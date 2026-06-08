from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Autoencoder.data.datasets import get_loader

from .common import (
    AverageMeter, apply_nested_tail, batch_metric_mean,
    format_metrics, print_epoch, print_run_header, psnr_per_image, recon_loss,
    resolve_path, sample_nested_m, seed_everything, should_save_latest, should_validate,
)
from .io import build_config, build_models, ckpt_path, forward_parts, load_experiment_checkpoint, save_checkpoint
from .models import TailCVQ, append_sample_pool, restart_dead_codes, stats_from_hist
from .stage1 import validate_stage1_or_3

@torch.no_grad()
def cvq_stats_on_loader(loader, encoder: nn.Module, cvq: TailCVQ, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    cvq.eval()
    hist_a = torch.zeros(cvq.cvq_a.num_codes, dtype=torch.long)
    hist_b = torch.zeros(cvq.cvq_b.num_codes, dtype=torch.long)
    mse_a_sum = 0.0
    mse_b_sum = 0.0
    mse_a_count = 0
    mse_b_count = 0
    tail_sum = 0.0
    tail_sq_sum = 0.0
    tail_count = 0
    qerr_sum = 0.0
    device = next(encoder.parameters()).device
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, _s, _y_prefix, tail = forward_parts(imgs, encoder, args, noisy=False)
        tail_q, _idx, aux = cvq.encode(tail)
        idx_a = aux["idx_a"].reshape(-1).detach().cpu()
        idx_b = aux["idx_b"].reshape(-1).detach().cpu()
        hist_a += torch.bincount(idx_a, minlength=cvq.cvq_a.num_codes)
        hist_b += torch.bincount(idx_b, minlength=cvq.cvq_b.num_codes)
        err_a = (tail_q[:, : cvq.split_a].float() - tail[:, : cvq.split_a].float()).square()
        err_b = (tail_q[:, cvq.split_a :].float() - tail[:, cvq.split_a :].float()).square()
        mse_a_sum += float(err_a.sum().item())
        mse_b_sum += float(err_b.sum().item())
        mse_a_count += int(err_a.numel())
        mse_b_count += int(err_b.numel())
        tail_f = tail.float()
        tail_sum += float(tail_f.sum().item())
        tail_sq_sum += float(tail_f.square().sum().item())
        tail_count += int(tail_f.numel())
        qerr_sum += float(err_a.sum().item() + err_b.sum().item())
    out = {}
    out.update(stats_from_hist(hist_a, mse_a_sum, mse_a_count, "cvq_a"))
    out.update(stats_from_hist(hist_b, mse_b_sum, mse_b_count, "cvq_b"))
    tail_mean = tail_sum / max(1, tail_count)
    tail_power = tail_sq_sum / max(1, tail_count)
    tail_var = max(0.0, tail_power - tail_mean * tail_mean)
    out["tail_power"] = tail_power
    out["tail_var"] = tail_var
    out["tail_quant_mse"] = qerr_sum / max(1, tail_count)
    out["tail_nmse"] = out["tail_quant_mse"] / max(tail_power, 1e-12)
    return out

def vq_lambda_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    ramp = max(1, int(args.vq_ramp_epochs))
    frac = min(1.0, epoch / float(ramp))
    return float(args.lambda_vq_start) + frac * (float(args.lambda_vq_end) - float(args.lambda_vq_start))

def freeze_module(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(trainable)

def encoder_output_head(encoder: nn.Module) -> nn.Module | None:
    base = encoder.module if hasattr(encoder, "module") else encoder
    inner = getattr(base, "encoder", None)
    return getattr(inner, "head_list", None)

def register_prefix_head_mask(encoder: nn.Module, prefix_ch: int) -> bool:
    head = encoder_output_head(encoder)
    if head is None or not hasattr(head, "weight"):
        return False
    if getattr(head, "_cvq_prefix_mask_registered", False):
        return True

    prefix_ch = int(prefix_ch)

    def mask_weight_grad(grad: torch.Tensor) -> torch.Tensor:
        out = grad.clone()
        out[:prefix_ch] = 0
        return out

    head.weight.register_hook(mask_weight_grad)
    if getattr(head, "bias", None) is not None:
        def mask_bias_grad(grad: torch.Tensor) -> torch.Tensor:
            out = grad.clone()
            out[:prefix_ch] = 0
            return out
        head.bias.register_hook(mask_bias_grad)
    head._cvq_prefix_mask_registered = True
    return True

def apply_encoder_trainability(encoder: nn.Module, args: argparse.Namespace, trainable: bool) -> bool:
    if not trainable:
        freeze_module(encoder, False)
        encoder.train(False)
        return False
    if bool(getattr(args, "stage3_freeze_encoder_body", False)):
        freeze_module(encoder, False)
        head = encoder_output_head(encoder)
        if head is None:
            raise RuntimeError("--stage3-freeze-encoder-body requires encoder.encoder.head_list")
        head.weight.requires_grad = True
        if getattr(head, "bias", None) is not None:
            head.bias.requires_grad = True
    else:
        freeze_module(encoder, True)
    if bool(getattr(args, "stage3_freeze_prefix_head", False)):
        if not register_prefix_head_mask(encoder, int(args.prefix_ch)):
            raise RuntimeError("--stage3-freeze-prefix-head requires encoder.encoder.head_list")
    encoder.train(True)
    return True

def cvq_usage_loss(tail: torch.Tensor, cvq: TailCVQ, tau: float) -> torch.Tensor:
    return cvq.usage_loss(tail, tau)

def tail_power_var(tail: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = tail.float()
    power = x.square().mean()
    var = x.var(unbiased=False)
    return power, var

def stage3_car_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "stage3_car_mode", "none")) != "none" and float(getattr(args, "lambda_stage3_car_rec", 0.0)) > 0.0

def stage3_fsq_car_soft_tail(
    car: nn.Module,
    y_prefix: torch.Tensor,
    cvq: TailCVQ,
    args: argparse.Namespace,
    *,
    scale: float,
    x_prefix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = str(getattr(args, "stage3_car_mode", "a"))
    if mode not in ("a", "all"):
        raise ValueError(f"unsupported Stage 3 CAR mode {mode!r}")
    bsz = y_prefix.shape[0]
    context = car._condition(y_prefix, x_prefix=x_prefix)
    values_a = cvq.cvq_a.level_values(device=y_prefix.device, dtype=y_prefix.float().dtype)
    values_b = cvq.cvq_b.level_values(device=y_prefix.device, dtype=y_prefix.float().dtype)
    embed_a = car.idx_embed_a.weight.to(device=y_prefix.device, dtype=y_prefix.float().dtype)
    embed_b = car.idx_embed_b.weight.to(device=y_prefix.device, dtype=y_prefix.float().dtype)
    tail_raw = torch.zeros(bsz, car.tail_ch, car.h, car.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
    idx = torch.zeros(bsz, car.tail_ch, car.h, car.w, device=y_prefix.device, dtype=torch.long)
    logits_a = []
    logits_b = []
    tau = max(float(getattr(args, "stage3_car_soft_tau", 1.0)), 1e-6)
    conf_floor = float(getattr(args, "stage3_car_conf_floor", 0.0))
    conf_gamma = float(getattr(args, "stage3_car_conf_gamma", 0.0))
    value_mode = str(getattr(args, "stage3_car_value_mode", "posterior"))
    direct_residual_scale = float(getattr(args, "stage3_car_direct_residual_scale", 1.0))
    max_t = car.tail_ch if mode == "all" else car.split_a
    for t in range(max_t):
        is_a = t < car.split_a
        if value_mode in ("direct", "residual") and hasattr(car, "_step_logits_value"):
            logits, value_raw = car._step_logits_value(context, t)
        else:
            logits = car._step_logits(context, t)
            value_raw = None
        k = car.k_a if is_a else car.k_b
        values = values_a if is_a else values_b
        embed_weight = embed_a if is_a else embed_b
        prob = F.softmax(logits.float() / tau, dim=-1)
        value_post = torch.einsum("bhwk,k->bhw", prob, values)
        if value_mode == "posterior":
            value = value_post
        elif value_mode == "direct":
            if value_raw is None:
                raise RuntimeError("stage3_car_value_mode=direct requires FSQSpatialCAR value heads")
            value = torch.tanh(value_raw.float()) * values.abs().max() * direct_residual_scale
        elif value_mode == "residual":
            if value_raw is None:
                raise RuntimeError("stage3_car_value_mode=residual requires FSQSpatialCAR value heads")
            step = (values.max() - values.min()) / max(1, int(k) - 1)
            value = value_post + torch.tanh(value_raw.float()) * step * direct_residual_scale
        else:
            raise ValueError(f"unknown stage3_car_value_mode={value_mode!r}")
        if conf_gamma > 0.0:
            conf = prob.max(dim=-1).values
            if conf_floor > 0.0:
                gate = ((conf - conf_floor) / max(1e-6, 1.0 - conf_floor)).clamp(0.0, 1.0)
            else:
                gate = conf.clamp(0.0, 1.0)
            value = value * gate.pow(conf_gamma)
        tail_raw[:, t] = value
        levels = torch.arange(k, device=logits.device, dtype=prob.dtype)
        idx[:, t] = torch.sum(prob * levels, dim=-1).round().long().clamp(0, k - 1)
        if is_a:
            logits_a.append(logits)
        else:
            logits_b.append(logits)
        if t + 1 < max_t:
            emb = torch.einsum("bhwk,kd->bdhw", prob, embed_weight)
            context = context + car.prev_scale * car.prev_proj(emb)
    if mode == "a":
        logits_b_t = torch.empty(0, device=y_prefix.device)
    else:
        logits_b_t = torch.stack(logits_b, dim=1)
    return tail_raw * float(scale), idx, torch.stack(logits_a, dim=1), logits_b_t

def stage3_car_ce_loss(car: nn.Module, logits_a: torch.Tensor, logits_b: torch.Tensor, idx: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    mode = str(getattr(args, "stage3_car_mode", "a"))
    loss_a = F.cross_entropy(logits_a.reshape(-1, car.k_a), idx[:, : car.split_a].reshape(-1))
    if mode == "a":
        return loss_a
    loss_b = F.cross_entropy(logits_b.reshape(-1, car.k_b), idx[:, car.split_a :].reshape(-1))
    return loss_a + loss_b

def stage3_car_ordinal_loss(car: nn.Module, logits_a: torch.Tensor, logits_b: torch.Tensor, idx: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    def part(logits: torch.Tensor, target: torch.Tensor, levels: int) -> torch.Tensor:
        prob = F.softmax(logits.float(), dim=-1)
        grid = torch.arange(int(levels), device=logits.device, dtype=prob.dtype)
        expect = torch.sum(prob * grid, dim=-1) / max(1, int(levels) - 1)
        target_f = target.float() / max(1, int(levels) - 1)
        target_onehot = F.one_hot(target.long().clamp(0, int(levels) - 1), num_classes=int(levels)).to(dtype=prob.dtype)
        emd = (prob.cumsum(dim=-1) - target_onehot.cumsum(dim=-1)).abs().mean()
        return F.smooth_l1_loss(expect, target_f) + emd

    losses = [part(logits_a, idx[:, : car.split_a], car.k_a)]
    if str(getattr(args, "stage3_car_mode", "a")) == "all":
        losses.append(part(logits_b, idx[:, car.split_a :], car.k_b))
    return sum(losses) / float(len(losses))

def stage3_car_alpha_values(args: argparse.Namespace) -> list[float]:
    raw = str(getattr(args, "stage3_car_alpha_sweep", ""))
    if not raw.strip():
        raw = str(getattr(args, "stage5_alpha_sweep", ""))
    if raw.strip():
        values = [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
    else:
        values = [float(getattr(args, "stage3_car_tail_scale", 1.0))]
    return sorted(set(values))

def stage3_car_predictable_lambda(args: argparse.Namespace, epoch: int) -> float:
    base = float(getattr(args, "lambda_stage3_car_predictable", 0.0))
    ramp = max(1, int(getattr(args, "stage3_car_predictable_ramp_epochs", 1)))
    return base * min(1.0, float(epoch) / float(ramp))

@torch.no_grad()
def validate_stage3_car(
    loader,
    encoder: nn.Module,
    decoder: nn.Module,
    cvq: TailCVQ,
    car: nn.Module,
    args: argparse.Namespace,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    cvq.eval()
    car.eval()
    alphas = stage3_car_alpha_values(args)
    meters = {k: AverageMeter() for k in ["car_ce", "car_acc_a", "car_acc_b", "car_acc_all", "psnr_car_raw", "psnr_vq_oracle"]}
    psnr_by_alpha = {alpha: AverageMeter() for alpha in alphas}
    for imgs, _labels in loader:
        imgs = imgs.to(next(encoder.parameters()).device, non_blocking=True)
        _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
        tail_q, idx, _aux = cvq.encode(tail)
        zero = torch.zeros_like(tail)
        x_prefix = decoder(torch.cat([y_prefix, zero], dim=1)).clamp(0.0, 1.0)
        tail_car_raw, pred_idx, logits_a, logits_b = stage3_fsq_car_soft_tail(
            car,
            y_prefix,
            cvq,
            args,
            scale=1.0,
            x_prefix=x_prefix if bool(getattr(args, "car_prefix_image_cond", False)) else None,
        )
        ce = stage3_car_ce_loss(car, logits_a, logits_b, idx, args)
        x_vq = decoder(torch.cat([y_prefix, tail_q], dim=1)).clamp(0.0, 1.0)
        x_raw = decoder(torch.cat([y_prefix, tail_car_raw * float(getattr(args, "stage3_car_tail_scale", 1.0))], dim=1)).clamp(0.0, 1.0)
        bsz = imgs.shape[0]
        pred_a = pred_idx[:, : car.split_a]
        idx_a = idx[:, : car.split_a]
        meters["car_ce"].update(float(ce.item()), bsz)
        meters["car_acc_a"].update(float((pred_a == idx_a).float().mean().item()), bsz)
        if str(getattr(args, "stage3_car_mode", "a")) == "all":
            pred_b = pred_idx[:, car.split_a :]
            idx_b = idx[:, car.split_a :]
            meters["car_acc_b"].update(float((pred_b == idx_b).float().mean().item()), bsz)
            meters["car_acc_all"].update(float((pred_idx == idx).float().mean().item()), bsz)
        else:
            meters["car_acc_b"].update(0.0, bsz)
            meters["car_acc_all"].update(float((pred_a == idx_a).float().mean().item()), bsz)
        meters["psnr_vq_oracle"].update(batch_metric_mean(psnr_per_image(x_vq, imgs)), bsz)
        meters["psnr_car_raw"].update(batch_metric_mean(psnr_per_image(x_raw, imgs)), bsz)
        for alpha in alphas:
            x_car = decoder(torch.cat([y_prefix, tail_car_raw * float(alpha)], dim=1)).clamp(0.0, 1.0)
            psnr_by_alpha[alpha].update(batch_metric_mean(psnr_per_image(x_car, imgs)), bsz)
    out = {k: v.avg for k, v in meters.items()}
    best_alpha = max(alphas, key=lambda alpha: psnr_by_alpha[alpha].avg)
    out["psnr_car"] = psnr_by_alpha[best_alpha].avg
    out["best_car_alpha"] = float(best_alpha)
    for alpha in alphas:
        key = f"{float(alpha):.2f}".replace("-", "m").replace(".", "p")
        out[f"psnr_car_alpha_{key}"] = psnr_by_alpha[alpha].avg
    return out

def train_stage3(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder, cvq, car = build_models(args, cfg.device)
    src = args.init_ckpt or ckpt_path(args, "stage2", "codebook_init")
    load_experiment_checkpoint(src, encoder=encoder, decoder=decoder, cvq=cvq, strict=True)
    car_enabled = stage3_car_enabled(args)
    if car_enabled:
        if str(args.cvq_mode) != "fsq" or not bool(getattr(car, "is_fsq_spatial", False)):
            raise RuntimeError("Stage 3 CAR-in-loop currently requires --cvq-mode fsq --car-arch swin")
        car_src = str(getattr(args, "stage3_init_car_ckpt", ""))
        if car_src:
            load_experiment_checkpoint(car_src, car=car, strict=False)
    teacher_encoder: nn.Module | None = None
    teacher_src: str | None = None
    if float(args.lambda_tail_distill) > 0.0:
        teacher_encoder, _teacher_decoder, _teacher_cvq, _teacher_car = build_models(args, cfg.device)
        teacher_src = args.stage3_teacher_ckpt or src
        load_experiment_checkpoint(teacher_src, encoder=teacher_encoder, strict=True)
        freeze_module(teacher_encoder, False)
        teacher_encoder.eval()
        del _teacher_decoder, _teacher_cvq, _teacher_car
    opt_groups = [{"params": list(encoder.parameters()) + list(decoder.parameters()) + list(cvq.parameters()), "lr": float(args.lr)}]
    if car_enabled:
        opt_groups.append({"params": car.parameters(), "lr": float(args.car_lr)})
    optimizer = optim.Adam(opt_groups)
    best = -1.0
    title = "Stage 3 | frozen encoder CVQ + decoder training"
    if bool(args.stage3_train_encoder) and bool(args.stage3_train_decoder):
        title = "Stage 3 | constrained end-to-end JSCC + CVQ training"
    elif not bool(args.stage3_train_encoder) and not bool(args.stage3_train_decoder):
        title = "Stage 3 | frozen JSCC, quantizer-only CVQ training"
    print_run_header(args, title, len(train_loader.dataset), len(val_loader.dataset))
    print(f"stage3_source_checkpoint={resolve_path(src)}", flush=True)
    if teacher_src is not None:
        print(f"stage3_teacher_checkpoint={resolve_path(teacher_src)}", flush=True)
    if car_enabled:
        print(
            "stage3_car_in_loop="
            f"mode={str(getattr(args, 'stage3_car_mode', 'none'))} "
            f"start_epoch={int(getattr(args, 'stage3_car_start_epoch', 1))} "
            f"lambda_car_rec={float(getattr(args, 'lambda_stage3_car_rec', 0.0)):g} "
            f"lambda_car_ce={float(getattr(args, 'lambda_stage3_car_ce', 0.0)):g} "
            f"lambda_car_value={float(getattr(args, 'lambda_stage3_car_value', 0.0)):g} "
            f"lambda_car_ordinal={float(getattr(args, 'lambda_stage3_car_ordinal', 0.0)):g} "
            f"lambda_car_gain={float(getattr(args, 'lambda_stage3_car_gain', 0.0)):g} "
            f"lambda_car_predictable={float(getattr(args, 'lambda_stage3_car_predictable', 0.0)):g} "
            f"tail_scale={float(getattr(args, 'stage3_car_tail_scale', 1.0)):g} "
            f"soft_tau={float(getattr(args, 'stage3_car_soft_tau', 1.0)):g} "
            f"conf_floor={float(getattr(args, 'stage3_car_conf_floor', 0.0)):g} "
            f"conf_gamma={float(getattr(args, 'stage3_car_conf_gamma', 0.0)):g} "
            f"value_mode={str(getattr(args, 'stage3_car_value_mode', 'posterior'))} "
            f"direct_residual_scale={float(getattr(args, 'stage3_car_direct_residual_scale', 1.0)):g} "
            f"prefix_image_cond={int(bool(getattr(args, 'car_prefix_image_cond', False)))} "
            f"prefix_image_scale_init={float(getattr(args, 'car_prefix_image_scale_init', 0.1)):g} "
            f"predictable_target={str(getattr(args, 'stage3_car_predictable_target', 'tail'))} "
            f"predictable_detach_car={int(bool(getattr(args, 'stage3_car_predictable_detach_car', True)))} "
            f"alpha_sweep={str(getattr(args, 'stage3_car_alpha_sweep', ''))}",
            flush=True,
        )
    restart_until = int(args.dead_code_restart_until)
    if restart_until < 0:
        restart_until = int(args.freeze_encoder_first)
    aux_unlocked = False
    for epoch in range(1, int(args.epochs) + 1):
        encoder_trainable = bool(args.stage3_train_encoder) and epoch > int(args.freeze_encoder_first)
        apply_encoder_trainability(encoder, args, encoder_trainable)
        decoder_trainable = bool(getattr(args, "stage3_train_decoder", True))
        freeze_module(decoder, decoder_trainable)
        decoder.train(decoder_trainable)
        cvq.train()
        car_active_epoch = car_enabled and epoch >= int(getattr(args, "stage3_car_start_epoch", 1))
        car.train(car_active_epoch)
        meters = {k: AverageMeter() for k in ["loss", "loss_rec", "loss_prefix", "loss_prefix_floor", "loss_gain", "loss_nd", "loss_vq", "loss_usage", "loss_tail_distill", "loss_cont", "loss_car_rec", "loss_car_ce", "loss_car_value", "loss_car_ordinal", "loss_car_gain", "loss_car_predictable", "psnr_vq", "psnr_prefix", "psnr_cont", "psnr_car", "tail_power", "tail_var"]}
        hist_a = torch.zeros(cvq.cvq_a.num_codes, dtype=torch.long)
        hist_b = torch.zeros(cvq.cvq_b.num_codes, dtype=torch.long)
        mse_a_sum = 0.0
        mse_b_sum = 0.0
        mse_a_count = 0
        mse_b_count = 0
        samples_a: list[torch.Tensor] = []
        samples_b: list[torch.Tensor] = []
        t0 = time.time()
        lam_vq = vq_lambda_for_epoch(args, epoch)
        lam_prefix_epoch = float(args.lambda_prefix_stage3) if aux_unlocked else 0.0
        lam_nested_epoch = float(args.lambda_nested_stage3) if aux_unlocked else 0.0
        lam_car_predictable = stage3_car_predictable_lambda(args, epoch)
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            if encoder_trainable:
                _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
            else:
                with torch.no_grad():
                    _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
            tail_q_st, _idx, loss_vq, aux = cvq(tail)
            tail_q_raw = torch.cat([aux["qa_raw"], aux["qb_raw"]], dim=1)
            tail_q_dec = tail_q_raw + (tail_q_st - tail_q_st.detach()) if encoder_trainable else tail_q_raw
            zero = torch.zeros_like(tail_q_dec)
            x_vq = decoder(torch.cat([y_prefix, tail_q_dec], dim=1))
            x_prefix = decoder(torch.cat([y_prefix, zero], dim=1))
            x_cont = decoder(torch.cat([y_prefix, tail], dim=1))
            if lam_nested_epoch > 0.0:
                tail_q_nd = apply_nested_tail(tail_q_dec, sample_nested_m(imgs.device, max_m=tail_q_dec.shape[1], batch_size=imgs.shape[0]))
                x_q_nd = decoder(torch.cat([y_prefix, tail_q_nd], dim=1))
                loss_nd = recon_loss(x_q_nd, imgs)
            else:
                loss_nd = x_vq.new_zeros(())
            loss_rec = recon_loss(x_vq, imgs)
            loss_prefix = recon_loss(x_prefix, imgs)
            loss_cont = recon_loss(x_cont, imgs)
            if float(args.lambda_prefix_floor_stage3) > 0.0:
                prefix_metric_mse = (x_prefix.float().clamp(0.0, 1.0) - imgs.float()).square().mean()
                prefix_floor_mse = 10.0 ** (-float(args.stage3_prefix_floor_db) / 10.0)
                loss_prefix_floor = F.relu(prefix_metric_mse - x_vq.new_tensor(prefix_floor_mse))
            else:
                loss_prefix_floor = x_vq.new_zeros(())
            if float(args.lambda_gain_stage3) > 0.0:
                gain_ratio = 10.0 ** (-float(args.stage3_gain_margin_db) / 10.0)
                loss_gain = F.relu(loss_rec - loss_prefix.detach() * gain_ratio)
            else:
                loss_gain = x_vq.new_zeros(())
            loss_usage = cvq_usage_loss(tail, cvq, float(args.stage3_usage_tau)) if float(args.lambda_usage) > 0.0 else x_vq.new_zeros(())
            if teacher_encoder is not None:
                with torch.no_grad():
                    _tz, _ts, _ty, teacher_tail = forward_parts(imgs, teacher_encoder, args, noisy=False)
                loss_tail_distill = F.mse_loss(tail.float(), teacher_tail.float())
            else:
                loss_tail_distill = x_vq.new_zeros(())
            loss_car_rec = x_vq.new_zeros(())
            loss_car_ce = x_vq.new_zeros(())
            loss_car_value = x_vq.new_zeros(())
            loss_car_ordinal = x_vq.new_zeros(())
            loss_car_gain = x_vq.new_zeros(())
            loss_car_predictable = x_vq.new_zeros(())
            x_car = None
            if car_active_epoch:
                tail_car_raw, _pred_idx, logits_a, logits_b = stage3_fsq_car_soft_tail(
                    car,
                    y_prefix,
                    cvq,
                    args,
                    scale=1.0,
                    x_prefix=x_prefix if bool(getattr(args, "car_prefix_image_cond", False)) else None,
                )
                tail_car = tail_car_raw * float(getattr(args, "stage3_car_tail_scale", 1.0))
                x_car = decoder(torch.cat([y_prefix, tail_car], dim=1))
                loss_car_rec = recon_loss(x_car, imgs)
                loss_car_ce = stage3_car_ce_loss(car, logits_a, logits_b, _idx, args)
                target_tail = tail_q_raw.detach()
                if str(getattr(args, "stage3_car_mode", "a")) == "a":
                    loss_car_value = F.mse_loss(tail_car_raw[:, : car.split_a].float(), target_tail[:, : car.split_a].float())
                else:
                    loss_car_value = F.mse_loss(tail_car_raw.float(), target_tail.float())
                loss_car_ordinal = stage3_car_ordinal_loss(car, logits_a, logits_b, _idx, args)
                if float(getattr(args, "lambda_stage3_car_gain", 0.0)) > 0.0:
                    ratio = 10.0 ** (-float(getattr(args, "stage3_car_gain_db", 0.5)) / 10.0)
                    loss_car_gain = F.relu(loss_car_rec - loss_prefix.detach() * ratio)
                if lam_car_predictable > 0.0 and encoder_trainable:
                    mode = str(getattr(args, "stage3_car_mode", "a"))
                    n_ch = car.split_a if mode == "a" else car.tail_ch
                    pred_tail = tail_car_raw[:, :n_ch]
                    if bool(getattr(args, "stage3_car_predictable_detach_car", True)):
                        pred_tail = pred_tail.detach()
                    target_name = str(getattr(args, "stage3_car_predictable_target", "tail"))
                    if target_name == "tail":
                        target_pred = tail[:, :n_ch]
                    elif target_name == "quant":
                        target_pred = tail_q_dec[:, :n_ch]
                    else:
                        raise ValueError(f"unknown stage3_car_predictable_target={target_name!r}")
                    loss_car_predictable = F.mse_loss(target_pred.float(), pred_tail.float())
            loss = (
                loss_rec
                + lam_prefix_epoch * loss_prefix
                + lam_nested_epoch * loss_nd
                + float(args.lambda_cont_stage3) * loss_cont
                + lam_vq * loss_vq
                + float(args.lambda_usage) * loss_usage
                + float(args.lambda_gain_stage3) * loss_gain
                + float(args.lambda_prefix_floor_stage3) * loss_prefix_floor
                + float(args.lambda_tail_distill) * loss_tail_distill
                + float(getattr(args, "lambda_stage3_car_rec", 0.0)) * loss_car_rec
                + float(getattr(args, "lambda_stage3_car_ce", 0.0)) * loss_car_ce
                + float(getattr(args, "lambda_stage3_car_value", 0.0)) * loss_car_value
                + float(getattr(args, "lambda_stage3_car_ordinal", 0.0)) * loss_car_ordinal
                + float(getattr(args, "lambda_stage3_car_gain", 0.0)) * loss_car_gain
                + lam_car_predictable * loss_car_predictable
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            power, var = tail_power_var(tail.detach())
            bsz = imgs.shape[0]
            meters["loss"].update(float(loss.item()), bsz)
            meters["loss_rec"].update(float(loss_rec.item()), bsz)
            meters["loss_prefix"].update(float(loss_prefix.item()), bsz)
            meters["loss_prefix_floor"].update(float(loss_prefix_floor.item()), bsz)
            meters["loss_gain"].update(float(loss_gain.item()), bsz)
            meters["loss_nd"].update(float(loss_nd.item()), bsz)
            meters["loss_vq"].update(float(loss_vq.item()), bsz)
            meters["loss_usage"].update(float(loss_usage.item()), bsz)
            meters["loss_tail_distill"].update(float(loss_tail_distill.item()), bsz)
            meters["loss_cont"].update(float(loss_cont.item()), bsz)
            meters["loss_car_rec"].update(float(loss_car_rec.item()), bsz)
            meters["loss_car_ce"].update(float(loss_car_ce.item()), bsz)
            meters["loss_car_value"].update(float(loss_car_value.item()), bsz)
            meters["loss_car_ordinal"].update(float(loss_car_ordinal.item()), bsz)
            meters["loss_car_gain"].update(float(loss_car_gain.item()), bsz)
            meters["loss_car_predictable"].update(float(loss_car_predictable.item()), bsz)
            meters["psnr_vq"].update(batch_metric_mean(psnr_per_image(x_vq, imgs)), bsz)
            meters["psnr_prefix"].update(batch_metric_mean(psnr_per_image(x_prefix, imgs)), bsz)
            meters["psnr_cont"].update(batch_metric_mean(psnr_per_image(x_cont, imgs)), bsz)
            if x_car is not None:
                meters["psnr_car"].update(batch_metric_mean(psnr_per_image(x_car, imgs)), bsz)
            meters["tail_power"].update(float(power.item()), bsz)
            meters["tail_var"].update(float(var.item()), bsz)
            with torch.no_grad():
                idx_a = aux["idx_a"].reshape(-1).detach().cpu()
                idx_b = aux["idx_b"].reshape(-1).detach().cpu()
                hist_a += torch.bincount(idx_a, minlength=cvq.cvq_a.num_codes)
                hist_b += torch.bincount(idx_b, minlength=cvq.cvq_b.num_codes)
                err_a = (tail_q_raw[:, : cvq.split_a].float() - tail[:, : cvq.split_a].float()).square()
                err_b = (tail_q_raw[:, cvq.split_a :].float() - tail[:, cvq.split_a :].float()).square()
                mse_a_sum += float(err_a.sum().item())
                mse_b_sum += float(err_b.sum().item())
                mse_a_count += int(err_a.numel())
                mse_b_count += int(err_b.numel())
                append_sample_pool(
                    samples_a,
                    tail[:, : cvq.split_a].reshape(-1, int(args.latent_h), int(args.latent_w)),
                    int(args.dead_code_restart_pool),
                )
                append_sample_pool(
                    samples_b,
                    tail[:, cvq.split_a :].reshape(-1, int(args.latent_h), int(args.latent_w)),
                    int(args.dead_code_restart_pool),
                )
        metrics = {k: v.avg for k, v in meters.items()}
        metrics["lambda_vq"] = lam_vq
        metrics["lambda_prefix_stage3_effective"] = lam_prefix_epoch
        metrics["lambda_nested_stage3_effective"] = lam_nested_epoch
        metrics["lambda_cont_stage3"] = float(args.lambda_cont_stage3)
        metrics["lambda_usage"] = float(args.lambda_usage)
        metrics["lambda_gain_stage3"] = float(args.lambda_gain_stage3)
        metrics["lambda_prefix_floor_stage3"] = float(args.lambda_prefix_floor_stage3)
        metrics["stage3_prefix_floor_db"] = float(args.stage3_prefix_floor_db)
        metrics["stage3_gain_margin_db"] = float(args.stage3_gain_margin_db)
        metrics["lambda_tail_distill"] = float(args.lambda_tail_distill)
        metrics["stage3_car_active"] = float(car_active_epoch)
        metrics["lambda_stage3_car_rec"] = float(getattr(args, "lambda_stage3_car_rec", 0.0))
        metrics["lambda_stage3_car_ce"] = float(getattr(args, "lambda_stage3_car_ce", 0.0))
        metrics["lambda_stage3_car_value"] = float(getattr(args, "lambda_stage3_car_value", 0.0))
        metrics["lambda_stage3_car_ordinal"] = float(getattr(args, "lambda_stage3_car_ordinal", 0.0))
        metrics["lambda_stage3_car_gain"] = float(getattr(args, "lambda_stage3_car_gain", 0.0))
        metrics["lambda_stage3_car_predictable"] = float(lam_car_predictable)
        metrics["stage3_car_tail_scale"] = float(getattr(args, "stage3_car_tail_scale", 1.0))
        metrics["stage3_car_conf_floor"] = float(getattr(args, "stage3_car_conf_floor", 0.0))
        metrics["stage3_car_conf_gamma"] = float(getattr(args, "stage3_car_conf_gamma", 0.0))
        metrics["stage3_car_direct_residual_scale"] = float(getattr(args, "stage3_car_direct_residual_scale", 1.0))
        metrics["aux_unlocked"] = float(aux_unlocked)
        metrics["encoder_trainable"] = float(encoder_trainable)
        metrics["stage3_train_encoder"] = float(bool(args.stage3_train_encoder))
        metrics["stage3_train_decoder"] = float(decoder_trainable)
        metrics["stage3_freeze_encoder_body"] = float(bool(getattr(args, "stage3_freeze_encoder_body", False)))
        metrics["stage3_freeze_prefix_head"] = float(bool(getattr(args, "stage3_freeze_prefix_head", False)))
        metrics.update(stats_from_hist(hist_a, mse_a_sum, mse_a_count, "train_cvq_a"))
        metrics.update(stats_from_hist(hist_b, mse_b_sum, mse_b_count, "train_cvq_b"))
        restart_epoch = (
            int(args.dead_code_restart_every) > 0
            and epoch <= restart_until
            and epoch % int(args.dead_code_restart_every) == 0
        )
        metrics["dead_restart_enabled"] = float(epoch <= restart_until)
        if restart_epoch:
            restarted_a = restart_dead_codes(cvq.cvq_a, hist_a, samples_a)
            restarted_b = restart_dead_codes(cvq.cvq_b, hist_b, samples_b)
            metrics["dead_restart_a"] = float(restarted_a)
            metrics["dead_restart_b"] = float(restarted_b)
        print_epoch("stage3", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate_stage1_or_3(val_loader, encoder, decoder, args, cvq=cvq)
            if car_enabled:
                car_metrics = validate_stage3_car(val_loader, encoder, decoder, cvq, car, args)
                val_metrics.update(car_metrics)
            val_metrics.update(cvq_stats_on_loader(val_loader, encoder, cvq, args))
            val_metrics["tail_gain_cont"] = val_metrics["psnr_cont"] - val_metrics["psnr_prefix"]
            val_metrics["tail_gain_vq"] = val_metrics["psnr_vq"] - val_metrics["psnr_prefix"]
            val_metrics["gap_vq"] = val_metrics["psnr_cont"] - val_metrics["psnr_vq"]
            if car_enabled:
                val_metrics["tail_gain_car"] = val_metrics["psnr_car"] - val_metrics["psnr_prefix"]
                val_metrics["gap_car_to_vq_oracle"] = val_metrics["psnr_vq_oracle"] - val_metrics["psnr_car"]
            score_name = str(getattr(args, "stage3_score", "tail_gain_vq"))
            if score_name not in val_metrics:
                raise ValueError(f"unknown Stage 3 score metric {score_name!r}; available={sorted(val_metrics)}")
            score = val_metrics[score_name]
            enough_codes = (
                val_metrics.get("cvq_a_used_codes", 0.0) >= float(args.stage3_min_used_a)
                and val_metrics.get("cvq_b_used_codes", 0.0) >= float(args.stage3_min_used_b)
            )
            enough_tail_gain = val_metrics["tail_gain_vq"] >= float(args.stage3_min_tail_gain_vq)
            if car_enabled and score_name in ("tail_gain_car", "psnr_car"):
                enough_tail_gain = val_metrics["tail_gain_car"] >= float(getattr(args, "stage3_min_tail_gain_car", 0.0))
            enough_cont_gain = val_metrics["tail_gain_cont"] >= float(args.stage3_min_tail_gain_cont)
            enough_prefix_base = val_metrics["psnr_prefix"] >= float(getattr(args, "stage3_min_psnr_prefix", 0.0))
            enough_perplexity = (
                val_metrics.get("cvq_a_perplexity", 0.0) >= float(args.stage3_min_perplexity_a)
                and val_metrics.get("cvq_b_perplexity", 0.0) >= float(args.stage3_min_perplexity_b)
            )
            if (
                not aux_unlocked
                and epoch >= int(args.stage3_aux_start_epoch)
                and val_metrics["tail_gain_vq"] >= float(args.stage3_aux_tail_gain_threshold)
            ):
                aux_unlocked = True
                print(f"[stage3 val {epoch:03d}] aux_unlocked=1", flush=True)
            val_metrics["score"] = score
            val_metrics["used_code_gate"] = float(enough_codes)
            val_metrics["tail_gain_gate"] = float(enough_tail_gain)
            val_metrics["tail_cont_gate"] = float(enough_cont_gain)
            val_metrics["prefix_base_gate"] = float(enough_prefix_base)
            val_metrics["perplexity_gate"] = float(enough_perplexity)
            val_metrics["min_tail_gain_vq"] = float(args.stage3_min_tail_gain_vq)
            if car_enabled:
                val_metrics["min_tail_gain_car"] = float(getattr(args, "stage3_min_tail_gain_car", 0.0))
            val_metrics["min_tail_gain_cont"] = float(args.stage3_min_tail_gain_cont)
            val_metrics["min_psnr_prefix"] = float(getattr(args, "stage3_min_psnr_prefix", 0.0))
            val_metrics["min_perplexity_a"] = float(args.stage3_min_perplexity_a)
            val_metrics["min_perplexity_b"] = float(args.stage3_min_perplexity_b)
            print(f"[stage3 val {epoch:03d}] {format_metrics(val_metrics)} score={score_name}")
            if restart_epoch:
                print(f"[stage3 val {epoch:03d}] skip_best_save=dead_code_restart_epoch", flush=True)
            elif not enough_codes:
                print(f"[stage3 val {epoch:03d}] skip_best_save=used_code_gate", flush=True)
            elif not enough_tail_gain:
                print(f"[stage3 val {epoch:03d}] skip_best_save=tail_gain_gate", flush=True)
            elif not enough_cont_gain:
                print(f"[stage3 val {epoch:03d}] skip_best_save=tail_cont_gate", flush=True)
            elif not enough_prefix_base:
                print(f"[stage3 val {epoch:03d}] skip_best_save=prefix_base_gate", flush=True)
            elif not enough_perplexity:
                print(f"[stage3 val {epoch:03d}] skip_best_save=perplexity_gate", flush=True)
            elif score > best:
                best = score
                save_checkpoint(ckpt_path(args, "stage3", "best"), stage="stage3", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car if car_enabled else None)
        if should_save_latest(args, epoch):
            save_checkpoint(ckpt_path(args, "stage3", "latest"), stage="stage3", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car if car_enabled else None)
    save_checkpoint(ckpt_path(args, "stage3", "latest"), stage="stage3", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car if car_enabled else None)
