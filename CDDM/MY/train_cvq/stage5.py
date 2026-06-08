from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from Autoencoder.data.datasets import get_loader

from .common import (
    AverageMeter,
    batch_metric_mean,
    format_metrics,
    print_epoch,
    print_run_header,
    psnr_per_image,
    recon_loss,
    seed_everything,
    should_save_latest,
    should_validate,
)
from .io import build_config, build_models, ckpt_path, forward_parts, load_experiment_checkpoint, save_checkpoint
from .models import TailCAR, TailCVQ
from .stage3 import encoder_output_head, freeze_module

def car_labels(imgs: torch.Tensor, encoder: nn.Module, cvq: TailCVQ, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
    _tail_q, idx, _aux = cvq.encode(tail)
    return y_prefix, idx

def is_fsq_spatial_car(cvq: TailCVQ, car: torch.nn.Module) -> bool:
    return str(getattr(cvq, "mode", "")) == "fsq" and bool(getattr(car, "is_fsq_spatial", False))

def stage5_best_allowed(args: argparse.Namespace, val_metrics: dict[str, float]) -> bool:
    return float(val_metrics.get("tail_gain_car", -1e9)) >= float(getattr(args, "stage5_min_tail_gain_car", 0.0))

def register_tail_head_mask(encoder: nn.Module, prefix_ch: int) -> bool:
    head = encoder_output_head(encoder)
    if head is None or not hasattr(head, "weight"):
        return False
    if getattr(head, "_cvq_tail_mask_registered", False):
        return True
    prefix_ch = int(prefix_ch)

    def mask_weight_grad(grad: torch.Tensor) -> torch.Tensor:
        out = grad.clone()
        out[prefix_ch:] = 0
        return out

    head.weight.register_hook(mask_weight_grad)
    if getattr(head, "bias", None) is not None:
        def mask_bias_grad(grad: torch.Tensor) -> torch.Tensor:
            out = grad.clone()
            out[prefix_ch:] = 0
            return out
        head.bias.register_hook(mask_bias_grad)
    head._cvq_tail_mask_registered = True
    return True

def apply_stage5_encoder_trainability(encoder: nn.Module, args: argparse.Namespace) -> bool:
    train_encoder = bool(getattr(args, "stage5_train_encoder", False))
    if not train_encoder:
        freeze_module(encoder, False)
        encoder.train(False)
        return False
    if bool(getattr(args, "stage5_freeze_encoder_body", False)):
        freeze_module(encoder, False)
        head = encoder_output_head(encoder)
        if head is None:
            raise RuntimeError("--stage5-freeze-encoder-body requires encoder.encoder.head_list")
        head.weight.requires_grad = True
        if getattr(head, "bias", None) is not None:
            head.bias.requires_grad = True
    else:
        freeze_module(encoder, True)
    if bool(getattr(args, "stage5_train_prefix_head_only", False)):
        if not register_tail_head_mask(encoder, int(args.prefix_ch)):
            raise RuntimeError("--stage5-train-prefix-head-only requires encoder.encoder.head_list")
    encoder.train(True)
    return True

def fsq_strategy(args: argparse.Namespace) -> str:
    return str(getattr(args, "stage5_fsq_strategy", "joint"))

def fsq_is_coarse_strategy(strategy: str) -> bool:
    return str(strategy) in ("coarse_a", "coarse_a_residual")

def fsq_is_posterior_strategy(strategy: str) -> bool:
    return str(strategy) in ("posterior_gated_a", "posterior_softar_a", "posterior_softar_all")

def fsq_is_softar_posterior_strategy(strategy: str) -> bool:
    return str(strategy) in ("posterior_softar_a", "posterior_softar_all")

def fsq_is_posterior_all_strategy(strategy: str) -> bool:
    return str(strategy) == "posterior_softar_all"

def fsq_gate_sweep_values(args: argparse.Namespace) -> list[float]:
    raw = str(getattr(args, "stage5_gate_sweep", ""))
    if raw.strip():
        vals = [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
    else:
        vals = [-1.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    return sorted(set(vals))

def fsq_alpha_sweep_values(args: argparse.Namespace) -> list[float]:
    raw = str(getattr(args, "stage5_alpha_sweep", ""))
    if raw.strip():
        vals = [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
    else:
        vals = [float(getattr(args, "stage5_tail_scale", 1.0))]
    return sorted(set(vals))

def fsq_tail_scale(args: argparse.Namespace) -> float:
    return float(getattr(args, "stage5_tail_scale", 1.0))

def fsq_metric_float_key(value: float) -> str:
    return f"{float(value):.2f}".replace("-", "m").replace(".", "p")

def fsq_coarse_levels(args: argparse.Namespace) -> int:
    return int(getattr(args, "stage5_fsq_coarse_levels", 4))

def fsq_coarse_factor(num_codes: int, coarse_levels: int) -> int:
    num_codes = int(num_codes)
    coarse_levels = int(coarse_levels)
    if coarse_levels < 2:
        raise ValueError(f"coarse levels must be >= 2, got {coarse_levels}")
    if num_codes % coarse_levels != 0:
        raise ValueError(f"FSQ levels={num_codes} must be divisible by coarse levels={coarse_levels}")
    return num_codes // coarse_levels

def fsq_coarse_logits(logits: torch.Tensor, num_codes: int, coarse_levels: int) -> torch.Tensor:
    factor = fsq_coarse_factor(num_codes, coarse_levels)
    return logits.reshape(*logits.shape[:-1], int(coarse_levels), factor).logsumexp(dim=-1)

def fsq_coarse_target(idx: torch.Tensor, num_codes: int, coarse_levels: int) -> torch.Tensor:
    factor = fsq_coarse_factor(num_codes, coarse_levels)
    return (idx.long() // factor).clamp(0, int(coarse_levels) - 1)

def fsq_center_index_from_coarse(coarse: torch.Tensor, num_codes: int, coarse_levels: int) -> torch.Tensor:
    factor = fsq_coarse_factor(num_codes, coarse_levels)
    return coarse.float() * float(factor) + 0.5 * float(factor - 1)

def fsq_decode_fractional_a(center_idx: torch.Tensor, cvq: TailCVQ, car: torch.nn.Module) -> torch.Tensor:
    scale = cvq.cvq_a._scale().to(device=center_idx.device, dtype=center_idx.float().dtype)
    values = ((center_idx.float() / float(car.k_a - 1)) * 2.0 - 1.0) * scale
    return values

def fsq_ce_a(car: torch.nn.Module, logits_a: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits_a.reshape(-1, car.k_a), idx[:, : car.split_a].reshape(-1))

def fsq_ce_b(car: torch.nn.Module, logits_b: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits_b.reshape(-1, car.k_b), idx[:, car.split_a :].reshape(-1))

def fsq_ce_a_coarse(car: torch.nn.Module, logits_a: torch.Tensor, idx: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    levels = fsq_coarse_levels(args)
    logits = fsq_coarse_logits(logits_a, car.k_a, levels)
    target = fsq_coarse_target(idx[:, : car.split_a], car.k_a, levels)
    return F.cross_entropy(logits.reshape(-1, levels), target.reshape(-1))

def fsq_residual_logits_a(
    car: torch.nn.Module,
    logits_a: torch.Tensor,
    coarse_idx: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    levels = fsq_coarse_levels(args)
    factor = fsq_coarse_factor(car.k_a, levels)
    grouped = logits_a.reshape(*logits_a.shape[:-1], levels, factor)
    gather_idx = coarse_idx.long().clamp(0, levels - 1).unsqueeze(-1).unsqueeze(-1).expand(*coarse_idx.shape, 1, factor)
    return grouped.gather(-2, gather_idx).squeeze(-2)

def fsq_ce_a_residual(car: torch.nn.Module, logits_a: torch.Tensor, idx: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    levels = fsq_coarse_levels(args)
    factor = fsq_coarse_factor(car.k_a, levels)
    idx_a = idx[:, : car.split_a].long()
    coarse = fsq_coarse_target(idx_a, car.k_a, levels)
    residual = (idx_a % factor).clamp(0, factor - 1)
    logits = fsq_residual_logits_a(car, logits_a, coarse, args)
    return F.cross_entropy(logits.reshape(-1, factor), residual.reshape(-1))

def fsq_hard_coarse_from_logits(logits: torch.Tensor, num_codes: int, coarse_levels: int, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    coarse_logits = fsq_coarse_logits(logits, num_codes, coarse_levels)
    prob = F.softmax(coarse_logits.float(), dim=-1)
    if str(mode) == "mean":
        grid = torch.arange(int(coarse_levels), device=logits.device, dtype=prob.dtype)
        coarse = torch.sum(prob * grid, dim=-1).round().long().clamp(0, int(coarse_levels) - 1)
    elif str(mode) == "argmax":
        coarse = coarse_logits.argmax(dim=-1)
    else:
        raise ValueError(f"unknown FSQ generate mode: {mode}")
    conf = prob.gather(-1, coarse.unsqueeze(-1)).squeeze(-1)
    return coarse, conf

def fsq_soft_coarse_a_tail(
    car: torch.nn.Module,
    logits_a: torch.Tensor,
    cvq: TailCVQ,
    args: argparse.Namespace,
    *,
    residual: bool,
) -> torch.Tensor:
    levels = fsq_coarse_levels(args)
    factor = fsq_coarse_factor(car.k_a, levels)
    grouped = logits_a.reshape(*logits_a.shape[:-1], levels, factor)
    coarse_logits = grouped.logsumexp(dim=-1)
    coarse_prob = F.softmax(coarse_logits.float() / max(float(args.stage5_soft_tau), 1e-6), dim=-1)
    coarse_grid = torch.arange(levels, device=logits_a.device, dtype=coarse_prob.dtype)
    if residual:
        res_prob = F.softmax(grouped.float() / max(float(args.stage5_soft_tau), 1e-6), dim=-1)
        res_grid = torch.arange(factor, device=logits_a.device, dtype=res_prob.dtype)
        res_expect = torch.sum(res_prob * res_grid, dim=-1)
        center_per_coarse = coarse_grid.view(*([1] * (coarse_prob.ndim - 1)), levels) * float(factor) + res_expect
    else:
        center_per_coarse = coarse_grid.view(*([1] * (coarse_prob.ndim - 1)), levels) * float(factor) + 0.5 * float(factor - 1)
    center_idx = torch.sum(coarse_prob * center_per_coarse, dim=-1)
    tail = torch.zeros(logits_a.shape[0], car.tail_ch, car.h, car.w, device=logits_a.device, dtype=logits_a.float().dtype)
    tail[:, : car.split_a] = fsq_decode_fractional_a(center_idx, cvq, car)
    return tail

@torch.no_grad()
def fsq_hard_coarse_a_tail_from_logits(
    car: torch.nn.Module,
    logits_a: torch.Tensor,
    cvq: TailCVQ,
    args: argparse.Namespace,
    *,
    residual: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    levels = fsq_coarse_levels(args)
    factor = fsq_coarse_factor(car.k_a, levels)
    coarse, conf = fsq_hard_coarse_from_logits(logits_a, car.k_a, levels, str(args.stage5_fsq_generate))
    center_idx = fsq_center_index_from_coarse(coarse, car.k_a, levels)
    if residual:
        logits_res = fsq_residual_logits_a(car, logits_a, coarse, args)
        res_prob = F.softmax(logits_res.float(), dim=-1)
        if str(args.stage5_fsq_generate) == "mean":
            res_grid = torch.arange(factor, device=logits_a.device, dtype=res_prob.dtype)
            residual_idx = torch.sum(res_prob * res_grid, dim=-1).round().long().clamp(0, factor - 1)
        else:
            residual_idx = logits_res.argmax(dim=-1)
        res_conf = res_prob.gather(-1, residual_idx.unsqueeze(-1)).squeeze(-1)
        th = float(getattr(args, "stage5_residual_threshold", -1.0))
        use_res = torch.ones_like(res_conf, dtype=torch.bool) if th < 0.0 else res_conf >= th
        residual_center = coarse.float() * float(factor) + residual_idx.float()
        center_idx = torch.where(use_res, residual_center, center_idx)
        conf = conf * torch.where(use_res, res_conf, torch.ones_like(res_conf))
    tail = torch.zeros(logits_a.shape[0], car.tail_ch, car.h, car.w, device=logits_a.device, dtype=logits_a.float().dtype)
    tail_a = fsq_decode_fractional_a(center_idx, cvq, car)
    if bool(getattr(args, "stage5_coarse_gate_to_zero", False)):
        tail_a = tail_a * (conf >= float(getattr(args, "stage5_gate_threshold", 0.55))).float()
    tail[:, : car.split_a] = tail_a
    return tail, coarse, conf

def fsq_ordinal_part(logits: torch.Tensor, target: torch.Tensor, levels: int) -> torch.Tensor:
    prob = F.softmax(logits.float(), dim=-1)
    grid = torch.arange(int(levels), device=logits.device, dtype=prob.dtype)
    expect = torch.sum(prob * grid, dim=-1) / max(1, int(levels) - 1)
    target_f = target.float() / max(1, int(levels) - 1)
    value = F.smooth_l1_loss(expect, target_f)
    target_onehot = F.one_hot(target.long().clamp(0, int(levels) - 1), num_classes=int(levels)).to(dtype=prob.dtype)
    emd = (prob.cumsum(dim=-1) - target_onehot.cumsum(dim=-1)).abs().mean()
    return value + emd

def fsq_ordinal_loss(car: torch.nn.Module, logits_a: torch.Tensor, logits_b: torch.Tensor, idx: torch.Tensor, part: str = "all") -> torch.Tensor:
    losses = []
    if part in ("all", "a"):
        losses.append(fsq_ordinal_part(logits_a, idx[:, : car.split_a], car.k_a))
    if part in ("all", "b"):
        losses.append(fsq_ordinal_part(logits_b, idx[:, car.split_a :], car.k_b))
    return sum(losses) / max(1, len(losses))

def fsq_a_only_tail(tail: torch.Tensor, car: torch.nn.Module) -> torch.Tensor:
    out = tail.clone()
    out[:, car.split_a :] = 0.0
    return out

def fsq_confidence_mask(conf: torch.Tensor, car: torch.nn.Module, args: argparse.Namespace) -> torch.Tensor:
    th_a = float(getattr(args, "stage5_gate_threshold", 0.55))
    th_b = float(getattr(args, "stage5_gate_b_threshold", -1.0))
    if th_b < 0.0:
        th_b = th_a
    mask = torch.empty_like(conf, dtype=torch.bool)
    mask[:, : car.split_a] = conf[:, : car.split_a] >= th_a
    mask[:, car.split_a :] = conf[:, car.split_a :] >= th_b
    return mask

def fsq_gated_tail(tail: torch.Tensor, conf: torch.Tensor, car: torch.nn.Module, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    mask = fsq_confidence_mask(conf, car, args)
    return tail * mask.float(), mask

def fsq_fsq_values_a(cvq: TailCVQ, logits_a: torch.Tensor) -> torch.Tensor:
    return cvq.cvq_a.level_values(device=logits_a.device, dtype=logits_a.float().dtype)

def fsq_posterior_a_tail_from_logits(
    car: torch.nn.Module,
    logits_a: torch.Tensor,
    cvq: TailCVQ,
    args: argparse.Namespace,
    *,
    threshold: float | None,
    soft_gate: bool,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prob = F.softmax(logits_a.float() / max(float(args.stage5_soft_tau), 1e-6), dim=-1)
    values = fsq_fsq_values_a(cvq, logits_a)
    tail_a = torch.einsum("bchwk,k->bchw", prob, values)
    conf = prob.max(dim=-1).values
    if threshold is None or float(threshold) < 0.0:
        gate = torch.ones_like(conf)
    elif soft_gate:
        temp = max(float(getattr(args, "stage5_gate_soft_temp", 0.08)), 1e-6)
        gate = torch.sigmoid((conf - float(threshold)) / temp)
    else:
        gate = (conf >= float(threshold)).to(dtype=tail_a.dtype)
    tail_scale = fsq_tail_scale(args) if scale is None else float(scale)
    tail = torch.zeros(logits_a.shape[0], car.tail_ch, car.h, car.w, device=logits_a.device, dtype=tail_a.dtype)
    tail[:, : car.split_a] = tail_a * gate * tail_scale
    return tail, conf, gate

def fsq_softar_posterior_a_tail(
    car: torch.nn.Module,
    y_prefix: torch.Tensor,
    cvq: TailCVQ,
    args: argparse.Namespace,
    *,
    threshold: float | None,
    soft_gate: bool,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = y_prefix.shape[0]
    context = car._condition(y_prefix)
    values = cvq.cvq_a.level_values(device=y_prefix.device, dtype=y_prefix.float().dtype)
    embed_weight = car.idx_embed_a.weight.to(device=y_prefix.device, dtype=y_prefix.float().dtype)
    idx = torch.zeros(bsz, car.split_a, car.h, car.w, device=y_prefix.device, dtype=torch.long)
    conf = torch.zeros(bsz, car.split_a, car.h, car.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
    gate = torch.zeros_like(conf)
    tail_a = torch.zeros_like(conf)
    logits_all = []
    for t in range(car.split_a):
        logits = car._step_logits(context, t)
        prob = F.softmax(logits.float() / max(float(args.stage5_soft_tau), 1e-6), dim=-1)
        conf_t = prob.max(dim=-1).values
        if threshold is None or float(threshold) < 0.0:
            gate_t = torch.ones_like(conf_t)
        elif soft_gate:
            temp = max(float(getattr(args, "stage5_gate_soft_temp", 0.08)), 1e-6)
            gate_t = torch.sigmoid((conf_t - float(threshold)) / temp)
        else:
            gate_t = (conf_t >= float(threshold)).to(dtype=conf_t.dtype)
        idx_t = car._logits_to_idx(logits, car.k_a, str(args.stage5_fsq_generate))
        tail_a[:, t] = torch.einsum("bhwk,k->bhw", prob, values) * gate_t
        idx[:, t] = idx_t
        conf[:, t] = conf_t
        gate[:, t] = gate_t
        logits_all.append(logits)
        if t + 1 < car.split_a:
            emb = torch.einsum("bhwk,kd->bdhw", prob, embed_weight)
            context = context + car.prev_scale * car.prev_proj(emb)
    tail_scale = fsq_tail_scale(args) if scale is None else float(scale)
    tail = torch.zeros(bsz, car.tail_ch, car.h, car.w, device=y_prefix.device, dtype=tail_a.dtype)
    tail[:, : car.split_a] = tail_a * tail_scale
    return tail, idx, conf, gate, torch.stack(logits_all, dim=1)

def fsq_softar_posterior_tail(
    car: torch.nn.Module,
    y_prefix: torch.Tensor,
    cvq: TailCVQ,
    args: argparse.Namespace,
    *,
    threshold: float | None,
    soft_gate: bool,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = y_prefix.shape[0]
    context = car._condition(y_prefix)
    values_a = cvq.cvq_a.level_values(device=y_prefix.device, dtype=y_prefix.float().dtype)
    values_b = cvq.cvq_b.level_values(device=y_prefix.device, dtype=y_prefix.float().dtype)
    embed_a = car.idx_embed_a.weight.to(device=y_prefix.device, dtype=y_prefix.float().dtype)
    embed_b = car.idx_embed_b.weight.to(device=y_prefix.device, dtype=y_prefix.float().dtype)
    idx = torch.zeros(bsz, car.tail_ch, car.h, car.w, device=y_prefix.device, dtype=torch.long)
    conf = torch.zeros(bsz, car.tail_ch, car.h, car.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
    gate = torch.zeros_like(conf)
    tail_raw = torch.zeros_like(conf)
    logits_a = []
    logits_b = []
    for t in range(car.tail_ch):
        logits = car._step_logits(context, t)
        is_a = t < car.split_a
        k = car.k_a if is_a else car.k_b
        values = values_a if is_a else values_b
        embed_weight = embed_a if is_a else embed_b
        prob = F.softmax(logits.float() / max(float(args.stage5_soft_tau), 1e-6), dim=-1)
        conf_t = prob.max(dim=-1).values
        if threshold is None or float(threshold) < 0.0:
            gate_t = torch.ones_like(conf_t)
        elif soft_gate:
            temp = max(float(getattr(args, "stage5_gate_soft_temp", 0.08)), 1e-6)
            gate_t = torch.sigmoid((conf_t - float(threshold)) / temp)
        else:
            gate_t = (conf_t >= float(threshold)).to(dtype=conf_t.dtype)
        idx_t = car._logits_to_idx(logits, k, str(args.stage5_fsq_generate))
        tail_raw[:, t] = torch.einsum("bhwk,k->bhw", prob, values) * gate_t
        idx[:, t] = idx_t
        conf[:, t] = conf_t
        gate[:, t] = gate_t
        if is_a:
            logits_a.append(logits)
        else:
            logits_b.append(logits)
        if t + 1 < car.tail_ch:
            emb = torch.einsum("bhwk,kd->bdhw", prob, embed_weight)
            context = context + car.prev_scale * car.prev_proj(emb)
    tail_scale = fsq_tail_scale(args) if scale is None else float(scale)
    return tail_raw * tail_scale, idx, conf, gate, torch.stack(logits_a, dim=1), torch.stack(logits_b, dim=1)

@torch.no_grad()
def fsq_generate_posterior_a_tail(
    car: torch.nn.Module,
    y_prefix: torch.Tensor,
    cvq: TailCVQ,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = y_prefix.shape[0]
    context = car._condition(y_prefix)
    idx = torch.zeros(bsz, car.split_a, car.h, car.w, device=y_prefix.device, dtype=torch.long)
    conf = torch.zeros(bsz, car.split_a, car.h, car.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
    tail_a = torch.zeros(bsz, car.split_a, car.h, car.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
    values = cvq.cvq_a.level_values(device=y_prefix.device, dtype=y_prefix.float().dtype)
    for t in range(car.split_a):
        logits = car._step_logits(context, t)
        prob = F.softmax(logits.float() / max(float(args.stage5_soft_tau), 1e-6), dim=-1)
        tail_a[:, t] = torch.einsum("bhwk,k->bhw", prob, values)
        conf[:, t] = prob.max(dim=-1).values
        idx_t = car._logits_to_idx(logits, car.k_a, str(args.stage5_fsq_generate))
        idx[:, t] = idx_t
        if t + 1 < car.split_a:
            context = car._update_context(context, idx_t, t)
    tail = torch.zeros(bsz, car.tail_ch, car.h, car.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
    tail[:, : car.split_a] = tail_a
    return tail, idx, conf

@torch.no_grad()
def fsq_sample_posterior_a_tail(
    car: torch.nn.Module,
    y_prefix: torch.Tensor,
    cvq: TailCVQ,
    args: argparse.Namespace,
) -> torch.Tensor:
    bsz = y_prefix.shape[0]
    context = car._condition(y_prefix)
    tail_a = torch.zeros(bsz, car.split_a, car.h, car.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
    values = cvq.cvq_a.level_values(device=y_prefix.device, dtype=y_prefix.float().dtype)
    for t in range(car.split_a):
        logits = car._step_logits(context, t)
        prob = F.softmax(logits.float() / max(float(args.stage5_soft_tau), 1e-6), dim=-1)
        idx_t = torch.multinomial(prob.reshape(-1, car.k_a), num_samples=1).reshape(bsz, car.h, car.w)
        tail_a[:, t] = values[idx_t]
        if t + 1 < car.split_a:
            context = car._update_context(context, idx_t, t)
    tail = torch.zeros(bsz, car.tail_ch, car.h, car.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
    tail[:, : car.split_a] = tail_a
    return tail

@torch.no_grad()
def fsq_generate_indices(car: torch.nn.Module, y_prefix: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    if fsq_strategy(args) == "maskgit":
        return car.masked_generate(y_prefix, iterations=int(getattr(args, "stage5_maskgit_iters", 6)), mode=str(args.stage5_fsq_generate))
    return car.generate_with_confidence(y_prefix, mode=str(args.stage5_fsq_generate))

@torch.no_grad()
def fsq_generate_coarse_a_tail(
    car: torch.nn.Module,
    y_prefix: torch.Tensor,
    cvq: TailCVQ,
    args: argparse.Namespace,
    *,
    residual: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    levels = fsq_coarse_levels(args)
    factor = fsq_coarse_factor(car.k_a, levels)
    bsz = y_prefix.shape[0]
    context = car._condition(y_prefix)
    coarse_idx = torch.zeros(bsz, car.split_a, car.h, car.w, device=y_prefix.device, dtype=torch.long)
    conf = torch.zeros(bsz, car.split_a, car.h, car.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
    center_idx = torch.zeros(bsz, car.split_a, car.h, car.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
    for t in range(car.split_a):
        logits = car._step_logits(context, t)
        coarse_t, conf_t = fsq_hard_coarse_from_logits(logits, car.k_a, levels, str(args.stage5_fsq_generate))
        center_t = fsq_center_index_from_coarse(coarse_t, car.k_a, levels)
        if residual:
            logits_res = fsq_residual_logits_a(car, logits.unsqueeze(1), coarse_t.unsqueeze(1), args).squeeze(1)
            res_prob = F.softmax(logits_res.float(), dim=-1)
            if str(args.stage5_fsq_generate) == "mean":
                res_grid = torch.arange(factor, device=y_prefix.device, dtype=res_prob.dtype)
                residual_t = torch.sum(res_prob * res_grid, dim=-1).round().long().clamp(0, factor - 1)
            else:
                residual_t = logits_res.argmax(dim=-1)
            res_conf = res_prob.gather(-1, residual_t.unsqueeze(-1)).squeeze(-1)
            th = float(getattr(args, "stage5_residual_threshold", -1.0))
            use_res = torch.ones_like(res_conf, dtype=torch.bool) if th < 0.0 else res_conf >= th
            residual_center = coarse_t.float() * float(factor) + residual_t.float()
            center_t = torch.where(use_res, residual_center, center_t)
            conf_t = conf_t * torch.where(use_res, res_conf, torch.ones_like(res_conf))
        coarse_idx[:, t] = coarse_t
        conf[:, t] = conf_t
        center_idx[:, t] = center_t
        context_idx = center_t.round().long().clamp(0, car.k_a - 1)
        if t + 1 < car.split_a:
            context = car._update_context(context, context_idx, t)
    tail = torch.zeros(bsz, car.tail_ch, car.h, car.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
    tail_a = fsq_decode_fractional_a(center_idx, cvq, car)
    if bool(getattr(args, "stage5_coarse_gate_to_zero", False)):
        tail_a = tail_a * (conf >= float(getattr(args, "stage5_gate_threshold", 0.55))).float()
    tail[:, : car.split_a] = tail_a
    return tail, coarse_idx, conf

@torch.no_grad()
def validate_car(loader, encoder: nn.Module, decoder: nn.Module, cvq: TailCVQ, car: TailCAR, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    cvq.eval()
    car.eval()
    meters = {k: AverageMeter() for k in ["ce", "acc_a_tf", "acc_b_tf", "acc_all_tf", "acc_a_ar", "acc_b_ar", "acc_all_ar", "psnr_prefix", "psnr_vq_oracle", "psnr_car"]}
    for imgs, _labels in loader:
        imgs = imgs.to(next(encoder.parameters()).device, non_blocking=True)
        _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
        tail_q, idx, _aux = cvq.encode(tail)
        logits_a, logits_b = car(y_prefix, idx)
        loss = car.ce_loss(logits_a, logits_b, idx)
        tf_idx = car.hard_indices_from_logits(logits_a, logits_b, mode=str(args.stage5_fsq_generate))
        pred_a = tf_idx[:, : car.split_a]
        pred_b = tf_idx[:, car.split_a :]
        ar_a, ar_b = car.generate(y_prefix, mode=str(args.stage5_fsq_generate))
        pred_idx = torch.cat([ar_a, ar_b], dim=1)
        pred_tail = cvq.decode_indices(pred_idx)
        zero = torch.zeros_like(tail)
        x_prefix = decoder(torch.cat([y_prefix, zero], dim=1)).clamp(0.0, 1.0)
        x_vq = decoder(torch.cat([y_prefix, tail_q], dim=1)).clamp(0.0, 1.0)
        x_car = decoder(torch.cat([y_prefix, pred_tail], dim=1)).clamp(0.0, 1.0)
        bsz = imgs.shape[0]
        meters["ce"].update(float(loss.item()), bsz)
        meters["acc_a_tf"].update(float((pred_a == idx[:, : car.split_a]).float().mean().item()), bsz)
        meters["acc_b_tf"].update(float((pred_b == idx[:, car.split_a :]).float().mean().item()), bsz)
        meters["acc_all_tf"].update(float((torch.cat([pred_a, pred_b], dim=1) == idx).float().mean().item()), bsz)
        meters["acc_a_ar"].update(float((ar_a == idx[:, : car.split_a]).float().mean().item()), bsz)
        meters["acc_b_ar"].update(float((ar_b == idx[:, car.split_a :]).float().mean().item()), bsz)
        meters["acc_all_ar"].update(float((pred_idx == idx).float().mean().item()), bsz)
        meters["psnr_prefix"].update(batch_metric_mean(psnr_per_image(x_prefix, imgs)), bsz)
        meters["psnr_vq_oracle"].update(batch_metric_mean(psnr_per_image(x_vq, imgs)), bsz)
        meters["psnr_car"].update(batch_metric_mean(psnr_per_image(x_car, imgs)), bsz)
    out = {k: v.avg for k, v in meters.items()}
    out["tail_gain_car"] = out["psnr_car"] - out["psnr_prefix"]
    out["gap_car_to_vq_oracle"] = out["psnr_vq_oracle"] - out["psnr_car"]
    return out

@torch.no_grad()
def validate_fsq_spatial_coarse_a(loader, encoder: nn.Module, decoder: nn.Module, cvq: TailCVQ, car: torch.nn.Module, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    cvq.eval()
    car.eval()
    strategy = fsq_strategy(args)
    residual = strategy == "coarse_a_residual"
    levels = fsq_coarse_levels(args)
    meters = {
        k: AverageMeter()
        for k in [
            "ce",
            "residual",
            "acc_a_tf",
            "acc_a_ar",
            "psnr_prefix",
            "psnr_vq_oracle",
            "psnr_car",
            "gate_keep_a",
        ]
    }
    for imgs, _labels in loader:
        imgs = imgs.to(next(encoder.parameters()).device, non_blocking=True)
        _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
        tail_q, idx, _aux = cvq.encode(tail)
        logits_a, logits_b = car(y_prefix, idx)
        del logits_b
        ce = fsq_ce_a_coarse(car, logits_a, idx, args)
        residual_loss = ce.new_tensor(0.0)
        if residual:
            residual_loss = fsq_ce_a_residual(car, logits_a, idx, args)
        tf_coarse, _tf_conf = fsq_hard_coarse_from_logits(logits_a, car.k_a, levels, str(args.stage5_fsq_generate))
        true_coarse = fsq_coarse_target(idx[:, : car.split_a], car.k_a, levels)
        pred_tail, ar_coarse, ar_conf = fsq_generate_coarse_a_tail(car, y_prefix, cvq, args, residual=residual)
        zero = torch.zeros_like(tail)
        x_prefix = decoder(torch.cat([y_prefix, zero], dim=1)).clamp(0.0, 1.0)
        x_vq = decoder(torch.cat([y_prefix, tail_q], dim=1)).clamp(0.0, 1.0)
        x_car = decoder(torch.cat([y_prefix, pred_tail], dim=1)).clamp(0.0, 1.0)
        bsz = imgs.shape[0]
        meters["ce"].update(float(ce.item()), bsz)
        meters["residual"].update(float(residual_loss.item()), bsz)
        meters["acc_a_tf"].update(float((tf_coarse == true_coarse).float().mean().item()), bsz)
        meters["acc_a_ar"].update(float((ar_coarse == true_coarse).float().mean().item()), bsz)
        meters["psnr_prefix"].update(batch_metric_mean(psnr_per_image(x_prefix, imgs)), bsz)
        meters["psnr_vq_oracle"].update(batch_metric_mean(psnr_per_image(x_vq, imgs)), bsz)
        meters["psnr_car"].update(batch_metric_mean(psnr_per_image(x_car, imgs)), bsz)
        meters["gate_keep_a"].update(float((ar_conf >= float(getattr(args, "stage5_gate_threshold", 0.55))).float().mean().item()), bsz)
    out = {k: v.avg for k, v in meters.items()}
    out["acc_b_tf"] = 0.0
    out["acc_all_tf"] = out["acc_a_tf"]
    out["acc_b_ar"] = 0.0
    out["acc_all_ar"] = out["acc_a_ar"]
    out["psnr_car_raw"] = out["psnr_car"]
    out["psnr_car_a"] = out["psnr_car"]
    out["psnr_car_gated"] = out["psnr_car"]
    out["gate_keep"] = out["gate_keep_a"] * 0.5
    out["gate_keep_b"] = 0.0
    out["tail_gain_car"] = out["psnr_car"] - out["psnr_prefix"]
    out["gap_car_to_vq_oracle"] = out["psnr_vq_oracle"] - out["psnr_car"]
    out["tail_gain_car_raw"] = out["tail_gain_car"]
    out["tail_gain_car_a"] = out["tail_gain_car"]
    out["tail_gain_car_gated"] = out["tail_gain_car"]
    out["coarse_levels"] = float(levels)
    out["coarse_residual"] = float(residual)
    return out

@torch.no_grad()
def validate_fsq_spatial_posterior_a(loader, encoder: nn.Module, decoder: nn.Module, cvq: TailCVQ, car: torch.nn.Module, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    cvq.eval()
    car.eval()
    softar_posterior = fsq_is_softar_posterior_strategy(fsq_strategy(args))
    posterior_all = fsq_is_posterior_all_strategy(fsq_strategy(args))
    thresholds = fsq_gate_sweep_values(args)
    alphas = fsq_alpha_sweep_values(args)
    default_alpha = fsq_tail_scale(args)
    mc_samples = max(0, int(getattr(args, "stage5_mc_samples", 0)))
    meters = {
        k: AverageMeter()
        for k in [
            "ce",
            "acc_a_tf",
            "acc_b_tf",
            "acc_all_tf",
            "acc_a_ar",
            "acc_b_ar",
            "acc_all_ar",
            "psnr_prefix",
            "psnr_vq_oracle",
            "psnr_car_raw",
            "posterior_conf_a",
        ]
    }
    psnr_by_pair = {(th, alpha): AverageMeter() for th in thresholds for alpha in alphas}
    keep_by_th = {th: AverageMeter() for th in thresholds}
    psnr_mc_by_alpha = {alpha: AverageMeter() for alpha in alphas} if mc_samples > 0 else {}
    for imgs, _labels in loader:
        imgs = imgs.to(next(encoder.parameters()).device, non_blocking=True)
        _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
        tail_q, idx, _aux = cvq.encode(tail)
        if posterior_all:
            pred_tail_raw, pred_ar, conf_ar, _gate_ar, logits_a, logits_b = fsq_softar_posterior_tail(
                car,
                y_prefix,
                cvq,
                args,
                threshold=None,
                soft_gate=False,
                scale=1.0,
            )
            ce = car.ce_loss(logits_a, logits_b, idx)
            pred_tf = pred_ar
        elif softar_posterior:
            pred_tail_raw, pred_ar, conf_ar, _gate_ar, logits_a = fsq_softar_posterior_a_tail(
                car,
                y_prefix,
                cvq,
                args,
                threshold=None,
                soft_gate=False,
                scale=1.0,
            )
            ce = fsq_ce_a(car, logits_a, idx)
            pred_tf = pred_ar
        else:
            logits_a, logits_b = car(y_prefix, idx)
            del logits_b
            ce = fsq_ce_a(car, logits_a, idx)
            pred_tf = car._logits_to_idx(logits_a, car.k_a, str(args.stage5_fsq_generate))
            pred_tail_raw, pred_ar, conf_ar = fsq_generate_posterior_a_tail(car, y_prefix, cvq, args)
        zero = torch.zeros_like(tail)
        x_prefix = decoder(torch.cat([y_prefix, zero], dim=1)).clamp(0.0, 1.0)
        x_vq = decoder(torch.cat([y_prefix, tail_q], dim=1)).clamp(0.0, 1.0)
        x_raw = decoder(torch.cat([y_prefix, pred_tail_raw * default_alpha], dim=1)).clamp(0.0, 1.0)
        bsz = imgs.shape[0]
        pred_tf_a = pred_tf[:, : car.split_a] if posterior_all else pred_tf
        pred_ar_a = pred_ar[:, : car.split_a] if posterior_all else pred_ar
        meters["ce"].update(float(ce.item()), bsz)
        meters["acc_a_tf"].update(float((pred_tf_a == idx[:, : car.split_a]).float().mean().item()), bsz)
        meters["acc_a_ar"].update(float((pred_ar_a == idx[:, : car.split_a]).float().mean().item()), bsz)
        if posterior_all:
            meters["acc_b_tf"].update(float((pred_tf[:, car.split_a :] == idx[:, car.split_a :]).float().mean().item()), bsz)
            meters["acc_b_ar"].update(float((pred_ar[:, car.split_a :] == idx[:, car.split_a :]).float().mean().item()), bsz)
            meters["acc_all_tf"].update(float((pred_tf == idx).float().mean().item()), bsz)
            meters["acc_all_ar"].update(float((pred_ar == idx).float().mean().item()), bsz)
        else:
            meters["acc_b_tf"].update(0.0, bsz)
            meters["acc_b_ar"].update(0.0, bsz)
            meters["acc_all_tf"].update(float((pred_tf_a == idx[:, : car.split_a]).float().mean().item()), bsz)
            meters["acc_all_ar"].update(float((pred_ar_a == idx[:, : car.split_a]).float().mean().item()), bsz)
        meters["psnr_prefix"].update(batch_metric_mean(psnr_per_image(x_prefix, imgs)), bsz)
        meters["psnr_vq_oracle"].update(batch_metric_mean(psnr_per_image(x_vq, imgs)), bsz)
        meters["psnr_car_raw"].update(batch_metric_mean(psnr_per_image(x_raw, imgs)), bsz)
        meters["posterior_conf_a"].update(float(conf_ar.mean().item()), bsz)
        for th in thresholds:
            if float(th) < 0.0:
                pred_tail = pred_tail_raw
                keep = torch.ones_like(conf_ar)
            else:
                keep = (conf_ar >= float(th)).to(dtype=pred_tail_raw.dtype)
            keep_by_th[th].update(float(keep.mean().item()), bsz)
            for alpha in alphas:
                if posterior_all:
                    pred_tail = pred_tail_raw * keep * float(alpha)
                else:
                    pred_tail = pred_tail_raw.clone()
                    pred_tail[:, : car.split_a] = pred_tail[:, : car.split_a] * keep * float(alpha)
                x_car = decoder(torch.cat([y_prefix, pred_tail], dim=1)).clamp(0.0, 1.0)
                psnr_by_pair[(th, alpha)].update(batch_metric_mean(psnr_per_image(x_car, imgs)), bsz)
        if mc_samples > 0:
            x_sum_by_alpha = {alpha: torch.zeros_like(x_prefix) for alpha in alphas}
            for _sample_idx in range(mc_samples):
                sample_tail = fsq_sample_posterior_a_tail(car, y_prefix, cvq, args)
                for alpha in alphas:
                    tail_alpha = sample_tail * float(alpha)
                    x_sample = decoder(torch.cat([y_prefix, tail_alpha], dim=1)).clamp(0.0, 1.0)
                    x_sum_by_alpha[alpha] = x_sum_by_alpha[alpha] + x_sample
            for alpha in alphas:
                x_mc = (x_sum_by_alpha[alpha] / float(mc_samples)).clamp(0.0, 1.0)
                psnr_mc_by_alpha[alpha].update(batch_metric_mean(psnr_per_image(x_mc, imgs)), bsz)
    out = {k: v.avg for k, v in meters.items()}
    best_th, best_alpha = max(psnr_by_pair, key=lambda pair: psnr_by_pair[pair].avg)
    best_psnr = psnr_by_pair[(best_th, best_alpha)].avg
    best_decode_mode = 0.0
    if mc_samples > 0:
        best_mc_alpha = max(alphas, key=lambda alpha: psnr_mc_by_alpha[alpha].avg)
        best_mc_psnr = psnr_mc_by_alpha[best_mc_alpha].avg
        out["psnr_car_mc"] = best_mc_psnr
        out["best_mc_alpha"] = float(best_mc_alpha)
        out["stage5_mc_samples"] = float(mc_samples)
        if best_mc_psnr > best_psnr:
            best_psnr = best_mc_psnr
            best_alpha = best_mc_alpha
            best_th = -1.0
            best_decode_mode = 1.0
    raw_th = -1.0 if -1.0 in thresholds else thresholds[0]
    out["psnr_car"] = best_psnr
    out["psnr_car_a"] = best_psnr
    out["psnr_car_gated"] = best_psnr
    out["gate_keep_a"] = keep_by_th[best_th].avg
    out["gate_keep_b"] = keep_by_th[best_th].avg if posterior_all else 0.0
    out["gate_keep"] = keep_by_th[best_th].avg if posterior_all else 0.5 * keep_by_th[best_th].avg
    out["best_gate_threshold"] = float(best_th)
    out["best_tail_alpha"] = float(best_alpha)
    out["best_decode_mode"] = best_decode_mode
    out["tail_alpha_default"] = float(default_alpha)
    out["tail_gain_car"] = out["psnr_car"] - out["psnr_prefix"]
    out["tail_gain_car_a"] = out["tail_gain_car"]
    out["tail_gain_car_gated"] = out["tail_gain_car"]
    out["tail_gain_car_raw"] = out["psnr_car_raw"] - out["psnr_prefix"]
    out["gap_car_to_vq_oracle"] = out["psnr_vq_oracle"] - out["psnr_car"]
    for th in thresholds:
        key = "none" if float(th) < 0.0 else fsq_metric_float_key(th)
        out[f"psnr_th_{key}"] = max(psnr_by_pair[(th, alpha)].avg for alpha in alphas)
        out[f"keep_th_{key}"] = keep_by_th[th].avg
    for alpha in alphas:
        key = fsq_metric_float_key(alpha)
        out[f"psnr_alpha_{key}"] = psnr_by_pair[(raw_th, alpha)].avg
        if mc_samples > 0:
            out[f"psnr_mc_alpha_{key}"] = psnr_mc_by_alpha[alpha].avg
    return out

@torch.no_grad()
def validate_fsq_spatial_car(loader, encoder: nn.Module, decoder: nn.Module, cvq: TailCVQ, car: torch.nn.Module, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    cvq.eval()
    car.eval()
    strategy = fsq_strategy(args)
    if fsq_is_posterior_strategy(strategy):
        return validate_fsq_spatial_posterior_a(loader, encoder, decoder, cvq, car, args)
    if fsq_is_coarse_strategy(strategy):
        return validate_fsq_spatial_coarse_a(loader, encoder, decoder, cvq, car, args)
    meters = {
        k: AverageMeter()
        for k in [
            "ce",
            "acc_a_tf",
            "acc_b_tf",
            "acc_all_tf",
            "acc_a_ar",
            "acc_b_ar",
            "acc_all_ar",
            "psnr_prefix",
            "psnr_vq_oracle",
            "psnr_car_raw",
            "psnr_car_a",
            "psnr_car_gated",
            "gate_keep",
            "gate_keep_a",
            "gate_keep_b",
        ]
    }
    for imgs, _labels in loader:
        imgs = imgs.to(next(encoder.parameters()).device, non_blocking=True)
        _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
        tail_q, idx, _aux = cvq.encode(tail)
        logits_a, logits_b = car(y_prefix, idx)
        loss = car.ce_loss(logits_a, logits_b, idx)
        tf_idx = car.hard_indices_from_logits(logits_a, logits_b, mode=str(args.stage5_fsq_generate))
        pred_a = tf_idx[:, : car.split_a]
        pred_b = tf_idx[:, car.split_a :]
        pred_idx, conf = fsq_generate_indices(car, y_prefix, args)
        pred_tail_raw = cvq.decode_indices(pred_idx)
        pred_tail_a = fsq_a_only_tail(pred_tail_raw, car)
        pred_tail_gated, gate = fsq_gated_tail(pred_tail_raw, conf, car, args)
        zero = torch.zeros_like(tail)
        x_prefix = decoder(torch.cat([y_prefix, zero], dim=1)).clamp(0.0, 1.0)
        x_vq = decoder(torch.cat([y_prefix, tail_q], dim=1)).clamp(0.0, 1.0)
        x_raw = decoder(torch.cat([y_prefix, pred_tail_raw], dim=1)).clamp(0.0, 1.0)
        x_a = decoder(torch.cat([y_prefix, pred_tail_a], dim=1)).clamp(0.0, 1.0)
        x_gated = decoder(torch.cat([y_prefix, pred_tail_gated], dim=1)).clamp(0.0, 1.0)
        idx_a = idx[:, : car.split_a]
        idx_b = idx[:, car.split_a :]
        bsz = imgs.shape[0]
        meters["ce"].update(float(loss.item()), bsz)
        meters["acc_a_tf"].update(float((pred_a == idx_a).float().mean().item()), bsz)
        meters["acc_b_tf"].update(float((pred_b == idx_b).float().mean().item()), bsz)
        meters["acc_all_tf"].update(float((tf_idx == idx).float().mean().item()), bsz)
        meters["acc_a_ar"].update(float((pred_idx[:, : car.split_a] == idx_a).float().mean().item()), bsz)
        meters["acc_b_ar"].update(float((pred_idx[:, car.split_a :] == idx_b).float().mean().item()), bsz)
        meters["acc_all_ar"].update(float((pred_idx == idx).float().mean().item()), bsz)
        meters["psnr_prefix"].update(batch_metric_mean(psnr_per_image(x_prefix, imgs)), bsz)
        meters["psnr_vq_oracle"].update(batch_metric_mean(psnr_per_image(x_vq, imgs)), bsz)
        meters["psnr_car_raw"].update(batch_metric_mean(psnr_per_image(x_raw, imgs)), bsz)
        meters["psnr_car_a"].update(batch_metric_mean(psnr_per_image(x_a, imgs)), bsz)
        meters["psnr_car_gated"].update(batch_metric_mean(psnr_per_image(x_gated, imgs)), bsz)
        meters["gate_keep"].update(float(gate.float().mean().item()), bsz)
        meters["gate_keep_a"].update(float(gate[:, : car.split_a].float().mean().item()), bsz)
        meters["gate_keep_b"].update(float(gate[:, car.split_a :].float().mean().item()), bsz)
    out = {k: v.avg for k, v in meters.items()}
    if strategy == "two_stage":
        out["psnr_car"] = max(out["psnr_car_a"], out["psnr_car_raw"])
    elif strategy == "gated":
        out["psnr_car"] = out["psnr_car_gated"]
    else:
        out["psnr_car"] = out["psnr_car_raw"]
    out["tail_gain_car"] = out["psnr_car"] - out["psnr_prefix"]
    out["gap_car_to_vq_oracle"] = out["psnr_vq_oracle"] - out["psnr_car"]
    out["tail_gain_car_raw"] = out["psnr_car_raw"] - out["psnr_prefix"]
    out["tail_gain_car_a"] = out["psnr_car_a"] - out["psnr_prefix"]
    out["tail_gain_car_gated"] = out["psnr_car_gated"] - out["psnr_prefix"]
    return out

def train_stage5_fsq_spatial(
    args: argparse.Namespace,
    train_loader,
    val_loader,
    cfg,
    encoder: nn.Module,
    decoder: nn.Module,
    cvq: TailCVQ,
    car: torch.nn.Module,
) -> None:
    opt_groups = [{"params": car.parameters(), "lr": float(args.car_lr)}]
    if bool(getattr(args, "stage5_train_encoder", False)):
        opt_groups.append({"params": [p for p in encoder.parameters() if p.requires_grad], "lr": float(args.lr)})
    if bool(getattr(args, "stage5_train_cvq", False)):
        opt_groups.append({"params": [p for p in cvq.parameters() if p.requires_grad], "lr": float(getattr(args, "stage5_cvq_lr", args.lr))})
    if bool(getattr(args, "stage5_train_decoder", False)):
        opt_groups.append({"params": decoder.parameters(), "lr": float(getattr(args, "decoder_lr", 2e-5))})
    optimizer = optim.Adam(opt_groups)
    best = -1.0
    strategy = fsq_strategy(args)
    print_run_header(args, f"Stage 5 | FSQ Spatial Swin CAR prediction q17:36 | {strategy}", len(train_loader.dataset), len(val_loader.dataset))
    print(
        f"stage5_fsq_spatial levels_a={car.k_a} levels_b={car.k_b} "
        f"window={getattr(car, 'window_size', '<none>')} "
        f"strategy={strategy} "
        f"lambda_stage5_rec={float(args.lambda_stage5_rec):g} "
        f"lambda_stage5_prefix={float(getattr(args, 'lambda_stage5_prefix', 0.0)):g} "
        f"lambda_stage5_gain_margin={float(getattr(args, 'lambda_stage5_gain_margin', 0.0)):g} "
        f"stage5_gain_margin_db={float(getattr(args, 'stage5_gain_margin_db', 0.5)):g} "
        f"lambda_stage5_value={float(args.lambda_stage5_value):g} "
        f"lambda_stage5_vq={float(getattr(args, 'lambda_stage5_vq', 0.0)):g} "
        f"lambda_stage5_ordinal={float(getattr(args, 'lambda_stage5_ordinal', 0.0)):g} "
        f"lambda_stage5_residual={float(getattr(args, 'lambda_stage5_residual', 1.0)):g} "
        f"lambda_stage5_gate_l1={float(getattr(args, 'lambda_stage5_gate_l1', 0.0)):g} "
        f"coarse_levels={int(getattr(args, 'stage5_fsq_coarse_levels', 4))} "
        f"residual_threshold={float(getattr(args, 'stage5_residual_threshold', -1.0)):g} "
        f"coarse_gate_to_zero={int(bool(getattr(args, 'stage5_coarse_gate_to_zero', False)))} "
        f"gate_threshold={float(getattr(args, 'stage5_gate_threshold', 0.55)):g} "
        f"gate_soft_temp={float(getattr(args, 'stage5_gate_soft_temp', 0.08)):g} "
        f"tail_scale={float(getattr(args, 'stage5_tail_scale', 1.0)):g} "
        f"alpha_sweep={str(getattr(args, 'stage5_alpha_sweep', ''))} "
        f"mc_samples={int(getattr(args, 'stage5_mc_samples', 0))} "
        f"gate_b_threshold={float(getattr(args, 'stage5_gate_b_threshold', -1.0)):g} "
        f"two_stage_a_epochs={int(getattr(args, 'stage5_two_stage_a_epochs', 0))} "
        f"maskgit_iters={int(getattr(args, 'stage5_maskgit_iters', 6))} "
        f"maskgit_mask_prob={float(getattr(args, 'stage5_maskgit_mask_prob', 0.6)):g} "
        f"train_encoder={int(bool(getattr(args, 'stage5_train_encoder', False)))} "
        f"freeze_encoder_body={int(bool(getattr(args, 'stage5_freeze_encoder_body', False)))} "
        f"train_prefix_head_only={int(bool(getattr(args, 'stage5_train_prefix_head_only', False)))} "
        f"encoder_lr={float(args.lr):g} "
        f"train_cvq={int(bool(getattr(args, 'stage5_train_cvq', False)))} "
        f"cvq_lr={float(getattr(args, 'stage5_cvq_lr', args.lr)):g} "
        f"train_decoder={int(bool(getattr(args, 'stage5_train_decoder', False)))} "
        f"decoder_lr={float(getattr(args, 'decoder_lr', 2e-5)):g} "
        f"min_tail_gain_car={float(args.stage5_min_tail_gain_car):g}"
    )
    if int(args.epochs) <= 0:
        val_metrics = validate_fsq_spatial_car(val_loader, encoder, decoder, cvq, car, args)
        print(f"[stage5 val 000] {format_metrics(val_metrics)} score=psnr_car")
        save_checkpoint(ckpt_path(args, "stage5", "latest"), stage="stage5", epoch=0, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
        return
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train(bool(getattr(args, "stage5_train_encoder", False)))
        decoder.train(bool(getattr(args, "stage5_train_decoder", False)))
        cvq.train(bool(getattr(args, "stage5_train_cvq", False)))
        car.train()
        meters = {k: AverageMeter() for k in ["loss", "ce", "rec", "prefix", "gain_margin", "value", "ordinal", "vq", "residual", "gate_l1", "acc_a", "acc_b", "acc_all"]}
        two_stage_phase_a = strategy == "two_stage" and epoch <= int(getattr(args, "stage5_two_stage_a_epochs", 0))
        coarse_strategy = fsq_is_coarse_strategy(strategy)
        posterior_strategy = fsq_is_posterior_strategy(strategy)
        softar_posterior = fsq_is_softar_posterior_strategy(strategy)
        posterior_all = fsq_is_posterior_all_strategy(strategy)
        coarse_residual = strategy == "coarse_a_residual"
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
            if bool(getattr(args, "stage5_train_cvq", False)):
                tail_q, idx, vq_loss, _aux = cvq(tail)
            else:
                with torch.no_grad():
                    tail_q, idx, _aux = cvq.encode(tail)
                vq_loss = torch.zeros((), device=cfg.device)

            residual_loss = torch.zeros((), device=cfg.device)
            gate_l1 = torch.zeros((), device=cfg.device)
            posterior_gate = None
            precomputed_tail_value = None
            precomputed_tail_soft = None
            if posterior_strategy:
                if softar_posterior:
                    if posterior_all:
                        precomputed_tail_value, _pred_ar, _conf_ar, posterior_gate, logits_a, logits_b = fsq_softar_posterior_tail(
                            car,
                            y_prefix,
                            cvq,
                            args,
                            threshold=float(getattr(args, "stage5_gate_threshold", 0.55)),
                            soft_gate=True,
                            scale=1.0,
                        )
                    else:
                        precomputed_tail_value, _pred_ar, _conf_ar, posterior_gate, logits_a = fsq_softar_posterior_a_tail(
                            car,
                            y_prefix,
                            cvq,
                            args,
                            threshold=float(getattr(args, "stage5_gate_threshold", 0.55)),
                            soft_gate=True,
                            scale=1.0,
                        )
                        logits_b = torch.empty(0, device=cfg.device)
                    precomputed_tail_soft = precomputed_tail_value * fsq_tail_scale(args)
                else:
                    logits_a, logits_b = car(y_prefix, idx)
                ce = car.ce_loss(logits_a, logits_b, idx) if posterior_all else fsq_ce_a(car, logits_a, idx)
            elif coarse_strategy:
                logits_a, logits_b = car(y_prefix, idx)
                ce = fsq_ce_a_coarse(car, logits_a, idx, args)
                if coarse_residual:
                    residual_loss = fsq_ce_a_residual(car, logits_a, idx, args)
                    ce = ce + float(getattr(args, "lambda_stage5_residual", 1.0)) * residual_loss
            elif strategy == "maskgit":
                visible_mask = torch.rand_like(idx.float()) > float(getattr(args, "stage5_maskgit_mask_prob", 0.6))
                target_mask = ~visible_mask
                if not bool(target_mask.any()):
                    target_mask[:, 0, 0, 0] = True
                    visible_mask[:, 0, 0, 0] = False
                logits_a, logits_b = car.masked_forward(y_prefix, idx, visible_mask)
                ce = car.masked_ce_loss(logits_a, logits_b, idx, target_mask)
            else:
                teacher_idx = idx
                if strategy == "two_stage" and not two_stage_phase_a:
                    with torch.no_grad():
                        pred_teacher, _conf_teacher = car.generate_with_confidence(y_prefix, mode=str(args.stage5_fsq_generate))
                    teacher_idx = idx.clone()
                    teacher_idx[:, : car.split_a] = pred_teacher[:, : car.split_a]
                logits_a, logits_b = car(y_prefix, teacher_idx)
                if two_stage_phase_a:
                    ce = fsq_ce_a(car, logits_a, idx)
                elif strategy == "two_stage":
                    ce = 0.5 * fsq_ce_a(car, logits_a, idx) + fsq_ce_b(car, logits_b, idx)
                else:
                    ce = car.ce_loss(logits_a, logits_b, idx)

            rec = ce.new_tensor(0.0)
            prefix_loss = ce.new_tensor(0.0)
            gain_margin_loss = ce.new_tensor(0.0)
            value = ce.new_tensor(0.0)
            ordinal = ce.new_tensor(0.0)
            tail_soft = precomputed_tail_soft
            loss = ce + float(getattr(args, "lambda_stage5_vq", 0.0)) * vq_loss
            ordinal_weight = float(getattr(args, "lambda_stage5_ordinal", 0.0))
            if ordinal_weight > 0.0 and not coarse_strategy:
                ordinal_part = "all" if posterior_all else ("a" if (two_stage_phase_a or posterior_strategy) else "all")
                ordinal = fsq_ordinal_loss(car, logits_a, logits_b, idx, part=ordinal_part)
                loss = loss + ordinal_weight * ordinal
            if float(args.lambda_stage5_value) > 0.0:
                if posterior_strategy:
                    if precomputed_tail_value is None:
                        tail_value, _posterior_conf, posterior_gate = fsq_posterior_a_tail_from_logits(
                            car,
                            logits_a,
                            cvq,
                            args,
                            threshold=float(getattr(args, "stage5_gate_threshold", 0.55)),
                            soft_gate=True,
                            scale=1.0,
                        )
                    else:
                        tail_value = precomputed_tail_value
                    if posterior_all:
                        value = (tail_value.float() - tail_q.float()).square().mean()
                    else:
                        value = (tail_value[:, : car.split_a].float() - tail_q[:, : car.split_a].float()).square().mean()
                    if tail_soft is None:
                        tail_soft = tail_value * fsq_tail_scale(args)
                elif coarse_strategy:
                    tail_soft = fsq_soft_coarse_a_tail(car, logits_a, cvq, args, residual=coarse_residual)
                    value = (tail_soft[:, : car.split_a].float() - tail_q[:, : car.split_a].float()).square().mean()
                else:
                    tail_soft = car.soft_decode(logits_a, logits_b, cvq, tau=float(args.stage5_soft_tau))
                if two_stage_phase_a:
                    value = (tail_soft[:, : car.split_a].float() - tail_q[:, : car.split_a].float()).square().mean()
                elif not coarse_strategy and not posterior_strategy:
                    value = (tail_soft.float() - tail_q.float()).square().mean()
                loss = loss + float(args.lambda_stage5_value) * value
            gate_l1_weight = float(getattr(args, "lambda_stage5_gate_l1", 0.0))
            if posterior_strategy and gate_l1_weight > 0.0:
                if tail_soft is None or posterior_gate is None:
                    tail_soft, _posterior_conf, posterior_gate = fsq_posterior_a_tail_from_logits(
                        car,
                        logits_a,
                        cvq,
                        args,
                        threshold=float(getattr(args, "stage5_gate_threshold", 0.55)),
                        soft_gate=True,
                    )
                gate_l1 = posterior_gate.mean()
                loss = loss + gate_l1_weight * gate_l1
            if float(args.lambda_stage5_rec) > 0.0:
                if tail_soft is None:
                    if posterior_strategy:
                        tail_soft, _posterior_conf, posterior_gate = fsq_posterior_a_tail_from_logits(
                            car,
                            logits_a,
                            cvq,
                            args,
                            threshold=float(getattr(args, "stage5_gate_threshold", 0.55)),
                            soft_gate=True,
                        )
                    elif coarse_strategy:
                        tail_soft = fsq_soft_coarse_a_tail(car, logits_a, cvq, args, residual=coarse_residual)
                    else:
                        tail_soft = car.soft_decode(logits_a, logits_b, cvq, tau=float(args.stage5_soft_tau))
                if posterior_strategy:
                    x_soft = decoder(torch.cat([y_prefix, tail_soft], dim=1))
                    rec = recon_loss(x_soft, imgs)
                    loss = loss + float(args.lambda_stage5_rec) * rec
                else:
                    with torch.no_grad():
                        if coarse_strategy:
                            tail_hard, _hard_coarse, hard_conf = fsq_hard_coarse_a_tail_from_logits(car, logits_a, cvq, args, residual=coarse_residual)
                        else:
                            hard_idx = car.hard_indices_from_logits(logits_a, logits_b, mode=str(args.stage5_fsq_generate))
                            tail_hard = cvq.decode_indices(hard_idx)
                        if two_stage_phase_a:
                            tail_hard = fsq_a_only_tail(tail_hard, car)
                        elif strategy == "gated" and not coarse_strategy:
                            hard_conf = car.confidence_from_logits(logits_a, logits_b, hard_idx, mode=str(args.stage5_fsq_generate))
                            hard_gate = fsq_confidence_mask(hard_conf, car, args)
                            tail_hard = tail_hard * hard_gate.float()
                    tail_for_rec = tail_soft
                    if two_stage_phase_a:
                        tail_for_rec = fsq_a_only_tail(tail_for_rec, car)
                    elif coarse_strategy and bool(getattr(args, "stage5_coarse_gate_to_zero", False)):
                        hard_gate = hard_conf >= float(getattr(args, "stage5_gate_threshold", 0.55))
                        tail_for_rec = tail_for_rec.clone()
                        tail_for_rec[:, : car.split_a] = tail_for_rec[:, : car.split_a] * hard_gate.float()
                    elif strategy == "gated" and not coarse_strategy:
                        tail_for_rec = tail_for_rec * hard_gate.float()
                    tail_st = tail_hard + (tail_for_rec - tail_for_rec.detach())
                    x_soft = decoder(torch.cat([y_prefix, tail_st], dim=1))
                    rec = recon_loss(x_soft, imgs)
                    loss = loss + float(args.lambda_stage5_rec) * rec
            if float(getattr(args, "lambda_stage5_prefix", 0.0)) > 0.0:
                zero_tail = torch.zeros_like(tail)
                x_prefix_train = decoder(torch.cat([y_prefix, zero_tail], dim=1))
                prefix_loss = recon_loss(x_prefix_train, imgs)
                loss = loss + float(getattr(args, "lambda_stage5_prefix", 0.0)) * prefix_loss
            if float(getattr(args, "lambda_stage5_gain_margin", 0.0)) > 0.0:
                if rec.detach().item() == 0.0:
                    if tail_soft is None:
                        if posterior_strategy:
                            tail_soft, _posterior_conf, posterior_gate = fsq_posterior_a_tail_from_logits(
                                car,
                                logits_a,
                                cvq,
                                args,
                                threshold=float(getattr(args, "stage5_gate_threshold", 0.55)),
                                soft_gate=True,
                            )
                        elif coarse_strategy:
                            tail_soft = fsq_soft_coarse_a_tail(car, logits_a, cvq, args, residual=coarse_residual)
                        else:
                            tail_soft = car.soft_decode(logits_a, logits_b, cvq, tau=float(args.stage5_soft_tau))
                    rec = recon_loss(decoder(torch.cat([y_prefix, tail_soft], dim=1)), imgs)
                if prefix_loss.detach().item() == 0.0:
                    with torch.no_grad():
                        zero_tail = torch.zeros_like(tail)
                        prefix_for_margin = recon_loss(decoder(torch.cat([y_prefix, zero_tail], dim=1)), imgs)
                else:
                    prefix_for_margin = prefix_loss.detach()
                ratio = 10.0 ** (-float(getattr(args, "stage5_gain_margin_db", 0.5)) / 10.0)
                gain_margin_loss = F.relu(rec - prefix_for_margin.detach() * ratio)
                loss = loss + float(getattr(args, "lambda_stage5_gain_margin", 0.0)) * gain_margin_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                idx_a = idx[:, : car.split_a]
                idx_b = idx[:, car.split_a :]
                if posterior_strategy:
                    pred_a = car._logits_to_idx(logits_a, car.k_a, str(args.stage5_fsq_generate))
                    if posterior_all:
                        pred_b = car._logits_to_idx(logits_b, car.k_b, str(args.stage5_fsq_generate))
                        pred = torch.cat([pred_a, pred_b], dim=1)
                        idx_a_metric = idx_a
                        idx_b_metric = idx_b
                        idx_metric = idx
                    else:
                        pred_b = torch.zeros_like(idx_b)
                        pred = pred_a
                        idx_a_metric = idx_a
                        idx_b_metric = torch.zeros_like(idx_b)
                        idx_metric = idx_a
                elif coarse_strategy:
                    pred_a, _conf_a = fsq_hard_coarse_from_logits(logits_a, car.k_a, fsq_coarse_levels(args), str(args.stage5_fsq_generate))
                    idx_a_metric = fsq_coarse_target(idx_a, car.k_a, fsq_coarse_levels(args))
                    pred_b = torch.zeros_like(idx_b)
                    idx_b_metric = torch.zeros_like(idx_b)
                    pred = pred_a
                    idx_metric = idx_a_metric
                else:
                    pred = car.hard_indices_from_logits(logits_a, logits_b, mode=str(args.stage5_fsq_generate))
                    pred_a = pred[:, : car.split_a]
                    pred_b = pred[:, car.split_a :]
                    idx_a_metric = idx_a
                    idx_b_metric = idx_b
                    idx_metric = idx
                bsz = imgs.shape[0]
                meters["loss"].update(float(loss.item()), bsz)
                meters["ce"].update(float(ce.item()), bsz)
                meters["rec"].update(float(rec.item()), bsz)
                meters["prefix"].update(float(prefix_loss.item()), bsz)
                meters["gain_margin"].update(float(gain_margin_loss.item()), bsz)
                meters["value"].update(float(value.item()), bsz)
                meters["ordinal"].update(float(ordinal.item()), bsz)
                meters["vq"].update(float(vq_loss.item()), bsz)
                meters["residual"].update(float(residual_loss.item()), bsz)
                meters["gate_l1"].update(float(gate_l1.item()), bsz)
                meters["acc_a"].update(float((pred_a == idx_a_metric).float().mean().item()), bsz)
                if coarse_strategy or (posterior_strategy and not posterior_all):
                    meters["acc_b"].update(0.0, bsz)
                else:
                    meters["acc_b"].update(float((pred_b == idx_b_metric).float().mean().item()), bsz)
                meters["acc_all"].update(float((pred == idx_metric).float().mean().item()), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        metrics["two_stage_phase_a"] = 1.0 if two_stage_phase_a else 0.0
        print_epoch("stage5", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate_fsq_spatial_car(val_loader, encoder, decoder, cvq, car, args)
            val_metrics["min_tail_gain_car"] = float(args.stage5_min_tail_gain_car)
            score = val_metrics["psnr_car"]
            print(f"[stage5 val {epoch:03d}] {format_metrics(val_metrics)} score=psnr_car")
            if score > best and stage5_best_allowed(args, val_metrics):
                best = score
                save_checkpoint(ckpt_path(args, "stage5", "best"), stage="stage5", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
            elif score > best:
                print(f"[stage5 val {epoch:03d}] skip_best_save=tail_gain_car")
        if should_save_latest(args, epoch):
            save_checkpoint(ckpt_path(args, "stage5", "latest"), stage="stage5", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
    save_checkpoint(ckpt_path(args, "stage5", "latest"), stage="stage5", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)

def train_stage5(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder, cvq, car = build_models(args, cfg.device)
    src = args.init_ckpt or ckpt_path(args, "stage3", "best")
    load_experiment_checkpoint(src, encoder=encoder, decoder=decoder, cvq=cvq, strict=True)
    if str(getattr(args, "stage5_init_car_ckpt", "")):
        load_experiment_checkpoint(str(args.stage5_init_car_ckpt), car=car, strict=True)
    apply_stage5_encoder_trainability(encoder, args)
    freeze_module(decoder, bool(getattr(args, "stage5_train_decoder", False)))
    freeze_module(cvq, bool(getattr(args, "stage5_train_cvq", False)))
    if is_fsq_spatial_car(cvq, car):
        train_stage5_fsq_spatial(args, train_loader, val_loader, cfg, encoder, decoder, cvq, car)
        return
    optimizer = optim.Adam(car.parameters(), lr=float(args.car_lr))
    best = -1.0
    print_run_header(args, f"Stage 5 | CAR prediction q{int(args.prefix_ch) + 1}:{int(args.latent_ch)}", len(train_loader.dataset), len(val_loader.dataset))
    if int(args.epochs) <= 0:
        val_metrics = validate_car(val_loader, encoder, decoder, cvq, car, args)
        print(f"[stage5 val 000] {format_metrics(val_metrics)} score=psnr_car")
        save_checkpoint(ckpt_path(args, "stage5", "latest"), stage="stage5", epoch=0, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
        return
    for epoch in range(1, int(args.epochs) + 1):
        encoder.eval()
        cvq.eval()
        car.train()
        meters = {k: AverageMeter() for k in ["ce", "acc_a", "acc_b", "acc_all"]}
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            y_prefix, idx = car_labels(imgs, encoder, cvq, args)
            logits_a, logits_b = car(y_prefix, idx)
            loss = car.ce_loss(logits_a, logits_b, idx)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_a = logits_a.argmax(dim=-1)
                pred_b = logits_b.argmax(dim=-1)
                pred = torch.cat([pred_a, pred_b], dim=1)
                bsz = imgs.shape[0]
                meters["ce"].update(float(loss.item()), bsz)
                meters["acc_a"].update(float((pred_a == idx[:, : car.split_a]).float().mean().item()), bsz)
                meters["acc_b"].update(float((pred_b == idx[:, car.split_a :]).float().mean().item()), bsz)
                meters["acc_all"].update(float((pred == idx).float().mean().item()), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        print_epoch("stage5", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate_car(val_loader, encoder, decoder, cvq, car, args)
            val_metrics["min_tail_gain_car"] = float(args.stage5_min_tail_gain_car)
            score = val_metrics["psnr_car"]
            print(f"[stage5 val {epoch:03d}] {format_metrics(val_metrics)} score=psnr_car")
            if score > best and stage5_best_allowed(args, val_metrics):
                best = score
                save_checkpoint(ckpt_path(args, "stage5", "best"), stage="stage5", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
            elif score > best:
                print(f"[stage5 val {epoch:03d}] skip_best_save=tail_gain_car")
        if should_save_latest(args, epoch):
            save_checkpoint(ckpt_path(args, "stage5", "latest"), stage="stage5", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
    save_checkpoint(ckpt_path(args, "stage5", "latest"), stage="stage5", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
