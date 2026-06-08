#!/usr/bin/env python
"""Two-frequency receiver-normalized Swin Stage 01 with predicted high latent.

This variant trains the deployed receiver path directly:

    y = z_low / scale + AWGN
    h = z_high / scale
    h_pred = P(y)

    x_recv_pred = Decoder([y, h_pred])
    x_base = Decoder([y, 0])
    x_oracle = Decoder([y, h])

The main checkpoint score is validation PSNR of x_recv_pred.  Oracle PSNR is
only a weak auxiliary constraint and diagnostic.
"""

from __future__ import annotations

import argparse
import builtins
import os
import sys
import time
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
TRAIN_DIR = os.path.abspath(os.path.dirname(__file__))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from train_codex_orthogonal_highfreq import (  # noqa: E402
    _parse_amp,
    make_autocast,
    power_normalize_awgn,
    psnr_per_image,
    seed_everything,
    semiorth_error,
)
from train_hierarchical_swin_ar_awgn12 import (  # noqa: E402
    ARReceiver,
    build_semantic_modules,
    encode_a,
    fixed_select_a,
    make_loaders,
)
from train_route_a_sc import AverageMeter, GaussianBlur, TeeStream, load_state_dict_from_ckpt  # noqa: E402


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = resolve_path(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Twofreq predicted-receiver Stage 01 @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train receiver-normalized Swin Stage 01 through h_pred=P(y)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", action="store_false", dest="cache_decoded")
    p.add_argument("--cache_workers", type=int, default=12)
    p.add_argument("--prefetch_factor", type=int, default=4)

    p.add_argument("--init_sc_encoder_ckpt", type=str, default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_encoder_div2k_c16.pth")
    p.add_argument("--init_sc_decoder_ckpt", type=str, default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_decoder_div2k_c16.pth")
    p.add_argument("--init_hier_ckpt", type=str, default="", help="Optional Stage 01 checkpoint to fine-tune from")
    p.add_argument("--load_predictor_from_hier", action="store_true", help="Also load ar/predictor state from --init_hier_ckpt")
    p.add_argument("--snr_db", type=float, default=6.0)
    p.add_argument("--baseline_psnr", type=float, default=21.0085, help="Only used for logging gain vs previous frozen-receiver baseline")

    p.add_argument("--pred_hidden", type=int, default=160)
    p.add_argument("--pred_depth", type=int, default=4)
    p.add_argument("--pred_use_scale", action="store_true", default=False, help="Give log(scale) to P; default keeps P(y) strict")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--decoder_lr", type=float, default=2e-5)
    p.add_argument("--predictor_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    p.add_argument("--lambda_recv_pred", type=float, default=1.0, help="MSE(Decoder([y,P(y)]), x) weight")
    p.add_argument("--lambda_base", type=float, default=0.2, help="MSE(Decoder([y,0]), Blur(x)) weight")
    p.add_argument("--lambda_oracle", type=float, default=0.03, help="MSE(Decoder([y,h]), x) weight")
    p.add_argument("--blur_kernel", type=int, default=7)
    p.add_argument("--blur_sigma", type=float, default=1.0)
    p.add_argument("--lambda_kl", type=float, default=0.0)
    p.add_argument("--encoder_use_vae", action="store_true", default=False)
    p.add_argument("--no_encoder_vae", action="store_false", dest="encoder_use_vae")

    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints-ar/twofreq_receiver_norm_swin_snr6_v1/01B_receiver_norm_swin_predrecv_stage01")
    p.add_argument("--log_file", type=str, default="checkpoints-ar/twofreq_receiver_norm_swin_snr6_v1/01B_receiver_norm_swin_predrecv_stage01/train.log")
    return p.parse_args()


def load_encoder_decoder_init(
    encoder: nn.Module,
    decoder: nn.Module,
    predictor: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if str(args.init_hier_ckpt):
        ckpt_path = resolve_path(str(args.init_hier_ckpt))
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
        decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
        if bool(args.load_predictor_from_hier):
            pred_state = ckpt.get("predictor_state_dict", ckpt.get("ar_state_dict"))
            if pred_state is None:
                raise KeyError(f"--load_predictor_from_hier set but no predictor/ar state in {ckpt_path}")
            predictor.load_state_dict(pred_state, strict=True)
            print(f"loaded encoder/decoder/predictor from: {ckpt_path}")
        else:
            print(f"loaded encoder/decoder from: {ckpt_path}; predictor is freshly initialized")
        print(f"  init stage={ckpt.get('stage', 'unknown')} epoch={ckpt.get('epoch', 'unknown')}")
        return

    load_state_dict_from_ckpt(encoder, args.init_sc_encoder_ckpt, "semantic_encoder")
    load_state_dict_from_ckpt(decoder, args.init_sc_decoder_ckpt, "semantic_decoder")
    print("predictor is freshly initialized")


def make_twofreq_latents(
    z: torch.Tensor,
    y4_norm: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_view = scale.float().view(-1, 1, 1, 1).clamp_min(1e-12)
    h_oracle = z[:, 4:16].float() / scale_view
    z_base = torch.zeros_like(z.float())
    z_base[:, :4] = y4_norm.float()
    z_oracle = torch.cat([y4_norm.float(), h_oracle], dim=1)
    return z_base, z_oracle, h_oracle


def run_batch(
    *,
    imgs: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    predictor: nn.Module,
    a: torch.Tensor,
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator | None,
    blur: GaussianBlur,
    train: bool,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    device = imgs.device
    bsz = imgs.shape[0]
    snr_b = torch.full((bsz,), float(args.snr_db), device=device, dtype=torch.float32)

    with make_autocast(device, amp_enabled, amp_dtype):
        z, _mu, _logvar = encoder.encode(imgs, sample=False)
        z = z.float()
        y4 = encode_a(z, a)
        y4_norm, _y4_raw, scale = power_normalize_awgn(y4, snr_b, generator=generator)
        z_base, z_oracle, h_oracle = make_twofreq_latents(z, y4_norm, scale)

        pred_dtype = next(predictor.parameters()).dtype
        z_recv_pred, pred_groups = predictor(
            y4_norm.to(dtype=pred_dtype),
            y4_raw=y4_norm.to(dtype=pred_dtype),
            y4_norm=y4_norm.to(dtype=pred_dtype),
            scale=scale,
            snr_db=snr_b,
            z_gt=None,
            teacher_prob=0.0,
        )
        h_pred = torch.cat([g.float() for g in pred_groups], dim=1)
        z_recv_pred = torch.cat([y4_norm.float(), h_pred], dim=1)
        imgs_blur = blur(imgs.float()).clamp(0, 1)

        x_recv_pred = decoder(z_recv_pred).float().clamp(0, 1)
        x_base = decoder(z_base).float().clamp(0, 1)
        x_oracle = decoder(z_oracle).float().clamp(0, 1)

        loss_recv_pred = F.mse_loss(x_recv_pred, imgs.float())
        loss_base = F.mse_loss(x_base, imgs_blur)
        loss_oracle = F.mse_loss(x_oracle, imgs.float())
        loss = (
            float(args.lambda_recv_pred) * loss_recv_pred
            + float(args.lambda_base) * loss_base
            + float(args.lambda_oracle) * loss_oracle
        )

    psnr_base = float(psnr_per_image(x_base, imgs.float()).mean().item())
    psnr_recv = float(psnr_per_image(x_recv_pred, imgs.float()).mean().item())
    psnr_oracle = float(psnr_per_image(x_oracle, imgs.float()).mean().item())
    stats = {
        "loss": float(loss.detach().item()),
        "loss_recv_pred": float(loss_recv_pred.detach().item()),
        "loss_base": float(loss_base.detach().item()),
        "loss_oracle": float(loss_oracle.detach().item()),
        "psnr_base": psnr_base,
        "psnr_base_blur": float(psnr_per_image(x_base, imgs_blur).mean().item()),
        "psnr_recv": psnr_recv,
        "psnr_oracle": psnr_oracle,
        "gain_recv": psnr_recv - psnr_base,
        "gain_oracle": psnr_oracle - psnr_base,
        "gap_oracle_recv": psnr_oracle - psnr_recv,
        "h_pred_mse": float(F.mse_loss(h_pred, h_oracle).detach().item()),
        "h_pred_rms": float(h_pred.float().square().mean().sqrt().detach().item()),
        "h_oracle_rms": float(h_oracle.float().square().mean().sqrt().detach().item()),
    }
    return (loss if train else None), stats


def save_checkpoint(
    path: str,
    encoder: nn.Module,
    decoder: nn.Module,
    predictor: nn.Module,
    a: torch.Tensor,
    cfg,
    args: argparse.Namespace,
    metrics: dict,
    epoch: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    common = {
        "route": "twofreq_receiver_norm_swin_predrecv",
        "stage": str(args.stage),
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "semantic_config": asdict(cfg.semantic),
        "a_matrix": a.detach().cpu(),
        "aat": (a @ a.t()).detach().cpu(),
        "aat_error": semiorth_error(a),
        "fixed_channel_codec": True,
        "channel_encoder": "A=[I4,0]",
        "channel_decoder": "A^T zero-fill",
        "power_norm_after_channel_encoder": True,
        "receiver_observation": "norm",
        "use_scale": bool(args.pred_use_scale),
        "snr_db": float(args.snr_db),
        "objective": "L_recv_pred + lambda_base*L_base + lambda_oracle*L_oracle",
        "score_metric": "val_psnr_recv",
        "trainable_encoder": any(p.requires_grad for p in encoder.parameters()),
        "trainable_decoder": any(p.requires_grad for p in decoder.parameters()),
        "trainable_predictor": any(p.requires_grad for p in predictor.parameters()),
    }
    payload = {
        **common,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "ar_state_dict": predictor.state_dict(),
    }
    torch.save(payload, path)
    split_dir = os.path.dirname(path)
    torch.save({**common, "part": "semantic_encoder", "state_dict": encoder.state_dict()}, os.path.join(split_dir, "sc_encoder_hier_c16.pth"))
    torch.save({**common, "part": "semantic_decoder", "state_dict": decoder.state_dict()}, os.path.join(split_dir, "sc_decoder_hier_c16.pth"))
    torch.save({**common, "part": "high_predictor", "state_dict": predictor.state_dict()}, os.path.join(split_dir, "high_predictor_hier_c16_snr6.pth"))


def main() -> None:
    args = parse_args()
    args.stage = "twofreq_receiver_norm_swin_predrecv_stage01"
    save_dir = resolve_path(args.save_dir)
    log_file = resolve_path(args.log_file)
    os.makedirs(save_dir, exist_ok=True)
    setup_log_file(log_file)
    seed_everything(int(args.seed))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if device.type != "cuda":
        amp_enabled = False
    print(
        f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}, "
        f"stage={args.stage}, train/test snr={float(args.snr_db):g}dB"
    )
    print(
        "rule: z_low=z[:,0:4] is transmitted, y=z_low/scale+AWGN, "
        "h=z[:,4:16]/scale, h_pred=P(y), score=val_psnr_recv, PSNR=mean(per-image PSNR)"
    )

    train_ds, val_ds, train_loader, val_loader = make_loaders(args, device)
    print(f"train={len(train_ds)} valid={len(val_ds)} batch={args.batch_size} crop={args.crop_size}")

    encoder, decoder, cfg = build_semantic_modules(device, args)
    predictor = ARReceiver(hidden=int(args.pred_hidden), depth=int(args.pred_depth), use_scale=bool(args.pred_use_scale)).to(device)
    load_encoder_decoder_init(encoder, decoder, predictor, args, device)

    a = fixed_select_a(device=device, dtype=torch.float32)
    aat_err = semiorth_error(a)
    if aat_err > 1e-7:
        raise RuntimeError(f"A A^T != I4, err={aat_err:.3e}")
    print(f"A_shape={tuple(a.shape)} aat_error={aat_err:.3e}")
    print(
        f"predictor=ARReceiver hidden={args.pred_hidden} depth={args.pred_depth} "
        f"use_scale={bool(args.pred_use_scale)} params={sum(p.numel() for p in predictor.parameters())}"
    )
    print(
        f"loss=L_recv_pred + {float(args.lambda_base):g}*L_base(blur_sigma={float(args.blur_sigma):g}) "
        f"+ {float(args.lambda_oracle):g}*L_oracle"
    )

    params = [
        {"params": list(encoder.parameters()), "lr": float(args.lr), "name": "encoder"},
        {"params": list(decoder.parameters()), "lr": float(args.decoder_lr), "name": "decoder"},
        {"params": list(predictor.parameters()), "lr": float(args.predictor_lr), "name": "predictor"},
    ]
    optimizer = optim.AdamW(params, weight_decay=float(args.weight_decay), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    print(
        "optimizer_groups="
        + ", ".join(f"{g['name']}:n={sum(p.numel() for p in g['params'])}:lr={g['lr']}" for g in params)
    )
    blur = GaussianBlur(3, int(args.blur_kernel), float(args.blur_sigma)).to(device)

    meter_keys = (
        "loss",
        "loss_recv_pred",
        "loss_base",
        "loss_oracle",
        "psnr_base",
        "psnr_base_blur",
        "psnr_recv",
        "psnr_oracle",
        "gain_recv",
        "gain_oracle",
        "gap_oracle_recv",
        "h_pred_mse",
        "h_pred_rms",
        "h_oracle_rms",
    )
    best = -1.0
    ckpt_prefix = "twofreq_receiver_norm_swin_predrecv_stage01"
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train(True)
        decoder.train(True)
        predictor.train(True)
        meters = {k: AverageMeter() for k in meter_keys}
        for bi, batch in enumerate(train_loader):
            if int(args.max_train_batches) > 0 and bi >= int(args.max_train_batches):
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, stats = run_batch(
                imgs=imgs,
                encoder=encoder,
                decoder=decoder,
                predictor=predictor,
                a=a,
                args=args,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                generator=None,
                blur=blur,
                train=True,
            )
            assert loss is not None
            scaler.scale(loss).backward()
            if float(args.clip_grad_norm) > 0:
                scaler.unscale_(optimizer)
                trainable_params = [p for group in optimizer.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.clip_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            for k in meters:
                meters[k].update(stats[k], imgs.shape[0])

        metrics = {f"train_{k}": v.avg for k, v in meters.items()}
        do_eval = epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs)
        if do_eval:
            encoder.eval()
            decoder.eval()
            predictor.eval()
            val_meters = {k: AverageMeter() for k in meter_keys}
            gen = torch.Generator(device=device)
            gen.manual_seed(int(args.seed) + 1000)
            with torch.no_grad():
                for bi, batch in enumerate(val_loader):
                    if int(args.max_val_batches) > 0 and bi >= int(args.max_val_batches):
                        break
                    imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
                    imgs = imgs.to(device, non_blocking=True)
                    _loss, stats = run_batch(
                        imgs=imgs,
                        encoder=encoder,
                        decoder=decoder,
                        predictor=predictor,
                        a=a,
                        args=args,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        generator=gen,
                        blur=blur,
                        train=False,
                    )
                    for k in val_meters:
                        val_meters[k].update(stats[k], imgs.shape[0])
            metrics.update({f"val_{k}": v.avg for k, v in val_meters.items()})
            score = metrics["val_psnr_recv"]
            is_best = score > best
            if is_best:
                best = score
                save_checkpoint(os.path.join(save_dir, f"{ckpt_prefix}_best.pth"), encoder, decoder, predictor, a, cfg, args, metrics, epoch)
            save_checkpoint(os.path.join(save_dir, f"{ckpt_prefix}_latest.pth"), encoder, decoder, predictor, a, cfg, args, metrics, epoch)
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"recv_loss={meters['loss_recv_pred'].avg:.6f} | "
                f"base={metrics['val_psnr_base']:.4f} recv={metrics['val_psnr_recv']:.4f} "
                f"oracle={metrics['val_psnr_oracle']:.4f} gain_recv={metrics['val_gain_recv']:+.4f} "
                f"gap_oracle_recv={metrics['val_gap_oracle_recv']:+.4f} base_blur={metrics['val_psnr_base_blur']:.4f} "
                f"h_mse={metrics['val_h_pred_mse']:.4f} h_pred_rms={metrics['val_h_pred_rms']:.4f} "
                f"h_oracle_rms={metrics['val_h_oracle_rms']:.4f} score(val_recv)={score:.4f} "
                f"gain_prev_baseline={metrics['val_psnr_recv'] - float(args.baseline_psnr):+.4f} "
                f"aat_err={semiorth_error(a):.2e} {'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"base={meters['psnr_base'].avg:.4f} recv={meters['psnr_recv'].avg:.4f} "
                f"oracle={meters['psnr_oracle'].avg:.4f} gain_recv={meters['gain_recv'].avg:+.4f} "
                f"h_mse={meters['h_pred_mse'].avg:.4f}"
            )

    print(f"best_val_psnr_recv={best:.4f}")


if __name__ == "__main__":
    main()
