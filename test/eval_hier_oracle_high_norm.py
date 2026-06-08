#!/usr/bin/env python
"""Evaluate no-side-info oracle high-frequency upper bound.

The receiver input is the normalized channel observation:

    z0_rx_norm = y4_norm
    z_oracle_norm = concat(z0_rx_norm, z_gt[:, 4:16] / scale)

This isolates whether normalized true high-frequency latent channels can help
the frozen Swin decoder under the no-scale receiver setting.
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
TRAIN_DIR = os.path.join(PROJECT_ROOT, "train")
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
    build_semantic_modules,
    decode_a,
    encode_a,
    fixed_select_a,
)
from train_route_a_sc import AverageMeter  # noqa: E402
from src.cddm_mimo_ddnm import DIV2KDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oracle high-norm eval for hierarchical no-side-info Swin checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints-ar/hier_no_vae_no_sideinfo_five_stage_snr6_v1/"
            "01_swin_pretrain_no_vae_no_sideinfo_snr6/hierarchical_swin_pretrain_best.pth"
        ),
    )
    p.add_argument("--snr_db", type=float, default=6.0)
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", action="store_false", dest="cache_decoded")
    p.add_argument("--cache_workers", type=int, default=12)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260524)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--output",
        type=str,
        default="checkpoints-ar/hier_no_vae_no_sideinfo_five_stage_snr6_v1/oracle_high_norm_snr6.txt",
    )
    p.add_argument("--encoder_use_vae", action="store_true", default=False)
    p.add_argument("--lambda_kl", type=float, default=0.0)
    return p.parse_args()


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_checkpoint(path: str, encoder: torch.nn.Module, decoder: torch.nn.Module, device: torch.device) -> dict:
    ckpt_path = resolve_path(path)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "encoder_state_dict" not in ckpt or "decoder_state_dict" not in ckpt:
        raise KeyError(f"{ckpt_path} is not a hierarchical checkpoint with encoder/decoder state dicts")
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    return ckpt


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if device.type != "cuda":
        amp_enabled = False

    module_args = SimpleNamespace(encoder_use_vae=bool(args.encoder_use_vae), lambda_kl=float(args.lambda_kl))
    encoder, decoder, _cfg = build_semantic_modules(device, module_args)
    ckpt = load_checkpoint(args.checkpoint, encoder, decoder, device)
    encoder.eval()
    decoder.eval()

    a = fixed_select_a(device=device, dtype=torch.float32)
    aat_err = semiorth_error(a)
    if aat_err > 1e-7:
        raise RuntimeError(f"A A^T != I4, err={aat_err:.3e}")

    val_ds = DIV2KDataset(
        args.data_dir,
        crop_size=int(args.crop_size),
        split="valid",
        cache_decoded=bool(args.cache_decoded),
        cache_workers=int(args.cache_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.val_num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.val_num_workers) > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if int(args.val_num_workers) > 0 else None),
    )

    meters = {k: AverageMeter() for k in ("base", "oracle_high_norm", "full", "oracle_raw_high")}
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 1000)

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if int(args.max_val_batches) > 0 and bi >= int(args.max_val_batches):
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=True)
            bsz = imgs.shape[0]
            snr_b = torch.full((bsz,), float(args.snr_db), device=device, dtype=torch.float32)

            with make_autocast(device, amp_enabled, amp_dtype):
                z, _mu, _logvar = encoder.encode(imgs, sample=False)
            z = z.float()

            y4 = encode_a(z, a)
            y4_norm, _y4_raw, scale = power_normalize_awgn(y4, snr_b, generator=gen)
            scale_view = scale.float().view(-1, 1, 1, 1).clamp_min(1e-12)

            z_base = decode_a(y4_norm.float(), a)
            z_oracle = torch.cat([y4_norm.float(), z[:, 4:16].float() / scale_view], dim=1)
            z_oracle_raw_high = torch.cat([y4_norm.float(), z[:, 4:16].float()], dim=1)

            with make_autocast(device, amp_enabled, amp_dtype):
                x_base = decoder(z_base).float().clamp(0, 1)
                x_oracle = decoder(z_oracle).float().clamp(0, 1)
                x_full = decoder(z).float().clamp(0, 1)
                x_oracle_raw_high = decoder(z_oracle_raw_high).float().clamp(0, 1)

            meters["base"].update(float(psnr_per_image(x_base, imgs.float()).mean().item()), bsz)
            meters["oracle_high_norm"].update(float(psnr_per_image(x_oracle, imgs.float()).mean().item()), bsz)
            meters["full"].update(float(psnr_per_image(x_full, imgs.float()).mean().item()), bsz)
            meters["oracle_raw_high"].update(float(psnr_per_image(x_oracle_raw_high, imgs.float()).mean().item()), bsz)

    result = {
        "checkpoint": resolve_path(args.checkpoint),
        "ckpt_stage": ckpt.get("stage", ckpt.get("route", "unknown")),
        "ckpt_epoch": ckpt.get("epoch", "unknown"),
        "snr_db": float(args.snr_db),
        "valid_images": int(meters["base"].count),
        "aat_error": float(aat_err),
        "psnr_base_norm_rx": meters["base"].avg,
        "psnr_oracle_high_norm": meters["oracle_high_norm"].avg,
        "psnr_oracle_raw_high": meters["oracle_raw_high"].avg,
        "psnr_full_clean": meters["full"].avg,
        "gain_oracle_norm_minus_base": meters["oracle_high_norm"].avg - meters["base"].avg,
        "gain_raw_high_minus_base": meters["oracle_raw_high"].avg - meters["base"].avg,
    }

    lines = [
        "oracle_high_norm_eval",
        f"checkpoint={result['checkpoint']}",
        f"ckpt_stage={result['ckpt_stage']} ckpt_epoch={result['ckpt_epoch']}",
        f"snr_db={result['snr_db']:.4g} valid_images={result['valid_images']} aat_error={result['aat_error']:.3e}",
        f"psnr_base_norm_rx={result['psnr_base_norm_rx']:.4f}",
        f"psnr_oracle_high_norm={result['psnr_oracle_high_norm']:.4f}",
        f"psnr_oracle_raw_high={result['psnr_oracle_raw_high']:.4f}",
        f"psnr_full_clean={result['psnr_full_clean']:.4f}",
        f"gain_oracle_norm_minus_base={result['gain_oracle_norm_minus_base']:+.4f}",
        f"gain_raw_high_minus_base={result['gain_raw_high_minus_base']:+.4f}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")

    if args.output:
        out_path = resolve_path(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
