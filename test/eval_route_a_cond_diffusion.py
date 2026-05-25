#!/usr/bin/env python
"""Evaluate/sweep Route-A conditional diffusion checkpoints.

The current conditional diffusion model predicts the null-space latent
conditioned on the received low-space latent.  This script isolates sampler
choices from training choices: ``t_start``, DDIM steps, initialization, and
latent ensembling.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import DIV2KDataset  # noqa: E402
from src.cddm_mimo_ddnm.modules.ddnm import UNetDenoiser  # noqa: E402
from src.cddm_mimo_ddnm.modules.siso_channel import SISOChannel  # noqa: E402


def _load_module(rel_path: str, name: str):
    path = os.path.join(PROJECT_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_all = _load_module("test/eval_all.py", "eval_all_helpers")
cond_train = _load_module("train/train_route_a_cond_diffusion.py", "cond_diff_helpers")

build_system_for_ratio = eval_all.build_system_for_ratio
load_latent_refiner = eval_all.load_latent_refiner
psnr_per_image = eval_all.psnr_per_image
seed_everything = eval_all.seed_everything
_parse_amp = eval_all._parse_amp
make_batch_latents = cond_train.make_batch_latents
make_schedule = cond_train.make_schedule
null_project = cond_train.null_project
sample_null = cond_train.sample_null


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Route-A conditional diffusion sampler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--cache_workers", type=int, default=16)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=0)

    p.add_argument("--sc_encoder_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_encoder_div2k_c16.pth")
    p.add_argument("--sc_decoder_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_decoder_div2k_c16.pth")
    p.add_argument("--cc_dir", type=str, default="checkpoints-val-v2/route_a/cc_dct_c4")
    p.add_argument("--unet_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/unet_un/unet_un_div2k_c16_best.pth")
    p.add_argument("--cond_diff_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/cond_diff_null_h96_t300_awgn12/route_a_cond_diff_null_div2k_c16_awgn12_best.pth")
    p.add_argument("--refiner_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/null_refiner_h128d8_img10_awgn12/route_a_null_refiner_div2k_c16_awgn12_best.pth")
    p.add_argument("--refiner_hidden", type=int, default=0)
    p.add_argument("--refiner_depth", type=int, default=0)
    p.add_argument("--compression_ratio", type=float, default=0.25)
    p.add_argument("--snr_db", type=float, default=12.0)
    p.add_argument("--fading", type=str, default="awgn", choices=["awgn", "rayleigh"])

    p.add_argument("--sampler", type=str, default="freq_guided", choices=["vanilla", "freq_guided"])
    p.add_argument("--t_starts", type=int, nargs="+", default=[300])
    p.add_argument("--steps", type=int, nargs="+", default=[30])
    p.add_argument("--sample_inits", type=str, nargs="+", default=["zero"], choices=["zero", "noise"])
    p.add_argument("--ensembles", type=int, nargs="+", default=[1])
    p.add_argument("--z0_clip", type=float, default=-1.0,
                   help="Use checkpoint z0_clip when <0.")
    p.add_argument("--latent_lowpass_kernel", type=int, default=5)
    p.add_argument("--latent_lowpass_sigma", type=float, default=1.0)
    p.add_argument("--diff_high_scale", type=float, default=1.0)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--csv", type=str, default="log-v2/route_a/cond_diff_sampler_sweep_awgn12.csv")
    return p.parse_args()


def load_cond_model(path: str, device: torch.device):
    ckpt_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}
    hidden = int(ckpt_args.get("hidden", 96))
    model = UNetDenoiser(channels=16, hidden_dim=hidden, use_cond=True).to(device)
    ema_state = ckpt.get("ema_state_dict", {})
    if isinstance(ema_state, dict) and "shadow" in ema_state:
        model.load_state_dict(ema_state["shadow"], strict=True)
        weight_src = "ema"
    else:
        model.load_state_dict(ckpt["state_dict"], strict=True)
        weight_src = "state_dict"
    model.eval()
    schedule = str(ckpt_args.get("noise_schedule", "cosine"))
    num_train_steps = int(ckpt_args.get("num_train_steps", 1000))
    z0_clip = float(ckpt_args.get("z0_clip", 5.0))
    alpha_bars = make_schedule(num_train_steps, schedule, device)
    stats = {k: v.to(device) for k, v in ckpt["stats"].items()}
    a_matrix = torch.as_tensor(ckpt["a_matrix"], dtype=torch.float32, device=device)
    print(
        f"[cond_diff] {ckpt_path} hidden={hidden} weights={weight_src} "
        f"schedule={schedule} T={num_train_steps} epoch={ckpt.get('epoch')}"
    )
    if "metrics" in ckpt:
        print(f"  metrics={ckpt['metrics']}")
    return model, stats, a_matrix, alpha_bars, z0_clip


def gaussian_blur_latent(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    k = int(kernel_size)
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) * 0.5
    kernel = torch.exp(-(coords * coords) / (2.0 * float(sigma) * float(sigma)))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    c = x.shape[1]
    pad = k // 2
    kx = kernel.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    ky = kernel.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    y = torch.nn.functional.pad(x, (pad, pad, 0, 0), mode="reflect")
    y = torch.nn.functional.conv2d(y, kx, groups=c)
    y = torch.nn.functional.pad(y, (0, 0, pad, pad), mode="reflect")
    return torch.nn.functional.conv2d(y, ky, groups=c)


@torch.no_grad()
def sample_null_frequency_guided(
    model: torch.nn.Module,
    z_low: torch.Tensor,
    z_anchor: torch.Tensor,
    stats: dict,
    alpha_bars: torch.Tensor,
    a_matrix: torch.Tensor,
    *,
    num_steps: int,
    t_start: int,
    z0_clip: float,
    lowpass_kernel: int,
    lowpass_sigma: float,
    diff_high_scale: float,
) -> torch.Tensor:
    """Null-space DDIM where Refiner provides structure and diffusion provides highpass texture."""
    device = z_low.device
    low_norm = cond_train.norm(z_low, stats["low_mean"], stats["low_std"])
    anchor_null = null_project(z_anchor.float(), a_matrix)
    anchor_low = gaussian_blur_latent(anchor_null, lowpass_kernel, lowpass_sigma)
    anchor_norm = cond_train.norm(anchor_null, stats["null_mean"], stats["null_std"])

    n_total = int(alpha_bars.shape[0])
    t_start = max(0, min(int(t_start), n_total - 1))
    step_indices = torch.linspace(t_start, 0, int(num_steps), device=device).long()
    alpha_start = alpha_bars[t_start].to(device=device, dtype=z_low.dtype)
    z = torch.sqrt(alpha_start) * anchor_norm + torch.sqrt(1.0 - alpha_start) * torch.randn_like(anchor_norm)

    for i, idx in enumerate(step_indices):
        alpha_bar = alpha_bars[idx].to(device=device, dtype=z.dtype)
        alpha_prev = (
            alpha_bars[step_indices[i + 1]].to(device=device, dtype=z.dtype)
            if i + 1 < len(step_indices)
            else torch.tensor(1.0, device=device, dtype=z.dtype)
        )
        t_cont = torch.full((z.shape[0],), float(idx.item()) / float(max(1, n_total - 1)), device=device, dtype=z.dtype)
        eps = model(z, t_cont, cond=low_norm)
        z0_norm = (z - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar + 1e-8)
        if z0_clip > 0:
            z0_norm = z0_norm.clamp(-float(z0_clip), float(z0_clip))
        pred_null = null_project(cond_train.denorm(z0_norm, stats["null_mean"], stats["null_std"]), a_matrix)
        pred_low = gaussian_blur_latent(pred_null, lowpass_kernel, lowpass_sigma)
        pred_high = pred_null - pred_low
        guided_null = null_project(anchor_low + float(diff_high_scale) * pred_high, a_matrix)
        guided_norm = cond_train.norm(guided_null, stats["null_mean"], stats["null_std"])
        z = torch.sqrt(alpha_prev) * guided_norm + torch.sqrt(1.0 - alpha_prev) * eps

    final_null = null_project(cond_train.denorm(z, stats["null_mean"], stats["null_std"]), a_matrix)
    final_low = gaussian_blur_latent(final_null, lowpass_kernel, lowpass_sigma)
    final_high = final_null - final_low
    return null_project(anchor_low + float(diff_high_scale) * final_high, a_matrix)


@torch.no_grad()
def evaluate_combo(
    *,
    system,
    model,
    loader,
    stats,
    a_matrix,
    alpha_bars,
    sampler: str,
    refiner,
    t_start: int,
    steps: int,
    sample_init: str,
    ensemble: int,
    z0_clip: float,
    lowpass_kernel: int,
    lowpass_sigma: float,
    diff_high_scale: float,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
) -> dict:
    base_vals: list[torch.Tensor] = []
    refiner_vals: list[torch.Tensor] = []
    diff_vals: list[torch.Tensor] = []
    decoder_dtype = next(system.semantic_decoder.parameters()).dtype
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        _z_sem, z_low = make_batch_latents(system, imgs, amp_enabled, amp_dtype)
        z_ref = None
        if refiner is not None:
            z_ref = refiner(z_low.to(next(refiner.parameters()).dtype)).float()
        z_accum = torch.zeros_like(z_low.float())
        for _ in range(max(1, int(ensemble))):
            if sampler == "freq_guided":
                if z_ref is None:
                    raise ValueError("freq_guided sampler requires --refiner_ckpt")
                z_null = sample_null_frequency_guided(
                    model,
                    z_low.float(),
                    z_ref.float(),
                    stats,
                    alpha_bars,
                    a_matrix,
                    num_steps=int(steps),
                    t_start=int(t_start),
                    z0_clip=float(z0_clip),
                    lowpass_kernel=int(lowpass_kernel),
                    lowpass_sigma=float(lowpass_sigma),
                    diff_high_scale=float(diff_high_scale),
                )
            else:
                z_null = sample_null(
                    model,
                    z_low.float(),
                    stats,
                    alpha_bars,
                    a_matrix,
                    num_steps=int(steps),
                    t_start=int(t_start),
                    sample_init=sample_init,
                    z0_clip=float(z0_clip),
                )
            z_accum.add_(z_low.float() + z_null)
        z_hat = z_accum / float(max(1, int(ensemble)))
        autocast_cm = (
            torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with autocast_cm:
            x_base = system.semantic_decoder(z_low.to(decoder_dtype))
            x_ref = system.semantic_decoder(z_ref.to(decoder_dtype)) if z_ref is not None else None
            x_hat = system.semantic_decoder(z_hat.to(decoder_dtype))
        base_vals.append(psnr_per_image(x_base.float().clamp(0, 1), imgs.float()))
        if x_ref is not None:
            refiner_vals.append(psnr_per_image(x_ref.float().clamp(0, 1), imgs.float()))
        diff_vals.append(psnr_per_image(x_hat.float().clamp(0, 1), imgs.float()))
    base = torch.cat(base_vals).mean().item() if base_vals else float("nan")
    refiner_psnr = torch.cat(refiner_vals).mean().item() if refiner_vals else float("nan")
    diff = torch.cat(diff_vals).mean().item() if diff_vals else float("nan")
    return {
        "sampler": sampler,
        "t_start": int(t_start),
        "steps": int(steps),
        "sample_init": sample_init,
        "ensemble": int(ensemble),
        "base_psnr": float(base),
        "refiner_psnr": float(refiner_psnr),
        "diff_psnr": float(diff),
        "gain": float(diff - base),
        "gain_vs_refiner": float(diff - refiner_psnr),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

    ds = DIV2KDataset(
        args.data_dir,
        crop_size=args.crop_size,
        split="valid",
        cache_decoded=bool(args.cache_decoded),
        cache_workers=int(args.cache_workers),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if args.num_workers > 0 else None),
    )

    system, _unet_obj, tag = build_system_for_ratio(
        ratio=args.compression_ratio,
        sc_encoder_ckpt=args.sc_encoder_ckpt,
        sc_decoder_ckpt=args.sc_decoder_ckpt,
        cc_dir=args.cc_dir,
        unet_ckpt=args.unet_ckpt,
        use_ema=True,
        device=device,
    )
    system.eval()
    system.mimo = SISOChannel(snr_db=float(args.snr_db), fading=args.fading)
    for p in system.parameters():
        p.requires_grad_(False)
    model, stats, a_matrix, alpha_bars, ckpt_z0_clip = load_cond_model(args.cond_diff_ckpt, device)
    refiner = None
    if args.refiner_ckpt:
        refiner = load_latent_refiner(args.refiner_ckpt, args.refiner_hidden, args.refiner_depth, device)
    z0_clip = ckpt_z0_clip if float(args.z0_clip) < 0 else float(args.z0_clip)

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(PROJECT_ROOT, args.csv)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = []
    for sample_init in args.sample_inits:
        for ensemble in args.ensembles:
            for steps in args.steps:
                for t_start in args.t_starts:
                    row = evaluate_combo(
                        system=system,
                        model=model,
                        loader=loader,
                        stats=stats,
                        a_matrix=a_matrix,
                        alpha_bars=alpha_bars,
                        sampler=args.sampler,
                        refiner=refiner,
                        t_start=t_start,
                        steps=steps,
                        sample_init=sample_init,
                        ensemble=ensemble,
                        z0_clip=z0_clip,
                        lowpass_kernel=int(args.latent_lowpass_kernel),
                        lowpass_sigma=float(args.latent_lowpass_sigma),
                        diff_high_scale=float(args.diff_high_scale),
                        device=device,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        max_batches=int(args.max_batches),
                    )
                    rows.append(row)
                    print(
                        f"sampler={args.sampler} init={sample_init:>5} E={ensemble} steps={steps:3d} t={t_start:3d} "
                        f"base={row['base_psnr']:.4f} ref={row['refiner_psnr']:.4f} "
                        f"diff={row['diff_psnr']:.4f} gain={row['gain']:+.4f} "
                        f"vs_ref={row['gain_vs_refiner']:+.4f}"
                    )

    rows = sorted(rows, key=lambda r: r["diff_psnr"], reverse=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sampler", "sample_init", "ensemble", "steps", "t_start",
                "base_psnr", "refiner_psnr", "diff_psnr", "gain", "gain_vs_refiner",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTop results written to {csv_path}")
    for row in rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
