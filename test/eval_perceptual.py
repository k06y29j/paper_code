#!/usr/bin/env python
"""Perceptual full-link evaluation: PSNR / LPIPS / FID over SNR settings.

This script reuses ``test/eval_all.py`` for system assembly and sampling, then
adds perceptual metrics for the generated images.  LPIPS is averaged per image;
FID is computed between all real valid crops and generated crops for each
(ratio, fading, SNR, sampler) setting.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

FID_DIR = os.path.join(PROJECT_ROOT, "CDDM/open-master/open-master/FID")
if FID_DIR not in sys.path:
    sys.path.insert(0, FID_DIR)

import lpips  # noqa: E402
from inception import InceptionV3  # noqa: E402

from eval_all import (  # noqa: E402
    DIV2KDataset,
    SISOChannel,
    build_system_for_ratio,
    load_latent_refiner,
    psnr_per_image,
    resolve_unet_ckpt,
    seed_everything,
    _parse_amp,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate PSNR / LPIPS / FID for full-link reconstructions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=0)

    p.add_argument("--sc_encoder_ckpt", type=str, required=True)
    p.add_argument("--sc_decoder_ckpt", type=str, required=True)
    p.add_argument("--cc_dir", type=str, required=True)
    p.add_argument("--unet_ckpt", type=str, required=True)
    p.add_argument("--unet_model", type=str, default=None)
    p.add_argument("--no_ema", action="store_true")

    p.add_argument("--compression_ratios", type=float, nargs="+", default=[0.25])
    p.add_argument("--fadings", type=str, nargs="+", default=["awgn"], choices=["awgn", "rayleigh"])
    p.add_argument("--snrs", type=float, nargs="+", default=[0, 3, 6, 9, 12, 15])
    p.add_argument("--samplers", type=str, nargs="+", default=["none", "ddnm", "route_a"],
                   choices=["none", "ddnm", "route_a"])
    p.add_argument("--num_sample_steps", type=int, default=30)
    p.add_argument("--ddnm_t_start", type=int, default=100)
    p.add_argument("--ddnm_anchor", type=str, default="zcd", choices=["zcd", "pinv", "zero"])
    p.add_argument("--ddnm_blend", type=float, default=1.0)
    p.add_argument("--ddnm_repeat_per_step", type=int, default=3)
    p.add_argument("--ddnm_observation", type=str, default="rx", choices=["zcd", "rx"])
    p.add_argument("--ddnm_ridge", type=float, default=0.0)
    p.add_argument("--latent_std", type=float, default=0.0)
    p.add_argument("--latent_norm_stats", type=str, default="")

    p.add_argument("--latent_refiner_ckpt", type=str, default="")
    p.add_argument("--latent_refiner_hidden", type=int, default=0)
    p.add_argument("--latent_refiner_depth", type=int, default=0)
    p.add_argument("--latent_refiner_apply", type=str, default="post_channel",
                   choices=["post_channel", "post_ddnm"])
    p.add_argument("--latent_refiner_blend", type=float, default=1.0)
    p.add_argument("--rx_scale", type=float, default=1.0)
    p.add_argument("--zcd_scale", type=float, default=1.0)

    p.add_argument("--metrics", type=str, nargs="+", default=["psnr", "lpips", "fid"],
                   choices=["psnr", "lpips", "fid"])
    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--fid_dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--output_file", type=str, default="")

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    return p.parse_args()


def image_for_lpips(x: torch.Tensor) -> torch.Tensor:
    return x.float().clamp(0, 1) * 2.0 - 1.0


def image_for_fid(x: torch.Tensor) -> torch.Tensor:
    x = x.float().clamp(0, 1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[1] > 3:
        x = x[:, :3]
    return x


@torch.no_grad()
def inception_acts(model: torch.nn.Module, images: torch.Tensor) -> np.ndarray:
    pred = model(image_for_fid(images))[0]
    if pred.shape[2] != 1 or pred.shape[3] != 1:
        pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
    return pred.squeeze(-1).squeeze(-1).detach().cpu().double().numpy()


def frechet_distance(act1: np.ndarray, act2: np.ndarray, eps: float = 1e-6) -> float:
    if len(act1) < 2 or len(act2) < 2:
        return float("nan")
    mu1, mu2 = np.mean(act1, axis=0), np.mean(act2, axis=0)
    sigma1, sigma2 = np.cov(act1, rowvar=False), np.cov(act2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


@torch.no_grad()
def generate_batch(
    *,
    system,
    images: torch.Tensor,
    sampler: str,
    fading: str,
    snr_db: float,
    latent_std: float,
    latent_mean,
    latent_channel_std,
    args: argparse.Namespace,
    latent_refiner: torch.nn.Module | None,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    system.mimo = SISOChannel(snr_db=snr_db, fading=fading)
    autocast_cm = (
        torch.autocast(images.device.type, enabled=amp_enabled, dtype=amp_dtype)
        if images.device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with autocast_cm:
        z_sem = system.semantic_encoder(images)
        z_ch = system.channel_encoder(z_sem)

    z_rx, sigma_y, beta = system.mimo.forward(z_ch.float())
    if args.rx_scale != 1.0:
        z_rx = z_rx * float(args.rx_scale)

    with autocast_cm:
        z_cd = system.channel_decoder(z_rx.to(z_ch.dtype))
    if args.zcd_scale != 1.0:
        z_cd = z_cd * float(args.zcd_scale)

    z_cond = z_cd.float()
    if latent_refiner is not None and args.latent_refiner_apply == "post_channel":
        with autocast_cm:
            z_ref = latent_refiner(z_cd.to(next(latent_refiner.parameters()).dtype)).float()
        z_cond = float(args.latent_refiner_blend) * z_ref + (1.0 - float(args.latent_refiner_blend)) * z_cond

    if sampler == "none":
        z_out = z_cond
    elif sampler == "route_a":
        z_out = system.route_a_wiener_sample_normalized(
            z_anchor=z_cond,
            z_rx=z_rx.float(),
            beta=beta.float(),
            sigma_y=sigma_y,
            latent_std=latent_std,
            latent_mean=latent_mean,
            latent_channel_std=latent_channel_std,
            num_steps=args.num_sample_steps,
            t_start=args.ddnm_t_start,
            blend=args.ddnm_blend,
        )
    elif args.ddnm_observation == "rx":
        z_out = system.ddnm_sample_rx_normalized(
            z_anchor=z_cond,
            z_rx=z_rx.float(),
            beta=beta.float(),
            sigma_y=sigma_y,
            latent_std=latent_std,
            latent_mean=latent_mean,
            latent_channel_std=latent_channel_std,
            num_steps=args.num_sample_steps,
            t_start=args.ddnm_t_start,
            anchor=args.ddnm_anchor,
            blend=args.ddnm_blend,
            repeat_per_step=args.ddnm_repeat_per_step,
            ridge=args.ddnm_ridge,
        )
    else:
        z_out = system.ddnm_sample_normalized(
            z_cond=z_cond,
            beta=beta.float(),
            sigma_y=sigma_y,
            latent_std=latent_std,
            latent_mean=latent_mean,
            latent_channel_std=latent_channel_std,
            num_steps=args.num_sample_steps,
            t_start=args.ddnm_t_start,
            anchor=args.ddnm_anchor,
            blend=args.ddnm_blend,
            repeat_per_step=args.ddnm_repeat_per_step,
            ridge=args.ddnm_ridge,
        )

    if latent_refiner is not None and args.latent_refiner_apply == "post_ddnm" and sampler != "none":
        with autocast_cm:
            z_ref = latent_refiner(z_out.to(next(latent_refiner.parameters()).dtype)).float()
        z_out = float(args.latent_refiner_blend) * z_ref + (1.0 - float(args.latent_refiner_blend)) * z_out

    with autocast_cm:
        return system.semantic_decoder(z_out.to(z_ch.dtype)).float().clamp(0, 1)


def main() -> None:
    args = parse_args()
    args.unet_ckpt = resolve_unet_ckpt(args)
    seed_everything(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if device.type != "cuda":
        amp_enabled = False

    ds = DIV2KDataset(args.data_dir, crop_size=args.crop_size, split="valid")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    lpips_model = None
    if "lpips" in args.metrics:
        lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    fid_model = None
    real_acts = None
    if "fid" in args.metrics:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.fid_dims]
        fid_model = InceptionV3([block_idx]).to(device).eval()
        real_batches = []
        for bi, batch in enumerate(tqdm(loader, desc="real FID acts")):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
            real_batches.append(inception_acts(fid_model, images))
        real_acts = np.concatenate(real_batches, axis=0)

    print(f"DIV2K valid images={len(ds)}  evaluated_batches={'all' if args.max_batches <= 0 else args.max_batches}")
    print(f"metrics={args.metrics}, samplers={args.samplers}, SNRs={args.snrs}, t_start={args.ddnm_t_start}")
    results = defaultdict(dict)

    latent_refiner = None
    if args.latent_refiner_ckpt:
        latent_refiner = load_latent_refiner(
            args.latent_refiner_ckpt,
            args.latent_refiner_hidden,
            args.latent_refiner_depth,
            device,
        )

    for ratio in args.compression_ratios:
        system, unet_obj, tag = build_system_for_ratio(
            ratio=ratio,
            sc_encoder_ckpt=args.sc_encoder_ckpt,
            sc_decoder_ckpt=args.sc_decoder_ckpt,
            cc_dir=args.cc_dir,
            unet_ckpt=args.unet_ckpt,
            use_ema=not args.no_ema,
            device=device,
        )

        latent_mean = system.cfg.diffusion.latent_mean
        latent_channel_std = system.cfg.diffusion.latent_std_channels
        if args.latent_norm_stats:
            stats_obj = torch.load(args.latent_norm_stats, map_location="cpu", weights_only=False)
            latent_mean = stats_obj.get("latent_mean", stats_obj.get("mean", latent_mean))
            latent_channel_std = stats_obj.get("latent_std_channels", stats_obj.get("std", latent_channel_std))
            if hasattr(latent_mean, "tolist"):
                latent_mean = latent_mean.tolist()
            if hasattr(latent_channel_std, "tolist"):
                latent_channel_std = latent_channel_std.tolist()
            system.cfg.diffusion.latent_mean = latent_mean
            system.cfg.diffusion.latent_std_channels = latent_channel_std
            if latent_channel_std is not None:
                system.cfg.diffusion.latent_std = float(torch.as_tensor(latent_channel_std).mean().item())

        if latent_channel_std is not None:
            latent_std = float(system.cfg.diffusion.latent_std)
        elif args.latent_std > 0:
            latent_std = float(args.latent_std)
        else:
            latent_std = float(unet_obj.get("latent_std", system.cfg.diffusion.latent_std))
            if latent_std <= 0:
                raise ValueError("No valid latent_std found; pass --latent_std.")

        for sampler in args.samplers:
            for fading in args.fadings:
                for snr in args.snrs:
                    psnr_vals = []
                    lpips_sum = 0.0
                    lpips_n = 0
                    fake_acts_batches = []

                    desc = f"{tag} {sampler} {fading} {snr:g}dB"
                    for bi, batch in enumerate(tqdm(loader, desc=desc)):
                        if args.max_batches > 0 and bi >= args.max_batches:
                            break
                        images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
                        x_hat = generate_batch(
                            system=system,
                            images=images,
                            sampler=sampler,
                            fading=fading,
                            snr_db=float(snr),
                            latent_std=latent_std,
                            latent_mean=latent_mean,
                            latent_channel_std=latent_channel_std,
                            args=args,
                            latent_refiner=latent_refiner,
                            amp_enabled=amp_enabled,
                            amp_dtype=amp_dtype,
                        )

                        if "psnr" in args.metrics:
                            psnr_vals.append(psnr_per_image(x_hat, images.float()))
                        if lpips_model is not None:
                            dist = lpips_model(image_for_lpips(x_hat), image_for_lpips(images.float()))
                            lpips_sum += float(dist.view(-1).sum().item())
                            lpips_n += int(images.shape[0])
                        if fid_model is not None:
                            fake_acts_batches.append(inception_acts(fid_model, x_hat))

                    metrics = {}
                    if psnr_vals:
                        metrics["psnr"] = float(torch.cat(psnr_vals).mean().item())
                    if lpips_model is not None:
                        metrics["lpips"] = lpips_sum / max(1, lpips_n)
                    if fid_model is not None and real_acts is not None:
                        fake_acts = np.concatenate(fake_acts_batches, axis=0)
                        metrics["fid"] = frechet_distance(real_acts, fake_acts)
                    results[(ratio, sampler, fading)][float(snr)] = metrics

                    msg = f"[{tag}] sampler={sampler:<7} fading={fading:<8} SNR={snr:>5.1f} dB"
                    for name in args.metrics:
                        if name in metrics:
                            msg += f" | {name.upper()}={metrics[name]:.4f}"
                    print(msg, flush=True)

        del system
        if device.type == "cuda":
            torch.cuda.empty_cache()

    lines = []
    for metric in args.metrics:
        lines.append("\n" + "=" * 96)
        lines.append(f" {metric.upper()} summary")
        lines.append("=" * 96)
        header = f"{'ratio':>6} | {'sampler':<7} | {'fading':<8} | " + " | ".join(
            f"{snr:>7.1f}dB" for snr in args.snrs
        )
        lines.append(header)
        lines.append("-" * len(header))
        for ratio in args.compression_ratios:
            for sampler in args.samplers:
                for fading in args.fadings:
                    row = f"{ratio:>6.2f} | {sampler:<7} | {fading:<8} | "
                    vals = []
                    for snr in args.snrs:
                        value = results[(ratio, sampler, fading)][float(snr)].get(metric, float("nan"))
                        vals.append(f"{value:>10.4f}")
                    lines.append(row + " | ".join(vals))
        lines.append("=" * 96)

    text = "\n".join(lines)
    print(text)
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
