#!/usr/bin/env python
"""Route-A value/null-space diagnostics.

For each reconstruction method this reports three complementary views:

1. full: decoder(z_final) vs x
2. low_blur: decoder(A^T A z_final) vs GaussianBlur(x)
3. cleanlow_null: decoder(A^T A z_gt + (I - A^T A) z_final) vs x

The third view isolates null-space hallucination quality by removing channel
noise from the value-space component.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
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


class GaussianBlur(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, sigma: float) -> None:
        super().__init__()
        half = kernel_size // 2
        coords = torch.arange(kernel_size, dtype=torch.float32) - half
        kernel_1d = torch.exp(-(coords ** 2) / (2.0 * float(sigma) ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        weight = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.groups = int(channels)
        self.padding = int(half)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        return F.conv2d(x, weight, padding=self.padding, groups=self.groups)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose Route-A low/value-space denoising and null-space alignment.",
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

    p.add_argument("--compression_ratio", type=float, default=0.25)
    p.add_argument("--fading", type=str, default="awgn", choices=["awgn", "rayleigh"])
    p.add_argument("--snr", type=float, default=12.0)
    p.add_argument("--methods", type=str, nargs="+",
                   default=["none", "route_a", "refiner", "refiner_route_a"],
                   choices=["none", "route_a", "refiner", "refiner_route_a"])

    p.add_argument("--num_sample_steps", type=int, default=30)
    p.add_argument("--route_a_t_start", type=int, default=150)
    p.add_argument("--route_a_blend", type=float, default=1.0)
    p.add_argument("--route_a_keep_null", type=float, default=1.0)
    p.add_argument("--route_a_final_wiener", type=float, default=1.0)

    p.add_argument("--latent_std", type=float, default=0.0)
    p.add_argument("--latent_norm_stats", type=str, default="")
    p.add_argument("--latent_refiner_ckpt", type=str, default="")
    p.add_argument("--latent_refiner_hidden", type=int, default=0)
    p.add_argument("--latent_refiner_depth", type=int, default=0)
    p.add_argument("--latent_refiner_blend", type=float, default=0.75)

    p.add_argument("--blur_kernel", type=int, default=15)
    p.add_argument("--blur_sigma", type=float, default=3.0)
    p.add_argument("--metrics", type=str, nargs="+", default=["psnr", "lpips", "fid"],
                   choices=["psnr", "lpips", "fid"])
    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--fid_dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--output_file", type=str, default="")

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=20260521)
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


def project_low(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    aa = a.to(device=z.device, dtype=z.dtype)
    az = torch.einsum("oc,bchw->bohw", aa, z)
    return torch.einsum("oc,bohw->bchw", aa, az)


def null_part(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return z - project_low(z, a)


@torch.no_grad()
def decode(system, z: torch.Tensor, dtype: torch.dtype, amp_enabled: bool) -> torch.Tensor:
    autocast_cm = (
        torch.autocast(z.device.type, enabled=amp_enabled, dtype=dtype)
        if z.device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with autocast_cm:
        out = system.semantic_decoder(z.to(next(system.semantic_decoder.parameters()).dtype))
    return out.float().clamp(0, 1)


@torch.no_grad()
def method_latent(
    *,
    method: str,
    system,
    z_cd: torch.Tensor,
    z_rx: torch.Tensor,
    beta: torch.Tensor,
    sigma_y: float,
    latent_std: float,
    latent_mean,
    latent_channel_std,
    refiner: torch.nn.Module | None,
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    if method in ("refiner", "refiner_route_a"):
        if refiner is None:
            raise ValueError(f"method={method} requires --latent_refiner_ckpt")
        autocast_cm = (
            torch.autocast(z_cd.device.type, enabled=amp_enabled, dtype=amp_dtype)
            if z_cd.device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with autocast_cm:
            z_ref = refiner(z_cd.to(next(refiner.parameters()).dtype)).float()
        z_anchor = float(args.latent_refiner_blend) * z_ref + (1.0 - float(args.latent_refiner_blend)) * z_cd
    else:
        z_anchor = z_cd

    if method in ("route_a", "refiner_route_a"):
        return system.route_a_wiener_sample_normalized(
            z_anchor=z_anchor,
            z_rx=z_rx.float(),
            beta=beta.float(),
            sigma_y=sigma_y,
            latent_std=latent_std,
            latent_mean=latent_mean,
            latent_channel_std=latent_channel_std,
            num_steps=args.num_sample_steps,
            t_start=args.route_a_t_start,
            blend=args.route_a_blend,
            keep_null_space=args.route_a_keep_null,
            final_wiener=args.route_a_final_wiener,
        )
    return z_anchor


def _new_metric_state(args: argparse.Namespace) -> dict:
    return {
        "psnr": [],
        "lpips_sum": 0.0,
        "lpips_n": 0,
        "fid_batches": [],
    }


def _accumulate_metrics(
    state: dict,
    pred: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    lpips_model,
    fid_model,
) -> None:
    if "psnr" in args.metrics:
        state["psnr"].append(psnr_per_image(pred.float().clamp(0, 1), target.float().clamp(0, 1)))
    if lpips_model is not None:
        dist = lpips_model(image_for_lpips(pred), image_for_lpips(target))
        state["lpips_sum"] += float(dist.view(-1).sum().item())
        state["lpips_n"] += int(pred.shape[0])
    if fid_model is not None:
        state["fid_batches"].append(inception_acts(fid_model, pred))


def _finalize_state(state: dict, args: argparse.Namespace, real_acts: np.ndarray | None) -> dict:
    out = {}
    if state["psnr"]:
        out["psnr"] = float(torch.cat(state["psnr"]).mean().item())
    if "lpips" in args.metrics:
        out["lpips"] = state["lpips_sum"] / max(1, state["lpips_n"])
    if "fid" in args.metrics and real_acts is not None:
        fake = np.concatenate(state["fid_batches"], axis=0)
        out["fid"] = frechet_distance(real_acts, fake)
    return out


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

    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval() if "lpips" in args.metrics else None
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

    refiner = None
    if args.latent_refiner_ckpt:
        refiner = load_latent_refiner(
            args.latent_refiner_ckpt,
            args.latent_refiner_hidden,
            args.latent_refiner_depth,
            device,
        )

    system, unet_obj, tag = build_system_for_ratio(
        ratio=args.compression_ratio,
        sc_encoder_ckpt=args.sc_encoder_ckpt,
        sc_decoder_ckpt=args.sc_decoder_ckpt,
        cc_dir=args.cc_dir,
        unet_ckpt=args.unet_ckpt,
        use_ema=not args.no_ema,
        device=device,
    )
    system.eval()
    system.mimo = SISOChannel(snr_db=float(args.snr), fading=args.fading)

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

    a = system._channel_encoder_matrix().detach().float()
    blur = GaussianBlur(3, args.blur_kernel, args.blur_sigma).to(device)

    states = {
        method: {
            "full": _new_metric_state(args),
            "low_blur": _new_metric_state(argparse.Namespace(metrics=["psnr"])),
            "cleanlow_null": _new_metric_state(args),
        }
        for method in args.methods
    }

    autocast_dtype = amp_dtype
    for bi, batch in enumerate(tqdm(loader, desc=f"{tag} diagnostics {args.snr:g}dB")):
        if args.max_batches > 0 and bi >= args.max_batches:
            break
        images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        x_blur = blur(images.float()).clamp(0, 1)

        autocast_cm = (
            torch.autocast(device.type, enabled=amp_enabled, dtype=autocast_dtype)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with autocast_cm:
            z_gt = system.semantic_encoder(images)
            z_ch = system.channel_encoder(z_gt)
        z_rx, sigma_y, beta = system.mimo.forward(z_ch.float())
        with autocast_cm:
            z_cd = system.channel_decoder(z_rx.to(z_ch.dtype))
        z_gt_f = z_gt.float()
        z_cd_f = z_cd.float()
        z_low_clean = project_low(z_gt_f, a)

        for method in args.methods:
            z_final = method_latent(
                method=method,
                system=system,
                z_cd=z_cd_f,
                z_rx=z_rx.float(),
                beta=beta.float(),
                sigma_y=sigma_y,
                latent_std=latent_std,
                latent_mean=latent_mean,
                latent_channel_std=latent_channel_std,
                refiner=refiner,
                args=args,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            z_low_final = project_low(z_final.float(), a)
            z_cleanlow_null = z_low_clean + null_part(z_final.float(), a)

            x_full = decode(system, z_final, autocast_dtype, amp_enabled)
            x_low = decode(system, z_low_final, autocast_dtype, amp_enabled)
            x_cleanlow_null = decode(system, z_cleanlow_null, autocast_dtype, amp_enabled)

            _accumulate_metrics(states[method]["full"], x_full, images.float(), args, lpips_model, fid_model)
            _accumulate_metrics(
                states[method]["low_blur"],
                x_low,
                x_blur,
                argparse.Namespace(metrics=["psnr"]),
                None,
                None,
            )
            _accumulate_metrics(
                states[method]["cleanlow_null"],
                x_cleanlow_null,
                images.float(),
                args,
                lpips_model,
                fid_model,
            )

    results = {}
    for method, views in states.items():
        results[method] = {
            "full": _finalize_state(views["full"], args, real_acts),
            "low_blur": _finalize_state(views["low_blur"], argparse.Namespace(metrics=["psnr"]), None),
            "cleanlow_null": _finalize_state(views["cleanlow_null"], args, real_acts),
        }

    lines = []
    lines.append("=" * 116)
    lines.append(
        f"Route-A space diagnostics: tag={tag}, fading={args.fading}, snr={args.snr:g}dB, "
        f"steps={args.num_sample_steps}, t_start={args.route_a_t_start}, "
        f"keep_null={args.route_a_keep_null:g}, final_wiener={args.route_a_final_wiener:g}"
    )
    lines.append(
        f"blur: kernel={args.blur_kernel}, sigma={args.blur_sigma:g}; "
        f"refiner_blend={args.latent_refiner_blend:g}; images={len(ds) if args.max_batches <= 0 else 'subset'}"
    )
    lines.append("=" * 116)
    header = (
        f"{'method':<15} | {'full_psnr':>9} | {'full_lpips':>10} | {'full_fid':>9} | "
        f"{'low_vs_blur_psnr':>16} | {'cleanlow_psnr':>13} | {'cleanlow_lpips':>15} | {'cleanlow_fid':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for method in args.methods:
        full = results[method]["full"]
        low = results[method]["low_blur"]
        clean = results[method]["cleanlow_null"]
        lines.append(
            f"{method:<15} | "
            f"{full.get('psnr', float('nan')):9.4f} | "
            f"{full.get('lpips', float('nan')):10.4f} | "
            f"{full.get('fid', float('nan')):9.4f} | "
            f"{low.get('psnr', float('nan')):16.4f} | "
            f"{clean.get('psnr', float('nan')):13.4f} | "
            f"{clean.get('lpips', float('nan')):15.4f} | "
            f"{clean.get('fid', float('nan')):12.4f}"
        )
    lines.append("=" * 116)
    text = "\n".join(lines)
    print(text)
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
