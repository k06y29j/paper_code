#!/usr/bin/env python
"""Train Route-A residual conditional diffusion after a fixed low-frequency anchor.

The previous conditional diffusion sampled the full null-space latent.  That is
too unconstrained for PSNR because the sampled high-frequency field is not
spatially aligned with the image.  This script keeps a fixed anchor and trains
diffusion only on the remaining null-space residual:

    z_anchor = z_low                 (lightweight mode)
           or scaled-noise z_low     (lower-channel-noise diagnostic mode)
           or Wiener(A-space rx)     (noise-aware lightweight mode)
           or Refiner(z_low)         (strong-anchor mode)
    r_gt     = P_null(z_sem - z_anchor)
    z_hat    = z_anchor + r_sample

Thus A-space low-frequency information stays fixed while diffusion learns only
the high-frequency/null-space correction not already present in the anchor.
"""

from __future__ import annotations

import argparse
import builtins
import copy
import importlib.util
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import DIV2KDataset  # noqa: E402
from src.cddm_mimo_ddnm.modules.ddnm import UNetDenoiser  # noqa: E402
from src.cddm_mimo_ddnm.modules.siso_channel import SISOChannel  # noqa: E402


def _load_eval_helpers():
    path = os.path.join(PROJECT_ROOT, "test", "eval_all.py")
    spec = importlib.util.spec_from_file_location("eval_all_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_helpers = _load_eval_helpers()
build_system_for_ratio = eval_helpers.build_system_for_ratio
load_latent_refiner = eval_helpers.load_latent_refiner
psnr_per_image = eval_helpers.psnr_per_image
seed_everything = eval_helpers.seed_everything
_parse_amp = eval_helpers._parse_amp


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()}}

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> dict:
        backup = copy.deepcopy(model.state_dict())
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd:
                sd[k].copy_(v.to(device=sd[k].device, dtype=sd[k].dtype))
        return backup


class ResidualLatentDataset(Dataset):
    def __init__(self, z_ref: torch.Tensor, residual: torch.Tensor, imgs: torch.Tensor) -> None:
        self.z_ref = z_ref
        self.residual = residual
        self.imgs = imgs

    def __len__(self) -> int:
        return int(self.z_ref.shape[0])

    def __getitem__(self, idx):
        return self.z_ref[idx], self.residual[idx], self.imgs[idx]


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Route-A residual conditional diffusion session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train residual conditional diffusion after Route-A null refiner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", action="store_false", dest="cache_decoded")
    p.add_argument("--cache_workers", type=int, default=16)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--precache_repeats", type=int, default=4)

    p.add_argument("--sc_encoder_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_encoder_div2k_c16.pth")
    p.add_argument("--sc_decoder_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_decoder_div2k_c16.pth")
    p.add_argument("--cc_dir", type=str, default="checkpoints-val-v2/route_a/cc_dct_c4")
    p.add_argument("--unet_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/unet_un/unet_un_div2k_c16_best.pth")
    p.add_argument("--refiner_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/null_refiner_h128d8_img10_awgn12/route_a_null_refiner_div2k_c16_awgn12_best.pth")
    p.add_argument("--refiner_hidden", type=int, default=0)
    p.add_argument("--refiner_depth", type=int, default=0)
    p.add_argument(
        "--anchor_mode",
        type=str,
        default="refiner",
        choices=["refiner", "low", "noise_scaled_low", "wiener_low"],
        help=(
            "refiner: z_anchor=Refiner(z_low); low: z_anchor=z_low; "
            "noise_scaled_low: z_ch + scale * (z_rx - z_ch) diagnostic; "
            "wiener_low: AWGN-aware Wiener denoise in 4-channel A-space, then A^T."
        ),
    )
    p.add_argument("--low_noise_scale", type=float, default=1.0,
                   help="For anchor_mode=noise_scaled_low, use z_ch + scale * (z_rx - z_ch).")
    p.add_argument("--wiener_low_strength", type=float, default=1.0,
                   help="Blend strength for anchor_mode=wiener_low; 1 uses full Wiener estimate, 0 raw rx.")
    p.add_argument("--compression_ratio", type=float, default=0.25)
    p.add_argument("--snr_db", type=float, default=12.0)
    p.add_argument("--fading", type=str, default="awgn", choices=["awgn", "rayleigh"])

    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_train_steps", type=int, default=1000)
    p.add_argument("--noise_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--min_snr_gamma", type=float, default=5.0)
    p.add_argument("--lambda_image", type=float, default=5.0)
    p.add_argument("--image_loss_alpha_power", type=float, default=1.0)
    p.add_argument("--z0_clip", type=float, default=5.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    p.add_argument("--sample_steps", type=int, default=30)
    p.add_argument("--sample_t_start", type=int, default=200)
    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_file", type=str, default="log-v2/route_a/res_cond_diff_awgn12.txt")
    p.add_argument("--save_dir", type=str, default="checkpoints-val-v2/route_a/res_cond_diff_awgn12")
    p.add_argument("--resume_ckpt", type=str, default="")
    return p.parse_args()


def make_schedule(num_steps: int, schedule: str, device: torch.device) -> torch.Tensor:
    if schedule == "linear":
        betas = torch.linspace(1e-4, 2e-2, num_steps, device=device)
        return torch.cumprod(1.0 - betas, dim=0)
    if schedule == "cosine":
        steps = torch.arange(num_steps + 1, device=device, dtype=torch.float32)
        s = 0.008
        f = torch.cos(((steps / num_steps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bars = (f / f[0]).clamp_min(1e-8)
        betas = (1.0 - alpha_bars[1:] / alpha_bars[:-1]).clamp(1e-8, 0.999)
        return torch.cumprod(1.0 - betas, dim=0)
    raise ValueError(f"unknown noise_schedule={schedule!r}")


def project_low(z: torch.Tensor, a_matrix: torch.Tensor) -> torch.Tensor:
    a = a_matrix.to(device=z.device, dtype=z.dtype)
    az = torch.einsum("oc,bchw->bohw", a, z)
    return torch.einsum("oc,bohw->bchw", a, az)


def null_project(z: torch.Tensor, a_matrix: torch.Tensor) -> torch.Tensor:
    return z - project_low(z, a_matrix)


def awgn_wiener_low_anchor(
    z_rx: torch.Tensor,
    a_matrix: torch.Tensor,
    snr_db: float,
    strength: float = 1.0,
) -> torch.Tensor:
    """Denoise the received 4-channel A-space feature before A^T expansion.

    ``SISOChannel`` adds AWGN after per-sample complex-symbol power
    normalization and then scales back.  From the received feature power we
    estimate the clean power, convert it to real-channel noise variance, and do
    a spatial Wiener shrinkage per sample/channel:

        mu_y + var_signal / (var_signal + var_noise) * (y - mu_y)

    This keeps the low-frequency value space fixed but avoids conditioning the
    high-frequency model on the dirtiest version of the 4-channel observation.
    """
    if z_rx.shape[1] % 2 != 0:
        raise ValueError("wiener_low expects an even number of A-space channels")
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    z_use = z_rx.float()
    z_complex = torch.complex(z_use[:, 0::2], z_use[:, 1::2])
    dims_complex = tuple(range(1, z_complex.ndim))
    rx_power = (z_complex.real.square() + z_complex.imag.square()).mean(dim=dims_complex)
    clean_power = rx_power / (1.0 + 1.0 / snr_linear)
    noise_var = (clean_power / (2.0 * snr_linear)).view(-1, 1, 1, 1).clamp_min(1e-12)

    spatial_dims = (2, 3)
    mean_y = z_use.mean(dim=spatial_dims, keepdim=True)
    var_y = z_use.var(dim=spatial_dims, unbiased=False, keepdim=True)
    signal_var = (var_y - noise_var).clamp_min(0.0)
    gain = signal_var / (signal_var + noise_var)
    z_clean = mean_y + gain * (z_use - mean_y)
    if strength < 1.0:
        z_clean = float(strength) * z_clean + (1.0 - float(strength)) * z_use

    a = a_matrix.to(device=z_rx.device, dtype=z_clean.dtype)
    return torch.einsum("oc,bohw->bchw", a, z_clean).to(dtype=z_rx.dtype)


def norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean.to(device=x.device, dtype=x.dtype)) / std.to(device=x.device, dtype=x.dtype).clamp_min(1e-6)


def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std.to(device=x.device, dtype=x.dtype).clamp_min(1e-6) + mean.to(device=x.device, dtype=x.dtype)


def min_snr_weight(alpha_bar: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        return torch.ones_like(alpha_bar)
    snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-8)
    return torch.minimum(snr, torch.full_like(snr, float(gamma))) / snr.clamp_min(1e-8)


@torch.no_grad()
def make_anchor_batch(
    system,
    refiner,
    anchor_mode: str,
    imgs: torch.Tensor,
    a_matrix: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
):
    autocast_cm = (
        torch.autocast(imgs.device.type, enabled=amp_enabled, dtype=amp_dtype)
        if imgs.device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with autocast_cm:
        z_sem = system.semantic_encoder(imgs)
        z_ch = system.channel_encoder(z_sem)
    z_rx, _sigma_y, _beta = system.mimo.forward(z_ch.float())
    with autocast_cm:
        z_low = system.channel_decoder(z_rx.to(z_ch.dtype))
    if anchor_mode == "refiner":
        if refiner is None:
            raise ValueError("anchor_mode=refiner requires a refiner")
        z_anchor = refiner(z_low.to(next(refiner.parameters()).dtype)).float()
    elif anchor_mode == "low":
        z_anchor = z_low.float()
    elif anchor_mode == "noise_scaled_low":
        noise_scale = float(getattr(system, "_low_noise_scale", 1.0))
        z_rx_eff = z_ch.float() + noise_scale * (z_rx.float() - z_ch.float())
        with autocast_cm:
            z_anchor = system.channel_decoder(z_rx_eff.to(z_ch.dtype)).float()
    elif anchor_mode == "wiener_low":
        if getattr(system.mimo, "fading", "awgn") != "awgn":
            raise ValueError("anchor_mode=wiener_low currently supports AWGN only")
        z_anchor = awgn_wiener_low_anchor(
            z_rx.float(),
            a_matrix,
            snr_db=float(getattr(system.mimo, "snr_db", 12.0)),
            strength=float(getattr(system, "_wiener_low_strength", 1.0)),
        ).float()
    else:
        raise ValueError(f"unknown anchor_mode={anchor_mode!r}")
    residual = null_project(z_sem.float() - z_anchor.float(), a_matrix)
    return z_anchor.float(), residual.float()


@torch.no_grad()
def precache_latents(
    system,
    refiner,
    anchor_mode: str,
    loader,
    a_matrix,
    device,
    amp_enabled,
    amp_dtype,
    repeats: int,
    max_batches: int,
    split_name: str,
):
    refs, residuals, imgs_all = [], [], []
    for rep in range(max(1, int(repeats))):
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
            z_ref, residual = make_anchor_batch(
                system, refiner, anchor_mode, imgs, a_matrix, amp_enabled, amp_dtype
            )
            refs.append(z_ref.cpu())
            residuals.append(residual.cpu())
            imgs_all.append(imgs.float().cpu())
        print(f"  cached {split_name} repeat {rep + 1}/{max(1, int(repeats))}")
    return ResidualLatentDataset(torch.cat(refs), torch.cat(residuals), torch.cat(imgs_all))


def channel_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x64 = x.double()
    dims = (0, 2, 3)
    mean = x64.mean(dim=dims).float().view(1, -1, 1, 1)
    std = x64.std(dim=dims, unbiased=False).clamp_min(1e-6).float().view(1, -1, 1, 1)
    return mean, std


def load_or_make_stats(train_ds: ResidualLatentDataset):
    cond_mean, cond_std = channel_stats(train_ds.z_ref)
    res_mean, res_std = channel_stats(train_ds.residual)
    return {
        "cond_mean": cond_mean,
        "cond_std": cond_std,
        "res_mean": res_mean,
        "res_std": res_std,
    }


@torch.no_grad()
def sample_residual(model, z_ref, stats, alpha_bars, a_matrix, *, num_steps: int, t_start: int, z0_clip: float):
    device = z_ref.device
    cond_norm = norm(z_ref, stats["cond_mean"], stats["cond_std"])
    zero_res_norm = norm(torch.zeros_like(z_ref), stats["res_mean"], stats["res_std"])
    n_total = int(alpha_bars.shape[0])
    t_start = max(0, min(int(t_start), n_total - 1))
    alpha_start = alpha_bars[t_start].to(device=device, dtype=z_ref.dtype)
    z = torch.sqrt(alpha_start) * zero_res_norm + torch.sqrt(1.0 - alpha_start) * torch.randn_like(z_ref)
    step_indices = torch.linspace(t_start, 0, int(num_steps), device=device).long()
    for i, idx in enumerate(step_indices):
        alpha_bar = alpha_bars[idx].to(device=device, dtype=z.dtype)
        alpha_prev = (
            alpha_bars[step_indices[i + 1]].to(device=device, dtype=z.dtype)
            if i + 1 < len(step_indices)
            else torch.tensor(1.0, device=device, dtype=z.dtype)
        )
        t_cont = torch.full((z.shape[0],), float(idx.item()) / float(max(1, n_total - 1)), device=device, dtype=z.dtype)
        eps = model(z, t_cont, cond=cond_norm)
        z0_norm = (z - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar + 1e-8)
        if z0_clip > 0:
            z0_norm = z0_norm.clamp(-float(z0_clip), float(z0_clip))
        r0 = null_project(denorm(z0_norm, stats["res_mean"], stats["res_std"]), a_matrix)
        z0_norm = norm(r0, stats["res_mean"], stats["res_std"])
        z = torch.sqrt(alpha_prev) * z0_norm + torch.sqrt(1.0 - alpha_prev) * eps
    return null_project(denorm(z, stats["res_mean"], stats["res_std"]), a_matrix)


def save_checkpoint(path, model, ema, opt, scaler, args, a_matrix, stats, epoch, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "a_matrix": a_matrix.detach().cpu(),
            "stats": {k: v.detach().cpu() for k, v in stats.items()},
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
            "diffusion_target": f"{args.anchor_mode}_null_residual",
        },
        path,
    )


def load_resume(path, model, ema, opt, scaler, device) -> tuple[int, float]:
    if not path:
        return 1, -1.0
    ckpt_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    ema_state = ckpt.get("ema_state_dict", {})
    if isinstance(ema_state, dict) and "shadow" in ema_state:
        model_state = model.state_dict()
        ema.decay = float(ema_state.get("decay", ema.decay))
        ema.shadow = {
            k: v.to(device=model_state[k].device, dtype=model_state[k].dtype)
            for k, v in ema_state["shadow"].items()
            if k in model_state
        }
    if "optimizer_state_dict" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    metrics = ckpt.get("metrics", {}) or {}
    best = float(metrics.get("val_psnr_diffusion", -1.0))
    print(f"resumed {ckpt_path} epoch={ckpt.get('epoch')} best={best:.4f}")
    return int(ckpt.get("epoch", 0)) + 1, best


@torch.no_grad()
def validate(system, model, ema, loader, stats, alpha_bars, a_matrix, args, device, amp_enabled, amp_dtype):
    model.eval()
    backup = ema.apply_to(model)
    anchor_vals, diff_vals = [], []
    decoder_dtype = next(system.semantic_decoder.parameters()).dtype
    for bi, batch in enumerate(loader):
        if args.max_val_batches > 0 and bi >= args.max_val_batches:
            break
        z_ref, _residual, imgs = batch
        z_ref = z_ref.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)
        r = sample_residual(
            model,
            z_ref.float(),
            stats,
            alpha_bars,
            a_matrix,
            num_steps=int(args.sample_steps),
            t_start=int(args.sample_t_start),
            z0_clip=float(args.z0_clip),
        )
        z_hat = z_ref.float() + r
        autocast_cm = (
            torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with autocast_cm:
            x_ref = system.semantic_decoder(z_ref.to(decoder_dtype))
            x_hat = system.semantic_decoder(z_hat.to(decoder_dtype))
        anchor_vals.append(psnr_per_image(x_ref.float().clamp(0, 1), imgs.float()))
        diff_vals.append(psnr_per_image(x_hat.float().clamp(0, 1), imgs.float()))
    model.load_state_dict(backup)
    ref = torch.cat(anchor_vals).mean().item() if anchor_vals else float("nan")
    diff = torch.cat(diff_vals).mean().item() if diff_vals else float("nan")
    return {
        "val_psnr_anchor": float(ref),
        "val_psnr_diffusion": float(diff),
        "val_gain_vs_anchor": float(diff - ref),
    }


def main() -> None:
    args = parse_args()
    setup_log_file(args.log_file)
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

    train_img_ds = DIV2KDataset(
        args.data_dir,
        crop_size=args.crop_size,
        split="train",
        cache_decoded=bool(args.cache_decoded),
        cache_workers=int(args.cache_workers),
    )
    val_img_ds = DIV2KDataset(
        args.data_dir,
        crop_size=args.crop_size,
        split="valid",
        cache_decoded=bool(args.cache_decoded),
        cache_workers=int(args.cache_workers),
    )
    train_img_loader = DataLoader(
        train_img_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if args.num_workers > 0 else None),
    )
    val_img_loader = DataLoader(
        val_img_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.val_num_workers > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if args.val_num_workers > 0 else None),
    )
    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}")
    print(
        f"train={len(train_img_ds)} valid={len(val_img_ds)} snr={args.snr_db:g}dB "
        f"target={args.anchor_mode}_null_residual"
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
    system._wiener_low_strength = float(args.wiener_low_strength)
    system._low_noise_scale = float(args.low_noise_scale)
    refiner = None
    if args.anchor_mode == "refiner":
        refiner = load_latent_refiner(args.refiner_ckpt, args.refiner_hidden, args.refiner_depth, device)
        refiner.eval()
    for p in system.parameters():
        p.requires_grad_(False)
    if refiner is not None:
        for p in refiner.parameters():
            p.requires_grad_(False)
    a_matrix = system._channel_encoder_matrix().detach().cpu().float()

    print(f"Pre-caching {args.anchor_mode} anchors and residual targets ...")
    train_ds = precache_latents(
        system, refiner, args.anchor_mode, train_img_loader, a_matrix, device, amp_enabled, amp_dtype,
        repeats=int(args.precache_repeats), max_batches=int(args.max_train_batches), split_name="train",
    )
    val_ds = precache_latents(
        system, refiner, args.anchor_mode, val_img_loader, a_matrix, device, amp_enabled, amp_dtype,
        repeats=1, max_batches=int(args.max_val_batches), split_name="valid",
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        prefetch_factor=max(2, int(args.prefetch_factor)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        prefetch_factor=max(2, int(args.prefetch_factor)),
    )
    stats = load_or_make_stats(train_ds)
    print(
        f"cached train={len(train_ds)} valid={len(val_ds)} "
        f"anchor_mode={args.anchor_mode} "
        f"res_std=[{stats['res_std'].min().item():.5f},{stats['res_std'].max().item():.5f}] "
        f"cond_std=[{stats['cond_std'].min().item():.5f},{stats['cond_std'].max().item():.5f}]"
    )

    alpha_bars = make_schedule(args.num_train_steps, args.noise_schedule, device)
    stats = {k: v.to(device) for k, v in stats.items()}
    model = UNetDenoiser(channels=16, hidden_dim=int(args.hidden), use_cond=True).to(device)
    ema = EMA(model, decay=float(args.ema_decay))
    opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    start_epoch, best_psnr = load_resume(args.resume_ckpt, model, ema, opt, scaler, device)

    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(PROJECT_ROOT, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "route_a_res_cond_diff_div2k_c16_awgn12_best.pth")
    latest_path = os.path.join(save_dir, "route_a_res_cond_diff_div2k_c16_awgn12_latest.pth")

    print(
        f"arch=UNetDenoiser(cond) hidden={args.hidden} tag={tag} schedule={args.noise_schedule} "
        f"anchor={args.anchor_mode} sample=zero_residual:{args.sample_steps}@{args.sample_t_start} "
        f"start_epoch={start_epoch}"
    )
    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        loss_meter = AverageMeter()
        eps_meter = AverageMeter()
        img_meter = AverageMeter()
        for bi, batch in enumerate(train_loader):
            z_ref, residual, imgs = batch
            z_ref = z_ref.to(device, non_blocking=True)
            residual = residual.to(device, non_blocking=True)
            imgs = imgs.to(device, non_blocking=True)
            res_norm = norm(residual, stats["res_mean"], stats["res_std"])
            cond_norm = norm(z_ref, stats["cond_mean"], stats["cond_std"])
            bsz = imgs.shape[0]
            t_idx = torch.randint(0, int(args.num_train_steps), (bsz,), device=device, dtype=torch.long)
            alpha_bar = alpha_bars[t_idx].view(-1, 1, 1, 1).to(dtype=res_norm.dtype)
            eps = torch.randn_like(res_norm)
            z_t = torch.sqrt(alpha_bar) * res_norm + torch.sqrt(1.0 - alpha_bar) * eps
            t_cont = t_idx.to(dtype=res_norm.dtype) / float(max(1, int(args.num_train_steps) - 1))

            autocast_cm = (
                torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            opt.zero_grad(set_to_none=True)
            with autocast_cm:
                eps_pred = model(z_t, t_cont, cond=cond_norm)
                weight = min_snr_weight(alpha_bar.float(), float(args.min_snr_gamma))
                loss_eps = (weight * (eps_pred.float() - eps.float()).pow(2)).mean()
                z0_norm = (z_t.float() - torch.sqrt(1.0 - alpha_bar.float()) * eps_pred.float()) / torch.sqrt(alpha_bar.float() + 1e-8)
                if args.z0_clip > 0:
                    z0_norm = z0_norm.clamp(-float(args.z0_clip), float(args.z0_clip))
                res_pred = null_project(denorm(z0_norm, stats["res_mean"], stats["res_std"]), a_matrix)
                z_hat = z_ref.float() + res_pred
                x_hat = system.semantic_decoder(z_hat.to(next(system.semantic_decoder.parameters()).dtype))
                loss_img_raw = F.mse_loss(x_hat.float(), imgs.float())
                alpha_img = alpha_bar.float().mean().detach().clamp_min(0.0).pow(float(args.image_loss_alpha_power))
                loss_img = float(args.lambda_image) * alpha_img * loss_img_raw
                loss = loss_eps + loss_img
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            n = imgs.shape[0]
            loss_meter.update(float(loss.item()), n)
            eps_meter.update(float(loss_eps.item()), n)
            img_meter.update(float(loss_img_raw.item()), n)

        metrics = {
            "train_loss": loss_meter.avg,
            "train_loss_eps": eps_meter.avg,
            "train_loss_img_raw": img_meter.avg,
        }
        if epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs):
            val_metrics = validate(system, model, ema, val_loader, stats, alpha_bars, a_matrix, args, device, amp_enabled, amp_dtype)
            metrics.update(val_metrics)
            is_best = val_metrics["val_psnr_diffusion"] > best_psnr
            if is_best:
                best_psnr = val_metrics["val_psnr_diffusion"]
                save_checkpoint(best_path, model, ema, opt, scaler, args, a_matrix, stats, epoch, metrics)
            save_checkpoint(latest_path, model, ema, opt, scaler, args, a_matrix, stats, epoch, metrics)
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={loss_meter.avg:.6f} "
                f"eps={eps_meter.avg:.6f} img={img_meter.avg:.6f} | "
                f"anchor={val_metrics['val_psnr_anchor']:.4f} diff={val_metrics['val_psnr_diffusion']:.4f} "
                f"vs_anchor={val_metrics['val_gain_vs_anchor']:+.4f} {'BEST' if is_best else ''}"
            )
        else:
            print(f"[epoch {epoch:03d}/{args.epochs}] loss={loss_meter.avg:.6f} eps={eps_meter.avg:.6f} img={img_meter.avg:.6f}")
    print(f"best_psnr={best_psnr:.4f} ckpt={best_path}")


if __name__ == "__main__":
    main()
