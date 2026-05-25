#!/usr/bin/env python
"""Train Route-A conditional diffusion in the null space.

The value-space latent is fixed by the received 4-channel observation:

    z_low = A^T y_rx

The diffusion model learns only the null-space distribution conditioned on
``z_low``:

    z_hat = z_low + P_null z_null_sample,  P_null = I - A^T A

This is the conditional counterpart to the deterministic null refiner and is
selected by AWGN12 full-link validation PSNR.
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
from torch.utils.data import DataLoader

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


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Route-A conditional diffusion session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Route-A conditional null-space diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_num_workers", type=int, default=2)
    p.add_argument("--cache_decoded", action="store_true")
    p.add_argument("--cache_workers", type=int, default=16)
    p.add_argument("--prefetch_factor", type=int, default=2)

    p.add_argument("--sc_encoder_ckpt", type=str, required=True)
    p.add_argument("--sc_decoder_ckpt", type=str, required=True)
    p.add_argument("--cc_dir", type=str, required=True)
    p.add_argument("--unet_ckpt", type=str, required=True,
                   help="Only used for system assembly and schedule metadata compatibility.")
    p.add_argument("--compression_ratio", type=float, default=0.25)
    p.add_argument("--snr_db", type=float, default=12.0)
    p.add_argument("--fading", type=str, default="awgn", choices=["awgn", "rayleigh"])

    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_train_steps", type=int, default=1000)
    p.add_argument("--noise_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--min_snr_gamma", type=float, default=5.0)
    p.add_argument("--lambda_image", type=float, default=1.0)
    p.add_argument("--image_loss_alpha_power", type=float, default=1.0,
                   help="Image loss multiplier uses alpha_bar**power; 1.0 strongly suppresses high-noise steps.")
    p.add_argument("--z0_clip", type=float, default=5.0,
                   help="Clamp predicted normalized z0 before denorm/image loss; <=0 disables.")
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    p.add_argument("--sample_steps", type=int, default=30)
    p.add_argument("--sample_t_start", type=int, default=999)
    p.add_argument("--sample_init", type=str, default="noise", choices=["noise", "zero"],
                   help="noise: start from pure Gaussian; zero: SDEdit-style noising from z_null=0.")
    p.add_argument("--eval_every_epochs", type=int, default=2)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--stat_batches", type=int, default=0,
                   help="Latent stats batches; 0 uses the full training loader.")
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_file", type=str, default="log-v2/route_a/cond_diff_awgn12.txt")
    p.add_argument("--save_dir", type=str, default="checkpoints-val-v2/route_a/cond_diff_awgn12")
    p.add_argument("--resume_ckpt", type=str, default="",
                   help="Resume model/EMA/optimizer state from a conditional diffusion checkpoint.")
    return p.parse_args()


def make_schedule(num_steps: int, schedule: str, device: torch.device) -> torch.Tensor:
    if schedule == "linear":
        betas = torch.linspace(1e-4, 2e-2, num_steps, device=device)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    elif schedule == "cosine":
        steps = torch.arange(num_steps + 1, device=device, dtype=torch.float32)
        s = 0.008
        f = torch.cos(((steps / num_steps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bars = (f / f[0]).clamp_min(1e-8)
        betas = (1.0 - alpha_bars[1:] / alpha_bars[:-1]).clamp(1e-8, 0.999)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    else:
        raise ValueError(f"unknown noise_schedule={schedule!r}")
    return alpha_bars


def project_low(z: torch.Tensor, a_matrix: torch.Tensor) -> torch.Tensor:
    a = a_matrix.to(device=z.device, dtype=z.dtype)
    az = torch.einsum("oc,bchw->bohw", a, z)
    return torch.einsum("oc,bohw->bchw", a, az)


def null_project(z: torch.Tensor, a_matrix: torch.Tensor) -> torch.Tensor:
    return z - project_low(z, a_matrix)


@torch.no_grad()
def make_batch_latents(system, imgs: torch.Tensor, amp_enabled: bool, amp_dtype: torch.dtype):
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
    return z_sem.float(), z_low.float()


def _stat_update(x: torch.Tensor, sum_: torch.Tensor, sumsq: torch.Tensor, count: int):
    x64 = x.detach().double()
    dims = (0, 2, 3)
    return sum_ + x64.sum(dim=dims).cpu(), sumsq + (x64 * x64).sum(dim=dims).cpu(), count + x.shape[0] * x.shape[2] * x.shape[3]


@torch.no_grad()
def estimate_stats(system, loader: DataLoader, a_matrix: torch.Tensor, device, amp_enabled, amp_dtype, max_batches: int):
    low_sum = torch.zeros(16, dtype=torch.float64)
    low_sumsq = torch.zeros(16, dtype=torch.float64)
    null_sum = torch.zeros(16, dtype=torch.float64)
    null_sumsq = torch.zeros(16, dtype=torch.float64)
    low_count = 0
    null_count = 0
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        z_sem, z_low = make_batch_latents(system, imgs, amp_enabled, amp_dtype)
        z_null = null_project(z_sem, a_matrix)
        low_sum, low_sumsq, low_count = _stat_update(z_low, low_sum, low_sumsq, low_count)
        null_sum, null_sumsq, null_count = _stat_update(z_null, null_sum, null_sumsq, null_count)
    low_mean = low_sum / max(1, low_count)
    low_var = (low_sumsq / max(1, low_count) - low_mean.square()).clamp_min(1e-8)
    null_mean = null_sum / max(1, null_count)
    null_var = (null_sumsq / max(1, null_count) - null_mean.square()).clamp_min(1e-8)
    return {
        "low_mean": low_mean.float().view(1, 16, 1, 1),
        "low_std": torch.sqrt(low_var).float().view(1, 16, 1, 1),
        "null_mean": null_mean.float().view(1, 16, 1, 1),
        "null_std": torch.sqrt(null_var).float().view(1, 16, 1, 1),
    }


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
def sample_null(
    model: nn.Module,
    z_low: torch.Tensor,
    stats: dict,
    alpha_bars: torch.Tensor,
    a_matrix: torch.Tensor,
    *,
    num_steps: int,
    t_start: int,
    sample_init: str = "noise",
    z0_clip: float = 5.0,
) -> torch.Tensor:
    device = z_low.device
    low_norm = norm(z_low, stats["low_mean"], stats["low_std"])
    n_total = int(alpha_bars.shape[0])
    t_start = max(0, min(int(t_start), n_total - 1))
    step_indices = torch.linspace(t_start, 0, int(num_steps), device=device).long()
    if sample_init == "zero":
        z0_anchor = norm(torch.zeros_like(z_low), stats["null_mean"], stats["null_std"])
        alpha_start = alpha_bars[t_start].to(device=device, dtype=z_low.dtype)
        z = torch.sqrt(alpha_start) * z0_anchor + torch.sqrt(1.0 - alpha_start) * torch.randn_like(z_low)
    else:
        z = torch.randn_like(z_low)
    for i, idx in enumerate(step_indices):
        alpha_bar = alpha_bars[idx].to(device=device, dtype=z.dtype)
        alpha_prev = (
            alpha_bars[step_indices[i + 1]].to(device=device, dtype=z.dtype)
            if i + 1 < len(step_indices)
            else torch.tensor(1.0, device=device, dtype=z.dtype)
        )
        t_cont = torch.full((z.shape[0],), float(idx.item()) / float(max(1, n_total - 1)), device=device, dtype=z.dtype)
        eps = model(z, t_cont, cond=low_norm)
        z0 = (z - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar + 1e-8)
        if z0_clip > 0:
            z0 = z0.clamp(-float(z0_clip), float(z0_clip))
        z = torch.sqrt(alpha_prev) * z0 + torch.sqrt(1.0 - alpha_prev) * eps
    z_null = denorm(z, stats["null_mean"], stats["null_std"])
    return null_project(z_null.float(), a_matrix)


def save_checkpoint(
    path: str,
    model: nn.Module,
    ema: EMA,
    args,
    a_matrix: torch.Tensor,
    stats: dict,
    epoch: int,
    metrics: dict,
    optimizer: optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "a_matrix": a_matrix.detach().cpu(),
        "stats": {k: v.detach().cpu() for k, v in stats.items()},
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_resume_checkpoint(
    path: str,
    model: nn.Module,
    ema: EMA,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float]:
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

    opt_state = ckpt.get("optimizer_state_dict")
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    scaler_state = ckpt.get("scaler_state_dict")
    if scaler_state:
        scaler.load_state_dict(scaler_state)

    epoch = int(ckpt.get("epoch", 0))
    metrics = ckpt.get("metrics", {}) or {}
    best_psnr = float(metrics.get("val_psnr_diffusion", -1.0))
    print(f"resumed checkpoint: {ckpt_path} epoch={epoch} best_psnr={best_psnr:.4f}")
    return epoch, best_psnr


@torch.no_grad()
def validate(
    *,
    system,
    model: nn.Module,
    ema: EMA,
    loader: DataLoader,
    a_matrix: torch.Tensor,
    stats: dict,
    alpha_bars: torch.Tensor,
    args,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
) -> dict:
    model.eval()
    backup = ema.apply_to(model)
    base_psnrs: list[torch.Tensor] = []
    diff_psnrs: list[torch.Tensor] = []
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        _z_sem, z_low = make_batch_latents(system, imgs, amp_enabled, amp_dtype)
        z_null = sample_null(
            model,
            z_low,
            stats,
            alpha_bars,
            a_matrix,
            num_steps=args.sample_steps,
            t_start=args.sample_t_start,
            sample_init=args.sample_init,
            z0_clip=float(args.z0_clip),
        )
        z_hat = z_low + z_null
        autocast_cm = (
            torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with autocast_cm:
            x_base = system.semantic_decoder(z_low.to(next(system.semantic_decoder.parameters()).dtype))
            x_hat = system.semantic_decoder(z_hat.to(next(system.semantic_decoder.parameters()).dtype))
        base_psnrs.append(psnr_per_image(x_base.float().clamp(0, 1), imgs.float()))
        diff_psnrs.append(psnr_per_image(x_hat.float().clamp(0, 1), imgs.float()))
    model.load_state_dict(backup)
    base = torch.cat(base_psnrs).mean().item() if base_psnrs else float("nan")
    refined = torch.cat(diff_psnrs).mean().item() if diff_psnrs else float("nan")
    return {
        "val_psnr_base": float(base),
        "val_psnr_diffusion": float(refined),
        "val_gain": float(refined - base),
    }


def main() -> None:
    args = parse_args()
    setup_log_file(args.log_file)
    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

    train_ds = DIV2KDataset(
        args.data_dir,
        crop_size=args.crop_size,
        split="train",
        cache_decoded=bool(args.cache_decoded),
        cache_workers=int(args.cache_workers),
    )
    val_ds = DIV2KDataset(
        args.data_dir,
        crop_size=args.crop_size,
        split="valid",
        cache_decoded=bool(args.cache_decoded),
        cache_workers=int(args.cache_workers),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.val_num_workers > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if args.val_num_workers > 0 else None),
    )

    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}")
    print(f"train={len(train_ds)} valid={len(val_ds)} snr={args.snr_db:g}dB fading={args.fading}")

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

    a_matrix = system._channel_encoder_matrix().detach().cpu().float()
    print("estimating low/null channel stats ...")
    stats = estimate_stats(
        system,
        train_loader,
        a_matrix,
        device,
        amp_enabled,
        amp_dtype,
        max_batches=int(args.stat_batches),
    )
    print(
        "stats: "
        f"low_std=[{stats['low_std'].min().item():.4f},{stats['low_std'].max().item():.4f}], "
        f"null_std=[{stats['null_std'].min().item():.4f},{stats['null_std'].max().item():.4f}]"
    )

    alpha_bars = make_schedule(args.num_train_steps, args.noise_schedule, device)
    model = UNetDenoiser(channels=16, hidden_dim=int(args.hidden), use_cond=True).to(device)
    ema = EMA(model, decay=float(args.ema_decay))
    opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(PROJECT_ROOT, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "route_a_cond_diff_null_div2k_c16_awgn12_best.pth")
    latest_path = os.path.join(save_dir, "route_a_cond_diff_null_div2k_c16_awgn12_latest.pth")
    best_psnr = -1.0
    start_epoch = 1
    if args.resume_ckpt:
        resume_epoch, resume_best = load_resume_checkpoint(
            args.resume_ckpt,
            model,
            ema,
            opt,
            scaler,
            device,
        )
        start_epoch = resume_epoch + 1
        best_psnr = max(best_psnr, resume_best)
    print(
        f"arch=UNetDenoiser(cond) hidden={args.hidden} tag={tag} "
        f"schedule={args.noise_schedule} T={args.num_train_steps} "
        f"sample={args.sample_init}:{args.sample_steps}@{args.sample_t_start} "
        f"z0_clip={args.z0_clip:g} image_alpha_power={args.image_loss_alpha_power:g} "
        f"start_epoch={start_epoch}"
    )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        loss_meter = AverageMeter()
        eps_meter = AverageMeter()
        img_raw_meter = AverageMeter()
        img_weighted_meter = AverageMeter()
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and bi >= args.max_train_batches:
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
            z_sem, z_low = make_batch_latents(system, imgs, amp_enabled, amp_dtype)
            z_null = null_project(z_sem, a_matrix)
            z_null_norm = norm(z_null, stats["null_mean"], stats["null_std"])
            z_low_norm = norm(z_low, stats["low_mean"], stats["low_std"])

            bsz = imgs.shape[0]
            t_idx = torch.randint(0, int(args.num_train_steps), (bsz,), device=device, dtype=torch.long)
            alpha_bar = alpha_bars[t_idx].view(-1, 1, 1, 1).to(dtype=z_null_norm.dtype)
            eps = torch.randn_like(z_null_norm)
            z_t = torch.sqrt(alpha_bar) * z_null_norm + torch.sqrt(1.0 - alpha_bar) * eps
            t_cont = t_idx.to(dtype=z_null_norm.dtype) / float(max(1, int(args.num_train_steps) - 1))

            autocast_cm = (
                torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            opt.zero_grad(set_to_none=True)
            with autocast_cm:
                eps_pred = model(z_t, t_cont, cond=z_low_norm)
                weight = min_snr_weight(alpha_bar.float(), float(args.min_snr_gamma))
                loss_eps = (weight * (eps_pred.float() - eps.float()).pow(2)).mean()
                z0_norm_pred = (z_t.float() - torch.sqrt(1.0 - alpha_bar.float()) * eps_pred.float()) / torch.sqrt(alpha_bar.float() + 1e-8)
                if args.z0_clip > 0:
                    z0_norm_pred = z0_norm_pred.clamp(-float(args.z0_clip), float(args.z0_clip))
                z_null_pred = null_project(denorm(z0_norm_pred, stats["null_mean"], stats["null_std"]), a_matrix)
                z_hat = z_low + z_null_pred
                x_hat = system.semantic_decoder(z_hat.to(next(system.semantic_decoder.parameters()).dtype))
                loss_img_raw = F.mse_loss(x_hat.float(), imgs.float())
                alpha_img = alpha_bar.float().mean().detach().clamp_min(0.0)
                if float(args.image_loss_alpha_power) != 1.0:
                    alpha_img = alpha_img.pow(float(args.image_loss_alpha_power))
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
            img_raw_meter.update(float(loss_img_raw.item()), n)
            img_weighted_meter.update(float(loss_img.item()), n)

        metrics = {
            "train_loss": loss_meter.avg,
            "train_loss_eps": eps_meter.avg,
            "train_loss_img_raw": img_raw_meter.avg,
            "train_loss_img_weighted": img_weighted_meter.avg,
        }
        if epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs):
            val_metrics = validate(
                system=system,
                model=model,
                ema=ema,
                loader=val_loader,
                a_matrix=a_matrix,
                stats=stats,
                alpha_bars=alpha_bars,
                args=args,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                max_batches=int(args.max_val_batches),
            )
            metrics.update(val_metrics)
            is_best = val_metrics["val_psnr_diffusion"] > best_psnr
            if is_best:
                best_psnr = val_metrics["val_psnr_diffusion"]
                save_checkpoint(best_path, model, ema, args, a_matrix, stats, epoch, metrics, opt, scaler)
            save_checkpoint(latest_path, model, ema, args, a_matrix, stats, epoch, metrics, opt, scaler)
            print(
                f"[epoch {epoch:03d}/{args.epochs}] "
                f"loss={loss_meter.avg:.6f} eps={eps_meter.avg:.6f} "
                f"img_raw={img_raw_meter.avg:.6f} img_w={img_weighted_meter.avg:.6f} | "
                f"val_base={val_metrics['val_psnr_base']:.4f} "
                f"val_diff={val_metrics['val_psnr_diffusion']:.4f} "
                f"gain={val_metrics['val_gain']:+.4f} "
                f"{'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] "
                f"loss={loss_meter.avg:.6f} eps={eps_meter.avg:.6f} "
                f"img_raw={img_raw_meter.avg:.6f} img_w={img_weighted_meter.avg:.6f}"
            )

    print(f"best_psnr={best_psnr:.4f} ckpt={best_path}")


if __name__ == "__main__":
    main()
