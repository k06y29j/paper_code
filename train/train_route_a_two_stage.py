#!/usr/bin/env python
"""Train Route-A low denoiser + high-frequency refiner from rule.md.

The frozen base path is:

    z_16 = SemanticEncoder(x)
    z_4 = A z_16
    y_4 = AWGN(z_4, SNR)

Stage 1 trains a conditional low-frequency denoiser:

    z_4_hat = y_4 - N_theta(y_4, snr)

Stage 2 freezes the denoiser and trains a conditional null-space refiner:

    z_init = A^T z_4_hat
    z_refined = z_init + P_null(A) R_phi(z_init, z_4_hat, snr)

The semantic encoder/decoder and channel matrix are never updated.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import DIV2KDataset  # noqa: E402


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


class CachedRouteADataset(Dataset):
    def __init__(self, z_sem: torch.Tensor, z_low: torch.Tensor, imgs: torch.Tensor) -> None:
        self.z_sem = z_sem
        self.z_low = z_low
        self.imgs = imgs

    def __len__(self) -> int:
        return int(self.z_sem.shape[0])

    def __getitem__(self, idx):
        return self.z_sem[idx], self.z_low[idx], self.imgs[idx]


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 64, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("embedding dim must be even")
        self.dim = int(dim)
        self.max_period = float(max_period)

    def forward(self, snr_db: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = snr_db.device
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(1, half - 1)
        )
        args = snr_db.float().view(-1, 1) * freqs.view(1, -1)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 4 * channels),
        )
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

    @staticmethod
    def _film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1)
        return x * (1.0 + gamma.to(dtype=x.dtype)) + beta.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g1, b1, g2, b2 = self.cond(cond).chunk(4, dim=1)
        h = self._film(self.norm1(x), g1, b1)
        h = self.conv1(F.silu(h))
        h = self._film(self.norm2(h), g2, b2)
        h = self.conv2(F.silu(h))
        return x + h


class ConditionalLowDenoiser(nn.Module):
    def __init__(self, in_channels: int = 4, hidden: int = 64, depth: int = 4, emb_dim: int = 64) -> None:
        super().__init__()
        self.snr_embed = SinusoidalEmbedding(emb_dim)
        self.cond = nn.Sequential(
            nn.Linear(emb_dim, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden * 2),
        )
        self.in_conv = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMResBlock(hidden, hidden * 2) for _ in range(int(depth))])
        self.out_norm = nn.GroupNorm(min(8, hidden), hidden)
        self.out_conv = nn.Conv2d(hidden, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.noise_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, y_low: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        cond = self.cond(self.snr_embed(snr_db).to(dtype=y_low.dtype))
        h = self.in_conv(y_low)
        for block in self.blocks:
            h = block(h, cond)
        noise_hat = self.out_conv(F.silu(self.out_norm(h)))
        return y_low - self.noise_scale.to(dtype=y_low.dtype) * noise_hat


class ConditionalNullRefiner(nn.Module):
    def __init__(
        self,
        low_channels: int = 4,
        latent_channels: int = 16,
        hidden: int = 96,
        depth: int = 6,
        emb_dim: int = 64,
        a_matrix: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.snr_embed = SinusoidalEmbedding(emb_dim)
        self.cond = nn.Sequential(
            nn.Linear(emb_dim, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden * 2),
        )
        self.in_conv = nn.Conv2d(latent_channels + low_channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMResBlock(hidden, hidden * 2) for _ in range(int(depth))])
        self.out_norm = nn.GroupNorm(min(8, hidden), hidden)
        self.out_conv = nn.Conv2d(hidden, latent_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        if a_matrix is None:
            a_matrix = torch.eye(low_channels, latent_channels, dtype=torch.float32)
        a = torch.as_tensor(a_matrix, dtype=torch.float32)
        self.register_buffer("a_matrix", a, persistent=True)
        self.register_buffer("a_pinv", torch.linalg.pinv(a), persistent=True)

    def null_project(self, z: torch.Tensor) -> torch.Tensor:
        a = self.a_matrix.to(device=z.device, dtype=z.dtype)
        a_pinv = self.a_pinv.to(device=z.device, dtype=z.dtype)
        az = torch.einsum("oc,bchw->bohw", a, z)
        low_part = torch.einsum("co,bohw->bchw", a_pinv, az)
        return z - low_part

    def forward(self, z_init: torch.Tensor, z_low_hat: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        cond = self.cond(self.snr_embed(snr_db).to(dtype=z_init.dtype))
        h = self.in_conv(torch.cat([z_init, z_low_hat], dim=1))
        for block in self.blocks:
            h = block(h, cond)
        residual = self.out_conv(F.silu(self.out_norm(h)))
        residual = self.null_project(residual.float()).to(dtype=z_init.dtype)
        return z_init + self.res_scale.to(dtype=z_init.dtype) * residual


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Route-A two-stage session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_snr_values(raw: str) -> list[float]:
    values = [float(v) for v in raw.replace(",", " ").split() if v.strip()]
    if not values:
        raise ValueError("at least one SNR value is required")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Route-A rule.md two-stage receiver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", action="store_false", dest="cache_decoded")
    p.add_argument("--cache_workers", type=int, default=12)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--precache_latents", action="store_true", default=True)
    p.add_argument("--no_precache_latents", action="store_false", dest="precache_latents")

    p.add_argument("--sc_encoder_ckpt", type=str, required=True)
    p.add_argument("--sc_decoder_ckpt", type=str, required=True)
    p.add_argument("--cc_dir", type=str, required=True)
    p.add_argument("--unet_ckpt", type=str, required=True)
    p.add_argument("--compression_ratio", type=float, default=0.25)
    p.add_argument("--snr_db", type=float, default=12.0)
    p.add_argument("--train_snr_values", type=str, default="12")

    p.add_argument("--low_hidden", type=int, default=64)
    p.add_argument("--low_depth", type=int, default=4)
    p.add_argument("--low_epochs", type=int, default=160)
    p.add_argument("--low_lr", type=float, default=2e-4)
    p.add_argument("--lambda_low_feat", type=float, default=1.0)
    p.add_argument("--lambda_low_noise", type=float, default=0.1)
    p.add_argument("--lambda_low_img", type=float, default=0.05)

    p.add_argument("--refiner_hidden", type=int, default=96)
    p.add_argument("--refiner_depth", type=int, default=6)
    p.add_argument("--refiner_epochs", type=int, default=240)
    p.add_argument("--refiner_lr", type=float, default=2e-4)
    p.add_argument("--lambda_refiner_latent", type=float, default=1.0)
    p.add_argument("--lambda_refiner_img", type=float, default=1.0)
    p.add_argument("--lambda_refiner_null", type=float, default=0.1)

    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260523)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_file", type=str, default="log-v2/two-stage/awgn12_rule.txt")
    p.add_argument("--save_dir", type=str, default="checkpoints-val-v2/two-stage/awgn12_rule")
    p.add_argument("--stage", type=str, default="both", choices=["low", "refiner", "both"])
    p.add_argument("--low_denoiser_ckpt", type=str, default="")
    return p.parse_args()


@torch.no_grad()
def add_awgn_siso_real(
    z_low: torch.Tensor,
    snr_db: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    bsz, channels, _h, _w = z_low.shape
    if channels % 2 != 0:
        raise ValueError("low channel count must be even for SISO complex pairing")
    z_use = z_low.float()
    z_complex = torch.complex(z_use[:, 0::2], z_use[:, 1::2])
    dims = tuple(range(1, z_complex.ndim))
    power = (z_complex.real.square() + z_complex.imag.square()).mean(dim=dims)
    scale = torch.sqrt(power.clamp_min(1e-12)).view(bsz, 1, 1, 1)
    x_norm = z_complex / scale.to(dtype=z_complex.dtype)

    snr_linear = torch.pow(torch.tensor(10.0, device=z_low.device), snr_db.float().view(bsz) / 10.0)
    sigma = torch.sqrt(1.0 / (2.0 * snr_linear)).view(bsz, 1, 1, 1)
    noise_r = torch.randn(
        x_norm.real.shape,
        device=z_low.device,
        dtype=x_norm.real.dtype,
        generator=generator,
    ) * sigma
    noise_i = torch.randn(
        x_norm.imag.shape,
        device=z_low.device,
        dtype=x_norm.imag.dtype,
        generator=generator,
    ) * sigma
    y = x_norm + torch.complex(noise_r, noise_i)
    y = y * scale.to(dtype=y.dtype)
    out = torch.empty_like(z_use)
    out[:, 0::2] = y.real
    out[:, 1::2] = y.imag
    return out.to(dtype=z_low.dtype)


def sample_snr(batch_size: int, values: list[float], device: torch.device) -> torch.Tensor:
    if len(values) == 1:
        return torch.full((batch_size,), float(values[0]), device=device, dtype=torch.float32)
    idx = torch.randint(0, len(values), (batch_size,), device=device)
    table = torch.tensor(values, device=device, dtype=torch.float32)
    return table[idx]


def make_autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device.type, enabled=enabled, dtype=dtype)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def cache_latents(system, loader, device, amp_enabled, amp_dtype, max_batches: int, split_name: str) -> CachedRouteADataset:
    z_sems: list[torch.Tensor] = []
    z_lows: list[torch.Tensor] = []
    imgs_all: list[torch.Tensor] = []
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        with make_autocast(device, amp_enabled, amp_dtype):
            z_sem = system.semantic_encoder(imgs)
            z_low = system.channel_encoder(z_sem)
        z_sems.append(z_sem.float().cpu())
        z_lows.append(z_low.float().cpu())
        imgs_all.append(imgs.float().cpu())
        if (bi + 1) % 10 == 0:
            print(f"  cached {split_name}: {bi + 1} batches")
    if not z_sems:
        raise RuntimeError(f"no {split_name} batches cached")
    return CachedRouteADataset(torch.cat(z_sems), torch.cat(z_lows), torch.cat(imgs_all))


def make_image_loaders(args: argparse.Namespace, device: torch.device):
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
    train_loader = DataLoader(
        train_img_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_img_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.val_num_workers > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if args.val_num_workers > 0 else None),
    )
    return train_img_ds, val_img_ds, train_loader, val_loader


def make_cached_loaders(args: argparse.Namespace, train_ds: Dataset, val_ds: Dataset, device: torch.device):
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
    return train_loader, val_loader


def decode_latent(system, z: torch.Tensor, amp_enabled: bool, amp_dtype: torch.dtype):
    device = z.device
    dec_dtype = next(system.semantic_decoder.parameters()).dtype
    with make_autocast(device, amp_enabled, amp_dtype):
        return system.semantic_decoder(z.to(dec_dtype))


def lift_low(system, z_low: torch.Tensor, amp_enabled: bool, amp_dtype: torch.dtype):
    device = z_low.device
    ch_dtype = next(system.channel_decoder.parameters()).dtype
    with make_autocast(device, amp_enabled, amp_dtype):
        return system.channel_decoder(z_low.to(ch_dtype)).float()


@dataclass
class Paths:
    low_best: str
    low_latest: str
    ref_best: str
    ref_latest: str


def make_paths(save_dir: str) -> Paths:
    os.makedirs(save_dir, exist_ok=True)
    return Paths(
        low_best=os.path.join(save_dir, "route_a_cond_low_denoiser_div2k_c16_awgn12_best.pth"),
        low_latest=os.path.join(save_dir, "route_a_cond_low_denoiser_div2k_c16_awgn12_latest.pth"),
        ref_best=os.path.join(save_dir, "route_a_cond_refiner_div2k_c16_awgn12_best.pth"),
        ref_latest=os.path.join(save_dir, "route_a_cond_refiner_div2k_c16_awgn12_latest.pth"),
    )


def save_low_checkpoint(path: str, model: nn.Module, args: argparse.Namespace, epoch: int, metrics: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "denoiser_type": "conditional_low_noise_predictor",
            "low_channels": 4,
            "hidden": int(args.low_hidden),
            "depth": int(args.low_depth),
            "snr_db": float(args.snr_db),
            "train_snr_values": parse_snr_values(args.train_snr_values),
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def save_refiner_checkpoint(
    path: str,
    refiner: nn.Module,
    denoiser: nn.Module,
    args: argparse.Namespace,
    a_matrix: torch.Tensor,
    epoch: int,
    metrics: dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": refiner.state_dict(),
            "low_denoiser_state_dict": denoiser.state_dict(),
            "refiner_type": "conditional_null_refiner",
            "denoiser_type": "conditional_low_noise_predictor",
            "low_channels": 4,
            "latent_channels": 16,
            "refiner_hidden": int(args.refiner_hidden),
            "refiner_depth": int(args.refiner_depth),
            "low_hidden": int(args.low_hidden),
            "low_depth": int(args.low_depth),
            "a_matrix": a_matrix.detach().cpu(),
            "snr_db": float(args.snr_db),
            "train_snr_values": parse_snr_values(args.train_snr_values),
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def load_low_checkpoint(path: str, model: nn.Module, device: torch.device) -> dict:
    obj = torch.load(path, map_location=device, weights_only=False)
    sd = obj.get("state_dict", obj)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  [low_denoiser] loaded {path}")
    if missing or unexpected:
        print(f"    missing={len(missing)}, unexpected={len(unexpected)}")
    if isinstance(obj, dict) and "metrics" in obj:
        print(f"    metrics={obj['metrics']}")
    return obj


@torch.no_grad()
def validate_low(
    *,
    system,
    denoiser: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator,
) -> dict:
    denoiser.eval()
    raw_vals: list[torch.Tensor] = []
    clean_vals: list[torch.Tensor] = []
    den_vals: list[torch.Tensor] = []
    feat_loss = AverageMeter()
    snr = torch.full((args.batch_size,), float(args.snr_db), device=device, dtype=torch.float32)
    for bi, batch in enumerate(loader):
        if args.max_val_batches > 0 and bi >= args.max_val_batches:
            break
        _z_sem, z_low, imgs = batch
        z_low = z_low.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)
        snr_b = snr[: z_low.shape[0]]
        y_low = add_awgn_siso_real(z_low, snr_b, generator=generator)
        z_hat = denoiser(y_low.float(), snr_b)
        feat_loss.update(float(F.mse_loss(z_hat.float(), z_low.float()).item()), int(z_low.shape[0]))

        z_raw = lift_low(system, y_low, amp_enabled, amp_dtype)
        z_clean = lift_low(system, z_low, amp_enabled, amp_dtype)
        z_den = lift_low(system, z_hat, amp_enabled, amp_dtype)
        x_raw = decode_latent(system, z_raw, amp_enabled, amp_dtype).float().clamp(0, 1)
        x_clean = decode_latent(system, z_clean, amp_enabled, amp_dtype).float().clamp(0, 1)
        x_den = decode_latent(system, z_den, amp_enabled, amp_dtype).float().clamp(0, 1)
        raw_vals.append(psnr_per_image(x_raw, imgs.float()))
        clean_vals.append(psnr_per_image(x_clean, imgs.float()))
        den_vals.append(psnr_per_image(x_den, imgs.float()))

    def avg(items: list[torch.Tensor]) -> float:
        return float(torch.cat(items).mean().item()) if items else float("nan")

    return {
        "val_low_feat_mse": feat_loss.avg,
        "val_psnr_low_raw": avg(raw_vals),
        "val_psnr_low_clean": avg(clean_vals),
        "val_psnr_low_denoised": avg(den_vals),
        "val_gain_low_vs_raw": avg(den_vals) - avg(raw_vals),
    }


@torch.no_grad()
def validate_refiner(
    *,
    system,
    denoiser: nn.Module,
    refiner: ConditionalNullRefiner,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator,
) -> dict:
    denoiser.eval()
    refiner.eval()
    raw_vals: list[torch.Tensor] = []
    den_vals: list[torch.Tensor] = []
    ref_vals: list[torch.Tensor] = []
    clean_low_vals: list[torch.Tensor] = []
    latent_loss = AverageMeter()
    null_loss = AverageMeter()
    snr = torch.full((args.batch_size,), float(args.snr_db), device=device, dtype=torch.float32)
    a = refiner.a_matrix.to(device=device, dtype=torch.float32)
    for bi, batch in enumerate(loader):
        if args.max_val_batches > 0 and bi >= args.max_val_batches:
            break
        z_sem, z_low, imgs = batch
        z_sem = z_sem.to(device, non_blocking=True)
        z_low = z_low.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)
        snr_b = snr[: z_low.shape[0]]
        y_low = add_awgn_siso_real(z_low, snr_b, generator=generator)
        z_low_hat = denoiser(y_low.float(), snr_b)
        z_raw = lift_low(system, y_low, amp_enabled, amp_dtype)
        z_clean_low = lift_low(system, z_low, amp_enabled, amp_dtype)
        z_init = lift_low(system, z_low_hat, amp_enabled, amp_dtype)
        z_ref = refiner(z_init.float(), z_low_hat.float(), snr_b)
        residual = z_ref.float() - z_init.float()
        ar = torch.einsum("oc,bchw->bohw", a, residual)
        latent_loss.update(float(F.mse_loss(z_ref.float(), z_sem.float()).item()), int(z_sem.shape[0]))
        null_loss.update(float(ar.square().mean().item()), int(z_sem.shape[0]))

        x_raw = decode_latent(system, z_raw, amp_enabled, amp_dtype).float().clamp(0, 1)
        x_clean = decode_latent(system, z_clean_low, amp_enabled, amp_dtype).float().clamp(0, 1)
        x_den = decode_latent(system, z_init, amp_enabled, amp_dtype).float().clamp(0, 1)
        x_ref = decode_latent(system, z_ref, amp_enabled, amp_dtype).float().clamp(0, 1)
        raw_vals.append(psnr_per_image(x_raw, imgs.float()))
        clean_low_vals.append(psnr_per_image(x_clean, imgs.float()))
        den_vals.append(psnr_per_image(x_den, imgs.float()))
        ref_vals.append(psnr_per_image(x_ref, imgs.float()))

    def avg(items: list[torch.Tensor]) -> float:
        return float(torch.cat(items).mean().item()) if items else float("nan")

    return {
        "val_refiner_latent_mse": latent_loss.avg,
        "val_refiner_null_mse": null_loss.avg,
        "val_psnr_raw": avg(raw_vals),
        "val_psnr_clean_low": avg(clean_low_vals),
        "val_psnr_denoised_low": avg(den_vals),
        "val_psnr_refined": avg(ref_vals),
        "val_gain_refined_vs_raw": avg(ref_vals) - avg(raw_vals),
        "val_gain_refined_vs_denoised_low": avg(ref_vals) - avg(den_vals),
    }


def train_low_stage(
    *,
    system,
    denoiser: ConditionalLowDenoiser,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    paths: Paths,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    values = parse_snr_values(args.train_snr_values)
    opt = optim.AdamW(denoiser.parameters(), lr=float(args.low_lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    best_psnr = -1.0
    print(
        f"\n--- Stage 1: conditional low denoiser ---\n"
        f"train_snr_values={values}, eval_snr={args.snr_db:g}, "
        f"hidden={args.low_hidden}, depth={args.low_depth}"
    )

    for epoch in range(1, int(args.low_epochs) + 1):
        denoiser.train()
        loss_meter = AverageMeter()
        feat_meter = AverageMeter()
        noise_meter = AverageMeter()
        img_meter = AverageMeter()
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and bi >= args.max_train_batches:
                break
            _z_sem, z_low, imgs = batch
            z_low = z_low.to(device, non_blocking=True)
            imgs = imgs.to(device, non_blocking=True)
            snr_b = sample_snr(z_low.shape[0], values, device)
            y_low = add_awgn_siso_real(z_low, snr_b)

            opt.zero_grad(set_to_none=True)
            with make_autocast(device, amp_enabled, amp_dtype):
                z_hat = denoiser(y_low.float(), snr_b)
                loss_feat = F.mse_loss(z_hat.float(), z_low.float())
                n_hat = y_low.float() - z_hat.float()
                n_gt = y_low.float() - z_low.float()
                loss_noise = F.mse_loss(n_hat, n_gt)
                if args.lambda_low_img > 0:
                    z_init = lift_low(system, z_hat, amp_enabled, amp_dtype)
                    x_low = decode_latent(system, z_init, amp_enabled, amp_dtype)
                    loss_img = F.mse_loss(x_low.float(), imgs.float())
                else:
                    loss_img = z_hat.new_zeros(())
                loss = (
                    float(args.lambda_low_feat) * loss_feat
                    + float(args.lambda_low_noise) * loss_noise
                    + float(args.lambda_low_img) * loss_img
                )
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), float(args.clip_grad_norm))
            scaler.step(opt)
            scaler.update()

            n = int(z_low.shape[0])
            loss_meter.update(float(loss.item()), n)
            feat_meter.update(float(loss_feat.item()), n)
            noise_meter.update(float(loss_noise.item()), n)
            img_meter.update(float(loss_img.item()), n)

        metrics = {
            "train_low_loss": loss_meter.avg,
            "train_low_feat_mse": feat_meter.avg,
            "train_low_noise_mse": noise_meter.avg,
            "train_low_img_mse": img_meter.avg,
        }
        if epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.low_epochs):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(args.seed) + 101)
            val_metrics = validate_low(
                system=system,
                denoiser=denoiser,
                loader=val_loader,
                args=args,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                generator=gen,
            )
            metrics.update(val_metrics)
            is_best = val_metrics["val_psnr_low_denoised"] > best_psnr
            if is_best:
                best_psnr = val_metrics["val_psnr_low_denoised"]
                save_low_checkpoint(paths.low_best, denoiser, args, epoch, metrics)
            save_low_checkpoint(paths.low_latest, denoiser, args, epoch, metrics)
            print(
                f"[low {epoch:03d}/{args.low_epochs}] "
                f"loss={loss_meter.avg:.6f} feat={feat_meter.avg:.6f} img={img_meter.avg:.6f} | "
                f"raw={val_metrics['val_psnr_low_raw']:.4f} "
                f"clean={val_metrics['val_psnr_low_clean']:.4f} "
                f"den={val_metrics['val_psnr_low_denoised']:.4f} "
                f"gain={val_metrics['val_gain_low_vs_raw']:+.4f} "
                f"{'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[low {epoch:03d}/{args.low_epochs}] "
                f"loss={loss_meter.avg:.6f} feat={feat_meter.avg:.6f} "
                f"noise={noise_meter.avg:.6f} img={img_meter.avg:.6f}"
            )
    print(f"stage1_best_psnr={best_psnr:.4f} ckpt={paths.low_best}")
    return best_psnr


def train_refiner_stage(
    *,
    system,
    denoiser: ConditionalLowDenoiser,
    refiner: ConditionalNullRefiner,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    paths: Paths,
    a_matrix: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    values = parse_snr_values(args.train_snr_values)
    for p in denoiser.parameters():
        p.requires_grad_(False)
    denoiser.eval()
    opt = optim.AdamW(refiner.parameters(), lr=float(args.refiner_lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    best_psnr = -1.0
    a = a_matrix.to(device=device, dtype=torch.float32)
    print(
        f"\n--- Stage 2: conditional null-space refiner ---\n"
        f"train_snr_values={values}, eval_snr={args.snr_db:g}, "
        f"hidden={args.refiner_hidden}, depth={args.refiner_depth}"
    )

    for epoch in range(1, int(args.refiner_epochs) + 1):
        refiner.train()
        loss_meter = AverageMeter()
        latent_meter = AverageMeter()
        image_meter = AverageMeter()
        null_meter = AverageMeter()
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and bi >= args.max_train_batches:
                break
            z_sem, z_low, imgs = batch
            z_sem = z_sem.to(device, non_blocking=True)
            z_low = z_low.to(device, non_blocking=True)
            imgs = imgs.to(device, non_blocking=True)
            snr_b = sample_snr(z_low.shape[0], values, device)
            y_low = add_awgn_siso_real(z_low, snr_b)

            with torch.no_grad():
                z_low_hat = denoiser(y_low.float(), snr_b)
                z_init = lift_low(system, z_low_hat, amp_enabled, amp_dtype)

            opt.zero_grad(set_to_none=True)
            with make_autocast(device, amp_enabled, amp_dtype):
                z_ref = refiner(z_init.float(), z_low_hat.float(), snr_b)
                residual = z_ref.float() - z_init.float()
                ar = torch.einsum("oc,bchw->bohw", a, residual)
                loss_latent = F.mse_loss(z_ref.float(), z_sem.float())
                x_ref = decode_latent(system, z_ref, amp_enabled, amp_dtype)
                loss_img = F.mse_loss(x_ref.float(), imgs.float())
                loss_null = ar.square().mean()
                loss = (
                    float(args.lambda_refiner_latent) * loss_latent
                    + float(args.lambda_refiner_img) * loss_img
                    + float(args.lambda_refiner_null) * loss_null
                )
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), float(args.clip_grad_norm))
            scaler.step(opt)
            scaler.update()

            n = int(z_sem.shape[0])
            loss_meter.update(float(loss.item()), n)
            latent_meter.update(float(loss_latent.item()), n)
            image_meter.update(float(loss_img.item()), n)
            null_meter.update(float(loss_null.item()), n)

        metrics = {
            "train_refiner_loss": loss_meter.avg,
            "train_refiner_latent_mse": latent_meter.avg,
            "train_refiner_img_mse": image_meter.avg,
            "train_refiner_null_mse": null_meter.avg,
        }
        if epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.refiner_epochs):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(args.seed) + 202)
            val_metrics = validate_refiner(
                system=system,
                denoiser=denoiser,
                refiner=refiner,
                loader=val_loader,
                args=args,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                generator=gen,
            )
            metrics.update(val_metrics)
            is_best = val_metrics["val_psnr_refined"] > best_psnr
            if is_best:
                best_psnr = val_metrics["val_psnr_refined"]
                save_refiner_checkpoint(paths.ref_best, refiner, denoiser, args, a_matrix, epoch, metrics)
            save_refiner_checkpoint(paths.ref_latest, refiner, denoiser, args, a_matrix, epoch, metrics)
            print(
                f"[ref {epoch:03d}/{args.refiner_epochs}] "
                f"loss={loss_meter.avg:.6f} latent={latent_meter.avg:.6f} "
                f"img={image_meter.avg:.6f} null={null_meter.avg:.6f} | "
                f"raw={val_metrics['val_psnr_raw']:.4f} "
                f"den_low={val_metrics['val_psnr_denoised_low']:.4f} "
                f"ref={val_metrics['val_psnr_refined']:.4f} "
                f"gain_raw={val_metrics['val_gain_refined_vs_raw']:+.4f} "
                f"gain_den={val_metrics['val_gain_refined_vs_denoised_low']:+.4f} "
                f"{'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[ref {epoch:03d}/{args.refiner_epochs}] "
                f"loss={loss_meter.avg:.6f} latent={latent_meter.avg:.6f} "
                f"img={image_meter.avg:.6f} null={null_meter.avg:.6f}"
            )
    print(f"stage2_best_psnr={best_psnr:.4f} ckpt={paths.ref_best}")
    return best_psnr


def main() -> None:
    args = parse_args()
    setup_log_file(args.log_file)
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(PROJECT_ROOT, args.save_dir)
    paths = make_paths(save_dir)
    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}")
    print(f"save_dir={save_dir}")
    print(f"rule: train/test SNR={float(args.snr_db):g} dB, PSNR=per-image mean")

    train_img_ds, val_img_ds, train_img_loader, val_img_loader = make_image_loaders(args, device)
    print(f"train={len(train_img_ds)} valid={len(val_img_ds)}")

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
    for p in system.parameters():
        p.requires_grad_(False)
    a_matrix = system._channel_encoder_matrix().detach().cpu().float()
    print(f"channel_matrix_shape={tuple(a_matrix.shape)} tag={tag}")

    if args.precache_latents:
        print("Pre-caching frozen SC/CC latents ...")
        train_cached = cache_latents(
            system,
            train_img_loader,
            device,
            amp_enabled,
            amp_dtype,
            max_batches=int(args.max_train_batches),
            split_name="train",
        )
        val_cached = cache_latents(
            system,
            val_img_loader,
            device,
            amp_enabled,
            amp_dtype,
            max_batches=int(args.max_val_batches),
            split_name="valid",
        )
        train_loader, val_loader = make_cached_loaders(args, train_cached, val_cached, device)
        print(f"cached train={len(train_cached)} valid={len(val_cached)}")
    else:
        raise RuntimeError("non-cached mode is not implemented for this training script")

    denoiser = ConditionalLowDenoiser(
        in_channels=int(a_matrix.shape[0]),
        hidden=int(args.low_hidden),
        depth=int(args.low_depth),
    ).to(device)
    if args.low_denoiser_ckpt:
        load_low_checkpoint(args.low_denoiser_ckpt, denoiser, device)

    if args.stage in ("low", "both") and not args.low_denoiser_ckpt:
        train_low_stage(
            system=system,
            denoiser=denoiser,
            train_loader=train_loader,
            val_loader=val_loader,
            args=args,
            paths=paths,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        load_low_checkpoint(paths.low_best, denoiser, device)

    if args.stage in ("refiner", "both"):
        if args.stage == "refiner" and not args.low_denoiser_ckpt and os.path.isfile(paths.low_best):
            load_low_checkpoint(paths.low_best, denoiser, device)
        refiner = ConditionalNullRefiner(
            low_channels=int(a_matrix.shape[0]),
            latent_channels=16,
            hidden=int(args.refiner_hidden),
            depth=int(args.refiner_depth),
            a_matrix=a_matrix,
        ).to(device)
        train_refiner_stage(
            system=system,
            denoiser=denoiser,
            refiner=refiner,
            train_loader=train_loader,
            val_loader=val_loader,
            args=args,
            paths=paths,
            a_matrix=a_matrix,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )


if __name__ == "__main__":
    main()
