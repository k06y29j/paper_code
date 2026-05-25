#!/usr/bin/env python
"""Fixed-link AWGN12 C=4 high-frequency recovery experiments.

Both routes keep the Swin semantic encoder/decoder and the channel
encoder/decoder frozen.  The receiver observes the fixed channel output

    z_sem -> A -> power normalization + AWGN12 -> fixed channel decoder

and learns only the missing null-space/high-frequency component.
"""

from __future__ import annotations

import argparse
import builtins
import copy
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
TRAIN_DIR = os.path.abspath(os.path.dirname(__file__))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from src.cddm_mimo_ddnm import DIV2KDataset  # noqa: E402
from src.cddm_mimo_ddnm.modules.ddnm import UNetDenoiser  # noqa: E402
from train_codex_orthogonal_highfreq import (  # noqa: E402
    AverageMeter,
    TeeStream,
    build_system_for_ratio,
    channel_matrix,
    decode_latent,
    encode_with_a,
    lift_with_at,
    lift_with_decoder,
    make_autocast,
    null_project,
    power_normalize_awgn,
    psnr_per_image,
    seed_everything,
    semiorth_error,
    _parse_amp,
)


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


class ConvResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConditionEncoder(nn.Module):
    """Compress receiver observations into a 16-channel diffusion condition."""

    def __init__(self, in_channels: int = 49, hidden: int = 128, depth: int = 4, out_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            *[ConvResBlock(hidden) for _ in range(int(depth))],
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DirectNullPredictor(nn.Module):
    def __init__(self, in_channels: int = 21, hidden: int = 192, depth: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            *[ConvResBlock(hidden) for _ in range(int(depth))],
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 16, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.res_scale.to(dtype=obs.dtype) * self.net(obs)


class MultiScaleNullPredictor(nn.Module):
    def __init__(self, in_channels: int = 21, hidden: int = 192, depth: int = 6) -> None:
        super().__init__()
        branch_ch = max(16, hidden // 4)
        self.stem = nn.Sequential(nn.Conv2d(in_channels, hidden, 3, padding=1), nn.SiLU())
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(hidden, branch_ch, 3, padding=d, dilation=d)
                for d in (1, 2, 3, 5)
            ]
        )
        self.fuse = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(branch_ch * 4, hidden, 1),
            *[ConvResBlock(hidden) for _ in range(int(depth))],
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 16, 3, padding=1),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.stem(obs)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.res_scale.to(dtype=obs.dtype) * self.fuse(x)


class UNetNullPredictor(nn.Module):
    def __init__(self, in_channels: int = 21, hidden: int = 160, depth: int = 2) -> None:
        super().__init__()
        h1 = hidden
        h2 = hidden * 2
        h3 = hidden * 3
        self.stem = nn.Sequential(nn.Conv2d(in_channels, h1, 3, padding=1), nn.SiLU())
        self.enc1 = nn.Sequential(*[ConvResBlock(h1) for _ in range(max(1, int(depth)))])
        self.down1 = nn.Sequential(nn.Conv2d(h1, h2, 3, stride=2, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(*[ConvResBlock(h2) for _ in range(max(1, int(depth)))])
        self.down2 = nn.Sequential(nn.Conv2d(h2, h3, 3, stride=2, padding=1), nn.SiLU())
        self.mid = nn.Sequential(*[ConvResBlock(h3) for _ in range(max(1, int(depth)))])
        self.up2 = nn.Sequential(nn.Conv2d(h3 + h2, h2, 3, padding=1), nn.SiLU(), ConvResBlock(h2))
        self.up1 = nn.Sequential(nn.Conv2d(h2 + h1, h1, 3, padding=1), nn.SiLU(), ConvResBlock(h1))
        self.head = nn.Sequential(nn.GroupNorm(min(8, h1), h1), nn.SiLU(), nn.Conv2d(h1, 16, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(self.stem(obs))
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        x = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, e1], dim=1))
        return self.res_scale.to(dtype=obs.dtype) * self.head(x)


class SpatialAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_in = self.norm(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ff(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class AttentionNullPredictor(nn.Module):
    def __init__(self, in_channels: int = 21, hidden: int = 160, depth: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, hidden, 3, padding=1), nn.SiLU())
        blocks = []
        for i in range(max(1, int(depth))):
            blocks.append(ConvResBlock(hidden))
            if i % 2 == 1:
                blocks.append(SpatialAttentionBlock(hidden, heads=4))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.GroupNorm(min(8, hidden), hidden), nn.SiLU(), nn.Conv2d(hidden, 16, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.res_scale.to(dtype=obs.dtype) * self.head(self.body(self.stem(obs)))


class FreqNullPredictor(nn.Module):
    def __init__(self, in_channels: int = 21, hidden: int = 192, depth: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 16, hidden, 3, padding=1),
            nn.SiLU(),
            *[ConvResBlock(hidden) for _ in range(int(depth))],
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 16, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z_base = obs[:, :16]
        local_low = F.avg_pool2d(z_base, kernel_size=3, stride=1, padding=1)
        high = z_base - local_low
        return self.res_scale.to(dtype=obs.dtype) * self.net(torch.cat([obs, high], dim=1))


class HybridFreqAttnUNetNullPredictor(nn.Module):
    """Frequency-augmented latent U-Net with bottleneck attention and skip fusion."""

    def __init__(self, in_channels: int = 49, hidden: int = 160, depth: int = 2) -> None:
        super().__init__()
        h1 = hidden
        h2 = hidden * 2
        h3 = hidden * 2
        d = max(1, int(depth))
        self.stem = nn.Sequential(nn.Conv2d(in_channels + 16, h1, 3, padding=1), nn.SiLU())
        self.enc1 = nn.Sequential(*[ConvResBlock(h1) for _ in range(d)])
        self.down1 = nn.Sequential(nn.Conv2d(h1, h2, 3, stride=2, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(*[ConvResBlock(h2) for _ in range(d)])
        self.down2 = nn.Sequential(nn.Conv2d(h2, h3, 3, stride=2, padding=1), nn.SiLU())
        mid_blocks: list[nn.Module] = []
        for i in range(max(2, d + 1)):
            mid_blocks.append(ConvResBlock(h3))
            if i == 1:
                mid_blocks.append(SpatialAttentionBlock(h3, heads=4))
        self.mid = nn.Sequential(*mid_blocks)
        branch_ch = max(16, h1 // 4)
        self.local_branches = nn.ModuleList(
            [nn.Conv2d(h1, branch_ch, 3, padding=r, dilation=r) for r in (1, 2, 3, 5)]
        )
        self.local_fuse = nn.Sequential(nn.SiLU(), nn.Conv2d(branch_ch * 4, h1, 1), ConvResBlock(h1))
        self.up2 = nn.Sequential(nn.Conv2d(h3, h2, 3, padding=1), nn.SiLU())
        self.fuse2 = nn.Sequential(nn.Conv2d(h2 + h2, h2, 1), ConvResBlock(h2), ConvResBlock(h2))
        self.up1 = nn.Sequential(nn.Conv2d(h2, h1, 3, padding=1), nn.SiLU())
        self.fuse1 = nn.Sequential(nn.Conv2d(h1 + h1 + h1, h1, 1), ConvResBlock(h1), ConvResBlock(h1))
        self.head = nn.Sequential(nn.GroupNorm(min(8, h1), h1), nn.SiLU(), nn.Conv2d(h1, 16, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z_base = obs[:, :16]
        high = z_base - F.avg_pool2d(z_base, kernel_size=3, stride=1, padding=1)
        x = torch.cat([obs, high.to(dtype=obs.dtype)], dim=1)
        e1 = self.enc1(self.stem(x))
        local = torch.cat([branch(e1) for branch in self.local_branches], dim=1)
        local = self.local_fuse(local)
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.fuse2(torch.cat([self.up2(d2), e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.fuse1(torch.cat([self.up1(d1), e1, local], dim=1))
        return self.res_scale.to(dtype=obs.dtype) * self.head(d1)


def build_direct_predictor(kind: str, in_channels: int, hidden: int, depth: int) -> nn.Module:
    kind = str(kind)
    if kind == "resstack":
        return DirectNullPredictor(in_channels=in_channels, hidden=hidden, depth=depth)
    if kind == "resunet":
        return UNetNullPredictor(in_channels=in_channels, hidden=hidden, depth=max(1, min(depth, 3)))
    if kind == "multiscale":
        return MultiScaleNullPredictor(in_channels=in_channels, hidden=hidden, depth=depth)
    if kind == "attn":
        return AttentionNullPredictor(in_channels=in_channels, hidden=hidden, depth=depth)
    if kind == "freq":
        return FreqNullPredictor(in_channels=in_channels, hidden=hidden, depth=depth)
    if kind == "hybrid":
        return HybridFreqAttnUNetNullPredictor(in_channels=in_channels, hidden=hidden, depth=max(1, min(depth, 4)))
    raise ValueError(f"unknown predictor={kind!r}")


class NullResidualDiffusion(nn.Module):
    def __init__(self, cond_hidden: int = 128, cond_depth: int = 4, unet_hidden: int = 96) -> None:
        super().__init__()
        self.cond = ConditionEncoder(hidden=cond_hidden, depth=cond_depth, out_channels=16)
        self.unet = UNetDenoiser(channels=16, hidden_dim=unet_hidden, use_cond=True)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        cond = self.cond(obs)
        return self.unet(z_t, t, cond=cond)


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Codex fixed high-frequency session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed Swin/channel AWGN12 C=4 high-frequency recovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--route", type=str, required=True, choices=["direct", "diffusion"])
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", action="store_false", dest="cache_decoded")
    p.add_argument("--cache_workers", type=int, default=12)
    p.add_argument("--prefetch_factor", type=int, default=4)

    p.add_argument("--sc_encoder_ckpt", type=str, required=True)
    p.add_argument("--sc_decoder_ckpt", type=str, required=True)
    p.add_argument("--cc_dir", type=str, required=True)
    p.add_argument("--unet_ckpt", type=str, required=True)
    p.add_argument("--compression_ratio", type=float, default=0.25)
    p.add_argument("--snr_db", type=float, default=12.0)
    p.add_argument("--baseline_psnr", type=float, default=22.419)

    p.add_argument("--predictor", type=str, default="resstack", choices=["resstack", "resunet", "multiscale", "attn", "freq", "hybrid"])
    p.add_argument("--direct_condition", type=str, default="compact", choices=["compact", "rich"])
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--lambda_res", type=float, default=0.1)
    p.add_argument("--img_mse_weight", type=float, default=0.8)
    p.add_argument("--img_charb_weight", type=float, default=0.2)
    p.add_argument("--charb_eps", type=float, default=1e-3)
    p.add_argument("--cond_hidden", type=int, default=128)
    p.add_argument("--cond_depth", type=int, default=4)
    p.add_argument("--unet_hidden", type=int, default=96)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    p.add_argument("--num_train_steps", type=int, default=1000)
    p.add_argument("--noise_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--min_snr_gamma", type=float, default=5.0)
    p.add_argument("--z0_clip", type=float, default=5.0)
    p.add_argument("--sample_steps", type=int, default=24)
    p.add_argument("--sample_t_start", type=int, default=180)
    p.add_argument("--sample_init", type=str, default="zero", choices=["zero", "noise"])

    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--stat_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260524)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints-codex/fixed_highfreq")
    p.add_argument("--log_file", type=str, default="checkpoints-codex/fixed_highfreq/train.log")
    return p.parse_args()


def make_schedule(num_steps: int, schedule: str, device: torch.device) -> torch.Tensor:
    if schedule == "linear":
        betas = torch.linspace(1e-4, 2e-2, int(num_steps), device=device)
        return torch.cumprod(1.0 - betas, dim=0)
    if schedule == "cosine":
        steps = torch.arange(int(num_steps) + 1, device=device, dtype=torch.float32)
        s = 0.008
        f = torch.cos(((steps / int(num_steps)) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bars = (f / f[0]).clamp_min(1e-8)
        betas = (1.0 - alpha_bars[1:] / alpha_bars[:-1]).clamp(1e-8, 0.999)
        return torch.cumprod(1.0 - betas, dim=0)
    raise ValueError(f"unknown noise_schedule={schedule!r}")


def norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean.to(device=x.device, dtype=x.dtype)) / std.to(device=x.device, dtype=x.dtype).clamp_min(1e-6)


def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std.to(device=x.device, dtype=x.dtype).clamp_min(1e-6) + mean.to(device=x.device, dtype=x.dtype)


def min_snr_weight(alpha_bar: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        return torch.ones_like(alpha_bar)
    snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-8)
    return torch.minimum(snr, torch.full_like(snr, float(gamma))) / snr.clamp_min(1e-8)


def charbonnier_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = x.float() - y.float()
    return torch.sqrt(diff.square() + float(eps) ** 2).mean()


def setup_loaders(args: argparse.Namespace, device: torch.device):
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
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if int(args.num_workers) > 0 else None),
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
    return train_ds, val_ds, train_loader, val_loader


@torch.no_grad()
def make_observation(
    *,
    system,
    imgs: torch.Tensor,
    a: torch.Tensor,
    snr_b: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    with make_autocast(imgs.device, amp_enabled, amp_dtype):
        z_sem = system.semantic_encoder(imgs).float()
    z_low = encode_with_a(z_sem, a)
    y_norm, y_raw, scale = power_normalize_awgn(z_low, snr_b, generator=generator)
    z_base = lift_with_decoder(system, y_raw.float(), amp_enabled, amp_dtype).float()
    z_at_raw = lift_with_at(y_raw.float(), a).float()
    z_norm_lift = lift_with_at(y_norm.float(), a).float()
    h, w = z_base.shape[-2:]
    scale_map = scale.float().log().view(-1, 1, 1, 1).expand(-1, 1, h, w)
    obs = torch.cat([z_base, z_at_raw, z_norm_lift, scale_map.to(dtype=z_base.dtype)], dim=1)
    direct_obs = torch.cat([z_base, y_raw.float(), scale_map.to(dtype=z_base.dtype)], dim=1)
    # The fixed learned channel decoder can already synthesize part of the
    # null-space content, so the high-frequency target is the remaining
    # null-space residual after the fixed noisy receiver base.
    z_null = null_project(z_sem - z_base, a)
    z_clean = lift_with_decoder(system, z_low.float(), amp_enabled, amp_dtype).float()
    return {
        "z_sem": z_sem,
        "z_base": z_base,
        "z_clean": z_clean,
        "z_null": z_null,
        "obs": obs,
        "direct_obs": direct_obs,
    }


@torch.no_grad()
def estimate_null_stats(system, loader, a, args, device, amp_enabled, amp_dtype) -> dict[str, torch.Tensor]:
    sum_ = torch.zeros(16, dtype=torch.float64)
    sumsq = torch.zeros(16, dtype=torch.float64)
    count = 0
    snr_template = torch.full((int(args.batch_size),), float(args.snr_db), device=device, dtype=torch.float32)
    for bi, batch in enumerate(loader):
        if int(args.stat_batches) > 0 and bi >= int(args.stat_batches):
            break
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=True)
        out = make_observation(
            system=system,
            imgs=imgs,
            a=a,
            snr_b=snr_template[: imgs.shape[0]],
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        z = out["z_null"].detach().double()
        sum_ += z.sum(dim=(0, 2, 3)).cpu()
        sumsq += (z * z).sum(dim=(0, 2, 3)).cpu()
        count += z.shape[0] * z.shape[2] * z.shape[3]
    mean = sum_ / max(1, count)
    var = (sumsq / max(1, count) - mean.square()).clamp_min(1e-8)
    return {
        "null_mean": mean.float().view(1, 16, 1, 1),
        "null_std": torch.sqrt(var).float().view(1, 16, 1, 1),
    }


@torch.no_grad()
def sample_diffusion(
    model: NullResidualDiffusion,
    obs: torch.Tensor,
    stats: dict[str, torch.Tensor],
    alpha_bars: torch.Tensor,
    a: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    device = obs.device
    n_total = int(alpha_bars.shape[0])
    t_start = max(0, min(int(args.sample_t_start), n_total - 1))
    steps = max(1, int(args.sample_steps))
    step_indices = torch.linspace(t_start, 0, steps, device=device).long()
    if str(args.sample_init) == "zero":
        z0 = norm(torch.zeros(obs.shape[0], 16, obs.shape[2], obs.shape[3], device=device), stats["null_mean"], stats["null_std"])
        alpha_start = alpha_bars[t_start].to(device=device, dtype=obs.dtype)
        z = torch.sqrt(alpha_start) * z0 + torch.sqrt(1.0 - alpha_start) * torch.randn_like(z0)
    else:
        z = torch.randn(obs.shape[0], 16, obs.shape[2], obs.shape[3], device=device, dtype=obs.dtype)

    for i, idx in enumerate(step_indices):
        alpha = alpha_bars[idx].to(device=device, dtype=z.dtype)
        alpha_prev = (
            alpha_bars[step_indices[i + 1]].to(device=device, dtype=z.dtype)
            if i + 1 < len(step_indices)
            else torch.tensor(1.0, device=device, dtype=z.dtype)
        )
        t = torch.full((z.shape[0],), float(idx.item()) / float(max(1, n_total - 1)), device=device, dtype=z.dtype)
        eps = model(z, t, obs)
        z0_norm = (z - torch.sqrt(1.0 - alpha) * eps) / torch.sqrt(alpha + 1e-8)
        if float(args.z0_clip) > 0:
            z0_norm = z0_norm.clamp(-float(args.z0_clip), float(args.z0_clip))
        z0_null = null_project(denorm(z0_norm.float(), stats["null_mean"], stats["null_std"]), a)
        z0_norm = norm(z0_null, stats["null_mean"], stats["null_std"]).to(dtype=z.dtype)
        z = torch.sqrt(alpha_prev) * z0_norm + torch.sqrt(1.0 - alpha_prev) * eps
    return null_project(denorm(z.float(), stats["null_mean"], stats["null_std"]), a)


@torch.no_grad()
def validate_direct(system, model, loader, a, args, device, amp_enabled, amp_dtype, generator) -> dict[str, float]:
    model.eval()
    vals = {k: [] for k in ("base", "clean", "recv", "full")}
    snr_template = torch.full((int(args.batch_size),), float(args.snr_db), device=device, dtype=torch.float32)
    cond_key = "obs" if str(args.direct_condition) == "rich" else "direct_obs"
    for bi, batch in enumerate(loader):
        if int(args.max_val_batches) > 0 and bi >= int(args.max_val_batches):
            break
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=True)
        out = make_observation(system=system, imgs=imgs, a=a, snr_b=snr_template[: imgs.shape[0]], amp_enabled=amp_enabled, amp_dtype=amp_dtype, generator=generator)
        pred_null = null_project(model(out[cond_key].to(dtype=next(model.parameters()).dtype)).float(), a)
        z_hat = out["z_base"] + pred_null
        x_hat = decode_latent(system, z_hat, amp_enabled, amp_dtype).float().clamp(0, 1)
        x_base = decode_latent(system, out["z_base"], amp_enabled, amp_dtype).float().clamp(0, 1)
        x_clean = decode_latent(system, out["z_clean"], amp_enabled, amp_dtype).float().clamp(0, 1)
        x_full = decode_latent(system, out["z_sem"], amp_enabled, amp_dtype).float().clamp(0, 1)
        vals["recv"].append(psnr_per_image(x_hat, imgs.float()))
        vals["base"].append(psnr_per_image(x_base, imgs.float()))
        vals["clean"].append(psnr_per_image(x_clean, imgs.float()))
        vals["full"].append(psnr_per_image(x_full, imgs.float()))
    return _metric_dict(vals, "direct", args)


@torch.no_grad()
def validate_diffusion(system, model, ema, loader, a, stats, alpha_bars, args, device, amp_enabled, amp_dtype, generator) -> dict[str, float]:
    model.eval()
    backup = ema.apply_to(model)
    vals = {k: [] for k in ("base", "clean", "recv", "full")}
    snr_template = torch.full((int(args.batch_size),), float(args.snr_db), device=device, dtype=torch.float32)
    for bi, batch in enumerate(loader):
        if int(args.max_val_batches) > 0 and bi >= int(args.max_val_batches):
            break
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=True)
        out = make_observation(system=system, imgs=imgs, a=a, snr_b=snr_template[: imgs.shape[0]], amp_enabled=amp_enabled, amp_dtype=amp_dtype, generator=generator)
        pred_null = sample_diffusion(model, out["obs"].to(dtype=next(model.parameters()).dtype), stats, alpha_bars, a, args)
        z_hat = out["z_base"] + pred_null
        x_hat = decode_latent(system, z_hat, amp_enabled, amp_dtype).float().clamp(0, 1)
        x_base = decode_latent(system, out["z_base"], amp_enabled, amp_dtype).float().clamp(0, 1)
        x_clean = decode_latent(system, out["z_clean"], amp_enabled, amp_dtype).float().clamp(0, 1)
        x_full = decode_latent(system, out["z_sem"], amp_enabled, amp_dtype).float().clamp(0, 1)
        vals["recv"].append(psnr_per_image(x_hat, imgs.float()))
        vals["base"].append(psnr_per_image(x_base, imgs.float()))
        vals["clean"].append(psnr_per_image(x_clean, imgs.float()))
        vals["full"].append(psnr_per_image(x_full, imgs.float()))
    model.load_state_dict(backup)
    return _metric_dict(vals, "diffusion", args)


def _avg(xs: list[torch.Tensor]) -> float:
    return float(torch.cat(xs).mean().item()) if xs else float("nan")


def _metric_dict(vals: dict[str, list[torch.Tensor]], route_name: str, args: argparse.Namespace) -> dict[str, float]:
    base = _avg(vals["base"])
    clean = _avg(vals["clean"])
    recv = _avg(vals["recv"])
    full = _avg(vals["full"])
    return {
        f"val_psnr_{route_name}": recv,
        "val_psnr_base": base,
        "val_psnr_clean": clean,
        "val_psnr_full": full,
        "val_gain_vs_base": recv - base,
        "val_gain_vs_baseline": recv - float(args.baseline_psnr),
    }


def save_checkpoint(path: str, model: nn.Module, args: argparse.Namespace, a: torch.Tensor, metrics: dict, epoch: int, ema: EMA | None = None, stats: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "stats": {k: v.detach().cpu() for k, v in (stats or {}).items()},
        "a_matrix": a.detach().cpu(),
        "aat": (a @ a.t()).detach().cpu(),
        "aat_error": semiorth_error(a),
        "route": str(args.route),
        "predictor": str(getattr(args, "predictor", "")),
        "fixed_swin": True,
        "fixed_channel_codec": True,
        "loss": "direct_image_mse_charbonnier_plus_null_residual_mse" if args.route == "direct" else "residual_conditional_diffusion_eps",
        "sampler": "projected_nullspace_ddim_zero_init" if args.route == "diffusion" else None,
        "snr_db": float(args.snr_db),
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(PROJECT_ROOT, args.save_dir)
    log_file = args.log_file if os.path.isabs(args.log_file) else os.path.join(PROJECT_ROOT, args.log_file)
    os.makedirs(save_dir, exist_ok=True)
    setup_log_file(log_file)
    seed_everything(int(args.seed))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}, route={args.route}")
    print("rule2: PSNR=mean(per-image PSNR), train/test SNR=12 dB, fixed Swin/channel codec, A A^T=I_4, power norm after A")
    train_ds, val_ds, train_loader, val_loader = setup_loaders(args, device)
    print(
        f"train={len(train_ds)} valid={len(val_ds)} input=[3,{args.crop_size},{args.crop_size}] "
        "cache_decoded=full images; train crop is dynamic in DIV2KDataset.__getitem__"
    )

    system, _unet_obj, tag = build_system_for_ratio(
        ratio=float(args.compression_ratio),
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
    a = channel_matrix(system, dtype=torch.float32).detach()
    aat_err = semiorth_error(a)
    if aat_err > 1e-5:
        raise RuntimeError(f"fixed channel encoder violates A A^T=I_4: error={aat_err:.3e}")
    print(f"channel_encoder_output=[4,16,16], tag={tag}, aat_error={aat_err:.3e}")

    snr_template = torch.full((int(args.batch_size),), float(args.snr_db), device=device, dtype=torch.float32)
    best = -1.0
    if args.route == "direct":
        direct_in_channels = 49 if str(args.direct_condition) == "rich" else 21
        model: nn.Module = build_direct_predictor(
            kind=str(args.predictor),
            in_channels=direct_in_channels,
            hidden=int(args.hidden),
            depth=int(args.depth),
        ).to(device)
        opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
        print(
            f"arch={model.__class__.__name__} predictor={args.predictor} hidden={args.hidden} depth={args.depth}; "
            f"loss=img({args.img_mse_weight}*mse+{args.img_charb_weight}*charb)+{args.lambda_res}*null_res_mse; "
            f"condition={args.direct_condition}, input_channels={direct_in_channels}, output=P_null(residual)"
        )
    else:
        model = NullResidualDiffusion(
            cond_hidden=int(args.cond_hidden),
            cond_depth=int(args.cond_depth),
            unet_hidden=int(args.unet_hidden),
        ).to(device)
        print("estimating null residual stats for diffusion target ...")
        stats = estimate_null_stats(system, train_loader, a, args, device, amp_enabled, amp_dtype)
        print(f"null_std=[{stats['null_std'].min().item():.4f},{stats['null_std'].max().item():.4f}]")
        alpha_bars = make_schedule(int(args.num_train_steps), str(args.noise_schedule), device)
        ema = EMA(model, decay=float(args.ema_decay))
        opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
        print(
            f"arch=NullResidualDiffusion cond_hidden={args.cond_hidden} cond_depth={args.cond_depth} "
            f"unet_hidden={args.unet_hidden}; sampler={args.sample_init}:{args.sample_steps}@{args.sample_t_start}"
        )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_meter = AverageMeter()
        img_meter = AverageMeter()
        res_meter = AverageMeter()
        for bi, batch in enumerate(train_loader):
            if int(args.max_train_batches) > 0 and bi >= int(args.max_train_batches):
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                out = make_observation(
                    system=system,
                    imgs=imgs,
                    a=a,
                    snr_b=snr_template[: imgs.shape[0]],
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            opt.zero_grad(set_to_none=True)
            if args.route == "direct":
                cond_key = "obs" if str(args.direct_condition) == "rich" else "direct_obs"
                with make_autocast(device, amp_enabled, amp_dtype):
                    pred_null = null_project(model(out[cond_key].to(dtype=next(model.parameters()).dtype)).float(), a)
                    z_hat = out["z_base"] + pred_null
                    x_hat = decode_latent(system, z_hat, amp_enabled, amp_dtype).float().clamp(0, 1)
                    loss_mse = F.mse_loss(x_hat.float(), imgs.float())
                    loss_charb = charbonnier_loss(x_hat, imgs, eps=float(args.charb_eps))
                    loss_img = float(args.img_mse_weight) * loss_mse + float(args.img_charb_weight) * loss_charb
                    loss_res = F.mse_loss(pred_null.float(), out["z_null"].float())
                    loss = loss_img + float(args.lambda_res) * loss_res
            else:
                z_null_norm = norm(out["z_null"], stats["null_mean"], stats["null_std"])
                bsz = imgs.shape[0]
                t_idx = torch.randint(0, int(args.num_train_steps), (bsz,), device=device, dtype=torch.long)
                alpha = alpha_bars[t_idx].view(-1, 1, 1, 1).to(dtype=z_null_norm.dtype)
                eps = torch.randn_like(z_null_norm)
                z_t = torch.sqrt(alpha) * z_null_norm + torch.sqrt(1.0 - alpha) * eps
                t = t_idx.to(dtype=z_null_norm.dtype) / float(max(1, int(args.num_train_steps) - 1))
                with make_autocast(device, amp_enabled, amp_dtype):
                    eps_pred = model(z_t, t, out["obs"].to(dtype=next(model.parameters()).dtype))
                    weight = min_snr_weight(alpha.float(), float(args.min_snr_gamma))
                    loss = (weight * (eps_pred.float() - eps.float()).pow(2)).mean()
            scaler.scale(loss).backward()
            if float(args.clip_grad_norm) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
            scaler.step(opt)
            scaler.update()
            if args.route == "diffusion":
                ema.update(model)
            loss_meter.update(float(loss.item()), imgs.shape[0])
            if args.route == "direct":
                img_meter.update(float(loss_img.item()), imgs.shape[0])
                res_meter.update(float(loss_res.item()), imgs.shape[0])

        metrics = {"train_loss": loss_meter.avg}
        if args.route == "direct":
            metrics.update({"train_loss_img": img_meter.avg, "train_loss_res": res_meter.avg})
        if epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(args.seed) + 1000)
            if args.route == "direct":
                val_metrics = validate_direct(system, model, val_loader, a, args, device, amp_enabled, amp_dtype, gen)
                score_key = "val_psnr_direct"
            else:
                val_metrics = validate_diffusion(system, model, ema, val_loader, a, stats, alpha_bars, args, device, amp_enabled, amp_dtype, gen)
                score_key = "val_psnr_diffusion"
            metrics.update(val_metrics)
            score = val_metrics[score_key]
            is_best = score > best
            if is_best:
                best = score
                save_checkpoint(
                    os.path.join(save_dir, f"fixed_{args.route}_awgn12_best.pth"),
                    model,
                    args,
                    a,
                    metrics,
                    epoch,
                    ema=(ema if args.route == "diffusion" else None),
                    stats=(stats if args.route == "diffusion" else None),
                )
            save_checkpoint(
                os.path.join(save_dir, f"fixed_{args.route}_awgn12_latest.pth"),
                model,
                args,
                a,
                metrics,
                epoch,
                ema=(ema if args.route == "diffusion" else None),
                stats=(stats if args.route == "diffusion" else None),
            )
            train_extra = (
                f" img={img_meter.avg:.6f} res={res_meter.avg:.6f}"
                if args.route == "direct"
                else ""
            )
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={loss_meter.avg:.6f}{train_extra} | "
                f"base={val_metrics['val_psnr_base']:.4f} clean={val_metrics['val_psnr_clean']:.4f} "
                f"recv={score:.4f} full={val_metrics['val_psnr_full']:.4f} "
                f"gain_recv_base={val_metrics['val_gain_vs_base']:+.4f} "
                f"gain_baseline={val_metrics['val_gain_vs_baseline']:+.4f} "
                f"aat_err={semiorth_error(a):.2e} "
                f"{'BEST' if is_best else ''}"
            )
        else:
            if args.route == "direct":
                print(
                    f"[epoch {epoch:03d}/{args.epochs}] loss={loss_meter.avg:.6f} "
                    f"img={img_meter.avg:.6f} res={res_meter.avg:.6f} aat_err={semiorth_error(a):.2e}"
                )
            else:
                print(f"[epoch {epoch:03d}/{args.epochs}] loss={loss_meter.avg:.6f} aat_err={semiorth_error(a):.2e}")
    print(f"best_psnr={best:.4f} target>{float(args.baseline_psnr) + 0.5:.4f}")


if __name__ == "__main__":
    main()
