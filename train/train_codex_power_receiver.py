#!/usr/bin/env python
"""Train a power-normalized AWGN12 C=4 receiver for rule2.md.

Frozen transmitter / image codec:
    x -> Swin semantic encoder -> z16 -> channel encoder A -> z4

Channel:
    x4 = power_norm(z4)
    y4 = x4 + AWGN(SNR=12 dB)

Learned receiver:
    y4, scale, snr -> clean low estimate -> full z16 estimate -> frozen decoder

The receiver is allowed to correct both AWGN corruption in the transmitted
4-channel subspace and the missing 12 latent channels caused by dimensionality
reduction.  Validation reports mean per-image PSNR.
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


class CachedLatentDataset(Dataset):
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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = value.device
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(1, half - 1)
        )
        args = value.float().view(-1, 1) * freqs.view(1, -1)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.to_film = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 4 * channels))
        nn.init.zeros_(self.to_film[-1].weight)
        nn.init.zeros_(self.to_film[-1].bias)

    @staticmethod
    def _apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1).to(dtype=x.dtype)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1).to(dtype=x.dtype)
        return x * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g1, b1, g2, b2 = self.to_film(cond).chunk(4, dim=1)
        h = self._apply_film(self.norm1(x), g1, b1)
        h = self.conv1(F.silu(h))
        h = self._apply_film(self.norm2(h), g2, b2)
        h = self.conv2(F.silu(h))
        return x + h


class ConditionalConvNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden: int, depth: int, cond_dim: int) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMResBlock(hidden, cond_dim) for _ in range(int(depth))])
        self.out_norm = nn.GroupNorm(min(8, hidden), hidden)
        self.out_conv = nn.Conv2d(hidden, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_conv(F.silu(self.out_norm(h)))


class PowerAwareReceiver(nn.Module):
    def __init__(
        self,
        *,
        low_channels: int = 4,
        latent_channels: int = 16,
        low_hidden: int = 96,
        low_depth: int = 5,
        ref_hidden: int = 160,
        ref_depth: int = 8,
        emb_dim: int = 64,
    ) -> None:
        super().__init__()
        cond_dim = emb_dim + 1
        self.snr_embed = SinusoidalEmbedding(emb_dim)
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, ref_hidden),
            nn.SiLU(),
            nn.Linear(ref_hidden, ref_hidden),
        )
        self.low_cond = nn.Linear(ref_hidden, low_hidden * 2)
        self.ref_cond = nn.Linear(ref_hidden, ref_hidden * 2)
        self.low_net = ConditionalConvNet(
            in_channels=low_channels * 2 + 1,
            out_channels=low_channels,
            hidden=low_hidden,
            depth=low_depth,
            cond_dim=low_hidden * 2,
        )
        self.ref_net = ConditionalConvNet(
            in_channels=latent_channels + low_channels * 2 + 1,
            out_channels=latent_channels,
            hidden=ref_hidden,
            depth=ref_depth,
            cond_dim=ref_hidden * 2,
        )
        self.low_noise_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.ref_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    @staticmethod
    def _scale_map(scale: torch.Tensor, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        return scale.float().log().view(-1, 1, 1, 1).expand(-1, 1, height, width).to(dtype=dtype)

    def condition(self, snr_db: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scale_log = scale.float().clamp_min(1e-12).log().view(-1, 1)
        emb = self.snr_embed(snr_db)
        return self.cond(torch.cat([emb, scale_log], dim=1))

    def denoise_low(self, y_norm: torch.Tensor, y_raw: torch.Tensor, scale: torch.Tensor, snr_db: torch.Tensor):
        cond = self.condition(snr_db, scale)
        scale_map = self._scale_map(scale, y_raw.shape[-2], y_raw.shape[-1], y_raw.dtype)
        inp = torch.cat([y_norm, y_raw, scale_map], dim=1)
        noise = self.low_net(inp, self.low_cond(cond).to(dtype=y_raw.dtype))
        z_low_hat = y_raw - self.low_noise_scale.to(dtype=y_raw.dtype) * noise
        return z_low_hat, cond

    def refine(
        self,
        z_init: torch.Tensor,
        z_low_hat: torch.Tensor,
        y_raw: torch.Tensor,
        scale: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        scale_map = self._scale_map(scale, y_raw.shape[-2], y_raw.shape[-1], y_raw.dtype)
        inp = torch.cat([z_init, z_low_hat, y_raw, scale_map], dim=1)
        residual = self.ref_net(inp, self.ref_cond(cond).to(dtype=z_init.dtype))
        return z_init + self.ref_scale.to(dtype=z_init.dtype) * residual


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Codex power receiver session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train rule2 power-normalized C=4 AWGN12 receiver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
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
    p.add_argument("--baseline_psnr", type=float, default=22.419)

    p.add_argument("--low_hidden", type=int, default=96)
    p.add_argument("--low_depth", type=int, default=5)
    p.add_argument("--ref_hidden", type=int, default=160)
    p.add_argument("--ref_depth", type=int, default=8)
    p.add_argument("--pretrain_low_epochs", type=int, default=80)
    p.add_argument("--joint_epochs", type=int, default=260)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    p.add_argument("--lambda_low", type=float, default=1.0)
    p.add_argument("--lambda_latent", type=float, default=1.0)
    p.add_argument("--lambda_image", type=float, default=2.0)
    p.add_argument("--lambda_channel", type=float, default=0.5)
    p.add_argument("--lambda_dist", type=float, default=0.2)
    p.add_argument("--train_channel_codec", action="store_true",
                   help="Jointly train the 16->4 channel encoder and 4->16 channel decoder at the matched SNR.")
    p.add_argument("--channel_codec_lr", type=float, default=1e-4)
    p.add_argument("--train_decoder", action="store_true",
                   help="Jointly adapt the semantic decoder to the receiver latent distribution.")
    p.add_argument("--decoder_lr", type=float, default=5e-5)
    p.add_argument("--image_refiner", action="store_true",
                   help="Train an image-domain residual refiner after latent decoding.")
    p.add_argument("--image_refiner_hidden", type=int, default=96)
    p.add_argument("--image_refiner_depth", type=int, default=8)
    p.add_argument("--image_refiner_lr", type=float, default=1e-4)
    p.add_argument("--lambda_pre_ref_image", type=float, default=0.2,
                   help="Auxiliary image loss before image refiner when --image_refiner is enabled.")
    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260523)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints-codex/awgn12_power_receiver")
    p.add_argument("--log_file", type=str, default="checkpoints-codex/awgn12_power_receiver/train.log")
    return p.parse_args()


def make_autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device.type, enabled=enabled, dtype=dtype)
    return torch.autocast("cpu", enabled=False)


def power_normalize_awgn(
    z_low: torch.Tensor,
    snr_db: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, channels, _h, _w = z_low.shape
    if channels % 2 != 0:
        raise ValueError("channel encoder output channels must be even")
    z_use = z_low.float()
    z_complex = torch.complex(z_use[:, 0::2], z_use[:, 1::2])
    dims = tuple(range(1, z_complex.ndim))
    power = (z_complex.real.square() + z_complex.imag.square()).mean(dim=dims)
    scale = torch.sqrt(power.clamp_min(1e-12))
    x_norm = z_complex / scale.view(bsz, 1, 1, 1).to(dtype=z_complex.dtype)

    snr_linear = torch.pow(torch.tensor(10.0, device=z_low.device), snr_db.float().view(bsz) / 10.0)
    sigma = torch.sqrt(1.0 / (2.0 * snr_linear)).view(bsz, 1, 1, 1)
    noise_r = torch.randn(x_norm.real.shape, device=z_low.device, dtype=x_norm.real.dtype, generator=generator) * sigma
    noise_i = torch.randn(x_norm.imag.shape, device=z_low.device, dtype=x_norm.imag.dtype, generator=generator) * sigma
    y_norm_c = x_norm + torch.complex(noise_r, noise_i)
    y_raw_c = y_norm_c * scale.view(bsz, 1, 1, 1).to(dtype=y_norm_c.dtype)

    y_norm = torch.empty_like(z_use)
    y_norm[:, 0::2] = y_norm_c.real
    y_norm[:, 1::2] = y_norm_c.imag
    y_raw = torch.empty_like(z_use)
    y_raw[:, 0::2] = y_raw_c.real
    y_raw[:, 1::2] = y_raw_c.imag
    return y_norm.to(dtype=z_low.dtype), y_raw.to(dtype=z_low.dtype), scale


def distribution_loss(z_hat: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    dims = (0, 2, 3)
    mean_hat = z_hat.float().mean(dim=dims)
    mean_tgt = z_target.float().mean(dim=dims)
    std_hat = z_hat.float().std(dim=dims, unbiased=False)
    std_tgt = z_target.float().std(dim=dims, unbiased=False)
    return F.mse_loss(mean_hat, mean_tgt) + F.mse_loss(std_hat, std_tgt)


def low_project(z: torch.Tensor, a_matrix: torch.Tensor) -> torch.Tensor:
    a = a_matrix.to(device=z.device, dtype=z.dtype)
    return torch.einsum("oc,bchw->bohw", a, z)


def encode_low(system, z: torch.Tensor, amp_enabled: bool, amp_dtype: torch.dtype):
    device = z.device
    ch_dtype = next(system.channel_encoder.parameters()).dtype
    with make_autocast(device, amp_enabled, amp_dtype):
        return system.channel_encoder(z.to(ch_dtype)).float()


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
    return train_ds, val_ds, train_loader, val_loader


@torch.no_grad()
def cache_latents(system, loader, device, amp_enabled, amp_dtype, max_batches: int, split_name: str):
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
        raise RuntimeError(f"no {split_name} samples cached")
    return CachedLatentDataset(torch.cat(z_sems), torch.cat(z_lows), torch.cat(imgs_all))


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


@dataclass
class Paths:
    best: str
    latest: str


def make_paths(save_dir: str) -> Paths:
    os.makedirs(save_dir, exist_ok=True)
    return Paths(
        best=os.path.join(save_dir, "codex_power_receiver_awgn12_best.pth"),
        latest=os.path.join(save_dir, "codex_power_receiver_awgn12_latest.pth"),
    )


def save_checkpoint(path: str, model: PowerAwareReceiver, args: argparse.Namespace, a_matrix: torch.Tensor, epoch: int, metrics: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
            "state_dict": model.state_dict(),
            "receiver_type": "power_aware_low_and_full_latent",
            "low_channels": 4,
            "latent_channels": 16,
            "a_matrix": a_matrix.detach().cpu(),
            "snr_db": float(args.snr_db),
            "baseline_psnr": float(args.baseline_psnr),
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
    }
    torch.save(payload, path)


def save_checkpoint_with_decoder(
    path: str,
    model: PowerAwareReceiver,
    system,
    args: argparse.Namespace,
    a_matrix: torch.Tensor,
    epoch: int,
    metrics: dict,
    image_refiner: ImageResidualRefiner | None = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "receiver_type": "power_aware_low_and_full_latent",
        "low_channels": 4,
        "latent_channels": 16,
        "a_matrix": a_matrix.detach().cpu(),
        "snr_db": float(args.snr_db),
        "baseline_psnr": float(args.baseline_psnr),
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
    }
    if bool(getattr(args, "train_decoder", False)):
        payload["semantic_decoder_state_dict"] = system.semantic_decoder.state_dict()
        payload["semantic_decoder_ckpt"] = args.sc_decoder_ckpt
    if bool(getattr(args, "train_channel_codec", False)):
        payload["channel_encoder_state_dict"] = system.channel_encoder.state_dict()
        payload["channel_decoder_state_dict"] = system.channel_decoder.state_dict()
        payload["channel_codec_ckpt_dir"] = args.cc_dir
        try:
            payload["a_matrix"] = system._channel_encoder_matrix().detach().cpu()
        except Exception:
            pass
    if image_refiner is not None:
        payload["image_refiner_state_dict"] = image_refiner.state_dict()
        payload["image_refiner_hidden"] = int(getattr(args, "image_refiner_hidden", 96))
        payload["image_refiner_depth"] = int(getattr(args, "image_refiner_depth", 8))
    torch.save(payload, path)


def forward_receiver(
    *,
    system,
    model: PowerAwareReceiver,
    z_low: torch.Tensor,
    snr_b: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator | None = None,
):
    y_norm, y_raw, scale = power_normalize_awgn(z_low, snr_b, generator=generator)
    z_low_hat, cond = model.denoise_low(y_norm.float(), y_raw.float(), scale, snr_b)
    z_init = lift_low(system, z_low_hat, amp_enabled, amp_dtype)
    z_hat = model.refine(z_init.float(), z_low_hat.float(), y_raw.float(), scale, cond)
    z_raw = lift_low(system, y_raw, amp_enabled, amp_dtype)
    return {
        "y_norm": y_norm,
        "y_raw": y_raw,
        "scale": scale,
        "z_low_hat": z_low_hat,
        "z_init": z_init,
        "z_hat": z_hat,
        "z_raw": z_raw,
    }


class ImageRefinerBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ImageResidualRefiner(nn.Module):
    def __init__(self, in_channels: int = 9, hidden: int = 96, depth: int = 8) -> None:
        super().__init__()
        feat_channels = int(in_channels) * 3
        layers: list[nn.Module] = []
        for _ in range(int(depth)):
            layers.append(ImageRefinerBlock(hidden))
        self.head = nn.Sequential(
            nn.Conv2d(feat_channels, hidden, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*layers)
        up_hidden = max(32, hidden // 2)
        self.tail = nn.Sequential(
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, up_hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(up_hidden, 3, 3, padding=1),
        )
        high_hidden = max(24, hidden // 4)
        self.high_head = nn.Conv2d(feat_channels, high_hidden, 3, padding=1)
        self.high_blocks = nn.Sequential(
            ImageRefinerBlock(high_hidden),
            ImageRefinerBlock(high_hidden),
        )
        self.high_tail = nn.Sequential(
            nn.GroupNorm(min(8, high_hidden), high_hidden),
            nn.SiLU(),
            nn.Conv2d(high_hidden, 3, 3, padding=1),
        )
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)
        nn.init.zeros_(self.high_tail[-1].weight)
        nn.init.zeros_(self.high_tail[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x_recv: torch.Tensor, x_raw: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        base = torch.cat([x_recv, x_raw, x_low], dim=1)
        low = F.avg_pool2d(base, kernel_size=5, stride=1, padding=2, count_include_pad=False)
        high = base - low
        inp = torch.cat([base, low, high], dim=1)
        residual_low = self.tail(self.blocks(self.head(inp)))
        residual_high = self.high_tail(self.high_blocks(self.high_head(inp)))
        residual = residual_low + residual_high
        return x_recv + self.res_scale.to(dtype=x_recv.dtype) * residual


@torch.no_grad()
def validate(
    *,
    system,
    model: PowerAwareReceiver,
    image_refiner: ImageResidualRefiner | None,
    loader: DataLoader,
    args: argparse.Namespace,
    a_matrix: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator,
) -> dict:
    model.eval()
    if image_refiner is not None:
        image_refiner.eval()
    system.channel_encoder.eval()
    system.channel_decoder.eval()
    system.semantic_decoder.eval()
    raw_vals: list[torch.Tensor] = []
    low_vals: list[torch.Tensor] = []
    pred_vals: list[torch.Tensor] = []
    clean_low_vals: list[torch.Tensor] = []
    low_m = AverageMeter()
    latent_m = AverageMeter()
    channel_m = AverageMeter()
    dist_m = AverageMeter()
    snr_template = torch.full((args.batch_size,), float(args.snr_db), device=device, dtype=torch.float32)
    for bi, batch in enumerate(loader):
        if args.max_val_batches > 0 and bi >= args.max_val_batches:
            break
        z_sem, z_low, imgs = batch
        z_sem = z_sem.to(device, non_blocking=True)
        z_low = z_low.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)
        if bool(getattr(args, "train_channel_codec", False)):
            z_low = encode_low(system, z_sem, amp_enabled, amp_dtype)
        snr_b = snr_template[: z_low.shape[0]]
        out = forward_receiver(
            system=system,
            model=model,
            z_low=z_low,
            snr_b=snr_b,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            generator=generator,
        )
        z_clean_low = lift_low(system, z_low, amp_enabled, amp_dtype)
        x_raw = decode_latent(system, out["z_raw"], amp_enabled, amp_dtype).float().clamp(0, 1)
        x_low = decode_latent(system, out["z_init"], amp_enabled, amp_dtype).float().clamp(0, 1)
        x_hat_pre = decode_latent(system, out["z_hat"], amp_enabled, amp_dtype).float()
        if image_refiner is not None:
            x_hat = image_refiner(
                x_hat_pre.clamp(0, 1),
                x_raw.float().clamp(0, 1),
                x_low.float().clamp(0, 1),
            ).float().clamp(0, 1)
        else:
            x_hat = x_hat_pre.float().clamp(0, 1)
        x_clean_low = decode_latent(system, z_clean_low, amp_enabled, amp_dtype).float().clamp(0, 1)

        low_loss = F.mse_loss(out["z_low_hat"].float(), z_low.float())
        latent_loss = F.mse_loss(out["z_hat"].float(), z_sem.float())
        if bool(getattr(args, "train_channel_codec", False)):
            z_relow = encode_low(system, out["z_hat"].float(), amp_enabled, amp_dtype)
        else:
            z_relow = low_project(out["z_hat"].float(), a_matrix)
        channel_loss = F.mse_loss(z_relow.float(), z_low.float())
        dist = distribution_loss(out["z_hat"].float(), z_sem.float())
        n = int(z_low.shape[0])
        low_m.update(float(low_loss.item()), n)
        latent_m.update(float(latent_loss.item()), n)
        channel_m.update(float(channel_loss.item()), n)
        dist_m.update(float(dist.item()), n)
        raw_vals.append(psnr_per_image(x_raw, imgs.float()))
        low_vals.append(psnr_per_image(x_low, imgs.float()))
        pred_vals.append(psnr_per_image(x_hat, imgs.float()))
        clean_low_vals.append(psnr_per_image(x_clean_low, imgs.float()))

    def avg(items: list[torch.Tensor]) -> float:
        return float(torch.cat(items).mean().item()) if items else float("nan")

    pred = avg(pred_vals)
    raw = avg(raw_vals)
    low = avg(low_vals)
    clean_low = avg(clean_low_vals)
    return {
        "val_low_mse": low_m.avg,
        "val_latent_mse": latent_m.avg,
        "val_channel_mse": channel_m.avg,
        "val_dist_mse": dist_m.avg,
        "val_psnr_raw_channel_decoder": raw,
        "val_psnr_clean_low_oracle": clean_low,
        "val_psnr_low_denoised": low,
        "val_psnr_receiver": pred,
        "val_gain_vs_raw": pred - raw,
        "val_gain_vs_baseline": pred - float(args.baseline_psnr),
    }


def train_epoch(
    *,
    system,
    model: PowerAwareReceiver,
    image_refiner: ImageResidualRefiner | None,
    loader: DataLoader,
    opt: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    a_matrix: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    pretrain_low_only: bool,
) -> dict:
    model.train()
    if image_refiner is not None:
        image_refiner.train()
    if bool(getattr(args, "train_channel_codec", False)):
        system.channel_encoder.train()
        system.channel_decoder.train()
    else:
        system.channel_encoder.eval()
        system.channel_decoder.eval()
    if bool(getattr(args, "train_decoder", False)) and not pretrain_low_only:
        system.semantic_decoder.train()
    else:
        system.semantic_decoder.eval()
    meters = {name: AverageMeter() for name in ("loss", "low", "latent", "image", "channel", "dist")}
    snr_template = torch.full((args.batch_size,), float(args.snr_db), device=device, dtype=torch.float32)
    for bi, batch in enumerate(loader):
        if args.max_train_batches > 0 and bi >= args.max_train_batches:
            break
        z_sem, z_low, imgs = batch
        z_sem = z_sem.to(device, non_blocking=True)
        z_low = z_low.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)
        if bool(getattr(args, "train_channel_codec", False)):
            z_low = encode_low(system, z_sem, amp_enabled, amp_dtype)
        snr_b = snr_template[: z_low.shape[0]]

        opt.zero_grad(set_to_none=True)
        with make_autocast(device, amp_enabled, amp_dtype):
            out = forward_receiver(
                system=system,
                model=model,
                z_low=z_low,
                snr_b=snr_b,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            loss_low = F.mse_loss(out["z_low_hat"].float(), z_low.float())
            loss_latent = F.mse_loss(out["z_hat"].float(), z_sem.float())
            x_hat_pre = decode_latent(system, out["z_hat"], amp_enabled, amp_dtype)
            if image_refiner is not None:
                with torch.no_grad():
                    x_raw = decode_latent(system, out["z_raw"], amp_enabled, amp_dtype).float()
                    x_low = decode_latent(system, out["z_init"], amp_enabled, amp_dtype).float()
                x_hat = image_refiner(
                    x_hat_pre.float().clamp(0, 1),
                    x_raw.clamp(0, 1),
                    x_low.clamp(0, 1),
                )
                loss_image = F.mse_loss(x_hat.float(), imgs.float())
                loss_pre_ref = F.mse_loss(x_hat_pre.float(), imgs.float())
            else:
                loss_image = F.mse_loss(x_hat_pre.float(), imgs.float())
                loss_pre_ref = loss_image.new_zeros(())
            if bool(getattr(args, "train_channel_codec", False)):
                z_relow = encode_low(system, out["z_hat"].float(), amp_enabled, amp_dtype)
            else:
                z_relow = low_project(out["z_hat"].float(), a_matrix)
            loss_channel = F.mse_loss(z_relow.float(), z_low.float())
            loss_dist = distribution_loss(out["z_hat"].float(), z_sem.float())
            if pretrain_low_only:
                loss = float(args.lambda_low) * loss_low
            else:
                loss = (
                    float(args.lambda_low) * loss_low
                    + float(args.lambda_latent) * loss_latent
                    + float(args.lambda_image) * loss_image
                    + float(args.lambda_pre_ref_image) * loss_pre_ref
                    + float(args.lambda_channel) * loss_channel
                    + float(args.lambda_dist) * loss_dist
                )
        scaler.scale(loss).backward()
        if args.clip_grad_norm > 0:
            scaler.unscale_(opt)
            params = [
                p
                for group in opt.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(params, float(args.clip_grad_norm))
        scaler.step(opt)
        scaler.update()

        n = int(z_low.shape[0])
        meters["loss"].update(float(loss.item()), n)
        meters["low"].update(float(loss_low.item()), n)
        meters["latent"].update(float(loss_latent.item()), n)
        meters["image"].update(float(loss_image.item()), n)
        meters["channel"].update(float(loss_channel.item()), n)
        meters["dist"].update(float(loss_dist.item()), n)
    return {f"train_{k}": v.avg for k, v in meters.items()}


def main() -> None:
    args = parse_args()
    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(PROJECT_ROOT, args.save_dir)
    log_file = args.log_file if os.path.isabs(args.log_file) else os.path.join(PROJECT_ROOT, args.log_file)
    os.makedirs(save_dir, exist_ok=True)
    setup_log_file(log_file)
    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False
    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}")
    print(f"save_dir={save_dir}")
    print(
        "rule2: PSNR=mean(per-image PSNR), train/test SNR=12 dB, "
        "power normalization after channel encoder"
    )

    train_img_ds, val_img_ds, train_img_loader, val_img_loader = setup_loaders(args, device)
    print(f"train={len(train_img_ds)} valid={len(val_img_ds)} input=[3,{args.crop_size},{args.crop_size}]")

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
    if bool(args.train_channel_codec):
        for p in system.channel_encoder.parameters():
            p.requires_grad_(True)
        for p in system.channel_decoder.parameters():
            p.requires_grad_(True)
        system.channel_encoder.train()
        system.channel_decoder.train()
    if bool(args.train_decoder):
        for p in system.semantic_decoder.parameters():
            p.requires_grad_(True)
        system.semantic_decoder.train()
    a_matrix = system._channel_encoder_matrix().detach().cpu().float()
    print(f"channel_encoder_output=[4,16,16], actual_matrix_shape={tuple(a_matrix.shape)}, tag={tag}")

    if not args.precache_latents:
        raise RuntimeError("This script requires --precache_latents for stable GPU utilization.")
    print("Pre-caching frozen Swin/channel latents ...")
    train_cached = cache_latents(system, train_img_loader, device, amp_enabled, amp_dtype, int(args.max_train_batches), "train")
    val_cached = cache_latents(system, val_img_loader, device, amp_enabled, amp_dtype, int(args.max_val_batches), "valid")
    train_loader, val_loader = make_cached_loaders(args, train_cached, val_cached, device)
    print(f"cached train={len(train_cached)} valid={len(val_cached)}")

    model = PowerAwareReceiver(
        low_channels=int(a_matrix.shape[0]),
        latent_channels=16,
        low_hidden=int(args.low_hidden),
        low_depth=int(args.low_depth),
        ref_hidden=int(args.ref_hidden),
        ref_depth=int(args.ref_depth),
    ).to(device)
    image_refiner = None
    if bool(args.image_refiner):
        image_refiner = ImageResidualRefiner(
            in_channels=9,
            hidden=int(args.image_refiner_hidden),
            depth=int(args.image_refiner_depth),
        ).to(device)
    opt_groups = [{"params": model.parameters(), "lr": float(args.lr)}]
    if bool(args.train_channel_codec):
        opt_groups.append({
            "params": list(system.channel_encoder.parameters()) + list(system.channel_decoder.parameters()),
            "lr": float(args.channel_codec_lr),
        })
    if bool(args.train_decoder):
        opt_groups.append({
            "params": system.semantic_decoder.parameters(),
            "lr": float(args.decoder_lr),
        })
    if image_refiner is not None:
        opt_groups.append({
            "params": image_refiner.parameters(),
            "lr": float(args.image_refiner_lr),
        })
    opt = optim.AdamW(opt_groups, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    paths = make_paths(save_dir)
    best_psnr = -1.0
    total_epochs = int(args.pretrain_low_epochs) + int(args.joint_epochs)
    print(
        f"arch=PowerAwareReceiver low_h={args.low_hidden} low_d={args.low_depth} "
        f"ref_h={args.ref_hidden} ref_d={args.ref_depth} epochs={total_epochs} "
        f"train_channel_codec={bool(args.train_channel_codec)} "
        f"channel_codec_lr={float(args.channel_codec_lr):g} "
        f"train_decoder={bool(args.train_decoder)} decoder_lr={float(args.decoder_lr):g} "
        f"image_refiner={bool(args.image_refiner)}"
    )

    for epoch in range(1, total_epochs + 1):
        pretrain = epoch <= int(args.pretrain_low_epochs)
        metrics = train_epoch(
            system=system,
            model=model,
            image_refiner=image_refiner,
            loader=train_loader,
            opt=opt,
            scaler=scaler,
            args=args,
            a_matrix=a_matrix,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            pretrain_low_only=pretrain,
        )
        phase = "low" if pretrain else "joint"
        if epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == total_epochs:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(args.seed) + 1000 + epoch)
            val_metrics = validate(
                system=system,
                model=model,
                image_refiner=image_refiner,
                loader=val_loader,
                args=args,
                a_matrix=a_matrix,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                generator=gen,
            )
            metrics.update(val_metrics)
            is_best = val_metrics["val_psnr_receiver"] > best_psnr
            if is_best:
                best_psnr = val_metrics["val_psnr_receiver"]
                save_checkpoint_with_decoder(paths.best, model, system, args, a_matrix, epoch, metrics, image_refiner)
            save_checkpoint_with_decoder(paths.latest, model, system, args, a_matrix, epoch, metrics, image_refiner)
            print(
                f"[{phase} {epoch:03d}/{total_epochs}] "
                f"loss={metrics['train_loss']:.6f} low={metrics['train_low']:.6f} "
                f"lat={metrics['train_latent']:.6f} img={metrics['train_image']:.6f} "
                f"ch={metrics['train_channel']:.6f} dist={metrics['train_dist']:.6f} | "
                f"raw={val_metrics['val_psnr_raw_channel_decoder']:.4f} "
                f"clean_low={val_metrics['val_psnr_clean_low_oracle']:.4f} "
                f"low={val_metrics['val_psnr_low_denoised']:.4f} "
                f"recv={val_metrics['val_psnr_receiver']:.4f} "
                f"gain_raw={val_metrics['val_gain_vs_raw']:+.4f} "
                f"gain_base={val_metrics['val_gain_vs_baseline']:+.4f} "
                f"{'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[{phase} {epoch:03d}/{total_epochs}] "
                f"loss={metrics['train_loss']:.6f} low={metrics['train_low']:.6f} "
                f"lat={metrics['train_latent']:.6f} img={metrics['train_image']:.6f} "
                f"ch={metrics['train_channel']:.6f} dist={metrics['train_dist']:.6f}"
            )
    print(f"best_psnr={best_psnr:.4f} target>{float(args.baseline_psnr) + 0.5:.4f} ckpt={paths.best}")


if __name__ == "__main__":
    main()
