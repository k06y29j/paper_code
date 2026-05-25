#!/usr/bin/env python
"""Train an AWGN12 C=4 receiver with a hard semi-orthogonal channel matrix.

Protocol:
  * frozen Swin semantic encoder and frozen semantic decoder
  * optional trainable 16->4 channel matrix A with hard constraint A A^T = I_4
  * A^T or frozen learned low-subspace lift, no learned low denoiser
  * receiver predicts only the missing/null-space latent residual
  * loss is directly aligned with mean per-image PSNR
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


class CachedSemanticDataset(Dataset):
    def __init__(self, z_sem: torch.Tensor, imgs: torch.Tensor) -> None:
        self.z_sem = z_sem
        self.imgs = imgs

    def __len__(self) -> int:
        return int(self.z_sem.shape[0])

    def __getitem__(self, idx):
        return self.z_sem[idx], self.imgs[idx]


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 64, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("embedding dim must be even")
        self.dim = int(dim)
        self.max_period = float(max_period)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=value.device, dtype=torch.float32)
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
    def _film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1).to(dtype=x.dtype)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1).to(dtype=x.dtype)
        return x * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g1, b1, g2, b2 = self.to_film(cond).chunk(4, dim=1)
        h = self.conv1(F.silu(self._film(self.norm1(x), g1, b1)))
        h = self.conv2(F.silu(self._film(self.norm2(h), g2, b2)))
        return x + h


class NullspaceHighFreqReceiver(nn.Module):
    def __init__(self, latent_channels: int = 16, hidden: int = 224, depth: int = 12, emb_dim: int = 64) -> None:
        super().__init__()
        cond_dim = emb_dim + 1
        self.snr_embed = SinusoidalEmbedding(emb_dim)
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden * 2),
        )
        self.in_conv = nn.Conv2d(latent_channels * 3 + 1, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMResBlock(hidden, hidden * 2) for _ in range(int(depth))])
        self.out_norm = nn.GroupNorm(min(8, hidden), hidden)
        self.out_conv = nn.Conv2d(hidden, latent_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    @staticmethod
    def _scale_map(scale: torch.Tensor, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        return scale.float().log().view(-1, 1, 1, 1).expand(-1, 1, height, width).to(dtype=dtype)

    def forward(
        self,
        z_base: torch.Tensor,
        z_norm_lift: torch.Tensor,
        z_at_lift: torch.Tensor,
        scale: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.cond(torch.cat([self.snr_embed(snr_db), scale.float().log().view(-1, 1)], dim=1))
        scale_map = self._scale_map(scale, z_base.shape[-2], z_base.shape[-1], z_base.dtype)
        h = self.in_conv(torch.cat([z_base, z_norm_lift, z_at_lift, scale_map], dim=1))
        cond = cond.to(dtype=h.dtype)
        for block in self.blocks:
            h = block(h, cond)
        residual = self.out_conv(F.silu(self.out_norm(h)))
        return self.res_scale.to(dtype=z_base.dtype) * residual


@dataclass
class Paths:
    best: str
    latest: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train semi-orthogonal AWGN12 C=4 high-frequency receiver",
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
    p.add_argument("--precache_semantics", action="store_true",
                   help="Pre-cache cropped Swin latents. Off by default because it fixes train crops.")
    p.add_argument("--semantic_cache_repeats", type=int, default=1,
                   help="Only used with --precache_semantics for smoke tests.")

    p.add_argument("--sc_encoder_ckpt", type=str, required=True)
    p.add_argument("--sc_decoder_ckpt", type=str, required=True)
    p.add_argument("--cc_dir", type=str, required=True)
    p.add_argument("--unet_ckpt", type=str, required=True)
    p.add_argument("--compression_ratio", type=float, default=0.25)
    p.add_argument("--snr_db", type=float, default=12.0)
    p.add_argument("--baseline_psnr", type=float, default=22.419)

    p.add_argument("--hidden", type=int, default=224)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--a_lr", type=float, default=5e-5)
    p.add_argument("--train_semantic_encoder", action="store_true",
                   help="Fine-tune the Swin semantic encoder with the PSNR objective; semantic decoder stays frozen.")
    p.add_argument("--semantic_encoder_lr", type=float, default=1e-5)
    p.add_argument("--lift_mode", type=str, default="learned", choices=["learned", "at"],
                   help="Use the whitened learned channel decoder or strict A^T as low-subspace lift.")
    p.add_argument("--train_channel_decoder", action="store_true",
                   help="Train channel decoder lift with the PSNR objective. Off by default.")
    p.add_argument("--channel_decoder_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260524)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints-codex/orth_highfreq_awgn12")
    p.add_argument("--log_file", type=str, default="checkpoints-codex/orth_highfreq_awgn12/train.log")
    return p.parse_args()


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Orthogonal high-frequency receiver @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def make_autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device.type, enabled=enabled, dtype=dtype)
    return torch.autocast("cpu", enabled=False)


def make_paths(save_dir: str) -> Paths:
    os.makedirs(save_dir, exist_ok=True)
    return Paths(
        best=os.path.join(save_dir, "orth_highfreq_awgn12_best.pth"),
        latest=os.path.join(save_dir, "orth_highfreq_awgn12_latest.pth"),
    )


def direct_channel_conv(system) -> nn.Conv2d:
    net = system.channel_encoder.net
    if not isinstance(net, nn.Conv2d):
        raise RuntimeError(f"expected direct Conv2d channel encoder, got {type(net)}")
    if net.weight.shape[:2] != (4, 16):
        raise RuntimeError(f"expected channel encoder weight [4,16,1,1], got {tuple(net.weight.shape)}")
    return net


def direct_channel_decoder_conv(system) -> nn.Conv2d:
    net = system.channel_decoder.net
    if not isinstance(net, nn.Conv2d):
        raise RuntimeError(f"expected direct Conv2d channel decoder, got {type(net)}")
    if net.weight.shape[:2] != (16, 4):
        raise RuntimeError(f"expected channel decoder weight [16,4,1,1], got {tuple(net.weight.shape)}")
    return net


@torch.no_grad()
def project_channel_rows_(conv: nn.Conv2d) -> torch.Tensor:
    w = conv.weight.detach().float().squeeze(-1).squeeze(-1)
    q, _ = torch.linalg.qr(w.t(), mode="reduced")
    w_orth = q.t().contiguous()
    conv.weight.copy_(w_orth.to(device=conv.weight.device, dtype=conv.weight.dtype).view_as(conv.weight))
    return w_orth


@torch.no_grad()
def orthogonalize_preserve_linear_chain_(system) -> torch.Tensor:
    enc = direct_channel_conv(system)
    dec = direct_channel_decoder_conv(system)
    a_old = enc.weight.detach().float().squeeze(-1).squeeze(-1)
    d_old = dec.weight.detach().float().squeeze(-1).squeeze(-1)
    gram = a_old @ a_old.t()
    eigvals, eigvecs = torch.linalg.eigh(gram)
    eigvals = eigvals.clamp_min(1e-8)
    inv_sqrt = eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.t()
    sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.t()
    a_new = inv_sqrt @ a_old
    d_new = d_old @ sqrt
    enc.weight.copy_(a_new.to(device=enc.weight.device, dtype=enc.weight.dtype).view_as(enc.weight))
    dec.weight.copy_(d_new.to(device=dec.weight.device, dtype=dec.weight.dtype).view_as(dec.weight))
    return a_new


def channel_matrix(system, dtype: torch.dtype | None = None) -> torch.Tensor:
    conv = direct_channel_conv(system)
    a = conv.weight.squeeze(-1).squeeze(-1)
    return a if dtype is None else a.to(dtype=dtype)


def semiorth_error(a: torch.Tensor) -> float:
    eye = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    return float((a @ a.t() - eye).abs().max().detach().cpu().item())


def encode_with_a(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.einsum("oc,bchw->bohw", a.to(device=z.device, dtype=z.dtype), z)


def lift_with_at(z_low: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.einsum("oc,bohw->bchw", a.to(device=z_low.device, dtype=z_low.dtype), z_low)


def lift_with_decoder(system, z_low: torch.Tensor, amp_enabled: bool, amp_dtype: torch.dtype) -> torch.Tensor:
    device = z_low.device
    dec_dtype = next(system.channel_decoder.parameters()).dtype
    with make_autocast(device, amp_enabled, amp_dtype):
        return system.channel_decoder(z_low.to(dec_dtype)).float()


def null_project(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    a = a.to(device=z.device, dtype=z.dtype)
    az = torch.einsum("oc,bchw->bohw", a, z)
    low = torch.einsum("oc,bohw->bchw", a, az)
    return z - low


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


def mean_neg_psnr_loss(x_hat: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
    mse_i = torch.mean((x_hat.float() - imgs.float()) ** 2, dim=(1, 2, 3)).clamp_min(1e-10)
    return (10.0 * torch.log10(mse_i)).mean()


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
def cache_semantics(system, loader, device, amp_enabled, amp_dtype, max_batches: int, split_name: str, repeats: int = 1):
    z_all: list[torch.Tensor] = []
    imgs_all: list[torch.Tensor] = []
    repeats = max(1, int(repeats))
    for ri in range(repeats):
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
            with make_autocast(device, amp_enabled, amp_dtype):
                z_sem = system.semantic_encoder(imgs)
            z_all.append(z_sem.float().cpu())
            imgs_all.append(imgs.float().cpu())
            if (bi + 1) % 10 == 0:
                print(f"  cached {split_name}: repeat {ri + 1}/{repeats}, batch {bi + 1}")
    if not z_all:
        raise RuntimeError(f"no {split_name} samples cached")
    return CachedSemanticDataset(torch.cat(z_all), torch.cat(imgs_all))


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


def forward_receiver(
    *,
    system,
    model: NullspaceHighFreqReceiver,
    z_sem: torch.Tensor,
    snr_b: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    lift_mode: str,
    generator: torch.Generator | None = None,
):
    a = channel_matrix(system, dtype=torch.float32)
    z_low = encode_with_a(z_sem.float(), a)
    y_norm, y_raw, scale = power_normalize_awgn(z_low, snr_b, generator=generator)
    z_at_raw = lift_with_at(y_raw.float(), a)
    z_norm_lift = lift_with_at(y_norm.float(), a)
    if lift_mode == "learned":
        z_base = lift_with_decoder(system, y_raw.float(), amp_enabled, amp_dtype)
        z_clean_ref = lift_with_decoder(system, z_low.float(), amp_enabled, amp_dtype)
    elif lift_mode == "at":
        z_base = z_at_raw
        z_clean_ref = lift_with_at(z_low.float(), a)
    else:
        raise ValueError(f"unknown lift_mode={lift_mode}")
    residual = model(z_base.float(), z_norm_lift.float(), z_at_raw.float(), scale, snr_b)
    z_hat = z_base + null_project(residual.float(), a)
    return {
        "a": a,
        "z_low": z_low,
        "y_norm": y_norm,
        "y_raw": y_raw,
        "scale": scale,
        "z_base": z_base,
        "z_clean_ref": z_clean_ref,
        "z_hat": z_hat,
    }


@torch.no_grad()
def validate(
    *,
    system,
    model: NullspaceHighFreqReceiver,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator,
) -> dict:
    model.eval()
    system.semantic_encoder.eval()
    system.channel_decoder.eval()
    system.semantic_decoder.eval()
    psnr_raw: list[torch.Tensor] = []
    psnr_clean: list[torch.Tensor] = []
    psnr_recv: list[torch.Tensor] = []
    psnr_full: list[torch.Tensor] = []
    loss_m = AverageMeter()
    use_cached_semantics = bool(args.precache_semantics) and not bool(args.train_semantic_encoder)
    snr_template = torch.full((args.batch_size,), float(args.snr_db), device=device, dtype=torch.float32)
    for bi, batch in enumerate(loader):
        if args.max_val_batches > 0 and bi >= args.max_val_batches:
            break
        if use_cached_semantics:
            z_sem, imgs = batch
            z_sem = z_sem.to(device, non_blocking=True)
            imgs = imgs.to(device, non_blocking=True)
        else:
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=True)
            with torch.set_grad_enabled(False):
                with make_autocast(device, amp_enabled, amp_dtype):
                    z_sem = system.semantic_encoder(imgs).float()
        snr_b = snr_template[: z_sem.shape[0]]
        out = forward_receiver(
            system=system,
            model=model,
            z_sem=z_sem,
            snr_b=snr_b,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            lift_mode=str(args.lift_mode),
            generator=generator,
        )
        x_hat = decode_latent(system, out["z_hat"], amp_enabled, amp_dtype).float().clamp(0, 1)
        x_raw = decode_latent(system, out["z_base"], amp_enabled, amp_dtype).float().clamp(0, 1)
        x_clean = decode_latent(system, out["z_clean_ref"], amp_enabled, amp_dtype).float().clamp(0, 1)
        x_full = decode_latent(system, z_sem.float(), amp_enabled, amp_dtype).float().clamp(0, 1)
        n = int(z_sem.shape[0])
        loss_m.update(float(mean_neg_psnr_loss(x_hat, imgs).item()), n)
        psnr_raw.append(psnr_per_image(x_raw, imgs.float()))
        psnr_clean.append(psnr_per_image(x_clean, imgs.float()))
        psnr_recv.append(psnr_per_image(x_hat, imgs.float()))
        psnr_full.append(psnr_per_image(x_full, imgs.float()))

    def avg(xs: list[torch.Tensor]) -> float:
        return float(torch.cat(xs).mean().item()) if xs else float("nan")

    recv = avg(psnr_recv)
    raw = avg(psnr_raw)
    clean = avg(psnr_clean)
    full = avg(psnr_full)
    a = channel_matrix(system, dtype=torch.float32)
    return {
        "val_loss_neg_psnr": loss_m.avg,
        "val_psnr_raw_at": raw,
        "val_psnr_clean_at": clean,
        "val_psnr_receiver": recv,
        "val_psnr_full_oracle": full,
        "val_gain_vs_raw": recv - raw,
        "val_gain_vs_baseline": recv - float(args.baseline_psnr),
        "val_aat_error": semiorth_error(a),
    }


def train_epoch(
    *,
    system,
    model: NullspaceHighFreqReceiver,
    loader: DataLoader,
    opt: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict:
    model.train()
    if bool(args.train_semantic_encoder):
        system.semantic_encoder.train()
    else:
        system.semantic_encoder.eval()
    direct_channel_conv(system).train()
    if bool(args.train_channel_decoder):
        system.channel_decoder.train()
    else:
        system.channel_decoder.eval()
    system.semantic_decoder.eval()
    meters = {name: AverageMeter() for name in ("loss", "psnr")}
    use_cached_semantics = bool(args.precache_semantics) and not bool(args.train_semantic_encoder)
    snr_template = torch.full((args.batch_size,), float(args.snr_db), device=device, dtype=torch.float32)
    for bi, batch in enumerate(loader):
        if args.max_train_batches > 0 and bi >= args.max_train_batches:
            break
        if use_cached_semantics:
            z_sem, imgs = batch
            z_sem = z_sem.to(device, non_blocking=True)
            imgs = imgs.to(device, non_blocking=True)
        else:
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=True)
            with torch.set_grad_enabled(bool(args.train_semantic_encoder)):
                with make_autocast(device, amp_enabled, amp_dtype):
                    z_sem = system.semantic_encoder(imgs).float()
        snr_b = snr_template[: z_sem.shape[0]]

        opt.zero_grad(set_to_none=True)
        with make_autocast(device, amp_enabled, amp_dtype):
            out = forward_receiver(
                system=system,
                model=model,
                z_sem=z_sem,
                snr_b=snr_b,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                lift_mode=str(args.lift_mode),
            )
            x_hat = decode_latent(system, out["z_hat"], amp_enabled, amp_dtype).float().clamp(0, 1)
            loss = mean_neg_psnr_loss(x_hat, imgs)
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
        if float(args.a_lr) > 0.0:
            project_channel_rows_(direct_channel_conv(system))

        n = int(z_sem.shape[0])
        meters["loss"].update(float(loss.item()), n)
        meters["psnr"].update(float(-loss.item()), n)
    return {f"train_{k}": v.avg for k, v in meters.items()}


def save_checkpoint(path: str, system, model: NullspaceHighFreqReceiver, args: argparse.Namespace, epoch: int, metrics: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = channel_matrix(system, dtype=torch.float32).detach().cpu()
    payload = {
        "receiver_state_dict": model.state_dict(),
        "semantic_encoder_state_dict": system.semantic_encoder.state_dict() if bool(args.train_semantic_encoder) else None,
        "channel_encoder_state_dict": system.channel_encoder.state_dict(),
        "channel_decoder_state_dict": system.channel_decoder.state_dict(),
        "a_matrix": a,
        "aat": a @ a.t(),
        "aat_error": semiorth_error(a),
        "receiver_type": "semiorthogonal_nullspace_highfreq",
        "loss": "mean_negative_per_image_psnr",
        "lift_mode": str(args.lift_mode),
        "train_channel_decoder": bool(args.train_channel_decoder),
        "snr_db": float(args.snr_db),
        "baseline_psnr": float(args.baseline_psnr),
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
    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False
    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}")
    print(f"save_dir={save_dir}")
    print("rule2: PSNR=mean(per-image PSNR), train/test SNR=12 dB, A A^T=I_4, power norm after A")

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
    if bool(args.train_semantic_encoder):
        for p in system.semantic_encoder.parameters():
            p.requires_grad_(True)
    conv = direct_channel_conv(system)
    dec_conv = direct_channel_decoder_conv(system)
    if str(args.lift_mode) == "learned":
        orthogonalize_preserve_linear_chain_(system)
    else:
        project_channel_rows_(conv)
    conv.weight.requires_grad_(float(args.a_lr) > 0.0)
    dec_conv.weight.requires_grad_(bool(args.train_channel_decoder))
    a0 = channel_matrix(system, dtype=torch.float32)
    print(f"channel_encoder_output=[4,16,16], tag={tag}, aat_error={semiorth_error(a0):.3e}")

    if bool(args.train_semantic_encoder) or not bool(args.precache_semantics):
        train_loader, val_loader = train_img_loader, val_img_loader
        print(
            "image pipeline: cache_decoded stores full decoded images; "
            "train crops are sampled in __getitem__ every epoch; semantic latent pre-cache is OFF"
        )
    else:
        print("Pre-caching frozen Swin latents (smoke/debug only; fixes train crops) ...")
        train_cached = cache_semantics(
            system,
            train_img_loader,
            device,
            amp_enabled,
            amp_dtype,
            int(args.max_train_batches),
            "train",
            repeats=int(args.semantic_cache_repeats),
        )
        val_cached = cache_semantics(
            system,
            val_img_loader,
            device,
            amp_enabled,
            amp_dtype,
            int(args.max_val_batches),
            "valid",
            repeats=1,
        )
        train_loader, val_loader = make_cached_loaders(args, train_cached, val_cached, device)
        print(f"cached train={len(train_cached)} valid={len(val_cached)}")

    model = NullspaceHighFreqReceiver(
        latent_channels=16,
        hidden=int(args.hidden),
        depth=int(args.depth),
    ).to(device)
    opt_groups = [{"params": model.parameters(), "lr": float(args.lr)}]
    if bool(args.train_semantic_encoder):
        opt_groups.append({"params": system.semantic_encoder.parameters(), "lr": float(args.semantic_encoder_lr)})
    if float(args.a_lr) > 0.0:
        opt_groups.append({"params": [conv.weight], "lr": float(args.a_lr)})
    if bool(args.train_channel_decoder):
        opt_groups.append({"params": [dec_conv.weight], "lr": float(args.channel_decoder_lr)})
    opt = optim.AdamW(opt_groups, weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    paths = make_paths(save_dir)
    best_psnr = -1.0
    print(
        f"arch=NullspaceHighFreqReceiver hidden={args.hidden} depth={args.depth} epochs={args.epochs} "
        f"lr={float(args.lr):g} a_lr={float(args.a_lr):g}; lift_mode={args.lift_mode}; "
        f"channel_decoder_train={bool(args.train_channel_decoder)}; semantic_decoder=frozen; low_denoiser=False; "
        f"semantic_encoder_train={bool(args.train_semantic_encoder)}; "
        "dist_loss=0 channel_loss=0"
    )

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_epoch(
            system=system,
            model=model,
            loader=train_loader,
            opt=opt,
            scaler=scaler,
            args=args,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        if epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(args.seed) + 1000 + epoch)
            val_metrics = validate(
                system=system,
                model=model,
                loader=val_loader,
                args=args,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                generator=gen,
            )
            metrics = {**train_metrics, **val_metrics}
            is_best = val_metrics["val_psnr_receiver"] > best_psnr
            if is_best:
                best_psnr = val_metrics["val_psnr_receiver"]
                save_checkpoint(paths.best, system, model, args, epoch, metrics)
            save_checkpoint(paths.latest, system, model, args, epoch, metrics)
            print(
                f"[epoch {epoch:03d}/{int(args.epochs)}] "
                f"train_loss={train_metrics['train_loss']:.4f} train_psnr_obj={train_metrics['train_psnr']:.4f} | "
                f"raw={val_metrics['val_psnr_raw_at']:.4f} clean={val_metrics['val_psnr_clean_at']:.4f} "
                f"recv={val_metrics['val_psnr_receiver']:.4f} full={val_metrics['val_psnr_full_oracle']:.4f} "
                f"gain_raw={val_metrics['val_gain_vs_raw']:+.4f} "
                f"gain_base={val_metrics['val_gain_vs_baseline']:+.4f} "
                f"aat_err={val_metrics['val_aat_error']:.2e} {'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{int(args.epochs)}] "
                f"train_loss={train_metrics['train_loss']:.4f} train_psnr_obj={train_metrics['train_psnr']:.4f} "
                f"aat_err={semiorth_error(channel_matrix(system, dtype=torch.float32)):.2e}"
            )
    print(f"best_psnr={best_psnr:.4f} target>{float(args.baseline_psnr) + 0.5:.4f} ckpt={paths.best}")


if __name__ == "__main__":
    main()
