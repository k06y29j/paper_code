#!/usr/bin/env python3
from __future__ import annotations

import argparse
import builtins
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


CDDM_ROOT = Path(__file__).resolve().parents[1]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))

from Autoencoder.data.datasets import FlatImageFolder  # noqa: E402
from Autoencoder.net.channel import Channel  # noqa: E402
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder  # noqa: E402


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_jscc_config(args: argparse.Namespace, device: torch.device) -> SimpleNamespace:
    cfg = SimpleNamespace()
    cfg.loss_function = "MSE"
    cfg.dataset = "DIV2K"
    cfg.C = int(args.C)
    cfg.SNRs = float(args.snr_db)
    cfg.CUDA = device.type == "cuda"
    cfg.device = device
    cfg.channel_type = args.channel_type.lower()
    cfg.image_dims = (3, int(args.crop_size), int(args.crop_size))
    cfg.encoder_kwargs = dict(
        img_size=(args.crop_size, args.crop_size),
        patch_size=2,
        in_chans=3,
        embed_dims=[128, 192, 256, 320],
        depths=[2, 2, 6, 2],
        num_heads=[4, 6, 8, 10],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=torch.nn.LayerNorm,
        patch_norm=True,
    )
    cfg.decoder_kwargs = dict(
        img_size=(args.crop_size, args.crop_size),
        embed_dims=[320, 256, 192, 128],
        depths=[2, 6, 2, 2],
        num_heads=[10, 8, 6, 4],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=torch.nn.LayerNorm,
        patch_norm=True,
    )
    return cfg


def make_loader(root: str, crop_size: int, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = FlatImageFolder(root=root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def make_train_loader(
    root: str,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.RandomCrop((crop_size, crop_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    dataset = FlatImageFolder(root=root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.ckpt_root).expanduser().resolve()
    base = f"snr{args.snr_db:g}_channel_{args.channel_type.lower()}_C{args.C}"
    enc = root / f"encoder_{base}.pt"
    dec = root / f"decoder_{base}.pt"
    if not enc.is_file():
        raise FileNotFoundError(enc)
    if not dec.is_file():
        raise FileNotFoundError(dec)
    return enc, dec


def load_models(cfg: SimpleNamespace, enc_path: Path, dec_path: Path) -> tuple[nn.Module, nn.Module]:
    encoder = JSCC_encoder(cfg, cfg.C).to(cfg.device)
    decoder = JSCC_decoder(cfg, cfg.C).to(cfg.device)
    encoder.load_state_dict(torch.load(enc_path, map_location=cfg.device, weights_only=True), strict=True)
    decoder.load_state_dict(torch.load(dec_path, map_location=cfg.device, weights_only=True), strict=True)
    encoder.eval()
    decoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)
    return encoder, decoder


def load_pca_basis(path: Path, device: torch.device, c: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if "mean" not in obj or "basis" not in obj:
        raise KeyError(f"{path} must contain mean and basis")
    if "eigvals" not in obj:
        raise KeyError(f"{path} must contain eigvals for PCA coefficient whitening")
    mean = obj["mean"].float()
    basis = obj["basis"].float()
    eigvals = obj["eigvals"].float()
    if tuple(mean.shape) == (c,):
        mean = mean.view(1, c, 1, 1)
    if tuple(mean.shape) != (1, c, 1, 1):
        raise ValueError(f"bad PCA mean shape {tuple(mean.shape)}, expected {(1, c, 1, 1)}")
    if tuple(basis.shape) != (c, c):
        raise ValueError(f"bad PCA basis shape {tuple(basis.shape)}, expected {(c, c)}")
    if tuple(eigvals.shape) != (c,):
        raise ValueError(f"bad PCA eigvals shape {tuple(eigvals.shape)}, expected {(c,)}")
    coeff_std = eigvals.clamp_min(1e-12).sqrt().view(1, c, 1, 1)
    return mean.to(device), basis.to(device), coeff_std.to(device), obj


def pca_project(
    z: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    coeff_std: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    u_keep = basis[:, :k]
    u_perp = basis[:, k:]
    centered = z - mean
    c_keep = torch.einsum("bchw,ck->bkhw", centered, u_keep) / coeff_std[:, :k]
    c_perp = torch.einsum("bchw,ck->bkhw", centered, u_perp) / coeff_std[:, k:]
    return c_keep, c_perp


def pca_inverse(
    c_keep: torch.Tensor,
    c_perp: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    coeff_std: torch.Tensor,
) -> torch.Tensor:
    k = c_keep.shape[1]
    u_keep = basis[:, :k]
    u_perp = basis[:, k : k + c_perp.shape[1]]
    z_keep = torch.einsum("bkhw,ck->bchw", c_keep * coeff_std[:, :k], u_keep)
    z_perp = torch.einsum("bqhw,cq->bchw", c_perp * coeff_std[:, k : k + c_perp.shape[1]], u_perp)
    return mean + z_keep + z_perp


def channel_to_decoder_input(
    feature: torch.Tensor,
    channel: Channel,
    snr_db: float,
    channel_type: str,
) -> torch.Tensor:
    noisy_y, pwr, h = channel.forward(feature, snr_db)
    ch = channel_type.lower()
    if ch == "rayleigh":
        sigma_square = 1.0 / (10 ** (snr_db / 10))
        noisy_y = torch.conj(h) * noisy_y / (torch.abs(h) ** 2 + sigma_square)
    elif ch != "awgn":
        raise ValueError(f"unsupported channel_type={channel_type}")
    return torch.cat((torch.real(noisy_y), torch.imag(noisy_y)), dim=2) * torch.sqrt(pwr)


def luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]


def radial_bins(height: int, width: int, bins: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    cy = height // 2
    cx = width // 2
    radius = torch.sqrt((yy.float() - cy) ** 2 + (xx.float() - cx) ** 2)
    radius = radius / radius.max().clamp_min(1e-12)
    idx = torch.clamp((radius * bins).long(), max=bins - 1)
    weight = 0.2 + (3.0 - 0.2) * radius.pow(1.5)
    return idx, weight


def batch_spectrum_stats(x_hat: torch.Tensor, x: torch.Tensor, bins: int) -> dict[str, torch.Tensor]:
    pred = luma(x_hat.float().clamp(0.0, 1.0))
    target = luma(x.float())
    pred = pred - pred.mean(dim=(-2, -1), keepdim=True)
    target = target - target.mean(dim=(-2, -1), keepdim=True)
    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target), dim=(-2, -1))
    _, _, height, width = x.shape
    bin_idx, fft_weight = radial_bins(height, width, bins, x.device)
    diff = pred_fft - target_fft
    complex_err = (fft_weight.unsqueeze(0) * diff.abs().square()).mean()
    cross = pred_fft * torch.conj(target_fft)
    phase_score = torch.real(cross / (cross.abs() + 1e-8)).mean()
    pred_power = pred_fft.abs().square()
    target_power = target_fft.abs().square()
    high_mask = (bin_idx.float() / float(max(1, bins - 1))) > 0.5
    hf_ratio = pred_power[:, high_mask].sum() / target_power[:, high_mask].sum().clamp_min(1e-12)
    pred_profile = torch.zeros(bins, dtype=torch.float64, device=x.device)
    target_profile = torch.zeros(bins, dtype=torch.float64, device=x.device)
    pred_profile.scatter_add_(0, bin_idx.flatten(), pred_power.double().sum(dim=0).flatten())
    target_profile.scatter_add_(0, bin_idx.flatten(), target_power.double().sum(dim=0).flatten())
    counts = torch.bincount(bin_idx.flatten(), minlength=bins).double().to(x.device).clamp_min(1.0)
    pred_profile = pred_profile / counts / max(1, x.shape[0])
    target_profile = target_profile / counts / max(1, x.shape[0])
    return {
        "complex_fft_err_sum": complex_err.detach().double() * x.shape[0],
        "phase_score_sum": phase_score.detach().double() * x.shape[0],
        "hf_ratio_sum": hf_ratio.detach().double() * x.shape[0],
        "pred_profile_sum": pred_profile.detach() * x.shape[0],
        "target_profile_sum": target_profile.detach() * x.shape[0],
    }


def weighted_complex_fft_loss(x_hat: torch.Tensor, x: torch.Tensor, bins: int) -> torch.Tensor:
    pred = luma(x_hat.float())
    target = luma(x.float())
    pred = pred - pred.mean(dim=(-2, -1), keepdim=True)
    target = target - target.mean(dim=(-2, -1), keepdim=True)
    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target), dim=(-2, -1))
    _, weight = radial_bins(pred.shape[-2], pred.shape[-1], bins, pred.device)
    return (weight.unsqueeze(0) * (pred_fft - target_fft).abs().square()).mean()


def log_amplitude_fft_loss(x_hat: torch.Tensor, x: torch.Tensor, bins: int) -> torch.Tensor:
    pred = luma(x_hat.float())
    target = luma(x.float())
    pred = pred - pred.mean(dim=(-2, -1), keepdim=True)
    target = target - target.mean(dim=(-2, -1), keepdim=True)
    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target), dim=(-2, -1))
    _, weight = radial_bins(pred.shape[-2], pred.shape[-1], bins, pred.device)
    pred_log = torch.log(pred_fft.abs() + 1e-6)
    target_log = torch.log(target_fft.abs() + 1e-6)
    return (weight.unsqueeze(0) * (pred_log - target_log).abs()).mean()


def radial_log_power_loss(x_hat: torch.Tensor, x: torch.Tensor, bins: int) -> torch.Tensor:
    pred = luma(x_hat.float())
    target = luma(x.float())
    pred = pred - pred.mean(dim=(-2, -1), keepdim=True)
    target = target - target.mean(dim=(-2, -1), keepdim=True)
    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target), dim=(-2, -1))
    bin_idx, weight = radial_bins(pred.shape[-2], pred.shape[-1], bins, pred.device)
    pred_power = pred_fft.abs().square()
    target_power = target_fft.abs().square()
    pred_bins = pred_power.new_zeros((pred.shape[0], bins))
    target_bins = target_power.new_zeros((target.shape[0], bins))
    pred_bins.scatter_add_(1, bin_idx.flatten().expand(pred.shape[0], -1), pred_power.flatten(1))
    target_bins.scatter_add_(1, bin_idx.flatten().expand(target.shape[0], -1), target_power.flatten(1))
    counts = torch.bincount(bin_idx.flatten(), minlength=bins).to(pred.device).float().clamp_min(1.0)
    pred_bins = pred_bins / counts.unsqueeze(0)
    target_bins = target_bins / counts.unsqueeze(0)
    weights = torch.linspace(float(weight.min()), float(weight.max()), bins, device=pred.device)
    return (weights.unsqueeze(0) * (torch.log(pred_bins + 1e-8) - torch.log(target_bins + 1e-8)).abs()).mean()


class MetricAccumulator:
    def __init__(self, bins: int):
        self.bins = bins
        self.seen = 0
        self.psnr_sum = 0.0
        self.mse_sum = 0.0
        self.c12_mse_sum = 0.0
        self.c12_count = 0
        self.c36_mse_sum = 0.0
        self.c36_count = 0
        self.complex_fft_err_sum = 0.0
        self.phase_score_sum = 0.0
        self.hf_ratio_sum = 0.0
        self.pred_profile_sum = np.zeros(bins, dtype=np.float64)
        self.target_profile_sum = np.zeros(bins, dtype=np.float64)

    def update(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        *,
        c12_hat: torch.Tensor | None = None,
        c12_gt: torch.Tensor | None = None,
        c36_hat: torch.Tensor | None = None,
        c36_gt: torch.Tensor | None = None,
    ) -> None:
        bsz = int(x.shape[0])
        mse_i = (x_hat.float().clamp(0.0, 1.0) - x.float()).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
        psnr_i = 10.0 * torch.log10(1.0 / mse_i)
        self.seen += bsz
        self.mse_sum += float(mse_i.sum().item())
        self.psnr_sum += float(psnr_i.sum().item())
        if c12_hat is not None and c12_gt is not None:
            self.c12_mse_sum += float((c12_hat.float() - c12_gt.float()).square().sum().item())
            self.c12_count += int(c12_gt.numel())
        if c36_hat is not None and c36_gt is not None:
            self.c36_mse_sum += float((c36_hat.float() - c36_gt.float()).square().sum().item())
            self.c36_count += int(c36_gt.numel())
        spec = batch_spectrum_stats(x_hat, x, self.bins)
        self.complex_fft_err_sum += float(spec["complex_fft_err_sum"].item())
        self.phase_score_sum += float(spec["phase_score_sum"].item())
        self.hf_ratio_sum += float(spec["hf_ratio_sum"].item())
        self.pred_profile_sum += spec["pred_profile_sum"].cpu().numpy()
        self.target_profile_sum += spec["target_profile_sum"].cpu().numpy()

    def finalize(self) -> dict[str, float]:
        pred_profile = self.pred_profile_sum / max(1, self.seen)
        target_profile = self.target_profile_sum / max(1, self.seen)
        radial_err = float(np.mean(np.abs(np.log(pred_profile + 1e-12) - np.log(target_profile + 1e-12))))
        return {
            "psnr": self.psnr_sum / max(1, self.seen),
            "mse": self.mse_sum / max(1, self.seen),
            "c12_mse": self.c12_mse_sum / self.c12_count if self.c12_count else math.nan,
            "c36_mse": self.c36_mse_sum / self.c36_count if self.c36_count else math.nan,
            "radial_err": radial_err,
            "complex_fft_err": self.complex_fft_err_sum / max(1, self.seen),
            "phase_score": self.phase_score_sum / max(1, self.seen),
            "hf_power_ratio": self.hf_ratio_sum / max(1, self.seen),
        }


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PCANullPredictor(nn.Module):
    def __init__(self, in_ch: int = 37, hidden: int = 128, out_ch: int = 12, depth: int = 6):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(depth):
            layers.append(ResBlock(hidden))
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, noisy_c36: torch.Tensor, snr_value: float) -> torch.Tensor:
        snr_map = noisy_c36.new_full(
            (noisy_c36.shape[0], 1, noisy_c36.shape[2], noisy_c36.shape[3]),
            float(snr_value) / 20.0,
        )
        x = torch.cat([noisy_c36, snr_map], dim=1)
        return self.head(self.body(x))


class PCAZBasePredictor(nn.Module):
    def __init__(self, in_ch: int = 85, hidden: int = 160, out_ch: int = 12, depth: int = 8):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(depth):
            layers.append(ResBlock(hidden))
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def make_input(self, noisy_c36: torch.Tensor, z_base48: torch.Tensor, snr_value: float) -> torch.Tensor:
        snr_map = noisy_c36.new_full(
            (noisy_c36.shape[0], 1, noisy_c36.shape[2], noisy_c36.shape[3]),
            float(snr_value) / 20.0,
        )
        return torch.cat([noisy_c36, z_base48, snr_map], dim=1)

    def forward(self, noisy_c36: torch.Tensor, z_base48: torch.Tensor, snr_value: float) -> torch.Tensor:
        return self.head(self.body(self.make_input(noisy_c36, z_base48, snr_value)))


class PCADenoisePredictor(nn.Module):
    def __init__(self, in_ch: int = 85, hidden: int = 160, depth: int = 8, init_alpha: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(depth):
            layers.append(ResBlock(hidden))
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, 48, kernel_size=3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        noisy_c36: torch.Tensor,
        z_base48: torch.Tensor,
        snr_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        snr_map = noisy_c36.new_full(
            (noisy_c36.shape[0], 1, noisy_c36.shape[2], noisy_c36.shape[3]),
            float(snr_value) / 20.0,
        )
        out = self.head(self.body(torch.cat([noisy_c36, z_base48, snr_map], dim=1)))
        delta_c36 = out[:, :36]
        c12_hat = out[:, 36:]
        c36_hat = noisy_c36 + self.alpha * delta_c36
        return c36_hat, c12_hat, delta_c36, self.alpha


class FreqAwarePCANullPredictor(nn.Module):
    def __init__(self, in_ch: int = 85, hidden: int = 160, out_ch: int = 12, depth: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.body = nn.Sequential(*[ResBlock(hidden) for _ in range(depth)])
        self.freq_gate = nn.Sequential(
            nn.Conv2d(48 * 3, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.Sigmoid(),
        )
        self.head = nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, noisy_c36: torch.Tensor, z_base48: torch.Tensor, snr_value: float) -> torch.Tensor:
        snr_map = noisy_c36.new_full(
            (noisy_c36.shape[0], 1, noisy_c36.shape[2], noisy_c36.shape[3]),
            float(snr_value) / 20.0,
        )
        feat = self.body(self.stem(torch.cat([noisy_c36, z_base48, snr_map], dim=1)))
        fft = torch.fft.fft2(z_base48.float(), dim=(-2, -1))
        amp = torch.log(fft.abs() + 1e-6).to(z_base48.dtype)
        denom = (fft.abs() + 1e-6).to(z_base48.dtype)
        cos_phase = (fft.real.to(z_base48.dtype) / denom).clamp(-1.0, 1.0)
        sin_phase = (fft.imag.to(z_base48.dtype) / denom).clamp(-1.0, 1.0)
        gate = self.freq_gate(torch.cat([amp, cos_phase, sin_phase], dim=1))
        feat = feat * (1.0 + gate)
        return self.head(feat)


class ARPCANullPredictor(nn.Module):
    def __init__(self, keep_ch: int = 36, out_ch: int = 12, group_ch: int = 4, hidden: int = 160, depth: int = 6):
        super().__init__()
        if out_ch % group_ch != 0:
            raise ValueError(f"out_ch={out_ch} must be divisible by group_ch={group_ch}")
        self.keep_ch = keep_ch
        self.out_ch = out_ch
        self.group_ch = group_ch
        self.blocks = nn.ModuleList()
        groups = out_ch // group_ch
        for group_idx in range(groups):
            in_ch = keep_ch + group_idx * group_ch + 1
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
                nn.GELU(),
            ]
            for _ in range(depth):
                layers.append(ResBlock(hidden))
            head = nn.Conv2d(hidden, group_ch, kernel_size=3, padding=1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            layers.append(head)
            self.blocks.append(nn.Sequential(*layers))

    def forward(self, noisy_c36: torch.Tensor, snr_value: float) -> torch.Tensor:
        pred_groups: list[torch.Tensor] = []
        for block in self.blocks:
            snr_map = noisy_c36.new_full(
                (noisy_c36.shape[0], 1, noisy_c36.shape[2], noisy_c36.shape[3]),
                float(snr_value) / 20.0,
            )
            x = torch.cat([noisy_c36] + pred_groups + [snr_map], dim=1)
            pred_groups.append(block(x))
        return torch.cat(pred_groups, dim=1)


class DiffusionC12Predictor(nn.Module):
    def __init__(self, in_ch: int = 98, hidden: int = 160, out_ch: int = 12, depth: int = 8):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(depth):
            layers.append(ResBlock(hidden))
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, c12_t: torch.Tensor, noisy_c36: torch.Tensor, z_base48: torch.Tensor, t_norm: torch.Tensor, snr_value: float) -> torch.Tensor:
        snr_map = noisy_c36.new_full(
            (noisy_c36.shape[0], 1, noisy_c36.shape[2], noisy_c36.shape[3]),
            float(snr_value) / 20.0,
        )
        t_map = t_norm.view(-1, 1, 1, 1).expand(-1, 1, noisy_c36.shape[2], noisy_c36.shape[3]).to(noisy_c36.dtype)
        x = torch.cat([c12_t, noisy_c36, z_base48, snr_map, t_map], dim=1)
        return self.head(self.body(x))


METHOD_TITLES = {
    "exp01": "exp01_direct_c12_resnet at SNR=9 dB",
    "exp02": "exp02_c36_zbase_c12_resnet at SNR=9 dB",
    "exp03": "exp03_denoise_c36_predict_c12 at SNR=9 dB",
    "exp04": "exp04_complex_fft_loss at SNR=9 dB",
    "exp05": "exp05_amp_radial_loss at SNR=9 dB",
    "exp06": "exp06_freq_aware_gate_predictor at SNR=9 dB",
    "exp07": "exp07_ar_pca_residual at SNR=9 dB",
    "exp08": "exp08_diffusion_c12_only at SNR=9 dB",
}


def method_name(args: argparse.Namespace) -> str:
    fixed = {
        "exp01": "exp01_direct_c12_resnet",
        "exp02": "exp02_c36_zbase_c12_resnet",
        "exp03": "exp03_denoise_c36_predict_c12",
        "exp06": "exp06_freq_aware_gate_predictor",
        "exp07": "exp07_ar_pca_residual",
        "exp08": "exp08_diffusion_c12_only",
    }
    if args.mode in fixed:
        return fixed[args.mode]
    name = Path(args.exp_name).name
    return name if name else args.mode


def predictor_kind(args: argparse.Namespace) -> str:
    if args.mode == "exp01":
        return "exp01_direct_c12_resnet"
    if args.mode in {"exp04", "exp05"} and args.predictor_base == "exp03":
        return "pca_denoise_c36_predict_c12"
    if args.mode in {"exp02", "exp04", "exp05"}:
        return "pca_zbase_c12_resnet"
    if args.mode == "exp03":
        return "pca_denoise_c36_predict_c12"
    if args.mode == "exp06":
        return "freq_aware_pca_null_predictor"
    if args.mode == "exp07":
        return "ar_pca_residual_predictor"
    raise ValueError(args.mode)


def load_baseline_refs(path: Path) -> dict[str, dict[str, float]]:
    refs: dict[str, dict[str, float]] = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            refs[row["method"]] = {
                k: float(v)
                for k, v in row.items()
                if k != "method" and v not in {"", "nan", "NaN"}
            }
    required = {"full_c48_jscc", "pca36_zerofill", "oracle_c12"}
    missing = required - set(refs)
    if missing:
        raise KeyError(f"missing baseline rows in {path}: {sorted(missing)}")
    return refs


def read_existing_val_metrics(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float] = {"method": row["method"]}  # type: ignore[dict-item]
            for key, value in row.items():
                if key == "method":
                    continue
                parsed[key] = float(value) if value not in {"", "nan", "NaN"} else math.nan
            rows.append(parsed)
    return rows


def read_existing_spectrum_metrics(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        columns = [c for c in reader.fieldnames if c.endswith("_luma_power")]
        values = {c.removesuffix("_luma_power"): [] for c in columns}
        for row in reader:
            for c in columns:
                values[c.removesuffix("_luma_power")].append(float(row[c]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in values.items()}


def args_match_checkpoint(saved: dict, args: argparse.Namespace) -> bool:
    checks = (
        "mode",
        "exp_name",
        "C",
        "keep_ch",
        "snr_db",
        "channel_type",
        "hidden",
        "depth",
        "lambda_c",
        "lambda_c36",
        "lambda_fft",
        "lambda_amp",
        "amp_loss_type",
        "predictor_base",
        "ar_group_ch",
        "diffusion_steps",
        "diffusion_sample_steps",
    )
    current = vars(args)
    for key in checks:
        if key in saved and saved[key] != current[key]:
            return False
    return True


def maybe_resume_predictor(
    out_dir: Path,
    predictor: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, float] | None, dict[str, np.ndarray], int]:
    latest_path = out_dir / "latest_checkpoint.pt"
    val_path = out_dir / "val_metrics.csv"
    if not latest_path.is_file() or not val_path.is_file():
        return [], None, {}, 1
    rows = read_existing_val_metrics(val_path)
    if not rows:
        return [], None, {}, 1
    checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    if isinstance(saved_args, dict) and not args_match_checkpoint(saved_args, args):
        print(f"resume skipped: args mismatch in {latest_path}", flush=True)
        return [], None, {}, 1
    latest_epoch = int(checkpoint.get("epoch", 0))
    if latest_epoch <= 0:
        return [], None, {}, 1
    predictor.load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    best_row = max(rows, key=lambda r: r["psnr"])
    best_profiles = read_existing_spectrum_metrics(out_dir / "spectrum_metrics.csv")
    start_epoch = latest_epoch + 1
    print(
        f"resume_from={latest_path} latest_epoch={latest_epoch} "
        f"next_epoch={start_epoch} existing_val_rows={len(rows)}",
        flush=True,
    )
    return rows, best_row, best_profiles, start_epoch


@torch.inference_mode()
def run_baselines(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, float]], dict[str, np.ndarray]]:
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    cfg = build_jscc_config(args, device)
    enc_path, dec_path = checkpoint_paths(args)
    mean, basis, coeff_std, pca_obj = load_pca_basis(Path(args.pca_basis).expanduser().resolve(), device, int(args.C))
    encoder, decoder = load_models(cfg, enc_path, dec_path)
    val_loader = make_loader(args.val_dir, args.crop_size, args.batch_size, args.num_workers, device.type == "cuda")
    channels = {
        "full_c48_jscc": Channel(cfg),
        "pca36_zerofill": Channel(cfg),
        "oracle_c12": Channel(cfg),
    }
    acc = {
        "full_c48_jscc": MetricAccumulator(args.spectrum_bins),
        "pca36_zerofill": MetricAccumulator(args.spectrum_bins),
        "pca36_clean_upper": MetricAccumulator(args.spectrum_bins),
        "oracle_c12": MetricAccumulator(args.spectrum_bins),
    }
    zero_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}
    start = time.time()
    print(f"device={device} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"encoder={enc_path}", flush=True)
    print(f"decoder={dec_path}", flush=True)
    print(f"pca_basis={args.pca_basis}", flush=True)
    print(f"val_dir={args.val_dir} images={len(val_loader.dataset)}", flush=True)
    print(f"snr_db={args.snr_db:g} channel={args.channel_type} seed={args.seed}", flush=True)

    for batch_idx, (imgs, _labels) in enumerate(tqdm(val_loader, desc="pca c48 snr9 baselines", dynamic_ncols=True)):
        if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
            break
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        z, _ = encoder(imgs)
        c36, c12 = pca_project(z, mean, basis, coeff_std, args.keep_ch)
        shape = tuple(c12.shape)
        if shape not in zero_cache:
            zero_cache[shape] = torch.zeros(shape, dtype=c12.dtype, device=c12.device)
        zero_c12 = zero_cache[shape]

        torch.manual_seed(args.val_noise_seed + batch_idx)
        torch.cuda.manual_seed_all(args.val_noise_seed + batch_idx)
        z_full_rx = channel_to_decoder_input(z, channels["full_c48_jscc"], args.snr_db, args.channel_type)
        x_full = decoder(z_full_rx).clamp(0.0, 1.0)
        acc["full_c48_jscc"].update(x_full, imgs)

        torch.manual_seed(args.val_noise_seed + batch_idx)
        torch.cuda.manual_seed_all(args.val_noise_seed + batch_idx)
        c36_rx = channel_to_decoder_input(c36, channels["pca36_zerofill"], args.snr_db, args.channel_type)
        z_zero = pca_inverse(c36_rx, zero_c12, mean, basis, coeff_std)
        x_zero = decoder(z_zero).clamp(0.0, 1.0)
        acc["pca36_zerofill"].update(x_zero, imgs, c12_hat=zero_c12, c12_gt=c12, c36_hat=c36_rx, c36_gt=c36)

        z_clean = pca_inverse(c36, zero_c12, mean, basis, coeff_std)
        x_clean = decoder(z_clean).clamp(0.0, 1.0)
        acc["pca36_clean_upper"].update(x_clean, imgs, c12_hat=zero_c12, c12_gt=c12, c36_hat=c36, c36_gt=c36)

        torch.manual_seed(args.val_noise_seed + batch_idx)
        torch.cuda.manual_seed_all(args.val_noise_seed + batch_idx)
        c36_rx_oracle = channel_to_decoder_input(c36, channels["oracle_c12"], args.snr_db, args.channel_type)
        z_oracle = pca_inverse(c36_rx_oracle, c12, mean, basis, coeff_std)
        x_oracle = decoder(z_oracle).clamp(0.0, 1.0)
        acc["oracle_c12"].update(x_oracle, imgs, c12_hat=c12, c12_gt=c12, c36_hat=c36_rx_oracle, c36_gt=c36)

    rows: list[dict[str, float]] = []
    for method in ["full_c48_jscc", "pca36_zerofill", "pca36_clean_upper", "oracle_c12"]:
        row = {"method": method, "epoch": 0.0}
        row.update(acc[method].finalize())
        rows.append(row)

    by_method = {row["method"]: row for row in rows}
    pca_psnr = by_method["pca36_zerofill"]["psnr"]
    full_psnr = by_method["full_c48_jscc"]["psnr"]
    oracle_psnr = by_method["oracle_c12"]["psnr"]
    for row in rows:
        row["gain_vs_pca36"] = row["psnr"] - pca_psnr
        row["gap_to_full"] = full_psnr - row["psnr"]
        row["gap_to_oracle"] = oracle_psnr - row["psnr"]

    profiles = {method: acc[method].pred_profile_sum / max(1, acc[method].seen) for method in acc}
    profiles["ground_truth"] = acc["full_c48_jscc"].target_profile_sum / max(1, acc["full_c48_jscc"].seen)
    elapsed = time.time() - start
    print(f"elapsed_sec={elapsed:.1f}", flush=True)
    if device.type == "cuda":
        print(f"max_gpu_mem_mb={torch.cuda.max_memory_allocated() / 1024**2:.1f}", flush=True)
    for row in rows:
        print(
            f"{row['method']} psnr={row['psnr']:.4f} mse={row['mse']:.8f} "
            f"gain_vs_pca36={row['gain_vs_pca36']:+.4f} gap_to_full={row['gap_to_full']:+.4f}",
            flush=True,
        )
    torch.save(
        {
            "kind": "baseline_reference_no_trainable_predictor",
            "args": vars(args),
            "metrics": rows,
            "pca_metadata": {
                k: (tuple(v.shape) if torch.is_tensor(v) else v)
                for k, v in pca_obj.items()
                if k not in {"mean", "basis", "eigvals", "explained"}
            },
        },
        out_dir / "best_checkpoint.pt",
    )
    return rows, profiles


@torch.inference_mode()
def validate_exp01(
    predictor: PCANullPredictor,
    encoder: nn.Module,
    decoder: nn.Module,
    channel: Channel,
    val_loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    basis: torch.Tensor,
    coeff_std: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
    baseline_refs: dict[str, dict[str, float]],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    predictor.eval()
    acc = MetricAccumulator(args.spectrum_bins)
    for batch_idx, (imgs, _labels) in enumerate(tqdm(val_loader, desc=f"val exp01 epoch {epoch}", dynamic_ncols=True)):
        if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
            break
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        z, _ = encoder(imgs)
        c36, c12 = pca_project(z, mean, basis, coeff_std, args.keep_ch)
        torch.manual_seed(args.val_noise_seed + batch_idx)
        torch.cuda.manual_seed_all(args.val_noise_seed + batch_idx)
        noisy_c36 = channel_to_decoder_input(c36, channel, args.snr_db, args.channel_type)
        c12_hat = predictor(noisy_c36, args.snr_db)
        z_hat = pca_inverse(noisy_c36, c12_hat, mean, basis, coeff_std)
        x_hat = decoder(z_hat).clamp(0.0, 1.0)
        acc.update(x_hat, imgs, c12_hat=c12_hat, c12_gt=c12, c36_hat=noisy_c36, c36_gt=c36)

    row = {"method": "exp01_direct_c12_resnet", "epoch": float(epoch)}
    row.update(acc.finalize())
    pca_psnr = baseline_refs["pca36_zerofill"]["psnr"]
    full_psnr = baseline_refs["full_c48_jscc"]["psnr"]
    oracle_psnr = baseline_refs["oracle_c12"]["psnr"]
    row["gain_vs_pca36"] = row["psnr"] - pca_psnr
    row["gap_to_full"] = full_psnr - row["psnr"]
    row["gap_to_oracle"] = oracle_psnr - row["psnr"]
    profiles = {
        row["method"]: acc.pred_profile_sum / max(1, acc.seen),
        "ground_truth": acc.target_profile_sum / max(1, acc.seen),
    }
    return row, profiles


def run_exp01(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, float]], dict[str, np.ndarray]]:
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    cfg = build_jscc_config(args, device)
    enc_path, dec_path = checkpoint_paths(args)
    mean, basis, coeff_std, _pca_obj = load_pca_basis(Path(args.pca_basis).expanduser().resolve(), device, int(args.C))
    encoder, decoder = load_models(cfg, enc_path, dec_path)
    train_loader = make_train_loader(args.train_dir, args.crop_size, args.batch_size, args.num_workers, device.type == "cuda")
    val_loader = make_loader(args.val_dir, args.crop_size, args.val_batch_size, args.val_num_workers, device.type == "cuda")
    baseline_refs = load_baseline_refs(Path(args.baseline_metrics).expanduser().resolve())
    predictor = PCANullPredictor(in_ch=args.keep_ch + 1, hidden=args.hidden, out_ch=args.C - args.keep_ch, depth=args.depth).to(device)
    optimizer = optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    channel = Channel(cfg)
    rows: list[dict[str, float]] = []
    best_row: dict[str, float] | None = None
    best_profiles: dict[str, np.ndarray] = {}
    start = time.time()
    print(f"device={device} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"encoder={enc_path}", flush=True)
    print(f"decoder={dec_path}", flush=True)
    print(f"pca_basis={args.pca_basis}", flush=True)
    print(f"baseline_metrics={args.baseline_metrics}", flush=True)
    print(f"train_images={len(train_loader.dataset)} val_images={len(val_loader.dataset)}", flush=True)
    print(
        f"exp01 direct c12 predictor hidden={args.hidden} depth={args.depth} "
        f"epochs={args.epochs} batch={args.batch_size} lr={args.lr} lambda_c={args.lambda_c}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        epoch_start = time.time()
        loss_sum = 0.0
        img_loss_sum = 0.0
        c12_loss_sum = 0.0
        psnr_sum = 0.0
        seen = 0
        for step, (imgs, _labels) in enumerate(tqdm(train_loader, desc=f"train exp01 epoch {epoch}", dynamic_ncols=True), start=1):
            if args.max_train_steps > 0 and step > args.max_train_steps:
                break
            imgs = imgs.to(device, non_blocking=device.type == "cuda")
            with torch.no_grad():
                z, _ = encoder(imgs)
                c36, c12 = pca_project(z, mean, basis, coeff_std, args.keep_ch)
                noisy_c36 = channel_to_decoder_input(c36, channel, args.snr_db, args.channel_type)
            c12_hat = predictor(noisy_c36, args.snr_db)
            z_hat = pca_inverse(noisy_c36, c12_hat, mean, basis, coeff_std)
            x_hat = decoder(z_hat)
            img_loss = F.mse_loss(x_hat.float(), imgs.float())
            c12_loss = F.mse_loss(c12_hat.float(), c12.float())
            loss = img_loss + args.lambda_c * c12_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
            optimizer.step()

            bsz = int(imgs.shape[0])
            with torch.no_grad():
                x_hat_eval = x_hat.float().clamp(0.0, 1.0)
                mse_i = (x_hat_eval - imgs.float()).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
                psnr_i = 10.0 * torch.log10(1.0 / mse_i)
            loss_sum += float(loss.item()) * bsz
            img_loss_sum += float(img_loss.item()) * bsz
            c12_loss_sum += float(c12_loss.item()) * bsz
            psnr_sum += float(psnr_i.sum().item())
            seen += bsz

        do_val = epoch == 1 or epoch == args.epochs or epoch % args.val_every == 0
        msg = (
            f"epoch={epoch:03d} train_loss={loss_sum / max(1, seen):.8f} "
            f"img={img_loss_sum / max(1, seen):.8f} c12={c12_loss_sum / max(1, seen):.8f} "
            f"train_psnr={psnr_sum / max(1, seen):.4f} elapsed={time.time() - epoch_start:.1f}s"
        )
        if device.type == "cuda":
            msg += f" gpu_mem_mb={torch.cuda.max_memory_allocated() / 1024**2:.1f}"
        print(msg, flush=True)

        if do_val:
            row, profiles = validate_exp01(
                predictor,
                encoder,
                decoder,
                channel,
                val_loader,
                device,
                mean,
                basis,
                coeff_std,
                args,
                epoch,
                baseline_refs,
            )
            rows.append(row)
            print(
                f"val epoch={epoch:03d} psnr={row['psnr']:.4f} mse={row['mse']:.8f} "
                f"c12_mse={row['c12_mse']:.8f} gain_vs_pca36={row['gain_vs_pca36']:+.4f} "
                f"gap_to_full={row['gap_to_full']:+.4f} gap_to_oracle={row['gap_to_oracle']:+.4f} "
                f"radial_err={row['radial_err']:.6f} phase={row['phase_score']:.6f}",
                flush=True,
            )
            if best_row is None or row["psnr"] > best_row["psnr"]:
                best_row = dict(row)
                best_profiles = profiles
                torch.save(
                    {
                        "kind": "exp01_direct_c12_resnet",
                        "epoch": epoch,
                        "args": vars(args),
                        "model": predictor.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metrics": best_row,
                    },
                    out_dir / "best_checkpoint.pt",
                )
                print(f"saved_best epoch={epoch} psnr={row['psnr']:.4f}", flush=True)
            if best_row is not None:
                write_val_metrics(out_dir / "val_metrics.csv", rows)
                write_spectrum_metrics(out_dir / "spectrum_metrics.csv", best_profiles)
                write_predictor_summary(out_dir / "summary.txt", rows, args)

        torch.save(
            {
                "kind": "exp01_direct_c12_resnet_latest",
                "epoch": epoch,
                "args": vars(args),
                "model": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            out_dir / "latest_checkpoint.pt",
        )

    if best_row is None:
        raise RuntimeError("no validation rows produced")
    print(f"elapsed_sec={time.time() - start:.1f}", flush=True)
    print(f"best_epoch={int(best_row['epoch'])} best_psnr={best_row['psnr']:.4f}", flush=True)
    return rows, best_profiles


def make_zero_c12(c36: torch.Tensor, c: int, keep_ch: int) -> torch.Tensor:
    return c36.new_zeros((c36.shape[0], c - keep_ch, c36.shape[2], c36.shape[3]))


def predictor_forward_variant(
    predictor: nn.Module,
    noisy_c36: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    coeff_std: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    zero_c12 = make_zero_c12(noisy_c36, args.C, args.keep_ch)
    z_base48 = pca_inverse(noisy_c36, zero_c12, mean, basis, coeff_std)
    if args.mode in {"exp02", "exp04", "exp05"} and predictor_kind(args) == "pca_zbase_c12_resnet":
        c12_hat = predictor(noisy_c36, z_base48, args.snr_db)
        return noisy_c36, c12_hat, None, None
    if args.mode == "exp03" or (args.mode in {"exp04", "exp05"} and predictor_kind(args) == "pca_denoise_c36_predict_c12"):
        c36_hat, c12_hat, delta_c36, alpha = predictor(noisy_c36, z_base48, args.snr_db)
        return c36_hat, c12_hat, delta_c36, alpha
    if args.mode == "exp06":
        c12_hat = predictor(noisy_c36, z_base48, args.snr_db)
        return noisy_c36, c12_hat, None, None
    if args.mode == "exp07":
        c12_hat = predictor(noisy_c36, args.snr_db)
        return noisy_c36, c12_hat, None, None
    raise ValueError(args.mode)


@torch.inference_mode()
def validate_variant(
    predictor: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    channel: Channel,
    val_loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    basis: torch.Tensor,
    coeff_std: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
    baseline_refs: dict[str, dict[str, float]],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    predictor.eval()
    acc = MetricAccumulator(args.spectrum_bins)
    delta_num = 0.0
    delta_den = 0.0
    for batch_idx, (imgs, _labels) in enumerate(tqdm(val_loader, desc=f"val {args.mode} epoch {epoch}", dynamic_ncols=True)):
        if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
            break
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        z, _ = encoder(imgs)
        c36, c12 = pca_project(z, mean, basis, coeff_std, args.keep_ch)
        torch.manual_seed(args.val_noise_seed + batch_idx)
        torch.cuda.manual_seed_all(args.val_noise_seed + batch_idx)
        noisy_c36 = channel_to_decoder_input(c36, channel, args.snr_db, args.channel_type)
        c36_hat, c12_hat, delta_c36, alpha = predictor_forward_variant(predictor, noisy_c36, mean, basis, coeff_std, args)
        z_hat = pca_inverse(c36_hat, c12_hat, mean, basis, coeff_std)
        x_hat = decoder(z_hat).clamp(0.0, 1.0)
        acc.update(x_hat, imgs, c12_hat=c12_hat, c12_gt=c12, c36_hat=c36_hat, c36_gt=c36)
        if delta_c36 is not None:
            delta_num += float((alpha * delta_c36).float().square().sum().item())
            delta_den += float(noisy_c36.float().square().sum().item())

    method = method_name(args)
    row = {"method": method, "epoch": float(epoch)}
    row.update(acc.finalize())
    pca_psnr = baseline_refs["pca36_zerofill"]["psnr"]
    full_psnr = baseline_refs["full_c48_jscc"]["psnr"]
    oracle_psnr = baseline_refs["oracle_c12"]["psnr"]
    row["gain_vs_pca36"] = row["psnr"] - pca_psnr
    row["gap_to_full"] = full_psnr - row["psnr"]
    row["gap_to_oracle"] = oracle_psnr - row["psnr"]
    if delta_den > 0:
        row["delta_ratio"] = math.sqrt(delta_num / max(delta_den, 1e-30))
    profiles = {
        row["method"]: acc.pred_profile_sum / max(1, acc.seen),
        "ground_truth": acc.target_profile_sum / max(1, acc.seen),
    }
    return row, profiles


def run_variant(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, float]], dict[str, np.ndarray]]:
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    cfg = build_jscc_config(args, device)
    enc_path, dec_path = checkpoint_paths(args)
    mean, basis, coeff_std, _pca_obj = load_pca_basis(Path(args.pca_basis).expanduser().resolve(), device, int(args.C))
    encoder, decoder = load_models(cfg, enc_path, dec_path)
    train_loader = make_train_loader(args.train_dir, args.crop_size, args.batch_size, args.num_workers, device.type == "cuda")
    val_loader = make_loader(args.val_dir, args.crop_size, args.val_batch_size, args.val_num_workers, device.type == "cuda")
    baseline_refs = load_baseline_refs(Path(args.baseline_metrics).expanduser().resolve())
    kind_key = predictor_kind(args)
    if kind_key == "pca_zbase_c12_resnet":
        predictor: nn.Module = PCAZBasePredictor(
            in_ch=args.keep_ch + args.C + 1,
            hidden=args.hidden,
            out_ch=args.C - args.keep_ch,
            depth=args.depth,
        ).to(device)
        kind = method_name(args)
    elif kind_key == "pca_denoise_c36_predict_c12":
        predictor = PCADenoisePredictor(
            in_ch=args.keep_ch + args.C + 1,
            hidden=args.hidden,
            depth=args.depth,
            init_alpha=args.init_alpha,
        ).to(device)
        kind = method_name(args)
    elif kind_key == "freq_aware_pca_null_predictor":
        predictor = FreqAwarePCANullPredictor(
            in_ch=args.keep_ch + args.C + 1,
            hidden=args.hidden,
            out_ch=args.C - args.keep_ch,
            depth=args.depth,
        ).to(device)
        kind = method_name(args)
    elif kind_key == "ar_pca_residual_predictor":
        predictor = ARPCANullPredictor(
            keep_ch=args.keep_ch,
            out_ch=args.C - args.keep_ch,
            group_ch=args.ar_group_ch,
            hidden=args.hidden,
            depth=args.depth,
        ).to(device)
        kind = method_name(args)
    else:
        raise ValueError(args.mode)
    optimizer = optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    channel = Channel(cfg)
    rows, best_row, best_profiles, start_epoch = maybe_resume_predictor(out_dir, predictor, optimizer, args, device)
    start = time.time()
    print(f"device={device} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"encoder={enc_path}", flush=True)
    print(f"decoder={dec_path}", flush=True)
    print(f"pca_basis={args.pca_basis}", flush=True)
    print(f"baseline_metrics={args.baseline_metrics}", flush=True)
    print(f"train_images={len(train_loader.dataset)} val_images={len(val_loader.dataset)}", flush=True)
    print(
        f"{kind} hidden={args.hidden} depth={args.depth} epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.lr} lambda_c={args.lambda_c} "
        f"lambda_c36={args.lambda_c36} lambda_fft={args.lambda_fft} lambda_amp={args.lambda_amp} "
        f"amp_loss={args.amp_loss_type} init_alpha={args.init_alpha} kind={predictor_kind(args)}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        predictor.train()
        epoch_start = time.time()
        loss_sum = 0.0
        img_loss_sum = 0.0
        c12_loss_sum = 0.0
        c36_loss_sum = 0.0
        fft_loss_sum = 0.0
        amp_loss_sum = 0.0
        psnr_sum = 0.0
        delta_num = 0.0
        delta_den = 0.0
        seen = 0
        for step, (imgs, _labels) in enumerate(tqdm(train_loader, desc=f"train {args.mode} epoch {epoch}", dynamic_ncols=True), start=1):
            if args.max_train_steps > 0 and step > args.max_train_steps:
                break
            imgs = imgs.to(device, non_blocking=device.type == "cuda")
            with torch.no_grad():
                z, _ = encoder(imgs)
                c36, c12 = pca_project(z, mean, basis, coeff_std, args.keep_ch)
                noisy_c36 = channel_to_decoder_input(c36, channel, args.snr_db, args.channel_type)
            c36_hat, c12_hat, delta_c36, alpha = predictor_forward_variant(predictor, noisy_c36, mean, basis, coeff_std, args)
            z_hat = pca_inverse(c36_hat, c12_hat, mean, basis, coeff_std)
            x_hat = decoder(z_hat)
            img_loss = F.mse_loss(x_hat.float(), imgs.float())
            c12_loss = F.mse_loss(c12_hat.float(), c12.float())
            c36_loss = (
                F.mse_loss(c36_hat.float(), c36.float())
                if predictor_kind(args) == "pca_denoise_c36_predict_c12"
                else c36_hat.new_tensor(0.0)
            )
            fft_loss = weighted_complex_fft_loss(x_hat, imgs, args.spectrum_bins) if args.lambda_fft > 0 else x_hat.new_tensor(0.0)
            if args.lambda_amp > 0:
                if args.amp_loss_type == "radial":
                    amp_loss = radial_log_power_loss(x_hat, imgs, args.spectrum_bins)
                elif args.amp_loss_type == "log_amp":
                    amp_loss = log_amplitude_fft_loss(x_hat, imgs, args.spectrum_bins)
                else:
                    raise ValueError(args.amp_loss_type)
            else:
                amp_loss = x_hat.new_tensor(0.0)
            loss = (
                img_loss
                + args.lambda_c * c12_loss
                + args.lambda_c36 * c36_loss
                + args.lambda_fft * fft_loss
                + args.lambda_amp * amp_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
            optimizer.step()

            bsz = int(imgs.shape[0])
            with torch.no_grad():
                x_hat_eval = x_hat.float().clamp(0.0, 1.0)
                mse_i = (x_hat_eval - imgs.float()).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
                psnr_i = 10.0 * torch.log10(1.0 / mse_i)
            loss_sum += float(loss.item()) * bsz
            img_loss_sum += float(img_loss.item()) * bsz
            c12_loss_sum += float(c12_loss.item()) * bsz
            c36_loss_sum += float(c36_loss.item()) * bsz
            fft_loss_sum += float(fft_loss.item()) * bsz
            amp_loss_sum += float(amp_loss.item()) * bsz
            psnr_sum += float(psnr_i.sum().item())
            if delta_c36 is not None and alpha is not None:
                delta_num += float((alpha.detach() * delta_c36.detach()).float().square().sum().item())
                delta_den += float(noisy_c36.detach().float().square().sum().item())
            seen += bsz

        delta_ratio = math.sqrt(delta_num / max(delta_den, 1e-30)) if delta_den > 0 else 0.0
        do_val = epoch == 1 or epoch == args.epochs or epoch % args.val_every == 0
        msg = (
            f"epoch={epoch:03d} train_loss={loss_sum / max(1, seen):.8f} "
            f"img={img_loss_sum / max(1, seen):.8f} c12={c12_loss_sum / max(1, seen):.8f} "
            f"c36={c36_loss_sum / max(1, seen):.8f} fft={fft_loss_sum / max(1, seen):.8f} "
            f"amp={amp_loss_sum / max(1, seen):.8f} train_psnr={psnr_sum / max(1, seen):.4f} "
            f"delta_ratio={delta_ratio:.6f} elapsed={time.time() - epoch_start:.1f}s"
        )
        if predictor_kind(args) == "pca_denoise_c36_predict_c12":
            msg += f" alpha={float(predictor.alpha.detach().item()):.6f}"  # type: ignore[attr-defined]
        if device.type == "cuda":
            msg += f" gpu_mem_mb={torch.cuda.max_memory_allocated() / 1024**2:.1f}"
        print(msg, flush=True)

        if do_val:
            row, profiles = validate_variant(
                predictor,
                encoder,
                decoder,
                channel,
                val_loader,
                device,
                mean,
                basis,
                coeff_std,
                args,
                epoch,
                baseline_refs,
            )
            rows.append(row)
            delta_part = f" delta_ratio={row['delta_ratio']:.6f}" if "delta_ratio" in row else ""
            print(
                f"val epoch={epoch:03d} psnr={row['psnr']:.4f} mse={row['mse']:.8f} "
                f"c12_mse={row['c12_mse']:.8f} c36_mse={row['c36_mse']:.8f} "
                f"gain_vs_pca36={row['gain_vs_pca36']:+.4f} gap_to_full={row['gap_to_full']:+.4f} "
                f"gap_to_oracle={row['gap_to_oracle']:+.4f} radial_err={row['radial_err']:.6f} "
                f"phase={row['phase_score']:.6f}{delta_part}",
                flush=True,
            )
            if best_row is None or row["psnr"] > best_row["psnr"]:
                best_row = dict(row)
                best_profiles = profiles
                torch.save(
                    {
                        "kind": kind,
                        "epoch": epoch,
                        "args": vars(args),
                        "model": predictor.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metrics": best_row,
                    },
                    out_dir / "best_checkpoint.pt",
                )
                print(f"saved_best epoch={epoch} psnr={row['psnr']:.4f}", flush=True)
            if best_row is not None:
                write_val_metrics(out_dir / "val_metrics.csv", rows)
                write_spectrum_metrics(out_dir / "spectrum_metrics.csv", best_profiles)
                write_predictor_summary(out_dir / "summary.txt", rows, args)

        torch.save(
            {
                "kind": f"{kind}_latest",
                "epoch": epoch,
                "args": vars(args),
                "model": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            out_dir / "latest_checkpoint.pt",
        )

    if best_row is None:
        raise RuntimeError("no validation rows produced")
    print(f"elapsed_sec={time.time() - start:.1f}", flush=True)
    print(f"best_epoch={int(best_row['epoch'])} best_psnr={best_row['psnr']:.4f}", flush=True)
    return rows, best_profiles


def make_diffusion_schedule(steps: int, device: torch.device) -> torch.Tensor:
    betas = torch.linspace(1e-4, 2e-2, steps, device=device)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


@torch.inference_mode()
def sample_diffusion_c12(
    predictor: DiffusionC12Predictor,
    noisy_c36: torch.Tensor,
    z_base48: torch.Tensor,
    args: argparse.Namespace,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    predictor.eval()
    c12_t = torch.randn(
        noisy_c36.shape[0],
        args.C - args.keep_ch,
        noisy_c36.shape[2],
        noisy_c36.shape[3],
        device=noisy_c36.device,
        dtype=noisy_c36.dtype,
    )
    t_values = torch.linspace(args.diffusion_steps - 1, 0, args.diffusion_sample_steps, device=noisy_c36.device).round().long()
    for idx, t in enumerate(t_values):
        t_batch = torch.full((noisy_c36.shape[0],), int(t.item()), device=noisy_c36.device, dtype=torch.long)
        t_norm = t_batch.float() / float(max(1, args.diffusion_steps - 1))
        eps_pred = predictor(c12_t, noisy_c36, z_base48, t_norm, args.snr_db)
        ab_t = alpha_bar[t_batch].view(-1, 1, 1, 1)
        x0 = (c12_t - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(ab_t).clamp_min(1e-8)
        if idx == len(t_values) - 1:
            c12_t = x0
        else:
            next_t = t_values[idx + 1]
            ab_next = alpha_bar[next_t].view(1, 1, 1, 1)
            c12_t = torch.sqrt(ab_next) * x0 + torch.sqrt(1.0 - ab_next) * eps_pred
    return c12_t


@torch.inference_mode()
def validate_diffusion(
    predictor: DiffusionC12Predictor,
    encoder: nn.Module,
    decoder: nn.Module,
    channel: Channel,
    val_loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    basis: torch.Tensor,
    coeff_std: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
    baseline_refs: dict[str, dict[str, float]],
    alpha_bar: torch.Tensor,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    predictor.eval()
    acc = MetricAccumulator(args.spectrum_bins)
    for batch_idx, (imgs, _labels) in enumerate(tqdm(val_loader, desc=f"val exp08 epoch {epoch}", dynamic_ncols=True)):
        if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
            break
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        z, _ = encoder(imgs)
        c36, c12 = pca_project(z, mean, basis, coeff_std, args.keep_ch)
        torch.manual_seed(args.val_noise_seed + batch_idx)
        torch.cuda.manual_seed_all(args.val_noise_seed + batch_idx)
        noisy_c36 = channel_to_decoder_input(c36, channel, args.snr_db, args.channel_type)
        z_base48 = pca_inverse(noisy_c36, make_zero_c12(noisy_c36, args.C, args.keep_ch), mean, basis, coeff_std)
        torch.manual_seed(args.val_noise_seed + 100000 + batch_idx)
        torch.cuda.manual_seed_all(args.val_noise_seed + 100000 + batch_idx)
        c12_hat = sample_diffusion_c12(predictor, noisy_c36, z_base48, args, alpha_bar)
        z_hat = pca_inverse(noisy_c36, c12_hat, mean, basis, coeff_std)
        x_hat = decoder(z_hat).clamp(0.0, 1.0)
        acc.update(x_hat, imgs, c12_hat=c12_hat, c12_gt=c12, c36_hat=noisy_c36, c36_gt=c36)
    row = {"method": method_name(args), "epoch": float(epoch)}
    row.update(acc.finalize())
    pca_psnr = baseline_refs["pca36_zerofill"]["psnr"]
    full_psnr = baseline_refs["full_c48_jscc"]["psnr"]
    oracle_psnr = baseline_refs["oracle_c12"]["psnr"]
    row["gain_vs_pca36"] = row["psnr"] - pca_psnr
    row["gap_to_full"] = full_psnr - row["psnr"]
    row["gap_to_oracle"] = oracle_psnr - row["psnr"]
    profiles = {
        row["method"]: acc.pred_profile_sum / max(1, acc.seen),
        "ground_truth": acc.target_profile_sum / max(1, acc.seen),
    }
    return row, profiles


def run_diffusion(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, float]], dict[str, np.ndarray]]:
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    cfg = build_jscc_config(args, device)
    enc_path, dec_path = checkpoint_paths(args)
    mean, basis, coeff_std, _pca_obj = load_pca_basis(Path(args.pca_basis).expanduser().resolve(), device, int(args.C))
    encoder, decoder = load_models(cfg, enc_path, dec_path)
    train_loader = make_train_loader(args.train_dir, args.crop_size, args.batch_size, args.num_workers, device.type == "cuda")
    val_loader = make_loader(args.val_dir, args.crop_size, args.val_batch_size, args.val_num_workers, device.type == "cuda")
    baseline_refs = load_baseline_refs(Path(args.baseline_metrics).expanduser().resolve())
    predictor = DiffusionC12Predictor(
        in_ch=(args.C - args.keep_ch) + args.keep_ch + args.C + 2,
        hidden=args.hidden,
        out_ch=args.C - args.keep_ch,
        depth=args.depth,
    ).to(device)
    optimizer = optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    channel = Channel(cfg)
    alpha_bar = make_diffusion_schedule(args.diffusion_steps, device)
    rows, best_row, best_profiles, start_epoch = maybe_resume_predictor(out_dir, predictor, optimizer, args, device)
    start = time.time()
    print(f"device={device} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"encoder={enc_path}", flush=True)
    print(f"decoder={dec_path}", flush=True)
    print(f"pca_basis={args.pca_basis}", flush=True)
    print(f"baseline_metrics={args.baseline_metrics}", flush=True)
    print(f"train_images={len(train_loader.dataset)} val_images={len(val_loader.dataset)}", flush=True)
    print(
        f"{method_name(args)} hidden={args.hidden} depth={args.depth} epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.lr} diffusion_steps={args.diffusion_steps} "
        f"sample_steps={args.diffusion_sample_steps}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        predictor.train()
        epoch_start = time.time()
        loss_sum = 0.0
        seen = 0
        for step, (imgs, _labels) in enumerate(tqdm(train_loader, desc=f"train exp08 epoch {epoch}", dynamic_ncols=True), start=1):
            if args.max_train_steps > 0 and step > args.max_train_steps:
                break
            imgs = imgs.to(device, non_blocking=device.type == "cuda")
            with torch.no_grad():
                z, _ = encoder(imgs)
                c36, c12 = pca_project(z, mean, basis, coeff_std, args.keep_ch)
                noisy_c36 = channel_to_decoder_input(c36, channel, args.snr_db, args.channel_type)
                z_base48 = pca_inverse(noisy_c36, make_zero_c12(noisy_c36, args.C, args.keep_ch), mean, basis, coeff_std)
            bsz = int(imgs.shape[0])
            t = torch.randint(0, args.diffusion_steps, (bsz,), device=device)
            eps = torch.randn_like(c12)
            ab_t = alpha_bar[t].view(-1, 1, 1, 1)
            c12_t = torch.sqrt(ab_t) * c12 + torch.sqrt(1.0 - ab_t) * eps
            t_norm = t.float() / float(max(1, args.diffusion_steps - 1))
            eps_pred = predictor(c12_t, noisy_c36, z_base48, t_norm, args.snr_db)
            loss = F.mse_loss(eps_pred.float(), eps.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += float(loss.item()) * bsz
            seen += bsz

        do_val = epoch == 1 or epoch == args.epochs or epoch % args.val_every == 0
        msg = f"epoch={epoch:03d} train_eps_loss={loss_sum / max(1, seen):.8f} elapsed={time.time() - epoch_start:.1f}s"
        if device.type == "cuda":
            msg += f" gpu_mem_mb={torch.cuda.max_memory_allocated() / 1024**2:.1f}"
        print(msg, flush=True)
        if do_val:
            row, profiles = validate_diffusion(
                predictor,
                encoder,
                decoder,
                channel,
                val_loader,
                device,
                mean,
                basis,
                coeff_std,
                args,
                epoch,
                baseline_refs,
                alpha_bar,
            )
            rows.append(row)
            print(
                f"val epoch={epoch:03d} psnr={row['psnr']:.4f} mse={row['mse']:.8f} "
                f"c12_mse={row['c12_mse']:.8f} gain_vs_pca36={row['gain_vs_pca36']:+.4f} "
                f"gap_to_full={row['gap_to_full']:+.4f} gap_to_oracle={row['gap_to_oracle']:+.4f} "
                f"radial_err={row['radial_err']:.6f} phase={row['phase_score']:.6f}",
                flush=True,
            )
            if best_row is None or row["psnr"] > best_row["psnr"]:
                best_row = dict(row)
                best_profiles = profiles
                torch.save(
                    {
                        "kind": method_name(args),
                        "epoch": epoch,
                        "args": vars(args),
                        "model": predictor.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metrics": best_row,
                    },
                    out_dir / "best_checkpoint.pt",
                )
                print(f"saved_best epoch={epoch} psnr={row['psnr']:.4f}", flush=True)
            if best_row is not None:
                write_val_metrics(out_dir / "val_metrics.csv", rows)
                write_spectrum_metrics(out_dir / "spectrum_metrics.csv", best_profiles)
                write_predictor_summary(out_dir / "summary.txt", rows, args)

        torch.save(
            {
                "kind": f"{method_name(args)}_latest",
                "epoch": epoch,
                "args": vars(args),
                "model": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            out_dir / "latest_checkpoint.pt",
        )

    if best_row is None:
        raise RuntimeError("no validation rows produced")
    print(f"elapsed_sec={time.time() - start:.1f}", flush=True)
    print(f"best_epoch={int(best_row['epoch'])} best_psnr={best_row['psnr']:.4f}", flush=True)
    return rows, best_profiles


def write_val_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    fields = [
        "method",
        "epoch",
        "psnr",
        "mse",
        "c12_mse",
        "c36_mse",
        "radial_err",
        "complex_fft_err",
        "phase_score",
        "hf_power_ratio",
        "gain_vs_pca36",
        "gap_to_full",
        "gap_to_oracle",
        "delta_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_spectrum_metrics(path: Path, profiles: dict[str, np.ndarray]) -> None:
    methods = list(profiles.keys())
    bins = len(next(iter(profiles.values())))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin", "freq_min_norm", "freq_max_norm"] + [f"{m}_luma_power" for m in methods])
        for i in range(bins):
            row = [i, i / bins, (i + 1) / bins]
            row.extend(float(profiles[m][i]) for m in methods)
            writer.writerow(row)


def write_summary(path: Path, rows: list[dict[str, float]], args: argparse.Namespace) -> None:
    by = {r["method"]: r for r in rows}
    lines = [
        "PCA36/null-space JSCC baselines at SNR=9 dB",
        "",
        f"JSCC checkpoint root: {Path(args.ckpt_root).resolve()}",
        f"PCA basis: {Path(args.pca_basis).resolve()}",
        "PCA coefficients: whitened by sqrt(eigvals) before channel/predictor; unwhitened before inverse PCA",
        f"Validation: {Path(args.val_dir).resolve()}",
        f"Validation AWGN seed: {args.val_noise_seed}",
        "",
        f"Full C48 JSCC PSNR: {by['full_c48_jscc']['psnr']:.4f} dB",
        f"PCA36 zero-fill PSNR: {by['pca36_zerofill']['psnr']:.4f} dB",
        f"PCA36 clean upper reference PSNR: {by['pca36_clean_upper']['psnr']:.4f} dB",
        f"PCA36 oracle c12 PSNR: {by['oracle_c12']['psnr']:.4f} dB",
        "",
        "Conclusion:",
        (
            "The predictor target is valid if future experiments improve over PCA36 zero-fill "
            "and reduce the gap to the oracle c12 reference without modifying transmitted c36."
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def write_predictor_summary(path: Path, rows: list[dict[str, float]], args: argparse.Namespace) -> None:
    best = max(rows, key=lambda r: r["psnr"])
    lines = [
        METHOD_TITLES.get(args.mode, f"{args.mode} at SNR=9 dB"),
        "",
        f"JSCC checkpoint root: {Path(args.ckpt_root).resolve()}",
        f"PCA basis: {Path(args.pca_basis).resolve()}",
        "PCA coefficients: whitened by sqrt(eigvals) before channel/predictor; unwhitened before inverse PCA",
        f"Baseline metrics: {Path(args.baseline_metrics).resolve()}",
        f"Validation AWGN seed: {args.val_noise_seed}",
        f"Experiment name: {args.exp_name}",
        f"Predictor kind: {predictor_kind(args) if args.mode != 'exp08' else 'diffusion_c12_only'}",
        f"Loss weights: lambda_c={args.lambda_c}, lambda_c36={args.lambda_c36}, lambda_fft={args.lambda_fft}, lambda_amp={args.lambda_amp}",
        "",
        f"Best epoch: {int(best['epoch'])}",
        f"Best PSNR: {best['psnr']:.4f} dB",
        f"Gain vs PCA36 zero-fill: {best['gain_vs_pca36']:+.4f} dB",
        f"Gap to full C48: {best['gap_to_full']:+.4f} dB",
        f"Gap to oracle c12: {best['gap_to_oracle']:+.4f} dB",
        f"c12 MSE: {best['c12_mse']:.8f}",
        f"Radial spectrum error: {best['radial_err']:.6f}",
        f"Complex FFT error: {best['complex_fft_err']:.6f}",
        f"Phase consistency: {best['phase_score']:.6f}",
        f"High-frequency power ratio: {best['hf_power_ratio']:.6f}",
        f"Delta c36 ratio: {best.get('delta_ratio', math.nan):.8f}",
        "",
        "Conclusion:",
        "This experiment preserves the requested PCA/AWGN/inverse-PCA contract and reports the best validation checkpoint. Visual artifact review is not automated; use the metric tradeoff to screen perceptual variants.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCA null-space residual experiments for C48/SNR9 JSCC.")
    p.add_argument(
        "--mode",
        choices=("baselines", "exp01", "exp02", "exp03", "exp04", "exp05", "exp06", "exp07", "exp08"),
        default="baselines",
    )
    p.add_argument("--train-dir", default="/workspace/yongjia/datasets/DIV2K/DIV2K_train_HR")
    p.add_argument("--val-dir", default="/workspace/yongjia/datasets/DIV2K/DIV2K_valid_HR")
    p.add_argument("--ckpt-root", default=str(CDDM_ROOT / "MY" / "checkpoints-jscc"))
    p.add_argument("--pca-basis", default=str(CDDM_ROOT / "MY" / "pca" / "pca_basis_train_snr9_c48.pt"))
    p.add_argument("--out-root", default=str(CDDM_ROOT / "MY" / "checkpoints-pca-c48-snr9"))
    p.add_argument("--baseline-metrics", default=str(CDDM_ROOT / "MY" / "checkpoints-pca-c48-snr9" / "baselines" / "val_metrics.csv"))
    p.add_argument("--exp-name", default="baselines")
    p.add_argument("--C", type=int, default=48)
    p.add_argument("--keep-ch", type=int, default=36)
    p.add_argument("--snr-db", type=float, default=9.0)
    p.add_argument("--channel-type", choices=("awgn", "rayleigh"), default="awgn")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--spectrum-bins", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-c", type=float, default=0.1)
    p.add_argument("--lambda-c36", type=float, default=0.05)
    p.add_argument("--lambda-fft", type=float, default=0.0)
    p.add_argument("--lambda-amp", type=float, default=0.0)
    p.add_argument("--amp-loss-type", choices=("radial", "log_amp"), default="radial")
    p.add_argument("--predictor-base", choices=("exp02", "exp03"), default="exp02")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--init-alpha", type=float, default=0.1)
    p.add_argument("--ar-group-ch", type=int, default=4)
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--diffusion-sample-steps", type=int, default=25)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--val-noise-seed", type=int, default=20260603)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_root).expanduser().resolve() / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    log_path = out_dir / "train.log"
    resume_log = args.mode != "baselines" and (out_dir / "latest_checkpoint.pt").is_file() and (out_dir / "val_metrics.csv").is_file()
    with log_path.open("a" if resume_log else "w", buffering=1) as log_f:
        original_print = builtins.print
        def tee_print(*a, **k):
            if "file" in k:
                original_print(*a, **k)
            else:
                original_print(*a, file=TeeStream(sys.__stdout__, log_f), **k)

        builtins.print = tee_print
        try:
            if resume_log:
                print("", flush=True)
                print("=== resume run ===", flush=True)
            print(f"start_time={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            if args.mode == "baselines":
                rows, profiles = run_baselines(args, out_dir)
            elif args.mode == "exp01":
                rows, profiles = run_exp01(args, out_dir)
            elif args.mode in {"exp02", "exp03", "exp04", "exp05", "exp06", "exp07"}:
                rows, profiles = run_variant(args, out_dir)
            elif args.mode == "exp08":
                rows, profiles = run_diffusion(args, out_dir)
            else:
                raise ValueError(args.mode)
            write_val_metrics(out_dir / "val_metrics.csv", rows)
            write_spectrum_metrics(out_dir / "spectrum_metrics.csv", profiles)
            if args.mode == "baselines":
                write_summary(out_dir / "summary.txt", rows, args)
            else:
                write_predictor_summary(out_dir / "summary.txt", rows, args)
            print(f"outputs={out_dir}", flush=True)
        finally:
            builtins.print = original_print


if __name__ == "__main__":
    main()
