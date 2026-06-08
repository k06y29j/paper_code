#!/usr/bin/env python3
from __future__ import annotations

import argparse
import builtins
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_pca_null_residual import (
    Channel,
    MetricAccumulator,
    ResBlock,
    TeeStream,
    build_jscc_config,
    channel_to_decoder_input,
    checkpoint_paths,
    load_baseline_refs,
    load_models,
    load_pca_basis,
    make_loader,
    make_train_loader,
    make_zero_c12,
    pca_inverse,
    pca_project,
    radial_bins,
    seed_all,
    write_spectrum_metrics,
)


def channel_weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((pred.float() - target.float()).square() * weights).mean()


def latent_log_power_loss(z_hat: torch.Tensor, z_target: torch.Tensor, bins: int) -> torch.Tensor:
    pred = z_hat.float() - z_hat.float().mean(dim=(-2, -1), keepdim=True)
    target = z_target.float() - z_target.float().mean(dim=(-2, -1), keepdim=True)
    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred, norm="ortho", dim=(-2, -1)), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target, norm="ortho", dim=(-2, -1)), dim=(-2, -1))
    bin_idx, weight = radial_bins(pred.shape[-2], pred.shape[-1], bins, pred.device)
    pred_power = pred_fft.abs().square()
    target_power = target_fft.abs().square()
    pred_bins = pred_power.new_zeros((pred.shape[0], pred.shape[1], bins))
    target_bins = target_power.new_zeros((target.shape[0], target.shape[1], bins))
    expand_idx = bin_idx.flatten().view(1, 1, -1).expand(pred.shape[0], pred.shape[1], -1)
    pred_bins.scatter_add_(2, expand_idx, pred_power.flatten(2))
    target_bins.scatter_add_(2, expand_idx, target_power.flatten(2))
    counts = torch.bincount(bin_idx.flatten(), minlength=bins).to(pred.device).float().clamp_min(1.0)
    pred_bins = pred_bins / counts.view(1, 1, -1)
    target_bins = target_bins / counts.view(1, 1, -1)
    weights = torch.linspace(float(weight.min()), float(weight.max()), bins, device=pred.device)
    return (weights.view(1, 1, -1) * (torch.log(pred_bins + 1e-8) - torch.log(target_bins + 1e-8)).abs()).mean()


def image_log_power_loss(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    bins: int = 64,
    eps: float = 1e-8,
    high_freq_power: float = 1.5,
    min_w: float = 0.2,
    max_w: float = 3.0,
) -> torch.Tensor:
    x_hat = x_hat.float()
    x_gt = x_gt.float()
    pred = 0.299 * x_hat[:, 0] + 0.587 * x_hat[:, 1] + 0.114 * x_hat[:, 2]
    target = 0.299 * x_gt[:, 0] + 0.587 * x_gt[:, 1] + 0.114 * x_gt[:, 2]
    pred = pred - pred.mean(dim=(-2, -1), keepdim=True)
    target = target - target.mean(dim=(-2, -1), keepdim=True)
    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred, norm="ortho"), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target, norm="ortho"), dim=(-2, -1))
    pred_power = pred_fft.abs().square()
    target_power = target_fft.abs().square()
    batch, height, width = pred.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=x_hat.device),
        torch.arange(width, device=x_hat.device),
        indexing="ij",
    )
    cy, cx = height // 2, width // 2
    radius = torch.sqrt((yy.float() - cy) ** 2 + (xx.float() - cx) ** 2)
    radius = radius / radius.max().clamp_min(eps)
    bin_idx = torch.clamp((radius * bins).long(), max=bins - 1)
    flat_idx = bin_idx.flatten().unsqueeze(0).expand(batch, -1)
    pred_bins = pred_power.new_zeros(batch, bins)
    target_bins = target_power.new_zeros(batch, bins)
    pred_bins.scatter_add_(1, flat_idx, pred_power.flatten(1))
    target_bins.scatter_add_(1, flat_idx, target_power.flatten(1))
    counts = torch.bincount(bin_idx.flatten(), minlength=bins).to(x_hat.device).float().clamp_min(1.0)
    pred_bins = pred_bins / counts.unsqueeze(0)
    target_bins = target_bins / counts.unsqueeze(0)
    centers = (torch.arange(bins, device=x_hat.device).float() + 0.5) / bins
    weights = min_w + (max_w - min_w) * centers.pow(high_freq_power)
    return (weights.unsqueeze(0) * (torch.log(pred_bins + eps) - torch.log(target_bins + eps)).abs()).mean()


def clean_t_weight(t: torch.Tensor, diffusion_steps: int, cutoff: float = 0.35, power: float = 2.0) -> torch.Tensor:
    t_norm = t.float() / float(max(1, diffusion_steps - 1))
    weight = (cutoff - t_norm).clamp_min(0.0) / max(cutoff, 1e-8)
    return weight.pow(power)


def make_schedule(steps: int, kind: str, device: torch.device) -> torch.Tensor:
    if kind == "linear":
        betas = torch.linspace(1e-4, 2e-2, steps, device=device)
        return torch.cumprod(1.0 - betas, dim=0).clamp_min(1e-8)
    if kind == "cosine":
        s = 0.008
        x = torch.linspace(0, steps, steps + 1, device=device)
        alpha = torch.cos(((x / steps) + s) / (1.0 + s) * math.pi * 0.5).square()
        alpha = alpha / alpha[0]
        return alpha[1:].clamp_min(1e-8)
    raise ValueError(kind)


def load_channel_weights(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    out_ch = args.C - args.keep_ch
    if not args.channel_r2_csv:
        return torch.ones((1, out_ch, 1, 1), device=device)
    path = Path(args.channel_r2_csv).expanduser().resolve()
    rows = list(csv.DictReader(path.open()))
    score_by_component = {int(float(r["component_1based"])): float(r[args.weight_score_col]) for r in rows}
    scores = torch.tensor(
        [max(0.0, score_by_component.get(args.keep_ch + i + 1, 0.0)) for i in range(out_ch)],
        dtype=torch.float32,
        device=device,
    )
    if float(scores.max().item()) <= 0.0:
        raw = torch.ones_like(scores)
    else:
        norm = (scores / scores.max().clamp_min(1e-12)).pow(args.weight_power)
        raw = args.unpredictable_weight + (args.predictable_weight_max - args.unpredictable_weight) * norm
    if args.normalize_channel_weights:
        raw = raw / raw.mean().clamp_min(1e-12)
    return raw.view(1, out_ch, 1, 1)


class FrequencyBandGate(nn.Module):
    def __init__(
        self,
        latent_ch: int = 48,
        hidden: int = 192,
        bands: int = 8,
        use_phase: bool = True,
        gate_scale: float = 0.1,
    ):
        super().__init__()
        self.latent_ch = latent_ch
        self.bands = bands
        self.use_phase = use_phase
        self.gate_scale = gate_scale
        in_dim = latent_ch * bands * (3 if use_phase else 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _band_index(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        cy, cx = height // 2, width // 2
        radius = torch.sqrt((yy.float() - cy) ** 2 + (xx.float() - cx) ** 2)
        radius = radius / radius.max().clamp_min(1e-8)
        return torch.clamp((radius * self.bands).long(), max=self.bands - 1)

    def forward(self, feat: torch.Tensor, z_base48: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = z_base48.shape
        if channels != self.latent_ch:
            raise ValueError(f"FrequencyBandGate expected {self.latent_ch} latent channels, got {channels}")
        fft = torch.fft.fftshift(
            torch.fft.fft2(z_base48.float(), norm="ortho", dim=(-2, -1)),
            dim=(-2, -1),
        )
        amp = fft.abs()
        power = amp.square()
        band_idx = self._band_index(height, width, z_base48.device)
        flat_idx = band_idx.flatten().view(1, 1, -1).expand(batch, channels, -1)
        counts = torch.bincount(band_idx.flatten(), minlength=self.bands).to(z_base48.device).float().clamp_min(1.0)

        power_bins = power.new_zeros(batch, channels, self.bands)
        power_bins.scatter_add_(2, flat_idx, power.flatten(2))
        power_bins = power_bins / counts.view(1, 1, -1)
        log_power = torch.log(power_bins + 1e-8)
        log_power = log_power - log_power.mean(dim=(1, 2), keepdim=True)
        log_power = log_power / (log_power.std(dim=(1, 2), keepdim=True) + 1e-6)
        features = [log_power]

        if self.use_phase:
            reliability = amp / (amp.mean(dim=(-2, -1), keepdim=True) + amp + 1e-6)
            cos_phase = (fft.real / (amp + 1e-6)) * reliability
            sin_phase = (fft.imag / (amp + 1e-6)) * reliability
            cos_bins = power.new_zeros(batch, channels, self.bands)
            sin_bins = power.new_zeros(batch, channels, self.bands)
            cos_bins.scatter_add_(2, flat_idx, cos_phase.flatten(2))
            sin_bins.scatter_add_(2, flat_idx, sin_phase.flatten(2))
            features.extend([cos_bins / counts.view(1, 1, -1), sin_bins / counts.view(1, 1, -1)])

        gate = self.mlp(torch.cat(features, dim=1).flatten(1)).view(batch, -1, 1, 1)
        return feat * (1.0 + self.gate_scale * torch.tanh(gate))


class ConditionStem(nn.Module):
    def __init__(self, in_ch: int, hidden: int, depth: int, out_ch: int, freq_gate: bool):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1), nn.GELU())
        self.body = nn.Sequential(*[ResBlock(hidden) for _ in range(depth)])
        self.freq_gate = None
        if freq_gate:
            self.freq_gate = FrequencyBandGate(latent_ch=48, hidden=hidden, bands=8, use_phase=True, gate_scale=0.1)
        self.head = nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def apply_gate(self, feat: torch.Tensor, z_base48: torch.Tensor) -> torch.Tensor:
        if self.freq_gate is None:
            return feat
        return self.freq_gate(feat, z_base48)

    def forward_body(self, x: torch.Tensor, z_base48: torch.Tensor) -> torch.Tensor:
        feat = self.body(self.stem(x))
        return self.apply_gate(feat, z_base48)


class DirectPriorPredictor(nn.Module):
    def __init__(self, keep_ch: int, latent_ch: int, hidden: int, depth: int, out_ch: int, freq_gate: bool):
        super().__init__()
        self.net = ConditionStem(keep_ch + latent_ch + 1, hidden, depth, out_ch, freq_gate)

    def forward(self, noisy_c36: torch.Tensor, z_base48: torch.Tensor, snr_value: float) -> torch.Tensor:
        snr_map = noisy_c36.new_full((noisy_c36.shape[0], 1, noisy_c36.shape[2], noisy_c36.shape[3]), float(snr_value) / 20.0)
        x = torch.cat([noisy_c36, z_base48, snr_map], dim=1)
        return self.net.head(self.net.forward_body(x, z_base48))


class ResidualDDNMPredictor(nn.Module):
    def __init__(self, keep_ch: int, latent_ch: int, hidden: int, depth: int, out_ch: int, freq_gate: bool):
        super().__init__()
        self.net = ConditionStem(out_ch + keep_ch + latent_ch + 2, hidden, depth, out_ch, freq_gate)

    def forward(self, r_t: torch.Tensor, noisy_c36: torch.Tensor, z_base48: torch.Tensor, t_norm: torch.Tensor, snr_value: float) -> torch.Tensor:
        snr_map = noisy_c36.new_full((noisy_c36.shape[0], 1, noisy_c36.shape[2], noisy_c36.shape[3]), float(snr_value) / 20.0)
        t_map = t_norm.view(-1, 1, 1, 1).expand(-1, 1, noisy_c36.shape[2], noisy_c36.shape[3]).to(noisy_c36.dtype)
        x = torch.cat([r_t, noisy_c36, z_base48, snr_map, t_map], dim=1)
        return self.net.head(self.net.forward_body(x, z_base48))


@torch.inference_mode()
def sample_residual_ddim(
    ddnm: ResidualDDNMPredictor,
    noisy_c36: torch.Tensor,
    z_base48: torch.Tensor,
    args: argparse.Namespace,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    if args.sample_init == "zero":
        r_t = noisy_c36.new_zeros((noisy_c36.shape[0], args.C - args.keep_ch, noisy_c36.shape[2], noisy_c36.shape[3]))
    elif args.sample_init == "noise":
        r_t = torch.randn(noisy_c36.shape[0], args.C - args.keep_ch, noisy_c36.shape[2], noisy_c36.shape[3], device=noisy_c36.device, dtype=noisy_c36.dtype)
    else:
        raise ValueError(args.sample_init)
    t_values = torch.linspace(args.diffusion_steps - 1, 0, args.diffusion_sample_steps, device=noisy_c36.device).round().long()
    for idx, t in enumerate(t_values):
        t_batch = torch.full((noisy_c36.shape[0],), int(t.item()), device=noisy_c36.device, dtype=torch.long)
        t_norm = t_batch.float() / float(max(1, args.diffusion_steps - 1))
        eps_pred = ddnm(r_t, noisy_c36, z_base48, t_norm, args.snr_db)
        ab_t = alpha_bar[t_batch].view(-1, 1, 1, 1)
        r0 = (r_t - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(ab_t).clamp_min(1e-8)
        if args.clip_x0 > 0:
            r0 = r0.clamp(-args.clip_x0, args.clip_x0)
        if idx == len(t_values) - 1:
            r_t = r0
        else:
            next_t = t_values[idx + 1]
            ab_next = alpha_bar[next_t].view(1, 1, 1, 1)
            r_t = torch.sqrt(ab_next) * r0 + torch.sqrt(1.0 - ab_next) * eps_pred
    return r_t


@torch.inference_mode()
def sample_pca48_ddnm_plus(
    prior: DirectPriorPredictor,
    model: ResidualDDNMPredictor,
    noisy_c36: torch.Tensor,
    z_base48: torch.Tensor,
    args: argparse.Namespace,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    batch, _keep_ch, height, width = noisy_c36.shape
    latent_ch = args.C
    keep_ch = args.keep_ch
    out_ch = latent_ch - keep_ch
    start_t = int(getattr(args, "sample_start_t", 50))
    start_t = max(0, min(start_t, args.diffusion_steps - 1))
    temperature = float(getattr(args, "sample_temperature", 0.1))

    mu = prior(noisy_c36, z_base48, args.snr_db)
    x_anchor = torch.cat(
        [noisy_c36, noisy_c36.new_zeros(batch, out_ch, height, width)],
        dim=1,
    )
    ab_start = alpha_bar[start_t].view(1, 1, 1, 1)
    noise = torch.randn_like(x_anchor)
    x_t = torch.sqrt(ab_start) * x_anchor + torch.sqrt(1.0 - ab_start) * temperature * noise
    t_values = torch.linspace(start_t, 0, args.diffusion_sample_steps, device=noisy_c36.device).round().long().unique_consecutive()
    sigma_n2 = 1.0 / (10.0 ** (float(args.snr_db) / 10.0))

    for idx, t in enumerate(t_values):
        t_batch = torch.full((batch,), int(t.item()), device=noisy_c36.device, dtype=torch.long)
        t_norm = t_batch.float() / float(max(1, args.diffusion_steps - 1))
        ab_t = alpha_bar[t_batch].view(-1, 1, 1, 1)

        # A5 was trained on residual r_t = sqrt(alpha_bar_t) * (c12 - mu) + noise.
        r_t = x_t[:, keep_ch:] - torch.sqrt(ab_t) * mu
        eps_c12 = model(r_t, noisy_c36, z_base48, t_norm, args.snr_db)
        r0_pred = (r_t - torch.sqrt(1.0 - ab_t) * eps_c12) / torch.sqrt(ab_t).clamp_min(1e-8)
        c12_pred = mu + r0_pred

        c36_pred = x_t[:, :keep_ch] / torch.sqrt(ab_t).clamp_min(1e-8)
        if args.clip_x0 > 0:
            c36_pred = c36_pred.clamp(-args.clip_x0, args.clip_x0)
            c12_pred = c12_pred.clamp(-args.clip_x0, args.clip_x0)

        sigma_t2 = (1.0 - ab_t) / ab_t.clamp_min(1e-8)
        lam = sigma_t2 / (sigma_t2 + sigma_n2)
        c36_dc = lam * noisy_c36 + (1.0 - lam) * c36_pred
        x0_dc = torch.cat([c36_dc, c12_pred], dim=1)

        if idx == len(t_values) - 1:
            x_t = x0_dc
        else:
            next_t = t_values[idx + 1]
            ab_next = alpha_bar[next_t].view(1, 1, 1, 1)
            eps_full = torch.cat([torch.zeros_like(noisy_c36), eps_c12], dim=1)
            x_t = torch.sqrt(ab_next) * x0_dc + torch.sqrt(1.0 - ab_next) * eps_full
    return x_t[:, keep_ch:]


def batch_context(args, encoder, channel, imgs, mean, basis, coeff_std):
    z, _ = encoder(imgs)
    c36, c12 = pca_project(z, mean, basis, coeff_std, args.keep_ch)
    noisy_c36 = channel_to_decoder_input(c36, channel, args.snr_db, args.channel_type)
    z_base48 = pca_inverse(noisy_c36, make_zero_c12(noisy_c36, args.C, args.keep_ch), mean, basis, coeff_std)
    z_oracle = pca_inverse(noisy_c36, c12, mean, basis, coeff_std)
    return c36, c12, noisy_c36, z_base48, z_oracle


def train_prior_epoch(args, prior, encoder, decoder, channel, loader, optimizer, mean, basis, coeff_std, channel_weights, device):
    prior.train()
    loss_sum = c12_sum = img_sum = fft_sum = 0.0
    seen = 0
    for step, (imgs, _labels) in enumerate(tqdm(loader, desc="warmup direct prior", dynamic_ncols=True), start=1):
        if args.max_train_steps > 0 and step > args.max_train_steps:
            break
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        with torch.no_grad():
            _c36, c12, noisy_c36, z_base48, z_oracle = batch_context(args, encoder, channel, imgs, mean, basis, coeff_std)
        mu = prior(noisy_c36, z_base48, args.snr_db)
        c12_loss = channel_weighted_mse(mu, c12, channel_weights)
        z_hat = pca_inverse(noisy_c36, mu, mean, basis, coeff_std)
        x_hat_raw = decoder(z_hat) if args.lambda_img > 0 or args.lambda_fft > 0 else None
        img_loss = F.mse_loss(x_hat_raw.float(), imgs.float()) if args.lambda_img > 0 else mu.new_tensor(0.0)
        if args.lambda_fft > 0 and args.fft_domain == "image":
            fft_loss = image_log_power_loss(x_hat_raw, imgs, args.spectrum_bins)
        elif args.lambda_fft > 0 and args.fft_domain == "latent":
            fft_loss = latent_log_power_loss(z_hat, z_oracle, args.spectrum_bins)
        else:
            fft_loss = mu.new_tensor(0.0)
        loss = args.lambda_c12 * c12_loss + args.lambda_img * img_loss + args.lambda_fft * fft_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(prior.parameters(), args.grad_clip)
        optimizer.step()
        bsz = int(imgs.shape[0])
        loss_sum += float(loss.item()) * bsz
        c12_sum += float(c12_loss.item()) * bsz
        img_sum += float(img_loss.item()) * bsz
        fft_sum += float(fft_loss.item()) * bsz
        seen += bsz
    denom = max(1, seen)
    return {"loss": loss_sum / denom, "c12": c12_sum / denom, "img": img_sum / denom, "fft": fft_sum / denom}


def train_ddnm_epoch(args, prior, ddnm, encoder, decoder, channel, loader, optimizer, mean, basis, coeff_std, channel_weights, alpha_bar, device):
    prior.eval()
    ddnm.train()
    loss_sum = eps_sum = c12_sum = img_sum = fft_sum = 0.0
    aux_weight_sum = 0.0
    seen = 0
    for step, (imgs, _labels) in enumerate(tqdm(loader, desc="train residual ddnm", dynamic_ncols=True), start=1):
        if args.max_train_steps > 0 and step > args.max_train_steps:
            break
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        with torch.no_grad():
            _c36, c12, noisy_c36, z_base48, z_oracle = batch_context(args, encoder, channel, imgs, mean, basis, coeff_std)
            mu = prior(noisy_c36, z_base48, args.snr_db)
            residual = c12 - mu
        bsz = int(imgs.shape[0])
        t = torch.randint(0, args.diffusion_steps, (bsz,), device=device)
        eps = torch.randn_like(residual)
        ab_t = alpha_bar[t].view(-1, 1, 1, 1)
        r_t = torch.sqrt(ab_t) * residual + torch.sqrt(1.0 - ab_t) * eps
        t_norm = t.float() / float(max(1, args.diffusion_steps - 1))
        eps_pred = ddnm(r_t, noisy_c36, z_base48, t_norm, args.snr_db)
        eps_loss = F.mse_loss(eps_pred.float(), eps.float())
        r0_pred = (r_t - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(ab_t).clamp_min(1e-8)
        if args.clip_x0 > 0:
            r0_pred = r0_pred.clamp(-args.clip_x0, args.clip_x0)
        c12_hat = mu + r0_pred
        w_aux = clean_t_weight(t, args.diffusion_steps, args.aux_t_cutoff, args.aux_t_power).view(-1, 1, 1, 1)
        aux_denom = w_aux.sum().clamp_min(1.0)
        if args.lambda_x0_c12 > 0:
            c12_loss_per = (channel_weights * (c12_hat.float() - c12.float()).square()).mean(dim=(1, 2, 3), keepdim=True)
            c12_loss = (w_aux * c12_loss_per).sum() / aux_denom
        else:
            c12_loss = eps_pred.new_tensor(0.0)

        if args.lambda_img > 0 or args.lambda_fft > 0:
            z_hat = pca_inverse(noisy_c36, c12_hat, mean, basis, coeff_std)
            x_hat_raw = decoder(z_hat)
        else:
            z_hat = None
            x_hat_raw = None

        if args.lambda_img > 0:
            img_loss_per = (x_hat_raw.float() - imgs.float()).square().mean(dim=(1, 2, 3), keepdim=True)
            img_loss = (w_aux * img_loss_per).sum() / aux_denom
        else:
            img_loss = eps_pred.new_tensor(0.0)

        if args.lambda_fft > 0 and args.fft_domain == "image":
            low_mask = w_aux.flatten() > 0
            fft_loss = image_log_power_loss(x_hat_raw[low_mask], imgs[low_mask], args.spectrum_bins) if bool(low_mask.any()) else eps_pred.new_tensor(0.0)
        elif args.lambda_fft > 0 and args.fft_domain == "latent":
            low_mask = w_aux.flatten() > 0
            fft_loss = latent_log_power_loss(z_hat[low_mask], z_oracle[low_mask], args.spectrum_bins) if bool(low_mask.any()) else eps_pred.new_tensor(0.0)
        else:
            fft_loss = eps_pred.new_tensor(0.0)
        loss = eps_loss + args.lambda_x0_c12 * c12_loss + args.lambda_img * img_loss + args.lambda_fft * fft_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ddnm.parameters(), args.grad_clip)
        optimizer.step()
        loss_sum += float(loss.item()) * bsz
        eps_sum += float(eps_loss.item()) * bsz
        c12_sum += float(c12_loss.item()) * bsz
        img_sum += float(img_loss.item()) * bsz
        fft_sum += float(fft_loss.item()) * bsz
        aux_weight_sum += float(w_aux.mean().item()) * bsz
        seen += bsz
    denom = max(1, seen)
    return {
        "loss": loss_sum / denom,
        "eps": eps_sum / denom,
        "c12": c12_sum / denom,
        "img": img_sum / denom,
        "fft": fft_sum / denom,
        "aux_w": aux_weight_sum / denom,
    }


@torch.inference_mode()
def validate(
    args,
    prior,
    ddnm,
    encoder,
    decoder,
    channel,
    loader,
    mean,
    basis,
    coeff_std,
    alpha_bar,
    baseline_refs,
    device,
    epoch,
    phase,
    include_ddim: bool,
):
    prior.eval()
    ddnm.eval()
    accs = {"direct_prior": MetricAccumulator(args.spectrum_bins)}
    if include_ddim:
        accs["ddim_sample"] = MetricAccumulator(args.spectrum_bins)
    if include_ddim and args.diag_t >= 0:
        accs[f"x0_t{args.diag_t}"] = MetricAccumulator(args.spectrum_bins)
    for batch_idx, (imgs, _labels) in enumerate(tqdm(loader, desc=f"val {phase} epoch {epoch}", dynamic_ncols=True)):
        if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
            break
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        torch.manual_seed(args.val_noise_seed + batch_idx)
        torch.cuda.manual_seed_all(args.val_noise_seed + batch_idx)
        _c36, c12, noisy_c36, z_base48, _z_oracle = batch_context(args, encoder, channel, imgs, mean, basis, coeff_std)
        mu = prior(noisy_c36, z_base48, args.snr_db)

        z_direct = pca_inverse(noisy_c36, mu, mean, basis, coeff_std)
        x_direct = decoder(z_direct).clamp(0.0, 1.0)
        accs["direct_prior"].update(x_direct, imgs, c12_hat=mu, c12_gt=c12, c36_hat=noisy_c36, c36_gt=noisy_c36)

        if include_ddim:
            torch.manual_seed(args.val_noise_seed + 100000 + batch_idx)
            torch.cuda.manual_seed_all(args.val_noise_seed + 100000 + batch_idx)
            if getattr(args, "sampling_method", "residual_ddim") == "pca48_ddnm_plus":
                c12_sample = sample_pca48_ddnm_plus(prior, ddnm, noisy_c36, z_base48, args, alpha_bar)
            else:
                r_sample = sample_residual_ddim(ddnm, noisy_c36, z_base48, args, alpha_bar)
                c12_sample = mu + r_sample
            z_sample = pca_inverse(noisy_c36, c12_sample, mean, basis, coeff_std)
            x_sample = decoder(z_sample).clamp(0.0, 1.0)
            accs["ddim_sample"].update(x_sample, imgs, c12_hat=c12_sample, c12_gt=c12, c36_hat=noisy_c36, c36_gt=noisy_c36)

        if include_ddim and args.diag_t >= 0:
            t_val = min(args.diffusion_steps - 1, int(args.diag_t))
            t_batch = torch.full((imgs.shape[0],), t_val, device=device, dtype=torch.long)
            residual = c12 - mu
            torch.manual_seed(args.val_noise_seed + 200000 + batch_idx)
            torch.cuda.manual_seed_all(args.val_noise_seed + 200000 + batch_idx)
            eps = torch.randn_like(residual)
            ab_t = alpha_bar[t_batch].view(-1, 1, 1, 1)
            r_t = torch.sqrt(ab_t) * residual + torch.sqrt(1.0 - ab_t) * eps
            t_norm = t_batch.float() / float(max(1, args.diffusion_steps - 1))
            eps_pred = ddnm(r_t, noisy_c36, z_base48, t_norm, args.snr_db)
            r0 = (r_t - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(ab_t).clamp_min(1e-8)
            c12_x0 = mu + r0
            x_x0 = decoder(pca_inverse(noisy_c36, c12_x0, mean, basis, coeff_std)).clamp(0.0, 1.0)
            accs[f"x0_t{args.diag_t}"].update(x_x0, imgs, c12_hat=c12_x0, c12_gt=c12, c36_hat=noisy_c36, c36_gt=noisy_c36)

    rows = []
    profiles = {}
    for method, acc in accs.items():
        row = {"method": method, "phase": phase, "epoch": float(epoch)}
        row.update(acc.finalize())
        row["gain_vs_pca36"] = row["psnr"] - baseline_refs["pca36_zerofill"]["psnr"]
        row["gap_to_full"] = baseline_refs["full_c48_jscc"]["psnr"] - row["psnr"]
        row["gap_to_oracle"] = baseline_refs["oracle_c12"]["psnr"] - row["psnr"]
        row["delta_ratio"] = 0.0
        rows.append(row)
        profiles[method] = acc.pred_profile_sum / max(1, acc.seen)
    if accs:
        profiles["ground_truth"] = next(iter(accs.values())).target_profile_sum / max(1, next(iter(accs.values())).seen)
    return rows, profiles


def write_val_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    fields = [
        "method",
        "phase",
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


def write_summary(path: Path, args: argparse.Namespace, rows: list[dict[str, float]], channel_weights: torch.Tensor) -> None:
    best = max(rows, key=lambda r: r["psnr"]) if rows else None
    weights = channel_weights.detach().flatten().cpu().tolist()
    lines = [
        "Weighted c12 residual DDNM at SNR=9 dB",
        "",
        f"JSCC checkpoint root: {Path(args.ckpt_root).resolve()}",
        f"PCA basis: {Path(args.pca_basis).resolve()}",
        "PCA coefficients: whitened by sqrt(eigvals); c12 residual diffusion is trained in whitened null coordinates.",
        "Method note: conditional PCA null-space residual diffusion with data-consistent inverse PCA, not a generic DDNM projection loop.",
        f"c12 predictability CSV: {Path(args.channel_r2_csv).resolve() if args.channel_r2_csv else '<none>'}",
        f"channel_weights(c37..c48): {[round(w, 6) for w in weights]}",
        f"c36 policy: fixed noisy_c36 is always passed to inverse PCA; predictor never outputs delta_c36.",
        f"schedule={args.schedule} diffusion_steps={args.diffusion_steps} sample_steps={args.diffusion_sample_steps} sample_init={args.sample_init}",
        f"loss: eps + low_t({args.aux_t_cutoff},{args.aux_t_power})*({args.lambda_x0_c12}*weighted_x0_c12 + {args.lambda_img}*img + {args.lambda_fft}*{args.fft_domain}_fft)",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                f"Best method: {best['method']} phase={best['phase']} epoch={int(best['epoch'])}",
                f"Best PSNR: {best['psnr']:.4f} dB",
                f"Gain vs PCA36: {best['gain_vs_pca36']:+.4f} dB",
                f"Gap to oracle c12: {best['gap_to_oracle']:+.4f} dB",
                f"c12 MSE: {best['c12_mse']:.8f}",
                f"delta_ratio: {best['delta_ratio']:.8f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def save_checkpoint(path: Path, args, prior, ddnm, prior_opt, ddnm_opt, phase: str, epoch: int, rows: list[dict[str, float]], channel_weights: torch.Tensor) -> None:
    torch.save(
        {
            "kind": "weighted_c12_residual_ddnm",
            "phase": phase,
            "epoch": epoch,
            "args": vars(args),
            "prior": prior.state_dict(),
            "ddnm": ddnm.state_dict(),
            "prior_optimizer": prior_opt.state_dict(),
            "ddnm_optimizer": ddnm_opt.state_dict(),
            "rows": rows,
            "channel_weights": channel_weights.detach().cpu(),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weighted c12 residual DDNM for PCA36 C48 JSCC at SNR=9 dB.")
    p.add_argument("--train-dir", default="/workspace/yongjia/datasets/DIV2K/DIV2K_train_HR")
    p.add_argument("--val-dir", default="/workspace/yongjia/datasets/DIV2K/DIV2K_valid_HR")
    p.add_argument("--ckpt-root", default="MY/checkpoints-jscc")
    p.add_argument("--pca-basis", default="MY/pca/pca_basis_train_snr9_c48.pt")
    p.add_argument("--baseline-metrics", default="MY/checkpoints-pca-c48-snr9/baselines/val_metrics.csv")
    p.add_argument("--channel-r2-csv", default="MY/checkpoints-pca-c48-snr9/c12_predictability_analysis/channel_r2.csv")
    p.add_argument("--out-dir", default="MY/checkpoints-pca-c48-snr9/exp09_weighted_c12_residual_ddnm")
    p.add_argument("--prior-checkpoint", default="")
    p.add_argument("--C", type=int, default=48)
    p.add_argument("--keep-ch", type=int, default=36)
    p.add_argument("--snr-db", type=float, default=9.0)
    p.add_argument("--channel-type", choices=("awgn", "rayleigh"), default="awgn")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--freq-gate", action="store_true")
    p.add_argument("--prior-freq-gate", action="store_true")
    p.add_argument("--warmup-epochs", type=int, default=600)
    p.add_argument("--ddnm-epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-c12", type=float, default=0.1)
    p.add_argument("--lambda-x0-c12", type=float, default=0.05)
    p.add_argument("--lambda-img", type=float, default=0.25)
    p.add_argument("--lambda-fft", type=float, default=0.0)
    p.add_argument("--fft-domain", choices=("image", "latent"), default="image")
    p.add_argument("--aux-t-cutoff", type=float, default=0.35)
    p.add_argument("--aux-t-power", type=float, default=2.0)
    p.add_argument("--weight-score-col", default="predictable_sensitive_score")
    p.add_argument("--unpredictable-weight", type=float, default=0.1)
    p.add_argument("--predictable-weight-max", type=float, default=2.0)
    p.add_argument("--weight-power", type=float, default=0.5)
    p.add_argument("--normalize-channel-weights", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--schedule", choices=("linear", "cosine"), default="cosine")
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--diffusion-sample-steps", type=int, default=25)
    p.add_argument("--sampling-method", choices=("residual_ddim", "pca48_ddnm_plus"), default="residual_ddim")
    p.add_argument("--sample-start-t", type=int, default=50)
    p.add_argument("--sample-temperature", type=float, default=0.1)
    p.add_argument("--sample-init", choices=("noise", "zero"), default="noise")
    p.add_argument("--clip-x0", type=float, default=0.0)
    p.add_argument("--diag-t", type=int, default=50)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--spectrum-bins", type=int, default=64)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--val-noise-seed", type=int, default=20260603)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    log_path = out_dir / "train.log"
    with log_path.open("w", buffering=1) as log_f:
        original_print = builtins.print

        def tee_print(*a, **k):
            if "file" in k:
                original_print(*a, **k)
            else:
                original_print(*a, file=TeeStream(sys.__stdout__, log_f), **k)

        builtins.print = tee_print
        try:
            seed_all(args.seed)
            device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
            if device.type == "cuda":
                torch.cuda.set_device(0)
            cfg = build_jscc_config(args, device)
            enc_path, dec_path = checkpoint_paths(args)
            mean, basis, coeff_std, _pca_obj = load_pca_basis(Path(args.pca_basis).expanduser().resolve(), device, args.C)
            encoder, decoder = load_models(cfg, enc_path, dec_path)
            train_loader = make_train_loader(args.train_dir, args.crop_size, args.batch_size, args.num_workers, device.type == "cuda")
            val_loader = make_loader(args.val_dir, args.crop_size, args.val_batch_size, args.val_num_workers, device.type == "cuda")
            baseline_refs = load_baseline_refs(Path(args.baseline_metrics).expanduser().resolve())
            channel = Channel(cfg)
            out_ch = args.C - args.keep_ch
            channel_weights = load_channel_weights(args, device)
            prior = DirectPriorPredictor(args.keep_ch, args.C, args.hidden, args.depth, out_ch, args.prior_freq_gate).to(device)
            ddnm = ResidualDDNMPredictor(args.keep_ch, args.C, args.hidden, args.depth, out_ch, args.freq_gate).to(device)
            if args.prior_checkpoint:
                prior_path = Path(args.prior_checkpoint).expanduser().resolve()
                obj = torch.load(prior_path, map_location=device, weights_only=False)
                if "prior" not in obj:
                    raise KeyError(f"{prior_path} does not contain a prior state_dict")
                prior.load_state_dict(obj["prior"], strict=True)
                print(f"loaded_prior_checkpoint={prior_path}", flush=True)
            elif args.warmup_epochs <= 0 and args.ddnm_epochs > 0:
                raise ValueError("--warmup-epochs 0 with ddnm training requires --prior-checkpoint")
            prior_opt = optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            ddnm_opt = optim.AdamW(ddnm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            alpha_bar = make_schedule(args.diffusion_steps, args.schedule, device)
            all_rows: list[dict[str, float]] = []
            best_psnr = -1.0
            best_profiles: dict[str, np.ndarray] = {}
            start = time.time()
            print(f"start_time={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            print(f"device={device} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
            print(f"encoder={enc_path}", flush=True)
            print(f"decoder={dec_path}", flush=True)
            print(f"pca_basis={args.pca_basis}", flush=True)
            print(f"channel_weights_c37_c48={channel_weights.detach().flatten().cpu().tolist()}", flush=True)
            print(f"train_images={len(train_loader.dataset)} val_images={len(val_loader.dataset)}", flush=True)

            for epoch in range(1, args.warmup_epochs + 1):
                epoch_start = time.time()
                stats = train_prior_epoch(args, prior, encoder, decoder, channel, train_loader, prior_opt, mean, basis, coeff_std, channel_weights, device)
                print(
                    f"warmup epoch={epoch:03d} loss={stats['loss']:.8f} c12={stats['c12']:.8f} "
                    f"img={stats['img']:.8f} fft={stats['fft']:.8f} elapsed={time.time() - epoch_start:.1f}s",
                    flush=True,
                )
                if epoch == 1 or epoch == args.warmup_epochs or epoch % args.val_every == 0:
                    rows, profiles = validate(
                        args,
                        prior,
                        ddnm,
                        encoder,
                        decoder,
                        channel,
                        val_loader,
                        mean,
                        basis,
                        coeff_std,
                        alpha_bar,
                        baseline_refs,
                        device,
                        epoch,
                        "warmup",
                        include_ddim=False,
                    )
                    all_rows.extend(rows)
                    for row in rows:
                        print(
                            f"val phase=warmup epoch={epoch:03d} method={row['method']} psnr={row['psnr']:.4f} "
                            f"c12_mse={row['c12_mse']:.8f} gain={row['gain_vs_pca36']:+.4f} "
                            f"gap_oracle={row['gap_to_oracle']:+.4f} delta={row['delta_ratio']:.6f}",
                            flush=True,
                        )
                    if rows and max(r["psnr"] for r in rows) > best_psnr:
                        best_psnr = max(r["psnr"] for r in rows)
                        best_profiles = profiles
                        save_checkpoint(out_dir / "best_checkpoint.pt", args, prior, ddnm, prior_opt, ddnm_opt, "warmup", epoch, all_rows, channel_weights)
                if epoch % args.save_every == 0 or epoch == args.warmup_epochs:
                    save_checkpoint(out_dir / "latest_checkpoint.pt", args, prior, ddnm, prior_opt, ddnm_opt, "warmup", epoch, all_rows, channel_weights)

            for epoch in range(1, args.ddnm_epochs + 1):
                epoch_start = time.time()
                stats = train_ddnm_epoch(args, prior, ddnm, encoder, decoder, channel, train_loader, ddnm_opt, mean, basis, coeff_std, channel_weights, alpha_bar, device)
                print(
                    f"ddnm epoch={epoch:03d} loss={stats['loss']:.8f} eps={stats['eps']:.8f} "
                    f"c12={stats['c12']:.8f} img={stats['img']:.8f} fft={stats['fft']:.8f} "
                    f"aux_w={stats['aux_w']:.6f} elapsed={time.time() - epoch_start:.1f}s",
                    flush=True,
                )
                if epoch == 1 or epoch == args.ddnm_epochs or epoch % args.val_every == 0:
                    rows, profiles = validate(
                        args,
                        prior,
                        ddnm,
                        encoder,
                        decoder,
                        channel,
                        val_loader,
                        mean,
                        basis,
                        coeff_std,
                        alpha_bar,
                        baseline_refs,
                        device,
                        epoch,
                        "ddnm",
                        include_ddim=True,
                    )
                    all_rows.extend(rows)
                    for row in rows:
                        print(
                            f"val phase=ddnm epoch={epoch:03d} method={row['method']} psnr={row['psnr']:.4f} "
                            f"c12_mse={row['c12_mse']:.8f} gain={row['gain_vs_pca36']:+.4f} "
                            f"gap_oracle={row['gap_to_oracle']:+.4f} delta={row['delta_ratio']:.6f}",
                            flush=True,
                        )
                    if rows and max(r["psnr"] for r in rows) > best_psnr:
                        best_psnr = max(r["psnr"] for r in rows)
                        best_profiles = profiles
                        save_checkpoint(out_dir / "best_checkpoint.pt", args, prior, ddnm, prior_opt, ddnm_opt, "ddnm", epoch, all_rows, channel_weights)
                if epoch % args.save_every == 0 or epoch == args.ddnm_epochs:
                    save_checkpoint(out_dir / "latest_checkpoint.pt", args, prior, ddnm, prior_opt, ddnm_opt, "ddnm", epoch, all_rows, channel_weights)

            write_val_metrics(out_dir / "val_metrics.csv", all_rows)
            if best_profiles:
                write_spectrum_metrics(out_dir / "spectrum_metrics.csv", best_profiles)
            write_summary(out_dir / "summary.txt", args, all_rows, channel_weights)
            print(f"elapsed_sec={time.time() - start:.1f}", flush=True)
            print(f"outputs={out_dir}", flush=True)
        finally:
            builtins.print = original_print


if __name__ == "__main__":
    main()
