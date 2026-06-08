#!/usr/bin/env python
"""Conditional x0 diffusion with partial zero sampling for receiver-normalized high Swin latents.

Stage 03B:

    cond   = z0_rx_norm = z0 / scale + AWGN
    target = z[:, 4:16] / scale

The receiver-normalized Swin encoder/decoder from stage 01 are frozen.
The diffusion model only generates the missing 12 high channels.
The transmitted 4-channel observation is never rewritten.

Supported modes:
    --prediction_type eps
    --prediction_type x0

For x0 prediction:
    model(z_t, cond, t) -> clean high latent x0_pred ~= z_high_norm

Partial zero sampling:
    Start from h_init = 0 at selected t_start values, then denoise t_start -> 0.
"""

from __future__ import annotations

import argparse
import builtins
import math
import os
import sys
import time
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
TRAIN_DIR = os.path.abspath(os.path.dirname(__file__))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from train_codex_orthogonal_highfreq import (  # noqa: E402
    _parse_amp,
    make_autocast,
    power_normalize_awgn,
    psnr_per_image,
    seed_everything,
    semiorth_error,
)
from train_hierarchical_swin_ar_awgn12 import (  # noqa: E402
    build_semantic_modules,
    decode_a,
    encode_a,
    fixed_select_a,
    make_loaders,
)
from train_route_a_sc import AverageMeter, TeeStream  # noqa: E402


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Receiver-normalized high x0/eps diffusion @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time_dim must be even")
        self.dim = int(dim)
        self.max_period = float(max_period)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor, num_steps: int) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(1, half - 1)
        )
        _ = num_steps
        args = t.float().view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


class TimeResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.to_shift = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * channels))
        nn.init.zeros_(self.to_shift[-1].weight)
        nn.init.zeros_(self.to_shift[-1].bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_shift(temb).chunk(2, dim=1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1).to(dtype=x.dtype)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1).to(dtype=x.dtype)
        h = self.norm1(x)
        h = h * (1.0 + gamma) + beta
        h = self.conv1(F.silu(h))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class TimeBlockStack(nn.Module):
    def __init__(self, channels: int, time_dim: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TimeResBlock(channels, time_dim) for _ in range(max(1, int(depth)))])

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, temb)
        return x


class DiffHighUNet(nn.Module):
    """Latent U-Net for eps/x0 prediction on 12 high channels conditioned by z0_rx."""

    def __init__(
        self,
        high_channels: int = 12,
        cond_channels: int = 4,
        base: int = 128,
        depth: int = 3,
        time_dim: int = 256,
        num_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        h1, h2, h3 = int(base), int(base) * 2, int(base) * 4
        self.time = TimeEmbedding(int(time_dim))
        self.stem = nn.Sequential(nn.Conv2d(high_channels + cond_channels, h1, 3, padding=1), nn.SiLU())
        self.enc1 = TimeBlockStack(h1, int(time_dim), int(depth))
        self.down1 = nn.Sequential(nn.Conv2d(h1, h2, 3, stride=2, padding=1), nn.SiLU())
        self.enc2 = TimeBlockStack(h2, int(time_dim), int(depth))
        self.down2 = nn.Sequential(nn.Conv2d(h2, h3, 3, stride=2, padding=1), nn.SiLU())
        self.mid = TimeBlockStack(h3, int(time_dim), int(depth) + 1)
        self.up2 = nn.Sequential(nn.Conv2d(h3, h2, 3, padding=1), nn.SiLU())
        self.fuse2 = nn.Conv2d(h2 + h2, h2, 1)
        self.dec2 = TimeBlockStack(h2, int(time_dim), 2)
        self.up1 = nn.Sequential(nn.Conv2d(h2, h1, 3, padding=1), nn.SiLU())
        self.fuse1 = nn.Conv2d(h1 + h1, h1, 1)
        self.dec1 = TimeBlockStack(h1, int(time_dim), 2)
        self.out_norm = nn.GroupNorm(min(8, h1), h1)
        self.out = nn.Conv2d(h1, high_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time(t, self.num_steps).to(dtype=z_t.dtype)
        x = torch.cat([z_t, cond.to(dtype=z_t.dtype)], dim=1)
        e1 = self.enc1(self.stem(x), temb)
        e2 = self.enc2(self.down1(e1), temb)
        m = self.mid(self.down2(e2), temb)

        d2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(self.fuse2(torch.cat([self.up2(d2), e2], dim=1)), temb)

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(self.fuse1(torch.cat([self.up1(d1), e1], dim=1)), temb)

        return self.out(F.silu(self.out_norm(d1)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train high-latent x0/eps diffusion conditioned on receiver-normalized z0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", action="store_false", dest="cache_decoded")
    p.add_argument("--cache_workers", type=int, default=12)
    p.add_argument("--prefetch_factor", type=int, default=4)

    p.add_argument("--init_hier_ckpt", type=str, required=True)
    p.add_argument("--snr_db", type=float, default=6.0)

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    p.add_argument("--diffusion_steps", type=int, default=200)
    p.add_argument("--noise_schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--prediction_type", type=str, default="x0", choices=["eps", "x0"])
    p.add_argument("--lambda_x0", type=float, default=0.0, help="Only used when prediction_type=eps")
    p.add_argument("--clip_x0", type=float, default=3.0, help="Clamp sampled/diagnostic x0 predictions; <=0 disables")

    p.add_argument("--unet_base", type=int, default=128)
    p.add_argument("--unet_depth", type=int, default=3)
    p.add_argument("--time_dim", type=int, default=256)

    p.add_argument("--ddim_steps_a", type=int, default=20)
    p.add_argument("--ddim_steps_b", type=int, default=50)
    p.add_argument("--sample_init", type=str, default="noise", choices=["noise", "zero"])

    p.add_argument("--eval_partial", action="store_true", default=False)
    p.add_argument("--partial_t_starts", type=str, default="10,25,50,75,100")
    p.add_argument("--diag_timesteps", type=str, default="25,50,100,199")

    p.add_argument("--encoder_use_vae", action="store_true", default=False)
    p.add_argument("--lambda_kl", type=float, default=0.0)
    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints-ar/x0_partial_high")
    p.add_argument("--log_file", type=str, default="checkpoints-ar/x0_partial_high/train.log")
    return p.parse_args()


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def make_schedule(num_steps: int, schedule: str, device: torch.device) -> dict[str, torch.Tensor]:
    if schedule == "linear":
        betas = torch.linspace(1e-4, 2e-2, int(num_steps), device=device, dtype=torch.float32)
    elif schedule == "cosine":
        steps = torch.arange(int(num_steps) + 1, device=device, dtype=torch.float32)
        s = 0.008
        alpha_bar = torch.cos(((steps / int(num_steps)) + s) / (1.0 + s) * math.pi * 0.5).square()
        alpha_bar = (alpha_bar / alpha_bar[0]).clamp_min(1e-8)
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-8, 0.999)
    else:
        raise ValueError(f"unknown noise_schedule={schedule!r}")

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0).clamp_min(1e-8)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt((1.0 - alpha_bar).clamp_min(0.0)),
    }


def gather_4d(values: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return values.gather(0, t.long()).view(t.shape[0], 1, 1, 1).to(device=like.device, dtype=like.dtype)


def parse_int_list(text: str, max_steps: int) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        v = int(item)
        v = max(0, min(int(max_steps) - 1, v))
        if v not in out:
            out.append(v)
    return out


def load_hier_encoder_decoder(path: str, encoder: nn.Module, decoder: nn.Module, device: torch.device) -> dict:
    ckpt_path = resolve_path(path)
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    print(f"loaded frozen Swin: {ckpt_path}, stage={ckpt.get('stage', 'unknown')}, epoch={ckpt.get('epoch', 'unknown')}")
    return ckpt


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(bool(trainable))


@torch.no_grad()
def make_receiver_batch(
    imgs: torch.Tensor,
    encoder: nn.Module,
    a: torch.Tensor,
    snr_db: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = imgs.shape[0]
    snr_b = torch.full((bsz,), float(snr_db), device=imgs.device, dtype=torch.float32)
    z, _mu, _logvar = encoder.encode(imgs, sample=False)
    z = z.float()
    y4 = encode_a(z, a)
    y4_norm, _y4_raw, scale = power_normalize_awgn(y4, snr_b, generator=generator)
    scale_view = scale.float().view(-1, 1, 1, 1).clamp_min(1e-12)
    z_high_norm = z[:, 4:16].float() / scale_view
    return y4_norm.float(), z_high_norm


def diffusion_loss(
    model: nn.Module,
    cond: torch.Tensor,
    target: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    bsz = target.shape[0]
    t = torch.randint(0, int(args.diffusion_steps), (bsz,), device=target.device)
    eps = torch.randn_like(target)

    a = gather_4d(schedule["sqrt_alpha_bar"], t, target)
    s = gather_4d(schedule["sqrt_one_minus_alpha_bar"], t, target)
    z_t = a * target + s * eps

    with make_autocast(target.device, amp_enabled, amp_dtype):
        pred = model(z_t, cond, t).float()

        if str(args.prediction_type) == "eps":
            eps_pred = pred
            loss_eps = F.mse_loss(eps_pred, eps)
            x0_pred = (z_t - s * eps_pred) / a.clamp_min(1e-8)
            loss_x0 = F.mse_loss(x0_pred, target)
            loss = loss_eps + float(args.lambda_x0) * loss_x0

        elif str(args.prediction_type) == "x0":
            x0_pred = pred
            loss_x0 = F.mse_loss(x0_pred, target)
            eps_pred = (z_t - a * x0_pred) / s.clamp_min(1e-8)
            loss_eps = F.mse_loss(eps_pred, eps)
            loss = loss_x0

        else:
            raise ValueError(f"unknown prediction_type={args.prediction_type}")

    return loss, {
        "loss": float(loss.detach().item()),
        "loss_eps": float(loss_eps.detach().item()),
        "loss_x0": float(loss_x0.detach().item()),
    }


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    cond: torch.Tensor,
    shape: tuple[int, int, int, int],
    schedule: dict[str, torch.Tensor],
    steps: int,
    num_train_steps: int,
    sample_init: str,
    generator: torch.Generator | None,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    prediction_type: str = "eps",
    init_noise: torch.Tensor | None = None,
    clip_x0: float = 0.0,
    t_start: int | None = None,
    init_x0: torch.Tensor | None = None,
) -> torch.Tensor:
    alpha_bar = schedule["alpha_bar"]

    if t_start is None:
        start_t = int(num_train_steps) - 1
        if sample_init == "zero":
            z = torch.zeros(shape, device=cond.device, dtype=torch.float32)
        elif init_noise is not None:
            z = init_noise.detach().clone().to(device=cond.device, dtype=torch.float32)
        else:
            z = torch.randn(shape, device=cond.device, dtype=torch.float32, generator=generator)
    else:
        start_t = max(0, min(int(num_train_steps) - 1, int(t_start)))
        if init_x0 is None:
            init_x0 = torch.zeros(shape, device=cond.device, dtype=torch.float32)
        else:
            init_x0 = init_x0.detach().clone().to(device=cond.device, dtype=torch.float32)

        if init_noise is not None:
            noise = init_noise.detach().clone().to(device=cond.device, dtype=torch.float32)
        else:
            noise = torch.randn(shape, device=cond.device, dtype=torch.float32, generator=generator)

        a_start = alpha_bar[start_t].view(1, 1, 1, 1).to(device=cond.device, dtype=torch.float32)
        z = torch.sqrt(a_start) * init_x0 + torch.sqrt((1.0 - a_start).clamp_min(0.0)) * noise

    step_ids = torch.linspace(start_t, 0, max(1, int(steps)), device=cond.device).round().long()
    step_ids = torch.unique_consecutive(step_ids)

    last_x0_pred = None

    for i, ti in enumerate(step_ids):
        t = torch.full((shape[0],), int(ti.item()), device=cond.device, dtype=torch.long)
        a_t = alpha_bar[t].view(shape[0], 1, 1, 1).to(dtype=z.dtype)
        sqrt_a_t = torch.sqrt(a_t).clamp_min(1e-8)
        s_t = torch.sqrt((1.0 - a_t).clamp_min(0.0))

        with make_autocast(cond.device, amp_enabled, amp_dtype):
            pred = model(z, cond, t).float()

        if prediction_type == "eps":
            eps_pred = pred
            x0_pred = (z - s_t * eps_pred) / sqrt_a_t
        elif prediction_type == "x0":
            x0_pred = pred
            if float(clip_x0) > 0:
                x0_pred = x0_pred.clamp(-float(clip_x0), float(clip_x0))
            eps_pred = (z - sqrt_a_t * x0_pred) / s_t.clamp_min(1e-8)
        else:
            raise ValueError(f"unknown prediction_type={prediction_type}")

        if float(clip_x0) > 0:
            x0_pred = x0_pred.clamp(-float(clip_x0), float(clip_x0))

        last_x0_pred = x0_pred

        if i == len(step_ids) - 1:
            z = x0_pred
        else:
            next_t = int(step_ids[i + 1].item())
            a_next = alpha_bar[next_t].view(1, 1, 1, 1).to(device=cond.device, dtype=z.dtype)
            z = torch.sqrt(a_next) * x0_pred + torch.sqrt((1.0 - a_next).clamp_min(0.0)) * eps_pred

    return z.float() if last_x0_pred is None else last_x0_pred.float()


@torch.no_grad()
def one_step_x0_diagnostic(
    model: nn.Module,
    cond: torch.Tensor,
    target: torch.Tensor,
    t_value: int,
    schedule: dict[str, torch.Tensor],
    generator: torch.Generator,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    clip_x0: float,
    prediction_type: str,
) -> torch.Tensor:
    t = torch.full((target.shape[0],), int(t_value), device=target.device, dtype=torch.long)
    eps = torch.randn(target.shape, device=target.device, dtype=torch.float32, generator=generator)

    a = gather_4d(schedule["sqrt_alpha_bar"], t, target)
    s = gather_4d(schedule["sqrt_one_minus_alpha_bar"], t, target)
    z_t = a * target + s * eps

    with make_autocast(target.device, amp_enabled, amp_dtype):
        pred = model(z_t, cond, t).float()

    if prediction_type == "eps":
        eps_pred = pred
        x0_pred = (z_t - s * eps_pred) / a.clamp_min(1e-8)
    elif prediction_type == "x0":
        x0_pred = pred
    else:
        raise ValueError(f"unknown prediction_type={prediction_type}")

    if float(clip_x0) > 0:
        x0_pred = x0_pred.clamp(-float(clip_x0), float(clip_x0))
    return x0_pred.float()


def save_checkpoint(
    path: str,
    encoder: nn.Module,
    decoder: nn.Module,
    model: nn.Module,
    a: torch.Tensor,
    cfg,
    args: argparse.Namespace,
    metrics: dict,
    epoch: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "route": "receiver_norm_diffusion_high",
        "stage": "03B_x0_partial_high_unet_frozen_swin",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "semantic_config": asdict(cfg.semantic),
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "diffusion_state_dict": model.state_dict(),
        "a_matrix": a.detach().cpu(),
        "aat": (a @ a.t()).detach().cpu(),
        "aat_error": semiorth_error(a),
        "fixed_channel_codec": True,
        "channel_encoder": "A=[I4,0]",
        "channel_decoder": "A^T zero-fill",
        "power_norm_after_channel_encoder": True,
        "receiver_observation": "norm",
        "use_scale": False,
        "target": "z_high_norm=z[:,4:16]/scale_from_z0",
        "prediction_type": str(args.prediction_type),
        "eval_partial": bool(args.eval_partial),
        "partial_t_starts": str(args.partial_t_starts),
        "snr_db": float(args.snr_db),
        "trainable_encoder": False,
        "trainable_decoder": False,
        "trainable_diffusion": True,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    save_dir = resolve_path(args.save_dir)
    log_file = resolve_path(args.log_file)
    os.makedirs(save_dir, exist_ok=True)
    setup_log_file(log_file)
    seed_everything(int(args.seed))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if device.type != "cuda":
        amp_enabled = False

    print(
        f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}, "
        f"stage=03B_x0_partial_high_unet_frozen_swin, snr={float(args.snr_db):g}dB"
    )
    print("rule: frozen 01 Swin, no VAE, no side information, scale comes from transmitted z0 only, PSNR=mean(per-image PSNR), A=[I4,0]")

    train_ds, val_ds, train_loader, val_loader = make_loaders(args, device)
    print(f"train={len(train_ds)} valid={len(val_ds)} batch={args.batch_size} crop={args.crop_size}")

    encoder, decoder, cfg = build_semantic_modules(device, args)
    load_hier_encoder_decoder(args.init_hier_ckpt, encoder, decoder, device)
    set_trainable(encoder, False)
    set_trainable(decoder, False)
    encoder.eval()
    decoder.eval()

    a = fixed_select_a(device=device, dtype=torch.float32)
    aat_err = semiorth_error(a)
    if aat_err > 1e-7:
        raise RuntimeError(f"A A^T != I4, err={aat_err:.3e}")

    model = DiffHighUNet(
        high_channels=12,
        cond_channels=4,
        base=int(args.unet_base),
        depth=int(args.unet_depth),
        time_dim=int(args.time_dim),
        num_steps=int(args.diffusion_steps),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    schedule = make_schedule(int(args.diffusion_steps), str(args.noise_schedule), device)

    diag_timesteps = parse_int_list(str(args.diag_timesteps), int(args.diffusion_steps))
    partial_t_starts = parse_int_list(str(args.partial_t_starts), int(args.diffusion_steps))

    print(
        f"model=DiffHighUNet base={args.unet_base} depth={args.unet_depth} time_dim={args.time_dim} "
        f"params={sum(p.numel() for p in model.parameters())} lr={float(args.lr):g} "
        f"T={args.diffusion_steps} schedule={args.noise_schedule} prediction={args.prediction_type} "
        f"lambda_x0={float(args.lambda_x0):g} clip_x0={float(args.clip_x0):g} "
        f"ddim={args.ddim_steps_a}/{args.ddim_steps_b} full_init={args.sample_init} "
        f"eval_partial={bool(args.eval_partial)} partial_t={partial_t_starts} diag_t={diag_timesteps} "
        f"aat_error={aat_err:.3e}"
    )

    train_keys = ("loss", "loss_eps", "loss_x0")
    val_keys = [
        "psnr_base",
        "psnr_recv_a",
        "psnr_recv_b",
        "psnr_oracle",
        "mse_high_a",
        "mse_high_b",
        "target_mean",
        "target_std",
        "target_abs",
        "sample_mean",
        "sample_std",
        "sample_abs",
    ]

    for t_diag in diag_timesteps:
        val_keys.append(f"psnr_x0_t{t_diag}")
        val_keys.append(f"mse_x0_t{t_diag}")

    if bool(args.eval_partial):
        for ts in partial_t_starts:
            val_keys.append(f"psnr_partial_t{ts}")
            val_keys.append(f"mse_partial_t{ts}")
            val_keys.append(f"std_partial_t{ts}")

    best = -1.0
    best_name = "x0_partial_high_unet_frozen_swin_best.pth"
    latest_name = "x0_partial_high_unet_frozen_swin_latest.pth"

    for epoch in range(1, int(args.epochs) + 1):
        model.train(True)
        meters = {k: AverageMeter() for k in train_keys}

        for bi, batch in enumerate(train_loader):
            if int(args.max_train_batches) > 0 and bi >= int(args.max_train_batches):
                break

            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=True)

            with torch.no_grad():
                cond, target = make_receiver_batch(imgs, encoder, a, float(args.snr_db), generator=None)

            optimizer.zero_grad(set_to_none=True)
            loss, stats = diffusion_loss(model, cond, target, schedule, args, amp_enabled, amp_dtype)

            scaler.scale(loss).backward()
            if float(args.clip_grad_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
            scaler.step(optimizer)
            scaler.update()

            for k in train_keys:
                meters[k].update(stats[k], imgs.shape[0])

        metrics = {f"train_{k}": v.avg for k, v in meters.items()}
        do_eval = epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs)

        if do_eval:
            model.eval()
            val_meters = {k: AverageMeter() for k in val_keys}
            gen_awgn = torch.Generator(device=device)
            gen_awgn.manual_seed(int(args.seed) + 1000)

            with torch.no_grad():
                for bi, batch in enumerate(val_loader):
                    if int(args.max_val_batches) > 0 and bi >= int(args.max_val_batches):
                        break

                    imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
                    imgs = imgs.to(device, non_blocking=True)

                    cond, target = make_receiver_batch(imgs, encoder, a, float(args.snr_db), generator=gen_awgn)

                    gen_init = torch.Generator(device=device)
                    gen_init.manual_seed(int(args.seed) + 2000 + bi)
                    init_noise = (
                        torch.randn(tuple(target.shape), device=device, dtype=torch.float32, generator=gen_init)
                        if str(args.sample_init) == "noise"
                        else None
                    )

                    high_a = ddim_sample(
                        model,
                        cond,
                        tuple(target.shape),
                        schedule,
                        int(args.ddim_steps_a),
                        int(args.diffusion_steps),
                        str(args.sample_init),
                        None,
                        amp_enabled,
                        amp_dtype,
                        prediction_type=str(args.prediction_type),
                        init_noise=init_noise,
                        clip_x0=float(args.clip_x0),
                    )

                    high_b = ddim_sample(
                        model,
                        cond,
                        tuple(target.shape),
                        schedule,
                        int(args.ddim_steps_b),
                        int(args.diffusion_steps),
                        str(args.sample_init),
                        None,
                        amp_enabled,
                        amp_dtype,
                        prediction_type=str(args.prediction_type),
                        init_noise=init_noise,
                        clip_x0=float(args.clip_x0),
                    )

                    z_base = decode_a(cond, a)
                    z_oracle = torch.cat([cond, target], dim=1)

                    x_base = decoder(z_base).float().clamp(0, 1)
                    x_recv_a = decoder(torch.cat([cond, high_a], dim=1)).float().clamp(0, 1)
                    x_recv_b = decoder(torch.cat([cond, high_b], dim=1)).float().clamp(0, 1)
                    x_oracle = decoder(z_oracle).float().clamp(0, 1)

                    stats = {
                        "psnr_base": float(psnr_per_image(x_base, imgs.float()).mean().item()),
                        "psnr_recv_a": float(psnr_per_image(x_recv_a, imgs.float()).mean().item()),
                        "psnr_recv_b": float(psnr_per_image(x_recv_b, imgs.float()).mean().item()),
                        "psnr_oracle": float(psnr_per_image(x_oracle, imgs.float()).mean().item()),
                        "mse_high_a": float(F.mse_loss(high_a, target).item()),
                        "mse_high_b": float(F.mse_loss(high_b, target).item()),
                        "target_mean": float(target.float().mean().item()),
                        "target_std": float(target.float().std(unbiased=False).item()),
                        "target_abs": float(target.float().abs().mean().item()),
                        "sample_mean": float(high_b.float().mean().item()),
                        "sample_std": float(high_b.float().std(unbiased=False).item()),
                        "sample_abs": float(high_b.float().abs().mean().item()),
                    }

                    for t_diag in diag_timesteps:
                        gen_diag = torch.Generator(device=device)
                        gen_diag.manual_seed(int(args.seed) + 4000 + bi * 1000 + int(t_diag))

                        high_diag = one_step_x0_diagnostic(
                            model,
                            cond,
                            target,
                            int(t_diag),
                            schedule,
                            gen_diag,
                            amp_enabled,
                            amp_dtype,
                            float(args.clip_x0),
                            str(args.prediction_type),
                        )
                        x_diag = decoder(torch.cat([cond, high_diag], dim=1)).float().clamp(0, 1)

                        stats[f"psnr_x0_t{t_diag}"] = float(psnr_per_image(x_diag, imgs.float()).mean().item())
                        stats[f"mse_x0_t{t_diag}"] = float(F.mse_loss(high_diag, target).item())

                    if bool(args.eval_partial):
                        zero_init = torch.zeros_like(target)

                        for ts in partial_t_starts:
                            gen_partial = torch.Generator(device=device)
                            gen_partial.manual_seed(int(args.seed) + 5000 + bi * 1000 + int(ts))

                            partial_noise = torch.randn(
                                tuple(target.shape),
                                device=device,
                                dtype=torch.float32,
                                generator=gen_partial,
                            )

                            partial_steps = min(int(args.ddim_steps_b), int(ts) + 1)

                            high_partial = ddim_sample(
                                model,
                                cond,
                                tuple(target.shape),
                                schedule,
                                partial_steps,
                                int(args.diffusion_steps),
                                "noise",
                                None,
                                amp_enabled,
                                amp_dtype,
                                prediction_type=str(args.prediction_type),
                                init_noise=partial_noise,
                                clip_x0=float(args.clip_x0),
                                t_start=int(ts),
                                init_x0=zero_init,
                            )

                            x_partial = decoder(torch.cat([cond, high_partial], dim=1)).float().clamp(0, 1)

                            stats[f"psnr_partial_t{ts}"] = float(psnr_per_image(x_partial, imgs.float()).mean().item())
                            stats[f"mse_partial_t{ts}"] = float(F.mse_loss(high_partial, target).item())
                            stats[f"std_partial_t{ts}"] = float(high_partial.float().std(unbiased=False).item())

                    for k in val_keys:
                        val_meters[k].update(stats[k], imgs.shape[0])

            metrics.update({f"val_{k}": v.avg for k, v in val_meters.items()})

            score = metrics["val_psnr_recv_b"]
            if bool(args.eval_partial):
                partial_scores = [metrics[f"val_psnr_partial_t{ts}"] for ts in partial_t_starts]
                score = max([score] + partial_scores)

            is_best = score > best
            if is_best:
                best = score
                save_checkpoint(os.path.join(save_dir, best_name), encoder, decoder, model, a, cfg, args, metrics, epoch)
            save_checkpoint(os.path.join(save_dir, latest_name), encoder, decoder, model, a, cfg, args, metrics, epoch)

            diag_msg = " ".join(
                f"x0_t{t_diag}={metrics[f'val_psnr_x0_t{t_diag}']:.2f}/mse{metrics[f'val_mse_x0_t{t_diag}']:.4f}"
                for t_diag in diag_timesteps
            )

            partial_msg = ""
            if bool(args.eval_partial):
                parts = []
                for ts in partial_t_starts:
                    p = metrics[f"val_psnr_partial_t{ts}"]
                    g = p - metrics["val_psnr_base"]
                    m = metrics[f"val_mse_partial_t{ts}"]
                    st = metrics[f"val_std_partial_t{ts}"]
                    parts.append(f"pt{ts}={p:.4f}/g{g:+.4f}/mse{m:.4f}/std{st:.3f}")
                partial_msg = " " + " ".join(parts)

            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"eps={meters['loss_eps'].avg:.6f} x0={meters['loss_x0'].avg:.6f} | "
                f"base={metrics['val_psnr_base']:.4f} "
                f"recv{args.ddim_steps_a}={metrics['val_psnr_recv_a']:.4f} "
                f"recv{args.ddim_steps_b}={metrics['val_psnr_recv_b']:.4f} "
                f"oracle={metrics['val_psnr_oracle']:.4f} "
                f"gain{args.ddim_steps_b}={metrics['val_psnr_recv_b'] - metrics['val_psnr_base']:+.4f} "
                f"gap_oracle{args.ddim_steps_b}={metrics['val_psnr_oracle'] - metrics['val_psnr_recv_b']:+.4f} "
                f"mse_high{args.ddim_steps_b}={metrics['val_mse_high_b']:.6f} "
                f"std_gt={metrics['val_target_std']:.4f} std_sample={metrics['val_sample_std']:.4f} "
                f"mean_gt={metrics['val_target_mean']:+.4f} mean_sample={metrics['val_sample_mean']:+.4f} "
                f"abs_gt={metrics['val_target_abs']:.4f} abs_sample={metrics['val_sample_abs']:.4f} "
                f"{diag_msg}{partial_msg} score={score:.4f} "
                f"aat_err={semiorth_error(a):.2e} {'BEST' if is_best else ''}"
            )

        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"eps={meters['loss_eps'].avg:.6f} x0={meters['loss_x0'].avg:.6f}"
            )

    print(f"best_val_score={best:.4f}")


if __name__ == "__main__":
    main()
