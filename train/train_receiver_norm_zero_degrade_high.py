#!/usr/bin/env python
"""Zero-degradation x0 predictor for receiver-normalized high Swin latents.

Stage 03D:

    cond   = z0_rx_norm = z0 / scale + AWGN
    target = z[:, 4:16] / scale

Instead of standard diffusion that degrades target -> pure Gaussian noise, this
script uses a task-matched cold/zero degradation:

    z_t = m_t * target + sigma_t * eps
    m_t = 1 - t / (T - 1)
    sigma_t = sigma_max * t / (T - 1)

Thus the terminal state is near zero-high, matching the receiver baseline
[cond, 0].  The model predicts x0 directly:

    model(z_t, cond, t) -> target

The receiver-normalized Swin encoder/decoder from stage 01 are frozen.  The
model only predicts the missing 12 high channels; the transmitted 4-channel
observation is never rewritten.
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
    builtins.print(f"\n=== Receiver-normalized zero-degradation high x0 @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
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


class ZeroDegHighUNet(nn.Module):
    """Latent U-Net for x0 prediction on 12 high channels conditioned by z0_rx."""

    def __init__(
        self,
        high_channels: int = 12,
        cond_channels: int = 4,
        base: int = 128,
        depth: int = 3,
        time_dim: int = 256,
        num_steps: int = 200,
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
        e1 = self.enc1(self.stem(torch.cat([z_t, cond.to(dtype=z_t.dtype)], dim=1)), temb)
        e2 = self.enc2(self.down1(e1), temb)
        m = self.mid(self.down2(e2), temb)
        d2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(self.fuse2(torch.cat([self.up2(d2), e2], dim=1)), temb)
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(self.fuse1(torch.cat([self.up1(d1), e1], dim=1)), temb)
        return self.out(F.silu(self.out_norm(d1)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train zero-degradation x0 high-latent predictor conditioned on receiver-normalized z0",
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
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--num_steps", type=int, default=200, help="Number of zero-degradation timesteps")
    p.add_argument("--sigma_max", type=float, default=0.10, help="Terminal noise std relative to high latent scale")
    p.add_argument("--endpoint_zero_prob", type=float, default=0.25, help="Probability of training exactly from z_t=0 at t=T-1")
    p.add_argument("--clip_x0", type=float, default=3.0, help="Clamp predicted high latent; <=0 disables")
    p.add_argument("--unet_base", type=int, default=128)
    p.add_argument("--unet_depth", type=int, default=3)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--zero_eval_timesteps", type=str, default="50,100,199", help="Evaluate single-step model(0,cond,t)")
    p.add_argument("--refine_t_starts", type=str, default="50,100,199", help="Evaluate deterministic multi-step refinement from zero")
    p.add_argument("--refine_steps", type=int, default=20)
    p.add_argument("--diag_timesteps", type=str, default="25,50,100,199", help="Evaluate one-step degraded target diagnostic")
    p.add_argument("--encoder_use_vae", action="store_true", default=False)
    p.add_argument("--lambda_kl", type=float, default=0.0)
    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints-ar/zero_degrade_high")
    p.add_argument("--log_file", type=str, default="checkpoints-ar/zero_degrade_high/train.log")
    return p.parse_args()


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def parse_int_list(text: str, max_steps: int) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        t = int(item)
        t = max(0, min(int(max_steps) - 1, t))
        if t not in out:
            out.append(t)
    return out


def zero_coefficients(t: torch.Tensor, num_steps: int, sigma_max: float, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denom = max(1, int(num_steps) - 1)
    frac = t.float().view(-1, 1, 1, 1) / float(denom)
    frac = frac.to(device=like.device, dtype=like.dtype).clamp(0.0, 1.0)
    m = 1.0 - frac
    sigma = float(sigma_max) * frac
    return m, sigma


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


def zero_degrade_loss(
    model: nn.Module,
    cond: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    bsz = target.shape[0]
    num_steps = int(args.num_steps)
    t = torch.randint(0, num_steps, (bsz,), device=target.device)
    eps = torch.randn_like(target)
    m, sigma = zero_coefficients(t, num_steps, float(args.sigma_max), target)
    z_t = m * target + sigma * eps

    # Explicitly train the true receiver start: high channels exactly zero.
    p0 = float(args.endpoint_zero_prob)
    if p0 > 0:
        mask = (torch.rand((bsz,), device=target.device) < p0)
        if mask.any():
            t = t.clone()
            t[mask] = num_steps - 1
            z_t = z_t.clone()
            z_t[mask] = 0.0

    with make_autocast(target.device, amp_enabled, amp_dtype):
        x0_pred = model(z_t, cond, t).float()
        if float(args.clip_x0) > 0:
            x0_pred = x0_pred.clamp(-float(args.clip_x0), float(args.clip_x0))
        loss_x0 = F.mse_loss(x0_pred, target)

    return loss_x0, {
        "loss": float(loss_x0.detach().item()),
        "loss_x0": float(loss_x0.detach().item()),
        "zt_abs": float(z_t.detach().abs().mean().item()),
    }


@torch.no_grad()
def predict_from_zero(
    model: nn.Module,
    cond: torch.Tensor,
    shape: tuple[int, int, int, int],
    t_value: int,
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    z_t = torch.zeros(shape, device=cond.device, dtype=torch.float32)
    t = torch.full((shape[0],), int(t_value), device=cond.device, dtype=torch.long)
    with make_autocast(cond.device, amp_enabled, amp_dtype):
        x0_pred = model(z_t, cond, t).float()
    if float(args.clip_x0) > 0:
        x0_pred = x0_pred.clamp(-float(args.clip_x0), float(args.clip_x0))
    return x0_pred.float()


@torch.no_grad()
def refine_from_zero(
    model: nn.Module,
    cond: torch.Tensor,
    shape: tuple[int, int, int, int],
    t_start: int,
    refine_steps: int,
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    # Deterministic cold refinement. No injected random noise at inference.
    z_t = torch.zeros(shape, device=cond.device, dtype=torch.float32)
    step_ids = torch.linspace(int(t_start), 0, max(1, int(refine_steps)), device=cond.device).round().long()
    step_ids = torch.unique_consecutive(step_ids)
    last_x0 = None
    for i, ti in enumerate(step_ids):
        t = torch.full((shape[0],), int(ti.item()), device=cond.device, dtype=torch.long)
        with make_autocast(cond.device, amp_enabled, amp_dtype):
            x0_pred = model(z_t, cond, t).float()
        if float(args.clip_x0) > 0:
            x0_pred = x0_pred.clamp(-float(args.clip_x0), float(args.clip_x0))
        last_x0 = x0_pred
        if i < len(step_ids) - 1:
            next_t = torch.full((shape[0],), int(step_ids[i + 1].item()), device=cond.device, dtype=torch.long)
            m_next, _sigma_next = zero_coefficients(next_t, int(args.num_steps), float(args.sigma_max), x0_pred)
            z_t = m_next * x0_pred
    return last_x0.float() if last_x0 is not None else z_t.float()


@torch.no_grad()
def degraded_target_diagnostic(
    model: nn.Module,
    cond: torch.Tensor,
    target: torch.Tensor,
    t_value: int,
    args: argparse.Namespace,
    generator: torch.Generator,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    t = torch.full((target.shape[0],), int(t_value), device=target.device, dtype=torch.long)
    eps = torch.randn(tuple(target.shape), device=target.device, dtype=torch.float32, generator=generator)
    m, sigma = zero_coefficients(t, int(args.num_steps), float(args.sigma_max), target)
    z_t = m * target + sigma * eps
    with make_autocast(target.device, amp_enabled, amp_dtype):
        x0_pred = model(z_t, cond, t).float()
    if float(args.clip_x0) > 0:
        x0_pred = x0_pred.clamp(-float(args.clip_x0), float(args.clip_x0))
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
        "route": "receiver_norm_zero_degrade_high",
        "stage": "03D_zero_degrade_x0_high_unet_frozen_swin",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "semantic_config": asdict(cfg.semantic),
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "zero_degrade_state_dict": model.state_dict(),
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
        "prediction_type": "x0",
        "forward_degradation": "z_t=(1-t/T)*z_high_norm + sigma_max*(t/T)*eps, endpoint zero mixed",
        "snr_db": float(args.snr_db),
        "trainable_encoder": False,
        "trainable_decoder": False,
        "trainable_zero_degrade": True,
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

    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}, stage=03D_zero_degrade_x0_high, snr={float(args.snr_db):g}dB")
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

    model = ZeroDegHighUNet(
        high_channels=12,
        cond_channels=4,
        base=int(args.unet_base),
        depth=int(args.unet_depth),
        time_dim=int(args.time_dim),
        num_steps=int(args.num_steps),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    zero_eval_ts = parse_int_list(str(args.zero_eval_timesteps), int(args.num_steps))
    refine_ts = parse_int_list(str(args.refine_t_starts), int(args.num_steps))
    diag_ts = parse_int_list(str(args.diag_timesteps), int(args.num_steps))
    print(
        f"model=ZeroDegHighUNet base={args.unet_base} depth={args.unet_depth} time_dim={args.time_dim} "
        f"params={sum(p.numel() for p in model.parameters())} lr={float(args.lr):g} "
        f"T={args.num_steps} sigma_max={float(args.sigma_max):g} endpoint_zero_prob={float(args.endpoint_zero_prob):g} "
        f"clip_x0={float(args.clip_x0):g} zero_eval={zero_eval_ts} refine={refine_ts}/steps{args.refine_steps} diag={diag_ts} "
        f"aat_error={aat_err:.3e}"
    )

    train_keys = ("loss", "loss_x0", "zt_abs")
    val_keys: list[str] = [
        "psnr_base", "psnr_oracle", "target_mean", "target_std", "target_abs"
    ]
    for ts in zero_eval_ts:
        val_keys += [f"psnr_zero_t{ts}", f"mse_zero_t{ts}", f"std_zero_t{ts}"]
    for ts in refine_ts:
        val_keys += [f"psnr_refine_t{ts}", f"mse_refine_t{ts}", f"std_refine_t{ts}"]
    for ts in diag_ts:
        val_keys += [f"psnr_diag_t{ts}", f"mse_diag_t{ts}"]

    best = -1.0
    best_name = "zero_degrade_high_unet_frozen_swin_best.pth"
    latest_name = "zero_degrade_high_unet_frozen_swin_latest.pth"

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
            loss, stats = zero_degrade_loss(model, cond, target, args, amp_enabled, amp_dtype)
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
                    shape = tuple(target.shape)

                    z_base = decode_a(cond, a)
                    z_oracle = torch.cat([cond, target], dim=1)
                    x_base = decoder(z_base).float().clamp(0, 1)
                    x_oracle = decoder(z_oracle).float().clamp(0, 1)
                    stats = {
                        "psnr_base": float(psnr_per_image(x_base, imgs.float()).mean().item()),
                        "psnr_oracle": float(psnr_per_image(x_oracle, imgs.float()).mean().item()),
                        "target_mean": float(target.float().mean().item()),
                        "target_std": float(target.float().std(unbiased=False).item()),
                        "target_abs": float(target.float().abs().mean().item()),
                    }

                    for ts in zero_eval_ts:
                        high_zero = predict_from_zero(model, cond, shape, int(ts), args, amp_enabled, amp_dtype)
                        x_zero = decoder(torch.cat([cond, high_zero], dim=1)).float().clamp(0, 1)
                        stats[f"psnr_zero_t{ts}"] = float(psnr_per_image(x_zero, imgs.float()).mean().item())
                        stats[f"mse_zero_t{ts}"] = float(F.mse_loss(high_zero, target).item())
                        stats[f"std_zero_t{ts}"] = float(high_zero.float().std(unbiased=False).item())

                    for ts in refine_ts:
                        high_refine = refine_from_zero(model, cond, shape, int(ts), int(args.refine_steps), args, amp_enabled, amp_dtype)
                        x_refine = decoder(torch.cat([cond, high_refine], dim=1)).float().clamp(0, 1)
                        stats[f"psnr_refine_t{ts}"] = float(psnr_per_image(x_refine, imgs.float()).mean().item())
                        stats[f"mse_refine_t{ts}"] = float(F.mse_loss(high_refine, target).item())
                        stats[f"std_refine_t{ts}"] = float(high_refine.float().std(unbiased=False).item())

                    for ts in diag_ts:
                        gen_diag = torch.Generator(device=device)
                        gen_diag.manual_seed(int(args.seed) + 4000 + bi * 1000 + int(ts))
                        high_diag = degraded_target_diagnostic(model, cond, target, int(ts), args, gen_diag, amp_enabled, amp_dtype)
                        x_diag = decoder(torch.cat([cond, high_diag], dim=1)).float().clamp(0, 1)
                        stats[f"psnr_diag_t{ts}"] = float(psnr_per_image(x_diag, imgs.float()).mean().item())
                        stats[f"mse_diag_t{ts}"] = float(F.mse_loss(high_diag, target).item())

                    for k in val_keys:
                        val_meters[k].update(stats[k], imgs.shape[0])

            metrics.update({f"val_{k}": v.avg for k, v in val_meters.items()})
            candidate_scores = []
            for ts in zero_eval_ts:
                candidate_scores.append(metrics[f"val_psnr_zero_t{ts}"])
            for ts in refine_ts:
                candidate_scores.append(metrics[f"val_psnr_refine_t{ts}"])
            score = max(candidate_scores) if candidate_scores else -1.0
            is_best = score > best
            if is_best:
                best = score
                save_checkpoint(os.path.join(save_dir, best_name), encoder, decoder, model, a, cfg, args, metrics, epoch)
            save_checkpoint(os.path.join(save_dir, latest_name), encoder, decoder, model, a, cfg, args, metrics, epoch)

            zero_msg = " ".join(
                f"z{ts}={metrics[f'val_psnr_zero_t{ts}']:.4f}/g{metrics[f'val_psnr_zero_t{ts}']-metrics['val_psnr_base']:+.4f}/mse{metrics[f'val_mse_zero_t{ts}']:.4f}/std{metrics[f'val_std_zero_t{ts}']:.3f}"
                for ts in zero_eval_ts
            )
            refine_msg = " ".join(
                f"r{ts}={metrics[f'val_psnr_refine_t{ts}']:.4f}/g{metrics[f'val_psnr_refine_t{ts}']-metrics['val_psnr_base']:+.4f}/mse{metrics[f'val_mse_refine_t{ts}']:.4f}/std{metrics[f'val_std_refine_t{ts}']:.3f}"
                for ts in refine_ts
            )
            diag_msg = " ".join(
                f"d{ts}={metrics[f'val_psnr_diag_t{ts}']:.2f}/mse{metrics[f'val_mse_diag_t{ts}']:.4f}"
                for ts in diag_ts
            )
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"zt_abs={meters['zt_abs'].avg:.4f} | "
                f"base={metrics['val_psnr_base']:.4f} oracle={metrics['val_psnr_oracle']:.4f} "
                f"gap_oracle_best={metrics['val_psnr_oracle'] - score:+.4f} "
                f"std_gt={metrics['val_target_std']:.4f} mean_gt={metrics['val_target_mean']:+.4f} abs_gt={metrics['val_target_abs']:.4f} "
                f"{zero_msg} {refine_msg} {diag_msg} score={score:.4f} "
                f"aat_err={semiorth_error(a):.2e} {'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"zt_abs={meters['zt_abs'].avg:.4f}"
            )

    print(f"best_val_psnr_zero_or_refine={best:.4f}")


if __name__ == "__main__":
    main()
