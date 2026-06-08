#!/usr/bin/env python
"""Coarse-to-high residual refiner for receiver-normalized two-frequency Swin.

Stage 02C freezes the Stage 01B encoder, decoder, and ARReceiver P.  The
trainable residual refiner R learns whether there is deterministic correction
left after the coarse high prediction:

    y = z_low / scale + AWGN
    h = z_high / scale
    h0 = P(y).detach()
    r = h - h0

Training degradation is applied to the high state, not to a zero residual:

    h_t = h0 + m_t * (h - h0) + sigma_t * eps

Inference is single step and deterministic:

    r_hat = R(h0, y, h0, T)
    x_recv = Decoder([y, h0 + r_hat])
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
    ARReceiver,
    build_semantic_modules,
    decode_a,
    encode_a,
    fixed_select_a,
    make_loaders,
)
from train_route_a_sc import AverageMeter, TeeStream  # noqa: E402


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = resolve_path(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Coarse-degrade residual refiner @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
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
        groups = min(8, int(channels))
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


class CoarseResidualRefinerUNet(nn.Module):
    """Latent U-Net that predicts residual high correction r_hat.

    Input channels are concat(h_t, y, h0) = 12 + 4 + 12 = 28.
    """

    def __init__(
        self,
        in_channels: int = 28,
        out_channels: int = 12,
        base: int = 128,
        depth: int = 3,
        time_dim: int = 256,
        num_steps: int = 200,
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        h1, h2, h3 = int(base), int(base) * 2, int(base) * 4
        self.time = TimeEmbedding(int(time_dim))
        self.stem = nn.Sequential(nn.Conv2d(int(in_channels), h1, 3, padding=1), nn.SiLU())
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
        self.out = nn.Conv2d(h1, int(out_channels), 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h_t: torch.Tensor, y: torch.Tensor, h0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time(t, self.num_steps).to(dtype=h_t.dtype)
        x = torch.cat([h_t, y.to(dtype=h_t.dtype), h0.to(dtype=h_t.dtype)], dim=1)
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
        description="Train frozen Stage 01B coarse-to-high residual refiner",
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

    p.add_argument("--init_stage01b_ckpt", type=str, required=True)
    p.add_argument("--snr_db", type=float, default=6.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--sigma_max", type=float, default=0.05)
    p.add_argument("--lambda_res", type=float, default=0.03)
    p.add_argument("--clip_r", type=float, default=2.0)
    p.add_argument("--unet_base", type=int, default=128)
    p.add_argument("--unet_depth", type=int, default=3)
    p.add_argument("--time_dim", type=int, default=256)

    p.add_argument("--pred_hidden", type=int, default=160)
    p.add_argument("--pred_depth", type=int, default=4)
    p.add_argument("--pred_use_scale", action="store_true", default=False)
    p.add_argument("--encoder_use_vae", action="store_true", default=False)
    p.add_argument("--no_encoder_vae", action="store_false", dest="encoder_use_vae")
    p.add_argument("--lambda_kl", type=float, default=0.0)
    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints-ar/twofreq_receiver_norm_swin_snr6_v1/02C_coarse_degrade_residual_refiner_frozen_swin_snr6_v1")
    p.add_argument("--log_file", type=str, default="checkpoints-ar/twofreq_receiver_norm_swin_snr6_v1/02C_coarse_degrade_residual_refiner_frozen_swin_snr6_v1/train.log")
    return p.parse_args()


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(bool(trainable))


def load_stage01b(
    path: str,
    encoder: nn.Module,
    decoder: nn.Module,
    predictor: nn.Module,
    device: torch.device,
) -> dict:
    ckpt_path = resolve_path(path)
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    pred_state = ckpt.get("predictor_state_dict", ckpt.get("ar_state_dict"))
    if pred_state is None:
        raise KeyError(f"no predictor_state_dict/ar_state_dict in {ckpt_path}")
    predictor.load_state_dict(pred_state, strict=True)
    metrics = ckpt.get("metrics", {})
    print(
        f"loaded frozen Stage01B: {ckpt_path}, stage={ckpt.get('stage', 'unknown')}, "
        f"epoch={ckpt.get('epoch', 'unknown')}, val_recv={metrics.get('val_psnr_recv', 'na')}, "
        f"val_base={metrics.get('val_psnr_base', 'na')}, val_oracle={metrics.get('val_psnr_oracle', 'na')}"
    )
    return ckpt


def coarse_coefficients(t: torch.Tensor, num_steps: int, sigma_max: float, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denom = max(1, int(num_steps) - 1)
    frac = t.float().view(-1, 1, 1, 1) / float(denom)
    frac = frac.to(device=like.device, dtype=like.dtype).clamp(0.0, 1.0)
    m = 1.0 - frac
    sigma = float(sigma_max) * frac
    return m, sigma


@torch.no_grad()
def make_coarse_batch(
    imgs: torch.Tensor,
    encoder: nn.Module,
    predictor: nn.Module,
    a: torch.Tensor,
    snr_db: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = imgs.shape[0]
    snr_b = torch.full((bsz,), float(snr_db), device=imgs.device, dtype=torch.float32)
    z, _mu, _logvar = encoder.encode(imgs, sample=False)
    z = z.float()
    y4 = encode_a(z, a)
    y4_norm, _y4_raw, scale = power_normalize_awgn(y4, snr_b, generator=generator)
    scale_view = scale.float().view(-1, 1, 1, 1).clamp_min(1e-12)
    h = z[:, 4:16].float() / scale_view

    pred_dtype = next(predictor.parameters()).dtype
    _z_recv, pred_groups = predictor(
        y4_norm.to(dtype=pred_dtype),
        y4_raw=y4_norm.to(dtype=pred_dtype),
        y4_norm=y4_norm.to(dtype=pred_dtype),
        scale=scale,
        snr_db=snr_b,
        z_gt=None,
        teacher_prob=0.0,
    )
    h0 = torch.cat([g.float() for g in pred_groups], dim=1)
    return y4_norm.float(), h.float(), h0.float().detach()


def train_refiner_loss(
    model: nn.Module,
    decoder: nn.Module,
    imgs: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor,
    h0: torch.Tensor,
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    r = h - h0
    bsz = h.shape[0]
    t = torch.randint(0, int(args.num_steps), (bsz,), device=h.device)
    eps = torch.randn_like(h)
    m, sigma = coarse_coefficients(t, int(args.num_steps), float(args.sigma_max), h)
    h_t = h0 + m * r + sigma * eps

    with make_autocast(h.device, amp_enabled, amp_dtype):
        r_hat = model(h_t, y, h0, t).float()
        if float(args.clip_r) > 0:
            r_hat = r_hat.clamp(-float(args.clip_r), float(args.clip_r))
        h_hat = h0 + r_hat
        x_recv = decoder(torch.cat([y, h_hat], dim=1)).float().clamp(0, 1)
        loss_img = F.mse_loss(x_recv, imgs.float())
        loss_res = F.mse_loss(r_hat, r)
        loss = loss_img + float(args.lambda_res) * loss_res

    return loss, {
        "loss": float(loss.detach().item()),
        "loss_img": float(loss_img.detach().item()),
        "loss_res": float(loss_res.detach().item()),
        "ht_abs": float(h_t.detach().abs().mean().item()),
        "r_mse": float(loss_res.detach().item()),
        "r_gt_rms": float(r.float().square().mean().sqrt().detach().item()),
        "r_hat_rms": float(r_hat.float().square().mean().sqrt().detach().item()),
        "psnr_recv": float(psnr_per_image(x_recv, imgs.float()).mean().item()),
    }


@torch.no_grad()
def eval_single_step(
    model: nn.Module,
    decoder: nn.Module,
    imgs: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor,
    h0: torch.Tensor,
    a: torch.Tensor,
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    r = h - h0
    t_terminal = int(args.num_steps) - 1
    t = torch.full((h.shape[0],), t_terminal, device=h.device, dtype=torch.long)
    with make_autocast(h.device, amp_enabled, amp_dtype):
        r_hat = model(h0, y, h0, t).float()
        if float(args.clip_r) > 0:
            r_hat = r_hat.clamp(-float(args.clip_r), float(args.clip_r))
    h_hat = h0 + r_hat

    x_base = decoder(decode_a(y, a)).float().clamp(0, 1)
    x_coarse = decoder(torch.cat([y, h0], dim=1)).float().clamp(0, 1)
    x_recv = decoder(torch.cat([y, h_hat], dim=1)).float().clamp(0, 1)
    x_oracle = decoder(torch.cat([y, h], dim=1)).float().clamp(0, 1)
    base = float(psnr_per_image(x_base, imgs.float()).mean().item())
    coarse = float(psnr_per_image(x_coarse, imgs.float()).mean().item())
    recv = float(psnr_per_image(x_recv, imgs.float()).mean().item())
    oracle = float(psnr_per_image(x_oracle, imgs.float()).mean().item())
    return {
        "base": base,
        "coarse": coarse,
        "recv": recv,
        "oracle": oracle,
        "gain_vs_base": recv - base,
        "gain_vs_coarse": recv - coarse,
        "gap_oracle": oracle - recv,
        "r_mse": float(F.mse_loss(r_hat, r).item()),
        "r_gt_rms": float(r.float().square().mean().sqrt().item()),
        "r_hat_rms": float(r_hat.float().square().mean().sqrt().item()),
        "h0_rms": float(h0.float().square().mean().sqrt().item()),
        "h_oracle_rms": float(h.float().square().mean().sqrt().item()),
    }


def save_checkpoint(
    path: str,
    encoder: nn.Module,
    decoder: nn.Module,
    predictor: nn.Module,
    model: nn.Module,
    a: torch.Tensor,
    cfg,
    args: argparse.Namespace,
    metrics: dict,
    epoch: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "route": "twofreq_receiver_norm_coarse_degrade_residual_refiner",
        "stage": "02C_coarse_degrade_residual_refiner_frozen_swin",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "semantic_config": asdict(cfg.semantic),
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "residual_refiner_state_dict": model.state_dict(),
        "a_matrix": a.detach().cpu(),
        "aat": (a @ a.t()).detach().cpu(),
        "aat_error": semiorth_error(a),
        "fixed_channel_codec": True,
        "channel_encoder": "A=[I4,0]",
        "channel_decoder": "A^T zero-fill",
        "power_norm_after_channel_encoder": True,
        "receiver_observation": "norm",
        "snr_db": float(args.snr_db),
        "forward_degradation": "h_t=h0+(1-t/T)*(h-h0)+sigma_max*(t/T)*eps",
        "inference": "single_step_no_noise: r_hat=R(h0,y,h0,T), h_hat=h0+r_hat",
        "loss": "MSE(D([y,h0+r_hat]),x)+lambda_res*MSE(r_hat,h-h0)",
        "score_metric": "val_gain_vs_coarse",
        "trainable_encoder": False,
        "trainable_decoder": False,
        "trainable_predictor": False,
        "trainable_residual_refiner": True,
    }
    torch.save(payload, path)
    torch.save(
        {
            "route": payload["route"],
            "stage": payload["stage"],
            "part": "residual_refiner",
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
            "state_dict": model.state_dict(),
        },
        os.path.join(os.path.dirname(path), "coarse_residual_refiner_snr6.pth"),
    )


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
    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}, stage=02C, snr={float(args.snr_db):g}dB")
    print(
        "rule: frozen Stage01B Encoder/Decoder/P; h0=P(y).detach(); "
        "degrade high toward coarse h0; eval single-step R(h0,y,h0,T); score=recv-coarse"
    )

    train_ds, val_ds, train_loader, val_loader = make_loaders(args, device)
    print(f"train={len(train_ds)} valid={len(val_ds)} batch={args.batch_size} crop={args.crop_size}")

    encoder, decoder, cfg = build_semantic_modules(device, args)
    predictor = ARReceiver(hidden=int(args.pred_hidden), depth=int(args.pred_depth), use_scale=bool(args.pred_use_scale)).to(device)
    stage01b_ckpt = load_stage01b(args.init_stage01b_ckpt, encoder, decoder, predictor, device)
    _ = stage01b_ckpt
    set_trainable(encoder, False)
    set_trainable(decoder, False)
    set_trainable(predictor, False)
    encoder.eval()
    decoder.eval()
    predictor.eval()

    a = fixed_select_a(device=device, dtype=torch.float32)
    aat_err = semiorth_error(a)
    if aat_err > 1e-7:
        raise RuntimeError(f"A A^T != I4, err={aat_err:.3e}")

    model = CoarseResidualRefinerUNet(
        in_channels=28,
        out_channels=12,
        base=int(args.unet_base),
        depth=int(args.unet_depth),
        time_dim=int(args.time_dim),
        num_steps=int(args.num_steps),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    print(
        f"model=CoarseResidualRefinerUNet in=28 out=12 base={args.unet_base} depth={args.unet_depth} "
        f"time_dim={args.time_dim} params={sum(p.numel() for p in model.parameters())} lr={float(args.lr):g} "
        f"T={args.num_steps} sigma_max={float(args.sigma_max):g} lambda_res={float(args.lambda_res):g} "
        f"clip_r={float(args.clip_r):g} aat_error={aat_err:.3e}"
    )

    train_keys = ("loss", "loss_img", "loss_res", "ht_abs", "r_mse", "r_gt_rms", "r_hat_rms", "psnr_recv")
    val_keys = (
        "base",
        "coarse",
        "recv",
        "oracle",
        "gain_vs_base",
        "gain_vs_coarse",
        "gap_oracle",
        "r_mse",
        "r_gt_rms",
        "r_hat_rms",
        "h0_rms",
        "h_oracle_rms",
    )
    best = float("-inf")
    best_name = "coarse_degrade_residual_refiner_frozen_swin_best.pth"
    latest_name = "coarse_degrade_residual_refiner_frozen_swin_latest.pth"

    for epoch in range(1, int(args.epochs) + 1):
        model.train(True)
        encoder.eval()
        decoder.eval()
        predictor.eval()
        meters = {k: AverageMeter() for k in train_keys}
        for bi, batch in enumerate(train_loader):
            if int(args.max_train_batches) > 0 and bi >= int(args.max_train_batches):
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=True)
            y, h, h0 = make_coarse_batch(imgs, encoder, predictor, a, float(args.snr_db), generator=None)

            optimizer.zero_grad(set_to_none=True)
            loss, stats = train_refiner_loss(model, decoder, imgs, y, h, h0, args, amp_enabled, amp_dtype)
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
                    y, h, h0 = make_coarse_batch(imgs, encoder, predictor, a, float(args.snr_db), generator=gen_awgn)
                    stats = eval_single_step(model, decoder, imgs, y, h, h0, a, args, amp_enabled, amp_dtype)
                    for k in val_keys:
                        val_meters[k].update(stats[k], imgs.shape[0])

            metrics.update({f"val_{k}": v.avg for k, v in val_meters.items()})
            score = metrics["val_gain_vs_coarse"]
            is_best = score > best
            if is_best:
                best = score
                save_checkpoint(os.path.join(save_dir, best_name), encoder, decoder, predictor, model, a, cfg, args, metrics, epoch)
            save_checkpoint(os.path.join(save_dir, latest_name), encoder, decoder, predictor, model, a, cfg, args, metrics, epoch)
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"img={meters['loss_img'].avg:.6f} res={meters['loss_res'].avg:.6f} | "
                f"base={metrics['val_base']:.4f} coarse={metrics['val_coarse']:.4f} "
                f"recv={metrics['val_recv']:.4f} oracle={metrics['val_oracle']:.4f} "
                f"gain_vs_base={metrics['val_gain_vs_base']:+.4f} "
                f"gain_vs_coarse={metrics['val_gain_vs_coarse']:+.4f} "
                f"gap_oracle={metrics['val_gap_oracle']:+.4f} "
                f"r_mse={metrics['val_r_mse']:.4f} r_gt_rms={metrics['val_r_gt_rms']:.4f} "
                f"r_hat_rms={metrics['val_r_hat_rms']:.4f} h0_rms={metrics['val_h0_rms']:.4f} "
                f"h_rms={metrics['val_h_oracle_rms']:.4f} score(recv-coarse)={score:+.4f} "
                f"aat_err={semiorth_error(a):.2e} {'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"img={meters['loss_img'].avg:.6f} res={meters['loss_res'].avg:.6f} "
                f"train_recv={meters['psnr_recv'].avg:.4f} r_gt_rms={meters['r_gt_rms'].avg:.4f} "
                f"r_hat_rms={meters['r_hat_rms'].avg:.4f}"
            )

    print(f"best_val_gain_vs_coarse={best:+.4f}")


if __name__ == "__main__":
    main()
