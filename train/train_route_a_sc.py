#!/usr/bin/env python
"""Route A Stage 2: fixed orthogonal truncation + dual-branch Swin-VAE pretraining.

This trains SemanticEncoder/SemanticDecoder with a frozen 4x16 DCT projection A:

  full branch: x -> z16 -> Decoder(z16)
  sub branch : x -> z16 -> A^T A z16 -> Decoder(A^T A z16)

The checkpoint saves split encoder/decoder weights plus the fixed A matrix so
Stage 3 can train the unconditional latent diffusion on full z16 latents.
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
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import SystemConfig, get_cifar10_config, get_div2k_config
from src.cddm_mimo_ddnm.datasets import get_cifar10_loaders, get_div2k_loaders
from src.cddm_mimo_ddnm.loss import kl_loss
from src.cddm_mimo_ddnm.modules.orthogonal_projection import FixedOrthogonalProjector
from src.cddm_mimo_ddnm.modules.semantic_codec import SemanticDecoder, SemanticEncoder


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


def setup_log_file(log_path: str | None):
    if not log_path:
        return None
    abs_path = log_path if os.path.isabs(log_path) else os.path.join(PROJECT_ROOT, log_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Route A SC session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean of per-image PSNR values, matching rule.md."""
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.mean().item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Route A Stage 2 - fixed A dual-branch Swin-VAE training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default="div2k", choices=["cifar10", "div2k"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--use_lmdb", action="store_true")
    p.add_argument("--train_lmdb_path", type=str, default=None)
    p.add_argument("--val_lmdb_path", type=str, default=None)
    p.add_argument("--cache_decoded", dest="cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", dest="cache_decoded", action="store_false")

    p.add_argument("--embed_dim", type=int, default=16)
    p.add_argument("--a_out_dim", type=int, default=4)
    p.add_argument("--a_init", type=str, default="dct", choices=["dct"])
    p.add_argument("--channel_noise", type=str, default="none", choices=["none", "awgn"],
                   help="Subspace branch channel noise. 'awgn' injects noise into Az before A^T.")
    p.add_argument("--snr_db", type=float, default=12.0,
                   help="AWGN SNR in dB when --channel_noise awgn.")
    p.add_argument("--init_sc_encoder_ckpt", type=str,
                   default=os.path.join(PROJECT_ROOT, "checkpoints-val/sc/sc_encoder_div2k_c16.pth"))
    p.add_argument("--init_sc_decoder_ckpt", type=str,
                   default=os.path.join(PROJECT_ROOT, "checkpoints-val/sc/sc_decoder_div2k_c16.pth"))

    p.add_argument("--lambda_sub", type=float, default=1.0)
    p.add_argument("--lambda_reg", type=float, default=1e-3)
    p.add_argument("--lambda_kl_max", type=float, default=1e-6)
    p.add_argument("--kl_anneal_epochs", type=float, default=50.0)
    p.add_argument("--blur_kernel", type=int, default=15)
    p.add_argument("--blur_sigma", type=float, default=3.0)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--eval_every_epochs", type=int, default=10)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_file", type=str,
                   default="log-v1/route_a/sc_dct_c4.txt")
    p.add_argument("--save_dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "checkpoints-val-v1/route_a/sc_dct_c4"))
    p.add_argument("--resume", type=str, default="")
    args = p.parse_args()
    if args.dataset == "div2k" and args.data_dir is None:
        args.data_dir = "/workspace/yongjia/datasets/DIV2K"
    if args.blur_kernel < 1 or args.blur_kernel % 2 == 0:
        raise SystemExit("--blur_kernel must be a positive odd integer")
    return args


def build_config(args: argparse.Namespace) -> SystemConfig:
    cfg = get_div2k_config() if args.dataset == "div2k" else get_cifar10_config()
    cfg.semantic.embed_dim = int(args.embed_dim)
    cfg.semantic.use_vae = True
    cfg.semantic.lambda_kl = float(args.lambda_kl_max)
    cfg.channel.input_channels = int(args.embed_dim)
    cfg.channel.channel_symbols = int(args.a_out_dim)
    cfg.unet_uncond.input_channel = int(args.embed_dim)
    return cfg


def build_dataloaders(args: argparse.Namespace):
    if args.dataset == "div2k":
        return get_div2k_loaders(
            data_dir=args.data_dir or "",
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            num_workers=args.num_workers,
            distributed=False,
            use_lmdb=args.use_lmdb,
            train_lmdb_path=args.train_lmdb_path,
            val_lmdb_path=args.val_lmdb_path,
            val_num_workers=args.val_num_workers,
            prefetch_factor=args.prefetch_factor,
            cache_decoded=bool(args.cache_decoded),
        )[:2]
    return get_cifar10_loaders(
        data_dir=args.data_dir or "",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=False,
        val_num_workers=args.val_num_workers,
        prefetch_factor=args.prefetch_factor,
    )[:2]


def build_semantic_modules(cfg: SystemConfig, device: torch.device) -> tuple[SemanticEncoder, SemanticDecoder]:
    sc = cfg.semantic
    enc = SemanticEncoder(
        in_channels=sc.image_channels,
        embed_dim=sc.embed_dim,
        patch_size=sc.patch_size,
        num_heads=sc.num_heads,
        window_size=sc.window_size,
        num_blocks=sc.num_swin_blocks,
        stage_embed_dims=sc.stage_embed_dims,
        stage_depths=sc.stage_depths,
        stage_num_heads=sc.stage_num_heads,
        stem_stride=sc.stem_stride,
        stage_downsample=sc.stage_downsample,
        use_vae=True,
    ).to(device)
    dec = SemanticDecoder(
        out_channels=sc.image_channels,
        embed_dim=sc.embed_dim,
        patch_size=sc.patch_size,
        num_heads=sc.num_heads,
        window_size=sc.window_size,
        num_refine_blocks=sc.num_decoder_refine_blocks,
        stage_embed_dims=sc.stage_embed_dims,
        stage_depths=sc.stage_depths,
        stage_num_heads=sc.stage_num_heads,
        stem_stride=sc.stem_stride,
        stage_downsample=sc.stage_downsample,
    ).to(device)
    return enc, dec


def load_state_dict_from_ckpt(module: nn.Module, ckpt_path: str, name: str) -> None:
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        print(f"  [{name}] init checkpoint not found, skip: {ckpt_path}")
        return
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    missing, unexpected = module.load_state_dict(sd, strict=False)
    print(f"  [{name}] loaded init: {ckpt_path}")
    if missing:
        print(f"    missing={len(missing)}: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"    unexpected={len(unexpected)}: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")


class GaussianBlur(nn.Module):
    def __init__(self, channels: int, kernel_size: int, sigma: float) -> None:
        super().__init__()
        half = kernel_size // 2
        coords = torch.arange(kernel_size, dtype=torch.float32) - half
        kernel_1d = torch.exp(-(coords ** 2) / (2.0 * float(sigma) ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        weight = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.groups = int(channels)
        self.padding = int(half)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        return F.conv2d(x, weight, padding=self.padding, groups=self.groups)


class RouteASemanticVAE(nn.Module):
    def __init__(
        self,
        encoder: SemanticEncoder,
        decoder: SemanticDecoder,
        projector: FixedOrthogonalProjector,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector

    def forward(self, x: torch.Tensor, *, sample: bool) -> dict[str, torch.Tensor | None]:
        z, mu, logvar = self.encoder.encode(x, sample=sample)
        y_clean = self.projector.encode(z)
        z_sub = self.projector.decode(y_clean)
        return {
            "z": z,
            "z_sub": z_sub,
            "z_null": z - z_sub,
            "y_clean": y_clean,
            "x_full": self.decoder(z),
            "x_sub": self.decoder(z_sub),
            "mu": mu,
            "logvar": logvar,
        }

    def forward_awgn(
        self,
        x: torch.Tensor,
        *,
        sample: bool,
        snr_db: float,
    ) -> dict[str, torch.Tensor | None]:
        z, mu, logvar = self.encoder.encode(x, sample=sample)
        y_clean = self.projector.encode(z)
        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        dims = tuple(range(1, y_clean.ndim))
        signal_power = y_clean.float().pow(2).mean(dim=dims, keepdim=True).clamp_min(1e-12)
        noise_std = torch.sqrt(signal_power / snr_linear).to(device=y_clean.device, dtype=y_clean.dtype)
        noise = torch.randn_like(y_clean) * noise_std
        y_noisy = y_clean + noise
        z_sub = self.projector.decode(y_noisy)
        return {
            "z": z,
            "z_sub": z_sub,
            "z_null": z - self.projector.decode(y_clean),
            "y_clean": y_clean,
            "y_noisy": y_noisy,
            "channel_noise_std": noise_std.detach().float().mean(),
            "x_full": self.decoder(z),
            "x_sub": self.decoder(z_sub),
            "mu": mu,
            "logvar": logvar,
        }


def kl_weight_for_epoch(epoch: int, args: argparse.Namespace) -> float:
    if args.lambda_kl_max <= 0:
        return 0.0
    if args.kl_anneal_epochs <= 0:
        return float(args.lambda_kl_max)
    return float(args.lambda_kl_max) * min(1.0, float(epoch + 1) / float(args.kl_anneal_epochs))


def compute_route_a_loss(
    out: dict[str, torch.Tensor | None],
    x: torch.Tensor,
    x_blur: torch.Tensor,
    *,
    args: argparse.Namespace,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    x_full = out["x_full"]
    x_sub = out["x_sub"]
    z_null = out["z_null"]
    mu = out["mu"]
    logvar = out["logvar"]
    assert isinstance(x_full, torch.Tensor)
    assert isinstance(x_sub, torch.Tensor)
    assert isinstance(z_null, torch.Tensor)

    loss_full = F.mse_loss(x_full, x)
    loss_sub = F.mse_loss(x_sub, x_blur)
    loss_reg = torch.mean(z_null.float() ** 2).to(dtype=loss_full.dtype)
    if mu is not None and logvar is not None and kl_weight > 0:
        kl_raw = kl_loss(mu.float(), logvar.float()).to(dtype=loss_full.dtype)
    else:
        kl_raw = loss_full.new_tensor(0.0)
    loss = loss_full + float(args.lambda_sub) * loss_sub + float(args.lambda_reg) * loss_reg
    loss = loss + float(kl_weight) * kl_raw
    stats = {
        "loss": float(loss.detach().item()),
        "loss_full": float(loss_full.detach().item()),
        "loss_sub": float(loss_sub.detach().item()),
        "loss_reg": float(loss_reg.detach().item()),
        "kl_raw": float(kl_raw.detach().item()),
        "kl_weight": float(kl_weight),
    }
    if isinstance(out.get("channel_noise_std"), torch.Tensor):
        stats["channel_noise_std"] = float(out["channel_noise_std"].detach().item())
    else:
        stats["channel_noise_std"] = 0.0
    return loss, stats


def save_route_a_checkpoint(
    path: str,
    model: RouteASemanticVAE,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
    cfg: SystemConfig,
    *,
    save_split: bool = True,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    common = {
        "route": "A_fixed_orthogonal_dct",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "semantic_config": asdict(cfg.semantic),
        "projection_init": model.projector.init,
        "projection_A": model.projector.A.detach().cpu(),
        "projection_orth_error": model.projector.orthogonality_error(),
        "channel_noise": args.channel_noise,
        "snr_db": float(args.snr_db),
    }
    torch.save(
        {
            **common,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )
    if save_split:
        enc_path = os.path.join(os.path.dirname(path), f"sc_encoder_{args.dataset}_c{args.embed_dim}.pth")
        dec_path = os.path.join(os.path.dirname(path), f"sc_decoder_{args.dataset}_c{args.embed_dim}.pth")
        torch.save({**common, "part": "semantic_encoder", "state_dict": model.encoder.state_dict()}, enc_path)
        torch.save({**common, "part": "semantic_decoder", "state_dict": model.decoder.state_dict()}, dec_path)


def load_resume(path: str, model: RouteASemanticVAE, optimizer, scheduler) -> tuple[int, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(obj["model_state_dict"], strict=False)
    if optimizer is not None and obj.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(obj["optimizer_state_dict"])
    if scheduler is not None and obj.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(obj["scheduler_state_dict"])
    epoch = int(obj.get("epoch", -1)) + 1
    metrics = obj.get("metrics", {})
    print(f"  resumed: {path} -> start_epoch={epoch}")
    return epoch, metrics


def make_scheduler(args: argparse.Namespace, optimizer, steps_per_epoch: int):
    total_steps = max(1, int(args.epochs) * max(1, steps_per_epoch))
    warmup_steps = min(max(0, int(args.warmup_steps)), max(1, total_steps - 1))
    min_ratio = max(0.0, float(args.min_lr_ratio))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-6, float(step + 1) / float(warmup_steps))
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        t = min(max(t, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: RouteASemanticVAE,
    blur: GaussianBlur,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict:
    model.train()
    meters = {k: AverageMeter() for k in (
        "loss", "loss_full", "loss_sub", "loss_reg", "kl_raw", "channel_noise_std",
        "psnr_full", "psnr_sub"
    )}
    kl_weight = kl_weight_for_epoch(epoch, args)
    accum = max(1, int(args.grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    max_batches = int(args.max_train_batches)
    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device, non_blocking=True)
        should_step = ((i + 1) % accum == 0) or (i + 1 == len(loader)) or (
            max_batches > 0 and i + 1 >= max_batches
        )
        with torch.autocast("cuda", **autocast_kw):
            x_blur = blur(x)
            if args.channel_noise == "awgn":
                out = model.forward_awgn(x, sample=True, snr_db=float(args.snr_db))
            else:
                out = model(x, sample=True)
            loss, stats = compute_route_a_loss(out, x, x_blur, args=args, kl_weight=kl_weight)
            loss_back = loss / float(accum)
        if scaler.is_enabled():
            scaler.scale(loss_back).backward()
        else:
            loss_back.backward()
        if should_step:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad_norm))
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        bs = x.shape[0]
        for key in ("loss", "loss_full", "loss_sub", "loss_reg", "kl_raw", "channel_noise_std"):
            meters[key].update(stats[key], bs)
        x_full = out["x_full"]
        x_sub = out["x_sub"]
        assert isinstance(x_full, torch.Tensor)
        assert isinstance(x_sub, torch.Tensor)
        meters["psnr_full"].update(compute_psnr(x_full.detach().float().clamp(0, 1), x.float()), bs)
        meters["psnr_sub"].update(compute_psnr(x_sub.detach().float().clamp(0, 1), x_blur.float()), bs)

        if (i + 1) % args.log_freq == 0 or i + 1 == len(loader) or (
            max_batches > 0 and i + 1 >= max_batches
        ):
            elapsed = time.time() - t0
            it_s = elapsed / max(1, i + 1)
            print(
                f"  [{epoch+1}/{args.epochs}][{i+1}/{len(loader)}] "
                f"loss={meters['loss'].avg:.5f} full={meters['loss_full'].avg:.5f} "
                f"sub={meters['loss_sub'].avg:.5f} reg={meters['loss_reg'].avg:.5f} "
                f"kl={meters['kl_raw'].avg:.5f}*{kl_weight:.2e} "
                f"noise_std={meters['channel_noise_std'].avg:.3e} "
                f"psnr_full={meters['psnr_full'].avg:.2f} psnr_sub={meters['psnr_sub'].avg:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} {it_s:.2f}s/it"
            )
    out = {k: meter.avg for k, meter in meters.items()}
    out["kl_weight"] = kl_weight
    return out


@torch.no_grad()
def validate(
    model: RouteASemanticVAE,
    blur: GaussianBlur,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict:
    model.eval()
    meters = {k: AverageMeter() for k in (
        "loss", "loss_full", "loss_sub", "loss_reg", "kl_raw", "channel_noise_std",
        "psnr_full", "psnr_sub"
    )}
    kl_weight = kl_weight_for_epoch(epoch, args)
    max_batches = int(args.max_val_batches)
    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", **autocast_kw):
            x_blur = blur(x)
            if args.channel_noise == "awgn":
                out = model.forward_awgn(x, sample=False, snr_db=float(args.snr_db))
            else:
                out = model(x, sample=False)
            _loss, stats = compute_route_a_loss(out, x, x_blur, args=args, kl_weight=kl_weight)
        bs = x.shape[0]
        for key in ("loss", "loss_full", "loss_sub", "loss_reg", "kl_raw", "channel_noise_std"):
            meters[key].update(stats[key], bs)
        x_full = out["x_full"]
        x_sub = out["x_sub"]
        assert isinstance(x_full, torch.Tensor)
        assert isinstance(x_sub, torch.Tensor)
        meters["psnr_full"].update(compute_psnr(x_full.float().clamp(0, 1), x.float()), bs)
        meters["psnr_sub"].update(compute_psnr(x_sub.float().clamp(0, 1), x_blur.float()), bs)
    out = {k: meter.avg for k, meter in meters.items()}
    out["kl_weight"] = kl_weight
    return out


def main() -> None:
    args = parse_args()
    setup_log_file(args.log_file)
    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_enabled = args.amp_dtype != "none" and device.type == "cuda"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.amp_dtype, torch.bfloat16)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    cfg = build_config(args)
    enc, dec = build_semantic_modules(cfg, device)
    load_state_dict_from_ckpt(enc, args.init_sc_encoder_ckpt, "semantic_encoder")
    load_state_dict_from_ckpt(dec, args.init_sc_decoder_ckpt, "semantic_decoder")

    projector = FixedOrthogonalProjector(
        in_dim=int(args.embed_dim),
        out_dim=int(args.a_out_dim),
        init=args.a_init,
    ).to(device)
    for p in projector.parameters():
        p.requires_grad = False
    model = RouteASemanticVAE(enc, dec, projector).to(device)
    blur = GaussianBlur(cfg.semantic.image_channels, args.blur_kernel, args.blur_sigma).to(device)

    train_loader, val_loader = build_dataloaders(args)
    effective_train_len = int(args.max_train_batches) if args.max_train_batches > 0 else len(train_loader)
    steps_per_epoch = max(1, math.ceil(effective_train_len / max(1, int(args.grad_accum_steps))))
    optimizer = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
    )
    scheduler = make_scheduler(args, optimizer, steps_per_epoch)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        start_epoch, metrics = load_resume(args.resume, model, optimizer, scheduler)
        best_loss = float(metrics.get("v_loss", best_loss))

    print("=" * 80)
    print("Route A Stage 2: fixed orthogonal DCT A + dual-branch Swin-VAE")
    print(f"  dataset        : {args.dataset}")
    print(f"  data_dir       : {args.data_dir}")
    print(f"  device         : {device}")
    if device.type == "cuda":
        print(f"  gpu            : {torch.cuda.get_device_name(0)}")
    print(f"  latent C       : {args.embed_dim}")
    print(f"  A shape        : {args.a_out_dim} x {args.embed_dim}")
    print(f"  A init         : {args.a_init}")
    print(f"  A orth error   : {projector.orthogonality_error():.3e}")
    print(f"  channel_noise  : {args.channel_noise}")
    if args.channel_noise == "awgn":
        print(f"  train/test SNR : {float(args.snr_db):g} dB")
    print(f"  lambdas        : sub={args.lambda_sub:g}, reg={args.lambda_reg:g}, kl_max={args.lambda_kl_max:g}")
    print(f"  KL anneal      : {args.kl_anneal_epochs:g} epochs")
    print(f"  blur           : kernel={args.blur_kernel}, sigma={args.blur_sigma:g}")
    print(f"  batches        : train={len(train_loader)}, val={len(val_loader)}")
    print(f"  save_dir       : {args.save_dir}")
    print("=" * 80)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, f"route_a_sc_{args.dataset}_c{args.embed_dim}_best.pth")
    last_path = os.path.join(args.save_dir, f"route_a_sc_{args.dataset}_c{args.embed_dim}_last.pth")
    wall = time.time()
    last_train_metrics: dict = {}

    for epoch in range(start_epoch, int(args.epochs)):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_metrics = train_one_epoch(
            model, blur, train_loader, optimizer, scheduler, scaler,
            device, args, epoch, amp_enabled, amp_dtype,
        )
        last_train_metrics = train_metrics
        print(
            f"  train: loss={train_metrics['loss']:.5f} "
            f"full={train_metrics['loss_full']:.5f} sub={train_metrics['loss_sub']:.5f} "
            f"reg={train_metrics['loss_reg']:.5f} "
            f"psnr_full={train_metrics['psnr_full']:.3f}dB psnr_sub={train_metrics['psnr_sub']:.3f}dB"
        )

        should_eval = ((epoch + 1) % max(1, int(args.eval_every_epochs)) == 0) or (epoch + 1 == int(args.epochs))
        if should_eval:
            val_metrics = validate(model, blur, val_loader, device, args, epoch, amp_enabled, amp_dtype)
            merged = {
                **train_metrics,
                **{f"v_{k}": v for k, v in val_metrics.items()},
            }
            print(
                f"  val: loss={val_metrics['loss']:.5f} "
                f"full={val_metrics['loss_full']:.5f} sub={val_metrics['loss_sub']:.5f} "
                f"reg={val_metrics['loss_reg']:.5f} "
                f"psnr_full={val_metrics['psnr_full']:.3f}dB psnr_sub={val_metrics['psnr_sub']:.3f}dB"
            )
            if val_metrics["loss"] < best_loss:
                best_loss = float(val_metrics["loss"])
                save_route_a_checkpoint(best_path, model, optimizer, scheduler, epoch, merged, args, cfg)
                print(f"  *** save best -> {best_path}  val_loss={best_loss:.6f} ***")
        if epoch + 1 == int(args.epochs):
            save_route_a_checkpoint(
                last_path, model, optimizer, scheduler, epoch, last_train_metrics, args, cfg,
                save_split=False,
            )

    elapsed = (time.time() - wall) / 60.0
    print("=" * 80)
    print(f"Done. elapsed={elapsed:.1f} min best_val_loss={best_loss:.6f}")
    print(f"best={best_path}")
    print(f"split encoder={os.path.join(args.save_dir, f'sc_encoder_{args.dataset}_c{args.embed_dim}.pth')}")
    print(f"split decoder={os.path.join(args.save_dir, f'sc_decoder_{args.dataset}_c{args.embed_dim}.pth')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
