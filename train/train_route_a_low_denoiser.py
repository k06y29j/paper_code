#!/usr/bin/env python
"""Train a deployable Route-A low-space denoiser before the fixed Refiner.

The previous diagnostic showed that an oracle lower-noise low-frequency anchor,

    z_rx_oracle = z_ch + scale * (z_rx - z_ch)
    z_low_oracle = ChannelDecoder(z_rx_oracle)

substantially improves the frozen deterministic Refiner.  This script trains a
small network that only sees the deployable input ``z_low = ChannelDecoder(z_rx)``
and learns to approximate ``z_low_oracle``.  Validation selects by full-link
PSNR after the frozen Refiner:

    z_low -> LowDenoiser -> Refiner -> SemanticDecoder.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import DIV2KDataset  # noqa: E402
from src.cddm_mimo_ddnm.modules.siso_channel import SISOChannel  # noqa: E402


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
load_latent_refiner = eval_helpers.load_latent_refiner
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


class LowDenoiseDataset(Dataset):
    def __init__(self, z_low: torch.Tensor, z_target: torch.Tensor, imgs: torch.Tensor) -> None:
        self.z_low = z_low
        self.z_target = z_target
        self.imgs = imgs

    def __len__(self) -> int:
        return int(self.z_low.shape[0])

    def __getitem__(self, idx):
        return self.z_low[idx], self.z_target[idx], self.imgs[idx]


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LowDenoiser(nn.Module):
    def __init__(self, channels: int = 16, hidden: int = 96, depth: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, hidden, 3, padding=1)]
        for _ in range(int(depth)):
            layers.append(ResBlock(hidden))
        layers.extend([nn.GELU(), nn.Conv2d(hidden, channels, 3, padding=1)])
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, z_low: torch.Tensor) -> torch.Tensor:
        return z_low + self.res_scale.to(dtype=z_low.dtype) * self.net(z_low)


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Route-A low denoiser session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train deployable low denoiser before Route-A null refiner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", action="store_false", dest="cache_decoded")
    p.add_argument("--cache_workers", type=int, default=16)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--precache_repeats", type=int, default=4)

    p.add_argument("--sc_encoder_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_encoder_div2k_c16.pth")
    p.add_argument("--sc_decoder_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_decoder_div2k_c16.pth")
    p.add_argument("--cc_dir", type=str, default="checkpoints-val-v2/route_a/cc_dct_c4")
    p.add_argument("--unet_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/unet_un/unet_un_div2k_c16_best.pth")
    p.add_argument("--refiner_ckpt", type=str,
                   default="checkpoints-val-v2/route_a/null_refiner_h128d8_img10_awgn12/route_a_null_refiner_div2k_c16_awgn12_best.pth")
    p.add_argument("--refiner_hidden", type=int, default=0)
    p.add_argument("--refiner_depth", type=int, default=0)
    p.add_argument("--compression_ratio", type=float, default=0.25)
    p.add_argument("--snr_db", type=float, default=12.0)
    p.add_argument("--fading", type=str, default="awgn", choices=["awgn"])
    p.add_argument("--target_noise_scale", type=float, default=0.5)

    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_low", type=float, default=1.0)
    p.add_argument("--lambda_refiner_img", type=float, default=2.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_file", type=str, default="log-v2/route_a/low_denoiser_refiner_awgn12.txt")
    p.add_argument("--save_dir", type=str, default="checkpoints-val-v2/route_a/low_denoiser_refiner_awgn12")
    return p.parse_args()


@torch.no_grad()
def make_low_pair(system, imgs: torch.Tensor, target_noise_scale: float, amp_enabled: bool, amp_dtype: torch.dtype):
    autocast_cm = (
        torch.autocast(imgs.device.type, enabled=amp_enabled, dtype=amp_dtype)
        if imgs.device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with autocast_cm:
        z_sem = system.semantic_encoder(imgs)
        z_ch = system.channel_encoder(z_sem)
    z_rx, _sigma_y, _beta = system.mimo.forward(z_ch.float())
    z_rx_target = z_ch.float() + float(target_noise_scale) * (z_rx.float() - z_ch.float())
    with autocast_cm:
        z_low = system.channel_decoder(z_rx.to(z_ch.dtype))
        z_target = system.channel_decoder(z_rx_target.to(z_ch.dtype))
    return z_low.float(), z_target.float()


@torch.no_grad()
def precache(system, loader, device, target_noise_scale, amp_enabled, amp_dtype, repeats, max_batches, split_name):
    lows, targets, imgs_all = [], [], []
    for rep in range(max(1, int(repeats))):
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
            z_low, z_target = make_low_pair(system, imgs, target_noise_scale, amp_enabled, amp_dtype)
            lows.append(z_low.cpu())
            targets.append(z_target.cpu())
            imgs_all.append(imgs.float().cpu())
        print(f"  cached {split_name} repeat {rep + 1}/{max(1, int(repeats))}")
    return LowDenoiseDataset(torch.cat(lows), torch.cat(targets), torch.cat(imgs_all))


def save_checkpoint(path, model, args, epoch, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "denoiser_type": "low_residual",
            "channels": 16,
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


@torch.no_grad()
def validate(system, refiner, model, loader, args, device, amp_enabled, amp_dtype):
    model.eval()
    base_vals, oracle_vals, den_vals, ref_raw_vals, ref_den_vals = [], [], [], [], []
    low_losses = AverageMeter()
    dec_dtype = next(system.semantic_decoder.parameters()).dtype
    ref_dtype = next(refiner.parameters()).dtype
    for bi, batch in enumerate(loader):
        if args.max_val_batches > 0 and bi >= args.max_val_batches:
            break
        z_low, z_target, imgs = batch
        z_low = z_low.to(device, non_blocking=True)
        z_target = z_target.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)
        z_den = model(z_low.float())
        low_loss = F.mse_loss(z_den.float(), z_target.float())
        autocast_cm = (
            torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with autocast_cm:
            x_low = system.semantic_decoder(z_low.to(dec_dtype))
            x_oracle = system.semantic_decoder(z_target.to(dec_dtype))
            x_den = system.semantic_decoder(z_den.to(dec_dtype))
            z_ref_raw = refiner(z_low.to(ref_dtype)).float()
            z_ref_den = refiner(z_den.to(ref_dtype)).float()
            x_ref_raw = system.semantic_decoder(z_ref_raw.to(dec_dtype))
            x_ref_den = system.semantic_decoder(z_ref_den.to(dec_dtype))
        base_vals.append(psnr_per_image(x_low.float().clamp(0, 1), imgs.float()).cpu())
        oracle_vals.append(psnr_per_image(x_oracle.float().clamp(0, 1), imgs.float()).cpu())
        den_vals.append(psnr_per_image(x_den.float().clamp(0, 1), imgs.float()).cpu())
        ref_raw_vals.append(psnr_per_image(x_ref_raw.float().clamp(0, 1), imgs.float()).cpu())
        ref_den_vals.append(psnr_per_image(x_ref_den.float().clamp(0, 1), imgs.float()).cpu())
        low_losses.update(float(low_loss.item()), int(imgs.shape[0]))
    def avg(items):
        return float(torch.cat(items).mean().item()) if items else float("nan")
    return {
        "val_low_loss": low_losses.avg,
        "val_psnr_low_raw": avg(base_vals),
        "val_psnr_low_oracle": avg(oracle_vals),
        "val_psnr_low_denoised": avg(den_vals),
        "val_psnr_refiner_raw": avg(ref_raw_vals),
        "val_psnr_refiner_denoised": avg(ref_den_vals),
        "val_gain_vs_refiner_raw": avg(ref_den_vals) - avg(ref_raw_vals),
    }


def main() -> None:
    args = parse_args()
    setup_log_file(args.log_file)
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

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
    train_img_loader = DataLoader(
        train_img_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if args.num_workers > 0 else None),
    )
    val_img_loader = DataLoader(
        val_img_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.val_num_workers > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if args.val_num_workers > 0 else None),
    )

    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}")
    print(f"train={len(train_img_ds)} valid={len(val_img_ds)} snr={args.snr_db:g}dB target_noise_scale={args.target_noise_scale:g}")
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
    system.mimo = SISOChannel(snr_db=float(args.snr_db), fading=args.fading)
    refiner = load_latent_refiner(args.refiner_ckpt, args.refiner_hidden, args.refiner_depth, device)
    refiner.eval()
    for p in system.parameters():
        p.requires_grad_(False)
    for p in refiner.parameters():
        p.requires_grad_(False)

    print("Pre-caching low denoiser inputs/targets ...")
    train_ds = precache(
        system, train_img_loader, device, float(args.target_noise_scale), amp_enabled, amp_dtype,
        repeats=int(args.precache_repeats), max_batches=int(args.max_train_batches), split_name="train",
    )
    val_ds = precache(
        system, val_img_loader, device, float(args.target_noise_scale), amp_enabled, amp_dtype,
        repeats=1, max_batches=int(args.max_val_batches), split_name="valid",
    )
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

    model = LowDenoiser(16, hidden=int(args.hidden), depth=int(args.depth)).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(PROJECT_ROOT, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "route_a_low_denoiser_refiner_div2k_c16_awgn12_best.pth")
    latest_path = os.path.join(save_dir, "route_a_low_denoiser_refiner_div2k_c16_awgn12_latest.pth")
    best_psnr = -1.0
    print(f"arch=LowDenoiser hidden={args.hidden} depth={args.depth} tag={tag}")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_meter = AverageMeter()
        low_meter = AverageMeter()
        img_meter = AverageMeter()
        for z_low, z_target, imgs in train_loader:
            z_low = z_low.to(device, non_blocking=True)
            z_target = z_target.to(device, non_blocking=True)
            imgs = imgs.to(device, non_blocking=True)
            autocast_cm = (
                torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            opt.zero_grad(set_to_none=True)
            with autocast_cm:
                z_den = model(z_low.float())
                loss_low = F.mse_loss(z_den.float(), z_target.float())
                z_ref = refiner(z_den.to(next(refiner.parameters()).dtype)).float()
                x_ref = system.semantic_decoder(z_ref.to(next(system.semantic_decoder.parameters()).dtype))
                loss_img = F.mse_loss(x_ref.float(), imgs.float())
                loss = float(args.lambda_low) * loss_low + float(args.lambda_refiner_img) * loss_img
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
            scaler.step(opt)
            scaler.update()
            n = int(imgs.shape[0])
            loss_meter.update(float(loss.item()), n)
            low_meter.update(float(loss_low.item()), n)
            img_meter.update(float(loss_img.item()), n)

        metrics = {
            "train_loss": loss_meter.avg,
            "train_loss_low": low_meter.avg,
            "train_loss_refiner_img": img_meter.avg,
        }
        if epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs):
            val_metrics = validate(system, refiner, model, val_loader, args, device, amp_enabled, amp_dtype)
            metrics.update(val_metrics)
            is_best = val_metrics["val_psnr_refiner_denoised"] > best_psnr
            if is_best:
                best_psnr = val_metrics["val_psnr_refiner_denoised"]
                save_checkpoint(best_path, model, args, epoch, metrics)
            save_checkpoint(latest_path, model, args, epoch, metrics)
            print(
                f"[epoch {epoch:03d}/{args.epochs}] "
                f"loss={loss_meter.avg:.6f} low={low_meter.avg:.6f} img={img_meter.avg:.6f} | "
                f"raw={val_metrics['val_psnr_low_raw']:.4f} oracle={val_metrics['val_psnr_low_oracle']:.4f} "
                f"den={val_metrics['val_psnr_low_denoised']:.4f} "
                f"ref_raw={val_metrics['val_psnr_refiner_raw']:.4f} "
                f"ref_den={val_metrics['val_psnr_refiner_denoised']:.4f} "
                f"gain={val_metrics['val_gain_vs_refiner_raw']:+.4f} "
                f"{'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] "
                f"loss={loss_meter.avg:.6f} low={low_meter.avg:.6f} img={img_meter.avg:.6f}"
            )
    print(f"best_psnr={best_psnr:.4f} ckpt={best_path}")


if __name__ == "__main__":
    main()
