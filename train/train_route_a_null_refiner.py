#!/usr/bin/env python
"""Train a light Route-A conditional null-space refiner.

The model is intentionally deterministic and small:

    z_low = A^T y  ->  z_hat = z_low + (I - A^T A) f(z_low)

It tests whether a direct conditional high-frequency predictor can improve the
fixed-SNR AWGN12 full-link PSNR before spending more compute on conditional
diffusion.
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
from torch.utils.data import DataLoader

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
LatentNullPredictor = eval_helpers.LatentNullPredictor
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


class CachedLatentDataset(torch.utils.data.Dataset):
    """Serves pre-cached (z_sem, z_low, img) tuples from CPU memory."""

    def __init__(self, z_sems: torch.Tensor, z_lows: torch.Tensor, imgs: torch.Tensor):
        self.z_sems = z_sems
        self.z_lows = z_lows
        self.imgs = imgs

    def __len__(self):
        return len(self.z_sems)

    def __getitem__(self, idx):
        return self.z_sems[idx], self.z_lows[idx], self.imgs[idx]


@torch.no_grad()
def precache_latents(system, loader, device, amp_enabled, amp_dtype):
    """Pre-compute all (z_sem, z_low, imgs) for a DataLoader to eliminate
    repeated frozen-system forward passes during training."""
    all_z_sem, all_z_low, all_imgs = [], [], []
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=True)
        z_sem, z_low = make_batch_latents(system, imgs, amp_enabled, amp_dtype)
        all_z_sem.append(z_sem.cpu())
        all_z_low.append(z_low.cpu())
        all_imgs.append(imgs.cpu())
    return torch.cat(all_z_sem), torch.cat(all_z_low), torch.cat(all_imgs)


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Route-A null refiner session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train deterministic Route-A null-space refiner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", action="store_false", dest="cache_decoded",
                   help="Disable image decoding cache")
    p.add_argument("--cache_workers", type=int, default=16)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--precache_latents", action="store_true",
                   help="Pre-compute all (z_sem, z_low) before training to maximize GPU util")

    p.add_argument("--sc_encoder_ckpt", type=str, required=True)
    p.add_argument("--sc_decoder_ckpt", type=str, required=True)
    p.add_argument("--cc_dir", type=str, required=True)
    p.add_argument("--unet_ckpt", type=str, required=True,
                   help="Only used to reuse eval_all system assembly and latent metadata.")
    p.add_argument("--compression_ratio", type=float, default=0.25)
    p.add_argument("--snr_db", type=float, default=12.0)
    p.add_argument("--fading", type=str, default="awgn", choices=["awgn", "rayleigh"])

    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_latent", type=float, default=1.0)
    p.add_argument("--lambda_image", type=float, default=1.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_file", type=str, default="log-v2/route_a/null_refiner_awgn12.txt")
    p.add_argument("--save_dir", type=str, default="checkpoints-val-v2/route_a/null_refiner_awgn12")
    return p.parse_args()


def null_project(z: torch.Tensor, a_matrix: torch.Tensor) -> torch.Tensor:
    a = a_matrix.to(device=z.device, dtype=z.dtype)
    az = torch.einsum("oc,bchw->bohw", a, z)
    low = torch.einsum("oc,bohw->bchw", a, az)
    return z - low


def save_checkpoint(
    path: str,
    model: nn.Module,
    args: argparse.Namespace,
    a_matrix: torch.Tensor,
    epoch: int,
    metrics: dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "refiner_type": "null_predictor",
            "refiner_hidden": int(args.hidden),
            "refiner_depth": int(args.depth),
            "a_matrix": a_matrix.detach().cpu(),
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


@torch.no_grad()
def make_batch_latents(system, imgs: torch.Tensor, amp_enabled: bool, amp_dtype: torch.dtype):
    autocast_cm = (
        torch.autocast(imgs.device.type, enabled=amp_enabled, dtype=amp_dtype)
        if imgs.device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with autocast_cm:
        z_sem = system.semantic_encoder(imgs)
        z_ch = system.channel_encoder(z_sem)
    z_rx, _sigma_y, _beta = system.mimo.forward(z_ch.float())
    with autocast_cm:
        z_low = system.channel_decoder(z_rx.to(z_ch.dtype))
    return z_sem.float(), z_low.float()


def validate(
    *,
    system,
    model: nn.Module,
    loader: DataLoader,
    a_matrix: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
    precached: bool = False,
) -> dict:
    model.eval()
    base_psnrs: list[torch.Tensor] = []
    ref_psnrs: list[torch.Tensor] = []
    losses = AverageMeter()
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        if precached:
            z_sem, z_low, imgs = batch
            z_sem = z_sem.to(device, non_blocking=True)
            z_low = z_low.to(device, non_blocking=True)
            imgs = imgs.to(device, non_blocking=True)
        else:
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
            z_sem, z_low = make_batch_latents(system, imgs, amp_enabled, amp_dtype)
        z_null_gt = null_project(z_sem, a_matrix)
        z_hat = model(z_low)
        z_null_hat = null_project(z_hat, a_matrix)
        loss = F.mse_loss(z_null_hat, z_null_gt)

        autocast_cm = (
            torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with autocast_cm:
            x_base = system.semantic_decoder(z_low.to(next(system.semantic_decoder.parameters()).dtype))
            x_hat = system.semantic_decoder(z_hat.to(next(system.semantic_decoder.parameters()).dtype))
        base_psnrs.append(psnr_per_image(x_base.float().clamp(0, 1), imgs.float()))
        ref_psnrs.append(psnr_per_image(x_hat.float().clamp(0, 1), imgs.float()))
        losses.update(float(loss.item()), imgs.shape[0])

    base = torch.cat(base_psnrs).mean().item() if base_psnrs else float("nan")
    ref = torch.cat(ref_psnrs).mean().item() if ref_psnrs else float("nan")
    return {
        "val_loss_null": losses.avg,
        "val_psnr_base": float(base),
        "val_psnr_refined": float(ref),
        "val_gain": float(ref - base),
    }


def main() -> None:
    args = parse_args()
    setup_log_file(args.log_file)
    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

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

    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}")
    print(f"train={len(train_ds)} valid={len(val_ds)} snr={args.snr_db:g}dB fading={args.fading}")

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
    for p in system.parameters():
        p.requires_grad_(False)

    precached = False
    if args.precache_latents:
        print("Pre-caching training latents ...")
        train_z_sem, train_z_low, train_imgs = precache_latents(
            system, train_loader, device, amp_enabled, amp_dtype
        )
        print(f"  cached {len(train_z_sem)} train samples")
        print("Pre-caching validation latents ...")
        val_z_sem, val_z_low, val_imgs = precache_latents(
            system, val_loader, device, amp_enabled, amp_dtype
        )
        print(f"  cached {len(val_z_sem)} val samples")
        train_ds = CachedLatentDataset(train_z_sem, train_z_low, train_imgs)
        val_ds = CachedLatentDataset(val_z_sem, val_z_low, val_imgs)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
            persistent_workers=True, prefetch_factor=4,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
            persistent_workers=True, prefetch_factor=4,
        )
        precached = True

    a_matrix = system._channel_encoder_matrix().detach().cpu().float()
    model = LatentNullPredictor(16, hidden=args.hidden, depth=args.depth, a_matrix=a_matrix).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(PROJECT_ROOT, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "route_a_null_refiner_div2k_c16_awgn12_best.pth")
    latest_path = os.path.join(save_dir, "route_a_null_refiner_div2k_c16_awgn12_latest.pth")
    best_psnr = -1.0

    print(
        f"arch=null_predictor hidden={args.hidden} depth={args.depth} tag={tag} "
        f"lambda_latent={args.lambda_latent:g} lambda_image={args.lambda_image:g}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        latent_meter = AverageMeter()
        image_meter = AverageMeter()
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and bi >= args.max_train_batches:
                break
            if precached:
                z_sem, z_low, imgs = batch
                z_sem = z_sem.to(device, non_blocking=True)
                z_low = z_low.to(device, non_blocking=True)
                imgs = imgs.to(device, non_blocking=True)
            else:
                imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
                imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
                z_sem, z_low = make_batch_latents(system, imgs, amp_enabled, amp_dtype)
            z_null_gt = null_project(z_sem, a_matrix)

            autocast_cm = (
                torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            opt.zero_grad(set_to_none=True)
            with autocast_cm:
                z_hat = model(z_low)
                z_null_hat = null_project(z_hat, a_matrix)
                loss_latent = F.mse_loss(z_null_hat.float(), z_null_gt.float())
                x_hat = system.semantic_decoder(z_hat.to(next(system.semantic_decoder.parameters()).dtype))
                loss_image = F.mse_loss(x_hat.float(), imgs.float())
                loss = float(args.lambda_latent) * loss_latent + float(args.lambda_image) * loss_image
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
            scaler.step(opt)
            scaler.update()

            n = imgs.shape[0]
            loss_meter.update(float(loss.item()), n)
            latent_meter.update(float(loss_latent.item()), n)
            image_meter.update(float(loss_image.item()), n)

        metrics = {
            "train_loss": loss_meter.avg,
            "train_loss_null": latent_meter.avg,
            "train_loss_img": image_meter.avg,
        }
        if epoch == 1 or epoch % max(1, args.eval_every_epochs) == 0 or epoch == args.epochs:
            val_metrics = validate(
                system=system,
                model=model,
                loader=val_loader,
                a_matrix=a_matrix,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                max_batches=args.max_val_batches,
                precached=precached,
            )
            metrics.update(val_metrics)
            is_best = val_metrics["val_psnr_refined"] > best_psnr
            if is_best:
                best_psnr = val_metrics["val_psnr_refined"]
                save_checkpoint(best_path, model, args, a_matrix, epoch, metrics)
            save_checkpoint(latest_path, model, args, a_matrix, epoch, metrics)
            print(
                f"[epoch {epoch:03d}/{args.epochs}] "
                f"loss={loss_meter.avg:.6f} null={latent_meter.avg:.6f} img={image_meter.avg:.6f} | "
                f"val_base={val_metrics['val_psnr_base']:.4f} "
                f"val_refined={val_metrics['val_psnr_refined']:.4f} "
                f"gain={val_metrics['val_gain']:+.4f} "
                f"{'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] "
                f"loss={loss_meter.avg:.6f} null={latent_meter.avg:.6f} img={image_meter.avg:.6f}"
            )

    print(f"best_psnr={best_psnr:.4f} ckpt={best_path}")


if __name__ == "__main__":
    main()
