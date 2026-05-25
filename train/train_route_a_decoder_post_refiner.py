#!/usr/bin/env python
"""Post-train the Route-A Swin decoder for deterministic refiner latents.

This stage keeps the physical low-frequency path fixed:

    z_low = A^T y_rx
    z_ref = Refiner(z_low) = z_low + P_null f(z_low)

Only ``SemanticDecoder`` is updated.  The loss improves the decoded image from
``z_ref`` while guarding the low-only branch ``Decoder(z_low)`` against
regression with both blur-target and old-decoder distillation terms.
"""

from __future__ import annotations

import argparse
import builtins
import copy
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


class CachedDecoderDataset(Dataset):
    def __init__(
        self,
        z_sem: torch.Tensor,
        z_low: torch.Tensor,
        z_ref: torch.Tensor,
        imgs: torch.Tensor,
        imgs_blur: torch.Tensor,
        x_low_old: torch.Tensor,
        x_ref_old: torch.Tensor,
    ) -> None:
        self.tensors = (z_sem, z_low, z_ref, imgs, imgs_blur, x_low_old, x_ref_old)

    def __len__(self) -> int:
        return int(self.tensors[0].shape[0])

    def __getitem__(self, idx: int):
        return tuple(t[idx] for t in self.tensors)


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Route-A decoder post-refiner session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-train Route-A SemanticDecoder for deterministic refiner output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", action="store_false", dest="cache_decoded")
    p.add_argument("--cache_workers", type=int, default=16)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--precache_latents", action="store_true", default=True)
    p.add_argument("--no_precache_latents", action="store_false", dest="precache_latents")
    p.add_argument("--precache_repeats", type=int, default=4,
                   help="Number of random-crop passes cached for training.")

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
    p.add_argument("--fading", type=str, default="awgn", choices=["awgn", "rayleigh"])

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_ref", type=float, default=1.0)
    p.add_argument("--lambda_highfreq", type=float, default=0.5)
    p.add_argument("--lambda_low_blur", type=float, default=1.0)
    p.add_argument("--lambda_low_distill", type=float, default=5.0)
    p.add_argument("--lambda_ref_distill", type=float, default=0.0)
    p.add_argument("--lambda_clean", type=float, default=0.1)
    p.add_argument("--blur_kernel_size", type=int, default=11)
    p.add_argument("--blur_sigma", type=float, default=2.0)
    p.add_argument("--low_guard_margin_db", type=float, default=0.02,
                   help="Guarded best permits at most this low-vs-blur PSNR drop.")
    p.add_argument("--min_ref_gain_db", type=float, default=0.0,
                   help="Guarded best also requires refiner-output PSNR gain over the old decoder.")
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_every_epochs", type=int, default=2)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume_decoder_ckpt", type=str, default="")
    p.add_argument("--log_file", type=str, default="log-v2/route_a/sc_decoder_post_refiner_awgn12.txt")
    p.add_argument("--save_dir", type=str, default="checkpoints-val-v2/route_a/sc_decoder_post_refiner_awgn12")
    return p.parse_args()


def gaussian_blur_tensor(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    k = int(kernel_size)
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) * 0.5
    kernel = torch.exp(-(coords * coords) / (2.0 * float(sigma) * float(sigma)))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    c = x.shape[1]
    pad = k // 2
    kernel_x = kernel.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    kernel_y = kernel.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    y = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    y = F.conv2d(y, kernel_x, groups=c)
    y = F.pad(y, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(y, kernel_y, groups=c)


def highpass(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    return x - gaussian_blur_tensor(x, kernel_size, sigma)


@torch.no_grad()
def make_latents(system, refiner: nn.Module, imgs: torch.Tensor, amp_enabled: bool, amp_dtype: torch.dtype):
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
    z_ref = refiner(z_low.float())
    return z_sem.float(), z_low.float(), z_ref.float()


@torch.no_grad()
def precache_dataset(
    *,
    system,
    old_decoder: nn.Module,
    refiner: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    repeats: int,
    max_batches: int,
    blur_kernel_size: int,
    blur_sigma: float,
    split_name: str,
) -> CachedDecoderDataset:
    z_sems, z_lows, z_refs = [], [], []
    imgs_all, imgs_blur_all, low_old_all, ref_old_all = [], [], [], []
    decoder_dtype = next(old_decoder.parameters()).dtype
    n_repeat = max(1, int(repeats))
    for rep in range(n_repeat):
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
            z_sem, z_low, z_ref = make_latents(system, refiner, imgs, amp_enabled, amp_dtype)
            imgs_blur = gaussian_blur_tensor(imgs.float(), blur_kernel_size, blur_sigma)
            autocast_cm = (
                torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with autocast_cm:
                x_low_old = old_decoder(z_low.to(decoder_dtype))
                x_ref_old = old_decoder(z_ref.to(decoder_dtype))
            z_sems.append(z_sem.cpu())
            z_lows.append(z_low.cpu())
            z_refs.append(z_ref.cpu())
            imgs_all.append(imgs.float().cpu())
            imgs_blur_all.append(imgs_blur.cpu())
            low_old_all.append(x_low_old.float().cpu())
            ref_old_all.append(x_ref_old.float().cpu())
        print(f"  cached {split_name} repeat {rep + 1}/{n_repeat}")
    return CachedDecoderDataset(
        torch.cat(z_sems),
        torch.cat(z_lows),
        torch.cat(z_refs),
        torch.cat(imgs_all),
        torch.cat(imgs_blur_all),
        torch.cat(low_old_all),
        torch.cat(ref_old_all),
    )


def unpack_batch(batch, device: torch.device):
    z_sem, z_low, z_ref, imgs, imgs_blur, x_low_old, x_ref_old = batch
    return (
        z_sem.to(device, non_blocking=True),
        z_low.to(device, non_blocking=True),
        z_ref.to(device, non_blocking=True),
        imgs.to(device, non_blocking=True),
        imgs_blur.to(device, non_blocking=True),
        x_low_old.to(device, non_blocking=True),
        x_ref_old.to(device, non_blocking=True),
    )


@torch.no_grad()
def build_batch_on_the_fly(
    *,
    system,
    old_decoder: nn.Module,
    refiner: nn.Module,
    imgs: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    blur_kernel_size: int,
    blur_sigma: float,
) -> tuple[torch.Tensor, ...]:
    z_sem, z_low, z_ref = make_latents(system, refiner, imgs, amp_enabled, amp_dtype)
    imgs_blur = gaussian_blur_tensor(imgs.float(), blur_kernel_size, blur_sigma)
    decoder_dtype = next(old_decoder.parameters()).dtype
    autocast_cm = (
        torch.autocast(imgs.device.type, enabled=amp_enabled, dtype=amp_dtype)
        if imgs.device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with autocast_cm:
        x_low_old = old_decoder(z_low.to(decoder_dtype))
        x_ref_old = old_decoder(z_ref.to(decoder_dtype))
    return z_sem, z_low, z_ref, imgs.float(), imgs_blur, x_low_old.float(), x_ref_old.float()


def save_decoder_checkpoint(
    path: str,
    decoder: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "source_sc_decoder_ckpt": args.sc_decoder_ckpt,
            "refiner_ckpt": args.refiner_ckpt,
            "args": vars(args),
        },
        path,
    )


def maybe_resume_decoder(path: str, decoder: nn.Module, optimizer, scaler, device) -> tuple[int, float, float]:
    if not path:
        return 1, -1.0, -1.0
    ckpt_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    ckpt = torch.load(ckpt_path, map_location=device)
    decoder.load_state_dict(ckpt["state_dict"], strict=False)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"]:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    metrics = ckpt.get("metrics", {}) or {}
    print(f"resumed decoder: {ckpt_path} epoch={ckpt.get('epoch', 0)} metrics={metrics}")
    best_psnr = float(metrics.get("val_psnr_ref_new", metrics.get("val_psnr_refined", -1.0)))
    return (
        int(ckpt.get("epoch", 0)) + 1,
        best_psnr,
        best_psnr,
    )


@torch.no_grad()
def validate(
    *,
    system,
    old_decoder: nn.Module,
    refiner: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
    precached: bool,
) -> dict:
    decoder = system.semantic_decoder
    decoder.eval()
    decoder_dtype = next(decoder.parameters()).dtype
    meters = {k: [] for k in (
        "ref_old", "ref_new", "low_old", "low_new",
        "low_blur_old", "low_blur_new", "clean_new",
    )}
    loss_ref = AverageMeter()
    loss_low_blur = AverageMeter()
    loss_low_distill = AverageMeter()
    for bi, batch in enumerate(loader):
        if args.max_val_batches > 0 and bi >= args.max_val_batches:
            break
        if precached:
            z_sem, z_low, z_ref, imgs, imgs_blur, x_low_old, x_ref_old = unpack_batch(batch, device)
        else:
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
            z_sem, z_low, z_ref, imgs, imgs_blur, x_low_old, x_ref_old = build_batch_on_the_fly(
                system=system,
                old_decoder=old_decoder,
                refiner=refiner,
                imgs=imgs,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                blur_kernel_size=args.blur_kernel_size,
                blur_sigma=args.blur_sigma,
            )
        autocast_cm = (
            torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with autocast_cm:
            x_ref = decoder(z_ref.to(decoder_dtype))
            x_low = decoder(z_low.to(decoder_dtype))
            x_clean = decoder(z_sem.to(decoder_dtype))
        x_ref_f = x_ref.float()
        x_low_f = x_low.float()
        x_clean_f = x_clean.float()
        imgs_f = imgs.float()
        imgs_blur_f = imgs_blur.float()
        x_low_old_f = x_low_old.float()
        x_ref_old_f = x_ref_old.float()
        n = imgs.shape[0]
        loss_ref.update(float(F.mse_loss(x_ref_f, imgs_f).item()), n)
        loss_low_blur.update(float(F.mse_loss(x_low_f, imgs_blur_f).item()), n)
        loss_low_distill.update(float(F.mse_loss(x_low_f, x_low_old_f).item()), n)
        meters["ref_old"].append(psnr_per_image(x_ref_old_f.clamp(0, 1), imgs_f))
        meters["ref_new"].append(psnr_per_image(x_ref_f.clamp(0, 1), imgs_f))
        meters["low_old"].append(psnr_per_image(x_low_old_f.clamp(0, 1), imgs_f))
        meters["low_new"].append(psnr_per_image(x_low_f.clamp(0, 1), imgs_f))
        meters["low_blur_old"].append(psnr_per_image(x_low_old_f.clamp(0, 1), imgs_blur_f))
        meters["low_blur_new"].append(psnr_per_image(x_low_f.clamp(0, 1), imgs_blur_f))
        meters["clean_new"].append(psnr_per_image(x_clean_f.clamp(0, 1), imgs_f))

    out = {
        "val_loss_ref": loss_ref.avg,
        "val_loss_low_blur": loss_low_blur.avg,
        "val_loss_low_distill": loss_low_distill.avg,
    }
    for name, vals in meters.items():
        out[f"val_psnr_{name}"] = torch.cat(vals).mean().item() if vals else float("nan")
    out["val_gain_ref"] = out["val_psnr_ref_new"] - out["val_psnr_ref_old"]
    out["val_drop_low"] = out["val_psnr_low_new"] - out["val_psnr_low_old"]
    out["val_drop_low_blur"] = out["val_psnr_low_blur_new"] - out["val_psnr_low_blur_old"]
    out["low_guard_ok"] = bool(out["val_drop_low_blur"] >= -float(args.low_guard_margin_db))
    return out


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
    print(
        f"train={len(train_ds)} valid={len(val_ds)} snr={args.snr_db:g}dB fading={args.fading} "
        f"cache_decoded={args.cache_decoded} precache_latents={args.precache_latents} repeats={args.precache_repeats}"
    )

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

    old_decoder = copy.deepcopy(system.semantic_decoder).to(device).eval()
    for p in old_decoder.parameters():
        p.requires_grad_(False)
    for p in system.parameters():
        p.requires_grad_(False)
    for p in system.semantic_decoder.parameters():
        p.requires_grad_(True)

    precached = False
    if args.precache_latents:
        print("Pre-caching train latents/refiner outputs/old decoder targets ...")
        train_cached = precache_dataset(
            system=system,
            old_decoder=old_decoder,
            refiner=refiner,
            loader=train_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            repeats=int(args.precache_repeats),
            max_batches=int(args.max_train_batches),
            blur_kernel_size=int(args.blur_kernel_size),
            blur_sigma=float(args.blur_sigma),
            split_name="train",
        )
        print("Pre-caching validation latents/refiner outputs/old decoder targets ...")
        val_cached = precache_dataset(
            system=system,
            old_decoder=old_decoder,
            refiner=refiner,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            repeats=1,
            max_batches=int(args.max_val_batches),
            blur_kernel_size=int(args.blur_kernel_size),
            blur_sigma=float(args.blur_sigma),
            split_name="valid",
        )
        train_loader = DataLoader(
            train_cached,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=max(2, int(args.prefetch_factor)),
        )
        val_loader = DataLoader(
            val_cached,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=max(2, int(args.prefetch_factor)),
        )
        precached = True
        print(f"cached train={len(train_cached)} valid={len(val_cached)}")

    decoder = system.semantic_decoder
    opt = optim.AdamW(decoder.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    start_epoch, best_guarded, best_any = maybe_resume_decoder(args.resume_decoder_ckpt, decoder, opt, scaler, device)

    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(PROJECT_ROOT, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "sc_decoder_div2k_c16_post_refiner_awgn12_best.pth")
    best_any_path = os.path.join(save_dir, "sc_decoder_div2k_c16_post_refiner_awgn12_best_any.pth")
    latest_path = os.path.join(save_dir, "sc_decoder_div2k_c16_post_refiner_awgn12_latest.pth")

    print(
        f"decoder_posttrain tag={tag} lr={args.lr:g} "
        f"path=y_rx->z_low->Refiner(z_low)->z_ref->Decoder(z_ref) "
        f"loss=(ref:{args.lambda_ref:g}, hf:{args.lambda_highfreq:g}, "
        f"low_blur:{args.lambda_low_blur:g}, low_distill:{args.lambda_low_distill:g}, "
        f"ref_distill:{args.lambda_ref_distill:g}, clean:{args.lambda_clean:g}) "
        f"low_guard_margin={args.low_guard_margin_db:g}dB start_epoch={start_epoch}"
    )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        decoder.train()
        meters = {k: AverageMeter() for k in ("loss", "ref", "hf", "low_blur", "low_distill", "ref_distill", "clean")}
        for bi, batch in enumerate(train_loader):
            if not precached and args.max_train_batches > 0 and bi >= args.max_train_batches:
                break
            if precached:
                z_sem, z_low, z_ref, imgs, imgs_blur, x_low_old, x_ref_old = unpack_batch(batch, device)
            else:
                imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
                imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
                z_sem, z_low, z_ref, imgs, imgs_blur, x_low_old, x_ref_old = build_batch_on_the_fly(
                    system=system,
                    old_decoder=old_decoder,
                    refiner=refiner,
                    imgs=imgs,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    blur_kernel_size=args.blur_kernel_size,
                    blur_sigma=args.blur_sigma,
                )

            decoder_dtype = next(decoder.parameters()).dtype
            autocast_cm = (
                torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            opt.zero_grad(set_to_none=True)
            with autocast_cm:
                x_ref = decoder(z_ref.to(decoder_dtype))
                x_low = decoder(z_low.to(decoder_dtype))
                loss_ref = F.mse_loss(x_ref.float(), imgs.float())
                loss_hf = F.mse_loss(
                    highpass(x_ref.float(), args.blur_kernel_size, args.blur_sigma),
                    highpass(imgs.float(), args.blur_kernel_size, args.blur_sigma),
                )
                loss_low_blur = F.mse_loss(x_low.float(), imgs_blur.float())
                loss_low_distill = F.mse_loss(x_low.float(), x_low_old.float())
                loss_ref_distill = F.mse_loss(x_ref.float(), x_ref_old.float())
                if float(args.lambda_clean) > 0:
                    x_clean = decoder(z_sem.to(decoder_dtype))
                    loss_clean = F.mse_loss(x_clean.float(), imgs.float())
                else:
                    loss_clean = x_ref.float().new_tensor(0.0)
                loss = (
                    float(args.lambda_ref) * loss_ref
                    + float(args.lambda_highfreq) * loss_hf
                    + float(args.lambda_low_blur) * loss_low_blur
                    + float(args.lambda_low_distill) * loss_low_distill
                    + float(args.lambda_ref_distill) * loss_ref_distill
                    + float(args.lambda_clean) * loss_clean
                )
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), float(args.clip_grad_norm))
            scaler.step(opt)
            scaler.update()

            n = imgs.shape[0]
            meters["loss"].update(float(loss.item()), n)
            meters["ref"].update(float(loss_ref.item()), n)
            meters["hf"].update(float(loss_hf.item()), n)
            meters["low_blur"].update(float(loss_low_blur.item()), n)
            meters["low_distill"].update(float(loss_low_distill.item()), n)
            meters["ref_distill"].update(float(loss_ref_distill.item()), n)
            meters["clean"].update(float(loss_clean.item()), n)

        metrics = {f"train_{k}": v.avg for k, v in meters.items()}
        if epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs):
            val_metrics = validate(
                system=system,
                old_decoder=old_decoder,
                refiner=refiner,
                loader=val_loader,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                args=args,
                precached=precached,
            )
            metrics.update(val_metrics)
            is_best_any = val_metrics["val_psnr_ref_new"] > best_any
            if is_best_any:
                best_any = val_metrics["val_psnr_ref_new"]
                save_decoder_checkpoint(best_any_path, decoder, opt, scaler, args, epoch, metrics)
            is_best_guarded = (
                bool(val_metrics["low_guard_ok"])
                and val_metrics["val_gain_ref"] >= float(args.min_ref_gain_db)
                and val_metrics["val_psnr_ref_new"] > best_guarded
            )
            if is_best_guarded:
                best_guarded = val_metrics["val_psnr_ref_new"]
                save_decoder_checkpoint(best_path, decoder, opt, scaler, args, epoch, metrics)
            save_decoder_checkpoint(latest_path, decoder, opt, scaler, args, epoch, metrics)
            print(
                f"[epoch {epoch:03d}/{args.epochs}] "
                f"loss={meters['loss'].avg:.6f} ref={meters['ref'].avg:.6f} hf={meters['hf'].avg:.6f} "
                f"low_blur={meters['low_blur'].avg:.6f} low_distill={meters['low_distill'].avg:.6f} | "
                f"ref_old={val_metrics['val_psnr_ref_old']:.4f} ref_new={val_metrics['val_psnr_ref_new']:.4f} "
                f"gain={val_metrics['val_gain_ref']:+.4f} "
                f"low_blur_drop={val_metrics['val_drop_low_blur']:+.4f} "
                f"guard={'OK' if val_metrics['low_guard_ok'] else 'FAIL'} "
                f"{'BEST' if is_best_guarded else ('BEST_ANY' if is_best_any else '')}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] "
                f"loss={meters['loss'].avg:.6f} ref={meters['ref'].avg:.6f} hf={meters['hf'].avg:.6f} "
                f"low_blur={meters['low_blur'].avg:.6f} low_distill={meters['low_distill'].avg:.6f}"
            )

    print(f"best_guarded={best_guarded:.4f} ckpt={best_path}")
    print(f"best_any={best_any:.4f} ckpt={best_any_path}")


if __name__ == "__main__":
    main()
