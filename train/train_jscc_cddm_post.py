#!/usr/bin/env python
"""Train original CDDM latent denoiser and post decoder from existing JSCC splits.

This keeps the CDDM training recipe from ``CDDM/main.py`` and
``CDDM/Diffusion/Train.py`` while loading the current repo's
``SemanticEncoder`` / ``SemanticDecoder`` split checkpoints.
"""

from __future__ import annotations

import argparse
import builtins
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CDDM.Autoencoder.data.datasets import get_loader as get_cddm_loader
from CDDM.Diffusion.Diffusion import ChannelDiffusionSampler, ChannelDiffusionTrainer
from CDDM.Diffusion.Model import UNet
from CDDM.Scheduler import GradualWarmupScheduler
from src.cddm_mimo_ddnm import get_div2k_config
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
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


@contextmanager
def suppress_single_tensor_print():
    old_print = builtins.print

    def filtered_print(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            return None
        return old_print(*args, **kwargs)

    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = old_print


def setup_log(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    f = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== JSCC->CDDM->redecoder @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {path}")
    return f, old_stdout, old_stderr


def seed_torch(seed: int = 1024):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--decoder_ckpt", type=str, required=True)
    p.add_argument("--snr_db", type=float, required=True)
    p.add_argument("--embed_dim", type=int, default=4)
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--channel_type", type=str, default="awgn", choices=["awgn", "rayleigh"])
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--val_num_workers", type=int, default=8)

    # Original CDDM CHDDIM_config defaults.
    p.add_argument("--cddm_epochs", type=int, default=400)
    p.add_argument("--cddm_batch", type=int, default=16)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--noise_schedule", type=int, default=1)
    p.add_argument("--t_max", type=float, default=10.0)
    p.add_argument("--large_snr", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--snr_max", type=float, default=1e-4)
    p.add_argument("--snr_min", type=float, default=0.02)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--multiplier", type=float, default=2.0)
    p.add_argument("--re_weight", action="store_true", default=True)
    p.add_argument("--no_re_weight", action="store_false", dest="re_weight")

    # Original DIV2K decoder post-training defaults.
    p.add_argument("--redecoder_epochs", type=int, default=20)
    p.add_argument("--redecoder_batch", type=int, default=4)
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--max_batches", type=int, default=0, help="Debug only; 0 means full epoch.")
    p.add_argument("--skip_cddm", action="store_true")
    p.add_argument("--skip_redecoder", action="store_true")
    return p.parse_args()


def cddm_data_config(args: argparse.Namespace, batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="DIV2K",
        channel_type=args.channel_type,
        SNRs=float(args.snr_db),
        C=int(args.embed_dim),
        CUDA=True,
        device=torch.device(args.device),
        image_dims=(3, 256, 256),
        train_data_dir=os.path.join(args.data_dir, "DIV2K_train_HR"),
        test_data_dir=os.path.join(args.data_dir, "DIV2K_valid_HR"),
        batch_size=int(batch_size),
        test_batch=1,
        num_workers=int(args.num_workers),
        val_num_workers=int(args.val_num_workers),
        pin_memory=True,
        persistent_workers=False,
    )


def build_sc(args: argparse.Namespace, device: torch.device):
    cfg = get_div2k_config()
    cfg.semantic.embed_dim = int(args.embed_dim)
    cfg.semantic.use_vae = False
    cfg.semantic.lambda_kl = 0.0
    sc = cfg.semantic
    encoder = SemanticEncoder(
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
        use_vae=False,
    ).to(device)
    decoder = SemanticDecoder(
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
    load_split(encoder, args.encoder_ckpt, "encoder")
    load_split(decoder, args.decoder_ckpt, "decoder")
    return encoder, decoder


def load_split(module: torch.nn.Module, path: str, name: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    missing, unexpected = module.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"{name} load mismatch: missing={missing}, unexpected={unexpected}")
    print(f"loaded {name}: {path}")


def make_cddm(args: argparse.Namespace, device: torch.device) -> UNet:
    return UNet(
        T=int(args.T),
        ch=int(16 * args.embed_dim),
        ch_mult=[1, 2, 2],
        attn=[1],
        num_res_blocks=2,
        dropout=0.1,
        input_channel=int(args.embed_dim),
    ).to(device)


def complex_normalize(x: torch.Tensor, power: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    pwr = torch.mean(x ** 2) * 2.0
    return math.sqrt(power) * x / torch.sqrt(pwr), pwr


def legacy_channel_forward(
    x: torch.Tensor,
    snr_db: float,
    channel_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tx, pwr = complex_normalize(x, power=1.0)
    height = tx.shape[2]
    x_complex = torch.complex(tx[:, :, : height // 2, :], tx[:, :, height // 2 :, :])
    sigma = math.sqrt(1.0 / (2.0 * 10.0 ** (float(snr_db) / 10.0)))
    noise = torch.complex(
        torch.randn_like(x_complex.real) * sigma,
        torch.randn_like(x_complex.imag) * sigma,
    )
    if channel_type == "awgn":
        return x_complex + noise, pwr, torch.ones_like(x_complex)
    h = torch.complex(
        torch.randn_like(x_complex.real),
        torch.randn_like(x_complex.imag),
    ) / math.sqrt(2.0)
    return x_complex * h + noise, pwr, h


def compute_batch_psnr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    mse = F.mse_loss(x_hat.float().clamp(0.0, 1.0) * 255.0, x.float() * 255.0)
    return 10.0 * math.log10(255.0 * 255.0 / max(float(mse.item()), 1e-12))


def train_cddm(args: argparse.Namespace, encoder: torch.nn.Module, device: torch.device, cddm_path: str):
    data_cfg = cddm_data_config(args, args.cddm_batch)
    train_loader, _ = get_cddm_loader(data_cfg)
    cddm = make_cddm(args, device)
    trainer = ChannelDiffusionTrainer(
        model=cddm,
        noise_schedule=int(args.noise_schedule),
        re_weight=bool(args.re_weight),
        beta_1=float(args.snr_max),
        beta_T=float(args.snr_min),
        T=int(args.T),
    ).to(device)
    optimizer = optim.AdamW(cddm.parameters(), lr=float(args.lr), weight_decay=1e-4)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=int(args.cddm_epochs),
        eta_min=0,
        last_epoch=-1,
    )
    scheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=float(args.multiplier),
        warm_epoch=0.1,
        after_scheduler=cosine,
    )
    encoder.eval()
    print(
        f"[CDDM] epochs={args.cddm_epochs} batch={args.cddm_batch} T={args.T} "
        f"schedule={args.noise_schedule} beta=({args.snr_max},{args.snr_min}) "
        f"lr={args.lr} re_weight={args.re_weight}"
    )
    for epoch in range(int(args.cddm_epochs)):
        cddm.train()
        meter = AverageMeter()
        t0 = time.time()
        for i, (images, _labels) in enumerate(train_loader):
            if args.max_batches and i >= args.max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            x0 = images.to(device, non_blocking=True)
            with torch.no_grad():
                feature = encoder(x0)
                y, _pwr = complex_normalize(feature, power=1.0)
                if args.channel_type == "awgn":
                    h = torch.ones_like(y)
                else:
                    half_h = y.shape[2] // 2
                    h_c = torch.complex(
                        torch.randn_like(y[:, :, :half_h, :]),
                        torch.randn_like(y[:, :, :half_h, :]),
                    ).abs() / math.sqrt(2.0)
                    h = torch.cat((h_c, h_c), dim=2)
            with suppress_single_tensor_print():
                loss = trainer(y, h, float(args.snr_db), channel_type=args.channel_type)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cddm.parameters(), float(args.grad_clip))
            optimizer.step()
            meter.update(float(loss.item()), int(x0.shape[0]))
            if (i + 1) % int(args.log_freq) == 0 or (i + 1) == len(train_loader):
                print(
                    f"[CDDM][{epoch+1:03d}/{args.cddm_epochs}][{i+1}/{len(train_loader)}] "
                    f"loss={meter.avg:.6f} lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"{(time.time()-t0)/max(1, i+1):.2f}s/it"
                )
        scheduler.step()
    torch.save(cddm.state_dict(), cddm_path)
    print(f"[CDDM] saved: {cddm_path}")


def train_redecoder(
    args: argparse.Namespace,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    cddm_path: str,
    redecoder_path: str,
):
    data_cfg = cddm_data_config(args, args.redecoder_batch)
    train_loader, _ = get_cddm_loader(data_cfg)
    cddm = make_cddm(args, device)
    cddm.load_state_dict(torch.load(cddm_path, map_location=device, weights_only=False))
    cddm.eval()
    sampler = ChannelDiffusionSampler(
        model=cddm,
        noise_schedule=int(args.noise_schedule),
        t_max=float(args.t_max),
        beta_1=float(args.snr_max),
        beta_T=float(args.snr_min),
        T=int(args.T),
    ).to(device)
    encoder.eval()
    decoder.train()
    optimizer = optim.AdamW(decoder.parameters(), lr=float(args.lr), weight_decay=1e-4)
    post_snr = float(args.snr_db) - float(args.large_snr)
    print(
        f"[redecoder] epochs={args.redecoder_epochs} batch={args.redecoder_batch} "
        f"post_snr={post_snr:g} train_snr={args.snr_db:g} large_snr={args.large_snr:g} "
        f"t_max={args.t_max:g}"
    )
    for epoch in range(int(args.redecoder_epochs)):
        loss_meter = AverageMeter()
        psnr_meter = AverageMeter()
        t0 = time.time()
        for i, (images, _labels) in enumerate(train_loader):
            if args.max_batches and i >= args.max_batches:
                break
            x0 = images.to(device, non_blocking=True)
            with torch.no_grad():
                feature = encoder(x0)
                y_complex, pwr, h = legacy_channel_forward(feature, post_snr, args.channel_type)
                sigma_square = 1.0 / (2.0 * 10.0 ** (post_snr / 10.0))
                y_complex = y_complex / math.sqrt(1.0 + sigma_square)
                feature_hat = sampler(
                    y_complex,
                    post_snr,
                    post_snr + float(args.large_snr),
                    h,
                    args.channel_type,
                )
                feature_hat = feature_hat * torch.sqrt(pwr)
            x_hat = decoder(feature_hat)
            loss = F.mse_loss(x_hat.float(), x0.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_meter.update(float(loss.item()), int(x0.shape[0]))
            psnr_meter.update(compute_batch_psnr(x_hat, x0), int(x0.shape[0]))
            if (i + 1) % int(args.log_freq) == 0 or (i + 1) == len(train_loader):
                print(
                    f"[redecoder][{epoch+1:03d}/{args.redecoder_epochs}][{i+1}/{len(train_loader)}] "
                    f"loss={loss_meter.avg:.6f} psnr={psnr_meter.avg:.4f} "
                    f"{(time.time()-t0)/max(1, i+1):.2f}s/it"
                )
    meta = {
        "state_dict": decoder.state_dict(),
        "dataset": "div2k",
        "embed_dim": int(args.embed_dim),
        "snr_db": float(args.snr_db),
        "post_snr_db": post_snr,
        "large_snr": float(args.large_snr),
        "source_decoder_ckpt": args.decoder_ckpt,
        "cddm_ckpt": cddm_path,
        "stage": "cddm_redecoder",
    }
    torch.save(meta, redecoder_path)
    print(f"[redecoder] saved: {redecoder_path}")


def main():
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    log_f, old_stdout, old_stderr = setup_log(os.path.join(args.run_dir, "cddm_post_train.log"))
    try:
        seed_torch(int(args.seed))
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        encoder, decoder = build_sc(args, device)
        cddm_path = os.path.join(args.run_dir, f"cddm_snr{args.snr_db:g}_awgn_c{args.embed_dim}.pt")
        redecoder_path = os.path.join(
            args.run_dir,
            f"redecoder_cddm_snr{args.snr_db - args.large_snr:g}_awgn_c{args.embed_dim}.pth",
        )
        print(f"run_dir={args.run_dir}")
        print(f"encoder_ckpt={args.encoder_ckpt}")
        print(f"decoder_ckpt={args.decoder_ckpt}")
        print(f"cddm_path={cddm_path}")
        print(f"redecoder_path={redecoder_path}")
        if not args.skip_cddm:
            train_cddm(args, encoder, device, cddm_path)
        if not args.skip_redecoder:
            train_redecoder(args, encoder, decoder, device, cddm_path, redecoder_path)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_f.close()


if __name__ == "__main__":
    main()
