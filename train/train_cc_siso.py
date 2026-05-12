#!/usr/bin/env python
"""语义编解码器训练：瓶颈经 SISO AWGN 信道（默认 SNR=12 dB，C=36）。

链路：image → SemanticEncoder → SISOChannel(AWGN) → SemanticDecoder → image

与 ``train/train_cc.py`` 风格一致：单卡、同类数据与日志参数；区别在于训练的是**语义编解码器**
（非冻结 SC + 线性信道编解码器），并在 ``embed_dim`` 空间上模拟物理 SISO。

默认假设 DIV2K 分层配置 + ``embed_dim=36``（须为偶数以配对复数符号）。

用法示例::

    CUDA_VISIBLE_DEVICES=2 python train/train_cc_siso.py \
        --dataset div2k --batch_size 16 --epochs 400 --lr 1e-4

权重默认目录：项目根下 ``checkpoints_snr/``；日志默认：``log_snr/train_cc_siso.txt``（可用 ``--save_dir`` / ``--log_file`` 覆盖）。
"""

from __future__ import annotations

import argparse
import builtins
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import SystemConfig, get_cifar10_config, get_div2k_config
from src.cddm_mimo_ddnm.datasets import get_cifar10_loaders, get_div2k_loaders
from src.cddm_mimo_ddnm.loss import semantic_codec_loss
from src.cddm_mimo_ddnm.modules.semantic_codec import SemanticDecoder, SemanticEncoder
from src.cddm_mimo_ddnm.modules.siso_channel import SISOChannel


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_log_file(log_path: str | None):
    if not log_path:
        return None
    abs_path = log_path if os.path.isabs(log_path) else os.path.join(PROJECT_ROOT, log_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== SC + SISO-AWGN session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse_val = torch.mean((pred - target) ** 2).item()
    if mse_val < 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse_val)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="语义编解码器 + SISO AWGN（默认 SNR=12dB，C=36）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default="div2k", choices=["cifar10", "div2k"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--use_lmdb", action="store_true")
    p.add_argument("--train_lmdb_path", type=str, default=None)
    p.add_argument("--val_lmdb_path", type=str, default=None)
    p.add_argument("--cache_decoded", action="store_true", default=True)

    p.add_argument("--embed_dim", type=int, default=36, help="语义瓶颈 C（须为偶数）")
    p.add_argument("--snr_db", type=float, default=12.0, help="SISO AWGN 信噪比（dB）")
    p.add_argument(
        "--fading",
        type=str,
        default="awgn",
        choices=["awgn", "rayleigh"],
        help="默认 awgn；可切 rayleigh 做对比",
    )

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--eval_every_epochs", type=int, default=50)
    p.add_argument("--log_freq", type=int, default=50)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--log_file",
        type=str,
        default=os.path.join("log_snr", "train_cc_siso.txt"),
        help="相对项目根；默认 log_snr/train_cc_siso.txt；传空字符串则不落盘",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="默认 checkpoints_snr/（相对项目根）",
    )
    p.add_argument("--save_name", type=str, default=None, help="最优权重名；缺省自动生成")

    args = p.parse_args()
    if args.dataset == "div2k" and args.data_dir is None:
        args.data_dir = "/workspace/yongjia/datasets/DIV2K"
    if args.save_dir is None:
        args.save_dir = os.path.join(PROJECT_ROOT, "checkpoints_snr/snr_12db")
    else:
        args.save_dir = (
            args.save_dir
            if os.path.isabs(args.save_dir)
            else os.path.join(PROJECT_ROOT, args.save_dir)
        )
    if not args.log_file:
        args.log_file = None
    if int(args.embed_dim) % 2 != 0:
        raise SystemExit("--embed_dim 须为偶数（SISO 复数符号配对）。")
    return args


def build_config(args: argparse.Namespace) -> SystemConfig:
    cfg = get_div2k_config() if args.dataset == "div2k" else get_cifar10_config()
    cfg.semantic.embed_dim = int(args.embed_dim)
    cfg.mimo.snr_db = float(args.snr_db)
    cfg.mimo.fading = str(args.fading)
    cfg.mimo.mode = "siso"
    return cfg


def build_dataloaders(args: argparse.Namespace):
    if args.dataset == "div2k":
        train_loader, val_loader, _ = get_div2k_loaders(
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
        )
    else:
        train_loader, val_loader, _ = get_cifar10_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=False,
            val_num_workers=args.val_num_workers,
            prefetch_factor=args.prefetch_factor,
        )
    return train_loader, val_loader


class SISOSemanticCodec(nn.Module):
    """image → SC enc → SISO → SC dec → image。"""

    def __init__(
        self,
        sc_enc: SemanticEncoder,
        sc_dec: SemanticDecoder,
        snr_db: float,
        fading: str,
    ) -> None:
        super().__init__()
        self.sc_enc = sc_enc
        self.sc_dec = sc_dec
        self.snr_db = float(snr_db)
        self.fading = fading
        self.channel = SISOChannel(snr_db=self.snr_db, fading=self.fading)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        z_sem, mu, logvar = self.sc_enc.encode(x, sample=self.training)
        z_rx, _, _ = self.channel.forward(z_sem)
        x_hat = self.sc_dec(z_rx)
        return x_hat, mu, logvar


@torch.no_grad()
def evaluate(
    model: SISOSemanticCodec,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    cfg: SystemConfig,
) -> dict:
    model.eval()
    lam_kl = cfg.semantic.lambda_kl
    use_vae = cfg.semantic.use_vae
    loss_m = AverageMeter()
    psnr_m = AverageMeter()
    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)
    for batch in val_loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", **autocast_kw):
            x_hat, mu, logvar = model(images)
            loss = semantic_codec_loss(
                x_hat=x_hat,
                x=images,
                mu=mu,
                logvar=logvar,
                lambda_kl=lam_kl,
                use_vae=use_vae,
            )
        bs = images.shape[0]
        loss_m.update(loss.item(), bs)
        psnr_m.update(compute_psnr(x_hat.float().clamp(0, 1), images.float()), bs)
    return {"loss": loss_m.avg, "psnr": psnr_m.avg}


def train_loop(
    model: SISOSemanticCodec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    cfg: SystemConfig,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict:
    params = list(model.sc_enc.parameters()) + list(model.sc_dec.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup = min(args.warmup_steps, max(1, total_steps - 1))
    min_ratio = max(0.0, args.min_lr_ratio)
    lam_kl = cfg.semantic.lambda_kl
    use_vae = cfg.semantic.use_vae

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(1e-6, step / max(1, warmup))
        t = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))
        return min_ratio + (1.0 - min_ratio) * cos

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)

    print(f"  trainable SC params: {sum(p.numel() for p in params):,}")
    print(f"  total steps: {total_steps}  warmup: {warmup}")

    best_psnr = -float("inf")
    best_metrics: dict = {}
    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"{args.dataset}_c{args.embed_dim}_snr{args.snr_db:g}_{args.fading}"
    if args.save_name:
        base = args.save_name[:-4] if args.save_name.endswith(".pth") else args.save_name
        best_enc = os.path.join(args.save_dir, f"{base}_encoder_best.pth")
        best_dec = os.path.join(args.save_dir, f"{base}_decoder_best.pth")
    else:
        best_enc = os.path.join(args.save_dir, f"sc_encoder_{tag}_best.pth")
        best_dec = os.path.join(args.save_dir, f"sc_decoder_{tag}_best.pth")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        loss_m = AverageMeter()
        psnr_m = AverageMeter()
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", **autocast_kw):
                x_hat, mu, logvar = model(images)
                loss = semantic_codec_loss(
                    x_hat=x_hat,
                    x=images,
                    mu=mu,
                    logvar=logvar,
                    lambda_kl=lam_kl,
                    use_vae=use_vae,
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=args.clip_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            global_step += 1
            bs = images.shape[0]
            loss_m.update(loss.item(), bs)
            psnr_m.update(compute_psnr(x_hat.detach().float().clamp(0, 1), images.float()), bs)
            if (i + 1) % args.log_freq == 0 or i + 1 == len(train_loader):
                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                it_s = elapsed / (i + 1)
                print(
                    f"  [{epoch+1}/{args.epochs}][{i+1}/{len(train_loader)}]  "
                    f"loss={loss_m.avg:.4f}  PSNR={psnr_m.avg:.2f}dB  "
                    f"LR={lr_now:.2e}  step={global_step}  {it_s:.2f}s/it"
                )

        nepoch = epoch + 1
        if nepoch % max(1, args.eval_every_epochs) == 0 or nepoch == args.epochs:
            metrics = evaluate(model, val_loader, device, amp_enabled, amp_dtype, cfg)
            print(
                f"  [eval@epoch {nepoch}]  val_loss={metrics['loss']:.4f}  "
                f"val_PSNR={metrics['psnr']:.4f}dB"
            )
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                best_metrics = metrics
                save_sc_checkpoint(model, best_enc, best_dec, args, metrics, nepoch)
                print(f"  *** save best -> {best_enc} , {best_dec}  PSNR={best_psnr:.4f}dB ***")

    return {"best_psnr": best_psnr, **best_metrics, "encoder_path": best_enc, "decoder_path": best_dec}


def save_sc_checkpoint(
    model: SISOSemanticCodec,
    enc_path: str,
    dec_path: str,
    args: argparse.Namespace,
    metrics: dict | None,
    epoch: int,
) -> None:
    os.makedirs(os.path.dirname(enc_path), exist_ok=True)
    meta = {
        "dataset": args.dataset,
        "embed_dim": args.embed_dim,
        "snr_db": args.snr_db,
        "fading": args.fading,
        "metrics": metrics,
        "epoch": epoch,
    }
    torch.save({"state_dict": model.sc_enc.state_dict(), **meta}, enc_path)
    torch.save({"state_dict": model.sc_dec.state_dict(), **meta}, dec_path)


def main():
    args = parse_args()
    setup_log_file(args.log_file)
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    cfg = build_config(args)
    sc = cfg.semantic

    print("=" * 80)
    print("  SC + SISO 语义编解码训练")
    print(f"  dataset      : {args.dataset}")
    print(f"  device       : {device}")
    print(f"  embed_dim C  : {args.embed_dim}")
    print(f"  SISO SNR     : {args.snr_db} dB")
    print(f"  fading       : {args.fading}")
    print(f"  use_vae      : {sc.use_vae}")
    print(f"  save_dir     : {args.save_dir}")
    print("=" * 80)

    sc_enc = SemanticEncoder(
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
        use_vae=sc.use_vae,
    ).to(device)
    sc_dec = SemanticDecoder(
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

    model = SISOSemanticCodec(sc_enc, sc_dec, snr_db=args.snr_db, fading=args.fading).to(device)

    train_loader, val_loader = build_dataloaders(args)
    print(f"  train batches/epoch: {len(train_loader)}")
    print(f"  val   batches/epoch: {len(val_loader)}")

    amp_enabled = args.amp_dtype != "none"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        args.amp_dtype, torch.bfloat16
    )

    summary = train_loop(model, train_loader, val_loader, device, args, cfg, amp_enabled, amp_dtype)
    print("=" * 80)
    print(
        f"  Done.  best val PSNR={summary['best_psnr']:.4f} dB  "
        f"enc={summary.get('encoder_path')}  "
        f"dec={summary.get('decoder_path')}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
