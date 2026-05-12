#!/usr/bin/env python
"""信道编解码器（Stage 2）三种权重设计方案 + 两种压缩率。

链路：image → SemanticEncoder(冻结) → ChannelEncoder → ChannelDecoder → SemanticDecoder(冻结) → image

权重设计 (--mode)：
  model1 / random_random : encoder、decoder 都做正交随机初始化，不训练（仅评估保存）
  model2 / random_pinv   : encoder 正交随机；decoder 设为 encoder 权重的 Moore–Penrose 伪逆
  model3 / trained       : 冻结语义编解码器，端到端训练 ChannelEncoder + ChannelDecoder

两种压缩率（基于 c16 语义瓶颈，空间 16×16 不变）：
  --out_channels 12  ->  16,16,16  →  12,16,16   (率 = 12/16 = 0.75)
  --out_channels 4   ->  16,16,16  →   4,16,16   (率 =  4/16 = 0.25)

默认输入：semantic encoder/decoder 取 checkpoints-val/sc/sc_{encoder,decoder}_div2k_c16.pth
默认输出：checkpoints-val/cc/model{1,2,3}/cc_div2k_c16to{12,4}.pth

用法示例：
    # 模式 1：随机 + 随机
    python train/train_cc.py --mode random_random --out_channels 12
    python train/train_cc.py --mode random_random --out_channels  4

    # 模式 2：随机 + 伪逆
    python train/train_cc.py --mode random_pinv --out_channels 12
    python train/train_cc.py --mode random_pinv --out_channels  4

    # 模式 3：端到端训练（单卡）
    CUDA_VISIBLE_DEVICES=0 python train/train_cc.py \
        --mode trained --out_channels 12 \
        --batch_size 16 --epochs 200 --lr 1e-3 \
        --log_file log/cc/trained_c16to12.txt
    CUDA_VISIBLE_DEVICES=0 python train/train_cc.py \
        --mode trained --out_channels 4 \
        --batch_size 16 --epochs 200 --lr 1e-3 \
        --log_file log/cc/trained_c16to4.txt
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import (
    SystemConfig,
    get_cifar10_config,
    get_div2k_config,
)
from src.cddm_mimo_ddnm.datasets import (
    get_cifar10_loaders,
    get_div2k_loaders,
)
from src.cddm_mimo_ddnm.modules.semantic_codec import (
    SemanticDecoder,
    SemanticEncoder,
)


# ===========================================================================
# 命令行参数
# ===========================================================================

MODE_CHOICES = ("random_random", "random_pinv", "trained")
MODE_TO_DIR = {
    "random_random": "model1",
    "random_pinv": "model2",
    "trained": "model3",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2 - 信道编解码器（3 种权重设计 + 2 种压缩率）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 核心设计 ----
    p.add_argument(
        "--mode",
        type=str,
        default="trained",
        choices=MODE_CHOICES,
        help="权重设计方案：random_random=都随机；random_pinv=随机+伪逆；trained=端到端训练",
    )
    p.add_argument(
        "--out_channels",
        type=int,
        default=12,
        choices=[4, 12],
        help="信道编码器输出通道数（即压缩后的 C）；语义瓶颈固定 16，故支持 16→12 或 16→4",
    )

    # ---- 数据集 ----
    p.add_argument("--dataset", type=str, default="div2k", choices=["cifar10", "div2k"])
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="数据集根目录。DIV2K 默认 /workspace/yongjia/datasets/DIV2K；CIFAR10 必填。",
    )
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--use_lmdb", action="store_true")
    p.add_argument("--train_lmdb_path", type=str, default=None)
    p.add_argument("--val_lmdb_path", type=str, default=None)
    p.add_argument("--cache_decoded", action="store_true", default=True)

    # ---- 已训练好的语义编/解码器（c16）----
    p.add_argument(
        "--sc_encoder_ckpt",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints-val/sc/sc_encoder_div2k_c16.pth"),
    )
    p.add_argument(
        "--sc_decoder_ckpt",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints-val/sc/sc_decoder_div2k_c16.pth"),
    )

    # ---- 训练超参（仅 mode=trained 生效）----
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument(
        "--lambda_feat",
        type=float,
        default=0.5,
        help="特征空间 L2 损失权重（约束 z_cd ≈ z_sem）",
    )
    p.add_argument(
        "--lambda_img",
        type=float,
        default=1.0,
        help="图像空间 SmoothL1 损失权重",
    )
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--eval_every_epochs", type=int, default=10)
    p.add_argument("--log_freq", type=int, default=50)

    # ---- 通用 ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_file", type=str, default=None, help="终端日志同时落盘的文件路径（可选）")
    p.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="保存目录；缺省按 --mode 选择 checkpoints-val/cc/model{1,2,3}/",
    )
    p.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="保存文件名；缺省 cc_<dataset>_c16to{out_channels}[_best].pth",
    )
    p.add_argument(
        "--eval_max_batches",
        type=int,
        default=0,
        help="评估时截断的 batch 数；<=0 表示遍历整个 val_loader",
    )

    args = p.parse_args()
    if args.dataset == "div2k" and args.data_dir is None:
        args.data_dir = "/workspace/yongjia/datasets/DIV2K"
    if args.save_dir is None:
        args.save_dir = os.path.join(PROJECT_ROOT, "checkpoints-val/cc", MODE_TO_DIR[args.mode])
    return args


# ===========================================================================
# 工具
# ===========================================================================

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
    builtins.print(f"\n=== Stage2 ChannelCodec session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
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


def build_config(dataset: str) -> SystemConfig:
    return get_div2k_config() if dataset == "div2k" else get_cifar10_config()


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


def load_state_dict_from_ckpt(model: nn.Module, ckpt_path: str, name: str) -> None:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{name} 权重不存在：{ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  载入 {name}: {ckpt_path}")
    if missing:
        print(f"    missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"    unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")


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


# ===========================================================================
# 信道编解码器（线性 1×1，bias=False）
# ===========================================================================

class LinearChannelCodec(nn.Module):
    """以 1×1 卷积实现的线性信道编/解码器对，便于做 SVD/伪逆等线性分析。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.decoder = nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, z_sem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_ch = self.encoder(z_sem)
        z_cd = self.decoder(z_ch)
        return z_ch, z_cd


def _orthogonal_init_(conv: nn.Conv2d) -> None:
    """对 1×1 Conv 的权重做正交初始化（对应矩阵 [out_C, in_C]）。

    对 in_C ≠ out_C 的情形：torch.nn.init.orthogonal_ 会生成「半正交」矩阵：
    短边方向上为单位列/行向量，长边正常正交，是常见的随机投影选择。
    """
    nn.init.orthogonal_(conv.weight)


def init_random_random(codec: LinearChannelCodec) -> None:
    """模式 1：encoder/decoder 各自独立做正交随机初始化。"""
    _orthogonal_init_(codec.encoder)
    _orthogonal_init_(codec.decoder)


@torch.no_grad()
def init_random_pinv(codec: LinearChannelCodec) -> None:
    """模式 2：encoder 正交随机；decoder 设为 encoder 权重矩阵的 Moore–Penrose 伪逆。

    设 encoder 权重 W ∈ R^{out_C × in_C}（squeeze 后），decoder 权重应为
    W^† ∈ R^{in_C × out_C}（pinv），1×1 卷积形状 [in_C, out_C, 1, 1]。
    """
    _orthogonal_init_(codec.encoder)
    w_enc = codec.encoder.weight.detach().squeeze(-1).squeeze(-1).to(torch.float64)
    w_dec = torch.linalg.pinv(w_enc)
    w_dec = w_dec.to(codec.decoder.weight.dtype).unsqueeze(-1).unsqueeze(-1)
    codec.decoder.weight.copy_(w_dec)


# ===========================================================================
# 端到端封装：image → SC enc(冻结) → CC → SC dec(冻结) → image
# ===========================================================================

class CCSystem(nn.Module):
    def __init__(self, sc_enc: SemanticEncoder, sc_dec: SemanticDecoder, codec: LinearChannelCodec):
        super().__init__()
        self.sc_enc = sc_enc
        self.sc_dec = sc_dec
        self.codec = codec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z_sem, _, _ = self.sc_enc.encode(x, sample=False)
        z_ch, z_cd = self.codec(z_sem)
        x_hat = self.sc_dec(z_cd)
        return x_hat, z_sem, z_cd


# ===========================================================================
# 评估：在 val_loader 上跑全链路 PSNR / 特征 MSE
# ===========================================================================

@torch.no_grad()
def evaluate(
    system: CCSystem,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 0,
) -> dict:
    system.eval()
    img_psnr = AverageMeter()
    img_mse = AverageMeter()
    feat_mse = AverageMeter()
    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)

    for i, batch in enumerate(val_loader):
        if max_batches > 0 and i >= max_batches:
            break
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", **autocast_kw):
            x_hat, z_sem, z_cd = system(images)
        x_hat_f = x_hat.float().clamp(0, 1)
        bs = images.shape[0]
        img_mse.update(F.mse_loss(x_hat_f, images.float()).item(), bs)
        img_psnr.update(compute_psnr(x_hat_f, images.float()), bs)
        feat_mse.update(F.mse_loss(z_cd.float(), z_sem.float()).item(), bs)

    return {
        "img_psnr": img_psnr.avg,
        "img_mse": img_mse.avg,
        "feat_mse": feat_mse.avg,
    }


# ===========================================================================
# 训练循环（仅 mode=trained 生效）
# ===========================================================================

def train_loop(
    system: CCSystem,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict:
    optimizer = optim.Adam(
        system.codec.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup = min(args.warmup_steps, max(1, total_steps - 1))
    min_ratio = max(0.0, args.min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(1e-6, step / max(1, warmup))
        t = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))
        return min_ratio + (1.0 - min_ratio) * cos

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)

    print(f"  trainable codec params: {sum(p.numel() for p in system.codec.parameters()):,}")
    print(f"  total steps: {total_steps}  warmup: {warmup}")

    best_psnr = -float("inf")
    best_metrics: dict = {}
    save_dir = args.save_dir
    save_name = args.save_name or f"cc_{args.dataset}_c16to{args.out_channels}_best.pth"
    best_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        system.train()
        system.sc_enc.eval()
        system.sc_dec.eval()

        loss_meter = AverageMeter()
        psnr_meter = AverageMeter()
        t_epoch = time.time()
        for i, batch in enumerate(train_loader):
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)

            with torch.autocast("cuda", **autocast_kw):
                x_hat, z_sem, z_cd = system(images)
                loss_img = F.smooth_l1_loss(x_hat, images, beta=0.1)
                loss_feat = F.mse_loss(z_cd, z_sem)
                loss = args.lambda_img * loss_img + args.lambda_feat * loss_feat

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(system.codec.parameters(), max_norm=args.clip_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            global_step += 1

            bs = images.shape[0]
            loss_meter.update(loss.item(), bs)
            psnr_meter.update(
                compute_psnr(x_hat.detach().float().clamp(0, 1), images.float()),
                bs,
            )
            if (i + 1) % args.log_freq == 0 or i + 1 == len(train_loader):
                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_epoch
                it_s = elapsed / (i + 1)
                eta = it_s * (len(train_loader) - i - 1)
                print(
                    f"  [{epoch+1}/{args.epochs}][{i+1}/{len(train_loader)}]  "
                    f"loss={loss_meter.avg:.4f}  PSNR={psnr_meter.avg:.2f}dB  "
                    f"LR={lr_now:.2e}  step={global_step}  "
                    f"{it_s:.2f}s/it  ETA={eta:.0f}s"
                )

        nepoch = epoch + 1
        if nepoch % max(1, args.eval_every_epochs) == 0 or nepoch == args.epochs:
            metrics = evaluate(system, val_loader, device, amp_enabled, amp_dtype, max_batches=args.eval_max_batches)
            print(
                f"  [eval@epoch {nepoch}]  "
                f"val_PSNR={metrics['img_psnr']:.4f}dB  "
                f"val_MSE={metrics['img_mse']:.6f}  "
                f"val_featMSE={metrics['feat_mse']:.6f}"
            )
            if metrics["img_psnr"] > best_psnr:
                best_psnr = metrics["img_psnr"]
                best_metrics = metrics
                save_codec(system.codec, best_path, args, metrics, epoch=nepoch)
                print(f"  *** save best -> {best_path}  PSNR={best_psnr:.4f}dB ***")

    return {"best_psnr": best_psnr, **best_metrics, "save_path": best_path}


# ===========================================================================
# 保存
# ===========================================================================

def save_codec(
    codec: LinearChannelCodec,
    path: str,
    args: argparse.Namespace,
    metrics: dict | None = None,
    epoch: int | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "state_dict": codec.state_dict(),
        "in_channels": codec.in_channels,
        "out_channels": codec.out_channels,
        "mode": args.mode,
        "dataset": args.dataset,
        "metrics": metrics,
        "epoch": epoch,
        "compression_ratio": codec.out_channels / codec.in_channels,
    }
    torch.save(state, path)


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    args = parse_args()
    setup_log_file(args.log_file)
    seed_everything(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print("=" * 80)
    print(f"  mode               : {args.mode}  (-> {MODE_TO_DIR[args.mode]})")
    print(f"  dataset            : {args.dataset}")
    print(f"  device             : {device}")
    print(f"  sc_encoder_ckpt    : {args.sc_encoder_ckpt}")
    print(f"  sc_decoder_ckpt    : {args.sc_decoder_ckpt}")
    print(f"  save_dir           : {args.save_dir}")
    print("=" * 80)

    cfg = build_config(args.dataset)
    sc = cfg.semantic
    if int(sc.embed_dim) != 16:
        print(
            f"  [WARN] cfg.semantic.embed_dim={sc.embed_dim} ≠ 16；"
            "当前脚本默认假设语义瓶颈 C=16（与 c16 ckpt 对齐）。"
        )

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

    load_state_dict_from_ckpt(sc_enc, args.sc_encoder_ckpt, "semantic_encoder")
    load_state_dict_from_ckpt(sc_dec, args.sc_decoder_ckpt, "semantic_decoder")

    for p in sc_enc.parameters():
        p.requires_grad = False
    for p in sc_dec.parameters():
        p.requires_grad = False
    sc_enc.eval()
    sc_dec.eval()

    in_c = int(sc.embed_dim)
    out_c = int(args.out_channels)
    if out_c >= in_c:
        raise ValueError(
            f"--out_channels={out_c} 必须严格小于语义瓶颈 in_channels={in_c}（这里固定为 16）。"
        )
    codec = LinearChannelCodec(in_channels=in_c, out_channels=out_c).to(device)
    print(f"  codec              : in={in_c}  out={out_c}  压缩率={out_c/in_c:.3f}")

    if args.mode == "random_random":
        init_random_random(codec)
        print("  [init] random_random : 两个 1x1 Conv 各自正交随机初始化")
    elif args.mode == "random_pinv":
        init_random_pinv(codec)
        w_e = codec.encoder.weight.detach().squeeze(-1).squeeze(-1).double()
        w_d = codec.decoder.weight.detach().squeeze(-1).squeeze(-1).double()
        prod = w_d @ w_e
        eye = torch.eye(in_c, device=prod.device, dtype=prod.dtype)
        residual = (prod - eye).norm().item()
        print(
            f"  [init] random_pinv : ||W_dec W_enc - I||_F = {residual:.4f} "
            f"(理论：当 out_c<in_c 时无法等于 I，秩 ≤ {out_c})"
        )
    else:
        init_random_random(codec)
        print("  [init] trained     : 先做正交随机初始化，再端到端训练")

    system = CCSystem(sc_enc, sc_dec, codec).to(device)

    train_loader, val_loader = build_dataloaders(args)
    print(f"  train batches/epoch: {len(train_loader)}")
    print(f"  val   batches/epoch: {len(val_loader)}")

    amp_enabled = args.amp_dtype != "none"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.amp_dtype, torch.bfloat16)

    if args.mode in ("random_random", "random_pinv"):
        save_name = args.save_name or f"cc_{args.dataset}_c16to{out_c}.pth"
        save_path = os.path.join(args.save_dir, save_name)

        metrics = evaluate(system, val_loader, device, amp_enabled, amp_dtype, max_batches=args.eval_max_batches)
        print(
            f"  [eval@init]  "
            f"val_PSNR={metrics['img_psnr']:.4f}dB  "
            f"val_MSE={metrics['img_mse']:.6f}  "
            f"val_featMSE={metrics['feat_mse']:.6f}"
        )
        save_codec(codec, save_path, args, metrics, epoch=None)
        print(f"  [save] {save_path}")
        print("=" * 80)
        return

    summary = train_loop(system, train_loader, val_loader, device, args, amp_enabled, amp_dtype)
    print("=" * 80)
    print(f"  Done.  best PSNR={summary['best_psnr']:.4f}dB  ckpt={summary.get('save_path')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
