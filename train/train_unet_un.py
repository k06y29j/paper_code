#!/usr/bin/env python
"""Stage 3：在冻结的语义编码器输出空间上训练**无条件**扩散模型 UNetUncond。

适配 `sc_encoder_div2k_c{C}.pth`（embed_dim=C，对应 latent shape [B, C, 16, 16]）。
默认配置面向 DIV2K c=16，即 [B, 16, 16, 16]，T=1000。

设计要点（针对 `pipeline.py` / `loss.py` 的不合理之处）：
  1) 强制 `semantic_encoder.eval()` + `sample=False`：用 mu 作为 z_0，避免 VAE 抽样
     与 dropout/norm 的训练态噪声污染扩散目标分布。
  2) Latent 标准化：训练前 dry-run 估计 latent 全局 std（参考 LDM 的 scaling factor），
     使 z_0 → z_0/std 后近似单位方差，匹配 DDPM 线性 β 调度的 SNR 假设。
  3) Loss：默认 ε-prediction MSE；可选 `--min_snr_gamma` 启用 min-SNR-γ 加权
     （Hang et al. 2023），平衡不同 t 的梯度贡献，加快收敛。
  4) EMA：扩散模型采样质量对 EMA 极敏感，默认 decay=0.9999。
  5) 评估指标（默认每 20 个 epoch 在验证集上评估；仅当 val_eps_mse 创新低时保存 best）：
       - val_eps_mse           跨 t 平均 ε MSE（直接训练目标）
       - val_eps_mse_t_q[0..3] t 切 4 段后的 ε MSE，定位“哪些 t 没学好”
       - z0_psnr@t=250/500/750 单步 z_0 估计的 PSNR（latent 空间）
       - decoded_psnr@t_start  从 t_start 反向 DDIM 到 0 后 decode 的图像 PSNR
                                （需要 `--sc_decoder_ckpt`，否则跳过）

用法（单卡）:
    CUDA_VISIBLE_DEVICES=4 python train/train_unet_un.py \
        --dataset div2k \
        --sc_encoder_ckpt checkpoints-val/sc/sc_encoder_div2k_c16.pth \
        --sc_decoder_ckpt checkpoints-val/sc/sc_decoder_div2k_c16.pth \
        --batch_size 32 --epochs 600 \
        --num_workers 8 --val_num_workers 4 --prefetch_factor 4 \
        --log_file log/unet_un/div2k_c16.txt \
        --save_dir checkpoints-val/unet_un
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import copy
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
    SemanticCommSystem,
    SystemConfig,
    get_cifar10_config,
    get_div2k_config,
)
from src.cddm_mimo_ddnm.datasets import (
    get_cifar10_loaders,
    get_div2k_loaders,
)
from src.cddm_mimo_ddnm.loss import min_snr_weighted_eps_loss


# ===========================================================================
# 命令行参数
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3 - 无条件 UNet 扩散先验训练（freeze SC encoder/decoder）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据
    p.add_argument("--dataset", type=str, default="div2k", choices=["cifar10", "div2k"])
    p.add_argument("--data_dir", type=str, default=None,
                   help="数据集根目录。DIV2K 默认 /workspace/yongjia/datasets/DIV2K；CIFAR10 必填。")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--use_lmdb", action="store_true")
    p.add_argument("--train_lmdb_path", type=str, default=None)
    p.add_argument("--val_lmdb_path", type=str, default=None)
    p.add_argument("--cache_decoded", action="store_true", default=True)

    # 已训练好的语义编/解码器
    p.add_argument("--sc_encoder_ckpt", type=str,
                   default=os.path.join(PROJECT_ROOT, "checkpoints-val/sc/sc_encoder_div2k_c16.pth"),
                   help="冻结的 semantic encoder 权重（state_dict 形式）。")
    p.add_argument("--sc_decoder_ckpt", type=str,
                   default=os.path.join(PROJECT_ROOT, "checkpoints-val/sc/sc_decoder_div2k_c16.pth"),
                   help="可选：semantic decoder 权重，用于评估 decoded_psnr。")

    # 训练超参
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--amp_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "none"])
    p.add_argument("--seed", type=int, default=1234)

    # 扩散损失
    p.add_argument("--min_snr_gamma", type=float, default=5.0,
                   help=">0 时启用 min-SNR-γ 加权噪声损失；<=0 退回普通 MSE。")
    p.add_argument("--latent_std", type=float, default=0.0,
                   help=">0 时直接使用该值作为 latent 标准化系数；=0 则在训练前 dry-run 估计。")
    p.add_argument("--latent_std_batches", type=int, default=20,
                   help="dry-run 估计 latent_std 用的 batch 数。")

    # EMA
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_warmup_steps", type=int, default=1000,
                   help="前 N 步用线性增长 decay，避免冷启动 EMA 偏向初始权重。")

    # 评估（默认按 epoch；数据管线中的 val_loader，即划分上的验证/测试集）
    p.add_argument("--eval_every_epochs", type=int, default=20,
                   help="每 N 个 epoch 结束时在 val_loader 上评估并尝试更新 best；最后一轮也会评估。")
    p.add_argument("--eval_every_steps", type=int, default=0,
                   help=">0 时每隔 N 个 optimizer step 也评估（额外开销）；0 表示仅用 eval_every_epochs。")
    p.add_argument("--eval_max_batches", type=int, default=20,
                   help="每次评估用 val_loader 的前 N 个 batch；DIV2K val=100 张可覆盖全集。")
    p.add_argument("--eval_recon_t_starts", type=str, default="500",
                   help="逗号分隔的反向 DDIM 起点 t（步索引），用于评估 decoded_psnr。")
    p.add_argument("--eval_recon_steps", type=int, default=50,
                   help="评估时反向 DDIM 步数。")
    p.add_argument("--eval_z0_t_list", type=str, default="250,500,750",
                   help="评估单步 z0 估计 PSNR 用的时间步索引。")

    # I/O
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--save_dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "checkpoints-val/unet_un"))
    p.add_argument("--resume", type=str, default=None)

    args = p.parse_args()
    if args.dataset == "div2k" and args.data_dir is None:
        args.data_dir = "/workspace/yongjia/datasets/DIV2K"
    return args


# ===========================================================================
# 工具
# ===========================================================================

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
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
    builtins.print(f"\n=== Stage3 UNetUncond session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
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


def load_state_dict_from_ckpt(model: nn.Module, ckpt_path: str, name: str) -> None:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{name} 权重不存在：{ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
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
# Latent 取样：强制确定性、无 dropout/norm 训练态干扰
# ===========================================================================

@torch.no_grad()
def encode_latent(system: SemanticCommSystem, x: torch.Tensor) -> torch.Tensor:
    """以 evaluation 模式调用 semantic encoder 并取确定性 latent（mu，若有 VAE）。

    返回 z_0：[B, C_emb, H', W']，与 UNetUncond.input_channel 对齐。
    """
    enc = system.semantic_encoder
    enc.eval()
    z, _, _ = enc.encode(x, sample=False)
    return z


@torch.no_grad()
def estimate_latent_std(system: SemanticCommSystem,
                        loader: DataLoader,
                        device: torch.device,
                        max_batches: int = 20) -> float:
    """对若干 batch 估计全局 latent std；用作 LDM 风格的 scaling factor。"""
    sums = 0.0
    sums_sq = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        z = encode_latent(system, images).float()
        sums += z.sum().item()
        sums_sq += (z * z).sum().item()
        n += z.numel()
    if n == 0:
        return 1.0
    mean = sums / n
    var = max(1e-12, sums_sq / n - mean * mean)
    return float(math.sqrt(var))


# ===========================================================================
# EMA
# ===========================================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.step_count = 0

    def _cur_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        ratio = min(1.0, (self.step_count + 1) / float(self.warmup_steps))
        return min(self.decay, 1.0 - (1.0 - self.decay) ** ratio)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self._cur_decay()
        msd = model.state_dict()
        for k, v in msd.items():
            if not torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
                continue
            self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
        self.step_count += 1

    def state_dict(self) -> dict:
        return {"decay": self.decay, "warmup_steps": self.warmup_steps,
                "step_count": self.step_count, "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd.get("decay", self.decay)
        self.warmup_steps = sd.get("warmup_steps", self.warmup_steps)
        self.step_count = sd.get("step_count", 0)
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}

    @contextlib.contextmanager
    def swap_in(self, model: nn.Module):
        """临时把 EMA 权重换进 model 用于评估/采样，结束后恢复原权重。"""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)


# ===========================================================================
# 评估
# ===========================================================================

@torch.no_grad()
def evaluate(
    system: SemanticCommSystem,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    latent_std: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    have_decoder: bool,
) -> dict:
    """在验证集上评估 ε MSE / 单步 z0 PSNR / 反向 DDIM decoded PSNR。"""
    system.unet_denoiser.eval()
    system.semantic_encoder.eval()
    if have_decoder:
        system.semantic_decoder.eval()

    alpha_bars = system.alpha_bars
    T = int(alpha_bars.shape[0])

    eps_meter_sum = 0.0
    eps_meter_n = 0
    bin_sums = [0.0] * 4
    bin_ns = [0] * 4

    z0_t_list = [int(t.strip()) for t in args.eval_z0_t_list.split(",") if t.strip()]
    z0_psnr_sum = {t: 0.0 for t in z0_t_list}
    z0_psnr_n = 0

    recon_t_starts = [int(t.strip()) for t in args.eval_recon_t_starts.split(",") if t.strip()]
    decoded_psnr_sum = {t: 0.0 for t in recon_t_starts}
    decoded_psnr_n = 0

    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)

    for i, batch in enumerate(val_loader):
        if i >= args.eval_max_batches:
            break
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        z0 = encode_latent(system, images) / latent_std  # 归一化

        bsz = z0.shape[0]
        # ---- 1) ε MSE（跨 t 平均 + 4 段分位）----
        t_idx = torch.randint(0, T, (bsz,), device=device, dtype=torch.long)
        eps = torch.randn_like(z0)
        ab = alpha_bars[t_idx].view(-1, 1, 1, 1).to(z0.dtype)
        z_t = ab.sqrt() * z0 + (1 - ab).sqrt() * eps
        with torch.autocast("cuda", **autocast_kw):
            eps_pred = system.unet_denoiser(z_t, t_idx)
        per_sample_mse = ((eps_pred.float() - eps.float()) ** 2).mean(dim=(1, 2, 3))
        eps_meter_sum += per_sample_mse.sum().item()
        eps_meter_n += bsz
        # bin by t
        bin_id = (t_idx.float() * 4 / T).clamp(max=3.999).long()
        for b in range(4):
            sel = (bin_id == b)
            if sel.any():
                bin_sums[b] += per_sample_mse[sel].sum().item()
                bin_ns[b] += int(sel.sum().item())

        # ---- 2) 固定 t 的单步 z0 PSNR（latent 空间）----
        for t_fix in z0_t_list:
            t_idx2 = torch.full((bsz,), t_fix, device=device, dtype=torch.long)
            eps2 = torch.randn_like(z0)
            ab2 = alpha_bars[t_idx2].view(-1, 1, 1, 1).to(z0.dtype)
            z_t2 = ab2.sqrt() * z0 + (1 - ab2).sqrt() * eps2
            with torch.autocast("cuda", **autocast_kw):
                eps_pred2 = system.unet_denoiser(z_t2, t_idx2)
            z0_hat = (z_t2 - (1 - ab2).sqrt() * eps_pred2) / ab2.sqrt().clamp(min=1e-8)
            mse = F.mse_loss(z0_hat.float(), z0.float()).item()
            psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12)) if mse > 0 else float("inf")
            z0_psnr_sum[t_fix] += psnr * bsz
        z0_psnr_n += bsz

        # ---- 3) 反向 DDIM 到 0 后 decode → decoded_psnr ----
        if have_decoder:
            for t_start in recon_t_starts:
                eps3 = torch.randn_like(z0)
                ab3 = alpha_bars[torch.full((bsz,), t_start, device=device, dtype=torch.long)].view(-1, 1, 1, 1).to(z0.dtype)
                z_start = ab3.sqrt() * z0 + (1 - ab3).sqrt() * eps3
                # 从 t_start 到 0 的 DDIM 反向
                step_indices = torch.linspace(t_start, 0, args.eval_recon_steps, device=device).long()
                z = z_start
                for j, idx in enumerate(step_indices):
                    t_emb = torch.full((bsz,), int(idx.item()), device=device, dtype=torch.long)
                    with torch.autocast("cuda", **autocast_kw):
                        eps_p = system.unet_denoiser(z, t_emb)
                    a = alpha_bars[idx].to(z.dtype)
                    if j + 1 < len(step_indices):
                        a_prev = alpha_bars[step_indices[j + 1]].to(z.dtype)
                    else:
                        a_prev = torch.tensor(1.0, device=device, dtype=z.dtype)
                    z0_pred = (z - (1 - a).sqrt() * eps_p) / a.sqrt().clamp(min=1e-8)
                    z = a_prev.sqrt() * z0_pred + (1 - a_prev).sqrt() * eps_p
                # 反归一化后解码
                z_decode = z * latent_std
                with torch.autocast("cuda", **autocast_kw):
                    x_hat = system.semantic_decoder(z_decode).float().clamp(0, 1)
                mse = F.mse_loss(x_hat, images.float()).item()
                psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12)) if mse > 0 else float("inf")
                decoded_psnr_sum[t_start] += psnr * bsz
            decoded_psnr_n += bsz

    out = {
        "val_eps_mse": eps_meter_sum / max(1, eps_meter_n),
    }
    for b in range(4):
        out[f"val_eps_mse_t_q{b}"] = bin_sums[b] / max(1, bin_ns[b])
    for t_fix in z0_t_list:
        out[f"z0_psnr@t={t_fix}"] = z0_psnr_sum[t_fix] / max(1, z0_psnr_n)
    if have_decoder:
        for t_start in recon_t_starts:
            out[f"decoded_psnr@t_start={t_start}"] = decoded_psnr_sum[t_start] / max(1, decoded_psnr_n)
    return out


# ===========================================================================
# 训练
# ===========================================================================

def main():
    args = parse_args()
    setup_log_file(args.log_file)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print(f"  dataset           : {args.dataset}")
    print(f"  batch_size        : {args.batch_size}  (grad_accum={args.grad_accum_steps})")
    print(f"  epochs            : {args.epochs}")
    print(f"  amp_dtype         : {args.amp_dtype}")
    print(f"  sc_encoder_ckpt   : {args.sc_encoder_ckpt}")
    print(f"  sc_decoder_ckpt   : {args.sc_decoder_ckpt}")
    print(f"  save_dir          : {args.save_dir}")
    print(f"  eval_every_epochs : {args.eval_every_epochs}  (eval_every_steps={args.eval_every_steps})")
    print("=" * 80)

    cfg = build_config(args.dataset)
    print(f"  semantic.embed_dim    = {cfg.semantic.embed_dim}")
    print(f"  unet_uncond.input_ch  = {cfg.unet_uncond.input_channel}")
    print(f"  unet_uncond.ch        = {cfg.unet_uncond.ch}")
    print(f"  unet_uncond.ch_mult   = {cfg.unet_uncond.ch_mult}")
    print(f"  unet_uncond.attn      = {cfg.unet_uncond.attn}")
    print(f"  diffusion.num_train_steps = {cfg.diffusion.num_train_steps}")
    if cfg.unet_uncond.input_channel != cfg.semantic.embed_dim:
        raise ValueError(
            f"unet_uncond.input_channel({cfg.unet_uncond.input_channel}) "
            f"≠ semantic.embed_dim({cfg.semantic.embed_dim})。请检查 config 与 sc_encoder ckpt 是否对齐。"
        )

    system = SemanticCommSystem(cfg).to(device)
    load_state_dict_from_ckpt(system.semantic_encoder, args.sc_encoder_ckpt, "semantic_encoder")
    have_decoder = bool(args.sc_decoder_ckpt) and os.path.isfile(args.sc_decoder_ckpt)
    if have_decoder:
        load_state_dict_from_ckpt(system.semantic_decoder, args.sc_decoder_ckpt, "semantic_decoder")
    else:
        print(f"  [WARN] 未提供 sc_decoder_ckpt，将跳过 decoded_psnr 评估。")

    # 冻结 SC encoder/decoder + 信道 + MIMO；仅训练 unet_denoiser
    for p in system.parameters():
        p.requires_grad = False
    for p in system.unet_denoiser.parameters():
        p.requires_grad = True

    train_loader, val_loader = build_dataloaders(args)
    print(f"  train batches/epoch   : {len(train_loader)}")
    print(f"  val   batches/epoch   : {len(val_loader)}")

    # ---- 估计 latent_std（LDM scaling factor）----
    if args.latent_std > 0:
        latent_std = float(args.latent_std)
        print(f"  latent_std (manual)   : {latent_std:.6f}")
    else:
        t0 = time.time()
        latent_std = estimate_latent_std(system, train_loader, device, max_batches=args.latent_std_batches)
        print(f"  latent_std (dry-run)  : {latent_std:.6f}  (used {args.latent_std_batches} batches, {time.time()-t0:.1f}s)")

    # ---- 优化器 / 调度器 ----
    optimizer = optim.AdamW(
        system.unet_denoiser.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )
    accum = max(1, int(args.grad_accum_steps))
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup_steps, max(1, total_steps - 1))
    min_ratio = max(0.0, args.min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(1e-6, step / max(1, warmup))
        t = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))
        return min_ratio + (1.0 - min_ratio) * cos

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---- AMP ----
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.amp_dtype, torch.bfloat16)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    # ---- EMA ----
    ema = EMA(system.unet_denoiser, decay=args.ema_decay, warmup_steps=args.ema_warmup_steps)

    # ---- 续训 ----
    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        system.unet_denoiser.load_state_dict(ckpt["unet_state_dict"])
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        latent_std = float(ckpt.get("latent_std", latent_std))
        best_metric = float(ckpt.get("best_val_eps_mse", float("inf")))
        print(f"  resumed from {args.resume}  epoch={start_epoch}  step={global_step}  "
              f"latent_std={latent_std:.6f}  best_val_eps_mse={best_metric:.6f}")

    os.makedirs(args.save_dir, exist_ok=True)
    base_tag = f"unet_un_{args.dataset}_c{cfg.semantic.embed_dim}"
    best_path = os.path.join(args.save_dir, f"{base_tag}_best.pth")

    # ---- 训练循环 ----
    alpha_bars = system.alpha_bars
    T = int(alpha_bars.shape[0])
    use_min_snr = args.min_snr_gamma > 0

    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)
    print(f"  total_steps={total_steps}  warmup={warmup}  use_min_snr_gamma={use_min_snr}")

    for epoch in range(start_epoch, args.epochs):
        system.unet_denoiser.train()
        system.semantic_encoder.eval()
        if have_decoder:
            system.semantic_decoder.eval()

        loss_sum = 0.0
        loss_n = 0
        t_epoch = time.time()

        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            z0 = encode_latent(system, images) / latent_std

            bsz = z0.shape[0]
            t_idx = torch.randint(0, T, (bsz,), device=device, dtype=torch.long)
            eps = torch.randn_like(z0)
            ab = alpha_bars[t_idx].view(-1, 1, 1, 1).to(z0.dtype)
            z_t = ab.sqrt() * z0 + (1 - ab).sqrt() * eps

            with torch.autocast("cuda", **autocast_kw):
                eps_pred = system.unet_denoiser(z_t, t_idx)
                if use_min_snr:
                    loss = min_snr_weighted_eps_loss(
                        eps_pred, eps, alpha_bars, t_idx, gamma=args.min_snr_gamma,
                    )
                else:
                    loss = F.mse_loss(eps_pred, eps)
            loss_for_back = loss / accum

            if scaler.is_enabled():
                scaler.scale(loss_for_back).backward()
            else:
                loss_for_back.backward()

            should_step = ((i + 1) % accum == 0) or (i + 1 == len(train_loader))
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        system.unet_denoiser.parameters(),
                        max_norm=args.clip_grad_norm,
                    )
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(system.unet_denoiser)
                global_step += 1

                # 中途评估
                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    with ema.swap_in(system.unet_denoiser):
                        m = evaluate(system, val_loader, device, args, latent_std,
                                     amp_enabled, amp_dtype, have_decoder)
                    print(f"  [eval@step {global_step}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
                    monitor = m["val_eps_mse"]
                    if monitor < best_metric:
                        best_metric = monitor
                        torch.save(
                            _pack_ckpt(
                                system, ema, optimizer, scheduler, args, cfg,
                                latent_std, epoch, global_step, m,
                                best_val_eps_mse=best_metric,
                            ),
                            best_path,
                        )
                        print(f"  [save best] {best_path}  val_eps_mse={monitor:.4f}")
                    system.unet_denoiser.train()

            loss_sum += loss.item() * bsz
            loss_n += bsz
            if (i + 1) % args.log_freq == 0 or i + 1 == len(train_loader):
                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_epoch
                it_s = elapsed / (i + 1)
                eta = it_s * (len(train_loader) - i - 1)
                print(
                    f"  [{epoch+1}/{args.epochs}][{i+1}/{len(train_loader)}]  "
                    f"loss={loss_sum/loss_n:.4f}  LR={lr_now:.2e}  step={global_step}  "
                    f"{it_s:.2f}s/it  ETA={eta:.0f}s"
                )

        # 每 eval_every_epochs 轮在验证集上评估；最后一轮必评；仅 val_eps_mse 更好时覆盖 best
        nepoch = epoch + 1
        if args.eval_every_epochs > 0 and (
            nepoch % args.eval_every_epochs == 0 or nepoch == args.epochs
        ):
            system.unet_denoiser.eval()
            with ema.swap_in(system.unet_denoiser):
                m = evaluate(
                    system, val_loader, device, args, latent_std,
                    amp_enabled, amp_dtype, have_decoder,
                )
            print(
                "  [eval@epoch %d] " % nepoch
                + "  ".join(f"{k}={v:.4f}" for k, v in m.items())
            )
            monitor = m["val_eps_mse"]
            if monitor < best_metric:
                best_metric = monitor
                torch.save(
                    _pack_ckpt(
                        system, ema, optimizer, scheduler, args, cfg,
                        latent_std, nepoch, global_step, m,
                        best_val_eps_mse=best_metric,
                    ),
                    best_path,
                )
                print(f"  [save best] {best_path}  val_eps_mse={monitor:.4f}")

    print("训练完成。")


def _pack_ckpt(system, ema, optimizer, scheduler, args, cfg,
               latent_std, epoch, global_step, metrics,
               best_val_eps_mse: float | None = None):
    out = {
        "unet_state_dict": system.unet_denoiser.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "latent_std": latent_std,
        "args": vars(args),
        "cfg_unet_uncond": vars(cfg.unet_uncond) if hasattr(cfg.unet_uncond, "__dict__") else None,
        "cfg_diffusion": vars(cfg.diffusion) if hasattr(cfg.diffusion, "__dict__") else None,
        "metrics": metrics,
    }
    if best_val_eps_mse is not None:
        out["best_val_eps_mse"] = float(best_val_eps_mse)
    return out


if __name__ == "__main__":
    main()
