#!/usr/bin/env python
"""信道感知（Channel-Aware）的信道编解码器（Stage 2）训练脚本。

与 ``train/train_cc.py`` 的关键差异：在 forward 中把 SISO 信道**插入**到 CC 编/解码器之间，
让 1×1 Conv 端到端学到信道感知的最优线性码本，类比 CDDM 的 JSCC 训练。

链路（训练时）：
  image → SemanticEncoder(冻结) → ChannelEncoder → **SISO 信道 (AWGN/Rayleigh, MMSE)**
       → ChannelDecoder → SemanticDecoder(冻结) → image

SNR 训练策略（``--train_snr_mode``）：
  fixed   : 每个 batch 用同一固定 SNR=train_snr_db（用于复现 CDDM 的「snr_X 专用模型」曲线）
  uniform : 每个样本独立从 [low, high] 均匀采样 SNR（一个鲁棒模型横扫全 SNR 范围，**默认**）

保存目录：``checkpoints-val/cc/aware_<fading>_<snr_tag>``，文件名同 train_cc.py：
  cc_encoder_div2k_c16to{4,12}.pth
  cc_decoder_div2k_c16to{4,12}.pth

用法示例（在 paper_code 根目录）::
  # 单卡 / AWGN / 随机 SNR ∈ [0, 15] dB / C=4
  CUDA_VISIBLE_DEVICES=0 python train/train_cc_aware.py \
      --mode trained --out_channels 4 --train_snr_mode uniform \
      --train_snr_db_low 0 --train_snr_db_high 15 --train_fading awgn \
      --batch_size 16 --epochs 200 --lr 1e-3 \
      --log_file log/cc/aware_awgn_u0_15_c4.txt

  # 固定 SNR=6（对标 CDDM 的 snr_6 曲线）
  CUDA_VISIBLE_DEVICES=0 python train/train_cc_aware.py \
      --mode trained --out_channels 4 --train_snr_mode fixed \
      --train_snr_db 6 --train_fading awgn \
      --batch_size 16 --epochs 150 --log_file log/cc/aware_awgn_snr6_c4.txt

训练完成后，用 ``test/eval_all.py --cc_dir checkpoints-val/cc/aware_<...>`` 评估。
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
from src.cddm_mimo_ddnm.modules.siso_channel import SISOChannel

# 直接复用 train_cc.py 的工具与 codec 定义
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cc import (  # noqa: E402
    AverageMeter,
    LinearChannelCodec,
    TeeStream,
    build_config,
    build_dataloaders,
    compute_psnr,
    init_random_random,
    load_state_dict_from_ckpt,
    save_codec,
    seed_everything,
    setup_log_file,
)


# ===========================================================================
# 可微信道 forward（逐样本随机 SNR）
# ===========================================================================

def _siso_awgn_batched(
    z: torch.Tensor,
    snr_db_per_sample: torch.Tensor,
    fading: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SISO 信道的批内逐样本随机 SNR 版本（与 ``SISOChannel.forward`` 等价数学，但 SNR 可变）。

    返回:
        z_rx:      [B, C, H', W']  反归一化后的实数特征
        beta_mean: [B]              MMSE 等效增益均值（AWGN 时恒为 1）
    """
    B, C, _Hp, _Wp = z.shape
    assert C % 2 == 0, "通道数 C 必须为偶数以配对复数符号"

    in_dtype = z.dtype
    z_f = z.float()
    z_complex = torch.complex(z_f[:, 0::2, :, :], z_f[:, 1::2, :, :])
    dims = tuple(range(1, z_complex.ndim))
    pwr = (z_complex.real ** 2 + z_complex.imag ** 2).mean(dim=dims)
    scale = torch.sqrt(pwr.clamp_min(1e-12))
    scale_b = scale.view(-1, *([1] * (z_complex.ndim - 1))).to(z_complex.dtype)
    x_norm = z_complex / scale_b

    snr_linear = (10.0 ** (snr_db_per_sample.float() / 10.0)).to(z.device)
    sigma2 = 1.0 / snr_linear
    sigma2_b = sigma2.view(-1, *([1] * (z_complex.ndim - 1)))
    sigma = torch.sqrt(sigma2_b / 2.0)

    n_r = torch.randn_like(x_norm.real) * sigma
    n_i = torch.randn_like(x_norm.imag) * sigma
    noise = torch.complex(n_r, n_i)

    if fading == "awgn":
        y = x_norm + noise
        x_hat = y
        beta_mean = torch.ones(B, device=z.device, dtype=in_dtype)
    else:
        h_r = torch.randn_like(x_norm.real) / math.sqrt(2.0)
        h_i = torch.randn_like(x_norm.imag) / math.sqrt(2.0)
        h = torch.complex(h_r, h_i)
        y = h * x_norm + noise
        h_abs2 = h.real ** 2 + h.imag ** 2
        x_hat = h.conj() * y / (h_abs2 + sigma2_b)
        beta_map = (h_abs2 / (h_abs2 + sigma2_b)).clamp(1e-3, 1 - 1e-3)
        beta_mean = beta_map.mean(dim=tuple(range(1, beta_map.ndim))).to(dtype=in_dtype)

    x_hat = x_hat * scale_b
    out = torch.empty_like(z_f)
    out[:, 0::2, :, :] = x_hat.real
    out[:, 1::2, :, :] = x_hat.imag
    return out.to(in_dtype), beta_mean


class CCSystemAware(nn.Module):
    """端到端（含信道）的 CC 训练系统。

    SNR 采样：
      - ``snr_mode='fixed'`` : 每个 batch 都用 ``snr_db`` (float)
      - ``snr_mode='uniform'``: 每个样本独立采样 SNR ∈ [snr_low, snr_high]
    """

    def __init__(
        self,
        sc_enc: SemanticEncoder,
        sc_dec: SemanticDecoder,
        codec: LinearChannelCodec,
        *,
        snr_mode: str,
        snr_db: float,
        snr_low: float,
        snr_high: float,
        fading: str,
    ) -> None:
        super().__init__()
        self.sc_enc = sc_enc
        self.sc_dec = sc_dec
        self.codec = codec
        if snr_mode not in ("fixed", "uniform"):
            raise ValueError(f"snr_mode 必须为 fixed/uniform，收到 {snr_mode!r}")
        if fading not in ("awgn", "rayleigh"):
            raise ValueError(f"fading 必须为 awgn/rayleigh，收到 {fading!r}")
        self.snr_mode = snr_mode
        self.snr_db = float(snr_db)
        self.snr_low = float(snr_low)
        self.snr_high = float(snr_high)
        self.fading = fading

    def _sample_snr_db(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.snr_mode == "fixed":
            return torch.full((batch_size,), self.snr_db, device=device, dtype=torch.float32)
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        return self.snr_low + (self.snr_high - self.snr_low) * u

    def forward(
        self, x: torch.Tensor, *, eval_snr_db: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (x_hat, z_sem, z_cd, snr_db_used [B])。

        若提供 ``eval_snr_db``，则评估时用该固定 SNR 替换采样（便于 val 可控）。
        """
        with torch.no_grad():
            z_sem, _, _ = self.sc_enc.encode(x, sample=False)
        z_ch = self.codec.encoder(z_sem)

        bsz = z_ch.shape[0]
        if eval_snr_db is None:
            snr_db = self._sample_snr_db(bsz, z_ch.device)
        else:
            snr_db = torch.full((bsz,), float(eval_snr_db), device=z_ch.device, dtype=torch.float32)

        z_rx, _ = _siso_awgn_batched(z_ch, snr_db, self.fading)
        z_cd = self.codec.decoder(z_rx)
        x_hat = self.sc_dec(z_cd)
        return x_hat, z_sem, z_cd, snr_db


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2 - 信道感知的 CC 训练（含信道 forward）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", type=str, default="trained", choices=["trained"],
                   help="（保留旧字段；本脚本只支持 trained 模式）")
    p.add_argument("--out_channels", type=int, default=12, choices=[4, 12])

    p.add_argument("--dataset", type=str, default="div2k", choices=["cifar10", "div2k"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--use_lmdb", action="store_true")
    p.add_argument("--train_lmdb_path", type=str, default=None)
    p.add_argument("--val_lmdb_path", type=str, default=None)
    p.add_argument("--cache_decoded", action="store_true", default=True)

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

    # 信道训练超参
    p.add_argument("--train_snr_mode", type=str, default="uniform", choices=["fixed", "uniform"],
                   help="fixed: 训练时固定 SNR=train_snr_db；uniform: 每样本随机 ∈[low, high]")
    p.add_argument("--train_snr_db", type=float, default=6.0,
                   help="snr_mode=fixed 时的训练 SNR (dB)")
    p.add_argument("--train_snr_db_low", type=float, default=0.0,
                   help="snr_mode=uniform 时的下界 (dB)")
    p.add_argument("--train_snr_db_high", type=float, default=15.0,
                   help="snr_mode=uniform 时的上界 (dB)")
    p.add_argument("--train_fading", type=str, default="awgn", choices=["awgn", "rayleigh"])

    # eval 期固定 SNR 网格
    p.add_argument(
        "--eval_snr_grid",
        type=float,
        nargs="+",
        default=[0.0, 3.0, 6.0, 9.0, 12.0, 15.0],
        help="评估时使用的 SNR 网格 (dB)，用于报告各 SNR 下的 PSNR",
    )

    # 训练超参
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--lambda_feat", type=float, default=0.5,
                   help="特征空间 MSE 损失权重")
    p.add_argument("--lambda_img", type=float, default=1.0,
                   help="图像空间 SmoothL1 损失权重")
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--eval_every_epochs", type=int, default=10)
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None,
                   help="保存目录；缺省 checkpoints-val/cc/aware_<fading>_<snr_tag>/")
    p.add_argument("--eval_max_batches", type=int, default=0)

    args = p.parse_args()
    if args.dataset == "div2k" and args.data_dir is None:
        args.data_dir = "/workspace/yongjia/datasets/DIV2K"

    if args.save_dir is None:
        if args.train_snr_mode == "fixed":
            snr_tag = f"snr{int(round(args.train_snr_db))}"
        else:
            snr_tag = f"u{int(round(args.train_snr_db_low))}_{int(round(args.train_snr_db_high))}"
        args.save_dir = os.path.join(
            PROJECT_ROOT, "checkpoints-val/cc",
            f"aware_{args.train_fading}_{snr_tag}",
        )
    return args


# ===========================================================================
# 评估：在每个 SNR 网格点上跑全链路 PSNR / feat MSE
# ===========================================================================

@torch.no_grad()
def evaluate_snr_grid(
    system: CCSystemAware,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    snr_grid: list[float],
    max_batches: int = 0,
) -> dict:
    """对每个 SNR 评估一遍，返回 {snr: {psnr, mse, feat_mse}, mean_psnr}.

    注意：每个 SNR 都遍历整个 val_loader，所以总开销 ≈ len(snr_grid) × 一遍。
    """
    system.eval()
    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)

    per_snr: dict[float, dict] = {}
    for snr in snr_grid:
        img_psnr = AverageMeter()
        img_mse = AverageMeter()
        feat_mse = AverageMeter()
        for i, batch in enumerate(val_loader):
            if max_batches > 0 and i >= max_batches:
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", **autocast_kw):
                x_hat, z_sem, z_cd, _ = system(images, eval_snr_db=float(snr))
            x_hat_f = x_hat.float().clamp(0, 1)
            bs = images.shape[0]
            img_mse.update(F.mse_loss(x_hat_f, images.float()).item(), bs)
            img_psnr.update(compute_psnr(x_hat_f, images.float()), bs)
            feat_mse.update(F.mse_loss(z_cd.float(), z_sem.float()).item(), bs)
        per_snr[float(snr)] = {
            "img_psnr": img_psnr.avg,
            "img_mse": img_mse.avg,
            "feat_mse": feat_mse.avg,
        }
    mean_psnr = sum(d["img_psnr"] for d in per_snr.values()) / max(1, len(per_snr))
    return {"per_snr": per_snr, "mean_psnr": mean_psnr}


# ===========================================================================
# 训练循环
# ===========================================================================

def train_loop(
    system: CCSystemAware,
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
    print(f"  channel: fading={system.fading}  mode={system.snr_mode}  "
          f"snr_db={system.snr_db if system.snr_mode == 'fixed' else (system.snr_low, system.snr_high)}")

    best_mean_psnr = -float("inf")
    best_metrics: dict = {}

    enc_save_name = f"cc_encoder_{args.dataset}_c16to{args.out_channels}.pth"
    dec_save_name = f"cc_decoder_{args.dataset}_c16to{args.out_channels}.pth"
    enc_path = os.path.join(args.save_dir, enc_save_name)
    dec_path = os.path.join(args.save_dir, dec_save_name)
    os.makedirs(args.save_dir, exist_ok=True)

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
                x_hat, z_sem, z_cd, snr_used = system(images)
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
                    f"snr_used~[{snr_used.min().item():.1f},{snr_used.max().item():.1f}]  "
                    f"LR={lr_now:.2e}  step={global_step}  "
                    f"{it_s:.2f}s/it  ETA={eta:.0f}s"
                )

        nepoch = epoch + 1
        if nepoch % max(1, args.eval_every_epochs) == 0 or nepoch == args.epochs:
            metrics = evaluate_snr_grid(
                system, val_loader, device, amp_enabled, amp_dtype,
                snr_grid=list(args.eval_snr_grid),
                max_batches=args.eval_max_batches,
            )
            line = "  | ".join(
                f"SNR={k:>4.1f}: {v['img_psnr']:.3f}dB"
                for k, v in metrics["per_snr"].items()
            )
            print(f"  [eval@epoch {nepoch}]  mean={metrics['mean_psnr']:.3f}dB  | {line}")

            if metrics["mean_psnr"] > best_mean_psnr:
                best_mean_psnr = metrics["mean_psnr"]
                best_metrics = metrics
                # 分别保存 encoder/decoder（兼容 eval_all.py 的加载格式）
                save_codec_pair(
                    system.codec, enc_path, dec_path,
                    args, metrics, epoch=nepoch,
                )
                print(f"  *** save best -> {args.save_dir}  mean_PSNR={best_mean_psnr:.4f}dB ***")

    return {"best_mean_psnr": best_mean_psnr, **best_metrics, "save_dir": args.save_dir}


# ===========================================================================
# 保存：encoder / decoder 分文件（兼容 eval_all.py 的加载格式 {'state_dict':{'weight':...}}）
# ===========================================================================

def save_codec_pair(
    codec: LinearChannelCodec,
    enc_path: str,
    dec_path: str,
    args: argparse.Namespace,
    metrics: dict | None = None,
    epoch: int | None = None,
) -> None:
    """分别保存 encoder/decoder 的 1x1 Conv 权重，键名 'weight'（与 model3 一致）。"""
    common = dict(
        in_channels=codec.in_channels,
        out_channels=codec.out_channels,
        mode="trained",
        dataset=args.dataset,
        compression_ratio=codec.out_channels / codec.in_channels,
        channel_aware=True,
        train_fading=args.train_fading,
        train_snr_mode=args.train_snr_mode,
        train_snr_db=args.train_snr_db,
        train_snr_db_low=args.train_snr_db_low,
        train_snr_db_high=args.train_snr_db_high,
        epoch=epoch,
        metrics=metrics,
    )
    os.makedirs(os.path.dirname(enc_path), exist_ok=True)

    enc_state = {"weight": codec.encoder.weight.detach().cpu()}
    dec_state = {"weight": codec.decoder.weight.detach().cpu()}
    torch.save({**common, "part": "encoder", "state_dict": enc_state}, enc_path)
    torch.save({**common, "part": "decoder", "state_dict": dec_state}, dec_path)


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> None:
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
    print(f"  mode               : channel-aware trained")
    print(f"  dataset            : {args.dataset}")
    print(f"  device             : {device}")
    print(f"  train_fading       : {args.train_fading}")
    print(f"  train_snr_mode     : {args.train_snr_mode}")
    if args.train_snr_mode == "fixed":
        print(f"  train_snr_db       : {args.train_snr_db}")
    else:
        print(f"  train_snr_db_range : [{args.train_snr_db_low}, {args.train_snr_db_high}]")
    print(f"  eval_snr_grid      : {args.eval_snr_grid}")
    print(f"  out_channels       : {args.out_channels}  (CBR={args.out_channels/16:.3f})")
    print(f"  save_dir           : {args.save_dir}")
    print("=" * 80)

    cfg = build_config(args.dataset)
    sc = cfg.semantic

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
    codec = LinearChannelCodec(in_channels=in_c, out_channels=out_c).to(device)
    init_random_random(codec)
    print(f"  codec init: orthogonal random, in={in_c}  out={out_c}  CBR={out_c/in_c:.3f}")

    system = CCSystemAware(
        sc_enc, sc_dec, codec,
        snr_mode=args.train_snr_mode,
        snr_db=args.train_snr_db,
        snr_low=args.train_snr_db_low,
        snr_high=args.train_snr_db_high,
        fading=args.train_fading,
    ).to(device)

    train_loader, val_loader = build_dataloaders(args)
    print(f"  train batches/epoch: {len(train_loader)}")
    print(f"  val   batches/epoch: {len(val_loader)}")

    amp_enabled = args.amp_dtype != "none"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.amp_dtype, torch.bfloat16)

    summary = train_loop(system, train_loader, val_loader, device, args, amp_enabled, amp_dtype)
    print("=" * 80)
    print(f"  Done.  best mean_PSNR={summary['best_mean_psnr']:.4f}dB")
    print(f"  save_dir={summary.get('save_dir')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
