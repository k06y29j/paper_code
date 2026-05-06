#!/usr/bin/env python
"""测量固定时间步上的噪声预测 MSE，以及 编码→加噪→DDIM 去噪→解码 的图像 PSNR。

时间步集合默认 T∈{0,100,500,999}（与训练相同的 DDPM 索引，∈[0, T-1]）。

用法示例:
    CUDA_VISIBLE_DEVICES=0 python test/eval_dm_mse.py \\
        --dataset div2k \\
        --sc_encoder_ckpt checkpoints-val/sc/sc_encoder_div2k_c16.pth \\
        --sc_decoder_ckpt checkpoints-val/sc/sc_decoder_div2k_c16.pth \\
        --unet_ckpt checkpoints-val/unet_un/unet_un_div2k_c16_best.pth \\
        --batch_size 8 --max_batches 20 --recon_steps 50
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 复用 train_unet_un 中的编码与权重加载（避免维护重复实现）
_TUN_PATH = os.path.join(PROJECT_ROOT, "train", "train_unet_un.py")
_spec = importlib.util.spec_from_file_location("train_unet_un_evaldeps", _TUN_PATH)
_tun = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tun)

encode_latent = _tun.encode_latent
estimate_latent_std = _tun.estimate_latent_std
load_state_dict_from_ckpt = _tun.load_state_dict_from_ckpt

from src.cddm_mimo_ddnm import SemanticCommSystem, SystemConfig, get_div2k_config, get_cifar10_config
from src.cddm_mimo_ddnm.datasets import get_div2k_loaders, get_cifar10_loaders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="扩散模型：固定 t 的 ε MSE + 编解码 PSNR")
    p.add_argument("--dataset", type=str, default="div2k", choices=["cifar10", "div2k"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--use_lmdb", action="store_true")
    p.add_argument("--train_lmdb_path", type=str, default=None)
    p.add_argument("--val_lmdb_path", type=str, default=None)
    p.add_argument("--cache_decoded", action="store_true", default=True)

    p.add_argument("--sc_encoder_ckpt", type=str, required=True)
    p.add_argument("--sc_decoder_ckpt", type=str, required=True)
    p.add_argument("--unet_ckpt", type=str, required=True, help="train_unet_un 保存的 best/last，含 unet 与可选 ema_state_dict")

    p.add_argument("--fixed_ts", type=str, default="0,100,500,999",
                   help="逗号分隔的扩散时间步索引（与训练 T=1000 一致）")
    p.add_argument("--latent_std", type=float, default=0.0, help=">0 强制使用；0 则优先从 unet_ckpt 读取，否则 dry-run 估计")
    p.add_argument("--latent_std_batches", type=int, default=20)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=20, help="验证集上前多少个 batch")
    p.add_argument("--recon_steps", type=int, default=50, help="DDIM 从 t 反推到 0 的步数（t=0 时不走循环）")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", action="store_true", help="仅用 unet_state_dict，忽略 EMA")
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    if args.no_ema:
        args.use_ema = False
    if args.dataset == "div2k" and args.data_dir is None:
        args.data_dir = "/workspace/yongjia/datasets/DIV2K"
    return args


def build_config(dataset: str) -> SystemConfig:
    return get_div2k_config() if dataset == "div2k" else get_cifar10_config()


def build_val_loader(args: argparse.Namespace):
    if args.dataset == "div2k":
        _, val_loader, _ = get_div2k_loaders(
            data_dir=args.data_dir or "",
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            num_workers=args.num_workers,
            distributed=False,
            use_lmdb=args.use_lmdb,
            train_lmdb_path=args.train_lmdb_path,
            val_lmdb_path=args.val_lmdb_path,
            val_num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            cache_decoded=bool(args.cache_decoded),
        )
    else:
        _, val_loader, _ = get_cifar10_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=False,
            val_num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
    return val_loader


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def ddim_latent_to_zero(
    unet,
    z_start: torch.Tensor,
    t_start: int,
    alpha_bars: torch.Tensor,
    bsz: int,
    device: torch.device,
    recon_steps: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """与 train_unet_un.evaluate 中相同的 DDIM 形式，从 z @ t_start 推到近似 z0（归一化潜空间）。"""
    if t_start <= 0:
        return z_start
    step_indices = torch.linspace(t_start, 0, recon_steps, device=device).long()
    z = z_start
    for j, idx in enumerate(step_indices):
        t_emb = torch.full((bsz,), int(idx.item()), device=device, dtype=torch.long)
        with torch.autocast(device.type, enabled=use_amp and device.type == "cuda", dtype=amp_dtype):
            eps_p = unet(z, t_emb)
        a = alpha_bars[idx].to(z.dtype)
        if j + 1 < len(step_indices):
            a_prev = alpha_bars[step_indices[j + 1]].to(z.dtype)
        else:
            a_prev = torch.tensor(1.0, device=device, dtype=z.dtype)
        z0_pred = (z - (1 - a).sqrt() * eps_p) / a.sqrt().clamp(min=1e-8)
        z = a_prev.sqrt() * z0_pred + (1 - a_prev).sqrt() * eps_p
    return z


@torch.no_grad()
def main_eval():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_ts = [int(x.strip()) for x in args.fixed_ts.split(",") if x.strip()]

    cfg = build_config(args.dataset)
    if cfg.unet_uncond.input_channel != cfg.semantic.embed_dim:
        raise ValueError("unet input_channel 与 semantic.embed_dim 不一致，请检查 config。")

    system = SemanticCommSystem(cfg).to(device)
    load_state_dict_from_ckpt(system.semantic_encoder, args.sc_encoder_ckpt, "semantic_encoder")
    load_state_dict_from_ckpt(system.semantic_decoder, args.sc_decoder_ckpt, "semantic_decoder")

    ckpt = torch.load(args.unet_ckpt, map_location="cpu")
    if "unet_state_dict" not in ckpt:
        raise KeyError(f"{args.unet_ckpt} 缺少 unet_state_dict")
    system.unet_denoiser.load_state_dict(ckpt["unet_state_dict"], strict=True)
    backup_weights = {k: v.detach().clone() for k, v in system.unet_denoiser.state_dict().items()}
    used_ema = bool(
        args.use_ema and "ema_state_dict" in ckpt and "shadow" in ckpt["ema_state_dict"]
    )
    if used_ema:
        system.unet_denoiser.load_state_dict(ckpt["ema_state_dict"]["shadow"], strict=True)
        print("  UNet 权重: EMA shadow")
    else:
        print("  UNet 权重: unet_state_dict（未使用 EMA）")

    system.eval()
    alpha_bars = system.alpha_bars.to(device)
    T = int(alpha_bars.shape[0])
    for t in fixed_ts:
        if t < 0 or t >= T:
            raise ValueError(f"时间步 {t} 超出 [0, {T - 1}]")

    if args.latent_std > 0:
        latent_std = float(args.latent_std)
        print(f"  latent_std (manual) = {latent_std:.6f}")
    else:
        latent_std = float(ckpt.get("latent_std", 0.0))
        if latent_std <= 0:
            train_loader, _, _ = (
                get_div2k_loaders(
                    data_dir=args.data_dir or "",
                    batch_size=args.batch_size,
                    crop_size=args.crop_size,
                    num_workers=args.num_workers,
                    distributed=False,
                    use_lmdb=args.use_lmdb,
                    train_lmdb_path=args.train_lmdb_path,
                    val_lmdb_path=args.val_lmdb_path,
                    val_num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                    cache_decoded=bool(args.cache_decoded),
                )
                if args.dataset == "div2k"
                else get_cifar10_loaders(
                    data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    distributed=False,
                    val_num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )
            )
            loader_for_std = train_loader
            latent_std = estimate_latent_std(system, loader_for_std, device, max_batches=args.latent_std_batches)
            print(f"  latent_std (dry-run) = {latent_std:.6f}")
        else:
            print(f"  latent_std (from ckpt) = {latent_std:.6f}")

    val_loader = build_val_loader(args)
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.amp_dtype]

    # ε MSE 按 t 累计
    eps_mse_sum = {t: 0.0 for t in fixed_ts}
    eps_mse_n = {t: 0 for t in fixed_ts}
    # 图像 PSNR 按 t 累计（编码→加噪→DDIM→解码）
    psnr_sum = {t: 0.0 for t in fixed_ts}
    psnr_n = 0

    for bi, batch in enumerate(val_loader):
        if bi >= args.max_batches:
            break
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        z0_raw = encode_latent(system, images)
        z0 = z0_raw / latent_std
        bsz = z0.shape[0]

        for t in fixed_ts:
            t_idx = torch.full((bsz,), t, device=device, dtype=torch.long)
            eps = torch.randn_like(z0)
            ab = alpha_bars[t_idx].view(-1, 1, 1, 1).to(z0.dtype)
            z_t = ab.sqrt() * z0 + (1 - ab).sqrt() * eps
            with torch.autocast(device.type, enabled=amp_enabled and device.type == "cuda", dtype=amp_dtype):
                eps_pred = system.unet_denoiser(z_t, t_idx)
            per_sample_mse = ((eps_pred.float() - eps.float()) ** 2).mean(dim=(1, 2, 3))
            eps_mse_sum[t] += per_sample_mse.sum().item()
            eps_mse_n[t] += bsz

            z_clean = ddim_latent_to_zero(
                system.unet_denoiser,
                z_t,
                t,
                alpha_bars,
                bsz,
                device,
                args.recon_steps,
                amp_enabled,
                amp_dtype,
            )
            z_decode = z_clean * latent_std
            with torch.autocast(device.type, enabled=amp_enabled and device.type == "cuda", dtype=amp_dtype):
                x_hat = system.semantic_decoder(z_decode).float().clamp(0, 1)
            mse_img = F.mse_loss(x_hat, images.float()).item()
            psnr = 10.0 * math.log10(1.0 / max(mse_img, 1e-12)) if mse_img > 0 else float("inf")
            psnr_sum[t] += psnr * bsz

        psnr_n += bsz

    if used_ema:
        system.unet_denoiser.load_state_dict(backup_weights, strict=True)

    print("\n" + "=" * 72)
    print(" 固定时间步 ε 预测 MSE（batch 内全空间平均后再对样本平均）")
    print("=" * 72)
    for t in fixed_ts:
        n = max(1, eps_mse_n[t])
        print(f"  t={t:4d}  eps_mse = {eps_mse_sum[t] / n:.6f}")

    print("\n" + "=" * 72)
    print(f" 编码 → 加噪 → DDIM({args.recon_steps}步) → 解码  图像 PSNR [0,1] 动态范围")
    print("=" * 72)
    for t in fixed_ts:
        print(f"  t={t:4d}  decoded_psnr = {psnr_sum[t] / max(1, psnr_n):.4f} dB")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main_eval()
