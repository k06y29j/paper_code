#!/usr/bin/env python
"""评估 Stage 1 语义编解码 checkpoint：DIV2K 验证集重建 PSNR + 训练集 Latent 通道分布统计。

功能：
  1. 验证集 PSNR 评估（与原版一致）
  2. 训练集全量 encode → 统计 latent [B, C, H, W] 在通道维度 C 上的逐通道均值/方差
  3. 绘制直方图：
     - 逐通道均值分布 (per-channel mean)
     - 逐通道方差分布 (per-channel variance)
     - 全体 latent 值的分布 (all elements)
     - 16 个通道各自的值分布叠加

用法:
  python test/eval_sc_div2k_psnr.py \
      --checkpoint checkpoints-val/sc/sc_div2k_c12_best.pth 

  # 不画图，仅统计
  python test/eval_sc_div2k_psnr.py --checkpoint ... --train-dir ...
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import SemanticCommSystem, SystemConfig, get_div2k_config  # noqa: E402
from src.cddm_mimo_ddnm.datasets import DIV2KDataset  # noqa: E402
from train.train_sc import Stage1Wrapper  # noqa: E402


def _infer_embed_dim_from_state_dict(sd: dict) -> int | None:
    """从 checkpoint 中 semantic_encoder.head 形状推断 embed_dim（与 Stage1Wrapper 键名兼容）。"""
    tail = "semantic_encoder.head.weight"
    for prefix in ("system.", "", "module.system.", "module."):
        w = sd.get(prefix + tail)
        if w is not None and w.ndim >= 2:
            return int(w.shape[0])
    return None


def _infer_use_vae_from_state_dict(sd: dict) -> bool:
    """若 state_dict 含语义编码器 vae_proj，则与训练时 use_vae=True 一致。"""
    for k in sd:
        if "semantic_encoder" in k and "vae_proj" in k:
            return True
    return False


def _load_weights_compat(model: torch.nn.Module, state_dict: dict) -> None:
    """兼容加载权重：先用 strict=True 尝试，若因信道编解码器结构不匹配则
    只加载语义编解码器部分（Stage 1 评估不依赖信道编解码器权重）。"""
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as e:
        err_msg = str(e)

    # 宽松加载：只加载 key 匹配的权重
    model_sd = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
    missing = [k for k in model_sd if k not in matched]
    unexpected = [k for k in state_dict if k not in model_sd]

    model.load_state_dict(matched, strict=False)

    # 只报告语义编解码器相关的缺失（信道编解码器缺失不影响 Stage 1）
    sc_missing = [k for k in missing if "semantic_" in k]
    sc_unexpected = [k for k in unexpected if "semantic_" in k]
    if sc_missing or sc_unexpected:
        print(f"[WARN] 语义编解码器权重不匹配: missing={sc_missing}, unexpected={sc_unexpected}")
    else:
        n_skip_missing = len(missing)
        n_skip_unexpected = len(unexpected)
        if n_skip_missing or n_skip_unexpected:
            print(f"[INFO] 跳过不匹配的信道编解码器权重 "
                  f"(missing={n_skip_missing}, unexpected={n_skip_unexpected})，"
                  f"不影响 Stage 1 评估")


def _parse_amp(s: str) -> tuple[bool, torch.dtype]:
    if s == "none":
        return False, torch.float32
    if s == "bfloat16":
        return True, torch.bfloat16
    if s == "float16":
        return True, torch.float16
    raise ValueError(f"Unknown --amp-dtype={s}")


def _psnr_batch_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """每样本 PSNR (dB)，[B,C,H,W]、[0,1]，峰值=1。CPU float64 tensor [B]。"""
    mse_img = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3)).clamp(min=1e-12)
    pi = (10.0 * torch.log10(1.0 / mse_img)).detach().cpu().double()
    return pi


# ---------------------------------------------------------------------------
# 新增：训练集 latent 全量编码 + 通道维度统计
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_train_set(
    *,
    cfg: SystemConfig,
    model_state_dict: dict,
    train_root: str,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict:
    """对训练集全量 encode，收集 latent 并统计通道维度分布。

    Returns dict:
        all_z:          [N_total, C]  (flatten 后的所有 latent 向量，按通道排列)
        per_ch_mean:    [C] 逐通道均值 (跨所有样本和空间位置)
        per_ch_var:     [C] 逐通道方差
        per_ch_std:     [C] 逐通道标准差
        global_mean:    float 全局均值
        global_var:     float 全局方差
        z_shape:        tuple 最后一批的 shape
        n_samples:      int 总样本数
    """
    core = SemanticCommSystem(cfg).to(device)
    wrapped = Stage1Wrapper(core).to(device)
    _load_weights_compat(wrapped, model_state_dict)
    wrapped.eval()

    ds = DIV2KDataset(train_root, crop_size=crop_size, split="train")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=(4 if num_workers > 0 else None),
    )

    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)
    nb = device.type == "cuda"
    autocast_cm = (
        torch.autocast("cuda", **autocast_kw)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    # 逐通道双精度累加器: [C]
    C = cfg.semantic.embed_dim
    sum_c = torch.zeros(C, dtype=torch.float64)
    sum_c2 = torch.zeros(C, dtype=torch.float64)
    n_per_ch = 0  # 每个通道的元素数 (相同) = N * H * W

    # 收集所有 z (float32) 用于画图，按 batch 拼接 [N_total, C, H, W]
    z_blocks: list[torch.Tensor] = []
    last_shape = None
    total_samples = 0

    for batch in loader:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=nb)

        with autocast_cm:
            z_sem, _, _ = core.semantic_encoder.encode(imgs, sample=False)

        z_f = z_sem.detach().float()  # [B, C, H, W]
        B, _C, H, W = z_f.shape
        last_shape = (B, _C, H, W)
        total_samples += B

        # 逐通道统计: 对 (B, H, W) 维度求和
        z64 = z_f.double()
        # sum over B, H, W → [C]
        sum_c += z64.sum(dim=(0, 2, 3)).cpu()
        sum_c2 += z64.pow(2).sum(dim=(0, 2, 3)).cpu()
        n_per_ch += B * H * W

        # 收集到 CPU 用于画图（float32 节省内存）
        z_blocks.append(z_f.cpu())

    # 逐通道均值/方差
    n = float(max(n_per_ch, 1))
    per_ch_mean = sum_c / n                           # [C]
    per_ch_var = sum_c2 / n - per_ch_mean.pow(2)       # [C]
    per_ch_var = per_ch_var.clamp(min=0.0)
    per_ch_std = per_ch_var.sqrt()

    # 全局统计
    all_z = torch.cat(z_blocks, dim=0)  # [N_total, C, H, W]
    global_mean = float(per_ch_mean.mean())
    global_var = float(per_ch_var.mean())

    return {
        "all_z": all_z,             # [N_total, C, H, W] float32 on CPU
        "per_ch_mean": per_ch_mean,  # [C] float64
        "per_ch_var": per_ch_var,    # [C] float64
        "per_ch_std": per_ch_std,    # [C] float64
        "global_mean": global_mean,
        "global_var": global_var,
        "z_shape": last_shape,
        "n_samples": total_samples,
    }


# ---------------------------------------------------------------------------
# 原版验证集评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    *,
    cfg: SystemConfig,
    model_state_dict: dict,
    valid_root: str,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[int, torch.Tensor, float, float, float, tuple[int, ...]]:
    """返回 (N, 各图 PSNR向量, z 元素均值, z 元素方差, z 元素个数, 最后一批 z 的 shape)。"""
    core = SemanticCommSystem(cfg).to(device)
    wrapped = Stage1Wrapper(core).to(device)
    _load_weights_compat(wrapped, model_state_dict)
    wrapped.eval()

    ds = DIV2KDataset(valid_root, crop_size=crop_size, split="valid")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=(4 if num_workers > 0 else None),
    )

    per_blocks: list[torch.Tensor] = []
    sum_z = 0.0
    sum_z2 = 0.0
    n_elem_z = 0

    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)
    nb = device.type == "cuda"
    autocast_cm = (
        torch.autocast("cuda", **autocast_kw)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    for batch in loader:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=nb)

        with autocast_cm:
            z_sem, _, _ = core.semantic_encoder.encode(imgs, sample=False)
            x_hat = core.semantic_decoder(z_sem)

        z_f = z_sem.detach().float()
        z64 = z_f.double()
        sum_z += z64.sum().cpu().item()
        sum_z2 += z64.pow(2).sum().cpu().item()
        n_elem_z += z_f.numel()

        xf = x_hat.float().clamp(0, 1)
        tf = imgs.float()

        per_blocks.append(_psnr_batch_per_image(xf, tf))

    per_image_psnr = torch.cat(per_blocks)
    n_img = len(ds)
    denom = float(max(n_elem_z, 1))
    mean_z = sum_z / denom
    mean_z2 = sum_z2 / denom
    var_z = mean_z2 - mean_z * mean_z
    if var_z < 0 and var_z > -1e-9:
        var_z = 0.0

    return n_img, per_image_psnr, mean_z, var_z, float(n_elem_z), z_sem.shape


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_latent_distribution(stats: dict, save_path: str, cfg: SystemConfig) -> None:
    """绘制 4 张子图：逐通道均值、逐通道方差、全体值分布、各通道分布叠加。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_z = stats["all_z"]          # [N, C, H, W]
    per_ch_mean = stats["per_ch_mean"]  # [C]
    per_ch_var = stats["per_ch_var"]    # [C]
    per_ch_std = stats["per_ch_std"]    # [C]
    C = all_z.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Training Set Latent Distribution  |  "
        f"z shape: {list(all_z.shape)}  |  "
        f"embed_dim={cfg.semantic.embed_dim}  use_vae={cfg.semantic.use_vae}",
        fontsize=12,
    )

    # ---- (1) 逐通道均值 ----
    ax = axes[0, 0]
    ch_indices = list(range(C))
    ax.bar(ch_indices, per_ch_mean.numpy(), color="steelblue", alpha=0.8)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Mean")
    ax.set_title(f"Per-Channel Mean  (global mean={stats['global_mean']:.4f})")
    ax.axhline(y=stats["global_mean"], color="red", linestyle="--", linewidth=1, label=f"global={stats['global_mean']:.4f}")
    ax.legend(fontsize=8)

    # ---- (2) 逐通道方差 ----
    ax = axes[0, 1]
    ax.bar(ch_indices, per_ch_var.numpy(), color="darkorange", alpha=0.8)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Variance")
    ax.set_title(f"Per-Channel Variance  (global var={stats['global_var']:.4f})")
    ax.axhline(y=stats["global_var"], color="red", linestyle="--", linewidth=1, label=f"global={stats['global_var']:.4f}")
    ax.legend(fontsize=8)

    # ---- (3) 全体 latent 值分布直方图 ----
    ax = axes[1, 0]
    all_values = all_z.flatten().numpy()
    n_bins = min(200, max(50, len(all_values) // 10000))
    ax.hist(all_values, bins=n_bins, density=True, color="teal", alpha=0.75, edgecolor="none")
    ax.set_xlabel("Latent Value")
    ax.set_ylabel("Density")
    ax.set_title(
        f"All Latent Values  |  μ={stats['global_mean']:.4f}  "
        f"σ²={stats['global_var']:.4f}  σ={math.sqrt(stats['global_var']):.4f}"
    )
    # 叠加标准正态参考线
    import numpy as np
    x_norm = np.linspace(all_values.min(), all_values.max(), 300)
    # 匹配均值和方差的高斯
    g = stats["global_mean"]
    s = math.sqrt(max(stats["global_var"], 1e-12))
    y_norm = (1.0 / (s * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x_norm - g) / s) ** 2)
    ax.plot(x_norm, y_norm, "r--", linewidth=1.5, label=f"N({g:.2f}, {s:.2f}²)")
    ax.legend(fontsize=8)

    # ---- (4) 各通道分布叠加 ----
    ax = axes[1, 1]
    # 对每个通道 flatten 后画直方图 (用半透明叠加)
    cmap = plt.cm.get_cmap("tab20", C) if C <= 20 else plt.cm.get_cmap("hsv", C)
    for c in range(C):
        ch_vals = all_z[:, c, :, :].flatten().numpy()
        ax.hist(ch_vals, bins=min(100, max(30, len(ch_vals) // 5000)),
                density=True, alpha=0.3, color=cmap(c), label=f"ch{c}" if C <= 8 else None)
    ax.set_xlabel("Latent Value")
    ax.set_ylabel("Density")
    ax.set_title("Per-Channel Distribution Overlay")
    if C <= 8:
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  >> 直方图已保存至: {save_path}")


def _checkpoint_default() -> str:
    return os.path.join(PROJECT_ROOT, "checkpoints-val/val_12", "sc_div2k_c12_best.pth")


def main() -> None:
    vd = "/workspace/yongjia/datasets/DIV2K/DIV2K_valid_HR/"
    td = "/workspace/yongjia/datasets/DIV2K/DIV2K_train_HR/"
    p = argparse.ArgumentParser(
        description="DIV2K 验证 PSNR + 训练集 Latent 通道维度分布统计与直方图"
    )
    p.add_argument("--checkpoint", type=str, default=_checkpoint_default())
    p.add_argument(
        "--embed-dim", "--embed_dim", type=int, default=None, dest="embed_dim",
        help="覆盖 semantic.embed_dim，不设则从权重推断",
    )
    p.add_argument(
        "--use-vae", action=argparse.BooleanOptionalAction, default=None,
        help="是否使用 VAE（不设则从权重推断）",
    )
    p.add_argument("--valid-dir", type=str, default=vd)
    p.add_argument("--train-dir", type=str, default=td,
                   help="训练集目录（用于 latent 分布统计）")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp-dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "none"])
    p.add_argument("--plot", action="store_true",
                   help="绘制并保存 latent 分布直方图")
    p.add_argument("--plot-save", type=str, default=None,
                   help="直方图保存路径（默认自动生成）")
    p.add_argument("--skip-val", action="store_true",
                   help="跳过验证集 PSNR 评估（仅做训练集 latent 统计）")
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

    ckpt_path = os.path.abspath(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "metrics" in ckpt:
        print(f"ckpt.metrics: {ckpt['metrics']}")

    cfg = get_div2k_config()
    sd = ckpt["model_state_dict"]
    inferred = _infer_embed_dim_from_state_dict(sd)
    cfg_base_embed = cfg.semantic.embed_dim
    if args.embed_dim is not None:
        cfg.semantic.embed_dim = int(args.embed_dim)
    elif inferred is not None:
        cfg.semantic.embed_dim = inferred
        if inferred != cfg_base_embed:
            print(f"已从 checkpoint 推断 semantic.embed_dim={inferred}（config 预设为 {cfg_base_embed}）")
    else:
        print(
            "警告：无法从 state_dict 推断 embed_dim，使用 config.py 默认值 "
            f"{cfg.semantic.embed_dim}；若不匹配请加 --embed-dim / --embed_dim"
        )

    inferred_vae = _infer_use_vae_from_state_dict(sd)
    cfg_base_vae = cfg.semantic.use_vae
    if args.use_vae is not None:
        cfg.semantic.use_vae = bool(args.use_vae)
        if cfg.semantic.use_vae != cfg_base_vae:
            print(f"按命令行 semantic.use_vae={cfg.semantic.use_vae}（config 预设为 {cfg_base_vae}）")
    else:
        cfg.semantic.use_vae = inferred_vae
        if inferred_vae != cfg_base_vae:
            print(f"已从 checkpoint 推断 semantic.use_vae={inferred_vae}（config 预设为 {cfg_base_vae}）")

    # ---- 验证集 PSNR ----
    if not args.skip_val:
        n_img, imgs_psnr, mean_z, var_z, n_z, z_shape = evaluate(
            cfg=cfg,
            model_state_dict=ckpt["model_state_dict"],
            valid_root=os.path.abspath(args.valid_dir.rstrip(os.sep)),
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        sum_psnr = float(imgs_psnr.sum().item())
        avg_psnr = sum_psnr / float(max(n_img, 1))
        std_z = math.sqrt(max(var_z, 0.0))

        print(f"\n{'='*60}")
        print(f"  验证集 PSNR 评估")
        print(f"{'='*60}")
        print(f"checkpoint: {ckpt_path}")
        print(f"semantic: embed_dim={cfg.semantic.embed_dim}  use_vae={cfg.semantic.use_vae}")
        print(f"valid-dir: {os.path.abspath(args.valid_dir.rstrip(os.sep))}")
        print(f"crop_size: {args.crop_size}")
        print(f"图像数量: {n_img}")
        print(f"PSNR (ΣPSNR_i/N): {avg_psnr:.6f} dB  (= {sum_psnr:.6f} / {n_img})")
        print(f"z 全局: 均值={mean_z:.8f}, 方差={var_z:.8e}, σ={std_z:.8e}")
        print(f"z shape: {z_shape}")

    # ---- 训练集 latent 通道分布统计 ----
    train_dir = os.path.abspath(args.train_dir.rstrip(os.sep))
    if not os.path.isdir(train_dir):
        print(f"\n[WARN] 训练集目录不存在: {train_dir}，跳过 latent 统计")
    else:
        print(f"\n{'='*60}")
        print(f"  训练集 Latent 通道维度分布统计")
        print(f"{'='*60}")
        print(f"train-dir: {train_dir}")

        stats = encode_train_set(
            cfg=cfg,
            model_state_dict=ckpt["model_state_dict"],
            train_root=train_dir,
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        per_ch_mean = stats["per_ch_mean"]
        per_ch_var = stats["per_ch_var"]
        per_ch_std = stats["per_ch_std"]
        C = len(per_ch_mean)

        print(f"z shape: {stats['z_shape']}  (总样本: {stats['n_samples']})")
        print(f"\n逐通道统计 (共 {C} 通道):")
        print(f"  {'ch':>4s}  {'mean':>12s}  {'var':>12s}  {'std':>12s}")
        print(f"  {'----':>4s}  {'------------':>12s}  {'------------':>12s}  {'------------':>12s}")
        for c in range(C):
            print(f"  {c:>4d}  {per_ch_mean[c]:>12.6f}  {per_ch_var[c]:>12.6f}  {per_ch_std[c]:>12.6f}")
        print(f"\n全局: mean={stats['global_mean']:.6f}  "
              f"var={stats['global_var']:.6f}  "
              f"std={math.sqrt(stats['global_var']):.6f}")
        print(f"通道均值范围: [{per_ch_mean.min():.6f}, {per_ch_mean.max():.6f}]")
        print(f"通道方差范围: [{per_ch_var.min():.6f}, {per_ch_var.max():.6f}]")

        # ---- 绘图 ----
        if args.plot:
            if args.plot_save:
                plot_path = args.plot_save
            else:
                ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
                plot_path = os.path.join(
                    PROJECT_ROOT, "test", "plots",
                    f"latent_dist_{ckpt_name}_c{C}.png",
                )
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plot_latent_distribution(stats, plot_path, cfg)


if __name__ == "__main__":
    main()
