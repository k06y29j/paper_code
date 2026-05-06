#!/usr/bin/env python
"""语义潜空间 AWGN：在 encode(sample=False) 输出 z 上加噪（按样本平均功率约束 SNR），再经解码器测 PSNR。

- 潜空间形状与训练一致：DIV2K 256 中心裁剪下为 [B, C_bottleneck, 16, 16]（C ∈ {4,12,16} 随 checkpoint）。
- SNR（dB）定义：对每个样本 \(z\)，\( \\mathrm{SNR}_{\\mathrm{linear}} = \\mathbb{E}[z^2] / \\sigma_n^2 \)（期望在通道与空间维上），
  加性噪声 \(\epsilon \\sim \\mathcal{N}(0, \\sigma_n^2)\)；即 \( \\mathrm{SNR}_{\\mathrm{dB}} = 10\\log_{10}( \\mathbb{E}[z^2] / \\sigma_n^2 ) \)。
- PSNR：对每张图像单独计算（[0,1]、峰值 1），再 **各图 PSNR 之和除以张数**，与 ``eval_sc_div2k_psnr.py`` 一致。

用法（在 paper_code 根目录）::
  conda activate cddm_ddnm
  MPLCONFIGDIR=/tmp/mpl python test/eval_psnr_snr.py

默认使用三张 DIV2K 验证最优权重，并把 PSNR-SNR 曲线存到 ``image/psnr_vs_snr_c{C}.png``。
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 在无写权限环境下避免 mpl 写 ~/.config/matplotlib
_mpl_cfg = os.environ.get("MPLCONFIGDIR") or os.path.join(PROJECT_ROOT, ".matplotlib_config")
os.makedirs(_mpl_cfg, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cfg)

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

import matplotlib.pyplot as plt  # noqa: E402

import torch
from torch.utils.data import DataLoader

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import SemanticCommSystem, SystemConfig, get_div2k_config  # noqa: E402
from src.cddm_mimo_ddnm.datasets import DIV2KDataset  # noqa: E402
from train.train_sc import Stage1Wrapper  # noqa: E402

SNR_DB_VALUES = [0, 3, 6, 9, 12, 15]


def _infer_embed_dim_from_state_dict(sd: dict) -> int | None:
    tail = "semantic_encoder.head.weight"
    for prefix in ("system.", "", "module.system.", "module."):
        w = sd.get(prefix + tail)
        if w is not None and w.ndim >= 2:
            return int(w.shape[0])
    return None


def _parse_amp(s: str) -> tuple[bool, torch.dtype]:
    if s == "none":
        return False, torch.float32
    if s == "bfloat16":
        return True, torch.bfloat16
    if s == "float16":
        return True, torch.float16
    raise ValueError(f"Unknown --amp-dtype={s}")


def normalize_valid_root(p: str) -> str:
    """支持父目录 DIV2K 或直接指向 DIV2K_valid_HR 文件夹。"""
    p = os.path.abspath(p.rstrip(os.sep))
    base = os.path.basename(p)
    if base == "DIV2K_valid_HR" and os.path.isdir(p):
        return p
    return p


def latent_awgn_clean(z_clean: torch.Tensor, snr_db: float, noise_f: torch.Tensor) -> torch.Tensor:
    """在 z_clean 上叠加与 z_clean 同 dtype 的设备张量噪声；noise_f 为标准正态 float32/[B,C,H,W]。"""
    # 每样本平均功率 [B]，与空间/通道维度一致
    per_sample_pwr = z_clean.detach().pow(2).flatten(1).mean(dim=1).clamp(min=1e-20)
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma2 = (per_sample_pwr / snr_lin).view(-1, 1, 1, 1)
    sigma = sigma2.sqrt()
    nz = noise_f * sigma.to(noise_f.dtype)
    return z_clean + nz.to(dtype=z_clean.dtype)


def _psnr_batch_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse_img = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3)).clamp(min=1e-12)
    return (10.0 * torch.log10(1.0 / mse_img)).detach().cpu().double()


@torch.no_grad()
def evaluate_snr_curve(
    *,
    cfg: SystemConfig,
    model_state_dict: dict,
    valid_root: str,
    snr_db_list: list[float],
    crop_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    base_seed: int,
) -> tuple[int, dict[float, float], list[float]]:
    """返回 (N, {snr: avg_psnr}, snr_sorted_list)。"""
    core = SemanticCommSystem(cfg).to(device)
    wrapped = Stage1Wrapper(core).to(device)
    wrapped.load_state_dict(model_state_dict, strict=True)
    wrapped.eval()

    root = normalize_valid_root(valid_root)
    ds = DIV2KDataset(root, crop_size=crop_size, split="valid")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=(4 if num_workers > 0 else None),
    )
    nb = device.type == "cuda"
    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)
    autocast_cm = (
        torch.autocast("cuda", **autocast_kw)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    n_img = len(ds)
    avg_by_snr: dict[float, float] = {}

    for snr_db in snr_db_list:
        generator = torch.Generator(device=device)
        generator.manual_seed(base_seed + int(snr_db) * 10_013)
        psnr_chunks: list[torch.Tensor] = []

        for batch in loader:
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=nb)

            with autocast_cm:
                z_clean, _, _ = core.semantic_encoder.encode(imgs, sample=False)
            # 加噪在高精度上算 sigma，避免 bf16 下功率过小误判
            z32 = z_clean.float()
            noise = torch.randn(
                z32.shape,
                generator=generator,
                device=z32.device,
                dtype=z32.dtype,
            )
            z_noisy32 = latent_awgn_clean(z32, snr_db, noise)

            z_in = z_noisy32.to(dtype=z_clean.dtype)
            with autocast_cm:
                x_hat = core.semantic_decoder(z_in)

            xf = x_hat.float().clamp(0, 1)
            tf = imgs.float()
            psnr_chunks.append(_psnr_batch_per_image(xf, tf))

        avg_by_snr[snr_db] = float(torch.cat(psnr_chunks).sum().item()) / float(max(n_img, 1))

    return n_img, avg_by_snr, list(snr_db_list)


def plot_one_curve(snr_vals: list[float], psnr_vals: list[float], title: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(snr_vals, psnr_vals, marker="o", linewidth=2, markersize=7)
    ax.set_xlabel("SNR (dB) latent AWGN")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.set_xticks(snr_vals)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def default_checkpoints() -> list[str]:
    base = os.path.join(PROJECT_ROOT, "checkpoints-val", "sc")
    return [
        os.path.join(base, "sc_div2k_c4_best.pth"),
        os.path.join(base, "sc_div2k_c12_best.pth"),
        os.path.join(base, "sc_div2k_c16_best.pth"),
    ]


def main() -> None:
    default_valid = "/workspace/yongjia/datasets/DIV2K"
    p = argparse.ArgumentParser(description="语义潜空间噪声 SNR vs 解码 PSNR（DIV2K valid）")
    p.add_argument(
        "--checkpoints",
        nargs="+",
        default=default_checkpoints(),
        help="Stage1 语义 checkpoint（含 model_state_dict）列表，顺序对应输出图文件名 c4/c12/c16",
    )
    p.add_argument(
        "--valid-dir",
        type=str,
        default=default_valid,
        help=(
            "验证集目录：若为 DIV2K 根目录应含子目录 DIV2K_valid_HR；"
            "也可直接传入已含 PNG 的 DIV2K_valid_HR 文件夹。"
        ),
    )
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=20260502)
    p.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        default=SNR_DB_VALUES,
        help="信道（潜空间）SNR（dB）列表",
    )
    p.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "none"],
    )
    p.add_argument(
        "--image-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "image"),
        help="曲线 PNG 输出目录（每张 checkpoint 一幅图）",
    )
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

    snr_list = sorted(set(float(x) for x in args.snr_db))
    ckpts = [os.path.abspath(c) for c in args.checkpoints]

    print("SNR values (dB):", snr_list)
    print("valid-root:", normalize_valid_root(args.valid_dir))

    for ci, ckpt_path in enumerate(ckpts):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "metrics" in ckpt:
            print(f"\n[{os.path.basename(ckpt_path)}] ckpt.metrics: {ckpt['metrics']}")

        cfg = get_div2k_config()
        inferred = _infer_embed_dim_from_state_dict(ckpt["model_state_dict"])
        if inferred is None:
            raise RuntimeError(f"Cannot infer embed_dim from {ckpt_path}")
        cfg.semantic.embed_dim = inferred

        n_img, avg_map, snr_ordered = evaluate_snr_curve(
            cfg=cfg,
            model_state_dict=ckpt["model_state_dict"],
            valid_root=args.valid_dir,
            snr_db_list=snr_list,
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            base_seed=args.seed + ci * 9973,
        )

        psnr_list = [avg_map[s] for s in snr_ordered]

        infer_c = inferred
        out_png = os.path.join(args.image_dir, f"psnr_vs_snr_c{infer_c}.png")

        plot_one_curve(
            snr_ordered,
            psnr_list,
            title=f"DIV2K valid PSNR vs latent SNR (bottleneck C={infer_c})",
            out_path=out_png,
        )

        print(f"\n=== {ckpt_path} ===")
        print(f"embed_dim (inferred): {infer_c}  images: {n_img}")
        for s, v in zip(snr_ordered, psnr_list):
            print(f"  SNR={s:>4} dB -> avg PSNR = {v:.4f} dB")
        print(f"curve saved -> {out_png}")


if __name__ == "__main__":
    main()
