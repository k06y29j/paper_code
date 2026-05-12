#!/usr/bin/env python
"""评估 Stage 1 语义编解码 checkpoint：DIV2K 验证集重建 PSNR + 语义编码器输出分布统计。

- PSNR：对每张验证图单独算 PSNR，再 (**总和 / 图像张数**，即算术平均 Σ PSNR_i / N)。
- **语义编码器输出空间**：`encode(..., sample=False)` 得到的 z（与训练中 eval 一致，无 VAE 时为 head 输出，有 VAE 时为 μ），
  在整份验证集上对所有 z 的元素聚合 **均值、方差**（总体统计：σ² = E[z²] − E[z]²，双精度累计）。

用法（在 paper_code 根目录）:
  python test/eval_sc_div2k_psnr.py \
      --checkpoint /workspace/yongjia/paper_code/checkpoints-val/sc/sc_div2k_c4_best.pth \
      --embed-dim 4 \
      --use-vae

``--embed-dim`` / ``--embed_dim``、``--use-vae`` / ``--no-use-vae`` 均为可选；
不设时 ``embed_dim`` 从权重推断，``use_vae`` 由权重是否含 ``vae_proj`` 推断。
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
    wrapped.load_state_dict(model_state_dict, strict=True)
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


def _checkpoint_default() -> str:
    return os.path.join(PROJECT_ROOT, "checkpoints-val/val_12", "sc_div2k_c12_best.pth")


def main() -> None:
    vd = "/workspace/yongjia/datasets/DIV2K/DIV2K_valid_HR/"
    p = argparse.ArgumentParser(description="DIV2K 验证：PSNR(按图均值) + 语义编码输出 z 的均值/方差")
    p.add_argument("--checkpoint", type=str, default=_checkpoint_default())
    p.add_argument(
        "--embed-dim",
        "--embed_dim",
        type=int,
        default=None,
        dest="embed_dim",
        help=(
            "覆盖 config 的 semantic.embed_dim，须与 checkpoint 一致；"
            "不设则从权重里自动推断（与数据集预设不同时会打印）。"
        ),
    )
    p.add_argument(
        "--use-vae",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "是否构建带 VAE 的语义编码器（须与训练一致）。"
            "不设则根据 checkpoint 是否含 semantic_encoder.vae_proj 自动推断。"
        ),
    )
    p.add_argument("--valid-dir", type=str, default=vd)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "none"],
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
            print(
                f"按命令行 semantic.use_vae={cfg.semantic.use_vae}（config 预设为 {cfg_base_vae}）"
            )
    else:
        cfg.semantic.use_vae = inferred_vae
        if inferred_vae != cfg_base_vae:
            print(
                f"已从 checkpoint 推断 semantic.use_vae={inferred_vae}（config 预设为 {cfg_base_vae}）"
            )

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

    print(f"checkpoint: {ckpt_path}")
    print(f"semantic: embed_dim={cfg.semantic.embed_dim}  use_vae={cfg.semantic.use_vae}")
    print(f"valid-dir: {os.path.abspath(args.valid_dir.rstrip(os.sep))}")
    print(f"crop_size: {args.crop_size}")
    print(f"图像数量: {n_img}")
    print(f"PSNR (各图之和/张数 ΣPSNR_i/N): {avg_psnr:.6f} dB  (= {sum_psnr:.6f} / {n_img})")
    print(
        "语义编码器输出 z（encode(sample=False)，验证集全体元素）："
        f"均值={mean_z:.8f}, 方差={var_z:.8e}, σ={std_z:.8e}, 元素个数={int(n_z)}"
    )
    print(f"z shape: {z_shape}")

if __name__ == "__main__":
    main()
