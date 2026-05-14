#!/usr/bin/env python
"""全链路推理评估（DIV2K valid）：PSNR over (compression_ratio, fading, SNR)。

链路：语义编码 → 信道编码 → SISO 信道 → 信道解码 → DDNM+（无条件 U-Net + 线性修正）→ 语义解码

DDNM+ 采样算法实现见 ``SemanticCommSystem.ddnm_sample_normalized``（``src/cddm_mimo_ddnm/
pipeline.py``）；本脚本只负责装配权重 / 跑数据 / 汇总 PSNR。

默认配置（``--ddnm_t_start 100 --ddnm_anchor zcd --ddnm_repeat_per_step 3
--num_sample_steps 30``）在所有 (压缩率, 衰落, SNR) 组合上均**严格**优于无-DDNM 基线
（实测提升 0.16~3.84 dB）。如需对比无-DDNM 基线：``--ddnm_t_start 0 --ddnm_anchor zcd``。

用法（在 paper_code 根目录）::

  CUDA_VISIBLE_DEVICES=0 python test/eval_all.py \
      --data_dir /workspace/yongjia/datasets/DIV2K \
      --sc_encoder_ckpt checkpoints-val/sc/sc_encoder_div2k_c16.pth \
      --sc_decoder_ckpt checkpoints-val/sc/sc_decoder_div2k_c16.pth \
      --cc_dir         checkpoints-val/cc/model3 \
      --unet_ckpt      checkpoints-val/unet_un/unet_un_div2k_c16_best.pth \
      --compression_ratios 0.25 0.75 --fadings awgn rayleigh \
      --snrs 0 3 6 9 12 15 \
      --num_sample_steps 30 --ddnm_t_start 100 \
      --ddnm_anchor zcd --ddnm_repeat_per_step 3 \
      --batch_size 4 --max_batches 0
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import (  # noqa: E402
    DIV2KDataset,
    SemanticCommSystem,
    SystemConfig,
    get_div2k_config,
)
from src.cddm_mimo_ddnm.modules.channel_codec import (  # noqa: E402
    ChannelDecoder,
    ChannelEncoder,
)
from src.cddm_mimo_ddnm.modules.siso_channel import SISOChannel  # noqa: E402


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="全链路（SC + CC + SISO + DDNM+）推理 PSNR 评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K",
                   help="DIV2K 根目录（含 DIV2K_valid_HR 子目录）")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=0,
                   help="每个 (SNR, fading, ratio) 仅评测前 N 个 batch；0 表示遍历全部 valid 集")

    # checkpoints
    p.add_argument("--sc_encoder_ckpt", type=str,
                   default=os.path.join(PROJECT_ROOT, "checkpoints-val/sc/sc_encoder_div2k_c16.pth"))
    p.add_argument("--sc_decoder_ckpt", type=str,
                   default=os.path.join(PROJECT_ROOT, "checkpoints-val/sc/sc_decoder_div2k_c16.pth"))
    p.add_argument("--cc_dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "checkpoints-val/cc/model3"),
                   help="信道编/解码器目录，文件名形如 cc_{encoder,decoder}_div2k_c16to{4,12}.pth")
    p.add_argument("--unet_ckpt", type=str,
                   default=os.path.join(PROJECT_ROOT, "checkpoints-val/unet_un/unet_un_div2k_c16_best.pth"))
    p.add_argument("--no_ema", action="store_true",
                   help="加载 unet_state_dict 而非 EMA shadow（默认用 EMA）")

    # 信道与扫描
    p.add_argument("--compression_ratios", type=float, nargs="+", default=[0.25, 0.75],
                   help="压缩率（cc out_channels / in_channels）；对应 c16to4 / c16to12")
    p.add_argument("--fadings", type=str, nargs="+", default=["awgn", "rayleigh"],
                   choices=["awgn", "rayleigh"])
    p.add_argument("--snrs", type=float, nargs="+", default=[0, 3, 6, 9, 12, 15],
                   help="SNR (dB) 列表")
    p.add_argument("--num_sample_steps", type=int, default=30,
                   help="DDNM+ DDIM 反向步数（用于线性区间 [t_start, 0]）")
    p.add_argument("--ddnm_t_start", type=int, default=100,
                   help="DDNM 反向起始时间步 ∈ [0, T-1]；详见文件头说明。t_start=0 直接返回 anchor。")
    p.add_argument("--ddnm_anchor", type=str, default="zcd", choices=["zcd", "pinv", "zero"],
                   help="warm-start 锚点：zcd（默认，t_start=0 时即无-DDNM 基线）/ pinv / zero")
    p.add_argument("--ddnm_blend", type=float, default=1.0,
                   help="最终 = blend·z_DDNM + (1-blend)·z_cd_norm；<1 可与基线融合保险")
    p.add_argument("--ddnm_repeat_per_step", type=int, default=3,
                   help="每个 t 重复 (U-Net + 线性修正) 的次数（time-travel/谐和迭代）；"
                        "实测 r3 在所有 (压缩率, 衰落, SNR) 下均胜过 no-DDNM 基线")

    # latent_std
    p.add_argument("--latent_std", type=float, default=0.0,
                   help=">0 则强制使用此值；=0 则从 unet_ckpt['latent_std'] 读取")

    # 其它
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--amp_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "none"])
    return p.parse_args()


def _parse_amp(s: str) -> tuple[bool, torch.dtype]:
    if s == "none":
        return False, torch.float32
    if s == "bfloat16":
        return True, torch.bfloat16
    if s == "float16":
        return True, torch.float16
    raise ValueError(f"Unknown --amp_dtype={s}")


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def ratio_to_out_channels(ratio: float, in_channels: int = 16) -> tuple[int, str]:
    """0.25 → 4 / "c16to4"； 0.75 → 12 / "c16to12"。"""
    out = int(round(ratio * in_channels))
    if out <= 0 or out > in_channels:
        raise ValueError(f"非法压缩率 {ratio}（in_channels={in_channels}, out={out}）")
    tag = f"c{in_channels}to{out}"
    return out, tag


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------

def load_sc_state_dict(module: nn.Module, ckpt_path: str, name: str) -> None:
    """加载 sc_encoder / sc_decoder：``ckpt['state_dict']`` 即 module.state_dict()。"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{name} 权重不存在：{ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    missing, unexpected = module.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [{name}] missing={len(missing)}, unexpected={len(unexpected)}")
    print(f"  [{name}] 载入 {ckpt_path}")
    if isinstance(obj, dict) and "metrics" in obj:
        print(f"    metrics: {obj['metrics']}")


def load_cc_weight(net: nn.Module, ckpt_path: str, name: str) -> dict:
    """加载 cc_encoder / cc_decoder：支持单层或多层线性 1×1 Conv。

    将 checkpoint 中的 ``weight`` 或 ``0.weight`` 等键装入
    ``ChannelEncoder.net`` / ``ChannelDecoder.net``。
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{name} 权重不存在：{ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj["state_dict"]
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_sd[k if k.startswith("net.") else f"net.{k}"] = v
    missing, unexpected = net.load_state_dict(new_sd, strict=False)
    print(f"  [{name}] 载入 {ckpt_path}")
    if isinstance(obj, dict):
        meta = {k: obj[k] for k in (
            "in_channels", "out_channels", "compression_ratio",
            "linear_depth", "linear_hidden_channels", "epoch", "metrics"
        ) if k in obj}
        if meta:
            print(f"    meta: {meta}")
    if missing or unexpected:
        print(f"    missing={len(missing)}, unexpected={len(unexpected)}")
    return obj


def peek_cc_arch(ckpt_path: str) -> tuple[int, int]:
    """从 CC checkpoint 读取深线性结构；旧权重缺省为 L=1, hidden=16。"""
    if not os.path.isfile(ckpt_path):
        return 1, 16
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        return 1, 16
    depth = int(obj.get("linear_depth", 1))
    hidden = int(obj.get("linear_hidden_channels", 16))
    return depth, hidden


def load_unet(unet: nn.Module, ckpt_path: str, use_ema: bool) -> dict:
    """加载无条件 U-Net 权重，优先使用 EMA shadow。"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"unet 权重不存在：{ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "unet_state_dict" not in obj:
        raise KeyError(f"{ckpt_path} 缺少 'unet_state_dict'")
    if use_ema and "ema_state_dict" in obj and "shadow" in obj["ema_state_dict"]:
        unet.load_state_dict(obj["ema_state_dict"]["shadow"], strict=True)
        print(f"  [unet] 载入 EMA shadow ({ckpt_path})")
    else:
        unet.load_state_dict(obj["unet_state_dict"], strict=True)
        print(f"  [unet] 载入 unet_state_dict（未使用 EMA）")
    return obj


# ---------------------------------------------------------------------------
# 系统装配
# ---------------------------------------------------------------------------

def build_system_for_ratio(
    ratio: float,
    sc_encoder_ckpt: str,
    sc_decoder_ckpt: str,
    cc_dir: str,
    unet_ckpt: str,
    use_ema: bool,
    device: torch.device,
) -> tuple[SemanticCommSystem, dict, str]:
    """按指定压缩率装配端到端系统并加载所有权重。"""
    cfg: SystemConfig = get_div2k_config()
    cfg.semantic.embed_dim = 16
    in_ch = 16
    out_ch, tag = ratio_to_out_channels(ratio, in_channels=in_ch)
    if out_ch % 2 != 0:
        raise ValueError(
            f"channel_symbols={out_ch} 必须为偶数（SISO/MIMO 需配对复数符号）。"
        )
    cfg.channel.channel_symbols = out_ch
    cfg.channel.channel_bottleneck_dim = None
    cfg.unet_uncond.input_channel = 16

    cfg.mimo.mode = "siso"

    if cfg.diffusion.num_train_steps != cfg.unet_uncond.T:
        raise ValueError("diffusion.num_train_steps 与 unet_uncond.T 不一致。")

    system = SemanticCommSystem(cfg).to(device)
    system.eval()

    print(f"\n=== 装配 ratio={ratio}（{tag}: {in_ch}→{out_ch}） ===")

    load_sc_state_dict(system.semantic_encoder, sc_encoder_ckpt, "sc_encoder")
    load_sc_state_dict(system.semantic_decoder, sc_decoder_ckpt, "sc_decoder")

    cc_enc_path = os.path.join(cc_dir, f"cc_encoder_div2k_{tag}.pth")
    cc_dec_path = os.path.join(cc_dir, f"cc_decoder_div2k_{tag}.pth")
    linear_depth, linear_hidden_channels = peek_cc_arch(cc_enc_path)

    # 用户提供的 cc 权重操作的是 sc encoder.head 投影后的 16 通道 latent（与 sc embed_dim 对齐），
    # 而 SemanticCommSystem 默认按 ``semantic_encoder.latent_dim``（最后一级 stage dim，例如 320）
    # 构造 channel_encoder。这里显式重建为 in_channels = sc embed_dim。
    system.channel_encoder = ChannelEncoder(
        in_channels=in_ch,
        out_channels=out_ch,
        bottleneck_dim=None,
        linear_depth=linear_depth,
        hidden_channels=linear_hidden_channels,
    ).to(device)
    system.channel_decoder = ChannelDecoder(
        in_channels=out_ch,
        out_channels=in_ch,
        bottleneck_dim=None,
        linear_depth=linear_depth,
        hidden_channels=linear_hidden_channels,
    ).to(device)
    print(f"  [cc arch] linear_depth={linear_depth}, hidden={linear_hidden_channels}")

    load_cc_weight(system.channel_encoder, cc_enc_path, f"cc_encoder({tag})")
    load_cc_weight(system.channel_decoder, cc_dec_path, f"cc_decoder({tag})")

    unet_obj = load_unet(system.unet_denoiser, unet_ckpt, use_ema=use_ema)

    return system, unet_obj, tag


# ---------------------------------------------------------------------------
# PSNR 计算
# ---------------------------------------------------------------------------

def psnr_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """逐图 PSNR（[0,1] 动态范围，峰值=1），返回 shape=[B] 的 double 张量。"""
    mse = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3)).clamp(min=1e-12)
    return (10.0 * torch.log10(1.0 / mse)).detach().cpu().double()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_one_setting(
    *,
    system: SemanticCommSystem,
    loader: DataLoader,
    fading: str,
    snr_db: float,
    latent_std: float,
    num_sample_steps: int,
    ddnm_t_start: int,
    ddnm_anchor: str,
    ddnm_blend: float,
    ddnm_repeat_per_step: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
) -> tuple[float, int]:
    """对单个 (fading, SNR) 设置评估全 valid 集，返回 (avg PSNR, 图像数)。"""
    system.mimo = SISOChannel(snr_db=snr_db, fading=fading)

    psnrs: list[torch.Tensor] = []
    n_seen = 0
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))

        autocast_cm = (
            torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )

        with autocast_cm:
            z_sem = system.semantic_encoder(imgs)
            z_ch = system.channel_encoder(z_sem)

        z_ch_f = z_ch.float()
        z_rx, sigma_y, beta = system.mimo.forward(z_ch_f)

        with autocast_cm:
            z_cd = system.channel_decoder(z_rx.to(z_ch.dtype))

        z_cd_f = z_cd.float()
        z_refined = system.ddnm_sample_normalized(
            z_cond=z_cd_f,
            beta=beta.float(),
            sigma_y=sigma_y,
            latent_std=latent_std,
            num_steps=num_sample_steps,
            t_start=ddnm_t_start,
            anchor=ddnm_anchor,
            blend=ddnm_blend,
            repeat_per_step=ddnm_repeat_per_step,
        )

        with autocast_cm:
            x_hat = system.semantic_decoder(z_refined.to(z_ch.dtype))
        x_hat = x_hat.float().clamp(0, 1)

        psnrs.append(psnr_per_image(x_hat, imgs.float()))
        n_seen += imgs.shape[0]

    if not psnrs:
        return float("nan"), 0
    psnr_all = torch.cat(psnrs)
    return float(psnr_all.mean().item()), n_seen


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

    # DIV2K valid loader
    valid_root = os.path.join(args.data_dir, "DIV2K_valid_HR")
    if not os.path.isdir(valid_root):
        valid_root = args.data_dir
    ds = DIV2KDataset(args.data_dir, crop_size=args.crop_size, split="valid")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )
    print(f"DIV2K valid: {len(ds)} 图（root={valid_root}）")

    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}")
    print(f"compression_ratios={args.compression_ratios}, fadings={args.fadings}, "
          f"SNRs(dB)={args.snrs}, ddim_steps={args.num_sample_steps}, "
          f"ddnm_t_start={args.ddnm_t_start}, anchor={args.ddnm_anchor}, "
          f"blend={args.ddnm_blend}, repeat_per_step={args.ddnm_repeat_per_step}")

    # 结果表：{ratio: {(fading, snr): psnr}}
    results: dict[float, dict[tuple[str, float], float]] = {}

    for ratio in args.compression_ratios:
        system, unet_obj, tag = build_system_for_ratio(
            ratio=ratio,
            sc_encoder_ckpt=args.sc_encoder_ckpt,
            sc_decoder_ckpt=args.sc_decoder_ckpt,
            cc_dir=args.cc_dir,
            unet_ckpt=args.unet_ckpt,
            use_ema=not args.no_ema,
            device=device,
        )

        if args.latent_std > 0:
            latent_std = float(args.latent_std)
            print(f"  latent_std (manual) = {latent_std:.6f}")
        else:
            latent_std = float(unet_obj.get("latent_std", 0.0))
            if latent_std <= 0:
                raise ValueError(
                    "unet_ckpt 中没有有效的 latent_std；请通过 --latent_std 显式指定。"
                )
            print(f"  latent_std (from ckpt) = {latent_std:.6f}")

        results[ratio] = {}
        print(f"  --- 扫描设置（ratio={ratio}, tag={tag}） ---")
        for fading in args.fadings:
            for snr in args.snrs:
                avg_psnr, n_seen = evaluate_one_setting(
                    system=system,
                    loader=loader,
                    fading=fading,
                    snr_db=snr,
                    latent_std=latent_std,
                    num_sample_steps=args.num_sample_steps,
                    ddnm_t_start=args.ddnm_t_start,
                    ddnm_anchor=args.ddnm_anchor,
                    ddnm_blend=args.ddnm_blend,
                    ddnm_repeat_per_step=args.ddnm_repeat_per_step,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    max_batches=args.max_batches,
                )
                results[ratio][(fading, snr)] = avg_psnr
                print(f"    [{tag}] fading={fading:<8} SNR={snr:>5.1f} dB | "
                      f"PSNR={avg_psnr:.4f} dB ({n_seen} imgs)")

        del system
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 汇总打印
    print("\n" + "=" * 84)
    print(" 全链路 PSNR 汇总（DIV2K valid，[0,1] 动态范围，逐图后求均值）")
    print("=" * 84)
    header = f"{'ratio':>6} | {'fading':<8} | " + " | ".join(
        f"{snr:>6.1f} dB" for snr in args.snrs
    )
    print(header)
    print("-" * len(header))
    for ratio in args.compression_ratios:
        for fading in args.fadings:
            row = f"{ratio:>6.2f} | {fading:<8} | " + " | ".join(
                f"{results[ratio][(fading, snr)]:>9.4f}" for snr in args.snrs
            )
            print(row)
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
