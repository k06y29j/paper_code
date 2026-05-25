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
      --cc_dir         /workspace/yongjia/paper_code/checkpoints-val/cc/aware_awgn_snr12_L1\
      --unet_ckpt      checkpoints-val/unet_un/unet_un_div2k_c16_best.pth \
      --compression_ratios 0.75 --fadings awgn  \
      --snrs 0 3 6 9 12 15 \
      --num_sample_steps 30 --ddnm_t_start 0 \
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
    Residual1x1Codec,
    ResidualSpatialCodec,
)
from src.cddm_mimo_ddnm.modules.siso_channel import SISOChannel  # noqa: E402


UNET_MODEL_PRESETS = {
    "A1_channel_vanilla": "checkpoints-val/unet_un/A1_channel_vanilla/unet_un_div2k_c16_best.pth",
    "A2_channel_minsnr5": "checkpoints-val/unet_un/A2_channel_minsnr5/unet_un_div2k_c16_best.pth",
    "A2_channel_minsnr5_cosine": "checkpoints-val/unet_un/A2_channel_minsnr5_cosine/unet_un_div2k_c16_best.pth",
}


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LatentResidualRefiner(nn.Module):
    def __init__(self, channels: int = 16, hidden: int = 96, depth: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
        ]
        for _ in range(depth):
            layers.append(ResBlock(hidden))
        layers.extend([
            nn.GELU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        ])
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.res_scale.to(dtype=z.dtype) * self.net(z)


class LatentUNetRefiner(nn.Module):
    def __init__(self, channels: int = 16, base: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, base, 3, padding=1),
            nn.GELU(),
        )
        self.enc0 = nn.Sequential(*[ResBlock(base) for _ in range(depth)])
        self.down1 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.enc1 = nn.Sequential(*[ResBlock(base * 2) for _ in range(depth)])
        self.down2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.mid = nn.Sequential(*[ResBlock(base * 4) for _ in range(max(1, depth + 1))])
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.GELU(),
            *[ResBlock(base * 2) for _ in range(depth)],
        )
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.fuse0 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.GELU(),
            *[ResBlock(base) for _ in range(depth)],
        )
        self.out = nn.Conv2d(base, channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x0 = self.enc0(self.stem(z))
        x1 = self.enc1(self.down1(x0))
        xm = self.mid(self.down2(x1))
        y1 = self.up1(xm)
        if y1.shape[-2:] != x1.shape[-2:]:
            y1 = torch.nn.functional.interpolate(y1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.fuse1(torch.cat([y1, x1], dim=1))
        y0 = self.up0(y1)
        if y0.shape[-2:] != x0.shape[-2:]:
            y0 = torch.nn.functional.interpolate(y0, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y0 = self.fuse0(torch.cat([y0, x0], dim=1))
        return z + self.res_scale.to(dtype=z.dtype) * self.out(y0)


class LatentNullPredictor(nn.Module):
    def __init__(
        self,
        channels: int = 16,
        hidden: int = 64,
        depth: int = 4,
        a_matrix: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
        ]
        for _ in range(depth):
            layers.append(ResBlock(hidden))
        layers.extend([
            nn.GELU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        ])
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        if a_matrix is None:
            a_matrix = torch.eye(4, channels, dtype=torch.float32)
        self.register_buffer("a_matrix", a_matrix.float(), persistent=True)

    def _null_project(self, z: torch.Tensor) -> torch.Tensor:
        a = self.a_matrix.to(device=z.device, dtype=z.dtype)
        az = torch.einsum("oc,bchw->bohw", a, z)
        low = torch.einsum("oc,bohw->bchw", a, az)
        return z - low

    def forward(self, z_low: torch.Tensor) -> torch.Tensor:
        pred_null = self._null_project(self.net(z_low))
        return z_low + self.res_scale.to(dtype=z_low.dtype) * pred_null


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
    p.add_argument("--unet_model", type=str, default=None,
                   choices=sorted(UNET_MODEL_PRESETS.keys()),
                   help="快捷选择 checkpoints-val/unet_un 下的三个通道归一化扩散模型；设置后覆盖 --unet_ckpt")
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
    p.add_argument("--ddnm_ensemble", type=int, default=1,
                   help="每个接收 latent 重复随机 DDNM 采样 N 次并平均；1 表示原始行为")
    p.add_argument("--ddnm_ensemble_mode", type=str, default="latent", choices=["latent", "image"],
                   help="latent: 平均 refined latent 后解码；image: 解码后平均图像")
    p.add_argument("--ddnm_observation", type=str, default="zcd", choices=["zcd", "rx"],
                   help="DDNM 一致性空间：zcd 使用 W_dec W_enc；rx 使用 W_enc 和信道接收 z_rx。")
    p.add_argument("--ddnm_ridge", type=float, default=0.0,
                   help="线性一致性修正的 ridge 系数；0 为 Moore-Penrose pinv。实际 ridge 会乘 sigma_y_norm^2。")
    p.add_argument("--sampler", type=str, default="ddnm", choices=["ddnm", "route_a"],
                   help="采样方式：ddnm 为原 DDNM+；route_a 为 rule.md 的零空间 + 维纳子空间 CDDM 采样。")
    p.add_argument("--route_a_t_start", type=int, default=None,
                   help="Route-A 专用 t_start；未设置时沿用 --ddnm_t_start")
    p.add_argument("--route_a_blend", type=float, default=None,
                   help="Route-A 专用最终融合系数；未设置时沿用 --ddnm_blend")
    p.add_argument("--route_a_keep_null", type=float, default=1.0,
                   help="Route-A 最终零空间保留系数；1 保留 U-Net 高频，0 只保留低频子空间")
    p.add_argument("--route_a_final_wiener", type=float, default=1.0,
                   help="Route-A 最终静态 Wiener 低频融合强度；1 全启用，0 关闭")

    # latent normalization
    p.add_argument("--latent_std", type=float, default=0.0,
                   help="仅用于旧版 scalar latent scaling；通道归一化 checkpoint 会优先使用自身 mean/std metadata")
    p.add_argument("--latent_norm_stats", type=str, default="",
                   help="可选：覆盖 U-Net checkpoint 的 latent mean/std 统计（torch 保存的 dict）。")
    p.add_argument("--latent_refiner_ckpt", type=str, default="",
                   help="可选：信道解码 latent 残差 refiner 权重；为空则保持原始 eval_all 行为")
    p.add_argument("--latent_refiner_hidden", type=int, default=0,
                   help="refiner hidden channels；0 表示从 checkpoint metadata 读取")
    p.add_argument("--latent_refiner_depth", type=int, default=0,
                   help="refiner residual depth；0 表示从 checkpoint metadata 读取")
    p.add_argument("--latent_refiner_apply", type=str, default="post_channel",
                   choices=["post_channel", "post_ddnm"],
                   help="post_channel: DDNM 前修正 z_cd；post_ddnm: DDNM 后再修正")
    p.add_argument("--latent_refiner_blend", type=float, default=1.0,
                   help="refined = blend*refiner(z) + (1-blend)*z；用于扫接收端去噪强度")
    p.add_argument("--rx_scale", type=float, default=1.0,
                   help="测试时缩放信道输出 z_rx 后再进信道解码器；1.0 表示不变")
    p.add_argument("--zcd_scale", type=float, default=1.0,
                   help="测试时缩放信道解码 latent z_cd 后再进 DDNM/语义解码；1.0 表示不变")

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


def resolve_unet_ckpt(args: argparse.Namespace) -> str:
    """Resolve --unet_model / --unet_ckpt; allow passing a checkpoint directory."""
    if args.unet_model:
        rel = UNET_MODEL_PRESETS[args.unet_model]
        return os.path.join(PROJECT_ROOT, rel)
    ckpt = args.unet_ckpt
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, "unet_un_div2k_c16_best.pth")
    return ckpt


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
        raw_k = k[4:] if k.startswith("net.") else k
        if isinstance(getattr(net, "net", None), (Residual1x1Codec, ResidualSpatialCodec)):
            if raw_k == "weight":
                new_sd["net.base.weight"] = v
            elif raw_k.startswith(("base.", "res.")) or raw_k == "res_scale":
                new_sd[f"net.{raw_k}"] = v
            else:
                new_sd[k if k.startswith("net.") else f"net.{k}"] = v
        else:
            new_sd[k if k.startswith("net.") else f"net.{k}"] = v
    missing, unexpected = net.load_state_dict(new_sd, strict=False)
    print(f"  [{name}] 载入 {ckpt_path}")
    if isinstance(obj, dict):
        meta = {k: obj[k] for k in (
            "in_channels", "out_channels", "compression_ratio",
            "linear_depth", "linear_hidden_channels", "codec_mode",
            "residual_hidden_channels", "codec_init",
            "lambda_enc_orth", "cov_max_batches", "epoch", "metrics"
        ) if k in obj}
        if meta:
            print(f"    meta: {meta}")
    if missing or unexpected:
        print(f"    missing={len(missing)}, unexpected={len(unexpected)}")
    return obj


def peek_cc_arch(ckpt_path: str) -> tuple[int, int, str, int]:
    """从 CC checkpoint 读取深线性结构；旧权重缺省为 L=1, hidden=16。"""
    if not os.path.isfile(ckpt_path):
        return 1, 16, "linear", 0
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        return 1, 16, "linear", 0
    depth = int(obj.get("linear_depth", 1))
    hidden = int(obj.get("linear_hidden_channels", 16))
    codec_mode = str(obj.get("codec_mode", "linear"))
    residual_hidden = int(obj.get("residual_hidden_channels", 0))
    return depth, hidden, codec_mode, residual_hidden


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


def load_latent_refiner(
    ckpt_path: str,
    hidden_override: int,
    depth_override: int,
    device: torch.device,
) -> LatentResidualRefiner:
    """Load optional latent residual refiner trained for the C=12 AWGN12 link."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"latent_refiner 权重不存在：{ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = hidden_override or int(obj.get("refiner_hidden", 96))
    depth = depth_override or int(obj.get("refiner_depth", 6))
    refiner_type = str(obj.get("refiner_type", "resnet"))
    if refiner_type == "null_predictor":
        a_matrix = obj.get("a_matrix", None)
        if a_matrix is not None:
            a_matrix = torch.as_tensor(a_matrix, dtype=torch.float32)
        refiner = LatentNullPredictor(16, hidden=hidden, depth=depth, a_matrix=a_matrix).to(device)
    elif refiner_type == "unet":
        refiner = LatentUNetRefiner(16, base=hidden, depth=depth).to(device)
    else:
        refiner = LatentResidualRefiner(16, hidden=hidden, depth=depth).to(device)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    missing, unexpected = refiner.load_state_dict(sd, strict=False)
    refiner.eval()
    print(f"  [latent_refiner] 载入 {ckpt_path}")
    print(f"    arch: type={refiner_type}, hidden={hidden}, depth={depth}")
    if isinstance(obj, dict) and "metrics" in obj:
        print(f"    metrics: {obj['metrics']}")
    if missing or unexpected:
        print(f"    missing={len(missing)}, unexpected={len(unexpected)}")
    return refiner


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
    linear_depth, linear_hidden_channels, codec_mode, residual_hidden_channels = peek_cc_arch(cc_enc_path)
    residual_hidden_channels = int(residual_hidden_channels or linear_hidden_channels)

    # 用户提供的 cc 权重操作的是 sc encoder.head 投影后的 16 通道 latent（与 sc embed_dim 对齐），
    # 而 SemanticCommSystem 默认按 ``semantic_encoder.latent_dim``（最后一级 stage dim，例如 320）
    # 构造 channel_encoder。这里显式重建为 in_channels = sc embed_dim。
    system.channel_encoder = ChannelEncoder(
        in_channels=in_ch,
        out_channels=out_ch,
        bottleneck_dim=None,
        linear_depth=linear_depth,
        hidden_channels=linear_hidden_channels,
        codec_mode=codec_mode,
        residual_hidden_channels=residual_hidden_channels,
    ).to(device)
    system.channel_decoder = ChannelDecoder(
        in_channels=out_ch,
        out_channels=in_ch,
        bottleneck_dim=None,
        linear_depth=linear_depth,
        hidden_channels=linear_hidden_channels,
        codec_mode=codec_mode,
        residual_hidden_channels=residual_hidden_channels,
    ).to(device)
    print(
        f"  [cc arch] mode={codec_mode}, linear_depth={linear_depth}, "
        f"hidden={linear_hidden_channels}, residual_hidden={residual_hidden_channels}"
    )

    load_cc_weight(system.channel_encoder, cc_enc_path, f"cc_encoder({tag})")
    load_cc_weight(system.channel_decoder, cc_dec_path, f"cc_decoder({tag})")

    unet_obj = load_unet(system.unet_denoiser, unet_ckpt, use_ema=use_ema)
    system.cfg.apply_unet_checkpoint_metadata(unet_obj)
    system.refresh_diffusion_schedule()

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
    latent_refiner: nn.Module | None,
    loader: DataLoader,
    fading: str,
    snr_db: float,
    latent_std: float,
    latent_mean,
    latent_channel_std,
    num_sample_steps: int,
    ddnm_t_start: int,
    ddnm_anchor: str,
    ddnm_blend: float,
    ddnm_repeat_per_step: int,
    ddnm_ensemble: int,
    ddnm_ensemble_mode: str,
    ddnm_observation: str,
    ddnm_ridge: float,
    sampler: str,
    route_a_t_start: int | None,
    route_a_blend: float | None,
    route_a_keep_null: float,
    route_a_final_wiener: float,
    latent_refiner_apply: str,
    latent_refiner_blend: float,
    rx_scale: float,
    zcd_scale: float,
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
        if rx_scale != 1.0:
            z_rx = z_rx * float(rx_scale)

        with autocast_cm:
            z_cd = system.channel_decoder(z_rx.to(z_ch.dtype))
        if zcd_scale != 1.0:
            z_cd = z_cd * float(zcd_scale)

        z_cd_f = z_cd.float()
        z_cond = z_cd_f
        if latent_refiner is not None and latent_refiner_apply == "post_channel":
            with autocast_cm:
                z_ref = latent_refiner(z_cd.to(next(latent_refiner.parameters()).dtype)).float()
                z_cond = float(latent_refiner_blend) * z_ref + (1.0 - float(latent_refiner_blend)) * z_cd_f

        n_ens = max(1, int(ddnm_ensemble))
        if n_ens == 1 or ddnm_ensemble_mode == "latent":
            z_acc = None
            for _ in range(n_ens):
                if sampler == "route_a":
                    t_start_eff = ddnm_t_start if route_a_t_start is None else int(route_a_t_start)
                    blend_eff = ddnm_blend if route_a_blend is None else float(route_a_blend)
                    z_one = system.route_a_wiener_sample_normalized(
                        z_anchor=z_cond,
                        z_rx=z_rx.float(),
                        beta=beta.float(),
                        sigma_y=sigma_y,
                        latent_std=latent_std,
                        latent_mean=latent_mean,
                        latent_channel_std=latent_channel_std,
                        num_steps=num_sample_steps,
                        t_start=t_start_eff,
                        blend=blend_eff,
                        keep_null_space=route_a_keep_null,
                        final_wiener=route_a_final_wiener,
                    )
                elif ddnm_observation == "rx":
                    z_one = system.ddnm_sample_rx_normalized(
                        z_anchor=z_cond,
                        z_rx=z_rx.float(),
                        beta=beta.float(),
                        sigma_y=sigma_y,
                        latent_std=latent_std,
                        latent_mean=latent_mean,
                        latent_channel_std=latent_channel_std,
                        num_steps=num_sample_steps,
                        t_start=ddnm_t_start,
                        anchor=ddnm_anchor,
                        blend=ddnm_blend,
                        repeat_per_step=ddnm_repeat_per_step,
                        ridge=ddnm_ridge,
                    )
                else:
                    z_one = system.ddnm_sample_normalized(
                        z_cond=z_cond,
                        beta=beta.float(),
                        sigma_y=sigma_y,
                        latent_std=latent_std,
                        latent_mean=latent_mean,
                        latent_channel_std=latent_channel_std,
                        num_steps=num_sample_steps,
                        t_start=ddnm_t_start,
                        anchor=ddnm_anchor,
                        blend=ddnm_blend,
                        repeat_per_step=ddnm_repeat_per_step,
                        ridge=ddnm_ridge,
                    )
                if latent_refiner is not None and latent_refiner_apply == "post_ddnm":
                    with autocast_cm:
                        z_ref = latent_refiner(z_one.to(next(latent_refiner.parameters()).dtype)).float()
                        z_one = float(latent_refiner_blend) * z_ref + (1.0 - float(latent_refiner_blend)) * z_one
                z_acc = z_one if z_acc is None else z_acc + z_one
            z_refined = z_acc / float(n_ens)
            with autocast_cm:
                x_hat = system.semantic_decoder(z_refined.to(z_ch.dtype))
        else:
            x_acc = None
            for _ in range(n_ens):
                if sampler == "route_a":
                    t_start_eff = ddnm_t_start if route_a_t_start is None else int(route_a_t_start)
                    blend_eff = ddnm_blend if route_a_blend is None else float(route_a_blend)
                    z_one = system.route_a_wiener_sample_normalized(
                        z_anchor=z_cond,
                        z_rx=z_rx.float(),
                        beta=beta.float(),
                        sigma_y=sigma_y,
                        latent_std=latent_std,
                        latent_mean=latent_mean,
                        latent_channel_std=latent_channel_std,
                        num_steps=num_sample_steps,
                        t_start=t_start_eff,
                        blend=blend_eff,
                        keep_null_space=route_a_keep_null,
                        final_wiener=route_a_final_wiener,
                    )
                elif ddnm_observation == "rx":
                    z_one = system.ddnm_sample_rx_normalized(
                        z_anchor=z_cond,
                        z_rx=z_rx.float(),
                        beta=beta.float(),
                        sigma_y=sigma_y,
                        latent_std=latent_std,
                        latent_mean=latent_mean,
                        latent_channel_std=latent_channel_std,
                        num_steps=num_sample_steps,
                        t_start=ddnm_t_start,
                        anchor=ddnm_anchor,
                        blend=ddnm_blend,
                        repeat_per_step=ddnm_repeat_per_step,
                        ridge=ddnm_ridge,
                    )
                else:
                    z_one = system.ddnm_sample_normalized(
                        z_cond=z_cond,
                        beta=beta.float(),
                        sigma_y=sigma_y,
                        latent_std=latent_std,
                        latent_mean=latent_mean,
                        latent_channel_std=latent_channel_std,
                        num_steps=num_sample_steps,
                        t_start=ddnm_t_start,
                        anchor=ddnm_anchor,
                        blend=ddnm_blend,
                        repeat_per_step=ddnm_repeat_per_step,
                        ridge=ddnm_ridge,
                    )
                if latent_refiner is not None and latent_refiner_apply == "post_ddnm":
                    with autocast_cm:
                        z_ref = latent_refiner(z_one.to(next(latent_refiner.parameters()).dtype)).float()
                        z_one = float(latent_refiner_blend) * z_ref + (1.0 - float(latent_refiner_blend)) * z_one
                with autocast_cm:
                    x_one = system.semantic_decoder(z_one.to(z_ch.dtype)).float()
                x_acc = x_one if x_acc is None else x_acc + x_one
            x_hat = x_acc / float(n_ens)
        x_hat = x_hat.float().clamp(0, 1)

        psnrs.append(psnr_per_image(x_hat, imgs.float()))
        n_seen += imgs.shape[0]

    if not psnrs:
        return float("nan"), 0
    psnr_all = torch.cat(psnrs)
    return float(psnr_all.mean().item()), n_seen


def main() -> None:
    args = parse_args()
    args.unet_ckpt = resolve_unet_ckpt(args)
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
    if args.unet_model:
        print(f"unet_model={args.unet_model}")
    print(f"unet_ckpt={args.unet_ckpt}")
    print(f"compression_ratios={args.compression_ratios}, fadings={args.fadings}, "
          f"SNRs(dB)={args.snrs}, ddim_steps={args.num_sample_steps}, "
          f"ddnm_t_start={args.ddnm_t_start}, anchor={args.ddnm_anchor}, "
          f"blend={args.ddnm_blend}, repeat_per_step={args.ddnm_repeat_per_step}, "
          f"ensemble={args.ddnm_ensemble}/{args.ddnm_ensemble_mode}, "
          f"sampler={args.sampler}, "
          f"observation={args.ddnm_observation}, ridge={args.ddnm_ridge:g}, "
          f"rx_scale={args.rx_scale:g}, zcd_scale={args.zcd_scale:g}")
    if args.sampler == "route_a":
        print(
            "route_a: "
            f"t_start={args.route_a_t_start if args.route_a_t_start is not None else args.ddnm_t_start}, "
            f"blend={args.route_a_blend if args.route_a_blend is not None else args.ddnm_blend}, "
            f"keep_null={args.route_a_keep_null:g}, "
            f"final_wiener={args.route_a_final_wiener:g}"
        )
    latent_refiner = None
    if args.latent_refiner_ckpt:
        latent_refiner = load_latent_refiner(
            args.latent_refiner_ckpt,
            args.latent_refiner_hidden,
            args.latent_refiner_depth,
            device,
        )
        print(f"latent_refiner_apply={args.latent_refiner_apply}, blend={args.latent_refiner_blend}")

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

        latent_mean = system.cfg.diffusion.latent_mean
        latent_channel_std = system.cfg.diffusion.latent_std_channels
        if args.latent_norm_stats:
            stats_obj = torch.load(args.latent_norm_stats, map_location="cpu", weights_only=False)
            latent_mean = stats_obj.get("latent_mean", stats_obj.get("mean", latent_mean))
            latent_channel_std = stats_obj.get("latent_std_channels", stats_obj.get("std", latent_channel_std))
            if hasattr(latent_mean, "tolist"):
                latent_mean = latent_mean.tolist()
            if hasattr(latent_channel_std, "tolist"):
                latent_channel_std = latent_channel_std.tolist()
            system.cfg.diffusion.latent_mean = latent_mean
            system.cfg.diffusion.latent_std_channels = latent_channel_std
            if latent_channel_std is not None:
                system.cfg.diffusion.latent_std = float(torch.as_tensor(latent_channel_std).mean().item())
            print(f"  latent_norm override: {args.latent_norm_stats}")
        if latent_channel_std is not None:
            latent_std = float(system.cfg.diffusion.latent_std)
            mean = latent_mean or [0.0 for _ in latent_channel_std]
            std = latent_channel_std
            print(
                "  latent_norm (from ckpt) = channel  "
                f"mean_range=[{min(mean):.6f},{max(mean):.6f}]  "
                f"std_range=[{min(std):.6f},{max(std):.6f}]  "
                f"legacy_std={latent_std:.6f}"
            )
        elif args.latent_std > 0:
            latent_std = float(args.latent_std)
            print(f"  latent_norm (manual) = scalar  std={latent_std:.6f}")
        else:
            latent_std = float(unet_obj.get("latent_std", system.cfg.diffusion.latent_std))
            if latent_std <= 0:
                raise ValueError(
                    "unet_ckpt 中没有有效的 scalar latent_std；请通过 --latent_std 显式指定。"
                )
            print(f"  latent_norm (from ckpt) = scalar  std={latent_std:.6f}")
        print(f"  diffusion.noise_schedule = {system.cfg.diffusion.noise_schedule}")

        results[ratio] = {}
        print(f"  --- 扫描设置（ratio={ratio}, tag={tag}） ---")
        for fading in args.fadings:
            for snr in args.snrs:
                avg_psnr, n_seen = evaluate_one_setting(
                    system=system,
                    latent_refiner=latent_refiner,
                    loader=loader,
                    fading=fading,
                    snr_db=snr,
                    latent_std=latent_std,
                    latent_mean=latent_mean,
                    latent_channel_std=latent_channel_std,
                    num_sample_steps=args.num_sample_steps,
                    ddnm_t_start=args.ddnm_t_start,
                    ddnm_anchor=args.ddnm_anchor,
                    ddnm_blend=args.ddnm_blend,
                    ddnm_repeat_per_step=args.ddnm_repeat_per_step,
                    ddnm_ensemble=args.ddnm_ensemble,
                    ddnm_ensemble_mode=args.ddnm_ensemble_mode,
                    ddnm_observation=args.ddnm_observation,
                    ddnm_ridge=args.ddnm_ridge,
                    sampler=args.sampler,
                    route_a_t_start=args.route_a_t_start,
                    route_a_blend=args.route_a_blend,
                    route_a_keep_null=args.route_a_keep_null,
                    route_a_final_wiener=args.route_a_final_wiener,
                    latent_refiner_apply=args.latent_refiner_apply,
                    latent_refiner_blend=args.latent_refiner_blend,
                    rx_scale=args.rx_scale,
                    zcd_scale=args.zcd_scale,
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
