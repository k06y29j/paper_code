#!/usr/bin/env python3
"""
JSCC + 条件扩散（CHDDIM）验证集 PSNR，流程对齐 ``Diffusion/Train.eval_JSCC_with_CDDM``，
多 **评估信道 SNR** 扫频方式对齐 ``eval_JSCC_with_CDDM_SNRs``（``sampler(..., snr_test, train_snr, ...)``）。

默认假定权重布局与 ``main.py`` / ``test/eval_jscc_norm_feature_stats.py`` 一致：

- ``{jscc_root}/encoder_snr{S}_channel_{awgn|rayleigh}_C{C}.pt``
- ``{jscc_root}/redecoder_snr{S - redecoder_offset}_channel_*_C{C}.pt``（与 main 中 ``SNR-3`` 一致时可设 ``--redecoder-offset 3``）
- ``{cddm_root}/CDDM_snr{S}_channel_*_C{C}.pt``

用法（在 CODM 根目录）::

  python test/eval_jscc_cddm_psnr.py --channel-type awgn --train-snr 12 --C 12 --eval-snrs 50
  python test/eval_jscc_cddm_psnr.py --eval-snrs 0,3,15,50             # 任意多个、逗号分隔 SNR(dB)
  python test/eval_jscc_cddm_psnr.py --sweep  # 训练 SNR {0,6,12} × C {4,12} × 当前 channel-type

若某物理 GPU 存在 ECC/硬件问题：请传 ``--gpu N``（脚本会在 **import torch 之前** 屏蔽其它卡），或在外层执行
``CUDA_VISIBLE_DEVICES=N python ...``。**勿**在已 ``import torch`` 之后再改环境变量，否则会仍可能访问到故障的 0 号卡。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Sequence


def _apply_cuda_visible_devices_before_torch() -> None:
    """让 ``--gpu`` 在 **import torch 之前** 生效；否则 ``CUDA_VISIBLE_DEVICES`` 设得太晚，仍可能用到物理 0 号卡（易触发坏卡 ECC）。"""
    if "--gpu" not in sys.argv:
        return
    i = sys.argv.index("--gpu")
    if i + 1 >= len(sys.argv):
        return
    val = sys.argv[i + 1]
    if val.startswith("-"):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = val


_apply_cuda_visible_devices_before_torch()

import torch
from tqdm import tqdm

_CDDM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CDDM_ROOT not in sys.path:
    sys.path.insert(0, _CDDM_ROOT)

from Autoencoder.data.datasets import get_loader  # noqa: E402
from Autoencoder.net import channel, network  # noqa: E402
from Diffusion import ChannelDiffusionSampler  # noqa: E402
from Diffusion.Model import UNet  # noqa: E402


def default_jscc_snr_ckpt_root() -> str:
    return os.path.join(_CDDM_ROOT, "checkpoints", "JSCC", "DIV2K", "MSE", "SNRs")


def default_cddm_snr_ckpt_root() -> str:
    return os.path.join(_CDDM_ROOT, "checkpoints", "CDDM", "DIV2K", "MSE", "SNRs")


def _snr_ckpt_base(train_snr: int, channel_type: str, C: int) -> str:
    return "snr{}_channel_{}_C{}".format(train_snr, channel_type.lower(), C)


def resolve_encoder_path(ckpt_root: str, train_snr: int, channel_type: str, C: int) -> str:
    p = os.path.join(ckpt_root, "encoder_{}.pt".format(_snr_ckpt_base(train_snr, channel_type, C)))
    if not os.path.isfile(p):
        raise FileNotFoundError("未找到 encoder: {}".format(p))
    return p


def resolve_jscc_decoder_paths(
    ckpt_root: str,
    train_snr: int,
    channel_type: str,
    C: int,
    redecoder_offset: int,
) -> tuple[str, str]:
    """返回 (decoder 路径, **redecoder** · CDDM eval 所用)。decoder 不存在时回退为 redecoder 路径。"""
    base_dec = os.path.join(ckpt_root, "decoder_{}.pt".format(_snr_ckpt_base(train_snr, channel_type, C)))
    rdec_snr = int(train_snr - redecoder_offset)
    base_re = os.path.join(ckpt_root, "redecoder_{}.pt".format(_snr_ckpt_base(rdec_snr, channel_type, C)))
    if not os.path.isfile(base_re):
        raise FileNotFoundError("未找到 redecoder（偏移 {}）：{}".format(redecoder_offset, base_re))
    if not os.path.isfile(base_dec):
        base_dec = base_re
    return base_dec, base_re


def resolve_cddm_unet_path(ckpt_root: str, train_snr: int, channel_type: str, C: int) -> str:
    p = os.path.join(
        ckpt_root,
        "CDDM_{}.pt".format(_snr_ckpt_base(train_snr, channel_type, C)),
    )
    if not os.path.isfile(p):
        raise FileNotFoundError("未找到 CHDDIM U-Net: {}".format(p))
    return p


def parse_int_csv(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("列表不能为空")
    return tuple(int(x) for x in parts)


def parse_float_csv(s: str) -> tuple[float, ...]:
    """逗号分隔的若干 dB SNR（可任意实数）；例如 ``0,3,12,50`` 或 ``-3,0``.

    （仅含一个 SNR 时勿写成括号 ``(... )``——注意必须写逗号分隔或传字符串 ``"50"``。）
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("eval-snrs 列表不能为空（例如：0,3,12）")
    return tuple(float(x) for x in parts)


def normalize_eval_snrs(snrs: object) -> tuple[float, ...]:
    """统一为 SNR 元组；兼容误把默认值写成 ``(50.0)``（结果为 float、不可迭代）的情况。"""
    if isinstance(snrs, (list, tuple)):
        out = tuple(float(x) for x in snrs)
        if not out:
            raise ValueError("eval-snrs 为空")
        return out
    return (float(snrs),)


def build_config_ns(
    *,
    train_snr: int,
    C: int,
    channel_type: str,
    test_data_dir: str,
    encoder_path: str,
    decoder_path: str,
    re_decoder_path: str,
) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.loss_function = "MSE"
    ns.dataset = "DIV2K"
    ns.C = C
    ns.SNRs = float(train_snr)
    ns.seed = 1024
    ns.CUDA = torch.cuda.is_available()
    ns.device = torch.device("cuda:0" if ns.CUDA else "cpu")
    ns.database_address = "mongodb://localhost:27017"
    ns.channel_type = channel_type.lower()
    ns.image_dims = (3, 256, 256)
    ns.encoder_kwargs = dict(
        img_size=(256, 256),
        patch_size=2,
        in_chans=3,
        embed_dims=[128, 192, 256, 320],
        depths=[2, 2, 6, 2],
        num_heads=[4, 6, 8, 10],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=torch.nn.LayerNorm,
        patch_norm=True,
    )
    ns.decoder_kwargs = dict(
        img_size=(256, 256),
        embed_dims=[320, 256, 192, 128],
        depths=[2, 6, 2, 2],
        num_heads=[10, 8, 6, 4],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=torch.nn.LayerNorm,
        patch_norm=True,
    )
    ns.encoder_path = encoder_path
    ns.decoder_path = decoder_path
    ns.re_decoder_path = re_decoder_path
    ns.train_data_dir = test_data_dir
    ns.test_data_dir = test_data_dir
    ns.test_batch = 1
    ns.batch_size = 1
    ns.CDDM_batch = 1
    ns.num_workers = 4
    ns.val_num_workers = 4
    ns.pin_memory = ns.CUDA
    ns.preload_div2k_cpu = False
    ns.load_val_data = True

    return ns


def load_state(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def avg_psnr_jscc_cddm_multi_snr(
    config,
    chddim_dict: dict,
    eval_snrs: Sequence[float],
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """
    对每个 eval_snr（信道 SNR dB）在验证集上算平均 PSNR。
    ``chddim_dict`` 中含 UNet 与各超参以便构建 ``ChannelDiffusionSampler``。
    """
    device = config.device

    encoder = network.JSCC_encoder(config, config.C).to(device)
    decoder = network.JSCC_decoder(config, config.C).to(device)
    encoder.load_state_dict(load_state(config.encoder_path, device))
    decoder.load_state_dict(load_state(config.re_decoder_path, device))
    encoder.eval()
    decoder.eval()

    _, test_loader = get_loader(config)
    assert test_loader is not None

    C = config.C
    unet = UNet(
        T=chddim_dict["T"],
        ch=int(16 * C),
        ch_mult=chddim_dict["channel_mult"],
        attn=chddim_dict["attn"],
        num_res_blocks=chddim_dict["num_res_blocks"],
        dropout=chddim_dict["dropout"],
        input_channel=C,
    ).to(device)
    unet.load_state_dict(load_state(chddim_dict["save_path"], device))
    unet.eval()

    sampler = ChannelDiffusionSampler(
        model=unet,
        noise_schedule=chddim_dict["noise_schedule"],
        t_max=chddim_dict["t_max"],
        beta_1=chddim_dict["snr_max"],
        beta_T=chddim_dict["snr_min"],
        T=chddim_dict["T"],
    ).to(device)

    pass_ch = channel.Channel(config)
    train_snr = float(config.SNRs)

    last_cbr = 0.0
    result: list[tuple[float, float]] = []

    for snr_test in eval_snrs:
        snr_f = float(snr_test)
        sum_psnr = 0.0
        i = -1

        for i, (images, _) in enumerate(
            tqdm(test_loader, dynamic_ncols=True, desc="CDDM SNR_eval={}".format(snr_f))
        ):
            x_0 = images.to(device)
            feature, _ = encoder(x_0)
            y_c = feature
            y_c, pwr, h_p = pass_ch.forward(y_c, snr_f)
            sigma_square = 1.0 / (2 * 10 ** (snr_f / 10))
            last_cbr = feature.numel() / 2 / x_0.numel()

            y_c = y_c / math.sqrt(1 + sigma_square)
            feat_hat = sampler(y_c, snr_f, train_snr, h_p, config.channel_type)

            feat_hat = feat_hat * torch.sqrt(pwr)
            x_hat = decoder(feat_hat)

            mse = torch.nn.MSELoss()(x_0 * 255.0, x_hat.clamp(0.0, 1.0) * 255.0)
            psnr = 10.0 * math.log10(255.0 * 255.0 / max(mse.item(), 1e-18))
            sum_psnr += psnr

        n = max(i + 1, 1)
        result.append((snr_f, sum_psnr / n))

    meta = {"CBR_approx": float(last_cbr)}
    return result, meta


def run_one_setting(
    *,
    train_snr: int,
    C: int,
    channel_type: str,
    jscc_root: str,
    cddm_root: str,
    test_dir: str,
    eval_snrs: Sequence[float],
    redecoder_offset: int,
    large_snr_unused: float,
    noise_schedule: int,
    t_max: int,
) -> tuple[list[tuple[float, float]], argparse.Namespace]:
    _ = large_snr_unused
    encoder_path = resolve_encoder_path(jscc_root, train_snr, channel_type, C)
    _decoder_std, re_decoder_path = resolve_jscc_decoder_paths(
        jscc_root, train_snr, channel_type, C, redecoder_offset
    )
    cddm_path = resolve_cddm_unet_path(cddm_root, train_snr, channel_type, C)

    cfg = build_config_ns(
        train_snr=train_snr,
        C=C,
        channel_type=channel_type,
        test_data_dir=test_dir,
        encoder_path=encoder_path,
        decoder_path=_decoder_std,
        re_decoder_path=re_decoder_path,
    )

    chddim_dict = dict(
        T=1000,
        channel_mult=[1, 2, 2],
        attn=[1],
        num_res_blocks=2,
        dropout=0.1,
        noise_schedule=noise_schedule,
        t_max=int(t_max),
        snr_max=1e-4,
        snr_min=0.02,
        save_path=cddm_path,
    )

    rows, _meta = avg_psnr_jscc_cddm_multi_snr(cfg, chddim_dict, eval_snrs)
    return rows, cfg


def print_result_table(
    train_snr: int,
    C: int,
    channel_type: str,
    rows: list[tuple[float, float]],
) -> None:
    print(
        "\n=== train_SNR {} dB · C {} · {} ===".format(train_snr, C, channel_type)
    )
    print("eval SNR(dB):" + "".join("{:>8}".format(r[0]) for r in rows))
    vals = "".join("{:>8.3f}".format(r[1]) for r in rows)
    print("PSNR (dB):   " + vals)


def append_csv(csv_path: str, row_fields: Sequence) -> None:
    new = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["train_snr", "C", "channel", "eval_snr", "PSNR"])
        w.writerow(list(row_fields))


def main() -> None:
    p = argparse.ArgumentParser(description="DIV2K：JSCC+CHDDIM 多评估 SNR PSNR（对齐 Train.eval_JSCC_with_CDDM / _SNRs）")
    p.add_argument("--train-snr", type=int, default=None, help="当前权重对应的训练 SNR（文件名 snr{S}_...）；--sweep 时忽略")
    p.add_argument("--C", type=int, default=None)
    p.add_argument("--channel-type", type=str, default="awgn", choices=("awgn", "rayleigh"))
    p.add_argument(
        "--eval-snrs",
        type=parse_float_csv,
        default=(50.0),
        help=(
            "评估信道 SNR（dB，逗号分隔，可任意多个实数）；例：--eval-snrs 0,3,50 或单一值 --eval-snrs 50（请用字符串）"
        ),
    )
    p.add_argument("--train-snrs-sweep", type=parse_int_csv, default=(0, 6, 12))
    p.add_argument("--c-sweep", type=parse_int_csv, default=(4, 12))
    p.add_argument("--sweep", action="store_true", help="对 train-snrs-sweep × c-sweep × channel-type 全组合评估")
    p.add_argument("--jscc-root", type=str, default=default_jscc_snr_ckpt_root())
    p.add_argument("--cddm-root", type=str, default=default_cddm_snr_ckpt_root())
    p.add_argument(
        "--test-dir",
        type=str,
        default="/workspace/yongjia/datasets/DIV2K/DIV2K_valid_HR/",
    )
    p.add_argument(
        "--redecoder-offset",
        type=int,
        default=3,
        help="redecoder 文件名中的 SNR = train_snr - offset（与 main.py 约定一致默认为 3）",
    )
    p.add_argument("--large-snr", type=float, default=3.0, help="仅保留接口；sampler 沿用 train_SNR，与原版 eval 等价")
    p.add_argument("--noise-schedule", type=int, default=1)
    p.add_argument("--t-max", type=int, default=10)
    p.add_argument("--gpu", type=str, default=None, help="见文首说明：须在 import torch 之前生效；本脚本已据此在启动时解析 argv 并设置 CUDA_VISIBLE_DEVICES（如 3）")
    p.add_argument("--csv-out", type=str, default=None, help="追加写入 CSV：train_snr,C,channel,eval_snr,PSNR")

    args = p.parse_args()
    eval_snrs = normalize_eval_snrs(args.eval_snrs)

    if args.sweep:
        jobs = [(ts, cc) for ts in args.train_snrs_sweep for cc in args.c_sweep]
    else:
        if args.train_snr is None or args.C is None:
            raise SystemExit("--train-snr 与 --C 为必填（或改用 --sweep）")
        jobs = [(args.train_snr, args.C)]

    for ts, Cc in jobs:
        try:
            rows, _cfg = run_one_setting(
                train_snr=ts,
                C=Cc,
                channel_type=args.channel_type,
                jscc_root=args.jscc_root,
                cddm_root=args.cddm_root,
                test_dir=args.test_dir,
                eval_snrs=eval_snrs,
                redecoder_offset=args.redecoder_offset,
                large_snr_unused=args.large_snr,
                noise_schedule=args.noise_schedule,
                t_max=args.t_max,
            )
        except FileNotFoundError as e:
            print("[跳过 train_snr={} C={}] {}".format(ts, Cc, e))
            continue
        print_result_table(ts, Cc, args.channel_type, rows)
        if args.csv_out:
            for snr_ev, pv in rows:
                append_csv(
                    args.csv_out,
                    (ts, Cc, args.channel_type, snr_ev, "{:.6f}".format(pv)),
                )


if __name__ == "__main__":
    main()
