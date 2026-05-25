#!/usr/bin/env python3
"""
在 DIV2K 验证集上评估 JSCC（或 JSCC+CDDM）重建的 **LPIPS** 与 **FID**。

- **默认（JSCC）**：encoder + decoder，多评估信道 SNR 扫频；对齐 ``test/eval_jscc_norm_feature_stats.py``。
- **`--cddm`**：JSCC + 条件扩散（CHDDIM）；对齐 ``test/eval_jscc_cddm_psnr.py``。

LPIPS：原图与重建图逐样本 LPIPS 的验证集平均（越低越好，VGG backbone）。
FID：原图集合 vs 重建图集合的 Fréchet Inception Distance（越低越好；Inception 权重与 pytorch-fid 一致）。

用法（在 CDDM 根目录）::

  python test/eval_jscc_lpips_fid.py --train-snr 12 --channel-type awgn --C 12
  python test/eval_jscc_lpips_fid.py --train-snr 12 --C 12 --eval-snrs 0,3,12,50
  python test/eval_jscc_lpips_fid.py --cddm --train-snr 12 --C 12 --eval-snrs 12
  CUDA_VISIBLE_DEVICES=2 python test/eval_jscc_lpips_fid.py --gpu 2 --train-snr 0 --C 4

若某物理 GPU 存在 ECC/硬件问题：请传 ``--gpu N``（须在 import torch 之前生效），或外层
``CUDA_VISIBLE_DEVICES=N python ...``。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Sequence


def _apply_cuda_visible_devices_before_torch() -> None:
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

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

_CDDM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CDDM_ROOT not in sys.path:
    sys.path.insert(0, _CDDM_ROOT)
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
_FID_DIR = os.path.join(_CDDM_ROOT, "open-master", "open-master", "FID")
if _FID_DIR not in sys.path:
    sys.path.insert(0, _FID_DIR)

from Autoencoder.data.datasets import FlatImageFolder, get_loader  # noqa: E402
from Autoencoder.net import channel, network  # noqa: E402
from Autoencoder.net.channel import Channel  # noqa: E402
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder  # noqa: E402
from Diffusion import ChannelDiffusionSampler  # noqa: E402
from Diffusion.LPIPS import util_of_lpips  # noqa: E402
from Diffusion.Model import UNet  # noqa: E402
from eval_jscc_norm_feature_stats import (  # noqa: E402
    DEFAULT_EVAL_SNRS,
    build_eval_namespace,
    default_jscc_snr_ckpt_root,
    feature_to_decoder_input_noisy,
    parse_eval_snrs,
    resolve_jscc_snr_checkpoints,
)
from inception import InceptionV3  # noqa: E402

DEFAULT_CDDM_CKPT_ROOT = os.path.join(_CDDM_ROOT, "checkpoints", "CDDM", "DIV2K", "MSE", "SNRs")
FID_DIMS = 2048


def _snr_ckpt_base(train_snr: int, channel_type: str, C: int) -> str:
    return f"snr{train_snr}_channel_{channel_type.lower()}_C{C}"


def resolve_cddm_unet_path(ckpt_root: str, train_snr: int, channel_type: str, C: int) -> str:
    p = os.path.join(ckpt_root, f"CDDM_{_snr_ckpt_base(train_snr, channel_type, C)}.pt")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"未找到 CHDDIM U-Net: {p}")
    return p


def resolve_jscc_decoder_paths_cddm(
    ckpt_root: str,
    train_snr: int,
    channel_type: str,
    C: int,
    redecoder_offset: int,
) -> tuple[str, str]:
    base_dec = os.path.join(ckpt_root, f"decoder_{_snr_ckpt_base(train_snr, channel_type, C)}.pt")
    rdec_snr = int(train_snr - redecoder_offset)
    base_re = os.path.join(
        ckpt_root,
        f"redecoder_{_snr_ckpt_base(rdec_snr, channel_type, C)}.pt",
    )
    if not os.path.isfile(base_re):
        raise FileNotFoundError(f"未找到 redecoder（偏移 {redecoder_offset}）：{base_re}")
    if not os.path.isfile(base_dec):
        base_dec = base_re
    return base_dec, base_re


def load_state(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device, weights_only=False)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"FID 协方差平方根含虚部: {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


@torch.inference_mode()
def inception_activations(
    images: torch.Tensor,
    inception_model: InceptionV3,
) -> np.ndarray:
    """``images`` 为 [0, 1] 的 B×3×H×W float tensor，返回 (B, 2048) numpy。"""
    pred = inception_model(images)[0]
    if pred.shape[2] != 1 or pred.shape[3] != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    return pred.squeeze(-1).squeeze(-1).cpu().numpy()


def activation_stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def to_lpips_tensor(x: torch.Tensor) -> torch.Tensor:
    """[0, 1] → [-1, 1]（LPIPS 约定）。"""
    return x.clamp(0.0, 1.0) * 2.0 - 1.0


@torch.inference_mode()
def jscc_reconstruct(
    img: torch.Tensor,
    encoder: JSCC_encoder,
    decoder: JSCC_decoder,
    pass_ch: Channel,
    eval_snr_db: float,
    channel_type: str,
    *,
    use_amp: bool,
) -> torch.Tensor:
    with torch.autocast(
        device_type="cuda" if img.is_cuda else "cpu",
        dtype=torch.float16,
        enabled=use_amp and img.is_cuda,
    ):
        feature, _ = encoder(img)
        dec_in = feature_to_decoder_input_noisy(feature, pass_ch, eval_snr_db, channel_type)
        recon = decoder(dec_in)
    return recon.clamp(0.0, 1.0)


@torch.inference_mode()
def cddm_reconstruct(
    img: torch.Tensor,
    encoder: network.JSCC_encoder,
    decoder: network.JSCC_decoder,
    pass_ch: channel.Channel,
    sampler: ChannelDiffusionSampler,
    eval_snr_db: float,
    train_snr_db: float,
    channel_type: str,
) -> torch.Tensor:
    feature, _ = encoder(img)
    y_c, pwr, h_p = pass_ch.forward(feature, eval_snr_db)
    sigma_square = 1.0 / (2 * 10 ** (eval_snr_db / 10))
    y_c = y_c / math.sqrt(1 + sigma_square)
    feat_hat = sampler(y_c, eval_snr_db, train_snr_db, h_p, channel_type)
    feat_hat = feat_hat * torch.sqrt(pwr)
    x_hat = decoder(feat_hat)
    return x_hat.clamp(0.0, 1.0)


@torch.inference_mode()
def eval_lpips_fid_jscc(
    cfg,
    loader: DataLoader,
    eval_snrs: Sequence[float],
    *,
    lpips_net: str,
    use_amp: bool,
    metrics: set[str],
) -> list[tuple[float, float | None, float | None]]:
    dev = cfg.device
    nb = dev.type == "cuda"
    use_amp_b = bool(use_amp) and dev.type == "cuda"
    ch_type = cfg.channel_type

    encoder = JSCC_encoder(cfg, cfg.C).to(dev)
    decoder = JSCC_decoder(cfg, cfg.C).to(dev)
    pass_ch = Channel(cfg)
    encoder.load_state_dict(load_state(cfg.encoder_path, dev))
    decoder.load_state_dict(load_state(cfg.decoder_path, dev))
    encoder.eval()
    decoder.eval()

    lpips_fn = None
    if "lpips" in metrics:
        lpips_fn = util_of_lpips(lpips_net, use_gpu=nb)
        lpips_fn.loss_fn.to(dev).eval()

    inception_model = None
    if "fid" in metrics:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIMS]
        inception_model = InceptionV3([block_idx]).to(dev)
        inception_model.eval()

    results: list[tuple[float, float | None, float | None]] = []

    for eval_snr in eval_snrs:
        sum_lpips = 0.0
        seen = 0
        real_feats: list[np.ndarray] = []
        fake_feats: list[np.ndarray] = []

        desc = f"JSCC SNR={eval_snr:g} dB"
        with tqdm(loader, desc=desc, dynamic_ncols=True) as tl:
            for img, _ in tl:
                img = img.to(dev, non_blocking=nb)
                recon = jscc_reconstruct(
                    img,
                    encoder,
                    decoder,
                    pass_ch,
                    float(eval_snr),
                    ch_type,
                    use_amp=use_amp_b,
                )

                bs = int(img.shape[0])
                if lpips_fn is not None:
                    dist = lpips_fn.calc_lpips(to_lpips_tensor(img), to_lpips_tensor(recon))
                    sum_lpips += float(dist.sum().detach().cpu())
                seen += bs

                if inception_model is not None:
                    real_feats.append(inception_activations(img.float(), inception_model))
                    fake_feats.append(inception_activations(recon.float(), inception_model))

                if lpips_fn is not None and seen > 0:
                    tl.set_postfix(lpips=f"{sum_lpips / seen:.4f}")

        avg_lpips = (sum_lpips / seen) if (lpips_fn is not None and seen > 0) else None
        fid_val = None
        if inception_model is not None and real_feats:
            real_all = np.concatenate(real_feats, axis=0)
            fake_all = np.concatenate(fake_feats, axis=0)
            mu_r, sig_r = activation_stats(real_all)
            mu_f, sig_f = activation_stats(fake_all)
            fid_val = calculate_frechet_distance(mu_r, sig_r, mu_f, sig_f)

        results.append((float(eval_snr), avg_lpips, fid_val))

    return results


@torch.inference_mode()
def eval_lpips_fid_cddm(
    cfg,
    chddim_dict: dict,
    eval_snrs: Sequence[float],
    *,
    lpips_net: str,
    metrics: set[str],
) -> list[tuple[float, float | None, float | None]]:
    dev = cfg.device
    nb = dev.type == "cuda"
    train_snr = float(cfg.SNRs)

    encoder = network.JSCC_encoder(cfg, cfg.C).to(dev)
    decoder = network.JSCC_decoder(cfg, cfg.C).to(dev)
    encoder.load_state_dict(load_state(cfg.encoder_path, dev))
    decoder.load_state_dict(load_state(cfg.re_decoder_path, dev))
    encoder.eval()
    decoder.eval()

    unet = UNet(
        T=chddim_dict["T"],
        ch=int(16 * cfg.C),
        ch_mult=chddim_dict["channel_mult"],
        attn=chddim_dict["attn"],
        num_res_blocks=chddim_dict["num_res_blocks"],
        dropout=chddim_dict["dropout"],
        input_channel=cfg.C,
    ).to(dev)
    unet.load_state_dict(load_state(chddim_dict["save_path"], dev))
    unet.eval()

    sampler = ChannelDiffusionSampler(
        model=unet,
        noise_schedule=chddim_dict["noise_schedule"],
        t_max=chddim_dict["t_max"],
        beta_1=chddim_dict["snr_max"],
        beta_T=chddim_dict["snr_min"],
        T=chddim_dict["T"],
    ).to(dev)
    pass_ch = channel.Channel(cfg)

    _, test_loader = get_loader(cfg)
    assert test_loader is not None

    lpips_fn = None
    if "lpips" in metrics:
        lpips_fn = util_of_lpips(lpips_net, use_gpu=nb)
        lpips_fn.loss_fn.to(dev).eval()

    inception_model = None
    if "fid" in metrics:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIMS]
        inception_model = InceptionV3([block_idx]).to(dev)
        inception_model.eval()

    results: list[tuple[float, float | None, float | None]] = []

    for eval_snr in eval_snrs:
        sum_lpips = 0.0
        seen = 0
        real_feats: list[np.ndarray] = []
        fake_feats: list[np.ndarray] = []

        desc = f"CDDM SNR={eval_snr:g} dB"
        for img, _ in tqdm(test_loader, desc=desc, dynamic_ncols=True):
            img = img.to(dev, non_blocking=nb)
            recon = cddm_reconstruct(
                img,
                encoder,
                decoder,
                pass_ch,
                sampler,
                float(eval_snr),
                train_snr,
                cfg.channel_type,
            )

            bs = int(img.shape[0])
            if lpips_fn is not None:
                dist = lpips_fn.calc_lpips(to_lpips_tensor(img), to_lpips_tensor(recon))
                sum_lpips += float(dist.sum().detach().cpu())
            seen += bs

            if inception_model is not None:
                real_feats.append(inception_activations(img.float(), inception_model))
                fake_feats.append(inception_activations(recon.float(), inception_model))

        avg_lpips = (sum_lpips / seen) if (lpips_fn is not None and seen > 0) else None
        fid_val = None
        if inception_model is not None and real_feats:
            real_all = np.concatenate(real_feats, axis=0)
            fake_all = np.concatenate(fake_feats, axis=0)
            mu_r, sig_r = activation_stats(real_all)
            mu_f, sig_f = activation_stats(fake_all)
            fid_val = calculate_frechet_distance(mu_r, sig_r, mu_f, sig_f)

        results.append((float(eval_snr), avg_lpips, fid_val))

    return results


def build_cddm_config_ns(
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


def parse_metrics(s: str) -> set[str]:
    parts = {p.strip().lower() for p in s.split(",") if p.strip()}
    allowed = {"lpips", "fid"}
    unknown = parts - allowed
    if unknown:
        raise argparse.ArgumentTypeError(f"未知指标: {unknown}，可选 {allowed}")
    if not parts:
        raise argparse.ArgumentTypeError("metrics 不能为空")
    return parts


def print_result_table(
    *,
    mode: str,
    train_snr: int,
    C: int,
    channel_type: str,
    rows: list[tuple[float, float | None, float | None]],
    metrics: set[str],
) -> None:
    print(f"\n=== {mode} · train_SNR {train_snr} dB · C {C} · {channel_type} ===")
    header = f"{'eval_SNR(dB)':>12}"
    if "lpips" in metrics:
        header += f"{'LPIPS':>10}"
    if "fid" in metrics:
        header += f"{'FID':>10}"
    print(header)
    for snr, lp, fid in rows:
        if math.isfinite(snr) and snr == int(snr):
            snr_disp = str(int(snr))
        else:
            snr_disp = f"{snr:g}"
        line = f"{snr_disp:>12}"
        if "lpips" in metrics:
            line += f"{lp:>10.6f}" if lp is not None else f"{'N/A':>10}"
        if "fid" in metrics:
            line += f"{fid:>10.4f}" if fid is not None else f"{'N/A':>10}"
        print(line)


def append_csv(
    csv_path: str,
    *,
    mode: str,
    train_snr: int,
    C: int,
    channel_type: str,
    rows: list[tuple[float, float | None, float | None]],
    metrics: set[str],
) -> None:
    new = not os.path.isfile(csv_path)
    cols = ["mode", "train_snr", "C", "channel", "eval_snr"]
    if "lpips" in metrics:
        cols.append("LPIPS")
    if "fid" in metrics:
        cols.append("FID")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(cols)
        for snr, lp, fid in rows:
            row = [mode, train_snr, C, channel_type, snr]
            if "lpips" in metrics:
                row.append(f"{lp:.8f}" if lp is not None else "")
            if "fid" in metrics:
                row.append(f"{fid:.6f}" if fid is not None else "")
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="DIV2K：JSCC / JSCC+CDDM 多 SNR LPIPS 与 FID 评估")
    p.add_argument("--cddm", action="store_true", help="使用 JSCC+CDDM 流程（默认仅 JSCC）")
    p.add_argument("--train-snr", type=int, required=True, help="训练 SNR（匹配 checkpoint 文件名）")
    p.add_argument("--C", type=int, required=True, help="JSCC 通道数 C")
    p.add_argument("--channel-type", type=str, default="awgn", choices=("awgn", "rayleigh"))
    p.add_argument(
        "--eval-snrs",
        type=parse_eval_snrs,
        default=",".join(str(x) for x in DEFAULT_EVAL_SNRS),
        help="逗号分隔的评估信道 SNR（dB）",
    )
    p.add_argument(
        "--metrics",
        type=parse_metrics,
        default="lpips,fid",
        help="要计算的指标，逗号分隔：lpips,fid",
    )
    p.add_argument("--lpips-net", type=str, default="vgg", choices=("vgg", "alex"))
    p.add_argument("--jscc-root", type=str, default=default_jscc_snr_ckpt_root())
    p.add_argument("--cddm-root", type=str, default=DEFAULT_CDDM_CKPT_ROOT)
    p.add_argument(
        "--test-dir",
        type=str,
        default="/workspace/yongjia/datasets/DIV2K/DIV2K_valid_HR/",
    )
    p.add_argument("--batch-size", type=int, default=4, help="JSCC 模式 DataLoader batch size")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--redecoder-offset",
        type=int,
        default=3,
        help="CDDM 模式：redecoder 文件名 SNR = train_snr - offset",
    )
    p.add_argument("--noise-schedule", type=int, default=1)
    p.add_argument("--t-max", type=int, default=10)
    p.add_argument("--gpu", type=str, default=None)
    p.add_argument("--cpu", action="store_true", help="强制 CPU")
    p.add_argument("--cuda-device", type=str, default=None, help="写入 CUDA_VISIBLE_DEVICES（与 --cpu 互斥）")
    p.add_argument("--no-amp", action="store_true", help="JSCC 模式关闭 autocast")
    p.add_argument("--csv-out", type=str, default=None, help="追加写入 CSV")

    args = p.parse_args()
    if args.cpu and args.cuda_device is not None:
        p.error("--cpu 与 --cuda-device 不能同时使用")

    metrics = args.metrics
    test_dir = os.path.abspath(args.test_dir.rstrip("/"))
    jscc_root = os.path.abspath(args.jscc_root.rstrip("/"))
    mode = "CDDM" if args.cddm else "JSCC"

    if args.cddm:
        enc_path = os.path.join(
            jscc_root,
            f"encoder_{_snr_ckpt_base(args.train_snr, args.channel_type, args.C)}.pt",
        )
        if not os.path.isfile(enc_path):
            raise FileNotFoundError(f"未找到 encoder: {enc_path}")
        _dec_std, re_dec_path = resolve_jscc_decoder_paths_cddm(
            jscc_root, args.train_snr, args.channel_type, args.C, args.redecoder_offset
        )
        cddm_path = resolve_cddm_unet_path(
            os.path.abspath(args.cddm_root.rstrip("/")),
            args.train_snr,
            args.channel_type,
            args.C,
        )

        cfg = build_cddm_config_ns(
            train_snr=args.train_snr,
            C=args.C,
            channel_type=args.channel_type,
            test_data_dir=test_dir,
            encoder_path=enc_path,
            decoder_path=_dec_std,
            re_decoder_path=re_dec_path,
        )
        if args.cpu:
            cfg.CUDA = False
            cfg.device = torch.device("cpu")

        chddim_dict = dict(
            T=1000,
            channel_mult=[1, 2, 2],
            attn=[1],
            num_res_blocks=2,
            dropout=0.1,
            noise_schedule=args.noise_schedule,
            t_max=int(args.t_max),
            snr_max=1e-4,
            snr_min=0.02,
            save_path=cddm_path,
        )
        rows = eval_lpips_fid_cddm(
            cfg,
            chddim_dict,
            args.eval_snrs,
            lpips_net=args.lpips_net,
            metrics=metrics,
        )
        n_img = len(get_loader(cfg)[1].dataset)
    else:
        enc_path, dec_path = resolve_jscc_snr_checkpoints(
            jscc_root, args.train_snr, args.channel_type, args.C
        )
        cfg = build_eval_namespace(
            encoder_path=enc_path,
            decoder_path=dec_path,
            test_data_dir=test_dir,
            image_hw=(256, 256),
            C=args.C,
            channel_type=args.channel_type,
            cuda_device=args.cuda_device,
            use_amp=not args.no_amp,
            force_cpu=bool(args.cpu),
        )

        transform_test = transforms.Compose([
            transforms.CenterCrop((cfg.image_dims[1], cfg.image_dims[2])),
            transforms.ToTensor(),
        ])
        ds = FlatImageFolder(root=cfg.test_data_dir, transform=transform_test)
        n_img = len(ds)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=cfg.device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        rows = eval_lpips_fid_jscc(
            cfg,
            loader,
            args.eval_snrs,
            lpips_net=args.lpips_net,
            use_amp=not args.no_amp,
            metrics=metrics,
        )

    print(f"视觉感知质量评估（{mode}）")
    print(f"device: {cfg.device}")
    print(f"训练 SNR: {args.train_snr} dB | 信道: {args.channel_type} | C: {args.C}")
    print(f"数据集: {test_dir} | 图像数: {n_img}")
    print(f"指标: {','.join(sorted(metrics))} | LPIPS backbone: {args.lpips_net}")

    print_result_table(
        mode=mode,
        train_snr=args.train_snr,
        C=args.C,
        channel_type=args.channel_type,
        rows=rows,
        metrics=metrics,
    )

    if args.csv_out:
        append_csv(
            args.csv_out,
            mode=mode,
            train_snr=args.train_snr,
            C=args.C,
            channel_type=args.channel_type,
            rows=rows,
            metrics=metrics,
        )


if __name__ == "__main__":
    main()
