#!/usr/bin/env python3
"""
在 DIV2K 验证集上评估 JSCC 的 PSNR。

链路：JSCC 编码器 → 归一化(complex_normalize) → 信道(AWGN) → JSCC 解码器
注意：使用 decoder，不使用 redecoder。

根据训练 SNR、信道类型、C 定位 checkpoints/JSCC/DIV2K/MSE/SNRs/ 下的权重，
在多个评估 SNR 上计算验证集平均 PSNR。

用法（在 CDDM 根目录）:
  python test/eval_jscc_psnr.py
  CUDA_VISIBLE_DEVICES=2 python test/eval_jscc_psnr.py --eval-snrs 3,6,9,12
  python test/eval_jscc_psnr.py --train-snr 12 --channel-type awgn --C 4
  python test/eval_jscc_psnr.py --train-snr 12 --channel-type awgn --C 36 --eval-snrs 12
  python test/eval_jscc_psnr.py --include-latest --eval-snrs 6
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def _apply_cuda_visible_devices_before_torch() -> None:
    for flag in ("--gpu", "--cuda-device"):
        if flag not in sys.argv:
            continue
        i = sys.argv.index(flag)
        if i + 1 >= len(sys.argv):
            continue
        val = sys.argv[i + 1]
        if not val.startswith("-"):
            os.environ["CUDA_VISIBLE_DEVICES"] = val


_apply_cuda_visible_devices_before_torch()

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

_CDDM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CDDM_ROOT not in sys.path:
    sys.path.insert(0, _CDDM_ROOT)
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from Autoencoder.data.datasets import FlatImageFolder  # noqa: E402
from Autoencoder.net.channel import Channel  # noqa: E402
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder  # noqa: E402

DEFAULT_EVAL_SNRS = (0, 3, 6, 9, 12, 15, 50)
CKPT_RE = re.compile(
    r"^encoder_snr(?P<snr>-?\d+(?:\.\d+)?)_channel_(?P<channel>[A-Za-z0-9]+)_C(?P<C>\d+)(?P<latest>_latest)?\.pt$"
)


@dataclass(frozen=True)
class JsccCheckpointPair:
    train_snr: float
    channel_type: str
    C: int
    variant: str
    encoder_path: str
    decoder_path: str


def default_jscc_snr_ckpt_root() -> str:
    return os.path.join(_CDDM_ROOT, "MY/checkpoints-jscc")


def _snr_ckpt_base(train_snr: int, channel_type: str, C: int) -> str:
    return f"snr{train_snr}_channel_{channel_type.lower()}_C{C}"


def _format_snr_token(snr: float) -> str:
    return str(int(snr)) if math.isfinite(snr) and snr == int(snr) else f"{snr:g}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_plain_state(path: str, device: torch.device) -> dict:
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=device, weights_only=False)
    return obj.get("state_dict", obj) if isinstance(obj, dict) else obj


def discover_jscc_checkpoint_pairs(
    ckpt_root: str,
    *,
    channel_type: str | None,
    include_latest: bool,
    only_latest: bool,
) -> list[JsccCheckpointPair]:
    root = Path(ckpt_root)
    pairs: list[JsccCheckpointPair] = []
    for enc in sorted(root.glob("encoder_snr*_channel_*_C*.pt")):
        m = CKPT_RE.match(enc.name)
        if not m:
            continue
        ch = m.group("channel").lower()
        if channel_type is not None and ch != channel_type.lower():
            continue
        is_latest = bool(m.group("latest"))
        if only_latest and not is_latest:
            continue
        if is_latest and not include_latest and not only_latest:
            continue
        snr = float(m.group("snr"))
        C = int(m.group("C"))
        suffix = "_latest" if is_latest else ""
        dec = root / f"decoder_snr{m.group('snr')}_channel_{m.group('channel')}_C{C}{suffix}.pt"
        if not dec.is_file():
            print(f"skip missing decoder for {enc}: {dec}", file=sys.stderr)
            continue
        pairs.append(
            JsccCheckpointPair(
                train_snr=snr,
                channel_type=ch,
                C=C,
                variant="latest" if is_latest else "best",
                encoder_path=str(enc),
                decoder_path=str(dec),
            )
        )
    return sorted(pairs, key=lambda x: (x.train_snr, x.C, x.channel_type, x.variant))


def resolve_jscc_snr_checkpoints(
    ckpt_root: str,
    train_snr: int,
    channel_type: str,
    C: int,
) -> tuple[str, str]:
    base = _snr_ckpt_base(train_snr, channel_type, C)
    enc = os.path.join(ckpt_root, f"encoder_{base}.pt")
    dec = os.path.join(ckpt_root, f"decoder_{base}.pt")
    if not os.path.isfile(enc):
        raise FileNotFoundError(f"未找到 encoder: {enc}")
    if not os.path.isfile(dec):
        raise FileNotFoundError(f"未找到 decoder: {dec}")
    return enc, dec


def make_single_pair(ckpt_root: str, train_snr: int, channel_type: str, C: int) -> JsccCheckpointPair:
    enc, dec = resolve_jscc_snr_checkpoints(ckpt_root, train_snr, channel_type, C)
    return JsccCheckpointPair(
        train_snr=float(train_snr),
        channel_type=channel_type.lower(),
        C=int(C),
        variant="best",
        encoder_path=enc,
        decoder_path=dec,
    )


def build_config(
    encoder_path: str,
    decoder_path: str,
    test_data_dir: str,
    *,
    image_hw: tuple[int, int] = (256, 256),
    C: int,
    channel_type: str,
    cuda_device: str | None,
    use_amp: bool,
    force_cpu: bool = False,
) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.loss_function = "MSE"
    ns.dataset = "DIV2K"
    ns.C = C
    ns.seed = 1024
    if force_cpu:
        ns.CUDA = False
        ns.device = torch.device("cpu")
    else:
        ns.CUDA = torch.cuda.is_available()
        ns.device = torch.device("cuda:0" if ns.CUDA else "cpu")
    ns.channel_type = channel_type.lower()
    ns.image_dims = (3, image_hw[0], image_hw[1])
    ns.encoder_kwargs = dict(
        img_size=(image_hw[0], image_hw[1]),
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
        img_size=(image_hw[0], image_hw[1]),
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
    ns.test_data_dir = test_data_dir
    ns.use_amp = use_amp and ns.CUDA

    return ns


def feature_to_decoder_input_noisy(
    feature: torch.Tensor,
    channel: Channel,
    snr_db: float,
    channel_type: str,
) -> torch.Tensor:
    """encoder feature → complex_normalize → 信道 → decoder 入口张量。"""
    noisy_y, pwr, h = channel.forward(feature, snr_db)
    ch = channel_type.lower()
    if ch == "rayleigh":
        sigma_square = 1.0 / (10 ** (snr_db / 10))
        noisy_y = torch.conj(h) * noisy_y / (torch.abs(h) ** 2 + sigma_square)
    elif ch == "awgn":
        pass
    else:
        raise ValueError(f"不支持的信道类型: {channel_type}")

    dec_in = torch.cat((torch.real(noisy_y), torch.imag(noisy_y)), dim=2) * torch.sqrt(pwr)
    return dec_in


def parse_eval_snrs(s: str) -> tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("eval-snrs 不能为空")
    return tuple(float(x) for x in parts)


def make_loader(cfg, batch_size: int, num_workers: int, max_images: int) -> DataLoader:
    nb = cfg.device.type == "cuda"
    transform_test = transforms.Compose([
        transforms.CenterCrop((cfg.image_dims[1], cfg.image_dims[2])),
        transforms.ToTensor(),
    ])
    ds = FlatImageFolder(root=cfg.test_data_dir, transform=transform_test)
    if max_images > 0:
        ds.paths = ds.paths[: int(max_images)]
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=nb,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


@torch.inference_mode()
def run_avg_psnr_at_snr(
    cfg,
    encoder: JSCC_encoder,
    decoder: JSCC_decoder,
    channel: Channel,
    loader: DataLoader,
    eval_snr_db: float,
    *,
    use_amp: bool,
) -> float:
    """各图 PSNR 之和 / 张数。"""
    dev = cfg.device
    nb = dev.type == "cuda"
    use_amp_b = bool(use_amp) and dev.type == "cuda"
    ch_type = cfg.channel_type

    sum_psnr = 0.0
    seen = 0

    with tqdm(loader, desc=f"SNR={eval_snr_db:g} dB", dynamic_ncols=True, leave=False) as tl:
        for img, _ in tl:
            img = img.to(dev, non_blocking=nb)
            with torch.autocast(
                device_type="cuda" if dev.type == "cuda" else "cpu",
                dtype=torch.float16,
                enabled=use_amp_b,
            ):
                feature, _ = encoder(img)
                dec_in = feature_to_decoder_input_noisy(feature, channel, eval_snr_db, ch_type)
                recon = decoder(dec_in)
            recon_c = recon.clamp(0.0, 1.0)

            tgt = img.float() * 255.0
            pred = recon_c.float() * 255.0
            mse_i = torch.mean((tgt - pred) ** 2, dim=(1, 2, 3)).clamp(min=1e-18)
            psnr_i = 10.0 * torch.log10((255.0 * 255.0) / mse_i)
            bs = int(img.shape[0])
            sum_psnr += float(psnr_i.sum().cpu())
            seen += bs
            tl.set_postfix(avg_psnr=f"{sum_psnr / max(seen, 1):.2f} dB")

    n_img = len(loader.dataset)
    assert seen == n_img, f"计数不一致: seen={seen} n_img={n_img}"
    return sum_psnr / float(max(n_img, 1))


@torch.inference_mode()
def run_multi_snr_psnr_for_pair(
    cfg,
    loader: DataLoader,
    eval_snrs: tuple[float, ...],
    seed: int,
) -> tuple[list[tuple[float, float]], int]:
    dev = cfg.device
    use_amp = bool(cfg.use_amp) and dev.type == "cuda"
    n_img = len(loader.dataset)

    encoder = JSCC_encoder(cfg, cfg.C).to(dev)
    decoder = JSCC_decoder(cfg, cfg.C).to(dev)
    channel = Channel(cfg)
    encoder.load_state_dict(load_plain_state(cfg.encoder_path, dev), strict=True)
    decoder.load_state_dict(load_plain_state(cfg.decoder_path, dev), strict=True)
    encoder.eval()
    decoder.eval()

    out: list[tuple[float, float]] = []
    for idx, snr in enumerate(eval_snrs):
        seed_everything(int(seed) + 100003 * idx)
        avg = run_avg_psnr_at_snr(
            cfg,
            encoder,
            decoder,
            channel,
            loader,
            snr,
            use_amp=use_amp,
        )
        out.append((snr, avg))

    return out, n_img


def write_csv(path: str, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="JSCC PSNR 评估：encoder → complex_normalize → 信道 → decoder"
    )
    p.add_argument(
        "--train-snr",
        type=int,
        default=None,
        help="训练该 JSCC 模型时使用的 SNR（用于匹配 encoder_snr{train_snr}_... 文件名）",
    )
    p.add_argument(
        "--channel-type",
        type=str,
        default="awgn",
        choices=("awgn", "rayleigh"),
        help="信道类型，与检查点文件名中的 channel_* 一致",
    )
    p.add_argument("--C", type=int, default=None, help="JSCC 通道数 C（与检查点文件名 C* 一致）")
    p.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="评估 ckpt-root 下所有匹配的 JSCC encoder/decoder 权重对；未指定 --train-snr/--C 时默认启用",
    )
    p.add_argument("--include-latest", action="store_true", help="批量模式同时评估 *_latest.pt")
    p.add_argument("--only-latest", action="store_true", help="批量模式只评估 *_latest.pt")
    p.add_argument(
        "--ckpt-root",
        type=str,
        default=default_jscc_snr_ckpt_root(),
        help="含 encoder_snr*_channel_*_C*.pt 的目录",
    )
    p.add_argument(
        "--test-dir",
        type=str,
        default="/workspace/yongjia/datasets/DIV2K/DIV2K_valid_HR/",
    )
    p.add_argument(
        "--eval-snrs",
        type=parse_eval_snrs,
        default=",".join(str(x) for x in DEFAULT_EVAL_SNRS),
        help="逗号分隔的评估信道 SNR（dB），默认 0,3,6,9,12,15,50",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-images", type=int, default=0, help="只评估前 N 张图；0 表示全验证集")
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument(
        "--cuda-device",
        type=str,
        default=None,
        help='可见 GPU 编号（写入 CUDA_VISIBLE_DEVICES，如 "1" 换卡；与 --cpu 互斥）',
    )
    p.add_argument("--gpu", type=str, default=None, help="同 --cuda-device；在 import torch 前设置 CUDA_VISIBLE_DEVICES")
    p.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用 CPU",
    )
    p.add_argument("--amp", action="store_true", help="启用 CUDA float16 autocast；默认 fp32")
    p.add_argument("--no-amp", action="store_true", help="兼容旧参数；默认已经不启用 AMP")
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="结果 CSV 路径；批量模式默认写到 ckpt-root/eval_jscc_psnr_all.csv",
    )
    args = p.parse_args()

    if args.gpu is not None and args.cuda_device is not None and args.gpu != args.cuda_device:
        p.error("--gpu 与 --cuda-device 指定不一致")
    if args.cpu and (args.cuda_device is not None or args.gpu is not None):
        p.error("--cpu 与 --gpu/--cuda-device 不能同时使用")
    if args.include_latest and args.only_latest:
        p.error("--include-latest 与 --only-latest 不能同时使用")
    if args.amp and args.no_amp:
        p.error("--amp 与 --no-amp 不能同时使用")

    ckpt_root = os.path.abspath(args.ckpt_root.rstrip("/"))
    batch_mode = bool(args.all_checkpoints or (args.train_snr is None and args.C is None))
    if batch_mode:
        pairs = discover_jscc_checkpoint_pairs(
            ckpt_root,
            channel_type=args.channel_type,
            include_latest=bool(args.include_latest),
            only_latest=bool(args.only_latest),
        )
    else:
        if args.train_snr is None or args.C is None:
            p.error("单模型评估必须同时指定 --train-snr 和 --C；或使用 --all-checkpoints")
        pairs = [make_single_pair(ckpt_root, args.train_snr, args.channel_type, args.C)]
    if not pairs:
        raise FileNotFoundError(f"未在 {ckpt_root} 找到匹配的 JSCC encoder/decoder 权重对")

    first = pairs[0]
    base_cfg = build_config(
        encoder_path=first.encoder_path,
        decoder_path=first.decoder_path,
        test_data_dir=os.path.abspath(args.test_dir.rstrip("/")),
        image_hw=(256, 256),
        C=first.C,
        channel_type=first.channel_type,
        cuda_device=args.gpu or args.cuda_device,
        use_amp=bool(args.amp),
        force_cpu=bool(args.cpu),
    )
    loader = make_loader(base_cfg, args.batch_size, args.num_workers, args.max_images)

    print("JSCC PSNR 评估（encoder → 归一化 → 信道 → decoder）")
    print(f"device: {base_cfg.device} | visible_cuda: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"ckpt-root: {ckpt_root}")
    print(f"checkpoint pairs: {len(pairs)}")
    print(f"数据集: {base_cfg.test_data_dir}")
    print(f"图像数: {len(loader.dataset)} | batch_size: {args.batch_size}")
    print(f"amp: {'float16-on-cuda' if args.amp else 'off'}")
    print("")

    csv_rows: list[dict[str, object]] = []
    for pair_idx, pair in enumerate(pairs, start=1):
        cfg = build_config(
            encoder_path=pair.encoder_path,
            decoder_path=pair.decoder_path,
            test_data_dir=base_cfg.test_data_dir,
            image_hw=(256, 256),
            C=pair.C,
            channel_type=pair.channel_type,
            cuda_device=args.gpu or args.cuda_device,
            use_amp=bool(args.amp),
            force_cpu=bool(args.cpu),
        )
        print(
            f"[{pair_idx}/{len(pairs)}] train_snr={_format_snr_token(pair.train_snr)} "
            f"channel={pair.channel_type} C={pair.C} variant={pair.variant}",
            flush=True,
        )
        rows, n_img = run_multi_snr_psnr_for_pair(
            cfg,
            loader=loader,
            eval_snrs=tuple(args.eval_snrs),
            seed=int(args.seed),
        )
        for snr, psnr_v in rows:
            print(f"  eval_snr={_format_snr_token(snr):>4} dB | PSNR={psnr_v:.6f} dB", flush=True)
            csv_rows.append(
                {
                    "train_snr": pair.train_snr,
                    "channel_type": pair.channel_type,
                    "C": pair.C,
                    "variant": pair.variant,
                    "eval_snr": snr,
                    "psnr_db": psnr_v,
                    "n_images": n_img,
                    "encoder_path": pair.encoder_path,
                    "decoder_path": pair.decoder_path,
                }
            )

    csv_path = args.csv
    if not csv_path and batch_mode:
        suffix = "latest" if args.only_latest else "with_latest" if args.include_latest else "best"
        csv_path = os.path.join(ckpt_root, f"eval_jscc_psnr_all_{suffix}.csv")
    if csv_path:
        csv_abs = os.path.abspath(csv_path)
        write_csv(csv_abs, csv_rows)
        print(f"\nwrote csv: {csv_abs}")


if __name__ == "__main__":
    main()
