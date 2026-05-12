"""
根据 **训练时 SNR**、**信道类型**、**通道数 C** 在
`checkpoints/JSCC/DIV2K/MSE/SNRs/` 下定位 JSCC 权重：

- **默认**：encoder + decoder，在多个 **信道评估 SNR** 上计算验证集平均 PSNR（与 eval_jscc_psnr.py 一致）。
- **`--norm-stats-only`**：仅加载 encoder，对编码器输出经 `Channel.complex_normalize` 后的 **channel_tx**
  在整份验证集上做元素均值 / 总体方差统计（与 Train.py / train_jscc 一致）。

用法（在 CDDM 目录下）:
  python test/eval_jscc_norm_feature_stats.py --train-snr 12 --channel-type awgn --C 12
  python test/eval_jscc_norm_feature_stats.py --train-snr 0 --channel-type awgn --C 4 --norm-stats-only
  python test/eval_jscc_norm_feature_stats.py --norm-stats-only --encoder /path/encoder.pt --C 12
      --channel-type awgn --train-snr 0
"""
from __future__ import annotations

import argparse
import math
import os
import sys

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


def default_jscc_snr_ckpt_root() -> str:
    return os.path.join(_CDDM_ROOT, "checkpoints", "JSCC", "DIV2K", "MSE", "SNRs")


def _snr_ckpt_base(train_snr: int, channel_type: str, C: int) -> str:
    return f"snr{train_snr}_channel_{channel_type.lower()}_C{C}"


def resolve_jscc_snr_encoder_path(
    ckpt_root: str,
    train_snr: int,
    channel_type: str,
    C: int,
) -> str:
    enc = os.path.join(ckpt_root, f"encoder_{_snr_ckpt_base(train_snr, channel_type, C)}.pt")
    if not os.path.isfile(enc):
        raise FileNotFoundError(f"未找到 encoder: {enc}")
    return enc


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


def build_eval_namespace(
    encoder_path: str,
    decoder_path: str,
    test_data_dir: str,
    *,
    image_hw: tuple[int, int],
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

    if not force_cpu and cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    return ns


def build_eval_namespace_encoder_only(
    encoder_path: str,
    test_data_dir: str,
    *,
    image_hw: tuple[int, int],
    C: int,
    channel_type: str,
    cuda_device: str | None,
    use_amp: bool,
    force_cpu: bool = False,
) -> argparse.Namespace:
    """仅 encoder + Channel（用于 complex_normalize 统计），无需 decoder。"""
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
    ns.encoder_path = encoder_path
    ns.test_data_dir = test_data_dir
    ns.use_amp = use_amp and ns.CUDA

    if not force_cpu and cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    return ns


@torch.inference_mode()
def run_norm_feature_stats(
    cfg,
    batch_size: int,
    num_workers: int,
    norm_power: float,
) -> tuple[float, float, float, float, float, float, float, float, int]:
    """返回 (
        mean_out, var_out, std_out, n_elem,
        avg_pwr_batches, sum_pwr_batches, min_pwr, max_pwr,
        n_img,
    )。
    avg_pwr_batches = (各 batch 的 pwr 之和) / batch 数；归一化后元素统计跨全验证集 pooled。
    """
    dev = cfg.device
    nb = dev.type == "cuda"
    use_amp = bool(cfg.use_amp) and dev.type == "cuda"

    transform_test = transforms.Compose([
        transforms.CenterCrop((cfg.image_dims[1], cfg.image_dims[2])),
        transforms.ToTensor(),
    ])
    ds = FlatImageFolder(root=cfg.test_data_dir, transform=transform_test)
    n_img = len(ds)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=nb,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    encoder = JSCC_encoder(cfg, cfg.C).to(dev)
    channel = Channel(cfg)
    encoder.load_state_dict(torch.load(cfg.encoder_path, map_location=dev, weights_only=True))
    encoder.eval()

    sum_y = 0.0
    sum_y2 = 0.0
    n_y = 0
    sum_pwr = 0.0
    n_batches = 0
    min_pwr = float("inf")
    max_pwr = float("-inf")

    desc = os.path.basename(os.path.normpath(cfg.test_data_dir))
    with tqdm(loader, desc=f"complex_normalize [{desc}]", dynamic_ncols=True) as tl:
        for img, _ in tl:
            img = img.to(dev, non_blocking=nb)
            with torch.autocast(
                device_type="cuda" if dev.type == "cuda" else "cpu",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                feature, _ = encoder(img)
                channel_tx, pwr_t = channel.complex_normalize(feature, power=norm_power)

            pwr_f = float(pwr_t.detach().float().mean().cpu().item())
            sum_pwr += pwr_f
            n_batches += 1
            min_pwr = min(min_pwr, pwr_f)
            max_pwr = max(max_pwr, pwr_f)

            yf = channel_tx.detach().float()
            yd = yf.double()
            sum_y += float(yd.sum().cpu())
            sum_y2 += float(yd.pow(2).sum().cpu())
            n_y += int(yf.numel())

            tl.set_postfix(pwr=f"{pwr_f:.4f}")

    denom = float(max(n_y, 1))
    mean_y = sum_y / denom
    mean_y2 = sum_y2 / denom
    var_y = mean_y2 - mean_y * mean_y
    if var_y < 0 and var_y > -1e-9:
        var_y = 0.0
    std_y = math.sqrt(max(var_y, 0.0))

    avg_pwr = sum_pwr / max(n_batches, 1)

    return mean_y, var_y, std_y, float(n_y), avg_pwr, sum_pwr, min_pwr, max_pwr, n_img


def feature_to_decoder_input_noisy(
    feature: torch.Tensor,
    channel: Channel,
    snr_db: float,
    channel_type: str,
) -> torch.Tensor:
    """与 train.py `eval_only_JSCC` 一致：encoder feature → 信道 → decoder 入口张量。"""
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
def run_multi_snr_psnr(
    cfg,
    batch_size: int,
    num_workers: int,
    eval_snrs: tuple[float, ...],
) -> tuple[list[tuple[float, float]], int]:
    dev = cfg.device
    nb = dev.type == "cuda"
    use_amp = bool(cfg.use_amp) and dev.type == "cuda"

    transform_test = transforms.Compose([
        transforms.CenterCrop((cfg.image_dims[1], cfg.image_dims[2])),
        transforms.ToTensor(),
    ])
    ds = FlatImageFolder(root=cfg.test_data_dir, transform=transform_test)
    n_img = len(ds)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=nb,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    encoder = JSCC_encoder(cfg, cfg.C).to(dev)
    decoder = JSCC_decoder(cfg, cfg.C).to(dev)
    channel = Channel(cfg)
    encoder.load_state_dict(torch.load(cfg.encoder_path, map_location=dev, weights_only=True))
    decoder.load_state_dict(torch.load(cfg.decoder_path, map_location=dev, weights_only=True))
    encoder.eval()
    decoder.eval()

    out: list[tuple[float, float]] = []
    for snr in eval_snrs:
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


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "按训练 SNR / 信道 / C 加载 JSCC：默认做多 SNR PSNR；"
            "加 --norm-stats-only 则只做 complex_normalize 后特征统计（仅需 encoder）"
        )
    )
    p.add_argument(
        "--norm-stats-only",
        action="store_true",
        help="仅统计编码器输出经 complex_normalize 后 channel_tx 的全局均值/方差（不加载 decoder、不测 PSNR）",
    )
    p.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="仅 --norm-stats-only：显式指定 encoder.pt，覆盖由 train-snr/信道/C 在 ckpt-root 下解析的路径",
    )
    p.add_argument(
        "--norm-power",
        type=float,
        default=1.0,
        help="仅 --norm-stats-only：传给 complex_normalize 的 power（默认 1）",
    )
    p.add_argument(
        "--train-snr",
        type=int,
        required=True,
        help="训练该 JSCC 模型时使用的 SNR（用于匹配 encoder_snr{train_snr}_... 文件名）",
    )
    p.add_argument(
        "--channel-type",
        type=str,
        default="awgn",
        choices=("awgn", "rayleigh"),
        help="信道类型，与检查点文件名中的 channel_* 一致",
    )
    p.add_argument("--C", type=int, required=True, help="JSCC 通道数 C（与检查点文件名 C* 一致）")
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
        help="逗号分隔的评估信道 SNR（dB），默认 0,3,6,9,12,15,18,21",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--cuda-device",
        type=str,
        default=None,
        help='可见 GPU 编号（写入 CUDA_VISIBLE_DEVICES，如 "1" 换卡；与 --cpu 互斥）',
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用 CPU，不调用当前 CUDA 设备（GPU 报错/ECC 时可先试此项）",
    )
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    if args.encoder is not None and not args.norm_stats_only:
        p.error("--encoder 仅可与 --norm-stats-only 一起使用")
    if args.cpu and args.cuda_device is not None:
        p.error("--cpu 与 --cuda-device 不能同时使用")

    ckpt_root = os.path.abspath(args.ckpt_root.rstrip("/"))

    if args.norm_stats_only:
        if args.encoder is not None:
            enc_path = os.path.abspath(args.encoder)
        else:
            enc_path = resolve_jscc_snr_encoder_path(
                ckpt_root, args.train_snr, args.channel_type, args.C
            )

        cfg = build_eval_namespace_encoder_only(
            encoder_path=enc_path,
            test_data_dir=os.path.abspath(args.test_dir.rstrip("/")),
            image_hw=(256, 256),
            C=args.C,
            channel_type=args.channel_type,
            cuda_device=args.cuda_device,
            use_amp=not args.no_amp,
            force_cpu=bool(args.cpu),
        )

        mean_y, var_y, std_y, n_elem, avg_pwr, _, min_pw, max_pw, nimg = run_norm_feature_stats(
            cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            norm_power=args.norm_power,
        )

        print("JSCC complex_normalize 后 channel_tx 元素统计（验证集 pooled）")
        print(f"device: {cfg.device}")
        print(f"ckpt-root: {ckpt_root}")
        print(f"训练 SNR 标识: {args.train_snr} dB | 信道: {args.channel_type} | C: {args.C}")
        print(f"encoder: {cfg.encoder_path}")
        print(f"数据集: {cfg.test_data_dir}")
        print(f"norm_power (complex_normalize): {args.norm_power}")
        print(f"图像数: {nimg}")
        print(
            f"各 batch 的 pwr=mean(feature²)·2:  batch 均值={avg_pwr:.8f}  "
            f"min={min_pw:.8f}  max={max_pw:.8f}"
        )
        print(
            "complex_normalize 后 channel_tx（验证集全体元素）："
            f"均值={mean_y:.8f}, 方差={var_y:.8e}, σ={std_y:.8e}, 元素个数={int(n_elem)}"
        )
        return

    enc_path, dec_path = resolve_jscc_snr_checkpoints(
        ckpt_root, args.train_snr, args.channel_type, args.C
    )

    cfg = build_eval_namespace(
        encoder_path=enc_path,
        decoder_path=dec_path,
        test_data_dir=os.path.abspath(args.test_dir.rstrip("/")),
        image_hw=(256, 256),
        C=args.C,
        channel_type=args.channel_type,
        cuda_device=args.cuda_device,
        use_amp=not args.no_amp,
        force_cpu=bool(args.cpu),
    )

    rows, n_img = run_multi_snr_psnr(
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_snrs=tuple(args.eval_snrs),
    )

    print("JSCC PSNR（各图 PSNR 平均）")
    print(f"device: {cfg.device}")
    print(f"ckpt-root: {ckpt_root}")
    print(f"训练 SNR: {args.train_snr} dB | 信道: {args.channel_type} | C: {args.C}")
    print(f"encoder: {cfg.encoder_path}")
    print(f"decoder: {cfg.decoder_path}")
    print(f"数据集: {cfg.test_data_dir}")
    print(f"图像数: {n_img}")
    print("")
    print(f"{'eval_SNR_db':>12}  {'PSNR_db':>10}")
    for snr, psnr_v in rows:
        if math.isfinite(snr) and snr == int(snr):
            snr_disp = str(int(snr))
        else:
            snr_disp = f"{snr:g}"
        print(f"{snr_disp:>12}  {psnr_v:10.6f}")


if __name__ == "__main__":
    main()
