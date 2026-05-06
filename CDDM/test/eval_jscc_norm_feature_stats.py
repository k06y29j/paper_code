"""
对 JSCC 编码器输出先做 `Channel.complex_normalize`（与 Train.py / train_jscc 一致），
再在整份验证集上对 **归一化后张量 channel_tx** 的所有元素累计 **均值 / 总体方差**（fp64）。

`complex_normalize` 定义：`pwr = mean(x²)·2`，`out = sqrt(power) * x / sqrt(pwr)`。

用法（在 CDDM 目录下）:
  python test/eval_jscc_norm_feature_stats.py
  python test/eval_jscc_norm_feature_stats.py --encoder path/to/encoder.pt \\
      --test-dir /data/small-datasets-1/DIV2K/DIV2K_valid_HR/
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
from Autoencoder.net.network import JSCC_encoder  # noqa: E402


def build_eval_namespace(
    encoder_path: str,
    test_data_dir: str,
    *,
    image_hw: tuple[int, int],
    C: int,
    cuda_device: str | None,
    use_amp: bool,
) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.loss_function = "MSE"
    ns.dataset = "DIV2K"
    ns.C = C
    ns.seed = 1024
    ns.CUDA = torch.cuda.is_available()
    ns.device = torch.device("cuda:0" if ns.CUDA else "cpu")
    ns.channel_type = "awgn"
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

    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    return ns


@torch.inference_mode()
def run_norm_feature_stats(
    cfg,
    batch_size: int,
    num_workers: int,
    norm_power: float,
) -> tuple[float, float, float, float, int, float, float, int]:
    """返回 (
        mean_out, var_out, std_out, n_elem,
        avg_pwr_batches, sum_pwr_batches, min_pwr, max_pwr,
        n_img,
    )。
    avg_pwr_batches = (各 batch 的 pwr 之和) / batch 数；归一化后元素统计跨全验证集 pooled。
    """
    dev = cfg.device
    use_amp = bool(cfg.use_amp) and dev.type == "cuda"
    nb = True

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


def main() -> None:
    div2k_ckpt = os.path.join(_CDDM_ROOT, "checkpoints", "JSCC", "DIV2K")
    p = argparse.ArgumentParser(
        description="JSCC 编码器输出经 complex_normalize 后的元素均值/方差（验证集 pooled）"
    )
    p.add_argument(
        "--encoder",
        type=str,
        default=os.path.join(div2k_ckpt, "encoder.pt"),
    )
    p.add_argument(
        "--test-dir",
        type=str,
        default="/data/small-datasets-1/DIV2K/DIV2K_valid_HR/",
    )
    p.add_argument("--C", type=int, default=18)
    p.add_argument("--norm-power", type=float, default=1.0, help="complex_normalize 的 power 参数")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--cuda-device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    cfg = build_eval_namespace(
        encoder_path=os.path.abspath(args.encoder),
        test_data_dir=os.path.abspath(args.test_dir.rstrip("/")),
        image_hw=(256, 256),
        C=args.C,
        cuda_device=args.cuda_device,
        use_amp=not args.no_amp,
    )

    mean_y, var_y, std_y, n_elem, avg_pwr, _, min_pw, max_pw, nimg = run_norm_feature_stats(
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        norm_power=args.norm_power,
    )

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


if __name__ == "__main__":
    main()
