"""
加载 JSCC encoder/decoder checkpoint，对 DIV2K 验证集（或指定目录）评估。

- PSNR：每张图单独在像素尺度 [0,255] 上计算，再 (**各图 PSNR 之和 / 图像数**，Σ PSNR_i / N)。
- JSCC 编码器输出：正向中 `encoder(img)` 返回的 **feature**（送入信道 `_encoder_to_decoder_input` 之前），
  在整张验证集上对所有 feature 元素做 **均值 / 总体方差**（E[z²] − E[z]²，fp64 累加）。

用法（在 CDDM 目录下）:
  python test/eval_jscc_psnr.py
  python test/eval_jscc_psnr.py --encoder /path/encoder.pt --decoder /path/decoder.pt \\
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
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder  # noqa: E402
from train_jscc import _encoder_to_decoder_input  # noqa: E402


def build_eval_namespace(
    encoder_path: str,
    decoder_path: str,
    test_data_dir: str,
    *,
    image_hw: tuple[int, int],
    C: int,
    cuda_device: str | None,
    use_amp: bool,
):
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

    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    return ns


@torch.inference_mode()
def run_eval(
    cfg,
    batch_size: int,
    num_workers: int,
) -> tuple[float, float, float, float, float, float, int]:
    """返回 (avg_psnr, sum_psnr, mean_feature, var_feature, std_feature, n_elem_feature, n_img)。"""
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
    decoder = JSCC_decoder(cfg, cfg.C).to(dev)
    channel = Channel(cfg)
    encoder.load_state_dict(torch.load(cfg.encoder_path, map_location=dev, weights_only=True))
    decoder.load_state_dict(torch.load(cfg.decoder_path, map_location=dev, weights_only=True))
    encoder.eval()
    decoder.eval()

    sum_psnr = 0.0
    sum_feat = 0.0
    sum_feat2 = 0.0
    n_feat = 0
    seen = 0

    desc = os.path.basename(os.path.normpath(cfg.test_data_dir))
    with tqdm(loader, desc=f"JSCC PSNR [{desc}]", dynamic_ncols=True) as tl:
        for img, _ in tl:
            img = img.to(dev, non_blocking=nb)
            with torch.autocast(
                device_type="cuda" if dev.type == "cuda" else "cpu",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                feature, _ = encoder(img)
                dec_in = _encoder_to_decoder_input(channel, feature)
                recon = decoder(dec_in)
            recon_c = recon.clamp(0.0, 1.0)

            # 每张图 PSNR，动态范围 [0,255]，与训练中整图 MSE 定义一致（除的是 C×H×W）
            tgt = img.float() * 255.0
            pred = recon_c.float() * 255.0
            mse_i = torch.mean((tgt - pred) ** 2, dim=(1, 2, 3)).clamp(min=1e-18)
            psnr_i = 10.0 * torch.log10((255.0 * 255.0) / mse_i)
            bs = int(img.shape[0])
            sum_psnr += float(psnr_i.sum())
            seen += bs

            xf = feature.detach().float()
            xd = xf.double()
            sum_feat += float(xd.sum().cpu())
            sum_feat2 += float(xd.pow(2).sum().cpu())
            n_feat += int(xf.numel())

            tl.set_postfix(avg_psnr=f"{sum_psnr / max(seen, 1):.2f} dB")

    denom = float(max(n_img, 1))
    avg_psnr = sum_psnr / denom
    denom_f = float(max(n_feat, 1))
    mean_f = sum_feat / denom_f
    mean_f2 = sum_feat2 / denom_f
    var_f = mean_f2 - mean_f * mean_f
    if var_f < 0 and var_f > -1e-9:
        var_f = 0.0
    std_f = math.sqrt(max(var_f, 0.0))

    assert seen == n_img, f"计数不一致: seen={seen} n_img={n_img}"

    return avg_psnr, sum_psnr, mean_f, var_f, std_f, float(n_feat), n_img


def main():
    p = argparse.ArgumentParser(description="JSCC DIV2K 验证集 PSNR + 编码器特征统计")
    div2k_ckpt = os.path.join(_CDDM_ROOT, "checkpoints", "JSCC", "DIV2K")
    p.add_argument(
        "--encoder",
        type=str,
        default=os.path.join(div2k_ckpt, "encoder.pt"),
    )
    p.add_argument(
        "--decoder",
        type=str,
        default=os.path.join(div2k_ckpt, "decoder.pt"),
    )
    p.add_argument(
        "--test-dir",
        type=str,
        default="/data/small-datasets-1/DIV2K/DIV2K_valid_HR/",
    )
    p.add_argument("--C", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--cuda-device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    cfg = build_eval_namespace(
        encoder_path=os.path.abspath(args.encoder),
        decoder_path=os.path.abspath(args.decoder),
        test_data_dir=os.path.abspath(args.test_dir.rstrip("/")),
        image_hw=(256, 256),
        C=args.C,
        cuda_device=args.cuda_device,
        use_amp=not args.no_amp,
    )

    avg_psnr, sum_psnr, mean_f, var_f, std_f, n_elem_f, nimg = run_eval(
        cfg, batch_size=args.batch_size, num_workers=args.num_workers
    )

    print(f"数据集: {cfg.test_data_dir}")
    print(f"encoder: {cfg.encoder_path}")
    print(f"decoder: {cfg.decoder_path}")
    print(f"图像数: {nimg}")
    print(f"PSNR (各图之和/张数 ΣPSNR_i/N): {avg_psnr:.6f} dB  (= {sum_psnr:.6f} / {nimg})")
    print(
        "JSCC 编码器输出 feature（验证集全体元素）："
        f"均值={mean_f:.8f}, 方差={var_f:.8e}, σ={std_f:.8e}, 元素个数={int(n_elem_f)}"
    )


if __name__ == "__main__":
    main()