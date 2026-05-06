"""
仅训练 JSCC 自动编码器（图像经编码器 → 归一化与实虚拼接 → 解码器），无 CDDM 等。
用法（在 CDDM 目录下）:
  python test/pipeline.py
  python test/pipeline.py --basepath /path/to/checkpoints  # 默认可写目录在 CDDM/checkpoints/JSCC
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

_CDDM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CDDM_ROOT not in sys.path:
    sys.path.insert(0, _CDDM_ROOT)
_CUR = os.path.dirname(os.path.abspath(__file__))
if _CUR not in sys.path:
    sys.path.insert(0, _CUR)
import train_jscc

train_jscc_seqeratly = train_jscc.train_jscc_seqeratly


def seed_torch(seed=1024, fast_cudnn: bool = True):
    """fast_cudnn=True 时打开 cudnn benchmark、关闭全确定性，以显著加快训练/验证卷积（可复现性略降）。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if fast_cudnn:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class config:
    """DIV2K JSCC 自动编码器训练所需超参（无信道扩散、无多余数据集分支）。"""

    def __init__(self, encoder_path, decoder_path, C=18):
        self.loss_function = "MSE"
        self.dataset = "DIV2K"
        self.C = C
        self.seed = 1024
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.CUDA else "cpu")
        self.learning_rate = 0.0001
        self.epoch = 600
        self.save_model_freq = 100
        # 800=20×40，在 drop_last 下无丢弃；可按显存再调大
        self.batch_size = 40
        self.CDDM_batch = 16
        # 数据加载（不构建验证集以加快启动与省内存）
        self.load_val_data = False
        self.num_workers = 32
        self.persistent_workers = True
        # 自动混合精度（需 CUDA，通常有显著加速）
        self.use_amp = True
        self.image_dims = (3, 256, 256)
        self.train_data_dir = "/data/small-datasets-1/DIV2K/DIV2K_train_HR/"
        self.test_data_dir = "/data/small-datasets-1/DIV2K/DIV2K_valid_HR/"
        self.encoder_kwargs = dict(
            img_size=(self.image_dims[1], self.image_dims[2]),
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
        self.decoder_kwargs = dict(
            img_size=(self.image_dims[1], self.image_dims[2]),
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
        self.channel_type = "awgn"
        # 为 True 时在每次存 encoder.pt/decoder.pt 时额外写入 encoder_epoch{N}.pt / decoder_epoch{N}.pt
        self.save_checkpoint_history = True
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path


def main():
    parser = argparse.ArgumentParser(description="DIV2K JSCC 仅自动编码器训练")
    parser.add_argument(
        "--basepath",
        type=str,
        default=os.path.join(_CDDM_ROOT, "checkpoints", "JSCC"),
        help="checkpoint 根目录（默认可写：CDDM 下的 checkpoints/JSCC）",
    )
    parser.add_argument(
        "--cuda-device",
        type=str,
        default=None,
        help='若设置则写入环境变量 CUDA_VISIBLE_DEVICES，例如 "0"',
    )
    parser.add_argument(
        "--C",
        type=int,
        default=18,
        help="通道维度（与 JSCC 瓶颈宽度一致）",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="关闭自动混合精度",
    )
    parser.add_argument(
        "--no-checkpoint-history",
        action="store_true",
        help="不对每个保存点额外写入 encoder_epoch{N}.pt / decoder_epoch{N}.pt",
    )
    args = parser.parse_args()
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    basepath = args.basepath
    save_dir = os.path.join(basepath, "DIV2K")
    os.makedirs(save_dir, exist_ok=True)
    encoder_path = os.path.join(save_dir, "encoder.pt")
    decoder_path = os.path.join(save_dir, "decoder.pt")

    jscc_config = config(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        C=args.C,
    )
    if args.no_amp:
        jscc_config.use_amp = False
    jscc_config.save_checkpoint_history = not args.no_checkpoint_history
    seed_torch(jscc_config.seed, fast_cudnn=True)
    print("开始训练 JSCC 自动编码器: DIV2K, epoch={}, 每 {} epoch 保存".format(
        jscc_config.epoch, jscc_config.save_model_freq
    ))
    print("  训练集:", jscc_config.train_data_dir)
    print("  encoder ->", encoder_path)
    print("  decoder ->", decoder_path)
    train_jscc_seqeratly(jscc_config)


if __name__ == "__main__":
    main()
