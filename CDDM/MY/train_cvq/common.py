from __future__ import annotations

import builtins
import json
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CDDM_ROOT = Path(__file__).resolve().parents[2]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.meshgrid.*", category=UserWarning)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


@dataclass
class CDDMJSCCConfig:
    C: int
    SNRs: float
    channel_type: str
    batch_size: int
    test_batch: int
    num_workers: int
    val_num_workers: int
    train_data_dir: str
    test_data_dir: str
    CUDA: bool = True
    dataset: str = "DIV2K"
    loss_function: str = "MSE"
    image_dims: tuple[int, int, int] = (3, 256, 256)
    pin_memory: bool = True
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        self.device = torch.device("cuda:0" if self.CUDA and torch.cuda.is_available() else "cpu")
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else CDDM_ROOT / p)


def setup_log_file(path: str | None) -> object | None:
    if not path:
        return None
    abs_path = Path(resolve_path(path))
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== CVQ-v2 @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def prefix_power_normalize(
    z: torch.Tensor,
    prefix_ch: int = 16,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if z.ndim != 4:
        raise ValueError(f"expected latent [B,C,H,W], got {tuple(z.shape)}")
    if z.shape[1] < int(prefix_ch):
        raise ValueError(f"latent channels {z.shape[1]} < prefix_ch {prefix_ch}")
    prefix_power = z[:, : int(prefix_ch)].float().square().mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.rsqrt(prefix_power + eps).to(dtype=z.dtype)
    return z * scale, prefix_power


def real_awgn(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    noise_std = 10.0 ** (-float(snr_db) / 20.0)
    return x + noise_std * torch.randn_like(x)


def sample_c2_nested_prefix_mask(
    batch_size: int,
    c2_ch: int,
    nested_drop_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ratio = float(nested_drop_ratio)
    idx = torch.arange(int(c2_ch), device=device).view(1, int(c2_ch), 1, 1)
    k = torch.randint(0, int(c2_ch) + 1, (int(batch_size),), device=device)
    use_nested = torch.rand(int(batch_size), device=device) < ratio
    k = torch.where(use_nested, k, torch.full_like(k, int(c2_ch)))
    return (idx < k.view(-1, 1, 1, 1)).to(dtype=dtype)


def sample_c2_channel_mask(
    batch_size: int,
    c2_ch: int,
    dropout_prob: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return sample_c2_nested_prefix_mask(batch_size, c2_ch, dropout_prob, device, dtype)


def split_c1_c2(z: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    c1 = int(args.c1_ch)
    return z[:, :c1], z[:, c1:]


def freeze_module(module: torch.nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(trainable)
    module.train(bool(trainable))


def recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon.float(), target.float(), reduction="mean")


def psnr_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse = (x_hat.float().clamp(0.0, 1.0) - x.float()).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def batch_metric_mean(values: torch.Tensor) -> float:
    return float(values.float().mean().item())


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.6g}" for k, v in sorted(metrics.items()))


def print_epoch(stage: str, epoch: int, total: int, metrics: dict[str, float], elapsed: float) -> None:
    print(f"[{stage} epoch {epoch:03d}/{total:03d}] {format_metrics(metrics)} time={elapsed:.1f}s", flush=True)


def should_validate(args, epoch: int) -> bool:
    return int(args.val_every) > 0 and (epoch % int(args.val_every) == 0 or epoch == int(args.epochs))


def should_save_latest(args, epoch: int) -> bool:
    return int(args.latest_every) > 0 and epoch % int(args.latest_every) == 0


def write_json(path: str | Path, payload: dict) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def check_args(args) -> None:
    if int(args.latent_ch) != 36:
        raise ValueError("CVQ-v2 is configured for C=36; use --latent-ch 36.")
    if int(args.c1_ch) != 16:
        raise ValueError("CVQ-v2 stage design requires C1=16; use --c1-ch 16.")
    if int(args.latent_ch) - int(args.c1_ch) != 20:
        raise ValueError("CVQ-v2 stage design requires C2=20.")
    if int(args.latent_h) != 16 or int(args.latent_w) != 16:
        raise ValueError("The CDDM JSCC C36 encoder is expected to produce 16x16 latents for 256x256 crops.")
    if int(args.k) != 16384:
        raise ValueError("This requested design uses K=16384; use --k 16384.")


def print_run_header(args, title: str, train_n: int, val_n: int) -> None:
    print(f"=== {title} ===")
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"save_dir={resolve_path(args.save_dir)}")
    if getattr(args, "init_ckpt", ""):
        print(f"init_ckpt={resolve_path(args.init_ckpt)}")
    if getattr(args, "init_jscc_encoder", ""):
        print(f"init_jscc_encoder={resolve_path(args.init_jscc_encoder)}")
    if getattr(args, "init_jscc_decoder", ""):
        print(f"init_jscc_decoder={resolve_path(args.init_jscc_decoder)}")
    print(f"latent_ch={args.latent_ch} C1={args.c1_ch} C2={int(args.latent_ch) - int(args.c1_ch)} K={args.k}")
    print(f"power_norm=all_latents_scaled_by_C1_mean_square stage={args.stage} snr_db={args.snr_db:g}")
    if hasattr(args, "nested_drop_ratio"):
        print(f"c2_dropout=prefix_nested_uniform_k_0_to_{int(args.latent_ch) - int(args.c1_ch)} ratio={float(args.nested_drop_ratio):g}")
    if int(getattr(args, "stage", 0)) == 1:
        print(
            "loss_stage1="
            f"{float(args.lambda_c1):g}*L_rec_c1_only+"
            f"{float(args.lambda_drop):g}*L_rec_nested_drop+"
            f"{float(args.lambda_full):g}*L_rec_full+"
            f"{float(args.lambda_vq):g}*L_vq"
        )
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}")
