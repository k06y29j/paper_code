from __future__ import annotations

import argparse
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

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

class AverageMeter:
    def __init__(self):
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

def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = Path(resolve_path(path))
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== CVQ tail JSCC @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f

def build_config(args: argparse.Namespace, batch_size: int | None = None) -> CDDMJSCCConfig:
    return CDDMJSCCConfig(
        C=int(args.latent_ch),
        SNRs=float(args.snr_db),
        channel_type="awgn",
        batch_size=int(args.batch_size if batch_size is None else batch_size),
        test_batch=int(args.test_batch),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.val_num_workers),
        train_data_dir=resolve_path(os.path.join(args.data_dir, "DIV2K_train_HR")),
        test_data_dir=resolve_path(os.path.join(args.data_dir, "DIV2K_valid_HR")),
        CUDA=(not bool(args.cpu)),
    )

def prefix16_norm_all(z: torch.Tensor, prefix_ch: int = 16, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    power = z[:, :prefix_ch].float().square().mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.rsqrt(power + eps).to(dtype=z.dtype)
    return z * scale, scale

def real_awgn(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    noise_std = 10.0 ** (-float(snr_db) / 20.0)
    return x + noise_std * torch.randn_like(x)

def sample_nested_m(device: torch.device, max_m: int = 16, batch_size: int | None = None) -> int | torch.Tensor:
    max_m = int(max_m)
    if max_m < 1:
        raise ValueError(f"nested dropout max_m must be positive, got {max_m}")
    if batch_size is None:
        r = torch.rand((), device=device)
        if float(r.item()) < 0.25:
            return 0
        if float(r.item()) < 0.45:
            return max_m
        return int(torch.randint(1, max_m, (), device=device).item()) if max_m > 1 else max_m
    r = torch.rand(batch_size, device=device)
    m = torch.randint(1, max_m, (batch_size,), device=device) if max_m > 1 else torch.ones(batch_size, device=device, dtype=torch.long)
    m = torch.where(r < 0.25, torch.zeros_like(m), m)
    m = torch.where((r >= 0.25) & (r < 0.45), torch.full_like(m, max_m), m)
    return m

def apply_nested_tail(tail: torch.Tensor, m: int | torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(tail)
    if isinstance(m, int):
        if m > 0:
            out[:, :m] = tail[:, :m]
        return out
    idx = torch.arange(tail.shape[1], device=tail.device).view(1, -1, 1, 1)
    mask = idx < m.view(-1, 1, 1, 1)
    return tail * mask.to(dtype=tail.dtype)

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

def should_validate(args: argparse.Namespace, epoch: int) -> bool:
    return int(args.val_every) > 0 and (epoch % int(args.val_every) == 0 or epoch == int(args.epochs))

def should_save_latest(args: argparse.Namespace, epoch: int) -> bool:
    return int(args.latest_every) > 0 and epoch % int(args.latest_every) == 0

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

def print_run_header(args: argparse.Namespace, title: str, train_n: int, val_n: int) -> None:
    print(f"=== {title} ===")
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"save_dir={resolve_path(args.save_dir)}")
    print(f"snr_db={float(args.snr_db):g} channel=real_awgn latent_ch={args.latent_ch} prefix_ch={args.prefix_ch}")
    print(
        f"latent_hw=({args.latent_h},{args.latent_w}) K_A={args.k_a} K_B={args.k_b} "
        f"cvq_mode={getattr(args, 'cvq_mode', 'single')} car_arch={getattr(args, 'car_arch', 'legacy')}"
    )
    print(f"init_jscc_encoder={args.init_jscc_encoder or '<none>'}")
    print(f"init_jscc_decoder={args.init_jscc_decoder or '<none>'}")
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}")

def check_args(args: argparse.Namespace) -> None:
    if int(args.latent_ch) != 36 or int(args.prefix_ch) != 16:
        raise ValueError("This run is configured for C=36 with prefix16/tail20; use --latent-ch 36 --prefix-ch 16.")
    if int(args.latent_ch) - int(args.prefix_ch) != 20:
        raise ValueError("C36 CVQ/CAR expects exactly 20 tail channels.")
    if int(args.latent_h) != 16 or int(args.latent_w) != 16:
        raise ValueError("Current CDDM JSCC encoder produces 16x16 latents for 256x256 DIV2K crops.")

def default_log_name(stage: int, snr_db: float, latent_ch: int) -> str:
    snr = f"{float(snr_db):g}"
    names = {
        1: f"stage1_jscc_tail_nested_c{int(latent_ch)}_snr{snr}.log",
        2: f"stage2_codebook_init_c{int(latent_ch)}_snr{snr}.log",
        3: f"stage3_jscc_cvq_joint_c{int(latent_ch)}_snr{snr}.log",
        4: f"stage4_oracle_c{int(latent_ch)}_snr{snr}.log",
        5: f"stage5_car_c{int(latent_ch)}_snr{snr}.log",
        6: f"stage6_car_decoder_finetune_c{int(latent_ch)}_snr{snr}.log",
    }
    return names[int(stage)]
