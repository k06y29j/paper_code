#!/usr/bin/env python3
import argparse
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


CDDM_ROOT = Path(__file__).resolve().parents[1]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.meshgrid.*", category=UserWarning)

from Autoencoder.data.datasets import get_loader  # noqa: E402
from Autoencoder.net.channel import Channel  # noqa: E402
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder  # noqa: E402


def seed_torch(seed: int = 1024) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class TrainConfig:
    loss_function: str
    channel_type: str
    dataset: str
    SNRs: float
    C: int
    encoder_path: str
    decoder_path: str
    train_data_dir: str
    test_data_dir: str
    batch_size: int
    test_batch: int
    num_workers: int
    val_num_workers: int
    epoch: int
    save_model_freq: int
    learning_rate: float
    seed: int = 1024
    CUDA: bool = True
    image_dims: tuple[int, int, int] = (3, 256, 256)

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


def default_ckpt_root() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints" / "JSCC" / "DIV2K" / "MSE" / "SNRs")


def build_config(args: argparse.Namespace) -> TrainConfig:
    ckpt_root = Path(args.ckpt_root).resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)
    base = f"snr{args.snr_db:g}_channel_{args.channel_type}_C{args.C}"
    return TrainConfig(
        loss_function="MSE",
        channel_type=args.channel_type,
        dataset="DIV2K",
        SNRs=float(args.snr_db),
        C=int(args.C),
        encoder_path=str(ckpt_root / f"encoder_{base}.pt"),
        decoder_path=str(ckpt_root / f"decoder_{base}.pt"),
        train_data_dir=str(Path(args.train_dir).resolve()),
        test_data_dir=str(Path(args.val_dir).resolve()),
        batch_size=int(args.batch_size),
        test_batch=int(args.test_batch),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.val_num_workers),
        epoch=int(args.epochs),
        save_model_freq=int(args.save_every),
        learning_rate=float(args.lr),
        seed=int(args.seed),
        CUDA=not bool(args.cpu),
    )


def save_pair(encoder: nn.Module, decoder: nn.Module, cfg: TrainConfig) -> None:
    torch.save(encoder.state_dict(), cfg.encoder_path)
    torch.save(decoder.state_dict(), cfg.decoder_path)


def save_latest_pair(encoder: nn.Module, decoder: nn.Module, cfg: TrainConfig) -> None:
    enc_latest = str(Path(cfg.encoder_path).with_name(Path(cfg.encoder_path).stem + "_latest.pt"))
    dec_latest = str(Path(cfg.decoder_path).with_name(Path(cfg.decoder_path).stem + "_latest.pt"))
    torch.save(encoder.state_dict(), enc_latest)
    torch.save(decoder.state_dict(), dec_latest)


def load_state(module: nn.Module, path: str, name: str) -> None:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    missing, unexpected = module.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"{name} load mismatch: missing={missing}, unexpected={unexpected}")
    print(f"loaded {name}: {path}", flush=True)


def channel_to_decoder_input(feature: torch.Tensor, channel: Channel, cfg: TrainConfig) -> torch.Tensor:
    noisy_y, pwr, h = channel.forward(feature, cfg.SNRs)
    if cfg.channel_type == "rayleigh":
        sigma_square = 1.0 / (10 ** (cfg.SNRs / 10))
        noisy_y = torch.conj(h) * noisy_y / (torch.abs(h) ** 2 + sigma_square)
    elif cfg.channel_type != "awgn":
        raise ValueError(f"unsupported channel_type={cfg.channel_type}")
    return torch.cat((torch.real(noisy_y), torch.imag(noisy_y)), dim=2) * torch.sqrt(pwr)


@torch.inference_mode()
def validate_jscc(
    encoder: nn.Module,
    decoder: nn.Module,
    channel: Channel,
    val_loader,
    cfg: TrainConfig,
) -> tuple[float, float]:
    encoder.eval()
    decoder.eval()
    device = cfg.device
    psnr_sum = 0.0
    loss_sum = 0.0
    seen = 0
    for imgs, _labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        feature, _ = encoder(imgs)
        dec_in = channel_to_decoder_input(feature, channel, cfg)
        recon = decoder(dec_in).clamp(0.0, 1.0)
        mse_i = (recon.float() - imgs.float()).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
        psnr_i = 10.0 * torch.log10(1.0 / mse_i)
        bsz = imgs.shape[0]
        psnr_sum += float(psnr_i.sum().item())
        loss_sum += float(mse_i.sum().item())
        seen += bsz
    return loss_sum / max(1, seen), psnr_sum / max(1, seen)


def train_jscc(
    cfg: TrainConfig,
    latest_every: int,
    val_every: int,
    init_encoder: str,
    init_decoder: str,
    show_progress: bool = False,
    print_header: bool = False,
) -> None:
    seed_torch(cfg.seed)
    device = cfg.device
    if device.type == "cuda":
        torch.cuda.set_device(0)

    encoder = JSCC_encoder(cfg, cfg.C).to(device)
    decoder = JSCC_decoder(cfg, cfg.C).to(device)
    channel = Channel(cfg)

    train_loader, val_loader = get_loader(cfg)
    if val_loader is None:
        raise RuntimeError("validation loader is required for best-checkpoint training")
    if init_encoder:
        load_state(encoder, init_encoder, "init JSCC encoder")
    if init_decoder:
        load_state(decoder, init_decoder, "init JSCC decoder")
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=cfg.learning_rate)
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=cfg.learning_rate)

    if print_header:
        print(f"=== CDDM JSCC C{cfg.C} train ===", flush=True)
        print(f"device={device} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
        print(
            f"dataset={cfg.dataset} train={len(train_loader.dataset)} valid={len(val_loader.dataset)} "
            f"batch={cfg.batch_size} test_batch={cfg.test_batch}",
            flush=True,
        )
        print(f"snr={cfg.SNRs:g} channel={cfg.channel_type} C={cfg.C} epochs={cfg.epoch}", flush=True)
        print(f"val_every={val_every} latest_every={latest_every}", flush=True)
        print(f"loss=F.mse_loss(clamp(recon,0,1), input, reduction='sum') / batch", flush=True)
        print("score=validation mean(per-image PSNR)", flush=True)
        print(f"encoder_path={cfg.encoder_path}", flush=True)
        print(f"decoder_path={cfg.decoder_path}", flush=True)

    best_psnr = -1.0
    for epoch in range(cfg.epoch):
        encoder.train()
        decoder.train()
        t0 = time.time()
        loss_sum = 0.0
        psnr_sum = 0.0
        n_seen = 0
        cbr_last = float("nan")

        iterator = tqdm(train_loader, dynamic_ncols=True) if show_progress else train_loader
        for imgs, _labels in iterator:
            imgs = imgs.to(device, non_blocking=True)
            feature, _ = encoder(imgs)
            noisy_y = channel_to_decoder_input(feature, channel, cfg)
            recon = decoder(noisy_y)

            mse255 = nn.MSELoss()(imgs * 255.0, recon.clamp(0.0, 1.0) * 255.0)
            loss = F.mse_loss(recon.clamp(0.0, 1.0), imgs, reduction="sum") / imgs.shape[0]
            psnr = 10 * (torch.log(255.0 * 255.0 / mse255) / np.log(10)).item()
            cbr_last = feature.numel() / 2 / imgs.numel()

            optimizer_encoder.zero_grad(set_to_none=True)
            optimizer_decoder.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_decoder.step()
            optimizer_encoder.step()

            bsz = imgs.shape[0]
            loss_sum += float(loss.item()) * bsz
            psnr_sum += float(psnr) * bsz
            n_seen += bsz
            if show_progress:
                iterator.set_postfix(
                    ordered_dict={
                        "epoch": epoch + 1,
                        "state": "trainMSE",
                        "SNR": cfg.SNRs,
                        "CBR": cbr_last,
                        "loss": float(loss.item()),
                        "psnr": psnr,
                    }
                )

        avg_loss = loss_sum / max(1, n_seen)
        avg_psnr = psnr_sum / max(1, n_seen)
        elapsed = time.time() - t0
        print(
            f"[epoch {epoch + 1:03d}/{cfg.epoch}] "
            f"loss={avg_loss:.6f} train_psnr={avg_psnr:.4f} cbr={cbr_last:.6f} time={elapsed:.1f}s",
            flush=True,
        )

        do_val = val_every > 0 and ((epoch + 1) % val_every == 0 or (epoch + 1) == cfg.epoch)
        if do_val:
            val_loss, val_psnr = validate_jscc(encoder, decoder, channel, val_loader, cfg)
            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
                save_pair(encoder, decoder, cfg)
            print(
                f"[epoch {epoch + 1:03d}/{cfg.epoch}] "
                f"val_mse={val_loss:.8f} val_psnr={val_psnr:.4f} "
                f"best_val_psnr={best_psnr:.4f} {'BEST' if is_best else ''}",
                flush=True,
            )

        if latest_every > 0 and (epoch + 1) % latest_every == 0:
            save_latest_pair(encoder, decoder, cfg)
    save_latest_pair(encoder, decoder, cfg)
    print(f"best_val_psnr={best_psnr:.4f}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CDDM Autoencoder JSCC on DIV2K with the original CDDM channel path.")
    p.add_argument("--snr-db", type=float, default=6.0)
    p.add_argument("--C", type=int, default=4)
    p.add_argument("--channel-type", choices=("awgn", "rayleigh"), default="awgn")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=600)
    p.add_argument("--latest-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--init-encoder", type=str, default="")
    p.add_argument("--init-decoder", type=str, default="")
    p.add_argument("--progress", action="store_true", help="show per-batch tqdm progress on stderr")
    p.add_argument("--print-header", action="store_true", help="print run configuration before epoch logs")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--ckpt-root", type=str, default=default_ckpt_root())
    p.add_argument("--train-dir", type=str, default="/workspace/yongjia/datasets/DIV2K/DIV2K_train_HR")
    p.add_argument("--val-dir", type=str, default="/workspace/yongjia/datasets/DIV2K/DIV2K_valid_HR")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    train_jscc(
        cfg,
        latest_every=int(args.latest_every),
        val_every=int(args.val_every),
        init_encoder=str(args.init_encoder),
        init_decoder=str(args.init_decoder),
        show_progress=bool(args.progress),
        print_header=bool(args.print_header),
    )


if __name__ == "__main__":
    main()
