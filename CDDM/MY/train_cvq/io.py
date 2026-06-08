from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from Autoencoder.net.network import JSCC_decoder, JSCC_encoder

from .common import CDDM_ROOT, build_config, prefix16_norm_all, real_awgn, resolve_path
from .models import FSQFactorizedCAR, FSQSpatialCAR, TailCAR, TailCVQ

def load_state(module: nn.Module, state: dict[str, torch.Tensor], name: str, strict: bool = True) -> None:
    missing, unexpected = module.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"{name} load mismatch: missing={missing}, unexpected={unexpected}")
    print(f"loaded {name}: missing={missing} unexpected={unexpected}")

def load_module_checkpoint(module: nn.Module, path: str, name: str, strict: bool = True) -> None:
    if not path:
        return
    ckpt_path = resolve_path(path)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    load_state(module, state, name, strict=strict)
    print(f"loaded {name} checkpoint: {ckpt_path}")

def load_experiment_checkpoint(
    path: str,
    *,
    encoder: nn.Module | None = None,
    decoder: nn.Module | None = None,
    cvq: TailCVQ | None = None,
    car: nn.Module | None = None,
    strict: bool = True,
) -> dict:
    ckpt_path = resolve_path(path)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if encoder is not None and "encoder_state_dict" in obj:
        load_state(encoder, obj["encoder_state_dict"], "encoder", strict=strict)
    if decoder is not None and "decoder_state_dict" in obj:
        load_state(decoder, obj["decoder_state_dict"], "decoder", strict=strict)
    if cvq is not None and "cvq_state_dict" in obj:
        load_state(cvq, obj["cvq_state_dict"], "cvq", strict=strict)
    if car is not None and "car_state_dict" in obj:
        load_state(car, obj["car_state_dict"], "car", strict=strict)
    print(f"loaded checkpoint: {ckpt_path}")
    return obj

def save_checkpoint(
    path: str,
    *,
    stage: str,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    encoder: nn.Module,
    decoder: nn.Module,
    cvq: TailCVQ | None = None,
    car: nn.Module | None = None,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "route": f"cvq_tail_c{int(args.latent_ch)}_prefix{int(args.prefix_ch)}",
        "stage": stage,
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "snr_db": float(args.snr_db),
        "latent_ch": int(args.latent_ch),
        "prefix_ch": int(args.prefix_ch),
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
    }
    if cvq is not None:
        payload["cvq_state_dict"] = cvq.state_dict()
    if car is not None:
        payload["car_state_dict"] = car.state_dict()
    torch.save(payload, out)
    print(f"saved checkpoint: {out}")

def default_save_dir() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints-cvq")

def default_jscc_encoder_c36() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints-jscc" / "encoder_snr9_channel_awgn_C36.pt")

def default_jscc_decoder_c36() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints-jscc" / "decoder_snr9_channel_awgn_C36.pt")

def ckpt_path(args: argparse.Namespace, stage: str, suffix: str) -> str:
    return str(Path(resolve_path(args.save_dir)) / f"cvq_c{int(args.latent_ch)}_snr{args.snr_db:g}_{stage}_{suffix}.pth")

def build_models(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module, TailCVQ, nn.Module]:
    cfg = build_config(args)
    encoder = JSCC_encoder(cfg, int(args.latent_ch)).to(device)
    decoder = JSCC_decoder(cfg, int(args.latent_ch)).to(device)
    cvq = TailCVQ(
        h=int(args.latent_h),
        w=int(args.latent_w),
        tail_ch=int(args.latent_ch) - int(args.prefix_ch),
        k_a=int(args.k_a),
        k_b=int(args.k_b),
        beta=float(args.vq_beta),
        chunk_size=int(args.vq_chunk_size),
        mode=str(args.cvq_mode),
        rvq_stages=int(args.rvq_stages),
        patch_size=int(args.patch_size),
        fsq_levels_a=int(args.fsq_levels_a),
        fsq_levels_b=int(args.fsq_levels_b),
        fsq_scale=float(args.fsq_scale),
    ).to(device)
    car_arch = str(getattr(args, "car_arch", "auto"))
    if car_arch == "auto":
        car_arch = "swin" if str(args.cvq_mode) == "fsq" else "legacy"
    if car_arch == "swin":
        if str(args.cvq_mode) != "fsq":
            raise ValueError("--car-arch swin is currently implemented for --cvq-mode fsq only")
        car = FSQSpatialCAR(
            h=int(args.latent_h),
            w=int(args.latent_w),
            tail_ch=int(args.latent_ch) - int(args.prefix_ch),
            levels_a=int(args.fsq_levels_a),
            levels_b=int(args.fsq_levels_b),
            prefix_ch=int(args.prefix_ch),
            d_model=int(args.car_dim),
            nhead=int(args.car_heads),
            layers=int(args.car_layers),
            dropout=float(args.car_dropout),
            window_size=int(args.car_window_size),
            prefix_image_cond=bool(getattr(args, "car_prefix_image_cond", False)),
            prefix_image_scale_init=float(getattr(args, "car_prefix_image_scale_init", 0.1)),
        ).to(device)
    elif car_arch == "factorized_swin":
        if str(args.cvq_mode) != "fsq":
            raise ValueError("--car-arch factorized_swin is currently implemented for --cvq-mode fsq only")
        car = FSQFactorizedCAR(
            h=int(args.latent_h),
            w=int(args.latent_w),
            tail_ch=int(args.latent_ch) - int(args.prefix_ch),
            levels_a=int(args.fsq_levels_a),
            levels_b=int(args.fsq_levels_b),
            prefix_ch=int(args.prefix_ch),
            d_model=int(args.car_dim),
            nhead=int(args.car_heads),
            layers=int(args.car_layers),
            dropout=float(args.car_dropout),
            window_size=int(args.car_window_size),
            prefix_image_cond=bool(getattr(args, "car_prefix_image_cond", False)),
            prefix_image_scale_init=float(getattr(args, "car_prefix_image_scale_init", 0.1)),
        ).to(device)
    else:
        car = TailCAR(
            h=int(args.latent_h),
            w=int(args.latent_w),
            tail_ch=int(args.latent_ch) - int(args.prefix_ch),
            k_a=int(args.k_a),
            k_b=int(args.k_b),
            prefix_ch=int(args.prefix_ch),
            d_model=int(args.car_dim),
            nhead=int(args.car_heads),
            layers=int(args.car_layers),
            dropout=float(args.car_dropout),
        ).to(device)
    return encoder, decoder, cvq, car

def forward_parts(
    imgs: torch.Tensor,
    encoder: nn.Module,
    args: argparse.Namespace,
    *,
    noisy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z, _ = encoder(imgs)
    if z.shape[1] != int(args.latent_ch):
        raise RuntimeError(f"expected latent channels={args.latent_ch}, got {z.shape[1]}")
    s, _scale = prefix16_norm_all(z.float(), prefix_ch=int(args.prefix_ch))
    prefix = s[:, : int(args.prefix_ch)]
    tail = s[:, int(args.prefix_ch) :]
    y_prefix = real_awgn(prefix, float(args.snr_db)) if noisy else prefix
    return z, s, y_prefix, tail
