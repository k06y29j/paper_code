from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from Autoencoder.net.network import JSCC_decoder, JSCC_encoder

from common import CDDMJSCCConfig, CDDM_ROOT, prefix_power_normalize, resolve_path
from model import EMAFullChannelQuantizer, FullChannelQuantizer, ScaledWhitenedVQ, SimVQFullChannelQuantizer


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


def default_save_dir() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints-cvq-v2-c36-snr12-k16384")


def default_jscc_encoder_c36_snr12() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints-jscc" / "encoder_snr12_channel_awgn_C16.pt")


def default_jscc_decoder_c36_snr12() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints-jscc" / "decoder_snr12_channel_awgn_C16.pt")


def ckpt_path(args: argparse.Namespace, stage: str, suffix: str) -> str:
    return str(Path(resolve_path(args.save_dir)) / f"cvq_v2_c{int(args.latent_ch)}_snr{args.snr_db:g}_{stage}_{suffix}.pth")


def load_state(module: nn.Module, state: dict[str, torch.Tensor], name: str, strict: bool = True) -> None:
    missing, unexpected = module.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"{name} load mismatch: missing={missing}, unexpected={unexpected}")
    print(f"loaded {name}: missing={missing} unexpected={unexpected}")


def load_module_checkpoint(module: nn.Module, path: str, name: str, strict: bool = True) -> None:
    if not path:
        return
    ckpt_path_abs = resolve_path(path)
    obj = torch.load(ckpt_path_abs, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    load_state(module, state, name, strict=strict)
    print(f"loaded {name} checkpoint: {ckpt_path_abs}")


def load_experiment_checkpoint(
    path: str,
    *,
    encoder: nn.Module | None = None,
    decoder: nn.Module | None = None,
    quantizer: FullChannelQuantizer | None = None,
    strict: bool = True,
) -> dict:
    ckpt_path_abs = resolve_path(path)
    obj = torch.load(ckpt_path_abs, map_location="cpu", weights_only=False)
    if encoder is not None and "encoder_state_dict" in obj:
        load_state(encoder, obj["encoder_state_dict"], "encoder", strict=strict)
    if decoder is not None and "decoder_state_dict" in obj:
        load_state(decoder, obj["decoder_state_dict"], "decoder", strict=strict)
    if quantizer is not None and "quantizer_state_dict" in obj:
        load_state(quantizer, obj["quantizer_state_dict"], "quantizer", strict=strict)
    print(f"loaded experiment checkpoint: {ckpt_path_abs}")
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
    quantizer: FullChannelQuantizer,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "route": "cvq_v2_c36_c1_16_c2_20_k16384",
        "stage": stage,
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "snr_db": float(args.snr_db),
        "latent_ch": int(args.latent_ch),
        "c1_ch": int(args.c1_ch),
        "k": int(args.k),
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "quantizer_state_dict": quantizer.state_dict(),
    }
    torch.save(payload, out)
    print(f"saved checkpoint: {out}")


def build_models(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module]:
    cfg = build_config(args)
    encoder = JSCC_encoder(cfg, int(args.latent_ch)).to(device)
    decoder = JSCC_decoder(cfg, int(args.latent_ch)).to(device)
    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    vq_mode = str(getattr(args, "stage1_vq_mode", "plain")).lower()
    codebook_update = str(getattr(args, "stage1_codebook_update", "grad")).lower()
    if vq_mode == "scaled_whitened":
        quantizer = ScaledWhitenedVQ(
            num_codes=int(args.k),
            channels=c2_ch,
            h=int(args.latent_h),
            w=int(args.latent_w),
            beta=float(args.vq_beta),
            chunk_size=int(args.vq_chunk_size),
            eps=float(getattr(args, "scaled_vq_eps", 1e-6)),
            codebook_update=codebook_update,
            ema_decay=float(getattr(args, "vq_ema_decay", 0.99)),
            ema_eps=float(getattr(args, "vq_ema_eps", 1e-5)),
            ema_initial_count=float(getattr(args, "vq_ema_initial_count", 1.0)),
        ).to(device)
    elif vq_mode == "plain":
        quantizer_cls = EMAFullChannelQuantizer if codebook_update == "ema" else FullChannelQuantizer
        if codebook_update not in {"grad", "ema"}:
            raise ValueError(f"unknown --stage1-codebook-update {codebook_update!r}")
        kwargs = dict(
            num_codes=int(args.k),
            h=int(args.latent_h),
            w=int(args.latent_w),
            beta=float(args.vq_beta),
            chunk_size=int(args.vq_chunk_size),
        )
        if codebook_update == "ema":
            kwargs.update(
                ema_decay=float(getattr(args, "vq_ema_decay", 0.99)),
                ema_eps=float(getattr(args, "vq_ema_eps", 1e-5)),
                ema_initial_count=float(getattr(args, "vq_ema_initial_count", 1.0)),
            )
        quantizer = quantizer_cls(**kwargs).to(device)
    elif vq_mode == "simvq":
        quantizer = SimVQFullChannelQuantizer(
            num_codes=int(args.k),
            h=int(args.latent_h),
            w=int(args.latent_w),
            beta=float(args.vq_beta),
            chunk_size=int(args.vq_chunk_size),
            codebook_std=float(getattr(args, "simvq_codebook_std", 0.0)),
            freeze_codebook=not bool(getattr(args, "simvq_train_codebook", False)),
            legacy=bool(getattr(args, "simvq_legacy", True)),
            proj_bias=bool(getattr(args, "simvq_proj_bias", True)),
        ).to(device)
    else:
        raise ValueError(f"unknown --stage1-vq-mode {vq_mode!r}")
    return encoder, decoder, quantizer


def encode_normalized(
    imgs: torch.Tensor,
    encoder: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z, _ = encoder(imgs)
    if z.shape[1] != int(args.latent_ch):
        raise RuntimeError(f"expected latent channels={args.latent_ch}, got {z.shape[1]}")
    z_norm, c1_power = prefix_power_normalize(z.float(), prefix_ch=int(args.c1_ch))
    return z, z_norm, c1_power
