from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from common import CDDM_ROOT, JSCCFConfig, resolve_path


ROUTE = "TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256"


def build_config(args: argparse.Namespace, batch_size: int | None = None, encoder_in_chans: int = 3) -> JSCCFConfig:
    return JSCCFConfig(
        C=int(getattr(args, "latent_ch", 16)),
        batch_size=int(args.batch_size if batch_size is None else batch_size),
        test_batch=int(args.test_batch),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.val_num_workers),
        train_data_dir=resolve_path(os.path.join(args.data_dir, "DIV2K_train_HR")),
        test_data_dir=resolve_path(os.path.join(args.data_dir, "DIV2K_valid_HR")),
        CUDA=(not bool(args.cpu)),
        encoder_in_chans=int(encoder_in_chans),
    )


def default_save_dir() -> str:
    return str(CDDM_ROOT / "MY-V2" / "jscc-f" / "checkpoints")


def default_init_ckpt() -> str:
    return str(CDDM_ROOT / "MY" / "jscc-no-awgn" / "cvq_v2_c16_stage0_best.pth")


def safe_artifact_name(value: str) -> str:
    text = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_")


def ckpt_path(args: argparse.Namespace, stage: str, suffix: str) -> str:
    version = safe_artifact_name(getattr(args, "version", "default"))
    version_part = f"_{version}" if version else ""
    return str(Path(resolve_path(args.save_dir)) / f"jscc_f{version_part}_{stage}_{suffix}.pth")


def load_state(module: nn.Module, state: dict[str, torch.Tensor], name: str, strict: bool = True) -> None:
    missing, unexpected = module.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"{name} load mismatch: missing={missing}, unexpected={unexpected}")
    print(f"loaded {name}: missing={missing} unexpected={unexpected}")


def load_module_checkpoint(module: nn.Module, path: str, name: str, strict: bool = True) -> None:
    if not path:
        return
    abs_path = resolve_path(path)
    obj = torch.load(abs_path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    load_state(module, state, name, strict=strict)
    print(f"loaded {name} checkpoint: {abs_path}")


def load_checkpoint(path: str) -> dict:
    abs_path = resolve_path(path)
    obj = torch.load(abs_path, map_location="cpu", weights_only=False)
    print(f"loaded checkpoint: {abs_path}")
    return obj


def load_layer1_compatible_checkpoint(path: str, e1: nn.Module, d1: nn.Module, strict: bool = True) -> dict:
    ckpt = load_checkpoint(path)
    if "e1_state_dict" in ckpt and "d1_state_dict" in ckpt:
        load_state(e1, ckpt["e1_state_dict"], "E1", strict=strict)
        load_state(d1, ckpt["d1_state_dict"], "D1", strict=strict)
        return ckpt
    if "encoder_state_dict" in ckpt and "decoder_state_dict" in ckpt:
        load_state(e1, ckpt["encoder_state_dict"], "E1", strict=strict)
        load_state(d1, ckpt["decoder_state_dict"], "D1", strict=strict)
        return ckpt
    raise KeyError(f"checkpoint does not contain layer1 states: {resolve_path(path)}")


def save_layer1_checkpoint(path: str, *, epoch: int, args: argparse.Namespace, metrics: dict[str, float], e1: nn.Module, d1: nn.Module) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": ROUTE,
            "stage": "layer1",
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
            "version": str(getattr(args, "version", "")),
            "e1_state_dict": e1.state_dict(),
            "d1_state_dict": d1.state_dict(),
            "latent": {"z1": [int(args.latent_ch), int(args.latent_h), int(args.latent_w)], "z2": None},
        },
        out,
    )
    print(f"saved checkpoint: {out}")


def save_layer2_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    e1: nn.Module,
    d1: nn.Module,
    e2: nn.Module,
    d2: nn.Module,
    combiner: nn.Module,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": ROUTE,
            "stage": "layer2",
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
            "version": str(getattr(args, "version", "")),
            "e1_state_dict": e1.state_dict(),
            "d1_state_dict": d1.state_dict(),
            "e2_state_dict": e2.state_dict(),
            "d2_state_dict": d2.state_dict(),
            "combiner_state_dict": combiner.state_dict(),
            "variant": str(args.variant),
            "latent": {"z1": [int(args.latent_ch), int(args.latent_h), int(args.latent_w)], "z2": [20, int(args.latent_h), int(args.latent_w)]},
        },
        out,
    )
    print(f"saved checkpoint: {out}")


def save_layer3_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    e1: nn.Module,
    d1: nn.Module,
    e2: nn.Module,
    d2: nn.Module,
    combiner: nn.Module,
    quantizer: nn.Module,
    indexnet: nn.Module,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    quantizer_type = str(getattr(args, "quantizer", "simvq"))
    if quantizer_type == "vq":
        quantizer_codes = int(getattr(args, "vq_k", 64))
    elif quantizer_type == "fsq":
        fsq_text = str(getattr(args, "fsq_levels", "15")).replace("x", ",")
        fsq_levels = [int(part.strip()) for part in fsq_text.split(",") if part.strip()]
        if len(fsq_levels) == 1:
            fsq_levels = fsq_levels * 20
        quantizer_codes = max(fsq_levels)
    elif quantizer_type == "cvq":
        quantizer_codes = int(getattr(args, "cvq_k", 2048))
    elif quantizer_type == "fullmap_simvq":
        quantizer_codes = int(getattr(args, "fullmap_simvq_k", 2048))
    else:
        quantizer_codes = int(getattr(args, "simvq_k", 64))
    is_fullmap = quantizer_type in {"cvq", "fullmap_simvq"}
    if quantizer_type == "vq":
        train_codebook = True
    elif quantizer_type == "fsq":
        train_codebook = not bool(getattr(args, "fsq_freeze_affine", False))
    elif quantizer_type == "simvq":
        train_codebook = bool(getattr(args, "simvq_train_codebook", False))
    elif quantizer_type == "fullmap_simvq":
        train_codebook = bool(getattr(args, "fullmap_simvq_train_codebook", False))
    else:
        train_codebook = True
    payload = {
        "route": ROUTE,
        "stage": f"stage3_{quantizer_type}_oracle_index",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "version": str(getattr(args, "version", "")),
        "e1_state_dict": e1.state_dict(),
        "d1_state_dict": d1.state_dict(),
        "e2_state_dict": e2.state_dict(),
        "d2_state_dict": d2.state_dict(),
        "combiner_state_dict": combiner.state_dict(),
        "quantizer_state_dict": quantizer.state_dict(),
        "indexnet_state_dict": indexnet.state_dict(),
        "quantizer": {
            "type": quantizer_type,
            "num_codes": quantizer_codes,
            "embedding_dim": 20 if quantizer_type in {"vq", "simvq", "fsq"} else None,
            "codebook_shape": [quantizer_codes, int(args.latent_h), int(args.latent_w)] if is_fullmap else None,
            "tokens": 20 if is_fullmap else int(args.latent_h) * int(args.latent_w),
            "fsq_levels": fsq_levels if quantizer_type == "fsq" else None,
            "fsq_init_stats": bool(getattr(args, "fsq_init_stats", True)) if quantizer_type == "fsq" else None,
            "fsq_init_quantile": float(getattr(args, "fsq_init_quantile", 0.001)) if quantizer_type == "fsq" else None,
            "beta_commit": float(args.beta_commit),
            "train_codebook": train_codebook,
            "init_codebook_method": str(getattr(args, "init_codebook_method", "none")),
            "codebook_init_samples": int(getattr(args, "codebook_init_samples", 0)),
            "kmeans_iters": int(getattr(args, "kmeans_iters", 0)),
        },
        "latent": {
            "z1": [int(args.latent_ch), int(args.latent_h), int(args.latent_w)],
            "z2": [20, int(args.latent_h), int(args.latent_w)],
            "total": [int(args.latent_ch) + 20, int(args.latent_h), int(args.latent_w)],
        },
    }
    if quantizer_type in {"simvq", "fullmap_simvq"}:
        payload["simvq_state_dict"] = quantizer.state_dict()
    torch.save(
        payload,
        out,
    )
    print(f"saved checkpoint: {out}")
