"""Adapters for the existing JSCC-f CNN/Swin Layer1 checkpoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import Tensor, nn


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
CDDM_ROOT = JSCCF_DIR.parents[1]

DEFAULT_LAYER1_CKPT = {
    "cnn": "MY-V2/jscc-f/checkpoints/jscc_f_cnn_layer1_cnn_best.pth",
    "swin": "MY-V2/jscc-f/checkpoints/jscc_f_no-c1_layer1_best.pth",
}


def _load_module(name: str, path: Path):
    if str(JSCCF_DIR) not in sys.path:
        sys.path.insert(0, str(JSCCF_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _jscc_modules():
    return (
        _load_module("jsccf_codeformer_io", JSCCF_DIR / "io.py"),
        _load_module("jsccf_codeformer_model", JSCCF_DIR / "model.py"),
        _load_module("jsccf_codeformer_test_ed", JSCCF_DIR / "test_ed.py"),
    )


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else CDDM_ROOT / path


def _encode(encoder: nn.Module, image: Tensor) -> Tensor:
    output = encoder(image)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Layer1 encoder returned {type(output)!r}, expected Tensor")
    return output


def build_layer1(args, device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Build the exact architecture required by an existing Layer1 checkpoint."""
    _io, jsccf_model, test_ed = _jscc_modules()
    if args.layer1_arch == "swin":
        return jsccf_model.build_layer1(args, device)
    if args.layer1_arch == "cnn":
        encoder = test_ed.CNNAnalysisEncoder(
            base_ch=int(args.layer1_cnn_base_ch),
            bottleneck_ch=int(args.latent_ch),
            num_res=int(args.layer1_cnn_num_res),
        ).to(device)
        decoder = test_ed.CNNBottleneckDecoder(
            base_ch=int(args.layer1_cnn_base_ch),
            bottleneck_ch=int(args.latent_ch),
            num_res=int(args.layer1_cnn_num_res),
            output_activation="none",
        ).to(device)
        return encoder, decoder
    raise ValueError(f"unsupported --layer1-arch={args.layer1_arch!r}")


def load_layer1_checkpoint(args, encoder: nn.Module, decoder: nn.Module) -> Path:
    path = _resolve(args.layer1_ckpt or DEFAULT_LAYER1_CKPT[str(args.layer1_arch)])
    if not path.is_file():
        raise FileNotFoundError(f"Layer1 checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Layer1 checkpoint must be a dict, got {type(checkpoint)!r}")
    encoder_state = checkpoint.get("e1_state_dict", checkpoint.get("encoder_state_dict"))
    decoder_state = checkpoint.get("d1_state_dict", checkpoint.get("decoder_state_dict"))
    if encoder_state is None or decoder_state is None:
        raise KeyError("Layer1 checkpoint needs e1/d1_state_dict or encoder/decoder_state_dict")
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Layer1 encoder mismatch: missing={missing} unexpected={unexpected}")
    missing, unexpected = decoder.load_state_dict(decoder_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Layer1 decoder mismatch: missing={missing} unexpected={unexpected}")
    return path


def freeze_layer1(encoder: nn.Module, decoder: nn.Module) -> None:
    for module in (encoder, decoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)


@torch.no_grad()
def layer1_x1(encoder: nn.Module, decoder: nn.Module, image: Tensor) -> Tensor:
    """Return the only degradation input to CodeFormer: Layer1's clipped x1."""
    z1 = _encode(encoder, image)
    x1 = decoder(z1)
    if tuple(x1.shape[1:]) != (3, 256, 256):
        raise RuntimeError(f"Layer1 must reconstruct [B,3,256,256], got {tuple(x1.shape)}")
    return x1.clamp(0.0, 1.0)
