"""Isolated adapters for the existing JSCC-f CNN/Swin Layer1 checkpoints.

``codeformer-v2/model.py`` intentionally has the conventional module name
``model``.  The historical JSCC-f ``test_ed.py`` imports its own sibling
``model.py`` by that name, so loading it needs a short-lived module alias to
avoid accidentally importing CodeFormer-v2's model.
"""

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


def _exec_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _jscc_modules():
    if str(JSCCF_DIR) in sys.path:
        sys.path.remove(str(JSCCF_DIR))
    sys.path.insert(0, str(JSCCF_DIR))
    io_module = _exec_module("jsccf_codeformer_v2_io", JSCCF_DIR / "io.py")
    model_module = _exec_module("jsccf_codeformer_v2_jscc_model", JSCCF_DIR / "model.py")

    # ``test_ed.py`` has ``from model import ...``.  Let it bind to the old
    # JSCC-f module only while it executes, then restore CodeFormer-v2's alias.
    prior_model = sys.modules.get("model")
    try:
        sys.modules.pop("model", None)
        _exec_module("model", JSCCF_DIR / "model.py")
        test_ed = _exec_module("jsccf_codeformer_v2_test_ed", JSCCF_DIR / "test_ed.py")
    finally:
        sys.modules.pop("model", None)
        if prior_model is not None:
            sys.modules["model"] = prior_model
    return io_module, model_module, test_ed


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
    e1_state = checkpoint.get("e1_state_dict", checkpoint.get("encoder_state_dict"))
    d1_state = checkpoint.get("d1_state_dict", checkpoint.get("decoder_state_dict"))
    if e1_state is None or d1_state is None:
        raise KeyError("Layer1 checkpoint needs E1/D1 or encoder/decoder state dicts")
    missing, unexpected = encoder.load_state_dict(e1_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Layer1 encoder mismatch: missing={missing} unexpected={unexpected}")
    missing, unexpected = decoder.load_state_dict(d1_state, strict=True)
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
    z1 = _encode(encoder, image)
    x1 = decoder(z1)
    if tuple(x1.shape[1:]) != (3, 256, 256):
        raise RuntimeError(f"Layer1 must reconstruct [B,3,256,256], got {tuple(x1.shape)}")
    return x1.clamp(0.0, 1.0)
