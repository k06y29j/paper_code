"""CodeFormer-v2 topology.

Stage1 learns an HQ residual codec from ``concat(img, x1)``.  Stage2 receives
only ``x1`` at the model boundary, looks up the frozen Stage1 codebook, decodes
``u2``, and forms ``x2_hat = combiner(concat(x1, u2))``.  The Stage2 trainable
set is exactly the LQ encoder plus the decoder's SFT/CFT blocks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import Tensor, nn


THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent / "codeformer"


def _load_base_architecture():
    spec = importlib.util.spec_from_file_location(
        "jsccf_codeformer_v2_base_architecture", BASE_DIR / "architecture.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_DIR / 'architecture.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base_architecture()
IMAGE_SIZE = _base.IMAGE_SIZE
LATENT_SIZE = _base.LATENT_SIZE
LATENT_CHANNELS = _base.LATENT_CHANNELS
FUSE_RESOLUTIONS = _base.FUSE_RESOLUTIONS
ImageEncoder = _base.ImageEncoder
ImageDecoder = _base.ImageDecoder
VectorQuantizer = _base.VectorQuantizer


class ConditionalEncoder(ImageEncoder):
    """The shared 256→16 backbone with an explicitly selected input contract."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int = 64,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__(base_channels, LATENT_CHANNELS, num_res_blocks)
        self.input_channels = int(input_channels)
        old_stem = self.input_conv
        self.input_conv = nn.Conv2d(
            self.input_channels,
            old_stem.out_channels,
            old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None,
        )

    def forward(self, image: Tensor, return_features: bool = False):
        expected = (self.input_channels, IMAGE_SIZE, IMAGE_SIZE)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected:
            raise ValueError(f"expected image [B,{expected[0]},256,256], got {tuple(image.shape)}")
        h = self.input_conv(image)
        features: dict[int, Tensor] = {}
        for level, stage in enumerate(self.stages):
            h = stage(h)
            resolution = self.resolutions[level]
            if resolution in FUSE_RESOLUTIONS:
                features[resolution] = h
            if level < len(self.downsample):
                h = self.downsample[level](h)
        latent = self.output(self.mid(h))
        if tuple(latent.shape[1:]) != (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE):
            raise RuntimeError(f"encoder contract broke: got {tuple(latent.shape)}")
        return (latent, features) if return_features else latent


class OutputsCombiner(nn.Module):
    """The required [x1,u2] six-channel image combiner."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 48, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(48, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x1: Tensor, u2: Tensor) -> Tensor:
        if tuple(x1.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected x1 [B,3,256,256], got {tuple(x1.shape)}")
        if tuple(u2.shape) != tuple(x1.shape):
            raise ValueError(f"u2 must match x1, got u2={tuple(u2.shape)} x1={tuple(x1.shape)}")
        return self.net(torch.cat([x1, u2], dim=1))


class Stage1HQCodec(nn.Module):
    """Stage1: ``[img,x1] -> E_HQ -> VQ -> D -> u2 -> combiner -> x2``."""

    def __init__(
        self,
        codebook_size: int = 1024,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.hq_encoder = ConditionalEncoder(6, base_channels, num_res_blocks)
        self.quantizer = VectorQuantizer(codebook_size, LATENT_CHANNELS, beta)
        self.decoder = ImageDecoder(base_channels, num_res_blocks)
        self.combiner = OutputsCombiner()

    def forward(self, image: Tensor, x1: Tensor) -> dict[str, Tensor]:
        hq_input = torch.cat([image, x1], dim=1)
        z2 = self.hq_encoder(hq_input)
        q2_st, q2, indices, stats = self.quantizer(z2)
        u2 = self.decoder(q2_st)
        x2 = self.combiner(x1, u2)
        return {
            "hq_input": hq_input,
            "z2": z2,
            "q2": q2,
            "indices": indices,
            "u2": u2,
            "x2": x2,
            **stats,
        }


class Stage2LQRestorer(nn.Module):
    """Stage2 deployment module; its public forward input is x1 only."""

    def __init__(
        self,
        codebook_size: int = 1024,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.lq_encoder = ConditionalEncoder(3, base_channels, num_res_blocks)
        self.quantizer = VectorQuantizer(codebook_size, LATENT_CHANNELS, beta)
        self.decoder = ImageDecoder(base_channels, num_res_blocks)
        self.combiner = OutputsCombiner()

    @torch.no_grad()
    def initialize_from_stage1(self, payload: dict) -> dict[str, int]:
        required = {
            "hq_encoder_state_dict",
            "quantizer_state_dict",
            "decoder_state_dict",
            "combiner_state_dict",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise KeyError(f"Stage1 checkpoint misses {missing}")
        source = payload["hq_encoder_state_dict"]
        target = self.lq_encoder.state_dict()
        copied = {
            key: value
            for key, value in source.items()
            if key in target and tuple(value.shape) == tuple(target[key].shape)
        }
        # Stage1 concatenates [img,x1].  The LQ encoder's stem is initialized
        # from the x1 branch (channels 3:6), never from the privileged img part.
        stem_key = "input_conv.weight"
        source_stem = source.get(stem_key)
        if source_stem is None or source_stem.shape[1] != 6:
            raise RuntimeError("Stage1 HQ encoder must have a 6-channel input stem")
        copied[stem_key] = source_stem[:, 3:6].contiguous()
        if "input_conv.bias" in source:
            copied["input_conv.bias"] = source["input_conv.bias"]
        missing_keys, unexpected_keys = self.lq_encoder.load_state_dict(copied, strict=False)
        if unexpected_keys:
            raise RuntimeError(f"unexpected LQ encoder initialization keys: {unexpected_keys}")
        q_missing, q_unexpected = self.quantizer.load_state_dict(payload["quantizer_state_dict"], strict=True)
        d_missing, d_unexpected = self.decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
        c_missing, c_unexpected = self.combiner.load_state_dict(payload["combiner_state_dict"], strict=True)
        if q_missing or q_unexpected or d_missing or d_unexpected or c_missing or c_unexpected:
            raise RuntimeError("strict Stage1 component loading failed")
        return {
            "lq_encoder_copied_tensors": len(copied),
            "lq_encoder_total_tensors": len(target),
            "lq_encoder_missing_tensors": len(missing_keys),
        }

    def freeze_stage1_modules(self) -> None:
        """Freeze codebook/base decoder/combiner; leave LQ encoder and CFT live."""
        for parameter in self.quantizer.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        for block in self.decoder.fuse_blocks.values():
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.combiner.parameters():
            parameter.requires_grad_(False)
        for parameter in self.lq_encoder.parameters():
            parameter.requires_grad_(True)

    def forward(self, x1: Tensor, fusion_weight: float = 1.0) -> dict[str, Tensor]:
        """Deployment path: does not accept image, HQ codes, or sender latent."""
        z2, lq_features = self.lq_encoder(x1, return_features=True)
        q2_st, q2, indices, stats = self.quantizer(z2)
        u2 = self.decoder(q2_st, lq_features, fusion_weight=float(fusion_weight))
        x2_hat = self.combiner(x1, u2)
        return {
            "z2": z2,
            "q2": q2,
            "indices": indices,
            "u2": u2,
            "x2_hat": x2_hat,
            **stats,
        }


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]
