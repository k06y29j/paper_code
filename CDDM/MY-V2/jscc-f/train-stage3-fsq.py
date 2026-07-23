from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


THIS_DIR = Path(__file__).resolve().parent
CDDM_ROOT = THIS_DIR.parents[1]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from Autoencoder.data.datasets import get_loader

from common import (
    averaged,
    batch_metric_mean,
    check_jsccf_args,
    meters,
    mse_per_image,
    psnr_per_image,
    recon_loss,
    resolve_path,
    seed_everything,
    setup_log_file,
    should_save_latest,
    should_validate,
    ssim_per_image,
    write_json,
)
from model import OutputsCombiner, build_jscc_decoder, build_jscc_encoder, build_layer1
from test_ed import CNNAnalysisEncoder, CNNBottleneckDecoder, ConvNormAct


FSQ_LEVEL = 2
DEFAULT_LAYER1_CKPTS = {
    "swin": "MY-V2/jscc-f/checkpoints/jscc_f_no-c1_layer1_best.pth",
    "cnn": "MY-V2/jscc-f/checkpoints/jscc_f_cnn_layer1_cnn_best.pth",
}
DEFAULT_LAYER2_CKPTS = {
    "swin": (
        "MY-V2/jscc-f/checkpoints/"
        "jscc_f_swin320_layer2_swin_no_compressor_combiner_best.pth"
    ),
    "cnn": (
        "MY-V2/jscc-f/checkpoints/"
        "jscc_f_cnn-stage2-compressor-gpu2_layer2_v2_compressor_combiner_best.pth"
    ),
    "bar": "",
}
DEFAULT_SAVE_DIR = str(CDDM_ROOT / "MY-V2" / "jscc-f" / "checkpoints-fsq-c16")
BAR_SOURCE_URL = "https://github.com/amazon-far/BAR"


METRIC_NAMES = [
    "loss",
    "loss_final",
    "mse_x1",
    "psnr_x1",
    "ssim_x1",
    "mse_u2_as_img",
    "psnr_u2_as_img",
    "mse_final",
    "psnr_final",
    "ssim_final",
    "delta_x1",
    "z2_abs_mean",
    "q2_abs_mean",
    "fsq_mse",
]

VAL_ABLATION_METRICS = [
    "psnr_code0",
    "psnr_shuffle",
    "drop_code0",
    "drop_shuffle",
]

DISPLAY_METRICS = [
    "loss",
    "loss_final",
    "psnr_x1",
    "psnr_final",
    "delta_x1",
    "code_used",
    "code_entropy_bits",
    "code_perplexity",
    "code_usage_ratio",
    "code_coverage_of_sample_ceiling",
    "bit_one_frac_mean",
    "bit_one_frac_min",
    "bit_one_frac_max",
    "bit_entropy_bits_mean",
    "bit_dead_channels",
    "empirical_bpp",
    "marginal_entropy_bpp",
    "psnr_code0",
    "psnr_shuffle",
    "drop_code0",
    "drop_shuffle",
]


def load_local_io() -> ModuleType:
    spec = importlib.util.spec_from_file_location("jsccf_stage3_fsq_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = load_local_io()


def parse_fsq_levels(
    levels: str | int | list[int] | tuple[int, ...], channels: int
) -> list[int]:
    if isinstance(levels, int):
        parsed = [int(levels)]
    elif isinstance(levels, str):
        parts = [part.strip() for part in levels.replace("x", ",").split(",") if part.strip()]
        parsed = [int(part) for part in parts]
    else:
        parsed = [int(value) for value in levels]
    if len(parsed) == 1:
        parsed *= int(channels)
    if len(parsed) != int(channels):
        raise ValueError(
            f"expected one FSQ level or {int(channels)} levels, got {parsed}"
        )
    if min(parsed) < 2:
        raise ValueError(f"FSQ levels must be >= 2, got {parsed}")
    return parsed


def vocab_size(levels: list[int]) -> int:
    size = 1
    for level in levels:
        size *= int(level)
    return int(size)


def round_ste(value: torch.Tensor) -> torch.Tensor:
    return value + (value.round() - value).detach()


def encode_tensor(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    out = module(value)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"encoder returned unsupported type {type(out)!r}")
    return out


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(bool(trainable))
    module.train(bool(trainable))


def trainable_state(module: nn.Module) -> str:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    if total == 0:
        return "no_params"
    if trainable == 0:
        return "frozen"
    if trainable == total:
        return "trainable"
    return f"partial_trainable({100.0 * trainable / float(total):.1f}%)"


class IFSQQuantizer(nn.Module):
    """Binary scalar FSQ with a packed-bit representation for arbitrary C."""

    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        channels: int,
        use_pre_norm: bool = True,
    ) -> None:
        super().__init__()
        parsed = parse_fsq_levels(levels, int(channels))
        self.channels = int(channels)
        self.register_buffer("levels", torch.tensor(parsed, dtype=torch.long))
        if parsed != [FSQ_LEVEL] * self.channels:
            raise ValueError("this entrypoint requires binary FSQ ([2] x C)")

        # A scalar int64 joint index is useful for legacy C<=63 artifacts only.
        # Packed bytes are canonical and continue to work for BAR-scale C=64..256.
        if self.channels <= 63:
            multipliers = [1 << (self.channels - channel - 1) for channel in range(self.channels)]
        else:
            multipliers = []
        self.register_buffer("multipliers", torch.tensor(multipliers, dtype=torch.long))
        self.pre_norm = (
            nn.GroupNorm(1, self.channels, affine=True)
            if bool(use_pre_norm)
            else nn.Identity()
        )

    @property
    def vocab_size(self) -> int:
        return vocab_size([int(value) for value in self.levels.detach().cpu().tolist()])

    @property
    def supports_scalar_indices(self) -> bool:
        return self.channels <= 63

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        self._check_codes(codes)
        if not self.supports_scalar_indices:
            raise ValueError(
                f"C={self.channels} cannot be represented by one signed int64 index; "
                "use packed binary codes"
            )
        multipliers = self.multipliers.to(device=codes.device).view(
            1, self.channels, 1, 1
        )
        return (codes.long() * multipliers).sum(dim=1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        if not self.supports_scalar_indices:
            raise ValueError(
                f"C={self.channels} cannot be decoded from one signed int64 index"
            )
        if indices.ndim != 3:
            raise ValueError(f"expected FSQ indices [B,H,W], got {tuple(indices.shape)}")
        if indices.numel() > 0:
            index_min = int(indices.detach().min().item())
            index_max = int(indices.detach().max().item())
            if index_min < 0 or index_max >= self.vocab_size:
                raise ValueError(
                    f"FSQ indices must be in [0,{self.vocab_size}), "
                    f"got min={index_min} max={index_max}"
                )
        values = indices.long().unsqueeze(1)
        multipliers = self.multipliers.to(device=indices.device).view(
            1, self.channels, 1, 1
        )
        levels = self.levels.to(device=indices.device).view(
            1, self.channels, 1, 1
        )
        return torch.div(values, multipliers, rounding_mode="floor").remainder(levels)

    def indices_to_quantized(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codes_to_quantized(self.indices_to_codes(indices))

    def pack_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Pack channels little-endian into [B,ceil(C/8),H,W] uint8 bytes."""
        self._check_codes(codes)
        batch, _channels, height, width = codes.shape
        padding = (-self.channels) % 8
        bits = codes.to(dtype=torch.uint8)
        if padding:
            bits = F.pad(bits, (0, 0, 0, 0, 0, padding))
        bits = bits.view(batch, (self.channels + 7) // 8, 8, height, width)
        weights = (1 << torch.arange(8, device=codes.device, dtype=torch.int64)).view(
            1, 1, 8, 1, 1
        )
        return (bits.long() * weights).sum(dim=2).to(dtype=torch.uint8)

    def unpack_codes(self, packed: torch.Tensor) -> torch.Tensor:
        expected_bytes = (self.channels + 7) // 8
        if packed.ndim != 4 or int(packed.shape[1]) != expected_bytes:
            raise ValueError(
                f"expected packed codes [B,{expected_bytes},H,W], got {tuple(packed.shape)}"
            )
        shifts = torch.arange(8, device=packed.device, dtype=torch.int64).view(
            1, 1, 8, 1, 1
        )
        bits = ((packed.long().unsqueeze(2) >> shifts) & 1).to(dtype=torch.long)
        batch, _bytes, _bits, height, width = bits.shape
        return bits.view(batch, expected_bytes * 8, height, width)[:, : self.channels]

    def packed_to_quantized(self, packed: torch.Tensor) -> torch.Tensor:
        return self.codes_to_quantized(self.unpack_codes(packed))

    def codes_to_quantized(self, codes: torch.Tensor) -> torch.Tensor:
        self._check_codes(codes)
        levels = self.levels.to(device=codes.device, dtype=torch.float32).view(
            1, self.channels, 1, 1
        )
        span = (levels - 1.0).clamp_min(1.0)
        return codes.float() / span * 2.0 - 1.0

    def _check_codes(self, codes: torch.Tensor) -> None:
        if codes.ndim != 4 or int(codes.shape[1]) != self.channels:
            raise ValueError(
                f"expected FSQ codes [B,{self.channels},H,W], got {tuple(codes.shape)}"
            )
        if codes.numel() > 0:
            levels = self.levels.to(device=codes.device).view(1, self.channels, 1, 1)
            if bool(((codes < 0) | (codes >= levels)).any().item()):
                raise ValueError("FSQ codes are outside their per-channel level ranges")

    def forward(self, z2: torch.Tensor) -> dict[str, torch.Tensor]:
        if z2.ndim != 4 or int(z2.shape[1]) != self.channels:
            raise ValueError(
                f"expected z2 [B,{self.channels},H,W], got {tuple(z2.shape)}"
            )
        z2_norm = torch.tanh(self.pre_norm(z2))
        levels = self.levels.to(device=z2.device, dtype=z2_norm.dtype).view(
            1, self.channels, 1, 1
        )
        span = (levels - 1.0).clamp_min(1.0)
        positions = (z2_norm + 1.0) * 0.5 * span
        codes_float = round_ste(positions).clamp_min(0.0).minimum(span)
        codes = codes_float.detach().long()
        q2_hard = codes_float / span * 2.0 - 1.0
        q2 = z2_norm + (q2_hard - z2_norm).detach()
        result = {
            "z2_norm": z2_norm,
            "q2": q2,
            "q2_hard": q2_hard.detach(),
            "codes": codes,
            "packed_codes": self.pack_codes(codes),
            "fsq_mse": F.mse_loss(
                q2_hard.detach().float(), z2_norm.detach().float()
            ),
        }
        if self.supports_scalar_indices:
            result["idx2"] = self.codes_to_indices(codes)
        return result


class BARMLP(nn.Module):
    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(width), int(hidden_width))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(hidden_width), int(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(value)))


class BARAttention(nn.Module):
    """Native-PyTorch equivalent of BAR's self-attention block."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if int(width) % int(heads) != 0:
            raise ValueError(f"BAR width={width} must be divisible by heads={heads}")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.qkv = nn.Linear(self.width, self.width * 3, bias=True)
        self.proj = nn.Linear(self.width, self.width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _width = value.shape
        qkv = self.qkv(value).view(
            batch, tokens, 3, self.heads, self.head_width
        ).permute(2, 0, 3, 1, 4)
        query, key, val = qkv.unbind(dim=0)
        attended = F.scaled_dot_product_attention(query, key, val)
        attended = attended.transpose(1, 2).reshape(batch, tokens, self.width)
        return self.proj(attended)


class BARBlock(nn.Module):
    """Pre-norm ViT block matching amazon-far/BAR's decoder block contract."""

    def __init__(self, width: int, heads: int, mlp_width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(int(width))
        self.attention = BARAttention(int(width), int(heads))
        self.norm2 = nn.LayerNorm(int(width))
        self.mlp = BARMLP(int(width), int(mlp_width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attention(self.norm1(value))
        return value + self.mlp(self.norm2(value))


def initialize_bar_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class BARLayer2Encoder(nn.Module):
    """BAR/SigLIP2 token-encoder topology adapted to concat(img,x1).

    BAR's original encoder loads a large pretrained SigLIP2 model and retains a
    frozen teacher branch. This direct Layer2 variant intentionally omits that
    teacher branch, uses a native patch embedding, and keeps the same 16x16
    token grid and 1152->C projection contract.
    """

    def __init__(self, args: argparse.Namespace, latent_channels: int) -> None:
        super().__init__()
        image_size = 256
        patch_size = int(args.bar_patch_size)
        width = int(args.bar_encoder_width)
        layers = int(args.bar_encoder_layers)
        heads = int(args.bar_encoder_heads)
        mlp_width = int(args.bar_encoder_mlp_width)
        if image_size % patch_size != 0:
            raise ValueError("BAR patch size must divide the 256x256 crop")
        self.grid_size = image_size // patch_size
        self.width = width
        self.input_adapter = nn.Conv2d(6, 3, kernel_size=1, bias=True)
        self.patch_embed = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.positional_embedding = nn.Parameter(
            torch.empty(1, self.grid_size * self.grid_size, width)
        )
        self.transformer = nn.ModuleList(
            [BARBlock(width, heads, mlp_width) for _ in range(layers)]
        )
        self.post_norm = nn.LayerNorm(width)
        self.to_latent = nn.Linear(width, int(latent_channels))
        self.apply(initialize_bar_weights)
        nn.init.trunc_normal_(self.positional_embedding, mean=0.0, std=width**-0.5)
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            self.input_adapter.bias.zero_()
            for channel in range(3):
                self.input_adapter.weight[channel, channel, 0, 0] = 1.0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or int(value.shape[1]) != 6:
            raise ValueError(f"BAR E2 expected [B,6,H,W], got {tuple(value.shape)}")
        # Official BAR tokenizer consumes pixels in [-1,1].
        value = self.input_adapter(value.mul(2.0).sub(1.0))
        value = self.patch_embed(value)
        batch, _width, height, width = value.shape
        if (height, width) != (self.grid_size, self.grid_size):
            raise RuntimeError(
                f"BAR E2 expected a {self.grid_size}x{self.grid_size} token grid, "
                f"got {height}x{width}"
            )
        tokens = value.flatten(2).transpose(1, 2)
        tokens = tokens + self.positional_embedding.to(dtype=tokens.dtype)
        for block in self.transformer:
            tokens = block(tokens)
        tokens = self.to_latent(self.post_norm(tokens))
        return tokens.transpose(1, 2).reshape(
            batch, -1, self.grid_size, self.grid_size
        )


class BARSigLIP2Layer2Encoder(nn.Module):
    """Wrapper around BAR's pretrained SigLIP2 encoder path, without teacher output."""

    def __init__(self, args: argparse.Namespace, latent_channels: int) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError(
                "--bar-encoder-backend=siglip2 requires the transformers package"
            ) from error

        model_name = str(args.bar_siglip_model)
        try:
            full_model = AutoModel.from_pretrained(
                model_name,
                local_files_only=not bool(args.bar_allow_download),
            )
        except (OSError, ValueError) as error:
            mode = "download allowed" if bool(args.bar_allow_download) else "local files only"
            raise RuntimeError(
                f"failed to load BAR SigLIP2 encoder {model_name!r} ({mode}); "
                "provide a cached/local model or pass --bar-allow-download"
            ) from error

        vision = full_model.vision_model
        self.width = int(vision.config.hidden_size)
        self.patch_size = 16
        self.grid_size = 16
        self.input_adapter = nn.Conv2d(6, 3, kernel_size=1, bias=True)
        initialize_bar_weights(self.input_adapter)
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            self.input_adapter.bias.zero_()
            for channel in range(3):
                self.input_adapter.weight[channel, channel, 0, 0] = 1.0

        # BAR keeps pretrained embeddings frozen, and trains a copied pretrained
        # transformer/post-norm. The CLIP teacher copy is intentionally omitted.
        self.embeddings = vision.embeddings
        self.embeddings.requires_grad_(False)
        self.embeddings.eval()
        self.transformer = copy.deepcopy(vision.encoder)
        self.post_norm = copy.deepcopy(vision.post_layernorm)
        self.transformer.requires_grad_(True)
        self.post_norm.requires_grad_(True)
        self.to_latent = nn.Linear(self.width, int(latent_channels))
        initialize_bar_weights(self.to_latent)
        self.register_buffer(
            "spatial_shapes_template",
            torch.tensor([[self.grid_size, self.grid_size]], dtype=torch.long),
            persistent=False,
        )
        del vision
        del full_model

    def train(self, mode: bool = True) -> BARSigLIP2Layer2Encoder:
        super().train(mode)
        self.embeddings.eval()
        return self

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or tuple(value.shape[1:]) != (6, 256, 256):
            raise ValueError(
                f"BAR SigLIP2 E2 expected [B,6,256,256], got {tuple(value.shape)}"
            )
        pixels = self.input_adapter(value.mul(2.0).sub(1.0))
        batch = int(pixels.shape[0])
        patch = self.patch_size
        grid = self.grid_size
        patches = (
            pixels.view(batch, 3, grid, patch, grid, patch)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(batch, grid * grid, patch * patch * 3)
        )
        spatial_shapes = self.spatial_shapes_template.expand(batch, -1)
        self.embeddings.eval()
        tokens = self.embeddings(
            pixel_values=patches,
            spatial_shapes=spatial_shapes,
        )
        for layer in self.transformer.layers:
            tokens = layer(tokens, attention_mask=None)
            if isinstance(tokens, (tuple, list)):
                tokens = tokens[0]
        tokens = self.to_latent(self.post_norm(tokens))
        return tokens.transpose(1, 2).reshape(batch, -1, grid, grid)


class BARLayer2Decoder(nn.Module):
    """BAR SigLIP2Decoder port: C tokens -> ViT -> RGB patch reconstruction."""

    def __init__(self, args: argparse.Namespace, latent_channels: int) -> None:
        super().__init__()
        image_size = 256
        patch_size = int(args.bar_patch_size)
        width = int(args.bar_decoder_width)
        layers = int(args.bar_decoder_layers)
        heads = int(args.bar_decoder_heads)
        mlp_width = int(args.bar_decoder_mlp_width)
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.width = width
        self.from_latent = nn.Linear(int(latent_channels), width)
        self.class_embedding = nn.Parameter(torch.empty(1, 1, width))
        self.positional_embedding = nn.Parameter(
            torch.empty(1, self.grid_size * self.grid_size + 1, width)
        )
        self.pre_norm = nn.LayerNorm(width)
        self.transformer = nn.ModuleList(
            [BARBlock(width, heads, mlp_width) for _ in range(layers)]
        )
        self.post_norm = nn.LayerNorm(width)
        self.patch_head = nn.Conv2d(
            width, patch_size * patch_size * 3, kernel_size=1, bias=True
        )
        self.conv_out = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True)
        self.apply(initialize_bar_weights)
        nn.init.trunc_normal_(self.class_embedding, mean=0.0, std=width**-0.5)
        nn.init.trunc_normal_(self.positional_embedding, mean=0.0, std=width**-0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"BAR D2 expected [B,C,H,W], got {tuple(value.shape)}")
        batch, _channels, height, width = value.shape
        if (height, width) != (self.grid_size, self.grid_size):
            raise RuntimeError(
                f"BAR D2 expected {self.grid_size}x{self.grid_size} tokens, "
                f"got {height}x{width}"
            )
        tokens = value.flatten(2).transpose(1, 2)
        tokens = self.from_latent(tokens)
        cls = self.class_embedding.expand(batch, -1, -1).to(dtype=tokens.dtype)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pre_norm(
            tokens + self.positional_embedding.to(dtype=tokens.dtype)
        )
        for block in self.transformer:
            tokens = block(tokens)
        tokens = self.post_norm(tokens[:, 1:])
        feature_map = tokens.transpose(1, 2).reshape(
            batch, self.width, self.grid_size, self.grid_size
        )
        patches = self.patch_head(feature_map)
        patch = self.patch_size
        image = patches.view(
            batch, patch, patch, 3, self.grid_size, self.grid_size
        ).permute(0, 3, 4, 1, 5, 2).reshape(
            batch, 3, self.grid_size * patch, self.grid_size * patch
        )
        return self.conv_out(image)


def build_layer1_modules(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, nn.Module]:
    if str(args.layer1_arch) == "swin":
        return build_layer1(args, device)
    if str(args.layer1_arch) == "cnn":
        e1 = CNNAnalysisEncoder(
            base_ch=int(args.layer1_cnn_base_ch),
            bottleneck_ch=int(args.latent_ch),
            num_res=int(args.layer1_cnn_num_res),
        ).to(device)
        d1 = CNNBottleneckDecoder(
            base_ch=int(args.layer1_cnn_base_ch),
            bottleneck_ch=int(args.latent_ch),
            num_res=int(args.layer1_cnn_num_res),
            output_activation="none",
        ).to(device)
        return e1, d1
    raise ValueError(f"unknown Layer1 architecture {args.layer1_arch!r}")


def build_layer2_modules(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, nn.Module, str]:
    channels = int(args.layer2_c)
    architecture = str(args.layer2_arch)
    if architecture == "swin":
        return (
            build_jscc_encoder(args, device, latent_ch=channels, in_chans=6),
            build_jscc_decoder(args, device, latent_ch=channels),
            "zero_one",
        )
    if architecture == "cnn":
        base = int(args.layer2_cnn_base_ch)
        e2 = CNNAnalysisEncoder(
            base_ch=base,
            bottleneck_ch=channels,
            num_res=int(args.layer2_cnn_num_res),
        ).to(device)
        e2.stem = ConvNormAct(6, base, kernel=3, stride=1).to(device)
        d2 = CNNBottleneckDecoder(
            base_ch=base,
            bottleneck_ch=channels,
            num_res=int(args.layer2_cnn_num_res),
            output_activation="none",
        ).to(device)
        if channels == base * 16:
            e2.compressor = nn.Identity()
            d2.expander = nn.Identity()
        return e2, d2, "zero_one"
    if architecture == "bar":
        if str(args.bar_encoder_backend) == "siglip2":
            bar_encoder = BARSigLIP2Layer2Encoder(args, channels).to(device)
        else:
            bar_encoder = BARLayer2Encoder(args, channels).to(device)
        return (
            bar_encoder,
            BARLayer2Decoder(args, channels).to(device),
            "minus_one_one",
        )
    raise ValueError(f"unknown Layer2 architecture {architecture!r}")


class Layer2FSQCodec(nn.Module):
    """Trainable Layer2: architecture-specific E2/D2 around binary FSQ."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        super().__init__()
        self.architecture = str(args.layer2_arch)
        self.latent_channels = int(args.layer2_c)
        self.latent_height = int(args.latent_h)
        self.latent_width = int(args.latent_w)
        self.e2, self.d2, self.decoder_range = build_layer2_modules(args, device)
        self.quantizer = IFSQQuantizer(
            [FSQ_LEVEL] * self.latent_channels,
            channels=self.latent_channels,
            use_pre_norm=not bool(args.no_pre_norm),
        ).to(device)
        self.combiner = OutputsCombiner().to(device)

    def encode(self, img: torch.Tensor, x1: torch.Tensor) -> dict[str, torch.Tensor]:
        e2_input = torch.cat([img, x1], dim=1)
        z2 = encode_tensor(self.e2, e2_input)
        expected = (self.latent_channels, self.latent_height, self.latent_width)
        if tuple(z2.shape[1:]) != expected:
            raise RuntimeError(
                f"{self.architecture} E2 must output [B,{expected[0]},"
                f"{expected[1]},{expected[2]}], got {tuple(z2.shape)}"
            )
        encoded = self.quantizer(z2)
        return {**encoded, "e2_input": e2_input, "z2": z2}

    def decode(self, q2: torch.Tensor, x1: torch.Tensor) -> dict[str, torch.Tensor]:
        u2_raw = self.d2(q2)
        if self.decoder_range == "minus_one_one":
            # BAR predicts the normalized-image domain but has no terminal tanh.
            # Keep this affine conversion unclamped so final-only training retains
            # gradients even while fresh decoder outputs temporarily exceed [-1,1].
            u2 = u2_raw.add(1.0).mul(0.5)
        else:
            u2 = u2_raw.clamp(0.0, 1.0)
        final = self.combiner(x1, u2)
        return {"u2_raw": u2_raw, "u2": u2, "final": final}

    @staticmethod
    def shuffle_q2(q2: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = q2.shape
        flat = q2.permute(0, 2, 3, 1).reshape(-1, channels)
        permutation = torch.randperm(flat.shape[0], device=q2.device)
        return (
            flat[permutation]
            .view(batch, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def forward(
        self,
        img: torch.Tensor,
        x1: torch.Tensor,
        *,
        q_mode: str = "normal",
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(img, x1)
        q2 = encoded["q2"]
        if q_mode == "code0":
            q2 = self.quantizer.codes_to_quantized(
                torch.zeros_like(encoded["codes"])
            ).to(device=q2.device, dtype=q2.dtype)
        elif q_mode == "shuffle":
            q2 = self.shuffle_q2(q2)
        elif q_mode != "normal":
            raise ValueError(f"unknown q_mode {q_mode!r}")
        decoded = self.decode(q2, x1)
        return {**encoded, **decoded, "q2_used": q2}


@dataclass
class FSQSystem:
    e1: nn.Module
    d1: nn.Module
    codec: Layer2FSQCodec
    init_report: dict[str, dict[str, int | float | str]]

    @torch.no_grad()
    def layer1(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        z1 = encode_tensor(self.e1, imgs)
        x1_raw = self.d1(z1)
        return {
            "z1": z1,
            "x1_raw": x1_raw,
            "x1": x1_raw.clamp(0.0, 1.0),
        }

    def forward(self, imgs: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        layer1_out = self.layer1(imgs)
        out = self.codec(imgs, layer1_out["x1"])
        return layer1_out, out

    def train_codec(self) -> None:
        self.e1.eval()
        self.d1.eval()
        self.codec.train()

    def eval(self) -> None:
        self.e1.eval()
        self.d1.eval()
        self.codec.eval()


def load_compatible_state(
    module: nn.Module,
    source_state: dict[str, torch.Tensor],
    label: str,
) -> dict[str, int | float | str]:
    target_state = module.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    missing, unexpected = module.load_state_dict(compatible, strict=False)
    matched_numel = sum(int(target_state[key].numel()) for key in compatible)
    total_numel = sum(int(value.numel()) for value in target_state.values())
    report: dict[str, int | float | str] = {
        "label": label,
        "matched_tensors": len(compatible),
        "total_tensors": len(target_state),
        "matched_numel": matched_numel,
        "total_numel": total_numel,
        "matched_ratio": matched_numel / float(max(1, total_numel)),
        "missing_tensors": len(missing),
        "missing_keys": ",".join(missing),
        "unexpected_tensors": len(unexpected),
    }
    print(
        f"[compatible init] {label}: tensors={len(compatible)}/{len(target_state)} "
        f"numel={matched_numel}/{total_numel} "
        f"({100.0 * float(report['matched_ratio']):.2f}%) "
        f"fresh={list(missing)}",
        flush=True,
    )
    return report


def state_family(state: dict[str, torch.Tensor]) -> str:
    keys = tuple(state)
    if any(key.startswith("input_adapter.") for key in keys):
        return "bar"
    if any(key.startswith("stem.") for key in keys):
        return "cnn"
    if any(key.startswith("encoder.") for key in keys):
        return "swin"
    return "unknown"


def checkpoint_layer2_family(checkpoint: dict) -> str:
    architecture = checkpoint.get("architecture", {})
    if isinstance(architecture, dict) and str(architecture.get("layer2", "")):
        return str(architecture["layer2"]).lower()
    source_args = checkpoint.get("args", {})
    for key in ("layer2_arch", "arch"):
        value = str(source_args.get(key, "")).lower()
        if value in {"swin", "cnn", "bar"}:
            return value
    text = " ".join(
        [
            str(checkpoint.get("stage", "")),
            str(checkpoint.get("version", "")),
            str(checkpoint.get("stage2_codec", {}).get("arch", "")),
        ]
    ).lower()
    for family in ("bar", "cnn", "swin"):
        if family in text:
            return family
    if "e2_state_dict" in checkpoint:
        return state_family(checkpoint["e2_state_dict"])
    return "unknown"


def checkpoint_layer2_contract(
    checkpoint: dict, family: str
) -> tuple[int | None, int | None, int | None]:
    """Return actual (E2 input C, E2 output C, D2 input C) when inferable."""
    codec = checkpoint.get("stage2_codec", {})
    architecture = checkpoint.get("architecture", {})
    e2_state = checkpoint.get("e2_state_dict", {})
    d2_state = checkpoint.get("d2_state_dict", {})

    e2_input = codec.get("e2_in_ch")
    e2_output = codec.get("z2_ch")
    d2_input = codec.get("d2_in_ch")
    if isinstance(architecture, dict):
        e2_input = architecture.get("e2_in_ch", e2_input)
        e2_output = architecture.get("layer2_c", e2_output)
        d2_input = architecture.get("d2_in_ch", d2_input)

    if family == "swin":
        if "encoder.patch_embed.proj.weight" in e2_state:
            e2_input = int(e2_state["encoder.patch_embed.proj.weight"].shape[1])
        if "encoder.head_list.weight" in e2_state:
            e2_output = int(e2_state["encoder.head_list.weight"].shape[0])
        elif e2_output is None:
            e2_output = 320
        if "decoder.head_list.weight" in d2_state:
            d2_input = int(d2_state["decoder.head_list.weight"].shape[1])
        elif d2_input is None:
            d2_input = 320
    elif family == "cnn":
        if "stem.net.0.weight" in e2_state:
            e2_input = int(e2_state["stem.net.0.weight"].shape[1])
        if "compressor.7.weight" in e2_state:
            e2_output = int(e2_state["compressor.7.weight"].shape[0])
        elif e2_output is None and "refine.0.conv1.weight" in e2_state:
            e2_output = int(e2_state["refine.0.conv1.weight"].shape[0])
        if "expander.0.weight" in d2_state:
            d2_input = int(d2_state["expander.0.weight"].shape[1])
        elif d2_input is None and "init.0.weight" in d2_state:
            d2_input = int(d2_state["init.0.weight"].shape[1])
    elif family == "bar":
        if "input_adapter.weight" in e2_state:
            e2_input = int(e2_state["input_adapter.weight"].shape[1])
        if "to_latent.weight" in e2_state:
            e2_output = int(e2_state["to_latent.weight"].shape[0])
        if "from_latent.weight" in d2_state:
            d2_input = int(d2_state["from_latent.weight"].shape[1])

    return (
        int(e2_input) if e2_input is not None else None,
        int(e2_output) if e2_output is not None else None,
        int(d2_input) if d2_input is not None else None,
    )


def layer1_states(checkpoint: dict) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if "e1_state_dict" in checkpoint and "d1_state_dict" in checkpoint:
        return checkpoint["e1_state_dict"], checkpoint["d1_state_dict"]
    if "encoder_state_dict" in checkpoint and "decoder_state_dict" in checkpoint:
        return checkpoint["encoder_state_dict"], checkpoint["decoder_state_dict"]
    raise KeyError("Layer1 checkpoint must contain E1/D1 or encoder/decoder state dicts")


def checkpoint_layer1_family(checkpoint: dict) -> str:
    architecture = checkpoint.get("architecture", {})
    if isinstance(architecture, dict):
        family = str(architecture.get("layer1", "")).lower()
        if family in {"swin", "cnn"}:
            return family
    source_args = checkpoint.get("args", {})
    family = str(source_args.get("layer1_arch", "")).lower()
    if family in {"swin", "cnn"}:
        return family
    for key in ("e1_state_dict", "encoder_state_dict"):
        if key in checkpoint:
            return state_family(checkpoint[key])
    return "unknown"


def load_initialization_checkpoints(args: argparse.Namespace) -> dict:
    if not args.layer1_ckpt:
        args.layer1_ckpt = DEFAULT_LAYER1_CKPTS[str(args.layer1_arch)]
    layer1_checkpoint = jsccf_io.load_checkpoint(args.layer1_ckpt)
    e1_state, _d1_state = layer1_states(layer1_checkpoint)
    source_layer1_family = state_family(e1_state)
    if source_layer1_family != str(args.layer1_arch):
        raise ValueError(
            f"Layer1 checkpoint family={source_layer1_family!r} does not match "
            f"--layer1-arch={args.layer1_arch!r}"
        )

    layer2_checkpoint = None
    if str(args.codec_init) == "compatible":
        if not args.layer2_ckpt:
            args.layer2_ckpt = DEFAULT_LAYER2_CKPTS[str(args.layer2_arch)]
        if not args.layer2_ckpt:
            raise ValueError(
                f"--codec-init=compatible for {args.layer2_arch} requires --layer2-ckpt"
            )
        layer2_checkpoint = jsccf_io.load_checkpoint(args.layer2_ckpt)
        required = {"e2_state_dict", "d2_state_dict", "combiner_state_dict"}
        missing = sorted(required.difference(layer2_checkpoint))
        if missing:
            raise KeyError(
                f"Layer2 checkpoint {resolve_path(args.layer2_ckpt)} misses {missing}"
            )
        source_family = checkpoint_layer2_family(layer2_checkpoint)
        if source_family != str(args.layer2_arch):
            raise ValueError(
                "cross-family compatible initialization is not allowed: "
                f"source={source_family!r}, target={args.layer2_arch!r}"
            )
        source_args = layer2_checkpoint.get("args", {})
        if source_family == "bar":
            source_bar_backend = str(source_args.get("bar_encoder_backend", ""))
            if source_bar_backend and source_bar_backend != str(args.bar_encoder_backend):
                raise ValueError(
                    "BAR compatible initialization requires the same encoder backend: "
                    f"source={source_bar_backend!r}, target={args.bar_encoder_backend!r}"
                )
        source_variant = str(
            layer2_checkpoint.get("variant", source_args.get("variant", ""))
        )
        if source_variant and source_variant != "combiner":
            raise ValueError(
                "compatible initialization requires variant=combiner, "
                f"got {source_variant!r}"
            )
        source_e2_input, source_e2_output, source_d2_input = checkpoint_layer2_contract(
            layer2_checkpoint, source_family
        )
        if source_e2_input is not None and source_e2_input != 6:
            raise ValueError(
                "compatible Layer2 source must encode concat(img,x1) with 6 channels"
            )
        if (
            source_e2_output is not None
            and source_d2_input is not None
            and source_e2_output != source_d2_input
        ):
            raise ValueError(
                "compatible Layer2 source is not a direct E2->D2 codec: "
                f"E2 outputs C={source_e2_output}, D2 expects C={source_d2_input}"
            )

    print(
        f"initialization: layer1={resolve_path(args.layer1_ckpt)} "
        f"layer2={resolve_path(args.layer2_ckpt) if args.layer2_ckpt else '<fresh>'} "
        f"codec_init={args.codec_init}",
        flush=True,
    )
    return {"layer1": layer1_checkpoint, "layer2": layer2_checkpoint}


def build_system(
    args: argparse.Namespace,
    initialization: dict,
    device: torch.device,
) -> FSQSystem:
    e1, d1 = build_layer1_modules(args, device)
    e1_state, d1_state = layer1_states(initialization["layer1"])
    jsccf_io.load_state(e1, e1_state, "frozen_E1", strict=True)
    jsccf_io.load_state(d1, d1_state, "frozen_D1", strict=True)
    set_trainable(e1, False)
    set_trainable(d1, False)
    e1.eval()
    d1.eval()

    codec = Layer2FSQCodec(args, device)
    layer2_checkpoint = initialization["layer2"]
    init_report: dict[str, dict[str, int | float | str]] = {
        "layer1_source": {
            "path": str(args.layer1_ckpt),
            "family": str(args.layer1_arch),
        },
        "layer2_source": {
            "path": str(args.layer2_ckpt or ""),
            "family": str(args.layer2_arch),
            "mode": str(args.codec_init),
        },
    }
    if layer2_checkpoint is not None:
        init_report["e2"] = load_compatible_state(
            codec.e2,
            layer2_checkpoint["e2_state_dict"],
            f"source_{args.layer2_arch}_E2->C{int(args.layer2_c)}_E2",
        )
        init_report["d2"] = load_compatible_state(
            codec.d2,
            layer2_checkpoint["d2_state_dict"],
            f"source_{args.layer2_arch}_D2->C{int(args.layer2_c)}_D2",
        )
        if "quantizer_state_dict" in layer2_checkpoint:
            source_quantizer_state = layer2_checkpoint["quantizer_state_dict"]
            source_levels_tensor = source_quantizer_state.get("levels")
            source_levels = (
                [int(value) for value in source_levels_tensor.detach().cpu().tolist()]
                if torch.is_tensor(source_levels_tensor)
                else []
            )
            expected_levels = [FSQ_LEVEL] * int(args.layer2_c)
            if source_levels == expected_levels:
                # levels/multipliers define the current binary wire format and
                # must never be overwritten from an initialization checkpoint.
                source_pre_norm = {
                    key: value
                    for key, value in source_quantizer_state.items()
                    if key.startswith("pre_norm.")
                }
                init_report["quantizer"] = load_compatible_state(
                    codec.quantizer,
                    source_pre_norm,
                    f"source_binary_FSQ_affine->C{int(args.layer2_c)}_FSQ",
                )
            else:
                init_report["quantizer"] = {
                    "label": "fresh_FSQ_nonbinary_or_incompatible_source",
                    "source_levels": ",".join(str(value) for value in source_levels),
                    "target_levels": ",".join(str(value) for value in expected_levels),
                }
        source_layer1_family = checkpoint_layer1_family(layer2_checkpoint)
        if source_layer1_family == str(args.layer1_arch):
            source_combiner_state = layer2_checkpoint["combiner_state_dict"]
            if any(key.startswith("inner.") for key in source_combiner_state):
                source_combiner_state = {
                    key.removeprefix("inner."): value
                    for key, value in source_combiner_state.items()
                    if key.startswith("inner.")
                }
            init_report["combiner"] = load_compatible_state(
                codec.combiner,
                source_combiner_state,
                "source_combiner",
            )
        else:
            init_report["combiner"] = {
                "label": "fresh_combiner_unmatched_layer1_family",
                "source_layer1_family": source_layer1_family,
                "target_layer1_family": str(args.layer1_arch),
            }
    return FSQSystem(e1=e1, d1=d1, codec=codec, init_report=init_report)


def experiment_name(args: argparse.Namespace) -> str:
    normalizer = "none" if bool(args.no_pre_norm) else "group"
    architecture_detail = ""
    if str(args.layer2_arch) == "bar":
        if str(args.bar_encoder_backend) == "native":
            architecture_detail = (
                f"_{args.bar_encoder_backend}-e{int(args.bar_encoder_width)}"
                f"l{int(args.bar_encoder_layers)}h{int(args.bar_encoder_heads)}"
                f"m{int(args.bar_encoder_mlp_width)}"
            )
        else:
            siglip_name = jsccf_io.safe_artifact_name(
                str(args.bar_siglip_model).rstrip("/").split("/")[-1]
            )
            architecture_detail = f"_{args.bar_encoder_backend}-{siglip_name}"
        architecture_detail += (
            f"-d{int(args.bar_decoder_width)}l{int(args.bar_decoder_layers)}"
            f"h{int(args.bar_decoder_heads)}m{int(args.bar_decoder_mlp_width)}"
        )
    return (
        f"stage3_fsq_l1-{args.layer1_arch}_l2-{args.layer2_arch}_"
        f"c{int(args.layer2_c)}{architecture_detail}_binary_direct_"
        f"{normalizer}_{args.codec_init}"
    )


def display_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {name: metrics[name] for name in DISPLAY_METRICS if name in metrics}


def compute_losses(
    out: dict[str, torch.Tensor],
    imgs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    loss_final = recon_loss(out["final"], imgs)
    return {
        "loss": loss_final,
        "loss_final": loss_final,
    }


def update_joint_counter(counter: Counter[bytes], packed_codes: torch.Tensor) -> None:
    if packed_codes.ndim != 4 or packed_codes.dtype != torch.uint8:
        raise ValueError(
            "packed FSQ codes must be uint8 with shape [B,ceil(C/8),H,W]"
        )
    rows = (
        packed_codes.detach()
        .permute(0, 2, 3, 1)
        .reshape(-1, int(packed_codes.shape[1]))
        .contiguous()
        .cpu()
        .numpy()
    )
    counter.update(row.tobytes() for row in rows)


def update_bit_histogram(histogram: torch.Tensor, codes: torch.Tensor) -> None:
    if histogram.ndim != 2 or int(histogram.shape[1]) != FSQ_LEVEL:
        raise ValueError(f"expected bit histogram [C,2], got {tuple(histogram.shape)}")
    channels = int(histogram.shape[0])
    if codes.ndim != 4 or int(codes.shape[1]) != channels:
        raise ValueError(
            f"expected binary codes [B,{channels},H,W], got {tuple(codes.shape)}"
        )
    if codes.numel() > 0:
        code_min = int(codes.detach().min().item())
        code_max = int(codes.detach().max().item())
        if code_min < 0 or code_max >= FSQ_LEVEL:
            raise ValueError(
                f"binary FSQ codes must be in {{0,1}}, got min={code_min} max={code_max}"
            )
    flat = codes.detach().permute(1, 0, 2, 3).reshape(channels, -1).cpu()
    for channel in range(channels):
        counts = torch.bincount(flat[channel], minlength=FSQ_LEVEL).float()
        histogram[channel] += counts[:FSQ_LEVEL]


def update_metrics(
    metric_meters: dict,
    out: dict[str, torch.Tensor],
    layer1_out: dict[str, torch.Tensor],
    imgs: torch.Tensor,
    losses: dict[str, torch.Tensor],
) -> None:
    batch_size = int(imgs.shape[0])
    for name, value in losses.items():
        metric_meters[name].update(float(value.detach().item()), batch_size)

    psnr_x1 = batch_metric_mean(psnr_per_image(layer1_out["x1"], imgs))
    psnr_final = batch_metric_mean(psnr_per_image(out["final"], imgs))
    metric_meters["mse_x1"].update(
        batch_metric_mean(mse_per_image(layer1_out["x1"], imgs)), batch_size
    )
    metric_meters["psnr_x1"].update(psnr_x1, batch_size)
    metric_meters["ssim_x1"].update(
        batch_metric_mean(ssim_per_image(layer1_out["x1"], imgs)), batch_size
    )
    metric_meters["mse_u2_as_img"].update(
        batch_metric_mean(mse_per_image(out["u2"], imgs)), batch_size
    )
    metric_meters["psnr_u2_as_img"].update(
        batch_metric_mean(psnr_per_image(out["u2"], imgs)), batch_size
    )
    metric_meters["mse_final"].update(
        batch_metric_mean(mse_per_image(out["final"], imgs)), batch_size
    )
    metric_meters["psnr_final"].update(psnr_final, batch_size)
    metric_meters["ssim_final"].update(
        batch_metric_mean(ssim_per_image(out["final"], imgs)), batch_size
    )
    metric_meters["delta_x1"].update(psnr_final - psnr_x1, batch_size)
    metric_meters["z2_abs_mean"].update(
        float(out["z2"].detach().float().abs().mean().item()), batch_size
    )
    metric_meters["q2_abs_mean"].update(
        float(out["q2_hard"].detach().float().abs().mean().item()), batch_size
    )
    metric_meters["fsq_mse"].update(
        float(out["fsq_mse"].detach().item()), batch_size
    )


@torch.no_grad()
def update_ablation_metrics(
    metric_meters: dict,
    system: FSQSystem,
    out: dict[str, torch.Tensor],
    layer1_out: dict[str, torch.Tensor],
    imgs: torch.Tensor,
) -> None:
    batch_size = int(imgs.shape[0])
    x1 = layer1_out["x1"]
    code0 = system.codec.decode(
        system.codec.quantizer.codes_to_quantized(torch.zeros_like(out["codes"]))
        .to(device=out["q2"].device, dtype=out["q2"].dtype),
        x1,
    )
    shuffled = system.codec.decode(system.codec.shuffle_q2(out["q2"]), x1)
    psnr_final = batch_metric_mean(psnr_per_image(out["final"], imgs))
    psnr_code0 = batch_metric_mean(psnr_per_image(code0["final"], imgs))
    psnr_shuffle = batch_metric_mean(psnr_per_image(shuffled["final"], imgs))
    metric_meters["psnr_code0"].update(psnr_code0, batch_size)
    metric_meters["psnr_shuffle"].update(psnr_shuffle, batch_size)
    metric_meters["drop_code0"].update(psnr_final - psnr_code0, batch_size)
    metric_meters["drop_shuffle"].update(psnr_final - psnr_shuffle, batch_size)


def finalize_metrics(
    metric_meters: dict,
    joint_counter: Counter[bytes],
    bit_histogram: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    metrics = averaged(metric_meters)
    channels = int(args.layer2_c)
    total_count = int(sum(joint_counter.values()))
    total = float(total_count)
    if total_count > 0:
        counts = torch.tensor(list(joint_counter.values()), dtype=torch.float64)
        probabilities = (counts / total).clamp_min(1e-12)
        entropy_bits = float(
            -(probabilities * probabilities.log2()).sum().item()
        )
        used_codes = float(len(joint_counter))
        sample_ceiling_codes = min(1 << channels, total_count)
        usage_ratio = math.ldexp(used_codes, -channels) if channels < 1075 else 0.0
        sample_usage_ratio = (
            math.ldexp(float(sample_ceiling_codes), -channels)
            if channels < 1075
            else 0.0
        )
        metrics.update(
            {
                "code_used": used_codes,
                "code_usage_ratio": usage_ratio,
                "code_usage_sample_ceiling": sample_usage_ratio,
                "code_coverage_of_sample_ceiling": used_codes
                / max(1.0, float(sample_ceiling_codes)),
                "code_entropy_bits": entropy_bits,
                "code_perplexity": float(2.0**entropy_bits),
                "code_top1_frac": max(joint_counter.values()) / total,
                "joint_entropy_sample_ceiling_bits": math.log2(
                    max(1, sample_ceiling_codes)
                ),
            }
        )
    else:
        metrics.update(
            {
                "code_used": 0.0,
                "code_usage_ratio": 0.0,
                "code_usage_sample_ceiling": 0.0,
                "code_coverage_of_sample_ceiling": 0.0,
                "code_entropy_bits": 0.0,
                "code_perplexity": 0.0,
                "code_top1_frac": 0.0,
                "joint_entropy_sample_ceiling_bits": 0.0,
            }
        )

    bit_totals = bit_histogram.sum(dim=1).clamp_min(1.0)
    bit_one_fraction = bit_histogram[:, 1] / bit_totals
    bit_probabilities = (bit_histogram / bit_totals[:, None]).clamp_min(1e-12)
    bit_entropy = -(bit_probabilities * bit_probabilities.log2()).sum(dim=1)
    bit_used = (bit_histogram > 0).sum(dim=1)
    joint_entropy_bits_per_token = float(metrics["code_entropy_bits"])
    marginal_entropy_bits_per_token = float(bit_entropy.sum().item())
    spatial_tokens = int(args.latent_h) * int(args.latent_w)
    fixed_bits_per_token = channels
    fixed_bits_per_image = fixed_bits_per_token * spatial_tokens
    metrics.update(
        {
            "bit_one_frac_mean": float(bit_one_fraction.mean().item()),
            "bit_one_frac_min": float(bit_one_fraction.min().item()),
            "bit_one_frac_max": float(bit_one_fraction.max().item()),
            "bit_entropy_bits_mean": float(bit_entropy.mean().item()),
            "bit_entropy_bits_min": float(bit_entropy.min().item()),
            "bit_dead_channels": float((bit_used < FSQ_LEVEL).sum().item()),
            "empirical_bits_per_token": joint_entropy_bits_per_token,
            "empirical_bits_per_image": joint_entropy_bits_per_token * spatial_tokens,
            "empirical_bpp": joint_entropy_bits_per_token
            * spatial_tokens
            / float(256 * 256),
            "marginal_entropy_bits_per_token": marginal_entropy_bits_per_token,
            "marginal_entropy_bits_per_image": marginal_entropy_bits_per_token
            * spatial_tokens,
            "marginal_entropy_bpp": marginal_entropy_bits_per_token
            * spatial_tokens
            / float(256 * 256),
            "log2_vocab_size": float(channels),
            "fixed_bits_per_token": float(fixed_bits_per_token),
            "fixed_bits_per_image": float(fixed_bits_per_image),
            "fixed_bpp": fixed_bits_per_image / float(256 * 256),
        }
    )
    if channels <= 52:
        metrics["vocab_size"] = float(1 << channels)
    return metrics


@torch.no_grad()
def validate(
    loader,
    system: FSQSystem,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    system.eval()
    names = METRIC_NAMES + (
        VAL_ABLATION_METRICS if bool(args.val_ablation) else []
    )
    metric_meters = meters(names)
    joint_counter: Counter[bytes] = Counter()
    bit_histogram = torch.zeros(int(args.layer2_c), FSQ_LEVEL, dtype=torch.float32)
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if int(args.max_val_batches) > 0 and batch_index > int(args.max_val_batches):
            break
        imgs = imgs.to(device, non_blocking=True)
        layer1_out, out = system.forward(imgs)
        losses = compute_losses(out, imgs)
        update_metrics(metric_meters, out, layer1_out, imgs, losses)
        update_joint_counter(joint_counter, out["packed_codes"])
        update_bit_histogram(bit_histogram, out["codes"])
        if bool(args.val_ablation):
            update_ablation_metrics(metric_meters, system, out, layer1_out, imgs)
    return finalize_metrics(
        metric_meters, joint_counter, bit_histogram, args
    )


def print_run_header(
    args: argparse.Namespace,
    system: FSQSystem,
    train_size: int,
    val_size: int,
) -> None:
    channels = int(args.layer2_c)
    spatial_tokens = int(args.latent_h) * int(args.latent_w)
    fixed_bits = channels * spatial_tokens
    val_tokens = int(val_size) * spatial_tokens
    val_usage_ceiling = (
        min(1.0, math.ldexp(float(val_tokens), -channels))
        if channels < 1075
        else 0.0
    )
    val_entropy_ceiling = math.log2(max(1, min(val_tokens, 1 << channels)))
    print(
        f"=== Stage 3 | Layer1={args.layer1_arch} | "
        f"Layer2={args.layer2_arch} binary FSQ C={channels} ===",
        flush=True,
    )
    print(
        f"device={'cpu' if args.cpu else 'cuda:0'} "
        f"visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    print(f"save_dir={resolve_path(args.save_dir)}", flush=True)
    print("实验设计", flush=True)
    print(
        f"  Layer1={args.layer1_arch} frozen; Layer2={args.layer2_arch}; "
        "no parallel teacher forward",
        flush=True,
    )
    print(
        "  image -> E1/D1 -> x1; concat(image,x1) -> E2 -> "
        f"z2[B,{channels},16,16] -> FSQ -> q2[B,{channels},16,16] -> D2 -> u2; "
        "combiner(x1,u2) -> final",
        flush=True,
    )
    print(
        f"  FSQ levels=[2]x{channels}; one bit/channel; "
        f"packed=[B,{(channels + 7) // 8},16,16]; K=2^{channels}; "
        f"Layer2 incremental fixed_bits/token={channels} "
        f"fixed_bits/image={fixed_bits} "
        f"fixed_bpp={fixed_bits / float(256 * 256):.6f} (excluding Layer1)",
        flush=True,
    )
    print(
        f"  validation joint-stat sample ceiling: usage_ratio<={val_usage_ceiling:.6f} "
        f"entropy<={val_entropy_ceiling:.6f}bit for {val_size} images",
        flush=True,
    )
    print(
        f"  layer1_checkpoint={resolve_path(args.layer1_ckpt)} "
        f"layer2_checkpoint={resolve_path(args.layer2_ckpt) if args.layer2_ckpt else '<fresh>'} "
        f"codec_init={args.codec_init}; checkpoint is initialization only",
        flush=True,
    )
    print("loss设计", flush=True)
    print("  L=MSE(combiner(x1,D2(FSQ(E2(concat(image,x1))))), image)", flush=True)
    print("  teacher alignment=off; no auxiliary target from a continuous Layer2", flush=True)
    print("模块选择", flush=True)
    print(
        f"  E1={system.e1.__class__.__name__}({trainable_state(system.e1)}) "
        f"D1={system.d1.__class__.__name__}({trainable_state(system.d1)})",
        flush=True,
    )
    print(
        f"  E2={system.codec.e2.__class__.__name__}(6->{channels},{trainable_state(system.codec.e2)}) "
        f"FSQ=IFSQQuantizer(d={channels},levels=[2]x{channels},"
        f"{trainable_state(system.codec.quantizer)}) "
        f"D2={system.codec.d2.__class__.__name__}({channels}->3,{trainable_state(system.codec.d2)}) "
        f"combiner={trainable_state(system.codec.combiner)}",
        flush=True,
    )
    if str(args.layer2_arch) == "bar":
        print(
            f"  BAR codec source={BAR_SOURCE_URL}; encoder_backend={args.bar_encoder_backend}; "
            "decoder raw normalized-domain output is mapped by (raw+1)/2 without clamp",
            flush=True,
        )
        if int(args.batch_size) > 1:
            print(
                "  WARNING: official-scale BAR is very large; start with --batch-size 1 "
                "and increase only after checking GPU memory.",
                flush=True,
            )
    print(f"  compatible_init={system.init_report}", flush=True)
    print(
        f"epochs={args.epochs} train={train_size} valid={val_size} "
        f"batch={args.batch_size} test_batch={args.test_batch} "
        f"lr={float(args.lr):g} weight_decay={float(args.weight_decay):g}",
        flush=True,
    )


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    metrics_split: str,
    system: FSQSystem,
) -> None:
    output = Path(resolve_path(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    channels = int(args.layer2_c)
    spatial_tokens = int(args.latent_h) * int(args.latent_w)
    payload = {
        "route": getattr(
            jsccf_io,
            "ROUTE",
            "TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256",
        ),
        "stage": str(args.stage),
        "epoch": int(epoch),
        "metrics": metrics,
        "metrics_split": str(metrics_split),
        "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "version": str(args.version),
        "architecture": {
            "layer1": str(args.layer1_arch),
            "layer2": str(args.layer2_arch),
            "layer2_c": channels,
            "layer2_backend": (
                str(args.bar_encoder_backend)
                if str(args.layer2_arch) == "bar"
                else str(args.layer2_arch)
            ),
            "e2_in_ch": 6,
            "d2_in_ch": channels,
        },
        "source_layer1_ckpt": str(args.layer1_ckpt),
        "source_layer2_ckpt": str(args.layer2_ckpt),
        "e1_state_dict": system.e1.state_dict(),
        "d1_state_dict": system.d1.state_dict(),
        "e2_state_dict": system.codec.e2.state_dict(),
        "d2_state_dict": system.codec.d2.state_dict(),
        "combiner_state_dict": system.codec.combiner.state_dict(),
        "quantizer_state_dict": system.codec.quantizer.state_dict(),
        "init_report": system.init_report,
        "latent": {
            "z1": [int(args.latent_ch), int(args.latent_h), int(args.latent_w)],
            "z2": [channels, int(args.latent_h), int(args.latent_w)],
            "q2": [channels, int(args.latent_h), int(args.latent_w)],
        },
        "quantizer": {
            "type": "binary_fsq",
            "channels": channels,
            "levels": [FSQ_LEVEL] * channels,
            "bits_per_channel": 1,
            "vocab_size": 1 << channels,
            "log2_vocab_size": channels,
            "codes_shape": [
                channels,
                int(args.latent_h),
                int(args.latent_w),
            ],
            "packed_codes_shape": [
                (channels + 7) // 8,
                int(args.latent_h),
                int(args.latent_w),
            ],
            "index_encoding": "packed_little_endian_bits",
            "scalar_index_available": channels <= 63,
            "index_shape": (
                [int(args.latent_h), int(args.latent_w)] if channels <= 63 else None
            ),
            "fixed_bits_per_token": channels,
            "fixed_bits_per_image": channels * spatial_tokens,
            "bit_budget_scope": "layer2_incremental_excluding_layer1",
            "incremental_fixed_bpp": channels * spatial_tokens / float(256 * 256),
        },
        "objective": {
            "final_reconstruction": "MSE(final,image)",
            "teacher_alignment": False,
        },
    }
    torch.save(payload, output)
    print(f"saved checkpoint: {output}", flush=True)


def train(args: argparse.Namespace, initialization: dict) -> None:
    seed_everything(int(args.seed))
    config = jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = get_loader(config)
    system = build_system(args, initialization, config.device)
    parameters = list(system.codec.parameters())
    optimizer = optim.AdamW(
        parameters,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    print_run_header(
        args,
        system,
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    if bool(args.eval_init_only):
        val_metrics = validate(val_loader, system, args, config.device)
        print(
            f"[stage3-fsq init val] {display_metrics(val_metrics)} score=psnr_final",
            flush=True,
        )
        output = (
            Path(resolve_path(args.save_dir))
            / f"{experiment_name(args)}_jscc_f_"
            f"{jsccf_io.safe_artifact_name(args.version)}_init_eval.json"
        )
        write_json(
            output,
            {
                "args": {
                    key: value
                    for key, value in vars(args).items()
                    if not key.startswith("_")
                },
                "metrics": val_metrics,
                "init_report": system.init_report,
            },
        )
        print(f"[stage3-fsq init val] wrote {output}", flush=True)
        return

    best = -math.inf
    last_metrics: dict[str, float] = {}
    last_checkpoint_metrics: dict[str, float] = {}
    last_checkpoint_split = "train"
    for epoch in range(1, int(args.epochs) + 1):
        system.train_codec()
        metric_meters = meters(METRIC_NAMES)
        joint_counter: Counter[bytes] = Counter()
        bit_histogram = torch.zeros(int(args.layer2_c), FSQ_LEVEL, dtype=torch.float32)
        start_time = time.time()
        for batch_index, (imgs, _labels) in enumerate(train_loader, start=1):
            if (
                int(args.max_train_batches) > 0
                and batch_index > int(args.max_train_batches)
            ):
                break
            imgs = imgs.to(config.device, non_blocking=True)
            layer1_out, out = system.forward(imgs)
            losses = compute_losses(out, imgs)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(args.grad_clip_norm)
                )
            optimizer.step()
            update_metrics(metric_meters, out, layer1_out, imgs, losses)
            update_joint_counter(joint_counter, out["packed_codes"])
            update_bit_histogram(bit_histogram, out["codes"])

        last_metrics = finalize_metrics(
            metric_meters, joint_counter, bit_histogram, args
        )
        print(
            f"[stage3-fsq train {epoch:03d}/{int(args.epochs):03d}] "
            f"{display_metrics(last_metrics)} time={time.time() - start_time:.1f}s",
            flush=True,
        )

        checkpoint_metrics = last_metrics
        checkpoint_split = "train"
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, system, args, config.device)
            checkpoint_metrics = val_metrics
            checkpoint_split = "val"
            score = float(val_metrics["psnr_final"])
            print(
                f"[stage3-fsq val {epoch:03d}] {display_metrics(val_metrics)} "
                "score=psnr_final",
                flush=True,
            )
            if score > best:
                best = score
                save_checkpoint(
                    jsccf_io.ckpt_path(args, experiment_name(args), "best"),
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    metrics_split="val",
                    system=system,
                )

        last_checkpoint_metrics = checkpoint_metrics
        last_checkpoint_split = checkpoint_split

        if should_save_latest(args, epoch):
            save_checkpoint(
                jsccf_io.ckpt_path(args, experiment_name(args), "latest"),
                epoch=epoch,
                args=args,
                metrics=checkpoint_metrics,
                metrics_split=checkpoint_split,
                system=system,
            )

    save_checkpoint(
        jsccf_io.ckpt_path(args, experiment_name(args), "latest"),
        epoch=int(args.epochs),
        args=args,
        metrics=last_checkpoint_metrics or last_metrics,
        metrics_split=last_checkpoint_split,
        system=system,
    )


def smoke_shapes(args: argparse.Namespace, initialization: dict) -> None:
    seed_everything(int(args.seed))
    device = torch.device(
        "cuda:0"
        if (not bool(args.cpu)) and torch.cuda.is_available()
        else "cpu"
    )
    system = build_system(args, initialization, device)
    system.eval()
    imgs = torch.rand(
        int(args.smoke_batch_size), 3, 256, 256, device=device
    )
    layer1_out, out = system.forward(imgs)
    expected_latent = (
        int(args.smoke_batch_size),
        int(args.layer2_c),
        int(args.latent_h),
        int(args.latent_w),
    )
    expected_packed = (
        int(args.smoke_batch_size),
        (int(args.layer2_c) + 7) // 8,
        int(args.latent_h),
        int(args.latent_w),
    )
    expected_index = (
        int(args.smoke_batch_size),
        int(args.latent_h),
        int(args.latent_w),
    )
    expected_image = (int(args.smoke_batch_size), 3, 256, 256)
    index_shape = tuple(out["idx2"].shape) if "idx2" in out else None
    print(
        f"[smoke] layer1={args.layer1_arch} layer2={args.layer2_arch} "
        f"C={int(args.layer2_c)} x1={tuple(layer1_out['x1'].shape)} "
        f"z2={tuple(out['z2'].shape)} codes={tuple(out['codes'].shape)} "
        f"q2={tuple(out['q2'].shape)} packed={tuple(out['packed_codes'].shape)} "
        f"idx2={index_shape} "
        f"u2={tuple(out['u2'].shape)} final={tuple(out['final'].shape)}",
        flush=True,
    )
    for name in ("z2", "codes", "q2"):
        if tuple(out[name].shape) != expected_latent:
            raise RuntimeError(
                f"expected {name} {expected_latent}, got {tuple(out[name].shape)}"
            )
    if tuple(out["packed_codes"].shape) != expected_packed:
        raise RuntimeError(
            f"expected packed codes {expected_packed}, got {tuple(out['packed_codes'].shape)}"
        )
    if tuple(out["u2"].shape) != expected_image or tuple(out["final"].shape) != expected_image:
        raise RuntimeError(
            f"expected image outputs {expected_image}, got "
            f"u2={tuple(out['u2'].shape)} final={tuple(out['final'].shape)}"
        )
    if int(out["codes"].min().item()) < 0 or int(out["codes"].max().item()) > 1:
        raise RuntimeError("binary FSQ produced a code outside {0,1}")
    roundtrip_codes = system.codec.quantizer.unpack_codes(out["packed_codes"])
    if not torch.equal(roundtrip_codes, out["codes"]):
        raise RuntimeError("FSQ packed bytes -> binary codes roundtrip failed")
    roundtrip_q2 = system.codec.quantizer.packed_to_quantized(out["packed_codes"]).to(
        device=out["q2_hard"].device,
        dtype=out["q2_hard"].dtype,
    )
    roundtrip_error = float((roundtrip_q2 - out["q2_hard"]).abs().max().item())
    if roundtrip_error != 0.0:
        raise RuntimeError(f"FSQ packed bytes -> q2 roundtrip error is {roundtrip_error}")
    if system.codec.quantizer.supports_scalar_indices:
        if tuple(out["idx2"].shape) != expected_index:
            raise RuntimeError(
                f"expected scalar idx2 {expected_index}, got {tuple(out['idx2'].shape)}"
            )
        if int(out["idx2"].min().item()) < 0 or int(out["idx2"].max().item()) >= (
            1 << int(args.layer2_c)
        ):
            raise RuntimeError("mixed-radix FSQ index is out of range")
        scalar_roundtrip = system.codec.quantizer.indices_to_codes(out["idx2"])
        if not torch.equal(scalar_roundtrip, out["codes"]):
            raise RuntimeError("FSQ scalar index -> binary codes roundtrip failed")
    fixed_bits = int(args.layer2_c) * int(args.latent_h) * int(args.latent_w)
    smoke_joint_counter: Counter[bytes] = Counter()
    smoke_bit_histogram = torch.zeros(
        int(args.layer2_c), FSQ_LEVEL, dtype=torch.float32
    )
    update_joint_counter(smoke_joint_counter, out["packed_codes"])
    update_bit_histogram(smoke_bit_histogram, out["codes"])
    smoke_rate_metrics = finalize_metrics(
        meters(METRIC_NAMES), smoke_joint_counter, smoke_bit_histogram, args
    )
    if int(smoke_rate_metrics["fixed_bits_per_image"]) != fixed_bits:
        raise RuntimeError(
            "dynamic-C rate accounting mismatch: "
            f"expected {fixed_bits}, got {smoke_rate_metrics['fixed_bits_per_image']}"
        )
    losses = compute_losses(out, imgs)
    losses["loss"].backward()
    for label, module in (("E1", system.e1), ("D1", system.d1)):
        if any(parameter.grad is not None for parameter in module.parameters()):
            raise RuntimeError(f"frozen {label} unexpectedly received a gradient")
    gradient_norms: dict[str, float] = {}
    trainable_modules = {
        "E2": system.codec.e2,
        "D2": system.codec.d2,
        "combiner": system.codec.combiner,
        "FSQ": system.codec.quantizer,
    }
    for label, module in trainable_modules.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            if label == "FSQ" and trainable_state(module) == "no_params":
                continue
            raise RuntimeError(f"trainable {label} received no gradient")
        if any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
            raise RuntimeError(f"trainable {label} received a non-finite gradient")
        gradient_norm = math.sqrt(
            sum(float(gradient.detach().float().square().sum().item()) for gradient in gradients)
        )
        if gradient_norm <= 0.0:
            raise RuntimeError(f"trainable {label} gradient norm is zero")
        gradient_norms[label] = gradient_norm
    print(
        f"[smoke] binary FSQ contract passed: {int(args.layer2_c)}x1bit, "
        f"{fixed_bits} bits/image; packed/q roundtrip exact; "
        f"gradient_norms={gradient_norms}",
        flush=True,
    )


def validate_args(args: argparse.Namespace) -> None:
    check_jsccf_args(args)
    if str(args.codec_init) == "auto":
        args.codec_init = (
            "fresh"
            if str(args.layer2_arch) == "bar" and not args.layer2_ckpt
            else "compatible"
        )
    channels = int(args.layer2_c)
    if channels < 1:
        raise ValueError("--layer2-c must be >= 1")
    levels = parse_fsq_levels(args.fsq_levels, channels)
    if str(args.variant) != "combiner":
        raise ValueError("this entrypoint requires D2 output and x1 to enter combiner")
    if levels != [FSQ_LEVEL] * channels:
        raise ValueError(
            f"--fsq-levels must be 2 for every one of the {channels} channels "
            "(one bit/channel)"
        )
    if int(args.latent_h) != 16 or int(args.latent_w) != 16:
        raise ValueError("E2 spatial output is fixed to 16x16")
    if int(args.latent_ch) != 16 or int(args.c1_ch) != 16:
        raise ValueError("the provided frozen Layer1 checkpoints require z1 C=16")
    for name in (
        "layer1_cnn_base_ch",
        "layer2_cnn_base_ch",
        "bar_patch_size",
        "bar_encoder_width",
        "bar_encoder_heads",
        "bar_encoder_mlp_width",
        "bar_decoder_width",
        "bar_decoder_heads",
        "bar_decoder_mlp_width",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    for name in (
        "layer1_cnn_num_res",
        "layer2_cnn_num_res",
        "bar_encoder_layers",
        "bar_decoder_layers",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if str(args.layer2_arch) == "bar":
        if int(args.bar_patch_size) != 16:
            raise ValueError("BAR Layer2 requires --bar-patch-size=16 for a 16x16 grid")
        if (
            str(args.bar_encoder_backend) == "native"
            and int(args.bar_encoder_width) % int(args.bar_encoder_heads) != 0
        ):
            raise ValueError("BAR encoder width must be divisible by encoder heads")
        if int(args.bar_decoder_width) % int(args.bar_decoder_heads) != 0:
            raise ValueError("BAR decoder width must be divisible by decoder heads")
    if int(args.epochs) < 1:
        raise ValueError("--epochs must be >= 1")
    for name in ("batch_size", "test_batch", "smoke_batch_size"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    for name in (
        "num_workers",
        "val_num_workers",
        "val_every",
        "latest_every",
        "max_train_batches",
        "max_val_batches",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be > 0")
    if float(args.weight_decay) < 0.0:
        raise ValueError("--weight-decay must be >= 0")
    if float(args.grad_clip_norm) < 0.0:
        raise ValueError("--grad-clip-norm must be >= 0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--layer1-arch",
        dest="layer1_arch",
        type=str,
        default="swin",
        choices=["swin", "cnn"],
        help="Frozen Layer1 encoder/decoder family.",
    )
    parser.add_argument(
        "--layer2-arch",
        type=str,
        default="bar",
        choices=["swin", "cnn", "bar"],
        help="Trainable Layer2 encoder/decoder family.",
    )
    parser.add_argument(
        "--swin-codec",
        type=str,
        default="compressed",
        choices=["compressed"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--variant", type=str, default="combiner", choices=["combiner"])
    parser.add_argument("--version", type=str, default="c-16")
    parser.add_argument(
        "--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K"
    )
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument(
        "--layer1-ckpt",
        type=str,
        default="",
        help="Strict frozen Layer1 checkpoint; empty selects the family default.",
    )
    parser.add_argument(
        "--layer2-ckpt",
        type=str,
        default="",
        help=(
            "Optional same-family Layer2 initialization source. Empty selects the "
            "Swin/CNN default in compatible mode; BAR defaults to fresh."
        ),
    )
    parser.add_argument(
        "--codec-init",
        type=str,
        default="auto",
        choices=["auto", "compatible", "fresh"],
        help="Auto uses compatible Swin/CNN initialization and fresh BAR initialization.",
    )

    parser.add_argument(
        "--layer2-c",
        "--fsq-d",
        dest="layer2_c",
        type=int,
        default=16,
        help="E2/FSQ/D2 channel count C; --fsq-d is a compatibility alias.",
    )
    parser.add_argument(
        "--fsq-levels",
        type=str,
        default=str(FSQ_LEVEL),
        help="Fixed binary level count; one value expands to all Layer2 C channels.",
    )
    parser.add_argument(
        "--no-pre-norm",
        action="store_true",
        help="Disable trainable GroupNorm before tanh and binary FSQ.",
    )

    parser.add_argument("--latent-ch", type=int, default=16)
    parser.add_argument("--c1-ch", type=int, default=16)
    parser.add_argument("--latent-h", type=int, default=16)
    parser.add_argument("--latent-w", type=int, default=16)

    parser.add_argument("--layer1-cnn-base-ch", type=int, default=16)
    parser.add_argument("--layer1-cnn-num-res", type=int, default=2)
    parser.add_argument(
        "--layer2-cnn-base-ch",
        "--cnn-base-ch",
        dest="layer2_cnn_base_ch",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--layer2-cnn-num-res",
        "--cnn-num-res",
        dest="layer2_cnn_num_res",
        type=int,
        default=2,
    )

    parser.add_argument("--bar-patch-size", type=int, default=16)
    parser.add_argument(
        "--bar-encoder-backend",
        type=str,
        default="siglip2",
        choices=["native", "siglip2"],
        help=(
            "siglip2 follows BAR's pretrained encoder path; native is the explicit "
            "offline/from-scratch topology alternative."
        ),
    )
    parser.add_argument(
        "--bar-siglip-model",
        type=str,
        default="google/siglip2-so400m-patch16-naflex",
    )
    parser.add_argument(
        "--bar-allow-download",
        action="store_true",
        help="Allow Transformers to download --bar-siglip-model when it is not cached.",
    )
    parser.add_argument("--bar-encoder-width", type=int, default=1152)
    parser.add_argument("--bar-encoder-layers", type=int, default=27)
    parser.add_argument("--bar-encoder-heads", type=int, default=16)
    parser.add_argument("--bar-encoder-mlp-width", type=int, default=4304)
    parser.add_argument("--bar-decoder-width", type=int, default=1024)
    parser.add_argument("--bar-decoder-layers", type=int, default=24)
    parser.add_argument("--bar-decoder-heads", type=int, default=16)
    parser.add_argument("--bar-decoder-mlp-width", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--test-batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--latest-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--no-val-ablation",
        dest="val_ablation",
        action="store_false",
        help="Skip all-code-0 and spatial-shuffle validation ablations.",
    )
    parser.set_defaults(val_ablation=True)
    parser.add_argument("--eval-init-only", action="store_true")
    parser.add_argument("--smoke-shapes", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    bar_stage_part = (
        f"_{args.bar_encoder_backend}" if str(args.layer2_arch) == "bar" else ""
    )
    args.stage = (
        f"stage3_fsq_l1_{args.layer1_arch}_l2_{args.layer2_arch}_"
        f"c{int(args.layer2_c)}{bar_stage_part}_binary_direct"
    )
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(
            Path(resolve_path(args.save_dir))
            / f"{experiment_name(args)}_jscc_f_"
            f"{jsccf_io.safe_artifact_name(args.version)}.log"
        )
    setup_log_file(args.log_file)
    initialization = load_initialization_checkpoints(args)
    write_json(
        Path(resolve_path(args.save_dir))
        / f"{experiment_name(args)}_jscc_f_"
        f"{jsccf_io.safe_artifact_name(args.version)}_args.json",
        {key: value for key, value in vars(args).items() if not key.startswith("_")},
    )
    if bool(args.smoke_shapes):
        smoke_shapes(args, initialization)
        return
    train(args, initialization)


if __name__ == "__main__":
    main()
