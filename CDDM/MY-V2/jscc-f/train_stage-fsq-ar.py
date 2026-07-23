#!/usr/bin/env python3
"""iFSQ/LlamaGen-style conditional raster autoregression for direct Layer-2 FSQ.

Training-only oracle graph::

    image -> frozen Layer1 -> (y1=z1, x1)
    (image, x1) -> frozen E2/FSQ -> oracle idx2 [B,16,16]

Receiver/deployment graph::

    prefix:       (y1, x1) -> spatial condition prefix -> cached decoder -> idx2_hat
    per-token-y1: y1[i,j] -> fuse with BOS/q2[i-1] at position (i,j) -> cached decoder -> idx2_hat
    idx2_hat -> frozen FSQ inverse -> q2_hat -> frozen D2/combiner -> x2_hat

The raw image, oracle z2/q2, and future indices never enter the deployment
forward.  The three supported tokenizers use FSQ levels [5,5,5], [9,9,9], or
[17,17,17] and are selected with ``--layer2-level``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time
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


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


direct = load_module(
    "jsccf_stage_ar_direct_support",
    THIS_DIR / "explore" / "train_layer2_fsq_direct.py",
)
base = direct.base

# The exploration wrapper currently delegates Layer-1 construction through
# ``base.load_script_module``.  Some local canonical Stage-3 revisions omit
# that utility, so provide the same path-local loader here rather than making
# this receiver depend on an unrelated entrypoint revision.
if not hasattr(base, "load_script_module"):
    def _load_stage_script(name: str, filename: str) -> ModuleType:
        return load_module(name, THIS_DIR / filename)

    base.load_script_module = _load_stage_script


DEFAULT_LAYER1_CHECKPOINT = "MY-V2/jscc-f/checkpoints/jscc_f_cnn_layer1_cnn_best.pth"
DEFAULT_LAYER2_CHECKPOINTS = {
    5: (
        "MY-V2/jscc-f/explore/checkpoints-direct/"
        "direct-cnn-d3-l5x5x5-group-compatible-blend-e100/"
        "jscc_f_direct-cnn-d3-l5x5x5-group-compatible-blend-e100_"
        "layer2_fsq_direct_cnn_d3_l5x5x5_group_compatible_blend_best.pth"
    ),
    9: (
        "MY-V2/jscc-f/explore/checkpoints-direct/"
        "direct-cnn-d3-l9x9x9-group-compatible-blend-e100/"
        "jscc_f_direct-cnn-d3-l9x9x9-group-compatible-blend-e100_"
        "layer2_fsq_direct_cnn_d3_l9x9x9_group_compatible_blend_best.pth"
    ),
    17: (
        "MY-V2/jscc-f/explore/checkpoints-direct/"
        "direct-cnn-d3-l17x17x17-group-compatible-blend-e100/"
        "jscc_f_direct-cnn-d3-l17x17x17-group-compatible-blend-e100_"
        "layer2_fsq_direct_cnn_d3_l17x17x17_group_compatible_blend_best.pth"
    ),
}


def group_count(channels: int) -> int:
    for candidate in (16, 8, 4, 2, 1):
        if int(channels) % candidate == 0:
            return candidate
    return 1


class LegacyIFSQQuantizer(nn.Module):
    """The 3-D scalar FSQ used by the direct Layer2 checkpoints.

    ``train-stage3-fsq.py`` is currently the binary-FSQ/BAR entrypoint, so its
    ``IFSQQuantizer`` deliberately rejects ``[5,5,5]``.  The direct CNN
    artifacts were trained with the earlier arbitrary-level implementation;
    keep this small, state-dict-compatible copy local to the AR loader rather
    than changing the user's canonical Stage-3 entrypoint.
    """

    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        channels: int,
        use_pre_norm: bool = True,
    ) -> None:
        super().__init__()
        parsed = base.parse_fsq_levels(levels, int(channels))
        self.channels = int(channels)
        self.register_buffer("levels", torch.tensor(parsed, dtype=torch.long))
        multipliers: list[int] = []
        running = 1
        for level in reversed(parsed[1:]):
            running *= int(level)
            multipliers.append(running)
        self.register_buffer(
            "multipliers",
            torch.tensor(list(reversed(multipliers)) + [1], dtype=torch.long),
        )
        self.pre_norm = (
            nn.GroupNorm(1, self.channels, affine=True)
            if bool(use_pre_norm)
            else nn.Identity()
        )

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 4 or int(codes.shape[1]) != self.channels:
            raise ValueError(
                f"expected FSQ codes [B,{self.channels},H,W], got {tuple(codes.shape)}"
            )
        multipliers = self.multipliers.to(device=codes.device).view(1, self.channels, 1, 1)
        return (codes.long() * multipliers).sum(dim=1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.ndim != 4 or int(z.shape[1]) != self.channels:
            raise ValueError(f"expected z3 [B,{self.channels},H,W], got {tuple(z.shape)}")
        z_norm = torch.tanh(self.pre_norm(z))
        levels = self.levels.to(device=z.device, dtype=z_norm.dtype).view(1, self.channels, 1, 1)
        span = (levels - 1.0).clamp_min(1.0)
        positions = (z_norm + 1.0) * 0.5 * span
        codes_float = (
            positions + (positions.round() - positions).detach()
        ).clamp_min(0.0).minimum(span)
        codes = codes_float.detach().long()
        q_hard = codes_float / span * 2.0 - 1.0
        q3 = z_norm + (q_hard - z_norm).detach()
        return {
            "z3_norm": z_norm,
            "q3": q3,
            "q3_hard": q_hard.detach(),
            "codes": codes,
            "idx3": self.codes_to_indices(codes),
            "fsq_mse": F.mse_loss(q_hard.detach().float(), z_norm.detach().float()),
        }


class DirectFSQTokenizer(nn.Module):
    """Exact E2 -> arbitrary-level FSQ -> D2 graph in direct checkpoints."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        super().__init__()
        self.fsq_d = int(args.fsq_d)
        self.levels = base.parse_fsq_levels(args.fsq_levels, self.fsq_d)
        if str(args.arch) != "cnn":
            raise ValueError("stage-ar direct checkpoint loader currently supports CNN only")
        self.e3 = base.CNNAnalysisEncoder(
            base_ch=int(args.e3_base_ch),
            bottleneck_ch=self.fsq_d,
            num_res=int(args.e3_num_res),
        )
        self.e3.stem = base.ConvNormAct(
            6, int(args.e3_base_ch), kernel=3, stride=1
        )
        self.d3 = base.CNNBottleneckDecoder(
            base_ch=int(args.d3_base_ch),
            bottleneck_ch=self.fsq_d,
            num_res=int(args.d3_num_res),
            output_activation="none",
        )
        self.quantizer = LegacyIFSQQuantizer(
            self.levels,
            channels=self.fsq_d,
            use_pre_norm=not bool(args.no_pre_norm),
        )
        self.to(device)

    def encode(self, img: torch.Tensor, x1: torch.Tensor) -> dict[str, torch.Tensor]:
        z3 = base.encode_tensor(self.e3, torch.cat([x1, img], dim=1))
        quantized = self.quantizer(z3)
        quantized["z3"] = z3
        return quantized

    def decode(
        self,
        q3: torch.Tensor,
        x1: torch.Tensor,
        z1: torch.Tensor,
        combiner: nn.Module,
    ) -> dict[str, torch.Tensor]:
        del z1  # condition_mode=none is part of the direct checkpoint contract.
        u2_raw = self.d3(q3)
        u2_hat = u2_raw.clamp(0.0, 1.0)
        return {
            "d3_in": q3,
            "u2_raw": u2_raw,
            "u2_hat": u2_hat,
            "final": combiner(x1, u2_hat),
        }

    @staticmethod
    def shuffle_q3(q3: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = q3.shape
        flat = q3.permute(0, 2, 3, 1).reshape(-1, channels)
        perm = torch.randperm(flat.shape[0], device=q3.device)
        return flat[perm].view(bsz, height, width, channels).permute(0, 3, 1, 2).contiguous()


class ConditionResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.silu(self.norm1(x))
        residual = self.conv1(residual)
        residual = self.conv2(F.silu(self.norm2(residual)))
        return x + residual


class ReceiverConditionEncoder(nn.Module):
    """Fuse received Layer1 latent y1 and reconstruction x1 at 16x16."""

    def __init__(self, y1_channels: int, hidden: int, blocks: int) -> None:
        super().__init__()
        self.y1_stem = nn.Sequential(
            nn.Conv2d(int(y1_channels), hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.x1_stem = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.body = nn.Sequential(*(ConditionResidualBlock(hidden) for _ in range(int(blocks))))
        self.out_norm = nn.GroupNorm(group_count(hidden), hidden)

    def forward(self, y1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        if y1.ndim != 4 or x1.ndim != 4:
            raise ValueError(f"condition requires y1/x1 BCHW tensors, got {y1.shape} and {x1.shape}")
        x1_small = F.interpolate(x1, size=tuple(y1.shape[-2:]), mode="bilinear", align_corners=False)
        y_feature = self.y1_stem(y1)
        x_feature = self.x1_stem(x1_small)
        fused = self.fuse(
            torch.cat(
                [y_feature, x_feature, y_feature - x_feature, y_feature * x_feature],
                dim=1,
            )
        )
        return F.silu(self.out_norm(self.body(fused)))


class PerTokenY1ConditionEncoder(nn.Module):
    """Keep the received y1 grid aligned one-to-one with the q2 token grid.

    This encoder deliberately does not consume x1.  It maps y1[B,C,16,16] to
    a hidden feature at every spatial site, which is subsequently fused with
    the decoder input that predicts the q2 token at that same site.
    """

    def __init__(self, y1_channels: int, hidden: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(int(y1_channels), hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.body = nn.Sequential(*(ConditionResidualBlock(hidden) for _ in range(int(blocks))))
        self.out_norm = nn.GroupNorm(group_count(hidden), hidden)

    def forward(self, y1: torch.Tensor) -> torch.Tensor:
        if y1.ndim != 4:
            raise ValueError(f"per-token y1 condition requires BCHW, got {tuple(y1.shape)}")
        return F.silu(self.out_norm(self.body(self.stem(y1))))


class RMSNorm(nn.Module):
    """RMSNorm used by the iFSQ/LlamaGen autoregressive transformer."""

    def __init__(self, hidden: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(hidden)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.type_as(value) * self.weight


def _make_multiple(value: int, multiple: int = 256) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


class SwiGLUFeedForward(nn.Module):
    """LlamaGen/iFSQ SwiGLU MLP, including its two input projections."""

    def __init__(self, hidden: int, multiplier: float, dropout: float) -> None:
        super().__init__()
        inner = _make_multiple(int(round(int(hidden) * float(multiplier))))
        self.w1 = nn.Linear(hidden, inner, bias=False)
        self.w3 = nn.Linear(hidden, inner, bias=False)
        self.w2 = nn.Linear(inner, hidden, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(value)) * self.w3(value)))


def precompute_2d_rope(grid_size: int, head_dim: int, prefix_tokens: int, base: float = 10000.0) -> torch.Tensor:
    """Copy the 2-D RoPE layout from iFSQ/LlamaGen.

    The optional condition prefix deliberately has zero rotation, as LlamaGen
    does for its class-conditioning prefix.  The 16x16 FSQ positions receive
    rasterized 2-D rotary coordinates after that prefix.  In per-token-y1
    mode ``prefix_tokens`` is zero, so the BOS/q2 input at position i directly
    receives the RoPE coordinate for q2[i].
    """
    if int(head_dim) % 4 != 0:
        raise ValueError("iFSQ 2-D RoPE requires --hidden/--heads to be divisible by 4")
    half_dim = int(head_dim) // 2
    frequencies = 1.0 / (float(base) ** (torch.arange(0, half_dim, 2).float() / half_dim))
    positions = torch.arange(int(grid_size), dtype=torch.float32)
    phase = torch.outer(positions, frequencies)
    grid_phase = torch.cat(
        [
            phase[:, None, :].expand(-1, int(grid_size), -1),
            phase[None, :, :].expand(int(grid_size), -1, -1),
        ],
        dim=-1,
    )
    spatial = torch.stack([torch.cos(grid_phase), torch.sin(grid_phase)], dim=-1).flatten(0, 1)
    prefix = torch.zeros(int(prefix_tokens), int(head_dim) // 2, 2)
    return torch.cat([prefix, spatial], dim=0)


def apply_rope(value: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Apply iFSQ/LlamaGen real-valued RoPE to [B,T,H,D] queries or keys."""
    shaped = value.float().reshape(*value.shape[:-1], -1, 2)
    rotation = frequencies.view(1, shaped.shape[1], 1, shaped.shape[3], 2)
    rotated = torch.stack(
        [
            shaped[..., 0] * rotation[..., 0] - shaped[..., 1] * rotation[..., 1],
            shaped[..., 1] * rotation[..., 0] + shaped[..., 0] * rotation[..., 1],
        ],
        dim=-1,
    )
    return rotated.flatten(3).type_as(value)


class KVCache(nn.Module):
    """Fixed-size KV cache following iFSQ/LlamaGen's prefill/decode contract."""

    def __init__(self, batch: int, length: int, heads: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        shape = (int(batch), int(heads), int(length), int(head_dim))
        self.register_buffer("keys", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("values", torch.zeros(shape, device=device, dtype=dtype), persistent=False)

    def update(self, positions: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(keys.shape[0]) != int(self.keys.shape[0]):
            raise ValueError("KV cache batch differs from the prefill batch")
        self.keys[:, :, positions] = keys.to(self.keys.dtype)
        self.values[:, :, positions] = values.to(self.values.dtype)
        return self.keys, self.values


class IFSQAttention(nn.Module):
    """LlamaGen self-attention with 2-D RoPE and an explicit KV cache."""

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        if int(hidden) % int(heads) != 0:
            raise ValueError("--hidden must be divisible by --heads")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.head_dim = int(hidden) // int(heads)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.cache: KVCache | None = None

    def setup_cache(self, batch: int, length: int, device: torch.device, dtype: torch.dtype) -> None:
        self.cache = KVCache(batch, length, self.heads, self.head_dim, device, dtype)

    def forward(
        self,
        value: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = value.shape
        query, key, val = self.qkv(value).chunk(3, dim=-1)
        query = apply_rope(query.view(batch, length, self.heads, self.head_dim), frequencies).transpose(1, 2)
        key = apply_rope(key.view(batch, length, self.heads, self.head_dim), frequencies).transpose(1, 2)
        val = val.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        if positions is not None:
            if self.cache is None:
                raise RuntimeError("KV cache was not initialized before autoregressive decoding")
            key, val = self.cache.update(positions, key, val)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            val,
            attn_mask=causal_mask,
            is_causal=positions is None,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.hidden)
        return self.dropout(self.proj(attended))


class IFSQTransformerBlock(nn.Module):
    """Pre-norm LlamaGen block: RMSNorm, RoPE attention, and SwiGLU."""

    def __init__(self, hidden: int, heads: int, ff_mult: float, dropout: float) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden)
        self.attention = IFSQAttention(hidden, heads, dropout)
        self.ffn_norm = RMSNorm(hidden)
        self.feed_forward = SwiGLUFeedForward(hidden, ff_mult, dropout)

    def forward(
        self,
        value: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = value + self.attention(self.attention_norm(value), frequencies, positions=positions, causal_mask=causal_mask)
        return value + self.feed_forward(self.ffn_norm(value))


class IFSQPrefixRasterAR(nn.Module):
    """Conditional LlamaGen-style AR over a 16x16 joint-FSQ index map.

    ``prefix`` follows iFSQ/LlamaGen: a 16x16 condition grid is prepended to
    the causal q2 sequence.  ``per-token-y1`` instead keeps y1 spatially
    aligned with q2: the feature for site i is fused into the BOS/previous-q2
    input that predicts q2[i].  Both variants are decoder-only and use no
    cross-attention.
    """

    def __init__(
        self,
        y1_channels: int,
        height: int,
        width: int,
        vocabulary: int,
        *,
        hidden: int,
        layers: int,
        heads: int,
        condition_blocks: int,
        ff_mult: float,
        dropout: float,
        condition_mode: str = "prefix",
    ) -> None:
        super().__init__()
        if int(hidden) % int(heads) != 0:
            raise ValueError("--hidden must be divisible by --heads")
        if str(condition_mode) not in {"prefix", "per-token-y1"}:
            raise ValueError(f"unknown condition mode: {condition_mode}")
        self.height = int(height)
        self.width = int(width)
        self.tokens = self.height * self.width
        self.vocabulary = int(vocabulary)
        self.hidden = int(hidden)
        self.condition_mode = str(condition_mode)
        self.prefix_tokens = self.tokens + 1 if self.condition_mode == "prefix" else 0
        self.total_tokens = self.prefix_tokens + self.tokens

        # Instantiate only the selected branch.  This preserves strict loading
        # of the existing prefix checkpoints, while keeping per-token-y1 a
        # genuinely distinct architecture that must be trained from scratch.
        if self.condition_mode == "prefix":
            self.condition_encoder = ReceiverConditionEncoder(y1_channels, hidden, condition_blocks)
            self.condition_position = nn.Parameter(torch.empty(1, self.tokens, hidden))
            nn.init.normal_(self.condition_position, std=0.02)
        else:
            self.per_token_condition_encoder = PerTokenY1ConditionEncoder(y1_channels, hidden, condition_blocks)
            self.per_token_position = nn.Parameter(torch.empty(1, self.tokens, hidden))
            self.per_token_fusion = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden, bias=False),
            )
            nn.init.normal_(self.per_token_position, std=0.02)
            # Start as additive y1/token fusion.  The residual MLP then learns
            # a richer content-dependent fusion without disrupting early AR.
            nn.init.zeros_(self.per_token_fusion[-1].weight)
        self.bos_embedding = nn.Parameter(torch.empty(1, 1, hidden))
        self.token_embedding = nn.Embedding(self.vocabulary, hidden)
        self.token_dropout = nn.Dropout(float(dropout))
        self.blocks = nn.ModuleList(
            IFSQTransformerBlock(hidden, heads, ff_mult, dropout) for _ in range(int(layers))
        )
        self.norm = RMSNorm(hidden)
        self.head = nn.Linear(hidden, self.vocabulary, bias=False)
        self.register_buffer(
            "rope_frequencies",
            precompute_2d_rope(self.height, int(hidden) // int(heads), self.prefix_tokens),
            persistent=False,
        )
        self.causal_mask: torch.Tensor | None = None
        nn.init.normal_(self.bos_embedding, std=0.02)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.constant_(self.head.weight, 0.0)  # iFSQ/LlamaGen output initialization

    def encode_prefix_condition(self, y1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        if self.condition_mode != "prefix":
            raise RuntimeError("encode_prefix_condition is valid only for --condition-mode=prefix")
        feature = self.condition_encoder(y1, x1)
        if tuple(feature.shape[-2:]) != (self.height, self.width):
            feature = F.interpolate(feature, size=(self.height, self.width), mode="bilinear", align_corners=False)
        return feature.flatten(2).transpose(1, 2) + self.condition_position

    def encode_per_token_y1(self, y1: torch.Tensor) -> torch.Tensor:
        if self.condition_mode != "per-token-y1":
            raise RuntimeError("encode_per_token_y1 is valid only for --condition-mode=per-token-y1")
        feature = self.per_token_condition_encoder(y1)
        if tuple(feature.shape[-2:]) != (self.height, self.width):
            feature = F.interpolate(feature, size=(self.height, self.width), mode="bilinear", align_corners=False)
        return feature.flatten(2).transpose(1, 2) + self.per_token_position

    def fuse_per_token_y1(self, token_input: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if tuple(token_input.shape) != tuple(condition.shape):
            raise ValueError(
                "per-token fusion requires matching [B,T,H] token and y1 condition tensors, got "
                f"{tuple(token_input.shape)} and {tuple(condition.shape)}"
            )
        return token_input + condition + self.per_token_fusion(torch.cat([token_input, condition], dim=-1))

    def setup_caches(self, batch: int, device: torch.device, dtype: torch.dtype) -> None:
        for block in self.blocks:
            block.attention.setup_cache(batch, self.total_tokens, device, dtype)
        mask = torch.tril(torch.ones(self.total_tokens, self.total_tokens, dtype=torch.bool, device=device))
        self.causal_mask = mask.unsqueeze(0).expand(int(batch), -1, -1)

    def _run(self, value: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if positions is None:
            frequencies = self.rope_frequencies[: value.shape[1]].to(value.device)
            causal_mask = None
        else:
            if self.causal_mask is None:
                raise RuntimeError("call setup_caches before cached AR decoding")
            frequencies = self.rope_frequencies[positions].to(value.device)
            causal_mask = self.causal_mask[: value.shape[0], None, positions]
        value = self.token_dropout(value)
        for block in self.blocks:
            value = block(value, frequencies, positions=positions, causal_mask=causal_mask)
        return value

    def forward_sequence(self, y1: torch.Tensor, x1: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Teacher-forced LlamaGen shift for a [B,T] index sequence."""
        if indices.ndim != 2 or not 1 <= int(indices.shape[1]) <= self.tokens:
            raise ValueError(f"FSQ sequence must be [B,T] with 1<=T<={self.tokens}, got {tuple(indices.shape)}")
        if int(indices.min()) < 0 or int(indices.max()) >= self.vocabulary:
            raise ValueError("FSQ index is outside the selected vocabulary")
        batch, length = indices.shape
        bos = self.bos_embedding.expand(int(batch), -1, -1)
        history = self.token_embedding(indices[:, :-1].long()) if int(length) > 1 else bos[:, :0]
        token_inputs = torch.cat([bos, history], dim=1)
        if self.condition_mode == "prefix":
            condition = self.encode_prefix_condition(y1, x1)
            value = torch.cat([condition, token_inputs], dim=1)
            start = self.prefix_tokens - 1
        else:
            condition = self.encode_per_token_y1(y1)
            value = self.fuse_per_token_y1(token_inputs, condition[:, : int(length)])
            start = 0
        hidden = self._run(value)
        return self.head(self.norm(hidden[:, start : start + int(length)]))

    def forward_teacher(self, y1: torch.Tensor, x1: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 3 or tuple(indices.shape[-2:]) != (self.height, self.width):
            raise ValueError(
                f"oracle FSQ indices must be [B,{self.height},{self.width}], got {tuple(indices.shape)}"
            )
        return self.forward_sequence(y1, x1, indices.flatten(1).long())

    @staticmethod
    def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        filtered = logits.clone()
        if int(top_k) > 0:
            keep = min(int(top_k), int(filtered.shape[-1]))
            threshold = filtered.topk(keep, dim=-1).values[..., -1, None]
            filtered.masked_fill_(filtered < threshold, float("-inf"))
        if float(top_p) < 1.0:
            sorted_logits, sorted_indices = filtered.sort(dim=-1, descending=True)
            cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative > float(top_p)
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            filtered.scatter_(1, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
        return filtered

    @classmethod
    def _sample(cls, logits: torch.Tensor, temperature: float, top_k: int, top_p: float, sample_logits: bool) -> torch.Tensor:
        if float(temperature) <= 0.0:
            raise ValueError("--temperature must be positive")
        filtered = cls._filter_logits(logits / float(temperature), top_k, top_p)
        if not sample_logits:
            return filtered.argmax(dim=-1)
        return torch.multinomial(filtered.softmax(dim=-1), num_samples=1).squeeze(1)

    @torch.no_grad()
    def generate(
        self,
        y1: torch.Tensor,
        x1: torch.Tensor,
        *,
        steps: int | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        sample_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached greedy/sample AR; prefix prefill or position-aligned y1 fusion."""
        total = self.tokens if steps is None else int(steps)
        if not 1 <= total <= self.tokens:
            raise ValueError(f"generation steps must be in [1,{self.tokens}], got {total}")
        if not 0.0 < float(top_p) <= 1.0 or int(top_k) < 0:
            raise ValueError("--top-p must be in (0,1] and --top-k must be non-negative")
        batch = int(y1.shape[0])
        if self.condition_mode == "prefix":
            condition = self.encode_prefix_condition(y1, x1)
            prefill = torch.cat([condition, self.bos_embedding.expand(batch, -1, -1)], dim=1)
            prefill_positions = torch.arange(self.prefix_tokens, device=y1.device)
        else:
            condition = self.encode_per_token_y1(y1)
            prefill = self.fuse_per_token_y1(
                self.bos_embedding.expand(batch, -1, -1),
                condition[:, :1],
            )
            prefill_positions = torch.zeros(1, dtype=torch.long, device=y1.device)
        self.setup_caches(batch, y1.device, prefill.dtype)
        hidden = self._run(prefill, prefill_positions)
        logits = self.head(self.norm(hidden[:, -1]))
        logits_out: list[torch.Tensor] = [logits]
        generated: list[torch.Tensor] = [self._sample(logits, temperature, top_k, top_p, sample_logits)]
        for step in range(1, total):
            position_value = self.prefix_tokens + step - 1 if self.condition_mode == "prefix" else step
            position = torch.tensor([position_value], device=y1.device)
            current = self.token_embedding(generated[-1]).unsqueeze(1)
            if self.condition_mode == "per-token-y1":
                current = self.fuse_per_token_y1(current, condition[:, step : step + 1])
            hidden = self._run(current, position)
            logits = self.head(self.norm(hidden[:, 0]))
            logits_out.append(logits)
            generated.append(self._sample(logits, temperature, top_k, top_p, sample_logits))
        return torch.stack(logits_out, dim=1), torch.stack(generated, dim=1)


class FrozenFSQSystem(nn.Module):
    """Frozen Layer1 plus selected direct-FSQ Layer2 oracle/receiver."""

    def __init__(self, bundle: direct.DirectBundle, levels: list[int], height: int, width: int) -> None:
        super().__init__()
        self.e1 = bundle.e1
        self.d1 = bundle.d1
        self.tokenizer = bundle.tokenizer
        self.combiner = bundle.combiner
        self.height = int(height)
        self.width = int(width)
        parsed = [int(value) for value in levels]
        multipliers: list[int] = []
        for dimension in range(len(parsed)):
            multiplier = 1
            for following in parsed[dimension + 1 :]:
                multiplier *= int(following)
            multipliers.append(multiplier)
        self.register_buffer("levels", torch.tensor(parsed, dtype=torch.long))
        self.register_buffer("multipliers", torch.tensor(multipliers, dtype=torch.long))
        vocabulary = math.prod(parsed)
        ids = torch.arange(vocabulary, dtype=torch.long)
        code_columns = [(ids // multiplier) % level for level, multiplier in zip(parsed, multipliers)]
        codes = torch.stack(code_columns, dim=1).float()
        level_values = torch.tensor(parsed, dtype=torch.float32).view(1, -1)
        table = codes / (level_values - 1.0) * 2.0 - 1.0
        self.register_buffer("fsq_table", table)
        self.requires_grad_(False)
        self.eval()

    @property
    def vocabulary(self) -> int:
        return int(self.fsq_table.shape[0])

    @property
    def fsq_d(self) -> int:
        return int(self.fsq_table.shape[1])

    def train(self, mode: bool = True):
        # This container is an immutable oracle/receiver even while AR trains.
        return super().train(False)

    @torch.no_grad()
    def oracle_targets(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        y1 = base.encode_tensor(self.e1, images)
        x1 = self.d1(y1).clamp(0.0, 1.0)
        encoded = self.tokenizer.encode(images, x1)
        decoded = self.tokenizer.decode(encoded["q3"], x1, y1, self.combiner)
        indices = encoded["idx3"].long()
        if tuple(indices.shape[-2:]) != (self.height, self.width):
            raise RuntimeError(f"FSQ checkpoint produced index map {tuple(indices.shape)}, expected 16x16")
        return {
            "y1": y1.detach(),
            "x1": x1.detach(),
            "indices": indices.detach(),
            "q2": encoded["q3_hard"].detach(),
            "oracle": decoded["final"].detach(),
        }

    def indices_to_q(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 3 or tuple(indices.shape[-2:]) != (self.height, self.width):
            raise ValueError(f"indices must be [B,{self.height},{self.width}], got {tuple(indices.shape)}")
        if int(indices.min()) < 0 or int(indices.max()) >= self.vocabulary:
            raise ValueError("predicted FSQ index outside vocabulary")
        flat = self.fsq_table[indices.long()]
        return flat.permute(0, 3, 1, 2).contiguous()

    def logits_to_soft_q(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if logits.ndim != 3 or int(logits.shape[1]) != self.height * self.width:
            raise ValueError(f"logits must be [B,{self.height * self.width},K], got {tuple(logits.shape)}")
        probabilities = F.softmax(logits.float() / float(temperature), dim=-1)
        flat = torch.matmul(probabilities, self.fsq_table.float())
        return flat.view(int(logits.shape[0]), self.height, self.width, self.fsq_d).permute(0, 3, 1, 2)

    def decode_q(self, q2: torch.Tensor, y1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(q2, x1, y1, self.combiner)["final"]


def resolved(path: str) -> str:
    return str(Path(base.resolve_path(path)).resolve())


def load_frozen_system(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FrozenFSQSystem, dict, argparse.Namespace, str]:
    layer1_path = resolved(args.layer1_checkpoint)
    selected_layer2 = args.layer2_checkpoint or DEFAULT_LAYER2_CHECKPOINTS[int(args.layer2_level)]
    layer2_path = resolved(selected_layer2)
    layer1_payload = torch.load(layer1_path, map_location="cpu", weights_only=False)
    layer2_payload = torch.load(layer2_path, map_location="cpu", weights_only=False)
    if str(layer1_payload.get("stage", "")) != "layer1":
        raise ValueError(f"not a Layer1 checkpoint: {layer1_path}")
    if str(layer2_payload.get("stage", "")) != "layer2_fsq_direct":
        raise ValueError(f"not a direct Layer2 FSQ checkpoint: {layer2_path}")
    tokenizer_meta = dict(layer2_payload.get("tokenizer", {}))
    levels = [int(value) for value in tokenizer_meta.get("fsq_levels", [])]
    expected_levels = [int(args.layer2_level)] * 3
    if levels != expected_levels:
        raise ValueError(f"selected Layer2 levels are {levels}, expected {expected_levels}")
    if str(tokenizer_meta.get("arch")) != "cnn":
        raise ValueError(f"stage-ar currently requires the requested CNN Layer2 checkpoints, got {tokenizer_meta}")

    saved_args = argparse.Namespace(**copy.deepcopy(dict(layer2_payload["args"])))
    saved_args.cpu = bool(args.cpu)
    saved_args._usage_weight = 0.0

    # Do not call ``direct.build_direct_bundle`` here.  That helper delegates
    # to ``base.Layer3FSQTokenizer``, which belonged to the old canonical
    # Stage-3 entrypoint and is intentionally absent from the current BAR
    # binary-FSQ ``train-stage3-fsq.py``.  The checkpoint already contains
    # the complete direct E2/D2/FSQ/combiner graph, so restore that graph
    # directly and leave the current Stage-3 entrypoint untouched.
    e1, d1 = direct.build_layer1(args=saved_args, source_ckpt=layer1_payload, device=device)
    tokenizer = DirectFSQTokenizer(saved_args, device)
    base.jsccf_io.load_state(
        tokenizer.e3,
        layer2_payload["e2_state_dict"],
        "stage_ar_frozen_E2",
        strict=True,
    )
    base.jsccf_io.load_state(
        tokenizer.d3,
        layer2_payload["d2_state_dict"],
        "stage_ar_frozen_D2",
        strict=True,
    )
    tokenizer_state = dict(layer2_payload.get("tokenizer_state_dict", {}))
    quantizer_state = {
        key[len("quantizer.") :]: value
        for key, value in tokenizer_state.items()
        if key.startswith("quantizer.")
    }
    if not quantizer_state:
        raise KeyError("direct Layer2 checkpoint has no tokenizer.quantizer state")
    base.jsccf_io.load_state(
        tokenizer.quantizer,
        quantizer_state,
        "stage_ar_frozen_FSQ",
        strict=True,
    )
    inner = base.OutputsCombiner().to(device)
    combiner = direct.SafeBlendCombiner(
        inner,
        mode=str(saved_args.combiner_mode),
        init_alpha=float(saved_args.blend_init),
    ).to(device)
    base.jsccf_io.load_state(
        combiner,
        layer2_payload["combiner_state_dict"],
        "stage_ar_frozen_combiner",
        strict=True,
    )
    bundle = direct.DirectBundle(
        e1=e1,
        d1=d1,
        tokenizer=tokenizer,
        combiner=combiner,
        init_report={},
    )
    latent = dict(layer2_payload.get("latent", {}))
    index_shape = latent.get("idx2", [16, 16])
    if [int(value) for value in index_shape] != [16, 16]:
        raise ValueError(f"stage-ar expects the mentor-specified 16x16 index map, got {index_shape}")
    system = FrozenFSQSystem(bundle, levels, 16, 16).to(device)
    return system, layer2_payload, saved_args, layer2_path


class MeanMetrics:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weights: dict[str, int] = {}

    def add(self, name: str, value: torch.Tensor | float, weight: int) -> None:
        number = float(value.detach().item()) if torch.is_tensor(value) else float(value)
        self.sums[name] = self.sums.get(name, 0.0) + number * int(weight)
        self.weights[name] = self.weights.get(name, 0) + int(weight)

    def result(self) -> dict[str, float]:
        return {name: total / max(1, self.weights[name]) for name, total in self.sums.items()}


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(1).clamp_min(1e-12)
    return -10.0 * mse.log10()


def train_epoch(
    loader,
    system: FrozenFSQSystem,
    model: IFSQPrefixRasterAR,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    system.eval()
    metrics = MeanMetrics()
    for batch_index, (images, _labels) in enumerate(loader, start=1):
        if int(args.max_train_batches) > 0 and batch_index > int(args.max_train_batches):
            break
        images = images.to(device, non_blocking=True)
        target = system.oracle_targets(images)
        logits = model.forward_teacher(target["y1"], target["x1"], target["indices"])
        flat_target = target["indices"].flatten(1)
        loss_ce = F.cross_entropy(
            logits.reshape(-1, system.vocabulary),
            flat_target.reshape(-1),
            label_smoothing=float(args.label_smoothing),
        )
        loss_recon = logits.new_zeros(())
        soft_final = None
        if float(args.lambda_recon) > 0.0:
            soft_q = system.logits_to_soft_q(logits, float(args.soft_temperature))
            soft_final = system.decode_q(soft_q, target["y1"], target["x1"])
            loss_recon = F.mse_loss(soft_final.float(), images.float())
        loss = loss_ce + float(args.lambda_recon) * loss_recon
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
        optimizer.step()

        batch = int(images.shape[0])
        accuracy = (logits.argmax(dim=-1) == flat_target).float().mean()
        metrics.add("loss", loss, batch)
        metrics.add("loss_ce", loss_ce, batch)
        metrics.add("loss_recon", loss_recon, batch)
        metrics.add("teacher_token_accuracy", accuracy, batch)
        if soft_final is not None:
            metrics.add("teacher_soft_psnr", psnr_per_image(soft_final, images).mean(), batch)
    result = metrics.result()
    result["nll_bits_per_token"] = result.get("loss_ce", 0.0) / math.log(2.0)
    result["teacher_perplexity"] = math.exp(min(30.0, result.get("loss_ce", 0.0)))
    return result


@torch.no_grad()
def validate(
    loader,
    system: FrozenFSQSystem,
    model: IFSQPrefixRasterAR,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    system.eval()
    metrics = MeanMetrics()
    for batch_index, (images, _labels) in enumerate(loader, start=1):
        if int(args.max_val_batches) > 0 and batch_index > int(args.max_val_batches):
            break
        images = images.to(device, non_blocking=True)
        target = system.oracle_targets(images)
        teacher_logits = model.forward_teacher(target["y1"], target["x1"], target["indices"])
        flat_target = target["indices"].flatten(1)
        loss_ce = F.cross_entropy(teacher_logits.reshape(-1, system.vocabulary), flat_target.reshape(-1))
        rollout_logits, generated = model.generate(
            target["y1"],
            target["x1"],
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            sample_logits=bool(args.sample_logits),
        )
        generated_map = generated.view(int(images.shape[0]), system.height, system.width)
        q_hat = system.indices_to_q(generated_map)
        final = system.decode_q(q_hat, target["y1"], target["x1"])
        zero = system.decode_q(torch.zeros_like(q_hat), target["y1"], target["x1"])
        shuffled_q = system.tokenizer.shuffle_q3(q_hat)
        shuffled = system.decode_q(shuffled_q, target["y1"], target["x1"])

        batch = int(images.shape[0])
        x1_psnr = psnr_per_image(target["x1"], images)
        oracle_psnr = psnr_per_image(target["oracle"], images)
        final_psnr = psnr_per_image(final, images)
        zero_psnr = psnr_per_image(zero, images)
        shuffle_psnr = psnr_per_image(shuffled, images)
        metrics.add("loss_ce", loss_ce, batch)
        metrics.add("teacher_token_accuracy", (teacher_logits.argmax(dim=-1) == flat_target).float().mean(), batch)
        metrics.add("rollout_token_accuracy", (generated == flat_target).float().mean(), batch)
        metrics.add("psnr_x1", x1_psnr.mean(), batch)
        metrics.add("psnr_oracle", oracle_psnr.mean(), batch)
        metrics.add("psnr_x2_hat", final_psnr.mean(), batch)
        metrics.add("delta_x1_hat", (final_psnr - x1_psnr).mean(), batch)
        metrics.add("gap_oracle", (oracle_psnr - final_psnr).mean(), batch)
        metrics.add("psnr_zero", zero_psnr.mean(), batch)
        metrics.add("psnr_shuffle", shuffle_psnr.mean(), batch)
        metrics.add("drop_zero", (final_psnr - zero_psnr).mean(), batch)
        metrics.add("drop_shuffle", (final_psnr - shuffle_psnr).mean(), batch)
        metrics.add("rollout_logit_abs_mean", rollout_logits.abs().mean(), batch)
    result = metrics.result()
    result["nll_bits_per_token"] = result.get("loss_ce", 0.0) / math.log(2.0)
    result["teacher_perplexity"] = math.exp(min(30.0, result.get("loss_ce", 0.0)))
    result["receiver_only"] = 1.0
    return result


def artifact_stem(args: argparse.Namespace) -> str:
    version = base.jsccf_io.safe_artifact_name(args.version)
    mode_suffix = "" if str(args.condition_mode) == "prefix" else f"_{base.jsccf_io.safe_artifact_name(args.condition_mode)}"
    return (
        f"jscc_f_{version}_stage_ifsq_ar_fsq_"
        f"l{args.layer2_level}x{args.layer2_level}x{args.layer2_level}{mode_suffix}"
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    args: argparse.Namespace,
    model: IFSQPrefixRasterAR,
    optimizer: optim.Optimizer,
    metrics: dict[str, float],
    layer2_path: str,
    best_psnr: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(args.condition_mode) == "prefix":
        ar_framework = "iFSQ/LlamaGen decoder-only spatial prefix + 2D RoPE + KV cache"
        ar_inputs = ["y1", "x1", "generated_index_prefix"]
    else:
        ar_framework = "decoder-only per-token y1 fusion + 2D RoPE + KV cache"
        ar_inputs = ["y1", "generated_index_prefix"]
    torch.save(
        {
            "route": f"FSQ-iFSQ-{args.condition_mode}-AR-receiver",
            "stage": "stage_ifsq_prefix_ar_fsq",
            "ar_framework": ar_framework,
            "condition_mode": str(args.condition_mode),
            "epoch": int(epoch),
            "version": str(args.version),
            "args": vars(args),
            "metrics": metrics,
            "best_psnr": float(best_psnr),
            "layer1_checkpoint": resolved(args.layer1_checkpoint),
            "layer2_checkpoint": layer2_path,
            "fsq_levels": [int(args.layer2_level)] * 3,
            "index_shape": [16, 16],
            "vocabulary": int(args.layer2_level) ** 3,
            "ar_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "receiver_contract": {
                "deployment_inputs": ar_inputs,
                "training_only": ["image", "z2", "q2", "oracle_indices"],
                "output": "FSQ_idx2_hat_16x16",
            },
        },
        path,
    )
    print(f"saved checkpoint: {path}", flush=True)


def load_resume(
    path: str,
    model: IFSQPrefixRasterAR,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    layer2_path: str,
) -> tuple[int, float]:
    if not path:
        return 1, float("-inf")
    payload = torch.load(resolved(path), map_location="cpu", weights_only=False)
    if str(payload.get("stage", "")) != "stage_ifsq_prefix_ar_fsq":
        raise ValueError(
            "not an iFSQ-prefix AR checkpoint; the previous cross-attention AR format is intentionally unsupported"
        )
    if [int(value) for value in payload.get("fsq_levels", [])] != [int(args.layer2_level)] * 3:
        raise ValueError("resume FSQ levels do not match --layer2-level")
    if resolved(payload.get("layer1_checkpoint", "")) != resolved(args.layer1_checkpoint):
        raise ValueError("resume Layer1 checkpoint does not match")
    if resolved(payload.get("layer2_checkpoint", "")) != resolved(layer2_path):
        raise ValueError("resume Layer2 checkpoint does not match")
    saved_mode = str(payload.get("condition_mode", payload.get("args", {}).get("condition_mode", "prefix")))
    if saved_mode != str(args.condition_mode):
        raise ValueError(
            f"resume condition mode is {saved_mode!r}, but --condition-mode is {args.condition_mode!r}; "
            "prefix and per-token-y1 have different sequence layouts and cannot share a checkpoint"
        )
    model.load_state_dict(payload["ar_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    start = int(payload.get("epoch", 0)) + 1
    print(f"resumed {resolved(path)} at epoch {start}", flush=True)
    return start, float(payload.get("best_psnr", float("-inf")))


def print_header(
    args: argparse.Namespace,
    system: FrozenFSQSystem,
    layer2_payload: dict,
    layer2_path: str,
    model: IFSQPrefixRasterAR,
    train_size: int | None,
    val_size: int | None,
) -> None:
    print("=== JSCC-f | conditional spatial FSQ autoregression ===", flush=True)
    print("实验设计", flush=True)
    print(f"  Layer1={resolved(args.layer1_checkpoint)} (frozen CNN E1/D1); current no-channel route uses y1=z1", flush=True)
    print(f"  Layer2={layer2_path} (frozen E2/FSQ/D2/combiner)", flush=True)
    print(
        f"  oracle idx2=[B,16,16] -> raster sequence N=256; levels={system.levels.tolist()} "
        f"K={system.vocabulary}; condition_mode={args.condition_mode}",
        flush=True,
    )
    oracle_metrics = dict(layer2_payload.get("metrics", {}))
    print(
        f"  selected Layer2 checkpoint reference: psnr_x1={oracle_metrics.get('psnr_x1')} "
        f"psnr_oracle={oracle_metrics.get('psnr_final')} delta={oracle_metrics.get('delta_x1')}",
        flush=True,
    )
    print("loss设计", flush=True)
    print(
        f"  L=teacher-forced CE + {float(args.lambda_recon):g}*teacher-logit soft-FSQ MSE; "
        f"label_smoothing={float(args.label_smoothing):g}; validation is strict cached 256-step rollout",
        flush=True,
    )
    print("模块选择", flush=True)
    if str(args.condition_mode) == "prefix":
        condition_description = "condition=CNN(y1,x1)->256 spatial prefix tokens + learned BOS"
    else:
        condition_description = (
            "condition=y1[B,16,16,16]->256 aligned features; feature(i,j) fuses with "
            "BOS/q2(i-1) while predicting q2(i,j), x1 is not an AR input"
        )
    print(
        f"  {condition_description}; LlamaGen decoder={len(model.blocks)}x{model.hidden}, "
        "RMSNorm/SwiGLU/2-D RoPE/KV cache; cross-attention=disabled",
        flush=True,
    )
    print(
        f"  trainable=AR only; Layer1/Layer2 frozen; batch={args.batch_size}/{args.test_batch} "
        f"train/valid={train_size}/{val_size}; device={'cpu' if args.cpu else 'cuda:0'} "
        f"visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )


@torch.no_grad()
def smoke_shapes(
    args: argparse.Namespace,
    system: FrozenFSQSystem,
    model: IFSQPrefixRasterAR,
    device: torch.device,
) -> None:
    model.eval()
    images = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    target = system.oracle_targets(images)
    teacher_logits = model.forward_teacher(target["y1"], target["x1"], target["indices"])
    rollout_logits, generated = model.generate(
        target["y1"],
        target["x1"],
        steps=int(args.smoke_rollout_steps),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        sample_logits=bool(args.sample_logits),
    )
    recovered_q = system.indices_to_q(target["indices"])
    recovered_final = system.decode_q(recovered_q, target["y1"], target["x1"])
    q_error = float((recovered_q - target["q2"]).abs().max().item())
    print(
        f"[smoke ifsq-prefix-ar] y1={tuple(target['y1'].shape)} x1={tuple(target['x1'].shape)} "
        f"indices={tuple(target['indices'].shape)} teacher_logits={tuple(teacher_logits.shape)} "
        f"rollout_logits={tuple(rollout_logits.shape)} generated={tuple(generated.shape)} "
        f"q2={tuple(recovered_q.shape)} final={tuple(recovered_final.shape)} inverse_max_error={q_error:.3g}",
        flush=True,
    )
    expected_logits = (int(args.smoke_batch_size), 256, system.vocabulary)
    if tuple(teacher_logits.shape) != expected_logits:
        raise RuntimeError(f"teacher logits must be {expected_logits}, got {tuple(teacher_logits.shape)}")
    if q_error > 1e-6:
        raise RuntimeError(f"mixed-radix FSQ inverse mismatch: max error {q_error}")
    cached_teacher = model.forward_sequence(target["y1"], target["x1"], generated)
    cache_error = float((cached_teacher - rollout_logits).abs().max().item())
    print(f"[smoke ifsq-prefix-ar] cached_teacher_max_error={cache_error:.3g}", flush=True)
    if cache_error > 5e-4:
        raise RuntimeError(f"iFSQ cached rollout disagrees with teacher sequence: max error {cache_error}")


def build_loaders(saved_args: argparse.Namespace, args: argparse.Namespace):
    loader_args = copy.deepcopy(saved_args)
    loader_args.data_dir = str(args.data_dir)
    loader_args.batch_size = int(args.batch_size)
    loader_args.test_batch = int(args.test_batch)
    loader_args.num_workers = int(args.num_workers)
    loader_args.val_num_workers = int(args.val_num_workers)
    loader_args.cpu = bool(args.cpu)
    cfg = base.jsccf_io.build_config(loader_args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(cfg)
    return train_loader, val_loader, cfg.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--layer1-checkpoint", default=DEFAULT_LAYER1_CHECKPOINT)
    parser.add_argument(
        "--layer2-level",
        type=int,
        choices=[5, 9, 17],
        default=5,
        help="Select the matching [L,L,L] direct-FSQ Layer2 checkpoint.",
    )
    parser.add_argument(
        "--layer2-checkpoint",
        default="MY-V2/jscc-f/explore/checkpoints-direct/direct-cnn-d3-l5x5x5-group-compatible-blend-e100/jscc_f_direct-cnn-d3-l5x5x5-group-compatible-blend-e100_layer2_fsq_direct_cnn_d3_l5x5x5_group_compatible_blend_best.pth",
        help="Optional path override; its FSQ levels must still match --layer2-level.",
    )
    parser.add_argument("--version", default="ifsq-prefix-ar-k125")
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument("--save-dir", default="MY-V2/jscc-f/checkpoints-ar-ifsq")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--history-json", default="")

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--condition-blocks", type=int, default=4)
    parser.add_argument(
        "--condition-mode",
        choices=["prefix", "per-token-y1"],
        default="prefix",
        help=(
            "prefix: CNN(y1,x1) is prepended as 256 condition tokens; "
            "per-token-y1: y1[i,j] is fused with the BOS/previous-q2 input that predicts q2[i,j]."
        ),
    )
    parser.add_argument("--ff-mult", type=float, default=8.0 / 3.0, help="iFSQ/LlamaGen SwiGLU expansion multiplier")
    parser.add_argument("--dropout", type=float, default=0.1, help="shared token/residual/FFN dropout")

    parser.add_argument("--lambda-recon", type=float, default=0.0)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sample-logits", action="store_true", help="sample instead of greedy argmax; disabled for accuracy evaluation")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--latest-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--test-batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-shapes", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    parser.add_argument("--smoke-rollout-steps", type=int, default=4)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.hidden) <= 0 or int(args.hidden) % int(args.heads) != 0:
        raise ValueError("--hidden must be positive and divisible by --heads")
    if min(int(args.layers), int(args.condition_blocks)) <= 0 or float(args.ff_mult) <= 0:
        raise ValueError("--layers, --condition-blocks and --ff-mult must be positive")
    if not 0.0 <= float(args.dropout) < 1.0:
        raise ValueError("--dropout must be in [0,1)")
    if float(args.lambda_recon) < 0.0:
        raise ValueError("--lambda-recon must be non-negative")
    if min(float(args.soft_temperature), float(args.temperature)) <= 0.0:
        raise ValueError("temperatures must be positive")
    if int(args.top_k) < 0 or not 0.0 < float(args.top_p) <= 1.0:
        raise ValueError("--top-k must be non-negative and --top-p must be in (0,1]")
    if not 0.0 <= float(args.label_smoothing) < 1.0:
        raise ValueError("--label-smoothing must be in [0,1)")
    if min(int(args.epochs), int(args.val_every), int(args.latest_every)) <= 0:
        raise ValueError("--epochs/--val-every/--latest-every must be positive")
    if not 1 <= int(args.smoke_rollout_steps) <= 256:
        raise ValueError("--smoke-rollout-steps must be in [1,256]")


def main() -> None:
    args = parse_args()
    validate_args(args)
    base.seed_everything(int(args.seed))
    save_root = Path(base.resolve_path(args.save_dir))
    save_root.mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(save_root / f"{artifact_stem(args)}.log")
    if not bool(args.smoke_shapes):
        base.setup_log_file(args.log_file)

    device = torch.device("cpu" if args.cpu else "cuda:0" if torch.cuda.is_available() else "cpu")
    system, layer2_payload, saved_args, layer2_path = load_frozen_system(args, device)
    layer1_latent = dict(torch.load(resolved(args.layer1_checkpoint), map_location="cpu", weights_only=False).get("latent", {}))
    y1_shape = layer1_latent.get("z1", [16, 16, 16])
    model = IFSQPrefixRasterAR(
        int(y1_shape[0]),
        system.height,
        system.width,
        system.vocabulary,
        hidden=int(args.hidden),
        layers=int(args.layers),
        heads=int(args.heads),
        condition_blocks=int(args.condition_blocks),
        ff_mult=float(args.ff_mult),
        dropout=float(args.dropout),
        condition_mode=str(args.condition_mode),
    ).to(device)

    if bool(args.smoke_shapes):
        print_header(args, system, layer2_payload, layer2_path, model, None, None)
        smoke_shapes(args, system, model, device)
        return

    train_loader, val_loader, loader_device = build_loaders(saved_args, args)
    if loader_device != device:
        raise RuntimeError(f"loader/model device mismatch: {loader_device} vs {device}")
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    start_epoch, best_psnr = load_resume(args.resume, model, optimizer, args, layer2_path)
    print_header(
        args,
        system,
        layer2_payload,
        layer2_path,
        model,
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    if bool(args.eval_only):
        metrics = validate(val_loader, system, model, args, device)
        print(f"[ifsq-prefix-ar eval] {metrics}", flush=True)
        return
    if start_epoch > int(args.epochs):
        raise ValueError(f"resume starts at epoch {start_epoch}, beyond --epochs={args.epochs}")

    history: list[dict] = []
    for epoch in range(start_epoch, int(args.epochs) + 1):
        started = time.time()
        train_metrics = train_epoch(train_loader, system, model, optimizer, args, device)
        print(
            f"[ifsq-prefix-ar train {epoch:03d}/{int(args.epochs):03d}] {train_metrics} "
            f"time={time.time() - started:.1f}s",
            flush=True,
        )
        item: dict[str, object] = {"epoch": epoch, "train": train_metrics}
        val_metrics: dict[str, float] | None = None
        if epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            val_metrics = validate(val_loader, system, model, args, device)
            print(f"[ifsq-prefix-ar val {epoch:03d}] {val_metrics}", flush=True)
            item["val"] = val_metrics
            score = float(val_metrics["psnr_x2_hat"])
            if score > best_psnr:
                best_psnr = score
                save_checkpoint(
                    save_root / f"{artifact_stem(args)}_best.pth",
                    epoch=epoch,
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    metrics=val_metrics,
                    layer2_path=layer2_path,
                    best_psnr=best_psnr,
                )
        history.append(item)
        if epoch % int(args.latest_every) == 0 or epoch == int(args.epochs):
            save_checkpoint(
                save_root / f"{artifact_stem(args)}_latest.pth",
                epoch=epoch,
                args=args,
                model=model,
                optimizer=optimizer,
                metrics=val_metrics or train_metrics,
                layer2_path=layer2_path,
                best_psnr=best_psnr,
            )
        if args.history_json:
            history_path = Path(base.resolve_path(args.history_json))
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
