#!/usr/bin/env python3
"""Conditional BAR generation for a frozen binary-FSQ Layer-2 codec.

This script is deliberately separate from ``train-stage3-fsq.py``.  The
latter trains the Layer-2 *tokenizer*::

    image, x1 -> E2 -> binary FSQ [B,16,16,16] -> D2/combiner.

Here that tokenizer and Layer1 are frozen.  The trainable receiver generator
uses only ``(z1, x1)`` at deployment and follows BAR's two-part generation
contract::

    spatial Transformer (raster causal AR, 256 steps)
        -> one condition vector per current spatial token
        -> Masked Bit Modeling (four confidence-ranked unmasking rounds)
        -> 16 binary FSQ channels at that location.

It intentionally never creates a 65536-way softmax.  A spatial token is a
length-16 binary vector, so the only prediction head is ``16 x 2`` logits.
The input image and the true FSQ bits are used only to create supervision
during training; ``generate`` has no path from either one to its output.

Supported frozen checkpoints under ``checkpoints-fsq-c16`` are exactly:

* ``cnn-cnn``
* ``cnn-bar``
* ``swin-swin``

The implementation uses a cached causal spatial Transformer during rollout
and a non-causal token-local MBM Transformer for iterative bit unmasking.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

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


stage3 = load_module("jsccf_binary_fsq_stage3", THIS_DIR / "train-stage3-fsq.py")


DEFAULT_CHECKPOINTS = {
    ("cnn", "cnn"): (
        "MY-V2/jscc-f/checkpoints-fsq-c16/"
        "jscc_f_c-16_stage3_fsq_l1-cnn_l2-cnn_c16_binary_direct_group_compatible_best.pth"
    ),
    ("cnn", "bar"): (
        "MY-V2/jscc-f/checkpoints-fsq-c16/"
        "jscc_f_c-16_stage3_fsq_l1-cnn_l2-bar_c16_native-e1152l27h16m4304-"
        "d1024l24h16m4096_binary_direct_group_fresh_best.pth"
    ),
    ("swin", "swin"): (
        "MY-V2/jscc-f/checkpoints-fsq-c16/"
        "jscc_f_c-16_stage3_fsq_l1-swin_l2-swin_c16_binary_direct_group_compatible_best.pth"
    ),
}


def resolve(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = CDDM_ROOT / candidate
    return str(candidate.resolve())


def set_frozen(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(1)
    return -10.0 * mse.clamp_min(1e-12).log10()


class MeanMetrics:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weights: dict[str, int] = {}

    def add(self, name: str, value: torch.Tensor | float, weight: int) -> None:
        number = float(value.detach().item()) if torch.is_tensor(value) else float(value)
        self.sums[name] = self.sums.get(name, 0.0) + number * int(weight)
        self.weights[name] = self.weights.get(name, 0) + int(weight)

    def result(self) -> dict[str, float]:
        return {
            name: total / float(max(1, self.weights[name]))
            for name, total in self.sums.items()
        }


class FrozenBinaryFSQSystem(nn.Module):
    """Strictly restored Layer1 + direct binary-FSQ Layer2 checkpoint."""

    def __init__(
        self,
        e1: nn.Module,
        d1: nn.Module,
        codec: nn.Module,
        *,
        height: int,
        width: int,
        channels: int,
    ) -> None:
        super().__init__()
        self.e1 = e1
        self.d1 = d1
        self.codec = codec
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        set_frozen(self.e1)
        set_frozen(self.d1)
        set_frozen(self.codec)

    def train(self, mode: bool = True) -> FrozenBinaryFSQSystem:
        # The oracle must remain frozen even while the generator trains.
        super().train(False)
        return self

    @torch.no_grad()
    def oracle_targets(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        z1 = stage3.encode_tensor(self.e1, images)
        x1 = self.d1(z1).clamp(0.0, 1.0)
        encoded = self.codec.encode(images, x1)
        codes = encoded["codes"]
        expected = (int(images.shape[0]), self.channels, self.height, self.width)
        if tuple(codes.shape) != expected:
            raise RuntimeError(
                f"checkpoint binary FSQ codes must be {expected}, got {tuple(codes.shape)}"
            )
        oracle = self.codec.decode(encoded["q2_hard"], x1)["final"]
        return {
            "z1": z1.detach(),
            "x1": x1.detach(),
            "codes": codes.detach(),
            "q2": encoded["q2_hard"].detach(),
            "oracle": oracle.detach(),
        }

    def codes_to_q(self, codes: torch.Tensor) -> torch.Tensor:
        return self.codec.quantizer.codes_to_quantized(codes)

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(self.codes_to_q(codes), x1)["final"]


def validate_checkpoint(payload: dict, layer1_arch: str, layer2_arch: str) -> None:
    if not str(payload.get("stage", "")).endswith("binary_direct"):
        raise ValueError("--checkpoint is not a direct binary-FSQ Layer2 checkpoint")
    architecture = dict(payload.get("architecture", {}))
    saved_args = dict(payload.get("args", {}))
    saved_l1 = str(architecture.get("layer1", saved_args.get("layer1_arch", "")))
    saved_l2 = str(architecture.get("layer2", saved_args.get("layer2_arch", "")))
    if (saved_l1, saved_l2) != (str(layer1_arch), str(layer2_arch)):
        raise ValueError(
            "checkpoint architecture does not match the requested combination: "
            f"checkpoint={saved_l1}-{saved_l2}, requested={layer1_arch}-{layer2_arch}"
        )
    latent = dict(payload.get("latent", {}))
    q2_shape = [int(value) for value in latent.get("q2", [])]
    quantizer = dict(payload.get("quantizer", {}))
    if q2_shape != [16, 16, 16] or int(quantizer.get("channels", 0)) != 16:
        raise ValueError(
            "BAR generator requires the C=16, 16x16 binary-FSQ checkpoint contract; "
            f"got q2={q2_shape}, channels={quantizer.get('channels')}"
        )
    if [int(value) for value in quantizer.get("levels", [])] != [2] * 16:
        raise ValueError("BAR generator requires binary FSQ levels [2] x 16")


def load_frozen_system(
    args: argparse.Namespace, device: torch.device
) -> tuple[FrozenBinaryFSQSystem, dict, argparse.Namespace, str]:
    checkpoint_path = resolve(args.checkpoint or DEFAULT_CHECKPOINTS[(args.layer1_arch, args.layer2_arch)])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint must be a mapping, got {type(payload)!r}")
    validate_checkpoint(payload, args.layer1_arch, args.layer2_arch)
    saved_args = argparse.Namespace(**copy.deepcopy(dict(payload["args"])))
    saved_args.cpu = bool(args.cpu)
    e1, d1 = stage3.build_layer1_modules(saved_args, device)
    codec = stage3.Layer2FSQCodec(saved_args, device)
    stage3.jsccf_io.load_state(e1, payload["e1_state_dict"], "BAR-AR frozen E1", strict=True)
    stage3.jsccf_io.load_state(d1, payload["d1_state_dict"], "BAR-AR frozen D1", strict=True)
    stage3.jsccf_io.load_state(codec.e2, payload["e2_state_dict"], "BAR-AR frozen E2", strict=True)
    stage3.jsccf_io.load_state(codec.d2, payload["d2_state_dict"], "BAR-AR frozen D2", strict=True)
    stage3.jsccf_io.load_state(
        codec.quantizer,
        payload["quantizer_state_dict"],
        "BAR-AR frozen binary FSQ",
        strict=True,
    )
    stage3.jsccf_io.load_state(
        codec.combiner,
        payload["combiner_state_dict"],
        "BAR-AR frozen combiner",
        strict=True,
    )
    system = FrozenBinaryFSQSystem(
        e1,
        d1,
        codec,
        height=16,
        width=16,
        channels=16,
    ).to(device)
    return system, payload, saved_args, checkpoint_path


class ReceiverConditionEncoder(nn.Module):
    """Map receiver-visible ``(z1, x1)`` to one feature per FSQ location."""

    def __init__(self, z1_channels: int, hidden: int, blocks: int) -> None:
        super().__init__()
        stem_channels = max(64, int(hidden) // 2)
        self.z1_stem = nn.Sequential(
            nn.Conv2d(int(z1_channels), stem_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, 3, padding=1),
        )
        self.x1_stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, 3, padding=1),
        )
        layers: list[nn.Module] = [
            nn.Conv2d(2 * stem_channels, int(hidden), 1),
            nn.GELU(),
        ]
        for _ in range(int(blocks)):
            layers.extend(
                [
                    nn.GroupNorm(1, int(hidden)),
                    nn.GELU(),
                    nn.Conv2d(int(hidden), int(hidden), 3, padding=1),
                ]
            )
        self.fuse = nn.Sequential(*layers)

    def forward(self, z1: torch.Tensor, x1: torch.Tensor, height: int, width: int) -> torch.Tensor:
        z1 = F.interpolate(z1, size=(height, width), mode="bilinear", align_corners=False)
        x1 = F.interpolate(x1, size=(height, width), mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.z1_stem(z1), self.x1_stem(x1)], dim=1))


class CausalSelfAttention(nn.Module):
    """Causal attention with a small per-layer KV cache for rollout."""

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        if int(hidden) % int(heads) != 0:
            raise ValueError("--hidden must be divisible by --heads")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.head_dim = self.hidden // self.heads
        self.qkv = nn.Linear(self.hidden, 3 * self.hidden, bias=False)
        self.proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.dropout = float(dropout)

    def forward(
        self,
        value: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, length, _ = value.shape
        query, key, val = self.qkv(value).chunk(3, dim=-1)
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        val = val.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        if use_cache:
            if cache is not None:
                key = torch.cat([cache[0], key], dim=2)
                val = torch.cat([cache[1], val], dim=2)
            attended = F.scaled_dot_product_attention(
                query, key, val, dropout_p=0.0, is_causal=False
            )
            next_cache = (key, val)
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                val,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            next_cache = None
        attended = attended.transpose(1, 2).reshape(batch, length, self.hidden)
        return self.proj(attended), next_cache


class SpatialARBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        mlp_hidden = max(int(hidden), int(round(float(hidden) * float(mlp_ratio))))
        self.norm1 = nn.LayerNorm(int(hidden))
        self.attention = CausalSelfAttention(hidden, heads, dropout)
        self.norm2 = nn.LayerNorm(int(hidden))
        self.mlp = nn.Sequential(
            nn.Linear(int(hidden), mlp_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_hidden, int(hidden)),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        value: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual, next_cache = self.attention(self.norm1(value), cache, use_cache=use_cache)
        value = value + residual
        return value + self.mlp(self.norm2(value)), next_cache


class MBMBlock(nn.Module):
    """Non-causal Transformer block inside one spatial token's bit vector."""

    def __init__(self, hidden: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        mlp_hidden = max(int(hidden), int(round(float(hidden) * float(mlp_ratio))))
        self.norm1 = nn.LayerNorm(int(hidden))
        self.attention = nn.MultiheadAttention(
            int(hidden), int(heads), dropout=float(dropout), batch_first=True
        )
        self.norm2 = nn.LayerNorm(int(hidden))
        self.mlp = nn.Sequential(
            nn.Linear(int(hidden), mlp_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_hidden, int(hidden)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        attended = self.attention(self.norm1(value), self.norm1(value), self.norm1(value), need_weights=False)[0]
        value = value + attended
        return value + self.mlp(self.norm2(value))


class MaskBitModeling(nn.Module):
    """BAR-style iterative masked-bit modeling for a C=16 binary FSQ token."""

    mask_token_id = 2

    def __init__(
        self,
        channels: int,
        condition_width: int,
        hidden: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.token_embedding = nn.Embedding(3, int(hidden))
        self.bit_position = nn.Parameter(torch.empty(1, self.channels, int(hidden)))
        self.condition = nn.Linear(int(condition_width), int(hidden))
        self.mask_ratio = nn.Sequential(
            nn.Linear(1, int(hidden)), nn.SiLU(), nn.Linear(int(hidden), int(hidden))
        )
        self.blocks = nn.ModuleList(
            MBMBlock(hidden, heads, mlp_ratio, dropout) for _ in range(int(layers))
        )
        self.norm = nn.LayerNorm(int(hidden))
        self.head = nn.Linear(int(hidden), 2)
        nn.init.normal_(self.bit_position, std=0.02)

    def forward(
        self, ids: torch.Tensor, conditions: torch.Tensor, mask_ratio: torch.Tensor
    ) -> torch.Tensor:
        if ids.ndim != 2 or int(ids.shape[1]) != self.channels:
            raise ValueError(f"MBM ids must be [N,{self.channels}], got {tuple(ids.shape)}")
        if conditions.ndim != 2 or int(conditions.shape[0]) != int(ids.shape[0]):
            raise ValueError("MBM conditions must be [N,H] with the same N as ids")
        ratio = mask_ratio.reshape(-1, 1).to(dtype=conditions.dtype)
        if int(ratio.shape[0]) == 1:
            ratio = ratio.expand(int(ids.shape[0]), -1)
        if int(ratio.shape[0]) != int(ids.shape[0]):
            raise ValueError("mask_ratio must be scalar or one value per spatial token")
        value = self.token_embedding(ids.long()) + self.bit_position
        value = value + self.condition(conditions).unsqueeze(1)
        value = value + self.mask_ratio(ratio).unsqueeze(1)
        for block in self.blocks:
            value = block(value)
        return self.head(self.norm(value))

    def training_loss(
        self, target: torch.Tensor, conditions: torch.Tensor, mask_ratio: float | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Random-mask BAR objective; masked bits have weight 1, known bits 0.1."""
        if target.ndim != 2 or int(target.shape[1]) != self.channels:
            raise ValueError(f"MBM target must be [N,{self.channels}], got {tuple(target.shape)}")
        if int(target.min()) < 0 or int(target.max()) > 1:
            raise ValueError("binary FSQ target contains a value outside {0,1}")
        batch = int(target.shape[0])
        if mask_ratio is None:
            # Cosine draws emphasize difficult high-mask examples, as in MaskGIT/BAR.
            ratios = torch.cos(torch.rand(batch, device=target.device) * math.pi / 2.0)
        else:
            ratios = torch.full((batch,), float(mask_ratio), device=target.device)
        masks = torch.rand_like(target.float()) < ratios.unsqueeze(1)
        # Do not let a sample silently become an all-known identity task.
        empty = ~masks.any(dim=1)
        if bool(empty.any().item()):
            masks[empty, 0] = True
        ids = torch.where(masks, torch.full_like(target, self.mask_token_id), target)
        logits = self(ids, conditions, masks.float().mean(dim=1))
        losses = F.cross_entropy(logits.transpose(1, 2), target.long(), reduction="none")
        weights = masks.float() + (~masks).float() * 0.1
        loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
        return loss, {
            "masked_fraction": masks.float().mean(),
            "masked_bit_accuracy": (logits.argmax(dim=-1)[masks] == target[masks]).float().mean(),
            "all_bit_accuracy": (logits.argmax(dim=-1) == target).float().mean(),
        }

    @torch.no_grad()
    def sample(
        self,
        conditions: torch.Tensor,
        tokens_allocation: tuple[int, ...],
        *,
        temperature: float,
        sample_logits: bool,
    ) -> torch.Tensor:
        if sum(tokens_allocation) != self.channels or any(int(value) < 1 for value in tokens_allocation):
            raise ValueError(
                f"--tokens-allocation must contain positive values summing to C={self.channels}, "
                f"got {tokens_allocation}"
            )
        if float(temperature) <= 0.0:
            raise ValueError("--temperature must be positive")
        batch = int(conditions.shape[0])
        ids = torch.full(
            (batch, self.channels), self.mask_token_id, dtype=torch.long, device=conditions.device
        )
        for round_index, allocated in enumerate(tokens_allocation):
            active = ids.eq(self.mask_token_id)
            ratio = active.float().mean(dim=1)
            logits = self(ids, conditions, ratio) / float(temperature)
            probabilities = logits.softmax(dim=-1)
            if sample_logits:
                sampled = torch.multinomial(
                    probabilities.reshape(-1, 2), num_samples=1
                ).reshape(batch, self.channels)
            else:
                sampled = probabilities.argmax(dim=-1)
            confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            confidence = confidence.masked_fill(~active, float("-inf"))
            if round_index == len(tokens_allocation) - 1:
                keep = active
            else:
                keep_count = min(int(allocated), self.channels)
                selected = confidence.topk(keep_count, dim=1).indices
                keep = torch.zeros_like(active)
                keep.scatter_(1, selected, True)
                keep &= active
            ids = torch.where(keep, sampled, ids)
        if bool(ids.eq(self.mask_token_id).any().item()):
            raise RuntimeError("MBM sampler left a bit masked after its final round")
        return ids


class BARGenerator(nn.Module):
    """Raster causal outer generator plus token-local iterative MBM."""

    def __init__(
        self,
        *,
        z1_channels: int,
        height: int,
        width: int,
        channels: int,
        hidden: int,
        layers: int,
        heads: int,
        bit_embedding: int,
        condition_blocks: int,
        mbm_hidden: int,
        mbm_layers: int,
        mbm_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(hidden) % int(heads) != 0 or int(mbm_hidden) % int(mbm_heads) != 0:
            raise ValueError("outer and MBM hidden widths must be divisible by their head counts")
        self.height = int(height)
        self.width = int(width)
        self.tokens = self.height * self.width
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.condition_encoder = ReceiverConditionEncoder(z1_channels, hidden, condition_blocks)
        self.row_position = nn.Embedding(self.height, int(hidden))
        self.col_position = nn.Embedding(self.width, int(hidden))
        self.bos = nn.Parameter(torch.empty(1, 1, int(hidden)))
        self.bit_embeddings = nn.ModuleList(
            nn.Embedding(2, int(bit_embedding)) for _ in range(self.channels)
        )
        self.input_merge = nn.Sequential(
            nn.Linear(self.channels * int(bit_embedding), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(hidden)),
        )
        self.blocks = nn.ModuleList(
            SpatialARBlock(hidden, heads, mlp_ratio, dropout) for _ in range(int(layers))
        )
        self.norm = nn.LayerNorm(int(hidden))
        self.mbm = MaskBitModeling(
            channels,
            hidden,
            mbm_hidden,
            mbm_layers,
            mbm_heads,
            mlp_ratio,
            dropout,
        )
        nn.init.normal_(self.bos, std=0.02)
        for embedding in self.bit_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)

    def _positions(self, device: torch.device) -> torch.Tensor:
        rows = torch.arange(self.height, device=device).view(self.height, 1).expand(-1, self.width).reshape(-1)
        cols = torch.arange(self.width, device=device).view(1, self.width).expand(self.height, -1).reshape(-1)
        return self.row_position(rows) + self.col_position(cols)

    def _condition(self, z1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        feature = self.condition_encoder(z1, x1, self.height, self.width)
        return feature.flatten(2).transpose(1, 2) + self._positions(feature.device).unsqueeze(0)

    def _embed_bits(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.ndim != 3 or int(bits.shape[-1]) != self.channels:
            raise ValueError(f"spatial bits must be [B,T,{self.channels}], got {tuple(bits.shape)}")
        if bits.numel() and (int(bits.min()) < 0 or int(bits.max()) > 1):
            raise ValueError("spatial bit tokens contain a value outside {0,1}")
        pieces = [embedding(bits[..., index].long()) for index, embedding in enumerate(self.bit_embeddings)]
        return self.input_merge(torch.cat(pieces, dim=-1))

    def _outer_full(self, condition: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = target_bits.shape
        if (tokens, channels) != (self.tokens, self.channels):
            raise ValueError(
                f"target bits must be [B,{self.tokens},{self.channels}], got {tuple(target_bits.shape)}"
            )
        history = self._embed_bits(target_bits[:, :-1]) if self.tokens > 1 else condition[:, :0]
        history = torch.cat([self.bos.expand(batch, -1, -1), history], dim=1)
        value = condition + history
        for block in self.blocks:
            value, _ = block(value, use_cache=False)
        return self.norm(value)

    def teacher_conditions(self, z1: torch.Tensor, x1: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        bits = codes.permute(0, 2, 3, 1).reshape(int(codes.shape[0]), self.tokens, self.channels)
        return self._outer_full(self._condition(z1, x1), bits)

    def training_loss(
        self, z1: torch.Tensor, x1: torch.Tensor, codes: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = codes.permute(0, 2, 3, 1).reshape(-1, self.channels)
        conditions = self.teacher_conditions(z1, x1, codes).reshape(-1, self.hidden)
        return self.mbm.training_loss(target, conditions)

    @torch.no_grad()
    def teacher_bit_logits(self, z1: torch.Tensor, x1: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        target = codes.permute(0, 2, 3, 1).reshape(-1, self.channels)
        conditions = self.teacher_conditions(z1, x1, codes).reshape(-1, self.hidden)
        ids = torch.full_like(target, self.mbm.mask_token_id)
        return self.mbm(ids, conditions, torch.ones(target.shape[0], device=target.device))

    @torch.no_grad()
    def generate(
        self,
        z1: torch.Tensor,
        x1: torch.Tensor,
        tokens_allocation: tuple[int, ...],
        *,
        temperature: float,
        sample_logits: bool,
        return_conditions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Strict deployment route: only z1, x1, and generated spatial prefixes."""
        condition = self._condition(z1, x1)
        batch = int(condition.shape[0])
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.blocks)
        generated: list[torch.Tensor] = []
        generated_conditions: list[torch.Tensor] = []
        previous = self.bos.expand(batch, -1, -1)
        for step in range(self.tokens):
            value = condition[:, step : step + 1] + previous
            next_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = []
            for block, cache in zip(self.blocks, caches):
                value, cache = block(value, cache, use_cache=True)
                next_caches.append(cache)
            caches = next_caches
            current_condition = self.norm(value[:, 0])
            current = self.mbm.sample(
                current_condition,
                tokens_allocation,
                temperature=temperature,
                sample_logits=sample_logits,
            )
            generated.append(current)
            generated_conditions.append(current_condition)
            previous = self._embed_bits(current.unsqueeze(1))
        flat = torch.stack(generated, dim=1)
        codes = flat.reshape(batch, self.height, self.width, self.channels).permute(0, 3, 1, 2).contiguous()
        if return_conditions:
            return codes, torch.stack(generated_conditions, dim=1)
        return codes


def parse_allocation(value: str, channels: int) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in str(value).replace("x", ",").split(",") if part.strip())
    if not result or any(item < 1 for item in result) or sum(result) != int(channels):
        raise ValueError(
            f"--tokens-allocation must be positive integers summing to {channels}, got {value!r}"
        )
    return result


def build_model(args: argparse.Namespace, system: FrozenBinaryFSQSystem, z1_channels: int) -> BARGenerator:
    return BARGenerator(
        z1_channels=int(z1_channels),
        height=system.height,
        width=system.width,
        channels=system.channels,
        hidden=int(args.hidden),
        layers=int(args.layers),
        heads=int(args.heads),
        bit_embedding=int(args.bit_embedding),
        condition_blocks=int(args.condition_blocks),
        mbm_hidden=int(args.mbm_hidden),
        mbm_layers=int(args.mbm_layers),
        mbm_heads=int(args.mbm_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
    )


def train_epoch(
    loader,
    system: FrozenBinaryFSQSystem,
    generator: BARGenerator,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    generator.train()
    system.eval()
    metrics = MeanMetrics()
    for batch_index, (images, _labels) in enumerate(loader, start=1):
        if int(args.max_train_batches) > 0 and batch_index > int(args.max_train_batches):
            break
        images = images.to(device, non_blocking=True)
        target = system.oracle_targets(images)
        loss, details = generator.training_loss(target["z1"], target["x1"], target["codes"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip_norm) > 0.0:
            nn.utils.clip_grad_norm_(generator.parameters(), float(args.grad_clip_norm))
        optimizer.step()
        batch = int(images.shape[0])
        metrics.add("loss", loss, batch)
        for name, value in details.items():
            metrics.add(name, value, batch)
    result = metrics.result()
    result["masked_nll_bits"] = result.get("loss", 0.0) / math.log(2.0)
    return result


@torch.no_grad()
def validate(
    loader,
    system: FrozenBinaryFSQSystem,
    generator: BARGenerator,
    allocation: tuple[int, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    generator.eval()
    system.eval()
    metrics = MeanMetrics()
    for batch_index, (images, _labels) in enumerate(loader, start=1):
        if int(args.max_val_batches) > 0 and batch_index > int(args.max_val_batches):
            break
        images = images.to(device, non_blocking=True)
        target = system.oracle_targets(images)
        teacher_logits = generator.teacher_bit_logits(target["z1"], target["x1"], target["codes"])
        target_bits = target["codes"].permute(0, 2, 3, 1).reshape(-1, system.channels)
        teacher_ce = F.cross_entropy(teacher_logits.transpose(1, 2), target_bits.long())
        generated = generator.generate(
            target["z1"],
            target["x1"],
            allocation,
            temperature=float(args.temperature),
            sample_logits=bool(args.sample_logits),
        )
        final = system.decode_codes(generated, target["x1"])
        zero = system.decode_codes(torch.zeros_like(generated), target["x1"])
        shuffled = generated.permute(0, 2, 3, 1).reshape(-1, system.channels)
        shuffled = shuffled[torch.randperm(shuffled.shape[0], device=device)]
        shuffled = shuffled.reshape(int(images.shape[0]), system.height, system.width, system.channels).permute(0, 3, 1, 2)
        shuffled_final = system.decode_codes(shuffled, target["x1"])
        batch = int(images.shape[0])
        x1_psnr = psnr_per_image(target["x1"], images)
        oracle_psnr = psnr_per_image(target["oracle"], images)
        final_psnr = psnr_per_image(final, images)
        zero_psnr = psnr_per_image(zero, images)
        shuffled_psnr = psnr_per_image(shuffled_final, images)
        metrics.add("teacher_bit_ce", teacher_ce, batch)
        metrics.add("teacher_bit_accuracy", (teacher_logits.argmax(dim=-1) == target_bits).float().mean(), batch)
        metrics.add("rollout_bit_accuracy", (generated == target["codes"]).float().mean(), batch)
        metrics.add("psnr_x1", x1_psnr.mean(), batch)
        metrics.add("psnr_oracle", oracle_psnr.mean(), batch)
        metrics.add("psnr_x2_hat", final_psnr.mean(), batch)
        metrics.add("delta_x1_hat", (final_psnr - x1_psnr).mean(), batch)
        metrics.add("gap_oracle", (oracle_psnr - final_psnr).mean(), batch)
        metrics.add("psnr_code0", zero_psnr.mean(), batch)
        metrics.add("psnr_shuffle", shuffled_psnr.mean(), batch)
        metrics.add("drop_code0", (final_psnr - zero_psnr).mean(), batch)
        metrics.add("drop_shuffle", (final_psnr - shuffled_psnr).mean(), batch)
    result = metrics.result()
    result["teacher_nll_bits_per_bit"] = result.get("teacher_bit_ce", 0.0) / math.log(2.0)
    result["receiver_only"] = 1.0
    return result


def artifact_stem(args: argparse.Namespace) -> str:
    version = stage3.jsccf_io.safe_artifact_name(str(args.version))
    return f"jscc_f_{version}_bar_ar_l1-{args.layer1_arch}_l2-{args.layer2_arch}_c16"


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    args: argparse.Namespace,
    generator: BARGenerator,
    optimizer: optim.Optimizer,
    metrics: dict[str, float],
    source_checkpoint: str,
    best_psnr: float,
    allocation: tuple[int, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "bar_binary_fsq_receiver_generator",
            "route": "conditional-spatial-BAR-MBM-receiver",
            "epoch": int(epoch),
            "args": vars(args),
            "metrics": metrics,
            "best_psnr": float(best_psnr),
            "source_layer2_checkpoint": source_checkpoint,
            "generator_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "token_contract": {
                "spatial_shape": [16, 16],
                "spatial_steps": 256,
                "binary_channels_per_token": 16,
                "implicit_joint_vocab_size": 65536,
                "logit_shape_per_token": [16, 2],
                "tokens_allocation": list(allocation),
            },
            "generation_contract": {
                "outer": "raster-causal spatial autoregression with KV cache",
                "inner": "parallel masked-bit modeling with confidence-ranked iterative unmasking",
                "not_bitwise_autoregression": True,
            },
            "receiver_contract": {
                "deployment_inputs": ["z1", "x1", "generated_spatial_bit_prefix"],
                "training_only": ["image", "true_z2", "true_q2", "true_fsq_bits"],
                "output": "q2_hat [B,16,16,16] decoded by frozen D2/combiner",
            },
        },
        path,
    )
    print(f"saved checkpoint: {path}", flush=True)


def load_resume(
    path: str,
    generator: BARGenerator,
    optimizer: optim.Optimizer,
    source_checkpoint: str,
    allocation: tuple[int, ...],
) -> tuple[int, float]:
    if not path:
        return 1, float("-inf")
    payload = torch.load(resolve(path), map_location="cpu", weights_only=False)
    if str(payload.get("stage", "")) != "bar_binary_fsq_receiver_generator":
        raise ValueError("--resume is not a BAR binary-FSQ receiver generator checkpoint")
    if resolve(str(payload.get("source_layer2_checkpoint", ""))) != resolve(source_checkpoint):
        raise ValueError("resume checkpoint was trained against a different frozen Layer2 checkpoint")
    saved_allocation = tuple(int(value) for value in payload.get("token_contract", {}).get("tokens_allocation", []))
    if saved_allocation != allocation:
        raise ValueError("resume --tokens-allocation differs from the saved generator contract")
    generator.load_state_dict(payload["generator_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload.get("epoch", 0)) + 1, float(payload.get("best_psnr", float("-inf")))


def build_loaders(saved_args: argparse.Namespace, args: argparse.Namespace):
    loader_args = copy.deepcopy(saved_args)
    loader_args.data_dir = str(args.data_dir)
    loader_args.batch_size = int(args.batch_size)
    loader_args.test_batch = int(args.test_batch)
    loader_args.num_workers = int(args.num_workers)
    loader_args.val_num_workers = int(args.val_num_workers)
    loader_args.cpu = bool(args.cpu)
    config = stage3.jsccf_io.build_config(loader_args, encoder_in_chans=3)
    train_loader, val_loader = stage3.get_loader(config)
    return train_loader, val_loader, config.device


@torch.no_grad()
def smoke_shapes(
    args: argparse.Namespace,
    system: FrozenBinaryFSQSystem,
    generator: BARGenerator,
    allocation: tuple[int, ...],
    device: torch.device,
) -> None:
    generator.eval()
    system.eval()
    images = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    target = system.oracle_targets(images)
    teacher_logits = generator.teacher_bit_logits(target["z1"], target["x1"], target["codes"])
    generated, cached_conditions = generator.generate(
        target["z1"],
        target["x1"],
        allocation,
        temperature=1.0,
        sample_logits=False,
        return_conditions=True,
    )
    final = system.decode_codes(generated, target["x1"])
    roundtrip_final = system.decode_codes(target["codes"], target["x1"])
    inverse_error = float((roundtrip_final - target["oracle"]).abs().max().item())
    teacher_conditions = generator.teacher_conditions(target["z1"], target["x1"], generated)
    cache_error = float((cached_conditions - teacher_conditions).abs().max().item())
    expected_logits = (int(args.smoke_batch_size) * 256, 16, 2)
    expected_codes = (int(args.smoke_batch_size), 16, 16, 16)
    if tuple(teacher_logits.shape) != expected_logits:
        raise RuntimeError(f"teacher MBM logits must be {expected_logits}, got {tuple(teacher_logits.shape)}")
    if tuple(generated.shape) != expected_codes or tuple(final.shape) != tuple(images.shape):
        raise RuntimeError(
            f"rollout must produce codes={expected_codes}, image={tuple(images.shape)}; "
            f"got codes={tuple(generated.shape)}, image={tuple(final.shape)}"
        )
    if inverse_error != 0.0:
        raise RuntimeError(f"binary FSQ code inverse changed frozen decoder output: {inverse_error}")
    if cache_error > 1e-5:
        raise RuntimeError(
            "cached outer autoregression disagrees with the teacher-forced causal pass: "
            f"max error={cache_error}"
        )
    print(
        f"[smoke bar-ar] z1={tuple(target['z1'].shape)} x1={tuple(target['x1'].shape)} "
        f"codes={tuple(target['codes'].shape)} teacher_logits={tuple(teacher_logits.shape)} "
        f"generated={tuple(generated.shape)} final={tuple(final.shape)} "
        f"inverse_max_error={inverse_error:.3g} cache_max_error={cache_error:.3g}; "
        f"outer_steps=256 inner_rounds={len(allocation)}",
        flush=True,
    )


def print_header(
    args: argparse.Namespace,
    system: FrozenBinaryFSQSystem,
    payload: dict,
    source_checkpoint: str,
    generator: BARGenerator,
    allocation: tuple[int, ...],
    train_size: int | None,
    val_size: int | None,
) -> None:
    print("=== JSCC-f | conditional BAR spatial AR + Masked Bit Modeling ===", flush=True)
    print(
        f"  frozen codec={source_checkpoint}\n"
        f"  combination={args.layer1_arch}-{args.layer2_arch}; checkpoint epoch={payload.get('epoch')}\n"
        f"  FSQ: [B,16,16,16] binary bits; 256 outer raster steps; "
        f"inner MBM allocation={list(allocation)}; no 65536-way softmax\n"
        f"  deployment only: (z1,x1,generated prefixes); image/true z2/q2/bits are training-only\n"
        f"  generator: outer={len(generator.blocks)}x{generator.hidden}; "
        f"MBM={len(generator.mbm.blocks)}x{generator.mbm.token_embedding.embedding_dim}; "
        f"train/valid={train_size}/{val_size}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--layer1-arch", choices=["cnn", "swin"], default="cnn")
    parser.add_argument("--layer2-arch", choices=["cnn", "bar", "swin"], default="bar")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Frozen C=16 binary-FSQ checkpoint; empty selects the requested combination's best checkpoint.",
    )
    parser.add_argument("--version", default="bar-ar")
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument("--save-dir", default="MY-V2/jscc-f/checkpoints-bar-ar")
    parser.add_argument("--history-json", default="")
    parser.add_argument("--log-file", default="")

    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--bit-embedding", type=int, default=32)
    parser.add_argument("--condition-blocks", type=int, default=3)
    parser.add_argument("--mbm-hidden", type=int, default=256)
    parser.add_argument("--mbm-layers", type=int, default=4)
    parser.add_argument("--mbm-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tokens-allocation", default="4,4,4,4")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-logits", action="store_true")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--test-batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--latest-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-shapes", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    combination = (str(args.layer1_arch), str(args.layer2_arch))
    if combination not in DEFAULT_CHECKPOINTS:
        supported = ", ".join(f"{left}-{right}" for left, right in DEFAULT_CHECKPOINTS)
        raise ValueError(f"unsupported combination {combination}; supported combinations: {supported}")
    for value in (
        args.hidden,
        args.layers,
        args.heads,
        args.bit_embedding,
        args.condition_blocks,
        args.mbm_hidden,
        args.mbm_layers,
        args.mbm_heads,
        args.epochs,
        args.batch_size,
        args.test_batch,
        args.val_every,
        args.latest_every,
        args.smoke_batch_size,
    ):
        if int(value) < 1:
            raise ValueError("all model widths/counts, epochs, and batch sizes must be positive")
    if int(args.hidden) % int(args.heads) or int(args.mbm_hidden) % int(args.mbm_heads):
        raise ValueError("--hidden/--heads and --mbm-hidden/--mbm-heads must divide exactly")
    if float(args.mlp_ratio) <= 0.0 or not 0.0 <= float(args.dropout) < 1.0:
        raise ValueError("--mlp-ratio must be positive and --dropout must be in [0,1)")
    if min(float(args.lr), float(args.temperature)) <= 0.0 or float(args.weight_decay) < 0.0:
        raise ValueError("--lr/--temperature must be positive and --weight-decay non-negative")
    if min(int(args.num_workers), int(args.val_num_workers), int(args.max_train_batches), int(args.max_val_batches)) < 0:
        raise ValueError("worker and max-batch counts must be non-negative")
    return parse_allocation(args.tokens_allocation, 16)


def main() -> None:
    args = parse_args()
    allocation = validate_args(args)
    stage3.seed_everything(int(args.seed))
    device = torch.device("cpu" if bool(args.cpu) or not torch.cuda.is_available() else "cuda:0")
    system, payload, saved_args, source_checkpoint = load_frozen_system(args, device)
    z1_shape = [int(value) for value in dict(payload.get("latent", {})).get("z1", [16, 16, 16])]
    if len(z1_shape) != 3 or z1_shape[1:] != [16, 16]:
        raise ValueError(f"checkpoint z1 contract must be [C,16,16], got {z1_shape}")
    generator = build_model(args, system, z1_shape[0]).to(device)
    if bool(args.smoke_shapes):
        print_header(args, system, payload, source_checkpoint, generator, allocation, None, None)
        smoke_shapes(args, system, generator, allocation, device)
        return

    save_root = Path(resolve(args.save_dir))
    save_root.mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(save_root / f"{artifact_stem(args)}.log")
    stage3.setup_log_file(args.log_file)
    train_loader, val_loader, loader_device = build_loaders(saved_args, args)
    if loader_device != device:
        raise RuntimeError(f"loader/model device mismatch: {loader_device} vs {device}")
    optimizer = optim.AdamW(generator.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    start_epoch, best_psnr = load_resume(args.resume, generator, optimizer, source_checkpoint, allocation)
    print_header(
        args, system, payload, source_checkpoint, generator, allocation,
        len(train_loader.dataset), len(val_loader.dataset),
    )
    if bool(args.eval_only):
        print(f"[bar-ar eval] {validate(val_loader, system, generator, allocation, args, device)}", flush=True)
        return
    if start_epoch > int(args.epochs):
        raise ValueError(f"resume begins at epoch {start_epoch}, beyond --epochs={args.epochs}")

    history: list[dict[str, object]] = []
    for epoch in range(start_epoch, int(args.epochs) + 1):
        started = time.time()
        train_metrics = train_epoch(train_loader, system, generator, optimizer, args, device)
        print(f"[bar-ar train {epoch:03d}/{args.epochs:03d}] {train_metrics} time={time.time() - started:.1f}s", flush=True)
        record: dict[str, object] = {"epoch": epoch, "train": train_metrics}
        checkpoint_metrics: dict[str, float] = train_metrics
        if epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            val_metrics = validate(val_loader, system, generator, allocation, args, device)
            print(f"[bar-ar val {epoch:03d}] {val_metrics}", flush=True)
            record["val"] = val_metrics
            checkpoint_metrics = val_metrics
            score = float(val_metrics["psnr_x2_hat"])
            if score > best_psnr:
                best_psnr = score
                save_checkpoint(
                    save_root / f"{artifact_stem(args)}_best.pth", epoch=epoch, args=args,
                    generator=generator, optimizer=optimizer, metrics=val_metrics,
                    source_checkpoint=source_checkpoint, best_psnr=best_psnr, allocation=allocation,
                )
        history.append(record)
        if epoch % int(args.latest_every) == 0 or epoch == int(args.epochs):
            save_checkpoint(
                save_root / f"{artifact_stem(args)}_latest.pth", epoch=epoch, args=args,
                generator=generator, optimizer=optimizer, metrics=checkpoint_metrics,
                source_checkpoint=source_checkpoint, best_psnr=best_psnr, allocation=allocation,
            )
        if args.history_json:
            history_path = Path(resolve(args.history_json))
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
