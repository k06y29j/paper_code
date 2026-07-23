#!/usr/bin/env python3
"""Receiver-only joint predictor for the exact K=125 FSQ grid.

The deployment contract is deliberately narrow: :meth:`Joint125Predictor.forward`
accepts only the receiver-known Layer1 tensors ``z1`` and ``x1``.  Sender-only
quantities such as ``img``, ``z2``, ``q2`` and target indices are not accepted by
the public inference path.

For ``levels=[5, 5, 5]`` the canonical Layer2 FSQ quantizer uses mixed-radix
multipliers ``[25, 5, 1]``::

    index = code[0] * 25 + code[1] * 5 + code[2]

Each scalar code in ``[0, 4]`` is mapped to the exact FSQ value
``code / 4 * 2 - 1``.  Consequently, this module's buffers and index ordering
match ``IFSQQuantizer.codes_to_indices`` in ``train-stage3-fsq.py`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


LEVELS: tuple[int, int, int] = (5, 5, 5)
JOINT_MULTIPLIERS: tuple[int, int, int] = (25, 5, 1)
VOCAB_SIZE = 125


def _groups(channels: int, maximum: int = 32) -> int:
    """Return the largest valid GroupNorm group count up to ``maximum``."""

    for groups in range(min(int(maximum), int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class GatedResidualBlock(nn.Module):
    """State-dict-compatible copy of the existing receiver trunk block."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = int(channels) * int(expansion)
        self.norm = nn.GroupNorm(_groups(channels), int(channels))
        self.in_proj = nn.Conv2d(int(channels), hidden * 2, 3, padding=1)
        self.out_proj = nn.Conv2d(hidden, int(channels), 3, padding=1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected, gate = self.in_proj(F.silu(self.norm(value))).chunk(2, dim=1)
        return value + self.scale * self.out_proj(projected * torch.sigmoid(gate))


class SpatialAttentionBlock(nn.Module):
    """State-dict-compatible spatial attention used by DirectQPredictor."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if int(channels) % int(heads) != 0:
            raise ValueError(f"hidden={channels} must be divisible by heads={heads}")
        self.norm1 = nn.LayerNorm(int(channels))
        self.attn = nn.MultiheadAttention(
            int(channels), int(heads), batch_first=True
        )
        self.norm2 = nn.LayerNorm(int(channels))
        self.mlp = nn.Sequential(
            nn.Linear(int(channels), int(channels) * 2),
            nn.GELU(),
            nn.Linear(int(channels) * 2, int(channels)),
        )
        self.attn_scale = nn.Parameter(torch.tensor(0.1))
        self.mlp_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        tokens = value.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        attended, _weights = self.attn(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + self.attn_scale * attended
        tokens = tokens + self.mlp_scale * self.mlp(self.norm2(tokens))
        return (
            tokens.transpose(1, 2)
            .reshape(batch, channels, height, width)
            .contiguous()
        )


class ReceiverZ1X1Trunk(nn.Module):
    """Fuse ``z1`` and decoded ``x1`` at the ``z1`` spatial resolution.

    Module names and tensor shapes intentionally match
    ``explore-2/receiver_models.py::ReceiverTrunk`` with
    ``condition_mode='z1_x1'``.  A trained DirectQPredictor trunk can therefore
    be migrated without touching its old three-channel regression head.
    """

    def __init__(
        self,
        z1_channels: int,
        *,
        hidden: int = 128,
        blocks: int = 8,
        attention_every: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if int(z1_channels) < 1:
            raise ValueError(f"z1_channels must be positive, got {z1_channels}")
        if int(hidden) < 2:
            raise ValueError(f"hidden must be at least 2, got {hidden}")
        if int(blocks) < 0:
            raise ValueError(f"blocks must be non-negative, got {blocks}")
        self.z1_channels = int(z1_channels)
        self.hidden = int(hidden)
        self.z1_stem = nn.Sequential(
            nn.Conv2d(self.z1_channels, self.hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1),
        )
        self.x1_stem = nn.Sequential(
            nn.Conv2d(3, self.hidden // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden // 2, self.hidden, 3, padding=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(self.hidden * 4, self.hidden, 1),
            nn.SiLU(),
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1),
        )
        body: list[nn.Module] = []
        for index in range(int(blocks)):
            body.append(GatedResidualBlock(self.hidden))
            if int(attention_every) > 0 and (
                (index + 1) % int(attention_every) == 0
            ):
                body.append(SpatialAttentionBlock(self.hidden, int(heads)))
        self.body = nn.Sequential(*body)
        self.out_norm = nn.GroupNorm(_groups(self.hidden), self.hidden)

    def forward(self, z1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        _validate_receiver_inputs(z1, x1, self.z1_channels)
        x1_small = F.interpolate(
            x1,
            size=tuple(int(value) for value in z1.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        z_feature = self.z1_stem(z1)
        x_feature = self.x1_stem(x1_small)
        fused = torch.cat(
            [
                z_feature,
                x_feature,
                z_feature - x_feature,
                z_feature * x_feature,
            ],
            dim=1,
        )
        return F.silu(self.out_norm(self.body(self.fuse(fused))))


@dataclass(frozen=True)
class Joint125Prediction:
    """Outputs of the receiver-only joint categorical prediction.

    Attributes:
        logits: Joint-token logits with shape ``[B,125,H,W]``.
        q_soft: Posterior mean on the fixed FSQ grid, ``[B,3,H,W]``.
        q_hard: Argmax grid vector, ``[B,3,H,W]``.
        q_st: ``q_hard`` in the forward pass with ``q_soft`` gradients.
        indices: Argmax joint token in ``[0,125)``, shape ``[B,H,W]``.
        codes: The corresponding scalar codes in ``[0,5)``, ``[B,3,H,W]``.
        q_base: Continuous prediction from the migrated DirectQPredictor head.
    """

    logits: torch.Tensor
    q_soft: torch.Tensor
    q_hard: torch.Tensor
    q_st: torch.Tensor
    indices: torch.Tensor
    codes: torch.Tensor
    q_base: torch.Tensor


class Joint125Predictor(nn.Module):
    """Predict one of 125 complete FSQ tokens from receiver-known ``z1/x1``.

    The joint head captures dependencies among the three FSQ scalar channels.
    It is intentionally non-autoregressive at spatial sites; every site is
    predicted in parallel from the Layer1 receiver condition.

    Args:
        z1_channels: Number of channels in Layer1 latent ``z1``.
        hidden: Receiver trunk width.  Use 128 for the existing continuous
            DirectQPredictor checkpoint.
        blocks: Number of gated residual blocks; use 8 for that checkpoint.
        attention_every: Insert attention after every N residual blocks.
        heads: Attention heads; use 4 for the existing checkpoint.
        base_temperature: Temperature of the fixed-grid distance logits derived
            from the continuous DirectQPredictor output.
        base_head_trainable: Whether the migrated continuous three-channel head
            is trainable.  It can be frozen while the 125-way residual head is
            optimized.
    """

    levels = LEVELS
    vocab_size = VOCAB_SIZE

    def __init__(
        self,
        z1_channels: int,
        *,
        hidden: int = 128,
        blocks: int = 8,
        attention_every: int = 2,
        heads: int = 4,
        base_temperature: float = 0.1,
        base_head_trainable: bool = False,
    ) -> None:
        super().__init__()
        if float(base_temperature) <= 0.0:
            raise ValueError(
                f"base_temperature must be positive, got {base_temperature}"
            )
        self.z1_channels = int(z1_channels)
        self.hidden = int(hidden)
        self.base_temperature = float(base_temperature)
        self.trunk = ReceiverZ1X1Trunk(
            self.z1_channels,
            hidden=self.hidden,
            blocks=int(blocks),
            attention_every=int(attention_every),
            heads=int(heads),
        )
        # This is exactly the old DirectQPredictor head.  Once initialized from
        # its checkpoint it supplies the +0.22 dB continuous receiver starting
        # point instead of discarding that learned public-condition mapping.
        self.base_head = nn.Sequential(
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden, 3, 3, padding=1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.base_head[-2].weight)
        nn.init.zeros_(self.base_head[-2].bias)
        self.set_base_head_trainable(bool(base_head_trainable))
        # These are residual joint logits.  Their zero initialization makes the
        # initial hard output exactly the continuous base prediction snapped to
        # the K=125 grid (including explicit FSQ-compatible tie breaking).
        self.head = nn.Conv2d(self.hidden, VOCAB_SIZE, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        multipliers = torch.tensor(JOINT_MULTIPLIERS, dtype=torch.long)
        indices = torch.arange(VOCAB_SIZE, dtype=torch.long)
        codes = torch.stack(
            [
                (indices // int(multiplier)) % int(level)
                for level, multiplier in zip(LEVELS, JOINT_MULTIPLIERS)
            ],
            dim=1,
        )
        q_vectors = codes.float() / 4.0 * 2.0 - 1.0
        self.register_buffer("joint_multipliers", multipliers)
        self.register_buffer("codebook_codes", codes.contiguous())
        self.register_buffer("codebook_q", q_vectors.contiguous())
        self._assert_codebook_contract()

    def set_base_head_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze only the migrated continuous three-channel head."""

        for parameter in self.base_head.parameters():
            parameter.requires_grad_(bool(trainable))

    def _assert_codebook_contract(self) -> None:
        if tuple(self.codebook_codes.shape) != (VOCAB_SIZE, 3):
            raise RuntimeError(
                f"invalid K=125 codebook shape {tuple(self.codebook_codes.shape)}"
            )
        reconstructed = (
            self.codebook_codes * self.joint_multipliers.view(1, 3)
        ).sum(dim=1)
        expected = torch.arange(VOCAB_SIZE, device=reconstructed.device)
        if not torch.equal(reconstructed, expected):
            raise RuntimeError("mixed-radix [25,5,1] codebook ordering is invalid")

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert ``[B,3,H,W]`` scalar codes to canonical joint indices."""

        if codes.ndim != 4 or int(codes.shape[1]) != 3:
            raise ValueError(
                f"FSQ codes must be [B,3,H,W], got {tuple(codes.shape)}"
            )
        if codes.numel() == 0:
            raise ValueError("FSQ codes cannot be empty")
        if torch.is_floating_point(codes):
            if not torch.equal(codes, codes.round()):
                raise ValueError("floating FSQ codes must contain integer values")
        codes_long = codes.long()
        if int(codes_long.min()) < 0 or int(codes_long.max()) >= 5:
            raise ValueError("K=125 scalar FSQ codes must lie in [0,5)")
        multipliers = self.joint_multipliers.to(device=codes.device).view(
            1, 3, 1, 1
        )
        return (codes_long * multipliers).sum(dim=1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Vectorized inverse mixed-radix map to ``[B,3,H,W]`` codes."""

        if indices.ndim != 3:
            raise ValueError(
                f"joint FSQ indices must be [B,H,W], got {tuple(indices.shape)}"
            )
        if indices.numel() == 0:
            raise ValueError("joint FSQ indices cannot be empty")
        indices_long = indices.long()
        if int(indices_long.min()) < 0 or int(indices_long.max()) >= VOCAB_SIZE:
            raise ValueError(f"joint FSQ indices must lie in [0,{VOCAB_SIZE})")
        return (
            F.embedding(indices_long, self.codebook_codes)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def _base_logits(self, q_base: torch.Tensor) -> torch.Tensor:
        """Convert continuous ``q_base`` to differentiable fixed-grid logits."""

        q_float = q_base.float()
        codebook = self.codebook_q.float()
        q_squared = q_float.square().sum(dim=1, keepdim=True)
        code_squared = codebook.square().sum(dim=1).view(1, VOCAB_SIZE, 1, 1)
        cross = torch.einsum("bchw,kc->bkhw", q_float, codebook)
        squared_distance = (q_squared + code_squared - 2.0 * cross).clamp_min(0.0)
        logits = -squared_distance / self.base_temperature

        # torch.round is the contract used by IFSQQuantizer.  A tiny bias only
        # resolves exact equal-distance boundaries, making argmax(base_logits)
        # reproduce its round-to-even snap rather than argmax's lower-index tie.
        snapped_codes = ((q_float.clamp(-1.0, 1.0) + 1.0) * 2.0).round().long()
        snapped_indices = (
            snapped_codes
            * self.joint_multipliers.to(q_base.device).view(1, 3, 1, 1)
        ).sum(dim=1, keepdim=True)
        tie_bias = torch.zeros_like(logits).scatter_(1, snapped_indices, 1.0e-6)
        return logits + tie_bias

    def forward(self, z1: torch.Tensor, x1: torch.Tensor) -> Joint125Prediction:
        """Run receiver inference using only Layer1 ``z1`` and decoded ``x1``."""

        feature = self.trunk(z1, x1)
        q_base = self.base_head(feature)
        logits = self._base_logits(q_base).to(dtype=feature.dtype) + self.head(feature)
        if tuple(logits.shape[1:]) != (
            VOCAB_SIZE,
            int(z1.shape[-2]),
            int(z1.shape[-1]),
        ):
            raise RuntimeError(f"unexpected joint logits shape {tuple(logits.shape)}")

        # Softmax is evaluated in fp32 for stable posterior expectations under
        # AMP, then the expected q is returned in the model/logit dtype.
        probabilities = logits.float().softmax(dim=1)
        q_soft = torch.einsum(
            "bkhw,kc->bchw", probabilities, self.codebook_q.float()
        ).to(dtype=logits.dtype)
        indices = logits.argmax(dim=1)
        codes = self.indices_to_codes(indices)
        q_hard = (
            F.embedding(indices, self.codebook_q)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(dtype=logits.dtype)
        )
        q_st = q_soft + (q_hard - q_soft).detach()
        return Joint125Prediction(
            logits, q_soft, q_hard, q_st, indices, codes, q_base
        )


@dataclass(frozen=True)
class TrunkLoadReport:
    """Audit record returned when migrating a continuous predictor trunk."""

    source: str
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_trunk_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    ignored_non_trunk_keys: tuple[str, ...]


@dataclass(frozen=True)
class DirectQInitializationReport:
    """Audit record for complete DirectQ trunk plus base-head initialization."""

    source: str
    trunk: TrunkLoadReport
    loaded_base_head_keys: tuple[str, ...]
    missing_base_head_keys: tuple[str, ...]
    unexpected_head_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]


def _extract_predictor_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("predictor_state_dict", "state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    if payload and all(isinstance(key, str) for key in payload):
        return payload
    raise ValueError("checkpoint does not contain a predictor state dict")


def _strip_wrapper_prefixes(key: str) -> str:
    stripped = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "predictor."):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                changed = True
    return stripped


def load_direct_q_trunk(
    model: Joint125Predictor,
    checkpoint_or_state: str | PathLike[str] | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> TrunkLoadReport:
    """Load only a continuous DirectQPredictor's reusable ``trunk`` weights.

    ``checkpoint_or_state`` may be the full explore-2 receiver checkpoint, its
    ``predictor_state_dict``, or a wrapped state dict.  Keys from the old
    three-channel head (``head.0.*`` and ``head.2.*``) are intentionally
    ignored because this model replaces it with one 125-way categorical head.

    With ``strict=True`` (recommended), every trunk tensor must exist and match
    shape before any weights are changed.  ``strict=False`` permits a partial
    migration and reports every omission/mismatch for explicit audit.
    """

    if not isinstance(model, Joint125Predictor):
        raise TypeError(
            f"model must be Joint125Predictor, got {type(model).__name__}"
        )
    if isinstance(checkpoint_or_state, (str, PathLike)):
        source = str(checkpoint_or_state)
        payload = torch.load(source, map_location=map_location)
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint {source!r} is not a mapping")
    elif isinstance(checkpoint_or_state, Mapping):
        source = "<mapping>"
        payload = checkpoint_or_state
    else:
        raise TypeError(
            "checkpoint_or_state must be a path or mapping, got "
            f"{type(checkpoint_or_state).__name__}"
        )

    source_state = _extract_predictor_state(payload)
    target_state = model.trunk.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    mismatches: list[str] = []
    ignored: list[str] = []
    for original_key, value in source_state.items():
        key = _strip_wrapper_prefixes(str(original_key))
        if not key.startswith("trunk."):
            ignored.append(str(original_key))
            continue
        trunk_key = key[len("trunk.") :]
        if trunk_key not in target_state:
            unexpected.append(str(original_key))
            continue
        if not isinstance(value, torch.Tensor):
            mismatches.append(f"{original_key}:not-a-tensor")
            continue
        if tuple(value.shape) != tuple(target_state[trunk_key].shape):
            mismatches.append(
                f"{original_key}:{tuple(value.shape)}!={tuple(target_state[trunk_key].shape)}"
            )
            continue
        compatible[trunk_key] = value

    missing = sorted(set(target_state) - set(compatible))
    unexpected = sorted(unexpected)
    mismatches = sorted(mismatches)
    if strict and (missing or unexpected or mismatches):
        raise RuntimeError(
            "DirectQPredictor trunk is incompatible: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={mismatches}"
        )
    model.trunk.load_state_dict(compatible, strict=False)
    return TrunkLoadReport(
        source=source,
        loaded_keys=tuple(sorted(compatible)),
        missing_keys=tuple(missing),
        unexpected_trunk_keys=tuple(unexpected),
        shape_mismatches=tuple(mismatches),
        ignored_non_trunk_keys=tuple(sorted(ignored)),
    )


def initialize_from_direct_q_state(
    model: Joint125Predictor,
    checkpoint_or_state: str | PathLike[str] | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    base_head_trainable: bool | None = None,
) -> DirectQInitializationReport:
    """Initialize the complete continuous base predictor and joint residual.

    This loads both ``trunk.*`` and the old ``head.0.*``/``head.2.*`` tensors
    from an explore-2 DirectQPredictor.  The old head becomes ``base_head``;
    the new 125-way ``head`` remains zero initialized and therefore begins as a
    residual correction on top of distance logits from the learned ``q_base``.

    The first hard prediction after loading is the old continuous output
    snapped to the exact K=125 FSQ grid.  Set ``base_head_trainable=False`` to
    retain the old head as a frozen anchor, or ``True`` to co-adapt it.  A value
    of ``None`` preserves the constructor's trainability setting.
    """

    if not isinstance(model, Joint125Predictor):
        raise TypeError(
            f"model must be Joint125Predictor, got {type(model).__name__}"
        )
    if isinstance(checkpoint_or_state, (str, PathLike)):
        source = str(checkpoint_or_state)
        payload = torch.load(source, map_location=map_location)
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint {source!r} is not a mapping")
    elif isinstance(checkpoint_or_state, Mapping):
        source = "<mapping>"
        payload = checkpoint_or_state
    else:
        raise TypeError(
            "checkpoint_or_state must be a path or mapping, got "
            f"{type(checkpoint_or_state).__name__}"
        )

    # The trunk loader intentionally ignores all head keys.  Passing the
    # already loaded mapping avoids deserializing a full checkpoint twice.
    trunk_report_raw = load_direct_q_trunk(model, payload, strict=strict)
    trunk_report = TrunkLoadReport(
        source=source,
        loaded_keys=trunk_report_raw.loaded_keys,
        missing_keys=trunk_report_raw.missing_keys,
        unexpected_trunk_keys=trunk_report_raw.unexpected_trunk_keys,
        shape_mismatches=trunk_report_raw.shape_mismatches,
        ignored_non_trunk_keys=trunk_report_raw.ignored_non_trunk_keys,
    )

    source_state = _extract_predictor_state(payload)
    target_state = model.base_head.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    mismatches: list[str] = []
    for original_key, value in source_state.items():
        key = _strip_wrapper_prefixes(str(original_key))
        if not key.startswith("head."):
            continue
        head_key = key[len("head.") :]
        if head_key not in target_state:
            unexpected.append(str(original_key))
            continue
        if not isinstance(value, torch.Tensor):
            mismatches.append(f"{original_key}:not-a-tensor")
            continue
        if tuple(value.shape) != tuple(target_state[head_key].shape):
            mismatches.append(
                f"{original_key}:{tuple(value.shape)}!={tuple(target_state[head_key].shape)}"
            )
            continue
        compatible[head_key] = value

    missing = sorted(set(target_state) - set(compatible))
    unexpected = sorted(unexpected)
    mismatches = sorted(mismatches)
    if strict and (missing or unexpected or mismatches):
        raise RuntimeError(
            "DirectQPredictor base head is incompatible: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={mismatches}"
        )
    model.base_head.load_state_dict(compatible, strict=False)
    if base_head_trainable is not None:
        model.set_base_head_trainable(bool(base_head_trainable))
    return DirectQInitializationReport(
        source=source,
        trunk=trunk_report,
        loaded_base_head_keys=tuple(sorted(compatible)),
        missing_base_head_keys=tuple(missing),
        unexpected_head_keys=tuple(unexpected),
        shape_mismatches=tuple(mismatches),
    )


def _validate_receiver_inputs(
    z1: torch.Tensor, x1: torch.Tensor, expected_z1_channels: int
) -> None:
    if not isinstance(z1, torch.Tensor) or not isinstance(x1, torch.Tensor):
        raise TypeError("z1 and x1 must both be torch.Tensor instances")
    if z1.ndim != 4 or int(z1.shape[1]) != int(expected_z1_channels):
        raise ValueError(
            f"z1 must be [B,{expected_z1_channels},H,W], got {tuple(z1.shape)}"
        )
    if x1.ndim != 4 or int(x1.shape[1]) != 3:
        raise ValueError(f"x1 must be RGB [B,3,H,W], got {tuple(x1.shape)}")
    if int(z1.shape[0]) != int(x1.shape[0]):
        raise ValueError(
            f"z1/x1 batch mismatch: {int(z1.shape[0])} != {int(x1.shape[0])}"
        )
    if int(z1.shape[0]) < 1 or int(z1.shape[-2]) < 1 or int(z1.shape[-1]) < 1:
        raise ValueError(f"z1 must have non-empty batch/spatial axes, got {tuple(z1.shape)}")
    if int(x1.shape[-2]) < 1 or int(x1.shape[-1]) < 1:
        raise ValueError(f"x1 must have non-empty spatial axes, got {tuple(x1.shape)}")
    if z1.device != x1.device:
        raise ValueError(f"z1/x1 device mismatch: {z1.device} != {x1.device}")
    if z1.dtype != x1.dtype:
        raise ValueError(f"z1/x1 dtype mismatch: {z1.dtype} != {x1.dtype}")
    if not torch.is_floating_point(z1) or not torch.is_floating_point(x1):
        raise TypeError("z1 and x1 must use floating-point dtypes")


__all__ = [
    "DirectQInitializationReport",
    "JOINT_MULTIPLIERS",
    "LEVELS",
    "VOCAB_SIZE",
    "Joint125Prediction",
    "Joint125Predictor",
    "TrunkLoadReport",
    "initialize_from_direct_q_state",
    "load_direct_q_trunk",
]
