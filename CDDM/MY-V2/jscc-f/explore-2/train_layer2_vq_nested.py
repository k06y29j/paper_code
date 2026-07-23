#!/usr/bin/env python3
"""Direct Layer2 nested image-VQ/channel-VQ with receiver predictability.

Sender/oracle path::

    img -> frozen E1 -> z1 -> frozen D1 -> x1
    concat(img,x1) -> exact source E2_320 -> analysis(320->C)
    -> shared-prefix VQ -> synthesis(C->320) -> exact source D2_320
    -> residual combiner(x1,u2) -> x2

Receiver path::

    ReceiverCondition(z1,x1) -> direct q predictor -> nearest shared code
    -> q2_hat -> shared or independent synthesis/D2/combiner -> x2_hat

Only the first path receives ``img``.  It creates training labels and oracle
measurements.  The predictor forward signature accepts only ReceiverCondition.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
EXPLORE_DIR = JSCCF_DIR / "explore"
CDDM_ROOT = JSCCF_DIR.parents[1]
for path in (THIS_DIR, JSCCF_DIR, EXPLORE_DIR, CDDM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contracts import (  # noqa: E402
    assert_div2k_crop_protocol,
    assert_receiver_only_module,
    assert_training_targets_are_not_inputs,
    make_receiver_condition,
)
from receiver_models import ReceiverTrunk  # noqa: E402
from vq_modules import NestedPrefixVQ, build_nested_vq  # noqa: E402


DEFAULT_SOURCES = {
    "cnn": "MY-V2/jscc-f/checkpoints/jscc_f_cnn-stage2-no_compressor-gpu2_layer2_v2_no_compressor_combiner_best.pth",
    "swin": "MY-V2/jscc-f/checkpoints/jscc_f_swin320_layer2_swin_no_compressor_combiner_best.pth",
}


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


direct = load_module("jsccf_explore2_vq_direct_support", EXPLORE_DIR / "train_layer2_fsq_direct.py")
base = direct.base


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else CDDM_ROOT / value


def parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not parsed or any(item < 2 for item in parsed):
        raise ValueError(f"expected comma-separated K values >=2, got {value!r}")
    if parsed != sorted(set(parsed)):
        raise ValueError(f"K prefixes must be unique and increasing, got {parsed}")
    return parsed


def parse_float_list(value: str, count: int) -> list[float]:
    parsed = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(parsed) == 1:
        parsed *= int(count)
    if len(parsed) != int(count) or any(item < 0 for item in parsed):
        raise ValueError(f"expected one or {count} non-negative weights, got {parsed}")
    total = sum(parsed)
    if total <= 0:
        raise ValueError("at least one rate weight must be positive")
    return [item / total for item in parsed]


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass
class SourceLayer2:
    arch: str
    args: argparse.Namespace
    checkpoint: dict
    layer2_arch: str
    layer2_args: argparse.Namespace
    layer2_checkpoint: dict
    e1: nn.Module
    d1: nn.Module
    e2: nn.Module
    d2: nn.Module
    combiner: nn.Module

    @torch.no_grad()
    def layer1(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        z1 = base.encode_tensor(self.e1, imgs)
        x1_raw = self.d1(z1)
        return {"z1": z1, "x1_raw": x1_raw, "x1": x1_raw.clamp(0.0, 1.0)}


def make_source_e2_input(img: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Reproduce the Stage2 checkpoint's exact six-channel input contract.

    Canonical ``model.layer2_forward`` and ``train_stage2-cnn.py`` both use
    ``torch.cat([img, x1], dim=1)``.  Reversing the two RGB halves is shape
    compatible but changes the learned E2 semantics, so validate and centralize
    the ordering here instead of spelling an ambiguous concat at call sites.
    """

    if img.ndim != 4 or x1.ndim != 4:
        raise ValueError(f"source E2 expects BCHW img/x1, got {tuple(img.shape)} and {tuple(x1.shape)}")
    if tuple(img.shape) != tuple(x1.shape) or int(img.shape[1]) != 3:
        raise ValueError(
            f"source E2 expects matching RGB img/x1, got {tuple(img.shape)} and {tuple(x1.shape)}"
        )
    if img.device != x1.device:
        raise ValueError(f"source E2 img/x1 devices differ: {img.device} versus {x1.device}")
    return torch.cat([img, x1], dim=1)


class BuiltinResidualLayer2Encoder(nn.Module):
    """Architecture-neutral Layer2 encoder for img/x1 residual information."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(64, 96, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 160, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(160, 320, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(320, 320, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class BuiltinResidualLayer2Decoder(nn.Module):
    """Architecture-neutral q feature decoder producing a Layer2 RGB proposal."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(320, 320, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(320, 160, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(160, 96, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
        )
        # The additive combiner below interprets D2 output as a signed RGB
        # correction.  Zero initialization therefore gives the exact Layer1
        # reconstruction while still providing an immediate gradient to D2.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class BuiltinLayer2CombinerPlaceholder(nn.Module):
    def forward(self, x1: torch.Tensor, _u2: torch.Tensor) -> torch.Tensor:
        return x1


class AdditiveResidualCombiner(nn.Module):
    """Combine a signed Layer2 RGB correction with x1.

    ``uses_raw_u2`` is an explicit interface flag: unlike source JSCC
    combiners, this combiner must receive the signed D2 tensor before display
    clamping.  It remains a genuine ``(x1,u2)->x2`` module and has no sender
    image access.
    """

    uses_raw_u2 = True

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        return (x1 + u2).clamp(0.0, 1.0)


def _build_source_models(
    arch: str,
    checkpoint: dict,
    device: torch.device,
    *,
    cpu: bool,
    tag: str,
) -> tuple[argparse.Namespace, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    source_args = argparse.Namespace(**checkpoint["args"])
    source_args.cpu = bool(cpu)
    if str(arch) == "cnn":
        module = base.load_script_module(
            f"jsccf_explore2_vq_source_cnn_{tag}", "train_stage2-cnn.py"
        )
        module.validate_args(source_args)
        e1, d1, e2, d2, combiner = module.build_layer2_cnn(source_args, device)
    else:
        module = base.load_script_module(
            f"jsccf_explore2_vq_source_swin_{tag}", "train_stage2-swin.py"
        )
        e1, d1, e2, d2, combiner = module.build_stage2(source_args, device)
    return source_args, e1, d1, e2, d2, combiner


def load_source(args: argparse.Namespace, device: torch.device) -> SourceLayer2:
    layer1_path = args.source_checkpoint or DEFAULT_SOURCES[str(args.arch)]
    layer1_checkpoint = base.jsccf_io.load_checkpoint(str(resolve_path(layer1_path)))
    layer1_args, e1, d1, matched_e2, matched_d2, matched_combiner = _build_source_models(
        str(args.arch),
        layer1_checkpoint,
        device,
        cpu=bool(args.cpu),
        tag="layer1",
    )
    for name, model in (("E1", e1), ("D1", d1)):
        base.jsccf_io.load_state(
            model,
            layer1_checkpoint[f"{name.lower()}_state_dict"],
            f"vq_layer1_{name}",
            strict=True,
        )

    requested_layer2_arch = str(getattr(args, "layer2_arch", "match"))
    layer2_arch = str(args.arch) if requested_layer2_arch == "match" else requested_layer2_arch
    explicit_layer2_path = str(getattr(args, "layer2_source_checkpoint", "") or "")
    layer2_path = explicit_layer2_path or (
        "builtin:residual-cnn"
        if layer2_arch == "residual-cnn"
        else (str(layer1_path) if layer2_arch == str(args.arch) else DEFAULT_SOURCES[layer2_arch])
    )
    if layer2_arch == "residual-cnn":
        if explicit_layer2_path:
            raise ValueError("builtin residual-cnn Layer2 does not accept --layer2-source-checkpoint")
        layer2_args = argparse.Namespace(latent_h=16, latent_w=16, latent_ch=320)
        layer2_checkpoint = {
            "stage": "explore2_builtin_residual_cnn_layer2",
            "args": vars(layer2_args),
        }
        e2 = BuiltinResidualLayer2Encoder().to(device)
        d2 = BuiltinResidualLayer2Decoder().to(device)
        combiner = BuiltinLayer2CombinerPlaceholder().to(device)
    elif layer2_arch == str(args.arch) and resolve_path(layer2_path) == resolve_path(layer1_path):
        layer2_checkpoint = layer1_checkpoint
        layer2_args = layer1_args
        e2, d2, combiner = matched_e2, matched_d2, matched_combiner
    else:
        layer2_checkpoint = base.jsccf_io.load_checkpoint(str(resolve_path(layer2_path)))
        layer2_args, _unused_e1, _unused_d1, e2, d2, combiner = _build_source_models(
            layer2_arch,
            layer2_checkpoint,
            device,
            cpu=bool(args.cpu),
            tag="layer2",
        )
    if layer2_arch != "residual-cnn":
        for name, model in (("E2", e2), ("D2", d2), ("combiner", combiner)):
            base.jsccf_io.load_state(
                model,
                layer2_checkpoint[f"{name.lower()}_state_dict"],
                f"vq_layer2_{name}",
                strict=True,
            )
    e1.requires_grad_(False)
    d1.requires_grad_(False)
    e1.eval()
    d1.eval()
    args.source_checkpoint = str(layer1_path)
    args.layer2_source_checkpoint = str(layer2_path)
    args._layer2_arch = layer2_arch
    return SourceLayer2(
        str(args.arch),
        layer1_args,
        layer1_checkpoint,
        layer2_arch,
        layer2_args,
        layer2_checkpoint,
        e1,
        d1,
        e2,
        d2,
        combiner,
    )


class SynthesisAdapter(nn.Module):
    def __init__(self, channels: int, hidden: int = 320) -> None:
        super().__init__()
        self.linear = nn.Conv2d(int(channels), 320, 1)
        self.residual = nn.Sequential(
            nn.Conv2d(int(channels), int(hidden), 1),
            nn.SiLU(),
            nn.Conv2d(int(hidden), 320, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.linear(q) + self.residual(q)


def _identity_like_linear_(weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    """Initialize a rectangular projection as an identity on shared axes."""

    with torch.no_grad():
        weight.zero_()
        diagonal = min(int(weight.shape[0]), int(weight.shape[1]))
        rows = torch.arange(diagonal, device=weight.device)
        weight[rows, rows] = 1.0
        if bias is not None:
            bias.zero_()


class ImageEmbeddingEncoder(nn.Module):
    """Map one spatial image token from C coordinates to arbitrary D."""

    def __init__(self, channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(int(channels), int(embedding_dim), 1)
        _identity_like_linear_(
            self.projection.weight[:, :, 0, 0], self.projection.bias
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.projection(z)


class ImageEmbeddingDecoder(nn.Module):
    """Map a D-dimensional image-VQ token back to the Layer2 latent width."""

    def __init__(self, embedding_dim: int, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(int(embedding_dim), int(channels), 1)
        _identity_like_linear_(
            self.projection.weight[:, :, 0, 0], self.projection.bias
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.projection(q)


class ImageTokenRMSNorm(nn.Module):
    """Fix the RMS of every image-VQ token without adding a collapseable gain.

    The high-D Swin PCA path can otherwise reduce every token's magnitude
    before the weak early VQ term becomes active.  A fixed per-token RMS makes
    a code-vector collapse costly in direction as well as magnitude.  This is
    deliberately parameter-free: a learnable scale would recreate the same
    all-zero shortcut that this opt-in stabilizer is intended to remove.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        if self.eps <= 0.0:
            raise ValueError(f"ImageTokenRMSNorm eps must be positive, got {eps}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"image token RMS norm expects BCHW, got {tuple(z.shape)}")
        # Compute the statistic in fp32 so bfloat16/float16 runs keep the
        # intended unit RMS even when a token initially has a small norm.
        rms = z.float().square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return z / rms.to(dtype=z.dtype)


class ChannelEmbeddingEncoder(nn.Module):
    """Map each complete HxW channel token to an arbitrary D-dimensional vector."""

    def __init__(self, height: int, width: int, embedding_dim: int) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.embedding_dim = int(embedding_dim)
        self.projection = nn.Linear(self.height * self.width, self.embedding_dim)
        _identity_like_linear_(self.projection.weight, self.projection.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4 or tuple(int(v) for v in z.shape[2:]) != (self.height, self.width):
            raise ValueError(
                f"channel embedding encoder expects [B,C,{self.height},{self.width}], "
                f"got {tuple(z.shape)}"
            )
        batch, channels = int(z.shape[0]), int(z.shape[1])
        flat = z.reshape(batch, channels, self.height * self.width)
        return self.projection(flat).reshape(batch, channels, 1, self.embedding_dim)


class ChannelEmbeddingDecoder(nn.Module):
    """Map each D-dimensional channel-VQ token back to its HxW map."""

    def __init__(self, embedding_dim: int, height: int, width: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.height = int(height)
        self.width = int(width)
        self.projection = nn.Linear(self.embedding_dim, self.height * self.width)
        _identity_like_linear_(self.projection.weight, self.projection.bias)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim != 4 or tuple(int(v) for v in q.shape[2:]) != (1, self.embedding_dim):
            raise ValueError(
                f"channel embedding decoder expects [B,C,1,{self.embedding_dim}], "
                f"got {tuple(q.shape)}"
            )
        batch, channels = int(q.shape[0]), int(q.shape[1])
        flat = q.reshape(batch, channels, self.embedding_dim)
        return self.projection(flat).reshape(batch, channels, self.height, self.width)


class ProjectedSynthesis(nn.Module):
    """Embedding decoder followed by the existing q2->z320 synthesis adapter."""

    def __init__(self, token_decoder: nn.Module, synthesis: SynthesisAdapter) -> None:
        super().__init__()
        self.token_decoder = token_decoder
        self.base = synthesis

    @property
    def linear(self) -> nn.Conv2d:
        # Preserve the PCA initialization interface used by native-dimensional runs.
        return self.base.linear

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.base(self.token_decoder(q))


class ResidualCorrectionCombiner(nn.Module):
    """Exact x1 identity at initialization, without a global alpha ceiling."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 48, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(48, 3, 3, padding=1),
        )
        source_net = getattr(source, "net", None)
        if isinstance(source_net, nn.Sequential) and len(source_net) >= 3:
            self.net[0].load_state_dict(source_net[0].state_dict(), strict=True)
            self.net[1].load_state_dict(source_net[1].state_dict(), strict=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        correction = self.net(torch.cat([x1, u2], dim=1))
        return (x1 + correction).clamp(0.0, 1.0)


class CorrectionResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, int(channels))
        self.body = nn.Sequential(
            nn.Conv2d(
                int(channels),
                int(channels) * 2,
                3,
                padding=int(dilation),
                dilation=int(dilation),
            ),
            nn.SiLU(),
            nn.Conv2d(int(channels) * 2, int(channels), 3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.scale * self.body(self.norm(value))


class EnhancedResidualCorrectionCombiner(nn.Module):
    """Higher-capacity Layer2 combiner with an exact x1 identity start."""

    def __init__(self, source: nn.Module, width: int = 64, blocks: int = 8) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(6, int(width), 3, padding=1),
            nn.PReLU(),
        )
        source_net = getattr(source, "net", None)
        if isinstance(source_net, nn.Sequential) and len(source_net) >= 2:
            source_first = source_net[0]
            if (
                isinstance(source_first, nn.Conv2d)
                and tuple(source_first.weight.shape) == tuple(self.stem[0].weight.shape)
            ):
                self.stem[0].load_state_dict(source_first.state_dict(), strict=True)
        dilations = (1, 2, 4, 1)
        self.body = nn.Sequential(
            *[
                CorrectionResidualBlock(int(width), dilations[index % len(dilations)])
                for index in range(int(blocks))
            ]
        )
        self.head = nn.Conv2d(int(width), 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        correction = self.head(self.body(self.stem(torch.cat([x1, u2], dim=1))))
        return (x1 + correction).clamp(0.0, 1.0)


class NestedVQCodec(nn.Module):
    def __init__(self, source: SourceLayer2, args: argparse.Namespace, device: torch.device) -> None:
        super().__init__()
        self.e2 = source.e2
        self.d2 = source.d2
        self.base_height = int(source.layer2_args.latent_h)
        self.base_width = int(source.layer2_args.latent_w)
        self.embedding_dim = int(args._embedding_dim)
        self.bound_latent = bool(getattr(args, "bound_latent", False))
        self.image_vq_stabilizer = str(
            getattr(args, "image_vq_stabilizer", "none")
        )
        self.analysis = nn.Conv2d(320, int(args.latent_c), 1)
        base_synthesis = SynthesisAdapter(int(args.latent_c), int(args.adapter_hidden))
        if str(args.vq_family) == "image-vq":
            # In the normal/default path D=C remains the historical direct
            # adapter.  The opt-in RMS path intentionally inserts an exact
            # identity encoder/decoder pair, isolating the normalized VQ
            # coordinates from the source latent without changing D128 or any
            # existing checkpoint topology.
            use_projection_pair = (
                self.embedding_dim != int(args.latent_c)
                or self.image_vq_stabilizer != "none"
            )
            if not use_projection_pair:
                self.token_encoder = nn.Identity()
                self.synthesis = base_synthesis
            else:
                self.token_encoder = ImageEmbeddingEncoder(
                    int(args.latent_c), self.embedding_dim
                )
                self.synthesis = ProjectedSynthesis(
                    ImageEmbeddingDecoder(self.embedding_dim, int(args.latent_c)),
                    base_synthesis,
                )
            if self.image_vq_stabilizer == "none":
                self.token_stabilizer = nn.Identity()
            elif self.image_vq_stabilizer == "rmsnorm":
                self.token_stabilizer = ImageTokenRMSNorm(
                    float(getattr(args, "image_vq_rms_eps", 1e-6))
                )
            else:
                raise ValueError(
                    f"unknown image_vq_stabilizer {self.image_vq_stabilizer!r}"
                )
            quantizer_channels = self.embedding_dim
            quantizer_h = self.base_height
            quantizer_w = self.base_width
        else:
            self.token_stabilizer = nn.Identity()
            native_dim = self.base_height * self.base_width
            if self.embedding_dim == native_dim:
                self.token_encoder = nn.Identity()
                self.synthesis = base_synthesis
                # Preserve the historical channel-map state-dict layout
                # [K,H,W] when D is the native H*W.  Explicit non-native D
                # uses the decoupled [K,1,D] layout below.
                quantizer_h = self.base_height
                quantizer_w = self.base_width
            else:
                self.token_encoder = ChannelEmbeddingEncoder(
                    self.base_height, self.base_width, self.embedding_dim
                )
                self.synthesis = ProjectedSynthesis(
                    ChannelEmbeddingDecoder(
                        self.embedding_dim, self.base_height, self.base_width
                    ),
                    base_synthesis,
                )
                quantizer_h = 1
                quantizer_w = self.embedding_dim
            quantizer_channels = int(args.latent_c)
        self.quantizer = build_nested_vq(
            args.vq_family,
            max(args._rates),
            channels=quantizer_channels,
            h=quantizer_h,
            w=quantizer_w,
            beta=float(args.beta_commit),
            query_chunk_size=int(args.query_chunk_size),
            codebook_chunk_size=int(args.codebook_chunk_size),
            usage_decay=float(args.usage_decay),
            channel_codebook_mode=str(args.channel_codebook_mode),
        )
        self.to(device)

    def encode(self, img: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z320 = base.encode_tensor(self.e2, make_source_e2_input(img, x1))
        z = self.token_encoder(self.analysis(z320))
        if self.bound_latent:
            z = torch.tanh(z)
        z = self.token_stabilizer(z)
        return z, z320

    def decode(self, q: torch.Tensor, x1: torch.Tensor, combiner: nn.Module) -> dict[str, torch.Tensor]:
        z320_hat = self.synthesis(q)
        u2_raw = self.d2(z320_hat)
        u2_hat = u2_raw.clamp(0.0, 1.0)
        u2_for_combiner = u2_raw if bool(getattr(combiner, "uses_raw_u2", False)) else u2_hat
        final = combiner(x1, u2_for_combiner)
        return {
            "z320_hat": z320_hat,
            "u2_raw": u2_raw,
            "u2_hat": u2_hat,
            "final": final,
        }


class ReceiverDecodeStack(nn.Module):
    """Receiver-owned q2 decoder, isolated from the frozen sender oracle.

    The quantizer is deliberately not copied: sender and receiver use the same
    frozen codebook (including grouped channel ownership).  Only modules after
    q2 are receiver-adapted.
    """

    def __init__(self, synthesis: nn.Module, d2: nn.Module, combiner: nn.Module) -> None:
        super().__init__()
        self.synthesis = copy.deepcopy(synthesis)
        self.d2 = copy.deepcopy(d2)
        self.combiner = copy.deepcopy(combiner)

    @torch.no_grad()
    def initialize_from_sender(
        self,
        synthesis: nn.Module,
        d2: nn.Module,
        combiner: nn.Module,
    ) -> None:
        self.synthesis.load_state_dict(synthesis.state_dict(), strict=True)
        self.d2.load_state_dict(d2.state_dict(), strict=True)
        self.combiner.load_state_dict(combiner.state_dict(), strict=True)

    def decode(self, q: torch.Tensor, x1: torch.Tensor) -> dict[str, torch.Tensor]:
        z320_hat = self.synthesis(q)
        u2_raw = self.d2(z320_hat)
        u2_hat = u2_raw.clamp(0.0, 1.0)
        u2_for_combiner = (
            u2_raw if bool(getattr(self.combiner, "uses_raw_u2", False)) else u2_hat
        )
        final = self.combiner(x1, u2_for_combiner)
        return {
            "z320_hat": z320_hat,
            "u2_raw": u2_raw,
            "u2_hat": u2_hat,
            "final": final,
        }


class ReceiverVQPredictor(nn.Module):
    def __init__(
        self,
        z1_channels: int,
        out_channels: int,
        args: argparse.Namespace,
        *,
        height: int,
        width: int,
    ) -> None:
        super().__init__()
        self.trunk = ReceiverTrunk(
            int(z1_channels),
            hidden=int(args.predictor_hidden),
            blocks=int(args.predictor_blocks),
            attention_every=int(args.predictor_attention_every),
            heads=int(args.predictor_heads),
            condition_mode=str(args.condition_mode),
        )
        head_channels = (
            int(args._embedding_dim)
            if str(args.vq_family) == "image-vq"
            else int(out_channels)
        )
        self.head = nn.Sequential(
            nn.Conv2d(int(args.predictor_hidden), int(args.predictor_hidden), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(int(args.predictor_hidden), head_channels, 3, padding=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        native_channel_dim = int(height) * int(width)
        self.output_projection = (
            ChannelEmbeddingEncoder(int(height), int(width), int(args._embedding_dim))
            if str(args.vq_family) == "channel-vq"
            and int(args._embedding_dim) != native_channel_dim
            else nn.Identity()
        )

    def forward(self, condition):
        return self.output_projection(self.head(self.trunk(condition)))


@dataclass
class VQBundle:
    source: SourceLayer2
    codec: NestedVQCodec
    combiner: nn.Module
    predictor: ReceiverVQPredictor
    receiver_stack: ReceiverDecodeStack | None
    sender_state_hash: str = ""


def receiver_phase(args: argparse.Namespace) -> str:
    return str(getattr(args, "receiver_phase", "none"))


def independent_receiver(args: argparse.Namespace) -> bool:
    return str(getattr(args, "receiver_stack", "shared")) == "independent"


def image_vq_stabilizer_warmup_active(
    args: argparse.Namespace,
    *,
    epoch: int,
    train: bool,
) -> bool:
    """Whether the opt-in high-D image-VQ initialization is held fixed."""

    return (
        bool(train)
        and str(getattr(args, "vq_family", "")) == "image-vq"
        and str(getattr(args, "image_vq_stabilizer", "none")) != "none"
        and 1 <= int(epoch) <= int(getattr(args, "image_vq_stabilize_epochs", 0))
    )


def configure_image_vq_stabilizer_warmup(
    bundle: VQBundle,
    args: argparse.Namespace,
    *,
    epoch: int,
    train: bool,
) -> bool:
    """Temporarily freeze the pre-VQ distribution and codebook at startup.

    Parameters stay in the optimizer from construction time, so they resume
    normally after the selected number of epochs; this is unlike a permanent
    ``--freeze-*`` choice.  Decoder/D2/combiner remain trainable and learn to
    use the real K-means code distribution before encoder/codebook updates can
    co-adapt into a single code.
    """

    # Receiver-only phases freeze the whole sender permanently.  Never undo
    # that topology while merely checking whether a sender warmup would apply.
    if bool(args.receiver_only) or receiver_phase(args) != "none":
        return False
    active = image_vq_stabilizer_warmup_active(args, epoch=epoch, train=train)
    encoder_trainable = not bool(args.freeze_encoder) and not active
    for module in (bundle.codec.e2, bundle.codec.analysis):
        module.requires_grad_(encoder_trainable)
    # ``--freeze-encoder`` historically covered E2/analysis only.  The token
    # projection is held only by this opt-in warmup and must be restored even
    # when that legacy flag permanently freezes E2/analysis.
    bundle.codec.token_encoder.requires_grad_(not active)
    bundle.codec.quantizer.codebook.requires_grad_(
        not bool(args.freeze_codebook) and not active
    )
    return active


def decode_receiver(bundle: VQBundle, q: torch.Tensor, x1: torch.Tensor) -> dict[str, torch.Tensor]:
    if bundle.receiver_stack is not None:
        return bundle.receiver_stack.decode(q, x1)
    return bundle.codec.decode(q, x1, bundle.combiner)


def assert_receiver_topology(bundle: VQBundle, args: argparse.Namespace) -> None:
    if not independent_receiver(args):
        if bundle.receiver_stack is not None:
            raise AssertionError("shared receiver unexpectedly owns an independent decoder stack")
        return
    if bundle.receiver_stack is None:
        raise AssertionError("independent receiver decoder stack was not constructed")
    sender_ids = {
        id(parameter)
        for module in (bundle.codec.synthesis, bundle.codec.d2, bundle.combiner)
        for parameter in module.parameters()
    }
    receiver_ids = {id(parameter) for parameter in bundle.receiver_stack.parameters()}
    overlap = sender_ids & receiver_ids
    if overlap:
        raise AssertionError(f"independent receiver shares {len(overlap)} parameters with sender")
    sender_buffer_ids = {
        id(buffer)
        for module in (bundle.codec.synthesis, bundle.codec.d2, bundle.combiner)
        for buffer in module.buffers()
    }
    receiver_buffer_ids = {id(buffer) for buffer in bundle.receiver_stack.buffers()}
    buffer_overlap = sender_buffer_ids & receiver_buffer_ids
    if buffer_overlap:
        raise AssertionError(
            f"independent receiver shares {len(buffer_overlap)} buffers with sender"
        )
    if receiver_phase(args) != "none":
        sender_trainable = [
            name
            for prefix, module in (("codec", bundle.codec), ("combiner", bundle.combiner))
            for name, parameter in module.named_parameters(prefix=prefix)
            if parameter.requires_grad
        ]
        if sender_trainable:
            raise AssertionError(
                "receiver phases require a frozen sender oracle; trainable="
                f"{sender_trainable[:8]}"
            )
        if any(parameter.requires_grad for parameter in bundle.codec.quantizer.parameters()):
            raise AssertionError("receiver phases require the shared quantizer/codebook to be frozen")
        if (
            str(args.vq_family) == "channel-vq"
            and str(args.channel_codebook_mode) == "grouped"
            and bundle.codec.quantizer.channel_codebook_mode != "grouped"
        ):
            raise AssertionError("grouped receiver lost the channel-owned quantizer contract")


def module_state_sha256(*named_modules: tuple[str, nn.Module]) -> str:
    """Hash parameters and buffers without relying on serialization ordering."""

    digest = hashlib.sha256()
    for prefix, module in named_modules:
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{prefix}.{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sender_state_sha256(bundle: VQBundle) -> str:
    return module_state_sha256(("codec", bundle.codec), ("combiner", bundle.combiner))


def audit_sender_state(bundle: VQBundle, args: argparse.Namespace) -> float:
    if receiver_phase(args) == "none":
        return 1.0
    actual = sender_state_sha256(bundle)
    if not bundle.sender_state_hash:
        bundle.sender_state_hash = actual
    if actual != bundle.sender_state_hash:
        raise AssertionError(
            "frozen sender state changed during receiver phase: "
            f"expected={bundle.sender_state_hash} actual={actual}"
        )
    return 1.0


def build_bundle(args: argparse.Namespace, source: SourceLayer2, device: torch.device) -> VQBundle:
    codec = NestedVQCodec(source, args, device)
    if str(args.combiner) == "source":
        combiner = source.combiner
    elif str(args.combiner) == "additive":
        combiner = AdditiveResidualCombiner().to(device)
    elif str(args.combiner) == "enhanced":
        combiner = EnhancedResidualCorrectionCombiner(
            source.combiner,
            width=int(args.enhanced_combiner_width),
            blocks=int(args.enhanced_combiner_blocks),
        ).to(device)
    else:
        combiner = ResidualCorrectionCombiner(source.combiner).to(device)
    predictor = ReceiverVQPredictor(
        int(source.args.latent_ch),
        int(args.latent_c),
        args,
        height=int(source.layer2_args.latent_h),
        width=int(source.layer2_args.latent_w),
    ).to(device)
    receiver_stack = (
        ReceiverDecodeStack(codec.synthesis, codec.d2, combiner).to(device)
        if independent_receiver(args)
        else None
    )
    if bool(args.freeze_encoder):
        codec.e2.requires_grad_(False)
        codec.analysis.requires_grad_(False)
    if bool(args.freeze_codebook):
        codec.quantizer.codebook.requires_grad_(False)
    if bool(args.freeze_source_d2):
        codec.d2.requires_grad_(False)
    if bool(args.freeze_combiner):
        combiner.requires_grad_(False)
    if bool(args.receiver_only):
        codec.requires_grad_(False)
        combiner.requires_grad_(False)
    if bool(args.oracle_only):
        predictor.requires_grad_(False)
    phase = receiver_phase(args)
    if receiver_stack is not None:
        receiver_stack.requires_grad_(phase in {"decoder-warmup", "joint"})
    if phase == "q-pretrain":
        predictor.requires_grad_(True)
    elif phase == "decoder-warmup":
        predictor.requires_grad_(False)
    elif phase == "joint":
        predictor.requires_grad_(True)
    assert_receiver_only_module(predictor)
    bundle = VQBundle(source, codec, combiner, predictor, receiver_stack)
    assert_receiver_topology(bundle, args)
    return bundle


@torch.no_grad()
def initialize_pca(loader, bundle: VQBundle, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    if str(args.adapter_init) == "random":
        return {"pca_batches": 0.0, "pca_explained_ratio": 0.0}
    samples: list[torch.Tensor] = []
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if batch_index > int(args.pca_init_batches):
            break
        imgs = imgs.to(device, non_blocking=True)
        layer1 = bundle.source.layer1(imgs)
        z320 = base.encode_tensor(bundle.codec.e2, make_source_e2_input(imgs, layer1["x1"]))
        samples.append(z320.permute(0, 2, 3, 1).reshape(-1, 320).float().cpu())
    if not samples:
        raise RuntimeError("PCA initialization collected no E2 tokens")
    values = torch.cat(samples).double()
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.t().matmul(centered) / float(max(1, int(centered.shape[0]) - 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order[: int(args.latent_c)]]
    std = eigenvalues[: int(args.latent_c)].clamp_min(1e-10).sqrt()
    analysis_weight = (components / std.view(1, -1)).t()
    analysis_bias = -analysis_weight.matmul(mean)
    synthesis_weight = components * std.view(1, -1)
    bundle.codec.analysis.weight.copy_(
        analysis_weight.float().view(int(args.latent_c), 320, 1, 1).to(device)
    )
    bundle.codec.analysis.bias.copy_(analysis_bias.float().to(device))
    bundle.codec.synthesis.linear.weight.copy_(
        synthesis_weight.float().view(320, int(args.latent_c), 1, 1).to(device)
    )
    bundle.codec.synthesis.linear.bias.copy_(mean.float().to(device))
    explained = float(eigenvalues[: int(args.latent_c)].sum() / eigenvalues.clamp_min(0).sum())
    result = {
        "pca_batches": float(len(samples)),
        "pca_tokens": float(values.shape[0]),
        "pca_explained_ratio": explained,
    }
    print(f"[vq PCA init] {result}", flush=True)
    return result


@torch.no_grad()
def collect_codebook_samples(
    loader,
    bundle: VQBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    collected: list[torch.Tensor] = []
    count = 0
    maximum = int(args.codebook_init_max_samples)
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if batch_index > int(args.codebook_init_batches) or count >= maximum:
            break
        imgs = imgs.to(device, non_blocking=True)
        layer1 = bundle.source.layer1(imgs)
        z, _z320 = bundle.codec.encode(imgs, layer1["x1"])
        tokens = bundle.codec.quantizer.flatten_tokens(z).detach().float().cpu()
        remaining = maximum - count
        if int(tokens.shape[0]) > remaining:
            choice = torch.randperm(int(tokens.shape[0]))[:remaining]
            tokens = tokens[choice]
        collected.append(tokens)
        count += int(tokens.shape[0])
    if not collected:
        raise RuntimeError("codebook initialization collected no VQ tokens")
    return torch.cat(collected, dim=0)


@torch.no_grad()
def collect_channel_codebook_samples(
    loader,
    bundle: VQBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    """Collect channel-VQ sources without losing their channel identity.

    The returned tensor is ``[N,C,H,W]``.  ``codebook_init_max_samples`` keeps
    its existing meaning as a maximum number of channel-map tokens, but only
    whole images are retained so every collected round has all ``C`` channels.
    """

    quantizer = bundle.codec.quantizer
    if str(quantizer.family) != "channel-vq":
        raise ValueError("channel-balanced collection is only valid for channel-vq")
    channels = int(args.latent_c)
    maximum_tokens = int(args.codebook_init_max_samples)
    if maximum_tokens < channels:
        raise ValueError(
            "--codebook-init-max-samples must be at least --latent-c for channel-balanced init "
            f"({maximum_tokens} < {channels})"
        )
    maximum_images = maximum_tokens // channels
    collected: list[torch.Tensor] = []
    image_count = 0
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if batch_index > int(args.codebook_init_batches) or image_count >= maximum_images:
            break
        imgs = imgs.to(device, non_blocking=True)
        layer1 = bundle.source.layer1(imgs)
        z, _z320 = bundle.codec.encode(imgs, layer1["x1"])
        if int(z.shape[1]) != channels:
            raise RuntimeError(f"channel-balanced expected C={channels}, got z={tuple(z.shape)}")
        remaining = maximum_images - image_count
        z_cpu = z.detach().float().cpu()
        if int(z_cpu.shape[0]) > remaining:
            z_cpu = z_cpu[:remaining]
        collected.append(z_cpu)
        image_count += int(z_cpu.shape[0])
    if not collected:
        raise RuntimeError("channel-balanced initialization collected no BCHW latent samples")
    return torch.cat(collected, dim=0)


@torch.no_grad()
def initialize_channel_balanced_codebook(
    samples: torch.Tensor,
    bundle: VQBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | str]:
    """Initialize a round-major channel-map codebook from real latent maps.

    Row ``round*C + channel`` is sampled from that channel.  Consequently a
    prefix of ``C``, ``4C`` or ``16C`` contains one, four or sixteen maps from
    every channel.  Arbitrary K values use all complete rounds followed by a
    deterministic channel-order truncation of the next round.
    """

    quantizer = bundle.codec.quantizer
    if str(args.vq_family) != "channel-vq" or str(quantizer.family) != "channel-vq":
        raise ValueError("--codebook-init channel-balanced is only valid with --vq-family channel-vq")
    if samples.ndim != 4:
        raise ValueError(f"channel-balanced samples must be [N,C,H,W], got {tuple(samples.shape)}")
    sample_images, channels, height, width = (int(value) for value in samples.shape)
    if channels != int(args.latent_c):
        raise ValueError(f"channel-balanced sample C={channels} != --latent-c {int(args.latent_c)}")
    if (height, width) != tuple(int(value) for value in quantizer.embedding_shape):
        raise ValueError(
            f"channel-balanced map shape {(height, width)} != codebook {quantizer.embedding_shape}"
        )
    k_max = max(args._rates)
    rounds = int(math.ceil(float(k_max) / float(channels)))
    per_channel_indices: list[torch.Tensor] = []
    for channel in range(channels):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(args.seed) + 104729 * int(channel))
        choices: list[torch.Tensor] = []
        remaining = rounds
        while remaining > 0:
            # A complete permutation is consumed before any sample index is
            # reused for this channel.  Repeats occur only when rounds>N.
            permutation = torch.randperm(sample_images, generator=generator)
            take = min(remaining, sample_images)
            choices.append(permutation[:take])
            remaining -= take
        per_channel_indices.append(torch.cat(choices, dim=0))

    rows: list[torch.Tensor] = []
    for round_index in range(rounds):
        for channel in range(channels):
            if len(rows) >= k_max:
                break
            image_index = int(per_channel_indices[channel][round_index])
            rows.append(samples[image_index, channel])
    centers = torch.stack(rows, dim=0)
    if tuple(centers.shape) != (k_max, height, width):
        raise RuntimeError(
            f"channel-balanced built centers {tuple(centers.shape)}, expected {(k_max, height, width)}"
        )
    quantizer.codebook[:k_max].copy_(
        centers.to(device=device, dtype=quantizer.codebook.dtype)
    )
    quantizer.reset_usage()

    result: dict[str, float | str] = {
        "codebook_init_method": "channel-balanced",
        "codebook_init_selection": "seeded-channel-rounds",
        "channel_codebook_mode": str(args.channel_codebook_mode),
        "codebook_init_source_images": float(sample_images),
        "codebook_init_samples": float(sample_images * channels),
        "codebook_init_selected_codes": float(k_max),
        "codebook_vector_dim": float(height * width),
        "codebook_kmax": float(k_max),
        "codebook_init_rounds": float(rounds),
        "codebook_init_unique_rounds_per_channel": float(min(rounds, sample_images)),
        "codebook_init_reuse_rounds_per_channel": float(max(0, rounds - sample_images)),
    }
    for rate in args._rates:
        result[f"codebook_full_rounds_k{int(rate)}"] = float(int(rate) // channels)
        result[f"codebook_partial_channels_k{int(rate)}"] = float(int(rate) % channels)
    return result


@torch.no_grad()
def initialize_codebook(
    loader,
    bundle: VQBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | str]:
    method = str(args.codebook_init)
    if method == "channel-balanced":
        samples = collect_channel_codebook_samples(loader, bundle, args, device)
        result = initialize_channel_balanced_codebook(samples, bundle, args, device)
        print(f"[vq codebook init] method={method} {result}", flush=True)
        return result

    samples = collect_codebook_samples(loader, bundle, args, device)
    kwargs: dict[str, object] = {"seed": int(args.seed)}
    if method == "kmeans":
        kwargs.update(
            iterations=int(args.kmeans_iters),
            max_samples=int(args.codebook_init_max_samples),
        )
    bundle.codec.quantizer.initialize_from_samples(
        samples.to(device),
        k=max(args._rates),
        method=method,
        **kwargs,
    )
    result = {
        "codebook_init_method": method,
        "codebook_init_samples": float(samples.shape[0]),
        "codebook_vector_dim": float(samples.shape[1]),
        "codebook_kmax": float(max(args._rates)),
    }
    print(f"[vq codebook init] method={method} {result}", flush=True)
    return result


class MetricSums:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def add(self, name: str, value: float, weight: float) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value) * float(weight)
        self.weights[name] = self.weights.get(name, 0.0) + float(weight)

    def means(self) -> dict[str, float]:
        return {name: total / max(1.0, self.weights[name]) for name, total in self.sums.items()}


def psnr_values(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    return -10.0 * mse.clamp_min(1e-12).log10()


def index_change_rate(indices: torch.Tensor) -> torch.Tensor:
    """Fraction of neighbouring token indices that change within each sample.

    Image-VQ has a spatial ``[B,H,W]`` index map, so both horizontal and
    vertical edges are counted.  Channel-VQ has ``[B,C]`` indices, where the
    corresponding collapse diagnostic is change between adjacent channels.
    """

    changes: list[torch.Tensor] = []
    if indices.ndim == 3:
        if int(indices.shape[1]) > 1:
            changes.append((indices[:, 1:, :] != indices[:, :-1, :]).float().flatten(1))
        if int(indices.shape[2]) > 1:
            changes.append((indices[:, :, 1:] != indices[:, :, :-1]).float().flatten(1))
    elif indices.ndim == 2:
        if int(indices.shape[1]) > 1:
            changes.append((indices[:, 1:] != indices[:, :-1]).float().flatten(1))
    else:
        raise ValueError(f"VQ indices must be [B,H,W] or [B,C], got {tuple(indices.shape)}")
    if not changes:
        return torch.zeros((), device=indices.device, dtype=torch.float32)
    return torch.cat(changes, dim=1).mean()


def vq_weight(args: argparse.Namespace, epoch: int) -> float:
    if int(epoch) <= int(args.vq_warmup_epochs):
        return 0.0
    progress = (int(epoch) - int(args.vq_warmup_epochs)) / float(max(1, int(args.vq_ramp_epochs)))
    return float(args.lambda_vq) * min(1.0, max(0.0, progress))


def receiver_weight(args: argparse.Namespace, epoch: int) -> float:
    if bool(args.oracle_only):
        return 0.0
    if receiver_phase(args) != "none":
        return 1.0
    if int(epoch) <= int(args.receiver_warmup_epochs):
        return 0.0
    return 1.0


def sample_soft_usage_tokens(
    quantizer: NestedPrefixVQ,
    z: torch.Tensor,
    *,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a bounded, differentiable subset of pre-VQ tokens.

    ``positions`` always indexes the family-specific flattened-token layout.
    It is only needed by grouped channel-VQ, where the candidate code subset
    depends on the owning channel.  Sampling changes neither the hard VQ path
    nor its EMA usage accounting; it is exclusively for this training loss.
    """

    if int(max_tokens) < 0:
        raise ValueError(f"soft usage max_tokens must be non-negative, got {max_tokens}")
    tokens = quantizer.flatten_tokens(z)
    total = int(tokens.shape[0])
    positions = torch.arange(total, device=tokens.device, dtype=torch.long)
    if int(max_tokens) > 0 and total > int(max_tokens):
        positions = torch.randperm(total, device=tokens.device)[: int(max_tokens)]
        tokens = tokens.index_select(0, positions)
    return tokens, positions


def soft_batch_code_usage_entropy(
    quantizer: NestedPrefixVQ,
    tokens: torch.Tensor,
    token_positions: torch.Tensor,
    rate: int,
    *,
    temperature: float,
    query_chunk_size: int,
) -> dict[str, torch.Tensor]:
    """Differentiable batch-marginal code-usage entropy for one nested prefix.

    For image-VQ and global channel-VQ, every token sees the full ``K`` prefix
    and the loss is ``1 - H(mean_token_soft_assignment) / log(K)``.  Scores are
    evaluated in token chunks, so a K=4096 run is bounded to at most
    ``min(sampled_tokens, query_chunk_size) x K`` temporary scores.  For
    grouped channel-VQ, each channel sees only its round-major local
    candidates; entropy is computed per channel and then
    averaged, which avoids rewarding the fixed channel ownership itself.

    No value is detached: gradients reach the selected z2 tokens and the
    active codebook prefix.  Callers deliberately invoke this only during
    sender training, never for validation/deployment.
    """

    if tokens.ndim != 2 or int(tokens.shape[0]) < 1:
        raise ValueError(f"soft usage tokens must be non-empty [N,D], got {tuple(tokens.shape)}")
    if token_positions.ndim != 1 or int(token_positions.shape[0]) != int(tokens.shape[0]):
        raise ValueError(
            "soft usage token positions must be [N] matching tokens, got "
            f"tokens={tuple(tokens.shape)} positions={tuple(token_positions.shape)}"
        )
    if float(temperature) <= 0.0:
        raise ValueError(f"soft usage temperature must be positive, got {temperature}")
    if int(query_chunk_size) < 1:
        raise ValueError(f"soft usage query_chunk_size must be positive, got {query_chunk_size}")

    checked_rate = quantizer._checked_k(int(rate))
    prefix = quantizer.codebook_at_k(checked_rate)
    if int(tokens.shape[1]) != int(prefix[0].numel()):
        raise ValueError(
            "soft usage token/codebook dimensions differ: "
            f"{int(tokens.shape[1])} versus {int(prefix[0].numel())}"
        )
    # Float32 keeps the softmax and entropy stable for high-D image vectors,
    # while the casts remain differentiable with respect to the original
    # encoder output and codebook parameter.
    tokens_f = tokens.float()
    chunk = min(int(query_chunk_size), int(tokens_f.shape[0]))
    eps = torch.finfo(tokens_f.dtype).tiny

    if quantizer.channel_codebook_mode != "grouped":
        codebook = prefix.reshape(checked_rate, -1).float()
        code_norm = codebook.square().sum(dim=1)
        masses: list[torch.Tensor] = []
        for start in range(0, int(tokens_f.shape[0]), chunk):
            query = tokens_f[start : start + chunk]
            distance = (
                query.square().sum(dim=1, keepdim=True)
                + code_norm.unsqueeze(0)
                - 2.0 * query.matmul(codebook.t())
            ).clamp_min(0.0)
            probabilities = F.softmax(-distance / float(temperature), dim=1)
            masses.append(probabilities.sum(dim=0))
        marginal = torch.stack(masses, dim=0).sum(dim=0) / float(tokens_f.shape[0])
        entropy = -(marginal * marginal.clamp_min(eps).log()).sum()
        max_entropy = entropy.new_tensor(math.log(float(checked_rate)))
        ratio = entropy / max_entropy
        loss = 1.0 - ratio
        candidate_count = entropy.new_tensor(float(checked_rate))
    else:
        # ``flatten_tokens`` orders channel tokens as [batch, channel], hence
        # ``position % C`` is the owner of a selected token.  Codebook rows are
        # round-major, so reshape [round, channel, D] before gathering each
        # token's allowed local candidates.
        channels = int(quantizer.channels)
        rounds = int(quantizer.local_code_count(checked_rate))
        zero = tokens_f.sum() * 0.0
        if rounds <= 1:
            # K=C has no source-dependent index choice.  It should contribute
            # no artificial penalty (or gradient) to a capacity regularizer.
            entropy = zero
            ratio = zero + 1.0
            loss = zero
            candidate_count = zero + 1.0
        else:
            grouped_codebook = (
                prefix.reshape(rounds, channels, -1).permute(1, 0, 2).contiguous().float()
            )
            local_masses: list[torch.Tensor] = []
            owners = token_positions.remainder(channels)
            for start in range(0, int(tokens_f.shape[0]), chunk):
                query = tokens_f[start : start + chunk]
                owner = owners[start : start + chunk]
                candidates = grouped_codebook.index_select(0, owner)
                distance = (
                    query.square().sum(dim=1, keepdim=True)
                    + candidates.square().sum(dim=2)
                    - 2.0 * torch.einsum("nd,nrd->nr", query, candidates)
                ).clamp_min(0.0)
                probabilities = F.softmax(-distance / float(temperature), dim=1)
                local_masses.append(
                    probabilities.new_zeros((channels, rounds)).index_add(0, owner, probabilities)
                )
            masses_by_channel = torch.stack(local_masses, dim=0).sum(dim=0)
            totals = masses_by_channel.sum(dim=1)
            valid = totals > 0.0
            marginal = masses_by_channel[valid] / totals[valid].unsqueeze(1).clamp_min(eps)
            entropy_per_channel = -(marginal * marginal.clamp_min(eps).log()).sum(dim=1)
            entropy = entropy_per_channel.mean()
            max_entropy = entropy.new_tensor(math.log(float(rounds)))
            ratio = entropy / max_entropy
            loss = 1.0 - ratio
            candidate_count = entropy.new_tensor(float(rounds))

    return {
        "loss": loss,
        "entropy": entropy,
        "entropy_ratio": ratio,
        "sampled_tokens": entropy.new_tensor(float(tokens_f.shape[0])),
        "candidate_count": candidate_count,
    }


def soft_batch_code_usage_regularizer(
    quantizer: NestedPrefixVQ,
    z: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    """Aggregate the soft batch-usage objective across all nested K values."""

    tokens, positions = sample_soft_usage_tokens(
        quantizer,
        z,
        max_tokens=int(args.soft_usage_sample_tokens),
    )
    weighted_loss: list[torch.Tensor] = []
    weighted_entropy: list[torch.Tensor] = []
    weighted_ratio: list[torch.Tensor] = []
    result: dict[str, torch.Tensor] = {}
    for weight, rate in zip(args._rate_weights, args._rates):
        current = soft_batch_code_usage_entropy(
            quantizer,
            tokens,
            positions,
            int(rate),
            temperature=float(args.soft_usage_temperature),
            query_chunk_size=int(args.query_chunk_size),
        )
        weighted_loss.append(float(weight) * current["loss"])
        weighted_entropy.append(float(weight) * current["entropy"])
        weighted_ratio.append(float(weight) * current["entropy_ratio"])
        result[f"soft_usage_loss_k{int(rate)}"] = current["loss"]
        result[f"soft_usage_entropy_k{int(rate)}"] = current["entropy"]
        result[f"soft_usage_entropy_ratio_k{int(rate)}"] = current["entropy_ratio"]
        result[f"soft_usage_candidate_count_k{int(rate)}"] = current["candidate_count"]
    result.update(
        {
            "loss_soft_usage_entropy": torch.stack(weighted_loss).sum(),
            "soft_usage_entropy": torch.stack(weighted_entropy).sum(),
            "soft_usage_entropy_ratio": torch.stack(weighted_ratio).sum(),
            # The same sampled token set is intentionally shared by every K.
            "soft_usage_sampled_tokens": tokens.new_tensor(float(tokens.shape[0])),
            "soft_usage_enabled": tokens.new_tensor(1.0),
        }
    )
    return result


def forward_bundle(
    bundle: VQBundle,
    imgs: torch.Tensor,
    args: argparse.Namespace,
    *,
    train: bool,
    epoch: int = 0,
) -> dict:
    with torch.no_grad():
        layer1 = bundle.source.layer1(imgs)
    z, z320 = bundle.codec.encode(imgs, layer1["x1"])
    branches: dict[int, dict] = {}
    for rate in args._rates:
        q_st, q_hard, indices, stats = bundle.codec.quantizer.forward_at_k(
            z,
            int(rate),
            # Nested-prefix liveness must be aggregated across every operating
            # point.  Updating from Kmax alone can mark a row dead even when it
            # remains essential to a smaller prefix.
            update_usage=False,
        )
        continuous_warmup = int(getattr(args, "continuous_recon_warmup_epochs", 0))
        transition_epochs = int(getattr(args, "continuous_recon_transition_epochs", 0))
        if train and int(epoch) <= continuous_warmup:
            hard_fraction = 0.0
        elif train and transition_epochs > 0 and int(epoch) <= continuous_warmup + transition_epochs:
            hard_fraction = (int(epoch) - continuous_warmup) / float(transition_epochs)
        else:
            hard_fraction = 1.0
        decode_token = z + float(hard_fraction) * (q_st - z)
        decoded = bundle.codec.decode(decode_token, layer1["x1"], bundle.combiner)
        branches[int(rate)] = {
            "q_st": q_st,
            "q_hard": q_hard,
            "indices": indices,
            "stats": stats,
            **decoded,
        }
    if train and not bool(args.receiver_only) and bool(bundle.codec.quantizer.track_usage):
        # Concatenating along the batch axis preserves the family-specific
        # index contract ([B,H,W] for image-VQ, [B,C] for channel-VQ) while one
        # EMA update protects a code used by *any* configured prefix.
        combined_indices = torch.cat(
            [branches[int(rate)]["indices"].detach() for rate in args._rates],
            dim=0,
        )
        bundle.codec.quantizer.update_usage_ema(combined_indices, max(args._rates))
    condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
    q_raw = bundle.predictor(condition)
    q_pred_hard, pred_indices = bundle.codec.quantizer.quantize_input(
        q_raw,
        max(args._rates),
        detach_codebook=True,
    )
    q_pred_st = q_raw + (q_pred_hard - q_raw).detach()
    pred_decoded = decode_receiver(bundle, q_pred_st, layer1["x1"])
    result = {
        "layer1": layer1,
        "z": z,
        "z320": z320,
        "branches": branches,
        "condition": condition,
        "q_raw": q_raw,
        "q_pred_hard": q_pred_hard,
        "pred_indices": pred_indices,
        "pred": pred_decoded,
    }

    phase = receiver_phase(args)
    anchors_enabled = phase in {"decoder-warmup", "joint"} and any(
        float(value) > 0
        for value in (
            args.lambda_zero_anchor,
            args.lambda_shuffle_anchor,
            args.lambda_condition_anchor,
        )
    )
    if train and anchors_enabled:
        # Negative branches train only the receiver decoder stack.  Detaching
        # their q values prevents the anti-collapse anchors from teaching the
        # predictor to emit a constant code.
        q_anchor = q_pred_hard.detach()
        result["anchor_zero"] = decode_receiver(
            bundle,
            torch.zeros_like(q_anchor),
            layer1["x1"],
        )["final"]
        result["anchor_shuffle"] = decode_receiver(
            bundle,
            bundle.codec.quantizer.shuffle_tokens(q_anchor),
            layer1["x1"],
        )["final"]
        if int(imgs.shape[0]) > 1:
            permutation = torch.roll(torch.arange(int(imgs.shape[0]), device=imgs.device), shifts=1)
            wrong_condition = make_receiver_condition(
                condition.z1[permutation],
                condition.x1[permutation],
                detach=True,
            )
            wrong_q_raw = bundle.predictor(wrong_condition)
            wrong_q_hard, _wrong_indices = bundle.codec.quantizer.quantize_input(
                wrong_q_raw,
                max(args._rates),
                detach_codebook=True,
            )
            result["anchor_condition"] = decode_receiver(
                bundle,
                wrong_q_hard.detach(),
                layer1["x1"],
            )["final"]
    return result


def compute_losses(
    out: dict,
    imgs: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
    *,
    quantizer: NestedPrefixVQ,
    train: bool,
) -> dict[str, torch.Tensor]:
    recon_terms: list[torch.Tensor] = []
    vq_terms: list[torch.Tensor] = []
    per_image_mse: dict[int, torch.Tensor] = {}
    if float(args.hard_example_power) > 0.0:
        baseline_mse = (
            out["layer1"]["x1"].float() - imgs.float()
        ).square().flatten(1).mean(dim=1).detach()
        hard_weights = baseline_mse.pow(float(args.hard_example_power))
        hard_weights = hard_weights / hard_weights.mean().clamp_min(1e-8)
        hard_weights = hard_weights.clamp(
            min=float(args.hard_example_min_weight),
            max=float(args.hard_example_max_weight),
        )
    else:
        hard_weights = imgs.new_ones((int(imgs.shape[0]),), dtype=torch.float32)
    for weight, rate in zip(args._rate_weights, args._rates):
        final = out["branches"][rate]["final"]
        error = (final.float() - imgs.float()).square().flatten(1).mean(dim=1)
        per_image_mse[rate] = error
        recon_terms.append(float(weight) * (hard_weights * error).mean())
        vq_terms.append(float(weight) * out["branches"][rate]["stats"]["vq_loss"])
    loss_recon = torch.stack(recon_terms).sum()
    loss_vq = torch.stack(vq_terms).sum()
    monotonic_terms = [
        F.relu(
            per_image_mse[high] - per_image_mse[low].detach() + float(args.monotonic_margin)
        ).mean()
        for low, high in zip(args._rates[:-1], args._rates[1:])
    ]
    loss_monotonic = torch.stack(monotonic_terms).mean() if monotonic_terms else loss_recon.new_zeros(())
    high = out["branches"][max(args._rates)]
    q_target = high["q_hard"].detach().float()
    q_scale = q_target.square().mean().clamp_min(1e-6)
    loss_predict_q_raw = F.mse_loss(out["q_raw"].float(), q_target)
    loss_predict_q_normalized = loss_predict_q_raw / q_scale
    loss_predict_q_hard = F.mse_loss(out["q_pred_hard"].float(), q_target) / q_scale
    loss_predict_q = (
        loss_predict_q_normalized
        if receiver_phase(args) != "none"
        else loss_predict_q_raw
    )
    loss_predictable_code = F.mse_loss(high["q_hard"].float(), out["q_raw"].detach().float())
    loss_predictable_z = F.mse_loss(
        out["z"].float(), out["q_raw"].detach().float()
    ) / q_scale
    loss_predict_final = F.mse_loss(out["pred"]["final"].float(), imgs.float())
    zero = loss_recon.new_zeros(())
    loss_zero_anchor = (
        F.mse_loss(out["anchor_zero"].float(), out["layer1"]["x1"].detach().float())
        if "anchor_zero" in out
        else zero
    )
    loss_shuffle_anchor = (
        F.mse_loss(out["anchor_shuffle"].float(), out["layer1"]["x1"].detach().float())
        if "anchor_shuffle" in out
        else zero
    )
    loss_condition_anchor = (
        F.mse_loss(out["anchor_condition"].float(), out["layer1"]["x1"].detach().float())
        if "anchor_condition" in out
        else zero
    )
    soft_usage_lambda = float(getattr(args, "lambda_soft_usage_entropy", 0.0))
    soft_usage_active = (
        bool(train)
        and soft_usage_lambda > 0.0
        and not bool(args.receiver_only)
        and receiver_phase(args) == "none"
    )
    if soft_usage_active:
        soft_usage_metrics = soft_batch_code_usage_regularizer(quantizer, out["z"], args)
    else:
        # Keep default training bit-for-bit independent of this opt-in term:
        # no soft assignment is materialized and the sender objective below is
        # left untouched when lambda is zero.
        soft_usage_metrics = {
            "loss_soft_usage_entropy": zero,
            "soft_usage_entropy": zero,
            "soft_usage_entropy_ratio": zero,
            "soft_usage_sampled_tokens": zero,
            "soft_usage_enabled": zero,
        }
    receive = receiver_weight(args, epoch)
    sender_objective = (
        loss_recon
        + vq_weight(args, epoch) * loss_vq
        + float(args.lambda_monotonic) * loss_monotonic
        if receiver_phase(args) == "none"
        else loss_recon.new_zeros(())
    )
    if soft_usage_active:
        sender_objective = sender_objective + soft_usage_lambda * soft_usage_metrics[
            "loss_soft_usage_entropy"
        ]
    loss = (
        sender_objective
        + receive * float(args.lambda_predict_q) * loss_predict_q
        + receive * float(args.lambda_predictability) * loss_predictable_code
        + receive * float(args.lambda_predictable_z) * loss_predictable_z
        + receive * float(args.lambda_predict_final) * loss_predict_final
        + receive * float(args.lambda_zero_anchor) * loss_zero_anchor
        + receive * float(args.lambda_shuffle_anchor) * loss_shuffle_anchor
        + receive * float(args.lambda_condition_anchor) * loss_condition_anchor
    )
    return {
        "loss": loss,
        "loss_recon": loss_recon,
        "loss_sender_objective": sender_objective,
        "loss_vq": loss_vq,
        "loss_monotonic": loss_monotonic,
        "loss_predict_q": loss_predict_q,
        "loss_predict_q_raw": loss_predict_q_raw,
        "loss_predict_q_normalized": loss_predict_q_normalized,
        "loss_predict_q_hard": loss_predict_q_hard,
        "q_target_scale": q_scale,
        "loss_predictable_code": loss_predictable_code,
        "loss_predictable_z": loss_predictable_z,
        "loss_predict_final": loss_predict_final,
        "loss_zero_anchor": loss_zero_anchor,
        "loss_shuffle_anchor": loss_shuffle_anchor,
        "loss_condition_anchor": loss_condition_anchor,
        **soft_usage_metrics,
    }


def histogram_metrics(hist: torch.Tensor, rate: int) -> dict[str, float]:
    total = float(hist.sum())
    if total <= 0:
        return {"used": 0.0, "ppl": 0.0, "ppl_ratio": 0.0, "top1": 0.0}
    probabilities = hist / total
    active = probabilities > 0
    entropy = -(probabilities[active] * probabilities[active].log()).sum()
    return {
        "used": float(active.sum()),
        "ppl": float(entropy.exp()),
        "ppl_ratio": float(entropy.exp() / float(rate)),
        "top1": float(probabilities.max()),
    }


def add_grouped_local_counts(
    counts: torch.Tensor,
    local_indices: torch.Tensor,
) -> None:
    """Accumulate [B,C] local round ids into a [C,R] histogram."""

    if local_indices.ndim != 2 or int(local_indices.shape[1]) != int(counts.shape[0]):
        raise ValueError(
            f"local grouped indices/counts mismatch: {tuple(local_indices.shape)} vs {tuple(counts.shape)}"
        )
    rounds = int(counts.shape[1])
    local = local_indices.detach().long().cpu()
    channel_ids = torch.arange(int(counts.shape[0]), dtype=torch.long).view(1, -1)
    keys = channel_ids * rounds + local
    counts += torch.bincount(keys.reshape(-1), minlength=int(counts.numel())).view_as(counts).double()


def grouped_local_metrics(counts: torch.Tensor) -> dict[str, float]:
    """Local-round usage plus variation across samples within each channel."""

    if counts.ndim != 2 or int(counts.shape[0]) < 1 or int(counts.shape[1]) < 1:
        raise ValueError(f"grouped local counts must be [C,R], got {tuple(counts.shape)}")
    rounds = int(counts.shape[1])
    usage = histogram_metrics(counts.sum(dim=0), rounds)
    totals = counts.sum(dim=1)
    valid = totals > 0
    per_channel_top1 = torch.ones_like(totals, dtype=torch.float64)
    if bool(valid.any()):
        per_channel_top1[valid] = (
            counts[valid].max(dim=1).values / totals[valid]
        ).double()
    variation = 1.0 - per_channel_top1
    variant_channels = (counts > 0).sum(dim=1) > 1
    return {
        "local_used": usage["used"],
        "local_ppl": usage["ppl"],
        "local_ppl_ratio": usage["ppl_ratio"],
        "local_top1": usage["top1"],
        "channel_variation": float(variation.mean()),
        "channel_variation_min": float(variation.min()),
        "variant_channel_frac": float(variant_channels.double().mean()),
    }


def nominal_rate_metrics(args: argparse.Namespace, rate: int) -> dict[str, float]:
    """Report the actual candidate count seen by one transmitted token."""

    if str(args.vq_family) == "image-vq":
        tokens_per_image = 16 * 16
        candidates = int(rate)
    else:
        tokens_per_image = int(args.latent_c)
        candidates = (
            int(rate) // int(args.latent_c)
            if str(args.channel_codebook_mode) == "grouped"
            else int(rate)
        )
    bits_per_token = math.log2(float(candidates)) if candidates > 1 else 0.0
    bits_per_image = float(tokens_per_image) * bits_per_token
    return {
        "tokens_per_image": float(tokens_per_image),
        "candidates_per_token": float(candidates),
        "bits_per_token": bits_per_token,
        "bits_per_image": bits_per_image,
        "bpp": bits_per_image / float(256 * 256),
    }


def update_basic_metrics(
    meters: MetricSums,
    losses: dict[str, torch.Tensor],
    out: dict,
    imgs: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    batch = int(imgs.shape[0])
    for name, value in losses.items():
        meters.add(name, float(value.detach()), batch)
    if str(args.vq_family) == "image-vq":
        token_rms = out["z"].float().square().mean(dim=1).sqrt().mean()
        meters.add("image_vq_token_rms", float(token_rms), batch)
    x1_psnr = psnr_values(out["layer1"]["x1"], imgs)
    meters.add("psnr_x1", float(x1_psnr.mean()), batch)
    previous: torch.Tensor | None = None
    for rate in args._rates:
        current = psnr_values(out["branches"][rate]["final"], imgs)
        current_indices = out["branches"][rate]["indices"]
        meters.add(f"psnr_k{rate}", float(current.mean()), batch)
        meters.add(f"delta_x1_k{rate}", float((current - x1_psnr).mean()), batch)
        meters.add(f"vq_mse_k{rate}", float(out["branches"][rate]["stats"]["vq_mse"]), batch)
        change = float(index_change_rate(current_indices))
        meters.add(f"index_change_k{rate}", change, batch)
        meters.add(f"token_change_k{rate}", change, batch)
        if previous is not None:
            meters.add(f"paired_strict_k{rate}", float((current > previous).float().mean()), batch)
            meters.add(f"paired_gain_k{rate}", float((current - previous).mean()), batch)
        previous = current
    pred_psnr = psnr_values(out["pred"]["final"], imgs)
    meters.add("psnr_pred", float(pred_psnr.mean()), batch)
    meters.add("delta_x1_pred", float((pred_psnr - x1_psnr).mean()), batch)
    oracle_indices = out["branches"][max(args._rates)]["indices"]
    accuracy = (out["pred_indices"] == oracle_indices).float().mean()
    meters.add("pred_index_accuracy", float(accuracy), batch)
    pred_change = float(index_change_rate(out["pred_indices"]))
    meters.add("pred_index_change", pred_change, batch)
    meters.add("pred_token_change", pred_change, batch)


@torch.no_grad()
def add_validation_ablations(
    meters: MetricSums,
    out: dict,
    imgs: torch.Tensor,
    bundle: VQBundle,
    args: argparse.Namespace,
) -> None:
    batch = int(imgs.shape[0])
    x1 = out["layer1"]["x1"]
    continuous = bundle.codec.decode(out["z"], x1, bundle.combiner)["final"]
    meters.add("psnr_continuous", float(psnr_values(continuous, imgs).mean()), batch)
    pred_psnr = psnr_values(out["pred"]["final"], imgs)
    pred_zero = decode_receiver(bundle, torch.zeros_like(out["q_pred_hard"]), x1)["final"]
    pred_shuffle_q = bundle.codec.quantizer.shuffle_tokens(out["q_pred_hard"])
    pred_shuffle = decode_receiver(bundle, pred_shuffle_q, x1)["final"]
    meters.add("psnr_pred_zero", float(psnr_values(pred_zero, imgs).mean()), batch)
    meters.add("psnr_pred_shuffle", float(psnr_values(pred_shuffle, imgs).mean()), batch)
    meters.add(
        "pred_drop_zero",
        float((pred_psnr - psnr_values(pred_zero, imgs)).mean()),
        batch,
    )
    meters.add(
        "pred_drop_shuffle",
        float((pred_psnr - psnr_values(pred_shuffle, imgs)).mean()),
        batch,
    )
    if batch > 1:
        # Shuffle z1/x1 together so the predictor still receives a valid paired
        # Layer1 condition from another image.  Decode with the *original* x1;
        # otherwise the Layer1 bypass itself would make this test pass even if
        # the predictor completely ignored its condition.
        permutation = torch.roll(torch.arange(batch, device=x1.device), shifts=1)
        condition = out["condition"]
        shuffled_condition = make_receiver_condition(
            condition.z1[permutation],
            condition.x1[permutation],
            detach=True,
        )
        shuffled_q_raw = bundle.predictor(shuffled_condition)
        shuffled_q_hard, _shuffled_indices = bundle.codec.quantizer.quantize_input(
            shuffled_q_raw,
            max(args._rates),
            detach_codebook=True,
        )
        condition_shuffled = decode_receiver(bundle, shuffled_q_hard, x1)["final"]
        condition_shuffle_drop = float(
            (pred_psnr - psnr_values(condition_shuffled, imgs)).mean()
        )
        condition_shuffle_psnr = float(psnr_values(condition_shuffled, imgs).mean())
    else:
        # A singleton batch has no different condition to pair with.  Recording
        # zero keeps the gate conservative instead of fabricating evidence.
        condition_shuffle_drop = 0.0
        condition_shuffle_psnr = float(pred_psnr.mean())
    meters.add("psnr_condition_shuffle", condition_shuffle_psnr, batch)
    meters.add("condition_shuffle_drop", condition_shuffle_drop, batch)
    for rate in args._rates:
        branch = out["branches"][rate]
        final_psnr = psnr_values(branch["final"], imgs)
        zero = bundle.codec.decode(torch.zeros_like(branch["q_hard"]), x1, bundle.combiner)["final"]
        shuffled_q = bundle.codec.quantizer.shuffle_tokens(branch["q_hard"])
        shuffled = bundle.codec.decode(shuffled_q, x1, bundle.combiner)["final"]
        zero_psnr = psnr_values(zero, imgs)
        shuffle_psnr = psnr_values(shuffled, imgs)
        meters.add(f"drop_zero_k{rate}", float((final_psnr - zero_psnr).mean()), batch)
        meters.add(f"drop_shuffle_k{rate}", float((final_psnr - shuffle_psnr).mean()), batch)


def clear_optimizer_rows(optimizer: optim.Optimizer, parameter: nn.Parameter, rows: torch.Tensor) -> None:
    if int(rows.numel()) == 0:
        return
    state = optimizer.state.get(parameter, {})
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(name)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and int(value.shape[0]) == int(parameter.shape[0]):
            value[rows] = 0


def run_epoch(
    loader,
    *,
    bundle: VQBundle,
    optimizer: optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    train: bool,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor] | None]:
    stabilizer_warmup = configure_image_vq_stabilizer_warmup(
        bundle,
        args,
        epoch=int(epoch),
        train=bool(train),
    )
    bundle.source.e1.eval()
    bundle.source.d1.eval()
    bundle.codec.train(train)
    bundle.combiner.train(train)
    bundle.predictor.train(
        train
        and not bool(args.oracle_only)
        and any(parameter.requires_grad for parameter in bundle.predictor.parameters())
    )
    if bundle.receiver_stack is not None:
        bundle.receiver_stack.train(
            train and any(parameter.requires_grad for parameter in bundle.receiver_stack.parameters())
        )
    if bool(args.freeze_encoder):
        bundle.codec.e2.eval()
        bundle.codec.analysis.eval()
    if stabilizer_warmup:
        bundle.codec.e2.eval()
        bundle.codec.analysis.eval()
        bundle.codec.token_encoder.eval()
    if bool(args.freeze_source_d2):
        bundle.codec.d2.eval()
    if bool(args.freeze_combiner):
        bundle.combiner.eval()
    if bool(args.receiver_only):
        bundle.codec.eval()
        bundle.combiner.eval()
    meters = MetricSums()
    histograms = {rate: torch.zeros(rate, dtype=torch.float64) for rate in args._rates}
    grouped = str(args.vq_family) == "channel-vq" and str(args.channel_codebook_mode) == "grouped"
    local_counts = (
        {
            rate: torch.zeros(
                int(args.latent_c), int(rate) // int(args.latent_c), dtype=torch.float64
            )
            for rate in args._rates
        }
        if grouped
        else {}
    )
    pred_local_counts = (
        torch.zeros(
            int(args.latent_c),
            max(args._rates) // int(args.latent_c),
            dtype=torch.float64,
        )
        if grouped
        else None
    )
    maximum = int(args.max_train_batches if train else args.max_val_batches)
    last_refresh: tuple[torch.Tensor, torch.Tensor] | None = None
    audited = False
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if maximum > 0 and batch_index > maximum:
            break
        imgs = imgs.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = forward_bundle(bundle, imgs, args, train=train, epoch=epoch)
            if not audited:
                assert_training_targets_are_not_inputs(
                    bundle.predictor,
                    out["condition"],
                    source_targets={
                        "img": imgs,
                        "z2": out["z"],
                        "oracle_q2": out["branches"][max(args._rates)]["q_hard"],
                        "oracle_indices": out["branches"][max(args._rates)]["indices"],
                    },
                )
                audited = True
            losses = compute_losses(
                out,
                imgs,
                args,
                epoch,
                quantizer=bundle.codec.quantizer,
                train=train,
            )
            if train:
                if optimizer is None:
                    raise RuntimeError("training requires optimizer")
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if float(args.grad_clip_norm) > 0:
                    modules = [bundle.codec, bundle.combiner, bundle.predictor]
                    if bundle.receiver_stack is not None:
                        modules.append(bundle.receiver_stack)
                    parameters = [
                        parameter
                        for module in modules
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    ]
                    torch.nn.utils.clip_grad_norm_(parameters, float(args.grad_clip_norm))
                optimizer.step()
        update_basic_metrics(meters, losses, out, imgs, args)
        meters.add("image_vq_stabilizer_warmup", float(stabilizer_warmup), int(imgs.shape[0]))
        if not train:
            add_validation_ablations(meters, out, imgs, bundle, args)
        for rate in args._rates:
            counts = torch.bincount(
                out["branches"][rate]["indices"].detach().reshape(-1).cpu(),
                minlength=rate,
            ).double()
            histograms[rate] += counts[:rate]
            if grouped:
                local = bundle.codec.quantizer.indices_to_local(
                    out["branches"][rate]["indices"], rate
                )
                add_grouped_local_counts(local_counts[rate], local)
        if grouped and pred_local_counts is not None:
            pred_local = bundle.codec.quantizer.indices_to_local(
                out["pred_indices"], max(args._rates)
            )
            add_grouped_local_counts(pred_local_counts, pred_local)
        last_refresh = (
            out["z"].detach(),
            out["branches"][max(args._rates)]["indices"].detach(),
        )
    metrics = meters.means()
    metrics["receiver_only_audit"] = 1.0 if audited else 0.0
    metrics["sender_hash_audit"] = audit_sender_state(bundle, args)
    for rate, histogram in histograms.items():
        for name, value in histogram_metrics(histogram, rate).items():
            metrics[f"{name}_k{rate}"] = value
        for name, value in nominal_rate_metrics(args, rate).items():
            metrics[f"{name}_k{rate}"] = value
        if grouped:
            for name, value in grouped_local_metrics(local_counts[rate]).items():
                metrics[f"{name}_k{rate}"] = value
    if grouped and pred_local_counts is not None:
        for name, value in grouped_local_metrics(pred_local_counts).items():
            metrics[f"pred_{name}"] = value
    if not train:
        health = bundle.codec.quantizer.codebook_metrics(
            max(args._rates),
            l2_max_samples=int(args.codebook_l2_samples),
            rank_max_samples=int(args.codebook_rank_samples),
            seed=int(args.seed),
        )
        metrics.update(health)
        deltas = [metrics[f"delta_x1_k{rate}"] for rate in args._rates]
        drops_ok = all(
            metrics[f"drop_zero_k{rate}"] >= float(args.min_ablation_drop)
            and metrics[f"drop_shuffle_k{rate}"] >= float(args.min_ablation_drop)
            for rate in args._rates
        )
        monotonic = all(
            metrics[f"psnr_k{high}"] > metrics[f"psnr_k{low}"]
            for low, high in zip(args._rates[:-1], args._rates[1:])
        )
        paired_capacity = all(
            metrics[f"paired_gain_k{high}"] >= float(args.min_paired_gain_db)
            and metrics[f"paired_strict_k{high}"] >= float(args.min_paired_strict)
            for _low, high in zip(args._rates[:-1], args._rates[1:])
        )
        if grouped:
            noncollapse = all(
                metrics[f"local_ppl_k{rate}"] >= float(args.min_local_ppl)
                and metrics[f"local_top1_k{rate}"] <= float(args.max_local_top1)
                and metrics[f"channel_variation_k{rate}"] >= float(args.min_channel_variation)
                and metrics[f"variant_channel_frac_k{rate}"] >= float(args.min_variant_channel_frac)
                for rate in args._rates
            )
            usage_scaling = all(
                metrics[f"local_ppl_k{high}"] > metrics[f"local_ppl_k{low}"]
                for low, high in zip(args._rates[:-1], args._rates[1:])
            )
            # K=C has one candidate per channel, carries zero index bits, and
            # cannot pass a shuffle-relevance test.  It remains available only
            # as an explicitly requested structural diagnostic; no gate
            # exemption may promote it as a capacity operating point.
            promotion_eligible = all(int(rate) > int(args.latent_c) for rate in args._rates)
            metrics["grouped_promotion_eligible"] = float(promotion_eligible)
        else:
            noncollapse = all(
                metrics[f"ppl_ratio_k{rate}"] >= float(args.min_ppl_ratio)
                and metrics[f"top1_k{rate}"] <= float(args.max_top1)
                and metrics[f"index_change_k{rate}"] >= float(args.min_index_change)
                for rate in args._rates
            )
            usage_scaling = all(
                metrics[f"ppl_k{high}"] > metrics[f"ppl_k{low}"]
                for low, high in zip(args._rates[:-1], args._rates[1:])
            )
        metrics["noncollapse_goal_met"] = float(noncollapse)
        metrics["usage_scaling_goal_met"] = float(usage_scaling)
        metrics["capacity_goal_met"] = float(monotonic and paired_capacity)
        metrics["oracle_goal_met"] = float(
            all(delta > 0 for delta in deltas)
            and drops_ok
            and monotonic
            and paired_capacity
            and noncollapse
            and usage_scaling
            and (not grouped or promotion_eligible)
        )
        receiver_local_ok = (
            not grouped
            or (
                metrics["pred_local_ppl"] >= float(args.min_local_ppl)
                and metrics["pred_local_top1"] <= float(args.max_local_top1)
                and metrics["pred_channel_variation"] >= float(args.min_channel_variation)
                and metrics["pred_variant_channel_frac"] >= float(args.min_variant_channel_frac)
            )
        )
        metrics["receiver_goal_met"] = float(
            metrics["delta_x1_pred"] > float(args.min_receiver_delta)
            and metrics["pred_drop_zero"] >= float(args.min_ablation_drop)
            and metrics["pred_drop_shuffle"] >= float(args.min_ablation_drop)
            and metrics["condition_shuffle_drop"] >= float(args.min_condition_shuffle_drop)
            and receiver_local_ok
        )
        metrics["goal_met"] = float(metrics["oracle_goal_met"] and metrics["receiver_goal_met"])
    return metrics, last_refresh


DISPLAY_BASE = (
    "loss",
    "loss_recon",
    "loss_sender_objective",
    "loss_vq",
    "loss_monotonic",
    "loss_soft_usage_entropy",
    "soft_usage_entropy",
    "soft_usage_entropy_ratio",
    "soft_usage_sampled_tokens",
    "soft_usage_enabled",
    "loss_predict_q",
    "loss_predict_q_raw",
    "loss_predict_q_normalized",
    "loss_predict_q_hard",
    "loss_predictable_code",
    "loss_predictable_z",
    "q_target_scale",
    "loss_predict_final",
    "loss_zero_anchor",
    "loss_shuffle_anchor",
    "loss_condition_anchor",
    "image_vq_token_rms",
    "image_vq_stabilizer_warmup",
    "psnr_x1",
    "psnr_continuous",
    "psnr_pred",
    "delta_x1_pred",
    "pred_index_accuracy",
    "pred_index_change",
    "pred_token_change",
    "pred_local_used",
    "pred_local_ppl",
    "pred_local_ppl_ratio",
    "pred_local_top1",
    "pred_channel_variation",
    "pred_channel_variation_min",
    "pred_variant_channel_frac",
    "pred_drop_zero",
    "pred_drop_shuffle",
    "psnr_pred_zero",
    "psnr_pred_shuffle",
    "psnr_condition_shuffle",
    "condition_shuffle_drop",
    "receiver_only_audit",
    "sender_hash_audit",
    "oracle_goal_met",
    "receiver_goal_met",
    "capacity_goal_met",
    "noncollapse_goal_met",
    "usage_scaling_goal_met",
    "grouped_promotion_eligible",
    "goal_met",
)


def display(metrics: dict[str, float], rates: list[int]) -> dict[str, float]:
    result = {name: metrics[name] for name in DISPLAY_BASE if name in metrics}
    for rate in rates:
        for prefix in (
            "psnr",
            "delta_x1",
            "used",
            "ppl",
            "ppl_ratio",
            "top1",
            "drop_zero",
            "drop_shuffle",
            "paired_strict",
            "paired_gain",
            "vq_mse",
            "soft_usage_loss",
            "soft_usage_entropy",
            "soft_usage_entropy_ratio",
            "soft_usage_candidate_count",
            "index_change",
            "token_change",
            "local_used",
            "local_ppl",
            "local_ppl_ratio",
            "local_top1",
            "channel_variation",
            "channel_variation_min",
            "variant_channel_frac",
            "tokens_per_image",
            "candidates_per_token",
            "bits_per_token",
            "bits_per_image",
            "bpp",
        ):
            key = f"{prefix}_k{rate}"
            if key in metrics:
                result[key] = metrics[key]
    return result


def experiment_name(args: argparse.Namespace) -> str:
    rates = "-".join(str(rate) for rate in args._rates)
    mode = f"_{args.channel_codebook_mode}" if str(args.vq_family) == "channel-vq" else ""
    return (
        f"nested_{args.arch}_l2{args._layer2_arch}_{args.vq_family}{mode}_c{int(args.latent_c)}_"
        f"d{int(args._embedding_dim)}_k{rates}"
    )


def soft_usage_checkpoint_record(
    args: argparse.Namespace,
    train_metrics: dict[str, float] | None,
) -> dict[str, object]:
    """Persist opt-in configuration plus the last *training* entropy values.

    Validation intentionally does not materialize this regularizer, so its
    metrics are zero there.  Keeping the last train-side subset next to the
    usual validation ``metrics`` prevents a checkpoint from silently losing
    the actual entropy signal that produced it.
    """

    recorded = {
        name: float(value)
        for name, value in (train_metrics or {}).items()
        if name == "loss_soft_usage_entropy" or name.startswith("soft_usage_")
    }
    return {
        "lambda": float(args.lambda_soft_usage_entropy),
        "temperature": float(args.soft_usage_temperature),
        "sample_tokens": int(args.soft_usage_sample_tokens),
        "query_chunk_size": int(args.query_chunk_size),
        "train_only": True,
        "enabled": bool(float(args.lambda_soft_usage_entropy) > 0.0),
        "last_train_metrics": recorded,
    }


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    bundle: VQBundle,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    metrics: dict[str, float],
    init_stats: dict[str, float | str],
    best: float,
    best_goal: float,
    best_q: float,
    train_metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sender_hash = sender_state_sha256(bundle)
    if receiver_phase(args) != "none" and sender_hash != bundle.sender_state_hash:
        raise AssertionError("refusing to save a receiver checkpoint with a mutated sender")
    payload = {
            "stage": "explore2_layer2_nested_vq",
            "epoch": int(epoch),
            "metrics": metrics,
            "args": {name: value for name, value in vars(args).items() if not name.startswith("_")},
            "e2_input_order": str(args.e2_input_order),
            "channel_codebook_mode": str(args.channel_codebook_mode),
            "rates": list(args._rates),
            "rate_weights": list(args._rate_weights),
            "source_layer1_checkpoint": str(args.source_checkpoint),
            "source_layer2_checkpoint": str(args.layer2_source_checkpoint),
            "e1_state_dict": bundle.source.e1.state_dict(),
            "d1_state_dict": bundle.source.d1.state_dict(),
            "codec_state_dict": bundle.codec.state_dict(),
            "combiner_state_dict": bundle.combiner.state_dict(),
            "predictor_state_dict": bundle.predictor.state_dict(),
            "receiver_stack_state_dict": (
                bundle.receiver_stack.state_dict() if bundle.receiver_stack is not None else None
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "init_stats": init_stats,
            "best_psnr": float(best),
            "best_goal_psnr": float(best_goal),
            "best_q_hard_nmse": float(best_q),
            "soft_batch_code_usage_entropy": soft_usage_checkpoint_record(args, train_metrics),
            "sender_state_hash": sender_hash,
            "receiver_phase": receiver_phase(args),
            "receiver_stack": str(args.receiver_stack),
            "receiver_contract": {
                "inputs": ["z1", "x1"],
                "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
                "output": "hard_codebook_q2_hat",
                "train_transform": "RandomCrop(256)+RandomHorizontalFlip+ToTensor",
                "validation_transform": "CenterCrop(256)+ToTensor",
            },
        }
    torch.save(payload, path)
    print(f"saved checkpoint: {path}", flush=True)


def load_resume(
    path: str,
    bundle: VQBundle,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, float, float, float, dict[str, float | str]]:
    payload = torch.load(resolve_path(path), map_location=device, weights_only=False)
    if str(payload.get("stage", "")) != "explore2_layer2_nested_vq":
        raise ValueError(f"not an explore-2 nested VQ checkpoint: {path}")
    saved = dict(payload.get("args", {}))
    # Checkpoints created before grouped channel-VQ existed are unambiguously
    # the legacy global mode and remain valid for receiver-only continuation.
    saved.setdefault("channel_codebook_mode", "global")
    # Early explore-2 checkpoints serialized these newly introduced options as
    # ``None``.  Their actual topology was the old matched Layer1/Layer2 path,
    # so normalize only the legacy null spelling; explicit new values still
    # participate in the strict resume contract below.
    if saved.get("layer2_arch") is None:
        saved["layer2_arch"] = "match"
    if saved.get("layer2_source_checkpoint") is None:
        saved["layer2_source_checkpoint"] = ""
    # Native-dimensional checkpoints predate the explicit embedding-dim knob.
    saved.setdefault("embedding_dim", 0)
    # The default keeps the historical direct image-VQ topology, so old
    # checkpoints remain resumable while opt-in RMS runs are topology-checked.
    saved.setdefault("image_vq_stabilizer", "none")
    saved.setdefault("image_vq_rms_eps", 1e-6)
    if str(payload.get("e2_input_order", "")) != str(args.e2_input_order):
        raise ValueError(
            "resume contract mismatch e2_input_order: "
            f"{payload.get('e2_input_order')!r} != {args.e2_input_order!r}"
        )
    for key in (
        "arch",
        "layer2_arch",
        "vq_family",
        "channel_codebook_mode",
        "latent_c",
        "embedding_dim",
        "image_vq_stabilizer",
        "image_vq_rms_eps",
        "rates",
        "e2_input_order",
        "codebook_init",
    ):
        if key == "rates" and bool(args.allow_eval_rate_override):
            saved_rates = parse_int_list(str(saved.get("rates", "")))
            requested_rates = [int(value) for value in args._rates]
            if (
                not saved_rates
                or not requested_rates
                or max(saved_rates) != max(requested_rates)
                or any(rate <= 0 or rate > max(saved_rates) for rate in requested_rates)
            ):
                raise ValueError(
                    "eval rate override must keep the checkpoint Kmax and use positive prefixes: "
                    f"saved={saved_rates}, requested={requested_rates}"
                )
            continue
        if str(saved.get(key)) != str(getattr(args, key)):
            raise ValueError(f"resume contract mismatch {key}: {saved.get(key)!r} != {getattr(args, key)!r}")
    bundle.codec.load_state_dict(payload["codec_state_dict"], strict=True)
    bundle.combiner.load_state_dict(payload["combiner_state_dict"], strict=True)
    if not bool(args.reset_predictor_on_resume):
        bundle.predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    else:
        print("reset predictor on resume: checkpoint predictor state skipped", flush=True)
    if bundle.receiver_stack is not None:
        receiver_state = payload.get("receiver_stack_state_dict")
        if isinstance(receiver_state, dict):
            bundle.receiver_stack.load_state_dict(receiver_state, strict=True)
            print("loaded independent receiver decode stack from checkpoint", flush=True)
        else:
            bundle.receiver_stack.initialize_from_sender(
                bundle.codec.synthesis,
                bundle.codec.d2,
                bundle.combiner,
            )
            print("initialized independent receiver decode stack from loaded sender", flush=True)
    actual_sender_hash = sender_state_sha256(bundle)
    saved_sender_hash = str(payload.get("sender_state_hash", ""))
    if saved_sender_hash and saved_sender_hash != actual_sender_hash:
        raise ValueError(
            "resume sender hash mismatch: "
            f"checkpoint={saved_sender_hash} loaded={actual_sender_hash}"
        )
    bundle.sender_state_hash = actual_sender_hash
    if not bool(args.reset_optimizer_on_resume):
        saved_phase = str(payload.get("receiver_phase", saved.get("receiver_phase", "none")))
        saved_stack = str(payload.get("receiver_stack", saved.get("receiver_stack", "shared")))
        if saved_phase != receiver_phase(args) or saved_stack != str(args.receiver_stack):
            raise ValueError(
                "changing receiver phase/topology requires --reset-optimizer-on-resume: "
                f"saved={saved_phase}/{saved_stack} current={receiver_phase(args)}/{args.receiver_stack}"
            )
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return (
        int(payload["epoch"]) + 1,
        float(payload.get("best_psnr", float("-inf"))),
        float(payload.get("best_goal_psnr", float("-inf"))),
        float(payload.get("best_q_hard_nmse", float("inf"))),
        dict(payload.get("init_stats", {})),
    )


def print_header(
    args: argparse.Namespace,
    bundle: VQBundle,
    init_stats: dict[str, float | str],
    train_n: int,
    val_n: int,
) -> None:
    print(f"=== explore-2 | nested direct Layer2 {args.vq_family} | {args.arch} ===", flush=True)
    print("实验设计", flush=True)
    print(
        "  sender=img+x1 -> exact E2_320 -> analysis -> shared-prefix VQ -> synthesis -> "
        "exact D2_320 -> residual combiner -> x2",
        flush=True,
    )
    print(
        "  receiver=(z1,x1) -> direct q predictor -> hard nearest shared code -> "
        f"{'independent synthesis/D2/combiner' if independent_receiver(args) else 'sender synthesis/D2/combiner'} -> x2_hat",
        flush=True,
    )
    print("  predictor forbidden inputs=img,z2,q2,oracle_indices", flush=True)
    print(
        f"  layer1_arch={args.arch} layer2_arch={args._layer2_arch} "
        f"family={args.vq_family} C={int(args.latent_c)} "
        f"embedding_dim={int(args._embedding_dim)} rates={args._rates} "
        f"weights={args._rate_weights} channel_codebook_mode={args.channel_codebook_mode} "
        f"e2_input_order={args.e2_input_order} init={init_stats}",
        flush=True,
    )
    print("loss设计", flush=True)
    print(
        f"  weighted MSE(x2_K,img) + ramp({float(args.lambda_vq):g})*VQ "
        f"+{float(args.lambda_monotonic):g}*monotonic_hinge "
        f"+receiver[{float(args.lambda_predict_q):g}*"
        f"{'normalized_' if receiver_phase(args) != 'none' else ''}q + "
        f"{float(args.lambda_predict_final):g}*MSE(x2_hat,img) + anchors="
        f"{float(args.lambda_zero_anchor):g}/{float(args.lambda_shuffle_anchor):g}/"
        f"{float(args.lambda_condition_anchor):g}]",
        flush=True,
    )
    print(
        "  soft_batch_code_usage_entropy: "
        f"lambda={float(args.lambda_soft_usage_entropy):g} "
        f"temperature={float(args.soft_usage_temperature):g} "
        f"sample_tokens={int(args.soft_usage_sample_tokens)}(0=all) "
        f"query_chunk={int(args.query_chunk_size)}; "
        f"training_only=True status={'enabled' if float(args.lambda_soft_usage_entropy) > 0.0 else 'disabled(default)'}",
        flush=True,
    )
    if bool(args.bound_latent) or int(args.continuous_recon_warmup_epochs) > 0:
        print(
            f"  fresh-codec curriculum: bound_latent={bool(args.bound_latent)}; "
            f"continuous_warmup={int(args.continuous_recon_warmup_epochs)}; "
            f"hard_transition={int(args.continuous_recon_transition_epochs)}; "
            "validation is always hard-VQ",
            flush=True,
        )
    print("模块选择", flush=True)
    if receiver_phase(args) != "none":
        print(
            "  sender E1/D1/E2/analysis/codebook/synthesis/D2/combiner=frozen; "
            f"predictor={'trainable' if any(p.requires_grad for p in bundle.predictor.parameters()) else 'frozen'}; "
            f"receiver synthesis/D2/combiner="
            f"{'trainable' if bundle.receiver_stack is not None and any(p.requires_grad for p in bundle.receiver_stack.parameters()) else 'frozen'}; "
            f"condition={args.condition_mode}",
            flush=True,
        )
    else:
        print(
            f"  E1/D1=frozen; E2_320/D2_320/adapters/quantizer/combiner=configured by freeze flags; "
            f"predictor={bundle.predictor.__class__.__name__} condition={args.condition_mode}",
            flush=True,
        )
    print(
        f"  combiner={args.combiner} freeze_encoder={bool(args.freeze_encoder)} "
        f"freeze_codebook={bool(args.freeze_codebook)} freeze_source_d2={bool(args.freeze_source_d2)} "
        f"freeze_combiner={bool(args.freeze_combiner)}",
        flush=True,
    )
    if str(args.vq_family) == "image-vq":
        print(
            f"  image_vq_stabilizer={args.image_vq_stabilizer} "
            f"stabilize_epochs={int(args.image_vq_stabilize_epochs)} "
            f"rms_eps={float(args.image_vq_rms_eps):g}; "
            "warmup freezes only pre-VQ encoder/codebook, not decoder/D2/combiner",
            flush=True,
        )
    print(
        f"  oracle_only_phase={bool(args.oracle_only)} "
        f"receiver_only_phase={bool(args.receiver_only)} receiver_phase={receiver_phase(args)} "
        f"receiver_stack={args.receiver_stack} predictor_lr={float(args.predictor_lr):g} "
        f"receiver_decoder_lr={float(args.receiver_decoder_lr):g}",
        flush=True,
    )
    if receiver_phase(args) != "none":
        print(
            f"  sender_frozen_hash={bundle.sender_state_hash} no_alias=PASS shared_quantizer=PASS",
            flush=True,
        )
    if receiver_phase(args) != "none":
        dead_refresh = "disabled(receiver_sender_frozen)"
    elif bool(args.freeze_codebook):
        dead_refresh = "disabled(frozen_codebook)"
    elif (
        str(args.vq_family) == "image-vq"
        and str(args.image_vq_stabilizer) != "none"
        and int(args.image_vq_stabilize_epochs) > 0
    ):
        dead_refresh = (
            f"every_{int(args.dead_refresh_every)}_epochs_after_rms_warmup_"
            f"{int(args.image_vq_stabilize_epochs)}"
        )
    else:
        dead_refresh = f"every_{int(args.dead_refresh_every)}_epochs"
    print(
        f"  gates: paired_gain>={float(args.min_paired_gain_db):g}dB "
        f"paired_strict>={float(args.min_paired_strict):g} "
        f"index_change>={float(args.min_index_change):g} "
        f"condition_shuffle_drop>={float(args.min_condition_shuffle_drop):g}dB; "
        f"dead_refresh={dead_refresh}",
        flush=True,
    )
    if str(args.vq_family) == "channel-vq" and str(args.channel_codebook_mode) == "grouped":
        rate_contract = {
            int(rate): {
                "candidates_per_channel": int(rate) // int(args.latent_c),
                "bits_per_token": nominal_rate_metrics(args, int(rate))["bits_per_token"],
                "bits_per_image": nominal_rate_metrics(args, int(rate))["bits_per_image"],
                "bpp": nominal_rate_metrics(args, int(rate))["bpp"],
            }
            for rate in args._rates
        }
        diagnostic_rates = [
            int(rate) for rate in args._rates if int(rate) == int(args.latent_c)
        ]
        print(
            f"  grouped_local_gates: ppl>={float(args.min_local_ppl):g} "
            f"top1<={float(args.max_local_top1):g} "
            f"variation>={float(args.min_channel_variation):g} "
            f"variant_channel_frac>={float(args.min_variant_channel_frac):g}; "
            f"rate_contract={rate_contract}",
            flush=True,
        )
        print(
            "  grouped_rate_semantics: K is total round-major rows; each channel sees K/C "
            "candidates; honest bits/image=C*log2(K/C). These K values are not rate-equivalent "
            "to image-VQ/global-channel K. "
            f"promotion_scope={'NONPROMOTABLE_K_EQUALS_C_DIAGNOSTIC' if diagnostic_rates else 'FORMAL_ALL_K_GREATER_THAN_C'} "
            f"diagnostic_rates={diagnostic_rates}; dead refresh is same-channel only",
            flush=True,
        )
    print(
        f"  train={train_n} val={val_n} batch={int(args.batch_size)} workers="
        f"{int(args.num_workers)}/{int(args.val_num_workers)}; "
        "crop=train:RandomCrop(256)+RandomHorizontalFlip, val:CenterCrop(256)",
        flush=True,
    )


def build_loaders(args: argparse.Namespace, source_args: argparse.Namespace):
    source_args.data_dir = str(args.data_dir)
    source_args.batch_size = int(args.batch_size)
    source_args.test_batch = int(args.test_batch)
    source_args.num_workers = int(args.num_workers)
    source_args.val_num_workers = int(args.val_num_workers)
    source_args.cpu = bool(args.cpu)
    config = base.jsccf_io.build_config(source_args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(config)
    assert_div2k_crop_protocol(train_loader, val_loader)
    return train_loader, val_loader, config.device


def build_optimizer(args: argparse.Namespace, bundle: VQBundle) -> optim.Optimizer:
    phase = receiver_phase(args)
    if phase == "q-pretrain":
        groups = [
            {
                "params": [p for p in bundle.predictor.parameters() if p.requires_grad],
                "lr": float(args.predictor_lr),
                "name": "predictor",
            }
        ]
    elif phase == "decoder-warmup":
        if bundle.receiver_stack is None:
            raise AssertionError("decoder-warmup requires an independent receiver stack")
        groups = [
            {
                "params": [p for p in bundle.receiver_stack.parameters() if p.requires_grad],
                "lr": float(args.receiver_decoder_lr),
                "name": "receiver_decoder",
            }
        ]
    elif phase == "joint":
        if bundle.receiver_stack is None:
            raise AssertionError("joint receiver phase requires an independent receiver stack")
        groups = [
            {
                "params": [p for p in bundle.predictor.parameters() if p.requires_grad],
                "lr": float(args.predictor_lr),
                "name": "predictor",
            },
            {
                "params": [p for p in bundle.receiver_stack.parameters() if p.requires_grad],
                "lr": float(args.receiver_decoder_lr),
                "name": "receiver_decoder",
            },
        ]
    else:
        parameters = [
            parameter
            for module in (bundle.codec, bundle.combiner, bundle.predictor)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        groups = [{"params": parameters, "lr": float(args.lr), "name": "default"}]
    groups = [group for group in groups if group["params"]]
    if not groups:
        raise ValueError("configuration freezes every trainable module")
    seen: set[int] = set()
    for group in groups:
        for parameter in group["params"]:
            if id(parameter) in seen:
                raise AssertionError("optimizer parameter appears in more than one group")
            seen.add(id(parameter))
    return optim.AdamW(groups, weight_decay=float(args.weight_decay))


def resolve_embedding_dim(args: argparse.Namespace, source: SourceLayer2) -> int:
    """Resolve D only after the selected Layer1/Layer2 checkpoint is known."""

    native = (
        int(args.latent_c)
        if str(args.vq_family) == "image-vq"
        else int(source.layer2_args.latent_h) * int(source.layer2_args.latent_w)
    )
    resolved = int(args.embedding_dim) if int(args.embedding_dim) > 0 else native
    if resolved < 1:
        raise ValueError(f"resolved embedding dimension must be positive, got {resolved}")
    args._embedding_dim = resolved
    return resolved


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    source_path = args.source_checkpoint or DEFAULT_SOURCES[str(args.arch)]
    checkpoint = base.jsccf_io.load_checkpoint(str(resolve_path(source_path)))
    probe_args = argparse.Namespace(**checkpoint["args"])
    train_loader, val_loader, device = build_loaders(args, probe_args)
    source = load_source(args, device)
    resolve_embedding_dim(args, source)
    bundle = build_bundle(args, source, device)
    assert_receiver_topology(bundle, args)
    if bool(args.oracle_only) and any(
        parameter.requires_grad for parameter in bundle.predictor.parameters()
    ):
        raise AssertionError("--oracle-only must freeze every predictor parameter")
    optimizer = build_optimizer(args, bundle)
    start = 1
    best = float("-inf")
    best_goal = float("-inf")
    best_q = float("inf")
    init_stats: dict[str, float | str]
    if args.resume:
        start, best, best_goal, best_q, init_stats = load_resume(
            args.resume, bundle, optimizer, args, device
        )
        if bool(args.receiver_only):
            best = float("-inf")
            best_goal = float("-inf")
            if receiver_phase(args) == "q-pretrain":
                best_q = float("inf")
    else:
        init_stats = initialize_pca(train_loader, bundle, args, device)
        init_stats.update(initialize_codebook(train_loader, bundle, args, device))
    assert_receiver_topology(bundle, args)
    if receiver_phase(args) != "none":
        if not bundle.sender_state_hash:
            bundle.sender_state_hash = sender_state_sha256(bundle)
        audit_sender_state(bundle, args)
    print_header(args, bundle, init_stats, len(train_loader.dataset), len(val_loader.dataset))
    if bool(args.eval_init_only):
        init_metrics, _unused = run_epoch(
            val_loader,
            bundle=bundle,
            optimizer=None,
            args=args,
            device=device,
            epoch=0,
            train=False,
        )
        print(f"[nested-vq init val] {display(init_metrics,args._rates)}", flush=True)
        return
    output_dir = resolve_path(args.save_dir) / str(args.version)
    name = experiment_name(args)
    latest: dict[str, float] = {}
    for epoch in range(start, int(args.epochs) + 1):
        began = time.time()
        train_metrics, refresh_data = run_epoch(
            train_loader,
            bundle=bundle,
            optimizer=optimizer,
            args=args,
            device=device,
            epoch=epoch,
            train=True,
        )
        print(
            f"[nested-vq train {epoch:03d}/{int(args.epochs):03d}] {display(train_metrics,args._rates)} "
            f"vq_weight={vq_weight(args,epoch):g} time={time.time()-began:.1f}s",
            flush=True,
        )
        latest = train_metrics
        if (
            refresh_data is not None
            and not bool(args.receiver_only)
            and not bool(args.freeze_codebook)
            and not image_vq_stabilizer_warmup_active(args, epoch=epoch, train=True)
            and int(args.dead_refresh_every) > 0
            and epoch >= int(args.dead_refresh_start)
            and epoch % int(args.dead_refresh_every) == 0
        ):
            refreshed = bundle.codec.quantizer.refresh_dead_codes(
                refresh_data[0],
                refresh_data[1],
                k=max(args._rates),
                threshold=float(args.dead_threshold),
                patience=int(args.dead_patience),
                max_refresh=int(args.dead_max_refresh),
                seed=int(args.seed) + epoch,
            )
            clear_optimizer_rows(optimizer, bundle.codec.quantizer.codebook, refreshed)
            print(f"[nested-vq dead refresh] epoch={epoch} refreshed={int(refreshed.numel())}", flush=True)
        if epoch == 1 or epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            val_metrics, _unused = run_epoch(
                val_loader,
                bundle=bundle,
                optimizer=None,
                args=args,
                device=device,
                epoch=epoch,
                train=False,
            )
            print(f"[nested-vq val {epoch:03d}] {display(val_metrics,args._rates)}", flush=True)
            latest = val_metrics
            if receiver_phase(args) == "q-pretrain":
                score = -float(val_metrics["loss_predict_q_hard"])
            else:
                score = float(
                    val_metrics["psnr_pred"]
                    if bool(args.receiver_only)
                    else val_metrics[f"psnr_k{max(args._rates)}"]
                )
            goal = bool(val_metrics["goal_met"])
            improved = score > best
            improved_goal = goal and score > best_goal
            hard_q = float(val_metrics["loss_predict_q_hard"])
            improved_q = receiver_phase(args) == "q-pretrain" and hard_q < best_q
            if improved:
                best = score
                save_checkpoint(
                    output_dir / f"{name}_best.pth",
                    epoch=epoch,
                    bundle=bundle,
                    optimizer=optimizer,
                    args=args,
                    metrics=val_metrics,
                    init_stats=init_stats,
                    best=best,
                    best_goal=best_goal,
                    best_q=best_q,
                    train_metrics=train_metrics,
                )
            if improved_q:
                best_q = hard_q
                save_checkpoint(
                    output_dir / f"{name}_q_best.pth",
                    epoch=epoch,
                    bundle=bundle,
                    optimizer=optimizer,
                    args=args,
                    metrics=val_metrics,
                    init_stats=init_stats,
                    best=best,
                    best_goal=best_goal,
                    best_q=best_q,
                    train_metrics=train_metrics,
                )
            if improved_goal:
                best_goal = score
                save_checkpoint(
                    output_dir / f"{name}_goal_best.pth",
                    epoch=epoch,
                    bundle=bundle,
                    optimizer=optimizer,
                    args=args,
                    metrics=val_metrics,
                    init_stats=init_stats,
                    best=best,
                    best_goal=best_goal,
                    best_q=best_q,
                    train_metrics=train_metrics,
                )
        if epoch % int(args.latest_every) == 0 or epoch == int(args.epochs):
            save_checkpoint(
                output_dir / f"{name}_latest.pth",
                epoch=epoch,
                bundle=bundle,
                optimizer=optimizer,
                args=args,
                metrics=latest,
                init_stats=init_stats,
                best=best,
                best_goal=best_goal,
                best_q=best_q,
                train_metrics=train_metrics,
            )


def smoke_soft_batch_code_usage_entropy(
    bundle: VQBundle,
    out: dict,
    args: argparse.Namespace,
) -> dict[str, float]:
    """CPU/GPU shape-smoke for the differentiable, chunked usage objective.

    This deliberately runs even with the production lambda left at its default
    zero: it tests the opt-in branch without changing the normal smoke's model
    loss.  A 64-token cap makes K=4096 inexpensive on CPU.
    """

    quantizer = bundle.codec.quantizer
    with torch.enable_grad():
        z = out["z"].detach().clone().requires_grad_(True)
        available = int(quantizer.flatten_tokens(z).shape[0])
        tokens, positions = sample_soft_usage_tokens(
            quantizer,
            z,
            max_tokens=min(64, available),
        )
        terms: dict[int, dict[str, torch.Tensor]] = {}
        for rate in args._rates:
            terms[int(rate)] = soft_batch_code_usage_entropy(
                quantizer,
                tokens,
                positions,
                int(rate),
                temperature=float(args.soft_usage_temperature),
                query_chunk_size=min(int(args.query_chunk_size), int(tokens.shape[0])),
            )
        objective = torch.stack([terms[int(rate)]["loss"] for rate in args._rates]).mean()
        if not objective.requires_grad:
            raise AssertionError("soft batch code-usage entropy lost its differentiable path")
        quantizer.zero_grad(set_to_none=True)
        objective.backward()
        if z.grad is None or not bool(torch.isfinite(z.grad).all()):
            raise AssertionError("soft batch code-usage entropy produced no finite z2 gradient")
        gradient_l1 = float(z.grad.detach().abs().sum())
        quantizer.zero_grad(set_to_none=True)
    return {
        "loss": float(objective.detach()),
        "gradient_l1": gradient_l1,
        "sampled_tokens": float(tokens.shape[0]),
        **{f"entropy_k{rate}": float(value["entropy"].detach()) for rate, value in terms.items()},
        **{f"ratio_k{rate}": float(value["entropy_ratio"].detach()) for rate, value in terms.items()},
    }


@torch.no_grad()
def smoke(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    source = load_source(args, device)
    resolve_embedding_dim(args, source)
    bundle = build_bundle(args, source, device)
    imgs = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    out = forward_bundle(bundle, imgs, args, train=False)
    soft_usage_smoke = smoke_soft_batch_code_usage_entropy(bundle, out, args)
    e2_input = make_source_e2_input(imgs, out["layer1"]["x1"])
    if not torch.equal(e2_input[:, :3], imgs) or not torch.equal(e2_input[:, 3:], out["layer1"]["x1"]):
        raise AssertionError("source E2 runtime contract must be concat([img,x1])")
    first = args._rates[0]
    high = args._rates[-1]
    if str(args.channel_codebook_mode) == "grouped":
        expected_channels = torch.arange(int(args.latent_c), device=device).view(1, -1)
        for rate in args._rates:
            indices = out["branches"][rate]["indices"]
            if not torch.equal(
                indices % int(args.latent_c),
                expected_channels.expand_as(indices),
            ):
                raise AssertionError(f"grouped K={rate} selected a row owned by another channel")
    max_identity_error = float((out["branches"][first]["final"] - out["layer1"]["x1"]).abs().max())
    token_rms = out["z"].float().square().mean(dim=1).sqrt()
    if str(args.image_vq_stabilizer) == "rmsnorm":
        nonzero = out["z"].float().abs().sum(dim=1) > 0.0
        if not bool(nonzero.any()):
            raise AssertionError("RMS-normalized image-VQ smoke unexpectedly produced only zero tokens")
        rms_error = float((token_rms[nonzero] - 1.0).abs().max())
        if rms_error > 2e-4:
            raise AssertionError(f"image-VQ RMS stabilizer error={rms_error:.3g}")
    else:
        rms_error = 0.0
    print(
        f"[nested-vq smoke] arch={args.arch} family={args.vq_family} "
        f"z1={tuple(out['layer1']['z1'].shape)} z={tuple(out['z'].shape)} "
        f"idx_k{first}={tuple(out['branches'][first]['indices'].shape)} "
        f"q_k{high}={tuple(out['branches'][high]['q_hard'].shape)} "
        f"q2_hat={tuple(out['q_pred_hard'].shape)} x2_hat={tuple(out['pred']['final'].shape)} "
        f"identity_error={max_identity_error:.3g} e2_input_order={args.e2_input_order}=PASS "
        f"channel_codebook_mode={args.channel_codebook_mode}=PASS "
        f"oracle_only={bool(args.oracle_only)}=PASS "
        f"receiver_stack={args.receiver_stack}=PASS "
        f"image_vq_stabilizer={args.image_vq_stabilizer}=PASS token_rms="
        f"{float(token_rms.mean()):.4g} rms_error={rms_error:.3g} "
        f"soft_usage_entropy={soft_usage_smoke}=PASS "
        f"receiver_only=PASS",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--arch", choices=["cnn", "swin"], required=True)
    parser.add_argument(
        "--layer2-arch",
        choices=["match", "cnn", "swin", "residual-cnn"],
        default="match",
        help="Layer2 codec source, independent of the Layer1 --arch; match preserves old runs.",
    )
    parser.add_argument("--vq-family", choices=["image-vq", "channel-vq"], required=True)
    parser.add_argument(
        "--channel-codebook-mode",
        choices=["global", "grouped"],
        default="global",
        help="For channel-vq, use one global prefix or channel-restricted round-major groups.",
    )
    parser.add_argument(
        "--allow-single-candidate-grouped-diagnostic",
        action="store_true",
        help=(
            "Allow grouped K=C only as a non-promotable zero-index-bit structural diagnostic; "
            "formal grouped sweeps require every K>C."
        ),
    )
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--layer2-source-checkpoint", default="")
    parser.add_argument("--latent-c", type=int, default=256)
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=0,
        help=(
            "VQ code vector dimension D, independent of Layer2 C/H/W. "
            "Zero selects the family-native dimension: C for image-vq and H*W for channel-vq."
        ),
    )
    parser.add_argument(
        "--image-vq-stabilizer",
        choices=["none", "rmsnorm"],
        default="none",
        help=(
            "Opt-in high-D image-VQ stabilizer. rmsnorm fixes each image-token RMS and "
            "uses an identity-initialized D->D projection pair when D=C; default none "
            "preserves the existing topology."
        ),
    )
    parser.add_argument(
        "--image-vq-stabilize-epochs",
        type=int,
        default=0,
        help=(
            "With --image-vq-stabilizer rmsnorm, temporarily freeze E2/analysis/pre-VQ "
            "projection and codebook for these initial full-hard training epochs."
        ),
    )
    parser.add_argument(
        "--image-vq-rms-eps",
        type=float,
        default=1e-6,
        help="Positive numerical epsilon for --image-vq-stabilizer rmsnorm.",
    )
    parser.add_argument("--rates", default="256,1024,4096")
    parser.add_argument("--rate-weights", default="1,1,1")
    parser.add_argument("--adapter-hidden", type=int, default=320)
    parser.add_argument(
        "--combiner",
        choices=["residual", "enhanced", "source", "additive"],
        default="residual",
        help="additive consumes signed u2 and is reserved for the built-in residual-cnn Layer2.",
    )
    parser.add_argument("--enhanced-combiner-width", type=int, default=48)
    parser.add_argument("--enhanced-combiner-blocks", type=int, default=4)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--freeze-codebook", action="store_true")
    parser.add_argument("--freeze-source-d2", action="store_true")
    parser.add_argument("--freeze-combiner", action="store_true")
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Train only the sender oracle path; freeze the predictor and set every receiver loss weight to zero.",
    )
    parser.add_argument("--receiver-only", action="store_true")
    parser.add_argument(
        "--receiver-phase",
        choices=["none", "q-pretrain", "decoder-warmup", "joint"],
        default="none",
        help="Explicit receiver-only phase; non-none phases require an independent receiver stack.",
    )
    parser.add_argument(
        "--receiver-stack",
        choices=["shared", "independent"],
        default="shared",
        help="Use the sender decoder or an isolated deepcopy of synthesis+D2+combiner.",
    )
    parser.add_argument("--adapter-init", choices=["pca", "random"], default="pca")
    parser.add_argument("--pca-init-batches", type=int, default=3)
    parser.add_argument(
        "--codebook-init",
        choices=["random", "kmeans", "channel-balanced"],
        default="random",
    )
    parser.add_argument("--codebook-init-batches", type=int, default=8)
    parser.add_argument("--codebook-init-max-samples", type=int, default=65536)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--beta-commit", type=float, default=0.25)
    parser.add_argument(
        "--bound-latent",
        action="store_true",
        help="Apply tanh to sender VQ vectors; useful for fresh Layer2 codecs.",
    )
    parser.add_argument(
        "--continuous-recon-warmup-epochs",
        type=int,
        default=0,
        help="Training-only epochs decoded from continuous z before gradual hard-VQ transition.",
    )
    parser.add_argument(
        "--continuous-recon-transition-epochs",
        type=int,
        default=0,
        help="Training-only linear transition length from continuous z to straight-through hard q.",
    )
    parser.add_argument("--lambda-vq", type=float, default=0.02)
    parser.add_argument("--vq-warmup-epochs", type=int, default=2)
    parser.add_argument("--vq-ramp-epochs", type=int, default=18)
    parser.add_argument(
        "--lambda-soft-usage-entropy",
        type=float,
        default=0.0,
        help=(
            "Training-only differentiable soft batch code-usage entropy weight. "
            "Zero disables it exactly (default); positive values add the normalized "
            "log(K)-H batch-marginal deficit for every nested K."
        ),
    )
    parser.add_argument(
        "--soft-usage-temperature",
        type=float,
        default=1.0,
        help="Positive softmax temperature for --lambda-soft-usage-entropy.",
    )
    parser.add_argument(
        "--soft-usage-sample-tokens",
        type=int,
        default=1024,
        help=(
            "Maximum pre-VQ tokens sampled once per training batch for the soft usage loss; "
            "0 uses all tokens. Query chunks remain bounded by --query-chunk-size."
        ),
    )
    parser.add_argument("--lambda-monotonic", type=float, default=0.1)
    parser.add_argument("--hard-example-power", type=float, default=0.0)
    parser.add_argument("--hard-example-min-weight", type=float, default=0.25)
    parser.add_argument("--hard-example-max-weight", type=float, default=4.0)
    parser.add_argument("--monotonic-margin", type=float, default=1e-5)
    parser.add_argument("--condition-mode", choices=["z1", "x1", "z1_x1"], default="z1_x1")
    parser.add_argument("--predictor-hidden", type=int, default=128)
    parser.add_argument("--predictor-blocks", type=int, default=6)
    parser.add_argument("--predictor-attention-every", type=int, default=2)
    parser.add_argument("--predictor-heads", type=int, default=4)
    parser.add_argument("--receiver-warmup-epochs", type=int, default=10)
    parser.add_argument("--lambda-predict-q", type=float, default=0.02)
    parser.add_argument("--lambda-predictability", type=float, default=0.002)
    parser.add_argument(
        "--lambda-predictable-z",
        type=float,
        default=0.0,
        help="Align sender pre-quantization z2 to receiver q prediction so indices become condition-sufficient.",
    )
    parser.add_argument("--lambda-predict-final", type=float, default=1.0)
    parser.add_argument("--lambda-zero-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-shuffle-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-condition-anchor", type=float, default=0.0)
    parser.add_argument("--usage-decay", type=float, default=0.99)
    parser.add_argument("--dead-refresh-start", type=int, default=10)
    parser.add_argument("--dead-refresh-every", type=int, default=5)
    parser.add_argument("--dead-threshold", type=float, default=0.01)
    parser.add_argument("--dead-patience", type=int, default=100)
    parser.add_argument("--dead-max-refresh", type=int, default=256)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--codebook-chunk-size", type=int, default=2048)
    parser.add_argument("--min-ablation-drop", type=float, default=0.1)
    parser.add_argument("--min-receiver-delta", type=float, default=0.0)
    parser.add_argument("--min-condition-shuffle-drop", type=float, default=0.1)
    parser.add_argument("--min-paired-gain-db", type=float, default=0.01)
    parser.add_argument("--min-paired-strict", type=float, default=0.55)
    parser.add_argument("--min-index-change", type=float, default=0.01)
    parser.add_argument("--min-ppl-ratio", type=float, default=0.01)
    parser.add_argument("--max-top1", type=float, default=0.25)
    parser.add_argument("--min-local-ppl", type=float, default=1.1)
    parser.add_argument("--max-local-top1", type=float, default=0.95)
    parser.add_argument("--min-channel-variation", type=float, default=0.01)
    parser.add_argument("--min-variant-channel-frac", type=float, default=0.1)
    parser.add_argument("--codebook-l2-samples", type=int, default=512)
    parser.add_argument("--codebook-rank-samples", type=int, default=2048)
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument("--save-dir", default="MY-V2/jscc-f/explore-2/checkpoints-vq")
    parser.add_argument("--version", default="nested-vq-v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--predictor-lr", type=float, default=1e-4)
    parser.add_argument("--receiver-decoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--latest-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--reset-predictor-on-resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-shapes", action="store_true")
    parser.add_argument("--eval-init-only", action="store_true")
    parser.add_argument(
        "--allow-eval-rate-override",
        action="store_true",
        help="With --eval-init-only, evaluate alternate nested prefixes while preserving checkpoint Kmax.",
    )
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    args = parser.parse_args()
    # Checkpoint/deployment contract.  This is deliberately not user-settable:
    # older [x1,img] exploratory checkpoints must fail resume validation rather
    # than silently entering the canonical [img,x1] result pool.
    args.e2_input_order = "img_x1"
    args._rates = parse_int_list(args.rates)
    args._rate_weights = parse_float_list(args.rate_weights, len(args._rates))
    if not 1 <= int(args.latent_c) <= 320:
        raise ValueError("--latent-c must be in [1,320] for the exact-width source adapter")
    if int(args.embedding_dim) < 0:
        raise ValueError("--embedding-dim must be non-negative; zero means family-native")
    if int(args.image_vq_stabilize_epochs) < 0:
        raise ValueError("--image-vq-stabilize-epochs must be non-negative")
    if float(args.image_vq_rms_eps) <= 0.0:
        raise ValueError("--image-vq-rms-eps must be positive")
    if str(args.image_vq_stabilizer) != "none" and str(args.vq_family) != "image-vq":
        raise ValueError("--image-vq-stabilizer requires --vq-family image-vq")
    if int(args.image_vq_stabilize_epochs) > 0 and str(args.image_vq_stabilizer) == "none":
        raise ValueError(
            "--image-vq-stabilize-epochs requires --image-vq-stabilizer rmsnorm"
        )
    if int(args.continuous_recon_warmup_epochs) < 0 or int(args.continuous_recon_transition_epochs) < 0:
        raise ValueError("continuous reconstruction warmup/transition epochs must be non-negative")
    if float(args.lambda_soft_usage_entropy) < 0.0:
        raise ValueError("--lambda-soft-usage-entropy must be non-negative")
    if float(args.soft_usage_temperature) <= 0.0:
        raise ValueError("--soft-usage-temperature must be positive")
    if int(args.soft_usage_sample_tokens) < 0:
        raise ValueError("--soft-usage-sample-tokens must be non-negative; 0 means all tokens")
    if int(args.enhanced_combiner_width) < 8 or int(args.enhanced_combiner_width) % 8 != 0:
        raise ValueError("--enhanced-combiner-width must be >=8 and divisible by 8")
    if int(args.enhanced_combiner_blocks) < 1:
        raise ValueError("--enhanced-combiner-blocks must be positive")
    if int(args.dead_max_refresh) < 1:
        raise ValueError("--dead-max-refresh must be positive")
    if float(args.min_paired_gain_db) < 0.0:
        raise ValueError("--min-paired-gain-db must be non-negative")
    if not 0.0 <= float(args.min_paired_strict) <= 1.0:
        raise ValueError("--min-paired-strict must lie in [0,1]")
    if not 0.0 <= float(args.min_index_change) <= 1.0:
        raise ValueError("--min-index-change must lie in [0,1]")
    if float(args.min_condition_shuffle_drop) < 0.0:
        raise ValueError("--min-condition-shuffle-drop must be non-negative")
    if float(args.min_local_ppl) < 1.0:
        raise ValueError("--min-local-ppl must be at least 1")
    if not 0.0 <= float(args.max_local_top1) <= 1.0:
        raise ValueError("--max-local-top1 must lie in [0,1]")
    if not 0.0 <= float(args.min_channel_variation) <= 1.0:
        raise ValueError("--min-channel-variation must lie in [0,1]")
    if not 0.0 <= float(args.min_variant_channel_frac) <= 1.0:
        raise ValueError("--min-variant-channel-frac must lie in [0,1]")
    if str(args.codebook_init) == "channel-balanced" and str(args.vq_family) != "channel-vq":
        raise ValueError("--codebook-init channel-balanced requires --vq-family channel-vq")
    if str(args.layer2_arch) == "residual-cnn" and str(args.combiner) == "source":
        raise ValueError("builtin residual-cnn Layer2 requires residual, enhanced, or additive combiner")
    if str(args.combiner) == "additive" and str(args.layer2_arch) != "residual-cnn":
        raise ValueError("--combiner additive requires --layer2-arch residual-cnn")
    if str(args.channel_codebook_mode) == "grouped":
        if str(args.vq_family) != "channel-vq":
            raise ValueError("--channel-codebook-mode grouped requires --vq-family channel-vq")
        invalid = [rate for rate in args._rates if int(rate) % int(args.latent_c) != 0]
        if invalid:
            raise ValueError(
                "grouped channel-vq requires every K divisible by C; "
                f"C={int(args.latent_c)} invalid_rates={invalid}"
            )
        single_candidate = [rate for rate in args._rates if int(rate) == int(args.latent_c)]
        if single_candidate and not bool(args.allow_single_candidate_grouped_diagnostic):
            raise ValueError(
                "formal grouped channel-vq requires every K>C so every channel has at least "
                "two source-dependent candidates; K=C carries zero index bits.  Pass "
                "--allow-single-candidate-grouped-diagnostic only for a non-promotable "
                f"structural diagnostic; found {single_candidate}."
            )
        if str(args.codebook_init) != "channel-balanced" and not args.resume:
            raise ValueError(
                "new grouped channel-vq runs require --codebook-init channel-balanced "
                "so row round*C+channel owns a real map from that channel"
            )
    if bool(args.receiver_only) and not args.resume:
        raise ValueError("--receiver-only requires --resume from a trained oracle checkpoint")
    if bool(args.oracle_only) and bool(args.receiver_only):
        raise ValueError("--oracle-only and --receiver-only are mutually exclusive phases")
    phase = receiver_phase(args)
    if phase != "none":
        if not bool(args.receiver_only):
            raise ValueError("a non-none --receiver-phase requires --receiver-only")
        if not independent_receiver(args):
            raise ValueError("explicit receiver phases require --receiver-stack independent")
    if bool(args.reset_predictor_on_resume) and not args.resume:
        raise ValueError("--reset-predictor-on-resume requires --resume")
    if bool(args.allow_eval_rate_override) and (not bool(args.eval_init_only) or not args.resume):
        raise ValueError("--allow-eval-rate-override requires --eval-init-only and --resume")
    for name in (
        "lambda_zero_anchor",
        "lambda_shuffle_anchor",
        "lambda_condition_anchor",
        "lambda_predictable_z",
        "predictor_lr",
        "receiver_decoder_lr",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if args.smoke_shapes:
        smoke(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
