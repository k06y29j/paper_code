#!/usr/bin/env python3
"""Paper-faithful CVQ tokenizer sweep with a variable-resolution Layer-2 E2.

This is intentionally separate from ``explore-2/train_layer2_vq_nested.py``.
That entrypoint always consumes the source checkpoint's 16x16 E2 latent;
therefore changing its code-vector dimension does *not* change E2's
downsampling ratio.  Here a fresh Layer-2 encoder has exactly ``n`` stride-2
stages, so a 256x256 crop produces 256 / 2**n spatial maps (n=4 -> 16x16,
n=3 -> 32x32).  The quantizer is the paper's global channel-wise VQ:

    z in R[B,C,H,W] -> one code index for each complete HxW channel map.

The source D2 consumes 320x16x16.  A learned spatial bridge maps the selected
CVQ resolution to this fixed interface; it is part of Layer 2 and is trained
with E2, codebook, D2 and combiner.  Layer 1 stays frozen.

Nested channel dropout follows arXiv:2605.26089v2: during tokenizer training
we retain only the first randomly sampled number of channels.  Validation is
always full-token reconstruction, while partial-token validation is reported
separately.  This establishes an ordered coarse-to-fine channel sequence for
the later CAR receiver without leaking sender-side inputs to that receiver.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
EXPLORE2_DIR = JSCCF_DIR / "explore-2"
for path in (THIS_DIR, EXPLORE2_DIR, JSCCF_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cvq_predictability import (
    Z1ConditionalChannelPrior,
    masked_channel_marginal_entropy_deficit,
    masked_hard_index_nll,
    masked_soft_rate_nll,
    soft_channel_assignments,
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nested = load_module("explore4_nested_support", EXPLORE2_DIR / "train_layer2_vq_nested.py")
vq = load_module("explore4_vq_support", EXPLORE2_DIR / "vq_modules.py")
base = nested.base


def parse_int_list(value: str) -> list[int]:
    result = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not result or result != sorted(set(result)) or any(item < 2 for item in result):
        raise ValueError(f"--rates must be increasing unique integers >=2, got {value!r}")
    return result


def parse_float_list(value: str, count: int) -> list[float]:
    values = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(values) == 1:
        values *= int(count)
    if len(values) != int(count) or any(item < 0 for item in values) or sum(values) <= 0:
        raise ValueError(f"invalid --rate-weights {value!r}")
    total = sum(values)
    return [item / total for item in values]


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int) -> None:
        super().__init__()
        groups = min(16, int(out_ch))
        while int(out_ch) % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4 if stride == 2 else 3, stride=stride, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialLayer2Encoder(nn.Module):
    """Fresh E2 with exactly ``downsamples`` stride-2 operations."""

    def __init__(self, downsamples: int) -> None:
        super().__init__()
        if not 2 <= int(downsamples) <= 5:
            raise ValueError("--e2-downsamples must be in [2,5]")
        widths = [64, 128, 192, 256, 320]
        blocks: list[nn.Module] = []
        in_ch = 6
        for stage in range(int(downsamples)):
            out_ch = widths[stage]
            blocks.append(ConvNormAct(in_ch, out_ch, stride=2))
            in_ch = out_ch
        blocks += [
            nn.Conv2d(in_ch, 320, 3, padding=1),
            nn.GroupNorm(16, 320),
            nn.SiLU(),
            nn.Conv2d(320, 320, 3, padding=1),
        ]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialBridge(nn.Module):
    """Map a variable CVQ map to the q2 portion of the fixed D2 contract."""

    def __init__(self, latent_c: int, input_size: int, output_channels: int) -> None:
        super().__init__()
        if input_size not in {8, 16, 32, 64}:
            raise ValueError(f"unsupported CVQ spatial size {input_size}; expected 8/16/32/64")
        self.input_size = int(input_size)
        self.pre = nn.Sequential(
            nn.Conv2d(int(latent_c), 320, 3, padding=1),
            nn.GroupNorm(16, 320),
            nn.SiLU(),
        )
        blocks: list[nn.Module] = []
        current = int(input_size)
        while current > 16:
            blocks += [nn.Conv2d(320, 320, 4, stride=2, padding=1), nn.GroupNorm(16, 320), nn.SiLU()]
            current //= 2
        while current < 16:
            blocks += [nn.ConvTranspose2d(320, 320, 4, stride=2, padding=1), nn.GroupNorm(16, 320), nn.SiLU()]
            current *= 2
        self.resample = nn.Sequential(*blocks)
        self.output_channels = int(output_channels)
        self.out = nn.Conv2d(320, self.output_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.resample(self.pre(x)))


class CVQSpatialCodec(nn.Module):
    def __init__(self, args: argparse.Namespace, source=None) -> None:
        super().__init__()
        self.spatial_size = 256 // (2 ** int(args.e2_downsamples))
        self.native_dim = self.spatial_size * self.spatial_size
        self.embedding_dim = int(args.embedding_dim) or self.native_dim
        self.source_e2_init = bool(getattr(args, "source_e2_init", False))
        self.channel_rms_normalize = bool(getattr(args, "channel_rms_normalize", False))
        if self.source_e2_init:
            if source is None or int(args.e2_downsamples) != 4 or self.spatial_size != 16:
                raise ValueError("--source-e2-init requires the matched 16x16 source Layer2 E2")
            self.e2 = copy.deepcopy(source.e2)
        else:
            self.e2 = SpatialLayer2Encoder(int(args.e2_downsamples))
        self.analysis = nn.Conv2d(320, int(args.latent_c), 1)
        if self.source_e2_init:
            # Preserve the first C source channels at step zero.  This lets
            # the quantizer see a trained residual representation rather than
            # a random projection of a near-zero Swin residual.
            with torch.no_grad():
                self.analysis.weight.zero_()
                self.analysis.bias.zero_()
                diagonal = min(320, int(args.latent_c))
                rows = torch.arange(diagonal)
                self.analysis.weight[rows, rows, 0, 0] = 1.0
        if self.embedding_dim == self.native_dim:
            self.token_encoder = nn.Identity()
            self.token_decoder = nn.Identity()
            q_h, q_w = self.spatial_size, self.spatial_size
        else:
            self.token_encoder = nested.ChannelEmbeddingEncoder(
                self.spatial_size, self.spatial_size, self.embedding_dim
            )
            self.token_decoder = nested.ChannelEmbeddingDecoder(
                self.embedding_dim, self.spatial_size, self.spatial_size
            )
            q_h, q_w = 1, self.embedding_dim
        self.quantizer = vq.build_nested_vq(
            "channel-vq",
            max(args.rates_list),
            channels=int(args.latent_c),
            h=q_h,
            w=q_w,
            beta=float(args.beta_commit),
            query_chunk_size=int(args.query_chunk_size),
            codebook_chunk_size=int(args.codebook_chunk_size),
            usage_decay=float(args.usage_decay),
            channel_codebook_mode="global",
        )
        bridge_channels = int(args.d2_input_channels) - int(args.d2_z1_channels)
        if self.source_e2_init and self.embedding_dim == self.native_dim and int(args.latent_c) == bridge_channels:
            # At matched 16x16 resolution q2 already has the D2 q-branch
            # layout; retaining it exactly preserves the pretrained E2
            # channel geometry during continuous and VQ warmup.
            self.bridge = nn.Identity()
        else:
            self.bridge = SpatialBridge(int(args.latent_c), self.spatial_size, bridge_channels)

    def encode(self, img: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        z320 = self.e2(nested.make_source_e2_input(img, x1))
        # The matched Swin Stage2 E2 exposes an auxiliary return alongside
        # its latent, while the CNN implementation returns the latent alone.
        # CVQ always consumes the canonical first latent output.
        if isinstance(z320, (tuple, list)):
            z320 = z320[0]
        if tuple(z320.shape[-2:]) != (self.spatial_size, self.spatial_size):
            raise RuntimeError(
                f"E2 downsample contract expected {self.spatial_size}x{self.spatial_size}, got {tuple(z320.shape[-2:])}"
            )
        maps = self.analysis(z320)
        if self.channel_rms_normalize:
            # Global CVQ compares complete channel maps against one shared
            # Euclidean codebook.  Pretrained feature channels can have very
            # different energies, which otherwise makes lookup select only a
            # few low-norm centers.  Unit-RMS maps give the codebook a common
            # channel geometry while preserving each map's spatial direction.
            maps = maps / maps.float().square().mean(dim=(-2, -1), keepdim=True).add(1e-6).sqrt().to(dtype=maps.dtype)
        return self.token_encoder(maps)

    def decode_latent(self, q: torch.Tensor) -> torch.Tensor:
        return self.bridge(self.token_decoder(q))


class StrictResidualDeltaCombiner(nn.Module):
    """Make q2 the only route from x1 to an enhancement residual.

    ``D2(z1, 0)`` is a z1-conditioned reference, so its subtraction removes
    the legal Layer-1 condition from the enhancement signal.  Consequently
    q2=0 is identically x1, not merely encouraged to resemble x1 by a loss.
    """

    def __init__(self, initial_gain: float = 0.01) -> None:
        super().__init__()
        initial_gain = float(initial_gain)
        if not 0.0 < initial_gain < 1.0:
            raise ValueError("initial residual gain must lie in (0,1)")
        self.residual_gain_logit = nn.Parameter(torch.tensor(math.log(initial_gain / (1.0 - initial_gain))))

    def forward(self, x1: torch.Tensor, u2: torch.Tensor, u2_zero: torch.Tensor) -> torch.Tensor:
        gain = self.residual_gain_logit.sigmoid()
        return (x1 + gain * (u2 - u2_zero)).clamp(0.0, 1.0)


class SourceCombinerResidualDelta(nn.Module):
    """Residualize the pretrained Swin output combiner without bypassing q2.

    The Swin Layer-2 checkpoint was trained as ``H(x1, clamp(D2(z2)))``;
    its D2 output is therefore not an additive RGB residual.  Taking the
    difference in H's *output* domain preserves that semantic contract while
    making the receiver fallback exact: q2=0 implies x2=x1.
    """

    def __init__(self, source_combiner: nn.Module) -> None:
        super().__init__()
        self.source_combiner = copy.deepcopy(source_combiner)

    def forward(self, x1: torch.Tensor, u2: torch.Tensor, u2_zero: torch.Tensor) -> torch.Tensor:
        enhanced = self.source_combiner(x1, u2.clamp(0.0, 1.0))
        reference = self.source_combiner(x1, u2_zero.clamp(0.0, 1.0))
        return (x1 + enhanced - reference).clamp(0.0, 1.0)


class SenderCVQ(nn.Module):
    def __init__(self, source, args: argparse.Namespace) -> None:
        super().__init__()
        self.source = source
        self.codec = CVQSpatialCodec(args, source)
        self.use_swin_source_combiner = bool(getattr(args, "swin_source_combiner_residual", False))
        if self.use_swin_source_combiner and str(source.arch) != "swin":
            raise ValueError("--swin-source-combiner-residual is only valid with --arch swin")
        if self.use_swin_source_combiner:
            # Keep a distinct trainable student and a completely frozen
            # teacher.  Sharing source modules would silently update the
            # teacher through the student optimizer and invalidate distillation.
            self.d2 = copy.deepcopy(source.d2)
            self.combiner = SourceCombinerResidualDelta(source.combiner)
            self.has_source_teacher = True
            source.e2.requires_grad_(False); source.d2.requires_grad_(False); source.combiner.requires_grad_(False)
            source.e2.eval(); source.d2.eval(); source.combiner.eval()
            if bool(getattr(args, "freeze_source_combiner", False)):
                self.combiner.source_combiner.requires_grad_(False)
        else:
            self.d2 = source.d2
            self.combiner = StrictResidualDeltaCombiner(
                initial_gain=float(getattr(args, "initial_residual_gain", 0.01))
            )
            self.has_source_teacher = False
        self.d2_input_channels = int(args.d2_input_channels)
        self.source_d2_input_channels = int(args.source_d2_input_channels)
        self.d2_z1_channels = int(args.d2_z1_channels)
        if not 0 <= self.d2_z1_channels < self.d2_input_channels:
            raise ValueError("--d2-z1-channels must be nonnegative and smaller than --d2-input-channels")
        # The experimental Layer-2 interface is exactly [z1(16), q2(240)].
        # The inherited source D2 was trained with 320 channels, so this
        # trainable 1x1 front end is considered part of our new D2.  It lets
        # us retain its upsampling trunk without pretending that q2 has 304
        # channel tokens.
        self.d2_frontend = nn.Conv2d(self.d2_input_channels, self.source_d2_input_channels, 1)
        with torch.no_grad():
            self.d2_frontend.weight.zero_()
            self.d2_frontend.bias.zero_()
            # The combiner's 0.01 residual gate keeps this identity front end
            # stable at startup while retaining a nonzero E2/codebook gradient.
            diagonal = min(self.d2_input_channels, self.source_d2_input_channels)
            identity = torch.arange(diagonal)
            self.d2_frontend.weight[identity, identity, 0, 0] = 1.0
            if bool(getattr(args, "source_e2_init", False)):
                # Input is [z1(16), q2(240)].  Map q2 channel j back to
                # source-D2 channel j, preserving the pretrained E2/D2 path
                # as far as the mandated 16+240 contract permits.
                self.d2_frontend.weight.zero_()
                q_channels = min(
                    int(args.d2_input_channels) - int(args.d2_z1_channels),
                    self.source_d2_input_channels,
                )
                source_rows = torch.arange(q_channels)
                input_cols = source_rows + int(args.d2_z1_channels)
                self.d2_frontend.weight[source_rows, input_cols, 0, 0] = 1.0
        if bool(getattr(args, "freeze_student_decoder", False)):
            self.d2.requires_grad_(False)
            self.d2_frontend.requires_grad_(False)
            self.codec.bridge.requires_grad_(False)
            self.codec.token_decoder.requires_grad_(False)
            self.combiner.requires_grad_(False)
        if bool(getattr(args, "conditional_prior", False)):
            self.predictability_prior: Z1ConditionalChannelPrior | None = Z1ConditionalChannelPrior(
                int(args.d2_z1_channels), int(args.latent_c), max(args.rates_list),
                hidden=int(args.prior_hidden), layers=int(args.prior_layers), heads=int(args.prior_heads),
                dropout=float(args.prior_dropout),
            )
        else:
            self.predictability_prior = None

    def decode(self, q: torch.Tensor, z1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Decode ``[z1[:,:16], q2]``; only q2 receives token dropout.

        The new Layer-2 front end is exactly [z1(16),q2(240)].  We subtract
        D2(z1,0) before adding the result to x1, so D2 cannot use z1 or a
        learned combiner bypass as a substitute for transmitted q2.
        """
        return self.decode_outputs(q, z1, x1)["final"]

    def decode_outputs(self, q: torch.Tensor, z1: torch.Tensor, x1: torch.Tensor) -> dict[str, torch.Tensor]:
        u2 = self._decode_u2(q, z1)
        u2_zero = self._decode_u2(torch.zeros_like(q), z1)
        return {"u2": u2, "u2_zero": u2_zero, "final": self.combiner(x1, u2, u2_zero)}

    def set_source_teacher_eval(self) -> None:
        if self.has_source_teacher:
            self.source.e2.eval(); self.source.d2.eval(); self.source.combiner.eval()

    @torch.no_grad()
    def source_teacher_outputs(self, img: torch.Tensor, x1: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Full original Swin Layer2, kept outside sender state/checkpoints."""

        if not self.has_source_teacher:
            return None
        z2 = self.source.e2(nested.make_source_e2_input(img, x1))
        if isinstance(z2, (tuple, list)):
            z2 = z2[0]
        u2 = self.source.d2(z2).clamp(0.0, 1.0)
        return {"z2": z2, "u2": u2, "final": self.source.combiner(x1, u2)}

    def _decode_u2(self, q: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        q_part = self.codec.decode_latent(q)
        if int(z1.shape[1]) < self.d2_z1_channels:
            raise RuntimeError(f"z1 has {z1.shape[1]} channels but needs {self.d2_z1_channels}")
        z1_part = z1[:, :self.d2_z1_channels]
        if tuple(z1_part.shape[-2:]) != tuple(q_part.shape[-2:]):
            raise RuntimeError(f"z1/D2 spatial mismatch {tuple(z1_part.shape)} vs {tuple(q_part.shape)}")
        d2_input = torch.cat([z1_part, q_part], dim=1)
        if int(d2_input.shape[1]) != self.d2_input_channels:
            raise RuntimeError(f"D2 input must have {self.d2_input_channels} channels, got {d2_input.shape[1]}")
        return self.d2(self.d2_frontend(d2_input))


def nested_channel_mask(channels: int, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    keep = torch.randint(1, int(channels) + 1, (int(batch),), device=device)
    channel_ids = torch.arange(int(channels), device=device).view(1, -1)
    mask = (channel_ids < keep.view(-1, 1)).to(dtype=torch.float32).view(batch, channels, 1, 1)
    return mask, keep


@torch.no_grad()
def initialize_channel_codebook(loader, model: SenderCVQ, args: argparse.Namespace, device: torch.device) -> dict[str, float | str]:
    """Paper-aligned channel-balanced real-map initialization for a global CVQ table."""

    max_images = max(1, int(args.codebook_init_max_samples) // int(args.latent_c))
    samples: list[torch.Tensor] = []
    total = 0
    model.eval()
    for batch_index, (img, _label) in enumerate(loader, start=1):
        if batch_index > int(args.codebook_init_batches) or total >= max_images:
            break
        img = img.to(device, non_blocking=True)
        x1 = model.source.layer1(img)["x1"]
        z = model.codec.encode(img, x1).detach().float().cpu()
        take = min(int(z.shape[0]), max_images - total)
        samples.append(z[:take])
        total += take
    if not samples:
        raise RuntimeError("CVQ codebook initialization collected no samples")
    maps = torch.cat(samples, dim=0)
    q = model.codec.quantizer
    kmax = max(args.rates_list)
    rounds = math.ceil(kmax / int(args.latent_c))
    rows: list[torch.Tensor] = []
    # Round-major, channel-balanced selection gives every source channel a
    # real map before any channel is repeated.  It is an initializer only;
    # global CVQ lookup remains exactly the paper's shared codebook lookup.
    for round_index in range(rounds):
        for channel in range(int(args.latent_c)):
            if len(rows) >= kmax:
                break
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(args.seed) + round_index * 1009 + channel * 104729)
            image_index = int(torch.randint(int(maps.shape[0]), (1,), generator=generator))
            rows.append(maps[image_index, channel])
    centers = torch.stack(rows)
    if tuple(centers.shape[1:]) != tuple(q.embedding_shape):
        raise RuntimeError(f"initializer/codebook shape mismatch {tuple(centers.shape)} vs {tuple(q.codebook.shape)}")
    q.codebook.data[:kmax].copy_(centers.to(device=device, dtype=q.codebook.dtype))
    q.reset_usage()
    return {
        "method": "channel-balanced-real-map",
        "source_images": float(maps.shape[0]),
        "kmax": float(kmax),
        "embedding_dim": float(centers[0].numel()),
        "rounds": float(rounds),
    }


@torch.no_grad()
def initialize_swin_pca_adapter(loader, model: SenderCVQ, args: argparse.Namespace, device: torch.device) -> dict[str, float | str]:
    """Initialize 320->240->320 student maps with source-latent PCA.

    The mandated receiver contract has only 240 q2 channels, whereas the
    pretrained Swin D2 consumes all 320 source-z2 channels.  Copying the first
    240 channels discards an arbitrary 80-channel suffix and starts far away
    from the trained D2/combiner manifold.  Centered PCA gives the optimal
    rank-C linear reconstruction of real source E2 maps at initialization:

        q = A (z2 - mu),      z2_hat = A^T q + mu.

    It is a structural low-rank initialization, not a codebook-use heuristic.
    """

    if not bool(args.swin_source_pca_init) or not model.use_swin_source_combiner:
        return {"adapter_init": "not-requested"}
    if not bool(args.source_e2_init):
        raise ValueError("--swin-source-pca-init requires --source-e2-init")
    latent_c = int(args.latent_c)
    source_c = int(args.source_d2_input_channels)
    if latent_c > source_c:
        raise ValueError("PCA adapter requires latent-c <= source-d2-input-channels")
    maximum = int(args.swin_pca_max_tokens)
    samples: list[torch.Tensor] = []
    total = 0
    model.set_source_teacher_eval()
    for batch_index, (img, _label) in enumerate(loader, start=1):
        if batch_index > int(args.swin_pca_batches) or total >= maximum:
            break
        img = img.to(device, non_blocking=True)
        x1 = model.source.layer1(img)["x1"]
        z2 = model.source.e2(nested.make_source_e2_input(img, x1))
        if isinstance(z2, (tuple, list)):
            z2 = z2[0]
        tokens = z2.detach().float().permute(0, 2, 3, 1).reshape(-1, source_c)
        remaining = maximum - total
        if int(tokens.shape[0]) > remaining:
            choice = torch.randperm(int(tokens.shape[0]), device=tokens.device)[:remaining]
            tokens = tokens[choice]
        samples.append(tokens.cpu())
        total += int(tokens.shape[0])
    if total < source_c:
        raise RuntimeError(f"PCA adapter collected only {total} source tokens for {source_c} channels")
    matrix = torch.cat(samples, dim=0)
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    covariance = centered.t().matmul(centered) / float(max(1, int(centered.shape[0]) - 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    # Rows are orthonormal analysis directions ordered by descending variance.
    analysis = eigenvectors[:, -latent_c:].t().contiguous()
    explained = eigenvalues[-latent_c:].sum() / eigenvalues.clamp_min(0).sum().clamp_min(1e-12)
    with torch.no_grad():
        model.codec.analysis.weight.zero_(); model.codec.analysis.bias.zero_()
        model.codec.analysis.weight[:latent_c, :source_c, 0, 0].copy_(analysis.to(device=device, dtype=model.codec.analysis.weight.dtype))
        model.codec.analysis.bias[:latent_c].copy_((-analysis @ mean).to(device=device, dtype=model.codec.analysis.bias.dtype))
        model.d2_frontend.weight.zero_(); model.d2_frontend.bias.zero_()
        q_start = int(args.d2_z1_channels)
        model.d2_frontend.weight[:source_c, q_start:q_start + latent_c, 0, 0].copy_(analysis.t().to(device=device, dtype=model.d2_frontend.weight.dtype))
        model.d2_frontend.bias[:source_c].copy_(mean.to(device=device, dtype=model.d2_frontend.bias.dtype))
    return {
        "adapter_init": "source-z2-centered-pca",
        "adapter_pca_tokens": float(total),
        "adapter_pca_rank": float(latent_c),
        "adapter_pca_explained_variance": float(explained),
    }


def psnr_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred.float() - target.float()).square().flatten(1).mean(dim=1).clamp_min(1e-10)
    return -10.0 * torch.log10(mse)


def histogram_metrics(hist: torch.Tensor, rate: int) -> dict[str, float]:
    total = float(hist.sum())
    if total <= 0:
        return {"used": 0.0, "ppl": 0.0, "ppl_ratio": 0.0, "top1": 0.0}
    prob = hist / total
    active = prob > 0
    entropy = -(prob[active] * prob[active].log()).sum()
    return {
        "used": float(active.sum()),
        "ppl": float(entropy.exp()),
        "ppl_ratio": float(entropy.exp() / float(rate)),
        "top1": float(prob.max()),
    }


def soft_usage_entropy_loss(z: torch.Tensor, quantizer, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable batch-marginal code-usage deficit across nested K.

    CVQ should naturally retain broad code usage, but a freshly trained JSCC
    decoder can otherwise settle into the x1 bypass before its codebook has a
    reconstruction role.  This term is computed on a bounded sample of real
    channel maps and explicitly measures ``1 - H(p_batch)/log(K)``.  It is not
    an index histogram surrogate: gradients reach both E2 and codebook.
    """

    if float(args.lambda_soft_usage) <= 0.0:
        zero = z.new_zeros(())
        return zero, zero
    tokens = quantizer.flatten_tokens(z)
    maximum = min(int(args.soft_usage_samples), int(tokens.shape[0]))
    if maximum < int(tokens.shape[0]):
        selection = torch.randperm(int(tokens.shape[0]), device=z.device)[:maximum]
        tokens = tokens[selection]
    tokens_f = tokens.float()
    token_norm = tokens_f.square().sum(dim=1, keepdim=True)
    losses: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    for weight, rate in zip(args.rate_weights_list, args.rates_list):
        codebook = quantizer.codebook_at_k(int(rate)).reshape(int(rate), -1).float()
        distance = token_norm + codebook.square().sum(dim=1).unsqueeze(0) - 2.0 * tokens_f @ codebook.t()
        probabilities = F.softmax(-distance / float(args.soft_usage_temperature), dim=1)
        marginal = probabilities.mean(dim=0).clamp_min(1e-12)
        entropy = -(marginal * marginal.log()).sum()
        ratio = entropy / math.log(float(rate))
        losses.append(float(weight) * (1.0 - ratio))
        ratios.append(float(weight) * ratio)
    return torch.stack(losses).sum(), torch.stack(ratios).sum()


def active_prefix_vq_stats(
    z: torch.Tensor, q_hard: torch.Tensor, active_mask: torch.Tensor, beta: float,
) -> dict[str, torch.Tensor]:
    """Paper Appendix-B.1 VQ objective restricted to nested active channels."""

    if tuple(active_mask.shape[:2]) != tuple(z.shape[:2]):
        raise ValueError(f"active mask/token shape mismatch {tuple(active_mask.shape)} vs {tuple(z.shape)}")
    mask = active_mask.to(device=z.device, dtype=torch.float32)
    denominator = (mask.sum() * int(z.shape[-2]) * int(z.shape[-1])).clamp_min(1.0)
    codebook = ((q_hard.float() - z.detach().float()).square() * mask).sum() / denominator
    commitment = ((z.float() - q_hard.detach().float()).square() * mask).sum() / denominator
    return {
        "codebook_loss": codebook,
        "commit_loss": commitment,
        "commitment_loss": commitment,
        "vq_loss": codebook + float(beta) * commitment,
        "loss": codebook + float(beta) * commitment,
        "vq_mse": ((q_hard.detach().float() - z.detach().float()).square() * mask).sum() / denominator,
    }


@torch.no_grad()
def update_usage_ema_flat(quantizer, indices: torch.Tensor, rate: int) -> None:
    """EMA usage update for an explicitly selected set of channel indices.

    The shared quantizer API validates a rectangular [B,C] tensor.  Nested
    dropout selects a ragged active prefix per image, so padding it would add
    fake code occurrences.  This is the same EMA update as the quantizer
    method, applied to exactly the active index population.
    """

    flat = indices.detach().long().reshape(-1)
    if flat.numel() < 1 or int(flat.min()) < 0 or int(flat.max()) >= int(rate):
        raise ValueError("active CVQ usage indices are empty or out of range")
    counts = torch.bincount(flat, minlength=int(rate)).float().to(device=quantizer.ema_count.device)
    quantizer.ema_count[:int(rate)].mul_(quantizer.usage_decay).add_(counts, alpha=1.0 - quantizer.usage_decay)
    used = counts > 0
    quantizer.inactive_steps[:int(rate)].add_(1)
    quantizer.inactive_steps[:int(rate)][used] = 0


def run_epoch(loader, model: SenderCVQ, optimizer, args: argparse.Namespace, device: torch.device, *, train: bool, continuous_q: bool = False) -> dict[str, float]:
    model.source.e1.eval()
    model.source.d1.eval()
    model.train(train)
    model.source.e1.eval()
    model.source.d1.eval()
    if model.has_source_teacher:
        model.set_source_teacher_eval()
        if bool(getattr(args, "freeze_source_combiner", False)):
            model.combiner.source_combiner.eval()
    sums: dict[str, float] = {}
    count = 0
    histograms = {rate: torch.zeros(rate, dtype=torch.float64) for rate in args.rates_list}
    max_batches = int(args.max_train_batches if train else args.max_val_batches)
    # CVQ's large-K experiments rely on many channel maps contributing to one
    # codebook update (the reference uses batch 256).  Accumulation preserves
    # the exact MSE+commitment objective while matching that token statistics
    # with a memory-bounded physical batch.
    accum_steps = int(args.accum_steps) if train else 1
    pending_usage: list[torch.Tensor] = []
    pending_backward = False
    if train:
        optimizer.zero_grad(set_to_none=True)

    def optimizer_step() -> None:
        nonlocal pending_backward, pending_usage
        if not pending_backward:
            return
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if model.predictability_prior is not None:
            prior_ids = {id(parameter) for parameter in model.predictability_prior.parameters() if parameter.requires_grad}
            tokenizer_params = [parameter for parameter in trainable if id(parameter) not in prior_ids]
            if tokenizer_params:
                torch.nn.utils.clip_grad_norm_(tokenizer_params, float(args.grad_clip_norm))
            prior_params = [parameter for parameter in trainable if id(parameter) in prior_ids]
            if prior_params:
                torch.nn.utils.clip_grad_norm_(prior_params, float(args.prior_grad_clip_norm))
        else:
            torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not continuous_q and pending_usage:
            update_usage_ema_flat(model.codec.quantizer, torch.cat(pending_usage, dim=0), max(args.rates_list))
        pending_usage = []
        pending_backward = False

    for batch_index, (img, _label) in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        img = img.to(device, non_blocking=True)
        with torch.no_grad():
            layer1 = model.source.layer1(img)
            teacher_outputs = model.source_teacher_outputs(img, layer1["x1"])
        with torch.set_grad_enabled(train):
            z = model.codec.encode(img, layer1["x1"])
            branches: dict[int, dict[str, torch.Tensor]] = {}
            recon_terms: list[torch.Tensor] = []
            vq_terms: list[torch.Tensor] = []
            mse_terms: dict[int, torch.Tensor] = {}
            keep_sum = img.new_zeros(())
            # Paper Eq. (9): nested channel dropout is stochastic at the
            # iteration level.  With probability 1-alpha we train the full
            # representation, rather than masking q2 on every step.
            use_nested_dropout = bool(
                train and args.nested_channel_dropout
                and torch.rand((), device=img.device) < float(args.nested_dropout_prob)
            )
            if use_nested_dropout:
                q_mask, q_keep = nested_channel_mask(int(args.latent_c), int(img.shape[0]), img.device)
            else:
                q_mask = None
                q_keep = torch.full((int(img.shape[0]),), int(args.latent_c), device=img.device)
            for weight, rate in zip(args.rate_weights_list, args.rates_list):
                if continuous_q:
                    q_st, q_hard = z, z.detach()
                    indices = torch.zeros((int(img.shape[0]), int(args.latent_c)), dtype=torch.long, device=img.device)
                    stats = {"vq_loss": z.new_zeros(()), "vq_mse": z.new_zeros(())}
                else:
                    q_st, q_hard, indices, stats = model.codec.quantizer.forward_at_k(z, rate, update_usage=False)
                    if use_nested_dropout and bool(args.nested_vq_active_only):
                        stats.update(active_prefix_vq_stats(z, q_hard, q_mask, float(args.beta_commit)))
                if use_nested_dropout:
                    q_for_decode = q_st * q_mask.to(dtype=q_st.dtype)
                else:
                    q_for_decode = q_st
                keep_sum = keep_sum + q_keep.float().mean()
                decoded = model.decode_outputs(q_for_decode, layer1["z1"], layer1["x1"])
                final = decoded["final"]
                mse = (final.float() - img.float()).square().flatten(1).mean(dim=1)
                recon_terms.append(float(weight) * mse.mean())
                vq_terms.append(float(weight) * stats["vq_loss"])
                mse_terms[rate] = mse
                branches[rate] = {"q_hard": q_hard, "indices": indices, "final": final, "u2": decoded["u2"], "vq_mse": stats["vq_mse"]}
            loss_recon = torch.stack(recon_terms).sum()
            loss_vq = torch.stack(vq_terms).sum()
            x1_mse = (layer1["x1"].float() - img.float()).square().flatten(1).mean(dim=1)
            mono = [F.relu(mse_terms[high] - mse_terms[low].detach() + float(args.monotonic_margin)).mean() for low, high in zip(args.rates_list[:-1], args.rates_list[1:])]
            loss_mono = torch.stack(mono).mean() if mono else loss_recon.new_zeros(())
            if train and float(args.lambda_q_relevance) > 0.0:
                high = max(args.rates_list)
                zero_final = model.decode(torch.zeros_like(branches[high]["q_hard"]), layer1["z1"], layer1["x1"])
                zero_mse = (zero_final.float() - img.float()).square().flatten(1).mean(dim=1)
                # Keep the zero-code branch semantically equal to the Layer-1
                # fallback.  Without this anchor a relevance hinge could be
                # satisfied by deliberately degrading zero-q rather than by
                # making transmitted code information improve x1.
                loss_zero_anchor = F.mse_loss(zero_final.float(), layer1["x1"].detach().float())
                # Require the actual code path to outperform the Layer-1-like
                # zero-code path by a nonzero MSE margin.  This prevents the
                # learned combiner from treating q as a decorative bypass.
                loss_relevance = F.relu(
                    float(args.q_relevance_margin) - (zero_mse - mse_terms[high])
                ).mean()
            else:
                loss_relevance = loss_recon.new_zeros(())
                loss_zero_anchor = loss_recon.new_zeros(())
            if train and float(args.lambda_direct_gain) > 0.0 and float(args.direct_gain_db) > 0.0:
                # Directly optimize the acceptance quantity against frozen x1,
                # not merely against a learned zero-q fallback.  A G-dB PSNR
                # improvement means mse_q <= mse_x1 * 10^(-G/10).
                high = max(args.rates_list)
                required_reduction = x1_mse * (1.0 - 10.0 ** (-float(args.direct_gain_db) / 10.0))
                achieved_reduction = x1_mse - mse_terms[high]
                loss_direct_gain = F.relu(required_reduction - achieved_reduction).mean()
            else:
                loss_direct_gain = loss_recon.new_zeros(())
            if teacher_outputs is not None:
                teacher_final_terms = [
                    float(weight) * F.mse_loss(branches[rate]["final"].float(), teacher_outputs["final"].float())
                    for weight, rate in zip(args.rate_weights_list, args.rates_list)
                ]
                teacher_u2_terms = [
                    float(weight) * F.mse_loss(branches[rate]["u2"].float().clamp(0.0, 1.0), teacher_outputs["u2"].float())
                    for weight, rate in zip(args.rate_weights_list, args.rates_list)
                ]
                loss_teacher_final = torch.stack(teacher_final_terms).sum()
                loss_teacher_u2 = torch.stack(teacher_u2_terms).sum()
            else:
                loss_teacher_final = loss_recon.new_zeros(())
                loss_teacher_u2 = loss_recon.new_zeros(())
            if train and not continuous_q and model.predictability_prior is None:
                loss_usage, usage_ratio = soft_usage_entropy_loss(z, model.codec.quantizer, args)
            else:
                loss_usage, usage_ratio = loss_recon.new_zeros(()), loss_recon.new_zeros(())
            if model.predictability_prior is not None and not continuous_q:
                active_for_prior = q_mask if use_nested_dropout else None
                prior_ce_terms: list[torch.Tensor] = []
                prior_rate_terms: list[torch.Tensor] = []
                prior_entropy_terms: list[torch.Tensor] = []
                prior_accuracy_terms: list[torch.Tensor] = []
                prior_usage_terms: list[torch.Tensor] = []
                prior_usage_ratio_terms: list[torch.Tensor] = []
                z1_prior = layer1["z1"][:, :int(args.d2_z1_channels)]
                for weight, rate in zip(args.rate_weights_list, args.rates_list):
                    indices = branches[rate]["indices"]
                    logits = model.predictability_prior.forward_teacher(z1_prior, indices)[:, :, :int(rate)]
                    prior_ce_terms.append(float(weight) * masked_hard_index_nll(logits, indices, active_for_prior))
                    assignments = soft_channel_assignments(z, model.codec.quantizer, int(rate), float(args.predictability_temperature))
                    soft_rate, assignment_entropy = masked_soft_rate_nll(assignments, logits, active_for_prior)
                    prior_rate_terms.append(float(weight) * soft_rate)
                    prior_entropy_terms.append(float(weight) * assignment_entropy)
                    if train and float(args.lambda_soft_usage) > 0.0:
                        usage_deficit, usage_ratio_rate = masked_channel_marginal_entropy_deficit(assignments, active_for_prior)
                    else:
                        usage_deficit, usage_ratio_rate = loss_recon.new_zeros(()), loss_recon.new_zeros(())
                    prior_usage_terms.append(float(weight) * usage_deficit)
                    prior_usage_ratio_terms.append(float(weight) * usage_ratio_rate)
                    if active_for_prior is None:
                        prior_accuracy = (logits.argmax(dim=-1) == indices).float().mean()
                    else:
                        mask = active_for_prior[:, :, 0, 0].to(dtype=torch.float32)
                        prior_accuracy = ((logits.argmax(dim=-1) == indices).float() * mask).sum() / mask.sum().clamp_min(1.0)
                    prior_accuracy_terms.append(float(weight) * prior_accuracy)
                loss_prior_ce = torch.stack(prior_ce_terms).sum()
                loss_predictability = torch.stack(prior_rate_terms).sum()
                prior_assignment_entropy = torch.stack(prior_entropy_terms).sum()
                prior_index_accuracy = torch.stack(prior_accuracy_terms).sum()
                loss_usage = torch.stack(prior_usage_terms).sum()
                usage_ratio = torch.stack(prior_usage_ratio_terms).sum()
            else:
                loss_prior_ce = loss_recon.new_zeros(())
                loss_predictability = loss_recon.new_zeros(())
                prior_assignment_entropy = loss_recon.new_zeros(())
                prior_index_accuracy = loss_recon.new_zeros(())
            loss = (
                loss_recon
                + float(args.lambda_vq) * loss_vq
                + float(args.lambda_monotonic) * loss_mono
                + float(args.lambda_q_relevance) * loss_relevance
                + float(args.lambda_zero_anchor) * loss_zero_anchor
                + float(args.lambda_direct_gain) * loss_direct_gain
                + float(args.lambda_soft_usage) * loss_usage
                + float(args.lambda_teacher_final) * loss_teacher_final
                + float(args.lambda_teacher_u2) * loss_teacher_u2
                + float(args.lambda_prior_ce) * loss_prior_ce
                + float(args.lambda_predictability) * loss_predictability
            )
            if train:
                # Divide only the gradient; reported losses retain their
                # physical-batch scale for comparability with prior runs.
                (loss / float(accum_steps)).backward()
                pending_backward = True
                if not continuous_q:
                    index_parts = []
                    for rate in args.rates_list:
                        index = branches[rate]["indices"].detach()
                        if use_nested_dropout and bool(args.nested_vq_active_only):
                            active = q_mask[:, :, 0, 0].bool()
                            index = index[active]
                        index_parts.append(index.reshape(-1))
                    joined = torch.cat(index_parts, dim=0)
                    pending_usage.append(joined)
                if batch_index % accum_steps == 0:
                    optimizer_step()
        batch = int(img.shape[0])
        count += batch
        x1_psnr = psnr_per_image(layer1["x1"], img)
        values = {
            "loss": float(loss.detach()),
            "loss_recon": float(loss_recon.detach()),
            "loss_vq": float(loss_vq.detach()),
            "loss_monotonic": float(loss_mono.detach()),
            "loss_q_relevance": float(loss_relevance.detach()),
            "loss_zero_anchor": float(loss_zero_anchor.detach()),
            "loss_direct_gain": float(loss_direct_gain.detach()),
            "loss_teacher_final": float(loss_teacher_final.detach()),
            "loss_teacher_u2": float(loss_teacher_u2.detach()),
            "loss_prior_ce": float(loss_prior_ce.detach()),
            "loss_predictability": float(loss_predictability.detach()),
            "prior_assignment_entropy": float(prior_assignment_entropy.detach()),
            "prior_index_accuracy": float(prior_index_accuracy.detach()),
            "loss_soft_usage": float(loss_usage.detach()),
            "soft_usage_entropy_ratio": float(usage_ratio.detach()),
            "psnr_x1": float(x1_psnr.mean()),
        }
        if teacher_outputs is not None:
            teacher_psnr = psnr_per_image(teacher_outputs["final"], img)
            values["psnr_source_teacher"] = float(teacher_psnr.mean())
            values["delta_x1_source_teacher"] = float((teacher_psnr - x1_psnr).mean())
        if train and bool(args.nested_channel_dropout):
            values["train_channels_kept_mean"] = float(keep_sum.detach() / len(args.rates_list))
        for rate in args.rates_list:
            branch_psnr = psnr_per_image(branches[rate]["final"], img)
            values[f"psnr_k{rate}"] = float(branch_psnr.mean())
            values[f"delta_x1_k{rate}"] = float((branch_psnr - x1_psnr).mean())
            values[f"vq_mse_k{rate}"] = float(branches[rate]["vq_mse"])
            if not train:
                zero = model.decode(torch.zeros_like(branches[rate]["q_hard"]), layer1["z1"], layer1["x1"])
                shuffle = model.decode(model.codec.quantizer.shuffle_tokens(branches[rate]["q_hard"]), layer1["z1"], layer1["x1"])
                values[f"drop_zero_k{rate}"] = float((branch_psnr - psnr_per_image(zero, img)).mean())
                values[f"drop_shuffle_k{rate}"] = float((branch_psnr - psnr_per_image(shuffle, img)).mean())
            if not continuous_q:
                histograms[rate] += torch.bincount(branches[rate]["indices"].detach().cpu().reshape(-1), minlength=rate).double()[:rate]
        for low, high in zip(args.rates_list[:-1], args.rates_list[1:]):
            lower = psnr_per_image(branches[low]["final"], img)
            higher = psnr_per_image(branches[high]["final"], img)
            values[f"paired_gain_k{high}"] = float((higher - lower).mean())
            values[f"paired_strict_k{high}"] = float((higher > lower).float().mean())
        for name, value in values.items():
            sums[name] = sums.get(name, 0.0) + value * batch
    # Flush a partial effective batch at the end of an epoch.  DIV2K's
    # loader length is not assumed to be divisible by ``accum_steps``.
    if train:
        optimizer_step()
    if count == 0:
        raise RuntimeError("empty epoch")
    result = {name: value / count for name, value in sums.items()}
    result["continuous_q_warmup"] = float(continuous_q)
    for rate, hist in histograms.items():
        result.update({f"{name}_k{rate}": value for name, value in histogram_metrics(hist, rate).items()})
        result[f"tokens_per_image_k{rate}"] = float(args.latent_c)
        result[f"bits_per_image_k{rate}"] = float(args.latent_c) * math.log2(float(rate))
        result[f"bpp_k{rate}"] = result[f"bits_per_image_k{rate}"] / (256.0 * 256.0)
    if not train:
        health = model.codec.quantizer.codebook_metrics(max(args.rates_list), seed=int(args.seed))
        result.update(health)
        monotonic = all(result[f"psnr_k{high}"] > result[f"psnr_k{low}"] for low, high in zip(args.rates_list[:-1], args.rates_list[1:]))
        noncollapse = all(result[f"ppl_ratio_k{rate}"] >= float(args.min_ppl_ratio) and result[f"top1_k{rate}"] <= float(args.max_top1) for rate in args.rates_list)
        relevant = all(result[f"drop_zero_k{rate}"] >= float(args.min_ablation_drop) and result[f"drop_shuffle_k{rate}"] >= float(args.min_ablation_drop) for rate in args.rates_list)
        result["capacity_goal_met"] = float(monotonic)
        result["noncollapse_goal_met"] = float(noncollapse)
        result["oracle_goal_met"] = float(monotonic and noncollapse and relevant and all(result[f"delta_x1_k{rate}"] > 0.0 for rate in args.rates_list))
    return result


def print_header(args: argparse.Namespace, source, init: dict[str, float | str]) -> None:
    print("=== explore-4 | paper-2605.26089v2 CVQ spatial sweep ===", flush=True)
    print("实验设计", flush=True)
    e2_mode = "source-E2 initialized" if bool(getattr(args, "source_e2_init", False)) else "fresh"
    print(f"  Layer1={args.arch} frozen; Layer2 E2={e2_mode}, downsample={args.e2_downsamples} -> {args.spatial_size}x{args.spatial_size}; CVQ=global channel maps; K={args.rates_list}; D={args.resolved_embedding_dim}", flush=True)
    decode_contract = (
        "x2=x1+[H_source(x1,clamp(D2(z1,q2)))-H_source(x1,clamp(D2(z1,0)))]"
        if bool(getattr(args, "swin_source_combiner_residual", False))
        else "x2=x1+[D2(z1,q2)-D2(z1,0)]"
    )
    print(f"  sender=img+x1 -> E2 -> channel-CVQ(C={args.latent_c}) -> [z1[:,:{args.d2_z1_channels}] + q2({args.d2_input_channels-args.d2_z1_channels})]={args.d2_input_channels}ch -> D2; {decode_contract}", flush=True)
    print(f"  continuous residual warmup epochs={args.continuous_warmup_epochs}; initial residual gain={args.initial_residual_gain}; channel_rms_normalize={bool(getattr(args, 'channel_rms_normalize', False))}; codebook is initialized from post-warmup channel maps; physical/effective batch={args.batch_size}/{args.batch_size * args.accum_steps}", flush=True)
    print("loss设计", flush=True)
    print(f"  sum_K[w_K*MSE(x2_K,img)] + {args.lambda_vq}*commitment + {args.lambda_monotonic}*nested-K monotonic + {args.lambda_q_relevance}*q-relevance + {args.lambda_zero_anchor}*zero-anchor + {args.lambda_direct_gain}*direct-{args.direct_gain_db}dB-vs-x1 + {args.lambda_teacher_final}*teacher-final + {args.lambda_teacher_u2}*teacher-u2 + {args.lambda_prior_ce}*z1-prior-CE + {args.lambda_predictability}*conditional-rate + {args.lambda_soft_usage}*soft-usage; nested_channel_dropout={args.nested_channel_dropout}, alpha={args.nested_dropout_prob}, active-prefix-vq={args.nested_vq_active_only}", flush=True)
    if bool(getattr(args, "conditional_prior", False)):
        print(
            f"  conditional tokenizer objective: Hhat(I|z1,I<k) and the matched active-prefix marginal Hhat(I); "
            f"both use soft-assignment temperature={args.predictability_temperature}",
            flush=True,
        )
    print("模块选择", flush=True)
    print(f"  CVQ codebook is global [K,H,W] (or [K,1,D]); q2=0 is exactly x1; nested channel dropout affects q2 only; z1-only conditional-prior={bool(getattr(args, 'conditional_prior', False))}; source-combiner-residual={bool(getattr(args, 'swin_source_combiner_residual', False))}; frozen-student-decoder={bool(getattr(args, 'freeze_student_decoder', False))}; continuous-valid gate={args.continuous_min_delta}dB; Layer1 uses RandomCrop+Flip train and CenterCrop valid.", flush=True)
    print(f"  source_layer1={args.source_checkpoint}; source_layer2={args.layer2_source_checkpoint}; init={init}", flush=True)


def save_checkpoint(path: Path, model: SenderCVQ, optimizer, args: argparse.Namespace, epoch: int, metrics: dict[str, float], init: dict[str, float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "explore4_paper_cvq_spatial",
        "paper": "arXiv:2605.26089v2 Channel-wise Vector Quantization",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "init": init,
        "receiver_contract": {"allowed_inputs": ["z1", "x1"], "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"], "note": "CAR is trained separately after this sender tokenizer is frozen."},
        "data_contract": {"train": "RandomCrop(256)+RandomHorizontalFlip", "validation": "CenterCrop(256)"},
    }
    torch.save(payload, path)
    print(f"saved checkpoint: {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--arch", choices=["cnn", "swin"], required=True, help="frozen Layer1 architecture")
    parser.add_argument("--e2-downsamples", type=int, choices=[2, 3, 4, 5], required=True)
    parser.add_argument("--source-e2-init", action="store_true", help="initialize matched 16x16 E2 and D2-front q branch from the pretrained source Layer2")
    parser.add_argument("--channel-rms-normalize", action="store_true", help="normalize each channel map to unit RMS before global channel-VQ lookup")
    parser.add_argument("--swin-source-combiner-residual", action="store_true", help="Swin only: preserve the pretrained source combiner through an exact q=0 residualized wrapper and create a frozen full Layer2 teacher")
    parser.add_argument("--freeze-source-combiner", action="store_true", help="with --swin-source-combiner-residual, keep the copied source combiner frozen while adapting E2/D2")
    parser.add_argument("--freeze-student-decoder", action="store_true", help="freeze student D2, combiner and q2 synthesis bridge; useful while first aligning a new CVQ codebook to a validated decoder manifold")
    parser.add_argument("--swin-source-pca-init", action="store_true", help="Swin only: initialize the mandated 320->q2->320 adapter with centered PCA of source E2 latents")
    parser.add_argument("--swin-pca-batches", type=int, default=16, help="maximum source-data batches used for the PCA adapter fit")
    parser.add_argument("--swin-pca-max-tokens", type=int, default=65536, help="maximum 320-D source E2 spatial vectors used for PCA")
    parser.add_argument("--latent-c", type=int, default=240, help="number of q2 channel tokens")
    parser.add_argument("--d2-input-channels", type=int, default=256, help="new D2 front-end input channels: z1 + q2")
    parser.add_argument("--source-d2-input-channels", type=int, default=320, help="inherited source D2 trunk input-channel contract")
    parser.add_argument("--d2-z1-channels", type=int, default=16, help="reserved non-dropout z1 channels concatenated before q2")
    parser.add_argument("--embedding-dim", type=int, default=0, help="CVQ code-vector D; 0 means native H*W")
    parser.add_argument("--rates", default="256,1024,4096", help="nested global K prefixes")
    parser.add_argument("--rate-weights", default="1", help="one weight broadcasts to every requested K")
    parser.add_argument("--nested-channel-dropout", action="store_true", help="paper nested channel dropout during tokenizer training")
    parser.add_argument("--nested-dropout-prob", type=float, default=0.5, help="paper alpha: probability of applying nested q2 dropout on a training step")
    parser.add_argument("--nested-vq-active-only", action="store_true", help="when nested dropout is active, compute VQ/EMA updates only over the retained channel prefix (paper Appendix B.1)")
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--layer2-source-checkpoint", default="")
    parser.add_argument("--combiner-width", type=int, default=64)
    parser.add_argument("--combiner-blocks", type=int, default=8)
    parser.add_argument("--beta-commit", type=float, default=0.25)
    parser.add_argument("--lambda-vq", type=float, default=0.02)
    parser.add_argument("--lambda-monotonic", type=float, default=0.0)
    parser.add_argument("--monotonic-margin", type=float, default=1e-5)
    parser.add_argument("--lambda-q-relevance", type=float, default=0.0)
    parser.add_argument("--q-relevance-margin", type=float, default=1e-5)
    parser.add_argument("--lambda-zero-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-direct-gain", type=float, default=0.0, help="weight for direct x2-vs-frozen-x1 PSNR gain hinge")
    parser.add_argument("--direct-gain-db", type=float, default=0.0, help="required oracle PSNR gain in dB for the direct hinge")
    parser.add_argument("--lambda-teacher-final", type=float, default=0.0, help="weight of frozen full-source final-image distillation; active only for the Swin source-combiner path")
    parser.add_argument("--lambda-teacher-u2", type=float, default=0.0, help="weight of frozen full-source clamped-D2 distillation; active only for the Swin source-combiner path")
    parser.add_argument("--conditional-prior", action="store_true", help="train a z1-only causal channel-index prior alongside the tokenizer")
    parser.add_argument("--prior-hidden", type=int, default=192)
    parser.add_argument("--prior-layers", type=int, default=4)
    parser.add_argument("--prior-heads", type=int, default=6)
    parser.add_argument("--prior-dropout", type=float, default=0.0)
    parser.add_argument("--lambda-prior-ce", type=float, default=0.0, help="hard-index CE weight for the z1-only conditional prior")
    parser.add_argument("--lambda-predictability", type=float, default=0.0, help="soft conditional-rate RD weight that moves E2/codebook assignments toward the z1-only prior")
    parser.add_argument("--predictability-temperature", type=float, default=1.0, help="soft CVQ-assignment temperature for the conditional-rate loss")
    parser.add_argument("--prior-lr", type=float, default=0.0, help="z1-only prior learning rate; 0 reuses --lr")
    parser.add_argument("--prior-grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--lambda-soft-usage", type=float, default=0.0,
        help=(
            "coefficient of code-usage entropy; with --conditional-prior this is the "
            "matched per-channel marginal-entropy term in H(I)-H(I|z1,i<), otherwise "
            "the legacy batch-marginal anti-collapse term"
        ),
    )
    parser.add_argument("--soft-usage-temperature", type=float, default=0.1)
    parser.add_argument("--soft-usage-samples", type=int, default=512)
    parser.add_argument("--usage-decay", type=float, default=0.99)
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--codebook-chunk-size", type=int, default=1024)
    parser.add_argument("--codebook-init-batches", type=int, default=16)
    parser.add_argument("--codebook-init-max-samples", type=int, default=65536)
    parser.add_argument("--min-ppl-ratio", type=float, default=0.01)
    parser.add_argument("--max-top1", type=float, default=0.25)
    parser.add_argument("--min-ablation-drop", type=float, default=0.1)
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument("--save-dir", default="MY-V2/jscc-f/explore-4/checkpoints-cvq")
    parser.add_argument("--log-json", default="")
    parser.add_argument("--version", default="cvq-spatial-v1")
    parser.add_argument("--resume", default="", help="resume a compatible explore-4 CVQ sender; optimizer is intentionally reset so optional new prior modules can be added")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--continuous-warmup-epochs", type=int, default=0, help="train residual codec with continuous q2 before codebook initialization")
    parser.add_argument("--continuous-min-delta", type=float, default=0.0, help="required strict-valid full-q PSNR gain before VQ starts")
    parser.add_argument("--require-continuous-goal", action="store_true", help="stop after continuous warmup if --continuous-min-delta is not met on CenterCrop validation")
    parser.add_argument("--initial-residual-gain", type=float, default=0.01, help="initial scale for the strict q2-only enhancement residual")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=1, help="gradient accumulation; preserves objective while increasing CVQ token statistics per codebook update")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--latest-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.rates_list = parse_int_list(args.rates)
    args.rate_weights_list = parse_float_list(args.rate_weights, len(args.rates_list))
    args.spatial_size = 256 // (2 ** int(args.e2_downsamples))
    args.resolved_embedding_dim = int(args.embedding_dim) or int(args.spatial_size) ** 2
    # Unlike the inherited source checkpoint, this fresh Layer-2 E2 is
    # followed by a trainable 320->C analysis projection.  Paper CVQ's 32x32
    # setting naturally uses C=1024 channel tokens, so C must not inherit the
    # old source latent's 320-channel ceiling.
    if not 1 <= int(args.latent_c) <= 1024:
        raise ValueError("--latent-c must be in [1,1024]")
    if bool(args.source_e2_init) and int(args.e2_downsamples) != 4:
        raise ValueError("--source-e2-init only applies to the matched 16x16 E2")
    if bool(args.swin_source_combiner_residual) and str(args.arch) != "swin":
        raise ValueError("--swin-source-combiner-residual requires --arch swin")
    if bool(args.freeze_source_combiner) and not bool(args.swin_source_combiner_residual):
        raise ValueError("--freeze-source-combiner requires --swin-source-combiner-residual")
    if bool(args.swin_source_pca_init) and (not bool(args.swin_source_combiner_residual) or not bool(args.source_e2_init)):
        raise ValueError("--swin-source-pca-init requires --swin-source-combiner-residual and --source-e2-init")
    if int(args.swin_pca_batches) < 1 or int(args.swin_pca_max_tokens) < 320:
        raise ValueError("--swin-pca-batches must be >=1 and --swin-pca-max-tokens must be >=320")
    if (float(args.lambda_teacher_final) > 0.0 or float(args.lambda_teacher_u2) > 0.0) and not bool(args.swin_source_combiner_residual):
        raise ValueError("teacher distillation losses require --swin-source-combiner-residual")
    if (float(args.lambda_prior_ce) > 0.0 or float(args.lambda_predictability) > 0.0) and not bool(args.conditional_prior):
        raise ValueError("conditional-rate weights require --conditional-prior")
    if bool(args.conditional_prior) and (int(args.prior_hidden) < 1 or int(args.prior_layers) < 1 or int(args.prior_heads) < 1 or int(args.prior_hidden) % int(args.prior_heads)):
        raise ValueError("conditional-prior hidden/layers/heads are incompatible")
    if float(args.predictability_temperature) <= 0.0:
        raise ValueError("--predictability-temperature must be positive")
    if float(args.prior_lr) < 0.0 or float(args.prior_grad_clip_norm) <= 0.0:
        raise ValueError("--prior-lr must be nonnegative and --prior-grad-clip-norm positive")
    if not 0 <= int(args.d2_z1_channels) < int(args.d2_input_channels):
        raise ValueError("--d2-z1-channels must be in [0, d2-input-channels)")
    if int(args.resolved_embedding_dim) < 1:
        raise ValueError("--embedding-dim must be positive")
    if float(args.soft_usage_temperature) <= 0.0 or int(args.soft_usage_samples) < 1:
        raise ValueError("soft usage temperature/samples must be positive")
    if not 0.0 <= float(args.nested_dropout_prob) <= 1.0:
        raise ValueError("--nested-dropout-prob must be in [0,1]")
    if int(args.continuous_warmup_epochs) < 0:
        raise ValueError("--continuous-warmup-epochs must be nonnegative")
    if float(args.continuous_min_delta) < 0.0:
        raise ValueError("--continuous-min-delta must be nonnegative")
    if bool(args.require_continuous_goal) and int(args.continuous_warmup_epochs) < 1:
        raise ValueError("--require-continuous-goal requires --continuous-warmup-epochs >= 1")
    if not 0.0 < float(args.initial_residual_gain) < 1.0:
        raise ValueError("--initial-residual-gain must lie in (0,1)")
    if int(args.accum_steps) < 1:
        raise ValueError("--accum-steps must be >= 1")
    if bool(args.cpu):
        args.num_workers = 0
        args.val_num_workers = 0
    return args


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    source_path = args.source_checkpoint or nested.DEFAULT_SOURCES[str(args.arch)]
    args.source_checkpoint = str(source_path)
    checkpoint = base.jsccf_io.load_checkpoint(str(nested.resolve_path(source_path)))
    source_args = argparse.Namespace(**checkpoint["args"])
    train_loader, val_loader, device = nested.build_loaders(args, source_args)
    source = nested.load_source(args, device)
    args.layer2_source_checkpoint = str(args.layer2_source_checkpoint)
    model = SenderCVQ(source, args).to(device)
    start_epoch = 0
    if args.resume:
        resume_path = nested.resolve_path(args.resume)
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("stage") != "explore4_paper_cvq_spatial":
            raise ValueError(f"not an explore-4 spatial CVQ checkpoint: {resume_path}")
        saved_args = dict(payload.get("args", {}))
        for name in ("arch", "e2_downsamples", "latent_c", "d2_input_channels", "source_d2_input_channels", "d2_z1_channels"):
            if name in saved_args and getattr(args, name) != saved_args[name]:
                raise ValueError(f"--resume incompatible {name}: current={getattr(args, name)!r}, saved={saved_args[name]!r}")
        saved_rates = list(saved_args.get("rates_list", []))
        if saved_rates and [int(item) for item in args.rates_list] != [int(item) for item in saved_rates]:
            raise ValueError(f"--resume incompatible rates: current={args.rates_list}, saved={saved_rates}")
        missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
        allowed_missing = {name for name in model.state_dict() if name.startswith("predictability_prior.")}
        # A pre-prior checkpoint legitimately has every prior key missing;
        # a later prior-calibration/MI checkpoint legitimately has none.  In
        # both cases, only *unexpected* or non-prior missing keys break the
        # tokenizer contract.
        if not set(missing).issubset(allowed_missing) or unexpected:
            raise RuntimeError(f"incompatible resume state missing={missing} unexpected={unexpected}")
        start_epoch = int(payload.get("epoch", 0))
        init = dict(payload.get("init", {"method": "resumed"}))
        adapter_init = dict(init)
        best = float(payload.get("metrics", {}).get(f"psnr_k{max(args.rates_list)}", float("-inf")))
        print(f"resumed CVQ sender from {resume_path} at epoch={start_epoch}; optimizer reset", flush=True)
    else:
        adapter_init = initialize_swin_pca_adapter(train_loader, model, args, device)
        if int(args.continuous_warmup_epochs) > 0:
            init = {"method": "deferred-after-continuous-residual-warmup", **adapter_init}
        else:
            init = initialize_channel_codebook(train_loader, model, args, device)
            init.update(adapter_init)
        best = float("-inf")
    if args.resume and int(args.continuous_warmup_epochs) > 0:
        raise ValueError("--resume requires --continuous-warmup-epochs=0; resume begins from an already initialized VQ tokenizer")
    if start_epoch >= int(args.epochs):
        raise ValueError(f"--epochs={args.epochs} must exceed resumed epoch {start_epoch}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("CVQ model has no trainable parameters")
    if model.predictability_prior is not None:
        prior_ids = {id(parameter) for parameter in model.predictability_prior.parameters() if parameter.requires_grad}
        tokenizer_params = [parameter for parameter in trainable if id(parameter) not in prior_ids]
        prior_params = [parameter for parameter in trainable if id(parameter) in prior_ids]
        groups = []
        if tokenizer_params:
            groups.append({"params": tokenizer_params, "lr": float(args.lr)})
        if prior_params:
            groups.append({"params": prior_params, "lr": float(args.prior_lr) or float(args.lr)})
        optimizer = optim.AdamW(groups, weight_decay=float(args.weight_decay))
    else:
        optimizer = optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    print_header(args, source, init)
    if bool(args.smoke):
        args.max_train_batches = 1
        args.max_val_batches = 1
        args.epochs = 1
    save_root = nested.resolve_path(args.save_dir) / str(args.version)
    history: list[dict[str, object]] = []
    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        began = time.time()
        continuous_q = epoch <= int(args.continuous_warmup_epochs)
        train_metrics = run_epoch(train_loader, model, optimizer, args, device, train=True, continuous_q=continuous_q)
        phase = "continuous warmup" if continuous_q else "cvq spatial train"
        print(f"[{phase} {epoch:03d}/{args.epochs}] {train_metrics} time={time.time()-began:.1f}s", flush=True)
        record: dict[str, object] = {"epoch": epoch, "train": train_metrics}
        should_validate = (
            epoch % int(args.val_every) == 0
            or epoch == int(args.epochs)
            or (continuous_q and epoch == int(args.continuous_warmup_epochs))
        )
        if should_validate:
            val_metrics = run_epoch(val_loader, model, None, args, device, train=False, continuous_q=continuous_q)
            if continuous_q:
                high = max(args.rates_list)
                continuous_gain = float(val_metrics[f"delta_x1_k{high}"])
                val_metrics["continuous_goal_met"] = float(continuous_gain >= float(args.continuous_min_delta))
                print(f"[continuous strict val {epoch:03d}] {val_metrics}", flush=True)
            else:
                print(f"[cvq spatial val {epoch:03d}] {val_metrics}", flush=True)
            record["val"] = val_metrics
            if not continuous_q:
                current = float(val_metrics[f"psnr_k{max(args.rates_list)}"])
                if current > best:
                    best = current
                    save_checkpoint(save_root / "best.pth", model, optimizer, args, epoch, val_metrics, init)
        if continuous_q and epoch == int(args.continuous_warmup_epochs):
            if bool(args.require_continuous_goal) and not bool(record.get("val", {}).get("continuous_goal_met", 0.0)):
                history.append(record)
                if args.log_json:
                    path = nested.resolve_path(args.log_json)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
                print(
                    f"continuous strict-valid gate failed: required {args.continuous_min_delta:.3f}dB; VQ initialization/training skipped",
                    flush=True,
                )
                return
            init = initialize_channel_codebook(train_loader, model, args, device)
            init.update(adapter_init)
            print(f"continuous warmup complete; reinitialized CVQ codebook: {init}", flush=True)
        history.append(record)
        if not continuous_q and (epoch % int(args.latest_every) == 0 or epoch == int(args.epochs)):
            latest_metrics = record.get("val", train_metrics)
            save_checkpoint(save_root / "latest.pth", model, optimizer, args, epoch, latest_metrics, init)
        if args.log_json:
            path = nested.resolve_path(args.log_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train(parse_args())
