#!/usr/bin/env python3
"""Train a receiver-only q2_hat model on a successful direct-FSQ oracle.

Inference graph (the graph that matters for the no-leakage requirement)::

    received z1 -> frozen D1 -> x1
    ReceiverPredictor(z1, x1) -> q2_hat
    frozen D2(q2_hat) -> u2_hat
    frozen combiner(x1, u2_hat) -> x2_hat

``img`` and oracle ``q2`` are supervised targets during training only.  They
are not members of ``ReceiverCondition`` and cannot be passed to the predictor
forward method.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import os
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
EXPLORE_DIR = JSCCF_DIR / "explore"
CDDM_ROOT = JSCCF_DIR.parents[1]
for path in (THIS_DIR, JSCCF_DIR, EXPLORE_DIR, CDDM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contracts import (  # noqa: E402
    ReceiverCondition,
    assert_div2k_crop_protocol,
    assert_receiver_only_module,
    assert_training_targets_are_not_inputs,
    make_receiver_condition,
)
from receiver_models import (  # noqa: E402
    AutoregressiveFSQJointTokenPredictor,
    QResidualPredictor,
    ReceiverPrediction,
    build_receiver_predictor,
)


DEFAULT_ORACLE = (
    "MY-V2/jscc-f/explore/checkpoints-direct/"
    "direct-cnn-d3-l17x17x17-group-compatible-blend-e100/"
    "jscc_f_direct-cnn-d3-l17x17x17-group-compatible-blend-e100_"
    "layer2_fsq_direct_cnn_d3_l17x17x17_group_compatible_blend_best.pth"
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


direct = load_module("jsccf_explore2_direct_support", EXPLORE_DIR / "train_layer2_fsq_direct.py")
base = direct.base


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return CDDM_ROOT / candidate


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def set_frozen(module: nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()


def load_fsq_oracle(path: str, device: torch.device) -> tuple[direct.DirectBundle, argparse.Namespace, dict]:
    checkpoint_path = resolve_path(path)
    checkpoint = base.jsccf_io.load_checkpoint(str(checkpoint_path))
    if str(checkpoint.get("stage", "")) != "layer2_fsq_direct":
        raise ValueError(f"receiver trainer needs layer2_fsq_direct checkpoint, got {checkpoint.get('stage')!r}")
    oracle_args = argparse.Namespace(**checkpoint["args"])
    source_path = checkpoint.get("source_layer2_ckpt") or getattr(oracle_args, "layer2_ckpt", "")
    if not source_path:
        raise ValueError("oracle checkpoint does not identify its Layer2 source checkpoint")
    source_checkpoint = base.jsccf_io.load_checkpoint(str(resolve_path(source_path)))
    direct.explore.ExploreIFSQQuantizer.config = oracle_args
    bundle = direct.build_direct_bundle(oracle_args, source_checkpoint, device)
    base.jsccf_io.load_state(bundle.e1, checkpoint["e1_state_dict"], "receiver_oracle_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, checkpoint["d1_state_dict"], "receiver_oracle_D1", strict=True)
    base.jsccf_io.load_state(
        bundle.tokenizer,
        checkpoint["tokenizer_state_dict"],
        "receiver_oracle_tokenizer",
        strict=True,
    )
    base.jsccf_io.load_state(
        bundle.combiner,
        checkpoint["combiner_state_dict"],
        "receiver_oracle_combiner",
        strict=True,
    )
    for module in (bundle.e1, bundle.d1, bundle.tokenizer, bundle.combiner):
        set_frozen(module)
    return bundle, oracle_args, checkpoint


class MetricSums:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def add(self, name: str, value: float, weight: int | float) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value) * float(weight)
        self.weights[name] = self.weights.get(name, 0.0) + float(weight)

    def means(self) -> dict[str, float]:
        return {name: value / max(1.0, self.weights[name]) for name, value in self.sums.items()}


class ReceiverResidualCombiner(nn.Module):
    """Identity-safe receiver combiner used for deployable joint finetuning."""

    def __init__(self, oracle_combiner: nn.Module) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 48, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(48, 3, 3, padding=1),
        )
        source_inner = getattr(oracle_combiner, "inner", oracle_combiner)
        source_net = getattr(source_inner, "net", None)
        if isinstance(source_net, nn.Sequential) and len(source_net) >= 3:
            self.net[0].load_state_dict(source_net[0].state_dict(), strict=True)
            self.net[1].load_state_dict(source_net[1].state_dict(), strict=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        return (x1 + self.net(torch.cat([x1, u2], dim=1))).clamp(0.0, 1.0)


class ReceiverQOnlyResidualCombiner(nn.Module):
    """Identity-safe combiner whose correction must pass through q2_hat/D2.

    ``x1`` is used only for the mandated residual addition.  The correction
    network cannot learn an x1-only postprocessor that ignores ``u2_hat``;
    consequently zero/shuffle/condition ablations test the actual receiver
    latent path rather than a six-channel combiner bypass.
    """

    def __init__(self, oracle_combiner: nn.Module) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(48, 3, 3, padding=1),
        )
        source_inner = getattr(oracle_combiner, "inner", oracle_combiner)
        source_net = getattr(source_inner, "net", None)
        if isinstance(source_net, nn.Sequential) and len(source_net) >= 3:
            source_first = source_net[0]
            if (
                isinstance(source_first, nn.Conv2d)
                and tuple(source_first.weight.shape) == (48, 6, 3, 3)
            ):
                # The source combiner consumes [x1,u2]; retain only its u2
                # filters so initialization still respects the no-bypass path.
                self.net[0].weight.data.copy_(source_first.weight.data[:, 3:6])
                if source_first.bias is not None and self.net[0].bias is not None:
                    self.net[0].bias.data.copy_(source_first.bias.data)
            if isinstance(source_net[1], nn.PReLU):
                self.net[1].load_state_dict(source_net[1].state_dict(), strict=True)
        # Exact x1 identity at initialization; gradients first train the last
        # layer and then propagate into D2/G once the correction path opens.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        return (x1 + self.net(u2)).clamp(0.0, 1.0)


class ReceiverQOnlyUNetCombiner(nn.Module):
    """Multi-scale q-path combiner; x1 is only the final residual anchor."""

    class Block(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            groups = min(32, int(channels))
            while int(channels) % groups:
                groups -= 1
            self.net = nn.Sequential(
                nn.GroupNorm(groups, int(channels)),
                nn.SiLU(),
                nn.Conv2d(int(channels), int(channels) * 2, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(int(channels) * 2, int(channels), 3, padding=1),
            )
            self.scale = nn.Parameter(torch.tensor(0.1))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.scale * self.net(value)

    def __init__(self, width: int = 32, blocks: int = 2) -> None:
        super().__init__()
        width = int(width)

        def refine(channels: int) -> nn.Sequential:
            return nn.Sequential(*[self.Block(channels) for _ in range(int(blocks))])

        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.enc0 = refine(width)
        self.down0 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.enc1 = refine(width * 2)
        self.down1 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.middle = refine(width * 4)
        self.up1 = nn.Conv2d(width * 6, width * 2, 3, padding=1)
        self.dec1 = refine(width * 2)
        self.up0 = nn.Conv2d(width * 3, width, 3, padding=1)
        self.dec0 = refine(width)
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        skip0 = self.enc0(self.stem(u2))
        skip1 = self.enc1(F.silu(self.down0(skip0)))
        middle = self.middle(F.silu(self.down1(skip1)))
        value = F.interpolate(middle, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec1(F.silu(self.up1(torch.cat([value, skip1], dim=1))))
        value = F.interpolate(value, size=skip0.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec0(F.silu(self.up0(torch.cat([value, skip0], dim=1))))
        return (x1 + self.head(value)).clamp(0.0, 1.0)


class ReceiverQGatedRefinerCombiner(nn.Module):
    """Preserve a trained q-path combiner and add a q-gated multi-scale refinement.

    The frozen ``base`` is an already validated receiver combiner.  The new
    correction is deliberately *not* an x1-only postprocessor: every feature
    entering its decoder originates from ``u2_hat`` and is merely modulated by
    x1.  With an all-zero u2 input, the bias-free q branch is identically zero
    and so is the new correction.  This keeps deployment exactly on
    ``q2_hat -> D2 -> combiner(x1, u2_hat)`` while letting x1 disambiguate
    artifacts in the generated Layer2 proposal.
    """

    class Block(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(int(channels), int(channels) * 2, 3, padding=1, bias=False),
                nn.SiLU(),
                nn.Conv2d(int(channels) * 2, int(channels), 3, padding=1, bias=False),
            )
            self.scale = nn.Parameter(torch.tensor(0.1))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.scale * self.net(value)

    def __init__(self, oracle_combiner: nn.Module, width: int = 48, blocks: int = 2) -> None:
        super().__init__()
        width = int(width)
        self.base = copy.deepcopy(oracle_combiner)
        # The base is initialized from a receiver checkpoint and deliberately
        # frozen.  Its receiver-q relevance is retained while this branch is
        # trained as a small, explicitly q-gated residual.
        self.base.requires_grad_(False)

        def refine(channels: int) -> nn.Sequential:
            return nn.Sequential(*[self.Block(channels) for _ in range(int(blocks))])

        self.u_stem = nn.Conv2d(3, width, 3, padding=1, bias=False)
        self.x_stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
        )
        self.x_gate0 = nn.Conv2d(width, width, 1)
        self.enc0 = refine(width)
        self.down0 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1, bias=False)
        self.x_gate1 = nn.Conv2d(width, width * 2, 1)
        self.enc1 = refine(width * 2)
        self.down1 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1, bias=False)
        self.x_gate2 = nn.Conv2d(width, width * 4, 1)
        self.middle = refine(width * 4)
        self.up1 = nn.Conv2d(width * 6, width * 2, 3, padding=1, bias=False)
        self.dec1 = refine(width * 2)
        self.up0 = nn.Conv2d(width * 3, width, 3, padding=1, bias=False)
        self.dec0 = refine(width)
        self.head = nn.Conv2d(width, 3, 3, padding=1, bias=False)
        nn.init.zeros_(self.head.weight)

    @staticmethod
    def _resize(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _modulate(q_feature: torch.Tensor, x_feature: torch.Tensor, gate: nn.Module) -> torch.Tensor:
        # q_feature is the only signal that can reach the correction decoder.
        # x_feature can select/modulate it, but cannot create a correction.
        return q_feature * torch.sigmoid(gate(x_feature))

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        base = self.base(x1, u2)
        x_feature = self.x_stem(x1)
        skip0 = self.enc0(self._modulate(self.u_stem(u2), x_feature, self.x_gate0))
        q1 = F.silu(self.down0(skip0))
        x1_small = self._resize(x_feature, q1)
        skip1 = self.enc1(self._modulate(q1, x1_small, self.x_gate1))
        q2 = F.silu(self.down1(skip1))
        x2_small = self._resize(x_feature, q2)
        middle = self.middle(self._modulate(q2, x2_small, self.x_gate2))
        value = self.dec1(
            F.silu(self.up1(torch.cat([self._resize(middle, skip1), skip1], dim=1)))
        )
        value = self.dec0(
            F.silu(self.up0(torch.cat([self._resize(value, skip0), skip0], dim=1)))
        )
        return (base + self.head(value)).clamp(0.0, 1.0)


class ReceiverFSQResidualD2(nn.Module):
    """Decode a three-map receiver FSQ latent into a centered RGB residual signal."""

    class Block(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(int(channels), int(channels), 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(int(channels), int(channels), 3, padding=1),
            )
            self.scale = nn.Parameter(torch.tensor(0.1))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.scale * self.body(value)

    def __init__(self, in_channels: int, width: int = 64, blocks: int = 2) -> None:
        super().__init__()
        width = int(width)
        self.stem = nn.Conv2d(int(in_channels), width, 3, padding=1)
        stages: list[nn.Module] = []
        for _ in range(4):
            stages.extend(self.Block(width) for _ in range(int(blocks)))
            stages.extend(
                [
                    nn.Conv2d(width, width * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.SiLU(),
                ]
            )
        self.stages = nn.Sequential(*stages)
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, q2_hat: torch.Tensor) -> torch.Tensor:
        residual = 0.5 * torch.tanh(self.head(self.stages(F.silu(self.stem(q2_hat)))))
        # decode_receiver retains its ordinary [0,1] u2 contract.  The paired
        # combiner below recenters this signal without seeing sender tensors.
        return 0.5 + residual


class ReceiverResidualSignalCombiner(nn.Module):
    """Fixed x1 anchor for the centered residual emitted by ReceiverFSQResidualD2."""

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        return (x1 + (u2 - 0.5)).clamp(0.0, 1.0)


def _normalization_groups(channels: int, maximum: int = 8) -> int:
    """Return a valid, small GroupNorm group count for a receiver branch."""

    for groups in range(min(int(channels), int(maximum)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class ReceiverFSQQOnlyHighResolutionResidualD2(nn.Module):
    """Preserve a validated FSQ D2 and add a strictly q2_hat-only RGB residual.

    The base is a private copy of the ordinary receiver D2.  The added branch
    never sees ``x1``/``z1`` (nor any sender tensor), and its RGB head is zero
    initialized.  Loading a legacy receiver checkpoint therefore gives exactly
    the legacy ``q2_hat -> D2`` output at startup, while subsequent training
    can use the extra high-resolution q2_hat capacity.

    ``requires_qonly_input`` is consumed by :func:`decode_receiver` and turns
    a conditional tokenizer/D2 concatenation into a hard error rather than an
    accidental relaxation of the required q2_hat-only D2 contract.
    """

    requires_qonly_input = True

    class Block(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.norm = nn.GroupNorm(_normalization_groups(int(channels)), int(channels))
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

    def __init__(
        self,
        base_d2: nn.Module,
        in_channels: int,
        *,
        width: int = 64,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        self.base = copy.deepcopy(base_d2)
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.blocks_per_scale = int(blocks)
        if self.width < 8:
            raise ValueError(f"q-only highres receiver D2 width must be >=8, got {self.width}")
        if self.blocks_per_scale < 1:
            raise ValueError(
                "q-only highres receiver D2 blocks per scale must be positive, "
                f"got {self.blocks_per_scale}"
            )
        self.stem = nn.Conv2d(self.in_channels, self.width, 3, padding=1)
        stages: list[nn.Module] = []
        dilations = (1, 2, 4, 1)
        for scale in range(4):
            stages.extend(
                self.Block(self.width, dilations[(scale + block) % len(dilations)])
                for block in range(self.blocks_per_scale)
            )
            stages.extend(
                [
                    nn.Conv2d(self.width, self.width * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.SiLU(),
                ]
            )
        self.stages = nn.Sequential(*stages)
        self.residual_head = nn.Conv2d(self.width, 3, 3, padding=1)
        self.reset_new_branch_to_zero()

    def reset_new_branch_to_zero(self) -> None:
        """Make a legacy-D2 initialization exactly output-equivalent."""

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, q2_hat: torch.Tensor) -> torch.Tensor:
        """Decode exactly one argument: generated receiver ``q2_hat``."""

        base = self.base(q2_hat)
        residual = self.residual_head(self.stages(F.silu(self.stem(q2_hat))))
        if tuple(residual.shape[-2:]) != tuple(base.shape[-2:]):
            residual = F.interpolate(
                residual,
                size=tuple(base.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
        return base + residual


def assert_qonly_d2_contract(tokenizer: nn.Module, receiver_d2: nn.Module) -> None:
    """Check the strict ``q2_hat -> D2`` interface of an opt-in D2 path."""

    if not bool(getattr(receiver_d2, "requires_qonly_input", False)):
        return
    parameters = list(inspect.signature(receiver_d2.forward).parameters.values())
    names = [parameter.name for parameter in parameters]
    if names != ["q2_hat"]:
        raise AssertionError(
            "strict q-only receiver D2 must expose forward(q2_hat), "
            f"got forward({', '.join(names)})"
        )
    # Layer3FSQTokenizer uses these modules to form the D2 side condition.
    # A non-None one means decode_receiver would concatenate x1/z1 features
    # into D2, which is intentionally prohibited for this new path.
    condition_modules = {
        name: getattr(tokenizer, name, None)
        for name in ("x1_cond", "z1_cond")
    }
    active = [name for name, module in condition_modules.items() if module is not None]
    if active:
        raise ValueError(
            "strict q-only receiver D2 requires tokenizer.condition(x1,z1) to be None; "
            f"active tokenizer condition modules={active}"
        )


def build_receiver_d2(
    args: argparse.Namespace,
    oracle_d2: nn.Module,
    device: torch.device,
    *,
    oracle_fsq_d: int,
) -> nn.Module:
    if str(args.receiver_d2_arch) == "qonly-highres-residual":
        return ReceiverFSQQOnlyHighResolutionResidualD2(
            oracle_d2,
            int(oracle_fsq_d),
            width=int(args.receiver_d2_width),
            blocks=int(args.receiver_d2_blocks),
        ).to(device)
    if str(args.receiver_d2_arch) == "residual-upsampler":
        return ReceiverFSQResidualD2(
            int(oracle_fsq_d),
            width=int(args.receiver_d2_width),
            blocks=int(args.receiver_d2_blocks),
        ).to(device)
    if bool(args.independent_receiver_d2):
        return copy.deepcopy(oracle_d2).to(device)
    return oracle_d2


def build_receiver_combiner(
    args: argparse.Namespace,
    oracle_combiner: nn.Module,
    device: torch.device,
) -> nn.Module:
    if str(args.receiver_combiner) == "oracle":
        receiver_combiner = (
            copy.deepcopy(oracle_combiner).to(device)
            if bool(args.independent_receiver_d2)
            else oracle_combiner
        )
        if bool(args.finetune_d2):
            receiver_combiner.requires_grad_(True)
        return receiver_combiner
    if str(args.receiver_combiner) == "qonly-residual":
        return ReceiverQOnlyResidualCombiner(oracle_combiner).to(device)
    if str(args.receiver_combiner) == "qonly-unet":
        return ReceiverQOnlyUNetCombiner(
            width=int(args.receiver_unet_width), blocks=int(args.receiver_unet_blocks)
        ).to(device)
    if str(args.receiver_combiner) == "qgated-refiner":
        return ReceiverQGatedRefinerCombiner(
            oracle_combiner,
            width=int(args.receiver_unet_width),
            blocks=int(args.receiver_unet_blocks),
        ).to(device)
    if str(args.receiver_combiner) == "residual-signal":
        return ReceiverResidualSignalCombiner().to(device)
    return ReceiverResidualCombiner(oracle_combiner).to(device)


def assert_receiver_topology(
    args: argparse.Namespace,
    bundle: direct.DirectBundle,
    receiver_d2: nn.Module,
    receiver_combiner: nn.Module,
) -> None:
    assert_qonly_d2_contract(bundle.tokenizer, receiver_d2)
    if not bool(args.independent_receiver_d2):
        if receiver_d2 is not bundle.tokenizer.d3:
            raise AssertionError("shared receiver mode must reuse sender D2")
        return
    if receiver_d2 is bundle.tokenizer.d3:
        raise AssertionError("independent receiver D2 unexpectedly aliases sender D2")
    if receiver_combiner is bundle.combiner:
        raise AssertionError("independent receiver combiner unexpectedly aliases sender combiner")
    sender_ids = {
        id(parameter)
        for module in (bundle.tokenizer.d3, bundle.combiner)
        for parameter in module.parameters()
    }
    receiver_ids = {
        id(parameter)
        for module in (receiver_d2, receiver_combiner)
        for parameter in module.parameters()
    }
    overlap = sender_ids & receiver_ids
    if overlap:
        raise AssertionError(f"independent receiver modules share {len(overlap)} parameters with sender modules")
    sender_trainable = [
        parameter
        for module in (bundle.tokenizer.d3, bundle.combiner)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if sender_trainable and not bool(getattr(args, "sender_aligned_q", False)):
        raise AssertionError(
            "independent receiver mode requires frozen sender D2/combiner; "
            f"found {len(sender_trainable)} trainable tensors"
        )
    if bool(getattr(args, "sender_aligned_q", False)) and not sender_trainable:
        raise AssertionError("--sender-aligned-q requires a trainable sender D2/combiner")


def decode_receiver(
    tokenizer: nn.Module,
    receiver_d2: nn.Module,
    q2_hat: torch.Tensor,
    x1: torch.Tensor,
    z1: torch.Tensor,
    receiver_combiner: nn.Module,
) -> dict[str, torch.Tensor]:
    """Decode a receiver prediction without consulting sender-only tensors."""
    condition = tokenizer.condition(x1, z1)
    if bool(getattr(receiver_d2, "requires_qonly_input", False)) and condition is not None:
        raise AssertionError(
            "strict q-only receiver D2 received a tokenizer condition; "
            "D2 must be called only as D2(q2_hat)"
        )
    d2_input = q2_hat if condition is None else torch.cat([q2_hat, condition], dim=1)
    u2_raw = receiver_d2(d2_input)
    u2_hat = u2_raw.clamp(0.0, 1.0)
    final = receiver_combiner(x1, u2_hat)
    return {
        "d3_in": d2_input,
        "u2_raw": u2_raw,
        "u2_hat": u2_hat,
        "final": final,
    }


def prediction_index_loss(
    prediction: ReceiverPrediction,
    target_codes: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    if not prediction.logits:
        return target_codes.new_zeros((), dtype=torch.float32)
    if prediction.joint_indices is not None:
        return F.cross_entropy(prediction.logits[0].float(), target_indices.long())
    losses = [
        F.cross_entropy(logits.float(), target_codes[:, channel].long())
        for channel, logits in enumerate(prediction.logits)
    ]
    return torch.stack(losses).mean()


def compute_losses(
    prediction: ReceiverPrediction,
    target_q: torch.Tensor,
    target_codes: torch.Tensor,
    target_indices: torch.Tensor,
    predicted_final: torch.Tensor,
    oracle_final: torch.Tensor,
    imgs: torch.Tensor,
    x1: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    loss_q = F.mse_loss(prediction.q_continuous.float(), target_q.float())
    loss_index = prediction_index_loss(prediction, target_codes, target_indices)
    loss_flow = (
        prediction.loss_flow
        if prediction.loss_flow is not None
        else loss_q.new_zeros(())
    )
    loss_flow_mse = (
        prediction.loss_flow_mse
        if prediction.loss_flow_mse is not None
        else loss_q.new_zeros(())
    )
    loss_flow_cosine = (
        prediction.loss_flow_cosine
        if prediction.loss_flow_cosine is not None
        else loss_q.new_zeros(())
    )
    final_mse_per_image = (
        predicted_final.float() - imgs.float()
    ).square().flatten(1).mean(dim=1)
    if float(args.hard_example_power) > 0.0:
        baseline_mse = (x1.float() - imgs.float()).square().flatten(1).mean(dim=1).detach()
        hard_weights = baseline_mse.pow(float(args.hard_example_power))
        hard_weights = hard_weights / hard_weights.mean().clamp_min(1e-8)
        hard_weights = hard_weights.clamp(
            min=float(args.hard_example_min_weight),
            max=float(args.hard_example_max_weight),
        )
    else:
        hard_weights = torch.ones_like(final_mse_per_image)
    loss_final = (hard_weights * final_mse_per_image).mean()
    loss_oracle = F.mse_loss(predicted_final.float(), oracle_final.float())
    loss = (
        float(getattr(args, "lambda_flow", 0.0)) * loss_flow
        + float(args.lambda_q) * loss_q
        + float(args.lambda_index) * loss_index
        + float(args.lambda_final) * loss_final
        + float(args.lambda_oracle) * loss_oracle
    )
    return {
        "loss": loss,
        "loss_q": loss_q,
        "loss_flow": loss_flow,
        "loss_flow_mse": loss_flow_mse,
        "loss_flow_cosine": loss_flow_cosine,
        "loss_index": loss_index,
        "loss_final": loss_final,
        "loss_oracle": loss_oracle,
    }


def compute_residual_phase_losses(
    prediction: ReceiverPrediction,
    oracle: dict[str, torch.Tensor],
    decoded: dict[str, torch.Tensor],
    condition: ReceiverCondition,
    imgs: torch.Tensor,
    bundle: direct.DirectBundle,
    receiver_d2: nn.Module,
    receiver_combiner: nn.Module,
    args: argparse.Namespace,
    *,
    phase: int,
    train: bool,
) -> dict[str, torch.Tensor]:
    q_true = oracle["q3_hard"].detach()
    q_hat = prediction.q_train if train else (
        prediction.q_hard if bool(args.hard_fsq) else prediction.q_continuous
    )
    loss_q = F.mse_loss(prediction.q_continuous.float(), q_true.float())
    loss_final = F.mse_loss(decoded["final"].float(), imgs.float())
    zero = loss_final.new_zeros(())
    losses: dict[str, torch.Tensor] = {
        "loss_q": loss_q,
        "loss_final": loss_final,
        "loss_index": zero,
        "loss_oracle": F.mse_loss(decoded["final"].float(), oracle["final"].detach().float()),
        "loss_receiver_true": zero,
        "loss_receiver_mid": zero,
        "loss_receiver_consistency": zero,
        "residual_phase": loss_final.new_tensor(float(phase)),
    }
    if int(phase) == 1:
        losses["loss"] = (
            float(args.residual_phase1_lambda_q) * loss_q
            + float(args.residual_phase1_lambda_final) * loss_final
        )
        return losses
    q_mid = 0.5 * (q_hat + q_true)
    decoded_true = decode_receiver(
        bundle.tokenizer,
        receiver_d2,
        q_true,
        condition.x1,
        condition.z1,
        receiver_combiner,
    )
    decoded_mid = decode_receiver(
        bundle.tokenizer,
        receiver_d2,
        q_mid,
        condition.x1,
        condition.z1,
        receiver_combiner,
    )
    loss_receiver_true = F.mse_loss(decoded_true["final"].float(), imgs.float())
    loss_receiver_mid = F.mse_loss(decoded_mid["final"].float(), imgs.float())
    loss_receiver_consistency = F.mse_loss(
        decoded["final"].float(),
        decoded_true["final"].detach().float(),
    )
    losses["loss_receiver_true"] = loss_receiver_true
    losses["loss_receiver_mid"] = loss_receiver_mid
    losses["loss_receiver_consistency"] = loss_receiver_consistency
    losses["loss"] = (
        float(args.residual_phase2_lambda_final) * loss_final
        + float(args.residual_phase2_lambda_true) * loss_receiver_true
        + float(args.residual_phase2_lambda_mid) * loss_receiver_mid
        + float(args.residual_phase2_lambda_consistency) * loss_receiver_consistency
        + float(args.residual_phase2_lambda_q) * loss_q
    )
    return losses


def receiver_forward(
    predictor: nn.Module,
    bundle: direct.DirectBundle,
    receiver_d2: nn.Module,
    receiver_combiner: nn.Module,
    imgs: torch.Tensor,
    *,
    train: bool,
    joint_predictable_oracle: bool,
    ar_history_corruption_prob: float = 0.0,
    ar_rollout_history_batch: int = 0,
) -> tuple[
    ReceiverCondition,
    ReceiverPrediction,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    # Sender-only oracle computation is isolated and has no gradient path into
    # the receiver predictor.  At deployment this entire block is absent.
    with torch.no_grad():
        layer1 = bundle.layer1(imgs)
    if bool(joint_predictable_oracle):
        oracle = bundle.tokenizer(imgs, layer1["x1"], layer1["z1"], bundle.combiner)
    else:
        with torch.no_grad():
            oracle = bundle.tokenizer(imgs, layer1["x1"], layer1["z1"], bundle.combiner)
    condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
    if train and hasattr(predictor, "forward_teacher"):
        # AR teacher forcing is a training-only loss path.  The public
        # deployment forward remains forward(condition) and performs greedy
        # generation without img/z2/q2/oracle indices.
        teacher_target = (
            oracle["q3_hard"].detach()
            if str(getattr(predictor, "teacher_target", "indices")) == "q2"
            else oracle["idx3"].detach()
        )
        teacher_history = teacher_target
        corruption_prob = float(ar_history_corruption_prob)
        if corruption_prob > 0.0:
            if not isinstance(predictor, AutoregressiveFSQJointTokenPredictor):
                raise ValueError(
                    "AR history corruption currently requires --route ar_joint_index"
                )
            # Scheduled-sampling histories must come from the same recursive
            # greedy rollout used at deployment.  A parallel teacher-prefix
            # proposal is not equivalent: every proposed site would still
            # have observed the oracle prefix and the exposure gap would stay
            # hidden.  Limit the rollout to a configurable sub-batch because
            # a 16x16 raster requires 256 causal model evaluations.
            rollout_batch = min(
                int(teacher_target.shape[0]),
                max(1, int(ar_rollout_history_batch)),
            )
            with torch.no_grad():
                rollout_history = predictor.greedy_rollout_indices(
                    condition,
                    batch_limit=rollout_batch,
                )
            corrupt = (
                torch.rand_like(teacher_target[:rollout_batch].float())
                < corruption_prob
            )
            teacher_history = teacher_target.clone()
            teacher_history[:rollout_batch] = torch.where(
                corrupt,
                rollout_history.detach(),
                teacher_target[:rollout_batch],
            )
        prediction = predictor.forward_teacher(condition, teacher_history)
    else:
        prediction = predictor(condition)
    q_used = prediction.q_train if train else (
        prediction.q_hard if bool(getattr(predictor, "hard_fsq", True)) else prediction.q_continuous
    )
    decoded = decode_receiver(
        bundle.tokenizer,
        receiver_d2,
        q_used,
        condition.x1,
        condition.z1,
        receiver_combiner,
    )
    return condition, prediction, oracle, decoded


def update_metrics(
    metrics: MetricSums,
    losses: dict[str, torch.Tensor],
    prediction: ReceiverPrediction,
    oracle: dict[str, torch.Tensor],
    decoded: dict[str, torch.Tensor],
    condition: ReceiverCondition,
    imgs: torch.Tensor,
    *,
    prediction_prefix: str = "",
) -> None:
    batch = int(imgs.shape[0])
    for name, value in losses.items():
        metrics.add(name, float(value.detach().item()), batch)
    psnr_x1 = float(base.batch_metric_mean(base.psnr_per_image(condition.x1, imgs)))
    psnr_oracle = float(base.batch_metric_mean(base.psnr_per_image(oracle["final"], imgs)))
    psnr_pred = float(base.batch_metric_mean(base.psnr_per_image(decoded["final"], imgs)))
    metrics.add("psnr_x1", psnr_x1, batch)
    metrics.add("psnr_oracle", psnr_oracle, batch)
    metrics.add(f"{prediction_prefix}psnr_pred", psnr_pred, batch)
    metrics.add(f"{prediction_prefix}delta_x1", psnr_pred - psnr_x1, batch)
    metrics.add("delta_oracle", psnr_oracle - psnr_x1, batch)
    metrics.add(f"{prediction_prefix}gap_oracle", psnr_oracle - psnr_pred, batch)
    metrics.add(
        f"{prediction_prefix}q_mse_hard",
        float(F.mse_loss(prediction.q_hard, oracle["q3_hard"]).item()),
        batch,
    )
    if prediction.q_base is not None:
        metrics.add(
            "q_base_mse",
            float(F.mse_loss(prediction.q_base.float(), oracle["q3_hard"].float()).item()),
            batch,
        )
    if prediction.q_residual is not None:
        metrics.add("q_residual_abs_mean", float(prediction.q_residual.float().abs().mean().item()), batch)
        metrics.add("q_residual_rms", float(prediction.q_residual.float().square().mean().sqrt().item()), batch)
    if prediction.joint_indices is not None:
        accuracy = (prediction.joint_indices == oracle["idx3"]).float().mean()
        metrics.add(f"{prediction_prefix}index_accuracy", float(accuracy.item()), batch)
    elif prediction.codes is not None:
        accuracy = (prediction.codes == oracle["codes"]).float().mean()
        metrics.add(f"{prediction_prefix}index_accuracy", float(accuracy.item()), batch)


@torch.no_grad()
def update_train_deploy_metrics(
    metrics: MetricSums,
    predictor: nn.Module,
    bundle: direct.DirectBundle,
    receiver_d2: nn.Module,
    receiver_combiner: nn.Module,
    condition: ReceiverCondition,
    oracle: dict[str, torch.Tensor],
    imgs: torch.Tensor,
    *,
    batch_limit: int,
) -> None:
    """Measure a small train subset through the exact public inference path.

    These diagnostics are deliberately separate from differentiable
    teacher/short-endpoint losses.  They consume only z1/x1 and therefore have
    the same information set and hard/continuous choice as strict validation.
    """

    selected = min(int(imgs.shape[0]), max(1, int(batch_limit)))
    deploy_condition = make_receiver_condition(
        condition.z1[:selected],
        condition.x1[:selected],
        detach=True,
    )
    was_training = bool(predictor.training)
    predictor.eval()
    try:
        deploy_prediction = predictor(deploy_condition)
    finally:
        predictor.train(was_training)
    deploy_q = (
        deploy_prediction.q_hard
        if bool(getattr(predictor, "hard_fsq", True))
        else deploy_prediction.q_continuous
    )
    deploy_decoded = decode_receiver(
        bundle.tokenizer,
        receiver_d2,
        deploy_q,
        deploy_condition.x1,
        deploy_condition.z1,
        receiver_combiner,
    )
    target_imgs = imgs[:selected]
    target_q = oracle["q3_hard"][:selected]
    psnr_x1 = float(base.batch_metric_mean(base.psnr_per_image(deploy_condition.x1, target_imgs)))
    psnr_pred = float(base.batch_metric_mean(base.psnr_per_image(deploy_decoded["final"], target_imgs)))
    metrics.add("train_deploy_psnr_pred", psnr_pred, selected)
    metrics.add("train_deploy_delta_x1", psnr_pred - psnr_x1, selected)
    metrics.add(
        "train_deploy_q_mse_hard",
        float(F.mse_loss(deploy_prediction.q_hard, target_q).item()),
        selected,
    )
    if deploy_prediction.joint_indices is not None:
        accuracy = (
            deploy_prediction.joint_indices == oracle["idx3"][:selected]
        ).float().mean()
    elif deploy_prediction.codes is not None:
        accuracy = (
            deploy_prediction.codes == oracle["codes"][:selected]
        ).float().mean()
    else:
        accuracy = None
    if accuracy is not None:
        metrics.add("train_deploy_index_accuracy", float(accuracy.item()), selected)
    metrics.add("train_deploy_evaluated_images", float(selected), 1)


def run_loader(
    loader,
    *,
    predictor: nn.Module,
    bundle: direct.DirectBundle,
    receiver_d2: nn.Module,
    receiver_combiner: nn.Module,
    optimizer: optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
    train: bool,
    residual_phase: int = 0,
) -> dict[str, float]:
    predictor.train(train and not bool(getattr(args, "sender_aligned_q", False)))
    for module in (bundle.e1, bundle.d1, bundle.tokenizer, bundle.combiner):
        module.eval()
    sender_training = bool(args.joint_predictable_oracle) or bool(
        getattr(args, "sender_aligned_q", False)
    )
    if sender_training:
        bundle.tokenizer.e3.train(train)
        bundle.tokenizer.quantizer.train(train)
        bundle.combiner.train(train and any(parameter.requires_grad for parameter in bundle.combiner.parameters()))
    if bool(getattr(args, "sender_aligned_q", False)):
        bundle.tokenizer.d3.train(
            train and any(parameter.requires_grad for parameter in bundle.tokenizer.d3.parameters())
        )
    receiver_d2.train(train and any(parameter.requires_grad for parameter in receiver_d2.parameters()))
    receiver_combiner.train(train and any(parameter.requires_grad for parameter in receiver_combiner.parameters()))
    # During the optional highres-D2 warmup, preserve the validated v5
    # predictor/base-D2/combiner not only by disabling gradients but also by
    # keeping their BatchNorm/dropout state in eval mode.  The new q-only
    # branch itself remains in train mode through its parent receiver_d2.
    if train and bool(getattr(args, "_qonly_highres_warmup_active", False)):
        predictor.eval()
        if not isinstance(receiver_d2, ReceiverFSQQOnlyHighResolutionResidualD2):
            raise AssertionError("qonly highres warmup active with an incompatible receiver D2")
        receiver_d2.base.eval()
        receiver_combiner.eval()
    metrics = MetricSums()
    maximum = int(args.max_train_batches if train else args.max_val_batches)
    audited = False
    image_count = 0
    batch_count = 0
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if maximum > 0 and batch_index > maximum:
            break
        imgs = imgs.to(device, non_blocking=True)
        image_count += int(imgs.shape[0])
        batch_count += 1
        condition, prediction, oracle, decoded = receiver_forward(
            predictor,
            bundle,
            receiver_d2,
            receiver_combiner,
            imgs,
            train=train,
            joint_predictable_oracle=sender_training,
            ar_history_corruption_prob=(
                float(getattr(args, "_ar_history_corruption_prob", 0.0))
                if train
                else 0.0
            ),
            ar_rollout_history_batch=(
                int(getattr(args, "ar_rollout_history_batch", 2))
                if train
                else 0
            ),
        )
        if not audited:
            assert_training_targets_are_not_inputs(
                predictor,
                condition,
                source_targets={
                    "img": imgs,
                    "oracle_q2": oracle["q3_hard"],
                    "oracle_indices": oracle["codes"],
                    "z2": oracle["z3"],
                },
            )
            audited = True
        losses = compute_losses(
            prediction,
            oracle["q3_hard"].detach(),
            oracle["codes"].detach(),
            oracle["idx3"].detach(),
            decoded["final"],
            oracle["final"].detach(),
            imgs,
            condition.x1,
            args,
        )
        if train and float(getattr(args, "_ar_history_corruption_prob", 0.0)) > 0.0:
            losses["ar_history_corruption_prob"] = losses["loss"].new_tensor(
                float(args._ar_history_corruption_prob)
            )
        if bool(args.residual_q):
            losses = compute_residual_phase_losses(
                prediction,
                oracle,
                decoded,
                condition,
                imgs,
                bundle,
                receiver_d2,
                receiver_combiner,
                args,
                phase=int(residual_phase),
                train=train,
            )
        if sender_training:
            loss_sender_final = F.mse_loss(oracle["final"].float(), imgs.float())
            loss_sender_predictability = F.mse_loss(
                oracle["q3"].float(),
                prediction.q_continuous.detach().float(),
            )
            losses["loss_sender_final"] = loss_sender_final
            losses["loss_sender_predictability"] = loss_sender_predictability
            if bool(getattr(args, "sender_aligned_q", False)):
                # v9 changes the sender representation instead of asking G0
                # to recover sender-only information.  G0 is frozen, so qhat
                # is a receiver-only, stop-gradient alignment target.
                sender_terms: list[torch.Tensor] = []
                if float(args.sender_align_lambda_final) > 0:
                    sender_terms.append(float(args.sender_align_lambda_final) * loss_sender_final)
                if float(args.sender_align_lambda_predictability) > 0:
                    sender_terms.append(
                        float(args.sender_align_lambda_predictability)
                        * loss_sender_predictability
                    )
                loss_sender_qhat_final = loss_sender_final.new_zeros(())
                if float(args.sender_align_lambda_qhat_final) > 0:
                    # Direct deployability loss.  The explicit detach is the
                    # gradient boundary: this term may train only the current
                    # sender D2/combiner.  Sender E2/FSQ and G0 cannot receive
                    # gradients from img through this path.
                    qhat_detached = prediction.q_continuous.detach()
                    sender_qhat_final = bundle.tokenizer.decode(
                        qhat_detached,
                        condition.x1,
                        condition.z1,
                        bundle.combiner,
                    )["final"]
                    loss_sender_qhat_final = F.mse_loss(
                        sender_qhat_final.float(),
                        imgs.float(),
                    )
                    sender_terms.append(
                        float(args.sender_align_lambda_qhat_final)
                        * loss_sender_qhat_final
                    )
                losses["loss_sender_qhat_final"] = loss_sender_qhat_final
                if bool(args.finetune_d2) and float(args.sender_align_lambda_receiver_final) > 0:
                    sender_terms.append(
                        float(args.sender_align_lambda_receiver_final) * losses["loss_final"]
                    )
                if not sender_terms:
                    raise RuntimeError("sender-aligned-q has no active loss term")
                losses["loss"] = torch.stack(sender_terms).sum()
            else:
                losses["loss"] = (
                    losses["loss"]
                    + float(args.lambda_sender_final) * loss_sender_final
                    + float(args.lambda_sender_predictability) * loss_sender_predictability
                )
        if train and (float(args.lambda_zero_anchor) > 0 or float(args.lambda_shuffle_anchor) > 0):
            q_train = prediction.q_train
            loss_zero_anchor = losses["loss"].new_zeros(())
            loss_shuffle_anchor = losses["loss"].new_zeros(())
            if float(args.lambda_zero_anchor) > 0:
                zero_final = decode_receiver(
                    bundle.tokenizer,
                    receiver_d2,
                    torch.zeros_like(q_train),
                    condition.x1,
                    condition.z1,
                    receiver_combiner,
                )["final"]
                loss_zero_anchor = F.mse_loss(zero_final.float(), condition.x1.float())
            if float(args.lambda_shuffle_anchor) > 0:
                shuffled_final = decode_receiver(
                    bundle.tokenizer,
                    receiver_d2,
                    bundle.tokenizer.shuffle_q3(q_train),
                    condition.x1,
                    condition.z1,
                    receiver_combiner,
                )["final"]
                loss_shuffle_anchor = F.mse_loss(shuffled_final.float(), condition.x1.float())
            losses["loss_zero_anchor"] = loss_zero_anchor
            losses["loss_shuffle_anchor"] = loss_shuffle_anchor
            losses["loss"] = (
                losses["loss"]
                + float(args.lambda_zero_anchor) * loss_zero_anchor
                + float(args.lambda_shuffle_anchor) * loss_shuffle_anchor
            )
        if train:
            if optimizer is None:
                raise RuntimeError("training requires an optimizer")
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if float(args.grad_clip_norm) > 0:
                unique: dict[int, nn.Parameter] = {}
                for module in (predictor, bundle.tokenizer, receiver_d2, receiver_combiner):
                    for parameter in module.parameters():
                        if parameter.requires_grad:
                            unique[id(parameter)] = parameter
                trainable = list(unique.values())
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip_norm))
            optimizer.step()
        prediction_prefix = ""
        if train and hasattr(predictor, "forward_teacher"):
            route = str(getattr(args, "route", ""))
            if route == "flow_matching":
                prediction_prefix = "train_endpoint_"
            elif route == "categorical_diffusion":
                prediction_prefix = "train_teacher_"
            else:
                prediction_prefix = "train_proxy_"
        update_metrics(
            metrics,
            losses,
            prediction,
            oracle,
            decoded,
            condition,
            imgs,
            prediction_prefix=prediction_prefix,
        )
        if (
            train
            and hasattr(predictor, "forward_teacher")
            and batch_index <= int(getattr(args, "train_deploy_metric_batches", 0))
        ):
            update_train_deploy_metrics(
                metrics,
                predictor,
                bundle,
                receiver_d2,
                receiver_combiner,
                condition,
                oracle,
                imgs,
                batch_limit=int(getattr(args, "train_deploy_metric_batch_size", 2)),
            )
        if not train:
            q_eval = (
                prediction.q_hard
                if bool(getattr(predictor, "hard_fsq", True))
                else prediction.q_continuous
            )
            condition_permutation = torch.roll(
                torch.arange(int(imgs.shape[0]), device=imgs.device),
                shifts=1,
            )
            shuffled_condition = make_receiver_condition(
                condition.z1[condition_permutation],
                condition.x1[condition_permutation],
                detach=True,
            )
            shuffled_condition_prediction = predictor(shuffled_condition)
            shuffled_condition_q = (
                shuffled_condition_prediction.q_hard
                if bool(getattr(predictor, "hard_fsq", True))
                else shuffled_condition_prediction.q_continuous
            )
            condition_shuffled = decode_receiver(
                bundle.tokenizer,
                receiver_d2,
                shuffled_condition_q,
                condition.x1,
                condition.z1,
                receiver_combiner,
            )["final"]
            zero = decode_receiver(
                bundle.tokenizer,
                receiver_d2,
                torch.zeros_like(q_eval),
                condition.x1,
                condition.z1,
                receiver_combiner,
            )["final"]
            shuffled = decode_receiver(
                bundle.tokenizer,
                receiver_d2,
                bundle.tokenizer.shuffle_q3(q_eval),
                condition.x1,
                condition.z1,
                receiver_combiner,
            )["final"]
            predicted_psnr = base.psnr_per_image(decoded["final"], imgs)
            condition_shuffle_psnr = base.psnr_per_image(condition_shuffled, imgs)
            metrics.add(
                "psnr_condition_shuffle",
                float(condition_shuffle_psnr.mean()),
                int(imgs.shape[0]),
            )
            metrics.add(
                "condition_shuffle_drop",
                float((predicted_psnr - condition_shuffle_psnr).mean()),
                int(imgs.shape[0]),
            )
            metrics.add(
                "pred_drop_zero",
                float((predicted_psnr - base.psnr_per_image(zero, imgs)).mean()),
                int(imgs.shape[0]),
            )
            metrics.add(
                "pred_drop_shuffle",
                float((predicted_psnr - base.psnr_per_image(shuffled, imgs)).mean()),
                int(imgs.shape[0]),
            )
            # Keep sender usefulness separate from receiver usefulness.  This
            # prevents a jointly trained E2/FSQ from satisfying predictability
            # by collapsing q2 to an ignorable constant.
            oracle_q = oracle["q3"]
            oracle_zero = bundle.tokenizer.decode(
                torch.zeros_like(oracle_q),
                condition.x1,
                condition.z1,
                bundle.combiner,
            )["final"]
            oracle_shuffled = bundle.tokenizer.decode(
                bundle.tokenizer.shuffle_q3(oracle_q),
                condition.x1,
                condition.z1,
                bundle.combiner,
            )["final"]
            oracle_psnr = base.psnr_per_image(oracle["final"], imgs)
            metrics.add(
                "oracle_drop_zero",
                float((oracle_psnr - base.psnr_per_image(oracle_zero, imgs)).mean()),
                int(imgs.shape[0]),
            )
            metrics.add(
                "oracle_drop_shuffle",
                float((oracle_psnr - base.psnr_per_image(oracle_shuffled, imgs)).mean()),
                int(imgs.shape[0]),
            )
            if bool(getattr(args, "sender_aligned_q", False)):
                # Deployment audit: q_eval was produced above strictly from
                # ReceiverCondition(z1, x1).  Only after prediction do we use
                # the current sender D2/combiner; img/z2/qtrue never enter G0.
                sender_qhat = bundle.tokenizer.decode(
                    q_eval,
                    condition.x1,
                    condition.z1,
                    bundle.combiner,
                )["final"]
                sender_qhat_zero = bundle.tokenizer.decode(
                    torch.zeros_like(q_eval),
                    condition.x1,
                    condition.z1,
                    bundle.combiner,
                )["final"]
                sender_qhat_shuffled = bundle.tokenizer.decode(
                    bundle.tokenizer.shuffle_q3(q_eval),
                    condition.x1,
                    condition.z1,
                    bundle.combiner,
                )["final"]
                sender_qhat_psnr = base.psnr_per_image(sender_qhat, imgs)
                x1_psnr = base.psnr_per_image(condition.x1, imgs)
                metrics.add(
                    "psnr_sender_qhat",
                    float(sender_qhat_psnr.mean()),
                    int(imgs.shape[0]),
                )
                metrics.add(
                    "delta_sender_qhat_x1",
                    float((sender_qhat_psnr - x1_psnr).mean()),
                    int(imgs.shape[0]),
                )
                metrics.add(
                    "sender_qhat_drop_zero",
                    float(
                        (
                            sender_qhat_psnr
                            - base.psnr_per_image(sender_qhat_zero, imgs)
                        ).mean()
                    ),
                    int(imgs.shape[0]),
                )
                metrics.add(
                    "sender_qhat_drop_shuffle",
                    float(
                        (
                            sender_qhat_psnr
                            - base.psnr_per_image(sender_qhat_shuffled, imgs)
                        ).mean()
                    ),
                    int(imgs.shape[0]),
                )
    result = metrics.means()
    dataset_size = int(len(loader.dataset)) if hasattr(loader, "dataset") else image_count
    full_validation = bool(
        not train
        and int(args.max_val_batches) <= 0
        and image_count == dataset_size
    )
    result["evaluated_images"] = float(image_count)
    result["evaluated_batches"] = float(batch_count)
    result["full_validation"] = 1.0 if full_validation else 0.0
    result["goal_delta_db"] = float(args.goal_delta_db)
    result["goal_met"] = float(
        result.get("delta_x1", float("-inf")) >= float(args.goal_delta_db)
        and result.get("delta_oracle", float("-inf")) >= float(args.min_oracle_delta_db)
        and result.get("pred_drop_zero", float("-inf")) >= float(args.min_pred_ablation_drop)
        and result.get("pred_drop_shuffle", float("-inf")) >= float(args.min_pred_ablation_drop)
        and result.get("condition_shuffle_drop", float("-inf"))
        >= float(args.min_condition_shuffle_drop)
        and result.get("oracle_drop_zero", float("-inf")) >= float(args.min_oracle_ablation_drop)
        and result.get("oracle_drop_shuffle", float("-inf")) >= float(args.min_oracle_ablation_drop)
    )
    result["receiver_only_audit"] = 1.0 if audited else 0.0
    sender_quality_gate = bool(
        full_validation
        and result.get("delta_oracle", float("-inf")) >= float(args.sender_align_min_oracle_delta_db)
        and result.get("oracle_drop_zero", float("-inf"))
        >= float(args.sender_align_min_oracle_ablation_drop)
        and result.get("oracle_drop_shuffle", float("-inf"))
        >= float(args.sender_align_min_oracle_ablation_drop)
    )
    result["sender_quality_gate_met"] = 1.0 if sender_quality_gate else 0.0
    sender_deployment_gate = bool(
        bool(getattr(args, "sender_aligned_q", False))
        and full_validation
        and result.get("delta_sender_qhat_x1", float("-inf"))
        >= float(args.sender_deployment_min_delta_db)
        and result.get("sender_qhat_drop_zero", float("-inf"))
        >= float(args.sender_deployment_min_ablation_drop)
        and result.get("sender_qhat_drop_shuffle", float("-inf"))
        >= float(args.sender_deployment_min_ablation_drop)
    )
    result["sender_deployment_gate_met"] = 1.0 if sender_deployment_gate else 0.0
    result["sender_align_goal_met"] = float(
        bool(getattr(args, "sender_aligned_q", False))
        and sender_quality_gate
        and sender_deployment_gate
        and result.get("loss_sender_predictability", float("inf"))
        <= float(args.sender_align_max_q_mse)
    )
    # The deployable receiver path is the actual objective.  q-MSE remains a
    # required diagnostic/gate, but selecting the ordinary best checkpoint by
    # q-MSE can prefer a representation whose D2/combiner reconstructs worse.
    result["sender_align_selection_score"] = (
        float(result.get("delta_sender_qhat_x1", float("-inf")))
        if bool(getattr(args, "sender_aligned_q", False)) and sender_quality_gate
        else float("-inf")
    )
    return result


DISPLAY_KEYS = (
    "loss",
    "ar_history_corruption_prob",
    "loss_q",
    "loss_flow",
    "loss_flow_mse",
    "loss_flow_cosine",
    "loss_index",
    "loss_final",
    "loss_oracle",
    "loss_zero_anchor",
    "loss_shuffle_anchor",
    "loss_sender_final",
    "loss_sender_predictability",
    "loss_sender_qhat_final",
    "loss_receiver_true",
    "loss_receiver_mid",
    "loss_receiver_consistency",
    "residual_phase",
    "psnr_x1",
    "psnr_oracle",
    "psnr_pred",
    "delta_x1",
    "delta_oracle",
    "gap_oracle",
    "q_mse_hard",
    "q_base_mse",
    "q_residual_abs_mean",
    "q_residual_rms",
    "index_accuracy",
    "train_proxy_psnr_pred",
    "train_proxy_delta_x1",
    "train_proxy_gap_oracle",
    "train_proxy_q_mse_hard",
    "train_proxy_index_accuracy",
    "train_endpoint_psnr_pred",
    "train_endpoint_delta_x1",
    "train_endpoint_gap_oracle",
    "train_endpoint_q_mse_hard",
    "train_endpoint_index_accuracy",
    "train_teacher_psnr_pred",
    "train_teacher_delta_x1",
    "train_teacher_gap_oracle",
    "train_teacher_q_mse_hard",
    "train_teacher_index_accuracy",
    "train_deploy_psnr_pred",
    "train_deploy_delta_x1",
    "train_deploy_q_mse_hard",
    "train_deploy_index_accuracy",
    "train_deploy_evaluated_images",
    "psnr_condition_shuffle",
    "condition_shuffle_drop",
    "pred_drop_zero",
    "pred_drop_shuffle",
    "oracle_drop_zero",
    "oracle_drop_shuffle",
    "psnr_sender_qhat",
    "delta_sender_qhat_x1",
    "sender_qhat_drop_zero",
    "sender_qhat_drop_shuffle",
    "evaluated_images",
    "full_validation",
    "sender_quality_gate_met",
    "sender_deployment_gate_met",
    "sender_align_goal_met",
    "goal_met",
    "receiver_only_audit",
)


def display(metrics: dict[str, float]) -> dict[str, float]:
    return {name: metrics[name] for name in DISPLAY_KEYS if name in metrics}


def checkpoint_selection_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    if bool(args.sender_aligned_q):
        return float(metrics.get("sender_align_selection_score", float("-inf")))
    return float(metrics.get("psnr_pred", float("-inf")))


def clean_args(args: argparse.Namespace) -> dict:
    return {name: value for name, value in vars(args).items() if not name.startswith("_")}


def experiment_name(args: argparse.Namespace, oracle_args: argparse.Namespace) -> str:
    discrete_routes = {
        "parallel_index",
        "joint_index",
        "ar_index",
        "ar_joint_index",
        "ar_residual_index",
    }
    hard = (
        "hard"
        if bool(args.hard_fsq) or str(args.route) in discrete_routes
        else "continuous"
    )
    topology = "_independent-d2" if bool(args.independent_receiver_d2) else ""
    residual = (
        f"_residual-q-h{int(args.residual_hidden)}-b{int(args.residual_blocks)}"
        if bool(args.residual_q)
        else ""
    )
    sender_align = "_sender-aligned-q" if bool(args.sender_aligned_q) else ""
    deploy = (
        f"_d2ft-{args.receiver_combiner}"
        if bool(args.finetune_d2) or str(args.receiver_combiner) != "oracle"
        else "_frozen"
    )
    return (
        f"fsq_receiver_{args.route}_{args.condition_mode}_{hard}_"
        f"{oracle_args.arch}_d{int(oracle_args.fsq_d)}_l{str(oracle_args.fsq_levels).replace(',', 'x')}"
        f"{deploy}{topology}{residual}{sender_align}"
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    predictor: nn.Module,
    receiver_d2: nn.Module,
    oracle_tokenizer: nn.Module,
    oracle_combiner: nn.Module,
    receiver_combiner: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    oracle_args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    strict_qonly_d2 = bool(getattr(receiver_d2, "requires_qonly_input", False))
    tokenizer_condition_modules = {
        name: bool(getattr(oracle_tokenizer, name, None) is not None)
        for name in ("x1_cond", "z1_cond")
    }
    torch.save(
        {
            "stage": "explore2_fsq_receiver",
            "epoch": int(epoch),
            "metrics": metrics,
            "selection_score": checkpoint_selection_score(metrics, args),
            "args": clean_args(args),
            "oracle_args": vars(oracle_args),
            "oracle_checkpoint": str(resolve_path(args.oracle_checkpoint)),
            "predictor_state_dict": predictor.state_dict(),
            "receiver_d2_state_dict": receiver_d2.state_dict(),
            "oracle_tokenizer_state_dict": oracle_tokenizer.state_dict(),
            "oracle_combiner_state_dict": oracle_combiner.state_dict(),
            "receiver_combiner_state_dict": receiver_combiner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "receiver_topology": {
                "independent_receiver_d2": bool(args.independent_receiver_d2),
                "receiver_combiner_isolated": bool(args.independent_receiver_d2)
                or str(args.receiver_combiner) != "oracle",
                "receiver_d2_arch": str(args.receiver_d2_arch),
                "receiver_d2_input_contract": (
                    "q2_hat_only" if strict_qonly_d2 else "q2_hat_plus_tokenizer_condition_if_configured"
                ),
                "tokenizer_condition_modules": tokenizer_condition_modules,
                "strict_qonly_d2_verified": strict_qonly_d2,
                "residual_q": bool(args.residual_q),
                "sender_aligned_q": bool(args.sender_aligned_q),
                "sender_receiver_decoder_disjoint": bool(args.independent_receiver_d2),
                "sender_decoder_trainable": bool(args.sender_aligned_q),
            },
            "receiver_contract": {
                "inputs": ["z1", "x1"],
                "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
                "output": "q2_hat",
                "receiver_decode": (
                    "q2_hat -> D2(q2_hat) -> u2_hat -> combiner(x1,u2_hat)"
                    if strict_qonly_d2
                    else "q2_hat -> D2(q2_hat[, tokenizer_condition]) -> u2_hat -> combiner(x1,u2_hat)"
                ),
                "train_transform": "RandomCrop(256)+RandomHorizontalFlip+ToTensor",
                "validation_transform": "CenterCrop(256)+ToTensor",
            },
            "sender_deployment_stack": {
                "predictor": "predictor_state_dict",
                "sender_d2": "oracle_tokenizer_state_dict[d3.*]",
                "sender_combiner": "oracle_combiner_state_dict",
                "inputs": ["z1", "x1"],
                "forbidden_predictor_inputs": ["img", "z2", "q2", "oracle_indices"],
            },
        },
        path,
    )
    print(f"saved checkpoint: {path}", flush=True)


def load_resume(
    path: str,
    predictor: nn.Module,
    receiver_d2: nn.Module,
    oracle_tokenizer: nn.Module,
    oracle_combiner: nn.Module,
    receiver_combiner: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, float]:
    payload = torch.load(resolve_path(path), map_location=device, weights_only=False)
    if str(payload.get("stage", "")) != "explore2_fsq_receiver":
        raise ValueError(f"not an explore-2 receiver checkpoint: {path}")
    saved_independent = bool(
        payload.get("receiver_topology", {}).get(
            "independent_receiver_d2",
            payload.get("args", {}).get("independent_receiver_d2", False),
        )
    )
    if saved_independent != bool(args.independent_receiver_d2):
        raise ValueError(
            "resume receiver topology mismatch: "
            f"checkpoint independent_receiver_d2={saved_independent}, "
            f"current={bool(args.independent_receiver_d2)}"
        )
    saved_d2_arch = str(
        payload.get("receiver_topology", {}).get(
            "receiver_d2_arch",
            payload.get("args", {}).get("receiver_d2_arch", "oracle"),
        )
    )
    if saved_d2_arch != str(args.receiver_d2_arch):
        raise ValueError(
            "resume receiver D2 architecture mismatch: "
            f"checkpoint={saved_d2_arch!r}, current={str(args.receiver_d2_arch)!r}"
        )
    saved_residual_q = bool(
        payload.get("receiver_topology", {}).get(
            "residual_q",
            payload.get("args", {}).get("residual_q", False),
        )
    )
    if saved_residual_q != bool(args.residual_q):
        raise ValueError(
            "resume residual topology mismatch: "
            f"checkpoint residual_q={saved_residual_q}, current={bool(args.residual_q)}"
        )
    saved_sender_aligned_q = bool(
        payload.get("receiver_topology", {}).get(
            "sender_aligned_q",
            payload.get("args", {}).get("sender_aligned_q", False),
        )
    )
    if saved_sender_aligned_q != bool(args.sender_aligned_q):
        raise ValueError(
            "resume sender-aligned topology mismatch: "
            f"checkpoint sender_aligned_q={saved_sender_aligned_q}, "
            f"current={bool(args.sender_aligned_q)}"
        )
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    if "oracle_tokenizer_state_dict" in payload:
        oracle_tokenizer.load_state_dict(payload["oracle_tokenizer_state_dict"], strict=True)
    if "oracle_combiner_state_dict" in payload:
        oracle_combiner.load_state_dict(payload["oracle_combiner_state_dict"], strict=True)
    if "receiver_d2_state_dict" in payload:
        receiver_d2.load_state_dict(payload["receiver_d2_state_dict"], strict=True)
    if "receiver_combiner_state_dict" in payload:
        receiver_combiner.load_state_dict(payload["receiver_combiner_state_dict"], strict=True)
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    fallback_score = payload.get("metrics", {}).get("psnr_pred", float("-inf"))
    return int(payload.get("epoch", 0)), float(payload.get("selection_score", fallback_score))


def print_header(
    args: argparse.Namespace,
    oracle_args: argparse.Namespace,
    oracle_checkpoint: dict,
    predictor: nn.Module,
    receiver_d2: nn.Module,
    receiver_combiner: nn.Module,
    train_n: int,
    val_n: int,
) -> None:
    levels = base.parse_fsq_levels(oracle_args.fsq_levels, int(oracle_args.fsq_d))
    print("=== explore-2 | receiver-only FSQ q2 prediction ===", flush=True)
    print("实验设计", flush=True)
    print(
        f"  oracle_init={resolve_path(args.oracle_checkpoint)} epoch={oracle_checkpoint.get('epoch')} "
        f"oracle_psnr={oracle_checkpoint.get('metrics', {}).get('psnr_final')}",
        flush=True,
    )
    if args.init_receiver_checkpoint:
        print(
            f"  receiver_init={resolve_path(args.init_receiver_checkpoint)} "
            "loads=predictor+sender_tokenizer+sender_combiner optimizer=RESET",
            flush=True,
        )
    print(
        f"  inference=z1+x1 -> {args.route} -> q2_hat -> receiver D2 -> u2_hat "
        f"-> receiver combiner({args.receiver_combiner}) -> x2_hat",
        flush=True,
    )
    print("  forbidden predictor inputs=img,z2,q2,oracle_indices", flush=True)
    print(
        f"  target=psnr_x2_hat-psnr_x1>={float(args.goal_delta_db):g} dB; "
        f"train={train_n} val={val_n}; crop=train:RandomCrop(256)+RandomHorizontalFlip, "
        "val:CenterCrop(256)",
        flush=True,
    )
    print(
        f"  gates=condition_shuffle_drop>={float(args.min_condition_shuffle_drop):g} dB, "
        f"pred_q_drop>={float(args.min_pred_ablation_drop):g} dB, "
        f"oracle_delta>={float(args.min_oracle_delta_db):g} dB, "
        f"oracle_q_drop>={float(args.min_oracle_ablation_drop):g} dB",
        flush=True,
    )
    if bool(args.sender_aligned_q):
        print(
            "  sender_align_full_val_gate="
            f"MSE(q_sender,G0)<={float(args.sender_align_max_q_mse):g}, "
            f"sender_oracle_delta>={float(args.sender_align_min_oracle_delta_db):g} dB, "
            "sender_oracle_zero/shuffle_drop>="
            f"{float(args.sender_align_min_oracle_ablation_drop):g} dB",
            flush=True,
        )
        print(
            "  sender_deployment_full_val_gate="
            "G0(z1,x1)->current_sender_D2+sender_combiner; "
            f"delta_sender_qhat_x1>={float(args.sender_deployment_min_delta_db):g} dB, "
            "sender_qhat_zero/shuffle_drop>="
            f"{float(args.sender_deployment_min_ablation_drop):g} dB",
            flush=True,
        )
    print("loss设计", flush=True)
    print(
        f"  L={float(args.lambda_final):g}*MSE(x2_hat,img)"
        f"+{float(args.lambda_oracle):g}*MSE(x2_hat,x2_oracle)"
        f"+{float(args.lambda_q):g}*MSE(q2_hat,q2_oracle)"
        f"+{float(args.lambda_index):g}*CE(index_hat,index_oracle)"
        f"+{float(getattr(args, 'lambda_flow', 0.0)):g}*MSE(v_hat,q2-epsilon)"
        f"+{float(args.lambda_zero_anchor):g}*MSE(x2_zero,x1)"
        f"+{float(args.lambda_shuffle_anchor):g}*MSE(x2_shuffle,x1)",
        flush=True,
    )
    if bool(getattr(args, "residual_q", False)):
        print(
            "  residual_phase1="
            f"{float(args.residual_phase1_lambda_q):g}*MSE(qhat,qtrue)"
            f"+{float(args.residual_phase1_lambda_final):g}*MSE(xhat,img); "
            "residual_phase2="
            f"{float(args.residual_phase2_lambda_final):g}*Limg(qhat)"
            f"+{float(args.residual_phase2_lambda_true):g}*Limg(qtrue)"
            f"+{float(args.residual_phase2_lambda_mid):g}*Limg(qmid)"
            f"+{float(args.residual_phase2_lambda_consistency):g}*Lcons"
            f"+{float(args.residual_phase2_lambda_q):g}*Lq",
            flush=True,
        )
    if bool(args.sender_aligned_q):
        receiver_term = (
            f"+{float(args.sender_align_lambda_receiver_final):g}*MSE(x2_hat,img)"
            if bool(args.finetune_d2)
            else ""
        )
        print(
            "  sender_align="
            f"{float(args.sender_align_lambda_final):g}*MSE(x2_sender,img)"
            f"+{float(args.sender_align_lambda_predictability):g}*MSE(q_sender,G0(z1,x1))"
            f"+{float(args.sender_align_lambda_qhat_final):g}*"
            "MSE(sender_D2_combiner(stopgrad(G0(z1,x1))),img)"
            f"{receiver_term}; G0=stopgrad",
            flush=True,
        )
    print("模块选择", flush=True)
    print(
        f"  Layer1={oracle_args.arch} frozen; q2=[B,{len(levels)},16,16] levels={levels}; "
        f"condition={args.condition_mode}",
        flush=True,
    )
    print(
        f"  predictor={predictor.__class__.__name__} hidden={int(args.hidden)} blocks={int(args.blocks)} "
        f"attention_every={int(args.attention_every)} heads={int(args.heads)}",
        flush=True,
    )
    if str(args.route) == "ar_joint_index":
        print(
            "  AR history conditioning="
            f"model-error corruption {float(args.ar_history_corruption_start):g}"
            f"->{float(args.ar_history_corruption_end):g}; "
            f"real greedy rollout sub-batch={int(args.ar_rollout_history_batch)}; "
            "deployment remains greedy receiver-only raster generation",
            flush=True,
        )
    if str(args.route) == "flow_matching":
        print(
            "  diffusion endpoint alignment="
            f"train Euler steps={int(args.flow_train_sample_steps)} "
            f"deploy Euler steps={int(args.flow_sample_steps)} "
            f"initial_noise={args.flow_sample_noise}; "
            f"t_sampling={args.flow_timestep_sampling}; "
            f"velocity_cosine_weight={float(args.flow_cosine_loss_weight):g}; "
            "q/index/final train on autonomous endpoint",
            flush=True,
        )
        if bool(getattr(predictor, "uses_base_anchor", False)):
            print(
                "  diffusion base anchor=ACTIVE; "
                f"frozen_direct_q={resolve_path(args.flow_base_checkpoint)}; "
                "q_t=(1-t)*q_base+t*q_oracle; public Euler starts at q_base",
                flush=True,
            )
    if str(args.route) == "categorical_diffusion":
        output_mode = "terminal_argmax_hard_grid" if bool(args.hard_fsq) else "posterior_mean_continuous"
        print(
            "  CDCD categorical posterior="
            "fixed K=125 grid; train t~U[0,1], "
            "q_t=alpha(t)*q_oracle+sigma(t)*epsilon, CE(125 logits,index); "
            f"schedule={args.cdcd_schedule} (local assumption), "
            f"DDIM NFE={int(args.cdcd_sample_steps)}, "
            f"prior=N(0,{float(args.cdcd_prior_scale):g}^2 I), "
            f"deploy={output_mode}; teacher metrics and complete-rollout metrics are separate",
            flush=True,
        )
    if bool(args.residual_q):
        print(
            f"  residual_q=q0+{float(args.residual_scale):g}*tanh(R); base_G0=frozen; "
            f"R=multiscale_x1+z1 hidden={int(args.residual_hidden)} blocks={int(args.residual_blocks)} "
            f"attention_every={int(args.residual_attention_every)} heads={int(args.residual_heads)}; "
            f"phase1_epochs={int(args.residual_phase1_epochs)} "
            f"phase1_lr={float(args.residual_phase1_lr):g} "
            f"phase2_R_lr={float(args.residual_phase2_lr):g} "
            f"phase2_D2_lr={float(args.residual_phase2_decoder_lr):g}",
            flush=True,
        )
    d2_topology = (
        "receiver-new-qonly-highres-residual"
        if str(args.receiver_d2_arch) == "qonly-highres-residual"
        else "receiver-new"
        if str(args.receiver_d2_arch) == "residual-upsampler"
        else "independent-copy"
        if bool(args.independent_receiver_d2)
        else "shared-sender"
    )
    print(
        f"  receiver_D2={receiver_d2.__class__.__name__} "
        f"topology={d2_topology} "
        f"state={'trainable' if any(p.requires_grad for p in receiver_d2.parameters()) else 'frozen'} "
        f"decoder_lr={float(args.decoder_lr):g}",
        flush=True,
    )
    if bool(getattr(receiver_d2, "requires_qonly_input", False)):
        print(
            "  strict_qonly_D2=ACTIVE; decoder signature=forward(q2_hat); "
            "tokenizer.condition(x1,z1)=None is enforced; combiner=forward(x1,u2_hat)",
            flush=True,
        )
        print(
            "  qonly_highres_base=private receiver D2 loaded from init checkpoint; "
            "new RGB branch=q2_hat-only four-scale residual, zero-output at startup; "
            f"warmup_epochs={int(args.qonly_highres_warmup_epochs)}",
            flush=True,
        )
    combiner_topology = (
        "independent-copy"
        if bool(args.independent_receiver_d2) and str(args.receiver_combiner) == "oracle"
        else "receiver-owned"
        if str(args.receiver_combiner) != "oracle"
        else "shared-sender"
    )
    print(
        f"  receiver_combiner={receiver_combiner.__class__.__name__} topology={combiner_topology}; "
        "sender_oracle_path=disjoint original D2 + original combiner "
        f"state={'trainable' if bool(args.sender_aligned_q) else 'frozen'}"
        if bool(args.independent_receiver_d2)
        else f"  receiver_combiner={receiver_combiner.__class__.__name__} topology={combiner_topology}",
        flush=True,
    )
    if bool(args.joint_predictable_oracle):
        print(
            "  legacy_joint_predictable_oracle=ACTIVE; "
            f"sender_lr={float(args.sender_lr):g} "
            f"lambda_sender_final={float(args.lambda_sender_final):g} "
            f"lambda_sender_predictability={float(args.lambda_sender_predictability):g}",
            flush=True,
        )
    else:
        print(
            "  legacy_joint_predictable_oracle=inactive; "
            "legacy --lambda-sender-final/--lambda-sender-predictability are ignored",
            flush=True,
        )
    if bool(args.sender_aligned_q):
        print(
            "  sender_aligned_q=True; predictor_G0=frozen; "
            "sender=(E2+FSQ+D2+combiner) trainable; "
            "sender_receiver_D2_shared=False; sender_receiver_combiner_shared=False; "
            f"sender_E2_lr={float(args.sender_align_e2_lr):g} "
            f"sender_D2_lr={float(args.sender_align_decoder_lr):g} "
            f"receiver_D2={'trainable' if bool(args.finetune_d2) else 'frozen'}",
            flush=True,
        )


def build_data_loaders(args: argparse.Namespace, oracle_args: argparse.Namespace):
    oracle_args.data_dir = str(args.data_dir)
    oracle_args.batch_size = int(args.batch_size)
    oracle_args.test_batch = int(args.test_batch)
    oracle_args.num_workers = int(args.num_workers)
    oracle_args.val_num_workers = int(args.val_num_workers)
    oracle_args.cpu = bool(args.cpu)
    config = base.jsccf_io.build_config(oracle_args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(config)
    assert_div2k_crop_protocol(train_loader, val_loader)
    return train_loader, val_loader, config.device


def build_predictor(args: argparse.Namespace, oracle_args: argparse.Namespace, device: torch.device) -> nn.Module:
    levels = base.parse_fsq_levels(oracle_args.fsq_levels, int(oracle_args.fsq_d))
    common = dict(
        z1_channels=int(oracle_args.latent_ch),
        levels=levels,
        hidden=int(args.hidden),
        blocks=int(args.blocks),
        attention_every=int(args.attention_every),
        heads=int(args.heads),
        condition_mode=str(args.condition_mode),
        height=16,
        width=16,
    )
    flow_base_predictor: nn.Module | None = None
    if str(getattr(args, "flow_base_checkpoint", "")):
        if str(args.route) != "flow_matching":
            raise ValueError(
                "--flow-base-checkpoint is only valid with --route flow_matching"
            )
        flow_base_predictor = build_receiver_predictor(
            "direct_q",
            hard_fsq=False,
            **common,
        ).to(device)
    predictor: nn.Module = build_receiver_predictor(
        args.route,
        hard_fsq=bool(args.hard_fsq),
        flow_sample_steps=int(getattr(args, "flow_sample_steps", 32)),
        flow_train_sample_steps=int(getattr(args, "flow_train_sample_steps", 2)),
        flow_sample_noise=str(getattr(args, "flow_sample_noise", "gaussian")),
        flow_sample_seed=int(getattr(args, "flow_sample_seed", 20260713)),
        flow_time_scale=float(getattr(args, "flow_time_scale", 1000.0)),
        flow_timestep_sampling=str(
            getattr(args, "flow_timestep_sampling", "uniform")
        ),
        flow_cosine_loss_weight=float(
            getattr(args, "flow_cosine_loss_weight", 0.0)
        ),
        flow_base_predictor=flow_base_predictor,
        cdcd_sample_steps=int(getattr(args, "cdcd_sample_steps", 12)),
        cdcd_sample_seed=int(getattr(args, "cdcd_sample_seed", 20260713)),
        cdcd_time_scale=float(getattr(args, "cdcd_time_scale", 1000.0)),
        cdcd_prior_scale=float(getattr(args, "cdcd_prior_scale", 1.0)),
        cdcd_schedule=str(getattr(args, "cdcd_schedule", "cosine_vp")),
        **common,
    ).to(device)
    if bool(getattr(args, "residual_q", False)):
        predictor = QResidualPredictor(
            predictor,
            z1_channels=int(oracle_args.latent_ch),
            out_channels=len(levels),
            levels=levels,
            hidden=int(args.residual_hidden),
            blocks=int(args.residual_blocks),
            attention_every=int(args.residual_attention_every),
            heads=int(args.residual_heads),
            residual_scale=float(args.residual_scale),
            hard_fsq=bool(args.hard_fsq),
        ).to(device)
    assert_receiver_only_module(predictor)
    return predictor


def predictor_init_target(predictor: nn.Module) -> nn.Module:
    if isinstance(predictor, QResidualPredictor):
        return predictor.base_predictor
    return predictor


def initialize_predictor_from_checkpoint(
    path: str,
    predictor: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not path:
        return
    payload = torch.load(resolve_path(path), map_location=device, weights_only=False)
    if str(payload.get("stage", "")) != "explore2_fsq_receiver":
        raise ValueError(f"--init-predictor-checkpoint is not a receiver checkpoint: {path}")
    saved = payload.get("args", {})
    for key in ("route", "condition_mode", "hidden", "blocks", "attention_every", "heads"):
        if str(saved.get(key)) != str(getattr(args, key)):
            raise ValueError(
                f"predictor init contract mismatch {key}: {saved.get(key)!r} != {getattr(args,key)!r}"
            )
    predictor_init_target(predictor).load_state_dict(payload["predictor_state_dict"], strict=True)
    print(
        f"initialized predictor from {resolve_path(path)} epoch={payload.get('epoch')} "
        f"saved_delta={payload.get('metrics', {}).get('delta_x1')}",
        flush=True,
    )


def initialize_ar_base_predictor(
    path: str,
    predictor: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if str(args.route) != "ar_residual_index":
        if path:
            raise ValueError("--ar-base-checkpoint is only valid with --route ar_residual_index")
        return
    if args.resume:
        return
    if not path:
        raise ValueError("new ar_residual_index runs require --ar-base-checkpoint")
    if not hasattr(predictor, "base_predictor"):
        raise AssertionError("ar_residual_index predictor has no base_predictor")
    payload = torch.load(resolve_path(path), map_location=device, weights_only=False)
    if str(payload.get("stage", "")) != "explore2_fsq_receiver":
        raise ValueError(f"--ar-base-checkpoint is not a receiver checkpoint: {path}")
    saved = payload.get("args", {})
    if str(saved.get("route")) != "direct_q":
        raise ValueError(f"AR base must be a direct_q checkpoint, got {saved.get('route')!r}")
    for key in ("condition_mode", "hidden", "blocks", "attention_every", "heads"):
        if str(saved.get(key)) != str(getattr(args, key)):
            raise ValueError(
                f"AR base predictor contract mismatch {key}: "
                f"{saved.get(key)!r} != {getattr(args, key)!r}"
            )
    predictor.base_predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    predictor.base_predictor.requires_grad_(False)
    predictor.base_predictor.eval()
    print(
        f"initialized frozen AR direct-q base from {resolve_path(path)} "
        f"epoch={payload.get('epoch')} saved_delta={payload.get('metrics', {}).get('delta_x1')}",
        flush=True,
    )


def initialize_flow_base_predictor(
    path: str,
    predictor: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if str(args.route) != "flow_matching":
        if path:
            raise ValueError(
                "--flow-base-checkpoint is only valid with --route flow_matching"
            )
        return
    if not path or bool(args.resume):
        return
    if not hasattr(predictor, "load_base_predictor_state_dict"):
        raise AssertionError("base-anchored flow has no base predictor loader")
    payload = torch.load(resolve_path(path), map_location=device, weights_only=False)
    if str(payload.get("stage", "")) != "explore2_fsq_receiver":
        raise ValueError(f"--flow-base-checkpoint is not a receiver checkpoint: {path}")
    saved = payload.get("args", {})
    if str(saved.get("route")) != "direct_q":
        raise ValueError(
            f"flow base must be a direct_q checkpoint, got {saved.get('route')!r}"
        )
    for key in ("condition_mode", "hidden", "blocks", "attention_every", "heads"):
        if str(saved.get(key)) != str(getattr(args, key)):
            raise ValueError(
                f"flow base predictor contract mismatch {key}: "
                f"{saved.get(key)!r} != {getattr(args, key)!r}"
            )
    predictor.load_base_predictor_state_dict(
        payload["predictor_state_dict"],
        strict=True,
    )
    print(
        f"initialized frozen flow direct-q base from {resolve_path(path)} "
        f"epoch={payload.get('epoch')} saved_delta={payload.get('metrics', {}).get('delta_x1')}",
        flush=True,
    )


def initialize_from_receiver_checkpoint(
    path: str,
    predictor: nn.Module,
    bundle: direct.DirectBundle,
    args: argparse.Namespace,
    oracle_args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Initialize predictor and adapted sender state without restoring optimizer state."""
    payload = torch.load(resolve_path(path), map_location=device, weights_only=False)
    if str(payload.get("stage", "")) != "explore2_fsq_receiver":
        raise ValueError(f"--init-receiver-checkpoint is not a receiver checkpoint: {path}")
    saved = payload.get("args", {})
    for key in ("route", "condition_mode", "hidden", "blocks", "attention_every", "heads"):
        if str(saved.get(key)) != str(getattr(args, key)):
            raise ValueError(
                f"receiver init predictor contract mismatch {key}: "
                f"{saved.get(key)!r} != {getattr(args, key)!r}"
            )
    saved_oracle_args = payload.get("oracle_args", {})
    for key in ("arch", "fsq_d", "fsq_levels", "latent_ch", "latent_h", "latent_w"):
        if key in saved_oracle_args and str(saved_oracle_args[key]) != str(getattr(oracle_args, key)):
            raise ValueError(
                f"receiver init oracle contract mismatch {key}: "
                f"{saved_oracle_args[key]!r} != {getattr(oracle_args, key)!r}"
            )
    if "oracle_tokenizer_state_dict" not in payload:
        raise ValueError(f"receiver init checkpoint has no adapted sender tokenizer: {path}")
    predictor_init_target(predictor).load_state_dict(payload["predictor_state_dict"], strict=True)
    bundle.tokenizer.load_state_dict(payload["oracle_tokenizer_state_dict"], strict=True)
    if "oracle_combiner_state_dict" in payload:
        bundle.combiner.load_state_dict(payload["oracle_combiner_state_dict"], strict=True)
    else:
        saved_independent = bool(
            payload.get("receiver_topology", {}).get(
                "independent_receiver_d2",
                saved.get("independent_receiver_d2", False),
            )
        )
        if saved_independent:
            raise ValueError(
                "legacy independent receiver checkpoint lacks oracle_combiner_state_dict; "
                "cannot recover the sender combiner unambiguously"
            )
        if "receiver_combiner_state_dict" not in payload:
            raise ValueError(f"receiver init checkpoint has no sender combiner state: {path}")
        # Legacy shared-D2 checkpoints used the same object for sender and
        # receiver combiner, so the receiver payload is the sender state too.
        bundle.combiner.load_state_dict(payload["receiver_combiner_state_dict"], strict=True)
    metrics = payload.get("metrics", {})
    print(
        f"initialized predictor+sender from {resolve_path(path)} epoch={payload.get('epoch')} "
        f"saved_delta={metrics.get('delta_x1')} oracle_delta={metrics.get('delta_oracle')} "
        f"q_mse={metrics.get('loss_q')} optimizer=RESET",
        flush=True,
    )


def initialize_requested_state(
    args: argparse.Namespace,
    predictor: nn.Module,
    bundle: direct.DirectBundle,
    oracle_args: argparse.Namespace,
    device: torch.device,
) -> None:
    if args.resume and (args.init_predictor_checkpoint or args.init_receiver_checkpoint):
        raise ValueError("--resume cannot be combined with checkpoint initialization options")
    if args.init_predictor_checkpoint and args.init_receiver_checkpoint:
        raise ValueError(
            "choose either --init-predictor-checkpoint or --init-receiver-checkpoint, not both"
        )
    if args.init_receiver_checkpoint:
        initialize_from_receiver_checkpoint(
            args.init_receiver_checkpoint,
            predictor,
            bundle,
            args,
            oracle_args,
            device,
        )
    else:
        initialize_predictor_from_checkpoint(args.init_predictor_checkpoint, predictor, args, device)
    initialize_flow_base_predictor(
        str(getattr(args, "flow_base_checkpoint", "")),
        predictor,
        args,
        device,
    )
    initialize_ar_base_predictor(args.ar_base_checkpoint, predictor, args, device)


def initialize_private_receiver_modules(
    args: argparse.Namespace,
    receiver_d2: nn.Module,
    receiver_combiner: nn.Module,
    device: torch.device,
) -> None:
    private_init = (
        bool(args.residual_q)
        or bool(args.sender_aligned_q)
        or str(args.receiver_d2_arch) != "oracle"
        or str(args.receiver_combiner) != "oracle"
    )
    if not private_init or not args.init_receiver_checkpoint:
        return
    payload = torch.load(resolve_path(args.init_receiver_checkpoint), map_location=device, weights_only=False)
    for key in ("receiver_d2_state_dict", "receiver_combiner_state_dict"):
        if key not in payload:
            raise ValueError(
                f"private receiver initialization requires {key} in {args.init_receiver_checkpoint}"
            )
    saved_args = payload.get("args", {})
    saved_d2_arch = str(saved_args.get("receiver_d2_arch", "oracle"))
    if isinstance(receiver_d2, ReceiverFSQQOnlyHighResolutionResidualD2):
        if saved_d2_arch == "qonly-highres-residual":
            receiver_d2.load_state_dict(payload["receiver_d2_state_dict"], strict=True)
            d2_init = "qonly_highres_strict"
        elif saved_d2_arch == "oracle":
            # v5/v21 store the validated private CNNBottleneckDecoder directly.
            # Preserve it as the base exactly and leave the new q-only RGB
            # branch at its zero-output initialization.
            receiver_d2.base.load_state_dict(payload["receiver_d2_state_dict"], strict=True)
            receiver_d2.reset_new_branch_to_zero()
            d2_init = "legacy_receiver_d2_base_strict+new_qonly_branch_zero"
        else:
            raise ValueError(
                "cannot initialize qonly-highres-residual D2 from "
                f"receiver_d2_arch={saved_d2_arch!r}; expected oracle or qonly-highres-residual"
            )
    else:
        receiver_d2.load_state_dict(payload["receiver_d2_state_dict"], strict=True)
        d2_init = "loaded_full"
    if isinstance(receiver_combiner, ReceiverQGatedRefinerCombiner):
        # The checkpoint predates this opt-in refiner.  Recover its validated
        # private combiner as the frozen q-path base, while leaving the new
        # bias-free gated correction at its exact-zero initialization.
        receiver_combiner.base.load_state_dict(
            payload["receiver_combiner_state_dict"], strict=True
        )
        receiver_combiner.base.requires_grad_(False)
        combiner_init = "loaded_checkpoint_as_frozen_q_path_base"
    else:
        receiver_combiner.load_state_dict(payload["receiver_combiner_state_dict"], strict=True)
        combiner_init = "loaded_full"
    print(
        f"initialized private receiver D2+combiner from {resolve_path(args.init_receiver_checkpoint)} "
        f"epoch={payload.get('epoch')} d2={d2_init} combiner={combiner_init} optimizer=RESET",
        flush=True,
    )


def smoke_shapes(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    bundle, oracle_args, _checkpoint = load_fsq_oracle(args.oracle_checkpoint, device)
    predictor = build_predictor(args, oracle_args, device)
    initialize_requested_state(args, predictor, bundle, oracle_args, device)
    receiver_d2 = build_receiver_d2(
        args,
        bundle.tokenizer.d3,
        device,
        oracle_fsq_d=int(oracle_args.fsq_d),
    )
    if bool(args.finetune_d2):
        receiver_d2.requires_grad_(True)
    receiver_combiner = build_receiver_combiner(args, bundle.combiner, device)
    initialize_private_receiver_modules(args, receiver_d2, receiver_combiner, device)
    if bool(args.sender_aligned_q):
        predictor.requires_grad_(False)
        for module in (
            bundle.tokenizer.e3,
            bundle.tokenizer.quantizer,
            bundle.tokenizer.d3,
            bundle.combiner,
        ):
            module.requires_grad_(True)
    elif bool(args.joint_predictable_oracle):
        bundle.tokenizer.e3.requires_grad_(True)
        bundle.tokenizer.quantizer.requires_grad_(True)
    assert_receiver_topology(args, bundle, receiver_d2, receiver_combiner)
    qonly_warmup = configure_qonly_highres_warmup(
        args,
        predictor,
        receiver_d2,
        receiver_combiner,
        epoch=1,
        announce=False,
    )
    if qonly_warmup:
        if any(parameter.requires_grad for parameter in predictor.parameters()):
            raise AssertionError("qonly highres warmup smoke left predictor trainable")
        if not isinstance(receiver_d2, ReceiverFSQQOnlyHighResolutionResidualD2):
            raise AssertionError("qonly highres warmup smoke built an incompatible D2")
        if any(parameter.requires_grad for parameter in receiver_d2.base.parameters()):
            raise AssertionError("qonly highres warmup smoke left legacy D2 base trainable")
        if any(parameter.requires_grad for parameter in receiver_combiner.parameters()):
            raise AssertionError("qonly highres warmup smoke left legacy combiner trainable")
        new_branch = (receiver_d2.stem, receiver_d2.stages, receiver_d2.residual_head)
        if not all(parameter.requires_grad for module in new_branch for parameter in module.parameters()):
            raise AssertionError("qonly highres warmup smoke froze the new q-only branch")
    batch = int(args.smoke_batch_size)
    condition = make_receiver_condition(
        torch.randn(batch, int(oracle_args.latent_ch), int(oracle_args.latent_h), int(oracle_args.latent_w), device=device),
        torch.rand(batch, 3, 256, 256, device=device),
    )
    prediction = predictor(condition)
    decoded = decode_receiver(
        bundle.tokenizer,
        receiver_d2,
        prediction.q_train,
        condition.x1,
        condition.z1,
        receiver_combiner,
    )
    sender_decoded = bundle.tokenizer.decode(
        prediction.q_train,
        condition.x1,
        condition.z1,
        bundle.combiner,
    )
    assert tuple(prediction.q_train.shape) == (
        batch,
        int(oracle_args.fsq_d),
        int(oracle_args.latent_h),
        int(oracle_args.latent_w),
    )
    assert tuple(decoded["final"].shape) == (batch, 3, 256, 256)
    assert tuple(sender_decoded["final"].shape) == (batch, 3, 256, 256)
    if bool(args.residual_q):
        if prediction.q_base is None or prediction.q_residual is None:
            raise AssertionError("residual-q smoke did not return q_base/q_residual")
        if tuple(prediction.q_residual.shape) != tuple(prediction.q_train.shape):
            raise AssertionError("residual-q correction shape mismatch")
    qonly_zero_start = False
    if isinstance(receiver_d2, ReceiverFSQQOnlyHighResolutionResidualD2):
        assert_qonly_d2_contract(bundle.tokenizer, receiver_d2)
        if bundle.tokenizer.condition(condition.x1, condition.z1) is not None:
            raise AssertionError("strict q-only smoke unexpectedly produced a tokenizer D2 condition")
        d2_parameters = list(inspect.signature(receiver_d2.forward).parameters.values())
        if [parameter.name for parameter in d2_parameters] != ["q2_hat"]:
            raise AssertionError("strict q-only smoke found a non-q2_hat D2 signature")
        combiner_parameters = list(inspect.signature(receiver_combiner.forward).parameters.values())
        if [parameter.name for parameter in combiner_parameters] != ["x1", "u2"]:
            raise AssertionError(
                "strict q-only smoke requires combiner.forward(x1,u2), got "
                f"{[parameter.name for parameter in combiner_parameters]}"
            )
        with torch.no_grad():
            base_u2 = receiver_d2.base(prediction.q_train)
            wrapped_u2 = receiver_d2(prediction.q_train)
        if not torch.equal(base_u2, wrapped_u2):
            raise AssertionError("zero-initialized q-only highres branch changed its legacy D2 base")
        legacy_final = receiver_combiner(condition.x1, base_u2.clamp(0.0, 1.0))
        if not torch.equal(decoded["final"], legacy_final):
            raise AssertionError(
                "zero-initialized q-only highres branch changed the legacy D2+combiner startup output"
            )
        qonly_zero_start = True
    assert_receiver_only_module(predictor)
    assert_training_targets_are_not_inputs(
        predictor,
        condition,
        source_targets={
            "img": torch.rand_like(condition.x1),
            "oracle_q2": torch.rand_like(prediction.q_train),
            "oracle_indices": torch.zeros_like(prediction.q_train, dtype=torch.long),
            "z2": torch.rand_like(prediction.q_train),
        },
    )
    # Structural crop audit for a no-data CPU smoke.  Real training/evaluation
    # separately calls build_data_loaders(), which checks the actual DIV2K
    # transforms before the first batch.
    train_loader = SimpleNamespace(
        dataset=SimpleNamespace(transform="RandomCrop(256)+RandomHorizontalFlip+ToTensor")
    )
    val_loader = SimpleNamespace(dataset=SimpleNamespace(transform="CenterCrop(256)+ToTensor"))
    assert_div2k_crop_protocol(train_loader, val_loader)
    d2_topology = (
        "receiver-new-qonly-highres-residual"
        if str(args.receiver_d2_arch) == "qonly-highres-residual"
        else "receiver-new"
        if str(args.receiver_d2_arch) == "residual-upsampler"
        else "independent-copy"
        if bool(args.independent_receiver_d2)
        else "shared-sender"
    )
    print(
        f"[smoke] route={args.route} condition={args.condition_mode} "
        f"q2_hat={tuple(prediction.q_train.shape)} u2_hat={tuple(decoded['u2_hat'].shape)} "
        f"x2_hat={tuple(decoded['final'].shape)} "
        f"d2_topology={d2_topology} "
        f"residual_q={bool(args.residual_q)} "
        f"strict_qonly_d2={int(qonly_zero_start)} warmup_active={int(qonly_warmup)} "
        "receiver_only=PASS crop_contract=PASS",
        flush=True,
    )


def residual_phase_for_epoch(args: argparse.Namespace, epoch: int) -> int:
    if not bool(args.residual_q):
        return 0
    return 1 if int(epoch) <= int(args.residual_phase1_epochs) else 2


def configure_residual_phase(
    args: argparse.Namespace,
    predictor: nn.Module,
    receiver_d2: nn.Module,
    receiver_combiner: nn.Module,
    optimizer: optim.Optimizer,
    phase: int,
    *,
    announce: bool,
) -> None:
    if not bool(args.residual_q):
        return
    if not isinstance(predictor, QResidualPredictor):
        raise TypeError("--residual-q requires QResidualPredictor")
    predictor.base_predictor.requires_grad_(False)
    for parameter in predictor.residual_parameters():
        parameter.requires_grad_(True)
    train_decoder = int(phase) >= 2
    receiver_d2.requires_grad_(train_decoder)
    receiver_combiner.requires_grad_(train_decoder)
    for group in optimizer.param_groups:
        name = str(group.get("group_name", ""))
        if name == "residual_predictor":
            group["lr"] = float(
                args.residual_phase1_lr if int(phase) == 1 else args.residual_phase2_lr
            )
        elif name == "residual_decoder":
            group["lr"] = float(args.residual_phase2_decoder_lr) if train_decoder else 0.0
    if announce:
        print(
            f"[residual phase {int(phase)}] "
            f"R_lr={float(args.residual_phase1_lr if int(phase) == 1 else args.residual_phase2_lr):.3e} "
            f"receiver_D2_combiner={'trainable' if train_decoder else 'frozen'} "
            f"decoder_lr={float(args.residual_phase2_decoder_lr) if train_decoder else 0.0:.3e}",
            flush=True,
        )


def qonly_highres_warmup_active(args: argparse.Namespace, epoch: int) -> bool:
    return bool(
        str(args.receiver_d2_arch) == "qonly-highres-residual"
        and int(args.qonly_highres_warmup_epochs) > 0
        and int(epoch) <= int(args.qonly_highres_warmup_epochs)
    )


def configure_qonly_highres_warmup(
    args: argparse.Namespace,
    predictor: nn.Module,
    receiver_d2: nn.Module,
    receiver_combiner: nn.Module,
    *,
    epoch: int,
    announce: bool,
) -> bool:
    """Optionally train only the new q-only residual branch before finetuning.

    This avoids erasing the v5 receiver predictor/base-D2/combiner before the
    zero-initialized high-resolution branch has learned a useful correction.
    All parameters remain in the optimizer groups, so leaving warmup simply
    re-enables their gradients without rebuilding optimizer state.
    """

    if str(args.receiver_d2_arch) != "qonly-highres-residual":
        setattr(args, "_qonly_highres_warmup_active", False)
        return False
    if not isinstance(receiver_d2, ReceiverFSQQOnlyHighResolutionResidualD2):
        raise AssertionError("qonly-highres-residual did not build its expected receiver D2")
    active = qonly_highres_warmup_active(args, int(epoch))
    predictor.requires_grad_(not active)
    receiver_d2.base.requires_grad_(not active)
    for module in (receiver_d2.stem, receiver_d2.stages, receiver_d2.residual_head):
        module.requires_grad_(True)
    receiver_combiner.requires_grad_(not active)
    setattr(args, "_qonly_highres_warmup_active", bool(active))
    if announce:
        if active:
            print(
                "[qonly-highres warmup] active: train=new q2_hat-only highres branch; "
                "freeze/eval=v5 predictor+base-D2+combiner",
                flush=True,
            )
        else:
            print(
                "[qonly-highres warmup] complete: unfreeze predictor+base-D2+combiner; "
                "continue strict q2_hat-only D2 finetuning",
                flush=True,
            )
    return bool(active)


def validate_residual_args(args: argparse.Namespace) -> None:
    if not bool(args.residual_q):
        return
    if str(args.route) != "direct_q":
        raise ValueError("--residual-q currently requires --route direct_q")
    if str(args.condition_mode) != "z1_x1":
        raise ValueError("--residual-q requires --condition-mode z1_x1")
    if bool(args.hard_fsq):
        raise ValueError("residual-q phase1/phase2 are continuous; do not use --hard-fsq")
    if not bool(args.independent_receiver_d2) or not bool(args.finetune_d2):
        raise ValueError("--residual-q requires --independent-receiver-d2 --finetune-d2")
    if str(args.receiver_combiner) != "oracle":
        raise ValueError("v5 residual-q initialization currently requires --receiver-combiner oracle")
    if bool(args.joint_predictable_oracle):
        raise ValueError("residual-q freezes the sender; do not use --joint-predictable-oracle")
    if not args.resume and not args.init_receiver_checkpoint:
        raise ValueError("new residual-q runs require --init-receiver-checkpoint (use v5 best)")
    if int(args.residual_phase1_epochs) < 1:
        raise ValueError("--residual-phase1-epochs must be positive")


def validate_qonly_highres_args(args: argparse.Namespace) -> None:
    """Keep the high-resolution extension a strict v5-compatible opt-in path."""

    if str(args.receiver_d2_arch) != "qonly-highres-residual":
        return
    if str(args.route) != "direct_q" or str(args.condition_mode) != "z1_x1":
        raise ValueError(
            "--receiver-d2-arch qonly-highres-residual requires "
            "--route direct_q --condition-mode z1_x1"
        )
    if int(args.qonly_highres_warmup_epochs) < 0:
        raise ValueError("--qonly-highres-warmup-epochs must be non-negative")
    if not bool(args.independent_receiver_d2) or not bool(args.finetune_d2):
        raise ValueError(
            "qonly-highres-residual requires --independent-receiver-d2 --finetune-d2"
        )
    if str(args.receiver_combiner) != "oracle":
        raise ValueError(
            "qonly-highres-residual starts from the validated v5 D2+oracle-combiner stack; "
            "use --receiver-combiner oracle"
        )
    if bool(args.residual_q) or bool(args.sender_aligned_q) or bool(args.joint_predictable_oracle):
        raise ValueError(
            "qonly-highres-residual is a receiver-only D2 extension; do not combine it with "
            "--residual-q, --sender-aligned-q, or --joint-predictable-oracle"
        )
    if not args.resume and not args.init_receiver_checkpoint:
        raise ValueError(
            "new qonly-highres-residual runs require --init-receiver-checkpoint "
            "so their base D2/combiner is loaded exactly"
        )


def validate_sender_aligned_args(args: argparse.Namespace) -> None:
    if not bool(args.sender_aligned_q):
        return
    if bool(args.residual_q):
        raise ValueError("--sender-aligned-q and --residual-q are separate opt-in routes")
    if bool(args.joint_predictable_oracle):
        raise ValueError("--sender-aligned-q replaces --joint-predictable-oracle")
    if str(args.route) != "direct_q" or str(args.condition_mode) != "z1_x1":
        raise ValueError("--sender-aligned-q requires --route direct_q --condition-mode z1_x1")
    if bool(args.hard_fsq):
        raise ValueError("G0 supplies a continuous receiver-only target; do not use --hard-fsq")
    if not bool(args.independent_receiver_d2):
        raise ValueError(
            "--sender-aligned-q requires --independent-receiver-d2 so sender/receiver D2 are disjoint"
        )
    if str(args.receiver_combiner) != "oracle":
        raise ValueError("v9 sender alignment currently requires --receiver-combiner oracle")
    if not args.resume and not args.init_receiver_checkpoint:
        raise ValueError("new --sender-aligned-q runs require --init-receiver-checkpoint (v4/v5)")
    sender_loss_weights = {
        "--sender-align-lambda-final": float(args.sender_align_lambda_final),
        "--sender-align-lambda-predictability": float(args.sender_align_lambda_predictability),
        "--sender-align-lambda-qhat-final": float(args.sender_align_lambda_qhat_final),
        "--sender-align-lambda-receiver-final": (
            float(args.sender_align_lambda_receiver_final) if bool(args.finetune_d2) else 0.0
        ),
    }
    negative = [name for name, value in sender_loss_weights.items() if value < 0]
    if negative:
        raise ValueError(f"sender-align loss weights must be non-negative: {negative}")
    if not any(value > 0 for value in sender_loss_weights.values()):
        raise ValueError("--sender-aligned-q requires at least one positive sender-align loss weight")
    if float(args.sender_align_e2_lr) <= 0 or float(args.sender_align_decoder_lr) <= 0:
        raise ValueError("sender E2 and sender D2 learning rates must be positive")
    if float(args.sender_align_max_q_mse) <= 0:
        raise ValueError("--sender-align-max-q-mse must be positive")
    if float(args.sender_deployment_min_delta_db) < 0:
        raise ValueError("--sender-deployment-min-delta-db must be non-negative")
    if float(args.sender_deployment_min_ablation_drop) < 0:
        raise ValueError("--sender-deployment-min-ablation-drop must be non-negative")


def validate_ar_history_args(args: argparse.Namespace) -> None:
    start = float(args.ar_history_corruption_start)
    end = float(args.ar_history_corruption_end)
    if str(args.route) == "ar_joint_index" and not bool(args.hard_fsq):
        raise ValueError(
            "--route ar_joint_index generates discrete K-way FSQ tokens and "
            "therefore requires --hard-fsq"
        )
    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
        raise ValueError("AR history corruption probabilities must lie in [0,1]")
    if (start > 0.0 or end > 0.0) and str(args.route) != "ar_joint_index":
        raise ValueError(
            "AR history corruption is an opt-in route for --route ar_joint_index"
        )
    if (start > 0.0 or end > 0.0) and int(args.ar_rollout_history_batch) < 1:
        raise ValueError(
            "AR history corruption requires --ar-rollout-history-batch >= 1"
        )
    if int(args.flow_train_sample_steps) < 1:
        raise ValueError("--flow-train-sample-steps must be positive")
    if int(args.train_deploy_metric_batches) < 0:
        raise ValueError("--train-deploy-metric-batches must be non-negative")
    if int(args.train_deploy_metric_batch_size) < 1:
        raise ValueError("--train-deploy-metric-batch-size must be positive")
    if str(getattr(args, "flow_base_checkpoint", "")) and str(args.route) != "flow_matching":
        raise ValueError(
            "--flow-base-checkpoint is only valid with --route flow_matching"
        )
    if str(args.route) == "categorical_diffusion":
        if int(args.cdcd_sample_steps) < 2:
            raise ValueError("--cdcd-sample-steps must be at least 2")
        if float(args.cdcd_time_scale) <= 0.0:
            raise ValueError("--cdcd-time-scale must be positive")
        if float(args.cdcd_prior_scale) <= 0.0:
            raise ValueError("--cdcd-prior-scale must be positive")
        if float(args.lambda_index) <= 0.0:
            raise ValueError("categorical_diffusion requires a positive --lambda-index CE weight")
        forbidden = {
            "--lambda-flow": float(args.lambda_flow),
            "--lambda-q": float(args.lambda_q),
            "--lambda-final": float(args.lambda_final),
            "--lambda-oracle": float(args.lambda_oracle),
            "--lambda-zero-anchor": float(args.lambda_zero_anchor),
            "--lambda-shuffle-anchor": float(args.lambda_shuffle_anchor),
        }
        active = [name for name, value in forbidden.items() if value != 0.0]
        if active:
            raise ValueError(
                "categorical_diffusion is the paper-driven CE-only route; disable "
                + ", ".join(active)
            )


def ar_history_corruption_probability(args: argparse.Namespace, epoch: int) -> float:
    start = float(args.ar_history_corruption_start)
    end = float(args.ar_history_corruption_end)
    if int(args.epochs) <= 1:
        return end
    progress = (int(epoch) - 1) / max(1, int(args.epochs) - 1)
    progress = min(1.0, max(0.0, float(progress)))
    return start + progress * (end - start)


def train(args: argparse.Namespace) -> None:
    validate_residual_args(args)
    validate_qonly_highres_args(args)
    validate_sender_aligned_args(args)
    validate_ar_history_args(args)
    seed_everything(int(args.seed))
    # Data config chooses the same device used for model construction.
    checkpoint_probe = base.jsccf_io.load_checkpoint(str(resolve_path(args.oracle_checkpoint)))
    oracle_args_probe = argparse.Namespace(**checkpoint_probe["args"])
    train_loader, val_loader, device = build_data_loaders(args, oracle_args_probe)
    bundle, oracle_args, oracle_checkpoint = load_fsq_oracle(args.oracle_checkpoint, device)
    predictor = build_predictor(args, oracle_args, device)
    initialize_requested_state(args, predictor, bundle, oracle_args, device)
    receiver_d2 = build_receiver_d2(
        args,
        bundle.tokenizer.d3,
        device,
        oracle_fsq_d=int(oracle_args.fsq_d),
    )
    if bool(args.finetune_d2):
        receiver_d2.requires_grad_(True)
    receiver_combiner = build_receiver_combiner(args, bundle.combiner, device)
    initialize_private_receiver_modules(args, receiver_d2, receiver_combiner, device)
    if bool(args.sender_aligned_q):
        predictor.requires_grad_(False)
        predictor.eval()
        for module in (
            bundle.tokenizer.e3,
            bundle.tokenizer.quantizer,
            bundle.tokenizer.d3,
            bundle.combiner,
        ):
            module.requires_grad_(True)
    elif bool(args.joint_predictable_oracle):
        # load_fsq_oracle() freezes the complete sender/receiver tokenizer.  A
        # predictable-oracle run must explicitly reopen the sender-side E2/FSQ
        # parameters or loss_sender_predictability has no trainable path.
        bundle.tokenizer.e3.requires_grad_(True)
        bundle.tokenizer.quantizer.requires_grad_(True)
    assert_receiver_topology(args, bundle, receiver_d2, receiver_combiner)
    if bool(args.residual_q):
        if not isinstance(predictor, QResidualPredictor):
            raise TypeError("--residual-q did not build QResidualPredictor")
        residual_parameters = list(predictor.residual_parameters())
        decoder_parameters = [
            parameter
            for module in (receiver_d2, receiver_combiner)
            for parameter in module.parameters()
        ]
        parameter_groups: list[dict] = [
            {
                "params": residual_parameters,
                "lr": float(args.residual_phase1_lr),
                "group_name": "residual_predictor",
            },
            {
                "params": decoder_parameters,
                "lr": 0.0,
                "group_name": "residual_decoder",
            },
        ]
    elif bool(args.sender_aligned_q):
        sender_e2_parameters = [
            parameter
            for module in (bundle.tokenizer.e3, bundle.tokenizer.quantizer)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        sender_decoder_parameters = [
            parameter
            for module in (bundle.tokenizer.d3, bundle.combiner)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        decoder_parameters = [
            parameter
            for module in (receiver_d2, receiver_combiner)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if not sender_e2_parameters or not sender_decoder_parameters:
            raise RuntimeError("sender-aligned-q found no trainable sender E2/FSQ or D2/combiner parameters")
        parameter_groups = [
            {
                "params": sender_e2_parameters,
                "lr": float(args.sender_align_e2_lr),
                "group_name": "sender_e2_fsq",
            },
            {
                "params": sender_decoder_parameters,
                "lr": float(args.sender_align_decoder_lr),
                "group_name": "sender_d2_combiner",
            },
        ]
        if decoder_parameters:
            parameter_groups.append(
                {
                    "params": decoder_parameters,
                    "lr": float(args.decoder_lr),
                    "group_name": "receiver_d2_combiner",
                }
            )
        sender_ids = {id(parameter) for parameter in sender_decoder_parameters}
        receiver_ids = {id(parameter) for parameter in decoder_parameters}
        if sender_ids & receiver_ids:
            raise AssertionError("sender/receiver decoder optimizer groups unexpectedly overlap")
        print(
            "[sender-aligned-q trainability] "
            f"G0_trainable=0 sender_E2_FSQ={sum(p.numel() for p in sender_e2_parameters)} "
            f"sender_D2_combiner={sum(p.numel() for p in sender_decoder_parameters)} "
            f"receiver_D2_combiner={sum(p.numel() for p in decoder_parameters)} "
            "sender_receiver_shared_params=0 status=PASS",
            flush=True,
        )
    else:
        parameter_groups = [
            {"params": list(predictor.parameters()), "lr": float(args.lr)},
        ]
        decoder_parameters = [
            parameter
            for module in (receiver_d2, receiver_combiner)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if decoder_parameters:
            parameter_groups.append({"params": decoder_parameters, "lr": float(args.decoder_lr)})
    if bool(args.joint_predictable_oracle):
        decoder_ids = {id(parameter) for parameter in decoder_parameters}
        sender_parameters = [
            parameter
            for module in (bundle.tokenizer.e3, bundle.tokenizer.quantizer)
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in decoder_ids
        ]
        if not sender_parameters:
            raise RuntimeError(
                "--joint-predictable-oracle requires trainable E2/FSQ sender parameters, "
                "but none were found"
            )
        parameter_groups.append({"params": sender_parameters, "lr": float(args.sender_lr)})
        print(
            "[sender-trainability] "
            f"tensors={len(sender_parameters)} params={sum(p.numel() for p in sender_parameters)} "
            f"lr={float(args.sender_lr):.3e} status=PASS",
            flush=True,
        )
    optimizer = optim.AdamW(parameter_groups, weight_decay=float(args.weight_decay))
    name = experiment_name(args, oracle_args)
    save_dir = resolve_path(args.save_dir) / str(args.version)
    save_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best = float("-inf")
    best_goal = float("-inf")
    if args.resume:
        start_epoch, best = load_resume(
            args.resume,
            predictor,
            receiver_d2,
            bundle.tokenizer,
            bundle.combiner,
            receiver_combiner,
            optimizer,
            args,
            device,
        )
        print(f"resumed {args.resume} at epoch={start_epoch} best_score={best:.6f}", flush=True)
    configured_residual_phase = residual_phase_for_epoch(
        args,
        max(1, start_epoch if bool(args.eval_only) else start_epoch + 1),
    )
    configure_residual_phase(
        args,
        predictor,
        receiver_d2,
        receiver_combiner,
        optimizer,
        configured_residual_phase,
        announce=bool(args.residual_q),
    )
    configured_qonly_warmup = qonly_highres_warmup_active(
        args,
        max(1, start_epoch if bool(args.eval_only) else start_epoch + 1),
    )
    configure_qonly_highres_warmup(
        args,
        predictor,
        receiver_d2,
        receiver_combiner,
        epoch=max(1, start_epoch if bool(args.eval_only) else start_epoch + 1),
        announce=(
            str(args.receiver_d2_arch) == "qonly-highres-residual"
            and int(args.qonly_highres_warmup_epochs) > 0
        ),
    )
    print_header(
        args,
        oracle_args,
        oracle_checkpoint,
        predictor,
        receiver_d2,
        receiver_combiner,
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    if bool(args.eval_only):
        if not args.resume:
            raise ValueError("--eval-only requires --resume")
        metrics = run_loader(
            val_loader,
            predictor=predictor,
            bundle=bundle,
            receiver_d2=receiver_d2,
            receiver_combiner=receiver_combiner,
            optimizer=None,
            args=args,
            device=device,
            train=False,
            residual_phase=configured_residual_phase,
        )
        print(f"[receiver eval] {display(metrics)}", flush=True)
        return

    latest_metrics: dict[str, float] = {}
    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        args._ar_history_corruption_prob = ar_history_corruption_probability(args, epoch)
        epoch_residual_phase = residual_phase_for_epoch(args, epoch)
        if epoch_residual_phase != configured_residual_phase:
            configured_residual_phase = epoch_residual_phase
            configure_residual_phase(
                args,
                predictor,
                receiver_d2,
                receiver_combiner,
                optimizer,
                configured_residual_phase,
                announce=True,
            )
        epoch_qonly_warmup = qonly_highres_warmup_active(args, epoch)
        if epoch_qonly_warmup != configured_qonly_warmup:
            configured_qonly_warmup = epoch_qonly_warmup
            configure_qonly_highres_warmup(
                args,
                predictor,
                receiver_d2,
                receiver_combiner,
                epoch=epoch,
                announce=True,
            )
        started = time.time()
        train_metrics = run_loader(
            train_loader,
            predictor=predictor,
            bundle=bundle,
            receiver_d2=receiver_d2,
            receiver_combiner=receiver_combiner,
            optimizer=optimizer,
            args=args,
            device=device,
            train=True,
            residual_phase=configured_residual_phase,
        )
        print(
            f"[receiver train {epoch:03d}/{int(args.epochs):03d}] {display(train_metrics)} "
            f"time={time.time() - started:.1f}s",
            flush=True,
        )
        latest_metrics = train_metrics
        if epoch == 1 or epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            with torch.no_grad():
                val_metrics = run_loader(
                    val_loader,
                    predictor=predictor,
                    bundle=bundle,
                    receiver_d2=receiver_d2,
                    receiver_combiner=receiver_combiner,
                    optimizer=None,
                    args=args,
                    device=device,
                    train=False,
                    residual_phase=configured_residual_phase,
                )
            print(f"[receiver val {epoch:03d}] {display(val_metrics)}", flush=True)
            latest_metrics = val_metrics
            score = checkpoint_selection_score(val_metrics, args)
            if score > best:
                best = score
                save_checkpoint(
                    save_dir / f"{name}_best.pth",
                    epoch=epoch,
                    predictor=predictor,
                    receiver_d2=receiver_d2,
                    oracle_tokenizer=bundle.tokenizer,
                    oracle_combiner=bundle.combiner,
                    receiver_combiner=receiver_combiner,
                    optimizer=optimizer,
                    args=args,
                    oracle_args=oracle_args,
                    metrics=val_metrics,
                )
            goal_key = "sender_align_goal_met" if bool(args.sender_aligned_q) else "goal_met"
            goal_suffix = "sender_goal_best" if bool(args.sender_aligned_q) else "goal_best"
            if bool(val_metrics.get(goal_key, 0.0)) and score > best_goal:
                best_goal = score
                save_checkpoint(
                    save_dir / f"{name}_{goal_suffix}.pth",
                    epoch=epoch,
                    predictor=predictor,
                    receiver_d2=receiver_d2,
                    oracle_tokenizer=bundle.tokenizer,
                    oracle_combiner=bundle.combiner,
                    receiver_combiner=receiver_combiner,
                    optimizer=optimizer,
                    args=args,
                    oracle_args=oracle_args,
                    metrics=val_metrics,
                )
        if epoch % int(args.latest_every) == 0 or epoch == int(args.epochs):
            save_checkpoint(
                save_dir / f"{name}_latest.pth",
                epoch=epoch,
                predictor=predictor,
                receiver_d2=receiver_d2,
                oracle_tokenizer=bundle.tokenizer,
                oracle_combiner=bundle.combiner,
                receiver_combiner=receiver_combiner,
                optimizer=optimizer,
                args=args,
                oracle_args=oracle_args,
                metrics=latest_metrics,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--oracle-checkpoint", default=DEFAULT_ORACLE)
    parser.add_argument(
        "--route",
        choices=[
            "direct_q",
            "parallel_index",
            "joint_index",
            "ar_index",
            "ar_joint_index",
            "ar_residual_index",
            "flow_matching",
            "categorical_diffusion",
        ],
        default="direct_q",
    )
    parser.add_argument("--condition-mode", choices=["z1", "x1", "z1_x1"], default="z1_x1")
    parser.add_argument("--hard-fsq", action="store_true", help="Snap direct_q to the oracle FSQ grid with STE.")
    parser.add_argument("--finetune-d2", action="store_true", help="Jointly adapt receiver D2 without exposing sender tensors.")
    parser.add_argument(
        "--independent-receiver-d2",
        action="store_true",
        help=(
            "Decode q2_hat with private copies of the oracle D2 and, in oracle-combiner mode, "
            "the oracle combiner; sender/receiver parameters never alias. The sender copy remains "
            "frozen except in the explicit --sender-aligned-q route."
        ),
    )
    parser.add_argument(
        "--receiver-combiner",
        choices=[
            "oracle",
            "residual",
            "qonly-residual",
            "qonly-unet",
            "qgated-refiner",
            "residual-signal",
        ],
        default="oracle",
    )
    parser.add_argument(
        "--receiver-d2-arch",
        choices=["oracle", "residual-upsampler", "qonly-highres-residual"],
        default="oracle",
    )
    parser.add_argument("--receiver-d2-width", type=int, default=64)
    parser.add_argument("--receiver-d2-blocks", type=int, default=2)
    parser.add_argument(
        "--qonly-highres-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Only for --receiver-d2-arch qonly-highres-residual: first train just the "
            "new q2_hat-only residual branch while preserving the loaded v5 stack; 0 disables."
        ),
    )
    parser.add_argument("--receiver-unet-width", type=int, default=32)
    parser.add_argument("--receiver-unet-blocks", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--attention-every", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--flow-sample-steps",
        type=int,
        default=32,
        help="Euler NFE for flow deployment; the iFSQ paper reports 250 by default.",
    )
    parser.add_argument(
        "--flow-train-sample-steps",
        type=int,
        default=2,
        help=(
            "Differentiable autonomous Euler steps used for flow q/index/final "
            "training; velocity supervision remains at random t."
        ),
    )
    parser.add_argument(
        "--flow-sample-noise",
        choices=["gaussian", "zero"],
        default="gaussian",
    )
    parser.add_argument("--flow-sample-seed", type=int, default=20260713)
    parser.add_argument("--flow-time-scale", type=float, default=1000.0)
    parser.add_argument(
        "--flow-timestep-sampling",
        choices=["uniform", "logit_normal"],
        default="uniform",
        help=(
            "Training-time distribution for t. The released iFSQ transport "
            "configuration uses logit_normal."
        ),
    )
    parser.add_argument(
        "--flow-cosine-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on 1-cos(v_hat,v_target); the released iFSQ transport "
            "configuration adds it with weight 1."
        ),
    )
    parser.add_argument(
        "--flow-base-checkpoint",
        default="",
        help=(
            "Optional frozen receiver-only direct_q checkpoint used as the "
            "initial state of a base-anchored residual flow."
        ),
    )
    parser.add_argument(
        "--cdcd-sample-steps",
        type=int,
        default=12,
        help=(
            "Deterministic DDIM model evaluations for categorical-posterior "
            "diffusion; the source paper reports 5/8/12/25-step TTS results."
        ),
    )
    parser.add_argument("--cdcd-sample-seed", type=int, default=20260713)
    parser.add_argument("--cdcd-time-scale", type=float, default=1000.0)
    parser.add_argument(
        "--cdcd-prior-scale",
        type=float,
        default=1.0,
        help=(
            "Gaussian DDIM initial-state standard deviation. The theory uses "
            "1; the paper's TTS implementation reports an empirical 2/3."
        ),
    )
    parser.add_argument(
        "--cdcd-schedule",
        choices=["cosine_vp"],
        default="cosine_vp",
        help=(
            "Explicit local alpha/sigma assumption because arXiv:2606.09962 "
            "does not publish its exact schedule."
        ),
    )
    parser.add_argument(
        "--ar-history-corruption-start",
        type=float,
        default=0.0,
        help="Training-only probability of replacing an oracle AR history token by the model proposal.",
    )
    parser.add_argument(
        "--ar-history-corruption-end",
        type=float,
        default=0.0,
        help="Final linearly scheduled AR model-error history corruption probability.",
    )
    parser.add_argument(
        "--ar-rollout-history-batch",
        type=int,
        default=2,
        help=(
            "Number of leading samples per train batch whose scheduled-sampling "
            "history is generated by the true 256-step greedy AR rollout."
        ),
    )
    parser.add_argument(
        "--train-deploy-metric-batches",
        type=int,
        default=0,
        help=(
            "Number of leading train batches per epoch evaluated through the exact "
            "public receiver forward; use 1 for AR/flow diagnostics."
        ),
    )
    parser.add_argument(
        "--train-deploy-metric-batch-size",
        type=int,
        default=2,
        help="Sub-batch size for exact train-deployment diagnostics.",
    )
    parser.add_argument("--residual-q", action="store_true")
    parser.add_argument("--residual-hidden", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=12)
    parser.add_argument("--residual-attention-every", type=int, default=3)
    parser.add_argument("--residual-heads", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--residual-phase1-epochs", type=int, default=40)
    parser.add_argument("--residual-phase1-lr", type=float, default=1e-4)
    parser.add_argument("--residual-phase2-lr", type=float, default=2e-5)
    parser.add_argument("--residual-phase2-decoder-lr", type=float, default=2e-5)
    parser.add_argument("--residual-phase1-lambda-q", type=float, default=1.0)
    parser.add_argument("--residual-phase1-lambda-final", type=float, default=0.25)
    parser.add_argument("--residual-phase2-lambda-final", type=float, default=1.0)
    parser.add_argument("--residual-phase2-lambda-true", type=float, default=0.5)
    parser.add_argument("--residual-phase2-lambda-mid", type=float, default=0.25)
    parser.add_argument("--residual-phase2-lambda-consistency", type=float, default=0.1)
    parser.add_argument("--residual-phase2-lambda-q", type=float, default=0.05)
    parser.add_argument("--lambda-q", type=float, default=0.02)
    parser.add_argument("--lambda-flow", type=float, default=0.0)
    parser.add_argument("--lambda-index", type=float, default=0.01)
    parser.add_argument("--lambda-final", type=float, default=1.0)
    parser.add_argument("--hard-example-power", type=float, default=0.0)
    parser.add_argument("--hard-example-min-weight", type=float, default=0.25)
    parser.add_argument("--hard-example-max-weight", type=float, default=4.0)
    parser.add_argument("--lambda-oracle", type=float, default=0.25)
    parser.add_argument("--lambda-zero-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-shuffle-anchor", type=float, default=0.0)
    parser.add_argument("--joint-predictable-oracle", action="store_true")
    parser.add_argument("--lambda-sender-final", type=float, default=1.0)
    parser.add_argument("--lambda-sender-predictability", type=float, default=0.01)
    parser.add_argument(
        "--sender-aligned-q",
        action="store_true",
        help=(
            "Freeze receiver-only G0 and reshape sender E2/FSQ toward G0 while training a "
            "sender-only D2/combiner disjoint from the independent receiver D2/combiner."
        ),
    )
    parser.add_argument("--sender-align-lambda-predictability", type=float, default=0.5)
    parser.add_argument("--sender-align-lambda-final", type=float, default=1.0)
    parser.add_argument(
        "--sender-align-lambda-qhat-final",
        type=float,
        default=0.0,
        help=(
            "Direct deployment loss on current sender D2/combiner using detached frozen-G0 qhat; "
            "zero preserves the original sender-aligned-q training graph."
        ),
    )
    parser.add_argument("--sender-align-lambda-receiver-final", type=float, default=1.0)
    parser.add_argument("--sender-align-e2-lr", type=float, default=2e-5)
    parser.add_argument("--sender-align-decoder-lr", type=float, default=2e-5)
    parser.add_argument("--sender-align-max-q-mse", type=float, default=0.0015)
    parser.add_argument("--sender-align-min-oracle-delta-db", type=float, default=0.8)
    parser.add_argument("--sender-align-min-oracle-ablation-drop", type=float, default=0.5)
    parser.add_argument("--sender-deployment-min-delta-db", type=float, default=0.5)
    parser.add_argument("--sender-deployment-min-ablation-drop", type=float, default=0.1)
    parser.add_argument("--goal-delta-db", type=float, default=0.5)
    parser.add_argument("--min-pred-ablation-drop", type=float, default=0.1)
    parser.add_argument("--min-condition-shuffle-drop", type=float, default=0.1)
    parser.add_argument("--min-oracle-delta-db", type=float, default=0.8)
    parser.add_argument("--min-oracle-ablation-drop", type=float, default=0.5)
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument("--save-dir", default="MY-V2/jscc-f/explore-2/checkpoints-receiver")
    parser.add_argument("--version", default="cnn-fsq-k4913-rx-v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--sender-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--latest-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-predictor-checkpoint", default="")
    parser.add_argument(
        "--ar-base-checkpoint",
        default="",
        help="Frozen direct_q receiver checkpoint used as the base prior for ar_residual_index.",
    )
    parser.add_argument(
        "--init-receiver-checkpoint",
        default="",
        help=(
            "Initialize predictor plus adapted sender tokenizer/combiner from a receiver checkpoint, "
            "then construct the requested receiver topology with a fresh optimizer."
        ),
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-shapes", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_residual_args(args)
    validate_qonly_highres_args(args)
    validate_sender_aligned_args(args)
    validate_ar_history_args(args)
    if args.smoke_shapes:
        smoke_shapes(args)
        return
    train(args)


if __name__ == "__main__":
    main()
