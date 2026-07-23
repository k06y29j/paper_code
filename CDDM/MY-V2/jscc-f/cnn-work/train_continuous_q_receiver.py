#!/usr/bin/env python3
"""Receiver-only continuous q2_hat baseline with a Layer1-initialized D2.

Deployment is strictly::

    (z1, x1) -> q2_hat -> receiver D2 -> combiner(x1, u2_hat) -> x2_hat

The sender image is a supervised training target only.  Sender z2/q2 and E2
are never constructed in the receiver forward path.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import math
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_layer2_vq_nested as nested
from contracts import (
    assert_div2k_crop_protocol,
    assert_receiver_only_module,
    assert_training_targets_are_not_inputs,
    make_receiver_condition,
)
from receiver_models import ReceiverTrunk


base = nested.base
ROOT = nested.CDDM_ROOT


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def identity_projection_(layer: nn.Conv2d) -> None:
    with torch.no_grad():
        layer.weight.zero_()
        diagonal = min(int(layer.weight.shape[0]), int(layer.weight.shape[1]))
        rows = torch.arange(diagonal, device=layer.weight.device)
        layer.weight[rows, rows, 0, 0] = 1.0
        if layer.bias is not None:
            layer.bias.zero_()


class ContinuousQGenerator(nn.Module):
    """Generate an arbitrary-D image embedding from receiver information."""

    def __init__(
        self,
        z1_channels: int,
        embedding_dim: int,
        *,
        hidden: int,
        blocks: int,
        attention_every: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.base = nn.Conv2d(int(z1_channels), int(embedding_dim), 1)
        identity_projection_(self.base)
        self.trunk = ReceiverTrunk(
            int(z1_channels),
            hidden=int(hidden),
            blocks=int(blocks),
            attention_every=int(attention_every),
            heads=int(heads),
            condition_mode="z1_x1",
        )
        self.residual = nn.Sequential(
            nn.Conv2d(int(hidden), int(hidden), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(int(hidden), int(embedding_dim), 3, padding=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, condition):
        condition.validate()
        return self.base(condition.z1) + self.residual(self.trunk(condition))

    def initial_q(self, z1: torch.Tensor) -> torch.Tensor:
        return self.base(z1)


class Layer1InitializedD2(nn.Module):
    """Map q2_hat back to z1 width, then decode with a trainable D1 clone."""

    def __init__(self, embedding_dim: int, z1_channels: int, d1: nn.Module) -> None:
        super().__init__()
        self.to_z1 = nn.Conv2d(int(embedding_dim), int(z1_channels), 1)
        identity_projection_(self.to_z1)
        self.decoder = copy.deepcopy(d1)

    def forward(self, q2_hat: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.to_z1(q2_hat))


def _normalization_groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(channels), int(maximum)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class QOnlyHighResolutionResidualD2(nn.Module):
    """Optional q-only high-resolution residual extension of the legacy D2.

    The base is the exact legacy ``D=embedding_dim -> D1-clone`` D2.  The
    new branch sees only ``q2_hat`` and upsamples it directly to RGB.  Its
    final head is zero initialized, so immediately after construction (and
    after a legacy-v9 initialization) this module is exactly equal to the
    legacy D2.  This gives a stable opt-in transition without introducing an
    x1/image side path into D2.
    """

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
        embedding_dim: int,
        z1_channels: int,
        d1: nn.Module,
        *,
        width: int = 64,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.width = int(width)
        self.blocks_per_scale = int(blocks)
        if self.width < 8:
            raise ValueError(f"q-only highres D2 width must be >=8, got {self.width}")
        if self.blocks_per_scale < 1:
            raise ValueError(
                f"q-only highres D2 blocks must be positive, got {self.blocks_per_scale}"
            )
        self.base = Layer1InitializedD2(int(embedding_dim), int(z1_channels), d1)
        self.stem = nn.Conv2d(int(embedding_dim), self.width, 3, padding=1)
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
        """Keep the added q-only residual exactly zero at initialization."""

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, q2_hat: torch.Tensor) -> torch.Tensor:
        # The exact one-argument signature is audited below.  In particular,
        # x1, img, z2 and true q2 cannot enter this decoder.
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


def _forward_parameter_names(module: nn.Module) -> list[str]:
    return [parameter.name for parameter in inspect.signature(module.forward).parameters.values()]


def assert_q_only_d2_module(module: nn.Module) -> None:
    """Require a D2 whose only dynamic input is generated q2_hat."""

    names = _forward_parameter_names(module)
    if names != ["q2_hat"]:
        raise AssertionError(
            "receiver D2 must be forward(q2_hat) only; "
            f"got parameters={names}"
        )


def assert_x1_u2_combiner_module(module: nn.Module) -> None:
    """Require the post-D2 fusion to retain the mandated x1/u2 interface."""

    names = _forward_parameter_names(module)
    if names != ["x1", "u2"]:
        raise AssertionError(
            "continuous receiver combiner must be forward(x1,u2) only; "
            f"got parameters={names}"
        )


class ReceiverCombiner(nn.Module):
    """Fuse x1 and the receiver D2 proposal, initialized to exact D2 output."""

    def __init__(self, width: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(9, int(width), 3, padding=1),
            nn.SiLU(),
        )
        body: list[nn.Module] = []
        for _ in range(int(blocks)):
            body.append(nested.CorrectionResidualBlock(int(width), 1))
        self.body = nn.Sequential(*body)
        self.head = nn.Conv2d(int(width), 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        features = torch.cat([x1, u2, u2 - x1], dim=1)
        correction = self.head(self.body(self.stem(features)))
        return (u2 + correction).clamp(0.0, 1.0)


class ReceiverUNetCombiner(nn.Module):
    """Multi-scale receiver image refiner with an exact u2 identity start."""

    def __init__(self, width: int, blocks: int) -> None:
        super().__init__()
        width = int(width)
        depth = max(1, int(blocks))

        def refine(channels: int) -> nn.Sequential:
            return nn.Sequential(
                *[
                    nested.CorrectionResidualBlock(channels, (1, 2, 1)[index % 3])
                    for index in range(depth)
                ]
            )

        self.stem = nn.Conv2d(9, width, 3, padding=1)
        self.enc0 = refine(width)
        self.down0 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.enc1 = refine(width * 2)
        self.down1 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.enc2 = refine(width * 4)
        self.down2 = nn.Conv2d(width * 4, width * 8, 4, stride=2, padding=1)
        self.middle = refine(width * 8)
        self.up2 = nn.Conv2d(width * 8 + width * 4, width * 4, 3, padding=1)
        self.dec2 = refine(width * 4)
        self.up1 = nn.Conv2d(width * 4 + width * 2, width * 2, 3, padding=1)
        self.dec1 = refine(width * 2)
        self.up0 = nn.Conv2d(width * 2 + width, width, 3, padding=1)
        self.dec0 = refine(width)
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _up(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        value = torch.cat([x1, u2, u2 - x1], dim=1)
        skip0 = self.enc0(self.stem(value))
        skip1 = self.enc1(F.silu(self.down0(skip0)))
        skip2 = self.enc2(F.silu(self.down1(skip1)))
        middle = self.middle(F.silu(self.down2(skip2)))
        value = self.dec2(F.silu(self.up2(torch.cat([self._up(middle, skip2), skip2], dim=1))))
        value = self.dec1(F.silu(self.up1(torch.cat([self._up(value, skip1), skip1], dim=1))))
        value = self.dec0(F.silu(self.up0(torch.cat([self._up(value, skip0), skip0], dim=1))))
        return (u2 + self.head(value)).clamp(0.0, 1.0)


class ReceiverHybridCombiner(nn.Module):
    """Preserve a trained local combiner, then add a zero-init multi-scale refiner."""

    def __init__(
        self,
        *,
        base_width: int,
        base_blocks: int,
        unet_width: int,
        unet_blocks: int,
    ) -> None:
        super().__init__()
        self.base = ReceiverCombiner(int(base_width), int(base_blocks))
        self.refiner = ReceiverUNetCombiner(int(unet_width), int(unet_blocks))

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        base = self.base(x1, u2)
        return self.refiner(x1, base)


class ContinuousReceiver(nn.Module):
    def __init__(self, generator: nn.Module, d2: nn.Module, combiner: nn.Module) -> None:
        super().__init__()
        self.generator = generator
        self.d2 = d2
        self.combiner = combiner

    def forward(self, condition):
        q2_hat = self.generator(condition)
        u2_raw = self.d2(q2_hat)
        u2_hat = u2_raw.clamp(0.0, 1.0)
        x2_hat = self.combiner(condition.x1, u2_hat)
        return {"q2_hat": q2_hat, "u2_hat": u2_hat, "x2_hat": x2_hat}


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    model_state = model.state_dict()
    for name, value in ema.state_dict().items():
        source = model_state[name]
        if value.is_floating_point():
            value.mul_(float(decay)).add_(source.detach(), alpha=1.0 - float(decay))
        else:
            value.copy_(source)


def build_source_and_loaders(args: argparse.Namespace):
    source_path = args.source_checkpoint or nested.DEFAULT_SOURCES[str(args.arch)]
    checkpoint = base.jsccf_io.load_checkpoint(str(resolve_path(source_path)))
    probe_args = argparse.Namespace(**checkpoint["args"])
    train_loader, val_loader, device = nested.build_loaders(args, probe_args)
    # Own the protocol here too: train must never silently become center crop.
    assert_div2k_crop_protocol(train_loader, val_loader)
    args.source_checkpoint = str(source_path)
    source = nested.load_source(args, device)
    return source, train_loader, val_loader, device


def build_receiver(source: nested.SourceLayer2, args: argparse.Namespace, device: torch.device):
    z1_channels = int(source.args.latent_ch)
    # Evaluation tools reconstruct their Namespace from historical checkpoints.
    # Those payloads predate the opt-in high-resolution D2 flags, so preserve
    # the original Layer1-clone topology when the fields are absent.
    d2_type = str(getattr(args, "d2_type", "layer1"))
    d2_highres_width = int(getattr(args, "d2_highres_width", 64))
    d2_highres_blocks = int(getattr(args, "d2_highres_blocks", 2))
    generator = ContinuousQGenerator(
        z1_channels,
        int(args.embedding_dim),
        hidden=int(args.hidden),
        blocks=int(args.blocks),
        attention_every=int(args.attention_every),
        heads=int(args.heads),
    )
    if d2_type == "layer1":
        # Historical default: preserve the v1--v12 topology and checkpoint
        # state keys byte-for-byte.
        d2: nn.Module = Layer1InitializedD2(int(args.embedding_dim), z1_channels, source.d1)
    elif d2_type == "qonly-highres-residual":
        d2 = QOnlyHighResolutionResidualD2(
            int(args.embedding_dim),
            z1_channels,
            source.d1,
            width=d2_highres_width,
            blocks=d2_highres_blocks,
        )
    else:
        raise ValueError(f"unsupported --d2-type {d2_type!r}")
    if str(args.combiner_type) == "hybrid":
        combiner = ReceiverHybridCombiner(
            base_width=int(args.base_combiner_width),
            base_blocks=int(args.base_combiner_blocks),
            unet_width=int(args.combiner_width),
            unet_blocks=int(args.combiner_blocks),
        )
    elif str(args.combiner_type) == "unet":
        combiner = ReceiverUNetCombiner(int(args.combiner_width), int(args.combiner_blocks))
    else:
        combiner = ReceiverCombiner(int(args.combiner_width), int(args.combiner_blocks))
    receiver = ContinuousReceiver(generator, d2, combiner).to(device)
    assert_continuous_receiver_contract(receiver)
    return receiver


def assert_continuous_receiver_contract(receiver: ContinuousReceiver) -> None:
    """Audit the complete deployment wiring, including the q-only D2."""

    assert_receiver_only_module(receiver)
    assert_receiver_only_module(receiver.generator)
    assert_q_only_d2_module(receiver.d2)
    assert_x1_u2_combiner_module(receiver.combiner)


def initialize_receiver_from_checkpoint(
    receiver: ContinuousReceiver, args: argparse.Namespace, device: torch.device
) -> None:
    if not args.init_checkpoint:
        return
    path = resolve_path(args.init_checkpoint)
    payload = torch.load(path, map_location=device, weights_only=False)
    if str(payload.get("stage", "")) != "explore2_continuous_q_receiver":
        raise ValueError(f"not a continuous-q receiver checkpoint: {path}")
    saved_args = dict(payload.get("args", {}))
    if int(saved_args.get("embedding_dim", -1)) != int(args.embedding_dim):
        raise ValueError(
            f"init embedding D mismatch: {saved_args.get('embedding_dim')} != {args.embedding_dim}"
        )
    state = payload.get("ema_state_dict") or payload["receiver_state_dict"]

    def component(prefix: str) -> dict[str, torch.Tensor]:
        marker = f"{prefix}."
        return {key[len(marker) :]: value for key, value in state.items() if key.startswith(marker)}

    receiver.generator.load_state_dict(component("generator"), strict=True)
    d2_type = str(getattr(args, "d2_type", "layer1"))
    saved_d2_type = str(saved_args.get("d2_type", "layer1"))
    saved_d2 = component("d2")
    if d2_type == "layer1":
        if saved_d2_type != "layer1":
            raise ValueError(
                "cannot load a highres-D2 checkpoint into legacy --d2-type layer1; "
                "select the matching D2 type explicitly"
            )
        # Exact legacy load: intentionally strict for v9 and every existing
        # default-topology checkpoint.
        receiver.d2.load_state_dict(saved_d2, strict=True)
        d2_status = "legacy_strict"
    elif d2_type == "qonly-highres-residual":
        if not isinstance(receiver.d2, QOnlyHighResolutionResidualD2):
            raise AssertionError("highres D2 selection did not construct QOnlyHighResolutionResidualD2")
        if saved_d2_type == "layer1":
            # v9 (and all pre-flag checkpoints) store exactly to_z1/decoder.
            # Load those fields strictly into the base and leave the newly
            # introduced q-only highres branch at its zero-output init.
            receiver.d2.base.load_state_dict(saved_d2, strict=True)
            receiver.d2.reset_new_branch_to_zero()
            d2_status = "legacy_base_strict+new_qonly_branch_zero"
        elif saved_d2_type == "qonly-highres-residual":
            receiver.d2.load_state_dict(saved_d2, strict=True)
            d2_status = "highres_strict"
        else:
            raise ValueError(
                f"unsupported saved d2_type={saved_d2_type!r}; expected layer1 or qonly-highres-residual"
            )
    else:
        raise ValueError(f"unsupported --d2-type {d2_type!r}")
    saved_combiner = component("combiner")
    if isinstance(receiver.combiner, ReceiverHybridCombiner):
        receiver.combiner.base.load_state_dict(saved_combiner, strict=True)
        combiner_status = "loaded_into_hybrid_base"
    elif str(saved_args.get("combiner_type", "residual")) == str(args.combiner_type):
        receiver.combiner.load_state_dict(saved_combiner, strict=True)
        combiner_status = "loaded"
    else:
        combiner_status = "reset_topology_changed"
    print(
        f"initialized continuous receiver from {path} epoch={payload.get('epoch')} "
        f"saved_delta={payload.get('metrics', {}).get('delta_x1')} "
        f"d2={d2_status} combiner={combiner_status}",
        flush=True,
    )


def configure_trainable_parameters(receiver: ContinuousReceiver, args: argparse.Namespace) -> None:
    """Optionally preserve an initialized receiver and train only its new refiner."""

    if not bool(args.freeze_initialized_backbone):
        return
    if not args.init_checkpoint:
        raise ValueError("--freeze-initialized-backbone requires --init-checkpoint")
    if not isinstance(receiver.combiner, ReceiverHybridCombiner):
        raise ValueError("--freeze-initialized-backbone requires --combiner-type hybrid")
    receiver.generator.requires_grad_(False)
    receiver.d2.requires_grad_(False)
    receiver.combiner.base.requires_grad_(False)
    receiver.combiner.refiner.requires_grad_(True)


def run_epoch(
    loader,
    *,
    source: nested.SourceLayer2,
    receiver: ContinuousReceiver,
    optimizer: optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    source.e1.eval()
    source.d1.eval()
    receiver.train(train)
    assert_continuous_receiver_contract(receiver)
    totals: dict[str, float] = {}
    count = 0
    audited = False
    max_batches = int(args.max_train_batches if train else args.max_val_batches)
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad():
            layer1 = source.layer1(imgs)
            condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            output = receiver(condition)
            if not audited:
                # img appears only after receiver(condition) has executed, as
                # a loss target.  The q-only D2 and x1/u2 combiner interfaces
                # are audited separately by assert_continuous_receiver_contract.
                assert_training_targets_are_not_inputs(
                    receiver,
                    condition,
                    source_targets={"img": imgs},
                )
                audited = True
            q_initial = receiver.generator.initial_q(condition.z1).detach()
            q_scale = q_initial.square().mean().clamp_min(1e-6)
            loss_q_residual = (output["q2_hat"] - q_initial).square().mean() / q_scale
            loss_u2 = F.mse_loss(output["u2_hat"].float(), imgs.float())
            final_mse_per_image = (
                output["x2_hat"].float() - imgs.float()
            ).square().flatten(1).mean(dim=1)
            loss_final_mse = final_mse_per_image.mean()
            if float(args.hard_example_power) > 0.0:
                baseline_mse = (
                    condition.x1.float() - imgs.float()
                ).square().flatten(1).mean(dim=1).detach()
                hard_weights = baseline_mse.pow(float(args.hard_example_power))
                hard_weights = hard_weights / hard_weights.mean().clamp_min(1e-8)
                hard_weights = hard_weights.clamp(
                    min=float(args.hard_example_min_weight),
                    max=float(args.hard_example_max_weight),
                )
            else:
                hard_weights = torch.ones_like(final_mse_per_image)
            if str(args.final_loss) == "log-mse":
                # Mean per-image PSNR is an affine transform of
                # -mean(log(MSE_i)).  The scale keeps gradient magnitudes near
                # the ordinary MSE baseline while preserving its optimum.
                loss_final = float(args.log_mse_scale) * (
                    hard_weights * torch.log(final_mse_per_image.clamp_min(1e-8))
                ).mean()
            else:
                loss_final = (hard_weights * final_mse_per_image).mean()
            loss = (
                float(args.lambda_final) * loss_final
                + float(args.lambda_u2) * loss_u2
                + float(args.lambda_q_residual) * loss_q_residual
            )
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(receiver.parameters(), float(args.grad_clip_norm))
                optimizer.step()
        batch = int(imgs.shape[0])
        values = {
            "loss": float(loss.detach()),
            "loss_final": float(loss_final.detach()),
            "loss_final_mse": float(loss_final_mse.detach()),
            "hard_weight_mean": float(hard_weights.detach().mean()),
            "loss_u2": float(loss_u2.detach()),
            "loss_q_residual": float(loss_q_residual.detach()),
            "psnr_x1": float(psnr_per_image(condition.x1, imgs).mean()),
            "psnr_pred": float(psnr_per_image(output["x2_hat"], imgs).mean()),
            "q2_hat_rms": float(output["q2_hat"].detach().square().mean().sqrt()),
            "receiver_only_audit": 1.0,
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch
        count += batch
    if count < 1:
        raise RuntimeError("receiver epoch processed no images")
    metrics = {key: value / count for key, value in totals.items()}
    metrics["delta_x1"] = metrics["psnr_pred"] - metrics["psnr_x1"]
    metrics["evaluated_images"] = float(count)
    metrics["full_validation"] = float((not train) and count == len(loader.dataset))
    return metrics


@torch.no_grad()
def receiver_ablations(
    loader,
    *,
    source: nested.SourceLayer2,
    receiver: ContinuousReceiver,
    device: torch.device,
) -> dict[str, float]:
    receiver.eval()
    totals = {"condition_shuffle_drop": 0.0, "pred_drop_zero": 0.0, "pred_drop_shuffle": 0.0}
    count = 0
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        layer1 = source.layer1(imgs)
        condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
        output = receiver(condition)
        batch = int(imgs.shape[0])
        if batch > 1:
            permutation = torch.roll(torch.arange(batch, device=device), shifts=1)
        else:
            permutation = torch.arange(batch, device=device)
        wrong = make_receiver_condition(
            condition.z1[permutation], condition.x1[permutation], detach=True
        )
        wrong_q = receiver.generator(wrong)
        # Decode wrong generated q against the current sample's x1.  This is
        # an ablation only; deployment always uses the matching condition.
        wrong_u2 = receiver.d2(wrong_q).clamp(0.0, 1.0)
        wrong_x2 = receiver.combiner(condition.x1, wrong_u2)
        zero_u2 = receiver.d2(torch.zeros_like(output["q2_hat"])).clamp(0.0, 1.0)
        zero_x2 = receiver.combiner(condition.x1, zero_u2)
        shuffled_q = output["q2_hat"][permutation]
        shuffled_u2 = receiver.d2(shuffled_q).clamp(0.0, 1.0)
        shuffled_x2 = receiver.combiner(condition.x1, shuffled_u2)
        base_psnr = psnr_per_image(output["x2_hat"], imgs)
        values = {
            "condition_shuffle_drop": base_psnr - psnr_per_image(wrong_x2, imgs),
            "pred_drop_zero": base_psnr - psnr_per_image(zero_x2, imgs),
            "pred_drop_shuffle": base_psnr - psnr_per_image(shuffled_x2, imgs),
        }
        for key, value in values.items():
            totals[key] += float(value.sum())
        count += batch
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    receiver: ContinuousReceiver,
    ema: ContinuousReceiver,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    metrics: dict[str, float],
    best: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "explore2_continuous_q_receiver",
            "epoch": int(epoch),
            "args": vars(args),
            "receiver_state_dict": receiver.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_delta_x1": float(best),
            "receiver_contract": {
                "deployment_inputs": ["z1", "x1"],
                "generated": "continuous_q2_hat",
                "decode_path": "q2_hat -> receiver_D2 -> combiner(x1,u2_hat)",
                "d2_type": str(args.d2_type),
                "d2_inputs": ["q2_hat"],
                "d2_forbidden_inputs": ["img", "z2", "q2", "x1"],
                "combiner_inputs": ["x1", "u2_hat"],
                "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
                "train_transform": "RandomCrop(256)+RandomHorizontalFlip+ToTensor",
                "validation_transform": "CenterCrop(256)+ToTensor",
            },
        },
        path,
    )
    print(f"saved checkpoint: {path}", flush=True)


def train(args: argparse.Namespace) -> None:
    nested.seed_everything(int(args.seed))
    source, train_loader, val_loader, device = build_source_and_loaders(args)
    receiver = build_receiver(source, args, device)
    initialize_receiver_from_checkpoint(receiver, args, device)
    configure_trainable_parameters(receiver, args)
    # load_source also materializes sender Layer2 modules for checkpoint
    # compatibility.  This receiver never calls them, so release them after D1
    # has been copied into the receiver D2.
    source.e2 = nn.Identity().to(device)
    source.d2 = nn.Identity().to(device)
    source.combiner = nn.Identity().to(device)
    source.checkpoint = {}
    source.layer2_checkpoint = {}
    ema = copy.deepcopy(receiver).eval()
    ema.requires_grad_(False)
    parameter_groups = []
    for name, module, learning_rate in (
        ("generator", receiver.generator, float(args.lr)),
        ("d2", receiver.d2, float(args.decoder_lr)),
        ("combiner", receiver.combiner, float(args.lr)),
    ):
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if parameters:
            parameter_groups.append({"params": parameters, "lr": learning_rate, "name": name})
    if not parameter_groups:
        raise RuntimeError("receiver has no trainable parameters")
    optimizer = optim.AdamW(parameter_groups, weight_decay=float(args.weight_decay))
    output_dir = resolve_path(args.save_dir) / str(args.version)
    print("=== cnn-work | continuous q2_hat receiver ===", flush=True)
    print("实验设计", flush=True)
    print(
        f"  frozen Layer1={args.arch}; deployment=(z1,x1)->q2_hat[D={args.embedding_dim}]"
        "->receiver D2->combiner->x2_hat; forbidden=img,z2,q2,oracle_indices; "
        "train=RandomCrop(256)+RandomHorizontalFlip; val=CenterCrop(256)",
        flush=True,
    )
    print("loss设计", flush=True)
    print(
        f"  {args.lambda_final:g}*{args.final_loss}(x2_hat,img) + "
        f"{args.lambda_u2:g}*MSE(u2,img) + "
        f"{args.lambda_q_residual:g}*normalized_q_residual; "
        f"hard_example_power={args.hard_example_power:g}",
        flush=True,
    )
    print("模块选择", flush=True)
    print(
        f"  generator=identity(z1)+ReceiverTrunk residual hidden={args.hidden}; "
        f"D2={args.d2_type}"
        f"{'(legacy Layer1 D1 clone)' if args.d2_type == 'layer1' else f'(q-only highres width={args.d2_highres_width} blocks/scale={args.d2_highres_blocks})'}; "
        f"combiner=receiver {args.combiner_type}; EMA validation; "
        f"freeze_initialized_backbone={int(args.freeze_initialized_backbone)}",
        flush=True,
    )
    best = float("-inf")
    for epoch in range(1, int(args.epochs) + 1):
        began = time.time()
        train_metrics = run_epoch(
            train_loader,
            source=source,
            receiver=receiver,
            optimizer=optimizer,
            args=args,
            device=device,
            train=True,
        )
        update_ema(ema, receiver, float(args.ema_decay))
        print(
            f"[continuous-q train {epoch:03d}/{args.epochs:03d}] {train_metrics} "
            f"time={time.time()-began:.1f}s",
            flush=True,
        )
        if epoch == 1 or epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            val_metrics = run_epoch(
                val_loader,
                source=source,
                receiver=ema,
                optimizer=None,
                args=args,
                device=device,
                train=False,
            )
            val_metrics.update(
                receiver_ablations(val_loader, source=source, receiver=ema, device=device)
            )
            val_metrics["goal_met"] = float(
                val_metrics["delta_x1"] >= float(args.min_delta)
                and val_metrics["condition_shuffle_drop"] >= float(args.min_condition_drop)
                and val_metrics["full_validation"] == 1.0
            )
            print(f"[continuous-q val {epoch:03d}] {val_metrics}", flush=True)
            latest = output_dir / "continuous_q_receiver_latest.pth"
            save_checkpoint(
                latest,
                epoch=epoch,
                receiver=receiver,
                ema=ema,
                optimizer=optimizer,
                args=args,
                metrics=val_metrics,
                best=best,
            )
            if val_metrics["delta_x1"] > best:
                best = val_metrics["delta_x1"]
                save_checkpoint(
                    output_dir / "continuous_q_receiver_best.pth",
                    epoch=epoch,
                    receiver=receiver,
                    ema=ema,
                    optimizer=optimizer,
                    args=args,
                    metrics=val_metrics,
                    best=best,
                )


def run_smoke_shapes() -> None:
    """Small CPU-only proof of q-only D2 shape, zero start, and contracts."""

    class SmokeD1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Conv2d(16, 3, 1)

        def forward(self, z1: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                self.projection(z1), scale_factor=16, mode="bilinear", align_corners=False
            )

    torch.manual_seed(20260713)
    device = torch.device("cpu")
    d2 = QOnlyHighResolutionResidualD2(
        embedding_dim=32,
        z1_channels=16,
        d1=SmokeD1(),
        width=16,
        blocks=1,
    ).to(device)
    assert_q_only_d2_module(d2)
    q2_hat = torch.randn(1, 32, 16, 16, device=device)
    with torch.no_grad():
        base = d2.base(q2_hat)
        proposal = d2(q2_hat)
    if tuple(proposal.shape) != (1, 3, 256, 256):
        raise AssertionError(f"q-only highres D2 shape mismatch: {tuple(proposal.shape)}")
    if not torch.equal(proposal, base):
        raise AssertionError("zero-initialized q-only residual D2 must equal legacy base at init")

    generator = ContinuousQGenerator(
        16, 32, hidden=16, blocks=1, attention_every=1, heads=4
    ).to(device)
    combiner = ReceiverCombiner(width=8, blocks=1).to(device)
    receiver = ContinuousReceiver(generator, d2, combiner).to(device)
    assert_continuous_receiver_contract(receiver)
    condition = make_receiver_condition(
        torch.randn(1, 16, 16, 16, device=device),
        torch.rand(1, 3, 256, 256, device=device),
        detach=True,
    )
    output = receiver(condition)
    target_img = torch.rand_like(output["x2_hat"])
    assert_training_targets_are_not_inputs(
        receiver,
        condition,
        source_targets={"img": target_img},
    )
    train_loader = SimpleNamespace(
        dataset=SimpleNamespace(transform="RandomCrop(256)+RandomHorizontalFlip+ToTensor")
    )
    val_loader = SimpleNamespace(dataset=SimpleNamespace(transform="CenterCrop(256)+ToTensor"))
    assert_div2k_crop_protocol(train_loader, val_loader)
    print(
        "[PASS] continuous-q qonly-highres-residual CPU smoke "
        f"q={tuple(q2_hat.shape)} u2={tuple(output['u2_hat'].shape)} "
        f"x2={tuple(output['x2_hat'].shape)} base_zero_start=1 "
        "d2_qonly=1 combiner_x1_u2=1 no_leak=1 crop_contract=1",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--arch", choices=["cnn", "swin"], default="cnn")
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument(
        "--d2-type",
        choices=["layer1", "qonly-highres-residual"],
        default="layer1",
        help=(
            "Receiver D2 topology. layer1 preserves every existing checkpoint exactly; "
            "qonly-highres-residual adds a zero-output q2_hat-only high-resolution branch."
        ),
    )
    parser.add_argument(
        "--d2-highres-width",
        type=int,
        default=64,
        help="Width of the opt-in qonly-highres-residual D2 branch.",
    )
    parser.add_argument(
        "--d2-highres-blocks",
        type=int,
        default=2,
        help="Residual blocks per 2x scale in the opt-in qonly-highres-residual D2 branch.",
    )
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--attention-every", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--combiner-width", type=int, default=48)
    parser.add_argument("--combiner-blocks", type=int, default=4)
    parser.add_argument(
        "--combiner-type", choices=["residual", "unet", "hybrid"], default="residual"
    )
    parser.add_argument("--base-combiner-width", type=int, default=48)
    parser.add_argument("--base-combiner-blocks", type=int, default=4)
    parser.add_argument("--lambda-final", type=float, default=1.0)
    parser.add_argument("--final-loss", choices=["mse", "log-mse"], default="mse")
    parser.add_argument("--log-mse-scale", type=float, default=0.01)
    parser.add_argument("--hard-example-power", type=float, default=0.0)
    parser.add_argument("--hard-example-min-weight", type=float, default=0.25)
    parser.add_argument("--hard-example-max-weight", type=float, default=4.0)
    parser.add_argument("--lambda-u2", type=float, default=0.25)
    parser.add_argument("--lambda-q-residual", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.5)
    parser.add_argument("--min-condition-drop", type=float, default=0.1)
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument(
        "--save-dir", default="MY-V2/jscc-f/cnn-work/checkpoints-continuous-q"
    )
    parser.add_argument("--version", default="cnn-continuous-q-d512-v1")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument(
        "--freeze-initialized-backbone",
        action="store_true",
        help="with a hybrid combiner, freeze generator/D2/base and train only the new refiner",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--smoke-shapes",
        action="store_true",
        help="CPU-only q-only-D2/no-leak/crop structural smoke; does not load data or train.",
    )
    args = parser.parse_args()
    if int(args.embedding_dim) < 16:
        raise ValueError("--embedding-dim must be at least the Layer1 latent width")
    if int(args.d2_highres_width) < 8 or int(args.d2_highres_blocks) < 1:
        raise ValueError("--d2-highres-width must be >=8 and --d2-highres-blocks must be positive")
    # load_source uses these fixed compatibility fields; they do not alter the
    # receiver architecture and Layer2 sender modules are never called.
    args.layer2_arch = "match"
    args.layer2_source_checkpoint = ""
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if bool(parsed_args.smoke_shapes):
        run_smoke_shapes()
    else:
        train(parsed_args)
