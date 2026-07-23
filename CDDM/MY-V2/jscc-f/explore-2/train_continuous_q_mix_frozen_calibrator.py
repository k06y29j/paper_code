#!/usr/bin/env python3
"""Calibrate the frozen e60 continuous-q mix without changing its decoder.

This is an independent opt-in follow-up to ``train_continuous_q_mix_receiver``.
It reconstructs the original train-selected top-3 generators, restores the
*EMA* D2 and combiner from the archived e60 checkpoint, and freezes all four
networks.  The deployed graph is strictly::

    (z1, x1) -> {frozen G_i} -> simplex mix -> channel affine -> q2_hat
              -> frozen e60 EMA D2 -> u2_hat
              -> frozen e60 EMA combiner(x1, u2_hat) -> x2_hat

Only three global simplex logits and per-q-channel affine parameters ``a,b``
are trained by default.  A zero-output q-space bottleneck adapter is available
as an explicit opt-in.  The source image is introduced only after receiver
forward as the RandomCrop training objective; validation is always the full
100-image CenterCrop split.  Validation never selects a checkpoint: the final
epoch is the result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_continuous_q_mix_receiver as qmix  # noqa: E402
import train_continuous_q_receiver as continuous  # noqa: E402
from contracts import (  # noqa: E402
    assert_div2k_crop_protocol,
    assert_receiver_only_module,
    assert_training_targets_are_not_inputs,
    make_receiver_condition,
)


EXPECTED_E60_DELTA = 0.4898087310791013
EXPECTED_E60_EPOCH = 60


class ZeroInitQResidualAdapter(nn.Module):
    """Small q-only 1x1 bottleneck whose initial output is exactly zero."""

    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        if int(width) < 1:
            raise ValueError("adapter width must be positive")
        self.reduce = nn.Conv2d(int(channels), int(width), 1)
        self.expand = nn.Conv2d(int(width), int(channels), 1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.expand(F.silu(self.reduce(value)))


class CalibratedTrainSelectedQMix(nn.Module):
    """Frozen receiver generators plus a trainable low-dimensional q calibrator."""

    def __init__(
        self,
        members: Sequence[nn.Module],
        weights: Sequence[float],
        *,
        channels: int,
        adapter_type: str = "none",
        adapter_width: int = 32,
    ) -> None:
        super().__init__()
        if len(members) != 3:
            raise ValueError(f"this strict top-3 calibrator needs exactly 3 members, got {len(members)}")
        self.members = nn.ModuleList(list(members))
        base_weights = qmix._normalise_weights(  # pylint: disable=protected-access
            weights, members=len(self.members), device=torch.device("cpu")
        )
        self.register_buffer("base_weights", base_weights)
        # These are relative logits around the fixed train-selected simplex.
        # The multiplicative form below is mathematically
        # softmax(log(base_weights)+simplex_logits), while preserving the
        # archived float32 weights *exactly* when all relative logits are zero.
        self.simplex_logits = nn.Parameter(torch.zeros_like(base_weights))
        self.channel_a = nn.Parameter(torch.ones(1, int(channels), 1, 1))
        self.channel_b = nn.Parameter(torch.zeros(1, int(channels), 1, 1))
        checked_adapter = str(adapter_type)
        if checked_adapter == "none":
            self.adapter: nn.Module | None = None
        elif checked_adapter == "bottleneck-1x1":
            self.adapter = ZeroInitQResidualAdapter(int(channels), int(adapter_width))
        else:
            raise ValueError(f"unsupported adapter type {checked_adapter!r}")
        self.adapter_type = checked_adapter
        for member in self.members:
            assert_receiver_only_module(member)
            member.requires_grad_(False)
            member.eval()
        assert_receiver_only_module(self)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Generator batch-norm/dropout state is part of the fixed deployment
        # function and must never drift during calibration.
        for member in self.members:
            member.eval()
        return self

    def simplex_weights(self) -> torch.Tensor:
        stable_logits = self.simplex_logits - self.simplex_logits.max()
        unnormalised = self.base_weights.to(self.simplex_logits) * torch.exp(stable_logits)
        return unnormalised / unnormalised.sum()

    def forward(self, condition):
        condition.validate()
        with torch.no_grad():
            member_q = [member(condition) for member in self.members]
        stacked = torch.stack(member_q, dim=0)
        weights = self.simplex_weights().to(device=stacked.device, dtype=stacked.dtype)
        view = weights.view(len(member_q), *([1] * (stacked.ndim - 1)))
        mixed = (stacked * view).sum(dim=0)
        calibrated = self.channel_a.to(dtype=mixed.dtype) * mixed + self.channel_b.to(
            dtype=mixed.dtype
        )
        if self.adapter is not None:
            calibrated = calibrated + self.adapter(mixed)
        return calibrated

    def regularization(self) -> dict[str, torch.Tensor]:
        weights = self.simplex_weights()
        simplex_anchor = (weights - self.base_weights.to(weights)).square().mean()
        affine_anchor = (self.channel_a - 1.0).square().mean() + self.channel_b.square().mean()
        if self.adapter is None:
            adapter_anchor = self.channel_a.new_zeros(())
        else:
            values = [parameter.square().mean() for parameter in self.adapter.parameters()]
            adapter_anchor = torch.stack(values).mean()
        return {
            "simplex_anchor": simplex_anchor,
            "affine_anchor": affine_anchor,
            "adapter_anchor": adapter_anchor,
        }

    def calibration_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only learned calibration state, never frozen generator weights."""

        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }


class FrozenPostQCalibratedReceiver(continuous.ContinuousReceiver):
    """Continuous receiver that keeps its restored D2/combiner frozen and in eval."""

    generator: CalibratedTrainSelectedQMix

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.d2.eval()
        self.combiner.eval()
        for member in self.generator.members:
            member.eval()
        return self


def _resolved(value: str | Path) -> Path:
    return continuous.resolve_path(str(value))


def validate_e60_start(
    payload: dict[str, Any],
    *,
    path: Path,
    mix: dict[str, Any],
    spec_path: Path,
    expected_epoch: int,
    expected_delta: float,
) -> None:
    """Reject any drift from the archived full-valid e60 EMA starting point."""

    if str(payload.get("stage", "")) != "explore2_continuous_q_mix_receiver":
        raise ValueError(f"not an explore-2 q-mix receiver checkpoint: {path}")
    if int(payload.get("epoch", -1)) != int(expected_epoch):
        raise ValueError(
            f"strict calibrator requires epoch {expected_epoch}, got {payload.get('epoch')}"
        )
    metrics = dict(payload.get("metrics", {}))
    if float(metrics.get("full_validation", 0.0)) != 1.0 or int(
        metrics.get("evaluated_images", 0)
    ) != 100:
        raise ValueError("strict e60 starting checkpoint lacks full DIV2K valid-100 metrics")
    observed = float(metrics.get("delta_x1", float("nan")))
    if not math.isfinite(observed) or abs(observed - float(expected_delta)) > 1e-12:
        raise ValueError(
            "strict e60 starting delta differs: "
            f"checkpoint={observed:.15f} expected={float(expected_delta):.15f}"
        )
    saved_mix = dict(payload.get("fixed_q_mix", {}))
    qmix._assert_resume_fixed_mix(saved_mix, mix)  # pylint: disable=protected-access
    if _resolved(saved_mix.get("spec_path", "")) != spec_path:
        raise ValueError("e60 checkpoint fixed-q spec path differs from --ensemble-spec")
    post = payload.get("ema_decoder_combiner_state_dict")
    if not isinstance(post, dict) or set(post) != {"d2", "combiner"}:
        raise ValueError("e60 checkpoint lacks exactly EMA D2+combiner state")
    contract = dict(payload.get("receiver_contract", {}))
    if list(contract.get("deployment_inputs", [])) != ["z1", "x1"]:
        raise ValueError("e60 starting checkpoint lacks z1/x1-only deployment contract")
    if list(contract.get("forbidden_inputs", [])) != ["img", "z2", "q2", "oracle_indices"]:
        raise ValueError("e60 starting checkpoint no-leak contract differs")


def _loader_args(saved: dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    bridge = SimpleNamespace(
        cpu=bool(args.cpu),
        batch_size=int(args.batch_size),
        test_batch=int(args.test_batch),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.val_num_workers),
        max_train_batches=int(args.max_train_batches),
        max_val_batches=0,
        data_dir=str(args.data_dir),
        source_checkpoint="",
        d2_type=None,
    )
    return qmix._compatible_args(saved, bridge)  # pylint: disable=protected-access


def build_frozen_receiver(
    source,
    *,
    payload: dict[str, Any],
    topology_args: argparse.Namespace,
    members: Sequence[nn.Module],
    weights: Sequence[float],
    adapter_type: str,
    adapter_width: int,
    device: torch.device,
) -> FrozenPostQCalibratedReceiver:
    initialized = continuous.build_receiver(source, topology_args, device)
    post = dict(payload["ema_decoder_combiner_state_dict"])
    initialized.d2.load_state_dict(dict(post["d2"]), strict=True)
    initialized.combiner.load_state_dict(dict(post["combiner"]), strict=True)
    initialized.d2.requires_grad_(False).eval()
    initialized.combiner.requires_grad_(False).eval()
    generator = CalibratedTrainSelectedQMix(
        members,
        weights,
        channels=int(topology_args.embedding_dim),
        adapter_type=str(adapter_type),
        adapter_width=int(adapter_width),
    ).to(device)
    receiver = FrozenPostQCalibratedReceiver(
        generator, initialized.d2, initialized.combiner
    ).to(device)
    del initialized
    continuous.assert_continuous_receiver_contract(receiver)
    assert_frozen_post_q_contract(receiver)
    return receiver


def assert_frozen_post_q_contract(receiver: FrozenPostQCalibratedReceiver) -> None:
    assert_receiver_only_module(receiver)
    continuous.assert_continuous_receiver_contract(receiver)
    if any(parameter.requires_grad for parameter in receiver.d2.parameters()):
        raise AssertionError("e60 EMA D2 must be frozen")
    if any(parameter.requires_grad for parameter in receiver.combiner.parameters()):
        raise AssertionError("e60 EMA combiner must be frozen")
    if receiver.d2.training or receiver.combiner.training:
        raise AssertionError("e60 EMA D2+combiner must remain in eval mode")
    if any(member.training for member in receiver.generator.members):
        raise AssertionError("all three q generators must remain in eval mode")
    if any(
        parameter.requires_grad
        for member in receiver.generator.members
        for parameter in member.parameters()
    ):
        raise AssertionError("all three q generators must be frozen")
    trainable = {
        name for name, parameter in receiver.named_parameters() if parameter.requires_grad
    }
    allowed = {
        f"generator.{name}"
        for name, parameter in receiver.generator.named_parameters()
        if parameter.requires_grad
    }
    if trainable != allowed or not trainable:
        raise AssertionError(
            f"only calibrator parameters may train: trainable={sorted(trainable)}"
        )


def optimizer_groups(
    generator: CalibratedTrainSelectedQMix, args: argparse.Namespace
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = [
        {
            "name": "simplex_logits",
            "params": [generator.simplex_logits],
            "lr": float(args.simplex_lr),
        },
        {
            "name": "channel_affine",
            "params": [generator.channel_a, generator.channel_b],
            "lr": float(args.affine_lr),
        },
    ]
    if generator.adapter is not None:
        groups.append(
            {
                "name": "q_residual_adapter",
                "params": list(generator.adapter.parameters()),
                "lr": float(args.adapter_lr),
            }
        )
    grouped = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in generator.parameters() if parameter.requires_grad]
    if len({id(value) for value in grouped}) != len(grouped):
        raise AssertionError("calibrator optimizer parameter groups overlap")
    if {id(value) for value in grouped} != {id(value) for value in trainable}:
        raise AssertionError("calibrator optimizer groups do not cover exactly trainable parameters")
    return groups


def _final_loss(
    per_image_mse: torch.Tensor, hard_weights: torch.Tensor, args: argparse.Namespace
) -> torch.Tensor:
    if str(args.final_loss) == "mse":
        return (hard_weights * per_image_mse).mean()
    if str(args.final_loss) == "log-mse":
        return float(args.log_mse_scale) * (
            hard_weights * torch.log(per_image_mse.clamp_min(1e-8))
        ).mean()
    raise ValueError(f"unsupported --final-loss {args.final_loss!r}")


def run_epoch(
    loader,
    *,
    source,
    receiver: FrozenPostQCalibratedReceiver,
    optimizer: optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
    train: bool,
) -> dict[str, Any]:
    source.e1.eval()
    source.d1.eval()
    receiver.train(train)
    assert_frozen_post_q_contract(receiver)
    totals: dict[str, float] = {}
    examples = 0
    audited = False
    max_batches = int(args.max_train_batches) if train else 0
    for batch_index, (image, _label) in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        image = image.to(device, non_blocking=True)
        with torch.no_grad():
            layer1 = source.layer1(image)
            condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
        if train:
            if optimizer is None:
                raise AssertionError("training calibrator needs an optimizer")
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            output = receiver(condition)
            if not audited:
                assert_training_targets_are_not_inputs(
                    receiver, condition, source_targets={"img": image}
                )
                audited = True
            per_image_mse = (
                output["x2_hat"].float() - image.float()
            ).square().flatten(1).mean(dim=1)
            hard_weights = qmix._hard_weights(  # pylint: disable=protected-access
                condition.x1, image, args
            )
            loss_recon = _final_loss(per_image_mse, hard_weights, args)
            regularization = receiver.generator.regularization()
            loss = (
                loss_recon
                + float(args.lambda_simplex_anchor) * regularization["simplex_anchor"]
                + float(args.lambda_affine_anchor) * regularization["affine_anchor"]
                + float(args.lambda_adapter_anchor) * regularization["adapter_anchor"]
            )
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in receiver.generator.parameters()
                        if parameter.requires_grad
                    ],
                    float(args.grad_clip_norm),
                )
                optimizer.step()
        batch = int(image.shape[0])
        values = {
            "loss": float(loss.detach()),
            "loss_recon": float(loss_recon.detach()),
            "loss_final_mse": float(per_image_mse.detach().mean()),
            "loss_simplex_anchor": float(regularization["simplex_anchor"].detach()),
            "loss_affine_anchor": float(regularization["affine_anchor"].detach()),
            "loss_adapter_anchor": float(regularization["adapter_anchor"].detach()),
            "hard_weight_mean": float(hard_weights.detach().mean()),
            "psnr_x1": float(continuous.psnr_per_image(condition.x1, image).mean()),
            "psnr_pred": float(
                continuous.psnr_per_image(output["x2_hat"], image).mean()
            ),
            "q2_hat_rms": float(output["q2_hat"].detach().square().mean().sqrt()),
            "receiver_only_audit": 1.0,
            "frozen_post_q_audit": 1.0,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * batch
        examples += batch
    if examples < 1:
        raise RuntimeError("calibrator epoch processed no images")
    metrics: dict[str, Any] = {
        name: value / examples for name, value in totals.items()
    }
    metrics["delta_x1"] = metrics["psnr_pred"] - metrics["psnr_x1"]
    metrics["evaluated_images"] = float(examples)
    metrics["full_validation"] = float((not train) and examples == len(loader.dataset))
    weights = receiver.generator.simplex_weights().detach()
    metrics["simplex_weights"] = [float(value) for value in weights.cpu()]
    metrics["channel_a_mean"] = float(receiver.generator.channel_a.detach().mean())
    metrics["channel_a_min"] = float(receiver.generator.channel_a.detach().min())
    metrics["channel_a_max"] = float(receiver.generator.channel_a.detach().max())
    metrics["channel_b_rms"] = float(
        receiver.generator.channel_b.detach().square().mean().sqrt()
    )
    return metrics


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(name): _jsonable(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    receiver: FrozenPostQCalibratedReceiver,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    mix: dict[str, Any],
    e60_path: Path,
    e60_payload: dict[str, Any],
    initial_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    calibration_state = receiver.generator.calibration_state_dict()
    torch.save(
        {
            "stage": "explore2_continuous_q_mix_frozen_post_q_calibrator",
            "epoch": int(epoch),
            "args": {name: value for name, value in vars(args).items() if not name.startswith("_")},
            "calibration_state_dict": calibration_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "fixed_q_mix": _jsonable(mix),
            "frozen_post_q": {
                "checkpoint": str(e60_path),
                "state": "ema_decoder_combiner_state_dict",
                "epoch": int(e60_payload["epoch"]),
                "archived_full_valid_metrics": _jsonable(e60_payload["metrics"]),
            },
            "initial_full_valid_metrics": _jsonable(initial_metrics),
            "metrics": _jsonable(metrics),
            "checkpoint_selection": "final_epoch_only",
            "receiver_contract": {
                "deployment_inputs": ["z1", "x1"],
                "generation": (
                    "frozen_top3_G(z1,x1) -> global_simplex -> channel_affine"
                    + (" -> zero_init_q_adapter" if receiver.generator.adapter is not None else "")
                    + " -> q2_hat"
                ),
                "decode_path": (
                    "q2_hat -> frozen_e60_EMA_D2 -> "
                    "frozen_e60_EMA_combiner(x1,u2_hat)"
                ),
                "trainable_parameters": sorted(calibration_state),
                "frozen_modules": ["three_q_generators", "e60_EMA_D2", "e60_EMA_combiner"],
                "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
                "image_role": "RandomCrop train-only supervised objective",
                "train_transform": "RandomCrop(256)+RandomHorizontalFlip+ToTensor",
                "validation_transform": "CenterCrop(256)+ToTensor",
                "validation_images": 100,
                "validation_selects_checkpoint": False,
            },
        },
        path,
    )
    print(f"saved checkpoint: {path}", flush=True)


def train(args: argparse.Namespace) -> None:
    continuous.nested.seed_everything(int(args.seed))
    spec_path = _resolved(args.ensemble_spec)
    mix = qmix.load_train_selected_mix(spec_path)
    e60_path = _resolved(args.e60_checkpoint)
    e60_payload = torch.load(e60_path, map_location="cpu", weights_only=False)
    validate_e60_start(
        e60_payload,
        path=e60_path,
        mix=mix,
        spec_path=spec_path,
        expected_epoch=int(args.expected_start_epoch),
        expected_delta=float(args.expected_start_delta),
    )
    topology = dict(e60_payload.get("receiver_init", {}))
    if str(topology.get("state", "")) != "ema":
        raise ValueError("archived e60 run did not originate from the expected EMA topology")
    topology_args = _loader_args(dict(topology.get("args", {})), args)
    source, train_loader, val_loader, device = continuous.build_source_and_loaders(topology_args)
    assert_div2k_crop_protocol(train_loader, val_loader)
    if len(val_loader.dataset) != 100:
        raise AssertionError(
            f"strict calibrator requires exactly 100 DIV2K validation images, got {len(val_loader.dataset)}"
        )
    members = qmix.load_fixed_mix_members(source, mix, topology_args, device)
    receiver = build_frozen_receiver(
        source,
        payload=e60_payload,
        topology_args=topology_args,
        members=members,
        weights=mix["weights"],
        adapter_type=str(args.adapter_type),
        adapter_width=int(args.adapter_width),
        device=device,
    )
    # Sender Layer2 is irrelevant after Layer1 source reconstruction and is
    # explicitly removed before the first calibrated receiver forward.
    source.e2 = nn.Identity().to(device)
    source.d2 = nn.Identity().to(device)
    source.combiner = nn.Identity().to(device)
    source.checkpoint = {}
    source.layer2_checkpoint = {}
    groups = optimizer_groups(receiver.generator, args)
    optimizer = optim.AdamW(groups, weight_decay=float(args.weight_decay))
    output_dir = _resolved(args.save_dir) / str(args.version)

    print("=== explore-2 | frozen e60 post-q continuous-q mix calibrator ===", flush=True)
    print("实验设计", flush=True)
    print(
        "  strict_start=e60 EMA full-valid delta=+0.4898087310791013dB; "
        "deployment=(z1,x1)->frozen top3 G->trainable global simplex + per-channel "
        "affine->q2_hat->frozen e60 EMA D2->frozen e60 EMA combiner; "
        "forbidden=img,z2,q2,oracle_indices; train=RandomCrop(256)+RandomHorizontalFlip; "
        "val=CenterCrop(256), full100; validation never selects model; final epoch only",
        flush=True,
    )
    print("loss设计", flush=True)
    print(
        f"  {args.final_loss}(x2_hat,img) with hard_example_power={args.hard_example_power:g}; "
        f"anchors=simplex:{args.lambda_simplex_anchor:g},"
        f"affine:{args.lambda_affine_anchor:g},adapter:{args.lambda_adapter_anchor:g}; "
        "no q2/z2/index target",
        flush=True,
    )
    print("模块选择", flush=True)
    print(
        f"  spec={spec_path}; selected={mix['selected_indices']}; "
        f"initial_weights={receiver.generator.base_weights.tolist()}; "
        f"post_q={e60_path}:EMA; D2=frozen/eval combiner=frozen/eval generators=3xfrozen/eval; "
        f"trainable=simplex_logits(3)+channel_a,b({2*int(topology_args.embedding_dim)}); "
        f"adapter={args.adapter_type} width={args.adapter_width}; "
        f"lr=simplex:{args.simplex_lr:g},affine:{args.affine_lr:g},adapter:{args.adapter_lr:g}; "
        "checkpoint_selection=final_epoch_only",
        flush=True,
    )

    initial_metrics = run_epoch(
        val_loader,
        source=source,
        receiver=receiver,
        optimizer=None,
        args=args,
        device=device,
        train=False,
    )
    if float(initial_metrics["full_validation"]) != 1.0:
        raise AssertionError("initial reproduction was not full valid-100")
    reproduction_error = abs(
        float(initial_metrics["delta_x1"]) - float(args.expected_start_delta)
    )
    initial_metrics["archived_delta_x1"] = float(args.expected_start_delta)
    initial_metrics["reproduction_abs_error"] = float(reproduction_error)
    if reproduction_error > float(args.start_reproduction_tolerance):
        raise AssertionError(
            "epoch-0 e60 reproduction drifted: "
            f"observed={initial_metrics['delta_x1']:.9f} "
            f"archived={float(args.expected_start_delta):.9f} "
            f"error={reproduction_error:.9g} tolerance={args.start_reproduction_tolerance:g}"
        )
    print(f"[frozen calibrator val 000 strict-start] {initial_metrics}", flush=True)

    final_metrics: dict[str, Any] = dict(initial_metrics)
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
        print(
            f"[frozen calibrator train {epoch:03d}/{args.epochs:03d}] "
            f"{train_metrics} time={time.time()-began:.1f}s",
            flush=True,
        )
        should_validate = epoch % int(args.val_every) == 0 or epoch == int(args.epochs)
        if should_validate:
            final_metrics = run_epoch(
                val_loader,
                source=source,
                receiver=receiver,
                optimizer=None,
                args=args,
                device=device,
                train=False,
            )
            if float(final_metrics["full_validation"]) != 1.0:
                raise AssertionError("calibrator validation was not full valid-100")
            final_metrics["goal_met"] = float(
                float(final_metrics["delta_x1"]) >= float(args.min_delta)
            )
            print(f"[frozen calibrator val {epoch:03d} monitor-only] {final_metrics}", flush=True)
        # A tiny latest checkpoint supports crash recovery/inspection, but is
        # never a validation-selected result and is overwritten each epoch.
        save_checkpoint(
            output_dir / "continuous_q_mix_frozen_calibrator_latest.pth",
            epoch=epoch,
            receiver=receiver,
            optimizer=optimizer,
            args=args,
            mix=mix,
            e60_path=e60_path,
            e60_payload=e60_payload,
            initial_metrics=initial_metrics,
            metrics=(final_metrics if should_validate else train_metrics),
        )

    final_metrics.update(
        continuous.receiver_ablations(
            val_loader, source=source, receiver=receiver, device=device
        )
    )
    final_metrics["goal_met"] = float(
        float(final_metrics["delta_x1"]) >= float(args.min_delta)
        and float(final_metrics["condition_shuffle_drop"]) >= float(args.min_condition_drop)
        and float(final_metrics["full_validation"]) == 1.0
    )
    print(f"[frozen calibrator final full-val epoch={args.epochs:03d}] {final_metrics}", flush=True)
    save_checkpoint(
        output_dir / "continuous_q_mix_frozen_calibrator_final.pth",
        epoch=int(args.epochs),
        receiver=receiver,
        optimizer=optimizer,
        args=args,
        mix=mix,
        e60_path=e60_path,
        e60_payload=e60_payload,
        initial_metrics=initial_metrics,
        metrics=final_metrics,
    )


def run_contract_smoke() -> None:
    """CPU proof of exact zero start, gradient ownership, no leak, and crops."""

    class SmokeMember(nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.projection = nn.Conv2d(4, 8, 1)
            with torch.no_grad():
                self.projection.weight.zero_()
                self.projection.bias.fill_(float(value))

        def forward(self, condition):
            condition.validate()
            return self.projection(condition.z1)

    class SmokeD2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Conv2d(8, 3, 1)

        def forward(self, q2_hat: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                self.projection(q2_hat), scale_factor=4, mode="bilinear", align_corners=False
            )

    class SmokeCombiner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Conv2d(6, 3, 1)

        def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.projection(torch.cat([x1, u2], dim=1)))

    torch.manual_seed(20260713)
    device = torch.device("cpu")
    members = [SmokeMember(1.0), SmokeMember(2.0), SmokeMember(4.0)]
    base = qmix.FixedTrainSelectedQMix(members, [0.6, 0.3, 0.1]).to(device)
    default_calibrated = CalibratedTrainSelectedQMix(
        members, [0.6, 0.3, 0.1], channels=8
    ).to(device)
    if default_calibrated.adapter is not None:
        raise AssertionError("default frozen calibrator unexpectedly enabled an adapter")
    default_groups = optimizer_groups(
        default_calibrated,
        SimpleNamespace(simplex_lr=1e-3, affine_lr=1e-3, adapter_lr=1e-3),
    )
    if [str(group["name"]) for group in default_groups] != [
        "simplex_logits",
        "channel_affine",
    ]:
        raise AssertionError("default low-dimensional optimizer groups differ")
    calibrated = CalibratedTrainSelectedQMix(
        members, [0.6, 0.3, 0.1], channels=8, adapter_type="bottleneck-1x1", adapter_width=4
    ).to(device)
    d2 = SmokeD2().to(device).requires_grad_(False).eval()
    combiner = SmokeCombiner().to(device).requires_grad_(False).eval()
    receiver = FrozenPostQCalibratedReceiver(calibrated, d2, combiner).to(device)
    receiver.train(True)
    assert_frozen_post_q_contract(receiver)
    condition = make_receiver_condition(
        torch.randn(2, 4, 4, 4, device=device),
        torch.rand(2, 3, 16, 16, device=device),
        detach=True,
    )
    with torch.no_grad():
        expected_q = base(condition)
        default_q = default_calibrated(condition)
        observed_q = receiver.generator(condition)
    torch.testing.assert_close(default_q, expected_q, rtol=0.0, atol=0.0)
    torch.testing.assert_close(observed_q, expected_q, rtol=0.0, atol=0.0)
    if not torch.equal(receiver.generator.channel_a, torch.ones_like(receiver.generator.channel_a)):
        raise AssertionError("per-channel affine a did not initialize to one")
    if not torch.equal(receiver.generator.channel_b, torch.zeros_like(receiver.generator.channel_b)):
        raise AssertionError("per-channel affine b did not initialize to zero")
    if not torch.equal(receiver.generator.simplex_weights(), receiver.generator.base_weights):
        raise AssertionError("global simplex did not exactly reproduce fixed top-3 weights")
    output = receiver(condition)
    target = torch.rand_like(output["x2_hat"])
    assert_training_targets_are_not_inputs(receiver, condition, source_targets={"img": target})
    F.mse_loss(output["x2_hat"], target).backward()
    if any(parameter.grad is not None for member in members for parameter in member.parameters()):
        raise AssertionError("frozen top-3 generator received gradient")
    if any(parameter.grad is not None for parameter in receiver.d2.parameters()):
        raise AssertionError("frozen D2 received gradient")
    if any(parameter.grad is not None for parameter in receiver.combiner.parameters()):
        raise AssertionError("frozen combiner received gradient")
    for name in ("simplex_logits", "channel_a", "channel_b"):
        if getattr(receiver.generator, name).grad is None:
            raise AssertionError(f"calibration parameter {name} received no gradient")
    if receiver.generator.adapter is None or receiver.generator.adapter.expand.weight.grad is None:
        raise AssertionError("opt-in zero-init adapter output head received no gradient")
    smoke_args = SimpleNamespace(simplex_lr=1e-3, affine_lr=1e-3, adapter_lr=1e-3)
    groups = optimizer_groups(receiver.generator, smoke_args)
    if [str(group["name"]) for group in groups] != [
        "simplex_logits",
        "channel_affine",
        "q_residual_adapter",
    ]:
        raise AssertionError("calibrator optimizer groups differ")
    train_loader = SimpleNamespace(
        dataset=SimpleNamespace(transform="RandomCrop(256)+RandomHorizontalFlip+ToTensor")
    )
    val_loader = SimpleNamespace(dataset=SimpleNamespace(transform="CenterCrop(256)+ToTensor"))
    assert_div2k_crop_protocol(train_loader, val_loader)
    print(
        "[PASS] frozen e60 post-q q-mix calibrator CPU contract "
        "top3_exact_zero_start=1 default_lowdim_only=1 simplex3=1 "
        "channel_a1_b0=1 optional_adapter_zero=1 "
        "generators_frozen_eval=1 d2_frozen_eval=1 combiner_frozen_eval=1 "
        "trainable_ownership=calibrator_only no_leak=1 receiver_inputs=z1,x1 "
        "train_RandomCrop=1 val_CenterCrop=1 checkpoint_selection=final_epoch_only",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ensemble-spec", default="")
    parser.add_argument("--e60-checkpoint", default="")
    parser.add_argument("--expected-start-epoch", type=int, default=EXPECTED_E60_EPOCH)
    parser.add_argument("--expected-start-delta", type=float, default=EXPECTED_E60_DELTA)
    parser.add_argument("--start-reproduction-tolerance", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--final-loss", choices=["mse", "log-mse"], default="mse")
    parser.add_argument("--log-mse-scale", type=float, default=0.01)
    parser.add_argument("--hard-example-power", type=float, default=0.5)
    parser.add_argument("--hard-example-min-weight", type=float, default=0.25)
    parser.add_argument("--hard-example-max-weight", type=float, default=4.0)
    parser.add_argument("--simplex-lr", type=float, default=1e-3)
    parser.add_argument("--affine-lr", type=float, default=3e-4)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lambda-simplex-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-affine-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-adapter-anchor", type=float, default=0.0)
    parser.add_argument(
        "--adapter-type", choices=["none", "bottleneck-1x1"], default="none"
    )
    parser.add_argument("--adapter-width", type=int, default=32)
    parser.add_argument("--min-delta", type=float, default=0.5)
    parser.add_argument("--min-condition-drop", type=float, default=0.1)
    parser.add_argument(
        "--save-dir",
        default="MY-V2/jscc-f/explore-2/checkpoints-continuous-q-mix",
    )
    parser.add_argument(
        "--version", default="cnn-continuous-q-mix-top3-frozen-calibrator-e60-v1"
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--smoke-contract",
        action="store_true",
        help="CPU-only zero-start/frozen/no-leak/crop contract; loads no checkpoint or data",
    )
    args = parser.parse_args()
    if bool(args.smoke_contract):
        return args
    if not str(args.ensemble_spec) or not str(args.e60_checkpoint):
        parser.error("--ensemble-spec and --e60-checkpoint are required unless --smoke-contract")
    if int(args.epochs) < 1 or int(args.val_every) < 1:
        parser.error("--epochs and --val-every must be positive")
    if int(args.batch_size) < 1 or int(args.test_batch) < 1:
        parser.error("--batch-size and --test-batch must be positive")
    if int(args.num_workers) < 0 or int(args.val_num_workers) < 0:
        parser.error("worker counts must be non-negative")
    if int(args.max_train_batches) < 0:
        parser.error("--max-train-batches must be non-negative")
    if float(args.start_reproduction_tolerance) < 0.0:
        parser.error("--start-reproduction-tolerance must be non-negative")
    if int(args.adapter_width) < 1:
        parser.error("--adapter-width must be positive")
    for name in ("simplex_lr", "affine_lr", "adapter_lr", "grad_clip_norm"):
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "weight_decay",
        "lambda_simplex_anchor",
        "lambda_affine_anchor",
        "lambda_adapter_anchor",
        "hard_example_power",
    ):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if float(args.hard_example_min_weight) <= 0.0 or float(args.hard_example_max_weight) <= 0.0:
        parser.error("hard-example bounds must be positive")
    if float(args.hard_example_min_weight) > float(args.hard_example_max_weight):
        parser.error("hard-example min cannot exceed max")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if bool(parsed.smoke_contract):
        run_contract_smoke()
    else:
        train(parsed)
