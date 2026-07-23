#!/usr/bin/env python3
"""Train one strict receiver D2+combiner behind a fixed train-selected q mix.

This is deliberately *not* an image-space ensemble.  A previously saved
``train-simplex`` report supplies a fixed, train-only selected set of
continuous receiver generators and their convex weights.  The deployed graph
is exactly::

    (z1, x1) -> {frozen G_i} -> sum_i(w_i G_i(z1,x1)) -> q2_hat
              -> one trainable D2 -> u2_hat
              -> one trainable combiner(x1,u2_hat) -> x2_hat

The q-mix weights and members are immutable during this script.  DIV2K train
uses RandomCrop and is the only supervision for the single D2/combiner;
validation is a CenterCrop-only final/monitoring report.  No sender E2, z2,
true q2, oracle index, or source image enters the receiver forward path.
"""

from __future__ import annotations

import argparse
import copy
import json
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

import train_continuous_q_receiver as continuous  # noqa: E402
from contracts import (  # noqa: E402
    assert_div2k_crop_protocol,
    assert_receiver_only_module,
    assert_training_targets_are_not_inputs,
    make_receiver_condition,
)


def _state(payload: dict[str, Any], state_kind: str) -> dict[str, torch.Tensor]:
    if str(state_kind) == "raw":
        return payload["receiver_state_dict"]
    return payload.get("ema_state_dict") or payload["receiver_state_dict"]


def _compatible_args(saved: dict[str, Any], cli: argparse.Namespace) -> argparse.Namespace:
    """Restore a historical continuous-q namespace without topology drift."""

    values = dict(saved)
    values.setdefault("combiner_type", "residual")
    values.setdefault("base_combiner_width", int(values.get("combiner_width", 48)))
    values.setdefault("base_combiner_blocks", int(values.get("combiner_blocks", 4)))
    values.setdefault("freeze_initialized_backbone", False)
    values.setdefault("init_checkpoint", "")
    values.setdefault("final_loss", "mse")
    values.setdefault("log_mse_scale", 0.01)
    # v1--v12 predate the q-only high-resolution D2 flags.  Missing means
    # exactly the legacy Layer1-initialized D2 topology.
    values.setdefault("d2_type", "layer1")
    values.setdefault("d2_highres_width", 64)
    values.setdefault("d2_highres_blocks", 2)
    # The q-mix trainer owns the opt-in receiver-side D2 choice.  Evaluation
    # callers intentionally omit these fields so they reconstruct the saved
    # topology byte-for-byte instead of accidentally changing it.
    requested_d2_type = getattr(cli, "d2_type", None)
    if requested_d2_type is not None:
        values["d2_type"] = str(requested_d2_type)
        values["d2_highres_width"] = int(getattr(cli, "d2_highres_width", 64))
        values["d2_highres_blocks"] = int(getattr(cli, "d2_highres_blocks", 2))
    values.setdefault("layer2_arch", "match")
    values.setdefault("layer2_source_checkpoint", "")
    values["cpu"] = bool(cli.cpu)
    values["batch_size"] = int(cli.batch_size)
    values["test_batch"] = int(cli.test_batch)
    values["num_workers"] = int(cli.num_workers)
    values["val_num_workers"] = int(cli.val_num_workers)
    values["max_train_batches"] = int(cli.max_train_batches)
    values["max_val_batches"] = int(cli.max_val_batches)
    if str(cli.data_dir):
        values["data_dir"] = str(cli.data_dir)
    if str(cli.source_checkpoint):
        values["source_checkpoint"] = str(cli.source_checkpoint)
    return argparse.Namespace(**values)


def _normalise_weights(
    weights: torch.Tensor | Sequence[float], *, members: int, device: torch.device
) -> torch.Tensor:
    result = torch.as_tensor(weights, dtype=torch.float32, device=device).flatten()
    if int(result.numel()) != int(members):
        raise ValueError(f"fixed q mix needs {members} weights, got {int(result.numel())}")
    if not bool(torch.isfinite(result).all()) or bool((result < 0.0).any()):
        raise ValueError("fixed q mix weights must be finite and non-negative")
    mass = result.sum()
    if not bool(mass > 0.0):
        raise ValueError("fixed q mix weights must have positive mass")
    return result / mass


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def load_train_selected_mix(path: Path) -> dict[str, Any]:
    """Load only a reproducible RandomCrop-train fixed simplex artifact.

    The strict checks reject a validation-selected, conditional, or manual
    image-output ensemble.  This trainer is intentionally for the existing
    fixed ``train-simplex`` q2 artifact only.
    """

    report = json.loads(path.read_text(encoding="utf-8"))
    selection = dict(report.get("weight_selection", {}))
    if str(selection.get("mode", "")) != "train-simplex":
        raise ValueError(
            "--ensemble-spec must be an eval_continuous_q_ensemble.py "
            "--selection-mode train-simplex report (fixed global q weights)"
        )
    if str(selection.get("selection_split", "")) != "DIV2K train":
        raise ValueError("ensemble spec was not selected on DIV2K train")
    transform = str(selection.get("selection_transform", ""))
    if "RandomCrop" not in transform or "CenterCrop" in transform:
        raise ValueError(
            "ensemble spec must record RandomCrop-only train selection, got "
            f"{transform!r}"
        )
    if str(selection.get("selection_loss_target", "")) != "train img only":
        raise ValueError("ensemble spec does not prove train-image-only selection supervision")
    contract = dict(report.get("receiver_contract", {}))
    forbidden = set(contract.get("forbidden_inputs", []))
    if not {"img", "z2", "q2", "oracle_indices"}.issubset(forbidden):
        raise ValueError("ensemble spec lacks the required receiver no-leak declaration")

    members = [str(value) for value in report.get("generator_checkpoints", [])]
    raw_weights = [float(value) for value in selection.get("weights", [])]
    selected = [int(value) for value in selection.get("selected_indices", [])]
    if not members or len(members) != len(raw_weights) or not selected:
        raise ValueError("ensemble spec needs generator_checkpoints, matching weights, and selected_indices")
    if len(set(selected)) != len(selected) or any(index < 0 or index >= len(members) for index in selected):
        raise ValueError("ensemble spec selected_indices are invalid")
    outside = [raw_weights[index] for index in range(len(raw_weights)) if index not in set(selected)]
    if any(abs(value) > 1e-6 for value in outside):
        raise ValueError(
            "ensemble spec has nonzero mass outside selected_indices; refusing to silently alter q mix"
        )
    selected_weights = [raw_weights[index] for index in selected]
    if any(value <= 0.0 for value in selected_weights):
        raise ValueError("all fixed selected q-mix members must have strictly positive mass")
    ensemble_state = str(report.get("ensemble_state", ""))
    if ensemble_state not in {"ema", "raw"}:
        raise ValueError(f"unsupported ensemble state in spec: {ensemble_state!r}")
    reference = str(report.get("reference_checkpoint", ""))
    if not reference:
        raise ValueError("ensemble spec has no reference_checkpoint for D2/combiner initialization")

    return {
        "spec_path": str(path),
        "reference_checkpoint": reference,
        "ensemble_state": ensemble_state,
        "selected_indices": selected,
        "member_paths": [members[index] for index in selected],
        "weights": selected_weights,
        "selection_provenance": {
            "mode": str(selection["mode"]),
            "selection_split": str(selection["selection_split"]),
            "selection_transform": transform,
            "selection_loss_target": str(selection["selection_loss_target"]),
            "calibration_epochs": int(selection.get("calibration_epochs", 0)),
            "calibration_batches_per_epoch": int(
                selection.get("calibration_batches_per_epoch", 0)
            ),
            "calibration_loss": str(selection.get("calibration_loss", "")),
        },
    }


class FixedTrainSelectedQMix(nn.Module):
    """A frozen q2 generator ensemble with fixed global convex weights.

    The module deliberately exposes only ``forward(condition)``.  It owns no
    image target, sender Layer2 tensor, or per-validation choice.  Calling
    ``train(True)`` still keeps its members in evaluation mode so batch norm or
    dropout state cannot drift while D2/combiner are trained.
    """

    def __init__(self, members: Sequence[nn.Module], weights: Sequence[float]) -> None:
        super().__init__()
        if not members:
            raise ValueError("fixed q mix needs at least one member")
        self.members = nn.ModuleList(list(members))
        normalised = _normalise_weights(
            weights, members=len(self.members), device=torch.device("cpu")
        )
        self.register_buffer("weights", normalised)
        for member in self.members:
            assert_receiver_only_module(member)
            member.requires_grad_(False)
            member.eval()
        assert_receiver_only_module(self)

    def train(self, mode: bool = True):  # type: ignore[override]
        # Frozen member normalization/dropout must remain deployment-identical.
        super().train(False)
        return self

    def forward(self, condition):
        condition.validate()
        with torch.no_grad():
            q_members = [member(condition) for member in self.members]
        stacked = torch.stack(q_members, dim=0)
        view = self.weights.to(device=stacked.device, dtype=stacked.dtype).view(
            len(q_members), *([1] * (stacked.ndim - 1))
        )
        return (stacked * view).sum(dim=0)


def _generator_from_payload(
    source, payload: dict[str, Any], args: argparse.Namespace, state_kind: str, device: torch.device
) -> nn.Module:
    generator = continuous.ContinuousQGenerator(
        int(source.args.latent_ch),
        int(args.embedding_dim),
        hidden=int(args.hidden),
        blocks=int(args.blocks),
        attention_every=int(args.attention_every),
        heads=int(args.heads),
    ).to(device)
    state = _state(payload, state_kind)
    prefix = "generator."
    generator_state = {
        name[len(prefix) :]: value for name, value in state.items() if name.startswith(prefix)
    }
    if not generator_state:
        raise ValueError("continuous-q checkpoint has no generator state")
    generator.load_state_dict(generator_state, strict=True)
    generator.requires_grad_(False)
    generator.eval()
    assert_receiver_only_module(generator)
    return generator


def load_fixed_mix_members(
    source, mix: dict[str, Any], init_args: argparse.Namespace, device: torch.device
) -> list[nn.Module]:
    members: list[nn.Module] = []
    for member_path in mix["member_paths"]:
        path = continuous.resolve_path(member_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if str(payload.get("stage", "")) != "explore2_continuous_q_receiver":
            raise ValueError(f"not a continuous-q generator checkpoint: {path}")
        saved_args = dict(payload.get("args", {}))
        if str(saved_args.get("arch", "")) != str(init_args.arch):
            raise ValueError(
                f"fixed q member Layer1 architecture mismatch: {path} has "
                f"{saved_args.get('arch')!r}, expected {init_args.arch!r}"
            )
        if int(saved_args.get("embedding_dim", -1)) != int(init_args.embedding_dim):
            raise ValueError(
                f"fixed q member embedding mismatch: {path} has "
                f"D={saved_args.get('embedding_dim')}, expected D={init_args.embedding_dim}"
            )
        member_args = argparse.Namespace(**saved_args)
        members.append(
            _generator_from_payload(source, payload, member_args, mix["ensemble_state"], device)
        )
    return members


def _component_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    marker = f"{prefix}."
    result = {name[len(marker) :]: value for name, value in state.items() if name.startswith(marker)}
    if not result:
        raise ValueError(f"continuous-q initialization checkpoint has no {prefix} state")
    return result


def load_post_q_modules_from_initialization(
    receiver: continuous.ContinuousReceiver,
    *,
    init_payload: dict[str, Any],
    init_args: argparse.Namespace,
    init_state: str,
) -> tuple[str, str]:
    """Strictly load one legacy/highres D2 and its matching combiner.

    A fixed q-mix never reuses the initialization checkpoint's generator.  It
    may, however, preserve a proven D2+combiner stack (including v12's hybrid
    combiner) as the base behind its own generated q2_hat.  For the opt-in
    qonly-highres-residual D2, a legacy D2 is loaded *only* into ``.base``;
    its new four-scale q2_hat-only RGB branch remains exactly zero at start.
    """

    state = _state(init_payload, init_state)
    saved_args = dict(init_payload.get("args", {}))
    saved_d2_type = str(saved_args.get("d2_type") or "layer1")
    requested_d2_type = str(getattr(init_args, "d2_type", "layer1"))
    saved_d2 = _component_state(state, "d2")
    if requested_d2_type == "layer1":
        if saved_d2_type != "layer1":
            raise ValueError(
                "cannot load a highres-D2 initialization into legacy --d2-type layer1; "
                "select --d2-type qonly-highres-residual explicitly"
            )
        receiver.d2.load_state_dict(saved_d2, strict=True)
        d2_status = "legacy_d2_strict"
    elif requested_d2_type == "qonly-highres-residual":
        if not isinstance(receiver.d2, continuous.QOnlyHighResolutionResidualD2):
            raise AssertionError("qonly highres selection did not build QOnlyHighResolutionResidualD2")
        if saved_d2_type == "layer1":
            receiver.d2.base.load_state_dict(saved_d2, strict=True)
            receiver.d2.reset_new_branch_to_zero()
            d2_status = "legacy_d2_base_strict+new_qonly_branch_zero"
        elif saved_d2_type == "qonly-highres-residual":
            receiver.d2.load_state_dict(saved_d2, strict=True)
            d2_status = "qonly_highres_d2_strict"
        else:
            raise ValueError(
                "unsupported initialization d2_type="
                f"{saved_d2_type!r}; expected layer1 or qonly-highres-residual"
            )
    else:
        raise ValueError(f"unsupported q-mix --d2-type {requested_d2_type!r}")
    receiver.combiner.load_state_dict(_component_state(state, "combiner"), strict=True)
    return d2_status, "combiner_strict"


def build_mixed_receiver(
    source,
    *,
    init_payload: dict[str, Any],
    init_args: argparse.Namespace,
    init_state: str,
    members: Sequence[nn.Module],
    weights: Sequence[float],
    device: torch.device,
) -> continuous.ContinuousReceiver:
    """Clone one strict D2/combiner, then replace only its q generator."""

    initialized = continuous.build_receiver(source, init_args, device)
    d2_status, combiner_status = load_post_q_modules_from_initialization(
        initialized,
        init_payload=init_payload,
        init_args=init_args,
        init_state=init_state,
    )
    fixed_mix = FixedTrainSelectedQMix(members, weights).to(device)
    receiver = continuous.ContinuousReceiver(fixed_mix, initialized.d2, initialized.combiner).to(device)
    # The initial receiver's old learnable generator is not retained anywhere
    # in the deployed receiver after this assignment.
    del initialized
    continuous.assert_continuous_receiver_contract(receiver)
    setattr(receiver, "_qmix_d2_init_status", d2_status)
    setattr(receiver, "_qmix_combiner_init_status", combiner_status)
    return receiver


def qonly_highres_warmup_active(args: argparse.Namespace, epoch: int) -> bool:
    return bool(
        str(getattr(args, "d2_type", "layer1")) == "qonly-highres-residual"
        and int(getattr(args, "qonly_highres_warmup_epochs", 0)) > 0
        and int(epoch) <= int(args.qonly_highres_warmup_epochs)
    )


def configure_trainable_parameters(
    receiver: continuous.ContinuousReceiver,
    args: argparse.Namespace,
    *,
    epoch: int,
    announce: bool = False,
) -> bool:
    """Set the per-epoch post-q trainability without ever unfreezing q-mix.

    The optimizer is constructed with all non-permanently-frozen post-q
    parameters.  During qonly-highres warmup gradients flow only through the
    new q2_hat-only branch.  At the transition the original D2 base and the
    exact initialized combiner become trainable without rebuilding optimizer
    state; frozen generators remain immutable throughout.
    """

    receiver.generator.requires_grad_(False)
    active = qonly_highres_warmup_active(args, int(epoch))
    if active:
        if not isinstance(receiver.d2, continuous.QOnlyHighResolutionResidualD2):
            raise AssertionError("qonly highres warmup requires QOnlyHighResolutionResidualD2")
        receiver.d2.base.requires_grad_(False)
        receiver.d2.base.eval()
        for module in (receiver.d2.stem, receiver.d2.stages, receiver.d2.residual_head):
            module.requires_grad_(True)
        receiver.combiner.requires_grad_(False)
        receiver.combiner.eval()
    else:
        receiver.d2.requires_grad_(not bool(args.freeze_d2))
        receiver.combiner.requires_grad_(not bool(args.freeze_combiner))
    setattr(args, "_qonly_highres_warmup_active", bool(active))
    if announce:
        if active:
            print(
                "[q-mix qonly-highres warmup] active: train=new four-scale q2_hat-only "
                "branch; freeze=base-D2+combiner+fixed-q generators",
                flush=True,
            )
        elif str(getattr(args, "d2_type", "layer1")) == "qonly-highres-residual":
            print(
                "[q-mix qonly-highres warmup] complete: unfreeze configured base-D2+combiner; "
                "fixed-q generators remain frozen",
                flush=True,
            )
    return bool(active)


def post_q_optimizer_parameter_groups(
    receiver: continuous.ContinuousReceiver, args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Build disjoint post-q optimizer groups without exposing sender state.

    The high-resolution q-only branch starts from a zero RGB head and benefits
    from a short, larger-LR warmup.  Keep it separate from the initialized
    legacy D2 base so unfreezing the base after warmup still uses ``--d2-lr``.
    A zero ``--d2-highres-lr`` is deliberately identical to the historical
    D2 learning rate.
    """

    parameter_groups: list[dict[str, Any]] = []
    if not bool(args.freeze_d2):
        if isinstance(receiver.d2, continuous.QOnlyHighResolutionResidualD2):
            base_parameters = list(receiver.d2.base.parameters())
            highres_parameters = [
                parameter
                for module in (receiver.d2.stem, receiver.d2.stages, receiver.d2.residual_head)
                for parameter in module.parameters()
            ]
            d2_parameters = list(receiver.d2.parameters())
            base_ids = {id(parameter) for parameter in base_parameters}
            highres_ids = {id(parameter) for parameter in highres_parameters}
            d2_ids = {id(parameter) for parameter in d2_parameters}
            if len(base_ids) != len(base_parameters) or len(highres_ids) != len(highres_parameters):
                raise AssertionError("qonly highres D2 contains duplicate parameters")
            if base_ids & highres_ids or (base_ids | highres_ids) != d2_ids:
                raise AssertionError("qonly highres optimizer groups do not partition D2 parameters")
            highres_lr = float(getattr(args, "d2_highres_lr", 0.0))
            if highres_lr == 0.0:
                highres_lr = float(args.d2_lr)
            parameter_groups.extend(
                [
                    {"params": base_parameters, "lr": float(args.d2_lr), "name": "d2_base"},
                    {"params": highres_parameters, "lr": highres_lr, "name": "d2_highres"},
                ]
            )
        else:
            parameter_groups.append(
                {"params": list(receiver.d2.parameters()), "lr": float(args.d2_lr), "name": "d2"}
            )
    if not bool(args.freeze_combiner):
        parameter_groups.append(
            {
                "params": list(receiver.combiner.parameters()),
                "lr": float(args.combiner_lr),
                "name": "combiner",
            }
        )

    seen: set[int] = set()
    for group in parameter_groups:
        parameters = list(group["params"])
        if not parameters:
            raise AssertionError(f"optimizer group {group['name']!r} is empty")
        duplicate = seen.intersection(id(parameter) for parameter in parameters)
        if duplicate:
            raise AssertionError("post-q optimizer parameter appears in more than one group")
        seen.update(id(parameter) for parameter in parameters)
    if not parameter_groups:
        raise RuntimeError("both D2 and combiner are permanently frozen")
    return parameter_groups


_RESUME_ARG_DEFAULTS: dict[str, Any] = {
    # Checkpoints made before the opt-in high-resolution D2 work have these
    # fields absent.  Their only possible topology is the legacy Layer1 D2.
    "d2_type": "layer1",
    "d2_highres_width": 64,
    "d2_highres_blocks": 2,
    "qonly_highres_warmup_epochs": 0,
    "d2_highres_lr": 0.0,
    "log_mse_scale": 0.01,
}


_RESUME_LOCKED_ARGS = (
    "version",
    "batch_size",
    "test_batch",
    "num_workers",
    "val_num_workers",
    "val_every",
    "max_train_batches",
    "max_val_batches",
    "lambda_final",
    "lambda_u2",
    "final_loss",
    "log_mse_scale",
    "hard_example_power",
    "hard_example_min_weight",
    "hard_example_max_weight",
    "d2_type",
    "d2_highres_width",
    "d2_highres_blocks",
    "qonly_highres_warmup_epochs",
    "d2_lr",
    "d2_highres_lr",
    "combiner_lr",
    "weight_decay",
    "grad_clip_norm",
    "ema_decay",
    "freeze_d2",
    "freeze_combiner",
    "min_delta",
    "min_condition_drop",
    "checkpoint_selection",
    "seed",
)


def _resume_value(values: dict[str, Any], name: str) -> Any:
    return values.get(name, _RESUME_ARG_DEFAULTS.get(name))


def _assert_resume_args(saved: dict[str, Any], args: argparse.Namespace) -> None:
    """Refuse a semantic training-contract change while resuming a run."""

    current = vars(args)
    for name in _RESUME_LOCKED_ARGS:
        expected = _resume_value(saved, name)
        actual = _resume_value(current, name)
        if isinstance(expected, float) or isinstance(actual, float):
            if expected is None or actual is None or float(expected) != float(actual):
                raise ValueError(
                    f"resume training-contract mismatch {name}: saved={expected!r} current={actual!r}"
                )
        elif expected != actual:
            raise ValueError(
                f"resume training-contract mismatch {name}: saved={expected!r} current={actual!r}"
            )


def _canonical_member_paths(values: Sequence[str]) -> list[str]:
    return [str(continuous.resolve_path(value)) for value in values]


def _assert_resume_fixed_mix(saved: dict[str, Any], mix: dict[str, Any]) -> None:
    """Require the resumed post-q decoder to sit behind exactly the same q mix."""

    if [int(value) for value in saved.get("selected_indices", [])] != [
        int(value) for value in mix["selected_indices"]
    ]:
        raise ValueError("resume fixed q-mix selected_indices differ")
    if _canonical_member_paths(saved.get("member_paths", [])) != _canonical_member_paths(
        mix["member_paths"]
    ):
        raise ValueError("resume fixed q-mix generator members differ")
    saved_weights = torch.as_tensor(saved.get("weights", []), dtype=torch.float64)
    current_weights = torch.as_tensor(mix["weights"], dtype=torch.float64)
    if saved_weights.shape != current_weights.shape or not bool(
        torch.equal(saved_weights, current_weights)
    ):
        raise ValueError("resume fixed q-mix weights differ")
    if dict(saved.get("selection_provenance", {})) != dict(mix["selection_provenance"]):
        raise ValueError("resume fixed q-mix train-selection provenance differs")


def restore_qmix_resume(
    path: Path,
    *,
    receiver: continuous.ContinuousReceiver,
    ema: continuous.ContinuousReceiver,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    mix: dict[str, Any],
    init_path: Path,
    init_state: str,
    output_dir: Path,
) -> tuple[int, float]:
    """Restore only a contract-identical fixed-q post-decoder continuation.

    Generator members are intentionally rebuilt from the current immutable
    train-selected spec rather than loaded from the resume payload.  This
    proves that no sender tensor, changed q weight, or stale generator state
    enters the resumed receiver.
    """

    resolved = continuous.resolve_path(str(path))
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if str(payload.get("stage", "")) != "explore2_continuous_q_mix_receiver":
        raise ValueError(f"not a q-mix receiver resume checkpoint: {resolved}")
    if resolved.parent != output_dir:
        raise ValueError(
            "resume checkpoint must belong to the exact --save-dir/--version output directory"
        )
    saved_epoch = int(payload.get("epoch", 0))
    if saved_epoch < 1:
        raise ValueError(f"resume checkpoint has invalid epoch={saved_epoch}")
    if int(args.epochs) <= saved_epoch:
        raise ValueError(
            f"--epochs={args.epochs} must exceed resumed epoch={saved_epoch}"
        )
    saved_metrics = dict(payload.get("metrics", {}))
    if float(saved_metrics.get("full_validation", 0.0)) != 1.0:
        raise ValueError("resume checkpoint must be saved from a full valid-100 evaluation")
    _assert_resume_args(dict(payload.get("args", {})), args)
    _assert_resume_fixed_mix(dict(payload.get("fixed_q_mix", {})), mix)

    saved_init = dict(payload.get("receiver_init", {}))
    saved_init_path = continuous.resolve_path(str(saved_init.get("checkpoint", "")))
    if saved_init_path != init_path or str(saved_init.get("state", "")) != str(init_state):
        raise ValueError("resume receiver initialization checkpoint/state differs")
    saved_contract = dict(payload.get("receiver_contract", {}))
    if list(saved_contract.get("deployment_inputs", [])) != ["z1", "x1"]:
        raise ValueError("resume checkpoint lacks the receiver-only deployment-input contract")
    if list(saved_contract.get("forbidden_inputs", [])) != ["img", "z2", "q2", "oracle_indices"]:
        raise ValueError("resume checkpoint lacks the required no-leak forbidden-input contract")
    if str(saved_contract.get("q_mix_selection_transform", "")) != (
        "RandomCrop(256)+RandomHorizontalFlip+ToTensor"
    ) or str(saved_contract.get("validation_transform", "")) != "CenterCrop(256)+ToTensor":
        raise ValueError("resume checkpoint crop protocol differs")

    raw_post = dict(payload.get("decoder_combiner_state_dict", {}))
    ema_post = dict(payload.get("ema_decoder_combiner_state_dict", {}))
    for label, target, state in (
        ("raw", receiver, raw_post),
        ("ema", ema, ema_post),
    ):
        if set(state) != {"d2", "combiner"}:
            raise ValueError(f"resume {label} checkpoint lacks exactly D2+combiner state")
        target.d2.load_state_dict(state["d2"], strict=True)
        target.combiner.load_state_dict(state["combiner"], strict=True)
    if ema.generator is not receiver.generator:
        raise AssertionError("resumed q-mix EMA must retain the shared frozen generator")

    optimizer_state = dict(payload.get("optimizer_state_dict", {}))
    saved_groups = list(optimizer_state.get("param_groups", []))
    current_groups = list(optimizer.state_dict().get("param_groups", []))
    if [str(group.get("name", "")) for group in saved_groups] != [
        str(group.get("name", "")) for group in current_groups
    ] or [len(group.get("params", [])) for group in saved_groups] != [
        len(group.get("params", [])) for group in current_groups
    ]:
        raise ValueError("resume optimizer parameter-group topology differs")
    optimizer.load_state_dict(optimizer_state)

    best = float(payload.get("best_delta_x1", saved_metrics.get("delta_x1", float("-inf"))))
    if not bool(torch.isfinite(torch.tensor(best))):
        raise ValueError(f"resume checkpoint has non-finite best_delta_x1={best}")
    return saved_epoch + 1, best


def _hard_weights(
    baseline: torch.Tensor, target: torch.Tensor, args: argparse.Namespace
) -> torch.Tensor:
    if float(args.hard_example_power) <= 0.0:
        return torch.ones(int(target.shape[0]), device=target.device, dtype=target.dtype)
    baseline_mse = (baseline.float() - target.float()).square().flatten(1).mean(dim=1).detach()
    weights = baseline_mse.pow(float(args.hard_example_power))
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights.clamp(
        min=float(args.hard_example_min_weight), max=float(args.hard_example_max_weight)
    )


def run_epoch(
    loader,
    *,
    source,
    receiver: continuous.ContinuousReceiver,
    optimizer: optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    source.e1.eval()
    source.d1.eval()
    receiver.train(train)
    receiver.generator.eval()
    if train and qonly_highres_warmup_active(args, int(getattr(args, "_current_epoch", 0))):
        if not isinstance(receiver.d2, continuous.QOnlyHighResolutionResidualD2):
            raise AssertionError("qonly highres warmup receiver D2 type drifted")
        # `receiver.train(True)` above may otherwise update normalization
        # buffers in the frozen legacy base/combiner.  Warmup must be only the
        # new q2_hat-only branch in both gradients and module state.
        receiver.d2.base.eval()
        receiver.combiner.eval()
    continuous.assert_continuous_receiver_contract(receiver)
    totals: dict[str, float] = {}
    examples = 0
    audited = False
    max_batches = int(args.max_train_batches if train else args.max_val_batches)
    for batch_index, (image, _label) in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        image = image.to(device, non_blocking=True)
        with torch.no_grad():
            layer1 = source.layer1(image)
            condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
        if train:
            if optimizer is None:
                raise AssertionError("training epoch needs an optimizer")
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            output = receiver(condition)
            # The receiver has already consumed the sole allowed typed
            # condition before the source image becomes a loss target.
            if not audited:
                assert_training_targets_are_not_inputs(
                    receiver, condition, source_targets={"img": image}
                )
                audited = True
            final_mse_per_image = (
                output["x2_hat"].float() - image.float()
            ).square().flatten(1).mean(dim=1)
            loss_final_mse = final_mse_per_image.mean()
            hard_weights = _hard_weights(condition.x1, image, args)
            if str(args.final_loss) == "log-mse":
                loss_final = float(args.log_mse_scale) * (
                    hard_weights * torch.log(final_mse_per_image.clamp_min(1e-8))
                ).mean()
            elif str(args.final_loss) == "mse":
                loss_final = (hard_weights * final_mse_per_image).mean()
            else:
                raise ValueError(f"unknown --final-loss {args.final_loss!r}")
            loss_u2 = F.mse_loss(output["u2_hat"].float(), image.float())
            loss = float(args.lambda_final) * loss_final + float(args.lambda_u2) * loss_u2
            if train:
                loss.backward()
                trainable = [parameter for parameter in receiver.parameters() if parameter.requires_grad]
                nn.utils.clip_grad_norm_(trainable, float(args.grad_clip_norm))
                optimizer.step()
        batch = int(image.shape[0])
        values = {
            "loss": float(loss.detach()),
            "loss_final": float(loss_final.detach()),
            "loss_final_mse": float(loss_final_mse.detach()),
            "loss_u2": float(loss_u2.detach()),
            "hard_weight_mean": float(hard_weights.detach().mean()),
            "psnr_x1": float(continuous.psnr_per_image(condition.x1, image).mean()),
            "psnr_pred": float(continuous.psnr_per_image(output["x2_hat"], image).mean()),
            "q2_hat_rms": float(output["q2_hat"].detach().square().mean().sqrt()),
            "receiver_only_audit": 1.0,
            "fixed_q_mix_audit": 1.0,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * batch
        examples += batch
    if examples < 1:
        raise RuntimeError("receiver epoch processed no images")
    metrics = {name: value / examples for name, value in totals.items()}
    metrics["delta_x1"] = metrics["psnr_pred"] - metrics["psnr_x1"]
    metrics["evaluated_images"] = float(examples)
    metrics["full_validation"] = float((not train) and examples == len(loader.dataset))
    return metrics


def _decoder_combiner_state(receiver: continuous.ContinuousReceiver) -> dict[str, dict[str, torch.Tensor]]:
    return {"d2": receiver.d2.state_dict(), "combiner": receiver.combiner.state_dict()}


@torch.no_grad()
def update_post_qmix_ema(
    ema: continuous.ContinuousReceiver,
    receiver: continuous.ContinuousReceiver,
    decay: float,
) -> None:
    """EMA only the trainable post-q-mix modules.

    The frozen fixed-q generator is deliberately shared between ``receiver``
    and ``ema`` to avoid holding a second ensemble in GPU memory.  Calling the
    generic whole-receiver EMA updater on that shared state is unsafe: its
    in-place ``mul_`` changes the source before ``add_`` reads it, which
    silently decays the supposedly immutable generators.  D2 and the
    combiner are separate module copies, so update exactly those two states.
    """

    if ema.generator is not receiver.generator:
        raise AssertionError("q-mix EMA must share the frozen generator exactly")
    checked_decay = float(decay)
    if not 0.0 <= checked_decay < 1.0:
        raise ValueError(f"EMA decay must be in [0,1), got {checked_decay}")
    for name in ("d2", "combiner"):
        ema_state = getattr(ema, name).state_dict()
        source_state = getattr(receiver, name).state_dict()
        if ema_state.keys() != source_state.keys():
            raise AssertionError(f"q-mix EMA {name} state keys differ")
        for key, value in ema_state.items():
            source = source_state[key]
            if value.is_floating_point():
                value.mul_(checked_decay).add_(source.detach(), alpha=1.0 - checked_decay)
            else:
                value.copy_(source)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    receiver: continuous.ContinuousReceiver,
    ema: continuous.ContinuousReceiver,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    init_path: Path,
    init_args: argparse.Namespace,
    init_state: str,
    mix: dict[str, Any],
    metrics: dict[str, float],
    best: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "explore2_continuous_q_mix_receiver",
            "epoch": int(epoch),
            "args": vars(args),
            "receiver_init": {
                "checkpoint": str(init_path),
                "state": str(init_state),
                "args": vars(init_args),
            },
            # Generator states stay in their immutable original checkpoints;
            # this saves only the one trained post-q-mix receiver.
            "fixed_q_mix": _jsonable(mix),
            "decoder_combiner_state_dict": _decoder_combiner_state(receiver),
            "ema_decoder_combiner_state_dict": _decoder_combiner_state(ema),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_delta_x1": float(best),
            "receiver_contract": {
                "deployment_inputs": ["z1", "x1"],
                "generation": "fixed_train_selected_sum_i(w_i*G_i(z1,x1)) -> q2_hat",
                "decode_path": "q2_hat -> one_receiver_D2 -> combiner(x1,u2_hat)",
                "fixed_member_indices": [int(value) for value in mix["selected_indices"]],
                "fixed_member_weights": [float(value) for value in mix["weights"]],
                "q_mix_selection_split": "DIV2K train",
                "q_mix_selection_transform": "RandomCrop(256)+RandomHorizontalFlip+ToTensor",
                "d2_inputs": ["q2_hat"],
                "d2_type": str(getattr(init_args, "d2_type", "layer1")),
                "d2_highres_width": int(getattr(init_args, "d2_highres_width", 64)),
                "d2_highres_blocks": int(getattr(init_args, "d2_highres_blocks", 2)),
                "qonly_highres_warmup_epochs": int(
                    getattr(args, "qonly_highres_warmup_epochs", 0)
                ),
                "combiner_inputs": ["x1", "u2_hat"],
                "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
                "image_role": "RandomCrop train-only supervised objective",
                "validation_transform": "CenterCrop(256)+ToTensor",
                "validation_does_not_update_q_mix": True,
            },
        },
        path,
    )
    print(f"saved checkpoint: {path}", flush=True)


def train(args: argparse.Namespace) -> None:
    continuous.nested.seed_everything(int(args.seed))
    spec_path = continuous.resolve_path(args.ensemble_spec)
    mix = load_train_selected_mix(spec_path)
    init_path = continuous.resolve_path(
        args.receiver_init_checkpoint or mix["reference_checkpoint"]
    )
    init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
    if str(init_payload.get("stage", "")) != "explore2_continuous_q_receiver":
        raise ValueError(f"not a continuous-q receiver initialization checkpoint: {init_path}")
    init_state = (
        str(mix["ensemble_state"])
        if str(args.receiver_init_state) == "auto"
        else str(args.receiver_init_state)
    )
    init_args = _compatible_args(dict(init_payload["args"]), args)
    source, train_loader, val_loader, device = continuous.build_source_and_loaders(init_args)
    assert_div2k_crop_protocol(train_loader, val_loader)
    if len(val_loader.dataset) != 100:
        raise AssertionError(
            f"final receiver report requires exactly 100 DIV2K validation images, got {len(val_loader.dataset)}"
        )
    members = load_fixed_mix_members(source, mix, init_args, device)
    receiver = build_mixed_receiver(
        source,
        init_payload=init_payload,
        init_args=init_args,
        init_state=init_state,
        members=members,
        weights=mix["weights"],
        device=device,
    )
    configure_trainable_parameters(receiver, args, epoch=1, announce=False)
    # `load_source` creates sender Layer2 only to resolve the Layer1 source
    # checkpoint.  Delete it before the first receiver batch.
    source.e2 = nn.Identity().to(device)
    source.d2 = nn.Identity().to(device)
    source.combiner = nn.Identity().to(device)
    source.checkpoint = {}
    source.layer2_checkpoint = {}

    ema = copy.deepcopy(receiver).eval()
    # Do not duplicate a five-member frozen generator ensemble for EMA.  The
    # shared module has no trainable/buffer updates and is deployment-fixed.
    ema.generator = receiver.generator
    ema.requires_grad_(False)
    parameter_groups = post_q_optimizer_parameter_groups(receiver, args)
    optimizer = optim.AdamW(parameter_groups, weight_decay=float(args.weight_decay))
    output_dir = continuous.resolve_path(args.save_dir) / str(args.version)
    start_epoch = 1
    best = float("-inf")
    if str(args.resume_checkpoint):
        start_epoch, best = restore_qmix_resume(
            continuous.resolve_path(args.resume_checkpoint),
            receiver=receiver,
            ema=ema,
            optimizer=optimizer,
            args=args,
            mix=mix,
            init_path=init_path,
            init_state=init_state,
            output_dir=output_dir,
        )

    print("=== explore-2 | fixed train-selected q mix -> one D2+combiner ===", flush=True)
    print("实验设计", flush=True)
    print(
        "  deployment=(z1,x1)->fixed sum_i(w_i*G_i(z1,x1))->q2_hat->one D2"
        "->combiner(x1,u2_hat)->x2_hat; forbidden=img,z2,q2,oracle_indices; "
        "train=RandomCrop(256)+RandomHorizontalFlip; val=CenterCrop(256)",
        flush=True,
    )
    print("loss设计", flush=True)
    print(
        f"  {args.lambda_final:g}*{args.final_loss}(x2_hat,img) + "
        f"{args.lambda_u2:g}*MSE(u2_hat,img); hard_example_power={args.hard_example_power:g}; "
        "q-mix weights/frozen generators receive no gradient",
        flush=True,
    )
    print("模块选择", flush=True)
    print(
        f"  fixed q spec={spec_path}; selected original members={mix['selected_indices']}; "
        f"weights={_normalise_weights(mix['weights'], members=len(mix['weights']), device=torch.device('cpu')).tolist()}; "
        f"D2={init_args.d2_type}; combiner={init_args.combiner_type}; "
        f"D2_init={getattr(receiver, '_qmix_d2_init_status', 'unknown')}; "
        f"combiner_init={getattr(receiver, '_qmix_combiner_init_status', 'unknown')}; "
        f"d2_lr={float(args.d2_lr):g}; "
        f"d2_highres_lr={float(getattr(args, 'd2_highres_lr', 0.0)):g}"
        "(0=--d2-lr); "
        f"combiner_lr={float(args.combiner_lr):g}; "
        f"freeze_d2={int(args.freeze_d2)} freeze_combiner={int(args.freeze_combiner)}; "
        f"qonly_highres_warmup_epochs={int(args.qonly_highres_warmup_epochs)}; "
        f"checkpoint_selection={args.checkpoint_selection}; "
        f"resume={'none' if not str(args.resume_checkpoint) else str(args.resume_checkpoint)} "
        f"start_epoch={start_epoch}",
        flush=True,
    )
    if int(start_epoch) > 1:
        print(
            f"[q-mix resume] checkpoint={continuous.resolve_path(args.resume_checkpoint)} "
            f"next_epoch={start_epoch:03d}/{args.epochs:03d} restored_best_delta={best:.6f}",
            flush=True,
        )
    for epoch in range(int(start_epoch), int(args.epochs) + 1):
        began = time.time()
        args._current_epoch = int(epoch)
        warmup_active = configure_trainable_parameters(
            receiver,
            args,
            epoch=epoch,
            announce=(
                epoch == 1
                or (
                    str(args.d2_type) == "qonly-highres-residual"
                    and int(args.qonly_highres_warmup_epochs) > 0
                    and epoch == int(args.qonly_highres_warmup_epochs) + 1
                )
            ),
        )
        train_metrics = run_epoch(
            train_loader,
            source=source,
            receiver=receiver,
            optimizer=optimizer,
            args=args,
            device=device,
            train=True,
        )
        train_metrics["qonly_highres_warmup_active"] = float(warmup_active)
        update_post_qmix_ema(ema, receiver, float(args.ema_decay))
        print(
            f"[q-mix receiver train {epoch:03d}/{args.epochs:03d}] {train_metrics} "
            f"time={time.time() - began:.1f}s",
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
            if val_metrics["full_validation"] == 1.0:
                val_metrics.update(
                    continuous.receiver_ablations(
                        val_loader, source=source, receiver=ema, device=device
                    )
                )
            else:
                # A partial validation smoke must never silently run a full
                # 100-image ablation or be mistaken for the final gate.
                val_metrics["condition_shuffle_drop"] = float("nan")
                val_metrics["pred_drop_zero"] = float("nan")
                val_metrics["pred_drop_shuffle"] = float("nan")
                val_metrics["ablations_skipped_partial_validation"] = 1.0
            val_metrics["goal_met"] = float(
                val_metrics["delta_x1"] >= float(args.min_delta)
                and val_metrics["condition_shuffle_drop"] >= float(args.min_condition_drop)
                and val_metrics["full_validation"] == 1.0
            )
            print(f"[q-mix receiver val {epoch:03d}] {val_metrics}", flush=True)
            is_new_full_val_best = bool(
                val_metrics["full_validation"] == 1.0 and val_metrics["delta_x1"] > best
            )
            if is_new_full_val_best:
                best = val_metrics["delta_x1"]
            # Partial validation is a smoke/throughput diagnostic only, so
            # do not write an infinite best value into its checkpoint.
            best_for_save = best if best != float("-inf") else val_metrics["delta_x1"]
            save_checkpoint(
                output_dir / "continuous_q_mix_receiver_latest.pth",
                epoch=epoch,
                receiver=receiver,
                ema=ema,
                optimizer=optimizer,
                args=args,
                init_path=init_path,
                init_args=init_args,
                init_state=init_state,
                mix=mix,
                metrics=val_metrics,
                best=best_for_save,
            )
            if epoch == int(args.epochs):
                save_checkpoint(
                    output_dir / "continuous_q_mix_receiver_final.pth",
                    epoch=epoch,
                    receiver=receiver,
                    ema=ema,
                    optimizer=optimizer,
                    args=args,
                    init_path=init_path,
                    init_args=init_args,
                    init_state=init_state,
                    mix=mix,
                    metrics=val_metrics,
                    best=best_for_save,
                )
            if str(args.checkpoint_selection) == "best-val" and is_new_full_val_best:
                save_checkpoint(
                    output_dir / "continuous_q_mix_receiver_best_val.pth",
                    epoch=epoch,
                    receiver=receiver,
                    ema=ema,
                    optimizer=optimizer,
                    args=args,
                    init_path=init_path,
                    init_args=init_args,
                    init_state=init_state,
                    mix=mix,
                    metrics=val_metrics,
                    best=best,
                )


def run_contract_smoke() -> None:
    """CPU-only proof of fixed q mix, q-only D2, no-leak, and crop contract."""

    class SmokeMember(nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.projection = nn.Conv2d(16, 32, 1)
            with torch.no_grad():
                self.projection.weight.zero_()
                self.projection.bias.fill_(float(value))

        def forward(self, condition):
            condition.validate()
            return self.projection(condition.z1)

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
    mix = FixedTrainSelectedQMix(
        [SmokeMember(1.0), SmokeMember(2.0), SmokeMember(4.0)], [0.6, 0.3, 0.1]
    ).to(device)
    # Simulate a legacy/v12 initialization: load its legacy D2 strictly into
    # the highres wrapper's base, retain its hybrid combiner topology exactly,
    # and prove the newly added four-scale q-only branch starts as zero.
    legacy_d2 = continuous.Layer1InitializedD2(32, 16, SmokeD1()).to(device)
    legacy_combiner = continuous.ReceiverHybridCombiner(
        base_width=8, base_blocks=1, unet_width=8, unet_blocks=1
    ).to(device)
    highres_d2 = continuous.QOnlyHighResolutionResidualD2(
        32, 16, SmokeD1(), width=8, blocks=1
    ).to(device)
    combiner = continuous.ReceiverHybridCombiner(
        base_width=8, base_blocks=1, unet_width=8, unet_blocks=1
    ).to(device)
    receiver = continuous.ContinuousReceiver(mix, highres_d2, combiner).to(device)
    init_state = {
        **{f"d2.{name}": value.detach().clone() for name, value in legacy_d2.state_dict().items()},
        **{
            f"combiner.{name}": value.detach().clone()
            for name, value in legacy_combiner.state_dict().items()
        },
    }
    d2_status, combiner_status = load_post_q_modules_from_initialization(
        receiver,
        init_payload={
            "args": {"d2_type": "layer1"},
            "receiver_state_dict": init_state,
        },
        init_args=SimpleNamespace(d2_type="qonly-highres-residual"),
        init_state="raw",
    )
    if d2_status != "legacy_d2_base_strict+new_qonly_branch_zero" or combiner_status != "combiner_strict":
        raise AssertionError("legacy/v12 post-q initialization did not report strict load")
    continuous.assert_continuous_receiver_contract(receiver)
    probe_q = torch.randn(2, 32, 16, 16, device=device)
    torch.testing.assert_close(receiver.d2(probe_q), legacy_d2(probe_q), rtol=0.0, atol=0.0)
    condition = make_receiver_condition(
        torch.randn(2, 16, 16, 16, device=device),
        torch.rand(2, 3, 256, 256, device=device),
        detach=True,
    )
    warmup_args = SimpleNamespace(
        d2_type="qonly-highres-residual",
        qonly_highres_warmup_epochs=2,
        freeze_d2=False,
        freeze_combiner=False,
        d2_lr=2e-5,
        d2_highres_lr=3e-4,
        combiner_lr=1e-4,
    )
    warmup_active = configure_trainable_parameters(receiver, warmup_args, epoch=1)
    if not warmup_active:
        raise AssertionError("qonly highres warmup did not activate")
    if any(parameter.requires_grad for parameter in receiver.d2.base.parameters()):
        raise AssertionError("qonly highres warmup left legacy D2 base trainable")
    if any(parameter.requires_grad for parameter in receiver.combiner.parameters()):
        raise AssertionError("qonly highres warmup left initialized combiner trainable")
    for module in (receiver.d2.stem, receiver.d2.stages, receiver.d2.residual_head):
        if not all(parameter.requires_grad for parameter in module.parameters()):
            raise AssertionError("qonly highres warmup froze part of the new q2_hat-only branch")
    optimizer_groups = post_q_optimizer_parameter_groups(receiver, warmup_args)
    groups_by_name = {str(group["name"]): group for group in optimizer_groups}
    if set(groups_by_name) != {"d2_base", "d2_highres", "combiner"}:
        raise AssertionError("qonly highres optimizer groups are incomplete")
    if float(groups_by_name["d2_base"]["lr"]) != 2e-5:
        raise AssertionError("legacy D2 base did not retain --d2-lr")
    if float(groups_by_name["d2_highres"]["lr"]) != 3e-4:
        raise AssertionError("new qonly highres branch did not receive --d2-highres-lr")
    highres_ids = {
        id(parameter)
        for module in (receiver.d2.stem, receiver.d2.stages, receiver.d2.residual_head)
        for parameter in module.parameters()
    }
    if {id(parameter) for parameter in groups_by_name["d2_highres"]["params"]} != highres_ids:
        raise AssertionError("qonly highres optimizer group contains a non-branch parameter")
    all_group_ids = [
        id(parameter) for group in optimizer_groups for parameter in group["params"]
    ]
    if len(all_group_ids) != len(set(all_group_ids)):
        raise AssertionError("qonly highres optimizer groups overlap")
    fallback_args = SimpleNamespace(**vars(warmup_args))
    fallback_args.d2_highres_lr = 0.0
    fallback_groups = {
        str(group["name"]): group
        for group in post_q_optimizer_parameter_groups(receiver, fallback_args)
    }
    if float(fallback_groups["d2_highres"]["lr"]) != float(fallback_args.d2_lr):
        raise AssertionError("zero --d2-highres-lr did not fall back to --d2-lr")
    receiver.train()
    receiver.d2.base.eval()
    receiver.combiner.eval()
    if receiver.generator.training or any(member.training for member in receiver.generator.members):
        raise AssertionError("fixed q generator must remain in eval mode during D2/combiner training")
    output = receiver(condition)
    expected_q_value = 0.6 * 1.0 + 0.3 * 2.0 + 0.1 * 4.0
    torch.testing.assert_close(
        output["q2_hat"],
        torch.full_like(output["q2_hat"], expected_q_value),
        rtol=0.0,
        atol=1e-6,
    )
    legacy_u2 = legacy_d2(output["q2_hat"]).clamp(0.0, 1.0)
    torch.testing.assert_close(output["u2_hat"], legacy_u2, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        output["x2_hat"], legacy_combiner(condition.x1, legacy_u2), rtol=0.0, atol=0.0
    )
    target = torch.rand_like(output["x2_hat"])
    assert_training_targets_are_not_inputs(receiver, condition, source_targets={"img": target})
    F.mse_loss(output["x2_hat"], target).backward()
    if any(parameter.grad is not None for parameter in receiver.generator.parameters()):
        raise AssertionError("fixed q generators received a gradient")
    if any(parameter.grad is not None for parameter in receiver.d2.base.parameters()):
        raise AssertionError("warmup backpropagated into the frozen legacy D2 base")
    if any(parameter.grad is not None for parameter in receiver.combiner.parameters()):
        raise AssertionError("warmup backpropagated into the frozen initialized combiner")
    if receiver.d2.residual_head.weight.grad is None:
        raise AssertionError("new q2_hat-only highres residual head received no warmup gradient")
    receiver.zero_grad(set_to_none=True)
    if configure_trainable_parameters(receiver, warmup_args, epoch=3):
        raise AssertionError("qonly highres warmup failed to finish")
    if not all(parameter.requires_grad for parameter in receiver.d2.parameters()):
        raise AssertionError("post-warmup did not unfreeze the complete D2")
    if not all(parameter.requires_grad for parameter in receiver.combiner.parameters()):
        raise AssertionError("post-warmup did not unfreeze the initialized combiner")
    F.mse_loss(receiver(condition)["x2_hat"], target).backward()
    if not any(parameter.grad is not None for parameter in receiver.d2.base.parameters()):
        raise AssertionError("post-warmup base D2 received no gradient")
    if not any(parameter.grad is not None for parameter in receiver.combiner.parameters()):
        raise AssertionError("post-warmup combiner received no gradient")
    ema = copy.deepcopy(receiver).eval()
    ema.generator = receiver.generator
    ema.requires_grad_(False)
    q_before_ema = receiver.generator(condition).detach().clone()
    update_post_qmix_ema(ema, receiver, 0.5)
    torch.testing.assert_close(
        receiver.generator(condition), q_before_ema, rtol=0.0, atol=0.0
    )
    train_loader = SimpleNamespace(
        dataset=SimpleNamespace(transform="RandomCrop(256)+RandomHorizontalFlip+ToTensor")
    )
    val_loader = SimpleNamespace(dataset=SimpleNamespace(transform="CenterCrop(256)+ToTensor"))
    assert_div2k_crop_protocol(train_loader, val_loader)
    print(
        "[PASS] fixed train-selected q-mix -> one D2+combiner CPU smoke "
        f"q={tuple(output['q2_hat'].shape)} u2={tuple(output['u2_hat'].shape)} "
        f"x2={tuple(output['x2_hat'].shape)} fixed_q=1 qonly_d2=1 "
        "legacy_v12_base_strict=1 zero_start_equivalent=1 warmup_branch_only=1 "
        "highres_lr_group=1 highres_lr_fallback=1 combiner_x1_u2=1 "
        "no_leak=1 no_member_grad=1 ema_fixed_q=1 crop_contract=1",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--ensemble-spec",
        default="",
        help="JSON from eval_continuous_q_ensemble.py --selection-mode train-simplex",
    )
    parser.add_argument(
        "--receiver-init-checkpoint",
        default="",
        help="continuous-q checkpoint supplying the single D2+combiner initialization; default=spec reference",
    )
    parser.add_argument(
        "--receiver-init-state",
        choices=["auto", "ema", "raw"],
        default="auto",
        help="auto follows the fixed q-mix spec ensemble_state",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help=(
            "strictly resume the exact same fixed-q receiver run: restores only D2/combiner, "
            "EMA, optimizer, epoch and best metric after validating mix/init/crop/no-leak contracts"
        ),
    )
    parser.add_argument("--source-checkpoint", default="", help="optional Layer1 source override")
    parser.add_argument("--data-dir", default="", help="optional DIV2K override")
    parser.add_argument(
        "--d2-type",
        choices=["layer1", "qonly-highres-residual"],
        default="layer1",
        help=(
            "receiver D2 topology; qonly-highres-residual strictly loads a legacy/v12 "
            "D2 into its base and adds a zero-initialized q2_hat-only four-scale RGB branch"
        ),
    )
    parser.add_argument("--d2-highres-width", type=int, default=64)
    parser.add_argument("--d2-highres-blocks", type=int, default=2)
    parser.add_argument(
        "--qonly-highres-warmup-epochs",
        type=int,
        default=0,
        help=(
            "only with --d2-type qonly-highres-residual: first train only its new "
            "q2_hat-only branch; base D2, combiner, and fixed q generators stay frozen/eval"
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--lambda-final", type=float, default=1.0)
    parser.add_argument("--lambda-u2", type=float, default=0.25)
    parser.add_argument("--final-loss", choices=["mse", "log-mse"], default="log-mse")
    parser.add_argument("--log-mse-scale", type=float, default=0.01)
    parser.add_argument("--hard-example-power", type=float, default=0.5)
    parser.add_argument("--hard-example-min-weight", type=float, default=0.25)
    parser.add_argument("--hard-example-max-weight", type=float, default=4.0)
    parser.add_argument("--d2-lr", type=float, default=2e-5)
    parser.add_argument(
        "--d2-highres-lr",
        type=float,
        default=0.0,
        help=(
            "qonly-highres-residual new stem/stages/RGB-head LR; "
            "0 uses --d2-lr and preserves the legacy D2 LR"
        ),
    )
    parser.add_argument("--combiner-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--freeze-d2", action="store_true")
    parser.add_argument("--freeze-combiner", action="store_true")
    parser.add_argument("--min-delta", type=float, default=0.5)
    parser.add_argument("--min-condition-drop", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["final", "best-val"],
        default="final",
        help="final avoids validation-driven checkpoint selection; best-val is explicit opt-in",
    )
    parser.add_argument(
        "--save-dir", default="MY-V2/jscc-f/explore-2/checkpoints-continuous-q-mix"
    )
    parser.add_argument("--version", default="cnn-continuous-q-mix-top3-v1")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--smoke-contract",
        action="store_true",
        help="CPU-only fixed-mix/q-only-D2/no-leak/crop test; does not load data/checkpoints.",
    )
    args = parser.parse_args()
    if bool(args.smoke_contract):
        return args
    if not str(args.ensemble_spec):
        parser.error("--ensemble-spec is required unless --smoke-contract")
    if int(args.epochs) < 1 or int(args.val_every) < 1:
        parser.error("--epochs and --val-every must be positive")
    if int(args.batch_size) < 1 or int(args.test_batch) < 1:
        parser.error("--batch-size and --test-batch must be positive")
    if int(args.max_train_batches) < 0 or int(args.max_val_batches) < 0:
        parser.error("--max-train-batches and --max-val-batches must be non-negative")
    if int(args.d2_highres_width) < 8 or int(args.d2_highres_blocks) < 1:
        parser.error("--d2-highres-width must be >=8 and --d2-highres-blocks must be positive")
    if int(args.qonly_highres_warmup_epochs) < 0:
        parser.error("--qonly-highres-warmup-epochs must be non-negative")
    if str(args.d2_type) != "qonly-highres-residual" and int(args.qonly_highres_warmup_epochs) != 0:
        parser.error("--qonly-highres-warmup-epochs requires --d2-type qonly-highres-residual")
    if str(args.d2_type) == "qonly-highres-residual" and bool(args.freeze_d2):
        parser.error("--d2-type qonly-highres-residual cannot be combined with --freeze-d2")
    if float(args.d2_lr) <= 0.0 or float(args.combiner_lr) <= 0.0:
        parser.error("--d2-lr and --combiner-lr must be positive")
    if float(args.d2_highres_lr) < 0.0:
        parser.error("--d2-highres-lr must be zero or positive")
    if float(args.d2_highres_lr) > 0.0 and str(args.d2_type) != "qonly-highres-residual":
        parser.error("--d2-highres-lr requires --d2-type qonly-highres-residual")
    if float(args.lambda_final) < 0.0 or float(args.lambda_u2) < 0.0:
        parser.error("--lambda-final and --lambda-u2 must be non-negative")
    if float(args.hard_example_power) < 0.0:
        parser.error("--hard-example-power must be non-negative")
    if float(args.hard_example_min_weight) <= 0.0 or float(args.hard_example_max_weight) <= 0.0:
        parser.error("hard-example weights must be positive")
    if float(args.hard_example_min_weight) > float(args.hard_example_max_weight):
        parser.error("--hard-example-min-weight cannot exceed --hard-example-max-weight")
    if bool(args.freeze_d2) and bool(args.freeze_combiner):
        parser.error("cannot freeze both D2 and combiner")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if bool(parsed_args.smoke_contract):
        run_contract_smoke()
    else:
        train(parsed_args)
