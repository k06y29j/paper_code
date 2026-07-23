#!/usr/bin/env python3
"""Evaluate a receiver-only q2_hat generator ensemble.

Every member is a previously trained continuous generator with the same
receiver-visible contract ``(z1, x1) -> q2_hat``.  At deployment this utility
forms a fixed weighted mean of those q2 estimates and performs *one* reference
receiver decode::

    (z1,x1) -> {G_i} -> sum_i(w_i*q2_hat_i) -> D2_ref -> combiner_ref -> x2_hat

For ``--selection-mode train-simplex``, the weights (and optional top-k
member subset) are fitted *only* against random-cropped DIV2K training images.
``train-gated-simplex`` first performs that same fixed train-only selection,
then learns an optional per-image simplex from receiver-visible ``(z1,x1)``.
The final report is then a fresh, single full 100-image CenterCrop validation.
No E2/z2/true q2 is materialized after Layer1, so this remains a legitimate
q2-generator ensemble rather than an image-space bypass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn as nn


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
from receiver_models import ReceiverTrunk  # noqa: E402


def _compatible_args(saved: dict, cli: argparse.Namespace) -> argparse.Namespace:
    values = dict(saved)
    values.setdefault("combiner_type", "residual")
    values.setdefault("base_combiner_width", int(values.get("combiner_width", 48)))
    values.setdefault("base_combiner_blocks", int(values.get("combiner_blocks", 4)))
    values.setdefault("freeze_initialized_backbone", False)
    values.setdefault("init_checkpoint", "")
    values.setdefault("final_loss", "mse")
    values.setdefault("log_mse_scale", 0.01)
    # Checkpoints through v12 predate the optional q-only highres D2 flags.
    # The historical default is exactly the legacy Layer1-initialized D2.
    values.setdefault("d2_type", "layer1")
    values.setdefault("d2_highres_width", 64)
    values.setdefault("d2_highres_blocks", 2)
    values["cpu"] = bool(cli.cpu)
    values["batch_size"] = int(cli.batch_size)
    values["test_batch"] = int(cli.batch_size)
    values["num_workers"] = int(cli.num_workers)
    values["val_num_workers"] = int(cli.num_workers)
    values["max_train_batches"] = 0
    values["max_val_batches"] = 0
    if str(getattr(cli, "data_dir", "")):
        values["data_dir"] = str(cli.data_dir)
    return argparse.Namespace(**values)


def _state(payload: dict, state_kind: str) -> dict[str, torch.Tensor]:
    if str(state_kind) == "raw":
        return payload["receiver_state_dict"]
    return payload.get("ema_state_dict") or payload["receiver_state_dict"]


def _generator_from_payload(source, payload: dict, args: argparse.Namespace, device: torch.device):
    generator = continuous.ContinuousQGenerator(
        int(source.args.latent_ch),
        int(args.embedding_dim),
        hidden=int(args.hidden),
        blocks=int(args.blocks),
        attention_every=int(args.attention_every),
        heads=int(args.heads),
    ).to(device)
    state = _state(payload, str(args.ensemble_state))
    prefix = "generator."
    generator.load_state_dict(
        {name[len(prefix) :]: value for name, value in state.items() if name.startswith(prefix)},
        strict=True,
    )
    assert_receiver_only_module(generator)
    return generator.eval()


def _normalized_weights(
    weights: torch.Tensor | Sequence[float],
    *,
    members: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate a deployment-fixed convex q2 mixing vector."""

    value = torch.as_tensor(weights, dtype=torch.float32, device=device).flatten()
    if int(value.numel()) != int(members):
        raise ValueError(f"ensemble weights need {members} entries, got {int(value.numel())}")
    if not torch.isfinite(value).all() or bool((value < 0.0).any()):
        raise ValueError("ensemble weights must be finite and non-negative")
    total = value.sum()
    if not bool(total > 0.0):
        raise ValueError("ensemble weights must have positive total mass")
    return value / total


def _jsonable(value):
    """Convert calibration tensors into an auditable JSON artifact."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(name): _jsonable(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _weighted_q(q_members: Sequence[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Mix q2 estimates with global or receiver-condition-dependent weights.

    A one-dimensional vector is the original fixed deployment simplex.  A
    two-dimensional ``[B,M]`` tensor is produced by the optional gate below;
    each row remains a convex combination selected only from `(z1,x1)`.
    """

    if not q_members:
        raise ValueError("q2 ensemble needs at least one generator")
    stacked = torch.stack(list(q_members), dim=0)
    member_count = len(q_members)
    if weights.ndim == 1:
        if int(weights.numel()) != member_count:
            raise ValueError("q2 ensemble member/weight count mismatch")
        view = weights.to(device=stacked.device, dtype=stacked.dtype).view(
            member_count, *([1] * (stacked.ndim - 1))
        )
    elif weights.ndim == 2:
        if tuple(weights.shape) != (int(stacked.shape[1]), member_count):
            raise ValueError(
                "conditional q2 ensemble weights must be [B,M], got "
                f"{tuple(weights.shape)} for q stack {tuple(stacked.shape)}"
            )
        view = weights.to(device=stacked.device, dtype=stacked.dtype).transpose(0, 1).view(
            member_count, int(stacked.shape[1]), *([1] * (stacked.ndim - 2))
        )
    else:
        raise ValueError(f"q2 ensemble weights must be [M] or [B,M], got {tuple(weights.shape)}")
    return (stacked * view).sum(dim=0)


class ReceiverConditionalSimplexGate(nn.Module):
    """Per-image simplex gate that sees only the typed receiver condition.

    The zero-initialized output projection makes its first prediction exactly
    the train-selected global simplex given through ``prior_weights``.  This
    is intentionally a small refinement over the already validated global
    top-k ensemble rather than an image-space bypass.
    """

    def __init__(
        self,
        z1_channels: int,
        member_count: int,
        *,
        prior_weights: torch.Tensor,
        hidden: int,
        blocks: int,
        attention_every: int,
        heads: int,
    ) -> None:
        super().__init__()
        if int(member_count) < 1:
            raise ValueError(f"conditional simplex needs at least one member, got {member_count}")
        if int(hidden) % int(heads) != 0:
            raise ValueError(f"gate hidden={hidden} must be divisible by heads={heads}")
        prior = _normalized_weights(
            prior_weights, members=int(member_count), device=torch.device("cpu")
        )
        self.member_count = int(member_count)
        self.trunk = ReceiverTrunk(
            int(z1_channels),
            hidden=int(hidden),
            blocks=int(blocks),
            attention_every=int(attention_every),
            heads=int(heads),
            condition_mode="z1_x1",
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(int(hidden), self.member_count)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(prior.clamp_min(1e-8).log())

    def forward(self, condition):
        condition.validate()
        feature = self.pool(self.trunk(condition)).flatten(1)
        return torch.softmax(self.head(feature), dim=1)


def _decode_reference(reference, condition, q2_hat: torch.Tensor) -> torch.Tensor:
    """The only allowed ensemble decode: one fixed receiver D2/combiner."""

    u2_hat = reference.d2(q2_hat).clamp(0.0, 1.0)
    return reference.combiner(condition.x1, u2_hat)


def _member_qs(generators, condition) -> list[torch.Tensor]:
    """Generate q2 only from receiver-visible condition tensors."""

    with torch.no_grad():
        return [generator(condition) for generator in generators]


def calibrate_train_simplex(
    loader,
    *,
    source,
    reference,
    generators,
    active_indices: Sequence[int],
    device: torch.device,
    epochs: int,
    max_batches: int,
    lr: float,
    loss_name: str,
) -> dict:
    """Fit one convex q2 mixing vector strictly on random-cropped train data.

    ``img`` is used only as the supervised loss target.  Each generator still
    sees exactly ``ReceiverCondition(z1, x1)`` and gradients flow only into
    the scalar simplex logits through the fixed reference D2/combiner.
    """

    active = [int(index) for index in active_indices]
    if not active:
        raise ValueError("train simplex selection requires at least one active member")
    if int(epochs) < 1:
        raise ValueError("--calibration-epochs must be positive")
    if int(max_batches) < 0:
        raise ValueError("--calibration-batches must be non-negative")
    if float(lr) <= 0.0:
        raise ValueError("--calibration-lr must be positive")
    if str(loss_name) not in {"mse", "log-mse"}:
        raise ValueError(f"unknown calibration loss {loss_name!r}")

    for generator in generators:
        generator.eval().requires_grad_(False)
    reference.eval().requires_grad_(False)
    source.e1.eval()
    source.d1.eval()
    logits = torch.nn.Parameter(torch.zeros(len(active), device=device))
    optimizer = torch.optim.Adam([logits], lr=float(lr))
    epoch_metrics: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        loss_sum = 0.0
        psnr_sum = 0.0
        examples = 0
        batches = 0
        for batch_index, (image, _label) in enumerate(loader, start=1):
            if int(max_batches) > 0 and batch_index > int(max_batches):
                break
            image = image.to(device, non_blocking=True)
            with torch.no_grad():
                layer1 = source.layer1(image)
                condition = make_receiver_condition(
                    layer1["z1"], layer1["x1"], detach=True
                )
                all_q_members = _member_qs(generators, condition)
            q_members = [all_q_members[index] for index in active]
            weights = torch.softmax(logits, dim=0)
            q2_hat = _weighted_q(q_members, weights)
            prediction = _decode_reference(reference, condition, q2_hat)
            mse_per_image = (prediction.float() - image.float()).square().flatten(1).mean(dim=1)
            if str(loss_name) == "log-mse":
                loss = torch.log(mse_per_image.clamp_min(1e-8)).mean()
            else:
                loss = mse_per_image.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch = int(image.shape[0])
            loss_sum += float(loss.detach()) * batch
            psnr_sum += float(continuous.psnr_per_image(prediction, image).sum().detach())
            examples += batch
            batches += 1
        if examples < 1:
            raise RuntimeError("train simplex calibration processed no training images")
        epoch_metrics.append(
            {
                "epoch": float(epoch),
                "batches": float(batches),
                "images": float(examples),
                "loss": loss_sum / examples,
                "psnr": psnr_sum / examples,
            }
        )

    return {
        "active_indices": active,
        "weights": torch.softmax(logits.detach(), dim=0),
        "epochs": epoch_metrics,
    }


def select_train_weights(
    loader,
    *,
    source,
    reference,
    generators,
    device: torch.device,
    max_members: int,
    epochs: int,
    max_batches: int,
    lr: float,
    loss_name: str,
) -> dict:
    """Train-only simplex calibration with optional top-k subset refit."""

    count = len(generators)
    if int(max_members) < 0 or int(max_members) > count:
        raise ValueError(f"--max-members must lie in [0,{count}], got {max_members}")
    first = calibrate_train_simplex(
        loader,
        source=source,
        reference=reference,
        generators=generators,
        active_indices=list(range(count)),
        device=device,
        epochs=epochs,
        max_batches=max_batches,
        lr=lr,
        loss_name=loss_name,
    )
    selected = list(range(count))
    refit = None
    if 0 < int(max_members) < count:
        ranking = torch.argsort(first["weights"], descending=True).tolist()
        selected = sorted(int(index) for index in ranking[: int(max_members)])
        refit = calibrate_train_simplex(
            loader,
            source=source,
            reference=reference,
            generators=generators,
            active_indices=selected,
            device=device,
            epochs=epochs,
            max_batches=max_batches,
            lr=lr,
            loss_name=loss_name,
        )
        selected_weights = refit["weights"]
    else:
        selected_weights = first["weights"]
    weights = torch.zeros(count, dtype=torch.float32, device=device)
    weights[torch.tensor(selected, device=device)] = selected_weights
    return {
        "weights": _normalized_weights(weights, members=count, device=device),
        "selected_indices": selected,
        "first_fit": first,
        "subset_refit": refit,
    }


def calibrate_train_conditional_gate(
    loader,
    *,
    source,
    reference,
    generators,
    selected_indices: Sequence[int],
    global_weights: torch.Tensor,
    device: torch.device,
    hidden: int,
    blocks: int,
    attention_every: int,
    heads: int,
    epochs: int,
    max_batches: int,
    lr: float,
    loss_name: str,
    anchor_weight: float,
) -> tuple[ReceiverConditionalSimplexGate, dict]:
    """Fit a receiver-only conditional q2 simplex on RandomCrop train data.

    The global member/top-k choice is already train-selected.  This optional
    second stage only lets `(z1,x1)` redistribute that fixed subset per image.
    The target image appears after `gate(condition)` and q2 decoding; sender
    Layer2 tensors are never built.
    """

    active = [int(index) for index in selected_indices]
    if not active:
        raise ValueError("conditional simplex requires at least one selected generator")
    if int(epochs) < 1 or int(max_batches) < 0 or float(lr) <= 0.0:
        raise ValueError("invalid conditional-gate calibration epochs/batches/lr")
    if float(anchor_weight) < 0.0:
        raise ValueError("--gate-anchor-weight must be non-negative")
    prior = _normalized_weights(
        global_weights[torch.as_tensor(active, device=global_weights.device)],
        members=len(active),
        device=device,
    )
    gate = ReceiverConditionalSimplexGate(
        int(source.args.latent_ch),
        len(active),
        prior_weights=prior,
        hidden=int(hidden),
        blocks=int(blocks),
        attention_every=int(attention_every),
        heads=int(heads),
    ).to(device)
    assert_receiver_only_module(gate)
    for generator in generators:
        generator.eval().requires_grad_(False)
    reference.eval().requires_grad_(False)
    source.e1.eval()
    source.d1.eval()
    optimizer = torch.optim.AdamW(gate.parameters(), lr=float(lr), weight_decay=0.0)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        gate.train()
        loss_sum = 0.0
        reconstruction_sum = 0.0
        anchor_sum = 0.0
        entropy_sum = 0.0
        psnr_sum = 0.0
        examples = 0
        batches = 0
        audited = False
        for batch_index, (image, _label) in enumerate(loader, start=1):
            if int(max_batches) > 0 and batch_index > int(max_batches):
                break
            image = image.to(device, non_blocking=True)
            with torch.no_grad():
                layer1 = source.layer1(image)
                condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
                all_q_members = _member_qs(generators, condition)
            q_members = [all_q_members[index] for index in active]
            # This is the only conditional decision point and it receives the
            # typed receiver condition, never `image` or a Layer2 tensor.
            per_image_weights = gate(condition)
            q2_hat = _weighted_q(q_members, per_image_weights)
            prediction = _decode_reference(reference, condition, q2_hat)
            if not audited:
                assert_training_targets_are_not_inputs(
                    gate,
                    condition,
                    source_targets={"img": image},
                )
                audited = True
            mse_per_image = (prediction.float() - image.float()).square().flatten(1).mean(dim=1)
            if str(loss_name) == "log-mse":
                reconstruction = torch.log(mse_per_image.clamp_min(1e-8)).mean()
            elif str(loss_name) == "mse":
                reconstruction = mse_per_image.mean()
            else:
                raise ValueError(f"unknown gate calibration loss {loss_name!r}")
            anchor = (per_image_weights - prior.unsqueeze(0)).square().mean()
            loss = reconstruction + float(anchor_weight) * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimizer.step()

            batch = int(image.shape[0])
            loss_sum += float(loss.detach()) * batch
            reconstruction_sum += float(reconstruction.detach()) * batch
            anchor_sum += float(anchor.detach()) * batch
            entropy_sum += float(
                (-(per_image_weights * per_image_weights.clamp_min(1e-8).log()).sum(dim=1)).detach().sum()
            )
            psnr_sum += float(continuous.psnr_per_image(prediction, image).sum().detach())
            examples += batch
            batches += 1
        if examples < 1:
            raise RuntimeError("conditional simplex calibration processed no training images")
        history.append(
            {
                "epoch": float(epoch),
                "batches": float(batches),
                "images": float(examples),
                "loss": loss_sum / examples,
                "reconstruction_loss": reconstruction_sum / examples,
                "anchor_loss": anchor_sum / examples,
                "weight_entropy": entropy_sum / examples,
                "psnr": psnr_sum / examples,
            }
        )
    gate.eval()
    return gate, {
        "active_indices": active,
        "prior_weights": [float(value) for value in prior.detach().cpu()],
        "epochs": history,
        "anchor_weight": float(anchor_weight),
        "loss": str(loss_name),
    }


@torch.no_grad()
def evaluate(
    loader,
    *,
    source,
    reference,
    generators,
    weights,
    device: torch.device,
    gate: ReceiverConditionalSimplexGate | None = None,
    selected_indices: Sequence[int] | None = None,
) -> dict:
    totals = {
        "psnr_x1": 0.0,
        "psnr_ensemble": 0.0,
        "psnr_condition_shuffle": 0.0,
        "psnr_zero": 0.0,
        "psnr_q_shuffle": 0.0,
    }
    member_sums = [0.0 for _ in generators]
    images = 0
    reference.eval()
    weights = _normalized_weights(weights, members=len(generators), device=device)
    if gate is None:
        active = list(range(len(generators)))
        gate_weight_sum = None
        gate_entropy_sum = None
    else:
        active = [int(index) for index in (selected_indices or [])]
        if not active or len(active) != int(gate.member_count):
            raise ValueError("conditional gate selected_indices do not match its member count")
        if any(index < 0 or index >= len(generators) for index in active):
            raise ValueError("conditional gate selected index is out of range")
        gate.eval()
        assert_receiver_only_module(gate)
        gate_weight_sum = torch.zeros(len(active), dtype=torch.float64)
        gate_entropy_sum = 0.0
    for image, _label in loader:
        image = image.to(device, non_blocking=True)
        layer1 = source.layer1(image)
        condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
        q_members = _member_qs(generators, condition)
        if gate is None:
            q_ensemble = _weighted_q(q_members, weights)
        else:
            per_image_weights = gate(condition)
            q_ensemble = _weighted_q([q_members[index] for index in active], per_image_weights)
        prediction = _decode_reference(reference, condition, q_ensemble)
        batch = int(image.shape[0])
        totals["psnr_x1"] += float(continuous.psnr_per_image(condition.x1, image).sum())
        totals["psnr_ensemble"] += float(continuous.psnr_per_image(prediction, image).sum())
        for index, q_member in enumerate(q_members):
            member_prediction = _decode_reference(reference, condition, q_member)
            member_sums[index] += float(continuous.psnr_per_image(member_prediction, image).sum())
        if batch > 1:
            permutation = torch.roll(torch.arange(batch, device=device), shifts=1)
            wrong = make_receiver_condition(condition.z1[permutation], condition.x1[permutation], detach=True)
            wrong_members = _member_qs(generators, wrong)
            if gate is None:
                wrong_q = _weighted_q(wrong_members, weights)
            else:
                wrong_q = _weighted_q(
                    [wrong_members[index] for index in active], gate(wrong)
                )
            wrong_prediction = _decode_reference(reference, condition, wrong_q)
            totals["psnr_condition_shuffle"] += float(
                continuous.psnr_per_image(wrong_prediction, image).sum()
            )
        else:
            totals["psnr_condition_shuffle"] += float(continuous.psnr_per_image(prediction, image).sum())
        zero = _decode_reference(reference, condition, torch.zeros_like(q_ensemble))
        q_shuffle = q_ensemble[torch.roll(torch.arange(batch, device=device), shifts=1)]
        shuffled = _decode_reference(reference, condition, q_shuffle)
        totals["psnr_zero"] += float(continuous.psnr_per_image(zero, image).sum())
        totals["psnr_q_shuffle"] += float(continuous.psnr_per_image(shuffled, image).sum())
        if gate is not None:
            gate_weight_sum += per_image_weights.detach().double().sum(dim=0).cpu()
            gate_entropy_sum += float(
                (-(per_image_weights * per_image_weights.clamp_min(1e-8).log()).sum(dim=1)).sum()
            )
        images += batch
    if images != len(loader.dataset):
        raise AssertionError(f"ensemble evaluation covered {images}, expected {len(loader.dataset)}")
    result = {name: value / images for name, value in totals.items()}
    result["psnr_members_reference_decode"] = [value / images for value in member_sums]
    result["delta_x1"] = result["psnr_ensemble"] - result["psnr_x1"]
    result["condition_shuffle_drop"] = result["psnr_ensemble"] - result["psnr_condition_shuffle"]
    result["pred_drop_zero"] = result["psnr_ensemble"] - result["psnr_zero"]
    result["pred_drop_q_shuffle"] = result["psnr_ensemble"] - result["psnr_q_shuffle"]
    result["evaluated_images"] = images
    result["full_validation"] = 1.0
    result["receiver_only_audit"] = 1.0
    result["weights"] = [float(value) for value in weights.detach().cpu()]
    if gate is not None:
        assert gate_weight_sum is not None and gate_entropy_sum is not None
        result["conditional_gate"] = 1.0
        result["gate_selected_indices"] = active
        result["gate_mean_weights"] = [float(value) for value in (gate_weight_sum / images)]
        result["gate_mean_entropy"] = float(gate_entropy_sum / images)
    return result


def run_contract_smoke() -> None:
    """CPU-only conditional fusion contract check; no data/checkpoint access."""

    torch.manual_seed(20260713)
    device = torch.device("cpu")
    prior = torch.tensor([0.6, 0.3, 0.1], device=device)
    gate = ReceiverConditionalSimplexGate(
        16,
        3,
        prior_weights=prior,
        hidden=16,
        blocks=1,
        attention_every=0,
        heads=4,
    ).to(device)
    assert_receiver_only_module(gate)
    condition = make_receiver_condition(
        torch.randn(2, 16, 16, 16, device=device),
        torch.rand(2, 3, 256, 256, device=device),
        detach=True,
    )
    weights = gate(condition)
    expected = _normalized_weights(prior, members=3, device=device).expand(2, -1)
    torch.testing.assert_close(weights, expected, rtol=0.0, atol=1e-7)
    q_members = [torch.randn(2, 32, 16, 16, device=device) for _ in range(3)]
    q2_hat = _weighted_q(q_members, weights)
    if tuple(q2_hat.shape) != (2, 32, 16, 16):
        raise AssertionError(f"conditional q mix shape mismatch: {tuple(q2_hat.shape)}")
    image_target = torch.rand(2, 3, 256, 256, device=device)
    assert_training_targets_are_not_inputs(
        gate,
        condition,
        source_targets={"img": image_target},
    )
    train_loader = SimpleNamespace(
        dataset=SimpleNamespace(transform="RandomCrop(256)+RandomHorizontalFlip+ToTensor")
    )
    val_loader = SimpleNamespace(dataset=SimpleNamespace(transform="CenterCrop(256)+ToTensor"))
    assert_div2k_crop_protocol(train_loader, val_loader)
    print(
        "[PASS] conditional q-simplex CPU smoke "
        f"weights={weights[0].detach().tolist()} q={tuple(q2_hat.shape)} "
        "gate_z1_x1_only=1 no_leak=1 crop_contract=1",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reference-checkpoint", default="")
    parser.add_argument(
        "--generator-checkpoints",
        default="",
        help="comma-separated continuous-q receiver checkpoints; reference may be included",
    )
    parser.add_argument("--ensemble-state", choices=["ema", "raw"], default="ema")
    parser.add_argument(
        "--selection-mode",
        choices=["uniform", "train-simplex", "train-gated-simplex"],
        default="uniform",
        help=(
            "uniform preserves arithmetic q averaging; train-simplex fits global weights only "
            "on RandomCrop train; train-gated-simplex additionally learns per-image q weights "
            "from receiver-visible z1/x1 before the final CenterCrop validation"
        ),
    )
    parser.add_argument(
        "--max-members",
        type=int,
        default=0,
        help=(
            "with train-simplex or train-gated-simplex, keep top-k train-selected "
            "generators then refit their global weights; 0 keeps all"
        ),
    )
    parser.add_argument("--calibration-epochs", type=int, default=3)
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=0,
        help="training batches per calibration epoch; 0 means the complete train loader",
    )
    parser.add_argument("--calibration-lr", type=float, default=5e-2)
    parser.add_argument("--calibration-loss", choices=["mse", "log-mse"], default="mse")
    parser.add_argument("--gate-hidden", type=int, default=64)
    parser.add_argument("--gate-blocks", type=int, default=2)
    parser.add_argument("--gate-attention-every", type=int, default=0)
    parser.add_argument("--gate-heads", type=int, default=4)
    parser.add_argument("--gate-epochs", type=int, default=3)
    parser.add_argument(
        "--gate-batches",
        type=int,
        default=0,
        help="RandomCrop train batches per gate epoch; 0 means the complete train loader",
    )
    parser.add_argument("--gate-lr", type=float, default=1e-4)
    parser.add_argument(
        "--gate-anchor-weight",
        type=float,
        default=0.05,
        help="Train-only L2 penalty toward the already train-selected global simplex.",
    )
    parser.add_argument(
        "--gate-checkpoint",
        default="",
        help="Optional path for a replayable conditional-gate state; default derives from --output.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", default="", help="optional DIV2K override")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--smoke-contract",
        action="store_true",
        help="CPU-only conditional-gate/no-leak/crop smoke; does not load checkpoints or data.",
    )
    cli = parser.parse_args()
    if bool(cli.smoke_contract):
        run_contract_smoke()
        return
    if not str(cli.reference_checkpoint) or not str(cli.generator_checkpoints):
        parser.error("--reference-checkpoint and --generator-checkpoints are required unless --smoke-contract")
    if str(cli.selection_mode) == "train-gated-simplex":
        if int(cli.gate_hidden) < 8 or int(cli.gate_hidden) % int(cli.gate_heads) != 0:
            parser.error("--gate-hidden must be >=8 and divisible by --gate-heads")
        if (
            int(cli.gate_blocks) < 1
            or int(cli.gate_attention_every) < 0
            or int(cli.gate_epochs) < 1
            or int(cli.gate_batches) < 0
        ):
            parser.error(
                "--gate-blocks/--gate-epochs must be positive and "
                "--gate-attention-every/--gate-batches non-negative"
            )
        if float(cli.gate_lr) <= 0.0 or float(cli.gate_anchor_weight) < 0.0:
            parser.error("--gate-lr must be positive and --gate-anchor-weight non-negative")
    elif str(cli.gate_checkpoint):
        parser.error("--gate-checkpoint requires --selection-mode train-gated-simplex")
    continuous.nested.seed_everything(int(cli.seed))

    reference_path = continuous.resolve_path(cli.reference_checkpoint)
    reference_payload = torch.load(reference_path, map_location="cpu", weights_only=False)
    if str(reference_payload.get("stage", "")) != "explore2_continuous_q_receiver":
        raise ValueError(f"not a continuous-q receiver checkpoint: {reference_path}")
    reference_args = _compatible_args(reference_payload["args"], cli)
    reference_args.ensemble_state = str(cli.ensemble_state)
    source, train_loader, val_loader, device = continuous.build_source_and_loaders(reference_args)
    assert_div2k_crop_protocol(train_loader, val_loader)
    if len(val_loader.dataset) != 100:
        raise AssertionError(
            f"final ensemble report requires exactly 100 DIV2K validation images, "
            f"got {len(val_loader.dataset)}"
        )
    reference = continuous.build_receiver(source, reference_args, device)
    reference.load_state_dict(_state(reference_payload, str(cli.ensemble_state)), strict=True)
    # Sender Layer2 is not available to this evaluator after the reference D2
    # clone has been built.
    source.e2 = torch.nn.Identity().to(device)
    source.d2 = torch.nn.Identity().to(device)
    source.combiner = torch.nn.Identity().to(device)
    source.checkpoint = {}
    source.layer2_checkpoint = {}

    paths = [continuous.resolve_path(value.strip()) for value in cli.generator_checkpoints.split(",") if value.strip()]
    if not paths:
        raise ValueError("--generator-checkpoints is empty")
    generators = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if str(payload.get("stage", "")) != "explore2_continuous_q_receiver":
            raise ValueError(f"not a continuous-q receiver checkpoint: {path}")
        args = _compatible_args(payload["args"], cli)
        args.ensemble_state = str(cli.ensemble_state)
        if int(args.embedding_dim) != int(reference_args.embedding_dim):
            raise ValueError(f"generator D mismatch: {path}")
        if str(args.arch) != str(reference_args.arch):
            raise ValueError(
                f"generator Layer1 architecture mismatch: {path} has {args.arch!r}, "
                f"reference has {reference_args.arch!r}"
            )
        generators.append(_generator_from_payload(source, payload, args, device))

    gate = None
    gate_details = None
    if str(cli.selection_mode) in {"train-simplex", "train-gated-simplex"}:
        selection = select_train_weights(
            train_loader,
            source=source,
            reference=reference,
            generators=generators,
            device=device,
            max_members=int(cli.max_members),
            epochs=int(cli.calibration_epochs),
            max_batches=int(cli.calibration_batches),
            lr=float(cli.calibration_lr),
            loss_name=str(cli.calibration_loss),
        )
        weights = selection["weights"]
        selected_indices = [int(index) for index in selection["selected_indices"]]
        if str(cli.selection_mode) == "train-gated-simplex":
            gate, gate_details = calibrate_train_conditional_gate(
                train_loader,
                source=source,
                reference=reference,
                generators=generators,
                selected_indices=selected_indices,
                global_weights=weights,
                device=device,
                hidden=int(cli.gate_hidden),
                blocks=int(cli.gate_blocks),
                attention_every=int(cli.gate_attention_every),
                heads=int(cli.gate_heads),
                epochs=int(cli.gate_epochs),
                max_batches=int(cli.gate_batches),
                lr=float(cli.gate_lr),
                loss_name=str(cli.calibration_loss),
                anchor_weight=float(cli.gate_anchor_weight),
            )
    else:
        if int(cli.max_members) != 0:
            raise ValueError(
                "--max-members is only valid with --selection-mode train-simplex "
                "or train-gated-simplex"
            )
        weights = torch.full(
            (len(generators),), 1.0 / float(len(generators)), device=device
        )
        selected_indices = list(range(len(generators)))
        selection = None

    # This is deliberately after train-only selection.  Nothing in
    # ``evaluate`` can alter the fixed weights, so the 100 center crops remain
    # a clean final report rather than a tuning signal.
    metrics = evaluate(
        val_loader,
        source=source,
        reference=reference,
        generators=generators,
        weights=weights,
        device=device,
        gate=gate,
        selected_indices=selected_indices,
    )
    default_name = {
        "uniform": "continuous_q_ensemble.json",
        "train-simplex": "continuous_q_ensemble_train_selected.json",
        "train-gated-simplex": "continuous_q_ensemble_train_gated.json",
    }[str(cli.selection_mode)]
    output = Path(cli.output) if cli.output else HERE / "results-receiver" / default_name
    if not output.is_absolute():
        output = continuous.ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    gate_checkpoint = None
    if gate is not None:
        gate_checkpoint = (
            Path(cli.gate_checkpoint)
            if str(cli.gate_checkpoint)
            else output.with_suffix(".gate.pth")
        )
        if not gate_checkpoint.is_absolute():
            gate_checkpoint = continuous.ROOT / gate_checkpoint
        gate_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stage": "explore2_continuous_q_conditional_simplex_gate",
                "gate_state_dict": gate.state_dict(),
                "gate_config": {
                    "z1_channels": int(source.args.latent_ch),
                    "member_count": int(gate.member_count),
                    "hidden": int(cli.gate_hidden),
                    "blocks": int(cli.gate_blocks),
                    "attention_every": int(cli.gate_attention_every),
                    "heads": int(cli.gate_heads),
                },
                "selected_indices": selected_indices,
                "global_weights": [float(value) for value in weights.detach().cpu()],
                "selected_prior_weights": gate_details["prior_weights"],
                "reference_checkpoint": str(reference_path),
                "generator_checkpoints": [str(path) for path in paths],
                "ensemble_state": str(cli.ensemble_state),
                "receiver_contract": {
                    "deployment_inputs": ["z1", "x1"],
                    "generation": "gate(z1,x1)->per_image_w; sum_i(w_i*G_i(z1,x1))->q2_hat",
                    "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
                    "target_role": "RandomCrop train-only supervised objective",
                },
            },
            gate_checkpoint,
        )
    result = {
        "reference_checkpoint": str(reference_path),
        "generator_checkpoints": [str(path) for path in paths],
        "ensemble_state": str(cli.ensemble_state),
        "weight_selection": {
            "mode": str(cli.selection_mode),
            "weights": [float(value) for value in _normalized_weights(
                weights, members=len(generators), device=device
            ).detach().cpu()],
            "selected_indices": selected_indices,
            "selected_checkpoints": [str(paths[index]) for index in selected_indices],
            "selection_split": "DIV2K train",
            "selection_transform": "RandomCrop(256)+RandomHorizontalFlip+ToTensor",
            "selection_loss_target": "train img only",
            "calibration_epochs": int(cli.calibration_epochs)
            if str(cli.selection_mode) in {"train-simplex", "train-gated-simplex"}
            else 0,
            "calibration_batches_per_epoch": int(cli.calibration_batches)
            if str(cli.selection_mode) in {"train-simplex", "train-gated-simplex"}
            else 0,
            "calibration_loss": str(cli.calibration_loss)
            if str(cli.selection_mode) in {"train-simplex", "train-gated-simplex"}
            else "none",
            "details": _jsonable(selection),
        },
        "conditional_gate": {
            "enabled": bool(gate is not None),
            "checkpoint": str(gate_checkpoint) if gate_checkpoint is not None else "",
            "inputs": ["z1", "x1"],
            "output": "per-image simplex over the train-selected generator subset",
            "details": _jsonable(gate_details),
        },
        "receiver_contract": {
            "deployment_inputs": ["z1", "x1"],
            "generation": (
                "gate(z1,x1)->w; sum_i(w_i*G_i(z1,x1)) -> q2_hat"
                if gate is not None
                else "sum_i(w_i*G_i(z1,x1)) -> q2_hat"
            ),
            "decode_path": "q2_hat -> reference_D2 -> reference_combiner(x1,u2_hat)",
            "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
            "weight_selection_forbidden_inputs": ["z2", "q2", "oracle_indices"],
            "weight_selection_image_role": "train-only supervised objective",
            "validation_transform": "CenterCrop(256)+ToTensor",
        },
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if gate_checkpoint is not None:
        print(f"saved conditional gate: {gate_checkpoint}", flush=True)
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
