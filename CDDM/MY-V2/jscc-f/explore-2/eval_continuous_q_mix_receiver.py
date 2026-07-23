#!/usr/bin/env python3
"""Strict read-only full-validation evaluator for fixed train-selected q mixes.

The saved q-mix receiver checkpoint deliberately contains only the trained
post-generation modules.  This evaluator reconstructs the frozen Layer1 and
the immutable train-selected generators, then evaluates exactly::

    (z1, x1) -> fixed sum_i(w_i * G_i(z1, x1)) -> q2_hat
             -> one saved D2(q2_hat) -> u2_hat
             -> one saved combiner(x1, u2_hat) -> x2_hat

No sender Layer2 E2/z2/true q2/oracle index is present in the receiver graph.
The source image is used only to derive the evaluation Layer1 condition and,
after the receiver forward pass, as the PSNR target.  The evaluator is
read-only with respect to all checkpoints: it writes only one JSON report
under ``explore-2/results-receiver``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_continuous_q_mix_receiver as qmix_train  # noqa: E402
import train_continuous_q_receiver as continuous  # noqa: E402
from contracts import (  # noqa: E402
    assert_div2k_crop_protocol,
    assert_receiver_only_module,
    assert_training_targets_are_not_inputs,
    make_receiver_condition,
)


RESULTS_ROOT = HERE / "results-receiver"
EXPECTED_VALIDATION_IMAGES = 100
FORBIDDEN_RECEIVER_INPUTS = {"img", "z2", "q2", "oracle_indices"}


class SenderLayer2Forbidden(nn.Module):
    """A tripwire proving the evaluator cannot accidentally call sender Layer2."""

    def forward(self, *_args, **_kwargs):  # pragma: no cover - must never execute
        raise AssertionError(
            "sender Layer2 E2/D2/combiner is forbidden in the strict receiver evaluator"
        )


def _canonical_path(value: str | Path) -> str:
    return str(continuous.resolve_path(value).resolve())


def _as_float_list(value: Sequence[float] | torch.Tensor) -> list[float]:
    return [float(item) for item in torch.as_tensor(value, dtype=torch.float64).flatten().tolist()]


def _close_float_lists(left: Sequence[float], right: Sequence[float], *, atol: float = 1e-7) -> bool:
    if len(left) != len(right):
        return False
    return all(abs(float(a) - float(b)) <= float(atol) for a, b in zip(left, right))


def _required(mapping: dict[str, Any], name: str) -> Any:
    if name not in mapping:
        raise ValueError(f"saved q-mix checkpoint is missing required field {name!r}")
    return mapping[name]


def validate_saved_mix(saved_mix: dict[str, Any]) -> dict[str, Any]:
    """Validate saved train-only selection and compare it to its source JSON.

    The checkpoint records the selected generator paths and weights so that a
    later change to a report cannot silently alter deployment.  Re-reading the
    report here proves that the recorded mixture was originally selected on
    RandomCrop training data rather than on the validation set.
    """

    mix = dict(saved_mix)
    provenance = dict(_required(mix, "selection_provenance"))
    if str(provenance.get("mode", "")) != "train-simplex":
        raise ValueError("saved fixed q mix is not a train-simplex selection")
    if str(provenance.get("selection_split", "")) != "DIV2K train":
        raise ValueError("saved fixed q mix was not selected on DIV2K train")
    transform = str(provenance.get("selection_transform", ""))
    if "RandomCrop" not in transform or "CenterCrop" in transform:
        raise ValueError(
            "saved fixed q mix lacks RandomCrop-only selection provenance: " f"{transform!r}"
        )
    if str(provenance.get("selection_loss_target", "")) != "train img only":
        raise ValueError("saved fixed q mix does not prove train-image-only selection")

    selected_indices = [int(item) for item in _required(mix, "selected_indices")]
    member_paths = [str(item) for item in _required(mix, "member_paths")]
    weights = _as_float_list(_required(mix, "weights"))
    if not selected_indices or len(selected_indices) != len(member_paths) or len(weights) != len(member_paths):
        raise ValueError("saved fixed q mix has inconsistent selected members/weights")
    if len(set(selected_indices)) != len(selected_indices) or any(index < 0 for index in selected_indices):
        raise ValueError("saved fixed q mix has invalid original member indices")
    if not all(torch.isfinite(torch.tensor(weights)).tolist()) or any(weight <= 0.0 for weight in weights):
        raise ValueError("saved fixed q mix weights must be finite and strictly positive")
    total = float(sum(weights))
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"saved fixed q mix weights must sum to one, got {total:.9g}")
    ensemble_state = str(_required(mix, "ensemble_state"))
    if ensemble_state not in {"raw", "ema"}:
        raise ValueError(f"unsupported frozen generator state {ensemble_state!r}")
    reference = str(_required(mix, "reference_checkpoint"))
    if not reference:
        raise ValueError("saved fixed q mix has no reference checkpoint")

    spec_path = continuous.resolve_path(_required(mix, "spec_path"))
    if not spec_path.is_file():
        raise FileNotFoundError(f"saved train-selected q-mix source JSON is unavailable: {spec_path}")
    from_spec = qmix_train.load_train_selected_mix(spec_path)
    if [int(item) for item in from_spec["selected_indices"]] != selected_indices:
        raise ValueError("saved q-mix selected indices disagree with the train-simplex JSON")
    if [_canonical_path(item) for item in from_spec["member_paths"]] != [
        _canonical_path(item) for item in member_paths
    ]:
        raise ValueError("saved q-mix member checkpoints disagree with the train-simplex JSON")
    if not _close_float_lists(from_spec["weights"], weights):
        raise ValueError("saved q-mix weights disagree with the train-simplex JSON")
    if str(from_spec["ensemble_state"]) != ensemble_state:
        raise ValueError("saved q-mix frozen generator state disagrees with the train-simplex JSON")
    if _canonical_path(from_spec["reference_checkpoint"]) != _canonical_path(reference):
        raise ValueError("saved q-mix reference checkpoint disagrees with the train-simplex JSON")

    return {
        "spec_path": str(spec_path.resolve()),
        "reference_checkpoint": _canonical_path(reference),
        "ensemble_state": ensemble_state,
        "selected_indices": selected_indices,
        "member_paths": [_canonical_path(item) for item in member_paths],
        "weights": weights,
        "selection_provenance": {
            "mode": str(provenance["mode"]),
            "selection_split": str(provenance["selection_split"]),
            "selection_transform": transform,
            "selection_loss_target": str(provenance["selection_loss_target"]),
            "calibration_epochs": int(provenance.get("calibration_epochs", 0)),
            "calibration_batches_per_epoch": int(
                provenance.get("calibration_batches_per_epoch", 0)
            ),
            "calibration_loss": str(provenance.get("calibration_loss", "")),
        },
    }


def validate_checkpoint(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject any checkpoint that cannot prove the strict receiver wiring."""

    if str(payload.get("stage", "")) != "explore2_continuous_q_mix_receiver":
        raise ValueError("--checkpoint is not an explore-2 fixed q-mix receiver checkpoint")
    contract = dict(_required(payload, "receiver_contract"))
    if list(contract.get("deployment_inputs", [])) != ["z1", "x1"]:
        raise ValueError("saved receiver contract does not restrict deployment inputs to z1/x1")
    if list(contract.get("d2_inputs", [])) != ["q2_hat"]:
        raise ValueError("saved receiver contract does not prove q2_hat-only D2")
    if list(contract.get("combiner_inputs", [])) != ["x1", "u2_hat"]:
        raise ValueError("saved receiver contract does not prove x1/u2_hat combiner")
    forbidden = set(contract.get("forbidden_inputs", []))
    if not FORBIDDEN_RECEIVER_INPUTS.issubset(forbidden):
        raise ValueError("saved receiver contract lacks required forbidden receiver inputs")
    if "RandomCrop" not in str(contract.get("q_mix_selection_transform", "")):
        raise ValueError("saved receiver contract lacks RandomCrop q-mix selection provenance")
    if "CenterCrop" not in str(contract.get("validation_transform", "")):
        raise ValueError("saved receiver contract lacks CenterCrop validation provenance")

    receiver_init = dict(_required(payload, "receiver_init"))
    if not str(_required(receiver_init, "checkpoint")):
        raise ValueError("saved receiver has no initialization checkpoint")
    if str(_required(receiver_init, "state")) not in {"raw", "ema"}:
        raise ValueError("saved receiver initialization has unsupported state")
    init_args = dict(_required(receiver_init, "args"))
    if str(init_args.get("arch", "")) not in {"cnn", "swin"}:
        raise ValueError("saved receiver initialization lacks a supported Layer1 architecture")

    for key in ("decoder_combiner_state_dict", "ema_decoder_combiner_state_dict"):
        post = dict(_required(payload, key))
        if set(post) != {"d2", "combiner"}:
            raise ValueError(f"{key} must contain exactly d2 and combiner states")
        if not isinstance(post["d2"], dict) or not isinstance(post["combiner"], dict):
            raise ValueError(f"{key} d2/combiner states must be dictionaries")

    return validate_saved_mix(dict(_required(payload, "fixed_q_mix"))), receiver_init


def rebuild_init_args(receiver_init: dict[str, Any], cli: argparse.Namespace) -> argparse.Namespace:
    """Restore the historical post-q receiver topology without training options."""

    saved = dict(receiver_init["args"])
    compatibility_cli = SimpleNamespace(
        cpu=bool(cli.cpu),
        batch_size=int(cli.batch_size),
        test_batch=int(cli.test_batch),
        num_workers=int(cli.num_workers),
        val_num_workers=int(cli.val_num_workers),
        max_train_batches=0,
        max_val_batches=0,
        data_dir=str(cli.data_dir),
        source_checkpoint="",
    )
    restored = qmix_train._compatible_args(saved, compatibility_cli)
    # A full evaluator has no partial-validation mode.  Keep the train loader
    # only for its RandomCrop protocol assertion; it is never iterated.
    restored.max_train_batches = 0
    restored.max_val_batches = 0
    return restored


def load_post_q_state(payload: dict[str, Any], state: str) -> dict[str, dict[str, torch.Tensor]]:
    key = "decoder_combiner_state_dict" if state == "raw" else "ema_decoder_combiner_state_dict"
    saved = dict(payload[key])
    return {"d2": dict(saved["d2"]), "combiner": dict(saved["combiner"])}


def _component_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    marker = f"{prefix}."
    result = {name[len(marker) :]: value for name, value in state.items() if name.startswith(marker)}
    if not result:
        raise ValueError(f"continuous-q checkpoint has no {prefix} state")
    return result


def validate_external_post_initializer(
    payload: dict[str, Any], *, qmix_init_args: argparse.Namespace
) -> dict[str, Any]:
    """Validate a zero-training D2+combiner initializer such as hybrid v12.

    Its generator state is deliberately ignored.  The only deployed generator
    remains the fixed train-selected q mix restored above.  This lets us test
    a stronger saved D2/combiner behind exactly the same q2_hat distribution
    without learning or selecting on validation images.
    """

    if str(payload.get("stage", "")) != "explore2_continuous_q_receiver":
        raise ValueError("--post-q-init-checkpoint must be a continuous-q receiver checkpoint")
    saved_args = dict(_required(payload, "args"))
    if str(saved_args.get("arch", "")) != str(qmix_init_args.arch):
        raise ValueError("external post-q initializer Layer1 architecture differs from fixed q mix")
    if int(saved_args.get("embedding_dim", -1)) != int(qmix_init_args.embedding_dim):
        raise ValueError("external post-q initializer embedding D differs from fixed q mix")
    qmix_source = _canonical_path(qmix_init_args.source_checkpoint)
    external_source = _canonical_path(str(saved_args.get("source_checkpoint", "")))
    if external_source != qmix_source:
        raise ValueError(
            "external post-q initializer Layer1 source differs from fixed q mix: "
            f"{external_source} != {qmix_source}"
        )
    for key in ("receiver_state_dict", "ema_state_dict"):
        state = dict(_required(payload, key))
        _component_state(state, "d2")
        _component_state(state, "combiner")
    return saved_args


def load_external_post_q_state(
    payload: dict[str, Any], state: str
) -> dict[str, dict[str, torch.Tensor]]:
    key = "receiver_state_dict" if state == "raw" else "ema_state_dict"
    saved = dict(_required(payload, key))
    return {
        "d2": _component_state(saved, "d2"),
        "combiner": _component_state(saved, "combiner"),
    }


def assert_fixed_mix_runtime(
    fixed_mix: qmix_train.FixedTrainSelectedQMix, expected_weights: Sequence[float]
) -> None:
    """Check that the q generators are immutable and deployment-fixed."""

    assert_receiver_only_module(fixed_mix)
    fixed_mix.eval()
    if fixed_mix.training or any(member.training for member in fixed_mix.members):
        raise AssertionError("fixed q generators must remain in evaluation mode")
    if any(parameter.requires_grad for parameter in fixed_mix.parameters()):
        raise AssertionError("fixed q generators must not have trainable parameters in evaluation")
    observed = _as_float_list(fixed_mix.weights.detach().cpu())
    if not _close_float_lists(observed, _as_float_list(expected_weights)):
        raise AssertionError("runtime q-mix weights differ from the saved train-selected mix")


def build_receiver_for_state(
    source,
    *,
    init_args: argparse.Namespace,
    fixed_mix: qmix_train.FixedTrainSelectedQMix,
    post_state: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> continuous.ContinuousReceiver:
    """Build a fresh single D2+combiner and load either raw or EMA weights."""

    template = continuous.build_receiver(source, init_args, device)
    d2 = template.d2
    combiner = template.combiner
    del template
    d2.load_state_dict(post_state["d2"], strict=True)
    combiner.load_state_dict(post_state["combiner"], strict=True)
    receiver = continuous.ContinuousReceiver(fixed_mix, d2, combiner).to(device).eval()
    receiver.requires_grad_(False)
    continuous.assert_continuous_receiver_contract(receiver)
    return receiver


@torch.inference_mode()
def evaluate_full_validation(
    loader,
    *,
    source,
    receiver: continuous.ContinuousReceiver,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a receiver without allowing the target image into its graph."""

    source.e1.eval()
    source.d1.eval()
    receiver.eval()
    continuous.assert_continuous_receiver_contract(receiver)
    totals = {
        "psnr_x1": 0.0,
        "psnr_u2_hat": 0.0,
        "psnr_x2_hat": 0.0,
        "q2_hat_rms": 0.0,
    }
    examples = 0
    audited = False
    began = time.time()
    for batch_index, (images, _labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        # ``source.layer1`` is the measurement-side reconstruction needed to
        # construct the allowed receiver condition.  The source image is not
        # a receiver argument and becomes a metric target only after output.
        layer1 = source.layer1(images)
        condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
        output = receiver(condition)
        if not audited:
            assert_training_targets_are_not_inputs(
                receiver, condition, source_targets={"img": images}
            )
            audited = True
        batch = int(images.shape[0])
        values = {
            "psnr_x1": continuous.psnr_per_image(condition.x1, images).mean(),
            "psnr_u2_hat": continuous.psnr_per_image(output["u2_hat"], images).mean(),
            "psnr_x2_hat": continuous.psnr_per_image(output["x2_hat"], images).mean(),
            "q2_hat_rms": output["q2_hat"].float().square().mean().sqrt(),
        }
        for name, value in values.items():
            totals[name] += float(value) * batch
        examples += batch
        if batch_index % 10 == 0:
            print(
                f"[q-mix eval] batches={batch_index} images={examples}/{len(loader.dataset)} "
                f"elapsed={time.time() - began:.1f}s",
                flush=True,
            )
    if examples != EXPECTED_VALIDATION_IMAGES or examples != len(loader.dataset):
        raise AssertionError(
            "strict full validation must process exactly 100 DIV2K validation images, "
            f"got examples={examples} dataset={len(loader.dataset)}"
        )
    metrics = {name: value / examples for name, value in totals.items()}
    metrics.update(
        {
            "delta_x1": metrics["psnr_x2_hat"] - metrics["psnr_x1"],
            "evaluated_images": float(examples),
            "full_validation": 1.0,
            "receiver_only_audit": 1.0,
            "runtime_target_no_leak_audit": 1.0,
            "qonly_d2_audit": 1.0,
            "combiner_x1_u2_audit": 1.0,
        }
    )
    return metrics


def _report_output_path(
    checkpoint: Path, requested: str, *, post_initializer: Path | None = None
) -> Path:
    if requested:
        output = Path(requested).expanduser()
        if not output.is_absolute():
            output = HERE / output
    else:
        extra = (
            ""
            if post_initializer is None
            else f"_postinit-{post_initializer.parent.name}-{post_initializer.stem}"
        )
        output = RESULTS_ROOT / f"{checkpoint.parent.name}_{checkpoint.stem}{extra}_strict_fullval.json"
    output = output.resolve()
    try:
        output.relative_to(HERE.resolve())
    except ValueError as error:
        raise ValueError("--output must remain under MY-V2/jscc-f/explore-2") from error
    if output.suffix.lower() != ".json":
        raise ValueError("--output must have a .json suffix")
    return output


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def evaluate_checkpoint(cli: argparse.Namespace) -> Path:
    checkpoint_path = continuous.resolve_path(cli.checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_mix, receiver_init = validate_checkpoint(payload)
    init_args = rebuild_init_args(receiver_init, cli)

    # By default evaluate the saved q-mix receiver's own raw/EMA post-q
    # states.  An explicit continuous-q initializer is a *zero-training*
    # baseline: it contributes only its saved D2+combiner while the fixed
    # train-selected q generators and Layer1 stay identical.
    post_source = "saved_qmix_checkpoint"
    post_checkpoint_path: Path | None = None
    post_payload: dict[str, Any] | None = None
    post_init_args = init_args
    post_states: Iterable[str] = ("raw", "ema") if cli.state == "both" else (str(cli.state),)
    if str(cli.post_q_init_checkpoint):
        post_source = "external_continuous_q_initializer_zero_training"
        post_checkpoint_path = continuous.resolve_path(cli.post_q_init_checkpoint).resolve()
        post_payload = torch.load(post_checkpoint_path, map_location="cpu", weights_only=False)
        external_args = validate_external_post_initializer(
            post_payload, qmix_init_args=init_args
        )
        external_receiver_init = {
            "args": external_args,
            "checkpoint": str(post_checkpoint_path),
            "state": "ema",
        }
        post_init_args = rebuild_init_args(external_receiver_init, cli)
        post_states = (
            ("raw", "ema")
            if cli.post_q_init_state == "both"
            else (str(cli.post_q_init_state),)
        )

    source, train_loader, val_loader, device = continuous.build_source_and_loaders(init_args)
    assert_div2k_crop_protocol(train_loader, val_loader)
    if len(val_loader.dataset) != EXPECTED_VALIDATION_IMAGES:
        raise AssertionError(
            f"strict evaluator requires the complete 100-image DIV2K validation set, "
            f"got {len(val_loader.dataset)}"
        )

    # Build all receiver-side post-q modules before turning the unused sender
    # Layer2 objects into tripwires.  From this point onward, an accidental
    # E2/z2/true-q2 sender path fails immediately.
    members = qmix_train.load_fixed_mix_members(source, saved_mix, init_args, device)
    fixed_mix = qmix_train.FixedTrainSelectedQMix(members, saved_mix["weights"]).to(device)
    assert_fixed_mix_runtime(fixed_mix, saved_mix["weights"])
    source.e2 = SenderLayer2Forbidden().to(device)
    source.d2 = SenderLayer2Forbidden().to(device)
    source.combiner = SenderLayer2Forbidden().to(device)
    source.checkpoint = {}
    source.layer2_checkpoint = {}

    evaluations: dict[str, Any] = {}
    for state in post_states:
        if post_payload is None:
            post_state = load_post_q_state(payload, state)
            state_key = (
                "decoder_combiner_state_dict"
                if state == "raw"
                else "ema_decoder_combiner_state_dict"
            )
        else:
            post_state = load_external_post_q_state(post_payload, state)
            state_key = "receiver_state_dict" if state == "raw" else "ema_state_dict"
        receiver = build_receiver_for_state(
            source,
            init_args=post_init_args,
            fixed_mix=fixed_mix,
            post_state=post_state,
            device=device,
        )
        assert_fixed_mix_runtime(fixed_mix, saved_mix["weights"])
        metrics = evaluate_full_validation(val_loader, source=source, receiver=receiver, device=device)
        ablations = continuous.receiver_ablations(
            val_loader, source=source, receiver=receiver, device=device
        )
        metrics.update(ablations)
        metrics["crop_protocol_audit"] = 1.0
        metrics["sender_layer2_tripwire_audit"] = 1.0
        metrics["fixed_q_mix_audit"] = 1.0
        metrics["goal_min_delta_db"] = 0.5
        metrics["goal_min_condition_shuffle_drop_db"] = 0.1
        metrics["goal_met"] = float(
            metrics["delta_x1"] >= metrics["goal_min_delta_db"]
            and metrics["condition_shuffle_drop"] >= metrics["goal_min_condition_shuffle_drop_db"]
        )
        evaluations[state] = {
            "state_key": state_key,
            "metrics": metrics,
        }
        print(
            f"[q-mix strict full-val {state}] "
            f"PSNR(x1)={metrics['psnr_x1']:.6f} "
            f"PSNR(x2_hat)={metrics['psnr_x2_hat']:.6f} "
            f"delta={metrics['delta_x1']:+.6f}dB "
            f"condition_shuffle_drop={metrics['condition_shuffle_drop']:.6f}dB "
            f"goal_met={int(metrics['goal_met'])}",
            flush=True,
        )
        del receiver
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "evaluator": "eval_continuous_q_mix_receiver.py",
        "read_only_checkpoint_evaluation": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_saved_validation_metrics": _json_ready(payload.get("metrics", {})),
        "evaluated_states": list(evaluations),
        "post_q_state_source": post_source,
        "post_q_initializer": (
            None
            if post_payload is None
            else {
                "checkpoint": str(post_checkpoint_path),
                "epoch": int(post_payload.get("epoch", -1)),
                "saved_validation_metrics": _json_ready(post_payload.get("metrics", {})),
                "topology_args": {
                    key: _json_ready(getattr(post_init_args, key))
                    for key in (
                        "d2_type",
                        "d2_highres_width",
                        "d2_highres_blocks",
                        "combiner_type",
                        "combiner_width",
                        "combiner_blocks",
                        "base_combiner_width",
                        "base_combiner_blocks",
                    )
                },
                "zero_training": True,
            }
        ),
        "receiver_contract": {
            "deployment_inputs": ["z1", "x1"],
            "generation": "fixed_train_selected_sum_i(w_i*G_i(z1,x1)) -> q2_hat",
            "decode_path": "q2_hat -> one_saved_D2 -> combiner(x1,u2_hat) -> x2_hat",
            "forbidden_inputs": sorted(FORBIDDEN_RECEIVER_INPUTS),
            "sender_layer2_tripwire": True,
            "raw_image_role": "Layer1 condition source and post-forward metric target only",
        },
        "fixed_train_selected_q_mix": saved_mix,
        "layer1": {
            "arch": str(init_args.arch),
            "source_checkpoint": str(init_args.source_checkpoint),
            "latent_channels": int(source.args.latent_ch),
        },
        "validation": {
            "dataset": "DIV2K validation",
            "expected_images": EXPECTED_VALIDATION_IMAGES,
            "transform": repr(val_loader.dataset.transform),
            "train_transform_guard": repr(train_loader.dataset.transform),
            "crop_protocol": "train RandomCrop(256)+RandomHorizontalFlip+ToTensor; val CenterCrop(256)+ToTensor",
        },
        "evaluations": evaluations,
    }
    output = _report_output_path(
        checkpoint_path, cli.output, post_initializer=post_checkpoint_path
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved strict full-val JSON: {output}", flush=True)
    return output


def run_contract_smoke() -> None:
    """CPU-only structural proof for q-mix, q-only D2, no leak, and crops."""

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
    fixed_mix = qmix_train.FixedTrainSelectedQMix(
        [SmokeMember(1.0), SmokeMember(2.0), SmokeMember(4.0)], [0.6, 0.3, 0.1]
    ).to(device)
    d2 = continuous.Layer1InitializedD2(32, 16, SmokeD1()).to(device)
    combiner = continuous.ReceiverCombiner(width=8, blocks=1).to(device)
    receiver = continuous.ContinuousReceiver(fixed_mix, d2, combiner).to(device).eval()
    receiver.requires_grad_(False)
    continuous.assert_continuous_receiver_contract(receiver)
    assert_fixed_mix_runtime(fixed_mix, [0.6, 0.3, 0.1])
    condition = make_receiver_condition(
        torch.randn(2, 16, 16, 16, device=device),
        torch.rand(2, 3, 256, 256, device=device),
        detach=True,
    )
    output = receiver(condition)
    expected = 0.6 * 1.0 + 0.3 * 2.0 + 0.1 * 4.0
    torch.testing.assert_close(
        output["q2_hat"], torch.full_like(output["q2_hat"], expected), rtol=0.0, atol=1e-6
    )
    assert_training_targets_are_not_inputs(
        receiver, condition, source_targets={"img": torch.rand_like(output["x2_hat"])}
    )
    raw = {"d2": receiver.d2.state_dict(), "combiner": receiver.combiner.state_dict()}
    state = load_post_q_state(
        {
            "decoder_combiner_state_dict": raw,
            "ema_decoder_combiner_state_dict": raw,
        },
        "ema",
    )
    if set(state) != {"d2", "combiner"}:
        raise AssertionError("EMA post-q state selection failed")
    train_loader = SimpleNamespace(
        dataset=SimpleNamespace(transform="RandomCrop(256)+RandomHorizontalFlip+ToTensor")
    )
    val_loader = SimpleNamespace(dataset=SimpleNamespace(transform="CenterCrop(256)+ToTensor"))
    assert_div2k_crop_protocol(train_loader, val_loader)
    print(
        "[PASS] q-mix strict evaluator CPU smoke "
        f"q={tuple(output['q2_hat'].shape)} u2={tuple(output['u2_hat'].shape)} "
        f"x2={tuple(output['x2_hat'].shape)} raw_ema_state=1 qonly_d2=1 "
        "combiner_x1_u2=1 no_leak=1 fixed_q=1 crop_contract=1",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="saved explore2_continuous_q_mix_receiver checkpoint to inspect read-only",
    )
    parser.add_argument(
        "--state", choices=["raw", "ema", "both"], default="both", help="post-q state to evaluate"
    )
    parser.add_argument(
        "--post-q-init-checkpoint",
        default="",
        help=(
            "optional continuous-q checkpoint supplying a zero-training D2+combiner baseline; "
            "its generator is ignored and the saved fixed train-selected q mix remains active"
        ),
    )
    parser.add_argument(
        "--post-q-init-state",
        choices=["raw", "ema", "both"],
        default="ema",
        help="raw/EMA D2+combiner state for --post-q-init-checkpoint",
    )
    parser.add_argument("--data-dir", default="", help="optional DIV2K location override")
    parser.add_argument("--batch-size", type=int, default=2, help="unused train-loader batch size")
    parser.add_argument("--test-batch", type=int, default=2, help="validation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="train-loader workers for crop guard")
    parser.add_argument("--val-num-workers", type=int, default=0, help="validation DataLoader workers")
    parser.add_argument(
        "--output",
        default="",
        help="JSON path relative to explore-2; default=results-receiver/<checkpoint>_strict_fullval.json",
    )
    parser.add_argument("--cpu", action="store_true", help="force CPU evaluation")
    parser.add_argument(
        "--smoke-contract",
        action="store_true",
        help="CPU-only q-mix/q-only-D2/no-leak/crop structural test; does not load data/checkpoints",
    )
    args = parser.parse_args()
    if bool(args.smoke_contract):
        return args
    if not str(args.checkpoint):
        parser.error("--checkpoint is required unless --smoke-contract")
    for name in ("batch_size", "test_batch"):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("num_workers", "val_num_workers"):
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if bool(parsed.smoke_contract):
        run_contract_smoke()
    else:
        evaluate_checkpoint(parsed)
