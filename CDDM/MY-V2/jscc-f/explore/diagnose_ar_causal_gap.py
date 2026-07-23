#!/usr/bin/env python3
"""A1: decompose conditional, exposure, hard-decision, and decoder gaps.

This is a read-only analysis experiment.  It never constructs an optimizer or
updates a checkpoint.  E1/D1/E2/FSQ/D2/combiner remain in eval mode with
``requires_grad=False``.  Hashes for E2, FSQ, D2, and combiner are recorded
before and after each validation run.

For each existing AR checkpoint the script decodes four predictions:

* teacher_hard: per-position argmax under the complete oracle prefix;
* teacher_soft: posterior-mean FSQ value under the oracle prefix;
* rollout_hard: deployable greedy autoregressive generation;
* rollout_soft: posterior-mean FSQ value under generated prefixes.

Teacher-soft and rollout-soft are diagnostic continuous-code bounds, not valid
discrete deployment results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
if str(JSCCF_DIR) not in sys.path:
    sys.path.insert(0, str(JSCCF_DIR))


DEFAULT_CHECKPOINTS = [
    (
        "MY-V2/jscc-f/checkpoints-ar-ifsq/"
        "jscc_f_ifsq-prefix-ar-k125_stage_ifsq_ar_fsq_l5x5x5_best.pth"
    ),
    (
        "MY-V2/jscc-f/checkpoints-ar-ifsq/"
        "jscc_f_ifsq-prefix-ar-k125_stage_ifsq_ar_fsq_l5x5x5_latest.pth"
    ),
    (
        "MY-V2/jscc-f/checkpoints-ar-ifsq-per-token-y1/"
        "jscc_f_ifsq-per-token-y1-k125_stage_ifsq_ar_fsq_l5x5x5_per-token-y1_best.pth"
    ),
    (
        "MY-V2/jscc-f/checkpoints-ar-ifsq-per-token-y1/"
        "jscc_f_ifsq-per-token-y1-k125_stage_ifsq_ar_fsq_l5x5x5_per-token-y1_latest.pth"
    ),
    (
        "MY-V2/jscc-f/checkpoints-ar-ifsq-per-token-y1/"
        "jscc_f_ifsq-per-token-y1-k4913_stage_ifsq_ar_fsq_l17x17x17_per-token-y1_best.pth"
    ),
    (
        "MY-V2/jscc-f/checkpoints-ar-ifsq-per-token-y1/"
        "jscc_f_ifsq-per-token-y1-k4913_stage_ifsq_ar_fsq_l17x17x17_per-token-y1_latest.pth"
    ),
]


def load_ar():
    path = JSCCF_DIR / "train_stage-fsq-ar.py"
    spec = importlib.util.spec_from_file_location("jsccf_ar_causal_gap", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ar-checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output",
        default="MY-V2/jscc-f/explore/results-ar/ar_causal_gap_a1.json",
    )
    return parser.parse_args()


def module_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_audit(module: torch.nn.Module) -> dict[str, Any]:
    parameters = list(module.parameters())
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        "training": bool(module.training),
        "state_sha256": module_hash(module),
    }


def frozen_modules(system) -> dict[str, torch.nn.Module]:
    return {
        "E2_tokenizer_e3": system.tokenizer.e3,
        "FSQ_quantizer": system.tokenizer.quantizer,
        "D2_tokenizer_d3": system.tokenizer.d3,
        "combiner": system.combiner,
    }


def audit_frozen(system) -> dict[str, dict[str, Any]]:
    return {name: module_audit(module) for name, module in frozen_modules(system).items()}


def assert_frozen(audit: dict[str, dict[str, Any]], phase: str) -> None:
    failures = {
        name: values
        for name, values in audit.items()
        if int(values["trainable_parameter_count"]) != 0 or bool(values["training"])
    }
    if failures:
        raise RuntimeError(f"frozen module audit failed {phase}: {failures}")


def compare_audits(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    names = sorted(before)
    unchanged = {
        name: bool(before[name]["state_sha256"] == after[name]["state_sha256"])
        for name in names
    }
    if not all(unchanged.values()):
        raise RuntimeError(f"frozen module state changed: {unchanged}")
    return {
        "all_unchanged": bool(all(unchanged.values())),
        "unchanged": unchanged,
        "before": before,
        "after": after,
    }


def checkpoint_condition_mode(payload: dict[str, Any]) -> str:
    saved_args = dict(payload.get("args", {}))
    return str(payload.get("condition_mode", saved_args.get("condition_mode", "prefix")))


def make_model(ar, system, payload: dict[str, Any], device: torch.device):
    saved = argparse.Namespace(**payload["args"])
    latent = dict(
        torch.load(
            ar.resolved(saved.layer1_checkpoint),
            map_location="cpu",
            weights_only=False,
        ).get("latent", {})
    )
    y1_shape = latent.get("z1", [16, 16, 16])
    model = ar.IFSQPrefixRasterAR(
        int(y1_shape[0]),
        system.height,
        system.width,
        system.vocabulary,
        hidden=int(saved.hidden),
        layers=int(saved.layers),
        heads=int(saved.heads),
        condition_blocks=int(saved.condition_blocks),
        ff_mult=float(saved.ff_mult),
        dropout=float(saved.dropout),
        condition_mode=checkpoint_condition_mode(payload),
    ).to(device)
    model.load_state_dict(payload["ar_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def index_digits(indices: torch.Tensor, system) -> torch.Tensor:
    flat = indices.long().flatten(1)
    levels = system.levels.view(1, 1, -1)
    multipliers = system.multipliers.view(1, 1, -1)
    return (flat.unsqueeze(-1) // multipliers) % levels


def image_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).square().flatten(1).mean(dim=1)


def image_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -10.0 * image_mse(prediction, target).clamp_min(1e-12).log10()


def entropy_bits(counts: torch.Tensor) -> float:
    probability = counts.float() / counts.sum().clamp_min(1.0)
    positive = probability > 0
    return float(
        -(probability * probability.clamp_min(1e-12).log2())
        .masked_fill(~positive, 0.0)
        .sum()
        .item()
    )


class Accumulator:
    def __init__(self, tokens: int, dimensions: int, vocabulary: int) -> None:
        self.images = 0
        self.tokens = int(tokens)
        self.dimensions = int(dimensions)
        self.vocabulary = int(vocabulary)
        self.scalar_sums: dict[str, float] = {}
        self.image_sums: dict[str, float] = {}
        self.image_mse_sums: dict[str, float] = {}
        self.position_correct = {
            name: torch.zeros(self.tokens, dtype=torch.float64)
            for name in ("teacher_hard", "rollout_hard")
        }
        self.digit_correct = {
            name: torch.zeros(self.dimensions, dtype=torch.float64)
            for name in ("teacher_hard", "rollout_hard")
        }
        self.target_counts = torch.zeros(self.vocabulary, dtype=torch.long)
        self.generated_counts = torch.zeros(self.vocabulary, dtype=torch.long)
        self.max_cache_error = 0.0

    def add_scalar(self, name: str, value: float) -> None:
        self.scalar_sums[name] = self.scalar_sums.get(name, 0.0) + float(value)

    def add_image_quality(
        self, name: str, prediction: torch.Tensor, target: torch.Tensor
    ) -> None:
        self.image_sums[name] = self.image_sums.get(name, 0.0) + float(
            image_psnr(prediction, target).sum().item()
        )
        self.image_mse_sums[name] = self.image_mse_sums.get(name, 0.0) + float(
            image_mse(prediction, target).sum().item()
        )


def bins(values: torch.Tensor) -> dict[str, float]:
    edges = [0, 16, 64, 128, 192, int(values.numel())]
    return {
        f"{left}:{right}": float(values[left:right].mean().item())
        for left, right in zip(edges[:-1], edges[1:])
        if right > left
    }


@torch.no_grad()
def evaluate_checkpoint(
    ar,
    checkpoint: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = ar.resolved(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**payload["args"])
    saved.cpu = bool(args.cpu)
    system, layer2_payload, layer2_args, layer2_path = ar.load_frozen_system(saved, device)
    system.eval()
    model = make_model(ar, system, payload, device)

    runtime = argparse.Namespace(
        data_dir=str(saved.data_dir),
        batch_size=int(args.batch_size),
        test_batch=int(args.batch_size),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.num_workers),
        cpu=bool(args.cpu),
    )
    _train_loader, val_loader, loader_device = ar.build_loaders(layer2_args, runtime)
    if loader_device != device:
        raise RuntimeError(f"loader/model device mismatch: {loader_device} vs {device}")

    before = audit_frozen(system)
    assert_frozen(before, "before")
    accumulator = Accumulator(
        system.height * system.width,
        system.fsq_d,
        system.vocabulary,
    )
    cache_checked = False

    print(
        f"[A1 start] epoch={payload.get('epoch')} mode={model.condition_mode} "
        f"K={system.vocabulary} checkpoint={checkpoint_path}",
        flush=True,
    )
    for batch_index, (images, _labels) in enumerate(val_loader, start=1):
        if int(args.max_val_batches) > 0 and batch_index > int(args.max_val_batches):
            break
        images = images.to(device, non_blocking=True)
        target = system.oracle_targets(images)
        true_map = target["indices"].long()
        true_flat = true_map.flatten(1)
        batch = int(images.shape[0])

        teacher_logits = model.forward_teacher(
            target["y1"], target["x1"], true_map
        )
        teacher_ids = teacher_logits.argmax(dim=-1)
        rollout_logits, rollout_ids = model.generate(
            target["y1"],
            target["x1"],
            steps=system.height * system.width,
            sample_logits=False,
        )
        if not cache_checked:
            replay_logits = model.forward_sequence(
                target["y1"], target["x1"], rollout_ids.long()
            )
            accumulator.max_cache_error = max(
                accumulator.max_cache_error,
                float((replay_logits - rollout_logits).abs().max().item()),
            )
            cache_checked = True

        teacher_map = teacher_ids.view(batch, system.height, system.width)
        rollout_map = rollout_ids.view(batch, system.height, system.width)
        teacher_hard_q = system.indices_to_q(teacher_map)
        rollout_hard_q = system.indices_to_q(rollout_map)
        teacher_soft_q = system.logits_to_soft_q(
            teacher_logits, float(args.soft_temperature)
        )
        rollout_soft_q = system.logits_to_soft_q(
            rollout_logits, float(args.soft_temperature)
        )

        all_q = torch.cat(
            [teacher_hard_q, teacher_soft_q, rollout_hard_q, rollout_soft_q],
            dim=0,
        )
        all_y1 = torch.cat([target["y1"]] * 4, dim=0)
        all_x1 = torch.cat([target["x1"]] * 4, dim=0)
        all_final = system.decode_q(all_q, all_y1, all_x1)
        teacher_hard_final, teacher_soft_final, rollout_hard_final, rollout_soft_final = (
            all_final.chunk(4, dim=0)
        )

        quality_predictions = {
            "x1": target["x1"],
            "oracle": target["oracle"],
            "teacher_hard": teacher_hard_final,
            "teacher_soft": teacher_soft_final,
            "rollout_hard": rollout_hard_final,
            "rollout_soft": rollout_soft_final,
        }
        for name, prediction in quality_predictions.items():
            accumulator.add_image_quality(name, prediction, images)

        accumulator.add_scalar(
            "teacher_ce_sum",
            F.cross_entropy(
                teacher_logits.reshape(-1, system.vocabulary),
                true_flat.reshape(-1),
                reduction="sum",
            ).item(),
        )
        accumulator.add_scalar(
            "rollout_context_ce_sum",
            F.cross_entropy(
                rollout_logits.reshape(-1, system.vocabulary),
                true_flat.reshape(-1),
                reduction="sum",
            ).item(),
        )
        for name, predicted in (
            ("teacher_hard", teacher_ids),
            ("rollout_hard", rollout_ids),
        ):
            correct = predicted.eq(true_flat)
            accumulator.position_correct[name] += correct.double().sum(dim=0).cpu()
            predicted_digits = index_digits(
                predicted.view(batch, system.height, system.width), system
            )
            true_digits = index_digits(true_map, system)
            accumulator.digit_correct[name] += (
                predicted_digits.eq(true_digits)
                .double()
                .sum(dim=(0, 1))
                .cpu()
            )
            hard_q = teacher_hard_q if name == "teacher_hard" else rollout_hard_q
            accumulator.add_scalar(
                f"{name}_q_l1_sum",
                (hard_q.float() - target["q2"].float())
                .abs()
                .flatten(1)
                .mean(dim=1)
                .sum()
                .item(),
            )
            accumulator.add_scalar(
                f"{name}_q_mse_sum",
                (hard_q.float() - target["q2"].float())
                .square()
                .flatten(1)
                .mean(dim=1)
                .sum()
                .item(),
            )

        accumulator.target_counts += torch.bincount(
            true_flat.reshape(-1).cpu(), minlength=system.vocabulary
        )
        accumulator.generated_counts += torch.bincount(
            rollout_ids.reshape(-1).cpu(), minlength=system.vocabulary
        )
        accumulator.images += batch
        if batch_index % 10 == 0:
            print(
                f"[A1 progress] epoch={payload.get('epoch')} "
                f"batches={batch_index} images={accumulator.images}",
                flush=True,
            )

    if accumulator.images < 1:
        raise RuntimeError("validation loader produced no images")

    after = audit_frozen(system)
    assert_frozen(after, "after")
    freeze_report = compare_audits(before, after)
    images = float(accumulator.images)
    token_count = images * float(accumulator.tokens)
    digit_count = token_count
    quality = {
        name: {
            "psnr": accumulator.image_sums[name] / images,
            "mse": accumulator.image_mse_sums[name] / images,
        }
        for name in accumulator.image_sums
    }
    baseline_mse = quality["x1"]["mse"]
    oracle_mse = quality["oracle"]["mse"]
    recoverable_mse = max(1e-12, baseline_mse - oracle_mse)
    goal_psnr = quality["x1"]["psnr"] + 0.5
    for name in ("teacher_hard", "teacher_soft", "rollout_hard", "rollout_soft"):
        quality[name]["delta_psnr_x1"] = (
            quality[name]["psnr"] - quality["x1"]["psnr"]
        )
        quality[name]["oracle_mse_gain_fraction"] = (
            baseline_mse - quality[name]["mse"]
        ) / recoverable_mse
        quality[name]["meets_x1_plus_0p5"] = bool(
            quality[name]["psnr"] >= goal_psnr
        )

    position_accuracy = {
        name: values / images
        for name, values in accumulator.position_correct.items()
    }
    token_metrics: dict[str, Any] = {
        "teacher_ce": accumulator.scalar_sums["teacher_ce_sum"] / token_count,
        "teacher_nll_bits": (
            accumulator.scalar_sums["teacher_ce_sum"] / token_count / math.log(2.0)
        ),
        "rollout_context_ce": (
            accumulator.scalar_sums["rollout_context_ce_sum"] / token_count
        ),
        "rollout_context_nll_bits": (
            accumulator.scalar_sums["rollout_context_ce_sum"]
            / token_count
            / math.log(2.0)
        ),
        "teacher_hard_joint_accuracy": float(
            position_accuracy["teacher_hard"].mean().item()
        ),
        "rollout_hard_joint_accuracy": float(
            position_accuracy["rollout_hard"].mean().item()
        ),
        "teacher_hard_digit_accuracy": [
            float(value)
            for value in (
                accumulator.digit_correct["teacher_hard"] / digit_count
            )
        ],
        "rollout_hard_digit_accuracy": [
            float(value)
            for value in (
                accumulator.digit_correct["rollout_hard"] / digit_count
            )
        ],
        "teacher_hard_q_l1": (
            accumulator.scalar_sums["teacher_hard_q_l1_sum"] / images
        ),
        "teacher_hard_q_mse": (
            accumulator.scalar_sums["teacher_hard_q_mse_sum"] / images
        ),
        "rollout_hard_q_l1": (
            accumulator.scalar_sums["rollout_hard_q_l1_sum"] / images
        ),
        "rollout_hard_q_mse": (
            accumulator.scalar_sums["rollout_hard_q_mse_sum"] / images
        ),
        "target_entropy_bits": entropy_bits(accumulator.target_counts),
        "target_unique_tokens": int((accumulator.target_counts > 0).sum().item()),
        "rollout_entropy_bits": entropy_bits(accumulator.generated_counts),
        "rollout_unique_tokens": int(
            (accumulator.generated_counts > 0).sum().item()
        ),
        "max_cache_replay_error": float(accumulator.max_cache_error),
    }
    position_report = {
        name: {
            "bins": bins(values),
            "positions": {
                str(position): float(values[position].item())
                for position in (0, 1, 2, 3, 7, 15, 31, 63, 127, 191, 255)
            },
        }
        for name, values in position_accuracy.items()
    }
    report = {
        "checkpoint": checkpoint_path,
        "epoch": int(payload.get("epoch", -1)),
        "condition_mode": model.condition_mode,
        "layer2_level": int(saved.layer2_level),
        "vocabulary": int(system.vocabulary),
        "layer2_checkpoint": layer2_path,
        "layer2_reference_metrics": dict(layer2_payload.get("metrics", {})),
        "device": str(device),
        "images": int(accumulator.images),
        "soft_temperature": float(args.soft_temperature),
        "goal_psnr_x1_plus_0p5": float(goal_psnr),
        "quality": quality,
        "tokens": token_metrics,
        "position_accuracy": position_report,
        "frozen_state": freeze_report,
    }
    print(
        f"[A1 done] epoch={report['epoch']} mode={report['condition_mode']} "
        f"K={report['vocabulary']} teacher_hard={quality['teacher_hard']['psnr']:.4f} "
        f"teacher_soft={quality['teacher_soft']['psnr']:.4f} "
        f"rollout_hard={quality['rollout_hard']['psnr']:.4f} "
        f"rollout_soft={quality['rollout_soft']['psnr']:.4f} "
        f"frozen={freeze_report['all_unchanged']}",
        flush=True,
    )
    del model, system
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.max_val_batches) < 0:
        raise ValueError("--max-val-batches must be non-negative")
    if float(args.soft_temperature) <= 0.0:
        raise ValueError("--soft-temperature must be positive")
    ar = load_ar()
    ar.base.seed_everything(int(args.seed))
    device = torch.device(
        "cpu"
        if bool(args.cpu)
        else "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    reports = [
        evaluate_checkpoint(ar, checkpoint, args, device)
        for checkpoint in args.ar_checkpoints
    ]
    result = {
        "experiment": "A1_teacher_rollout_decoder_gap",
        "read_only": True,
        "optimizer_created": False,
        "protocol": {
            "validation": "DIV2K valid in checkpoint data configuration",
            "batch_size": int(args.batch_size),
            "max_val_batches": int(args.max_val_batches),
            "seed": int(args.seed),
            "teacher_soft_is_deployable": False,
            "rollout_soft_is_deployable": False,
            "goal": "mean PSNR >= mean x1 PSNR + 0.5 dB",
        },
        "runs": reports,
    }
    output = Path(ar.resolved(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[A1 saved] {output}", flush=True)
    return result


if __name__ == "__main__":
    run(parse_args())
