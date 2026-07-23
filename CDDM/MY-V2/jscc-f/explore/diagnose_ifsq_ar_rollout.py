#!/usr/bin/env python3
"""Diagnostics for the iFSQ prefix-AR rollout gap.

This script does not train or modify a checkpoint.  It separates four failure
modes which are otherwise conflated by one ``rollout_token_accuracy`` number:

* teacher-forced prediction versus generated-prefix prediction;
* error accumulation along the 256-token raster;
* cache/generation alignment;
* marginal token predictability and generated-token collapse.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
if str(JSCCF_DIR) not in sys.path:
    sys.path.insert(0, str(JSCCF_DIR))


def load_ar():
    path = JSCCF_DIR / "train_stage-fsq-ar.py"
    spec = importlib.util.spec_from_file_location("jsccf_ifsq_ar_diagnostic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--ar-checkpoint",
        default=(
            "MY-V2/jscc-f/checkpoints-ar-ifsq/"
            "jscc_f_ifsq-prefix-ar-k125_stage_ifsq_ar_fsq_l5x5x5_latest.pth"
        ),
    )
    parser.add_argument("--max-val-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=64,
        help="Number of generated raster tokens; 64 is enough to expose prefix drift and is much faster than 256.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def mean_or_nan(values: torch.Tensor) -> float:
    return float(values.float().mean().item()) if values.numel() else float("nan")


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(1).clamp_min(1e-12)
    return float((-10.0 * mse.log10()).mean().item())


def entropy_bits(counts: torch.Tensor) -> torch.Tensor:
    probability = counts.float() / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    positive = probability > 0
    terms = probability * probability.clamp_min(1e-12).log2()
    return -terms.masked_fill(~positive, 0.0).sum(dim=-1)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    ar = load_ar()
    checkpoint_path = ar.resolved(args.ar_checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**payload["args"])
    saved.cpu = bool(args.cpu)
    device = torch.device("cpu" if args.cpu else "cuda:0" if torch.cuda.is_available() else "cpu")

    system, layer2_payload, layer2_args, layer2_path = ar.load_frozen_system(saved, device)
    latent = dict(torch.load(ar.resolved(saved.layer1_checkpoint), map_location="cpu", weights_only=False).get("latent", {}))
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
    ).to(device)
    model.load_state_dict(payload["ar_state_dict"], strict=True)
    model.eval()

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

    full_tokens = int(system.height * system.width)
    rollout_tokens = min(full_tokens, int(args.rollout_steps))
    if rollout_tokens < 1:
        raise ValueError("--rollout-steps must be positive")
    vocabulary = int(system.vocabulary)
    teacher_correct = torch.zeros(rollout_tokens, dtype=torch.float64)
    rollout_correct = torch.zeros(rollout_tokens, dtype=torch.float64)
    prefix_exact = torch.zeros(rollout_tokens, dtype=torch.float64)
    conditional_correct = torch.zeros(rollout_tokens, dtype=torch.float64)
    conditional_count = torch.zeros(rollout_tokens, dtype=torch.float64)
    target_counts = torch.zeros(rollout_tokens, vocabulary, dtype=torch.long)
    generated_counts = torch.zeros(vocabulary, dtype=torch.long)
    total_images = 0
    total_cache_error = 0.0
    total_teacher_ce = 0.0
    total_condition_only_ce = 0.0
    total_generated_prefix_ce = 0.0
    condition_only_correct = torch.zeros(rollout_tokens, dtype=torch.float64)
    image_metrics = {name: 0.0 for name in ("psnr_x1", "psnr_oracle", "psnr_generated", "gap_generated")}

    for batch_index, (images, _labels) in enumerate(val_loader, start=1):
        if int(args.max_val_batches) > 0 and batch_index > int(args.max_val_batches):
            break
        images = images.to(device, non_blocking=True)
        target = system.oracle_targets(images)
        true = target["indices"].flatten(1).long()

        teacher_logits = model.forward_teacher(target["y1"], target["x1"], target["indices"])
        condition_history = torch.zeros_like(true[:, :rollout_tokens])
        condition_only_logits = model.forward_sequence(target["y1"], target["x1"], condition_history)
        rollout_logits, generated = model.generate(
            target["y1"], target["x1"], steps=rollout_tokens, sample_logits=False
        )
        true_rollout = true[:, :rollout_tokens]
        generated_prefix_logits = model.forward_sequence(target["y1"], target["x1"], generated.long())
        cache_error = float((generated_prefix_logits - rollout_logits).abs().max().item())
        total_cache_error = max(total_cache_error, cache_error)

        teacher_pred = teacher_logits.argmax(dim=-1)[:, :rollout_tokens]
        teacher_logits_rollout = teacher_logits[:, :rollout_tokens]
        condition_only_pred = condition_only_logits.argmax(dim=-1)
        rollout_pred = generated_prefix_logits.argmax(dim=-1)
        t_correct = teacher_pred.eq(true_rollout)
        r_correct = rollout_pred.eq(true_rollout)
        teacher_correct += t_correct.double().sum(dim=0).cpu()
        condition_only_correct += condition_only_pred.eq(true_rollout).double().sum(dim=0).cpu()
        rollout_correct += r_correct.double().sum(dim=0).cpu()
        prefix_ok = r_correct.cumprod(dim=1)
        prefix_exact += prefix_ok.double().sum(dim=0).cpu()
        previous_ok = torch.cat(
            [torch.ones_like(prefix_ok[:, :1]), prefix_ok[:, :-1]], dim=1
        )
        conditional_correct += (r_correct & previous_ok.bool()).double().sum(dim=0).cpu()
        conditional_count += previous_ok.double().sum(dim=0).cpu()

        total_teacher_ce += float(F.cross_entropy(teacher_logits_rollout.reshape(-1, vocabulary), true_rollout.reshape(-1), reduction="sum").item())
        total_condition_only_ce += float(F.cross_entropy(condition_only_logits.reshape(-1, vocabulary), true_rollout.reshape(-1), reduction="sum").item())
        total_generated_prefix_ce += float(F.cross_entropy(generated_prefix_logits.reshape(-1, vocabulary), true_rollout.reshape(-1), reduction="sum").item())
        for position in range(rollout_tokens):
            target_counts[position] += torch.bincount(true_rollout[:, position].cpu(), minlength=vocabulary)
        generated_counts += torch.bincount(generated.reshape(-1).cpu(), minlength=vocabulary)

        if rollout_tokens == full_tokens:
            final = system.decode_q(system.indices_to_q(generated.view(-1, system.height, system.width)), target["y1"], target["x1"])
            image_metrics["psnr_x1"] += psnr(target["x1"], images) * int(images.shape[0])
            image_metrics["psnr_oracle"] += psnr(target["oracle"], images) * int(images.shape[0])
            image_metrics["psnr_generated"] += psnr(final, images) * int(images.shape[0])
            image_metrics["gap_generated"] += (psnr(target["oracle"], images) - psnr(final, images)) * int(images.shape[0])
        total_images += int(images.shape[0])

    if total_images < 1:
        raise RuntimeError("diagnostic loader produced no images")
    teacher_acc = teacher_correct / float(total_images)
    condition_only_acc = condition_only_correct / float(total_images)
    rollout_acc = rollout_correct / float(total_images)
    prefix_acc = prefix_exact / float(total_images)
    conditional_acc = conditional_correct / conditional_count.clamp_min(1.0)
    mode_acc = target_counts.max(dim=1).values.double() / float(total_images)
    target_entropy = entropy_bits(target_counts).double()
    target_global_counts = target_counts.sum(dim=0)
    target_global_probability = target_global_counts.float() / target_global_counts.sum().clamp_min(1.0)
    target_global_entropy = float(
        -(target_global_probability.clamp_min(1e-12) * target_global_probability.clamp_min(1e-12).log2()).sum().item()
    )
    generated_probability = generated_counts.float() / generated_counts.sum().clamp_min(1.0)
    generated_entropy = float(-(generated_probability.clamp_min(1e-12) * generated_probability.clamp_min(1e-12).log2()).sum().item())

    def bins(values: torch.Tensor) -> dict[str, float]:
        edges = sorted(set([0, min(16, rollout_tokens), min(64, rollout_tokens), min(128, rollout_tokens), min(192, rollout_tokens), rollout_tokens]))
        return {
            f"{edges[i]}:{edges[i + 1]}": mean_or_nan(values[edges[i] : edges[i + 1]])
            for i in range(len(edges) - 1)
        }

    probe_positions = sorted(set([0, 1, 2, 3, 4, 7, 15, 31, 63, rollout_tokens - 1]))
    probe_positions = [position for position in probe_positions if position < rollout_tokens]

    def probes(values: torch.Tensor) -> dict[str, float]:
        return {str(position): float(values[position]) for position in probe_positions}

    report = {
        "checkpoint": checkpoint_path,
        "epoch": int(payload.get("epoch", -1)),
        "device": str(device),
        "images": int(total_images),
        "rollout_steps": int(rollout_tokens),
        "max_cache_error": float(total_cache_error),
        "teacher_ce": total_teacher_ce / float(total_images * rollout_tokens),
        "condition_only_ce": total_condition_only_ce / float(total_images * rollout_tokens),
        "generated_prefix_ce": total_generated_prefix_ce / float(total_images * rollout_tokens),
        "teacher_accuracy": mean_or_nan(teacher_acc),
        "condition_only_accuracy": mean_or_nan(condition_only_acc),
        "rollout_accuracy": mean_or_nan(rollout_acc),
        "first_token_teacher_accuracy": float(teacher_acc[0]),
        "first_token_rollout_accuracy": float(rollout_acc[0]),
        "prefix_exact_at_16": float(prefix_acc[min(15, rollout_tokens - 1)]),
        "prefix_exact_at_64": float(prefix_acc[min(63, rollout_tokens - 1)]),
        "prefix_exact_at_128": float(prefix_acc[min(127, rollout_tokens - 1)]),
        "prefix_exact_at_256": float(prefix_acc[min(255, rollout_tokens - 1)]) if rollout_tokens == full_tokens else None,
        "conditional_accuracy_bins": bins(conditional_acc),
        "conditional_accuracy_positions": probes(conditional_acc),
        "teacher_accuracy_bins": bins(teacher_acc),
        "condition_only_accuracy_bins": bins(condition_only_acc),
        "teacher_accuracy_positions": probes(teacher_acc),
        "condition_only_accuracy_positions": probes(condition_only_acc),
        "rollout_accuracy_bins": bins(rollout_acc),
        "rollout_accuracy_positions": probes(rollout_acc),
        "prefix_exact_bins": bins(prefix_acc),
        "prefix_exact_positions": probes(prefix_acc),
        "target_mode_accuracy_mean": mean_or_nan(mode_acc),
        "target_mode_accuracy_positions": probes(mode_acc),
        "target_entropy_bits_mean": mean_or_nan(target_entropy),
        "target_entropy_bits_positions": probes(target_entropy),
        "target_global_entropy_bits": target_global_entropy,
        "target_global_unique_tokens": int((target_global_counts > 0).sum().item()),
        "generated_entropy_bits": generated_entropy,
        "generated_unique_tokens": int((generated_counts > 0).sum().item()),
        "image_metrics": ({name: value / float(total_images) for name, value in image_metrics.items()} if rollout_tokens == full_tokens else None),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    run(parse_args())
