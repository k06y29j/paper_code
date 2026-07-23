#!/usr/bin/env python3
"""Compare candidate FSQ raster orders before retraining the AR decoder.

The experiment uses the frozen direct-FSQ tokenizer to measure target-sequence
statistics only.  A lower empirical H(token_t | token_{t-1}), higher Markov
top-1 accuracy, and smaller adjacent FSQ-code distance indicate a sequence
that should be easier for a causal decoder.  No AR checkpoint is reused for a
different order: changing order requires retraining the AR model afterwards.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
if str(JSCCF_DIR) not in sys.path:
    sys.path.insert(0, str(JSCCF_DIR))


def load_ar():
    path = JSCCF_DIR / "train_stage-fsq-ar.py"
    spec = importlib.util.spec_from_file_location("jsccf_scan_order_ar", path)
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
            "jscc_f_ifsq-prefix-ar-k125_stage_ifsq_ar_fsq_l5x5x5_best.pth"
        ),
    )
    parser.add_argument("--max-val-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def raster(h: int, w: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(h) for col in range(w)]


def column(h: int, w: int) -> list[tuple[int, int]]:
    return [(row, col) for col in range(w) for row in range(h)]


def snake(h: int, w: int) -> list[tuple[int, int]]:
    return [(row, col if row % 2 == 0 else w - 1 - col) for row in range(h) for col in range(w)]


def column_snake(h: int, w: int) -> list[tuple[int, int]]:
    return [(row if col % 2 == 0 else h - 1 - row, col) for col in range(w) for row in range(h)]


def morton(h: int, w: int) -> list[tuple[int, int]]:
    def code(row: int, col: int) -> int:
        value = 0
        for bit in range(max(h, w).bit_length()):
            value |= ((row >> bit) & 1) << (2 * bit)
            value |= ((col >> bit) & 1) << (2 * bit + 1)
        return value

    return sorted(((row, col) for row in range(h) for col in range(w)), key=lambda item: code(*item))


def hilbert_xy(order: int, distance: int) -> tuple[int, int]:
    """Wikipedia d2xy Hilbert mapping for a 2**order square."""
    x = 0
    y = 0
    value = int(distance)
    size = 1
    while size < (1 << int(order)):
        rx = 1 & (value // 2)
        ry = 1 & (value ^ rx)
        if ry == 0:
            if rx == 1:
                x = size - 1 - x
                y = size - 1 - y
            x, y = y, x
        x += size * rx
        y += size * ry
        value //= 4
        size *= 2
    return y, x


def hilbert(h: int, w: int) -> list[tuple[int, int]]:
    if h != w or h & (h - 1):
        raise ValueError("Hilbert order requires a square power-of-two grid")
    order = int(math.log2(h))
    return [hilbert_xy(order, distance) for distance in range(h * w)]


def diagonal(h: int, w: int) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for diagonal_index in range(h + w - 1):
        points = [(row, diagonal_index - row) for row in range(h) if 0 <= diagonal_index - row < w]
        if diagonal_index % 2:
            points.reverse()
        output.extend(points)
    return output


def orders(h: int, w: int) -> dict[str, list[tuple[int, int]]]:
    row = raster(h, w)
    row_snake = snake(h, w)
    return {
        "raster": row,
        "column": column(h, w),
        "snake": row_snake,
        "column_snake": column_snake(h, w),
        "reverse_raster": list(reversed(row)),
        "reverse_snake": list(reversed(row_snake)),
        "morton": morton(h, w),
        "hilbert": hilbert(h, w),
        "diagonal": diagonal(h, w),
    }


def entropy_from_counts(counts: torch.Tensor) -> torch.Tensor:
    probability = counts.float() / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    terms = probability * probability.clamp_min(1e-12).log2()
    return -terms.masked_fill(probability <= 0, 0.0).sum(dim=-1)


def sequence_stats(sequences: torch.Tensor, levels: list[int]) -> dict[str, object]:
    """Return entropy/Markov/locality statistics for [N,T] sequences."""
    samples, length = sequences.shape
    vocabulary = int(math.prod(levels))
    counts = torch.zeros(vocabulary, vocabulary, dtype=torch.long)
    previous = sequences[:, :-1].reshape(-1).long()
    following = sequences[:, 1:].reshape(-1).long()
    counts.index_put_((previous, following), torch.ones_like(previous), accumulate=True)
    row_total = counts.sum(dim=1)
    transition_entropy = entropy_from_counts(counts)
    conditional_entropy = float((transition_entropy * row_total.float()).sum().item() / max(1, row_total.sum().item()))
    markov_accuracy = float(counts.max(dim=1).values.sum().item() / max(1, row_total.sum().item()))
    marginal = torch.bincount(sequences.reshape(-1), minlength=vocabulary)
    global_entropy = float(entropy_from_counts(marginal.view(1, -1))[0].item())
    first_counts = torch.bincount(sequences[:, 0], minlength=vocabulary)
    first_entropy = float(entropy_from_counts(first_counts.view(1, -1))[0].item())
    first_mode = float(first_counts.max().item() / max(1, samples))

    level_values = []
    for level in levels:
        level_values.append(torch.arange(int(level), dtype=torch.float32) / float(int(level) - 1) * 2.0 - 1.0)
    multipliers = []
    for dimension in range(len(levels)):
        multiplier = 1
        for following_level in levels[dimension + 1 :]:
            multiplier *= int(following_level)
        multipliers.append(multiplier)
    codes = torch.stack(
        [((sequences // int(multiplier)) % int(level)).float() for multiplier, level in zip(multipliers, levels)], dim=-1
    )
    adjacent_code_l1 = float((codes[:, 1:] - codes[:, :-1]).abs().mean().item())
    adjacent_same = float(sequences[:, 1:].eq(sequences[:, :-1]).float().mean().item())
    return {
        "samples": int(samples),
        "tokens": int(length),
        "start_coord": None,
        "first_mode_accuracy": first_mode,
        "first_entropy_bits": first_entropy,
        "global_entropy_bits": global_entropy,
        "transition_conditional_entropy_bits": conditional_entropy,
        "transition_markov_top1_accuracy": markov_accuracy,
        "adjacent_same_token_ratio": adjacent_same,
        "adjacent_fsq_code_l1": adjacent_code_l1,
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    ar = load_ar()
    checkpoint_path = ar.resolved(args.ar_checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**payload["args"])
    saved.cpu = bool(args.cpu)
    device = torch.device("cpu" if args.cpu else "cuda:0" if torch.cuda.is_available() else "cpu")
    system, _layer2_payload, layer2_args, _layer2_path = ar.load_frozen_system(saved, device)

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

    candidate_orders = orders(system.height, system.width)
    collected: list[torch.Tensor] = []
    for batch_index, (images, _labels) in enumerate(val_loader, start=1):
        if int(args.max_val_batches) > 0 and batch_index > int(args.max_val_batches):
            break
        target = system.oracle_targets(images.to(device, non_blocking=True))
        collected.append(target["indices"].detach().cpu())
    if not collected:
        raise RuntimeError("diagnostic loader produced no images")
    index_maps = torch.cat(collected, dim=0)
    levels = [int(value) for value in system.levels.cpu().tolist()]
    report = {
        "checkpoint": checkpoint_path,
        "epoch": int(payload.get("epoch", -1)),
        "device": str(device),
        "images": int(index_maps.shape[0]),
        "grid": [int(system.height), int(system.width)],
        "levels": levels,
        "orders": {},
    }
    for name, coordinates in candidate_orders.items():
        row = torch.tensor([coord[0] for coord in coordinates], dtype=torch.long)
        col = torch.tensor([coord[1] for coord in coordinates], dtype=torch.long)
        sequence = index_maps[:, row, col]
        stats = sequence_stats(sequence, levels)
        stats["start_coord"] = [int(coordinates[0][0]), int(coordinates[0][1])]
        report["orders"][name] = stats
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    run(parse_args())
