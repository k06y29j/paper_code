#!/usr/bin/env python3
"""Inspect a trained Stage-3 FSQ tokenizer on the DIV2K validation split.

This is intentionally an exploration-side tool: it reuses the canonical
``train-stage3-fsq.py`` implementation without changing its checkpoints.
Besides the normal decoder result it measures whether the frozen Layer-2
combiner can safely fall back to ``combiner(x1, x1)`` and whether the E3
features actually carry spatial/token variation before FSQ.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
CDDM_ROOT = JSCCF_DIR.parents[1]
for path in (CDDM_ROOT, JSCCF_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Autoencoder.data.datasets import get_loader
from common import batch_metric_mean, psnr_per_image, resolve_path


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fsq = load_module("jsccf_stage3_fsq_explore_base", JSCCF_DIR / "train-stage3-fsq.py")


def mean_and_std(x: torch.Tensor) -> tuple[float, float]:
    x = x.detach().float()
    return float(x.mean().item()), float(x.std(unbiased=False).item())


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict[str, float]:
    checkpoint_path = Path(resolve_path(args.tokenizer_ckpt))
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("stage") != "stage3_fsq_tokenizer_u2":
        raise ValueError(f"not a Stage-3 FSQ tokenizer: {checkpoint_path}")

    run_args = SimpleNamespace(**payload["args"])
    run_args.cpu = bool(args.cpu)
    run_args.max_val_batches = int(args.max_val_batches)
    # This probe is often run from the restricted workspace sandbox, where
    # PyTorch worker shared-memory handles are unavailable.  It is validation
    # only, so a serial loader is both deterministic and sufficient.
    run_args.num_workers = 0
    run_args.val_num_workers = 0
    teacher_ckpt = fsq.load_teacher_checkpoint_for_args(run_args)
    device = torch.device("cuda:0" if (not run_args.cpu) and torch.cuda.is_available() else "cpu")
    cfg = fsq.jsccf_io.build_config(run_args, encoder_in_chans=3)
    _train_loader, val_loader = get_loader(cfg)
    teacher = fsq.build_teacher(run_args, teacher_ckpt, device)
    tokenizer = fsq.Layer3FSQTokenizer(run_args, device)
    if not bool(run_args.no_pre_norm):
        normalizer = str(getattr(run_args, "fsq_normalizer", "group"))
        if normalizer == "batch":
            tokenizer.quantizer.pre_norm = nn.BatchNorm2d(
                int(run_args.fsq_d), affine=True, track_running_stats=True
            ).to(device)
        elif normalizer == "batch_stateless":
            tokenizer.quantizer.pre_norm = nn.BatchNorm2d(
                int(run_args.fsq_d), affine=True, track_running_stats=False
            ).to(device)
        elif normalizer == "instance":
            tokenizer.quantizer.pre_norm = nn.InstanceNorm2d(
                int(run_args.fsq_d), affine=True, track_running_stats=False
            ).to(device)
    tokenizer.load_state_dict(payload["tokenizer_state_dict"], strict=True)
    tokenizer.eval()
    teacher.eval()

    totals = {
        "images": 0.0,
        "psnr_final": 0.0,
        "psnr_x1": 0.0,
        "psnr_x1_x1": 0.0,
        "psnr_zero": 0.0,
        "psnr_shuffle": 0.0,
        "z3_abs_mean": 0.0,
        "z3_spatial_std": 0.0,
        "z3_channel_std": 0.0,
        "z3_norm_spatial_std": 0.0,
        "q3_spatial_std": 0.0,
        "token_unique_per_image": 0.0,
        "token_spatial_change_ratio": 0.0,
    }
    for batch_idx, (imgs, _labels) in enumerate(val_loader, start=1):
        if int(args.max_val_batches) > 0 and batch_idx > int(args.max_val_batches):
            break
        imgs = imgs.to(device, non_blocking=True)
        teacher_out = teacher.forward(imgs)
        out = tokenizer(imgs, teacher_out["x1"], teacher_out["z1"], teacher.combiner)
        zero = tokenizer.decode(torch.zeros_like(out["q3"]), teacher_out["x1"], teacher_out["z1"], teacher.combiner)
        shuffled = tokenizer.decode(tokenizer.shuffle_q3(out["q3"]), teacher_out["x1"], teacher_out["z1"], teacher.combiner)
        x1_x1 = teacher.combiner(teacher_out["x1"], teacher_out["x1"])

        bsz = int(imgs.shape[0])
        z3 = out["z3"].float()
        z3_norm = out["z3_norm"].float()
        q3 = out["q3_hard"].float()
        indices = out["idx3"]
        spatial_std = z3.flatten(2).std(dim=2, unbiased=False).mean()
        channel_std = z3.mean(dim=(2, 3)).std(dim=1, unbiased=False).mean()
        norm_spatial_std = z3_norm.flatten(2).std(dim=2, unbiased=False).mean()
        q_spatial_std = q3.flatten(2).std(dim=2, unbiased=False).mean()
        unique_per_image = torch.tensor(
            [x.unique().numel() for x in indices], device=indices.device, dtype=torch.float32
        ).mean()
        changed = (indices[:, :, 1:] != indices[:, :, :-1]).float().mean()
        changed = 0.5 * (changed + (indices[:, 1:, :] != indices[:, :-1, :]).float().mean())

        values = {
            "psnr_final": batch_metric_mean(psnr_per_image(out["final"], imgs)),
            "psnr_x1": batch_metric_mean(psnr_per_image(teacher_out["x1"], imgs)),
            "psnr_x1_x1": batch_metric_mean(psnr_per_image(x1_x1, imgs)),
            "psnr_zero": batch_metric_mean(psnr_per_image(zero["final"], imgs)),
            "psnr_shuffle": batch_metric_mean(psnr_per_image(shuffled["final"], imgs)),
            "z3_abs_mean": float(z3.abs().mean().item()),
            "z3_spatial_std": float(spatial_std.item()),
            "z3_channel_std": float(channel_std.item()),
            "z3_norm_spatial_std": float(norm_spatial_std.item()),
            "q3_spatial_std": float(q_spatial_std.item()),
            "token_unique_per_image": float(unique_per_image.item()),
            "token_spatial_change_ratio": float(changed.item()),
        }
        totals["images"] += bsz
        for key, value in values.items():
            totals[key] += bsz * value

    if totals["images"] == 0:
        raise RuntimeError("validation loader produced no images")
    metrics = {key: value / totals["images"] for key, value in totals.items() if key != "images"}
    metrics["images"] = totals["images"]
    metrics["drop_zero"] = metrics["psnr_final"] - metrics["psnr_zero"]
    metrics["drop_shuffle"] = metrics["psnr_final"] - metrics["psnr_shuffle"]
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tokenizer-ckpt", required=True)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = analyze(args)
    print(json.dumps(metrics, sort_keys=True, indent=2), flush=True)
    if args.output:
        output = Path(resolve_path(args.output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
