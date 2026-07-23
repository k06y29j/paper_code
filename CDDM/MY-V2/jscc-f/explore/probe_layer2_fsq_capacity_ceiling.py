#!/usr/bin/env python3
"""Empirical ceiling ladder for trained direct Layer2-FSQ checkpoints.

The reported levels deliberately have different meanings:

``actual``
    Encoder-selected hard FSQ codes.
``hard_oracle_search``
    Per-image projected-gradient/STE search over the same finite FSQ grid and
    frozen decoder.  This is a reachable search estimate (a lower bound on the
    best code for that decoder), not a mathematical upper bound.
``continuous_latent_relax``
    Optimize an arbitrary d-dimensional latent in [-1,1], removing rounding.
``arbitrary_u2_relax``
    Optimize the full decoder output image supplied to the combiner, removing
    both the d-dimensional bottleneck and its rate constraint.
``source320_reference``
    The original 320-channel continuous Layer2 checkpoint, only a wide-model
    architecture reference.

This decomposition prevents the 42 dB source reference or a validation oracle
from being mislabeled as the K-specific FSQ limit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_layer2_fsq_direct as direct  # noqa: E402


base = direct.base


def parse_indices(value: str) -> list[int]:
    indices = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not indices or any(index < 0 for index in indices):
        raise ValueError("--val-indices must contain non-negative indices")
    if len(indices) != len(set(indices)):
        raise ValueError("--val-indices contains duplicates")
    return sorted(indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--val-indices", default=",".join(str(index) for index in range(0, 100, 5)))
    parser.add_argument("--hard-steps", type=int, default=100)
    parser.add_argument("--hard-lr", type=float, default=0.08)
    parser.add_argument("--continuous-steps", type=int, default=100)
    parser.add_argument("--continuous-lr", type=float, default=0.05)
    parser.add_argument("--u2-steps", type=int, default=150)
    parser.add_argument("--u2-lr", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--bn-calibration-batches", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    parse_indices(args.val_indices)
    for name in ("hard_steps", "continuous_steps", "u2_steps"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("hard_lr", "continuous_lr", "u2_lr"):
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def build_checkpoint(
    checkpoint_path: str,
    cli: argparse.Namespace,
) -> tuple[dict, argparse.Namespace, direct.DirectBundle, object, object, torch.device, dict]:
    payload = torch.load(base.resolve_path(checkpoint_path), map_location="cpu")
    if str(payload.get("stage", "")) != "layer2_fsq_direct":
        raise ValueError(f"not a direct Layer2-FSQ checkpoint: {checkpoint_path}")
    args = argparse.Namespace(**payload["args"])
    args.cpu = False
    args.test_batch = 1
    args.num_workers = int(cli.num_workers)
    args.val_num_workers = int(cli.val_num_workers)
    args.max_val_batches = 0
    args.seed = int(cli.seed)
    if int(cli.bn_calibration_batches) >= 0:
        args.bn_calibration_batches = int(cli.bn_calibration_batches)
    args._usage_weight = 0.0
    direct.explore.ExploreIFSQQuantizer.config = args
    base.seed_everything(int(args.seed))
    source = base.load_teacher_checkpoint_for_args(args)
    cfg = base.jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(cfg)
    bundle = direct.build_direct_bundle(args, source, cfg.device)
    base.jsccf_io.load_state(bundle.e1, payload["e1_state_dict"], "ceiling_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, payload["d1_state_dict"], "ceiling_D1", strict=True)
    base.jsccf_io.load_state(bundle.tokenizer, payload["tokenizer_state_dict"], "ceiling_codec", strict=True)
    base.jsccf_io.load_state(bundle.combiner, payload["combiner_state_dict"], "ceiling_combiner", strict=True)
    direct.calibrate_fsq_batch_norm(train_loader, bundle, args, cfg.device)
    for module in (bundle.e1, bundle.d1, bundle.tokenizer, bundle.combiner):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return payload, args, bundle, train_loader, val_loader, cfg.device, source


def hard_quantize_ste(latent: torch.Tensor, levels: list[int]) -> torch.Tensor:
    level_tensor = latent.new_tensor(levels).view(1, -1, 1, 1)
    span = (level_tensor - 1.0).clamp_min(1.0)
    bounded = latent.clamp(-1.0, 1.0)
    position = (bounded + 1.0) * 0.5 * span
    code = base.round_ste(position).clamp_min(0.0).minimum(span)
    hard = code / span * 2.0 - 1.0
    return bounded + (hard - bounded).detach()


def hard_quantize(latent: torch.Tensor, levels: list[int]) -> torch.Tensor:
    level_tensor = latent.new_tensor(levels).view(1, -1, 1, 1)
    span = (level_tensor - 1.0).clamp_min(1.0)
    code = (((latent.clamp(-1.0, 1.0) + 1.0) * 0.5 * span).round()).clamp_min(0.0).minimum(span)
    return code / span * 2.0 - 1.0


@torch.no_grad()
def psnr(final: torch.Tensor, target: torch.Tensor) -> float:
    return float(base.psnr_per_image(final, target).mean().item())


def optimize_hard_code(
    bundle: direct.DirectBundle,
    x1: torch.Tensor,
    z1: torch.Tensor,
    target: torch.Tensor,
    initial: torch.Tensor,
    levels: list[int],
    steps: int,
    lr: float,
) -> tuple[float, torch.Tensor]:
    variable = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([variable], lr=float(lr))
    with torch.no_grad():
        best_q = hard_quantize(variable, levels)
        best = psnr(bundle.tokenizer.decode(best_q, x1, z1, bundle.combiner)["final"], target)
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        q = hard_quantize_ste(variable, levels)
        final = bundle.tokenizer.decode(q, x1, z1, bundle.combiner)["final"]
        loss = base.recon_loss(final, target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            variable.clamp_(-1.0, 1.0)
            candidate = hard_quantize(variable, levels)
            score = psnr(bundle.tokenizer.decode(candidate, x1, z1, bundle.combiner)["final"], target)
            if score > best:
                best = score
                best_q = candidate.detach().clone()
    return best, best_q


def optimize_continuous_latent(
    bundle: direct.DirectBundle,
    x1: torch.Tensor,
    z1: torch.Tensor,
    target: torch.Tensor,
    initial: torch.Tensor,
    steps: int,
    lr: float,
) -> float:
    variable = initial.detach().clone().clamp(-1.0, 1.0).requires_grad_(True)
    optimizer = torch.optim.Adam([variable], lr=float(lr))
    with torch.no_grad():
        best = psnr(bundle.tokenizer.decode(variable, x1, z1, bundle.combiner)["final"], target)
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        final = bundle.tokenizer.decode(variable, x1, z1, bundle.combiner)["final"]
        base.recon_loss(final, target).backward()
        optimizer.step()
        with torch.no_grad():
            variable.clamp_(-1.0, 1.0)
            best = max(best, psnr(bundle.tokenizer.decode(variable, x1, z1, bundle.combiner)["final"], target))
    return best


def optimize_u2(
    bundle: direct.DirectBundle,
    x1: torch.Tensor,
    target: torch.Tensor,
    initial: torch.Tensor,
    steps: int,
    lr: float,
) -> float:
    variable = initial.detach().clone().clamp(0.0, 1.0).requires_grad_(True)
    optimizer = torch.optim.Adam([variable], lr=float(lr))
    with torch.no_grad():
        best = psnr(bundle.combiner(x1, variable), target)
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        final = bundle.combiner(x1, variable)
        base.recon_loss(final, target).backward()
        optimizer.step()
        with torch.no_grad():
            variable.clamp_(0.0, 1.0)
            best = max(best, psnr(bundle.combiner(x1, variable), target))
    return best


def mean(values: list[float]) -> float:
    return sum(values) / float(max(1, len(values)))


def probe_one(checkpoint_path: str, cli: argparse.Namespace) -> dict:
    payload, args, bundle, _train, val_loader, device, source = build_checkpoint(checkpoint_path, cli)
    selected = set(parse_indices(cli.val_indices))
    levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    records: list[dict[str, float | int]] = []
    sample_index = 0
    for imgs, _labels in val_loader:
        batch_size = int(imgs.shape[0])
        for offset in range(batch_size):
            index = sample_index + offset
            if index not in selected:
                continue
            target = imgs[offset : offset + 1].to(device, non_blocking=True)
            with torch.no_grad():
                layer1, out = direct.forward_direct(bundle, target)
                x1 = layer1["x1"]
                z1 = layer1["z1"]
                actual = psnr(out["final"], target)
                x1_score = psnr(x1, target)
            hard_score, hard_q = optimize_hard_code(
                bundle, x1, z1, target, out["q3"], levels,
                int(cli.hard_steps), float(cli.hard_lr),
            )
            continuous_score = optimize_continuous_latent(
                bundle, x1, z1, target, hard_q,
                int(cli.continuous_steps), float(cli.continuous_lr),
            )
            u2_score = optimize_u2(
                bundle, x1, target, out["u2_hat"], int(cli.u2_steps), float(cli.u2_lr)
            )
            records.append(
                {
                    "index": index,
                    "psnr_x1": x1_score,
                    "psnr_actual": actual,
                    "psnr_hard_oracle_search": hard_score,
                    "psnr_continuous_latent_relax": continuous_score,
                    "psnr_arbitrary_u2_relax": u2_score,
                }
            )
        sample_index += batch_size
        if sample_index > max(selected):
            break
    if len(records) != len(selected):
        raise RuntimeError(f"requested {len(selected)} validation samples but evaluated {len(records)}")
    x1_mean = mean([float(row["psnr_x1"]) for row in records])
    result = {
        "checkpoint": str(Path(base.resolve_path(checkpoint_path))),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "levels": levels,
        "vocab_size": base.vocab_size(levels),
        "fixed_bits_per_token": int(math.ceil(math.log2(float(base.vocab_size(levels))))),
        "sample_count": len(records),
        "val_indices": sorted(selected),
        "definitions": {
            "hard_oracle_search": "reachable same-K frozen-decoder search estimate; not a mathematical upper bound",
            "continuous_latent_relax": "rounding removed, frozen d-dimensional decoder",
            "arbitrary_u2_relax": "d/rate/decoder removed, frozen combiner",
            "source320_reference": "wide continuous architecture reference, not same-rate",
        },
        "psnr_x1": x1_mean,
        "psnr_actual": mean([float(row["psnr_actual"]) for row in records]),
        "psnr_hard_oracle_search": mean([float(row["psnr_hard_oracle_search"]) for row in records]),
        "psnr_continuous_latent_relax": mean(
            [float(row["psnr_continuous_latent_relax"]) for row in records]
        ),
        "psnr_arbitrary_u2_relax": mean([float(row["psnr_arbitrary_u2_relax"]) for row in records]),
        "source320_psnr_final": float(source.get("metrics", {}).get("psnr_final", float("nan"))),
        "source320_z2_channels": int(source.get("stage2_codec", {}).get("z2_ch", -1)),
        "per_image": records,
    }
    for key in (
        "actual",
        "hard_oracle_search",
        "continuous_latent_relax",
        "arbitrary_u2_relax",
    ):
        result[f"gain_{key}"] = float(result[f"psnr_{key}"]) - x1_mean
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2), flush=True)
    return result


def main() -> None:
    cli = parse_args()
    rows = [probe_one(path, cli) for path in cli.checkpoint]
    if cli.output_json:
        output = Path(base.resolve_path(cli.output_json))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
