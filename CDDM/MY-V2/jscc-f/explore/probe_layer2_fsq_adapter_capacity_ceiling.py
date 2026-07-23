#!/usr/bin/env python3
"""Empirical capacity-ceiling probes for adapter and bitplane checkpoints.

Supported checkpoint stages are ``layer2_fsq_adapter``,
``layer2_fsq_adapter_multirate``, and ``layer2_fsq_bitplane_multirate``.
Every optimization in this script is a finite numerical search with a frozen
trained decoder.  Consequently its result is only a *reachable estimate* (a
lower bound on the corresponding optimum), never a mathematical upper bound.

For shared bitplane checkpoints, the reachable hard envelope is constructed
in increasing-rate order.  The best lower-rate code, embedded by appending
inactive ``-1`` bits, is retained as a candidate at the next rate.  The actual
encoder code is retained at every rate as well.  This makes the reported hard
envelope non-decreasing per validation image by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_layer2_fsq_adapter as adapter  # noqa: E402
import train_layer2_fsq_adapter_multirate as adapter_multirate  # noqa: E402
import train_layer2_fsq_bitplane_multirate as bitplane  # noqa: E402
import train_layer2_fsq_direct as direct  # noqa: E402


base = direct.base
SUPPORTED_STAGES = {
    "layer2_fsq_adapter",
    "layer2_fsq_adapter_multirate",
    "layer2_fsq_bitplane_multirate",
}

DEFINITIONS = {
    "actual": "encoder-selected hard code evaluated by the frozen trained decoder",
    "hard_oracle_search": (
        "best retained same-grid candidate from actual-initialized STE search; shared nested routes "
        "also retain the embedded lower-rate best code; reachable search estimate only, not a "
        "mathematical upper bound"
    ),
    "continuous_latent_relax": (
        "rounding removed with the same active latent dimensions and frozen decoder; inactive "
        "bitplane dimensions remain -1; shared scalar routes search from every rate hard-best and "
        "report one common best; finite numerical reachable estimate, not the relaxed optimum"
    ),
    "arbitrary_u2_relax": (
        "full decoder-output image optimized through the frozen combiner, removing latent/rate/decoder "
        "constraints; finite numerical reachable estimate, not the relaxed optimum"
    ),
    "gain": "mean candidate PSNR minus mean Layer1 PSNR on the selected validation images",
    "hard_search_gap": "gain_hard_oracle_search minus gain_actual",
    "drop_zero": "mean actual PSNR minus mean all-zero/off-code PSNR on the same selected images",
    "drop_shuffle": "mean actual PSNR minus mean spatially shuffled-code PSNR on the same selected images",
    "effective": "gain_actual>0 and drop_zero>=0.1 dB and drop_shuffle>=0.1 dB",
    "ceiling_utilization": (
        "signed U=A/R where A=gain_actual and R=gain_hard_oracle_search; null when R<=0"
    ),
    "ceiling_utilization_positive": "max(A,0)/R; null when R<=0",
    "optimization_scope": (
        "all optimized values are reachable lower bounds on their respective best attainable values"
    ),
    "cross_rate_strict": (
        "strict means the higher-rate PSNR/gain is greater, not merely equal; per-image ratios require "
        "all adjacent rates to be strict unless a transition-specific ratio is named"
    ),
}


def parse_indices(value: str) -> list[int]:
    indices = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not indices or any(index < 0 for index in indices):
        raise ValueError("--val-indices must contain non-negative indices")
    if len(indices) != len(set(indices)):
        raise ValueError("--val-indices contains duplicates")
    return sorted(indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="adapter or shared-bitplane checkpoint; may be repeated",
    )
    parser.add_argument("--val-indices", default=",".join(str(index) for index in range(0, 100, 5)))
    parser.add_argument("--hard-steps", type=int, default=100)
    parser.add_argument("--hard-lr", type=float, default=0.08)
    parser.add_argument("--continuous-steps", type=int, default=100)
    parser.add_argument("--continuous-lr", type=float, default=0.05)
    parser.add_argument("--u2-steps", type=int, default=150)
    parser.add_argument("--u2-lr", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument(
        "--bn-calibration-batches",
        type=int,
        default=-1,
        help="negative keeps the checkpoint setting; zero disables recalibration",
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--cpu", action="store_true", help="run the probe on CPU")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    parse_indices(args.val_indices)
    if int(args.num_workers) < 0 or int(args.val_num_workers) < 0:
        raise ValueError("worker counts must be non-negative")
    for name in ("hard_steps", "continuous_steps", "u2_steps"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("hard_lr", "continuous_lr", "u2_lr"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def build_checkpoint(checkpoint_path: str, cli: argparse.Namespace) -> dict:
    resolved = Path(base.resolve_path(checkpoint_path))
    payload = torch.load(resolved, map_location="cpu")
    stage = str(payload.get("stage", ""))
    if stage not in SUPPORTED_STAGES:
        raise ValueError(
            f"unsupported checkpoint stage {stage!r} for {resolved}; expected one of {sorted(SUPPORTED_STAGES)}"
        )
    if not isinstance(payload.get("args"), dict):
        raise ValueError(f"checkpoint has no serialized args dictionary: {resolved}")

    saved_args = dict(payload["args"])
    # Checkpoints created before the residual-combiner experiment did not save
    # this newly introduced option; their exact historical behavior is original.
    saved_args.setdefault("adapter_combiner", "original")
    args = argparse.Namespace(**saved_args)
    args.cpu = bool(cli.cpu)
    args.test_batch = 1
    args.num_workers = int(cli.num_workers)
    args.val_num_workers = int(cli.val_num_workers)
    args.max_val_batches = 0
    args.seed = int(cli.seed)
    if int(cli.bn_calibration_batches) >= 0:
        args.bn_calibration_batches = int(cli.bn_calibration_batches)
    args._usage_weight = 0.0

    rates: list[int] = []
    levels: list[int] | None = None
    nested_levels: list[int] = []
    if stage == "layer2_fsq_bitplane_multirate":
        rates = [int(value) for value in payload.get("rate_bits", [])]
        if not rates:
            rates = bitplane.parse_rate_bits(args.rate_bits)
        if rates != sorted(set(rates)) or any(rate <= 0 for rate in rates):
            raise ValueError(f"invalid checkpoint bitplane rates: {rates}")
        args.fsq_d = int(rates[-1])
        args.fsq_levels = [2] * int(rates[-1])
    elif stage == "layer2_fsq_adapter_multirate":
        nested_levels = [int(value) for value in payload.get("nested_scalar_levels", [])]
        if not nested_levels:
            nested_levels = adapter_multirate.multirate.parse_nested_levels(
                args.nested_levels, int(args.fsq_d)
            )
        if nested_levels != sorted(set(nested_levels)) or any(level < 2 for level in nested_levels):
            raise ValueError(f"invalid checkpoint nested scalar levels: {nested_levels}")
        args.fsq_levels = adapter_multirate.multirate.repeated_levels(
            nested_levels[-1], int(args.fsq_d)
        )
    else:
        levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))

    direct.explore.ExploreIFSQQuantizer.config = args
    base.seed_everything(int(args.seed))
    source = base.load_teacher_checkpoint_for_args(args)
    cfg = base.jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(cfg)
    bundle = adapter.build_bundle(args, source, cfg.device)
    base.jsccf_io.load_state(bundle.e1, payload["e1_state_dict"], "ceiling_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, payload["d1_state_dict"], "ceiling_D1", strict=True)
    base.jsccf_io.load_state(
        bundle.tokenizer, payload["tokenizer_state_dict"], "ceiling_adapter", strict=True
    )
    base.jsccf_io.load_state(
        bundle.combiner, payload["combiner_state_dict"], "ceiling_combiner", strict=True
    )
    calibration = adapter.calibrate_batch_norm(train_loader, bundle, args, cfg.device)
    for module in (bundle.e1, bundle.d1, bundle.tokenizer, bundle.combiner):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return {
        "path": resolved,
        "payload": payload,
        "stage": stage,
        "args": args,
        "rates": rates,
        "levels": levels,
        "nested_levels": nested_levels,
        "bundle": bundle,
        "val_loader": val_loader,
        "device": cfg.device,
        "calibration": calibration,
    }


def fsq_project(latent: torch.Tensor, levels: list[int], *, ste: bool) -> torch.Tensor:
    """Project an adapter latent onto its exact Cartesian FSQ grid."""
    if latent.ndim != 4 or int(latent.shape[1]) != len(levels):
        raise ValueError(
            f"FSQ projection expected [B,{len(levels)},H,W], got {tuple(latent.shape)}"
        )
    level_tensor = latent.new_tensor(levels).view(1, -1, 1, 1)
    span = (level_tensor - 1.0).clamp_min(1.0)
    bounded = latent.clamp(-1.0, 1.0)
    position = (bounded + 1.0) * 0.5 * span
    rounded = position.round().clamp_min(0.0).minimum(span)
    hard = rounded / span * 2.0 - 1.0
    if ste:
        return bounded + (hard - bounded).detach()
    return hard


def bitplane_project(
    active: torch.Tensor,
    bits: int,
    max_bits: int,
    *,
    hard: bool,
    ste: bool = False,
) -> torch.Tensor:
    """Build a full bipolar code with inactive suffix dimensions fixed at -1."""
    if int(active.shape[1]) != int(bits):
        raise ValueError(f"active tensor has {active.shape[1]} channels, expected {bits}")
    bounded = active.clamp(-1.0, 1.0)
    if hard:
        binary = ((bounded + 1.0) * 0.5).round().clamp(0.0, 1.0) * 2.0 - 1.0
        active_value = bounded + (binary - bounded).detach() if ste else binary
    else:
        active_value = bounded
    full = active.new_full(
        (active.shape[0], int(max_bits), active.shape[2], active.shape[3]), -1.0
    )
    full[:, : int(bits)] = active_value
    return full


@torch.no_grad()
def psnr(final: torch.Tensor, target: torch.Tensor) -> float:
    return float(base.psnr_per_image(final, target).mean().item())


def optimize_hard_code(
    decode: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    target: torch.Tensor,
    initial_variable: torch.Tensor,
    hard_project: Callable[[torch.Tensor, bool], torch.Tensor],
    steps: int,
    lr: float,
    retained_candidates: list[torch.Tensor] | None = None,
) -> tuple[float, torch.Tensor]:
    """Actual-initialized STE search that never discards a retained candidate."""
    variable = initial_variable.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([variable], lr=float(lr))
    with torch.no_grad():
        best_q = hard_project(variable, False).detach().clone()
        best = psnr(decode(best_q)["final"], target)
        for candidate in retained_candidates or []:
            candidate = candidate.detach()
            score = psnr(decode(candidate)["final"], target)
            if score > best:
                best = score
                best_q = candidate.clone()
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        q = hard_project(variable, True)
        base.recon_loss(decode(q)["final"], target).backward()
        optimizer.step()
        with torch.no_grad():
            variable.clamp_(-1.0, 1.0)
            candidate = hard_project(variable, False)
            score = psnr(decode(candidate)["final"], target)
            if score > best:
                best = score
                best_q = candidate.detach().clone()
    return best, best_q


def optimize_continuous_latent(
    decode: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    target: torch.Tensor,
    initial_variable: torch.Tensor,
    continuous_project: Callable[[torch.Tensor], torch.Tensor],
    steps: int,
    lr: float,
) -> tuple[float, torch.Tensor]:
    variable = initial_variable.detach().clone().clamp(-1.0, 1.0).requires_grad_(True)
    optimizer = torch.optim.Adam([variable], lr=float(lr))
    with torch.no_grad():
        best_q = continuous_project(variable).detach().clone()
        best = psnr(decode(best_q)["final"], target)
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        base.recon_loss(decode(continuous_project(variable))["final"], target).backward()
        optimizer.step()
        with torch.no_grad():
            variable.clamp_(-1.0, 1.0)
            candidate = continuous_project(variable)
            score = psnr(decode(candidate)["final"], target)
            if score > best:
                best = score
                best_q = candidate.detach().clone()
    return best, best_q


def optimize_u2(
    combiner: torch.nn.Module,
    x1: torch.Tensor,
    target: torch.Tensor,
    initial: torch.Tensor,
    steps: int,
    lr: float,
) -> float:
    variable = initial.detach().clone().clamp(0.0, 1.0).requires_grad_(True)
    optimizer = torch.optim.Adam([variable], lr=float(lr))
    with torch.no_grad():
        best = psnr(combiner(x1, variable), target)
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        base.recon_loss(combiner(x1, variable), target).backward()
        optimizer.step()
        with torch.no_grad():
            variable.clamp_(0.0, 1.0)
            best = max(best, psnr(combiner(x1, variable), target))
    return best


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / float(len(values))


def aggregate(records: list[dict], metadata: dict) -> dict:
    x1_mean = mean([float(row["psnr_x1"]) for row in records])
    result = {
        **metadata,
        "sample_count": len(records),
        "psnr_x1": x1_mean,
        "psnr_actual": mean([float(row["psnr_actual"]) for row in records]),
        "psnr_hard_oracle_search": mean(
            [float(row["psnr_hard_oracle_search"]) for row in records]
        ),
        "psnr_continuous_latent_relax": mean(
            [float(row["psnr_continuous_latent_relax"]) for row in records]
        ),
        "psnr_arbitrary_u2_relax": mean(
            [float(row["psnr_arbitrary_u2_relax"]) for row in records]
        ),
        "psnr_zero": mean([float(row["psnr_zero"]) for row in records]),
        "psnr_shuffle": mean([float(row["psnr_shuffle"]) for row in records]),
        "per_image": records,
    }
    for key in ("actual", "hard_oracle_search", "continuous_latent_relax", "arbitrary_u2_relax"):
        result[f"gain_{key}"] = float(result[f"psnr_{key}"]) - x1_mean
    actual_gain = float(result["gain_actual"])
    reachable_gain = float(result["gain_hard_oracle_search"])
    result["hard_search_gap"] = reachable_gain - actual_gain
    result["drop_zero"] = float(result["psnr_actual"]) - float(result["psnr_zero"])
    result["drop_shuffle"] = float(result["psnr_actual"]) - float(result["psnr_shuffle"])
    result["effective_threshold_db"] = 0.1
    result["effective"] = bool(
        actual_gain > 0.0
        and float(result["drop_zero"]) >= 0.1
        and float(result["drop_shuffle"]) >= 0.1
    )
    if reachable_gain <= 0.0:
        result["ceiling_utilization"] = None
        result["ceiling_utilization_positive"] = None
        result["ceiling_utilization_status"] = "no_positive_reachable_gain"
    else:
        result["ceiling_utilization"] = actual_gain / reachable_gain
        result["ceiling_utilization_positive"] = max(actual_gain, 0.0) / reachable_gain
        result["ceiling_utilization_status"] = "positive_reachable_gain"
    return result


def evaluate_adapter(context: dict, cli: argparse.Namespace, selected: set[int]) -> list[dict]:
    bundle = context["bundle"]
    levels = list(context["levels"])
    records: list[dict] = []
    sample_index = 0
    for imgs, _labels in context["val_loader"]:
        batch_size = int(imgs.shape[0])
        for offset in range(batch_size):
            index = sample_index + offset
            if index not in selected:
                continue
            target = imgs[offset : offset + 1].to(context["device"], non_blocking=True)
            with torch.no_grad():
                layer1, out = direct.forward_direct(bundle, target)
                x1, z1 = layer1["x1"], layer1["z1"]
                x1_score = psnr(x1, target)
                actual = psnr(out["final"], target)

            def decode(q: torch.Tensor) -> dict[str, torch.Tensor]:
                return bundle.tokenizer.decode(q, x1, z1, bundle.combiner)

            with torch.no_grad():
                zero_score = psnr(decode(torch.zeros_like(out["q3"]))["final"], target)
                shuffled_q = bundle.tokenizer.shuffle_q3(out["q3"])
                shuffle_score = psnr(decode(shuffled_q)["final"], target)
            hard_score, hard_q = optimize_hard_code(
                decode,
                target,
                out["q3"],
                lambda value, ste: fsq_project(value, levels, ste=ste),
                int(cli.hard_steps),
                float(cli.hard_lr),
            )
            continuous_score, continuous_q = optimize_continuous_latent(
                decode,
                target,
                hard_q,
                lambda value: value.clamp(-1.0, 1.0),
                int(cli.continuous_steps),
                float(cli.continuous_lr),
            )
            with torch.no_grad():
                continuous_u2 = decode(continuous_q)["u2_hat"]
            u2_score = optimize_u2(
                bundle.combiner,
                x1,
                target,
                continuous_u2,
                int(cli.u2_steps),
                float(cli.u2_lr),
            )
            if continuous_score < hard_score:
                if continuous_score + 1e-5 < hard_score:
                    raise RuntimeError("reachable adapter continuous relaxation decreased unexpectedly")
                continuous_score = hard_score
            if u2_score < continuous_score:
                if u2_score + 1e-5 < continuous_score:
                    raise RuntimeError("reachable adapter u2 relaxation decreased unexpectedly")
                u2_score = continuous_score
            records.append(
                {
                    "index": index,
                    "psnr_x1": x1_score,
                    "psnr_actual": actual,
                    "psnr_hard_oracle_search": hard_score,
                    "psnr_continuous_latent_relax": continuous_score,
                    "psnr_arbitrary_u2_relax": u2_score,
                    "psnr_zero": zero_score,
                    "psnr_shuffle": shuffle_score,
                    "drop_zero": actual - zero_score,
                    "drop_shuffle": actual - shuffle_score,
                }
            )
            print(
                f"[adapter ceiling] index={index} x1={x1_score:.4f} actual={actual:.4f} "
                f"hard={hard_score:.4f} continuous={continuous_score:.4f} u2={u2_score:.4f}",
                flush=True,
            )
        sample_index += batch_size
        if sample_index > max(selected):
            break
    if len(records) != len(selected):
        raise RuntimeError(f"requested {len(selected)} validation samples but evaluated {len(records)}")
    vocab_size = base.vocab_size(levels)
    metadata = {
        "checkpoint": str(context["path"]),
        "checkpoint_stage": context["stage"],
        "checkpoint_epoch": int(context["payload"].get("epoch", -1)),
        "adapter_combiner": str(context["args"].adapter_combiner),
        "grid_kind": "cartesian_fsq_levels",
        "levels": levels,
        "latent_dimensions": int(context["args"].fsq_d),
        "vocab_size": vocab_size,
        "fixed_bits_per_token": int(math.ceil(math.log2(float(vocab_size)))),
        "val_indices": sorted(selected),
        "bn_calibration": context["calibration"],
    }
    return [aggregate(records, metadata)]


def evaluate_bitplanes(context: dict, cli: argparse.Namespace, selected: set[int]) -> list[dict]:
    bundle = context["bundle"]
    rates = list(context["rates"])
    max_bits = int(rates[-1])
    records_by_rate: dict[int, list[dict]] = {bits: [] for bits in rates}
    sample_index = 0
    for imgs, _labels in context["val_loader"]:
        batch_size = int(imgs.shape[0])
        for offset in range(batch_size):
            index = sample_index + offset
            if index not in selected:
                continue
            target = imgs[offset : offset + 1].to(context["device"], non_blocking=True)
            with torch.no_grad():
                layer1, z_norm, outputs = bitplane.forward_bitplanes(bundle, target, rates)
                x1, z1 = layer1["x1"], layer1["z1"]
                x1_score = psnr(x1, target)
                permutation = torch.randperm(
                    int(z_norm.shape[0] * z_norm.shape[2] * z_norm.shape[3]),
                    device=z_norm.device,
                )
                zero_q = bitplane.zero_branch(z_norm)
                zero_score = psnr(
                    bundle.tokenizer.decode(zero_q, x1, z1, bundle.combiner)["final"], target
                )

            inherited_best_q: torch.Tensor | None = None
            inherited_hard_score: float | None = None
            for bits in rates:
                out = outputs[bits]

                def decode(q: torch.Tensor) -> dict[str, torch.Tensor]:
                    return bundle.tokenizer.decode(q, x1, z1, bundle.combiner)

                actual = psnr(out["final"], target)
                with torch.no_grad():
                    shuffled_q = bitplane.multirate.shuffled_with_perm(out["q3"], permutation)
                    shuffle_score = psnr(decode(shuffled_q)["final"], target)
                retained = [inherited_best_q] if inherited_best_q is not None else []
                hard_score, hard_q = optimize_hard_code(
                    decode,
                    target,
                    out["q3"][:, :bits],
                    lambda value, ste, b=bits: bitplane_project(
                        value, b, max_bits, hard=True, ste=ste
                    ),
                    int(cli.hard_steps),
                    float(cli.hard_lr),
                    retained_candidates=retained,
                )
                if inherited_hard_score is not None and hard_score < inherited_hard_score:
                    if hard_score + 1e-5 < inherited_hard_score:
                        raise RuntimeError("bitplane inherited hard envelope decreased unexpectedly")
                    # The inherited tensor is a valid code at this rate.  Hide
                    # only harmless repeated-forward roundoff, not a real drop.
                    hard_score = inherited_hard_score
                    hard_q = inherited_best_q.detach().clone()
                continuous_score, continuous_q = optimize_continuous_latent(
                    decode,
                    target,
                    hard_q[:, :bits],
                    lambda value, b=bits: bitplane_project(
                        value, b, max_bits, hard=False
                    ),
                    int(cli.continuous_steps),
                    float(cli.continuous_lr),
                )
                with torch.no_grad():
                    continuous_u2 = decode(continuous_q)["u2_hat"]
                u2_score = optimize_u2(
                    bundle.combiner,
                    x1,
                    target,
                    continuous_u2,
                    int(cli.u2_steps),
                    float(cli.u2_lr),
                )
                if continuous_score < hard_score:
                    if continuous_score + 1e-5 < hard_score:
                        raise RuntimeError(
                            "reachable bitplane continuous relaxation decreased unexpectedly"
                        )
                    continuous_score = hard_score
                if u2_score < continuous_score:
                    if u2_score + 1e-5 < continuous_score:
                        raise RuntimeError("reachable bitplane u2 relaxation decreased unexpectedly")
                    u2_score = continuous_score
                records_by_rate[bits].append(
                    {
                        "index": index,
                        "psnr_x1": x1_score,
                        "psnr_actual": actual,
                        "psnr_hard_oracle_search": hard_score,
                        "psnr_continuous_latent_relax": continuous_score,
                        "psnr_arbitrary_u2_relax": u2_score,
                        "psnr_zero": zero_score,
                        "psnr_shuffle": shuffle_score,
                        "drop_zero": actual - zero_score,
                        "drop_shuffle": actual - shuffle_score,
                        "inherited_lower_rate_hard_candidate": inherited_best_q is not None,
                    }
                )
                print(
                    f"[bitplane ceiling b{bits}] index={index} x1={x1_score:.4f} "
                    f"actual={actual:.4f} hard={hard_score:.4f} continuous={continuous_score:.4f} "
                    f"u2={u2_score:.4f}",
                    flush=True,
                )
                inherited_best_q = hard_q.detach().clone()
                inherited_hard_score = hard_score
        sample_index += batch_size
        if sample_index > max(selected):
            break

    results = []
    for bits in rates:
        records = records_by_rate[bits]
        if len(records) != len(selected):
            raise RuntimeError(
                f"rate b{bits}: requested {len(selected)} validation samples but evaluated {len(records)}"
            )
        metadata = {
            "checkpoint": str(context["path"]),
            "checkpoint_stage": context["stage"],
            "checkpoint_epoch": int(context["payload"].get("epoch", -1)),
            "adapter_combiner": str(context["args"].adapter_combiner),
            "grid_kind": "nested_binary_bitplane",
            "rate_bits": bits,
            "max_model_bits": max_bits,
            "active_values": [-1, 1],
            "inactive_suffix_value": -1,
            "inherits_lower_rate_hard_candidate": bits != rates[0],
            "vocab_size": 2**bits,
            "fixed_bits_per_token": bits,
            "val_indices": sorted(selected),
            "bn_calibration": context["calibration"],
        }
        results.append(aggregate(records, metadata))
    return results


def probe_one(checkpoint_path: str, cli: argparse.Namespace) -> list[dict]:
    context = build_checkpoint(checkpoint_path, cli)
    selected = set(parse_indices(cli.val_indices))
    if context["stage"] == "layer2_fsq_adapter":
        return evaluate_adapter(context, cli, selected)
    return evaluate_bitplanes(context, cli, selected)


def cross_rate_summary(rows: list[dict]) -> dict | None:
    if len(rows) < 2 or any(row.get("grid_kind") != "nested_binary_bitplane" for row in rows):
        return None
    ordered = sorted(rows, key=lambda row: int(row["rate_bits"]))
    transitions = []
    actual_all_strict: dict[int, bool] = {
        int(row["index"]): True for row in ordered[0]["per_image"]
    }
    reachable_all_strict = dict(actual_all_strict)
    for lower, upper in zip(ordered, ordered[1:]):
        lower_records = {int(row["index"]): row for row in lower["per_image"]}
        upper_records = {int(row["index"]): row for row in upper["per_image"]}
        if lower_records.keys() != upper_records.keys():
            raise RuntimeError("cross-rate summaries require identical selected validation images")
        actual_flags = []
        reachable_flags = []
        for index in sorted(lower_records):
            actual_strict = float(upper_records[index]["psnr_actual"]) > float(
                lower_records[index]["psnr_actual"]
            )
            reachable_strict = float(
                upper_records[index]["psnr_hard_oracle_search"]
            ) > float(lower_records[index]["psnr_hard_oracle_search"])
            actual_flags.append(actual_strict)
            reachable_flags.append(reachable_strict)
            actual_all_strict[index] = actual_all_strict[index] and actual_strict
            reachable_all_strict[index] = reachable_all_strict[index] and reachable_strict
        transitions.append(
            {
                "lower_rate_bits": int(lower["rate_bits"]),
                "upper_rate_bits": int(upper["rate_bits"]),
                "actual_gain_delta": float(upper["gain_actual"]) - float(lower["gain_actual"]),
                "reachable_gain_delta": float(upper["gain_hard_oracle_search"])
                - float(lower["gain_hard_oracle_search"]),
                "mean_strict_actual": bool(
                    float(upper["gain_actual"]) > float(lower["gain_actual"])
                ),
                "mean_strict_reachable": bool(
                    float(upper["gain_hard_oracle_search"])
                    > float(lower["gain_hard_oracle_search"])
                ),
                "per_image_strict_ratio_actual": sum(actual_flags) / float(len(actual_flags)),
                "per_image_strict_ratio_reachable": sum(reachable_flags)
                / float(len(reachable_flags)),
            }
        )
    return {
        "checkpoint": str(ordered[0]["checkpoint"]),
        "checkpoint_epoch": int(ordered[0]["checkpoint_epoch"]),
        "rates_bits": [int(row["rate_bits"]) for row in ordered],
        "actual_mean_all_adjacent_strict": all(
            bool(transition["mean_strict_actual"]) for transition in transitions
        ),
        "reachable_mean_all_adjacent_strict": all(
            bool(transition["mean_strict_reachable"]) for transition in transitions
        ),
        "per_image_all_rates_strict_ratio_actual": sum(actual_all_strict.values())
        / float(len(actual_all_strict)),
        "per_image_all_rates_strict_ratio_reachable": sum(reachable_all_strict.values())
        / float(len(reachable_all_strict)),
        "transitions": transitions,
    }


def summary_row(row: dict) -> dict:
    keys = (
        "checkpoint",
        "checkpoint_stage",
        "checkpoint_epoch",
        "levels",
        "rate_bits",
        "vocab_size",
        "sample_count",
        "psnr_x1",
        "psnr_actual",
        "psnr_hard_oracle_search",
        "psnr_continuous_latent_relax",
        "psnr_arbitrary_u2_relax",
        "gain_actual",
        "gain_hard_oracle_search",
        "hard_search_gap",
        "drop_zero",
        "drop_shuffle",
        "effective",
        "ceiling_utilization",
        "ceiling_utilization_positive",
        "ceiling_utilization_status",
    )
    return {key: row[key] for key in keys if key in row}


def main() -> None:
    cli = parse_args()
    rows: list[dict] = []
    cross_rate: list[dict] = []
    for checkpoint in cli.checkpoint:
        checkpoint_rows = probe_one(checkpoint, cli)
        rows.extend(checkpoint_rows)
        cross = cross_rate_summary(checkpoint_rows)
        if cross is not None:
            cross_rate.append(cross)
    report = {
        "schema_version": 1,
        "definitions": DEFINITIONS,
        "optimization": {
            "hard_steps": int(cli.hard_steps),
            "hard_lr": float(cli.hard_lr),
            "continuous_steps": int(cli.continuous_steps),
            "continuous_lr": float(cli.continuous_lr),
            "u2_steps": int(cli.u2_steps),
            "u2_lr": float(cli.u2_lr),
        },
        "results": rows,
        "cross_rate": cross_rate,
    }
    print(
        json.dumps(
            {"results": [summary_row(row) for row in rows], "cross_rate": cross_rate}, indent=2
        ),
        flush=True,
    )
    if cli.output_json:
        output = Path(base.resolve_path(cli.output_json))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
