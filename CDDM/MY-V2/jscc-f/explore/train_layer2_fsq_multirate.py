#!/usr/bin/env python3
"""Shared-model nested-rate FSQ directly in JSCC-f Layer2.

This is an opt-in follow-up to :mod:`train_layer2_fsq_direct`.  It keeps one
trainable E2, one pre-FSQ normalizer, one D2, and one combiner, then evaluates
several *nested* scalar grids from the same continuous latent::

    image -> frozen E1/D1 -> x1
    concat(image, x1) -> E2 -> norm/tanh -> z_norm (computed once)
                                      |-> FSQ L=5  -> shared D2/combiner
                                      |-> FSQ L=9  -> shared D2/combiner
                                      `-> FSQ L=17 -> shared D2/combiner

No source Layer2 ``u2`` forward or teacher target is used.  In addition to the
mean reconstruction loss over rates, a per-image monotonic hinge pushes only
the higher-rate branch whenever it fails to beat the detached lower-rate
distortion.  Detaching the lower-rate target prevents the hinge from being
satisfied directly by degrading the lower-rate reconstruction.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_layer2_fsq_direct as direct  # noqa: E402


base = direct.base

OBJECTIVE_METRICS = [
    "loss",
    "loss_recon",
    "loss_monotonic",
    "all_rates_strict_ratio",
]


def parse_nested_levels(value: str, channels: int) -> list[int]:
    levels = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(levels) < 2:
        raise ValueError("--nested-levels requires at least two scalar levels")
    if levels != sorted(set(levels)):
        raise ValueError(f"--nested-levels must be strictly increasing, got {levels}")
    if any(level < 2 for level in levels):
        raise ValueError(f"all nested scalar levels must be >=2, got {levels}")
    for lower, higher in zip(levels, levels[1:]):
        if (higher - 1) % (lower - 1) != 0:
            raise ValueError(
                f"grids are not nested: (higher-1) must be divisible by (lower-1), got {lower}->{higher}"
            )
    if int(channels) <= 0:
        raise ValueError("--fsq-d must be positive")
    return levels


def parse_float_list(value: str, expected: int, label: str) -> list[float]:
    values = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(values) == 1 and expected > 1:
        values = values * expected
    if len(values) != expected:
        raise ValueError(f"{label} needs {expected} values (or one shared value), got {values}")
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"{label} values must be finite, got {values}")
    return values


def repeated_levels(level: int, channels: int) -> list[int]:
    return [int(level)] * int(channels)


def rate_args(args: argparse.Namespace, level: int) -> argparse.Namespace:
    current = copy.copy(args)
    current.fsq_levels = repeated_levels(level, int(args.fsq_d))
    return current


def transition_name(lower: int, higher: int, suffix: str) -> str:
    return f"l{int(lower)}_to_l{int(higher)}_{suffix}"


def objective_metric_names(levels: list[int]) -> list[str]:
    names = list(OBJECTIVE_METRICS)
    for lower, higher in zip(levels, levels[1:]):
        names.extend(
            [
                transition_name(lower, higher, "mse_gain"),
                transition_name(lower, higher, "raw_violation_ratio"),
                transition_name(lower, higher, "margin_violation_ratio"),
            ]
        )
    return names


def quantize_at_level(z_norm: torch.Tensor, level: int) -> dict[str, torch.Tensor]:
    channels = int(z_norm.shape[1])
    span_value = float(int(level) - 1)
    span = z_norm.new_full((1, channels, 1, 1), span_value)
    positions = (z_norm + 1.0) * 0.5 * span
    codes_float = base.round_ste(positions).clamp_min(0.0).minimum(span)
    codes = codes_float.detach().long()
    q_hard = codes_float / span * 2.0 - 1.0
    q = z_norm + (q_hard - z_norm).detach()
    multipliers = torch.tensor(
        [int(level) ** power for power in reversed(range(channels))],
        device=codes.device,
        dtype=torch.long,
    ).view(1, channels, 1, 1)
    indices = (codes * multipliers).sum(dim=1)
    zero = z_norm.new_zeros(())
    return {
        "z3_norm": z_norm,
        "q3": q,
        "q3_hard": q_hard.detach(),
        "codes": codes,
        "idx3": indices,
        "fsq_mse": F.mse_loss(q_hard.detach().float(), z_norm.detach().float()),
        "usage_kl": zero,
        "soft_level_entropy_bits": zero,
        "soft_usage_entropy_bits": zero,
    }


def encode_multirate(
    bundle: direct.DirectBundle,
    imgs: torch.Tensor,
    layer1_out: dict[str, torch.Tensor],
    levels: list[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    e2_in = torch.cat([layer1_out["x1"], imgs], dim=1)
    z3 = base.encode_tensor(bundle.tokenizer.e3, e2_in)
    z_norm = torch.tanh(bundle.tokenizer.quantizer.pre_norm(z3))
    outputs: dict[int, dict[str, torch.Tensor]] = {}
    for level in levels:
        encoded = quantize_at_level(z_norm, level)
        decoded = bundle.tokenizer.decode(
            encoded["q3"],
            layer1_out["x1"],
            layer1_out["z1"],
            bundle.combiner,
        )
        outputs[int(level)] = {**encoded, **decoded, "z3": z3, "q3_used": encoded["q3"]}
    return z3, z_norm, outputs


def forward_multirate(
    bundle: direct.DirectBundle,
    imgs: torch.Tensor,
    levels: list[int],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    with torch.no_grad():
        layer1_out = bundle.layer1(imgs)
    _z3, z_norm, outputs = encode_multirate(bundle, imgs, layer1_out, levels)
    return layer1_out, z_norm, outputs


def compute_multirate_loss(
    outputs: dict[int, dict[str, torch.Tensor]],
    imgs: torch.Tensor,
    levels: list[int],
    recon_weights: list[float],
    margins: list[float],
    lambda_monotonic: float,
) -> tuple[dict[str, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    branch_losses: dict[int, dict[str, torch.Tensor]] = {}
    per_image_mse: dict[int, torch.Tensor] = {}
    weighted = imgs.new_zeros(())
    weight_sum = max(sum(recon_weights), 1e-12)
    for level, weight in zip(levels, recon_weights):
        out = outputs[level]
        loss_final = base.recon_loss(out["final"], imgs)
        loss_u2_img = base.recon_loss(out["u2_raw"], imgs)
        zero = loss_final.new_zeros(())
        branch_losses[level] = {
            "loss": loss_final,
            "loss_final": loss_final,
            "loss_u2_img": loss_u2_img,
            "loss_usage": zero,
        }
        per_image_mse[level] = base.mse_per_image(out["final"], imgs)
        weighted = weighted + float(weight) * loss_final
    loss_recon = weighted / float(weight_sum)

    loss_monotonic = imgs.new_zeros(())
    transition_metrics: dict[str, torch.Tensor] = {}
    strict = torch.ones_like(per_image_mse[levels[0]], dtype=torch.bool)
    for transition, (lower, higher) in enumerate(zip(levels, levels[1:])):
        lower_mse = per_image_mse[lower]
        higher_mse = per_image_mse[higher]
        margin = float(margins[transition])
        hinge = F.relu(higher_mse - lower_mse.detach() + margin)
        loss_monotonic = loss_monotonic + hinge.mean()
        strict = strict & (higher_mse < lower_mse)
        transition_metrics[transition_name(lower, higher, "mse_gain")] = (lower_mse - higher_mse).mean()
        transition_metrics[transition_name(lower, higher, "raw_violation_ratio")] = (
            higher_mse >= lower_mse
        ).float().mean()
        transition_metrics[transition_name(lower, higher, "margin_violation_ratio")] = (
            higher_mse - lower_mse + margin > 0.0
        ).float().mean()

    loss = loss_recon + float(lambda_monotonic) * loss_monotonic
    objective = {
        "loss": loss,
        "loss_recon": loss_recon,
        "loss_monotonic": loss_monotonic,
        "all_rates_strict_ratio": strict.float().mean(),
        **transition_metrics,
    }
    return objective, branch_losses


def make_rate_states(args: argparse.Namespace, levels: list[int], *, validation: bool) -> dict[int, dict]:
    states: dict[int, dict] = {}
    names = direct.METRIC_NAMES + (direct.VAL_ABLATION_METRICS if validation and bool(args.val_ablation) else [])
    for level in levels:
        current_args = rate_args(args, level)
        rate_levels = repeated_levels(level, int(args.fsq_d))
        states[level] = {
            "args": current_args,
            "meters": base.meters(names),
            "hist": torch.zeros(base.vocab_size(rate_levels), dtype=torch.float32),
            "level_hists": direct.make_level_hists(rate_levels),
        }
    return states


def update_rate_state(
    state: dict,
    out: dict[str, torch.Tensor],
    layer1_out: dict[str, torch.Tensor],
    imgs: torch.Tensor,
    losses: dict[str, torch.Tensor],
    bundle: direct.DirectBundle,
) -> None:
    direct.update_metrics(state["meters"], out, layer1_out, imgs, losses, bundle.combiner, state["args"])
    base.update_code_hist(state["hist"], out["idx3"])
    direct.update_level_hists(state["level_hists"], out["codes"])


def update_objective_meters(meters: dict, objective: dict[str, torch.Tensor], batch_size: int) -> None:
    for name, value in objective.items():
        meters[name].update(float(value.detach().item()), int(batch_size))


def shuffled_with_perm(q: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = q.shape
    flat = q.permute(0, 2, 3, 1).reshape(-1, channels)
    if int(permutation.numel()) != int(flat.shape[0]):
        raise ValueError("shared shuffle permutation has the wrong token count")
    return flat[permutation].view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


def update_rate_ablation(
    state: dict,
    out: dict[str, torch.Tensor],
    imgs: torch.Tensor,
    continuous_final: torch.Tensor,
    zero_final: torch.Tensor,
    shuffle_final: torch.Tensor,
) -> None:
    batch_size = int(imgs.shape[0])
    psnr_final = base.batch_metric_mean(base.psnr_per_image(out["final"], imgs))
    psnr_continuous = base.batch_metric_mean(base.psnr_per_image(continuous_final, imgs))
    psnr_zero = base.batch_metric_mean(base.psnr_per_image(zero_final, imgs))
    psnr_shuffle = base.batch_metric_mean(base.psnr_per_image(shuffle_final, imgs))
    values = {
        "psnr_continuous": psnr_continuous,
        "gap_continuous": psnr_continuous - psnr_final,
        "psnr_zero": psnr_zero,
        "psnr_shuffle": psnr_shuffle,
        "drop_zero": psnr_final - psnr_zero,
        "drop_shuffle": psnr_final - psnr_shuffle,
    }
    for name, value in values.items():
        state["meters"][name].update(value, batch_size)


def finalize_all_metrics(
    objective_meters: dict,
    rate_states: dict[int, dict],
    levels: list[int],
) -> dict[str, float]:
    metrics = base.averaged(objective_meters)
    for level in levels:
        state = rate_states[level]
        rate_metrics = direct.finalize_metrics(
            state["meters"], state["hist"], state["level_hists"], state["args"]
        )
        for name, value in rate_metrics.items():
            metrics[f"l{level}_{name}"] = float(value)
    finals = [metrics[f"l{level}_psnr_final"] for level in levels]
    metrics["strict_psnr_monotonic"] = float(all(lower < higher for lower, higher in zip(finals, finals[1:])))
    for lower, higher in zip(levels, levels[1:]):
        metrics[transition_name(lower, higher, "psnr_gain")] = (
            metrics[f"l{higher}_psnr_final"] - metrics[f"l{lower}_psnr_final"]
        )
    return metrics


def display_metrics(metrics: dict[str, float], levels: list[int]) -> dict[str, float]:
    names = ["loss", "loss_recon", "loss_monotonic", "all_rates_strict_ratio", "strict_psnr_monotonic"]
    for lower, higher in zip(levels, levels[1:]):
        names.extend(
            [
                transition_name(lower, higher, "psnr_gain"),
                transition_name(lower, higher, "mse_gain"),
                transition_name(lower, higher, "raw_violation_ratio"),
                transition_name(lower, higher, "margin_violation_ratio"),
            ]
        )
    for level in levels:
        names.extend(
            [
                f"l{level}_psnr_x1",
                f"l{level}_psnr_final",
                f"l{level}_delta_x1",
                f"l{level}_psnr_continuous",
                f"l{level}_gap_continuous",
                f"l{level}_drop_zero",
                f"l{level}_drop_shuffle",
                f"l{level}_code_perplexity",
                f"l{level}_code_used",
                f"l{level}_fsq_mse",
            ]
        )
    return {name: metrics[name] for name in names if name in metrics}


def goal_eligible(metrics: dict[str, float], args: argparse.Namespace, levels: list[int]) -> bool:
    if not bool(metrics.get("strict_psnr_monotonic", 0.0)):
        return False
    for lower, higher in zip(levels, levels[1:]):
        if float(metrics.get(transition_name(lower, higher, "psnr_gain"), float("-inf"))) <= float(
            args.selection_min_rate_gain_db
        ):
            return False
    for level in levels:
        if float(metrics.get(f"l{level}_delta_x1", float("-inf"))) <= float(args.selection_min_delta_x1):
            return False
        if float(metrics.get(f"l{level}_drop_zero", float("-inf"))) < float(args.selection_min_drop_zero):
            return False
        if float(metrics.get(f"l{level}_drop_shuffle", float("-inf"))) < float(
            args.selection_min_drop_shuffle
        ):
            return False
    return float(metrics.get("all_rates_strict_ratio", 0.0)) >= float(args.selection_min_per_image_strict_ratio)


@torch.no_grad()
def validate(
    loader,
    bundle: direct.DirectBundle,
    args: argparse.Namespace,
    device: torch.device,
    levels: list[int],
    recon_weights: list[float],
    margins: list[float],
) -> dict[str, float]:
    bundle.e1.eval()
    bundle.d1.eval()
    bundle.tokenizer.eval()
    bundle.combiner.eval()
    rate_states = make_rate_states(args, levels, validation=True)
    objective_meters = base.meters(objective_metric_names(levels))
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if int(args.max_val_batches) > 0 and batch_index > int(args.max_val_batches):
            break
        imgs = imgs.to(device, non_blocking=True)
        layer1_out, z_norm, outputs = forward_multirate(bundle, imgs, levels)
        objective, branch_losses = compute_multirate_loss(
            outputs, imgs, levels, recon_weights, margins, float(args.lambda_monotonic)
        )
        update_objective_meters(objective_meters, objective, int(imgs.shape[0]))
        for level in levels:
            update_rate_state(rate_states[level], outputs[level], layer1_out, imgs, branch_losses[level], bundle)

        if bool(args.val_ablation):
            x1 = layer1_out["x1"]
            z1 = layer1_out["z1"]
            continuous = bundle.tokenizer.decode(z_norm, x1, z1, bundle.combiner)["final"]
            zero = bundle.tokenizer.decode(torch.zeros_like(z_norm), x1, z1, bundle.combiner)["final"]
            token_count = int(imgs.shape[0] * z_norm.shape[2] * z_norm.shape[3])
            permutation = torch.randperm(token_count, device=z_norm.device)
            for level in levels:
                shuffled = bundle.tokenizer.decode(
                    shuffled_with_perm(outputs[level]["q3"], permutation), x1, z1, bundle.combiner
                )["final"]
                update_rate_ablation(rate_states[level], outputs[level], imgs, continuous, zero, shuffled)
    return finalize_all_metrics(objective_meters, rate_states, levels)


def multirate_name(args: argparse.Namespace, levels: list[int]) -> str:
    tag = "-".join(str(level) for level in levels)
    return (
        f"layer2_fsq_multirate_{args.arch}_d{int(args.fsq_d)}_l{tag}_"
        f"{direct.normalizer_name(args)}_{args.codec_init}_{args.combiner_mode}"
    )


def load_direct_initialization(path: str, bundle: direct.DirectBundle, args: argparse.Namespace, levels: list[int]) -> None:
    if not path:
        return
    payload = torch.load(base.resolve_path(path), map_location="cpu")
    if str(payload.get("stage", "")) != "layer2_fsq_direct":
        raise ValueError(f"--init-direct-ckpt is not a direct Layer2 FSQ checkpoint: {path}")
    saved = payload.get("args", {})
    checks = {
        "arch": str(args.arch),
        "fsq_d": str(args.fsq_d),
        "fsq_normalizer": str(args.fsq_normalizer),
        "no_pre_norm": str(args.no_pre_norm),
        "combiner_mode": str(args.combiner_mode),
    }
    for key, current in checks.items():
        if str(saved.get(key)) != current:
            raise ValueError(f"direct init mismatch for {key}: checkpoint={saved.get(key)!r} current={current!r}")
    saved_levels = base.parse_fsq_levels(saved.get("fsq_levels"), int(args.fsq_d))
    required_high = repeated_levels(levels[-1], int(args.fsq_d))
    if saved_levels != required_high:
        raise ValueError(f"direct init must use highest grid {required_high}, got {saved_levels}")
    base.jsccf_io.load_state(bundle.e1, payload["e1_state_dict"], "multirate_init_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, payload["d1_state_dict"], "multirate_init_D1", strict=True)
    base.jsccf_io.load_state(
        bundle.tokenizer, payload["tokenizer_state_dict"], "multirate_init_direct_codec", strict=True
    )
    base.jsccf_io.load_state(
        bundle.combiner, payload["combiner_state_dict"], "multirate_init_direct_combiner", strict=True
    )
    print(f"initialized shared multirate model from direct checkpoint epoch={int(payload.get('epoch', -1))}: {path}", flush=True)


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    levels: list[int],
    metrics: dict[str, float],
    bundle: direct.DirectBundle,
    optimizer: optim.Optimizer,
    best_score: float,
    best_goal_score: float,
) -> None:
    output = Path(base.resolve_path(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    rate_metadata = {
        str(level): {
            "levels": repeated_levels(level, int(args.fsq_d)),
            "vocab_size": int(level) ** int(args.fsq_d),
            "fixed_bits_per_token": int(math.ceil(math.log2(float(int(level) ** int(args.fsq_d))))),
        }
        for level in levels
    }
    payload = {
        "route": getattr(base.jsccf_io, "ROUTE", "TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256"),
        "stage": "layer2_fsq_multirate",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "version": str(args.version),
        "source_layer2_ckpt": str(args.layer2_ckpt),
        "init_direct_ckpt": str(args.init_direct_ckpt),
        "nested_scalar_levels": levels,
        "rates": rate_metadata,
        "e1_state_dict": bundle.e1.state_dict(),
        "d1_state_dict": bundle.d1.state_dict(),
        "e2_state_dict": bundle.tokenizer.e3.state_dict(),
        "quantizer_state_dict": bundle.tokenizer.quantizer.state_dict(),
        "d2_state_dict": bundle.tokenizer.d3.state_dict(),
        "tokenizer_state_dict": bundle.tokenizer.state_dict(),
        "combiner_state_dict": bundle.combiner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": direct.capture_rng_state(),
        "best_score": float(best_score),
        "best_goal_score": float(best_goal_score),
        "init_report": bundle.init_report,
    }
    torch.save(payload, output)
    print(f"saved checkpoint: {output}", flush=True)


def load_resume(
    args: argparse.Namespace,
    bundle: direct.DirectBundle,
    optimizer: optim.Optimizer,
    levels: list[int],
) -> tuple[int, float, float]:
    if not args.resume:
        return 1, float("-inf"), float("-inf")
    if args.init_direct_ckpt:
        raise ValueError("--resume and --init-direct-ckpt are mutually exclusive")
    payload = torch.load(base.resolve_path(args.resume), map_location="cpu")
    if str(payload.get("stage", "")) != "layer2_fsq_multirate":
        raise ValueError(f"not a multirate Layer2 FSQ checkpoint: {args.resume}")
    if [int(value) for value in payload.get("nested_scalar_levels", [])] != levels:
        raise ValueError(
            f"resume nested levels mismatch: checkpoint={payload.get('nested_scalar_levels')} current={levels}"
        )
    saved = payload.get("args", {})
    for key in (
        "arch",
        "fsq_d",
        "fsq_normalizer",
        "no_pre_norm",
        "codec_init",
        "combiner_mode",
        "fresh_combiner",
        "freeze_combiner",
        "match_source_width",
        "lambda_monotonic",
        "monotonic_margins",
        "recon_weights",
    ):
        if str(saved.get(key)) != str(getattr(args, key)):
            raise ValueError(f"resume mismatch for {key}: checkpoint={saved.get(key)!r} current={getattr(args, key)!r}")
    gate_keys = (
        "selection_min_rate_gain_db",
        "selection_min_per_image_strict_ratio",
        "selection_min_delta_x1",
        "selection_min_drop_zero",
        "selection_min_drop_shuffle",
        "val_ablation",
    )
    changed_gates = [key for key in gate_keys if str(saved.get(key)) != str(getattr(args, key))]
    if changed_gates and not bool(args.reset_best_on_resume):
        raise ValueError(
            f"resume changes goal/validation protocol {changed_gates}; pass --reset-best-on-resume"
        )
    saved_calibration = int(saved.get("bn_calibration_batches", 0))
    if saved_calibration != int(args.bn_calibration_batches) and not bool(args.reset_best_on_resume):
        raise ValueError("changing --bn-calibration-batches on resume requires --reset-best-on-resume")
    base.jsccf_io.load_state(bundle.e1, payload["e1_state_dict"], "resume_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, payload["d1_state_dict"], "resume_D1", strict=True)
    base.jsccf_io.load_state(bundle.tokenizer, payload["tokenizer_state_dict"], "resume_multirate_codec", strict=True)
    base.jsccf_io.load_state(bundle.combiner, payload["combiner_state_dict"], "resume_combiner", strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if "rng_state" in payload:
        direct.restore_rng_state(payload["rng_state"])
    start = int(payload.get("epoch", 0)) + 1
    print(f"resumed {args.resume} at epoch {start}", flush=True)
    return start, float(payload.get("best_score", float("-inf"))), float(
        payload.get("best_goal_score", float("-inf"))
    )


def print_header(
    args: argparse.Namespace,
    bundle: direct.DirectBundle,
    levels: list[int],
    recon_weights: list[float],
    margins: list[float],
    train_count: int,
    val_count: int,
) -> None:
    rates = ", ".join(
        f"L={level}:K={level ** int(args.fsq_d)}:bits/token={math.ceil(math.log2(level ** int(args.fsq_d)))}"
        for level in levels
    )
    print(f"=== Layer 2 | shared nested multirate FSQ | {args.arch} ===", flush=True)
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"save_dir={base.resolve_path(args.save_dir)}", flush=True)
    print("实验设计", flush=True)
    print("  frozen Layer1 -> one trainable E2/norm -> nested FSQ rates -> shared trainable D2/combiner", flush=True)
    print("  u2_teacher=disabled; source Layer2 E2/D2 forward is never executed", flush=True)
    print(f"  nested_rates={rates}; same image, z3_norm, decoder, and epoch at every rate", flush=True)
    print(f"  init_direct_ckpt={args.init_direct_ckpt or '<none>'}", flush=True)
    print("loss设计", flush=True)
    print(
        f"  recon=weighted_mean(MSE(final_L,img)); weights={recon_weights}; "
        f"monotonic_margins={margins}; lambda_monotonic={float(args.lambda_monotonic):g}",
        flush=True,
    )
    print("  hinge=relu(MSE_higher-MSE_lower.detach()+margin), evaluated per image", flush=True)
    print("模块选择", flush=True)
    print(
        f"  E1={base.trainable_state(bundle.e1)} D1={base.trainable_state(bundle.d1)} "
        f"E2={base.trainable_state(bundle.tokenizer.e3)} D2={base.trainable_state(bundle.tokenizer.d3)} "
        f"combiner={base.trainable_state(bundle.combiner)}",
        flush=True,
    )
    print(
        f"  FSQ normalizer={direct.normalizer_name(args)} computed_once_per_batch; "
        f"combiner_mode={args.combiner_mode} bn_calibration_batches={int(args.bn_calibration_batches)}",
        flush=True,
    )
    print(
        "  psnr_continuous=learned d=3 tanh-normalized bypass diagnostic; "
        "it is not a source-z2 or K-specific capacity ceiling",
        flush=True,
    )
    print(
        f"  goal=all rates final>x1, adjacent PSNR gain>{float(args.selection_min_rate_gain_db):g}dB, "
        f"drop0>={float(args.selection_min_drop_zero):g}, dropshuffle>={float(args.selection_min_drop_shuffle):g}",
        flush=True,
    )
    print(
        f"epochs={int(args.epochs)} train={train_count} valid={val_count} batch={int(args.batch_size)} "
        f"test_batch={int(args.test_batch)} workers={int(args.num_workers)}/{int(args.val_num_workers)} "
        f"lr={float(args.lr):g} weight_decay={float(args.weight_decay):g}",
        flush=True,
    )


def train(args: argparse.Namespace, source_ckpt: dict) -> None:
    levels = parse_nested_levels(args.nested_levels, int(args.fsq_d))
    recon_weights = parse_float_list(args.recon_weights, len(levels), "--recon-weights")
    margins = parse_float_list(args.monotonic_margins, len(levels) - 1, "--monotonic-margins")
    base.seed_everything(int(args.seed))
    cfg = base.jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(cfg)
    bundle = direct.build_direct_bundle(args, source_ckpt, cfg.device)
    if args.init_direct_ckpt:
        load_direct_initialization(args.init_direct_ckpt, bundle, args, levels)
    params = list(bundle.tokenizer.parameters()) + [
        parameter for parameter in bundle.combiner.parameters() if parameter.requires_grad
    ]
    optimizer = optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    start_epoch, best_score, best_goal_score = load_resume(args, bundle, optimizer, levels)
    if args.resume and bool(args.reset_best_on_resume):
        best_score = float("-inf")
        best_goal_score = float("-inf")
        print("reset best scores after resume", flush=True)
    print_header(args, bundle, levels, recon_weights, margins, len(train_loader.dataset), len(val_loader.dataset))

    if bool(args.eval_init_only):
        calibration = direct.calibrate_fsq_batch_norm(train_loader, bundle, args, cfg.device)
        metrics = validate(val_loader, bundle, args, cfg.device, levels, recon_weights, margins)
        if calibration is not None:
            metrics.update({f"bn_calibration_{key}": value for key, value in calibration.items()})
        metrics["goal_eligible"] = float(goal_eligible(metrics, args, levels))
        print(
            f"[layer2-fsq-multirate init val] {display_metrics(metrics, levels)} "
            f"goal_eligible={bool(metrics['goal_eligible'])}",
            flush=True,
        )
        return

    if start_epoch > int(args.epochs):
        raise ValueError(f"resume starts at epoch {start_epoch}, beyond --epochs {int(args.epochs)}")

    last_metrics: dict[str, float] = {}
    last_checkpoint_metrics: dict[str, float] = {}
    name = multirate_name(args, levels)
    for epoch in range(start_epoch, int(args.epochs) + 1):
        bundle.e1.eval()
        bundle.d1.eval()
        bundle.tokenizer.train()
        bundle.combiner.train()
        if bool(args.freeze_combiner):
            bundle.combiner.inner.eval()
        rate_states = make_rate_states(args, levels, validation=False)
        objective_meters = base.meters(objective_metric_names(levels))
        started = time.time()
        for batch_index, (imgs, _labels) in enumerate(train_loader, start=1):
            if int(args.max_train_batches) > 0 and batch_index > int(args.max_train_batches):
                break
            imgs = imgs.to(cfg.device, non_blocking=True)
            layer1_out, _z_norm, outputs = forward_multirate(bundle, imgs, levels)
            objective, branch_losses = compute_multirate_loss(
                outputs, imgs, levels, recon_weights, margins, float(args.lambda_monotonic)
            )
            optimizer.zero_grad(set_to_none=True)
            objective["loss"].backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip_norm))
            optimizer.step()
            update_objective_meters(objective_meters, objective, int(imgs.shape[0]))
            for level in levels:
                update_rate_state(rate_states[level], outputs[level], layer1_out, imgs, branch_losses[level], bundle)

        last_metrics = finalize_all_metrics(objective_meters, rate_states, levels)
        print(
            f"[layer2-fsq-multirate train {epoch:03d}/{int(args.epochs):03d}] "
            f"{display_metrics(last_metrics, levels)} time={time.time() - started:.1f}s",
            flush=True,
        )

        checkpoint_metrics = last_metrics
        if base.should_validate(args, epoch):
            calibration = direct.calibrate_fsq_batch_norm(train_loader, bundle, args, cfg.device)
            val_metrics = validate(val_loader, bundle, args, cfg.device, levels, recon_weights, margins)
            if calibration is not None:
                val_metrics.update({f"bn_calibration_{key}": value for key, value in calibration.items()})
            eligible = goal_eligible(val_metrics, args, levels)
            val_metrics["goal_eligible"] = float(eligible)
            checkpoint_metrics = val_metrics
            print(
                f"[layer2-fsq-multirate val {epoch:03d}] {display_metrics(val_metrics, levels)} "
                f"goal_eligible={eligible}",
                flush=True,
            )
            score = sum(float(val_metrics[f"l{level}_psnr_final"]) for level in levels) / float(len(levels))
            improved = score > best_score
            improved_goal = eligible and score > best_goal_score
            if improved:
                best_score = score
            if improved_goal:
                best_goal_score = score
            if improved:
                save_checkpoint(
                    base.jsccf_io.ckpt_path(args, name, "best"),
                    epoch=epoch,
                    args=args,
                    levels=levels,
                    metrics=val_metrics,
                    bundle=bundle,
                    optimizer=optimizer,
                    best_score=best_score,
                    best_goal_score=best_goal_score,
                )
            if improved_goal:
                save_checkpoint(
                    base.jsccf_io.ckpt_path(args, name, "goal_best"),
                    epoch=epoch,
                    args=args,
                    levels=levels,
                    metrics=val_metrics,
                    bundle=bundle,
                    optimizer=optimizer,
                    best_score=best_score,
                    best_goal_score=best_goal_score,
                )

        last_checkpoint_metrics = checkpoint_metrics

        if base.should_save_latest(args, epoch):
            save_checkpoint(
                base.jsccf_io.ckpt_path(args, name, "latest"),
                epoch=epoch,
                args=args,
                levels=levels,
                metrics=checkpoint_metrics,
                bundle=bundle,
                optimizer=optimizer,
                best_score=best_score,
                best_goal_score=best_goal_score,
            )

    save_checkpoint(
        base.jsccf_io.ckpt_path(args, name, "latest"),
        epoch=int(args.epochs),
        args=args,
        levels=levels,
        metrics=last_checkpoint_metrics or last_metrics,
        bundle=bundle,
        optimizer=optimizer,
        best_score=best_score,
        best_goal_score=best_goal_score,
    )


@torch.no_grad()
def smoke_shapes(args: argparse.Namespace, source_ckpt: dict) -> None:
    levels = parse_nested_levels(args.nested_levels, int(args.fsq_d))
    base.seed_everything(int(args.seed))
    device = torch.device("cuda:0" if (not bool(args.cpu)) and torch.cuda.is_available() else "cpu")
    bundle = direct.build_direct_bundle(args, source_ckpt, device)
    if args.init_direct_ckpt:
        load_direct_initialization(args.init_direct_ckpt, bundle, args, levels)
    imgs = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    layer1_out, _z_norm, outputs = forward_multirate(bundle, imgs, levels)
    errors = [float(outputs[level]["fsq_mse"].item()) for level in levels]
    expected_q = (int(args.smoke_batch_size), int(args.fsq_d), int(args.latent_h), int(args.latent_w))
    for level in levels:
        if tuple(outputs[level]["q3"].shape) != expected_q:
            raise RuntimeError(f"L={level} expected q {expected_q}, got {tuple(outputs[level]['q3'].shape)}")
        if tuple(outputs[level]["final"].shape) != (int(args.smoke_batch_size), 3, 256, 256):
            raise RuntimeError(f"L={level} produced invalid image shape {tuple(outputs[level]['final'].shape)}")
    if any(higher > lower + 1e-7 for lower, higher in zip(errors, errors[1:])):
        raise RuntimeError(f"nested quantization error should be non-increasing, got {errors}")
    print(
        f"[smoke multirate] arch={args.arch} x1={tuple(layer1_out['x1'].shape)} "
        f"levels={levels} q={expected_q} fsq_mse={errors}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nested-levels", type=str, default="5,9,17")
    parser.add_argument("--recon-weights", type=str, default="1")
    parser.add_argument("--lambda-monotonic", type=float, default=1.0)
    parser.add_argument("--monotonic-margins", type=str, default="1e-5")
    parser.add_argument("--selection-min-rate-gain-db", type=float, default=0.0)
    parser.add_argument("--selection-min-per-image-strict-ratio", type=float, default=0.0)
    parser.add_argument("--init-direct-ckpt", type=str, default="")
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Shared multirate options: --nested-levels 5,9,17; --recon-weights 1,1,1; "
            "--lambda-monotonic FLOAT; --monotonic-margins 1e-5,1e-5; "
            "--selection-min-rate-gain-db FLOAT; --selection-min-per-image-strict-ratio FLOAT; "
            "--init-direct-ckpt PATH.\n",
            flush=True,
        )
    multirate, remaining = parser.parse_known_args()
    argv = sys.argv
    try:
        sys.argv = [argv[0], *remaining]
        args = direct.parse_args()
    finally:
        sys.argv = argv
    for key, value in vars(multirate).items():
        setattr(args, key, value)
    # High-gain Layer2 experiments must not inherit the safety bypass used by
    # early direct-route debugging.  An explicit CLI choice still wins.
    if not direct.cli_option_present(remaining, "--combiner-mode"):
        args.combiner_mode = "original"
    if not direct.cli_option_present(remaining, "--val-num-workers"):
        args.val_num_workers = 4
    if not direct.cli_option_present(remaining, "--selection-min-drop-zero"):
        args.selection_min_drop_zero = 0.1
    if not direct.cli_option_present(remaining, "--selection-min-drop-shuffle"):
        args.selection_min_drop_shuffle = 0.1
    if not math.isfinite(float(args.lambda_monotonic)) or float(args.lambda_monotonic) < 0.0:
        raise ValueError("--lambda-monotonic must be non-negative")
    if not 0.0 <= float(args.selection_min_per_image_strict_ratio) <= 1.0:
        raise ValueError("--selection-min-per-image-strict-ratio must be in [0,1]")
    return args


def main() -> None:
    args = parse_args()
    args.stage = "layer2_fsq_multirate"
    base.apply_preset(args)
    if str(args.preset) != "custom":
        raise ValueError("multirate comparisons require --preset custom")
    levels = parse_nested_levels(args.nested_levels, int(args.fsq_d))
    recon_weights = parse_float_list(args.recon_weights, len(levels), "--recon-weights")
    margins = parse_float_list(args.monotonic_margins, len(levels) - 1, "--monotonic-margins")
    if any(weight < 0.0 for weight in recon_weights) or sum(recon_weights) <= 0.0:
        raise ValueError(f"--recon-weights must be non-negative with positive sum, got {recon_weights}")
    if any(margin < 0.0 for margin in margins):
        raise ValueError(f"--monotonic-margins must be non-negative, got {margins}")
    args.fsq_levels = repeated_levels(levels[-1], int(args.fsq_d))
    if float(args.lambda_usage) != 0.0:
        raise ValueError("multirate direct FSQ does not use a usage/KL objective; set --lambda-usage 0")
    if float(args.lambda_u2_img) != 0.0:
        raise ValueError("multirate direct FSQ has no u2 teacher/image auxiliary; set --lambda-u2-img 0")
    direct.explore.validate_explore_args(args)
    source_ckpt = base.load_teacher_checkpoint_for_args(args)
    if str(args.condition_mode) != "none":
        raise ValueError("multirate direct Layer2 FSQ requires --condition-mode none")
    if str(args.variant) != "combiner":
        raise ValueError(f"multirate route requires a combiner Layer2 source, got variant={args.variant!r}")
    if bool(args.match_source_width) and str(args.arch) == "cnn" and str(args.codec_init) == "compatible":
        args.e3_base_ch = int(args.cnn_base_ch)
        args.d3_base_ch = int(args.cnn_base_ch)
        args.e3_num_res = int(args.cnn_num_res)
        args.d3_num_res = int(args.cnn_num_res)
    args._usage_weight = 0.0
    direct.explore.ExploreIFSQQuantizer.config = args
    base.check_jsccf_args(args)

    if not args.save_dir or str(args.save_dir) == str(THIS_DIR / "checkpoints-direct"):
        args.save_dir = str(THIS_DIR / "checkpoints-multirate")
    Path(base.resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    name = multirate_name(args, levels)
    if not args.log_file:
        args.log_file = str(
            THIS_DIR / "logs-multirate" / f"{name}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}.log"
        )
    base.setup_log_file(args.log_file)
    base.write_json(
        Path(base.resolve_path(args.save_dir))
        / f"{name}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}_args.json",
        {key: value for key, value in vars(args).items() if not key.startswith("_")},
    )
    if bool(args.smoke_shapes):
        smoke_shapes(args, source_ckpt)
        return
    train(args, source_ckpt)


if __name__ == "__main__":
    main()
