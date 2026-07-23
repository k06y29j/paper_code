#!/usr/bin/env python3
"""Summarize explore-2 nested Layer2 VQ checkpoints without re-evaluation.

The checkpoint-embedded ``args`` and validation ``metrics`` are the only metric
source used here.  One checkpoint is selected per embedded experiment version:
valid ``e2_input_order=img_x1`` first, then ``goal_best > best > latest > other``.
Missing evidence remains missing: this script never promotes a run by deriving
a successful gate from PSNR or codebook diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STAGE = "explore2_layer2_nested_vq"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "checkpoints-vq"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results-vq"
DEFAULT_EXPECTED_ARCHES = ("cnn", "swin")
DEFAULT_EXPECTED_FAMILIES = ("image-vq", "channel-vq")
CHECKPOINT_KIND_PRIORITY = {"other": 0, "latest": 1, "best": 2, "goal_best": 3}


@dataclass(frozen=True)
class Candidate:
    path: Path
    display_path: str
    checkpoint_kind: str
    priority: int
    version: str
    channel_codebook_mode: str
    epoch: int | None
    args: dict[str, Any]
    metrics: dict[str, Any]
    rates: tuple[int, ...]


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(vars(value))
    except (TypeError, AttributeError):
        return {}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _parse_ints(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        pieces: Iterable[Any] = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        pieces = value
    else:
        return ()
    result: list[int] = []
    for piece in pieces:
        try:
            number = int(str(piece).strip())
        except (TypeError, ValueError):
            return ()
        if number <= 0:
            return ()
        result.append(number)
    return tuple(result)


def _channel_codebook_mode(args: Mapping[str, Any]) -> str:
    """Normalize the embedded channel-VQ namespace; legacy runs are global."""

    raw = args.get("channel_codebook_mode", "global")
    mode = str(raw).strip().lower().replace("_", "-")
    return mode or "global"


def _checkpoint_kind(path: Path) -> tuple[str, int]:
    name = path.name
    if name.endswith("_goal_best.pth"):
        return "goal_best", 3
    if name.endswith("_best.pth"):
        return "best", 2
    if name.endswith("_latest.pth"):
        return "latest", 1
    return "other", 0


def _load_payload(path: Path) -> Any:
    """Load metadata while mapping tensor storages to meta when supported."""
    import torch

    kwargs = {"map_location": "meta", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError, ValueError) as error:
        # Old (non-zip) serialization and older torch releases may reject mmap.
        if "mmap" not in str(error).lower() and not isinstance(error, TypeError):
            raise
        return torch.load(path, **kwargs)


def _checkpoint_paths(roots: Sequence[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved.is_file():
            if resolved.suffix == ".pth":
                paths.add(resolved)
            continue
        if resolved.is_dir():
            paths.update(path.resolve() for path in resolved.rglob("*.pth") if path.is_file())
    return sorted(paths, key=lambda path: path.as_posix())


def discover_candidates(roots: Sequence[Path]) -> tuple[list[Candidate], dict[str, Any]]:
    paths = _checkpoint_paths(roots)
    candidates: list[Candidate] = []
    load_errors: list[dict[str, str]] = []
    ignored_non_stage = 0
    for path in paths:
        try:
            payload = _load_payload(path)
        except Exception as error:  # A bad checkpoint must be visible, not fatal to the report.
            load_errors.append(
                {
                    "checkpoint": _display_path(path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        try:
            if not isinstance(payload, Mapping) or str(payload.get("stage", "")) != STAGE:
                ignored_non_stage += 1
                continue
            saved_args = _plain_mapping(payload.get("args", {}))
            saved_metrics = _plain_mapping(payload.get("metrics", {}))
            rates = _parse_ints(payload.get("rates")) or _parse_ints(saved_args.get("rates"))
            kind, priority = _checkpoint_kind(path)
            embedded_version = str(saved_args.get("version", "")).strip()
            version = embedded_version or path.parent.name or path.stem
            channel_codebook_mode = _channel_codebook_mode(saved_args)
            candidates.append(
                Candidate(
                    path=path,
                    display_path=_display_path(path),
                    checkpoint_kind=kind,
                    priority=priority,
                    version=version,
                    channel_codebook_mode=channel_codebook_mode,
                    epoch=_integer(payload.get("epoch")),
                    args=saved_args,
                    metrics=saved_metrics,
                    rates=rates,
                )
            )
        finally:
            del payload
            gc.collect()
    discovery = {
        "pth_files_scanned": len(paths),
        "matching_stage_checkpoints": len(candidates),
        "ignored_non_stage_checkpoints": ignored_non_stage,
        "load_errors": load_errors,
    }
    return candidates, discovery


def select_versions(candidates: Sequence[Candidate]) -> tuple[list[Candidate], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.version, candidate.channel_codebook_mode)].append(candidate)

    selected: list[Candidate] = []
    discarded: list[dict[str, Any]] = []
    for (version, channel_codebook_mode), choices in sorted(grouped.items()):
        # A valid E2 contract dominates legacy snapshots.  Within the same
        # contract, goal_best priority dominates epoch by design.
        ordered = sorted(
            choices,
            key=lambda item: (
                int(str(item.args.get("e2_input_order", "")).strip() == "img_x1"),
                item.priority,
                item.epoch if item.epoch is not None else -1,
                item.path.stat().st_mtime_ns,
                item.display_path,
            ),
            reverse=True,
        )
        winner = ordered[0]
        selected.append(winner)
        for item in ordered[1:]:
            discarded.append(
                {
                    "version": version,
                    "channel_codebook_mode": channel_codebook_mode,
                    "checkpoint": item.display_path,
                    "checkpoint_kind": item.checkpoint_kind,
                    "e2_input_order": item.args.get("e2_input_order"),
                    "epoch": item.epoch,
                    "selected_checkpoint": winner.display_path,
                    "reason": (
                        "lower selection rank within the same embedded version "
                        "(valid e2_input_order first, then checkpoint kind and epoch)"
                    ),
                }
            )
    return selected, discarded


def _metric(metrics: Mapping[str, Any], key: str) -> float | None:
    return _finite_float(metrics.get(key))


def _first_number(sources: Sequence[tuple[str, Mapping[str, Any], Sequence[str]]]) -> tuple[float | None, str | None]:
    """Return the first finite embedded number and its fully qualified source."""
    for namespace, mapping, keys in sources:
        for key in keys:
            value = _finite_float(mapping.get(key))
            if value is not None:
                return value, f"embedded {namespace}.{key}"
    return None, None


def _paired_threshold(
    metrics: Mapping[str, Any],
    args: Mapping[str, Any],
    rate: int,
    name: str,
) -> tuple[float | None, str | None]:
    if name == "strict":
        stems = ("min_paired_strict", "paired_strict_threshold")
    elif name == "gain":
        stems = ("min_paired_gain_db", "min_paired_gain", "paired_gain_threshold")
    else:
        raise ValueError(f"unknown paired threshold: {name}")
    per_rate = tuple(f"{stem}_k{rate}" for stem in stems)
    return _first_number(
        (
            ("metrics", metrics, per_rate + stems),
            ("args", args, per_rate + stems),
        )
    )


def _gate(metrics: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = _finite_float(metrics.get(key))
    if raw is None:
        return {"status": "unknown", "raw": None, "source": f"embedded metrics.{key}"}
    return {
        "status": "pass" if raw == 1.0 else "fail",
        "raw": raw,
        "source": f"embedded metrics.{key}",
    }


def _run_from_candidate(candidate: Candidate) -> dict[str, Any]:
    args = candidate.args
    metrics = candidate.metrics
    rates = tuple(sorted(set(candidate.rates)))
    family = str(args.get("vq_family", "")) or None
    layer1_arch = str(args.get("arch", "")) or None
    requested_layer2_arch = str(args.get("layer2_arch", "match") or "match")
    layer2_arch = layer1_arch if requested_layer2_arch == "match" else requested_layer2_arch
    channel_codebook_mode = candidate.channel_codebook_mode
    latent_c = _integer(args.get("latent_c"))
    requested_embedding_dim = _integer(args.get("embedding_dim"))
    embedding_dim = (
        requested_embedding_dim
        if requested_embedding_dim is not None and requested_embedding_dim > 0
        else (latent_c if family == "image-vq" else 16 * 16)
    )
    grouped_channel_vq = family == "channel-vq" and channel_codebook_mode == "grouped"
    per_rate: list[dict[str, Any]] = []
    missing_metrics: list[str] = []
    missing_thresholds: list[str] = []
    collapse_triggers: list[dict[str, Any]] = []
    collapse_unchecked: list[dict[str, Any]] = []
    paired_evidence_ok: list[bool] = []

    required_per_rate = (
        "psnr",
        "delta_x1",
        "drop_zero",
        "drop_shuffle",
        "used",
        "ppl",
        "ppl_ratio",
        "top1",
        "vq_mse",
    )
    grouped_per_rate = (
        "local_used",
        "local_ppl",
        "local_ppl_ratio",
        "local_top1",
        "channel_variation",
        "channel_variation_min",
        "variant_channel_frac",
    )
    nominal_rate_per_rate = (
        "tokens_per_image",
        "candidates_per_token",
        "bits_per_token",
        "bits_per_image",
        "bpp",
    )
    for index, rate in enumerate(rates):
        row: dict[str, Any] = {"K": rate}
        for prefix in required_per_rate:
            key = f"{prefix}_k{rate}"
            row[prefix] = _metric(metrics, key)
            if row[prefix] is None:
                missing_metrics.append(key)
        for prefix in grouped_per_rate:
            row[prefix] = _metric(metrics, f"{prefix}_k{rate}")
        for prefix in nominal_rate_per_rate:
            row[prefix] = _metric(metrics, f"{prefix}_k{rate}")
        for prefix in ("paired_strict", "paired_gain"):
            key = f"{prefix}_k{rate}"
            row[prefix] = _metric(metrics, key)
            if index > 0 and row[prefix] is None:
                missing_metrics.append(key)

        strict_threshold, strict_threshold_source = _paired_threshold(metrics, args, rate, "strict")
        gain_threshold, gain_threshold_source = _paired_threshold(metrics, args, rate, "gain")
        row["paired_strict_threshold"] = strict_threshold
        row["paired_strict_threshold_source"] = strict_threshold_source
        row["paired_gain_threshold"] = gain_threshold
        row["paired_gain_threshold_source"] = gain_threshold_source
        embedded_pair_gate, embedded_pair_gate_source = _first_number(
            (
                (
                    "metrics",
                    metrics,
                    (
                        f"paired_goal_met_k{rate}",
                        f"paired_gate_k{rate}",
                        f"paired_threshold_met_k{rate}",
                    ),
                ),
            )
        )
        row["paired_embedded_gate"] = embedded_pair_gate
        row["paired_embedded_gate_source"] = embedded_pair_gate_source
        if index == 0:
            row["paired_threshold_status"] = "not_applicable"
            paired_evidence_ok.append(True)
        elif embedded_pair_gate is not None:
            row["paired_threshold_status"] = (
                "PASS (embedded)" if embedded_pair_gate == 1.0 else "FAIL (embedded)"
            )
            paired_evidence_ok.append(embedded_pair_gate == 1.0)
        elif strict_threshold is None or gain_threshold is None:
            row["paired_threshold_status"] = "UNKNOWN (threshold missing)"
            if strict_threshold is None:
                missing_thresholds.append(f"paired_strict_threshold_k{rate}")
            if gain_threshold is None:
                missing_thresholds.append(f"paired_gain_threshold_k{rate}")
            paired_evidence_ok.append(False)
        elif row["paired_strict"] is None or row["paired_gain"] is None:
            row["paired_threshold_status"] = "UNKNOWN (metric missing)"
            paired_evidence_ok.append(False)
        else:
            meets = row["paired_strict"] >= strict_threshold and row["paired_gain"] >= gain_threshold
            row["paired_threshold_status"] = (
                "meets embedded thresholds (observed)" if meets else "BELOW embedded thresholds"
            )
            paired_evidence_ok.append(meets)

        triggers: list[str] = []
        token_change_key = f"token_change_k{rate}"
        token_change = _metric(metrics, token_change_key)
        row["token_change"] = token_change
        token_change_threshold, token_change_threshold_source = _first_number(
            (
                (
                    "metrics",
                    metrics,
                    (
                        f"min_token_change_k{rate}",
                        f"min_index_change_k{rate}",
                        "min_token_change",
                        "min_index_change",
                    ),
                ),
                (
                    "args",
                    args,
                    (
                        f"min_token_change_k{rate}",
                        f"min_index_change_k{rate}",
                        "min_token_change",
                        "min_index_change",
                    ),
                ),
            )
        )
        row["token_change_threshold"] = token_change_threshold
        row["token_change_threshold_source"] = token_change_threshold_source
        collapse_evidence_missing = False
        if grouped_channel_vq:
            local_threshold_specs = (
                ("local_ppl", "min_local_ppl", "min", 1.1),
                ("local_top1", "max_local_top1", "max", 0.95),
                ("channel_variation", "min_channel_variation", "min", 0.01),
                ("variant_channel_frac", "min_variant_channel_frac", "min", 0.1),
            )
            for metric_name, threshold_name, direction, _documented_default in local_threshold_specs:
                threshold, threshold_source = _first_number(
                    (
                        ("metrics", metrics, (f"{threshold_name}_k{rate}", threshold_name)),
                        ("args", args, (f"{threshold_name}_k{rate}", threshold_name)),
                    )
                )
                row[f"{metric_name}_threshold"] = threshold
                row[f"{metric_name}_threshold_source"] = threshold_source
            local_failures: list[str] = []
            for metric_name, threshold_name, direction, _documented_default in local_threshold_specs:
                value = row[metric_name]
                threshold = row[f"{metric_name}_threshold"]
                if value is None:
                    missing_metrics.append(f"{metric_name}_k{rate}")
                    collapse_evidence_missing = True
                    continue
                if threshold is None:
                    missing_thresholds.append(f"{threshold_name}_k{rate}")
                    collapse_evidence_missing = True
                    continue
                failed = value < threshold if direction == "min" else value > threshold
                if failed:
                    comparator = "<" if direction == "min" else ">"
                    local_failures.append(
                        f"{metric_name}={value:.6g}{comparator}{threshold_name}={threshold:.6g}"
                    )
            triggers.extend(local_failures)
            if local_failures:
                row["local_noncollapse_status"] = "BELOW embedded threshold"
            elif collapse_evidence_missing:
                row["local_noncollapse_status"] = "UNKNOWN (metric/threshold missing)"
            else:
                row["local_noncollapse_status"] = "meets embedded thresholds (observed)"
            row["token_change_status"] = "not_used_for_grouped_noncollapse"
        else:
            if row["ppl"] is not None and row["ppl"] <= 1.1:
                triggers.append(f"PPL={row['ppl']:.6g}<=1.1")
            if row["top1"] is not None and row["top1"] >= 0.95:
                triggers.append(f"top1={row['top1']:.6g}>=0.95")
            if token_change is None:
                missing_metrics.append(token_change_key)
                collapse_evidence_missing = True
                row["token_change_status"] = "UNKNOWN (metric missing)"
            elif token_change_threshold is None:
                missing_thresholds.append(f"token_change_threshold_k{rate}")
                collapse_evidence_missing = True
                row["token_change_status"] = "UNKNOWN (threshold missing)"
            elif token_change < token_change_threshold:
                row["token_change_status"] = "BELOW embedded threshold"
                triggers.append(
                    f"token_change={token_change:.6g}<min={token_change_threshold:.6g}"
                )
            else:
                row["token_change_status"] = "meets embedded threshold (observed)"
        if triggers:
            row["collapse_status"] = "COLLAPSE"
            collapse_triggers.append({"K": rate, "triggers": triggers})
        elif collapse_evidence_missing:
            row["collapse_status"] = "unknown"
        else:
            row["collapse_status"] = "no_observed_trigger"
        if not grouped_channel_vq and token_change is None:
            collapse_unchecked.append({"K": rate, "missing_check": token_change_key})
        row["collapse_triggers"] = triggers
        per_rate.append(row)

    monotonic_pairs: list[dict[str, Any]] = []
    non_monotonic_pairs: list[dict[str, Any]] = []
    missing_monotonic_pairs: list[dict[str, Any]] = []
    for low, high in zip(per_rate[:-1], per_rate[1:]):
        low_psnr, high_psnr = low["psnr"], high["psnr"]
        pair = {"low_K": low["K"], "high_K": high["K"], "low_psnr": low_psnr, "high_psnr": high_psnr}
        if low_psnr is None or high_psnr is None:
            pair["status"] = "unknown"
            missing_monotonic_pairs.append(pair)
        elif high_psnr > low_psnr:
            pair["status"] = "strict_increase"
        else:
            pair["status"] = "NON_MONOTONIC"
            non_monotonic_pairs.append(pair)
        monotonic_pairs.append(pair)
    if non_monotonic_pairs:
        monotonic_status = "NON_MONOTONIC"
    elif missing_monotonic_pairs or len(per_rate) < 2:
        monotonic_status = "unknown"
    else:
        monotonic_status = "strict_increase_observed"

    # Freeze the sender-oracle evidence set before receiver-only diagnostics
    # append their own missing fields.  Oracle-only runs intentionally do not
    # train a predictor, so their canonical capacity verdict must not depend
    # on receiver ablations; receiver sufficiency is reported separately.
    oracle_missing_metrics = sorted(set(missing_metrics))
    oracle_missing_thresholds = sorted(set(missing_thresholds))

    base_metric_keys = (
        "psnr_x1",
        "psnr_continuous",
        "psnr_pred",
        "delta_x1_pred",
        "pred_drop_zero",
        "pred_drop_shuffle",
        "pred_index_accuracy",
        "pred_local_used",
        "pred_local_ppl",
        "pred_local_ppl_ratio",
        "pred_local_top1",
        "pred_channel_variation",
        "pred_channel_variation_min",
        "pred_variant_channel_frac",
        "condition_shuffle_drop",
        "receiver_only_audit",
        "oracle_goal_met",
        "receiver_goal_met",
        "goal_met",
    )
    base_metrics = {key: _metric(metrics, key) for key in base_metric_keys}
    condition_shuffle_drop, condition_shuffle_source = _first_number(
        (
            (
                "metrics",
                metrics,
                (
                    "condition_shuffle_drop",
                    "pred_condition_shuffle_drop",
                    "receiver_condition_shuffle_drop",
                ),
            ),
        )
    )
    base_metrics["condition_shuffle_drop"] = condition_shuffle_drop
    condition_shuffle_threshold, condition_shuffle_threshold_source = _first_number(
        (
            (
                "metrics",
                metrics,
                ("min_condition_shuffle_drop", "condition_shuffle_threshold"),
            ),
            (
                "args",
                args,
                ("min_condition_shuffle_drop", "condition_shuffle_threshold", "min_ablation_drop"),
            ),
        )
    )
    if condition_shuffle_drop is None:
        condition_shuffle_status = "UNKNOWN (metric missing)"
    elif condition_shuffle_threshold is None:
        condition_shuffle_status = "UNKNOWN (threshold missing)"
        missing_thresholds.append("condition_shuffle_threshold")
    elif condition_shuffle_drop >= condition_shuffle_threshold:
        condition_shuffle_status = "meets embedded threshold (observed)"
    else:
        condition_shuffle_status = "BELOW embedded threshold"
    grouped_pred_keys = {
        "pred_local_used",
        "pred_local_ppl",
        "pred_local_ppl_ratio",
        "pred_local_top1",
        "pred_channel_variation",
        "pred_channel_variation_min",
        "pred_variant_channel_frac",
    }
    missing_metrics.extend(
        key
        for key, value in base_metrics.items()
        if value is None and key not in grouped_pred_keys
    )
    if grouped_channel_vq and rates and (latent_c is None or max(rates) > latent_c):
        missing_metrics.extend(key for key in grouped_pred_keys if base_metrics[key] is None)
    gates = {
        "oracle": _gate(metrics, "oracle_goal_met"),
        "receiver": _gate(metrics, "receiver_goal_met"),
        "overall": _gate(metrics, "goal_met"),
    }

    oracle_strict_ok = (
        gates["oracle"]["status"] == "pass"
        and monotonic_status == "strict_increase_observed"
        and not collapse_triggers
        and not oracle_missing_metrics
        and not oracle_missing_thresholds
        and all(paired_evidence_ok)
    )
    receiver_strict_ok = (
        gates["receiver"]["status"] == "pass"
        and base_metrics["receiver_only_audit"] == 1.0
        and condition_shuffle_status == "meets embedded threshold (observed)"
    )

    observed_warnings: list[str] = []
    if not rates:
        observed_warnings.append("rates missing from checkpoint payload and embedded args")
    if monotonic_status == "NON_MONOTONIC":
        observed_warnings.append("capacity PSNR is non-monotonic")
    if collapse_triggers:
        observed_warnings.append("one or more K values trigger a collapse threshold")
    if missing_metrics:
        observed_warnings.append("required embedded metrics are missing")
    if missing_thresholds:
        observed_warnings.append("required paired/condition thresholds are missing")
    if condition_shuffle_status.startswith("UNKNOWN"):
        observed_warnings.append("condition-shuffle sufficiency is UNKNOWN")
    elif condition_shuffle_status.startswith("BELOW"):
        observed_warnings.append("condition-shuffle drop is below its embedded threshold")
    if any(not passed for passed in paired_evidence_ok):
        observed_warnings.append("paired threshold evidence is failed or UNKNOWN")
    if base_metrics["receiver_only_audit"] != 1.0:
        observed_warnings.append("receiver_only_audit is not an embedded PASS")
    if gates["overall"]["status"] == "pass" and (
        gates["oracle"]["status"] != "pass" or gates["receiver"]["status"] != "pass"
    ):
        observed_warnings.append("embedded overall gate conflicts with component gates")
    if gates["oracle"]["status"] == "pass" and monotonic_status != "strict_increase_observed":
        observed_warnings.append("embedded oracle gate conflicts with observed monotonicity evidence")

    strict_goal_ok = oracle_strict_ok and receiver_strict_ok and gates["overall"]["status"] == "pass"

    raw_e2_input_order = args.get("e2_input_order")
    e2_input_order = str(raw_e2_input_order).strip() if raw_e2_input_order is not None else None
    if not e2_input_order:
        e2_input_order = None
        e2_input_order_status = "LEGACY_UNKNOWN"
        observed_warnings.append("e2_input_order is missing; legacy run is excluded from formal cells")
    elif e2_input_order == "img_x1":
        e2_input_order_status = "VALID"
    else:
        e2_input_order_status = "INVALID"
        observed_warnings.append(
            f"e2_input_order={e2_input_order!r} is not img_x1; run is excluded from formal cells"
        )
    grouped_promotion_eligible = not grouped_channel_vq or (
        latent_c is not None and bool(rates) and all(rate > latent_c for rate in rates)
    )
    if not grouped_promotion_eligible:
        observed_warnings.append(
            "grouped K=C is a zero-index-bit structural diagnostic and is excluded from formal promotion"
        )
    receiver_phase = str(args.get("receiver_phase", "none") or "none")
    receiver_only_phase = bool(args.get("receiver_only", False)) or receiver_phase != "none"
    if receiver_only_phase:
        observed_warnings.append(
            "receiver-phase checkpoint is reported for sufficiency diagnostics but excluded "
            "from the canonical sender-oracle capacity cell"
        )
    formal_cell_eligible = (
        e2_input_order == "img_x1"
        and grouped_promotion_eligible
        and not receiver_only_phase
    )
    strict_goal_ok = strict_goal_ok and formal_cell_eligible

    return {
        "version": candidate.version,
        "arch": layer1_arch,
        "layer2_arch": layer2_arch,
        "family": family,
        "channel_codebook_mode": channel_codebook_mode,
        "C": latent_c,
        "embedding_dim": embedding_dim,
        "requested_embedding_dim": requested_embedding_dim or 0,
        "rates": list(rates),
        "seed": _integer(args.get("seed")),
        "epoch": candidate.epoch,
        "checkpoint": candidate.display_path,
        "checkpoint_kind": candidate.checkpoint_kind,
        "e2_input_order": e2_input_order,
        "e2_input_order_status": e2_input_order_status,
        "grouped_promotion_eligible": grouped_promotion_eligible,
        "receiver_phase": receiver_phase,
        "receiver_only_phase": receiver_only_phase,
        "formal_cell_eligible": formal_cell_eligible,
        "psnr_x1": base_metrics["psnr_x1"],
        "psnr_continuous": base_metrics["psnr_continuous"],
        "psnr_pred": base_metrics["psnr_pred"],
        "delta_x1_pred": base_metrics["delta_x1_pred"],
        "pred_drop_zero": base_metrics["pred_drop_zero"],
        "pred_drop_shuffle": base_metrics["pred_drop_shuffle"],
        "pred_index_accuracy": base_metrics["pred_index_accuracy"],
        "pred_grouped_metrics": {
            "local_used": base_metrics["pred_local_used"],
            "local_ppl": base_metrics["pred_local_ppl"],
            "local_ppl_ratio": base_metrics["pred_local_ppl_ratio"],
            "local_top1": base_metrics["pred_local_top1"],
            "channel_variation": base_metrics["pred_channel_variation"],
            "channel_variation_min": base_metrics["pred_channel_variation_min"],
            "variant_channel_frac": base_metrics["pred_variant_channel_frac"],
        },
        "condition_shuffle_drop": condition_shuffle_drop,
        "condition_shuffle_source": condition_shuffle_source,
        "condition_shuffle_threshold": condition_shuffle_threshold,
        "condition_shuffle_threshold_source": condition_shuffle_threshold_source,
        "condition_shuffle_status": condition_shuffle_status,
        "receiver_only_audit": base_metrics["receiver_only_audit"],
        "gates": gates,
        "per_K": per_rate,
        "diagnostics": {
            "oracle_strict_ok": oracle_strict_ok and formal_cell_eligible,
            "receiver_strict_ok": receiver_strict_ok,
            "monotonic_status": monotonic_status,
            "monotonic_pairs": monotonic_pairs,
            "non_monotonic_pairs": non_monotonic_pairs,
            "collapse_triggers": collapse_triggers,
            "collapse_unchecked": collapse_unchecked,
            "missing_metrics": sorted(set(missing_metrics)),
            "missing_thresholds": sorted(set(missing_thresholds)),
            "paired_thresholds_ok": all(paired_evidence_ok),
            "warnings": observed_warnings,
            "strict_goal_ok": strict_goal_ok,
        },
    }


def _canonical_score(run: Mapping[str, Any]) -> float | None:
    per_rate = run.get("per_K", [])
    if not per_rate:
        return None
    highest = max(per_rate, key=lambda row: row.get("K", -1))
    return _finite_float(highest.get("psnr"))


def _canonical_rank(run: Mapping[str, Any]) -> tuple[int, int, int, float, str]:
    score = _canonical_score(run)
    return (
        int(bool(run["diagnostics"]["oracle_strict_ok"])),
        CHECKPOINT_KIND_PRIORITY.get(str(run.get("checkpoint_kind")), -1),
        run["epoch"] if run.get("epoch") is not None else -1,
        score if score is not None else float("-inf"),
        str(run.get("version", "")),
    )


def _canonical_rank_record(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strict_goal_ok": bool(run["diagnostics"]["oracle_strict_ok"]),
        "checkpoint_kind": run.get("checkpoint_kind"),
        "checkpoint_kind_priority": CHECKPOINT_KIND_PRIORITY.get(str(run.get("checkpoint_kind")), -1),
        "epoch": run.get("epoch"),
        "max_K_psnr_score": _canonical_score(run),
    }


def select_canonical_cells(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one eligible acceptance run per exact cell while retaining every run."""
    grouped: dict[tuple[Any, Any, tuple[int, ...], Any, Any, Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (
            run["C"], run["embedding_dim"], tuple(run["rates"]), run["seed"], run["arch"], run["layer2_arch"], run["family"],
            run["channel_codebook_mode"],
        )
        grouped[key].append(run)

    policy = (
        "eligible sender-oracle run (e2_input_order=img_x1, grouped K>C, receiver_phase=none); "
        "then strict_goal_ok; then checkpoint kind "
        "goal_best>best>latest>other; then epoch; then max-K PSNR score; then version"
    )
    canonical_cells: list[dict[str, Any]] = []
    for key, members in sorted(
        grouped.items(),
        key=lambda item: tuple("" if value is None else str(value) for value in item[0]),
    ):
        eligible = [run for run in members if run["formal_cell_eligible"]]
        winner = sorted(eligible, key=_canonical_rank, reverse=True)[0] if eligible else None
        if winner is None:
            reason = (
                "no canonical run: every version fails sender-oracle eligibility "
                "(E2 input contract, grouped K>C, or receiver phase)"
            )
        else:
            winner_rank = _canonical_rank_record(winner)
            reason = (
                f"selected {winner['version']!r} by {policy}; rank="
                f"strict_goal_ok={winner_rank['strict_goal_ok']}, "
                f"kind={winner_rank['checkpoint_kind']}, epoch={winner_rank['epoch']}, "
                f"max_K_psnr={winner_rank['max_K_psnr_score']!r}; "
                f"eligible_alternatives={len(eligible) - 1}"
            )
        for run in members:
            is_winner = run is winner
            run["canonical"] = is_winner
            run["canonical_selected_version"] = winner["version"] if winner is not None else None
            run["canonical_rank"] = _canonical_rank_record(run)
            if is_winner:
                run["canonical_selection_reason"] = reason
            elif not run["formal_cell_eligible"]:
                run["canonical_selection_reason"] = (
                    f"excluded: e2_input_order status={run['e2_input_order_status']}; {policy}"
                )
            else:
                run["canonical_selection_reason"] = (
                    f"not canonical; {winner['version']!r} outranked this run under: {policy}"
                )
        latent_c, embedding_dim, rates, seed, arch, layer2_arch, family, channel_codebook_mode = key
        canonical_cells.append(
            {
                "C": latent_c,
                "embedding_dim": embedding_dim,
                "rates": list(rates),
                "seed": seed,
                "arch": arch,
                "layer2_arch": layer2_arch,
                "family": family,
                "channel_codebook_mode": channel_codebook_mode,
                "status": "selected" if winner is not None else "MISSING_VALID_RUN",
                "canonical_version": winner["version"] if winner is not None else None,
                "canonical_checkpoint": winner["checkpoint"] if winner is not None else None,
                "canonical_strict_goal_ok": (
                    bool(winner["diagnostics"]["oracle_strict_ok"]) if winner is not None else False
                ),
                "canonical_rank": _canonical_rank_record(winner) if winner is not None else None,
                "selection_reason": reason,
                "candidates": [
                    {
                        "version": run["version"],
                        "channel_codebook_mode": run["channel_codebook_mode"],
                        "checkpoint": run["checkpoint"],
                        "e2_input_order_status": run["e2_input_order_status"],
                        "eligible": run["formal_cell_eligible"],
                        "selected": run is winner,
                        "rank": _canonical_rank_record(run),
                    }
                    for run in sorted(members, key=_canonical_rank, reverse=True)
                ],
            }
        )
    return canonical_cells


def _matrix_groups(
    runs: Sequence[dict[str, Any]],
    expected_arches: Sequence[str],
    expected_families: Sequence[str],
    *,
    reference_runs: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    # The promoted 2x2 design deliberately uses different token semantics:
    # image-VQ is global K=256/1024/4096, whereas channel-VQ is the grouped
    # channel-owned K=512/1024/4096 design.  Requiring a single common
    # (rates,mode) tuple would manufacture impossible "missing" cross-cells.
    target_modes = {
        "image-vq": ("global",),
        # Both are genuine channel-VQ.  Grouped is the low-rate,
        # channel-owned AR contract; global gives every channel token the full
        # codebook and may be selected when oracle expression is the priority.
        "channel-vq": ("global", "grouped"),
    }
    # Layer2 architecture and embedding D are deliberately free design
    # variables.  The requested matrix is Layer1 x VQ family, so grouping by
    # Layer2/D would manufacture cross-product cells the experiment never
    # requires.  D scaling is audited separately below.
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    references = reference_runs if reference_runs is not None else runs
    for run in references:
        if str(run["channel_codebook_mode"]) not in target_modes.get(str(run["family"]), ("global",)):
            continue
        key = (run["C"], run["seed"])
        groups.setdefault(key, [])
    for run in runs:
        if str(run["channel_codebook_mode"]) not in target_modes.get(str(run["family"]), ("global",)):
            continue
        key = (run["C"], run["seed"])
        groups[key].append(run)
    result: list[dict[str, Any]] = []
    for (latent_c, seed), members in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0] is None,
            item[0][0] or -1,
            item[0][1] or -1,
        ),
    ):
        cells: list[dict[str, Any]] = []
        missing: list[dict[str, str]] = []
        for arch in expected_arches:
            for family in expected_families:
                allowed_modes = target_modes.get(str(family), ("global",))
                matching = [
                    run
                    for run in members
                    if run["arch"] == arch
                    and run["family"] == family
                    and run["channel_codebook_mode"] in allowed_modes
                ]
                matching = sorted(matching, key=_canonical_rank, reverse=True)
                cell = {
                    "arch": arch,
                    "family": family,
                    "channel_codebook_mode": (
                        matching[0]["channel_codebook_mode"] if matching else "/".join(allowed_modes)
                    ),
                    "status": "present" if matching else "MISSING",
                    "versions": [run["version"] for run in matching],
                    "checkpoints": [run["checkpoint"] for run in matching],
                    "rates": list(matching[0]["rates"]) if matching else [],
                    "embedding_dim": matching[0]["embedding_dim"] if matching else None,
                    "layer2_arch": matching[0]["layer2_arch"] if matching else None,
                    "strict_goal_ok": (
                        bool(matching[0]["diagnostics"]["oracle_strict_ok"]) if matching else False
                    ),
                }
                cells.append(cell)
                if not matching:
                    missing.append(
                        {
                            "arch": arch,
                            "family": family,
                            "channel_codebook_mode": "/".join(allowed_modes),
                        }
                    )
        result.append(
            {
                "C": latent_c,
                "rates": [],
                "seed": seed,
                "layer2_arch": "arbitrary-per-cell",
                "channel_codebook_mode": "family-specific",
                "family_contracts": {
                    family: {
                        "channel_codebook_mode": next(
                            (
                                cell["channel_codebook_mode"]
                                for cell in cells
                                if cell["family"] == family and cell["status"] == "present"
                            ),
                            "/".join(target_modes.get(str(family), ("global",))),
                        ),
                        "rates": next(
                            (
                                cell["rates"]
                                for cell in cells
                                if cell["family"] == family and cell["rates"]
                            ),
                            [],
                        ),
                    }
                    for family in expected_families
                },
                "cells": cells,
                "missing_cells": missing,
                "complete": not missing,
            }
        )
    return result


def _embedding_scaling_groups(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare D at fixed Layer1/Layer2/family/C/K/seed using canonical runs."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if not run.get("canonical") or not run.get("formal_cell_eligible"):
            continue
        key = (
            run["arch"],
            run["layer2_arch"],
            run["family"],
            run["channel_codebook_mode"],
            run["C"],
            tuple(run["rates"]),
            run["seed"],
        )
        grouped[key].append(run)
    result: list[dict[str, Any]] = []
    for key, members in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        by_dim = {
            int(run["embedding_dim"]): run
            for run in members
            if run["embedding_dim"] is not None
        }
        dimensions = sorted(by_dim)
        if len(dimensions) < 2:
            continue
        pairs: list[dict[str, Any]] = []
        for low_dim, high_dim in zip(dimensions[:-1], dimensions[1:]):
            low = by_dim[low_dim]
            high = by_dim[high_dim]
            low_psnr = {int(row["K"]): row["psnr"] for row in low["per_K"]}
            high_psnr = {int(row["K"]): row["psnr"] for row in high["per_K"]}
            per_k: list[dict[str, Any]] = []
            for rate in low["rates"]:
                low_value = low_psnr.get(int(rate))
                high_value = high_psnr.get(int(rate))
                delta = (
                    float(high_value) - float(low_value)
                    if low_value is not None and high_value is not None
                    else None
                )
                per_k.append(
                    {
                        "K": int(rate),
                        "low_psnr": low_value,
                        "high_psnr": high_value,
                        "delta_psnr": delta,
                        "strict_increase": delta is not None and delta > 0.0,
                    }
                )
            pairs.append(
                {
                    "low_D": low_dim,
                    "high_D": high_dim,
                    "low_version": low["version"],
                    "high_version": high["version"],
                    "per_K": per_k,
                    "strict_increase_all_K": all(row["strict_increase"] for row in per_k),
                }
            )
        arch, layer2_arch, family, mode, latent_c, rates, seed = key
        result.append(
            {
                "arch": arch,
                "layer2_arch": layer2_arch,
                "family": family,
                "channel_codebook_mode": mode,
                "C": latent_c,
                "rates": list(rates),
                "seed": seed,
                "dimensions": dimensions,
                "pairs": pairs,
                "strict_increase": all(pair["strict_increase_all_K"] for pair in pairs),
            }
        )
    return result


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "MISSING"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _md_escape(value: Any) -> str:
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def _gate_text(gate: Mapping[str, Any]) -> str:
    status = str(gate.get("status", "unknown"))
    if status == "pass":
        return "PASS (embedded)"
    if status == "fail":
        return "FAIL (embedded)"
    return "UNKNOWN (missing)"


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# explore-2 nested VQ matrix",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "> Gate results below are copied only from checkpoint-embedded validation metrics. "
        "Missing gates remain UNKNOWN; diagnostics never infer a PASS.",
        "> `COLLAPSE` uses global PPL/top1/token-change for image/global VQ. Grouped "
        "channel-VQ instead uses its embedded local-round thresholds for every K. `K=C` "
        "has one structural local choice, zero index bits, and is non-promotable rather than exempt. "
        "Missing required evidence remains UNKNOWN.",
        "",
        "## Summary",
        "",
        f"- Version-level runs retained: **{summary['selected_versions']}**",
        f"- Canonical formal runs: **{summary['canonical_runs']}**",
        f"- Runs excluded from formal sender-oracle cells: **{summary['excluded_e2_contract_runs']}**",
        f"- Matching stage checkpoints: **{summary['matching_stage_checkpoints']}**",
        f"- Missing 2x2 cells: **{summary['missing_matrix_cells']}**",
        f"- Canonical runs with failed/unknown sender-oracle strict goals: **{summary['strict_goal_failures']}**",
        f"- Embedding-scaling groups: **{summary['embedding_scaling_groups']}**; failures: **{summary['embedding_scaling_failures']}**",
        f"- All-run failed/unknown strict goals (diagnostic only): **{summary['all_run_strict_goal_failures']}**",
        f"- Strict completeness: **{'PASS' if summary['strict_complete'] else 'FAIL'}**",
        "",
        "Version selection: valid `e2_input_order=img_x1` first, then "
        "`goal_best > best > latest > other`; exactly one checkpoint per embedded "
        "`(args.version, args.channel_codebook_mode)` (legacy mode defaults to `global`).",
        "",
        "Canonical selection: sender-oracle eligibility requires exact embedded "
        "`args.e2_input_order=img_x1`, grouped K>C, and receiver_phase=none; then "
        "`oracle_strict_ok`, checkpoint kind, epoch, max-K PSNR score, and version. "
        "Historical runs do not vote on strict completeness.",
        "",
        "## 2x2 matrix completeness",
        "",
    ]
    matrix_groups = report["matrix_groups"]
    if not matrix_groups:
        lines.extend(
            [
                "**MISSING: no comparable `(C, rates, seed)` group was found; no 2x2 cell is proven present.**",
                "",
            ]
        )
    else:
        expected_arches = report["expected_matrix"]["arches"]
        expected_families = report["expected_matrix"]["families"]
        headers = ["C", "seed", "family-specific contracts"] + [
            f"{arch}/{family}" for arch in expected_arches for family in expected_families
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for group in matrix_groups:
            by_cell = {(cell["arch"], cell["family"]): cell for cell in group["cells"]}
            contracts = "; ".join(
                f"{family}:{spec['channel_codebook_mode']} K={_fmt(spec['rates'])}"
                for family, spec in group["family_contracts"].items()
            )
            values = [
                _fmt(group["C"]),
                _fmt(group["seed"]),
                contracts,
            ]
            for arch in expected_arches:
                for family in expected_families:
                    cell = by_cell[(arch, family)]
                    if cell["status"] == "present":
                        gate = "strict PASS" if cell["strict_goal_ok"] else "strict FAIL/UNKNOWN"
                        values.append(
                            f"`{cell['versions'][0]}` (D={cell['embedding_dim']}, "
                            f"L2={cell['layer2_arch']}, {gate})"
                        )
                    else:
                        values.append("**MISSING canonical**")
            lines.append("| " + " | ".join(_md_escape(value) for value in values) + " |")
        lines.append("")
        for group in matrix_groups:
            if group["missing_cells"]:
                missing = ", ".join(f"{cell['arch']}/{cell['family']}" for cell in group["missing_cells"])
                lines.append(
                    f"- **MISSING** for promoted family-specific matrix `C={_fmt(group['C'])}; "
                    f"seed={_fmt(group['seed'])}`: {missing}."
                )
        lines.append("")

    lines.extend(
        [
            "## Embedding-dimension scaling",
            "",
            "| Layer1 | Layer2 | family | mode | C | K set | seed | D pair | per-K PSNR delta | status |",
            "| --- | --- | --- | --- | ---: | --- | ---: | --- | --- | --- |",
        ]
    )
    for group in report["embedding_scaling_groups"]:
        for pair in group["pairs"]:
            deltas = ", ".join(
                f"K{row['K']}:{_fmt(row['delta_psnr'])}" for row in pair["per_K"]
            )
            values = [
                group["arch"],
                group["layer2_arch"],
                group["family"],
                group["channel_codebook_mode"],
                group["C"],
                group["rates"],
                group["seed"],
                f"D{pair['low_D']}->D{pair['high_D']}",
                deltas,
                "PASS" if pair["strict_increase_all_K"] else "FAIL",
            ]
            lines.append("| " + " | ".join(_md_escape(value) for value in values) + " |")
    if not report["embedding_scaling_groups"]:
        lines.append("| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | UNKNOWN |")
    lines.append("")

    lines.extend(
        [
            "## Canonical cell decisions",
            "",
            "| C | D | rates | seed | Layer1 | Layer2 | family | channel codebook mode | canonical version | status | sender-oracle strict goal | selection reason |",
            "| ---: | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for cell in report["canonical_cells"]:
        values = [
            cell["C"], cell["embedding_dim"], cell["rates"], cell["seed"], cell["arch"], cell["layer2_arch"], cell["family"],
            cell["channel_codebook_mode"],
            cell["canonical_version"], cell["status"],
            "PASS" if cell["canonical_strict_goal_ok"] else "FAIL/UNKNOWN",
            cell["selection_reason"],
        ]
        lines.append("| " + " | ".join(_md_escape(value) for value in values) + " |")
    if not report["canonical_cells"]:
        lines.append("| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | UNKNOWN | no candidates |")
    lines.append("")

    lines.extend(
        [
            "## All version-level runs and receiver/gates",
            "",
            "| canonical | version | Layer1 | Layer2 | family | channel codebook mode | C | D | rates | seed | epoch | kind | E2 order | E2 status | checkpoint | x1 PSNR | continuous PSNR | receiver PSNR | receiver delta | pred zero | pred shuffle | condition shuffle | condition min | condition status | index acc | pred local PPL | pred local top1 | pred channel variation | pred variant channel frac | receiver audit | oracle gate | receiver gate | overall gate |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for run in report["runs"]:
        values = [
            "YES" if run["canonical"] else "no", run["version"], run["arch"], run["layer2_arch"], run["family"],
            run["channel_codebook_mode"], run["C"], run["embedding_dim"],
            run["rates"], run["seed"], run["epoch"], run["checkpoint_kind"], run["e2_input_order"],
            run["e2_input_order_status"], run["checkpoint"], run["psnr_x1"], run["psnr_continuous"],
            run["psnr_pred"], run["delta_x1_pred"], run["pred_drop_zero"], run["pred_drop_shuffle"],
            run["condition_shuffle_drop"], run["condition_shuffle_threshold"], run["condition_shuffle_status"],
            run["pred_index_accuracy"],
            run["pred_grouped_metrics"]["local_ppl"],
            run["pred_grouped_metrics"]["local_top1"],
            run["pred_grouped_metrics"]["channel_variation"],
            run["pred_grouped_metrics"]["variant_channel_frac"],
            "PASS" if run["receiver_only_audit"] == 1.0 else ("FAIL" if run["receiver_only_audit"] is not None else "UNKNOWN"),
            _gate_text(run["gates"]["oracle"]), _gate_text(run["gates"]["receiver"]), _gate_text(run["gates"]["overall"]),
        ]
        lines.append("| " + " | ".join(_md_escape(value) for value in values) + " |")
    if not report["runs"]:
        lines.append("| no | MISSING | MISSING | MISSING | global | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | LEGACY_UNKNOWN | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | UNKNOWN | MISSING | MISSING | MISSING | MISSING | MISSING | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |")
    lines.extend(
        [
            "",
            "## Per-K oracle/codebook metrics",
            "",
            "| version | Layer1 | Layer2 | family | channel codebook mode | C | D | K | PSNR | delta vs x1 | zero drop | shuffle drop | used | PPL | PPL/K | top1 | local used | local PPL | local PPL/rounds | local top1 | channel variation | variant channel frac | tokens/image | candidates/token | bits/token | bits/image | bpp | token change | token min | token status | VQ MSE | paired strict | strict min | paired gain | gain min | paired threshold status | monotonic | collapse |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for run in report["runs"]:
        for row in run["per_K"]:
            values = [
                run["version"], run["arch"], run["layer2_arch"], run["family"], run["channel_codebook_mode"],
                run["C"], run["embedding_dim"], row["K"], row["psnr"], row["delta_x1"],
                row["drop_zero"], row["drop_shuffle"], row["used"], row["ppl"], row["ppl_ratio"], row["top1"],
                row["local_used"], row["local_ppl"], row["local_ppl_ratio"], row["local_top1"],
                row["channel_variation"], row["variant_channel_frac"], row["tokens_per_image"],
                row["candidates_per_token"], row["bits_per_token"], row["bits_per_image"], row["bpp"],
                row["token_change"], row["token_change_threshold"], row["token_change_status"], row["vq_mse"],
                row["paired_strict"], row["paired_strict_threshold"],
                row["paired_gain"], row["paired_gain_threshold"], row["paired_threshold_status"],
                run["diagnostics"]["monotonic_status"], row["collapse_status"],
            ]
            lines.append("| " + " | ".join(_md_escape(value) for value in values) + " |")
    if not any(run["per_K"] for run in report["runs"]):
        lines.append("| MISSING | MISSING | MISSING | global | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | UNKNOWN | MISSING | MISSING | MISSING | MISSING | MISSING | UNKNOWN | unknown | unknown |")

    lines.extend(["", "## Diagnostics", ""])
    diagnostic_rows = 0
    for run in report["runs"]:
        diagnostics = run["diagnostics"]
        messages: list[str] = []
        if diagnostics["non_monotonic_pairs"]:
            pairs = ", ".join(f"K{pair['low_K']}->K{pair['high_K']}" for pair in diagnostics["non_monotonic_pairs"])
            messages.append(f"**NON_MONOTONIC**: {pairs}")
        for collapse in diagnostics["collapse_triggers"]:
            messages.append(f"**COLLAPSE K{collapse['K']}**: {', '.join(collapse['triggers'])}")
        if diagnostics["missing_metrics"]:
            messages.append("**MISSING metrics**: " + ", ".join(diagnostics["missing_metrics"]))
        if diagnostics["missing_thresholds"]:
            messages.append("**MISSING thresholds**: " + ", ".join(diagnostics["missing_thresholds"]))
        messages.append("canonical: " + run["canonical_selection_reason"])
        messages.extend(diagnostics["warnings"])
        if not messages:
            messages.append("No reported threshold trigger; this is not an inferred gate PASS")
        lines.append(f"- `{run['version']}`: " + "; ".join(dict.fromkeys(messages)) + ".")
        diagnostic_rows += 1
    if diagnostic_rows == 0:
        lines.append("- **MISSING**: no selected run is available for diagnostics.")

    if report["discovery"]["load_errors"]:
        lines.extend(["", "## Checkpoint load errors", ""])
        for item in report["discovery"]["load_errors"]:
            lines.append(f"- `{item['checkpoint']}`: `{item['error']}`")
    lines.append("")
    return "\n".join(lines)


def build_report(
    roots: Sequence[Path],
    expected_arches: Sequence[str],
    expected_families: Sequence[str],
) -> dict[str, Any]:
    candidates, discovery = discover_candidates(roots)
    selected, discarded = select_versions(candidates)
    runs = [_run_from_candidate(candidate) for candidate in selected]
    runs.sort(
        key=lambda run: (
            str(run["arch"]), str(run["family"]), str(run["channel_codebook_mode"]),
            run["C"] if run["C"] is not None else -1,
            tuple(run["rates"]), run["seed"] if run["seed"] is not None else -1, run["version"],
        )
    )
    canonical_cells = select_canonical_cells(runs)
    canonical_runs = [run for run in runs if run["canonical"]]
    embedding_scaling_groups = _embedding_scaling_groups(canonical_runs)
    embedding_scaling_failures = sum(
        not group["strict_increase"] for group in embedding_scaling_groups
    )
    matrix_groups = _matrix_groups(
        canonical_runs,
        expected_arches,
        expected_families,
        reference_runs=runs,
    )
    missing_matrix_cells = sum(len(group["missing_cells"]) for group in matrix_groups)
    if not matrix_groups:
        # No group means none of the expected 2x2 evidence exists.
        missing_matrix_cells = len(expected_arches) * len(expected_families)
    promoted_versions = {
        cell["versions"][0]
        for group in matrix_groups
        for cell in group["cells"]
        if cell["status"] == "present" and cell["versions"]
    }
    promoted_canonical_runs = [
        run for run in canonical_runs if run["version"] in promoted_versions
    ]
    strict_goal_failures = sum(
        not run["diagnostics"]["oracle_strict_ok"] for run in promoted_canonical_runs
    )
    all_run_strict_goal_failures = sum(not run["diagnostics"]["strict_goal_ok"] for run in runs)
    matrix_complete = bool(matrix_groups) and all(group["complete"] for group in matrix_groups)
    strict_complete = (
        bool(promoted_canonical_runs)
        and matrix_complete
        and strict_goal_failures == 0
        and not discovery["load_errors"]
    )
    discovery["selected_versions"] = len(runs)
    discovery["discarded_same_version"] = discarded
    report = {
        "schema_version": 5,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": STAGE,
        "search_roots": [_display_path(root) for root in roots],
        "selection_policy": (
            "valid args.e2_input_order=img_x1 first; then goal_best > best > latest > other; "
            "one checkpoint per embedded (args.version, args.channel_codebook_mode); "
            "missing legacy mode defaults to global"
        ),
        "canonical_policy": (
            "exact (C, D, rates, seed, layer1_arch, layer2_arch, family, channel_codebook_mode) cell and "
            "sender-oracle eligibility (args.e2_input_order=img_x1, grouped K>C, "
            "receiver_phase=none); oracle_strict_ok first; then "
            "goal_best>best>latest>other; epoch; max-K PSNR score; version"
        ),
        "expected_matrix": {"arches": list(expected_arches), "families": list(expected_families)},
        "summary": {
            "pth_files_scanned": discovery["pth_files_scanned"],
            "matching_stage_checkpoints": discovery["matching_stage_checkpoints"],
            "selected_versions": len(runs),
            "canonical_runs": len(promoted_canonical_runs),
            "excluded_e2_contract_runs": sum(not run["formal_cell_eligible"] for run in runs),
            "matrix_groups": len(matrix_groups),
            "missing_matrix_cells": missing_matrix_cells,
            "strict_goal_failures": strict_goal_failures,
            "all_run_strict_goal_failures": all_run_strict_goal_failures,
            "embedding_scaling_groups": len(embedding_scaling_groups),
            "embedding_scaling_failures": embedding_scaling_failures,
            "strict_complete": strict_complete,
        },
        "matrix_groups": matrix_groups,
        "embedding_scaling_groups": embedding_scaling_groups,
        "canonical_cells": canonical_cells,
        "runs": runs,
        "discovery": discovery,
    }
    return report


def _csv_names(value: str, option: str) -> tuple[str, ...]:
    names = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not names:
        raise ValueError(f"{option} must contain at least one comma-separated name")
    if len(set(names)) != len(names):
        raise ValueError(f"{option} contains duplicates: {value!r}")
    return names


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively summarize explore2_layer2_nested_vq checkpoints. "
            "The default root excludes checkpoints-vq-smoke so smoke evidence cannot contaminate the formal matrix."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-root",
        "--root",
        dest="checkpoint_roots",
        action="append",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "Checkpoint file/directory to scan recursively; repeat for multiple roots "
            f"(default when omitted: {DEFAULT_CHECKPOINT_ROOT})"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-name", default="vq_matrix.json")
    parser.add_argument("--markdown-name", default="vq_matrix.md")
    parser.add_argument("--expected-arches", default=",".join(DEFAULT_EXPECTED_ARCHES))
    parser.add_argument("--expected-families", default=",".join(DEFAULT_EXPECTED_FAMILIES))
    parser.add_argument(
        "--strict-complete",
        action="store_true",
        help=(
            "Exit nonzero after writing reports if any canonical 2x2 cell is missing "
            "or any canonical run lacks passing goals/diagnostics"
        ),
    )
    args = parser.parse_args(argv)
    args.checkpoint_roots = getattr(args, "checkpoint_roots", None) or [DEFAULT_CHECKPOINT_ROOT]
    try:
        args.expected_arches = _csv_names(args.expected_arches, "--expected-arches")
        args.expected_families = _csv_names(args.expected_families, "--expected-families")
    except ValueError as error:
        parser.error(str(error))
    for name, option in ((args.json_name, "--json-name"), (args.markdown_name, "--markdown-name")):
        if Path(name).name != name or not name:
            parser.error(f"{option} must be a plain filename, got {name!r}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [path.expanduser().resolve() for path in args.checkpoint_roots]
    output_dir = args.output_dir.expanduser().resolve()
    report = build_report(roots, args.expected_arches, args.expected_families)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / args.json_name
    markdown_path = output_dir / args.markdown_name
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    summary = report["summary"]
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print(
        "selected_versions={selected_versions} canonical_runs={canonical_runs} "
        "excluded_e2_contract_runs={excluded_e2_contract_runs} "
        "missing_matrix_cells={missing_matrix_cells} canonical_strict_goal_failures={strict_goal_failures} "
        "strict_complete={strict_complete}".format(**summary)
    )
    if args.strict_complete and not summary["strict_complete"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
