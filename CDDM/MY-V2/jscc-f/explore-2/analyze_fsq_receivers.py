#!/usr/bin/env python3
"""Audit explore-2 FSQ receiver checkpoints without running the models.

Only checkpoint-embedded ``args``, ``oracle_args``, ``metrics``, topology, and
receiver-contract metadata are used.  Tensor storages are mmap'ed and mapped
to the meta device so scanning the large training checkpoints does not
materialize model or optimizer tensors in host memory.

One checkpoint is selected per experiment version using the fixed precedence
``goal_best > best > latest > other``.  A selected checkpoint becomes the
single strict canonical result only when every receiver, usefulness, and
no-leakage evidence gate below is explicitly present and passes.
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
from typing import Any, Mapping, Sequence


STAGE = "explore2_fsq_receiver"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "checkpoints-receiver"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results-receiver"
KIND_PRIORITY = {"other": 0, "latest": 1, "best": 2, "goal_best": 3}

STRICT_THRESHOLDS = {
    "receiver_only_audit": 1.0,
    "condition_shuffle_drop": 0.1,
    "pred_drop_zero": 0.1,
    "pred_drop_shuffle": 0.1,
    "delta_oracle": 0.8,
    "oracle_drop_zero": 0.5,
    "oracle_drop_shuffle": 0.5,
    "delta_x1": 0.5,
}

METRIC_KEYS = (
    "psnr_x1",
    "psnr_oracle",
    "psnr_pred",
    "delta_x1",
    "delta_oracle",
    "gap_oracle",
    "loss_q",
    "q_mse_hard",
    "index_accuracy",
    "psnr_condition_shuffle",
    "condition_shuffle_drop",
    "pred_drop_zero",
    "pred_drop_shuffle",
    "oracle_drop_zero",
    "oracle_drop_shuffle",
    "receiver_only_audit",
    "goal_met",
)


@dataclass(frozen=True)
class Candidate:
    path: Path
    display_path: str
    kind: str
    priority: int
    version: str
    epoch: int | None
    args: dict[str, Any]
    oracle_args: dict[str, Any]
    metrics: dict[str, Any]
    topology: dict[str, Any]
    contract: dict[str, Any]


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
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 0.0):
        return False
    if value in (1, 1.0):
        return True
    return None


def _levels(value: Any, dimension: int | None) -> tuple[int, ...]:
    if isinstance(value, str):
        pieces: Sequence[Any] = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        pieces = value
    elif value is not None and dimension is not None:
        pieces = [value] * dimension
    else:
        return ()
    parsed: list[int] = []
    for piece in pieces:
        try:
            number = int(str(piece).strip())
        except (TypeError, ValueError):
            return ()
        if number <= 0:
            return ()
        parsed.append(number)
    if dimension is not None and len(parsed) == 1 and dimension > 1:
        parsed *= dimension
    if dimension is not None and len(parsed) != dimension:
        return ()
    return tuple(parsed)


def _kind(path: Path) -> tuple[str, int]:
    if path.name.endswith("_goal_best.pth"):
        return "goal_best", KIND_PRIORITY["goal_best"]
    if path.name.endswith("_best.pth"):
        return "best", KIND_PRIORITY["best"]
    if path.name.endswith("_latest.pth"):
        return "latest", KIND_PRIORITY["latest"]
    return "other", KIND_PRIORITY["other"]


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _load_payload(path: Path) -> Any:
    """Load Python metadata while never allocating checkpoint tensor storage."""
    import torch

    kwargs = {"map_location": "meta", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError, ValueError) as error:
        # mmap is unavailable for legacy non-zip files and older torch builds;
        # map_location=meta still prevents tensor storage materialization.
        if "mmap" not in str(error).lower() and not isinstance(error, TypeError):
            raise
        return torch.load(path, **kwargs)


def _paths(roots: Sequence[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved.is_file() and resolved.suffix == ".pth":
            found.add(resolved)
        elif resolved.is_dir():
            found.update(path.resolve() for path in resolved.rglob("*.pth") if path.is_file())
    return sorted(found, key=lambda path: path.as_posix())


def discover(roots: Sequence[Path]) -> tuple[list[Candidate], dict[str, Any]]:
    candidates: list[Candidate] = []
    errors: list[dict[str, str]] = []
    ignored = 0
    paths = _paths(roots)
    for path in paths:
        try:
            payload = _load_payload(path)
        except Exception as error:  # Keep a corrupt snapshot visible in the report.
            errors.append({"checkpoint": _display_path(path), "error": f"{type(error).__name__}: {error}"})
            continue
        try:
            if not isinstance(payload, Mapping) or str(payload.get("stage", "")) != STAGE:
                ignored += 1
                continue
            args = _plain_mapping(payload.get("args"))
            kind, priority = _kind(path)
            version = str(args.get("version", "")).strip() or path.parent.name or path.stem
            candidates.append(
                Candidate(
                    path=path,
                    display_path=_display_path(path),
                    kind=kind,
                    priority=priority,
                    version=version,
                    epoch=_integer(payload.get("epoch")),
                    args=args,
                    oracle_args=_plain_mapping(payload.get("oracle_args")),
                    metrics=_plain_mapping(payload.get("metrics")),
                    topology=_plain_mapping(payload.get("receiver_topology")),
                    contract=_plain_mapping(payload.get("receiver_contract")),
                )
            )
        finally:
            del payload
            gc.collect()
    return candidates, {
        "pth_files_scanned": len(paths),
        "matching_stage_checkpoints": len(candidates),
        "ignored_non_stage_checkpoints": ignored,
        "load_errors": errors,
        "tensor_load_policy": "torch.load(map_location=meta, mmap=True); meta-only fallback for legacy files",
    }


def select_versions(candidates: Sequence[Candidate]) -> tuple[list[Candidate], list[dict[str, Any]]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.version].append(candidate)
    winners: list[Candidate] = []
    discarded: list[dict[str, Any]] = []
    for version, choices in sorted(grouped.items()):
        ordered = sorted(
            choices,
            key=lambda item: (
                item.priority,
                item.epoch if item.epoch is not None else -1,
                _finite_float(item.metrics.get("psnr_pred")) or float("-inf"),
                item.display_path,
            ),
            reverse=True,
        )
        winner = ordered[0]
        winners.append(winner)
        for item in ordered[1:]:
            discarded.append(
                {
                    "version": version,
                    "checkpoint": item.display_path,
                    "checkpoint_kind": item.kind,
                    "epoch": item.epoch,
                    "selected_checkpoint": winner.display_path,
                    "reason": "lower goal_best > best > latest > other selection rank within version",
                }
            )
    return winners, discarded


def _metric(metrics: Mapping[str, Any], key: str) -> float | None:
    return _finite_float(metrics.get(key))


def _metric_or_difference(
    metrics: Mapping[str, Any],
    key: str,
    high: str,
    low: str,
) -> tuple[float | None, str]:
    explicit = _metric(metrics, key)
    if explicit is not None:
        return explicit, f"metrics.{key}"
    high_value = _metric(metrics, high)
    low_value = _metric(metrics, low)
    if high_value is not None and low_value is not None:
        return high_value - low_value, f"derived from metrics.{high}-metrics.{low}"
    return None, "missing"


def _topology(candidate: Candidate) -> dict[str, Any]:
    independent = _bool(candidate.topology.get("independent_receiver_d2"))
    isolated = _bool(candidate.topology.get("receiver_combiner_isolated"))
    arg_independent = _bool(candidate.args.get("independent_receiver_d2"))
    explicit = independent is not None and isolated is not None and arg_independent is not None
    consistent = explicit and independent == arg_independent
    if independent is True:
        mode = "independent-d2"
    elif independent is False and isolated is True:
        mode = "shared-d2/isolated-combiner"
    elif independent is False:
        mode = "shared-sender"
    else:
        mode = "unknown"
    return {
        "mode": mode,
        "explicit": explicit,
        "consistent_with_args": bool(consistent),
        "independent_receiver_d2": independent,
        "receiver_combiner_isolated": isolated,
    }


def _contract(candidate: Candidate) -> dict[str, Any]:
    inputs = candidate.contract.get("inputs")
    forbidden = candidate.contract.get("forbidden_inputs")
    output = candidate.contract.get("output")
    input_list = [str(item) for item in inputs] if isinstance(inputs, (list, tuple)) else []
    forbidden_list = [str(item) for item in forbidden] if isinstance(forbidden, (list, tuple)) else []
    allowed_inputs = {"z1", "x1"}
    required_forbidden = {"img", "z2", "q2", "oracle_indices"}
    valid = (
        bool(input_list)
        and set(input_list).issubset(allowed_inputs)
        and required_forbidden.issubset(set(forbidden_list))
        and str(output) == "q2_hat"
        and str(candidate.args.get("route", "")) in {"direct_q", "parallel_index", "joint_index"}
        and str(candidate.args.get("condition_mode", "")) in {"z1", "x1", "z1_x1"}
    )
    return {
        "inputs": input_list,
        "forbidden_inputs": forbidden_list,
        "output": output,
        "explicit_no_sender_leakage_contract": bool(valid),
    }


def _gate(value: float | None, threshold: float, *, present: bool = True) -> dict[str, Any]:
    available = value is not None and present
    return {
        "value": value,
        "threshold": threshold,
        "present": bool(available),
        "pass": bool(available and value >= threshold),
    }


def _exact_gate(value: float | None, expected: float) -> dict[str, Any]:
    available = value is not None
    return {
        "value": value,
        "threshold": expected,
        "present": available,
        "pass": bool(available and value == expected),
    }


def _run(candidate: Candidate) -> dict[str, Any]:
    dimension = _integer(candidate.oracle_args.get("fsq_d"))
    levels = _levels(candidate.oracle_args.get("fsq_levels"), dimension)
    codebook_capacity = math.prod(levels) if levels else None
    metrics = {key: _metric(candidate.metrics, key) for key in METRIC_KEYS}
    delta_x1, delta_x1_source = _metric_or_difference(
        candidate.metrics, "delta_x1", "psnr_pred", "psnr_x1"
    )
    delta_oracle, delta_oracle_source = _metric_or_difference(
        candidate.metrics, "delta_oracle", "psnr_oracle", "psnr_x1"
    )
    metrics["delta_x1"] = delta_x1
    metrics["delta_oracle"] = delta_oracle
    topology = _topology(candidate)
    contract = _contract(candidate)
    known_v3_sender_bug = candidate.version == "cnn-fsq-k4913-joint-predictable-v3"

    gates = {
        "receiver_only_audit": _exact_gate(metrics["receiver_only_audit"], 1.0),
        # Presence is intentionally mandatory.  Old snapshots cannot infer a
        # condition-shuffle result from any other ablation.
        "condition_shuffle_drop": _gate(metrics["condition_shuffle_drop"], 0.1),
        "pred_drop_zero": _gate(metrics["pred_drop_zero"], 0.1),
        "pred_drop_shuffle": _gate(metrics["pred_drop_shuffle"], 0.1),
        "delta_oracle": _gate(delta_oracle, 0.8),
        "oracle_drop_zero": _gate(metrics["oracle_drop_zero"], 0.5),
        "oracle_drop_shuffle": _gate(metrics["oracle_drop_shuffle"], 0.5),
        "delta_x1": _gate(delta_x1, 0.5),
        "topology_contract_clear": {
            "value": bool(
                topology["explicit"]
                and topology["consistent_with_args"]
                and contract["explicit_no_sender_leakage_contract"]
            ),
            "threshold": True,
            "present": bool(candidate.topology and candidate.contract),
            "pass": bool(
                topology["explicit"]
                and topology["consistent_with_args"]
                and contract["explicit_no_sender_leakage_contract"]
            ),
        },
        "historical_sender_integrity": {
            "value": not known_v3_sender_bug,
            "threshold": True,
            "present": True,
            "pass": not known_v3_sender_bug,
        },
    }
    failures = [name for name, gate in gates.items() if not gate["pass"]]
    diagnostics: list[str] = []
    if known_v3_sender_bug:
        diagnostics.append(
            "KNOWN_INVALID: v3 requested joint sender training while E2/FSQ remained frozen; diagnostic only"
        )
    if metrics["condition_shuffle_drop"] is None:
        diagnostics.append("MISSING: condition_shuffle_drop; legacy no-condition audit cannot be strict")
    if not topology["explicit"]:
        diagnostics.append("MISSING: explicit receiver_topology metadata")
    if not contract["explicit_no_sender_leakage_contract"]:
        diagnostics.append("INVALID_OR_MISSING: explicit receiver no-sender-leakage contract")
    if not levels:
        diagnostics.append("INVALID_OR_MISSING: FSQ levels/dimension contract")
    if not diagnostics:
        diagnostics.append("checkpoint metadata contracts are explicit")

    return {
        "version": candidate.version,
        "checkpoint": candidate.display_path,
        "checkpoint_kind": candidate.kind,
        "epoch": candidate.epoch,
        "route": candidate.args.get("route"),
        "condition_mode": candidate.args.get("condition_mode"),
        "hard_fsq": _bool(candidate.args.get("hard_fsq")),
        "arch": candidate.oracle_args.get("arch"),
        "fsq_dimension": dimension,
        "levels": list(levels),
        "K": codebook_capacity,
        "topology": topology,
        "receiver_combiner": candidate.args.get("receiver_combiner"),
        "finetune_d2": _bool(candidate.args.get("finetune_d2")),
        "joint_sender": {
            "requested": _bool(candidate.args.get("joint_predictable_oracle")),
            "known_v3_sender_frozen_bug": known_v3_sender_bug,
        },
        "contract": contract,
        "metrics": metrics,
        "metric_sources": {
            "delta_x1": delta_x1_source,
            "delta_oracle": delta_oracle_source,
        },
        "strict": {
            "eligible": not failures and bool(levels),
            "canonical": False,
            "gates": gates,
            "failures": failures + ([] if levels else ["fsq_levels_contract"]),
        },
        "diagnostics": diagnostics,
    }


def _canonicalize(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [run for run in runs if run["strict"]["eligible"]]
    if not eligible:
        return None
    eligible.sort(
        key=lambda run: (
            KIND_PRIORITY.get(str(run["checkpoint_kind"]), -1),
            run["metrics"]["psnr_pred"] if run["metrics"]["psnr_pred"] is not None else float("-inf"),
            run["epoch"] if run["epoch"] is not None else -1,
            run["version"],
        ),
        reverse=True,
    )
    canonical = eligible[0]
    canonical["strict"]["canonical"] = True
    return canonical


def build_report(roots: Sequence[Path]) -> dict[str, Any]:
    candidates, discovery = discover(roots)
    selected, discarded = select_versions(candidates)
    runs = [_run(candidate) for candidate in selected]
    runs.sort(key=lambda run: (str(run["arch"]), run["K"] or -1, run["version"]))
    canonical = _canonicalize(runs)
    discovery["selected_versions"] = len(runs)
    discovery["discarded_same_version"] = discarded
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": STAGE,
        "search_roots": [_display_path(path) for path in roots],
        "selection_policy": "one checkpoint per args.version: goal_best > best > latest > other; then epoch and psnr_pred",
        "strict_policy": {
            "thresholds": STRICT_THRESHOLDS,
            "additional_requirements": [
                "explicit and args-consistent receiver_topology",
                "explicit receiver contract with inputs limited to z1/x1 and img/z2/q2/oracle_indices forbidden",
                "valid FSQ dimension/levels",
                "historical cnn-fsq-k4913-joint-predictable-v3 sender-freeze checkpoint is diagnostic-only",
            ],
            "canonical_selection": "strict-eligible first; goal_best > best > latest > other; then psnr_pred and epoch",
        },
        "summary": {
            "pth_files_scanned": discovery["pth_files_scanned"],
            "matching_stage_checkpoints": discovery["matching_stage_checkpoints"],
            "selected_versions": len(runs),
            "strict_eligible_versions": sum(run["strict"]["eligible"] for run in runs),
            "strict_canonical_checkpoint": canonical["checkpoint"] if canonical else None,
            "strict_complete": canonical is not None,
            "load_errors": len(discovery["load_errors"]),
        },
        "strict_canonical": canonical,
        "runs": runs,
        "discovery": discovery,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "MISSING"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value) if value else "MISSING"
    return str(value)


def _md(value: Any) -> str:
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# FSQ receiver checkpoint audit",
        "",
        "> Metrics come only from checkpoint-embedded metadata. Model and optimizer tensors were mapped to `meta`; no dataset re-evaluation was performed.",
        "",
        "## Outcome",
        "",
        f"- Strict complete: **{str(summary['strict_complete']).upper()}**",
        f"- Selected versions: {summary['selected_versions']}; strict eligible: {summary['strict_eligible_versions']}.",
        f"- Strict canonical checkpoint: `{summary['strict_canonical_checkpoint'] or 'MISSING'}`.",
        "- Per-version selection: `goal_best > best > latest > other`.",
        "",
        "## Route and contract",
        "",
        "| Version | Kind | Epoch | Route | Condition | Arch | Levels | K | D2 topology | Joint sender | Contract |",
        "|---|---:|---:|---|---|---|---:|---:|---|---|---|",
    ]
    for run in report["runs"]:
        joint = run["joint_sender"]["requested"]
        if run["joint_sender"]["known_v3_sender_frozen_bug"]:
            joint_text = "INVALID-v3-frozen"
        elif joint is True:
            joint_text = "yes"
        elif joint is False:
            joint_text = "no"
        else:
            joint_text = "MISSING"
        values = [
            run["version"], run["checkpoint_kind"], run["epoch"], run["route"],
            run["condition_mode"], run["arch"], run["levels"], run["K"],
            run["topology"]["mode"], joint_text,
            run["contract"]["explicit_no_sender_leakage_contract"],
        ]
        lines.append("| " + " | ".join(_md(value) for value in values) + " |")
    if not report["runs"]:
        lines.append("| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | FAIL |")

    lines.extend(
        [
            "",
            "## Reconstruction and prediction",
            "",
            "| Version | PSNR x1 | PSNR oracle | PSNR pred | Delta oracle | Delta pred-x1 | Oracle gap | q MSE hard | q loss | Index acc |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in report["runs"]:
        m = run["metrics"]
        values = [
            run["version"], m["psnr_x1"], m["psnr_oracle"], m["psnr_pred"],
            m["delta_oracle"], m["delta_x1"], m["gap_oracle"], m["q_mse_hard"],
            m["loss_q"], m["index_accuracy"],
        ]
        lines.append("| " + " | ".join(_md(value) for value in values) + " |")

    lines.extend(
        [
            "",
            "## Ablations and receiver audit",
            "",
            "| Version | Condition-shuffle PSNR | Condition drop | Pred zero drop | Pred shuffle drop | Oracle zero drop | Oracle shuffle drop | Receiver-only audit |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in report["runs"]:
        m = run["metrics"]
        values = [
            run["version"], m["psnr_condition_shuffle"], m["condition_shuffle_drop"],
            m["pred_drop_zero"], m["pred_drop_shuffle"], m["oracle_drop_zero"],
            m["oracle_drop_shuffle"], m["receiver_only_audit"],
        ]
        lines.append("| " + " | ".join(_md(value) for value in values) + " |")

    lines.extend(
        [
            "",
            "## Strict gates",
            "",
            "Thresholds: audit=1, condition/pred drops >=0.1 dB, oracle delta >=0.8 dB, oracle drops >=0.5 dB, predicted delta >=0.5 dB, with explicit topology and no-leakage contract.",
            "",
            "| Version | Eligible | Canonical | Failures |",
            "|---|---|---|---|",
        ]
    )
    for run in report["runs"]:
        failures = ", ".join(run["strict"]["failures"]) or "none"
        lines.append(
            "| "
            + " | ".join(
                _md(value)
                for value in (
                    run["version"], run["strict"]["eligible"],
                    run["strict"]["canonical"], failures,
                )
            )
            + " |"
        )

    lines.extend(["", "## Diagnostics", ""])
    if report["runs"]:
        for run in report["runs"]:
            lines.append(
                f"- `{run['version']}` (`{run['checkpoint']}`): "
                + "; ".join(run["diagnostics"])
                + "."
            )
    else:
        lines.append("- MISSING: no matching FSQ receiver checkpoint.")
    if report["discovery"]["load_errors"]:
        lines.extend(["", "## Checkpoint load errors", ""])
        for item in report["discovery"]["load_errors"]:
            lines.append(f"- `{item['checkpoint']}`: `{item['error']}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit explore-2 FSQ receiver checkpoints from embedded metadata only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-root",
        "--root",
        dest="checkpoint_roots",
        action="append",
        type=Path,
        default=argparse.SUPPRESS,
        help="Checkpoint file/directory to scan recursively; repeat for multiple roots.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-name", default="fsq_receivers.json")
    parser.add_argument("--markdown-name", default="fsq_receivers.md")
    parser.add_argument(
        "--strict-complete",
        action="store_true",
        help="Write both reports, then exit 2 unless a strict canonical receiver checkpoint exists.",
    )
    args = parser.parse_args(argv)
    args.checkpoint_roots = getattr(args, "checkpoint_roots", None) or [DEFAULT_CHECKPOINT_ROOT]
    for name, option in ((args.json_name, "--json-name"), (args.markdown_name, "--markdown-name")):
        if not name or Path(name).name != name:
            parser.error(f"{option} must be a plain filename, got {name!r}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [path.expanduser().resolve() for path in args.checkpoint_roots]
    output_dir = args.output_dir.expanduser().resolve()
    report = build_report(roots)
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
        "selected_versions={selected_versions} strict_eligible_versions={strict_eligible_versions} "
        "strict_complete={strict_complete} load_errors={load_errors}".format(**summary)
    )
    if args.strict_complete and not summary["strict_complete"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
