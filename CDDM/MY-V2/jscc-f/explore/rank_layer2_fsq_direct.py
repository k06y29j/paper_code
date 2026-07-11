#!/usr/bin/env python3
"""Rank direct Layer2-FSQ checkpoints by rate, reconstruction, and code relevance."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=str(THIS_DIR / "checkpoints-direct" / "**" / "*layer2_fsq_direct*_best.pth"),
    )
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.checkpoint]
    paths.extend(Path(path) for path in sorted(glob.glob(args.glob_pattern, recursive=True)))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise FileNotFoundError("no direct Layer2-FSQ checkpoint matched")

    rows: list[dict] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        if str(payload.get("stage", "")) != "layer2_fsq_direct":
            continue
        metrics = payload.get("metrics", {})
        cfg = payload.get("args", {})
        tokenizer = payload.get("tokenizer", {})
        rows.append(
            {
                "checkpoint": str(path),
                "arch": cfg.get("arch"),
                "epoch": int(payload.get("epoch", -1)),
                "levels": tokenizer.get("fsq_levels", cfg.get("fsq_levels")),
                "vocab_size": int(tokenizer.get("vocab_size", -1)),
                "normalizer": tokenizer.get("normalizer"),
                "codec_init": cfg.get("codec_init"),
                "combiner_mode": cfg.get("combiner_mode"),
                "goal_eligible": bool(metrics.get("goal_eligible", False)),
                "psnr_x1": float(metrics.get("psnr_x1", float("nan"))),
                "psnr_final": float(metrics.get("psnr_final", float("-inf"))),
                "delta_x1": float(metrics.get("delta_x1", float("nan"))),
                "psnr_continuous": float(metrics.get("psnr_continuous", float("nan"))),
                "gap_continuous": float(metrics.get("gap_continuous", float("nan"))),
                "code_perplexity": float(metrics.get("code_perplexity", float("nan"))),
                "code_used": float(metrics.get("code_used", float("nan"))),
                "level_entropy_bits_mean": float(metrics.get("level_entropy_bits_mean", float("nan"))),
                "empirical_bpp": float(metrics.get("empirical_bpp", float("nan"))),
                "drop_zero": float(metrics.get("drop_zero", float("nan"))),
                "drop_shuffle": float(metrics.get("drop_shuffle", float("nan"))),
                "fsq_mse": float(metrics.get("fsq_mse", float("nan"))),
            }
        )

    rows.sort(key=lambda row: (row["goal_eligible"], row["psnr_final"]), reverse=True)
    print(
        "goal arch epoch levels K norm init combiner psnr_x1 psnr_final delta continuous gap_q "
        "ppl used Hlevel bpp drop0 dropshuf fsq_mse checkpoint"
    )
    for row in rows:
        print(
            f"{str(row['goal_eligible']):5} {str(row['arch']):4} {row['epoch']:5d} "
            f"{str(row['levels']):14} {row['vocab_size']:5d} {str(row['normalizer']):8} "
            f"{str(row['codec_init']):10} {str(row['combiner_mode']):8} "
            f"{row['psnr_x1']:8.4f} {row['psnr_final']:10.4f} {row['delta_x1']:7.4f} "
            f"{row['psnr_continuous']:10.4f} {row['gap_continuous']:7.4f} "
            f"{row['code_perplexity']:8.2f} {row['code_used']:5.0f} "
            f"{row['level_entropy_bits_mean']:6.3f} {row['empirical_bpp']:7.4f} "
            f"{row['drop_zero']:7.4f} {row['drop_shuffle']:8.4f} {row['fsq_mse']:8.5f} "
            f"{row['checkpoint']}",
            flush=True,
        )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
