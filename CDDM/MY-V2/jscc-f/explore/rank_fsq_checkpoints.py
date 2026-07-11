#!/usr/bin/env python3
"""Rank FSQ tokenizer checkpoints by reconstruction *and* q3 relevance."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint path; may be repeated.")
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=str(THIS_DIR / "checkpoints" / "**" / "*stage3_fsq_tokenizer*_best.pth"),
    )
    parser.add_argument("--min-drop-zero", type=float, default=0.5)
    parser.add_argument("--min-drop-shuffle", type=float, default=0.5)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.checkpoint]
    paths.extend(Path(path) for path in sorted(glob.glob(args.glob_pattern, recursive=True)))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise FileNotFoundError("no checkpoint matched")

    rows: list[dict] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        metrics = payload.get("metrics", {})
        meta = payload.get("tokenizer", {})
        cfg = payload.get("args", {})
        drop_zero = float(metrics.get("drop_zero", float("-inf")))
        drop_shuffle = float(metrics.get("drop_shuffle", float("-inf")))
        eligible = drop_zero >= float(args.min_drop_zero) and drop_shuffle >= float(args.min_drop_shuffle)
        rows.append(
            {
                "checkpoint": str(path),
                "epoch": int(payload.get("epoch", -1)),
                "levels": meta.get("fsq_levels", cfg.get("fsq_levels")),
                "fsq_d": meta.get("fsq_d", cfg.get("fsq_d")),
                "normalizer": cfg.get("fsq_normalizer", "group"),
                "usage_objective": cfg.get("usage_objective", "uniform_kl"),
                "lambda_usage": float(cfg.get("lambda_usage", 0.0)),
                "fixed_bits_per_image": meta.get("fixed_bits_per_image"),
                "psnr_final": float(metrics.get("psnr_final", float("-inf"))),
                "psnr_x1": float(metrics.get("psnr_x1", float("nan"))),
                "code_perplexity": float(metrics.get("code_perplexity", float("nan"))),
                "code_used": float(metrics.get("code_used", float("nan"))),
                "drop_zero": drop_zero,
                "drop_shuffle": drop_shuffle,
                "eligible": eligible,
            }
        )

    rows.sort(key=lambda row: (row["eligible"], row["psnr_final"], min(row["drop_zero"], row["drop_shuffle"])), reverse=True)
    print("eligible epoch levels norm objective psnr_final psnr_x1 perplexity used drop_zero drop_shuffle checkpoint")
    for row in rows:
        print(
            f"{str(row['eligible']):8} {row['epoch']:5d} {str(row['levels']):14} "
            f"{row['normalizer']:15} {row['usage_objective']:13} {row['psnr_final']:10.4f} "
            f"{row['psnr_x1']:7.4f} {row['code_perplexity']:10.2f} {row['code_used']:6.0f} "
            f"{row['drop_zero']:9.4f} {row['drop_shuffle']:12.4f} {row['checkpoint']}",
            flush=True,
        )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
