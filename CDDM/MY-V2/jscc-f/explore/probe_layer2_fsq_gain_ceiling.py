#!/usr/bin/env python3
"""Measure how much a trained blend gate hides direct Layer2-FSQ gain.

For each direct checkpoint this probe keeps E1/E2/FSQ/D2/inner-combiner fixed
and evaluates four increasingly optimistic blend choices:

* the learned scalar alpha stored in the checkpoint;
* alpha=1 (the unattenuated inner combiner output);
* one validation-set alpha selected on a fixed grid;
* an analytic per-image alpha oracle (free side information, upper probe only).

The same global alpha chosen by the normal quantized path is also applied to
zero-code and shuffled-code paths.  This prevents an alpha search from making
the code-relevance ablation incomparable.  These are gating/decoder ceilings,
not information-theoretic rate-distortion bounds.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--alpha-step", type=float, default=0.02)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--bn-calibration-batches", type=int, default=-1)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()
    if not 0.0 < float(args.alpha_step) <= 1.0:
        raise ValueError("--alpha-step must be in (0,1]")
    if int(args.bn_calibration_batches) < -1:
        raise ValueError("--bn-calibration-batches must be -1 or non-negative")
    return args


def build_from_checkpoint(
    checkpoint_path: str,
    cli: argparse.Namespace,
) -> tuple[dict, argparse.Namespace, direct.DirectBundle, object, object, torch.device, dict]:
    path = Path(base.resolve_path(checkpoint_path))
    payload = torch.load(path, map_location="cpu")
    if str(payload.get("stage", "")) != "layer2_fsq_direct":
        raise ValueError(f"not a direct Layer2-FSQ checkpoint: {path}")
    run_args = argparse.Namespace(**payload["args"])
    run_args.cpu = bool(cli.cpu)
    run_args.num_workers = int(cli.num_workers)
    run_args.val_num_workers = int(cli.val_num_workers)
    run_args.max_val_batches = int(cli.max_val_batches)
    if cli.data_dir:
        run_args.data_dir = str(cli.data_dir)
    if int(cli.bn_calibration_batches) >= 0:
        run_args.bn_calibration_batches = int(cli.bn_calibration_batches)
    run_args._usage_weight = 0.0
    direct.explore.ExploreIFSQQuantizer.config = run_args
    base.seed_everything(int(run_args.seed))
    source = base.load_teacher_checkpoint_for_args(run_args)
    cfg = base.jsccf_io.build_config(run_args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(cfg)
    bundle = direct.build_direct_bundle(run_args, source, cfg.device)
    base.jsccf_io.load_state(bundle.e1, payload["e1_state_dict"], "probe_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, payload["d1_state_dict"], "probe_D1", strict=True)
    base.jsccf_io.load_state(bundle.tokenizer, payload["tokenizer_state_dict"], "probe_codec", strict=True)
    base.jsccf_io.load_state(bundle.combiner, payload["combiner_state_dict"], "probe_combiner", strict=True)
    return payload, run_args, bundle, train_loader, val_loader, cfg.device, source


def alpha_grid(step: float, device: torch.device) -> torch.Tensor:
    count = int(math.ceil(1.0 / float(step)))
    values = torch.arange(count + 1, device=device, dtype=torch.float32) * float(step)
    values = values.clamp_max(1.0)
    if float(values[-1].item()) < 1.0:
        values = torch.cat([values, values.new_ones(1)])
    return torch.unique(values)


def analytic_alpha(x1: torch.Tensor, refined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    direction = (refined - x1).float()
    residual = (x1 - target).float()
    numerator = -(residual * direction).sum(dim=(1, 2, 3))
    denominator = direction.square().sum(dim=(1, 2, 3)).clamp_min(1e-12)
    return (numerator / denominator).clamp(0.0, 1.0)


def blend(x1: torch.Tensor, refined: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
    if isinstance(alpha, torch.Tensor) and alpha.ndim == 1:
        alpha = alpha.view(-1, 1, 1, 1)
    return torch.lerp(x1, refined, alpha).clamp(0.0, 1.0)


@torch.no_grad()
def probe_one(
    checkpoint_path: str,
    cli: argparse.Namespace,
) -> dict[str, float | int | str | list[int]]:
    payload, run_args, bundle, train_loader, val_loader, device, source = build_from_checkpoint(
        checkpoint_path, cli
    )
    calibration = direct.calibrate_fsq_batch_norm(train_loader, bundle, run_args, device)
    bundle.e1.eval()
    bundle.d1.eval()
    bundle.tokenizer.eval()
    bundle.combiner.eval()
    grid = alpha_grid(float(cli.alpha_step), device)
    grid_normal = torch.zeros_like(grid, dtype=torch.float64)
    grid_continuous = torch.zeros_like(grid, dtype=torch.float64)
    grid_zero = torch.zeros_like(grid, dtype=torch.float64)
    grid_shuffle = torch.zeros_like(grid, dtype=torch.float64)
    sums = {
        "psnr_x1": 0.0,
        "psnr_learned": 0.0,
        "psnr_alpha1": 0.0,
        "psnr_continuous_alpha1": 0.0,
        "psnr_per_image_alpha_oracle": 0.0,
        "psnr_continuous_per_image_alpha_oracle": 0.0,
        "alpha_oracle": 0.0,
        "alpha_continuous_oracle": 0.0,
    }
    alpha_min = 1.0
    alpha_max = 0.0
    count = 0
    for batch_index, (imgs, _labels) in enumerate(val_loader, start=1):
        if int(run_args.max_val_batches) > 0 and batch_index > int(run_args.max_val_batches):
            break
        imgs = imgs.to(device, non_blocking=True)
        layer1_out, out = direct.forward_direct(bundle, imgs)
        x1 = layer1_out["x1"]
        z1 = layer1_out["z1"]
        refined = bundle.combiner.inner(x1, out["u2_hat"])
        continuous_decoded = bundle.tokenizer.decode(out["z3_norm"], x1, z1, bundle.combiner)
        zero_decoded = bundle.tokenizer.decode(torch.zeros_like(out["q3"]), x1, z1, bundle.combiner)
        shuffled_decoded = bundle.tokenizer.decode(
            bundle.tokenizer.shuffle_q3(out["q3"]), x1, z1, bundle.combiner
        )
        refined_continuous = bundle.combiner.inner(x1, continuous_decoded["u2_hat"])
        refined_zero = bundle.combiner.inner(x1, zero_decoded["u2_hat"])
        refined_shuffle = bundle.combiner.inner(x1, shuffled_decoded["u2_hat"])
        batch_size = int(imgs.shape[0])
        count += batch_size
        sums["psnr_x1"] += float(base.psnr_per_image(x1, imgs).sum().item())
        sums["psnr_learned"] += float(base.psnr_per_image(out["final"], imgs).sum().item())
        sums["psnr_alpha1"] += float(base.psnr_per_image(refined, imgs).sum().item())
        sums["psnr_continuous_alpha1"] += float(
            base.psnr_per_image(refined_continuous, imgs).sum().item()
        )

        alpha_oracle = analytic_alpha(x1, refined, imgs)
        alpha_continuous = analytic_alpha(x1, refined_continuous, imgs)
        sums["alpha_oracle"] += float(alpha_oracle.sum().item())
        sums["alpha_continuous_oracle"] += float(alpha_continuous.sum().item())
        alpha_min = min(alpha_min, float(alpha_oracle.min().item()))
        alpha_max = max(alpha_max, float(alpha_oracle.max().item()))
        sums["psnr_per_image_alpha_oracle"] += float(
            base.psnr_per_image(blend(x1, refined, alpha_oracle), imgs).sum().item()
        )
        sums["psnr_continuous_per_image_alpha_oracle"] += float(
            base.psnr_per_image(blend(x1, refined_continuous, alpha_continuous), imgs).sum().item()
        )

        for index, alpha in enumerate(grid):
            grid_normal[index] += float(base.psnr_per_image(blend(x1, refined, alpha), imgs).sum().item())
            grid_continuous[index] += float(
                base.psnr_per_image(blend(x1, refined_continuous, alpha), imgs).sum().item()
            )
            grid_zero[index] += float(base.psnr_per_image(blend(x1, refined_zero, alpha), imgs).sum().item())
            grid_shuffle[index] += float(
                base.psnr_per_image(blend(x1, refined_shuffle, alpha), imgs).sum().item()
            )
    if count <= 0:
        raise RuntimeError("validation loader produced no images")
    grid_normal /= float(count)
    grid_continuous /= float(count)
    grid_zero /= float(count)
    grid_shuffle /= float(count)
    best_index = int(grid_normal.argmax().item())
    best_continuous_index = int(grid_continuous.argmax().item())
    learned_alpha = float(bundle.combiner.alpha().item())
    levels = base.parse_fsq_levels(run_args.fsq_levels, int(run_args.fsq_d))
    vocab = base.vocab_size(levels)
    fixed_bits = int(math.ceil(math.log2(float(vocab))))
    psnr_x1 = sums["psnr_x1"] / float(count)
    result: dict[str, float | int | str | list[int]] = {
        "checkpoint": str(Path(base.resolve_path(checkpoint_path))),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "arch": str(run_args.arch),
        "levels": levels,
        "vocab_size": int(vocab),
        "fixed_bits_per_token": fixed_bits,
        "incremental_bpp": fixed_bits * int(run_args.latent_h) * int(run_args.latent_w) / float(256 * 256),
        "validation_images": int(count),
        "learned_alpha": learned_alpha,
        "psnr_x1": psnr_x1,
        "psnr_learned": sums["psnr_learned"] / float(count),
        "gain_learned": sums["psnr_learned"] / float(count) - psnr_x1,
        "psnr_alpha1": sums["psnr_alpha1"] / float(count),
        "gain_alpha1": sums["psnr_alpha1"] / float(count) - psnr_x1,
        "best_global_alpha": float(grid[best_index].item()),
        "psnr_best_global_alpha": float(grid_normal[best_index].item()),
        "gain_best_global_alpha": float(grid_normal[best_index].item()) - psnr_x1,
        "drop_zero_at_best_global_alpha": float(grid_normal[best_index] - grid_zero[best_index]),
        "drop_shuffle_at_best_global_alpha": float(grid_normal[best_index] - grid_shuffle[best_index]),
        "per_image_alpha_mean": sums["alpha_oracle"] / float(count),
        "per_image_alpha_min": alpha_min,
        "per_image_alpha_max": alpha_max,
        "psnr_per_image_alpha_oracle": sums["psnr_per_image_alpha_oracle"] / float(count),
        "gain_per_image_alpha_oracle": sums["psnr_per_image_alpha_oracle"] / float(count) - psnr_x1,
        "psnr_continuous_alpha1": sums["psnr_continuous_alpha1"] / float(count),
        "best_continuous_global_alpha": float(grid[best_continuous_index].item()),
        "psnr_continuous_best_global_alpha": float(grid_continuous[best_continuous_index].item()),
        "continuous_per_image_alpha_mean": sums["alpha_continuous_oracle"] / float(count),
        "psnr_continuous_per_image_alpha_oracle": sums[
            "psnr_continuous_per_image_alpha_oracle"
        ]
        / float(count),
        "source_z2_channels": int(source.get("stage2_codec", {}).get("z2_ch", -1)),
        "source_psnr_final": float(source.get("metrics", {}).get("psnr_final", float("nan"))),
        "source_delta_psnr": float(source.get("metrics", {}).get("delta_psnr", float("nan"))),
    }
    if calibration is not None:
        for key, value in calibration.items():
            result[f"bn_calibration_{key}"] = float(value)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
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
