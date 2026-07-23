#!/usr/bin/env python3
"""Train-set calibration of a receiver-only residual scale.

For a fixed continuous-q receiver, deployment is

    x2_hat(alpha) = clamp(x1 + alpha * (receiver(z1,x1) - x1)).

Alpha is selected once on DIV2K train and then frozen for full DIV2K val.  The
validation-optimal alpha is reported only as a diagnostic ceiling and is never
the promoted deployment result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_continuous_q_receiver as continuous
from contracts import make_receiver_condition


def compatible_args(saved: dict, cli: argparse.Namespace) -> argparse.Namespace:
    values = dict(saved)
    values.setdefault("combiner_type", "residual")
    values.setdefault("base_combiner_width", int(values.get("combiner_width", 48)))
    values.setdefault("base_combiner_blocks", int(values.get("combiner_blocks", 4)))
    values.setdefault("freeze_initialized_backbone", False)
    values.setdefault("init_checkpoint", "")
    values.setdefault("final_loss", "mse")
    values.setdefault("log_mse_scale", 0.01)
    # v1--v12 did not serialize the optional q-only highres D2 flags.
    values.setdefault("d2_type", "layer1")
    values.setdefault("d2_highres_width", 64)
    values.setdefault("d2_highres_blocks", 2)
    values["cpu"] = bool(cli.cpu)
    values["batch_size"] = int(cli.batch_size)
    values["test_batch"] = int(cli.batch_size)
    values["num_workers"] = int(cli.num_workers)
    values["val_num_workers"] = int(cli.num_workers)
    values["max_train_batches"] = 0
    values["max_val_batches"] = 0
    return argparse.Namespace(**values)


@torch.no_grad()
def score_grid(loader, source, receiver, alphas: torch.Tensor, device: torch.device):
    receiver.eval()
    psnr_sums = torch.zeros_like(alphas, dtype=torch.float64, device="cpu")
    base_x1 = 0.0
    base_receiver = 0.0
    count = 0
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        layer1 = source.layer1(imgs)
        condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
        prediction = receiver(condition)["x2_hat"]
        error = condition.x1.float() - imgs.float()
        residual = prediction.float() - condition.x1.float()
        # Per-image quadratic coefficients before clipping.  Candidate images
        # are clamped below so the metric exactly matches deployment.
        candidates = (
            condition.x1.float().unsqueeze(0)
            + alphas.to(device)[:, None, None, None, None] * residual.unsqueeze(0)
        ).clamp(0.0, 1.0)
        target = imgs.float().unsqueeze(0)
        mse = (candidates - target).square().flatten(2).mean(dim=2)
        psnr_sums += (-10.0 * torch.log10(mse.clamp_min(1e-10))).sum(dim=1).cpu().double()
        base_x1 += float(continuous.psnr_per_image(condition.x1, imgs).sum())
        base_receiver += float(continuous.psnr_per_image(prediction, imgs).sum())
        count += int(imgs.shape[0])
    return {
        "mean_psnr": (psnr_sums / count).tolist(),
        "psnr_x1": base_x1 / count,
        "psnr_receiver": base_receiver / count,
        "images": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.025)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="")
    parser.add_argument("--cpu", action="store_true")
    cli = parser.parse_args()
    checkpoint_path = continuous.resolve_path(cli.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if str(payload.get("stage", "")) != "explore2_continuous_q_receiver":
        raise ValueError(f"not a continuous-q receiver checkpoint: {checkpoint_path}")
    args = compatible_args(payload["args"], cli)
    source, train_loader, val_loader, device = continuous.build_source_and_loaders(args)
    receiver = continuous.build_receiver(source, args, device)
    receiver.load_state_dict(payload.get("ema_state_dict") or payload["receiver_state_dict"], strict=True)
    source.e2 = torch.nn.Identity().to(device)
    source.d2 = torch.nn.Identity().to(device)
    source.combiner = torch.nn.Identity().to(device)
    source.checkpoint = {}
    source.layer2_checkpoint = {}

    steps = round((float(cli.alpha_max) - float(cli.alpha_min)) / float(cli.alpha_step))
    alphas = torch.linspace(float(cli.alpha_min), float(cli.alpha_max), steps + 1)
    train_scores = score_grid(train_loader, source, receiver, alphas, device)
    selected_index = int(torch.tensor(train_scores["mean_psnr"]).argmax())
    val_scores = score_grid(val_loader, source, receiver, alphas, device)
    val_oracle_index = int(torch.tensor(val_scores["mean_psnr"]).argmax())
    selected_psnr = float(val_scores["mean_psnr"][selected_index])
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "receiver_contract": {
            "deployment_inputs": ["z1", "x1"],
            "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
            "calibration_source": "DIV2K train only",
        },
        "alpha_grid": [float(value) for value in alphas],
        "train_selected": {
            "alpha": float(alphas[selected_index]),
            "train_psnr": float(train_scores["mean_psnr"][selected_index]),
            "val_psnr": selected_psnr,
            "val_delta_x1": selected_psnr - float(val_scores["psnr_x1"]),
        },
        "validation_diagnostic_ceiling": {
            "not_for_promotion": True,
            "alpha": float(alphas[val_oracle_index]),
            "val_psnr": float(val_scores["mean_psnr"][val_oracle_index]),
            "val_delta_x1": float(val_scores["mean_psnr"][val_oracle_index])
            - float(val_scores["psnr_x1"]),
        },
        "train": train_scores,
        "validation": val_scores,
        "full_validation": float(val_scores["images"] == len(val_loader.dataset)),
        "receiver_only_audit": 1.0,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    output = Path(cli.output) if cli.output else HERE / "results-receiver" / "continuous_q_scale.json"
    if not output.is_absolute():
        output = continuous.ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
