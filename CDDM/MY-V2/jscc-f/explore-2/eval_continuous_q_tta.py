#!/usr/bin/env python3
"""Full-validation receiver-only D4 self-ensemble for continuous q2_hat.

The primary ``q-ensemble`` path transforms only the receiver-known z1/x1,
generates eight q candidates, maps them back to canonical orientation, averages
them into one q2_hat, then performs exactly one D2/combiner decode.  The target
image is used for metrics only.
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


def transform(value: torch.Tensor, rotation: int, flip: bool) -> torch.Tensor:
    value = torch.rot90(value, int(rotation), dims=(-2, -1))
    return torch.flip(value, dims=(-1,)) if flip else value


def inverse_transform(value: torch.Tensor, rotation: int, flip: bool) -> torch.Tensor:
    if flip:
        value = torch.flip(value, dims=(-1,))
    return torch.rot90(value, -int(rotation), dims=(-2, -1))


@torch.no_grad()
def evaluate(loader, source, receiver, device: torch.device) -> dict[str, float]:
    receiver.eval()
    sums = {"psnr_x1": 0.0, "psnr_base": 0.0, "psnr_q_ensemble": 0.0, "psnr_x_ensemble": 0.0}
    count = 0
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        layer1 = source.layer1(imgs)
        condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
        base = receiver(condition)["x2_hat"]
        canonical_q = []
        canonical_x = []
        for flip in (False, True):
            for rotation in range(4):
                augmented = make_receiver_condition(
                    transform(condition.z1, rotation, flip),
                    transform(condition.x1, rotation, flip),
                    detach=True,
                )
                output = receiver(augmented)
                canonical_q.append(inverse_transform(output["q2_hat"], rotation, flip))
                canonical_x.append(inverse_transform(output["x2_hat"], rotation, flip))
        q2_hat = torch.stack(canonical_q, dim=0).mean(dim=0)
        u2_hat = receiver.d2(q2_hat).clamp(0.0, 1.0)
        q_ensemble = receiver.combiner(condition.x1, u2_hat)
        x_ensemble = torch.stack(canonical_x, dim=0).mean(dim=0).clamp(0.0, 1.0)
        values = {
            "psnr_x1": continuous.psnr_per_image(condition.x1, imgs),
            "psnr_base": continuous.psnr_per_image(base, imgs),
            "psnr_q_ensemble": continuous.psnr_per_image(q_ensemble, imgs),
            "psnr_x_ensemble": continuous.psnr_per_image(x_ensemble, imgs),
        }
        batch = int(imgs.shape[0])
        for key, value in values.items():
            sums[key] += float(value.sum())
        count += batch
    metrics = {key: value / count for key, value in sums.items()}
    metrics.update(
        {
            "delta_base": metrics["psnr_base"] - metrics["psnr_x1"],
            "delta_q_ensemble": metrics["psnr_q_ensemble"] - metrics["psnr_x1"],
            "delta_x_ensemble": metrics["psnr_x_ensemble"] - metrics["psnr_x1"],
            "evaluated_images": float(count),
            "full_validation": float(count == len(loader.dataset)),
            "receiver_only_audit": 1.0,
            "q_ensemble_single_decode": 1.0,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--state",
        choices=["ema", "raw"],
        default="ema",
        help="evaluate the checkpoint EMA or same-epoch raw receiver weights",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--cpu", action="store_true")
    cli = parser.parse_args()

    checkpoint_path = continuous.resolve_path(cli.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if str(payload.get("stage", "")) != "explore2_continuous_q_receiver":
        raise ValueError(f"not a continuous-q receiver checkpoint: {checkpoint_path}")
    saved_args = dict(payload["args"])
    # Backward-compatible defaults for checkpoints written before the hybrid
    # combiner and freeze controls were added.
    saved_args.setdefault("combiner_type", "residual")
    saved_args.setdefault("base_combiner_width", int(saved_args.get("combiner_width", 48)))
    saved_args.setdefault("base_combiner_blocks", int(saved_args.get("combiner_blocks", 4)))
    saved_args.setdefault("freeze_initialized_backbone", False)
    saved_args.setdefault("init_checkpoint", "")
    # Historical continuous-q checkpoints predate the opt-in q-only highres
    # D2 fields.  Preserve their exact legacy D2 topology when reloading.
    saved_args.setdefault("d2_type", "layer1")
    saved_args.setdefault("d2_highres_width", 64)
    saved_args.setdefault("d2_highres_blocks", 2)
    args = argparse.Namespace(**saved_args)
    args.cpu = bool(cli.cpu)
    args.test_batch = int(cli.batch_size)
    args.batch_size = int(cli.batch_size)
    args.num_workers = int(cli.num_workers)
    args.val_num_workers = int(cli.num_workers)
    args.max_train_batches = 0
    args.max_val_batches = 0
    source, _train_loader, val_loader, device = continuous.build_source_and_loaders(args)
    receiver = continuous.build_receiver(source, args, device)
    state = (
        payload.get("ema_state_dict") or payload["receiver_state_dict"]
        if str(cli.state) == "ema"
        else payload["receiver_state_dict"]
    )
    receiver.load_state_dict(state, strict=True)
    # Sender Layer2 is never called by this evaluator.
    source.e2 = torch.nn.Identity().to(device)
    source.d2 = torch.nn.Identity().to(device)
    source.combiner = torch.nn.Identity().to(device)
    source.checkpoint = {}
    source.layer2_checkpoint = {}

    metrics = evaluate(val_loader, source, receiver, device)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "evaluated_state": str(cli.state),
        "receiver_contract": {
            "deployment_inputs": ["z1", "x1"],
            "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
            "q_ensemble": "D4(z1,x1) -> G -> inverse-D4 -> mean q2_hat -> one D2/combiner",
        },
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    output = Path(cli.output) if cli.output else HERE / "results-receiver" / "continuous_q_tta.json"
    if not output.is_absolute():
        output = continuous.ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
