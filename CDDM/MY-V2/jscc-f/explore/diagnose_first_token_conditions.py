#!/usr/bin/env python3
"""First-raster-token condition ablation for the iFSQ prefix AR model.

The same trained AR checkpoint is evaluated with both receiver conditions,
only z1, only x1, and neither condition.  Only the first-token logits are
used, so later autoregressive errors cannot contaminate this measurement.
Zeroing a branch is an inference ablation (not a separately retrained model);
the report therefore gives a direct same-model sensitivity test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
if str(JSCCF_DIR) not in sys.path:
    sys.path.insert(0, str(JSCCF_DIR))


def load_ar():
    path = JSCCF_DIR / "train_stage-fsq-ar.py"
    spec = importlib.util.spec_from_file_location("jsccf_first_token_ar", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--ar-checkpoint",
        default=(
            "MY-V2/jscc-f/checkpoints-ar-ifsq/"
            "jscc_f_ifsq-prefix-ar-k125_stage_ifsq_ar_fsq_l5x5x5_best.pth"
        ),
    )
    parser.add_argument("--max-val-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    ar = load_ar()
    checkpoint_path = ar.resolved(args.ar_checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**payload["args"])
    saved.cpu = bool(args.cpu)
    device = torch.device("cpu" if args.cpu else "cuda:0" if torch.cuda.is_available() else "cpu")

    system, _layer2_payload, layer2_args, _layer2_path = ar.load_frozen_system(saved, device)
    latent = dict(torch.load(ar.resolved(saved.layer1_checkpoint), map_location="cpu", weights_only=False).get("latent", {}))
    y1_shape = latent.get("z1", [16, 16, 16])
    model = ar.IFSQPrefixRasterAR(
        int(y1_shape[0]),
        system.height,
        system.width,
        system.vocabulary,
        hidden=int(saved.hidden),
        layers=int(saved.layers),
        heads=int(saved.heads),
        condition_blocks=int(saved.condition_blocks),
        ff_mult=float(saved.ff_mult),
        dropout=float(saved.dropout),
    ).to(device)
    model.load_state_dict(payload["ar_state_dict"], strict=True)
    model.eval()

    runtime = argparse.Namespace(
        data_dir=str(saved.data_dir),
        batch_size=int(args.batch_size),
        test_batch=int(args.batch_size),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.num_workers),
        cpu=bool(args.cpu),
    )
    _train_loader, val_loader, loader_device = ar.build_loaders(layer2_args, runtime)
    if loader_device != device:
        raise RuntimeError(f"loader/model device mismatch: {loader_device} vs {device}")

    modes = {
        "z1": {"zero_y1": False, "zero_x1": True},
        "x1": {"zero_y1": True, "zero_x1": False},
        "z1_x1": {"zero_y1": False, "zero_x1": False},
        "none": {"zero_y1": True, "zero_x1": True},
    }
    sums = {
        name: {"ce": 0.0, "correct": 0.0, "n": 0, "log_prob": 0.0}
        for name in modes
    }
    vocabulary = int(system.vocabulary)
    target_counts = torch.zeros(vocabulary, dtype=torch.long)

    for batch_index, (images, _labels) in enumerate(val_loader, start=1):
        if int(args.max_val_batches) > 0 and batch_index > int(args.max_val_batches):
            break
        images = images.to(device, non_blocking=True)
        target = system.oracle_targets(images)
        label = target["indices"][:, 0, 0].long()
        target_counts += torch.bincount(label.cpu(), minlength=vocabulary)
        for name, options in modes.items():
            y1 = torch.zeros_like(target["y1"]) if options["zero_y1"] else target["y1"]
            x1 = torch.zeros_like(target["x1"]) if options["zero_x1"] else target["x1"]
            logits = model.forward_teacher(y1, x1, target["indices"])[:, 0]
            log_prob = F.log_softmax(logits.float(), dim=-1)
            sums[name]["ce"] += float(F.nll_loss(log_prob, label, reduction="sum").item())
            sums[name]["correct"] += float(logits.argmax(dim=-1).eq(label).sum().item())
            sums[name]["log_prob"] += float(log_prob.gather(1, label[:, None]).sum().item())
            sums[name]["n"] += int(label.numel())

    total = int(sum(item["n"] for item in sums.values()) // max(1, len(sums)))
    if total < 1:
        raise RuntimeError("diagnostic loader produced no images")
    mode_count = int(target_counts.max().item())
    report = {
        "checkpoint": checkpoint_path,
        "epoch": int(payload.get("epoch", -1)),
        "device": str(device),
        "images": total,
        "vocabulary": vocabulary,
        "first_token_target_unique": int((target_counts > 0).sum().item()),
        "first_token_target_mode_accuracy": mode_count / float(total),
        "modes": {
            name: {
                "ce": values["ce"] / float(values["n"]),
                "nll_bits": values["ce"] / float(values["n"]) / torch.log(torch.tensor(2.0)).item(),
                "accuracy": values["correct"] / float(values["n"]),
                "mean_true_log_prob": values["log_prob"] / float(values["n"]),
            }
            for name, values in sums.items()
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    run(parse_args())
