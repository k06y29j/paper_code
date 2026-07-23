#!/usr/bin/env python3
"""Train equal-capacity probes for the first direct-FSQ token.

The frozen Layer1/direct-FSQ system supplies ``z1``, ``x1`` and the oracle
first raster index.  Three independently trained probes then use only z1,
only x1, or both.  This is a condition-information experiment, separate from
the long-horizon AR exposure-bias experiment.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
if str(JSCCF_DIR) not in sys.path:
    sys.path.insert(0, str(JSCCF_DIR))


def load_ar():
    path = JSCCF_DIR / "train_stage-fsq-ar.py"
    spec = importlib.util.spec_from_file_location("jsccf_first_token_probe_ar", path)
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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--target-row", type=int, default=0)
    parser.add_argument("--target-col", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class ProbeResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.silu(self.norm1(value)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return value + residual


class FirstTokenProbe(nn.Module):
    def __init__(self, mode: str, hidden: int, vocabulary: int) -> None:
        super().__init__()
        self.mode = str(mode)
        in_channels = {"z1": 16, "x1": 3, "z1_x1": 19}[self.mode]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, int(hidden), 3, padding=1),
            nn.GroupNorm(8, int(hidden)),
            nn.SiLU(),
        )
        self.body = nn.Sequential(*(ProbeResidual(int(hidden)) for _ in range(3)))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(vocabulary)),
        )

    def forward(self, z1: torch.Tensor, x1_small: torch.Tensor) -> torch.Tensor:
        if self.mode == "z1":
            value = z1
        elif self.mode == "x1":
            value = x1_small
        else:
            value = torch.cat([z1, x1_small], dim=1)
        return self.head(self.body(self.stem(value)))


@torch.no_grad()
def collect_features(
    loader, system, device: torch.device, target_row: int, target_col: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z1_all: list[torch.Tensor] = []
    x1_all: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    system.eval()
    for images, _labels in loader:
        images = images.to(device, non_blocking=True)
        target = system.oracle_targets(images)
        x1_small = F.interpolate(
            target["x1"], size=tuple(target["y1"].shape[-2:]), mode="bilinear", align_corners=False
        )
        z1_all.append(target["y1"].detach().cpu())
        x1_all.append(x1_small.detach().cpu())
        labels.append(target["indices"][:, int(target_row), int(target_col)].long().detach().cpu())
    return torch.cat(z1_all), torch.cat(x1_all), torch.cat(labels)


def mode_accuracy(labels: torch.Tensor) -> float:
    return float(labels.bincount().max().item()) / float(max(1, labels.numel()))


def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for z1, x1_small, labels in loader:
        logits = model(z1.to(device), x1_small.to(device))
        labels = labels.to(device)
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
        total_correct += int(logits.argmax(dim=-1).eq(labels).sum().item())
        total += int(labels.numel())
    ce = total_loss / float(max(1, total))
    return {
        "ce": ce,
        "nll_bits": ce / math.log(2.0),
        "perplexity": math.exp(min(30.0, ce)),
        "accuracy": total_correct / float(max(1, total)),
    }


def train_probe(mode: str, train_loader, val_loader, args, device, vocabulary: int) -> dict[str, float]:
    model = FirstTokenProbe(mode, int(args.hidden), vocabulary).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = {"ce": float("inf")}
    for _epoch in range(1, int(args.epochs) + 1):
        model.train()
        for z1, x1_small, labels in train_loader:
            z1 = z1.to(device, non_blocking=True)
            x1_small = x1_small.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = F.cross_entropy(model(z1, x1_small), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        if metrics["ce"] < best["ce"]:
            best = metrics
    return best


def run(args: argparse.Namespace) -> dict:
    seed_everything(int(args.seed))
    ar = load_ar()
    checkpoint_path = ar.resolved(args.ar_checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**payload["args"])
    saved.cpu = bool(args.cpu)
    device = torch.device("cpu" if args.cpu else "cuda:0" if torch.cuda.is_available() else "cpu")
    system, _layer2_payload, layer2_args, _layer2_path = ar.load_frozen_system(saved, device)

    runtime = argparse.Namespace(
        data_dir=str(saved.data_dir),
        batch_size=int(args.batch_size),
        test_batch=int(args.batch_size),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.num_workers),
        cpu=bool(args.cpu),
    )
    train_loader, val_loader, loader_device = ar.build_loaders(layer2_args, runtime)
    if loader_device != device:
        raise RuntimeError(f"loader/model device mismatch: {loader_device} vs {device}")

    if not 0 <= int(args.target_row) < int(system.height) or not 0 <= int(args.target_col) < int(system.width):
        raise ValueError("target coordinate is outside the FSQ grid")
    train_z1, train_x1, train_labels = collect_features(
        train_loader, system, device, int(args.target_row), int(args.target_col)
    )
    val_z1, val_x1, val_labels = collect_features(
        val_loader, system, device, int(args.target_row), int(args.target_col)
    )
    train_data = TensorDataset(train_z1, train_x1, train_labels)
    val_data = TensorDataset(val_z1, val_x1, val_labels)
    train_probe_loader = DataLoader(train_data, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_probe_loader = DataLoader(val_data, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    vocabulary = int(system.vocabulary)
    report = {
        "checkpoint": checkpoint_path,
        "epoch": int(payload.get("epoch", -1)),
        "device": str(device),
        "train_images": int(train_labels.numel()),
        "val_images": int(val_labels.numel()),
        "vocabulary": vocabulary,
        "target_coordinate": [int(args.target_row), int(args.target_col)],
        "train_mode_accuracy": mode_accuracy(train_labels),
        "val_mode_accuracy": mode_accuracy(val_labels),
        "modes": {},
    }
    for mode in ("z1", "x1", "z1_x1"):
        report["modes"][mode] = train_probe(
            mode, train_probe_loader, val_probe_loader, args, device, vocabulary
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    run(parse_args())
