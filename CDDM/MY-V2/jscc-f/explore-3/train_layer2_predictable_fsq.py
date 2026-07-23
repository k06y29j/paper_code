#!/usr/bin/env python3
"""Jointly train a predictable K=125 FSQ Layer2 codec.

This experiment moves receiver predictability into Layer2 training itself.
The sender still observes ``img`` when producing its FSQ target, while the
public receiver predictor has the deliberately narrow signature
``predictor(z1, x1)``.  Neither image, z2/q2, nor target indices are accepted by
the deployment forward.

The exact route is::

    sender:   cat(img, x1) -> E2 -> FSQ[5,5,5] -> q2 -> D2 -> combiner -> x2
    receiver: (z1, x1) -> joint 125-way predictor -> q2_hat
              -> the same D2 -> the same combiner -> x2_hat

Unlike a hard-label CE alone, ``loss_sender_shape`` is differentiable with
respect to E2.  It uses soft assignments around the fixed FSQ grid and has an
explicit entropy floor, so making codes predictable cannot be satisfied by a
constant/unused codebook.  Checkpoint selection is based on the real public
receiver reconstruction, never train-time teacher forcing.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
EXPLORE_DIR = JSCCF_DIR / "explore"
CDDM_ROOT = JSCCF_DIR.parents[1]
for path in (CDDM_ROOT, JSCCF_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_ORACLE = (
    "MY-V2/jscc-f/explore/checkpoints-direct/"
    "direct-cnn-d3-l5x5x5-group-compatible-blend-e100/"
    "jscc_f_direct-cnn-d3-l5x5x5-group-compatible-blend-e100_"
    "layer2_fsq_direct_cnn_d3_l5x5x5_group_compatible_blend_goal_best.pth"
)
DEFAULT_PREDICTOR = (
    "MY-V2/jscc-f/explore-2/checkpoints-125/"
    "cnn-fsq-k125-frozen-d2-combiner-hard-v1/"
    "fsq_receiver_direct_q_z1_x1_hard_cnn_d3_l5x5x5_frozen_best.pth"
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


direct = load_module(
    "jsccf_explore3_direct_support",
    EXPLORE_DIR / "train_layer2_fsq_direct.py",
)
base = direct.base
predictor_lib = load_module(
    "jsccf_explore3_joint125_predictor",
    THIS_DIR / "joint125_predictor.py",
)
Joint125Predictor = predictor_lib.Joint125Predictor
initialize_from_direct_q_state = predictor_lib.initialize_from_direct_q_state


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else CDDM_ROOT / candidate


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def set_trainable(module: nn.Module, enabled: bool) -> None:
    module.requires_grad_(bool(enabled))
    module.train(bool(enabled))


def unique_parameters(groups: Iterable[Iterable[nn.Parameter]]) -> list[nn.Parameter]:
    result: dict[int, nn.Parameter] = {}
    for group in groups:
        for parameter in group:
            result[id(parameter)] = parameter
    return list(result.values())


class MetricSums:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def add(self, name: str, value: float | torch.Tensor, weight: int | float) -> None:
        if torch.is_tensor(value):
            value = float(value.detach().item())
        self.sums[name] = self.sums.get(name, 0.0) + float(value) * float(weight)
        self.weights[name] = self.weights.get(name, 0.0) + float(weight)

    def means(self) -> dict[str, float]:
        return {
            name: value / max(1.0, self.weights[name])
            for name, value in self.sums.items()
        }


@dataclass(frozen=True)
class ForwardResult:
    layer1: dict[str, torch.Tensor]
    sender: dict[str, torch.Tensor]
    prediction: object
    anchor_prediction: object
    hard: dict[str, torch.Tensor]
    soft: dict[str, torch.Tensor]
    base: dict[str, torch.Tensor]


def assert_k125_contract(oracle_args: argparse.Namespace, checkpoint: dict) -> None:
    levels = base.parse_fsq_levels(oracle_args.fsq_levels, int(oracle_args.fsq_d))
    expected = {
        "arch": "cnn",
        "condition_mode": "none",
        "fsq_normalizer": "group",
        "fsq_d": 3,
        "latent_ch": 16,
        "latent_h": 16,
        "latent_w": 16,
    }
    for key, value in expected.items():
        if getattr(oracle_args, key) != value:
            raise ValueError(
                f"K125 oracle contract mismatch {key}: "
                f"{getattr(oracle_args, key)!r} != {value!r}"
            )
    if levels != [5, 5, 5] or math.prod(levels) != 125:
        raise ValueError(f"this trainer requires levels=[5,5,5], got {levels}")
    if str(checkpoint.get("stage", "")) != "layer2_fsq_direct":
        raise ValueError(
            f"oracle stage must be layer2_fsq_direct, got {checkpoint.get('stage')!r}"
        )


def load_bundle(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[direct.DirectBundle, argparse.Namespace, dict]:
    checkpoint = base.jsccf_io.load_checkpoint(str(resolve_path(checkpoint_path)))
    oracle_args = argparse.Namespace(**checkpoint["args"])
    assert_k125_contract(oracle_args, checkpoint)
    source_path = checkpoint.get("source_layer2_ckpt") or oracle_args.layer2_ckpt
    source = base.jsccf_io.load_checkpoint(str(resolve_path(source_path)))
    direct.explore.ExploreIFSQQuantizer.config = oracle_args
    bundle = direct.build_direct_bundle(oracle_args, source, device)
    base.jsccf_io.load_state(bundle.e1, checkpoint["e1_state_dict"], "explore3_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, checkpoint["d1_state_dict"], "explore3_D1", strict=True)
    base.jsccf_io.load_state(
        bundle.tokenizer,
        checkpoint["tokenizer_state_dict"],
        "explore3_tokenizer",
        strict=True,
    )
    base.jsccf_io.load_state(
        bundle.combiner,
        checkpoint["combiner_state_dict"],
        "explore3_combiner",
        strict=True,
    )
    set_trainable(bundle.e1, False)
    set_trainable(bundle.d1, False)
    return bundle, oracle_args, checkpoint


def build_predictor(
    args: argparse.Namespace,
    oracle_args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, dict]:
    predictor_checkpoint = torch.load(
        resolve_path(args.predictor_init), map_location="cpu", weights_only=False
    )
    saved = predictor_checkpoint.get("args", {})
    expected = {
        "route": "direct_q",
        "condition_mode": "z1_x1",
        "hidden": int(args.hidden),
        "blocks": int(args.blocks),
        "attention_every": int(args.attention_every),
        "heads": int(args.heads),
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(
                f"predictor init mismatch {key}: {saved.get(key)!r} != {value!r}"
            )
    predictor = Joint125Predictor(
        z1_channels=int(oracle_args.latent_ch),
        hidden=int(args.hidden),
        blocks=int(args.blocks),
        attention_every=int(args.attention_every),
        heads=int(args.heads),
        base_temperature=float(args.receiver_temperature),
        base_head_trainable=bool(args.train_base_head),
    ).to(device)
    report = initialize_from_direct_q_state(
        predictor,
        predictor_checkpoint,
        strict=True,
        base_head_trainable=bool(args.train_base_head),
    )
    anchor = copy.deepcopy(predictor).to(device)
    set_trainable(anchor, False)
    anchor.eval()
    signature = inspect.signature(predictor.forward)
    if tuple(signature.parameters) != ("z1", "x1"):
        raise AssertionError(
            f"receiver forward must accept only z1/x1, got {tuple(signature.parameters)}"
        )
    return predictor, anchor, {
        "source": report.source,
        "trunk_tensors": len(report.trunk.loaded_keys),
        "base_head_tensors": len(report.loaded_base_head_keys),
    }


def assert_crop_protocol(train_loader: DataLoader, val_loader: DataLoader) -> None:
    train_transform = repr(getattr(train_loader.dataset, "transform", None))
    val_transform = repr(getattr(val_loader.dataset, "transform", None))
    if "RandomCrop" not in train_transform or "RandomHorizontalFlip" not in train_transform:
        raise AssertionError(f"training must use random crop+flip, got {train_transform}")
    if "CenterCrop" not in val_transform or "RandomCrop" in val_transform:
        raise AssertionError(f"validation must use center crop, got {val_transform}")
    if len(train_loader.dataset) != 800:
        raise AssertionError(f"expected DIV2K train800, got {len(train_loader.dataset)}")
    if len(val_loader.dataset) != 100:
        raise AssertionError(f"expected DIV2K valid100, got {len(val_loader.dataset)}")
    if not bool(train_loader.drop_last):
        raise AssertionError("training loader must retain established drop_last=True contract")
    if bool(val_loader.drop_last):
        raise AssertionError("validation loader must use drop_last=False")


def build_loaders(
    args: argparse.Namespace,
    oracle_args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.device]:
    loader_args = copy.deepcopy(oracle_args)
    loader_args.data_dir = str(args.data_dir)
    loader_args.batch_size = int(args.batch_size)
    loader_args.test_batch = int(args.test_batch)
    loader_args.num_workers = int(args.num_workers)
    loader_args.val_num_workers = int(args.val_num_workers)
    loader_args.cpu = bool(args.cpu)
    config = base.jsccf_io.build_config(loader_args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(config)
    assert_crop_protocol(train_loader, val_loader)

    # Accuracy comparisons use the same deterministic crop/eval/deploy path on
    # both domains.  Comparing random augmented train batches to center-crop
    # validation was the source of the misleading historical accuracy gap.
    dataset_module = load_module(
        "jsccf_explore3_datasets",
        CDDM_ROOT / "Autoencoder" / "data" / "datasets.py",
    )
    center_dataset = dataset_module.FlatImageFolder(
        root=config.train_data_dir,
        transform=transforms.Compose(
            [transforms.CenterCrop((256, 256)), transforms.ToTensor()]
        ),
    )
    holdout_count = min(int(args.train_holdout_images), len(center_dataset))
    if holdout_count < 1:
        raise ValueError("--train-holdout-images must select at least one image")
    holdout_loader = DataLoader(
        Subset(center_dataset, list(range(holdout_count))),
        batch_size=int(args.test_batch),
        shuffle=False,
        num_workers=int(args.val_num_workers),
        pin_memory=not bool(args.cpu),
        drop_last=False,
    )
    return train_loader, val_loader, holdout_loader, config.device


def phase_weights(args: argparse.Namespace, epoch: int) -> tuple[str, float]:
    if int(epoch) <= int(args.warmup_epochs):
        return "receiver_warmup", 0.0
    ramp = max(1, int(args.shape_ramp_epochs))
    progress = min(1.0, (int(epoch) - int(args.warmup_epochs)) / float(ramp))
    return "joint_shaping", float(args.lambda_sender_shape) * progress


def configure_phase(
    bundle: direct.DirectBundle,
    predictor: nn.Module,
    args: argparse.Namespace,
    epoch: int,
) -> tuple[str, float]:
    phase, sender_shape_weight = phase_weights(args, epoch)
    sender_enabled = phase == "joint_shaping"
    set_trainable(bundle.tokenizer.e3, sender_enabled)
    set_trainable(bundle.tokenizer.quantizer, sender_enabled)
    set_trainable(bundle.tokenizer.d3, True)
    set_trainable(bundle.combiner, True)
    # Receiver co-design is trained from the first epoch.  Only the sender E2
    # waits for warmup: forcing the receiver to chase old, image-dependent
    # sender codes was empirically anti-correlated with deployment PSNR.
    set_trainable(predictor.trunk, True)
    set_trainable(predictor.head, True)
    predictor.set_base_head_trainable(bool(args.train_base_head) and sender_enabled)
    predictor.train()
    if not bool(args.train_base_head):
        predictor.base_head.eval()
    bundle.e1.eval()
    bundle.d1.eval()
    return phase, sender_shape_weight


def sender_joint_distribution(
    z_norm: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if tuple(z_norm.shape[1:]) != (3, 16, 16):
        raise ValueError(f"sender z_norm must be [B,3,16,16], got {tuple(z_norm.shape)}")
    centers = torch.linspace(
        -1.0, 1.0, 5, device=z_norm.device, dtype=z_norm.dtype
    ).view(1, 1, 5, 1, 1)
    scalar = F.softmax(
        -((z_norm.unsqueeze(2) - centers) / max(float(temperature), 1.0e-4)).square(),
        dim=2,
    )
    joint = torch.einsum(
        "bihw,bjhw,bkhw->bijkhw",
        scalar[:, 0],
        scalar[:, 1],
        scalar[:, 2],
    ).flatten(1, 3)
    if tuple(joint.shape[1:]) != (125, 16, 16):
        raise RuntimeError(f"unexpected sender joint distribution {tuple(joint.shape)}")
    return joint


def joint_entropy_bits(probabilities: torch.Tensor) -> torch.Tensor:
    marginal = probabilities.float().mean(dim=(0, 2, 3)).clamp_min(1.0e-12)
    return -(marginal * marginal.log2()).sum()


def forward_model(
    imgs: torch.Tensor,
    bundle: direct.DirectBundle,
    predictor: nn.Module,
    anchor: nn.Module,
) -> ForwardResult:
    with torch.no_grad():
        layer1 = bundle.layer1(imgs)
    sender = bundle.tokenizer.encode(imgs, layer1["x1"])
    sender.update(
        bundle.tokenizer.decode(
            sender["q3"], layer1["x1"], layer1["z1"], bundle.combiner
        )
    )
    # This is the only receiver call.  It deliberately receives no sender
    # tensor; all targets are consulted only after public inference completes.
    prediction = predictor(layer1["z1"].detach(), layer1["x1"].detach())
    with torch.no_grad():
        anchor_prediction = anchor(layer1["z1"].detach(), layer1["x1"].detach())
    hard = bundle.tokenizer.decode(
        prediction.q_st, layer1["x1"], layer1["z1"], bundle.combiner
    )
    soft = bundle.tokenizer.decode(
        prediction.q_soft, layer1["x1"], layer1["z1"], bundle.combiner
    )
    base_decoded = bundle.tokenizer.decode(
        prediction.q_base, layer1["x1"], layer1["z1"], bundle.combiner
    )
    return ForwardResult(
        layer1, sender, prediction, anchor_prediction, hard, soft, base_decoded
    )


def compute_losses(
    result: ForwardResult,
    imgs: torch.Tensor,
    args: argparse.Namespace,
    sender_shape_weight: float,
) -> dict[str, torch.Tensor]:
    prediction = result.prediction
    sender = result.sender
    sender_probs = sender_joint_distribution(
        sender["z3_norm"], float(args.sender_temperature)
    )
    log_receiver = F.log_softmax(prediction.logits.float(), dim=1)
    sender_probs_float = sender_probs.float().clamp_min(1.0e-12)
    log_sender = sender_probs_float.log()
    loss_sender_shape = (
        sender_probs_float * (log_sender - log_receiver.detach())
    ).sum(dim=1).mean() / math.log(125.0)
    entropy_bits = joint_entropy_bits(sender_probs)
    loss_entropy = F.relu(
        entropy_bits.new_tensor(float(args.entropy_floor_bits)) - entropy_bits
    ).square()
    receiver_probs = F.softmax(prediction.logits.float(), dim=1)
    receiver_entropy_bits = joint_entropy_bits(receiver_probs)
    loss_receiver_entropy = F.relu(
        receiver_entropy_bits.new_tensor(float(args.receiver_entropy_floor_bits))
        - receiver_entropy_bits
    ).square()

    loss_oracle = F.mse_loss(sender["final"].float(), imgs.float())
    loss_hard = F.mse_loss(result.hard["final"].float(), imgs.float())
    loss_soft = F.mse_loss(result.soft["final"].float(), imgs.float())
    loss_distill = F.mse_loss(
        result.hard["final"].float(), sender["final"].detach().float()
    )
    loss_receiver_code = F.cross_entropy(
        prediction.logits.float(), sender["idx3"].detach().long()
    ) / math.log(125.0)
    loss_receiver_q = F.mse_loss(
        prediction.q_soft.float(), sender["q3_hard"].detach().float()
    )
    loss_anchor = F.mse_loss(
        prediction.q_base.float(), result.anchor_prediction.q_base.detach().float()
    )
    loss = (
        float(args.lambda_oracle) * loss_oracle
        + float(args.lambda_hard) * loss_hard
        + float(args.lambda_soft) * loss_soft
        + float(args.lambda_distill) * loss_distill
        + float(args.lambda_receiver_code) * loss_receiver_code
        + float(args.lambda_receiver_q) * loss_receiver_q
        + float(sender_shape_weight) * loss_sender_shape
        + float(args.lambda_entropy) * loss_entropy
        + float(args.lambda_receiver_entropy) * loss_receiver_entropy
        + float(args.lambda_anchor) * loss_anchor
    )
    return {
        "loss": loss,
        "loss_oracle": loss_oracle,
        "loss_hard": loss_hard,
        "loss_soft": loss_soft,
        "loss_distill": loss_distill,
        "loss_receiver_code": loss_receiver_code,
        "loss_receiver_q": loss_receiver_q,
        "loss_sender_shape": loss_sender_shape,
        "loss_entropy": loss_entropy,
        "loss_receiver_entropy": loss_receiver_entropy,
        "loss_anchor": loss_anchor,
        "soft_sender_entropy_bits": entropy_bits,
        "soft_receiver_entropy_bits": receiver_entropy_bits,
    }


def add_psnr_metrics(
    meters: MetricSums,
    result: ForwardResult,
    imgs: torch.Tensor,
) -> None:
    batch = int(imgs.shape[0])
    psnr_x1 = base.psnr_per_image(result.layer1["x1"], imgs)
    psnr_oracle = base.psnr_per_image(result.sender["final"], imgs)
    psnr_hard = base.psnr_per_image(result.hard["final"], imgs)
    psnr_soft = base.psnr_per_image(result.soft["final"], imgs)
    psnr_base = base.psnr_per_image(result.base["final"], imgs)
    for name, values in (
        ("psnr_x1", psnr_x1),
        ("psnr_oracle", psnr_oracle),
        ("psnr_hard", psnr_hard),
        ("psnr_soft", psnr_soft),
        ("psnr_base", psnr_base),
        ("delta_oracle", psnr_oracle - psnr_x1),
        ("delta_hard", psnr_hard - psnr_x1),
        ("delta_soft", psnr_soft - psnr_x1),
        ("delta_base", psnr_base - psnr_x1),
    ):
        meters.add(name, float(values.mean().item()), batch)
    scalar_accuracy = (
        result.prediction.codes == result.sender["codes"]
    ).float().mean()
    joint_accuracy = (
        result.prediction.indices == result.sender["idx3"]
    ).float().mean()
    meters.add("scalar_accuracy", scalar_accuracy, batch)
    meters.add("joint_accuracy", joint_accuracy, batch)
    meters.add(
        "q_mse_hard",
        F.mse_loss(
            result.prediction.q_hard.float(), result.sender["q3_hard"].float()
        ),
        batch,
    )


def histogram_metrics(histogram: torch.Tensor, prefix: str) -> dict[str, float]:
    total = float(histogram.sum().item())
    if total <= 0.0:
        return {
            f"{prefix}_code_used": 0.0,
            f"{prefix}_code_entropy_bits": 0.0,
            f"{prefix}_code_perplexity": 0.0,
            f"{prefix}_code_top1_frac": 0.0,
        }
    active = histogram > 0
    probabilities = (histogram[active] / total).clamp_min(1.0e-12)
    entropy = float(-(probabilities * probabilities.log2()).sum().item())
    return {
        f"{prefix}_code_used": float(active.sum().item()),
        f"{prefix}_code_entropy_bits": entropy,
        f"{prefix}_code_perplexity": 2.0**entropy,
        f"{prefix}_code_top1_frac": float(histogram.max().item() / total),
    }


def level_metrics(level_hists: list[torch.Tensor], prefix: str) -> dict[str, float]:
    output: dict[str, float] = {}
    entropies: list[float] = []
    for channel, hist in enumerate(level_hists):
        total = float(hist.sum().item())
        active = hist > 0
        if total > 0:
            probabilities = (hist[active] / total).clamp_min(1.0e-12)
            entropy = float(-(probabilities * probabilities.log2()).sum().item())
        else:
            entropy = 0.0
        output[f"{prefix}_level{channel}_used"] = float(active.sum().item())
        output[f"{prefix}_level{channel}_entropy_bits"] = entropy
        entropies.append(entropy)
    output[f"{prefix}_level_entropy_bits_mean"] = sum(entropies) / 3.0
    return output


@torch.no_grad()
def validate(
    loader: DataLoader,
    bundle: direct.DirectBundle,
    predictor: nn.Module,
    anchor: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    bundle.e1.eval()
    bundle.d1.eval()
    bundle.tokenizer.eval()
    bundle.combiner.eval()
    predictor.eval()
    anchor.eval()
    meters = MetricSums()
    sender_hist = torch.zeros(125, dtype=torch.float64)
    pred_hist = torch.zeros(125, dtype=torch.float64)
    sender_levels = [torch.zeros(5, dtype=torch.float64) for _ in range(3)]
    pred_levels = [torch.zeros(5, dtype=torch.float64) for _ in range(3)]
    images = 0
    batches = 0
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if int(args.max_val_batches) > 0 and batch_index > int(args.max_val_batches):
            break
        imgs = imgs.to(device, non_blocking=True)
        result = forward_model(imgs, bundle, predictor, anchor)
        _phase, shape_weight = phase_weights(args, max(1, int(args._current_epoch)))
        losses = compute_losses(result, imgs, args, shape_weight)
        batch = int(imgs.shape[0])
        images += batch
        batches += 1
        for name, value in losses.items():
            meters.add(name, value, batch)
        add_psnr_metrics(meters, result, imgs)

        sender_hist += torch.bincount(
            result.sender["idx3"].reshape(-1).cpu(), minlength=125
        ).double()
        pred_hist += torch.bincount(
            result.prediction.indices.reshape(-1).cpu(), minlength=125
        ).double()
        for channel in range(3):
            sender_levels[channel] += torch.bincount(
                result.sender["codes"][:, channel].reshape(-1).cpu(), minlength=5
            ).double()
            pred_levels[channel] += torch.bincount(
                result.prediction.codes[:, channel].reshape(-1).cpu(), minlength=5
            ).double()

        permutation = torch.roll(
            torch.arange(batch, device=device), shifts=1
        )
        wrong_prediction = predictor(
            result.layer1["z1"][permutation], result.layer1["x1"][permutation]
        )
        wrong = bundle.tokenizer.decode(
            wrong_prediction.q_hard,
            result.layer1["x1"],
            result.layer1["z1"],
            bundle.combiner,
        )["final"]
        zero = bundle.tokenizer.decode(
            torch.zeros_like(result.prediction.q_hard),
            result.layer1["x1"],
            result.layer1["z1"],
            bundle.combiner,
        )["final"]
        shuffled = bundle.tokenizer.decode(
            bundle.tokenizer.shuffle_q3(result.prediction.q_hard),
            result.layer1["x1"],
            result.layer1["z1"],
            bundle.combiner,
        )["final"]
        oracle_zero = bundle.tokenizer.decode(
            torch.zeros_like(result.sender["q3"]),
            result.layer1["x1"],
            result.layer1["z1"],
            bundle.combiner,
        )["final"]
        oracle_shuffle = bundle.tokenizer.decode(
            bundle.tokenizer.shuffle_q3(result.sender["q3"]),
            result.layer1["x1"],
            result.layer1["z1"],
            bundle.combiner,
        )["final"]
        psnr_hard = base.psnr_per_image(result.hard["final"], imgs)
        psnr_oracle = base.psnr_per_image(result.sender["final"], imgs)
        for name, values in (
            ("condition_shuffle_drop", psnr_hard - base.psnr_per_image(wrong, imgs)),
            ("pred_drop_zero", psnr_hard - base.psnr_per_image(zero, imgs)),
            ("pred_drop_shuffle", psnr_hard - base.psnr_per_image(shuffled, imgs)),
            ("oracle_drop_zero", psnr_oracle - base.psnr_per_image(oracle_zero, imgs)),
            (
                "oracle_drop_shuffle",
                psnr_oracle - base.psnr_per_image(oracle_shuffle, imgs),
            ),
        ):
            meters.add(name, float(values.mean().item()), batch)

    metrics = meters.means()
    metrics.update(histogram_metrics(sender_hist, "sender"))
    metrics.update(histogram_metrics(pred_hist, "pred"))
    metrics.update(level_metrics(sender_levels, "sender"))
    metrics.update(level_metrics(pred_levels, "pred"))
    metrics["evaluated_images"] = float(images)
    metrics["evaluated_batches"] = float(batches)
    metrics["full_validation"] = float(
        int(args.max_val_batches) == 0 and images == len(loader.dataset)
    )
    metrics["receiver_only_audit"] = 1.0
    if int(args.max_val_batches) == 0:
        if len(loader.dataset) != 100 or images != 100:
            raise AssertionError(
                f"strict validation requires exactly valid100, dataset={len(loader.dataset)} evaluated={images}"
            )
    metrics["goal_route_delta"] = max(
        float(metrics.get("delta_hard", -math.inf)),
        float(metrics.get("delta_soft", -math.inf)),
    )
    metrics["quality_gate"] = float(quality_gate(metrics, args))
    metrics["goal_met"] = float(
        quality_gate(metrics, args)
        and metrics["goal_route_delta"] >= float(args.goal_delta_db)
    )
    return metrics


@torch.no_grad()
def evaluate_accuracy_holdout(
    loader: DataLoader,
    bundle: direct.DirectBundle,
    predictor: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    bundle.e1.eval()
    bundle.d1.eval()
    bundle.tokenizer.eval()
    predictor.eval()
    scalar_correct = 0
    scalar_total = 0
    joint_correct = 0
    joint_total = 0
    images = 0
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        layer1 = bundle.layer1(imgs)
        sender = bundle.tokenizer.encode(imgs, layer1["x1"])
        prediction = predictor(layer1["z1"], layer1["x1"])
        scalar_correct += int((prediction.codes == sender["codes"]).sum().item())
        scalar_total += int(sender["codes"].numel())
        joint_correct += int((prediction.indices == sender["idx3"]).sum().item())
        joint_total += int(sender["idx3"].numel())
        images += int(imgs.shape[0])
    return {
        "train_center_scalar_accuracy": scalar_correct / float(max(1, scalar_total)),
        "train_center_joint_accuracy": joint_correct / float(max(1, joint_total)),
        "train_center_evaluated_images": float(images),
    }


def quality_gate(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    scalar_levels_used = min(
        float(metrics.get(f"sender_level{channel}_used", 0.0))
        for channel in range(3)
    )
    return bool(
        metrics.get("full_validation", 0.0) == 1.0
        and metrics.get("receiver_only_audit", 0.0) == 1.0
        and metrics.get("delta_oracle", -math.inf) >= float(args.min_oracle_delta)
        and metrics.get("condition_shuffle_drop", -math.inf)
        >= float(args.min_condition_drop)
        and metrics.get("pred_drop_zero", -math.inf) >= float(args.min_pred_drop)
        and metrics.get("pred_drop_shuffle", -math.inf) >= float(args.min_pred_drop)
        and metrics.get("oracle_drop_zero", -math.inf) >= float(args.min_oracle_drop)
        and metrics.get("oracle_drop_shuffle", -math.inf) >= float(args.min_oracle_drop)
        and metrics.get("sender_code_entropy_bits", -math.inf)
        >= float(args.min_sender_entropy)
        and metrics.get("sender_code_used", 0.0) >= float(args.min_sender_codes)
        and scalar_levels_used >= 5.0
    )


DISPLAY_KEYS = (
    "loss",
    "loss_oracle",
    "loss_hard",
    "loss_soft",
    "loss_receiver_code",
    "loss_receiver_q",
    "loss_sender_shape",
    "loss_entropy",
    "loss_receiver_entropy",
    "soft_receiver_entropy_bits",
    "psnr_x1",
    "psnr_oracle",
    "psnr_hard",
    "psnr_soft",
    "delta_oracle",
    "delta_hard",
    "delta_soft",
    "scalar_accuracy",
    "joint_accuracy",
    "q_mse_hard",
    "condition_shuffle_drop",
    "pred_drop_zero",
    "pred_drop_shuffle",
    "oracle_drop_zero",
    "oracle_drop_shuffle",
    "sender_code_used",
    "sender_code_entropy_bits",
    "sender_code_perplexity",
    "pred_code_used",
    "pred_code_entropy_bits",
    "train_center_scalar_accuracy",
    "train_valid_scalar_accuracy_gap",
    "goal_route_delta",
    "quality_gate",
    "goal_met",
)


def display(metrics: dict[str, float]) -> dict[str, float]:
    return {key: metrics[key] for key in DISPLAY_KEYS if key in metrics}


def checkpoint_payload(
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    bundle: direct.DirectBundle,
    predictor: nn.Module,
    anchor: nn.Module,
    optimizer: torch.optim.Optimizer,
    predictor_report: dict,
    best_score: float,
    best_goal_score: float,
) -> dict:
    return {
        "stage": "explore3_layer2_predictable_fsq",
        "epoch": int(epoch),
        "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "metrics": metrics,
        "source_oracle_checkpoint": str(args.oracle_checkpoint),
        "source_predictor_checkpoint": str(args.predictor_init),
        "e1_state_dict": bundle.e1.state_dict(),
        "d1_state_dict": bundle.d1.state_dict(),
        "e2_state_dict": bundle.tokenizer.e3.state_dict(),
        "quantizer_state_dict": bundle.tokenizer.quantizer.state_dict(),
        "d2_state_dict": bundle.tokenizer.d3.state_dict(),
        "tokenizer_state_dict": bundle.tokenizer.state_dict(),
        "combiner_state_dict": bundle.combiner.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "anchor_predictor_state_dict": anchor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": direct.capture_rng_state(),
        "best_score": float(best_score),
        "best_goal_score": float(best_goal_score),
        "predictor_init_report": predictor_report,
        "tokenizer": {
            "type": "FSQ",
            "fsq_d": 3,
            "fsq_levels": [5, 5, 5],
            "multipliers": [25, 5, 1],
            "vocab_size": 125,
            "normalizer": "group",
        },
        "latent": {
            "z1": [16, 16, 16],
            "q2": [3, 16, 16],
            "idx2": [16, 16],
        },
        "receiver_contract": {
            "inputs": ["z1", "x1"],
            "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
            "hard_output": "exact K125 FSQ grid",
            "continuous_output": "posterior mean over exact K125 FSQ grid",
        },
        "crop_contract": {
            "train": "RandomCrop(256)+RandomHorizontalFlip",
            "validation": "CenterCrop(256)",
            "validation_images": 100,
            "train_accuracy_probe": "first N train images with CenterCrop(256)",
        },
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"saved checkpoint: {path}", flush=True)


def restore_checkpoint(
    path: str,
    bundle: direct.DirectBundle,
    predictor: nn.Module,
    anchor: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float, float]:
    if not path:
        return 1, -math.inf, -math.inf
    payload = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    if payload.get("stage") != "explore3_layer2_predictable_fsq":
        raise ValueError(f"not an explore-3 predictable FSQ checkpoint: {path}")
    base.jsccf_io.load_state(bundle.e1, payload["e1_state_dict"], "resume_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, payload["d1_state_dict"], "resume_D1", strict=True)
    base.jsccf_io.load_state(
        bundle.tokenizer, payload["tokenizer_state_dict"], "resume_tokenizer", strict=True
    )
    base.jsccf_io.load_state(
        bundle.combiner, payload["combiner_state_dict"], "resume_combiner", strict=True
    )
    base.jsccf_io.load_state(
        predictor, payload["predictor_state_dict"], "resume_predictor", strict=True
    )
    base.jsccf_io.load_state(
        anchor, payload["anchor_predictor_state_dict"], "resume_anchor", strict=True
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if "rng_state" in payload:
        direct.restore_rng_state(payload["rng_state"])
    return (
        int(payload["epoch"]) + 1,
        float(payload.get("best_score", -math.inf)),
        float(payload.get("best_goal_score", -math.inf)),
    )


def print_header(
    args: argparse.Namespace,
    bundle: direct.DirectBundle,
    predictor_report: dict,
    train_count: int,
    val_count: int,
) -> None:
    print("=== explore-3 | Layer2 predictable K125 FSQ ===", flush=True)
    print("实验设计", flush=True)
    print(
        "  sender=img+x1->E2->FSQ[5,5,5]->shared D2/combiner->x2; "
        "receiver=(z1,x1)->joint125->q2_hat->same D2/combiner->x2_hat",
        flush=True,
    )
    print(
        "  receiver forbidden inputs=img,z2,q2,oracle_indices; "
        "public forward signature=forward(z1,x1)",
        flush=True,
    )
    print(
        "  crop=train RandomCrop(256)+flip; valid CenterCrop(256), strict valid100; "
        "train accuracy probe=CenterCrop deploy path",
        flush=True,
    )
    print(f"  oracle={resolve_path(args.oracle_checkpoint)}", flush=True)
    print(f"  predictor_init={resolve_path(args.predictor_init)} report={predictor_report}", flush=True)
    print("loss设计", flush=True)
    print(
        f"  L={args.lambda_oracle:g}*MSE(x2,img)+{args.lambda_hard:g}*MSE(x2_hat_hard,img)"
        f"+{args.lambda_soft:g}*MSE(x2_hat_soft,img)+{args.lambda_distill:g}*distill"
        f"+{args.lambda_receiver_code:g}*joint125_CE/log125+{args.lambda_receiver_q:g}*qMSE"
        f"+ramp({args.lambda_sender_shape:g})*KL(sender_soft||receiver)+"
        f"{args.lambda_entropy:g}*entropy_floor(H>={args.entropy_floor_bits:g})"
        f"+{args.lambda_receiver_entropy:g}*receiver_entropy_floor(H>={args.receiver_entropy_floor_bits:g})"
        f"+{args.lambda_anchor:g}*receiver_anchor",
        flush=True,
    )
    print("模块选择", flush=True)
    print(
        f"  E1/D1=frozen; E2/FSQ=warmup {args.warmup_epochs} epochs frozen then lr={args.sender_lr:g}; "
        f"D2/combiner=shared trainable lr={args.decoder_lr:g}",
        flush=True,
    )
    print(
        f"  predictor=Joint125Predictor h={args.hidden} b={args.blocks} "
        f"attention_every={args.attention_every} heads={args.heads} lr={args.predictor_lr:g}; "
        f"base_head_trainable={args.train_base_head}",
        flush=True,
    )
    print(
        f"  counts train={train_count} valid={val_count}; batch={args.batch_size}/{args.test_batch}; "
        f"epochs={args.epochs}; device={'cpu' if args.cpu else 'cuda:0'} "
        f"visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    print(
        f"  goal=max(hard,codebook-posterior-mean) delta>={args.goal_delta_db:g} dB; "
        f"oracle_delta>={args.min_oracle_delta:g}; sender_entropy>={args.min_sender_entropy:g}",
        flush=True,
    )
    print(
        f"  initial blend_alpha={float(bundle.combiner.alpha().detach().item()):.6f}",
        flush=True,
    )


def smoke(
    args: argparse.Namespace,
    bundle: direct.DirectBundle,
    predictor: nn.Module,
    anchor: nn.Module,
    device: torch.device,
) -> None:
    imgs = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    result = forward_model(imgs, bundle, predictor, anchor)
    expected_q = (int(args.smoke_batch_size), 3, 16, 16)
    expected_img = (int(args.smoke_batch_size), 3, 256, 256)
    if tuple(result.prediction.q_hard.shape) != expected_q:
        raise AssertionError(f"q2_hat mismatch {tuple(result.prediction.q_hard.shape)}")
    if tuple(result.hard["final"].shape) != expected_img:
        raise AssertionError(f"x2_hat mismatch {tuple(result.hard['final'].shape)}")
    losses = compute_losses(result, imgs, args, sender_shape_weight=0.001)
    losses["loss"].backward()
    if bundle.e1.training or bundle.d1.training:
        raise AssertionError("Layer1 must stay frozen/eval")
    print(
        f"[smoke PASS] z1={tuple(result.layer1['z1'].shape)} logits={tuple(result.prediction.logits.shape)} "
        f"q2_hat={tuple(result.prediction.q_hard.shape)} x2_hat={tuple(result.hard['final'].shape)} "
        "receiver_inputs=(z1,x1)",
        flush=True,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    # Probe checkpoint args first so data/device and model construction share
    # the exact latent contract embedded by the oracle.
    oracle_probe = torch.load(
        resolve_path(args.oracle_checkpoint), map_location="cpu", weights_only=False
    )
    oracle_args_probe = argparse.Namespace(**oracle_probe["args"])
    train_loader, val_loader, holdout_loader, device = build_loaders(
        args, oracle_args_probe
    )
    bundle, oracle_args, oracle_checkpoint = load_bundle(args.oracle_checkpoint, device)
    predictor, anchor, predictor_report = build_predictor(args, oracle_args, device)

    set_trainable(bundle.tokenizer.e3, True)
    set_trainable(bundle.tokenizer.quantizer, True)
    set_trainable(bundle.tokenizer.d3, True)
    set_trainable(bundle.combiner, True)
    predictor_params = unique_parameters(
        [predictor.trunk.parameters(), predictor.head.parameters(), predictor.base_head.parameters()]
    )
    sender_params = unique_parameters(
        [bundle.tokenizer.e3.parameters(), bundle.tokenizer.quantizer.parameters()]
    )
    decoder_params = unique_parameters(
        [bundle.tokenizer.d3.parameters(), bundle.combiner.parameters()]
    )
    if set(map(id, predictor_params)) & set(map(id, sender_params + decoder_params)):
        raise AssertionError("optimizer parameter groups overlap")
    optimizer = torch.optim.AdamW(
        [
            {"params": predictor_params, "lr": float(args.predictor_lr), "name": "predictor"},
            {"params": sender_params, "lr": float(args.sender_lr), "name": "sender_e2_fsq"},
            {"params": decoder_params, "lr": float(args.decoder_lr), "name": "shared_d2_combiner"},
        ],
        weight_decay=float(args.weight_decay),
    )
    start_epoch, best_score, best_goal_score = restore_checkpoint(
        args.resume, bundle, predictor, anchor, optimizer
    )
    print_header(
        args,
        bundle,
        predictor_report,
        len(train_loader.dataset),
        len(val_loader.dataset),
    )
    if bool(args.smoke_shapes):
        configure_phase(bundle, predictor, args, max(1, start_epoch))
        smoke(args, bundle, predictor, anchor, device)
        return

    args._current_epoch = max(0, start_epoch - 1)
    if not args.resume or bool(args.eval_init):
        init_metrics = validate(
            val_loader, bundle, predictor, anchor, args, device
        )
        init_holdout = evaluate_accuracy_holdout(
            holdout_loader, bundle, predictor, device
        )
        init_metrics.update(init_holdout)
        init_metrics["train_valid_scalar_accuracy_gap"] = (
            init_metrics["train_center_scalar_accuracy"]
            - init_metrics["scalar_accuracy"]
        )
        print(f"[init strict valid100] {display(init_metrics)}", flush=True)
        init_path = Path(args.save_dir) / f"{args.version}_init_metrics.json"
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.write_text(
            json.dumps(init_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if bool(args.eval_init):
            return

    if start_epoch > int(args.epochs):
        raise ValueError(f"resume starts at {start_epoch}, beyond epochs={args.epochs}")
    last_train: dict[str, float] = {}
    for epoch in range(start_epoch, int(args.epochs) + 1):
        args._current_epoch = epoch
        phase, sender_shape_weight = configure_phase(
            bundle, predictor, args, epoch
        )
        meters = MetricSums()
        start = time.time()
        for batch_index, (imgs, _labels) in enumerate(train_loader, start=1):
            if int(args.max_train_batches) > 0 and batch_index > int(args.max_train_batches):
                break
            imgs = imgs.to(device, non_blocking=True)
            result = forward_model(imgs, bundle, predictor, anchor)
            losses = compute_losses(
                result, imgs, args, sender_shape_weight=sender_shape_weight
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    unique_parameters([predictor_params, sender_params, decoder_params]),
                    float(args.grad_clip_norm),
                )
            optimizer.step()
            batch = int(imgs.shape[0])
            for name, value in losses.items():
                meters.add(name, value, batch)
            add_psnr_metrics(meters, result, imgs)
        last_train = meters.means()
        last_train["sender_shape_weight"] = float(sender_shape_weight)
        print(
            f"[train {epoch:03d}/{int(args.epochs):03d}] phase={phase} "
            f"{display(last_train)} shape_weight={sender_shape_weight:.6g} "
            f"time={time.time() - start:.1f}s",
            flush=True,
        )

        should_validate = (
            epoch == 1
            or epoch == int(args.warmup_epochs)
            or epoch % int(args.val_every) == 0
            or epoch == int(args.epochs)
        )
        if should_validate:
            metrics = validate(
                val_loader, bundle, predictor, anchor, args, device
            )
            holdout = evaluate_accuracy_holdout(
                holdout_loader, bundle, predictor, device
            )
            metrics.update(holdout)
            metrics["train_valid_scalar_accuracy_gap"] = (
                metrics["train_center_scalar_accuracy"] - metrics["scalar_accuracy"]
            )
            print(
                f"[strict valid100 {epoch:03d}] phase={phase} {display(metrics)}",
                flush=True,
            )
            score = max(float(metrics["psnr_hard"]), float(metrics["psnr_soft"]))
            improved = score > best_score
            improved_goal = bool(metrics["goal_met"]) and score > best_goal_score
            if improved:
                best_score = score
            if improved_goal:
                best_goal_score = score
            payload = checkpoint_payload(
                epoch=epoch,
                args=args,
                metrics=metrics,
                bundle=bundle,
                predictor=predictor,
                anchor=anchor,
                optimizer=optimizer,
                predictor_report=predictor_report,
                best_score=best_score,
                best_goal_score=best_goal_score,
            )
            if improved:
                save_checkpoint(Path(args.save_dir) / f"{args.version}_best.pth", payload)
            if improved_goal:
                save_checkpoint(
                    Path(args.save_dir) / f"{args.version}_goal_best.pth", payload
                )
            metrics_path = Path(args.save_dir) / f"{args.version}_latest_metrics.json"
            metrics_path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        if epoch % int(args.latest_every) == 0 or epoch == int(args.epochs):
            payload = checkpoint_payload(
                epoch=epoch,
                args=args,
                metrics=last_train,
                bundle=bundle,
                predictor=predictor,
                anchor=anchor,
                optimizer=optimizer,
                predictor_report=predictor_report,
                best_score=best_score,
                best_goal_score=best_goal_score,
            )
            save_checkpoint(Path(args.save_dir) / f"{args.version}_latest.pth", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly train a receiver-predictable K125 FSQ Layer2 codec"
    )
    parser.add_argument("--oracle-checkpoint", default=DEFAULT_ORACLE)
    parser.add_argument("--predictor-init", default=DEFAULT_PREDICTOR)
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument("--save-dir", default=str(THIS_DIR / "checkpoints"))
    parser.add_argument("--log-file", default="")
    parser.add_argument("--version", default="k125-joint125-predictable-v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--shape-ramp-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--train-holdout-images", type=int, default=100)
    parser.add_argument("--predictor-lr", type=float, default=1.0e-4)
    parser.add_argument("--sender-lr", type=float, default=1.0e-5)
    parser.add_argument("--decoder-lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--attention-every", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--receiver-temperature", type=float, default=0.1)
    parser.add_argument("--sender-temperature", type=float, default=0.25)
    parser.add_argument("--train-base-head", action="store_true")
    parser.add_argument("--lambda-oracle", type=float, default=1.0)
    parser.add_argument("--lambda-hard", type=float, default=1.5)
    parser.add_argument("--lambda-soft", type=float, default=0.5)
    parser.add_argument("--lambda-distill", type=float, default=0.0)
    parser.add_argument("--lambda-receiver-code", type=float, default=0.0)
    parser.add_argument("--lambda-receiver-q", type=float, default=0.0)
    parser.add_argument("--lambda-sender-shape", type=float, default=0.0005)
    parser.add_argument("--lambda-entropy", type=float, default=0.05)
    parser.add_argument("--lambda-receiver-entropy", type=float, default=0.001)
    parser.add_argument("--lambda-anchor", type=float, default=0.005)
    parser.add_argument("--entropy-floor-bits", type=float, default=6.0)
    parser.add_argument("--receiver-entropy-floor-bits", type=float, default=4.5)
    parser.add_argument("--goal-delta-db", type=float, default=0.5)
    parser.add_argument("--min-oracle-delta", type=float, default=0.8)
    parser.add_argument("--min-condition-drop", type=float, default=0.1)
    parser.add_argument("--min-pred-drop", type=float, default=0.1)
    parser.add_argument("--min-oracle-drop", type=float, default=0.5)
    parser.add_argument("--min-sender-entropy", type=float, default=5.5)
    parser.add_argument("--min-sender-codes", type=float, default=100.0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--latest-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--resume", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--eval-init", action="store_true")
    parser.add_argument("--smoke-shapes", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    args = parser.parse_args()
    for name in (
        "epochs",
        "batch_size",
        "test_batch",
        "val_every",
        "latest_every",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "lambda_oracle",
        "lambda_hard",
        "lambda_soft",
        "lambda_distill",
        "lambda_receiver_code",
        "lambda_receiver_q",
        "lambda_sender_shape",
        "lambda_entropy",
        "lambda_receiver_entropy",
        "lambda_anchor",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if int(args.max_val_batches) > 0 and not (
        bool(args.smoke_shapes) or bool(args.eval_init)
    ):
        print(
            "[warn] max_val_batches>0 disables strict valid100 acceptance",
            flush=True,
        )
    args.save_dir = str(resolve_path(args.save_dir))
    if not args.log_file:
        args.log_file = str(THIS_DIR / "logs" / f"{args.version}.log")
    args.log_file = str(resolve_path(args.log_file))
    return args


def main() -> None:
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    base.setup_log_file(args.log_file)
    args_path = Path(args.save_dir) / f"{args.version}_args.json"
    args_path.write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    train(args)


if __name__ == "__main__":
    main()
