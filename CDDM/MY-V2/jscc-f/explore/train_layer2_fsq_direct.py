#!/usr/bin/env python3
"""Train FSQ directly in Layer2, without an ``u2`` teacher target.

This is deliberately separate from ``train-stage3-fsq.py``.  The frozen
Layer2 checkpoint is used only as a source of the established Layer1 weights
and, optionally, shape-compatible E2/D2/combiner initialization.  Its E2/D2
forward result and its ``u2`` are never used as supervision.

Data flow::

    image -> frozen E1/D1 -> (z1, x1)
    concat(image, x1) -> trainable E2 -> FSQ -> trainable D2
    (x1, u2) -> trainable combiner -> final

The primary loss is MSE(final, image).  Validation includes continuous,
zero-code, and shuffled-code ablations so a large occupied vocabulary cannot
be mistaken for useful transmitted information.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
CDDM_ROOT = JSCCF_DIR.parents[1]
for path in (CDDM_ROOT, JSCCF_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


explore = load_module("jsccf_layer2_direct_fsq_support", THIS_DIR / "train_stage3_fsq_explore.py")
base = explore.base


METRIC_NAMES = [
    "loss",
    "loss_final",
    "loss_u2_img",
    "loss_usage",
    "mse_x1",
    "psnr_x1",
    "ssim_x1",
    "mse_final",
    "psnr_final",
    "ssim_final",
    "delta_x1",
    "mse_u2_as_img",
    "psnr_u2_as_img",
    "z3_abs_mean",
    "q3_abs_mean",
    "z3_norm_abs_mean",
    "z3_norm_saturation_frac",
    "code_edge_frac",
    "fsq_mse",
    "soft_level_entropy_bits",
    "soft_usage_entropy_bits",
    "blend_alpha",
]

VAL_ABLATION_METRICS = [
    "psnr_continuous",
    "gap_continuous",
    "psnr_zero",
    "psnr_shuffle",
    "drop_zero",
    "drop_shuffle",
]

DISPLAY_METRICS = [
    "loss",
    "loss_final",
    "loss_u2_img",
    "loss_usage",
    "psnr_x1",
    "psnr_final",
    "delta_x1",
    "psnr_continuous",
    "gap_continuous",
    "code_used",
    "code_entropy_bits",
    "code_perplexity",
    "level_entropy_bits_mean",
    "level_used_ratio_mean",
    "empirical_bpp",
    "drop_zero",
    "drop_shuffle",
    "fsq_mse",
    "soft_usage_entropy_bits",
    "z3_norm_saturation_frac",
    "code_edge_frac",
    "blend_alpha",
]


def display_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {name: metrics[name] for name in DISPLAY_METRICS if name in metrics}


def normalizer_name(args: argparse.Namespace) -> str:
    return "none" if bool(args.no_pre_norm) else str(args.fsq_normalizer)


def cli_option_present(tokens: list[str], option: str) -> bool:
    """Recognize both ``--flag value`` and ``--flag=value`` spellings."""
    return option in tokens or any(token.startswith(f"{option}=") for token in tokens)


class SafeBlendCombiner(nn.Module):
    """Optional x1 bypass around the established Layer2 combiner.

    A small nonzero initial blend keeps gradients flowing while making a fresh
    FSQ bottleneck start close to the fixed Layer1 baseline.  The normal
    ``original`` mode is also available as an exact architecture control.
    """

    def __init__(self, inner: nn.Module, mode: str, init_alpha: float) -> None:
        super().__init__()
        self.inner = inner
        self.mode = str(mode)
        if self.mode == "blend":
            alpha = min(max(float(init_alpha), 1e-4), 1.0 - 1e-4)
            self.blend_logit = nn.Parameter(torch.tensor(math.log(alpha / (1.0 - alpha))))
        elif self.mode == "original":
            self.register_parameter("blend_logit", None)
        else:
            raise ValueError(f"unsupported combiner mode {self.mode!r}")

    def alpha(self) -> torch.Tensor:
        if self.blend_logit is None:
            return next(self.inner.parameters()).new_ones(())
        return self.blend_logit.sigmoid()

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        refined = self.inner(x1, u2)
        if self.blend_logit is None:
            return refined
        alpha = self.alpha().to(dtype=refined.dtype)
        return torch.lerp(x1, refined, alpha).clamp(0.0, 1.0)


@dataclass
class DirectBundle:
    e1: nn.Module
    d1: nn.Module
    tokenizer: nn.Module
    combiner: SafeBlendCombiner
    init_report: dict[str, dict[str, int | float | str]]

    @torch.no_grad()
    def layer1(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        z1 = base.encode_tensor(self.e1, imgs)
        x1_raw = self.d1(z1)
        return {"z1": z1, "x1_raw": x1_raw, "x1": x1_raw.clamp(0.0, 1.0)}


def load_compatible_state(module: nn.Module, source: dict[str, torch.Tensor], label: str) -> dict[str, int | float | str]:
    target = module.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    module.load_state_dict(matched, strict=False)
    matched_numel = sum(int(target[key].numel()) for key in matched)
    total_numel = sum(int(value.numel()) for value in target.values())
    report: dict[str, int | float | str] = {
        "label": label,
        "matched_tensors": len(matched),
        "total_tensors": len(target),
        "matched_numel": matched_numel,
        "total_numel": total_numel,
        "matched_ratio": matched_numel / float(max(1, total_numel)),
    }
    print(
        f"[compatible init] {label}: tensors={len(matched)}/{len(target)} "
        f"numel={matched_numel}/{total_numel} ({100.0 * float(report['matched_ratio']):.2f}%)",
        flush=True,
    )
    return report


def build_layer1(args: argparse.Namespace, source_ckpt: dict, device: torch.device) -> tuple[nn.Module, nn.Module]:
    if str(args.arch) == "cnn":
        stage2 = base.load_script_module("jsccf_layer2_direct_source_cnn", "train_stage2-cnn.py")
        if hasattr(stage2, "validate_args"):
            stage2.validate_args(args)
        e1, d1 = stage2.build_cnn_layer1(args, device)
    else:
        stage2 = base.load_script_module("jsccf_layer2_direct_source_swin", "train_stage2-swin.py")
        e1, d1 = stage2.build_layer1(args, device)
    base.jsccf_io.load_state(e1, source_ckpt["e1_state_dict"], "direct_E1", strict=True)
    base.jsccf_io.load_state(d1, source_ckpt["d1_state_dict"], "direct_D1", strict=True)
    base.set_trainable(e1, False)
    base.set_trainable(d1, False)
    e1.eval()
    d1.eval()
    return e1, d1


def build_direct_bundle(args: argparse.Namespace, source_ckpt: dict, device: torch.device) -> DirectBundle:
    e1, d1 = build_layer1(args, source_ckpt, device)

    # Layer3FSQTokenizer already has exactly the desired E2(d)-FSQ-D2(d)
    # bottleneck.  Reuse its tested shape contract, but train it here as Layer2.
    base.IFSQQuantizer = explore.ExploreIFSQQuantizer
    explore.ExploreIFSQQuantizer.config = args
    tokenizer = base.Layer3FSQTokenizer(args, device)

    init_report: dict[str, dict[str, int | float | str]] = {}
    if str(args.codec_init) == "compatible":
        init_report["e2"] = load_compatible_state(tokenizer.e3, source_ckpt["e2_state_dict"], "E2->direct_E2")
        init_report["d2"] = load_compatible_state(tokenizer.d3, source_ckpt["d2_state_dict"], "D2->direct_D2")

    inner = base.OutputsCombiner().to(device)
    if not bool(args.fresh_combiner):
        base.jsccf_io.load_state(inner, source_ckpt["combiner_state_dict"], "direct_combiner_init", strict=True)
        init_report["combiner"] = {
            "label": "combiner",
            "matched_tensors": len(inner.state_dict()),
            "total_tensors": len(inner.state_dict()),
            "matched_numel": sum(int(value.numel()) for value in inner.state_dict().values()),
            "total_numel": sum(int(value.numel()) for value in inner.state_dict().values()),
            "matched_ratio": 1.0,
        }
    combiner = SafeBlendCombiner(inner, mode=str(args.combiner_mode), init_alpha=float(args.blend_init)).to(device)
    if bool(args.freeze_combiner):
        base.set_trainable(combiner.inner, False)
    return DirectBundle(e1=e1, d1=d1, tokenizer=tokenizer, combiner=combiner, init_report=init_report)


def direct_name(args: argparse.Namespace) -> str:
    levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    usage = ""
    if float(args.lambda_usage) > 0.0:
        usage = f"_{args.usage_objective}_u{explore._format_float(args.lambda_usage)}"
    return (
        f"layer2_fsq_direct_{args.arch}_d{int(args.fsq_d)}_{base.fsq_level_name(levels)}"
        f"_{normalizer_name(args)}_{args.codec_init}_{args.combiner_mode}{usage}"
    )


def compute_losses(
    out: dict[str, torch.Tensor],
    imgs: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    loss_final = base.recon_loss(out["final"], imgs)
    loss_u2_img = base.recon_loss(out["u2_raw"], imgs)
    loss_usage = out["usage_kl"]
    loss = (
        loss_final
        + float(args.lambda_u2_img) * loss_u2_img
        + float(getattr(args, "_usage_weight", args.lambda_usage)) * loss_usage
    )
    return {
        "loss": loss,
        "loss_final": loss_final,
        "loss_u2_img": loss_u2_img,
        "loss_usage": loss_usage,
    }


def make_level_hists(levels: list[int]) -> list[torch.Tensor]:
    return [torch.zeros(int(level), dtype=torch.float32) for level in levels]


def update_level_hists(level_hists: list[torch.Tensor], codes: torch.Tensor) -> None:
    if codes.ndim != 4 or int(codes.shape[1]) != len(level_hists):
        raise ValueError(f"level histogram expected codes [B,{len(level_hists)},H,W], got {tuple(codes.shape)}")
    for channel, hist in enumerate(level_hists):
        counts = torch.bincount(codes[:, channel].detach().reshape(-1).cpu(), minlength=hist.numel()).float()
        hist += counts[: hist.numel()]


def add_level_metrics(metrics: dict[str, float], level_hists: list[torch.Tensor], args: argparse.Namespace) -> None:
    entropies: list[float] = []
    used_ratios: list[float] = []
    for channel, hist in enumerate(level_hists):
        total = float(hist.sum().item())
        if total <= 0.0:
            used = 0.0
            entropy_bits = 0.0
            perplexity = 0.0
            top1_frac = 0.0
        else:
            active = hist > 0
            probs = (hist[active] / total).clamp_min(1e-12)
            entropy_bits = float(-(probs * probs.log2()).sum().item())
            perplexity = 2.0**entropy_bits
            used = float(active.sum().item())
            top1_frac = float(hist.max().item() / total)
        used_ratio = used / float(max(1, hist.numel()))
        metrics[f"level{channel}_used"] = used
        metrics[f"level{channel}_used_ratio"] = used_ratio
        metrics[f"level{channel}_entropy_bits"] = entropy_bits
        metrics[f"level{channel}_perplexity"] = perplexity
        metrics[f"level{channel}_top1_frac"] = top1_frac
        entropies.append(entropy_bits)
        used_ratios.append(used_ratio)
    metrics["level_entropy_bits_mean"] = sum(entropies) / float(max(1, len(entropies)))
    metrics["level_used_ratio_mean"] = sum(used_ratios) / float(max(1, len(used_ratios)))
    empirical_bits_per_image = float(metrics.get("code_entropy_bits", 0.0)) * int(args.latent_h) * int(args.latent_w)
    metrics["empirical_bits_per_image"] = empirical_bits_per_image
    metrics["empirical_bpp"] = empirical_bits_per_image / float(256 * 256)


def finalize_metrics(
    meters: dict,
    joint_hist: torch.Tensor,
    level_hists: list[torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, float]:
    metrics = base.finalize_metrics(meters, joint_hist, args)
    add_level_metrics(metrics, level_hists, args)
    return metrics


def update_metrics(
    meters: dict,
    out: dict[str, torch.Tensor],
    layer1_out: dict[str, torch.Tensor],
    imgs: torch.Tensor,
    losses: dict[str, torch.Tensor],
    combiner: SafeBlendCombiner,
    args: argparse.Namespace,
) -> None:
    bsz = int(imgs.shape[0])
    for name, value in losses.items():
        meters[name].update(float(value.detach().item()), bsz)

    psnr_x1 = base.batch_metric_mean(base.psnr_per_image(layer1_out["x1"], imgs))
    psnr_final = base.batch_metric_mean(base.psnr_per_image(out["final"], imgs))
    meters["mse_x1"].update(base.batch_metric_mean(base.mse_per_image(layer1_out["x1"], imgs)), bsz)
    meters["psnr_x1"].update(psnr_x1, bsz)
    meters["ssim_x1"].update(base.batch_metric_mean(base.ssim_per_image(layer1_out["x1"], imgs)), bsz)
    meters["mse_final"].update(base.batch_metric_mean(base.mse_per_image(out["final"], imgs)), bsz)
    meters["psnr_final"].update(psnr_final, bsz)
    meters["ssim_final"].update(base.batch_metric_mean(base.ssim_per_image(out["final"], imgs)), bsz)
    meters["delta_x1"].update(psnr_final - psnr_x1, bsz)
    meters["mse_u2_as_img"].update(base.batch_metric_mean(base.mse_per_image(out["u2_hat"], imgs)), bsz)
    meters["psnr_u2_as_img"].update(base.batch_metric_mean(base.psnr_per_image(out["u2_hat"], imgs)), bsz)
    meters["z3_abs_mean"].update(float(out["z3"].detach().float().abs().mean().item()), bsz)
    meters["q3_abs_mean"].update(float(out["q3_hard"].detach().float().abs().mean().item()), bsz)
    z3_norm = out["z3_norm"].detach().float()
    codes = out["codes"].detach()
    levels = torch.tensor(
        base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d)),
        device=codes.device,
        dtype=codes.dtype,
    ).view(1, -1, 1, 1)
    meters["z3_norm_abs_mean"].update(float(z3_norm.abs().mean().item()), bsz)
    meters["z3_norm_saturation_frac"].update(float((z3_norm.abs() >= 0.95).float().mean().item()), bsz)
    meters["code_edge_frac"].update(float(((codes == 0) | (codes == levels - 1)).float().mean().item()), bsz)
    meters["fsq_mse"].update(float(out["fsq_mse"].detach().item()), bsz)
    meters["soft_level_entropy_bits"].update(float(out["soft_level_entropy_bits"].detach().item()), bsz)
    meters["soft_usage_entropy_bits"].update(float(out["soft_usage_entropy_bits"].detach().item()), bsz)
    meters["blend_alpha"].update(float(combiner.alpha().detach().item()), bsz)


def forward_direct(bundle: DirectBundle, imgs: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    with torch.no_grad():
        layer1_out = bundle.layer1(imgs)
    out = bundle.tokenizer(
        imgs,
        layer1_out["x1"],
        layer1_out["z1"],
        bundle.combiner,
    )
    return layer1_out, out


@torch.no_grad()
def calibrate_fsq_batch_norm(
    loader,
    bundle: DirectBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float] | None:
    """Recompute exact pre-FSQ BatchNorm moments for the current E2.

    The E2 distribution moves throughout training, while the usual EMA
    running moments can lag behind it.  Validation then sees a different FSQ
    grid occupancy from training.  This pass measures the current z3 moments
    directly without running D2 or changing any trainable parameter.
    """
    max_batches = int(args.bn_calibration_batches)
    pre_norm = bundle.tokenizer.quantizer.pre_norm
    if max_batches <= 0 or not isinstance(pre_norm, nn.BatchNorm2d) or not bool(pre_norm.track_running_stats):
        return None

    bundle.e1.eval()
    bundle.d1.eval()
    bundle.tokenizer.eval()
    channels = int(args.fsq_d)
    total = 0
    value_sum = torch.zeros(channels, device=device, dtype=torch.float64)
    square_sum = torch.zeros_like(value_sum)
    batches = 0
    rng_state = capture_rng_state()
    try:
        for batch_idx, (imgs, _labels) in enumerate(loader, start=1):
            if batch_idx > max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            layer1_out = bundle.layer1(imgs)
            z3 = base.encode_tensor(bundle.tokenizer.e3, torch.cat([layer1_out["x1"], imgs], dim=1)).double()
            value_sum += z3.sum(dim=(0, 2, 3))
            square_sum += z3.square().sum(dim=(0, 2, 3))
            total += int(z3.shape[0] * z3.shape[2] * z3.shape[3])
            batches += 1
    finally:
        # Calibration is an evaluation-side measurement and must not alter the
        # next epoch's shuffle, crop, augmentation, or stochastic model path.
        restore_rng_state(rng_state)
    if total <= 0:
        raise RuntimeError("BatchNorm calibration loader produced no values")
    mean = value_sum / float(total)
    variance = (square_sum / float(total) - mean.square()).clamp_min(1e-8)
    pre_norm.running_mean.copy_(mean.to(dtype=pre_norm.running_mean.dtype))
    pre_norm.running_var.copy_(variance.to(dtype=pre_norm.running_var.dtype))
    pre_norm.num_batches_tracked.fill_(int(batches))
    stats = {
        "batches": float(batches),
        "values_per_channel": float(total),
        "mean_abs": float(mean.abs().mean().item()),
        "variance_mean": float(variance.mean().item()),
    }
    print(f"[fsq BN calibration] {stats}", flush=True)
    return stats


def goal_eligible(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        float(metrics.get("delta_x1", float("-inf"))) >= float(args.selection_min_delta_x1)
        and float(metrics.get("drop_zero", float("-inf"))) >= float(args.selection_min_drop_zero)
        and float(metrics.get("drop_shuffle", float("-inf"))) >= float(args.selection_min_drop_shuffle)
    )


@torch.no_grad()
def validate(loader, bundle: DirectBundle, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    bundle.e1.eval()
    bundle.d1.eval()
    bundle.tokenizer.eval()
    bundle.combiner.eval()
    meter_names = METRIC_NAMES + (VAL_ABLATION_METRICS if bool(args.val_ablation) else [])
    meters = base.meters(meter_names)
    levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    hist = torch.zeros(base.vocab_size(levels), dtype=torch.float32)
    level_hists = make_level_hists(levels)
    for batch_idx, (imgs, _labels) in enumerate(loader, start=1):
        if int(args.max_val_batches) > 0 and batch_idx > int(args.max_val_batches):
            break
        imgs = imgs.to(device, non_blocking=True)
        layer1_out, out = forward_direct(bundle, imgs)
        losses = compute_losses(out, imgs, args)
        update_metrics(meters, out, layer1_out, imgs, losses, bundle.combiner, args)
        base.update_code_hist(hist, out["idx3"])
        update_level_hists(level_hists, out["codes"])

        if bool(args.val_ablation):
            x1 = layer1_out["x1"]
            z1 = layer1_out["z1"]
            continuous = bundle.tokenizer.decode(out["z3_norm"], x1, z1, bundle.combiner)
            zero = bundle.tokenizer.decode(torch.zeros_like(out["q3"]), x1, z1, bundle.combiner)
            shuffled = bundle.tokenizer.decode(bundle.tokenizer.shuffle_q3(out["q3"]), x1, z1, bundle.combiner)
            bsz = int(imgs.shape[0])
            psnr_final = base.batch_metric_mean(base.psnr_per_image(out["final"], imgs))
            psnr_continuous = base.batch_metric_mean(base.psnr_per_image(continuous["final"], imgs))
            psnr_zero = base.batch_metric_mean(base.psnr_per_image(zero["final"], imgs))
            psnr_shuffle = base.batch_metric_mean(base.psnr_per_image(shuffled["final"], imgs))
            meters["psnr_continuous"].update(psnr_continuous, bsz)
            meters["gap_continuous"].update(psnr_continuous - psnr_final, bsz)
            meters["psnr_zero"].update(psnr_zero, bsz)
            meters["psnr_shuffle"].update(psnr_shuffle, bsz)
            meters["drop_zero"].update(psnr_final - psnr_zero, bsz)
            meters["drop_shuffle"].update(psnr_final - psnr_shuffle, bsz)
    return finalize_metrics(meters, hist, level_hists, args)


def print_header(
    args: argparse.Namespace,
    bundle: DirectBundle,
    train_n: int,
    val_n: int,
) -> None:
    levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    vocab = base.vocab_size(levels)
    bits_per_token = int(math.ceil(math.log2(float(vocab))))
    print(f"=== Layer 2 | direct FSQ | {args.arch} ===", flush=True)
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"save_dir={base.resolve_path(args.save_dir)}", flush=True)
    print("实验设计", flush=True)
    print("  path=frozen Layer1 -> trainable E2 -> FSQ -> trainable D2/combiner", flush=True)
    print("  u2_teacher=disabled; source Layer2 E2/D2 forward is not executed", flush=True)
    print(
        f"  source_ckpt={base.resolve_path(args.layer2_ckpt)} use=E1/D1 weights"
        f"+{args.codec_init} codec init+{'fresh' if args.fresh_combiner else 'compatible'} combiner init",
        flush=True,
    )
    print(
        f"  E2 input=concat(img,x1) q=[B,{int(args.fsq_d)},16,16] "
        f"levels={levels} vocab={vocab} fixed_bits/token={bits_per_token} "
        f"fixed_bits/image={bits_per_token * int(args.latent_h) * int(args.latent_w)}",
        flush=True,
    )
    print("loss设计", flush=True)
    print(
        f"  L=MSE(final,img)+{float(args.lambda_u2_img):g}*MSE(u2_raw,img)"
        f"+usage_weight*{args.usage_objective}; lambda_usage={float(args.lambda_usage):g}",
        flush=True,
    )
    print("模块选择", flush=True)
    print(
        f"  E1={base.trainable_state(bundle.e1)} D1={base.trainable_state(bundle.d1)} "
        f"E2={base.trainable_state(bundle.tokenizer.e3)} D2={base.trainable_state(bundle.tokenizer.d3)} "
        f"combiner={base.trainable_state(bundle.combiner)}",
        flush=True,
    )
    print(
        f"  FSQ normalizer={normalizer_name(args)} combiner_mode={args.combiner_mode} "
        f"blend_alpha_init={float(bundle.combiner.alpha().detach().item()):.6f} "
        f"bn_calibration_batches={int(args.bn_calibration_batches)}",
        flush=True,
    )
    print(
        f"  selection goal: delta_x1>={float(args.selection_min_delta_x1):g}, "
        f"drop_zero>={float(args.selection_min_drop_zero):g}, "
        f"drop_shuffle>={float(args.selection_min_drop_shuffle):g}",
        flush=True,
    )
    print(
        f"epochs={int(args.epochs)} train={train_n} valid={val_n} batch={int(args.batch_size)} "
        f"test_batch={int(args.test_batch)} workers={int(args.num_workers)}/{int(args.val_num_workers)} "
        f"lr={float(args.lr):g} weight_decay={float(args.weight_decay):g}",
        flush=True,
    )


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    bundle: DirectBundle,
    optimizer: optim.Optimizer,
    best_psnr: float,
    best_goal_psnr: float,
) -> None:
    output = Path(base.resolve_path(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    payload = {
        "route": getattr(base.jsccf_io, "ROUTE", "TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256"),
        "stage": "layer2_fsq_direct",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "version": str(args.version),
        "source_layer2_ckpt": str(args.layer2_ckpt),
        "e1_state_dict": bundle.e1.state_dict(),
        "d1_state_dict": bundle.d1.state_dict(),
        "e2_state_dict": bundle.tokenizer.e3.state_dict(),
        "quantizer_state_dict": bundle.tokenizer.quantizer.state_dict(),
        "d2_state_dict": bundle.tokenizer.d3.state_dict(),
        "tokenizer_state_dict": bundle.tokenizer.state_dict(),
        "combiner_state_dict": bundle.combiner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "best_psnr": float(best_psnr),
        "best_goal_psnr": float(best_goal_psnr),
        "init_report": bundle.init_report,
        "tokenizer": {
            "arch": str(args.arch),
            "fsq_d": int(args.fsq_d),
            "fsq_levels": levels,
            "vocab_size": base.vocab_size(levels),
            "normalizer": normalizer_name(args),
            "fixed_bits_per_token": int(math.ceil(math.log2(float(base.vocab_size(levels))))),
            "fixed_bits_per_image": int(math.ceil(math.log2(float(base.vocab_size(levels)))))
            * int(args.latent_h)
            * int(args.latent_w),
        },
        "latent": {
            "z1": [int(args.latent_ch), int(args.latent_h), int(args.latent_w)],
            "z2": [int(args.fsq_d), int(args.latent_h), int(args.latent_w)],
            "q2": [int(args.fsq_d), int(args.latent_h), int(args.latent_w)],
            "idx2": [int(args.latent_h), int(args.latent_w)],
        },
    }
    torch.save(payload, output)
    print(f"saved checkpoint: {output}", flush=True)


def load_resume(
    args: argparse.Namespace,
    bundle: DirectBundle,
    optimizer: optim.Optimizer,
) -> tuple[int, float, float]:
    if not args.resume:
        return 1, float("-inf"), float("-inf")
    payload = torch.load(base.resolve_path(args.resume), map_location="cpu")
    if str(payload.get("stage", "")) != "layer2_fsq_direct":
        raise ValueError(f"not a direct Layer2 FSQ checkpoint: {args.resume}")
    saved = payload.get("args", {})
    for key in (
        "arch",
        "fsq_d",
        "fsq_levels",
        "condition_mode",
        "fsq_normalizer",
        "no_pre_norm",
        "codec_init",
        "combiner_mode",
        "fresh_combiner",
        "match_source_width",
    ):
        if str(saved.get(key)) != str(getattr(args, key)):
            raise ValueError(f"resume mismatch for {key}: checkpoint={saved.get(key)!r} current={getattr(args, key)!r}")
    saved_calibration = int(saved.get("bn_calibration_batches", 0))
    current_calibration = int(args.bn_calibration_batches)
    if saved_calibration != current_calibration and not bool(args.reset_best_on_resume):
        raise ValueError(
            "resume changes --bn-calibration-batches from "
            f"{saved_calibration} to {current_calibration}; pass --reset-best-on-resume to start a new validation regime"
        )
    base.jsccf_io.load_state(bundle.e1, payload["e1_state_dict"], "resume_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, payload["d1_state_dict"], "resume_D1", strict=True)
    base.jsccf_io.load_state(bundle.tokenizer, payload["tokenizer_state_dict"], "resume_direct_codec", strict=True)
    base.jsccf_io.load_state(bundle.combiner, payload["combiner_state_dict"], "resume_combiner", strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if "rng_state" in payload:
        restore_rng_state(payload["rng_state"])
    else:
        print("[warn] resume checkpoint has no RNG state; continuation is not trajectory-exact", flush=True)
    start = int(payload.get("epoch", 0)) + 1
    print(f"resumed {args.resume} at epoch {start}", flush=True)
    return start, float(payload.get("best_psnr", float("-inf"))), float(payload.get("best_goal_psnr", float("-inf")))


def train(args: argparse.Namespace, source_ckpt: dict) -> None:
    base.seed_everything(int(args.seed))
    cfg = base.jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(cfg)
    bundle = build_direct_bundle(args, source_ckpt, cfg.device)
    params = list(bundle.tokenizer.parameters()) + [parameter for parameter in bundle.combiner.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    start_epoch, best_psnr, best_goal_psnr = load_resume(args, bundle, optimizer)
    if args.resume and bool(args.reset_best_on_resume):
        best_psnr = float("-inf")
        best_goal_psnr = float("-inf")
        print("reset best scores after resume", flush=True)
    print_header(args, bundle, len(train_loader.dataset), len(val_loader.dataset))

    if bool(args.eval_init_only):
        args._usage_weight = explore.usage_weight(args, max(1, start_epoch))
        calibration = calibrate_fsq_batch_norm(train_loader, bundle, args, cfg.device)
        metrics = validate(val_loader, bundle, args, cfg.device)
        if calibration is not None:
            metrics.update({f"bn_calibration_{key}": value for key, value in calibration.items()})
        metrics["goal_eligible"] = float(goal_eligible(metrics, args))
        print(f"[layer2-fsq-direct init val] {display_metrics(metrics)} goal_eligible={bool(metrics['goal_eligible'])}", flush=True)
        return

    if start_epoch > int(args.epochs):
        raise ValueError(f"resume starts at epoch {start_epoch}, beyond --epochs {int(args.epochs)}")

    last_metrics: dict[str, float] = {}
    for epoch in range(start_epoch, int(args.epochs) + 1):
        args._usage_weight = explore.usage_weight(args, epoch)
        bundle.e1.eval()
        bundle.d1.eval()
        bundle.tokenizer.train()
        bundle.combiner.train()
        if bool(args.freeze_combiner):
            bundle.combiner.inner.eval()
        meters = base.meters(METRIC_NAMES)
        levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
        hist = torch.zeros(base.vocab_size(levels), dtype=torch.float32)
        level_hists = make_level_hists(levels)
        t0 = time.time()
        for batch_idx, (imgs, _labels) in enumerate(train_loader, start=1):
            if int(args.max_train_batches) > 0 and batch_idx > int(args.max_train_batches):
                break
            imgs = imgs.to(cfg.device, non_blocking=True)
            layer1_out, out = forward_direct(bundle, imgs)
            losses = compute_losses(out, imgs, args)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip_norm))
            optimizer.step()
            update_metrics(meters, out, layer1_out, imgs, losses, bundle.combiner, args)
            base.update_code_hist(hist, out["idx3"])
            update_level_hists(level_hists, out["codes"])

        last_metrics = finalize_metrics(meters, hist, level_hists, args)
        last_metrics["usage_weight"] = float(args._usage_weight)
        print(
            f"[layer2-fsq-direct train {epoch:03d}/{int(args.epochs):03d}] "
            f"{display_metrics(last_metrics)} usage_weight={float(args._usage_weight):g} "
            f"time={time.time() - t0:.1f}s",
            flush=True,
        )

        if base.should_validate(args, epoch):
            calibration = calibrate_fsq_batch_norm(train_loader, bundle, args, cfg.device)
            val_metrics = validate(val_loader, bundle, args, cfg.device)
            if calibration is not None:
                val_metrics.update({f"bn_calibration_{key}": value for key, value in calibration.items()})
            eligible = goal_eligible(val_metrics, args)
            val_metrics["goal_eligible"] = float(eligible)
            psnr = float(val_metrics["psnr_final"])
            print(
                f"[layer2-fsq-direct val {epoch:03d}] {display_metrics(val_metrics)} "
                f"goal_eligible={eligible}",
                flush=True,
            )
            improved_best = psnr > best_psnr
            improved_goal = eligible and psnr > best_goal_psnr
            if improved_best:
                best_psnr = psnr
            if improved_goal:
                best_goal_psnr = psnr
            if improved_best:
                save_checkpoint(
                    base.jsccf_io.ckpt_path(args, direct_name(args), "best"),
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    bundle=bundle,
                    optimizer=optimizer,
                    best_psnr=best_psnr,
                    best_goal_psnr=best_goal_psnr,
                )
            if improved_goal:
                save_checkpoint(
                    base.jsccf_io.ckpt_path(args, direct_name(args), "goal_best"),
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    bundle=bundle,
                    optimizer=optimizer,
                    best_psnr=best_psnr,
                    best_goal_psnr=best_goal_psnr,
                )

        if base.should_save_latest(args, epoch):
            save_checkpoint(
                base.jsccf_io.ckpt_path(args, direct_name(args), "latest"),
                epoch=epoch,
                args=args,
                metrics=last_metrics,
                bundle=bundle,
                optimizer=optimizer,
                best_psnr=best_psnr,
                best_goal_psnr=best_goal_psnr,
            )

    save_checkpoint(
        base.jsccf_io.ckpt_path(args, direct_name(args), "latest"),
        epoch=int(args.epochs),
        args=args,
        metrics=last_metrics,
        bundle=bundle,
        optimizer=optimizer,
        best_psnr=best_psnr,
        best_goal_psnr=best_goal_psnr,
    )


@torch.no_grad()
def smoke_shapes(args: argparse.Namespace, source_ckpt: dict) -> None:
    base.seed_everything(int(args.seed))
    device = torch.device("cuda:0" if (not bool(args.cpu)) and torch.cuda.is_available() else "cpu")
    bundle = build_direct_bundle(args, source_ckpt, device)
    args._usage_weight = explore.usage_weight(args, 1)
    imgs = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    layer1_out, out = forward_direct(bundle, imgs)
    expected_q = (int(args.smoke_batch_size), int(args.fsq_d), int(args.latent_h), int(args.latent_w))
    expected_img = (int(args.smoke_batch_size), 3, 256, 256)
    print(
        f"[smoke direct] arch={args.arch} x1={tuple(layer1_out['x1'].shape)} "
        f"z3={tuple(out['z3'].shape)} q3={tuple(out['q3'].shape)} "
        f"u2={tuple(out['u2_hat'].shape)} final={tuple(out['final'].shape)} "
        f"alpha={float(bundle.combiner.alpha().item()):.6f}",
        flush=True,
    )
    if tuple(out["q3"].shape) != expected_q:
        raise RuntimeError(f"expected q3 {expected_q}, got {tuple(out['q3'].shape)}")
    if tuple(out["final"].shape) != expected_img:
        raise RuntimeError(f"expected final {expected_img}, got {tuple(out['final'].shape)}")


def parse_args() -> argparse.Namespace:
    direct_parser = argparse.ArgumentParser(add_help=False)
    direct_parser.add_argument("--codec-init", choices=["compatible", "fresh"], default="compatible")
    direct_parser.add_argument("--fresh-combiner", action="store_true")
    direct_parser.add_argument("--freeze-combiner", action="store_true")
    direct_parser.add_argument("--combiner-mode", choices=["blend", "original"], default="blend")
    direct_parser.add_argument("--blend-init", type=float, default=0.1)
    direct_parser.add_argument("--lambda-u2-img", type=float, default=0.0)
    direct_parser.add_argument("--selection-min-delta-x1", type=float, default=0.0)
    direct_parser.add_argument(
        "--bn-calibration-batches",
        type=int,
        default=0,
        help="Recompute exact pre-FSQ BatchNorm moments from this many train batches before validation; 0 disables it.",
    )
    direct_parser.add_argument("--resume", type=str, default="")
    direct_parser.add_argument("--reset-best-on-resume", action="store_true")
    direct_parser.add_argument("--match-source-width", dest="match_source_width", action="store_true")
    direct_parser.add_argument("--no-match-source-width", dest="match_source_width", action="store_false")
    direct_parser.set_defaults(match_source_width=True)
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Direct Layer2 options: --codec-init {compatible,fresh}; --fresh-combiner; "
            "--freeze-combiner; --combiner-mode {blend,original}; --blend-init FLOAT; "
            "--lambda-u2-img FLOAT; --selection-min-delta-x1 FLOAT; --bn-calibration-batches INT; --resume PATH; "
            "--reset-best-on-resume; "
            "--[no-]match-source-width.\n",
            flush=True,
        )
    direct_args, remaining = direct_parser.parse_known_args()
    if not 0.0 < float(direct_args.blend_init) < 1.0:
        raise ValueError("--blend-init must be strictly between 0 and 1")
    if float(direct_args.lambda_u2_img) < 0.0:
        raise ValueError("--lambda-u2-img must be non-negative")
    if int(direct_args.bn_calibration_batches) < 0:
        raise ValueError("--bn-calibration-batches must be non-negative")

    argv = sys.argv
    try:
        sys.argv = [argv[0], *remaining]
        args = explore.parse_args()
    finally:
        sys.argv = argv

    for key, value in vars(direct_args).items():
        setattr(args, key, value)
    if not cli_option_present(remaining, "--save-dir"):
        args.save_dir = str(THIS_DIR / "checkpoints-direct")
    if not cli_option_present(remaining, "--log-file"):
        args.log_file = ""
    if not cli_option_present(remaining, "--version"):
        args.version = "layer2-direct-fsq"
    return args


def main() -> None:
    args = parse_args()
    args.stage = "layer2_fsq_direct"
    base.apply_preset(args)
    base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    explore.validate_explore_args(args)
    source_ckpt = base.load_teacher_checkpoint_for_args(args)
    if str(args.condition_mode) != "none":
        raise ValueError("direct capacity comparisons currently require --condition-mode none")
    if str(args.variant) != "combiner":
        raise ValueError(f"direct route currently requires a combiner Layer2 source, got variant={args.variant!r}")
    if bool(args.match_source_width) and str(args.arch) == "cnn" and str(args.codec_init) == "compatible":
        args.e3_base_ch = int(args.cnn_base_ch)
        args.d3_base_ch = int(args.cnn_base_ch)
        args.e3_num_res = int(args.cnn_num_res)
        args.d3_num_res = int(args.cnn_num_res)
    args._usage_weight = explore.usage_weight(args, 1)
    explore.ExploreIFSQQuantizer.config = args
    base.check_jsccf_args(args)

    Path(base.resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(
            THIS_DIR / "logs-direct" / f"{direct_name(args)}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}.log"
        )
    base.setup_log_file(args.log_file)
    base.write_json(
        Path(base.resolve_path(args.save_dir))
        / f"{direct_name(args)}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}_args.json",
        {key: value for key, value in vars(args).items() if not key.startswith("_")},
    )
    if bool(args.smoke_shapes):
        smoke_shapes(args, source_ckpt)
        return
    train(args, source_ckpt)


if __name__ == "__main__":
    main()
