#!/usr/bin/env python3
"""Train the direct Layer2 path with a continuous d-dimensional bottleneck.

This control uses the same frozen Layer1, E2/D2 family, BatchNorm/tanh range,
and original combiner as the FSQ runs, but bypasses rounding during both train
and validation.  It is an empirical same-architecture continuous control for
diagnosing whether finite FSQ precision or the learned d-dimensional codec is
the bottleneck.  It is not an information-theoretic rate-distortion bound.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_layer2_fsq_direct as direct  # noqa: E402


base = direct.base


class ContinuousDiagnosticQuantizer(nn.Module):
    """Keep pre-normalization but pass tanh(z) to D2 without rounding.

    Hard codes are still computed for occupancy diagnostics only.  They never
    enter the reconstruction path or its gradient.
    """

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.channels = int(source.channels)
        self.register_buffer("levels", source.levels.detach().clone())
        self.register_buffer("multipliers", source.multipliers.detach().clone())
        self.pre_norm = source.pre_norm

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        multipliers = self.multipliers.to(device=codes.device, dtype=torch.long).view(
            1, self.channels, 1, 1
        )
        return (codes.long() * multipliers).sum(dim=1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.ndim != 4 or int(z.shape[1]) != self.channels:
            raise ValueError(f"expected z [B,{self.channels},H,W], got {tuple(z.shape)}")
        z_norm = torch.tanh(self.pre_norm(z))
        levels = self.levels.to(device=z.device, dtype=z_norm.dtype).view(1, self.channels, 1, 1)
        span = (levels - 1.0).clamp_min(1.0)
        codes = (((z_norm + 1.0) * 0.5 * span).round().clamp_min(0.0).minimum(span)).long()
        zero = z_norm.new_zeros(())
        return {
            "z3_norm": z_norm,
            "q3": z_norm,
            "q3_hard": z_norm.detach(),
            "codes": codes,
            "idx3": self.codes_to_indices(codes),
            "fsq_mse": zero,
            "usage_kl": zero,
            "soft_level_entropy_bits": zero,
            "soft_usage_entropy_bits": zero,
        }


def build_bundle(args: argparse.Namespace, source_ckpt: dict, device: torch.device) -> direct.DirectBundle:
    bundle = direct.build_direct_bundle(args, source_ckpt, device)
    bundle.tokenizer.quantizer = ContinuousDiagnosticQuantizer(bundle.tokenizer.quantizer).to(device)
    return bundle


def control_name(args: argparse.Namespace) -> str:
    return (
        f"layer2_continuous_direct_{args.arch}_d{int(args.fsq_d)}_"
        f"{direct.normalizer_name(args)}_{args.codec_init}_{args.combiner_mode}"
    )


def print_header(args: argparse.Namespace, bundle: direct.DirectBundle, train_n: int, val_n: int) -> None:
    print(f"=== Layer 2 | continuous d={int(args.fsq_d)} control | {args.arch} ===", flush=True)
    print(f"save_dir={base.resolve_path(args.save_dir)}", flush=True)
    print("实验设计", flush=True)
    print("  frozen Layer1 -> trainable E2 -> BatchNorm/tanh -> trainable D2/original combiner", flush=True)
    print("  FSQ rounding=disabled; u2_teacher=disabled; source Layer2 forward=disabled", flush=True)
    print("  role=same-d empirical continuous control, not a K-specific or mathematical upper bound", flush=True)
    print("loss设计", flush=True)
    print("  L=MSE(final,img); no u2 target; no usage/KL", flush=True)
    print("模块选择", flush=True)
    print(
        f"  E1={base.trainable_state(bundle.e1)} D1={base.trainable_state(bundle.d1)} "
        f"E2={base.trainable_state(bundle.tokenizer.e3)} D2={base.trainable_state(bundle.tokenizer.d3)} "
        f"combiner={base.trainable_state(bundle.combiner)}",
        flush=True,
    )
    print(
        f"  normalizer={direct.normalizer_name(args)} combiner={args.combiner_mode} "
        f"bn_calibration_batches={int(args.bn_calibration_batches)} workers={int(args.num_workers)}/{int(args.val_num_workers)}",
        flush=True,
    )
    print(
        f"epochs={int(args.epochs)} train={train_n} valid={val_n} batch={int(args.batch_size)} "
        f"lr={float(args.lr):g} weight_decay={float(args.weight_decay):g}",
        flush=True,
    )


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    bundle: direct.DirectBundle,
    optimizer: optim.Optimizer,
    best_psnr: float,
    best_goal_psnr: float,
) -> None:
    output = Path(base.resolve_path(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": getattr(base.jsccf_io, "ROUTE", "TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256"),
            "stage": "layer2_continuous_direct",
            "epoch": int(epoch),
            "metrics": metrics,
            "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
            "version": str(args.version),
            "source_layer2_ckpt": str(args.layer2_ckpt),
            "e1_state_dict": bundle.e1.state_dict(),
            "d1_state_dict": bundle.d1.state_dict(),
            "tokenizer_state_dict": bundle.tokenizer.state_dict(),
            "combiner_state_dict": bundle.combiner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": direct.capture_rng_state(),
            "best_psnr": float(best_psnr),
            "best_goal_psnr": float(best_goal_psnr),
            "init_report": bundle.init_report,
            "latent": {
                "z1": [int(args.latent_ch), int(args.latent_h), int(args.latent_w)],
                "z2_continuous": [int(args.fsq_d), int(args.latent_h), int(args.latent_w)],
            },
        },
        output,
    )
    print(f"saved checkpoint: {output}", flush=True)


def load_resume(
    args: argparse.Namespace,
    bundle: direct.DirectBundle,
    optimizer: optim.Optimizer,
) -> tuple[int, float, float]:
    if not args.resume:
        return 1, float("-inf"), float("-inf")
    payload = torch.load(base.resolve_path(args.resume), map_location="cpu")
    if str(payload.get("stage", "")) != "layer2_continuous_direct":
        raise ValueError(f"not a continuous Layer2 control checkpoint: {args.resume}")
    saved = payload.get("args", {})
    for key in (
        "arch",
        "fsq_d",
        "fsq_normalizer",
        "no_pre_norm",
        "codec_init",
        "combiner_mode",
        "fresh_combiner",
        "freeze_combiner",
        "match_source_width",
    ):
        if str(saved.get(key)) != str(getattr(args, key)):
            raise ValueError(f"resume mismatch for {key}: checkpoint={saved.get(key)!r} current={getattr(args, key)!r}")
    if int(saved.get("bn_calibration_batches", 0)) != int(args.bn_calibration_batches) and not bool(
        args.reset_best_on_resume
    ):
        raise ValueError("changing BN calibration on resume requires --reset-best-on-resume")
    base.jsccf_io.load_state(bundle.e1, payload["e1_state_dict"], "resume_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, payload["d1_state_dict"], "resume_D1", strict=True)
    base.jsccf_io.load_state(bundle.tokenizer, payload["tokenizer_state_dict"], "resume_continuous", strict=True)
    base.jsccf_io.load_state(bundle.combiner, payload["combiner_state_dict"], "resume_combiner", strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if "rng_state" in payload:
        direct.restore_rng_state(payload["rng_state"])
    start = int(payload.get("epoch", 0)) + 1
    print(f"resumed {args.resume} at epoch {start}", flush=True)
    return start, float(payload.get("best_psnr", float("-inf"))), float(
        payload.get("best_goal_psnr", float("-inf"))
    )


def train(args: argparse.Namespace, source_ckpt: dict) -> None:
    base.seed_everything(int(args.seed))
    cfg = base.jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(cfg)
    bundle = build_bundle(args, source_ckpt, cfg.device)
    params = list(bundle.tokenizer.parameters()) + [
        parameter for parameter in bundle.combiner.parameters() if parameter.requires_grad
    ]
    optimizer = optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    start_epoch, best_psnr, best_goal_psnr = load_resume(args, bundle, optimizer)
    if args.resume and bool(args.reset_best_on_resume):
        best_psnr = float("-inf")
        best_goal_psnr = float("-inf")
    print_header(args, bundle, len(train_loader.dataset), len(val_loader.dataset))

    if bool(args.eval_init_only):
        calibration = direct.calibrate_fsq_batch_norm(train_loader, bundle, args, cfg.device)
        metrics = direct.validate(val_loader, bundle, args, cfg.device)
        if calibration is not None:
            metrics.update({f"bn_calibration_{key}": value for key, value in calibration.items()})
        eligible = direct.goal_eligible(metrics, args)
        print(f"[layer2-continuous init val] {direct.display_metrics(metrics)} goal_eligible={eligible}", flush=True)
        return

    last_train: dict[str, float] = {}
    last_checkpoint: dict[str, float] = {}
    name = control_name(args)
    levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    for epoch in range(start_epoch, int(args.epochs) + 1):
        bundle.e1.eval()
        bundle.d1.eval()
        bundle.tokenizer.train()
        bundle.combiner.train()
        if bool(args.freeze_combiner):
            bundle.combiner.inner.eval()
        meters = base.meters(direct.METRIC_NAMES)
        hist = torch.zeros(base.vocab_size(levels), dtype=torch.float32)
        level_hists = direct.make_level_hists(levels)
        started = time.time()
        for batch_index, (imgs, _labels) in enumerate(train_loader, start=1):
            if int(args.max_train_batches) > 0 and batch_index > int(args.max_train_batches):
                break
            imgs = imgs.to(cfg.device, non_blocking=True)
            layer1_out, out = direct.forward_direct(bundle, imgs)
            losses = direct.compute_losses(out, imgs, args)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip_norm))
            optimizer.step()
            direct.update_metrics(meters, out, layer1_out, imgs, losses, bundle.combiner, args)
            base.update_code_hist(hist, out["idx3"])
            direct.update_level_hists(level_hists, out["codes"])
        last_train = direct.finalize_metrics(meters, hist, level_hists, args)
        print(
            f"[layer2-continuous train {epoch:03d}/{int(args.epochs):03d}] "
            f"{direct.display_metrics(last_train)} time={time.time() - started:.1f}s",
            flush=True,
        )

        checkpoint_metrics = last_train
        if base.should_validate(args, epoch):
            calibration = direct.calibrate_fsq_batch_norm(train_loader, bundle, args, cfg.device)
            val_metrics = direct.validate(val_loader, bundle, args, cfg.device)
            if calibration is not None:
                val_metrics.update({f"bn_calibration_{key}": value for key, value in calibration.items()})
            eligible = direct.goal_eligible(val_metrics, args)
            val_metrics["goal_eligible"] = float(eligible)
            checkpoint_metrics = val_metrics
            psnr = float(val_metrics["psnr_final"])
            print(
                f"[layer2-continuous val {epoch:03d}] {direct.display_metrics(val_metrics)} "
                f"goal_eligible={eligible}",
                flush=True,
            )
            improved = psnr > best_psnr
            improved_goal = eligible and psnr > best_goal_psnr
            if improved:
                best_psnr = psnr
            if improved_goal:
                best_goal_psnr = psnr
            if improved:
                save_checkpoint(
                    base.jsccf_io.ckpt_path(args, name, "best"), epoch=epoch, args=args,
                    metrics=val_metrics, bundle=bundle, optimizer=optimizer,
                    best_psnr=best_psnr, best_goal_psnr=best_goal_psnr,
                )
            if improved_goal:
                save_checkpoint(
                    base.jsccf_io.ckpt_path(args, name, "goal_best"), epoch=epoch, args=args,
                    metrics=val_metrics, bundle=bundle, optimizer=optimizer,
                    best_psnr=best_psnr, best_goal_psnr=best_goal_psnr,
                )
        last_checkpoint = checkpoint_metrics
        if base.should_save_latest(args, epoch):
            save_checkpoint(
                base.jsccf_io.ckpt_path(args, name, "latest"), epoch=epoch, args=args,
                metrics=checkpoint_metrics, bundle=bundle, optimizer=optimizer,
                best_psnr=best_psnr, best_goal_psnr=best_goal_psnr,
            )
    save_checkpoint(
        base.jsccf_io.ckpt_path(args, name, "latest"), epoch=int(args.epochs), args=args,
        metrics=last_checkpoint or last_train, bundle=bundle, optimizer=optimizer,
        best_psnr=best_psnr, best_goal_psnr=best_goal_psnr,
    )


@torch.no_grad()
def smoke_shapes(args: argparse.Namespace, source_ckpt: dict) -> None:
    device = torch.device("cuda:0" if not bool(args.cpu) and torch.cuda.is_available() else "cpu")
    bundle = build_bundle(args, source_ckpt, device)
    imgs = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    layer1_out, out = direct.forward_direct(bundle, imgs)
    if not torch.equal(out["q3"], out["z3_norm"]):
        raise RuntimeError("continuous control unexpectedly changed z_norm")
    print(
        f"[smoke continuous] arch={args.arch} x1={tuple(layer1_out['x1'].shape)} "
        f"z={tuple(out['z3_norm'].shape)} final={tuple(out['final'].shape)} fsq_mse={float(out['fsq_mse']):g}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    args = direct.parse_args()
    if not direct.cli_option_present(sys.argv[1:], "--combiner-mode"):
        args.combiner_mode = "original"
    if not direct.cli_option_present(sys.argv[1:], "--val-num-workers"):
        args.val_num_workers = 4
    if not direct.cli_option_present(sys.argv[1:], "--selection-min-drop-zero"):
        args.selection_min_drop_zero = 0.1
    if not direct.cli_option_present(sys.argv[1:], "--selection-min-drop-shuffle"):
        args.selection_min_drop_shuffle = 0.1
    return args


def main() -> None:
    args = parse_args()
    args.stage = "layer2_continuous_direct"
    base.apply_preset(args)
    if str(args.preset) != "custom":
        raise ValueError("continuous control requires --preset custom")
    base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    if str(args.combiner_mode) != "original":
        raise ValueError("continuous ceiling control requires --combiner-mode original")
    if float(args.lambda_usage) != 0.0 or float(args.lambda_u2_img) != 0.0:
        raise ValueError("continuous control requires --lambda-usage 0 and --lambda-u2-img 0")
    source_ckpt = base.load_teacher_checkpoint_for_args(args)
    if str(args.condition_mode) != "none" or str(args.variant) != "combiner":
        raise ValueError("continuous control requires --condition-mode none and combiner source")
    args._usage_weight = 0.0
    direct.explore.ExploreIFSQQuantizer.config = args
    base.check_jsccf_args(args)
    if not direct.cli_option_present(sys.argv[1:], "--save-dir"):
        args.save_dir = str(THIS_DIR / "checkpoints-continuous")
    if not direct.cli_option_present(sys.argv[1:], "--version"):
        args.version = "layer2-continuous-d3"
    name = control_name(args)
    if not direct.cli_option_present(sys.argv[1:], "--log-file"):
        args.log_file = str(
            THIS_DIR / "logs-continuous" / f"{name}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}.log"
        )
    Path(base.resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    base.setup_log_file(args.log_file)
    base.write_json(
        Path(base.resolve_path(args.save_dir))
        / f"{name}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}_args.json",
        {key: value for key, value in vars(args).items() if not key.startswith("_")},
    )
    if bool(args.smoke_shapes):
        smoke_shapes(args, source_ckpt)
        return
    train(args, source_ckpt)


if __name__ == "__main__":
    main()
