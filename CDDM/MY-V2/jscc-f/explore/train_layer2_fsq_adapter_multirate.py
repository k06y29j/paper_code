#!/usr/bin/env python3
"""Shared nested-rate FSQ inside the exact width-320 Swin Layer2 codec.

This opt-in route combines the two useful controls from the existing
exploration scripts:

* keep the pretrained source E2/D2 at their native 320-channel width and put a
  PCA-initialized ``320 -> d -> 320`` adapter around the discrete bottleneck;
* quantize one shared ``d=3`` normalized latent on nested scalar grids
  ``L=5,9,17`` and decode every rate with the same synthesis adapter, D2, and
  identity-safe residual combiner.

The training graph is therefore::

    image -> frozen E1/D1 -> x1
    cat(x1,image) -> source E2_320 -> analysis adapter -> pre_norm/tanh
                                                        |-> FSQ L=5  --|
                                                        |-> FSQ L=9  --+-> shared synthesis/D2/residual combiner
                                                        `-> FSQ L=17 --|

There is no source-u2 teacher forward, usage/KL objective, or continuous side
path during training.  The continuous, zero-code, and shared-permutation
shuffle branches are validation diagnostics only.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_layer2_fsq_adapter as adapter  # noqa: E402
import train_layer2_fsq_direct as direct  # noqa: E402
import train_layer2_fsq_multirate as multirate  # noqa: E402


base = direct.base


def encode_adapter_multirate(
    bundle: direct.DirectBundle,
    imgs: torch.Tensor,
    layer1_out: dict[str, torch.Tensor],
    levels: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    """Encode E2 once, normalize once, then form every nested FSQ branch."""

    e2_input = torch.cat([layer1_out["x1"], imgs], dim=1)
    z320 = base.encode_tensor(bundle.tokenizer.e3, e2_input)
    z3 = bundle.tokenizer.analysis_adapter(z320)
    z_norm = torch.tanh(bundle.tokenizer.quantizer.pre_norm(z3))
    outputs: dict[int, dict[str, torch.Tensor]] = {}
    for level in levels:
        encoded = multirate.quantize_at_level(z_norm, int(level))
        decoded = bundle.tokenizer.decode(
            encoded["q3"],
            layer1_out["x1"],
            layer1_out["z1"],
            bundle.combiner,
        )
        outputs[int(level)] = {
            **encoded,
            **decoded,
            "z320": z320,
            "z3": z3,
            "q3_used": encoded["q3"],
        }
    return z320, z3, z_norm, outputs


def forward_adapter_multirate(
    bundle: direct.DirectBundle,
    imgs: torch.Tensor,
    levels: list[int],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    with torch.no_grad():
        layer1_out = bundle.layer1(imgs)
    _z320, _z3, z_norm, outputs = encode_adapter_multirate(bundle, imgs, layer1_out, levels)
    return layer1_out, z_norm, outputs


@torch.no_grad()
def validate(
    loader,
    bundle: direct.DirectBundle,
    args: argparse.Namespace,
    device: torch.device,
    levels: list[int],
    recon_weights: list[float],
    margins: list[float],
) -> dict[str, float]:
    bundle.e1.eval()
    bundle.d1.eval()
    bundle.tokenizer.eval()
    bundle.combiner.eval()
    rate_states = multirate.make_rate_states(args, levels, validation=True)
    objective_meters = base.meters(multirate.objective_metric_names(levels))

    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if int(args.max_val_batches) > 0 and batch_index > int(args.max_val_batches):
            break
        imgs = imgs.to(device, non_blocking=True)
        layer1_out, z_norm, outputs = forward_adapter_multirate(bundle, imgs, levels)
        objective, branch_losses = multirate.compute_multirate_loss(
            outputs,
            imgs,
            levels,
            recon_weights,
            margins,
            float(args.lambda_monotonic),
        )
        multirate.update_objective_meters(objective_meters, objective, int(imgs.shape[0]))
        for level in levels:
            multirate.update_rate_state(
                rate_states[level],
                outputs[level],
                layer1_out,
                imgs,
                branch_losses[level],
                bundle,
            )

        if bool(args.val_ablation):
            x1 = layer1_out["x1"]
            z1 = layer1_out["z1"]
            continuous_final = bundle.tokenizer.decode(z_norm, x1, z1, bundle.combiner)["final"]
            zero_final = bundle.tokenizer.decode(
                torch.zeros_like(z_norm), x1, z1, bundle.combiner
            )["final"]
            token_count = int(imgs.shape[0] * z_norm.shape[2] * z_norm.shape[3])
            shared_permutation = torch.randperm(token_count, device=z_norm.device)
            for level in levels:
                shuffled_q = multirate.shuffled_with_perm(
                    outputs[level]["q3"], shared_permutation
                )
                shuffle_final = bundle.tokenizer.decode(
                    shuffled_q, x1, z1, bundle.combiner
                )["final"]
                multirate.update_rate_ablation(
                    rate_states[level],
                    outputs[level],
                    imgs,
                    continuous_final,
                    zero_final,
                    shuffle_final,
                )

    return multirate.finalize_all_metrics(objective_meters, rate_states, levels)


def route_name(args: argparse.Namespace, levels: list[int]) -> str:
    tag = "-".join(str(level) for level in levels)
    return (
        f"layer2_fsq_adapter_multirate_{args.arch}_d{int(args.fsq_d)}_l{tag}_"
        f"{direct.normalizer_name(args)}_{args.adapter_init}_h{int(args.adapter_hidden)}_"
        f"{args.adapter_combiner}"
    )


def rate_metadata(args: argparse.Namespace, levels: list[int]) -> dict[str, dict[str, int | list[int]]]:
    return {
        str(level): {
            "levels": multirate.repeated_levels(level, int(args.fsq_d)),
            "vocab_size": int(level) ** int(args.fsq_d),
            "fixed_bits_per_token": int(
                math.ceil(math.log2(float(int(level) ** int(args.fsq_d))))
            ),
        }
        for level in levels
    }


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    levels: list[int],
    metrics: dict[str, float],
    bundle: direct.DirectBundle,
    optimizer: optim.Optimizer,
    best_score: float,
    best_goal_score: float,
    init_stats: dict[str, float],
) -> None:
    output = Path(base.resolve_path(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "route": getattr(base.jsccf_io, "ROUTE", "TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256"),
        "stage": "layer2_fsq_adapter_multirate",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "version": str(args.version),
        "source_layer2_ckpt": str(args.layer2_ckpt),
        "nested_scalar_levels": list(levels),
        "rates": rate_metadata(args, levels),
        "adapter_combiner": str(args.adapter_combiner),
        "e1_state_dict": bundle.e1.state_dict(),
        "d1_state_dict": bundle.d1.state_dict(),
        "tokenizer_state_dict": bundle.tokenizer.state_dict(),
        "combiner_state_dict": bundle.combiner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": direct.capture_rng_state(),
        "best_score": float(best_score),
        "best_goal_score": float(best_goal_score),
        "init_stats": init_stats,
        "init_report": bundle.init_report,
    }
    torch.save(payload, output)
    print(f"saved checkpoint: {output}", flush=True)


MODEL_RESUME_KEYS = (
    "arch",
    "swin_codec",
    "variant",
    "condition_mode",
    "fsq_d",
    "fsq_normalizer",
    "no_pre_norm",
    "adapter_init",
    "adapter_hidden",
    "adapter_combiner",
    "lambda_monotonic",
    "monotonic_margins",
    "recon_weights",
)

# These fields can alter a validation score, an ablation, or goal eligibility.
# A user may change them only by explicitly starting a new best-score regime.
VALIDATION_PROTOCOL_KEYS = (
    "data_dir",
    "test_batch",
    "max_val_batches",
    "val_ablation",
    "bn_calibration_batches",
    "seed",
    "selection_min_rate_gain_db",
    "selection_min_per_image_strict_ratio",
    "selection_min_delta_x1",
    "selection_min_drop_zero",
    "selection_min_drop_shuffle",
)


def load_resume(
    args: argparse.Namespace,
    bundle: direct.DirectBundle,
    optimizer: optim.Optimizer,
    levels: list[int],
) -> tuple[int, float, float, dict[str, float] | None]:
    if not args.resume:
        return 1, -math.inf, -math.inf, None
    payload = torch.load(base.resolve_path(args.resume), map_location="cpu")
    if str(payload.get("stage", "")) != "layer2_fsq_adapter_multirate":
        raise ValueError(f"not an adapter multirate checkpoint: {args.resume}")
    saved_levels = [int(value) for value in payload.get("nested_scalar_levels", [])]
    if saved_levels != levels:
        raise ValueError(f"resume nested levels mismatch: {saved_levels} != {levels}")
    if str(payload.get("adapter_combiner", "")) != str(args.adapter_combiner):
        raise ValueError(
            "resume adapter_combiner mismatch: "
            f"{payload.get('adapter_combiner')!r} != {args.adapter_combiner!r}"
        )
    saved = payload.get("args", {})
    for key in MODEL_RESUME_KEYS:
        if str(saved.get(key)) != str(getattr(args, key)):
            raise ValueError(
                f"resume mismatch for {key}: checkpoint={saved.get(key)!r} "
                f"current={getattr(args, key)!r}"
            )
    changed_protocol = [
        key for key in VALIDATION_PROTOCOL_KEYS if str(saved.get(key)) != str(getattr(args, key))
    ]
    if changed_protocol and not bool(args.reset_best_on_resume):
        raise ValueError(
            "resume changes validation protocol "
            f"{changed_protocol}; pass --reset-best-on-resume to start a new selection regime"
        )

    base.jsccf_io.load_state(bundle.e1, payload["e1_state_dict"], "resume_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, payload["d1_state_dict"], "resume_D1", strict=True)
    base.jsccf_io.load_state(
        bundle.tokenizer,
        payload["tokenizer_state_dict"],
        "resume_adapter_multirate",
        strict=True,
    )
    base.jsccf_io.load_state(
        bundle.combiner,
        payload["combiner_state_dict"],
        "resume_residual_combiner",
        strict=True,
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if "rng_state" in payload:
        direct.restore_rng_state(payload["rng_state"])
    best_score = float(payload.get("best_score", -math.inf))
    best_goal_score = float(payload.get("best_goal_score", -math.inf))
    if bool(args.reset_best_on_resume):
        best_score = -math.inf
        best_goal_score = -math.inf
        print("reset best scores after validation-protocol change", flush=True)
    start_epoch = int(payload.get("epoch", 0)) + 1
    print(f"resumed {args.resume} at epoch {start_epoch}", flush=True)
    return start_epoch, best_score, best_goal_score, payload.get("init_stats")


def print_header(
    args: argparse.Namespace,
    bundle: direct.DirectBundle,
    levels: list[int],
    recon_weights: list[float],
    margins: list[float],
    train_count: int,
    val_count: int,
    init_stats: dict[str, float],
) -> None:
    rates = ", ".join(
        f"L={level}:K={int(level) ** int(args.fsq_d)}:"
        f"bits/token={math.ceil(math.log2(int(level) ** int(args.fsq_d)))}"
        for level in levels
    )
    print("=== Layer 2 | width320 PCA-adapter shared nested FSQ | Swin ===", flush=True)
    print(
        f"device={'cpu' if args.cpu else 'cuda:0'} "
        f"visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    print(f"save_dir={base.resolve_path(args.save_dir)}", flush=True)
    print("实验设计", flush=True)
    print(
        "  frozen Layer1; exact source E2_320/D2_320; one PCA analysis/synthesis adapter; "
        "one residual combiner",
        flush=True,
    )
    print(
        "  shared E2_320 -> analysis -> pre_norm/tanh z_norm; nested FSQ rates share "
        "synthesis/D2_320/combiner",
        flush=True,
    )
    print(f"  rates={rates}; init_stats={init_stats}", flush=True)
    print("  no u2 teacher forward, no usage/KL loss, no continuous training bypass", flush=True)
    print("loss设计", flush=True)
    print(
        f"  recon=weighted_mean(MSE(final_L,img)); weights={recon_weights}; "
        f"lambda_monotonic={float(args.lambda_monotonic):g}; margins={margins}",
        flush=True,
    )
    print("  hinge=relu(MSE_higher-MSE_lower.detach()+margin), per image", flush=True)
    print("模块选择", flush=True)
    print(
        f"  E1={base.trainable_state(bundle.e1)} D1={base.trainable_state(bundle.d1)} "
        f"E2_320={base.trainable_state(bundle.tokenizer.e3)} adapters=trainable "
        f"D2_320={base.trainable_state(bundle.tokenizer.d3)} "
        f"combiner={args.adapter_combiner}:{base.trainable_state(bundle.combiner)}",
        flush=True,
    )
    print(
        f"  normalizer={direct.normalizer_name(args)} BN_calibration={int(args.bn_calibration_batches)}; "
        "validation=continuous/zero/shared-permutation-shuffle per rate",
        flush=True,
    )
    print(
        f"  goal=all final>x1; adjacent gain>{float(args.selection_min_rate_gain_db):g}dB; "
        f"per-image strict>={float(args.selection_min_per_image_strict_ratio):g}; "
        f"drop0>={float(args.selection_min_drop_zero):g}; "
        f"dropshuffle>={float(args.selection_min_drop_shuffle):g}",
        flush=True,
    )
    print(
        f"epochs={int(args.epochs)} train={train_count} valid={val_count} "
        f"batch={int(args.batch_size)} test_batch={int(args.test_batch)} "
        f"workers={int(args.num_workers)}/{int(args.val_num_workers)} "
        f"lr={float(args.lr):g} weight_decay={float(args.weight_decay):g}",
        flush=True,
    )


def train(args: argparse.Namespace, source_ckpt: dict) -> None:
    levels = multirate.parse_nested_levels(args.nested_levels, int(args.fsq_d))
    recon_weights = multirate.parse_float_list(
        args.recon_weights, len(levels), "--recon-weights"
    )
    margins = multirate.parse_float_list(
        args.monotonic_margins, len(levels) - 1, "--monotonic-margins"
    )
    base.seed_everything(int(args.seed))
    cfg = base.jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(cfg)
    bundle = adapter.build_bundle(args, source_ckpt, cfg.device)
    params = list(bundle.tokenizer.parameters()) + list(bundle.combiner.parameters())
    optimizer = optim.AdamW(
        params, lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    start_epoch, best_score, best_goal_score, saved_init = load_resume(
        args, bundle, optimizer, levels
    )
    init_stats = (
        saved_init
        if saved_init is not None
        else adapter.initialize_pca(train_loader, bundle, args, cfg.device)
    )
    print_header(
        args,
        bundle,
        levels,
        recon_weights,
        margins,
        len(train_loader.dataset),
        len(val_loader.dataset),
        init_stats,
    )

    if bool(args.eval_init_only):
        calibration = adapter.calibrate_batch_norm(train_loader, bundle, args, cfg.device)
        metrics = validate(
            val_loader, bundle, args, cfg.device, levels, recon_weights, margins
        )
        if calibration is not None:
            metrics.update({f"bn_calibration_{key}": value for key, value in calibration.items()})
        metrics["goal_eligible"] = float(multirate.goal_eligible(metrics, args, levels))
        print(
            f"[layer2-fsq-adapter-multirate init val] "
            f"{multirate.display_metrics(metrics, levels)} "
            f"goal_eligible={bool(metrics['goal_eligible'])}",
            flush=True,
        )
        return

    if start_epoch > int(args.epochs):
        raise ValueError(
            f"resume starts at epoch {start_epoch}, beyond --epochs {int(args.epochs)}"
        )

    name = route_name(args, levels)
    last_train: dict[str, float] = {}
    last_checkpoint: dict[str, float] = {}
    for epoch in range(start_epoch, int(args.epochs) + 1):
        bundle.e1.eval()
        bundle.d1.eval()
        bundle.tokenizer.train()
        bundle.combiner.train()
        rate_states = multirate.make_rate_states(args, levels, validation=False)
        objective_meters = base.meters(multirate.objective_metric_names(levels))
        started = time.time()

        for batch_index, (imgs, _labels) in enumerate(train_loader, start=1):
            if int(args.max_train_batches) > 0 and batch_index > int(args.max_train_batches):
                break
            imgs = imgs.to(cfg.device, non_blocking=True)
            layer1_out, _z_norm, outputs = forward_adapter_multirate(bundle, imgs, levels)
            objective, branch_losses = multirate.compute_multirate_loss(
                outputs,
                imgs,
                levels,
                recon_weights,
                margins,
                float(args.lambda_monotonic),
            )
            optimizer.zero_grad(set_to_none=True)
            objective["loss"].backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip_norm))
            optimizer.step()
            multirate.update_objective_meters(
                objective_meters, objective, int(imgs.shape[0])
            )
            for level in levels:
                multirate.update_rate_state(
                    rate_states[level],
                    outputs[level],
                    layer1_out,
                    imgs,
                    branch_losses[level],
                    bundle,
                )

        last_train = multirate.finalize_all_metrics(objective_meters, rate_states, levels)
        print(
            f"[layer2-fsq-adapter-multirate train {epoch:03d}/{int(args.epochs):03d}] "
            f"{multirate.display_metrics(last_train, levels)} "
            f"time={time.time() - started:.1f}s",
            flush=True,
        )

        checkpoint_metrics = last_train
        if base.should_validate(args, epoch):
            calibration = adapter.calibrate_batch_norm(train_loader, bundle, args, cfg.device)
            val_metrics = validate(
                val_loader, bundle, args, cfg.device, levels, recon_weights, margins
            )
            if calibration is not None:
                val_metrics.update(
                    {f"bn_calibration_{key}": value for key, value in calibration.items()}
                )
            eligible = multirate.goal_eligible(val_metrics, args, levels)
            val_metrics["goal_eligible"] = float(eligible)
            checkpoint_metrics = val_metrics
            print(
                f"[layer2-fsq-adapter-multirate val {epoch:03d}] "
                f"{multirate.display_metrics(val_metrics, levels)} goal_eligible={eligible}",
                flush=True,
            )
            score = sum(
                float(val_metrics[f"l{level}_psnr_final"]) for level in levels
            ) / float(len(levels))
            improved = score > best_score
            improved_goal = eligible and score > best_goal_score
            if improved:
                best_score = score
            if improved_goal:
                best_goal_score = score
            if improved:
                save_checkpoint(
                    base.jsccf_io.ckpt_path(args, name, "best"),
                    epoch=epoch,
                    args=args,
                    levels=levels,
                    metrics=val_metrics,
                    bundle=bundle,
                    optimizer=optimizer,
                    best_score=best_score,
                    best_goal_score=best_goal_score,
                    init_stats=init_stats,
                )
            if improved_goal:
                save_checkpoint(
                    base.jsccf_io.ckpt_path(args, name, "goal_best"),
                    epoch=epoch,
                    args=args,
                    levels=levels,
                    metrics=val_metrics,
                    bundle=bundle,
                    optimizer=optimizer,
                    best_score=best_score,
                    best_goal_score=best_goal_score,
                    init_stats=init_stats,
                )

        last_checkpoint = checkpoint_metrics
        if base.should_save_latest(args, epoch):
            save_checkpoint(
                base.jsccf_io.ckpt_path(args, name, "latest"),
                epoch=epoch,
                args=args,
                levels=levels,
                metrics=checkpoint_metrics,
                bundle=bundle,
                optimizer=optimizer,
                best_score=best_score,
                best_goal_score=best_goal_score,
                init_stats=init_stats,
            )

    save_checkpoint(
        base.jsccf_io.ckpt_path(args, name, "latest"),
        epoch=int(args.epochs),
        args=args,
        levels=levels,
        metrics=last_checkpoint or last_train,
        bundle=bundle,
        optimizer=optimizer,
        best_score=best_score,
        best_goal_score=best_goal_score,
        init_stats=init_stats,
    )


def smoke_shapes_and_backward(args: argparse.Namespace, source_ckpt: dict) -> None:
    """Run an actual CPU forward/backward through all shared-rate branches."""

    levels = multirate.parse_nested_levels(args.nested_levels, int(args.fsq_d))
    recon_weights = multirate.parse_float_list(
        args.recon_weights, len(levels), "--recon-weights"
    )
    margins = multirate.parse_float_list(
        args.monotonic_margins, len(levels) - 1, "--monotonic-margins"
    )
    base.seed_everything(int(args.seed))
    device = torch.device("cpu")
    bundle = adapter.build_bundle(args, source_ckpt, device)
    bundle.e1.eval()
    bundle.d1.eval()
    bundle.tokenizer.train()
    bundle.combiner.train()
    imgs = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    layer1_out, z_norm, outputs = forward_adapter_multirate(bundle, imgs, levels)
    objective, _branch_losses = multirate.compute_multirate_loss(
        outputs,
        imgs,
        levels,
        recon_weights,
        margins,
        float(args.lambda_monotonic),
    )
    objective["loss"].backward()

    expected_q = (
        int(args.smoke_batch_size),
        int(args.fsq_d),
        int(args.latent_h),
        int(args.latent_w),
    )
    expected_img = (int(args.smoke_batch_size), 3, 256, 256)
    for level in levels:
        if tuple(outputs[level]["q3"].shape) != expected_q:
            raise RuntimeError(
                f"L={level} expected q3 {expected_q}, got {tuple(outputs[level]['q3'].shape)}"
            )
        if tuple(outputs[level]["final"].shape) != expected_img:
            raise RuntimeError(
                f"L={level} expected final {expected_img}, got {tuple(outputs[level]['final'].shape)}"
            )
    if tuple(z_norm.shape) != expected_q:
        raise RuntimeError(f"expected shared z_norm {expected_q}, got {tuple(z_norm.shape)}")
    finite_grad_parameters = sum(
        1
        for parameter in list(bundle.tokenizer.parameters()) + list(bundle.combiner.parameters())
        if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
    )
    if finite_grad_parameters <= 0:
        raise RuntimeError("backward smoke produced no finite trainable gradients")
    initial_deltas = [
        float((outputs[level]["final"] - layer1_out["x1"]).abs().max().detach())
        for level in levels
    ]
    print(
        f"[smoke adapter multirate backward] device=cpu x1={tuple(layer1_out['x1'].shape)} "
        f"z_norm={tuple(z_norm.shape)} levels={levels} q={expected_q} "
        f"loss={float(objective['loss'].detach()):.6g} "
        f"finite_grad_parameters={finite_grad_parameters} "
        f"initial_max_abs_final_minus_x1={initial_deltas}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nested-levels", type=str, default="5,9,17")
    parser.add_argument("--recon-weights", type=str, default="1")
    parser.add_argument("--lambda-monotonic", type=float, default=1.0)
    parser.add_argument("--monotonic-margins", type=str, default="1e-5")
    parser.add_argument("--selection-min-rate-gain-db", type=float, default=0.0)
    parser.add_argument("--selection-min-per-image-strict-ratio", type=float, default=0.0)
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Width320 PCA-adapter multirate options: --nested-levels 5,9,17; "
            "--recon-weights 1,1,1; --lambda-monotonic FLOAT; "
            "--monotonic-margins 1e-5,1e-5; --selection-min-rate-gain-db FLOAT; "
            "--selection-min-per-image-strict-ratio FLOAT. "
            "--smoke-shapes performs a CPU shape + one-batch backward check.\n",
            flush=True,
        )
    route_args, remaining = parser.parse_known_args()
    argv = sys.argv
    try:
        sys.argv = [argv[0], *remaining]
        args = adapter.parse_args()
    finally:
        sys.argv = argv
    for key, value in vars(route_args).items():
        setattr(args, key, value)

    original = argv[1:]
    if not direct.cli_option_present(original, "--epochs"):
        args.epochs = 100
    if not direct.cli_option_present(original, "--num-workers"):
        args.num_workers = 16
    if not direct.cli_option_present(original, "--val-num-workers"):
        args.val_num_workers = 4
    if not direct.cli_option_present(original, "--bn-calibration-batches"):
        args.bn_calibration_batches = 64
    if not direct.cli_option_present(original, "--fsq-normalizer"):
        args.fsq_normalizer = "batch"
    if not direct.cli_option_present(original, "--adapter-combiner"):
        args.adapter_combiner = "residual"
    if not direct.cli_option_present(original, "--selection-min-drop-zero"):
        args.selection_min_drop_zero = 0.1
    if not direct.cli_option_present(original, "--selection-min-drop-shuffle"):
        args.selection_min_drop_shuffle = 0.1
    return args


def validate_args(args: argparse.Namespace) -> tuple[list[int], list[float], list[float]]:
    if str(args.preset) != "custom":
        raise ValueError("adapter multirate comparisons require --preset custom")
    if str(args.arch) != "swin" or str(args.swin_codec) != "no_compressor":
        raise ValueError("adapter multirate route requires Swin no_compressor")
    if int(args.fsq_d) != 3:
        raise ValueError("adapter multirate route fixes the shared normalized latent at d=3")
    if bool(args.no_pre_norm):
        raise ValueError("adapter multirate route requires pre_norm followed by tanh")
    if str(args.adapter_init) != "pca":
        raise ValueError("adapter multirate route requires --adapter-init pca")
    if str(args.adapter_combiner) != "residual":
        raise ValueError("adapter multirate route requires --adapter-combiner residual")
    if str(args.combiner_mode) != "original" or str(args.condition_mode) != "none":
        raise ValueError("adapter multirate route requires original/no-condition source contract")
    if str(args.variant) != "combiner":
        raise ValueError(f"adapter multirate route requires combiner source, got {args.variant!r}")
    if bool(args.freeze_combiner):
        raise ValueError("adapter multirate route requires a trainable residual combiner")
    if float(args.lambda_usage) != 0.0 or float(args.lambda_u2_img) != 0.0:
        raise ValueError("adapter multirate route has no usage/KL or u2 teacher loss")
    if int(args.pca_init_batches) <= 0 or int(args.adapter_hidden) < 0:
        raise ValueError("invalid PCA adapter settings")
    if int(args.bn_calibration_batches) < 0:
        raise ValueError("--bn-calibration-batches must be non-negative")
    if not math.isfinite(float(args.lambda_monotonic)) or float(args.lambda_monotonic) < 0.0:
        raise ValueError("--lambda-monotonic must be finite and non-negative")
    if not 0.0 <= float(args.selection_min_per_image_strict_ratio) <= 1.0:
        raise ValueError("--selection-min-per-image-strict-ratio must be in [0,1]")

    levels = multirate.parse_nested_levels(args.nested_levels, int(args.fsq_d))
    weights = multirate.parse_float_list(args.recon_weights, len(levels), "--recon-weights")
    margins = multirate.parse_float_list(
        args.monotonic_margins, len(levels) - 1, "--monotonic-margins"
    )
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError(f"invalid reconstruction weights: {weights}")
    if any(margin < 0.0 for margin in margins):
        raise ValueError(f"invalid monotonic margins: {margins}")
    return levels, weights, margins


def main() -> None:
    args = parse_args()
    args.stage = "layer2_fsq_adapter_multirate"
    base.apply_preset(args)
    levels, _weights, _margins = validate_args(args)
    args.fsq_levels = multirate.repeated_levels(levels[-1], int(args.fsq_d))
    source_ckpt = base.load_teacher_checkpoint_for_args(args)
    # Check again because source checkpoint arguments can update the codec contract.
    validate_args(args)
    args._usage_weight = 0.0
    direct.explore.ExploreIFSQQuantizer.config = args
    direct.explore.validate_explore_args(args)
    base.check_jsccf_args(args)

    if not direct.cli_option_present(sys.argv[1:], "--save-dir"):
        args.save_dir = str(THIS_DIR / "checkpoints-adapter-multirate")
    if not direct.cli_option_present(sys.argv[1:], "--version"):
        args.version = "layer2-fsq-adapter-multirate"
    name = route_name(args, levels)
    if not direct.cli_option_present(sys.argv[1:], "--log-file"):
        args.log_file = str(
            THIS_DIR
            / "logs-adapter-multirate"
            / f"{name}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}.log"
        )
    Path(base.resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    base.setup_log_file(args.log_file)
    base.write_json(
        Path(base.resolve_path(args.save_dir))
        / f"{name}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}_args.json",
        {key: value for key, value in vars(args).items() if not key.startswith("_")},
    )

    if bool(args.smoke_shapes):
        smoke_shapes_and_backward(args, source_ckpt)
        return
    train(args, source_ckpt)


if __name__ == "__main__":
    main()
