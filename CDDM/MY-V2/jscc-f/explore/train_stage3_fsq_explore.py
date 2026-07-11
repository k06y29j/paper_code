#!/usr/bin/env python3
"""FSQ experiments isolated from the canonical Stage-3 training entrypoint.

The established ``train-stage3-fsq.py`` stays the reference implementation.
This wrapper adds two opt-in mechanisms needed for the large-vocabulary
experiments:

* ``--fsq-normalizer batch`` uses channel-wise BatchNorm before tanh+FSQ,
  rather than per-image GroupNorm over the whole 3x16x16 map.
* ``--lambda-usage`` adds a differentiable KL-to-uniform loss over soft FSQ
  assignments.  ``joint`` acts on the full Cartesian product vocabulary and
  is therefore appropriate for a 5,120-word ``20x16x16`` tokenizer.

All exploration outputs default under ``MY-V2/jscc-f/explore``.  With the
default zero weight and ``group`` normalizer, the quantizer math remains the
same as the canonical script.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
CDDM_ROOT = JSCCF_DIR.parents[1]
for path in (CDDM_ROOT, JSCCF_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_base() -> ModuleType:
    spec = importlib.util.spec_from_file_location("jsccf_stage3_fsq_explore_base", JSCCF_DIR / "train-stage3-fsq.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load canonical train-stage3-fsq.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


def _format_float(value: float) -> str:
    return f"{float(value):.0e}".replace("-", "m").replace("+", "p")


def _cli_option_present(tokens: list[str], option: str) -> bool:
    return option in tokens or any(token.startswith(f"{option}=") for token in tokens)


class ExploreIFSQQuantizer(base.IFSQQuantizer):
    """Canonical IFSQ plus normalizer and differentiable usage objectives."""

    config: argparse.Namespace | None = None

    def __init__(self, levels, channels: int, use_pre_norm: bool = True) -> None:
        super().__init__(levels, channels=channels, use_pre_norm=use_pre_norm)
        cfg = self.config
        if cfg is None:
            raise RuntimeError("ExploreIFSQQuantizer requires an active experiment config")
        normalizer = str(cfg.fsq_normalizer)
        if bool(use_pre_norm) and normalizer == "batch":
            # Unlike GroupNorm(1, D), this preserves each map's spatial
            # contrast and uses accumulated validation statistics.
            self.pre_norm = nn.BatchNorm2d(int(channels), affine=True, track_running_stats=True)
        elif bool(use_pre_norm) and normalizer == "batch_stateless":
            # Batch statistics in both train and eval prevent running-stat
            # drift from turning a healthy train token distribution into a
            # collapsed validation distribution.
            self.pre_norm = nn.BatchNorm2d(int(channels), affine=True, track_running_stats=False)
        elif bool(use_pre_norm) and normalizer == "instance":
            self.pre_norm = nn.InstanceNorm2d(int(channels), affine=True, track_running_stats=False)
        elif normalizer != "group":
            raise ValueError(f"unsupported fsq normalizer {normalizer!r}")

    def _soft_assignment(self, z_norm: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return per-dimension probabilities, KL-to-uniform, entropy bits."""
        tau = max(float(self.config.usage_tau), 1e-4)
        probs: list[torch.Tensor] = []
        loss = z_norm.new_zeros(())
        entropy_bits = z_norm.new_zeros(())
        for channel, level in enumerate(self.levels.detach().cpu().tolist()):
            level = int(level)
            centers = torch.linspace(-1.0, 1.0, level, device=z_norm.device, dtype=z_norm.dtype)
            values = z_norm[:, channel].reshape(-1, 1)
            assignment = F.softmax(-((values - centers) / tau).square(), dim=1)
            marginal = assignment.mean(dim=0).clamp_min(1e-12)
            probs.append(assignment)
            loss = loss + math.log(float(level)) + (marginal * marginal.log()).sum()
            entropy_bits = entropy_bits - (marginal * marginal.log2()).sum()
        return probs, loss / float(len(probs)), entropy_bits / float(len(probs))

    def _joint_usage_kl(self, assignments: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # N=Bx16x16, K<=5,120 in the planned experiments: the temporary
        # N x K distribution is ~60 MB at float32 and gives direct gradients
        # to the actual transmitted joint codeword rather than only marginals.
        dims = len(assignments)
        joint = assignments[0].reshape(assignments[0].shape[0], assignments[0].shape[1], *([1] * (dims - 1)))
        for dim, assignment in enumerate(assignments[1:], start=1):
            shape = [assignment.shape[0]] + [1] * dims
            shape[dim + 1] = assignment.shape[1]
            joint = joint * assignment.reshape(shape)
        marginal = joint.flatten(1).mean(dim=0).clamp_min(1e-12)
        vocab = int(marginal.numel())
        kl = math.log(float(vocab)) + (marginal * marginal.log()).sum()
        entropy_bits = -(marginal * marginal.log2()).sum()
        return kl, entropy_bits

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out = super().forward(z)
        # The BatchNorm-only SWIN ablation should be computationally identical
        # to canonical FSQ apart from its normalizer.  Avoid materializing an
        # N x vocab soft joint distribution when it cannot affect gradients.
        if float(getattr(self.config, "_usage_weight", self.config.lambda_usage)) == 0.0:
            zero = out["z3_norm"].new_zeros(())
            out["usage_kl"] = zero
            out["soft_level_entropy_bits"] = zero
            out["soft_usage_entropy_bits"] = zero
            return out
        assignments, level_kl, level_entropy_bits = self._soft_assignment(out["z3_norm"])
        if str(self.config.usage_mode) == "joint":
            usage_kl, usage_entropy_bits = self._joint_usage_kl(assignments)
        else:
            usage_kl, usage_entropy_bits = level_kl, level_entropy_bits
        if str(self.config.usage_objective) == "entropy_floor":
            deficit = (float(self.config.usage_target_bits) - usage_entropy_bits).clamp_min(0.0)
            usage_kl = deficit.square()
        out["usage_kl"] = usage_kl
        out["soft_level_entropy_bits"] = level_entropy_bits
        out["soft_usage_entropy_bits"] = usage_entropy_bits
        return out


def usage_weight(args: argparse.Namespace, epoch: int) -> float:
    """Warm up reconstruction first, then linearly enable usage pressure."""
    target = float(args.lambda_usage)
    if target == 0.0 or int(epoch) <= int(args.usage_warmup_epochs):
        return 0.0
    ramp = int(args.usage_ramp_epochs)
    if ramp <= 0:
        return target
    progress = min(1.0, float(int(epoch) - int(args.usage_warmup_epochs)) / float(ramp))
    return target * progress


def selection_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    """Keep a high-PSNR checkpoint only when q3 is decoder-relevant."""
    if float(metrics.get("drop_zero", float("-inf"))) < float(args.selection_min_drop_zero):
        return float("-inf")
    if float(metrics.get("drop_shuffle", float("-inf"))) < float(args.selection_min_drop_shuffle):
        return float("-inf")
    return float(metrics["psnr_final"])


def install_exploration_hooks() -> None:
    original_compute_losses = base.compute_losses
    original_update_metrics = base.update_metrics
    original_stage3_name = base.stage3_name
    original_header = base.print_tokenizer_header

    def compute_losses(out, teacher_out, imgs, args):
        losses = original_compute_losses(out, teacher_out, imgs, args)
        loss_usage = out["usage_kl"]
        losses["loss_usage"] = loss_usage
        losses["loss"] = losses["loss"] + float(getattr(args, "_usage_weight", args.lambda_usage)) * loss_usage
        return losses

    def update_metrics(m, out, teacher_out, imgs, losses):
        original_update_metrics(m, out, teacher_out, imgs, losses)
        bsz = int(imgs.shape[0])
        m["soft_level_entropy_bits"].update(float(out["soft_level_entropy_bits"].detach().item()), bsz)
        m["soft_usage_entropy_bits"].update(float(out["soft_usage_entropy_bits"].detach().item()), bsz)

    def stage3_name(args):
        name = original_stage3_name(args)
        usage = (
            f"{args.usage_mode}_{args.usage_objective}_u{_format_float(args.lambda_usage)}"
            f"_w{int(args.usage_warmup_epochs)}r{int(args.usage_ramp_epochs)}"
        )
        if str(args.usage_objective) == "entropy_floor":
            usage += f"_h{float(args.usage_target_bits):g}".replace(".", "p")
        return f"{name}_{args.fsq_normalizer}_{usage}"

    def print_header(args, tokenizer, teacher, train_n, val_n):
        original_header(args, tokenizer, teacher, train_n, val_n)
        print(
            "  explore_fsq="
            f"normalizer={args.fsq_normalizer} usage_mode={args.usage_mode} "
            f"usage_objective={args.usage_objective} lambda_usage={float(args.lambda_usage):g} "
            f"usage_tau={float(args.usage_tau):g} warmup={int(args.usage_warmup_epochs)} "
            f"ramp={int(args.usage_ramp_epochs)} target_bits={float(args.usage_target_bits):g}",
            flush=True,
        )
        print(
            "  selection="
            f"psnr_final subject_to drop_zero>={float(args.selection_min_drop_zero):g} "
            f"drop_shuffle>={float(args.selection_min_drop_shuffle):g}",
            flush=True,
        )

    base.IFSQQuantizer = ExploreIFSQQuantizer
    base.compute_losses = compute_losses
    base.update_metrics = update_metrics
    base.stage3_name = stage3_name
    base.print_tokenizer_header = print_header
    for name in ("loss_usage", "soft_level_entropy_bits", "soft_usage_entropy_bits"):
        if name not in base.METRIC_NAMES:
            base.METRIC_NAMES.append(name)
    for name in ("loss_usage", "soft_usage_entropy_bits"):
        if name not in base.DISPLAY_METRICS:
            base.DISPLAY_METRICS.append(name)


def train(args: argparse.Namespace, teacher_ckpt: dict) -> None:
    """Canonical loop with scheduled usage pressure and semantic best gating."""
    base.seed_everything(int(args.seed))
    cfg = base.jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(cfg)
    teacher = base.build_teacher(args, teacher_ckpt, cfg.device)
    tokenizer = base.Layer3FSQTokenizer(args, cfg.device)
    opt = optim.AdamW(tokenizer.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    base.print_tokenizer_header(args, tokenizer, teacher, len(train_loader.dataset), len(val_loader.dataset))

    if bool(args.eval_init_only):
        args._usage_weight = usage_weight(args, 1)
        val_metrics = base.validate(val_loader, tokenizer, teacher, args, cfg.device)
        score = selection_score(val_metrics, args)
        print(
            f"[stage3-fsq init val] {base.display_metrics(val_metrics)} "
            f"usage_weight={float(args._usage_weight):g} score={score:.6f}",
            flush=True,
        )
        out = Path(base.resolve_path(args.save_dir)) / (
            f"{base.stage3_name(args)}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}_init_eval.json"
        )
        base.write_json(
            out,
            {
                "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
                "metrics": val_metrics,
                "selection_score": score,
            },
        )
        print(f"[stage3-fsq init val] wrote {out}", flush=True)
        return

    best = float("-inf")
    metrics: dict[str, float] = {}
    for epoch in range(1, int(args.epochs) + 1):
        args._usage_weight = usage_weight(args, epoch)
        tokenizer.train()
        teacher.eval()
        m = base.meters(base.METRIC_NAMES)
        hist = torch.zeros(base.vocab_size(base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))), dtype=torch.float32)
        t0 = time.time()
        for batch_idx, (imgs, _labels) in enumerate(train_loader, start=1):
            if int(args.max_train_batches) > 0 and batch_idx > int(args.max_train_batches):
                break
            imgs = imgs.to(cfg.device, non_blocking=True)
            with torch.no_grad():
                teacher_out = teacher.forward(imgs)
            out = tokenizer(imgs, teacher_out["x1"], teacher_out["z1"], teacher.combiner)
            losses = base.compute_losses(out, teacher_out, imgs, args)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), float(args.grad_clip_norm))
            opt.step()
            base.update_metrics(m, out, teacher_out, imgs, losses)
            base.update_code_hist(hist, out["idx3"])

        metrics = base.finalize_metrics(m, hist, args)
        print(
            f"[stage3-fsq train {epoch:03d}/{int(args.epochs):03d}] "
            f"{base.display_metrics(metrics)} usage_weight={float(args._usage_weight):g} "
            f"time={time.time() - t0:.1f}s",
            flush=True,
        )
        if base.should_validate(args, epoch):
            val_metrics = base.validate(val_loader, tokenizer, teacher, args, cfg.device)
            score = selection_score(val_metrics, args)
            print(
                f"[stage3-fsq val {epoch:03d}] {base.display_metrics(val_metrics)} "
                f"usage_weight={float(args._usage_weight):g} score={score:.6f}",
                flush=True,
            )
            if score > best:
                best = score
                val_metrics["selection_score"] = score
                base.save_tokenizer_checkpoint(
                    base.jsccf_io.ckpt_path(args, base.stage3_name(args), "best"),
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    tokenizer=tokenizer,
                )
        if base.should_save_latest(args, epoch):
            metrics["usage_weight"] = float(args._usage_weight)
            base.save_tokenizer_checkpoint(
                base.jsccf_io.ckpt_path(args, base.stage3_name(args), "latest"),
                epoch=epoch,
                args=args,
                metrics=metrics,
                tokenizer=tokenizer,
            )

    metrics["usage_weight"] = float(args._usage_weight)
    base.save_tokenizer_checkpoint(
        base.jsccf_io.ckpt_path(args, base.stage3_name(args), "latest"),
        epoch=int(args.epochs),
        args=args,
        metrics=metrics,
        tokenizer=tokenizer,
    )


def parse_args() -> argparse.Namespace:
    # Parse only exploration flags first, then delegate all canonical arguments
    # unchanged to the original parser.  This prevents a second copy of the
    # Stage-3 CLI from drifting away from the reference entrypoint.
    explore_parser = argparse.ArgumentParser(add_help=False)
    explore_parser.add_argument(
        "--fsq-normalizer",
        choices=["group", "batch", "batch_stateless", "instance"],
        default="group",
    )
    explore_parser.add_argument("--lambda-usage", type=float, default=0.0)
    explore_parser.add_argument("--usage-mode", choices=["joint", "level"], default="joint")
    explore_parser.add_argument("--usage-objective", choices=["uniform_kl", "entropy_floor"], default="uniform_kl")
    explore_parser.add_argument(
        "--usage-target-bits",
        type=float,
        default=0.0,
        help="Entropy floor in bits.  For --usage-mode joint, this is the full token entropy.",
    )
    explore_parser.add_argument("--usage-warmup-epochs", type=int, default=0)
    explore_parser.add_argument("--usage-ramp-epochs", type=int, default=0)
    explore_parser.add_argument("--usage-tau", type=float, default=0.12)
    explore_parser.add_argument("--selection-min-drop-zero", type=float, default=-1e9)
    explore_parser.add_argument("--selection-min-drop-shuffle", type=float, default=-1e9)
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Exploration options: --fsq-normalizer {group,batch,batch_stateless,instance}; "
            "--lambda-usage FLOAT; --usage-objective {uniform_kl,entropy_floor}; "
            "--usage-target-bits FLOAT; --usage-warmup-epochs INT; --usage-ramp-epochs INT; "
            "--selection-min-drop-zero FLOAT; --selection-min-drop-shuffle FLOAT.\n",
            flush=True,
        )
    explore_args, remaining = explore_parser.parse_known_args()
    if float(explore_args.lambda_usage) < 0.0:
        raise ValueError("--lambda-usage must be non-negative")
    if float(explore_args.usage_tau) <= 0.0:
        raise ValueError("--usage-tau must be positive")
    if int(explore_args.usage_warmup_epochs) < 0 or int(explore_args.usage_ramp_epochs) < 0:
        raise ValueError("usage warmup/ramp epochs must be non-negative")

    argv = sys.argv
    try:
        sys.argv = [argv[0], *remaining]
        args = base.parse_args()
    finally:
        sys.argv = argv
    args.fsq_normalizer = str(explore_args.fsq_normalizer)
    args.lambda_usage = float(explore_args.lambda_usage)
    args.usage_mode = str(explore_args.usage_mode)
    args.usage_objective = str(explore_args.usage_objective)
    args.usage_target_bits = float(explore_args.usage_target_bits)
    args.usage_warmup_epochs = int(explore_args.usage_warmup_epochs)
    args.usage_ramp_epochs = int(explore_args.usage_ramp_epochs)
    args.usage_tau = float(explore_args.usage_tau)
    args.selection_min_drop_zero = float(explore_args.selection_min_drop_zero)
    args.selection_min_drop_shuffle = float(explore_args.selection_min_drop_shuffle)

    if not _cli_option_present(remaining, "--save-dir"):
        args.save_dir = str(THIS_DIR / "checkpoints")
    if not _cli_option_present(remaining, "--log-file"):
        args.log_file = ""
    if not _cli_option_present(remaining, "--version"):
        args.version = "fsq-explore"
    return args


def validate_explore_args(args: argparse.Namespace) -> None:
    if str(args.usage_objective) != "entropy_floor" or float(args.lambda_usage) == 0.0:
        return
    if float(args.usage_target_bits) <= 0.0:
        raise ValueError("--usage-target-bits must be positive for --usage-objective entropy_floor")
    levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    if str(args.usage_mode) == "joint":
        max_bits = math.log2(float(base.vocab_size(levels)))
    else:
        max_bits = sum(math.log2(float(level)) for level in levels) / float(len(levels))
    if float(args.usage_target_bits) > max_bits + 1e-6:
        raise ValueError(
            f"usage target {float(args.usage_target_bits):g} bits exceeds the {args.usage_mode} maximum {max_bits:g}"
        )


def main() -> None:
    install_exploration_hooks()
    args = parse_args()
    args.stage = "stage3_fsq_tokenizer_explore"
    ExploreIFSQQuantizer.config = args
    base.apply_preset(args)
    base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    validate_explore_args(args)
    Path(base.resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(
            THIS_DIR
            / "logs"
            / f"{base.stage3_name(args)}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}.log"
        )
    base.setup_log_file(args.log_file)
    teacher_ckpt = base.load_teacher_checkpoint_for_args(args)
    base.check_jsccf_args(args)
    base.write_json(
        Path(base.resolve_path(args.save_dir)) / f"{base.stage3_name(args)}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}_args.json",
        {key: value for key, value in vars(args).items() if not key.startswith("_")},
    )
    if bool(args.smoke_shapes):
        base.smoke_shapes(args, teacher_ckpt)
        return
    train(args, teacher_ckpt)


if __name__ == "__main__":
    main()
