#!/usr/bin/env python3
"""Training-only receiver-information restoration ceiling diagnostic.

This file is deliberately *not* a new receiver deployment route.  It asks a
different, narrower diagnostic question: with only the information already
available after Layer1, how much image residual can a sufficiently flexible
regressor recover?

The graph is strictly::

    img --frozen E1/D1--> ReceiverCondition(z1, x1)
    Teacher(ReceiverCondition(z1, x1)) --> residual_hat
    x_teacher = clamp(x1 + residual_hat)

``img`` is used *after* ``Teacher.forward`` only as the reconstruction loss
target.  The teacher's public forward signature takes exactly one
``ReceiverCondition`` and therefore cannot receive ``img``, ``z2``, ``q2`` or
an oracle index.  No Layer2 sender module is executed by this script.

The resulting PSNR is an empirical, finite-model restoration diagnostic --
not a mathematical information-theoretic bound, and not a deployable
``q2_hat -> D2 -> combiner`` result.  Its checkpoints and JSON/Markdown
reports are explicitly marked ``deployment_prohibited``.  They may be used
only as *training-time teachers*, for example to choose a representation or
add a distillation target inside a separately audited q2/index receiver
training run.  A production/deployment claim must still be evaluated through
the required q2/index generation and receiver D2/combiner graph.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_layer2_vq_nested as nested  # noqa: E402
from contracts import (  # noqa: E402
    ReceiverCondition,
    assert_div2k_crop_protocol,
    assert_receiver_only_module,
    assert_training_targets_are_not_inputs,
    make_receiver_condition,
)
from receiver_models import ReceiverTrunk  # noqa: E402


base = nested.base
ROOT = nested.CDDM_ROOT


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


class ReceiverInformationResidualTeacher(nn.Module):
    """A high-capacity residual regressor with a receiver-only interface.

    The zero-initialized RGB head makes the initial output exactly ``x1``.
    Four PixelShuffle stages match the normal JSCC-f 16x16 -> 256x256 Layer1
    geometry.  The final interpolation only makes synthetic smoke tensors
    robust; normal DIV2K inputs already have the exact 16x scale ratio.
    """

    def __init__(
        self,
        z1_channels: int,
        *,
        hidden: int,
        blocks: int,
        attention_every: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.trunk = ReceiverTrunk(
            int(z1_channels),
            hidden=int(hidden),
            blocks=int(blocks),
            attention_every=int(attention_every),
            heads=int(heads),
            condition_mode="z1_x1",
        )
        stages: list[nn.Module] = []
        for _stage in range(4):
            stages.extend(
                [
                    nn.Conv2d(int(hidden), int(hidden) * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.SiLU(),
                ]
            )
        self.upsample = nn.Sequential(*stages)
        self.residual_head = nn.Conv2d(int(hidden), 3, 3, padding=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, condition: ReceiverCondition) -> torch.Tensor:
        """Return ``x1 + residual_hat`` using only ``condition.z1/x1``."""

        condition.validate()
        residual = self.residual_head(self.upsample(self.trunk(condition)))
        if tuple(residual.shape[-2:]) != tuple(condition.x1.shape[-2:]):
            residual = F.interpolate(
                residual,
                size=tuple(condition.x1.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
        return (condition.x1 + residual).clamp(0.0, 1.0)


def _assert_frozen_layer1(source: nested.SourceLayer2) -> None:
    frozen = list(source.e1.parameters()) + list(source.d1.parameters())
    if any(parameter.requires_grad for parameter in frozen):
        raise AssertionError("training-only teacher must keep E1/D1 frozen")


def _assert_layer2_discarded(source: nested.SourceLayer2) -> None:
    if not all(isinstance(getattr(source, name), nn.Identity) for name in ("e2", "d2", "combiner")):
        raise AssertionError("training-only teacher must not retain executable Layer2 sender modules")


def load_layer1_and_loaders(
    args: argparse.Namespace,
) -> tuple[nested.SourceLayer2, Any, Any, torch.device]:
    """Reuse the shared DIV2K loader/checkpoint path, then discard Layer2.

    ``nested.load_source`` is used for the exact existing checkpoint contract.
    Immediately replacing its unused Layer2 modules makes it impossible for
    this diagnostic to accidentally execute the sender Layer2 path.
    """

    source_path = args.source_checkpoint or nested.DEFAULT_SOURCES[str(args.arch)]
    checkpoint = base.jsccf_io.load_checkpoint(str(resolve_path(source_path)))
    source_args = argparse.Namespace(**checkpoint["args"])
    train_loader, val_loader, device = nested.build_loaders(args, source_args)
    # Keep this explicit even though nested.build_loaders already enforces it:
    # this diagnostic owns its RandomCrop-train / CenterCrop-val contract.
    assert_div2k_crop_protocol(train_loader, val_loader)
    args.source_checkpoint = str(source_path)
    args.layer2_arch = "match"
    args.layer2_source_checkpoint = ""
    source = nested.load_source(args, device)
    _assert_frozen_layer1(source)
    # No Layer2 forward is available after this point.  E1/D1 are the only
    # source modules retained to turn each training image into (z1,x1).
    source.e2 = nn.Identity().to(device)
    source.d2 = nn.Identity().to(device)
    source.combiner = nn.Identity().to(device)
    source.layer2_checkpoint = {}
    _assert_layer2_discarded(source)
    return source, train_loader, val_loader, device


def build_teacher(source: nested.SourceLayer2, args: argparse.Namespace, device: torch.device) -> nn.Module:
    teacher = ReceiverInformationResidualTeacher(
        int(source.args.latent_ch),
        hidden=int(args.hidden),
        blocks=int(args.blocks),
        attention_every=int(args.attention_every),
        heads=int(args.heads),
    ).to(device)
    assert_receiver_only_module(teacher)
    return teacher


def run_epoch(
    loader,
    *,
    source: nested.SourceLayer2,
    teacher: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    max_batches: int,
    train: bool,
) -> dict[str, float]:
    """Run one teacher epoch; the image exists only outside Teacher.forward."""

    source.e1.eval()
    source.d1.eval()
    _assert_layer2_discarded(source)
    teacher.train(bool(train))
    totals: dict[str, float] = {}
    image_count = 0
    batches = 0
    for batch_index, (images, _labels) in enumerate(loader, start=1):
        if int(max_batches) > 0 and batch_index > int(max_batches):
            break
        images = images.to(device, non_blocking=True)
        # Layer1 simulation belongs outside the teacher inference graph.  The
        # resulting typed condition is all the teacher can see.
        with torch.no_grad():
            layer1 = source.layer1(images)
            condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)

        if train:
            if optimizer is None:
                raise AssertionError("training requires an optimizer")
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(bool(train)):
            prediction = teacher(condition)
            # `images` is intentionally introduced only after teacher(condition)
            # has returned.  It is a supervised image target, never a forward
            # input and never a residual/model attribute.
            loss = F.mse_loss(prediction.float(), images.float())
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
                optimizer.step()

        if batch_index == 1:
            assert_training_targets_are_not_inputs(
                teacher,
                condition,
                source_targets={"img": images},
            )
        batch_size = int(images.shape[0])
        values = {
            "loss": float(loss.detach()),
            "psnr_x1": float(psnr_per_image(condition.x1, images).mean()),
            "psnr_teacher": float(psnr_per_image(prediction, images).mean()),
            "receiver_only_audit": 1.0,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        image_count += batch_size
        batches += 1
    if image_count < 1:
        raise RuntimeError("teacher epoch processed no images")
    metrics = {name: value / image_count for name, value in totals.items()}
    metrics["delta_x1"] = metrics["psnr_teacher"] - metrics["psnr_x1"]
    metrics["evaluated_images"] = float(image_count)
    metrics["evaluated_batches"] = float(batches)
    metrics["full_validation"] = float((not train) and image_count == len(loader.dataset))
    return metrics


def _report_payload(
    *,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "stage": "explore2_training_only_receiver_information_teacher_ceiling",
        "epoch": int(epoch),
        "deployment_prohibited": True,
        "interpretation": (
            "Finite-model receiver-information restoration diagnostic only; not a "
            "mathematical ceiling and not a q2/index receiver deployment result."
        ),
        "teacher_contract": {
            "teacher_inputs": ["z1", "x1"],
            "teacher_output": "clamp(x1 + residual_hat)",
            "supervision_target_only": "img",
            "forbidden_teacher_inputs": ["img", "z2", "q2", "oracle_indices"],
            "layer2_execution": "prohibited; Layer2 modules replaced by Identity",
            "train_transform": "RandomCrop(256)+RandomHorizontalFlip+ToTensor",
            "validation_transform": "CenterCrop(256)+ToTensor",
            "approved_use": "training-time teacher/distillation target only",
        },
        "args": vars(args),
        "metrics": metrics,
    }


def write_report(
    *,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    output_dir = resolve_path(args.results_dir) / str(args.version)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _report_payload(args=args, epoch=epoch, metrics=metrics)
    json_path = output_dir / "teacher_ceiling_latest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = output_dir / "teacher_ceiling_latest.md"
    md_path.write_text(
        "# Training-only receiver-information restoration teacher\n\n"
        "**Not a deployment result.** This finite model estimates how much residual can be "
        "regressed from receiver-visible `(z1, x1)`. It does not generate `q2_hat`, does not "
        "execute receiver `D2/combiner`, and must not be cited as a receiver deployment gain.\n\n"
        f"- Epoch: {int(epoch)}\n"
        f"- Full DIV2K validation: `{int(metrics['full_validation'])}` "
        f"({int(metrics['evaluated_images'])} images)\n"
        f"- PSNR(x1): `{metrics['psnr_x1']:.6f} dB`\n"
        f"- PSNR(teacher): `{metrics['psnr_teacher']:.6f} dB`\n"
        f"- Delta vs x1: `{metrics['delta_x1']:+.6f} dB`\n"
        f"- No-leak audit: `{int(metrics['receiver_only_audit'])}`\n"
        "- Crop contract: train `RandomCrop(256)+RandomHorizontalFlip`; validation `CenterCrop(256)`\n\n"
        "Permitted use: use the residual/image prediction only as a training-time teacher or "
        "auxiliary/distillation target for a separately audited q2/index generator. The latter "
        "must still demonstrate `q2_hat -> D2 -> combiner(x1,u2_hat)` on full validation.\n",
        encoding="utf-8",
    )
    print(f"saved training-only teacher report: {json_path}", flush=True)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    teacher: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    metrics: dict[str, float],
    best_delta_x1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _report_payload(args=args, epoch=epoch, metrics=metrics)
    payload.update(
        {
            "teacher_state_dict": teacher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_delta_x1": float(best_delta_x1),
        }
    )
    torch.save(payload, path)
    print(f"saved training-only teacher checkpoint: {path}", flush=True)


def train(args: argparse.Namespace) -> None:
    nested.seed_everything(int(args.seed))
    source, train_loader, val_loader, device = load_layer1_and_loaders(args)
    teacher = build_teacher(source, args, device)
    optimizer = optim.AdamW(
        teacher.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    output_dir = resolve_path(args.save_dir) / str(args.version)

    print("=== explore-2 | TRAINING-ONLY receiver-information teacher ceiling ===", flush=True)
    print("实验设计", flush=True)
    print(
        "  frozen Layer1 img->E1->z1->D1->x1 prepares ReceiverCondition; "
        "Teacher(z1,x1)->x1+residual_hat. Teacher forward never accepts img/z2/q2/index.",
        flush=True,
    )
    print("loss设计", flush=True)
    print("  MSE(x1+residual_hat, img); img is supervised target only after Teacher.forward.", flush=True)
    print("模块选择", flush=True)
    print(
        f"  ReceiverTrunk+4x PixelShuffle residual teacher hidden={args.hidden} blocks={args.blocks}; "
        "Layer2 sender modules discarded; checkpoint/report deployment_prohibited=1.",
        flush=True,
    )
    print(
        f"  crop=train RandomCrop(256)+RandomHorizontalFlip; val CenterCrop(256); "
        f"full-val default=100 images; source={args.source_checkpoint}",
        flush=True,
    )

    best = float("-inf")
    for epoch in range(1, int(args.epochs) + 1):
        started = time.time()
        train_metrics = run_epoch(
            train_loader,
            source=source,
            teacher=teacher,
            optimizer=optimizer,
            device=device,
            max_batches=int(args.max_train_batches),
            train=True,
        )
        print(
            f"[teacher-ceiling train {epoch:03d}/{args.epochs:03d}] {train_metrics} "
            f"time={time.time()-started:.1f}s",
            flush=True,
        )
        if epoch == 1 or epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            val_metrics = run_epoch(
                val_loader,
                source=source,
                teacher=teacher,
                optimizer=None,
                device=device,
                max_batches=int(args.max_val_batches),
                train=False,
            )
            print(f"[teacher-ceiling val {epoch:03d}] {val_metrics}", flush=True)
            write_report(args=args, epoch=epoch, metrics=val_metrics)
            is_best = val_metrics["delta_x1"] > best
            if is_best:
                best = val_metrics["delta_x1"]
            save_checkpoint(
                output_dir / "teacher_ceiling_latest.pth",
                epoch=epoch,
                teacher=teacher,
                optimizer=optimizer,
                args=args,
                metrics=val_metrics,
                best_delta_x1=best,
            )
            if is_best:
                save_checkpoint(
                    output_dir / "teacher_ceiling_best.pth",
                    epoch=epoch,
                    teacher=teacher,
                    optimizer=optimizer,
                    args=args,
                    metrics=val_metrics,
                    best_delta_x1=best,
                )


def run_smoke() -> None:
    """CPU-only structural smoke: no real image/checkpoint and no long job."""

    torch.manual_seed(20260713)
    device = torch.device("cpu")
    teacher = ReceiverInformationResidualTeacher(
        16, hidden=16, blocks=1, attention_every=1, heads=4
    ).to(device)
    assert_receiver_only_module(teacher)
    condition = make_receiver_condition(
        torch.randn(2, 16, 1, 1, device=device),
        torch.rand(2, 3, 16, 16, device=device),
        detach=True,
    )
    target = torch.rand(2, 3, 16, 16, device=device)
    prediction = teacher(condition)
    assert tuple(prediction.shape) == tuple(target.shape)
    assert torch.isfinite(prediction).all()
    assert_training_targets_are_not_inputs(teacher, condition, source_targets={"img": target})
    # Exercise the explicit crop contract without touching a real dataset.
    train_loader = SimpleNamespace(dataset=SimpleNamespace(transform="RandomCrop(256)+ToTensor"))
    val_loader = SimpleNamespace(dataset=SimpleNamespace(transform="CenterCrop(256)+ToTensor"))
    assert_div2k_crop_protocol(train_loader, val_loader)
    print(
        "[PASS] training-only teacher CPU smoke "
        f"shape={tuple(prediction.shape)} delta_init="
        f"{float(psnr_per_image(prediction, target).mean() - psnr_per_image(condition.x1, target).mean()):+.6f}dB "
        "no_leak=1 crop_contract=1 deployment_prohibited=1",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--arch", choices=["cnn", "swin"], default="cnn")
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--attention-every", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--save-dir", default="MY-V2/jscc-f/explore-2/checkpoints-receiver-ceiling-teacher"
    )
    parser.add_argument(
        "--results-dir", default="MY-V2/jscc-f/explore-2/results-receiver-ceiling-teacher"
    )
    parser.add_argument("--version", default="cnn-receiver-information-teacher-v1")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small CPU structural/no-leak/crop smoke without loading data or a checkpoint.",
    )
    args = parser.parse_args()
    if int(args.hidden) < 8 or int(args.hidden) % int(args.heads) != 0:
        raise ValueError("--hidden must be >=8 and divisible by --heads")
    if int(args.blocks) < 1 or int(args.epochs) < 1:
        raise ValueError("--blocks and --epochs must be positive")
    if int(args.val_every) < 1:
        raise ValueError("--val-every must be positive")
    return args


def main() -> None:
    args = parse_args()
    if bool(args.smoke):
        run_smoke()
    else:
        train(args)


if __name__ == "__main__":
    main()
