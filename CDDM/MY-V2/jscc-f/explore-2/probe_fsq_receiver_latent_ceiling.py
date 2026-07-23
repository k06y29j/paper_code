#!/usr/bin/env python3
"""Probe the latent-side ceiling of an explore-2 FSQ receiver checkpoint.

The five evaluated paths share the same frozen Layer1 output ``(z1, x1)``:

``x1``
    Layer1 reconstruction baseline.
``sender_oracle``
    Sender E2/FSQ -> sender D2 -> sender combiner.
``receiver_qhat``
    Receiver-only predictor(z1, x1) -> receiver D2/combiner.
``receiver_true_q``
    Exact sender hard-FSQ q is decoded by the receiver D2/combiner.  This is
    an empirical latent-side ceiling for the fixed receiver decoder, not a
    mathematical upper bound and not a deployable receiver input.
``receiver_mid_q``
    The continuous diagnostic q_mid = 0.5 * (qhat + q_true), decoded by the
    receiver D2/combiner.  It is not claimed to be a valid hard FSQ token.

All evaluation runs under ``torch.inference_mode``.  Sender-only ``img`` and
``q_true`` are never passed into the receiver predictor; they are used only
for the oracle/ceiling diagnostics after qhat has been predicted from z1/x1.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_fsq_receiver as receiver  # noqa: E402


DEFAULT_CHECKPOINT = (
    THIS_DIR
    / "checkpoints-receiver"
    / "cnn-fsq-k4913-independent-d2-v5"
    / "fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_"
    "d2ft-oracle_independent-d2_best.pth"
)
DEFAULT_OUTPUT_DIR = THIS_DIR / "results-receiver-ceiling"
PATH_NAMES = (
    "x1",
    "sender_oracle",
    "receiver_qhat",
    "receiver_true_q",
    "receiver_mid_q",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def select_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device {value!r} requests CUDA, but CUDA is unavailable")
    return device


def load_payload(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        # Keep the unused optimizer snapshot off the GPU.  load_state_dict
        # copies only the model tensors into the selected evaluation device.
        "map_location": "cpu",
        "weights_only": False,
    }
    try:
        payload = torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError, ValueError) as error:
        if "mmap" not in str(error).lower() and not isinstance(error, TypeError):
            raise
        payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict) or str(payload.get("stage", "")) != "explore2_fsq_receiver":
        raise ValueError(f"not an explore-2 FSQ receiver checkpoint: {path}")
    for key in ("args", "predictor_state_dict", "receiver_d2_state_dict", "receiver_combiner_state_dict"):
        if key not in payload:
            raise KeyError(f"receiver checkpoint is missing {key!r}: {path}")
    return payload


def check_saved_contract(payload: dict[str, Any]) -> None:
    contract = dict(payload.get("receiver_contract", {}))
    inputs = set(contract.get("inputs", ()))
    forbidden = set(contract.get("forbidden_inputs", ()))
    if inputs and inputs != {"z1", "x1"}:
        raise ValueError(f"checkpoint receiver inputs are not exactly z1/x1: {sorted(inputs)}")
    required_forbidden = {"img", "z2", "q2", "oracle_indices"}
    if forbidden and not required_forbidden.issubset(forbidden):
        raise ValueError(
            "checkpoint receiver contract does not forbid all sender-only inputs: "
            f"missing={sorted(required_forbidden - forbidden)}"
        )


def build_models(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    argparse.Namespace,
    argparse.Namespace,
    Any,
    torch.nn.Module,
    torch.nn.Module,
    torch.nn.Module,
]:
    payload = load_payload(checkpoint_path)
    check_saved_contract(payload)
    saved_args = argparse.Namespace(**dict(payload["args"]))
    # v5 predates the opt-in residual-q wrapper.  Preserve its original base
    # predictor topology when evaluated by a newer trainer support module.
    if not hasattr(saved_args, "residual_q"):
        saved_args.residual_q = False
    oracle_path = payload.get("oracle_checkpoint") or getattr(saved_args, "oracle_checkpoint", "")
    if not oracle_path:
        raise ValueError(f"receiver checkpoint does not identify its sender oracle: {checkpoint_path}")

    bundle, oracle_args, _oracle_payload = receiver.load_fsq_oracle(str(oracle_path), device)

    if "oracle_tokenizer_state_dict" in payload:
        receiver.base.jsccf_io.load_state(
            bundle.tokenizer,
            payload["oracle_tokenizer_state_dict"],
            "ceiling_sender_tokenizer",
            strict=True,
        )
    if "oracle_combiner_state_dict" in payload:
        receiver.base.jsccf_io.load_state(
            bundle.combiner,
            payload["oracle_combiner_state_dict"],
            "ceiling_sender_combiner",
            strict=True,
        )

    predictor = receiver.build_predictor(saved_args, oracle_args, device)
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)

    receiver_d2 = receiver.build_receiver_d2(saved_args, bundle.tokenizer.d3, device)
    receiver_combiner = receiver.build_receiver_combiner(saved_args, bundle.combiner, device)
    receiver.assert_receiver_topology(saved_args, bundle, receiver_d2, receiver_combiner)
    receiver_d2.load_state_dict(payload["receiver_d2_state_dict"], strict=True)
    receiver_combiner.load_state_dict(payload["receiver_combiner_state_dict"], strict=True)

    modules = (
        bundle.e1,
        bundle.d1,
        bundle.tokenizer,
        bundle.combiner,
        predictor,
        receiver_d2,
        receiver_combiner,
    )
    for module in modules:
        module.requires_grad_(False)
        module.eval()
    trainable = sum(int(parameter.numel()) for module in modules for parameter in module.parameters() if parameter.requires_grad)
    if trainable != 0:
        raise AssertionError(f"strict no-grad probe found {trainable} trainable parameters")
    return payload, saved_args, oracle_args, bundle, predictor, receiver_d2, receiver_combiner


def build_val_loader(
    oracle_args: argparse.Namespace,
    *,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    loader_args = argparse.Namespace(**vars(oracle_args))
    loader_args.data_dir = str(data_dir)
    loader_args.batch_size = int(batch_size)
    loader_args.test_batch = int(batch_size)
    loader_args.num_workers = int(num_workers)
    loader_args.val_num_workers = int(num_workers)
    loader_args.cpu = device.type == "cpu"
    config = receiver.base.jsccf_io.build_config(loader_args, encoder_in_chans=3)
    _train_loader, val_loader = receiver.base.get_loader(config)
    if val_loader is None:
        raise RuntimeError("DIV2K validation loader is unavailable")
    return val_loader


def add_psnr_sums(
    sums: dict[str, float],
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> None:
    for name in PATH_NAMES:
        scores = receiver.base.psnr_per_image(outputs[name], targets).detach().double()
        if scores.ndim != 1 or int(scores.numel()) != int(targets.shape[0]):
            raise AssertionError(f"{name} PSNR is not one value per image: shape={tuple(scores.shape)}")
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError(f"{name} produced non-finite PSNR")
        sums[name] += float(scores.sum().item())


@torch.inference_mode()
def evaluate(
    val_loader,
    *,
    bundle,
    predictor: torch.nn.Module,
    receiver_d2: torch.nn.Module,
    receiver_combiner: torch.nn.Module,
    device: torch.device,
    max_val_batches: int,
) -> dict[str, Any]:
    if torch.is_grad_enabled():
        raise AssertionError("evaluate must run with autograd disabled")
    psnr_sums = {name: 0.0 for name in PATH_NAMES}
    q_squared_error = 0.0
    q_element_count = 0
    image_count = 0
    batch_count = 0

    for batch_index, (imgs, _labels) in enumerate(val_loader):
        if int(max_val_batches) > 0 and batch_index >= int(max_val_batches):
            break
        imgs = imgs.to(device, non_blocking=True)
        layer1 = bundle.layer1(imgs)
        x1 = layer1["x1"]
        z1 = layer1["z1"]

        # img enters only the sender oracle.  The receiver predictor is called
        # separately and accepts the typed (z1, x1) condition only.
        sender = bundle.tokenizer(imgs, x1, z1, bundle.combiner)
        condition = receiver.make_receiver_condition(z1, x1, detach=True)
        if batch_index == 0:
            receiver.assert_training_targets_are_not_inputs(
                predictor,
                condition,
                source_targets={
                    "img": imgs,
                    "z2": sender["z3"],
                    "oracle_q2": sender["q3_hard"],
                    "oracle_indices": sender["codes"],
                },
            )
        prediction = predictor(condition)
        qhat = (
            prediction.q_hard
            if bool(getattr(predictor, "hard_fsq", True))
            else prediction.q_continuous
        )
        q_true = sender["q3_hard"]
        if tuple(qhat.shape) != tuple(q_true.shape):
            raise AssertionError(f"qhat/q_true shape mismatch: {tuple(qhat.shape)} != {tuple(q_true.shape)}")
        q_mid = 0.5 * (qhat + q_true)

        receiver_qhat = receiver.decode_receiver(
            bundle.tokenizer, receiver_d2, qhat, x1, z1, receiver_combiner
        )["final"]
        receiver_true_q = receiver.decode_receiver(
            bundle.tokenizer, receiver_d2, q_true, x1, z1, receiver_combiner
        )["final"]
        receiver_mid_q = receiver.decode_receiver(
            bundle.tokenizer, receiver_d2, q_mid, x1, z1, receiver_combiner
        )["final"]
        outputs = {
            "x1": x1,
            "sender_oracle": sender["final"],
            "receiver_qhat": receiver_qhat,
            "receiver_true_q": receiver_true_q,
            "receiver_mid_q": receiver_mid_q,
        }
        if any(tensor.requires_grad or tensor.grad_fn is not None for tensor in outputs.values()):
            raise AssertionError("strict no-grad contract violated by an evaluated output")
        add_psnr_sums(psnr_sums, outputs, imgs)

        difference = qhat.detach().double() - q_true.detach().double()
        q_squared_error += float(difference.square().sum().item())
        q_element_count += int(difference.numel())
        image_count += int(imgs.shape[0])
        batch_count += 1
        if batch_count == 1 or batch_count % 10 == 0:
            print(f"[receiver ceiling] batches={batch_count} images={image_count}", flush=True)

    if image_count == 0:
        raise RuntimeError("no DIV2K validation images were evaluated")
    psnr = {name: value / float(image_count) for name, value in psnr_sums.items()}
    delta = {name: psnr[name] - psnr["x1"] for name in PATH_NAMES if name != "x1"}
    gap_to_true = {name: psnr["receiver_true_q"] - psnr[name] for name in PATH_NAMES}
    metrics = {
        "psnr_x1_db": psnr["x1"],
        "psnr_sender_oracle_db": psnr["sender_oracle"],
        "psnr_receiver_qhat_db": psnr["receiver_qhat"],
        "psnr_receiver_true_q_db": psnr["receiver_true_q"],
        "psnr_receiver_mid_q_db": psnr["receiver_mid_q"],
        "delta_sender_oracle_vs_x1_db": delta["sender_oracle"],
        "delta_receiver_qhat_vs_x1_db": delta["receiver_qhat"],
        "delta_receiver_true_q_vs_x1_db": delta["receiver_true_q"],
        "delta_receiver_mid_q_vs_x1_db": delta["receiver_mid_q"],
        "gap_receiver_true_q_minus_qhat_db": psnr["receiver_true_q"] - psnr["receiver_qhat"],
        "gap_receiver_true_q_minus_mid_q_db": psnr["receiver_true_q"] - psnr["receiver_mid_q"],
        "gap_receiver_mid_q_minus_qhat_db": psnr["receiver_mid_q"] - psnr["receiver_qhat"],
        "gap_sender_oracle_minus_qhat_db": psnr["sender_oracle"] - psnr["receiver_qhat"],
        "gap_sender_oracle_minus_receiver_true_q_db": psnr["sender_oracle"] - psnr["receiver_true_q"],
        "gap_sender_oracle_minus_receiver_mid_q_db": psnr["sender_oracle"] - psnr["receiver_mid_q"],
        "qhat_true_q_mse": q_squared_error / float(max(1, q_element_count)),
    }
    return {
        "image_count": image_count,
        "batch_count": batch_count,
        "receiver_only_audit": True,
        "psnr_db": psnr,
        "delta_vs_x1_db": delta,
        "gap_to_receiver_true_q_db": gap_to_true,
        "metrics": metrics,
    }


def markdown_report(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    psnr = evaluation["psnr_db"]
    delta = evaluation["delta_vs_x1_db"]
    gap = evaluation["gap_to_receiver_true_q_db"]
    rows = []
    for name in PATH_NAMES:
        delta_text = "0.000000" if name == "x1" else f"{float(delta[name]):.6f}"
        rows.append(
            f"| `{name}` | {float(psnr[name]):.6f} | {delta_text} | {float(gap[name]):.6f} |"
        )
    metrics = evaluation["metrics"]
    scope = result["dataset"]
    return "\n".join(
        [
            "# FSQ receiver latent-ceiling probe",
            "",
            f"- Checkpoint: `{result['checkpoint']['path']}`",
            f"- Epoch: `{result['checkpoint']['epoch']}`; version: `{result['checkpoint']['version']}`",
            f"- Device: `{result['runtime']['device']}`",
            f"- DIV2K val: `{evaluation['image_count']}/{scope['dataset_images']}` images, "
            f"`{evaluation['batch_count']}` batches; full validation: `{scope['full_validation']}`",
            f"- Reproduce: `{result['runtime']['command']}`",
            "",
            "## Definitions",
            "",
            "- `sender_oracle`: true hard FSQ q decoded by the frozen sender D2/combiner.",
            "- `receiver_qhat`: qhat predicted strictly from `(z1, x1)`, decoded by receiver D2/combiner.",
            "- `receiver_true_q`: true hard FSQ q decoded by receiver D2/combiner; this is an empirical "
            "fixed-decoder latent ceiling, not a deployable input or mathematical upper bound.",
            "- `receiver_mid_q`: `q_mid = 0.5 * (qhat + q_true)`; this continuous diagnostic is not "
            "necessarily a valid hard FSQ token.",
            "- Delta is PSNR minus `x1`; gap in the table is `PSNR(receiver_true_q) - PSNR(path)`.",
            "",
            "## Validation-set means",
            "",
            "| Path | PSNR (dB) | Delta vs x1 (dB) | Gap to receiver true-q (dB) |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            "## Latent gaps",
            "",
            f"- Receiver true-q minus qhat: `{metrics['gap_receiver_true_q_minus_qhat_db']:.6f} dB`",
            f"- Receiver mid-q minus qhat: `{metrics['gap_receiver_mid_q_minus_qhat_db']:.6f} dB`",
            f"- Receiver true-q minus mid-q: `{metrics['gap_receiver_true_q_minus_mid_q_db']:.6f} dB`",
            f"- Sender oracle minus receiver qhat: `{metrics['gap_sender_oracle_minus_qhat_db']:.6f} dB`",
            f"- qhat/true-q element MSE: `{metrics['qhat_true_q_mse']:.9f}`",
            "",
            "The entire evaluation loop uses `torch.inference_mode`; all loaded modules are frozen and in eval mode.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (auto, cpu, cuda, cuda:0, ...).",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="0 evaluates the complete DIV2K validation set; positive values are smoke/debug limits.",
    )
    args = parser.parse_args()
    if int(args.batch_size) <= 0:
        parser.error("--batch-size must be positive")
    if int(args.num_workers) < 0:
        parser.error("--num-workers must be non-negative")
    if int(args.max_val_batches) < 0:
        parser.error("--max-val-batches must be non-negative")
    return args


def main() -> None:
    cli = parse_args()
    checkpoint_path = resolve_path(cli.checkpoint)
    data_dir = resolve_path(cli.data_dir)
    output_dir = resolve_path(cli.output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not (data_dir / "DIV2K_valid_HR").is_dir():
        raise FileNotFoundError(data_dir / "DIV2K_valid_HR")
    device = select_device(cli.device)

    payload, saved_args, oracle_args, bundle, predictor, receiver_d2, receiver_combiner = build_models(
        checkpoint_path, device
    )
    seed = int(getattr(saved_args, "seed", 0))
    receiver.seed_everything(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    val_loader = build_val_loader(
        oracle_args,
        data_dir=data_dir,
        batch_size=int(cli.batch_size),
        num_workers=int(cli.num_workers),
        device=device,
    )
    evaluation = evaluate(
        val_loader,
        bundle=bundle,
        predictor=predictor,
        receiver_d2=receiver_d2,
        receiver_combiner=receiver_combiner,
        device=device,
        max_val_batches=int(cli.max_val_batches),
    )
    dataset_images = int(len(val_loader.dataset))
    full_validation = int(evaluation["image_count"]) == dataset_images
    if int(cli.max_val_batches) == 0 and not full_validation:
        raise AssertionError(
            f"full validation requested but processed {evaluation['image_count']}/{dataset_images} images"
        )

    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    topology = dict(payload.get("receiver_topology", {}))
    result = {
        "schema": "explore2_fsq_receiver_latent_ceiling_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": display_path(checkpoint_path),
            "epoch": int(payload.get("epoch", -1)),
            "version": str(getattr(saved_args, "version", checkpoint_path.parent.name)),
            "stage": str(payload.get("stage")),
            "saved_metrics": dict(payload.get("metrics", {})),
            "receiver_topology": topology,
            "receiver_contract": dict(payload.get("receiver_contract", {})),
        },
        "runtime": {
            "device": str(device),
            "torch_version": str(torch.__version__),
            "seed": seed,
            "strict_no_grad": True,
            "all_modules_frozen": True,
            "command": command,
        },
        "dataset": {
            "name": "DIV2K_valid_HR",
            "data_dir": display_path(data_dir),
            "dataset_images": dataset_images,
            "batch_size": int(cli.batch_size),
            "num_workers": int(cli.num_workers),
            "max_val_batches": int(cli.max_val_batches),
            "full_validation": full_validation,
            "transform": "CenterCrop(256x256) + ToTensor",
        },
        "definitions": {
            "qhat": "predictor(z1, x1) evaluation output; predictor receives no img, z2, q_true, or indices",
            "q_true": "sender E2/FSQ hard quantized q3_hard; sender-only diagnostic",
            "q_mid": "0.5 * (qhat + q_true); continuous interpolation diagnostic, not necessarily a hard FSQ token",
            "receiver_true_q_ceiling": (
                "exact sender hard q decoded by the fixed receiver D2/combiner; empirical latent-side ceiling, "
                "not a mathematical upper bound"
            ),
            "delta": "path PSNR minus x1 PSNR, averaged over per-image PSNR",
            "gap_to_receiver_true_q": "receiver_true_q PSNR minus path PSNR",
        },
        "evaluation": evaluation,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{checkpoint_path.stem}_latent_ceiling"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(evaluation["metrics"], indent=2, sort_keys=True), flush=True)
    print(f"wrote JSON: {json_path}", flush=True)
    print(f"wrote Markdown: {md_path}", flush=True)


if __name__ == "__main__":
    main()
