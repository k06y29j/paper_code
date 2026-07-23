#!/usr/bin/env python3
"""Receiver-only conditional autoregressive prediction for channel-VQ indices.

The frozen sender oracle creates index labels during training only.  The
deployment graph is strictly::

    (z1, x1) -> conditional AR(indices) -> shared codebook -> q2_hat
             -> receiver synthesis/D2/combiner -> x2_hat

Neither ``img`` nor ``z2`` is accepted by the predictor's deployment forward.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from contracts import (
    assert_receiver_only_module,
    assert_training_targets_are_not_inputs,
    make_receiver_condition,
)
from receiver_models import ChannelVQAutoregressiveIndexPredictor
import train_layer2_vq_nested as vqtrain


def psnr_values(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    return -10.0 * mse.clamp_min(1e-12).log10()


class Means:
    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.weight: dict[str, float] = {}

    def add(self, name: str, value: float | torch.Tensor, weight: int) -> None:
        scalar = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
        self.total[name] = self.total.get(name, 0.0) + scalar * int(weight)
        self.weight[name] = self.weight.get(name, 0.0) + int(weight)

    def result(self) -> dict[str, float]:
        return {name: value / self.weight[name] for name, value in self.total.items()}


def checkpoint_namespace(payload: dict, cli: argparse.Namespace) -> argparse.Namespace:
    saved = dict(payload.get("args", {}))
    saved.setdefault("channel_codebook_mode", "global")
    saved.setdefault("layer2_arch", "match")
    saved.setdefault("layer2_source_checkpoint", "")
    saved.setdefault("embedding_dim", 0)
    saved.setdefault("receiver_phase", "none")
    saved.setdefault("receiver_stack", "shared")
    saved.setdefault("lambda_zero_anchor", 0.0)
    saved.setdefault("lambda_shuffle_anchor", 0.0)
    saved.setdefault("lambda_condition_anchor", 0.0)
    saved.setdefault("predictor_hidden", 128)
    saved.setdefault("predictor_blocks", 6)
    saved.setdefault("predictor_attention_every", 2)
    saved.setdefault("predictor_heads", 4)
    saved.setdefault("condition_mode", "z1_x1")
    saved.setdefault("combiner", "residual")
    saved.setdefault("enhanced_combiner_width", 48)
    saved.setdefault("enhanced_combiner_blocks", 4)
    saved.setdefault("freeze_encoder", False)
    saved.setdefault("freeze_codebook", False)
    saved.setdefault("freeze_source_d2", False)
    saved.setdefault("freeze_combiner", False)
    saved.setdefault("receiver_only", False)
    saved.setdefault("oracle_only", True)
    saved["cpu"] = bool(cli.cpu)
    args = argparse.Namespace(**saved)
    args._rates = [int(value) for value in payload.get("rates", vqtrain.parse_int_list(args.rates))]
    args._rate_weights = list(
        payload.get("rate_weights", vqtrain.parse_float_list(args.rate_weights, len(args._rates)))
    )
    return args


def load_oracle(
    path: Path, cli: argparse.Namespace, device: torch.device
) -> tuple[vqtrain.VQBundle, argparse.Namespace, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("stage", "")) != "explore2_layer2_nested_vq":
        raise ValueError(f"not an explore-2 nested VQ checkpoint: {path}")
    args = checkpoint_namespace(payload, cli)
    if str(args.vq_family) != "channel-vq":
        raise ValueError("channel AR requires a channel-vq oracle checkpoint")
    mode = str(args.channel_codebook_mode)
    if mode not in {"global", "grouped"}:
        raise ValueError(f"unsupported channel-vq codebook mode {mode!r}")
    source = vqtrain.load_source(args, device)
    vqtrain.resolve_embedding_dim(args, source)
    bundle = vqtrain.build_bundle(args, source, device)
    bundle.codec.load_state_dict(payload["codec_state_dict"], strict=True)
    bundle.combiner.load_state_dict(payload["combiner_state_dict"], strict=True)
    bundle.codec.requires_grad_(False).eval()
    bundle.combiner.requires_grad_(False).eval()
    bundle.source.e1.requires_grad_(False).eval()
    bundle.source.d1.requires_grad_(False).eval()
    return bundle, args, payload


@torch.no_grad()
def sender_targets(
    bundle: vqtrain.VQBundle,
    imgs: torch.Tensor,
    rate: int,
) -> dict[str, torch.Tensor]:
    layer1 = bundle.source.layer1(imgs)
    z2, _z320 = bundle.codec.encode(imgs, layer1["x1"])
    _q_st, q_hard, global_indices, _stats = bundle.codec.quantizer.forward_at_k(
        z2, int(rate), update_usage=False
    )
    # The historical field name is retained for grouped checkpoints.  For a
    # global codebook this is an identity operation, so these are exact K-row
    # labels rather than per-channel round ids.
    local_indices = bundle.codec.quantizer.indices_to_local(global_indices, int(rate))
    oracle = bundle.codec.decode(q_hard, layer1["x1"], bundle.combiner)["final"]
    condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
    return {
        "x1": layer1["x1"],
        "z1": layer1["z1"],
        "z2": z2,
        "q2": q_hard,
        "global_indices": global_indices,
        "local_indices": local_indices,
        "oracle": oracle,
        "condition": condition,
    }


def local_to_q(
    bundle: vqtrain.VQBundle, local_indices: torch.Tensor, rate: int
) -> torch.Tensor:
    """Decode AR labels using the sender's exact shared-prefix lookup rule.

    Grouped mode predicts a channel-local round id and maps it to a
    channel-owned global row.  Global mode predicts the global codebook row
    directly.  The branch is receiver-side bookkeeping only: it never sees a
    sender image, z2, or q2 target.
    """

    quantizer = bundle.codec.quantizer
    if str(quantizer.channel_codebook_mode) == "grouped":
        global_indices = quantizer.local_to_global_indices(local_indices, int(rate))
    else:
        global_indices = local_indices
    return bundle.codec.quantizer.get_codebook_entry(
        global_indices, int(rate), detach_codebook=True
    )


def logits_to_soft_q(
    bundle: vqtrain.VQBundle,
    logits: torch.Tensor,
    rate: int,
    temperature: float,
) -> torch.Tensor:
    quantizer = bundle.codec.quantizer
    channels = int(quantizer.channels)
    vocabulary = int(quantizer.local_code_count(int(rate)))
    if tuple(int(value) for value in logits.shape[1:]) != (channels, vocabulary):
        raise ValueError(
            f"AR logits must be [B,{channels},{vocabulary}], got {tuple(logits.shape)}"
        )
    codebook = quantizer.codebook_at_k(int(rate)).detach()
    vector_dim = int(math.prod(quantizer.embedding_shape))
    probabilities = F.softmax(logits / float(temperature), dim=-1)
    if str(quantizer.channel_codebook_mode) == "grouped":
        # Sender rows are round-major [round, channel]; expose the expected
        # per-channel vocabulary [channel, round] to the AR logits.
        grouped = (
            codebook.reshape(vocabulary, channels, vector_dim)
            .permute(1, 0, 2)
            .contiguous()
        )
        soft = torch.einsum("bcr,crd->bcd", probabilities, grouped)
    else:
        # Every global channel token uses the same K-row prefix.
        global_codebook = codebook.reshape(int(rate), vector_dim)
        soft = torch.einsum("bck,kd->bcd", probabilities, global_codebook)
    return soft.reshape(
        int(logits.shape[0]), channels, int(quantizer.h), int(quantizer.w)
    )


def run_epoch(
    loader,
    *,
    bundle: vqtrain.VQBundle,
    predictor: ChannelVQAutoregressiveIndexPredictor,
    receiver: vqtrain.ReceiverDecodeStack,
    optimizer: optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    predictor.train(train)
    receiver.train(train)
    meters = Means()
    maximum = int(args.max_train_batches if train else args.max_val_batches)
    audited = False
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if maximum > 0 and batch_index > maximum:
            break
        imgs = imgs.to(device, non_blocking=True)
        target = sender_targets(bundle, imgs, int(args.rate))
        condition = target["condition"]
        if not audited:
            assert_training_targets_are_not_inputs(
                predictor,
                condition,
                source_targets={
                    "img": imgs,
                    "z2": target["z2"],
                    "q2": target["q2"],
                    "oracle_indices": target["global_indices"],
                },
            )
            audited = True

        with torch.set_grad_enabled(train):
            teacher_logits = predictor.forward_teacher(condition, target["local_indices"])
            loss_index = F.cross_entropy(
                teacher_logits.reshape(-1, int(args.local_code_count)),
                target["local_indices"].reshape(-1),
                label_smoothing=float(args.label_smoothing),
            )
            if train and int(imgs.shape[0]) > 1 and float(args.lambda_condition_ce) > 0.0:
                permutation = torch.roll(
                    torch.arange(int(imgs.shape[0]), device=device), shifts=1
                )
                wrong_condition = make_receiver_condition(
                    condition.z1[permutation], condition.x1[permutation], detach=True
                )
                wrong_logits = predictor.forward_teacher(
                    wrong_condition, target["local_indices"]
                )
                loss_wrong_condition = F.cross_entropy(
                    wrong_logits.reshape(-1, int(args.local_code_count)),
                    target["local_indices"].reshape(-1),
                    label_smoothing=float(args.label_smoothing),
                )
                loss_condition_ce = F.relu(
                    float(args.condition_ce_margin)
                    + loss_index
                    - loss_wrong_condition
                )
            else:
                loss_wrong_condition = loss_index.detach()
                loss_condition_ce = loss_index.new_zeros(())
            # Deployment remains autoregressive.  Hard mode decodes generated
            # indices; soft mode decodes the expected code embedding from the
            # same causal logits and allows final-image gradients into them.
            if str(args.decode_mode) == "soft":
                greedy_logits, predicted_local = predictor(condition)
                q_pred = logits_to_soft_q(
                    bundle,
                    greedy_logits,
                    int(args.rate),
                    float(args.soft_temperature),
                )
            else:
                with torch.no_grad():
                    _greedy_logits, predicted_local = predictor(condition)
                    q_pred = local_to_q(bundle, predicted_local, int(args.rate))
            predicted = receiver.decode(q_pred, target["x1"])["final"]
            loss_final = F.mse_loss(predicted.float(), imgs.float())
            loss = (
                float(args.lambda_index) * loss_index
                + float(args.lambda_condition_ce) * loss_condition_ce
                + float(args.lambda_final) * loss_final
            )
            if train:
                if optimizer is None:
                    raise RuntimeError("training requires an optimizer")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(predictor.parameters()) + list(receiver.parameters()),
                    float(args.grad_clip_norm),
                )
                optimizer.step()

        batch = int(imgs.shape[0])
        x1_psnr = psnr_values(target["x1"], imgs)
        pred_psnr = psnr_values(predicted, imgs)
        oracle_psnr = psnr_values(target["oracle"], imgs)
        meters.add("loss", loss, batch)
        meters.add("loss_index", loss_index, batch)
        meters.add("loss_wrong_condition", loss_wrong_condition, batch)
        meters.add("loss_condition_ce", loss_condition_ce, batch)
        meters.add("loss_final", loss_final, batch)
        meters.add("psnr_x1", x1_psnr.mean(), batch)
        meters.add("psnr_oracle", oracle_psnr.mean(), batch)
        meters.add("psnr_pred", pred_psnr.mean(), batch)
        meters.add("delta_oracle", (oracle_psnr - x1_psnr).mean(), batch)
        meters.add("delta_x1", (pred_psnr - x1_psnr).mean(), batch)
        meters.add(
            "index_accuracy",
            (predicted_local == target["local_indices"]).float().mean(),
            batch,
        )
        if not train:
            zero = receiver.decode(torch.zeros_like(q_pred), target["x1"])["final"]
            shuffled = receiver.decode(
                bundle.codec.quantizer.shuffle_tokens(q_pred), target["x1"]
            )["final"]
            meters.add("pred_drop_zero", (pred_psnr - psnr_values(zero, imgs)).mean(), batch)
            meters.add(
                "pred_drop_shuffle", (pred_psnr - psnr_values(shuffled, imgs)).mean(), batch
            )
            if batch > 1:
                permutation = torch.roll(torch.arange(batch, device=device), 1)
                wrong_condition = make_receiver_condition(
                    condition.z1[permutation], condition.x1[permutation], detach=True
                )
                wrong_logits, wrong_local = predictor(wrong_condition)
                wrong_q = (
                    logits_to_soft_q(
                        bundle,
                        wrong_logits,
                        int(args.rate),
                        float(args.soft_temperature),
                    )
                    if str(args.decode_mode) == "soft"
                    else local_to_q(bundle, wrong_local, int(args.rate))
                )
                wrong = receiver.decode(wrong_q, target["x1"])["final"]
                meters.add(
                    "condition_shuffle_drop",
                    (pred_psnr - psnr_values(wrong, imgs)).mean(),
                    batch,
                )
    result = meters.result()
    result["receiver_only_audit"] = float(audited)
    if not train:
        result["full_validation"] = float(
            int(result.get("evaluated_images", len(loader.dataset))) == len(loader.dataset)
            if "evaluated_images" in result
            else maximum == 0
        )
        result["goal_met"] = float(
            result.get("delta_x1", -math.inf) >= float(args.min_delta_db)
            and result.get("pred_drop_zero", -math.inf) >= float(args.min_ablation_drop)
            and result.get("condition_shuffle_drop", -math.inf)
            >= float(args.min_condition_drop)
        )
    return result


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    predictor: ChannelVQAutoregressiveIndexPredictor,
    receiver: vqtrain.ReceiverDecodeStack,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    oracle_path: Path,
    metrics: dict[str, float],
    best: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "explore2_channel_vq_ar_receiver",
            "epoch": int(epoch),
            "args": vars(args),
            "metrics": metrics,
            "best_delta_x1": float(best),
            "oracle_checkpoint": str(oracle_path),
            "predictor_state_dict": predictor.state_dict(),
            "receiver_stack_state_dict": receiver.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "receiver_contract": {
                "inputs": ["z1", "x1", "generated_past_indices"],
                "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
                "index_mode": str(getattr(args, "ar_index_mode", "unknown")),
                "output": "channel_vq_autoregressive_indices_and_q2_hat",
            },
        },
        path,
    )
    print(f"saved checkpoint: {path}", flush=True)


def train(args: argparse.Namespace) -> None:
    vqtrain.seed_everything(int(args.seed))
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    oracle_path = vqtrain.resolve_path(args.oracle_checkpoint)
    bundle, oracle_args, oracle_payload = load_oracle(oracle_path, args, device)
    args.rate = int(args.rate) if int(args.rate) > 0 else max(oracle_args._rates)
    if args.rate not in oracle_args._rates:
        raise ValueError(f"rate K={args.rate} is not in oracle rates {oracle_args._rates}")
    quantizer = bundle.codec.quantizer
    channels = int(quantizer.channels)
    mode = str(quantizer.channel_codebook_mode)
    # Preserve the oracle's index semantics in every receiver checkpoint.
    # Grouped labels are per-channel rounds; global labels are K-prefix rows.
    args.channel_codebook_mode = mode
    args.ar_index_mode = "per_channel_round" if mode == "grouped" else "global_codebook_row"
    args.local_code_count = int(quantizer.local_code_count(int(args.rate)))
    if int(args.local_code_count) < 2:
        raise ValueError(
            f"channel AR requires at least two index candidates, got "
            f"mode={mode} K={args.rate} vocabulary={args.local_code_count}"
        )
    probe = argparse.Namespace(**bundle.source.checkpoint["args"])
    train_loader, val_loader, loader_device = vqtrain.build_loaders(args, probe)
    if loader_device.type != device.type:
        raise RuntimeError(f"loader/source device mismatch: {loader_device} vs {device}")
    device = loader_device
    predictor = ChannelVQAutoregressiveIndexPredictor(
        int(bundle.source.args.latent_ch),
        channels,
        int(args.local_code_count),
        hidden=int(args.hidden),
        blocks=int(args.blocks),
        attention_every=int(args.attention_every),
        heads=int(args.heads),
        condition_mode=str(args.condition_mode),
    ).to(device)
    assert_receiver_only_module(predictor)
    receiver_combiner = (
        vqtrain.EnhancedResidualCorrectionCombiner(
            bundle.combiner,
            width=int(args.receiver_combiner_width),
            blocks=int(args.receiver_combiner_blocks),
        ).to(device)
        if str(args.receiver_combiner) == "enhanced"
        else bundle.combiner
    )
    receiver = vqtrain.ReceiverDecodeStack(
        bundle.codec.synthesis, bundle.codec.d2, receiver_combiner
    ).to(device)
    optimizer = optim.AdamW(
        [
            {"params": predictor.parameters(), "lr": float(args.lr)},
            {"params": receiver.parameters(), "lr": float(args.decoder_lr)},
        ],
        weight_decay=float(args.weight_decay),
    )
    start_epoch = 1
    best = -math.inf
    if args.resume:
        resume_path = vqtrain.resolve_path(args.resume)
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        if str(resume_payload.get("stage", "")) != "explore2_channel_vq_ar_receiver":
            raise ValueError(f"not a channel-VQ AR receiver checkpoint: {resume_path}")
        predictor.load_state_dict(resume_payload["predictor_state_dict"], strict=True)
        receiver.load_state_dict(resume_payload["receiver_stack_state_dict"], strict=True)
        if not bool(args.reset_optimizer_on_resume):
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        start_epoch = int(resume_payload["epoch"]) + 1
        best = float(resume_payload.get("best_delta_x1", -math.inf))
        print(
            f"resumed receiver: {resume_path} start_epoch={start_epoch} best={best:.6f}",
            flush=True,
        )
    print("=== explore-2 | channel-VQ conditional AR receiver ===", flush=True)
    print("实验设计", flush=True)
    print(
        f"  frozen oracle={oracle_path} mode={mode} C={channels} K={args.rate} "
        f"index_vocab={args.local_code_count}; deployment=(z1,x1,past_generated_indices)->q2_hat",
        flush=True,
    )
    print("  forbidden deployment inputs=img,z2,q2,oracle_indices", flush=True)
    print("loss设计", flush=True)
    print(
        f"  {args.lambda_index:g}*teacher-forced causal CE + "
        f"{args.lambda_condition_ce:g}*condition-ranking hinge + "
        f"{args.lambda_final:g}*MSE(receiver_x2_hat,img); decoder sees greedy deployment q2_hat",
        flush=True,
    )
    print("模块选择", flush=True)
    print(
        f"  predictor=ChannelVQAutoregressiveIndexPredictor hidden={args.hidden}; "
        f"decode_mode={args.decode_mode} temperature={args.soft_temperature:g}; "
        f"sender/codebook=frozen; receiver synthesis/D2/combiner={args.receiver_combiner} "
        "independent trainable",
        flush=True,
    )
    output = vqtrain.resolve_path(args.save_dir) / str(args.version)
    for epoch in range(start_epoch, int(args.epochs) + 1):
        began = time.time()
        train_metrics = run_epoch(
            train_loader,
            bundle=bundle,
            predictor=predictor,
            receiver=receiver,
            optimizer=optimizer,
            args=args,
            device=device,
            train=True,
        )
        print(
            f"[channel-ar train {epoch:03d}/{int(args.epochs):03d}] {train_metrics} "
            f"time={time.time()-began:.1f}s",
            flush=True,
        )
        if epoch == 1 or epoch % int(args.val_every) == 0:
            with torch.no_grad():
                val_metrics = run_epoch(
                    val_loader,
                    bundle=bundle,
                    predictor=predictor,
                    receiver=receiver,
                    optimizer=None,
                    args=args,
                    device=device,
                    train=False,
                )
            print(f"[channel-ar val {epoch:03d}] {val_metrics}", flush=True)
            latest = output / "channel_vq_ar_latest.pth"
            save_checkpoint(
                latest,
                epoch=epoch,
                predictor=predictor,
                receiver=receiver,
                optimizer=optimizer,
                args=args,
                oracle_path=oracle_path,
                metrics=val_metrics,
                best=best,
            )
            if float(val_metrics["delta_x1"]) > best:
                best = float(val_metrics["delta_x1"])
                save_checkpoint(
                    output / "channel_vq_ar_best.pth",
                    epoch=epoch,
                    predictor=predictor,
                    receiver=receiver,
                    optimizer=optimizer,
                    args=args,
                    oracle_path=oracle_path,
                    metrics=val_metrics,
                    best=best,
                )
    summary = {"best_delta_x1": best, "oracle_metrics": oracle_payload.get("metrics", {})}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--oracle-checkpoint", required=True)
    parser.add_argument("--rate", type=int, default=0)
    parser.add_argument("--condition-mode", choices=["z1", "x1", "z1_x1"], default="z1_x1")
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--attention-every", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lambda-index", type=float, default=1.0)
    parser.add_argument("--lambda-condition-ce", type=float, default=0.2)
    parser.add_argument("--condition-ce-margin", type=float, default=0.1)
    parser.add_argument("--lambda-final", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--decode-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--soft-temperature", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--receiver-combiner", choices=["oracle", "enhanced"], default="enhanced")
    parser.add_argument("--receiver-combiner-width", type=int, default=32)
    parser.add_argument("--receiver-combiner-blocks", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--min-delta-db", type=float, default=0.5)
    parser.add_argument("--min-ablation-drop", type=float, default=0.1)
    parser.add_argument("--min-condition-drop", type=float, default=0.1)
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument("--save-dir", default="MY-V2/jscc-f/explore-2/checkpoints-channel-ar")
    parser.add_argument("--version", default="channel-vq-ar-v1")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
