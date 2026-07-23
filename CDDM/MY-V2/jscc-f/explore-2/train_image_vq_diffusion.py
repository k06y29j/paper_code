#!/usr/bin/env python3
"""Receiver-only conditional diffusion generation of image-VQ q2 embeddings."""

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
from receiver_models import ImageVQConditionalDiffusionGenerator
from train_channel_vq_ar import Means, checkpoint_namespace, psnr_values
import train_layer2_vq_nested as vqtrain


def load_oracle(
    path: Path, cli: argparse.Namespace, device: torch.device
) -> tuple[vqtrain.VQBundle, argparse.Namespace, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("stage", "")) != "explore2_layer2_nested_vq":
        raise ValueError(f"not an explore-2 nested VQ checkpoint: {path}")
    args = checkpoint_namespace(payload, cli)
    if str(args.vq_family) != "image-vq":
        raise ValueError("image-VQ diffusion requires an image-vq oracle checkpoint")
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
    bundle: vqtrain.VQBundle, imgs: torch.Tensor, rate: int
) -> dict[str, torch.Tensor]:
    layer1 = bundle.source.layer1(imgs)
    z2, _z320 = bundle.codec.encode(imgs, layer1["x1"])
    _q_st, q_hard, indices, _stats = bundle.codec.quantizer.forward_at_k(
        z2, int(rate), update_usage=False
    )
    oracle = bundle.codec.decode(q_hard, layer1["x1"], bundle.combiner)["final"]
    return {
        "x1": layer1["x1"],
        "z1": layer1["z1"],
        "z2": z2,
        "q2": q_hard,
        "indices": indices,
        "oracle": oracle,
        "condition": make_receiver_condition(layer1["z1"], layer1["x1"], detach=True),
    }


def seeded_generation(
    generator: ImageVQConditionalDiffusionGenerator,
    condition,
    seed: int,
) -> torch.Tensor:
    devices = [condition.z1.device] if condition.z1.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if condition.z1.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        return generator(condition)


def run_epoch(
    loader,
    *,
    bundle: vqtrain.VQBundle,
    generator: ImageVQConditionalDiffusionGenerator,
    receiver: vqtrain.ReceiverDecodeStack,
    optimizer: optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    generator.train(train)
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
                generator,
                condition,
                source_targets={
                    "img": imgs,
                    "z2": target["z2"],
                    "q2": target["q2"],
                    "oracle_indices": target["indices"],
                },
            )
            audited = True

        with torch.set_grad_enabled(train):
            if train:
                diffusion = generator.training_predictions(condition, target["q2"])
                q_pred = diffusion["predicted_q"]
                loss_noise = F.mse_loss(diffusion["predicted_noise"], diffusion["noise"])
                loss_mean = diffusion["loss_mean_q"]
            else:
                q_pred = seeded_generation(
                    generator, condition, int(args.eval_seed) + int(batch_index)
                )
                loss_noise = q_pred.new_zeros(())
                loss_mean = F.mse_loss(
                    generator.predict_mean(condition).float(), target["q2"].float()
                ) / generator.q_scale.float().square().clamp_min(1e-6)
            predicted = receiver.decode(q_pred, target["x1"])["final"]
            loss_final = F.mse_loss(predicted.float(), imgs.float())
            loss = (
                float(args.lambda_noise) * loss_noise
                + float(args.lambda_mean) * loss_mean
                + float(args.lambda_final) * loss_final
            )
            if train:
                if optimizer is None:
                    raise RuntimeError("training requires an optimizer")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(generator.parameters()) + list(receiver.parameters()),
                    float(args.grad_clip_norm),
                )
                optimizer.step()

        with torch.no_grad():
            q_hard, predicted_indices = bundle.codec.quantizer.quantize_input(
                q_pred, int(args.rate), detach_codebook=True
            )
        batch = int(imgs.shape[0])
        x1_psnr = psnr_values(target["x1"], imgs)
        pred_psnr = psnr_values(predicted, imgs)
        oracle_psnr = psnr_values(target["oracle"], imgs)
        meters.add("loss", loss, batch)
        meters.add("loss_noise", loss_noise, batch)
        meters.add("loss_mean", loss_mean, batch)
        meters.add("loss_final", loss_final, batch)
        meters.add("psnr_x1", x1_psnr.mean(), batch)
        meters.add("psnr_oracle", oracle_psnr.mean(), batch)
        meters.add("psnr_pred", pred_psnr.mean(), batch)
        meters.add("delta_oracle", (oracle_psnr - x1_psnr).mean(), batch)
        meters.add("delta_x1", (pred_psnr - x1_psnr).mean(), batch)
        meters.add(
            "nearest_index_accuracy",
            (predicted_indices == target["indices"]).float().mean(),
            batch,
        )
        meters.add(
            "q_nmse",
            F.mse_loss(q_pred.float(), target["q2"].float())
            / target["q2"].float().square().mean().clamp_min(1e-6),
            batch,
        )
        if not train:
            mean_only = receiver.decode(
                generator.predict_mean(condition), target["x1"]
            )["final"]
            zero = receiver.decode(torch.zeros_like(q_pred), target["x1"])["final"]
            shuffled = receiver.decode(
                bundle.codec.quantizer.shuffle_tokens(q_pred), target["x1"]
            )["final"]
            hard = receiver.decode(q_hard, target["x1"])["final"]
            meters.add("psnr_pred_hard", psnr_values(hard, imgs).mean(), batch)
            meters.add("psnr_pred_mean", psnr_values(mean_only, imgs).mean(), batch)
            meters.add(
                "delta_x1_mean",
                (psnr_values(mean_only, imgs) - x1_psnr).mean(),
                batch,
            )
            meters.add("pred_drop_zero", (pred_psnr - psnr_values(zero, imgs)).mean(), batch)
            meters.add(
                "pred_drop_shuffle", (pred_psnr - psnr_values(shuffled, imgs)).mean(), batch
            )
            if batch > 1:
                permutation = torch.roll(torch.arange(batch, device=device), 1)
                wrong_condition = make_receiver_condition(
                    condition.z1[permutation], condition.x1[permutation], detach=True
                )
                wrong_q = seeded_generation(
                    generator, wrong_condition, int(args.eval_seed) + int(batch_index)
                )
                wrong = receiver.decode(wrong_q, target["x1"])["final"]
                meters.add(
                    "condition_shuffle_drop",
                    (pred_psnr - psnr_values(wrong, imgs)).mean(),
                    batch,
                )
    result = meters.result()
    result["receiver_only_audit"] = float(audited)
    result["residual_sample_scale"] = float(generator.residual_sample_scale)
    if not train:
        result["full_validation"] = float(maximum == 0)
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
    generator: ImageVQConditionalDiffusionGenerator,
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
            "stage": "explore2_image_vq_diffusion_receiver",
            "epoch": int(epoch),
            "args": vars(args),
            "metrics": metrics,
            "best_delta_x1": float(best),
            "oracle_checkpoint": str(oracle_path),
            "generator_state_dict": generator.state_dict(),
            "receiver_stack_state_dict": receiver.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "receiver_contract": {
                "inputs": ["z1", "x1", "internally_sampled_noise"],
                "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"],
                "output": "continuous_image_vq_q2_hat",
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
    embedding_dim = int(quantizer.channels)
    height, width = int(quantizer.h), int(quantizer.w)
    q_scale = float(quantizer.codebook[: int(args.rate)].float().square().mean().sqrt())
    probe = argparse.Namespace(**bundle.source.checkpoint["args"])
    train_loader, val_loader, loader_device = vqtrain.build_loaders(args, probe)
    if loader_device.type != device.type:
        raise RuntimeError(f"loader/source device mismatch: {loader_device} vs {device}")
    device = loader_device
    generator = ImageVQConditionalDiffusionGenerator(
        int(bundle.source.args.latent_ch),
        embedding_dim,
        height,
        width,
        hidden=int(args.hidden),
        blocks=int(args.blocks),
        attention_every=int(args.attention_every),
        heads=int(args.heads),
        condition_mode=str(args.condition_mode),
        diffusion_steps=int(args.diffusion_steps),
        sample_steps=int(args.sample_steps),
        q_scale=q_scale,
        residual_mean=bool(args.residual_mean),
    ).to(device)
    assert_receiver_only_module(generator)
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
            {"params": generator.parameters(), "lr": float(args.lr)},
            {"params": receiver.parameters(), "lr": float(args.decoder_lr)},
        ],
        weight_decay=float(args.weight_decay),
    )
    start_epoch = 1
    best = -math.inf
    if args.resume:
        resume_path = vqtrain.resolve_path(args.resume)
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        if str(resume_payload.get("stage", "")) != "explore2_image_vq_diffusion_receiver":
            raise ValueError(f"not an image-VQ diffusion receiver checkpoint: {resume_path}")
        generator.load_state_dict(resume_payload["generator_state_dict"], strict=True)
        receiver.load_state_dict(resume_payload["receiver_stack_state_dict"], strict=True)
        if not bool(args.reset_optimizer_on_resume):
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        start_epoch = int(resume_payload["epoch"]) + 1
        best = float(resume_payload.get("best_delta_x1", -math.inf))
        print(
            f"resumed receiver: {resume_path} start_epoch={start_epoch} best={best:.6f}",
            flush=True,
        )
    print("=== explore-2 | image-VQ conditional diffusion receiver ===", flush=True)
    print("实验设计", flush=True)
    print(
        f"  frozen oracle={oracle_path} K={args.rate} q=[B,{embedding_dim},{height},{width}] "
        f"scale={q_scale:.6g}; deployment=(z1,x1,internal_noise)->q2_hat",
        flush=True,
    )
    print("  forbidden deployment inputs=img,z2,q2,oracle_indices", flush=True)
    print("loss设计", flush=True)
    print(
        f"  {args.lambda_noise:g}*noise_MSE + {args.lambda_mean:g}*mean_q_NMSE + "
        f"{args.lambda_final:g}*MSE(x2_hat,img); "
        f"DDIM={args.sample_steps}/{args.diffusion_steps} steps",
        flush=True,
    )
    print("模块选择", flush=True)
    print(
        f"  diffusion_hidden={args.hidden}; sender/codebook=frozen; "
        f"receiver synthesis/D2/combiner={args.receiver_combiner} independent trainable",
        flush=True,
    )
    output = vqtrain.resolve_path(args.save_dir) / str(args.version)
    for epoch in range(start_epoch, int(args.epochs) + 1):
        residual_scale = (
            float(args.residual_max_scale)
            * min(1.0, float(epoch) / float(max(1, int(args.residual_ramp_epochs))))
            if bool(args.residual_mean)
            else 1.0
        )
        generator.set_residual_sample_scale(residual_scale)
        began = time.time()
        train_metrics = run_epoch(
            train_loader,
            bundle=bundle,
            generator=generator,
            receiver=receiver,
            optimizer=optimizer,
            args=args,
            device=device,
            train=True,
        )
        print(
            f"[image-diffusion train {epoch:03d}/{int(args.epochs):03d}] {train_metrics} "
            f"time={time.time()-began:.1f}s",
            flush=True,
        )
        if epoch == 1 or epoch % int(args.val_every) == 0:
            with torch.no_grad():
                val_metrics = run_epoch(
                    val_loader,
                    bundle=bundle,
                    generator=generator,
                    receiver=receiver,
                    optimizer=None,
                    args=args,
                    device=device,
                    train=False,
                )
            print(f"[image-diffusion val {epoch:03d}] {val_metrics}", flush=True)
            save_checkpoint(
                output / "image_vq_diffusion_latest.pth",
                epoch=epoch,
                generator=generator,
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
                    output / "image_vq_diffusion_best.pth",
                    epoch=epoch,
                    generator=generator,
                    receiver=receiver,
                    optimizer=optimizer,
                    args=args,
                    oracle_path=oracle_path,
                    metrics=val_metrics,
                    best=best,
                )
    output.mkdir(parents=True, exist_ok=True)
    summary = {"best_delta_x1": best, "oracle_metrics": oracle_payload.get("metrics", {})}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--oracle-checkpoint", required=True)
    parser.add_argument("--rate", type=int, default=0)
    parser.add_argument("--condition-mode", choices=["z1", "x1", "z1_x1"], default="z1_x1")
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--attention-every", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--residual-mean", action="store_true")
    parser.add_argument("--residual-ramp-epochs", type=int, default=30)
    parser.add_argument("--residual-max-scale", type=float, default=1.0)
    parser.add_argument("--lambda-noise", type=float, default=1.0)
    parser.add_argument("--lambda-mean", type=float, default=1.0)
    parser.add_argument("--lambda-final", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    parser.add_argument("--save-dir", default="MY-V2/jscc-f/explore-2/checkpoints-image-diffusion")
    parser.add_argument("--version", default="image-vq-diffusion-v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--eval-seed", type=int, default=2026071301)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
