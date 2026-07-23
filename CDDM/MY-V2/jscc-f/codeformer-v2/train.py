#!/usr/bin/env python3
"""Train CodeFormer-v2's requested Stage1/Stage2 contracts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import AdamW

from layer1 import build_layer1, freeze_layer1, layer1_x1, load_layer1_checkpoint
from model import (
    IMAGE_SIZE,
    LATENT_CHANNELS,
    LATENT_SIZE,
    Stage1HQCodec,
    Stage2LQRestorer,
    trainable_parameters,
)


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
CDDM_ROOT = JSCCF_DIR.parents[1]
DEFAULT_SAVE_DIR = THIS_DIR / "checkpoints"


def _load_jscc_io():
    if str(JSCCF_DIR) not in sys.path:
        sys.path.insert(0, str(JSCCF_DIR))
    spec = importlib.util.spec_from_file_location("jsccf_codeformer_v2_io", JSCCF_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import JSCC-f io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = _load_jscc_io()
from Autoencoder.data.datasets import get_loader  # noqa: E402


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else CDDM_ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(prediction: Tensor, target: Tensor) -> Tensor:
    mse = (prediction.float().clamp(0, 1) - target.float()).square().mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


class Meter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int) -> None:
        self.total += float(value) * int(count)
        self.count += int(count)

    @property
    def average(self) -> float:
        return self.total / max(1, self.count)


class CodeUsage:
    def __init__(self, codebook_size: int) -> None:
        self.codebook_size = int(codebook_size)
        self.counts = torch.zeros(self.codebook_size, dtype=torch.float64)

    @torch.no_grad()
    def update(self, indices: Tensor) -> None:
        flat = indices.detach().long().reshape(-1)
        if flat.numel() == 0:
            return
        low, high = int(flat.min().item()), int(flat.max().item())
        if low < 0 or high >= self.codebook_size:
            raise ValueError(f"code IDs must be in [0,{self.codebook_size}), got [{low},{high}]")
        self.counts += torch.bincount(flat, minlength=self.codebook_size).cpu().to(torch.float64)

    def metrics(self) -> dict[str, float]:
        tokens = float(self.counts.sum().item())
        if tokens == 0.0:
            return {key: 0.0 for key in ("code_used", "code_usage_ratio", "code_entropy_bits", "code_perplexity", "code_top1_frac", "code_tokens")}
        active = self.counts[self.counts > 0]
        probability = active / tokens
        entropy = float(-(probability * probability.log2()).sum().item())
        return {
            "code_used": float(active.numel()),
            "code_usage_ratio": float(active.numel()) / float(self.codebook_size),
            "code_entropy_bits": entropy,
            "code_perplexity": float(2.0**entropy),
            "code_top1_frac": float(self.counts.max().item()) / tokens,
            "code_tokens": tokens,
        }


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_log_file(args: argparse.Namespace) -> tuple[object, object, object]:
    if args.log_file:
        path = resolve(args.log_file)
    else:
        safe_version = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(args.version))
        path = resolve(args.save_dir) / f"jsccf_codeformer_v2_{args.stage}_{safe_version or 'default'}.log"
        args.log_file = str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stdout, stderr = sys.stdout, sys.stderr
    handle = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = TeeStream(stdout, handle), TeeStream(stderr, handle)
    print(f"\n=== JSCC-f CodeFormer-v2 @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"log file: {path}")
    print(json.dumps(vars(args), sort_keys=True))
    return handle, stdout, stderr


def update(meters: dict[str, Meter], key: str, value: Tensor | float, batch: int) -> None:
    value = float(value.detach().float().mean().item()) if torch.is_tensor(value) else float(value)
    meters[key].update(value, batch)


def summarize(meters: dict[str, Meter]) -> dict[str, float]:
    return {key: value.average for key, value in meters.items()}


def build_stage1(args: argparse.Namespace) -> Stage1HQCodec:
    return Stage1HQCodec(args.codebook_size, args.base_channels, args.num_res_blocks, args.beta)


def build_stage2(args: argparse.Namespace) -> Stage2LQRestorer:
    return Stage2LQRestorer(args.codebook_size, args.base_channels, args.num_res_blocks, args.beta)


def architecture(args: argparse.Namespace) -> dict:
    return {
        "image_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
        "stage1_hq_encoder_input": [6, IMAGE_SIZE, IMAGE_SIZE],
        "stage2_lq_encoder_input": [3, IMAGE_SIZE, IMAGE_SIZE],
        "latent_shape": [LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
        "stage1_output": "u2 then combiner(x1,u2)->x2",
        "stage2_output": "u2 then combiner(x1,u2)->x2_hat",
        "stage2_trainable": ["lq_encoder", "decoder.fuse_blocks"],
        "fuse_resolutions": [128, 64, 32],
        "codebook_size": int(args.codebook_size),
    }


def make_loader(args: argparse.Namespace):
    config = jsccf_io.build_config(args)
    return get_loader(config), config.device


def checkpoint_path(args: argparse.Namespace, suffix: str) -> Path:
    return resolve(args.save_dir) / f"jsccf_codeformer_v2_{args.stage}_{args.version}_{suffix}.pth"


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"saved checkpoint: {path}", flush=True)


def stage1_metrics(out: dict[str, Tensor], image: Tensor, x1: Tensor, meters: dict[str, Meter]) -> None:
    batch = image.shape[0]
    update(meters, "psnr_x1", psnr(x1, image), batch)
    update(meters, "psnr_x2", psnr(out["x2"], image), batch)
    update(meters, "delta_x1", psnr(out["x2"], image).mean() - psnr(x1, image).mean(), batch)
    update(meters, "vq", out["vq_loss"], batch)
    update(meters, "perplexity", out["perplexity"], batch)


@torch.no_grad()
def validate_stage1(model, e1, d1, loader, device, max_batches: int) -> dict[str, float]:
    model.eval()
    e1.eval(); d1.eval()
    meters = {key: Meter() for key in ("loss", "psnr_x1", "psnr_x2", "delta_x1", "vq", "perplexity")}
    usage = CodeUsage(model.quantizer.codebook_size)
    for step, (image, _label) in enumerate(loader, start=1):
        if max_batches and step > max_batches:
            break
        image = image.to(device, non_blocking=True)
        x1 = layer1_x1(e1, d1, image)
        out = model(image, x1)
        loss = F.mse_loss(out["x2"], image) + out["vq_loss"]
        update(meters, "loss", loss, image.shape[0])
        stage1_metrics(out, image, x1, meters)
        usage.update(out["indices"])
    metrics = summarize(meters)
    metrics.update(usage.metrics())
    return metrics


def train_stage1(args: argparse.Namespace) -> None:
    (train_loader, val_loader), device = make_loader(args)
    model = build_stage1(args).to(device)
    e1, d1 = build_layer1(args, device)
    layer1_path = load_layer1_checkpoint(args, e1, d1)
    freeze_layer1(e1, d1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("-inf")
    print(f"Stage1 frozen Layer1: {layer1_path}")
    for epoch in range(1, args.epochs + 1):
        model.train(); e1.eval(); d1.eval()
        meters = {key: Meter() for key in ("loss", "psnr_x1", "psnr_x2", "delta_x1", "vq", "perplexity")}
        usage = CodeUsage(model.quantizer.codebook_size)
        for step, (image, _label) in enumerate(train_loader, start=1):
            if args.max_train_batches and step > args.max_train_batches:
                break
            image = image.to(device, non_blocking=True)
            x1 = layer1_x1(e1, d1, image)
            out = model(image, x1)
            x2_loss = F.mse_loss(out["x2"], image)
            loss = x2_loss + args.lambda_vq * out["vq_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            update(meters, "loss", loss, image.shape[0])
            stage1_metrics(out, image, x1, meters)
            usage.update(out["indices"])
        train_metrics = summarize(meters); train_metrics.update(usage.metrics())
        print(f"[stage1 {epoch:03d}/{args.epochs}] train {json.dumps(train_metrics, sort_keys=True)}")
        if epoch % args.val_every == 0 or epoch == args.epochs:
            metrics = validate_stage1(model, e1, d1, val_loader, device, args.max_val_batches)
            print(f"[stage1 {epoch:03d}/{args.epochs}] valid {json.dumps(metrics, sort_keys=True)}")
            payload = {
                "format": "jsccf_codeformer_v2_stage1_v1", "stage": "stage1", "epoch": epoch,
                "args": vars(args), "architecture": architecture(args), "metrics": metrics,
                "layer1_arch": args.layer1_arch, "layer1_checkpoint": str(layer1_path),
                "hq_encoder_state_dict": model.hq_encoder.state_dict(),
                "quantizer_state_dict": model.quantizer.state_dict(),
                "decoder_state_dict": model.decoder.state_dict(),
                "combiner_state_dict": model.combiner.state_dict(),
            }
            save(checkpoint_path(args, "latest"), payload)
            if metrics["psnr_x2"] >= best:
                best = metrics["psnr_x2"]
                save(checkpoint_path(args, "best"), payload)


def load_stage1(path: str) -> dict:
    checkpoint = resolve(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"--stage1-ckpt does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("stage") != "stage1":
        raise ValueError("--stage1-ckpt must be a CodeFormer-v2 stage1 checkpoint")
    return payload


def stage2_metrics(out: dict[str, Tensor], image: Tensor, x1: Tensor, meters: dict[str, Meter]) -> None:
    batch = image.shape[0]
    update(meters, "psnr_x1", psnr(x1, image), batch)
    update(meters, "psnr_x2_hat", psnr(out["x2_hat"], image), batch)
    update(meters, "delta_x1", psnr(out["x2_hat"], image).mean() - psnr(x1, image).mean(), batch)
    update(meters, "vq_mse", F.mse_loss(out["q2"], out["z2"]), batch)


@torch.no_grad()
def validate_stage2(model, e1, d1, loader, device, fusion_weight: float, max_batches: int) -> dict[str, float]:
    model.eval(); e1.eval(); d1.eval()
    meters = {key: Meter() for key in ("loss", "psnr_x1", "psnr_x2_hat", "delta_x1", "vq_mse")}
    usage = CodeUsage(model.quantizer.codebook_size)
    for step, (image, _label) in enumerate(loader, start=1):
        if max_batches and step > max_batches:
            break
        image = image.to(device, non_blocking=True)
        x1 = layer1_x1(e1, d1, image)
        out = model(x1, fusion_weight)
        update(meters, "loss", F.mse_loss(out["x2_hat"], image), image.shape[0])
        stage2_metrics(out, image, x1, meters)
        usage.update(out["indices"])
    metrics = summarize(meters); metrics.update(usage.metrics())
    return metrics


def train_stage2(args: argparse.Namespace) -> None:
    if not args.stage1_ckpt:
        raise ValueError("--stage stage2 requires --stage1-ckpt")
    payload = load_stage1(args.stage1_ckpt)
    source_arch = payload.get("layer1_arch")
    if source_arch and source_arch != args.layer1_arch:
        raise ValueError(f"Stage1 used Layer1 {source_arch!r}, not requested {args.layer1_arch!r}")
    (train_loader, val_loader), device = make_loader(args)
    model = build_stage2(args).to(device)
    report = model.initialize_from_stage1(payload)
    model.freeze_stage1_modules()
    e1, d1 = build_layer1(args, device)
    layer1_path = load_layer1_checkpoint(args, e1, d1)
    freeze_layer1(e1, d1)
    parameters = trainable_parameters(model)
    optimizer = AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    print(f"Stage2 frozen Layer1: {layer1_path}; init={report}; trainable_params={sum(p.numel() for p in parameters):,}")
    best = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); e1.eval(); d1.eval()
        meters = {key: Meter() for key in ("loss", "psnr_x1", "psnr_x2_hat", "delta_x1", "vq_mse")}
        usage = CodeUsage(model.quantizer.codebook_size)
        for step, (image, _label) in enumerate(train_loader, start=1):
            if args.max_train_batches and step > args.max_train_batches:
                break
            image = image.to(device, non_blocking=True)
            x1 = layer1_x1(e1, d1, image)
            out = model(x1, args.fusion_weight)
            loss = F.mse_loss(out["x2_hat"], image)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm)
            optimizer.step()
            update(meters, "loss", loss, image.shape[0])
            stage2_metrics(out, image, x1, meters)
            usage.update(out["indices"])
        train_metrics = summarize(meters); train_metrics.update(usage.metrics())
        print(f"[stage2 {epoch:03d}/{args.epochs}] train {json.dumps(train_metrics, sort_keys=True)}")
        if epoch % args.val_every == 0 or epoch == args.epochs:
            metrics = validate_stage2(model, e1, d1, val_loader, device, args.fusion_weight, args.max_val_batches)
            print(f"[stage2 {epoch:03d}/{args.epochs}] valid {json.dumps(metrics, sort_keys=True)}")
            checkpoint = {
                "format": "jsccf_codeformer_v2_stage2_v1", "stage": "stage2", "epoch": epoch,
                "args": vars(args), "architecture": architecture(args), "metrics": metrics,
                "layer1_arch": args.layer1_arch, "layer1_checkpoint": str(layer1_path),
                "stage1_checkpoint": str(resolve(args.stage1_ckpt)), "stage1_init_report": report,
                "lq_encoder_state_dict": model.lq_encoder.state_dict(),
                "quantizer_state_dict": model.quantizer.state_dict(),
                "decoder_state_dict": model.decoder.state_dict(),
                "combiner_state_dict": model.combiner.state_dict(),
            }
            save(checkpoint_path(args, "latest"), checkpoint)
            if metrics["psnr_x2_hat"] >= best:
                best = metrics["psnr_x2_hat"]
                save(checkpoint_path(args, "best"), checkpoint)


def smoke_shapes(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    image = torch.rand(args.smoke_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    x1 = torch.rand_like(image)
    stage1 = build_stage1(args).to(device)
    out1 = stage1(image, x1)
    assert tuple(out1["hq_input"].shape[1:]) == (6, 256, 256)
    assert tuple(out1["q2"].shape[1:]) == (256, 16, 16)
    assert tuple(out1["u2"].shape[1:]) == (3, 256, 256)
    assert tuple(out1["x2"].shape[1:]) == (3, 256, 256)
    stage1_payload = {
        "hq_encoder_state_dict": stage1.hq_encoder.state_dict(),
        "quantizer_state_dict": stage1.quantizer.state_dict(),
        "decoder_state_dict": stage1.decoder.state_dict(),
        "combiner_state_dict": stage1.combiner.state_dict(),
    }
    stage2 = build_stage2(args).to(device)
    stage2.initialize_from_stage1(stage1_payload)
    stage2.freeze_stage1_modules()
    out2 = stage2(x1, args.fusion_weight)
    loss = F.mse_loss(out2["x2_hat"], image)
    loss.backward()
    assert tuple(out2["z2"].shape[1:]) == (256, 16, 16)
    assert tuple(out2["u2"].shape[1:]) == (3, 256, 256)
    assert tuple(out2["x2_hat"].shape[1:]) == (3, 256, 256)
    assert not any(p.requires_grad for p in stage2.quantizer.parameters())
    assert not any(p.requires_grad for p in stage2.combiner.parameters())
    assert not any(
        parameter.requires_grad
        for name, parameter in stage2.decoder.named_parameters()
        if not name.startswith("fuse_blocks.")
    )
    assert any(p.requires_grad for p in stage2.lq_encoder.parameters())
    assert any(p.requires_grad for p in stage2.decoder.fuse_blocks.parameters())
    if not any(p.grad is not None for p in trainable_parameters(stage2)):
        raise RuntimeError("Stage2 smoke had no LQ encoder/SFT gradient")
    print(
        "CodeFormer-v2 smoke passed: "
        f"stage1_hq_input={tuple(out1['hq_input'].shape)} q2={tuple(out1['q2'].shape)} "
        f"u2={tuple(out1['u2'].shape)} x2={tuple(out1['x2'].shape)} "
        f"stage2_x1={tuple(x1.shape)} x2_hat={tuple(out2['x2_hat'].shape)}",
        flush=True,
    )


@torch.no_grad()
def smoke_layer1(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    e1, d1 = build_layer1(args, device)
    checkpoint = load_layer1_checkpoint(args, e1, d1)
    freeze_layer1(e1, d1)
    image = torch.rand(args.smoke_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    x1 = layer1_x1(e1, d1, image)
    print(
        f"Layer1-v2 smoke passed: arch={args.layer1_arch} checkpoint={checkpoint} "
        f"x1={tuple(x1.shape)} frozen={not any(p.requires_grad for p in e1.parameters())}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage2")
    parser.add_argument("--version", default="cnn")
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--log-file", default="")
    parser.add_argument("--stage1-ckpt", default="MY-V2/jscc-f/codeformer-v2/checkpoints/jsccf_codeformer_v2_stage1_cnn_best.pth", help="Required for stage2.")
    parser.add_argument("--layer1-arch", choices=["cnn", "swin"], default="cnn")
    parser.add_argument("--layer1-ckpt", default="")
    parser.add_argument("--layer1-cnn-base-ch", type=int, default=16)
    parser.add_argument("--layer1-cnn-num-res", type=int, default=2)
    # Existing Layer1 Swin builder contract.
    parser.add_argument("--latent-ch", type=int, default=16)
    parser.add_argument("--c1-ch", type=int, default=16)
    parser.add_argument("--latent-h", type=int, default=16)
    parser.add_argument("--latent-w", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--fusion-weight", type=float, default=1.0)
    parser.add_argument("--lambda-vq", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-shapes", action="store_true")
    parser.add_argument("--smoke-layer1", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (args.latent_ch, args.c1_ch, args.latent_h, args.latent_w) != (16, 16, 16, 16):
        raise ValueError("Layer1 contract requires latent_ch=c1_ch=16 and 16x16 latent")
    if args.codebook_size < 2:
        raise ValueError("--codebook-size must be >= 2")
    if args.val_every < 1:
        raise ValueError("--val-every must be >= 1")


def main() -> None:
    args = parse_args(); validate_args(args); seed_everything(args.seed)
    handle, stdout, stderr = setup_log_file(args)
    try:
        if args.smoke_shapes:
            smoke_shapes(args)
        if args.smoke_layer1:
            smoke_layer1(args)
        if args.smoke_shapes or args.smoke_layer1:
            return
        if args.stage == "stage1":
            train_stage1(args)
        else:
            train_stage2(args)
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        handle.close()


if __name__ == "__main__":
    main()
