#!/usr/bin/env python3
"""Train the JSCC-f 256px CodeFormer adaptation.

``vq`` is CodeFormer stage I (HQ VQ autoencoder).  ``codeformer`` is stage II:
HQ images provide target code indices only during training, while the predictor
and all CFT/SFT conditions consume the frozen Layer1 reconstruction ``x1``.
"""

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

from architecture import (
    IMAGE_SIZE,
    LATENT_CHANNELS,
    LATENT_SIZE,
    CodeFormer,
    VQAutoencoder,
    trainable_parameters,
)
from layer1 import build_layer1, freeze_layer1, layer1_x1, load_layer1_checkpoint


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
CDDM_ROOT = JSCCF_DIR.parents[1]
DEFAULT_SAVE_DIR = THIS_DIR / "checkpoints"


def _load_jscc_io():
    if str(JSCCF_DIR) not in sys.path:
        sys.path.insert(0, str(JSCCF_DIR))
    spec = importlib.util.spec_from_file_location("jsccf_codeformer_train_io", JSCCF_DIR / "io.py")
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


def psnr(reconstruction: Tensor, target: Tensor) -> Tensor:
    mse = (reconstruction.float().clamp(0, 1) - target.float()).square().mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


class Meter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        return self.total / max(1, self.count)


class TeeStream:
    """Mirror stdout/stderr to one append-only experiment log."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class CodeUsage:
    """Dataset/epoch-level hard-code occupancy statistics.

    Per-batch perplexity is already emitted by the VQ module.  This accumulator
    instead combines every 16x16 token in an epoch before calculating usage, so
    a short or skewed final batch cannot dominate the reported code health.
    """

    def __init__(self, codebook_size: int) -> None:
        self.codebook_size = int(codebook_size)
        self.counts = torch.zeros(self.codebook_size, dtype=torch.float64)

    @torch.no_grad()
    def update(self, indices: Tensor) -> None:
        flat = indices.detach().long().reshape(-1)
        if flat.numel() == 0:
            return
        minimum = int(flat.min().item())
        maximum = int(flat.max().item())
        if minimum < 0 or maximum >= self.codebook_size:
            raise ValueError(
                f"code IDs must lie in [0,{self.codebook_size}), got [{minimum},{maximum}]"
            )
        self.counts += torch.bincount(flat, minlength=self.codebook_size).cpu().to(torch.float64)

    def metrics(self, prefix: str = "") -> dict[str, float]:
        key = lambda name: f"{prefix}{name}"
        total = float(self.counts.sum().item())
        if total <= 0.0:
            return {
                key("code_used"): 0.0,
                key("code_usage_ratio"): 0.0,
                key("code_entropy_bits"): 0.0,
                key("code_perplexity"): 0.0,
                key("code_top1_frac"): 0.0,
                key("code_tokens"): 0.0,
            }
        active = self.counts[self.counts > 0]
        probability = active / total
        entropy_bits = float(-(probability * probability.log2()).sum().item())
        return {
            key("code_used"): float(active.numel()),
            key("code_usage_ratio"): float(active.numel()) / float(self.codebook_size),
            key("code_entropy_bits"): entropy_bits,
            key("code_perplexity"): float(2.0**entropy_bits),
            key("code_top1_frac"): float(self.counts.max().item()) / total,
            key("code_tokens"): total,
        }


def setup_log_file(args: argparse.Namespace) -> tuple[object, object, object, Path]:
    """Install a Tee logger; the default filename is unique per stage/version."""
    if args.log_file:
        log_path = resolve(args.log_file)
    else:
        safe_version = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in str(args.version)
        ).strip("_") or "default"
        log_path = resolve(args.save_dir) / f"jsccf_codeformer_{args.stage}_{safe_version}.log"
        args.log_file = str(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout, stderr = sys.stdout, sys.stderr
    handle = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(stdout, handle)
    sys.stderr = TeeStream(stderr, handle)
    print(f"\n=== JSCC-f CodeFormer @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print(f"log file: {log_path}", flush=True)
    print(json.dumps(vars(args), sort_keys=True), flush=True)
    return handle, stdout, stderr, log_path


def update(meters: dict[str, Meter], name: str, value: Tensor | float, batch: int) -> None:
    value = float(value.detach().float().mean().item()) if torch.is_tensor(value) else float(value)
    meters[name].update(value, batch)


def summarize(meters: dict[str, Meter]) -> dict[str, float]:
    return {key: meter.average for key, meter in meters.items()}


def build_vq(args: argparse.Namespace) -> VQAutoencoder:
    return VQAutoencoder(
        codebook_size=int(args.codebook_size),
        base_channels=int(args.base_channels),
        num_res_blocks=int(args.num_res_blocks),
        beta=float(args.beta),
    )


def architecture_config(args: argparse.Namespace) -> dict[str, int | float | list[int]]:
    return {
        "image_size": IMAGE_SIZE,
        "latent_shape": [LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
        "downsample_stages": 4,
        "fuse_resolutions": [128, 64, 32],
        "removed_fuse_resolution": 256,
        "base_channels": int(args.base_channels),
        "num_res_blocks": int(args.num_res_blocks),
        "codebook_size": int(args.codebook_size),
        "beta": float(args.beta),
    }


def load_vq_checkpoint(vq: VQAutoencoder, path: str) -> dict:
    checkpoint_path = resolve(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"--vq-ckpt does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"VQ checkpoint must be a dict, got {type(payload)!r}")
    state = payload.get("vq_state_dict", payload.get("model_state_dict"))
    if state is None:
        raise KeyError(f"{checkpoint_path} does not contain vq_state_dict/model_state_dict")
    missing, unexpected = vq.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"VQ checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return payload


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"saved checkpoint: {path}", flush=True)


def checkpoint_path(args: argparse.Namespace, suffix: str) -> Path:
    name = f"jsccf_codeformer_{args.stage}_{args.version}_{suffix}.pth"
    return resolve(args.save_dir) / name


def make_loader(args: argparse.Namespace):
    config = jsccf_io.build_config(args)
    return get_loader(config), config.device


def validate_vq(model: VQAutoencoder, loader, device: torch.device, max_batches: int) -> dict[str, float]:
    model.eval()
    meters = {name: Meter() for name in ("loss", "recon", "psnr", "vq", "perplexity")}
    usage = CodeUsage(model.quantizer.codebook_size)
    with torch.no_grad():
        for step, (images, _labels) in enumerate(loader, start=1):
            if max_batches and step > max_batches:
                break
            images = images.to(device, non_blocking=True)
            out = model(images)
            reconstruction = F.mse_loss(out["reconstruction"], images)
            loss = reconstruction + out["vq_loss"]
            batch = images.shape[0]
            update(meters, "loss", loss, batch)
            update(meters, "recon", reconstruction, batch)
            update(meters, "psnr", psnr(out["reconstruction"], images), batch)
            update(meters, "vq", out["vq_loss"], batch)
            update(meters, "perplexity", out["perplexity"], batch)
            usage.update(out["indices"])
    metrics = summarize(meters)
    metrics.update(usage.metrics())
    return metrics


def train_vq(args: argparse.Namespace) -> None:
    (train_loader, val_loader), device = make_loader(args)
    model = build_vq(args).to(device)
    optimizer = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_psnr = float("-inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        meters = {name: Meter() for name in ("loss", "recon", "psnr", "vq", "perplexity")}
        usage = CodeUsage(model.quantizer.codebook_size)
        for step, (images, _labels) in enumerate(train_loader, start=1):
            if int(args.max_train_batches) and step > int(args.max_train_batches):
                break
            images = images.to(device, non_blocking=True)
            out = model(images)
            reconstruction = F.mse_loss(out["reconstruction"], images)
            loss = reconstruction + float(args.lambda_vq) * out["vq_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
            optimizer.step()
            batch = images.shape[0]
            update(meters, "loss", loss, batch)
            update(meters, "recon", reconstruction, batch)
            update(meters, "psnr", psnr(out["reconstruction"], images), batch)
            update(meters, "vq", out["vq_loss"], batch)
            update(meters, "perplexity", out["perplexity"], batch)
            usage.update(out["indices"])
        train_metrics = summarize(meters)
        train_metrics.update(usage.metrics())
        print(f"[vq {epoch:03d}/{args.epochs}] train {json.dumps(train_metrics, sort_keys=True)}", flush=True)
        if epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            metrics = validate_vq(model, val_loader, device, int(args.max_val_batches))
            print(f"[vq {epoch:03d}/{args.epochs}] valid {json.dumps(metrics, sort_keys=True)}", flush=True)
            payload = {
                "format": "jsccf_codeformer_vq_v1",
                "stage": "vq",
                "epoch": epoch,
                "args": vars(args),
                "architecture": architecture_config(args),
                "metrics": metrics,
                "vq_state_dict": model.state_dict(),
            }
            save_checkpoint(checkpoint_path(args, "latest"), payload)
            if metrics["psnr"] >= best_psnr:
                best_psnr = metrics["psnr"]
                save_checkpoint(checkpoint_path(args, "best"), payload)


def _target_indices(vq: VQAutoencoder, hq: Tensor) -> Tensor:
    # HQ is supervision only.  CodeFormer.restore itself never accepts it.
    with torch.no_grad():
        _straight, _quantized, indices, _stats = vq.encode(hq)
    return indices


def codeformer_metrics(
    out: dict[str, Tensor], target: Tensor, x1: Tensor, hq: Tensor, ce: Tensor, meters: dict[str, Meter]
) -> None:
    batch = hq.shape[0]
    update(meters, "ce", ce, batch)
    update(meters, "psnr_x1", psnr(x1, hq), batch)
    update(meters, "psnr_final", psnr(out["output"], hq), batch)
    update(meters, "delta_x1", psnr(out["output"], hq).mean() - psnr(x1, hq).mean(), batch)
    update(meters, "index_acc", (out["indices"] == target).float(), batch)


def validate_codeformer(
    model: CodeFormer,
    layer1_encoder: nn.Module,
    layer1_decoder: nn.Module,
    loader,
    device: torch.device,
    fusion_weight: float,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    layer1_encoder.eval()
    layer1_decoder.eval()
    meters = {name: Meter() for name in ("ce", "rec", "psnr_x1", "psnr_final", "delta_x1", "index_acc")}
    predicted_usage = CodeUsage(model.vq.quantizer.codebook_size)
    target_usage = CodeUsage(model.vq.quantizer.codebook_size)
    with torch.no_grad():
        for step, (hq, _labels) in enumerate(loader, start=1):
            if max_batches and step > max_batches:
                break
            hq = hq.to(device, non_blocking=True)
            x1 = layer1_x1(layer1_encoder, layer1_decoder, hq)
            target = _target_indices(model.vq, hq)
            out = model.restore(x1, fusion_weight=fusion_weight, soft_decode=False)
            ce = F.cross_entropy(out["logits"].flatten(0, 1), target.flatten())
            codeformer_metrics(out, target, x1, hq, ce, meters)
            update(meters, "rec", F.mse_loss(out["output"], hq), hq.shape[0])
            predicted_usage.update(out["indices"])
            target_usage.update(target)
    metrics = summarize(meters)
    metrics.update(predicted_usage.metrics("pred_"))
    metrics.update(target_usage.metrics("target_"))
    return metrics


def train_codeformer(args: argparse.Namespace) -> None:
    if not args.vq_ckpt:
        raise ValueError("--stage codeformer requires a trained --vq-ckpt from --stage vq")
    (train_loader, val_loader), device = make_loader(args)
    vq = build_vq(args)
    vq_payload = load_vq_checkpoint(vq, args.vq_ckpt)
    model = CodeFormer(
        vq,
        transformer_width=int(args.transformer_width),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        transformer_mlp_ratio=float(args.transformer_mlp_ratio),
    ).to(device)
    model.freeze_codebook_and_generator(bool(args.freeze_vq_encoder))

    layer1_encoder, layer1_decoder = build_layer1(args, device)
    layer1_path = load_layer1_checkpoint(args, layer1_encoder, layer1_decoder)
    freeze_layer1(layer1_encoder, layer1_decoder)
    optimizer = AdamW(
        trainable_parameters([model]),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"frozen Layer1: {layer1_path}; CodeFormer trainable parameters: {trainable:,}", flush=True)

    best_psnr = float("-inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        layer1_encoder.eval()
        layer1_decoder.eval()
        meters = {name: Meter() for name in ("loss", "ce", "rec", "psnr_x1", "psnr_final", "delta_x1", "index_acc")}
        predicted_usage = CodeUsage(model.vq.quantizer.codebook_size)
        target_usage = CodeUsage(model.vq.quantizer.codebook_size)
        for step, (hq, _labels) in enumerate(train_loader, start=1):
            if int(args.max_train_batches) and step > int(args.max_train_batches):
                break
            hq = hq.to(device, non_blocking=True)
            x1 = layer1_x1(layer1_encoder, layer1_decoder, hq)
            target = _target_indices(model.vq, hq)
            # Soft codebook decoding makes reconstruction/CFT losses differentiable
            # through the code logits; validation uses hard argmax deployment codes.
            out = model.restore(x1, fusion_weight=float(args.fusion_weight), soft_decode=True)
            ce = F.cross_entropy(out["logits"].flatten(0, 1), target.flatten())
            reconstruction = F.mse_loss(out["output"], hq)
            loss = float(args.lambda_ce) * ce + float(args.lambda_rec) * reconstruction
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0:
                nn.utils.clip_grad_norm_(trainable_parameters([model]), float(args.grad_clip_norm))
            optimizer.step()
            batch = hq.shape[0]
            update(meters, "loss", loss, batch)
            codeformer_metrics(out, target, x1, hq, ce, meters)
            update(meters, "rec", reconstruction, batch)
            predicted_usage.update(out["indices"])
            target_usage.update(target)
        train_metrics = summarize(meters)
        train_metrics.update(predicted_usage.metrics("pred_"))
        train_metrics.update(target_usage.metrics("target_"))
        print(f"[codeformer {epoch:03d}/{args.epochs}] train {json.dumps(train_metrics, sort_keys=True)}", flush=True)
        if epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            metrics = validate_codeformer(
                model, layer1_encoder, layer1_decoder, val_loader, device,
                float(args.fusion_weight), int(args.max_val_batches),
            )
            print(f"[codeformer {epoch:03d}/{args.epochs}] valid {json.dumps(metrics, sort_keys=True)}", flush=True)
            payload = {
                "format": "jsccf_codeformer_stage2_v1",
                "stage": "codeformer",
                "epoch": epoch,
                "args": vars(args),
                "architecture": {
                    **architecture_config(args),
                    "transformer_width": int(args.transformer_width),
                    "transformer_layers": int(args.transformer_layers),
                    "transformer_heads": int(args.transformer_heads),
                },
                "metrics": metrics,
                "layer1_arch": args.layer1_arch,
                "layer1_checkpoint": str(layer1_path),
                "vq_checkpoint": str(resolve(args.vq_ckpt)),
                "vq_checkpoint_epoch": int(vq_payload.get("epoch", -1)),
                "model_state_dict": model.state_dict(),
            }
            save_checkpoint(checkpoint_path(args, "latest"), payload)
            if metrics["psnr_final"] >= best_psnr:
                best_psnr = metrics["psnr_final"]
                save_checkpoint(checkpoint_path(args, "best"), payload)


def smoke_shapes(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    vq = build_vq(args).to(device).eval()
    image = torch.rand(int(args.smoke_batch_size), 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        stage1 = vq(image)
    assert tuple(stage1["quantized"].shape[1:]) == (256, 16, 16)
    assert tuple(stage1["indices"].shape[1:]) == (16, 16)
    codeformer = CodeFormer(
        vq,
        transformer_width=int(args.transformer_width),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        transformer_mlp_ratio=float(args.transformer_mlp_ratio),
    ).to(device).eval()
    with torch.no_grad():
        result = codeformer.restore(image, fusion_weight=1.0, soft_decode=False)
    assert tuple(result["output"].shape[1:]) == (3, 256, 256)
    assert tuple(result["logits"].shape[1:]) == (256, int(args.codebook_size))
    # Exercise the differentiable stage-II route, too: the hard deployment
    # path above cannot prove gradients reach the Transformer/SFT parameters.
    codeformer.train()
    codeformer.freeze_codebook_and_generator(freeze_encoder=False)
    with torch.no_grad():
        target = vq.encode(image)[2]
    soft = codeformer.restore(image, fusion_weight=1.0, soft_decode=True)
    smoke_loss = F.cross_entropy(soft["logits"].flatten(0, 1), target.flatten()) + F.mse_loss(soft["output"], image)
    smoke_loss.backward()
    if not any(parameter.grad is not None for parameter in codeformer.parameters() if parameter.requires_grad):
        raise RuntimeError("stage-II smoke had no trainable gradient")
    vq_usage = CodeUsage(vq.quantizer.codebook_size)
    vq_usage.update(stage1["indices"])
    predicted_usage = CodeUsage(vq.quantizer.codebook_size)
    predicted_usage.update(result["indices"])
    print(
        "shape smoke passed: "
        f"x1={tuple(image.shape)} latent={tuple(stage1['quantized'].shape)} "
        f"tokens={tuple(result['logits'].shape)} output={tuple(result['output'].shape)} "
        "fuse_resolutions=[128,64,32] (256 removed); soft stage-II backward passed; "
        f"vq_usage={vq_usage.metrics()} pred_usage={predicted_usage.metrics('pred_')}",
        flush=True,
    )


@torch.no_grad()
def smoke_layer1(args: argparse.Namespace) -> None:
    """Strict-load a selectable existing Layer1 checkpoint and produce x1."""
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    encoder, decoder = build_layer1(args, device)
    checkpoint = load_layer1_checkpoint(args, encoder, decoder)
    freeze_layer1(encoder, decoder)
    image = torch.rand(int(args.smoke_batch_size), 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    x1 = layer1_x1(encoder, decoder, image)
    print(
        f"Layer1 smoke passed: arch={args.layer1_arch} checkpoint={checkpoint} "
        f"x1={tuple(x1.shape)} frozen={not any(p.requires_grad for p in encoder.parameters())}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=["vq", "codeformer"], default="codeformer")
    parser.add_argument("--version", default="swin-vq")
    parser.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--log-file", default="", help="Append stdout/stderr here; empty auto-saves under --save-dir.")
    parser.add_argument("--vq-ckpt", default="MY-V2/jscc-f/codeformer/checkpoints/jsccf_codeformer_vq_v1_best.pth", help="Required frozen HQ VQ checkpoint for stage=codeformer.")
    parser.add_argument("--layer1-arch", choices=["cnn", "swin"], default="cnn")
    parser.add_argument("--layer1-ckpt", default="", help="Empty selects the matching JSCC-f Layer1 default.")
    parser.add_argument("--layer1-cnn-base-ch", type=int, default=16)
    parser.add_argument("--layer1-cnn-num-res", type=int, default=2)
    # Existing Swin builder contract.
    parser.add_argument("--latent-ch", type=int, default=16)
    parser.add_argument("--c1-ch", type=int, default=16)
    parser.add_argument("--latent-h", type=int, default=16)
    parser.add_argument("--latent-w", type=int, default=16)

    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--transformer-width", type=int, default=512)
    parser.add_argument("--transformer-layers", type=int, default=9)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-mlp-ratio", type=float, default=2.0)
    parser.add_argument("--freeze-vq-encoder", action="store_true")
    parser.add_argument("--fusion-weight", type=float, default=1.0)
    parser.add_argument("--lambda-vq", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--lambda-rec", type=float, default=1.0)

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
    parser.add_argument("--smoke-layer1", action="store_true", help="Strict-load the chosen existing Layer1 checkpoint and check x1.")
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (int(args.latent_ch), int(args.c1_ch), int(args.latent_h), int(args.latent_w)) != (16, 16, 16, 16):
        raise ValueError("Layer1 contract must stay latent_ch=c1_ch=16 and latent_h=latent_w=16")
    if int(args.codebook_size) < 2:
        raise ValueError("--codebook-size must be >= 2")
    if int(args.transformer_width) % int(args.transformer_heads):
        raise ValueError("--transformer-width must be divisible by --transformer-heads")
    if int(args.val_every) < 1:
        raise ValueError("--val-every must be >= 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(int(args.seed))
    log_handle, stdout, stderr, _log_path = setup_log_file(args)
    try:
        if args.smoke_shapes:
            smoke_shapes(args)
        if args.smoke_layer1:
            smoke_layer1(args)
        if args.smoke_shapes or args.smoke_layer1:
            return
        if args.stage == "vq":
            train_vq(args)
        else:
            train_codeformer(args)
    finally:
        sys.stdout = stdout
        sys.stderr = stderr
        log_handle.close()


if __name__ == "__main__":
    main()
