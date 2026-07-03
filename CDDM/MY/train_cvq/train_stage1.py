from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (
    AverageMeter,
    batch_metric_mean,
    check_args,
    format_metrics,
    print_epoch,
    print_run_header,
    psnr_per_image,
    real_awgn,
    recon_loss,
    resolve_path,
    sample_c2_nested_prefix_mask,
    seed_everything,
    setup_log_file,
    should_save_latest,
    should_validate,
    split_c1_c2,
    write_json,
)
from model import FullChannelQuantizer

from Autoencoder.data.datasets import get_loader


def load_local_io():
    spec = importlib.util.spec_from_file_location("cvq_v2_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cvq_io = load_local_io()


def default_stage0_best_ckpt() -> str:
    return str(Path(cvq_io.default_save_dir()) / "cvq_v2_c36_snr12_stage0_best.pth")


def default_stage1_warmup_ckpt() -> str:
    return str(Path(cvq_io.CDDM_ROOT) / "MY" / "checkpoints-cvq" / "cvq_c36_snr9_stage1_best.pth")


def stage_key(args: argparse.Namespace) -> str:
    return f"stage{int(getattr(args, 'stage', 1))}"


def default_latent_cache_path(args: argparse.Namespace) -> str:
    maps = int(args.latent_cache_maps)
    label = "262k" if maps == 262144 else str(maps)
    return str(Path(resolve_path(args.save_dir)) / f"cvq_c2_latent_cache_{label}.pt")


@torch.no_grad()
def collect_cvq_latent_cache(
    train_loader,
    encoder: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    target_maps = int(args.latent_cache_maps)
    if target_maps < int(args.k) * 10:
        raise ValueError(f"--latent-cache-maps must be at least 10*K={int(args.k) * 10}, got {target_maps}")
    cache_path = Path(resolve_path(args.latent_cache_path or default_latent_cache_path(args)))
    if bool(args.reuse_latent_cache) and cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.ndim != 3 or tuple(cache.shape[1:]) != (int(args.latent_h), int(args.latent_w)):
            raise RuntimeError(f"bad latent cache shape {tuple(cache.shape)} at {cache_path}")
        if cache.shape[0] < target_maps:
            raise RuntimeError(f"latent cache has {cache.shape[0]} maps, need {target_maps}: {cache_path}")
        print(f"loaded latent cache: {cache_path} shape={tuple(cache.shape)}")
        return cache[:target_maps].float()

    samples = []
    seen = 0
    passes = 0
    encoder.eval()
    print(f"collecting latent cache: target_maps={target_maps} path={cache_path}")
    while seen < target_maps:
        passes += 1
        for imgs, _labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
            _c1, c2 = split_c1_c2(z_norm, args)
            maps = c2.detach().float().cpu().reshape(-1, int(args.latent_h), int(args.latent_w))
            samples.append(maps)
            seen += int(maps.shape[0])
            if seen >= target_maps:
                break
        print(f"[latent-cache] pass={passes} seen={seen}/{target_maps}", flush=True)
    cache = torch.cat(samples, dim=0)[:target_maps].contiguous()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache.half(), cache_path)
    print(f"saved latent cache: {cache_path} shape={tuple(cache.shape)} dtype=float16")
    return cache


@torch.no_grad()
def kmeans_init_codebook_from_cache(
    cache: torch.Tensor,
    quantizer: FullChannelQuantizer,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    k = int(args.k)
    x = cache.float().reshape(cache.shape[0], -1).to(device)
    n, d = x.shape
    if n < k:
        raise ValueError(f"kmeans cache size {n} < K {k}")
    if d != int(args.latent_h) * int(args.latent_w):
        raise ValueError(f"kmeans vector dim {d} does not match latent map size")
    perm = torch.randperm(n, device=device)[:k]
    centers = x[perm].clone()
    counts = torch.zeros(k, device=device)
    batch_size = min(n, max(1, int(args.kmeans_batch_size)))
    assign_chunk = max(1, int(args.kmeans_assign_chunk))
    for it in range(int(args.kmeans_iters)):
        batch_idx = torch.randint(0, n, (batch_size,), device=device)
        batch = x[batch_idx]
        sums = torch.zeros_like(centers)
        batch_counts = torch.zeros(k, device=device)
        center_norm = centers.square().sum(dim=1).view(1, -1)
        for start in range(0, batch.shape[0], assign_chunk):
            q = batch[start : start + assign_chunk]
            dist = q.square().sum(dim=1, keepdim=True) + center_norm - 2.0 * q @ centers.t()
            idx = dist.argmin(dim=1)
            batch_counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
            sums.index_add_(0, idx, q)
        touched = batch_counts > 0
        counts[touched] = counts[touched] + batch_counts[touched]
        eta = (batch_counts[touched] / counts[touched]).unsqueeze(1)
        batch_means = sums[touched] / batch_counts[touched].unsqueeze(1)
        centers[touched] = centers[touched] * (1.0 - eta) + batch_means * eta
        dead = counts == 0
        min_count = float(counts[touched].min().item()) if bool(touched.any()) else 0.0
        max_count = float(counts.max().item()) if counts.numel() else 0.0
        print(
            f"[kmeans {it + 1:02d}/{int(args.kmeans_iters):02d}] "
            f"batch={batch_size} touched={int(touched.sum().item())} "
            f"dead={int(dead.sum().item())} min_count={min_count:.1f} max_count={max_count:.1f}",
            flush=True,
        )
    quantizer.codebook.data.copy_(
        centers.reshape(k, quantizer.h, quantizer.w).to(
            device=quantizer.codebook.device,
            dtype=quantizer.codebook.dtype,
        )
    )


@torch.no_grad()
def validate_kmeans_init(loader, encoder: nn.Module, decoder: nn.Module, quantizer: FullChannelQuantizer, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    quantizer.eval()
    meters = {k: AverageMeter() for k in ["quant_mse_init", "psnr_q_c2_init", "psnr_c1_only_init"]}
    device = next(encoder.parameters()).device
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        q_c2, _idx = quantizer.encode(c2)
        y_c1 = real_awgn(c1, float(args.snr_db))
        x_q_c2 = decoder(torch.cat([y_c1, q_c2], dim=1)).clamp(0.0, 1.0)
        x_c1_only = decoder(torch.cat([y_c1, torch.zeros_like(q_c2)], dim=1)).clamp(0.0, 1.0)
        bsz = imgs.shape[0]
        meters["quant_mse_init"].update(float(torch.mean((q_c2.float() - c2.float()).square()).item()), bsz)
        meters["psnr_q_c2_init"].update(batch_metric_mean(psnr_per_image(x_q_c2, imgs)), bsz)
        meters["psnr_c1_only_init"].update(batch_metric_mean(psnr_per_image(x_c1_only, imgs)), bsz)
    return {k: v.avg for k, v in meters.items()}


def init_codebook_with_cache_kmeans(
    train_loader,
    val_loader,
    encoder: nn.Module,
    decoder: nn.Module,
    quantizer: FullChannelQuantizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    if not bool(args.init_codebook_from_jscc):
        return {}
    cache = collect_cvq_latent_cache(train_loader, encoder, args, device)
    kmeans_device = torch.device(args.kmeans_device) if args.kmeans_device else device
    kmeans_init_codebook_from_cache(cache, quantizer, args, kmeans_device)
    metrics = validate_kmeans_init(val_loader, encoder, decoder, quantizer, args)
    key = stage_key(args)
    print(f"[{key} kmeans init] {format_metrics(metrics)}")
    write_json(Path(args.save_dir) / f"{key}_kmeans_init_metrics.json", metrics)
    cvq_io.save_checkpoint(
        cvq_io.ckpt_path(args, key, "kmeans_init"),
        stage=f"{key}_kmeans_init",
        epoch=0,
        args=args,
        metrics=metrics,
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
    )


def update_codebook_hist(hist: torch.Tensor, idx: torch.Tensor) -> None:
    hist += torch.bincount(idx.reshape(-1).detach().cpu(), minlength=hist.numel()).to(hist.dtype)


def codebook_usage_metrics(hist: torch.Tensor) -> dict[str, float]:
    total = hist.sum().clamp_min(1.0)
    prob = hist.float() / total
    nz = prob[prob > 0]
    used = int((hist > 0).sum().item())
    entropy = -(nz * nz.log()).sum() if nz.numel() else torch.zeros(())
    return {
        "code_used": float(used),
        "code_usage": used / float(hist.numel()),
        "code_perplexity": float(torch.exp(entropy).item()),
    }


def optimizer_stage1(encoder: nn.Module, decoder: nn.Module, quantizer: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    return optim.Adam(
        [
            {"params": encoder.parameters(), "lr": float(args.lr)},
            {"params": decoder.parameters(), "lr": float(args.lr)},
            {"params": quantizer.parameters(), "lr": float(args.codebook_lr)},
        ]
    )


@torch.no_grad()
def validate_stage1(loader, encoder: nn.Module, decoder: nn.Module, quantizer: FullChannelQuantizer, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    quantizer.eval()
    meters = {
        k: AverageMeter()
        for k in [
            "loss",
            "loss_c1_rec",
            "loss_drop_rec",
            "loss_full_rec",
            "vq",
            "psnr_c1_only",
            "psnr_drop",
            "psnr_full",
            "quant_mse",
        ]
    }
    device = next(encoder.parameters()).device
    code_hist = torch.zeros(int(args.k), dtype=torch.long)
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        q_c2, idx, vq_loss, q_raw = quantizer(c2)
        update_codebook_hist(code_hist, idx)
        y_c1 = real_awgn(c1, float(args.snr_db))
        mask = sample_c2_nested_prefix_mask(imgs.shape[0], q_c2.shape[1], float(args.nested_drop_ratio), imgs.device, q_c2.dtype)
        x_full = decoder(torch.cat([y_c1, q_c2], dim=1)).clamp(0.0, 1.0)
        x_nested_drop = decoder(torch.cat([y_c1, q_c2 * mask], dim=1)).clamp(0.0, 1.0)
        x_c1_only = decoder(torch.cat([y_c1, torch.zeros_like(q_c2)], dim=1)).clamp(0.0, 1.0)
        bsz = imgs.shape[0]
        loss_c1_rec = recon_loss(x_c1_only, imgs)
        loss_drop_rec = recon_loss(x_nested_drop, imgs)
        loss_full_rec = recon_loss(x_full, imgs)
        loss = (
            float(args.lambda_c1) * loss_c1_rec
            + float(args.lambda_drop) * loss_drop_rec
            + float(args.lambda_full) * loss_full_rec
            + float(args.lambda_vq) * vq_loss
        )
        meters["loss"].update(float(loss.item()), bsz)
        meters["loss_c1_rec"].update(float(loss_c1_rec.item()), bsz)
        meters["loss_drop_rec"].update(float(loss_drop_rec.item()), bsz)
        meters["loss_full_rec"].update(float(loss_full_rec.item()), bsz)
        meters["vq"].update(float(vq_loss.item()), bsz)
        meters["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1_only, imgs)), bsz)
        meters["psnr_drop"].update(batch_metric_mean(psnr_per_image(x_nested_drop, imgs)), bsz)
        meters["psnr_full"].update(batch_metric_mean(psnr_per_image(x_full, imgs)), bsz)
        meters["quant_mse"].update(float(torch.mean((q_raw.float() - c2.float()).square()).item()), bsz)
    metrics = {k: v.avg for k, v in meters.items()}
    metrics.update(codebook_usage_metrics(code_hist))
    return metrics


def train_stage1(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder, quantizer = cvq_io.build_models(args, cfg.device)
    if args.init_ckpt:
        ckpt_obj = cvq_io.load_experiment_checkpoint(args.init_ckpt, encoder=encoder, decoder=decoder, quantizer=quantizer, strict=True)
        if "quantizer_state_dict" not in ckpt_obj:
            print("init checkpoint has no quantizer_state_dict; initializing codebook with latent-cache KMeans")
            init_codebook_with_cache_kmeans(train_loader, val_loader, encoder, decoder, quantizer, args, cfg.device)
    else:
        if not args.init_jscc_encoder:
            args.init_jscc_encoder = cvq_io.default_jscc_encoder_c36_snr9()
        if not args.init_jscc_decoder:
            args.init_jscc_decoder = cvq_io.default_jscc_decoder_c36_snr9()
        cvq_io.load_module_checkpoint(encoder, args.init_jscc_encoder, "init JSCC encoder", strict=True)
        cvq_io.load_module_checkpoint(decoder, args.init_jscc_decoder, "init JSCC decoder", strict=True)
        init_codebook_with_cache_kmeans(train_loader, val_loader, encoder, decoder, quantizer, args, cfg.device)
    optimizer = optimizer_stage1(encoder, decoder, quantizer, args)
    best = -1.0
    key = stage_key(args)
    print_run_header(args, f"Stage {int(getattr(args, 'stage', 1))} | end-to-end JSCC + C2-only K16384 codebook", len(train_loader.dataset), len(val_loader.dataset))
    print("power_norm=all_36_channels_scaled_by_C1_mean_square cache=z_norm_C2_only no_per_channel_map_norm")
    print(f"c1_channel=real_awgn_after_power_norm snr_db={args.snr_db:g} c2_quantizer=shared_spatial_codebook K={int(args.k)}")
    metrics: dict[str, float] = {}
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train()
        decoder.train()
        quantizer.train()
        meters = {
            k: AverageMeter()
            for k in [
                "loss",
                "loss_c1_rec",
                "loss_drop_rec",
                "loss_full_rec",
                "vq",
                "psnr_c1_only",
                "psnr_drop",
                "psnr_full",
                "drop_keep",
            ]
        }
        code_hist = torch.zeros(int(args.k), dtype=torch.long)
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
            c1, c2 = split_c1_c2(z_norm, args)
            q_c2, idx, vq_loss, _q_raw = quantizer(c2)
            update_codebook_hist(code_hist, idx)
            y_c1 = real_awgn(c1, float(args.snr_db))
            mask = sample_c2_nested_prefix_mask(imgs.shape[0], q_c2.shape[1], float(args.nested_drop_ratio), imgs.device, q_c2.dtype)
            x_full = decoder(torch.cat([y_c1, q_c2], dim=1))
            x_nested_drop = decoder(torch.cat([y_c1, q_c2 * mask], dim=1))
            x_c1_only = decoder(torch.cat([y_c1, torch.zeros_like(q_c2)], dim=1))
            loss_c1_rec = recon_loss(x_c1_only, imgs)
            loss_drop_rec = recon_loss(x_nested_drop, imgs)
            loss_full_rec = recon_loss(x_full, imgs)
            loss = (
                float(args.lambda_c1) * loss_c1_rec
                + float(args.lambda_drop) * loss_drop_rec
                + float(args.lambda_full) * loss_full_rec
                + float(args.lambda_vq) * vq_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bsz = imgs.shape[0]
            meters["loss"].update(float(loss.item()), bsz)
            meters["loss_c1_rec"].update(float(loss_c1_rec.item()), bsz)
            meters["loss_drop_rec"].update(float(loss_drop_rec.item()), bsz)
            meters["loss_full_rec"].update(float(loss_full_rec.item()), bsz)
            meters["vq"].update(float(vq_loss.item()), bsz)
            meters["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1_only, imgs)), bsz)
            meters["psnr_drop"].update(batch_metric_mean(psnr_per_image(x_nested_drop, imgs)), bsz)
            meters["psnr_full"].update(batch_metric_mean(psnr_per_image(x_full, imgs)), bsz)
            meters["drop_keep"].update(float(mask.float().mean().item()), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        metrics.update(codebook_usage_metrics(code_hist))
        print_epoch(key, epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate_stage1(val_loader, encoder, decoder, quantizer, args)
            score = val_metrics["psnr_full"]
            print(f"[{key} val {epoch:03d}] {format_metrics(val_metrics)} score=psnr_full")
            if score > best:
                best = score
                cvq_io.save_checkpoint(cvq_io.ckpt_path(args, key, "best"), stage=key, epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)
        if should_save_latest(args, epoch):
            cvq_io.save_checkpoint(cvq_io.ckpt_path(args, key, "latest"), stage=key, epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)
    cvq_io.save_checkpoint(cvq_io.ckpt_path(args, key, "latest"), stage=key, epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=cvq_io.default_save_dir())
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default=default_stage1_warmup_ckpt())
    p.add_argument("--init-jscc-encoder", type=str, default="", help="Fallback JSCC encoder used only when --init-ckpt is empty.")
    p.add_argument("--init-jscc-decoder", type=str, default="", help="Fallback JSCC decoder used only when --init-ckpt is empty.")
    p.add_argument("--snr-db", type=float, default=9.0)
    p.add_argument("--latent-ch", type=int, default=36)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--k", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--codebook-lr", type=float, default=1e-4)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--vq-chunk-size", type=int, default=128)
    p.add_argument("--lambda-c1", type=float, default=0.5)
    p.add_argument("--lambda-drop", type=float, default=0.01)
    p.add_argument("--lambda-full", type=float, default=1.0)
    p.add_argument("--lambda-vq", type=float, default=1.0)
    p.add_argument(
        "--nested-drop-ratio",
        "--c2-dropout-prob",
        dest="nested_drop_ratio",
        type=float,
        default=1.0,
        help="Probability of applying C2 prefix nested dropout; otherwise full C2 is kept.",
    )
    p.add_argument("--init-codebook-from-jscc", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--latent-cache-maps", type=int, default=262144)
    p.add_argument("--latent-cache-path", type=str, default="")
    p.add_argument("--reuse-latent-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kmeans-iters", type=int, default=30)
    p.add_argument("--kmeans-batch-size", type=int, default=16384)
    p.add_argument("--kmeans-assign-chunk", type=int, default=4096)
    p.add_argument("--kmeans-device", type=str, default="")
    p.add_argument("--abort-bad-kmeans-init", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-init-quant-mse", type=float, default=0.15)
    p.add_argument("--min-init-psnr-q-full", type=float, default=20.0)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.stage = 1
    check_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(Path(args.save_dir) / f"stage1_c2shared_cvq_c36_snr{args.snr_db:g}_k{int(args.k)}.log")
    setup_log_file(args.log_file)
    write_json(Path(args.save_dir) / "stage1_args.json", vars(args))
    train_stage1(args)


if __name__ == "__main__":
    main()
