from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    recon_loss,
    real_awgn,
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


def default_latent_cache_path(args: argparse.Namespace) -> str:
    maps = int(args.latent_cache_maps)
    label = "262k" if maps == 262144 else str(maps)
    mode = str(getattr(args, "stage1_vq_mode", "plain")).lower()
    suffix = "" if mode == "plain" else f"_{mode}"
    return str(Path(resolve_path(args.save_dir)) / f"cvq_stage1_c2_latent_cache_{label}{suffix}.pt")


def quantizer_cache_mode(quantizer: nn.Module) -> str:
    return str(getattr(quantizer, "cache_mode", "plain"))


@torch.no_grad()
def fit_scaled_vq_stats(
    train_loader,
    encoder: nn.Module,
    quantizer: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    if not hasattr(quantizer, "set_channel_stats"):
        return {}
    max_batches = int(args.scaled_vq_stat_batches)
    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    sums = torch.zeros(c2_ch, dtype=torch.float64)
    sq_sums = torch.zeros(c2_ch, dtype=torch.float64)
    count = 0
    batches = 0
    encoder.eval()
    for imgs, _labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
        _z_c1, z_c2 = split_c1_c2(z_norm, args)
        x = z_c2.detach().double().cpu()
        sums += x.sum(dim=(0, 2, 3))
        sq_sums += x.square().sum(dim=(0, 2, 3))
        count += int(x.shape[0] * x.shape[2] * x.shape[3])
        batches += 1
        if max_batches > 0 and batches >= max_batches:
            break
    if count <= 0:
        raise RuntimeError("cannot fit scaled VQ stats from an empty train loader")
    mean = sums / float(count)
    var = (sq_sums / float(count) - mean.square()).clamp_min(float(args.scaled_vq_eps) ** 2)
    quantizer.set_channel_stats(mean.float().to(device), var.float().to(device))
    sigma = torch.sqrt(var)
    stats = {
        "scaled_vq_stat_batches": float(batches),
        "scaled_vq_stat_count_per_channel": float(count),
        "scaled_vq_mean_abs_avg": float(mean.abs().mean().item()),
        "scaled_vq_sigma_avg": float(sigma.mean().item()),
        "scaled_vq_sigma_min": float(sigma.min().item()),
        "scaled_vq_sigma_max": float(sigma.max().item()),
    }
    print(f"[scaled-vq stats] {format_metrics(stats)}")
    return stats


@torch.no_grad()
def collect_cvq_latent_cache(
    train_loader,
    encoder: nn.Module,
    quantizer: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    target_maps = int(args.latent_cache_maps)
    if target_maps < int(args.k) * 10:
        raise ValueError(f"--latent-cache-maps must be at least 10*K={int(args.k) * 10}, got {target_maps}")
    cache_path = Path(resolve_path(args.latent_cache_path or default_latent_cache_path(args)))
    cache_mode = quantizer_cache_mode(quantizer)
    if bool(args.reuse_latent_cache) and cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            saved_mode = str(obj.get("mode", "plain"))
            if saved_mode != cache_mode:
                raise RuntimeError(f"latent cache mode mismatch at {cache_path}: saved={saved_mode} current={cache_mode}")
            cache = obj["samples"]
            if cache_mode != "plain" and hasattr(quantizer, "set_channel_stats") and "running_mean" in obj and "running_var" in obj:
                quantizer.set_channel_stats(obj["running_mean"].to(device), obj["running_var"].to(device))
        else:
            if cache_mode != "plain":
                raise RuntimeError(
                    f"latent cache at {cache_path} is a legacy raw tensor; use a scaled cache path or pass --no-reuse-latent-cache"
                )
            cache = obj
        if cache.ndim != 3 or tuple(cache.shape[1:]) != (int(args.latent_h), int(args.latent_w)):
            raise RuntimeError(f"bad latent cache shape {tuple(cache.shape)} at {cache_path}")
        if cache.shape[0] < target_maps:
            raise RuntimeError(f"latent cache has {cache.shape[0]} maps, need {target_maps}: {cache_path}")
        print(f"loaded latent cache: {cache_path} mode={cache_mode} shape={tuple(cache.shape)}")
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
            _z_c1, z_c2 = split_c1_c2(z_norm, args)
            if hasattr(quantizer, "extract_codebook_samples"):
                maps = quantizer.extract_codebook_samples(z_c2).detach().float().cpu()
            else:
                maps = z_c2.detach().float().cpu().reshape(-1, int(args.latent_h), int(args.latent_w))
            samples.append(maps)
            seen += int(maps.shape[0])
            if seen >= target_maps:
                break
        print(f"[latent-cache] pass={passes} seen={seen}/{target_maps}", flush=True)
    cache = torch.cat(samples, dim=0)[:target_maps].contiguous()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_mode == "plain":
        payload = cache.half()
    else:
        payload = {
            "mode": cache_mode,
            "samples": cache.half(),
            "running_mean": quantizer.running_mean.detach().cpu(),
            "running_var": quantizer.running_var.detach().cpu(),
        }
    torch.save(payload, cache_path)
    print(f"saved latent cache: {cache_path} mode={cache_mode} shape={tuple(cache.shape)} dtype=float16")
    return cache


@torch.no_grad()
def kmeans_init_codebook_from_cache(
    cache: torch.Tensor,
    quantizer: FullChannelQuantizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    k = int(args.k)
    x = cache.float().reshape(cache.shape[0], -1).to(device)
    n, d = x.shape
    if n < k:
        raise ValueError(f"kmeans cache size {n} < K {k}")
    if d != int(args.latent_h) * int(args.latent_w):
        raise ValueError(f"kmeans vector dim {d} does not match latent map size")
    perm = torch.randperm(n, device=device)[:k]
    centers = x[perm].clone()
    assign_chunk = max(1, int(args.kmeans_assign_chunk))
    usage_stats: dict[str, float] = {}
    for it in range(int(args.kmeans_iters)):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device)
        center_norm = centers.square().sum(dim=1).view(1, -1)
        for start in range(0, n, assign_chunk):
            q = x[start : start + assign_chunk]
            dist = q.square().sum(dim=1, keepdim=True) + center_norm - 2.0 * q @ centers.t()
            idx = dist.argmin(dim=1)
            counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
            sums.index_add_(0, idx, q)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        empty = ~nonempty
        if bool(empty.any()):
            num_empty = int(empty.sum().item())
            refill = x[torch.randint(0, n, (num_empty,), device=device)]
            centers[empty] = refill
        min_count = float(counts[nonempty].min().item()) if bool(nonempty.any()) else 0.0
        max_count = float(counts.max().item()) if counts.numel() else 0.0
        mean_count = float(counts.float().mean().item()) if counts.numel() else 0.0
        probs = counts / counts.sum().clamp_min(1.0)
        nz = probs[probs > 0]
        entropy = -(nz * torch.log2(nz)).sum() if nz.numel() else torch.zeros((), device=device)
        perplexity = torch.pow(torch.tensor(2.0, device=device), entropy)
        active_count = int(nonempty.sum().item())
        count1 = int((counts == 1).sum().item())
        counts_f = counts.float()
        p50 = float(torch.quantile(counts_f, 0.50).item())
        p90 = float(torch.quantile(counts_f, 0.90).item())
        p99 = float(torch.quantile(counts_f, 0.99).item())
        max_over_mean = max_count / max(mean_count, 1e-12)
        usage_stats = {
            "kmeans_active_count": float(active_count),
            "kmeans_perplexity": float(perplexity.item()),
            "kmeans_usage_entropy_bits": float(entropy.item()),
            "kmeans_count1": float(count1),
            "kmeans_count_p50": p50,
            "kmeans_count_p90": p90,
            "kmeans_count_p99": p99,
            "kmeans_max_count": max_count,
            "kmeans_mean_count": mean_count,
            "kmeans_max_over_mean": float(max_over_mean),
        }
        print(
            f"[kmeans {it + 1:02d}/{int(args.kmeans_iters):02d}] "
            f"empty={int(empty.sum().item())} active={active_count} "
            f"min_count={min_count:.1f} max_count={max_count:.1f} "
            f"perplexity={float(perplexity.item()):.1f} "
            f"count1={count1} "
            f"p50={p50:.1f} p90={p90:.1f} p99={p99:.1f} "
            f"max/mean={max_over_mean:.1f}",
            flush=True,
        )
    quantizer.codebook.data.copy_(
        centers.reshape(k, quantizer.h, quantizer.w).to(
            device=quantizer.codebook.device,
            dtype=quantizer.codebook.dtype,
        )
    )
    sync_codebook_ema(quantizer)
    return usage_stats


def codebook_usage_from_hist(hist: torch.Tensor, k: int, prefix: str = "codebook") -> dict[str, float]:
    hist = hist.detach().float().cpu()
    total = hist.sum().clamp_min(1.0)
    prob = hist / total
    nz = prob[prob > 0]
    entropy_bits = -(nz * nz.log2()).sum() if nz.numel() else torch.zeros(())
    perplexity = torch.pow(torch.tensor(2.0), entropy_bits)
    used = int((hist > 0).sum().item())
    sorted_hist = torch.sort(hist, descending=True).values
    return {
        f"{prefix}_used": float(used),
        f"{prefix}_usage": float(used / max(1, int(k))),
        f"{prefix}_perplexity": float(perplexity.item()),
        f"{prefix}_entropy_bits": float(entropy_bits.item()),
        f"{prefix}_top1_ratio": float(sorted_hist[:1].sum().item() / float(total.item())),
        f"{prefix}_top10_ratio": float(sorted_hist[:10].sum().item() / float(total.item())),
    }


def usage_regularizer_enabled(args: argparse.Namespace) -> bool:
    return float(args.lambda_usage_entropy) != 0.0 or float(args.lambda_usage_topk) != 0.0 or bool(args.usage_reg_eval)


def usage_regularizer_input(z_c2: torch.Tensor, quantizer: nn.Module) -> torch.Tensor:
    if hasattr(quantizer, "normalize"):
        return quantizer.normalize(z_c2)
    return z_c2


def soft_codebook_usage_regularizer(
    z_c2: torch.Tensor,
    quantizer: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    x = usage_regularizer_input(z_c2, quantizer)
    bsz, channels, h, w = x.shape
    flat = x.reshape(bsz * channels, h * w).float()
    if hasattr(quantizer, "effective_codebook"):
        codebook = quantizer.effective_codebook().float().flatten(1)
    else:
        codebook = quantizer.codebook.float().flatten(1)
    if bool(args.usage_reg_detach_codebook):
        codebook = codebook.detach()
    k = int(codebook.shape[0])
    chunk = max(1, int(args.usage_reg_chunk_size))
    tau = max(float(args.usage_reg_tau), float(args.usage_reg_eps))
    eps = float(args.usage_reg_eps)
    cb_norm = codebook.square().sum(dim=1).view(1, -1)
    prob_sum = None
    for start in range(0, flat.shape[0], chunk):
        q = flat[start : start + chunk]
        dist = q.square().sum(dim=1, keepdim=True) + cb_norm - 2.0 * q @ codebook.t()
        probs = torch.softmax(-dist / tau, dim=1)
        chunk_sum = probs.sum(dim=0)
        prob_sum = chunk_sum if prob_sum is None else prob_sum + chunk_sum
    avg_prob = prob_sum / float(max(1, flat.shape[0]))
    entropy = -(avg_prob * torch.log(avg_prob.clamp_min(eps))).sum()
    log_k = torch.log(torch.tensor(float(k), device=avg_prob.device, dtype=avg_prob.dtype))
    entropy_norm = entropy / log_k.clamp_min(eps)
    top = torch.sort(avg_prob, descending=True).values
    top1_ratio = top[:1].sum()
    top10_ratio = top[: min(10, k)].sum()
    loss_usage_entropy = (1.0 - entropy_norm).clamp_min(0.0)
    loss_usage_topk = F.relu(top1_ratio - float(args.usage_reg_top1_target)) + F.relu(
        top10_ratio - float(args.usage_reg_top10_target)
    )
    loss_usage = float(args.lambda_usage_entropy) * loss_usage_entropy + float(args.lambda_usage_topk) * loss_usage_topk
    entropy_bits = entropy / torch.log(torch.tensor(2.0, device=avg_prob.device, dtype=avg_prob.dtype))
    metrics = {
        "loss_usage": float(loss_usage.detach().item()),
        "loss_usage_entropy": float(loss_usage_entropy.detach().item()),
        "loss_usage_topk": float(loss_usage_topk.detach().item()),
        "soft_codebook_entropy_bits": float(entropy_bits.detach().item()),
        "soft_codebook_entropy_norm": float(entropy_norm.detach().item()),
        "soft_codebook_perplexity": float(torch.exp(entropy).detach().item()),
        "soft_codebook_top1_ratio": float(top1_ratio.detach().item()),
        "soft_codebook_top10_ratio": float(top10_ratio.detach().item()),
    }
    return loss_usage, metrics


def update_codebook_hist(hist: torch.Tensor, idx: torch.Tensor, k: int) -> None:
    counts = torch.bincount(idx.reshape(-1).detach().cpu(), minlength=int(k)).to(dtype=hist.dtype)
    hist += counts


@torch.no_grad()
def sync_codebook_ema(quantizer: nn.Module) -> None:
    if hasattr(quantizer, "sync_ema_from_codebook"):
        quantizer.sync_ema_from_codebook()


@torch.no_grad()
def validate_kmeans_init(loader, encoder: nn.Module, decoder: nn.Module, quantizer: FullChannelQuantizer, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    quantizer.eval()
    meters = {k: AverageMeter() for k in ["quant_mse_init", "psnr_c1_only_init", "psnr_q_full_init", "psnr_mix_init"]}
    device = next(encoder.parameters()).device
    codebook_hist = torch.zeros(int(args.k), dtype=torch.float64)
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
        c1_cont, c2 = split_c1_c2(z_norm, args)
        q_c2, idx = quantizer.encode(c2)
        update_codebook_hist(codebook_hist, idx, int(args.k))
        x_c1_only = decoder(torch.cat([c1_cont, torch.zeros_like(q_c2)], dim=1)).clamp(0.0, 1.0)
        x_mix = decoder(torch.cat([c1_cont, q_c2], dim=1)).clamp(0.0, 1.0)
        bsz = imgs.shape[0]
        meters["quant_mse_init"].update(float(torch.mean((q_c2.float() - c2.float()).square()).item()), bsz)
        meters["psnr_c1_only_init"].update(batch_metric_mean(psnr_per_image(x_c1_only, imgs)), bsz)
        meters["psnr_q_full_init"].update(batch_metric_mean(psnr_per_image(x_mix, imgs)), bsz)
        meters["psnr_mix_init"].update(batch_metric_mean(psnr_per_image(x_mix, imgs)), bsz)
    metrics = {k: v.avg for k, v in meters.items()}
    metrics.update(codebook_usage_from_hist(codebook_hist, int(args.k)))
    return metrics


def init_codebook_with_cache_kmeans(
    train_loader,
    val_loader,
    encoder: nn.Module,
    decoder: nn.Module,
    quantizer: FullChannelQuantizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    cache = collect_cvq_latent_cache(train_loader, encoder, quantizer, args, device)
    kmeans_device = torch.device(args.kmeans_device) if args.kmeans_device else device
    kmeans_stats = kmeans_init_codebook_from_cache(cache, quantizer, args, kmeans_device)
    metrics = validate_kmeans_init(val_loader, encoder, decoder, quantizer, args)
    metrics.update(kmeans_stats)
    print(f"[stage1 kmeans init] {format_metrics(metrics)}")
    write_json(Path(args.save_dir) / "stage1_kmeans_init_metrics.json", metrics)
    cvq_io.save_checkpoint(
        cvq_io.ckpt_path(args, "stage1", "kmeans_init"),
        stage="stage1_kmeans_init",
        epoch=0,
        args=args,
        metrics=metrics,
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
    )
    # if bool(args.abort_bad_kmeans_init):
    #     bad = metrics["quant_mse_init"] > float(args.max_init_quant_mse) or metrics["psnr_q_full_init"] < float(args.min_init_psnr_q_full)
    #     if bad:
    #         raise RuntimeError(
    #             "bad KMeans initialization: "
    #             f"quant_mse_init={metrics['quant_mse_init']:.6g} "
    #             f"psnr_q_full_init={metrics['psnr_q_full_init']:.6g}; "
    #             f"thresholds quant_mse<={float(args.max_init_quant_mse):g}, "
    #             f"psnr_q_full>={float(args.min_init_psnr_q_full):g}"
    #         )
    return metrics


@torch.no_grad()
def init_codebook_with_random_samples(
    train_loader,
    val_loader,
    encoder: nn.Module,
    decoder: nn.Module,
    quantizer: FullChannelQuantizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    cache = collect_cvq_latent_cache(train_loader, encoder, quantizer, args, device)
    quantizer.init_from_samples(cache)
    metrics = validate_kmeans_init(val_loader, encoder, decoder, quantizer, args)
    metrics["init_codebook_samples"] = float(cache.shape[0])
    print(f"[stage1 random_samples init] {format_metrics(metrics)}")
    write_json(Path(args.save_dir) / "stage1_random_samples_init_metrics.json", metrics)
    cvq_io.save_checkpoint(
        cvq_io.ckpt_path(args, "stage1", "random_samples_init"),
        stage="stage1_random_samples_init",
        epoch=0,
        args=args,
        metrics=metrics,
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
    )
    return metrics


@torch.no_grad()
def init_codebook_with_random_normal(
    val_loader,
    encoder: nn.Module,
    decoder: nn.Module,
    quantizer: FullChannelQuantizer,
    args: argparse.Namespace,
) -> dict[str, float]:
    if quantizer_cache_mode(quantizer) == "scaled_whitened":
        std = float(args.scaled_vq_random_codebook_std)
    elif hasattr(quantizer, "default_codebook_std"):
        std = float(args.simvq_codebook_std) if float(args.simvq_codebook_std) > 0.0 else float(quantizer.default_codebook_std)
    else:
        std = float(args.random_codebook_std)
    quantizer.codebook.normal_(mean=0.0, std=std)
    sync_codebook_ema(quantizer)
    metrics = validate_kmeans_init(val_loader, encoder, decoder, quantizer, args)
    metrics["random_codebook_std"] = std
    print(f"[stage1 random_normal init] {format_metrics(metrics)}")
    write_json(Path(args.save_dir) / "stage1_random_normal_init_metrics.json", metrics)
    cvq_io.save_checkpoint(
        cvq_io.ckpt_path(args, "stage1", "random_normal_init-v2"),
        stage="stage1_random_normal_init",
        epoch=0,
        args=args,
        metrics=metrics,
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
    )
    return metrics


def init_codebook(
    train_loader,
    val_loader,
    encoder: nn.Module,
    decoder: nn.Module,
    quantizer: FullChannelQuantizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    method = str(args.init_codebook_method).lower()
    if method == "kmeans":
        return init_codebook_with_cache_kmeans(train_loader, val_loader, encoder, decoder, quantizer, args, device)
    if method == "random_samples":
        return init_codebook_with_random_samples(train_loader, val_loader, encoder, decoder, quantizer, args, device)
    if method in {"random", "random_normal"}:
        return init_codebook_with_random_normal(val_loader, encoder, decoder, quantizer, args)
    raise ValueError(f"unknown init_codebook_method: {method}")


def optimizer_stage1(encoder: nn.Module, decoder: nn.Module, quantizer: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    return optim.Adam(
        [
            {"params": encoder.parameters(), "lr": float(args.stage1_encoder_lr_phase1), "name": "encoder"},
            {"params": decoder.parameters(), "lr": float(args.stage1_decoder_lr_phase1), "name": "decoder"},
            {"params": quantizer.parameters(), "lr": float(args.stage1_codebook_lr_phase1), "name": "codebook"},
        ]
    )


def set_stage1_lrs(optimizer: optim.Optimizer, args: argparse.Namespace, epoch: int) -> dict[str, float]:
    epoch = int(epoch)
    phase1_end = int(args.stage1_phase1_end_epoch)
    phase2_end = int(args.stage1_phase2_end_epoch)
    if epoch <= phase1_end:
        lrs = {
            "encoder": float(args.stage1_encoder_lr_phase1),
            "decoder": float(args.stage1_decoder_lr_phase1),
            "codebook": float(args.stage1_codebook_lr_phase1),
            "phase": 1.0,
        }
    elif epoch <= phase2_end:
        lrs = {
            "encoder": float(args.stage1_encoder_lr_phase2),
            "decoder": float(args.stage1_decoder_lr_phase2),
            "codebook": float(args.stage1_codebook_lr_phase2),
            "phase": 2.0,
        }
    else:
        lrs = {
            "encoder": float(args.stage1_encoder_lr_phase3),
            "decoder": float(args.stage1_decoder_lr_phase3),
            "codebook": float(args.stage1_codebook_lr_phase3),
            "phase": 3.0,
        }
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        if name in lrs:
            group["lr"] = lrs[name]
    return lrs


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
            "loss_all_z_rec",
            "loss_usage",
            "loss_usage_entropy",
            "loss_usage_topk",
            "vq",
            "psnr_c1_only",
            "psnr_drop",
            "psnr_full",
            "psnr_all_z",
            "drop_keep",
            "soft_codebook_entropy_bits",
            "soft_codebook_entropy_norm",
            "soft_codebook_perplexity",
            "soft_codebook_top1_ratio",
            "soft_codebook_top10_ratio",
        ]
    }
    device = next(encoder.parameters()).device
    codebook_hist = torch.zeros(int(args.k), dtype=torch.float64)
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
        z_c1, z_c2 = split_c1_c2(z_norm, args)
        q_c2, idx, vq_loss, _q_raw = quantizer(z_c2)
        update_codebook_hist(codebook_hist, idx, int(args.k))
        x_full = decoder(torch.cat([z_c1, q_c2], dim=1)).clamp(0.0, 1.0)
        x_all_z = decoder(torch.cat([z_c1, z_c2], dim=1)).clamp(0.0, 1.0)
        x_c1_only = decoder(torch.cat([z_c1, torch.zeros_like(q_c2)], dim=1)).clamp(0.0, 1.0)
        mask = sample_c2_nested_prefix_mask(
            imgs.shape[0],
            q_c2.shape[1],
            float(args.nested_drop_ratio),
            imgs.device,
            q_c2.dtype,
        )
        x_nested_drop = decoder(torch.cat([z_c1, q_c2 * mask], dim=1)).clamp(0.0, 1.0)
        bsz = imgs.shape[0]
        loss_c1_rec = recon_loss(x_c1_only, imgs)
        loss_drop_rec = recon_loss(x_nested_drop, imgs)
        loss_full_rec = recon_loss(x_full, imgs)
        loss_all_z_rec = recon_loss(x_all_z, imgs)
        if usage_regularizer_enabled(args):
            loss_usage, usage_metrics = soft_codebook_usage_regularizer(z_c2, quantizer, args)
        else:
            loss_usage = loss_full_rec.new_zeros(())
            usage_metrics = {}
        loss = (
            float(args.lambda_c1) * loss_c1_rec
            + float(args.lambda_drop) * loss_drop_rec
            + float(args.lambda_full) * loss_full_rec
            + float(args.lambda_vq) * vq_loss
            + loss_usage
            + float(args.lambda_all_z) * loss_all_z_rec
        )
        meters["loss"].update(float(loss.item()), bsz)
        meters["loss_c1_rec"].update(float(loss_c1_rec.item()), bsz)
        meters["loss_drop_rec"].update(float(loss_drop_rec.item()), bsz)
        meters["loss_full_rec"].update(float(loss_full_rec.item()), bsz)
        meters["loss_all_z_rec"].update(float(loss_all_z_rec.item()), bsz)
        meters["loss_usage"].update(float(loss_usage.item()), bsz)
        for key in [
            "loss_usage_entropy",
            "loss_usage_topk",
            "soft_codebook_entropy_bits",
            "soft_codebook_entropy_norm",
            "soft_codebook_perplexity",
            "soft_codebook_top1_ratio",
            "soft_codebook_top10_ratio",
        ]:
            meters[key].update(float(usage_metrics.get(key, 0.0)), bsz)
        meters["vq"].update(float(vq_loss.item()), bsz)
        meters["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1_only, imgs)), bsz)
        meters["psnr_drop"].update(batch_metric_mean(psnr_per_image(x_nested_drop, imgs)), bsz)
        meters["psnr_full"].update(batch_metric_mean(psnr_per_image(x_full, imgs)), bsz)
        meters["psnr_all_z"].update(batch_metric_mean(psnr_per_image(x_all_z, imgs)), bsz)
        meters["drop_keep"].update(float(mask.float().mean().item()), bsz)
    metrics = {k: v.avg for k, v in meters.items()}
    metrics.update(codebook_usage_from_hist(codebook_hist, int(args.k)))
    return metrics


def train_stage1(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder, quantizer = cvq_io.build_models(args, cfg.device)
    if args.init_ckpt:
        ckpt_obj = cvq_io.load_experiment_checkpoint(args.init_ckpt, encoder=encoder, decoder=decoder, quantizer=quantizer, strict=True)
        if "quantizer_state_dict" not in ckpt_obj:
            print(f"init checkpoint has no quantizer_state_dict; initializing codebook with {args.init_codebook_method}")
            fit_scaled_vq_stats(train_loader, encoder, quantizer, args, cfg.device)
            init_codebook(train_loader, val_loader, encoder, decoder, quantizer, args, cfg.device)
    else:
        if args.init_jscc_encoder:
            cvq_io.load_module_checkpoint(encoder, args.init_jscc_encoder, "init JSCC encoder", strict=True)
        else:
            print("init JSCC encoder: random")
        if args.init_jscc_decoder:
            cvq_io.load_module_checkpoint(decoder, args.init_jscc_decoder, "init JSCC decoder", strict=True)
        else:
            print("init JSCC decoder: random")
        fit_scaled_vq_stats(train_loader, encoder, quantizer, args, cfg.device)
        init_codebook(train_loader, val_loader, encoder, decoder, quantizer, args, cfg.device)
    optimizer = optimizer_stage1(encoder, decoder, quantizer, args)
    best = -1.0
    print_run_header(args, "Stage 1 | end-to-end JSCC + K16384 codebook, no channel", len(train_loader.dataset), len(val_loader.dataset))
    print(
        f"vq_mode={args.stage1_vq_mode} codebook_update={args.stage1_codebook_update} "
        f"codebook_init_method={args.init_codebook_method} "
        f"random_codebook_std={float(args.random_codebook_std):g} "
        f"scaled_vq_random_codebook_std={float(args.scaled_vq_random_codebook_std):g} "
        f"vq_ema_decay={float(args.vq_ema_decay):g} "
        f"vq_ema_initial_count={float(args.vq_ema_initial_count):g}"
    )
    if str(args.stage1_vq_mode).lower() == "simvq":
        print(
            "simvq="
            f"freeze_codebook={not bool(args.simvq_train_codebook)} "
            f"legacy={bool(args.simvq_legacy)} "
            f"proj_bias={bool(args.simvq_proj_bias)} "
            f"codebook_std={float(args.simvq_codebook_std):g}"
        )
    print(
        "usage_regularizer="
        f"lambda_entropy={float(args.lambda_usage_entropy):g} "
        f"lambda_topk={float(args.lambda_usage_topk):g} "
        f"tau={float(args.usage_reg_tau):g} "
        f"top1_target={float(args.usage_reg_top1_target):g} "
        f"top10_target={float(args.usage_reg_top10_target):g} "
        f"detach_codebook={bool(args.usage_reg_detach_codebook)} "
        f"eval={bool(args.usage_reg_eval)}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        lrs = set_stage1_lrs(optimizer, args, epoch)
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
                "loss_all_z_rec",
                "loss_usage",
                "loss_usage_entropy",
                "loss_usage_topk",
                "vq",
                "psnr_c1_only",
                "psnr_drop",
                "psnr_full",
                "psnr_all_z",
                "drop_keep",
                "soft_codebook_entropy_bits",
                "soft_codebook_entropy_norm",
                "soft_codebook_perplexity",
                "soft_codebook_top1_ratio",
                "soft_codebook_top10_ratio",
            ]
        }
        codebook_hist = torch.zeros(int(args.k), dtype=torch.float64)
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
            # q_all, idx, vq_loss, _q_raw = quantizer(z_norm)
            z_c1, z_c2 = split_c1_c2(z_norm, args)
            # q_c1, q_c2 = split_c1_c2(q_all, args)
            q_c2, idx, vq_loss, _q_raw = quantizer(z_c2)
            update_codebook_hist(codebook_hist, idx, int(args.k))
            mask = sample_c2_nested_prefix_mask(
                imgs.shape[0],
                q_c2.shape[1],
                float(args.nested_drop_ratio),
                imgs.device,
                q_c2.dtype,
            )
            x_full = decoder(torch.cat([z_c1, q_c2], dim=1))
            x_all_z = decoder(torch.cat([z_c1, z_c2], dim=1))
            x_nested_drop = decoder(torch.cat([z_c1, q_c2 * mask], dim=1))
            x_c1_only = decoder(torch.cat([z_c1, torch.zeros_like(q_c2)], dim=1))
            loss_c1_rec = recon_loss(x_c1_only, imgs)
            loss_drop_rec = recon_loss(x_nested_drop, imgs)
            loss_full_rec = recon_loss(x_full, imgs)
            loss_all_z_rec = recon_loss(x_all_z, imgs)
            if usage_regularizer_enabled(args):
                loss_usage, usage_metrics = soft_codebook_usage_regularizer(z_c2, quantizer, args)
            else:
                loss_usage = loss_full_rec.new_zeros(())
                usage_metrics = {}
            loss = (
                float(args.lambda_c1) * loss_c1_rec
                + float(args.lambda_drop) * loss_drop_rec
                + float(args.lambda_full) * loss_full_rec
                + float(args.lambda_vq) * vq_loss
                # + loss_usage
                + float(args.lambda_all_z) * loss_all_z_rec
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bsz = imgs.shape[0]
            meters["loss"].update(float(loss.item()), bsz)
            meters["loss_c1_rec"].update(float(loss_c1_rec.item()), bsz)
            meters["loss_drop_rec"].update(float(loss_drop_rec.item()), bsz)
            meters["loss_full_rec"].update(float(loss_full_rec.item()), bsz)
            meters["loss_all_z_rec"].update(float(loss_all_z_rec.item()), bsz)
            meters["loss_usage"].update(float(loss_usage.item()), bsz)
            for key in [
                "loss_usage_entropy",
                "loss_usage_topk",
                "soft_codebook_entropy_bits",
                "soft_codebook_entropy_norm",
                "soft_codebook_perplexity",
                "soft_codebook_top1_ratio",
                "soft_codebook_top10_ratio",
            ]:
                meters[key].update(float(usage_metrics.get(key, 0.0)), bsz)
            meters["vq"].update(float(vq_loss.item()), bsz)
            meters["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1_only, imgs)), bsz)
            meters["psnr_drop"].update(batch_metric_mean(psnr_per_image(x_nested_drop, imgs)), bsz)
            meters["psnr_full"].update(batch_metric_mean(psnr_per_image(x_full, imgs)), bsz)
            meters["drop_keep"].update(float(mask.float().mean().item()), bsz)
            meters["psnr_all_z"].update(batch_metric_mean(psnr_per_image(x_all_z, imgs)), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        metrics["lr_encoder"] = lrs["encoder"]
        metrics["lr_decoder"] = lrs["decoder"]
        metrics["lr_codebook"] = lrs["codebook"]
        metrics["lr_phase"] = lrs["phase"]
        metrics.update(codebook_usage_from_hist(codebook_hist, int(args.k)))
        print_epoch("stage1", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate_stage1(val_loader, encoder, decoder, quantizer, args)
            score = val_metrics["psnr_full"]
            print(f"[stage1 val {epoch:03d}] {format_metrics(val_metrics)} score=psnr_full")
            if score > best:
                best = score
                cvq_io.save_checkpoint(cvq_io.ckpt_path(args, "stage1", f"best-v2-{args.version}"), stage="stage1", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)
        if should_save_latest(args, epoch):
            cvq_io.save_checkpoint(cvq_io.ckpt_path(args, "stage1", f"latest-v2-{args.version}"), stage="stage1", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)
    cvq_io.save_checkpoint(cvq_io.ckpt_path(args, "stage1", f"latest-v2-{args.version}"), stage="stage1", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", type=str, default="cldistill", help="String label for this run, used in saved checkpoint names. Does not affect behavior except for --init-ckpt default.")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default="/workspace/yongjia/paper_code/CDDM/MY/VQ-Stage1")
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default="MY/jscc-no-awgn/cvq_v2_c36_stage0_v2_best_c1distill-c16.pth", help="Optional experiment checkpoint. Empty means random JSCC encoder/decoder and random codebook.")
    p.add_argument("--init-jscc-encoder", type=str, default="", help="Optional JSCC encoder checkpoint used only when --init-ckpt is empty. Empty keeps random initialization.")
    p.add_argument("--init-jscc-decoder", type=str, default="", help="Optional JSCC decoder checkpoint used only when --init-ckpt is empty. Empty keeps random initialization.")
    p.add_argument("--snr-db", type=float, default=12.0)
    p.add_argument("--latent-ch", type=int, default=36)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--k", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--stage1-phase1-end-epoch", type=int, default=0, help="Stage1 LR phase 1 uses epochs <= this value.")
    p.add_argument("--stage1-phase2-end-epoch", type=int, default=0, help="Stage1 LR phase 2 uses epochs <= this value; later epochs use phase 3.")
    p.add_argument("--stage1-encoder-lr-phase1",type=float,default=0.0,)
    p.add_argument("--stage1-encoder-lr-phase2", type=float, default=0.0)
    p.add_argument("--stage1-encoder-lr-phase3",type=float,default=1e-5,)
    p.add_argument("--stage1-decoder-lr-phase1", type=float, default=0.0)
    p.add_argument("--stage1-decoder-lr-phase2",type=float,default=1e-5,)
    p.add_argument("--stage1-decoder-lr-phase3", type=float, default=1e-5)
    p.add_argument("--stage1-codebook-lr-phase1",type=float,default=5e-4,)
    p.add_argument("--stage1-codebook-lr-phase2", type=float, default=1e-4)
    p.add_argument("--stage1-codebook-lr-phase3",type=float,default=5e-5,)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--vq-chunk-size", type=int, default=128)
    p.add_argument(
        "--stage1-codebook-update",
        type=str,
        choices=["grad", "ema"],
        default="ema",
        help="How to update the VQ codebook: gradient loss or EMA assignment means.",
    )
    p.add_argument("--vq-ema-decay", type=float, default=0.99, help="EMA decay for --stage1-codebook-update ema.")
    p.add_argument("--vq-ema-eps", type=float, default=1e-5, help="Numerical epsilon for EMA codebook normalization.")
    p.add_argument(
        "--vq-ema-initial-count",
        type=float,
        default=1.0,
        help="Initial pseudo-count per code when syncing EMA state from the initialized codebook.",
    )
    p.add_argument(
        "--stage1-vq-mode",
        type=str,
        choices=["plain", "scaled_whitened", "simvq"],
        default="simvq",
        help="plain quantizes z2 directly; scaled_whitened quantizes fixed per-channel normalized C2; simvq uses a frozen codebook plus trainable linear projection.",
    )
    p.add_argument(
        "--simvq-codebook-std",
        type=float,
        default=0.0,
        help="Initial SimVQ raw-codebook std. Use 0 for upstream e_dim**-0.5.",
    )
    p.add_argument(
        "--simvq-train-codebook",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the raw SimVQ codebook too. Default keeps it frozen following upstream SimVQ.",
    )
    p.add_argument(
        "--simvq-legacy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use upstream SimVQ legacy loss weighting: commit + beta * codebook/projection loss.",
    )
    p.add_argument(
        "--simvq-proj-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bias in the SimVQ linear projection layer.",
    )
    p.add_argument("--scaled-vq-eps", type=float, default=1e-6)
    p.add_argument(
        "--scaled-vq-stat-batches",
        type=int,
        default=0,
        help="Number of train batches used to fit fixed C2 mean/std for scaled_whitened; 0 means one full loader pass.",
    )
    p.add_argument(
        "--scaled-vq-random-codebook-std",
        type=float,
        default=1.0,
        help="Random-normal codebook std in whitened space when --stage1-vq-mode scaled_whitened.",
    )
    p.add_argument("--lambda-c1", type=float, default=0.0)
    p.add_argument("--lambda-drop", type=float, default=0.0)
    p.add_argument("--lambda-full", type=float, default=1.0)
    p.add_argument("--lambda-vq", type=float, default=0.05)
    p.add_argument("--lambda-all-z", type=float, default=0.01)
    p.add_argument(
        "--lambda-usage-entropy",
        type=float,
        default=0.0,
        help="Weight for the differentiable soft-assignment usage entropy loss; minimizes 1 - normalized entropy.",
    )
    p.add_argument(
        "--lambda-usage-topk",
        type=float,
        default=0.0,
        help="Weight for soft-assignment top1/top10 concentration penalties.",
    )
    p.add_argument(
        "--usage-reg-tau",
        type=float,
        default=1.0,
        help="Temperature for softmax(-distance/tau) used by the soft codebook usage regularizer.",
    )
    p.add_argument("--usage-reg-top1-target", type=float, default=0.05)
    p.add_argument("--usage-reg-top10-target", type=float, default=0.25)
    p.add_argument("--usage-reg-eps", type=float, default=1e-8)
    p.add_argument(
        "--usage-reg-chunk-size",
        type=int,
        default=128,
        help="Number of C2 maps per soft-assignment chunk.",
    )
    p.add_argument(
        "--usage-reg-detach-codebook",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach codebook in usage regularizer so the entropy term pushes encoder assignments rather than moving code vectors directly.",
    )
    p.add_argument(
        "--usage-reg-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute soft-assignment usage metrics even when usage regularizer weights are zero.",
    )
    p.add_argument(
        "--nested-drop-ratio",
        "--c2-dropout-prob",
        dest="nested_drop_ratio",
        type=float,
        default=1.0,
        help="Probability of applying C2 prefix nested dropout; otherwise full C2 is kept.",
    )
    p.add_argument(
        "--init-codebook-method",
        type=str,
        choices=["kmeans", "random_samples", "random_normal"],
        default="random_normal",
        help="How to initialize the stage1 codebook when not loaded from an init checkpoint.",
    )
    p.add_argument("--init-codebook-from-jscc", action=argparse.BooleanOptionalAction, default=False, help="Deprecated compatibility switch; codebook initialization is controlled by --init-codebook-method.")
    p.add_argument("--random-codebook-std", type=float, default=0.02)
    p.add_argument("--latent-cache-maps", type=int, default=262144)
    p.add_argument("--latent-cache-path", type=str, default="")
    p.add_argument("--reuse-latent-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kmeans-iters", type=int, default=30)
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
    #日志打印处
    if not args.log_file:
        args.log_file = str(Path(args.save_dir) / f"stage1_cvq_v2_c36_snr{args.snr_db:g}_k{int(args.k)}-{args.version}.log")
    setup_log_file(args.log_file)
    write_json(Path(args.save_dir) / "stage1_args.json", vars(args))
    train_stage1(args)


if __name__ == "__main__":
    main()
