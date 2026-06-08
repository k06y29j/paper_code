from __future__ import annotations

import argparse
from pathlib import Path

import torch

from Autoencoder.data.datasets import get_loader

from .common import format_metrics, print_run_header, resolve_path, seed_everything, write_json
from .io import build_config, build_models, ckpt_path, forward_parts, load_experiment_checkpoint, save_checkpoint

@torch.no_grad()
def init_codebook_from_stage1(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = build_config(args)
    train_loader, _val_loader = get_loader(cfg)
    encoder, decoder, cvq, _car = build_models(args, cfg.device)
    src = args.init_ckpt or ckpt_path(args, "stage1", "best")
    print(f"stage2_source_checkpoint={resolve_path(src)}")
    load_experiment_checkpoint(src, encoder=encoder, decoder=decoder, strict=True)
    encoder.eval()
    need_a = max(int(args.k_a), int(args.min_init_vectors), int(args.k_a) * int(args.init_vectors_per_code))
    need_b = max(int(args.k_b), int(args.min_init_vectors), int(args.k_b) * int(args.init_vectors_per_code))
    print_run_header(args, "Stage 2 | extract normalized tail and initialize codebook", len(train_loader.dataset), 0)
    if bool(getattr(cvq, "needs_tail_channel_samples", False)):
        samples_tail = []
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            _z, _s, _y_prefix, tail = forward_parts(imgs, encoder, args, noisy=False)
            samples_tail.append(tail.detach().cpu())
            if sum(s.shape[0] for s in samples_tail) >= max(need_a, need_b):
                break
        sample_tail = torch.cat(samples_tail, dim=0)
        if int(args.kmeans_iters) > 0:
            init_metrics = cvq.init_from_tail_samples_kmeans(
                sample_tail,
                iters=int(args.kmeans_iters),
                chunk_size=int(args.kmeans_chunk_size),
            )
        else:
            init_metrics = cvq.init_from_tail_samples(sample_tail)
        sample_a = sample_tail[:, : cvq.split_a]
        sample_b = sample_tail[:, cvq.split_a :]
        metrics = {
            "kmeans_iters": float(args.kmeans_iters),
            "sample_a_init_quant_mse": float(init_metrics["sample_a_init_quant_mse"]),
            "sample_a_vectors": float(sample_a.shape[0] * sample_a.shape[1]),
            "sample_a_vectors_per_channel": float(sample_a.shape[0]),
            "sample_b_init_quant_mse": float(init_metrics["sample_b_init_quant_mse"]),
            "sample_b_vectors": float(sample_b.shape[0] * sample_b.shape[1]),
            "sample_b_vectors_per_channel": float(sample_b.shape[0]),
            "target_a_vectors": float(need_a),
            "target_b_vectors": float(need_b),
            "sample_a_std": float(sample_a.std().item()),
            "sample_b_std": float(sample_b.std().item()),
        }
    else:
        samples_a = []
        samples_b = []
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            _z, _s, _y_prefix, tail = forward_parts(imgs, encoder, args, noisy=False)
            samples_a.append(tail[:, : cvq.split_a].detach().cpu().reshape(-1, int(args.latent_h), int(args.latent_w)))
            samples_b.append(tail[:, cvq.split_a :].detach().cpu().reshape(-1, int(args.latent_h), int(args.latent_w)))
            if sum(s.shape[0] for s in samples_a) >= need_a and sum(s.shape[0] for s in samples_b) >= need_b:
                break
        sample_a = torch.cat(samples_a, dim=0)
        sample_b = torch.cat(samples_b, dim=0)
        if int(args.kmeans_iters) > 0:
            init_mse_a = cvq.cvq_a.init_from_samples_kmeans(sample_a, iters=int(args.kmeans_iters), chunk_size=int(args.kmeans_chunk_size))
            init_mse_b = cvq.cvq_b.init_from_samples_kmeans(sample_b, iters=int(args.kmeans_iters), chunk_size=int(args.kmeans_chunk_size))
        else:
            cvq.cvq_a.init_from_samples(sample_a)
            cvq.cvq_b.init_from_samples(sample_b)
            init_mse_a = float("nan")
            init_mse_b = float("nan")
        metrics = {
            "kmeans_iters": float(args.kmeans_iters),
            "sample_a_init_quant_mse": float(init_mse_a),
            "sample_a_vectors": float(sample_a.shape[0]),
            "sample_b_init_quant_mse": float(init_mse_b),
            "sample_b_vectors": float(sample_b.shape[0]),
            "target_a_vectors": float(need_a),
            "target_b_vectors": float(need_b),
            "sample_a_std": float(sample_a.std().item()),
            "sample_b_std": float(sample_b.std().item()),
        }
    save_checkpoint(ckpt_path(args, "stage2", "codebook_init"), stage="stage2", epoch=0, args=args, metrics=metrics, encoder=encoder, decoder=decoder, cvq=cvq)
    write_json(Path(resolve_path(args.save_dir)) / "stage2_codebook_init_stats.json", metrics)
    print(f"[stage2] {format_metrics(metrics)}")
