# 作用：评估 patch_vq Stage2 中“C1 局部聚类 -> C2 code 候选集合”的覆盖率。
# 输出：coverage_summary.json、coverage_by_cluster_G*_M*.csv，以及可选的 hist/candidate/c1_kmeans .pt 文件。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/c1_cluster_candidate_coverage.py --stat-epochs 1 --max-batches 1 --g-values 8 --m-values 32 --val-num-workers 0

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR.parent
TRAIN_DIR = PACKAGE_DIR / "train"
CDDM_ROOT = PACKAGE_DIR.parents[1]
for path in (TRAIN_DIR, PACKAGE_DIR, CDDM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import check_args, real_awgn, resolve_path, seed_everything, setup_log_file  # noqa: E402
from shared import build_quantizer, cvq_io, get_loader, split_c1_c2  # noqa: E402
from Autoencoder.data.datasets import worker_init_fn_seed  # noqa: E402
from Autoencoder.net.network import JSCC_encoder  # noqa: E402


DEFAULT_STAGE2_CKPT = (
    CDDM_ROOT
    / "MY"
    / "checkpoints-cvq-v2-v01-c36-snr9-k4096"
    / "cvq_v2_v01_c36_snr9_k2048_stage2_best.pth"
)


def parse_int_list(text: str) -> list[int]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError("empty integer list")
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage2-ckpt", type=str, default=str(DEFAULT_STAGE2_CKPT))
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--g-values", type=str, default="8,16,32")
    p.add_argument("--m-values", type=str, default="32,64,128")
    p.add_argument("--split", type=str, choices=["train", "valid"], default="train")
    p.add_argument("--stat-epochs", type=int, default=20, help="number of random-crop passes over the selected split")
    p.add_argument("--max-batches", type=int, default=0, help="0 means full split per stat epoch")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--val-num-workers", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--vq-chunk-size", type=int, default=None)
    p.add_argument("--kmeans-iters", type=int, default=20)
    p.add_argument("--kmeans-chunk-size", type=int, default=65536)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--save-hist", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-candidates", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-file", type=str, default="")
    return p.parse_args()


def checkpoint_args(ckpt: dict, cli: argparse.Namespace) -> argparse.Namespace:
    base = dict(ckpt.get("args", {}))
    for key in ["snr_db", "latent_ch", "c1_ch", "k"]:
        if key in ckpt and key not in base:
            base[key] = ckpt[key]
    args = argparse.Namespace(**base)

    overrides = {
        "batch_size": cli.batch_size,
        "num_workers": cli.num_workers,
        "val_num_workers": cli.val_num_workers,
        "data_dir": cli.data_dir,
        "vq_chunk_size": cli.vq_chunk_size,
        "seed": cli.seed,
        "cpu": cli.cpu,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(args, key, value)

    defaults = {
        "quantizer": "patch_vq",
        "patch_size": 4,
        "vq_beta": 0.25,
        "vq_chunk_size": 128,
        "latent_h": 16,
        "latent_w": 16,
        "test_batch": 1,
        "num_workers": 16,
        "val_num_workers": 8,
        "cpu": False,
        "seed": 20260610,
        "save_dir": str(Path(resolve_path(cli.stage2_ckpt)).parent),
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.stage = 2
    check_args(args)
    if str(args.quantizer).lower() != "patch_vq":
        raise ValueError(f"this diagnostic requires patch_vq, got quantizer={args.quantizer}")
    return args


def default_out_dir(stage2_ckpt: str, split: str, stat_epochs: int) -> Path:
    stem = Path(stage2_ckpt).stem
    return THIS_DIR / "outputs" / f"{stem}_{split}_c1_cluster_coverage_e{int(stat_epochs)}"


def make_stats_loader(loader, cfg, args: argparse.Namespace, split: str):
    if split != "train":
        return loader
    nw = int(getattr(cfg, "num_workers", int(args.num_workers)))
    pin = bool(getattr(cfg, "pin_memory", True))
    pw = bool(getattr(cfg, "persistent_workers", False)) and nw > 0
    pf = 2 if nw > 0 else None
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    return torch.utils.data.DataLoader(
        dataset=loader.dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=pw,
        prefetch_factor=pf,
        worker_init_fn=worker_init_fn_seed,
        generator=generator,
        drop_last=False,
    )


def load_stage2(args: argparse.Namespace, ckpt: dict, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    cfg = cvq_io.build_config(args)
    encoder = JSCC_encoder(cfg, int(args.latent_ch)).to(device)
    quantizer = build_quantizer(args, device)
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    quantizer.load_state_dict(ckpt["quantizer_state_dict"], strict=True)
    encoder.eval()
    quantizer.eval()
    return encoder, quantizer


@torch.no_grad()
def collect_features_and_indices(
    loader,
    encoder: torch.nn.Module,
    quantizer: torch.nn.Module,
    args: argparse.Namespace,
    *,
    stat_epochs: int,
    max_batches: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    device = next(encoder.parameters()).device
    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    gh = int(args.latent_h) // int(args.patch_size)
    gw = int(args.latent_w) // int(args.patch_size)
    features = []
    indices = []
    total_images = 0
    total_batches = 0

    for epoch in range(1, int(stat_epochs) + 1):
        epoch_images = 0
        epoch_batches = 0
        for batch_idx, (imgs, _labels) in enumerate(loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
            c1, c2 = split_c1_c2(z_norm, args)
            c1_rx = real_awgn(c1, float(args.snr_db))
            c1_local = F.avg_pool2d(c1_rx.float(), kernel_size=int(args.patch_size), stride=int(args.patch_size))
            if tuple(c1_local.shape[2:]) != (gh, gw):
                raise RuntimeError(f"expected c1_local [B,C1,{gh},{gw}], got {tuple(c1_local.shape)}")
            _q, idx = quantizer.encode(c2)
            if tuple(idx.shape[1:]) != (c2_ch, gh, gw):
                raise RuntimeError(f"expected idx [B,{c2_ch},{gh},{gw}], got {tuple(idx.shape)}")
            features.append(c1_local.permute(0, 2, 3, 1).reshape(-1, int(args.c1_ch)).detach().cpu())
            indices.append(idx.detach().cpu().to(torch.int16))
            bsz = int(imgs.shape[0])
            total_images += bsz
            total_batches += 1
            epoch_images += bsz
            epoch_batches += 1
            if max_batches > 0 and batch_idx >= max_batches:
                break
        print(
            f"collected stat_epoch={epoch:02d}/{int(stat_epochs):02d} "
            f"epoch_batches={epoch_batches} epoch_images={epoch_images} total_images={total_images}",
            flush=True,
        )

    return torch.cat(features, dim=0), torch.cat(indices, dim=0), total_images, total_batches


@torch.no_grad()
def assign_nearest(x: torch.Tensor, centers: torch.Tensor, chunk_size: int) -> torch.Tensor:
    labels = []
    center_norm = centers.square().sum(dim=1).view(1, -1)
    chunk = max(1, int(chunk_size))
    for start in range(0, x.shape[0], chunk):
        xb = x[start : start + chunk]
        dist = xb.square().sum(dim=1, keepdim=True) + center_norm - 2.0 * xb @ centers.t()
        labels.append(dist.argmin(dim=1).to(torch.long))
    return torch.cat(labels, dim=0)


@torch.no_grad()
def kmeans(x_cpu: torch.Tensor, clusters: int, *, iters: int, chunk_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_cpu.to(device=device, dtype=torch.float32)
    n, dim = x.shape
    g = int(clusters)
    if n < g:
        raise ValueError(f"kmeans needs samples >= clusters, got samples={n} clusters={g}")
    centers = x[torch.randperm(n, device=device)[:g]].clone()
    labels = torch.empty(n, device=device, dtype=torch.long)

    for step in range(1, max(1, int(iters)) + 1):
        sums = torch.zeros(g, dim, device=device, dtype=torch.float32)
        counts = torch.zeros(g, device=device, dtype=torch.float32)
        total_dist = torch.zeros((), device=device, dtype=torch.float32)
        center_norm = centers.square().sum(dim=1).view(1, -1)
        chunk = max(1, int(chunk_size))
        for start in range(0, n, chunk):
            xb = x[start : start + chunk]
            dist = xb.square().sum(dim=1, keepdim=True) + center_norm - 2.0 * xb @ centers.t()
            idx = dist.argmin(dim=1)
            labels[start : start + xb.shape[0]] = idx
            sums.index_add_(0, idx, xb)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            total_dist += dist.gather(1, idx.view(-1, 1)).sum()
        empty = counts == 0
        if bool(empty.any()):
            repl = torch.randint(0, n, (int(empty.sum().item()),), device=device)
            sums[empty] = x[repl]
            counts[empty] = 1.0
        centers = sums / counts.clamp_min(1.0).unsqueeze(1)
        print(
            f"kmeans G={g} iter={step:02d}/{int(iters)} "
            f"mse={float(total_dist.item() / max(1, n * dim)):.6g} empty={int(empty.sum().item())}",
            flush=True,
        )

    labels = assign_nearest(x, centers, chunk_size)
    return centers.detach().cpu(), labels.detach().cpu()


def build_cluster_hist(labels: torch.Tensor, idx: torch.Tensor, *, g: int, c2_ch: int, gh: int, gw: int, k: int) -> torch.Tensor:
    n = int(idx.shape[0])
    cluster = labels.reshape(n, gh, gw).long()
    idx_long = idx.long()
    hist = torch.zeros(int(g), c2_ch, gh, gw, k, dtype=torch.long)
    pos_count = c2_ch * gh * gw
    token_count = n * pos_count
    offsets = (
        cluster.unsqueeze(1).expand(n, c2_ch, gh, gw) * (pos_count * k)
        + torch.arange(pos_count, dtype=torch.long).view(1, c2_ch, gh, gw) * k
    )
    linear = (offsets + idx_long).reshape(-1)
    counts = torch.bincount(linear, minlength=int(g) * pos_count * k)
    hist.view(int(g), pos_count, k).copy_(counts.reshape(int(g), pos_count, k))
    if int(hist.sum().item()) != token_count:
        raise RuntimeError(f"hist sum mismatch: hist={int(hist.sum().item())} expected={token_count}")
    return hist


def summarize_cluster_hist(hist: torch.Tensor, m_values: list[int]) -> tuple[dict, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    total = int(hist.sum().item())
    if total <= 0:
        raise RuntimeError("empty histogram")
    out = {}
    candidates = {}
    candidate_counts = {}
    cell_total = hist.sum(dim=-1)
    nonempty = cell_total > 0
    for m in m_values:
        top = hist.topk(min(int(m), hist.shape[-1]), dim=-1)
        hit = top.values.sum(dim=-1)
        coverage = float(hit.sum().item() / total)
        per_cell = torch.zeros_like(cell_total, dtype=torch.float32)
        per_cell[nonempty] = hit[nonempty].float() / cell_total[nonempty].float()
        out[str(m)] = {
            "coverage": coverage,
            "nonempty_cell_mean": float(per_cell[nonempty].mean().item()),
            "nonempty_cell_min": float(per_cell[nonempty].min().item()),
            "nonempty_cell_max": float(per_cell[nonempty].max().item()),
            "nonempty_cell_std": float(per_cell[nonempty].std(unbiased=False).item()),
            "nonempty_cells": int(nonempty.sum().item()),
            "total_cells": int(nonempty.numel()),
        }
        candidates[int(m)] = top.indices.cpu()
        candidate_counts[int(m)] = top.values.cpu()
    return out, candidates, candidate_counts


def write_cluster_coverage_csv(path: Path, per_cell: torch.Tensor, cell_total: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "c2_channel", "patch_u", "patch_v", "count", "coverage"])
        g, c2_ch, gh, gw = per_cell.shape
        for cluster in range(g):
            for j in range(c2_ch):
                for u in range(gh):
                    for v in range(gw):
                        writer.writerow(
                            [
                                cluster,
                                j,
                                u,
                                v,
                                int(cell_total[cluster, j, u, v].item()),
                                float(per_cell[cluster, j, u, v].item()),
                            ]
                        )


def target_summary(all_summary: dict) -> dict[str, object]:
    checks = {
        "G16_M64_ge_50": all_summary.get("16", {}).get("64", {}).get("coverage", 0.0) >= 0.50,
        "G16_M128_ge_65": all_summary.get("16", {}).get("128", {}).get("coverage", 0.0) >= 0.65,
        "G32_M128_ge_75": all_summary.get("32", {}).get("128", {}).get("coverage", 0.0) >= 0.75,
    }
    return {
        "checks": checks,
        "all_targets_met": all(bool(v) for v in checks.values()),
    }


def main() -> None:
    cli = parse_args()
    stage2_ckpt = resolve_path(cli.stage2_ckpt)
    ckpt = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    args = checkpoint_args(ckpt, cli)
    g_values = parse_int_list(cli.g_values)
    m_values = parse_int_list(cli.m_values)
    out_dir = Path(resolve_path(cli.out_dir)) if cli.out_dir else default_out_dir(stage2_ckpt, cli.split, int(cli.stat_epochs))
    out_dir.mkdir(parents=True, exist_ok=True)
    if cli.log_file:
        setup_log_file(cli.log_file)
    seed_everything(int(args.seed))

    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    base_loader = train_loader if cli.split == "train" else val_loader
    loader = make_stats_loader(base_loader, cfg, args, cli.split)
    encoder, quantizer = load_stage2(args, ckpt, cfg.device)

    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    gh = int(args.latent_h) // int(args.patch_size)
    gw = int(args.latent_w) // int(args.patch_size)
    k = int(args.k)

    print("=== C1-cluster conditioned Patch-VQ candidate coverage ===")
    print(f"stage2_ckpt={stage2_ckpt}")
    print(f"split={cli.split} dataset_size={len(loader.dataset)} stat_epochs={int(cli.stat_epochs)} max_batches={int(cli.max_batches)}")
    print(f"K={k} C1={int(args.c1_ch)} C2={c2_ch} patch_size={int(args.patch_size)} G={g_values} M={m_values}")
    print(f"out_dir={out_dir}")

    features, idx, total_images, total_batches = collect_features_and_indices(
        loader,
        encoder,
        quantizer,
        args,
        stat_epochs=int(cli.stat_epochs),
        max_batches=int(cli.max_batches),
    )
    print(f"feature_shape={tuple(features.shape)} idx_shape={tuple(idx.shape)}", flush=True)

    summary = {}
    for g in g_values:
        centers, labels = kmeans(
            features,
            int(g),
            iters=int(cli.kmeans_iters),
            chunk_size=int(cli.kmeans_chunk_size),
            device=cfg.device,
        )
        hist = build_cluster_hist(labels, idx, g=int(g), c2_ch=c2_ch, gh=gh, gw=gw, k=k)
        g_summary, candidates, candidate_counts = summarize_cluster_hist(hist, m_values)
        summary[str(g)] = g_summary
        cell_total = hist.sum(dim=-1)
        torch.save(
            {
                "centers": centers,
                "labels": labels.reshape(idx.shape[0], gh, gw).to(torch.int16),
                "G": int(g),
                "stage2_ckpt": stage2_ckpt,
            },
            out_dir / f"c1_kmeans_G{g}.pt",
        )
        if bool(cli.save_hist):
            torch.save({"hist": hist, "G": int(g), "stage2_ckpt": stage2_ckpt}, out_dir / f"hist_G{g}.pt")
        for m in m_values:
            hit = candidate_counts[m].sum(dim=-1)
            per_cell = torch.zeros_like(cell_total, dtype=torch.float32)
            nonempty = cell_total > 0
            per_cell[nonempty] = hit[nonempty].float() / cell_total[nonempty].float()
            write_cluster_coverage_csv(out_dir / f"coverage_by_cluster_G{g}_M{m}.csv", per_cell, cell_total)
            if bool(cli.save_candidates):
                torch.save(
                    {
                        "candidate": candidates[m],
                        "candidate_count": candidate_counts[m],
                        "coverage_by_cluster_position": per_cell.cpu(),
                        "cell_total": cell_total.cpu(),
                        "G": int(g),
                        "M": int(m),
                        "stage2_ckpt": stage2_ckpt,
                    },
                    out_dir / f"candidate_G{g}_top{m}.pt",
                )
        print("coverage:")
        for m in m_values:
            print(f"  G={g} M={m}: coverage={g_summary[str(m)]['coverage']:.4%}", flush=True)

    payload = {
        "stage2_ckpt": stage2_ckpt,
        "split": cli.split,
        "stat_epochs": int(cli.stat_epochs),
        "total_images": total_images,
        "total_batches": total_batches,
        "feature_shape": list(features.shape),
        "idx_shape": list(idx.shape),
        "g_values": g_values,
        "m_values": m_values,
        "coverage": summary,
        "targets": target_summary(summary),
        "args": vars(args),
        "kmeans_iters": int(cli.kmeans_iters),
    }
    with open(out_dir / "coverage_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print("target checks:")
    for name, ok in payload["targets"]["checks"].items():
        print(f"  {name}: {bool(ok)}")
    print(f"all_targets_met={bool(payload['targets']['all_targets_met'])}")
    print(f"saved: {out_dir / 'coverage_summary.json'}")


if __name__ == "__main__":
    main()
