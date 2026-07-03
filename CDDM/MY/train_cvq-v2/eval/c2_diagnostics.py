# 作用：对 Stage2 C2 量化 checkpoint 做综合诊断，包括 decoder 敏感度、空间/PCA 结构、索引可预测性和基底扰动。
# 输出：summary.json、decoder_sensitivity.csv、spatial_structure.csv、pca_structure.csv、entropy_summary.csv 等诊断文件。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/c2_diagnostics.py --analyses sensitivity,spatial,pca --max-batches 1 --eval-batch-size 1 --val-num-workers 0

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
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

from common import batch_metric_mean, check_args, psnr_per_image, real_awgn, resolve_path, seed_everything  # noqa: E402
from shared import build_encoder_decoder, build_quantizer as build_basic_quantizer, cvq_io, get_loader, split_c1_c2  # noqa: E402
from Autoencoder.data.datasets import worker_init_fn_seed  # noqa: E402


DEFAULT_CKPT = (
    CDDM_ROOT
    / "MY"
    / "checkpoints-cvq-v2-v01-c36-snr9-pca-frequency-rvq-g10"
    / "cvq_v2_v01_c36_snr9_k512_shared_level_pca_frequency_rvq_stage2_best.pth"
)
RVQ_PATH = TRAIN_DIR / "train_stage2_c2_rvq.py"


def load_rvq_module():
    spec = importlib.util.spec_from_file_location("cvq_v2_stage2_rvq", RVQ_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {RVQ_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RVQ = load_rvq_module()


def parse_int_list(text: str) -> list[int]:
    out = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not out:
        raise ValueError("empty int list")
    return out


def parse_float_list(text: str) -> list[float]:
    out = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def parse_groups(text: str, c2_ch: int) -> list[list[int]]:
    groups = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = [int(v) for v in item.split("-", 1)]
            groups.append(list(range(start, end + 1)))
        else:
            groups.append([int(item)])
    if not groups:
        groups = [list(range(c2_ch))]
    for group in groups:
        for ch in group:
            if ch < 0 or ch >= c2_ch:
                raise ValueError(f"group channel {ch} outside C2 range 0..{c2_ch - 1}")
    return groups


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def checkpoint_args(ckpt: dict, cli: argparse.Namespace) -> argparse.Namespace:
    base = dict(ckpt.get("args", {}))
    for key in ["snr_db", "latent_ch", "c1_ch", "k"]:
        if key in ckpt and key not in base:
            base[key] = ckpt[key]
    args = argparse.Namespace(**base)
    overrides = {
        "batch_size": cli.batch_size,
        "test_batch": cli.test_batch,
        "num_workers": cli.num_workers,
        "val_num_workers": cli.val_num_workers,
        "data_dir": cli.data_dir,
        "seed": cli.seed,
        "cpu": cli.cpu,
        "vq_chunk_size": cli.vq_chunk_size,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(args, key, value)
    defaults = {
        "latent_h": 16,
        "latent_w": 16,
        "c1_ch": 16,
        "latent_ch": 36,
        "snr_db": 9.0,
        "k": 0,
        "quantizer": "none",
        "vq_beta": 0.25,
        "vq_chunk_size": 128,
        "simvq_proj_dim": 256,
        "patch_size": 4,
        "block_size": 2,
        "rvq_levels": 10,
        "prefix_levels": [1, 3, 5, 10],
        "freq_group_sizes": list(RVQ.DEFAULT_FREQ_GROUP_SIZES),
        "pca_group_sizes": list(RVQ.DEFAULT_FREQ_GROUP_SIZES),
        "pca_k_list": list(RVQ.DEFAULT_PCA_K_LIST),
        "batch_size": 8,
        "test_batch": 1,
        "num_workers": 8,
        "val_num_workers": 4,
        "seed": 20260610,
        "cpu": False,
        "save_dir": str(Path(resolve_path(cli.checkpoint)).parent),
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.stage = int(getattr(args, "stage", 2))
    check_args(args)
    return args


def build_stage_quantizer(args: argparse.Namespace, ckpt: dict, device: torch.device):
    if "quantizer_state_dict" not in ckpt:
        return None
    kind = str(getattr(args, "quantizer", "none")).lower()
    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    if kind == "shared_level_pca_frequency_rvq":
        q = RVQ.SharedLevelPCAFrequencyRVQQuantizer(
            c2_ch=c2_ch,
            h=int(args.latent_h),
            w=int(args.latent_w),
            group_sizes=tuple(int(v) for v in args.pca_group_sizes),
            k_list=tuple(int(v) for v in args.pca_k_list),
            beta=float(args.vq_beta),
            chunk_size=int(args.vq_chunk_size),
            prefix_levels=tuple(int(v) for v in args.prefix_levels),
        ).to(device)
    elif kind == "shared_level_frequency_rvq":
        q = RVQ.SharedLevelFrequencyRVQQuantizer(
            c2_ch=c2_ch,
            h=int(args.latent_h),
            w=int(args.latent_w),
            num_codes=int(args.k),
            group_sizes=tuple(int(v) for v in args.freq_group_sizes),
            beta=float(args.vq_beta),
            chunk_size=int(args.vq_chunk_size),
            prefix_levels=tuple(int(v) for v in args.prefix_levels),
        ).to(device)
    elif kind == "shared_level_fullmap_rvq_l":
        q = RVQ.SharedLevelFullMapRVQQuantizer(
            c2_ch=c2_ch,
            levels=int(args.rvq_levels),
            num_codes=int(args.k),
            h=int(args.latent_h),
            w=int(args.latent_w),
            beta=float(args.vq_beta),
            chunk_size=int(args.vq_chunk_size),
            prefix_levels=tuple(int(v) for v in args.prefix_levels),
        ).to(device)
    elif kind in {"vq", "simvq", "patch_vq", "cross_channel_block_vq", "cc_block_vq", "block_vq"}:
        q = build_basic_quantizer(args, device)
    else:
        raise ValueError(f"checkpoint has quantizer_state_dict but quantizer={kind} is unknown")
    q.load_state_dict(ckpt["quantizer_state_dict"], strict=True)
    q.eval()
    return q


def make_loader(base_loader, cfg, args: argparse.Namespace, split: str):
    if split != "train":
        return base_loader
    nw = int(getattr(cfg, "num_workers", int(args.num_workers)))
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    return torch.utils.data.DataLoader(
        dataset=base_loader.dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=bool(getattr(cfg, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg, "persistent_workers", False)) and nw > 0,
        prefetch_factor=2 if nw > 0 else None,
        worker_init_fn=worker_init_fn_seed,
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def encode_token_indices(quantizer, c2: torch.Tensor):
    if quantizer is None:
        return None, None
    if hasattr(quantizer, "encode"):
        q, idx = quantizer.encode(c2)
        return q, idx
    out = quantizer(c2)
    if not isinstance(out, tuple) or len(out) < 3:
        raise TypeError("RVQ quantizer must return tuple with q_all and idx")
    return out[1], out[2]


@torch.no_grad()
def collect_latents(loader, encoder, quantizer, args: argparse.Namespace, *, stat_epochs: int, max_batches: int) -> dict[str, torch.Tensor | list[torch.Tensor] | int]:
    device = next(encoder.parameters()).device
    imgs_all, c1_all, c2_all, q_all, idx_all = [], [], [], [], []
    total_batches = 0
    total_images = 0
    for epoch in range(1, int(stat_epochs) + 1):
        epoch_images = 0
        for batch_idx, (imgs, _labels) in enumerate(loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
            c1, c2 = split_c1_c2(z_norm, args)
            c1_rx = real_awgn(c1, float(args.snr_db))
            q, idx = encode_token_indices(quantizer, c2)
            imgs_all.append(imgs.detach().cpu())
            c1_all.append(c1_rx.detach().cpu())
            c2_all.append(c2.detach().cpu())
            if q is not None:
                q_all.append(q.detach().cpu())
            if idx is not None:
                idx_all.append(idx.detach().cpu().to(torch.long))
            total_batches += 1
            total_images += int(imgs.shape[0])
            epoch_images += int(imgs.shape[0])
            if max_batches > 0 and batch_idx >= max_batches:
                break
        print(f"collected epoch={epoch:02d}/{int(stat_epochs):02d} epoch_images={epoch_images} total_images={total_images}", flush=True)
    out = {
        "imgs": torch.cat(imgs_all, dim=0),
        "c1_rx": torch.cat(c1_all, dim=0),
        "c2": torch.cat(c2_all, dim=0),
        "total_images": total_images,
        "total_batches": total_batches,
    }
    if q_all:
        out["q"] = torch.cat(q_all, dim=0)
    if idx_all:
        out["idx"] = torch.cat(idx_all, dim=0)
    return out


def iter_tensor_batches(data: dict[str, torch.Tensor], batch_size: int):
    n = int(data["imgs"].shape[0])
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        yield {k: v[start:end] for k, v in data.items() if isinstance(v, torch.Tensor)}


@torch.no_grad()
def decoder_psnr(decoder, imgs: torch.Tensor, c1_rx: torch.Tensor, c2: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = decoder(torch.cat([c1_rx.to(device), c2.to(device)], dim=1)).clamp(0.0, 1.0)
    return psnr_per_image(x, imgs.to(device)).detach().cpu()


@torch.no_grad()
def analyze_decoder_sensitivity(data: dict, decoder, args, out_dir: Path, noise_scales: list[float], batch_size: int) -> list[dict]:
    c2 = data["c2"].float()
    c2_ch = int(c2.shape[1])
    rms = c2.square().mean(dim=(0, 2, 3)).sqrt()
    var = c2.var(dim=(0, 2, 3), unbiased=False)
    device = next(decoder.parameters()).device
    sums = {
        "c1": 0.0,
        "full": 0.0,
        "keep": torch.zeros(c2_ch),
        "drop": torch.zeros(c2_ch),
        "noise": torch.zeros(c2_ch),
    }
    count = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 17)
    for batch in iter_tensor_batches(data, batch_size):
        imgs = batch["imgs"]
        c1_rx = batch["c1_rx"]
        c2_b = batch["c2"]
        bsz = int(imgs.shape[0])
        zero = torch.zeros_like(c2_b)
        psnr_c1 = decoder_psnr(decoder, imgs, c1_rx, zero, device)
        psnr_full = decoder_psnr(decoder, imgs, c1_rx, c2_b, device)
        sums["c1"] += float(psnr_c1.sum().item())
        sums["full"] += float(psnr_full.sum().item())
        count += bsz
        for ch in range(c2_ch):
            keep = zero.clone()
            keep[:, ch] = c2_b[:, ch]
            drop = c2_b.clone()
            drop[:, ch] = 0.0
            sums["keep"][ch] += float((decoder_psnr(decoder, imgs, c1_rx, keep, device) - psnr_c1).sum().item())
            sums["drop"][ch] += float((psnr_full - decoder_psnr(decoder, imgs, c1_rx, drop, device)).sum().item())
            sens_vals = []
            for scale in noise_scales:
                noisy = c2_b.clone().to(device)
                sigma = float(scale) * float(rms[ch].item())
                noise = torch.randn(noisy[:, ch].shape, generator=gen, device=device, dtype=noisy.dtype) * sigma
                noisy[:, ch] = noisy[:, ch] + noise
                psnr_noisy = decoder_psnr(decoder, imgs, c1_rx, noisy.cpu(), device)
                sens_vals.append((psnr_full - psnr_noisy) / max(float(scale), 1e-6))
            sums["noise"][ch] += float(torch.stack(sens_vals, dim=0).mean(dim=0).sum().item())
    rows = []
    for ch in range(c2_ch):
        keep_gain = float(sums["keep"][ch].item() / max(1, count))
        drop_damage = float(sums["drop"][ch].item() / max(1, count))
        noise_sens = float(sums["noise"][ch].item() / max(1, count))
        rows.append(
            {
                "channel_id": ch,
                "rms": float(rms[ch].item()),
                "variance": float(var[ch].item()),
                "keep_only_gain": keep_gain,
                "drop_one_damage": drop_damage,
                "noise_sensitivity": noise_sens,
                "decoder_importance_score": keep_gain + drop_damage + noise_sens,
            }
        )
    rows = sorted(rows, key=lambda r: r["decoder_importance_score"], reverse=True)
    write_csv(out_dir / "decoder_sensitivity.csv", rows)
    return rows


def build_dct_matrix(n: int, dtype=torch.float32) -> torch.Tensor:
    i = torch.arange(n, dtype=dtype).view(1, n)
    k = torch.arange(n, dtype=dtype).view(n, 1)
    mat = torch.cos(torch.pi * (i + 0.5) * k / float(n))
    mat[0] *= (1.0 / float(n)) ** 0.5
    mat[1:] *= (2.0 / float(n)) ** 0.5
    return mat


def dct2_cpu(x: torch.Tensor) -> torch.Tensor:
    h = int(x.shape[-2])
    d = build_dct_matrix(h, dtype=x.dtype)
    flat = x.reshape(-1, h, h)
    return (d @ flat @ d.t()).reshape_as(x)


def radial_profile(mat: torch.Tensor) -> torch.Tensor:
    h, w = int(mat.shape[-2]), int(mat.shape[-1])
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    center = torch.tensor([(h - 1) / 2.0, (w - 1) / 2.0])
    rr = torch.sqrt((yy.float() - center[0]).square() + (xx.float() - center[1]).square()).round().long()
    out = []
    for r in range(int(rr.max().item()) + 1):
        mask = rr == r
        out.append(mat[mask].mean())
    return torch.stack(out)


def autocorr_length(x: torch.Tensor) -> float:
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    spec = torch.fft.fft2(x.float(), dim=(-2, -1))
    ac = torch.fft.ifft2(spec * spec.conj(), dim=(-2, -1)).real
    ac = torch.fft.fftshift(ac, dim=(-2, -1))
    ac = ac / ac.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    prof = radial_profile(ac.mean(dim=0))
    below = torch.nonzero(prof <= math.exp(-1), as_tuple=False)
    return float(below[0].item()) if below.numel() else float(len(prof) - 1)


def patch_variance(x: torch.Tensor, patch: int) -> float:
    patches = F.unfold(x.unsqueeze(1), kernel_size=int(patch), stride=int(patch))
    return float(patches.var(dim=1, unbiased=False).mean().item())


def analyze_spatial_structure(data: dict, out_dir: Path) -> list[dict]:
    c2 = data["c2"].float()
    n, c, h, w = c2.shape
    dct = dct2_cpu(c2)
    fft_energy = torch.fft.fftshift(torch.fft.fft2(c2, dim=(-2, -1)), dim=(-2, -1)).abs().square()
    dct_energy = dct.square()
    rows = []
    spectra = {"dct_energy_mean": dct_energy.mean(dim=0), "fft_energy_mean": fft_energy.mean(dim=0)}
    center = h // 2
    for ch in range(c):
        de = dct_energy[:, ch]
        fe = fft_energy[:, ch]
        d_total = float(de.sum(dim=(-2, -1)).mean().item())
        f_total = float(fe.sum(dim=(-2, -1)).mean().item())
        fft4 = fe[:, center - 2 : center + 2, center - 2 : center + 2].sum(dim=(-2, -1)).mean()
        fft8 = fe[:, center - 4 : center + 4, center - 4 : center + 4].sum(dim=(-2, -1)).mean()
        dct4 = de[:, :4, :4].sum(dim=(-2, -1)).mean()
        dct8 = de[:, :8, :8].sum(dim=(-2, -1)).mean()
        rows.append(
            {
                "channel_id": ch,
                "dct_low4_ratio": float(dct4.item() / max(d_total, 1e-12)),
                "dct_low8_ratio": float(dct8.item() / max(d_total, 1e-12)),
                "dct_high_ratio": float(1.0 - dct8.item() / max(d_total, 1e-12)),
                "fft_low4_ratio": float(fft4.item() / max(f_total, 1e-12)),
                "fft_low8_ratio": float(fft8.item() / max(f_total, 1e-12)),
                "fft_high_ratio": float(1.0 - fft8.item() / max(f_total, 1e-12)),
                "autocorr_length": autocorr_length(c2[:, ch]),
                "patch4_variance": patch_variance(c2[:, ch], 4),
                "patch8_variance": patch_variance(c2[:, ch], 8),
            }
        )
    write_csv(out_dir / "spatial_structure.csv", rows)
    torch.save(spectra, out_dir / "spatial_spectra.pt")
    return rows


def pca_eigs(tokens: torch.Tensor, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = tokens.float()
    if weights is None:
        mean = x.mean(dim=0)
        centered = x - mean
        cov = centered.t().matmul(centered) / max(1, x.shape[0] - 1)
    else:
        w = weights.float().view(-1, 1).clamp_min(0.0)
        w = w / w.sum().clamp_min(1e-12)
        mean = (x * w).sum(dim=0)
        centered = x - mean
        cov = (centered * w).t().matmul(centered)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    return eigvals[order].clamp_min(0.0), eigvecs[:, order], mean


def pca_row(mode: str, name: str, eigvals: torch.Tensor, dims: list[int], n: int) -> dict:
    total = eigvals.sum().clamp_min(1e-12)
    cum = eigvals.cumsum(0) / total
    row = {"mode": mode, "name": name, "samples": int(n), "total_variance": float(total.item())}
    for dim in dims:
        d = min(int(dim), int(eigvals.numel()))
        row[f"explained_{dim}"] = float(cum[d - 1].item())
    for thresh in [0.5, 0.8, 0.9, 0.95]:
        row[f"dim_for_{int(thresh * 100)}pct"] = int((cum >= thresh).nonzero(as_tuple=False)[0].item() + 1) if bool((cum >= thresh).any()) else int(eigvals.numel())
    return row


def analyze_pca_structure(data: dict, out_dir: Path, dims: list[int], groups: list[list[int]], sensitivity_rows: list[dict] | None) -> dict:
    c2 = data["c2"].float()
    n, c, h, w = c2.shape
    flat = c2.reshape(n, c, h * w)
    rows = []
    eig_dump = {}
    eig, vec, mean = pca_eigs(flat.reshape(n * c, h * w))
    rows.append(pca_row("shared", "all_channels", eig, dims, n * c))
    eig_dump["shared_eigvals"] = eig
    eig_dump["shared_components"] = vec
    eig_dump["shared_mean"] = mean
    for ch in range(c):
        eig_ch, _vec_ch, _mean_ch = pca_eigs(flat[:, ch])
        rows.append(pca_row("per_channel", f"channel_{ch}", eig_ch, dims, n))
    for group in groups:
        tokens = flat[:, group].reshape(n * len(group), h * w)
        eig_g, _vec_g, _mean_g = pca_eigs(tokens)
        rows.append(pca_row("channel_group", "_".join(str(v) for v in group), eig_g, dims, int(tokens.shape[0])))
    if sensitivity_rows:
        by_ch = {int(r["channel_id"]): float(r["decoder_importance_score"]) for r in sensitivity_rows}
        weights = torch.tensor([max(by_ch.get(ch, 0.0), 0.0) for ch in range(c)], dtype=torch.float32)
        if float(weights.sum().item()) <= 0.0:
            weights = torch.ones(c)
        sample_w = weights.view(1, c).expand(n, c).reshape(n * c)
        eig_w, _vec_w, _mean_w = pca_eigs(flat.reshape(n * c, h * w), sample_w)
        rows.append(pca_row("decoder_weighted_shared", "all_channels", eig_w, dims, n * c))
        eig_dump["decoder_weighted_shared_eigvals"] = eig_w
    write_csv(out_dir / "pca_structure.csv", rows)
    torch.save(eig_dump, out_dir / "pca_eigens.pt")
    return eig_dump


@torch.no_grad()
def assign_nearest(x: torch.Tensor, centers: torch.Tensor, chunk_size: int) -> torch.Tensor:
    labels = []
    center_norm = centers.square().sum(dim=1).view(1, -1)
    for start in range(0, x.shape[0], int(chunk_size)):
        xb = x[start : start + int(chunk_size)]
        dist = xb.square().sum(dim=1, keepdim=True) + center_norm - 2.0 * xb @ centers.t()
        labels.append(dist.argmin(dim=1).long())
    return torch.cat(labels, dim=0)


@torch.no_grad()
def kmeans(x_cpu: torch.Tensor, clusters: int, iters: int, chunk_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_cpu.float().to(device)
    n, dim = x.shape
    centers = x[torch.randperm(n, device=device)[: int(clusters)]].clone()
    for step in range(1, int(iters) + 1):
        labels = assign_nearest(x, centers, chunk_size)
        sums = torch.zeros_like(centers)
        counts = torch.zeros(int(clusters), device=device)
        sums.index_add_(0, labels, x)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        empty = counts == 0
        if bool(empty.any()):
            repl = torch.randint(0, n, (int(empty.sum().item()),), device=device)
            sums[empty] = x[repl]
            counts[empty] = 1.0
        centers = sums / counts.clamp_min(1.0).unsqueeze(1)
        print(f"kmeans G={int(clusters)} iter={step:02d}/{int(iters)} empty={int(empty.sum().item())}", flush=True)
    labels = assign_nearest(x, centers, chunk_size)
    return centers.detach().cpu(), labels.detach().cpu()


def entropy_from_hist(hist: torch.Tensor) -> float:
    prob = hist.float() / hist.sum().clamp_min(1.0)
    nz = prob[prob > 0]
    return float((-(nz * torch.log2(nz))).sum().item()) if nz.numel() else 0.0


def token_groups(data: dict, args) -> list[dict]:
    if "idx" not in data:
        return []
    idx = data["idx"].long()
    c1 = data["c1_rx"].float()
    bsz = int(idx.shape[0])
    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    kind = str(getattr(args, "quantizer", "")).lower()
    groups = []
    if idx.ndim == 2:
        feat = c1.mean(dim=(2, 3))
        groups.append({"name": "fullmap", "idx": idx.reshape(-1), "features": feat[:, None, :].expand(bsz, idx.shape[1], feat.shape[1]).reshape(-1, feat.shape[1]), "channel": torch.arange(idx.shape[1]).view(1, -1).expand_as(idx).reshape(-1), "k": int(args.k)})
    elif idx.ndim == 3 and kind.startswith("shared_level"):
        feat = c1.mean(dim=(2, 3))
        k_list = list(getattr(args, "pca_k_list", [])) if kind == "shared_level_pca_frequency_rvq" else [int(args.k)] * int(idx.shape[2])
        for level in range(int(idx.shape[2])):
            values = idx[:, :, level]
            groups.append({"name": f"level_{level + 1}", "idx": values.reshape(-1), "features": feat[:, None, :].expand(bsz, c2_ch, feat.shape[1]).reshape(-1, feat.shape[1]), "channel": torch.arange(c2_ch).view(1, -1).expand_as(values).reshape(-1), "k": int(k_list[level])})
    elif idx.ndim == 3:
        gh, gw = int(idx.shape[1]), int(idx.shape[2])
        feat = F.avg_pool2d(c1, kernel_size=int(args.latent_h) // gh, stride=int(args.latent_h) // gh).permute(0, 2, 3, 1)
        groups.append({"name": "spatial_block", "idx": idx.reshape(-1), "features": feat.reshape(-1, feat.shape[-1]), "channel": torch.full((idx.numel(),), -1, dtype=torch.long), "k": int(args.k)})
    elif idx.ndim == 4:
        gh, gw = int(idx.shape[2]), int(idx.shape[3])
        feat = F.avg_pool2d(c1, kernel_size=int(args.latent_h) // gh, stride=int(args.latent_h) // gh).permute(0, 2, 3, 1)
        features = feat[:, None].expand(bsz, idx.shape[1], gh, gw, feat.shape[-1]).reshape(-1, feat.shape[-1])
        channel = torch.arange(idx.shape[1]).view(1, -1, 1, 1).expand_as(idx).reshape(-1)
        groups.append({"name": "patch", "idx": idx.reshape(-1), "features": features, "channel": channel, "k": int(args.k)})
    else:
        raise ValueError(f"unsupported idx shape {tuple(idx.shape)}")
    return groups


def conditional_entropy(labels: torch.Tensor, idx: torch.Tensor, k: int) -> float:
    total = int(idx.numel())
    out = 0.0
    for value in labels.unique(sorted=True):
        mask = labels == value
        hist = torch.bincount(idx[mask], minlength=int(k))
        out += float(mask.sum().item()) / max(1, total) * entropy_from_hist(hist)
    return out


def cluster_recall(labels: torch.Tensor, idx: torch.Tensor, k: int, m_values: list[int]) -> dict[int, float]:
    hits = {int(m): 0 for m in m_values}
    total = int(idx.numel())
    for value in labels.unique(sorted=True):
        mask = labels == value
        hist = torch.bincount(idx[mask], minlength=int(k))
        order = torch.argsort(hist, descending=True)
        for m in m_values:
            top = order[: min(int(m), int(k))]
            hits[int(m)] += int(torch.isin(idx[mask], top).sum().item())
    return {m: hits[m] / max(1, total) for m in hits}


def analyze_predictability(data: dict, args, out_dir: Path, g_values: list[int], m_values: list[int], kmeans_iters: int, chunk_size: int, device: torch.device) -> list[dict]:
    groups = token_groups(data, args)
    rows, entropy_rows = [], []
    for group in groups:
        idx = group["idx"].long()
        features = group["features"].float()
        channel = group["channel"].long()
        k = int(group["k"])
        h_index = entropy_from_hist(torch.bincount(idx, minlength=k))
        h_channel = conditional_entropy(channel, idx, k) if int(channel.min().item()) >= 0 else float("nan")
        entropy_rows.append({"token_group": group["name"], "tokens": int(idx.numel()), "K": k, "H_index": h_index, "H_index_given_channel_id": h_channel})
        for g in g_values:
            _centers, labels = kmeans(features, int(g), int(kmeans_iters), int(chunk_size), device)
            h_cluster = conditional_entropy(labels, idx, k)
            recall = cluster_recall(labels, idx, k, m_values)
            for m, value in recall.items():
                rows.append({"token_group": group["name"], "G": int(g), "M": int(m), "topM_recall_under_C1_cluster": value, "H_index": h_index, "H_index_given_channel_id": h_channel, "H_index_given_C1_cluster": h_cluster})
            print(f"predictability group={group['name']} G={g} H={h_index:.4f} H|cluster={h_cluster:.4f}", flush=True)
    write_csv(out_dir / "entropy_summary.csv", entropy_rows)
    write_csv(out_dir / "cluster_recall_summary.csv", rows)
    return rows


def make_basis_maps(kind: str, c2: torch.Tensor, max_dirs: int, eig_dump: dict | None) -> tuple[torch.Tensor, torch.Tensor]:
    _n, _c, h, w = c2.shape
    if kind == "pca_shared":
        if eig_dump is None or "shared_components" not in eig_dump:
            eig, vec, _mean = pca_eigs(c2.reshape(-1, h * w))
            energy = eig
            maps = vec.t().reshape(-1, h, w)
        else:
            maps = eig_dump["shared_components"].t().reshape(-1, h, w)
            energy = eig_dump["shared_eigvals"]
    elif kind == "dct":
        d = build_dct_matrix(h)
        maps, energy_vals = [], []
        coords = sorted(((i, j) for i in range(h) for j in range(w)), key=lambda xy: (xy[0] + xy[1], xy[0]))
        coeff = dct2_cpu(c2)
        for i, j in coords:
            basis = torch.outer(d[i], d[j])
            maps.append(basis)
            energy_vals.append(coeff[:, :, i, j].square().mean())
        maps = torch.stack(maps, dim=0)
        energy = torch.stack(energy_vals)
    else:
        raise ValueError(f"unknown basis kind {kind}")
    maps = maps[: int(max_dirs)].float()
    maps = maps / maps.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    energy = energy[: int(max_dirs)].float()
    energy = energy / energy.sum().clamp_min(1e-12)
    return maps, energy


@torch.no_grad()
def analyze_basis_sensitivity(data: dict, decoder, out_dir: Path, basis_kinds: list[str], max_dirs: int, eps: float, eig_dump: dict | None, batch_size: int) -> list[dict]:
    device = next(decoder.parameters()).device
    c2 = data["c2"].float()
    c2_ch = int(c2.shape[1])
    rms = c2.square().mean(dim=(0, 2, 3)).sqrt()
    rows = []
    for kind in basis_kinds:
        maps, energy = make_basis_maps(kind, c2, max_dirs, eig_dump)
        damages = torch.zeros(c2_ch, int(maps.shape[0]))
        count = 0
        for batch in iter_tensor_batches(data, batch_size):
            imgs, c1_rx, c2_b = batch["imgs"], batch["c1_rx"], batch["c2"]
            psnr_full = decoder_psnr(decoder, imgs, c1_rx, c2_b, device)
            count += int(imgs.shape[0])
            for d_idx, basis in enumerate(maps):
                basis_device = basis.to(device=device, dtype=c2_b.dtype)
                for ch in range(c2_ch):
                    pert = c2_b.clone().to(device)
                    amp = float(eps) * float(rms[ch].item()) * math.sqrt(float(basis.numel()))
                    pert[:, ch] = pert[:, ch] + amp * basis_device
                    psnr_pert = decoder_psnr(decoder, imgs, c1_rx, pert.cpu(), device)
                    damages[ch, d_idx] += float((psnr_full - psnr_pert).sum().item())
        for ch in range(c2_ch):
            for d_idx in range(int(maps.shape[0])):
                sens = float(damages[ch, d_idx].item() / max(1, count) / max(float(eps) ** 2, 1e-12))
                rows.append({"basis": kind, "basis_id": d_idx, "channel_id": ch, "energy_ratio": float(energy[d_idx].item()), "decoder_sensitivity": sens, "energy_times_decoder_sensitivity": float(energy[d_idx].item()) * sens})
    rows = sorted(rows, key=lambda r: r["energy_times_decoder_sensitivity"], reverse=True)
    write_csv(out_dir / "decoder_aware_basis_sensitivity.csv", rows)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--split", choices=["valid", "train"], default="valid")
    p.add_argument("--analyses", type=str, default="all", help="comma list: all,sensitivity,spatial,pca,predictability,basis")
    p.add_argument("--stat-epochs", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--test-batch", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--val-num-workers", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--vq-chunk-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--noise-scales", type=str, default="0.25,0.5,1.0")
    p.add_argument("--pca-dims", type=str, default="16,56,112,208,256")
    p.add_argument("--pca-groups", type=str, default="0-4,5-9,10-14,15-19")
    p.add_argument("--g-values", type=str, default="16,32,64")
    p.add_argument("--m-values", type=str, default="32,64,128")
    p.add_argument("--kmeans-iters", type=int, default=20)
    p.add_argument("--kmeans-chunk-size", type=int, default=65536)
    p.add_argument("--basis-kinds", type=str, default="pca_shared,dct")
    p.add_argument("--basis-max-dirs", type=int, default=64)
    p.add_argument("--basis-eps", type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    ckpt_path = resolve_path(cli.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = checkpoint_args(ckpt, cli)
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    loader = make_loader(train_loader if cli.split == "train" else val_loader, cfg, args, cli.split)
    encoder, decoder = build_encoder_decoder(args, cfg.device)
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    encoder.eval()
    decoder.eval()
    quantizer = build_stage_quantizer(args, ckpt, cfg.device)
    stem = Path(ckpt_path).stem
    out_dir = Path(resolve_path(cli.out_dir)) if cli.out_dir else THIS_DIR / "outputs" / f"{stem}_{cli.split}_c2_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    analyses = {item.strip() for item in str(cli.analyses).split(",") if item.strip()}
    if "all" in analyses:
        analyses = {"sensitivity", "spatial", "pca", "predictability", "basis"}

    print("=== C2 diagnostic analyses ===")
    print(f"checkpoint={ckpt_path}")
    print(f"quantizer={getattr(args, 'quantizer', 'none')} split={cli.split} stat_epochs={int(cli.stat_epochs)} max_batches={int(cli.max_batches)}")
    print(f"out_dir={out_dir}")
    data = collect_latents(loader, encoder, quantizer, args, stat_epochs=int(cli.stat_epochs), max_batches=int(cli.max_batches))
    print(f"collected imgs={tuple(data['imgs'].shape)} c1_rx={tuple(data['c1_rx'].shape)} c2={tuple(data['c2'].shape)}")
    if "idx" in data:
        print(f"token_idx={tuple(data['idx'].shape)}")

    sensitivity_rows = None
    eig_dump = None
    if "sensitivity" in analyses:
        sensitivity_rows = analyze_decoder_sensitivity(data, decoder, args, out_dir, parse_float_list(cli.noise_scales), int(cli.eval_batch_size))
    if "spatial" in analyses:
        analyze_spatial_structure(data, out_dir)
    if "pca" in analyses:
        groups = parse_groups(cli.pca_groups, int(args.latent_ch) - int(args.c1_ch))
        eig_dump = analyze_pca_structure(data, out_dir, parse_int_list(cli.pca_dims), groups, sensitivity_rows)
    if "predictability" in analyses:
        if "idx" not in data:
            print("skip predictability: checkpoint has no quantizer indices")
        else:
            analyze_predictability(data, args, out_dir, parse_int_list(cli.g_values), parse_int_list(cli.m_values), int(cli.kmeans_iters), int(cli.kmeans_chunk_size), cfg.device)
    if "basis" in analyses:
        if eig_dump is None:
            eig_dump = analyze_pca_structure(data, out_dir, parse_int_list(cli.pca_dims), parse_groups(cli.pca_groups, int(args.latent_ch) - int(args.c1_ch)), sensitivity_rows)
        analyze_basis_sensitivity(data, decoder, out_dir, [v.strip() for v in cli.basis_kinds.split(",") if v.strip()], int(cli.basis_max_dirs), float(cli.basis_eps), eig_dump, int(cli.eval_batch_size))

    summary = {
        "checkpoint": ckpt_path,
        "split": cli.split,
        "stat_epochs": int(cli.stat_epochs),
        "max_batches": int(cli.max_batches),
        "total_images": int(data["total_images"]),
        "total_batches": int(data["total_batches"]),
        "c2_shape": list(data["c2"].shape),
        "idx_shape": list(data["idx"].shape) if "idx" in data else None,
        "quantizer": str(getattr(args, "quantizer", "none")),
        "analyses": sorted(analyses),
        "outputs": {
            "decoder_sensitivity": str(out_dir / "decoder_sensitivity.csv") if "sensitivity" in analyses else None,
            "spatial_structure": str(out_dir / "spatial_structure.csv") if "spatial" in analyses else None,
            "pca_structure": str(out_dir / "pca_structure.csv") if "pca" in analyses or "basis" in analyses else None,
            "entropy_summary": str(out_dir / "entropy_summary.csv") if "predictability" in analyses and "idx" in data else None,
            "cluster_recall_summary": str(out_dir / "cluster_recall_summary.csv") if "predictability" in analyses and "idx" in data else None,
            "decoder_aware_basis_sensitivity": str(out_dir / "decoder_aware_basis_sensitivity.csv") if "basis" in analyses else None,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"saved summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
