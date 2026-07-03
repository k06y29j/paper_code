from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from shared import (
    add_common_cli,
    averaged,
    batch_metric_mean,
    build_encoder_decoder,
    ckpt_path,
    cvq_io,
    ensure_common_args,
    format_metrics,
    freeze_module,
    get_loader,
    load_v01_checkpoint,
    meters,
    print_epoch,
    print_v01_header,
    psnr_per_image,
    quantizer_artifact_part,
    real_awgn,
    recon_loss,
    resolve_path,
    sample_uniform_channel_keep_mask,
    save_v01_checkpoint,
    seed_everything,
    setup_stage_log,
    should_save_latest,
    should_validate,
    split_c1_c2,
    with_log_keys,
    write_json,
)


PREFIX_LEVELS = (1, 3, 5, 10)
DEFAULT_FREQ_GROUP_SIZES = (4, 4, 8, 8, 16, 16, 32, 32, 68, 68)
DEFAULT_PCA_K_LIST = (2048, 1024, 1024, 512, 512, 512, 256, 256, 256, 256)


def normalize_pca_k_list(k_list: tuple[int, ...], levels: int) -> tuple[int, ...]:
    levels = int(levels)
    values = tuple(int(v) for v in k_list)
    if levels < 1:
        raise ValueError(f"--rvq-levels must be positive, got {levels}")
    if not values:
        raise ValueError("--pca-k-list cannot be empty")
    if len(values) == levels:
        return values
    if len(values) > levels:
        return values[:levels]
    return values + (values[-1],) * (levels - len(values))


def load_stage1(args: argparse.Namespace, encoder: nn.Module, decoder: nn.Module) -> None:
    obj = load_v01_checkpoint(args.init_stage1_ckpt)
    encoder.load_state_dict(obj["encoder_state_dict"], strict=True)
    decoder.load_state_dict(obj["decoder_state_dict"], strict=True)
    print(f"stage2_rvq_source_stage1={resolve_path(args.init_stage1_ckpt)}")


def nearest_full_map(x: torch.Tensor, codebook: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3:
        raise ValueError(f"expected x [N,H,W], got {tuple(x.shape)}")
    if codebook.ndim != 3:
        raise ValueError(f"expected codebook [K,H,W], got {tuple(codebook.shape)}")
    cb = codebook.float().flatten(1)
    cb_norm = cb.square().sum(dim=1).view(1, -1)
    x2 = x.float().flatten(1)
    quants = []
    indices = []
    chunk = max(1, int(chunk_size))
    for start in range(0, x2.shape[0], chunk):
        xb = x2[start : start + chunk]
        dist = xb.square().sum(dim=1, keepdim=True) + cb_norm - 2.0 * xb @ cb.t()
        idx = dist.argmin(dim=1)
        indices.append(idx)
        quants.append(codebook[idx])
    return torch.cat(quants, dim=0).to(dtype=x.dtype), torch.cat(indices, dim=0)


def nearest_codebook_2d(x: torch.Tensor, codebook: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"expected x [N,D], got {tuple(x.shape)}")
    if codebook.ndim != 2:
        raise ValueError(f"expected codebook [K,D], got {tuple(codebook.shape)}")
    if int(x.shape[1]) != int(codebook.shape[1]):
        raise ValueError(f"token dim mismatch: x={int(x.shape[1])} codebook={int(codebook.shape[1])}")
    cb = codebook.float()
    cb_norm = cb.square().sum(dim=1).view(1, -1)
    quants = []
    indices = []
    chunk = max(1, int(chunk_size))
    for start in range(0, x.shape[0], chunk):
        xb = x[start : start + chunk].float()
        dist = xb.square().sum(dim=1, keepdim=True) + cb_norm - 2.0 * xb @ cb.t()
        idx = dist.argmin(dim=1)
        indices.append(idx)
        quants.append(codebook[idx])
    return torch.cat(quants, dim=0).to(dtype=x.dtype), torch.cat(indices, dim=0)


def build_dct_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    n = int(n)
    i = torch.arange(n, device=device, dtype=dtype).view(1, n)
    k = torch.arange(n, device=device, dtype=dtype).view(n, 1)
    mat = torch.cos(torch.pi * (i + 0.5) * k / float(n))
    mat[0] *= (1.0 / float(n)) ** 0.5
    if n > 1:
        mat[1:] *= (2.0 / float(n)) ** 0.5
    return mat


def dct2(x: torch.Tensor, dct_matrix: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"dct2 expects [B,C,H,W], got {tuple(x.shape)}")
    bsz, channels, h, w = x.shape
    if h != w or int(dct_matrix.shape[0]) != h:
        raise ValueError(f"DCT matrix {tuple(dct_matrix.shape)} does not match input {tuple(x.shape)}")
    d = dct_matrix.to(device=x.device, dtype=x.dtype)
    flat = x.reshape(bsz * channels, h, w)
    out = torch.matmul(torch.matmul(d, flat), d.t())
    return out.reshape(bsz, channels, h, w)


def idct2(y: torch.Tensor, dct_matrix: torch.Tensor) -> torch.Tensor:
    if y.ndim != 4:
        raise ValueError(f"idct2 expects [B,C,H,W], got {tuple(y.shape)}")
    bsz, channels, h, w = y.shape
    if h != w or int(dct_matrix.shape[0]) != h:
        raise ValueError(f"DCT matrix {tuple(dct_matrix.shape)} does not match input {tuple(y.shape)}")
    d = dct_matrix.to(device=y.device, dtype=y.dtype)
    flat = y.reshape(bsz * channels, h, w)
    out = torch.matmul(torch.matmul(d.t(), flat), d)
    return out.reshape(bsz, channels, h, w)


def zigzag_indices(h: int = 16, w: int = 16) -> list[tuple[int, int]]:
    return sorted(((i, j) for i in range(int(h)) for j in range(int(w))), key=lambda item: (item[0] + item[1], item[0]))


def make_frequency_groups(h: int, w: int, group_sizes: tuple[int, ...]) -> list[list[tuple[int, int]]]:
    coords = zigzag_indices(h, w)
    if sum(int(v) for v in group_sizes) != len(coords):
        raise ValueError(f"freq group sizes must sum to {len(coords)}, got {sum(int(v) for v in group_sizes)}")
    groups = []
    start = 0
    for size in group_sizes:
        end = start + int(size)
        groups.append(coords[start:end])
        start = end
    return groups


@torch.no_grad()
def kmeans_full_map(
    samples: torch.Tensor,
    k: int,
    *,
    iters: int,
    chunk_size: int,
    device: torch.device,
    label: str,
) -> torch.Tensor:
    if samples.ndim != 3:
        raise ValueError(f"kmeans expects [N,H,W], got {tuple(samples.shape)}")
    if samples.shape[0] < 1:
        raise ValueError(f"{label}: empty samples")
    x = samples.detach().float().reshape(samples.shape[0], -1).to(device=device)
    k = int(k)
    if x.shape[0] >= k:
        pick = torch.randperm(x.shape[0], device=device)[:k]
    else:
        pick = torch.randint(0, x.shape[0], (k,), device=device)
    centers = x[pick].clone()
    chunk = max(1, int(chunk_size))
    print(f"running {label} kmeans: samples={x.shape[0]} K={k} dim={x.shape[1]} iters={int(iters)} chunk={chunk}")
    for step in range(1, max(1, int(iters)) + 1):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device, dtype=torch.float32)
        total_dist = torch.zeros((), device=device, dtype=torch.float32)
        for start in range(0, x.shape[0], chunk):
            xb = x[start : start + chunk]
            dist = xb.square().sum(dim=1, keepdim=True) + centers.square().sum(dim=1).view(1, -1) - 2.0 * xb @ centers.t()
            idx = dist.argmin(dim=1)
            sums.index_add_(0, idx, xb)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            total_dist += dist.gather(1, idx.view(-1, 1)).sum()
        empty = counts == 0
        if bool(empty.any()):
            repl = torch.randint(0, x.shape[0], (int(empty.sum().item()),), device=device)
            sums[empty] = x[repl]
            counts[empty] = 1.0
        centers = sums / counts.clamp_min(1.0).unsqueeze(1)
        mse = float(total_dist.item() / max(1, x.shape[0] * x.shape[1]))
        print(f"{label} kmeans iter {step:02d}/{int(iters)} mse={mse:.6g} empty={int(empty.sum().item())}")
    return centers.reshape(k, samples.shape[1], samples.shape[2])


@torch.no_grad()
def kmeans_vectors(
    samples: torch.Tensor,
    k: int,
    *,
    iters: int,
    chunk_size: int,
    device: torch.device,
    label: str,
) -> tuple[torch.Tensor, float]:
    if samples.ndim != 2:
        raise ValueError(f"kmeans_vectors expects [N,D], got {tuple(samples.shape)}")
    if samples.shape[0] < 1:
        raise ValueError(f"{label}: empty samples")
    x = samples.detach().float().to(device=device)
    k = int(k)
    if x.shape[0] >= k:
        pick = torch.randperm(x.shape[0], device=device)[:k]
    else:
        pick = torch.randint(0, x.shape[0], (k,), device=device)
    centers = x[pick].clone()
    chunk = max(1, int(chunk_size))
    final_mse = 0.0
    print(f"running {label} kmeans: samples={x.shape[0]} K={k} dim={x.shape[1]} iters={int(iters)} chunk={chunk}")
    for step in range(1, max(1, int(iters)) + 1):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device, dtype=torch.float32)
        total_dist = torch.zeros((), device=device, dtype=torch.float32)
        for start in range(0, x.shape[0], chunk):
            xb = x[start : start + chunk]
            dist = xb.square().sum(dim=1, keepdim=True) + centers.square().sum(dim=1).view(1, -1) - 2.0 * xb @ centers.t()
            idx = dist.argmin(dim=1)
            sums.index_add_(0, idx, xb)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            total_dist += dist.gather(1, idx.view(-1, 1)).sum()
        empty = counts == 0
        if bool(empty.any()):
            repl = torch.randint(0, x.shape[0], (int(empty.sum().item()),), device=device)
            sums[empty] = x[repl]
            counts[empty] = 1.0
        centers = sums / counts.clamp_min(1.0).unsqueeze(1)
        final_mse = float(total_dist.item() / max(1, x.shape[0] * x.shape[1]))
        print(f"{label} kmeans iter {step:02d}/{int(iters)} mse={final_mse:.6g} empty={int(empty.sum().item())}")
    return centers, final_mse


@torch.no_grad()
def random_sample_vectors(
    samples: torch.Tensor,
    k: int,
    *,
    chunk_size: int,
    device: torch.device,
    label: str,
) -> tuple[torch.Tensor, float]:
    if samples.ndim != 2:
        raise ValueError(f"random_sample_vectors expects [N,D], got {tuple(samples.shape)}")
    if samples.shape[0] < 1:
        raise ValueError(f"{label}: empty samples")
    x = samples.detach().float().to(device=device)
    k = int(k)
    if x.shape[0] >= k:
        pick = torch.randperm(x.shape[0], device=device)[:k]
    else:
        pick = torch.randint(0, x.shape[0], (k,), device=device)
    centers = x[pick].clone()
    q, _idx = nearest_codebook_2d(x, centers, int(chunk_size))
    mse = float(F.mse_loss(q.float(), x.float()).item())
    print(f"running {label} random_samples: samples={x.shape[0]} K={k} dim={x.shape[1]} mse={mse:.6g}")
    return centers, mse


@torch.no_grad()
def init_codebook_vectors(
    samples: torch.Tensor,
    k: int,
    *,
    method: str,
    iters: int,
    chunk_size: int,
    device: torch.device,
    label: str,
) -> tuple[torch.Tensor, float]:
    method = str(method).lower()
    if method == "kmeans":
        return kmeans_vectors(samples, k, iters=iters, chunk_size=chunk_size, device=device, label=label)
    if method in {"random_samples", "random_sample"}:
        return random_sample_vectors(samples, k, chunk_size=chunk_size, device=device, label=label)
    raise ValueError(f"unknown codebook init method: {method}")


class SharedLevelFullMapRVQQuantizer(nn.Module):
    def __init__(
        self,
        *,
        c2_ch: int = 20,
        levels: int = 10,
        num_codes: int = 512,
        h: int = 16,
        w: int = 16,
        beta: float = 0.25,
        chunk_size: int = 128,
        prefix_levels: tuple[int, ...] = PREFIX_LEVELS,
    ) -> None:
        super().__init__()
        self.c2_ch = int(c2_ch)
        self.levels = int(levels)
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.prefix_levels = tuple(int(v) for v in prefix_levels if 1 <= int(v) <= self.levels)
        self.codebooks = nn.Parameter(torch.randn(self.levels, self.num_codes, self.h, self.w) * 0.02)
        self.register_buffer("channel_rms", torch.ones(self.c2_ch, 1, 1))

    def _check(self, c2: torch.Tensor) -> None:
        if c2.ndim != 4:
            raise ValueError(f"expected C2 [B,20,16,16], got {tuple(c2.shape)}")
        expected = (self.c2_ch, self.h, self.w)
        if tuple(c2.shape[1:]) != expected:
            raise ValueError(f"expected C2 shape [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(c2.shape)}")

    def normalize(self, c2: torch.Tensor) -> torch.Tensor:
        scale = self.channel_rms.view(1, self.c2_ch, 1, 1).to(device=c2.device, dtype=c2.dtype).clamp_min(1e-6)
        return c2 / scale

    def denormalize(self, q: torch.Tensor) -> torch.Tensor:
        scale = self.channel_rms.view(1, self.c2_ch, 1, 1).to(device=q.device, dtype=q.dtype)
        return q * scale

    def forward(self, c2: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        self._check(c2)
        bsz, channels, h, w = c2.shape
        x = self.normalize(c2)
        q_sum = torch.zeros_like(x)
        q_sum_raw = torch.zeros_like(x)
        residual = x
        idx_levels = []
        prefix = {}
        raw_prefix = {}
        vq_losses = []

        for level in range(self.levels):
            cb = self.codebooks[level]
            flat = residual.reshape(bsz * channels, h, w)
            q_raw_flat, idx_flat = nearest_full_map(flat, cb, self.chunk_size)
            q_raw = q_raw_flat.reshape(bsz, channels, h, w)
            idx_level = idx_flat.reshape(bsz, channels)
            codebook_loss = F.mse_loss(q_raw.float(), residual.detach().float())
            commit_loss = F.mse_loss(q_raw.detach().float(), residual.float())
            vq_losses.append(codebook_loss + self.beta * commit_loss)
            q_level = residual + (q_raw - residual).detach()
            q_sum = q_sum + q_level
            q_sum_raw = q_sum_raw + q_raw
            residual = x - q_sum.detach()
            idx_levels.append(idx_level)
            prefix_n = level + 1
            if prefix_n in self.prefix_levels:
                prefix[prefix_n] = self.denormalize(q_sum)
                raw_prefix[prefix_n] = self.denormalize(q_sum_raw)

        idx = torch.stack(idx_levels, dim=2).reshape(bsz, channels, self.levels)
        vq_loss = torch.stack(vq_losses).sum()
        return prefix, self.denormalize(q_sum), idx, vq_loss, raw_prefix

    @torch.no_grad()
    def init_from_c2_samples(
        self,
        c2_samples: torch.Tensor,
        *,
        iters: int,
        chunk_size: int,
        device: torch.device,
        codebook_init: str = "kmeans",
    ) -> None:
        if c2_samples.ndim != 4:
            raise ValueError(f"expected C2 samples [N,20,16,16], got {tuple(c2_samples.shape)}")
        expected = (self.c2_ch, self.h, self.w)
        if tuple(c2_samples.shape[1:]) != expected:
            raise ValueError(f"expected sample shape [N,{expected[0]},{expected[1]},{expected[2]}], got {tuple(c2_samples.shape)}")
        samples = c2_samples.detach().float().cpu()
        rms = samples.square().mean(dim=(0, 2, 3)).sqrt().clamp_min(1e-6)
        self.channel_rms.copy_(rms.view(self.c2_ch, 1, 1).to(device=self.channel_rms.device, dtype=self.channel_rms.dtype))
        print(
            "initialized channel_rms "
            f"min={float(rms.min().item()):.6g} max={float(rms.max().item()):.6g} mean={float(rms.mean().item()):.6g}"
        )
        norm = samples / rms.view(1, self.c2_ch, 1, 1)
        residual = norm.reshape(-1, self.h, self.w).contiguous()
        print(f"initializing shared-level RVQ: merged_tokens={residual.shape[0]} levels={self.levels} K={self.num_codes}")
        for level in range(self.levels):
            centers = kmeans_full_map(
                residual,
                self.num_codes,
                iters=int(iters),
                chunk_size=int(chunk_size),
                device=device,
                label=f"shared RVQ level {level + 1:02d}",
            )
            self.codebooks.data[level].copy_(centers.to(device=self.codebooks.device, dtype=self.codebooks.dtype))
            q, _idx = nearest_full_map(
                residual.to(device=device),
                self.codebooks.data[level].to(device=device),
                int(chunk_size),
            )
            residual = (residual.to(device=device) - q).cpu()
            print(f"shared RVQ level {level + 1:02d} residual_mse_after_level={float(residual.float().square().mean().item()):.6g}")


class SharedLevelFrequencyRVQQuantizer(nn.Module):
    def __init__(
        self,
        *,
        c2_ch: int = 20,
        h: int = 16,
        w: int = 16,
        num_codes: int = 512,
        group_sizes: tuple[int, ...] = DEFAULT_FREQ_GROUP_SIZES,
        beta: float = 0.25,
        chunk_size: int = 128,
        prefix_levels: tuple[int, ...] = PREFIX_LEVELS,
    ) -> None:
        super().__init__()
        if int(h) != int(w):
            raise ValueError("frequency RVQ currently expects square C2 maps")
        self.c2_ch = int(c2_ch)
        self.h = int(h)
        self.w = int(w)
        self.num_codes = int(num_codes)
        self.group_sizes = tuple(int(v) for v in group_sizes)
        self.levels = len(self.group_sizes)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.prefix_levels = tuple(int(v) for v in prefix_levels if 1 <= int(v) <= self.levels)
        groups = make_frequency_groups(self.h, self.w, self.group_sizes)
        max_dim = max(len(g) for g in groups)
        rows = torch.full((self.levels, max_dim), -1, dtype=torch.long)
        cols = torch.full((self.levels, max_dim), -1, dtype=torch.long)
        sizes = torch.tensor([len(g) for g in groups], dtype=torch.long)
        for group_id, group in enumerate(groups):
            rows[group_id, : len(group)] = torch.tensor([xy[0] for xy in group], dtype=torch.long)
            cols[group_id, : len(group)] = torch.tensor([xy[1] for xy in group], dtype=torch.long)
        dct_mat = build_dct_matrix(self.h)
        ortho_err = float((dct_mat.t() @ dct_mat - torch.eye(self.h)).abs().max().item())
        if ortho_err > 1e-5:
            raise RuntimeError(f"DCT matrix is not orthogonal enough: max_err={ortho_err:.6g}")
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(self.num_codes, dim) * 0.02) for dim in self.group_sizes])
        self.register_buffer("channel_rms", torch.ones(self.c2_ch, 1, 1))
        self.register_buffer("dct_matrix", dct_mat)
        self.register_buffer("freq_group_rows", rows)
        self.register_buffer("freq_group_cols", cols)
        self.register_buffer("freq_group_sizes", sizes)

    def _check(self, c2: torch.Tensor) -> None:
        if c2.ndim != 4:
            raise ValueError(f"expected C2 [B,20,16,16], got {tuple(c2.shape)}")
        expected = (self.c2_ch, self.h, self.w)
        if tuple(c2.shape[1:]) != expected:
            raise ValueError(f"expected C2 shape [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(c2.shape)}")

    def normalize(self, c2: torch.Tensor) -> torch.Tensor:
        scale = self.channel_rms.view(1, self.c2_ch, 1, 1).to(device=c2.device, dtype=c2.dtype).clamp_min(1e-6)
        return c2 / scale

    def denormalize(self, q: torch.Tensor) -> torch.Tensor:
        scale = self.channel_rms.view(1, self.c2_ch, 1, 1).to(device=q.device, dtype=q.dtype)
        return q * scale

    def dct2(self, x: torch.Tensor) -> torch.Tensor:
        return dct2(x, self.dct_matrix)

    def idct2(self, y: torch.Tensor) -> torch.Tensor:
        return idct2(y, self.dct_matrix)

    def _group_coords(self, group_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        size = int(self.freq_group_sizes[group_id].item())
        rows = self.freq_group_rows[group_id, :size].to(device=device)
        cols = self.freq_group_cols[group_id, :size].to(device=device)
        return rows, cols

    def _gather_group(self, freq: torch.Tensor, group_id: int) -> torch.Tensor:
        rows, cols = self._group_coords(group_id, freq.device)
        return freq[:, :, rows, cols]

    def _scatter_group(self, freq_q: torch.Tensor, group_id: int, values: torch.Tensor) -> None:
        rows, cols = self._group_coords(group_id, freq_q.device)
        freq_q[:, :, rows, cols] = values

    def forward(self, c2: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        self._check(c2)
        bsz, channels, h, w = c2.shape
        c2_norm = self.normalize(c2)
        freq = self.dct2(c2_norm)
        freq_q_sum = torch.zeros_like(freq)
        idx_list = []
        prefix = {}
        raw_prefix = {}
        vq_losses = []

        for group_id, codebook in enumerate(self.codebooks):
            token = self._gather_group(freq, group_id)
            flat = token.reshape(bsz * channels, int(token.shape[-1]))
            q_flat, idx_flat = nearest_codebook_2d(flat, codebook, self.chunk_size)
            q_token = q_flat.reshape(bsz, channels, int(token.shape[-1]))
            self._scatter_group(freq_q_sum, group_id, q_token)
            codebook_loss = F.mse_loss(q_token.float(), token.detach().float())
            commit_loss = F.mse_loss(q_token.detach().float(), token.float())
            vq_losses.append(codebook_loss + self.beta * commit_loss)
            idx_list.append(idx_flat.reshape(bsz, channels))

            prefix_n = group_id + 1
            if prefix_n in self.prefix_levels:
                q_norm = self.idct2(freq_q_sum)
                q_raw = self.denormalize(q_norm)
                raw_prefix[prefix_n] = q_raw
                prefix[prefix_n] = c2 + (q_raw - c2).detach()

        idx = torch.stack(idx_list, dim=2).reshape(bsz, channels, self.levels)
        vq_loss = torch.stack(vq_losses).sum()
        q_all = prefix[self.levels] if self.levels in prefix else c2 + (self.denormalize(self.idct2(freq_q_sum)) - c2).detach()
        return prefix, q_all, idx, vq_loss, raw_prefix

    @torch.no_grad()
    def init_from_c2_samples(
        self,
        c2_samples: torch.Tensor,
        *,
        iters: int,
        chunk_size: int,
        device: torch.device,
        codebook_init: str = "kmeans",
    ) -> None:
        if c2_samples.ndim != 4:
            raise ValueError(f"expected C2 samples [N,20,16,16], got {tuple(c2_samples.shape)}")
        expected = (self.c2_ch, self.h, self.w)
        if tuple(c2_samples.shape[1:]) != expected:
            raise ValueError(f"expected sample shape [N,{expected[0]},{expected[1]},{expected[2]}], got {tuple(c2_samples.shape)}")
        samples = c2_samples.detach().float().cpu()
        rms = samples.square().mean(dim=(0, 2, 3)).sqrt().clamp_min(1e-6)
        self.channel_rms.copy_(rms.view(self.c2_ch, 1, 1).to(device=self.channel_rms.device, dtype=self.channel_rms.dtype))
        print(
            "initialized channel_rms "
            f"min={float(rms.min().item()):.6g} max={float(rms.max().item()):.6g} mean={float(rms.mean().item()):.6g}"
        )
        norm = samples / rms.view(1, self.c2_ch, 1, 1)
        freq = dct2(norm, build_dct_matrix(self.h))
        freq_q_sum = torch.zeros_like(freq)
        for group_id, codebook in enumerate(self.codebooks):
            size = int(self.freq_group_sizes[group_id].item())
            rows = self.freq_group_rows[group_id, :size].to(device=freq.device)
            cols = self.freq_group_cols[group_id, :size].to(device=freq.device)
            tokens = freq[:, :, rows, cols].reshape(-1, size).contiguous()
            centers, mse = kmeans_vectors(
                tokens,
                self.num_codes,
                iters=int(iters),
                chunk_size=int(chunk_size),
                device=device,
                label=f"freq group {group_id + 1:02d}",
            )
            codebook.data.copy_(centers.to(device=codebook.device, dtype=codebook.dtype))
            q, _idx = nearest_codebook_2d(tokens.to(device=device), codebook.data.to(device=device), int(chunk_size))
            q_cpu = q.cpu().reshape(samples.shape[0], self.c2_ch, size)
            freq_q_sum[:, :, rows, cols] = q_cpu
            print(f"freq_group_{group_id + 1:02d} dim={size} kmeans_mse={mse:.6g}")
            prefix_n = group_id + 1
            if prefix_n in self.prefix_levels:
                prefix_mse = float((freq - freq_q_sum).float().square().mean().item())
                print(f"freq_prefix_mse_q{prefix_n}={prefix_mse:.6g}")


class SharedLevelPCAFrequencyRVQQuantizer(nn.Module):
    def __init__(
        self,
        *,
        c2_ch: int = 20,
        h: int = 16,
        w: int = 16,
        group_sizes: tuple[int, ...] = DEFAULT_FREQ_GROUP_SIZES,
        k_list: tuple[int, ...] = DEFAULT_PCA_K_LIST,
        beta: float = 0.25,
        chunk_size: int = 128,
        prefix_levels: tuple[int, ...] = PREFIX_LEVELS,
    ) -> None:
        super().__init__()
        self.c2_ch = int(c2_ch)
        self.h = int(h)
        self.w = int(w)
        self.dim = self.h * self.w
        self.group_sizes = tuple(int(v) for v in group_sizes)
        self.k_list = tuple(int(v) for v in k_list)
        self.levels = len(self.group_sizes)
        if sum(self.group_sizes) != self.dim:
            raise ValueError(f"PCA group sizes must sum to {self.dim}, got {sum(self.group_sizes)}")
        if len(self.k_list) != self.levels:
            raise ValueError(f"k_list length must equal groups, got {len(self.k_list)} vs {self.levels}")
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.prefix_levels = tuple(int(v) for v in prefix_levels if 1 <= int(v) <= self.levels)
        starts = []
        ends = []
        start = 0
        for size in self.group_sizes:
            end = start + int(size)
            starts.append(start)
            ends.append(end)
            start = end
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(k, dim) * 0.02) for k, dim in zip(self.k_list, self.group_sizes)]
        )
        self.register_buffer("channel_rms", torch.ones(self.c2_ch, 1, 1))
        self.register_buffer("pca_mean", torch.zeros(self.dim))
        self.register_buffer("pca_components", torch.eye(self.dim))
        self.register_buffer("pca_eigvals", torch.zeros(self.dim))
        self.register_buffer("group_starts", torch.tensor(starts, dtype=torch.long))
        self.register_buffer("group_ends", torch.tensor(ends, dtype=torch.long))
        self.register_buffer("group_sizes_buffer", torch.tensor(self.group_sizes, dtype=torch.long))
        self.register_buffer("k_list_buffer", torch.tensor(self.k_list, dtype=torch.long))

    def _check(self, c2: torch.Tensor) -> None:
        if c2.ndim != 4:
            raise ValueError(f"expected C2 [B,20,16,16], got {tuple(c2.shape)}")
        expected = (self.c2_ch, self.h, self.w)
        if tuple(c2.shape[1:]) != expected:
            raise ValueError(f"expected C2 shape [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(c2.shape)}")

    def normalize(self, c2: torch.Tensor) -> torch.Tensor:
        scale = self.channel_rms.view(1, self.c2_ch, 1, 1).to(device=c2.device, dtype=c2.dtype).clamp_min(1e-6)
        return c2 / scale

    def denormalize(self, q: torch.Tensor) -> torch.Tensor:
        scale = self.channel_rms.view(1, self.c2_ch, 1, 1).to(device=q.device, dtype=q.dtype)
        return q * scale

    def pca_transform(self, c2: torch.Tensor) -> torch.Tensor:
        self._check(c2)
        bsz, channels, _h, _w = c2.shape
        x = self.normalize(c2)
        flat = x.reshape(bsz, channels, self.dim)
        mean = self.pca_mean.to(device=c2.device, dtype=c2.dtype).view(1, 1, self.dim)
        comp = self.pca_components.to(device=c2.device, dtype=c2.dtype)
        return (flat - mean) @ comp

    def pca_inverse(self, coeff: torch.Tensor) -> torch.Tensor:
        if coeff.ndim != 3 or int(coeff.shape[1]) != self.c2_ch or int(coeff.shape[2]) != self.dim:
            raise ValueError(f"expected coeff [B,{self.c2_ch},{self.dim}], got {tuple(coeff.shape)}")
        bsz, channels, _dim = coeff.shape
        comp = self.pca_components.to(device=coeff.device, dtype=coeff.dtype)
        mean = self.pca_mean.to(device=coeff.device, dtype=coeff.dtype).view(1, 1, self.dim)
        flat = coeff @ comp.t() + mean
        return self.denormalize(flat.reshape(bsz, channels, self.h, self.w))

    def _group_slice(self, group_id: int) -> tuple[int, int]:
        return int(self.group_starts[group_id].item()), int(self.group_ends[group_id].item())

    def forward(self, c2: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        self._check(c2)
        bsz, channels, _h, _w = c2.shape
        coeff = self.pca_transform(c2)
        coeff_q = torch.zeros_like(coeff)
        idx_list = []
        prefix = {}
        raw_prefix = {}
        vq_losses = []

        for group_id, codebook in enumerate(self.codebooks):
            start, end = self._group_slice(group_id)
            token = coeff[:, :, start:end]
            flat = token.reshape(bsz * channels, end - start)
            q_flat, idx_flat = nearest_codebook_2d(flat, codebook, self.chunk_size)
            q_token = q_flat.reshape(bsz, channels, end - start)
            coeff_q[:, :, start:end] = q_token
            codebook_loss = F.mse_loss(q_token.float(), token.detach().float())
            commit_loss = F.mse_loss(q_token.detach().float(), token.float())
            vq_losses.append(codebook_loss + self.beta * commit_loss)
            idx_list.append(idx_flat.reshape(bsz, channels))

            prefix_n = group_id + 1
            if prefix_n in self.prefix_levels:
                q_raw = self.pca_inverse(coeff_q)
                raw_prefix[prefix_n] = q_raw
                prefix[prefix_n] = c2 + (q_raw - c2).detach()

        idx = torch.stack(idx_list, dim=2).reshape(bsz, channels, self.levels)
        vq_loss = torch.stack(vq_losses).sum()
        q_all = prefix[self.levels] if self.levels in prefix else c2 + (self.pca_inverse(coeff_q) - c2).detach()
        return prefix, q_all, idx, vq_loss, raw_prefix

    @torch.no_grad()
    def init_from_c2_samples(
        self,
        c2_samples: torch.Tensor,
        *,
        iters: int,
        chunk_size: int,
        device: torch.device,
        codebook_init: str = "kmeans",
    ) -> None:
        if c2_samples.ndim != 4:
            raise ValueError(f"expected C2 samples [N,20,16,16], got {tuple(c2_samples.shape)}")
        expected = (self.c2_ch, self.h, self.w)
        if tuple(c2_samples.shape[1:]) != expected:
            raise ValueError(f"expected sample shape [N,{expected[0]},{expected[1]},{expected[2]}], got {tuple(c2_samples.shape)}")
        samples = c2_samples.detach().float().cpu()
        rms = samples.square().mean(dim=(0, 2, 3)).sqrt().clamp_min(1e-6)
        self.channel_rms.copy_(rms.view(self.c2_ch, 1, 1).to(device=self.channel_rms.device, dtype=self.channel_rms.dtype))
        print(
            "initialized channel_rms "
            f"min={float(rms.min().item()):.6g} max={float(rms.max().item()):.6g} mean={float(rms.mean().item()):.6g}"
        )
        norm = samples / rms.view(1, self.c2_ch, 1, 1)
        tokens = norm.reshape(-1, self.dim).contiguous()
        mean = tokens.mean(dim=0)
        centered = tokens - mean
        cov = centered.t().matmul(centered) / max(1, tokens.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order].clamp_min(0.0).contiguous()
        eigvecs = eigvecs[:, order].contiguous()
        self.pca_mean.copy_(mean.to(device=self.pca_mean.device, dtype=self.pca_mean.dtype))
        self.pca_components.copy_(eigvecs.to(device=self.pca_components.device, dtype=self.pca_components.dtype))
        self.pca_eigvals.copy_(eigvals.to(device=self.pca_eigvals.device, dtype=self.pca_eigvals.dtype))
        explained = eigvals / eigvals.sum().clamp_min(1e-12)
        cum = explained.cumsum(0)
        checkpoints = [int(v) for v in self.group_ends.detach().cpu().tolist()]
        print(
            "PCA explained variance "
            + " ".join(f"q{idx + 1}_dim{end}={float(cum[end - 1].item()):.6g}" for idx, end in enumerate(checkpoints))
        )
        coeff = centered @ eigvecs
        coeff_q = torch.zeros_like(coeff)
        for group_id, codebook in enumerate(self.codebooks):
            start, end = self._group_slice(group_id)
            group_tokens = coeff[:, start:end].contiguous()
            centers, mse = init_codebook_vectors(
                group_tokens,
                int(self.k_list[group_id]),
                method=codebook_init,
                iters=int(iters),
                chunk_size=int(chunk_size),
                device=device,
                label=f"PCA group {group_id + 1:02d}",
            )
            codebook.data.copy_(centers.to(device=codebook.device, dtype=codebook.dtype))
            q, _idx = nearest_codebook_2d(group_tokens.to(device=device), codebook.data.to(device=device), int(chunk_size))
            coeff_q[:, start:end] = q.cpu()
            print(f"pca_group_{group_id + 1:02d} dim={end - start} K={int(self.k_list[group_id])} init={codebook_init} mse={mse:.6g}")
            prefix_n = group_id + 1
            if prefix_n in self.prefix_levels:
                rec = coeff_q @ eigvecs.t() + mean
                prefix_mse = float(F.mse_loss(rec.float(), tokens.float()).item())
                print(f"pca_prefix_mse_q{prefix_n}={prefix_mse:.6g}")


class PerChannelPCAFrequencyRVQQuantizer(nn.Module):
    def __init__(
        self,
        *,
        c2_ch: int = 20,
        h: int = 16,
        w: int = 16,
        group_sizes: tuple[int, ...] = DEFAULT_FREQ_GROUP_SIZES,
        k_list: tuple[int, ...] = DEFAULT_PCA_K_LIST,
        beta: float = 0.25,
        chunk_size: int = 128,
        prefix_levels: tuple[int, ...] = PREFIX_LEVELS,
    ) -> None:
        super().__init__()
        self.c2_ch = int(c2_ch)
        self.h = int(h)
        self.w = int(w)
        self.dim = self.h * self.w
        self.group_sizes = tuple(int(v) for v in group_sizes)
        self.k_list = tuple(int(v) for v in k_list)
        self.levels = len(self.group_sizes)
        if sum(self.group_sizes) != self.dim:
            raise ValueError(f"PCA group sizes must sum to {self.dim}, got {sum(self.group_sizes)}")
        if len(self.k_list) != self.levels:
            raise ValueError(f"k_list length must equal groups, got {len(self.k_list)} vs {self.levels}")
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.prefix_levels = tuple(int(v) for v in prefix_levels if 1 <= int(v) <= self.levels)
        starts = []
        ends = []
        start = 0
        for size in self.group_sizes:
            end = start + int(size)
            starts.append(start)
            ends.append(end)
            start = end
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(self.c2_ch, k, dim) * 0.02) for k, dim in zip(self.k_list, self.group_sizes)]
        )
        self.register_buffer("channel_rms", torch.ones(self.c2_ch, 1, 1))
        self.register_buffer("pca_mean", torch.zeros(self.c2_ch, self.dim))
        self.register_buffer("pca_components", torch.eye(self.dim).repeat(self.c2_ch, 1, 1))
        self.register_buffer("pca_eigvals", torch.zeros(self.c2_ch, self.dim))
        self.register_buffer("group_starts", torch.tensor(starts, dtype=torch.long))
        self.register_buffer("group_ends", torch.tensor(ends, dtype=torch.long))
        self.register_buffer("group_sizes_buffer", torch.tensor(self.group_sizes, dtype=torch.long))
        self.register_buffer("k_list_buffer", torch.tensor(self.k_list, dtype=torch.long))

    def _check(self, c2: torch.Tensor) -> None:
        if c2.ndim != 4:
            raise ValueError(f"expected C2 [B,20,16,16], got {tuple(c2.shape)}")
        expected = (self.c2_ch, self.h, self.w)
        if tuple(c2.shape[1:]) != expected:
            raise ValueError(f"expected C2 shape [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(c2.shape)}")

    def normalize(self, c2: torch.Tensor) -> torch.Tensor:
        scale = self.channel_rms.view(1, self.c2_ch, 1, 1).to(device=c2.device, dtype=c2.dtype).clamp_min(1e-6)
        return c2 / scale

    def denormalize(self, q: torch.Tensor) -> torch.Tensor:
        scale = self.channel_rms.view(1, self.c2_ch, 1, 1).to(device=q.device, dtype=q.dtype)
        return q * scale

    def pca_transform(self, c2: torch.Tensor) -> torch.Tensor:
        self._check(c2)
        bsz, channels, _h, _w = c2.shape
        x = self.normalize(c2)
        flat = x.reshape(bsz, channels, self.dim)
        mean = self.pca_mean.to(device=c2.device, dtype=c2.dtype).view(1, self.c2_ch, self.dim)
        comp = self.pca_components.to(device=c2.device, dtype=c2.dtype)
        return torch.einsum("bcd,cdk->bck", flat - mean, comp)

    def pca_inverse(self, coeff: torch.Tensor) -> torch.Tensor:
        if coeff.ndim != 3 or int(coeff.shape[1]) != self.c2_ch or int(coeff.shape[2]) != self.dim:
            raise ValueError(f"expected coeff [B,{self.c2_ch},{self.dim}], got {tuple(coeff.shape)}")
        bsz, channels, _dim = coeff.shape
        comp = self.pca_components.to(device=coeff.device, dtype=coeff.dtype)
        mean = self.pca_mean.to(device=coeff.device, dtype=coeff.dtype).view(1, self.c2_ch, self.dim)
        flat = torch.einsum("bck,cdk->bcd", coeff, comp) + mean
        return self.denormalize(flat.reshape(bsz, channels, self.h, self.w))

    def _group_slice(self, group_id: int) -> tuple[int, int]:
        return int(self.group_starts[group_id].item()), int(self.group_ends[group_id].item())

    def _nearest_per_channel_group(self, token: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if token.ndim != 3 or int(token.shape[1]) != self.c2_ch:
            raise ValueError(f"expected token [B,{self.c2_ch},D], got {tuple(token.shape)}")
        q_parts = []
        idx_parts = []
        for ch in range(self.c2_ch):
            q_ch, idx_ch = nearest_codebook_2d(token[:, ch, :], codebook[ch], self.chunk_size)
            q_parts.append(q_ch)
            idx_parts.append(idx_ch)
        return torch.stack(q_parts, dim=1), torch.stack(idx_parts, dim=1)

    def forward(self, c2: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        self._check(c2)
        bsz, channels, _h, _w = c2.shape
        coeff = self.pca_transform(c2)
        coeff_q = torch.zeros_like(coeff)
        idx_list = []
        prefix = {}
        raw_prefix = {}
        vq_losses = []

        for group_id, codebook in enumerate(self.codebooks):
            start, end = self._group_slice(group_id)
            token = coeff[:, :, start:end]
            q_token, idx = self._nearest_per_channel_group(token, codebook)
            coeff_q[:, :, start:end] = q_token
            codebook_loss = F.mse_loss(q_token.float(), token.detach().float())
            commit_loss = F.mse_loss(q_token.detach().float(), token.float())
            vq_losses.append(codebook_loss + self.beta * commit_loss)
            idx_list.append(idx)

            prefix_n = group_id + 1
            if prefix_n in self.prefix_levels:
                q_raw = self.pca_inverse(coeff_q)
                raw_prefix[prefix_n] = q_raw
                prefix[prefix_n] = c2 + (q_raw - c2).detach()

        idx = torch.stack(idx_list, dim=2).reshape(bsz, channels, self.levels)
        vq_loss = torch.stack(vq_losses).sum()
        q_all = prefix[self.levels] if self.levels in prefix else c2 + (self.pca_inverse(coeff_q) - c2).detach()
        return prefix, q_all, idx, vq_loss, raw_prefix

    @torch.no_grad()
    def init_from_c2_samples(
        self,
        c2_samples: torch.Tensor,
        *,
        iters: int,
        chunk_size: int,
        device: torch.device,
        codebook_init: str = "kmeans",
    ) -> None:
        if c2_samples.ndim != 4:
            raise ValueError(f"expected C2 samples [N,20,16,16], got {tuple(c2_samples.shape)}")
        expected = (self.c2_ch, self.h, self.w)
        if tuple(c2_samples.shape[1:]) != expected:
            raise ValueError(f"expected sample shape [N,{expected[0]},{expected[1]},{expected[2]}], got {tuple(c2_samples.shape)}")
        samples = c2_samples.detach().float().cpu()
        rms = samples.square().mean(dim=(0, 2, 3)).sqrt().clamp_min(1e-6)
        self.channel_rms.copy_(rms.view(self.c2_ch, 1, 1).to(device=self.channel_rms.device, dtype=self.channel_rms.dtype))
        print(
            "initialized per-channel channel_rms "
            f"min={float(rms.min().item()):.6g} max={float(rms.max().item()):.6g} mean={float(rms.mean().item()):.6g}"
        )

        norm = samples / rms.view(1, self.c2_ch, 1, 1)
        flat = norm.reshape(samples.shape[0], self.c2_ch, self.dim).contiguous()
        coeff_q = torch.zeros_like(flat)
        cum_by_channel = []
        for ch in range(self.c2_ch):
            tokens = flat[:, ch, :].contiguous()
            mean = tokens.mean(dim=0)
            centered = tokens - mean
            cov = centered.t().matmul(centered) / max(1, tokens.shape[0] - 1)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            order = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[order].clamp_min(0.0).contiguous()
            eigvecs = eigvecs[:, order].contiguous()
            self.pca_mean[ch].copy_(mean.to(device=self.pca_mean.device, dtype=self.pca_mean.dtype))
            self.pca_components[ch].copy_(eigvecs.to(device=self.pca_components.device, dtype=self.pca_components.dtype))
            self.pca_eigvals[ch].copy_(eigvals.to(device=self.pca_eigvals.device, dtype=self.pca_eigvals.dtype))
            explained = eigvals / eigvals.sum().clamp_min(1e-12)
            cum_by_channel.append(explained.cumsum(0))
            coeff = centered @ eigvecs
            for group_id, codebook in enumerate(self.codebooks):
                start, end = self._group_slice(group_id)
                group_tokens = coeff[:, start:end].contiguous()
                centers, mse = init_codebook_vectors(
                    group_tokens,
                    int(self.k_list[group_id]),
                    method=codebook_init,
                    iters=int(iters),
                    chunk_size=int(chunk_size),
                    device=device,
                    label=f"per-channel PCA ch{ch + 1:02d} group{group_id + 1:02d}",
                )
                codebook.data[ch].copy_(centers.to(device=codebook.device, dtype=codebook.dtype))
                q, _idx = nearest_codebook_2d(group_tokens.to(device=device), codebook.data[ch].to(device=device), int(chunk_size))
                coeff_q[:, ch, start:end] = q.cpu()

        cum = torch.stack(cum_by_channel, dim=0)
        checkpoints = [int(v) for v in self.group_ends.detach().cpu().tolist()]
        print(
            "per-channel PCA explained variance mean "
            + " ".join(f"q{idx + 1}_dim{end}={float(cum[:, end - 1].mean().item()):.6g}" for idx, end in enumerate(checkpoints))
        )
        for prefix_n in self.prefix_levels:
            rec = torch.zeros_like(flat)
            end = int(self.group_ends[int(prefix_n) - 1].item())
            for ch in range(self.c2_ch):
                coeff_ch = torch.zeros(flat.shape[0], self.dim)
                coeff_ch[:, :end] = coeff_q[:, ch, :end]
                rec[:, ch, :] = coeff_ch @ self.pca_components[ch].detach().cpu().float().t() + self.pca_mean[ch].detach().cpu().float()
            prefix_mse = float(F.mse_loss(rec.float(), flat.float()).item())
            print(f"per_channel_pca_prefix_mse_q{prefix_n}={prefix_mse:.6g}")


@torch.no_grad()
def collect_c2_init_samples(train_loader, encoder: nn.Module, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    encoder.eval()
    samples = []
    seen = 0
    max_images = int(args.init_max_images)
    print(f"collecting C2 RVQ init samples: random_crop_epochs={int(args.init_collect_epochs)} max_images={max_images}")
    for crop_epoch in range(1, int(args.init_collect_epochs) + 1):
        for imgs, _labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
            _c1, c2 = split_c1_c2(z_norm, args)
            samples.append(c2.detach().float().cpu())
            seen += int(c2.shape[0])
            if max_images > 0 and seen >= max_images:
                break
        print(f"  collected crop_epoch={crop_epoch}/{int(args.init_collect_epochs)} images={seen}")
        if max_images > 0 and seen >= max_images:
            break
    c2_samples = torch.cat(samples, dim=0)
    if max_images > 0:
        c2_samples = c2_samples[:max_images]
    print(f"C2 RVQ init tensor: {tuple(c2_samples.shape)}")
    return c2_samples


def build_quantizer(args: argparse.Namespace, device: torch.device) -> nn.Module:
    quantizer_name = str(args.quantizer)
    common_kwargs = dict(
        c2_ch=int(args.latent_ch) - int(args.c1_ch),
        h=int(args.latent_h),
        w=int(args.latent_w),
        k_list=tuple(int(v) for v in args.pca_k_list),
        beta=float(args.vq_beta),
        chunk_size=int(args.vq_chunk_size),
        prefix_levels=tuple(int(v) for v in args.prefix_levels),
    )
    if quantizer_name == "shared_level_pca_frequency_rvq":
        return SharedLevelPCAFrequencyRVQQuantizer(
            group_sizes=tuple(int(v) for v in args.pca_group_sizes),
            **common_kwargs,
        ).to(device)
    if quantizer_name == "per_channel_pca_frequency_rvq":
        return PerChannelPCAFrequencyRVQQuantizer(
            group_sizes=tuple(int(v) for v in args.pca_group_sizes),
            **common_kwargs,
        ).to(device)
    raise ValueError(f"unknown --quantizer {quantizer_name!r}")


def metric_names(prefix_levels: tuple[int, ...], levels: int) -> list[str]:
    names = [
        "loss",
        "loss_q10_rec",
        "loss_c1_rec",
        "loss_q10_drop_rec",
        "loss_prefix_dropout",
        "vq",
        "psnr_c1_only",
        "psnr_real_c2_full",
        "gap_q10_to_real",
        "used_codes_per_channel_level",
        "channel_rms_min",
        "channel_rms_max",
        "channel_rms_mean",
    ]
    for n in prefix_levels:
        names.extend([f"loss_q{n}_rec", f"psnr_q{n}_gt", f"gain_q{n}_over_c1", f"quant_mse_q{n}", f"pca_mse_q{n}", f"pca_explained_q{n}"])
    for group_id in range(1, int(levels) + 1):
        names.extend([f"perplexity_g{group_id}", f"used_codes_g{group_id}"])
    return list(dict.fromkeys(names))


@torch.no_grad()
def rvq_usage_stats(
    idx: torch.Tensor,
    raw_prefix: dict[int, torch.Tensor],
    target: torch.Tensor,
    *,
    levels: int,
    quantizer: nn.Module,
) -> dict[str, float]:
    stats: dict[str, float] = {}
    target_coeff = quantizer.pca_transform(target)
    eigvals = quantizer.pca_eigvals.detach().float().cpu().clamp_min(0.0)
    if eigvals.ndim == 1:
        eig_cum = eigvals.cumsum(0) / eigvals.sum().clamp_min(1e-12)
    elif eigvals.ndim == 2:
        eig_cum = eigvals.cumsum(1) / eigvals.sum(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        raise ValueError(f"unexpected pca_eigvals shape: {tuple(eigvals.shape)}")
    for n, q in raw_prefix.items():
        stats[f"quant_mse_q{n}"] = float(F.mse_loss(q.float(), target.float()).item())
        q_coeff = quantizer.pca_transform(q)
        stats[f"pca_mse_q{n}"] = float(F.mse_loss(q_coeff.float(), target_coeff.float()).item())
        end = int(quantizer.group_ends[int(n) - 1].item())
        if eig_cum.ndim == 1:
            stats[f"pca_explained_q{n}"] = float(eig_cum[end - 1].item())
        else:
            stats[f"pca_explained_q{n}"] = float(eig_cum[:, end - 1].mean().item())
    used_counts = []
    for group_id in range(levels):
        values = idx[:, :, group_id].reshape(-1).detach().cpu()
        k = int(quantizer.k_list_buffer[group_id].item())
        hist = torch.bincount(values, minlength=k).float()
        used = float((hist > 0).sum().item())
        used_counts.append(used)
        prob = hist / hist.sum().clamp_min(1.0)
        nz = prob[prob > 0]
        stats[f"perplexity_g{group_id + 1}"] = float(torch.exp(-(nz * nz.log()).sum()).item()) if nz.numel() else 0.0
        stats[f"used_codes_g{group_id + 1}"] = used
    stats["used_codes_per_channel_level"] = float(sum(used_counts) / max(1, len(used_counts)))
    rms = quantizer.channel_rms.detach().float().cpu()
    stats["channel_rms_min"] = float(rms.min().item())
    stats["channel_rms_max"] = float(rms.max().item())
    stats["channel_rms_mean"] = float(rms.mean().item())
    return stats


def decode_prefixes(decoder: nn.Module, c1_rx: torch.Tensor, prefixes: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {n: decoder(torch.cat([c1_rx, q], dim=1)) for n, q in prefixes.items()}


def compute_losses_and_outputs(
    imgs: torch.Tensor,
    c1_rx: torch.Tensor,
    c2: torch.Tensor,
    decoder: nn.Module,
    quantizer: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor | dict[int, torch.Tensor] | int]]:
    prefixes, q_all, idx, vq_loss, raw_prefix = quantizer(c2)
    zero_c2 = torch.zeros_like(c2)
    x_c1 = decoder(torch.cat([c1_rx, zero_c2], dim=1))
    x_real = decoder(torch.cat([c1_rx, c2], dim=1))
    x_prefix = decode_prefixes(decoder, c1_rx, prefixes)
    mask = sample_uniform_channel_keep_mask(imgs.shape[0], c2.shape[1], imgs.device, c2.dtype)
    x_drop = decoder(torch.cat([c1_rx, q_all * mask], dim=1))

    dropout_n = int(args.prefix_levels[torch.randint(0, len(args.prefix_levels), (1,), device=imgs.device).item()])
    x_prefix_dropout = x_prefix[dropout_n]

    prefix_losses = {n: recon_loss(x_prefix[n], imgs) for n in prefixes}
    loss_c1 = recon_loss(x_c1, imgs)
    loss_drop = recon_loss(x_drop, imgs)
    loss_prefix_dropout = recon_loss(x_prefix_dropout, imgs)
    loss = (
        float(args.lambda_q10) * prefix_losses[10]
        + float(args.lambda_q1) * prefix_losses[1]
        + float(args.lambda_q3) * prefix_losses[3]
        + float(args.lambda_q5) * prefix_losses[5]
        + float(args.lambda_c1) * loss_c1
        + float(args.lambda_q10_drop) * loss_drop
        + float(args.lambda_prefix_dropout) * loss_prefix_dropout
        + float(args.lambda_vq) * vq_loss
    )
    losses = {
        "loss_q10_rec": prefix_losses[10],
        "loss_q1_rec": prefix_losses[1],
        "loss_q3_rec": prefix_losses[3],
        "loss_q5_rec": prefix_losses[5],
        "loss_c1_rec": loss_c1,
        "loss_q10_drop_rec": loss_drop,
        "loss_prefix_dropout": loss_prefix_dropout,
        "vq": vq_loss,
    }
    outputs = {
        "x_c1": x_c1,
        "x_real": x_real,
        "x_prefix": x_prefix,
        "x_drop": x_drop,
        "idx": idx,
        "raw_prefix": raw_prefix,
        "dropout_n": dropout_n,
    }
    return loss, losses, outputs


def update_metrics(
    m: dict,
    *,
    imgs: torch.Tensor,
    loss: torch.Tensor,
    losses: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor | dict[int, torch.Tensor] | int],
    stats: dict[str, float],
    prefix_levels: tuple[int, ...],
) -> None:
    bsz = int(imgs.shape[0])
    m["loss"].update(float(loss.item()), bsz)
    for key, value in losses.items():
        m[key].update(float(value.item()), bsz)
    x_c1 = outputs["x_c1"]
    x_real = outputs["x_real"]
    x_prefix = outputs["x_prefix"]
    if not isinstance(x_c1, torch.Tensor) or not isinstance(x_real, torch.Tensor) or not isinstance(x_prefix, dict):
        raise TypeError("unexpected metric output types")
    psnr_c1 = batch_metric_mean(psnr_per_image(x_c1, imgs))
    psnr_real = batch_metric_mean(psnr_per_image(x_real, imgs))
    m["psnr_c1_only"].update(psnr_c1, bsz)
    m["psnr_real_c2_full"].update(psnr_real, bsz)
    for n in prefix_levels:
        x_n = x_prefix[n]
        psnr_n = batch_metric_mean(psnr_per_image(x_n, imgs))
        m[f"psnr_q{n}_gt"].update(psnr_n, bsz)
        m[f"gain_q{n}_over_c1"].update(psnr_n - psnr_c1, bsz)
    m["gap_q10_to_real"].update(psnr_real - batch_metric_mean(psnr_per_image(x_prefix[10], imgs)), bsz)
    for key, value in stats.items():
        if key in m:
            m[key].update(float(value), bsz)


@torch.no_grad()
def validate(loader, encoder, decoder, quantizer: nn.Module, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    quantizer.eval()
    prefix_levels = tuple(int(v) for v in args.prefix_levels)
    m = meters(metric_names(prefix_levels, int(args.rvq_levels)))
    device = next(decoder.parameters()).device
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        c1_rx = real_awgn(c1, float(args.snr_db))
        loss, losses, outputs = compute_losses_and_outputs(imgs, c1_rx, c2, decoder, quantizer, args)
        raw_prefix = outputs["raw_prefix"]
        idx = outputs["idx"]
        if not isinstance(raw_prefix, dict) or not isinstance(idx, torch.Tensor):
            raise TypeError("unexpected RVQ outputs")
        stats = rvq_usage_stats(idx, raw_prefix, c2, levels=int(args.rvq_levels), quantizer=quantizer)
        update_metrics(m, imgs=imgs, loss=loss, losses=losses, outputs=outputs, stats=stats, prefix_levels=prefix_levels)
    return averaged(m)


def rvq_stage2_score(metrics: dict[str, float]) -> float:
    return (
        float(metrics["gain_q10_over_c1"])
        + 0.3 * float(metrics["gain_q5_over_c1"])
        + 0.2 * float(metrics["gain_q3_over_c1"])
        + 0.1 * float(metrics["gain_q1_over_c1"])
    )


def print_init_validation(metrics: dict[str, float]) -> None:
    fields = [
        ("init_psnr_c1_only", "psnr_c1_only"),
        ("init_psnr_q1_gt", "psnr_q1_gt"),
        ("init_psnr_q3_gt", "psnr_q3_gt"),
        ("init_psnr_q5_gt", "psnr_q5_gt"),
        ("init_psnr_q10_gt", "psnr_q10_gt"),
        ("init_gain_q1_over_c1", "gain_q1_over_c1"),
        ("init_gain_q3_over_c1", "gain_q3_over_c1"),
        ("init_gain_q5_over_c1", "gain_q5_over_c1"),
        ("init_gain_q10_over_c1", "gain_q10_over_c1"),
        ("init_pca_mse_q1", "pca_mse_q1"),
        ("init_pca_mse_q3", "pca_mse_q3"),
        ("init_pca_mse_q5", "pca_mse_q5"),
        ("init_pca_mse_q10", "pca_mse_q10"),
    ]
    print("[stage2-pca-freq-rvq init val] " + " ".join(f"{name}={float(metrics[key]):.6g}" for name, key in fields))
    if float(metrics["gain_q10_over_c1"]) <= 0.0:
        print("[stage2-pca-freq-rvq init warning] gain_q10_over_c1<=0; PCA frequency RVQ initialization did not improve over C1-only.")


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder = build_encoder_decoder(args, cfg.device)
    quantizer = build_quantizer(args, cfg.device)
    load_stage1(args, encoder, decoder)
    if args.init_stage2_ckpt:
        obj = load_v01_checkpoint(args.init_stage2_ckpt)
        if "quantizer_state_dict" in obj:
            quantizer.load_state_dict(obj["quantizer_state_dict"], strict=True)
    else:
        samples = collect_c2_init_samples(train_loader, encoder, args, cfg.device)
        quantizer.init_from_c2_samples(
            samples,
            iters=int(args.kmeans_iters),
            chunk_size=int(args.kmeans_chunk_size),
            device=cfg.device,
            codebook_init=str(args.codebook_init),
        )
    init_metrics = validate(val_loader, encoder, decoder, quantizer, args)
    print_init_validation(init_metrics)

    freeze_module(encoder, bool(args.train_encoder))
    params = list(decoder.parameters()) + [p for p in quantizer.parameters() if p.requires_grad]
    if bool(args.train_encoder):
        params += list(encoder.parameters())
    params = [p for p in params if p.requires_grad]
    opt = optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    prefix_levels = tuple(int(v) for v in args.prefix_levels)
    best = -1.0
    stage_title = {
        "shared_level_pca_frequency_rvq": "Stage 2 | Shared-Level PCA-Frequency RVQ",
        "per_channel_pca_frequency_rvq": "Stage 2 | Per-Channel PCA-Frequency RVQ",
    }.get(str(args.quantizer), "Stage 2 | PCA-Frequency RVQ")
    print_v01_header(args, stage_title, len(train_loader.dataset), len(val_loader.dataset))
    transform_desc = (
        "20 independent channel PCAs, grouped PCA dims per channel"
        if str(args.quantizer) == "per_channel_pca_frequency_rvq"
        else "PCA/KLT learned from train C2"
    )
    print(
        "PCA-Frequency-RVQ config "
        f"C2=[B,20,16,16] transform={transform_desc} "
        f"groups={int(args.rvq_levels)} group_sizes={list(args.pca_group_sizes)} "
        f"K_list={list(args.pca_k_list)} codebooks=[variable K, variable dim] "
        f"idx=[B,20,{int(args.rvq_levels)}] prefixes={list(prefix_levels)} "
        f"channel_rms_norm=True codebook_init={str(args.codebook_init)} train_encoder={bool(args.train_encoder)}"
    )
    print(
        "PCA-Frequency-RVQ loss设计 "
        f"L={float(args.lambda_q10):g}*q10+{float(args.lambda_q1):g}*q1+"
        f"{float(args.lambda_q3):g}*q3+{float(args.lambda_q5):g}*q5+"
        f"{float(args.lambda_c1):g}*c1+{float(args.lambda_q10_drop):g}*channel_drop(q10)+"
        f"{float(args.lambda_prefix_dropout):g}*prefix_dropout+{float(args.lambda_vq):g}*sum(vq_losses)"
    )
    metrics = {}
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train(bool(args.train_encoder))
        decoder.train()
        quantizer.train()
        m = meters(metric_names(prefix_levels, int(args.rvq_levels)))
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            with torch.set_grad_enabled(bool(args.train_encoder)):
                _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
            if not bool(args.train_encoder):
                z_norm = z_norm.detach()
            c1, c2 = split_c1_c2(z_norm, args)
            c1_rx = real_awgn(c1, float(args.snr_db))
            loss, losses, outputs = compute_losses_and_outputs(imgs, c1_rx, c2, decoder, quantizer, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            raw_prefix = outputs["raw_prefix"]
            idx = outputs["idx"]
            if not isinstance(raw_prefix, dict) or not isinstance(idx, torch.Tensor):
                raise TypeError("unexpected RVQ outputs")
            stats = rvq_usage_stats(idx, raw_prefix, c2, levels=int(args.rvq_levels), quantizer=quantizer)
            update_metrics(m, imgs=imgs, loss=loss, losses=losses, outputs=outputs, stats=stats, prefix_levels=prefix_levels)
        metrics = averaged(m)
        print_epoch("stage2-pca-freq-rvq", epoch, int(args.epochs), with_log_keys(metrics), time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, encoder, decoder, quantizer, args)
            score = rvq_stage2_score(val_metrics)
            print(f"[stage2-pca-freq-rvq val {epoch:03d}] {format_metrics(with_log_keys(val_metrics))} score={score:.6g} score_key=multi_prefix_gain")
            if score > best:
                best = score
                save_v01_checkpoint(ckpt_path(args, "stage2", "best"), stage="stage2", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)
        if should_save_latest(args, epoch):
            save_v01_checkpoint(ckpt_path(args, "stage2", "latest"), stage="stage2", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)
    save_v01_checkpoint(ckpt_path(args, "stage2", "latest"), stage="stage2", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_cli(p, default_k=512)
    p.set_defaults(save_dir="MY/checkpoints-cvq-v2-v01-c36-snr9-channel-pca", epochs=100)
    p.add_argument("--init-stage1-ckpt", type=str, default="MY/checkpoints-cvq-v2-v01-c36-snr9-k4096/cvq_v2_v01_c36_snr9_k4096_stage1_best.pth")
    p.add_argument("--init-stage2-ckpt", type=str, default="")
    p.add_argument(
        "--quantizer",
        type=str,
        choices=["shared_level_pca_frequency_rvq", "per_channel_pca_frequency_rvq"],
        default="per_channel_pca_frequency_rvq",
    )
    p.add_argument("--rvq-levels", type=int, default=10)
    p.add_argument("--prefix-levels", type=int, nargs="+", default=list(PREFIX_LEVELS))
    p.add_argument("--pca-group-sizes", type=int, nargs="+", default=list(DEFAULT_FREQ_GROUP_SIZES))
    p.add_argument("--pca-k-list", type=int, nargs="+", default=list(DEFAULT_PCA_K_LIST))
    p.add_argument("--init-collect-epochs", type=int, default=10)
    p.add_argument("--init-max-images", type=int, default=0)
    p.add_argument("--codebook-init", type=str, choices=["kmeans", "random_samples"], default="random_samples")
    p.add_argument("--kmeans-iters", type=int, default=20)
    p.add_argument("--kmeans-chunk-size", type=int, default=4096)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--vq-chunk-size", type=int, default=128)
    p.add_argument("--train-encoder", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-q10", type=float, default=1.0)
    p.add_argument("--lambda-q1", type=float, default=0.5)
    p.add_argument("--lambda-q3", type=float, default=0.3)
    p.add_argument("--lambda-q5", type=float, default=0.3)
    p.add_argument("--lambda-c1", type=float, default=0.25)
    p.add_argument("--lambda-q10-drop", type=float, default=0.01)
    p.add_argument("--lambda-prefix-dropout", type=float, default=0.1)
    p.add_argument("--lambda-vq", type=float, default=0.03)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.prefix_levels = tuple(sorted(set(int(v) for v in args.prefix_levels)))
    args.pca_group_sizes = tuple(int(v) for v in args.pca_group_sizes)
    args.pca_k_list = tuple(int(v) for v in args.pca_k_list)
    args.rvq_levels = int(args.rvq_levels)
    quantizer_name = str(args.quantizer)
    if quantizer_name == "shared_level_pca_frequency_rvq":
        if args.rvq_levels != len(args.pca_group_sizes):
            raise ValueError(f"--rvq-levels must equal len(--pca-group-sizes), got {args.rvq_levels} vs {len(args.pca_group_sizes)}")
        if sum(args.pca_group_sizes) != int(args.latent_h) * int(args.latent_w):
            raise ValueError(f"--pca-group-sizes must sum to latent area {int(args.latent_h) * int(args.latent_w)}")
        if args.rvq_levels != len(args.pca_k_list):
            raise ValueError(f"--rvq-levels must equal len(--pca-k-list), got {args.rvq_levels} vs {len(args.pca_k_list)}")
    elif quantizer_name == "per_channel_pca_frequency_rvq":
        if sum(args.pca_group_sizes) != int(args.latent_h) * int(args.latent_w):
            raise ValueError(f"--pca-group-sizes must sum to latent area {int(args.latent_h) * int(args.latent_w)}")
        if args.rvq_levels != len(args.pca_group_sizes):
            print(f"overriding --rvq-levels from {args.rvq_levels} to len(--pca-group-sizes)={len(args.pca_group_sizes)} for per-channel grouped PCA")
            args.rvq_levels = len(args.pca_group_sizes)
        old_len = len(args.pca_k_list)
        args.pca_k_list = normalize_pca_k_list(args.pca_k_list, args.rvq_levels)
        if old_len != len(args.pca_k_list):
            print(f"expanded --pca-k-list from {old_len} to {len(args.pca_k_list)} for per-channel PCA")
    else:
        raise ValueError(f"unknown --quantizer {quantizer_name!r}")
    required = set(PREFIX_LEVELS)
    if not required.issubset(set(args.prefix_levels)):
        raise ValueError(f"--prefix-levels must include {sorted(required)} for the configured Stage2 losses")
    if max(args.prefix_levels) > int(args.rvq_levels):
        raise ValueError("--prefix-levels cannot exceed --rvq-levels")
    args.predictor = "none"
    args.gate = "none"
    ensure_common_args(args, stage=2)
    setup_stage_log(args, "stage2_pca_freq_rvq_v01")
    write_json(Path(resolve_path(args.save_dir)) / f"stage2_pca_freq_rvq_v01{quantizer_artifact_part(args, 'stage2')}_args.json", vars(args))
    train(args)


if __name__ == "__main__":
    main()
