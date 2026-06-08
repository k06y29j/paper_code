from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Autoencoder.net.encoder import SwinTransformerBlock

def _nearest_codebook(x: torch.Tensor, codebook: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cb = codebook.float().flatten(1)
    cb_norm = cb.square().sum(dim=1).view(1, -1)
    x2 = x.float().flatten(1)
    indices = []
    quants = []
    for start in range(0, x2.shape[0], int(chunk_size)):
        q = x2[start : start + int(chunk_size)]
        dist = q.square().sum(dim=1, keepdim=True) + cb_norm - 2.0 * q @ cb.t()
        idx = dist.argmin(dim=1)
        indices.append(idx)
        quants.append(codebook[idx])
    return torch.cat(quants, dim=0).to(dtype=x.dtype), torch.cat(indices, dim=0)

@torch.no_grad()
def _kmeans_centers(samples: torch.Tensor, num_codes: int, iters: int, chunk_size: int) -> tuple[torch.Tensor, float]:
    x = samples.detach().float().flatten(1)
    n, dim = x.shape
    if n < 1:
        raise ValueError("cannot initialize k-means from empty samples")
    k = int(num_codes)
    if n >= k:
        centers = x[torch.randperm(n, device=x.device)[:k]].clone()
    else:
        centers = x[torch.randint(0, n, (k,), device=x.device)].clone()
    chunk = max(1, int(chunk_size))
    for _ in range(max(1, int(iters))):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(k, device=x.device, dtype=torch.long)
        for start in range(0, n, chunk):
            q = x[start : start + chunk]
            dist = q.square().sum(dim=1, keepdim=True) + centers.square().sum(dim=1).view(1, -1) - 2.0 * q @ centers.t()
            idx = dist.argmin(dim=1)
            sums.index_add_(0, idx, q)
            counts += torch.bincount(idx, minlength=k)
        live = counts > 0
        centers[live] = sums[live] / counts[live].float().unsqueeze(1)
        if bool((~live).any()):
            centers[~live] = x[torch.randint(0, n, (int((~live).sum().item()),), device=x.device)]
    total = 0.0
    count = 0
    for start in range(0, n, chunk):
        q = x[start : start + chunk]
        dist = q.square().sum(dim=1, keepdim=True) + centers.square().sum(dim=1).view(1, -1) - 2.0 * q @ centers.t()
        total += float(dist.min(dim=1).values.sum().item())
        count += int(q.shape[0] * dim)
    return centers, total / max(1, count)

class SingleChannelCVQ(nn.Module):
    def __init__(self, num_codes: int, h: int, w: int, beta: float = 0.25, chunk_size: int = 128):
        super().__init__()
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.codebook = nn.Parameter(torch.randn(self.num_codes, self.h, self.w) * 0.02)

    def _nearest(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _nearest_codebook(x_flat, self.codebook, self.chunk_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, channels, h, w = x.shape
        if (h, w) != (self.h, self.w):
            raise ValueError(f"CVQ codeword size {(self.h, self.w)} does not match input {(h, w)}")
        flat = x.reshape(bsz * channels, h, w)
        quant, idx = self._nearest(flat)
        codebook_loss = F.mse_loss(quant, flat.detach())
        commit_loss = F.mse_loss(quant.detach(), flat)
        vq_loss = codebook_loss + self.beta * commit_loss
        quant_st = flat + (quant - flat).detach()
        return quant_st.reshape(bsz, channels, h, w), idx.reshape(bsz, channels), vq_loss, quant.reshape(bsz, channels, h, w)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, channels, h, w = x.shape
        quant, idx = self._nearest(x.reshape(bsz * channels, h, w))
        return quant.reshape(bsz, channels, h, w), idx.reshape(bsz, channels)

    @torch.no_grad()
    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, channels = idx.shape
        return self.codebook[idx.reshape(-1)].reshape(bsz, channels, self.h, self.w)

    @torch.no_grad()
    def init_from_samples(self, samples: torch.Tensor) -> None:
        samples = samples.detach().float().cpu()
        if samples.ndim != 3:
            raise ValueError(f"expected samples [N,H,W], got {tuple(samples.shape)}")
        if tuple(samples.shape[1:]) != (self.h, self.w):
            raise ValueError(f"sample size {tuple(samples.shape[1:])} does not match {(self.h, self.w)}")
        n = samples.shape[0]
        if n >= self.num_codes:
            perm = torch.randperm(n)[: self.num_codes]
        else:
            perm = torch.randint(0, n, (self.num_codes,))
        self.codebook.copy_(samples[perm].to(device=self.codebook.device, dtype=self.codebook.dtype))

    @torch.no_grad()
    def init_from_samples_kmeans(self, samples: torch.Tensor, iters: int = 25, chunk_size: int = 1024) -> float:
        samples = samples.detach().float().to(device=self.codebook.device)
        if samples.ndim != 3:
            raise ValueError(f"expected samples [N,H,W], got {tuple(samples.shape)}")
        if tuple(samples.shape[1:]) != (self.h, self.w):
            raise ValueError(f"sample size {tuple(samples.shape[1:])} does not match {(self.h, self.w)}")
        centers, mse = _kmeans_centers(samples, self.num_codes, iters, chunk_size)
        self.codebook.copy_(centers.reshape(self.num_codes, self.h, self.w).to(dtype=self.codebook.dtype))
        return mse

    def usage_loss(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        x2 = x.float().reshape(-1, self.h * self.w)
        cb = self.codebook.float().flatten(1)
        dist = x2.square().sum(dim=1, keepdim=True) + cb.square().sum(dim=1).view(1, -1) - 2.0 * x2 @ cb.t()
        prob = torch.softmax(-dist / max(float(tau), 1e-6), dim=1)
        avg = prob.mean(dim=0).clamp_min(1e-8)
        return (avg * (avg * avg.numel()).log()).sum()

    @torch.no_grad()
    def stats(self, idx: torch.Tensor, quant: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        hist = torch.bincount(idx.reshape(-1).detach().cpu(), minlength=self.num_codes).float()
        prob = hist / hist.sum().clamp_min(1.0)
        used = int((hist > 0).sum().item())
        perplexity = float(torch.exp(-(prob[prob > 0] * prob[prob > 0].log()).sum()).item())
        return {
            "K": float(self.num_codes),
            "usage": used / float(self.num_codes),
            "used_codes": float(used),
            "perplexity": perplexity,
            "quant_mse": float(F.mse_loss(quant.float(), target.float()).item()),
        }

class ResidualChannelCVQ(nn.Module):
    def __init__(self, num_codes: int, h: int, w: int, beta: float = 0.25, chunk_size: int = 128, stages: int = 2):
        super().__init__()
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.stages = int(stages)
        if self.stages < 1:
            raise ValueError(f"rvq stages must be >= 1, got {self.stages}")
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(self.num_codes, self.h, self.w) * 0.02) for _ in range(self.stages)]
        )

    def _nearest(self, x_flat: torch.Tensor, stage: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _nearest_codebook(x_flat, self.codebooks[int(stage)], self.chunk_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, channels, h, w = x.shape
        if (h, w) != (self.h, self.w):
            raise ValueError(f"RVQ codeword size {(self.h, self.w)} does not match input {(h, w)}")
        flat = x.reshape(bsz * channels, h, w)
        residual = flat
        quant_total = torch.zeros_like(flat)
        idxs = []
        losses = []
        for stage in range(self.stages):
            quant, idx = self._nearest(residual, stage)
            losses.append(F.mse_loss(quant, residual.detach()) + self.beta * F.mse_loss(quant.detach(), residual))
            quant_total = quant_total + quant
            residual = residual - quant.detach()
            idxs.append(idx)
        quant_st = flat + (quant_total - flat).detach()
        idx = torch.stack(idxs, dim=-1).reshape(bsz, channels, self.stages)
        return quant_st.reshape(bsz, channels, h, w), idx, torch.stack(losses).sum(), quant_total.reshape(bsz, channels, h, w)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, channels, h, w = x.shape
        flat = x.reshape(bsz * channels, h, w)
        residual = flat
        quant_total = torch.zeros_like(flat)
        idxs = []
        for stage in range(self.stages):
            quant, idx = self._nearest(residual, stage)
            quant_total = quant_total + quant
            residual = residual - quant
            idxs.append(idx)
        return quant_total.reshape(bsz, channels, h, w), torch.stack(idxs, dim=-1).reshape(bsz, channels, self.stages)

    @torch.no_grad()
    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, channels, stages = idx.shape
        if stages != self.stages:
            raise RuntimeError(f"expected RVQ stages={self.stages}, got {stages}")
        out = torch.zeros(bsz * channels, self.h, self.w, device=idx.device, dtype=self.codebooks[0].dtype)
        flat_idx = idx.reshape(bsz * channels, stages)
        for stage in range(self.stages):
            out = out + self.codebooks[stage][flat_idx[:, stage]]
        return out.reshape(bsz, channels, self.h, self.w)

    @torch.no_grad()
    def init_from_samples(self, samples: torch.Tensor) -> None:
        samples = samples.detach().float().to(device=self.codebooks[0].device)
        residual = samples
        for stage, codebook in enumerate(self.codebooks):
            n = residual.shape[0]
            pick = torch.randperm(n, device=residual.device)[: self.num_codes] if n >= self.num_codes else torch.randint(0, n, (self.num_codes,), device=residual.device)
            codebook.copy_(residual[pick].to(dtype=codebook.dtype))
            quant, _idx = self._nearest(residual, stage)
            residual = residual - quant

    @torch.no_grad()
    def init_from_samples_kmeans(self, samples: torch.Tensor, iters: int = 25, chunk_size: int = 1024) -> float:
        samples = samples.detach().float().to(device=self.codebooks[0].device)
        if samples.ndim != 3:
            raise ValueError(f"expected samples [N,H,W], got {tuple(samples.shape)}")
        if tuple(samples.shape[1:]) != (self.h, self.w):
            raise ValueError(f"sample size {tuple(samples.shape[1:])} does not match {(self.h, self.w)}")
        residual = samples
        quant_total = torch.zeros_like(samples)
        for stage, codebook in enumerate(self.codebooks):
            centers, _mse = _kmeans_centers(residual, self.num_codes, iters, chunk_size)
            codebook.copy_(centers.reshape(self.num_codes, self.h, self.w).to(dtype=codebook.dtype))
            quant, _idx = self._nearest(residual, stage)
            quant_total = quant_total + quant
            residual = residual - quant
        return float(F.mse_loss(quant_total.float(), samples.float()).item())

    def usage_loss(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        flat = x.float().reshape(-1, self.h, self.w)
        residual = flat
        losses = []
        for stage in range(self.stages):
            cb = self.codebooks[stage].float().flatten(1)
            x2 = residual.flatten(1)
            dist = x2.square().sum(dim=1, keepdim=True) + cb.square().sum(dim=1).view(1, -1) - 2.0 * x2 @ cb.t()
            prob = torch.softmax(-dist / max(float(tau), 1e-6), dim=1)
            avg = prob.mean(dim=0).clamp_min(1e-8)
            losses.append((avg * (avg * avg.numel()).log()).sum())
            with torch.no_grad():
                idx = dist.argmin(dim=1)
                residual = residual - self.codebooks[stage][idx].float()
        return torch.stack(losses).sum()

class PerChannelResidualCVQ(nn.Module):
    def __init__(
        self,
        num_codes: int,
        h: int,
        w: int,
        channels: int,
        beta: float = 0.25,
        chunk_size: int = 128,
        stages: int = 1,
    ):
        super().__init__()
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.channels = int(channels)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.stages = int(stages)
        if self.channels < 1:
            raise ValueError(f"per-channel CVQ channels must be >= 1, got {self.channels}")
        if self.stages < 1:
            raise ValueError(f"per-channel RVQ stages must be >= 1, got {self.stages}")
        self.codebooks = nn.Parameter(torch.randn(self.stages, self.channels, self.num_codes, self.h, self.w) * 0.02)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.channels:
            raise ValueError(f"expected channels={self.channels}, got {x.shape[1]}")
        if tuple(x.shape[2:]) != (self.h, self.w):
            raise ValueError(f"per-channel RVQ size {(self.h, self.w)} does not match input {tuple(x.shape[2:])}")

    def _nearest(self, x_ch: torch.Tensor, stage: int, channel: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _nearest_codebook(x_ch, self.codebooks[int(stage), int(channel)], self.chunk_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_input(x)
        bsz, channels, h, w = x.shape
        residual = x
        quant_total = torch.zeros_like(x)
        idxs = []
        stage_losses = []
        for stage in range(self.stages):
            quant_stage = torch.zeros_like(x)
            idx_stage = []
            losses = []
            for ch in range(channels):
                quant, idx = self._nearest(residual[:, ch], stage, ch)
                quant_stage[:, ch] = quant
                losses.append(F.mse_loss(quant, residual[:, ch].detach()) + self.beta * F.mse_loss(quant.detach(), residual[:, ch]))
                idx_stage.append(idx)
            quant_total = quant_total + quant_stage
            residual = residual - quant_stage.detach()
            idxs.append(torch.stack(idx_stage, dim=1))
            stage_losses.append(torch.stack(losses).mean())
        quant_st = x + (quant_total - x).detach()
        return quant_st.reshape(bsz, channels, h, w), torch.stack(idxs, dim=-1), torch.stack(stage_losses).sum(), quant_total

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_input(x)
        bsz, channels, h, w = x.shape
        residual = x
        quant_total = torch.zeros_like(x)
        idxs = []
        for stage in range(self.stages):
            quant_stage = torch.zeros_like(x)
            idx_stage = []
            for ch in range(channels):
                quant, idx = self._nearest(residual[:, ch], stage, ch)
                quant_stage[:, ch] = quant
                idx_stage.append(idx)
            quant_total = quant_total + quant_stage
            residual = residual - quant_stage
            idxs.append(torch.stack(idx_stage, dim=1))
        return quant_total.reshape(bsz, channels, h, w), torch.stack(idxs, dim=-1)

    @torch.no_grad()
    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 3:
            raise RuntimeError(f"expected per-channel RVQ idx [B,C,S], got {tuple(idx.shape)}")
        bsz, channels, stages = idx.shape
        if channels != self.channels or stages != self.stages:
            raise RuntimeError(f"expected per-channel RVQ idx [B,{self.channels},{self.stages}], got {tuple(idx.shape)}")
        out = torch.zeros(bsz, self.channels, self.h, self.w, device=idx.device, dtype=self.codebooks.dtype)
        for stage in range(self.stages):
            for ch in range(self.channels):
                out[:, ch] = out[:, ch] + self.codebooks[stage, ch][idx[:, ch, stage].long()]
        return out

    @torch.no_grad()
    def init_from_channel_samples(self, samples: torch.Tensor) -> float:
        samples = samples.detach().float().to(device=self.codebooks.device)
        self._check_input(samples)
        residual = samples
        quant_total = torch.zeros_like(samples)
        for stage in range(self.stages):
            quant_stage = torch.zeros_like(samples)
            for ch in range(self.channels):
                n = residual.shape[0]
                pick = (
                    torch.randperm(n, device=residual.device)[: self.num_codes]
                    if n >= self.num_codes
                    else torch.randint(0, n, (self.num_codes,), device=residual.device)
                )
                self.codebooks[stage, ch].copy_(residual[pick, ch].to(dtype=self.codebooks.dtype))
                quant, _idx = self._nearest(residual[:, ch], stage, ch)
                quant_stage[:, ch] = quant
            quant_total = quant_total + quant_stage
            residual = residual - quant_stage
        return float(F.mse_loss(quant_total.float(), samples.float()).item())

    @torch.no_grad()
    def init_from_channel_samples_kmeans(self, samples: torch.Tensor, iters: int = 25, chunk_size: int = 1024) -> float:
        samples = samples.detach().float().to(device=self.codebooks.device)
        self._check_input(samples)
        residual = samples
        quant_total = torch.zeros_like(samples)
        for stage in range(self.stages):
            quant_stage = torch.zeros_like(samples)
            for ch in range(self.channels):
                centers, _mse = _kmeans_centers(residual[:, ch], self.num_codes, iters, chunk_size)
                self.codebooks[stage, ch].copy_(centers.reshape(self.num_codes, self.h, self.w).to(dtype=self.codebooks.dtype))
                quant, _idx = self._nearest(residual[:, ch], stage, ch)
                quant_stage[:, ch] = quant
            quant_total = quant_total + quant_stage
            residual = residual - quant_stage
        return float(F.mse_loss(quant_total.float(), samples.float()).item())

    def usage_loss(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        self._check_input(x)
        residual = x.float()
        stage_losses = []
        for stage in range(self.stages):
            quant_stage = torch.zeros_like(residual)
            losses = []
            for ch in range(self.channels):
                cb = self.codebooks[stage, ch].float().flatten(1)
                x2 = residual[:, ch].flatten(1)
                dist = x2.square().sum(dim=1, keepdim=True) + cb.square().sum(dim=1).view(1, -1) - 2.0 * x2 @ cb.t()
                prob = torch.softmax(-dist / max(float(tau), 1e-6), dim=1)
                avg = prob.mean(dim=0).clamp_min(1e-8)
                losses.append((avg * (avg * avg.numel()).log()).sum())
                with torch.no_grad():
                    idx = dist.argmin(dim=1)
                    quant_stage[:, ch] = self.codebooks[stage, ch][idx].float()
            stage_losses.append(torch.stack(losses).mean())
            residual = residual - quant_stage
        return torch.stack(stage_losses).sum()

class PatchChannelCVQ(nn.Module):
    def __init__(self, num_codes: int, h: int, w: int, patch_size: int = 4, beta: float = 0.25, chunk_size: int = 128):
        super().__init__()
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.patch_size = int(patch_size)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        if self.h % self.patch_size != 0 or self.w % self.patch_size != 0:
            raise ValueError(f"patch_size={self.patch_size} must divide {(self.h, self.w)}")
        self.num_patches = (self.h // self.patch_size) * (self.w // self.patch_size)
        self.codebook = nn.Parameter(torch.randn(self.num_codes, self.patch_size, self.patch_size) * 0.02)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        n, h, w = x.shape
        if (h, w) != (self.h, self.w):
            raise ValueError(f"patch CVQ size {(self.h, self.w)} does not match input {(h, w)}")
        patches = F.unfold(x.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        return patches.reshape(n * self.num_patches, self.patch_size, self.patch_size)

    def _from_patches(self, patches: torch.Tensor, n: int) -> torch.Tensor:
        p = patches.reshape(n, self.num_patches, self.patch_size * self.patch_size).transpose(1, 2)
        return F.fold(p, output_size=(self.h, self.w), kernel_size=self.patch_size, stride=self.patch_size).squeeze(1)

    def _nearest(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _nearest_codebook(patches, self.codebook, self.chunk_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, channels, h, w = x.shape
        flat = x.reshape(bsz * channels, h, w)
        patches = self._to_patches(flat)
        quant_patches, idx = self._nearest(patches)
        codebook_loss = F.mse_loss(quant_patches, patches.detach())
        commit_loss = F.mse_loss(quant_patches.detach(), patches)
        quant = self._from_patches(quant_patches, flat.shape[0])
        quant_st = flat + (quant - flat).detach()
        return (
            quant_st.reshape(bsz, channels, h, w),
            idx.reshape(bsz, channels, self.num_patches),
            codebook_loss + self.beta * commit_loss,
            quant.reshape(bsz, channels, h, w),
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, channels, h, w = x.shape
        flat = x.reshape(bsz * channels, h, w)
        quant_patches, idx = self._nearest(self._to_patches(flat))
        quant = self._from_patches(quant_patches, flat.shape[0])
        return quant.reshape(bsz, channels, h, w), idx.reshape(bsz, channels, self.num_patches)

    @torch.no_grad()
    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, channels, num_patches = idx.shape
        if num_patches != self.num_patches:
            raise RuntimeError(f"expected patch tokens={self.num_patches}, got {num_patches}")
        patches = self.codebook[idx.reshape(-1)]
        return self._from_patches(patches, bsz * channels).reshape(bsz, channels, self.h, self.w)

    @torch.no_grad()
    def init_from_samples(self, samples: torch.Tensor) -> None:
        samples = samples.detach().float().to(device=self.codebook.device)
        patches = self._to_patches(samples)
        n = patches.shape[0]
        pick = torch.randperm(n, device=patches.device)[: self.num_codes] if n >= self.num_codes else torch.randint(0, n, (self.num_codes,), device=patches.device)
        self.codebook.copy_(patches[pick].to(dtype=self.codebook.dtype))

    @torch.no_grad()
    def init_from_samples_kmeans(self, samples: torch.Tensor, iters: int = 25, chunk_size: int = 1024) -> float:
        samples = samples.detach().float().to(device=self.codebook.device)
        patches = self._to_patches(samples)
        centers, mse = _kmeans_centers(patches, self.num_codes, iters, chunk_size)
        self.codebook.copy_(centers.reshape(self.num_codes, self.patch_size, self.patch_size).to(dtype=self.codebook.dtype))
        return mse

    def usage_loss(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        patches = self._to_patches(x.float().reshape(-1, self.h, self.w))
        cb = self.codebook.float().flatten(1)
        p = patches.flatten(1)
        dist = p.square().sum(dim=1, keepdim=True) + cb.square().sum(dim=1).view(1, -1) - 2.0 * p @ cb.t()
        prob = torch.softmax(-dist / max(float(tau), 1e-6), dim=1)
        avg = prob.mean(dim=0).clamp_min(1e-8)
        return (avg * (avg * avg.numel()).log()).sum()

class ScalarFSQChannel(nn.Module):
    def __init__(self, levels: int, h: int, w: int, beta: float = 0.25, init_scale: float = 3.0):
        super().__init__()
        self.num_codes = int(levels)
        self.h = int(h)
        self.w = int(w)
        self.beta = float(beta)
        if self.num_codes < 2:
            raise ValueError(f"FSQ levels must be >= 2, got {self.num_codes}")
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(init_scale))))

    def _scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp_min(1e-3)

    def level_values(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        scale = self._scale()
        if device is not None:
            scale = scale.to(device=device)
        if dtype is not None:
            scale = scale.to(dtype=dtype)
        return torch.linspace(-1.0, 1.0, self.num_codes, device=scale.device, dtype=scale.dtype) * scale

    def _quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale().to(device=x.device, dtype=x.float().dtype)
        y = (x.float() / scale).clamp(-1.0, 1.0)
        pos = (y + 1.0) * 0.5 * float(self.num_codes - 1)
        idx = pos.round().long().clamp(0, self.num_codes - 1)
        q = ((idx.float() / float(self.num_codes - 1)) * 2.0 - 1.0) * scale
        return q.to(dtype=x.dtype), idx

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, channels, h, w = x.shape
        if (h, w) != (self.h, self.w):
            raise ValueError(f"FSQ size {(self.h, self.w)} does not match input {(h, w)}")
        quant, idx = self._quantize(x)
        codebook_loss = F.mse_loss(quant, x.detach())
        commit_loss = F.mse_loss(quant.detach(), x)
        loss = codebook_loss + self.beta * commit_loss
        quant_st = x + (quant - x).detach()
        return quant_st, idx.reshape(bsz, channels, h, w), loss, quant

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quant, idx = self._quantize(x)
        return quant, idx

    @torch.no_grad()
    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        values = self.level_values(device=idx.device, dtype=torch.float32)
        return values[idx.long()]

    @torch.no_grad()
    def init_from_samples(self, samples: torch.Tensor) -> None:
        samples = samples.detach().float().to(device=self.log_scale.device)
        scale = torch.quantile(samples.abs().flatten(), 0.995).clamp_min(1e-3)
        self.log_scale.copy_(scale.log())

    @torch.no_grad()
    def init_from_samples_kmeans(self, samples: torch.Tensor, iters: int = 25, chunk_size: int = 1024) -> float:
        del iters, chunk_size
        self.init_from_samples(samples)
        quant, _idx = self._quantize(samples.to(device=self.log_scale.device))
        return float(F.mse_loss(quant.float(), samples.to(device=self.log_scale.device).float()).item())

    def usage_loss(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        values = torch.linspace(-1.0, 1.0, self.num_codes, device=x.device, dtype=x.float().dtype) * self._scale().to(device=x.device)
        dist = (x.float().reshape(-1, 1) - values.view(1, -1)).square()
        prob = torch.softmax(-dist / max(float(tau), 1e-6), dim=1)
        avg = prob.mean(dim=0).clamp_min(1e-8)
        return (avg * (avg * avg.numel()).log()).sum()

class TailCVQ(nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
        tail_ch: int = 20,
        k_a: int = 4096,
        k_b: int = 2048,
        beta: float = 0.25,
        chunk_size: int = 128,
        mode: str = "single",
        rvq_stages: int = 2,
        patch_size: int = 4,
        fsq_levels_a: int = 16,
        fsq_levels_b: int = 16,
        fsq_scale: float = 3.0,
    ):
        super().__init__()
        self.h = int(h)
        self.w = int(w)
        self.tail_ch = int(tail_ch)
        self.mode = str(mode)
        if self.tail_ch < 2:
            raise ValueError(f"tail_ch must be >= 2, got {self.tail_ch}")
        self.split_a = self.tail_ch // 2
        self.split_b = self.tail_ch - self.split_a
        if self.mode == "single":
            self.cvq_a = SingleChannelCVQ(k_a, h, w, beta=beta, chunk_size=chunk_size)
            self.cvq_b = SingleChannelCVQ(k_b, h, w, beta=beta, chunk_size=chunk_size)
        elif self.mode == "rvq":
            self.cvq_a = ResidualChannelCVQ(k_a, h, w, beta=beta, chunk_size=chunk_size, stages=rvq_stages)
            self.cvq_b = ResidualChannelCVQ(k_b, h, w, beta=beta, chunk_size=chunk_size, stages=rvq_stages)
        elif self.mode == "rvq_per_channel":
            self.needs_tail_channel_samples = True
            self.cvq_a = PerChannelResidualCVQ(
                k_a,
                h,
                w,
                channels=self.split_a,
                beta=beta,
                chunk_size=chunk_size,
                stages=rvq_stages,
            )
            self.cvq_b = PerChannelResidualCVQ(
                k_b,
                h,
                w,
                channels=self.split_b,
                beta=beta,
                chunk_size=chunk_size,
                stages=rvq_stages,
            )
        elif self.mode == "patch":
            self.cvq_a = PatchChannelCVQ(k_a, h, w, patch_size=patch_size, beta=beta, chunk_size=chunk_size)
            self.cvq_b = PatchChannelCVQ(k_b, h, w, patch_size=patch_size, beta=beta, chunk_size=chunk_size)
        elif self.mode == "fsq":
            self.cvq_a = ScalarFSQChannel(fsq_levels_a, h, w, beta=beta, init_scale=fsq_scale)
            self.cvq_b = ScalarFSQChannel(fsq_levels_b, h, w, beta=beta, init_scale=fsq_scale)
        elif self.mode == "fsq_shared":
            self.needs_tail_channel_samples = True
            self.cvq_shared = ScalarFSQChannel(fsq_levels_a, h, w, beta=beta, init_scale=fsq_scale)
            self.cvq_a = self.cvq_shared
            self.cvq_b = self.cvq_shared
        else:
            raise ValueError(f"unknown cvq mode: {self.mode}")

    def forward(self, tail: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if tail.shape[1] != self.tail_ch:
            raise RuntimeError(f"expected tail channels={self.tail_ch}, got {tail.shape[1]}")
        if self.mode == "fsq_shared":
            q, idx, loss, q_raw = self.cvq_shared(tail)
            aux = {
                "idx_a": idx[:, : self.split_a],
                "idx_b": idx[:, self.split_a :],
                "qa_raw": q_raw[:, : self.split_a],
                "qb_raw": q_raw[:, self.split_a :],
            }
            return q, idx, loss, aux
        tail_a = tail[:, : self.split_a]
        tail_b = tail[:, self.split_a :]
        qa, idx_a, loss_a, qa_raw = self.cvq_a(tail_a)
        qb, idx_b, loss_b, qb_raw = self.cvq_b(tail_b)
        idx = torch.cat([idx_a, idx_b], dim=1)
        aux = {"idx_a": idx_a, "idx_b": idx_b, "qa_raw": qa_raw, "qb_raw": qb_raw}
        return torch.cat([qa, qb], dim=1), idx, loss_a + loss_b, aux

    @torch.no_grad()
    def encode(self, tail: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if tail.shape[1] != self.tail_ch:
            raise RuntimeError(f"expected tail channels={self.tail_ch}, got {tail.shape[1]}")
        if self.mode == "fsq_shared":
            q, idx = self.cvq_shared.encode(tail)
            return q, idx, {"idx_a": idx[:, : self.split_a], "idx_b": idx[:, self.split_a :]}
        qa, idx_a = self.cvq_a.encode(tail[:, : self.split_a])
        qb, idx_b = self.cvq_b.encode(tail[:, self.split_a :])
        aux = {"idx_a": idx_a, "idx_b": idx_b}
        return torch.cat([qa, qb], dim=1), torch.cat([idx_a, idx_b], dim=1), aux

    @torch.no_grad()
    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.shape[1] != self.tail_ch:
            raise RuntimeError(f"expected index channels={self.tail_ch}, got {idx.shape[1]}")
        if self.mode == "fsq_shared":
            return self.cvq_shared.decode_indices(idx)
        qa = self.cvq_a.decode_indices(idx[:, : self.split_a])
        qb = self.cvq_b.decode_indices(idx[:, self.split_a :])
        return torch.cat([qa, qb], dim=1)

    def usage_loss(self, tail: torch.Tensor, tau: float) -> torch.Tensor:
        if self.mode == "fsq_shared":
            return self.cvq_shared.usage_loss(tail, tau)
        return self.cvq_a.usage_loss(tail[:, : self.split_a], tau) + self.cvq_b.usage_loss(tail[:, self.split_a :], tau)

    @torch.no_grad()
    def init_from_tail_samples(self, samples: torch.Tensor) -> dict[str, float]:
        samples = samples.detach().float()
        if samples.ndim != 4:
            raise ValueError(f"expected tail samples [N,C,H,W], got {tuple(samples.shape)}")
        if samples.shape[1] != self.tail_ch or tuple(samples.shape[2:]) != (self.h, self.w):
            raise ValueError(f"tail sample shape {tuple(samples.shape)} does not match [N,{self.tail_ch},{self.h},{self.w}]")
        if self.mode == "rvq_per_channel":
            mse_a = self.cvq_a.init_from_channel_samples(samples[:, : self.split_a])
            mse_b = self.cvq_b.init_from_channel_samples(samples[:, self.split_a :])
        elif self.mode == "fsq_shared":
            self.cvq_shared.init_from_samples(samples.reshape(-1, self.h, self.w))
            q, _idx = self.cvq_shared.encode(samples.to(device=self.cvq_shared.log_scale.device))
            mse_a = float(F.mse_loss(q[:, : self.split_a].float(), samples[:, : self.split_a].to(device=q.device).float()).item())
            mse_b = float(F.mse_loss(q[:, self.split_a :].float(), samples[:, self.split_a :].to(device=q.device).float()).item())
        else:
            raise RuntimeError(f"init_from_tail_samples is only used by tail-preserving modes, got {self.mode}")
        return {"sample_a_init_quant_mse": float(mse_a), "sample_b_init_quant_mse": float(mse_b)}

    @torch.no_grad()
    def init_from_tail_samples_kmeans(self, samples: torch.Tensor, iters: int = 25, chunk_size: int = 1024) -> dict[str, float]:
        samples = samples.detach().float()
        if samples.ndim != 4:
            raise ValueError(f"expected tail samples [N,C,H,W], got {tuple(samples.shape)}")
        if samples.shape[1] != self.tail_ch or tuple(samples.shape[2:]) != (self.h, self.w):
            raise ValueError(f"tail sample shape {tuple(samples.shape)} does not match [N,{self.tail_ch},{self.h},{self.w}]")
        if self.mode == "rvq_per_channel":
            mse_a = self.cvq_a.init_from_channel_samples_kmeans(
                samples[:, : self.split_a],
                iters=iters,
                chunk_size=chunk_size,
            )
            mse_b = self.cvq_b.init_from_channel_samples_kmeans(
                samples[:, self.split_a :],
                iters=iters,
                chunk_size=chunk_size,
            )
        elif self.mode == "fsq_shared":
            _mse_all = self.cvq_shared.init_from_samples_kmeans(
                samples.reshape(-1, self.h, self.w),
                iters=iters,
                chunk_size=chunk_size,
            )
            q, _idx = self.cvq_shared.encode(samples.to(device=self.cvq_shared.log_scale.device))
            mse_a = float(F.mse_loss(q[:, : self.split_a].float(), samples[:, : self.split_a].to(device=q.device).float()).item())
            mse_b = float(F.mse_loss(q[:, self.split_a :].float(), samples[:, self.split_a :].to(device=q.device).float()).item())
        else:
            raise RuntimeError(f"init_from_tail_samples_kmeans is only used by tail-preserving modes, got {self.mode}")
        return {"sample_a_init_quant_mse": float(mse_a), "sample_b_init_quant_mse": float(mse_b)}

def stats_from_hist(hist: torch.Tensor, quant_mse_sum: float, quant_mse_count: int, prefix: str) -> dict[str, float]:
    hist = hist.detach().float().cpu()
    prob = hist / hist.sum().clamp_min(1.0)
    used = float((hist > 0).sum().item())
    nz = prob > 0
    perplexity = float(torch.exp(-(prob[nz] * prob[nz].log()).sum()).item()) if bool(nz.any()) else 0.0
    return {
        f"{prefix}_usage": used / float(hist.numel()),
        f"{prefix}_used_codes": used,
        f"{prefix}_perplexity": perplexity,
        f"{prefix}_quant_mse": float(quant_mse_sum) / max(1, int(quant_mse_count)),
    }

def append_sample_pool(pool: list[torch.Tensor], samples: torch.Tensor, max_vectors: int) -> None:
    current = sum(x.shape[0] for x in pool)
    if current >= int(max_vectors):
        return
    need = int(max_vectors) - current
    pool.append(samples.detach().cpu()[:need])

@torch.no_grad()
def restart_dead_codes(codebook: SingleChannelCVQ, hist: torch.Tensor, sample_pool: list[torch.Tensor]) -> int:
    if not sample_pool:
        return 0
    dead = hist.to(device=codebook.codebook.device) == 0
    count = int(dead.sum().item())
    if count == 0:
        return 0
    samples = torch.cat(sample_pool, dim=0).to(device=codebook.codebook.device, dtype=codebook.codebook.dtype)
    pick = torch.randint(0, samples.shape[0], (count,), device=codebook.codebook.device)
    codebook.codebook[dead] = samples[pick]
    return count

class TailCAR(nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
        tail_ch: int = 20,
        k_a: int = 4096,
        k_b: int = 2048,
        prefix_ch: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.h = int(h)
        self.w = int(w)
        self.tail_ch = int(tail_ch)
        if self.tail_ch < 2:
            raise ValueError(f"tail_ch must be >= 2, got {self.tail_ch}")
        self.split_a = self.tail_ch // 2
        self.split_b = self.tail_ch - self.split_a
        self.k_a = int(k_a)
        self.k_b = int(k_b)
        self.d_model = int(d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.token_a = nn.Embedding(k_a, d_model)
        self.token_b = nn.Embedding(k_b, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.tail_ch, d_model))
        self.prefix_net = nn.Sequential(
            nn.Conv2d(prefix_ch, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head_a = nn.Linear(d_model, k_a)
        self.head_b = nn.Linear(d_model, k_b)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.bos, std=0.02)

    def _teacher_inputs(self, idx: torch.Tensor) -> torch.Tensor:
        bsz = idx.shape[0]
        if idx.shape[1] != self.tail_ch:
            raise RuntimeError(f"expected teacher index channels={self.tail_ch}, got {idx.shape[1]}")
        parts = [self.bos.expand(bsz, 1, -1)]
        if idx.shape[1] > 1:
            prev_idx = idx[:, : self.tail_ch - 1]
            prev_parts = []
            n_a = min(self.split_a, prev_idx.shape[1])
            if n_a > 0:
                prev_parts.append(self.token_a(prev_idx[:, :n_a].clamp_max(self.k_a - 1)))
            if prev_idx.shape[1] > n_a:
                prev_parts.append(self.token_b(prev_idx[:, n_a:].clamp_max(self.k_b - 1)))
            parts.append(torch.cat(prev_parts, dim=1))
        return torch.cat(parts, dim=1)

    def forward(self, y_prefix: torch.Tensor, teacher_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if teacher_idx is None:
            return self.generate(y_prefix)
        x = self._teacher_inputs(teacher_idx)
        cond = self.prefix_net(y_prefix.float()).unsqueeze(1)
        x = x + self.pos + cond
        mask = torch.triu(torch.ones(self.tail_ch, self.tail_ch, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.transformer(x, mask=mask)
        return self.head_a(h[:, : self.split_a]), self.head_b(h[:, self.split_a :])

    @torch.no_grad()
    def generate(self, y_prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = y_prefix.shape[0]
        idx = torch.zeros(bsz, self.tail_ch, device=y_prefix.device, dtype=torch.long)
        for t in range(self.tail_ch):
            logits_a, logits_b = self.forward(y_prefix, idx)
            if t < self.split_a:
                idx[:, t] = logits_a[:, t].argmax(dim=-1)
            else:
                idx[:, t] = logits_b[:, t - self.split_a].argmax(dim=-1)
        return idx[:, : self.split_a], idx[:, self.split_a :]

    def ce_loss(self, logits_a: torch.Tensor, logits_b: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        loss_a = F.cross_entropy(logits_a.reshape(-1, self.k_a), idx[:, : self.split_a].reshape(-1))
        loss_b = F.cross_entropy(logits_b.reshape(-1, self.k_b), idx[:, self.split_a :].reshape(-1))
        return loss_a + loss_b

    def soft_decode(self, logits_a: torch.Tensor, logits_b: torch.Tensor, cvq: TailCVQ, tau: float = 1.0) -> torch.Tensor:
        pa = F.softmax(logits_a.float() / float(tau), dim=-1)
        pb = F.softmax(logits_b.float() / float(tau), dim=-1)
        ca = cvq.cvq_a.codebook.float().flatten(1)
        cb = cvq.cvq_b.codebook.float().flatten(1)
        qa = torch.einsum("btk,kh->bth", pa, ca).reshape(logits_a.shape[0], self.split_a, cvq.h, cvq.w)
        qb = torch.einsum("btk,kh->bth", pb, cb).reshape(logits_b.shape[0], self.split_b, cvq.h, cvq.w)
        return torch.cat([qa, qb], dim=1)


class FSQSpatialCAR(nn.Module):
    is_fsq_spatial = True

    def __init__(
        self,
        h: int,
        w: int,
        tail_ch: int = 20,
        levels_a: int = 16,
        levels_b: int = 16,
        prefix_ch: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
        window_size: int = 4,
        prefix_image_cond: bool = False,
        prefix_image_scale_init: float = 0.1,
    ):
        super().__init__()
        self.h = int(h)
        self.w = int(w)
        self.tail_ch = int(tail_ch)
        if self.tail_ch < 2:
            raise ValueError(f"tail_ch must be >= 2, got {self.tail_ch}")
        self.split_a = self.tail_ch // 2
        self.split_b = self.tail_ch - self.split_a
        self.k_a = int(levels_a)
        self.k_b = int(levels_b)
        self.d_model = int(d_model)
        self.window_size = int(window_size)
        self.prefix_image_cond = bool(prefix_image_cond)
        if self.k_a < 2 or self.k_b < 2:
            raise ValueError(f"FSQ levels must be >= 2, got A={self.k_a}, B={self.k_b}")
        if self.h % self.window_size != 0 or self.w % self.window_size != 0:
            raise ValueError(f"car_window_size={self.window_size} must divide latent size {(self.h, self.w)}")

        self.prefix_net = nn.Sequential(
            nn.Conv2d(prefix_ch, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        if self.prefix_image_cond:
            image_hidden = max(16, d_model // 4)
            self.prefix_image_net = nn.Sequential(
                nn.Conv2d(3, image_hidden, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(image_hidden, image_hidden, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(image_hidden, image_hidden, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(image_hidden, image_hidden, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(image_hidden, d_model, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.prefix_image_scale = nn.Parameter(torch.tensor(float(prefix_image_scale_init)))
        self.idx_embed_a = nn.Embedding(self.k_a, d_model)
        self.idx_embed_b = nn.Embedding(self.k_b, d_model)
        self.prev_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.prev_scale = nn.Parameter(torch.tensor(0.1))
        self.channel_pos = nn.Parameter(torch.zeros(1, self.tail_ch, d_model))
        self.drop = nn.Dropout(float(dropout))
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=d_model,
                    input_resolution=(self.h, self.w),
                    num_heads=nhead,
                    window_size=self.window_size,
                    shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                    mlp_ratio=4.0,
                )
                for i in range(int(layers))
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head_a = nn.Linear(d_model, self.k_a)
        self.head_b = nn.Linear(d_model, self.k_b)
        self.value_head_a = nn.Linear(d_model, 1)
        self.value_head_b = nn.Linear(d_model, 1)
        nn.init.trunc_normal_(self.channel_pos, std=0.02)
        nn.init.zeros_(self.value_head_a.weight)
        nn.init.zeros_(self.value_head_a.bias)
        nn.init.zeros_(self.value_head_b.weight)
        nn.init.zeros_(self.value_head_b.bias)

    def _embed_idx(self, idx_t: torch.Tensor, t: int) -> torch.Tensor:
        if int(t) < self.split_a:
            emb = self.idx_embed_a(idx_t.clamp(0, self.k_a - 1))
        else:
            emb = self.idx_embed_b(idx_t.clamp(0, self.k_b - 1))
        return emb.permute(0, 3, 1, 2).contiguous()

    def _condition(self, y_prefix: torch.Tensor, x_prefix: torch.Tensor | None = None) -> torch.Tensor:
        cond = self.prefix_net(y_prefix.float())
        if self.prefix_image_cond and x_prefix is not None:
            img_cond = self.prefix_image_net(x_prefix.float())
            if img_cond.shape[-2:] != cond.shape[-2:]:
                img_cond = F.interpolate(img_cond, size=cond.shape[-2:], mode="bilinear", align_corners=False)
            cond = cond + self.prefix_image_scale * img_cond
        return cond

    def _step_feat(self, context: torch.Tensor, t: int) -> torch.Tensor:
        bsz = context.shape[0]
        tokens = context.flatten(2).transpose(1, 2)
        tokens = tokens + self.channel_pos[:, int(t) : int(t) + 1]
        tokens = self.drop(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens.reshape(bsz, self.h, self.w, self.d_model)

    def _step_logits(self, context: torch.Tensor, t: int) -> torch.Tensor:
        feat = self._step_feat(context, t)
        if int(t) < self.split_a:
            return self.head_a(feat)
        return self.head_b(feat)

    def _step_logits_value(self, context: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self._step_feat(context, t)
        if int(t) < self.split_a:
            return self.head_a(feat), self.value_head_a(feat).squeeze(-1)
        return self.head_b(feat), self.value_head_b(feat).squeeze(-1)

    def _update_context(self, context: torch.Tensor, idx_t: torch.Tensor, t: int) -> torch.Tensor:
        return context + self.prev_scale * self.prev_proj(self._embed_idx(idx_t, int(t)).float())

    def _logits_to_idx(self, logits: torch.Tensor, k: int, mode: str) -> torch.Tensor:
        if str(mode) == "mean":
            prob = F.softmax(logits.float(), dim=-1)
            levels = torch.arange(int(k), device=logits.device, dtype=prob.dtype)
            return torch.sum(prob * levels, dim=-1).round().long().clamp(0, int(k) - 1)
        if str(mode) != "argmax":
            raise ValueError(f"unknown FSQ generate mode: {mode}")
        return logits.argmax(dim=-1)

    def _stack_logits_from_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits_a = []
        logits_b = []
        for t in range(self.tail_ch):
            logits = self._step_logits(context, t)
            if t < self.split_a:
                logits_a.append(logits)
            else:
                logits_b.append(logits)
        return torch.stack(logits_a, dim=1), torch.stack(logits_b, dim=1)

    def hard_indices_from_logits(self, logits_a: torch.Tensor, logits_b: torch.Tensor, mode: str = "argmax") -> torch.Tensor:
        idx_a = self._logits_to_idx(logits_a, self.k_a, mode)
        idx_b = self._logits_to_idx(logits_b, self.k_b, mode)
        return torch.cat([idx_a, idx_b], dim=1)

    def confidence_from_logits(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        idx: torch.Tensor | None = None,
        mode: str = "argmax",
    ) -> torch.Tensor:
        if idx is None:
            idx = self.hard_indices_from_logits(logits_a, logits_b, mode=mode)
        pa = F.softmax(logits_a.float(), dim=-1)
        pb = F.softmax(logits_b.float(), dim=-1)
        conf_a = pa.gather(-1, idx[:, : self.split_a].long().unsqueeze(-1)).squeeze(-1)
        conf_b = pb.gather(-1, idx[:, self.split_a :].long().unsqueeze(-1)).squeeze(-1)
        return torch.cat([conf_a, conf_b], dim=1)

    def forward(self, y_prefix: torch.Tensor, teacher_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if teacher_idx is None:
            return self.generate(y_prefix)
        if teacher_idx.ndim != 4 or teacher_idx.shape[1:] != (self.tail_ch, self.h, self.w):
            raise RuntimeError(f"expected FSQ teacher idx [B,{self.tail_ch},{self.h},{self.w}], got {tuple(teacher_idx.shape)}")
        context = self._condition(y_prefix)
        logits_a = []
        logits_b = []
        for t in range(self.tail_ch):
            logits = self._step_logits(context, t)
            if t < self.split_a:
                logits_a.append(logits)
            else:
                logits_b.append(logits)
            if t + 1 < self.tail_ch:
                context = self._update_context(context, teacher_idx[:, t].long(), t)
        return torch.stack(logits_a, dim=1), torch.stack(logits_b, dim=1)

    @torch.no_grad()
    def generate(self, y_prefix: torch.Tensor, mode: str = "argmax") -> tuple[torch.Tensor, torch.Tensor]:
        idx, _conf = self.generate_with_confidence(y_prefix, mode=mode)
        return idx[:, : self.split_a], idx[:, self.split_a :]

    @torch.no_grad()
    def generate_with_confidence(self, y_prefix: torch.Tensor, mode: str = "argmax") -> tuple[torch.Tensor, torch.Tensor]:
        bsz = y_prefix.shape[0]
        context = self._condition(y_prefix)
        idx = torch.zeros(bsz, self.tail_ch, self.h, self.w, device=y_prefix.device, dtype=torch.long)
        conf = torch.zeros(bsz, self.tail_ch, self.h, self.w, device=y_prefix.device, dtype=y_prefix.float().dtype)
        for t in range(self.tail_ch):
            logits = self._step_logits(context, t)
            k = self.k_a if t < self.split_a else self.k_b
            idx_t = self._logits_to_idx(logits, k, mode)
            idx[:, t] = idx_t
            prob = F.softmax(logits.float(), dim=-1)
            conf[:, t] = prob.gather(-1, idx_t.unsqueeze(-1)).squeeze(-1)
            if t + 1 < self.tail_ch:
                context = self._update_context(context, idx[:, t], t)
        return idx, conf

    def masked_forward(
        self,
        y_prefix: torch.Tensor,
        visible_idx: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visible_idx.ndim != 4 or visible_idx.shape[1:] != (self.tail_ch, self.h, self.w):
            raise RuntimeError(f"expected visible_idx [B,{self.tail_ch},{self.h},{self.w}], got {tuple(visible_idx.shape)}")
        if visible_mask.shape != visible_idx.shape:
            raise RuntimeError(f"visible_mask shape {tuple(visible_mask.shape)} does not match {tuple(visible_idx.shape)}")
        context = self._condition(y_prefix)
        mask = visible_mask.float()
        for t in range(self.tail_ch):
            emb = self._embed_idx(visible_idx[:, t].long(), t).float()
            context = context + self.prev_scale * self.prev_proj(emb * mask[:, t : t + 1])
        return self._stack_logits_from_context(context)

    @torch.no_grad()
    def masked_generate(
        self,
        y_prefix: torch.Tensor,
        iterations: int = 6,
        mode: str = "argmax",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = y_prefix.shape[0]
        total = self.tail_ch * self.h * self.w
        steps = max(1, int(iterations))
        idx = torch.zeros(bsz, self.tail_ch, self.h, self.w, device=y_prefix.device, dtype=torch.long)
        visible = torch.zeros_like(idx, dtype=torch.bool)
        conf = torch.zeros_like(idx, dtype=y_prefix.float().dtype)
        for step in range(steps):
            logits_a, logits_b = self.masked_forward(y_prefix, idx, visible)
            pred = self.hard_indices_from_logits(logits_a, logits_b, mode=mode)
            pred_conf = self.confidence_from_logits(logits_a, logits_b, pred, mode=mode)
            idx = torch.where(visible, idx, pred)
            conf = torch.where(visible, conf, pred_conf)
            keep = total if step + 1 >= steps else max(1, int(round(total * float(step + 1) / float(steps))))
            flat_visible = visible.flatten(1)
            flat_conf = conf.flatten(1).masked_fill(flat_visible, 2.0)
            new_visible = torch.zeros_like(flat_visible)
            top = torch.topk(flat_conf, k=keep, dim=1).indices
            new_visible.scatter_(1, top, True)
            visible = new_visible.view_as(visible)
        return idx, conf

    def ce_loss(self, logits_a: torch.Tensor, logits_b: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 4:
            raise RuntimeError(f"expected FSQ idx [B,C,H,W], got {tuple(idx.shape)}")
        loss_a = F.cross_entropy(logits_a.reshape(-1, self.k_a), idx[:, : self.split_a].reshape(-1))
        loss_b = F.cross_entropy(logits_b.reshape(-1, self.k_b), idx[:, self.split_a :].reshape(-1))
        return loss_a + loss_b

    def masked_ce_loss(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.shape != idx.shape:
            raise RuntimeError(f"mask shape {tuple(mask.shape)} does not match idx shape {tuple(idx.shape)}")
        loss_a = F.cross_entropy(logits_a.reshape(-1, self.k_a), idx[:, : self.split_a].reshape(-1), reduction="none")
        loss_b = F.cross_entropy(logits_b.reshape(-1, self.k_b), idx[:, self.split_a :].reshape(-1), reduction="none")
        loss_a = loss_a.view_as(idx[:, : self.split_a])
        loss_b = loss_b.view_as(idx[:, self.split_a :])
        loss = torch.cat([loss_a, loss_b], dim=1)
        m = mask.float()
        return (loss * m).sum() / m.sum().clamp_min(1.0)

    def soft_decode(self, logits_a: torch.Tensor, logits_b: torch.Tensor, cvq: TailCVQ, tau: float = 1.0) -> torch.Tensor:
        pa = F.softmax(logits_a.float() / max(float(tau), 1e-6), dim=-1)
        pb = F.softmax(logits_b.float() / max(float(tau), 1e-6), dim=-1)
        va = cvq.cvq_a.level_values(device=logits_a.device, dtype=pa.dtype)
        vb = cvq.cvq_b.level_values(device=logits_b.device, dtype=pb.dtype)
        qa = torch.einsum("bchwk,k->bchw", pa, va)
        qb = torch.einsum("bchwk,k->bchw", pb, vb)
        return torch.cat([qa, qb], dim=1)


class FSQFactorizedCAR(FSQSpatialCAR):
    is_fsq_factorized = True

    def _factorized_feat(self, context: torch.Tensor) -> torch.Tensor:
        bsz = context.shape[0]
        tokens = context.flatten(2).transpose(1, 2).unsqueeze(1).expand(-1, self.tail_ch, -1, -1).contiguous()
        tokens = tokens + self.channel_pos[:, :, None, :]
        tokens = self.drop(tokens)
        tokens = tokens.reshape(bsz * self.tail_ch, self.h * self.w, self.d_model)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens.reshape(bsz, self.tail_ch, self.h, self.w, self.d_model)

    def _factorized_logits(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self._factorized_feat(context)
        return self.head_a(feat[:, : self.split_a]), self.head_b(feat[:, self.split_a :])

    def forward(self, y_prefix: torch.Tensor, teacher_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del teacher_idx
        return self._factorized_logits(self._condition(y_prefix))

    @torch.no_grad()
    def generate_with_confidence(self, y_prefix: torch.Tensor, mode: str = "argmax") -> tuple[torch.Tensor, torch.Tensor]:
        logits_a, logits_b = self._factorized_logits(self._condition(y_prefix))
        idx_a = self._logits_to_idx(logits_a, self.k_a, mode)
        idx_b = self._logits_to_idx(logits_b, self.k_b, mode)
        prob_a = F.softmax(logits_a.float(), dim=-1)
        prob_b = F.softmax(logits_b.float(), dim=-1)
        conf_a = prob_a.gather(-1, idx_a.unsqueeze(-1)).squeeze(-1)
        conf_b = prob_b.gather(-1, idx_b.unsqueeze(-1)).squeeze(-1)
        return torch.cat([idx_a, idx_b], dim=1), torch.cat([conf_a, conf_b], dim=1)

    @torch.no_grad()
    def masked_generate(
        self,
        y_prefix: torch.Tensor,
        iterations: int = 6,
        mode: str = "argmax",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del iterations
        return self.generate_with_confidence(y_prefix, mode=mode)

    def masked_forward(
        self,
        y_prefix: torch.Tensor,
        visible_idx: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del visible_idx, visible_mask
        return self.forward(y_prefix)
