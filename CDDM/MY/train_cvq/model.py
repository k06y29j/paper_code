from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def nearest_codebook(
    x: torch.Tensor,
    codebook: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cb = codebook.float().flatten(1)
    cb_norm = cb.square().sum(dim=1).view(1, -1)
    x2 = x.float().flatten(1)
    quants = []
    indices = []
    chunk = max(1, int(chunk_size))
    for start in range(0, x2.shape[0], chunk):
        q = x2[start : start + chunk]
        dist = q.square().sum(dim=1, keepdim=True) + cb_norm - 2.0 * q @ cb.t()
        idx = dist.argmin(dim=1)
        indices.append(idx)
        quants.append(codebook[idx])
    return torch.cat(quants, dim=0).to(dtype=x.dtype), torch.cat(indices, dim=0)


class FullChannelQuantizer(nn.Module):
    """Shared K-entry spatial codebook applied independently to every latent channel."""

    def __init__(
        self,
        num_codes: int = 16384,
        h: int = 16,
        w: int = 16,
        beta: float = 0.25,
        chunk_size: int = 128,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.codebook = nn.Parameter(torch.randn(self.num_codes, self.h, self.w) * 0.02)

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        if tuple(x.shape[2:]) != (self.h, self.w):
            raise ValueError(f"latent size {tuple(x.shape[2:])} does not match codebook {(self.h, self.w)}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check(x)
        bsz, channels, h, w = x.shape
        flat = x.reshape(bsz * channels, h, w)
        quant, idx = nearest_codebook(flat, self.codebook, self.chunk_size)
        codebook_loss = F.mse_loss(quant.float(), flat.detach().float())
        commit_loss = F.mse_loss(quant.detach().float(), flat.float())
        vq_loss = codebook_loss + self.beta * commit_loss
        quant_st = flat + (quant - flat).detach()
        return quant_st.reshape(bsz, channels, h, w), idx.reshape(bsz, channels), vq_loss, quant.reshape(bsz, channels, h, w)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check(x)
        bsz, channels, h, w = x.shape
        quant, idx = nearest_codebook(x.reshape(bsz * channels, h, w), self.codebook, self.chunk_size)
        return quant.reshape(bsz, channels, h, w), idx.reshape(bsz, channels)

    @torch.no_grad()
    def init_from_samples(self, samples: torch.Tensor) -> None:
        samples = samples.detach().float().cpu()
        if samples.ndim != 3:
            raise ValueError(f"expected samples [N,H,W], got {tuple(samples.shape)}")
        if tuple(samples.shape[1:]) != (self.h, self.w):
            raise ValueError(f"sample size {tuple(samples.shape[1:])} does not match {(self.h, self.w)}")
        n = int(samples.shape[0])
        if n < 1:
            raise ValueError("cannot initialize codebook from empty samples")
        if n >= self.num_codes:
            pick = torch.randperm(n)[: self.num_codes]
        else:
            pick = torch.randint(0, n, (self.num_codes,))
        self.codebook.copy_(samples[pick].to(device=self.codebook.device, dtype=self.codebook.dtype))

    @torch.no_grad()
    def stats(self, idx: torch.Tensor, quant: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        hist = torch.bincount(idx.reshape(-1).detach().cpu(), minlength=self.num_codes).float()
        prob = hist / hist.sum().clamp_min(1.0)
        used = int((hist > 0).sum().item())
        perplexity = float(torch.exp(-(prob[prob > 0] * prob[prob > 0].log()).sum()).item())
        return {
            "used_codes": float(used),
            "usage": used / float(self.num_codes),
            "perplexity": perplexity,
            "quant_mse": float(F.mse_loss(quant.float(), target.float()).item()),
        }
