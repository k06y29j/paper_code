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


class SimVQFullChannelQuantizer(nn.Module):
    """SimVQ-style shared spatial codebook.

    The base codebook is frozen by default. A trainable linear layer projects
    code vectors into the quantization space before nearest-neighbor lookup.
    """

    def __init__(
        self,
        num_codes: int = 16384,
        h: int = 16,
        w: int = 16,
        beta: float = 0.25,
        chunk_size: int = 128,
        codebook_std: float = 0.0,
        freeze_codebook: bool = True,
        legacy: bool = True,
        proj_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.legacy = bool(legacy)
        self.freeze_codebook = bool(freeze_codebook)
        self.dim = self.h * self.w
        self.default_codebook_std = float(codebook_std) if float(codebook_std) > 0.0 else float(self.dim) ** -0.5
        self.codebook = nn.Parameter(torch.empty(self.num_codes, self.h, self.w))
        nn.init.normal_(self.codebook, mean=0.0, std=self.default_codebook_std)
        self.codebook.requires_grad_(not self.freeze_codebook)
        self.embedding_proj = nn.Linear(self.dim, self.dim, bias=bool(proj_bias))

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        if tuple(x.shape[2:]) != (self.h, self.w):
            raise ValueError(f"latent size {tuple(x.shape[2:])} does not match codebook {(self.h, self.w)}")

    def effective_codebook(self) -> torch.Tensor:
        quant_codebook = self.embedding_proj(self.codebook.float().flatten(1))
        return quant_codebook.reshape(self.num_codes, self.h, self.w)

    def _nearest(self, flat: torch.Tensor, quant_codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cb = quant_codebook.float().flatten(1)
        cb_norm = cb.square().sum(dim=1).view(1, -1)
        x2 = flat.float().flatten(1)
        quants = []
        indices = []
        chunk = max(1, int(self.chunk_size))
        for start in range(0, x2.shape[0], chunk):
            q = x2[start : start + chunk]
            dist = q.square().sum(dim=1, keepdim=True) + cb_norm - 2.0 * q @ cb.t()
            idx = dist.argmin(dim=1)
            indices.append(idx)
            quants.append(quant_codebook[idx])
        return torch.cat(quants, dim=0).to(dtype=flat.dtype), torch.cat(indices, dim=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check(x)
        bsz, channels, h, w = x.shape
        flat = x.reshape(bsz * channels, h, w)
        quant_codebook = self.effective_codebook()
        quant, idx = self._nearest(flat, quant_codebook)
        codebook_loss = F.mse_loss(quant.float(), flat.detach().float())
        commit_loss = F.mse_loss(quant.detach().float(), flat.float())
        if self.legacy:
            vq_loss = commit_loss + self.beta * codebook_loss
        else:
            vq_loss = codebook_loss + self.beta * commit_loss
        quant_st = flat + (quant - flat).detach()
        return quant_st.reshape(bsz, channels, h, w), idx.reshape(bsz, channels), vq_loss, quant.reshape(bsz, channels, h, w)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check(x)
        bsz, channels, h, w = x.shape
        quant_codebook = self.effective_codebook()
        quant, idx = self._nearest(x.reshape(bsz * channels, h, w), quant_codebook)
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
            "simvq_codebook_trainable": float(self.codebook.requires_grad),
            "simvq_legacy": float(self.legacy),
        }


class EMAFullChannelQuantizer(nn.Module):
    """Shared spatial VQ with codebook vectors updated by EMA assignment means."""

    def __init__(
        self,
        num_codes: int = 16384,
        h: int = 16,
        w: int = 16,
        beta: float = 0.25,
        chunk_size: int = 128,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        ema_initial_count: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.ema_initial_count = float(ema_initial_count)
        self.codebook = nn.Parameter(torch.randn(self.num_codes, self.h, self.w) * 0.02, requires_grad=False)
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_codes))
        self.register_buffer("ema_codebook_sum", torch.zeros(self.num_codes, self.h, self.w))
        self.register_buffer("ema_initialized", torch.zeros((), dtype=torch.bool))

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        if tuple(x.shape[2:]) != (self.h, self.w):
            raise ValueError(f"latent size {tuple(x.shape[2:])} does not match codebook {(self.h, self.w)}")

    @torch.no_grad()
    def sync_ema_from_codebook(self) -> None:
        init_count = max(float(self.ema_initial_count), float(self.ema_eps))
        self.ema_cluster_size.fill_(init_count)
        self.ema_codebook_sum.copy_(self.codebook.detach().float() * init_count)
        self.ema_initialized.fill_(True)

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, idx: torch.Tensor) -> None:
        if not bool(self.ema_initialized.item()):
            self.sync_ema_from_codebook()
        flat_f = flat.detach().float()
        counts = torch.bincount(idx.detach(), minlength=self.num_codes).to(device=flat_f.device, dtype=flat_f.dtype)
        sums = torch.zeros(self.num_codes, self.h, self.w, device=flat_f.device, dtype=flat_f.dtype)
        sums.index_add_(0, idx.detach(), flat_f)
        decay = float(self.ema_decay)
        self.ema_cluster_size.mul_(decay).add_(counts, alpha=1.0 - decay)
        self.ema_codebook_sum.mul_(decay).add_(sums, alpha=1.0 - decay)
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + float(self.ema_eps)) / (n + self.num_codes * float(self.ema_eps)) * n
        updated = self.ema_codebook_sum / smoothed.clamp_min(float(self.ema_eps)).view(-1, 1, 1)
        self.codebook.copy_(updated.to(device=self.codebook.device, dtype=self.codebook.dtype))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check(x)
        bsz, channels, h, w = x.shape
        flat = x.reshape(bsz * channels, h, w)
        quant, idx = nearest_codebook(flat, self.codebook, self.chunk_size)
        if self.training:
            self._ema_update(flat, idx)
        commit_loss = F.mse_loss(quant.detach().float(), flat.float())
        vq_loss = self.beta * commit_loss
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
        self.sync_ema_from_codebook()

    @torch.no_grad()
    def stats(self, idx: torch.Tensor, quant: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        hist = torch.bincount(idx.reshape(-1).detach().cpu(), minlength=self.num_codes).float()
        prob = hist / hist.sum().clamp_min(1.0)
        used = int((hist > 0).sum().item())
        perplexity = float(torch.exp(-(prob[prob > 0] * prob[prob > 0].log()).sum()).item())
        active_ema = int((self.ema_cluster_size.detach().cpu() > float(self.ema_eps)).sum().item())
        return {
            "used_codes": float(used),
            "usage": used / float(self.num_codes),
            "perplexity": perplexity,
            "quant_mse": float(F.mse_loss(quant.float(), target.float()).item()),
            "ema_active_codes": float(active_ema),
            "ema_decay": float(self.ema_decay),
        }


class ScaledWhitenedVQ(nn.Module):
    """Whiten C2 with fixed channel stats before applying a shared spatial VQ."""

    def __init__(
        self,
        num_codes: int = 16384,
        channels: int = 20,
        h: int = 16,
        w: int = 16,
        beta: float = 0.25,
        chunk_size: int = 128,
        eps: float = 1e-6,
        codebook_update: str = "grad",
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        ema_initial_count: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.channels = int(channels)
        self.h = int(h)
        self.w = int(w)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.eps = float(eps)
        self.codebook_update = str(codebook_update)
        self.cache_mode = "scaled_whitened"
        if self.codebook_update == "ema":
            self.vq = EMAFullChannelQuantizer(
                num_codes=self.num_codes,
                h=self.h,
                w=self.w,
                beta=self.beta,
                chunk_size=self.chunk_size,
                ema_decay=ema_decay,
                ema_eps=ema_eps,
                ema_initial_count=ema_initial_count,
            )
        elif self.codebook_update == "grad":
            self.vq = FullChannelQuantizer(
                num_codes=self.num_codes,
                h=self.h,
                w=self.w,
                beta=self.beta,
                chunk_size=self.chunk_size,
            )
        else:
            raise ValueError(f"unknown codebook_update: {self.codebook_update}")
        self.register_buffer("running_mean", torch.zeros(1, self.channels, 1, 1))
        self.register_buffer("running_var", torch.ones(1, self.channels, 1, 1))
        self.register_buffer("stats_initialized", torch.zeros((), dtype=torch.bool))

    @property
    def codebook(self) -> nn.Parameter:
        return self.vq.codebook

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"expected C2 channels={self.channels}, got {x.shape[1]}")
        if tuple(x.shape[2:]) != (self.h, self.w):
            raise ValueError(f"latent size {tuple(x.shape[2:])} does not match codebook {(self.h, self.w)}")

    @torch.no_grad()
    def set_channel_stats(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        mean = mean.detach().float().reshape(1, self.channels, 1, 1)
        var = var.detach().float().reshape(1, self.channels, 1, 1).clamp_min(self.eps * self.eps)
        self.running_mean.copy_(mean.to(device=self.running_mean.device, dtype=self.running_mean.dtype))
        self.running_var.copy_(var.to(device=self.running_var.device, dtype=self.running_var.dtype))
        self.stats_initialized.fill_(True)

    def sigma(self) -> torch.Tensor:
        return torch.sqrt(self.running_var.clamp_min(self.eps * self.eps)) + self.eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        mean = self.running_mean.to(device=x.device, dtype=x.dtype)
        sigma = self.sigma().to(device=x.device, dtype=x.dtype)
        return (x - mean) / sigma

    def denormalize(self, u: torch.Tensor) -> torch.Tensor:
        self._check(u)
        mean = self.running_mean.to(device=u.device, dtype=u.dtype)
        sigma = self.sigma().to(device=u.device, dtype=u.dtype)
        return u * sigma + mean

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.normalize(x)
        quant_u, idx, vq_loss, raw_u = self.vq(u)
        return self.denormalize(quant_u), idx, vq_loss, self.denormalize(raw_u)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quant_u, idx = self.vq.encode(self.normalize(x))
        return self.denormalize(quant_u), idx

    @torch.no_grad()
    def init_from_samples(self, samples: torch.Tensor) -> None:
        self.vq.init_from_samples(samples)

    @torch.no_grad()
    def sync_ema_from_codebook(self) -> None:
        if hasattr(self.vq, "sync_ema_from_codebook"):
            self.vq.sync_ema_from_codebook()

    @torch.no_grad()
    def extract_codebook_samples(self, x: torch.Tensor) -> torch.Tensor:
        u = self.normalize(x)
        return u.detach().reshape(-1, self.h, self.w)

    @torch.no_grad()
    def stats(self, idx: torch.Tensor, quant: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        out = self.vq.stats(idx, quant, target)
        out["c2_running_mean_abs"] = float(self.running_mean.detach().abs().mean().item())
        out["c2_running_sigma_mean"] = float(torch.sqrt(self.running_var.detach().clamp_min(0.0)).mean().item())
        return out
