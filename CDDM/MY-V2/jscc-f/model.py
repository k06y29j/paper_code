from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CDDM_ROOT = THIS_DIR.parents[1]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

from Autoencoder.net.modules import PatchEmbed
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder

from common import freeze_module, total_latent_ch, z1_ch, z2_ch


def load_local_io():
    spec = importlib.util.spec_from_file_location("jsccf_model_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = load_local_io()


class OutputsCombiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 48, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(48, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x1, u2], dim=1))


def nearest_codebook(
    x: torch.Tensor,
    codebook: torch.Tensor,
    chunk_size: int = 128,
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


def select_fullmap_codebook_samples(samples: torch.Tensor, num_codes: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if samples.ndim != 3 or tuple(samples.shape[1:]) != (int(h), int(w)):
        raise ValueError(f"expected samples [N,{h},{w}], got {tuple(samples.shape)}")
    if samples.shape[0] < 1:
        raise ValueError("cannot initialize codebook from empty samples")
    samples = samples.detach().to(device=device, dtype=dtype)
    idx = torch.randint(0, int(samples.shape[0]), (int(num_codes),), device=device)
    return samples[idx].contiguous()


class VQQuantizer(nn.Module):
    """Per-location VQ over z2 channel vectors."""

    def __init__(
        self,
        num_codes: int = 64,
        embedding_dim: int = 20,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.embedding_dim = int(embedding_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.num_codes, self.embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=float(self.embedding_dim) ** -0.5)

    def effective_codebook(self) -> torch.Tensor:
        return self.embedding.weight

    @torch.no_grad()
    def initialize_from_vectors(self, vectors: torch.Tensor) -> None:
        if vectors.ndim != 2 or tuple(vectors.shape) != (self.num_codes, self.embedding_dim):
            raise ValueError(f"expected vectors [{self.num_codes},{self.embedding_dim}], got {tuple(vectors.shape)}")
        self.embedding.weight.copy_(vectors.to(device=self.embedding.weight.device, dtype=self.embedding.weight.dtype))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if z.ndim != 4:
            raise ValueError(f"expected z [B,C,H,W], got {tuple(z.shape)}")
        if int(z.shape[1]) != self.embedding_dim:
            raise ValueError(f"expected z channel dim {self.embedding_dim}, got {int(z.shape[1])}")

        bsz, _channels, h, w = z.shape
        z_hw = z.permute(0, 2, 3, 1).contiguous()
        flat = z_hw.reshape(-1, self.embedding_dim)
        codebook = self.effective_codebook()
        dist = (
            flat.float().square().sum(dim=1, keepdim=True)
            + codebook.float().square().sum(dim=1).view(1, -1)
            - 2.0 * flat.float() @ codebook.float().t()
        )
        indices = dist.argmin(dim=1)
        q_hw = F.embedding(indices, codebook).view(bsz, h, w, self.embedding_dim).to(dtype=z.dtype)
        q2 = q_hw.permute(0, 3, 1, 2).contiguous()
        q2_st = z + (q2 - z).detach()

        codebook_loss = F.mse_loss(q2.float(), z.detach().float())
        commit_loss = F.mse_loss(q2.detach().float(), z.float())
        vq_loss = codebook_loss + self.beta * commit_loss
        index_map = indices.view(bsz, h, w)
        stats = {
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
            "vq_loss": vq_loss,
            "vq_mse": F.mse_loss(q2.detach().float(), z.detach().float()),
        }
        return q2_st, q2, index_map, stats

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _q2_st, q2, indices, _stats = self.forward(z)
        return q2, indices

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 3:
            raise ValueError(f"expected indices [B,H,W], got {tuple(indices.shape)}")
        bsz, h, w = indices.shape
        q_hw = F.embedding(indices.long().reshape(-1), self.effective_codebook())
        return q_hw.view(bsz, h, w, self.embedding_dim).permute(0, 3, 1, 2).contiguous()


class SimVQQuantizer(nn.Module):
    """Per-location SimVQ over z2 channel vectors.

    This follows the SimVQ pattern: keep a base embedding table and pass it
    through a trainable linear layer before nearest-neighbor lookup.
    """

    def __init__(
        self,
        num_codes: int = 64,
        embedding_dim: int = 20,
        beta: float = 0.25,
        freeze_codebook: bool = True,
        proj_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.embedding_dim = int(embedding_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.num_codes, self.embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=float(self.embedding_dim) ** -0.5)
        self.embedding.weight.requires_grad_(not bool(freeze_codebook))
        self.embedding_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=bool(proj_bias))

    def effective_codebook(self) -> torch.Tensor:
        return self.embedding_proj(self.embedding.weight.float())

    @torch.no_grad()
    def initialize_from_vectors(self, vectors: torch.Tensor) -> None:
        if vectors.ndim != 2 or tuple(vectors.shape) != (self.num_codes, self.embedding_dim):
            raise ValueError(f"expected vectors [{self.num_codes},{self.embedding_dim}], got {tuple(vectors.shape)}")
        self.embedding.weight.copy_(vectors.to(device=self.embedding.weight.device, dtype=self.embedding.weight.dtype))
        eye = torch.eye(self.embedding_dim, device=self.embedding_proj.weight.device, dtype=self.embedding_proj.weight.dtype)
        self.embedding_proj.weight.copy_(eye)
        if self.embedding_proj.bias is not None:
            self.embedding_proj.bias.zero_()

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if z.ndim != 4:
            raise ValueError(f"expected z [B,C,H,W], got {tuple(z.shape)}")
        if int(z.shape[1]) != self.embedding_dim:
            raise ValueError(f"expected z channel dim {self.embedding_dim}, got {int(z.shape[1])}")

        bsz, _channels, h, w = z.shape
        z_hw = z.permute(0, 2, 3, 1).contiguous()
        flat = z_hw.reshape(-1, self.embedding_dim)
        codebook = self.effective_codebook()
        dist = (
            flat.float().square().sum(dim=1, keepdim=True)
            + codebook.float().square().sum(dim=1).view(1, -1)
            - 2.0 * flat.float() @ codebook.float().t()
        )
        indices = dist.argmin(dim=1)
        q_hw = F.embedding(indices, codebook).view(bsz, h, w, self.embedding_dim).to(dtype=z.dtype)
        q2 = q_hw.permute(0, 3, 1, 2).contiguous()
        q2_st = z + (q2 - z).detach()

        codebook_loss = F.mse_loss(q2.float(), z.detach().float())
        commit_loss = F.mse_loss(q2.detach().float(), z.float())
        vq_loss = codebook_loss + self.beta * commit_loss
        index_map = indices.view(bsz, h, w)
        stats = {
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
            "vq_loss": vq_loss,
            "vq_mse": F.mse_loss(q2.detach().float(), z.detach().float()),
        }
        return q2_st, q2, index_map, stats

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _q2_st, q2, indices, _stats = self.forward(z)
        return q2, indices

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 3:
            raise ValueError(f"expected indices [B,H,W], got {tuple(indices.shape)}")
        bsz, h, w = indices.shape
        q_hw = F.embedding(indices.long().reshape(-1), self.effective_codebook())
        return q_hw.view(bsz, h, w, self.embedding_dim).permute(0, 3, 1, 2).contiguous()


def parse_fsq_levels(levels: str | int | list[int] | tuple[int, ...], channels: int) -> list[int]:
    if isinstance(levels, int):
        parsed = [int(levels)]
    elif isinstance(levels, str):
        text = levels.strip()
        if not text:
            raise ValueError("--fsq-levels must not be empty")
        parsed = [int(part.strip()) for part in text.replace("x", ",").split(",") if part.strip()]
    else:
        parsed = [int(v) for v in levels]
    if len(parsed) == 1:
        parsed = parsed * int(channels)
    if len(parsed) != int(channels):
        raise ValueError(f"expected one FSQ level or {channels} comma-separated levels, got {len(parsed)}")
    if min(parsed) < 2:
        raise ValueError(f"FSQ levels must be >= 2, got {parsed}")
    return parsed


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


class ScalarFSQQuantizer(nn.Module):
    """Scalar finite scalar quantization over each z2 channel.

    The quantizer maps every z2 scalar to a finite per-channel grid. A learned
    affine range lets a pretrained continuous E2/D2 path be calibrated before
    optional finetuning.
    """

    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        embedding_dim: int = 20,
        train_affine: bool = True,
    ) -> None:
        super().__init__()
        levels = parse_fsq_levels(list(levels), int(embedding_dim))
        self.embedding_dim = int(embedding_dim)
        self.num_codes = int(max(levels))
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long))
        self.center = nn.Parameter(torch.zeros(self.embedding_dim), requires_grad=bool(train_affine))
        self.log_scale = nn.Parameter(torch.zeros(self.embedding_dim), requires_grad=bool(train_affine))

    def _shape_params(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        levels = self.levels.to(device=z.device, dtype=z.dtype).view(1, self.embedding_dim, 1, 1)
        center = self.center.to(dtype=z.dtype).view(1, self.embedding_dim, 1, 1)
        scale = self.log_scale.to(dtype=z.dtype).exp().clamp_min(1e-6).view(1, self.embedding_dim, 1, 1)
        return levels, center, scale

    def _quantize(self, z: torch.Tensor, *, ste: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 4:
            raise ValueError(f"expected z [B,C,H,W], got {tuple(z.shape)}")
        if int(z.shape[1]) != self.embedding_dim:
            raise ValueError(f"expected z channel dim {self.embedding_dim}, got {int(z.shape[1])}")
        levels, center, scale = self._shape_params(z)
        normalized = torch.tanh((z - center) / scale)
        level_span = (levels - 1.0).clamp_min(1.0)
        level_pos = (normalized + 1.0) * 0.5 * level_span
        quant_pos = round_ste(level_pos) if bool(ste) else level_pos.round()
        quant_pos = quant_pos.clamp_min(0.0).minimum(level_span)
        q_norm = quant_pos / level_span * 2.0 - 1.0
        q = q_norm * scale + center
        return q, quant_pos.detach().long()

    @torch.no_grad()
    def initialize_from_data(self, values: torch.Tensor, quantile: float = 0.001) -> None:
        if values.ndim == 4:
            values = values.detach().permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        if values.ndim != 2 or int(values.shape[1]) != self.embedding_dim:
            raise ValueError(f"expected FSQ init values [N,{self.embedding_dim}], got {tuple(values.shape)}")
        q = float(quantile)
        if not 0.0 <= q < 0.5:
            raise ValueError(f"FSQ init quantile must be in [0,0.5), got {quantile}")
        data = values.detach().float().to(device=self.center.device)
        if data.shape[0] < 1:
            raise ValueError("cannot initialize FSQ from empty values")
        lo = torch.quantile(data, q, dim=0)
        hi = torch.quantile(data, 1.0 - q, dim=0)
        center = 0.5 * (lo + hi)
        scale = (0.5 * (hi - lo)).clamp_min(1e-4)
        self.center.copy_(center.to(dtype=self.center.dtype))
        self.log_scale.copy_(scale.log().to(dtype=self.log_scale.dtype))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        q2, indices = self._quantize(z, ste=True)
        q2_st = z + (q2 - z).detach()
        q2_for_affine, _indices_for_affine = self._quantize(z.detach(), ste=True)
        codebook_loss = F.mse_loss(q2_for_affine.float(), z.detach().float())
        commit_loss = F.mse_loss(q2.detach().float(), z.float())
        vq_loss = codebook_loss + commit_loss
        stats = {
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
            "vq_loss": vq_loss,
            "vq_mse": F.mse_loss(q2.detach().float(), z.detach().float()),
        }
        return q2_st, q2, indices, stats

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q2, indices = self._quantize(z, ste=False)
        return q2, indices

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 4:
            raise ValueError(f"expected FSQ indices [B,C,H,W], got {tuple(indices.shape)}")
        if int(indices.shape[1]) != self.embedding_dim:
            raise ValueError(f"expected FSQ index channel dim {self.embedding_dim}, got {int(indices.shape[1])}")
        levels, center, scale = self._shape_params(indices.float())
        level_span = (levels - 1.0).clamp_min(1.0)
        q_norm = indices.float().clamp_min(0.0).minimum(level_span) / level_span * 2.0 - 1.0
        return q_norm * scale + center

    @torch.no_grad()
    def extra_metrics(self) -> dict[str, float]:
        scale = self.log_scale.detach().float().exp()
        levels = self.levels.detach().float()
        return {
            "fsq_levels_mean": float(levels.mean().item()),
            "fsq_levels_min": float(levels.min().item()),
            "fsq_levels_max": float(levels.max().item()),
            "fsq_scale_mean": float(scale.mean().item()),
            "fsq_scale_min": float(scale.min().item()),
            "fsq_scale_max": float(scale.max().item()),
        }


class FullMapCVQQuantizer(nn.Module):
    """Shared full-map codebook applied independently to each z2 channel.

    Input z2 is [B,20,16,16]. The K-entry codebook is [K,16,16], so each of
    the 20 z2 channel maps selects one full spatial code, producing indices
    [B,20].
    """

    def __init__(
        self,
        num_codes: int = 2048,
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

    @torch.no_grad()
    def initialize_from_samples(self, samples: torch.Tensor) -> None:
        selected = select_fullmap_codebook_samples(samples, self.num_codes, self.h, self.w, self.codebook.device, self.codebook.dtype)
        self.codebook.copy_(selected)

    def _check(self, z: torch.Tensor) -> None:
        if z.ndim != 4:
            raise ValueError(f"expected z [B,C,H,W], got {tuple(z.shape)}")
        if tuple(z.shape[2:]) != (self.h, self.w):
            raise ValueError(f"latent size {tuple(z.shape[2:])} does not match codebook {(self.h, self.w)}")

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._check(z)
        bsz, channels, h, w = z.shape
        flat = z.reshape(bsz * channels, h, w)
        q_flat, idx = nearest_codebook(flat, self.codebook, self.chunk_size)
        q2 = q_flat.reshape(bsz, channels, h, w)
        q2_st = z + (q2 - z).detach()
        codebook_loss = F.mse_loss(q2.float(), z.detach().float())
        commit_loss = F.mse_loss(q2.detach().float(), z.float())
        vq_loss = codebook_loss + self.beta * commit_loss
        stats = {
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
            "vq_loss": vq_loss,
            "vq_mse": F.mse_loss(q2.detach().float(), z.detach().float()),
        }
        return q2_st, q2, idx.reshape(bsz, channels), stats

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check(z)
        bsz, channels, h, w = z.shape
        q_flat, idx = nearest_codebook(z.reshape(bsz * channels, h, w), self.codebook, self.chunk_size)
        return q_flat.reshape(bsz, channels, h, w), idx.reshape(bsz, channels)

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 2:
            raise ValueError(f"expected indices [B,C], got {tuple(indices.shape)}")
        bsz, channels = indices.shape
        q_flat = F.embedding(indices.long().reshape(-1), self.codebook.flatten(1))
        return q_flat.view(bsz, channels, self.h, self.w).contiguous()


class FullMapSimVQQuantizer(nn.Module):
    """SimVQ variant of the full-map CVQ quantizer.

    The base codebook has shape [K,16,16]. Nearest-neighbor lookup uses the
    trainable projected/effective codebook, while the base codebook is frozen
    by default.
    """

    def __init__(
        self,
        num_codes: int = 2048,
        h: int = 16,
        w: int = 16,
        beta: float = 0.25,
        chunk_size: int = 128,
        freeze_codebook: bool = True,
        proj_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        dim = self.h * self.w
        self.codebook = nn.Parameter(torch.randn(self.num_codes, self.h, self.w) * (float(dim) ** -0.5))
        self.codebook.requires_grad_(not bool(freeze_codebook))
        self.embedding_proj = nn.Linear(dim, dim, bias=bool(proj_bias))

    @torch.no_grad()
    def initialize_from_samples(self, samples: torch.Tensor) -> None:
        selected = select_fullmap_codebook_samples(samples, self.num_codes, self.h, self.w, self.codebook.device, self.codebook.dtype)
        self.codebook.copy_(selected)
        dim = self.h * self.w
        eye = torch.eye(dim, device=self.embedding_proj.weight.device, dtype=self.embedding_proj.weight.dtype)
        self.embedding_proj.weight.copy_(eye)
        if self.embedding_proj.bias is not None:
            self.embedding_proj.bias.zero_()

    def _check(self, z: torch.Tensor) -> None:
        if z.ndim != 4:
            raise ValueError(f"expected z [B,C,H,W], got {tuple(z.shape)}")
        if tuple(z.shape[2:]) != (self.h, self.w):
            raise ValueError(f"latent size {tuple(z.shape[2:])} does not match codebook {(self.h, self.w)}")

    def effective_codebook(self) -> torch.Tensor:
        cb = self.codebook.float().flatten(1)
        return self.embedding_proj(cb).reshape(self.num_codes, self.h, self.w)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._check(z)
        bsz, channels, h, w = z.shape
        flat = z.reshape(bsz * channels, h, w)
        effective_codebook = self.effective_codebook()
        q_flat, idx = nearest_codebook(flat, effective_codebook, self.chunk_size)
        q2 = q_flat.reshape(bsz, channels, h, w)
        q2_st = z + (q2 - z).detach()
        codebook_loss = F.mse_loss(q2.float(), z.detach().float())
        commit_loss = F.mse_loss(q2.detach().float(), z.float())
        vq_loss = codebook_loss + self.beta * commit_loss
        stats = {
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
            "vq_loss": vq_loss,
            "vq_mse": F.mse_loss(q2.detach().float(), z.detach().float()),
        }
        return q2_st, q2, idx.reshape(bsz, channels), stats

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check(z)
        bsz, channels, h, w = z.shape
        q_flat, idx = nearest_codebook(z.reshape(bsz * channels, h, w), self.effective_codebook(), self.chunk_size)
        return q_flat.reshape(bsz, channels, h, w), idx.reshape(bsz, channels)

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 2:
            raise ValueError(f"expected indices [B,C], got {tuple(indices.shape)}")
        bsz, channels = indices.shape
        q_flat = F.embedding(indices.long().reshape(-1), self.effective_codebook().flatten(1))
        return q_flat.view(bsz, channels, self.h, self.w).contiguous()


class SpatialIndexNet(nn.Module):
    def __init__(self, in_ch: int = 16, num_codes: int = 64, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = int(in_ch)
        for _idx in range(max(1, int(depth))):
            layers.append(nn.Conv2d(ch, int(hidden), kernel_size=3, padding=1))
            layers.append(nn.GELU())
            ch = int(hidden)
        layers.append(nn.Conv2d(ch, int(num_codes), kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z1: torch.Tensor) -> torch.Tensor:
        return self.net(z1)


class ScalarFSQIndexNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 16,
        num_channels: int = 20,
        num_levels: int = 7,
        hidden: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_levels = int(num_levels)
        layers: list[nn.Module] = []
        ch = int(in_ch)
        for _idx in range(max(1, int(depth))):
            layers.append(nn.Conv2d(ch, int(hidden), kernel_size=3, padding=1))
            layers.append(nn.GELU())
            ch = int(hidden)
        layers.append(nn.Conv2d(ch, self.num_channels * self.num_levels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z1: torch.Tensor) -> torch.Tensor:
        logits = self.net(z1)
        bsz, _channels, h, w = logits.shape
        return logits.view(bsz, self.num_channels, self.num_levels, h, w).permute(0, 1, 3, 4, 2).contiguous()


class FullMapIndexNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 16,
        num_tokens: int = 20,
        num_codes: int = 2048,
        latent_h: int = 16,
        latent_w: int = 16,
        hidden: int = 128,
        depth: int = 3,
        heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if int(hidden) % int(heads) != 0:
            raise ValueError(f"FullMapIndexNet requires hidden divisible by heads, got hidden={hidden} heads={heads}")
        self.num_tokens = int(num_tokens)
        self.num_codes = int(num_codes)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.stem = nn.Sequential(
            nn.Conv2d(int(in_ch), int(hidden), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden), int(hidden), kernel_size=3, padding=1),
        )
        self.pos = nn.Parameter(torch.zeros(1, self.latent_h * self.latent_w, int(hidden)))
        self.queries = nn.Parameter(torch.randn(1, self.num_tokens, int(hidden)) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden),
            nhead=int(heads),
            dim_feedforward=int(hidden * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(depth)))
        self.cross = nn.MultiheadAttention(int(hidden), int(heads), dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(int(hidden))
        self.head = nn.Linear(int(hidden), self.num_codes)

    def forward(self, z1: torch.Tensor) -> torch.Tensor:
        bsz = z1.shape[0]
        feat = self.stem(z1).flatten(2).transpose(1, 2)
        feat = self.encoder(feat + self.pos[:, : feat.shape[1]])
        queries = self.queries.expand(bsz, -1, -1)
        out, _attn = self.cross(queries, feat, feat, need_weights=False)
        return self.head(self.norm(out))


def _replace_encoder_patch_embed(encoder: nn.Module, in_chans: int) -> None:
    inner = encoder.encoder
    old = inner.patch_embed
    if int(getattr(old, "in_chans", 3)) == int(in_chans):
        return
    device = old.proj.weight.device
    dtype = old.proj.weight.dtype
    new = PatchEmbed(
        img_size=old.img_size,
        patch_size=old.patch_size,
        in_chans=int(in_chans),
        embed_dim=old.embed_dim,
        norm_layer=None,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        new.proj.weight.zero_()
        copy_ch = min(old.proj.weight.shape[1], int(in_chans))
        new.proj.weight[:, :copy_ch].copy_(old.proj.weight[:, :copy_ch])
        if int(in_chans) > copy_ch:
            mean_w = old.proj.weight.mean(dim=1, keepdim=True)
            new.proj.weight[:, copy_ch:].copy_(mean_w.expand(-1, int(in_chans) - copy_ch, -1, -1))
        new.proj.bias.copy_(old.proj.bias)
    inner.patch_embed = new


def build_jscc_encoder(args, device: torch.device, latent_ch: int, in_chans: int) -> nn.Module:
    cfg = jsccf_io.build_config(args, encoder_in_chans=int(in_chans))
    encoder = JSCC_encoder(cfg, int(latent_ch)).to(device)
    _replace_encoder_patch_embed(encoder, int(in_chans))
    return encoder


def build_jscc_decoder(args, device: torch.device, latent_ch: int) -> nn.Module:
    cfg = jsccf_io.build_config(args, encoder_in_chans=3)
    return JSCC_decoder(cfg, int(latent_ch)).to(device)


def build_layer1(args, device: torch.device) -> tuple[nn.Module, nn.Module]:
    e1 = build_jscc_encoder(args, device, latent_ch=z1_ch(args), in_chans=3)
    d1 = build_jscc_decoder(args, device, latent_ch=z1_ch(args))
    return e1, d1


def build_layer2(args, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, OutputsCombiner]:
    e1, d1 = build_layer1(args, device)
    e2_in = 3 if str(args.variant) == "residual_input" else 6
    e2 = build_jscc_encoder(args, device, latent_ch=z2_ch(args), in_chans=e2_in)
    d2 = build_jscc_decoder(args, device, latent_ch=3)
    combiner = OutputsCombiner().to(device)
    return e1, d1, e2, d2, combiner


def build_layer3(
    args,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, OutputsCombiner, nn.Module, nn.Module]:
    e1, d1 = build_layer1(args, device)
    e2 = build_jscc_encoder(args, device, latent_ch=z2_ch(args), in_chans=6)
    d2 = build_jscc_decoder(args, device, latent_ch=20)
    combiner = OutputsCombiner().to(device)
    quantizer_name = str(getattr(args, "quantizer", "simvq"))
    if quantizer_name == "cvq":
        quantizer = FullMapCVQQuantizer(
            num_codes=int(getattr(args, "cvq_k", 2048)),
            h=int(getattr(args, "latent_h", 16)),
            w=int(getattr(args, "latent_w", 16)),
            beta=float(getattr(args, "beta_commit", 0.25)),
            chunk_size=int(getattr(args, "cvq_chunk_size", 128)),
        ).to(device)
        indexnet = FullMapIndexNet(
            in_ch=z1_ch(args),
            num_tokens=z2_ch(args),
            num_codes=int(getattr(args, "cvq_k", 2048)),
            latent_h=int(getattr(args, "latent_h", 16)),
            latent_w=int(getattr(args, "latent_w", 16)),
            hidden=int(getattr(args, "index_hidden", 128)),
            depth=int(getattr(args, "index_depth", 3)),
            heads=int(getattr(args, "index_heads", 4)),
        ).to(device)
    elif quantizer_name == "fullmap_simvq":
        quantizer = FullMapSimVQQuantizer(
            num_codes=int(getattr(args, "fullmap_simvq_k", 2048)),
            h=int(getattr(args, "latent_h", 16)),
            w=int(getattr(args, "latent_w", 16)),
            beta=float(getattr(args, "beta_commit", 0.25)),
            chunk_size=int(getattr(args, "fullmap_simvq_chunk_size", 128)),
            freeze_codebook=not bool(getattr(args, "fullmap_simvq_train_codebook", False)),
        ).to(device)
        indexnet = FullMapIndexNet(
            in_ch=z1_ch(args),
            num_tokens=z2_ch(args),
            num_codes=int(getattr(args, "fullmap_simvq_k", 2048)),
            latent_h=int(getattr(args, "latent_h", 16)),
            latent_w=int(getattr(args, "latent_w", 16)),
            hidden=int(getattr(args, "index_hidden", 128)),
            depth=int(getattr(args, "index_depth", 3)),
            heads=int(getattr(args, "index_heads", 4)),
        ).to(device)
    elif quantizer_name == "vq":
        quantizer = VQQuantizer(
            num_codes=int(getattr(args, "vq_k", 64)),
            embedding_dim=z2_ch(args),
            beta=float(getattr(args, "beta_commit", 0.25)),
        ).to(device)
        indexnet = SpatialIndexNet(
            in_ch=z1_ch(args),
            num_codes=int(getattr(args, "vq_k", 64)),
            hidden=int(getattr(args, "index_hidden", 128)),
            depth=int(getattr(args, "index_depth", 3)),
        ).to(device)
    elif quantizer_name == "fsq":
        fsq_levels = parse_fsq_levels(getattr(args, "fsq_levels", "7"), z2_ch(args))
        quantizer = ScalarFSQQuantizer(
            levels=fsq_levels,
            embedding_dim=z2_ch(args),
            train_affine=not bool(getattr(args, "fsq_freeze_affine", False)),
        ).to(device)
        indexnet = ScalarFSQIndexNet(
            in_ch=z1_ch(args),
            num_channels=z2_ch(args),
            num_levels=max(fsq_levels),
            hidden=int(getattr(args, "index_hidden", 128)),
            depth=int(getattr(args, "index_depth", 3)),
        ).to(device)
    elif quantizer_name == "simvq":
        quantizer = SimVQQuantizer(
            num_codes=int(getattr(args, "simvq_k", 64)),
            embedding_dim=z2_ch(args),
            beta=float(getattr(args, "beta_commit", 0.25)),
            freeze_codebook=not bool(getattr(args, "simvq_train_codebook", False)),
        ).to(device)
        indexnet = SpatialIndexNet(
            in_ch=z1_ch(args),
            num_codes=int(getattr(args, "simvq_k", 64)),
            hidden=int(getattr(args, "index_hidden", 128)),
            depth=int(getattr(args, "index_depth", 3)),
        ).to(device)
    else:
        raise ValueError(f"unknown Stage3 quantizer {quantizer_name!r}")
    return e1, d1, e2, d2, combiner, quantizer, indexnet


def freeze_layer1(e1: nn.Module, d1: nn.Module) -> None:
    freeze_module(e1, trainable=False)
    freeze_module(d1, trainable=False)


def layer1_forward(e1: nn.Module, d1: nn.Module, img: torch.Tensor) -> dict[str, torch.Tensor]:
    z1, _ = e1(img)
    x1_raw = d1(z1)
    x1 = x1_raw.clamp(0.0, 1.0)
    return {"z1": z1, "x1_raw": x1_raw, "x1": x1}


def layer2_forward(
    e1: nn.Module,
    d1: nn.Module,
    e2: nn.Module,
    d2: nn.Module,
    combiner: OutputsCombiner,
    img: torch.Tensor,
    variant: str,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        base = layer1_forward(e1, d1, img)
    x1 = base["x1"]
    if str(variant) == "residual_input":
        e2_in = img - x1
    else:
        e2_in = torch.cat([img, x1], dim=1)
    z2, _ = e2(e2_in)
    # u2_raw = d2(torch.cat([base["z1"], z2], dim=1))
    u2_raw = d2(z2)
    u2 = u2_raw.clamp(0.0, 1.0)
    x2_hat = combiner(x1, u2)
    final = u2 if str(variant) == "no_combiner" else x2_hat
    return {
        "z1": base["z1"],
        "x1": x1,
        "z2": z2,
        "u2_raw": u2_raw,
        "u2": u2,
        "x2_hat": x2_hat,
        "final": final,
    }


def layer3_forward(
    e1: nn.Module,
    d1: nn.Module,
    e2: nn.Module,
    d2: nn.Module,
    combiner: OutputsCombiner,
    quantizer: nn.Module,
    indexnet: nn.Module,
    img: torch.Tensor,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    with torch.no_grad():
        base = layer1_forward(e1, d1, img)
    z1 = base["z1"]
    x1 = base["x1"]
    e2_in = torch.cat([img, x1], dim=1)
    z2, _ = e2(e2_in)
    q2_st, q2, q2_index, vq_stats = quantizer(z2)
    # u2_raw = d2(torch.cat([z1, q2_st], dim=1))
    u2_raw = d2(q2_st)
    u2 = u2_raw.clamp(0.0, 1.0)
    final_oracle = combiner(x1, u2)
    logits = indexnet(z1.detach())
    return {
        "z1": z1,
        "x1": x1,
        "z2": z2,
        "q2_st": q2_st,
        "q2": q2,
        "q2_index": q2_index,
        "vq_stats": vq_stats,
        "index_logits": logits,
        "u2_raw": u2_raw,
        "u2": u2,
        "final_oracle": final_oracle,
        "final": final_oracle,
    }
