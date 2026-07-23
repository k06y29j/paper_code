#!/usr/bin/env python3
"""Trainable nested-prefix vector quantizers for Layer2 exploration.

The two supported tokenizations deliberately have different token units:

``image-vq``
    One token is the channel vector at a spatial location.  For an input
    ``[B,C,H,W]`` the codebook is ``[K_max,C]`` and indices are ``[B,H,W]``.

``channel-vq``
    One token is a complete channel map.  For an input ``[B,C,H,W]`` the
    codebook is ``[K_max,H,W]`` and indices are ``[B,C]``.

Every operating point ``K`` uses ``codebook[:K]``.  Consequently all rates
share parameters and the K-small codebook is an exact prefix of K-large,
rather than a separately trained table that only happens to have the same
shape.  This module is self-contained so the exploration entrypoints do not
depend on one of the historical Stage3 quantizers.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


VQFamily = Literal["image-vq", "channel-vq"]


def canonical_vq_family(family: str) -> VQFamily:
    """Return the canonical spelling for a supported VQ family."""

    value = str(family).strip().lower().replace("_", "-")
    image_aliases = {"image", "image-vq", "per-location", "location", "vector"}
    channel_aliases = {"channel", "channel-vq", "channel-map", "full-map", "fullmap"}
    if value in image_aliases:
        return "image-vq"
    if value in channel_aliases:
        return "channel-vq"
    raise ValueError(f"unknown VQ family {family!r}; expected 'image-vq' or 'channel-vq'")


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _resolve_k(k_max: int | None, num_codes: int | None, max_codes: int | None) -> int:
    supplied = [int(value) for value in (k_max, num_codes, max_codes) if value is not None]
    if not supplied:
        raise ValueError("one of k_max, num_codes, or max_codes must be provided")
    if len(set(supplied)) != 1:
        raise ValueError(f"conflicting codebook sizes: {supplied}")
    return _positive_int("k_max", supplied[0])


def _flatten_codebook(codebook: torch.Tensor) -> torch.Tensor:
    if codebook.ndim < 2:
        raise ValueError(f"codebook must be [K,...], got {tuple(codebook.shape)}")
    if int(codebook.shape[0]) < 1:
        raise ValueError("codebook must contain at least one embedding")
    return codebook.reshape(int(codebook.shape[0]), -1)


@torch.no_grad()
def nearest_codebook_indices_chunked(
    tokens: torch.Tensor,
    codebook: torch.Tensor,
    *,
    query_chunk_size: int = 4096,
    codebook_chunk_size: int = 4096,
    return_distances: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Find exact nearest neighbours without materializing the full N x K matrix.

    Distances are accumulated in float32 and both the query and codebook axes
    are chunked.  Ties select the lowest code index, matching a single
    ``argmin`` over the complete shared prefix.
    """

    if tokens.ndim != 2:
        raise ValueError(f"tokens must be [N,D], got {tuple(tokens.shape)}")
    cb = _flatten_codebook(codebook)
    if int(tokens.shape[0]) < 1:
        raise ValueError("cannot quantize an empty token tensor")
    if int(tokens.shape[1]) != int(cb.shape[1]):
        raise ValueError(
            f"token/codebook dimensions differ: {int(tokens.shape[1])} versus {int(cb.shape[1])}"
        )
    if tokens.device != cb.device:
        raise ValueError(f"tokens and codebook must share a device, got {tokens.device} and {cb.device}")
    query_chunk_size = _positive_int("query_chunk_size", query_chunk_size)
    codebook_chunk_size = _positive_int("codebook_chunk_size", codebook_chunk_size)

    # Fail early: an argmin in the presence of NaN/Inf silently produces an
    # arbitrary index and can make apparent codebook collapse very difficult
    # to diagnose.
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("tokens contain NaN or Inf")
    if not bool(torch.isfinite(cb).all()):
        raise ValueError("codebook contains NaN or Inf")

    tokens_f = tokens.detach().float()
    codebook_f = cb.detach().float()
    cb_norm = codebook_f.square().sum(dim=1)
    all_indices: list[torch.Tensor] = []
    all_distances: list[torch.Tensor] = []

    for query_start in range(0, int(tokens_f.shape[0]), query_chunk_size):
        query = tokens_f[query_start : query_start + query_chunk_size]
        query_norm = query.square().sum(dim=1, keepdim=True)
        best_distance = torch.full(
            (int(query.shape[0]),), float("inf"), device=query.device, dtype=torch.float32
        )
        best_index = torch.zeros((int(query.shape[0]),), device=query.device, dtype=torch.long)

        for code_start in range(0, int(codebook_f.shape[0]), codebook_chunk_size):
            code = codebook_f[code_start : code_start + codebook_chunk_size]
            distance = query_norm + cb_norm[code_start : code_start + int(code.shape[0])].unsqueeze(0)
            distance = distance - 2.0 * (query @ code.t())
            # Roundoff may make a squared Euclidean distance slightly negative.
            distance.clamp_min_(0.0)
            chunk_distance, chunk_index = distance.min(dim=1)
            update = chunk_distance < best_distance
            best_distance = torch.where(update, chunk_distance, best_distance)
            best_index = torch.where(update, chunk_index + code_start, best_index)

        all_indices.append(best_index)
        if return_distances:
            all_distances.append(best_distance)

    indices = torch.cat(all_indices, dim=0)
    if return_distances:
        return indices, torch.cat(all_distances, dim=0)
    return indices


def nearest_codebook_chunked(
    tokens: torch.Tensor,
    codebook: torch.Tensor,
    *,
    query_chunk_size: int = 4096,
    codebook_chunk_size: int = 4096,
    detach_codebook: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hard nearest-code vectors and indices with chunked assignment."""

    flat_codebook = _flatten_codebook(codebook)
    indices = nearest_codebook_indices_chunked(
        tokens,
        flat_codebook,
        query_chunk_size=query_chunk_size,
        codebook_chunk_size=codebook_chunk_size,
    )
    lookup = flat_codebook.detach() if detach_codebook else flat_codebook
    quantized = F.embedding(indices, lookup).to(dtype=tokens.dtype)
    return quantized, indices


def usage_statistics(indices: torch.Tensor, num_codes: int) -> dict[str, torch.Tensor]:
    """Tensor-valued occupancy, perplexity and top-1 usage statistics."""

    num_codes = _positive_int("num_codes", num_codes)
    flat = indices.detach().long().reshape(-1)
    if int(flat.numel()) < 1:
        raise ValueError("indices must not be empty")
    if int(flat.min()) < 0 or int(flat.max()) >= num_codes:
        raise ValueError(
            f"indices must lie in [0,{num_codes}), got min={int(flat.min())}, max={int(flat.max())}"
        )
    counts = torch.bincount(flat, minlength=num_codes)
    probabilities = counts.float() / float(flat.numel())
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    perplexity = entropy.exp()
    occupancy = nonzero.sum()
    top1 = probabilities.max()
    return {
        "counts": counts,
        "occupancy": occupancy,
        "used_codes": occupancy,
        "dead_codes": occupancy.new_tensor(num_codes) - occupancy,
        "occupancy_ratio": occupancy.float() / float(num_codes),
        "dead_ratio": 1.0 - occupancy.float() / float(num_codes),
        "perplexity": perplexity,
        "ppl": perplexity,
        "perplexity_ratio": perplexity / float(num_codes),
        "top1": top1,
        "top1_share": top1,
        "top1_frac": top1,
    }


@torch.no_grad()
def usage_metrics(indices: torch.Tensor, num_codes: int) -> dict[str, float]:
    """JSON/log-friendly form of :func:`usage_statistics`."""

    stats = usage_statistics(indices, num_codes)
    return {
        key: float(value.item())
        for key, value in stats.items()
        if key != "counts"
    }


def _sample_codebook_rows(
    codebook: torch.Tensor,
    max_samples: int,
    seed: int,
) -> torch.Tensor:
    flat = _flatten_codebook(codebook).detach().float()
    max_samples = _positive_int("max_samples", max_samples)
    if int(flat.shape[0]) <= max_samples:
        return flat
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected = torch.randperm(int(flat.shape[0]), generator=generator)[:max_samples]
    return flat[selected.to(device=flat.device)]


@torch.no_grad()
def sampled_codebook_l2_metrics(
    codebook: torch.Tensor,
    *,
    max_samples: int = 512,
    seed: int = 0,
) -> dict[str, float]:
    """Pairwise L2 summary on a deterministic codebook-row sample."""

    sample = _sample_codebook_rows(codebook, max_samples=max_samples, seed=seed)
    if int(sample.shape[0]) < 2:
        return {
            "codebook_l2_mean": 0.0,
            "codebook_l2_std": 0.0,
            "codebook_l2_min": 0.0,
            "codebook_l2_max": 0.0,
            "codebook_l2_sample_size": float(sample.shape[0]),
        }
    distances = torch.pdist(sample, p=2)
    return {
        "codebook_l2_mean": float(distances.mean().item()),
        "codebook_l2_std": float(distances.std(unbiased=False).item()),
        "codebook_l2_min": float(distances.min().item()),
        "codebook_l2_max": float(distances.max().item()),
        "codebook_l2_sample_size": float(sample.shape[0]),
    }


@torch.no_grad()
def codebook_effective_rank_metrics(
    codebook: torch.Tensor,
    *,
    max_samples: int = 2048,
    seed: int = 0,
    center: bool = True,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Entropy effective rank, stable rank and numerical rank of embeddings."""

    matrix = _sample_codebook_rows(codebook, max_samples=max_samples, seed=seed)
    if center and int(matrix.shape[0]) > 1:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(matrix)
    total = singular_values.sum()
    if int(singular_values.numel()) == 0 or float(total.item()) <= eps:
        effective_rank = 0.0
        stable_rank = 0.0
        numerical_rank = 0.0
        leading_share = 0.0
    else:
        probabilities = (singular_values / total).clamp_min(eps)
        effective_rank = float((-(probabilities * probabilities.log()).sum()).exp().item())
        squared = singular_values.square()
        stable_rank = float((squared.sum() / squared.max().clamp_min(eps)).item())
        tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular_values.max()
        numerical_rank = float((singular_values > tolerance).sum().item())
        leading_share = float(probabilities[0].item())
    return {
        "codebook_effective_rank": effective_rank,
        "codebook_stable_rank": stable_rank,
        "codebook_numerical_rank": numerical_rank,
        "codebook_singular_top1_share": leading_share,
        "codebook_rank_sample_size": float(matrix.shape[0]),
    }


@torch.no_grad()
def codebook_health_metrics(
    codebook: torch.Tensor,
    *,
    l2_max_samples: int = 512,
    rank_max_samples: int = 2048,
    seed: int = 0,
) -> dict[str, float]:
    """Combined sampled L2 and rank diagnostics."""

    result = sampled_codebook_l2_metrics(codebook, max_samples=l2_max_samples, seed=seed)
    result.update(
        codebook_effective_rank_metrics(codebook, max_samples=rank_max_samples, seed=seed)
    )
    return result


def _randperm_on(tensor: torch.Tensor, *, generator: torch.Generator | None, seed: int | None) -> torch.Tensor:
    if generator is not None and seed is not None:
        raise ValueError("pass generator or seed, not both")
    if seed is not None:
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(int(seed))
    return torch.randperm(int(tensor.shape[0]), device=tensor.device, generator=generator)


def shuffle_vq_tokens(
    value: torch.Tensor,
    family: str,
    *,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Shuffle complete VQ tokens while preserving each token's internal axes."""

    family = canonical_vq_family(family)
    if value.ndim != 4:
        raise ValueError(f"quantized value must be BCHW, got {tuple(value.shape)}")
    bsz, channels, h, w = value.shape
    if family == "image-vq":
        tokens = value.permute(0, 2, 3, 1).contiguous().reshape(bsz * h * w, channels)
        shuffled = tokens[_randperm_on(tokens, generator=generator, seed=seed)]
        return shuffled.view(bsz, h, w, channels).permute(0, 3, 1, 2).contiguous()
    tokens = value.reshape(bsz * channels, h, w)
    shuffled = tokens[_randperm_on(tokens, generator=generator, seed=seed)]
    return shuffled.view(bsz, channels, h, w)


def shuffle_vq_indices(
    indices: torch.Tensor,
    family: str,
    *,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Shuffle one scalar index per image-VQ or channel-VQ token."""

    family = canonical_vq_family(family)
    expected_ndim = 3 if family == "image-vq" else 2
    if indices.ndim != expected_ndim:
        raise ValueError(
            f"{family} indices must have rank {expected_ndim}, got {tuple(indices.shape)}"
        )
    flat = indices.reshape(-1)
    return flat[_randperm_on(flat, generator=generator, seed=seed)].view_as(indices)


def zero_vq_tokens(value: torch.Tensor) -> torch.Tensor:
    """Zero ablation with exactly the same tensor metadata as ``value``."""

    return torch.zeros_like(value)


class NestedPrefixVQ(nn.Module):
    """One trainable codebook shared by all prefix capacities ``K <= K_max``."""

    def __init__(
        self,
        family: str,
        k_max: int | None = None,
        *,
        num_codes: int | None = None,
        max_codes: int | None = None,
        channels: int = 16,
        embedding_dim: int | None = None,
        h: int = 16,
        w: int = 16,
        beta: float = 0.25,
        query_chunk_size: int = 4096,
        codebook_chunk_size: int = 4096,
        usage_decay: float = 0.99,
        track_usage: bool = True,
        channel_codebook_mode: str = "global",
    ) -> None:
        super().__init__()
        self.family = canonical_vq_family(family)
        self.k_max = _resolve_k(k_max, num_codes, max_codes)
        self.num_codes = self.k_max
        self.max_codes = self.k_max
        if embedding_dim is not None:
            if int(channels) != 16 and int(channels) != int(embedding_dim):
                raise ValueError(f"conflicting channels={channels} and embedding_dim={embedding_dim}")
            channels = int(embedding_dim)
        self.channels = _positive_int("channels", channels)
        self.embedding_dim = self.channels
        self.h = _positive_int("h", h)
        self.w = _positive_int("w", w)
        self.channel_codebook_mode = str(channel_codebook_mode).strip().lower().replace("_", "-")
        if self.channel_codebook_mode not in {"global", "grouped"}:
            raise ValueError(
                f"channel_codebook_mode must be 'global' or 'grouped', got {channel_codebook_mode!r}"
            )
        if self.family != "channel-vq" and self.channel_codebook_mode != "global":
            raise ValueError("channel_codebook_mode='grouped' is only valid for channel-vq")
        if self.channel_codebook_mode == "grouped" and self.k_max % self.channels != 0:
            raise ValueError(
                f"grouped channel-vq requires Kmax divisible by C, got Kmax={self.k_max}, C={self.channels}"
            )
        self.beta = float(beta)
        if self.beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        self.query_chunk_size = _positive_int("query_chunk_size", query_chunk_size)
        self.codebook_chunk_size = _positive_int("codebook_chunk_size", codebook_chunk_size)
        self.usage_decay = float(usage_decay)
        if not 0.0 <= self.usage_decay < 1.0:
            raise ValueError(f"usage_decay must be in [0,1), got {self.usage_decay}")
        self.track_usage = bool(track_usage)

        if self.family == "image-vq":
            code_shape = (self.k_max, self.channels)
            vector_dim = self.channels
        else:
            code_shape = (self.k_max, self.h, self.w)
            vector_dim = self.h * self.w
        scale = float(vector_dim) ** -0.5
        self.codebook = nn.Parameter(torch.randn(*code_shape) * scale)
        self.register_buffer("ema_count", torch.zeros(self.k_max, dtype=torch.float32))
        self.register_buffer("inactive_steps", torch.zeros(self.k_max, dtype=torch.long))

    @property
    def Kmax(self) -> int:  # noqa: N802 - useful spelling in experiment tables.
        return self.k_max

    @property
    def embedding_shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.codebook.shape[1:])

    def extra_repr(self) -> str:
        return (
            f"family={self.family!r}, k_max={self.k_max}, embedding_shape={self.embedding_shape}, "
            f"channel_codebook_mode={self.channel_codebook_mode!r}, "
            f"beta={self.beta}, query_chunk_size={self.query_chunk_size}, "
            f"codebook_chunk_size={self.codebook_chunk_size}"
        )

    def _checked_k(self, k: int | None) -> int:
        if k is None:
            return self.k_max
        if isinstance(k, bool):
            raise TypeError("k must be an integer, not bool")
        k = int(k)
        if not 1 <= k <= self.k_max:
            raise ValueError(f"k must lie in [1,{self.k_max}], got {k}")
        if self.channel_codebook_mode == "grouped" and k % self.channels != 0:
            raise ValueError(
                f"grouped channel-vq requires K divisible by C, got K={k}, C={self.channels}"
            )
        return k

    def local_code_count(self, k: int | None = None) -> int:
        """Number of candidates available to one token at prefix K."""

        checked = self._checked_k(k)
        return checked // self.channels if self.channel_codebook_mode == "grouped" else checked

    def codebook_at_k(self, k: int | None = None) -> torch.Tensor:
        """Return a differentiable view of the exact shared prefix."""

        return self.codebook[: self._checked_k(k)]

    def effective_codebook(self, k: int | None = None) -> torch.Tensor:
        """Compatibility alias; there is no hidden projection in this VQ."""

        return self.codebook_at_k(k)

    def _validate_input(self, z: torch.Tensor) -> None:
        if z.ndim != 4:
            raise ValueError(f"expected z [B,C,{self.h},{self.w}], got {tuple(z.shape)}")
        if int(z.shape[0]) < 1 or int(z.shape[1]) < 1:
            raise ValueError(f"batch and channel dimensions must be non-empty, got {tuple(z.shape)}")
        if tuple(int(v) for v in z.shape[2:]) != (self.h, self.w):
            raise ValueError(f"expected spatial size [{self.h},{self.w}], got {tuple(z.shape[2:])}")
        if self.family == "image-vq" and int(z.shape[1]) != self.channels:
            raise ValueError(f"image-vq expects {self.channels} channels, got {int(z.shape[1])}")
        if z.device != self.codebook.device:
            raise ValueError(f"z and codebook must share a device, got {z.device} and {self.codebook.device}")

    def flatten_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """Flatten BCHW according to the selected family's indivisible token."""

        self._validate_input(z)
        bsz, channels, h, w = z.shape
        if self.family == "image-vq":
            return z.permute(0, 2, 3, 1).contiguous().view(bsz * h * w, channels)
        return z.contiguous().view(bsz * channels, h * w)

    def unflatten_tokens(self, flat: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        self._validate_input(reference)
        bsz, channels, h, w = reference.shape
        expected = (bsz * h * w, channels) if self.family == "image-vq" else (bsz * channels, h * w)
        if tuple(flat.shape) != expected:
            raise ValueError(f"expected flat tokens {expected}, got {tuple(flat.shape)}")
        if self.family == "image-vq":
            return flat.view(bsz, h, w, channels).permute(0, 3, 1, 2).contiguous()
        return flat.view(bsz, channels, h, w)

    def _reshape_indices(self, flat_indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        bsz, channels, h, w = z.shape
        if self.family == "image-vq":
            return flat_indices.view(bsz, h, w)
        return flat_indices.view(bsz, channels)

    def indices_to_local(self, indices: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Convert grouped global row ids to per-channel round ids.

        Global/image modes return the original indices because every token sees
        the complete prefix.  Grouped channel-VQ validates that channel ``c``
        only references rows ``c + C*round`` before returning ``round``.
        """

        checked = self._checked_k(k)
        self._validate_indices(indices, checked)
        if self.channel_codebook_mode != "grouped":
            return indices
        return torch.div(indices.long(), self.channels, rounding_mode="floor")

    def local_to_global_indices(self, local_indices: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Convert grouped round ids ``[B,C]`` to round-major codebook rows."""

        checked = self._checked_k(k)
        if self.channel_codebook_mode != "grouped":
            raise ValueError("local_to_global_indices is only defined for grouped channel-vq")
        if local_indices.ndim != 2 or int(local_indices.shape[1]) != self.channels:
            raise ValueError(f"grouped local indices must be [B,{self.channels}], got {tuple(local_indices.shape)}")
        rounds = checked // self.channels
        flat = local_indices.detach().long()
        if int(flat.numel()) < 1 or int(flat.min()) < 0 or int(flat.max()) >= rounds:
            raise ValueError(f"local indices must lie in [0,{rounds})")
        channel_ids = torch.arange(self.channels, device=local_indices.device).view(1, self.channels)
        return local_indices.long() * self.channels + channel_ids

    def _quantize_channel_grouped(
        self,
        z: torch.Tensor,
        k: int,
        *,
        detach_codebook: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest lookup restricted to the round-major group of each channel."""

        bsz, channels, h, w = z.shape
        if channels != self.channels:
            raise ValueError(f"grouped channel-vq expects C={self.channels}, got {channels}")
        rounds = k // channels
        vector_dim = h * w
        prefix = self.codebook_at_k(k)
        grouped_codebook = prefix.reshape(rounds, channels, vector_dim).permute(1, 0, 2).contiguous()
        tokens = z.reshape(bsz, channels, vector_dim)
        with torch.no_grad():
            token_values = tokens.detach().float()
            code_values = grouped_codebook.detach().float()
            distances = token_values.square().sum(dim=2, keepdim=True)
            distances = distances + code_values.square().sum(dim=2).unsqueeze(0)
            distances = distances - 2.0 * torch.einsum("bcd,crd->bcr", token_values, code_values)
            local_indices = distances.argmin(dim=2)
            global_indices = self.local_to_global_indices(local_indices, k)
        lookup = _flatten_codebook(prefix.detach() if detach_codebook else prefix)
        q_flat = F.embedding(global_indices.reshape(-1), lookup).to(dtype=z.dtype)
        return q_flat.view(bsz, channels, h, w), global_indices

    def quantize_input(
        self,
        z: torch.Tensor,
        k: int | None = None,
        *,
        detach_codebook: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard ``q`` and indices, without STE or auxiliary losses.

        This is the receiver direct-q primitive.  A caller that predicts a raw
        q tensor can explicitly form ``raw + (q_hard - raw).detach()`` while
        still using exactly the sender quantizer's K-prefix lookup rule.
        """

        self._validate_input(z)
        k = self._checked_k(k)
        if self.channel_codebook_mode == "grouped":
            return self._quantize_channel_grouped(z, k, detach_codebook=detach_codebook)
        tokens = self.flatten_tokens(z)
        prefix = self.codebook_at_k(k)
        q_flat, flat_indices = nearest_codebook_chunked(
            tokens,
            prefix,
            query_chunk_size=self.query_chunk_size,
            codebook_chunk_size=self.codebook_chunk_size,
            detach_codebook=detach_codebook,
        )
        q_hard = self.unflatten_tokens(q_flat, z)
        return q_hard, self._reshape_indices(flat_indices, z)

    @torch.no_grad()
    def encode(self, z: torch.Tensor, k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """No-gradient hard encoder using the requested shared prefix."""

        return self.quantize_input(z, k, detach_codebook=True)

    def forward_at_k(
        self,
        z: torch.Tensor,
        k: int,
        *,
        update_usage: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Quantize at prefix K and return STE q, hard q, indices and stats."""

        k = self._checked_k(k)
        q_hard, indices = self.quantize_input(z, k, detach_codebook=False)
        q_st = z + (q_hard - z).detach()
        codebook_loss = F.mse_loss(q_hard.float(), z.detach().float())
        commit_loss = F.mse_loss(z.float(), q_hard.detach().float())
        vq_loss = codebook_loss + self.beta * commit_loss
        usage = usage_statistics(indices, k)
        stats: dict[str, torch.Tensor] = {
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
            "commitment_loss": commit_loss,
            "vq_loss": vq_loss,
            "loss": vq_loss,
            "vq_mse": F.mse_loss(q_hard.detach().float(), z.detach().float()),
            "occupancy": usage["occupancy"],
            "used_codes": usage["used_codes"],
            "dead_codes": usage["dead_codes"],
            "occupancy_ratio": usage["occupancy_ratio"],
            "dead_ratio": usage["dead_ratio"],
            "perplexity": usage["perplexity"],
            "ppl": usage["ppl"],
            "perplexity_ratio": usage["perplexity_ratio"],
            "top1": usage["top1"],
            "top1_share": usage["top1_share"],
            "top1_frac": usage["top1_frac"],
        }
        if self.channel_codebook_mode == "grouped":
            local_indices = self.indices_to_local(indices, k)
            local_usage = usage_statistics(local_indices, self.local_code_count(k))
            stats.update(
                {
                    "local_indices": local_indices,
                    "local_used_codes": local_usage["used_codes"],
                    "local_perplexity": local_usage["perplexity"],
                    "local_ppl": local_usage["ppl"],
                    "local_perplexity_ratio": local_usage["perplexity_ratio"],
                    "local_ppl_ratio": local_usage["ppl"].float() / float(self.local_code_count(k)),
                    "local_top1": local_usage["top1"],
                }
            )
        should_update = self.training and self.track_usage if update_usage is None else bool(update_usage)
        if should_update:
            self.update_usage_ema(indices, k)
        stats["ema_occupancy"] = (self.ema_count[:k] > 0).sum()
        stats["ema_occupancy_ratio"] = stats["ema_occupancy"].float() / float(k)
        return q_st, q_hard, indices, stats

    def forward(
        self,
        z: torch.Tensor,
        k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return self.forward_at_k(z, self._checked_k(k))

    def _validate_indices(self, indices: torch.Tensor, k: int) -> None:
        expected_ndim = 3 if self.family == "image-vq" else 2
        if indices.ndim != expected_ndim:
            raise ValueError(
                f"{self.family} indices must have rank {expected_ndim}, got {tuple(indices.shape)}"
            )
        if indices.dtype.is_floating_point or indices.dtype.is_complex:
            raise TypeError(f"indices must have an integer dtype, got {indices.dtype}")
        if int(indices.numel()) < 1:
            raise ValueError("indices must not be empty")
        flat = indices.detach().long().reshape(-1)
        if int(flat.min()) < 0 or int(flat.max()) >= k:
            raise ValueError(f"indices must lie in [0,{k}), got min={int(flat.min())}, max={int(flat.max())}")
        if self.channel_codebook_mode == "grouped":
            if int(indices.shape[1]) != self.channels:
                raise ValueError(f"grouped channel indices must be [B,{self.channels}], got {tuple(indices.shape)}")
            expected_channels = torch.arange(self.channels, device=indices.device).view(1, self.channels)
            if not bool(((indices.long() % self.channels) == expected_channels).all()):
                raise ValueError("grouped indices reference a codebook row owned by a different channel")

    def get_codebook_entry(
        self,
        indices: torch.Tensor,
        k: int | None = None,
        *,
        detach_codebook: bool = False,
    ) -> torch.Tensor:
        """Decode index tokens back to BCHW q with family-correct layout."""

        k = self._checked_k(k)
        self._validate_indices(indices, k)
        prefix = self.codebook_at_k(k)
        if indices.device != prefix.device:
            raise ValueError(
                f"indices and codebook must share a device, got {indices.device} and {prefix.device}"
            )
        if detach_codebook:
            prefix = prefix.detach()
        flat_q = F.embedding(indices.long().reshape(-1), _flatten_codebook(prefix))
        if self.family == "image-vq":
            bsz, h, w = indices.shape
            if (int(h), int(w)) != (self.h, self.w):
                raise ValueError(f"image-vq index map must be [{self.h},{self.w}], got [{h},{w}]")
            return flat_q.view(bsz, h, w, self.channels).permute(0, 3, 1, 2).contiguous()
        bsz, channels = indices.shape
        return flat_q.view(bsz, channels, self.h, self.w)

    decode_indices = get_codebook_entry

    def shuffle_tokens(
        self,
        value: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        self._validate_input(value)
        if self.channel_codebook_mode == "grouped":
            if generator is not None and seed is not None:
                raise ValueError("pass generator or seed, not both")
            if seed is not None:
                generator = torch.Generator(device=value.device)
                generator.manual_seed(int(seed))
            shuffled = torch.empty_like(value)
            for channel in range(self.channels):
                permutation = torch.randperm(
                    int(value.shape[0]), device=value.device, generator=generator
                )
                shuffled[:, channel] = value[permutation, channel]
            return shuffled
        return shuffle_vq_tokens(value, self.family, generator=generator, seed=seed)

    shuffle = shuffle_tokens
    shuffle_quantized = shuffle_tokens

    def shuffle_indices(
        self,
        indices: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        if self.channel_codebook_mode == "grouped":
            self._validate_indices(indices, self.k_max)
            if generator is not None and seed is not None:
                raise ValueError("pass generator or seed, not both")
            if seed is not None:
                generator = torch.Generator(device=indices.device)
                generator.manual_seed(int(seed))
            shuffled = torch.empty_like(indices)
            for channel in range(self.channels):
                permutation = torch.randperm(
                    int(indices.shape[0]), device=indices.device, generator=generator
                )
                shuffled[:, channel] = indices[permutation, channel]
            return shuffled
        return shuffle_vq_indices(indices, self.family, generator=generator, seed=seed)

    @staticmethod
    def zero_tokens(value: torch.Tensor) -> torch.Tensor:
        return zero_vq_tokens(value)

    zero = zero_tokens
    zero_quantized = zero_tokens

    @torch.no_grad()
    def update_usage_ema(self, indices: torch.Tensor, k: int | None = None) -> None:
        """Update active-prefix usage EMA and consecutive-inactive counters."""

        k = self._checked_k(k)
        self._validate_indices(indices, k)
        counts = torch.bincount(indices.detach().long().reshape(-1), minlength=k).float()
        counts = counts.to(device=self.ema_count.device)
        self.ema_count[:k].mul_(self.usage_decay).add_(counts, alpha=1.0 - self.usage_decay)
        used = counts > 0
        self.inactive_steps[:k].add_(1)
        self.inactive_steps[:k][used] = 0

    @torch.no_grad()
    def reset_usage(self) -> None:
        self.ema_count.zero_()
        self.inactive_steps.zero_()

    def _tokens_from_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Accept BCHW samples or already flattened family tokens."""

        if not isinstance(samples, torch.Tensor):
            raise TypeError(f"samples must be a tensor, got {type(samples).__name__}")
        if samples.ndim == 4:
            # Initialization samples are commonly accumulated on CPU while the
            # quantizer already lives on CUDA.  Validate the family shape here
            # without imposing the live-forward same-device requirement.
            bsz, channels, h, w = samples.shape
            if int(bsz) < 1 or int(channels) < 1:
                raise ValueError(f"sample batch and channels must be non-empty, got {tuple(samples.shape)}")
            if (int(h), int(w)) != (self.h, self.w):
                raise ValueError(
                    f"expected sample spatial size [{self.h},{self.w}], got {tuple(samples.shape[2:])}"
                )
            if self.family == "image-vq":
                if int(channels) != self.channels:
                    raise ValueError(f"image-vq expects {self.channels} sample channels, got {int(channels)}")
                return samples.permute(0, 2, 3, 1).contiguous().view(bsz * h * w, channels)
            return samples.contiguous().view(bsz * channels, h * w)
        expected_dim = self.channels if self.family == "image-vq" else self.h * self.w
        if self.family == "channel-vq" and samples.ndim == 3 and tuple(samples.shape[1:]) == (self.h, self.w):
            return samples.contiguous().view(int(samples.shape[0]), expected_dim)
        if samples.ndim == 2 and int(samples.shape[1]) == expected_dim:
            return samples
        expected = f"[N,{expected_dim}]"
        if self.family == "channel-vq":
            expected += f" or [N,{self.h},{self.w}]"
        raise ValueError(f"invalid {self.family} samples {tuple(samples.shape)}; expected BCHW or {expected}")

    @torch.no_grad()
    def initialize_random_from_samples(
        self,
        samples: torch.Tensor,
        k: int | None = None,
        *,
        seed: int = 0,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Initialize a prefix from randomly sampled real family tokens."""

        k = self._checked_k(k)
        tokens = self._tokens_from_samples(samples).detach().to(device=self.codebook.device).float()
        if int(tokens.shape[0]) < 1:
            raise ValueError("cannot initialize from empty samples")
        generator = torch.Generator(device=tokens.device)
        generator.manual_seed(int(seed))
        if int(tokens.shape[0]) >= k:
            selected = torch.randperm(int(tokens.shape[0]), device=tokens.device, generator=generator)[:k]
        else:
            selected = torch.randint(0, int(tokens.shape[0]), (k,), device=tokens.device, generator=generator)
        centers = tokens[selected].clone()
        if float(noise_std) != 0.0:
            noise = torch.randn(centers.shape, device=centers.device, dtype=centers.dtype, generator=generator)
            centers.add_(noise, alpha=float(noise_std))
        self.codebook[:k].copy_(centers.view_as(self.codebook[:k]).to(dtype=self.codebook.dtype))
        self.ema_count[:k].zero_()
        self.inactive_steps[:k].zero_()
        return self.codebook[:k].detach().clone()

    def _kmeans_random_centers(
        self,
        tokens: torch.Tensor,
        k: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        count = int(tokens.shape[0])
        if count >= k:
            selected = torch.randperm(count, device=tokens.device, generator=generator)[:k]
        else:
            selected = torch.randint(0, count, (k,), device=tokens.device, generator=generator)
        return tokens[selected].clone()

    @torch.no_grad()
    def initialize_kmeans_from_samples(
        self,
        samples: torch.Tensor,
        k: int | None = None,
        *,
        iterations: int = 20,
        max_samples: int = 100_000,
        seed: int = 0,
        tolerance: float = 1e-4,
    ) -> torch.Tensor:
        """Run exact-assignment Lloyd k-means and copy centers into a prefix.

        Empty clusters are reseeded from the currently highest-error samples.
        For ``N < K`` duplicate samples are allowed, which keeps this hook
        usable for small smoke datasets while making the limitation explicit in
        the resulting usage metrics.
        """

        k = self._checked_k(k)
        iterations = _positive_int("iterations", iterations)
        max_samples = _positive_int("max_samples", max_samples)
        tokens = self._tokens_from_samples(samples).detach().to(device=self.codebook.device).float()
        if int(tokens.shape[0]) < 1:
            raise ValueError("cannot initialize from empty samples")
        generator = torch.Generator(device=tokens.device)
        generator.manual_seed(int(seed))
        if int(tokens.shape[0]) > max_samples:
            selected = torch.randperm(
                int(tokens.shape[0]), device=tokens.device, generator=generator
            )[:max_samples]
            tokens = tokens[selected]
        centers = self._kmeans_random_centers(tokens, k, generator)

        for _ in range(iterations):
            assignments, errors = nearest_codebook_indices_chunked(
                tokens,
                centers,
                query_chunk_size=self.query_chunk_size,
                codebook_chunk_size=self.codebook_chunk_size,
                return_distances=True,
            )
            counts = torch.bincount(assignments, minlength=k)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, assignments, tokens)
            updated = sums / counts.clamp_min(1).to(dtype=sums.dtype).unsqueeze(1)
            empty = counts == 0
            if bool(empty.any()):
                num_empty = int(empty.sum().item())
                high_error = errors.argsort(descending=True)
                if int(high_error.numel()) < num_empty:
                    repeats = math.ceil(num_empty / int(high_error.numel()))
                    high_error = high_error.repeat(repeats)
                updated[empty] = tokens[high_error[:num_empty]]
            movement = (updated - centers).square().sum(dim=1).sqrt().max()
            centers = updated
            if float(movement.item()) <= float(tolerance):
                break

        self.codebook[:k].copy_(centers.view_as(self.codebook[:k]).to(dtype=self.codebook.dtype))
        self.ema_count[:k].zero_()
        self.inactive_steps[:k].zero_()
        return self.codebook[:k].detach().clone()

    @torch.no_grad()
    def initialize_from_samples(
        self,
        samples: torch.Tensor,
        k: int | None = None,
        *,
        method: str = "random",
        **kwargs: object,
    ) -> torch.Tensor:
        """Dispatch to random-sample or k-means initialization."""

        method = str(method).strip().lower().replace("_", "-")
        if method in {"random", "sample", "random-sample"}:
            return self.initialize_random_from_samples(samples, k, **kwargs)
        if method in {"kmeans", "k-means", "lloyd"}:
            return self.initialize_kmeans_from_samples(samples, k, **kwargs)
        raise ValueError(f"unknown initialization method {method!r}")

    init_from_samples = initialize_from_samples

    @torch.no_grad()
    def refresh_dead_codes(
        self,
        tokens: torch.Tensor,
        indices: torch.Tensor | None = None,
        *,
        k: int | None = None,
        threshold: float = 1e-3,
        patience: int = 0,
        strategy: str = "high-error",
        max_refresh: int | None = None,
        seed: int = 0,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Replace persistently dead prefix codes with current real tokens.

        The method is explicit rather than mutating the codebook unexpectedly
        during ``forward``.  It returns refreshed code indices so a trainer may
        also clear row-wise optimizer state when desired.  Grouped channel-VQ
        preserves row ownership: a dead row ``round*C+c`` is refreshed only
        from channel-``c`` maps in the current BCHW sample batch.
        """

        k = self._checked_k(k)
        threshold = float(threshold)
        patience = int(patience)
        if threshold < 0.0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")
        if self.channel_codebook_mode == "grouped":
            if tokens.ndim != 4 or int(tokens.shape[1]) != self.channels:
                raise ValueError(
                    "grouped dead-code refresh requires BCHW samples so each replacement "
                    f"retains its channel owner; got {tuple(tokens.shape)}"
                )
            if tuple(int(value) for value in tokens.shape[2:]) != (self.h, self.w):
                raise ValueError(
                    f"grouped refresh expects spatial size [{self.h},{self.w}], "
                    f"got {tuple(tokens.shape[2:])}"
                )
        flat_tokens = self._tokens_from_samples(tokens).detach().to(device=self.codebook.device).float()
        if int(flat_tokens.shape[0]) < 1:
            raise ValueError("cannot refresh from empty tokens")
        dead_mask = self.ema_count[:k] <= threshold
        if patience > 0:
            dead_mask &= self.inactive_steps[:k] >= patience
        dead = dead_mask.nonzero(as_tuple=False).flatten()
        if max_refresh is not None:
            max_refresh = _positive_int("max_refresh", max_refresh)
            dead = dead[:max_refresh]
        if int(dead.numel()) == 0:
            return dead

        strategy = str(strategy).strip().lower().replace("_", "-")
        generator = torch.Generator(device=flat_tokens.device)
        generator.manual_seed(int(seed))
        count = int(dead.numel())
        if self.channel_codebook_mode == "grouped":
            # BCHW flattening is [b0c0, b0c1, ..., b1c0, ...].  Keep the
            # explicit [B,C,D] view so every dead code can sample only maps
            # belonging to its row owner ``row % C``.
            grouped_tokens = flat_tokens.view(int(tokens.shape[0]), self.channels, -1)
            if strategy in {"high-error", "error", "hard"}:
                if indices is None:
                    sample_values = tokens.detach().to(
                        device=self.codebook.device,
                        dtype=self.codebook.dtype,
                    )
                    q_hard, selected_indices = self.quantize_input(
                        sample_values,
                        k,
                        detach_codebook=True,
                    )
                    errors_by_channel = (
                        sample_values.float() - q_hard.float()
                    ).square().flatten(2).sum(dim=2)
                    del selected_indices
                else:
                    self._validate_indices(indices, k)
                    if tuple(indices.shape) != (int(tokens.shape[0]), self.channels):
                        raise ValueError(
                            "grouped refresh indices must align with BCHW samples; "
                            f"got indices={tuple(indices.shape)} samples={tuple(tokens.shape)}"
                        )
                    flat_indices = indices.detach().long().reshape(-1).to(device=flat_tokens.device)
                    selected_codes = _flatten_codebook(self.codebook_at_k(k).detach()).float()[flat_indices]
                    errors_by_channel = (
                        flat_tokens - selected_codes
                    ).square().sum(dim=1).view(int(tokens.shape[0]), self.channels)
                replacements = torch.empty(
                    count,
                    int(grouped_tokens.shape[2]),
                    device=flat_tokens.device,
                    dtype=flat_tokens.dtype,
                )
                for channel in (dead % self.channels).unique(sorted=True).tolist():
                    positions = ((dead % self.channels) == int(channel)).nonzero(as_tuple=False).flatten()
                    ranked = errors_by_channel[:, int(channel)].argsort(descending=True)
                    needed = int(positions.numel())
                    if int(ranked.numel()) < needed:
                        ranked = ranked.repeat(math.ceil(needed / int(ranked.numel())))
                    replacements[positions] = grouped_tokens[ranked[:needed], int(channel)]
            elif strategy in {"random", "sample", "random-sample"}:
                replacements = torch.empty(
                    count,
                    int(grouped_tokens.shape[2]),
                    device=flat_tokens.device,
                    dtype=flat_tokens.dtype,
                )
                for channel in (dead % self.channels).unique(sorted=True).tolist():
                    positions = ((dead % self.channels) == int(channel)).nonzero(as_tuple=False).flatten()
                    selected = torch.randint(
                        0,
                        int(grouped_tokens.shape[0]),
                        (int(positions.numel()),),
                        device=flat_tokens.device,
                        generator=generator,
                    )
                    replacements[positions] = grouped_tokens[selected, int(channel)]
            else:
                raise ValueError(f"unknown dead-code refresh strategy {strategy!r}")
        elif strategy in {"high-error", "error", "hard"}:
            if indices is None:
                flat_indices, errors = nearest_codebook_indices_chunked(
                    flat_tokens,
                    self.codebook_at_k(k),
                    query_chunk_size=self.query_chunk_size,
                    codebook_chunk_size=self.codebook_chunk_size,
                    return_distances=True,
                )
            else:
                self._validate_indices(indices, k)
                flat_indices = indices.detach().long().reshape(-1).to(device=flat_tokens.device)
                if int(flat_indices.numel()) != int(flat_tokens.shape[0]):
                    raise ValueError(
                        f"indices contain {int(flat_indices.numel())} tokens but samples contain "
                        f"{int(flat_tokens.shape[0])}"
                    )
                selected_codes = _flatten_codebook(self.codebook_at_k(k).detach()).float()[flat_indices]
                errors = (flat_tokens - selected_codes).square().sum(dim=1)
            del flat_indices
            selected = errors.argsort(descending=True)
            if int(selected.numel()) < count:
                selected = selected.repeat(math.ceil(count / int(selected.numel())))
            replacements = flat_tokens[selected[:count]].clone()
        elif strategy in {"random", "sample", "random-sample"}:
            selected = torch.randint(
                0, int(flat_tokens.shape[0]), (count,), device=flat_tokens.device, generator=generator
            )
            replacements = flat_tokens[selected].clone()
        else:
            raise ValueError(f"unknown dead-code refresh strategy {strategy!r}")

        if float(noise_std) != 0.0:
            noise = torch.randn(
                replacements.shape,
                device=replacements.device,
                dtype=replacements.dtype,
                generator=generator,
            )
            replacements.add_(noise, alpha=float(noise_std))
        replacement_rows = replacements.reshape(
            int(dead.numel()), *self.embedding_shape
        ).to(dtype=self.codebook.dtype)
        # ``self.codebook[dead].copy_(...)`` would mutate an advanced-indexing
        # temporary rather than the Parameter itself.  ``index_copy_`` is the
        # required in-place row update for both grouped and legacy modes.
        self.codebook.index_copy_(0, dead, replacement_rows)
        alive = self.ema_count[:k][~dead_mask]
        reset_count = max(2.0 * threshold, float(alive.mean().item()) if int(alive.numel()) else 1.0)
        self.ema_count[dead] = reset_count
        self.inactive_steps[dead] = 0
        return dead

    @torch.no_grad()
    def usage_metrics(self, indices: torch.Tensor, k: int | None = None) -> dict[str, float]:
        k = self._checked_k(k)
        self._validate_indices(indices, k)
        return usage_metrics(indices, k)

    @torch.no_grad()
    def codebook_metrics(
        self,
        k: int | None = None,
        *,
        l2_max_samples: int = 512,
        rank_max_samples: int = 2048,
        seed: int = 0,
    ) -> dict[str, float]:
        return codebook_health_metrics(
            self.codebook_at_k(k),
            l2_max_samples=l2_max_samples,
            rank_max_samples=rank_max_samples,
            seed=seed,
        )


class ImageVQ(NestedPrefixVQ):
    """Convenience constructor for image-VQ shared prefixes."""

    def __init__(self, k_max: int | None = None, **kwargs: object) -> None:
        super().__init__("image-vq", k_max, **kwargs)


class ChannelVQ(NestedPrefixVQ):
    """Convenience constructor for channel-VQ shared prefixes."""

    def __init__(self, k_max: int | None = None, **kwargs: object) -> None:
        super().__init__("channel-vq", k_max, **kwargs)


# Descriptive aliases make call sites readable while retaining one canonical
# implementation and one state-dict layout.
NestedSharedPrefixVQ = NestedPrefixVQ
NestedSharedPrefixVectorQuantizer = NestedPrefixVQ
ImageVectorNestedVQ = ImageVQ
ChannelMapNestedVQ = ChannelVQ


def build_nested_vq(family: str, k_max: int, **kwargs: object) -> NestedPrefixVQ:
    """Small factory used by CLI-driven experiment scripts."""

    return NestedPrefixVQ(family, k_max, **kwargs)


__all__ = [
    "ChannelMapNestedVQ",
    "ChannelVQ",
    "ImageVQ",
    "ImageVectorNestedVQ",
    "NestedPrefixVQ",
    "NestedSharedPrefixVQ",
    "NestedSharedPrefixVectorQuantizer",
    "VQFamily",
    "build_nested_vq",
    "canonical_vq_family",
    "codebook_effective_rank_metrics",
    "codebook_health_metrics",
    "nearest_codebook_chunked",
    "nearest_codebook_indices_chunked",
    "sampled_codebook_l2_metrics",
    "shuffle_vq_indices",
    "shuffle_vq_tokens",
    "usage_metrics",
    "usage_statistics",
    "zero_vq_tokens",
]
