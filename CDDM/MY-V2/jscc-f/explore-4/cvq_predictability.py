"""z1-only conditional rate objective for channel-VQ tokenizer training.

The receiver later has only z1 and previously generated channel indices.  A
tokenizer trained only for oracle reconstruction can therefore place useful
image information in indices with high H(I | z1, I_<k), making even a strong
CAR unable to recover it.  This module supplies the conditional-rate term

    E_{q_soft(i_k | z_k)}[-log p_phi(i_k | z1, i_<k)]

while the ordinary hard-index CE trains p_phi.  The expected term is detached
from the prior logits, so its gradient makes E2/codebook assignments more
predictable rather than allowing the prior to chase arbitrary assignments.
It never accepts x1, img, z2, or q2 as a conditioning input.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Z1ConditionalChannelPrior(nn.Module):
    """Small causal visual prior p(index_k | z1, index_<k)."""

    def __init__(
        self,
        z1_channels: int,
        channels: int,
        vocabulary: int,
        *,
        hidden: int = 192,
        layers: int = 4,
        heads: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("conditional-prior hidden must divide heads")
        self.channels = int(channels)
        self.vocabulary = int(vocabulary)
        self.hidden = int(hidden)
        groups = min(16, hidden)
        while hidden % groups:
            groups -= 1
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(int(z1_channels), hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden), nn.SiLU(),
        )
        self.channel_queries = nn.Embedding(self.channels, hidden)
        self.cross_attention = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=dropout)
        # The final row is a BOS visual-token marker.
        self.token_embedding = nn.Embedding(self.vocabulary + 1, hidden)
        self.position_embedding = nn.Embedding(self.channels, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            batch_first=True, norm_first=True, activation="gelu", dropout=dropout,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=int(layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, self.vocabulary)

    def _context(self, z1: torch.Tensor) -> torch.Tensor:
        if z1.ndim != 4:
            raise ValueError(f"z1 must be BCHW, got {tuple(z1.shape)}")
        memory = self.condition_encoder(z1).flatten(2).transpose(1, 2)
        queries = self.channel_queries.weight.unsqueeze(0).expand(int(z1.shape[0]), -1, -1)
        context, _ = self.cross_attention(queries, memory, memory, need_weights=False)
        return context + queries

    def forward_teacher(self, z1: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 2 or int(indices.shape[1]) != self.channels:
            raise ValueError(f"indices must be [B,{self.channels}], got {tuple(indices.shape)}")
        if int(indices.min()) < 0 or int(indices.max()) >= self.vocabulary:
            raise ValueError("conditional-prior target index out of vocabulary")
        batch = int(indices.shape[0])
        bos = torch.full((batch, 1), self.vocabulary, device=indices.device, dtype=torch.long)
        shifted = torch.cat([bos, indices[:, :-1].long()], dim=1)
        positions = torch.arange(self.channels, device=indices.device).view(1, -1)
        value = self.token_embedding(shifted) + self.position_embedding(positions) + self._context(z1)
        causal = torch.ones((self.channels, self.channels), dtype=torch.bool, device=indices.device).triu(1)
        return self.head(self.norm(self.decoder(value, mask=causal)))


def soft_channel_assignments(z: torch.Tensor, quantizer, rate: int, temperature: float) -> torch.Tensor:
    """Differentiable p_soft(index | complete channel map), shape [B,C,K]."""

    if z.ndim != 4:
        raise ValueError(f"channel CVQ tensor must be BCHW, got {tuple(z.shape)}")
    if float(temperature) <= 0.0:
        raise ValueError("soft assignment temperature must be positive")
    tokens = quantizer.flatten_tokens(z).float()
    codebook = quantizer.codebook_at_k(int(rate)).reshape(int(rate), -1).float()
    distance = (
        tokens.square().sum(dim=1, keepdim=True)
        + codebook.square().sum(dim=1).unsqueeze(0)
        - 2.0 * tokens @ codebook.t()
    )
    probabilities = F.softmax(-distance / float(temperature), dim=-1)
    return probabilities.view(int(z.shape[0]), int(z.shape[1]), int(rate))


def masked_hard_index_nll(logits: torch.Tensor, indices: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Prior CE; only the active nested prefix contributes when supplied."""

    losses = F.cross_entropy(logits.flatten(0, 1).float(), indices.flatten().long(), reduction="none").view_as(indices)
    if active_mask is None:
        return losses.mean()
    mask = active_mask[:, :, 0, 0].to(device=losses.device, dtype=losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def masked_soft_rate_nll(
    soft_assignments: torch.Tensor,
    logits: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rate surrogate for E2/codebook plus its assignment entropy diagnostic."""

    if tuple(soft_assignments.shape) != tuple(logits.shape):
        raise ValueError(f"soft assignment/logit mismatch {tuple(soft_assignments.shape)} vs {tuple(logits.shape)}")
    # Detaching p_phi is deliberate: the predictor is fitted by hard CE while
    # this RD term moves assignments toward what the receiver can predict.
    nll = -(soft_assignments * F.log_softmax(logits.float(), dim=-1).detach()).sum(dim=-1)
    entropy = -(soft_assignments * soft_assignments.clamp_min(1e-12).log()).sum(dim=-1)
    if active_mask is None:
        return nll.mean(), entropy.mean()
    mask = active_mask[:, :, 0, 0].to(device=nll.device, dtype=nll.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (nll * mask).sum() / denom, (entropy * mask).sum() / denom


def masked_channel_marginal_entropy_deficit(
    soft_assignments: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``1-H(I_k)/log(K)`` for the same soft assignments as the rate.

    The conditional-rate term alone minimizes ``H(I_k | z1, I_<k)``.  That
    has a degenerate optimum in which every channel selects one easy-to-predict
    code.  For a *positioned* CAR sequence, the matching marginal is per
    channel position, not one histogram pooled across all channel positions:

        mean_k H( E_batch q(i_k | z_k) ).

    Together with :func:`masked_soft_rate_nll`, minimizing this deficit
    maximizes the conditional mutual information
    ``I(I_k ; z1,I_<k | k)``.  The optional nested-dropout mask is deliberately
    shared with the rate term, so an inactive suffix cannot manufacture code
    diversity that the decoder did not train on in that step.
    """

    if soft_assignments.ndim != 3:
        raise ValueError(f"soft assignments must be [B,C,K], got {tuple(soft_assignments.shape)}")
    _batch, channels, vocabulary = map(int, soft_assignments.shape)
    if vocabulary < 2:
        raise ValueError("marginal entropy needs a vocabulary of at least two codes")
    if active_mask is None:
        marginal = soft_assignments.mean(dim=0)
        valid = torch.ones(channels, device=soft_assignments.device, dtype=torch.bool)
    else:
        if tuple(active_mask.shape[:2]) != tuple(soft_assignments.shape[:2]):
            raise ValueError(
                f"active mask/assignment mismatch {tuple(active_mask.shape)} vs {tuple(soft_assignments.shape)}"
            )
        weights = active_mask[:, :, 0, 0].to(device=soft_assignments.device, dtype=soft_assignments.dtype)
        counts = weights.sum(dim=0)
        marginal = (soft_assignments * weights.unsqueeze(-1)).sum(dim=0) / counts.clamp_min(1.0).unsqueeze(-1)
        valid = counts > 0
    entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum(dim=-1)
    ratio = entropy / math.log(float(vocabulary))
    if not bool(valid.any()):
        zero = soft_assignments.new_zeros(())
        return zero, zero
    ratio = ratio[valid].mean()
    return 1.0 - ratio, ratio
