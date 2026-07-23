#!/usr/bin/env python3
"""Receiver-only q2 predictors shared by FSQ and later VQ experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from contracts import ReceiverCondition


def _groups(channels: int, maximum: int = 32) -> int:
    for groups in range(min(int(maximum), int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class GatedResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = int(channels) * int(expansion)
        self.norm = nn.GroupNorm(_groups(channels), int(channels))
        self.in_proj = nn.Conv2d(int(channels), hidden * 2, 3, padding=1)
        self.out_proj = nn.Conv2d(hidden, int(channels), 3, padding=1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(F.silu(self.norm(x))).chunk(2, dim=1)
        return x + self.scale * self.out_proj(value * torch.sigmoid(gate))


class SpatialAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if int(channels) % int(heads) != 0:
            raise ValueError(f"hidden={channels} must be divisible by heads={heads}")
        self.norm1 = nn.LayerNorm(int(channels))
        self.attn = nn.MultiheadAttention(int(channels), int(heads), batch_first=True)
        self.norm2 = nn.LayerNorm(int(channels))
        self.mlp = nn.Sequential(
            nn.Linear(int(channels), int(channels) * 2),
            nn.GELU(),
            nn.Linear(int(channels) * 2, int(channels)),
        )
        self.attn_scale = nn.Parameter(torch.tensor(0.1))
        self.mlp_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        attended, _weights = self.attn(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.attn_scale * attended
        tokens = tokens + self.mlp_scale * self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(bsz, channels, height, width).contiguous()


class ReceiverTrunk(nn.Module):
    """Fuse received z1 and its decoded image x1 at latent resolution."""

    def __init__(
        self,
        z1_channels: int,
        hidden: int = 128,
        blocks: int = 8,
        attention_every: int = 2,
        heads: int = 4,
        condition_mode: str = "z1_x1",
    ) -> None:
        super().__init__()
        self.condition_mode = str(condition_mode)
        if self.condition_mode not in {"z1", "x1", "z1_x1"}:
            raise ValueError(f"unknown receiver condition mode {self.condition_mode!r}")
        self.z1_stem = nn.Sequential(
            nn.Conv2d(int(z1_channels), int(hidden), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(int(hidden), int(hidden), 3, padding=1),
        )
        self.x1_stem = nn.Sequential(
            nn.Conv2d(3, int(hidden) // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(int(hidden) // 2, int(hidden), 3, padding=1),
        )
        fuse_parts = 1 if self.condition_mode != "z1_x1" else 4
        self.fuse = nn.Sequential(
            nn.Conv2d(int(hidden) * fuse_parts, int(hidden), 1),
            nn.SiLU(),
            nn.Conv2d(int(hidden), int(hidden), 3, padding=1),
        )
        body: list[nn.Module] = []
        for index in range(int(blocks)):
            body.append(GatedResidualBlock(int(hidden)))
            if int(attention_every) > 0 and (index + 1) % int(attention_every) == 0:
                body.append(SpatialAttentionBlock(int(hidden), int(heads)))
        self.body = nn.Sequential(*body)
        self.out_norm = nn.GroupNorm(_groups(hidden), int(hidden))

    def forward(self, condition: ReceiverCondition) -> torch.Tensor:
        condition.validate()
        z1 = condition.z1
        x1_small = F.interpolate(
            condition.x1,
            size=tuple(z1.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        z_feature = self.z1_stem(z1)
        x_feature = self.x1_stem(x1_small)
        if self.condition_mode == "z1":
            parts = [z_feature]
        elif self.condition_mode == "x1":
            parts = [x_feature]
        else:
            parts = [z_feature, x_feature, z_feature - x_feature, z_feature * x_feature]
        return F.silu(self.out_norm(self.body(self.fuse(torch.cat(parts, dim=1)))))


class ChannelVQAutoregressiveIndexPredictor(nn.Module):
    """Conditional AR model for channel-VQ index sequences.

    Deployment ``forward`` consumes only the receiver condition and its own
    previously generated indices.  ``local_code_count`` is the per-channel
    grouped vocabulary for grouped VQ and the full K-prefix vocabulary for
    global channel-VQ.  ``forward_teacher`` is a training-only likelihood path
    whose shifted targets never expose the current/future token.
    """

    def __init__(
        self,
        z1_channels: int,
        channels: int,
        local_code_count: int,
        *,
        hidden: int = 192,
        blocks: int = 6,
        attention_every: int = 2,
        heads: int = 4,
        condition_mode: str = "z1_x1",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.local_code_count = int(local_code_count)
        self.hidden = int(hidden)
        if self.channels < 1 or self.local_code_count < 2:
            raise ValueError(
                f"channel AR requires channels>=1 and at least two local codes, "
                f"got C={self.channels}, R={self.local_code_count}"
            )
        if self.hidden % int(heads) != 0:
            raise ValueError(f"hidden={self.hidden} must be divisible by heads={heads}")
        self.trunk = ReceiverTrunk(
            int(z1_channels),
            hidden=self.hidden,
            blocks=int(blocks),
            attention_every=int(attention_every),
            heads=int(heads),
            condition_mode=str(condition_mode),
        )
        self.channel_queries = nn.Embedding(self.channels, self.hidden)
        self.cross_norm = nn.LayerNorm(self.hidden)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden, int(heads), batch_first=True
        )
        # Row R is the start token; generated indices occupy [0,R).
        self.index_embedding = nn.Embedding(self.local_code_count + 1, self.hidden)
        self.input_projection = nn.Linear(self.hidden * 2, self.hidden)
        self.initial_hidden = nn.Linear(self.hidden, self.hidden)
        self.gru = nn.GRU(self.hidden, self.hidden, batch_first=True)
        self.logit_head = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.local_code_count),
        )

    def _condition_tokens(
        self, condition: ReceiverCondition
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.trunk(condition)
        batch = int(feature.shape[0])
        memory = feature.flatten(2).transpose(1, 2)
        queries = self.channel_queries.weight.unsqueeze(0).expand(batch, -1, -1)
        channel_context, _weights = self.cross_attention(
            self.cross_norm(queries), self.cross_norm(memory), memory, need_weights=False
        )
        channel_context = queries + channel_context
        initial = torch.tanh(self.initial_hidden(memory.mean(dim=1))).unsqueeze(0)
        return channel_context, initial

    def _checked_targets(self, local_indices: torch.Tensor) -> torch.Tensor:
        if local_indices.ndim != 2 or tuple(int(v) for v in local_indices.shape[1:]) != (
            self.channels,
        ):
            raise ValueError(
                f"channel AR indices must be [B,{self.channels}], "
                f"got {tuple(local_indices.shape)}"
            )
        target = local_indices.long()
        if int(target.numel()) < 1 or int(target.min()) < 0 or int(target.max()) >= self.local_code_count:
            raise ValueError(f"channel AR indices must lie in [0,{self.local_code_count})")
        return target

    def forward_teacher(
        self, condition: ReceiverCondition, local_indices: torch.Tensor
    ) -> torch.Tensor:
        target = self._checked_targets(local_indices)
        context, hidden = self._condition_tokens(condition)
        start = torch.full(
            (int(target.shape[0]), 1),
            self.local_code_count,
            device=target.device,
            dtype=torch.long,
        )
        shifted = torch.cat([start, target[:, :-1]], dim=1)
        inputs = self.input_projection(
            torch.cat([self.index_embedding(shifted), context], dim=-1)
        )
        states, _hidden = self.gru(inputs, hidden)
        return self.logit_head(states)

    def forward(self, condition: ReceiverCondition) -> tuple[torch.Tensor, torch.Tensor]:
        context, hidden = self._condition_tokens(condition)
        batch = int(context.shape[0])
        previous = torch.full(
            (batch,),
            self.local_code_count,
            device=context.device,
            dtype=torch.long,
        )
        indices: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for channel in range(self.channels):
            step = self.input_projection(
                torch.cat(
                    [self.index_embedding(previous), context[:, channel]], dim=-1
                )
            ).unsqueeze(1)
            state, hidden = self.gru(step, hidden)
            step_logits = self.logit_head(state[:, 0])
            previous = step_logits.argmax(dim=-1)
            logits.append(step_logits)
            indices.append(previous)
        return torch.stack(logits, dim=1), torch.stack(indices, dim=1)


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / float(max(1, half - 1))
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        value = torch.cat([angles.sin(), angles.cos()], dim=1)
        if int(value.shape[1]) < self.channels:
            value = F.pad(value, (0, self.channels - int(value.shape[1])))
        return value


class ImageVQConditionalDenoiser(nn.Module):
    """Noise estimator for diffusion directly in image-VQ embedding space."""

    def __init__(
        self,
        z1_channels: int,
        embedding_dim: int,
        *,
        hidden: int = 192,
        blocks: int = 8,
        attention_every: int = 2,
        heads: int = 4,
        condition_mode: str = "z1_x1",
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden = int(hidden)
        self.condition_trunk = ReceiverTrunk(
            int(z1_channels),
            hidden=self.hidden,
            blocks=max(1, int(blocks) // 2),
            attention_every=int(attention_every),
            heads=int(heads),
            condition_mode=str(condition_mode),
        )
        self.noisy_projection = nn.Conv2d(self.embedding_dim, self.hidden, 1)
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(self.hidden),
            nn.Linear(self.hidden, self.hidden * 2),
            nn.SiLU(),
            nn.Linear(self.hidden * 2, self.hidden),
        )
        self.fuse = nn.Conv2d(self.hidden * 4, self.hidden, 1)
        body: list[nn.Module] = []
        for index in range(int(blocks)):
            body.append(GatedResidualBlock(self.hidden))
            if int(attention_every) > 0 and (index + 1) % int(attention_every) == 0:
                body.append(SpatialAttentionBlock(self.hidden, int(heads)))
        self.body = nn.Sequential(*body)
        self.out_norm = nn.GroupNorm(_groups(self.hidden), self.hidden)
        self.output = nn.Conv2d(self.hidden, self.embedding_dim, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy_q: torch.Tensor,
        timesteps: torch.Tensor,
        condition: ReceiverCondition,
    ) -> torch.Tensor:
        condition_feature = self.condition_trunk(condition)
        noisy_feature = self.noisy_projection(noisy_q)
        if tuple(noisy_feature.shape[-2:]) != tuple(condition_feature.shape[-2:]):
            raise ValueError(
                f"diffusion q/condition spatial mismatch: {tuple(noisy_feature.shape)} "
                f"vs {tuple(condition_feature.shape)}"
            )
        time = self.time_embedding(timesteps).view(int(noisy_q.shape[0]), self.hidden, 1, 1)
        noisy_feature = noisy_feature + time
        value = self.fuse(
            torch.cat(
                [
                    noisy_feature,
                    condition_feature,
                    noisy_feature - condition_feature,
                    noisy_feature * condition_feature,
                ],
                dim=1,
            )
        )
        return self.output(F.silu(self.out_norm(self.body(value))))


class ImageVQConditionalDiffusionGenerator(nn.Module):
    """Receiver-only DDIM generator for q2 embeddings.

    The public deployment ``forward(condition)`` samples without any sender
    tensor.  ``training_predictions`` is explicitly supervision-only.
    """

    def __init__(
        self,
        z1_channels: int,
        embedding_dim: int,
        height: int,
        width: int,
        *,
        hidden: int = 192,
        blocks: int = 8,
        attention_every: int = 2,
        heads: int = 4,
        condition_mode: str = "z1_x1",
        diffusion_steps: int = 100,
        sample_steps: int = 20,
        q_scale: float = 1.0,
        residual_mean: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.height = int(height)
        self.width = int(width)
        self.diffusion_steps = int(diffusion_steps)
        self.sample_steps = int(sample_steps)
        self.residual_mean = bool(residual_mean)
        if self.diffusion_steps < 2 or not 1 <= self.sample_steps <= self.diffusion_steps:
            raise ValueError("diffusion requires 2<=diffusion_steps and 1<=sample_steps<=steps")
        self.denoiser = ImageVQConditionalDenoiser(
            int(z1_channels),
            self.embedding_dim,
            hidden=int(hidden),
            blocks=int(blocks),
            attention_every=int(attention_every),
            heads=int(heads),
            condition_mode=str(condition_mode),
        )
        if self.residual_mean:
            self.mean_trunk = ReceiverTrunk(
                int(z1_channels),
                hidden=int(hidden),
                blocks=max(1, int(blocks) // 2),
                attention_every=int(attention_every),
                heads=int(heads),
                condition_mode=str(condition_mode),
            )
            self.mean_head = nn.Conv2d(int(hidden), self.embedding_dim, 3, padding=1)
            nn.init.zeros_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)
        else:
            self.mean_trunk = None
            self.mean_head = None
        beta = torch.linspace(1e-4, 2e-2, self.diffusion_steps, dtype=torch.float32)
        alpha = 1.0 - beta
        self.register_buffer("alpha_bar", torch.cumprod(alpha, dim=0))
        self.register_buffer("q_scale", torch.tensor(float(max(q_scale, 1e-6))))
        self.register_buffer("residual_sample_scale", torch.tensor(1.0))

    def set_residual_sample_scale(self, value: float) -> None:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"residual diffusion scale must lie in [0,1], got {value}")
        self.residual_sample_scale.fill_(value)

    def predict_mean(self, condition: ReceiverCondition) -> torch.Tensor:
        if self.mean_trunk is None or self.mean_head is None:
            return torch.zeros(
                int(condition.z1.shape[0]),
                self.embedding_dim,
                self.height,
                self.width,
                device=condition.z1.device,
                dtype=condition.z1.dtype,
            )
        return self.mean_head(self.mean_trunk(condition))

    def training_predictions(
        self, condition: ReceiverCondition, q_target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if tuple(int(v) for v in q_target.shape[1:]) != (
            self.embedding_dim,
            self.height,
            self.width,
        ):
            raise ValueError(
                f"diffusion target must be [B,{self.embedding_dim},{self.height},{self.width}], "
                f"got {tuple(q_target.shape)}"
            )
        batch = int(q_target.shape[0])
        timesteps = torch.randint(
            0, self.diffusion_steps, (batch,), device=q_target.device, dtype=torch.long
        )
        noise = torch.randn_like(q_target)
        mean_q = self.predict_mean(condition)
        target = (q_target - mean_q.detach()) / self.q_scale
        alpha_bar = self.alpha_bar[timesteps].view(batch, 1, 1, 1)
        noisy = alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise
        predicted_noise = self.denoiser(noisy, timesteps, condition)
        predicted_x0 = (
            noisy - (1.0 - alpha_bar).sqrt() * predicted_noise
        ) / alpha_bar.sqrt().clamp_min(1e-6)
        return {
            "timesteps": timesteps,
            "noise": noise,
            "predicted_noise": predicted_noise,
            "predicted_q": mean_q
            + self.residual_sample_scale
            * predicted_x0.clamp(-6.0, 6.0)
            * self.q_scale,
            "predicted_mean_q": mean_q,
            "loss_mean_q": F.mse_loss(mean_q.float(), q_target.float())
            / self.q_scale.float().square().clamp_min(1e-6),
        }

    def forward(self, condition: ReceiverCondition) -> torch.Tensor:
        condition.validate()
        batch = int(condition.z1.shape[0])
        mean_q = self.predict_mean(condition)
        value = torch.randn(
            batch,
            self.embedding_dim,
            self.height,
            self.width,
            device=condition.z1.device,
            dtype=condition.z1.dtype,
        )
        schedule = torch.linspace(
            self.diffusion_steps - 1,
            0,
            self.sample_steps,
            device=condition.z1.device,
        ).round().long().unique_consecutive()
        predicted_x0 = value
        for index, timestep in enumerate(schedule):
            times = torch.full(
                (batch,), int(timestep), device=value.device, dtype=torch.long
            )
            alpha_bar = self.alpha_bar[int(timestep)]
            predicted_noise = self.denoiser(value, times, condition)
            predicted_x0 = (
                value - (1.0 - alpha_bar).sqrt() * predicted_noise
            ) / alpha_bar.sqrt().clamp_min(1e-6)
            predicted_x0 = predicted_x0.clamp(-6.0, 6.0)
            if index + 1 < int(schedule.numel()):
                next_alpha_bar = self.alpha_bar[int(schedule[index + 1])]
                value = (
                    next_alpha_bar.sqrt() * predicted_x0
                    + (1.0 - next_alpha_bar).sqrt() * predicted_noise
                )
            else:
                value = predicted_x0
        return mean_q + self.residual_sample_scale * value * self.q_scale


@dataclass
class ReceiverPrediction:
    q_continuous: torch.Tensor
    q_hard: torch.Tensor
    q_train: torch.Tensor
    logits: list[torch.Tensor]
    codes: torch.Tensor | None
    joint_indices: torch.Tensor | None = None
    q_base: torch.Tensor | None = None
    q_residual: torch.Tensor | None = None


def fsq_codes_to_q(codes: torch.Tensor, levels: Sequence[int]) -> torch.Tensor:
    if codes.ndim != 4 or int(codes.shape[1]) != len(levels):
        raise ValueError(f"expected FSQ codes [B,{len(levels)},H,W], got {tuple(codes.shape)}")
    span = torch.tensor(
        [max(1, int(level) - 1) for level in levels],
        device=codes.device,
        dtype=torch.float32,
    ).view(1, -1, 1, 1)
    return codes.float() / span * 2.0 - 1.0


def snap_to_fsq(q: torch.Tensor, levels: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    span = torch.tensor(
        [max(1, int(level) - 1) for level in levels],
        device=q.device,
        dtype=q.dtype,
    ).view(1, -1, 1, 1)
    positions = ((q.clamp(-1.0, 1.0) + 1.0) * 0.5 * span).round().clamp_min(0.0).minimum(span)
    q_hard = positions / span * 2.0 - 1.0
    return q_hard, positions.long()


class DirectQPredictor(nn.Module):
    """Predict q2 values directly; optional STE snapping keeps them on FSQ."""

    def __init__(
        self,
        z1_channels: int,
        out_channels: int,
        levels: Sequence[int],
        *,
        hidden: int,
        blocks: int,
        attention_every: int,
        heads: int,
        condition_mode: str,
        hard_fsq: bool,
    ) -> None:
        super().__init__()
        self.levels = tuple(int(level) for level in levels)
        self.hard_fsq = bool(hard_fsq)
        self.trunk = ReceiverTrunk(
            z1_channels,
            hidden,
            blocks,
            attention_every,
            heads,
            condition_mode,
        )
        self.head = nn.Sequential(
            nn.Conv2d(int(hidden), int(hidden), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(int(hidden), int(out_channels), 3, padding=1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

    def forward(self, condition: ReceiverCondition) -> ReceiverPrediction:
        q_continuous = self.head(self.trunk(condition))
        q_hard, codes = snap_to_fsq(q_continuous, self.levels)
        q_train = q_continuous + (q_hard - q_continuous).detach() if self.hard_fsq else q_continuous
        return ReceiverPrediction(q_continuous, q_hard, q_train, [], codes)


class ParallelFSQIndexPredictor(nn.Module):
    """Parallel receiver-only prediction of each FSQ scalar index."""

    def __init__(
        self,
        z1_channels: int,
        levels: Sequence[int],
        *,
        hidden: int,
        blocks: int,
        attention_every: int,
        heads: int,
        condition_mode: str,
    ) -> None:
        super().__init__()
        self.levels = tuple(int(level) for level in levels)
        self.trunk = ReceiverTrunk(
            z1_channels,
            hidden,
            blocks,
            attention_every,
            heads,
            condition_mode,
        )
        self.heads = nn.ModuleList(nn.Conv2d(int(hidden), level, 1) for level in self.levels)
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, condition: ReceiverCondition) -> ReceiverPrediction:
        feature = self.trunk(condition)
        logits = [head(feature) for head in self.heads]
        soft_parts: list[torch.Tensor] = []
        hard_codes: list[torch.Tensor] = []
        for level, channel_logits in zip(self.levels, logits):
            values = torch.linspace(-1.0, 1.0, int(level), device=feature.device, dtype=feature.dtype)
            probabilities = channel_logits.float().softmax(dim=1).to(dtype=feature.dtype)
            soft_parts.append((probabilities * values.view(1, -1, 1, 1)).sum(dim=1, keepdim=True))
            hard_codes.append(channel_logits.argmax(dim=1, keepdim=True))
        q_continuous = torch.cat(soft_parts, dim=1)
        codes = torch.cat(hard_codes, dim=1)
        q_hard = fsq_codes_to_q(codes, self.levels).to(dtype=q_continuous.dtype)
        q_train = q_continuous + (q_hard - q_continuous).detach()
        return ReceiverPrediction(q_continuous, q_hard, q_train, logits, codes)


class JointFSQIndexPredictor(nn.Module):
    """Predict the complete mixed-radix FSQ token at every spatial site."""

    def __init__(
        self,
        z1_channels: int,
        levels: Sequence[int],
        *,
        hidden: int,
        blocks: int,
        attention_every: int,
        heads: int,
        condition_mode: str,
    ) -> None:
        super().__init__()
        self.levels = tuple(int(level) for level in levels)
        self.vocab_size = int(math.prod(self.levels))
        self.trunk = ReceiverTrunk(
            z1_channels,
            hidden,
            blocks,
            attention_every,
            heads,
            condition_mode,
        )
        self.head = nn.Conv2d(int(hidden), self.vocab_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        indices = torch.arange(self.vocab_size, dtype=torch.long)
        multipliers: list[int] = []
        running = 1
        for level in reversed(self.levels[1:]):
            running *= int(level)
            multipliers.append(running)
        multipliers = list(reversed(multipliers)) + [1]
        codes = torch.stack(
            [(indices // int(multiplier)) % int(level) for level, multiplier in zip(self.levels, multipliers)],
            dim=1,
        )
        q_codebook = fsq_codes_to_q(codes.t().unsqueeze(0).unsqueeze(-1), self.levels)
        self.register_buffer("codebook", q_codebook.squeeze(0).squeeze(-1).t().contiguous())
        self.register_buffer("codebook_codes", codes.contiguous())

    def forward(self, condition: ReceiverCondition) -> ReceiverPrediction:
        logits = self.head(self.trunk(condition))
        probabilities = logits.float().softmax(dim=1)
        q_continuous = torch.einsum("bkhw,kc->bchw", probabilities, self.codebook.float()).to(logits.dtype)
        joint_indices = logits.argmax(dim=1)
        codes = F.embedding(joint_indices, self.codebook_codes).permute(0, 3, 1, 2).contiguous()
        q_hard = F.embedding(joint_indices, self.codebook).permute(0, 3, 1, 2).contiguous().to(logits.dtype)
        q_train = q_continuous + (q_hard - q_continuous).detach()
        return ReceiverPrediction(q_continuous, q_hard, q_train, [logits], codes, joint_indices)


class CausalMaskedConv2d(nn.Conv2d):
    """Raster-causal convolution used by the receiver-side AR generator."""

    def __init__(self, mask_type: str, *args, **kwargs) -> None:
        if str(mask_type) not in {"A", "B"}:
            raise ValueError(f"mask_type must be A or B, got {mask_type!r}")
        super().__init__(*args, **kwargs)
        if self.kernel_size[0] != self.kernel_size[1] or self.kernel_size[0] % 2 != 1:
            raise ValueError("causal masked convolution requires an odd square kernel")
        mask = torch.ones_like(self.weight)
        center = int(self.kernel_size[0]) // 2
        mask[:, :, center + 1 :, :] = 0
        mask[:, :, center, center + (1 if str(mask_type) == "B" else 0) :] = 0
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self.causal_mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CausalConditionedBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre = nn.Conv2d(int(channels), int(channels), 1)
        self.condition = nn.Conv2d(int(channels), int(channels), 1)
        self.causal = CausalMaskedConv2d(
            "B",
            int(channels),
            int(channels),
            3,
            padding=1,
        )
        self.post = nn.Conv2d(int(channels), int(channels), 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.pre(x) + self.condition(condition))
        return x + self.scale * self.post(F.silu(self.causal(hidden)))


class AutoregressiveFSQIndexPredictor(nn.Module):
    """Generate the three FSQ scalar indices in raster order.

    Training uses standard masked-convolution teacher forcing through
    ``forward_teacher``.  Deployment calls ``forward(condition)`` and greedily
    generates all 16x16 sites using only z1/x1 and already generated indices.
    The public forward signature deliberately remains receiver-only so the
    executable no-leak contract still applies.
    """

    def __init__(
        self,
        z1_channels: int,
        levels: Sequence[int],
        *,
        hidden: int,
        blocks: int,
        attention_every: int,
        heads: int,
        condition_mode: str,
    ) -> None:
        super().__init__()
        self.levels = tuple(int(level) for level in levels)
        self.vocab_size = int(math.prod(self.levels))
        self.hard_fsq = True
        self.condition_trunk = ReceiverTrunk(
            z1_channels,
            hidden,
            max(2, int(blocks) // 2),
            attention_every,
            heads,
            condition_mode,
        )
        self.token_embedding = nn.Embedding(self.vocab_size, int(hidden))
        self.input_causal = CausalMaskedConv2d(
            "A",
            int(hidden),
            int(hidden),
            5,
            padding=2,
        )
        self.condition_in = nn.Conv2d(int(hidden), int(hidden), 1)
        self.causal_blocks = nn.ModuleList(
            CausalConditionedBlock(int(hidden)) for _ in range(int(blocks))
        )
        self.output = nn.Sequential(
            nn.Conv2d(int(hidden), int(hidden), 1),
            nn.SiLU(),
        )
        self.index_heads = nn.ModuleList(
            nn.Conv2d(int(hidden), int(level), 1) for level in self.levels
        )
        for head in self.index_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        multipliers: list[int] = []
        for channel in range(len(self.levels)):
            multiplier = math.prod(self.levels[channel + 1 :])
            multipliers.append(int(multiplier))
        self.register_buffer(
            "joint_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
        )

    def _logits_from_feature(
        self,
        condition_feature: torch.Tensor,
        joint_indices: torch.Tensor,
        prior_logits: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        embedded = self.token_embedding(joint_indices.long()).permute(0, 3, 1, 2).contiguous()
        hidden = self.input_causal(embedded) + self.condition_in(condition_feature)
        for block in self.causal_blocks:
            hidden = block(hidden, condition_feature)
        hidden = self.output(hidden)
        logits = [head(hidden) for head in self.index_heads]
        if prior_logits is not None:
            logits = [value + prior for value, prior in zip(logits, prior_logits)]
        return logits

    def _condition_feature_and_prior(
        self,
        condition: ReceiverCondition,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        return self.condition_trunk(condition), None

    def _prediction(self, logits: list[torch.Tensor]) -> ReceiverPrediction:
        soft_parts: list[torch.Tensor] = []
        hard_codes: list[torch.Tensor] = []
        for level, channel_logits in zip(self.levels, logits):
            values = torch.linspace(
                -1.0,
                1.0,
                int(level),
                device=channel_logits.device,
                dtype=channel_logits.dtype,
            )
            probabilities = channel_logits.float().softmax(dim=1).to(channel_logits.dtype)
            soft_parts.append(
                (probabilities * values.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            )
            hard_codes.append(channel_logits.argmax(dim=1, keepdim=True))
        q_continuous = torch.cat(soft_parts, dim=1)
        codes = torch.cat(hard_codes, dim=1)
        q_hard = fsq_codes_to_q(codes, self.levels).to(dtype=q_continuous.dtype)
        q_train = q_continuous + (q_hard - q_continuous).detach()
        return ReceiverPrediction(q_continuous, q_hard, q_train, logits, codes)

    def forward_teacher(
        self,
        condition: ReceiverCondition,
        target_joint_indices: torch.Tensor,
    ) -> ReceiverPrediction:
        condition_feature, prior_logits = self._condition_feature_and_prior(condition)
        if tuple(target_joint_indices.shape) != (
            int(condition.z1.shape[0]),
            int(condition.z1.shape[-2]),
            int(condition.z1.shape[-1]),
        ):
            raise ValueError(
                "AR teacher indices must be [B,H,W] at the z1 resolution, got "
                f"{tuple(target_joint_indices.shape)}"
            )
        logits = self._logits_from_feature(
            condition_feature,
            target_joint_indices.detach(),
            prior_logits,
        )
        return self._prediction(logits)

    def forward(self, condition: ReceiverCondition) -> ReceiverPrediction:
        condition_feature, prior_logits = self._condition_feature_and_prior(condition)
        batch, _channels, height, width = condition_feature.shape
        generated = torch.zeros(batch, height, width, dtype=torch.long, device=condition_feature.device)
        for flat_index in range(int(height) * int(width)):
            row, column = divmod(flat_index, int(width))
            logits = self._logits_from_feature(condition_feature, generated, prior_logits)
            codes_here = torch.stack(
                [channel_logits[:, :, row, column].argmax(dim=1) for channel_logits in logits],
                dim=1,
            )
            generated[:, row, column] = (
                codes_here * self.joint_multipliers.view(1, -1)
            ).sum(dim=1)
        return self._prediction(
            self._logits_from_feature(condition_feature, generated, prior_logits)
        )


class BaseInitializedAutoregressiveFSQIndexPredictor(AutoregressiveFSQIndexPredictor):
    """AR refinement initialized by a frozen receiver-only direct-q predictor."""

    def __init__(self, *args, prior_strength: float = 8.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        z1_channels = int(kwargs["z1_channels"] if "z1_channels" in kwargs else args[0])
        hidden = int(kwargs["hidden"])
        blocks = int(kwargs["blocks"])
        attention_every = int(kwargs["attention_every"])
        heads = int(kwargs["heads"])
        condition_mode = str(kwargs["condition_mode"])
        self.base_predictor = DirectQPredictor(
            z1_channels=z1_channels,
            out_channels=len(self.levels),
            levels=self.levels,
            hidden=hidden,
            blocks=blocks,
            attention_every=attention_every,
            heads=heads,
            condition_mode=condition_mode,
            hard_fsq=False,
        )
        self.base_predictor.requires_grad_(False)
        self.base_q_proj = nn.Conv2d(len(self.levels), hidden, 1)
        self.prior_strength = float(prior_strength)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_predictor.eval()
        return self

    def _condition_feature_and_prior(
        self,
        condition: ReceiverCondition,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        with torch.no_grad():
            base = self.base_predictor(condition)
        feature = self.condition_trunk(condition) + self.base_q_proj(base.q_continuous.detach())
        priors = [
            F.one_hot(base.codes[:, channel].long(), num_classes=int(level))
            .permute(0, 3, 1, 2)
            .to(dtype=feature.dtype)
            * self.prior_strength
            for channel, level in enumerate(self.levels)
        ]
        return feature, priors


class MultiScaleX1Encoder(nn.Module):
    """Encode full-resolution x1 with learned downsampling instead of interpolation."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        widths = [64, 96, 128, 192, int(hidden)]
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1),
            nn.GroupNorm(_groups(widths[0]), widths[0]),
            nn.SiLU(),
        )
        stages: list[nn.Module] = []
        for in_channels, out_channels in zip(widths[:-1], widths[1:]):
            stages.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                    nn.GroupNorm(_groups(out_channels), out_channels),
                    nn.SiLU(),
                    GatedResidualBlock(out_channels),
                )
            )
        self.stages = nn.Sequential(*stages)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x1))


class QResidualPredictor(nn.Module):
    """Learn a bounded receiver-only correction around a frozen q predictor."""

    def __init__(
        self,
        base_predictor: nn.Module,
        *,
        z1_channels: int,
        out_channels: int,
        levels: Sequence[int],
        hidden: int = 256,
        blocks: int = 12,
        attention_every: int = 3,
        heads: int = 8,
        residual_scale: float = 0.25,
        hard_fsq: bool = False,
    ) -> None:
        super().__init__()
        self.base_predictor = base_predictor
        self.base_predictor.requires_grad_(False)
        self.base_predictor.eval()
        self.levels = tuple(int(level) for level in levels)
        self.hard_fsq = bool(hard_fsq)
        self.residual_scale = float(residual_scale)
        self.x1_encoder = MultiScaleX1Encoder(int(hidden))
        self.z1_encoder = nn.Sequential(
            nn.Conv2d(int(z1_channels), int(hidden), 3, padding=1),
            nn.GroupNorm(_groups(hidden), int(hidden)),
            nn.SiLU(),
            GatedResidualBlock(int(hidden)),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(int(hidden) * 4, int(hidden), 1),
            nn.SiLU(),
            nn.Conv2d(int(hidden), int(hidden), 3, padding=1),
        )
        body: list[nn.Module] = []
        for index in range(int(blocks)):
            body.append(GatedResidualBlock(int(hidden)))
            if int(attention_every) > 0 and (index + 1) % int(attention_every) == 0:
                body.append(SpatialAttentionBlock(int(hidden), int(heads)))
        self.body = nn.Sequential(*body)
        self.out_norm = nn.GroupNorm(_groups(hidden), int(hidden))
        self.head = nn.Sequential(
            nn.Conv2d(int(hidden), int(hidden), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(int(hidden), int(out_channels), 3, padding=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_predictor.eval()
        return self

    def residual_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("base_predictor."):
                yield parameter

    def forward(self, condition: ReceiverCondition) -> ReceiverPrediction:
        condition.validate()
        with torch.no_grad():
            base_prediction = self.base_predictor(condition)
            q_base = base_prediction.q_continuous.detach()
        x_feature = self.x1_encoder(condition.x1)
        z_feature = self.z1_encoder(condition.z1)
        if tuple(x_feature.shape[-2:]) != tuple(z_feature.shape[-2:]):
            raise ValueError(
                f"residual x1/z1 feature mismatch: {tuple(x_feature.shape)} vs {tuple(z_feature.shape)}"
            )
        feature = self.fuse(
            torch.cat(
                [z_feature, x_feature, z_feature - x_feature, z_feature * x_feature],
                dim=1,
            )
        )
        feature = F.silu(self.out_norm(self.body(feature)))
        q_residual = float(self.residual_scale) * torch.tanh(self.head(feature))
        q_continuous = (q_base + q_residual).clamp(-1.0, 1.0)
        q_hard, codes = snap_to_fsq(q_continuous, self.levels)
        q_train = q_continuous + (q_hard - q_continuous).detach() if self.hard_fsq else q_continuous
        return ReceiverPrediction(
            q_continuous,
            q_hard,
            q_train,
            [],
            codes,
            q_base=q_base,
            q_residual=q_residual,
        )


def build_receiver_predictor(
    route: str,
    *,
    z1_channels: int,
    levels: Sequence[int],
    hidden: int,
    blocks: int,
    attention_every: int,
    heads: int,
    condition_mode: str,
    hard_fsq: bool,
) -> nn.Module:
    common = dict(
        z1_channels=int(z1_channels),
        hidden=int(hidden),
        blocks=int(blocks),
        attention_every=int(attention_every),
        heads=int(heads),
        condition_mode=str(condition_mode),
    )
    if str(route) == "direct_q":
        return DirectQPredictor(
            out_channels=len(levels),
            levels=levels,
            hard_fsq=bool(hard_fsq),
            **common,
        )
    if str(route) == "parallel_index":
        return ParallelFSQIndexPredictor(levels=levels, **common)
    if str(route) == "joint_index":
        return JointFSQIndexPredictor(levels=levels, **common)
    if str(route) == "ar_index":
        return AutoregressiveFSQIndexPredictor(levels=levels, **common)
    if str(route) == "ar_residual_index":
        return BaseInitializedAutoregressiveFSQIndexPredictor(levels=levels, **common)
    raise ValueError(f"unknown receiver prediction route {route!r}")
