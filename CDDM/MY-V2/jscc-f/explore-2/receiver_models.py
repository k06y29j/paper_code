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
    loss_flow: torch.Tensor | None = None
    loss_flow_mse: torch.Tensor | None = None
    loss_flow_cosine: torch.Tensor | None = None


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


class FSQCategoricalPosteriorDenoiser(ImageVQConditionalDenoiser):
    """Predict a categorical posterior over the exact FSQ Cartesian grid."""

    def __init__(
        self,
        z1_channels: int,
        embedding_dim: int,
        vocab_size: int,
        *,
        hidden: int,
        blocks: int,
        attention_every: int,
        heads: int,
        condition_mode: str,
    ) -> None:
        super().__init__(
            int(z1_channels),
            int(embedding_dim),
            hidden=int(hidden),
            blocks=int(blocks),
            attention_every=int(attention_every),
            heads=int(heads),
            condition_mode=str(condition_mode),
        )
        self.vocab_size = int(vocab_size)
        # The inherited network consumes a noisy q tensor with embedding_dim
        # channels.  CDCD changes only its output parameterization: one logit
        # per complete Cartesian-grid token at every spatial site.
        self.output = nn.Conv2d(self.hidden, self.vocab_size, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


class FSQCategoricalPosteriorDiffusionGenerator(nn.Module):
    """Receiver-only categorical-posterior diffusion on the K=125 FSQ grid.

    This is a minimal local adaptation of arXiv:2606.09962.  Training samples
    ``t ~ U[0,1]`` and

    ``q_t = alpha(t) * q_target + sigma(t) * epsilon``

    before predicting 125 token logits with cross entropy.  The posterior
    mean over the fixed ``[-1,-.5,0,.5,1]^3`` grid is the continuous q2_hat;
    posterior argmax is the hard q2_hat.  Deployment performs a deterministic
    DDIM-style reverse rollout and accepts only :class:`ReceiverCondition`.

    The paper does not publish its exact alpha/sigma schedule.  ``cosine_vp``
    (alpha=cos(pi*t/2), sigma=sin(pi*t/2)) is therefore an explicit local
    assumption, not claimed as a reproduction detail.
    """

    teacher_target = "q2"

    def __init__(
        self,
        z1_channels: int,
        levels: Sequence[int],
        *,
        height: int = 16,
        width: int = 16,
        hidden: int = 192,
        blocks: int = 8,
        attention_every: int = 2,
        heads: int = 4,
        condition_mode: str = "z1_x1",
        sample_steps: int = 12,
        sample_seed: int = 20260713,
        time_scale: float = 1000.0,
        prior_scale: float = 1.0,
        schedule: str = "cosine_vp",
        hard_fsq: bool = False,
    ) -> None:
        super().__init__()
        self.levels = tuple(int(level) for level in levels)
        if self.levels != (5, 5, 5):
            raise ValueError(
                "categorical-posterior diffusion is deliberately fixed to "
                f"K=125 levels=(5,5,5), got {self.levels}"
            )
        self.embedding_dim = 3
        self.vocab_size = 125
        self.height = int(height)
        self.width = int(width)
        self.sample_steps = int(sample_steps)
        self.sample_seed = int(sample_seed)
        self.time_scale = float(time_scale)
        self.prior_scale = float(prior_scale)
        self.schedule = str(schedule)
        self.hard_fsq = bool(hard_fsq)
        if self.sample_steps < 2:
            raise ValueError(
                f"categorical diffusion requires at least two DDIM evaluations, got {self.sample_steps}"
            )
        if self.time_scale <= 0.0:
            raise ValueError(f"CDCD time_scale must be positive, got {self.time_scale}")
        if self.prior_scale <= 0.0:
            raise ValueError(f"CDCD prior_scale must be positive, got {self.prior_scale}")
        if self.schedule != "cosine_vp":
            raise ValueError(
                "only the explicit local CDCD schedule assumption cosine_vp "
                f"is implemented, got {self.schedule!r}"
            )

        multipliers = torch.tensor((25, 5, 1), dtype=torch.long)
        indices = torch.arange(self.vocab_size, dtype=torch.long)
        codes = torch.stack(
            [
                (indices // int(multiplier)) % int(level)
                for level, multiplier in zip(self.levels, multipliers.tolist())
            ],
            dim=1,
        )
        q_codebook = fsq_codes_to_q(
            codes.t().unsqueeze(0).unsqueeze(-1), self.levels
        )
        self.register_buffer("joint_multipliers", multipliers)
        self.register_buffer("codebook_codes", codes.contiguous())
        self.register_buffer(
            "codebook", q_codebook.squeeze(0).squeeze(-1).t().contiguous()
        )
        self.denoiser = FSQCategoricalPosteriorDenoiser(
            int(z1_channels),
            self.embedding_dim,
            self.vocab_size,
            hidden=int(hidden),
            blocks=int(blocks),
            attention_every=int(attention_every),
            heads=int(heads),
            condition_mode=str(condition_mode),
        )

    def _checked_q(self, q: torch.Tensor) -> torch.Tensor:
        expected = (self.embedding_dim, self.height, self.width)
        if tuple(int(value) for value in q.shape[1:]) != expected:
            raise ValueError(
                f"CDCD target/state must be [B,{self.embedding_dim},{self.height},{self.width}], "
                f"got {tuple(q.shape)}"
            )
        return q

    def _alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Variance-preserving cosine schedule assumed by this adaptation."""

        t = t.clamp(0.0, 1.0)
        angle = t * (math.pi * 0.5)
        alpha = torch.cos(angle).clamp_min(0.0)
        sigma = torch.sin(angle).clamp_min(0.0)
        return alpha, sigma

    def _target_indices(self, q_target: torch.Tensor) -> torch.Tensor:
        q_hard, codes = snap_to_fsq(q_target, self.levels)
        if not torch.allclose(q_target.float(), q_hard.float(), rtol=0.0, atol=1e-6):
            raise ValueError("CDCD supervision must already lie on the exact FSQ grid")
        return (
            codes.long()
            * self.joint_multipliers.view(1, self.embedding_dim, 1, 1)
        ).sum(dim=1)

    def _prediction_from_logits(self, logits: torch.Tensor) -> ReceiverPrediction:
        expected = (self.vocab_size, self.height, self.width)
        if tuple(int(value) for value in logits.shape[1:]) != expected:
            raise ValueError(
                f"CDCD logits must be [B,{self.vocab_size},{self.height},{self.width}], "
                f"got {tuple(logits.shape)}"
            )
        probabilities = logits.float().softmax(dim=1)
        q_continuous = torch.einsum(
            "bkhw,kc->bchw", probabilities, self.codebook.float()
        ).to(logits.dtype)
        joint_indices = logits.argmax(dim=1)
        codes = (
            F.embedding(joint_indices, self.codebook_codes)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        q_hard = (
            F.embedding(joint_indices, self.codebook)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(logits.dtype)
        )
        q_train = (
            q_continuous + (q_hard - q_continuous).detach()
            if self.hard_fsq
            else q_continuous
        )
        return ReceiverPrediction(
            q_continuous,
            q_hard,
            q_train,
            [logits],
            codes,
            joint_indices,
        )

    def forward_teacher(
        self,
        condition: ReceiverCondition,
        q_target: torch.Tensor,
    ) -> ReceiverPrediction:
        """Single noisy teacher sample used only for the CDCD CE objective."""

        condition.validate()
        q_target = self._checked_q(q_target)
        # Validate that the CE target is an exact member of the frozen grid.
        self._target_indices(q_target)
        batch = int(q_target.shape[0])
        t = torch.rand(batch, device=q_target.device, dtype=q_target.dtype)
        alpha, sigma = self._alpha_sigma(t)
        shape = (batch, 1, 1, 1)
        q_t = (
            alpha.view(shape) * q_target
            + sigma.view(shape) * torch.randn_like(q_target)
        )
        logits = self.denoiser(q_t, t * self.time_scale, condition)
        return self._prediction_from_logits(logits)

    def _initial_noise(self, condition: ReceiverCondition) -> torch.Tensor:
        condition.validate()
        generator = torch.Generator(device=condition.z1.device)
        generator.manual_seed(self.sample_seed)
        return self.prior_scale * torch.randn(
            int(condition.z1.shape[0]),
            self.embedding_dim,
            self.height,
            self.width,
            generator=generator,
            device=condition.z1.device,
            dtype=condition.z1.dtype,
        )

    def forward(self, condition: ReceiverCondition) -> ReceiverPrediction:
        """Complete receiver-only deterministic DDIM rollout."""

        value = self._initial_noise(condition)
        batch = int(value.shape[0])
        schedule = torch.linspace(
            1.0,
            0.0,
            self.sample_steps,
            device=value.device,
            dtype=value.dtype,
        )
        final_prediction: ReceiverPrediction | None = None
        for step, t_scalar in enumerate(schedule):
            t = torch.full(
                (batch,),
                float(t_scalar),
                device=value.device,
                dtype=value.dtype,
            )
            logits = self.denoiser(value, t * self.time_scale, condition)
            final_prediction = self._prediction_from_logits(logits)
            if step + 1 == int(schedule.numel()):
                break
            alpha, sigma = self._alpha_sigma(t)
            next_t = torch.full_like(t, float(schedule[step + 1]))
            next_alpha, next_sigma = self._alpha_sigma(next_t)
            view = (batch, 1, 1, 1)
            # DDIM reuses the noise implied by the categorical posterior mean.
            epsilon = (
                value - alpha.view(view) * final_prediction.q_continuous
            ) / sigma.view(view).clamp_min(1e-6)
            value = (
                next_alpha.view(view) * final_prediction.q_continuous
                + next_sigma.view(view) * epsilon
            )
        if final_prediction is None:
            raise AssertionError("CDCD DDIM schedule produced no model evaluation")
        return final_prediction


class FSQConditionalFlowMatchingGenerator(nn.Module):
    """Conditional rectified-flow generator for receiver-side FSQ q2.

    The velocity-supervision path implements Eq. (9) from the iFSQ appendix:

    ``q_t = t*q + (1-t)*eps`` and ``v_target = q-eps``.

    Optionally, ``eps`` is replaced by a frozen receiver-only Direct-Q anchor
    ``q_base(z1,x1)``.  This gives a residual flow
    ``q_t=(1-t)*q_base+t*q`` with velocity target ``q-q_base``; both training
    rollout and deployment start at that same base.  Because the velocity
    head is zero-initialized, a newly constructed anchored flow reproduces the
    base predictor exactly before its first update.

    The q/index/reconstruction training path is deliberately separate: it
    integrates the learned velocity from the configured receiver-side initial
    state with a short differentiable Euler rollout.  Deployment uses the same
    integrator with ``sample_steps`` (normally more steps), then snaps the
    result to the exact FSQ Cartesian grid.  Thus training metrics no longer
    measure a target-assisted one-step endpoint while validation measures an
    autonomous rollout.  Its public ``forward`` receives only
    :class:`ReceiverCondition`; the q2 target exists solely in
    ``forward_teacher`` for velocity supervision.
    """

    teacher_target = "q2"

    def __init__(
        self,
        z1_channels: int,
        levels: Sequence[int],
        *,
        height: int = 16,
        width: int = 16,
        hidden: int = 192,
        blocks: int = 8,
        attention_every: int = 2,
        heads: int = 4,
        condition_mode: str = "z1_x1",
        sample_steps: int = 32,
        train_sample_steps: int = 4,
        sample_noise: str = "gaussian",
        sample_seed: int = 20260713,
        time_scale: float = 1000.0,
        timestep_sampling: str = "uniform",
        cosine_loss_weight: float = 0.0,
        hard_fsq: bool = True,
        base_predictor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.levels = tuple(int(level) for level in levels)
        self.embedding_dim = len(self.levels)
        self.height = int(height)
        self.width = int(width)
        self.sample_steps = int(sample_steps)
        self.train_sample_steps = int(train_sample_steps)
        self.sample_noise = str(sample_noise)
        self.sample_seed = int(sample_seed)
        self.time_scale = float(time_scale)
        self.timestep_sampling = str(timestep_sampling)
        self.cosine_loss_weight = float(cosine_loss_weight)
        self.hard_fsq = bool(hard_fsq)
        if self.sample_steps < 1:
            raise ValueError(f"flow sample_steps must be positive, got {self.sample_steps}")
        if self.train_sample_steps < 1:
            raise ValueError(
                "flow train_sample_steps must be positive, "
                f"got {self.train_sample_steps}"
            )
        if self.sample_noise not in {"gaussian", "zero"}:
            raise ValueError(
                f"flow sample_noise must be gaussian or zero, got {self.sample_noise!r}"
            )
        if self.time_scale <= 0.0:
            raise ValueError(f"flow time_scale must be positive, got {self.time_scale}")
        if self.timestep_sampling not in {"uniform", "logit_normal"}:
            raise ValueError(
                "flow timestep_sampling must be uniform or logit_normal, got "
                f"{self.timestep_sampling!r}"
            )
        if self.cosine_loss_weight < 0.0:
            raise ValueError(
                "flow cosine_loss_weight must be non-negative, got "
                f"{self.cosine_loss_weight}"
            )
        self.denoiser = ImageVQConditionalDenoiser(
            int(z1_channels),
            self.embedding_dim,
            hidden=int(hidden),
            blocks=int(blocks),
            attention_every=int(attention_every),
            heads=int(heads),
            condition_mode=str(condition_mode),
        )
        # Register the optional module under a stable name so checkpoints and
        # optimizer filtering can identify it unambiguously.  It may also be
        # attached after loading an older non-anchored flow checkpoint.
        self.base_predictor: nn.Module | None = None
        if base_predictor is not None:
            self.attach_base_predictor(base_predictor)
        multipliers: list[int] = []
        for channel in range(self.embedding_dim):
            multipliers.append(int(math.prod(self.levels[channel + 1 :])))
        self.register_buffer(
            "joint_multipliers", torch.tensor(multipliers, dtype=torch.long)
        )

    @property
    def uses_base_anchor(self) -> bool:
        """Whether this flow starts from a frozen receiver-only direct-q base."""

        return self.base_predictor is not None

    def attach_base_predictor(
        self,
        base_predictor: nn.Module,
        *,
        replace: bool = False,
    ) -> "FSQConditionalFlowMatchingGenerator":
        """Attach and freeze a receiver-only :class:`DirectQPredictor` base.

        Post-construction attachment lets callers first restore an older flow
        checkpoint whose state dict predates the optional ``base_predictor``
        subtree, then load the direct-q checkpoint through the explicit helper
        below.  The base is never optimized and remains in evaluation mode even
        when the residual flow itself is trained.
        """

        if not isinstance(base_predictor, DirectQPredictor):
            raise TypeError(
                "base-anchored FSQ flow requires DirectQPredictor, got "
                f"{type(base_predictor).__name__}"
            )
        if self.base_predictor is not None and not bool(replace):
            raise RuntimeError("flow base_predictor is already attached")
        if tuple(int(level) for level in base_predictor.levels) != self.levels:
            raise ValueError(
                "flow/base FSQ levels differ: "
                f"{self.levels} != {tuple(base_predictor.levels)}"
            )
        self.base_predictor = base_predictor
        self.base_predictor.requires_grad_(False)
        self.base_predictor.eval()
        return self

    def load_base_predictor_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ):
        """Load a direct-q predictor state into the attached frozen base."""

        if self.base_predictor is None:
            raise RuntimeError("attach a DirectQPredictor before loading its state")
        result = self.base_predictor.load_state_dict(state_dict, strict=bool(strict))
        self.base_predictor.requires_grad_(False)
        self.base_predictor.eval()
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        if self.base_predictor is not None:
            self.base_predictor.eval()
        return self

    def _checked_target(self, q_target: torch.Tensor) -> torch.Tensor:
        expected = (self.embedding_dim, self.height, self.width)
        if tuple(int(v) for v in q_target.shape[1:]) != expected:
            raise ValueError(
                f"flow target must be [B,{self.embedding_dim},{self.height},{self.width}], "
                f"got {tuple(q_target.shape)}"
            )
        return q_target

    def _prediction(
        self,
        q_continuous: torch.Tensor,
        *,
        loss_flow: torch.Tensor | None = None,
        loss_flow_mse: torch.Tensor | None = None,
        loss_flow_cosine: torch.Tensor | None = None,
        q_base: torch.Tensor | None = None,
    ) -> ReceiverPrediction:
        q_continuous = q_continuous.clamp(-1.0, 1.0)
        q_hard, codes = snap_to_fsq(q_continuous, self.levels)
        q_train = (
            q_continuous + (q_hard - q_continuous).detach()
            if self.hard_fsq
            else q_continuous
        )
        joint_indices = (
            codes.long()
            * self.joint_multipliers.view(1, self.embedding_dim, 1, 1)
        ).sum(dim=1)
        return ReceiverPrediction(
            q_continuous,
            q_hard,
            q_train,
            [],
            codes,
            joint_indices,
            q_base=q_base,
            loss_flow=loss_flow,
            loss_flow_mse=loss_flow_mse,
            loss_flow_cosine=loss_flow_cosine,
        )

    def _sample_training_time(self, q_target: torch.Tensor) -> torch.Tensor:
        """Sample flow time using the selected, checkpointed training contract.

        The released iFSQ transport configuration uses logit-normal time
        sampling.  Uniform remains the default so existing checkpoints and
        launchers retain their original behavior.
        """

        batch = int(q_target.shape[0])
        if self.timestep_sampling == "logit_normal":
            return torch.randn(
                batch, device=q_target.device, dtype=q_target.dtype
            ).sigmoid()
        return torch.rand(batch, device=q_target.device, dtype=q_target.dtype)

    def forward_teacher(
        self,
        condition: ReceiverCondition,
        q_target: torch.Tensor,
    ) -> ReceiverPrediction:
        condition.validate()
        q_target = self._checked_target(q_target)
        batch = int(q_target.shape[0])
        t = self._sample_training_time(q_target)
        t_map = t.view(batch, 1, 1, 1)
        endpoint_start = self._initial_state(condition)
        # A base-anchored residual flow uses the exact same receiver-only
        # q_base for velocity supervision and autonomous integration:
        # q_t=(1-t)q_base+t*q_target, v*=q_target-q_base.  Without a learned
        # base, retain the configured zero/Gaussian flow behavior.
        if self.base_predictor is not None:
            flow_start = endpoint_start
        elif self.sample_noise == "zero":
            flow_start = torch.zeros_like(q_target)
        else:
            flow_start = torch.randn_like(q_target)
        q_t = (1.0 - t_map) * flow_start + t_map * q_target
        target_velocity = q_target - flow_start
        predicted_velocity = self.denoiser(
            q_t,
            t * self.time_scale,
            condition,
        )
        loss_flow_mse = F.mse_loss(
            predicted_velocity.float(), target_velocity.float()
        )
        loss_flow_cosine = (
            1.0
            - F.cosine_similarity(
                predicted_velocity.float(), target_velocity.float(), dim=1
            )
        ).mean()
        loss_flow = (
            loss_flow_mse
            + self.cosine_loss_weight * loss_flow_cosine
        )
        # q/index/final losses and their logged metrics must be measured on an
        # autonomous receiver rollout.  q_target is intentionally absent from
        # this branch, so it has exactly the public-forward information set.
        predicted_q = self._integrate_euler(
            condition,
            steps=self.train_sample_steps,
            initial_value=endpoint_start,
        )
        return self._prediction(
            predicted_q,
            loss_flow=loss_flow,
            loss_flow_mse=loss_flow_mse,
            loss_flow_cosine=loss_flow_cosine,
            q_base=endpoint_start if self.base_predictor is not None else None,
        )

    def _initial_noise(self, condition: ReceiverCondition) -> torch.Tensor:
        shape = (
            int(condition.z1.shape[0]),
            self.embedding_dim,
            self.height,
            self.width,
        )
        if self.sample_noise == "zero":
            return torch.zeros(
                shape, device=condition.z1.device, dtype=condition.z1.dtype
            )
        generator = torch.Generator(device=condition.z1.device)
        generator.manual_seed(self.sample_seed)
        return torch.randn(
            shape,
            generator=generator,
            device=condition.z1.device,
            dtype=condition.z1.dtype,
        )

    def _initial_state(self, condition: ReceiverCondition) -> torch.Tensor:
        condition.validate()
        if self.base_predictor is None:
            return self._initial_noise(condition)
        # This is a frozen receiver-only function of z1/x1.  Detaching here is
        # both the optimizer boundary and what keeps multi-step residual-flow
        # activation memory independent of the base network depth.
        self.base_predictor.eval()
        with torch.no_grad():
            base_prediction = self.base_predictor(condition)
            q_base = base_prediction.q_continuous.detach()
        return self._checked_target(q_base)

    def _integrate_euler(
        self,
        condition: ReceiverCondition,
        *,
        steps: int,
        initial_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition.validate()
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"flow Euler steps must be positive, got {steps}")
        value = (
            self._initial_state(condition)
            if initial_value is None
            else self._checked_target(initial_value)
        )
        batch = int(value.shape[0])
        step_size = 1.0 / float(steps)
        for step in range(steps):
            t = torch.full(
                (batch,),
                float(step) * step_size,
                device=value.device,
                dtype=value.dtype,
            )
            velocity = self.denoiser(value, t * self.time_scale, condition)
            value = value + step_size * velocity
        return value

    def forward(self, condition: ReceiverCondition) -> ReceiverPrediction:
        initial_value = self._initial_state(condition)
        value = self._integrate_euler(
            condition,
            steps=self.sample_steps,
            initial_value=initial_value,
        )
        return self._prediction(
            value,
            q_base=initial_value if self.base_predictor is not None else None,
        )


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


class AutoregressiveFSQJointTokenPredictor(AutoregressiveFSQIndexPredictor):
    """Raster-AR generator over the paper's mixed-radix FSQ token.

    For ``levels=[5,5,5]`` each spatial site is one token in ``[0,125)``.
    This differs intentionally from :class:`AutoregressiveFSQIndexPredictor`,
    whose history is joint but whose output factorizes into three five-way
    heads.  The single K-way likelihood here is the tokenization used by the
    iFSQ paper's autoregressive objective.

    ``forward_teacher`` is supervision-only and remains strictly causal;
    deployment ``forward(condition)`` consumes only receiver-visible z1/x1
    and previously generated joint tokens.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Keep the legacy factorized AR route load-compatible and add this as
        # a separate opt-in model instead of changing its checkpoint schema.
        del self.index_heads
        hidden = int(self.condition_in.in_channels)
        self.joint_head = nn.Conv2d(hidden, self.vocab_size, 1)
        nn.init.zeros_(self.joint_head.weight)
        nn.init.zeros_(self.joint_head.bias)

        indices = torch.arange(self.vocab_size, dtype=torch.long)
        codes = torch.stack(
            [
                (indices // int(multiplier)) % int(level)
                for level, multiplier in zip(
                    self.levels, self.joint_multipliers.detach().cpu().tolist()
                )
            ],
            dim=1,
        )
        q_codebook = fsq_codes_to_q(
            codes.t().unsqueeze(0).unsqueeze(-1), self.levels
        )
        self.register_buffer(
            "codebook", q_codebook.squeeze(0).squeeze(-1).t().contiguous()
        )
        self.register_buffer("codebook_codes", codes.contiguous())

    def _joint_logits_from_feature(
        self,
        condition_feature: torch.Tensor,
        joint_indices: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.token_embedding(joint_indices.long()).permute(0, 3, 1, 2).contiguous()
        hidden = self.input_causal(embedded) + self.condition_in(condition_feature)
        for block in self.causal_blocks:
            hidden = block(hidden, condition_feature)
        return self.joint_head(self.output(hidden))

    def _joint_prediction(self, logits: torch.Tensor) -> ReceiverPrediction:
        probabilities = logits.float().softmax(dim=1)
        q_continuous = torch.einsum(
            "bkhw,kc->bchw", probabilities, self.codebook.float()
        ).to(logits.dtype)
        joint_indices = logits.argmax(dim=1)
        codes = F.embedding(joint_indices, self.codebook_codes).permute(0, 3, 1, 2).contiguous()
        q_hard = (
            F.embedding(joint_indices, self.codebook)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(logits.dtype)
        )
        q_train = q_continuous + (q_hard - q_continuous).detach()
        return ReceiverPrediction(
            q_continuous,
            q_hard,
            q_train,
            [logits],
            codes,
            joint_indices,
        )

    def forward_teacher(
        self,
        condition: ReceiverCondition,
        target_joint_indices: torch.Tensor,
    ) -> ReceiverPrediction:
        condition_feature = self.condition_trunk(condition)
        expected = (
            int(condition.z1.shape[0]),
            int(condition.z1.shape[-2]),
            int(condition.z1.shape[-1]),
        )
        if tuple(target_joint_indices.shape) != expected:
            raise ValueError(
                "joint-token AR teacher indices must be [B,H,W] at the z1 "
                f"resolution, expected {expected}, got {tuple(target_joint_indices.shape)}"
            )
        target = target_joint_indices.detach().long()
        if int(target.numel()) and (
            int(target.min()) < 0 or int(target.max()) >= self.vocab_size
        ):
            raise ValueError(
                f"joint-token AR targets must lie in [0,{self.vocab_size})"
            )
        return self._joint_prediction(
            self._joint_logits_from_feature(condition_feature, target)
        )

    @torch.no_grad()
    def _greedy_rollout_from_feature(
        self,
        condition_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Return a genuinely free-running raster history.

        Each token is sampled greedily from a prefix containing only tokens
        generated by earlier iterations.  In particular, this is not the
        parallel argmax of a teacher-forced pass, which would still condition
        every location on the oracle prefix and therefore underestimate the
        train/deploy exposure gap.
        """

        batch, _channels, height, width = condition_feature.shape
        generated = torch.zeros(
            batch,
            height,
            width,
            dtype=torch.long,
            device=condition_feature.device,
        )
        for flat_index in range(int(height) * int(width)):
            row, column = divmod(flat_index, int(width))
            logits = self._joint_logits_from_feature(condition_feature, generated)
            generated[:, row, column] = logits[:, :, row, column].argmax(dim=1)
        return generated

    @torch.no_grad()
    def greedy_rollout_indices(
        self,
        condition: ReceiverCondition,
        *,
        batch_limit: int = 0,
    ) -> torch.Tensor:
        """Generate receiver-only histories for scheduled sampling.

        ``batch_limit=0`` rolls out the complete batch.  A positive value
        rolls out only the first ``min(batch_limit, B)`` examples, allowing a
        training loop to pay the 16x16 raster-generation cost on a small
        sub-batch and/or only every few optimizer steps.  The returned tensor
        is therefore ``[B',H,W]`` with ``B' <= B``.  Only ``z1`` and ``x1`` are
        consumed; sender images, q2, z2, and oracle indices are neither
        accepted nor retained.
        """

        condition.validate()
        batch = int(condition.z1.shape[0])
        requested = int(batch_limit)
        if requested < 0:
            raise ValueError(f"batch_limit must be non-negative, got {requested}")
        selected = batch if requested == 0 else min(requested, batch)
        if selected < 1:
            raise ValueError("cannot roll out an empty receiver batch")
        if selected != batch:
            condition = ReceiverCondition(
                z1=condition.z1[:selected],
                x1=condition.x1[:selected],
            ).validate()
        condition_feature = self.condition_trunk(condition)
        return self._greedy_rollout_from_feature(condition_feature)

    def forward(self, condition: ReceiverCondition) -> ReceiverPrediction:
        condition_feature = self.condition_trunk(condition)
        generated = self._greedy_rollout_from_feature(condition_feature)
        return self._joint_prediction(
            self._joint_logits_from_feature(condition_feature, generated)
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
    height: int = 16,
    width: int = 16,
    flow_sample_steps: int = 32,
    flow_train_sample_steps: int = 4,
    flow_sample_noise: str = "gaussian",
    flow_sample_seed: int = 20260713,
    flow_time_scale: float = 1000.0,
    flow_timestep_sampling: str = "uniform",
    flow_cosine_loss_weight: float = 0.0,
    flow_base_predictor: nn.Module | None = None,
    cdcd_sample_steps: int = 12,
    cdcd_sample_seed: int = 20260713,
    cdcd_time_scale: float = 1000.0,
    cdcd_prior_scale: float = 1.0,
    cdcd_schedule: str = "cosine_vp",
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
    if str(route) == "ar_joint_index":
        return AutoregressiveFSQJointTokenPredictor(levels=levels, **common)
    if str(route) == "ar_residual_index":
        return BaseInitializedAutoregressiveFSQIndexPredictor(levels=levels, **common)
    if str(route) == "flow_matching":
        return FSQConditionalFlowMatchingGenerator(
            levels=levels,
            height=int(height),
            width=int(width),
            sample_steps=int(flow_sample_steps),
            train_sample_steps=int(flow_train_sample_steps),
            sample_noise=str(flow_sample_noise),
            sample_seed=int(flow_sample_seed),
            time_scale=float(flow_time_scale),
            timestep_sampling=str(flow_timestep_sampling),
            cosine_loss_weight=float(flow_cosine_loss_weight),
            hard_fsq=bool(hard_fsq),
            base_predictor=flow_base_predictor,
            **common,
        )
    if str(route) == "categorical_diffusion":
        return FSQCategoricalPosteriorDiffusionGenerator(
            levels=levels,
            height=int(height),
            width=int(width),
            sample_steps=int(cdcd_sample_steps),
            sample_seed=int(cdcd_sample_seed),
            time_scale=float(cdcd_time_scale),
            prior_scale=float(cdcd_prior_scale),
            schedule=str(cdcd_schedule),
            hard_fsq=bool(hard_fsq),
            **common,
        )
    raise ValueError(f"unknown receiver prediction route {route!r}")
