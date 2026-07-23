#!/usr/bin/env python3
"""Qwen3-initialised z1-conditioned Channel AutoRegressive receiver.

This is the receiver-side CAR analogue of arXiv:2605.26089v2 for JSCC.  The
generator has the deliberately narrow deployment contract

    q2_hat_indices = CAR_Qwen3(z1)

and never receives x1, img, z2, q2, or oracle indices.  Each CVQ index is
represented by its frozen 16x16 codebook map (256 scalars), exactly the channel
embedding dimensionality used by the paper.  The 16 z1 channel maps form a
causal condition prefix; x1 is used only *after* generation by the required
D2+combiner reconstruction path.

Stage I follows the paper: Qwen3 backbone frozen, train the two-layer 256->H
projector and K-way channel head.  ``--unfreeze-backbone`` exposes Stage II for
the 4B model when memory permits; do not use it for the first feasibility run.

The optional predictor-reconstruction calibration keeps the deployment contract
unchanged.  Its hard branch has an exact argmax/codebook forward pass and uses
only a straight-through posterior for gradients; its continuous branch is the
posterior mean over the *same* frozen CVQ codebook.  Both are disabled by
default so the paper-faithful CE Stage-I baseline remains reproducible.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import AutoModel

from train_cvq_car import (
    Means, ReceiverDecode, indices_to_q, load_sender, logits_to_soft_q, psnr,
    sender_targets,
)


class Qwen3Z1CAR(nn.Module):
    """Pretrained decoder-only Qwen3 with a z1-map prefix and CVQ-map tokens."""

    def __init__(self, model_name: str, codebook: torch.Tensor, z1_channels: int, channels: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        if codebook.ndim != 3:
            raise ValueError(f"CVQ codebook must be [K,H,W], got {tuple(codebook.shape)}")
        self.vocabulary, height, width = map(int, codebook.shape)
        self.embedding_dim = height * width
        self.z1_channels = int(z1_channels)
        self.channels = int(channels)
        self.backbone = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
        )
        hidden = int(self.backbone.config.hidden_size)
        # Paper Stage-I projector: 256-dimensional channel map -> LLM hidden.
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2), nn.GELU(),
            nn.Linear(self.embedding_dim * 2, hidden),
        )
        self.z1_type = nn.Parameter(torch.zeros(1, 1, hidden))
        self.q_type = nn.Parameter(torch.zeros(1, 1, hidden))
        self.bos_map = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        # This is the CAR visual-token head, not Qwen's text LM head.
        self.head = nn.Linear(hidden, self.vocabulary, bias=False)
        self.register_buffer("codebook_maps", codebook.detach().flatten(1).contiguous(), persistent=True)
        nn.init.normal_(self.z1_type, std=0.02)
        nn.init.normal_(self.q_type, std=0.02)
        nn.init.normal_(self.bos_map, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def set_stage(self, *, unfreeze_backbone: bool) -> None:
        self.backbone.requires_grad_(bool(unfreeze_backbone))

    def _z1_prefix(self, z1: torch.Tensor) -> torch.Tensor:
        if z1.ndim != 4 or int(z1.shape[1]) != self.z1_channels:
            raise ValueError(f"z1 must be [B,{self.z1_channels},H,W], got {tuple(z1.shape)}")
        maps = z1.flatten(2)
        if int(maps.shape[-1]) != self.embedding_dim:
            raise ValueError(f"z1 maps need {self.embedding_dim} elements, got {tuple(maps.shape)}")
        # JSCC sender tensors remain FP32; the Qwen Stage-I projector and
        # frozen backbone use BF16.  This is a representation boundary, not
        # a change to z1 conditioning or the source model.
        maps = maps.to(dtype=self.projector[0].weight.dtype)
        return self.projector(maps) + self.z1_type

    def _map_embeddings(self, maps: torch.Tensor) -> torch.Tensor:
        """Project arbitrary channel maps; used by both hard and continuous CAR."""
        if maps.ndim != 3 or int(maps.shape[-1]) != self.embedding_dim:
            raise ValueError(f"channel maps must be [B,L,{self.embedding_dim}], got {tuple(maps.shape)}")
        return self.projector(maps.to(dtype=self.projector[0].weight.dtype)) + self.q_type

    def _token_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.vocabulary):
            raise ValueError("CVQ index out of range")
        maps = self.codebook_maps[indices.long()].to(dtype=self.projector[0].weight.dtype)
        return self._map_embeddings(maps)

    def _bos_embedding(self, batch: int) -> torch.Tensor:
        return self.projector(self.bos_map.expand(batch, -1, -1)) + self.q_type

    def forward_teacher(self, z1: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Parallel teacher-forced next-channel logits; targets are not inputs at deployment."""
        if indices.ndim != 2:
            raise ValueError(f"indices must be [B,C], got {tuple(indices.shape)}")
        prefix = self._z1_prefix(z1)
        prior = self._bos_embedding(int(indices.shape[0]))
        if int(indices.shape[1]) > 1:
            prior = torch.cat([prior, self._token_embeddings(indices[:, :-1])], dim=1)
        sequence = torch.cat([prefix, prior], dim=1)
        states = self.backbone(inputs_embeds=sequence, use_cache=False, return_dict=True).last_hidden_state
        return self.head(states[:, int(prefix.shape[1]):])

    def forward_scheduled_suffix(
        self, z1: torch.Tensor, indices: torch.Tensor, *, probability: float, steps: int,
        backprop_projector: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Head-only scheduled sampling on actual rollout states.

        Parallel teacher forcing is retained for the projector alignment.  This
        short suffix pass cures the otherwise severe train/inference mismatch:
        it predicts a random channel span after a history containing a mixture
        of its own indices and oracle indices.  The frozen Qwen backbone is
        normally evaluated under ``no_grad``: the visual head still receives
        CE gradients for the states it will encounter at deployment.  The
        optional projector-backprop path instead retains the frozen Qwen
        activations so that the map projector, z1 prefix, BOS, and token-type
        embeddings can align to the actual rollout-state distribution.  This
        is the principled Stage-I correction for teacher-forcing exposure
        bias; Qwen weights remain frozen in either mode.
        """
        if indices.ndim != 2 or int(indices.shape[1]) < 1:
            raise ValueError("scheduled indices must be nonempty [B,C]")
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("scheduled sampling probability must lie in [0,1]")
        span = min(max(1, int(steps)), int(indices.shape[1]))
        start = int(torch.randint(int(indices.shape[1]) - span + 1, (), device=indices.device).item())
        with torch.set_grad_enabled(bool(backprop_projector)):
            prefix = self._z1_prefix(z1)
            prior = self._bos_embedding(int(indices.shape[0]))
            if start:
                prior = torch.cat([prior, self._token_embeddings(indices[:, :start])], dim=1)
            output = self.backbone(
                inputs_embeds=torch.cat([prefix, prior], dim=1), use_cache=True, return_dict=True
            )
            cache = output.past_key_values
            state = output.last_hidden_state[:, -1]
            states: list[torch.Tensor] = []
            for offset in range(span):
                states.append(state)
                prediction = self.head(state).argmax(dim=-1)
                oracle = indices[:, start + offset]
                if float(probability) == 0.0:
                    next_index = oracle
                elif float(probability) == 1.0:
                    next_index = prediction
                else:
                    use_prediction = torch.rand(int(indices.shape[0]), device=indices.device) < float(probability)
                    next_index = torch.where(use_prediction, prediction, oracle)
                if offset + 1 < span:
                    output = self.backbone(
                        inputs_embeds=self._token_embeddings(next_index[:, None]),
                        past_key_values=cache, use_cache=True, return_dict=True,
                    )
                    cache = output.past_key_values
                    state = output.last_hidden_state[:, -1]
        return self.head(torch.stack(states, dim=1)), indices[:, start:start + span]

    def forward_scheduled_suffix_continuous(
        self, z1: torch.Tensor, indices: torch.Tensor, *, probability: float, steps: int,
        temperature: float, backprop_projector: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scheduled sampling on continuous posterior-mean feedback states.

        The deployment path of a continuous receiver feeds a posterior mean
        ``sum_j p(j) e_j`` back to the next CAR step.  Training a CE head only
        on discrete oracle-history states therefore leaves a real exposure
        gap.  This method uses the same causal map feedback for a sampled
        suffix, while keeping the true prefix only as a bounded-cost context.
        It is deliberately separate from the hard-index version above: a
        continuous receiver must never silently revert to argmax feedback.

        With ``backprop_projector=False`` this is a head-only DAgger-style
        correction: the frozen Qwen backbone supplies rollout states under
        ``no_grad`` and CE updates the visual head.  Enabling it propagates
        through the frozen Qwen computation into the visual projector and the
        earlier continuous maps, but never changes Qwen backbone weights.
        """
        if indices.ndim != 2 or int(indices.shape[1]) < 1:
            raise ValueError("scheduled indices must be nonempty [B,C]")
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("scheduled sampling probability must lie in [0,1]")
        if float(temperature) <= 0.0:
            raise ValueError("scheduled continuous temperature must be positive")
        span = min(max(1, int(steps)), int(indices.shape[1]))
        start = int(torch.randint(int(indices.shape[1]) - span + 1, (), device=indices.device).item())
        codebook = self.codebook_maps.float()
        with torch.set_grad_enabled(bool(backprop_projector)):
            prefix = self._z1_prefix(z1)
            prior = self._bos_embedding(int(indices.shape[0]))
            if start:
                # Before the sampled rollout suffix, retain only legal
                # tokenizer targets as a training context.  At deployment the
                # full prefix is replaced by generated maps via rollout_continuous.
                prior = torch.cat([prior, self._token_embeddings(indices[:, :start])], dim=1)
            output = self.backbone(
                inputs_embeds=torch.cat([prefix, prior], dim=1), use_cache=True, return_dict=True
            )
            cache = output.past_key_values
            state = output.last_hidden_state[:, -1]
            states: list[torch.Tensor] = []
            for offset in range(span):
                states.append(state)
                logits = self.head(state)
                predicted_map = torch.matmul(
                    F.softmax(logits.float() / float(temperature), dim=-1), codebook
                )
                oracle_map = codebook[indices[:, start + offset].long()]
                if float(probability) == 0.0:
                    next_map = oracle_map
                elif float(probability) == 1.0:
                    next_map = predicted_map
                else:
                    use_prediction = torch.rand(int(indices.shape[0]), device=indices.device) < float(probability)
                    next_map = torch.where(use_prediction[:, None], predicted_map, oracle_map)
                if offset + 1 < span:
                    output = self.backbone(
                        inputs_embeds=self._map_embeddings(next_map[:, None]),
                        past_key_values=cache, use_cache=True, return_dict=True,
                    )
                    cache = output.past_key_values
                    state = output.last_hidden_state[:, -1]
        return self.head(torch.stack(states, dim=1)), indices[:, start:start + span]

    @torch.no_grad()
    def forward(self, z1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Strict autoregressive BOS->C rollout.  Deployment input is z1 only."""
        return self.rollout(z1, self.channels)

    @torch.no_grad()
    def rollout(self, z1: torch.Tensor, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = self._z1_prefix(z1)
        output = self.backbone(
            inputs_embeds=torch.cat([prefix, self._bos_embedding(int(z1.shape[0]))], dim=1),
            use_cache=True, return_dict=True,
        )
        cache = output.past_key_values
        state = output.last_hidden_state[:, -1]
        logits_list: list[torch.Tensor] = []
        indices_list: list[torch.Tensor] = []
        for _ in range(int(channels)):
            logits = self.head(state)
            indices = logits.argmax(dim=-1)
            logits_list.append(logits)
            indices_list.append(indices)
            output = self.backbone(
                inputs_embeds=self._token_embeddings(indices[:, None]),
                past_key_values=cache, use_cache=True, return_dict=True,
            )
            cache = output.past_key_values
            state = output.last_hidden_state[:, -1]
        return torch.stack(logits_list, dim=1), torch.stack(indices_list, dim=1)

    @torch.no_grad()
    def rollout_continuous(
        self, z1: torch.Tensor, channels: int, *, temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Continuous CAR rollout: posterior-mean code maps are fed back causally.

        This is the legal continuous-q2_hat counterpart of ``rollout``.  It
        never converts a generated map back to a sender index before the next
        channel; doing so would reintroduce a hard-history mismatch into the
        continuous deployment route.
        """
        if float(temperature) <= 0.0:
            raise ValueError("continuous rollout temperature must be positive")
        prefix = self._z1_prefix(z1)
        output = self.backbone(
            inputs_embeds=torch.cat([prefix, self._bos_embedding(int(z1.shape[0]))], dim=1),
            use_cache=True, return_dict=True,
        )
        cache = output.past_key_values
        state = output.last_hidden_state[:, -1]
        logits_list: list[torch.Tensor] = []
        maps_list: list[torch.Tensor] = []
        indices_list: list[torch.Tensor] = []
        # Decode-side continuous q_hat must use the exact FP32 CVQ entries.
        # The projector below explicitly casts its input to Qwen's BF16
        # boundary, so keeping this buffer FP32 does not alter the backbone.
        codebook = self.codebook_maps.float()
        for _ in range(int(channels)):
            logits = self.head(state)
            probabilities = F.softmax(logits.float() / float(temperature), dim=-1)
            maps = torch.matmul(probabilities, codebook)
            logits_list.append(logits)
            maps_list.append(maps)
            indices_list.append(logits.argmax(dim=-1))
            output = self.backbone(
                inputs_embeds=self._map_embeddings(maps[:, None]),
                past_key_values=cache, use_cache=True, return_dict=True,
            )
            cache = output.past_key_values
            state = output.last_hidden_state[:, -1]
        maps_out = torch.stack(maps_list, dim=1)
        return torch.stack(logits_list, dim=1), maps_out, torch.stack(indices_list, dim=1)


class _HardCodebookPosteriorST(torch.autograd.Function):
    """Exact codebook lookup forward, posterior-mean Jacobian backward."""

    @staticmethod
    def forward(ctx, probabilities: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(codebook)
        return codebook[probabilities.argmax(dim=-1)]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (codebook,) = ctx.saved_tensors
        # d sum_j p_j e_j / d p_j = e_j.  The codebook is deliberately
        # stop-gradient in predictor calibration, so no second return exists.
        grad_probabilities = torch.einsum("bcd,kd->bck", grad_output, codebook)
        return grad_probabilities, None


def logits_to_st_hard_q(sender, logits: torch.Tensor, rate: int, temperature: float) -> torch.Tensor:
    """Exact hard-CVQ forward values with posterior-mean straight-through gradients.

    ``argmax`` and the codebook lookup are the actual forward deployment path;
    only their backward surrogate is soft.  This must not be substituted by a
    posterior mean when measuring or optimizing the hard receiver route.
    """
    if float(temperature) <= 0.0:
        raise ValueError("--hard-st-temperature must be positive")
    codebook = sender.codec.quantizer.codebook_at_k(int(rate)).detach().reshape(int(rate), -1)
    probabilities = F.softmax(logits / float(temperature), dim=-1)
    flat = _HardCodebookPosteriorST.apply(probabilities, codebook)
    h, w = sender.codec.quantizer.embedding_shape
    return flat.view(int(logits.shape[0]), int(logits.shape[1]), int(h), int(w))


def qwen_generator_audit(model: Qwen3Z1CAR) -> None:
    # The parameter/buffer graph is self-contained: no sender or receiver
    # module may be reachable from the CAR generator.
    forbidden = ("x1", "img", "z2", "oracle", "receiver", "sender")
    names = " ".join(name.lower() for name, _ in model.named_modules())
    if any(word in names for word in forbidden):
        raise AssertionError("Qwen CAR generator contains a forbidden receiver/source module")


def run_epoch(loader, sender, car, receiver, optimizer, args, device, *, train: bool) -> dict[str, float]:
    car.train(train)
    # Stage-I backbone stays in eval mode (dropout-free pretrained prior).
    if not bool(args.unfreeze_backbone):
        car.backbone.eval()
    receiver.eval() if bool(args.freeze_receiver_decoder) else receiver.train(train)
    meters = Means(); audited = False
    maximum = int(args.max_train_batches if train else args.max_val_batches)
    accum_steps = max(1, int(args.accum_steps))
    pending_backward = False
    if train:
        optimizer.zero_grad(set_to_none=True)

    def optimizer_step() -> None:
        nonlocal pending_backward
        if not pending_backward:
            return
        trainable = [parameter for parameter in list(car.parameters()) + list(receiver.parameters()) if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        pending_backward = False

    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if maximum and batch_index > maximum:
            break
        imgs = imgs.to(device, non_blocking=True)
        target = sender_targets(sender, imgs, int(args.rate))
        z1 = target["condition"].z1  # x1 does not enter the generator.
        if not audited:
            qwen_generator_audit(car)
            if car.z1_channels != int(z1.shape[1]):
                raise AssertionError("generator z1 channel contract changed")
            audited = True
        with torch.set_grad_enabled(train):
            teacher = car.forward_teacher(z1, target["indices"])
            loss_ce = F.cross_entropy(teacher.flatten(0, 1).float(), target["indices"].flatten(), label_smoothing=float(args.label_smoothing))
            loss_scheduled = loss_ce.new_zeros(())
            if train and float(args.lambda_scheduled_ce) > 0.0 and int(args.scheduled_sampling_steps) > 0 and batch_index % int(args.scheduled_sampling_every) == 0:
                if args.scheduled_history_mode == "continuous":
                    scheduled_logits, scheduled_targets = car.forward_scheduled_suffix_continuous(
                        z1, target["indices"], probability=float(args.scheduled_sampling_probability),
                        steps=int(args.scheduled_sampling_steps),
                        temperature=float(args.rollout_temperature),
                        backprop_projector=bool(args.scheduled_backprop_projector),
                    )
                else:
                    scheduled_logits, scheduled_targets = car.forward_scheduled_suffix(
                        z1, target["indices"], probability=float(args.scheduled_sampling_probability),
                        steps=int(args.scheduled_sampling_steps),
                        backprop_projector=bool(args.scheduled_backprop_projector),
                    )
                loss_scheduled = F.cross_entropy(scheduled_logits.flatten(0, 1).float(), scheduled_targets.flatten())
            hard_final = mean_final = mean_hard_history_final = teacher_hard_final = teacher_mean_final = hard_st_final = None
            if train:
                q_decode = logits_to_soft_q(sender, teacher.float(), int(args.rate), float(args.soft_temperature))
                generated = teacher.argmax(dim=-1)
                final = receiver(q_decode, z1, target["x1"])
                # Optional hard predictor calibration.  Its forward q is
                # exactly a CVQ codebook lookup at argmax indices, while its
                # gradient is the posterior-mean straight-through surrogate.
                if float(args.lambda_hard_st_recon) > 0.0:
                    q_hard_st = logits_to_st_hard_q(
                        sender, teacher.float(), int(args.rate), float(args.hard_st_temperature)
                    )
                    hard_st_final = receiver(q_hard_st, z1, target["x1"])
            else:
                channels = int(target["indices"].shape[1])
                # The hard route always uses hard generated history.
                rollout_logits, hard_generated = car.rollout(z1, channels)
                q_hard = indices_to_q(sender, hard_generated, int(args.rate))
                q_mean_hard_history = logits_to_soft_q(
                    sender, rollout_logits.float(), int(args.rate), float(args.rollout_temperature)
                )
                hard_final = receiver(q_hard, z1, target["x1"])
                mean_hard_history_final = receiver(q_mean_hard_history, z1, target["x1"])
                if args.rollout_q_mode == "mean":
                    # A continuous-q2_hat deployment must also feed its
                    # continuous posterior mean to the next channel.  Feeding
                    # argmax indices here would measure a different hybrid
                    # route and causes an avoidable exposure mismatch.
                    _continuous_logits, continuous_maps, generated = car.rollout_continuous(
                        z1, channels, temperature=float(args.rollout_temperature)
                    )
                    h, w = sender.codec.quantizer.embedding_shape
                    q_mean = continuous_maps.view(int(imgs.shape[0]), channels, int(h), int(w))
                    mean_final = receiver(q_mean, z1, target["x1"])
                else:
                    generated = hard_generated
                    q_mean = q_mean_hard_history
                    mean_final = mean_hard_history_final
                # Diagnostic only: oracle previous indices are supplied to
                # the teacher-forced logits.  This is not a legal receiver
                # output, but separates conditional-token quality from the
                # exposure error of the real BOS->C rollout.
                teacher_hard_final = receiver(indices_to_q(sender, teacher.argmax(dim=-1), int(args.rate)), z1, target["x1"])
                # The continuous counterpart is equally diagnostic: it is the
                # posterior codebook mean under *true* previous channels.  It
                # remains an oracle-history probe, never a deployment input.
                teacher_mean_final = receiver(
                    logits_to_soft_q(sender, teacher.float(), int(args.rate), float(args.rollout_temperature)),
                    z1,
                    target["x1"],
                )
                q_decode = q_hard if args.rollout_q_mode == "hard" else q_mean
                final = hard_final if args.rollout_q_mode == "hard" else mean_final
            loss_recon = F.mse_loss(final.float(), imgs.float())
            loss_hard_st_recon = (
                F.mse_loss(hard_st_final.float(), imgs.float()) if hard_st_final is not None else loss_recon.new_zeros(())
            )
            # This is deliberately separate from the legacy --lambda-recon
            # term.  A calibration run sets the legacy term to zero and uses
            # this explicit posterior-mean weight instead.
            loss_continuous_code_recon = (
                loss_recon if train and float(args.lambda_continuous_code_recon) > 0.0 else loss_recon.new_zeros(())
            )
            loss = (
                float(args.lambda_ce) * loss_ce
                + float(args.lambda_recon) * loss_recon
                + float(args.lambda_hard_st_recon) * loss_hard_st_recon
                + float(args.lambda_continuous_code_recon) * loss_continuous_code_recon
                + float(args.lambda_scheduled_ce) * loss_scheduled
            )
            if train:
                # Preserve the physical loss for logs while accumulating an
                # effective Stage-II batch that is large enough for a stable
                # full-Qwen update on one GPU.
                (loss / float(accum_steps)).backward()
                pending_backward = True
                if batch_index % accum_steps == 0:
                    optimizer_step()
        batch = int(imgs.shape[0])
        x1_psnr = psnr(target["x1"], imgs); oracle_psnr = psnr(target["oracle"], imgs); pred_psnr = psnr(final, imgs)
        meters.add("loss", loss, batch); meters.add("loss_ce", loss_ce, batch); meters.add("loss_scheduled", loss_scheduled, batch); meters.add("loss_recon", loss_recon, batch)
        meters.add("loss_hard_st_recon", loss_hard_st_recon, batch); meters.add("loss_continuous_code_recon", loss_continuous_code_recon, batch)
        meters.add("psnr_x1", x1_psnr.mean(), batch); meters.add("psnr_oracle", oracle_psnr.mean(), batch); meters.add("psnr_x2_hat", pred_psnr.mean(), batch)
        meters.add("delta_x1_hat", (pred_psnr - x1_psnr).mean(), batch); meters.add("delta_x1_oracle", (oracle_psnr - x1_psnr).mean(), batch)
        meters.add("teacher_index_accuracy", (teacher.argmax(-1) == target["indices"]).float().mean(), batch)
        meters.add("index_accuracy", (generated == target["indices"]).float().mean(), batch)
        if not train:
            hard_psnr = psnr(hard_final, imgs); mean_psnr = psnr(mean_final, imgs)
            meters.add("psnr_x2_hat_hard", hard_psnr.mean(), batch); meters.add("delta_x1_hat_hard", (hard_psnr - x1_psnr).mean(), batch)
            meters.add("psnr_x2_hat_mean", mean_psnr.mean(), batch); meters.add("delta_x1_hat_mean", (mean_psnr - x1_psnr).mean(), batch)
            mean_hard_history_psnr = psnr(mean_hard_history_final, imgs)
            meters.add("psnr_x2_hat_mean_hard_history", mean_hard_history_psnr.mean(), batch)
            meters.add("delta_x1_hat_mean_hard_history", (mean_hard_history_psnr - x1_psnr).mean(), batch)
            teacher_hard_psnr = psnr(teacher_hard_final, imgs)
            meters.add("psnr_x2_teacher_history_hard", teacher_hard_psnr.mean(), batch)
            meters.add("delta_x1_teacher_history_hard", (teacher_hard_psnr - x1_psnr).mean(), batch)
            teacher_mean_psnr = psnr(teacher_mean_final, imgs)
            meters.add("psnr_x2_teacher_history_mean", teacher_mean_psnr.mean(), batch)
            meters.add("delta_x1_teacher_history_mean", (teacher_mean_psnr - x1_psnr).mean(), batch)
            zero = receiver(torch.zeros_like(q_decode), z1, target["x1"])
            shuffled = receiver(sender.codec.quantizer.shuffle_tokens(q_decode), z1, target["x1"])
            meters.add("drop_zero", (pred_psnr - psnr(zero, imgs)).mean(), batch)
            meters.add("drop_shuffle", (pred_psnr - psnr(shuffled, imgs)).mean(), batch)
            if batch > 1:
                if args.rollout_q_mode == "mean":
                    _wrong_logits, wrong_maps, _wrong_indices = car.rollout_continuous(
                        z1.roll(1, 0), channels, temperature=float(args.rollout_temperature)
                    )
                    h, w = sender.codec.quantizer.embedding_shape
                    wrong_q = wrong_maps.view(int(imgs.shape[0]), channels, int(h), int(w))
                else:
                    _wrong_logits, wrong_indices = car.rollout(z1.roll(1, 0), channels)
                    wrong_q = indices_to_q(sender, wrong_indices, int(args.rate))
                wrong = receiver(wrong_q, z1, target["x1"])
                meters.add("condition_shuffle_drop", (pred_psnr - psnr(wrong, imgs)).mean(), batch)
    if train:
        # DIV2K does not guarantee an epoch length divisible by the requested
        # effective batch.  Flush the final partial group rather than silently
        # discarding its gradients.
        optimizer_step()
    result = meters.result(); result["receiver_only_audit"] = float(audited)
    if not train:
        result["goal_met"] = float(
            result["delta_x1_hat"] >= args.min_delta
            and result["drop_zero"] >= args.min_ablation_drop
            and result["drop_shuffle"] >= args.min_ablation_drop
            and result.get("condition_shuffle_drop", float("-inf")) >= args.min_condition_ablation_drop
        )
    return result


def save(path: Path, sender_path: str, car: Qwen3Z1CAR, receiver, optimizer, args, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not duplicate the 4B public backbone in every experiment checkpoint.
    trainable = {name: tensor.detach().cpu() for name, tensor in car.state_dict().items() if not name.startswith("backbone.")}
    torch.save({"stage": "explore4_qwen3_z1_car", "paper": "arXiv:2605.26089v2 CAR Stage-I/II", "epoch": epoch,
                "sender_checkpoint": sender_path, "qwen_model": args.qwen_model, "args": vars(args), "metrics": metrics,
                "car_trainable_state_dict": trainable, "receiver_state_dict": receiver.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "receiver_contract": {"generator_inputs": ["z1"], "reconstruction_inputs": ["z1", "x1"],
                                      "forbidden_generator_inputs": ["img", "x1", "z2", "q2", "oracle_indices"],
                                      "output": "hard_or_posterior_mean_CVQ_q2_hat"}}, path)
    print(f"saved checkpoint: {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sender-checkpoint", required=True); parser.add_argument("--rate", type=int, default=4096)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-4B"); parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="checkpoint frozen-Qwen activations during Stage II; enables a practical full-Qwen batch on one GPU")
    parser.add_argument("--lambda-ce", type=float, default=1.0); parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-hard-st-recon", type=float, default=0.0, help="opt-in teacher-history hard-CVQ reconstruction weight; forward is argmax+same codebook")
    parser.add_argument("--lambda-continuous-code-recon", type=float, default=0.0, help="opt-in teacher-history posterior-mean reconstruction weight over the same codebook")
    parser.add_argument("--hard-st-temperature", type=float, default=1.0, help="softmax temperature used only by the hard straight-through backward surrogate")
    parser.add_argument("--lambda-scheduled-ce", type=float, default=0.0, help="weight of rollout-state CE; 0 keeps pure teacher forcing")
    parser.add_argument("--scheduled-sampling-probability", type=float, default=0.5, help="probability of feeding a generated history token/map in scheduled suffixes")
    parser.add_argument("--scheduled-sampling-steps", type=int, default=0, help="rollout-state suffix length; 0 disables scheduled sampling")
    parser.add_argument("--scheduled-sampling-every", type=int, default=8, help="apply scheduled suffix CE every N training batches")
    parser.add_argument("--scheduled-history-mode", choices=["hard", "continuous"], default="hard", help="generated history used by scheduled suffix CE; continuous is required for posterior-mean q2_hat deployment")
    parser.add_argument("--scheduled-backprop-projector", action="store_true", help="backprop scheduled-rollout CE through frozen Qwen into the visual projector/prefix; uses more VRAM")
    parser.add_argument("--soft-temperature", type=float, default=1.0); parser.add_argument("--rollout-q-mode", choices=["hard", "mean"], default="hard")
    parser.add_argument("--rollout-temperature", type=float, default=0.2); parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--freeze-receiver-decoder", action="store_true"); parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--decoder-lr", type=float, default=2e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9); parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--accum-steps", type=int, default=1, help="gradient-accumulation steps; physical batch stays within GPU memory while Stage-II uses a stable effective batch")
    parser.add_argument("--epochs", type=int, default=40); parser.add_argument("--val-every", type=int, default=5); parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--test-batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8); parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--save-dir", default="MY-V2/jscc-f/explore-4/checkpoints-car"); parser.add_argument("--log-json", default=""); parser.add_argument("--version", default="qwen3-z1-car-v1")
    parser.add_argument("--resume", default="", help="Qwen CAR checkpoint: restores non-backbone trainables, then reloads the public backbone")
    parser.add_argument("--reset-optimizer", action="store_true", help="do not restore AdamW state on resume; required when changing frozen Stage I to full-Qwen Stage II")
    parser.add_argument("--no-checkpoint", action="store_true", help="metrics-only run: never serialize a model or optimizer; useful for a bounded Stage-II memory/learning probe when no checkpoint quota is available")
    parser.add_argument("--eval-only", action="store_true", help="run strict receiver-only validation from --resume without taking a training step")
    parser.add_argument("--min-delta", type=float, default=0.5); parser.add_argument("--min-ablation-drop", type=float, default=0.1)
    parser.add_argument("--min-condition-ablation-drop", type=float, default=0.1, help="required PSNR drop when z1 is shuffled across the batch; proves the CAR actually uses z1")
    parser.add_argument("--max-train-batches", type=int, default=0); parser.add_argument("--max-val-batches", type=int, default=0); parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= float(args.scheduled_sampling_probability) <= 1.0:
        raise ValueError("--scheduled-sampling-probability must lie in [0,1]")
    if int(args.scheduled_sampling_steps) < 0 or int(args.scheduled_sampling_every) < 1:
        raise ValueError("--scheduled-sampling-steps must be >=0 and --scheduled-sampling-every >=1")
    if float(args.hard_st_temperature) <= 0.0:
        raise ValueError("--hard-st-temperature must be positive")
    if float(args.lambda_hard_st_recon) < 0.0 or float(args.lambda_continuous_code_recon) < 0.0:
        raise ValueError("predictor reconstruction weights must be non-negative")
    if not 0.0 <= float(args.adam_beta1) < 1.0 or not 0.0 <= float(args.adam_beta2) < 1.0:
        raise ValueError("AdamW beta values must lie in [0,1)")
    if bool(args.gradient_checkpointing) and not bool(args.unfreeze_backbone):
        raise ValueError("--gradient-checkpointing is only meaningful with --unfreeze-backbone Stage II")
    if int(args.accum_steps) < 1:
        raise ValueError("--accum-steps must be >=1")
    continuous_training = bool(
        args.rollout_q_mode == "mean"
        or float(args.lambda_continuous_code_recon) > 0.0
        or (args.scheduled_history_mode == "continuous" and float(args.lambda_scheduled_ce) > 0.0)
    )
    if continuous_training and abs(float(args.soft_temperature) - float(args.rollout_temperature)) > 1e-8:
        raise ValueError(
            "continuous q2_hat training/deployment must share one temperature: "
            "set --soft-temperature equal to --rollout-temperature"
        )
    return args


def train(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    sender, sender_args, _payload, train_loader, val_loader, device = load_sender(Path(args.sender_checkpoint), device, args)
    if int(args.rate) not in [int(v) for v in sender_args.rates_list]:
        raise ValueError(f"rate {args.rate} was not trained by sender")
    with torch.no_grad():
        first = next(iter(train_loader))[0][:1].to(device); z1_channels = int(sender.source.layer1(first)["z1"].shape[1])
    codebook = sender.codec.quantizer.codebook_at_k(int(args.rate)).detach().cpu()
    car = Qwen3Z1CAR(args.qwen_model, codebook, z1_channels, int(sender_args.latent_c), dtype=torch.bfloat16).to(device=device, dtype=torch.bfloat16)
    # ``Module.to(dtype=bf16)`` also converts buffers.  Keep the frozen CVQ
    # codebook itself FP32 so continuous posterior means are formed on the
    # same values used by receiver-side hard lookup; visual inputs are cast at
    # the projector boundary inside ``_map_embeddings``.
    car.codebook_maps = codebook.to(device=device, dtype=torch.float32).flatten(1).contiguous()
    car.set_stage(unfreeze_backbone=bool(args.unfreeze_backbone))
    if bool(args.gradient_checkpointing):
        car.backbone.gradient_checkpointing_enable()
    receiver = ReceiverDecode(sender).to(device)
    if args.freeze_receiver_decoder:
        receiver.requires_grad_(False).eval()
    trainable = [p for p in car.parameters() if p.requires_grad] + ([] if args.freeze_receiver_decoder else list(receiver.parameters()))
    optimizer = optim.AdamW(
        trainable, lr=float(args.lr), weight_decay=float(args.weight_decay),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
    )
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if payload.get("stage") != "explore4_qwen3_z1_car":
            raise ValueError(f"not a Qwen3 CAR checkpoint: {resume_path}")
        if str(payload.get("qwen_model")) != str(args.qwen_model):
            raise ValueError("--resume Qwen model differs from --qwen-model")
        saved_sender = payload.get("sender_checkpoint")
        if not saved_sender:
            raise ValueError("--resume checkpoint has no sender checkpoint identity")
        if Path(str(saved_sender)).expanduser().resolve(strict=False) != Path(args.sender_checkpoint).expanduser().resolve(strict=False):
            raise ValueError("--resume sender checkpoint differs from --sender-checkpoint")
        state = payload.get("car_trainable_state_dict")
        if not isinstance(state, dict):
            raise ValueError("resume checkpoint has no CAR trainable state")
        missing, unexpected = car.load_state_dict(state, strict=False)
        # All missing keys must be the deliberately omitted pretrained Qwen
        # backbone; anything else signals a broken receiver contract.
        if any(not key.startswith("backbone.") for key in missing) or unexpected:
            raise RuntimeError(f"incompatible CAR resume state missing={missing} unexpected={unexpected}")
        # Checkpoints made before the FP32-buffer refinement may contain a
        # BF16 copy.  The sender checkpoint is the immutable codebook source
        # of truth for every resumed receiver experiment.
        car.codebook_maps = codebook.to(device=device, dtype=torch.float32).flatten(1).contiguous()
        if not args.freeze_receiver_decoder:
            receiver.load_state_dict(payload["receiver_state_dict"], strict=True)
        if bool(args.reset_optimizer):
            print("resume requested with a reset optimizer", flush=True)
        else:
            try:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            except ValueError as error:
                raise ValueError(
                    "resume optimizer parameter groups differ; pass --reset-optimizer when changing stages"
                ) from error
        start_epoch = int(payload.get("epoch", 0))
        print(f"resumed Stage-I trainables from {resume_path} at epoch={start_epoch}", flush=True)
    print("=== explore-4 | Qwen3 z1-conditioned CAR ===", flush=True)
    print("实验设计", flush=True); print(f"  Qwen={args.qwen_model}; stage={'II unfreeze backbone' if args.unfreeze_backbone else 'I frozen backbone'}; C={sender_args.latent_c}, K={args.rate}; generator=z1 only", flush=True)
    print("loss设计", flush=True); print(f"  AdamW(lr={args.lr}, beta=({args.adam_beta1},{args.adam_beta2}), wd={args.weight_decay}, physical/effective-batch={args.batch_size}/{args.batch_size * args.accum_steps}); {args.lambda_ce}*teacher CE + {args.lambda_scheduled_ce}*rollout-state CE (history={args.scheduled_history_mode}, p={args.scheduled_sampling_probability}, steps={args.scheduled_sampling_steps}, every={args.scheduled_sampling_every}, projector_grad={args.scheduled_backprop_projector}) + {args.lambda_recon}*legacy-soft-code MSE + {args.lambda_hard_st_recon}*hard-ST-codebook MSE(T={args.hard_st_temperature}) + {args.lambda_continuous_code_recon}*posterior-mean-codebook MSE; valid=real cached BOS->C rollout", flush=True)
    print("模块选择", flush=True); print(f"  z1 16x16 maps are condition-prefix tokens; frozen CVQ maps are 256D CAR tokens -> 2-layer projector -> Qwen3; x1 only enters D2+combiner after q2_hat; train=random/valid=center; checkpoint={'disabled (metrics-only)' if args.no_checkpoint else 'enabled'}", flush=True)
    root = Path(args.save_dir) / args.version; best = float("-inf"); history = []
    if bool(args.eval_only):
        if not args.resume:
            raise ValueError("--eval-only requires --resume")
        val_metrics = run_epoch(val_loader, sender, car, receiver, None, args, device, train=False)
        print(f"[Qwen CAR eval {start_epoch:03d}] {val_metrics}", flush=True)
        if args.log_json:
            out = Path(args.log_json); out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps([{"epoch": start_epoch, "eval": val_metrics}], indent=2), encoding="utf-8")
        return
    if start_epoch >= int(args.epochs):
        raise ValueError(f"--epochs={args.epochs} must exceed resumed epoch {start_epoch}")
    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        started = time.time(); train_metrics = run_epoch(train_loader, sender, car, receiver, optimizer, args, device, train=True)
        print(f"[Qwen CAR train {epoch:03d}/{args.epochs}] {train_metrics} time={time.time()-started:.1f}s", flush=True)
        item = {"epoch": epoch, "train": train_metrics}
        if epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            val_metrics = run_epoch(val_loader, sender, car, receiver, None, args, device, train=False)
            print(f"[Qwen CAR val {epoch:03d}] {val_metrics}", flush=True); item["val"] = val_metrics
            # Keep the exact state associated with the latest strict
            # validation, even if its real rollout PSNR is temporarily below
            # an earlier best.  This is needed to diagnose teacher-history
            # versus exposure error without silently discarding a checkpoint.
            if bool(args.no_checkpoint):
                print("checkpoint skipped by --no-checkpoint", flush=True)
            else:
                save(root / "latest.pth", args.sender_checkpoint, car, receiver, optimizer, args, epoch, val_metrics)
                if val_metrics["psnr_x2_hat"] > best:
                    best = val_metrics["psnr_x2_hat"]; save(root / "best.pth", args.sender_checkpoint, car, receiver, optimizer, args, epoch, val_metrics)
        history.append(item)
        if args.log_json:
            out = Path(args.log_json); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train(parse_args())
