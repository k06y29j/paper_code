#!/usr/bin/env python3
"""Executable receiver-only contract tests for ``explore-2``.

The default run executes the lightweight tests and, when the saved K=125
receiver checkpoint is present, closes the real checkpoint graph as well.  A
lightweight-only run is available with ``--real-k125 skip``.

The deployment graph tested here is intentionally narrower than the training
graph::

    ReceiverCondition(z1, x1) -> predictor -> q2_hat
        -> tokenizer.decode -> u2_hat -> combiner -> x2_hat

In particular, neither ``img`` nor the Layer2 sender encoder (E2/e3) is part of
that graph.
"""

from __future__ import annotations

import argparse
import inspect
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator, Sequence

import torch
import torch.nn as nn
from torchvision import transforms

from contracts import (
    ReceiverCondition,
    assert_div2k_crop_protocol,
    assert_receiver_only_module,
    assert_training_targets_are_not_inputs,
    make_receiver_condition,
)
from receiver_models import (
    ChannelVQAutoregressiveIndexPredictor,
    ImageVQConditionalDiffusionGenerator,
    JointFSQIndexPredictor,
    ReceiverPrediction,
    build_receiver_predictor,
    fsq_codes_to_q,
    snap_to_fsq,
)
import train_channel_vq_ar as channel_ar_train
import train_fsq_receiver as receiver_train
from train_continuous_q_receiver import ContinuousQGenerator
from train_continuous_q_mix_receiver import FixedTrainSelectedQMix
from vq_modules import ChannelVQ


THIS_DIR = Path(__file__).resolve().parent
CDDM_ROOT = THIS_DIR.parents[2]
DEFAULT_K125_RECEIVER = (
    "MY-V2/jscc-f/explore-2/checkpoints-receiver/"
    "cnn-fsq-k125-joint-index-v1/"
    "fsq_receiver_joint_index_z1_x1_continuous_cnn_d3_l5x5x5_best.pth"
)


def _mixed_radix_multipliers(levels: Sequence[int]) -> torch.Tensor:
    multipliers = [math.prod(int(level) for level in levels[index + 1 :]) for index in range(len(levels))]
    return torch.tensor(multipliers, dtype=torch.long)


def _codes_to_indices(codes: torch.Tensor, levels: Sequence[int]) -> torch.Tensor:
    if codes.ndim != 2 or int(codes.shape[1]) != len(levels):
        raise ValueError(f"expected codes [N,{len(levels)}], got {tuple(codes.shape)}")
    multipliers = _mixed_radix_multipliers(levels).to(codes.device)
    return (codes.long() * multipliers.view(1, -1)).sum(dim=1)


def _tiny_predictor(
    route: str,
    *,
    levels: Sequence[int] = (5, 5, 5),
    z1_channels: int = 4,
    hard_fsq: bool = True,
) -> nn.Module:
    predictor = build_receiver_predictor(
        route,
        z1_channels=int(z1_channels),
        levels=levels,
        hidden=8,
        blocks=1,
        attention_every=0,
        heads=1,
        condition_mode="z1_x1",
        hard_fsq=bool(hard_fsq),
        height=4,
        width=4,
        flow_sample_steps=2,
        flow_sample_noise="zero",
        flow_sample_seed=20260713,
        cdcd_sample_steps=3,
        cdcd_sample_seed=20260713,
    )
    predictor.eval()
    return predictor


def _tiny_condition(*, z1_channels: int = 4, latent_size: int = 4, image_size: int = 16) -> ReceiverCondition:
    return make_receiver_condition(
        torch.randn(1, int(z1_channels), int(latent_size), int(latent_size)),
        torch.rand(1, 3, int(image_size), int(image_size)),
    )


def _assert_same_prediction(left: ReceiverPrediction, right: ReceiverPrediction) -> None:
    for name in ("q_continuous", "q_hard", "q_train"):
        assert torch.equal(getattr(left, name), getattr(right, name)), f"prediction field {name} changed"
    assert len(left.logits) == len(right.logits)
    for index, (left_logits, right_logits) in enumerate(zip(left.logits, right.logits)):
        assert torch.equal(left_logits, right_logits), f"prediction logits[{index}] changed"
    for name in ("codes", "joint_indices"):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        assert (left_value is None) == (right_value is None), f"prediction field {name} optionality changed"
        if left_value is not None:
            assert torch.equal(left_value, right_value), f"prediction field {name} changed"


def test_full_mixed_radix_roundtrip() -> None:
    """Enumerate every token for K=125/729/4913 plus a nonuniform radix."""

    level_sets = ((5, 5, 5), (9, 9, 9), (17, 17, 17), (5, 9, 17))
    for levels in level_sets:
        predictor = JointFSQIndexPredictor(
            z1_channels=4,
            levels=levels,
            hidden=8,
            blocks=0,
            attention_every=0,
            heads=1,
            condition_mode="z1_x1",
        )
        vocab_size = math.prod(levels)
        expected_indices = torch.arange(vocab_size, dtype=torch.long)
        codes = predictor.codebook_codes.cpu()
        assert tuple(codes.shape) == (vocab_size, len(levels))
        assert torch.equal(_codes_to_indices(codes, levels), expected_indices)

        # code -> q -> code is exact for every scalar level, including 17.
        codes_bchw = codes.t().contiguous().view(1, len(levels), vocab_size, 1)
        q = fsq_codes_to_q(codes_bchw, levels)
        q_roundtrip, codes_roundtrip = snap_to_fsq(q, levels)
        assert torch.equal(codes_roundtrip, codes_bchw)
        assert torch.allclose(q_roundtrip, q, rtol=0.0, atol=1e-7)

        # The joint predictor's registered q/code tables must describe the
        # same complete mixed-radix ordering used by target idx3.
        registered_q = q.squeeze(0).squeeze(-1).t().contiguous()
        assert torch.allclose(predictor.codebook.cpu(), registered_q, rtol=0.0, atol=1e-7)
        assert torch.equal(_codes_to_indices(codes_roundtrip.squeeze(0).squeeze(-1).t(), levels), expected_indices)


def test_predictor_forward_contract() -> None:
    condition = _tiny_condition()
    assert set(ReceiverCondition.__dataclass_fields__) == {"z1", "x1"}
    for route in (
        "direct_q",
        "parallel_index",
        "joint_index",
        "ar_index",
        "ar_joint_index",
        "ar_residual_index",
        "flow_matching",
        "categorical_diffusion",
    ):
        predictor = _tiny_predictor(route)
        assert_receiver_only_module(predictor)
        parameters = list(inspect.signature(predictor.forward).parameters.values())
        assert [parameter.name for parameter in parameters] == ["condition"]
        annotation = parameters[0].annotation
        assert annotation in (ReceiverCondition, "ReceiverCondition"), (
            f"{route} forward input must be annotated ReceiverCondition, got {annotation!r}"
        )
        with torch.inference_mode():
            prediction = predictor(condition)
        assert tuple(prediction.q_train.shape) == (1, 3, 4, 4)
        try:
            predictor(condition, img=torch.zeros(1, 3, 16, 16))
        except TypeError:
            pass
        else:
            raise AssertionError(f"{route} unexpectedly accepted sender-only img")


def test_supervision_invariance() -> None:
    """Changing fake sender targets cannot alter a z1/x1-only prediction."""

    condition = _tiny_condition()
    target_sets = (
        {
            "img": torch.zeros(1, 3, 16, 16),
            "z2": torch.zeros(1, 3, 4, 4),
            "q2": torch.zeros(1, 3, 4, 4),
            "oracle_indices": torch.zeros(1, 3, 4, 4, dtype=torch.long),
        },
        {
            "img": torch.randn(1, 3, 16, 16) * 100.0,
            "z2": torch.randn(1, 3, 4, 4) * 100.0,
            "q2": torch.randn(1, 3, 4, 4) * 100.0,
            "oracle_indices": torch.full((1, 3, 4, 4), 4, dtype=torch.long),
        },
    )
    for route in (
        "direct_q",
        "parallel_index",
        "joint_index",
        "ar_index",
        "ar_joint_index",
        "ar_residual_index",
        "flow_matching",
        "categorical_diffusion",
    ):
        predictor = _tiny_predictor(route)
        predictions: list[ReceiverPrediction] = []
        with torch.inference_mode():
            for source_targets in target_sets:
                assert_training_targets_are_not_inputs(
                    predictor,
                    condition,
                    source_targets=source_targets,
                )
                # The targets are deliberately not forwarded.  This is the
                # complete receiver predictor invocation.
                predictions.append(predictor(condition))
        _assert_same_prediction(predictions[0], predictions[1])


class _ContractOnlyReceiver(nn.Module):
    """Tiny receiver root used to exercise static audit failure modes."""

    def forward(self, condition: ReceiverCondition) -> torch.Tensor:
        return condition.z1


class _NestedAliasHolder(nn.Module):
    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        # ``detach`` intentionally creates a distinct Tensor object that still
        # aliases target storage.  Identity-only audits would miss this.
        self.payload = [{"nested": SimpleNamespace(cache=target.detach())}]


class _BufferAliasHolder(nn.Module):
    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("receiver_cache", target.detach())


class _ContainerAliasReceiver(_ContractOnlyReceiver):
    def __init__(self, target: torch.Tensor, *, mode: str) -> None:
        super().__init__()
        if mode == "module_dict":
            self.branches = nn.ModuleDict({"nested": _NestedAliasHolder(target)})
        elif mode == "module_list_buffer":
            self.blocks = nn.ModuleList([_BufferAliasHolder(target)])
        else:
            raise ValueError(f"unknown container test mode {mode!r}")


class _NamedStateReceiver(_ContractOnlyReceiver):
    def __init__(self, state_name: str, value: torch.Tensor) -> None:
        super().__init__()
        self.child = nn.Module()
        setattr(self.child, state_name, value)


class _NamedContainerReceiver(_ContractOnlyReceiver):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        # A clone is deliberately not a storage alias.  Its explicit sender
        # target key must nevertheless be rejected through nested containers.
        self.payload = ({"savedOracleQ2": value.clone()},)


class _LegitimateNamedReceiver(_ContractOnlyReceiver):
    def __init__(self) -> None:
        super().__init__()
        # These are intentionally close to source-target spellings but are
        # receiver implementation state, not sender supervision tensors.
        self.image_projection = nn.Identity()
        self.target_projection = nn.Identity()
        self.q2_hat = torch.zeros(1, 3, 1, 1)
        self.generated = {
            "q2_hat": torch.zeros(1, 3, 1, 1),
            "image_size": 256,
            "code_indices": torch.arange(3),
        }


def _expect_contract_failure(callback: Callable[[], None], required_text: str) -> None:
    try:
        callback()
    except AssertionError as error:
        assert required_text in str(error), f"unexpected failure: {error}"
    else:
        raise AssertionError(f"expected receiver contract failure containing {required_text!r}")


def test_recursive_sender_target_retention_audit() -> None:
    """A target alias must be found through children, containers, and buffers."""

    condition = _tiny_condition()
    target_q2 = torch.randn(1, 3, 4, 4)
    for mode, expected_path in (
        ("module_dict", "branches.nested.payload"),
        ("module_list_buffer", "blocks.0.receiver_cache"),
    ):
        receiver = _ContainerAliasReceiver(target_q2, mode=mode)
        # No forbidden *name* is present; this must reach the recursive tensor
        # alias audit instead of failing the simpler root-name check.
        assert_receiver_only_module(receiver)
        _expect_contract_failure(
            lambda receiver=receiver: assert_training_targets_are_not_inputs(
                receiver,
                condition,
                source_targets={"q2": target_q2},
            ),
            expected_path,
        )


def test_expanded_sender_names_and_legitimate_receiver_names() -> None:
    """Catch common sender aliases while retaining non-sender module names."""

    source_aliases = (
        "targetImage",
        "senderZ2",
        "cachedOracleQ2",
        "q_target",
        "cachedOracleQ3",
        "targetIndices",
        "reference_image",
        "source_idx",
    )
    for name in source_aliases:
        receiver = _NamedStateReceiver(name, torch.randn(1, 3, 4, 4))
        _expect_contract_failure(
            lambda receiver=receiver: assert_receiver_only_module(receiver),
            name,
        )

    nested_named = _NamedContainerReceiver(torch.randn(1, 3, 4, 4))
    _expect_contract_failure(
        lambda: assert_receiver_only_module(nested_named),
        "savedOracleQ2",
    )

    condition = _tiny_condition()
    receiver = _LegitimateNamedReceiver()
    # Caching a received Layer1 value is permitted even if an unrelated fake
    # img target is supplied to the supervision audit.
    receiver.cached_condition = {"x1": condition.x1.detach()}
    assert_receiver_only_module(receiver)
    for name in (
        "target_image",
        "oracle_q2",
        "sender_z2",
        "target_indices",
        "targetImage",
        "senderZ2",
        "cachedOracleQ2",
        "q_target",
        "cachedOracleQ3",
        "targetIndices",
    ):
        assert_training_targets_are_not_inputs(
            receiver,
            condition,
            source_targets={name: torch.randn(1, 3, 4, 4)},
        )


def test_ar_raster_causality() -> None:
    """Future teacher tokens must have exactly zero influence on a current AR logit."""

    predictor = _tiny_predictor("ar_index")
    for head in predictor.index_heads:
        nn.init.normal_(head.weight, std=0.02)
    condition = _tiny_condition()
    baseline = torch.zeros(1, 4, 4, dtype=torch.long)
    future_changed = baseline.clone()
    future_changed.view(-1)[6] = 5
    past_changed = baseline.clone()
    past_changed.view(-1)[0] = 5
    with torch.inference_mode():
        base_logits = predictor.forward_teacher(condition, baseline).logits
        future_logits = predictor.forward_teacher(condition, future_changed).logits
        past_logits = predictor.forward_teacher(condition, past_changed).logits
    current = (1, 0)
    future_effect = max(
        float((left[:, :, current[0], current[1]] - right[:, :, current[0], current[1]]).abs().max())
        for left, right in zip(base_logits, future_logits)
    )
    past_effect = max(
        float((left[:, :, current[0], current[1]] - right[:, :, current[0], current[1]]).abs().max())
        for left, right in zip(base_logits, past_logits)
    )
    assert future_effect == 0.0, f"AR future-token leakage={future_effect}"
    assert past_effect > 0.0, "AR masked stack does not consume already-generated past tokens"


def test_joint_token_ar_raster_causality() -> None:
    """The paper-style K-way token head must obey the same raster mask."""

    predictor = _tiny_predictor("ar_joint_index")
    nn.init.normal_(predictor.joint_head.weight, std=0.02)
    condition = _tiny_condition()
    baseline = torch.zeros(1, 4, 4, dtype=torch.long)
    future_changed = baseline.clone()
    future_changed.view(-1)[6] = 124
    past_changed = baseline.clone()
    past_changed.view(-1)[0] = 124
    with torch.inference_mode():
        base = predictor.forward_teacher(condition, baseline).logits[0]
        future = predictor.forward_teacher(condition, future_changed).logits[0]
        past = predictor.forward_teacher(condition, past_changed).logits[0]
    current = (1, 0)
    future_effect = float(
        (base[:, :, current[0], current[1]] - future[:, :, current[0], current[1]])
        .abs()
        .max()
    )
    past_effect = float(
        (base[:, :, current[0], current[1]] - past[:, :, current[0], current[1]])
        .abs()
        .max()
    )
    assert future_effect == 0.0, f"joint-token AR future leakage={future_effect}"
    assert past_effect > 0.0, "joint-token AR does not consume its generated past"


def test_channel_vq_ar_causality() -> None:
    """Channel AR may consume past indices, never future ones."""

    torch.manual_seed(20260713)
    predictor = ChannelVQAutoregressiveIndexPredictor(
        z1_channels=4,
        channels=6,
        local_code_count=4,
        hidden=8,
        blocks=1,
        attention_every=0,
        heads=1,
        condition_mode="z1_x1",
    ).eval()
    assert_receiver_only_module(predictor)
    condition = _tiny_condition(z1_channels=4, latent_size=4, image_size=16)
    target = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)
    future_changed = target.clone()
    future_changed[:, 4:] = torch.tensor([[3, 2]])
    with torch.no_grad():
        original = predictor.forward_teacher(condition, target)
        changed = predictor.forward_teacher(condition, future_changed)
        greedy_logits, greedy_indices = predictor(condition)
    # Position 4 is conditioned only on positions 0..3.  A changed target at
    # position 4 can first influence the prediction for position 5.
    future_effect = float((original[:, :5] - changed[:, :5]).abs().max())
    past_effect = float((original[:, 5:] - changed[:, 5:]).abs().max())
    assert future_effect == 0.0, f"channel AR future-token leakage={future_effect}"
    assert past_effect > 0.0, "channel AR does not consume its generated past"
    assert tuple(greedy_logits.shape) == (1, 6, 4)
    assert tuple(greedy_indices.shape) == (1, 6)
    assert int(greedy_indices.min()) >= 0 and int(greedy_indices.max()) < 4


def test_channel_vq_global_and_grouped_index_decode() -> None:
    """Both channel-VQ modes must map causal AR labels back to exact q2 tokens."""

    torch.manual_seed(20260713)
    cases = (
        ("global", 7, 7, torch.tensor([[0, 3, 6]], dtype=torch.long)),
        ("grouped", 6, 2, torch.tensor([[0, 1, 0]], dtype=torch.long)),
    )
    for mode, rate, vocabulary, labels in cases:
        quantizer = ChannelVQ(
            k_max=rate,
            channels=3,
            h=1,
            w=5,
            channel_codebook_mode=mode,
        )
        with torch.no_grad():
            quantizer.codebook.copy_(
                torch.arange(rate * 5, dtype=torch.float32).reshape(rate, 1, 5)
            )
        bundle = SimpleNamespace(codec=SimpleNamespace(quantizer=quantizer))
        q_hard = channel_ar_train.local_to_q(bundle, labels, rate)
        expected_indices = (
            labels
            if mode == "global"
            else quantizer.local_to_global_indices(labels, rate)
        )
        expected = quantizer.get_codebook_entry(expected_indices, rate, detach_codebook=True)
        assert torch.equal(q_hard, expected)
        assert tuple(q_hard.shape) == (1, 3, 1, 5)

        logits = torch.full((1, 3, vocabulary), -100.0)
        logits.scatter_(2, labels.unsqueeze(-1), 100.0)
        q_soft = channel_ar_train.logits_to_soft_q(
            bundle, logits, rate, temperature=1.0
        )
        assert torch.allclose(q_soft, expected, rtol=0.0, atol=1e-6)

        predictor = ChannelVQAutoregressiveIndexPredictor(
            z1_channels=4,
            channels=3,
            local_code_count=vocabulary,
            hidden=8,
            blocks=1,
            attention_every=0,
            heads=1,
            condition_mode="z1_x1",
        ).eval()
        assert_receiver_only_module(predictor)
        condition = _tiny_condition(z1_channels=4, latent_size=4, image_size=16)
        with torch.no_grad():
            teacher_logits = predictor.forward_teacher(condition, labels)
            generated_logits, generated_indices = predictor(condition)
        assert tuple(teacher_logits.shape) == (1, 3, vocabulary)
        assert tuple(generated_logits.shape) == (1, 3, vocabulary)
        assert tuple(generated_indices.shape) == (1, 3)
        assert int(generated_indices.min()) >= 0 and int(generated_indices.max()) < vocabulary
        index_loss = torch.nn.functional.cross_entropy(
            teacher_logits.reshape(-1, vocabulary), labels.reshape(-1)
        )
        assert torch.isfinite(index_loss)


def test_image_vq_diffusion_receiver_contract() -> None:
    torch.manual_seed(20260713)
    generator = ImageVQConditionalDiffusionGenerator(
        z1_channels=4,
        embedding_dim=6,
        height=4,
        width=4,
        hidden=8,
        blocks=1,
        attention_every=0,
        heads=1,
        diffusion_steps=4,
        sample_steps=2,
        q_scale=0.5,
    ).eval()
    assert_receiver_only_module(generator)
    condition = _tiny_condition(z1_channels=4, latent_size=4, image_size=16)
    target_q = torch.randn(1, 6, 4, 4)
    assert_training_targets_are_not_inputs(
        generator,
        condition,
        source_targets={"q2": target_q, "img": condition.x1, "z2": target_q.clone()},
    )
    with torch.no_grad():
        generated = generator(condition)
        training = generator.training_predictions(condition, target_q)
    assert tuple(generated.shape) == (1, 6, 4, 4)
    assert tuple(training["predicted_noise"].shape) == (1, 6, 4, 4)
    assert tuple(training["predicted_q"].shape) == (1, 6, 4, 4)
    assert torch.isfinite(generated).all()


def test_continuous_q_receiver_contract() -> None:
    torch.manual_seed(20260713)
    generator = ContinuousQGenerator(
        z1_channels=4,
        embedding_dim=9,
        hidden=8,
        blocks=1,
        attention_every=0,
        heads=1,
    ).eval()
    assert_receiver_only_module(generator)
    condition = _tiny_condition(z1_channels=4, latent_size=4, image_size=16)
    fake_targets = {
        "img": torch.randn(1, 3, 16, 16),
        "z2": torch.randn(1, 9, 4, 4),
        "q2": torch.randn(1, 9, 4, 4),
        "oracle_indices": torch.zeros(1, 4, 4, dtype=torch.long),
    }
    assert_training_targets_are_not_inputs(generator, condition, source_targets=fake_targets)
    with torch.no_grad():
        first = generator(condition)
        for value in fake_targets.values():
            value.add_(1000)
        second = generator(condition)
    assert tuple(first.shape) == (1, 9, 4, 4)
    assert torch.equal(first, second)


class _ConstantQMixMember(nn.Module):
    """Minimal z1/x1-only member for the frozen q-mix contract test."""

    def __init__(self, value: float, embedding_dim: int = 9) -> None:
        super().__init__()
        self.value = float(value)
        self.embedding_dim = int(embedding_dim)

    def forward(self, condition):
        condition.validate()
        return torch.full(
            (
                int(condition.z1.shape[0]),
                self.embedding_dim,
                int(condition.z1.shape[-2]),
                int(condition.z1.shape[-1]),
            ),
            self.value,
            dtype=condition.z1.dtype,
            device=condition.z1.device,
        )


def test_fixed_train_selected_q_mix_contract() -> None:
    """Fixed global q weights stay receiver-only and cannot enter train mode."""

    condition = _tiny_condition(z1_channels=4, latent_size=4, image_size=16)
    q_mix = FixedTrainSelectedQMix(
        [_ConstantQMixMember(1.0), _ConstantQMixMember(3.0)], [0.6, 0.4]
    )
    assert_receiver_only_module(q_mix)
    q_mix.train()
    assert not q_mix.training
    assert all(not member.training for member in q_mix.members)
    targets = {
        "img": torch.randn(1, 3, 16, 16),
        "z2": torch.randn(1, 9, 4, 4),
        "q2": torch.randn(1, 9, 4, 4),
        "oracle_indices": torch.zeros(1, 4, 4, dtype=torch.long),
    }
    assert_training_targets_are_not_inputs(q_mix, condition, source_targets=targets)
    with torch.no_grad():
        first = q_mix(condition)
        for value in targets.values():
            value.add_(1000)
        second = q_mix(condition)
    assert tuple(first.shape) == (1, 9, 4, 4)
    assert torch.equal(first, second)
    assert torch.allclose(first, torch.full_like(first, 1.8), rtol=0.0, atol=1e-6)


class _MeanCombiner(nn.Module):
    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        return torch.lerp(x1, u2, 0.5).clamp(0.0, 1.0)


class _TinyLayer1(nn.Module):
    """Deterministic Layer1 stand-in with the real receiver tensor shapes."""

    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        z_rgb = torch.nn.functional.interpolate(
            imgs,
            size=(4, 4),
            mode="bilinear",
            align_corners=False,
        )
        z1 = torch.cat([z_rgb, z_rgb.mean(dim=1, keepdim=True)], dim=1)
        return {"z1": z1, "x1": imgs.mul(0.75).add(0.125)}


class _TinyFSQOracle(nn.Module):
    """Small K=125 sender oracle; its E2-equivalent stays outside public forward."""

    def __init__(self) -> None:
        super().__init__()
        self.e3 = nn.Identity()

    def condition(self, _x1: torch.Tensor, _z1: torch.Tensor) -> None:
        return None

    def shuffle_q3(self, q: torch.Tensor) -> torch.Tensor:
        return torch.roll(q, shifts=1, dims=0)

    def decode(
        self,
        q: torch.Tensor,
        x1: torch.Tensor,
        _z1: torch.Tensor,
        combiner: nn.Module,
    ) -> dict[str, torch.Tensor]:
        u2_hat = torch.nn.functional.interpolate(
            q[:, :3],
            size=tuple(int(value) for value in x1.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        ).add(1.0).mul(0.5).clamp(0.0, 1.0)
        return {"u2_hat": u2_hat, "final": combiner(x1, u2_hat)}

    def forward(
        self,
        imgs: torch.Tensor,
        x1: torch.Tensor,
        _z1: torch.Tensor,
        _combiner: nn.Module,
    ) -> dict[str, torch.Tensor]:
        q_source = (
            torch.nn.functional.interpolate(
                imgs,
                size=(4, 4),
                mode="bilinear",
                align_corners=False,
            )
            * 2.0
            - 1.0
        )
        q_hard, codes = snap_to_fsq(q_source, (5, 5, 5))
        multipliers = torch.tensor((25, 5, 1), device=codes.device).view(1, 3, 1, 1)
        idx3 = (codes * multipliers).sum(dim=1)
        return {
            "z3": q_source,
            "q3": q_hard,
            "q3_hard": q_hard,
            "codes": codes,
            "idx3": idx3,
            "final": x1,
        }


class _CaptureUpsampleD2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        self.last_input = q.detach().clone()
        return torch.nn.functional.interpolate(
            q[:, :3],
            size=(16, 16),
            mode="bilinear",
            align_corners=False,
        ).add(1.0).mul(0.5)


def _tiny_receiver_bundle() -> SimpleNamespace:
    layer1 = _TinyLayer1()
    tokenizer = _TinyFSQOracle()
    combiner = _MeanCombiner()
    return SimpleNamespace(
        layer1=layer1,
        e1=layer1,
        d1=nn.Identity(),
        tokenizer=tokenizer,
        combiner=combiner,
    )


@contextmanager
def _count_prediction_paths(predictor: nn.Module) -> Iterator[dict[str, int]]:
    """Count public deployment versus supervision-only predictor calls."""

    state = {"public": 0, "teacher": 0}
    original_forward = predictor.forward

    def counted_public(*args, **kwargs):
        state["public"] += 1
        return original_forward(*args, **kwargs)

    predictor.forward = counted_public  # type: ignore[method-assign]
    original_teacher = getattr(predictor, "forward_teacher", None)
    if original_teacher is not None:
        def counted_teacher(*args, **kwargs):
            state["teacher"] += 1
            return original_teacher(*args, **kwargs)

        predictor.forward_teacher = counted_teacher  # type: ignore[attr-defined,method-assign]
    try:
        yield state
    finally:
        predictor.forward = original_forward  # type: ignore[method-assign]
        if original_teacher is not None:
            predictor.forward_teacher = original_teacher  # type: ignore[attr-defined,method-assign]


def _tiny_run_args(route: str) -> argparse.Namespace:
    """Only the options exercised by a one-batch production ``run_loader``."""

    return argparse.Namespace(
        route=str(route),
        max_train_batches=1,
        max_val_batches=0,
        joint_predictable_oracle=False,
        sender_aligned_q=False,
        residual_q=False,
        _ar_history_corruption_prob=0.0,
        lambda_flow=0.1,
        lambda_q=0.1,
        lambda_index=0.1,
        lambda_final=0.1,
        lambda_oracle=0.1,
        hard_example_power=0.0,
        hard_example_min_weight=0.0,
        hard_example_max_weight=10.0,
        lambda_zero_anchor=0.0,
        lambda_shuffle_anchor=0.0,
        grad_clip_norm=0.0,
        train_deploy_metric_batches=1,
        train_deploy_metric_batch_size=1,
        goal_delta_db=0.5,
        min_oracle_delta_db=-100.0,
        min_pred_ablation_drop=-100.0,
        min_condition_shuffle_drop=-100.0,
        min_oracle_ablation_drop=-100.0,
        sender_align_min_oracle_delta_db=-100.0,
        sender_align_min_oracle_ablation_drop=-100.0,
        sender_deployment_min_delta_db=-100.0,
        sender_deployment_min_ablation_drop=-100.0,
        sender_align_max_q_mse=1.0e9,
    )


def test_k125_ar_flow_public_path() -> None:
    """Cover K=125 AR, flow, and CDCD hard/continuous deployment paths."""

    torch.manual_seed(20260713)
    imgs = torch.rand(2, 3, 16, 16)
    for route, hard_fsq in (
        ("ar_joint_index", True),
        ("flow_matching", False),
        ("categorical_diffusion", False),
        ("categorical_diffusion", True),
    ):
        predictor = _tiny_predictor(route, hard_fsq=hard_fsq)
        bundle = _tiny_receiver_bundle()
        receiver_d2 = _CaptureUpsampleD2()
        assert_receiver_only_module(predictor)

        # Validation/deployment must call only forward(condition).  Its D2
        # input is hard FSQ for AR and continuous q2_hat for flow.
        with _count_prediction_paths(predictor) as calls:
            condition, deployed, oracle, _decoded = receiver_train.receiver_forward(
                predictor,
                bundle,
                receiver_d2,
                bundle.combiner,
                imgs,
                train=False,
                joint_predictable_oracle=False,
            )
            assert calls == {"public": 1, "teacher": 0}
            expected_deploy_q = deployed.q_hard if hard_fsq else deployed.q_continuous
            assert receiver_d2.last_input is not None
            assert torch.equal(receiver_d2.last_input, expected_deploy_q)

            # Training may call its explicitly named supervision/endpoint
            # path, but it must not silently call public forward instead.
            _condition, train_prediction, _oracle, _decoded = receiver_train.receiver_forward(
                predictor,
                bundle,
                receiver_d2,
                bundle.combiner,
                imgs,
                train=True,
                joint_predictable_oracle=False,
            )
            assert calls == {"public": 1, "teacher": 1}
            assert receiver_d2.last_input is not None
            assert torch.equal(receiver_d2.last_input, train_prediction.q_train)

        fake_targets = {
            "img": torch.randn_like(imgs),
            "z2": torch.randn_like(oracle["z3"]),
            "q2": torch.randn_like(oracle["q3_hard"]),
            "oracle_indices": torch.randint(0, 125, oracle["idx3"].shape),
        }
        assert_training_targets_are_not_inputs(
            predictor,
            condition,
            source_targets=fake_targets,
        )
        with torch.inference_mode():
            before = predictor(condition)
            for target in fake_targets.values():
                target.add_(1000)
            after = predictor(condition)
        _assert_same_prediction(before, after)


def test_cdcd_categorical_posterior_contract() -> None:
    """Exercise exact K=125 posterior means, CE, and full DDIM rollout."""

    torch.manual_seed(20260713)
    condition = _tiny_condition()
    target, target_codes = snap_to_fsq(
        torch.randn(1, 3, 4, 4), (5, 5, 5)
    )
    target_indices = (
        target_codes * torch.tensor((25, 5, 1)).view(1, 3, 1, 1)
    ).sum(dim=1)
    for hard_fsq in (False, True):
        predictor = _tiny_predictor(
            "categorical_diffusion", hard_fsq=hard_fsq
        )
        assert tuple(predictor.codebook.shape) == (125, 3)
        assert tuple(predictor.codebook_codes.shape) == (125, 3)
        assert torch.equal(
            _codes_to_indices(predictor.codebook_codes.cpu(), (5, 5, 5)),
            torch.arange(125),
        )
        assert_receiver_only_module(predictor)
        alpha, sigma = predictor._alpha_sigma(torch.tensor((0.0, 0.5, 1.0)))
        assert torch.allclose(
            alpha.square() + sigma.square(), torch.ones(3), atol=1e-6
        )
        assert torch.allclose(alpha[[0, 2]], torch.tensor((1.0, 0.0)), atol=1e-6)
        assert torch.allclose(sigma[[0, 2]], torch.tensor((0.0, 1.0)), atol=1e-6)

        # Teacher prediction is a single noisy q_t sample.  Its only paper
        # loss is 125-way CE; posterior mean and argmax share the exact grid.
        predictor.train()
        teacher = predictor.forward_teacher(condition, target)
        assert tuple(teacher.logits[0].shape) == (1, 125, 4, 4)
        probabilities = teacher.logits[0].float().softmax(dim=1)
        expected_mean = torch.einsum(
            "bkhw,kc->bchw", probabilities, predictor.codebook.float()
        ).to(teacher.q_continuous.dtype)
        assert torch.allclose(
            teacher.q_continuous, expected_mean, rtol=0.0, atol=1e-7
        )
        snapped, snapped_codes = snap_to_fsq(teacher.q_hard, (5, 5, 5))
        assert torch.equal(teacher.q_hard, snapped)
        assert torch.equal(teacher.codes, snapped_codes)
        expected_train_q = teacher.q_hard if hard_fsq else teacher.q_continuous
        assert torch.equal(teacher.q_train, expected_train_q)
        loss = torch.nn.functional.cross_entropy(
            teacher.logits[0].float(), target_indices.long()
        )
        production_loss = receiver_train.prediction_index_loss(
            teacher, target_codes, target_indices
        )
        assert torch.equal(loss, production_loss)
        loss.backward()
        assert torch.isfinite(loss)
        assert predictor.denoiser.output.weight.grad is not None
        assert torch.isfinite(predictor.denoiser.output.weight.grad).all()

        # Public forward is a complete deterministic rollout that consumes
        # only z1/x1 and makes exactly sample_steps denoiser evaluations.
        predictor.eval()
        calls = {"count": 0}

        def count_denoiser(_module, _inputs, _output):
            calls["count"] += 1

        hook = predictor.denoiser.register_forward_hook(count_denoiser)
        with torch.inference_mode():
            deployed = predictor(condition)
        hook.remove()
        assert calls["count"] == predictor.sample_steps
        assert tuple(deployed.q_continuous.shape) == (1, 3, 4, 4)
        assert tuple(deployed.joint_indices.shape) == (1, 4, 4)
        assert torch.isfinite(deployed.q_continuous).all()

    try:
        build_receiver_predictor(
            "categorical_diffusion",
            z1_channels=4,
            levels=(3, 3, 3),
            hidden=8,
            blocks=1,
            attention_every=0,
            heads=1,
            condition_mode="z1_x1",
            hard_fsq=False,
            height=4,
            width=4,
            cdcd_sample_steps=3,
        )
    except ValueError as error:
        assert "fixed to K=125" in str(error)
    else:
        raise AssertionError("CDCD unexpectedly accepted a non-K=125 grid")

    ce_only = argparse.Namespace(
        route="categorical_diffusion",
        hard_fsq=False,
        ar_history_corruption_start=0.0,
        ar_history_corruption_end=0.0,
        ar_rollout_history_batch=1,
        flow_train_sample_steps=1,
        flow_base_checkpoint="",
        train_deploy_metric_batches=1,
        train_deploy_metric_batch_size=1,
        cdcd_sample_steps=3,
        cdcd_time_scale=1000.0,
        cdcd_prior_scale=1.0,
        lambda_index=1.0,
        lambda_flow=0.0,
        lambda_q=0.0,
        lambda_final=0.0,
        lambda_oracle=0.0,
        lambda_zero_anchor=0.0,
        lambda_shuffle_anchor=0.0,
    )
    receiver_train.validate_ar_history_args(ce_only)
    ce_only.lambda_final = 0.1
    try:
        receiver_train.validate_ar_history_args(ce_only)
    except ValueError as error:
        assert "CE-only" in str(error)
    else:
        raise AssertionError("CDCD unexpectedly accepted an image-loss auxiliary")


def test_ifsq_transport_training_contract() -> None:
    """Cover the released iFSQ Gaussian/logit-normal/cosine flow options."""

    torch.manual_seed(20260713)
    condition = _tiny_condition()
    predictor = build_receiver_predictor(
        "flow_matching",
        z1_channels=4,
        levels=(5, 5, 5),
        hidden=8,
        blocks=1,
        attention_every=0,
        heads=1,
        condition_mode="z1_x1",
        hard_fsq=False,
        height=4,
        width=4,
        flow_sample_steps=2,
        flow_train_sample_steps=2,
        flow_sample_noise="gaussian",
        flow_sample_seed=20260713,
        flow_timestep_sampling="logit_normal",
        flow_cosine_loss_weight=1.0,
    )
    target, _codes = snap_to_fsq(torch.randn(1, 3, 4, 4), (5, 5, 5))
    sampled_t = predictor._sample_training_time(target)
    assert tuple(sampled_t.shape) == (1,)
    assert bool(((sampled_t > 0.0) & (sampled_t < 1.0)).all())

    prediction = predictor.forward_teacher(condition, target)
    assert prediction.loss_flow is not None
    assert prediction.loss_flow_mse is not None
    assert prediction.loss_flow_cosine is not None
    assert torch.isfinite(prediction.loss_flow)
    assert torch.isfinite(prediction.loss_flow_mse)
    assert torch.isfinite(prediction.loss_flow_cosine)
    assert torch.allclose(
        prediction.loss_flow,
        prediction.loss_flow_mse + prediction.loss_flow_cosine,
        rtol=1e-6,
        atol=1e-7,
    )


def test_ar_joint_route_requires_hard_fsq() -> None:
    """Discrete K-way AR is a hard-index route, never continuous q2_hat."""

    common = dict(
        route="ar_joint_index",
        ar_history_corruption_start=0.0,
        ar_history_corruption_end=0.0,
        ar_rollout_history_batch=1,
        flow_train_sample_steps=1,
        train_deploy_metric_batches=1,
        train_deploy_metric_batch_size=1,
    )
    receiver_train.validate_ar_history_args(
        argparse.Namespace(**common, hard_fsq=True)
    )
    try:
        receiver_train.validate_ar_history_args(
            argparse.Namespace(**common, hard_fsq=False)
        )
    except ValueError as error:
        assert "requires --hard-fsq" in str(error)
    else:
        raise AssertionError("ar_joint_index unexpectedly accepted continuous q2_hat mode")


def test_k125_generation_metric_scopes() -> None:
    """Proxy/endpoint metrics cannot masquerade as deploy accuracy."""

    imgs = torch.rand(1, 3, 16, 16)
    loader = [(imgs, torch.zeros(1, dtype=torch.long))]
    expected_train_prefix = {
        "ar_joint_index": "train_proxy_",
        "flow_matching": "train_endpoint_",
        "categorical_diffusion": "train_teacher_",
    }
    route_modes = {
        "ar_joint_index": True,
        "flow_matching": False,
        "categorical_diffusion": False,
    }
    for route, prefix in expected_train_prefix.items():
        torch.manual_seed(20260713)
        predictor = _tiny_predictor(route, hard_fsq=route_modes[route])
        bundle = _tiny_receiver_bundle()
        receiver_d2 = _CaptureUpsampleD2()
        optimizer = torch.optim.SGD(predictor.parameters(), lr=0.0)
        args = _tiny_run_args(route)
        train_metrics = receiver_train.run_loader(
            loader,
            predictor=predictor,
            bundle=bundle,
            receiver_d2=receiver_d2,
            receiver_combiner=bundle.combiner,
            optimizer=optimizer,
            args=args,
            device=torch.device("cpu"),
            train=True,
        )
        assert f"{prefix}index_accuracy" in train_metrics
        assert "train_deploy_index_accuracy" in train_metrics
        assert "index_accuracy" not in train_metrics, (
            f"{route} train-only prediction was mislabeled as deploy index_accuracy"
        )
        for other_prefix in ("train_proxy_", "train_endpoint_", "train_teacher_"):
            if other_prefix != prefix:
                assert f"{other_prefix}index_accuracy" not in train_metrics

        # Strict validation calls the public receiver path and therefore
        # retains the unprefixed deployment name only.
        val_metrics = receiver_train.run_loader(
            loader,
            predictor=predictor,
            bundle=bundle,
            receiver_d2=receiver_d2,
            receiver_combiner=bundle.combiner,
            optimizer=None,
            args=args,
            device=torch.device("cpu"),
            train=False,
        )
        assert "index_accuracy" in val_metrics
        assert "train_proxy_index_accuracy" not in val_metrics
        assert "train_endpoint_index_accuracy" not in val_metrics
        assert "train_teacher_index_accuracy" not in val_metrics
        assert "train_deploy_index_accuracy" not in val_metrics


def test_direct_q_hard_continuous_metric_contract_unchanged() -> None:
    """The metric split is opt-in to generative forward_teacher routes only."""

    imgs = torch.rand(1, 3, 16, 16)
    loader = [(imgs, torch.zeros(1, dtype=torch.long))]
    for hard_fsq in (True, False):
        torch.manual_seed(20260713)
        predictor = _tiny_predictor("direct_q", hard_fsq=hard_fsq)
        assert not hasattr(predictor, "forward_teacher")
        bundle = _tiny_receiver_bundle()
        receiver_d2 = _CaptureUpsampleD2()
        optimizer = torch.optim.SGD(predictor.parameters(), lr=0.0)
        args = _tiny_run_args("direct_q")
        train_metrics = receiver_train.run_loader(
            loader,
            predictor=predictor,
            bundle=bundle,
            receiver_d2=receiver_d2,
            receiver_combiner=bundle.combiner,
            optimizer=optimizer,
            args=args,
            device=torch.device("cpu"),
            train=True,
        )
        assert train_metrics["receiver_only_audit"] == 1.0
        assert "index_accuracy" in train_metrics
        assert not any(
            name.startswith(("train_proxy_", "train_endpoint_", "train_deploy_"))
            for name in train_metrics
        )

        # receiver_forward itself still selects q_hard versus q_continuous by
        # the existing hard_fsq flag; the metric fix must not change decoding.
        _condition, prediction, _oracle, _decoded = receiver_train.receiver_forward(
            predictor,
            bundle,
            receiver_d2,
            bundle.combiner,
            imgs,
            train=False,
            joint_predictable_oracle=False,
        )
        expected_q = prediction.q_hard if hard_fsq else prediction.q_continuous
        assert receiver_d2.last_input is not None
        assert torch.equal(receiver_d2.last_input, expected_q)


@contextmanager
def _sender_encoder_raises(module: nn.Module) -> Iterator[dict[str, int]]:
    original_forward = module.forward
    state = {"calls": 0}

    def forbidden_sender_forward(*_args, **_kwargs):
        state["calls"] += 1
        raise AssertionError("sender-only Layer2 E2/e3 was called by receiver decode")

    module.forward = forbidden_sender_forward  # type: ignore[method-assign]
    try:
        yield state
    finally:
        module.forward = original_forward  # type: ignore[method-assign]


def _tiny_real_tokenizer(device: torch.device) -> nn.Module:
    args = argparse.Namespace(
        arch="cnn",
        condition_mode="none",
        fsq_d=3,
        fsq_levels="5,5,5",
        e3_base_ch=4,
        e3_num_res=1,
        x1_cond_ch=4,
        z1_cond_ch=4,
        z1_cond_depth=1,
        cond_base_ch=4,
        cond_num_res=1,
        d3_base_ch=4,
        d3_num_res=1,
        latent_ch=4,
        no_pre_norm=True,
    )
    return receiver_train.base.Layer3FSQTokenizer(args, device).eval()


def test_decode_without_sender_encoder() -> None:
    """Exercise the production tokenizer.decode while E2/e3 is forbidden."""

    device = torch.device("cpu")
    tokenizer = _tiny_real_tokenizer(device)
    predictor = _tiny_predictor("joint_index", z1_channels=4).to(device)
    combiner = _MeanCombiner().to(device)
    condition = make_receiver_condition(
        torch.randn(1, 4, 16, 16, device=device),
        torch.rand(1, 3, 256, 256, device=device),
    )
    with torch.inference_mode(), _sender_encoder_raises(tokenizer.e3) as sender_state:
        prediction = predictor(condition)
        decoded = tokenizer.decode(prediction.q_hard, condition.x1, condition.z1, combiner)
    assert sender_state["calls"] == 0
    assert tuple(decoded["u2_hat"].shape) == (1, 3, 256, 256)
    assert tuple(decoded["final"].shape) == (1, 3, 256, 256)
    assert torch.isfinite(decoded["final"]).all()


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else CDDM_ROOT / candidate


def _resolve_device(name: str, cuda_index: int) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device(f"cuda:{int(cuda_index)}")
    if name != "auto":
        raise ValueError(f"unknown device mode {name!r}")
    return torch.device(f"cuda:{int(cuda_index)}" if torch.cuda.is_available() else "cpu")


def run_real_k125_goal_best_closure(
    checkpoint_path: Path,
    *,
    device: torch.device,
    goal_delta_db: float,
    require_receiver_goal: bool,
) -> dict[str, float | str]:
    """Close E1/D1 -> saved receiver -> D2/combiner without invoking E2."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert str(payload.get("stage", "")) == "explore2_fsq_receiver"
    contract = payload.get("receiver_contract", {})
    assert contract.get("inputs") == ["z1", "x1"]
    assert {"img", "z2", "q2", "oracle_indices"}.issubset(set(contract.get("forbidden_inputs", [])))

    receiver_args = argparse.Namespace(**payload["args"])
    oracle_path = _resolve_path(payload.get("oracle_checkpoint") or receiver_args.oracle_checkpoint)
    assert oracle_path.is_file(), f"K=125 oracle checkpoint is missing: {oracle_path}"
    assert oracle_path.name.endswith("_goal_best.pth"), f"expected a goal_best oracle, got {oracle_path.name}"

    bundle, oracle_args, oracle_payload = receiver_train.load_fsq_oracle(str(oracle_path), device)
    levels = receiver_train.base.parse_fsq_levels(oracle_args.fsq_levels, int(oracle_args.fsq_d))
    assert tuple(levels) == (5, 5, 5), f"expected K=125 levels, got {levels}"
    assert math.prod(levels) == 125
    oracle_metrics = oracle_payload.get("metrics", {})
    oracle_delta = float(oracle_metrics.get("delta_x1", float("-inf")))
    assert float(oracle_metrics.get("goal_eligible", 0.0)) == 1.0
    assert oracle_delta >= float(goal_delta_db), (
        f"K=125 goal_best oracle delta_x1={oracle_delta:.6f} dB < {goal_delta_db:.6f} dB"
    )

    predictor = receiver_train.build_predictor(receiver_args, oracle_args, device)
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    predictor.eval()
    assert_receiver_only_module(predictor)

    # A real frozen Layer1 creates the receiver condition.  The source image is
    # discarded before the patched E2/e3 boundary is entered.
    with torch.inference_mode():
        image = torch.rand(1, 3, 256, 256, device=device)
        layer1 = bundle.layer1(image)
        condition = make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
        del image
        with _sender_encoder_raises(bundle.tokenizer.e3) as sender_state:
            prediction = predictor(condition)
            q_used = prediction.q_hard if bool(getattr(predictor, "hard_fsq", True)) else prediction.q_continuous
            decoded = bundle.tokenizer.decode(q_used, condition.x1, condition.z1, bundle.combiner)

    assert sender_state["calls"] == 0
    assert tuple(prediction.q_hard.shape) == (1, 3, 16, 16)
    assert tuple(decoded["u2_hat"].shape) == (1, 3, 256, 256)
    assert tuple(decoded["final"].shape) == (1, 3, 256, 256)
    assert torch.isfinite(decoded["final"]).all()

    receiver_metrics = payload.get("metrics", {})
    receiver_delta = float(receiver_metrics.get("delta_x1", float("-inf")))
    receiver_goal_met = receiver_delta >= float(goal_delta_db)
    if require_receiver_goal:
        assert receiver_goal_met, (
            f"saved receiver delta_x1={receiver_delta:.6f} dB < {goal_delta_db:.6f} dB; "
            "the structural receiver-only closure passes, but the quality goal does not"
        )
    return {
        "device": str(device),
        "oracle_delta_x1_db": oracle_delta,
        "receiver_saved_delta_x1_db": receiver_delta,
        "receiver_goal_met": float(receiver_goal_met),
    }


def test_div2k_crop_protocol() -> None:
    train_loader = SimpleNamespace(
        dataset=SimpleNamespace(
            transform=transforms.Compose(
                [
                    transforms.RandomCrop((256, 256)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
        )
    )
    val_loader = SimpleNamespace(
        dataset=SimpleNamespace(
            transform=transforms.Compose(
                [transforms.CenterCrop((256, 256)), transforms.ToTensor()]
            )
        )
    )
    assert_div2k_crop_protocol(train_loader, val_loader)
    try:
        assert_div2k_crop_protocol(val_loader, train_loader)
    except AssertionError:
        pass
    else:
        raise AssertionError("swapped random/center crop protocol was not rejected")


def _run_named(name: str, test: Callable[[], None]) -> None:
    test()
    print(f"[PASS] {name}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--real-k125",
        choices=("auto", "require", "skip"),
        default="auto",
        help="Run the saved K=125 goal_best closure when available, require it, or skip it.",
    )
    parser.add_argument("--k125-receiver-checkpoint", default=DEFAULT_K125_RECEIVER)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--goal-delta-db", type=float, default=0.5)
    parser.add_argument(
        "--require-receiver-goal",
        action="store_true",
        help="Also fail unless the saved receiver validation delta is at least --goal-delta-db.",
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))

    _run_named("full_mixed_radix_roundtrip", test_full_mixed_radix_roundtrip)
    _run_named("predictor_forward_contract", test_predictor_forward_contract)
    _run_named("supervision_invariance", test_supervision_invariance)
    _run_named("recursive_sender_target_retention_audit", test_recursive_sender_target_retention_audit)
    _run_named(
        "expanded_sender_names_and_legitimate_receiver_names",
        test_expanded_sender_names_and_legitimate_receiver_names,
    )
    _run_named("ar_raster_causality", test_ar_raster_causality)
    _run_named("joint_token_ar_raster_causality", test_joint_token_ar_raster_causality)
    _run_named("channel_vq_ar_causality", test_channel_vq_ar_causality)
    _run_named(
        "channel_vq_global_and_grouped_index_decode",
        test_channel_vq_global_and_grouped_index_decode,
    )
    _run_named("image_vq_diffusion_receiver_contract", test_image_vq_diffusion_receiver_contract)
    _run_named("continuous_q_receiver_contract", test_continuous_q_receiver_contract)
    _run_named("fixed_train_selected_q_mix_contract", test_fixed_train_selected_q_mix_contract)
    _run_named(
        "k125_ar_flow_public_path",
        test_k125_ar_flow_public_path,
    )
    _run_named(
        "cdcd_categorical_posterior_contract",
        test_cdcd_categorical_posterior_contract,
    )
    _run_named(
        "ifsq_transport_training_contract",
        test_ifsq_transport_training_contract,
    )
    _run_named("ar_joint_route_requires_hard_fsq", test_ar_joint_route_requires_hard_fsq)
    _run_named("k125_generation_metric_scopes", test_k125_generation_metric_scopes)
    _run_named(
        "direct_q_hard_continuous_metric_contract_unchanged",
        test_direct_q_hard_continuous_metric_contract_unchanged,
    )
    _run_named("decode_without_sender_encoder", test_decode_without_sender_encoder)
    _run_named("div2k_crop_protocol", test_div2k_crop_protocol)

    checkpoint_path = _resolve_path(args.k125_receiver_checkpoint)
    if str(args.real_k125) == "skip":
        print("[SKIP] real_k125_goal_best_closure (--real-k125 skip)", flush=True)
    elif not checkpoint_path.is_file():
        if str(args.real_k125) == "require":
            raise FileNotFoundError(f"required K=125 receiver checkpoint is missing: {checkpoint_path}")
        print(f"[SKIP] real_k125_goal_best_closure (missing {checkpoint_path})", flush=True)
    else:
        device = _resolve_device(str(args.device), int(args.cuda_index))
        result = run_real_k125_goal_best_closure(
            checkpoint_path,
            device=device,
            goal_delta_db=float(args.goal_delta_db),
            require_receiver_goal=bool(args.require_receiver_goal),
        )
        print(f"[PASS] real_k125_goal_best_closure {result}", flush=True)
    print("receiver contract tests: PASS", flush=True)


if __name__ == "__main__":
    main()
