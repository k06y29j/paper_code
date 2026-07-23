#!/usr/bin/env python3
"""Receiver/source tensor contracts for the explore-2 experiments.

The receiver-side generator is intentionally given a small typed object rather
than a free-form dictionary.  This makes ``img``/``z2``/oracle ``q2`` leakage a
construction error instead of a convention hidden inside a training loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn


# These are *sender-side targets*, not generic words that happen to occur in a
# receiver model.  In particular, do not turn this into a substring test: an
# ``image_projection`` or ``q2_hat`` receiver module is legitimate, whereas
# ``cached_oracle_q2`` and ``target_image`` are not.
SOURCE_ONLY_NAMES = frozenset(
    {
        # Raw source image / image supervision aliases.
        "img",
        "imgs",
        "image",
        "images",
        "source",
        "source_img",
        "source_image",
        "source_images",
        "sender_img",
        "sender_image",
        "sender_images",
        "target_img",
        "target_image",
        "target_images",
        "oracle_img",
        "oracle_image",
        "ground_truth",
        "groundtruth",
        "gt",
        "gt_img",
        "gt_image",
        "reference_img",
        "reference_image",
        "original_img",
        "original_image",
        # Layer2 sender latent aliases.
        "z2",
        "z_2",
        "source_z2",
        "sender_z2",
        "target_z2",
        "oracle_z2",
        "z2_source",
        "z2_sender",
        "z2_target",
        "z2_oracle",
        "layer2_z",
        "layer2_latent",
        "latent2",
        "e2_latent",
        "e2_output",
        "z_target",
        "target_z",
        "latent_target",
        "target_latent",
        # Older FSQ helpers call the same Layer2 signal z3/q3.  Treat those
        # names identically so a wrapper cannot bypass the audit by spelling.
        "z3",
        "z_3",
        "source_z3",
        "sender_z3",
        "target_z3",
        "oracle_z3",
        "z3_source",
        "z3_sender",
        "z3_target",
        "z3_oracle",
        # Quantized Layer2 sender target aliases.
        "q2",
        "q_2",
        "source_q2",
        "sender_q2",
        "target_q2",
        "oracle_q2",
        "q2_source",
        "q2_sender",
        "q2_target",
        "q2_oracle",
        "quantized_q2",
        "q_target",
        "target_q",
        "q_oracle",
        "oracle_q",
        "q_sender",
        "sender_q",
        "q_source",
        "source_q",
        "q_hard_target",
        "q_target_hard",
        "q3",
        "q_3",
        "source_q3",
        "sender_q3",
        "target_q3",
        "oracle_q3",
        "q3_source",
        "q3_sender",
        "q3_target",
        "q3_oracle",
        # Layer2 code/index supervision aliases.  Bare ``indices`` is not
        # forbidden because static codebook tables may legitimately use it.
        "idx2",
        "idx_2",
        "index2",
        "indices2",
        "q2_idx",
        "q2_index",
        "q2_indices",
        "z2_idx",
        "z2_index",
        "z2_indices",
        "oracle_idx",
        "oracle_index",
        "oracle_indices",
        "target_idx",
        "target_index",
        "target_indices",
        "sender_idx",
        "sender_index",
        "sender_indices",
        "source_idx",
        "source_index",
        "source_indices",
        "oracle_codes",
        "target_codes",
        "sender_codes",
        "source_codes",
        "idx3",
        "idx_3",
        "index3",
        "indices3",
        "q3_idx",
        "q3_index",
        "q3_indices",
        "z3_idx",
        "z3_index",
        "z3_indices",
    }
)

_NAME_TOKEN_RE = re.compile(r"([a-z0-9])([A-Z])")
_PROVENANCE_TOKENS = frozenset(
    {
        "source",
        "sender",
        "target",
        "oracle",
        "ground",
        "truth",
        "gt",
        "reference",
        "original",
    }
)
_RETENTION_TOKENS = frozenset(
    {"cache", "cached", "save", "saved", "store", "stored", "retain", "retained"}
)
_DIRECT_LAYER2_TARGET_TOKENS = frozenset(
    {
        "z2",
        "q2",
        "idx2",
        "index2",
        "indices2",
        "latent2",
        "z3",
        "q3",
        "idx3",
        "index3",
        "indices3",
    }
)
_GENERIC_INDEX_TOKENS = frozenset({"idx", "index", "indices"})
_IMAGE_TOKENS = frozenset({"img", "imgs", "image", "images"})

# Internal ``nn.Module`` structures are traversed deliberately below.  Hook
# registries and Python bookkeeping are not model state and walking arbitrary
# closures would make the audit both noisy and non-deterministic.
_MODULE_INTERNAL_SKIP = frozenset(
    {
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_is_full_backward_hook",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    }
)


@dataclass(frozen=True)
class ReceiverCondition:
    """Information available after Layer1 decoding at the receiver.

    ``z1`` is the received Layer1 latent and ``x1`` is deterministically
    decoded from it.  No Layer2 sender tensor can be stored in this object.
    """

    z1: torch.Tensor
    x1: torch.Tensor

    def validate(self) -> "ReceiverCondition":
        if self.z1.ndim != 4:
            raise ValueError(f"receiver z1 must be BCHW, got {tuple(self.z1.shape)}")
        if self.x1.ndim != 4 or int(self.x1.shape[1]) != 3:
            raise ValueError(f"receiver x1 must be [B,3,H,W], got {tuple(self.x1.shape)}")
        if int(self.z1.shape[0]) != int(self.x1.shape[0]):
            raise ValueError("receiver z1/x1 batch sizes differ")
        if self.z1.device != self.x1.device:
            raise ValueError("receiver z1/x1 must be on the same device")
        return self


def make_receiver_condition(z1: torch.Tensor, x1: torch.Tensor, *, detach: bool = False) -> ReceiverCondition:
    if detach:
        z1 = z1.detach()
        x1 = x1.detach()
    return ReceiverCondition(z1=z1, x1=x1).validate()


def _name_tokens(name: str) -> tuple[str, ...]:
    """Split Python/camel-case names while preserving ``z2``/``q2`` tokens."""

    snake = _NAME_TOKEN_RE.sub(r"\1_\2", str(name)).lower()
    raw = [piece for piece in re.split(r"[^a-z0-9]+", snake) if piece]
    merged: list[str] = []
    index = 0
    while index < len(raw):
        if (
            raw[index] in {"z", "q", "idx", "index", "indices", "latent"}
            and index + 1 < len(raw)
            and raw[index + 1] in {"2", "3"}
        ):
            merged.append(f"{raw[index]}{raw[index + 1]}")
            index += 2
        else:
            merged.append(raw[index])
            index += 1
    return tuple(merged)


def _normalised_name(name: str) -> str:
    return "".join(_name_tokens(name))


_NORMALISED_SOURCE_ONLY_NAMES = frozenset(_normalised_name(name) for name in SOURCE_ONLY_NAMES)


def _is_forbidden_sender_target_name(name: str) -> bool:
    """Recognise sender target aliases without matching harmless head names.

    The explicit allow-by-structure rule matters for modules such as
    ``image_projection`` and a generated ``q2_hat``: neither is a sender
    tensor.  A provenance/retention qualifier is required for generic image
    and index names, while bare Layer2 target names (``z2``/``q2``/``idx2``)
    remain forbidden.
    """

    normalised = _normalised_name(name)
    if normalised in _NORMALISED_SOURCE_ONLY_NAMES:
        return True
    tokens = set(_name_tokens(name))
    if not tokens:
        return False
    qualifiers = tokens & (_PROVENANCE_TOKENS | _RETENTION_TOKENS)
    layer2_tokens = tokens & _DIRECT_LAYER2_TARGET_TOKENS
    if layer2_tokens and qualifiers:
        return True
    if (tokens & _IMAGE_TOKENS) and qualifiers:
        return True
    if (tokens & _GENERIC_INDEX_TOKENS) and (tokens & _PROVENANCE_TOKENS):
        return True
    return False


def _tensor_storage_key(value: torch.Tensor) -> tuple[str, int | None, int, int] | None:
    """Return a conservative dense-storage identity for alias detection."""

    try:
        if value.layout != torch.strided or value.device.type == "meta":
            return None
        storage = value.untyped_storage()
        pointer = int(storage.data_ptr())
        # Empty tensors may legitimately have a shared/null storage pointer;
        # object identity still catches direct retention in that case.
        if pointer == 0:
            return None
        return (str(value.device.type), value.device.index, pointer, int(storage.nbytes()))
    except (AttributeError, RuntimeError):
        return None


def _tensors_alias(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left is right:
        return True
    left_key = _tensor_storage_key(left)
    right_key = _tensor_storage_key(right)
    return left_key is not None and left_key == right_key


def _walk_retained_value(
    value: Any,
    path: str,
    *,
    visited_values: set[int],
    visit_module: Callable[..., Iterator[tuple[str, str | None, Any]]] | None = None,
) -> Iterator[tuple[str, str | None, Any]]:
    """Walk tensors hidden in ordinary Python containers and child modules."""

    if isinstance(value, torch.Tensor):
        yield path, None, value
        return
    if isinstance(value, nn.Module):
        # The enclosing module walker follows registered children separately.
        # This branch also catches a module hidden in a plain list/dict.
        if visit_module is not None:
            yield from visit_module(value, path)
        return
    if value is None or isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return
    if inspect.isroutine(value) or inspect.ismodule(value) or isinstance(value, type):
        return

    value_id = id(value)
    if value_id in visited_values:
        return
    visited_values.add(value_id)

    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            yield f"{path}[{key_text!r}]", key_text, nested
            yield from _walk_retained_value(
                nested,
                f"{path}[{key_text!r}]",
                visited_values=visited_values,
                visit_module=visit_module,
            )
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, nested in enumerate(value):
            nested_path = f"{path}[{index}]"
            yield nested_path, None, nested
            yield from _walk_retained_value(
                nested,
                nested_path,
                visited_values=visited_values,
                visit_module=visit_module,
            )
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            nested = getattr(value, field.name)
            nested_path = f"{path}.{field.name}"
            yield nested_path, field.name, nested
            yield from _walk_retained_value(
                nested,
                nested_path,
                visited_values=visited_values,
                visit_module=visit_module,
            )
        return
    # Small custom holders (for example SimpleNamespace) are legitimate
    # containers too.  Avoid crawling arbitrary callable/torch internals above
    # and keep the visited set so a back-reference cannot recurse forever.
    try:
        attributes = vars(value)
    except TypeError:
        return
    for name, nested in attributes.items():
        nested_path = f"{path}.{name}"
        yield nested_path, str(name), nested
        yield from _walk_retained_value(
            nested,
            nested_path,
            visited_values=visited_values,
            visit_module=visit_module,
        )


def _assert_no_forbidden_sender_names(module: nn.Module) -> None:
    """Reject sender-target-looking state anywhere below a receiver root."""

    # Child modules hidden in ordinary containers are uncommon, but still need
    # to be examined.  The nested walker closes this gap without changing the
    # normal registered-module traversal.
    visited_modules: set[int] = set()
    visited_values: set[int] = set()

    def visit_module(current: nn.Module, path: str) -> Iterator[tuple[str, str | None, Any]]:
        module_id = id(current)
        if module_id in visited_modules:
            return
        visited_modules.add(module_id)
        for name, value in getattr(current, "_parameters", {}).items():
            yield f"{path}.{name}", str(name), value
        for name, value in getattr(current, "_buffers", {}).items():
            yield f"{path}.{name}", str(name), value
        for name, value in vars(current).items():
            if name in _MODULE_INTERNAL_SKIP:
                continue
            attr_path = f"{path}.{name}"
            yield attr_path, str(name), value
            yield from _walk_retained_value(
                value,
                attr_path,
                visited_values=visited_values,
                visit_module=visit_module,
            )
        for name, child in getattr(current, "_modules", {}).items():
            if child is not None:
                yield from visit_module(child, f"{path}.{name}")

    for path, name, _value in visit_module(module, "<receiver>"):
        if name is not None and _is_forbidden_sender_target_name(name):
            raise AssertionError(f"receiver predictor stores forbidden sender-target state at {path!r}")


def _assert_no_retained_source_targets(
    module: nn.Module,
    condition: ReceiverCondition,
    source_targets: Mapping[str, torch.Tensor],
) -> None:
    """Catch direct and view/detach aliases retained below receiver modules."""

    allowed_receiver_tensors = (condition.z1, condition.x1)
    target_tensors = [
        (str(name), value)
        for name, value in source_targets.items()
        if isinstance(value, torch.Tensor)
        and not any(_tensors_alias(value, allowed) for allowed in allowed_receiver_tensors)
    ]
    if not target_tensors:
        return

    visited_modules: set[int] = set()
    visited_values: set[int] = set()

    def visit_module(current: nn.Module, path: str) -> Iterator[tuple[str, str | None, Any]]:
        module_id = id(current)
        if module_id in visited_modules:
            return
        visited_modules.add(module_id)
        for name, value in getattr(current, "_parameters", {}).items():
            yield f"{path}.{name}", str(name), value
        for name, value in getattr(current, "_buffers", {}).items():
            yield f"{path}.{name}", str(name), value
        for name, value in vars(current).items():
            if name in _MODULE_INTERNAL_SKIP:
                continue
            attr_path = f"{path}.{name}"
            yield attr_path, str(name), value
            yield from _walk_retained_value(
                value,
                attr_path,
                visited_values=visited_values,
                visit_module=visit_module,
            )
        for name, child in getattr(current, "_modules", {}).items():
            if child is not None:
                yield from visit_module(child, f"{path}.{name}")

    for path, _name, value in visit_module(module, "<receiver>"):
        if not isinstance(value, torch.Tensor):
            continue
        # z1/x1 are the explicit receiver inputs and may be cached by an
        # implementation.  They are never evidence of sender leakage.
        if any(_tensors_alias(value, allowed) for allowed in allowed_receiver_tensors):
            continue
        for target_name, target in target_tensors:
            if _tensors_alias(value, target):
                raise AssertionError(
                    "receiver predictor retained supervised sender target "
                    f"{target_name!r} at {path!r} (same tensor/storage alias)"
                )


def assert_receiver_only_module(module: nn.Module) -> None:
    """Fail if a predictor exposes a source tensor in its forward contract."""

    parameters = list(inspect.signature(module.forward).parameters.values())
    names = [parameter.name for parameter in parameters]
    if names != ["condition"]:
        raise AssertionError(
            f"receiver predictor forward must be forward(condition), got parameters={names}"
        )
    _assert_no_forbidden_sender_names(module)


def assert_training_targets_are_not_inputs(
    module: nn.Module,
    condition: ReceiverCondition,
    *,
    source_targets: dict[str, torch.Tensor],
) -> None:
    """Runtime audit used by smoke tests and the first real training batch.

    Supervised targets may contain sender tensors, but the predictor invocation
    is performed before this function looks at them and accepts only
    ``ReceiverCondition``.  The checks also catch accidental storage of a
    target tensor on the predictor object.
    """

    assert_receiver_only_module(module)
    condition.validate()
    for name in source_targets:
        if not _is_forbidden_sender_target_name(str(name)):
            raise AssertionError(f"unclassified supervised target {name!r}")
    _assert_no_retained_source_targets(module, condition, source_targets)


def assert_div2k_crop_protocol(train_loader, val_loader) -> None:
    """Enforce RandomCrop for training and CenterCrop for validation."""

    train_transform = repr(getattr(train_loader.dataset, "transform", None))
    val_transform = repr(getattr(val_loader.dataset, "transform", None))
    if "RandomCrop" not in train_transform:
        raise AssertionError(f"training transform must contain RandomCrop, got {train_transform}")
    if "CenterCrop" in train_transform:
        raise AssertionError(f"training transform must not contain CenterCrop, got {train_transform}")
    if "CenterCrop" not in val_transform:
        raise AssertionError(f"validation transform must contain CenterCrop, got {val_transform}")
    if "RandomCrop" in val_transform:
        raise AssertionError(f"validation transform must not contain RandomCrop, got {val_transform}")
