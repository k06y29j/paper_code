from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent
CDDM_ROOT = THIS_DIR.parents[1]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from Autoencoder.data.datasets import get_loader

from common import (
    averaged,
    batch_metric_mean,
    check_jsccf_args,
    meters,
    mse_per_image,
    psnr_per_image,
    recon_loss,
    resolve_path,
    seed_everything,
    setup_log_file,
    should_save_latest,
    should_validate,
    ssim_per_image,
    write_json,
)
from model import OutputsCombiner, build_jscc_decoder, build_jscc_encoder
from test_ed import CNNAnalysisEncoder, CNNBottleneckDecoder, ConvNormAct, ResidualBlock


def load_local_io() -> ModuleType:
    spec = importlib.util.spec_from_file_location("jsccf_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_script_module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, THIS_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = load_local_io()


PRESETS: dict[str, tuple[int, str]] = {
    "t0": (3, "8,8,8"),
    "t1": (3, "16,16,16"),
    "t2": (3, "20,16,16"),
    "t3": (4, "8,8,8,8"),
    "t4": (4, "16,16,16,16"),
}

DEFAULT_LAYER2_CKPTS = {
    "cnn": "MY-V2/jscc-f/checkpoints/jscc_f_cnn-stage2-no_compressor-gpu2_layer2_v2_no_compressor_combiner_best.pth",
    "swin": "MY-V2/jscc-f/checkpoints/jscc_f_swin320_layer2_swin_no_compressor_combiner_best.pth",
}
DEFAULT_SAVE_DIR = str(CDDM_ROOT / "MY-V2" / "jscc-f" / "checkpoints-fsq")

TEACHER_ARG_KEYS = {
    "variant",
    "latent_ch",
    "c1_ch",
    "latent_h",
    "latent_w",
    "cnn_codec",
    "cnn_base_ch",
    "cnn_bottleneck_ch",
    "cnn_num_res",
    "z1_concat_z2_ch",
    "layer1_cnn_base_ch",
    "layer1_cnn_num_res",
    "output_activation",
    "swin_codec",
}

METRIC_NAMES = [
    "loss",
    "loss_u2",
    "loss_img",
    "loss_l1",
    "mse_u2_teacher",
    "mse_x1",
    "psnr_x1",
    "ssim_x1",
    "mse_teacher",
    "psnr_teacher",
    "ssim_teacher",
    "mse_final",
    "psnr_final",
    "ssim_final",
    "delta_x1",
    "gap_teacher",
    "mse_u2_as_img",
    "psnr_u2_as_img",
    "z3_abs_mean",
    "q3_abs_mean",
    "fsq_mse",
]

VAL_ABLATION_METRICS = [
    "psnr_zero",
    "psnr_shuffle",
    "drop_zero",
    "drop_shuffle",
]

DISPLAY_METRICS = [
    "loss",
    "loss_u2",
    "loss_img",
    "loss_l1",
    "psnr_x1",
    "psnr_teacher",
    "psnr_final",
    "delta_x1",
    "gap_teacher",
    "mse_u2_teacher",
    "code_used",
    "code_entropy_bits",
    "code_perplexity",
    "code_usage_ratio",
    "psnr_zero",
    "psnr_shuffle",
    "drop_zero",
    "drop_shuffle",
]


def parse_fsq_levels(levels: str | list[int] | tuple[int, ...], d: int) -> list[int]:
    if isinstance(levels, str):
        parts = [p.strip() for p in levels.replace("x", ",").split(",") if p.strip()]
        parsed = [int(p) for p in parts]
    else:
        parsed = [int(v) for v in levels]
    if len(parsed) == 1:
        parsed = parsed * int(d)
    if len(parsed) != int(d):
        raise ValueError(f"expected one FSQ level or {d} levels, got {parsed}")
    if min(parsed) < 2:
        raise ValueError(f"FSQ levels must be >= 2, got {parsed}")
    return parsed


def fsq_level_name(levels: list[int]) -> str:
    return "l" + "x".join(str(v) for v in levels)


def vocab_size(levels: list[int]) -> int:
    out = 1
    for level in levels:
        out *= int(level)
    return int(out)


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def encode_tensor(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = module(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"encoder returned unsupported type {type(out)!r}")
    return out


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(bool(trainable))
    module.train(bool(trainable))


def trainable_state(module: nn.Module | None) -> str:
    if module is None:
        return "none"
    total = 0
    trainable = 0
    for param in module.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    if total == 0:
        return "no_params"
    if trainable == 0:
        return "frozen"
    if trainable == total:
        return "trainable"
    return f"partial_trainable({100.0 * trainable / float(total):.1f}%)"


class IFSQQuantizer(nn.Module):
    """Implicit finite scalar quantizer for tokenizer latents.

    This is intentionally separate from the older z2 ScalarFSQQuantizer: it has
    no center/scale calibration from teacher latents and no VQ auxiliary loss.
    """

    def __init__(self, levels: list[int] | tuple[int, ...], channels: int, use_pre_norm: bool = True) -> None:
        super().__init__()
        parsed = parse_fsq_levels(levels, int(channels))
        self.channels = int(channels)
        self.register_buffer("levels", torch.tensor(parsed, dtype=torch.long))
        multipliers: list[int] = []
        running = 1
        for level in reversed(parsed[1:]):
            running *= int(level)
            multipliers.append(running)
        multipliers = list(reversed(multipliers)) + [1]
        self.register_buffer("multipliers", torch.tensor(multipliers, dtype=torch.long))
        self.pre_norm = nn.GroupNorm(1, self.channels, affine=True) if bool(use_pre_norm) else nn.Identity()

    @property
    def vocab_size(self) -> int:
        return vocab_size([int(v) for v in self.levels.detach().cpu().tolist()])

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 4 or int(codes.shape[1]) != self.channels:
            raise ValueError(f"expected FSQ codes [B,{self.channels},H,W], got {tuple(codes.shape)}")
        multipliers = self.multipliers.to(device=codes.device, dtype=torch.long).view(1, self.channels, 1, 1)
        return (codes.long() * multipliers).sum(dim=1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.ndim != 4 or int(z.shape[1]) != self.channels:
            raise ValueError(f"expected z3 [B,{self.channels},H,W], got {tuple(z.shape)}")
        z_norm = torch.tanh(self.pre_norm(z))
        levels = self.levels.to(device=z.device, dtype=z_norm.dtype).view(1, self.channels, 1, 1)
        span = (levels - 1.0).clamp_min(1.0)
        positions = (z_norm + 1.0) * 0.5 * span
        codes_float = round_ste(positions).clamp_min(0.0).minimum(span)
        codes = codes_float.detach().long()
        q_hard = codes_float / span * 2.0 - 1.0
        q3 = z_norm + (q_hard - z_norm).detach()
        indices = self.codes_to_indices(codes)
        return {
            "z3_norm": z_norm,
            "q3": q3,
            "q3_hard": q_hard.detach(),
            "codes": codes,
            "idx3": indices,
            "fsq_mse": F.mse_loss(q_hard.detach().float(), z_norm.detach().float()),
        }


class CNNConditionEncoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base_ch: int, num_res: int) -> None:
        super().__init__()
        self.net = CNNAnalysisEncoder(base_ch=int(base_ch), bottleneck_ch=int(out_ch), num_res=int(num_res))
        self.net.stem = ConvNormAct(int(in_ch), int(base_ch), kernel=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Z1ConditionEncoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1),
            nn.SiLU(inplace=True),
        ]
        layers.extend(ResidualBlock(int(out_ch)) for _ in range(max(0, int(depth))))
        self.net = nn.Sequential(*layers)

    def forward(self, z1: torch.Tensor) -> torch.Tensor:
        return self.net(z1)


class Layer3FSQTokenizer(nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        super().__init__()
        self.args = args
        self.arch = str(args.arch)
        self.condition_mode = str(args.condition_mode)
        self.fsq_d = int(args.fsq_d)
        self.levels = parse_fsq_levels(args.fsq_levels, self.fsq_d)
        self.use_x1_cond = self.condition_mode in {"x1", "x1_only", "x1_z1"}
        self.use_z1_cond = self.condition_mode in {"z1", "z1_only", "x1_z1"}

        if self.arch == "cnn":
            self.e3 = CNNAnalysisEncoder(
                base_ch=int(args.e3_base_ch),
                bottleneck_ch=self.fsq_d,
                num_res=int(args.e3_num_res),
            )
            self.e3.stem = ConvNormAct(6, int(args.e3_base_ch), kernel=3, stride=1)
            self.x1_cond = (
                CNNConditionEncoder(3, int(args.x1_cond_ch), int(args.cond_base_ch), int(args.cond_num_res))
                if self.use_x1_cond
                else None
            )
            d3_in = self.fsq_d + (int(args.x1_cond_ch) if self.use_x1_cond else 0) + (int(args.z1_cond_ch) if self.use_z1_cond else 0)
            self.d3 = CNNBottleneckDecoder(
                base_ch=int(args.d3_base_ch),
                bottleneck_ch=d3_in,
                num_res=int(args.d3_num_res),
                output_activation="none",
            )
        elif self.arch == "swin":
            self.e3 = build_jscc_encoder(args, device, latent_ch=self.fsq_d, in_chans=6)
            self.x1_cond = (
                build_jscc_encoder(args, device, latent_ch=int(args.x1_cond_ch), in_chans=3)
                if self.use_x1_cond
                else None
            )
            d3_in = self.fsq_d + (int(args.x1_cond_ch) if self.use_x1_cond else 0) + (int(args.z1_cond_ch) if self.use_z1_cond else 0)
            self.d3 = build_jscc_decoder(args, device, latent_ch=d3_in)
        else:
            raise ValueError(f"unknown --arch {self.arch!r}")

        self.z1_cond = (
            Z1ConditionEncoder(int(args.latent_ch), int(args.z1_cond_ch), depth=int(args.z1_cond_depth))
            if self.use_z1_cond
            else None
        )
        self.quantizer = IFSQQuantizer(self.levels, channels=self.fsq_d, use_pre_norm=not bool(args.no_pre_norm))
        self.to(device)

    def encode(self, img: torch.Tensor, x1: torch.Tensor) -> dict[str, torch.Tensor]:
        e3_in = torch.cat([x1, img], dim=1)
        z3 = encode_tensor(self.e3, e3_in)
        q = self.quantizer(z3)
        q["z3"] = z3
        return q

    def condition(self, x1: torch.Tensor, z1: torch.Tensor) -> torch.Tensor | None:
        parts: list[torch.Tensor] = []
        if self.x1_cond is not None:
            parts.append(encode_tensor(self.x1_cond, x1))
        if self.z1_cond is not None:
            parts.append(self.z1_cond(z1))
        if not parts:
            return None
        return torch.cat(parts, dim=1)

    def decode(self, q3: torch.Tensor, x1: torch.Tensor, z1: torch.Tensor, combiner: OutputsCombiner) -> dict[str, torch.Tensor]:
        cond = self.condition(x1, z1)
        d3_in = q3 if cond is None else torch.cat([q3, cond], dim=1)
        u2_raw = self.d3(d3_in)
        u2_hat = u2_raw.clamp(0.0, 1.0)
        final = combiner(x1, u2_hat)
        return {
            "d3_in": d3_in,
            "u2_raw": u2_raw,
            "u2_hat": u2_hat,
            "final": final,
        }

    @staticmethod
    def shuffle_q3(q3: torch.Tensor) -> torch.Tensor:
        bsz, channels, h, w = q3.shape
        flat = q3.permute(0, 2, 3, 1).reshape(-1, channels)
        perm = torch.randperm(flat.shape[0], device=q3.device)
        return flat[perm].view(bsz, h, w, channels).permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        img: torch.Tensor,
        x1: torch.Tensor,
        z1: torch.Tensor,
        combiner: OutputsCombiner,
        *,
        q_mode: str = "normal",
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(img, x1)
        q3 = encoded["q3"]
        if q_mode == "zero":
            q3 = torch.zeros_like(q3)
        elif q_mode == "shuffle":
            q3 = self.shuffle_q3(q3)
        elif q_mode != "normal":
            raise ValueError(f"unknown q_mode {q_mode!r}")
        decoded = self.decode(q3, x1, z1, combiner)
        return {**encoded, **decoded, "q3_used": q3}


@dataclass
class TeacherBundle:
    arch: str
    module: ModuleType
    e1: nn.Module
    d1: nn.Module
    e2: nn.Module
    d2: nn.Module
    combiner: OutputsCombiner
    args: argparse.Namespace

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.arch == "cnn":
            return self.module.layer2_forward_cnn(self.e1, self.d1, self.e2, self.d2, self.combiner, imgs, self.args)
        return self.module.layer2_forward(self.e1, self.d1, self.e2, self.d2, self.combiner, imgs, self.args.variant)

    def eval(self) -> None:
        for module in (self.e1, self.d1, self.e2, self.d2, self.combiner):
            module.eval()


def load_teacher_checkpoint_for_args(args: argparse.Namespace) -> dict:
    if not args.layer2_ckpt:
        args.layer2_ckpt = DEFAULT_LAYER2_CKPTS[str(args.arch)]
    ckpt = jsccf_io.load_checkpoint(args.layer2_ckpt)
    if str(args.arch) == "cnn" and str(ckpt.get("stage", "")) != "layer2_v2_cnn":
        print(f"[warn] --arch cnn but checkpoint stage is {ckpt.get('stage')!r}", flush=True)
    if str(args.arch) == "swin" and "swin" not in str(ckpt.get("stage", "")) and "JSCC_encoder" not in str(ckpt.get("stage2_codec", {}).get("arch", "")):
        print(f"[warn] --arch swin but checkpoint stage is {ckpt.get('stage')!r}", flush=True)
    ckpt_args = ckpt.get("args", {})
    if not bool(args.ignore_ckpt_args):
        for key in TEACHER_ARG_KEYS:
            if key in ckpt_args:
                setattr(args, key, ckpt_args[key])
        if "variant" in ckpt:
            args.variant = ckpt["variant"]
    return ckpt


def build_teacher(args: argparse.Namespace, ckpt: dict, device: torch.device) -> TeacherBundle:
    if str(args.arch) == "cnn":
        stage2_module = load_script_module("jsccf_stage2_cnn", "train_stage2-cnn.py")
        if hasattr(stage2_module, "validate_args"):
            stage2_module.validate_args(args)
        e1, d1, e2, d2, combiner = stage2_module.build_layer2_cnn(args, device)
    else:
        stage2_module = load_script_module("jsccf_stage2_swin", "train_stage2-swin.py")
        e1, d1, e2, d2, combiner = stage2_module.build_stage2(args, device)

    jsccf_io.load_state(e1, ckpt["e1_state_dict"], "teacher_E1", strict=True)
    jsccf_io.load_state(d1, ckpt["d1_state_dict"], "teacher_D1", strict=True)
    jsccf_io.load_state(e2, ckpt["e2_state_dict"], "teacher_E2", strict=True)
    jsccf_io.load_state(d2, ckpt["d2_state_dict"], "teacher_D2", strict=True)
    jsccf_io.load_state(combiner, ckpt["combiner_state_dict"], "teacher_combiner", strict=True)
    for module in (e1, d1, e2, d2, combiner):
        set_trainable(module, False)
        module.eval()
    return TeacherBundle(str(args.arch), stage2_module, e1, d1, e2, d2, combiner, args)


def stage3_name(args: argparse.Namespace) -> str:
    levels = parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    return f"stage3_fsq_tokenizer_{args.arch}_d{int(args.fsq_d)}_{fsq_level_name(levels)}_{args.condition_mode}"


def display_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {k: metrics[k] for k in DISPLAY_METRICS if k in metrics}


def compute_losses(
    out: dict[str, torch.Tensor],
    teacher_out: dict[str, torch.Tensor],
    imgs: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    loss_u2 = F.mse_loss(out["u2_hat"].float(), teacher_out["u2"].detach().float())
    loss_img = recon_loss(out["final"], imgs)
    loss_l1 = F.l1_loss(out["u2_hat"].float(), teacher_out["u2"].detach().float())
    loss = float(args.lambda_u) * loss_u2 + float(args.lambda_img) * loss_img + float(args.lambda_l1) * loss_l1
    return {
        "loss": loss,
        "loss_u2": loss_u2,
        "loss_img": loss_img,
        "loss_l1": loss_l1,
    }


def update_code_hist(hist: torch.Tensor, idx3: torch.Tensor) -> None:
    counts = torch.bincount(idx3.detach().reshape(-1).cpu(), minlength=hist.numel()).float()
    hist += counts[: hist.numel()]


def update_metrics(
    m: dict,
    out: dict[str, torch.Tensor],
    teacher_out: dict[str, torch.Tensor],
    imgs: torch.Tensor,
    losses: dict[str, torch.Tensor],
) -> None:
    bsz = int(imgs.shape[0])
    for name, value in losses.items():
        m[name].update(float(value.detach().item()), bsz)

    psnr_x1 = batch_metric_mean(psnr_per_image(teacher_out["x1"], imgs))
    psnr_teacher = batch_metric_mean(psnr_per_image(teacher_out["final"], imgs))
    psnr_final = batch_metric_mean(psnr_per_image(out["final"], imgs))
    m["mse_u2_teacher"].update(float(F.mse_loss(out["u2_hat"].float(), teacher_out["u2"].detach().float()).item()), bsz)
    m["mse_x1"].update(batch_metric_mean(mse_per_image(teacher_out["x1"], imgs)), bsz)
    m["psnr_x1"].update(psnr_x1, bsz)
    m["ssim_x1"].update(batch_metric_mean(ssim_per_image(teacher_out["x1"], imgs)), bsz)
    m["mse_teacher"].update(batch_metric_mean(mse_per_image(teacher_out["final"], imgs)), bsz)
    m["psnr_teacher"].update(psnr_teacher, bsz)
    m["ssim_teacher"].update(batch_metric_mean(ssim_per_image(teacher_out["final"], imgs)), bsz)
    m["mse_final"].update(batch_metric_mean(mse_per_image(out["final"], imgs)), bsz)
    m["psnr_final"].update(psnr_final, bsz)
    m["ssim_final"].update(batch_metric_mean(ssim_per_image(out["final"], imgs)), bsz)
    m["delta_x1"].update(psnr_final - psnr_x1, bsz)
    m["gap_teacher"].update(psnr_teacher - psnr_final, bsz)
    m["mse_u2_as_img"].update(batch_metric_mean(mse_per_image(out["u2_hat"], imgs)), bsz)
    m["psnr_u2_as_img"].update(batch_metric_mean(psnr_per_image(out["u2_hat"], imgs)), bsz)
    m["z3_abs_mean"].update(float(out["z3"].detach().float().abs().mean().item()), bsz)
    m["q3_abs_mean"].update(float(out["q3_hard"].detach().float().abs().mean().item()), bsz)
    m["fsq_mse"].update(float(out["fsq_mse"].detach().item()), bsz)


@torch.no_grad()
def update_ablation_metrics(
    m: dict,
    tokenizer: Layer3FSQTokenizer,
    out: dict[str, torch.Tensor],
    teacher_out: dict[str, torch.Tensor],
    imgs: torch.Tensor,
    combiner: OutputsCombiner,
) -> None:
    bsz = int(imgs.shape[0])
    x1 = teacher_out["x1"]
    z1 = teacher_out["z1"]
    zero = tokenizer.decode(torch.zeros_like(out["q3"]), x1, z1, combiner)
    shuffle = tokenizer.decode(tokenizer.shuffle_q3(out["q3"]), x1, z1, combiner)
    psnr_final = batch_metric_mean(psnr_per_image(out["final"], imgs))
    psnr_zero = batch_metric_mean(psnr_per_image(zero["final"], imgs))
    psnr_shuffle = batch_metric_mean(psnr_per_image(shuffle["final"], imgs))
    m["psnr_zero"].update(psnr_zero, bsz)
    m["psnr_shuffle"].update(psnr_shuffle, bsz)
    m["drop_zero"].update(psnr_final - psnr_zero, bsz)
    m["drop_shuffle"].update(psnr_final - psnr_shuffle, bsz)


def finalize_metrics(m: dict, hist: torch.Tensor, args: argparse.Namespace) -> dict[str, float]:
    metrics = averaged(m)
    total = float(hist.sum().item())
    levels = parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    vocab = vocab_size(levels)
    if total > 0.0:
        probs = (hist / total).clamp_min(1e-12)
        active = hist > 0
        entropy_nats = float(-(probs[active] * probs[active].log()).sum().item())
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        metrics.update(
            {
                "code_used": float(active.sum().item()),
                "code_usage_ratio": float(active.float().mean().item()),
                "code_entropy_bits": entropy_nats / math.log(2.0),
                "code_perplexity": math.exp(entropy_nats),
                "code_top1_frac": float(hist.max().item() / total),
                "idx_min": float(active_idx.min().item()) if active_idx.numel() else 0.0,
                "idx_max": float(active_idx.max().item()) if active_idx.numel() else 0.0,
            }
        )
    else:
        metrics.update(
            {
                "code_used": 0.0,
                "code_usage_ratio": 0.0,
                "code_entropy_bits": 0.0,
                "code_perplexity": 0.0,
                "code_top1_frac": 0.0,
                "idx_min": 0.0,
                "idx_max": 0.0,
            }
        )
    bits_per_token = int(math.ceil(math.log2(float(vocab))))
    metrics["vocab_size"] = float(vocab)
    metrics["fixed_bits_per_token"] = float(bits_per_token)
    metrics["fixed_bits_per_image"] = float(bits_per_token * int(args.latent_h) * int(args.latent_w))
    return metrics


@torch.no_grad()
def validate(
    loader,
    tokenizer: Layer3FSQTokenizer,
    teacher: TeacherBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    tokenizer.eval()
    teacher.eval()
    names = METRIC_NAMES + (VAL_ABLATION_METRICS if bool(args.val_ablation) else [])
    m = meters(names)
    hist = torch.zeros(vocab_size(parse_fsq_levels(args.fsq_levels, int(args.fsq_d))), dtype=torch.float32)
    for batch_idx, (imgs, _labels) in enumerate(loader, start=1):
        if int(args.max_val_batches) > 0 and batch_idx > int(args.max_val_batches):
            break
        imgs = imgs.to(device, non_blocking=True)
        teacher_out = teacher.forward(imgs)
        out = tokenizer(imgs, teacher_out["x1"], teacher_out["z1"], teacher.combiner)
        losses = compute_losses(out, teacher_out, imgs, args)
        update_metrics(m, out, teacher_out, imgs, losses)
        update_code_hist(hist, out["idx3"])
        if bool(args.val_ablation):
            update_ablation_metrics(m, tokenizer, out, teacher_out, imgs, teacher.combiner)
    return finalize_metrics(m, hist, args)


def print_tokenizer_header(
    args: argparse.Namespace,
    tokenizer: Layer3FSQTokenizer,
    teacher: TeacherBundle,
    train_n: int,
    val_n: int,
) -> None:
    levels = parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    vocab = vocab_size(levels)
    bits_per_token = int(math.ceil(math.log2(float(vocab))))
    fixed_bits = bits_per_token * int(args.latent_h) * int(args.latent_w)
    print(f"=== Stage 3 | FSQ-tokenizer for u2 | {args.arch} ===", flush=True)
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"save_dir={resolve_path(args.save_dir)}", flush=True)
    print("实验设计", flush=True)
    print("  model=Layer3 FSQ tokenizer for u2; not z2 compression; AR/diffusion disabled", flush=True)
    print(
        f"  teacher_arch={args.arch} teacher_layer2_ckpt={resolve_path(args.layer2_ckpt)} "
        f"variant={args.variant}",
        flush=True,
    )
    print(
        f"  tokenizer={args.arch} E3 input=concat(x1,img) [B,6,256,256] "
        f"z3/q3=[B,{int(args.fsq_d)},{int(args.latent_h)},{int(args.latent_w)}] "
        f"idx3=[B,{int(args.latent_h)},{int(args.latent_w)}]",
        flush=True,
    )
    print(
        f"  fsq_levels={levels} vocab={vocab} fixed_bits/token={bits_per_token} "
        f"fixed_bits/image={fixed_bits}",
        flush=True,
    )
    print(f"  condition_mode={args.condition_mode} receiver_known=x1,z1", flush=True)
    print("loss设计", flush=True)
    print(
        f"  L={float(args.lambda_u):g}*MSE(u2_hat,u2_teacher)"
        f"+{float(args.lambda_img):g}*MSE(combiner(x1,u2_hat),img)"
        f"+{float(args.lambda_l1):g}*L1(u2_hat,u2_teacher)",
        flush=True,
    )
    print("  FSQ has no codebook/commitment loss; pre_norm is trainable per-sample GroupNorm unless disabled", flush=True)
    print("模块选择", flush=True)
    print(
        "  teacher=frozen "
        f"E1={trainable_state(teacher.e1)} D1={trainable_state(teacher.d1)} "
        f"E2={trainable_state(teacher.e2)} D2={trainable_state(teacher.d2)} "
        f"combiner={trainable_state(teacher.combiner)}",
        flush=True,
    )
    print(
        "  tokenizer=trainable "
        f"E3={tokenizer.e3.__class__.__name__} "
        f"FSQ=IFSQQuantizer(d={int(args.fsq_d)},levels={levels}) "
        f"D3={tokenizer.d3.__class__.__name__} "
        f"x1_cond={'on' if tokenizer.x1_cond is not None else 'off'} "
        f"z1_cond={'on' if tokenizer.z1_cond is not None else 'off'}",
        flush=True,
    )
    print(
        f"epochs={args.epochs} train={train_n} valid={val_n} "
        f"batch={args.batch_size} test_batch={args.test_batch} "
        f"lr={float(args.lr):g} weight_decay={float(args.weight_decay):g}",
        flush=True,
    )


def save_tokenizer_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    tokenizer: Layer3FSQTokenizer,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    levels = parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    payload = {
        "route": getattr(jsccf_io, "ROUTE", "TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256"),
        "stage": "stage3_fsq_tokenizer_u2",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "version": str(getattr(args, "version", "")),
        "tokenizer_state_dict": tokenizer.state_dict(),
        "teacher_layer2_ckpt": str(args.layer2_ckpt),
        "tokenizer": {
            "arch": str(args.arch),
            "condition_mode": str(args.condition_mode),
            "fsq_d": int(args.fsq_d),
            "fsq_levels": levels,
            "vocab_size": vocab_size(levels),
            "idx3_shape": [int(args.latent_h), int(args.latent_w)],
            "q3_shape": [int(args.fsq_d), int(args.latent_h), int(args.latent_w)],
            "fixed_bits_per_token": int(math.ceil(math.log2(float(vocab_size(levels))))),
            "fixed_bits_per_image": int(math.ceil(math.log2(float(vocab_size(levels))))) * int(args.latent_h) * int(args.latent_w),
        },
        "latent": {
            "z1": [int(args.latent_ch), int(args.latent_h), int(args.latent_w)],
            "z3": [int(args.fsq_d), int(args.latent_h), int(args.latent_w)],
        },
    }
    torch.save(payload, out)
    print(f"saved checkpoint: {out}", flush=True)


def train(args: argparse.Namespace, teacher_ckpt: dict) -> None:
    seed_everything(int(args.seed))
    cfg = jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = get_loader(cfg)
    teacher = build_teacher(args, teacher_ckpt, cfg.device)
    tokenizer = Layer3FSQTokenizer(args, cfg.device)
    opt = optim.AdamW(tokenizer.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    print_tokenizer_header(args, tokenizer, teacher, len(train_loader.dataset), len(val_loader.dataset))

    if bool(args.eval_init_only):
        val_metrics = validate(val_loader, tokenizer, teacher, args, cfg.device)
        print(f"[stage3-fsq init val] {display_metrics(val_metrics)} score=psnr_final", flush=True)
        out = Path(resolve_path(args.save_dir)) / f"{stage3_name(args)}_jscc_f_{jsccf_io.safe_artifact_name(args.version)}_init_eval.json"
        write_json(out, {"args": {k: v for k, v in vars(args).items() if not k.startswith("_")}, "metrics": val_metrics})
        print(f"[stage3-fsq init val] wrote {out}", flush=True)
        return

    best = -1.0
    metrics: dict[str, float] = {}
    for epoch in range(1, int(args.epochs) + 1):
        tokenizer.train()
        teacher.eval()
        m = meters(METRIC_NAMES)
        hist = torch.zeros(vocab_size(parse_fsq_levels(args.fsq_levels, int(args.fsq_d))), dtype=torch.float32)
        t0 = time.time()
        for batch_idx, (imgs, _labels) in enumerate(train_loader, start=1):
            if int(args.max_train_batches) > 0 and batch_idx > int(args.max_train_batches):
                break
            imgs = imgs.to(cfg.device, non_blocking=True)
            with torch.no_grad():
                teacher_out = teacher.forward(imgs)
            out = tokenizer(imgs, teacher_out["x1"], teacher_out["z1"], teacher.combiner)
            losses = compute_losses(out, teacher_out, imgs, args)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), float(args.grad_clip_norm))
            opt.step()
            update_metrics(m, out, teacher_out, imgs, losses)
            update_code_hist(hist, out["idx3"])

        metrics = finalize_metrics(m, hist, args)
        print(
            f"[stage3-fsq train {epoch:03d}/{int(args.epochs):03d}] "
            f"{display_metrics(metrics)} time={time.time() - t0:.1f}s",
            flush=True,
        )
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, tokenizer, teacher, args, cfg.device)
            score = float(val_metrics["psnr_final"])
            print(f"[stage3-fsq val {epoch:03d}] {display_metrics(val_metrics)} score=psnr_final", flush=True)
            if score > best:
                best = score
                save_tokenizer_checkpoint(
                    jsccf_io.ckpt_path(args, stage3_name(args), "best"),
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    tokenizer=tokenizer,
                )
        if should_save_latest(args, epoch):
            save_tokenizer_checkpoint(
                jsccf_io.ckpt_path(args, stage3_name(args), "latest"),
                epoch=epoch,
                args=args,
                metrics=metrics,
                tokenizer=tokenizer,
            )

    save_tokenizer_checkpoint(
        jsccf_io.ckpt_path(args, stage3_name(args), "latest"),
        epoch=int(args.epochs),
        args=args,
        metrics=metrics,
        tokenizer=tokenizer,
    )


@torch.no_grad()
def smoke_shapes(args: argparse.Namespace, teacher_ckpt: dict) -> None:
    seed_everything(int(args.seed))
    device = torch.device("cuda:0" if (not bool(args.cpu)) and torch.cuda.is_available() else "cpu")
    teacher = build_teacher(args, teacher_ckpt, device)
    tokenizer = Layer3FSQTokenizer(args, device)
    imgs = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    teacher_out = teacher.forward(imgs)
    out = tokenizer(imgs, teacher_out["x1"], teacher_out["z1"], teacher.combiner)
    expected_q3 = (int(args.smoke_batch_size), int(args.fsq_d), int(args.latent_h), int(args.latent_w))
    expected_idx = (int(args.smoke_batch_size), int(args.latent_h), int(args.latent_w))
    expected_u2 = (int(args.smoke_batch_size), 3, 256, 256)
    print(
        f"[smoke] arch={args.arch} x1={tuple(teacher_out['x1'].shape)} z1={tuple(teacher_out['z1'].shape)} "
        f"q3={tuple(out['q3'].shape)} idx3={tuple(out['idx3'].shape)} "
        f"u2_hat={tuple(out['u2_hat'].shape)} final={tuple(out['final'].shape)}",
        flush=True,
    )
    if tuple(out["q3"].shape) != expected_q3:
        raise RuntimeError(f"expected q3 {expected_q3}, got {tuple(out['q3'].shape)}")
    if tuple(out["idx3"].shape) != expected_idx:
        raise RuntimeError(f"expected idx3 {expected_idx}, got {tuple(out['idx3'].shape)}")
    if tuple(out["u2_hat"].shape) != expected_u2 or tuple(out["final"].shape) != expected_u2:
        raise RuntimeError(f"expected image outputs {expected_u2}, got u2={tuple(out['u2_hat'].shape)} final={tuple(out['final'].shape)}")


def apply_preset(args: argparse.Namespace) -> None:
    if str(args.preset) == "custom":
        return
    d, levels = PRESETS[str(args.preset)]
    args.fsq_d = int(d)
    args.fsq_levels = levels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--arch", type=str, default="swin", choices=["cnn", "swin"], help="Layer3 tokenizer architecture and default Layer2 teacher family.")
    p.add_argument("--version", type=str, default="fsq-tokenizer-t2")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--layer2-ckpt", type=str, default="", help="Frozen Stage2 teacher checkpoint. Default is chosen from --arch.")
    p.add_argument("--ignore-ckpt-args", action="store_true", help="Do not copy teacher architecture fields from the Stage2 checkpoint args.")

    p.add_argument("--preset", type=str, default="custom", choices=["custom", "t0", "t1", "t2", "t3", "t4"])
    p.add_argument("--fsq-d", type=int, default=3)
    p.add_argument("--fsq-levels", type=str, default="16,16,16")
    p.add_argument("--no-pre-norm", action="store_true", help="Disable tokenizer pre_norm before tanh+FSQ.")
    p.add_argument("--condition-mode", type=str, default="none", choices=["x1_z1", "x1_only", "x1", "z1_only", "z1", "none"])
    p.add_argument("--x1-cond-ch", type=int, default=16)
    p.add_argument("--z1-cond-ch", type=int, default=16)
    p.add_argument("--z1-cond-depth", type=int, default=2)

    p.add_argument("--latent-ch", type=int, default=16)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--lambda-u", type=float, default=1.0)
    p.add_argument("--lambda-img", type=float, default=1.0)
    p.add_argument("--lambda-l1", type=float, default=0.1)

    p.add_argument("--e3-base-ch", type=int, default=16)
    p.add_argument("--d3-base-ch", type=int, default=16)
    p.add_argument("--cond-base-ch", type=int, default=16)
    p.add_argument("--e3-num-res", type=int, default=2)
    p.add_argument("--d3-num-res", type=int, default=2)
    p.add_argument("--cond-num-res", type=int, default=2)

    p.add_argument("--variant", type=str, default="combiner", choices=["combiner", "no_combiner", "residual_input"])
    p.add_argument("--cnn-codec", type=str, default="no_compressor", choices=["compressor", "no_compressor", "z1_concat"])
    p.add_argument("--cnn-base-ch", type=int, default=20)
    p.add_argument("--cnn-bottleneck-ch", type=int, default=16)
    p.add_argument("--cnn-num-res", type=int, default=2)
    p.add_argument("--z1-concat-z2-ch", type=int, default=20)
    p.add_argument("--layer1-cnn-base-ch", type=int, default=16)
    p.add_argument("--layer1-cnn-num-res", type=int, default=2)
    p.add_argument("--output-activation", type=str, default="none", choices=["none", "sigmoid", "tanh"])
    p.add_argument("--swin-codec", type=str, default="no_compressor", choices=["no_compressor"])

    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--no-val-ablation", dest="val_ablation", action="store_false", help="Skip q3=0 and q3-shuffle validation ablations.")
    p.set_defaults(val_ablation=True)
    p.add_argument("--eval-init-only", action="store_true")
    p.add_argument("--smoke-shapes", action="store_true")
    p.add_argument("--smoke-batch-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.stage = "stage3_fsq_tokenizer"
    apply_preset(args)
    parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(
            Path(resolve_path(args.save_dir))
            / f"{stage3_name(args)}_jscc_f_{jsccf_io.safe_artifact_name(args.version)}.log"
        )
    setup_log_file(args.log_file)
    teacher_ckpt = load_teacher_checkpoint_for_args(args)
    check_jsccf_args(args)
    write_json(
        Path(resolve_path(args.save_dir)) / f"{stage3_name(args)}_jscc_f_{jsccf_io.safe_artifact_name(args.version)}_args.json",
        {k: v for k, v in vars(args).items() if not k.startswith("_")},
    )
    if bool(args.smoke_shapes):
        smoke_shapes(args, teacher_ckpt)
        return
    train(args, teacher_ckpt)


if __name__ == "__main__":
    main()
