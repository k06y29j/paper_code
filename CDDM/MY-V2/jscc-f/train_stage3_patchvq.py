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
    JSCCFConfig,
    averaged,
    batch_metric_mean,
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
import model as jsccf_model
from model import OutputsCombiner, build_jscc_decoder, build_jscc_encoder
from test_ed import (
    CNNAnalysisEncoder,
    CNNBottleneckDecoder,
    ConvNormAct,
    ImageVectorQuantizer,
    ImageVectorSimVQQuantizer,
    ResidualBlock,
    init_linear_identity,
    nearest_codebook_2d,
    soft_code_usage_kld_loss,
)


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


PATCHVQ_PRESETS: dict[str, tuple[int, int]] = {
    "t0": (8, 64),
    "t1": (16, 256),
    "t2": (16, 512),
    "t3": (32, 512),
    "t4": (32, 1024),
}

DEFAULT_LAYER2_CKPTS = {
    "cnn": "MY-V2/jscc-f/checkpoints/jscc_f_cnn-stage2-no_compressor-gpu2_layer2_v2_no_compressor_combiner_best.pth",
    "swin": "MY-V2/jscc-f/checkpoints/jscc_f_swin320_layer2_swin_no_compressor_combiner_best.pth",
}
DEFAULT_SAVE_DIR = str(CDDM_ROOT / "MY-V2" / "jscc-f" / "checkpoints-patchvq")

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
    "loss_vq",
    "loss_usage",
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
    "vq_mse",
    "codebook_loss",
    "commit_loss",
    "usage_kld_loss",
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
    "loss_vq",
    "loss_usage",
    "psnr_x1",
    "psnr_teacher",
    "psnr_final",
    "delta_x1",
    "gap_teacher",
    "mse_u2_teacher",
    "vq_mse",
    "codebook_loss",
    "commit_loss",
    "usage_kld_loss",
    "code_used",
    "code_entropy_bits",
    "code_perplexity",
    "code_usage_ratio",
    "code_top1_frac",
    "psnr_zero",
    "psnr_shuffle",
    "drop_zero",
    "drop_shuffle",
]


def normalize_image_args(args: argparse.Namespace) -> None:
    if int(getattr(args, "image_height", 0)) <= 0:
        args.image_height = int(args.image_size)
    if int(getattr(args, "image_width", 0)) <= 0:
        args.image_width = int(args.image_size)


def build_runtime_config(
    args: argparse.Namespace,
    batch_size: int | None = None,
    encoder_in_chans: int = 3,
) -> JSCCFConfig:
    h, w = int(args.image_height), int(args.image_width)
    return JSCCFConfig(
        C=int(getattr(args, "latent_ch", 16)),
        batch_size=int(args.batch_size if batch_size is None else batch_size),
        test_batch=int(args.test_batch),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.val_num_workers),
        train_data_dir=resolve_path(os.path.join(args.data_dir, "DIV2K_train_HR")),
        test_data_dir=resolve_path(os.path.join(args.data_dir, "DIV2K_valid_HR")),
        CUDA=(not bool(args.cpu)),
        encoder_in_chans=int(encoder_in_chans),
        image_dims=(3, h, w),
        persistent_workers=bool(args.persistent_workers),
    )


def patch_runtime_config() -> None:
    jsccf_io.build_config = build_runtime_config
    jsccf_model.jsccf_io.build_config = build_runtime_config


def validate_patchvq_args(args: argparse.Namespace) -> None:
    normalize_image_args(args)
    if int(args.c1_ch) != int(args.latent_ch):
        raise ValueError("JSCC-f layer1 uses --latent-ch as z1 channels; keep --c1-ch equal to --latent-ch.")
    if int(args.image_height) <= 0 or int(args.image_width) <= 0:
        raise ValueError("--image-height/--image-width must be positive")
    if int(args.image_height) % 16 != 0 or int(args.image_width) % 16 != 0:
        raise ValueError("current JSCC-f encoders require image height/width divisible by 16")
    expected_h = int(args.image_height) // 16
    expected_w = int(args.image_width) // 16
    if int(args.latent_h) != expected_h or int(args.latent_w) != expected_w:
        raise ValueError(
            f"--latent-h/--latent-w must match image/16 for this Stage3 path: "
            f"expected {expected_h}x{expected_w}, got {int(args.latent_h)}x{int(args.latent_w)}"
        )
    if int(args.token_ch) <= 0:
        raise ValueError("--token-ch must be positive")
    if int(args.vq_k) < 2:
        raise ValueError("--vq-k must be >= 2")
    if float(args.beta) < 0.0:
        raise ValueError("--beta must be non-negative")
    if int(args.vq_chunk_size) <= 0:
        raise ValueError("--vq-chunk-size must be positive")
    if float(args.usage_kld_tau) <= 0.0:
        raise ValueError("--usage-kld-tau must be positive")


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


def parameter_state(param: torch.Tensor) -> str:
    return "trainable" if bool(getattr(param, "requires_grad", False)) else "frozen"


def encode_tensor(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = module(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"encoder returned unsupported type {type(out)!r}")
    return out


def quantizer_codebook(quantizer: nn.Module, *, effective: bool) -> torch.Tensor:
    if effective and hasattr(quantizer, "effective_codebook"):
        return quantizer.effective_codebook().view(int(quantizer.num_codes), int(quantizer.channels))
    if hasattr(quantizer, "codebook"):
        return quantizer.codebook.view(int(quantizer.num_codes), int(quantizer.channels))
    raise TypeError(f"unsupported quantizer type {type(quantizer)!r}")


def set_quantizer_codebook(quantizer: nn.Module, vectors: torch.Tensor) -> None:
    k = int(quantizer.num_codes)
    c = int(quantizer.channels)
    if vectors.ndim != 2 or tuple(vectors.shape) != (k, c):
        raise ValueError(f"expected vectors [{k},{c}], got {tuple(vectors.shape)}")
    with torch.no_grad():
        quantizer.codebook.copy_(vectors.to(device=quantizer.codebook.device, dtype=quantizer.codebook.dtype).view(k, c, 1, 1))
        if hasattr(quantizer, "embedding_proj"):
            init_linear_identity(quantizer.embedding_proj)


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


class Layer3PatchVQTokenizer(nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        super().__init__()
        self.args = args
        self.arch = str(args.arch)
        self.condition_mode = str(args.condition_mode)
        self.token_ch = int(args.token_ch)
        self.vq_k = int(args.vq_k)
        self.use_x1_cond = self.condition_mode in {"x1", "x1_only", "x1_z1"}
        self.use_z1_cond = self.condition_mode in {"z1", "z1_only", "x1_z1"}

        if self.arch == "cnn":
            self.e3 = CNNAnalysisEncoder(
                base_ch=int(args.e3_base_ch),
                bottleneck_ch=self.token_ch,
                num_res=int(args.e3_num_res),
            )
            self.e3.stem = ConvNormAct(6, int(args.e3_base_ch), kernel=3, stride=1)
            self.x1_cond = (
                CNNConditionEncoder(3, int(args.x1_cond_ch), int(args.cond_base_ch), int(args.cond_num_res))
                if self.use_x1_cond
                else None
            )
            d3_in = self.token_ch + (int(args.x1_cond_ch) if self.use_x1_cond else 0) + (int(args.z1_cond_ch) if self.use_z1_cond else 0)
            self.d3 = CNNBottleneckDecoder(
                base_ch=int(args.d3_base_ch),
                bottleneck_ch=d3_in,
                num_res=int(args.d3_num_res),
                output_activation="none",
            )
        elif self.arch == "swin":
            self.e3 = build_jscc_encoder(args, device, latent_ch=self.token_ch, in_chans=6)
            self.x1_cond = (
                build_jscc_encoder(args, device, latent_ch=int(args.x1_cond_ch), in_chans=3)
                if self.use_x1_cond
                else None
            )
            d3_in = self.token_ch + (int(args.x1_cond_ch) if self.use_x1_cond else 0) + (int(args.z1_cond_ch) if self.use_z1_cond else 0)
            self.d3 = build_jscc_decoder(args, device, latent_ch=d3_in)
        else:
            raise ValueError(f"unknown --arch {self.arch!r}")

        self.z1_cond = (
            Z1ConditionEncoder(int(args.latent_ch), int(args.z1_cond_ch), depth=int(args.z1_cond_depth))
            if self.use_z1_cond
            else None
        )
        if str(args.quantizer) == "vq":
            self.quantizer = ImageVectorQuantizer(
                num_codes=self.vq_k,
                channels=self.token_ch,
                beta=float(args.beta),
                chunk_size=int(args.vq_chunk_size),
            )
        elif str(args.quantizer) == "simvq":
            self.quantizer = ImageVectorSimVQQuantizer(
                num_codes=self.vq_k,
                channels=self.token_ch,
                beta=float(args.beta),
                chunk_size=int(args.vq_chunk_size),
                freeze_codebook=not bool(args.simvq_train_codebook),
                proj_bias=not bool(args.simvq_no_proj_bias),
            )
        else:
            raise ValueError(f"unknown --quantizer {args.quantizer!r}")
        self.to(device)

    def encode(self, img: torch.Tensor, x1: torch.Tensor) -> dict[str, torch.Tensor]:
        e3_in = torch.cat([x1, img], dim=1)
        z3 = encode_tensor(self.e3, e3_in)
        q3, q3_hard, idx3, stats = self.quantizer(z3)
        usage_kld = z3.new_zeros(())
        if float(self.args.lambda_usage_kld) > 0.0:
            flat = z3.permute(0, 2, 3, 1).contiguous().view(-1, self.token_ch)
            codebook = quantizer_codebook(self.quantizer, effective=True).detach()
            usage_kld = soft_code_usage_kld_loss(flat, codebook, float(self.args.usage_kld_tau), int(self.args.vq_chunk_size))
        return {
            "z3": z3,
            "q3": q3,
            "q3_hard": q3_hard,
            "idx3": idx3,
            "usage_kld_loss": usage_kld,
            **stats,
        }

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
        stage2_module = load_script_module("jsccf_stage2_cnn_patchvq", "train_stage2-cnn.py")
        if hasattr(stage2_module, "validate_args"):
            stage2_module.validate_args(args)
        e1, d1, e2, d2, combiner = stage2_module.build_layer2_cnn(args, device)
    else:
        stage2_module = load_script_module("jsccf_stage2_swin_patchvq", "train_stage2-swin.py")
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
    return f"stage3_patchvq_tokenizer_{args.arch}_{args.quantizer}_k{int(args.vq_k)}_c{int(args.token_ch)}_{args.condition_mode}"


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
    loss_vq = out["vq_loss"]
    loss_usage = out["usage_kld_loss"]
    loss = (
        float(args.lambda_u) * loss_u2
        + float(args.lambda_img) * loss_img
        + float(args.lambda_l1) * loss_l1
        + float(args.lambda_vq) * loss_vq
        + float(args.lambda_usage_kld) * loss_usage
    )
    return {
        "loss": loss,
        "loss_u2": loss_u2,
        "loss_img": loss_img,
        "loss_l1": loss_l1,
        "loss_vq": loss_vq,
        "loss_usage": loss_usage,
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
    m["vq_mse"].update(float(out["vq_mse"].detach().item()), bsz)
    m["codebook_loss"].update(float(out["codebook_loss"].detach().item()), bsz)
    m["commit_loss"].update(float(out["commit_loss"].detach().item()), bsz)
    m["usage_kld_loss"].update(float(out["usage_kld_loss"].detach().item()), bsz)


@torch.no_grad()
def update_ablation_metrics(
    m: dict,
    tokenizer: Layer3PatchVQTokenizer,
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
    bits_per_token = int(math.ceil(math.log2(float(int(args.vq_k)))))
    metrics["vocab_size"] = float(int(args.vq_k))
    metrics["fixed_bits_per_token"] = float(bits_per_token)
    metrics["fixed_bits_per_image"] = float(bits_per_token * int(args.latent_h) * int(args.latent_w))
    return metrics


@torch.no_grad()
def validate(
    loader,
    tokenizer: Layer3PatchVQTokenizer,
    teacher: TeacherBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    tokenizer.eval()
    teacher.eval()
    names = METRIC_NAMES + (VAL_ABLATION_METRICS if bool(args.val_ablation) else [])
    m = meters(names)
    hist = torch.zeros(int(args.vq_k), dtype=torch.float32)
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


@torch.no_grad()
def collect_z3_vectors(
    loader,
    tokenizer: Layer3PatchVQTokenizer,
    teacher: TeacherBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    tokenizer.eval()
    teacher.eval()
    chunks: list[torch.Tensor] = []
    total = 0
    max_batches = int(args.init_max_batches)
    max_tokens = int(args.init_max_tokens)
    for batch_idx, (imgs, _labels) in enumerate(loader, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        teacher_out = teacher.forward(imgs)
        z3 = encode_tensor(tokenizer.e3, torch.cat([teacher_out["x1"], imgs], dim=1))
        flat = z3.permute(0, 2, 3, 1).contiguous().view(-1, int(args.token_ch))
        if flat.shape[0] > 0:
            remaining = max_tokens - total
            if remaining <= 0:
                break
            if flat.shape[0] > remaining:
                perm = torch.randperm(flat.shape[0], device=flat.device)[:remaining]
                flat = flat[perm]
            chunks.append(flat.detach().cpu())
            total += int(flat.shape[0])
    if not chunks:
        raise RuntimeError("no z3 vectors collected for codebook initialization")
    return torch.cat(chunks, dim=0)


def kmeans_vectors(samples: torch.Tensor, k: int, iters: int, chunk_size: int) -> torch.Tensor:
    if samples.ndim != 2:
        raise ValueError(f"expected samples [N,C], got {tuple(samples.shape)}")
    n, c = int(samples.shape[0]), int(samples.shape[1])
    if n < 1:
        raise ValueError("empty samples")
    device = samples.device
    idx = torch.randint(0, n, (int(k),), device=device)
    centers = samples[idx].contiguous()
    for _ in range(max(1, int(iters))):
        _q, labels = nearest_codebook_2d(samples, centers, int(chunk_size))
        counts = torch.bincount(labels, minlength=int(k)).to(device=device, dtype=samples.dtype)
        sums = torch.zeros(int(k), c, device=device, dtype=samples.dtype)
        sums.index_add_(0, labels.to(device), samples)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        if bool((~nonempty).any()):
            repl = torch.randint(0, n, (int((~nonempty).sum().item()),), device=device)
            centers[~nonempty] = samples[repl]
    return centers


@torch.no_grad()
def initialize_codebook_if_requested(
    train_loader,
    tokenizer: Layer3PatchVQTokenizer,
    teacher: TeacherBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    method = str(args.init_codebook)
    if method == "none":
        return
    vectors = collect_z3_vectors(train_loader, tokenizer, teacher, args, device).to(device=device, dtype=torch.float32)
    k = int(args.vq_k)
    if method == "random_samples":
        idx = torch.randint(0, int(vectors.shape[0]), (k,), device=device)
        centers = vectors[idx]
    elif method == "kmeans":
        centers = kmeans_vectors(vectors, k, int(args.init_kmeans_iters), int(args.vq_chunk_size))
    else:
        raise ValueError(f"unknown --init-codebook {method!r}")
    set_quantizer_codebook(tokenizer.quantizer, centers)
    print(
        f"[patchvq init] method={method} samples={int(vectors.shape[0])} "
        f"k={k} c={int(args.token_ch)} std={float(centers.std().item()):.6g}",
        flush=True,
    )


def print_tokenizer_header(
    args: argparse.Namespace,
    tokenizer: Layer3PatchVQTokenizer,
    teacher: TeacherBundle,
    train_n: int,
    val_n: int,
) -> None:
    bits_per_token = int(math.ceil(math.log2(float(int(args.vq_k)))))
    fixed_bits = bits_per_token * int(args.latent_h) * int(args.latent_w)
    h, w = int(args.image_height), int(args.image_width)
    print(f"=== Stage 3 | PatchVQ-tokenizer for u2 | {args.arch} ===", flush=True)
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"save_dir={resolve_path(args.save_dir)}", flush=True)
    print("实验设计", flush=True)
    print("  model=Layer3 PatchVQ tokenizer for u2; image-vector quantization; AR/diffusion disabled", flush=True)
    print(
        f"  teacher_arch={args.arch} teacher_layer2_ckpt={resolve_path(args.layer2_ckpt)} "
        f"variant={args.variant}",
        flush=True,
    )
    print(
        f"  tokenizer={args.arch} E3 input=concat(x1,img) [B,6,{h},{w}] "
        f"z3/q3=[B,{int(args.token_ch)},{int(args.latent_h)},{int(args.latent_w)}] "
        f"idx3=[B,{int(args.latent_h)},{int(args.latent_w)}]",
        flush=True,
    )
    print(
        f"  quantizer={args.quantizer} embedding=[{int(args.vq_k)},{int(args.token_ch)},1,1] "
        f"vocab={int(args.vq_k)} fixed_bits/token={bits_per_token} fixed_bits/image={fixed_bits}",
        flush=True,
    )
    print(f"  condition_mode={args.condition_mode} receiver_known=x1,z1", flush=True)
    print("loss设计", flush=True)
    print(
        f"  L={float(args.lambda_u):g}*MSE(u2_hat,u2_teacher)"
        f"+{float(args.lambda_img):g}*MSE(combiner(x1,u2_hat),img)"
        f"+{float(args.lambda_l1):g}*L1(u2_hat,u2_teacher)"
        f"+{float(args.lambda_vq):g}*VQ(codebook+beta*commit)"
        f"+{float(args.lambda_usage_kld):g}*soft_usage_KLD",
        flush=True,
    )
    print(f"  beta={float(args.beta):g} init_codebook={args.init_codebook} usage_kld_tau={float(args.usage_kld_tau):g}", flush=True)
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
        f"Q={tokenizer.quantizer.__class__.__name__}({args.quantizer},K={int(args.vq_k)},C={int(args.token_ch)}) "
        f"D3={tokenizer.d3.__class__.__name__} "
        f"x1_cond={'on' if tokenizer.x1_cond is not None else 'off'} "
        f"z1_cond={'on' if tokenizer.z1_cond is not None else 'off'}",
        flush=True,
    )
    if str(args.quantizer) == "simvq":
        print(
            f"  simvq_base_codebook={parameter_state(tokenizer.quantizer.codebook)} "
            f"simvq_proj={trainable_state(tokenizer.quantizer.embedding_proj)}",
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
    tokenizer: Layer3PatchVQTokenizer,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    bits_per_token = int(math.ceil(math.log2(float(int(args.vq_k)))))
    payload = {
        "route": getattr(jsccf_io, "ROUTE", "TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256"),
        "stage": "stage3_patchvq_tokenizer_u2",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "version": str(getattr(args, "version", "")),
        "tokenizer_state_dict": tokenizer.state_dict(),
        "teacher_layer2_ckpt": str(args.layer2_ckpt),
        "tokenizer": {
            "arch": str(args.arch),
            "condition_mode": str(args.condition_mode),
            "quantizer": str(args.quantizer),
            "vq_k": int(args.vq_k),
            "token_ch": int(args.token_ch),
            "embedding_shape": [int(args.vq_k), int(args.token_ch), 1, 1],
            "idx3_shape": [int(args.latent_h), int(args.latent_w)],
            "q3_shape": [int(args.token_ch), int(args.latent_h), int(args.latent_w)],
            "fixed_bits_per_token": bits_per_token,
            "fixed_bits_per_image": bits_per_token * int(args.latent_h) * int(args.latent_w),
        },
        "latent": {
            "z1": [int(args.latent_ch), int(args.latent_h), int(args.latent_w)],
            "z3": [int(args.token_ch), int(args.latent_h), int(args.latent_w)],
        },
        "image_dims": [3, int(args.image_height), int(args.image_width)],
    }
    torch.save(payload, out)
    print(f"saved checkpoint: {out}", flush=True)


def train(args: argparse.Namespace, teacher_ckpt: dict) -> None:
    seed_everything(int(args.seed))
    patch_runtime_config()
    cfg = build_runtime_config(args, encoder_in_chans=3)
    train_loader, val_loader = get_loader(cfg)
    teacher = build_teacher(args, teacher_ckpt, cfg.device)
    tokenizer = Layer3PatchVQTokenizer(args, cfg.device)
    print_tokenizer_header(args, tokenizer, teacher, len(train_loader.dataset), len(val_loader.dataset))
    initialize_codebook_if_requested(train_loader, tokenizer, teacher, args, cfg.device)
    opt = optim.AdamW(tokenizer.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    if bool(args.eval_init_only):
        val_metrics = validate(val_loader, tokenizer, teacher, args, cfg.device)
        print(f"[stage3-patchvq init val] {display_metrics(val_metrics)} score=psnr_final", flush=True)
        out = Path(resolve_path(args.save_dir)) / f"{stage3_name(args)}_jscc_f_{jsccf_io.safe_artifact_name(args.version)}_init_eval.json"
        write_json(out, {"args": {k: v for k, v in vars(args).items() if not k.startswith("_")}, "metrics": val_metrics})
        print(f"[stage3-patchvq init val] wrote {out}", flush=True)
        return

    best = -1.0
    metrics: dict[str, float] = {}
    for epoch in range(1, int(args.epochs) + 1):
        tokenizer.train()
        teacher.eval()
        m = meters(METRIC_NAMES)
        hist = torch.zeros(int(args.vq_k), dtype=torch.float32)
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
            f"[stage3-patchvq train {epoch:03d}/{int(args.epochs):03d}] "
            f"{display_metrics(metrics)} time={time.time() - t0:.1f}s",
            flush=True,
        )
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, tokenizer, teacher, args, cfg.device)
            score = float(val_metrics["psnr_final"])
            print(f"[stage3-patchvq val {epoch:03d}] {display_metrics(val_metrics)} score=psnr_final", flush=True)
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
    patch_runtime_config()
    device = torch.device("cuda:0" if (not bool(args.cpu)) and torch.cuda.is_available() else "cpu")
    teacher = build_teacher(args, teacher_ckpt, device)
    tokenizer = Layer3PatchVQTokenizer(args, device)
    imgs = torch.rand(int(args.smoke_batch_size), 3, int(args.image_height), int(args.image_width), device=device)
    teacher_out = teacher.forward(imgs)
    out = tokenizer(imgs, teacher_out["x1"], teacher_out["z1"], teacher.combiner)
    expected_q3 = (int(args.smoke_batch_size), int(args.token_ch), int(args.latent_h), int(args.latent_w))
    expected_idx = (int(args.smoke_batch_size), int(args.latent_h), int(args.latent_w))
    expected_img = (int(args.smoke_batch_size), 3, int(args.image_height), int(args.image_width))
    print(
        f"[smoke] arch={args.arch} quantizer={args.quantizer} x1={tuple(teacher_out['x1'].shape)} "
        f"z1={tuple(teacher_out['z1'].shape)} q3={tuple(out['q3'].shape)} idx3={tuple(out['idx3'].shape)} "
        f"u2_hat={tuple(out['u2_hat'].shape)} final={tuple(out['final'].shape)} "
        f"embedding=[{int(args.vq_k)},{int(args.token_ch)},1,1]",
        flush=True,
    )
    if tuple(out["q3"].shape) != expected_q3:
        raise RuntimeError(f"expected q3 {expected_q3}, got {tuple(out['q3'].shape)}")
    if tuple(out["idx3"].shape) != expected_idx:
        raise RuntimeError(f"expected idx3 {expected_idx}, got {tuple(out['idx3'].shape)}")
    if tuple(out["u2_hat"].shape) != expected_img or tuple(out["final"].shape) != expected_img:
        raise RuntimeError(f"expected image outputs {expected_img}, got u2={tuple(out['u2_hat'].shape)} final={tuple(out['final'].shape)}")


def apply_preset(args: argparse.Namespace) -> None:
    if str(args.preset) == "custom":
        return
    token_ch, vq_k = PATCHVQ_PRESETS[str(args.preset)]
    args.token_ch = int(token_ch)
    args.vq_k = int(vq_k)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--arch", type=str, default="cnn", choices=["cnn", "swin"], help="Layer3 tokenizer architecture and default Layer2 teacher family.")
    p.add_argument("--version", type=str, default="patchvq-tokenizer-t1")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--layer2-ckpt", type=str, default="", help="Frozen Stage2 teacher checkpoint. Default is chosen from --arch.")
    p.add_argument("--ignore-ckpt-args", action="store_true", help="Do not copy teacher architecture fields from the Stage2 checkpoint args.")

    p.add_argument("--preset", type=str, default="custom", choices=["custom", "t0", "t1", "t2", "t3", "t4"], help="Shortcut for token_ch and vq_k.")
    p.add_argument("--quantizer", type=str, default="vq", choices=["vq", "simvq"])
    p.add_argument("--simvq", action="store_true", help="Alias for --quantizer simvq.")
    p.add_argument("--token-ch", type=int, default=16, help="Per-location embedding channel dimension C for codebook [K,C,1,1].")
    p.add_argument("--vq-k", type=int, default=1024, help="Number of image-vector codebook entries.")
    p.add_argument("--beta", type=float, default=0.25, help="Commitment weight inside VQ loss.")
    p.add_argument("--vq-chunk-size", type=int, default=4096, help="Distance-search chunk size over flattened [B*H*W,C] tokens.")
    p.add_argument("--simvq-train-codebook", action="store_true", help="Make SimVQ base codebook trainable. Default keeps base frozen and trains projection.")
    p.add_argument("--simvq-no-proj-bias", action="store_true", help="Disable bias in SimVQ projection.")
    p.add_argument("--init-codebook", type=str, default="none", choices=["none", "random_samples", "kmeans"])
    p.add_argument("--init-max-batches", type=int, default=16)
    p.add_argument("--init-max-tokens", type=int, default=65536)
    p.add_argument("--init-kmeans-iters", type=int, default=10)

    p.add_argument("--condition-mode", type=str, default="none", choices=["x1_z1", "x1_only", "x1", "z1_only", "z1", "none"])
    p.add_argument("--x1-cond-ch", type=int, default=16)
    p.add_argument("--z1-cond-ch", type=int, default=16)
    p.add_argument("--z1-cond-depth", type=int, default=2)

    p.add_argument("--image-size", type=int, default=256, help="Square image size fallback for image-height/width.")
    p.add_argument("--image-height", type=int, default=0)
    p.add_argument("--image-width", type=int, default=0)
    p.add_argument("--latent-ch", type=int, default=16)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--lambda-u", type=float, default=1.0)
    p.add_argument("--lambda-img", type=float, default=0.0)
    p.add_argument("--lambda-l1", type=float, default=0.0)
    p.add_argument("--lambda-vq", type=float, default=0.05)
    p.add_argument("--lambda-usage-kld", type=float, default=0.0)
    p.add_argument("--usage-kld-tau", type=float, default=1.0)

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
    args.stage = "stage3_patchvq_tokenizer"
    if bool(args.simvq):
        args.quantizer = "simvq"
    apply_preset(args)
    normalize_image_args(args)
    patch_runtime_config()
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(
            Path(resolve_path(args.save_dir))
            / f"{stage3_name(args)}_jscc_f_{jsccf_io.safe_artifact_name(args.version)}.log"
        )
    setup_log_file(args.log_file)
    teacher_ckpt = load_teacher_checkpoint_for_args(args)
    validate_patchvq_args(args)
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
