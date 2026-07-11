from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

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
    AverageMeter,
    averaged,
    batch_metric_mean,
    check_jsccf_args,
    clamp_img,
    meters,
    mse_per_image,
    print_epoch,
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
from model import build_jscc_decoder, build_jscc_encoder


def load_local_io():
    spec = importlib.util.spec_from_file_location("jsccf_test_ed_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = load_local_io()


EXPERIMENTS = {
    "exp01_swin_channel": ("swin", "channel_map", "ed_exp01_swin_channelmap_k16384"),
    "exp02_swin_image": ("swin", "image_vector", "ed_exp02_swin_imagevec_k256"),
    "exp03_cnn_channel": ("cnn", "channel_map", "ed_exp03_cnn_channelmap_k16384"),
    "exp04_cnn_image": ("cnn", "image_vector", "ed_exp04_cnn_imagevec_k256"),
    "exp05_cnn_noquant": ("cnn", "none", "ed_exp05_cnn_noquant"),
    "exp06_swin_channel_simvq": ("swin", "channel_map_simvq", "ed_exp06_swin_channelmap_simvq_k16384"),
    "exp07_swin_image_simvq": ("swin", "image_vector_simvq", "ed_exp07_swin_imagevec_simvq_k256"),
    "exp08_cnn_channel_simvq": ("cnn", "channel_map_simvq", "ed_exp08_cnn_channelmap_simvq_k16384"),
    "exp09_cnn_image_simvq": ("cnn", "image_vector_simvq", "ed_exp09_cnn_imagevec_simvq_k256"),
    "exp10_swin320_channel_simvq": ("swin_320", "channel_map_simvq", "ed_exp10_swin320_channelmap_simvq_k16384"),
    "exp11_swin320_image_simvq": ("swin_320", "image_vector_simvq", "ed_exp11_swin320_imagevec_simvq_k256"),
    "exp12_swin320_channel": ("swin_320", "channel_map", "ed_exp12_swin320_channelmap_k16384"),
    "exp13_swin_image_ema": ("swin", "image_vector_ema", "ed_exp13_swin_imagevec_ema_k256"),
    "exp14_swin320_image_ema": ("swin_320", "image_vector_ema", "ed_exp14_swin320_imagevec_ema_k256"),
    "exp15_swin_image_ema_init_layer1": ("swin", "image_vector_ema", "ed_exp15_swin_imagevec_ema_init_layer1_k256"),
}

SWIN320_EXPERIMENTS = {"exp10_swin320_channel_simvq", "exp11_swin320_image_simvq", "exp12_swin320_channel", "exp14_swin320_image_ema"}
VQDEEPISC_EXPERIMENTS = {"exp13_swin_image_ema", "exp14_swin320_image_ema", "exp15_swin_image_ema_init_layer1"}
CODEC_INIT_EXPERIMENTS = {"exp15_swin_image_ema_init_layer1"}
DEFAULT_CODEC_INIT_CKPT = "MY-V2/jscc-f/checkpoints/jscc_f_no-c1_layer1_best.pth"

DEFAULT_EPOCHS = 400
DEFAULT_LR = 1e-4
DEFAULT_LR_CODEBOOK = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP_NORM = 0.0
DEFAULT_LR_SCHEDULER = "none"
DEFAULT_LR_MIN = 0.0
DEFAULT_LR_SCHEDULER_T_MAX = 0


def make_norm(channels: int, groups: int = 32) -> nn.GroupNorm:
    groups = min(int(groups), int(channels))
    while int(channels) % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int | None = None) -> None:
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=int(kernel), stride=int(stride), padding=int(padding)),
            make_norm(int(out_ch)),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = make_norm(channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.conv2(out)
        return x + out


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_res: int = 2) -> None:
        super().__init__()
        self.pre = nn.Sequential(*[ResidualBlock(int(in_ch)) for _ in range(int(num_res))])
        self.down = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, stride=2, padding=1)
        self.post = nn.Sequential(*[ResidualBlock(int(out_ch)) for _ in range(int(num_res))])
        self.tail = ConvNormAct(int(out_ch), int(out_ch), kernel=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.down(x)
        x = self.post(x)
        return self.tail(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_res: int = 2, up_mode: str = "bilinear") -> None:
        super().__init__()
        self.res = nn.Sequential(*[ResidualBlock(int(in_ch)) for _ in range(int(num_res))])
        self.up_mode = str(up_mode)
        self.conv = ConvNormAct(int(in_ch), int(out_ch), kernel=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        if self.up_mode == "bilinear":
            x = F.interpolate(x, scale_factor=2, mode=self.up_mode, align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.up_mode)
        return self.conv(x)


class CNNAnalysisEncoder(nn.Module):
    """Pasted-text architecture 2 encoder: [B,3,256,256] -> [B,16,16,16]."""

    def __init__(self, base_ch: int = 16, bottleneck_ch: int = 16, num_res: int = 2) -> None:
        super().__init__()
        base = int(base_ch)
        self.stem = ConvNormAct(3, base, kernel=3, stride=1)
        self.down0 = DownBlock(base, base * 2, num_res=num_res)
        self.down1 = DownBlock(base * 2, base * 4, num_res=num_res)
        self.down2 = DownBlock(base * 4, base * 8, num_res=num_res)
        self.down3 = DownBlock(base * 8, base * 16, num_res=num_res)
        high_ch = base * 16
        self.refine = nn.Sequential(ResidualBlock(high_ch), ResidualBlock(high_ch))
        self.compressor = nn.Sequential(
            ResidualBlock(high_ch),
            nn.Conv2d(high_ch, base * 4, kernel_size=1),
            make_norm(base * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 4, int(bottleneck_ch), kernel_size=1),
            make_norm(int(bottleneck_ch)),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(bottleneck_ch), int(bottleneck_ch), kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.refine(x)
        return self.compressor(x)


class CNNBottleneckDecoder(nn.Module):
    """Pasted-text architecture 2 decoder: [B,16,16,16] -> [B,3,256,256]."""

    def __init__(
        self,
        base_ch: int = 16,
        bottleneck_ch: int = 16,
        num_res: int = 2,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        base = int(base_ch)
        high_ch = base * 16
        bottleneck_ch = int(bottleneck_ch)
        self.expander = nn.Sequential(
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1),
            make_norm(bottleneck_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(bottleneck_ch, base * 4, kernel_size=1),
            make_norm(base * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 4, high_ch, kernel_size=1),
            make_norm(high_ch),
            nn.SiLU(inplace=True),
            ResidualBlock(high_ch),
            ResidualBlock(high_ch),
        )
        self.init = nn.Sequential(
            nn.Conv2d(high_ch, high_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            ResidualBlock(high_ch),
            ResidualBlock(high_ch),
        )
        self.up0 = UpBlock(high_ch, base * 8, num_res=num_res)
        self.up1 = UpBlock(base * 8, base * 4, num_res=num_res)
        self.up2 = UpBlock(base * 4, base * 2, num_res=num_res)
        self.up3 = UpBlock(base * 2, base, num_res=num_res)
        head: list[nn.Module] = [
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, 3, kernel_size=3, padding=1),
        ]
        if output_activation == "sigmoid":
            head.append(nn.Sigmoid())
        elif output_activation == "tanh":
            head.append(nn.Tanh())
        elif output_activation != "none":
            raise ValueError(f"unknown output activation {output_activation!r}")
        self.head = nn.Sequential(*head)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.expander(z)
        x = self.init(x)
        x = self.up0(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.head(x)


def nearest_codebook_2d(tokens: torch.Tensor, codebook: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 2 or codebook.ndim != 2:
        raise ValueError(f"expected 2D tokens/codebook, got {tuple(tokens.shape)} and {tuple(codebook.shape)}")
    cb = codebook.float()
    cb_norm = cb.square().sum(dim=1).view(1, -1)
    q_tokens = []
    q_indices = []
    chunk = max(1, int(chunk_size))
    for start in range(0, int(tokens.shape[0]), chunk):
        x = tokens[start : start + chunk].float()
        dist = x.square().sum(dim=1, keepdim=True) + cb_norm - 2.0 * x @ cb.t()
        idx = dist.argmin(dim=1)
        q_indices.append(idx)
        q_tokens.append(codebook[idx])
    return torch.cat(q_tokens, dim=0).to(dtype=tokens.dtype), torch.cat(q_indices, dim=0)


def init_linear_identity(linear: nn.Linear) -> None:
    if int(linear.weight.shape[0]) != int(linear.weight.shape[1]):
        raise ValueError(f"identity init requires square linear, got {tuple(linear.weight.shape)}")
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.weight.shape[0], device=linear.weight.device, dtype=linear.weight.dtype))
        if linear.bias is not None:
            linear.bias.zero_()


def soft_code_usage_kld_loss(tokens: torch.Tensor, codebook: torch.Tensor, tau: float, chunk_size: int) -> torch.Tensor:
    if float(tau) <= 0.0:
        raise ValueError(f"usage KLD tau must be positive, got {tau}")
    if tokens.ndim != 2 or codebook.ndim != 2:
        raise ValueError(f"expected 2D tokens/codebook, got {tuple(tokens.shape)} and {tuple(codebook.shape)}")
    cb = codebook.float()
    cb_norm = cb.square().sum(dim=1).view(1, -1)
    probs_sum = []
    chunk = max(1, int(chunk_size))
    for start in range(0, int(tokens.shape[0]), chunk):
        x = tokens[start : start + chunk].float()
        dist = x.square().sum(dim=1, keepdim=True) + cb_norm - 2.0 * x @ cb.t()
        probs_sum.append(F.softmax(-dist / float(tau), dim=1).sum(dim=0))
    probs = torch.stack(probs_sum, dim=0).sum(dim=0) / float(tokens.shape[0])
    probs = probs.clamp_min(1e-8)
    return (probs * (probs.log() + math.log(float(codebook.shape[0])))).sum()


class ImageVectorQuantizer(nn.Module):
    """Per spatial location VQ. Codebook embedding shape is [K,16,1,1]."""

    def __init__(self, num_codes: int = 256, channels: int = 16, beta: float = 0.25, chunk_size: int = 4096) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.channels = int(channels)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        weight = torch.randn(self.num_codes, self.channels, 1, 1) * (float(self.channels) ** -0.5)
        self.codebook = nn.Parameter(weight)

    @property
    def embedding_shape(self) -> tuple[int, int, int]:
        return (self.channels, 1, 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if z.ndim != 4 or int(z.shape[1]) != self.channels:
            raise ValueError(f"expected z [B,{self.channels},H,W], got {tuple(z.shape)}")
        bsz, channels, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        codebook = self.codebook.view(self.num_codes, channels)
        q_flat, indices = nearest_codebook_2d(flat, codebook, self.chunk_size)
        q = q_flat.view(bsz, h, w, channels).permute(0, 3, 1, 2).contiguous()
        return vq_outputs(z, q, indices.view(bsz, h, w), self.beta)


class ImageVectorEMAQuantizer(nn.Module):
    """VQ-DeepISC style patch-token VQ: codebook [K,C], indices [B,H,W], EMA updates."""

    def __init__(
        self,
        num_codes: int = 256,
        channels: int = 16,
        beta: float = 0.25,
        kld_weight: float = 0.05,
        usage_kld_tau: float = 1.0,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.channels = int(channels)
        self.beta = float(beta)
        self.kld_weight = float(kld_weight)
        self.usage_kld_tau = float(usage_kld_tau)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.chunk_size = int(chunk_size)
        weight = torch.randn(self.num_codes, self.channels, 1, 1) * (float(self.channels) ** -0.5)
        self.codebook = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("ema_count", torch.ones(self.num_codes))
        self.register_buffer("ema_sum", weight.view(self.num_codes, self.channels).clone())

    @property
    def embedding_shape(self) -> tuple[int, int, int]:
        return (self.channels, 1, 1)

    @torch.no_grad()
    def ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        flat_f = flat.detach().float()
        one_hot = F.one_hot(indices.reshape(-1), num_classes=self.num_codes).to(dtype=flat_f.dtype, device=flat_f.device)
        batch_count = one_hot.sum(dim=0)
        batch_sum = one_hot.t() @ flat_f
        decay = float(self.ema_decay)
        self.ema_count.mul_(decay).add_(batch_count, alpha=1.0 - decay)
        self.ema_sum.mul_(decay).add_(batch_sum, alpha=1.0 - decay)
        updated = self.ema_sum / (self.ema_count.unsqueeze(1) + float(self.ema_eps))
        self.codebook.data.copy_(updated.view_as(self.codebook))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if z.ndim != 4 or int(z.shape[1]) != self.channels:
            raise ValueError(f"expected z [B,{self.channels},H,W], got {tuple(z.shape)}")
        bsz, channels, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        codebook = self.codebook.view(self.num_codes, channels)
        q_flat, indices = nearest_codebook_2d(flat, codebook, self.chunk_size)
        q = q_flat.view(bsz, h, w, channels).permute(0, 3, 1, 2).contiguous()
        kld_loss = None
        if float(self.kld_weight) > 0.0:
            kld_loss = soft_code_usage_kld_loss(flat, codebook.detach().clone(), self.usage_kld_tau, self.chunk_size)
        if self.training:
            self.ema_update(flat, indices)
        return vq_outputs(
            z,
            q,
            indices.view(bsz, h, w),
            self.beta,
            kld_weight=self.kld_weight,
            kld_loss=kld_loss,
        )


class ImageVectorSimVQQuantizer(nn.Module):
    """Per spatial location SimVQ. Base codebook is [K,16,1,1], then projected."""

    def __init__(
        self,
        num_codes: int = 256,
        channels: int = 16,
        beta: float = 0.25,
        chunk_size: int = 4096,
        freeze_codebook: bool = True,
        proj_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.channels = int(channels)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        weight = torch.randn(self.num_codes, self.channels, 1, 1) * (float(self.channels) ** -0.5)
        self.codebook = nn.Parameter(weight)
        self.codebook.requires_grad_(not bool(freeze_codebook))
        self.embedding_proj = nn.Linear(self.channels, self.channels, bias=bool(proj_bias))
        init_linear_identity(self.embedding_proj)

    @property
    def embedding_shape(self) -> tuple[int, int, int]:
        return (self.channels, 1, 1)

    def effective_codebook(self) -> torch.Tensor:
        flat = self.codebook.float().view(self.num_codes, self.channels)
        return self.embedding_proj(flat).view(self.num_codes, self.channels, 1, 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if z.ndim != 4 or int(z.shape[1]) != self.channels:
            raise ValueError(f"expected z [B,{self.channels},H,W], got {tuple(z.shape)}")
        bsz, channels, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        codebook = self.effective_codebook().view(self.num_codes, channels)
        q_flat, indices = nearest_codebook_2d(flat, codebook, self.chunk_size)
        q = q_flat.view(bsz, h, w, channels).permute(0, 3, 1, 2).contiguous()
        return vq_outputs(z, q, indices.view(bsz, h, w), self.beta)


class ChannelMapQuantizer(nn.Module):
    """Per channel-map VQ. Codebook embedding shape is [K,16,16]."""

    def __init__(self, num_codes: int = 16384, h: int = 16, w: int = 16, beta: float = 0.25, chunk_size: int = 128) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        scale = float(self.h * self.w) ** -0.5
        self.codebook = nn.Parameter(torch.randn(self.num_codes, self.h, self.w) * scale)

    @property
    def embedding_shape(self) -> tuple[int, int]:
        return (self.h, self.w)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if z.ndim != 4 or tuple(z.shape[2:]) != (self.h, self.w):
            raise ValueError(f"expected z [B,C,{self.h},{self.w}], got {tuple(z.shape)}")
        bsz, channels, h, w = z.shape
        flat = z.reshape(bsz * channels, h * w)
        codebook = self.codebook.reshape(self.num_codes, h * w)
        q_flat, indices = nearest_codebook_2d(flat, codebook, self.chunk_size)
        q = q_flat.view(bsz, channels, h, w)
        return vq_outputs(z, q, indices.view(bsz, channels), self.beta)


class ChannelMapSimVQQuantizer(nn.Module):
    """Per channel-map SimVQ. Base codebook is [K,16,16], then projected."""

    def __init__(
        self,
        num_codes: int = 16384,
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
        init_linear_identity(self.embedding_proj)

    @property
    def embedding_shape(self) -> tuple[int, int]:
        return (self.h, self.w)

    def effective_codebook(self) -> torch.Tensor:
        flat = self.codebook.float().view(self.num_codes, self.h * self.w)
        return self.embedding_proj(flat).view(self.num_codes, self.h, self.w)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if z.ndim != 4 or tuple(z.shape[2:]) != (self.h, self.w):
            raise ValueError(f"expected z [B,C,{self.h},{self.w}], got {tuple(z.shape)}")
        bsz, channels, h, w = z.shape
        flat = z.reshape(bsz * channels, h * w)
        codebook = self.effective_codebook().reshape(self.num_codes, h * w)
        q_flat, indices = nearest_codebook_2d(flat, codebook, self.chunk_size)
        q = q_flat.view(bsz, channels, h, w)
        return vq_outputs(z, q, indices.view(bsz, channels), self.beta)


def vq_outputs(
    z: torch.Tensor,
    q: torch.Tensor,
    indices: torch.Tensor,
    beta: float,
    kld_weight: float = 0.0,
    kld_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    q_st = z + (q - z).detach()
    codebook_loss = F.mse_loss(q.float(), z.detach().float())
    commit_loss = F.mse_loss(q.detach().float(), z.float())
    if kld_loss is None:
        kld_loss = codebook_loss.new_zeros(())
    vq_loss = codebook_loss + float(beta) * commit_loss + float(kld_weight) * kld_loss
    stats = {
        "codebook_loss": codebook_loss,
        "commit_loss": commit_loss,
        "kld_loss": kld_loss,
        "vq_loss": vq_loss,
        "vq_mse": F.mse_loss(q.detach().float(), z.detach().float()),
    }
    return q_st, q, indices, stats


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Conv") != -1 and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator matching the SimVQ/taming discriminator shape."""

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        kw = 4
        padw = 1
        sequence: list[nn.Module] = [
            nn.Conv2d(int(input_nc), int(ndf), kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, int(n_layers)):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(int(ndf) * nf_mult_prev, int(ndf) * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(int(ndf) * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** int(n_layers), 8)
        sequence += [
            nn.Conv2d(int(ndf) * nf_mult_prev, int(ndf) * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(int(ndf) * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(ndf) * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def sigmoid_cross_entropy_with_logits(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return F.relu(logits) - logits * labels + torch.log1p(torch.exp(-torch.abs(logits)))


def non_saturate_gen_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    bsz = int(logits_fake.shape[0])
    logits = logits_fake.reshape(bsz, -1).mean(dim=-1)
    return sigmoid_cross_entropy_with_logits(torch.ones_like(logits), logits).mean()


def non_saturate_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    bsz = int(logits_fake.shape[0])
    real = logits_real.reshape(bsz, -1).mean(dim=-1)
    fake = logits_fake.reshape(bsz, -1).mean(dim=-1)
    real_loss = sigmoid_cross_entropy_with_logits(torch.ones_like(real), real)
    fake_loss = sigmoid_cross_entropy_with_logits(torch.zeros_like(fake), fake)
    return real_loss.mean() + fake_loss.mean()


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean())


def discriminator_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "hinge":
        return hinge_d_loss(logits_real, logits_fake)
    if mode == "vanilla":
        return vanilla_d_loss(logits_real, logits_fake)
    if mode == "non_saturate":
        return non_saturate_d_loss(logits_real, logits_fake)
    raise ValueError(f"unknown GAN discriminator loss {mode!r}")


def set_requires_grad(module: nn.Module | None, requires_grad: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(bool(requires_grad))


class LatentAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module | None,
        latent_ch: int = 16,
        latent_h: int = 16,
        latent_w: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.latent_shape = (int(latent_ch), int(latent_h), int(latent_w))

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        z = self.encoder(img)
        if isinstance(z, tuple):
            z = z[0]
        if tuple(z.shape[1:]) != self.latent_shape:
            raise ValueError(f"encoder must produce [B,{self.latent_shape}], got {tuple(z.shape)}")
        return z

    def forward(self, img: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        z = self.encode(img)
        if self.quantizer is None:
            zero = z.new_zeros(())
            stats = {
                "codebook_loss": zero,
                "commit_loss": zero,
                "kld_loss": zero,
                "vq_loss": zero,
                "vq_mse": zero,
            }
            z_path = z
            z_q = z
            indices = None
        else:
            z_path, z_q, indices, stats = self.quantizer(z)
        recon_raw = self.decoder(z_path)
        recon = clamp_img(recon_raw)
        return {
            "z": z,
            "z_q": z_q,
            "indices": indices,
            "vq_stats": stats,
            "recon_raw": recon_raw,
            "recon": recon,
        }


def build_encoder_decoder(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module]:
    if args.arch == "swin":
        encoder = build_jscc_encoder(args, device, latent_ch=int(args.latent_ch), in_chans=3)
        decoder = build_jscc_decoder(args, device, latent_ch=int(args.latent_ch))
        return encoder, decoder
    if args.arch == "swin_320":
        encoder = build_jscc_encoder(args, device, latent_ch=320, in_chans=3)
        decoder = build_jscc_decoder(args, device, latent_ch=320)
        encoder.encoder.head_list = nn.Identity()
        decoder.decoder.head_list = nn.Identity()
        return encoder, decoder
    if args.arch == "cnn":
        encoder = CNNAnalysisEncoder(
            base_ch=int(args.cnn_base_ch),
            bottleneck_ch=int(args.latent_ch),
            num_res=int(args.cnn_num_res),
        ).to(device)
        decoder = CNNBottleneckDecoder(
            base_ch=int(args.cnn_base_ch),
            bottleneck_ch=int(args.latent_ch),
            num_res=int(args.cnn_num_res),
            output_activation=str(args.output_activation),
        ).to(device)
        return encoder, decoder
    raise ValueError(f"unknown architecture {args.arch!r}")


def build_quantizer(args: argparse.Namespace, device: torch.device) -> nn.Module | None:
    if args.quantizer == "none":
        return None
    if args.quantizer == "image_vector":
        return ImageVectorQuantizer(
            num_codes=int(args.image_k),
            channels=int(args.latent_ch),
            beta=float(args.beta_commit),
            chunk_size=int(args.image_chunk_size),
        ).to(device)
    if args.quantizer == "image_vector_ema":
        return ImageVectorEMAQuantizer(
            num_codes=int(args.image_k),
            channels=int(args.latent_ch),
            beta=float(args.beta_commit),
            kld_weight=float(args.beta_kld),
            usage_kld_tau=float(args.usage_kld_tau),
            ema_decay=float(args.vq_ema_decay),
            ema_eps=float(args.vq_ema_eps),
            chunk_size=int(args.image_chunk_size),
        ).to(device)
    if args.quantizer == "image_vector_simvq":
        return ImageVectorSimVQQuantizer(
            num_codes=int(args.image_k),
            channels=int(args.latent_ch),
            beta=float(args.beta_commit),
            chunk_size=int(args.image_chunk_size),
            freeze_codebook=not bool(args.simvq_train_codebook),
            proj_bias=bool(args.simvq_proj_bias),
        ).to(device)
    if args.quantizer == "channel_map":
        return ChannelMapQuantizer(
            num_codes=int(args.channel_k),
            h=int(args.latent_h),
            w=int(args.latent_w),
            beta=float(args.beta_commit),
            chunk_size=int(args.channel_chunk_size),
        ).to(device)
    if args.quantizer == "channel_map_simvq":
        return ChannelMapSimVQQuantizer(
            num_codes=int(args.channel_k),
            h=int(args.latent_h),
            w=int(args.latent_w),
            beta=float(args.beta_commit),
            chunk_size=int(args.channel_chunk_size),
            freeze_codebook=not bool(args.simvq_train_codebook),
            proj_bias=bool(args.simvq_proj_bias),
        ).to(device)
    raise ValueError(f"unknown quantizer {args.quantizer!r}")


def build_model(args: argparse.Namespace, device: torch.device) -> LatentAutoencoder:
    encoder, decoder = build_encoder_decoder(args, device)
    quantizer = build_quantizer(args, device)
    return LatentAutoencoder(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        latent_ch=int(args.latent_ch),
        latent_h=int(args.latent_h),
        latent_w=int(args.latent_w),
    ).to(device)


def load_codec_initialization(args: argparse.Namespace, model: LatentAutoencoder) -> None:
    if not str(args.init_codec_ckpt):
        return
    jsccf_io.load_layer1_compatible_checkpoint(
        str(args.init_codec_ckpt),
        model.encoder,
        model.decoder,
        strict=bool(args.init_codec_strict),
    )
    print(
        f"initialized encoder/decoder from {resolve_path(args.init_codec_ckpt)} strict={bool(args.init_codec_strict)}",
        flush=True,
    )


def build_discriminator(args: argparse.Namespace, device: torch.device) -> nn.Module | None:
    if float(args.lambda_gan) <= 0.0:
        return None
    discriminator = NLayerDiscriminator(
        input_nc=3,
        ndf=int(args.disc_ndf),
        n_layers=int(args.disc_layers),
    ).to(device)
    discriminator.apply(weights_init)
    return discriminator


def trainable_params(module: nn.Module | None) -> list[nn.Parameter]:
    if module is None:
        return []
    return [p for p in module.parameters() if p.requires_grad]


def count_params(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def build_optimizer(args: argparse.Namespace, model: LatentAutoencoder) -> optim.Optimizer:
    param_groups: list[dict] = []
    codec_params = trainable_params(model.encoder) + trainable_params(model.decoder)
    if codec_params:
        param_groups.append({"params": codec_params, "lr": float(args.lr)})
    quantizer_params = trainable_params(model.quantizer)
    if quantizer_params:
        param_groups.append({"params": quantizer_params, "lr": float(args.lr_codebook)})
    if not param_groups:
        raise ValueError("no trainable parameters")
    return optim.AdamW(param_groups, weight_decay=float(args.weight_decay))


def build_discriminator_optimizer(args: argparse.Namespace, discriminator: nn.Module | None) -> optim.Optimizer | None:
    if discriminator is None:
        return None
    params = trainable_params(discriminator)
    if not params:
        raise ValueError("discriminator has no trainable parameters")
    return optim.AdamW(params, lr=float(args.disc_lr), weight_decay=float(args.disc_weight_decay))


def build_lr_scheduler(args: argparse.Namespace, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler | None:
    if str(args.lr_scheduler) == "none":
        return None
    if str(args.lr_scheduler) == "cosine":
        t_max = int(args.lr_scheduler_t_max) if int(args.lr_scheduler_t_max) > 0 else int(args.epochs)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(t_max)),
            eta_min=float(args.lr_min),
        )
    raise ValueError(f"unknown lr scheduler {args.lr_scheduler!r}")


def current_lr(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def quantizer_num_codes(args: argparse.Namespace) -> int:
    if args.quantizer in {"channel_map", "channel_map_simvq"}:
        return int(args.channel_k)
    if args.quantizer in {"image_vector", "image_vector_simvq", "image_vector_ema"}:
        return int(args.image_k)
    return 0


def update_hist(hist: torch.Tensor | None, indices: torch.Tensor | None) -> None:
    if hist is None or indices is None:
        return
    flat = indices.detach().reshape(-1).cpu()
    hist += torch.bincount(flat, minlength=int(hist.numel())).float()


def usage_metrics(hist: torch.Tensor | None) -> dict[str, float]:
    if hist is None or int(hist.sum().item()) == 0:
        return {}
    used = float((hist > 0).sum().item())
    probs = hist / hist.sum().clamp_min(1.0)
    nz = probs[probs > 0]
    entropy = float(-(nz * nz.log()).sum().item())
    return {
        "used_codes": used,
        "usage_percent": 100.0 * used / float(hist.numel()),
        "perplexity": float(torch.exp(torch.tensor(entropy)).item()),
        "index_entropy": entropy,
    }


def ssim_loss_value(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (1.0 - ssim_per_image(x_hat, x)).mean()


def gan_is_active(args: argparse.Namespace, epoch: int) -> bool:
    return float(args.lambda_gan) > 0.0 and int(epoch) >= int(args.gan_start_epoch)


def compute_losses(
    out: dict,
    imgs: torch.Tensor,
    args: argparse.Namespace,
    discriminator: nn.Module | None,
    gan_active: bool,
) -> dict[str, torch.Tensor]:
    stats = out["vq_stats"]
    loss_rec = recon_loss(out["recon_raw"], imgs)
    if float(args.lambda_ssim) > 0.0:
        loss_ssim = ssim_loss_value(out["recon_raw"], imgs)
    else:
        loss_ssim = loss_rec.new_zeros(())
    loss_vq = stats["vq_loss"]
    loss_gan = loss_rec.new_zeros(())
    logits_fake_g = loss_rec.new_zeros(())
    if gan_active and discriminator is not None:
        logits_fake = discriminator(out["recon_raw"].contiguous())
        loss_gan = non_saturate_gen_loss(logits_fake)
        logits_fake_g = logits_fake.detach().mean()
    loss = loss_rec + float(args.lambda_ssim) * loss_ssim + float(args.lambda_vq) * loss_vq + float(args.lambda_gan) * loss_gan
    return {
        "loss": loss,
        "loss_rec": loss_rec,
        "loss_ssim": loss_ssim,
        "loss_gan": loss_gan,
        "loss_vq": loss_vq,
        "loss_codebook": stats["codebook_loss"],
        "loss_commit": stats["commit_loss"],
        "loss_kld": stats["kld_loss"],
        "vq_mse": stats["vq_mse"],
        "logits_fake_g": logits_fake_g,
    }


def compute_discriminator_losses(
    discriminator: nn.Module | None,
    imgs: torch.Tensor,
    recon_raw: torch.Tensor,
    args: argparse.Namespace,
    gan_active: bool,
) -> dict[str, torch.Tensor]:
    zero = recon_raw.new_zeros(())
    if not gan_active or discriminator is None:
        return {
            "loss_disc": zero,
            "logits_real": zero,
            "logits_fake_d": zero,
        }
    logits_real = discriminator(imgs.contiguous().detach())
    logits_fake = discriminator(recon_raw.contiguous().detach())
    loss_disc = float(args.disc_factor) * discriminator_loss(logits_real, logits_fake, str(args.gan_loss))
    return {
        "loss_disc": loss_disc,
        "logits_real": logits_real.detach().mean(),
        "logits_fake_d": logits_fake.detach().mean(),
    }


def update_metrics(m: dict[str, AverageMeter], hist: torch.Tensor | None, out: dict, imgs: torch.Tensor, losses: dict[str, torch.Tensor]) -> None:
    bsz = int(imgs.shape[0])
    for name, value in losses.items():
        m[name].update(float(value.detach().item()), bsz)
    m["mse"].update(batch_metric_mean(mse_per_image(out["recon"], imgs)), bsz)
    m["psnr"].update(batch_metric_mean(psnr_per_image(out["recon"], imgs)), bsz)
    m["ssim"].update(batch_metric_mean(ssim_per_image(out["recon"], imgs)), bsz)
    m["latent_abs_mean"].update(float(out["z"].detach().abs().mean().item()), bsz)
    update_hist(hist, out["indices"])


def metric_names() -> list[str]:
    return [
        "loss",
        "loss_rec",
        "loss_ssim",
        "loss_gan",
        "loss_vq",
        "loss_disc",
        "loss_codebook",
        "loss_commit",
        "loss_kld",
        "vq_mse",
        "logits_real",
        "logits_fake_g",
        "logits_fake_d",
        "mse",
        "psnr",
        "ssim",
        "latent_abs_mean",
    ]


def display_metrics(metrics: dict[str, float]) -> dict[str, float]:
    order = [
        "loss",
        "loss_rec",
        "loss_ssim",
        "loss_gan",
        "loss_vq",
        "loss_kld",
        "loss_disc",
        "mse",
        "psnr",
        "ssim",
        "vq_mse",
        "logits_real",
        "logits_fake_g",
        "logits_fake_d",
        "used_codes",
        "usage_percent",
        "perplexity",
        "latent_abs_mean",
        "lr",
    ]
    return {k: metrics[k] for k in order if k in metrics}


def run_epoch(
    loader,
    model: LatentAutoencoder,
    discriminator: nn.Module | None,
    args: argparse.Namespace,
    optimizer: optim.Optimizer | None,
    optimizer_d: optim.Optimizer | None,
    device: torch.device,
    max_batches: int,
    epoch: int,
) -> dict[str, float]:
    training = optimizer is not None
    active_gan = gan_is_active(args, epoch)
    model.train(training)
    if discriminator is not None:
        discriminator.train(training)
    m = meters(metric_names())
    hist = torch.zeros(quantizer_num_codes(args), dtype=torch.float32) if quantizer_num_codes(args) > 0 else None
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, (imgs, _labels) in enumerate(loader):
            if int(max_batches) > 0 and batch_idx >= int(max_batches):
                break
            imgs = imgs.to(device, non_blocking=True)
            if training:
                set_requires_grad(discriminator, False)
            out = model(imgs)
            losses = compute_losses(out, imgs, args, discriminator, active_gan)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if float(args.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
                optimizer.step()
            if training and optimizer_d is not None and active_gan:
                set_requires_grad(discriminator, True)
            d_losses = compute_discriminator_losses(discriminator, imgs, out["recon_raw"], args, active_gan)
            if training and optimizer_d is not None and active_gan:
                optimizer_d.zero_grad(set_to_none=True)
                d_losses["loss_disc"].backward()
                optimizer_d.step()
            losses.update(d_losses)
            update_metrics(m, hist, out, imgs, losses)
    metrics = averaged(m)
    metrics.update(usage_metrics(hist))
    return metrics


def checkpoint_path(args: argparse.Namespace, suffix: str) -> str:
    version = jsccf_io.safe_artifact_name(args.version)
    return str(Path(resolve_path(args.save_dir)) / f"test_ed_{version}_{suffix}.pth")


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    model: LatentAutoencoder,
    discriminator: nn.Module | None,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "route": "test_ed-no-channel-no-powernorm",
        "stage": "joint_encoder_quantizer_decoder",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "experiment": str(args.experiment),
        "arch": str(args.arch),
        "quantizer": str(args.quantizer),
        "latent": [int(args.latent_ch), int(args.latent_h), int(args.latent_w)],
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "decoder_state_dict": model.decoder.state_dict(),
        "quantizer_state_dict": model.quantizer.state_dict() if model.quantizer is not None else None,
        "discriminator_state_dict": discriminator.state_dict() if discriminator is not None else None,
    }
    torch.save(payload, out)
    print(f"saved checkpoint: {out}", flush=True)


def print_run_header(args: argparse.Namespace, model: LatentAutoencoder, discriminator: nn.Module | None, train_n: int, val_n: int) -> None:
    model_device = next(model.parameters()).device
    print(f"=== test_ed | {args.version} ===", flush=True)
    print(f"device={model_device} cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"save_dir={resolve_path(args.save_dir)}", flush=True)
    print("实验设计", flush=True)
    print(f"  experiment={args.experiment} arch={args.arch} quantizer={args.quantizer}", flush=True)
    print(f"  image=[B,3,256,256] latent=[B,{args.latent_ch},{args.latent_h},{args.latent_w}]", flush=True)
    print("  path=image->encoder->quantizer(optional)->decoder->reconstruction", flush=True)
    if str(args.init_codec_ckpt):
        print(
            f"  channel=identity power_norm=none awgn=none codec_init={resolve_path(args.init_codec_ckpt)} "
            "quantizer_init=from_scratch",
            flush=True,
        )
    else:
        print("  channel=identity power_norm=none awgn=none init=from_scratch", flush=True)
    print("loss设计", flush=True)
    if args.quantizer == "image_vector_ema":
        print(
            f"  L=MSE(x_hat_raw,x)+{float(args.lambda_ssim):g}*(1-SSIM(x_hat,x))"
            f"+{float(args.lambda_vq):g}*(codebook+{float(args.beta_commit):g}*commit+{float(args.beta_kld):g}*KLD)",
            flush=True,
        )
    else:
        print(
            f"  L=MSE(x_hat_raw,x)+{float(args.lambda_ssim):g}*(1-SSIM(x_hat,x))"
            f"+{float(args.lambda_vq):g}*(codebook+{float(args.beta_commit):g}*commit)",
            flush=True,
        )
    print(
        f"  GAN=lambda_gan({float(args.lambda_gan):g})*non_saturating_G "
        f"+ disc_factor({float(args.disc_factor):g})*{args.gan_loss}_D "
        f"start_epoch={int(args.gan_start_epoch)}",
        flush=True,
    )
    print("模块选择", flush=True)
    if args.arch == "swin":
        print("  encoder=JSCC_encoder/SwinTransformer decoder=JSCC_decoder/SwinTransformer", flush=True)
    elif args.arch == "swin_320":
        print("  encoder=JSCC_encoder/SwinTransformer without 320->16 tail conv", flush=True)
        print("  decoder=JSCC_decoder/SwinTransformer without 16->320 input conv", flush=True)
    else:
        print("  encoder=CNNAnalysisEncoder+1x1Compressor decoder=1x1Expander+CNNSynthesisDecoder", flush=True)
    if args.quantizer == "channel_map":
        print(f"  quantizer=ChannelMapVQ embedding=[{args.latent_h},{args.latent_w}] K={args.channel_k} index_shape=[B,{args.latent_ch}]", flush=True)
    elif args.quantizer == "channel_map_simvq":
        print(
            f"  quantizer=ChannelMapSimVQ base_embedding=[{args.latent_h},{args.latent_w}] K={args.channel_k} "
            f"index_shape=[B,{args.latent_ch}] frozen_codebook={not bool(args.simvq_train_codebook)} trainable_linear=True",
            flush=True,
        )
    elif args.quantizer == "image_vector":
        print(f"  quantizer=ImageVectorVQ embedding=[{args.latent_ch},1,1] K={args.image_k} index_shape=[B,{args.latent_h},{args.latent_w}]", flush=True)
    elif args.quantizer == "image_vector_ema":
        print(
            f"  quantizer=ImageVectorEMA/VQ-DeepISC embedding=[{args.latent_ch},1,1] K={args.image_k} "
            f"index_shape=[B,{args.latent_h},{args.latent_w}] soft_kld_tau={float(args.usage_kld_tau):g} "
            f"ema_decay={float(args.vq_ema_decay):g} ema_eps={float(args.vq_ema_eps):g}",
            flush=True,
        )
    elif args.quantizer == "image_vector_simvq":
        print(
            f"  quantizer=ImageVectorSimVQ base_embedding=[{args.latent_ch},1,1] K={args.image_k} "
            f"index_shape=[B,{args.latent_h},{args.latent_w}] frozen_codebook={not bool(args.simvq_train_codebook)} trainable_linear=True",
            flush=True,
        )
    else:
        print("  quantizer=none continuous CNN baseline", flush=True)
    print(
        f"  params encoder={count_params(model.encoder)} decoder={count_params(model.decoder)} "
        f"quantizer={count_params(model.quantizer)} discriminator={count_params(discriminator)} "
        f"total={count_params(model) + count_params(discriminator)}",
        flush=True,
    )
    scheduler_t_max = int(args.lr_scheduler_t_max) if int(args.lr_scheduler_t_max) > 0 else int(args.epochs)
    print(
        f"优化配置 optimizer=AdamW lr={float(args.lr):g} lr_codebook={float(args.lr_codebook):g} "
        f"weight_decay={float(args.weight_decay):g} grad_clip_max_norm={float(args.grad_clip_norm):g}",
        flush=True,
    )
    print(
        f"LR Scheduler={args.lr_scheduler} T_max={scheduler_t_max} eta_min={float(args.lr_min):g}",
        flush=True,
    )
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}", flush=True)


def apply_experiment_preset(args: argparse.Namespace) -> None:
    if args.experiment == "custom":
        if args.version == "auto":
            args.version = f"ed_custom_{args.arch}_{args.quantizer}"
        return
    experiment = str(args.experiment)
    arch, quantizer, version = EXPERIMENTS[experiment]
    args.arch = arch
    args.quantizer = quantizer
    if experiment in SWIN320_EXPERIMENTS:
        args.latent_ch = 320
        args.c1_ch = 320
    if experiment in VQDEEPISC_EXPERIMENTS:
        if int(args.epochs) == DEFAULT_EPOCHS:
            args.epochs = 300
        if float(args.lr) == DEFAULT_LR:
            args.lr = 2e-4
        if float(args.lr_codebook) == DEFAULT_LR_CODEBOOK:
            args.lr_codebook = 2e-4
        if float(args.weight_decay) == DEFAULT_WEIGHT_DECAY:
            args.weight_decay = 1e-4
        if float(args.grad_clip_norm) == DEFAULT_GRAD_CLIP_NORM:
            args.grad_clip_norm = 5.0
        if str(args.lr_scheduler) == DEFAULT_LR_SCHEDULER:
            args.lr_scheduler = "cosine"
        if float(args.lr_min) == DEFAULT_LR_MIN:
            args.lr_min = 1e-6
        if int(args.lr_scheduler_t_max) == DEFAULT_LR_SCHEDULER_T_MAX:
            args.lr_scheduler_t_max = 300
    if experiment in CODEC_INIT_EXPERIMENTS and not str(args.init_codec_ckpt):
        args.init_codec_ckpt = DEFAULT_CODEC_INIT_CKPT
    if args.version == "auto":
        args.version = version


def smoke_shapes(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    device = torch.device("cuda:0" if (not bool(args.cpu)) and torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    load_codec_initialization(args, model)
    model.eval()
    imgs = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    with torch.no_grad():
        out = model(imgs)
    print(
        f"[smoke] arch={args.arch} quantizer={args.quantizer} "
        f"z={tuple(out['z'].shape)} recon={tuple(out['recon_raw'].shape)}",
        flush=True,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = get_loader(cfg)
    model = build_model(args, cfg.device)
    load_codec_initialization(args, model)
    discriminator = build_discriminator(args, cfg.device)
    optimizer = build_optimizer(args, model)
    optimizer_d = build_discriminator_optimizer(args, discriminator)
    scheduler = build_lr_scheduler(args, optimizer)
    print_run_header(args, model, discriminator, len(train_loader.dataset), len(val_loader.dataset) if val_loader is not None else 0)

    best = -1.0
    latest_metrics: dict[str, float] = {}
    if int(args.epochs) <= 0:
        if val_loader is not None:
            latest_metrics = run_epoch(
                val_loader,
                model,
                discriminator,
                args,
                optimizer=None,
                optimizer_d=None,
                device=cfg.device,
                max_batches=int(args.max_val_batches),
                epoch=0,
            )
            print(f"[test_ed val 000] {display_metrics(latest_metrics)} score=psnr", flush=True)
        save_checkpoint(checkpoint_path(args, "latest"), epoch=0, args=args, metrics=latest_metrics, model=model, discriminator=discriminator)
        return

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            train_loader,
            model,
            discriminator,
            args,
            optimizer=optimizer,
            optimizer_d=optimizer_d,
            device=cfg.device,
            max_batches=int(args.max_train_batches),
            epoch=epoch,
        )
        train_metrics["lr"] = current_lr(optimizer)
        latest_metrics = train_metrics
        print_epoch("test_ed", epoch, int(args.epochs), display_metrics(train_metrics), time.time() - t0)
        if should_validate(args, epoch) and val_loader is not None:
            val_metrics = run_epoch(
                val_loader,
                model,
                discriminator,
                args,
                optimizer=None,
                optimizer_d=None,
                device=cfg.device,
                max_batches=int(args.max_val_batches),
                epoch=epoch,
            )
            latest_metrics = val_metrics
            score = float(val_metrics["psnr"])
            print(f"[test_ed val {epoch:03d}] {display_metrics(val_metrics)} score=psnr", flush=True)
            if score > best:
                best = score
                save_checkpoint(checkpoint_path(args, "best"), epoch=epoch, args=args, metrics=val_metrics, model=model, discriminator=discriminator)
        if should_save_latest(args, epoch):
            save_checkpoint(checkpoint_path(args, "latest"), epoch=epoch, args=args, metrics=train_metrics, model=model, discriminator=discriminator)
        if scheduler is not None:
            scheduler.step()
    save_checkpoint(checkpoint_path(args, "latest"), epoch=int(args.epochs), args=args, metrics=latest_metrics, model=model, discriminator=discriminator)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--experiment", type=str, default="custom", choices=["custom", *EXPERIMENTS.keys()])
    p.add_argument("--version", type=str, default="auto")
    p.add_argument("--arch", type=str, default="cnn", choices=["swin", "swin_320", "cnn"])
    p.add_argument(
        "--quantizer",
        type=str,
        default="image_vector",
        choices=["channel_map", "image_vector", "image_vector_ema", "channel_map_simvq", "image_vector_simvq", "none"],
    )
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=str(Path(jsccf_io.default_save_dir()) / "test_ed"))
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-codec-ckpt", type=str, default="", help="Optional layer1-compatible checkpoint used to initialize encoder/decoder only.")
    p.add_argument("--init-codec-nonstrict", dest="init_codec_strict", action="store_false", help="Allow missing/unexpected encoder/decoder keys when loading --init-codec-ckpt.")
    p.add_argument("--latent-ch", type=int, default=16)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--channel-k", type=int, default=16384, help="Channel-map VQ K; embedding shape is [16,16].")
    p.add_argument("--image-k", type=int, default=256, help="Image/spatial-location VQ K; embedding shape is [16,1,1].")
    p.add_argument("--channel-chunk-size", type=int, default=128)
    p.add_argument("--image-chunk-size", type=int, default=4096)
    p.add_argument("--simvq-train-codebook", action="store_true", help="Also train the base SimVQ codebook; default freezes it and trains only the projection.")
    p.add_argument("--simvq-no-proj-bias", dest="simvq_proj_bias", action="store_false", help="Disable bias in the trainable SimVQ projection.")
    p.add_argument("--beta-commit", type=float, default=0.25)
    p.add_argument("--beta-kld", type=float, default=0.05, help="KLD weight inside ImageVectorEMA/VQ-DeepISC VQ loss.")
    p.add_argument("--usage-kld-tau", type=float, default=1.0, help="Temperature for differentiable softmax(-distance/tau) code-usage KLD.")
    p.add_argument("--vq-ema-decay", type=float, default=0.99, help="EMA decay for ImageVectorEMA/VQ-DeepISC codebook updates.")
    p.add_argument("--vq-ema-eps", type=float, default=1e-5, help="Denominator epsilon for ImageVectorEMA/VQ-DeepISC codebook updates.")
    p.add_argument("--lambda-ssim", type=float, default=0.0, help="Weight for differentiable SSIM loss term 1-SSIM; default keeps old MSE+VQ objective.")
    p.add_argument("--lambda-vq", type=float, default=1)
    p.add_argument("--lambda-gan", type=float, default=0.0, help="Weight for PatchGAN generator loss; 0 disables GAN training.")
    p.add_argument("--gan-start-epoch", type=int, default=1, help="First epoch where GAN losses are active.")
    p.add_argument("--gan-loss", type=str, default="hinge", choices=["hinge", "vanilla", "non_saturate"], help="Discriminator loss. Generator loss always uses SimVQ-style non-saturating loss.")
    p.add_argument("--disc-factor", type=float, default=1.0, help="Multiplier for discriminator loss after GAN starts.")
    p.add_argument("--disc-layers", type=int, default=3)
    p.add_argument("--disc-ndf", type=int, default=64)
    p.add_argument("--disc-lr", type=float, default=1e-4)
    p.add_argument("--disc-weight-decay", type=float, default=1e-4)
    p.add_argument("--cnn-base-ch", type=int, default=16)
    p.add_argument("--cnn-num-res", type=int, default=2)
    p.add_argument("--output-activation", type=str, default="none", choices=["none", "sigmoid", "tanh"])
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--lr-codebook", type=float, default=DEFAULT_LR_CODEBOOK)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--lr-scheduler", type=str, default=DEFAULT_LR_SCHEDULER, choices=["none", "cosine"])
    p.add_argument("--lr-min", type=float, default=DEFAULT_LR_MIN)
    p.add_argument("--lr-scheduler-t-max", type=int, default=DEFAULT_LR_SCHEDULER_T_MAX)
    p.add_argument("--grad-clip-norm", type=float, default=DEFAULT_GRAD_CLIP_NORM)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260701)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke-shapes", action="store_true", help="Build the requested model, run one random forward pass, and exit.")
    p.add_argument("--smoke-batch-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_experiment_preset(args)
    args.stage = "test_ed"
    args.init_ckpt = str(args.init_codec_ckpt)
    check_jsccf_args(args)
    if args.quantizer == "none" and args.experiment != "exp05_cnn_noquant" and args.arch != "cnn":
        print("[warn] no-quant baseline was requested outside exp05; continuing with the explicit CLI.", flush=True)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if args.smoke_shapes:
        smoke_shapes(args)
        return
    if not args.log_file:
        args.log_file = str(Path(resolve_path(args.save_dir)) / f"test_ed_{jsccf_io.safe_artifact_name(args.version)}.log")
    setup_log_file(args.log_file)
    write_json(Path(resolve_path(args.save_dir)) / f"test_ed_{jsccf_io.safe_artifact_name(args.version)}_args.json", vars(args))
    train(args)


if __name__ == "__main__":
    main()
