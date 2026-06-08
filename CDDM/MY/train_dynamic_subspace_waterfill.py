#!/usr/bin/env python3
"""Dynamic semantic subspace + water-filling on top of CDDM JSCC C16."""

from __future__ import annotations

import argparse
import builtins
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


CDDM_ROOT = Path(__file__).resolve().parents[1]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.meshgrid.*", category=UserWarning)

from Autoencoder.data.datasets import get_loader as get_cddm_loader  # noqa: E402
from Autoencoder.net.channel import Channel  # noqa: E402
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder  # noqa: E402


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


@dataclass
class CDDMJSCCConfig:
    C: int
    SNRs: float
    channel_type: str
    batch_size: int
    test_batch: int
    num_workers: int
    val_num_workers: int
    train_data_dir: str
    test_data_dir: str
    CUDA: bool = True
    dataset: str = "DIV2K"
    loss_function: str = "MSE"
    image_dims: tuple[int, int, int] = (3, 256, 256)
    pin_memory: bool = True
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        self.device = torch.device("cuda:0" if self.CUDA and torch.cuda.is_available() else "cpu")
        self.encoder_kwargs = dict(
            img_size=(self.image_dims[1], self.image_dims[2]),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 192, 256, 320],
            depths=[2, 2, 6, 2],
            num_heads=[4, 6, 8, 10],
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=torch.nn.LayerNorm,
            patch_norm=True,
        )
        self.decoder_kwargs = dict(
            img_size=(self.image_dims[1], self.image_dims[2]),
            embed_dims=[320, 256, 192, 128],
            depths=[2, 6, 2, 2],
            num_heads=[10, 8, 6, 4],
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=torch.nn.LayerNorm,
            patch_norm=True,
        )


class SemanticRouter(nn.Module):
    def __init__(self, in_ch: int = 16, hidden: int = 64, num_modes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_ch),
            nn.Linear(in_ch, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_modes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SemanticRouterV3(nn.Module):
    def __init__(
        self,
        in_ch: int = 48,
        bottleneck: int = 32,
        hidden: int = 128,
        num_modes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if bottleneck % 8 != 0:
            raise ValueError(f"router_v3 bottleneck must be divisible by 8, got {bottleneck}")
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, bottleneck, kernel_size=1, bias=False),
            nn.GroupNorm(8, bottleneck),
            nn.GELU(),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1, groups=bottleneck, bias=False),
            nn.GroupNorm(8, bottleneck),
            nn.GELU(),
        )
        stat_dim = bottleneck * 3
        self.head = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_modes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f = self.proj(z)
        f = self.dw(f)
        mean = f.mean(dim=(2, 3))
        std = f.std(dim=(2, 3), unbiased=False)
        low = F.avg_pool2d(f, 3, stride=1, padding=1)
        hf_energy = (f - low).pow(2).mean(dim=(2, 3))
        stat = torch.cat([mean, std, hf_energy], dim=1)
        return self.head(stat)


class NoRouter(nn.Module):
    def __init__(self, num_modes: int = 1):
        super().__init__()
        if int(num_modes) != 1:
            raise ValueError("router_type=none only supports num_modes=1")
        self.num_modes = int(num_modes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.new_zeros((z.shape[0], self.num_modes))


def init_A_raw_identity(num_modes: int, out_ch: int, in_ch: int) -> torch.Tensor:
    base = torch.zeros(out_ch, in_ch)
    base[:, :out_ch] = torch.eye(out_ch)
    A = torch.empty(num_modes, out_ch, in_ch)
    for k in range(num_modes):
        A[k] = base + 0.05 * torch.randn(out_ch, in_ch)
    return A


def init_A_raw_stiefel(num_modes: int, out_ch: int, in_ch: int, noise: float = 0.02) -> torch.Tensor:
    A = []
    for _k in range(num_modes):
        M = torch.randn(in_ch, out_ch)
        q, _ = torch.linalg.qr(M, mode="reduced")
        A.append(q.T + float(noise) * torch.randn(out_ch, in_ch))
    return torch.stack(A, dim=0)


def orthonormalize_rows(A_raw: torch.Tensor) -> torch.Tensor:
    rows = []
    for k in range(A_raw.shape[0]):
        q, _ = torch.linalg.qr(A_raw[k].float().T, mode="reduced")
        rows.append(q.T.to(dtype=A_raw.dtype))
    return torch.stack(rows, dim=0)


def power_normalize_per_sample(x: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    power = x.float().square().mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.sqrt(power + eps).to(dtype=x.dtype)
    return x / scale, scale


def psnr_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse = (x_hat.float().clamp(0.0, 1.0) - x.float()).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def format_metric_vector(values: dict[str, float], prefix: str, n: int, root: str = "") -> str:
    return "[" + ",".join(f"{values[f'{root}{prefix}{i}']:.2f}" for i in range(n)) + "]"


def format_meter_vector(meters: dict[str, AverageMeter], prefix: str, n: int) -> str:
    return "[" + ",".join(f"{meters[f'{prefix}{i}'].avg:.2f}" for i in range(n)) + "]"


class DynamicSubspaceChannel(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int = 16,
        out_ch: int = 4,
        num_modes: int = 4,
        router_hidden: int = 64,
        router_type: str = "avgpool",
        router_bottleneck: int = 32,
        router_dropout: float = 0.1,
        init_A: str = "identity",
        init_A_noise: float = 0.02,
        power_mode: str = "learnable",
        min_power: float = 0.05,
        channel_impl: str = "real_awgn",
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.num_modes = int(num_modes)
        self.min_power = float(min_power)
        self.channel_impl = str(channel_impl)
        self.power_mode = str(power_mode)
        if str(router_type) == "none":
            self.router = NoRouter(num_modes=num_modes)
        elif str(router_type) == "v3":
            self.router = SemanticRouterV3(
                in_ch=in_ch,
                bottleneck=router_bottleneck,
                hidden=router_hidden,
                num_modes=num_modes,
                dropout=router_dropout,
            )
        elif str(router_type) == "avgpool":
            self.router = SemanticRouter(in_ch=in_ch, hidden=router_hidden, num_modes=num_modes)
        else:
            raise ValueError(f"unsupported router_type={router_type}")
        if str(init_A) == "stiefel":
            A_init = init_A_raw_stiefel(num_modes, out_ch, in_ch, noise=init_A_noise)
        elif str(init_A) == "identity":
            A_init = init_A_raw_identity(num_modes, out_ch, in_ch)
        else:
            raise ValueError(f"unsupported init_A={init_A}")
        self.A_raw = nn.Parameter(A_init)
        self.power_logits = nn.Parameter(torch.zeros(num_modes, out_ch))
        self.power_logits.requires_grad_(self.power_mode == "learnable")

    def get_A(self) -> torch.Tensor:
        return orthonormalize_rows(self.A_raw)

    def get_power(self) -> torch.Tensor:
        if self.power_mode == "uniform":
            return torch.ones_like(self.power_logits, dtype=self.power_logits.dtype)
        if self.power_mode != "learnable":
            raise ValueError(f"unsupported power_mode={self.power_mode}")
        p = self.out_ch * F.softmax(self.power_logits.float(), dim=-1)
        p = p.clamp_min(self.min_power)
        p = self.out_ch * p / p.sum(dim=-1, keepdim=True)
        return p.to(dtype=self.power_logits.dtype)

    def select_mode(self, logits: torch.Tensor, tau: float, train_hard_argmax: bool) -> tuple[torch.Tensor, torch.Tensor]:
        prob = F.softmax(logits.float(), dim=-1)
        index = prob.argmax(dim=-1, keepdim=True)
        pi_hard = torch.zeros_like(prob).scatter_(1, index, 1.0)
        if not self.training:
            pi = pi_hard
        elif train_hard_argmax:
            pi = pi_hard - prob.detach() + prob
        else:
            pi = F.gumbel_softmax(logits.float(), tau=float(tau), hard=True, dim=-1)
        return pi.to(dtype=logits.dtype), prob.to(dtype=logits.dtype)

    def forward(
        self,
        z: torch.Tensor,
        *,
        channel: Channel,
        snr_db: float,
        tau: float,
        train_hard_argmax: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, C, H, W = z.shape
        if C != self.in_ch:
            raise ValueError(f"expected z channels={self.in_ch}, got {C}")

        z_float = z.float()
        logits = self.router(z_float)
        pi, prob = self.select_mode(logits, tau=tau, train_hard_argmax=train_hard_argmax)
        A_all = self.get_A().float()
        p_all = self.get_power().float()
        A_sel = torch.einsum("bk,koc->boc", pi.float(), A_all)
        p_sel = torch.einsum("bk,kc->bc", pi.float(), p_all)

        z_flat = z_float.permute(0, 2, 3, 1).reshape(B, H * W, C)
        u_flat = torch.einsum("boc,bnc->bno", A_sel, z_flat)
        u = u_flat.reshape(B, H, W, self.out_ch).permute(0, 3, 1, 2).contiguous()
        sqrt_p = torch.sqrt(p_sel.clamp_min(1e-8)).view(B, self.out_ch, 1, 1)
        u_power = u * sqrt_p

        if self.channel_impl == "cddm":
            y_complex, pwr, _h = channel.forward(u_power, float(snr_db))
            y = torch.cat((torch.real(y_complex), torch.imag(y_complex)), dim=2).float()
            if y.shape != u_power.shape:
                raise RuntimeError(f"CDDM channel shape mismatch: u_power={tuple(u_power.shape)} y={tuple(y.shape)}")
            scale = torch.sqrt(pwr.float().clamp_min(1e-12)).view(1, 1, 1, 1)
            u_rx_power = y * scale
        elif self.channel_impl == "real_awgn":
            u_norm, scale = power_normalize_per_sample(u_power)
            noise_std = 10.0 ** (-float(snr_db) / 20.0)
            y = u_norm + noise_std * torch.randn_like(u_norm)
            u_rx_power = y * scale
        else:
            raise ValueError(f"unsupported channel_impl={self.channel_impl}")

        u_rx = u_rx_power / sqrt_p.clamp_min(1e-6)
        u_rx_flat = u_rx.permute(0, 2, 3, 1).reshape(B, H * W, self.out_ch)
        z_hat_flat = torch.einsum("bno,boc->bnc", u_rx_flat, A_sel)
        z_hat = z_hat_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        aux = {
            "logits": logits,
            "prob": prob,
            "pi": pi,
            "A_all": A_all,
            "p_all": p_all,
            "p_sel": p_sel,
            "u_power": u_power,
            "u_rx_power": u_rx_power,
        }
        return z_hat, aux


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = Path(path)
    if not abs_path.is_absolute():
        abs_path = CDDM_ROOT / abs_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Dynamic subspace water-filling @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def default_jscc_root() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints" / "JSCC" / "DIV2K" / "MSE" / "SNRs")


def resolve_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else CDDM_ROOT / p)


def load_state(module: nn.Module, path: str, name: str) -> None:
    obj = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    missing, unexpected = module.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"{name} load mismatch: missing={missing}, unexpected={unexpected}")
    print(f"loaded {name}: {resolve_path(path)}")


def build_config(args: argparse.Namespace, batch_size: int) -> CDDMJSCCConfig:
    return CDDMJSCCConfig(
        C=int(args.latent_ch),
        SNRs=float(args.snr_db),
        channel_type=str(args.channel_type),
        batch_size=int(batch_size),
        test_batch=int(args.test_batch),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.val_num_workers),
        train_data_dir=resolve_path(os.path.join(args.data_dir, "DIV2K_train_HR")),
        test_data_dir=resolve_path(os.path.join(args.data_dir, "DIV2K_valid_HR")),
        CUDA=(not bool(args.cpu)),
    )


def orth_loss(A_all: torch.Tensor) -> torch.Tensor:
    K, m, _n = A_all.shape
    eye = torch.eye(m, device=A_all.device, dtype=A_all.dtype).unsqueeze(0).expand(K, -1, -1)
    gram = torch.einsum("koc,kdc->kod", A_all, A_all)
    return (gram - eye).square().mean()


def entropy_from_prob(prob: torch.Tensor) -> torch.Tensor:
    return -(prob.float() * prob.float().clamp_min(1e-8).log()).sum(dim=-1).mean()


@torch.no_grad()
def router_confidence(prob: torch.Tensor) -> torch.Tensor:
    return prob.float().max(dim=-1).values.mean()


@torch.no_grad()
def router_margin(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] < 2:
        return logits.float().new_tensor(0.0)
    top2 = logits.float().topk(2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).mean()


@torch.no_grad()
def usage_gap(pi: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    hard = pi.float().mean(dim=0)
    soft = prob.float().mean(dim=0)
    return (hard - soft).abs().mean()


def subspace_overlap_value(A_all: torch.Tensor) -> torch.Tensor:
    K = A_all.shape[0]
    vals = []
    for i in range(K):
        for j in range(i + 1, K):
            vals.append((A_all[i].float() @ A_all[j].float().T).pow(2).mean())
    return torch.stack(vals).mean() if vals else A_all.float().new_tensor(0.0)


@torch.no_grad()
def subspace_overlap(A_all: torch.Tensor) -> torch.Tensor:
    return subspace_overlap_value(A_all)


def power_entropy_value(p_all: torch.Tensor) -> torch.Tensor:
    q = p_all.float() / p_all.float().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(q * q.clamp_min(1e-8).log()).sum(dim=-1).mean()


@torch.no_grad()
def power_entropy(p_all: torch.Tensor) -> torch.Tensor:
    return power_entropy_value(p_all)


@torch.no_grad()
def power_var(p_all: torch.Tensor) -> torch.Tensor:
    p = p_all.float()
    p = p / p.mean(dim=-1, keepdim=True).clamp_min(1e-8)
    return p.var(dim=-1).mean()


def balance_weight_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    warm = getattr(args, "lambda_balance_warm", None)
    joint = getattr(args, "lambda_balance_joint", None)
    if warm is None and joint is None:
        return float(args.lambda_balance)
    if epoch <= int(args.warmup_epochs):
        return float(args.lambda_balance if warm is None else warm)
    return float(args.lambda_balance if joint is None else joint)


def tau_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    warm = max(1, int(args.warmup_epochs))
    if epoch <= warm:
        frac = (epoch - 1) / max(1, warm - 1)
        return float(args.tau_warm_start) + frac * (float(args.tau_warm_end) - float(args.tau_warm_start))
    remain = max(1, int(args.epochs) - warm)
    frac = (epoch - warm - 1) / max(1, remain - 1)
    return float(args.tau_joint_start) + frac * (float(args.tau_joint_end) - float(args.tau_joint_start))


def set_trainable(encoder: nn.Module, decoder: nn.Module, freeze: bool) -> None:
    for p in encoder.parameters():
        p.requires_grad = not freeze
    for p in decoder.parameters():
        p.requires_grad = not freeze


def run_batch(
    *,
    imgs: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    dyn: DynamicSubspaceChannel,
    channel: Channel,
    args: argparse.Namespace,
    epoch: int,
    train: bool,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    feature, _ = encoder(imgs)
    hard_argmax = train and int(args.hard_finetune_epochs) > 0 and epoch > int(args.epochs) - int(args.hard_finetune_epochs)
    z_hat, aux = dyn(
        feature.float(),
        channel=channel,
        snr_db=float(args.snr_db),
        tau=tau_for_epoch(args, epoch),
        train_hard_argmax=hard_argmax,
    )
    recon = decoder(z_hat).float().clamp(0.0, 1.0)
    loss_rec = F.mse_loss(recon, imgs.float())
    loss_orth = orth_loss(aux["A_all"])
    usage_hard = aux["pi"].float().mean(dim=0)
    usage_soft = aux["prob"].float().mean(dim=0)
    target = torch.full_like(usage_soft, 1.0 / float(args.num_modes))
    loss_balance = (usage_soft - target).square().mean()
    loss_subspace_div = subspace_overlap_value(aux["A_all"])
    loss_power_entropy = power_entropy_value(aux["p_all"])
    entropy = entropy_from_prob(aux["prob"])
    entropy_weight = -float(args.lambda_entropy_warm) if epoch <= int(args.warmup_epochs) else float(args.lambda_entropy_sharp)
    balance_weight = balance_weight_for_epoch(args, epoch)
    loss = (
        loss_rec
        + float(args.lambda_orth) * loss_orth
        + balance_weight * loss_balance
        + entropy_weight * entropy
        + float(args.lambda_subspace_div) * loss_subspace_div
        + float(args.lambda_power_entropy) * loss_power_entropy
    )

    p_all = aux["p_all"].detach().float()
    gram = torch.einsum("koc,kdc->kod", aux["A_all"], aux["A_all"]).detach().float()
    eye = torch.eye(gram.shape[-1], device=gram.device).view(1, gram.shape[-1], gram.shape[-1])
    aat_err = (gram - eye).abs().amax()
    p_sel = aux["p_sel"].detach().float()
    p_cpu = p_all.detach().float().cpu()
    p_sel_cpu = p_sel.cpu()
    stats = {
        "loss": float(loss.detach().item()),
        "loss_rec": float(loss_rec.detach().item()),
        "loss_orth": float(loss_orth.detach().item()),
        "loss_balance": float(loss_balance.detach().item()),
        "loss_subspace_div": float(loss_subspace_div.detach().item()),
        "loss_power_entropy": float(loss_power_entropy.detach().item()),
        "entropy": float(entropy.detach().item()),
        "router_conf": float(router_confidence(aux["prob"]).item()),
        "router_margin": float(router_margin(aux["logits"]).item()),
        "usage_gap": float(usage_gap(aux["pi"], aux["prob"]).item()),
        "subspace_overlap": float(subspace_overlap(aux["A_all"]).item()),
        "power_entropy": float(power_entropy(aux["p_all"]).item()),
        "power_var": float(power_var(aux["p_all"]).item()),
        "psnr": float(psnr_per_image(recon, imgs.float()).mean().item()),
        "aat_err": float(aat_err.item()),
        "p_min": float(p_cpu.min().item()),
        "p_max": float(p_cpu.max().item()),
        "p_sel_min": float(p_sel_cpu.min().item()),
        "p_sel_max": float(p_sel_cpu.max().item()),
    }
    usage_hard_cpu = usage_hard.detach().float().cpu()
    usage_soft_cpu = usage_soft.detach().float().cpu()
    for k in range(int(args.num_modes)):
        stats[f"usage{k}"] = float(usage_hard_cpu[k])
        stats[f"usage_soft{k}"] = float(usage_soft_cpu[k])
    p_sel_mean = p_sel_cpu.mean(dim=0)
    for c in range(int(args.subspace_out_ch)):
        stats[f"p_sel_mean{c}"] = float(p_sel_mean[c])
    return (loss if train else None), stats


def save_checkpoint(
    path: str,
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    dyn: DynamicSubspaceChannel,
    args: argparse.Namespace,
    metrics: dict[str, float],
    epoch: int,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": "dynamic_subspace_waterfill_cddm_jscc",
            "stage": str(args.stage_name),
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
            "source_jscc_encoder": resolve_path(args.encoder_ckpt),
            "source_jscc_decoder": resolve_path(args.decoder_ckpt),
            "channel": f"{args.channel_impl} snr={float(args.snr_db):g}dB",
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "dynamic_state_dict": dyn.state_dict(),
        },
        out,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--C", type=int, default=16, help="legacy alias for --latent_ch when --latent_ch is omitted")
    p.add_argument("--latent_ch", type=int, default=None, help="JSCC source latent channels Z")
    p.add_argument("--snr_db", type=float, default=6.0)
    p.add_argument("--channel_type", type=str, default="awgn", choices=["awgn"])
    p.add_argument("--channel_impl", type=str, default="real_awgn", choices=["cddm", "real_awgn"])
    p.add_argument("--jscc_root", type=str, default=default_jscc_root())
    p.add_argument("--encoder_ckpt", type=str, default="")
    p.add_argument("--decoder_ckpt", type=str, default="")

    p.add_argument("--num_modes", type=int, default=4)
    p.add_argument("--subspace_out_ch", type=int, default=4)
    p.add_argument("--router_hidden", type=int, default=64)
    p.add_argument("--router_type", type=str, default="avgpool", choices=["avgpool", "v3", "none"])
    p.add_argument("--router_bottleneck", type=int, default=32)
    p.add_argument("--router_dropout", type=float, default=0.1)
    p.add_argument("--init_A", type=str, default="identity", choices=["identity", "stiefel"])
    p.add_argument("--init_A_noise", type=float, default=0.02)
    p.add_argument("--power_mode", type=str, default="learnable", choices=["learnable", "uniform"])
    p.add_argument("--min_power", type=float, default=0.05)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--test_batch", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--warmup_epochs", type=int, default=30)
    p.add_argument("--hard_finetune_epochs", type=int, default=50)
    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)

    p.add_argument("--lr_dynamic", type=float, default=5e-5)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--lr_decoder", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--lambda_orth", type=float, default=0.0)
    p.add_argument("--lambda_balance", type=float, default=0.05)
    p.add_argument("--lambda_balance_warm", type=float, default=None)
    p.add_argument("--lambda_balance_joint", type=float, default=None)
    p.add_argument("--lambda_entropy_warm", type=float, default=0.01)
    p.add_argument("--lambda_entropy_sharp", type=float, default=0.0)
    p.add_argument("--lambda_subspace_div", type=float, default=0.0)
    p.add_argument("--lambda_power_entropy", type=float, default=0.0)
    p.add_argument("--tau_warm_start", type=float, default=2.0)
    p.add_argument("--tau_warm_end", type=float, default=1.0)
    p.add_argument("--tau_joint_start", type=float, default=1.0)
    p.add_argument("--tau_joint_end", type=float, default=0.5)
    p.add_argument("--amp_dtype", type=str, default="none", choices=["none", "bfloat16", "float16"])

    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--stage_name", type=str, default="dynamic_subspace_waterfill_snr6_c16_k4")
    p.add_argument("--ckpt_prefix", type=str, default="dynamic_subspace_waterfill")
    p.add_argument(
        "--save_dir",
        type=str,
        default="MY/checkpoints-dynamic/dynamic_subspace_waterfill_snr6_c16_k4_realawgn_v2",
    )
    p.add_argument(
        "--log_file",
        type=str,
        default="MY/checkpoints-dynamic/dynamic_subspace_waterfill_snr6_c16_k4_realawgn_v2/train.log",
    )
    args = p.parse_args()
    if args.latent_ch is None:
        args.latent_ch = int(args.C)
    args.C = int(args.latent_ch)
    if not args.encoder_ckpt:
        args.encoder_ckpt = os.path.join(args.jscc_root, f"encoder_snr{args.snr_db:g}_channel_{args.channel_type}_C{args.latent_ch}.pt")
    if not args.decoder_ckpt:
        args.decoder_ckpt = os.path.join(args.jscc_root, f"decoder_snr{args.snr_db:g}_channel_{args.channel_type}_C{args.latent_ch}.pt")
    return args


def main() -> None:
    args = parse_args()
    setup_log_file(args.log_file)
    seed_everything(int(args.seed))
    save_dir = Path(resolve_path(args.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_enabled = device.type == "cuda" and str(args.amp_dtype) != "none"
    amp_dtype = torch.bfloat16 if str(args.amp_dtype) == "bfloat16" else torch.float16

    cfg = build_config(args, int(args.batch_size))
    train_loader, val_loader = get_cddm_loader(cfg)
    if val_loader is None:
        raise RuntimeError("validation loader is required")

    encoder = JSCC_encoder(cfg, cfg.C).to(device)
    decoder = JSCC_decoder(cfg, cfg.C).to(device)
    dyn = DynamicSubspaceChannel(
        in_ch=int(args.latent_ch),
        out_ch=int(args.subspace_out_ch),
        num_modes=int(args.num_modes),
        router_hidden=int(args.router_hidden),
        router_type=str(args.router_type),
        router_bottleneck=int(args.router_bottleneck),
        router_dropout=float(args.router_dropout),
        init_A=str(args.init_A),
        init_A_noise=float(args.init_A_noise),
        power_mode=str(args.power_mode),
        min_power=float(args.min_power),
        channel_impl=str(args.channel_impl),
    ).to(device)
    channel = Channel(cfg)
    load_state(encoder, args.encoder_ckpt, "CDDM JSCC encoder")
    load_state(decoder, args.decoder_ckpt, "CDDM JSCC decoder")

    optimizer = optim.AdamW(
        [
            {"params": list(dyn.parameters()), "lr": float(args.lr_dynamic), "name": "dynamic"},
            {"params": list(encoder.parameters()), "lr": float(args.lr_encoder), "name": "encoder"},
            {"params": list(decoder.parameters()), "lr": float(args.lr_decoder), "name": "decoder"},
        ],
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    print(f"device={device}, visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, amp={amp_dtype if amp_enabled else 'fp32'}")
    print(f"stage={args.stage_name} epochs={args.epochs} warmup={args.warmup_epochs} hard_ft={args.hard_finetune_epochs}")
    print(f"train/test snr={float(args.snr_db):g}dB channel={args.channel_type}/{args.channel_impl} latent_ch={args.latent_ch}")
    print(
        f"dynamic: K={args.num_modes} A=[{args.subspace_out_ch},{args.latent_ch}] row-QR "
        f"router={args.router_type} init_A={args.init_A} power={args.power_mode} "
        f"p_sum={args.subspace_out_ch}, min_power={args.min_power}"
    )
    print(
        f"data train={len(train_loader.dataset)} valid={len(val_loader.dataset)} "
        f"batch={args.batch_size} grad_accum={args.grad_accum_steps} "
        f"effective_batch={int(args.batch_size) * int(args.grad_accum_steps)} test_batch={args.test_batch}"
    )
    print(f"encoder_ckpt={resolve_path(args.encoder_ckpt)}")
    print(f"decoder_ckpt={resolve_path(args.decoder_ckpt)}")
    print(f"save_dir={save_dir}")
    print(
        f"loss=L_rec + {args.lambda_orth:g}*L_orth + balance*L_balance "
        f"- {args.lambda_entropy_warm:g}*H(warm), + {args.lambda_entropy_sharp:g}*H(joint) "
        f"+ {args.lambda_subspace_div:g}*L_subspace_div + {args.lambda_power_entropy:g}*L_power_entropy "
        f"(balance warm={args.lambda_balance_warm if args.lambda_balance_warm is not None else args.lambda_balance:g}, "
        f"joint={args.lambda_balance_joint if args.lambda_balance_joint is not None else args.lambda_balance:g})"
    )

    base_meter_keys = [
        "loss",
        "loss_rec",
        "loss_orth",
        "loss_balance",
        "loss_subspace_div",
        "loss_power_entropy",
        "entropy",
        "router_conf",
        "router_margin",
        "usage_gap",
        "subspace_overlap",
        "power_entropy",
        "power_var",
        "psnr",
        "aat_err",
        "p_min",
        "p_max",
        "p_sel_min",
        "p_sel_max",
    ]
    meter_keys = tuple(
        base_meter_keys
        + [f"usage{k}" for k in range(int(args.num_modes))]
        + [f"usage_soft{k}" for k in range(int(args.num_modes))]
        + [f"p_sel_mean{c}" for c in range(int(args.subspace_out_ch))]
    )
    best = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        freeze_ed = epoch <= int(args.warmup_epochs)
        set_trainable(encoder, decoder, freeze=freeze_ed)
        encoder.train(not freeze_ed)
        decoder.train(not freeze_ed)
        dyn.train(True)
        meters = {k: AverageMeter() for k in meter_keys}
        optimizer.zero_grad(set_to_none=True)
        accum_steps = max(1, int(args.grad_accum_steps))
        pending_backward = 0
        for bi, batch in enumerate(train_loader):
            if int(args.max_train_batches) > 0 and bi >= int(args.max_train_batches):
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss, stats = run_batch(
                    imgs=imgs,
                    encoder=encoder,
                    decoder=decoder,
                    dyn=dyn,
                    channel=channel,
                    args=args,
                    epoch=epoch,
                    train=True,
                )
            assert loss is not None
            scaler.scale(loss / float(accum_steps)).backward()
            pending_backward += 1
            if pending_backward >= accum_steps:
                if float(args.clip_grad_norm) > 0:
                    scaler.unscale_(optimizer)
                    params = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(params, float(args.clip_grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_backward = 0
            for k in meters:
                meters[k].update(stats[k], imgs.shape[0])
        if pending_backward > 0:
            if float(args.clip_grad_norm) > 0:
                scaler.unscale_(optimizer)
                params = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params, float(args.clip_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        metrics = {f"train_{k}": v.avg for k, v in meters.items()}
        do_eval = epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs)
        if do_eval:
            encoder.eval()
            decoder.eval()
            dyn.eval()
            val_meters = {k: AverageMeter() for k in meter_keys}
            with torch.no_grad():
                for bi, batch in enumerate(val_loader):
                    if int(args.max_val_batches) > 0 and bi >= int(args.max_val_batches):
                        break
                    imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
                    imgs = imgs.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        _loss, stats = run_batch(
                            imgs=imgs,
                            encoder=encoder,
                            decoder=decoder,
                            dyn=dyn,
                            channel=channel,
                            args=args,
                            epoch=epoch,
                            train=False,
                        )
                    for k in val_meters:
                        val_meters[k].update(stats[k], imgs.shape[0])
            metrics.update({f"val_{k}": v.avg for k, v in val_meters.items()})
            score = metrics["val_psnr"]
            is_best = score > best
            if is_best:
                best = score
                save_checkpoint(
                    str(save_dir / f"{args.ckpt_prefix}_best.pth"),
                    encoder=encoder,
                    decoder=decoder,
                    dyn=dyn,
                    args=args,
                    metrics=metrics,
                    epoch=epoch,
                )
            save_checkpoint(
                str(save_dir / f"{args.ckpt_prefix}_latest.pth"),
                encoder=encoder,
                decoder=decoder,
                dyn=dyn,
                args=args,
                metrics=metrics,
                epoch=epoch,
            )
            p_all = dyn.get_power().detach().float().cpu().numpy()
            p_text = "; ".join("[" + ",".join(f"{v:.2f}" for v in row) + "]" for row in p_all)
            usage_text = format_metric_vector(metrics, "usage", int(args.num_modes), root="val_")
            usage_soft_text = format_metric_vector(metrics, "usage_soft", int(args.num_modes), root="val_")
            p_sel_mean_text = format_metric_vector(metrics, "p_sel_mean", int(args.subspace_out_ch), root="val_")
            print(
                f"[epoch {epoch:03d}/{args.epochs}] phase={'warm' if freeze_ed else 'joint'} "
                f"tau={tau_for_epoch(args, epoch):.3f} "
                f"loss={meters['loss'].avg:.6f} rec={meters['loss_rec'].avg:.6f} | "
                f"val_psnr={score:.4f} usage={usage_text} usage_soft={usage_soft_text} "
                f"H={metrics['val_entropy']:.3f} bal={metrics['val_loss_balance']:.5f} "
                f"conf={metrics['val_router_conf']:.3f} margin={metrics['val_router_margin']:.3f} "
                f"gap={metrics['val_usage_gap']:.4f} overlap={metrics['val_subspace_overlap']:.6f} "
                f"pH={metrics['val_power_entropy']:.3f} pvar={metrics['val_power_var']:.4f} "
                f"aat={metrics['val_aat_err']:.2e} "
                f"p_all_minmax=[{metrics['val_p_min']:.2f},{metrics['val_p_max']:.2f}] "
                f"p_sel_minmax=[{metrics['val_p_sel_min']:.2f},{metrics['val_p_sel_max']:.2f}] "
                f"p_sel_mean={p_sel_mean_text} "
                f"p_all={p_text} {'BEST' if is_best else ''}",
                flush=True,
            )
        else:
            usage_text = format_meter_vector(meters, "usage", int(args.num_modes))
            usage_soft_text = format_meter_vector(meters, "usage_soft", int(args.num_modes))
            p_sel_mean_text = format_meter_vector(meters, "p_sel_mean", int(args.subspace_out_ch))
            print(
                f"[epoch {epoch:03d}/{args.epochs}] phase={'warm' if freeze_ed else 'joint'} "
                f"tau={tau_for_epoch(args, epoch):.3f} "
                f"loss={meters['loss'].avg:.6f} rec={meters['loss_rec'].avg:.6f} "
                f"psnr={meters['psnr'].avg:.4f} usage={usage_text} usage_soft={usage_soft_text} "
                f"H={meters['entropy'].avg:.3f} conf={meters['router_conf'].avg:.3f} "
                f"margin={meters['router_margin'].avg:.3f} gap={meters['usage_gap'].avg:.4f} "
                f"overlap={meters['subspace_overlap'].avg:.6f} "
                f"pH={meters['power_entropy'].avg:.3f} pvar={meters['power_var'].avg:.4f} "
                f"p_sel_minmax=[{meters['p_sel_min'].avg:.2f},{meters['p_sel_max'].avg:.2f}] "
                f"p_sel_mean={p_sel_mean_text}",
                flush=True,
            )
    print(f"best_val_psnr={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
