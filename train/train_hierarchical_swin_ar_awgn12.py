#!/usr/bin/env python
"""Hierarchical Swin latent + AR receiver for AWGN12 C=4.

This experiment trains a frequency-ordered 16-channel Swin latent:

  z = [z0, z1, z2, z3], each group has 4 channels.

The fixed channel codec is A = [I4, 0], so A A^T = I4 exactly.  Only z0 is
transmitted through power-normalized AWGN12.  The receiver predicts z1/z2/z3
autoregressively from the noisy z0 and decodes the reconstructed z16.
"""

from __future__ import annotations

import argparse
import builtins
import os
import sys
import time
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
TRAIN_DIR = os.path.abspath(os.path.dirname(__file__))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from src.cddm_mimo_ddnm import DIV2KDataset, get_div2k_config  # noqa: E402
from src.cddm_mimo_ddnm.loss import kl_loss  # noqa: E402
from src.cddm_mimo_ddnm.modules.semantic_codec import SemanticDecoder, SemanticEncoder  # noqa: E402
from train_codex_orthogonal_highfreq import (  # noqa: E402
    make_autocast,
    power_normalize_awgn,
    psnr_per_image,
    seed_everything,
    semiorth_error,
    _parse_amp,
)
from train_route_a_sc import AverageMeter, GaussianBlur, TeeStream, load_state_dict_from_ckpt  # noqa: E402


class ConvResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GroupPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden: int, depth: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            *[ConvResBlock(hidden) for _ in range(int(depth))],
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 4, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_scale.to(dtype=x.dtype) * self.net(x)


class ARHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            ConvResBlock(hidden),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_scale.to(dtype=x.dtype) * self.net(x)


class SmallLatentUNet(nn.Module):
    """16x16 latent U-Net: only downsample to 8x8 and 4x4."""

    def __init__(self, in_channels: int = 14, base: int = 64, depth: int = 2) -> None:
        super().__init__()
        d = max(1, int(depth))
        h1, h2, h3 = base, base * 2, base * 4
        self.stem = nn.Sequential(nn.Conv2d(in_channels, h1, 3, padding=1), nn.SiLU())
        self.enc1 = nn.Sequential(*[ConvResBlock(h1) for _ in range(d)])
        self.down1 = nn.Sequential(nn.Conv2d(h1, h2, 3, stride=2, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(*[ConvResBlock(h2) for _ in range(d)])
        self.down2 = nn.Sequential(nn.Conv2d(h2, h3, 3, stride=2, padding=1), nn.SiLU())
        self.mid = nn.Sequential(*[ConvResBlock(h3) for _ in range(max(2, d + 1))])
        self.up2_conv = nn.Sequential(nn.Conv2d(h3, h2, 3, padding=1), nn.SiLU())
        self.up2_fuse = nn.Sequential(nn.Conv2d(h2 + h2, h2, 1), ConvResBlock(h2), ConvResBlock(h2))
        self.up1_conv = nn.Sequential(nn.Conv2d(h2, h1, 3, padding=1), nn.SiLU())
        self.up1_fuse = nn.Sequential(nn.Conv2d(h1 + h1, h1, 1), ConvResBlock(h1), ConvResBlock(h1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2_fuse(torch.cat([self.up2_conv(d2), e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.up1_fuse(torch.cat([self.up1_conv(d1), e1], dim=1))


class ARReceiver(nn.Module):
    def __init__(self, hidden: int = 160, depth: int = 4) -> None:
        super().__init__()
        self.pred1 = GroupPredictor(5, hidden, depth)
        self.pred2 = GroupPredictor(9, hidden, depth)
        self.pred3 = GroupPredictor(13, hidden, depth)

    def forward(
        self,
        z0_rx: torch.Tensor,
        *,
        y4_raw: torch.Tensor | None = None,
        y4_norm: torch.Tensor | None = None,
        scale: torch.Tensor,
        snr_db: torch.Tensor | None = None,
        z_gt: torch.Tensor | None = None,
        teacher_prob: float = 0.0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h, w = z0_rx.shape[-2:]
        scale_map = scale.float().log().view(-1, 1, 1, 1).expand(-1, 1, h, w).to(dtype=z0_rx.dtype)
        z1 = self.pred1(torch.cat([z0_rx, scale_map], dim=1))
        p = float(teacher_prob)
        z1_cond = z1
        if z_gt is not None and p > 0:
            z1_cond = p * z_gt[:, 4:8].to(dtype=z1.dtype) + (1.0 - p) * z1
        z2 = self.pred2(torch.cat([z0_rx, z1_cond, scale_map], dim=1))
        z2_cond = z2
        if z_gt is not None and p > 0:
            z2_cond = p * z_gt[:, 8:12].to(dtype=z2.dtype) + (1.0 - p) * z2
        z3 = self.pred3(torch.cat([z0_rx, z1_cond, z2_cond, scale_map], dim=1))
        return torch.cat([z0_rx, z1, z2, z3], dim=1), (z1, z2, z3)


class ARUNetReceiver(nn.Module):
    """Shared small U-Net plus progressive AR heads for z1/z2/z3."""

    def __init__(self, base: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.backbone = SmallLatentUNet(in_channels=14, base=base, depth=depth)
        self.head1 = ARHead(base, base, 4)
        self.head2 = ARHead(base + 4, base, 4)
        self.head3 = ARHead(base + 8, base, 4)

    def forward(
        self,
        z0_rx: torch.Tensor,
        *,
        y4_raw: torch.Tensor | None = None,
        y4_norm: torch.Tensor | None = None,
        scale: torch.Tensor,
        snr_db: torch.Tensor | None = None,
        z_gt: torch.Tensor | None = None,
        teacher_prob: float = 0.0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h, w = z0_rx.shape[-2:]
        if y4_raw is None:
            y4_raw = z0_rx
        if y4_norm is None:
            y4_norm = z0_rx
        if snr_db is None:
            snr_db = torch.full((z0_rx.shape[0],), 12.0, device=z0_rx.device, dtype=torch.float32)
        scale_map = scale.float().log().view(-1, 1, 1, 1).expand(-1, 1, h, w).to(dtype=z0_rx.dtype)
        snr_map = (snr_db.float().view(-1, 1, 1, 1) / 20.0).expand(-1, 1, h, w).to(dtype=z0_rx.dtype)
        cond = torch.cat([z0_rx, y4_raw.to(dtype=z0_rx.dtype), y4_norm.to(dtype=z0_rx.dtype), scale_map, snr_map], dim=1)
        shared = self.backbone(cond)
        z1 = self.head1(shared)
        p = float(teacher_prob)
        z1_cond = z1
        if z_gt is not None and p > 0:
            z1_cond = p * z_gt[:, 4:8].to(dtype=z1.dtype) + (1.0 - p) * z1
        z2 = self.head2(torch.cat([shared, z1_cond], dim=1))
        z2_cond = z2
        if z_gt is not None and p > 0:
            z2_cond = p * z_gt[:, 8:12].to(dtype=z2.dtype) + (1.0 - p) * z2
        z3 = self.head3(torch.cat([shared, z1_cond, z2_cond], dim=1))
        return torch.cat([z0_rx, z1, z2, z3], dim=1), (z1, z2, z3)


def teacher_force_prob(epoch: int, full_epochs: int, decay_epochs: int) -> float:
    if int(full_epochs) <= 0 and int(decay_epochs) <= 0:
        return 0.0
    if epoch <= int(full_epochs):
        return 1.0
    if int(decay_epochs) <= 0:
        return 0.0
    t = (epoch - int(full_epochs)) / float(max(1, int(decay_epochs)))
    return max(0.0, 1.0 - t)


def charbonnier_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((x.float() - y.float()).square() + float(eps) ** 2).mean()


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Hierarchical Swin AR AWGN12 session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train hierarchical Swin latent and AR receiver at AWGN12",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", action="store_false", dest="cache_decoded")
    p.add_argument("--cache_workers", type=int, default=12)
    p.add_argument("--prefetch_factor", type=int, default=4)

    p.add_argument("--init_sc_encoder_ckpt", type=str, default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_encoder_div2k_c16.pth")
    p.add_argument("--init_sc_decoder_ckpt", type=str, default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_decoder_div2k_c16.pth")
    p.add_argument("--init_hier_ckpt", type=str, default="", help="Optional checkpoint from a previous hierarchical stage")
    p.add_argument(
        "--stage",
        type=str,
        default="joint",
        choices=[
            "joint",
            "hierarchical_swin_pretrain",
            "hierarchical_swin_pretrain_ar_predictable",
            "hierarchical_group_ar_frozen_swin",
            "hierarchical_group_ar_decoder_tune",
            "hierarchical_group_ar_joint_tune",
        ],
    )
    p.add_argument("--snr_db", type=float, default=12.0)
    p.add_argument("--baseline_psnr", type=float, default=22.419)

    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--decoder_lr", type=float, default=0.0, help="Stage-specific decoder LR; <=0 uses --lr")
    p.add_argument("--ar_lr", type=float, default=0.0, help="Stage-specific AR LR; <=0 uses --lr * --ar_lr_mult")
    p.add_argument("--ar_lr_mult", type=float, default=2.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--ar_arch", type=str, default="simple", choices=["simple", "arunet"])
    p.add_argument("--ar_hidden", type=int, default=160)
    p.add_argument("--ar_depth", type=int, default=4)
    p.add_argument("--teacher_force_epochs", type=int, default=50)
    p.add_argument("--teacher_decay_epochs", type=int, default=100)

    p.add_argument("--lambda_recv", type=float, default=1.0)
    p.add_argument("--lambda_full", type=float, default=1.0)
    p.add_argument("--lambda_low", type=float, default=0.5)
    p.add_argument("--lambda_mid", type=float, default=0.4)
    p.add_argument("--lambda_midhigh", type=float, default=0.3)
    p.add_argument("--lambda_ar", type=float, default=0.1)
    p.add_argument("--lambda_ar_img", type=float, default=0.5)
    p.add_argument("--lambda_energy", type=float, default=1e-4)
    p.add_argument("--lambda_order", type=float, default=1e-3)
    p.add_argument("--lambda_kl", type=float, default=1e-6)
    p.add_argument("--recv_mse_weight", type=float, default=0.8)
    p.add_argument("--recv_charb_weight", type=float, default=0.2)
    p.add_argument("--charb_eps", type=float, default=1e-3)
    p.add_argument("--blur_low_sigma", type=float, default=4.0)
    p.add_argument("--blur_mid_sigma", type=float, default=2.0)
    p.add_argument("--blur_midhigh_sigma", type=float, default=1.0)
    p.add_argument("--blur_kernel", type=int, default=15)

    p.add_argument("--eval_every_epochs", type=int, default=2)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260524)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints-ar/hierarchical_swin_ar_awgn12")
    p.add_argument("--log_file", type=str, default="checkpoints-ar/hierarchical_swin_ar_awgn12/train.log")
    return p.parse_args()


def build_semantic_modules(device: torch.device) -> tuple[SemanticEncoder, SemanticDecoder, object]:
    cfg = get_div2k_config()
    cfg.semantic.embed_dim = 16
    cfg.semantic.use_vae = True
    cfg.semantic.lambda_kl = 1e-6
    cfg.channel.input_channels = 16
    cfg.channel.channel_symbols = 4
    cfg.unet_uncond.input_channel = 16
    sc = cfg.semantic
    enc = SemanticEncoder(
        in_channels=sc.image_channels,
        embed_dim=sc.embed_dim,
        patch_size=sc.patch_size,
        num_heads=sc.num_heads,
        window_size=sc.window_size,
        num_blocks=sc.num_swin_blocks,
        stage_embed_dims=sc.stage_embed_dims,
        stage_depths=sc.stage_depths,
        stage_num_heads=sc.stage_num_heads,
        stem_stride=sc.stem_stride,
        stage_downsample=sc.stage_downsample,
        use_vae=True,
    ).to(device)
    dec = SemanticDecoder(
        out_channels=sc.image_channels,
        embed_dim=sc.embed_dim,
        patch_size=sc.patch_size,
        num_heads=sc.num_heads,
        window_size=sc.window_size,
        num_refine_blocks=sc.num_decoder_refine_blocks,
        stage_embed_dims=sc.stage_embed_dims,
        stage_depths=sc.stage_depths,
        stage_num_heads=sc.stage_num_heads,
        stem_stride=sc.stem_stride,
        stage_downsample=sc.stage_downsample,
    ).to(device)
    return enc, dec, cfg


def fixed_select_a(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    a = torch.zeros(4, 16, device=device, dtype=dtype)
    a[:, :4] = torch.eye(4, device=device, dtype=dtype)
    return a


def encode_a(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.einsum("oc,bchw->bohw", a.to(device=z.device, dtype=z.dtype), z)


def decode_a(y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.einsum("oc,bohw->bchw", a.to(device=y.device, dtype=y.dtype), y)


def keep_groups(z: torch.Tensor, num_groups: int) -> torch.Tensor:
    out = torch.zeros_like(z)
    out[:, : 4 * int(num_groups)] = z[:, : 4 * int(num_groups)]
    return out


def order_loss(z: torch.Tensor) -> torch.Tensor:
    e = [z[:, i * 4 : (i + 1) * 4].float().square().mean() for i in range(4)]
    return F.relu(e[1] - e[0]) + F.relu(e[2] - e[1]) + F.relu(e[3] - e[2])


def make_loaders(args: argparse.Namespace, device: torch.device):
    train_ds = DIV2KDataset(
        args.data_dir,
        crop_size=args.crop_size,
        split="train",
        cache_decoded=bool(args.cache_decoded),
        cache_workers=int(args.cache_workers),
    )
    val_ds = DIV2KDataset(
        args.data_dir,
        crop_size=args.crop_size,
        split="valid",
        cache_decoded=bool(args.cache_decoded),
        cache_workers=int(args.cache_workers),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if int(args.num_workers) > 0 else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.val_num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.val_num_workers) > 0,
        prefetch_factor=(max(2, int(args.prefetch_factor)) if int(args.val_num_workers) > 0 else None),
    )
    return train_ds, val_ds, train_loader, val_loader


def save_checkpoint(path: str, encoder: nn.Module, decoder: nn.Module, ar: nn.Module, a: torch.Tensor, cfg, args: argparse.Namespace, metrics: dict, epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    common = {
        "route": "hierarchical_swin_ar_awgn12",
        "stage": str(args.stage),
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "semantic_config": asdict(cfg.semantic),
        "a_matrix": a.detach().cpu(),
        "aat": (a @ a.t()).detach().cpu(),
        "aat_error": semiorth_error(a),
        "fixed_channel_codec": True,
        "channel_encoder": "A=[I4,0]",
        "channel_decoder": "A^T zero-fill",
        "power_norm_after_channel_encoder": True,
        "snr_db": float(args.snr_db),
        "trainable_encoder": any(p.requires_grad for p in encoder.parameters()),
        "trainable_decoder": any(p.requires_grad for p in decoder.parameters()),
        "trainable_ar": any(p.requires_grad for p in ar.parameters()),
    }
    payload = {
        **common,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "ar_state_dict": ar.state_dict(),
    }
    torch.save(payload, path)
    split_dir = os.path.dirname(path)
    torch.save({**common, "part": "semantic_encoder", "state_dict": encoder.state_dict()}, os.path.join(split_dir, "sc_encoder_hier_c16.pth"))
    torch.save({**common, "part": "semantic_decoder", "state_dict": decoder.state_dict()}, os.path.join(split_dir, "sc_decoder_hier_c16.pth"))
    torch.save({**common, "part": "ar_receiver", "state_dict": ar.state_dict()}, os.path.join(split_dir, "ar_receiver_hier_c16_awgn12.pth"))


def load_hier_checkpoint(path: str, encoder: nn.Module, decoder: nn.Module, ar: nn.Module, device: torch.device) -> None:
    if not path:
        return
    ckpt_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    elif ckpt.get("part") == "semantic_encoder":
        encoder.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        raise KeyError(f"no encoder_state_dict in {ckpt_path}")
    if "decoder_state_dict" in ckpt:
        decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    elif ckpt.get("part") == "semantic_decoder":
        decoder.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        raise KeyError(f"no decoder_state_dict in {ckpt_path}")
    if "ar_state_dict" in ckpt:
        try:
            ar.load_state_dict(ckpt["ar_state_dict"], strict=True)
        except RuntimeError as exc:
            print(f"warning: skipped AR state from {ckpt_path}: {exc}")
    print(f"loaded hierarchical checkpoint: {ckpt_path}, stage={ckpt.get('stage', ckpt.get('route', 'unknown'))}, epoch={ckpt.get('epoch', 'unknown')}")


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(bool(trainable))


def stage_trainable(stage: str) -> tuple[bool, bool, bool]:
    if stage == "hierarchical_swin_pretrain":
        return True, True, False
    if stage == "hierarchical_swin_pretrain_ar_predictable":
        return True, True, True
    if stage == "hierarchical_group_ar_frozen_swin":
        return False, False, True
    if stage == "hierarchical_group_ar_decoder_tune":
        return False, True, False
    if stage == "hierarchical_group_ar_joint_tune":
        return False, True, True
    return True, True, True


def set_stage_mode(encoder: nn.Module, decoder: nn.Module, ar: nn.Module, stage: str, train: bool) -> None:
    enc_trainable, dec_trainable, ar_trainable = stage_trainable(stage)
    encoder.train(train and enc_trainable)
    decoder.train(train and dec_trainable)
    ar.train(train and ar_trainable)


def build_optimizer(args: argparse.Namespace, encoder: nn.Module, decoder: nn.Module, ar: nn.Module) -> optim.Optimizer:
    groups = []
    dec_lr = float(args.decoder_lr) if float(args.decoder_lr) > 0 else float(args.lr)
    ar_lr = float(args.ar_lr) if float(args.ar_lr) > 0 else float(args.lr) * float(args.ar_lr_mult)
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    dec_params = [p for p in decoder.parameters() if p.requires_grad]
    ar_params = [p for p in ar.parameters() if p.requires_grad]
    if enc_params:
        groups.append({"params": enc_params, "lr": float(args.lr), "name": "encoder"})
    if dec_params:
        groups.append({"params": dec_params, "lr": dec_lr, "name": "decoder"})
    if ar_params:
        groups.append({"params": ar_params, "lr": ar_lr, "name": "ar"})
    if not groups:
        raise RuntimeError(f"stage={args.stage} has no trainable parameters")
    print("optimizer_groups=" + ", ".join(f"{g['name']}:n={sum(p.numel() for p in g['params'])}:lr={g['lr']}" for g in groups))
    return optim.AdamW(groups, weight_decay=float(args.weight_decay), betas=(0.9, 0.999))


def run_batch(
    *,
    imgs: torch.Tensor,
    encoder: SemanticEncoder,
    decoder: SemanticDecoder,
    ar: nn.Module,
    a: torch.Tensor,
    blurs: tuple[GaussianBlur, GaussianBlur, GaussianBlur],
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator | None,
    train: bool,
    teacher_prob: float,
    sample_latent: bool,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    device = imgs.device
    bsz = imgs.shape[0]
    snr_b = torch.full((bsz,), float(args.snr_db), device=device, dtype=torch.float32)
    blur_low, blur_mid, blur_midhigh = blurs

    with make_autocast(device, amp_enabled, amp_dtype):
        z, mu, logvar = encoder.encode(imgs, sample=(train and sample_latent))
        z = z.float()
        x_low_t = blur_low(imgs).float().clamp(0, 1)
        x_mid_t = blur_mid(imgs).float().clamp(0, 1)
        x_mh_t = blur_midhigh(imgs).float().clamp(0, 1)

        y4 = encode_a(z, a)
        y4_norm, y4_raw, scale = power_normalize_awgn(y4, snr_b, generator=generator)
        z0_rx = y4_raw.float()
        aux_clean_ar = args.stage == "hierarchical_swin_pretrain_ar_predictable"
        if aux_clean_ar:
            ar_z0 = y4.float()
            ar_y4_raw = y4.float()
            ar_y4_norm = y4.float()
            ar_scale = torch.ones_like(scale)
        else:
            ar_z0 = z0_rx
            ar_y4_raw = y4_raw
            ar_y4_norm = y4_norm
            ar_scale = scale
        ar_dtype = next(ar.parameters()).dtype
        z_ar, pred_groups = ar(
            ar_z0.to(dtype=ar_dtype),
            y4_raw=ar_y4_raw.to(dtype=ar_dtype),
            y4_norm=ar_y4_norm.to(dtype=ar_dtype),
            scale=ar_scale,
            snr_db=snr_b,
            z_gt=(z if train else None),
            teacher_prob=(teacher_prob if train else 0.0),
        )
        z_base = decode_a(y4_raw.float(), a)

        x0 = decoder(keep_groups(z, 1)).float().clamp(0, 1)
        x1 = decoder(keep_groups(z, 2)).float().clamp(0, 1)
        x2 = decoder(keep_groups(z, 3)).float().clamp(0, 1)
        x_full = decoder(z).float().clamp(0, 1)
        x_recv = decoder(z_ar.float()).float().clamp(0, 1)

        loss_low = F.mse_loss(x0, x_low_t)
        loss_mid = F.mse_loss(x1, x_mid_t)
        loss_mh = F.mse_loss(x2, x_mh_t)
        loss_full = F.mse_loss(x_full, imgs.float())
        loss_recv_mse = F.mse_loss(x_recv, imgs.float())
        loss_recv_charb = charbonnier_loss(x_recv, imgs, eps=float(args.charb_eps))
        loss_recv = float(args.recv_mse_weight) * loss_recv_mse + float(args.recv_charb_weight) * loss_recv_charb
        z_det = z.detach()
        loss_ar = (
            F.mse_loss(pred_groups[0].float(), z_det[:, 4:8])
            + F.mse_loss(pred_groups[1].float(), z_det[:, 8:12])
            + F.mse_loss(pred_groups[2].float(), z_det[:, 12:16])
        ) / 3.0
        loss_energy = z.float().square().mean()
        loss_ord = order_loss(z)
        if mu is not None and logvar is not None and float(args.lambda_kl) > 0:
            loss_kl = kl_loss(mu.float(), logvar.float()).to(dtype=loss_full.dtype)
        else:
            loss_kl = loss_full.new_tensor(0.0)
        if args.stage == "hierarchical_swin_pretrain":
            loss = (
                float(args.lambda_full) * loss_full
                + float(args.lambda_low) * loss_low
                + float(args.lambda_mid) * loss_mid
                + float(args.lambda_midhigh) * loss_mh
                + float(args.lambda_energy) * loss_energy
                + float(args.lambda_order) * loss_ord
                + float(args.lambda_kl) * loss_kl
            )
        elif args.stage == "hierarchical_swin_pretrain_ar_predictable":
            loss = (
                float(args.lambda_full) * loss_full
                + float(args.lambda_low) * loss_low
                + float(args.lambda_mid) * loss_mid
                + float(args.lambda_midhigh) * loss_mh
                + float(args.lambda_ar) * loss_ar
                + float(args.lambda_ar_img) * loss_recv
                + float(args.lambda_energy) * loss_energy
                + float(args.lambda_order) * loss_ord
                + float(args.lambda_kl) * loss_kl
            )
        elif args.stage == "hierarchical_group_ar_frozen_swin":
            loss = float(args.lambda_recv) * loss_recv + float(args.lambda_ar) * loss_ar
        elif args.stage == "hierarchical_group_ar_decoder_tune":
            loss = float(args.lambda_recv) * loss_recv + float(args.lambda_full) * loss_full
        elif args.stage == "hierarchical_group_ar_joint_tune":
            loss = (
                float(args.lambda_recv) * loss_recv
                + float(args.lambda_full) * loss_full
                + float(args.lambda_low) * loss_low
                + float(args.lambda_mid) * loss_mid
                + float(args.lambda_midhigh) * loss_mh
                + float(args.lambda_ar) * loss_ar
            )
        else:
            loss = (
                float(args.lambda_recv) * loss_recv
                + float(args.lambda_full) * loss_full
                + float(args.lambda_low) * loss_low
                + float(args.lambda_mid) * loss_mid
                + float(args.lambda_midhigh) * loss_mh
                + float(args.lambda_ar) * loss_ar
                + float(args.lambda_energy) * loss_energy
                + float(args.lambda_order) * loss_ord
                + float(args.lambda_kl) * loss_kl
            )

    stats = {
        "loss": float(loss.detach().item()),
        "loss_recv": float(loss_recv.detach().item()),
        "loss_recv_mse": float(loss_recv_mse.detach().item()),
        "loss_recv_charb": float(loss_recv_charb.detach().item()),
        "loss_full": float(loss_full.detach().item()),
        "loss_low": float(loss_low.detach().item()),
        "loss_mid": float(loss_mid.detach().item()),
        "loss_midhigh": float(loss_mh.detach().item()),
        "loss_ar": float(loss_ar.detach().item()),
        "loss_energy": float(loss_energy.detach().item()),
        "loss_order": float(loss_ord.detach().item()),
        "loss_kl": float(loss_kl.detach().item()),
        "teacher_prob": float(teacher_prob if train else 0.0),
        "psnr_recv": float(psnr_per_image(x_recv, imgs.float()).mean().item()),
        "psnr_full": float(psnr_per_image(x_full, imgs.float()).mean().item()),
        "psnr_low": float(psnr_per_image(x0, x_low_t).mean().item()),
        "psnr_mid": float(psnr_per_image(x1, x_mid_t).mean().item()),
        "psnr_midhigh": float(psnr_per_image(x2, x_mh_t).mean().item()),
        "psnr_base": float(psnr_per_image(decoder(z_base.float()).float().clamp(0, 1), imgs.float()).mean().item()) if not train else 0.0,
    }
    return (loss if train else None), stats


def main() -> None:
    args = parse_args()
    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(PROJECT_ROOT, args.save_dir)
    log_file = args.log_file if os.path.isabs(args.log_file) else os.path.join(PROJECT_ROOT, args.log_file)
    os.makedirs(save_dir, exist_ok=True)
    setup_log_file(log_file)
    seed_everything(int(args.seed))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if device.type != "cuda":
        amp_enabled = False

    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}, stage={args.stage}")
    print("rule2: PSNR=mean(per-image PSNR), train/test SNR=12 dB, A=[I4,0], A A^T=I_4, power norm after channel encoder")
    train_ds, val_ds, train_loader, val_loader = make_loaders(args, device)
    print(
        f"train={len(train_ds)} valid={len(val_ds)} input=[3,{args.crop_size},{args.crop_size}] "
        "cache_decoded=full images; train crops are dynamic"
    )

    encoder, decoder, cfg = build_semantic_modules(device)
    load_state_dict_from_ckpt(encoder, args.init_sc_encoder_ckpt, "semantic_encoder")
    load_state_dict_from_ckpt(decoder, args.init_sc_decoder_ckpt, "semantic_decoder")
    if str(args.ar_arch) == "arunet":
        ar = ARUNetReceiver(base=int(args.ar_hidden), depth=int(args.ar_depth)).to(device)
    else:
        ar = ARReceiver(hidden=int(args.ar_hidden), depth=int(args.ar_depth)).to(device)
    load_hier_checkpoint(args.init_hier_ckpt, encoder, decoder, ar, device)

    enc_trainable, dec_trainable, ar_trainable = stage_trainable(str(args.stage))
    set_trainable(encoder, enc_trainable)
    set_trainable(decoder, dec_trainable)
    set_trainable(ar, ar_trainable)
    sample_latent = bool(enc_trainable)
    a = fixed_select_a(device=device, dtype=torch.float32)
    aat_err = semiorth_error(a)
    if aat_err > 1e-7:
        raise RuntimeError(f"A A^T != I4, err={aat_err:.3e}")
    print(f"channel_encoder_output=[4,16,16], A_shape={tuple(a.shape)}, aat_error={aat_err:.3e}")
    print(
        f"ar_arch={args.ar_arch} hidden={args.ar_hidden} depth={args.ar_depth}; "
        f"teacher_forcing=1.0 for {args.teacher_force_epochs} epochs then linear decay over {args.teacher_decay_epochs}; "
        f"recv_loss={args.recv_mse_weight}*mse+{args.recv_charb_weight}*charb"
    )
    print(
        f"trainable: encoder={enc_trainable} decoder={dec_trainable} ar={ar_trainable}; "
        f"latent_sample_train={sample_latent}"
    )

    blurs = (
        GaussianBlur(3, int(args.blur_kernel), float(args.blur_low_sigma)).to(device),
        GaussianBlur(3, int(args.blur_kernel), float(args.blur_mid_sigma)).to(device),
        GaussianBlur(3, int(args.blur_kernel), float(args.blur_midhigh_sigma)).to(device),
    )
    optimizer = build_optimizer(args, encoder, decoder, ar)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    best = -1.0
    score_metric = "val_psnr_full" if args.stage == "hierarchical_swin_pretrain" else "val_psnr_recv"
    ckpt_prefix = "hierarchical_swin_ar_awgn12" if args.stage == "joint" else str(args.stage)
    for epoch in range(1, int(args.epochs) + 1):
        tf_prob = teacher_force_prob(epoch, int(args.teacher_force_epochs), int(args.teacher_decay_epochs))
        set_stage_mode(encoder, decoder, ar, str(args.stage), train=True)
        meters = {k: AverageMeter() for k in (
            "loss", "loss_recv", "loss_full", "loss_low", "loss_mid", "loss_midhigh", "loss_ar",
            "psnr_recv", "psnr_full", "psnr_low", "psnr_mid", "psnr_midhigh"
        )}
        for bi, batch in enumerate(train_loader):
            if int(args.max_train_batches) > 0 and bi >= int(args.max_train_batches):
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, stats = run_batch(
                imgs=imgs,
                encoder=encoder,
                decoder=decoder,
                ar=ar,
                a=a,
                blurs=blurs,
                args=args,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                generator=None,
                train=True,
                teacher_prob=tf_prob,
                sample_latent=sample_latent,
            )
            assert loss is not None
            scaler.scale(loss).backward()
            if float(args.clip_grad_norm) > 0:
                scaler.unscale_(optimizer)
                trainable_params = [p for group in optimizer.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.clip_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            for k in meters:
                meters[k].update(stats[k], imgs.shape[0])

        metrics = {f"train_{k}": v.avg for k, v in meters.items()}
        do_eval = epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs)
        if do_eval:
            set_stage_mode(encoder, decoder, ar, str(args.stage), train=False)
            val_meters = {k: AverageMeter() for k in (
                "loss", "loss_recv", "loss_full", "loss_low", "loss_mid", "loss_midhigh", "loss_ar",
                "psnr_recv", "psnr_full", "psnr_low", "psnr_mid", "psnr_midhigh", "psnr_base"
            )}
            gen = torch.Generator(device=device)
            gen.manual_seed(int(args.seed) + 1000)
            with torch.no_grad():
                for bi, batch in enumerate(val_loader):
                    if int(args.max_val_batches) > 0 and bi >= int(args.max_val_batches):
                        break
                    imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
                    imgs = imgs.to(device, non_blocking=True)
                    _loss, stats = run_batch(
                        imgs=imgs,
                        encoder=encoder,
                        decoder=decoder,
                        ar=ar,
                        a=a,
                        blurs=blurs,
                        args=args,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        generator=gen,
                        train=False,
                        teacher_prob=0.0,
                        sample_latent=False,
                    )
                    for k in val_meters:
                        val_meters[k].update(stats[k], imgs.shape[0])
            metrics.update({f"val_{k}": v.avg for k, v in val_meters.items()})
            score = metrics[score_metric]
            is_best = score > best
            if is_best:
                best = score
                save_checkpoint(os.path.join(save_dir, f"{ckpt_prefix}_best.pth"), encoder, decoder, ar, a, cfg, args, metrics, epoch)
            save_checkpoint(os.path.join(save_dir, f"{ckpt_prefix}_latest.pth"), encoder, decoder, ar, a, cfg, args, metrics, epoch)
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} recv_loss={meters['loss_recv'].avg:.6f} "
                f"| base={metrics['val_psnr_base']:.4f} recv={metrics['val_psnr_recv']:.4f} full={metrics['val_psnr_full']:.4f} "
                f"low={metrics['val_psnr_low']:.4f} mid={metrics['val_psnr_mid']:.4f} mh={metrics['val_psnr_midhigh']:.4f} "
                f"score({score_metric})={score:.4f} gain_baseline={metrics['val_psnr_recv'] - float(args.baseline_psnr):+.4f} "
                f"tf={tf_prob:.3f} aat_err={semiorth_error(a):.2e} "
                f"{'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"recv={meters['psnr_recv'].avg:.4f} full={meters['psnr_full'].avg:.4f} tf={tf_prob:.3f} aat_err={semiorth_error(a):.2e}"
            )
    print(f"best_{score_metric}={best:.4f} target_recv>{float(args.baseline_psnr) + 0.5:.4f}")


if __name__ == "__main__":
    main()
