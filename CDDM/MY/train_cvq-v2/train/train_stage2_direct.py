from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
CDDM_ROOT = PARENT_DIR.parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))

from Autoencoder.net.encoder import SwinTransformerBlock

from shared import (
    add_common_cli,
    averaged,
    batch_metric_mean,
    build_encoder_decoder,
    ckpt_path,
    cvq_io,
    default_stage1_ckpt,
    default_v01_save_dir,
    ensure_common_args,
    format_metrics,
    freeze_module,
    get_loader,
    load_v01_checkpoint,
    meters,
    print_epoch,
    psnr_per_image,
    real_awgn,
    recon_loss,
    resolve_path,
    save_v01_checkpoint,
    seed_everything,
    setup_stage_log,
    should_save_latest,
    should_validate,
    split_c1_c2,
    with_log_keys,
    write_json,
)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(channels), int(channels), 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(int(channels), int(channels), 3, padding=1),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SingleChannelC2Prior(nn.Module):
    """One independent C1_rx -> one C2-channel Gaussian prior."""

    def __init__(
        self,
        c1_ch: int,
        hidden: int,
        depth: int,
        dropout: float,
        init_sigma: float,
        logvar_min: float,
        logvar_max: float,
    ) -> None:
        super().__init__()
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.stem = nn.Sequential(
            nn.Conv2d(int(c1_ch), int(hidden), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden), int(hidden), 3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualConvBlock(int(hidden), float(dropout)) for _ in range(int(depth))])
        self.mu_head = nn.Conv2d(int(hidden), 1, 3, padding=1)
        self.logvar_head = nn.Conv2d(int(hidden), 1, 3, padding=1)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, math.log(max(float(init_sigma), 1e-6) ** 2))

    def forward(self, c1_rx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.blocks(self.stem(c1_rx))
        mu = self.mu_head(feat)
        logvar = self.logvar_head(feat).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar


class SingleChannelSwinC2Prior(nn.Module):
    """One independent Swin prior for one C2 channel on the 16x16 latent grid."""

    def __init__(
        self,
        c1_ch: int,
        hidden: int,
        depth: int,
        heads: int,
        window_size: int,
        mlp_ratio: float,
        latent_h: int,
        latent_w: int,
        init_sigma: float,
        logvar_min: float,
        logvar_max: float,
    ) -> None:
        super().__init__()
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.stem = nn.Sequential(
            nn.Conv2d(int(c1_ch), int(hidden), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden), int(hidden), 3, padding=1),
        )
        self.pos = nn.Parameter(torch.zeros(1, self.latent_h * self.latent_w, int(hidden)))
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=int(hidden),
                    input_resolution=(self.latent_h, self.latent_w),
                    num_heads=int(heads),
                    window_size=int(window_size),
                    shift_size=0 if i % 2 == 0 else int(window_size) // 2,
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(hidden))
        self.mu_head = nn.Linear(int(hidden), 1)
        self.logvar_head = nn.Linear(int(hidden), 1)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, math.log(max(float(init_sigma), 1e-6) ** 2))

    def forward(self, c1_rx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = int(c1_rx.shape[0])
        tokens = self.stem(c1_rx).flatten(2).transpose(1, 2) + self.pos
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        mu = self.mu_head(tokens).transpose(1, 2).reshape(bsz, 1, self.latent_h, self.latent_w)
        logvar = self.logvar_head(tokens).transpose(1, 2).reshape(bsz, 1, self.latent_h, self.latent_w)
        return mu, logvar.clamp(self.logvar_min, self.logvar_max)


class IndependentChannelC2Prior(nn.Module):
    """Twenty separate-weight predictors instead of one shared multi-channel head."""

    def __init__(
        self,
        arch: str = "swin",
        c1_ch: int = 16,
        c2_ch: int = 20,
        hidden: int = 64,
        depth: int = 4,
        heads: int = 4,
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        latent_h: int = 16,
        latent_w: int = 16,
        dropout: float = 0.0,
        init_sigma: float = 1.0,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.arch = str(arch).lower()
        self.c2_ch = int(c2_ch)
        models = []
        for _ in range(self.c2_ch):
            if self.arch == "cnn":
                models.append(
                    SingleChannelC2Prior(
                        c1_ch=int(c1_ch),
                        hidden=int(hidden),
                        depth=int(depth),
                        dropout=float(dropout),
                        init_sigma=float(init_sigma),
                        logvar_min=float(logvar_min),
                        logvar_max=float(logvar_max),
                    )
                )
            elif self.arch == "swin":
                models.append(
                    SingleChannelSwinC2Prior(
                        c1_ch=int(c1_ch),
                        hidden=int(hidden),
                        depth=int(depth),
                        heads=int(heads),
                        window_size=int(window_size),
                        mlp_ratio=float(mlp_ratio),
                        latent_h=int(latent_h),
                        latent_w=int(latent_w),
                        init_sigma=float(init_sigma),
                        logvar_min=float(logvar_min),
                        logvar_max=float(logvar_max),
                    )
                )
            else:
                raise ValueError(f"unknown direct prior arch: {arch}")
        self.models = nn.ModuleList(models)

    def forward(self, c1_rx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mus: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []
        for model in self.models:
            mu, logvar = model(c1_rx)
            mus.append(mu)
            logvars.append(logvar)
        return torch.cat(mus, dim=1), torch.cat(logvars, dim=1)


def build_direct_prior(args: argparse.Namespace, device: torch.device) -> nn.Module:
    return IndependentChannelC2Prior(
        arch=str(args.direct_arch),
        c1_ch=int(args.c1_ch),
        c2_ch=int(args.latent_ch) - int(args.c1_ch),
        hidden=int(args.direct_hidden),
        depth=int(args.direct_depth),
        heads=int(args.direct_heads),
        window_size=int(args.direct_window_size),
        mlp_ratio=float(args.direct_mlp_ratio),
        latent_h=int(args.latent_h),
        latent_w=int(args.latent_w),
        dropout=float(args.direct_dropout),
        init_sigma=float(args.direct_init_sigma),
        logvar_min=float(args.direct_logvar_min),
        logvar_max=float(args.direct_logvar_max),
    ).to(device)


def load_stage1(args: argparse.Namespace, encoder: nn.Module, decoder: nn.Module) -> None:
    src = args.init_stage1_ckpt or default_stage1_ckpt(args)
    obj = load_v01_checkpoint(src)
    encoder.load_state_dict(obj["encoder_state_dict"], strict=True)
    decoder.load_state_dict(obj["decoder_state_dict"], strict=True)
    print(f"stage2_direct_source_stage1={resolve_path(src)}")


def gaussian_nll(mu: torch.Tensor, target: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    err2 = (target.float() - mu.float()).square()
    return 0.5 * (err2 * torch.exp(-logvar.float()) + logvar.float()).mean()


def sigma_from_logvar(logvar: torch.Tensor) -> torch.Tensor:
    return torch.exp(0.5 * logvar.float()).to(dtype=logvar.dtype)


def uncertainty_soft_mask(logvar: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    threshold = float(args.uncert_threshold)
    if threshold <= 0.0:
        return torch.ones_like(logvar)
    sigma = sigma_from_logvar(logvar)
    if str(args.uncert_granularity) == "channel":
        sigma = sigma.mean(dim=(2, 3), keepdim=True)
    return torch.sigmoid((threshold - sigma) * float(args.uncert_gate_temp)).to(dtype=logvar.dtype)


@torch.no_grad()
def uncertainty_hard_mask(logvar: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    threshold = float(args.uncert_threshold)
    if threshold <= 0.0:
        return torch.ones_like(logvar)
    sigma = sigma_from_logvar(logvar)
    if str(args.uncert_granularity) == "channel":
        sigma = sigma.mean(dim=(2, 3), keepdim=True)
    return (sigma <= threshold).to(dtype=logvar.dtype).expand_as(logvar)


def direct_losses(
    imgs: torch.Tensor,
    c1_rx: torch.Tensor,
    c2: torch.Tensor,
    decoder: nn.Module,
    prior: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    mu, logvar = prior(c1_rx)
    soft_mask = uncertainty_soft_mask(logvar, args)
    zero = torch.zeros_like(c2)
    x_c1 = decoder(torch.cat([c1_rx, zero], dim=1))
    x_pred = decoder(torch.cat([c1_rx, mu], dim=1))
    x_uncert_soft = decoder(torch.cat([c1_rx, mu * soft_mask], dim=1))
    loss_c1 = recon_loss(x_c1, imgs)
    loss_pred = recon_loss(x_pred, imgs)
    loss_uncert = recon_loss(x_uncert_soft, imgs)
    loss_latent = F.mse_loss(mu.float(), c2.float())
    loss_nll = gaussian_nll(mu, c2, logvar)
    sigma = sigma_from_logvar(logvar)
    loss_sigma_calib = F.l1_loss(sigma.float(), (mu.detach().float() - c2.float()).abs())
    loss = (
        float(args.lambda_pred_rec) * loss_pred
        + float(args.lambda_uncert_rec) * loss_uncert
        + float(args.lambda_latent) * loss_latent
        + float(args.lambda_nll) * loss_nll
        + float(args.lambda_sigma_calib) * loss_sigma_calib
        + float(args.lambda_c1) * loss_c1
    )
    losses = {
        "loss_c1_rec": loss_c1,
        "loss_pred_rec": loss_pred,
        "loss_uncert_rec": loss_uncert,
        "loss_latent": loss_latent,
        "loss_nll": loss_nll,
        "loss_sigma_calib": loss_sigma_calib,
    }
    outputs = {
        "mu": mu,
        "logvar": logvar,
        "soft_mask": soft_mask,
        "x_c1": x_c1,
        "x_pred": x_pred,
        "x_uncert_soft": x_uncert_soft,
    }
    return loss, losses, outputs


def update_direct_metrics(
    m: dict,
    *,
    imgs: torch.Tensor,
    c1_rx: torch.Tensor,
    c2: torch.Tensor,
    decoder: nn.Module,
    loss: torch.Tensor,
    losses: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
    include_full: bool,
) -> None:
    bsz = int(imgs.shape[0])
    mu = outputs["mu"]
    logvar = outputs["logvar"]
    hard_mask = uncertainty_hard_mask(logvar, args)
    with torch.no_grad():
        x_c1 = outputs["x_c1"].clamp(0.0, 1.0)
        x_pred = outputs["x_pred"].clamp(0.0, 1.0)
        x_uncert_soft = outputs["x_uncert_soft"].clamp(0.0, 1.0)
        x_uncert = decoder(torch.cat([c1_rx, mu * hard_mask], dim=1)).clamp(0.0, 1.0)
        psnr_c1 = batch_metric_mean(psnr_per_image(x_c1, imgs))
        psnr_pred = batch_metric_mean(psnr_per_image(x_pred, imgs))
        psnr_uncert_soft = batch_metric_mean(psnr_per_image(x_uncert_soft, imgs))
        psnr_uncert = batch_metric_mean(psnr_per_image(x_uncert, imgs))
        sigma = sigma_from_logvar(logvar)
        keep = float(hard_mask.float().mean().item())
        soft_keep = float(outputs["soft_mask"].float().mean().item())
        abs_err = (mu.float() - c2.float()).abs()
        nll = gaussian_nll(mu, c2, logvar)

    m["loss"].update(float(loss.item()), bsz)
    for key, value in losses.items():
        m[key].update(float(value.item()), bsz)
    m["psnr_c1_only"].update(psnr_c1, bsz)
    m["psnr_pred_all"].update(psnr_pred, bsz)
    m["psnr_pred_uncert"].update(psnr_uncert, bsz)
    m["psnr_pred_uncert_soft"].update(psnr_uncert_soft, bsz)
    m["pred_gain"].update(psnr_pred - psnr_c1, bsz)
    m["pred_uncert_gain"].update(psnr_uncert - psnr_c1, bsz)
    m["pred_uncert_soft_gain"].update(psnr_uncert_soft - psnr_c1, bsz)
    m["c2_pred_mse"].update(float(F.mse_loss(mu.float(), c2.float()).item()), bsz)
    m["c2_pred_mae"].update(float(abs_err.mean().item()), bsz)
    m["c2_nll"].update(float(nll.item()), bsz)
    m["uncert_keep_ratio"].update(keep, bsz)
    m["uncert_soft_keep_ratio"].update(soft_keep, bsz)
    m["uncert_sigma_mean"].update(float(sigma.float().mean().item()), bsz)
    m["uncert_sigma_max"].update(float(sigma.float().amax().item()), bsz)
    m["uncert_logvar_mean"].update(float(logvar.float().mean().item()), bsz)
    if include_full:
        with torch.no_grad():
            x_real = decoder(torch.cat([c1_rx, c2], dim=1)).clamp(0.0, 1.0)
            m["psnr_real_c2_full"].update(batch_metric_mean(psnr_per_image(x_real, imgs)), bsz)


def metric_names(include_full: bool) -> list[str]:
    names = [
        "loss",
        "loss_c1_rec",
        "loss_pred_rec",
        "loss_uncert_rec",
        "loss_latent",
        "loss_nll",
        "loss_sigma_calib",
        "psnr_c1_only",
        "psnr_pred_all",
        "psnr_pred_uncert",
        "psnr_pred_uncert_soft",
        "pred_gain",
        "pred_uncert_gain",
        "pred_uncert_soft_gain",
        "c2_pred_mse",
        "c2_pred_mae",
        "c2_nll",
        "uncert_keep_ratio",
        "uncert_soft_keep_ratio",
        "uncert_sigma_mean",
        "uncert_sigma_max",
        "uncert_logvar_mean",
    ]
    if include_full:
        names.append("psnr_real_c2_full")
    return names


@torch.no_grad()
def validate(loader, encoder, decoder, prior, args) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    prior.eval()
    device = next(prior.parameters()).device
    m = meters(metric_names(include_full=True))
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        c1_rx = real_awgn(c1, float(args.snr_db))
        loss, losses, outputs = direct_losses(imgs, c1_rx, c2, decoder, prior, args)
        update_direct_metrics(m, imgs=imgs, c1_rx=c1_rx, c2=c2, decoder=decoder, loss=loss, losses=losses, outputs=outputs, args=args, include_full=True)
    return averaged(m)


def print_direct_header(args: argparse.Namespace, train_n: int, val_n: int) -> None:
    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    print("=== Stage 2 | Direct continuous C2 prior ===")
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"save_dir={resolve_path(args.save_dir)}")
    print("实验设计")
    print(f"  latent_ch={args.latent_ch} C1={args.c1_ch} C2={c2_ch} latent_hw={args.latent_h}x{args.latent_w} snr_db={args.snr_db:g}")
    print("  real inference path: C1_rx -> 20 independent continuous C2 priors -> uncertainty gate -> decoder")
    print("loss设计")
    print(
        "  "
        f"L={float(args.lambda_pred_rec):g}*pred_rec+"
        f"{float(args.lambda_uncert_rec):g}*uncert_rec+"
        f"{float(args.lambda_latent):g}*latent_mse+"
        f"{float(args.lambda_nll):g}*gaussian_nll+"
        f"{float(args.lambda_sigma_calib):g}*sigma_calib+"
        f"{float(args.lambda_c1):g}*c1_rec"
    )
    print("模块选择")
    print(
        "  "
        f"predictor=independent_channel_direct models={c2_ch} "
        f"arch={args.direct_arch} hidden={int(args.direct_hidden)} depth={int(args.direct_depth)} "
        f"heads={int(args.direct_heads)} window={int(args.direct_window_size)} dropout={float(args.direct_dropout):g} "
        f"uncert_granularity={args.uncert_granularity} threshold={float(args.uncert_threshold):g} temp={float(args.uncert_gate_temp):g}"
    )
    print(f"  train_encoder={bool(args.train_encoder)} train_decoder={bool(args.train_decoder)} quantizer=none gate=uncertainty")
    if getattr(args, "init_stage1_ckpt", ""):
        print(f"init_stage1_ckpt={resolve_path(args.init_stage1_ckpt)}")
    if getattr(args, "init_stage2_direct_ckpt", ""):
        print(f"init_stage2_direct_ckpt={resolve_path(args.init_stage2_direct_ckpt)}")
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}")


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder = build_encoder_decoder(args, cfg.device)
    prior = build_direct_prior(args, cfg.device)
    load_stage1(args, encoder, decoder)
    if args.init_stage2_direct_ckpt:
        obj = load_v01_checkpoint(args.init_stage2_direct_ckpt)
        prior.load_state_dict(obj["predictor_state_dict"], strict=True)

    freeze_module(encoder, bool(args.train_encoder))
    freeze_module(decoder, bool(args.train_decoder))
    params = list(prior.parameters())
    if bool(args.train_encoder):
        params += list(encoder.parameters())
    if bool(args.train_decoder):
        params += list(decoder.parameters())
    params = [p for p in params if p.requires_grad]
    opt = optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = -1.0

    print_direct_header(args, len(train_loader.dataset), len(val_loader.dataset))
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train(bool(args.train_encoder))
        decoder.train(bool(args.train_decoder))
        prior.train()
        m = meters(metric_names(include_full=False))
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            if bool(args.train_encoder):
                _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
            else:
                with torch.no_grad():
                    _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
                z_norm = z_norm.detach()
            c1, c2 = split_c1_c2(z_norm, args)
            c1_rx = real_awgn(c1, float(args.snr_db))
            loss, losses, outputs = direct_losses(imgs, c1_rx, c2, decoder, prior, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip_norm))
            opt.step()
            update_direct_metrics(m, imgs=imgs, c1_rx=c1_rx, c2=c2, decoder=decoder, loss=loss, losses=losses, outputs=outputs, args=args, include_full=False)

        metrics = averaged(m)
        print_epoch("stage2-direct", epoch, int(args.epochs), with_log_keys(metrics), time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, encoder, decoder, prior, args)
            score_key = str(args.stage2_direct_score)
            score = float(val_metrics[score_key])
            print(f"[stage2-direct val {epoch:03d}] {format_metrics(with_log_keys(val_metrics))} score={score:g} score_key={score_key}")
            if score > best:
                best = score
                save_v01_checkpoint(ckpt_path(args, "stage2", "best"), stage="stage2_direct", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, predictor=prior)
        if should_save_latest(args, epoch):
            save_v01_checkpoint(ckpt_path(args, "stage2", "latest"), stage="stage2_direct", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, predictor=prior)
    save_v01_checkpoint(ckpt_path(args, "stage2", "latest"), stage="stage2_direct", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, predictor=prior)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_cli(p, default_k=4096)
    p.set_defaults(save_dir=default_v01_save_dir(4096), epochs=100, k=4096)
    p.add_argument("--init-stage1-ckpt", type=str, default="MY/checkpoints-cvq-v2-v01-c36-snr9-k4096/cvq_v2_v01_c36_snr9_k4096_stage1_best.pth")
    p.add_argument("--init-stage2-direct-ckpt", type=str, default="")
    p.add_argument("--direct-arch", type=str, choices=["swin", "cnn"], default="swin")
    p.add_argument("--direct-hidden", type=int, default=64)
    p.add_argument("--direct-depth", type=int, default=2)
    p.add_argument("--direct-heads", type=int, default=4)
    p.add_argument("--direct-window-size", type=int, default=4)
    p.add_argument("--direct-mlp-ratio", type=float, default=4.0)
    p.add_argument("--direct-dropout", type=float, default=0.0)
    p.add_argument("--direct-init-sigma", type=float, default=1.0)
    p.add_argument("--direct-logvar-min", type=float, default=-8.0)
    p.add_argument("--direct-logvar-max", type=float, default=4.0)
    p.add_argument("--uncert-threshold", type=float, default=1.0)
    p.add_argument("--uncert-gate-temp", type=float, default=8.0)
    p.add_argument("--uncert-granularity", type=str, choices=["channel", "pixel"], default="channel")
    p.add_argument("--train-encoder", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--train-decoder", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--lambda-pred-rec", type=float, default=1.0)
    p.add_argument("--lambda-uncert-rec", type=float, default=0.5)
    p.add_argument("--lambda-latent", type=float, default=0.05)
    p.add_argument("--lambda-nll", type=float, default=0.01)
    p.add_argument("--lambda-sigma-calib", type=float, default=0.01)
    p.add_argument("--lambda-c1", type=float, default=0.0)
    p.add_argument(
        "--stage2-direct-score",
        type=str,
        choices=["psnr_pred_all", "psnr_pred_uncert", "psnr_pred_uncert_soft", "pred_gain", "pred_uncert_gain", "pred_uncert_soft_gain"],
        default="psnr_pred_uncert_soft",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.quantizer = "direct"
    args.predictor = "independent_channel_direct"
    args.gate = "uncertainty"
    ensure_common_args(args, stage=2)
    setup_stage_log(args, "stage2_direct")
    write_json(Path(resolve_path(args.save_dir)) / "stage2_direct_args.json", vars(args))
    train(args)


if __name__ == "__main__":
    main()
