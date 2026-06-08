#!/usr/bin/env python
"""Direct high predictor for receiver-normalized Swin latents.

Stage 02A:

    input  = z0_rx_norm = z0 / scale + AWGN
    target = z[:, 4:16] / scale
    output = high_hat_norm [B, 12, 16, 16]

The Swin encoder/decoder are loaded from the receiver-normalized stage 01 and
kept frozen.  The predictor is trained without teacher forcing or high gates.
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
TRAIN_DIR = os.path.abspath(os.path.dirname(__file__))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from train_codex_orthogonal_highfreq import (  # noqa: E402
    _parse_amp,
    make_autocast,
    power_normalize_awgn,
    psnr_per_image,
    seed_everything,
    semiorth_error,
)
from train_hierarchical_swin_ar_awgn12 import (  # noqa: E402
    build_semantic_modules,
    decode_a,
    encode_a,
    fixed_select_a,
    make_loaders,
)
from train_route_a_sc import AverageMeter, TeeStream  # noqa: E402


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


class DirectHighUNet(nn.Module):
    """Small latent U-Net for 16x16 -> 16x16 high prediction."""

    def __init__(self, in_channels: int = 4, out_channels: int = 12, base: int = 128, depth: int = 3) -> None:
        super().__init__()
        d = max(1, int(depth))
        h1, h2, h3 = int(base), int(base) * 2, int(base) * 4
        self.stem = nn.Sequential(nn.Conv2d(in_channels, h1, 3, padding=1), nn.SiLU())
        self.enc1 = nn.Sequential(*[ConvResBlock(h1) for _ in range(d)])
        self.down1 = nn.Sequential(nn.Conv2d(h1, h2, 3, stride=2, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(*[ConvResBlock(h2) for _ in range(d)])
        self.down2 = nn.Sequential(nn.Conv2d(h2, h3, 3, stride=2, padding=1), nn.SiLU())
        self.mid = nn.Sequential(*[ConvResBlock(h3) for _ in range(d + 1)])
        self.up2 = nn.Sequential(nn.Conv2d(h3, h2, 3, padding=1), nn.SiLU())
        self.fuse2 = nn.Sequential(nn.Conv2d(h2 + h2, h2, 1), ConvResBlock(h2), ConvResBlock(h2))
        self.up1 = nn.Sequential(nn.Conv2d(h2, h1, 3, padding=1), nn.SiLU())
        self.fuse1 = nn.Sequential(nn.Conv2d(h1 + h1, h1, 1), ConvResBlock(h1), ConvResBlock(h1))
        self.out = nn.Sequential(
            nn.GroupNorm(min(8, h1), h1),
            nn.SiLU(),
            nn.Conv2d(h1, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.fuse2(torch.cat([self.up2(d2), e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.fuse1(torch.cat([self.up1(d1), e1], dim=1))
        return self.out(d1)


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
    builtins.print(f"\n=== DirectHigh receiver-normalized stage02 @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train direct high predictor on receiver-normalized Swin latents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--cache_decoded", action="store_true", default=True)
    p.add_argument("--no_cache_decoded", action="store_false", dest="cache_decoded")
    p.add_argument("--cache_workers", type=int, default=12)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--init_hier_ckpt", type=str, required=True)
    p.add_argument("--snr_db", type=float, default=6.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--predictor_base", type=int, default=128)
    p.add_argument("--predictor_depth", type=int, default=3)
    p.add_argument("--lambda_high", type=float, default=0.03)
    p.add_argument("--recv_mse_weight", type=float, default=0.8)
    p.add_argument("--recv_charb_weight", type=float, default=0.2)
    p.add_argument("--charb_eps", type=float, default=1e-3)
    p.add_argument("--encoder_use_vae", action="store_true", default=False)
    p.add_argument("--lambda_kl", type=float, default=0.0)
    p.add_argument("--eval_every_epochs", type=int, default=2)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints-ar/direct_high")
    p.add_argument("--log_file", type=str, default="checkpoints-ar/direct_high/train.log")
    return p.parse_args()


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_hier_encoder_decoder(path: str, encoder: nn.Module, decoder: nn.Module, device: torch.device) -> dict:
    ckpt_path = resolve_path(path)
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    print(f"loaded frozen Swin: {ckpt_path}, stage={ckpt.get('stage', 'unknown')}, epoch={ckpt.get('epoch', 'unknown')}")
    return ckpt


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(bool(trainable))


def save_checkpoint(path: str, encoder: nn.Module, decoder: nn.Module, predictor: nn.Module, a: torch.Tensor, cfg, args: argparse.Namespace, metrics: dict, epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "route": "receiver_norm_direct_high",
        "stage": "02_direct_high_unet_frozen_swin",
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "semantic_config": asdict(cfg.semantic),
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "a_matrix": a.detach().cpu(),
        "aat": (a @ a.t()).detach().cpu(),
        "aat_error": semiorth_error(a),
        "fixed_channel_codec": True,
        "channel_encoder": "A=[I4,0]",
        "channel_decoder": "A^T zero-fill",
        "power_norm_after_channel_encoder": True,
        "snr_db": float(args.snr_db),
        "trainable_encoder": False,
        "trainable_decoder": False,
        "trainable_predictor": True,
    }
    torch.save(payload, path)


def run_batch(
    *,
    imgs: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    predictor: nn.Module,
    a: torch.Tensor,
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator | None,
    train: bool,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    device = imgs.device
    bsz = imgs.shape[0]
    snr_b = torch.full((bsz,), float(args.snr_db), device=device, dtype=torch.float32)

    with make_autocast(device, amp_enabled, amp_dtype):
        z, _mu, _logvar = encoder.encode(imgs, sample=False)
        z = z.float()
        y4 = encode_a(z, a)
        y4_norm, _y4_raw, scale = power_normalize_awgn(y4, snr_b, generator=generator)
        z0_rx_norm = y4_norm.float()
        scale_view = scale.float().view(-1, 1, 1, 1).clamp_min(1e-12)
        z_high_norm = z[:, 4:16].float() / scale_view

        high_hat = predictor(z0_rx_norm.to(dtype=next(predictor.parameters()).dtype)).float()
        z_recv = torch.cat([z0_rx_norm, high_hat], dim=1)
        z_base = decode_a(z0_rx_norm, a)
        z_oracle = torch.cat([z0_rx_norm, z_high_norm], dim=1)

        x_recv = decoder(z_recv).float().clamp(0, 1)
        x_base = decoder(z_base).float().clamp(0, 1)
        x_oracle = decoder(z_oracle).float().clamp(0, 1)

        loss_recv_mse = F.mse_loss(x_recv, imgs.float())
        loss_recv_charb = charbonnier_loss(x_recv, imgs, eps=float(args.charb_eps))
        loss_recv = float(args.recv_mse_weight) * loss_recv_mse + float(args.recv_charb_weight) * loss_recv_charb
        loss_high = F.mse_loss(high_hat.float(), z_high_norm)
        loss = loss_recv + float(args.lambda_high) * loss_high

    stats = {
        "loss": float(loss.detach().item()),
        "loss_recv": float(loss_recv.detach().item()),
        "loss_recv_mse": float(loss_recv_mse.detach().item()),
        "loss_recv_charb": float(loss_recv_charb.detach().item()),
        "loss_high": float(loss_high.detach().item()),
        "psnr_base": float(psnr_per_image(x_base, imgs.float()).mean().item()),
        "psnr_recv": float(psnr_per_image(x_recv, imgs.float()).mean().item()),
        "psnr_oracle": float(psnr_per_image(x_oracle, imgs.float()).mean().item()),
    }
    return (loss if train else None), stats


def main() -> None:
    args = parse_args()
    save_dir = resolve_path(args.save_dir)
    log_file = resolve_path(args.log_file)
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

    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}, stage=02_direct_high_unet_frozen_swin, snr={float(args.snr_db):g}dB")
    print("rule: no VAE, no side information, scale comes from transmitted z0 only, PSNR=mean(per-image PSNR), A=[I4,0]")
    train_ds, val_ds, train_loader, val_loader = make_loaders(args, device)
    print(f"train={len(train_ds)} valid={len(val_ds)} batch={args.batch_size} crop={args.crop_size}")

    encoder, decoder, cfg = build_semantic_modules(device, args)
    load_hier_encoder_decoder(args.init_hier_ckpt, encoder, decoder, device)
    set_trainable(encoder, False)
    set_trainable(decoder, False)
    encoder.eval()
    decoder.eval()

    predictor = DirectHighUNet(base=int(args.predictor_base), depth=int(args.predictor_depth)).to(device)
    optimizer = optim.AdamW(predictor.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    a = fixed_select_a(device=device, dtype=torch.float32)
    aat_err = semiorth_error(a)
    if aat_err > 1e-7:
        raise RuntimeError(f"A A^T != I4, err={aat_err:.3e}")
    print(
        f"predictor=DirectHighUNet base={args.predictor_base} depth={args.predictor_depth} "
        f"params={sum(p.numel() for p in predictor.parameters())} lr={float(args.lr):g} lambda_high={float(args.lambda_high):g} "
        f"aat_error={aat_err:.3e}"
    )

    meter_keys = ("loss", "loss_recv", "loss_high", "psnr_base", "psnr_recv", "psnr_oracle")
    best = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        predictor.train(True)
        meters = {k: AverageMeter() for k in meter_keys}
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
                predictor=predictor,
                a=a,
                args=args,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                generator=None,
                train=True,
            )
            assert loss is not None
            scaler.scale(loss).backward()
            if float(args.clip_grad_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), float(args.clip_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            for k in meters:
                meters[k].update(stats[k], imgs.shape[0])

        metrics = {f"train_{k}": v.avg for k, v in meters.items()}
        do_eval = epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs)
        if do_eval:
            predictor.eval()
            val_meters = {k: AverageMeter() for k in meter_keys}
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
                        predictor=predictor,
                        a=a,
                        args=args,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        generator=gen,
                        train=False,
                    )
                    for k in val_meters:
                        val_meters[k].update(stats[k], imgs.shape[0])
            metrics.update({f"val_{k}": v.avg for k, v in val_meters.items()})
            score = metrics["val_psnr_recv"]
            is_best = score > best
            if is_best:
                best = score
                save_checkpoint(os.path.join(save_dir, "direct_high_unet_frozen_swin_best.pth"), encoder, decoder, predictor, a, cfg, args, metrics, epoch)
            save_checkpoint(os.path.join(save_dir, "direct_high_unet_frozen_swin_latest.pth"), encoder, decoder, predictor, a, cfg, args, metrics, epoch)
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} high_mse={metrics['val_loss_high']:.6f} "
                f"| base={metrics['val_psnr_base']:.4f} recv={metrics['val_psnr_recv']:.4f} "
                f"oracle={metrics['val_psnr_oracle']:.4f} gain={metrics['val_psnr_recv'] - metrics['val_psnr_base']:+.4f} "
                f"gap_oracle={metrics['val_psnr_oracle'] - metrics['val_psnr_recv']:+.4f} score={score:.4f} "
                f"aat_err={semiorth_error(a):.2e} {'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"recv={meters['psnr_recv'].avg:.4f} oracle={meters['psnr_oracle'].avg:.4f} high_mse={meters['loss_high'].avg:.6f}"
            )

    print(f"best_val_psnr_recv={best:.4f}")


if __name__ == "__main__":
    main()
