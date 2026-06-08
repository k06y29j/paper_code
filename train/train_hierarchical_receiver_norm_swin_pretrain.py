#!/usr/bin/env python
"""Two-frequency receiver-normalized Swin Stage 01.

This stage trains the Swin encoder/decoder directly in the strict receiver
coordinate system with only two latent frequency groups:

    z = Encoder(x)
    z_low = z[:, 0:4]      # transmitted
    z_high = z[:, 4:16]    # not transmitted; oracle receiver side only
    scale = rms(z_low) from the channel power normalization
    z0_rx_norm = z_low / scale + AWGN
    z_high_norm = z[:, 4:16] / scale

    x_base = Decoder([z0_rx_norm, 0])
    x_oracle = Decoder([z0_rx_norm, z_high_norm])

The loss makes the transmitted low group carry stable structure while the
non-transmitted high group carries residual detail:

    L = MSE(x_oracle, x) + 0.4 * MSE(x_base, GaussianBlur(x, sigma=1.0))
"""

from __future__ import annotations

import argparse
import builtins
import os
import sys
import time

import torch
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
    ARReceiver,
    build_semantic_modules,
    encode_a,
    fixed_select_a,
    make_loaders,
    save_checkpoint,
)
from train_route_a_sc import AverageMeter, GaussianBlur, TeeStream, load_state_dict_from_ckpt  # noqa: E402


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== Receiver-normalized Swin pretrain @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Swin decoder in receiver-normalized no-side-info latent space",
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

    p.add_argument("--init_sc_encoder_ckpt", type=str, default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_encoder_div2k_c16.pth")
    p.add_argument("--init_sc_decoder_ckpt", type=str, default="checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_decoder_div2k_c16.pth")
    p.add_argument("--snr_db", type=float, default=6.0)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--decoder_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    p.add_argument("--lambda_oracle", type=float, default=1.0, help="MSE(x_oracle, x) weight")
    p.add_argument("--lambda_base", type=float, default=0.4, help="MSE(x_base, blur(x)) weight")
    p.add_argument("--blur_kernel", type=int, default=7)
    p.add_argument("--blur_sigma", type=float, default=1.0)
    p.add_argument("--lambda_kl", type=float, default=0.0)
    p.add_argument("--encoder_use_vae", action="store_true", default=False)
    p.add_argument("--no_encoder_vae", action="store_false", dest="encoder_use_vae")

    p.add_argument("--eval_every_epochs", type=int, default=20)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints-ar/twofreq_receiver_norm_swin_snr6_v1/01_receiver_norm_swin_stage01")
    p.add_argument("--log_file", type=str, default="checkpoints-ar/twofreq_receiver_norm_swin_snr6_v1/01_receiver_norm_swin_stage01/train.log")
    return p.parse_args()


def make_twofreq_receiver_latents(
    z: torch.Tensor,
    y4_norm: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_view = scale.float().view(-1, 1, 1, 1).clamp_min(1e-12)
    z_high_norm = z[:, 4:16].float() / scale_view
    z_base = torch.zeros_like(z.float())
    z_base[:, :4] = y4_norm.float()
    z_oracle = torch.cat([y4_norm.float(), z_high_norm], dim=1)
    return z_base, z_oracle, z_high_norm


def run_batch(
    *,
    imgs: torch.Tensor,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    a: torch.Tensor,
    args: argparse.Namespace,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator | None,
    blur: GaussianBlur,
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
        z_base, z_oracle, z_high_norm = make_twofreq_receiver_latents(z, y4_norm, scale)
        imgs_blur = blur(imgs.float()).clamp(0, 1)

        x_base = decoder(z_base).float().clamp(0, 1)
        x_oracle = decoder(z_oracle).float().clamp(0, 1)

        loss_oracle = F.mse_loss(x_oracle, imgs.float())
        loss_base = F.mse_loss(x_base, imgs_blur)
        loss = float(args.lambda_oracle) * loss_oracle + float(args.lambda_base) * loss_base

    stats = {
        "loss": float(loss.detach().item()),
        "loss_oracle": float(loss_oracle.detach().item()),
        "loss_base": float(loss_base.detach().item()),
        "z_high_norm_rms": float(z_high_norm.float().square().mean().sqrt().detach().item()),
        "base_to_x": float(psnr_per_image(x_base, imgs.float()).mean().item()),
        "base_to_blur": float(psnr_per_image(x_base, imgs_blur).mean().item()),
        "oracle": float(psnr_per_image(x_oracle, imgs.float()).mean().item()),
    }
    stats["gain_oracle"] = stats["oracle"] - stats["base_to_x"]
    return (loss if train else None), stats


def main() -> None:
    args = parse_args()
    args.stage = "twofreq_receiver_norm_swin_stage01"
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
    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}, stage={args.stage}, snr={float(args.snr_db):g}dB")
    print(
        "rule: two frequency groups only; z_low=z[:,0:4] is transmitted, "
        "z_high=z[:,4:16] is oracle-only; scale comes from z_low; PSNR=mean(per-image PSNR)"
    )

    train_ds, val_ds, train_loader, val_loader = make_loaders(args, device)
    print(f"train={len(train_ds)} valid={len(val_ds)} batch={args.batch_size} crop={args.crop_size}")

    encoder, decoder, cfg = build_semantic_modules(device, args)
    load_state_dict_from_ckpt(encoder, args.init_sc_encoder_ckpt, "semantic_encoder")
    load_state_dict_from_ckpt(decoder, args.init_sc_decoder_ckpt, "semantic_decoder")
    ar = ARReceiver(hidden=160, depth=4, use_scale=False).to(device)
    for p in ar.parameters():
        p.requires_grad_(False)
    blur = GaussianBlur(3, int(args.blur_kernel), float(args.blur_sigma)).to(device)

    a = fixed_select_a(device=device, dtype=torch.float32)
    aat_err = semiorth_error(a)
    if aat_err > 1e-7:
        raise RuntimeError(f"A A^T != I4, err={aat_err:.3e}")
    print(f"A_shape={tuple(a.shape)} aat_error={aat_err:.3e}")

    enc_params = list(encoder.parameters())
    dec_params = list(decoder.parameters())
    params = [
        {"params": enc_params, "lr": float(args.lr), "name": "encoder"},
        {"params": dec_params, "lr": float(args.decoder_lr), "name": "decoder"},
    ]
    optimizer = optim.AdamW(params, weight_decay=float(args.weight_decay), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    print(
        "optimizer_groups="
        + ", ".join(f"{g['name']}:n={sum(p.numel() for p in g['params'])}:lr={g['lr']}" for g in params)
    )
    print(
        f"loss=lambda_oracle*MSE(x_oracle,x) + lambda_base*MSE(x_base,blur_sigma={args.blur_sigma:g}) "
        f"= {args.lambda_oracle:g}, {args.lambda_base:g}"
    )

    meter_keys = (
        "loss",
        "loss_oracle",
        "loss_base",
        "z_high_norm_rms",
        "base_to_x",
        "base_to_blur",
        "oracle",
        "gain_oracle",
    )
    best = -1.0
    ckpt_prefix = "twofreq_receiver_norm_swin_stage01"
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train(True)
        decoder.train(True)
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
                a=a,
                args=args,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                generator=None,
                blur=blur,
                train=True,
            )
            assert loss is not None
            scaler.scale(loss).backward()
            if float(args.clip_grad_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    float(args.clip_grad_norm),
                )
            scaler.step(optimizer)
            scaler.update()
            for k in meters:
                meters[k].update(stats[k], imgs.shape[0])

        metrics = {f"train_{k}": v.avg for k, v in meters.items()}
        do_eval = epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs)
        if do_eval:
            encoder.eval()
            decoder.eval()
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
                        a=a,
                        args=args,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        generator=gen,
                        blur=blur,
                        train=False,
                    )
                    for k in val_meters:
                        val_meters[k].update(stats[k], imgs.shape[0])
            metrics.update({f"val_{k}": v.avg for k, v in val_meters.items()})
            score = metrics["val_oracle"]
            is_best = score > best
            if is_best:
                best = score
                save_checkpoint(os.path.join(save_dir, f"{ckpt_prefix}_best.pth"), encoder, decoder, ar, a, cfg, args, metrics, epoch)
            save_checkpoint(os.path.join(save_dir, f"{ckpt_prefix}_latest.pth"), encoder, decoder, ar, a, cfg, args, metrics, epoch)
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"| base_to_x={metrics['val_base_to_x']:.4f} base_to_blur={metrics['val_base_to_blur']:.4f} "
                f"oracle={metrics['val_oracle']:.4f} gain_oracle={metrics['val_gain_oracle']:+.4f} "
                f"score(val_oracle)={score:.4f} z_high_norm_rms={metrics['val_z_high_norm_rms']:.4f} "
                f"aat_err={semiorth_error(a):.2e} "
                f"{'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"base_to_x={meters['base_to_x'].avg:.4f} base_to_blur={meters['base_to_blur'].avg:.4f} "
                f"oracle={meters['oracle'].avg:.4f} gain_oracle={meters['gain_oracle'].avg:+.4f}"
            )

    print(f"best_val_oracle={best:.4f}")


if __name__ == "__main__":
    main()
