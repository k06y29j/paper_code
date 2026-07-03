from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim

from shared import (
    add_common_cli,
    averaged,
    batch_metric_mean,
    build_encoder_decoder,
    ckpt_path,
    cvq_io,
    default_v01_save_dir,
    ensure_common_args,
    fixed_prefix_keep_mask,
    format_metrics,
    get_loader,
    load_encoder_decoder_initial,
    meters,
    print_epoch,
    print_v01_header,
    psnr_per_image,
    real_awgn,
    recon_loss,
    resolve_path,
    sample_uniform_channel_keep_mask,
    save_v01_checkpoint,
    seed_everything,
    setup_stage_log,
    should_save_latest,
    should_validate,
    split_c1_c2,
    with_log_keys,
    write_json,
)


@torch.no_grad()
def validate(loader, encoder, decoder, args) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    device = next(encoder.parameters()).device
    names = ["loss", "loss_c1_rec", "loss_drop_rec", "loss_full_rec", "psnr_c1_only", "psnr_c2_drop", "psnr_full_c2", "drop_keep_mean"]
    for keep in [0, 5, 10, 20]:
        names.append(f"drop_keep_{keep}_psnr")
    m = meters(names)
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        c1_rx = real_awgn(c1, float(args.snr_db))
        c2_rx = real_awgn(c2, float(args.snr_db))
        mask = sample_uniform_channel_keep_mask(imgs.shape[0], c2.shape[1], imgs.device, c2.dtype)
        zero = torch.zeros_like(c2_rx)
        x_c1 = decoder(torch.cat([c1_rx, zero], dim=1)).clamp(0.0, 1.0)
        x_drop = decoder(torch.cat([c1_rx, c2_rx * mask], dim=1)).clamp(0.0, 1.0)
        x_full = decoder(torch.cat([c1_rx, c2_rx], dim=1)).clamp(0.0, 1.0)
        loss_c1 = recon_loss(x_c1, imgs)
        loss_drop = recon_loss(x_drop, imgs)
        loss_full = recon_loss(x_full, imgs)
        loss = float(args.lambda_c1) * loss_c1 + float(args.lambda_drop) * loss_drop + float(args.lambda_full) * loss_full
        bsz = imgs.shape[0]
        m["loss"].update(float(loss.item()), bsz)
        m["loss_c1_rec"].update(float(loss_c1.item()), bsz)
        m["loss_drop_rec"].update(float(loss_drop.item()), bsz)
        m["loss_full_rec"].update(float(loss_full.item()), bsz)
        m["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1, imgs)), bsz)
        m["psnr_c2_drop"].update(batch_metric_mean(psnr_per_image(x_drop, imgs)), bsz)
        m["psnr_full_c2"].update(batch_metric_mean(psnr_per_image(x_full, imgs)), bsz)
        m["drop_keep_mean"].update(float(mask.float().mean().item()), bsz)
        for keep in [0, 5, 10, 20]:
            kmask = fixed_prefix_keep_mask(imgs.shape[0], c2.shape[1], keep, imgs.device, c2.dtype)
            x_keep = decoder(torch.cat([c1_rx, c2_rx * kmask], dim=1)).clamp(0.0, 1.0)
            m[f"drop_keep_{keep}_psnr"].update(batch_metric_mean(psnr_per_image(x_keep, imgs)), bsz)
    return averaged(m)


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder = build_encoder_decoder(args, cfg.device)
    if args.init_stage0_ckpt:
        args.init_ckpt = args.init_stage0_ckpt
    load_encoder_decoder_initial(args, encoder, decoder)
    opt = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = -1.0
    print_v01_header(args, "Stage 1 | JSCC random channel-wise C2 dropout backbone", len(train_loader.dataset), len(val_loader.dataset))
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train()
        decoder.train()
        m = meters(["loss", "loss_c1_rec", "loss_drop_rec", "loss_full_rec", "psnr_c1_only", "psnr_c2_drop", "psnr_full_c2", "drop_keep_mean"])
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
            c1, c2 = split_c1_c2(z_norm, args)
            c1_rx = real_awgn(c1, float(args.snr_db))
            c2_rx = real_awgn(c2, float(args.snr_db))
            mask = sample_uniform_channel_keep_mask(imgs.shape[0], c2.shape[1], imgs.device, c2.dtype)
            zero = torch.zeros_like(c2_rx)
            x_c1 = decoder(torch.cat([c1_rx, zero], dim=1))
            x_drop = decoder(torch.cat([c1_rx, c2_rx * mask], dim=1))
            x_full = decoder(torch.cat([c1_rx, c2_rx], dim=1))
            loss_c1 = recon_loss(x_c1, imgs)
            loss_drop = recon_loss(x_drop, imgs)
            loss_full = recon_loss(x_full, imgs)
            loss = float(args.lambda_c1) * loss_c1 + float(args.lambda_drop) * loss_drop + float(args.lambda_full) * loss_full
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsz = imgs.shape[0]
            m["loss"].update(float(loss.item()), bsz)
            m["loss_c1_rec"].update(float(loss_c1.item()), bsz)
            m["loss_drop_rec"].update(float(loss_drop.item()), bsz)
            m["loss_full_rec"].update(float(loss_full.item()), bsz)
            m["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1, imgs)), bsz)
            m["psnr_c2_drop"].update(batch_metric_mean(psnr_per_image(x_drop, imgs)), bsz)
            m["psnr_full_c2"].update(batch_metric_mean(psnr_per_image(x_full, imgs)), bsz)
            m["drop_keep_mean"].update(float(mask.float().mean().item()), bsz)
        metrics = averaged(m)
        print_epoch("stage1-v01", epoch, int(args.epochs), with_log_keys(metrics), time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, encoder, decoder, args)
            score = val_metrics["psnr_c2_drop"]
            print(f"[stage1-v01 val {epoch:03d}] {format_metrics(with_log_keys(val_metrics))} score=psnr_c2_drop")
            if score > best:
                best = score
                save_v01_checkpoint(ckpt_path(args, "stage1", "best"), stage="stage1", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder)
        if should_save_latest(args, epoch):
            save_v01_checkpoint(ckpt_path(args, "stage1", "latest"), stage="stage1", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder)
    save_v01_checkpoint(ckpt_path(args, "stage1", "latest"), stage="stage1", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_cli(p, default_k=4096)
    p.set_defaults(save_dir=default_v01_save_dir(4096), epochs=300, k=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-c1", type=float, default=1.0)
    p.add_argument("--lambda-drop", type=float, default=1.0)
    p.add_argument("--lambda-full", type=float, default=0.3)
    p.add_argument(
        "--init-stage0-ckpt",
        type=str,
        default="",
        help="Optional Stage0/experiment checkpoint used to initialize encoder and decoder; overrides --init-ckpt when set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.quantizer = "none"
    args.predictor = "none"
    args.gate = "none"
    ensure_common_args(args, stage=1)
    setup_stage_log(args, "stage1_v01")
    write_json(Path(resolve_path(args.save_dir)) / "stage1_v01_args.json", vars(args))
    train(args)


if __name__ == "__main__":
    main()
