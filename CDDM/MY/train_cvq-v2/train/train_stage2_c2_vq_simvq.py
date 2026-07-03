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
    build_quantizer,
    ckpt_path,
    cvq_io,
    default_stage1_ckpt,
    default_v01_save_dir,
    ensure_common_args,
    fixed_prefix_keep_mask,
    format_metrics,
    freeze_module,
    get_loader,
    init_c2_codebook_from_samples,
    load_v01_checkpoint,
    meters,
    print_epoch,
    print_v01_header,
    psnr_per_image,
    quantizer_artifact_part,
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
    vq_usage_stats,
    with_log_keys,
    write_json,
)


def load_stage1(args, encoder, decoder) -> None:
    src = args.init_stage1_ckpt or default_stage1_ckpt(args)
    obj = load_v01_checkpoint(src)
    encoder.load_state_dict(obj["encoder_state_dict"], strict=True)
    decoder.load_state_dict(obj["decoder_state_dict"], strict=True)
    print(f"stage2_source_stage1={resolve_path(src)}")


@torch.no_grad()
def validate(loader, encoder, decoder, quantizer, args) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    quantizer.eval()
    names = ["loss", "loss_q_rec", "loss_q_drop_rec", "loss_c1_rec", "vq", "psnr_c1_only", "psnr_q_c2_gt", "psnr_q_c2_drop", "psnr_real_c2_full", "quant_mse", "perplexity", "used_codes", "usage_top1_ratio", "usage_top10_ratio"]
    m = meters(names)
    device = next(decoder.parameters()).device
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        c1_rx = real_awgn(c1, float(args.snr_db))
        q_c2, idx, vq_loss, q_raw = quantizer(c2)
        mask = sample_uniform_channel_keep_mask(imgs.shape[0], c2.shape[1], imgs.device, c2.dtype)
        zero = torch.zeros_like(q_c2)
        x_q = decoder(torch.cat([c1_rx, q_c2], dim=1)).clamp(0.0, 1.0)
        x_q_drop = decoder(torch.cat([c1_rx, q_c2 * mask], dim=1)).clamp(0.0, 1.0)
        x_c1 = decoder(torch.cat([c1_rx, zero], dim=1)).clamp(0.0, 1.0)
        x_real = decoder(torch.cat([c1_rx, c2], dim=1)).clamp(0.0, 1.0)
        loss_q = recon_loss(x_q, imgs)
        loss_q_drop = recon_loss(x_q_drop, imgs)
        loss_c1 = recon_loss(x_c1, imgs)
        loss = float(args.lambda_q) * loss_q + float(args.lambda_q_drop) * loss_q_drop + float(args.lambda_c1) * loss_c1 + float(args.lambda_vq) * vq_loss
        stats = vq_usage_stats(idx, q_raw, c2, int(args.k))
        bsz = imgs.shape[0]
        for key, value in [("loss", loss), ("loss_q_rec", loss_q), ("loss_q_drop_rec", loss_q_drop), ("loss_c1_rec", loss_c1), ("vq", vq_loss)]:
            m[key].update(float(value.item()), bsz)
        m["psnr_q_c2_gt"].update(batch_metric_mean(psnr_per_image(x_q, imgs)), bsz)
        m["psnr_q_c2_drop"].update(batch_metric_mean(psnr_per_image(x_q_drop, imgs)), bsz)
        m["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1, imgs)), bsz)
        m["psnr_real_c2_full"].update(batch_metric_mean(psnr_per_image(x_real, imgs)), bsz)
        for key in ["quant_mse", "perplexity", "used_codes", "usage_top1_ratio", "usage_top10_ratio"]:
            m[key].update(stats[key], bsz)
    return averaged(m)


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder = build_encoder_decoder(args, cfg.device)
    quantizer = build_quantizer(args, cfg.device)
    load_stage1(args, encoder, decoder)
    if args.init_stage2_ckpt:
        obj = load_v01_checkpoint(args.init_stage2_ckpt)
        if "quantizer_state_dict" in obj:
            quantizer.load_state_dict(obj["quantizer_state_dict"], strict=True)
    else:
        init_c2_codebook_from_samples(train_loader, encoder, quantizer, args, cfg.device)
    freeze_module(encoder, bool(args.train_encoder))
    params = list(decoder.parameters()) + [p for p in quantizer.parameters() if p.requires_grad]
    if bool(args.train_encoder):
        params += list(encoder.parameters())
    params = [p for p in params if p.requires_grad]
    opt = optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = -1.0
    print_v01_header(args, "Stage 2 | C2-only VQ/SimVQ with continuous C1_rx", len(train_loader.dataset), len(val_loader.dataset))
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train(bool(args.train_encoder))
        decoder.train()
        quantizer.train()
        m = meters(["loss", "loss_q_rec", "loss_q_drop_rec", "loss_c1_rec", "vq", "psnr_c1_only", "psnr_q_c2_gt", "psnr_q_c2_drop", "quant_mse", "perplexity", "used_codes"])
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            with torch.set_grad_enabled(bool(args.train_encoder)):
                _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
            if not bool(args.train_encoder):
                z_norm = z_norm.detach()
            c1, c2 = split_c1_c2(z_norm, args)
            c1_rx = real_awgn(c1, float(args.snr_db))
            q_c2, idx, vq_loss, q_raw = quantizer(c2)
            mask = sample_uniform_channel_keep_mask(imgs.shape[0], c2.shape[1], imgs.device, c2.dtype)
            zero = torch.zeros_like(q_c2)
            x_q = decoder(torch.cat([c1_rx, q_c2], dim=1))
            x_q_drop = decoder(torch.cat([c1_rx, q_c2 * mask], dim=1))
            x_c1 = decoder(torch.cat([c1_rx, zero], dim=1))
            loss_q = recon_loss(x_q, imgs)
            loss_q_drop = recon_loss(x_q_drop, imgs)
            loss_c1 = recon_loss(x_c1, imgs)
            loss = float(args.lambda_q) * loss_q + float(args.lambda_q_drop) * loss_q_drop + float(args.lambda_c1) * loss_c1 + float(args.lambda_vq) * vq_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            stats = vq_usage_stats(idx, q_raw, c2, int(args.k))
            bsz = imgs.shape[0]
            for key, value in [("loss", loss), ("loss_q_rec", loss_q), ("loss_q_drop_rec", loss_q_drop), ("loss_c1_rec", loss_c1), ("vq", vq_loss)]:
                m[key].update(float(value.item()), bsz)
            m["psnr_q_c2_gt"].update(batch_metric_mean(psnr_per_image(x_q, imgs)), bsz)
            m["psnr_q_c2_drop"].update(batch_metric_mean(psnr_per_image(x_q_drop, imgs)), bsz)
            m["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1, imgs)), bsz)
            for key in ["quant_mse", "perplexity", "used_codes"]:
                m[key].update(stats[key], bsz)
        metrics = averaged(m)
        print_epoch("stage2-v01", epoch, int(args.epochs), with_log_keys(metrics), time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, encoder, decoder, quantizer, args)
            score = val_metrics["psnr_q_c2_gt"]
            print(f"[stage2-v01 val {epoch:03d}] {format_metrics(with_log_keys(val_metrics))} score=psnr_q_c2_gt")
            if score > best:
                best = score
                save_v01_checkpoint(ckpt_path(args, "stage2", "best"), stage="stage2", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)
        if should_save_latest(args, epoch):
            save_v01_checkpoint(ckpt_path(args, "stage2", "latest"), stage="stage2", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)
    save_v01_checkpoint(ckpt_path(args, "stage2", "latest"), stage="stage2", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_cli(p, default_k=19384)
    p.set_defaults(save_dir=default_v01_save_dir(19384), epochs=300)
    p.add_argument("--init-stage1-ckpt", type=str, default="MY/checkpoints-cvq-v2-v01-c36-snr9-k4096/cvq_v2_v01_c36_snr9_k4096_stage1_best.pth")
    p.add_argument("--init-stage2-ckpt", type=str, default="")
    p.add_argument("--quantizer", type=str, choices=["vq", "simvq", "patch_vq", "cross_channel_block_vq"], default="vq")
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=2)
    p.add_argument("--simvq-proj-dim", type=int, default=256)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--vq-chunk-size", type=int, default=128)
    p.add_argument("--init-codebook-method", type=str, choices=["random_samples", "kmeans"], default="kmeans")
    p.add_argument("--init-codebook-samples", type=int, default=1048576)
    p.add_argument("--kmeans-iters", type=int, default=20)
    p.add_argument("--kmeans-chunk-size", type=int, default=4096)
    p.add_argument("--train-encoder", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-q", type=float, default=1.0)
    p.add_argument("--lambda-q-drop", type=float, default=0.05)
    p.add_argument("--lambda-c1", type=float, default=0.5)
    p.add_argument("--lambda-vq", type=float, default=0.003)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.predictor = "none"
    args.gate = "none"
    ensure_common_args(args, stage=2)
    setup_stage_log(args, "stage2_v01")
    write_json(Path(resolve_path(args.save_dir)) / f"stage2_v01{quantizer_artifact_part(args, 'stage2')}_args.json", vars(args))
    train(args)


if __name__ == "__main__":
    main()
