from __future__ import annotations

import argparse
import time

import torch
import torch.optim as optim

from Autoencoder.data.datasets import get_loader

from .common import (
    AverageMeter, apply_nested_tail, batch_metric_mean, format_metrics,
    psnr_per_image, recon_loss, sample_nested_m, seed_everything,
    should_save_latest, should_validate, print_epoch, print_run_header,
)
from .io import build_config, build_models, ckpt_path, forward_parts, load_module_checkpoint, save_checkpoint
from .models import TailCVQ

@torch.no_grad()
def validate_stage1_or_3(
    loader,
    encoder: nn.Module,
    decoder: nn.Module,
    args: argparse.Namespace,
    cvq: TailCVQ | None = None,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    if cvq is not None:
        cvq.eval()
    meters = {k: AverageMeter() for k in ["psnr_prefix", "psnr_cont", "psnr_vq", "loss"]}
    for imgs, _labels in loader:
        imgs = imgs.to(next(encoder.parameters()).device, non_blocking=True)
        _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
        zero = torch.zeros_like(tail)
        x_prefix = decoder(torch.cat([y_prefix, zero], dim=1)).clamp(0.0, 1.0)
        x_cont = decoder(torch.cat([y_prefix, tail], dim=1)).clamp(0.0, 1.0)
        meters["psnr_prefix"].update(batch_metric_mean(psnr_per_image(x_prefix, imgs)), imgs.shape[0])
        meters["psnr_cont"].update(batch_metric_mean(psnr_per_image(x_cont, imgs)), imgs.shape[0])
        loss_val = recon_loss(x_prefix, imgs) + recon_loss(x_cont, imgs)
        if cvq is not None:
            tail_q, _idx, _aux = cvq.encode(tail)
            x_vq = decoder(torch.cat([y_prefix, tail_q], dim=1)).clamp(0.0, 1.0)
            meters["psnr_vq"].update(batch_metric_mean(psnr_per_image(x_vq, imgs)), imgs.shape[0])
            loss_val = loss_val + recon_loss(x_vq, imgs)
        meters["loss"].update(float(loss_val.item()), imgs.shape[0])
    return {k: v.avg for k, v in meters.items() if v.count > 0}

def train_stage1(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder, _cvq, _car = build_models(args, cfg.device)
    load_module_checkpoint(encoder, str(args.init_jscc_encoder), "init JSCC encoder", strict=True)
    load_module_checkpoint(decoder, str(args.init_jscc_decoder), "init JSCC decoder", strict=True)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(args.lr))
    best = -1.0
    print_run_header(args, f"Stage 1 | C={int(args.latent_ch)} JSCC + tail nested dropout warmup", len(train_loader.dataset), len(val_loader.dataset))
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train()
        decoder.train()
        meters = {k: AverageMeter() for k in ["loss", "loss_prefix", "loss_full", "loss_nd", "psnr_prefix", "psnr_full"]}
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
            zero = torch.zeros_like(tail)
            x_prefix = decoder(torch.cat([y_prefix, zero], dim=1))
            x_full = decoder(torch.cat([y_prefix, tail], dim=1))
            m = sample_nested_m(imgs.device, max_m=tail.shape[1], batch_size=imgs.shape[0])
            tail_nd = apply_nested_tail(tail, m)
            x_nd = decoder(torch.cat([y_prefix, tail_nd], dim=1))
            loss_prefix = recon_loss(x_prefix, imgs)
            loss_full = recon_loss(x_full, imgs)
            loss_nd = recon_loss(x_nd, imgs)
            if epoch <= int(args.stage1_full_warm_epochs):
                w_prefix = float(args.stage1_warm_lambda_prefix)
                w_full = float(args.stage1_warm_lambda_full)
                w_nested = 0.0
            else:
                w_prefix = float(args.lambda_prefix)
                w_full = float(args.lambda_full)
                w_nested = float(args.lambda_nested)
            loss = w_prefix * loss_prefix + w_full * loss_full + w_nested * loss_nd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bsz = imgs.shape[0]
            meters["loss"].update(float(loss.item()), bsz)
            meters["loss_prefix"].update(float(loss_prefix.item()), bsz)
            meters["loss_full"].update(float(loss_full.item()), bsz)
            meters["loss_nd"].update(float(loss_nd.item()), bsz)
            meters["psnr_prefix"].update(batch_metric_mean(psnr_per_image(x_prefix, imgs)), bsz)
            meters["psnr_full"].update(batch_metric_mean(psnr_per_image(x_full, imgs)), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        metrics["stage1_w_nested"] = w_nested
        print_epoch("stage1", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate_stage1_or_3(val_loader, encoder, decoder, args)
            score = val_metrics["psnr_cont"]
            print(f"[stage1 val {epoch:03d}] {format_metrics(val_metrics)} score=psnr_cont")
            if score > best:
                best = score
                save_checkpoint(ckpt_path(args, "stage1", "best"), stage="stage1", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder)
        if should_save_latest(args, epoch):
            save_checkpoint(ckpt_path(args, "stage1", "latest"), stage="stage1", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder)
    save_checkpoint(ckpt_path(args, "stage1", "latest"), stage="stage1", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder)
