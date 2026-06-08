from __future__ import annotations

import argparse
import time

import torch
import torch.optim as optim

from Autoencoder.data.datasets import get_loader

from .common import AverageMeter, batch_metric_mean, format_metrics, print_epoch, print_run_header, psnr_per_image, recon_loss, seed_everything, should_save_latest, should_validate
from .io import build_config, build_models, ckpt_path, forward_parts, load_experiment_checkpoint, save_checkpoint
from .models import TailCAR, TailCVQ
from .stage3 import freeze_module

@torch.no_grad()
def validate_stage6(loader, encoder: nn.Module, decoder: nn.Module, cvq: TailCVQ, car: TailCAR, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    cvq.eval()
    car.eval()
    meters = {k: AverageMeter() for k in ["psnr_car", "psnr_vq_oracle", "ce", "acc_all"]}
    for imgs, _labels in loader:
        imgs = imgs.to(next(encoder.parameters()).device, non_blocking=True)
        _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
        tail_q, idx, _aux = cvq.encode(tail)
        logits_a, logits_b = car(y_prefix, idx)
        ce = car.ce_loss(logits_a, logits_b, idx)
        ar_a, ar_b = car.generate(y_prefix)
        pred_idx = torch.cat([ar_a, ar_b], dim=1)
        pred_tail = cvq.decode_indices(pred_idx)
        x_car = decoder(torch.cat([y_prefix, pred_tail], dim=1)).clamp(0.0, 1.0)
        x_vq = decoder(torch.cat([y_prefix, tail_q], dim=1)).clamp(0.0, 1.0)
        bsz = imgs.shape[0]
        meters["psnr_car"].update(batch_metric_mean(psnr_per_image(x_car, imgs)), bsz)
        meters["psnr_vq_oracle"].update(batch_metric_mean(psnr_per_image(x_vq, imgs)), bsz)
        meters["ce"].update(float(ce.item()), bsz)
        meters["acc_all"].update(float((pred_idx == idx).float().mean().item()), bsz)
    return {k: v.avg for k, v in meters.items()}

def train_stage6(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder, cvq, car = build_models(args, cfg.device)
    src = args.init_ckpt or ckpt_path(args, "stage5", "best")
    load_experiment_checkpoint(src, encoder=encoder, decoder=decoder, cvq=cvq, car=car, strict=True)
    freeze_module(encoder, False)
    freeze_module(cvq, False)
    freeze_module(decoder, True)
    freeze_module(car, True)
    optimizer = optim.Adam(
        [
            {"params": decoder.parameters(), "lr": float(args.decoder_lr)},
            {"params": car.parameters(), "lr": float(args.car_lr)},
        ]
    )
    best = -1.0
    print_run_header(args, "Stage 6 | CAR + decoder finetune", len(train_loader.dataset), len(val_loader.dataset))
    if int(args.epochs) <= 0:
        val_metrics = validate_stage6(val_loader, encoder, decoder, cvq, car, args)
        print(f"[stage6 val 000] {format_metrics(val_metrics)} score=psnr_car")
        save_checkpoint(ckpt_path(args, "stage6", "best"), stage="stage6", epoch=0, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
        save_checkpoint(ckpt_path(args, "stage6", "latest"), stage="stage6", epoch=0, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
        return
    for epoch in range(1, int(args.epochs) + 1):
        encoder.eval()
        cvq.eval()
        decoder.train()
        car.train()
        meters = {k: AverageMeter() for k in ["loss", "ce", "rec", "psnr_soft"]}
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            _z, _s, y_prefix, tail = forward_parts(imgs, encoder, args)
            _tail_q, idx, _aux = cvq.encode(tail)
            logits_a, logits_b = car(y_prefix, idx)
            ce = car.ce_loss(logits_a, logits_b, idx)
            tail_soft = car.soft_decode(logits_a, logits_b, cvq, tau=float(args.stage6_soft_tau))
            with torch.no_grad():
                pred_idx = torch.cat([logits_a.argmax(dim=-1), logits_b.argmax(dim=-1)], dim=1)
                tail_hard = cvq.decode_indices(pred_idx)
            tail_st = tail_hard + (tail_soft - tail_soft.detach())
            x_soft = decoder(torch.cat([y_prefix, tail_st], dim=1))
            rec = recon_loss(x_soft, imgs)
            loss = ce + float(args.lambda_stage6_rec) * rec
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bsz = imgs.shape[0]
            meters["loss"].update(float(loss.item()), bsz)
            meters["ce"].update(float(ce.item()), bsz)
            meters["rec"].update(float(rec.item()), bsz)
            meters["psnr_soft"].update(batch_metric_mean(psnr_per_image(x_soft, imgs)), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        print_epoch("stage6", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate_stage6(val_loader, encoder, decoder, cvq, car, args)
            score = val_metrics["psnr_car"]
            print(f"[stage6 val {epoch:03d}] {format_metrics(val_metrics)} score=psnr_car")
            if score > best:
                best = score
                save_checkpoint(ckpt_path(args, "stage6", "best"), stage="stage6", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
        if should_save_latest(args, epoch):
            save_checkpoint(ckpt_path(args, "stage6", "latest"), stage="stage6", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
    save_checkpoint(ckpt_path(args, "stage6", "latest"), stage="stage6", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, cvq=cvq, car=car)
