from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (
    AverageMeter,
    batch_metric_mean,
    check_args,
    format_metrics,
    freeze_module,
    print_epoch,
    print_run_header,
    psnr_per_image,
    real_awgn,
    recon_loss,
    resolve_path,
    seed_everything,
    setup_log_file,
    should_save_latest,
    should_validate,
    split_c1_c2,
    write_json,
)
from model import FullChannelQuantizer

from Autoencoder.data.datasets import get_loader


def load_local_io():
    spec = importlib.util.spec_from_file_location("cvq_v2_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cvq_io = load_local_io()


@torch.no_grad()
def validate_stage2(loader, encoder: nn.Module, decoder: nn.Module, quantizer: FullChannelQuantizer, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    quantizer.eval()
    meters = {k: AverageMeter() for k in ["loss", "psnr", "quant_mse"]}
    device = next(decoder.parameters()).device
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        c2_q, _idx = quantizer.encode(c2)
        y_c1 = real_awgn(c1, float(args.snr_db))
        recon = decoder(torch.cat([y_c1, c2_q], dim=1)).clamp(0.0, 1.0)
        bsz = imgs.shape[0]
        meters["loss"].update(float(recon_loss(recon, imgs).item()), bsz)
        meters["psnr"].update(batch_metric_mean(psnr_per_image(recon, imgs)), bsz)
        meters["quant_mse"].update(float(torch.mean((c2_q.float() - c2.float()).square()).item()), bsz)
    return {k: v.avg for k, v in meters.items()}


def train_stage2(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder, quantizer = cvq_io.build_models(args, cfg.device)
    src = args.init_ckpt or cvq_io.ckpt_path(args, "stage1", "best")
    cvq_io.load_experiment_checkpoint(src, encoder=encoder, decoder=decoder, quantizer=quantizer, strict=True)
    freeze_module(encoder, False)
    freeze_module(quantizer, False)
    freeze_module(decoder, True)
    optimizer = optim.Adam(decoder.parameters(), lr=float(args.decoder_lr))
    best = -1.0
    print_run_header(args, "Stage 2 | frozen encoder/codebook, train decoder with C1 AWGN + C2 VQ", len(train_loader.dataset), len(val_loader.dataset))
    print(f"stage2_source_checkpoint={resolve_path(src)}")
    for epoch in range(1, int(args.epochs) + 1):
        decoder.train()
        encoder.eval()
        quantizer.eval()
        meters = {k: AverageMeter() for k in ["loss", "psnr", "quant_mse"]}
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            with torch.no_grad():
                _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
                c1, c2 = split_c1_c2(z_norm, args)
                c2_q, _idx = quantizer.encode(c2)
                y_c1 = real_awgn(c1, float(args.snr_db))
                dec_in = torch.cat([y_c1, c2_q], dim=1)
                q_mse = torch.mean((c2_q.float() - c2.float()).square())
            recon = decoder(dec_in)
            loss = recon_loss(recon, imgs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bsz = imgs.shape[0]
            meters["loss"].update(float(loss.item()), bsz)
            meters["psnr"].update(batch_metric_mean(psnr_per_image(recon, imgs)), bsz)
            meters["quant_mse"].update(float(q_mse.item()), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        print_epoch("stage2", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate_stage2(val_loader, encoder, decoder, quantizer, args)
            score = val_metrics["psnr"]
            print(f"[stage2 val {epoch:03d}] {format_metrics(val_metrics)} score=psnr")
            if score > best:
                best = score
                cvq_io.save_checkpoint(cvq_io.ckpt_path(args, "stage2", "best"), stage="stage2", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)
        if should_save_latest(args, epoch):
            cvq_io.save_checkpoint(cvq_io.ckpt_path(args, "stage2", "latest"), stage="stage2", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)
    cvq_io.save_checkpoint(cvq_io.ckpt_path(args, "stage2", "latest"), stage="stage2", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=cvq_io.default_save_dir())
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default="")
    p.add_argument("--init-jscc-encoder", type=str, default=cvq_io.default_jscc_encoder_c36_snr12())
    p.add_argument("--init-jscc-decoder", type=str, default=cvq_io.default_jscc_decoder_c36_snr12())
    p.add_argument("--snr-db", type=float, default=12.0)
    p.add_argument("--latent-ch", type=int, default=36)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--k", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--decoder-lr", type=float, default=2e-5)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--vq-chunk-size", type=int, default=128)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.stage = 2
    check_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(Path(args.save_dir) / f"stage2_cvq_v2_c36_snr{args.snr_db:g}_k{int(args.k)}.log")
    setup_log_file(args.log_file)
    write_json(Path(args.save_dir) / "stage2_args.json", vars(args))
    train_stage2(args)


if __name__ == "__main__":
    main()
