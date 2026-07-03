from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent
CDDM_ROOT = THIS_DIR.parents[1]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from Autoencoder.data.datasets import get_loader

from common import (
    averaged,
    batch_metric_mean,
    check_jsccf_args,
    meters,
    mse_per_image,
    print_epoch,
    print_run_header,
    psnr_per_image,
    recon_loss,
    resolve_path,
    seed_everything,
    setup_log_file,
    should_save_latest,
    should_validate,
    ssim_per_image,
    write_json,
)
from model import build_layer1, layer1_forward


def load_local_io():
    spec = importlib.util.spec_from_file_location("jsccf_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = load_local_io()


@torch.no_grad()
def validate(loader, e1, d1) -> dict[str, float]:
    e1.eval()
    d1.eval()
    device = next(e1.parameters()).device
    m = meters(["loss", "mse_x1", "psnr_x1", "ssim_x1"])
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = layer1_forward(e1, d1, imgs)
        loss = recon_loss(out["x1_raw"], imgs)
        bsz = imgs.shape[0]
        m["loss"].update(float(loss.item()), bsz)
        m["mse_x1"].update(batch_metric_mean(mse_per_image(out["x1"], imgs)), bsz)
        m["psnr_x1"].update(batch_metric_mean(psnr_per_image(out["x1"], imgs)), bsz)
        m["ssim_x1"].update(batch_metric_mean(ssim_per_image(out["x1"], imgs)), bsz)
    return averaged(m)


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = get_loader(cfg)
    e1, d1 = build_layer1(args, cfg.device)
    if args.init_ckpt:
        jsccf_io.load_layer1_compatible_checkpoint(args.init_ckpt, e1, d1, strict=True)
    if args.init_e1:
        jsccf_io.load_module_checkpoint(e1, args.init_e1, "init E1", strict=True)
    if args.init_d1:
        jsccf_io.load_module_checkpoint(d1, args.init_d1, "init D1", strict=True)
    opt = optim.AdamW(list(e1.parameters()) + list(d1.parameters()), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = -1.0
    print_run_header(args, "Layer 1 | E1-D1 JSCC base layer", len(train_loader.dataset), len(val_loader.dataset))
    for epoch in range(1, int(args.epochs) + 1):
        e1.train()
        d1.train()
        m = meters(["loss", "mse_x1", "psnr_x1", "ssim_x1"])
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            out = layer1_forward(e1, d1, imgs)
            loss = recon_loss(out["x1_raw"], imgs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsz = imgs.shape[0]
            m["loss"].update(float(loss.item()), bsz)
            m["mse_x1"].update(batch_metric_mean(mse_per_image(out["x1"], imgs)), bsz)
            m["psnr_x1"].update(batch_metric_mean(psnr_per_image(out["x1"], imgs)), bsz)
            m["ssim_x1"].update(batch_metric_mean(ssim_per_image(out["x1"], imgs)), bsz)
        metrics = averaged(m)
        print_epoch("layer1", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, e1, d1)
            score = val_metrics["psnr_x1"]
            print(f"[layer1 val {epoch:03d}] {val_metrics} score=psnr_x1")
            if score > best:
                best = score
                jsccf_io.save_layer1_checkpoint(jsccf_io.ckpt_path(args, "layer1", "best"), epoch=epoch, args=args, metrics=val_metrics, e1=e1, d1=d1)
        if should_save_latest(args, epoch):
            jsccf_io.save_layer1_checkpoint(jsccf_io.ckpt_path(args, "layer1", "latest"), epoch=epoch, args=args, metrics=metrics, e1=e1, d1=d1)
    jsccf_io.save_layer1_checkpoint(jsccf_io.ckpt_path(args, "layer1", "latest"), epoch=int(args.epochs), args=args, metrics=metrics, e1=e1, d1=d1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", type=str, default="no-c1", help="Version of the JSCC-f training; affects checkpoint and log names.")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=jsccf_io.default_save_dir())
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default="MY/jscc-no-awgn/cvq_v2_c16_stage0_best.pth")
    p.add_argument("--latent-ch", type=int, default=16)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--init-e1", type=str, default="")
    p.add_argument("--init-d1", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.stage = "layer1"
    check_jsccf_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(Path(resolve_path(args.save_dir)) / f"layer1_jscc_f_{args.version}.log")
    setup_log_file(args.log_file)
    write_json(Path(resolve_path(args.save_dir)) / "layer1_args.json", vars(args))
    train(args)


if __name__ == "__main__":
    main()
