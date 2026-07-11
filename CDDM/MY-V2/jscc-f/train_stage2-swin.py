from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
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
    print_run_header as common_print_run_header,
    psnr_per_image,
    recon_loss,
    resolve_path,
    seed_everything,
    setup_log_file,
    should_save_latest,
    should_validate,
    ssim_per_image,
    write_json,
    z2_ch,
)
from model import OutputsCombiner, build_jscc_decoder, build_jscc_encoder, build_layer1, build_layer2, freeze_layer1, layer2_forward


def load_local_io():
    spec = importlib.util.spec_from_file_location("jsccf_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = load_local_io()


def stage2_z_ch(args: argparse.Namespace) -> int:
    if str(args.swin_codec) == "no_compressor":
        return 320
    return z2_ch(args)


def stage2_in_ch(args: argparse.Namespace) -> int:
    return 3 if str(args.variant) == "residual_input" else 6


def build_layer2_no_compressor(args: argparse.Namespace, device: torch.device):
    e1, d1 = build_layer1(args, device)
    e2 = build_jscc_encoder(args, device, latent_ch=320, in_chans=stage2_in_ch(args))
    d2 = build_jscc_decoder(args, device, latent_ch=320)
    e2.encoder.head_list = nn.Identity()
    d2.decoder.head_list = nn.Identity()
    combiner = OutputsCombiner().to(device)
    return e1, d1, e2, d2, combiner


def build_stage2(args: argparse.Namespace, device: torch.device):
    if str(args.swin_codec) == "no_compressor":
        return build_layer2_no_compressor(args, device)
    return build_layer2(args, device)


def checkpoint_stage(args: argparse.Namespace) -> str:
    if str(args.swin_codec) == "no_compressor":
        return f"layer2_swin_no_compressor_{args.variant}"
    return f"layer2_{args.variant}"


def load_layer1(args, e1, d1) -> None:
    layer1_path = args.layer1_ckpt or args.init_ckpt
    if not layer1_path:
        raise ValueError("set --layer1-ckpt or --init-ckpt for Stage2 frozen E1-D1 initialization")
    args.layer1_ckpt = layer1_path
    jsccf_io.load_layer1_compatible_checkpoint(layer1_path, e1, d1, strict=True)


def collect_metrics(out: dict[str, torch.Tensor], imgs: torch.Tensor, m: dict) -> None:
    bsz = imgs.shape[0]
    m["mse_x1"].update(batch_metric_mean(mse_per_image(out["x1"], imgs)), bsz)
    m["psnr_x1"].update(batch_metric_mean(psnr_per_image(out["x1"], imgs)), bsz)
    m["ssim_x1"].update(batch_metric_mean(ssim_per_image(out["x1"], imgs)), bsz)
    m["mse_u2"].update(batch_metric_mean(mse_per_image(out["u2"], imgs)), bsz)
    m["psnr_u2"].update(batch_metric_mean(psnr_per_image(out["u2"], imgs)), bsz)
    m["ssim_u2"].update(batch_metric_mean(ssim_per_image(out["u2"], imgs)), bsz)
    m["mse_final"].update(batch_metric_mean(mse_per_image(out["final"], imgs)), bsz)
    psnr_final = batch_metric_mean(psnr_per_image(out["final"], imgs))
    psnr_x1 = batch_metric_mean(psnr_per_image(out["x1"], imgs))
    m["psnr_final"].update(psnr_final, bsz)
    m["ssim_final"].update(batch_metric_mean(ssim_per_image(out["final"], imgs)), bsz)
    m["delta_psnr"].update(psnr_final - psnr_x1, bsz)


def save_layer2_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    e1: nn.Module,
    d1: nn.Module,
    e2: nn.Module,
    d2: nn.Module,
    combiner: nn.Module,
) -> None:
    if str(args.swin_codec) != "no_compressor":
        jsccf_io.save_layer2_checkpoint(path, epoch=epoch, args=args, metrics=metrics, e1=e1, d1=d1, e2=e2, d2=d2, combiner=combiner)
        return
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": getattr(jsccf_io, "ROUTE", "TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256"),
            "stage": "layer2_swin_no_compressor",
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
            "version": str(getattr(args, "version", "")),
            "e1_state_dict": e1.state_dict(),
            "d1_state_dict": d1.state_dict(),
            "e2_state_dict": e2.state_dict(),
            "d2_state_dict": d2.state_dict(),
            "combiner_state_dict": combiner.state_dict(),
            "variant": str(args.variant),
            "stage2_codec": {
                "arch": "JSCC_encoder/JSCC_decoder",
                "mode": "swin_no_compressor",
                "e2_in_ch": stage2_in_ch(args),
                "z2_ch": 320,
                "encoder_head_list": "Identity",
                "decoder_head_list": "Identity",
            },
            "latent": {
                "z1": [int(args.latent_ch), int(args.latent_h), int(args.latent_w)],
                "z2": [320, int(args.latent_h), int(args.latent_w)],
            },
        },
        out,
    )
    print(f"saved checkpoint: {out}", flush=True)


def print_stage2_header(args: argparse.Namespace, train_n: int, val_n: int) -> None:
    if str(args.swin_codec) != "no_compressor":
        common_print_run_header(args, "Layer 2 | DeepJSCC-f refinement", train_n, val_n)
        return
    z1 = int(args.latent_ch)
    z2 = stage2_z_ch(args)
    latent_ratio = (z1 + z2) * int(args.latent_h) * int(args.latent_w) / float(3 * 256 * 256) * 100.0
    print("=== Layer 2 | DeepJSCC-f refinement | Swin no compressor ===", flush=True)
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"save_dir={resolve_path(args.save_dir)}", flush=True)
    if getattr(args, "init_ckpt", ""):
        print(f"init_ckpt={resolve_path(args.init_ckpt)}", flush=True)
    print("实验设计", flush=True)
    print("  model=TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256", flush=True)
    print(
        f"  version={args.version} z1={z1}x{args.latent_h}x{args.latent_w} "
        f"z2={z2}x{args.latent_h}x{args.latent_w} total_latent_ratio={latent_ratio:.2f}%",
        flush=True,
    )
    print("  channel=identity power_norm=none noise=none", flush=True)
    print("loss设计", flush=True)
    print(f"  variant={args.variant} L2=MSE(final,x)+{float(args.lambda_u2):g}*MSE(u2,x)", flush=True)
    print("模块选择", flush=True)
    print(f"  E1=JSCC_encoder(3->{z1}) D1=JSCC_decoder({z1}->3)", flush=True)
    print(
        f"  E2=JSCC_encoder({stage2_in_ch(args)}->320) with encoder.encoder.head_list=Identity",
        flush=True,
    )
    print("  D2=JSCC_decoder(320->3) with decoder.decoder.head_list=Identity", flush=True)
    print("  combiner=Conv3x3-48-PReLU-Conv3x3-Sigmoid", flush=True)
    print(f"  init_layer1_ckpt={resolve_path(args.layer1_ckpt)}", flush=True)
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}", flush=True)


@torch.no_grad()
def smoke_shapes(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    device = torch.device("cuda:0" if (not bool(args.cpu)) and torch.cuda.is_available() else "cpu")
    _e1, _d1, e2, d2, _combiner = build_stage2(args, device)
    e2.eval()
    d2.eval()
    e2_in = torch.rand(int(args.smoke_batch_size), stage2_in_ch(args), 256, 256, device=device)
    z2, _ = e2(e2_in)
    u2 = d2(z2)
    expected_z2 = (int(args.smoke_batch_size), stage2_z_ch(args), int(args.latent_h), int(args.latent_w))
    expected_u2 = (int(args.smoke_batch_size), 3, 256, 256)
    print(
        f"[smoke] swin_codec={args.swin_codec} e2_in={tuple(e2_in.shape)} z2={tuple(z2.shape)} u2={tuple(u2.shape)}",
        flush=True,
    )
    if tuple(z2.shape) != expected_z2:
        raise RuntimeError(f"expected z2 {expected_z2}, got {tuple(z2.shape)}")
    if tuple(u2.shape) != expected_u2:
        raise RuntimeError(f"expected u2 {expected_u2}, got {tuple(u2.shape)}")


@torch.no_grad()
def validate(loader, e1, d1, e2, d2, combiner, args) -> dict[str, float]:
    e1.eval()
    d1.eval()
    e2.eval()
    d2.eval()
    combiner.eval()
    device = next(e2.parameters()).device
    names = ["loss", "loss_final", "loss_u2", "mse_x1", "psnr_x1", "ssim_x1", "mse_u2", "psnr_u2", "ssim_u2", "mse_final", "psnr_final", "ssim_final", "delta_psnr"]
    m = meters(names)
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = layer2_forward(e1, d1, e2, d2, combiner, imgs, args.variant)
        loss_final = recon_loss(out["final"], imgs)
        loss_u2 = recon_loss(out["u2_raw"], imgs)
        aux = 0.0 if args.variant == "no_combiner" else float(args.lambda_u2)
        loss = loss_final + aux * loss_u2
        bsz = imgs.shape[0]
        m["loss"].update(float(loss.item()), bsz)
        m["loss_final"].update(float(loss_final.item()), bsz)
        m["loss_u2"].update(float(loss_u2.item()), bsz)
        collect_metrics(out, imgs, m)
    return averaged(m)


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = get_loader(cfg)
    e1, d1, e2, d2, combiner = build_stage2(args, cfg.device)
    load_layer1(args, e1, d1)
    freeze_layer1(e1, d1)
    params = list(e2.parameters()) + list(d2.parameters())
    if args.variant != "no_combiner":
        params += list(combiner.parameters())
    opt = optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = -1.0
    print_stage2_header(args, len(train_loader.dataset), len(val_loader.dataset))
    names = ["loss", "loss_final", "loss_u2", "mse_x1", "psnr_x1", "ssim_x1", "mse_u2", "psnr_u2", "ssim_u2", "mse_final", "psnr_final", "ssim_final", "delta_psnr"]
    for epoch in range(1, int(args.epochs) + 1):
        e1.eval()
        d1.eval()
        e2.train()
        d2.train()
        combiner.train(args.variant != "no_combiner")
        m = meters(names)
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            out = layer2_forward(e1, d1, e2, d2, combiner, imgs, args.variant)
            loss_final = recon_loss(out["final"], imgs)
            loss_u2 = recon_loss(out["u2_raw"], imgs)
            aux = 0.0 if args.variant == "no_combiner" else float(args.lambda_u2)
            loss = loss_final + aux * loss_u2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsz = imgs.shape[0]
            m["loss"].update(float(loss.item()), bsz)
            m["loss_final"].update(float(loss_final.item()), bsz)
            m["loss_u2"].update(float(loss_u2.item()), bsz)
            collect_metrics(out, imgs, m)
        metrics = averaged(m)
        print_epoch(f"layer2-{args.variant}", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, e1, d1, e2, d2, combiner, args)
            score = val_metrics["psnr_final"]
            print(f"[layer2 val {epoch:03d}] {val_metrics} score=psnr_final")
            if score > best:
                best = score
                save_layer2_checkpoint(jsccf_io.ckpt_path(args, checkpoint_stage(args), "best"), epoch=epoch, args=args, metrics=val_metrics, e1=e1, d1=d1, e2=e2, d2=d2, combiner=combiner)
        if should_save_latest(args, epoch):
            save_layer2_checkpoint(jsccf_io.ckpt_path(args, checkpoint_stage(args), "latest"), epoch=epoch, args=args, metrics=metrics, e1=e1, d1=d1, e2=e2, d2=d2, combiner=combiner)
    save_layer2_checkpoint(jsccf_io.ckpt_path(args, checkpoint_stage(args), "latest"), epoch=int(args.epochs), args=args, metrics=metrics, e1=e1, d1=d1, e2=e2, d2=d2, combiner=combiner)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", type=str, default="swin320", help="Version of the JSCC-f training; affects checkpoint and log names.")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=jsccf_io.default_save_dir())
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default="")
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
    p.add_argument("--layer1-ckpt", type=str, default="MY-V2/jscc-f/checkpoints/jscc_f_no-c1_layer1_best.pth")
    p.add_argument("--variant", type=str, default="combiner", choices=["combiner", "no_combiner", "residual_input"])
    p.add_argument("--lambda-u2", type=float, default=0.0)
    p.add_argument("--swin-codec", type=str, default="no_compressor", choices=["no_compressor", "no_compressor"], help="Use the normal compressed Swin codec or remove E2/D2 head_list so z2 is [B,320,16,16].")
    p.add_argument("--smoke-shapes", action="store_true", help="Build E2/D2, run one random shape check, and exit.")
    p.add_argument("--smoke-batch-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.stage = "layer2"
    check_jsccf_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if args.smoke_shapes:
        smoke_shapes(args)
        return
    if not args.log_file:
        codec_part = "" if str(args.swin_codec) == "compressed" else "_swin_no_compressor"
        args.log_file = str(Path(resolve_path(args.save_dir)) / f"layer2{codec_part}_{args.variant}_jscc_f_{args.version}.log")
    setup_log_file(args.log_file)
    args_name = f"layer2_{args.variant}_args.json" if str(args.swin_codec) == "compressed" else f"layer2_swin_no_compressor_{args.variant}_args.json"
    write_json(Path(resolve_path(args.save_dir)) / args_name, vars(args))
    train(args)


if __name__ == "__main__":
    main()
