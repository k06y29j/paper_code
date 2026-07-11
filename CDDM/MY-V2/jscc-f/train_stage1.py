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
)
from model import build_layer1
from test_ed import CNNAnalysisEncoder, CNNBottleneckDecoder


DEFAULT_INIT_CKPT = "MY/jscc-no-awgn/cvq_v2_c16_stage0_best.pth"


def load_local_io():
    spec = importlib.util.spec_from_file_location("jsccf_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = load_local_io()


def checkpoint_stage(args: argparse.Namespace) -> str:
    if str(args.arch) == "cnn":
        return "layer1_cnn"
    return "layer1"


def build_stage1(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module]:
    if str(args.arch) == "cnn":
        e1 = CNNAnalysisEncoder(
            base_ch=int(args.cnn_base_ch),
            bottleneck_ch=int(args.latent_ch),
            num_res=int(args.cnn_num_res),
        ).to(device)
        d1 = CNNBottleneckDecoder(
            base_ch=int(args.cnn_base_ch),
            bottleneck_ch=int(args.latent_ch),
            num_res=int(args.cnn_num_res),
            output_activation=str(args.output_activation),
        ).to(device)
        return e1, d1
    return build_layer1(args, device)


def encode_tensor(encoder: nn.Module, img: torch.Tensor) -> torch.Tensor:
    out = encoder(img)
    if isinstance(out, (tuple, list)):
        return out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"encoder returned unsupported type {type(out)!r}")
    return out


def layer1_forward(e1: nn.Module, d1: nn.Module, img: torch.Tensor) -> dict[str, torch.Tensor]:
    z1 = encode_tensor(e1, img)
    x1_raw = d1(z1)
    x1 = x1_raw.clamp(0.0, 1.0)
    return {"z1": z1, "x1_raw": x1_raw, "x1": x1}


def print_stage1_header(args: argparse.Namespace, train_n: int, val_n: int) -> None:
    if str(args.arch) != "cnn":
        common_print_run_header(args, "Layer 1 | E1-D1 JSCC base layer", train_n, val_n)
        return
    latent_ratio = int(args.latent_ch) * int(args.latent_h) * int(args.latent_w) / float(3 * 256 * 256) * 100.0
    print("=== Layer 1 | CNN E1-D1 JSCC base layer ===", flush=True)
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"save_dir={resolve_path(args.save_dir)}", flush=True)
    if getattr(args, "init_ckpt", ""):
        print(f"init_ckpt={resolve_path(args.init_ckpt)}", flush=True)
    print("实验设计", flush=True)
    print("  model=TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256", flush=True)
    print(
        f"  version={args.version} z1={args.latent_ch}x{args.latent_h}x{args.latent_w} "
        f"latent_ratio={latent_ratio:.2f}%",
        flush=True,
    )
    print("  channel=identity power_norm=none noise=none", flush=True)
    print("loss设计", flush=True)
    print("  L1=MSE(D1(E1(x)), x)", flush=True)
    print("模块选择", flush=True)
    print(
        f"  E1=CNNAnalysisEncoder(3->{args.latent_ch}) latent=[B,{args.latent_ch},{args.latent_h},{args.latent_w}]",
        flush=True,
    )
    print(f"  D1=CNNBottleneckDecoder({args.latent_ch}->3)", flush=True)
    print(f"  cnn_base_ch={int(args.cnn_base_ch)} cnn_num_res={int(args.cnn_num_res)}", flush=True)
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}", flush=True)


@torch.no_grad()
def smoke_shapes(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    device = torch.device("cuda:0" if (not bool(args.cpu)) and torch.cuda.is_available() else "cpu")
    e1, d1 = build_stage1(args, device)
    e1.eval()
    d1.eval()
    imgs = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    out = layer1_forward(e1, d1, imgs)
    expected_z1 = (int(args.smoke_batch_size), int(args.latent_ch), int(args.latent_h), int(args.latent_w))
    expected_x1 = (int(args.smoke_batch_size), 3, 256, 256)
    print(
        f"[smoke] arch={args.arch} img={tuple(imgs.shape)} z1={tuple(out['z1'].shape)} x1={tuple(out['x1_raw'].shape)}",
        flush=True,
    )
    if tuple(out["z1"].shape) != expected_z1:
        raise RuntimeError(f"expected z1 {expected_z1}, got {tuple(out['z1'].shape)}")
    if tuple(out["x1_raw"].shape) != expected_x1:
        raise RuntimeError(f"expected x1 {expected_x1}, got {tuple(out['x1_raw'].shape)}")


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
    e1, d1 = build_stage1(args, cfg.device)
    if args.init_ckpt:
        jsccf_io.load_layer1_compatible_checkpoint(args.init_ckpt, e1, d1, strict=True)
    if args.init_e1:
        jsccf_io.load_module_checkpoint(e1, args.init_e1, "init E1", strict=True)
    if args.init_d1:
        jsccf_io.load_module_checkpoint(d1, args.init_d1, "init D1", strict=True)
    opt = optim.AdamW(list(e1.parameters()) + list(d1.parameters()), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = -1.0
    print_stage1_header(args, len(train_loader.dataset), len(val_loader.dataset))
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
                jsccf_io.save_layer1_checkpoint(jsccf_io.ckpt_path(args, checkpoint_stage(args), "best"), epoch=epoch, args=args, metrics=val_metrics, e1=e1, d1=d1)
        if should_save_latest(args, epoch):
            jsccf_io.save_layer1_checkpoint(jsccf_io.ckpt_path(args, checkpoint_stage(args), "latest"), epoch=epoch, args=args, metrics=metrics, e1=e1, d1=d1)
    jsccf_io.save_layer1_checkpoint(jsccf_io.ckpt_path(args, checkpoint_stage(args), "latest"), epoch=int(args.epochs), args=args, metrics=metrics, e1=e1, d1=d1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", type=str, default="cnn", help="Version of the JSCC-f training; affects checkpoint and log names.")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=jsccf_io.default_save_dir())
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default=DEFAULT_INIT_CKPT)
    p.add_argument("--arch", type=str, default="cnn", choices=["swin", "cnn"], help="Layer1 encoder/decoder architecture.")
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
    p.add_argument("--cnn-base-ch", type=int, default=16)
    p.add_argument("--cnn-num-res", type=int, default=2)
    p.add_argument("--output-activation", type=str, default="none", choices=["none", "sigmoid", "tanh"])
    p.add_argument("--smoke-shapes", action="store_true", help="Build E1/D1, run one random shape check, and exit.")
    p.add_argument("--smoke-batch-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.stage = "layer1"
    if str(args.arch) == "cnn" and str(args.init_ckpt) == DEFAULT_INIT_CKPT:
        args.init_ckpt = ""
    check_jsccf_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if args.smoke_shapes:
        smoke_shapes(args)
        return
    if not args.log_file:
        arch_part = "" if str(args.arch) == "swin" else "_cnn"
        args.log_file = str(Path(resolve_path(args.save_dir)) / f"layer1{arch_part}_jscc_f_{args.version}.log")
    setup_log_file(args.log_file)
    args_name = "layer1_args.json" if str(args.arch) == "swin" else "layer1_cnn_args.json"
    write_json(Path(resolve_path(args.save_dir)) / args_name, vars(args))
    train(args)


if __name__ == "__main__":
    main()
