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
    format_metrics,
    print_epoch,
    psnr_per_image,
    resolve_path,
    sample_c2_nested_prefix_mask,
    seed_everything,
    setup_log_file,
    should_save_latest,
    should_validate,
    split_c1_c2,
    write_json,
    recon_loss,
)

from Autoencoder.data.datasets import get_loader
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder


def load_local_io():
    spec = importlib.util.spec_from_file_location("cvq_v2_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cvq_io = load_local_io()


def check_stage0_args(args: argparse.Namespace) -> None:
    # if int(args.latent_ch) != 36:
    #     raise ValueError("CVQ-v2 stage0 is configured for C=36; use --latent-ch 36.")
    if int(args.c1_ch) != 16:
        raise ValueError("CVQ-v2 stage0 requires C1=16; use --c1-ch 16.")
    # if int(args.latent_ch) - int(args.c1_ch) != 20:
    #     raise ValueError("CVQ-v2 stage0 requires C2=20.")
    if int(args.latent_h) != 16 or int(args.latent_w) != 16:
        raise ValueError("The CDDM JSCC C36 encoder is expected to produce 16x16 latents for 256x256 crops.")


def build_jscc_models(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module]:
    cfg = cvq_io.build_config(args)
    encoder = JSCC_encoder(cfg, int(args.latent_ch)).to(device)
    decoder = JSCC_decoder(cfg, int(args.latent_ch)).to(device)
    return encoder, decoder


def stage0_ckpt_path(args: argparse.Namespace, suffix: str) -> str:
    return str(Path(resolve_path(args.save_dir)) / f"cvq_v2_c{int(args.latent_ch)}_stage0_{suffix}.pth")


def save_stage0_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    encoder: nn.Module,
    decoder: nn.Module,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": "cvq_v2_stage0_jscc_only_c36_c1_16_c2_20",
            "stage": "stage0",
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
            "snr_db": float(args.snr_db),
            "latent_ch": int(args.latent_ch),
            "c1_ch": int(args.c1_ch),
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        },
        out,
    )
    print(f"saved checkpoint: {out}")

def psnr_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse = (x_hat.float().clamp(0.0, 1.0) - x.float()).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def stage0_recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(recon.float().clamp(0.0, 1.0), target.float(), reduction="sum") / target.shape[0]


def print_stage0_header(args: argparse.Namespace, train_n: int, val_n: int) -> None:
    print("=== Stage 0 | JSCC-only nested/drop warmup, no codebook ===")
    print(f"device={'cpu' if args.cpu else 'cuda:0'}")
    print(f"save_dir={resolve_path(args.save_dir)}")
    print(f"init_jscc_encoder={resolve_path(args.init_jscc_encoder) if args.init_jscc_encoder else 'random'}")
    print(f"init_jscc_decoder={resolve_path(args.init_jscc_decoder) if args.init_jscc_decoder else 'random'}")
    print(f"latent_ch={args.latent_ch} C1={args.c1_ch} C2={int(args.latent_ch) - int(args.c1_ch)} codebook=none")
    print(f"power_norm=all_latents_scaled_by_C1_mean_square stage=0 snr_db={args.snr_db:g}")
    print(f"c2_dropout=prefix_nested_uniform_k_0_to_{int(args.latent_ch) - int(args.c1_ch)} ratio={float(args.nested_drop_ratio):g}")
    print(
        "loss_stage0="
        f"{float(args.lambda_c1):g}*L_rec_c1_only+"
        f"{float(args.lambda_drop):g}*L_rec_nested_drop+"
        f"{float(args.lambda_full):g}*L_rec_full"
    )
    print("loss_reduction=F.mse_loss(clamp(recon,0,1), input, reduction='sum') / batch")
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}")


@torch.no_grad()
def validate_stage0(loader, encoder: nn.Module, decoder: nn.Module, args: argparse.Namespace) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    meters = {
        k: AverageMeter()
        for k in [
            "loss",
            "loss_c1_rec",
            "loss_drop_rec",
            "loss_full_rec",
            "psnr_c1_only",
            "psnr_drop",
            "psnr_full",
        ]
    }
    device = next(encoder.parameters()).device
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        mask = sample_c2_nested_prefix_mask(imgs.shape[0], c2.shape[1], float(args.nested_drop_ratio), imgs.device, c2.dtype)
        x_full = decoder(z_norm).clamp(0.0, 1.0)
        x_nested_drop = decoder(torch.cat([c1, c2 * mask], dim=1)).clamp(0.0, 1.0)
        x_c1_only = decoder(torch.cat([c1, torch.zeros_like(c2)], dim=1)).clamp(0.0, 1.0)
        loss_c1_rec = recon_loss(x_c1_only, imgs)
        # loss_drop_rec = recon_loss(x_nested_drop, imgs)
        loss_full_rec = recon_loss(x_full, imgs)
        loss = (
            float(args.lambda_c1) * loss_c1_rec
            # + float(args.lambda_drop) * loss_drop_rec
            + float(args.lambda_full) * loss_full_rec
        )
        bsz = imgs.shape[0]
        meters["loss"].update(float(loss.item()), bsz)
        meters["loss_c1_rec"].update(float(loss_c1_rec.item()), bsz)
        # meters["loss_drop_rec"].update(float(loss_drop_rec.item()), bsz)
        meters["loss_full_rec"].update(float(loss_full_rec.item()), bsz)
        meters["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1_only, imgs)), bsz)
        # meters["psnr_drop"].update(batch_metric_mean(psnr_per_image(x_nested_drop, imgs)), bsz)
        meters["psnr_full"].update(batch_metric_mean(psnr_per_image(x_full, imgs)), bsz)
    return {k: v.avg for k, v in meters.items()}


def train_stage0(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder = build_jscc_models(args, cfg.device)
    if args.init_jscc_encoder:
        cvq_io.load_module_checkpoint(encoder, args.init_jscc_encoder, "init JSCC encoder", strict=True)
    else:
        print("init JSCC encoder: random")
    if args.init_jscc_decoder:
        cvq_io.load_module_checkpoint(decoder, args.init_jscc_decoder, "init JSCC decoder", strict=True)
    else:
        print("init JSCC decoder: random")
    if args.init_ckpt:
        cvq_io.load_experiment_checkpoint(args.init_ckpt, encoder=encoder, decoder=decoder, strict=True)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(args.lr))
    best = -1.0
    print_stage0_header(args, len(train_loader.dataset), len(val_loader.dataset))
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train()
        decoder.train()
        meters = {
            k: AverageMeter()
            for k in [
                "loss",
                "loss_c1_rec",
                "loss_drop_rec",
                "loss_full_rec",
                "psnr_c1_only",
                "psnr_drop",
                "psnr_full",
                "drop_keep",
            ]
        }
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
            c1, c2 = split_c1_c2(z_norm, args)
            mask = sample_c2_nested_prefix_mask(imgs.shape[0], c2.shape[1], float(args.nested_drop_ratio), imgs.device, c2.dtype)
            x_full = decoder(z_norm)
            x_nested_drop = decoder(torch.cat([c1, c2 * mask], dim=1))
            x_c1_only = decoder(torch.cat([c1, torch.zeros_like(c2)], dim=1))
            loss_c1_rec = stage0_recon_loss(x_c1_only, imgs)
            # loss_drop_rec = stage0_recon_loss(x_nested_drop, imgs)
            loss_full_rec = stage0_recon_loss(x_full, imgs)
            loss = (
                # float(args.lambda_c1) * loss_c1_rec
                # + float(args.lambda_drop) * loss_drop_rec
                + 
                float(args.lambda_full) * loss_full_rec
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bsz = imgs.shape[0]
            meters["loss"].update(float(loss.item()), bsz)
            meters["loss_c1_rec"].update(float(loss_c1_rec.item()), bsz)
            # meters["loss_drop_rec"].update(float(loss_drop_rec.item()), bsz)
            meters["loss_full_rec"].update(float(loss_full_rec.item()), bsz)
            meters["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1_only, imgs)), bsz)
            # meters["psnr_drop"].update(batch_metric_mean(psnr_per_image(x_nested_drop, imgs)), bsz)
            meters["psnr_full"].update(batch_metric_mean(psnr_per_image(x_full, imgs)), bsz)
            # meters["drop_keep"].update(float(mask.float().mean().item()), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        print_epoch("stage0", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate_stage0(val_loader, encoder, decoder, args)
            score = val_metrics["psnr_full"]
            print(f"[stage0 val {epoch:03d}] {format_metrics(val_metrics)} score=psnr_full")
            if score > best:
                best = score
                save_stage0_checkpoint(stage0_ckpt_path(args, f"best_{args.version}"), epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder)
        if should_save_latest(args, epoch):
            save_stage0_checkpoint(stage0_ckpt_path(args, f"latest_{args.version}"), epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder)
    save_stage0_checkpoint(stage0_ckpt_path(args, f"latest_{args.version}"), epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", type=str, default="no-c1", help="Version of the CVQ-v2 training; affects default init JSCC encoder/decoder.")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default="/workspace/yongjia/paper_code/CDDM/MY/jscc-no-awgn")
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default="MY/jscc-no-awgn/cvq_v2_c16_stage0_best.pth")
    p.add_argument("--init-jscc-encoder", type=str, default=cvq_io.default_jscc_encoder_c36_snr12())
    p.add_argument("--init-jscc-decoder", type=str, default=cvq_io.default_jscc_decoder_c36_snr12())
    p.add_argument("--snr-db", type=float, default=12.0)
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
    p.add_argument("--lambda-c1", type=float, default=0.0)
    p.add_argument("--lambda-drop", type=float, default=0.0)
    p.add_argument("--lambda-full", type=float, default=1.0)
    p.add_argument(
        "--nested-drop-ratio",
        "--c2-dropout-prob",
        dest="nested_drop_ratio",
        type=float,
        default=1.0,
        help="Probability of applying C2 prefix nested dropout; otherwise full C2 is kept.",
    )
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def disable_mismatched_default_jscc_init(args: argparse.Namespace) -> None:
    default_init_ch = 16
    if int(args.latent_ch) == default_init_ch:
        return
    if args.init_jscc_encoder == cvq_io.default_jscc_encoder_c36_snr12():
        args.init_jscc_encoder = ""
    if args.init_jscc_decoder == cvq_io.default_jscc_decoder_c36_snr12():
        args.init_jscc_decoder = ""


def main() -> None:
    args = parse_args()
    args.stage = 0
    check_stage0_args(args)
    disable_mismatched_default_jscc_init(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(Path(args.save_dir) / f"stage0_jscc_v2_c{args.latent_ch}_snr{args.snr_db:g}_{args.version}.log")
    setup_log_file(args.log_file)
    write_json(Path(args.save_dir) / "stage0_args.json", vars(args))
    train_stage0(args)


if __name__ == "__main__":
    main()
