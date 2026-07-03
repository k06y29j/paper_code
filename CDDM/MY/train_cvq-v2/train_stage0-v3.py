from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    seed_everything,
    setup_log_file,
    should_save_latest,
    split_c1_c2,
    write_json,
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


def check_stage0_v3_args(args: argparse.Namespace) -> None:
    if int(args.latent_ch) != 36:
        raise ValueError("stage0-v3 trains a C=36 encoder; use --latent-ch 36.")
    if int(args.c1_ch) != 16:
        raise ValueError("stage0-v3 uses a C1=16 decoder first; use --c1-ch 16.")
    if int(args.latent_ch) - int(args.c1_ch) != 20:
        raise ValueError("stage0-v3 expects C2=20.")
    if int(args.latent_h) != 16 or int(args.latent_w) != 16:
        raise ValueError("The CDDM JSCC encoder is expected to produce 16x16 latents for 256x256 crops.")
    if int(args.phase1_epochs) <= 0:
        raise ValueError("--phase1-epochs must be positive.")
    if int(args.phase2_epochs) <= 0:
        raise ValueError("--phase2-epochs must be positive.")


def c1_args(args: argparse.Namespace) -> argparse.Namespace:
    out = copy.copy(args)
    out.latent_ch = int(args.c1_ch)
    out.c1_ch = int(args.c1_ch)
    return out


def build_models(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module]:
    cfg_full = cvq_io.build_config(args)
    encoder = JSCC_encoder(cfg_full, int(args.latent_ch)).to(device)

    cfg_c1 = cvq_io.build_config(c1_args(args))
    decoder_c1 = JSCC_decoder(cfg_c1, int(args.c1_ch)).to(device)
    decoder_full = JSCC_decoder(cfg_full, int(args.latent_ch)).to(device)
    return encoder, decoder_c1, decoder_full


def stage0_recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon.float().clamp(0.0, 1.0), target.float(), reduction="mean")


def decoder_head(decoder: nn.Module) -> nn.Conv2d:
    head = decoder.decoder.head_list
    if not isinstance(head, nn.Conv2d):
        raise TypeError(f"expected decoder.decoder.head_list to be Conv2d, got {type(head)!r}")
    return head


@torch.no_grad()
def copy_c1_decoder_to_full(decoder_c1: nn.Module, decoder_full: nn.Module, c1_ch: int) -> None:
    c1_state = decoder_c1.state_dict()
    full_state = decoder_full.state_dict()
    for name, value in c1_state.items():
        if name == "decoder.head_list.weight":
            full_state[name][:, : int(c1_ch)].copy_(value)
        elif name == "decoder.head_list.bias":
            full_state[name].copy_(value)
        elif name in full_state and full_state[name].shape == value.shape:
            full_state[name].copy_(value)
    decoder_full.load_state_dict(full_state, strict=True)


def attach_full_head_c1_mask(decoder_full: nn.Module, c1_ch: int) -> None:
    weight = decoder_head(decoder_full).weight
    mask = torch.ones_like(weight)
    mask[:, : int(c1_ch)] = 0.0
    weight.register_hook(lambda grad: grad * mask.to(device=grad.device, dtype=grad.dtype))
    bias = decoder_head(decoder_full).bias
    if bias is not None:
        bias.register_hook(lambda grad: torch.zeros_like(grad))


@torch.no_grad()
def resync_full_head_c1(decoder_c1: nn.Module, decoder_full: nn.Module, c1_ch: int) -> None:
    head_c1 = decoder_head(decoder_c1)
    head_full = decoder_head(decoder_full)
    head_full.weight[:, : int(c1_ch)].copy_(head_c1.weight)
    if head_full.bias is not None and head_c1.bias is not None:
        head_full.bias.copy_(head_c1.bias)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def stage0_v3_ckpt_path(args: argparse.Namespace, suffix: str) -> str:
    return str(Path(resolve_path(args.save_dir)) / f"cvq_v2_c{int(args.latent_ch)}_stage0_v3_{suffix}.pth")


def save_stage0_v3_checkpoint(
    path: str,
    *,
    phase: str,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    encoder: nn.Module,
    decoder_c1: nn.Module,
    decoder_full: nn.Module,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": "cvq_v2_stage0_v3_c36_collaborative_c1_decoder_then_full_decoder",
            "stage": "stage0_v3",
            "phase": phase,
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
            "latent_ch": int(args.latent_ch),
            "c1_ch": int(args.c1_ch),
            "encoder_state_dict": encoder.state_dict(),
            "decoder_c1_state_dict": decoder_c1.state_dict(),
            "decoder_full_state_dict": decoder_full.state_dict(),
        },
        out,
    )
    print(f"saved checkpoint: {out}")


def print_stage0_v3_header(args: argparse.Namespace, train_n: int, val_n: int) -> None:
    print("=== Stage 0-v3 | C36 collaborative JSCC with C1 decoder then full decoder ===")
    print("实验设计=C36 encoder outputs 36 channels; phase1 trains a C1=16 decoder on channels 0..15; phase2 trains a C36 decoder with channels 0..15 hard-synced to the C1 decoder")
    print(
        "loss设计="
        f"phase1: {float(args.lambda_phase1_c1):g}*L_rec_c1; "
        f"phase2: {float(args.lambda_full):g}*L_rec_full+"
        f"{float(args.lambda_c1_anchor):g}*L_rec_c1_anchor+"
        f"{float(args.lambda_c1_consistency):g}*L_consistency_full_c1"
    )
    print("模块选择=one JSCC_encoder(C36), two JSCC_decoder modules: decoder_c1(C16) and decoder_full(C36)")
    print(f"device={'cpu' if args.cpu else 'cuda:0'}")
    print(f"save_dir={resolve_path(args.save_dir)}")
    print(f"init_ckpt={resolve_path(args.init_ckpt) if args.init_ckpt else 'none'}")
    print(f"init_jscc_encoder={resolve_path(args.init_jscc_encoder) if args.init_jscc_encoder else 'random'}")
    print(f"latent_ch={args.latent_ch} C1={args.c1_ch} C2={int(args.latent_ch) - int(args.c1_ch)} channel=none codebook=none")
    print("full_decoder_c1_consistency=head_list.weight[:,0:16] and head_list.bias copied from decoder_c1 and kept fixed in phase2")
    print(f"epochs=phase1:{args.phase1_epochs} phase2:{args.phase2_epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}")


def load_initial_state(args: argparse.Namespace, encoder: nn.Module, decoder_c1: nn.Module, decoder_full: nn.Module) -> None:
    if args.init_jscc_encoder:
        cvq_io.load_module_checkpoint(encoder, args.init_jscc_encoder, "init JSCC encoder", strict=True)
    else:
        print("init JSCC encoder: random")
    if args.init_c1_decoder:
        cvq_io.load_module_checkpoint(decoder_c1, args.init_c1_decoder, "init C1 decoder", strict=True)
    else:
        print("init C1 decoder: random")
    if args.init_full_decoder:
        cvq_io.load_module_checkpoint(decoder_full, args.init_full_decoder, "init full decoder", strict=True)
    else:
        print("init full decoder: random")
    if args.init_ckpt:
        ckpt = cvq_io.load_experiment_checkpoint(args.init_ckpt, encoder=encoder, strict=False)
        if "decoder_c1_state_dict" in ckpt:
            cvq_io.load_state(decoder_c1, ckpt["decoder_c1_state_dict"], "decoder_c1", strict=True)
        if "decoder_full_state_dict" in ckpt:
            cvq_io.load_state(decoder_full, ckpt["decoder_full_state_dict"], "decoder_full", strict=True)
        elif "decoder_state_dict" in ckpt:
            cvq_io.load_state(decoder_full, ckpt["decoder_state_dict"], "decoder_full", strict=True)


def phase1_forward(
    imgs: torch.Tensor,
    encoder: nn.Module,
    decoder_c1: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
    c1, _c2 = split_c1_c2(z_norm, args)
    x_c1 = decoder_c1(c1)
    loss_c1 = stage0_recon_loss(x_c1, imgs)
    loss = float(args.lambda_phase1_c1) * loss_c1
    return loss, {"loss": loss, "loss_c1_rec": loss_c1}, {"x_c1": x_c1}


def phase2_forward(
    imgs: torch.Tensor,
    encoder: nn.Module,
    decoder_c1: nn.Module,
    decoder_full: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
    c1, c2 = split_c1_c2(z_norm, args)
    x_full = decoder_full(z_norm)
    x_c1_anchor = decoder_c1(c1)
    x_c1_target = x_c1_anchor.detach().clamp(0.0, 1.0)
    x_full_c1_only = decoder_full(torch.cat([c1, torch.zeros_like(c2)], dim=1))

    loss_full = stage0_recon_loss(x_full, imgs)
    loss_c1_anchor = stage0_recon_loss(x_c1_anchor, imgs)
    loss_c1_consistency = F.mse_loss(x_full_c1_only.float().clamp(0.0, 1.0), x_c1_target.float(), reduction="mean")
    loss = (
        float(args.lambda_full) * loss_full
        + float(args.lambda_c1_anchor) * loss_c1_anchor
        + float(args.lambda_c1_consistency) * loss_c1_consistency
    )
    losses = {
        "loss": loss,
        "loss_full_rec": loss_full,
        "loss_c1_anchor": loss_c1_anchor,
        "loss_c1_consistency": loss_c1_consistency,
    }
    outputs = {
        "x_full": x_full,
        "x_c1_anchor": x_c1_anchor,
        "x_full_c1_only": x_full_c1_only,
    }
    return loss, losses, outputs


@torch.no_grad()
def validate_stage0_v3(
    loader,
    encoder: nn.Module,
    decoder_c1: nn.Module,
    decoder_full: nn.Module,
    args: argparse.Namespace,
) -> dict[str, float]:
    encoder.eval()
    decoder_c1.eval()
    decoder_full.eval()
    meters = {
        k: AverageMeter()
        for k in [
            "loss_full_rec",
            "loss_c1_rec",
            "loss_c1_consistency",
            "psnr_full",
            "psnr_c1",
            "psnr_full_c1_only",
        ]
    }
    device = next(encoder.parameters()).device
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        x_c1 = decoder_c1(c1)
        x_full = decoder_full(z_norm)
        x_full_c1_only = decoder_full(torch.cat([c1, torch.zeros_like(c2)], dim=1))
        loss_full = stage0_recon_loss(x_full, imgs)
        loss_c1 = stage0_recon_loss(x_c1, imgs)
        loss_cons = F.mse_loss(x_full_c1_only.float().clamp(0.0, 1.0), x_c1.float().clamp(0.0, 1.0), reduction="mean")
        bsz = imgs.shape[0]
        meters["loss_full_rec"].update(float(loss_full.item()), bsz)
        meters["loss_c1_rec"].update(float(loss_c1.item()), bsz)
        meters["loss_c1_consistency"].update(float(loss_cons.item()), bsz)
        meters["psnr_full"].update(batch_metric_mean(psnr_per_image(x_full, imgs)), bsz)
        meters["psnr_c1"].update(batch_metric_mean(psnr_per_image(x_c1, imgs)), bsz)
        meters["psnr_full_c1_only"].update(batch_metric_mean(psnr_per_image(x_full_c1_only, imgs)), bsz)
    return {k: v.avg for k, v in meters.items()}


def train_stage0_v3(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder_c1, decoder_full = build_models(args, cfg.device)
    load_initial_state(args, encoder, decoder_c1, decoder_full)
    print_stage0_v3_header(args, len(train_loader.dataset), len(val_loader.dataset))

    optimizer1 = optim.Adam(list(encoder.parameters()) + list(decoder_c1.parameters()), lr=float(args.lr_phase1))
    best_c1 = -1.0
    for epoch in range(1, int(args.phase1_epochs) + 1):
        encoder.train()
        decoder_c1.train()
        meters = {"loss": AverageMeter(), "loss_c1_rec": AverageMeter(), "psnr_c1": AverageMeter()}
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            loss, losses, outputs = phase1_forward(imgs, encoder, decoder_c1, args)
            optimizer1.zero_grad(set_to_none=True)
            loss.backward()
            optimizer1.step()
            bsz = imgs.shape[0]
            meters["loss"].update(float(losses["loss"].item()), bsz)
            meters["loss_c1_rec"].update(float(losses["loss_c1_rec"].item()), bsz)
            meters["psnr_c1"].update(batch_metric_mean(psnr_per_image(outputs["x_c1"], imgs)), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        print_epoch("stage0-v3 phase1", epoch, int(args.phase1_epochs), metrics, time.time() - t0)
        if should_validate_phase(args.val_every_phase1, epoch, int(args.phase1_epochs)):
            val_metrics = validate_stage0_v3(val_loader, encoder, decoder_c1, decoder_full, args)
            score = val_metrics["psnr_c1"]
            print(f"[stage0-v3 phase1 val {epoch:03d}] {format_metrics(val_metrics)} score=psnr_c1")
            if score > best_c1:
                best_c1 = score
                save_stage0_v3_checkpoint(
                    stage0_v3_ckpt_path(args, f"phase1_best_{args.version}"),
                    phase="phase1",
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    encoder=encoder,
                    decoder_c1=decoder_c1,
                    decoder_full=decoder_full,
                )
        if should_save_latest(args, epoch):
            save_stage0_v3_checkpoint(
                stage0_v3_ckpt_path(args, f"phase1_latest_{args.version}"),
                phase="phase1",
                epoch=epoch,
                args=args,
                metrics=metrics,
                encoder=encoder,
                decoder_c1=decoder_c1,
                decoder_full=decoder_full,
            )

    copy_c1_decoder_to_full(decoder_c1, decoder_full, int(args.c1_ch))
    attach_full_head_c1_mask(decoder_full, int(args.c1_ch))
    freeze_module(decoder_c1)
    optimizer2 = optim.Adam(list(encoder.parameters()) + list(decoder_full.parameters()), lr=float(args.lr_phase2))
    best_full = -1.0
    for epoch in range(1, int(args.phase2_epochs) + 1):
        encoder.train()
        decoder_full.train()
        decoder_c1.eval()
        meters = {
            k: AverageMeter()
            for k in [
                "loss",
                "loss_full_rec",
                "loss_c1_anchor",
                "loss_c1_consistency",
                "psnr_full",
                "psnr_c1_anchor",
                "psnr_full_c1_only",
            ]
        }
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            loss, losses, outputs = phase2_forward(imgs, encoder, decoder_c1, decoder_full, args)
            optimizer2.zero_grad(set_to_none=True)
            loss.backward()
            optimizer2.step()
            resync_full_head_c1(decoder_c1, decoder_full, int(args.c1_ch))
            bsz = imgs.shape[0]
            for k, value in losses.items():
                meters[k].update(float(value.item()), bsz)
            meters["psnr_full"].update(batch_metric_mean(psnr_per_image(outputs["x_full"], imgs)), bsz)
            meters["psnr_c1_anchor"].update(batch_metric_mean(psnr_per_image(outputs["x_c1_anchor"], imgs)), bsz)
            meters["psnr_full_c1_only"].update(batch_metric_mean(psnr_per_image(outputs["x_full_c1_only"], imgs)), bsz)
        metrics = {k: v.avg for k, v in meters.items()}
        print_epoch("stage0-v3 phase2", epoch, int(args.phase2_epochs), metrics, time.time() - t0)
        if should_validate_phase(args.val_every, epoch, int(args.phase2_epochs)):
            val_metrics = validate_stage0_v3(val_loader, encoder, decoder_c1, decoder_full, args)
            score = val_metrics["psnr_full"]
            print(f"[stage0-v3 phase2 val {epoch:03d}] {format_metrics(val_metrics)} score=psnr_full")
            if score > best_full:
                best_full = score
                save_stage0_v3_checkpoint(
                    stage0_v3_ckpt_path(args, f"best_{args.version}"),
                    phase="phase2",
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    encoder=encoder,
                    decoder_c1=decoder_c1,
                    decoder_full=decoder_full,
                )
        if should_save_latest(args, epoch):
            save_stage0_v3_checkpoint(
                stage0_v3_ckpt_path(args, f"latest_{args.version}"),
                phase="phase2",
                epoch=epoch,
                args=args,
                metrics=metrics,
                encoder=encoder,
                decoder_c1=decoder_c1,
                decoder_full=decoder_full,
            )

    save_stage0_v3_checkpoint(
        stage0_v3_ckpt_path(args, f"latest_{args.version}"),
        phase="phase2",
        epoch=int(args.phase2_epochs),
        args=args,
        metrics=metrics,
        encoder=encoder,
        decoder_c1=decoder_c1,
        decoder_full=decoder_full,
    )


def should_validate_phase(val_every: int, epoch: int, total_epochs: int) -> bool:
    return int(val_every) > 0 and (epoch % int(val_every) == 0 or epoch == int(total_epochs))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", type=str, default="collab-c1thenfull")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default="/workspace/yongjia/paper_code/CDDM/MY/jscc-no-awgn")
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default="")
    p.add_argument("--init-jscc-encoder", type=str, default="")
    p.add_argument("--init-c1-decoder", type=str, default="")
    p.add_argument("--init-full-decoder", type=str, default="")
    p.add_argument("--snr-db", type=float, default=12.0)
    p.add_argument("--latent-ch", type=int, default=36)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--phase1-epochs", type=int, default=200)
    p.add_argument("--phase2-epochs", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--lr-phase1", type=float, default=1e-4)
    p.add_argument("--lr-phase2", type=float, default=1e-4)
    p.add_argument("--lambda-phase1-c1", type=float, default=1.0)
    p.add_argument("--lambda-full", type=float, default=1.0)
    p.add_argument("--lambda-c1-anchor", type=float, default=0.2)
    p.add_argument("--lambda-c1-consistency", type=float, default=0.2)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--val-every-phase1", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.stage = 0
    check_stage0_v3_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(Path(args.save_dir) / f"stage0_v3_c{args.latent_ch}_no_awgn_{args.version}.log")
    setup_log_file(args.log_file)
    write_json(Path(args.save_dir) / "stage0_v3_args.json", vars(args))
    train_stage0_v3(args)


if __name__ == "__main__":
    main()
