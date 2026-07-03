from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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


def check_stage0_v2_args(args: argparse.Namespace) -> None:
    if int(args.latent_ch) != 36:
        raise ValueError("stage0-v2 trains a C=36 JSCC; use --latent-ch 36.")
    if int(args.c1_ch) != 16:
        raise ValueError("stage0-v2 distills the first 16 channels; use --c1-ch 16.")
    if int(args.latent_ch) - int(args.c1_ch) != 20:
        raise ValueError("stage0-v2 expects C2=20.")
    if int(args.teacher_latent_ch) != int(args.c1_ch):
        raise ValueError("--teacher-latent-ch must match --c1-ch for C1 distillation.")
    if int(args.latent_h) != 16 or int(args.latent_w) != 16:
        raise ValueError("The CDDM JSCC encoder is expected to produce 16x16 latents for 256x256 crops.")


def build_jscc_models(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module]:
    cfg = cvq_io.build_config(args)
    encoder = JSCC_encoder(cfg, int(args.latent_ch)).to(device)
    decoder = JSCC_decoder(cfg, int(args.latent_ch)).to(device)
    return encoder, decoder


def build_teacher_models(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module]:
    teacher_args = copy.copy(args)
    teacher_args.latent_ch = int(args.teacher_latent_ch)
    teacher_args.c1_ch = int(args.teacher_latent_ch)
    cfg = cvq_io.build_config(teacher_args)
    encoder = JSCC_encoder(cfg, int(args.teacher_latent_ch)).to(device)
    decoder = JSCC_decoder(cfg, int(args.teacher_latent_ch)).to(device)
    return encoder, decoder


def load_teacher_checkpoint(
    path: str,
    encoder: nn.Module,
    decoder: nn.Module,
    device: torch.device,
) -> dict:
    ckpt_path_abs = resolve_path(path)
    ckpt = torch.load(ckpt_path_abs, map_location="cpu", weights_only=False)
    if "encoder_state_dict" not in ckpt or "decoder_state_dict" not in ckpt:
        raise RuntimeError(f"teacher checkpoint missing encoder/decoder state dicts: {ckpt_path_abs}")
    cvq_io.load_state(encoder, ckpt["encoder_state_dict"], "teacher encoder", strict=True)
    cvq_io.load_state(decoder, ckpt["decoder_state_dict"], "teacher decoder", strict=True)
    encoder.to(device).eval()
    decoder.to(device).eval()
    for module in (encoder, decoder):
        for p in module.parameters():
            p.requires_grad_(False)
    print(f"loaded teacher checkpoint: {ckpt_path_abs}")
    return ckpt


def stage0_v2_ckpt_path(args: argparse.Namespace, suffix: str) -> str:
    return str(Path(resolve_path(args.save_dir)) / f"cvq_v2_c{int(args.latent_ch)}_stage0_v2_{suffix}.pth")


def save_stage0_v2_checkpoint(
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
            "route": "cvq_v2_stage0_v2_c36_no_channel_c1_distill_c16_c2_recon",
            "stage": "stage0_v2",
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
            "teacher_ckpt": resolve_path(args.teacher_ckpt),
            "latent_ch": int(args.latent_ch),
            "c1_ch": int(args.c1_ch),
            "teacher_latent_ch": int(args.teacher_latent_ch),
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        },
        out,
    )
    print(f"saved checkpoint: {out}")


def stage0_recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon.float().clamp(0.0, 1.0), target.float(), reduction="mean")


def latent_distill_loss(student_c1: torch.Tensor, teacher_z: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_c1.float(), teacher_z.float(), reduction="mean")


def print_stage0_v2_header(args: argparse.Namespace, train_n: int, val_n: int, teacher_ckpt: dict) -> None:
    teacher_epoch = teacher_ckpt.get("epoch", "<unknown>")
    teacher_metrics = teacher_ckpt.get("metrics", {})
    teacher_psnr = teacher_metrics.get("psnr_full", teacher_metrics.get("psnr", "<unknown>"))
    print("=== Stage 0-v2 | C36 JSCC no-channel, C1 distills C16 teacher ===")
    print("实验设计=C36 JSCC without channel; channels 0..15 distill C16 no-AWGN teacher; channels 16..35 learn through full C36 reconstruction")
    print(
        "loss设计="
        f"{float(args.lambda_full):g}*L_rec_full+"
        f"{float(args.lambda_c1):g}*L_rec_c1_only+"
        f"{float(args.lambda_drop):g}*L_rec_nested_drop+"
        f"{float(args.lambda_c1_distill_latent):g}*L_distill_c1_latent+"
        f"{float(args.lambda_c1_distill_recon):g}*L_distill_c1_recon"
    )
    print("模块选择=student JSCC_encoder/JSCC_decoder C36; frozen teacher JSCC_encoder/JSCC_decoder C16")
    print(f"device={'cpu' if args.cpu else 'cuda:0'}")
    print(f"save_dir={resolve_path(args.save_dir)}")
    print(f"teacher_ckpt={resolve_path(args.teacher_ckpt)} teacher_epoch={teacher_epoch} teacher_psnr={teacher_psnr}")
    print(f"init_ckpt={resolve_path(args.init_ckpt) if args.init_ckpt else 'none'}")
    print(f"init_jscc_encoder={resolve_path(args.init_jscc_encoder) if args.init_jscc_encoder else 'random'}")
    print(f"init_jscc_decoder={resolve_path(args.init_jscc_decoder) if args.init_jscc_decoder else 'random'}")
    print(f"latent_ch={args.latent_ch} C1={args.c1_ch} C2={int(args.latent_ch) - int(args.c1_ch)} codebook=none channel=none")
    print(f"power_norm=all_latents_scaled_by_C1_mean_square teacher_power_norm=C16_mean_square")
    print(f"c2_dropout=prefix_nested_uniform_k_0_to_{int(args.latent_ch) - int(args.c1_ch)} ratio={float(args.nested_drop_ratio):g}")
    print("loss_recon_reduction=F.mse_loss(clamp(recon,0,1), input, reduction='sum') / batch")
    print("loss_distill_latent_reduction=F.mse_loss(student_c1_norm, teacher_z_norm, reduction='sum') / batch")
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}")


@torch.no_grad()
def teacher_forward(
    imgs: torch.Tensor,
    teacher_encoder: nn.Module,
    teacher_decoder: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_args = copy.copy(args)
    teacher_args.latent_ch = int(args.teacher_latent_ch)
    teacher_args.c1_ch = int(args.teacher_latent_ch)
    _z_teacher, z_teacher_norm, _teacher_power = cvq_io.encode_normalized(imgs, teacher_encoder, teacher_args)
    x_teacher = teacher_decoder(z_teacher_norm).clamp(0.0, 1.0)
    return z_teacher_norm.detach(), x_teacher.detach()


def compute_losses(
    imgs: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    teacher_encoder: nn.Module,
    teacher_decoder: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
    c1, c2 = split_c1_c2(z_norm, args)
    z_teacher, x_teacher = teacher_forward(imgs, teacher_encoder, teacher_decoder, args)

    x_full = decoder(z_norm)
    x_c1_only = decoder(torch.cat([c1, torch.zeros_like(c2)], dim=1))
    x_c1_teacher = decoder(torch.cat([z_teacher, torch.zeros_like(c2)], dim=1))
    mask = sample_c2_nested_prefix_mask(imgs.shape[0], c2.shape[1], float(args.nested_drop_ratio), imgs.device, c2.dtype)
    x_nested_drop = decoder(torch.cat([c1, c2 * mask], dim=1))

    loss_full_rec = recon_loss(x_full, imgs)
    loss_c1_rec = recon_loss(x_c1_only, imgs)
    loss_drop_rec = recon_loss(x_nested_drop, imgs)
    loss_c1_distill_latent = latent_distill_loss(c1, z_teacher)
    loss_c1_distill_recon = recon_loss(x_c1_only, x_teacher)

    loss = (
        float(args.lambda_full) * loss_full_rec
        + float(args.lambda_c1) * loss_c1_rec
        # + float(args.lambda_drop) * loss_drop_rec
        + float(args.lambda_c1_distill_latent) * loss_c1_distill_latent
        # + float(args.lambda_c1_distill_recon) * loss_c1_distill_recon
    )
    losses = {
        "loss": loss,
        "loss_full_rec": loss_full_rec,
        "loss_c1_rec": loss_c1_rec,
        "loss_drop_rec": loss_drop_rec,
        "loss_c1_distill_latent": loss_c1_distill_latent,
        "loss_c1_distill_recon": loss_c1_distill_recon,
    }
    outputs = {
        "x_full": x_full,
        "x_c1_only": x_c1_only,
        "x_nested_drop": x_nested_drop,
        "x_teacher": x_teacher,
        "mask": mask,
    }
    return loss, losses, outputs


@torch.no_grad()
def validate_stage0_v2(
    loader,
    encoder: nn.Module,
    decoder: nn.Module,
    teacher_encoder: nn.Module,
    teacher_decoder: nn.Module,
    args: argparse.Namespace,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    teacher_encoder.eval()
    teacher_decoder.eval()
    meter_names = [
        "loss",
        "loss_full_rec",
        "loss_c1_rec",
        "loss_drop_rec",
        "loss_c1_distill_latent",
        "loss_c1_distill_recon",
        "psnr_full",
        "psnr_c1_only",
        "psnr_drop",
        "psnr_teacher",
        "drop_keep",
    ]
    meters = {k: AverageMeter() for k in meter_names}
    device = next(encoder.parameters()).device
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _loss, losses, outputs = compute_losses(imgs, encoder, decoder, teacher_encoder, teacher_decoder, args)
        bsz = imgs.shape[0]
        for k, v in losses.items():
            meters[k].update(float(v.item()), bsz)
        meters["psnr_full"].update(batch_metric_mean(psnr_per_image(outputs["x_full"], imgs)), bsz)
        meters["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(outputs["x_c1_only"], imgs)), bsz)
        meters["psnr_drop"].update(batch_metric_mean(psnr_per_image(outputs["x_nested_drop"], imgs)), bsz)
        meters["psnr_teacher"].update(batch_metric_mean(psnr_per_image(outputs["x_teacher"], imgs)), bsz)
        meters["drop_keep"].update(float(outputs["mask"].float().mean().item()), bsz)
    return {k: v.avg for k, v in meters.items()}


def train_stage0_v2(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder = build_jscc_models(args, cfg.device)
    teacher_encoder, teacher_decoder = build_teacher_models(args, cfg.device)
    teacher_ckpt = load_teacher_checkpoint(args.teacher_ckpt, teacher_encoder, teacher_decoder, cfg.device)

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
    print_stage0_v2_header(args, len(train_loader.dataset), len(val_loader.dataset), teacher_ckpt)

    for epoch in range(1, int(args.epochs) + 1):
        encoder.train()
        decoder.train()
        teacher_encoder.eval()
        teacher_decoder.eval()
        meter_names = [
            "loss",
            "loss_full_rec",
            "loss_c1_rec",
            "loss_drop_rec",
            "loss_c1_distill_latent",
            "loss_c1_distill_recon",
            "psnr_full",
            "psnr_c1_only",
            "psnr_drop",
            "psnr_teacher",
            "drop_keep",
        ]
        meters = {k: AverageMeter() for k in meter_names}
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            loss, losses, outputs = compute_losses(imgs, encoder, decoder, teacher_encoder, teacher_decoder, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bsz = imgs.shape[0]
            for k, v in losses.items():
                meters[k].update(float(v.item()), bsz)
            meters["psnr_full"].update(batch_metric_mean(psnr_per_image(outputs["x_full"], imgs)), bsz)
            meters["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(outputs["x_c1_only"], imgs)), bsz)
            meters["psnr_drop"].update(batch_metric_mean(psnr_per_image(outputs["x_nested_drop"], imgs)), bsz)
            meters["psnr_teacher"].update(batch_metric_mean(psnr_per_image(outputs["x_teacher"], imgs)), bsz)
            meters["drop_keep"].update(float(outputs["mask"].float().mean().item()), bsz)

        metrics = {k: v.avg for k, v in meters.items()}
        print_epoch("stage0-v2", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate_stage0_v2(val_loader, encoder, decoder, teacher_encoder, teacher_decoder, args)
            score = val_metrics["psnr_full"]
            print(f"[stage0-v2 val {epoch:03d}] {format_metrics(val_metrics)} score=psnr_full")
            if score > best:
                best = score
                save_stage0_v2_checkpoint(
                    stage0_v2_ckpt_path(args, f"best_{args.version}"),
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    encoder=encoder,
                    decoder=decoder,
                )
        if should_save_latest(args, epoch):
            save_stage0_v2_checkpoint(
                stage0_v2_ckpt_path(args, f"latest_{args.version}"),
                epoch=epoch,
                args=args,
                metrics=metrics,
                encoder=encoder,
                decoder=decoder,
            )

    save_stage0_v2_checkpoint(
        stage0_v2_ckpt_path(args, f"latest_{args.version}"),
        epoch=int(args.epochs),
        args=args,
        metrics=metrics,
        encoder=encoder,
        decoder=decoder,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", type=str, default="c1distill-c16")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default="/workspace/yongjia/paper_code/CDDM/MY/jscc-no-awgn")
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--teacher-ckpt", type=str, default="MY/jscc-no-awgn/cvq_v2_c16_stage0_best_no-c1.pth")
    p.add_argument("--teacher-latent-ch", type=int, default=16)
    p.add_argument("--init-ckpt", type=str, default="MY/jscc-no-awgn/cvq_v2_c36_stage0_best.pth")
    p.add_argument("--init-jscc-encoder", type=str, default="")
    p.add_argument("--init-jscc-decoder", type=str, default="")
    p.add_argument("--snr-db", type=float, default=12.0)
    p.add_argument("--latent-ch", type=int, default=36)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-c1", type=float, default=0.2)
    p.add_argument("--lambda-drop", type=float, default=0.0)
    p.add_argument("--lambda-full", type=float, default=1.0)
    p.add_argument("--lambda-c1-distill-latent", type=float, default=1.0)
    p.add_argument("--lambda-c1-distill-recon", type=float, default=0.2)
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
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.stage = 0
    check_stage0_v2_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(Path(args.save_dir) / f"stage0_v2_c{args.latent_ch}_no_awgn_{args.version}.log")
    setup_log_file(args.log_file)
    write_json(Path(args.save_dir) / "stage0_v2_args.json", vars(args))
    train_stage0_v2(args)


if __name__ == "__main__":
    main()
