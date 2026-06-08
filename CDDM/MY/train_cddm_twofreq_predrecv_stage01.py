#!/usr/bin/env python3
"""CDDM Autoencoder/Channel version of twofreq predicted-receiver Stage 01."""

from __future__ import annotations

import argparse
import builtins
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CDDM_ROOT = PROJECT_ROOT / "CDDM"
TRAIN_DIR = PROJECT_ROOT / "train"
for p in (PROJECT_ROOT, CDDM_ROOT, TRAIN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.meshgrid.*", category=UserWarning)

from Autoencoder.data.datasets import get_loader as get_cddm_loader  # noqa: E402
from Autoencoder.net.channel import Channel  # noqa: E402
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder  # noqa: E402
from train_hierarchical_swin_ar_awgn12 import ARReceiver  # noqa: E402
from train_route_a_sc import AverageMeter, GaussianBlur, TeeStream  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_log_file(path: str | None):
    if not path:
        return None
    abs_path = Path(path)
    if not abs_path.is_absolute():
        abs_path = PROJECT_ROOT / abs_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== CDDM twofreq predicted-receiver Stage 01 @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


@dataclass
class CDDMJSCCConfig:
    C: int
    SNRs: float
    channel_type: str
    batch_size: int
    test_batch: int
    num_workers: int
    val_num_workers: int
    train_data_dir: str
    test_data_dir: str
    CUDA: bool = True
    dataset: str = "DIV2K"
    loss_function: str = "MSE"
    image_dims: tuple[int, int, int] = (3, 256, 256)
    pin_memory: bool = True
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        self.device = torch.device("cuda:0" if self.CUDA and torch.cuda.is_available() else "cpu")
        self.encoder_kwargs = dict(
            img_size=(self.image_dims[1], self.image_dims[2]),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 192, 256, 320],
            depths=[2, 2, 6, 2],
            num_heads=[4, 6, 8, 10],
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=torch.nn.LayerNorm,
            patch_norm=True,
        )
        self.decoder_kwargs = dict(
            img_size=(self.image_dims[1], self.image_dims[2]),
            embed_dims=[320, 256, 192, 128],
            depths=[2, 6, 2, 2],
            num_heads=[10, 8, 6, 4],
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=torch.nn.LayerNorm,
            patch_norm=True,
        )


def psnr_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse = (x_hat.float().clamp(0.0, 1.0) - x.float()).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def default_jscc_root() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints" / "JSCC" / "DIV2K" / "MSE" / "SNRs")


def resolve_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def load_state(module: nn.Module, path: str, name: str) -> None:
    obj = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    missing, unexpected = module.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"{name} load mismatch: missing={missing}, unexpected={unexpected}")
    print(f"loaded {name}: {resolve_path(path)}")


def load_stage_checkpoint(
    encoder: nn.Module,
    decoder: nn.Module,
    predictor: nn.Module,
    path: str,
    device: torch.device,
) -> None:
    ckpt_path = resolve_path(path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    pred_state = ckpt.get("predictor_state_dict", ckpt.get("ar_state_dict"))
    if pred_state is None:
        raise KeyError(f"No predictor_state_dict/ar_state_dict in {ckpt_path}")
    predictor.load_state_dict(pred_state, strict=True)
    print(f"loaded stage checkpoint encoder/decoder/predictor: {ckpt_path}")
    print(f"  init stage={ckpt.get('stage', 'unknown')} epoch={ckpt.get('epoch', 'unknown')}")


def build_config(args: argparse.Namespace, batch_size: int) -> CDDMJSCCConfig:
    return CDDMJSCCConfig(
        C=int(args.C),
        SNRs=float(args.snr_db),
        channel_type=str(args.channel_type),
        batch_size=int(batch_size),
        test_batch=int(args.test_batch),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.val_num_workers),
        train_data_dir=resolve_path(os.path.join(args.data_dir, "DIV2K_train_HR")),
        test_data_dir=resolve_path(os.path.join(args.data_dir, "DIV2K_valid_HR")),
        CUDA=(not bool(args.cpu)),
    )


def cddm_low_channel_norm(
    channel: Channel,
    z_low: torch.Tensor,
    snr_db: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized CDDM-channel observation and global sqrt(pwr)."""
    y_complex, pwr, _h = channel.forward(z_low.float(), float(snr_db))
    y_norm = torch.cat((torch.real(y_complex), torch.imag(y_complex)), dim=2).float()
    scale = torch.sqrt(pwr.float().clamp_min(1e-12)).view(1).expand(z_low.shape[0])
    return y_norm, scale


def make_twofreq_latents(
    z: torch.Tensor,
    y4_norm: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_view = scale.float().view(-1, 1, 1, 1).clamp_min(1e-12)
    h_oracle = z[:, 4:16].float() / scale_view
    z_base = torch.zeros_like(z.float())
    z_base[:, :4] = y4_norm.float()
    z_oracle = torch.cat([y4_norm.float(), h_oracle], dim=1)
    return z_base, z_oracle, h_oracle


def run_batch(
    *,
    imgs: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    predictor: nn.Module,
    channel: Channel,
    blur: GaussianBlur,
    args: argparse.Namespace,
    train: bool,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    feature, _ = encoder(imgs)
    z = feature.float()
    y4_norm, scale = cddm_low_channel_norm(channel, z[:, :4], float(args.snr_db))
    z_base, z_oracle, h_oracle = make_twofreq_latents(z, y4_norm, scale)

    pred_dtype = next(predictor.parameters()).dtype
    _z_recv, pred_groups = predictor(
        y4_norm.to(dtype=pred_dtype),
        y4_raw=y4_norm.to(dtype=pred_dtype),
        y4_norm=y4_norm.to(dtype=pred_dtype),
        scale=scale,
        snr_db=torch.full((imgs.shape[0],), float(args.snr_db), device=imgs.device, dtype=torch.float32),
        z_gt=None,
        teacher_prob=0.0,
    )
    h_pred = torch.cat([g.float() for g in pred_groups], dim=1)
    z_recv_pred = torch.cat([y4_norm.float(), h_pred], dim=1)
    imgs_blur = blur(imgs.float()).clamp(0.0, 1.0)

    x_recv_pred = decoder(z_recv_pred).float().clamp(0.0, 1.0)
    x_base = decoder(z_base).float().clamp(0.0, 1.0)
    x_oracle = decoder(z_oracle).float().clamp(0.0, 1.0)

    loss_recv_pred = F.mse_loss(x_recv_pred, imgs.float())
    loss_base = F.mse_loss(x_base, imgs_blur)
    loss_oracle = F.mse_loss(x_oracle, imgs.float())
    h_target = h_oracle.detach() if bool(args.pred_target_detach) else h_oracle
    loss_pred = F.mse_loss(h_pred, h_target)
    loss = (
        float(args.lambda_recv) * loss_recv_pred
        + float(args.lambda_base) * loss_base
        + float(args.lambda_pred) * loss_pred
        + float(args.lambda_oracle) * loss_oracle
    )

    psnr_base = float(psnr_per_image(x_base, imgs.float()).mean().item())
    psnr_recv = float(psnr_per_image(x_recv_pred, imgs.float()).mean().item())
    psnr_oracle = float(psnr_per_image(x_oracle, imgs.float()).mean().item())
    stats = {
        "loss": float(loss.detach().item()),
        "loss_recv_pred": float(loss_recv_pred.detach().item()),
        "loss_base": float(loss_base.detach().item()),
        "loss_pred": float(loss_pred.detach().item()),
        "loss_oracle": float(loss_oracle.detach().item()),
        "psnr_base": psnr_base,
        "psnr_base_blur": float(psnr_per_image(x_base, imgs_blur).mean().item()),
        "psnr_recv": psnr_recv,
        "psnr_oracle": psnr_oracle,
        "gain_recv": psnr_recv - psnr_base,
        "gap_oracle_recv": psnr_oracle - psnr_recv,
        "h_pred_mse": float(F.mse_loss(h_pred, h_oracle).detach().item()),
        "h_pred_rms": float(h_pred.float().square().mean().sqrt().detach().item()),
        "h_oracle_rms": float(h_oracle.float().square().mean().sqrt().detach().item()),
    }
    return (loss if train else None), stats


def save_checkpoint(
    path: str,
    encoder: nn.Module,
    decoder: nn.Module,
    predictor: nn.Module,
    args: argparse.Namespace,
    metrics: dict,
    epoch: int,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "route": "cddm_twofreq_receiver_norm_swin_predrecv",
        "stage": str(args.stage),
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "source_jscc_encoder": resolve_path(args.encoder_ckpt),
        "source_jscc_decoder": resolve_path(args.decoder_ckpt),
        "channel": "CDDM Autoencoder Channel.forward on z[:,0:4], height split real/imag",
        "split_rule": "z_low=z[:,0:4], h=z[:,4:16]/sqrt(pwr_low)",
        "objective": (
            f"{float(args.lambda_recv):g}*L_recv_pred + {float(args.lambda_base):g}*L_base"
            f"(blur_sigma={float(args.blur_sigma):g}) + {float(args.lambda_pred):g}*L_pred"
            f"(target={'detach' if bool(args.pred_target_detach) else 'live'})"
            f" + {float(args.lambda_oracle):g}*L_oracle"
        ),
        "score_metric": "val_psnr_recv",
        "snr_db": float(args.snr_db),
        "init_stage_ckpt": resolve_path(args.init_stage_ckpt) if str(args.init_stage_ckpt) else "",
    }
    torch.save(
        {
            **common,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "ar_state_dict": predictor.state_dict(),
        },
        out,
    )
    split_dir = out.parent
    torch.save({**common, "part": "cddm_jscc_encoder", "state_dict": encoder.state_dict()}, split_dir / "sc_encoder_hier_c16.pth")
    torch.save({**common, "part": "cddm_jscc_decoder", "state_dict": decoder.state_dict()}, split_dir / "sc_decoder_hier_c16.pth")
    torch.save({**common, "part": "high_predictor", "state_dict": predictor.state_dict()}, split_dir / "high_predictor_hier_c16_snr6.pth")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--C", type=int, default=16)
    p.add_argument("--snr_db", type=float, default=6.0)
    p.add_argument("--channel_type", type=str, default="awgn", choices=["awgn"])
    p.add_argument("--jscc_root", type=str, default=default_jscc_root())
    p.add_argument("--encoder_ckpt", type=str, default="")
    p.add_argument("--decoder_ckpt", type=str, default="")

    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--test_batch", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--eval_every_epochs", type=int, default=5)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--decoder_lr", type=float, default=2e-5)
    p.add_argument("--predictor_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--pred_hidden", type=int, default=160)
    p.add_argument("--pred_depth", type=int, default=4)
    p.add_argument("--pred_use_scale", action="store_true", default=False)
    p.add_argument("--blur_kernel", type=int, default=7)
    p.add_argument("--blur_sigma", type=float, default=1.0)
    p.add_argument("--baseline_psnr", type=float, default=21.0085)
    p.add_argument("--lambda_recv", type=float, default=1.0)
    p.add_argument("--lambda_base", type=float, default=0.2)
    p.add_argument("--lambda_pred", type=float, default=0.0)
    p.add_argument("--lambda_oracle", type=float, default=0.03)
    p.add_argument("--pred_target_detach", action="store_true", default=True)
    p.add_argument("--no_pred_target_detach", action="store_false", dest="pred_target_detach")
    p.add_argument("--init_stage_ckpt", type=str, default="", help="Optional 01C-style checkpoint to initialize encoder/decoder/predictor")
    p.add_argument("--amp_dtype", type=str, default="none", choices=["none", "bfloat16", "float16"])
    p.add_argument("--stage_name", type=str, default="cddm_twofreq_receiver_norm_swin_predrecv_stage01")
    p.add_argument("--ckpt_prefix", type=str, default="cddm_twofreq_receiver_norm_swin_predrecv_stage01")

    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--save_dir",
        type=str,
        default="CDDM/MY/checkpoints-ar/twofreq_receiver_norm_swin_snr6_v1/01C_receiver_norm_swin_predrecv_stage01",
    )
    p.add_argument(
        "--log_file",
        type=str,
        default="CDDM/MY/checkpoints-ar/twofreq_receiver_norm_swin_snr6_v1/01C_receiver_norm_swin_predrecv_stage01/train.log",
    )
    args = p.parse_args()
    if not args.encoder_ckpt:
        args.encoder_ckpt = os.path.join(args.jscc_root, f"encoder_snr{args.snr_db:g}_channel_{args.channel_type}_C{args.C}.pt")
    if not args.decoder_ckpt:
        args.decoder_ckpt = os.path.join(args.jscc_root, f"decoder_snr{args.snr_db:g}_channel_{args.channel_type}_C{args.C}.pt")
    args.stage = str(args.stage_name)
    return args


def main() -> None:
    args = parse_args()
    save_dir = Path(resolve_path(args.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_log_file(args.log_file)
    seed_everything(int(args.seed))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_enabled = device.type == "cuda" and str(args.amp_dtype) != "none"
    amp_dtype = torch.bfloat16 if str(args.amp_dtype) == "bfloat16" else torch.float16

    data_cfg = build_config(args, int(args.batch_size))
    val_cfg = build_config(args, int(args.batch_size))
    train_ds_loader, val_loader = get_cddm_loader(data_cfg)
    train_loader = train_ds_loader

    encoder = JSCC_encoder(data_cfg, data_cfg.C).to(device)
    decoder = JSCC_decoder(data_cfg, data_cfg.C).to(device)
    predictor = ARReceiver(hidden=int(args.pred_hidden), depth=int(args.pred_depth), use_scale=bool(args.pred_use_scale)).to(device)
    channel = Channel(data_cfg)
    load_state(encoder, args.encoder_ckpt, "CDDM JSCC encoder")
    load_state(decoder, args.decoder_ckpt, "CDDM JSCC decoder")
    if str(args.init_stage_ckpt):
        load_stage_checkpoint(encoder, decoder, predictor, str(args.init_stage_ckpt), device)

    print(f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}, stage={args.stage}, train/test snr={float(args.snr_db):g}dB")
    print(
        "rule: CDDM encoder z, z_low=z[:,0:4] transmitted by CDDM Channel, "
        "h=z[:,4:16]/sqrt(pwr_low), h_pred=P(y), score=val_psnr_recv, PSNR=mean(per-image PSNR)"
    )
    print(f"train={len(train_loader.dataset)} valid={len(val_loader.dataset)} batch={args.batch_size} crop=256")
    print(f"encoder_ckpt={resolve_path(args.encoder_ckpt)}")
    print(f"decoder_ckpt={resolve_path(args.decoder_ckpt)}")
    print(f"predictor=ARReceiver hidden={args.pred_hidden} depth={args.pred_depth} use_scale={bool(args.pred_use_scale)} params={sum(p.numel() for p in predictor.parameters())}")
    print(
        f"loss={float(args.lambda_recv):g}*L_recv_pred + "
        f"{float(args.lambda_base):g}*L_base(blur_sigma={float(args.blur_sigma):g}) + "
        f"{float(args.lambda_pred):g}*L_pred(target={'detach' if bool(args.pred_target_detach) else 'live'}) + "
        f"{float(args.lambda_oracle):g}*L_oracle"
    )

    params = [
        {"params": list(encoder.parameters()), "lr": float(args.lr), "name": "encoder"},
        {"params": list(decoder.parameters()), "lr": float(args.decoder_lr), "name": "decoder"},
        {"params": list(predictor.parameters()), "lr": float(args.predictor_lr), "name": "predictor"},
    ]
    optimizer = optim.AdamW(params, weight_decay=float(args.weight_decay), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    print(
        "optimizer_groups="
        + ", ".join(f"{g['name']}:n={sum(p.numel() for p in g['params'])}:lr={g['lr']}" for g in params)
    )
    blur = GaussianBlur(3, int(args.blur_kernel), float(args.blur_sigma)).to(device)
    meter_keys = (
        "loss",
        "loss_recv_pred",
        "loss_base",
        "loss_pred",
        "loss_oracle",
        "psnr_base",
        "psnr_base_blur",
        "psnr_recv",
        "psnr_oracle",
        "gain_recv",
        "gap_oracle_recv",
        "h_pred_mse",
        "h_pred_rms",
        "h_oracle_rms",
    )
    best = -1.0
    ckpt_prefix = str(args.ckpt_prefix)
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train(True)
        decoder.train(True)
        predictor.train(True)
        meters = {k: AverageMeter() for k in meter_keys}
        for bi, batch in enumerate(train_loader):
            if int(args.max_train_batches) > 0 and bi >= int(args.max_train_batches):
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss, stats = run_batch(
                    imgs=imgs,
                    encoder=encoder,
                    decoder=decoder,
                    predictor=predictor,
                    channel=channel,
                    blur=blur,
                    args=args,
                    train=True,
                )
            assert loss is not None
            scaler.scale(loss).backward()
            if float(args.clip_grad_norm) > 0:
                scaler.unscale_(optimizer)
                trainable_params = [p for group in optimizer.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.clip_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            for k in meters:
                meters[k].update(stats[k], imgs.shape[0])

        metrics = {f"train_{k}": v.avg for k, v in meters.items()}
        do_eval = epoch == 1 or epoch % max(1, int(args.eval_every_epochs)) == 0 or epoch == int(args.epochs)
        if do_eval:
            encoder.eval()
            decoder.eval()
            predictor.eval()
            val_meters = {k: AverageMeter() for k in meter_keys}
            with torch.no_grad():
                for bi, batch in enumerate(val_loader):
                    if int(args.max_val_batches) > 0 and bi >= int(args.max_val_batches):
                        break
                    imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
                    imgs = imgs.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        _loss, stats = run_batch(
                            imgs=imgs,
                            encoder=encoder,
                            decoder=decoder,
                            predictor=predictor,
                            channel=channel,
                            blur=blur,
                            args=args,
                            train=False,
                        )
                    for k in val_meters:
                        val_meters[k].update(stats[k], imgs.shape[0])
            metrics.update({f"val_{k}": v.avg for k, v in val_meters.items()})
            score = metrics["val_psnr_recv"]
            is_best = score > best
            if is_best:
                best = score
                save_checkpoint(str(save_dir / f"{ckpt_prefix}_best.pth"), encoder, decoder, predictor, args, metrics, epoch)
            save_checkpoint(str(save_dir / f"{ckpt_prefix}_latest.pth"), encoder, decoder, predictor, args, metrics, epoch)
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"recv_loss={meters['loss_recv_pred'].avg:.6f} pred_loss={meters['loss_pred'].avg:.6f} | "
                f"base={metrics['val_psnr_base']:.4f} recv={metrics['val_psnr_recv']:.4f} "
                f"oracle={metrics['val_psnr_oracle']:.4f} gain_recv={metrics['val_gain_recv']:+.4f} "
                f"gap_oracle_recv={metrics['val_gap_oracle_recv']:+.4f} base_blur={metrics['val_psnr_base_blur']:.4f} "
                f"pred={metrics['val_loss_pred']:.4f} h_mse={metrics['val_h_pred_mse']:.4f} h_pred_rms={metrics['val_h_pred_rms']:.4f} "
                f"h_oracle_rms={metrics['val_h_oracle_rms']:.4f} score(val_recv)={score:.4f} "
                f"gain_prev_baseline={metrics['val_psnr_recv'] - float(args.baseline_psnr):+.4f} "
                f"{'BEST' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] loss={meters['loss'].avg:.6f} "
                f"base={meters['psnr_base'].avg:.4f} recv={meters['psnr_recv'].avg:.4f} "
                f"oracle={meters['psnr_oracle'].avg:.4f} gain_recv={meters['gain_recv'].avg:+.4f} "
                f"pred={meters['loss_pred'].avg:.4f} h_mse={meters['h_pred_mse'].avg:.4f}"
            )
    print(f"best_val_psnr_recv={best:.4f}")


if __name__ == "__main__":
    main()
