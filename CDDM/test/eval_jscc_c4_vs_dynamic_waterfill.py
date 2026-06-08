#!/usr/bin/env python3
"""Compare C4 JSCC and C16->K4 dynamic water-filling on DIV2K valid PSNR."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Sequence


def _apply_cuda_visible_devices_before_torch() -> None:
    if "--gpu" not in sys.argv:
        return
    i = sys.argv.index("--gpu")
    if i + 1 >= len(sys.argv):
        return
    val = sys.argv[i + 1]
    if not val.startswith("-"):
        os.environ["CUDA_VISIBLE_DEVICES"] = val


_apply_cuda_visible_devices_before_torch()

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

CDDM_ROOT = Path(__file__).resolve().parents[1]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.meshgrid.*", category=UserWarning)

from Autoencoder.data.datasets import FlatImageFolder  # noqa: E402
from Autoencoder.net.channel import Channel  # noqa: E402
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder  # noqa: E402
from MY.train_dynamic_subspace_waterfill import DynamicSubspaceChannel  # noqa: E402


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else CDDM_ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float_csv(s: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in s.split(",") if x.strip())
    if not vals:
        raise argparse.ArgumentTypeError("empty SNR list")
    return vals


def build_cddm_cfg(*, C: int, snr_db: float, channel_type: str, device: torch.device) -> argparse.Namespace:
    cfg = argparse.Namespace()
    cfg.loss_function = "MSE"
    cfg.dataset = "DIV2K"
    cfg.C = int(C)
    cfg.SNRs = float(snr_db)
    cfg.seed = 1024
    cfg.CUDA = device.type == "cuda"
    cfg.device = device
    cfg.channel_type = channel_type.lower()
    cfg.image_dims = (3, 256, 256)
    cfg.encoder_kwargs = dict(
        img_size=(256, 256),
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
    cfg.decoder_kwargs = dict(
        img_size=(256, 256),
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
    return cfg


def load_plain_state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.get("state_dict", obj) if isinstance(obj, dict) else obj


def jscc_decoder_input(
    feature: torch.Tensor,
    channel: Channel,
    snr_db: float,
    channel_type: str,
) -> torch.Tensor:
    y_complex, pwr, h = channel.forward(feature, snr_db)
    if channel_type == "rayleigh":
        sigma_square = 1.0 / (10 ** (snr_db / 10))
        y_complex = torch.conj(h) * y_complex / (torch.abs(h) ** 2 + sigma_square)
    elif channel_type != "awgn":
        raise ValueError(f"unsupported channel_type={channel_type}")
    return torch.cat((torch.real(y_complex), torch.imag(y_complex)), dim=2) * torch.sqrt(pwr)


def psnr_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse = (x_hat.float().clamp(0.0, 1.0) - x.float()).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def make_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    ds = FlatImageFolder(
        root=str(resolve_path(args.test_dir)),
        transform=transforms.Compose(
            [
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
            ]
        ),
    )
    if int(args.max_images) > 0:
        ds.paths = ds.paths[: int(args.max_images)]
    workers = int(args.num_workers)
    return DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )


@torch.inference_mode()
def eval_jscc_c4(
    *,
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
    eval_snr: float,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> dict[str, float]:
    cfg = build_cddm_cfg(C=args.jscc_C, snr_db=eval_snr, channel_type=args.channel_type, device=device)
    encoder = JSCC_encoder(cfg, cfg.C).to(device)
    decoder = JSCC_decoder(cfg, cfg.C).to(device)
    encoder.load_state_dict(load_plain_state(resolve_path(args.jscc_encoder)), strict=True)
    decoder.load_state_dict(load_plain_state(resolve_path(args.jscc_decoder)), strict=True)
    encoder.eval()
    decoder.eval()
    channel = Channel(cfg)

    psnr_sum = 0.0
    seen = 0
    for imgs, _ in tqdm(loader, dynamic_ncols=True, leave=False, desc=f"JSCC C{cfg.C} SNR={eval_snr:g}"):
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            feature, _ = encoder(imgs)
            dec_in = jscc_decoder_input(feature, channel, float(eval_snr), args.channel_type)
            recon = decoder(dec_in).clamp(0.0, 1.0)
        psnr = psnr_per_image(recon, imgs)
        psnr_sum += float(psnr.sum().cpu())
        seen += int(imgs.shape[0])
    return {"psnr": psnr_sum / max(1, seen), "n": float(seen)}


def build_dynamic_from_ckpt(
    *,
    args: argparse.Namespace,
    ckpt: dict,
    device: torch.device,
    eval_snr: float,
) -> tuple[JSCC_encoder, JSCC_decoder, DynamicSubspaceChannel, Channel, argparse.Namespace]:
    ckpt_args = ckpt.get("args", {})
    C = int(args.dynamic_C if args.dynamic_C > 0 else ckpt_args.get("C", 16))
    out_ch = int(args.dynamic_out_ch if args.dynamic_out_ch > 0 else ckpt_args.get("subspace_out_ch", 4))
    num_modes = int(args.dynamic_modes if args.dynamic_modes > 0 else ckpt_args.get("num_modes", 4))
    router_hidden = int(ckpt_args.get("router_hidden", 64))
    min_power = float(ckpt_args.get("min_power", 0.05))
    channel_impl = str(args.dynamic_channel_impl or ckpt_args.get("channel_impl", "real_awgn"))

    cfg = build_cddm_cfg(C=C, snr_db=eval_snr, channel_type=args.channel_type, device=device)
    encoder = JSCC_encoder(cfg, cfg.C).to(device)
    decoder = JSCC_decoder(cfg, cfg.C).to(device)
    dyn = DynamicSubspaceChannel(
        in_ch=C,
        out_ch=out_ch,
        num_modes=num_modes,
        router_hidden=router_hidden,
        min_power=min_power,
        channel_impl=channel_impl,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    dyn.load_state_dict(ckpt["dynamic_state_dict"], strict=True)
    encoder.eval()
    decoder.eval()
    dyn.eval()
    channel = Channel(cfg)
    return encoder, decoder, dyn, channel, cfg


@torch.inference_mode()
def eval_dynamic(
    *,
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
    eval_snr: float,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> dict[str, float]:
    ckpt = torch.load(resolve_path(args.dynamic_ckpt), map_location="cpu", weights_only=False)
    encoder, decoder, dyn, channel, cfg = build_dynamic_from_ckpt(
        args=args,
        ckpt=ckpt,
        device=device,
        eval_snr=eval_snr,
    )

    psnr_sum = 0.0
    seen = 0
    usage_sum = torch.zeros(dyn.num_modes, dtype=torch.float64)
    usage_soft_sum = torch.zeros(dyn.num_modes, dtype=torch.float64)
    p_sel_sum = torch.zeros(dyn.out_ch, dtype=torch.float64)
    p_sel_count = 0
    aat_err = 0.0

    for imgs, _ in tqdm(loader, dynamic_ncols=True, leave=False, desc=f"Dynamic C{cfg.C}->K{dyn.out_ch} SNR={eval_snr:g}"):
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            feature, _ = encoder(imgs)
            z_hat, aux = dyn(
                feature.float(),
                channel=channel,
                snr_db=float(eval_snr),
                tau=1.0,
                train_hard_argmax=False,
            )
            recon = decoder(z_hat).float().clamp(0.0, 1.0)
        psnr = psnr_per_image(recon, imgs)
        bs = int(imgs.shape[0])
        psnr_sum += float(psnr.sum().cpu())
        seen += bs

        usage_sum += aux["pi"].detach().float().sum(dim=0).cpu().double()
        usage_soft_sum += aux["prob"].detach().float().sum(dim=0).cpu().double()
        p_sel_sum += aux["p_sel"].detach().float().sum(dim=0).cpu().double()
        p_sel_count += bs
        gram = torch.einsum("koc,kdc->kod", aux["A_all"].float(), aux["A_all"].float())
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).view(1, gram.shape[-1], gram.shape[-1])
        aat_err = max(aat_err, float((gram - eye).abs().amax().item()))

    out: dict[str, float] = {
        "psnr": psnr_sum / max(1, seen),
        "n": float(seen),
        "aat_err": aat_err,
    }
    usage = usage_sum / max(1, seen)
    usage_soft = usage_soft_sum / max(1, seen)
    p_sel = p_sel_sum / max(1, p_sel_count)
    for i, val in enumerate(usage.tolist()):
        out[f"usage{i}"] = float(val)
    for i, val in enumerate(usage_soft.tolist()):
        out[f"usage_soft{i}"] = float(val)
    for i, val in enumerate(p_sel.tolist()):
        out[f"p_sel_mean{i}"] = float(val)
    p_all = dyn.get_power().detach().float().cpu()
    out["p_all_min"] = float(p_all.min().item())
    out["p_all_max"] = float(p_all.max().item())
    return out


def write_csv(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def format_vec(metrics: dict[str, float], prefix: str, n: int) -> str:
    return "[" + ",".join(f"{metrics.get(f'{prefix}{i}', float('nan')):.3f}" for i in range(n)) + "]"


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--jscc-encoder", type=str, default="checkpoints/JSCC/DIV2K/MSE/SNRs/encoder_snr6_channel_awgn_C4.pt")
    p.add_argument("--jscc-decoder", type=str, default="checkpoints/JSCC/DIV2K/MSE/SNRs/decoder_snr6_channel_awgn_C4.pt")
    p.add_argument("--jscc-C", type=int, default=4)
    p.add_argument(
        "--dynamic-ckpt",
        type=str,
        default="MY/checkpoints-dynamic/dynamic_subspace_waterfill_snr6_c16_k4_realawgn_v2/dynamic_subspace_waterfill_best.pth",
    )
    p.add_argument("--dynamic-C", type=int, default=0, help="override dynamic input C; 0 reads checkpoint args")
    p.add_argument("--dynamic-out-ch", type=int, default=0, help="override dynamic transmitted subspace channels; 0 reads checkpoint args")
    p.add_argument("--dynamic-modes", type=int, default=0, help="override dynamic mode count; 0 reads checkpoint args")
    p.add_argument("--dynamic-channel-impl", type=str, default="", choices=["", "real_awgn", "cddm"])
    p.add_argument("--test-dir", type=str, default="/workspace/yongjia/datasets/DIV2K/DIV2K_valid_HR")
    p.add_argument("--channel-type", type=str, default="awgn", choices=["awgn", "rayleigh"])
    p.add_argument("--eval-snrs", type=parse_float_csv, default=(6.0,))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--mc-runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--amp-dtype", type=str, default="none", choices=["none", "bfloat16", "float16"])
    p.add_argument("--gpu", type=str, default=None, help="set CUDA_VISIBLE_DEVICES before torch import")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--csv", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    loader = make_loader(args, device)

    print("JSCC C4 vs dynamic semantic water-filling", flush=True)
    print(f"device={device} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} amp={args.amp_dtype}", flush=True)
    print(f"dataset={resolve_path(args.test_dir)} images={len(loader.dataset)} batch={args.batch_size}", flush=True)
    print(f"jscc_encoder={resolve_path(args.jscc_encoder)}", flush=True)
    print(f"jscc_decoder={resolve_path(args.jscc_decoder)}", flush=True)
    print(f"dynamic_ckpt={resolve_path(args.dynamic_ckpt)}", flush=True)
    print("", flush=True)

    rows: list[dict[str, float | str]] = []
    for snr_idx, snr in enumerate(args.eval_snrs):
        jscc_vals = []
        dyn_vals = []
        dyn_last: dict[str, float] = {}
        for run in range(int(args.mc_runs)):
            seed = int(args.seed) + 1009 * run + 100003 * snr_idx
            seed_everything(seed)
            jscc_metrics = eval_jscc_c4(
                args=args,
                loader=loader,
                device=device,
                eval_snr=float(snr),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            seed_everything(seed)
            dyn_metrics = eval_dynamic(
                args=args,
                loader=loader,
                device=device,
                eval_snr=float(snr),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            jscc_vals.append(jscc_metrics["psnr"])
            dyn_vals.append(dyn_metrics["psnr"])
            dyn_last = dyn_metrics

        jscc_mean = float(np.mean(jscc_vals))
        dyn_mean = float(np.mean(dyn_vals))
        jscc_std = float(np.std(jscc_vals, ddof=0))
        dyn_std = float(np.std(dyn_vals, ddof=0))
        gain = dyn_mean - jscc_mean
        num_modes = len([k for k in dyn_last if k.startswith("usage") and not k.startswith("usage_soft")])
        out_ch = len([k for k in dyn_last if k.startswith("p_sel_mean")])
        print(
            f"SNR={snr:g} dB | JSCC_C4={jscc_mean:.4f}"
            f"{' +/- ' + format(jscc_std, '.4f') if args.mc_runs > 1 else ''} | "
            f"dynamic_C16_K4={dyn_mean:.4f}"
            f"{' +/- ' + format(dyn_std, '.4f') if args.mc_runs > 1 else ''} | "
            f"gain={gain:+.4f}",
            flush=True,
        )
        print(
            f"  dynamic usage={format_vec(dyn_last, 'usage', num_modes)} "
            f"usage_soft={format_vec(dyn_last, 'usage_soft', num_modes)} "
            f"p_sel_mean={format_vec(dyn_last, 'p_sel_mean', out_ch)} "
            f"p_all_minmax=[{dyn_last.get('p_all_min', float('nan')):.3f},{dyn_last.get('p_all_max', float('nan')):.3f}] "
            f"aat_err={dyn_last.get('aat_err', float('nan')):.2e}",
            flush=True,
        )
        rows.append(
            {
                "eval_snr": float(snr),
                "mc_runs": int(args.mc_runs),
                "n_images": int(dyn_last.get("n", 0)),
                "jscc_c4_psnr": jscc_mean,
                "jscc_c4_psnr_std": jscc_std,
                "dynamic_c16_k4_psnr": dyn_mean,
                "dynamic_c16_k4_psnr_std": dyn_std,
                "gain_dynamic_minus_jscc": gain,
                "dynamic_aat_err": dyn_last.get("aat_err", float("nan")),
                "dynamic_p_all_min": dyn_last.get("p_all_min", float("nan")),
                "dynamic_p_all_max": dyn_last.get("p_all_max", float("nan")),
            }
        )

    if args.csv:
        csv_path = resolve_path(args.csv)
        write_csv(csv_path, rows)
        print(f"\nwrote csv: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
