#!/usr/bin/env python3
"""Validate whether C36 tail channels are ordered from coarse to fine."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

CDDM_ROOT = Path(__file__).resolve().parents[2]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))

from Autoencoder.data.datasets import get_loader  # noqa: E402
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder  # noqa: E402
from MY.train_cvq.common import (  # noqa: E402
    AverageMeter,
    CDDMJSCCConfig,
    prefix_power_normalize,
    psnr_per_image,
    real_awgn,
    resolve_path,
    seed_everything,
    write_json,
)


def default_save_dir() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints-cvq-v2-v01-c36-snr9-k4096")


def default_checkpoint() -> str:
    save_dir = Path(default_save_dir())
    stage3 = save_dir / "cvq_v2_v01_c36_snr9_k4096_stage3_best.pth"
    if stage3.exists():
        return str(stage3)
    return str(save_dir / "cvq_v2_v01_c36_snr9_k4096_stage1_best.pth")


def default_out_dir() -> str:
    return str(CDDM_ROOT / "MY" / "eval_cvq" / "results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=default_checkpoint())
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=default_save_dir())
    p.add_argument("--out-dir", type=str, default=default_out_dir())
    p.add_argument("--snr-db", type=float, default=9.0)
    p.add_argument("--latent-ch", type=int, default=36)
    p.add_argument("--prefix-ch", type=int, default=16)
    p.add_argument("--c1-ch", type=int, default=None, help="Alias for --prefix-ch; inferred from checkpoint when present.")
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--max-batches", type=int, default=0, help="0 evaluates the full validation split")
    p.add_argument("--freq-low", type=float, default=0.15)
    p.add_argument("--freq-high", type=float, default=0.35)
    p.add_argument("--random-trials", type=int, default=5)
    p.add_argument("--random-ms", type=str, default="1,2,4,8,10,20")
    return p.parse_args()


def build_config(args: argparse.Namespace, batch_size: int | None = None) -> CDDMJSCCConfig:
    return CDDMJSCCConfig(
        C=int(args.latent_ch),
        SNRs=float(args.snr_db),
        channel_type="awgn",
        batch_size=int(args.batch_size if batch_size is None else batch_size),
        test_batch=int(args.test_batch),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.val_num_workers),
        train_data_dir=resolve_path(Path(args.data_dir) / "DIV2K_train_HR"),
        test_data_dir=resolve_path(Path(args.data_dir) / "DIV2K_valid_HR"),
        CUDA=(not bool(args.cpu)),
    )


def validate_eval_args(args: argparse.Namespace, checkpoint: dict | None = None) -> None:
    if args.c1_ch is None:
        args.c1_ch = int(args.prefix_ch)
    else:
        args.prefix_ch = int(args.c1_ch)
    if checkpoint is not None:
        if "latent_ch" in checkpoint and int(checkpoint["latent_ch"]) != int(args.latent_ch):
            raise ValueError(f"checkpoint latent_ch={checkpoint['latent_ch']} but --latent-ch={args.latent_ch}")
        if "c1_ch" in checkpoint and int(checkpoint["c1_ch"]) != int(args.prefix_ch):
            raise ValueError(f"checkpoint c1_ch={checkpoint['c1_ch']} but --prefix-ch/--c1-ch={args.prefix_ch}")
    if int(args.latent_ch) != 36:
        raise ValueError("coarse_to_fine.py currently expects C=36; use --latent-ch 36.")
    if int(args.prefix_ch) != 16:
        raise ValueError("coarse_to_fine.py currently expects C1/prefix=16; use --prefix-ch 16.")
    if int(args.latent_ch) - int(args.prefix_ch) != 20:
        raise ValueError("coarse_to_fine.py expects a 20-channel C2 tail.")
    if int(args.latent_h) != 16 or int(args.latent_w) != 16:
        raise ValueError("The C36 JSCC encoder is expected to produce 16x16 latents for 256x256 crops.")


def build_encoder_decoder(args: argparse.Namespace, device: torch.device):
    encoder = JSCC_encoder(build_config(args), int(args.latent_ch)).to(device)
    decoder = JSCC_decoder(build_config(args), int(args.latent_ch)).to(device)
    return encoder, decoder


def load_state(module, state: dict[str, torch.Tensor], name: str, strict: bool = True) -> None:
    missing, unexpected = module.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"{name} load mismatch: missing={missing}, unexpected={unexpected}")
    print(f"loaded {name}: missing={missing} unexpected={unexpected}")


def load_eval_checkpoint(path: str, encoder, decoder) -> dict:
    abs_path = resolve_path(path)
    checkpoint = torch.load(abs_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"expected checkpoint dict, got {type(checkpoint)}")
    if "encoder_state_dict" not in checkpoint or "decoder_state_dict" not in checkpoint:
        raise KeyError("checkpoint must contain encoder_state_dict and decoder_state_dict")
    load_state(encoder, checkpoint["encoder_state_dict"], "encoder", strict=True)
    load_state(decoder, checkpoint["decoder_state_dict"], "decoder", strict=True)
    print(f"loaded checkpoint: {abs_path}")
    return checkpoint


def parse_m_list(text: str, tail_ch: int) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        m = int(part)
        if m < 0 or m > tail_ch:
            raise ValueError(f"random m must be in [0,{tail_ch}], got {m}")
        values.append(m)
    return sorted(set(values))


def make_eval_dir(args: argparse.Namespace) -> Path:
    ckpt_stem = Path(resolve_path(args.checkpoint)).stem
    out_dir = Path(resolve_path(args.out_dir)) / f"{ckpt_stem}_coarse_to_fine_snr{float(args.snr_db):g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def limit_reached(batch_idx: int, args: argparse.Namespace) -> bool:
    return int(args.max_batches) > 0 and batch_idx >= int(args.max_batches)


def reconstruct_prefix(decoder, y_prefix: torch.Tensor, tail: torch.Tensor, m: int) -> torch.Tensor:
    tail_m = torch.zeros_like(tail)
    if m > 0:
        tail_m[:, :m] = tail[:, :m]
    return decoder(torch.cat([y_prefix, tail_m], dim=1)).clamp(0.0, 1.0)


def reconstruct_with_tail(decoder, y_prefix: torch.Tensor, tail_m: torch.Tensor) -> torch.Tensor:
    return decoder(torch.cat([y_prefix, tail_m], dim=1)).clamp(0.0, 1.0)


def radial_masks(h: int, w: int, device: torch.device, low: float, high: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )
    rr = torch.sqrt(xx.square() + yy.square())
    rr = rr / rr.max().clamp_min(1e-8)
    return rr, rr <= float(low), (rr > float(low)) & (rr <= float(high)), rr > float(high)


def frequency_stats(delta: torch.Tensor, low: float, high: float, eps: float = 1e-12) -> dict[str, float]:
    x = delta.float()
    fft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    power = (fft.real.square() + fft.imag.square()).mean(dim=1)
    _rr, mask_low, mask_mid, mask_high = radial_masks(power.shape[-2], power.shape[-1], power.device, low, high)
    rr = _rr.reshape(1, -1)
    flat = power.flatten(1)
    total = flat.sum(dim=1).clamp_min(eps)
    low_e = power[:, mask_low].sum(dim=1) / total
    mid_e = power[:, mask_mid].sum(dim=1) / total
    high_e = power[:, mask_high].sum(dim=1) / total
    centroid = (flat * rr).sum(dim=1) / total
    l2 = x.square().mean(dim=(1, 2, 3)).sqrt()
    return {
        "delta_l2": float(l2.mean().item()),
        "low_energy": float(low_e.mean().item()),
        "mid_energy": float(mid_e.mean().item()),
        "high_energy": float(high_e.mean().item()),
        "high_ratio": float(high_e.mean().item()),
        "centroid": float(centroid.mean().item()),
    }


@torch.no_grad()
def eval_tail_prefix_curve(loader, encoder, decoder, args: argparse.Namespace) -> list[dict]:
    encoder.eval()
    decoder.eval()
    tail_ch = int(args.latent_ch) - int(args.prefix_ch)
    device = next(encoder.parameters()).device
    meters = {m: AverageMeter() for m in range(tail_ch + 1)}

    for batch_idx, (imgs, _labels) in enumerate(loader):
        if limit_reached(batch_idx, args):
            break
        imgs = imgs.to(device, non_blocking=True)
        _z, _s, y_prefix, tail = forward_parts_for_eval(imgs, encoder, args)
        for m in range(tail_ch + 1):
            out = reconstruct_prefix(decoder, y_prefix, tail, m)
            meters[m].update(float(psnr_per_image(out, imgs).sum().item()), imgs.shape[0])

    rows = []
    prev = None
    for m in range(tail_ch + 1):
        psnr = meters[m].avg
        gain = float("nan") if prev is None else psnr - prev
        rows.append(
            {
                "m": m,
                "channel_included": "none" if m == 0 else f"{int(args.prefix_ch) + 1}-{int(args.prefix_ch) + m}",
                "psnr": psnr,
                "delta_psnr": gain,
            }
        )
        prev = psnr
    return rows


@torch.no_grad()
def eval_tail_knockout(loader, encoder, decoder, args: argparse.Namespace) -> list[dict]:
    encoder.eval()
    decoder.eval()
    tail_ch = int(args.latent_ch) - int(args.prefix_ch)
    device = next(encoder.parameters()).device
    full_meter = AverageMeter()
    drop_meters = {i: AverageMeter() for i in range(tail_ch)}

    for batch_idx, (imgs, _labels) in enumerate(loader):
        if limit_reached(batch_idx, args):
            break
        imgs = imgs.to(device, non_blocking=True)
        _z, _s, y_prefix, tail = forward_parts_for_eval(imgs, encoder, args)
        x_full = reconstruct_with_tail(decoder, y_prefix, tail)
        full_meter.update(float(psnr_per_image(x_full, imgs).sum().item()), imgs.shape[0])
        for i in range(tail_ch):
            tail_drop = tail.clone()
            tail_drop[:, i] = 0.0
            out = reconstruct_with_tail(decoder, y_prefix, tail_drop)
            drop_meters[i].update(float(psnr_per_image(out, imgs).sum().item()), imgs.shape[0])

    full_psnr = full_meter.avg
    rows = []
    for i in range(tail_ch):
        psnr_drop = drop_meters[i].avg
        rows.append(
            {
                "tail_index": i + 1,
                "channel": int(args.prefix_ch) + i + 1,
                "psnr_full": full_psnr,
                "psnr_drop": psnr_drop,
                "damage_psnr": full_psnr - psnr_drop,
            }
        )
    return rows


@torch.no_grad()
def eval_tail_frequency_curve(loader, encoder, decoder, args: argparse.Namespace) -> list[dict]:
    encoder.eval()
    decoder.eval()
    tail_ch = int(args.latent_ch) - int(args.prefix_ch)
    device = next(encoder.parameters()).device
    meter_keys = ["delta_l2", "low_energy", "mid_energy", "high_energy", "high_ratio", "centroid"]
    meters = {m: {k: AverageMeter() for k in meter_keys} for m in range(1, tail_ch + 1)}

    for batch_idx, (imgs, _labels) in enumerate(loader):
        if limit_reached(batch_idx, args):
            break
        imgs = imgs.to(device, non_blocking=True)
        _z, _s, y_prefix, tail = forward_parts_for_eval(imgs, encoder, args)
        prev = reconstruct_prefix(decoder, y_prefix, tail, 0)
        for m in range(1, tail_ch + 1):
            out = reconstruct_prefix(decoder, y_prefix, tail, m)
            stats = frequency_stats(out - prev, float(args.freq_low), float(args.freq_high))
            for key, value in stats.items():
                meters[m][key].update(value, imgs.shape[0])
            prev = out

    rows = []
    for m in range(1, tail_ch + 1):
        row = {
            "m": m,
            "channel": int(args.prefix_ch) + m,
        }
        row.update({key: meters[m][key].avg for key in meter_keys})
        rows.append(row)
    return rows


@torch.no_grad()
def eval_random_order(loader, encoder, decoder, args: argparse.Namespace) -> list[dict]:
    encoder.eval()
    decoder.eval()
    tail_ch = int(args.latent_ch) - int(args.prefix_ch)
    m_list = parse_m_list(str(args.random_ms), tail_ch)
    device = next(encoder.parameters()).device
    ordered = {m: AverageMeter() for m in m_list}
    random_subset = {m: AverageMeter() for m in m_list}
    random_permuted_slots = {m: AverageMeter() for m in m_list}

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 991)

    for batch_idx, (imgs, _labels) in enumerate(loader):
        if limit_reached(batch_idx, args):
            break
        imgs = imgs.to(device, non_blocking=True)
        _z, _s, y_prefix, tail = forward_parts_for_eval(imgs, encoder, args)
        for m in m_list:
            ordered_out = reconstruct_prefix(decoder, y_prefix, tail, m)
            ordered[m].update(float(psnr_per_image(ordered_out, imgs).sum().item()), imgs.shape[0])

        for _trial in range(int(args.random_trials)):
            perm = torch.randperm(tail_ch, device=device, generator=gen)
            for m in m_list:
                tail_subset = torch.zeros_like(tail)
                tail_slots = torch.zeros_like(tail)
                if m > 0:
                    tail_subset[:, perm[:m]] = tail[:, perm[:m]]
                    tail_slots[:, :m] = tail[:, perm[:m]]
                subset_out = reconstruct_with_tail(decoder, y_prefix, tail_subset)
                slots_out = reconstruct_with_tail(decoder, y_prefix, tail_slots)
                random_subset[m].update(float(psnr_per_image(subset_out, imgs).sum().item()), imgs.shape[0])
                random_permuted_slots[m].update(float(psnr_per_image(slots_out, imgs).sum().item()), imgs.shape[0])

    rows = []
    for m in m_list:
        psnr_ordered = ordered[m].avg
        psnr_subset = random_subset[m].avg
        psnr_slots = random_permuted_slots[m].avg
        rows.append(
            {
                "m": m,
                "ordered_psnr": psnr_ordered,
                "random_subset_psnr": psnr_subset,
                "random_permuted_slots_psnr": psnr_slots,
                "ordered_minus_random_subset": psnr_ordered - psnr_subset,
                "ordered_minus_random_permuted_slots": psnr_ordered - psnr_slots,
                "random_trials": int(args.random_trials),
            }
        )
    return rows


def forward_parts_for_eval(imgs: torch.Tensor, encoder, args: argparse.Namespace):
    z, _ = encoder(imgs)
    if z.shape[1] != int(args.latent_ch):
        raise RuntimeError(f"expected latent channels={args.latent_ch}, got {z.shape[1]}")
    z_norm, _power = prefix_power_normalize(z.float(), prefix_ch=int(args.prefix_ch))
    y_prefix = real_awgn(z_norm[:, : int(args.prefix_ch)], float(args.snr_db))
    tail = z_norm[:, int(args.prefix_ch) :]
    return z, z_norm, y_prefix, tail


def finite_mean(values: list[float]) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return sum(xs) / max(1, len(xs))


def summarize(
    args: argparse.Namespace,
    prefix_rows: list[dict],
    knockout_rows: list[dict],
    freq_rows: list[dict],
    random_rows: list[dict],
) -> dict:
    tail_ch = int(args.latent_ch) - int(args.prefix_ch)
    half = tail_ch // 2
    psnr0 = float(prefix_rows[0]["psnr"])
    psnr_half = float(prefix_rows[half]["psnr"])
    psnr_full = float(prefix_rows[-1]["psnr"])
    gains = [float(r["delta_psnr"]) for r in prefix_rows[1:]]
    early_gains = gains[:half]
    late_gains = gains[half:]
    early_damage = [float(r["damage_psnr"]) for r in knockout_rows[:half]]
    late_damage = [float(r["damage_psnr"]) for r in knockout_rows[half:]]
    early_hf = [float(r["high_ratio"]) for r in freq_rows[:half]]
    late_hf = [float(r["high_ratio"]) for r in freq_rows[half:]]
    random_subset_gap = [float(r["ordered_minus_random_subset"]) for r in random_rows]
    random_slots_gap = [float(r["ordered_minus_random_permuted_slots"]) for r in random_rows]
    monotonic_steps = sum(1 for g in gains if g >= 0.0)
    return {
        "checkpoint": resolve_path(args.checkpoint),
        "snr_db": float(args.snr_db),
        "latent_ch": int(args.latent_ch),
        "prefix_ch": int(args.prefix_ch),
        "tail_ch": tail_ch,
        "max_batches": int(args.max_batches),
        "psnr_m0": psnr0,
        "psnr_m_half": psnr_half,
        "psnr_m_full": psnr_full,
        "tail_gain_total": psnr_full - psnr0,
        "tail_gain_first_half": psnr_half - psnr0,
        "tail_gain_second_half": psnr_full - psnr_half,
        "monotonic_nonnegative_steps": monotonic_steps,
        "monotonic_fraction": monotonic_steps / float(tail_ch),
        "early_delta_psnr_mean": finite_mean(early_gains),
        "late_delta_psnr_mean": finite_mean(late_gains),
        "early_damage_mean": finite_mean(early_damage),
        "late_damage_mean": finite_mean(late_damage),
        "early_high_ratio_mean": finite_mean(early_hf),
        "late_high_ratio_mean": finite_mean(late_hf),
        "ordered_minus_random_subset_mean": finite_mean(random_subset_gap),
        "ordered_minus_random_permuted_slots_mean": finite_mean(random_slots_gap),
    }


def write_markdown(path: Path, summary: dict, csv_names: list[str]) -> None:
    lines = [
        "# C36 Tail Coarse-to-Fine Validation",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- SNR: `{summary['snr_db']:g}` dB",
        f"- tail gain total: `{summary['tail_gain_total']:.4f}` dB",
        f"- first half gain: `{summary['tail_gain_first_half']:.4f}` dB",
        f"- second half gain: `{summary['tail_gain_second_half']:.4f}` dB",
        f"- monotonic nonnegative steps: `{summary['monotonic_nonnegative_steps']}/{summary['tail_ch']}`",
        f"- early vs late HF ratio: `{summary['early_high_ratio_mean']:.6f}` vs `{summary['late_high_ratio_mean']:.6f}`",
        f"- ordered minus random subset mean: `{summary['ordered_minus_random_subset_mean']:.4f}` dB",
        f"- ordered minus random permuted slots mean: `{summary['ordered_minus_random_permuted_slots_mean']:.4f}` dB",
        "",
        "## Outputs",
    ]
    lines.extend(f"- `{name}`" for name in csv_names)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_eval_args(args)
    seed_everything(int(args.seed))
    out_dir = make_eval_dir(args)
    cfg = build_config(args)
    _train_loader, val_loader = get_loader(cfg)
    encoder, decoder = build_encoder_decoder(args, cfg.device)
    checkpoint = load_eval_checkpoint(resolve_path(args.checkpoint), encoder, decoder)
    validate_eval_args(args, checkpoint)

    print("=== C36 tail coarse-to-fine evaluation ===")
    print(f"checkpoint={resolve_path(args.checkpoint)}")
    print(f"checkpoint_stage={checkpoint.get('stage', '<unknown>')} route={checkpoint.get('route', '<unknown>')}")
    print(f"out_dir={out_dir}")
    print(f"device={cfg.device} val={len(val_loader.dataset)} max_batches={args.max_batches}")
    t0 = time.time()

    prefix_rows = eval_tail_prefix_curve(val_loader, encoder, decoder, args)
    prefix_csv = out_dir / "tail_prefix_curve.csv"
    write_csv(prefix_csv, prefix_rows, ["m", "channel_included", "psnr", "delta_psnr"])
    print(f"wrote {prefix_csv}")

    knockout_rows = eval_tail_knockout(val_loader, encoder, decoder, args)
    knockout_csv = out_dir / "tail_knockout.csv"
    write_csv(knockout_csv, knockout_rows, ["tail_index", "channel", "psnr_full", "psnr_drop", "damage_psnr"])
    print(f"wrote {knockout_csv}")

    freq_rows = eval_tail_frequency_curve(val_loader, encoder, decoder, args)
    freq_csv = out_dir / "tail_frequency_curve.csv"
    write_csv(freq_csv, freq_rows, ["m", "channel", "delta_l2", "low_energy", "mid_energy", "high_energy", "high_ratio", "centroid"])
    print(f"wrote {freq_csv}")

    random_rows = eval_random_order(val_loader, encoder, decoder, args)
    random_csv = out_dir / "tail_random_order.csv"
    write_csv(
        random_csv,
        random_rows,
        [
            "m",
            "ordered_psnr",
            "random_subset_psnr",
            "random_permuted_slots_psnr",
            "ordered_minus_random_subset",
            "ordered_minus_random_permuted_slots",
            "random_trials",
        ],
    )
    print(f"wrote {random_csv}")

    summary = summarize(args, prefix_rows, knockout_rows, freq_rows, random_rows)
    summary["elapsed_sec"] = time.time() - t0
    write_json(out_dir / "summary.json", summary)
    write_markdown(out_dir / "summary.md", summary, [prefix_csv.name, knockout_csv.name, freq_csv.name, random_csv.name, "summary.json"])
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"summary tail_gain={summary['tail_gain_total']:.4f} monotonic={summary['monotonic_nonnegative_steps']}/{summary['tail_ch']} elapsed={summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
