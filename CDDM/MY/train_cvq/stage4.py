from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from Autoencoder.data.datasets import get_loader

try:
    from pytorch_msssim import ms_ssim as _ms_ssim
except Exception:  # pragma: no cover - optional metric
    _ms_ssim = None

try:
    import lpips as _lpips
except Exception:  # pragma: no cover - optional metric
    _lpips = None

from .common import AverageMeter, batch_metric_mean, format_metrics, psnr_per_image, real_awgn, resolve_path, seed_everything, write_json, print_run_header
from .io import build_config, build_models, ckpt_path, forward_parts, load_experiment_checkpoint

@torch.no_grad()
def oracle_eval(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = build_config(args)
    _train_loader, val_loader = get_loader(cfg)
    encoder, decoder, cvq, _car = build_models(args, cfg.device)
    src = args.init_ckpt or ckpt_path(args, "stage3", "best")
    load_experiment_checkpoint(src, encoder=encoder, decoder=decoder, cvq=cvq, strict=True)
    encoder.eval()
    decoder.eval()
    cvq.eval()
    lpips_metric = _lpips.LPIPS(net="alex").to(cfg.device).eval() if _lpips is not None and bool(args.with_lpips) else None
    rows = []
    code_stats_a = []
    code_stats_b = []
    print_run_header(args, "Stage 4 | oracle evaluation", 0, len(val_loader.dataset))
    meters = {
        name: {k: AverageMeter() for k in ["psnr", "ms_ssim", "lpips"]}
        for name in ["prefix_only", "continuous_tail_oracle", "codebook_tail_oracle", "full_reference"]
    }
    for imgs, _labels in val_loader:
        imgs = imgs.to(cfg.device, non_blocking=True)
        _z, s, y_prefix, tail = forward_parts(imgs, encoder, args)
        tail_q, idx, aux = cvq.encode(tail)
        zero = torch.zeros_like(tail)
        outputs = {
            "prefix_only": decoder(torch.cat([y_prefix, zero], dim=1)).clamp(0.0, 1.0),
            "continuous_tail_oracle": decoder(torch.cat([y_prefix, tail], dim=1)).clamp(0.0, 1.0),
            "codebook_tail_oracle": decoder(torch.cat([y_prefix, tail_q], dim=1)).clamp(0.0, 1.0),
            "full_reference": decoder(real_awgn(s, float(args.snr_db))).clamp(0.0, 1.0),
        }
        code_stats_a.append((aux["idx_a"].detach().cpu(), tail_q[:, : cvq.split_a].detach().cpu(), tail[:, : cvq.split_a].detach().cpu()))
        code_stats_b.append((aux["idx_b"].detach().cpu(), tail_q[:, cvq.split_a :].detach().cpu(), tail[:, cvq.split_a :].detach().cpu()))
        for name, out in outputs.items():
            bsz = imgs.shape[0]
            meters[name]["psnr"].update(batch_metric_mean(psnr_per_image(out, imgs)), bsz)
            if _ms_ssim is not None:
                val = batch_metric_mean(_ms_ssim(out.float(), imgs.float(), data_range=1.0, size_average=False))
                meters[name]["ms_ssim"].update(val, bsz)
            if lpips_metric is not None:
                val = batch_metric_mean(lpips_metric(out * 2.0 - 1.0, imgs * 2.0 - 1.0).reshape(-1))
                meters[name]["lpips"].update(val, bsz)
    for name, metric_meters in meters.items():
        row = {
            "method": name,
            "front": "AWGN all" if name == "full_reference" else "AWGN",
            "tail": {
                "prefix_only": "zero",
                "continuous_tail_oracle": "clean continuous",
                "codebook_tail_oracle": "CVQ quantized",
                "full_reference": "AWGN all",
            }[name],
            "psnr": metric_meters["psnr"].avg,
            "ms_ssim": metric_meters["ms_ssim"].avg if metric_meters["ms_ssim"].count else float("nan"),
            "lpips": metric_meters["lpips"].avg if metric_meters["lpips"].count else float("nan"),
        }
        rows.append(row)
    idx_a = torch.cat([x[0] for x in code_stats_a], dim=0)
    qa = torch.cat([x[1] for x in code_stats_a], dim=0)
    ta = torch.cat([x[2] for x in code_stats_a], dim=0)
    idx_b = torch.cat([x[0] for x in code_stats_b], dim=0)
    qb = torch.cat([x[1] for x in code_stats_b], dim=0)
    tb = torch.cat([x[2] for x in code_stats_b], dim=0)
    stats = {
        f"C_A_s{int(args.prefix_ch) + 1}_{int(args.prefix_ch) + cvq.split_a}": cvq.cvq_a.stats(idx_a, qa, ta),
        f"C_B_s{int(args.prefix_ch) + cvq.split_a + 1}_{int(args.latent_ch)}": cvq.cvq_b.stats(idx_b, qb, tb),
    }
    psnr = {row["method"]: row["psnr"] for row in rows}
    diagnostics = {
        "delta_tail": psnr["continuous_tail_oracle"] - psnr["prefix_only"],
        "delta_vq": psnr["codebook_tail_oracle"] - psnr["prefix_only"],
        "gap_vq": psnr["continuous_tail_oracle"] - psnr["codebook_tail_oracle"],
    }
    out_dir = Path(resolve_path(args.save_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"stage4_oracle_c{int(args.latent_ch)}_snr{args.snr_db:g}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "front", "tail", "psnr", "ms_ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / f"stage4_oracle_c{int(args.latent_ch)}_snr{args.snr_db:g}.json", {"rows": rows, "codebook": stats, "diagnostics": diagnostics})
    print(f"[stage4] wrote {csv_path}")
    print(f"[stage4] diagnostics {format_metrics(diagnostics)}")
    print(f"[stage4] codebook {json.dumps(stats, indent=2, sort_keys=True)}")
