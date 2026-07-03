from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

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
    meters,
    mse_per_image,
    psnr_per_image,
    resolve_path,
    seed_everything,
    ssim_per_image,
    write_json,
)
from model import build_layer1, build_layer2, layer1_forward, layer2_forward


def load_local_io():
    spec = importlib.util.spec_from_file_location("jsccf_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = load_local_io()


def save_visuals(out_dir: Path, idx: int, img: torch.Tensor, out: dict[str, torch.Tensor]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [img.cpu(), out["x1"].cpu()]
    if "u2" in out:
        rows.extend([out["u2"].cpu(), out["final"].cpu()])
        rows.extend([(img - out["x1"]).abs().cpu(), (img - out["final"]).abs().cpu()])
    else:
        rows.append((img - out["x1"]).abs().cpu())
    save_image(torch.cat(rows, dim=0), out_dir / f"sample_{idx:04d}.png", nrow=img.shape[0])


@torch.no_grad()
def eval_layer1(args) -> dict[str, float]:
    cfg = jsccf_io.build_config(args, encoder_in_chans=3)
    _train_loader, val_loader = get_loader(cfg)
    e1, d1 = build_layer1(args, cfg.device)
    ckpt = jsccf_io.load_checkpoint(args.layer1_ckpt)
    jsccf_io.load_state(e1, ckpt["e1_state_dict"], "E1", strict=True)
    jsccf_io.load_state(d1, ckpt["d1_state_dict"], "D1", strict=True)
    e1.eval()
    d1.eval()
    m = meters(["mse_x1", "psnr_x1", "ssim_x1"])
    rows = []
    for idx, (imgs, _labels) in enumerate(val_loader):
        imgs = imgs.to(cfg.device, non_blocking=True)
        out = layer1_forward(e1, d1, imgs)
        bsz = imgs.shape[0]
        metrics = {
            "mse_x1": batch_metric_mean(mse_per_image(out["x1"], imgs)),
            "psnr_x1": batch_metric_mean(psnr_per_image(out["x1"], imgs)),
            "ssim_x1": batch_metric_mean(ssim_per_image(out["x1"], imgs)),
        }
        for k, v in metrics.items():
            m[k].update(v, bsz)
        rows.append({"index": idx, **metrics})
        if idx < int(args.save_visuals):
            save_visuals(Path(resolve_path(args.out_dir)) / "visuals", idx, imgs, out)
    write_rows(args, rows)
    return averaged(m)


@torch.no_grad()
def eval_layer2(args) -> dict[str, float]:
    ckpt = jsccf_io.load_checkpoint(args.layer2_ckpt)
    ckpt_args = argparse.Namespace(**ckpt.get("args", {}))
    variant = str(getattr(args, "variant", "") or ckpt.get("variant", getattr(ckpt_args, "variant", "combiner")))
    args.variant = variant
    cfg = jsccf_io.build_config(args, encoder_in_chans=3)
    _train_loader, val_loader = get_loader(cfg)
    e1, d1, e2, d2, combiner = build_layer2(args, cfg.device)
    jsccf_io.load_state(e1, ckpt["e1_state_dict"], "E1", strict=True)
    jsccf_io.load_state(d1, ckpt["d1_state_dict"], "D1", strict=True)
    jsccf_io.load_state(e2, ckpt["e2_state_dict"], "E2", strict=True)
    jsccf_io.load_state(d2, ckpt["d2_state_dict"], "D2", strict=True)
    jsccf_io.load_state(combiner, ckpt["combiner_state_dict"], "combiner", strict=True)
    e1.eval()
    d1.eval()
    e2.eval()
    d2.eval()
    combiner.eval()
    names = ["mse_x1", "psnr_x1", "ssim_x1", "mse_u2", "psnr_u2", "ssim_u2", "mse_final", "psnr_final", "ssim_final", "delta_psnr"]
    m = meters(names)
    rows = []
    for idx, (imgs, _labels) in enumerate(val_loader):
        imgs = imgs.to(cfg.device, non_blocking=True)
        out = layer2_forward(e1, d1, e2, d2, combiner, imgs, args.variant)
        psnr_x1 = batch_metric_mean(psnr_per_image(out["x1"], imgs))
        psnr_final = batch_metric_mean(psnr_per_image(out["final"], imgs))
        metrics = {
            "mse_x1": batch_metric_mean(mse_per_image(out["x1"], imgs)),
            "psnr_x1": psnr_x1,
            "ssim_x1": batch_metric_mean(ssim_per_image(out["x1"], imgs)),
            "mse_u2": batch_metric_mean(mse_per_image(out["u2"], imgs)),
            "psnr_u2": batch_metric_mean(psnr_per_image(out["u2"], imgs)),
            "ssim_u2": batch_metric_mean(ssim_per_image(out["u2"], imgs)),
            "mse_final": batch_metric_mean(mse_per_image(out["final"], imgs)),
            "psnr_final": psnr_final,
            "ssim_final": batch_metric_mean(ssim_per_image(out["final"], imgs)),
            "delta_psnr": psnr_final - psnr_x1,
        }
        for k, v in metrics.items():
            m[k].update(v, imgs.shape[0])
        rows.append({"index": idx, **metrics})
        if idx < int(args.save_visuals):
            save_visuals(Path(resolve_path(args.out_dir)) / "visuals", idx, imgs, out)
    write_rows(args, rows)
    return averaged(m)


def write_rows(args, rows: list[dict]) -> None:
    out_dir = Path(resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(out_dir / "per_batch_metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", type=str, default="no-c1")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=jsccf_io.default_save_dir())
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default=jsccf_io.default_init_ckpt())
    p.add_argument("--latent-ch", type=int, default=16)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--latest-every", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--layer1-ckpt", type=str, default=str(Path(jsccf_io.default_save_dir()) / "jscc_f_layer1_best.pth"))
    p.add_argument("--layer2-ckpt", type=str, default="")
    p.add_argument("--variant", type=str, default="", choices=["", "combiner", "no_combiner", "residual_input"])
    p.add_argument("--out-dir", type=str, default=str(Path(jsccf_io.default_save_dir()) / "eval"))
    p.add_argument("--save-visuals", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    if args.layer2_ckpt:
        summary = eval_layer2(args)
    else:
        summary = eval_layer1(args)
    write_json(Path(resolve_path(args.out_dir)) / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
