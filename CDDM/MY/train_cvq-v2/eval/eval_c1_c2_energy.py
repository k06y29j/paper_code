# 作用：统计 Stage1 checkpoint 的 C1/C2 latent 能量，包括 power-normalized latent 和 encoder raw latent 两种口径。
# 输出：summary.json、per_image_energy.csv、per_channel_energy.csv。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/eval_c1_c2_energy.py --max-images 8 --val-num-workers 0

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms


THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR.parent
CDDM_ROOT = PACKAGE_DIR.parents[1]
for path in (PACKAGE_DIR, CDDM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Autoencoder.data.datasets import FlatImageFolder  # noqa: E402
from common import resolve_path, seed_everything, split_c1_c2  # noqa: E402


DEFAULT_CKPT = "MY/jscc-no-awgn/cvq_v2_c36_stage0_v2_best_c1distill-c16.pth"
DEFAULT_OUT_DIR = "MY/train_cvq-v2/eval/outputs/cvq_v2_c36_snr12_stage1_best_valid_c1_c2_energy-v2"


def load_cvq_io():
    spec = importlib.util.spec_from_file_location("cvq_v2_io", PACKAGE_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cvq_io = load_cvq_io()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def make_args(cli: argparse.Namespace, ckpt: dict) -> argparse.Namespace:
    saved_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    args = argparse.Namespace(**saved_args)
    args.data_dir = str(cli.data_dir)
    args.save_dir = str(Path(resolve_path(cli.ckpt)).parent)
    args.log_file = ""
    args.init_ckpt = str(Path(resolve_path(cli.ckpt)))
    args.snr_db = float(ckpt.get("snr_db", getattr(args, "snr_db", 12.0)))
    args.latent_ch = int(ckpt.get("latent_ch", getattr(args, "latent_ch", 36)))
    args.c1_ch = int(ckpt.get("c1_ch", getattr(args, "c1_ch", 16)))
    args.latent_h = int(getattr(args, "latent_h", 16))
    args.latent_w = int(getattr(args, "latent_w", 16))
    args.k = int(ckpt.get("k", getattr(args, "k", 16384)))
    args.batch_size = int(cli.batch_size)
    args.test_batch = int(cli.test_batch)
    args.num_workers = int(cli.num_workers)
    args.val_num_workers = int(cli.val_num_workers)
    args.vq_beta = float(getattr(args, "vq_beta", 0.25))
    args.vq_chunk_size = int(getattr(args, "vq_chunk_size", 128))
    args.seed = int(cli.seed)
    args.cpu = bool(cli.cpu)
    args.stage = int(ckpt.get("stage", saved_args.get("stage", 1)) == "stage1")
    return args


def make_valid_loader(args: argparse.Namespace, crop: str):
    cfg = cvq_io.build_config(args)
    crop_hw = (int(cfg.image_dims[1]), int(cfg.image_dims[2]))
    if str(crop) == "random":
        transform = transforms.Compose([transforms.RandomCrop(crop_hw), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.CenterCrop(crop_hw), transforms.ToTensor()])
    dataset = FlatImageFolder(root=cfg.test_data_dir, transform=transform)

    def worker_init_fn(worker_id: int) -> None:
        seed = int(args.seed) + int(worker_id)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=int(args.test_batch),
        shuffle=False,
        num_workers=int(args.val_num_workers),
        pin_memory=(not bool(args.cpu)),
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )


def load_encoder(args: argparse.Namespace, ckpt: dict, device: torch.device):
    encoder, _decoder, _quantizer = cvq_io.build_models(args, device)
    cvq_io.load_state(encoder, ckpt["encoder_state_dict"], "encoder", strict=True)
    encoder.eval()
    return encoder


def _stats(values: list[float], prefix: str) -> dict[str, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return {
        f"{prefix}_mean": float(t.mean().item()),
        f"{prefix}_std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        f"{prefix}_min": float(t.min().item()),
        f"{prefix}_p25": float(torch.quantile(t, 0.25).item()),
        f"{prefix}_median": float(torch.quantile(t, 0.50).item()),
        f"{prefix}_p75": float(torch.quantile(t, 0.75).item()),
        f"{prefix}_max": float(t.max().item()),
    }


@torch.no_grad()
def evaluate_energy(loader, encoder, args: argparse.Namespace, max_images: int) -> tuple[dict, list[dict], list[dict]]:
    device = next(encoder.parameters()).device
    total = {
        "norm_c1_sq_sum": 0.0,
        "norm_c2_sq_sum": 0.0,
        "raw_c1_sq_sum": 0.0,
        "raw_c2_sq_sum": 0.0,
        "norm_c1_numel": 0,
        "norm_c2_numel": 0,
        "raw_c1_numel": 0,
        "raw_c2_numel": 0,
    }
    per_image_rows: list[dict] = []
    norm_c1_ch_sum = torch.zeros(int(args.c1_ch), dtype=torch.float64)
    norm_c2_ch_sum = torch.zeros(int(args.latent_ch) - int(args.c1_ch), dtype=torch.float64)
    raw_c1_ch_sum = torch.zeros_like(norm_c1_ch_sum)
    raw_c2_ch_sum = torch.zeros_like(norm_c2_ch_sum)
    channel_spatial_count = 0
    seen = 0

    for imgs, _labels in loader:
        if int(max_images) > 0 and seen >= int(max_images):
            break
        if int(max_images) > 0 and seen + int(imgs.shape[0]) > int(max_images):
            imgs = imgs[: int(max_images) - seen]
        imgs = imgs.to(device, non_blocking=True)
        z_raw, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
        raw_c1, raw_c2 = split_c1_c2(z_raw.float(), args)
        norm_c1, norm_c2 = split_c1_c2(z_norm.float(), args)

        for name, c1, c2 in (("norm", norm_c1, norm_c2), ("raw", raw_c1, raw_c2)):
            total[f"{name}_c1_sq_sum"] += float(c1.square().sum().item())
            total[f"{name}_c2_sq_sum"] += float(c2.square().sum().item())
            total[f"{name}_c1_numel"] += int(c1.numel())
            total[f"{name}_c2_numel"] += int(c2.numel())

        norm_c1_ch_sum += norm_c1.square().sum(dim=(0, 2, 3)).detach().cpu().double()
        norm_c2_ch_sum += norm_c2.square().sum(dim=(0, 2, 3)).detach().cpu().double()
        raw_c1_ch_sum += raw_c1.square().sum(dim=(0, 2, 3)).detach().cpu().double()
        raw_c2_ch_sum += raw_c2.square().sum(dim=(0, 2, 3)).detach().cpu().double()
        channel_spatial_count += int(norm_c1.shape[0] * norm_c1.shape[2] * norm_c1.shape[3])

        norm_c1_img = norm_c1.square().mean(dim=(1, 2, 3)).detach().cpu()
        norm_c2_img = norm_c2.square().mean(dim=(1, 2, 3)).detach().cpu()
        raw_c1_img = raw_c1.square().mean(dim=(1, 2, 3)).detach().cpu()
        raw_c2_img = raw_c2.square().mean(dim=(1, 2, 3)).detach().cpu()
        for i in range(int(imgs.shape[0])):
            row = {
                "image_index": int(seen + i),
                "norm_c1_energy": float(norm_c1_img[i].item()),
                "norm_c2_energy": float(norm_c2_img[i].item()),
                "norm_c2_over_c1": float((norm_c2_img[i] / norm_c1_img[i].clamp_min(1e-12)).item()),
                "raw_c1_energy": float(raw_c1_img[i].item()),
                "raw_c2_energy": float(raw_c2_img[i].item()),
                "raw_c2_over_c1": float((raw_c2_img[i] / raw_c1_img[i].clamp_min(1e-12)).item()),
            }
            per_image_rows.append(row)
        seen += int(imgs.shape[0])

    if seen == 0:
        raise RuntimeError("no images evaluated")

    summary = {
        "eval_images": int(seen),
        "norm_c1_energy": total["norm_c1_sq_sum"] / max(1, total["norm_c1_numel"]),
        "norm_c2_energy": total["norm_c2_sq_sum"] / max(1, total["norm_c2_numel"]),
        "raw_c1_energy": total["raw_c1_sq_sum"] / max(1, total["raw_c1_numel"]),
        "raw_c2_energy": total["raw_c2_sq_sum"] / max(1, total["raw_c2_numel"]),
    }
    summary["norm_c2_over_c1"] = summary["norm_c2_energy"] / max(summary["norm_c1_energy"], 1e-12)
    summary["raw_c2_over_c1"] = summary["raw_c2_energy"] / max(summary["raw_c1_energy"], 1e-12)

    for key in ("norm_c1_energy", "norm_c2_energy", "norm_c2_over_c1", "raw_c1_energy", "raw_c2_energy", "raw_c2_over_c1"):
        summary.update(_stats([float(row[key]) for row in per_image_rows], f"image_{key}"))

    channel_rows: list[dict] = []
    for group, values in (
        ("norm_c1", norm_c1_ch_sum / max(1, channel_spatial_count)),
        ("norm_c2", norm_c2_ch_sum / max(1, channel_spatial_count)),
        ("raw_c1", raw_c1_ch_sum / max(1, channel_spatial_count)),
        ("raw_c2", raw_c2_ch_sum / max(1, channel_spatial_count)),
    ):
        for idx, value in enumerate(values.tolist()):
            channel_rows.append({"group": group, "channel_index": idx, "energy": float(value)})
    return summary, per_image_rows, channel_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--crop", type=str, choices=["center", "random"], default="center")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-num-workers", type=int, default=0)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    seed_everything(int(cli.seed))
    ckpt_path = Path(resolve_path(cli.ckpt))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "encoder_state_dict" not in ckpt:
        raise RuntimeError(f"checkpoint has no encoder_state_dict: {ckpt_path}")
    args = make_args(cli, ckpt)
    cfg = cvq_io.build_config(args)
    loader = make_valid_loader(args, str(cli.crop))
    encoder = load_encoder(args, ckpt, cfg.device)
    print(f"loaded checkpoint: {ckpt_path}")
    print(f"checkpoint stage={ckpt.get('stage')} epoch={ckpt.get('epoch')} route={ckpt.get('route')}")
    print(f"eval split=DIV2K_valid crop={cli.crop} images={len(loader.dataset)} max_images={int(cli.max_images)}")

    summary, per_image_rows, channel_rows = evaluate_energy(loader, encoder, args, int(cli.max_images))
    out_dir = Path(resolve_path(cli.out_dir))
    write_csv(out_dir / "per_image_energy.csv", per_image_rows)
    write_csv(out_dir / "per_channel_energy.csv", channel_rows)
    write_json(
        out_dir / "summary.json",
        {
            "script": str(Path(__file__).resolve()),
            "args": vars(cli),
            "checkpoint": {
                "path": str(ckpt_path),
                "stage": ckpt.get("stage"),
                "epoch": ckpt.get("epoch"),
                "route": ckpt.get("route"),
                "latent_ch": int(args.latent_ch),
                "c1_ch": int(args.c1_ch),
                "c2_ch": int(args.latent_ch) - int(args.c1_ch),
            },
            "summary": summary,
        },
    )
    print(
        "normalized "
        f"c1_energy={summary['norm_c1_energy']:.6g} "
        f"c2_energy={summary['norm_c2_energy']:.6g} "
        f"c2/c1={summary['norm_c2_over_c1']:.6g}"
    )
    print(
        "raw "
        f"c1_energy={summary['raw_c1_energy']:.6g} "
        f"c2_energy={summary['raw_c2_energy']:.6g} "
        f"c2/c1={summary['raw_c2_over_c1']:.6g}"
    )
    print(f"wrote: {out_dir / 'summary.json'}")
    print(f"wrote: {out_dir / 'per_image_energy.csv'}")
    print(f"wrote: {out_dir / 'per_channel_energy.csv'}")


if __name__ == "__main__":
    main()
