# 作用：评估 Stage1 checkpoint 中 C2 不经过码本量化时的 PSNR，并和量化 C2、C1-only 对照。
# 输出：metrics.csv 和 summary.json，核心指标为 psnr_c1_only、psnr_no_quant、psnr_quantized。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/eval_stage1_c2_no_quant_psnr.py --max-images 8 --val-num-workers 0

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
from common import (  # noqa: E402
    AverageMeter,
    batch_metric_mean,
    format_metrics,
    psnr_per_image,
    recon_loss,
    resolve_path,
    seed_everything,
    split_c1_c2,
)


DEFAULT_CKPT = "MY/VQ-Stage1/cvq_v2_c36_snr12_stage1_best-v2.pth"
DEFAULT_OUT_DIR = (
    "MY/train_cvq-v2/eval/outputs/"
    "cvq_v2_c36_snr12_stage1_best_valid_c2_no_quant-v2"
)


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
    args.test_batch = int(cli.test_batch)
    args.batch_size = int(cli.batch_size)
    args.num_workers = int(cli.num_workers)
    args.val_num_workers = int(cli.val_num_workers)
    args.vq_beta = float(getattr(args, "vq_beta", 0.25))
    args.vq_chunk_size = int(getattr(args, "vq_chunk_size", 128))
    args.lambda_full = float(getattr(args, "lambda_full", 1.0))
    args.lambda_vq = float(getattr(args, "lambda_vq", 0.25))
    args.seed = int(cli.seed)
    args.cpu = bool(cli.cpu)
    args.stage = 1
    return args


def encode_for_eval(imgs: torch.Tensor, encoder, args: argparse.Namespace, latent_power_norm: str):
    z_raw, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
    if latent_power_norm == "c1":
        return z_norm
    if latent_power_norm == "none":
        return z_raw.float()
    raise ValueError(f"unsupported latent_power_norm={latent_power_norm}")


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


def load_stage1(args: argparse.Namespace, ckpt: dict, device: torch.device):
    encoder, decoder, quantizer = cvq_io.build_models(args, device)
    cvq_io.load_state(encoder, ckpt["encoder_state_dict"], "encoder", strict=True)
    cvq_io.load_state(decoder, ckpt["decoder_state_dict"], "decoder", strict=True)
    if "quantizer_state_dict" in ckpt:
        cvq_io.load_state(quantizer, ckpt["quantizer_state_dict"], "quantizer", strict=True)
    encoder.eval()
    decoder.eval()
    quantizer.eval()
    return encoder, decoder, quantizer


@torch.no_grad()
def evaluate(
    loader,
    encoder,
    decoder,
    quantizer,
    args: argparse.Namespace,
    max_images: int,
    latent_power_norm: str,
) -> dict[str, float]:
    device = next(encoder.parameters()).device
    meter_keys = [
        "loss_no_quant",
        "loss_quantized",
        "loss_c1_only",
        "quant_mse",
        "psnr_no_quant",
        "psnr_quantized",
        "psnr_c1_only",
    ]
    meters = {key: AverageMeter() for key in meter_keys}
    seen = 0
    for imgs, _labels in loader:
        if int(max_images) > 0 and seen >= int(max_images):
            break
        if int(max_images) > 0 and seen + int(imgs.shape[0]) > int(max_images):
            imgs = imgs[: int(max_images) - seen]
        imgs = imgs.to(device, non_blocking=True)
        z_norm = encode_for_eval(imgs, encoder, args, latent_power_norm)
        z_c1, z_c2 = split_c1_c2(z_norm, args)
        q_c2, _idx = quantizer.encode(z_c2)

        x_no_quant = decoder(torch.cat([z_c1, z_c2], dim=1)).clamp(0.0, 1.0)
        x_quantized = decoder(torch.cat([z_c1, q_c2], dim=1)).clamp(0.0, 1.0)
        x_c1_only = decoder(torch.cat([z_c1, torch.zeros_like(z_c2)], dim=1)).clamp(0.0, 1.0)

        bsz = int(imgs.shape[0])
        meters["loss_no_quant"].update(float(recon_loss(x_no_quant, imgs).item()), bsz)
        meters["loss_quantized"].update(float(recon_loss(x_quantized, imgs).item()), bsz)
        meters["loss_c1_only"].update(float(recon_loss(x_c1_only, imgs).item()), bsz)
        meters["quant_mse"].update(float(torch.mean((q_c2.float() - z_c2.float()).square()).item()), bsz)
        meters["psnr_no_quant"].update(batch_metric_mean(psnr_per_image(x_no_quant, imgs)), bsz)
        meters["psnr_quantized"].update(batch_metric_mean(psnr_per_image(x_quantized, imgs)), bsz)
        meters["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1_only, imgs)), bsz)
        seen += bsz
    metrics = {key: value.avg for key, value in meters.items()}
    metrics["eval_images"] = float(seen)
    return metrics


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
    p.add_argument("--latent-power-norm", type=str, choices=["c1", "none"], default="c1")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    seed_everything(int(cli.seed))
    ckpt_path = Path(resolve_path(cli.ckpt))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if ckpt.get("stage") != "stage1":
        raise RuntimeError(f"expected stage1 checkpoint, got stage={ckpt.get('stage')}")
    args = make_args(cli, ckpt)
    cfg = cvq_io.build_config(args)
    loader = make_valid_loader(args, str(cli.crop))
    encoder, decoder, quantizer = load_stage1(args, ckpt, cfg.device)
    print(f"loaded checkpoint: {ckpt_path}")
    print(f"checkpoint stage={ckpt.get('stage')} epoch={ckpt.get('epoch')} route={ckpt.get('route')}")
    print(f"eval split=DIV2K_valid crop={cli.crop} images={len(loader.dataset)} max_images={int(cli.max_images)}")

    metrics = evaluate(loader, encoder, decoder, quantizer, args, int(cli.max_images), str(cli.latent_power_norm))
    row = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_stage": str(ckpt.get("stage", "")),
        "checkpoint_route": str(ckpt.get("route", "")),
        "checkpoint_psnr_full_quantized": float(ckpt.get("metrics", {}).get("psnr_full", float("nan"))),
        "latent_ch": int(args.latent_ch),
        "c1_ch": int(args.c1_ch),
        "c2_ch": int(args.latent_ch) - int(args.c1_ch),
        "k": int(args.k),
        "crop": str(cli.crop),
        "test_batch": int(args.test_batch),
        "val_num_workers": int(args.val_num_workers),
        "snr_db": float(args.snr_db),
        "latent_power_norm": str(cli.latent_power_norm),
    }
    row.update({key: float(value) for key, value in metrics.items()})
    out_dir = Path(resolve_path(cli.out_dir))
    write_csv(out_dir / "metrics.csv", [row])
    write_json(
        out_dir / "summary.json",
        {
            "script": str(Path(__file__).resolve()),
            "args": vars(cli),
            "checkpoint": row,
            "metrics": metrics,
        },
    )
    print(format_metrics(metrics))
    print(f"wrote: {out_dir / 'metrics.csv'}")
    print(f"wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
