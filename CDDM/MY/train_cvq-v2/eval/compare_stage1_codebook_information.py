# 作用：比较 Stage1 C2 码本是否真的携带信息，评估 C1-only、C2 不量化、C2 量化、随机码字和打乱码字的 PSNR/能量差异。
# 输出：metrics.csv 和 summary.json，包含 codebook usage、perplexity、quant_mse、C2/C1 能量比等指标。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/compare_stage1_codebook_information.py --ckpt MY/VQ-Stage1/cvq_v2_c36_snr12_stage1_best-v2.pth --out-dir /tmp/stage1_codebook_info --max-images 8

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
    psnr_per_image,
    recon_loss,
    resolve_path,
    seed_everything,
    split_c1_c2,
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


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
        writer.writerows(rows)


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
    args.seed = int(cli.seed)
    args.cpu = bool(cli.cpu)
    args.stage = 1
    return args


def make_valid_loader(args: argparse.Namespace, crop: str):
    cfg = cvq_io.build_config(args)
    crop_hw = (int(cfg.image_dims[1]), int(cfg.image_dims[2]))
    transform = transforms.Compose(
        [
            transforms.RandomCrop(crop_hw) if crop == "random" else transforms.CenterCrop(crop_hw),
            transforms.ToTensor(),
        ]
    )
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
    cvq_io.load_state(quantizer, ckpt["quantizer_state_dict"], "quantizer", strict=True)
    encoder.eval()
    decoder.eval()
    quantizer.eval()
    return encoder, decoder, quantizer


def codebook_usage(hist: torch.Tensor) -> dict[str, float]:
    prob = hist.double() / hist.double().sum().clamp_min(1.0)
    nonzero = prob > 0
    entropy = -(prob[nonzero] * prob[nonzero].log2()).sum()
    top = torch.sort(prob, descending=True).values
    return {
        "codebook_used": float(nonzero.sum().item()),
        "codebook_usage": float(nonzero.float().mean().item()),
        "codebook_entropy_bits": float(entropy.item()),
        "codebook_perplexity": float(torch.pow(torch.tensor(2.0), entropy).item()),
        "codebook_top1_ratio": float(top[0].item()) if top.numel() else 0.0,
        "codebook_top10_ratio": float(top[:10].sum().item()) if top.numel() else 0.0,
    }


@torch.no_grad()
def decoder_input_codebook(quantizer) -> tuple[torch.Tensor, torch.Tensor, bool]:
    raw_codebook = quantizer.codebook.detach()
    if hasattr(quantizer, "effective_codebook"):
        return quantizer.effective_codebook().detach(), raw_codebook, True
    return raw_codebook, raw_codebook, False


@torch.no_grad()
def evaluate(loader, encoder, decoder, quantizer, args: argparse.Namespace, max_images: int) -> dict[str, float]:
    device = next(encoder.parameters()).device
    keys = [
        "loss_c1_only",
        "loss_no_quant",
        "loss_quantized",
        "loss_random_code",
        "loss_shuffle_code",
        "psnr_c1_only",
        "psnr_no_quant",
        "psnr_quantized",
        "psnr_random_code",
        "psnr_shuffle_code",
        "norm_c1_energy",
        "norm_c2_energy",
        "quant_mse",
        "q_c2_energy",
    ]
    meters = {key: AverageMeter() for key in keys}
    hist = torch.zeros(int(args.k), dtype=torch.float64)
    seen = 0

    codebook, raw_codebook, uses_effective_codebook = decoder_input_codebook(quantizer)
    for imgs, _labels in loader:
        if int(max_images) > 0 and seen >= int(max_images):
            break
        if int(max_images) > 0 and seen + int(imgs.shape[0]) > int(max_images):
            imgs = imgs[: int(max_images) - seen]
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _c1_power = cvq_io.encode_normalized(imgs, encoder, args)
        z_c1, z_c2 = split_c1_c2(z_norm, args)
        q_c2, idx = quantizer.encode(z_c2)
        hist += torch.bincount(idx.reshape(-1).detach().cpu(), minlength=int(args.k)).double()

        flat_q = q_c2.reshape(-1, q_c2.shape[-2], q_c2.shape[-1])
        q_shuffle = torch.roll(flat_q, shifts=1, dims=0).reshape_as(q_c2)
        rand_idx = torch.randint(0, int(args.k), idx.shape, device=device)
        q_random = codebook[rand_idx.reshape(-1)].reshape_as(q_c2).to(q_c2.dtype)

        variants = {
            "c1_only": torch.zeros_like(q_c2),
            "no_quant": z_c2,
            "quantized": q_c2,
            "random_code": q_random,
            "shuffle_code": q_shuffle,
        }
        bsz = int(imgs.shape[0])
        for name, c2 in variants.items():
            recon = decoder(torch.cat([z_c1, c2], dim=1)).clamp(0.0, 1.0)
            meters[f"loss_{name}"].update(float(recon_loss(recon, imgs).item()), bsz)
            meters[f"psnr_{name}"].update(batch_metric_mean(psnr_per_image(recon, imgs)), bsz)
        meters["norm_c1_energy"].update(float(z_c1.float().square().mean().item()), bsz)
        meters["norm_c2_energy"].update(float(z_c2.float().square().mean().item()), bsz)
        meters["q_c2_energy"].update(float(q_c2.float().square().mean().item()), bsz)
        meters["quant_mse"].update(float((q_c2.float() - z_c2.float()).square().mean().item()), bsz)
        seen += bsz

    metrics = {key: meter.avg for key, meter in meters.items()}
    metrics["eval_images"] = float(seen)
    metrics["delta_psnr_full_minus_c1"] = metrics["psnr_quantized"] - metrics["psnr_c1_only"]
    metrics["delta_psnr_no_quant_minus_c1"] = metrics["psnr_no_quant"] - metrics["psnr_c1_only"]
    metrics["delta_psnr_random_minus_c1"] = metrics["psnr_random_code"] - metrics["psnr_c1_only"]
    metrics["delta_psnr_shuffle_minus_c1"] = metrics["psnr_shuffle_code"] - metrics["psnr_c1_only"]
    metrics["norm_c2_over_c1"] = metrics["norm_c2_energy"] / max(metrics["norm_c1_energy"], 1e-12)
    metrics["q_c2_over_z_c2_energy"] = metrics["q_c2_energy"] / max(metrics["norm_c2_energy"], 1e-12)
    metrics.update(codebook_usage(hist))
    metrics["codebook_metric_uses_effective"] = float(uses_effective_codebook)
    metrics["codebook_energy"] = float(codebook.float().square().mean().item())
    metrics["codebook_l2_mean"] = float(codebook.float().flatten(1).norm(dim=1).mean().item())
    if uses_effective_codebook:
        metrics["raw_codebook_energy"] = float(raw_codebook.float().square().mean().item())
        metrics["raw_codebook_l2_mean"] = float(raw_codebook.float().flatten(1).norm(dim=1).mean().item())
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--crop", type=str, choices=["center", "random"], default="center")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-batch", type=int, default=4)
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
    if ckpt.get("stage") != "stage1":
        raise RuntimeError(f"expected stage1 checkpoint, got stage={ckpt.get('stage')}")
    args = make_args(cli, ckpt)
    cfg = cvq_io.build_config(args)
    loader = make_valid_loader(args, str(cli.crop))
    encoder, decoder, quantizer = load_stage1(args, ckpt, cfg.device)
    metrics = evaluate(loader, encoder, decoder, quantizer, args, int(cli.max_images))
    row = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "crop": str(cli.crop),
        "test_batch": int(args.test_batch),
    }
    row.update(metrics)
    out_dir = Path(resolve_path(cli.out_dir))
    write_csv(out_dir / "metrics.csv", [row])
    write_json(out_dir / "summary.json", {"args": vars(cli), "checkpoint": row, "metrics": metrics})
    print(" ".join(f"{k}={v:.6g}" for k, v in sorted(metrics.items())))
    print(f"wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
