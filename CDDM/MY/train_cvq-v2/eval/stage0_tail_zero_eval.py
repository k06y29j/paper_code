# 作用：在 Stage0 checkpoint 上不经过信道直接解码，逐步把 C2 tail 通道置零，观察每个 tail 前缀对 PSNR 的贡献。
# 输出：tail_zero_passes.csv、tail_zero_summary.csv、summary.json。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/stage0_tail_zero_eval.py --include-full --max-images 8 --val-num-workers 0

from __future__ import annotations

import argparse
import csv
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
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder  # noqa: E402
from common import CDDMJSCCConfig, psnr_per_image, resolve_path, seed_everything, split_c1_c2  # noqa: E402

import importlib.util  # noqa: E402


DEFAULT_CKPT = (
    "MY/checkpoints-cvq-v2-c36-snr12-k16384/"
    "cvq_v2_c36_snr12_stage0_best.pth"
)
DEFAULT_OUT_DIR = (
    "MY/train_cvq-v2/eval/outputs/"
    "cvq_v2_c36_snr12_stage0_best_tail_zero_1to20_no_channel"
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


class SumMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update_tensor(self, values: torch.Tensor) -> None:
        vals = values.detach().float().cpu()
        self.sum += float(vals.sum().item())
        self.count += int(vals.numel())

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
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


def build_config(args: argparse.Namespace) -> CDDMJSCCConfig:
    return CDDMJSCCConfig(
        C=int(args.latent_ch),
        SNRs=float(args.snr_db),
        channel_type="awgn",
        batch_size=int(args.batch_size),
        test_batch=int(args.test_batch),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.val_num_workers),
        train_data_dir=resolve_path(str(Path(args.data_dir) / "DIV2K_train_HR")),
        test_data_dir=resolve_path(str(Path(args.data_dir) / "DIV2K_valid_HR")),
        CUDA=(not bool(args.cpu)),
    )


def load_stage0(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    cfg = build_config(args)
    encoder = JSCC_encoder(cfg, int(args.latent_ch)).to(device)
    decoder = JSCC_decoder(cfg, int(args.latent_ch)).to(device)
    ckpt_path = resolve_path(args.ckpt)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cvq_io.load_state(encoder, obj["encoder_state_dict"], "encoder", strict=True)
    cvq_io.load_state(decoder, obj["decoder_state_dict"], "decoder", strict=True)
    encoder.eval()
    decoder.eval()
    print(f"loaded stage0 checkpoint: {ckpt_path}")
    print(f"checkpoint stage={obj.get('stage')} epoch={obj.get('epoch')} route={obj.get('route')}")
    if "metrics" in obj:
        print(f"checkpoint metrics: {obj['metrics']}")
    return encoder, decoder, obj


def make_valid_loader(args: argparse.Namespace, pass_id: int):
    cfg = build_config(args)
    crop_size = (int(args.image_size), int(args.image_size))
    if str(args.crop) == "random":
        transform = transforms.Compose([transforms.RandomCrop(crop_size), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.CenterCrop(crop_size), transforms.ToTensor()])
    dataset = FlatImageFolder(root=cfg.test_data_dir, transform=transform)

    def worker_init_fn(worker_id: int) -> None:
        seed = int(args.seed) + int(pass_id) * 1009 + int(worker_id)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=int(args.test_batch),
        shuffle=False,
        num_workers=int(args.val_num_workers),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )


def zero_tail_counts(args: argparse.Namespace) -> list[int]:
    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    if c2_ch <= 0:
        raise ValueError("C2 channel count must be positive")
    counts = list(range(1, c2_ch + 1))
    if bool(args.include_full):
        counts = [0] + counts
    return counts


def tail_zero_c2(c2: torch.Tensor, keep: int) -> torch.Tensor:
    out = torch.zeros_like(c2)
    if int(keep) > 0:
        out[:, : int(keep)] = c2[:, : int(keep)]
    return out


@torch.no_grad()
def evaluate_pass(
    loader,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    args: argparse.Namespace,
    pass_id: int,
) -> tuple[list[dict], int]:
    device = next(encoder.parameters()).device
    zero_counts = zero_tail_counts(args)
    meters = {zero_count: SumMeter() for zero_count in zero_counts}
    total_images = 0
    max_images = int(args.max_images)
    for imgs, _labels in loader:
        if max_images > 0 and total_images >= max_images:
            break
        if max_images > 0 and total_images + int(imgs.shape[0]) > max_images:
            imgs = imgs[: max_images - total_images]
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        chunk = max(1, int(args.zero_candidates_per_forward))
        for start in range(0, len(zero_counts), chunk):
            part = zero_counts[start : start + chunk]
            c2_stack = []
            for zero_count in part:
                keep = c2.shape[1] - int(zero_count)
                c2_stack.append(tail_zero_c2(c2, keep))
            num = len(c2_stack)
            c2_cat = torch.stack(c2_stack, dim=0).reshape(num * imgs.shape[0], *c2.shape[1:])
            c1_cat = c1.unsqueeze(0).expand(num, -1, -1, -1, -1).reshape(num * imgs.shape[0], *c1.shape[1:])
            img_cat = imgs.unsqueeze(0).expand(num, -1, -1, -1, -1).reshape(num * imgs.shape[0], *imgs.shape[1:])
            recon = decoder(torch.cat([c1_cat, c2_cat], dim=1)).clamp(0.0, 1.0)
            vals = psnr_per_image(recon, img_cat).reshape(num, imgs.shape[0])
            for idx, zero_count in enumerate(part):
                meters[zero_count].update_tensor(vals[idx])
        total_images += int(imgs.shape[0])
    rows = []
    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    for zeroed in zero_counts:
        keep = c2_ch - int(zeroed)
        first_global_zero = int(args.c1_ch) + int(keep) if zeroed > 0 else ""
        last_global_zero = int(args.latent_ch) - 1 if zeroed > 0 else ""
        rows.append(
            {
                "pass": int(pass_id),
                "crop": str(args.crop),
                "images": int(total_images),
                "keep_c2": int(keep),
                "zero_c2_tail": int(zeroed),
                "global_zero_channels": (
                    f"{first_global_zero}-{last_global_zero}" if zeroed > 0 else ""
                ),
                "psnr": float(meters[zeroed].avg),
            }
        )
    return rows, total_images


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[float]] = {}
    images: dict[int, int] = {}
    for row in rows:
        zeroed = int(row["zero_c2_tail"])
        grouped.setdefault(zeroed, []).append(float(row["psnr"]))
        images[zeroed] = images.get(zeroed, 0) + int(row["images"])
    out = []
    full_mean = float(np.mean(grouped[0])) if 0 in grouped else None
    c2_ch = max(grouped)
    for zeroed in sorted(grouped.keys()):
        vals = np.asarray(grouped[zeroed], dtype=np.float64)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out.append(
            {
                "keep_c2": int(c2_ch - zeroed),
                "zero_c2_tail": int(zeroed),
                "passes": int(vals.size),
                "total_images": int(images[zeroed]),
                "psnr_mean": mean,
                "psnr_std": std,
                "gap_to_full_mean": float(full_mean - mean) if full_mean is not None else "",
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--snr-db", type=float, default=12.0)
    p.add_argument("--latent-ch", type=int, default=36)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--include-full", action="store_true", help="Also evaluate zero_tail=0 as a full-C2 reference.")
    p.add_argument("--crop", choices=["random", "center"], default="center")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--test-batch", type=int, default=4)
    p.add_argument("--zero-candidates-per-forward", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260615)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.latent_ch) != 36 or int(args.c1_ch) != 16:
        raise ValueError("expected latent_ch=36 and c1_ch=16 for this CVQ-v2 stage0 checkpoint")
    if int(args.latent_ch) - int(args.c1_ch) != 20:
        raise ValueError("expected C2=20")
    seed_everything(int(args.seed))
    cfg = build_config(args)
    out_dir = Path(resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder, decoder, ckpt = load_stage0(args, cfg.device)

    print("=== Stage0 tail-zero eval | no channel ===")
    print(f"device={cfg.device} checkpoint={resolve_path(args.ckpt)}")
    print(f"data={cfg.test_data_dir} crop={args.crop} passes={int(args.passes)}")
    print(f"C1=0..{int(args.c1_ch) - 1} C2={int(args.c1_ch)}..{int(args.latent_ch) - 1}")
    print(f"zero_tail_counts={zero_tail_counts(args)}; zero from C2 tail backward one channel at a time")

    rows: list[dict] = []
    for pass_id in range(1, int(args.passes) + 1):
        seed = int(args.seed) + pass_id * 1009
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        loader = make_valid_loader(args, pass_id)
        pass_rows, images = evaluate_pass(loader, encoder, decoder, args, pass_id)
        rows.extend(pass_rows)
        msg = " ".join(f"zero{r['zero_c2_tail']}={float(r['psnr']):.4f}" for r in pass_rows)
        print(f"[pass {pass_id:02d}/{int(args.passes):02d}] images={images} {msg}", flush=True)

    summary_rows = summarize(rows)
    write_csv(out_dir / "tail_zero_passes.csv", rows)
    write_csv(out_dir / "tail_zero_summary.csv", summary_rows)
    summary = {
        "checkpoint": resolve_path(args.ckpt),
        "checkpoint_stage": ckpt.get("stage"),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_metrics": ckpt.get("metrics", {}),
        "no_channel": True,
        "crop": str(args.crop),
        "passes": int(args.passes),
        "zero_tail_counts": zero_tail_counts(args),
        "summary": summary_rows,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"wrote {out_dir / 'tail_zero_passes.csv'}")
    print(f"wrote {out_dir / 'tail_zero_summary.csv'}")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
