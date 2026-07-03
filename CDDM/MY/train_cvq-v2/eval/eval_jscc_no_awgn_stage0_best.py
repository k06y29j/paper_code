# 作用：批量评估 no-AWGN Stage0 JSCC checkpoint 在不同 latent 通道数上的 full reconstruction PSNR。
# 输出：psnr_full_by_channel.csv 和 summary.json。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/eval_jscc_no_awgn_stage0_best.py --channels 16,24,36 --val-num-workers 0

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR.parent
CDDM_ROOT = PACKAGE_DIR.parents[1]
for path in (PACKAGE_DIR, CDDM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Autoencoder.data.datasets import get_loader  # noqa: E402
from common import format_metrics, resolve_path, seed_everything  # noqa: E402
from train_stage0 import build_jscc_models, cvq_io, validate_stage0  # noqa: E402


DEFAULT_CKPT_DIR = "MY/jscc-no-awgn"
DEFAULT_OUT_DIR = "MY/jscc-no-awgn/eval_psnr_full"


def parse_int_list(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("empty channel list")
    return out


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


def checkpoint_path(ckpt_dir: str, latent_ch: int) -> Path:
    return Path(resolve_path(ckpt_dir)) / f"cvq_v2_c{int(latent_ch)}_stage0_best_no-c1.pth"


def load_checkpoint_meta(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if "encoder_state_dict" not in obj or "decoder_state_dict" not in obj:
        raise RuntimeError(f"not a stage0 JSCC checkpoint: {path}")
    return obj


def make_model_args(cli: argparse.Namespace, ckpt: dict, ckpt_path_abs: Path, latent_ch: int) -> argparse.Namespace:
    saved_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    args = argparse.Namespace()
    args.data_dir = str(cli.data_dir)
    args.save_dir = str(ckpt_path_abs.parent)
    args.log_file = ""
    args.init_ckpt = str(ckpt_path_abs)
    args.init_jscc_encoder = ""
    args.init_jscc_decoder = ""
    args.snr_db = float(ckpt.get("snr_db", saved_args.get("snr_db", cli.snr_db)))
    args.latent_ch = int(ckpt.get("latent_ch", saved_args.get("latent_ch", latent_ch)))
    args.c1_ch = int(ckpt.get("c1_ch", saved_args.get("c1_ch", cli.c1_ch)))
    args.latent_h = int(saved_args.get("latent_h", cli.latent_h))
    args.latent_w = int(saved_args.get("latent_w", cli.latent_w))
    args.batch_size = int(cli.batch_size)
    args.test_batch = int(cli.test_batch)
    args.num_workers = int(cli.num_workers)
    args.val_num_workers = int(cli.val_num_workers)
    args.lr = float(saved_args.get("lr", 1e-4))
    args.lambda_c1 = float(saved_args.get("lambda_c1", 0.0))
    args.lambda_drop = float(saved_args.get("lambda_drop", 0.0))
    args.lambda_full = float(saved_args.get("lambda_full", 1.0))
    args.nested_drop_ratio = float(saved_args.get("nested_drop_ratio", cli.nested_drop_ratio))
    args.val_every = int(saved_args.get("val_every", 5))
    args.latest_every = int(saved_args.get("latest_every", 10))
    args.seed = int(cli.seed)
    args.cpu = bool(cli.cpu)
    args.stage = 0
    return args


def evaluate_checkpoint(cli: argparse.Namespace, latent_ch: int) -> dict:
    ckpt_path_abs = checkpoint_path(cli.ckpt_dir, latent_ch)
    ckpt = load_checkpoint_meta(ckpt_path_abs)
    args = make_model_args(cli, ckpt, ckpt_path_abs, latent_ch)
    seed_everything(int(args.seed))

    cfg = cvq_io.build_config(args)
    _train_loader, val_loader = get_loader(cfg)
    if val_loader is None:
        raise RuntimeError("validation loader is None")

    encoder, decoder = build_jscc_models(args, cfg.device)
    missing, unexpected = encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"encoder load mismatch: missing={missing} unexpected={unexpected}")
    missing, unexpected = decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"decoder load mismatch: missing={missing} unexpected={unexpected}")

    metrics = validate_stage0(val_loader, encoder, decoder, args)
    row = {
        "latent_ch": int(args.latent_ch),
        "c1_ch": int(args.c1_ch),
        "c2_ch": int(args.latent_ch) - int(args.c1_ch),
        "checkpoint": str(ckpt_path_abs),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_stage": str(ckpt.get("stage", "")),
        "checkpoint_route": str(ckpt.get("route", "")),
        "checkpoint_metric_psnr_full": float(ckpt.get("metrics", {}).get("psnr_full", float("nan"))),
        "eval_images": int(len(val_loader.dataset)),
        "test_batch": int(args.test_batch),
        "val_num_workers": int(args.val_num_workers),
        "snr_db": float(args.snr_db),
    }
    row.update({k: float(v) for k, v in metrics.items()})
    print(f"[eval c{int(args.latent_ch)}] {format_metrics(metrics)} ckpt_epoch={row['checkpoint_epoch']}")
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--channels", type=parse_int_list, default=parse_int_list("16"))
    p.add_argument("--snr-db", type=float, default=12.0)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--nested-drop-ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    out_dir = Path(resolve_path(cli.out_dir))
    rows = [evaluate_checkpoint(cli, c) for c in cli.channels]
    write_csv(out_dir / "psnr_full_by_channel.csv", rows)
    write_json(
        out_dir / "summary.json",
        {
            "script": str(Path(__file__).resolve()),
            "ckpt_dir": str(Path(resolve_path(cli.ckpt_dir))),
            "data_dir": str(Path(resolve_path(cli.data_dir))),
            "channels": [int(c) for c in cli.channels],
            "rows": rows,
        },
    )
    print(f"wrote: {out_dir / 'psnr_full_by_channel.csv'}")
    print(f"wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
