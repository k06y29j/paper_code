from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import check_args, resolve_path, setup_log_file, write_json
from train_stage0 import check_stage0_args, train_stage0
from train_stage1 import default_stage0_best_ckpt, train_stage1
from train_stage2 import train_stage2


def _load_local_io():
    spec = importlib.util.spec_from_file_location("cvq_v2_cli_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cvq_io = _load_local_io()


def _default_log_file(args: argparse.Namespace) -> str:
    save_dir = Path(resolve_path(args.save_dir))
    snr = f"{float(args.snr_db):g}"
    if int(args.stage) == 0:
        return str(save_dir / f"stage0_jscc_v2_c36_snr{snr}.log")
    if int(args.stage) == 1:
        return str(save_dir / f"stage1_cvq_v2_c36_snr{snr}_k{int(args.k)}.log")
    if int(args.stage) == 2:
        return str(save_dir / f"stage2_cvq_v2_c36_snr{snr}_k{int(args.k)}.log")
    raise ValueError(args.stage)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=cvq_io.default_save_dir())
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default="")
    p.add_argument("--init-jscc-encoder", type=str, default=cvq_io.default_jscc_encoder_c36_snr12())
    p.add_argument("--init-jscc-decoder", type=str, default=cvq_io.default_jscc_decoder_c36_snr12())
    p.add_argument("--snr-db", type=float, default=9.0)
    p.add_argument("--latent-ch", type=int, default=36)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--k", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--decoder-lr", type=float, default=2e-5)
    p.add_argument("--codebook-lr", type=float, default=1e-4)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--vq-chunk-size", type=int, default=128)
    p.add_argument("--lambda-c1", type=float, default=2.0)
    p.add_argument("--lambda-drop", type=float, default=1.0)
    p.add_argument("--lambda-full", type=float, default=0.25)
    p.add_argument("--lambda-vq", type=float, default=1.0)
    p.add_argument(
        "--nested-drop-ratio",
        "--c2-dropout-prob",
        dest="nested_drop_ratio",
        type=float,
        default=1.0,
        help="Probability of applying C2 prefix nested dropout; otherwise full C2 is kept.",
    )
    p.add_argument("--init-codebook-from-jscc", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--latent-cache-maps", type=int, default=262144)
    p.add_argument("--latent-cache-path", type=str, default="")
    p.add_argument("--reuse-latent-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kmeans-iters", type=int, default=30)
    p.add_argument("--kmeans-assign-chunk", type=int, default=4096)
    p.add_argument("--kmeans-device", type=str, default="")
    p.add_argument("--abort-bad-kmeans-init", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-init-quant-mse", type=float, default=0.15)
    p.add_argument("--min-init-psnr-q-full", type=float, default=20.0)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def _normalize_stage_defaults(args: argparse.Namespace) -> None:
    if int(args.stage) == 1 and not args.init_ckpt:
        args.init_ckpt = default_stage0_best_ckpt()
    if int(args.stage) == 1 and args.init_jscc_encoder == cvq_io.default_jscc_encoder_c36_snr12():
        args.init_jscc_encoder = ""
    if int(args.stage) == 1 and args.init_jscc_decoder == cvq_io.default_jscc_decoder_c36_snr12():
        args.init_jscc_decoder = ""


def main() -> None:
    args = parse_args()
    _normalize_stage_defaults(args)
    if int(args.stage) == 0:
        check_stage0_args(args)
    else:
        check_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = _default_log_file(args)
    setup_log_file(args.log_file)
    write_json(Path(args.save_dir) / f"stage{int(args.stage)}_args.json", vars(args))
    if int(args.stage) == 0:
        train_stage0(args)
    elif int(args.stage) == 1:
        train_stage1(args)
    elif int(args.stage) == 2:
        train_stage2(args)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
