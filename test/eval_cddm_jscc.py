#!/usr/bin/env python
"""Evaluate JSCC + CDDM with the original CDDM inference design.

For each run directory this script evaluates the same chain used by
``CDDM/Diffusion/Train.py::eval_JSCC_with_CDDM``:

  1. encode the DIV2K valid image;
  2. pass the latent through the original complex AWGN/Rayleigh channel at
     ``snr_db - large_snr``;
  3. normalize by ``sqrt(1 + sigma_square)``;
  4. run ``ChannelDiffusionSampler(y, snr_in, snr_db, h, channel_type)``;
  5. rescale by ``sqrt(pwr)`` and decode with either the CDDM post-trained
     decoder or the base JSCC decoder, depending on ``--cddm-decoder``.

It also reports direct JSCC baselines and the original latent MSE diagnostics
(``MSE_channel`` and ``MSE_channel+CDDM``).
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_train_deps():
    path = os.path.join(PROJECT_ROOT, "train", "train_jscc_cddm_post.py")
    spec = importlib.util.spec_from_file_location("train_jscc_cddm_post_evaldeps", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import eval deps from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deps = _load_train_deps()


@dataclass
class RunSpec:
    run_dir: str
    snr_db: float
    encoder_ckpt: str
    decoder_ckpt: str
    cddm_ckpt: str
    redecoder_ckpt: str | None


@dataclass
class EvalResult:
    run_dir: str
    n_images: int
    snr_db: float
    cddm_channel_snr_db: float
    jscc_train_psnr: float
    jscc_post_psnr: float
    cddm_psnr: float
    gain_vs_jscc_post: float
    gain_vs_jscc_train: float
    mse_channel: float
    mse_channel_cddm: float
    mse_channel_ratio_cddm_over_raw: float


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def seed_torch(seed: int) -> None:
    deps.seed_torch(seed)


def _fmt_num(v: float) -> str:
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return f"{v:g}"


def infer_snr_from_run_dir(run_dir: str) -> float:
    m = re.search(r"snr(-?\d+(?:\.\d+)?)", os.path.basename(run_dir))
    if not m:
        raise ValueError(f"cannot infer SNR from run dir name: {run_dir}")
    return float(m.group(1))


def default_run_dirs() -> list[str]:
    root = os.path.join(PROJECT_ROOT, "checkpoints-jscc")
    return [
        os.path.join(root, "cddm_mse_c4_awgn_snr0"),
        os.path.join(root, "cddm_mse_c4_awgn_snr6"),
        os.path.join(root, "cddm_mse_c4_awgn_snr12"),
    ]


def make_run_spec(
    run_dir: str,
    embed_dim: int,
    redecoder_snr: float,
    require_redecoder: bool,
) -> RunSpec:
    run_dir = os.path.abspath(run_dir)
    base = os.path.basename(run_dir.rstrip(os.sep))
    snr = infer_snr_from_run_dir(run_dir)
    redecoder_ckpt = os.path.join(
        run_dir, f"redecoder_cddm_snr{_fmt_num(redecoder_snr)}_awgn_c{embed_dim}.pth"
    )
    spec = RunSpec(
        run_dir=run_dir,
        snr_db=snr,
        encoder_ckpt=os.path.join(run_dir, f"{base}_encoder_best.pth"),
        decoder_ckpt=os.path.join(run_dir, f"{base}_decoder_best.pth"),
        cddm_ckpt=os.path.join(run_dir, f"cddm_snr{_fmt_num(snr)}_awgn_c{embed_dim}.pt"),
        redecoder_ckpt=redecoder_ckpt if require_redecoder or os.path.isfile(redecoder_ckpt) else None,
    )
    required = [spec.encoder_ckpt, spec.decoder_ckpt, spec.cddm_ckpt]
    if require_redecoder:
        required.append(redecoder_ckpt)
    for path in required:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    return spec


def data_args(args: argparse.Namespace, snr_db: float) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=args.data_dir,
        channel_type=args.channel_type,
        snr_db=snr_db,
        embed_dim=args.embed_dim,
        device=args.device,
        num_workers=args.num_workers,
        val_num_workers=args.val_num_workers,
    )


def model_args(args: argparse.Namespace, spec: RunSpec) -> SimpleNamespace:
    return SimpleNamespace(
        run_dir=spec.run_dir,
        encoder_ckpt=spec.encoder_ckpt,
        decoder_ckpt=spec.decoder_ckpt,
        snr_db=spec.snr_db,
        embed_dim=args.embed_dim,
        channel_type=args.channel_type,
        T=args.T,
        noise_schedule=args.noise_schedule,
        t_max=args.t_max,
        snr_max=args.snr_max,
        snr_min=args.snr_min,
    )


def complex_to_real_feature(y_complex: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.real(y_complex), torch.imag(y_complex)), dim=2)


def direct_channel_feature(
    y_complex: torch.Tensor,
    pwr: torch.Tensor,
    h: torch.Tensor,
    snr_db: float,
    channel_type: str,
) -> torch.Tensor:
    if channel_type == "awgn":
        y_eq = y_complex
    elif channel_type == "rayleigh":
        sigma_square_fix = 1.0 / (10.0 ** (float(snr_db) / 10.0))
        y_eq = y_complex * torch.conj(h) / (torch.abs(h) ** 2 + sigma_square_fix)
    else:
        raise ValueError(f"unsupported channel_type={channel_type}")
    return complex_to_real_feature(y_eq) * torch.sqrt(pwr)


def psnr_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse = (x_hat.float().clamp(0.0, 1.0) - x.float()).pow(2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def load_decoder_state(decoder: torch.nn.Module, path: str) -> None:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    missing, unexpected = decoder.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"redecoder load mismatch: missing={missing}, unexpected={unexpected}")


def resolve_test_snr(spec: RunSpec, args: argparse.Namespace) -> float:
    if args.test_snr_mode == "same":
        return float(spec.snr_db)
    if args.test_snr_mode == "original":
        return float(spec.snr_db) - float(args.large_snr)
    raise ValueError(f"unsupported test_snr_mode={args.test_snr_mode}")


def resolve_cddm_decoder_mode(args: argparse.Namespace) -> str:
    if args.cddm_decoder != "auto":
        return args.cddm_decoder
    return "redecoder" if args.test_snr_mode == "original" else "base"


@torch.no_grad()
def evaluate_one(spec: RunSpec, args: argparse.Namespace, device: torch.device) -> EvalResult:
    seed_torch(args.seed + int(round(spec.snr_db * 100)))
    margs = model_args(args, spec)
    encoder, base_decoder = deps.build_sc(margs, device)

    decoder_mode = resolve_cddm_decoder_mode(args)
    if decoder_mode == "redecoder":
        if spec.redecoder_ckpt is None:
            raise FileNotFoundError(
                f"redecoder checkpoint is required for {spec.run_dir} but was not found"
            )
        cddm_decoder = copy.deepcopy(base_decoder)
        load_decoder_state(cddm_decoder, spec.redecoder_ckpt)
    elif decoder_mode == "base":
        cddm_decoder = base_decoder
    else:
        raise ValueError(f"unsupported cddm_decoder={args.cddm_decoder}")

    cddm = deps.make_cddm(margs, device)
    cddm.load_state_dict(torch.load(spec.cddm_ckpt, map_location=device, weights_only=False))
    cddm.eval()
    sampler = deps.ChannelDiffusionSampler(
        model=cddm,
        noise_schedule=int(args.noise_schedule),
        t_max=float(args.t_max),
        beta_1=float(args.snr_max),
        beta_T=float(args.snr_min),
        T=int(args.T),
    ).to(device)

    encoder.eval()
    base_decoder.eval()
    cddm_decoder.eval()

    dargs = data_args(args, spec.snr_db)
    data_cfg = deps.cddm_data_config(dargs, args.batch_size)
    _train_loader, test_loader = deps.get_cddm_loader(data_cfg)
    if test_loader is None:
        raise RuntimeError("CDDM loader returned no test loader")

    test_snr = resolve_test_snr(spec, args)
    meters = {
        "jscc_train_psnr": AverageMeter(),
        "jscc_post_psnr": AverageMeter(),
        "cddm_psnr": AverageMeter(),
        "mse_channel": AverageMeter(),
        "mse_channel_cddm": AverageMeter(),
    }

    n_seen = 0
    progress = tqdm(
        test_loader,
        dynamic_ncols=True,
        desc=f"SNR {spec.snr_db:g} CDDM eval",
        disable=args.no_progress,
    )
    for images, _labels in progress:
        if args.max_images > 0 and n_seen >= args.max_images:
            break
        if args.max_images > 0 and n_seen + images.shape[0] > args.max_images:
            images = images[: args.max_images - n_seen]
        x0 = images.to(device, non_blocking=device.type == "cuda")
        bs = int(x0.shape[0])

        feature = encoder(x0)

        y_post, pwr_post, h_post = deps.legacy_channel_forward(feature, test_snr, args.channel_type)
        feat_post_direct = direct_channel_feature(
            y_post, pwr_post, h_post, test_snr, args.channel_type
        )
        x_jscc_post = base_decoder(feat_post_direct)

        if abs(test_snr - float(spec.snr_db)) < 1e-12:
            x_jscc_train = x_jscc_post
        else:
            y_train, pwr_train, h_train = deps.legacy_channel_forward(
                feature, float(spec.snr_db), args.channel_type
            )
            feat_train_direct = direct_channel_feature(
                y_train, pwr_train, h_train, float(spec.snr_db), args.channel_type
            )
            x_jscc_train = base_decoder(feat_train_direct)

        sigma_square = 1.0 / (2.0 * 10.0 ** (test_snr / 10.0))
        y_for_cddm = y_post / math.sqrt(1.0 + sigma_square)
        feature_hat_norm = sampler(
            y_for_cddm,
            test_snr,
            float(spec.snr_db),
            h_post,
            args.channel_type,
        )
        feature_hat = feature_hat_norm * torch.sqrt(pwr_post)
        x_cddm = cddm_decoder(feature_hat)

        target_norm = feature * math.sqrt(2.0) / torch.sqrt(pwr_post)
        if args.channel_type == "awgn":
            channel_norm = complex_to_real_feature(y_post) * math.sqrt(2.0)
        elif args.channel_type == "rayleigh":
            sigma_square_fix = 1.0 / (10.0 ** (test_snr / 10.0))
            y_mmse = y_post * torch.conj(h_post) / (torch.abs(h_post) ** 2 + sigma_square_fix)
            channel_norm = complex_to_real_feature(y_mmse) * math.sqrt(2.0)
        else:
            raise ValueError(f"unsupported channel_type={args.channel_type}")
        cddm_norm = feature_hat_norm * math.sqrt(2.0)

        meters["jscc_train_psnr"].update(psnr_per_image(x_jscc_train, x0).sum().item(), bs)
        meters["jscc_post_psnr"].update(psnr_per_image(x_jscc_post, x0).sum().item(), bs)
        meters["cddm_psnr"].update(psnr_per_image(x_cddm, x0).sum().item(), bs)
        meters["mse_channel"].update(F.mse_loss(channel_norm, target_norm).item(), bs)
        meters["mse_channel_cddm"].update(F.mse_loss(cddm_norm, target_norm).item(), bs)

        n_seen += bs
        if not args.no_progress:
            progress.set_postfix(
                {
                    "JSCC@Post": f"{meters['jscc_post_psnr'].avg:.3f}",
                    "CDDM": f"{meters['cddm_psnr'].avg:.3f}",
                }
            )

    mse_channel = meters["mse_channel"].avg
    mse_cddm = meters["mse_channel_cddm"].avg
    cddm_psnr = meters["cddm_psnr"].avg
    jscc_post_psnr = meters["jscc_post_psnr"].avg
    jscc_train_psnr = meters["jscc_train_psnr"].avg
    return EvalResult(
        run_dir=spec.run_dir,
        n_images=n_seen,
        snr_db=float(spec.snr_db),
        cddm_channel_snr_db=test_snr,
        jscc_train_psnr=jscc_train_psnr,
        jscc_post_psnr=jscc_post_psnr,
        cddm_psnr=cddm_psnr,
        gain_vs_jscc_post=cddm_psnr - jscc_post_psnr,
        gain_vs_jscc_train=cddm_psnr - jscc_train_psnr,
        mse_channel=mse_channel,
        mse_channel_cddm=mse_cddm,
        mse_channel_ratio_cddm_over_raw=mse_cddm / max(mse_channel, 1e-12),
    )


def write_outputs(results: list[EvalResult], args: argparse.Namespace) -> None:
    rows = [asdict(r) for r in results]
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "DIV2K_valid_HR",
        "psnr_rule": "mean per-image PSNR, peak=1",
        "cddm_rule": "original CDDM eval_JSCC_with_CDDM chain",
        "large_snr": float(args.large_snr),
        "T": int(args.T),
        "noise_schedule": int(args.noise_schedule),
        "t_max": float(args.t_max),
        "test_snr_mode": args.test_snr_mode,
        "cddm_decoder": resolve_cddm_decoder_mode(args),
        "results": rows,
    }
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate CDDM denoiser + post decoder following original CDDM design.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dirs", nargs="+", default=default_run_dirs())
    p.add_argument("--data-dir", default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--embed-dim", type=int, default=4)
    p.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1, help="Original CDDM eval uses test_batch=1.")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--max-images", type=int, default=0, help="Debug only; 0 evaluates full valid set.")
    p.add_argument("--seed", type=int, default=1024)

    p.add_argument("--large-snr", type=float, default=3.0)
    p.add_argument(
        "--test-snr-mode",
        choices=["original", "same"],
        default="original",
        help="original: test channel SNR = train SNR - large_snr; same: test channel SNR = train SNR.",
    )
    p.add_argument(
        "--cddm-decoder",
        choices=["auto", "base", "redecoder"],
        default="auto",
        help="auto uses post-trained redecoder for original mode and base JSCC decoder for same mode.",
    )
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--noise-schedule", type=int, default=1)
    p.add_argument("--t-max", type=float, default=10.0)
    p.add_argument("--snr-max", type=float, default=1e-4)
    p.add_argument("--snr-min", type=float, default=0.02)

    p.add_argument(
        "--out-json",
        default=os.path.join(PROJECT_ROOT, "checkpoints-jscc", "cddm_eval_results.json"),
    )
    p.add_argument(
        "--out-csv",
        default=os.path.join(PROJECT_ROOT, "checkpoints-jscc", "cddm_eval_results.csv"),
    )
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    decoder_mode = resolve_cddm_decoder_mode(args)
    specs = []
    for d in args.run_dirs:
        snr = infer_snr_from_run_dir(d)
        redecoder_snr = snr - args.large_snr if args.test_snr_mode == "original" else snr
        specs.append(
            make_run_spec(
                d,
                args.embed_dim,
                redecoder_snr=redecoder_snr,
                require_redecoder=decoder_mode == "redecoder",
            )
        )

    print("CDDM eval: original CDDM sampler + selected decoder")
    print(f"device={device}, batch={args.batch_size}, max_images={args.max_images or 'all'}")
    if args.test_snr_mode == "original":
        snr_rule = f"test_snr = train_snr - large_snr = train_snr - {args.large_snr:g}"
    else:
        snr_rule = "test_snr = train_snr"
    print(f"rule: {snr_rule}; cddm_decoder={decoder_mode}; PSNR=mean(per-image PSNR)")

    results: list[EvalResult] = []
    for spec in specs:
        res = evaluate_one(spec, args, device)
        results.append(res)
        print(
            f"[SNR {res.snr_db:g}] N={res.n_images} "
            f"JSCC@train={res.jscc_train_psnr:.4f}dB "
            f"JSCC@test({res.cddm_channel_snr_db:g})={res.jscc_post_psnr:.4f}dB "
            f"CDDM={res.cddm_psnr:.4f}dB "
            f"gain_test={res.gain_vs_jscc_post:+.4f}dB "
            f"MSE_channel={res.mse_channel:.6g} "
            f"MSE_CDDM={res.mse_channel_cddm:.6g}"
        )

    write_outputs(results, args)
    if args.out_json:
        print(f"wrote json: {args.out_json}")
    if args.out_csv:
        print(f"wrote csv: {args.out_csv}")


if __name__ == "__main__":
    main()
