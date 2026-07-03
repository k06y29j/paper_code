# 作用：对每个 C2 通道内部的 16x16 空间维做 PCA，并按 PCA group prefix 解码评估信息增益。
# 输出：summary.json、per_channel_spatial_pca_group_prefix.csv、per_channel_spatial_pca_structure.csv、per_channel_spatial_pca.pt。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/c2_per_channel_spatial_pca_prefix_probe.py --fit-epochs 1 --max-fit-images 16 --max-eval-images 8 --val-num-workers 0

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR.parent
TRAIN_DIR = PACKAGE_DIR / "train"
CDDM_ROOT = PACKAGE_DIR.parents[1]
for path in (TRAIN_DIR, PACKAGE_DIR, CDDM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import check_args, psnr_per_image, real_awgn, resolve_path, seed_everything  # noqa: E402
from shared import build_encoder_decoder, cvq_io, get_loader, split_c1_c2  # noqa: E402
from train_stage2_c2_rvq import DEFAULT_FREQ_GROUP_SIZES  # noqa: E402


DEFAULT_STAGE1_CKPT = (
    "MY/checkpoints-cvq-v2-v01-c36-snr9-k4096/"
    "cvq_v2_v01_c36_snr9_k4096_stage1_best.pth"
)


class SumMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update_tensor(self, values: torch.Tensor) -> None:
        values = values.detach().float().cpu()
        self.sum += float(values.sum().item())
        self.count += int(values.numel())

    def update_scalar(self, value: float, n: int) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

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


def load_stage1(args: argparse.Namespace, encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
    if args.init_stage1_ckpt:
        cvq_io.load_experiment_checkpoint(args.init_stage1_ckpt, encoder=encoder, decoder=decoder, strict=True)
        print(f"loaded stage1 checkpoint: {resolve_path(args.init_stage1_ckpt)}")
        return
    if args.init_ckpt:
        cvq_io.load_experiment_checkpoint(args.init_ckpt, encoder=encoder, decoder=decoder, strict=True)
        return
    cvq_io.load_module_checkpoint(encoder, args.init_jscc_encoder, "init JSCC encoder", strict=True)
    cvq_io.load_module_checkpoint(decoder, args.init_jscc_decoder, "init JSCC decoder", strict=True)


@torch.no_grad()
def collect_c2_samples(
    loader,
    encoder: torch.nn.Module,
    args: argparse.Namespace,
    *,
    epochs: int,
    max_images: int,
) -> torch.Tensor:
    device = next(encoder.parameters()).device
    samples = []
    seen = 0
    for epoch in range(1, int(epochs) + 1):
        epoch_seen = 0
        for imgs, _labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
            _c1, c2 = split_c1_c2(z_norm, args)
            samples.append(c2.detach().float().cpu())
            seen += int(c2.shape[0])
            epoch_seen += int(c2.shape[0])
            if int(max_images) > 0 and seen >= int(max_images):
                break
        print(f"fit collect epoch={epoch}/{int(epochs)} epoch_images={epoch_seen} total_images={seen}", flush=True)
        if int(max_images) > 0 and seen >= int(max_images):
            break
    if not samples:
        raise RuntimeError("no C2 samples collected for PCA")
    out = torch.cat(samples, dim=0)
    if int(max_images) > 0:
        out = out[: int(max_images)]
    print(f"fit C2 tensor={tuple(out.shape)}")
    return out


def fit_per_channel_spatial_pca(c2_samples: torch.Tensor) -> dict[str, torch.Tensor]:
    if c2_samples.ndim != 4:
        raise ValueError(f"expected C2 samples [N,20,16,16], got {tuple(c2_samples.shape)}")
    n, c, h, w = c2_samples.shape
    dim = h * w
    rms = c2_samples.square().mean(dim=(0, 2, 3)).sqrt().clamp_min(1e-6)
    flat = (c2_samples / rms.view(1, c, 1, 1)).reshape(n, c, dim).contiguous()
    means = torch.zeros(c, dim)
    components = torch.zeros(c, dim, dim)
    eigvals = torch.zeros(c, dim)
    for ch in range(c):
        tokens = flat[:, ch, :].contiguous()
        mean = tokens.mean(dim=0)
        centered = tokens - mean
        cov = centered.t().matmul(centered) / max(1, tokens.shape[0] - 1)
        vals, vecs = torch.linalg.eigh(cov)
        order = torch.argsort(vals, descending=True)
        vals = vals[order].clamp_min(0.0).contiguous()
        vecs = vecs[:, order].contiguous()
        means[ch] = mean
        components[ch] = vecs
        eigvals[ch] = vals
    explained = eigvals / eigvals.sum(dim=1, keepdim=True).clamp_min(1e-12)
    print(
        "per-channel spatial PCA fitted "
        f"N={n} C={c} dim={dim} "
        f"mean_top4={float(explained[:, :4].sum(dim=1).mean().item()):.6g} "
        f"mean_top16={float(explained[:, :16].sum(dim=1).mean().item()):.6g} "
        f"mean_top56={float(explained[:, :56].sum(dim=1).mean().item()):.6g} "
        f"mean_top256={float(explained[:, :256].sum(dim=1).mean().item()):.6g}"
    )
    return {
        "channel_rms": rms,
        "pca_mean": means,
        "pca_components": components,
        "pca_eigvals": eigvals,
        "pca_explained": explained,
    }


def pca_transform(c2: torch.Tensor, pca: dict[str, torch.Tensor]) -> torch.Tensor:
    bsz, c, h, w = c2.shape
    dim = h * w
    rms = pca["channel_rms"].to(device=c2.device, dtype=c2.dtype).view(1, c, 1, 1).clamp_min(1e-6)
    mean = pca["pca_mean"].to(device=c2.device, dtype=c2.dtype).view(1, c, dim)
    comp = pca["pca_components"].to(device=c2.device, dtype=c2.dtype)
    flat = (c2 / rms).reshape(bsz, c, dim)
    return torch.einsum("bcd,cdk->bck", flat - mean, comp)


def pca_inverse(coeff: torch.Tensor, pca: dict[str, torch.Tensor], h: int, w: int) -> torch.Tensor:
    bsz, c, dim = coeff.shape
    comp = pca["pca_components"].to(device=coeff.device, dtype=coeff.dtype)
    mean = pca["pca_mean"].to(device=coeff.device, dtype=coeff.dtype).view(1, c, dim)
    rms = pca["channel_rms"].to(device=coeff.device, dtype=coeff.dtype).view(1, c, 1, 1)
    flat = torch.einsum("bck,cdk->bcd", coeff, comp) + mean
    return flat.reshape(bsz, c, h, w) * rms


def reconstruct_prefix(c2: torch.Tensor, pca: dict[str, torch.Tensor], end_dim: int) -> torch.Tensor:
    bsz, c, h, w = c2.shape
    coeff = pca_transform(c2, pca)
    coeff_q = torch.zeros_like(coeff)
    if int(end_dim) > 0:
        coeff_q[:, :, : int(end_dim)] = coeff[:, :, : int(end_dim)]
    return pca_inverse(coeff_q, pca, h, w)


@torch.no_grad()
def decode_candidate_psnr(
    decoder: torch.nn.Module,
    imgs: torch.Tensor,
    c1_in: torch.Tensor,
    candidates: list[tuple[str, torch.Tensor]],
    *,
    candidates_per_forward: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    bsz = int(imgs.shape[0])
    chunk = max(1, int(candidates_per_forward))
    for start in range(0, len(candidates), chunk):
        part = candidates[start : start + chunk]
        keys = [item[0] for item in part]
        c2_stack = torch.stack([item[1] for item in part], dim=0)
        num = int(c2_stack.shape[0])
        c2_cat = c2_stack.reshape(num * bsz, *c2_stack.shape[2:])
        c1_cat = c1_in.unsqueeze(0).expand(num, -1, -1, -1, -1).reshape(num * bsz, *c1_in.shape[1:])
        img_cat = imgs.unsqueeze(0).expand(num, -1, -1, -1, -1).reshape(num * bsz, *imgs.shape[1:])
        recon = decoder(torch.cat([c1_cat, c2_cat], dim=1)).clamp(0.0, 1.0)
        vals = psnr_per_image(recon, img_cat).reshape(num, bsz).detach().cpu()
        for i, key in enumerate(keys):
            out[key] = vals[i]
    return out


@torch.no_grad()
def evaluate_prefixes(
    loader,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    pca: dict[str, torch.Tensor],
    args: argparse.Namespace,
    group_sizes: tuple[int, ...],
) -> tuple[dict[str, SumMeter], int]:
    device = next(encoder.parameters()).device
    ends = []
    start = 0
    for size in group_sizes:
        start += int(size)
        ends.append(start)
    meters = {name: SumMeter() for name in ["psnr_c1_zero", "psnr_full_c2", "pca_mse_q0"]}
    for level in range(0, len(ends) + 1):
        meters[f"psnr_q{level}"] = SumMeter()
        meters[f"pca_mse_q{level}"] = SumMeter()
        meters[f"raw_mse_q{level}"] = SumMeter()
    total_images = 0
    max_images = int(args.max_eval_images)
    for imgs, _labels in loader:
        if max_images > 0 and total_images >= max_images:
            break
        if max_images > 0 and total_images + int(imgs.shape[0]) > max_images:
            imgs = imgs[: max_images - total_images]
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        c1_in = c1 if str(args.c1_mode) == "clean" else real_awgn(c1, float(args.snr_db))
        zero_c2 = torch.zeros_like(c2)

        recs = [reconstruct_prefix(c2, pca, 0)]
        for end in ends:
            recs.append(reconstruct_prefix(c2, pca, end))
        candidates = [("c1_zero", zero_c2), ("full", c2)] + [(f"q{i}", rec) for i, rec in enumerate(recs)]
        vals = decode_candidate_psnr(decoder, imgs, c1_in, candidates, candidates_per_forward=int(args.decode_candidates_per_forward))
        meters["psnr_c1_zero"].update_tensor(vals["c1_zero"])
        meters["psnr_full_c2"].update_tensor(vals["full"])
        coeff = pca_transform(c2, pca)
        for level, rec in enumerate(recs):
            meters[f"psnr_q{level}"].update_tensor(vals[f"q{level}"])
            rec_coeff = pca_transform(rec, pca)
            meters[f"pca_mse_q{level}"].update_scalar(float(F.mse_loss(rec_coeff.float(), coeff.float()).item()), int(imgs.shape[0]))
            meters[f"raw_mse_q{level}"].update_scalar(float(F.mse_loss(rec.float(), c2.float()).item()), int(imgs.shape[0]))
        total_images += int(imgs.shape[0])
        print(f"eval images={total_images}", flush=True)
    return meters, total_images


def prefix_rows(
    meters: dict[str, SumMeter],
    pca: dict[str, torch.Tensor],
    group_sizes: tuple[int, ...],
) -> list[dict]:
    explained = pca["pca_explained"].float().cpu()
    c1 = meters["psnr_c1_zero"].avg
    full = meters["psnr_full_c2"].avg
    rows = []
    rows.append(
        {
            "q": 0,
            "end_dim": 0,
            "group_size": 0,
            "mean_cum_explained": 0.0,
            "psnr": meters["psnr_q0"].avg,
            "gain_over_c1": meters["psnr_q0"].avg - c1,
            "gap_to_full": full - meters["psnr_q0"].avg,
            "pca_mse": meters["pca_mse_q0"].avg,
            "raw_mse": meters["raw_mse_q0"].avg,
        }
    )
    end = 0
    for i, size in enumerate(group_sizes, start=1):
        end += int(size)
        rows.append(
            {
                "q": i,
                "end_dim": end,
                "group_size": int(size),
                "mean_cum_explained": float(explained[:, :end].sum(dim=1).mean().item()),
                "psnr": meters[f"psnr_q{i}"].avg,
                "gain_over_c1": meters[f"psnr_q{i}"].avg - c1,
                "gap_to_full": full - meters[f"psnr_q{i}"].avg,
                "pca_mse": meters[f"pca_mse_q{i}"].avg,
                "raw_mse": meters[f"raw_mse_q{i}"].avg,
            }
        )
    return rows


def channel_structure_rows(pca: dict[str, torch.Tensor], dims: list[int]) -> list[dict]:
    eigvals = pca["pca_eigvals"].float().cpu()
    explained = pca["pca_explained"].float().cpu()
    rows = []
    for ch in range(eigvals.shape[0]):
        row = {"channel": ch}
        for dim in dims:
            dim = min(int(dim), eigvals.shape[1])
            row[f"cum_explained_{dim}"] = float(explained[ch, :dim].sum().item())
        rows.append(row)
    mean_row = {"channel": "mean"}
    for dim in dims:
        dim = min(int(dim), eigvals.shape[1])
        mean_row[f"cum_explained_{dim}"] = float(explained[:, :dim].sum(dim=1).mean().item())
    rows.append(mean_row)
    return rows


def parse_group_sizes(values: list[int]) -> tuple[int, ...]:
    group_sizes = tuple(int(v) for v in values)
    if sum(group_sizes) != 256:
        raise ValueError(f"--pca-group-sizes must sum to 256, got {sum(group_sizes)}")
    return group_sizes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--out-dir", type=str, default="MY/train_cvq-v2/eval/outputs/cvq_v2_c36_snr9_per_channel_spatial_pca_prefix_probe")
    p.add_argument("--init-stage1-ckpt", type=str, default=DEFAULT_STAGE1_CKPT)
    p.add_argument("--init-ckpt", type=str, default="")
    p.add_argument("--init-jscc-encoder", type=str, default="MY/checkpoints-jscc/encoder_snr9_channel_awgn_C36.pt")
    p.add_argument("--init-jscc-decoder", type=str, default="MY/checkpoints-jscc/decoder_snr9_channel_awgn_C36.pt")
    p.add_argument("--snr-db", type=float, default=9.0)
    p.add_argument("--latent-ch", type=int, default=36)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--fit-epochs", type=int, default=10)
    p.add_argument("--max-fit-images", type=int, default=0)
    p.add_argument("--max-eval-images", type=int, default=0)
    p.add_argument("--c1-mode", type=str, choices=["rx", "clean"], default="rx")
    p.add_argument("--pca-group-sizes", type=int, nargs="+", default=list(DEFAULT_FREQ_GROUP_SIZES))
    p.add_argument("--decode-candidates-per-forward", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    check_args(args)
    group_sizes = parse_group_sizes(args.pca_group_sizes)
    seed_everything(int(args.seed))
    out_dir = Path(resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder = build_encoder_decoder(args, cfg.device)
    load_stage1(args, encoder, decoder)
    encoder.eval()
    decoder.eval()

    print("=== C2 per-channel spatial PCA prefix probe ===")
    print(f"device={cfg.device} c1_mode={args.c1_mode} out_dir={out_dir}")
    print(f"experiment: each C2 channel [16,16] -> 256-D PCA, then grouped by {list(group_sizes)}")
    fit_c2 = collect_c2_samples(train_loader, encoder, args, epochs=int(args.fit_epochs), max_images=int(args.max_fit_images))
    pca = fit_per_channel_spatial_pca(fit_c2)
    torch.save({key: value.cpu() for key, value in pca.items()}, out_dir / "per_channel_spatial_pca.pt")
    meters, eval_images = evaluate_prefixes(val_loader, encoder, decoder, pca, args, group_sizes)

    rows = prefix_rows(meters, pca, group_sizes)
    dims = [0]
    end = 0
    for size in group_sizes:
        end += int(size)
        dims.append(end)
    summary = {
        "fit_images": int(fit_c2.shape[0]),
        "fit_epochs": int(args.fit_epochs),
        "eval_images": int(eval_images),
        "c1_mode": str(args.c1_mode),
        "group_sizes": list(group_sizes),
        "psnr_c1_zero": meters["psnr_c1_zero"].avg,
        "psnr_full_c2": meters["psnr_full_c2"].avg,
        "psnr_q10": meters[f"psnr_q{len(group_sizes)}"].avg,
        "q10_minus_full": meters[f"psnr_q{len(group_sizes)}"].avg - meters["psnr_full_c2"].avg,
    }
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "per_channel_spatial_pca_group_prefix.csv", rows)
    write_csv(out_dir / "per_channel_spatial_pca_structure.csv", channel_structure_rows(pca, dims[1:]))

    print(
        "summary "
        f"psnr_c1_zero={summary['psnr_c1_zero']:.6g} "
        f"psnr_q1={meters['psnr_q1'].avg:.6g} "
        f"psnr_q3={meters['psnr_q3'].avg:.6g} "
        f"psnr_q5={meters['psnr_q5'].avg:.6g} "
        f"psnr_q10={summary['psnr_q10']:.6g} "
        f"psnr_full_c2={summary['psnr_full_c2']:.6g}"
    )
    print(f"wrote {out_dir / 'per_channel_spatial_pca_group_prefix.csv'}")
    print(f"wrote {out_dir / 'per_channel_spatial_pca_structure.csv'}")


if __name__ == "__main__":
    main()
