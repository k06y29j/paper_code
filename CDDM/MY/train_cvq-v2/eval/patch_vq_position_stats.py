# 作用：统计 patch_vq Stage2 中仅按 C2 通道和 patch 位置选 top-M code 的候选覆盖率。
# 输出：coverage_summary.json、coverage_by_position_M*.csv，以及可选 hist_train.pt/candidate_top*.pt。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/patch_vq_position_stats.py --stat-epochs 1 --max-batches 1 --m-values 64 --val-num-workers 0

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR.parent
TRAIN_DIR = PACKAGE_DIR / "train"
CDDM_ROOT = PACKAGE_DIR.parents[1]
for path in (TRAIN_DIR, PACKAGE_DIR, CDDM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import check_args, resolve_path, seed_everything, setup_log_file  # noqa: E402
from shared import build_quantizer, cvq_io, get_loader, split_c1_c2  # noqa: E402
from Autoencoder.data.datasets import worker_init_fn_seed  # noqa: E402
from Autoencoder.net.network import JSCC_encoder  # noqa: E402


DEFAULT_STAGE2_CKPT = (
    CDDM_ROOT
    / "MY"
    / "checkpoints-cvq-v2-v01-c36-snr9-k4096"
    / "cvq_v2_v01_c36_snr9_k2048_stage2_best.pth"
)


def parse_m_values(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("at least one M value is required")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage2-ckpt", type=str, default=str(DEFAULT_STAGE2_CKPT))
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--m-values", type=str, default="64,128,256")
    p.add_argument("--split", type=str, choices=["train", "valid"], default="train")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--val-num-workers", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--vq-chunk-size", type=int, default=None)
    p.add_argument("--stat-epochs", type=int, default=20, help="number of random-crop passes over the selected split")
    p.add_argument("--max-batches", type=int, default=0, help="0 means full split per stat epoch")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--save-hist", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-candidates", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-file", type=str, default="")
    return p.parse_args()


def checkpoint_args(ckpt: dict, cli: argparse.Namespace) -> argparse.Namespace:
    base = dict(ckpt.get("args", {}))
    for key in ["snr_db", "latent_ch", "c1_ch", "k"]:
        if key in ckpt and key not in base:
            base[key] = ckpt[key]
    args = argparse.Namespace(**base)

    overrides = {
        "batch_size": cli.batch_size,
        "num_workers": cli.num_workers,
        "val_num_workers": cli.val_num_workers,
        "data_dir": cli.data_dir,
        "vq_chunk_size": cli.vq_chunk_size,
        "seed": cli.seed,
        "cpu": cli.cpu,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(args, key, value)

    defaults = {
        "quantizer": "patch_vq",
        "patch_size": 4,
        "vq_beta": 0.25,
        "vq_chunk_size": 128,
        "latent_h": 16,
        "latent_w": 16,
        "test_batch": 1,
        "num_workers": 16,
        "val_num_workers": 8,
        "cpu": False,
        "seed": 20260610,
        "save_dir": str(Path(resolve_path(cli.stage2_ckpt)).parent),
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.stage = 2
    check_args(args)
    if str(args.quantizer).lower() != "patch_vq":
        raise ValueError(f"this diagnostic requires patch_vq, got quantizer={args.quantizer}")
    return args


def default_out_dir(stage2_ckpt: str, split: str, stat_epochs: int) -> Path:
    stem = Path(stage2_ckpt).stem
    return THIS_DIR / "outputs" / f"{stem}_{split}_position_stats_e{int(stat_epochs)}"


def load_stage2(args: argparse.Namespace, ckpt: dict, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    cfg = cvq_io.build_config(args)
    encoder = JSCC_encoder(cfg, int(args.latent_ch)).to(device)
    quantizer = build_quantizer(args, device)
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    quantizer.load_state_dict(ckpt["quantizer_state_dict"], strict=True)
    encoder.eval()
    quantizer.eval()
    return encoder, quantizer


def make_stats_loader(loader, cfg, args: argparse.Namespace, split: str):
    if split != "train":
        return loader
    nw = int(getattr(cfg, "num_workers", int(args.num_workers)))
    pin = bool(getattr(cfg, "pin_memory", True))
    pw = bool(getattr(cfg, "persistent_workers", False)) and nw > 0
    pf = 2 if nw > 0 else None
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    return torch.utils.data.DataLoader(
        dataset=loader.dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=pw,
        prefetch_factor=pf,
        worker_init_fn=worker_init_fn_seed,
        generator=generator,
        drop_last=False,
    )


def accumulate_hist(
    loader,
    encoder: torch.nn.Module,
    quantizer: torch.nn.Module,
    args: argparse.Namespace,
    *,
    stat_epochs: int,
    max_batches: int,
) -> tuple[torch.Tensor, int, int]:
    device = next(encoder.parameters()).device
    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    gh = int(args.latent_h) // int(args.patch_size)
    gw = int(args.latent_w) // int(args.patch_size)
    k = int(args.k)
    hist = torch.zeros(c2_ch, gh, gw, k, dtype=torch.long)
    total_images = 0
    total_batches = 0
    pos_count = c2_ch * gh * gw

    with torch.no_grad():
        for epoch in range(1, int(stat_epochs) + 1):
            epoch_images = 0
            epoch_batches = 0
            for batch_idx, (imgs, _labels) in enumerate(loader, start=1):
                imgs = imgs.to(device, non_blocking=True)
                _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
                _c1, c2 = split_c1_c2(z_norm, args)
                _q, idx = quantizer.encode(c2)
                if tuple(idx.shape[1:]) != (c2_ch, gh, gw):
                    raise RuntimeError(f"expected idx [B,{c2_ch},{gh},{gw}], got {tuple(idx.shape)}")
                idx_cpu = idx.detach().cpu().long().permute(1, 2, 3, 0).reshape(pos_count, -1)
                bsz = int(idx_cpu.shape[1])
                offsets = torch.arange(pos_count, dtype=torch.long).repeat_interleave(bsz) * k
                linear = offsets + idx_cpu.reshape(-1)
                counts = torch.bincount(linear, minlength=pos_count * k).reshape(pos_count, k)
                hist.view(pos_count, k).add_(counts)
                total_images += int(imgs.shape[0])
                total_batches += 1
                epoch_images += int(imgs.shape[0])
                epoch_batches += 1
                if max_batches > 0 and batch_idx >= max_batches:
                    break
            print(
                f"collected stat_epoch={epoch:02d}/{int(stat_epochs):02d} "
                f"epoch_batches={epoch_batches} epoch_images={epoch_images} total_images={total_images}",
                flush=True,
            )
    return hist, total_images, total_batches


def coverage_summary(hist: torch.Tensor, m_values: list[int]) -> tuple[dict, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    total = hist.sum().item()
    if total <= 0:
        raise RuntimeError("empty histogram")
    out = {}
    candidates = {}
    candidate_counts = {}
    cell_total = hist.sum(dim=-1).clamp_min(1)
    for m in m_values:
        top = hist.topk(min(int(m), hist.shape[-1]), dim=-1)
        hit = top.values.sum(dim=-1)
        per_cell = hit.float() / cell_total.float()
        coverage = float(hit.sum().item() / total)
        out[str(m)] = {
            "coverage": coverage,
            "per_cell_mean": float(per_cell.mean().item()),
            "per_cell_min": float(per_cell.min().item()),
            "per_cell_max": float(per_cell.max().item()),
            "per_cell_std": float(per_cell.std(unbiased=False).item()),
        }
        candidates[int(m)] = top.indices.cpu()
        candidate_counts[int(m)] = top.values.cpu()
    return out, candidates, candidate_counts


def write_coverage_csv(path: Path, per_cell: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["c2_channel", "patch_u", "patch_v", "coverage"])
        c2_ch, gh, gw = per_cell.shape
        for j in range(c2_ch):
            for u in range(gh):
                for v in range(gw):
                    writer.writerow([j, u, v, float(per_cell[j, u, v].item())])


def interpret(summary: dict) -> str:
    cov64 = summary.get("64", {}).get("coverage", 0.0)
    cov128 = summary.get("128", {}).get("coverage", 0.0)
    cov256 = summary.get("256", {}).get("coverage", 0.0)
    if cov64 >= 0.70:
        return "M=64 coverage >= 70%: channel/position statistics are very strong."
    if cov128 >= 0.80:
        return "M=128 coverage >= 80%: restricted predictor is a promising main path."
    if cov256 < 0.60:
        return "M=256 coverage < 60%: channel/position alone is weak; add C1-cluster conditioning."
    return "Intermediate regime: channel/position prior is useful, but likely needs stronger conditioning or predictor evidence."


def main() -> None:
    cli = parse_args()
    stage2_ckpt = resolve_path(cli.stage2_ckpt)
    ckpt = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    args = checkpoint_args(ckpt, cli)
    m_values = parse_m_values(cli.m_values)
    out_dir = Path(resolve_path(cli.out_dir)) if cli.out_dir else default_out_dir(stage2_ckpt, cli.split, int(cli.stat_epochs))
    out_dir.mkdir(parents=True, exist_ok=True)
    if cli.log_file:
        setup_log_file(cli.log_file)
    seed_everything(int(args.seed))

    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    base_loader = train_loader if cli.split == "train" else val_loader
    loader = make_stats_loader(base_loader, cfg, args, cli.split)
    encoder, quantizer = load_stage2(args, ckpt, cfg.device)

    print("=== Patch-VQ position prior statistics ===")
    print(f"stage2_ckpt={stage2_ckpt}")
    print(f"split={cli.split} dataset_size={len(loader.dataset)} stat_epochs={int(cli.stat_epochs)} max_batches={int(cli.max_batches)}")
    print(f"K={int(args.k)} C2={int(args.latent_ch) - int(args.c1_ch)} patch_size={int(args.patch_size)} M={m_values}")
    print(f"out_dir={out_dir}")

    hist, total_images, total_batches = accumulate_hist(
        loader,
        encoder,
        quantizer,
        args,
        stat_epochs=int(cli.stat_epochs),
        max_batches=int(cli.max_batches),
    )
    summary, candidates, candidate_counts = coverage_summary(hist, m_values)
    cell_total = hist.sum(dim=-1).clamp_min(1)

    payload = {
        "stage2_ckpt": stage2_ckpt,
        "split": cli.split,
        "stat_epochs": int(cli.stat_epochs),
        "total_images": total_images,
        "total_batches": total_batches,
        "hist_shape": list(hist.shape),
        "m_values": m_values,
        "coverage": summary,
        "interpretation": interpret(summary),
        "args": vars(args),
    }
    with open(out_dir / "coverage_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    for m in m_values:
        hit = candidate_counts[m].sum(dim=-1)
        per_cell = hit.float() / cell_total.float()
        write_coverage_csv(out_dir / f"coverage_by_position_M{m}.csv", per_cell)
        if bool(cli.save_candidates):
            torch.save(
                {
                    "candidate": candidates[m],
                    "candidate_count": candidate_counts[m],
                    "coverage_by_position": per_cell.cpu(),
                    "M": int(m),
                    "stage2_ckpt": stage2_ckpt,
                },
                out_dir / f"candidate_top{m}.pt",
            )
    if bool(cli.save_hist):
        torch.save({"hist": hist, "stage2_ckpt": stage2_ckpt, "args": vars(args)}, out_dir / "hist_train.pt")

    print("coverage:")
    for m in m_values:
        item = summary[str(m)]
        print(
            f"  M={m}: coverage={item['coverage']:.4%} "
            f"per_cell_mean={item['per_cell_mean']:.4%} "
            f"min={item['per_cell_min']:.4%} max={item['per_cell_max']:.4%}"
        )
    print(f"judgement: {payload['interpretation']}")
    print(f"saved: {out_dir / 'coverage_summary.json'}")


if __name__ == "__main__":
    main()
