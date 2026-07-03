# 作用：对 Stage1 连续 C2 的通道级 16x16 map 做 PCA，比较 PCA 前缀和原始通道前缀恢复 C2 后的解码 PSNR。
# 输出：summary.json、pca_channel_prefix.csv、raw_channel_prefix.csv、pca_channel_components.csv、channel_pca.pt。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/c2_channel_pca_prefix_probe.py --fit-epochs 1 --max-fit-images 16 --max-eval-images 8 --val-num-workers 0

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

from common import check_args, psnr_per_image, real_awgn, resolve_path, seed_everything  # noqa: E402
from shared import build_encoder_decoder, cvq_io, get_loader, split_c1_c2  # noqa: E402


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
        obj = cvq_io.load_experiment_checkpoint(args.init_stage1_ckpt, encoder=encoder, decoder=decoder, strict=True)
        print(f"loaded stage1 checkpoint: {resolve_path(args.init_stage1_ckpt)}")
        if "metrics" in obj:
            print(f"stage1 metrics keys: {sorted(obj['metrics'].keys())}")
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


def fit_channel_pca(
    c2_samples: torch.Tensor,
    *,
    normalize_channel_rms: bool,
    max_fit_tokens: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    if c2_samples.ndim != 4:
        raise ValueError(f"expected C2 samples [N,C,H,W], got {tuple(c2_samples.shape)}")
    n, c, h, w = c2_samples.shape
    tokens = c2_samples.permute(0, 2, 3, 1).reshape(n * h * w, c).float()
    if int(max_fit_tokens) > 0 and tokens.shape[0] > int(max_fit_tokens):
        g = torch.Generator()
        g.manual_seed(int(seed))
        perm = torch.randperm(tokens.shape[0], generator=g)[: int(max_fit_tokens)]
        tokens = tokens[perm].contiguous()
    channel_rms = tokens.square().mean(dim=0).sqrt().clamp_min(1e-8)
    scale = channel_rms if bool(normalize_channel_rms) else torch.ones_like(channel_rms)
    x = tokens / scale.view(1, c)
    mean = x.mean(dim=0)
    centered = x - mean.view(1, c)
    cov = centered.t().matmul(centered) / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0.0).contiguous()
    eigvecs = eigvecs[:, order].contiguous()
    explained = eigvals / eigvals.sum().clamp_min(1e-12)
    print(
        "channel PCA fitted "
        f"tokens={tokens.shape[0]} C={c} normalize_channel_rms={bool(normalize_channel_rms)} "
        f"top1_explained={float(explained[0].item()):.6g} "
        f"top5_explained={float(explained[:5].sum().item()):.6g} "
        f"top10_explained={float(explained[:10].sum().item()):.6g}"
    )
    return {
        "mean": mean,
        "components": eigvecs,
        "eigvals": eigvals,
        "explained": explained,
        "scale": scale,
        "channel_rms": channel_rms,
    }


def reconstruct_channel_pca_prefix(c2: torch.Tensor, pca: dict[str, torch.Tensor], k: int) -> torch.Tensor:
    bsz, c, h, w = c2.shape
    mean = pca["mean"].to(device=c2.device, dtype=c2.dtype)
    comp = pca["components"].to(device=c2.device, dtype=c2.dtype)
    scale = pca["scale"].to(device=c2.device, dtype=c2.dtype).clamp_min(1e-8)
    x = c2.permute(0, 2, 3, 1).reshape(-1, c) / scale.view(1, c)
    centered = x - mean.view(1, c)
    if int(k) <= 0:
        rec = mean.view(1, c).expand_as(x)
    else:
        basis = comp[:, : int(k)]
        coeff = centered.matmul(basis)
        rec = coeff.matmul(basis.t()) + mean.view(1, c)
    rec = rec * scale.view(1, c)
    return rec.reshape(bsz, h, w, c).permute(0, 3, 1, 2).contiguous()


def raw_channel_prefix(c2: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.zeros_like(c2)
    if int(k) > 0:
        out[:, : int(k)] = c2[:, : int(k)]
    return out


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
) -> tuple[dict[str, float], dict[str, SumMeter], dict[str, SumMeter], int]:
    device = next(encoder.parameters()).device
    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    base = {name: SumMeter() for name in ["psnr_c1_zero", "psnr_pca_q0", "psnr_full_c2"]}
    pca_meters = {f"q{k}": SumMeter() for k in range(1, c2_ch + 1)}
    raw_meters = {f"q{k}": SumMeter() for k in range(1, c2_ch + 1)}
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

        candidates: list[tuple[str, torch.Tensor]] = [("c1_zero", zero_c2), ("pca_q0", reconstruct_channel_pca_prefix(c2, pca, 0)), ("full", c2)]
        for k in range(1, c2_ch + 1):
            candidates.append((f"pca_q{k}", reconstruct_channel_pca_prefix(c2, pca, k)))
        for k in range(1, c2_ch + 1):
            candidates.append((f"raw_q{k}", raw_channel_prefix(c2, k)))

        vals = decode_candidate_psnr(decoder, imgs, c1_in, candidates, candidates_per_forward=int(args.decode_candidates_per_forward))
        base["psnr_c1_zero"].update_tensor(vals["c1_zero"])
        base["psnr_pca_q0"].update_tensor(vals["pca_q0"])
        base["psnr_full_c2"].update_tensor(vals["full"])
        for k in range(1, c2_ch + 1):
            pca_meters[f"q{k}"].update_tensor(vals[f"pca_q{k}"])
            raw_meters[f"q{k}"].update_tensor(vals[f"raw_q{k}"])
        total_images += int(imgs.shape[0])
        print(f"eval images={total_images}", flush=True)
    summary = {key: meter.avg for key, meter in base.items()}
    return summary, pca_meters, raw_meters, total_images


def component_rows(pca: dict[str, torch.Tensor]) -> list[dict]:
    comp = pca["components"].float().cpu()
    eigvals = pca["eigvals"].float().cpu()
    explained = pca["explained"].float().cpu()
    cum = explained.cumsum(0)
    rows = []
    for i in range(comp.shape[1]):
        vec = comp[:, i]
        abs_vec = vec.abs()
        weights = vec.square()
        participation = float(1.0 / weights.square().sum().clamp_min(1e-12).item())
        top_idx = int(abs_vec.argmax().item())
        order = torch.argsort(abs_vec, descending=True)
        rows.append(
            {
                "component": i + 1,
                "eigval": float(eigvals[i].item()),
                "explained": float(explained[i].item()),
                "cum_explained": float(cum[i].item()),
                "participation_ratio": participation,
                "top_channel": top_idx,
                "top_abs_loading": float(abs_vec[top_idx].item()),
                "top3_channels": " ".join(str(int(v.item())) for v in order[:3]),
                "top3_abs_loadings": " ".join(f"{float(abs_vec[v].item()):.6g}" for v in order[:3]),
            }
        )
    return rows


def prefix_rows(summary: dict[str, float], meters: dict[str, SumMeter], pca: dict[str, torch.Tensor], mode: str) -> list[dict]:
    full = float(summary["psnr_full_c2"])
    c1 = float(summary["psnr_c1_zero"])
    explained = pca["explained"].float().cpu()
    rows = []
    if mode == "pca":
        rows.append(
            {
                "q": 0,
                "cum_explained": 0.0,
                "psnr": float(summary["psnr_pca_q0"]),
                "gain_over_c1": float(summary["psnr_pca_q0"]) - c1,
                "gap_to_full": full - float(summary["psnr_pca_q0"]),
            }
        )
    for k in range(1, len(meters) + 1):
        psnr = meters[f"q{k}"].avg
        rows.append(
            {
                "q": k,
                "cum_explained": float(explained[:k].sum().item()) if mode == "pca" else "",
                "psnr": psnr,
                "gain_over_c1": psnr - c1,
                "gap_to_full": full - psnr,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--out-dir", type=str, default="MY/train_cvq-v2/eval/outputs/cvq_v2_c36_snr9_channel_pca_prefix_probe")
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
    p.add_argument("--max-fit-tokens", type=int, default=0)
    p.add_argument("--max-eval-images", type=int, default=0)
    p.add_argument("--c1-mode", type=str, choices=["rx", "clean"], default="rx")
    p.add_argument("--pca-normalize-channel-rms", action="store_true")
    p.add_argument("--decode-candidates-per-forward", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    check_args(args)
    seed_everything(int(args.seed))
    out_dir = Path(resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder = build_encoder_decoder(args, cfg.device)
    load_stage1(args, encoder, decoder)
    encoder.eval()
    decoder.eval()

    print("=== C2 channel PCA prefix probe ===")
    print(f"device={cfg.device} c1_mode={args.c1_mode} out_dir={out_dir}")
    print("experiment: keep C1 channels 0..15 unchanged by PCA; PCA is fitted over C2 channel vectors [20] at each spatial location.")
    fit_c2 = collect_c2_samples(train_loader, encoder, args, epochs=int(args.fit_epochs), max_images=int(args.max_fit_images))
    pca = fit_channel_pca(
        fit_c2,
        normalize_channel_rms=bool(args.pca_normalize_channel_rms),
        max_fit_tokens=int(args.max_fit_tokens),
        seed=int(args.seed),
    )
    torch.save({key: value.cpu() for key, value in pca.items()}, out_dir / "channel_pca.pt")

    summary, pca_meters, raw_meters, eval_images = evaluate_prefixes(val_loader, encoder, decoder, pca, args)
    summary["eval_images"] = int(eval_images)
    summary["fit_images"] = int(fit_c2.shape[0])
    summary["fit_epochs"] = int(args.fit_epochs)
    summary["c1_mode"] = str(args.c1_mode)
    summary["pca_normalize_channel_rms"] = bool(args.pca_normalize_channel_rms)
    summary["pca_q20_minus_full"] = pca_meters["q20"].avg - float(summary["psnr_full_c2"])
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "pca_channel_components.csv", component_rows(pca))
    write_csv(out_dir / "pca_channel_prefix.csv", prefix_rows(summary, pca_meters, pca, "pca"))
    write_csv(out_dir / "raw_channel_prefix.csv", prefix_rows(summary, raw_meters, pca, "raw"))

    print(
        "summary "
        f"psnr_c1_zero={summary['psnr_c1_zero']:.6g} "
        f"psnr_pca_q1={pca_meters['q1'].avg:.6g} "
        f"psnr_pca_q5={pca_meters['q5'].avg:.6g} "
        f"psnr_pca_q10={pca_meters['q10'].avg:.6g} "
        f"psnr_pca_q20={pca_meters['q20'].avg:.6g} "
        f"psnr_full_c2={summary['psnr_full_c2']:.6g}"
    )
    print(f"wrote {out_dir / 'pca_channel_prefix.csv'}")
    print(f"wrote {out_dir / 'raw_channel_prefix.csv'}")
    print(f"wrote {out_dir / 'pca_channel_components.csv'}")


if __name__ == "__main__":
    main()
