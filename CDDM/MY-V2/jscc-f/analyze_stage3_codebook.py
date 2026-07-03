from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
CDDM_ROOT = THIS_DIR.parents[1]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from Autoencoder.data.datasets import get_loader

from model import build_layer3, layer1_forward


def load_local_io():
    spec = importlib.util.spec_from_file_location("jsccf_analyze_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = load_local_io()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return CDDM_ROOT / p


def as_args(saved: dict[str, Any], cli: argparse.Namespace) -> SimpleNamespace:
    data = dict(saved)
    data.setdefault("stage", "stage3")
    data.setdefault("variant", "combiner")
    data.setdefault("lambda_u2", 0.0)
    data["cpu"] = bool(cli.cpu)
    if cli.data_dir:
        data["data_dir"] = str(resolve(cli.data_dir))
    for key in ("batch_size", "test_batch", "num_workers", "val_num_workers"):
        value = getattr(cli, key)
        if value is not None:
            data[key] = int(value)
    return SimpleNamespace(**data)


def load_state(module: torch.nn.Module, state: dict[str, torch.Tensor], name: str) -> None:
    missing, unexpected = module.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"{name} load mismatch: missing={missing} unexpected={unexpected}")


def flatten_codebook(codebook: torch.Tensor | None) -> torch.Tensor | None:
    if codebook is None or not isinstance(codebook, torch.Tensor) or codebook.ndim < 2:
        return None
    return codebook.detach().float().reshape(int(codebook.shape[0]), -1).cpu()


@torch.no_grad()
def effective_codebook(quantizer: torch.nn.Module) -> torch.Tensor | None:
    if hasattr(quantizer, "effective_codebook"):
        return flatten_codebook(quantizer.effective_codebook())
    if hasattr(quantizer, "codebook"):
        return flatten_codebook(quantizer.codebook)
    if hasattr(quantizer, "embedding") and hasattr(quantizer.embedding, "weight"):
        return flatten_codebook(quantizer.embedding.weight)
    return None


def raw_codebook(quantizer: torch.nn.Module) -> torch.Tensor | None:
    if hasattr(quantizer, "embedding") and hasattr(quantizer.embedding, "weight"):
        return flatten_codebook(quantizer.embedding.weight)
    if hasattr(quantizer, "codebook"):
        return flatten_codebook(quantizer.codebook)
    return None


def quantiles(values: torch.Tensor, prefix: str) -> dict[str, float]:
    if values.numel() == 0:
        return {}
    qs = torch.quantile(values.float(), torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99]))
    return {
        f"{prefix}_p01": float(qs[0].item()),
        f"{prefix}_p05": float(qs[1].item()),
        f"{prefix}_p50": float(qs[2].item()),
        f"{prefix}_p95": float(qs[3].item()),
        f"{prefix}_p99": float(qs[4].item()),
    }


def maybe_sample(x: torch.Tensor, sample: int, seed: int) -> torch.Tensor:
    n = int(x.shape[0])
    if sample <= 0 or sample >= n:
        return x
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    idx = torch.randperm(n, generator=gen)[:sample]
    return x.index_select(0, idx)


def matrix_metrics(codebook: torch.Tensor | None, prefix: str, pair_sample: int, seed: int, eps: float) -> dict[str, float]:
    if codebook is None or int(codebook.shape[0]) == 0:
        return {}
    x = codebook.float()
    n = int(x.shape[0])
    d = int(x.shape[1])
    norms = x.norm(p=2, dim=1)
    out = {
        f"{prefix}_codes": float(n),
        f"{prefix}_dim": float(d),
        f"{prefix}_norm_mean": float(norms.mean().item()),
        f"{prefix}_norm_std": float(norms.std(unbiased=False).item()),
        f"{prefix}_norm_min": float(norms.min().item()),
        f"{prefix}_norm_max": float(norms.max().item()),
    }
    out.update(quantiles(norms, f"{prefix}_norm"))
    if n < 2:
        return out

    sampled = maybe_sample(x, pair_sample, seed)
    pair = torch.pdist(sampled, p=2)
    out.update(
        {
            f"{prefix}_pair_sample": float(int(sampled.shape[0])),
            f"{prefix}_pair_mean": float(pair.mean().item()),
            f"{prefix}_pair_std": float(pair.std(unbiased=False).item()),
            f"{prefix}_pair_min": float(pair.min().item()),
            f"{prefix}_pair_max": float(pair.max().item()),
            f"{prefix}_duplicate_pairs_le_eps": float((pair <= float(eps)).sum().item()),
        }
    )
    out.update(quantiles(pair, f"{prefix}_pair"))

    dist = torch.cdist(sampled, sampled, p=2)
    dist.fill_diagonal_(float("inf"))
    nn_l2 = dist.min(dim=1).values
    out.update(
        {
            f"{prefix}_nn_l2_mean": float(nn_l2.mean().item()),
            f"{prefix}_nn_l2_std": float(nn_l2.std(unbiased=False).item()),
            f"{prefix}_nn_l2_min": float(nn_l2.min().item()),
            f"{prefix}_nn_l2_max": float(nn_l2.max().item()),
        }
    )
    out.update(quantiles(nn_l2, f"{prefix}_nn_l2"))

    unit = sampled / sampled.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    cosine = unit @ unit.t()
    mask = torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)
    offdiag = cosine[mask]
    out.update(
        {
            f"{prefix}_cos_mean": float(offdiag.mean().item()),
            f"{prefix}_cos_std": float(offdiag.std(unbiased=False).item()),
            f"{prefix}_cos_min": float(offdiag.min().item()),
            f"{prefix}_cos_max": float(offdiag.max().item()),
            f"{prefix}_cos_abs_mean": float(offdiag.abs().mean().item()),
            f"{prefix}_cos_abs_max": float(offdiag.abs().max().item()),
        }
    )
    return out


def gini_from_counts(counts: torch.Tensor) -> float:
    x = counts.double().flatten()
    total = x.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    sorted_x, _ = torch.sort(x)
    n = sorted_x.numel()
    rank = torch.arange(1, n + 1, dtype=torch.float64)
    return float((2.0 * (rank * sorted_x).sum() / (n * total) - (n + 1.0) / n).item())


def usage_metrics(hist: torch.Tensor, prefix: str = "usage") -> dict[str, float]:
    total = hist.double().sum().clamp_min(1.0)
    prob = hist.double() / total
    nonzero = prob > 0
    k = int(hist.numel())
    entropy_nats = float((-(prob[nonzero] * prob[nonzero].log()).sum()).item()) if bool(nonzero.any()) else 0.0
    entropy_bits = entropy_nats / math.log(2.0)
    perplexity = math.exp(entropy_nats)
    top = torch.sort(prob, descending=True).values
    out = {
        f"{prefix}_total_tokens": float(hist.sum().item()),
        f"{prefix}_used_codes": float(nonzero.sum().item()),
        f"{prefix}_dead_codes": float((~nonzero).sum().item()),
        f"{prefix}_active_ratio": float(nonzero.float().mean().item()),
        f"{prefix}_entropy_nats": entropy_nats,
        f"{prefix}_entropy_bits": entropy_bits,
        f"{prefix}_entropy_ratio": entropy_nats / math.log(max(2, k)),
        f"{prefix}_kl_to_uniform_nats": math.log(max(1, k)) - entropy_nats,
        f"{prefix}_perplexity": perplexity,
        f"{prefix}_perplexity_ratio": perplexity / max(1, k),
        f"{prefix}_top1_share": float(top[:1].sum().item()),
        f"{prefix}_top5_share": float(top[: min(5, k)].sum().item()),
        f"{prefix}_top10_share": float(top[: min(10, k)].sum().item()),
        f"{prefix}_gini": gini_from_counts(hist),
        f"{prefix}_count_mean": float(hist.double().mean().item()),
        f"{prefix}_count_std": float(hist.double().std(unbiased=False).item()),
        f"{prefix}_count_min": float(hist.min().item()),
        f"{prefix}_count_max": float(hist.max().item()),
    }
    out.update(quantiles(prob.float(), f"{prefix}_prob"))
    if bool(nonzero.any()):
        active_prob = prob[nonzero].float()
        out.update(
            {
                f"{prefix}_active_prob_min": float(active_prob.min().item()),
                f"{prefix}_active_prob_mean": float(active_prob.mean().item()),
                f"{prefix}_active_prob_max": float(active_prob.max().item()),
            }
        )
    return out


@torch.no_grad()
def collect_assignments(loader, e1, d1, e2, quantizer, args: SimpleNamespace, max_batches: int, progress_every: int) -> dict[str, float]:
    for module in (e1, d1, e2, quantizer):
        module.eval()
    device = next(e2.parameters()).device
    num_codes = int(getattr(args, "simvq_k", getattr(args, "vq_k", 0)))
    hist = torch.zeros(num_codes, dtype=torch.float64)
    quant_sqerr = 0.0
    z_sq = 0.0
    q_sq = 0.0
    element_count = 0
    token_count = 0
    assigned_l2: list[torch.Tensor] = []

    for batch_idx, (imgs, _labels) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        base = layer1_forward(e1, d1, imgs)
        z2, _ = e2(torch.cat([imgs, base["x1"]], dim=1))
        _q2_st, q2, idx, _stats = quantizer(z2)
        flat_idx = idx.detach().reshape(-1).cpu()
        hist += torch.bincount(flat_idx, minlength=num_codes).double()

        diff = q2.detach().float() - z2.detach().float()
        quant_sqerr += float(diff.square().sum().item())
        z_sq += float(z2.detach().float().square().sum().item())
        q_sq += float(q2.detach().float().square().sum().item())
        element_count += int(diff.numel())
        token_count += int(flat_idx.numel())
        vec_l2 = diff.permute(0, 2, 3, 1).reshape(-1, int(diff.shape[1])).norm(p=2, dim=1).cpu()
        assigned_l2.append(vec_l2)

        if progress_every > 0 and batch_idx % progress_every == 0:
            print(f"[assign] batches={batch_idx} tokens={token_count}", flush=True)
        if max_batches > 0 and batch_idx >= max_batches:
            break

    metrics = usage_metrics(hist)
    l2 = torch.cat(assigned_l2) if assigned_l2 else torch.empty(0)
    quant_mse = quant_sqerr / max(1, element_count)
    z_energy = z_sq / max(1, element_count)
    q_energy = q_sq / max(1, element_count)
    metrics.update(
        {
            "assign_l2_mean": float(l2.mean().item()) if l2.numel() else 0.0,
            "assign_l2_std": float(l2.std(unbiased=False).item()) if l2.numel() else 0.0,
            "assign_l2_min": float(l2.min().item()) if l2.numel() else 0.0,
            "assign_l2_max": float(l2.max().item()) if l2.numel() else 0.0,
            "quant_mse": float(quant_mse),
            "z_energy": float(z_energy),
            "q_energy": float(q_energy),
            "relative_quant_mse": float(quant_mse / max(z_energy, 1e-12)),
            "q_over_z_energy": float(q_energy / max(z_energy, 1e-12)),
        }
    )
    metrics.update(quantiles(l2, "assign_l2"))
    return metrics


def projection_metrics(quantizer: torch.nn.Module, eff: torch.Tensor | None, raw: torch.Tensor | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if eff is not None and raw is not None and tuple(eff.shape) == tuple(raw.shape):
        delta = (eff - raw).norm(p=2, dim=1)
        out.update(
            {
                "proj_delta_l2_mean": float(delta.mean().item()),
                "proj_delta_l2_std": float(delta.std(unbiased=False).item()),
                "proj_delta_l2_min": float(delta.min().item()),
                "proj_delta_l2_max": float(delta.max().item()),
            }
        )
        out.update(quantiles(delta, "proj_delta_l2"))
        raw_unit = raw / raw.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        eff_unit = eff / eff.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        same_cos = (raw_unit * eff_unit).sum(dim=1)
        out.update(
            {
                "raw_effective_cos_mean": float(same_cos.mean().item()),
                "raw_effective_cos_min": float(same_cos.min().item()),
                "raw_effective_cos_max": float(same_cos.max().item()),
            }
        )
    if hasattr(quantizer, "embedding_proj"):
        proj = quantizer.embedding_proj
        weight = proj.weight.detach().float().cpu()
        eye = torch.eye(weight.shape[0], weight.shape[1])
        out["proj_weight_minus_eye_fro"] = float((weight - eye).norm(p="fro").item())
        out["proj_weight_fro"] = float(weight.norm(p="fro").item())
        if proj.bias is not None:
            out["proj_bias_l2"] = float(proj.bias.detach().float().cpu().norm(p=2).item())
    return out


def analyze_checkpoint(path: Path, cli: argparse.Namespace) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = as_args(ckpt.get("args", {}), cli)
    device = torch.device("cpu" if bool(args.cpu) or not torch.cuda.is_available() else "cuda")
    e1, d1, e2, d2, combiner, quantizer, indexnet = build_layer3(args, device)
    load_state(e1, ckpt["e1_state_dict"], "E1")
    load_state(d1, ckpt["d1_state_dict"], "D1")
    load_state(e2, ckpt["e2_state_dict"], "E2")
    if "d2_state_dict" in ckpt:
        load_state(d2, ckpt["d2_state_dict"], "D2")
    if "combiner_state_dict" in ckpt:
        load_state(combiner, ckpt["combiner_state_dict"], "combiner")
    q_state = ckpt.get("quantizer_state_dict", ckpt.get("simvq_state_dict"))
    if q_state is None:
        raise KeyError(f"{path} has no quantizer_state_dict/simvq_state_dict")
    load_state(quantizer, q_state, "quantizer")
    if "indexnet_state_dict" in ckpt:
        load_state(indexnet, ckpt["indexnet_state_dict"], "indexnet")

    eff = effective_codebook(quantizer)
    raw = raw_codebook(quantizer)
    metrics: dict[str, Any] = {
        "checkpoint": str(path),
        "checkpoint_name": path.name,
        "epoch": int(ckpt.get("epoch", -1)),
        "stage": str(ckpt.get("stage", "")),
        "version": str(ckpt.get("version", getattr(args, "version", ""))),
        "quantizer_type": str(getattr(args, "quantizer", "")),
        "num_codes": int(eff.shape[0]) if eff is not None else int(getattr(args, "simvq_k", 0)),
        "embedding_dim": int(eff.shape[1]) if eff is not None else 0,
        "split": str(cli.split),
        "max_batches": int(cli.max_batches),
    }
    metrics.update(matrix_metrics(eff, "effective", int(cli.pair_sample), int(cli.seed), float(cli.duplicate_eps)))
    metrics.update(matrix_metrics(raw, "raw", int(cli.pair_sample), int(cli.seed) + 17, float(cli.duplicate_eps)))
    metrics.update(projection_metrics(quantizer, eff, raw))

    if not cli.codebook_only:
        cfg = jsccf_io.build_config(args, encoder_in_chans=3)
        train_loader, val_loader = get_loader(cfg)
        loader = train_loader if cli.split == "train" else val_loader
        metrics["dataset_size"] = int(len(loader.dataset))
        metrics.update(collect_assignments(loader, e1, d1, e2, quantizer, args, int(cli.max_batches), int(cli.progress_every)))
    return metrics


def write_outputs(rows: list[dict[str, Any]], output: Path, csv_output: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"wrote JSON: {output}", flush=True)
    if csv_output is None:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote CSV: {csv_output}", flush=True)


def print_summary(rows: list[dict[str, Any]]) -> None:
    cols = [
        "checkpoint_name",
        "epoch",
        "num_codes",
        "usage_perplexity",
        "usage_perplexity_ratio",
        "usage_used_codes",
        "usage_gini",
        "effective_pair_mean",
        "effective_nn_l2_mean",
        "effective_nn_l2_min",
        "proj_delta_l2_mean",
        "quant_mse",
        "relative_quant_mse",
    ]
    print("\t".join(cols), flush=True)
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        print("\t".join(vals), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", action="append", required=True, help="Stage3 checkpoint path. Repeat for multiple checkpoints.")
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--output", type=str, default="MY-V2/jscc-f/checkpoints/stage3_simvq_codebook_metrics.json")
    p.add_argument("--csv-output", type=str, default="MY-V2/jscc-f/checkpoints/stage3_simvq_codebook_metrics.csv")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--test-batch", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--val-num-workers", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--max-batches", type=int, default=0, help="0 means full split.")
    p.add_argument("--pair-sample", type=int, default=0, help="0 means all codewords.")
    p.add_argument("--duplicate-eps", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--codebook-only", action="store_true")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    rows = []
    for raw_path in cli.ckpt:
        path = resolve(raw_path)
        print(f"[analyze] {path}", flush=True)
        rows.append(analyze_checkpoint(path, cli))
    write_outputs(rows, resolve(cli.output), resolve(cli.csv_output) if cli.csv_output else None)
    print_summary(rows)


if __name__ == "__main__":
    main()
