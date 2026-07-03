from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent
CDDM_ROOT = THIS_DIR.parents[1]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from Autoencoder.data.datasets import get_loader

from common import (
    averaged,
    batch_metric_mean,
    check_jsccf_args,
    meters,
    mse_per_image,
    format_metrics,
    print_epoch,
    print_run_header,
    psnr_per_image,
    recon_loss,
    resolve_path,
    seed_everything,
    setup_log_file,
    should_save_latest,
    should_validate,
    ssim_per_image,
    write_json,
    z2_ch,
)
from model import build_layer2, build_layer3, layer1_forward, layer2_forward, layer3_forward, parse_fsq_levels


def load_local_io():
    spec = importlib.util.spec_from_file_location("jsccf_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = load_local_io()


METRIC_NAMES = [
    "loss",
    "loss_rec",
    "loss_vq",
    "loss_codebook",
    "loss_commit",
    "loss_index",
    "loss_u2_teacher",
    "lambda_index_eff",
    "lambda_u2_teacher_eff",
    "mse_x1",
    "psnr_x1",
    "ssim_x1",
    "mse_oracle",
    "psnr_oracle",
    "ssim_oracle",
    "delta_oracle",
    "vq_mse",
    "index_top1",
    "index_top5",
    "index_entropy",
]


DISPLAY_METRICS = [
    "loss",
    "loss_rec",
    "loss_vq",
    "loss_codebook",
    "loss_commit",
    "loss_index",
    "loss_u2_teacher",
    "lambda_index_eff",
    "lambda_u2_teacher_eff",
    "mse_x1",
    "psnr_x1",
    "ssim_x1",
    "mse_oracle",
    "psnr_oracle",
    "ssim_oracle",
    "delta_oracle",
    "vq_mse",
    "index_top1",
    "index_top5",
    "index_entropy",
    "used_codes",
    "fsq_levels_mean",
    "fsq_scale_mean",
]

def display_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {k: metrics[k] for k in DISPLAY_METRICS if k in metrics}

def stage3_name(args: argparse.Namespace) -> str:
    quantizer = str(getattr(args, "quantizer", "simvq"))
    if quantizer == "vq":
        return f"stage3_vq_k{int(args.vq_k)}_d{z2_ch(args)}_oracle_index"
    if quantizer == "fsq":
        levels = parse_fsq_levels(getattr(args, "fsq_levels", "7"), z2_ch(args))
        level_name = f"l{levels[0]}x{z2_ch(args)}" if len(set(levels)) == 1 else "l" + "x".join(str(v) for v in levels)
        return f"stage3_fsq_{level_name}_d{z2_ch(args)}_oracle_index"
    if quantizer == "cvq":
        return f"stage3_cvq_k{int(args.cvq_k)}_t{z2_ch(args)}_oracle_index"
    if quantizer == "fullmap_simvq":
        return f"stage3_fullmap_simvq_k{int(args.fullmap_simvq_k)}_t{z2_ch(args)}_oracle_index"
    return f"stage3_simvq_k{int(args.simvq_k)}_d{z2_ch(args)}_oracle_index"


def quantizer_num_codes(args: argparse.Namespace) -> int:
    quantizer = str(getattr(args, "quantizer", "simvq"))
    if quantizer == "vq":
        return int(args.vq_k)
    if quantizer == "fsq":
        return max(parse_fsq_levels(getattr(args, "fsq_levels", "7"), z2_ch(args)))
    if quantizer == "cvq":
        return int(args.cvq_k)
    if quantizer == "fullmap_simvq":
        return int(args.fullmap_simvq_k)
    return int(args.simvq_k)


def trainable_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def set_trainable(module: torch.nn.Module, trainable: bool = True) -> None:
    for param in module.parameters():
        param.requires_grad_(bool(trainable))


def u2_teacher_phase_enabled(args: argparse.Namespace, phase: str) -> bool:
    mode = str(getattr(args, "u2_teacher_phases", "finetune"))
    if mode == "none":
        return False
    if mode == "all":
        return True
    return mode == str(phase)


def build_optimizer(
    args: argparse.Namespace,
    e1: torch.nn.Module,
    d1: torch.nn.Module,
    e2: torch.nn.Module,
    d2: torch.nn.Module,
    combiner: torch.nn.Module,
    quantizer: torch.nn.Module,
    indexnet: torch.nn.Module,
) -> optim.Optimizer:
    param_groups: list[dict] = []
    q_params = trainable_params(quantizer)
    if q_params:
        param_groups.append({"params": q_params, "lr": float(args.lr_simvq)})
    codec_params = (
        trainable_params(e1)
        + trainable_params(d1)
        + trainable_params(e2)
        + trainable_params(d2)
        + trainable_params(combiner)
    )
    if codec_params:
        param_groups.append({"params": codec_params, "lr": float(args.lr)})
    if bool(getattr(args, "train_indexnet", False)):
        index_params = trainable_params(indexnet)
        if index_params:
            param_groups.append({"params": index_params, "lr": float(args.lr_indexnet)})
    if not param_groups:
        raise ValueError("no trainable parameters for Stage3")
    return optim.AdamW(param_groups, weight_decay=float(args.weight_decay))


def load_initial_weights(args, e1, d1, e2, d2, combiner) -> None:
    if args.layer2_ckpt:
        ckpt = jsccf_io.load_checkpoint(args.layer2_ckpt)
        jsccf_io.load_state(e1, ckpt["e1_state_dict"], "E1", strict=True)
        jsccf_io.load_state(d1, ckpt["d1_state_dict"], "D1", strict=True)
        jsccf_io.load_state(e2, ckpt["e2_state_dict"], "E2", strict=True)
        jsccf_io.load_state(d2, ckpt["d2_state_dict"], "D2", strict=True)
        jsccf_io.load_state(combiner, ckpt["combiner_state_dict"], "combiner", strict=True)
        return

    layer1_path = args.layer1_ckpt or args.init_ckpt
    if not layer1_path:
        raise ValueError("set --layer1-ckpt/--init-ckpt, or use --layer2-ckpt for Stage3 initialization")
    args.layer1_ckpt = layer1_path
    jsccf_io.load_layer1_compatible_checkpoint(layer1_path, e1, d1, strict=True)


def load_layer2_teacher(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module] | None:
    if float(getattr(args, "lambda_u2_teacher", 0.0)) <= 0.0:
        return None
    if not getattr(args, "layer2_ckpt", ""):
        raise ValueError("--lambda-u2-teacher requires --layer2-ckpt")
    e1_t, d1_t, e2_t, d2_t, combiner_t = build_layer2(args, device)
    ckpt = jsccf_io.load_checkpoint(args.layer2_ckpt)
    jsccf_io.load_state(e1_t, ckpt["e1_state_dict"], "teacher_E1", strict=True)
    jsccf_io.load_state(d1_t, ckpt["d1_state_dict"], "teacher_D1", strict=True)
    jsccf_io.load_state(e2_t, ckpt["e2_state_dict"], "teacher_E2", strict=True)
    jsccf_io.load_state(d2_t, ckpt["d2_state_dict"], "teacher_D2", strict=True)
    jsccf_io.load_state(combiner_t, ckpt["combiner_state_dict"], "teacher_combiner", strict=True)
    teacher = (e1_t, d1_t, e2_t, d2_t, combiner_t)
    for module in teacher:
        set_trainable(module, False)
        module.eval()
    print(f"[stage3 u2 teacher] loaded frozen Layer2 teacher from {resolve_path(args.layer2_ckpt)}", flush=True)
    return teacher


@torch.no_grad()
def forward_layer2_teacher(
    teacher: tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module] | None,
    imgs: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor] | None:
    if teacher is None:
        return None
    e1_t, d1_t, e2_t, d2_t, combiner_t = teacher
    return layer2_forward(e1_t, d1_t, e2_t, d2_t, combiner_t, imgs, variant=str(args.variant))


def layer3_forward_unfrozen(
    e1: torch.nn.Module,
    d1: torch.nn.Module,
    e2: torch.nn.Module,
    d2: torch.nn.Module,
    combiner: torch.nn.Module,
    quantizer: torch.nn.Module,
    indexnet: torch.nn.Module,
    img: torch.Tensor,
    *,
    train_index_path: bool,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    base = layer1_forward(e1, d1, img)
    z1 = base["z1"]
    x1 = base["x1"]
    e2_in = torch.cat([img, x1], dim=1)
    z2, _ = e2(e2_in)
    q2_st, q2, q2_index, vq_stats = quantizer(z2)
    u2_raw = d2(q2_st)
    u2 = u2_raw.clamp(0.0, 1.0)
    final_oracle = combiner(x1, u2)
    logits = indexnet(z1 if train_index_path else z1.detach())
    return {
        "z1": z1,
        "x1": x1,
        "z2": z2,
        "q2_st": q2_st,
        "q2": q2,
        "q2_index": q2_index,
        "vq_stats": vq_stats,
        "index_logits": logits,
        "u2_raw": u2_raw,
        "u2": u2,
        "final_oracle": final_oracle,
        "final": final_oracle,
    }




def update_code_hist(hist: torch.Tensor, q2_index: torch.Tensor) -> None:
    hist += torch.bincount(q2_index.detach().reshape(-1).cpu(), minlength=hist.numel()).float()


@torch.no_grad()
def flatten_codebook(codebook: torch.Tensor | None) -> torch.Tensor | None:
    if codebook is None or not isinstance(codebook, torch.Tensor) or codebook.ndim < 2:
        return None
    return codebook.detach().float().reshape(int(codebook.shape[0]), -1)


@torch.no_grad()
def codebook_effective_matrix(quantizer: torch.nn.Module) -> torch.Tensor | None:
    if hasattr(quantizer, "effective_codebook"):
        codebook = quantizer.effective_codebook()
    elif hasattr(quantizer, "codebook"):
        codebook = quantizer.codebook
    elif hasattr(quantizer, "embedding") and hasattr(quantizer.embedding, "weight"):
        codebook = quantizer.embedding.weight
    else:
        return None
    return flatten_codebook(codebook)


@torch.no_grad()
def codebook_raw_matrix(quantizer: torch.nn.Module) -> torch.Tensor | None:
    if hasattr(quantizer, "embedding") and hasattr(quantizer.embedding, "weight"):
        return flatten_codebook(quantizer.embedding.weight)
    if hasattr(quantizer, "codebook"):
        return flatten_codebook(quantizer.codebook)
    return None


@torch.no_grad()
def embedding_l2_summary(codebook: torch.Tensor | None, args: argparse.Namespace, prefix: str) -> dict[str, float]:
    if codebook is None or int(codebook.shape[0]) < 1:
        return {}

    norms = codebook.norm(p=2, dim=1)
    metrics = {
        f"{prefix}_norm_mean": float(norms.mean().item()),
        f"{prefix}_norm_std": float(norms.std(unbiased=False).item()),
        f"{prefix}_norm_min": float(norms.min().item()),
        f"{prefix}_norm_max": float(norms.max().item()),
    }

    sample_n = max(0, int(getattr(args, "codebook_dist_sample", 512)))
    num_codes = int(codebook.shape[0])
    if sample_n == 0 or num_codes < 2:
        return metrics

    sample_n = min(sample_n, num_codes)
    if sample_n < num_codes:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(getattr(args, "seed", 0)) + 104729)
        indices = torch.randperm(num_codes, generator=generator)[:sample_n].to(device=codebook.device)
        sampled = codebook.index_select(0, indices)
    else:
        sampled = codebook
    pair_l2 = torch.pdist(sampled, p=2)
    if pair_l2.numel() > 0:
        metrics.update(
            {
                f"{prefix}_pair_mean": float(pair_l2.mean().item()),
                # f"{prefix}_pair_std": float(pair_l2.std(unbiased=False).item()),
                # f"{prefix}_pair_min": float(pair_l2.min().item()),
                # f"{prefix}_pair_max": float(pair_l2.max().item()),
                # f"{prefix}_pair_sample": float(sample_n),
            }
        )
    return metrics


@torch.no_grad()
def codebook_l2_metrics(quantizer: torch.nn.Module, args: argparse.Namespace) -> dict[str, float]:
    effective = codebook_effective_matrix(quantizer)
    metrics = embedding_l2_summary(effective, args, "codebook_l2")

    if hasattr(quantizer, "embedding_proj"):
        raw = codebook_raw_matrix(quantizer)
        metrics.update(embedding_l2_summary(raw, args, "codebook_raw_l2"))
        if effective is not None and raw is not None and tuple(effective.shape) == tuple(raw.shape):
            delta = (effective - raw).norm(p=2, dim=1)
            metrics.update(
                {
                    "codebook_proj_delta_l2_mean": float(delta.mean().item()),
                    # "codebook_proj_delta_l2_std": float(delta.std(unbiased=False).item()),
                    # "codebook_proj_delta_l2_min": float(delta.min().item()),
                    # "codebook_proj_delta_l2_max": float(delta.max().item()),
                }
            )
    return metrics


def flatten_index_logits(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = target.long()
    if logits.ndim == 5 and target.ndim == 4 and tuple(logits.shape[:4]) == tuple(target.shape):
        return logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1)
    if logits.ndim == 4 and target.ndim == 3 and logits.shape[0] == target.shape[0] and tuple(logits.shape[2:]) == tuple(target.shape[1:]):
        return logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]).float(), target.reshape(-1)
    if logits.ndim == 3 and target.ndim == 2 and tuple(logits.shape[:2]) == tuple(target.shape):
        return logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1)
    raise ValueError(f"unsupported index logits/target shapes: logits={tuple(logits.shape)} target={tuple(target.shape)}")


def index_metrics(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    flat_logits, flat_target = flatten_index_logits(logits, target)
    pred = flat_logits.argmax(dim=1)
    top1 = (pred == flat_target).float().mean()
    topk = flat_logits.topk(min(5, int(flat_logits.shape[1])), dim=1).indices
    top5 = (topk == flat_target.unsqueeze(1)).any(dim=1).float().mean()
    log_prob = F.log_softmax(flat_logits.float(), dim=1)
    entropy = -(log_prob.exp() * log_prob).sum(dim=1).mean()
    return float(top1.item()), float(top5.item()), float(entropy.item())


def update_metrics(m: dict, hist: torch.Tensor, out: dict, imgs: torch.Tensor, losses: dict[str, torch.Tensor | float]) -> None:
    bsz = imgs.shape[0]
    for name in ["loss", "loss_rec", "loss_vq", "loss_codebook", "loss_commit", "loss_index", "loss_u2_teacher"]:
        value = losses[name]
        m[name].update(float(value.item()), bsz)
    m["lambda_index_eff"].update(float(losses["lambda_index_eff"]), bsz)
    m["lambda_u2_teacher_eff"].update(float(losses["lambda_u2_teacher_eff"]), bsz)

    x1 = out["x1"]
    final = out["final_oracle"]
    psnr_x1 = batch_metric_mean(psnr_per_image(x1, imgs))
    psnr_oracle = batch_metric_mean(psnr_per_image(final, imgs))
    m["mse_x1"].update(batch_metric_mean(mse_per_image(x1, imgs)), bsz)
    m["psnr_x1"].update(psnr_x1, bsz)
    m["ssim_x1"].update(batch_metric_mean(ssim_per_image(x1, imgs)), bsz)
    m["mse_oracle"].update(batch_metric_mean(mse_per_image(final, imgs)), bsz)
    m["psnr_oracle"].update(psnr_oracle, bsz)
    m["ssim_oracle"].update(batch_metric_mean(ssim_per_image(final, imgs)), bsz)
    m["delta_oracle"].update(psnr_oracle - psnr_x1, bsz)

    stats = out["vq_stats"]
    m["vq_mse"].update(float(stats["vq_mse"].item()), bsz)
    target = out["q2_index"].detach().long()
    top1, top5, entropy = index_metrics(out["index_logits"], target)
    m["index_top1"].update(top1, bsz)
    m["index_top5"].update(top5, bsz)
    m["index_entropy"].update(entropy, bsz)
    update_code_hist(hist, target)


def finalize_metrics(m: dict, hist: torch.Tensor) -> dict[str, float]:
    metrics = averaged(m)
    total = hist.sum().clamp_min(1.0)
    prob = hist / total
    used = hist > 0
    nonzero = prob > 0
    metrics["used_codes"] = float(used.sum().item())
    metrics["codebook_usage"] = float(used.float().mean().item())
    # metrics["dead_codes"] = float((~used).sum().item())
    metrics["perplexity"] = float(torch.exp(-(prob[nonzero] * prob[nonzero].log()).sum()).item()) if bool(nonzero.any()) else 0.0
    return metrics


def finalize_stage3_metrics(m: dict, hist: torch.Tensor, quantizer: torch.nn.Module, args: argparse.Namespace) -> dict[str, float]:
    metrics = finalize_metrics(m, hist)
    metrics.update(codebook_l2_metrics(quantizer, args))
    if hasattr(quantizer, "extra_metrics"):
        metrics.update(quantizer.extra_metrics())
    return metrics


@torch.no_grad()
def kmeans_vectors(samples: torch.Tensor, k: int, *, iters: int, chunk_size: int, device: torch.device, label: str) -> tuple[torch.Tensor, float]:
    if samples.ndim != 2:
        raise ValueError(f"{label} expects samples [N,D], got {tuple(samples.shape)}")
    if samples.shape[0] < 1:
        raise ValueError(f"{label} got empty samples")
    x = samples.detach().float().to(device=device)
    k = int(k)
    if x.shape[0] >= k:
        pick = torch.randperm(x.shape[0], device=device)[:k]
    else:
        pick = torch.randint(0, x.shape[0], (k,), device=device)
    centers = x[pick].clone()
    chunk = max(1, int(chunk_size))
    final_mse = 0.0
    print(f"[stage3 init] running {label}: samples={x.shape[0]} K={k} dim={x.shape[1]} iters={int(iters)} chunk={chunk}", flush=True)
    for step in range(1, max(1, int(iters)) + 1):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device, dtype=torch.float32)
        total_dist = torch.zeros((), device=device, dtype=torch.float32)
        for start in range(0, x.shape[0], chunk):
            xb = x[start : start + chunk]
            dist = xb.square().sum(dim=1, keepdim=True) + centers.square().sum(dim=1).view(1, -1) - 2.0 * xb @ centers.t()
            idx = dist.argmin(dim=1)
            sums.index_add_(0, idx, xb)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            total_dist += dist.gather(1, idx.view(-1, 1)).sum()
        empty = counts == 0
        if bool(empty.any()):
            repl = torch.randint(0, x.shape[0], (int(empty.sum().item()),), device=device)
            sums[empty] = x[repl]
            counts[empty] = 1.0
        centers = sums / counts.clamp_min(1.0).unsqueeze(1)
        final_mse = float(total_dist.item() / max(1, x.shape[0] * x.shape[1]))
        print(f"[stage3 init] {label} iter {step:02d}/{int(iters)} mse={final_mse:.6g} empty={int(empty.sum().item())}", flush=True)
    return centers.detach().cpu(), final_mse


@torch.no_grad()
def initialize_per_location_codebook(train_loader, e1, d1, e2, quantizer, args, device: torch.device) -> None:
    method = str(getattr(args, "init_codebook_method", "none")).lower()
    if method == "none":
        return
    if not hasattr(quantizer, "initialize_from_vectors"):
        print(f"[stage3 init] quantizer={args.quantizer} has no per-location vector initializer; skipped", flush=True)
        return
    if method not in {"random_samples", "kmeans"}:
        raise ValueError(f"unknown --init-codebook-method {method!r}")

    e1.eval()
    d1.eval()
    e2.eval()
    max_batches = int(getattr(args, "codebook_init_batches", 0))
    max_samples = int(getattr(args, "codebook_init_samples", 0))
    log_every = int(getattr(args, "codebook_init_log_every", 10))
    samples: list[torch.Tensor] = []
    seen = 0
    for batch_idx, (imgs, _labels) in enumerate(train_loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        base = layer1_forward(e1, d1, imgs)
        z2, _ = e2(torch.cat([imgs, base["x1"]], dim=1))
        vectors = z2.detach().permute(0, 2, 3, 1).reshape(-1, z2_ch(args)).float().cpu()
        if max_samples > 0 and seen + int(vectors.shape[0]) > max_samples:
            vectors = vectors[: max_samples - seen]
        samples.append(vectors)
        seen += int(vectors.shape[0])
        if max_samples > 0 and seen >= max_samples:
            break
        if log_every > 0 and batch_idx % log_every == 0:
            print(f"[stage3 init] collected z2 vectors batches={batch_idx} samples={seen}", flush=True)
        if max_batches > 0 and batch_idx >= max_batches:
            break

    if not samples:
        raise RuntimeError("failed to collect z2 vectors for per-location codebook initialization")
    sample_tensor = torch.cat(samples, dim=0)
    num_codes = quantizer_num_codes(args)
    if sample_tensor.shape[0] < 1:
        raise RuntimeError("empty z2 vector sample tensor")
    if method == "random_samples":
        if sample_tensor.shape[0] >= num_codes:
            pick = torch.randperm(sample_tensor.shape[0])[:num_codes]
        else:
            pick = torch.randint(0, sample_tensor.shape[0], (num_codes,))
        centers = sample_tensor[pick]
        init_mse = float("nan")
    else:
        centers, init_mse = kmeans_vectors(
            sample_tensor,
            num_codes,
            iters=int(getattr(args, "kmeans_iters", 20)),
            chunk_size=int(getattr(args, "kmeans_chunk_size", 4096)),
            device=device,
            label=f"{args.quantizer}_K{num_codes}xD{z2_ch(args)}",
        )
    quantizer.initialize_from_vectors(centers)
    print(
        f"[stage3 init] initialized {args.quantizer} K{num_codes}xD{z2_ch(args)} "
        f"from z2 vectors samples={sample_tensor.shape[0]} method={method} kmeans_mse={init_mse:.6g}",
        flush=True,
    )
    l2_metrics = codebook_l2_metrics(quantizer, args)
    if l2_metrics:
        print(f"[stage3 init codebook_l2] {format_metrics(l2_metrics)}", flush=True)


@torch.no_grad()
def initialize_fullmap_codebook(train_loader, e1, d1, e2, quantizer, args, device: torch.device) -> None:
    if not bool(getattr(args, "init_fullmap_codebook", True)):
        print("[stage3 init] full-map codebook init disabled", flush=True)
        return
    if not hasattr(quantizer, "initialize_from_samples"):
        print(f"[stage3 init] quantizer={args.quantizer} has no full-map sample initializer; skipped", flush=True)
        return
    e1.eval()
    d1.eval()
    e2.eval()
    max_batches = int(getattr(args, "codebook_init_batches", 0))
    samples: list[torch.Tensor] = []
    seen_maps = 0
    num_codes = quantizer_num_codes(args)
    for batch_idx, (imgs, _labels) in enumerate(train_loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        base = layer1_forward(e1, d1, imgs)
        z2, _ = e2(torch.cat([imgs, base["x1"]], dim=1))
        flat = z2.detach().float().reshape(-1, int(args.latent_h), int(args.latent_w)).cpu()
        samples.append(flat)
        seen_maps += int(flat.shape[0])
        if max_batches > 0 and batch_idx >= max_batches:
            break
        if max_batches <= 0 and seen_maps >= num_codes:
            break
    if not samples:
        raise RuntimeError("failed to collect z2 samples for full-map codebook initialization")
    sample_tensor = torch.cat(samples, dim=0)
    quantizer.initialize_from_samples(sample_tensor)
    print(f"[stage3 init] initialized full-map codebook from z2 samples maps={sample_tensor.shape[0]} codes={num_codes}", flush=True)
    l2_metrics = codebook_l2_metrics(quantizer, args)
    if l2_metrics:
        print(f"[stage3 init codebook_l2] {format_metrics(l2_metrics)}", flush=True)


@torch.no_grad()
def initialize_fsq_quantizer(train_loader, e1, d1, e2, quantizer, args, device: torch.device) -> None:
    if str(getattr(args, "quantizer", "")) != "fsq":
        return
    if not bool(getattr(args, "fsq_init_stats", True)):
        print("[stage3 init] FSQ stats init disabled", flush=True)
        return
    if not hasattr(quantizer, "initialize_from_data"):
        print("[stage3 init] FSQ quantizer has no stats initializer; skipped", flush=True)
        return

    e1.eval()
    d1.eval()
    e2.eval()
    max_batches = int(getattr(args, "codebook_init_batches", 0))
    max_samples = int(getattr(args, "codebook_init_samples", 0))
    log_every = int(getattr(args, "codebook_init_log_every", 10))
    samples: list[torch.Tensor] = []
    seen = 0
    for batch_idx, (imgs, _labels) in enumerate(train_loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        base = layer1_forward(e1, d1, imgs)
        z2, _ = e2(torch.cat([imgs, base["x1"]], dim=1))
        vectors = z2.detach().permute(0, 2, 3, 1).reshape(-1, z2_ch(args)).float().cpu()
        if max_samples > 0 and seen + int(vectors.shape[0]) > max_samples:
            vectors = vectors[: max_samples - seen]
        samples.append(vectors)
        seen += int(vectors.shape[0])
        if max_samples > 0 and seen >= max_samples:
            break
        if log_every > 0 and batch_idx % log_every == 0:
            print(f"[stage3 init] collected FSQ z2 vectors batches={batch_idx} samples={seen}", flush=True)
        if max_batches > 0 and batch_idx >= max_batches:
            break

    if not samples:
        raise RuntimeError("failed to collect z2 vectors for FSQ stats initialization")
    sample_tensor = torch.cat(samples, dim=0)
    quantizer.initialize_from_data(sample_tensor, quantile=float(getattr(args, "fsq_init_quantile", 0.001)))
    metrics = quantizer.extra_metrics() if hasattr(quantizer, "extra_metrics") else {}
    print(
        f"[stage3 init] initialized FSQ affine stats samples={sample_tensor.shape[0]} "
        f"quantile={float(getattr(args, 'fsq_init_quantile', 0.001)):g} {format_metrics(metrics)}",
        flush=True,
    )

def compute_losses(out: dict, imgs: torch.Tensor, args: argparse.Namespace, epoch: int, teacher_out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor | float]:
    target = out["q2_index"].detach().long()
    logits = out["index_logits"]
    stats = out["vq_stats"]
    loss_rec = recon_loss(out["final_oracle"], imgs)
    loss_vq = stats["codebook_loss"] + float(args.beta_commit) * stats["commit_loss"]
    if teacher_out is not None and "u2" in teacher_out:
        loss_u2_teacher = F.mse_loss(out["u2"].float(), teacher_out["u2"].detach().float())
    else:
        loss_u2_teacher = loss_rec.new_zeros(())
    flat_logits, flat_target = flatten_index_logits(logits, target)
    loss_index = F.cross_entropy(flat_logits, flat_target)
    use_index = bool(getattr(args, "train_indexnet", False)) and int(epoch) > int(args.warmup_index_epochs)
    lambda_index_eff = float(args.lambda_index) if use_index else 0.0
    lambda_u2_teacher_eff = float(args.lambda_u2_teacher) if teacher_out is not None and u2_teacher_phase_enabled(args, "finetune") else 0.0
    recon = str(getattr(args, "recon", "recon_x2"))
    if recon == "recon_x2":
        loss = loss_rec + float(args.lambda_vq) * loss_vq + lambda_index_eff * loss_index
    # loss = float(args.lambda_vq) * loss_vq + lambda_index_eff * loss_index
    elif recon == "recon_u2":
        loss = lambda_u2_teacher_eff * loss_u2_teacher + float(args.lambda_vq) * loss_vq + lambda_index_eff * loss_index
    else:
        raise ValueError(f"unknown --recon {recon!r}")
    return {
        "loss": loss,
        "loss_rec": loss_rec,
        "loss_vq": loss_vq,
        "loss_codebook": stats["codebook_loss"],
        "loss_commit": stats["commit_loss"],
        "loss_index": loss_index,
        "loss_u2_teacher": loss_u2_teacher,
        "lambda_index_eff": float(lambda_index_eff),
        "lambda_u2_teacher_eff": float(lambda_u2_teacher_eff),
    }

@torch.no_grad()
def validate(loader, e1, d1, e2, d2, combiner, quantizer, indexnet, teacher, args, epoch: int) -> dict[str, float]:
    e1.eval()
    d1.eval()
    e2.eval()
    d2.eval()
    combiner.eval()
    quantizer.eval()
    indexnet.eval()
    device = next(e2.parameters()).device
    m = meters(METRIC_NAMES)
    hist = torch.zeros(quantizer_num_codes(args), dtype=torch.float32)
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = layer3_forward(e1, d1, e2, d2, combiner, quantizer, indexnet, imgs)
        teacher_out = forward_layer2_teacher(teacher, imgs, args)
        losses = compute_losses(out, imgs, args, epoch, teacher_out)
        update_metrics(m, hist, out, imgs, losses)
    return finalize_stage3_metrics(m, hist, quantizer, args)


def best_guard_passed(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    if bool(getattr(args, "disable_best_guard", False)):
        return True
    margin = float(getattr(args, "best_psnr_margin", 0.0))
    return float(metrics["psnr_oracle"]) >= float(metrics["psnr_x1"]) + margin


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = get_loader(cfg)
    e1, d1, e2, d2, combiner, quantizer, indexnet = build_layer3(args, cfg.device)
    load_initial_weights(args, e1, d1, e2, d2, combiner)
    teacher = load_layer2_teacher(args, cfg.device)

    best = -1.0
    if str(args.quantizer) == "vq":
        title_quantizer = "Oracle-VQ"
    elif str(args.quantizer) == "fsq":
        title_quantizer = "Scalar-FSQ"
    elif str(args.quantizer) == "cvq":
        title_quantizer = "Full-map CVQ"
    elif str(args.quantizer) == "fullmap_simvq":
        title_quantizer = "Full-map SimVQ"
    else:
        title_quantizer = "Oracle-SimVQ"
    set_trainable(e1, False)
    set_trainable(d1, False)
    set_trainable(indexnet, bool(getattr(args, "train_indexnet", False)))
    print_run_header(
        args,
        f"Stage 3 | {title_quantizer} refinement with side IndexNet",
        len(train_loader.dataset),
        len(val_loader.dataset),
        modules={
            "E1": e1,
            "D1": d1,
            "E2": e2,
            "D2": d2,
            "combiner": combiner,
            "quantizer": quantizer,
            "IndexNet": indexnet,
        },
    )
    initialize_fsq_quantizer(train_loader, e1, d1, e2, quantizer, args, cfg.device)
    initialize_per_location_codebook(train_loader, e1, d1, e2, quantizer, args, cfg.device)
    initialize_fullmap_codebook(train_loader, e1, d1, e2, quantizer, args, cfg.device)
    if bool(getattr(args, "eval_init_only", False)):
        val_metrics = validate(val_loader, e1, d1, e2, d2, combiner, quantizer, indexnet, teacher, args, 0)
        print(f"[stage3 init eval] {display_metrics(val_metrics)} score=psnr_oracle", flush=True)
        version = jsccf_io.safe_artifact_name(getattr(args, "version", "default"))
        version_part = f"_jscc_f_{version}" if version else ""
        out = Path(resolve_path(args.save_dir)) / f"{stage3_name(args)}{version_part}_init_{args.init_codebook_method}_eval.json"
        write_json(out, {"args": vars(args), "metrics": val_metrics})
        print(f"[stage3 init eval] wrote {out}", flush=True)
        return
    # set_trainable(combiner, False)
    opt = build_optimizer(args, e1, d1, e2, d2, combiner, quantizer, indexnet)

    for epoch in range(1, int(args.epochs) + 1):
        train_index_path = bool(getattr(args, "train_indexnet", False)) and int(epoch) > int(args.warmup_index_epochs)
        quantizer.train()
        indexnet.train(bool(getattr(args, "train_indexnet", False)))
        m = meters(METRIC_NAMES)
        hist = torch.zeros(quantizer_num_codes(args), dtype=torch.float32)
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            out = layer3_forward_unfrozen(
                e1,
                d1,
                e2,
                d2,
                combiner,
                quantizer,
                indexnet,
                imgs,
                train_index_path=train_index_path,
            )
            teacher_out = forward_layer2_teacher(teacher, imgs, args)
            losses = compute_losses(out, imgs, args, epoch, teacher_out)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            update_metrics(m, hist, out, imgs, losses)

        metrics = finalize_stage3_metrics(m, hist, quantizer, args)
        print_epoch("stage3-unfrozen", epoch, int(args.epochs), display_metrics(metrics), time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, e1, d1, e2, d2, combiner, quantizer, indexnet, teacher, args, epoch)
            score = val_metrics["psnr_oracle"]
            # print(f"[stage3 val {epoch:03d}] {val_metrics} score=psnr_oracle")
            print(f"[stage3 val {epoch:03d}] {display_metrics(val_metrics)} score=psnr_oracle")
            guard_ok = best_guard_passed(val_metrics, args)
            if score > best and guard_ok:
                best = score
                jsccf_io.save_layer3_checkpoint(
                    jsccf_io.ckpt_path(args, stage3_name(args), "best"),
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    e1=e1,
                    d1=d1,
                    e2=e2,
                    d2=d2,
                    combiner=combiner,
                    quantizer=quantizer,
                    indexnet=indexnet,
                )
            elif score > best:
                print(
                    f"[stage3 guard] skip best save: psnr_oracle={score:.6g} "
                    f"psnr_x1={val_metrics['psnr_x1']:.6g} margin={float(args.best_psnr_margin):.6g}",
                    flush=True,
                )
        if should_save_latest(args, epoch):
            jsccf_io.save_layer3_checkpoint(
                jsccf_io.ckpt_path(args, stage3_name(args), "latest"),
                epoch=epoch,
                args=args,
                metrics=metrics,
                e1=e1,
                d1=d1,
                e2=e2,
                d2=d2,
                combiner=combiner,
                quantizer=quantizer,
                indexnet=indexnet,
            )

    jsccf_io.save_layer3_checkpoint(
        jsccf_io.ckpt_path(args, stage3_name(args), "latest"),
        epoch=int(args.epochs),
        args=args,
        metrics=metrics,
        e1=e1,
        d1=d1,
        e2=e2,
        d2=d2,
        combiner=combiner,
        quantizer=quantizer,
        indexnet=indexnet,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--recon", type=str, default="recon_x2",choices=["recon_u2","recon_x2"])
    p.add_argument("--version", type=str, default="simvq", help="Version of the JSCC-f training; affects checkpoint and log names.")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=jsccf_io.default_save_dir())
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default="")
    p.add_argument("--latent-ch", type=int, default=16)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5, help="LR for unfrozen E1/D1/E2/D2/Combiner.")
    p.add_argument("--lr-codebook", type=float, default=1e-4, help="Deprecated compatibility option; unfrozen Stage3 uses --lr-simvq for quantizer parameters.")
    p.add_argument("--lr-simvq", type=float, default=5e-5, help="LR for quantizer parameters: VQ/SimVQ embedding or CVQ codebook.")
    p.add_argument("--lr-indexnet", type=float, default=1e-5, help="LR for IndexNet.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--layer1-ckpt", type=str, default="MY-V2/jscc-f/checkpoints/jscc_f_no-c1_layer1_best.pth")
    p.add_argument("--layer2-ckpt", type=str, default="MY-V2/jscc-f/checkpoints/jscc_f_only-z2_layer2_combiner_best.pth", help="Optional warm start for E1/D1/E2/D2/Combiner from continuous Layer2.")
    p.add_argument("--quantizer", type=str, default="vq", choices=["simvq", "vq", "fsq", "cvq", "fullmap_simvq"], help="Stage3 z2 quantizer.")
    p.add_argument("--vq-k", type=int, default=128, help="Plain VQ codebook size; codebook shape is [K,z2_ch].")
    p.add_argument("--fsq-levels", type=str, default="15", help="Scalar FSQ levels. Set one integer for all z2 channels, or 20 comma-separated integers.")
    p.add_argument("--fsq-init-quantile", type=float, default=0.001, help="Lower/upper z2 quantile used to initialize FSQ affine range.")
    p.add_argument("--no-fsq-init-stats", dest="fsq_init_stats", action="store_false", help="Skip initializing FSQ affine scale/center from training z2 statistics.")
    p.add_argument("--fsq-freeze-affine", action="store_true", help="Freeze FSQ affine scale/center after initialization.")
    p.add_argument("--simvq-k", type=int, default=64)
    p.add_argument("--simvq-train-codebook", action="store_true", help="Also train the base SimVQ embedding table; default freezes it.")
    p.add_argument("--cvq-k", type=int, default=2048, help="Full-map CVQ codebook size; codebook shape is [K,16,16].")
    p.add_argument("--cvq-chunk-size", type=int, default=128, help="Nearest-neighbor chunk size for full-map CVQ lookup.")
    p.add_argument("--fullmap-simvq-k", type=int, default=16384, help="Full-map SimVQ base codebook size; codebook shape is [K,16,16].")
    p.add_argument("--fullmap-simvq-chunk-size", type=int, default=128, help="Nearest-neighbor chunk size for full-map SimVQ lookup.")
    p.add_argument("--fullmap-simvq-train-codebook", action="store_true", help="Also train the base full-map SimVQ codebook; default freezes it.")
    p.add_argument("--init-codebook-method", type=str, default="none", choices=["none", "random_samples", "kmeans"], help="Initialize per-location VQ/SimVQ codebook from training z2 vectors.")
    p.add_argument("--codebook-init-samples", type=int, default=0, help="Max per-location z2 vectors for codebook init; 0 means one full train-loader pass.")
    p.add_argument("--codebook-init-log-every", type=int, default=10, help="Print per-location codebook sample collection progress every N batches; set 0 to disable.")
    p.add_argument("--kmeans-iters", type=int, default=20, help="K-means iterations for per-location codebook init.")
    p.add_argument("--kmeans-chunk-size", type=int, default=4096, help="Vector chunk size for per-location k-means assignment.")
    p.add_argument("--eval-init-only", action="store_true", help="Initialize codebook, run validation once, write JSON metrics, and exit without training.")
    p.add_argument("--no-fullmap-codebook-init", dest="init_fullmap_codebook", action="store_false", help="Skip initializing full-map codebook entries from continuous z2 maps.")
    p.add_argument("--codebook-init-batches", type=int, default=0, help="Batches used for z2 codebook initialization; 0 means until codebook is filled or loader ends.")
    p.add_argument("--codebook-epochs", type=int, default=0, help="Deprecated compatibility option; unfrozen Stage3 uses one training phase.")
    p.add_argument("--codebook-dist-sample", type=int, default=512, help="Sampled codebook embeddings used for pairwise L2 distance metrics; set 0 to disable pairwise distances.")
    p.add_argument("--beta-commit", type=float, default=0.25)
    p.add_argument("--lambda-vq", type=float, default=0.05)
    p.add_argument("--lambda-u2-teacher", type=float, default=1.0, help="Weight for MSE between current Stage3 u2 and frozen continuous Layer2 teacher u2.")
    p.add_argument("--u2-teacher-phases", type=str, default="none", choices=["none", "codebook", "finetune", "all"], help="Compatibility option; in unfrozen Stage3, finetune/all enable the frozen Layer2 u2 teacher loss.")
    p.add_argument("--lambda-index", type=float, default=0.001)
    p.add_argument("--warmup-index-epochs", type=int, default=400)
    p.add_argument("--train-indexnet", action="store_true", help="Enable IndexNet optimization after warmup; default keeps Stage3 focused on oracle q2.")
    p.add_argument("--best-psnr-margin", type=float, default=0.0, help="Required psnr_oracle - psnr_x1 margin before saving best.")
    p.add_argument("--disable-best-guard", action="store_true", help="Allow best checkpoint saves even when psnr_oracle is below psnr_x1.")
    p.add_argument("--index-hidden", type=int, default=128)
    p.add_argument("--index-depth", type=int, default=3)
    p.add_argument("--index-heads", type=int, default=4, help="Attention heads for full-map CVQ IndexNet.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.stage = "stage3"
    args.variant = "combiner"
    args.lambda_u2 = 0.0
    check_jsccf_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(Path(resolve_path(args.save_dir)) / f"{stage3_name(args)}_jscc_f_{args.version}.log")
    setup_log_file(args.log_file)
    write_json(Path(resolve_path(args.save_dir)) / f"{stage3_name(args)}_args.json", vars(args))
    train(args)


if __name__ == "__main__":
    main()
