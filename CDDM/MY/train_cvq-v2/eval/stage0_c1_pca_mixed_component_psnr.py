# 作用：在 Stage0 checkpoint 上比较 C1 的 16维 PCA 与 C1+1 的 17维 PCA 基底，逐步替换低阶分量并测解码 PSNR。
# 输出：summary.json、replaced_components.csv、mixed_components.pt。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/stage0_c1_pca_mixed_component_psnr.py --max-images 8 --val-num-workers 0

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

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
from common import CDDMJSCCConfig, psnr_per_image, resolve_path, seed_everything  # noqa: E402


DEFAULT_CKPT = (
    "MY/checkpoints-cvq-v2-c36-snr12-k16384/"
    "cvq_v2_c36_snr12_stage0_best.pth"
)
DEFAULT_OUT_DIR = "MY/train_cvq-v2/eval/outputs/stage0_c1_pca_mixed_component_psnr"


def load_cvq_io():
    spec = importlib.util.spec_from_file_location("cvq_v2_io", PACKAGE_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cvq_io = load_cvq_io()


def build_config(args: argparse.Namespace) -> CDDMJSCCConfig:
    return CDDMJSCCConfig(
        C=int(args.latent_ch),
        SNRs=float(args.snr_db),
        channel_type="awgn",
        batch_size=int(args.batch_size),
        test_batch=int(args.batch_size),
        num_workers=0,
        val_num_workers=int(args.val_num_workers),
        train_data_dir=resolve_path(str(Path(args.data_dir) / "DIV2K_train_HR")),
        test_data_dir=resolve_path(str(Path(args.data_dir) / "DIV2K_valid_HR")),
        CUDA=(not bool(args.cpu)),
    )


def load_stage0_models(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
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
    return encoder, decoder, obj


def make_valid_loader(args: argparse.Namespace):
    cfg = build_config(args)
    transform = transforms.Compose(
        [
            transforms.CenterCrop((int(args.image_size), int(args.image_size))),
            transforms.ToTensor(),
        ]
    )
    dataset = FlatImageFolder(root=cfg.test_data_dir, transform=transform)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.val_num_workers),
        pin_memory=(not bool(args.cpu)),
        drop_last=False,
    )


@torch.no_grad()
def collect_eval_tensors(
    loader,
    encoder: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(encoder.parameters()).device
    imgs_all = []
    latents = []
    seen = 0
    max_images = int(args.max_images)
    for imgs, _labels in loader:
        if max_images > 0 and seen >= max_images:
            break
        if max_images > 0 and seen + int(imgs.shape[0]) > max_images:
            imgs = imgs[: max_images - seen]
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
        imgs_all.append(imgs.detach().float().cpu())
        latents.append(z_norm.detach().float().cpu())
        seen += int(z_norm.shape[0])
    if not latents:
        raise RuntimeError("no validation images collected")
    return torch.cat(imgs_all, dim=0), torch.cat(latents, dim=0)


def fit_pca(x: torch.Tensor) -> dict[str, torch.Tensor]:
    if x.ndim != 4:
        raise ValueError(f"expected [N,C,H,W], got {tuple(x.shape)}")
    n, c, h, w = x.shape
    tokens = x.permute(0, 2, 3, 1).reshape(n * h * w, c).float()
    mean = tokens.mean(dim=0, keepdim=True)
    centered = tokens - mean
    cov = centered.t().matmul(centered) / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    return {
        "mean": mean,
        "eigvals": eigvals[order].clamp_min(0.0).contiguous(),
        "eigvecs": eigvecs[:, order].contiguous(),
    }


def make_mixed_c16_components(pca16: dict[str, torch.Tensor], pca17: dict[str, torch.Tensor], replace_components: int) -> tuple[torch.Tensor, list[dict]]:
    base = pca16["eigvecs"].clone()
    vec16 = pca16["eigvecs"]
    vec17 = pca17["eigvecs"]
    rows = []
    limit = min(int(replace_components), vec16.shape[1], vec17.shape[1])
    for idx in range(limit):
        source = vec17[: vec16.shape[0], idx].clone()
        source_norm = source.norm().clamp_min(1e-12)
        sign = 1.0 if torch.dot(vec16[:, idx], source) >= 0 else -1.0
        source = sign * source / source_norm
        base[:, idx] = source
        diff = base[:, idx] - vec16[:, idx]
        rows.append(
            {
                "component": idx + 1,
                "sign": int(sign),
                "source_first16_norm_before_normalize": float(source_norm.item()),
                "abs_cos_first16": float(torch.dot(vec16[:, idx], source).abs().item()),
                "c17_ch17_loading": float(vec17[vec16.shape[0], idx].item()),
                "mean_abs_delta": float(diff.abs().mean().item()),
                "max_abs_delta": float(diff.abs().max().item()),
            }
        )
    return base, rows


def reconstruct_with_components(x: torch.Tensor, mean: torch.Tensor, components: torch.Tensor, keep: int) -> torch.Tensor:
    n, c, h, w = x.shape
    tokens = x.permute(0, 2, 3, 1).reshape(n * h * w, c).float()
    mean = mean.to(dtype=tokens.dtype)
    basis = components[:, : int(keep)].to(dtype=tokens.dtype)
    centered = tokens - mean
    coeff = centered.matmul(basis)
    rec = coeff.matmul(basis.t()) + mean
    return rec.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def decode_psnr(
    imgs: torch.Tensor,
    latent: torch.Tensor,
    c1_rec: torch.Tensor,
    decoder: torch.nn.Module,
    args: argparse.Namespace,
) -> float:
    device = next(decoder.parameters()).device
    c1 = int(args.c1_ch)
    batch = int(args.batch_size)
    vals = []
    for start in range(0, int(imgs.shape[0]), batch):
        end = min(start + batch, int(imgs.shape[0]))
        img_b = imgs[start:end].to(device)
        c1_b = c1_rec[start:end].to(device)
        c2_b = latent[start:end, c1:].to(device)
        recon = decoder(torch.cat([c1_b, c2_b], dim=1)).clamp(0.0, 1.0)
        vals.append(psnr_per_image(recon, img_b).detach().float().cpu())
    return float(torch.cat(vals).mean().item())


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    device = torch.device("cpu" if bool(args.cpu) or not torch.cuda.is_available() else "cuda:0")
    encoder, decoder, _obj = load_stage0_models(args, device)
    imgs, latent = collect_eval_tensors(make_valid_loader(args), encoder, args)
    c1 = int(args.c1_ch)
    c1_latent = latent[:, :c1]
    c17_latent = latent[:, : c1 + 1]

    pca16 = fit_pca(c1_latent)
    pca17 = fit_pca(c17_latent)
    mixed_components, replace_rows = make_mixed_c16_components(pca16, pca17, int(args.replace_components))

    c1_pca16 = reconstruct_with_components(c1_latent, pca16["mean"], pca16["eigvecs"], int(args.keep_components_c16))
    c1_mixed = reconstruct_with_components(c1_latent, pca16["mean"], mixed_components, int(args.keep_components_c16))
    c1_mse_pca16 = float(torch.mean((c1_pca16 - c1_latent).square()).item())
    c1_mse_mixed = float(torch.mean((c1_mixed - c1_latent).square()).item())

    psnr_pca16 = decode_psnr(imgs, latent, c1_pca16, decoder, args)
    psnr_mixed = decode_psnr(imgs, latent, c1_mixed, decoder, args)
    psnr_original = decode_psnr(imgs, latent, c1_latent, decoder, args)

    summary = {
        "checkpoint": resolve_path(args.ckpt),
        "images": int(imgs.shape[0]),
        "latent_shape": list(latent.shape),
        "replace_components": int(args.replace_components),
        "keep_components_c16": int(args.keep_components_c16),
        "psnr_original_c1": psnr_original,
        "psnr_pca16_components": psnr_pca16,
        "psnr_mixed_c17_components_1_to_n": psnr_mixed,
        "delta_mixed_minus_pca16": psnr_mixed - psnr_pca16,
        "delta_pca16_minus_original": psnr_pca16 - psnr_original,
        "delta_mixed_minus_original": psnr_mixed - psnr_original,
        "c1_mse_pca16": c1_mse_pca16,
        "c1_mse_mixed": c1_mse_mixed,
    }
    print("=== C1 PCA mixed component PSNR ===")
    for key, value in summary.items():
        print(f"{key}={value}")
    print("[replaced component stats]")
    for row in replace_rows:
        print(
            "component={component} sign={sign:+d} norm_before={source_first16_norm_before_normalize:.10g} "
            "abs_cos={abs_cos_first16:.10g} c17_ch17_loading={c17_ch17_loading:.10g} "
            "mean_abs_delta={mean_abs_delta:.10g} max_abs_delta={max_abs_delta:.10g}".format(**row)
        )

    out_dir = Path(resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "replaced_components.csv", replace_rows)
    torch.save(
        {
            "summary": summary,
            "pca16_mean": pca16["mean"],
            "pca16_components": pca16["eigvecs"],
            "pca17_components": pca17["eigvecs"],
            "mixed_c16_components": mixed_components,
        },
        out_dir / "mixed_components.pt",
    )
    print(f"saved summary: {out_dir / 'summary.json'}")
    print(f"saved replacement stats: {out_dir / 'replaced_components.csv'}")
    print(f"saved tensors: {out_dir / 'mixed_components.pt'}")


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
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-num-workers", type=int, default=0)
    p.add_argument("--keep-components-c16", type=int, default=16)
    p.add_argument("--replace-components", type=int, default=9)
    p.add_argument("--seed", type=int, default=20260616)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
