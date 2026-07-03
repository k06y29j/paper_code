# 作用：在 Stage0 checkpoint 上分别拟合 C1 前16通道和 C1+第17通道的 PCA，保存分量向量用于比较。
# 输出：summary.json、pca_components.pt、pca16_components.csv、pca17_components.csv。
# 示例：conda run -n cddm_ddnm python MY/train_cvq-v2/eval/stage0_c1_plus_one_pca_probe.py --max-images 8 --val-num-workers 0

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
from Autoencoder.net.network import JSCC_encoder  # noqa: E402
from common import CDDMJSCCConfig, resolve_path, seed_everything  # noqa: E402


DEFAULT_CKPT = (
    "MY/checkpoints-cvq-v2-c36-snr12-k16384/"
    "cvq_v2_c36_snr12_stage0_best.pth"
)
DEFAULT_OUT_DIR = "MY/train_cvq-v2/eval/outputs/stage0_c1_plus_one_pca_probe"


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
        batch_size=1,
        test_batch=1,
        num_workers=0,
        val_num_workers=int(args.val_num_workers),
        train_data_dir=resolve_path(str(Path(args.data_dir) / "DIV2K_train_HR")),
        test_data_dir=resolve_path(str(Path(args.data_dir) / "DIV2K_valid_HR")),
        CUDA=(not bool(args.cpu)),
    )


def load_stage0_encoder(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict]:
    cfg = build_config(args)
    encoder = JSCC_encoder(cfg, int(args.latent_ch)).to(device)
    ckpt_path = resolve_path(args.ckpt)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cvq_io.load_state(encoder, obj["encoder_state_dict"], "encoder", strict=True)
    encoder.eval()
    print(f"loaded stage0 checkpoint: {ckpt_path}")
    print(f"checkpoint stage={obj.get('stage')} epoch={obj.get('epoch')} route={obj.get('route')}")
    return encoder, obj


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


def fit_pca(x: torch.Tensor) -> dict[str, torch.Tensor | int]:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected [N,C,H,W] or [C,H,W], got {tuple(x.shape)}")
    n, c, h, w = x.shape
    tokens = x.permute(0, 2, 3, 1).reshape(n * h * w, c).float()
    mean = tokens.mean(dim=0, keepdim=True)
    centered = tokens - mean
    cov = centered.t().matmul(centered) / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0.0)
    eigvecs = eigvecs[:, order]
    return {
        "tokens": tokens,
        "mean": mean,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "channels": c,
        "height": h,
        "width": w,
        "images": n,
        "tokens_count": tokens.shape[0],
    }


def pca_reconstruction_mse(pca: dict[str, torch.Tensor | int], keep_components: int) -> tuple[float, float]:
    tokens = pca["tokens"]
    mean = pca["mean"]
    eigvals = pca["eigvals"]
    eigvecs = pca["eigvecs"]
    if not isinstance(tokens, torch.Tensor) or not isinstance(mean, torch.Tensor):
        raise TypeError("invalid PCA tokens/mean")
    if not isinstance(eigvals, torch.Tensor) or not isinstance(eigvecs, torch.Tensor):
        raise TypeError("invalid PCA eigensystem")
    keep = min(int(keep_components), int(eigvecs.shape[1]))
    centered = tokens - mean
    basis = eigvecs[:, :keep]
    coeff = centered.matmul(basis)
    rec = coeff.matmul(basis.t()) + mean
    mse = torch.mean((rec - tokens).square())
    explained = eigvals[:keep].sum() / eigvals.sum().clamp_min(1e-12)
    return float(mse.item()), float(explained.item())


def print_pca_spectrum(label: str, pca: dict[str, torch.Tensor | int]) -> None:
    eigvals = pca["eigvals"]
    if not isinstance(eigvals, torch.Tensor):
        raise TypeError("invalid PCA eigvals")
    explained = eigvals / eigvals.sum().clamp_min(1e-12)
    cumulative = torch.cumsum(explained, dim=0)
    print(f"[{label} components]")
    print("component,eigenvalue,explained,cumulative")
    for idx in range(eigvals.numel()):
        print(
            f"{idx + 1},"
            f"{float(eigvals[idx].item()):.10g},"
            f"{float(explained[idx].item()):.10g},"
            f"{float(cumulative[idx].item()):.10g}"
        )


def tensor_from_pca(pca: dict[str, torch.Tensor | int], key: str) -> torch.Tensor:
    value = pca[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"invalid PCA tensor: {key}")
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_components_csv(path: Path, pca: dict[str, torch.Tensor | int], label: str) -> None:
    eigvals = tensor_from_pca(pca, "eigvals")
    eigvecs = tensor_from_pca(pca, "eigvecs")
    explained = eigvals / eigvals.sum().clamp_min(1e-12)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pca",
                "component",
                "channel",
                "value",
                "eigenvalue",
                "explained",
            ],
        )
        writer.writeheader()
        for component in range(eigvecs.shape[1]):
            for channel in range(eigvecs.shape[0]):
                writer.writerow(
                    {
                        "pca": label,
                        "component": component + 1,
                        "channel": channel + 1,
                        "value": f"{float(eigvecs[channel, component].item()):.10g}",
                        "eigenvalue": f"{float(eigvals[component].item()):.10g}",
                        "explained": f"{float(explained[component].item()):.10g}",
                    }
                )


def print_component_vectors(label: str, pca: dict[str, torch.Tensor | int]) -> None:
    eigvecs = tensor_from_pca(pca, "eigvecs")
    print(f"[{label} component vectors]")
    header = ",".join(["component"] + [f"ch{idx + 1}" for idx in range(eigvecs.shape[0])])
    print(header)
    for component in range(eigvecs.shape[1]):
        values = [f"{float(eigvecs[channel, component].item()):.10g}" for channel in range(eigvecs.shape[0])]
        print(",".join([str(component + 1)] + values))


def save_pca_artifacts(
    out_dir: Path,
    pca16: dict[str, torch.Tensor | int],
    pca17: dict[str, torch.Tensor | int],
    summary: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pca16": {
            "mean": tensor_from_pca(pca16, "mean"),
            "eigvals": tensor_from_pca(pca16, "eigvals"),
            "eigvecs": tensor_from_pca(pca16, "eigvecs"),
            "images": int(pca16["images"]),
            "tokens_count": int(pca16["tokens_count"]),
        },
        "pca17": {
            "mean": tensor_from_pca(pca17, "mean"),
            "eigvals": tensor_from_pca(pca17, "eigvals"),
            "eigvecs": tensor_from_pca(pca17, "eigvecs"),
            "images": int(pca17["images"]),
            "tokens_count": int(pca17["tokens_count"]),
        },
        "summary": summary,
    }
    torch.save(payload, out_dir / "pca_components.pt")
    write_components_csv(out_dir / "pca16_components.csv", pca16, "C16")
    write_components_csv(out_dir / "pca17_components.csv", pca17, "C17")
    write_json(out_dir / "summary.json", summary)
    print(f"saved pca tensors: {out_dir / 'pca_components.pt'}")
    print(f"saved pca16 components csv: {out_dir / 'pca16_components.csv'}")
    print(f"saved pca17 components csv: {out_dir / 'pca17_components.csv'}")


def print_component_comparison(pca16: dict[str, torch.Tensor | int], pca17: dict[str, torch.Tensor | int], keep: int) -> None:
    eig16 = pca16["eigvals"]
    eig17 = pca17["eigvals"]
    vec16 = pca16["eigvecs"]
    vec17 = pca17["eigvecs"]
    if not isinstance(eig16, torch.Tensor) or not isinstance(eig17, torch.Tensor):
        raise TypeError("invalid PCA eigvals")
    if not isinstance(vec16, torch.Tensor) or not isinstance(vec17, torch.Tensor):
        raise TypeError("invalid PCA eigvecs")
    exp16 = eig16 / eig16.sum().clamp_min(1e-12)
    exp17 = eig17 / eig17.sum().clamp_min(1e-12)
    cum16 = torch.cumsum(exp16, dim=0)
    cum17 = torch.cumsum(exp17, dim=0)
    limit = min(int(keep), vec16.shape[1], vec17.shape[1])
    print("[component comparison: C16 PCA vs C17 PCA]")
    print(
        "component,"
        "eig_c16,eig_c17,"
        "explained_c16,explained_c17,"
        "cumulative_c16,cumulative_c17,"
        "abs_cos_first16,cos_first16,"
        "c17_first16_norm,c17_ch17_loading"
    )
    for idx in range(limit):
        v16 = vec16[:, idx]
        v17_first16 = vec17[: vec16.shape[0], idx]
        first16_norm = v17_first16.norm().clamp_min(1e-12)
        cos = torch.dot(v16, v17_first16 / first16_norm)
        extra = vec17[vec16.shape[0], idx]
        print(
            f"{idx + 1},"
            f"{float(eig16[idx].item()):.10g},"
            f"{float(eig17[idx].item()):.10g},"
            f"{float(exp16[idx].item()):.10g},"
            f"{float(exp17[idx].item()):.10g},"
            f"{float(cum16[idx].item()):.10g},"
            f"{float(cum17[idx].item()):.10g},"
            f"{float(cos.abs().item()):.10g},"
            f"{float(cos.item()):.10g},"
            f"{float(first16_norm.item()):.10g},"
            f"{float(extra.item()):.10g}"
        )


@torch.no_grad()
def run_probe(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    device = torch.device("cpu" if bool(args.cpu) or not torch.cuda.is_available() else "cuda:0")
    encoder, _obj = load_stage0_encoder(args, device)
    loader = make_valid_loader(args)
    latents = []
    labels_seen = []
    seen = 0
    max_images = int(args.max_images)
    for imgs, labels in loader:
        if max_images > 0 and seen >= max_images:
            break
        if max_images > 0 and seen + int(imgs.shape[0]) > max_images:
            imgs = imgs[: max_images - seen]
            labels = labels[: max_images - seen]
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
        latents.append(z_norm.detach().float().cpu())
        labels_seen.extend(labels)
        seen += int(z_norm.shape[0])
    if not latents:
        raise RuntimeError("no validation images collected")
    latent = torch.cat(latents, dim=0)
    c1 = int(args.c1_ch)
    if latent.shape[1] < c1 + 1:
        raise RuntimeError(f"latent channels {latent.shape[1]} < c1+1={c1 + 1}")

    pca16 = fit_pca(latent[:, :c1])
    pca17 = fit_pca(latent[:, : c1 + 1])
    keep16 = int(args.keep_components_c16)
    keep17 = int(args.keep_components_c17)
    mse_16, explained_16 = pca_reconstruction_mse(pca16, keep16)
    mse_17, explained_17 = pca_reconstruction_mse(pca17, keep17)

    print(f"images={int(latent.shape[0])}")
    print(f"image_labels={labels_seen[:5]}")
    print(f"latent_shape={tuple(latent.shape)}")
    print(f"pca_input_c16_shape={(int(latent.shape[0]), c1, int(args.latent_h), int(args.latent_w))} keep={keep16} tokens={int(pca16['tokens_count'])}")
    print(f"pca_input_c17_shape={(int(latent.shape[0]), c1 + 1, int(args.latent_h), int(args.latent_w))} keep={keep17} tokens={int(pca17['tokens_count'])}")
    print(f"pca_mse_c16={mse_16:.10g}")
    print(f"pca_mse_c17={mse_17:.10g}")
    print(f"pca_explained_c16={explained_16:.10g}")
    print(f"pca_explained_c17={explained_17:.10g}")
    print_pca_spectrum("C16", pca16)
    print_pca_spectrum("C17", pca17)
    print_component_comparison(pca16, pca17, min(keep16, keep17))
    if bool(args.print_component_vectors):
        print_component_vectors("C16", pca16)
        print_component_vectors("C17", pca17)
    if bool(args.save_components):
        summary = {
            "checkpoint": resolve_path(args.ckpt),
            "images": int(latent.shape[0]),
            "latent_shape": list(latent.shape),
            "c16_shape": [int(latent.shape[0]), c1, int(args.latent_h), int(args.latent_w)],
            "c17_shape": [int(latent.shape[0]), c1 + 1, int(args.latent_h), int(args.latent_w)],
            "keep_components_c16": keep16,
            "keep_components_c17": keep17,
            "pca_mse_c16": mse_16,
            "pca_mse_c17": mse_17,
            "pca_explained_c16": explained_16,
            "pca_explained_c17": explained_17,
        }
        save_pca_artifacts(Path(resolve_path(args.out_dir)), pca16, pca17, summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--snr-db", type=float, default=12.0)
    p.add_argument("--latent-ch", type=int, default=36)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--keep-components-c16", type=int, default=16)
    p.add_argument("--keep-components-c17", type=int, default=17)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--save-components", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print-component-vectors", action="store_true")
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260616)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
