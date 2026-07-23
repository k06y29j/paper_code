#!/usr/bin/env python3
"""Run a trained JSCC-f CodeFormer checkpoint from frozen Layer1 ``x1`` only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from architecture import CodeFormer, VQAutoencoder
from layer1 import build_layer1, freeze_layer1, layer1_x1, load_layer1_checkpoint


THIS_DIR = Path(__file__).resolve().parent
JSCCF_DIR = THIS_DIR.parent
CDDM_ROOT = JSCCF_DIR.parents[1]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else CDDM_ROOT / path


def _checkpoint_args(payload: dict, override: argparse.Namespace) -> SimpleNamespace:
    source = dict(payload.get("args", {}))
    if not source:
        raise KeyError("CodeFormer checkpoint does not contain its training args")
    # The Layer1 override is intentional; all model-shape fields remain bound to
    # the saved checkpoint so a mismatched architecture fails strict loading.
    source["layer1_arch"] = override.layer1_arch or payload.get("layer1_arch", source.get("layer1_arch", "cnn"))
    source["layer1_ckpt"] = override.layer1_ckpt or source.get("layer1_ckpt", "")
    return SimpleNamespace(**source)


def build_from_checkpoint(payload: dict, args: SimpleNamespace, device: torch.device) -> CodeFormer:
    vq = VQAutoencoder(
        codebook_size=int(args.codebook_size),
        base_channels=int(args.base_channels),
        num_res_blocks=int(args.num_res_blocks),
        beta=float(args.beta),
    )
    model = CodeFormer(
        vq,
        transformer_width=int(args.transformer_width),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        transformer_mlp_ratio=float(args.transformer_mlp_ratio),
    ).to(device)
    state = payload.get("model_state_dict")
    if state is None:
        raise KeyError("expected a stage=codeformer checkpoint with model_state_dict")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = sorted(path for path in input_path.iterdir() if path.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"no image files under {input_path}")
    return paths


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="stage=codeformer *_best.pth checkpoint")
    parser.add_argument("--input-path", required=True, help="One image or a directory of images")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer1-arch", choices=["cnn", "swin"], default=None, help="Default uses the architecture recorded in the checkpoint.")
    parser.add_argument("--layer1-ckpt", default="", help="Optional compatible Layer1 checkpoint override.")
    parser.add_argument("--fusion-weight", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    cli = parser.parse_args()

    checkpoint_path = resolve(cli.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("stage") != "codeformer":
        raise ValueError(f"expected stage=codeformer checkpoint, got {payload.get('stage')!r}")
    args = _checkpoint_args(payload, cli)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not cli.cpu else "cpu")
    model = build_from_checkpoint(payload, args, device)
    layer1_encoder, layer1_decoder = build_layer1(args, device)
    layer1_path = load_layer1_checkpoint(args, layer1_encoder, layer1_decoder)
    freeze_layer1(layer1_encoder, layer1_decoder)

    output_dir = resolve(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in image_paths(resolve(cli.input_path)):
        image = Image.open(source).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        hq = TF.to_tensor(image).unsqueeze(0).to(device)
        x1 = layer1_x1(layer1_encoder, layer1_decoder, hq)
        restored = model.restore(x1, fusion_weight=float(cli.fusion_weight), soft_decode=False)["output"]
        target = output_dir / f"{source.stem}_codeformer.png"
        TF.to_pil_image(restored[0].cpu()).save(target)
        print(f"saved {target}")
    print(f"Layer1={args.layer1_arch} checkpoint={layer1_path}; CodeFormer input was x1 only.")


if __name__ == "__main__":
    main()
