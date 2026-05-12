#!/usr/bin/env python
"""评估「语义编解码 + 线性信道编解码」全链路：DIV2K 验证集图像 PSNR / MSE。

链路：image → SemanticEncoder → ChannelEncoder → ChannelDecoder → SemanticDecoder → image

- 语义权重默认：``checkpoints-val/sc/sc_encoder_div2k_c16.pth``、``sc_decoder_div2k_c16.pth``（与 ``train_cc.py`` 一致）。
- 信道权重：用 ``--model``（1/2/3）与 ``--c`` 在 ``checkpoints-val/cc/model{i}/`` 下查找：

  1. **合并文件**（``train_cc.py`` ``save_codec``）：``cc_div2k_c16to{c}_best.pth`` / ``cc_div2k_c16to{c}.pth``
  2. **拆分文件**（与 ``eval_all.py`` 一致）：``cc_encoder_div2k_c16to{c}.pth`` + ``cc_decoder_div2k_c16to{c}.pth``

- **PSNR**：逐图 PSNR 后算术平均 ΣPSNR_i/N（与 ``eval_sc_div2k_psnr.py`` 一致）。
- **MSE**：整份验证集所有像素上 ``mean((x_hat - x)²)``（标量）。

用法::

  python test/eval_cc_psnr.py --model 3 --c 12
  python test/eval_cc_psnr.py --model 1 --c 4 --cc-root /path/to/checkpoints-val/cc
  python test/eval_cc_psnr.py --cc-ckpt /path/to/custom_codec.pth
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import SystemConfig, get_div2k_config  # noqa: E402
from src.cddm_mimo_ddnm.datasets import DIV2KDataset  # noqa: E402
from src.cddm_mimo_ddnm.modules.semantic_codec import SemanticDecoder, SemanticEncoder  # noqa: E402
from train.train_cc import CCSystem, LinearChannelCodec  # noqa: E402


MODEL_DIR = {1: "model1", 2: "model2", 3: "model3"}


def _load_obj(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _infer_use_vae_encoder(sd: dict) -> bool:
    for k in sd:
        if "vae_proj" in k:
            return True
    return False


def load_semantic_encoder(path: str, enc: SemanticEncoder) -> int:
    obj = _load_obj(path)
    sd = obj.get("state_dict", obj)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    emb = int(obj.get("embed_dim", 16))
    print(f"  semantic_encoder <- {path}")
    if missing:
        print(f"    missing ({len(missing)}): {missing[:3]}{' ...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"    unexpected ({len(unexpected)}): {unexpected[:3]}{' ...' if len(unexpected) > 3 else ''}")
    return emb


def load_semantic_decoder(path: str, dec: SemanticDecoder) -> None:
    obj = _load_obj(path)
    sd = obj.get("state_dict", obj)
    missing, unexpected = dec.load_state_dict(sd, strict=False)
    print(f"  semantic_decoder <- {path}")
    if missing:
        print(f"    missing ({len(missing)}): {missing[:3]}{' ...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"    unexpected ({len(unexpected)}): {unexpected[:3]}{' ...' if len(unexpected) > 3 else ''}")


def resolve_cc_checkpoint(
    *,
    cc_root: str,
    model_id: int,
    c_out: int,
    dataset: str,
    sem_c: int,
    explicit: str | None,
) -> str | tuple[str, str]:
    """返回合并 .pth 路径，或 (encoder.pth, decoder.pth) 拆分路径。"""
    if explicit:
        p = explicit if os.path.isabs(explicit) else os.path.join(PROJECT_ROOT, explicit)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"--cc-ckpt 路径不存在: {p}")
        return p

    sub = MODEL_DIR.get(model_id)
    if sub is None:
        raise ValueError(f"--model must be 1, 2 or 3; got {model_id}")

    base = os.path.join(cc_root, sub)
    stem = f"cc_{dataset}_c{sem_c}to{c_out}"
    candidates: list[str] = []
    if model_id == 3:
        candidates.append(os.path.join(base, f"{stem}_best.pth"))
    candidates.append(os.path.join(base, f"{stem}.pth"))

    tried: list[str] = list(candidates)
    for p in candidates:
        if os.path.isfile(p):
            return p

    enc_sp = os.path.join(base, f"cc_encoder_{dataset}_c{sem_c}to{c_out}.pth")
    dec_sp = os.path.join(base, f"cc_decoder_{dataset}_c{sem_c}to{c_out}.pth")
    tried.extend([enc_sp, dec_sp])
    if os.path.isfile(enc_sp) and os.path.isfile(dec_sp):
        return (enc_sp, dec_sp)

    raise FileNotFoundError(
        f"未找到信道权重。已尝试合并：{candidates}；拆分：{enc_sp} , {dec_sp}。"
        f" 目录：{base}。也可用 ``--cc-ckpt`` 指定合并文件。"
    )


def _remap_layer_sd(sd: dict, layer_name: str) -> dict:
    """单层 Conv 的 state_dict → ``LinearChannelCodec`` 的 ``{layer_name}.weight``。"""
    prefixed = {k: v for k, v in sd.items() if k.startswith(f"{layer_name}.")}
    if prefixed:
        return prefixed
    out: dict = {}
    for k, v in sd.items():
        if k in ("weight", "bias"):
            out[f"{layer_name}.{k}"] = v
    return out


def load_linear_codec(source: str | tuple[str, str], device: torch.device) -> LinearChannelCodec:
    if isinstance(source, tuple):
        enc_path, dec_path = source
        eobj = _load_obj(enc_path)
        dobj = _load_obj(dec_path)
        sd_e = eobj.get("state_dict", eobj)
        sd_d = dobj.get("state_dict", dobj)
        if not isinstance(sd_e, dict) or not isinstance(sd_d, dict):
            raise TypeError("encoder/decoder checkpoint 须含 state_dict")
        merged = {}
        merged.update(_remap_layer_sd(sd_e, "encoder"))
        merged.update(_remap_layer_sd(sd_d, "decoder"))
        if "encoder.weight" not in merged or "decoder.weight" not in merged:
            raise KeyError(f"无法从拆分权重组装 encoder/decoder，keys 示例: {list(sd_e.keys())}")
        in_c = int(
            eobj.get("in_channels", merged["encoder.weight"].shape[1])
        )
        out_c = int(
            eobj.get("out_channels", merged["encoder.weight"].shape[0])
        )
        codec = LinearChannelCodec(in_channels=in_c, out_channels=out_c).to(device)
        codec.load_state_dict(merged, strict=True)
        print(f"  channel codec (split) <-")
        print(f"    encoder: {enc_path}")
        print(f"    decoder: {dec_path}")
        print(f"    in_channels={in_c}  out_channels={out_c}  ratio={out_c / in_c:.4f}")
        return codec

    path = source
    obj = _load_obj(path)
    sd = obj.get("state_dict", obj)
    in_c = int(obj.get("in_channels", sd["encoder.weight"].shape[1]))
    out_c = int(obj.get("out_channels", sd["encoder.weight"].shape[0]))
    codec = LinearChannelCodec(in_channels=in_c, out_channels=out_c).to(device)
    codec.load_state_dict(sd, strict=True)
    print(f"  channel codec <- {path}")
    print(f"    in_channels={in_c}  out_channels={out_c}  ratio={out_c / in_c:.4f}")
    if "metrics" in obj and obj["metrics"]:
        print(f"    ckpt.metrics: {obj['metrics']}")
    return codec


def _psnr_batch_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse_img = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3)).clamp(min=1e-12)
    return (10.0 * torch.log10(1.0 / mse_img)).detach().cpu().double()


def _parse_amp(s: str) -> tuple[bool, torch.dtype]:
    if s == "none":
        return False, torch.float32
    if s == "bfloat16":
        return True, torch.bfloat16
    if s == "float16":
        return True, torch.float16
    raise ValueError(f"Unknown --amp-dtype={s}")


@torch.no_grad()
def evaluate(
    system: CCSystem,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float, float, int]:
    """返回 (avg_psnr ΣPSNR_i/N, global_mse, feat_mse, n_img)。"""
    system.eval()
    per_blocks: list[torch.Tensor] = []
    sum_sq = 0.0
    n_pix = 0
    sum_feat = 0.0
    n_feat = 0

    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)
    nb = device.type == "cuda"
    autocast_cm = (
        torch.autocast("cuda", **autocast_kw)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    for batch in loader:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device, non_blocking=nb)
        with autocast_cm:
            x_hat, z_sem, z_cd = system(imgs)
        xf = x_hat.float().clamp(0, 1)
        tf = imgs.float()
        per_blocks.append(_psnr_batch_per_image(xf, tf))
        diff = xf - tf
        sum_sq += (diff ** 2).sum().item()
        n_pix += diff.numel()
        sum_feat += F.mse_loss(z_cd.float(), z_sem.float(), reduction="sum").item()
        n_feat += z_sem.numel()

    per_image = torch.cat(per_blocks)
    n_img = per_image.shape[0]
    avg_psnr = float(per_image.mean().item())
    global_mse = sum_sq / max(n_pix, 1)
    feat_mse = sum_feat / max(n_feat, 1)
    return avg_psnr, global_mse, feat_mse, n_img


def main() -> None:
    vd = "/workspace/yongjia/datasets/DIV2K"
    p = argparse.ArgumentParser(description="SC + CC 全链路 PSNR / MSE（DIV2K valid）")
    p.add_argument("--model", type=int, default=3, choices=[1, 2, 3], help="对应 checkpoints-val/cc/model{i}")
    p.add_argument(
        "--c",
        "--out-channels",
        "--out_channels",
        type=int,
        default=12,
        dest="c_out",
        help="信道编码器输出通道数（16→c）",
    )
    p.add_argument(
        "--cc-root",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints-val", "cc"),
        help="信道权重根目录（其下含 model1/ model2/ model3/）",
    )
    p.add_argument(
        "--cc-ckpt",
        type=str,
        default=None,
        help="直接指定信道 .pth（设置后忽略 --model 路径解析）",
    )
    p.add_argument("--dataset", type=str, default="div2k", help="用于拼文件名：cc_{dataset}_c16to{c}")
    p.add_argument(
        "--sc-encoder",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints-val/sc/sc_encoder_div2k_c16.pth"),
    )
    p.add_argument(
        "--sc-decoder",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints-val/sc/sc_decoder_div2k_c16.pth"),
    )
    p.add_argument("--valid-dir", type=str, default=vd)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "none"],
    )
    p.add_argument("--sem-c", type=int, default=16, help="语义瓶颈维，应与文件名 c{sem_c}to{c} 一致")
    args = p.parse_args()

    if not (1 <= args.sem_c <= 512):
        raise SystemExit("--sem-c 非法")
    if args.c_out >= args.sem_c:
        raise SystemExit(f"--c / --out-channels 必须 < 语义瓶颈 --sem-c={args.sem_c}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled, amp_dtype = _parse_amp(args.amp_dtype)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

    cfg: SystemConfig = get_div2k_config()
    cfg.semantic.embed_dim = int(args.sem_c)
    sc = cfg.semantic

    enc_hint = _load_obj(os.path.abspath(args.sc_encoder))
    inferred_vae = _infer_use_vae_encoder(enc_hint.get("state_dict", {}))
    cfg.semantic.use_vae = inferred_vae

    sc_enc = SemanticEncoder(
        in_channels=sc.image_channels,
        embed_dim=sc.embed_dim,
        patch_size=sc.patch_size,
        num_heads=sc.num_heads,
        window_size=sc.window_size,
        num_blocks=sc.num_swin_blocks,
        stage_embed_dims=sc.stage_embed_dims,
        stage_depths=sc.stage_depths,
        stage_num_heads=sc.stage_num_heads,
        stem_stride=sc.stem_stride,
        stage_downsample=sc.stage_downsample,
        use_vae=sc.use_vae,
    ).to(device)
    sc_dec = SemanticDecoder(
        out_channels=sc.image_channels,
        embed_dim=sc.embed_dim,
        patch_size=sc.patch_size,
        num_heads=sc.num_heads,
        window_size=sc.window_size,
        num_refine_blocks=sc.num_decoder_refine_blocks,
        stage_embed_dims=sc.stage_embed_dims,
        stage_depths=sc.stage_depths,
        stage_num_heads=sc.stage_num_heads,
        stem_stride=sc.stem_stride,
        stage_downsample=sc.stage_downsample,
    ).to(device)

    emb_loaded = load_semantic_encoder(os.path.abspath(args.sc_encoder), sc_enc)
    load_semantic_decoder(os.path.abspath(args.sc_decoder), sc_dec)
    if emb_loaded != args.sem_c:
        print(
            f"  [WARN] checkpoint embed_dim={emb_loaded} 与 --sem-c={args.sem_c} 不一致；"
            "请以权重为准重新设 --sem-c 或检查文件。"
        )

    for m in (sc_enc, sc_dec):
        m.eval()
        for p_ in m.parameters():
            p_.requires_grad = False

    cc_resolved = resolve_cc_checkpoint(
        cc_root=os.path.abspath(args.cc_root),
        model_id=args.model,
        c_out=args.c_out,
        dataset=args.dataset,
        sem_c=args.sem_c,
        explicit=args.cc_ckpt,
    )
    codec = load_linear_codec(cc_resolved, device)

    if codec.in_channels != args.sem_c or codec.out_channels != args.c_out:
        print(
            f"  [WARN] codec  shape in={codec.in_channels} out={codec.out_channels}，"
            f"与 --sem-c={args.sem_c} --c={args.c_out} 不一致；以权重为准。"
        )

    system = CCSystem(sc_enc, sc_dec, codec).to(device)

    root = os.path.abspath(args.valid_dir.rstrip(os.sep))
    ds = DIV2KDataset(root, crop_size=args.crop_size, split="valid")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )

    avg_psnr, global_mse, feat_mse, n_img = evaluate(system, loader, device, amp_enabled, amp_dtype)

    print("=" * 72)
    print(f"  model={args.model}  ({MODEL_DIR[args.model]})  c_out={args.c_out}")
    if isinstance(cc_resolved, tuple):
        print(f"  channel_encoder: {cc_resolved[0]}")
        print(f"  channel_decoder: {cc_resolved[1]}")
    else:
        print(f"  channel_ckpt: {cc_resolved}")
    print(f"  DIV2K valid 图像数: {n_img}")
    print(
        f"  PSNR (各图算术平均 ΣPSNR_i/N): {avg_psnr:.6f} dB"
    )
    print(f"  MSE (全验证集像素 mean((x̂-x)²)):     {global_mse:.8e}")
    print(f"  特征 MSE mean((z_cd-z_sem)²) 逐元素: {feat_mse:.8e}")
    print("=" * 72)


if __name__ == "__main__":
    main()
