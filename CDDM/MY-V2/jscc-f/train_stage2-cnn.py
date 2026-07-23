from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
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
    print_epoch,
    psnr_per_image,
    recon_loss,
    resolve_path,
    seed_everything,
    setup_log_file,
    should_save_latest,
    should_validate,
    ssim_per_image,
    write_json,
)
from model import OutputsCombiner, freeze_layer1
from test_ed import CNNAnalysisEncoder, CNNBottleneckDecoder, ConvNormAct


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
    "loss_final",
    "loss_u2",
    "mse_x1",
    "psnr_x1",
    "ssim_x1",
    "mse_u2",
    "psnr_u2",
    "ssim_u2",
    "mse_final",
    "psnr_final",
    "ssim_final",
    "delta_psnr",
]


def stage2_in_ch(args: argparse.Namespace) -> int:
    return 3 if str(args.variant) == "residual_input" else 6


def use_z1_concat(args: argparse.Namespace) -> bool:
    return str(args.cnn_codec) == "z1_concat"


def stage2_z_ch(args: argparse.Namespace) -> int:
    if use_z1_concat(args):
        return int(args.z1_concat_z2_ch)
    if str(args.cnn_codec) == "compressor":
        return int(args.cnn_bottleneck_ch)
    if str(args.cnn_codec) == "no_compressor":
        return int(args.cnn_base_ch) * 16
    raise ValueError(f"unknown cnn codec {args.cnn_codec!r}")


def decoder_in_ch(args: argparse.Namespace) -> int:
    if use_z1_concat(args):
        return int(args.latent_ch) + stage2_z_ch(args)
    return stage2_z_ch(args)


def count_params(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters())


def trainable_state(module: nn.Module | None) -> str:
    if module is None:
        return "none"
    total = 0
    trainable = 0
    for param in module.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    if total == 0:
        return "no_params"
    if trainable == 0:
        return "frozen"
    if trainable == total:
        return "trainable"
    return f"partial_trainable({100.0 * trainable / float(total):.1f}%)"


def load_layer1(args: argparse.Namespace, e1: nn.Module, d1: nn.Module) -> None:
    layer1_path = args.layer1_ckpt or args.init_ckpt
    if not layer1_path:
        raise ValueError("set --layer1-ckpt or --init-ckpt for Stage2 frozen E1-D1 initialization")
    args.layer1_ckpt = layer1_path
    jsccf_io.load_layer1_compatible_checkpoint(layer1_path, e1, d1, strict=True)


def build_cnn_layer1(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module]:
    e1 = CNNAnalysisEncoder(
        base_ch=int(args.layer1_cnn_base_ch),
        bottleneck_ch=int(args.latent_ch),
        num_res=int(args.layer1_cnn_num_res),
    ).to(device)
    d1 = CNNBottleneckDecoder(
        base_ch=int(args.layer1_cnn_base_ch),
        bottleneck_ch=int(args.latent_ch),
        num_res=int(args.layer1_cnn_num_res),
        output_activation="none",
    ).to(device)
    return e1, d1


def build_cnn_encoder(args: argparse.Namespace, device: torch.device, in_chans: int) -> nn.Module:
    encoder = CNNAnalysisEncoder(
        base_ch=int(args.cnn_base_ch),
        bottleneck_ch=stage2_z_ch(args) if use_z1_concat(args) else int(args.cnn_bottleneck_ch),
        num_res=int(args.cnn_num_res),
    ).to(device)
    encoder.stem = ConvNormAct(int(in_chans), int(args.cnn_base_ch), kernel=3, stride=1).to(device)
    if str(args.cnn_codec) == "no_compressor":
        encoder.compressor = nn.Identity()
    return encoder


def build_cnn_decoder(args: argparse.Namespace, device: torch.device) -> nn.Module:
    decoder = CNNBottleneckDecoder(
        base_ch=int(args.cnn_base_ch),
        bottleneck_ch=decoder_in_ch(args),
        num_res=int(args.cnn_num_res),
        output_activation=str(args.output_activation),
    ).to(device)
    if str(args.cnn_codec) == "no_compressor":
        decoder.expander = nn.Identity()
    return decoder


def build_layer2_cnn(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, OutputsCombiner]:
    e1, d1 = build_cnn_layer1(args, device)
    e2 = build_cnn_encoder(args, device, in_chans=stage2_in_ch(args))
    d2 = build_cnn_decoder(args, device)
    combiner = OutputsCombiner().to(device)
    return e1, d1, e2, d2, combiner


def encode_tensor(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = encoder(x)
    if isinstance(out, (tuple, list)):
        return out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"encoder returned unsupported type {type(out)!r}")
    return out


def layer1_forward_cnn(e1: nn.Module, d1: nn.Module, img: torch.Tensor) -> dict[str, torch.Tensor]:
    z1 = encode_tensor(e1, img)
    x1_raw = d1(z1)
    x1 = x1_raw.clamp(0.0, 1.0)
    return {"z1": z1, "x1_raw": x1_raw, "x1": x1}


def layer2_forward_cnn(
    e1: nn.Module,
    d1: nn.Module,
    e2: nn.Module,
    d2: nn.Module,
    combiner: OutputsCombiner,
    img: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        base = layer1_forward_cnn(e1, d1, img)
    x1 = base["x1"]
    if str(args.variant) == "residual_input":
        e2_in = img - x1
    else:
        e2_in = torch.cat([img, x1], dim=1)
    z2 = encode_tensor(e2, e2_in)
    d2_in = torch.cat([base["z1"], z2], dim=1) if use_z1_concat(args) else z2
    u2_raw = d2(d2_in)
    u2 = u2_raw.clamp(0.0, 1.0)
    x2_hat = combiner(x1, u2)
    final = u2 if str(args.variant) == "no_combiner" else x2_hat
    return {
        "z1": base["z1"],
        "x1": x1,
        "z2": z2,
        "d2_in": d2_in,
        "u2_raw": u2_raw,
        "u2": u2,
        "x2_hat": x2_hat,
        "final": final,
    }


def collect_metrics(out: dict[str, torch.Tensor], imgs: torch.Tensor, m: dict) -> None:
    bsz = imgs.shape[0]
    m["mse_x1"].update(batch_metric_mean(mse_per_image(out["x1"], imgs)), bsz)
    m["psnr_x1"].update(batch_metric_mean(psnr_per_image(out["x1"], imgs)), bsz)
    m["ssim_x1"].update(batch_metric_mean(ssim_per_image(out["x1"], imgs)), bsz)
    m["mse_u2"].update(batch_metric_mean(mse_per_image(out["u2"], imgs)), bsz)
    m["psnr_u2"].update(batch_metric_mean(psnr_per_image(out["u2"], imgs)), bsz)
    m["ssim_u2"].update(batch_metric_mean(ssim_per_image(out["u2"], imgs)), bsz)
    m["mse_final"].update(batch_metric_mean(mse_per_image(out["final"], imgs)), bsz)
    psnr_final = batch_metric_mean(psnr_per_image(out["final"], imgs))
    psnr_x1 = batch_metric_mean(psnr_per_image(out["x1"], imgs))
    m["psnr_final"].update(psnr_final, bsz)
    m["ssim_final"].update(batch_metric_mean(ssim_per_image(out["final"], imgs)), bsz)
    m["delta_psnr"].update(psnr_final - psnr_x1, bsz)


@torch.no_grad()
def validate(loader, e1, d1, e2, d2, combiner, args) -> dict[str, float]:
    e1.eval()
    d1.eval()
    e2.eval()
    d2.eval()
    combiner.eval()
    device = next(e2.parameters()).device
    m = meters(METRIC_NAMES)
    max_batches = int(getattr(args, "max_val_batches", 0))
    for step, (imgs, _labels) in enumerate(loader, start=1):
        if max_batches > 0 and step > max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        out = layer2_forward_cnn(e1, d1, e2, d2, combiner, imgs, args)
        loss_final = recon_loss(out["final"], imgs)
        loss_u2 = recon_loss(out["u2_raw"], imgs)
        aux = 0.0 if args.variant == "no_combiner" else float(args.lambda_u2)
        loss = loss_final + aux * loss_u2
        bsz = imgs.shape[0]
        m["loss"].update(float(loss.item()), bsz)
        m["loss_final"].update(float(loss_final.item()), bsz)
        m["loss_u2"].update(float(loss_u2.item()), bsz)
        collect_metrics(out, imgs, m)
    return averaged(m)


def save_layer2_v2_checkpoint(
    path: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    e1: nn.Module,
    d1: nn.Module,
    e2: nn.Module,
    d2: nn.Module,
    combiner: nn.Module,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    z1 = int(args.latent_ch)
    z2 = stage2_z_ch(args)
    d2_ch = decoder_in_ch(args)
    torch.save(
        {
            "route": getattr(jsccf_io, "ROUTE", "TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256"),
            "stage": "layer2_v2_cnn",
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
            "version": str(getattr(args, "version", "")),
            "e1_state_dict": e1.state_dict(),
            "d1_state_dict": d1.state_dict(),
            "e2_state_dict": e2.state_dict(),
            "d2_state_dict": d2.state_dict(),
            "combiner_state_dict": combiner.state_dict(),
            "variant": str(args.variant),
            "stage2_codec": {
                "arch": "CNNAnalysisEncoder/CNNBottleneckDecoder",
                "mode": str(args.cnn_codec),
                "base_ch": int(args.cnn_base_ch),
                "num_res": int(args.cnn_num_res),
                "has_compressor": str(args.cnn_codec) != "no_compressor",
                "e2_in_ch": stage2_in_ch(args),
                "concat_z1": use_z1_concat(args),
                "z1_ch": z1,
                "z2_ch": z2,
                "d2_in_ch": d2_ch,
            },
            "latent": {
                "z1": [z1, int(args.latent_h), int(args.latent_w)],
                "z2": [z2, int(args.latent_h), int(args.latent_w)],
                "d2_in": [d2_ch, int(args.latent_h), int(args.latent_w)],
            },
        },
        out,
    )
    print(f"saved checkpoint: {out}", flush=True)


def checkpoint_stage(args: argparse.Namespace) -> str:
    return f"layer2_v2_{args.cnn_codec}_{args.variant}"


def print_cnn_run_header(args: argparse.Namespace, modules: dict[str, nn.Module], train_n: int, val_n: int) -> None:
    z1 = int(args.latent_ch)
    z2 = stage2_z_ch(args)
    d2_ch = decoder_in_ch(args)
    deep_ch = int(args.cnn_base_ch) * 16
    latent_shape = f"[B,{z2},{int(args.latent_h)},{int(args.latent_w)}]"
    latent_ratio = (z1 + z2) * int(args.latent_h) * int(args.latent_w) / float(3 * 256 * 256) * 100.0
    device = next(modules["E2"].parameters()).device
    print(f"=== Layer 2 v2 | CNN E2-D2 refinement | {args.cnn_codec} ===", flush=True)
    print(f"device={device} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"save_dir={resolve_path(args.save_dir)}", flush=True)
    print("实验设计", flush=True)
    print("  model=TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256", flush=True)
    print(
        f"  version={args.version} z1={z1}x{args.latent_h}x{args.latent_w} "
        f"z2={z2}x{args.latent_h}x{args.latent_w} total_latent_ratio={latent_ratio:.2f}%",
        flush=True,
    )
    print("  channel=identity power_norm=none noise=none", flush=True)
    print("  path=freeze(CNN-E1,CNN-D1); train CNN-E2/CNN-D2 plus optional combiner", flush=True)
    print("loss设计", flush=True)
    print(f"  variant={args.variant} L2=MSE(final,x)+{float(args.lambda_u2):g}*MSE(u2,x)", flush=True)
    print("模块选择", flush=True)
    print(
        f"  E1=CNNAnalysisEncoder(3->{z1}) frozen D1=CNNBottleneckDecoder({z1}->3) frozen",
        flush=True,
    )
    if use_z1_concat(args):
        print(
            f"  E2=CNNAnalysisEncoder({stage2_in_ch(args)}->{deep_ch}->{z2}) "
            f"latent={latent_shape}",
            flush=True,
        )
        print(
            f"  D2 input=concat(z1,z2) [B,{d2_ch},{args.latent_h},{args.latent_w}] "
            f"D2=CNNBottleneckDecoder({d2_ch}->{deep_ch}->3)",
            flush=True,
        )
    elif str(args.cnn_codec) == "compressor":
        print(
            f"  E2=CNNAnalysisEncoder({stage2_in_ch(args)}->{deep_ch}->{z2}) with compressor "
            f"latent={latent_shape}",
            flush=True,
        )
        print(f"  D2=CNNBottleneckDecoder({d2_ch}->{deep_ch}->3) with expander", flush=True)
    else:
        print(
            f"  E2=CNNAnalysisEncoder({stage2_in_ch(args)}->{deep_ch}) without compressor "
            f"latent={latent_shape}",
            flush=True,
        )
        print(f"  D2=CNNBottleneckDecoder({d2_ch}->3) without expander", flush=True)
    print(
        f"  layer1_cnn_base_ch={int(args.layer1_cnn_base_ch)} layer1_cnn_num_res={int(args.layer1_cnn_num_res)} "
        f"cnn_base_ch={int(args.cnn_base_ch)} cnn_num_res={int(args.cnn_num_res)}",
        flush=True,
    )
    print(f"  init_layer1_ckpt={resolve_path(args.layer1_ckpt)}", flush=True)
    print(
        "  trainable_status="
        f"E1={trainable_state(modules['E1'])} "
        f"D1={trainable_state(modules['D1'])} "
        f"E2={trainable_state(modules['E2'])} "
        f"D2={trainable_state(modules['D2'])} "
        f"combiner={trainable_state(modules['combiner'])}",
        flush=True,
    )
    print(
        "  params="
        f"E1={count_params(modules['E1'])} D1={count_params(modules['D1'])} "
        f"E2={count_params(modules['E2'])} D2={count_params(modules['D2'])} "
        f"combiner={count_params(modules['combiner'])}",
        flush=True,
    )
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}", flush=True)


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = get_loader(cfg)
    e1, d1, e2, d2, combiner = build_layer2_cnn(args, cfg.device)
    load_layer1(args, e1, d1)
    freeze_layer1(e1, d1)
    params = list(e2.parameters()) + list(d2.parameters())
    if args.variant != "no_combiner":
        params += list(combiner.parameters())
    opt = optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = -1.0
    latest_metrics: dict[str, float] = {}
    modules = {"E1": e1, "D1": d1, "E2": e2, "D2": d2, "combiner": combiner}
    print_cnn_run_header(args, modules, len(train_loader.dataset), len(val_loader.dataset))
    for epoch in range(1, int(args.epochs) + 1):
        e1.eval()
        d1.eval()
        e2.train()
        d2.train()
        combiner.train(args.variant != "no_combiner")
        m = meters(METRIC_NAMES)
        t0 = time.time()
        max_batches = int(getattr(args, "max_train_batches", 0))
        for step, (imgs, _labels) in enumerate(train_loader, start=1):
            if max_batches > 0 and step > max_batches:
                break
            imgs = imgs.to(cfg.device, non_blocking=True)
            out = layer2_forward_cnn(e1, d1, e2, d2, combiner, imgs, args)
            loss_final = recon_loss(out["final"], imgs)
            loss_u2 = recon_loss(out["u2_raw"], imgs)
            aux = 0.0 if args.variant == "no_combiner" else float(args.lambda_u2)
            loss = loss_final + aux * loss_u2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsz = imgs.shape[0]
            m["loss"].update(float(loss.item()), bsz)
            m["loss_final"].update(float(loss_final.item()), bsz)
            m["loss_u2"].update(float(loss_u2.item()), bsz)
            collect_metrics(out, imgs, m)
        metrics = averaged(m)
        latest_metrics = metrics
        print_epoch(f"layer2-v2-{args.cnn_codec}-{args.variant}", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, e1, d1, e2, d2, combiner, args)
            score = val_metrics["psnr_final"]
            latest_metrics = val_metrics
            print(f"[layer2-v2 val {epoch:03d}] {val_metrics} score=psnr_final", flush=True)
            if score > best:
                best = score
                save_layer2_v2_checkpoint(
                    jsccf_io.ckpt_path(args, checkpoint_stage(args), "best"),
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    e1=e1,
                    d1=d1,
                    e2=e2,
                    d2=d2,
                    combiner=combiner,
                )
        if should_save_latest(args, epoch):
            save_layer2_v2_checkpoint(
                jsccf_io.ckpt_path(args, checkpoint_stage(args), "latest"),
                epoch=epoch,
                args=args,
                metrics=metrics,
                e1=e1,
                d1=d1,
                e2=e2,
                d2=d2,
                combiner=combiner,
            )
    save_layer2_v2_checkpoint(
        jsccf_io.ckpt_path(args, checkpoint_stage(args), "latest"),
        epoch=int(args.epochs),
        args=args,
        metrics=latest_metrics,
        e1=e1,
        d1=d1,
        e2=e2,
        d2=d2,
        combiner=combiner,
    )


@torch.no_grad()
def smoke_shapes(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    device = torch.device("cuda:0" if (not bool(args.cpu)) and torch.cuda.is_available() else "cpu")
    e1, d1, e2, d2, combiner = build_layer2_cnn(args, device)
    e1.eval()
    d1.eval()
    e2.eval()
    d2.eval()
    combiner.eval()
    imgs = torch.rand(int(args.smoke_batch_size), 3, 256, 256, device=device)
    out = layer2_forward_cnn(e1, d1, e2, d2, combiner, imgs, args)
    expected_z1 = (int(args.smoke_batch_size), int(args.latent_ch), int(args.latent_h), int(args.latent_w))
    expected_z2 = (int(args.smoke_batch_size), stage2_z_ch(args), int(args.latent_h), int(args.latent_w))
    expected_d2_in = (int(args.smoke_batch_size), decoder_in_ch(args), int(args.latent_h), int(args.latent_w))
    expected_u2 = (int(args.smoke_batch_size), 3, 256, 256)
    print(
        f"[smoke] mode={args.cnn_codec} img={tuple(imgs.shape)} "
        f"z1={tuple(out['z1'].shape)} z2={tuple(out['z2'].shape)} "
        f"d2_in={tuple(out['d2_in'].shape)} u2={tuple(out['u2_raw'].shape)}",
        flush=True,
    )
    if tuple(out["z1"].shape) != expected_z1:
        raise RuntimeError(f"expected z1 {expected_z1}, got {tuple(out['z1'].shape)}")
    if tuple(out["z2"].shape) != expected_z2:
        raise RuntimeError(f"expected z2 {expected_z2}, got {tuple(out['z2'].shape)}")
    if tuple(out["d2_in"].shape) != expected_d2_in:
        raise RuntimeError(f"expected d2_in {expected_d2_in}, got {tuple(out['d2_in'].shape)}")
    if tuple(out["u2_raw"].shape) != expected_u2:
        raise RuntimeError(f"expected u2 {expected_u2}, got {tuple(out['u2_raw'].shape)}")


def validate_args(args: argparse.Namespace) -> None:
    check_jsccf_args(args)
    if int(args.cnn_base_ch) * 16 != 320:
        raise ValueError("Stage2 CNN high feature is defined as [B,320,16,16]; keep --cnn-base-ch 20.")
    if use_z1_concat(args) and int(args.z1_concat_z2_ch) <= 0:
        raise ValueError("--z1-concat-z2-ch must be positive.")
    if int(args.layer1_cnn_base_ch) != 16:
        raise ValueError("The default CNN layer1 checkpoint uses --layer1-cnn-base-ch 16.")
    if int(args.layer1_cnn_num_res) != 2:
        raise ValueError("The default CNN layer1 checkpoint uses --layer1-cnn-num-res 2.")
    if str(args.cnn_codec) == "compressor" and int(args.cnn_bottleneck_ch) <= 0:
        raise ValueError("--cnn-bottleneck-ch must be positive.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", type=str, default="cnn-stage2-c-4", help="Version of the JSCC-f training; affects checkpoint and log names.")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=jsccf_io.default_save_dir())
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default="")
    p.add_argument("--latent-ch", type=int, default=16)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--layer1-ckpt", type=str, default="MY-V2/jscc-f/checkpoints/jscc_f_cnn_layer1_cnn_best.pth")
    p.add_argument("--variant", type=str, default="combiner", choices=["combiner", "no_combiner", "residual_input"])
    p.add_argument("--lambda-u2", type=float, default=0.0)
    p.add_argument("--cnn-codec", type=str, default="compressor", choices=["compressor", "no_compressor", "z1_concat"])
    p.add_argument("--cnn-base-ch", type=int, default=20, help="CNN base width; 20 gives the Swin-like 320-channel deep feature.")
    p.add_argument("--cnn-bottleneck-ch", type=int, default=4, help="Compressor output channels for --cnn-codec compressor.")
    p.add_argument("--cnn-num-res", type=int, default=2)
    p.add_argument("--z1-concat-z2-ch", type=int, default=20, help="Layer2 z2 channels for --cnn-codec z1_concat.")
    p.add_argument("--layer1-cnn-base-ch", type=int, default=16, help="Frozen CNN layer1 base width.")
    p.add_argument("--layer1-cnn-num-res", type=int, default=2, help="Frozen CNN layer1 residual blocks per down/up block.")
    p.add_argument("--output-activation", type=str, default="none", choices=["none", "sigmoid", "tanh"])
    p.add_argument("--max-train-batches", type=int, default=0, help="Optional debug limit; 0 uses all train batches.")
    p.add_argument("--max-val-batches", type=int, default=0, help="Optional debug limit; 0 uses all validation batches.")
    p.add_argument("--smoke-shapes", action="store_true", help="Run random E2/D2 shape smoke and exit.")
    p.add_argument("--smoke-batch-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.stage = "layer2_v2"
    validate_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if args.smoke_shapes:
        smoke_shapes(args)
        return
    if not args.log_file:
        args.log_file = str(
            Path(resolve_path(args.save_dir))
            / f"layer2_v2_{args.cnn_codec}_{args.variant}_jscc_f_{jsccf_io.safe_artifact_name(args.version)}.log"
        )
    setup_log_file(args.log_file)
    write_json(
        Path(resolve_path(args.save_dir)) / f"layer2_v2_{args.cnn_codec}_{args.variant}_args.json",
        vars(args),
    )
    train(args)


if __name__ == "__main__":
    main()
