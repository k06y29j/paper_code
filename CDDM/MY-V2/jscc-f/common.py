from __future__ import annotations

import builtins
import json
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CDDM_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.meshgrid.*", category=UserWarning)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


@dataclass
class JSCCFConfig:
    C: int
    batch_size: int
    test_batch: int
    num_workers: int
    val_num_workers: int
    train_data_dir: str
    test_data_dir: str
    CUDA: bool = True
    dataset: str = "DIV2K"
    loss_function: str = "MSE"
    channel_type: str = "none"
    SNRs: float = 0.0
    image_dims: tuple[int, int, int] = (3, 256, 256)
    encoder_in_chans: int = 3
    pin_memory: bool = True
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        self.device = torch.device("cuda:0" if self.CUDA and torch.cuda.is_available() else "cpu")
        self.encoder_kwargs = dict(
            img_size=(self.image_dims[1], self.image_dims[2]),
            patch_size=2,
            in_chans=int(self.encoder_in_chans),
            embed_dims=[128, 192, 256, 320],
            depths=[2, 2, 6, 2],
            num_heads=[4, 6, 8, 10],
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=torch.nn.LayerNorm,
            patch_norm=True,
        )
        self.decoder_kwargs = dict(
            img_size=(self.image_dims[1], self.image_dims[2]),
            embed_dims=[320, 256, 192, 128],
            depths=[2, 6, 2, 2],
            num_heads=[10, 8, 6, 4],
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=torch.nn.LayerNorm,
            patch_norm=True,
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else CDDM_ROOT / p)


def setup_log_file(path: str | None) -> object | None:
    if not path:
        return None
    abs_path = Path(resolve_path(path))
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== JSCC-f @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def write_json(path: str | Path, payload: dict) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def freeze_module(module: torch.nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(trainable)
    module.train(bool(trainable))


def clamp_img(x: torch.Tensor) -> torch.Tensor:
    return x.float().clamp(0.0, 1.0)


def recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon.float(), target.float(), reduction="mean") 
    # return F.l1_loss(recon.float(), target.float(), reduction="mean")


def mse_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (clamp_img(x_hat) - x.float()).square().mean(dim=(1, 2, 3))


def psnr_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse = mse_per_image(x_hat, x).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def ssim_per_image(x_hat: torch.Tensor, x: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    x_hat = clamp_img(x_hat)
    x = x.float()
    channels = int(x.shape[1])
    pad = int(window_size) // 2
    weight = torch.ones(channels, 1, window_size, window_size, device=x.device, dtype=x.dtype)
    weight = weight / float(window_size * window_size)
    mu_x = F.conv2d(x, weight, padding=pad, groups=channels)
    mu_y = F.conv2d(x_hat, weight, padding=pad, groups=channels)
    sigma_x = F.conv2d(x * x, weight, padding=pad, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(x_hat * x_hat, weight, padding=pad, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(x * x_hat, weight, padding=pad, groups=channels) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim.mean(dim=(1, 2, 3))


def batch_metric_mean(values: torch.Tensor) -> float:
    return float(values.float().mean().item())


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.6g}" for k, v in sorted(metrics.items()))


def print_epoch(stage: str, epoch: int, total: int, metrics: dict[str, float], elapsed: float) -> None:
    print(f"[{stage} epoch {epoch:03d}/{total:03d}] {format_metrics(metrics)} time={elapsed:.1f}s", flush=True)


def should_validate(args, epoch: int) -> bool:
    return int(args.val_every) > 0 and (epoch % int(args.val_every) == 0 or epoch == int(args.epochs))


def should_save_latest(args, epoch: int) -> bool:
    return int(args.latest_every) > 0 and epoch % int(args.latest_every) == 0


def meters(names: list[str]) -> dict[str, AverageMeter]:
    return {name: AverageMeter() for name in names}


def averaged(m: dict[str, AverageMeter]) -> dict[str, float]:
    return {k: v.avg for k, v in m.items()}


def check_jsccf_args(args) -> None:
    if int(args.c1_ch) != int(args.latent_ch):
        raise ValueError("JSCC-f layer1 uses --latent-ch as z1 channels; keep --c1-ch equal to --latent-ch.")
    if int(args.latent_h) != 16 or int(args.latent_w) != 16:
        raise ValueError("The Swin JSCC encoder is expected to produce 16x16 latents for 256x256 crops.")


def z1_ch(args) -> int:
    return int(args.latent_ch)


def z2_ch(_args) -> int:
    return 20


def total_latent_ch(args) -> int:
    return z1_ch(args) + z2_ch(args)


def _module_train_state(module: torch.nn.Module | None, default: str) -> str:
    if module is None:
        return default
    total = 0
    trainable = 0
    for param in module.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    if total == 0:
        return "no_params"
    if trainable == 0:
        return "frozen"
    if trainable == total:
        return "trainable"
    return f"partial_trainable({100.0 * trainable / float(total):.1f}%)"


def _module_state(modules: dict[str, torch.nn.Module] | None, name: str, default: str) -> str:
    return _module_train_state(modules.get(name) if modules else None, default)


def _print_stage3_loss_design(args) -> None:
    recon = str(getattr(args, "recon", "recon_x2"))
    if recon == "recon_u2":
        rec_term = "lambda_u2_teacher_eff*MSE(u2,layer2_u2)"
    else:
        rec_term = "MSE(final_oracle,x)"
    print(f"  recon={recon} L_rec={rec_term}")
    print(
        "  "
        f"L=L_rec+{float(args.lambda_vq):g}*(codebook+{float(args.beta_commit):g}*commit)"
        "+lambda_index_eff*CE(IndexNet(z1),q2_index)"
    )
    print(
        "  "
        f"lambda_u2_teacher_eff={float(getattr(args, 'lambda_u2_teacher', 0.0)):g} "
        "only when teacher is loaded and u2_teacher_phases in {finetune,all}"
    )


def print_run_header(
    args,
    title: str,
    train_n: int,
    val_n: int,
    modules: dict[str, torch.nn.Module] | None = None,
) -> None:
    z1 = z1_ch(args)
    z2 = z2_ch(args)
    total = total_latent_ch(args)
    stage = str(getattr(args, "stage", "layer1"))
    latent_ratio = total * int(args.latent_h) * int(args.latent_w) / float(3 * 256 * 256) * 100.0
    print(f"=== {title} ===")
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"save_dir={resolve_path(args.save_dir)}")
    if getattr(args, "init_ckpt", ""):
        print(f"init_ckpt={resolve_path(args.init_ckpt)}")
    print("实验设计")
    print("  model=TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256")
    print(f"  version={args.version} z1={z1}x{args.latent_h}x{args.latent_w} z2={z2}x{args.latent_h}x{args.latent_w} total_latent_ratio={latent_ratio:.2f}%")
    print("  channel=identity power_norm=none noise=none")
    print("loss设计")
    if stage == "layer1":
        print("  L1=MSE(D1(E1(x)), x)")
    elif stage == "stage3":
        _print_stage3_loss_design(args)
        guard = "off" if bool(getattr(args, "disable_best_guard", False)) else f"psnr_oracle>=psnr_x1+{float(getattr(args, 'best_psnr_margin', 0.0)):g}"
        init_method = str(getattr(args, "init_codebook_method", "none"))
        print(
            f"  lr_codec={float(getattr(args, 'lr', 0.0)):g} "
            f"lr_quantizer={float(getattr(args, 'lr_simvq', 0.0)):g} "
            f"lambda_u2_teacher={float(getattr(args, 'lambda_u2_teacher', 0.0)):g} "
            f"u2_teacher_phases={getattr(args, 'u2_teacher_phases', 'finetune')} "
            f"train_indexnet={bool(getattr(args, 'train_indexnet', False))} "
            f"lambda_index={float(args.lambda_index):g} "
            f"index_warmup_epochs={int(args.warmup_index_epochs)} best_guard={guard}"
        )
        if init_method != "none":
            print(
                "  "
                f"codebook_init={init_method} samples={int(getattr(args, 'codebook_init_samples', 0))} "
                f"kmeans_iters={int(getattr(args, 'kmeans_iters', 0))} "
                f"eval_init_only={bool(getattr(args, 'eval_init_only', False))}"
            )
    else:
        print(
            "  "
            f"variant={args.variant} L2=MSE(final,x)+{float(args.lambda_u2):g}*MSE(u2,x)"
        )
    print("模块选择")
    print(f"  E1=JSCC_encoder(3->{z1}) D1=JSCC_decoder({z1}->3)")
    if stage != "layer1":
        e2_in = 3 if args.variant == "residual_input" else 6
        print(f"  E2=JSCC_encoder({e2_in}->{z2}) D2=JSCC_decoder({z2}->3) combiner=Conv3x3-48-PReLU-Conv3x3-Sigmoid")
        print(f"  init_layer1_ckpt={resolve_path(args.layer1_ckpt)}")
    if stage == "stage3":
        quantizer = str(getattr(args, "quantizer", "simvq"))
        if quantizer == "vq":
            print(f"  quantizer=VQ K{int(args.vq_k)}xD{z2} trainable_embedding=True")
            print(f"  IndexNet=Conv2d(z1->{int(args.vq_k)}) hidden={int(args.index_hidden)} depth={int(args.index_depth)}")
        elif quantizer == "fsq":
            fsq_levels = str(getattr(args, "fsq_levels", "15")).replace("x", ",")
            fsq_parts = [part.strip() for part in fsq_levels.split(",") if part.strip()]
            fsq_desc = f"{fsq_parts[0]}x{z2}" if len(fsq_parts) == 1 else ",".join(fsq_parts)
            max_level = max(int(part) for part in fsq_parts)
            print(
                f"  quantizer=ScalarFSQ levels={fsq_desc} index_shape=[B,{z2},{int(args.latent_h)},{int(args.latent_w)}] "
                f"stats_init={bool(getattr(args, 'fsq_init_stats', True))} train_affine={not bool(getattr(args, 'fsq_freeze_affine', False))}"
            )
            print(f"  IndexNet=ScalarFSQIndexNet(z1->[B,{z2},{int(args.latent_h)},{int(args.latent_w)},{max_level}]) hidden={int(args.index_hidden)} depth={int(args.index_depth)}")
        elif quantizer == "cvq":
            print(f"  quantizer=FullMapCVQ codebook=[{int(args.cvq_k)},{int(args.latent_h)},{int(args.latent_w)}] tokens={z2} index_shape=[B,{z2}]")
            print(f"  IndexNet=FullMapIndexNet(z1->[B,{z2},{int(args.cvq_k)}]) hidden={int(args.index_hidden)} depth={int(args.index_depth)} heads={int(args.index_heads)}")
        elif quantizer == "fullmap_simvq":
            print(f"  quantizer=FullMapSimVQ base_codebook=[{int(args.fullmap_simvq_k)},{int(args.latent_h)},{int(args.latent_w)}] tokens={z2} index_shape=[B,{z2}] frozen_codebook={not bool(args.fullmap_simvq_train_codebook)} trainable_linear=True")
            print(f"  IndexNet=FullMapIndexNet(z1->[B,{z2},{int(args.fullmap_simvq_k)}]) hidden={int(args.index_hidden)} depth={int(args.index_depth)} heads={int(args.index_heads)}")
        else:
            print(f"  quantizer=SimVQ K{int(args.simvq_k)}xD{z2} frozen_embedding={not bool(args.simvq_train_codebook)} trainable_linear=True")
            print(f"  IndexNet=Conv2d(z1->{int(args.simvq_k)}) hidden={int(args.index_hidden)} depth={int(args.index_depth)}")
        print(
            "  trainable_status="
            f"E1={_module_state(modules, 'E1', 'frozen')} "
            f"D1={_module_state(modules, 'D1', 'frozen')} "
            f"E2={_module_state(modules, 'E2', 'trainable')} "
            f"D2={_module_state(modules, 'D2', 'trainable')} "
            f"combiner={_module_state(modules, 'combiner', 'trainable')} "
            f"quantizer={_module_state(modules, 'quantizer', 'trainable')} "
            f"IndexNet={_module_state(modules, 'IndexNet', 'trainable' if bool(getattr(args, 'train_indexnet', False)) else 'frozen')}"
        )
        if getattr(args, "layer2_ckpt", ""):
            print(f"  init_layer2_ckpt={resolve_path(args.layer2_ckpt)}")
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}")
