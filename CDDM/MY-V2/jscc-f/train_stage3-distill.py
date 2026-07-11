from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
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
    freeze_module,
    meters,
    mse_per_image,
    print_epoch,
    psnr_per_image,
    resolve_path,
    seed_everything,
    setup_log_file,
    should_save_latest,
    should_validate,
    ssim_per_image,
    total_latent_ch,
    write_json,
    z1_ch,
    z2_ch,
)
from model import build_layer2, layer1_forward


def load_local_io():
    spec = importlib.util.spec_from_file_location("jsccf_io", THIS_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jsccf_io = load_local_io()


LOSS_NAMES = ["loss", "loss_z", "loss_u", "loss_final", "loss_t"]
METRIC_NAMES = LOSS_NAMES + [
    "mse_z2",
    "mse_u2",
    "mse_x1",
    "mse_student",
    "mse_teacher",
    "psnr_x1",
    "psnr_student",
    "psnr_teacher",
    "ssim_x1",
    "ssim_student",
    "ssim_teacher",
    "delta_student",
    "delta_teacher",
    "gap_to_teacher",
]


# class StudentResBlock(nn.Module):
#     def __init__(self, channels: int) -> None:
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1),
#             nn.PReLU(),
#             nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1),
#         )
#         self.act = nn.PReLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.act(x + self.net(x))


# class StudentZ2Predictor(nn.Module):
#     def __init__(
#         self,
#         in_ch: int = 19,
#         out_ch: int = 20,
#         hidden: int = 128,
#         blocks: int = 6,
#     ) -> None:
#         super().__init__()
#         hidden = int(hidden)
#         layers: list[nn.Module] = [
#             nn.Conv2d(int(in_ch), hidden, kernel_size=3, padding=1),
#             nn.PReLU(),
#         ]
#         for _idx in range(max(1, int(blocks))):
#             layers.append(StudentResBlock(hidden))
#         layers.append(nn.Conv2d(hidden, int(out_ch), kernel_size=3, padding=1))
#         self.net = nn.Sequential(*layers)

#     def forward(self, z1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
#         x1_down = F.interpolate(x1, size=tuple(z1.shape[-2:]), mode="bilinear", align_corners=False)
#         return self.net(torch.cat([z1, x1_down], dim=1))

def _best_group_count(channels: int, max_groups: int = 32) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if g <= max_groups and channels % g == 0:
            return g
    return 1


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B,H,W,C] -> LN -> [B,C,H,W]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, int(channels) // int(reduction))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(channels), hidden, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(hidden, int(channels), kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class GatedResBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        channels = int(channels)
        mid = channels * int(expansion)
        groups = _best_group_count(channels)

        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=3, padding=1)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(mid, channels * 2, kernel_size=3, padding=1)
        self.se = SEBlock(channels)

        # residual scale，防止强模型一开始破坏 teacher D2/Combiner 的输入分布
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(self.conv1(h))
        h = self.conv2(h)

        # gated residual: a * sigmoid(g)
        a, g = h.chunk(2, dim=1)
        h = a * torch.sigmoid(g)
        h = self.se(h)

        return x + self.res_scale * h


class LatentSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int = 6,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        heads = int(heads)
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

        self.attn_scale = nn.Parameter(torch.tensor(0.1))
        self.mlp_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # [B,HW,C]

        y = self.norm1(tokens)
        y, _ = self.attn(y, y, y, need_weights=False)
        tokens = tokens + self.attn_scale * y

        y = self.norm2(tokens)
        y = self.mlp(y)
        tokens = tokens + self.mlp_scale * y

        return tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()


class StudentZ2Predictor(nn.Module):
    """
    Stronger decoder-side z2 predictor.

    Input:
        z1: [B,16,16,16]
        x1: [B,3,256,256]

    Output:
        z2_hat: [B,20,16,16]
    """

    def __init__(
        self,
        z1_ch: int = 16,
        x1_ch: int = 3,
        out_ch: int = 20,
        hidden: int = 192,
        blocks: int = 8,
        attn_every: int = 2,
        heads: int = 6,
    ) -> None:
        super().__init__()
        hidden = int(hidden)

        self.z1_stem = nn.Sequential(
            nn.Conv2d(int(z1_ch), hidden, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        )

        self.x1_stem = nn.Sequential(
            nn.Conv2d(int(x1_ch), hidden // 2, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, padding=1),
        )

        # 融合 z1 feature、x1 feature、差分、乘积，增强条件建模能力
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        )

        body: list[nn.Module] = []
        for i in range(int(blocks)):
            body.append(GatedResBlock(hidden, expansion=2))
            if int(attn_every) > 0 and (i + 1) % int(attn_every) == 0:
                body.append(LatentSelfAttentionBlock(hidden, heads=int(heads), mlp_ratio=2.0))
        self.body = nn.Sequential(*body)

        self.out_norm = LayerNorm2d(hidden)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden, int(out_ch), kernel_size=3, padding=1),
        )

        # 稳定初始化：不要让初始 z2_hat 过大
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, z1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        x1_down = F.interpolate(
            x1,
            size=tuple(z1.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )

        fz = self.z1_stem(z1)
        fx = self.x1_stem(x1_down)

        fused = torch.cat(
            [
                fz,
                fx,
                fz - fx,
                fz * fx,
            ],
            dim=1,
        )

        h = self.fuse(fused)
        h = self.body(h)
        h = self.out_norm(h)
        z2_hat = self.head(h)
        return z2_hat


def stage3_v2_name(_args: argparse.Namespace) -> str:
    return "stage3_v2_student_z2"


def trainable_params(module: nn.Module) -> list[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def load_teacher_checkpoint(args: argparse.Namespace, e1: nn.Module, d1: nn.Module, e2: nn.Module, d2: nn.Module, combiner: nn.Module) -> None:
    if not args.layer2_ckpt:
        raise ValueError("Stage3-v2 requires --layer2-ckpt from the continuous layer2 teacher.")
    ckpt = jsccf_io.load_checkpoint(args.layer2_ckpt)
    jsccf_io.load_state(e1, ckpt["e1_state_dict"], "E1", strict=True)
    jsccf_io.load_state(d1, ckpt["d1_state_dict"], "D1", strict=True)
    jsccf_io.load_state(e2, ckpt["e2_state_dict"], "E2_teacher", strict=True)
    jsccf_io.load_state(d2, ckpt["d2_state_dict"], "D2_teacher", strict=True)
    jsccf_io.load_state(combiner, ckpt["combiner_state_dict"], "Combiner_teacher", strict=True)


def freeze_teacher(e1: nn.Module, d1: nn.Module, e2: nn.Module, d2: nn.Module, combiner: nn.Module) -> None:
    for module in [e1, d1, e2, d2, combiner]:
        freeze_module(module, trainable=False)
        module.eval()


def teacher_forward(e1: nn.Module, d1: nn.Module, e2: nn.Module, d2: nn.Module, combiner: nn.Module, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        base = layer1_forward(e1, d1, imgs)
        z1 = base["z1"]
        x1 = base["x1"]
        z2_t, _ = e2(torch.cat([imgs, x1], dim=1))
        u2_t_raw = d2(torch.cat([z1, z2_t], dim=1))
        u2_t = u2_t_raw.clamp(0.0, 1.0)
        final_t = combiner(x1, u2_t)
    return {
        "z1": z1,
        "x1": x1,
        "z2_t": z2_t,
        "u2_t_raw": u2_t_raw,
        "u2_t": u2_t,
        "final_t": final_t,
    }


def student_forward(student: StudentZ2Predictor, d2: nn.Module, combiner: nn.Module, teacher: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    z2_hat = student(teacher["z1"], teacher["x1"])
    u2_s_raw = d2(torch.cat([teacher["z1"], z2_hat], dim=1))
    u2_s = u2_s_raw.clamp(0.0, 1.0)
    final_s = combiner(teacher["x1"], u2_s)
    return {
        "z2_hat": z2_hat,
        "u2_s_raw": u2_s_raw,
        "u2_s": u2_s,
        "final_s": final_s,
    }


def compute_losses(teacher: dict[str, torch.Tensor], student_out: dict[str, torch.Tensor], imgs: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    loss_z = F.mse_loss(student_out["z2_hat"].float(), teacher["z2_t"].float(), reduction="mean")
    loss_u = F.mse_loss(student_out["u2_s"].float(), teacher["u2_t"].float(), reduction="mean")
    loss_final = F.mse_loss(student_out["final_s"].float(), imgs.float(), reduction="mean")
    loss_t = F.mse_loss(student_out["final_s"].float(), teacher["final_t"].float(), reduction="mean")
    loss = (
        float(args.lambda_z) * loss_z
        + float(args.lambda_u) * loss_u
        + float(args.lambda_final) * loss_final
        + float(args.lambda_t) * loss_t
    )
    return {
        "loss": loss,
        "loss_z": loss_z,
        "loss_u": loss_u,
        "loss_final": loss_final,
        "loss_t": loss_t,
    }


def update_metrics(
    m: dict,
    imgs: torch.Tensor,
    teacher: dict[str, torch.Tensor],
    student_out: dict[str, torch.Tensor],
    losses: dict[str, torch.Tensor],
) -> None:
    bsz = int(imgs.shape[0])
    for name in LOSS_NAMES:
        m[name].update(float(losses[name].item()), bsz)
    m["mse_z2"].update(float(F.mse_loss(student_out["z2_hat"].detach().float(), teacher["z2_t"].float()).item()), bsz)
    m["mse_u2"].update(float(F.mse_loss(student_out["u2_s"].detach().float(), teacher["u2_t"].float()).item()), bsz)

    psnr_x1 = batch_metric_mean(psnr_per_image(teacher["x1"], imgs))
    psnr_student = batch_metric_mean(psnr_per_image(student_out["final_s"], imgs))
    psnr_teacher = batch_metric_mean(psnr_per_image(teacher["final_t"], imgs))
    m["mse_x1"].update(batch_metric_mean(mse_per_image(teacher["x1"], imgs)), bsz)
    m["mse_student"].update(batch_metric_mean(mse_per_image(student_out["final_s"], imgs)), bsz)
    m["mse_teacher"].update(batch_metric_mean(mse_per_image(teacher["final_t"], imgs)), bsz)
    m["psnr_x1"].update(psnr_x1, bsz)
    m["psnr_student"].update(psnr_student, bsz)
    m["psnr_teacher"].update(psnr_teacher, bsz)
    m["ssim_x1"].update(batch_metric_mean(ssim_per_image(teacher["x1"], imgs)), bsz)
    m["ssim_student"].update(batch_metric_mean(ssim_per_image(student_out["final_s"], imgs)), bsz)
    m["ssim_teacher"].update(batch_metric_mean(ssim_per_image(teacher["final_t"], imgs)), bsz)
    m["delta_student"].update(psnr_student - psnr_x1, bsz)
    m["delta_teacher"].update(psnr_teacher - psnr_x1, bsz)
    m["gap_to_teacher"].update(psnr_teacher - psnr_student, bsz)


def run_batches(
    loader,
    *,
    e1: nn.Module,
    d1: nn.Module,
    e2: nn.Module,
    d2: nn.Module,
    combiner: nn.Module,
    student: StudentZ2Predictor,
    args: argparse.Namespace,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    student.train(train)
    m = meters(METRIC_NAMES)
    max_batches = int(args.max_train_batches if train else args.max_val_batches)
    for batch_idx, (imgs, _labels) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        teacher = teacher_forward(e1, d1, e2, d2, combiner, imgs)
        student_out = student_forward(student, d2, combiner, teacher)
        losses = compute_losses(teacher, student_out, imgs, args)
        if train:
            args.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.grad_clip_norm))
            args.optimizer.step()
        update_metrics(m, imgs, teacher, student_out, losses)
        if max_batches > 0 and batch_idx >= max_batches:
            break
    return averaged(m)


@torch.no_grad()
def validate(loader, e1, d1, e2, d2, combiner, student, args, device: torch.device) -> dict[str, float]:
    return run_batches(
        loader,
        e1=e1,
        d1=d1,
        e2=e2,
        d2=d2,
        combiner=combiner,
        student=student,
        args=args,
        device=device,
        train=False,
    )


def save_stage3_v2_checkpoint(
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
    student: StudentZ2Predictor,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": jsccf_io.ROUTE,
            "stage": "stage3_v2_student_z2",
            "epoch": int(epoch),
            "metrics": metrics,
            "args": vars(args),
            "version": str(getattr(args, "version", "")),
            "teacher_layer2_ckpt": resolve_path(args.layer2_ckpt),
            "e1_state_dict": e1.state_dict(),
            "d1_state_dict": d1.state_dict(),
            "e2_teacher_state_dict": e2.state_dict(),
            "d2_teacher_state_dict": d2.state_dict(),
            "combiner_teacher_state_dict": combiner.state_dict(),
            "student_state_dict": student.state_dict(),
            "student": {
                "type": "StudentZ2Predictor",
                "input": ["z1", "bilinear_downsample_x1"],
                "in_ch": int(z1_ch(args)) + 3,
                "hidden": int(args.student_hidden),
                "blocks": int(args.student_blocks),
                "out_ch": int(z2_ch(args)),
            },
            "latent": {
                "z1": [int(args.latent_ch), int(args.latent_h), int(args.latent_w)],
                "x1_down": [3, int(args.latent_h), int(args.latent_w)],
                "z2": [int(z2_ch(args)), int(args.latent_h), int(args.latent_w)],
                "total": [int(total_latent_ch(args)), int(args.latent_h), int(args.latent_w)],
            },
        },
        out,
    )
    print(f"saved checkpoint: {out}")


def print_stage3_v2_header(args: argparse.Namespace, train_n: int, val_n: int) -> None:
    z1 = z1_ch(args)
    z2 = z2_ch(args)
    total = total_latent_ch(args)
    latent_ratio = total * int(args.latent_h) * int(args.latent_w) / float(3 * 256 * 256) * 100.0
    print("=== Stage 3-v2 | Student z2 distillation from z1+x1 ===")
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"save_dir={resolve_path(args.save_dir)}")
    print("实验设计")
    print("  model=TwoLayer-DeepJSCCF-NoChannel-NoPowerNorm-256")
    print(f"  version={args.version} z1={z1}x{args.latent_h}x{args.latent_w} z2={z2}x{args.latent_h}x{args.latent_w} total_latent_ratio={latent_ratio:.2f}%")
    print("  task=Student(concat(z1,downsample(x1)))->z2_hat, no extra transmitted bits")
    print(f"  teacher_layer2_ckpt={resolve_path(args.layer2_ckpt)}")
    print("  channel=identity power_norm=none noise=none")
    print("loss设计")
    print(
        "  "
        f"L={float(args.lambda_z):g}*MSE(z2_hat,z2_t)"
        f"+{float(args.lambda_u):g}*MSE(u2_s,u2_t)"
        f"+{float(args.lambda_final):g}*MSE(final_s,img)"
        f"+{float(args.lambda_t):g}*MSE(final_s,final_t)"
    )
    print("模块选择")
    print(f"  Teacher frozen: E1=JSCC_encoder(3->{z1}) D1=JSCC_decoder({z1}->3)")
    print(f"  Teacher frozen: E2=JSCC_encoder(6->{z2}) D2=JSCC_decoder({total}->3) combiner=Conv3x3-48-PReLU-Conv3x3-Sigmoid")
    print(f"  StudentZ2Predictor input=[B,{z1 + 3},{args.latent_h},{args.latent_w}] hidden={int(args.student_hidden)} blocks={int(args.student_blocks)} output=[B,{z2},{args.latent_h},{args.latent_w}]")
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}")


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = get_loader(cfg)
    e1, d1, e2, d2, combiner = build_layer2(args, cfg.device)
    load_teacher_checkpoint(args, e1, d1, e2, d2, combiner)
    freeze_teacher(e1, d1, e2, d2, combiner)

    # student = StudentZ2Predictor(
    #     in_ch=z1_ch(args) + 3,
    #     out_ch=z2_ch(args),
    #     hidden=int(args.student_hidden),
    #     blocks=int(args.student_blocks),
    # ).to(cfg.device)

    student = StudentZ2Predictor(
    z1_ch=16,
    x1_ch=3,
    out_ch=20,
    hidden=192,
    blocks=8,
    attn_every=2,
    heads=6,
    ).to(cfg.device)
    opt = optim.AdamW(trainable_params(student), lr=float(args.lr), weight_decay=float(args.weight_decay))
    args.optimizer = opt
    best = -1.0e9
    print_stage3_v2_header(args, len(train_loader.dataset), len(val_loader.dataset))

    metrics: dict[str, float] = {}
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        metrics = run_batches(
            train_loader,
            e1=e1,
            d1=d1,
            e2=e2,
            d2=d2,
            combiner=combiner,
            student=student,
            args=args,
            device=cfg.device,
            train=True,
        )
        print_epoch("stage3-v2-student-z2", epoch, int(args.epochs), metrics, time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, e1, d1, e2, d2, combiner, student, args, cfg.device)
            score = float(val_metrics[str(args.score_metric)])
            print(f"[stage3-v2 val {epoch:03d}] {val_metrics} score={args.score_metric}")
            if score > best:
                best = score
                save_stage3_v2_checkpoint(
                    jsccf_io.ckpt_path(args, stage3_v2_name(args), "best"),
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    e1=e1,
                    d1=d1,
                    e2=e2,
                    d2=d2,
                    combiner=combiner,
                    student=student,
                )
        if should_save_latest(args, epoch):
            save_stage3_v2_checkpoint(
                jsccf_io.ckpt_path(args, stage3_v2_name(args), "latest"),
                epoch=epoch,
                args=args,
                metrics=metrics,
                e1=e1,
                d1=d1,
                e2=e2,
                d2=d2,
                combiner=combiner,
                student=student,
            )

    save_stage3_v2_checkpoint(
        jsccf_io.ckpt_path(args, stage3_v2_name(args), "latest"),
        epoch=int(args.epochs),
        args=args,
        metrics=metrics,
        e1=e1,
        d1=d1,
        e2=e2,
        d2=d2,
        combiner=combiner,
        student=student,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", type=str, default="no-c1", help="Version of the JSCC-f training; affects checkpoint and log names.")
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=jsccf_io.default_save_dir())
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--latent-ch", type=int, default=16)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="LR for StudentZ2Predictor.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--variant", type=str, default="combiner", choices=["combiner"], help="Stage3-v2 uses the continuous combiner teacher.")
    p.add_argument("--layer2-ckpt", type=str, default="MY-V2/jscc-f/checkpoints/jscc_f_no-c1_layer2_combiner_best.pth")
    p.add_argument("--student-hidden", type=int, default=128)
    p.add_argument("--student-blocks", type=int, default=6)
    p.add_argument("--lambda-z", type=float, default=1.0)
    p.add_argument("--lambda-u", type=float, default=0.2)
    p.add_argument("--lambda-final", type=float, default=1.0)
    p.add_argument("--lambda-t", type=float, default=0.5)
    p.add_argument("--score-metric", type=str, default="psnr_student", choices=["psnr_student", "delta_student", "gap_to_teacher"])
    p.add_argument("--max-train-batches", type=int, default=0, help="Debug only: limit train batches per epoch; 0 means full epoch.")
    p.add_argument("--max-val-batches", type=int, default=0, help="Debug only: limit validation batches; 0 means full validation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.stage = "stage3-v2"
    check_jsccf_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(Path(resolve_path(args.save_dir)) / f"{stage3_v2_name(args)}_jscc_f_{args.version}.log")
    setup_log_file(args.log_file)
    write_json(Path(resolve_path(args.save_dir)) / f"{stage3_v2_name(args)}_args.json", vars(args))
    train(args)


if __name__ == "__main__":
    main()
