#!/usr/bin/env python
"""Stage 1：语义编解码器（SemanticEncoder + SemanticDecoder）训练脚本。

针对多 GPU 服务器优化：
  - torch.compile 图编译加速前向/反向传播
  - AMP 混合精度（BF16，L40S/409D 原生支持）
  - DistributedDataParallel 多卡并行（推荐 torchrun 启动）
  - 大量 worker + pin_memory + prefetch 充分利用 128GB 内存预加载数据
  - 损失参数直接取自 config.py 的 SemanticConfig，无需命令行重复传入

用法:
    # CIFAR-10（5 卡 DDP 并行）
    torchrun --standalone --nproc_per_node=5 train/train_sc.py \
        --dataset cifar10 --data_dir /path/to/cifar10 \
        --batch_size 256 --epochs 200 --num_workers 16

    # DIV2K（默认：PNG，data_dir 下 DIV2K_train_HR / DIV2K_valid_HR）
    torchrun --standalone --nproc_per_node=5 train/train_sc.py \
        --dataset div2k --data_dir /workspace/yongjia/datasets/DIV2K \
        --batch_size 4 --epochs 600 --crop_size 256 --num_workers 12

    # DIV2K（可选 LMDB：同时指定两条路径并加 --use_lmdb）
    python train/train_sc.py --dataset div2k --use_lmdb \
        --train_lmdb_path /path/train-256.lmdb --val_lmdb_path /path/valid-256.lmdb

    # 单卡调试
    CUDA_VISIBLE_DEVICES=0 python train/train_sc.py --dataset cifar10 \
        --data_dir /path/to/cifar10 --batch_size 128 --compile off

    # 恢复训练
    python train/train_sc.py --resume checkpoints/sc_cifar10_best.pth

    # 最优时刻额外拆分保存语义编/解码器权重（文件名含 sc_encoder / sc_decoder、数据集、c{n}）
    python train/train_sc.py --dataset cifar10 --data_dir /path/to/cifar10 --save_sc_split_weights
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cddm_mimo_ddnm import (
    SystemConfig,
    SemanticCommSystem,
    get_cifar10_config,
    get_div2k_config,
)
from src.cddm_mimo_ddnm.datasets import (
    get_cifar10_loaders,
    get_div2k_loaders,
)
from src.cddm_mimo_ddnm.loss import kl_loss, semantic_codec_loss


# ===========================================================================
# 命令行参数
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 - 语义编解码器训练（多GPU/AMP/torch.compile 优化版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 数据集 ----
    parser.add_argument("--dataset", type=str, default="div2k",
                        choices=["cifar10", "div2k"], help="数据集")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default=None,
        help=(
            "数据集根目录。DIV2K 省略时默认为 "
            "/workspace/yongjia/datasets/DIV2K（其下须有 DIV2K_train_HR / DIV2K_valid_HR）；"
            "CIFAR10 必须指定。"
        ),
    )
    parser.add_argument("--crop_size", type=int, default=256, help="DIV2K 裁剪尺寸")
    parser.add_argument(
        "--use_lmdb",
        action="store_true",
        help=(
            "DIV2K：启用 LMDB（若 data_dir 下存在 train-{crop}.lmdb / valid-{crop}.lmdb）。"
            "默认关闭，直接使用 DIV2K_train_HR / DIV2K_valid_HR 下的 PNG。"
        ),
    )
    parser.add_argument(
        "--train_lmdb_path",
        type=str,
        default=None,
        help=(
            "可选：显式指定训练集 LMDB 路径；必须与 --val_lmdb_path 同时给出。"
            "给出后强制走 LMDB，无需再加 --use_lmdb。"
        ),
    )
    parser.add_argument(
        "--val_lmdb_path",
        type=str,
        default=None,
        help="可选：显式指定验证集 LMDB 路径；必须与 --train_lmdb_path 同时给出。",
    )
    parser.add_argument(
        "--cache_decoded",
        dest="cache_decoded",
        action="store_true",
        default=True,
        help=(
            "DIV2K PNG 模式：在 __init__ 中一次性把所有图像解码为 numpy 常驻内存，"
            "训练时只做随机裁剪+ToTensor，避免每个 batch 都解码 PNG 导致 GPU 周期性掉 0。"
            "DIV2K_train_HR 800 张约 8.6 GB；每个 DDP rank 各一份。默认开启。"
        ),
    )
    parser.add_argument(
        "--no_cache_decoded",
        dest="cache_decoded",
        action="store_false",
        help="关闭 --cache_decoded（每次 __getitem__ 重新读盘并解码 PNG）。",
    )
    parser.add_argument(
        "--cache_workers",
        type=int,
        default=None,
        help="启动期解码并发线程数；不传则用 min(16, os.cpu_count())。",
    )

    # ---- 训练超参（核心）----
    parser.add_argument("--epochs", type=int, default=2400, help="总轮次")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="每张 GPU 的批大小（总 batch_size = 此值 × GPU 数量）")
    parser.add_argument("--lr", type=float, default=1e-4, help="第 2 个 epoch 起的基础学习率（见 --lr_first_epoch）")
    parser.add_argument(
        "--lr_first_epoch",
        type=float,
        default=1e-4,
        help="仅第 1 个 epoch 使用的学习率；与 --lr 相同时不启用两阶段",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="梯度累积步数（>1 可在小 batch 下保持等效总 batch）",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=0.0,
        help="梯度范数裁剪上限（>0 启用 clip_grad_norm_，0 关闭）",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_warmup",
        choices=["cosine_warmup", "plateau", "cosine", "step", "none"],
        help=(
            "LR 调度策略：\n"
            "  cosine_warmup  推荐。按 optimizer step 调度：先线性 warmup，再余弦衰减到 min_lr。\n"
            "  plateau        按 val PSNR 触发：连续 patience 次不提升就乘 factor。\n"
            "  cosine         旧行为：CosineAnnealingLR(T_max=epochs)，按 epoch 级。\n"
            "  step           StepLR(step_size=100, gamma=0.5)，按 epoch 级。\n"
            "  none           不调度。"
        ),
    )
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="cosine_warmup 的 warmup（按 optimizer step 计；不是 batch）")
    parser.add_argument("--warmup_start_ratio", type=float, default=0.1,
                        help="cosine_warmup 起始 lr 与 base lr 的比例")
    parser.add_argument("--min_lr_ratio", type=float, default=0.05,
                        help="cosine_warmup / plateau 的 lr 下限与 base lr 的比例")
    parser.add_argument("--plateau_patience", type=int, default=3,
                        help="plateau：连续多少次 eval 没新 best 就降 lr")
    parser.add_argument("--plateau_factor", type=float, default=0.5,
                        help="plateau：每次降 lr 的乘数")

    # ---- 系统性能调优 ----
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help=(
            "训练 DataLoader 工作进程数（每个 GPU 任务）。\n"
            "经验：单卡 4-12 通常足够；多任务共用一台机器时务必降低，"
            "避免 (任务数 × num_workers × 2) > CPU 物理核数。\n"
            "示例：3 任务共用 144 核服务器 → 每任务 8-12 较合适。"
        ),
    )
    parser.add_argument(
        "--val_num_workers", type=int, default=None,
        help=(
            "验证 DataLoader 工作进程数。默认 max(2, num_workers // 4)。\n"
            "评估只是间歇触发，无须与训练一样多 worker。"
        ),
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=2,
        help="每个 worker 预取的 batch 数。多任务共用机器时建议 2，单任务可调到 4。",
    )
    parser.add_argument(
        "--val_persistent_workers", action="store_true",
        help="是否让验证 DataLoader 的 worker 常驻（默认 off，按需启停以释放 CPU）。",
    )
    parser.add_argument("--compile", type=str, default="reduce-overheads",
                        choices=["on", "off", "default", "reduce-overheads"],
                        help="torch.compile 模式（'reduce-overheads' 推荐，'off' 关闭）")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "none"],
                        help="AMP 混合精度类型（L40S/409D 推荐 bfloat16，显存紧张用 float16）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default=None,
                        help="指定设备（如 cuda:0），留空则自动选择所有可用 GPU")

    # ---- 日志 & 保存 ----
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "checkpoints-val/sc/no_vae"))
    parser.add_argument("--log_file", type=str, default="log/sc-no_vae.txt", help="终端日志保存路径")
    parser.add_argument("--log_freq", type=int, default=50, help="训练日志打印频率（按 batch）")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="epoch",
        choices=["batch", "epoch"],
        help=(
            "验证并据此保存最优 checkpoint 的时机："
            "batch=按全局 batch 计数，每隔 eval_every_batches 触发一次；"
            "epoch=按 epoch，间隔由 --eval_every_epochs 决定（最后一轮 epoch 总会验证）。"
            "二者互斥：epoch 模式下忽略 --eval_every_batches。"
        ),
    )
    parser.add_argument(
        "--eval_every_epochs",
        type=int,
        default=20,
        help=(
            "eval_mode=epoch 时：每隔多少个 epoch 做一次验证并尝试保存最优（默认 20；"
            "训练最后一个 epoch 结束时总会验证）。batch 模式下无效。"
        ),
    )
    parser.add_argument("--eval_every_batches", type=int, default=800,
                        help="eval_mode=batch 时：每多少个全局 batch 验证并尝试保存最优（≤0 表示本轮训内不做间隔验证）")
    parser.add_argument("--resume", type=str, default=None, help="恢复检查点路径")
    parser.add_argument(
        "--save_sc_split_weights",
        action="store_true",
        help=(
            "保存完整检查点的同时，将语义编码器/解码器各存一份独立权重；"
            "文件名含 sc_encoder / sc_decoder、数据集名与输出通道数 c{n}。"
        ),
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=None,
        help="若指定则覆盖 config 中的 semantic.embed_dim（潜空间瓶颈维）；不指定则用数据集预设。",
    )

    args = parser.parse_args()
    if args.eval_every_epochs < 1:
        parser.error("--eval_every_epochs 必须 >= 1")
    return args


# ===========================================================================
# 辅助工具
# ===========================================================================

class AverageMeter:
    """运行均值统计"""

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


class TeeStream:
    """将终端输出同时写入文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_log_file(log_path: str):
    abs_path = log_path if os.path.isabs(log_path) else os.path.join(PROJECT_ROOT, log_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    f = open(abs_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    builtins.print(f"\n=== New session @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    builtins.print(f"Log file: {abs_path}")
    return f


def is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not is_dist_ready()) or dist.get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def dist_barrier():
    if is_dist_ready():
        dist.barrier()


class Stage1Wrapper(nn.Module):
    """将 forward_stage1 封装为标准 forward，确保 DDP hook 生效。"""

    def __init__(self, core_system: nn.Module):
        super().__init__()
        self.system = core_system

    def forward(self, x: torch.Tensor):
        return self.system.forward_stage1(x)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算 PSNR (dB)，输入范围 [0,1]。"""
    mse_val = torch.mean((pred - target) ** 2).item()
    if mse_val < 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse_val)


# ===========================================================================
# 构建函数：配置 / 数据 / 模型 / 编译
# ===========================================================================

def build_config(dataset: str) -> SystemConfig:
    if dataset == "div2k":
        return get_div2k_config()
    return get_cifar10_config()


def build_dataloaders(args: argparse.Namespace, device: torch.device):
    """构建 DataLoader，充分利用 128GB RAM 做 I/O 预加载。

    num_workers 设大 + pin_memory=True + prefetch_factor=4 + persistent_workers
    可确保 GPU 几乎不会因数据读取而等待。
    """
    nw = args.num_workers
    val_nw = args.val_num_workers
    pf = args.prefetch_factor
    val_persist = bool(getattr(args, "val_persistent_workers", False))
    if args.dataset == "div2k":
        # 当显式指定 train/val LMDB 时，data_dir 可能为空；这里兜底为空字符串，
        # 以兼容 get_div2k_loaders 内部默认路径拼接逻辑。
        data_dir = args.data_dir or ""
        train_lp = getattr(args, "train_lmdb_path", None)
        val_lp = getattr(args, "val_lmdb_path", None)
        if train_lp is not None and isinstance(train_lp, str) and not train_lp.strip():
            train_lp = None
        if val_lp is not None and isinstance(val_lp, str) and not val_lp.strip():
            val_lp = None
        explicit_lmdb_paths = train_lp is not None and val_lp is not None
        use_lmdb = bool(getattr(args, "use_lmdb", False)) or explicit_lmdb_paths
        train_loader, val_loader, _ = get_div2k_loaders(
            data_dir=data_dir,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            num_workers=nw,
            distributed=args.distributed,
            use_lmdb=use_lmdb,
            train_lmdb_path=train_lp,
            val_lmdb_path=val_lp,
            val_num_workers=val_nw,
            prefetch_factor=pf,
            val_persistent_workers=val_persist,
            cache_decoded=bool(getattr(args, "cache_decoded", True)),
            cache_workers=getattr(args, "cache_workers", None),
        )
    else:
        train_loader, val_loader, _ = get_cifar10_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=nw,
            distributed=args.distributed,
            val_num_workers=val_nw,
            prefetch_factor=pf,
            val_persistent_workers=val_persist,
        )

    # DataLoader 初始化后不允许再直接修改 drop_last/prefetch_factor/pin_memory 等属性。
    # 这些参数已在 datasets.py 的 _make_loaders 中配置，这里不再覆盖。
    return train_loader, val_loader


def build_model_and_optimizer(cfg: SystemConfig, args, device: torch.device):
    """构建模型、DDP 包装、AMP GradScaler、优化器、scheduler。

    Stage 1 只优化 semantic_encoder + semantic_decoder 的参数。
    """
    core_system = SemanticCommSystem(cfg).to(device)
    model = Stage1Wrapper(core_system)

    # Stage1 仅训练语义编解码器；其余参数全部冻结，避免 DDP 等待无梯度参数。
    for p in core_system.parameters():
        p.requires_grad = False
    for p in core_system.semantic_encoder.parameters():
        p.requires_grad = True
    for p in core_system.semantic_decoder.parameters():
        p.requires_grad = True

    # ---- torch.compile 加速 ----
    compile_mode = args.compile
    if compile_mode != "off" and hasattr(torch, "compile"):
        if is_main_process():
            print(f"  torch.compile mode: {compile_mode}")
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception as e:
            if is_main_process():
                print(f"  [WARN] torch.compile 失败 ({e})，回退到 eager 模式")

    # ---- Stage 1 可训参数（仅语义编解码器）----
    sc_params = (
        list(core_system.semantic_encoder.parameters())
        + list(core_system.semantic_decoder.parameters())
    )

    optimizer = optim.Adam(sc_params, lr=args.lr, weight_decay=args.weight_decay)

    total_p = sum(p.numel() for p in core_system.parameters())
    train_p = sum(p.numel() for p in sc_params)
    if is_main_process():
        print(f"  模型总参数 : {total_p:,}")
        print(f"  可训练参数 : {train_p:,} ({train_p/total_p*100:.1f}%)")

    # ---- AMP GradScaler ----
    amp_enabled = args.amp_dtype != "none"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype = dtype_map.get(args.amp_dtype, torch.bfloat16)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    return model, optimizer, scaler, amp_enabled, amp_dtype


def build_lr_scheduler(args, optimizer, train_loader_len: int):
    """根据 args 创建 lr scheduler。

    返回 (scheduler, mode):
        mode == "step"    : 每个 optimizer.step() 后调度（warmup+cosine）
        mode == "epoch"   : 每个 epoch 末调度（旧的 CosineAnnealingLR / StepLR）
        mode == "plateau" : 每次 val 评估后用指标触发
        mode == "none"    : 不调度
    """
    name = args.lr_scheduler
    base_lr = float(args.lr)
    accum = max(1, int(args.grad_accum_steps))
    two_tier = abs(float(args.lr_first_epoch) - base_lr) > 1e-12

    if name == "cosine_warmup":
        if two_tier and is_main_process():
            print("  [LR] cosine_warmup 与两阶段 lr 不兼容：忽略 --lr_first_epoch。")
        steps_per_epoch = max(1, train_loader_len // accum)
        total_steps = max(1, steps_per_epoch * int(args.epochs))
        warmup_steps = max(0, int(args.warmup_steps))
        warmup_steps = min(warmup_steps, max(1, total_steps - 1))
        start_ratio = max(1e-6, float(args.warmup_start_ratio))
        min_ratio = max(0.0, float(args.min_lr_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                if warmup_steps == 0:
                    return 1.0
                t = step / float(max(1, warmup_steps))
                return start_ratio + (1.0 - start_ratio) * t
            t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            t = min(max(t, 0.0), 1.0)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))
            return min_ratio + (1.0 - min_ratio) * cos_factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if is_main_process():
            print(
                f"  [LR] cosine_warmup: total_steps={total_steps}  "
                f"warmup={warmup_steps}  start={base_lr*start_ratio:.2e} -> peak={base_lr:.2e} "
                f"-> min={base_lr*min_ratio:.2e}"
            )
        return scheduler, "step"

    if name == "plateau":
        if two_tier and is_main_process():
            print("  [LR] plateau 与两阶段 lr 不兼容：忽略 --lr_first_epoch。")
        min_lr = base_lr * max(0.0, float(args.min_lr_ratio))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
            threshold=1e-3,
            min_lr=min_lr,
        )
        if is_main_process():
            print(
                f"  [LR] plateau(monitor=val PSNR, max): "
                f"factor={args.plateau_factor}  patience={args.plateau_patience}  "
                f"min_lr={min_lr:.2e}"
            )
        return scheduler, "plateau"

    if name == "cosine":
        if two_tier:
            if is_main_process():
                print("  [LR] 两阶段 lr：第 1 epoch 用 lr_first_epoch，之后用 lr；已禁用 Cosine 调度。")
            return None, "none"
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs), "epoch"

    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5), "epoch"

    return None, "none"


# ===========================================================================
# 训练 & 验证循环
# ===========================================================================

@torch.enable_grad()
def train_one_epoch(
    model, loader: DataLoader, optimizer, scaler, device, epoch, args,
    amp_enabled: bool, amp_dtype: torch.dtype, cfg: SystemConfig,
    global_batch_start: int = 0,
    on_eval_interval=None,
    scheduler=None,
    scheduler_mode: str = "none",
):
    model.train()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    lam_kl = cfg.semantic.lambda_kl
    use_vae = cfg.semantic.use_vae

    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)

    t0 = time.time()
    accum_steps = max(1, int(args.grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        images_non_blocking = images.to(device, non_blocking=True)
        should_step = ((i + 1) % accum_steps == 0) or ((i + 1) == len(loader))
        sync_context = (
            model.no_sync() if args.distributed and not should_step else contextlib.nullcontext()
        )
        with sync_context:
            with torch.autocast("cuda", **autocast_kw):
                x_hat, mu, logvar = model(images_non_blocking)
            loss = semantic_codec_loss(
                x_hat=x_hat, x=images_non_blocking,
                mu=mu, logvar=logvar,
                lambda_kl=lam_kl, use_vae=use_vae,
            )
            loss_for_backward = loss / accum_steps
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
        if should_step:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            clip_val = getattr(args, "clip_grad_norm", 0.0)
            if clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=clip_val,
                )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and scheduler_mode == "step":
                scheduler.step()

        bs = images.shape[0]
        loss_meter.update(loss.item(), bs)
        psnr_meter.update(
            compute_psnr(
                x_hat.detach().float().clamp(0, 1),
                images_non_blocking.float(),
            ),
            bs,
        )

        if args.is_main and ((i + 1) % args.log_freq == 0 or i + 1 == len(loader)):
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            it_s = elapsed / (i + 1)
            remaining = it_s * (len(loader) - i - 1)
            print(f"  [{epoch+1}/{args.epochs}][{i+1}/{len(loader)}]  "
                  f"Loss:{loss_meter.avg:.4f}  PSNR:{psnr_meter.avg:.2f}dB  "
                  f"LR:{lr_now:.6f}  {it_s:.2f}s/it  ETA:{remaining:.0f}s")
        global_batch = global_batch_start + i + 1
        if (
            on_eval_interval is not None
            and getattr(args, "eval_mode", "batch") == "batch"
            and args.eval_every_batches > 0
            and (global_batch % args.eval_every_batches == 0)
        ):
            on_eval_interval(global_batch, epoch, {
                "loss": loss_meter.avg,
                "psnr": psnr_meter.avg,
            })
            model.train()

    return {"loss": loss_meter.avg, "psnr": psnr_meter.avg}, global_batch_start + len(loader)


@torch.no_grad()
def validate(model, loader: DataLoader, device, args,
             amp_enabled: bool, amp_dtype: torch.dtype, cfg: SystemConfig):
    """验证评估。use_vae 时同步估计 KL(未加权)、mu/logvar 的分布统计，便于看 KL 是否过弱。"""
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    vae = cfg.semantic.use_vae
    kl_raw_meter = AverageMeter()
    logvar_mean_meter = AverageMeter()
    mu_mean_meter = AverageMeter()
    mu_var_meter = AverageMeter()
    mu_ch_std_mean_meter = AverageMeter()

    lam_kl = cfg.semantic.lambda_kl
    autocast_kw = dict(enabled=amp_enabled, dtype=amp_dtype)

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        images_d = images.to(device, non_blocking=True)
        with torch.autocast("cuda", **autocast_kw):
            x_hat, mu, logvar = model(images_d)
            loss = semantic_codec_loss(
                x_hat=x_hat, x=images_d, mu=mu, logvar=logvar,
                lambda_kl=lam_kl, use_vae=cfg.semantic.use_vae,
            )

        bs = images.shape[0]
        loss_meter.update(loss.item(), bs)
        psnr_meter.update(compute_psnr(x_hat.float().clamp(0, 1), images_d.float()), bs)
        if vae and mu is not None and logvar is not None:
            # fp32 统计，避免半精度下 var/std 失准
            mu_f = mu.detach().float()
            logv_f = logvar.detach().float()
            klv = kl_loss(mu_f, logv_f)
            kl_raw_meter.update(klv.item(), bs)
            logvar_mean_meter.update(logv_f.mean().item(), bs)
            mu_mean_meter.update(mu_f.mean().item(), bs)
            mu_var_meter.update(mu_f.var().item(), bs)
            c = mu_f.shape[1]
            flat = mu_f.permute(1, 0, 2, 3).reshape(c, -1)
            ch_std = flat.std(dim=1)
            mu_ch_std_mean_meter.update(ch_std.mean().item(), bs)

    if is_dist_ready():
        if vae:
            stats = torch.tensor(
                [
                    loss_meter.sum,
                    psnr_meter.sum,
                    loss_meter.count,
                    kl_raw_meter.sum,
                    kl_raw_meter.count,
                    logvar_mean_meter.sum,
                    logvar_mean_meter.count,
                    mu_mean_meter.sum,
                    mu_mean_meter.count,
                    mu_var_meter.sum,
                    mu_var_meter.count,
                    mu_ch_std_mean_meter.sum,
                    mu_ch_std_mean_meter.count,
                ],
                device=device,
                dtype=torch.float64,
            )
        else:
            stats = torch.tensor(
                [loss_meter.sum, psnr_meter.sum, loss_meter.count],
                device=device,
                dtype=torch.float64,
            )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        avg_loss = (stats[0] / stats[2]).item()
        avg_psnr = (stats[1] / stats[2]).item()
        out: dict = {"loss": avg_loss, "psnr": avg_psnr}
        if vae:
            out["kl_raw"] = (stats[3] / stats[4]).item()
            out["logvar_mean"] = (stats[5] / stats[6]).item()
            out["mu_mean"] = (stats[7] / stats[8]).item()
            out["mu_var"] = (stats[9] / stats[10]).item()
            out["mu_ch_std_mean"] = (stats[11] / stats[12]).item()
        return out

    out = {"loss": loss_meter.avg, "psnr": psnr_meter.avg}
    if vae:
        out["kl_raw"] = kl_raw_meter.avg
        out["logvar_mean"] = logvar_mean_meter.avg
        out["mu_mean"] = mu_mean_meter.avg
        out["mu_var"] = mu_var_meter.avg
        out["mu_ch_std_mean"] = mu_ch_std_mean_meter.avg
    return out


# ===========================================================================
# 检查点管理
# ===========================================================================

def save_sc_split_weights(model: nn.Module, save_dir: str, dataset: str, metrics: dict | None) -> None:
    """将语义编解码器拆成两个权重文件（仅 state_dict 与少量元数据）。"""
    w = unwrap_model(model)
    system = w.system
    out_c = int(system.cfg.semantic.embed_dim)
    base = f"{dataset}_c{out_c}"
    enc_path = os.path.join(save_dir, f"sc_encoder_{base}.pth")
    dec_path = os.path.join(save_dir, f"sc_decoder_{base}.pth")
    os.makedirs(save_dir, exist_ok=True)
    common = {
        "dataset": dataset,
        "embed_dim": out_c,
        "metrics": metrics,
    }
    torch.save({**common, "state_dict": system.semantic_encoder.state_dict()}, enc_path)
    torch.save({**common, "state_dict": system.semantic_decoder.state_dict()}, dec_path)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args):
    core_model = unwrap_model(model)
    state = {
        "epoch": epoch,
        "metrics": metrics,
        "model_state_dict": core_model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if getattr(args, "save_sc_split_weights", False):
        save_sc_split_weights(model, os.path.dirname(path), args.dataset, metrics)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    core_model = unwrap_model(model)
    core_model.load_state_dict(state["model_state_dict"])
    start_epoch = state["epoch"] + 1
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    print(f"  >> Loaded {path}  resume epoch={start_epoch}")
    metrics = state.get("metrics", {})
    if metrics:
        print(f"     prev best: {metrics}")
    return start_epoch, metrics


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = world_size > 1
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    args.is_main = rank == 0

    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP 需要 CUDA 环境。")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    log_f = setup_log_file(args.log_file) if args.is_main else None

    div2k_default_root = "/workspace/yongjia/datasets/DIV2K"
    if args.dataset == "div2k":
        if args.data_dir is None or (
            isinstance(args.data_dir, str) and not args.data_dir.strip()
        ):
            args.data_dir = div2k_default_root
    if args.dataset == "div2k":
        t_lp = args.train_lmdb_path
        v_lp = args.val_lmdb_path
        if t_lp is not None and isinstance(t_lp, str) and not t_lp.strip():
            t_lp = None
        if v_lp is not None and isinstance(v_lp, str) and not v_lp.strip():
            v_lp = None
        if (t_lp is None) ^ (v_lp is None):
            raise ValueError(
                "DIV2K：--train_lmdb_path 与 --val_lmdb_path 必须同时指定或同时省略。"
            )
        explicit_lmdb = t_lp is not None and v_lp is not None
        if not explicit_lmdb and not (args.data_dir and str(args.data_dir).strip()):
            raise ValueError("DIV2K 模式下，未指定 LMDB 路径时必须提供非空 --data_dir。")
    else:
        if not args.data_dir:
            raise ValueError("CIFAR10 模式下必须提供 --data_dir。")

    # ---- 设备 ----
    if args.distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # CUDA 性能优化
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    seed = args.seed + args.local_rank
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ---- 配置（损失参数全来自这里，不再命令行重复传）----
    cfg = build_config(args.dataset)
    if args.embed_dim is not None:
        cfg.semantic.embed_dim = int(args.embed_dim)

    # ---- 打印信息 ----
    if args.is_main:
        print("=" * 64)
        print(f"  Stage 1: 语义编解码器训练  |  dataset={args.dataset}")
        print("=" * 64)
        print(f"  Device       : {device}")
        if device.type == "cuda":
            print(f"  GPU count    : {torch.cuda.device_count()}  "
                  f"({torch.cuda.get_device_name(0)})")
            print(f"  CUDA version  : {torch.version.cuda}")
            print(f"  Total memory  : ~128 GB RAM (server)")
            if args.distributed:
                print(f"  DDP          : world_size={args.world_size}")
        print(f"  Batch/GPU    : {args.batch_size}")
        print(f"  Grad Accum   : {max(1, int(args.grad_accum_steps))}")
        lfe = float(args.lr_first_epoch)
        lr_main = float(args.lr)
        if abs(lfe - lr_main) > 1e-12 and args.lr_scheduler in ("none", "cosine", "step"):
            print(
                f"  LR (两阶段)  : epoch1={lfe}  之后={lr_main}  |  epochs={args.epochs}"
            )
        else:
            print(f"  LR           : {args.lr}  |  epochs={args.epochs}")
        print(f"  LR scheduler : {args.lr_scheduler}")
        eff_val_nw = (
            args.val_num_workers
            if args.val_num_workers is not None
            else (max(2, args.num_workers // 4) if args.num_workers > 0 else 0)
        )
        print(
            f"  Workers      : train={args.num_workers}  val={eff_val_nw}  "
            f"prefetch={args.prefetch_factor}  val_persistent={bool(args.val_persistent_workers)}"
            f"  |  AMP={args.amp_dtype}  |  compile={args.compile}"
        )
        try:
            cpu_cnt = os.cpu_count() or 0
            ws = max(1, args.world_size)
            est_workers = ws * (args.num_workers + (eff_val_nw if args.val_persistent_workers else 0))
            if cpu_cnt and est_workers > cpu_cnt:
                print(
                    f"  [HINT] 本任务常驻 worker ≈ {est_workers}，CPU 物理核数 = {cpu_cnt}。\n"
                    f"         多任务共用机器时，请确保 (任务数 × num_workers) 不超过 CPU 核数，"
                    f"否则 GPU 利用率会大幅掉 0。"
                )
        except Exception:
            pass
        print(f"  save_sc_split_weights : {args.save_sc_split_weights}")
        clip_v = getattr(args, "clip_grad_norm", 0.0)
        print(f"  clip_grad_norm: {clip_v if clip_v > 0 else 'off'}")
        if args.eval_mode == "epoch":
            _eee = max(1, int(args.eval_every_epochs))
            print(
                f"  Eval/best ckpt : eval_mode=epoch，每 {_eee} 个 epoch 验证一次"
                "（最后一轮必定验证；忽略 --eval_every_batches）"
            )
        else:
            _eb = args.eval_every_batches
            print(
                f"  Eval/best ckpt : eval_mode=batch，每隔 {_eb} 个全局 batch"
                + ("（≤0 表示训练期内不做间隔验证）" if _eb <= 0 else "")
            )
        print(f"  Loss params  (from config.py SemanticConfig):")
        print(f"    lambda_kl    = {cfg.semantic.lambda_kl}")
        print(f"    use_vae      = {cfg.semantic.use_vae}")
        print(f"    embed_dim    = {cfg.semantic.embed_dim}" + (
            "" if args.embed_dim is None else "  (--embed_dim override)"
        ))
        if args.dataset == "div2k":
            _t = args.train_lmdb_path
            _v = args.val_lmdb_path
            _t = None if (_t is None or (isinstance(_t, str) and not _t.strip())) else _t
            _v = None if (_v is None or (isinstance(_v, str) and not _v.strip())) else _v
            _explicit = _t is not None and _v is not None
            print(f"  DIV2K data_dir : {args.data_dir}")
            if _explicit:
                print(f"  DIV2K loader   : LMDB（显式路径）")
                print(f"    train_lmdb : {_t}")
                print(f"    val_lmdb   : {_v}")
            elif args.use_lmdb:
                print(
                    "  DIV2K loader   : LMDB（data_dir 下 train-{crop}.lmdb / valid-{crop}.lmdb，若存在）"
                )
            else:
                print("  DIV2K loader   : PNG（不使用 LMDB）")
                print(f"    train : {os.path.join(args.data_dir, 'DIV2K_train_HR')}")
                print(f"    val   : {os.path.join(args.data_dir, 'DIV2K_valid_HR')}")
                if bool(getattr(args, "cache_decoded", True)):
                    print(
                        "    cache : ON（启动时一次性解码，常驻内存；__getitem__ 仅做随机裁剪）"
                    )
                else:
                    print(
                        "    cache : OFF（每次 __getitem__ 重新读盘并解码 PNG，CPU 重，"
                        "易导致 GPU 周期性掉 0；建议加 --cache_decoded）"
                    )

    # ---- 构建 ----
    train_loader, val_loader = build_dataloaders(args, device)
    model, optimizer, scaler, amp_enabled, amp_dtype = \
        build_model_and_optimizer(cfg, args, device)
    scheduler, scheduler_mode = build_lr_scheduler(args, optimizer, len(train_loader))

    # ---- 恢复 ----
    start_epoch = 0
    resumed_metrics: dict = {}
    if args.resume and os.path.isfile(args.resume):
        start_epoch, resumed_metrics = load_checkpoint(args.resume, model, optimizer, scheduler)
    dist_barrier()

    # ---- 训练循环 ----
    save_dir = args.save_dir
    prefix = f"sc_{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    best_psnr = float(resumed_metrics.get("v_psnr", 0.0))
    best_global_batch = -1
    all_metrics = {}
    global_batch = 0
    if args.is_main and best_psnr > 0:
        print(f"  Resumed best PSNR: {best_psnr:.4f} dB")

    wall_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        ep_t0 = time.time()
        lfe = float(args.lr_first_epoch)
        lr_main = float(args.lr)
        # 仅「从头训 + epoch 级或无调度器」时才允许两阶段覆盖；
        # 否则覆盖会废掉 step 级 / plateau 调度器。
        allow_per_epoch_override = scheduler_mode in ("none", "epoch")
        use_first_epoch_lr = (
            allow_per_epoch_override
            and (start_epoch == 0)
            and (epoch == 0)
            and abs(lfe - lr_main) > 1e-12
        )
        if allow_per_epoch_override:
            lr_run = lfe if use_first_epoch_lr else lr_main
            for g in optimizer.param_groups:
                g["lr"] = lr_run
        else:
            lr_run = float(optimizer.param_groups[0]["lr"])

        if args.distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if args.is_main:
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
            if allow_per_epoch_override and abs(lfe - lr_main) > 1e-12:
                print(f"  lr 已设为 {lr_run}  (两阶段: 第 1 轮={lfe}，之后={lr_main})")
            else:
                print(f"  lr 起始: {lr_run:.6g}  (scheduler={args.lr_scheduler}, mode={scheduler_mode})")

        def _eval_and_maybe_save(
            cur_global_batch: int,
            cur_epoch: int,
            train_snapshot: dict,
            *,
            eval_tag: str = "interval",
        ):
            nonlocal best_psnr, best_global_batch, all_metrics
            dist_barrier()
            val_metrics = validate(
                model, val_loader, device, args, amp_enabled, amp_dtype, cfg,
            )
            if args.is_main:
                all_metrics = {
                    **train_snapshot,
                    **{f"v_{k}": v for k, v in val_metrics.items()},
                    "global_batch": cur_global_batch,
                    "eval_tag": eval_tag,
                    "eval_epoch": cur_epoch + 1,
                }
                cur_psnr = float(val_metrics.get("psnr", 0.0))
                images_seen = cur_global_batch * args.batch_size * max(1, args.world_size)
                if eval_tag == "epoch":
                    line = (
                        f"  [Eval@end_epoch={cur_epoch + 1}/{args.epochs}, batch={cur_global_batch}, "
                        f"images~{images_seen}] "
                        f"Val L:{val_metrics['loss']:.4f}  PSNR:{cur_psnr:.2f}dB"
                    )
                else:
                    line = (
                        f"  [Eval@batch={cur_global_batch}, images~{images_seen}] "
                        f"Val L:{val_metrics['loss']:.4f}  PSNR:{cur_psnr:.2f}dB"
                    )
                if cfg.semantic.use_vae and "kl_raw" in val_metrics:
                    line += (
                        f"  KL(raw):{val_metrics['kl_raw']:.5f}  "
                        f"mu_m:{val_metrics['mu_mean']:.5f}  mu_v:{val_metrics['mu_var']:.5f}  "
                        f"mu_chstd_m:{val_metrics['mu_ch_std_mean']:.5f}  "
                        f"logvar_m:{val_metrics['logvar_mean']:.5f}"
                    )
                print(line)
                if cur_psnr > best_psnr:
                    best_psnr = cur_psnr
                    best_global_batch = cur_global_batch
                    save_checkpoint(
                        os.path.join(save_dir, f"{prefix}_c{int(cfg.semantic.embed_dim)}_best.pth"),
                        model, optimizer, scheduler, cur_epoch, all_metrics, args,
                    )
                    where = (
                        f"epoch {cur_epoch + 1}" if eval_tag == "epoch"
                        else f"batch {cur_global_batch}"
                    )
                    print(f"  *** New Best PSNR: {best_psnr:.4f} dB @ {where} ***")
            # plateau：所有 rank 都需要同步 step（用同一指标，避免漂移）
            if scheduler is not None and scheduler_mode == "plateau":
                metric_t = torch.tensor(
                    float(val_metrics.get("psnr", 0.0)),
                    device=device,
                    dtype=torch.float32,
                )
                if is_dist_ready():
                    dist.broadcast(metric_t, src=0)
                prev_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(metric_t.item())
                new_lr = optimizer.param_groups[0]["lr"]
                if args.is_main and abs(new_lr - prev_lr) > 1e-12:
                    print(f"  [LR] plateau 触发：{prev_lr:.2e} -> {new_lr:.2e}")
            dist_barrier()

        train_metrics, global_batch = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args,
            amp_enabled, amp_dtype, cfg,
            global_batch_start=global_batch,
            on_eval_interval=(
                _eval_and_maybe_save if args.eval_mode == "batch" else None
            ),
            scheduler=scheduler,
            scheduler_mode=scheduler_mode,
        )

        ep_time = time.time() - ep_t0

        msg = f"  Ep[{epoch+1}]  Train L:{train_metrics['loss']:.4f} P:{train_metrics['psnr']:.2f}"
        msg += f"  t:{ep_time:.1f}s"
        if args.is_main:
            print(msg)

        if args.eval_mode == "epoch":
            ep_done = epoch + 1
            ee = int(args.eval_every_epochs)
            if ep_done % ee == 0 or ep_done == args.epochs:
                _eval_and_maybe_save(global_batch, epoch, train_metrics, eval_tag="epoch")

        # 仅 epoch 级调度（旧的 cosine / step）在这里触发；
        # cosine_warmup 已在每个 optimizer.step() 之后触发；plateau 在 eval 时触发。
        if scheduler is not None and scheduler_mode == "epoch":
            scheduler.step()
    if best_global_batch < 0:
        val_metrics = validate(model, val_loader, device, args, amp_enabled, amp_dtype, cfg)
        if args.is_main:
            all_metrics = {**train_metrics, **{f"v_{k}": v for k, v in val_metrics.items()}, "global_batch": global_batch}
            best_psnr = float(val_metrics.get("psnr", 0.0))
            best_global_batch = global_batch
            save_checkpoint(
                os.path.join(save_dir, f"{prefix}_c{int(cfg.semantic.embed_dim)}_best.pth"),
                model, optimizer, scheduler, args.epochs - 1, all_metrics, args,
            )
            fline = f"  [Final Eval] L:{val_metrics['loss']:.4f}  PSNR:{best_psnr:.4f}dB @ batch {best_global_batch}"
            if cfg.semantic.use_vae and "kl_raw" in val_metrics:
                fline += (
                    f"  KL(raw):{val_metrics['kl_raw']:.5f}  "
                    f"mu_m:{val_metrics['mu_mean']:.5f}  mu_v:{val_metrics['mu_var']:.5f}  "
                    f"mu_chstd_m:{val_metrics['mu_ch_std_mean']:.5f}  logvar_m:{val_metrics['logvar_mean']:.5f}"
                )
            print(fline)

    if args.is_main:
        total_min = (time.time() - wall_start) / 60
        print(f"\n{'='*64}")
        best_where = (
            f"epoch 结束触发保存（batch={best_global_batch}）"
            if args.eval_mode == "epoch"
            else f"batch {best_global_batch}"
        )
        print(f"  Done!  {total_min:.1f}min  Best PSNR={best_psnr:.4f}dB @ {best_where}")
        print(f"  Best checkpoint: {os.path.join(save_dir, f'{prefix}_c{int(cfg.semantic.embed_dim)}_best.pth')}")
        print("=" * 64)
        log_f.close()
    if args.distributed and is_dist_ready():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
