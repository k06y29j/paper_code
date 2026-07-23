#!/usr/bin/env python3
"""Insert an FSQ bottleneck inside the pretrained 320-channel Swin Layer2.

Unlike ``train_layer2_fsq_direct.py``, this route does not replace the two
critical source heads with random ``320->d`` and ``d->320`` heads.  It keeps the
exact pretrained E2(320), D2(320), and original combiner contract, and inserts
an explicit adapter::

    E2_320 -> analysis 1x1 (320->d) -> BN/tanh -> FSQ
           -> synthesis adapter (d->320) -> D2_320 -> combiner(x1,u2)

The adapters can be initialized from PCA of current E2 latents and are then
trained end-to-end using only MSE(final,image).  No u2 teacher, usage KL, or
320-channel side path crosses the discrete bottleneck.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_layer2_fsq_direct as direct  # noqa: E402


base = direct.base


class SynthesisAdapter(nn.Module):
    def __init__(self, channels: int, output_channels: int = 320, hidden: int = 320) -> None:
        super().__init__()
        self.linear = nn.Conv2d(int(channels), int(output_channels), kernel_size=1)
        self.residual = None
        if int(hidden) > 0:
            self.residual = nn.Sequential(
                nn.Conv2d(int(channels), int(hidden), kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(int(hidden), int(output_channels), kernel_size=1),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        output = self.linear(q)
        if self.residual is not None:
            output = output + self.residual(q)
        return output


class ResidualCorrectionCombiner(nn.Module):
    """Identity-safe Layer2 combiner without a global alpha ceiling."""

    def __init__(self, source_inner: nn.Module) -> None:
        super().__init__()
        self.inner = nn.Sequential(
            nn.Conv2d(6, 48, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(48, 3, kernel_size=3, padding=1),
        )
        source_net = getattr(source_inner, "net", None)
        if not isinstance(source_net, nn.Sequential) or len(source_net) < 3:
            raise TypeError(
                "residual combiner requires the source Sequential Conv/PReLU/Conv layout"
            )
        self.inner[0].load_state_dict(source_net[0].state_dict(), strict=True)
        self.inner[1].load_state_dict(source_net[1].state_dict(), strict=True)
        nn.init.zeros_(self.inner[2].weight)
        nn.init.zeros_(self.inner[2].bias)

    def alpha(self) -> torch.Tensor:
        return self.inner[0].weight.new_ones(())

    def forward(self, x1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        correction = self.inner(torch.cat([x1, u2], dim=1))
        return (x1 + correction).clamp(0.0, 1.0)


class AdapterTokenizer(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        e2: nn.Module,
        d2: nn.Module,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.args = args
        self.e3 = e2
        self.d3 = d2
        self.analysis_adapter = nn.Conv2d(320, int(args.fsq_d), kernel_size=1)
        direct.explore.ExploreIFSQQuantizer.config = args
        self.quantizer = direct.explore.ExploreIFSQQuantizer(
            base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d)),
            channels=int(args.fsq_d),
            use_pre_norm=not bool(args.no_pre_norm),
        )
        self.synthesis_adapter = SynthesisAdapter(
            int(args.fsq_d), output_channels=320, hidden=int(args.adapter_hidden)
        )
        self.to(device)

    def encode(self, img: torch.Tensor, x1: torch.Tensor) -> dict[str, torch.Tensor]:
        z320 = base.encode_tensor(self.e3, torch.cat([x1, img], dim=1))
        z3 = self.analysis_adapter(z320)
        encoded = self.quantizer(z3)
        encoded.update({"z3": z3, "z320": z320})
        return encoded

    def decode(
        self,
        q3: torch.Tensor,
        x1: torch.Tensor,
        _z1: torch.Tensor,
        combiner: direct.SafeBlendCombiner,
    ) -> dict[str, torch.Tensor]:
        z320_hat = self.synthesis_adapter(q3)
        u2_raw = self.d3(z320_hat)
        u2_hat = u2_raw.clamp(0.0, 1.0)
        final = combiner(x1, u2_hat)
        return {
            "d3_in": z320_hat,
            "z320_hat": z320_hat,
            "u2_raw": u2_raw,
            "u2_hat": u2_hat,
            "final": final,
        }

    @staticmethod
    def shuffle_q3(q3: torch.Tensor) -> torch.Tensor:
        return base.Layer3FSQTokenizer.shuffle_q3(q3)

    def forward(
        self,
        img: torch.Tensor,
        x1: torch.Tensor,
        z1: torch.Tensor,
        combiner: direct.SafeBlendCombiner,
        *,
        q_mode: str = "normal",
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(img, x1)
        q3 = encoded["q3"]
        if q_mode == "zero":
            q3 = torch.zeros_like(q3)
        elif q_mode == "shuffle":
            q3 = self.shuffle_q3(q3)
        elif q_mode != "normal":
            raise ValueError(f"unknown q_mode {q_mode!r}")
        return {**encoded, **self.decode(q3, x1, z1, combiner), "q3_used": q3}


def build_bundle(args: argparse.Namespace, source: dict, device: torch.device) -> direct.DirectBundle:
    stage2 = base.load_script_module("jsccf_layer2_adapter_source_swin", "train_stage2-swin.py")
    e1, d1, e2, d2, inner = stage2.build_stage2(args, device)
    base.jsccf_io.load_state(e1, source["e1_state_dict"], "adapter_E1", strict=True)
    base.jsccf_io.load_state(d1, source["d1_state_dict"], "adapter_D1", strict=True)
    base.jsccf_io.load_state(e2, source["e2_state_dict"], "adapter_E2_320", strict=True)
    base.jsccf_io.load_state(d2, source["d2_state_dict"], "adapter_D2_320", strict=True)
    base.jsccf_io.load_state(inner, source["combiner_state_dict"], "adapter_combiner", strict=True)
    base.set_trainable(e1, False)
    base.set_trainable(d1, False)
    e1.eval()
    d1.eval()
    tokenizer = AdapterTokenizer(args, e2, d2, device)
    if str(args.adapter_combiner) == "original":
        combiner = direct.SafeBlendCombiner(inner, mode="original", init_alpha=0.5).to(device)
    else:
        combiner = ResidualCorrectionCombiner(inner).to(device)
    return direct.DirectBundle(
        e1=e1,
        d1=d1,
        tokenizer=tokenizer,
        combiner=combiner,
        init_report={
            "source_e2": {"matched_ratio": 1.0, "width": 320},
            "source_d2": {"matched_ratio": 1.0, "width": 320},
        },
    )


@torch.no_grad()
def initialize_pca(
    loader,
    bundle: direct.DirectBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    if str(args.adapter_init) == "random":
        return {"pca_batches": 0.0}
    rng = direct.capture_rng_state()
    samples: list[torch.Tensor] = []
    bundle.e1.eval()
    bundle.d1.eval()
    bundle.tokenizer.e3.eval()
    try:
        for batch_index, (imgs, _labels) in enumerate(loader, start=1):
            if batch_index > int(args.pca_init_batches):
                break
            imgs = imgs.to(device, non_blocking=True)
            layer1 = bundle.layer1(imgs)
            z320 = base.encode_tensor(bundle.tokenizer.e3, torch.cat([layer1["x1"], imgs], dim=1))
            samples.append(z320.permute(0, 2, 3, 1).reshape(-1, 320).float().cpu())
    finally:
        direct.restore_rng_state(rng)
    if not samples:
        raise RuntimeError("PCA adapter initialization collected no z320 samples")
    values = torch.cat(samples, dim=0).double()
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.t().matmul(centered) / float(max(1, int(centered.shape[0]) - 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order[: int(args.fsq_d)]]
    std = eigenvalues[: int(args.fsq_d)].clamp_min(1e-12).sqrt()
    analysis_weight = (components / std.view(1, -1)).t()
    analysis_bias = -analysis_weight.matmul(mean)
    standardized = centered.matmul(components) / std.view(1, -1)
    bounded = torch.tanh(standardized)
    raw_coeff = centered.matmul(components)
    synthesis_scale = (bounded * raw_coeff).sum(dim=0) / bounded.square().sum(dim=0).clamp_min(1e-12)
    synthesis_weight = components * synthesis_scale.view(1, -1)
    bundle.tokenizer.analysis_adapter.weight.copy_(
        analysis_weight.float().view(int(args.fsq_d), 320, 1, 1).to(device)
    )
    bundle.tokenizer.analysis_adapter.bias.copy_(analysis_bias.float().to(device))
    bundle.tokenizer.synthesis_adapter.linear.weight.copy_(
        synthesis_weight.float().view(320, int(args.fsq_d), 1, 1).to(device)
    )
    bundle.tokenizer.synthesis_adapter.linear.bias.copy_(mean.float().to(device))
    pre_norm = bundle.tokenizer.quantizer.pre_norm
    if isinstance(pre_norm, nn.BatchNorm2d):
        pre_norm.weight.fill_(1.0)
        pre_norm.bias.zero_()
        pre_norm.running_mean.zero_()
        pre_norm.running_var.fill_(1.0)
        pre_norm.num_batches_tracked.zero_()
    explained = float(eigenvalues[: int(args.fsq_d)].sum() / eigenvalues.clamp_min(0.0).sum())
    stats = {
        "pca_batches": float(min(int(args.pca_init_batches), len(loader))),
        "pca_tokens": float(values.shape[0]),
        "pca_explained_ratio": explained,
        "pca_top_eigenvalue": float(eigenvalues[0]),
    }
    print(f"[adapter PCA init] {stats}", flush=True)
    return stats


@torch.no_grad()
def calibrate_batch_norm(loader, bundle, args, device: torch.device) -> dict[str, float] | None:
    pre_norm = bundle.tokenizer.quantizer.pre_norm
    max_batches = int(args.bn_calibration_batches)
    if max_batches <= 0 or not isinstance(pre_norm, nn.BatchNorm2d) or not pre_norm.track_running_stats:
        return None
    rng = direct.capture_rng_state()
    total = 0
    channels = int(args.fsq_d)
    value_sum = torch.zeros(channels, device=device, dtype=torch.float64)
    square_sum = torch.zeros_like(value_sum)
    batches = 0
    bundle.e1.eval(); bundle.d1.eval(); bundle.tokenizer.eval()
    try:
        for batch_index, (imgs, _labels) in enumerate(loader, start=1):
            if batch_index > max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            layer1 = bundle.layer1(imgs)
            z320 = base.encode_tensor(bundle.tokenizer.e3, torch.cat([layer1["x1"], imgs], dim=1))
            z3 = bundle.tokenizer.analysis_adapter(z320).double()
            value_sum += z3.sum(dim=(0, 2, 3))
            square_sum += z3.square().sum(dim=(0, 2, 3))
            total += int(z3.shape[0] * z3.shape[2] * z3.shape[3])
            batches += 1
    finally:
        direct.restore_rng_state(rng)
    mean = value_sum / float(total)
    variance = (square_sum / float(total) - mean.square()).clamp_min(1e-8)
    pre_norm.running_mean.copy_(mean.to(pre_norm.running_mean))
    pre_norm.running_var.copy_(variance.to(pre_norm.running_var))
    pre_norm.num_batches_tracked.fill_(batches)
    stats = {"batches": float(batches), "mean_abs": float(mean.abs().mean()), "variance_mean": float(variance.mean())}
    print(f"[adapter BN calibration] {stats}", flush=True)
    return stats


def adapter_name(args: argparse.Namespace) -> str:
    levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    return (
        f"layer2_fsq_adapter_{args.arch}_d{int(args.fsq_d)}_{base.fsq_level_name(levels)}_"
        f"{direct.normalizer_name(args)}_{args.adapter_init}_h{int(args.adapter_hidden)}_{args.adapter_combiner}"
    )


def print_header(args, bundle, train_n: int, val_n: int, init_stats: dict[str, float]) -> None:
    levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d))
    print(f"=== Layer 2 | FSQ latent adapter | {args.arch} ===", flush=True)
    print(f"save_dir={base.resolve_path(args.save_dir)}", flush=True)
    print("实验设计", flush=True)
    print(
        "  frozen Layer1; exact source E2_320/D2_320; inserted 320->d->FSQ->320 adapter; "
        f"combiner={args.adapter_combiner}",
        flush=True,
    )
    print("  no u2 teacher, no usage/KL, no continuous 320-channel bypass", flush=True)
    print(
        f"  d={int(args.fsq_d)} levels={levels} K={base.vocab_size(levels)} adapter_init={args.adapter_init} "
        f"hidden={int(args.adapter_hidden)} init_stats={init_stats}", flush=True,
    )
    print("loss设计", flush=True)
    print("  L=MSE(final,img)", flush=True)
    print("模块选择", flush=True)
    print(
        f"  E1={base.trainable_state(bundle.e1)} D1={base.trainable_state(bundle.d1)} "
        f"E2_320={base.trainable_state(bundle.tokenizer.e3)} adapters=trainable "
        f"D2_320={base.trainable_state(bundle.tokenizer.d3)} "
        f"combiner={args.adapter_combiner}:{base.trainable_state(bundle.combiner)}",
        flush=True,
    )
    print(
        f"  normalizer={direct.normalizer_name(args)} BN_calibration={int(args.bn_calibration_batches)} "
        f"workers={int(args.num_workers)}/{int(args.val_num_workers)}",
        flush=True,
    )
    print(
        f"epochs={int(args.epochs)} train={train_n} valid={val_n} batch={int(args.batch_size)} "
        f"lr={float(args.lr):g} weight_decay={float(args.weight_decay):g}", flush=True,
    )


def save_checkpoint(path, *, epoch, args, metrics, bundle, optimizer, best_psnr, best_goal_psnr, init_stats):
    output = Path(base.resolve_path(path)); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stage": "layer2_fsq_adapter", "epoch": int(epoch), "metrics": metrics,
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "version": str(args.version), "source_layer2_ckpt": str(args.layer2_ckpt),
        "e1_state_dict": bundle.e1.state_dict(), "d1_state_dict": bundle.d1.state_dict(),
        "tokenizer_state_dict": bundle.tokenizer.state_dict(),
        "combiner_state_dict": bundle.combiner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "rng_state": direct.capture_rng_state(),
        "best_psnr": float(best_psnr), "best_goal_psnr": float(best_goal_psnr),
        "init_stats": init_stats,
    }, output)
    print(f"saved checkpoint: {output}", flush=True)


def load_resume(args, bundle, optimizer):
    if not args.resume: return 1, float("-inf"), float("-inf"), None
    payload = torch.load(base.resolve_path(args.resume), map_location="cpu")
    if str(payload.get("stage")) != "layer2_fsq_adapter": raise ValueError("not an adapter checkpoint")
    saved = payload.get("args", {})
    for key in ("arch", "fsq_d", "fsq_levels", "fsq_normalizer", "adapter_hidden", "adapter_init", "adapter_combiner"):
        saved_value = saved.get(key, "original" if key == "adapter_combiner" else None)
        if str(saved_value) != str(getattr(args, key)):
            raise ValueError(f"resume mismatch for {key}: {saved_value!r} != {getattr(args,key)!r}")
    validation_keys = (
        "bn_calibration_batches",
        "selection_min_delta_x1",
        "selection_min_drop_zero",
        "selection_min_drop_shuffle",
    )
    changed_validation = [
        key for key in validation_keys
        if str(saved.get(key)) != str(getattr(args, key))
    ]
    if changed_validation and not bool(args.reset_best_on_resume):
        raise ValueError(
            "resume changes validation protocol "
            f"{changed_validation}; pass --reset-best-on-resume to start a new selection regime"
        )
    base.jsccf_io.load_state(bundle.e1, payload["e1_state_dict"], "resume_E1", strict=True)
    base.jsccf_io.load_state(bundle.d1, payload["d1_state_dict"], "resume_D1", strict=True)
    base.jsccf_io.load_state(bundle.tokenizer, payload["tokenizer_state_dict"], "resume_adapter", strict=True)
    base.jsccf_io.load_state(bundle.combiner, payload["combiner_state_dict"], "resume_combiner", strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    direct.restore_rng_state(payload["rng_state"])
    best = float(payload.get("best_psnr", -math.inf))
    best_goal = float(payload.get("best_goal_psnr", -math.inf))
    if bool(args.reset_best_on_resume):
        best = float("-inf")
        best_goal = float("-inf")
        print("reset best scores after resume", flush=True)
    return int(payload["epoch"]) + 1, best, best_goal, payload.get("init_stats")


def train(args, source):
    base.seed_everything(int(args.seed)); cfg = base.jsccf_io.build_config(args, encoder_in_chans=3)
    train_loader, val_loader = base.get_loader(cfg); bundle = build_bundle(args, source, cfg.device)
    params = list(bundle.tokenizer.parameters()) + list(bundle.combiner.parameters())
    optimizer = optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    start, best, best_goal, saved_init = load_resume(args, bundle, optimizer)
    init_stats = saved_init if saved_init is not None else initialize_pca(train_loader, bundle, args, cfg.device)
    print_header(args, bundle, len(train_loader.dataset), len(val_loader.dataset), init_stats)
    levels = base.parse_fsq_levels(args.fsq_levels, int(args.fsq_d)); name = adapter_name(args)
    last_train = {}; last_checkpoint = {}
    for epoch in range(start, int(args.epochs) + 1):
        bundle.e1.eval(); bundle.d1.eval(); bundle.tokenizer.train(); bundle.combiner.train()
        meters = base.meters(direct.METRIC_NAMES); hist = torch.zeros(base.vocab_size(levels)); level_hists = direct.make_level_hists(levels)
        began = time.time()
        for batch_index, (imgs, _labels) in enumerate(train_loader, start=1):
            if int(args.max_train_batches) > 0 and batch_index > int(args.max_train_batches): break
            imgs = imgs.to(cfg.device, non_blocking=True); layer1, out = direct.forward_direct(bundle, imgs)
            losses = direct.compute_losses(out, imgs, args); optimizer.zero_grad(set_to_none=True); losses["loss"].backward()
            if float(args.grad_clip_norm) > 0: torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip_norm))
            optimizer.step(); direct.update_metrics(meters, out, layer1, imgs, losses, bundle.combiner, args)
            base.update_code_hist(hist, out["idx3"]); direct.update_level_hists(level_hists, out["codes"])
        last_train = direct.finalize_metrics(meters, hist, level_hists, args)
        print(f"[layer2-fsq-adapter train {epoch:03d}/{int(args.epochs):03d}] {direct.display_metrics(last_train)} time={time.time()-began:.1f}s", flush=True)
        checkpoint_metrics = last_train
        if base.should_validate(args, epoch):
            calibration = calibrate_batch_norm(train_loader, bundle, args, cfg.device)
            val = direct.validate(val_loader, bundle, args, cfg.device)
            if calibration: val.update({f"bn_calibration_{k}": v for k,v in calibration.items()})
            eligible = direct.goal_eligible(val, args); val["goal_eligible"] = float(eligible); checkpoint_metrics = val
            print(f"[layer2-fsq-adapter val {epoch:03d}] {direct.display_metrics(val)} goal_eligible={eligible}", flush=True)
            score = float(val["psnr_final"]); improved = score > best; improved_goal = eligible and score > best_goal
            if improved: best = score
            if improved_goal: best_goal = score
            if improved: save_checkpoint(base.jsccf_io.ckpt_path(args,name,"best"),epoch=epoch,args=args,metrics=val,bundle=bundle,optimizer=optimizer,best_psnr=best,best_goal_psnr=best_goal,init_stats=init_stats)
            if improved_goal: save_checkpoint(base.jsccf_io.ckpt_path(args,name,"goal_best"),epoch=epoch,args=args,metrics=val,bundle=bundle,optimizer=optimizer,best_psnr=best,best_goal_psnr=best_goal,init_stats=init_stats)
        last_checkpoint = checkpoint_metrics
        if base.should_save_latest(args, epoch): save_checkpoint(base.jsccf_io.ckpt_path(args,name,"latest"),epoch=epoch,args=args,metrics=checkpoint_metrics,bundle=bundle,optimizer=optimizer,best_psnr=best,best_goal_psnr=best_goal,init_stats=init_stats)
    save_checkpoint(base.jsccf_io.ckpt_path(args,name,"latest"),epoch=int(args.epochs),args=args,metrics=last_checkpoint or last_train,bundle=bundle,optimizer=optimizer,best_psnr=best,best_goal_psnr=best_goal,init_stats=init_stats)


@torch.no_grad()
def smoke(args, source):
    device = torch.device("cuda:0" if not args.cpu and torch.cuda.is_available() else "cpu"); bundle = build_bundle(args, source, device)
    imgs = torch.rand(int(args.smoke_batch_size),3,256,256,device=device); layer1,out = direct.forward_direct(bundle,imgs)
    print(
        f"[smoke adapter] x1={tuple(layer1['x1'].shape)} z320={tuple(out['z320'].shape)} "
        f"z3={tuple(out['z3'].shape)} z320_hat={tuple(out['z320_hat'].shape)} "
        f"final={tuple(out['final'].shape)} max_abs_final_minus_x1="
        f"{float((out['final']-layer1['x1']).abs().max()):.6g}",flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--adapter-init",choices=["pca","random"],default="pca")
    parser.add_argument("--pca-init-batches",type=int,default=3); parser.add_argument("--adapter-hidden",type=int,default=320)
    parser.add_argument("--adapter-combiner",choices=["original","residual"],default="original")
    adapter, remaining = parser.parse_known_args(); argv=sys.argv
    try: sys.argv=[argv[0],*remaining]; args=direct.parse_args()
    finally: sys.argv=argv
    for k,v in vars(adapter).items(): setattr(args,k,v)
    if not direct.cli_option_present(remaining,"--combiner-mode"): args.combiner_mode="original"
    if not direct.cli_option_present(remaining,"--val-num-workers"): args.val_num_workers=4
    if not direct.cli_option_present(remaining,"--selection-min-drop-zero"): args.selection_min_drop_zero=.1
    if not direct.cli_option_present(remaining,"--selection-min-drop-shuffle"): args.selection_min_drop_shuffle=.1
    return args


def main():
    args=parse_args(); args.stage="layer2_fsq_adapter"; base.apply_preset(args)
    if args.arch!="swin" or args.swin_codec!="no_compressor": raise ValueError("adapter route currently requires Swin no_compressor source")
    if args.combiner_mode!="original" or args.condition_mode!="none": raise ValueError("adapter route requires original combiner and no condition")
    if float(args.lambda_usage)!=0 or float(args.lambda_u2_img)!=0: raise ValueError("adapter route requires zero usage/u2 losses")
    if int(args.pca_init_batches)<=0 or int(args.adapter_hidden)<0: raise ValueError("invalid adapter initialization settings")
    source=base.load_teacher_checkpoint_for_args(args); args._usage_weight=0.; direct.explore.ExploreIFSQQuantizer.config=args; base.check_jsccf_args(args)
    if not direct.cli_option_present(sys.argv[1:],"--save-dir"): args.save_dir=str(THIS_DIR/"checkpoints-adapter")
    if not direct.cli_option_present(sys.argv[1:],"--version"): args.version="layer2-fsq-adapter"
    name=adapter_name(args)
    if not direct.cli_option_present(sys.argv[1:],"--log-file"): args.log_file=str(THIS_DIR/"logs-adapter"/f"{name}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}.log")
    Path(base.resolve_path(args.save_dir)).mkdir(parents=True,exist_ok=True); base.setup_log_file(args.log_file)
    base.write_json(Path(base.resolve_path(args.save_dir))/f"{name}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}_args.json",{k:v for k,v in vars(args).items() if not k.startswith('_')})
    if args.smoke_shapes: smoke(args,source)
    else: train(args,source)


if __name__=="__main__": main()
