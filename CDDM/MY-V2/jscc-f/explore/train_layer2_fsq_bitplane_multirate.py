#!/usr/bin/env python3
"""Shared fixed-bit FSQ capacity sweep with increasing discrete dimensions.

The original 5^3/9^3/17^3 sweep spends 7/10/13 bits per token only on finer
precision in the same three-dimensional latent.  This control instead uses
7/10/13 binary FSQ dimensions (K=128/1024/8192), so a larger bit budget adds
new discrete degrees of freedom.  It retains the exact 320-channel Swin
E2/D2 contract through the PCA-initialized adapter route.

Binary codes are represented as 0/1 and inactive higher-rate dimensions are
fixed to 0.  Therefore every lower-rate codeword is an exact candidate in the
higher-rate model after appending zero bits.  Before synthesis, 0/1 is mapped
to -1/+1 to match the PCA adapter scale.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_layer2_fsq_adapter as adapter  # noqa: E402
import train_layer2_fsq_multirate as multirate  # noqa: E402
import train_layer2_fsq_direct as direct  # noqa: E402


base = direct.base


def parse_rate_bits(value: str) -> list[int]:
    bits = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(bits) < 2 or bits != sorted(set(bits)) or any(bit <= 0 for bit in bits):
        raise ValueError(f"--rate-bits must be strictly increasing positive integers, got {bits}")
    return bits


def rate_args(args: argparse.Namespace, bits: int) -> argparse.Namespace:
    current = copy.copy(args)
    current.fsq_d = int(bits)
    current.fsq_levels = [2] * int(bits)
    return current


def make_rate_states(args, rates: list[int], *, validation: bool) -> dict[int, dict]:
    names = direct.METRIC_NAMES + (direct.VAL_ABLATION_METRICS if validation and args.val_ablation else [])
    states = {}
    for bits in rates:
        current = rate_args(args, bits)
        states[bits] = {
            "args": current,
            "meters": base.meters(names),
            "hist": torch.zeros(2**int(bits), dtype=torch.float32),
            "level_hists": direct.make_level_hists([2] * int(bits)),
        }
    return states


def binary_branch(z_norm: torch.Tensor, bits: int, max_bits: int) -> dict[str, torch.Tensor]:
    soft01 = (z_norm[:, :bits] + 1.0) * 0.5
    hard01_float = base.round_ste(soft01).clamp(0.0, 1.0)
    codes = hard01_float.detach().long()
    full01 = z_norm.new_zeros(z_norm.shape[0], max_bits, z_norm.shape[2], z_norm.shape[3])
    full01[:, :bits] = soft01 + (hard01_float - soft01).detach()
    q_bipolar = full01 * 2.0 - 1.0
    powers = torch.tensor(
        [2**power for power in reversed(range(bits))], device=codes.device, dtype=torch.long
    ).view(1, bits, 1, 1)
    indices = (codes * powers).sum(dim=1)
    hard_bipolar = hard01_float * 2.0 - 1.0
    zero = z_norm.new_zeros(())
    return {
        "q3": q_bipolar,
        "q3_hard": q_bipolar.detach(),
        "codes": codes,
        "idx3": indices,
        "fsq_mse": torch.nn.functional.mse_loss(hard_bipolar.detach(), z_norm[:, :bits].detach()),
        "usage_kl": zero,
        "soft_level_entropy_bits": zero,
        "soft_usage_entropy_bits": zero,
    }


def continuous_branch(z_norm: torch.Tensor, bits: int, max_bits: int) -> torch.Tensor:
    full01 = z_norm.new_zeros(z_norm.shape[0], max_bits, z_norm.shape[2], z_norm.shape[3])
    full01[:, :bits] = (z_norm[:, :bits] + 1.0) * 0.5
    return full01 * 2.0 - 1.0


def zero_branch(z_norm: torch.Tensor) -> torch.Tensor:
    return torch.full_like(z_norm, -1.0)


def forward_bitplanes(bundle, imgs: torch.Tensor, rates: list[int]):
    with torch.no_grad():
        layer1 = bundle.layer1(imgs)
    z320 = base.encode_tensor(bundle.tokenizer.e3, torch.cat([layer1["x1"], imgs], dim=1))
    z3 = bundle.tokenizer.analysis_adapter(z320)
    z_norm = torch.tanh(bundle.tokenizer.quantizer.pre_norm(z3))
    outputs = {}
    max_bits = int(rates[-1])
    for bits in rates:
        encoded = binary_branch(z_norm, bits, max_bits)
        decoded = bundle.tokenizer.decode(encoded["q3"], layer1["x1"], layer1["z1"], bundle.combiner)
        outputs[bits] = {
            **encoded,
            **decoded,
            "z3": z3,
            "z3_norm": z_norm,
            "z320": z320,
            "q3_used": encoded["q3"],
        }
    return layer1, z_norm, outputs


@torch.no_grad()
def validate(loader, bundle, args, device, rates, weights, margins):
    bundle.e1.eval(); bundle.d1.eval(); bundle.tokenizer.eval(); bundle.combiner.eval()
    states = make_rate_states(args, rates, validation=True)
    objective_meters = base.meters(multirate.objective_metric_names(rates))
    max_bits = rates[-1]
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if int(args.max_val_batches) > 0 and batch_index > int(args.max_val_batches): break
        imgs = imgs.to(device, non_blocking=True)
        layer1, z_norm, outputs = forward_bitplanes(bundle, imgs, rates)
        objective, branch_losses = multirate.compute_multirate_loss(
            outputs, imgs, rates, weights, margins, float(args.lambda_monotonic)
        )
        multirate.update_objective_meters(objective_meters, objective, int(imgs.shape[0]))
        for bits in rates:
            multirate.update_rate_state(states[bits], outputs[bits], layer1, imgs, branch_losses[bits], bundle)
        if args.val_ablation:
            token_count = int(imgs.shape[0] * z_norm.shape[2] * z_norm.shape[3])
            permutation = torch.randperm(token_count, device=device)
            zero_final = bundle.tokenizer.decode(
                zero_branch(z_norm), layer1["x1"], layer1["z1"], bundle.combiner
            )["final"]
            for bits in rates:
                continuous_final = bundle.tokenizer.decode(
                    continuous_branch(z_norm, bits, max_bits), layer1["x1"], layer1["z1"], bundle.combiner
                )["final"]
                shuffled = multirate.shuffled_with_perm(outputs[bits]["q3"], permutation)
                shuffle_final = bundle.tokenizer.decode(
                    shuffled, layer1["x1"], layer1["z1"], bundle.combiner
                )["final"]
                multirate.update_rate_ablation(
                    states[bits], outputs[bits], imgs, continuous_final, zero_final, shuffle_final
                )
    return multirate.finalize_all_metrics(objective_meters, states, rates)


def route_name(args, rates):
    tag = "-".join(str(bits) for bits in rates)
    return (
        f"layer2_fsq_bitplane_{args.arch}_b{tag}_{direct.normalizer_name(args)}_"
        f"{args.adapter_init}_h{int(args.adapter_hidden)}_{args.adapter_combiner}"
    )


def save_checkpoint(path, *, epoch, args, rates, metrics, bundle, optimizer, best, best_goal, init_stats):
    output = Path(base.resolve_path(path)); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stage": "layer2_fsq_bitplane_multirate", "epoch": int(epoch), "metrics": metrics,
        "args": {k:v for k,v in vars(args).items() if not k.startswith("_")}, "rate_bits": rates,
        "rates": {str(bits): {"vocab_size": 2**bits, "fixed_bits_per_token": bits} for bits in rates},
        "source_layer2_ckpt": str(args.layer2_ckpt), "tokenizer_state_dict": bundle.tokenizer.state_dict(),
        "e1_state_dict": bundle.e1.state_dict(), "d1_state_dict": bundle.d1.state_dict(),
        "combiner_state_dict": bundle.combiner.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": direct.capture_rng_state(), "best_score": float(best), "best_goal_score": float(best_goal),
        "init_stats": init_stats,
    }, output)
    print(f"saved checkpoint: {output}", flush=True)


def load_resume(args, rates, bundle, optimizer):
    if not args.resume: return 1, -math.inf, -math.inf, None
    payload=torch.load(base.resolve_path(args.resume),map_location="cpu")
    if payload.get("stage")!="layer2_fsq_bitplane_multirate" or payload.get("rate_bits")!=rates:
        raise ValueError("bitplane resume route/rates mismatch")
    for key in ("arch","fsq_normalizer","adapter_init","adapter_hidden","adapter_combiner","lambda_monotonic","monotonic_margins","recon_weights"):
        saved_value = payload["args"].get(key, "original" if key == "adapter_combiner" else None)
        if str(saved_value)!=str(getattr(args,key)): raise ValueError(f"resume mismatch {key}")
    validation_keys=("bn_calibration_batches","selection_min_delta_x1","selection_min_drop_zero","selection_min_drop_shuffle","selection_min_rate_gain_db","selection_min_per_image_strict_ratio")
    changed_validation=[key for key in validation_keys if str(payload["args"].get(key))!=str(getattr(args,key))]
    if changed_validation and not bool(args.reset_best_on_resume):
        raise ValueError(f"resume changes validation protocol {changed_validation}; pass --reset-best-on-resume")
    base.jsccf_io.load_state(bundle.e1,payload["e1_state_dict"],"resume_E1",strict=True)
    base.jsccf_io.load_state(bundle.d1,payload["d1_state_dict"],"resume_D1",strict=True)
    base.jsccf_io.load_state(bundle.tokenizer,payload["tokenizer_state_dict"],"resume_bitplane",strict=True)
    base.jsccf_io.load_state(bundle.combiner,payload["combiner_state_dict"],"resume_combiner",strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"]); direct.restore_rng_state(payload["rng_state"])
    best=float(payload.get("best_score",-math.inf));best_goal=float(payload.get("best_goal_score",-math.inf))
    if bool(args.reset_best_on_resume):
        best=-math.inf;best_goal=-math.inf;print("reset best scores after resume",flush=True)
    return int(payload["epoch"])+1,best,best_goal,payload.get("init_stats")


def print_header(args,bundle,rates,weights,margins,train_n,val_n,init_stats):
    print("=== Layer 2 | nested binary FSQ bit-budget control | Swin ===",flush=True)
    print("实验设计",flush=True)
    print("  exact E2/D2 width320 + PCA adapter; shared model; inactive bits=0; lower codes embed in higher codes",flush=True)
    print("  rate="+", ".join(f"{bits}bits:K={2**bits}" for bits in rates)+"; no u2 teacher/KL",flush=True)
    print("loss设计",flush=True)
    print(f"  mean recon weights={weights} + lambda={float(args.lambda_monotonic):g} monotonic hinge margins={margins}",flush=True)
    print("模块选择",flush=True)
    print(
        f"  E1/D1 frozen; E2_320/adapters/D2_320/{args.adapter_combiner}-combiner trainable; "
        f"init={init_stats}",flush=True,
    )
    print(f"  BN calibration={int(args.bn_calibration_batches)} workers={int(args.num_workers)}/{int(args.val_num_workers)} epochs={int(args.epochs)} train={train_n} val={val_n}",flush=True)


def train(args, source):
    rates=parse_rate_bits(args.rate_bits); weights=multirate.parse_float_list(args.recon_weights,len(rates),"weights")
    margins=multirate.parse_float_list(args.monotonic_margins,len(rates)-1,"margins")
    base.seed_everything(int(args.seed)); cfg=base.jsccf_io.build_config(args,encoder_in_chans=3)
    train_loader,val_loader=base.get_loader(cfg); bundle=adapter.build_bundle(args,source,cfg.device)
    params=list(bundle.tokenizer.parameters())+list(bundle.combiner.parameters()); optimizer=optim.AdamW(params,lr=float(args.lr),weight_decay=float(args.weight_decay))
    start,best,best_goal,saved_init=load_resume(args,rates,bundle,optimizer)
    init_stats=saved_init if saved_init is not None else adapter.initialize_pca(train_loader,bundle,args,cfg.device)
    print_header(args,bundle,rates,weights,margins,len(train_loader.dataset),len(val_loader.dataset),init_stats)
    name=route_name(args,rates); last_train={}; last_checkpoint={}
    for epoch in range(start,int(args.epochs)+1):
        bundle.e1.eval();bundle.d1.eval();bundle.tokenizer.train();bundle.combiner.train()
        states=make_rate_states(args,rates,validation=False); objective_meters=base.meters(multirate.objective_metric_names(rates)); began=time.time()
        for batch_index,(imgs,_labels) in enumerate(train_loader,start=1):
            if int(args.max_train_batches)>0 and batch_index>int(args.max_train_batches):break
            imgs=imgs.to(cfg.device,non_blocking=True);layer1,_z,outputs=forward_bitplanes(bundle,imgs,rates)
            objective,branch_losses=multirate.compute_multirate_loss(outputs,imgs,rates,weights,margins,float(args.lambda_monotonic))
            optimizer.zero_grad(set_to_none=True);objective["loss"].backward()
            if float(args.grad_clip_norm)>0:torch.nn.utils.clip_grad_norm_(params,float(args.grad_clip_norm))
            optimizer.step();multirate.update_objective_meters(objective_meters,objective,int(imgs.shape[0]))
            for bits in rates:multirate.update_rate_state(states[bits],outputs[bits],layer1,imgs,branch_losses[bits],bundle)
        last_train=multirate.finalize_all_metrics(objective_meters,states,rates)
        print(f"[layer2-fsq-bitplane train {epoch:03d}/{int(args.epochs):03d}] {multirate.display_metrics(last_train,rates)} time={time.time()-began:.1f}s",flush=True)
        checkpoint_metrics=last_train
        if base.should_validate(args,epoch):
            calibration=adapter.calibrate_batch_norm(train_loader,bundle,args,cfg.device)
            val=validate(val_loader,bundle,args,cfg.device,rates,weights,margins)
            if calibration:val.update({f"bn_calibration_{k}":v for k,v in calibration.items()})
            eligible=multirate.goal_eligible(val,args,rates);val["goal_eligible"]=float(eligible);checkpoint_metrics=val
            print(f"[layer2-fsq-bitplane val {epoch:03d}] {multirate.display_metrics(val,rates)} goal_eligible={eligible}",flush=True)
            score=sum(val[f"l{b}_psnr_final"] for b in rates)/len(rates);improved=score>best;improved_goal=eligible and score>best_goal
            if improved:best=score
            if improved_goal:best_goal=score
            if improved:save_checkpoint(base.jsccf_io.ckpt_path(args,name,"best"),epoch=epoch,args=args,rates=rates,metrics=val,bundle=bundle,optimizer=optimizer,best=best,best_goal=best_goal,init_stats=init_stats)
            if improved_goal:save_checkpoint(base.jsccf_io.ckpt_path(args,name,"goal_best"),epoch=epoch,args=args,rates=rates,metrics=val,bundle=bundle,optimizer=optimizer,best=best,best_goal=best_goal,init_stats=init_stats)
        last_checkpoint=checkpoint_metrics
        if base.should_save_latest(args,epoch):save_checkpoint(base.jsccf_io.ckpt_path(args,name,"latest"),epoch=epoch,args=args,rates=rates,metrics=checkpoint_metrics,bundle=bundle,optimizer=optimizer,best=best,best_goal=best_goal,init_stats=init_stats)
    save_checkpoint(base.jsccf_io.ckpt_path(args,name,"latest"),epoch=int(args.epochs),args=args,rates=rates,metrics=last_checkpoint or last_train,bundle=bundle,optimizer=optimizer,best=best,best_goal=best_goal,init_stats=init_stats)


@torch.no_grad()
def smoke(args,source):
    rates=parse_rate_bits(args.rate_bits);device=torch.device("cuda:0" if not args.cpu and torch.cuda.is_available() else "cpu");bundle=adapter.build_bundle(args,source,device)
    imgs=torch.rand(int(args.smoke_batch_size),3,256,256,device=device);layer1,_z,outputs=forward_bitplanes(bundle,imgs,rates)
    print(f"[smoke bitplane] x1={tuple(layer1['x1'].shape)} "+" ".join(f"b{b}:q{tuple(outputs[b]['q3'].shape)} idx=[{int(outputs[b]['idx3'].min())},{int(outputs[b]['idx3'].max())}]" for b in rates),flush=True)


def parse_args():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument("--rate-bits",default="7,10,13")
    parser.add_argument("--recon-weights",default="1");parser.add_argument("--lambda-monotonic",type=float,default=1.)
    parser.add_argument("--monotonic-margins",default="1e-5");parser.add_argument("--selection-min-rate-gain-db",type=float,default=0.)
    parser.add_argument("--selection-min-per-image-strict-ratio",type=float,default=0.)
    bitargs,remaining=parser.parse_known_args();argv=sys.argv
    try:sys.argv=[argv[0],*remaining];args=adapter.parse_args()
    finally:sys.argv=argv
    for k,v in vars(bitargs).items():setattr(args,k,v)
    return args


def main():
    args=parse_args();rates=parse_rate_bits(args.rate_bits);args.fsq_d=rates[-1];args.fsq_levels=[2]*rates[-1];args.stage="layer2_fsq_bitplane_multirate"
    base.apply_preset(args)
    if args.arch!="swin" or args.combiner_mode!="original" or args.condition_mode!="none":raise ValueError("bitplane route requires Swin/original/no-condition")
    if float(args.lambda_usage)!=0 or float(args.lambda_u2_img)!=0:raise ValueError("bitplane route requires no usage/u2 losses")
    if not math.isfinite(float(args.lambda_monotonic)) or float(args.lambda_monotonic)<0:raise ValueError("invalid monotonic weight")
    weights=multirate.parse_float_list(args.recon_weights,len(rates),"weights");margins=multirate.parse_float_list(args.monotonic_margins,len(rates)-1,"margins")
    if any(w<0 for w in weights) or sum(weights)<=0 or any(m<0 for m in margins):raise ValueError("invalid weights/margins")
    source=base.load_teacher_checkpoint_for_args(args);args._usage_weight=0.;direct.explore.ExploreIFSQQuantizer.config=args;base.check_jsccf_args(args)
    if not direct.cli_option_present(sys.argv[1:],"--save-dir"):args.save_dir=str(THIS_DIR/"checkpoints-bitplane")
    if not direct.cli_option_present(sys.argv[1:],"--version"):args.version="layer2-fsq-bitplane"
    name=route_name(args,rates)
    if not direct.cli_option_present(sys.argv[1:],"--log-file"):args.log_file=str(THIS_DIR/"logs-bitplane"/f"{name}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}.log")
    Path(base.resolve_path(args.save_dir)).mkdir(parents=True,exist_ok=True);base.setup_log_file(args.log_file)
    base.write_json(Path(base.resolve_path(args.save_dir))/f"{name}_jscc_f_{base.jsccf_io.safe_artifact_name(args.version)}_args.json",{k:v for k,v in vars(args).items() if not k.startswith('_')})
    if args.smoke_shapes:smoke(args,source)
    else:train(args,source)


if __name__=="__main__":main()
