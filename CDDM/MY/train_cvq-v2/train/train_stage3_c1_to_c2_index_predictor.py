from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from shared import (
    add_common_cli,
    averaged,
    batch_metric_mean,
    build_encoder_decoder,
    build_predictor,
    build_quantizer,
    ckpt_path,
    cvq_io,
    default_stage2_ckpt,
    default_v01_save_dir,
    ensure_common_args,
    format_metrics,
    freeze_module,
    get_loader,
    load_v01_checkpoint,
    meters,
    print_epoch,
    print_v01_header,
    psnr_per_image,
    confidence_keep_mask,
    quantizer_artifact_part,
    quantizer_lookup,
    real_awgn,
    recon_loss,
    resolve_path,
    save_v01_checkpoint,
    seed_everything,
    setup_stage_log,
    should_save_latest,
    should_validate,
    soft_quantizer_lookup,
    split_c1_c2,
    topk_accuracies,
    with_log_keys,
    write_json,
)


def load_stage2(args, encoder, decoder, quantizer) -> None:
    src = args.init_stage2_ckpt or default_stage2_ckpt(args)
    obj = load_v01_checkpoint(src)
    encoder.load_state_dict(obj["encoder_state_dict"], strict=True)
    decoder.load_state_dict(obj["decoder_state_dict"], strict=True)
    quantizer.load_state_dict(obj["quantizer_state_dict"], strict=True)
    print(f"stage3_source_stage2={resolve_path(src)}")


STAGE3_KEYS = [
    "loss",
    "loss_ce",
    "loss_prior_ce",
    "loss_soft_rec",
    "loss_c2_proposal",
    "top1_acc",
    "top5_acc",
    "top10_acc",
    "prior_top1_acc",
    "prior_top5_acc",
    "prior_top10_acc",
    "prior_recall64",
    "prior_recall128",
    "prior_recall256",
    "c2_proposal_mse",
    "psnr_c1_only",
    "psnr_q_c2_gt",
    "psnr_pred_all",
    "psnr_pred_safe",
    "psnr_soft",
    "pred_gain",
    "pred_safe_gain",
    "soft_gain",
    "safe_keep_ratio",
]


def unpack_predictor_output(out) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError(f"predictor tuple output must be (logits, c2_proposal), got len={len(out)}")
        return out[0], out[1]
    return out, None


def c2_proposal_prior_logits(quantizer, c2_proposal: torch.Tensor | None, args: argparse.Namespace) -> torch.Tensor | None:
    if c2_proposal is None:
        return None
    if not hasattr(quantizer, "patch_size"):
        raise ValueError("C2 proposal prior is currently implemented for patch_vq only")
    p = int(quantizer.patch_size)
    bsz, channels, h, w = c2_proposal.shape
    patches = F.unfold(c2_proposal.reshape(bsz * channels, 1, h, w), kernel_size=p, stride=p)
    patches = patches.transpose(1, 2).reshape(bsz * channels * int(quantizer.grid_h) * int(quantizer.grid_w), p, p)
    codebook = quantizer.codebook.float()
    tau = max(float(args.prior_tau), 1e-6)
    chunk = max(1, int(getattr(args, "prior_chunk_size", getattr(quantizer, "chunk_size", 128))))
    logits = []
    for start in range(0, patches.shape[0], chunk):
        xb = patches[start : start + chunk].float()
        dist = (xb.unsqueeze(1) - codebook.unsqueeze(0)).square().mean(dim=(2, 3))
        logits.append(-dist / tau)
    return torch.cat(logits, dim=0).reshape(bsz, channels, int(quantizer.grid_h), int(quantizer.grid_w), int(args.k))


def apply_c2_prior(cls_logits: torch.Tensor, prior_logits: torch.Tensor | None, args: argparse.Namespace) -> torch.Tensor:
    if prior_logits is None:
        return cls_logits
    logits = cls_logits + float(args.prior_alpha) * prior_logits.to(dtype=cls_logits.dtype)
    topm = int(args.prior_topm)
    if topm <= 0 or topm >= logits.shape[-1]:
        return logits
    keep_idx = prior_logits.topk(topm, dim=-1).indices
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep.scatter_(-1, keep_idx, True)
    return logits.masked_fill(~keep, -1.0e4)


def forward_stage3_logits(predictor, quantizer, c1_rx: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    cls_logits, c2_proposal = unpack_predictor_output(predictor(c1_rx))
    prior_logits = c2_proposal_prior_logits(quantizer, c2_proposal, args) if bool(args.use_c2_proposal_prior) else None
    logits = apply_c2_prior(cls_logits, prior_logits, args)
    return logits, prior_logits, c2_proposal


@torch.no_grad()
def prior_candidate_metrics(prior_logits: torch.Tensor | None, idx: torch.Tensor) -> dict[str, float]:
    if prior_logits is None:
        return {
            "prior_top1_acc": float("nan"),
            "prior_top5_acc": float("nan"),
            "prior_top10_acc": float("nan"),
            "prior_recall64": float("nan"),
            "prior_recall128": float("nan"),
            "prior_recall256": float("nan"),
        }
    out = {f"prior_{k}": v for k, v in topk_accuracies(prior_logits, idx).items()}
    for k in (64, 128, 256):
        kk = min(k, prior_logits.shape[-1])
        pred = prior_logits.topk(kk, dim=-1).indices
        out[f"prior_recall{k}"] = float(pred.eq(idx.unsqueeze(-1)).any(dim=-1).float().mean().item())
    return out


def prior_cross_entropy(prior_logits: torch.Tensor | None, idx: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if prior_logits is None:
        return torch.zeros((), device=idx.device)
    return F.cross_entropy(prior_logits.reshape(-1, int(args.k)), idx.reshape(-1), label_smoothing=float(args.label_smoothing))


def expand_token_mask(mask: torch.Tensor, q_pred: torch.Tensor, quantizer) -> torch.Tensor:
    if hasattr(quantizer, "patch_size") and mask.ndim == 4:
        p = int(quantizer.patch_size)
        return mask.unsqueeze(-1).unsqueeze(-1).repeat_interleave(p, dim=2).repeat_interleave(p, dim=3).reshape_as(q_pred)
    return mask.unsqueeze(-1).unsqueeze(-1).to(dtype=q_pred.dtype)


def selected_prediction(quantizer, logits: torch.Tensor, q_pred: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, float]:
    threshold = float(args.pred_safe_threshold)
    if threshold <= 0.0:
        return q_pred, 1.0
    keep = confidence_keep_mask(logits, threshold)
    latent_mask = expand_token_mask(keep.to(dtype=q_pred.dtype), q_pred, quantizer)
    return q_pred * latent_mask, float(keep.float().mean().item())


def update_recon_metrics(m: dict, *, x_c1, x_gt, x_pred, x_safe, x_soft, imgs, bsz: int, safe_keep: float) -> None:
    x_c1 = x_c1.detach()
    x_gt = x_gt.detach()
    x_pred = x_pred.detach()
    x_safe = x_safe.detach()
    x_soft = x_soft.detach()
    psnr_c1 = batch_metric_mean(psnr_per_image(x_c1, imgs))
    psnr_gt = batch_metric_mean(psnr_per_image(x_gt, imgs))
    psnr_pred = batch_metric_mean(psnr_per_image(x_pred, imgs))
    psnr_safe = batch_metric_mean(psnr_per_image(x_safe, imgs))
    psnr_soft = batch_metric_mean(psnr_per_image(x_soft, imgs))
    m["psnr_c1_only"].update(psnr_c1, bsz)
    m["psnr_q_c2_gt"].update(psnr_gt, bsz)
    m["psnr_pred_all"].update(psnr_pred, bsz)
    m["psnr_pred_safe"].update(psnr_safe, bsz)
    m["psnr_soft"].update(psnr_soft, bsz)
    m["pred_gain"].update(psnr_pred - psnr_c1, bsz)
    m["pred_safe_gain"].update(psnr_safe - psnr_c1, bsz)
    m["soft_gain"].update(psnr_soft - psnr_c1, bsz)
    m["safe_keep_ratio"].update(safe_keep, bsz)


@torch.no_grad()
def validate(loader, encoder, decoder, quantizer, predictor, args) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    quantizer.eval()
    predictor.eval()
    m = meters(STAGE3_KEYS)
    device = next(predictor.parameters()).device
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
        c1, c2 = split_c1_c2(z_norm, args)
        c1_rx = real_awgn(c1, float(args.snr_db))
        q_gt, idx = quantizer.encode(c2)
        logits, prior_logits, c2_proposal = forward_stage3_logits(predictor, quantizer, c1_rx, args)
        loss_ce = F.cross_entropy(logits.reshape(-1, int(args.k)), idx.reshape(-1), label_smoothing=float(args.label_smoothing))
        loss_prior_ce = prior_cross_entropy(prior_logits, idx, args)
        q_soft = soft_quantizer_lookup(quantizer, logits, tau=float(args.soft_tau))
        x_soft = decoder(torch.cat([c1_rx, q_soft], dim=1)).clamp(0.0, 1.0)
        loss_soft = recon_loss(x_soft, imgs)
        if c2_proposal is None:
            loss_c2 = torch.zeros((), device=imgs.device)
            c2_mse = float("nan")
        else:
            loss_c2 = F.mse_loss(c2_proposal.float(), c2.float())
            c2_mse = float(loss_c2.item())
        loss = (
            loss_ce
            + float(args.lambda_prior_ce) * loss_prior_ce
            + float(args.lambda_soft_rec) * loss_soft
            + float(args.lambda_c2_proposal) * loss_c2
        )
        idx_hat = logits.argmax(dim=-1)
        q_pred = quantizer_lookup(quantizer, idx_hat)
        q_safe, safe_keep = selected_prediction(quantizer, logits, q_pred, args)
        zero = torch.zeros_like(q_pred)
        x_c1 = decoder(torch.cat([c1_rx, zero], dim=1)).clamp(0.0, 1.0)
        x_gt = decoder(torch.cat([c1_rx, q_gt], dim=1)).clamp(0.0, 1.0)
        x_pred = decoder(torch.cat([c1_rx, q_pred], dim=1)).clamp(0.0, 1.0)
        x_safe = decoder(torch.cat([c1_rx, q_safe], dim=1)).clamp(0.0, 1.0)
        acc = topk_accuracies(logits, idx)
        bsz = imgs.shape[0]
        m["loss"].update(float(loss.item()), bsz)
        m["loss_ce"].update(float(loss_ce.item()), bsz)
        m["loss_prior_ce"].update(float(loss_prior_ce.item()), bsz)
        m["loss_soft_rec"].update(float(loss_soft.item()), bsz)
        m["loss_c2_proposal"].update(float(loss_c2.item()), bsz)
        m["c2_proposal_mse"].update(c2_mse, bsz)
        for key, value in acc.items():
            m[key].update(value, bsz)
        for key, value in prior_candidate_metrics(prior_logits, idx).items():
            m[key].update(value, bsz)
        update_recon_metrics(m, x_c1=x_c1, x_gt=x_gt, x_pred=x_pred, x_safe=x_safe, x_soft=x_soft, imgs=imgs, bsz=bsz, safe_keep=safe_keep)
    return averaged(m)


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder = build_encoder_decoder(args, cfg.device)
    quantizer = build_quantizer(args, cfg.device)
    predictor = build_predictor(args, cfg.device)
    load_stage2(args, encoder, decoder, quantizer)
    freeze_module(encoder, False)
    freeze_module(decoder, False)
    freeze_module(quantizer, False)
    opt = optim.AdamW(predictor.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = -1.0
    print_v01_header(args, "Stage 3 | frozen C1_rx to C2 VQ-index predictor", len(train_loader.dataset), len(val_loader.dataset))
    for epoch in range(1, int(args.epochs) + 1):
        encoder.eval()
        decoder.eval()
        quantizer.eval()
        predictor.train()
        m = meters(STAGE3_KEYS)
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            with torch.no_grad():
                _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
                c1, c2 = split_c1_c2(z_norm, args)
                c1_rx = real_awgn(c1, float(args.snr_db))
                q_gt, idx = quantizer.encode(c2)
            logits, prior_logits, c2_proposal = forward_stage3_logits(predictor, quantizer, c1_rx, args)
            loss_ce = F.cross_entropy(logits.reshape(-1, int(args.k)), idx.reshape(-1), label_smoothing=float(args.label_smoothing))
            loss_prior_ce = prior_cross_entropy(prior_logits, idx, args)
            q_soft = soft_quantizer_lookup(quantizer, logits, tau=float(args.soft_tau))
            x_soft = decoder(torch.cat([c1_rx, q_soft], dim=1))
            loss_soft = recon_loss(x_soft, imgs)
            if c2_proposal is None:
                loss_c2 = torch.zeros((), device=imgs.device)
                c2_mse = float("nan")
            else:
                loss_c2 = F.mse_loss(c2_proposal.float(), c2.float())
                c2_mse = float(loss_c2.item())
            loss = (
                loss_ce
                + float(args.lambda_prior_ce) * loss_prior_ce
                + float(args.lambda_soft_rec) * loss_soft
                + float(args.lambda_c2_proposal) * loss_c2
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                idx_hat = logits.argmax(dim=-1)
                q_pred = quantizer_lookup(quantizer, idx_hat)
                q_safe, safe_keep = selected_prediction(quantizer, logits, q_pred, args)
                zero = torch.zeros_like(q_pred)
                x_c1 = decoder(torch.cat([c1_rx, zero], dim=1))
                x_gt = decoder(torch.cat([c1_rx, q_gt], dim=1))
                x_pred = decoder(torch.cat([c1_rx, q_pred], dim=1))
                x_safe = decoder(torch.cat([c1_rx, q_safe], dim=1))
                acc = topk_accuracies(logits, idx)
            bsz = imgs.shape[0]
            m["loss"].update(float(loss.item()), bsz)
            m["loss_ce"].update(float(loss_ce.item()), bsz)
            m["loss_prior_ce"].update(float(loss_prior_ce.item()), bsz)
            m["loss_soft_rec"].update(float(loss_soft.item()), bsz)
            m["loss_c2_proposal"].update(float(loss_c2.item()), bsz)
            m["c2_proposal_mse"].update(c2_mse, bsz)
            for key, value in acc.items():
                m[key].update(value, bsz)
            for key, value in prior_candidate_metrics(prior_logits, idx).items():
                m[key].update(value, bsz)
            update_recon_metrics(m, x_c1=x_c1, x_gt=x_gt, x_pred=x_pred, x_safe=x_safe, x_soft=x_soft, imgs=imgs, bsz=bsz, safe_keep=safe_keep)
        metrics = averaged(m)
        print_epoch("stage3-v01", epoch, int(args.epochs), with_log_keys(metrics), time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, encoder, decoder, quantizer, predictor, args)
            score = val_metrics[str(args.stage3_score)]
            guard_name = str(args.stage3_save_guard)
            guard_value = float("inf") if guard_name == "none" else val_metrics[guard_name]
            guard_ok = guard_name == "none" or guard_value > float(args.min_save_guard)
            print(
                f"[stage3-v01 val {epoch:03d}] {format_metrics(with_log_keys(val_metrics))} "
                f"score={args.stage3_score} save_guard={guard_name}>{float(args.min_save_guard):g}"
            )
            if guard_ok and score > best:
                best = score
                save_v01_checkpoint(ckpt_path(args, "stage3", "best"), stage="stage3", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, quantizer=quantizer, predictor=predictor)
        if should_save_latest(args, epoch):
            save_v01_checkpoint(ckpt_path(args, "stage3", "latest"), stage="stage3", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer, predictor=predictor)
    save_v01_checkpoint(ckpt_path(args, "stage3", "latest"), stage="stage3", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer, predictor=predictor)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_cli(p, default_k=4096)
    p.set_defaults(save_dir=default_v01_save_dir(4096), epochs=150)
    p.add_argument("--init-stage2-ckpt", type=str, default="")
    p.add_argument("--quantizer", type=str, choices=["vq", "simvq", "patch_vq"], default="vq")
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--simvq-proj-dim", type=int, default=256)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--vq-chunk-size", type=int, default=128)
    p.add_argument("--predictor", type=str, default="patch_spatial_transformer")
    p.add_argument("--pred-embed-dim", type=int, default=256)
    p.add_argument("--pred-depth", type=int, default=4)
    p.add_argument("--pred-heads", type=int, default=8)
    p.add_argument("--pred-mlp-ratio", type=float, default=4.0)
    p.add_argument("--pred-dropout", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lambda-soft-rec", type=float, default=100.0)
    p.add_argument("--soft-tau", type=float, default=1.0)
    p.add_argument("--use-c2-proposal-prior", action="store_true")
    p.add_argument("--prior-alpha", type=float, default=0.5)
    p.add_argument("--prior-tau", type=float, default=1.0)
    p.add_argument("--prior-topm", type=int, default=0)
    p.add_argument("--prior-chunk-size", type=int, default=128)
    p.add_argument("--lambda-prior-ce", type=float, default=1.0)
    p.add_argument("--lambda-c2-proposal", type=float, default=0.1)
    p.add_argument("--pred-safe-threshold", type=float, default=0.02)
    p.add_argument("--stage3-score", type=str, choices=["pred_gain", "pred_safe_gain", "soft_gain", "psnr_pred_all", "psnr_pred_safe", "psnr_soft"], default="pred_gain")
    p.add_argument("--stage3-save-guard", type=str, choices=["pred_gain", "pred_safe_gain", "soft_gain", "none"], default="pred_safe_gain")
    p.add_argument("--min-save-guard", type=float, default=0.0)
    p.add_argument("--min-pred-gain", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.gate = "none"
    ensure_common_args(args, stage=3)
    setup_stage_log(args, "stage3_v01")
    write_json(Path(resolve_path(args.save_dir)) / f"stage3_v01{quantizer_artifact_part(args, 'stage3')}_args.json", vars(args))
    train(args)


if __name__ == "__main__":
    main()
