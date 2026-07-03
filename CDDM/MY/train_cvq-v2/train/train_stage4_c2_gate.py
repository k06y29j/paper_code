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
    build_gate,
    build_predictor,
    build_quantizer,
    ckpt_path,
    cvq_io,
    default_stage3_ckpt,
    default_v01_save_dir,
    ensure_common_args,
    format_metrics,
    freeze_module,
    gate_binary_metrics,
    get_loader,
    greedy_oracle_gate_labels,
    load_v01_checkpoint,
    logits_features,
    meters,
    print_epoch,
    print_v01_header,
    psnr_per_image,
    quantizer_lookup,
    real_awgn,
    resolve_path,
    save_v01_checkpoint,
    seed_everything,
    setup_stage_log,
    should_save_latest,
    should_validate,
    split_c1_c2,
    with_log_keys,
    write_json,
)


def load_stage3(args, encoder, decoder, quantizer, predictor) -> None:
    src = args.init_stage3_ckpt or default_stage3_ckpt(args)
    obj = load_v01_checkpoint(src)
    encoder.load_state_dict(obj["encoder_state_dict"], strict=True)
    decoder.load_state_dict(obj["decoder_state_dict"], strict=True)
    quantizer.load_state_dict(obj["quantizer_state_dict"], strict=True)
    predictor.load_state_dict(obj["predictor_state_dict"], strict=True)
    print(f"stage4_source_stage3={resolve_path(src)}")


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


def predictor_final_logits(predictor, quantizer, c1_rx: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    out = predictor(c1_rx)
    if isinstance(out, tuple):
        cls_logits, c2_proposal = out
    else:
        cls_logits, c2_proposal = out, None
    prior_logits = c2_proposal_prior_logits(quantizer, c2_proposal, args) if bool(args.use_c2_proposal_prior) else None
    return apply_c2_prior(cls_logits, prior_logits, args)


@torch.no_grad()
def evaluate_batch(imgs, encoder, decoder, quantizer, predictor, gate, args, train_gate: bool = False):
    _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
    c1, c2 = split_c1_c2(z_norm, args)
    c1_rx = real_awgn(c1, float(args.snr_db))
    logits = predictor_final_logits(predictor, quantizer, c1_rx, args)
    idx_hat = logits.argmax(dim=-1)
    q_pred = quantizer_lookup(quantizer, idx_hat)
    labels, q_oracle = greedy_oracle_gate_labels(decoder, c1_rx, q_pred, imgs, margin=float(args.gate_margin))
    conf, margin, entropy, q_norm = logits_features(logits, q_pred)
    gate_logits = gate(c1_rx, conf, margin, entropy, q_norm)
    prob = torch.sigmoid(gate_logits)
    learned_mask = (prob > float(args.gate_threshold)).to(dtype=q_pred.dtype).unsqueeze(-1).unsqueeze(-1)
    zero = torch.zeros_like(q_pred)
    x_c1 = decoder(torch.cat([c1_rx, zero], dim=1)).clamp(0.0, 1.0)
    x_pred = decoder(torch.cat([c1_rx, q_pred], dim=1)).clamp(0.0, 1.0)
    x_oracle = decoder(torch.cat([c1_rx, q_oracle], dim=1)).clamp(0.0, 1.0)
    x_learned = decoder(torch.cat([c1_rx, q_pred * learned_mask], dim=1)).clamp(0.0, 1.0)
    return {
        "gate_logits": gate_logits,
        "labels": labels,
        "prob": prob,
        "learned_mask": learned_mask.squeeze(-1).squeeze(-1),
        "x_c1": x_c1,
        "x_pred": x_pred,
        "x_oracle": x_oracle,
        "x_learned": x_learned,
    }


@torch.no_grad()
def validate(loader, encoder, decoder, quantizer, predictor, gate, args) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    quantizer.eval()
    predictor.eval()
    gate.eval()
    names = ["loss", "psnr_c1_only", "psnr_pred_all", "psnr_oracle_gate", "psnr_learned_gate", "oracle_keep_ratio", "learned_keep_ratio", "gate_acc", "gate_precision", "gate_recall"]
    m = meters(names)
    device = next(gate.parameters()).device
    for imgs, _labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = evaluate_batch(imgs, encoder, decoder, quantizer, predictor, gate, args)
        loss = F.binary_cross_entropy_with_logits(out["gate_logits"], out["labels"])
        bsz = imgs.shape[0]
        m["loss"].update(float(loss.item()), bsz)
        m["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(out["x_c1"], imgs)), bsz)
        m["psnr_pred_all"].update(batch_metric_mean(psnr_per_image(out["x_pred"], imgs)), bsz)
        m["psnr_oracle_gate"].update(batch_metric_mean(psnr_per_image(out["x_oracle"], imgs)), bsz)
        m["psnr_learned_gate"].update(batch_metric_mean(psnr_per_image(out["x_learned"], imgs)), bsz)
        m["oracle_keep_ratio"].update(float(out["labels"].mean().item()), bsz)
        m["learned_keep_ratio"].update(float(out["learned_mask"].float().mean().item()), bsz)
        bm = gate_binary_metrics(out["learned_mask"], out["labels"])
        for key, value in bm.items():
            m[key].update(value, bsz)
    return averaged(m)


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    cfg = cvq_io.build_config(args)
    train_loader, val_loader = get_loader(cfg)
    encoder, decoder = build_encoder_decoder(args, cfg.device)
    quantizer = build_quantizer(args, cfg.device)
    predictor = build_predictor(args, cfg.device)
    gate = build_gate(args, cfg.device)
    load_stage3(args, encoder, decoder, quantizer, predictor)
    for module in [encoder, decoder, quantizer, predictor]:
        freeze_module(module, False)
    opt = optim.AdamW(gate.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = -1.0
    print_v01_header(args, "Stage 4 | learned C2 selector gate from predictor confidence", len(train_loader.dataset), len(val_loader.dataset))
    for epoch in range(1, int(args.epochs) + 1):
        encoder.eval()
        decoder.eval()
        quantizer.eval()
        predictor.eval()
        gate.train()
        m = meters(["loss", "psnr_c1_only", "psnr_pred_all", "psnr_oracle_gate", "psnr_learned_gate", "oracle_keep_ratio", "learned_keep_ratio", "gate_acc", "gate_precision", "gate_recall"])
        t0 = time.time()
        for imgs, _labels in train_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            with torch.no_grad():
                _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
                c1, _c2 = split_c1_c2(z_norm, args)
                c1_rx = real_awgn(c1, float(args.snr_db))
                logits = predictor(c1_rx)
                idx_hat = logits.argmax(dim=-1)
                q_pred = quantizer_lookup(quantizer, idx_hat)
                labels, q_oracle = greedy_oracle_gate_labels(decoder, c1_rx, q_pred, imgs, margin=float(args.gate_margin))
                conf, pred_margin, entropy, q_norm = logits_features(logits, q_pred)
            gate_logits = gate(c1_rx, conf, pred_margin, entropy, q_norm)
            loss = F.binary_cross_entropy_with_logits(gate_logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                prob = torch.sigmoid(gate_logits)
                learned = (prob > float(args.gate_threshold)).to(dtype=q_pred.dtype)
                zero = torch.zeros_like(q_pred)
                x_c1 = decoder(torch.cat([c1_rx, zero], dim=1))
                x_pred = decoder(torch.cat([c1_rx, q_pred], dim=1))
                x_oracle = decoder(torch.cat([c1_rx, q_oracle], dim=1))
                x_learned = decoder(torch.cat([c1_rx, q_pred * learned.unsqueeze(-1).unsqueeze(-1)], dim=1))
                bm = gate_binary_metrics(learned, labels)
            bsz = imgs.shape[0]
            m["loss"].update(float(loss.item()), bsz)
            m["psnr_c1_only"].update(batch_metric_mean(psnr_per_image(x_c1, imgs)), bsz)
            m["psnr_pred_all"].update(batch_metric_mean(psnr_per_image(x_pred, imgs)), bsz)
            m["psnr_oracle_gate"].update(batch_metric_mean(psnr_per_image(x_oracle, imgs)), bsz)
            m["psnr_learned_gate"].update(batch_metric_mean(psnr_per_image(x_learned, imgs)), bsz)
            m["oracle_keep_ratio"].update(float(labels.mean().item()), bsz)
            m["learned_keep_ratio"].update(float(learned.float().mean().item()), bsz)
            for key, value in bm.items():
                m[key].update(value, bsz)
        metrics = averaged(m)
        print_epoch("stage4-v01", epoch, int(args.epochs), with_log_keys(metrics), time.time() - t0)
        if should_validate(args, epoch):
            val_metrics = validate(val_loader, encoder, decoder, quantizer, predictor, gate, args)
            score = val_metrics["psnr_learned_gate"]
            print(f"[stage4-v01 val {epoch:03d}] {format_metrics(with_log_keys(val_metrics))} score=psnr_learned_gate")
            if score > best:
                best = score
                save_v01_checkpoint(ckpt_path(args, "stage4", "best"), stage="stage4", epoch=epoch, args=args, metrics=val_metrics, encoder=encoder, decoder=decoder, quantizer=quantizer, predictor=predictor, gate=gate)
        if should_save_latest(args, epoch):
            save_v01_checkpoint(ckpt_path(args, "stage4", "latest"), stage="stage4", epoch=epoch, args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer, predictor=predictor, gate=gate)
    save_v01_checkpoint(ckpt_path(args, "stage4", "latest"), stage="stage4", epoch=int(args.epochs), args=args, metrics=metrics, encoder=encoder, decoder=decoder, quantizer=quantizer, predictor=predictor, gate=gate)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_cli(p, default_k=4096)
    p.set_defaults(save_dir=default_v01_save_dir(4096), epochs=100)
    p.add_argument("--init-stage3-ckpt", type=str, default="")
    p.add_argument("--quantizer", type=str, choices=["vq", "simvq", "patch_vq"], default="vq")
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--simvq-proj-dim", type=int, default=256)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--vq-chunk-size", type=int, default=128)
    p.add_argument("--predictor", type=str, default="query_transformer")
    p.add_argument("--pred-embed-dim", type=int, default=256)
    p.add_argument("--pred-depth", type=int, default=4)
    p.add_argument("--pred-heads", type=int, default=8)
    p.add_argument("--pred-mlp-ratio", type=float, default=4.0)
    p.add_argument("--pred-dropout", type=float, default=0.0)
    p.add_argument("--use-c2-proposal-prior", action="store_true")
    p.add_argument("--prior-alpha", type=float, default=0.5)
    p.add_argument("--prior-tau", type=float, default=1.0)
    p.add_argument("--prior-topm", type=int, default=0)
    p.add_argument("--prior-chunk-size", type=int, default=128)
    p.add_argument("--gate", type=str, default="mlp")
    p.add_argument("--gate-hidden", type=int, default=128)
    p.add_argument("--gate-c1-feat", type=int, default=64)
    p.add_argument("--gate-margin", type=float, default=0.0)
    p.add_argument("--gate-threshold", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_common_args(args, stage=4)
    setup_stage_log(args, "stage4_v01")
    write_json(Path(resolve_path(args.save_dir)) / "stage4_v01_args.json", vars(args))
    train(args)


if __name__ == "__main__":
    main()
