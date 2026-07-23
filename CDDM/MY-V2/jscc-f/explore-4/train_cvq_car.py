#!/usr/bin/env python3
"""Conditional Channel-wise AutoRegressive receiver for explore-4 CVQ.

This is a compact receiver-side implementation of CAR from arXiv:2605.26089v2.
The CVQ tokenizer turns an image into C channel indices.  A causal transformer
models next-channel likelihood p(i_k | i_<k, z1, x1); its deployment forward
has exactly one argument, ``ReceiverCondition(z1,x1)``.  Source img/z2/q2 and
true indices are formed only for teacher-forced training losses and audits.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


THIS_DIR = Path(__file__).resolve().parent
EXPLORE2_DIR = THIS_DIR.parent / "explore-2"
for path in (THIS_DIR, EXPLORE2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


spatial = load_module("explore4_spatial_support", THIS_DIR / "train_cvq_spatial.py")
nested = spatial.nested
base = spatial.base
contracts = load_module("explore4_contracts", EXPLORE2_DIR / "contracts.py")
receivers = load_module("explore4_receiver_models", EXPLORE2_DIR / "receiver_models.py")


class PaperCAR(nn.Module):
    """Decoder-only transformer with paper-style next-channel prediction."""

    def __init__(
        self,
        z1_channels: int,
        channels: int,
        vocabulary: int,
        *,
        hidden: int = 256,
        layers: int = 6,
        heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("--hidden must divide --heads")
        self.channels = int(channels)
        self.vocabulary = int(vocabulary)
        self.hidden = int(hidden)
        self.trunk = receivers.ReceiverTrunk(
            int(z1_channels), hidden=hidden, blocks=6, attention_every=2, heads=heads, condition_mode="z1_x1"
        )
        self.channel_queries = nn.Embedding(self.channels, hidden)
        self.cross_attention = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=dropout)
        self.token_embedding = nn.Embedding(self.vocabulary + 1, hidden)  # final row is BOS
        self.position_embedding = nn.Embedding(self.channels, hidden)
        self.input_projector = nn.Sequential(nn.Linear(hidden * 2, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, self.vocabulary)

    def _channel_context(self, condition) -> torch.Tensor:
        condition.validate()
        feature = self.trunk(condition)
        memory = feature.flatten(2).transpose(1, 2)
        queries = self.channel_queries.weight.unsqueeze(0).expand(int(memory.shape[0]), -1, -1)
        context, _ = self.cross_attention(queries, memory, memory, need_weights=False)
        return context + queries

    def _logits_from_context(self, context: torch.Tensor, shifted: torch.Tensor) -> torch.Tensor:
        if shifted.ndim != 2 or int(shifted.shape[1]) > self.channels:
            raise ValueError(f"shifted indices must be [B,<=C], got {tuple(shifted.shape)}")
        if int(shifted.min()) < 0 or int(shifted.max()) > self.vocabulary:
            raise ValueError("shifted index out of vocabulary")
        length = int(shifted.shape[1])
        context = context[:, :length]
        positions = torch.arange(length, device=shifted.device).view(1, -1)
        input_tokens = self.token_embedding(shifted) + self.position_embedding(positions)
        value = self.input_projector(torch.cat([input_tokens, context], dim=-1))
        causal = torch.ones((length, length), dtype=torch.bool, device=shifted.device).triu(1)
        return self.head(self.norm(self.decoder(value, mask=causal)))

    def _logits(self, condition, shifted: torch.Tensor) -> torch.Tensor:
        return self._logits_from_context(self._channel_context(condition), shifted)

    def forward_teacher(self, condition, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 2 or int(indices.shape[1]) != self.channels:
            raise ValueError(f"CAR targets must be [B,{self.channels}], got {tuple(indices.shape)}")
        if int(indices.min()) < 0 or int(indices.max()) >= self.vocabulary:
            raise ValueError("CAR target index out of vocabulary")
        bos = torch.full((int(indices.shape[0]), 1), self.vocabulary, device=indices.device, dtype=torch.long)
        return self._logits(condition, torch.cat([bos, indices[:, :-1].long()], dim=1))

    def forward(self, condition) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy rollout; the only permitted deployment graph."""
        batch = int(condition.z1.shape[0])
        context = self._channel_context(condition)
        sequence = torch.full((batch, 1), self.vocabulary, device=condition.z1.device, dtype=torch.long)
        logits: list[torch.Tensor] = []
        generated: list[torch.Tensor] = []
        for _channel in range(self.channels):
            step_logits = self._logits_from_context(context, sequence)[:, -1]
            token = step_logits.argmax(dim=-1)
            logits.append(step_logits)
            generated.append(token)
            sequence = torch.cat([sequence, token[:, None]], dim=1)
        return torch.stack(logits, dim=1), torch.stack(generated, dim=1)


class ReceiverDecode(nn.Module):
    """Receiver-owned copy after the frozen shared CVQ codebook."""

    def __init__(self, sender: spatial.SenderCVQ) -> None:
        super().__init__()
        self.token_decoder = copy.deepcopy(sender.codec.token_decoder)
        self.bridge = copy.deepcopy(sender.codec.bridge)
        self.d2_frontend = copy.deepcopy(sender.d2_frontend)
        self.d2 = copy.deepcopy(sender.d2)
        self.combiner = copy.deepcopy(sender.combiner)
        self.d2_input_channels = int(sender.d2_input_channels)
        self.source_d2_input_channels = int(sender.source_d2_input_channels)
        self.d2_z1_channels = int(sender.d2_z1_channels)

    def forward(self, q: torch.Tensor, z1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        u2 = self._decode_u2(q, z1)
        u2_zero = self._decode_u2(torch.zeros_like(q), z1)
        return self.combiner(x1, u2, u2_zero)

    def _decode_u2(self, q: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        q_part = self.bridge(self.token_decoder(q))
        z1_part = z1[:, :self.d2_z1_channels]
        if tuple(z1_part.shape[-2:]) != tuple(q_part.shape[-2:]):
            raise RuntimeError(f"z1/D2 spatial mismatch {tuple(z1_part.shape)} vs {tuple(q_part.shape)}")
        d2_input = torch.cat([z1_part, q_part], dim=1)
        if int(d2_input.shape[1]) != self.d2_input_channels:
            raise RuntimeError(f"D2 input must have {self.d2_input_channels} channels, got {d2_input.shape[1]}")
        return self.d2(self.d2_frontend(d2_input))


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(1).clamp_min(1e-12)
    return -10.0 * mse.log10()


def load_sender(path: Path, device: torch.device, cli: argparse.Namespace):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("stage") != "explore4_paper_cvq_spatial":
        raise ValueError(f"not an explore-4 CVQ checkpoint: {path}")
    saved_args = argparse.Namespace(**dict(payload["args"]))
    saved_args.cpu = bool(cli.cpu)
    # Data transforms are inherited from the frozen tokenizer checkpoint, but
    # receiver batch/worker settings belong to this CAR invocation.
    saved_args.batch_size = int(cli.batch_size)
    saved_args.test_batch = int(cli.test_batch)
    saved_args.num_workers = int(cli.num_workers)
    saved_args.val_num_workers = int(cli.val_num_workers)
    source_checkpoint = base.jsccf_io.load_checkpoint(str(nested.resolve_path(saved_args.source_checkpoint)))
    source_args = argparse.Namespace(**source_checkpoint["args"])
    _train_loader, _val_loader, device = nested.build_loaders(saved_args, source_args)
    source = nested.load_source(saved_args, device)
    sender = spatial.SenderCVQ(source, saved_args).to(device)
    sender.load_state_dict(payload["model_state_dict"], strict=True)
    sender.requires_grad_(False).eval()
    return sender, saved_args, payload, _train_loader, _val_loader, device


@torch.no_grad()
def sender_targets(sender: spatial.SenderCVQ, imgs: torch.Tensor, rate: int) -> dict[str, torch.Tensor]:
    layer1 = sender.source.layer1(imgs)
    z = sender.codec.encode(imgs, layer1["x1"])
    _st, q, indices, _stats = sender.codec.quantizer.forward_at_k(z, int(rate), update_usage=False)
    oracle = sender.decode(q, layer1["z1"], layer1["x1"])
    condition = contracts.make_receiver_condition(layer1["z1"], layer1["x1"], detach=True)
    return {"x1": layer1["x1"], "z": z, "q": q, "indices": indices, "oracle": oracle, "condition": condition}


def indices_to_q(sender: spatial.SenderCVQ, indices: torch.Tensor, rate: int) -> torch.Tensor:
    return sender.codec.quantizer.get_codebook_entry(indices.long(), int(rate), detach_codebook=True)


def logits_to_soft_q(sender: spatial.SenderCVQ, logits: torch.Tensor, rate: int, temperature: float) -> torch.Tensor:
    codebook = sender.codec.quantizer.codebook_at_k(int(rate)).detach().reshape(int(rate), -1)
    probabilities = F.softmax(logits / float(temperature), dim=-1)
    flat = torch.einsum("bck,kd->bcd", probabilities, codebook)
    h, w = sender.codec.quantizer.embedding_shape
    return flat.view(int(logits.shape[0]), int(logits.shape[1]), int(h), int(w))


class Means:
    def __init__(self) -> None:
        self.value: dict[str, float] = {}
        self.weight: dict[str, float] = {}

    def add(self, name: str, value: float | torch.Tensor, batch: int) -> None:
        scalar = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
        self.value[name] = self.value.get(name, 0.0) + scalar * int(batch)
        self.weight[name] = self.weight.get(name, 0.0) + int(batch)

    def result(self) -> dict[str, float]:
        return {name: total / self.weight[name] for name, total in self.value.items()}


def run_epoch(loader, sender, car, receiver, optimizer, args, device, *, train: bool) -> dict[str, float]:
    car.train(train)
    receiver.train(train and not bool(args.freeze_receiver_decoder))
    meters = Means()
    maximum = int(args.max_train_batches if train else args.max_val_batches)
    audited = False
    for batch_index, (imgs, _labels) in enumerate(loader, start=1):
        if maximum and batch_index > maximum:
            break
        imgs = imgs.to(device, non_blocking=True)
        target = sender_targets(sender, imgs, int(args.rate))
        if not audited:
            contracts.assert_receiver_only_module(car)
            contracts.assert_training_targets_are_not_inputs(
                car, target["condition"],
                source_targets={"img": imgs, "z2": target["z"], "q2": target["q"], "oracle_indices": target["indices"]},
            )
            audited = True
        with torch.set_grad_enabled(train):
            teacher = car.forward_teacher(target["condition"], target["indices"])
            loss_ce = F.cross_entropy(teacher.reshape(-1, int(args.rate)), target["indices"].reshape(-1), label_smoothing=float(args.label_smoothing))
            # Paper CAR trains next-channel likelihood in parallel under
            # teacher forcing.  Reusing those causal logits for the soft-code
            # reconstruction term preserves that objective and avoids an
            # O(C^3) greedy rollout inside every optimization step.  Strict
            # validation below still performs the real BOS->C hard rollout.
            hard_final = None
            mean_final = None
            if train and float(args.lambda_recon) > 0.0:
                q_decode = logits_to_soft_q(sender, teacher, int(args.rate), float(args.soft_temperature))
                pred_indices = teacher.argmax(dim=-1)
            else:
                rollout_logits, pred_indices = car(target["condition"])
                q_hard = indices_to_q(sender, pred_indices, int(args.rate))
                hard_final = receiver(q_hard, target["condition"].z1, target["x1"])
                # CAR models a categorical posterior at every generated
                # channel.  For a squared-error reconstruction target its
                # codebook expectation is the Bayes estimator; this remains
                # a fully autoregressive receiver path because logits arise
                # only from the BOS->generated prefix rollout.
                q_mean = logits_to_soft_q(
                    sender, rollout_logits, int(args.rate), float(args.rollout_temperature)
                )
                mean_final = receiver(q_mean, target["condition"].z1, target["x1"])
                q_decode = q_mean if args.rollout_q_mode == "mean" else q_hard
            if train:
                final = receiver(q_decode, target["condition"].z1, target["x1"])
            else:
                final = mean_final if args.rollout_q_mode == "mean" else hard_final
            loss_recon = F.mse_loss(final.float(), imgs.float())
            loss = float(args.lambda_ce) * loss_ce + float(args.lambda_recon) * loss_recon
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in list(car.parameters()) + list(receiver.parameters()) if p.requires_grad], float(args.grad_clip_norm))
                optimizer.step()
        batch = int(imgs.shape[0])
        x1_psnr, oracle_psnr, pred_psnr = psnr(target["x1"], imgs), psnr(target["oracle"], imgs), psnr(final, imgs)
        meters.add("loss", loss, batch); meters.add("loss_ce", loss_ce, batch); meters.add("loss_recon", loss_recon, batch)
        meters.add("psnr_x1", x1_psnr.mean(), batch); meters.add("psnr_oracle", oracle_psnr.mean(), batch); meters.add("psnr_x2_hat", pred_psnr.mean(), batch)
        meters.add("delta_x1_hat", (pred_psnr - x1_psnr).mean(), batch); meters.add("delta_x1_oracle", (oracle_psnr - x1_psnr).mean(), batch)
        # ``index_accuracy`` below is teacher-forced during training but is the
        # deployed BOS->C rollout during validation.  Keep an explicit
        # teacher-forced validation reference so an exposure-bias gap cannot
        # be misreported as a train/valid generalisation gap.
        meters.add("teacher_index_accuracy", (teacher.argmax(dim=-1) == target["indices"]).float().mean(), batch)
        meters.add("index_accuracy", (pred_indices == target["indices"]).float().mean(), batch)
        if not train:
            if hard_final is None or mean_final is None:
                raise RuntimeError("validation must evaluate both hard and posterior-mean CAR decodes")
            hard_psnr = psnr(hard_final, imgs)
            mean_psnr = psnr(mean_final, imgs)
            meters.add("psnr_x2_hat_hard", hard_psnr.mean(), batch)
            meters.add("delta_x1_hat_hard", (hard_psnr - x1_psnr).mean(), batch)
            meters.add("psnr_x2_hat_mean", mean_psnr.mean(), batch)
            meters.add("delta_x1_hat_mean", (mean_psnr - x1_psnr).mean(), batch)
        if not train:
            zero = receiver(torch.zeros_like(q_decode), target["condition"].z1, target["x1"])
            shuffle = receiver(sender.codec.quantizer.shuffle_tokens(q_decode), target["condition"].z1, target["x1"])
            meters.add("drop_zero", (pred_psnr - psnr(zero, imgs)).mean(), batch)
            meters.add("drop_shuffle", (pred_psnr - psnr(shuffle, imgs)).mean(), batch)
            if batch > 1:
                perm = torch.roll(torch.arange(batch, device=device), 1)
                wrong = contracts.make_receiver_condition(target["condition"].z1[perm], target["condition"].x1[perm], detach=True)
                _wrong_logits, wrong_indices = car(wrong)
                wrong_final = receiver(indices_to_q(sender, wrong_indices, int(args.rate)), wrong.z1, target["x1"])
                meters.add("condition_shuffle_drop", (pred_psnr - psnr(wrong_final, imgs)).mean(), batch)
    result = meters.result()
    result["receiver_only_audit"] = float(audited)
    if not train:
        result["goal_met"] = float(result["delta_x1_hat"] >= float(args.min_delta) and result.get("drop_zero", 0.0) >= float(args.min_ablation_drop) and result.get("drop_shuffle", 0.0) >= float(args.min_ablation_drop) and result.get("condition_shuffle_drop", 0.0) >= float(args.min_condition_drop))
    return result


def save(path: Path, sender_path: str, car, receiver, optimizer, args, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": "explore4_paper_car_receiver", "paper": "arXiv:2605.26089v2 CAR next-channel prediction", "epoch": epoch, "sender_checkpoint": sender_path, "args": vars(args), "metrics": metrics, "car_state_dict": car.state_dict(), "receiver_state_dict": receiver.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "receiver_contract": {"inputs": ["z1", "x1"], "forbidden_inputs": ["img", "z2", "q2", "oracle_indices"], "output": "hard_CVQ_codebook_q2_hat"}}, path)
    print(f"saved checkpoint: {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sender-checkpoint", required=True)
    parser.add_argument("--rate", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-q-mode", choices=["hard", "mean"], default="hard", help="receiver representation from autoregressive categorical logits")
    parser.add_argument("--rollout-temperature", type=float, default=1.0, help="temperature for the posterior-mean deployment representation")
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--freeze-receiver-decoder", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--save-dir", default="MY-V2/jscc-f/explore-4/checkpoints-car")
    parser.add_argument("--log-json", default="")
    parser.add_argument("--version", default="car-v1")
    parser.add_argument("--min-delta", type=float, default=0.5)
    parser.add_argument("--min-ablation-drop", type=float, default=0.1)
    parser.add_argument("--min-condition-drop", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    provisional = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    sender, sender_args, sender_payload, train_loader, val_loader, device = load_sender(Path(args.sender_checkpoint), provisional, args)
    if int(args.rate) not in [int(v) for v in sender_args.rates_list]:
        raise ValueError(f"--rate {args.rate} was not trained by sender prefixes {sender_args.rates_list}")
    with torch.no_grad():
        first = next(iter(train_loader))[0][:1].to(device)
        z1_channels = int(sender.source.layer1(first)["z1"].shape[1])
    car = PaperCAR(z1_channels, int(sender_args.latent_c), int(args.rate), hidden=int(args.hidden), layers=int(args.layers), heads=int(args.heads), dropout=float(args.dropout)).to(device)
    receiver = ReceiverDecode(sender).to(device)
    if bool(args.freeze_receiver_decoder):
        receiver.requires_grad_(False).eval()
    groups = [{"params": car.parameters(), "lr": float(args.lr)}]
    if not bool(args.freeze_receiver_decoder):
        groups.append({"params": receiver.parameters(), "lr": float(args.decoder_lr)})
    optimizer = optim.AdamW(groups, weight_decay=float(args.weight_decay))
    print("=== explore-4 | paper-2605.26089v2 conditional CAR ===", flush=True)
    print("实验设计", flush=True); print(f"  frozen CVQ={args.sender_checkpoint}; next-channel transformer C={sender_args.latent_c}, K={args.rate}; receiver input=(z1,x1) only", flush=True)
    print("loss设计", flush=True); print(f"  {args.lambda_ce}*teacher-forced CE + {args.lambda_recon}*teacher-logit soft-code reconstruction; valid rolls out BOS->C and reports hard MAP plus posterior-mean q2_hat", flush=True)
    print("模块选择", flush=True); print(f"  decoder-only causal transformer={args.layers}x{args.hidden}; receiver decoder frozen={args.freeze_receiver_decoder}; deployment_q={args.rollout_q_mode}@T={args.rollout_temperature}; crop train=random, valid=center", flush=True)
    root = nested.resolve_path(args.save_dir) / args.version
    best = float("-inf"); history = []
    for epoch in range(1, int(args.epochs) + 1):
        started = time.time()
        train_metrics = run_epoch(train_loader, sender, car, receiver, optimizer, args, device, train=True)
        print(f"[CAR train {epoch:03d}/{args.epochs}] {train_metrics} time={time.time()-started:.1f}s", flush=True)
        item = {"epoch": epoch, "train": train_metrics}
        if epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            val_metrics = run_epoch(val_loader, sender, car, receiver, None, args, device, train=False)
            print(f"[CAR val {epoch:03d}] {val_metrics}", flush=True)
            item["val"] = val_metrics
            if val_metrics["psnr_x2_hat"] > best:
                best = val_metrics["psnr_x2_hat"]
                save(root / "best.pth", args.sender_checkpoint, car, receiver, optimizer, args, epoch, val_metrics)
        history.append(item)
        if args.log_json:
            output = nested.resolve_path(args.log_json); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train(parse_args())
