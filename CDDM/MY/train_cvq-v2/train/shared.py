from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
CDDM_ROOT = PARENT_DIR.parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))

from common import (  # noqa: E402
    AverageMeter,
    batch_metric_mean,
    check_args,
    format_metrics,
    freeze_module,
    print_epoch,
    psnr_per_image,
    real_awgn,
    recon_loss,
    resolve_path,
    seed_everything,
    setup_log_file,
    should_save_latest,
    should_validate,
    split_c1_c2,
    write_json,
)
from model import FullChannelQuantizer, nearest_codebook  # noqa: E402

from Autoencoder.data.datasets import get_loader  # noqa: E402
from Autoencoder.net.network import JSCC_decoder, JSCC_encoder  # noqa: E402


def load_local_io():
    spec = importlib.util.spec_from_file_location("cvq_v2_io", PARENT_DIR / "io.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cvq_io = load_local_io()

LOG_KEYS = [
    "loss",
    "psnr_c1_only",
    "psnr_full_c2",
    "psnr_q_c2_gt",
    "psnr_pred_all",
    "psnr_pred_safe",
    "psnr_soft",
    "pred_gain",
    "pred_safe_gain",
    "soft_gain",
    "loss_ce",
    "loss_prior_ce",
    "loss_soft_rec",
    "loss_c2_proposal",
    "safe_keep_ratio",
    "c2_proposal_mse",
    "prior_top1_acc",
    "prior_top5_acc",
    "prior_top10_acc",
    "prior_recall64",
    "prior_recall128",
    "prior_recall256",
    "psnr_oracle_gate",
    "psnr_learned_gate",
    "quant_mse",
    "perplexity",
    "used_codes",
    "top1_acc",
    "top5_acc",
    "top10_acc",
    "oracle_keep_ratio",
    "learned_keep_ratio",
]


def default_v01_save_dir(k: int = 4096) -> str:
    return str(CDDM_ROOT / "MY" / f"checkpoints-cvq-v2-v01-c36-snr9-k{int(k)}")


def default_jscc_encoder_c36_snr9() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints-jscc" / "encoder_snr9_channel_awgn_C36.pt")


def default_jscc_decoder_c36_snr9() -> str:
    return str(CDDM_ROOT / "MY" / "checkpoints-jscc" / "decoder_snr9_channel_awgn_C36.pt")


def default_stage1_ckpt(args: argparse.Namespace) -> str:
    save_dir = Path(resolve_path(args.save_dir))
    exact = save_dir / f"cvq_v2_v01_c36_snr{args.snr_db:g}_k{int(args.k)}_stage1_best.pth"
    if exact.exists():
        return str(exact)
    matches = sorted(save_dir.glob(f"cvq_v2_v01_c36_snr{args.snr_db:g}_k*_stage1_best.pth"))
    if matches:
        return str(matches[0])
    return str(exact)


def safe_artifact_name(value: str) -> str:
    text = str(value).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def quantizer_artifact_part(args: argparse.Namespace, stage: str) -> str:
    if str(stage) not in {"stage2", "stage3"}:
        return ""
    quantizer = safe_artifact_name(str(getattr(args, "quantizer", "")))
    return f"_{quantizer}" if quantizer else ""


def default_stage2_ckpt(args: argparse.Namespace) -> str:
    save_dir = Path(resolve_path(args.save_dir))
    new_path = save_dir / f"cvq_v2_v01_c36_snr{args.snr_db:g}_k{int(args.k)}{quantizer_artifact_part(args, 'stage2')}_stage2_best.pth"
    old_path = save_dir / f"cvq_v2_v01_c36_snr{args.snr_db:g}_k{int(args.k)}_stage2_best.pth"
    return str(new_path if new_path.exists() or not old_path.exists() else old_path)


def default_stage3_ckpt(args: argparse.Namespace) -> str:
    save_dir = Path(resolve_path(args.save_dir))
    new_path = save_dir / f"cvq_v2_v01_c36_snr{args.snr_db:g}_k{int(args.k)}{quantizer_artifact_part(args, 'stage3')}_stage3_best.pth"
    old_path = save_dir / f"cvq_v2_v01_c36_snr{args.snr_db:g}_k{int(args.k)}_stage3_best.pth"
    return str(new_path if new_path.exists() or not old_path.exists() else old_path)


def ckpt_path(args: argparse.Namespace, stage: str, suffix: str) -> str:
    k_part = f"_k{int(args.k)}" if hasattr(args, "k") else ""
    q_part = quantizer_artifact_part(args, str(stage))
    return str(Path(resolve_path(args.save_dir)) / f"cvq_v2_v01_c36_snr{args.snr_db:g}{k_part}{q_part}_{stage}_{suffix}.pth")


def ensure_common_args(args: argparse.Namespace, stage: int) -> None:
    args.stage = stage
    if not hasattr(args, "k"):
        args.k = 0
    check_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)


def setup_stage_log(args: argparse.Namespace, stage_name: str) -> None:
    if not args.log_file:
        k_part = f"_k{int(args.k)}" if int(getattr(args, "k", 0)) > 0 else ""
        stage = str(stage_name).split("_", 1)[0]
        q_part = quantizer_artifact_part(args, stage)
        args.log_file = str(Path(resolve_path(args.save_dir)) / f"{stage_name}_cvq_v2_c36_snr{args.snr_db:g}{k_part}{q_part}.log")
    setup_log_file(args.log_file)


def print_v01_header(args: argparse.Namespace, title: str, train_n: int, val_n: int) -> None:
    c2_ch = int(args.latent_ch) - int(args.c1_ch)
    print(f"=== {title} ===")
    print(f"device={'cpu' if args.cpu else 'cuda:0'} visible_cuda={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"save_dir={resolve_path(args.save_dir)}")
    print("实验设计")
    print(f"  latent_ch={args.latent_ch} C1={args.c1_ch} C2={c2_ch} latent_hw={args.latent_h}x{args.latent_w} snr_db={args.snr_db:g}")
    print("  real inference path: C1_rx -> predictor -> C2 index -> codebook lookup -> gate -> decoder")
    print("loss设计")
    if hasattr(args, "lambda_c1"):
        fields = []
        for name in ["lambda_c1", "lambda_drop", "lambda_full", "lambda_q", "lambda_q_drop", "lambda_vq"]:
            if hasattr(args, name):
                fields.append(f"{name}={float(getattr(args, name)):g}")
        print("  " + " ".join(fields))
    if hasattr(args, "label_smoothing"):
        print(f"  CE label_smoothing={float(args.label_smoothing):g}")
    if hasattr(args, "lambda_soft_rec"):
        print(f"  soft_rec lambda={float(args.lambda_soft_rec):g} tau={float(getattr(args, 'soft_tau', 1.0)):g}")
    if getattr(args, "use_c2_proposal_prior", False):
        print(
            "  "
            f"c2_proposal_prior alpha={float(getattr(args, 'prior_alpha', 1.0)):g} "
            f"tau={float(getattr(args, 'prior_tau', 1.0)):g} "
            f"topm={int(getattr(args, 'prior_topm', 0))} "
            f"lambda_prior_ce={float(getattr(args, 'lambda_prior_ce', 0.0)):g} "
            f"lambda_c2={float(getattr(args, 'lambda_c2_proposal', 0.0)):g}"
        )
    if hasattr(args, "pred_safe_threshold"):
        print(
            "  "
            f"pred_safe threshold={float(args.pred_safe_threshold):g} "
            f"stage3_score={getattr(args, 'stage3_score', 'n/a')} "
            f"save_guard={getattr(args, 'stage3_save_guard', 'n/a')}"
        )
    if hasattr(args, "gate_margin"):
        print(f"  gate_bce oracle_margin_mse={float(args.gate_margin):g} threshold={float(args.gate_threshold):g}")
    print("模块选择")
    print(f"  K={int(getattr(args, 'k', 0))} quantizer={getattr(args, 'quantizer', 'none')} predictor={getattr(args, 'predictor', 'none')} gate={getattr(args, 'gate', 'none')}")
    if getattr(args, "quantizer", "") == "patch_vq":
        p = int(getattr(args, "patch_size", 4))
        print(f"  patch_vq patch_size={p} index_grid={int(args.latent_h) // p}x{int(args.latent_w) // p}")
    if getattr(args, "quantizer", "") == "simvq":
        print(f"  simvq proj_dim={int(getattr(args, 'simvq_proj_dim', 256))} codebook=frozen proj=trainable")
    if getattr(args, "quantizer", "") in {"cross_channel_block_vq", "cc_block_vq", "block_vq"}:
        p = int(getattr(args, "block_size", 2))
        c2 = int(getattr(args, "latent_ch", 36)) - int(getattr(args, "c1_ch", 16))
        print(f"  cross_channel_block_vq token_shape={c2}x{p}x{p} index_grid={int(args.latent_h) // p}x{int(args.latent_w) // p}")
    if hasattr(args, "init_codebook_method"):
        print(
            "  "
            f"codebook_init={args.init_codebook_method} "
            f"samples={int(getattr(args, 'init_codebook_samples', 0))} "
            f"kmeans_iters={int(getattr(args, 'kmeans_iters', 0))}"
        )
    for attr in ["init_ckpt", "init_stage0_ckpt", "init_stage1_ckpt", "init_stage2_ckpt", "init_stage3_ckpt", "init_jscc_encoder", "init_jscc_decoder"]:
        if getattr(args, attr, ""):
            print(f"{attr}={resolve_path(getattr(args, attr))}")
    print(f"epochs={args.epochs} train={train_n} valid={val_n} batch={args.batch_size} test_batch={args.test_batch}")


def build_encoder_decoder(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module]:
    cfg = cvq_io.build_config(args)
    encoder = JSCC_encoder(cfg, int(args.latent_ch)).to(device)
    decoder = JSCC_decoder(cfg, int(args.latent_ch)).to(device)
    return encoder, decoder


def load_encoder_decoder_initial(args: argparse.Namespace, encoder: nn.Module, decoder: nn.Module) -> None:
    if getattr(args, "init_ckpt", ""):
        cvq_io.load_experiment_checkpoint(args.init_ckpt, encoder=encoder, decoder=decoder, strict=True)
        return
    cvq_io.load_module_checkpoint(encoder, args.init_jscc_encoder, "init JSCC encoder", strict=True)
    cvq_io.load_module_checkpoint(decoder, args.init_jscc_decoder, "init JSCC decoder", strict=True)


def save_v01_checkpoint(
    path: str,
    *,
    stage: str,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    encoder: nn.Module | None = None,
    decoder: nn.Module | None = None,
    quantizer: nn.Module | None = None,
    predictor: nn.Module | None = None,
    gate: nn.Module | None = None,
) -> None:
    out = Path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "route": "cvq_v2_v01_c1_16_c2_20_predict_index_gate",
        "stage": stage,
        "epoch": int(epoch),
        "metrics": metrics,
        "args": vars(args),
        "snr_db": float(args.snr_db),
        "latent_ch": int(args.latent_ch),
        "c1_ch": int(args.c1_ch),
        "k": int(getattr(args, "k", 0)),
    }
    if encoder is not None:
        payload["encoder_state_dict"] = encoder.state_dict()
    if decoder is not None:
        payload["decoder_state_dict"] = decoder.state_dict()
    if quantizer is not None:
        payload["quantizer_state_dict"] = quantizer.state_dict()
        payload["quantizer_class"] = quantizer.__class__.__name__
    if predictor is not None:
        payload["predictor_state_dict"] = predictor.state_dict()
    if gate is not None:
        payload["gate_state_dict"] = gate.state_dict()
    torch.save(payload, out)
    print(f"saved checkpoint: {out}")


def load_v01_checkpoint(path: str) -> dict:
    abs_path = resolve_path(path)
    obj = torch.load(abs_path, map_location="cpu", weights_only=False)
    print(f"loaded checkpoint: {abs_path}")
    return obj


def sample_uniform_channel_keep_mask(
    batch_size: int,
    c2_ch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    p_keep = torch.rand(int(batch_size), 1, 1, 1, device=device)
    keep = torch.rand(int(batch_size), int(c2_ch), 1, 1, device=device) < p_keep
    return keep.to(dtype=dtype)


def fixed_prefix_keep_mask(
    batch_size: int,
    c2_ch: int,
    keep: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    idx = torch.arange(int(c2_ch), device=device).view(1, int(c2_ch), 1, 1)
    return (idx < int(keep)).expand(int(batch_size), -1, -1, -1).to(dtype=dtype)


def meters(names: list[str]) -> dict[str, AverageMeter]:
    return {name: AverageMeter() for name in names}


def averaged(m: dict[str, AverageMeter]) -> dict[str, float]:
    return {k: v.avg for k, v in m.items()}


def with_log_keys(metrics: dict[str, float]) -> dict[str, float]:
    out = {k: float("nan") for k in LOG_KEYS}
    out.update(metrics)
    return out


class SimVQQuantizer(nn.Module):
    def __init__(
        self,
        num_codes: int = 4096,
        h: int = 16,
        w: int = 16,
        beta: float = 0.25,
        chunk_size: int = 128,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        self.proj_dim = int(proj_dim)
        dim = self.h * self.w
        self.codebook = nn.Parameter(torch.randn(self.num_codes, self.h, self.w) * 0.02)
        self.codebook.requires_grad_(False)
        self.proj = nn.Parameter(torch.empty(dim, self.proj_dim))
        nn.init.orthogonal_(self.proj)
        self.ortho_weight = 1e-3

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        if tuple(x.shape[2:]) != (self.h, self.w):
            raise ValueError(f"latent size {tuple(x.shape[2:])} does not match codebook {(self.h, self.w)}")

    def _nearest(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cb = self.codebook.float().flatten(1)
        proj = self.proj.float()
        cbp = cb @ proj
        cb_norm = cbp.square().sum(dim=1).view(1, -1)
        x2 = x.float().flatten(1)
        indices = []
        quants = []
        chunk = max(1, int(self.chunk_size))
        for start in range(0, x2.shape[0], chunk):
            q = x2[start : start + chunk] @ proj
            dist = q.square().sum(dim=1, keepdim=True) + cb_norm - 2.0 * q @ cbp.t()
            idx = dist.argmin(dim=1)
            indices.append(idx)
            quants.append(self.codebook[idx])
        return torch.cat(quants, dim=0).to(dtype=x.dtype), torch.cat(indices, dim=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check(x)
        bsz, channels, h, w = x.shape
        flat = x.reshape(bsz * channels, h, w)
        quant, idx = self._nearest(flat)
        codebook_loss = F.mse_loss(quant.float(), flat.detach().float())
        commit_loss = F.mse_loss(quant.detach().float(), flat.float())
        flat2 = flat.float().flatten(1)
        quant2 = quant.detach().float().flatten(1)
        proj = self.proj.float()
        proj_commit_loss = F.mse_loss(flat2 @ proj, quant2 @ proj.detach())
        eye = torch.eye(self.proj_dim, device=proj.device, dtype=proj.dtype)
        ortho_loss = F.mse_loss(proj.t() @ proj, eye)
        vq_loss = codebook_loss + self.beta * commit_loss + proj_commit_loss + self.ortho_weight * ortho_loss
        quant_st = flat + (quant - flat).detach()
        return quant_st.reshape(bsz, channels, h, w), idx.reshape(bsz, channels), vq_loss, quant.reshape(bsz, channels, h, w)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check(x)
        bsz, channels, h, w = x.shape
        quant, idx = self._nearest(x.reshape(bsz * channels, h, w))
        return quant.reshape(bsz, channels, h, w), idx.reshape(bsz, channels)

    @torch.no_grad()
    def init_from_samples(self, samples: torch.Tensor) -> None:
        samples = samples.detach().float().cpu()
        if samples.ndim != 3:
            raise ValueError(f"expected samples [N,H,W], got {tuple(samples.shape)}")
        n = int(samples.shape[0])
        pick = torch.randperm(n)[: self.num_codes] if n >= self.num_codes else torch.randint(0, n, (self.num_codes,))
        self.codebook.copy_(samples[pick].to(device=self.codebook.device, dtype=self.codebook.dtype))


class PatchVQQuantizer(nn.Module):
    """Shared K-entry codebook over non-overlapping spatial patches of each C2 map."""

    def __init__(
        self,
        num_codes: int = 4096,
        h: int = 16,
        w: int = 16,
        patch_size: int = 4,
        beta: float = 0.25,
        chunk_size: int = 128,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.h = int(h)
        self.w = int(w)
        self.patch_size = int(patch_size)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        if self.h % self.patch_size != 0 or self.w % self.patch_size != 0:
            raise ValueError(f"patch_size={self.patch_size} must divide latent size {(self.h, self.w)}")
        self.grid_h = self.h // self.patch_size
        self.grid_w = self.w // self.patch_size
        self.codebook = nn.Parameter(torch.randn(self.num_codes, self.patch_size, self.patch_size) * 0.02)

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        if tuple(x.shape[2:]) != (self.h, self.w):
            raise ValueError(f"latent size {tuple(x.shape[2:])} does not match {(self.h, self.w)}")

    def _extract_flat_patches(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        self._check(x)
        bsz, channels, h, w = x.shape
        p = self.patch_size
        patches = F.unfold(x.reshape(bsz * channels, 1, h, w), kernel_size=p, stride=p)
        patches = patches.transpose(1, 2).reshape(bsz * channels * self.grid_h * self.grid_w, p, p)
        return patches, bsz, channels

    def _assemble_flat_patches(self, patches: torch.Tensor, bsz: int, channels: int) -> torch.Tensor:
        p = self.patch_size
        patches = patches.reshape(bsz * channels, self.grid_h * self.grid_w, p * p).transpose(1, 2)
        return F.fold(patches, output_size=(self.h, self.w), kernel_size=p, stride=p).reshape(bsz, channels, self.h, self.w)

    @torch.no_grad()
    def extract_codebook_samples(self, x: torch.Tensor) -> torch.Tensor:
        patches, _bsz, _channels = self._extract_flat_patches(x)
        return patches.detach()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat, bsz, channels = self._extract_flat_patches(x)
        quant, idx = nearest_codebook(flat, self.codebook, self.chunk_size)
        codebook_loss = F.mse_loss(quant.float(), flat.detach().float())
        commit_loss = F.mse_loss(quant.detach().float(), flat.float())
        vq_loss = codebook_loss + self.beta * commit_loss
        quant_st = flat + (quant - flat).detach()
        q_st = self._assemble_flat_patches(quant_st, bsz, channels)
        q_raw = self._assemble_flat_patches(quant, bsz, channels)
        idx = idx.reshape(bsz, channels, self.grid_h, self.grid_w)
        return q_st, idx, vq_loss, q_raw

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat, bsz, channels = self._extract_flat_patches(x)
        quant, idx = nearest_codebook(flat, self.codebook, self.chunk_size)
        return self._assemble_flat_patches(quant, bsz, channels), idx.reshape(bsz, channels, self.grid_h, self.grid_w)

    @torch.no_grad()
    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 4:
            raise ValueError(f"patch VQ indices must be [B,C,Gh,Gw], got {tuple(idx.shape)}")
        if tuple(idx.shape[2:]) != (self.grid_h, self.grid_w):
            raise ValueError(f"index grid {tuple(idx.shape[2:])} does not match {(self.grid_h, self.grid_w)}")
        bsz, channels, _gh, _gw = idx.shape
        patches = self.codebook[idx].to(dtype=self.codebook.dtype)
        patches = patches.reshape(bsz * channels, self.grid_h * self.grid_w, self.patch_size * self.patch_size).transpose(1, 2)
        return F.fold(patches, output_size=(self.h, self.w), kernel_size=self.patch_size, stride=self.patch_size).reshape(bsz, channels, self.h, self.w)

    @torch.no_grad()
    def init_from_samples(self, samples: torch.Tensor) -> None:
        samples = samples.detach().float().cpu()
        if samples.ndim != 3:
            raise ValueError(f"expected samples [N,P,P], got {tuple(samples.shape)}")
        if tuple(samples.shape[1:]) != (self.patch_size, self.patch_size):
            raise ValueError(f"sample size {tuple(samples.shape[1:])} does not match {(self.patch_size, self.patch_size)}")
        n = int(samples.shape[0])
        if n < 1:
            raise ValueError("cannot initialize codebook from empty samples")
        pick = torch.randperm(n)[: self.num_codes] if n >= self.num_codes else torch.randint(0, n, (self.num_codes,))
        self.codebook.copy_(samples[pick].to(device=self.codebook.device, dtype=self.codebook.dtype))


class CrossChannelBlockVQQuantizer(nn.Module):
    """Shared K-entry codebook over 2D spatial blocks containing all C2 channels."""

    def __init__(
        self,
        num_codes: int = 4096,
        c2_ch: int = 20,
        h: int = 16,
        w: int = 16,
        block_size: int = 2,
        beta: float = 0.25,
        chunk_size: int = 128,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.c2_ch = int(c2_ch)
        self.h = int(h)
        self.w = int(w)
        self.block_size = int(block_size)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)
        if self.h % self.block_size != 0 or self.w % self.block_size != 0:
            raise ValueError(f"block_size={self.block_size} must divide latent size {(self.h, self.w)}")
        self.grid_h = self.h // self.block_size
        self.grid_w = self.w // self.block_size
        self.codebook = nn.Parameter(torch.randn(self.num_codes, self.c2_ch, self.block_size, self.block_size) * 0.02)

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.c2_ch:
            raise ValueError(f"expected C2 channels={self.c2_ch}, got {int(x.shape[1])}")
        if tuple(x.shape[2:]) != (self.h, self.w):
            raise ValueError(f"latent size {tuple(x.shape[2:])} does not match {(self.h, self.w)}")

    def _extract_flat_blocks(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        self._check(x)
        bsz = int(x.shape[0])
        p = self.block_size
        blocks = x.unfold(2, p, p).unfold(3, p, p)
        blocks = blocks.permute(0, 2, 3, 1, 4, 5).contiguous()
        return blocks.reshape(bsz * self.grid_h * self.grid_w, self.c2_ch, p, p), bsz

    def _assemble_flat_blocks(self, blocks: torch.Tensor, bsz: int) -> torch.Tensor:
        p = self.block_size
        blocks = blocks.reshape(int(bsz), self.grid_h, self.grid_w, self.c2_ch, p, p)
        return blocks.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(int(bsz), self.c2_ch, self.h, self.w)

    @torch.no_grad()
    def extract_codebook_samples(self, x: torch.Tensor) -> torch.Tensor:
        blocks, _bsz = self._extract_flat_blocks(x)
        return blocks.detach()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat, bsz = self._extract_flat_blocks(x)
        quant, idx = nearest_codebook(flat, self.codebook, self.chunk_size)
        codebook_loss = F.mse_loss(quant.float(), flat.detach().float())
        commit_loss = F.mse_loss(quant.detach().float(), flat.float())
        vq_loss = codebook_loss + self.beta * commit_loss
        quant_st = flat + (quant - flat).detach()
        q_st = self._assemble_flat_blocks(quant_st, bsz)
        q_raw = self._assemble_flat_blocks(quant, bsz)
        return q_st, idx.reshape(bsz, self.grid_h, self.grid_w), vq_loss, q_raw

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat, bsz = self._extract_flat_blocks(x)
        quant, idx = nearest_codebook(flat, self.codebook, self.chunk_size)
        return self._assemble_flat_blocks(quant, bsz), idx.reshape(bsz, self.grid_h, self.grid_w)

    @torch.no_grad()
    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 3:
            raise ValueError(f"cross-channel block VQ indices must be [B,Gh,Gw], got {tuple(idx.shape)}")
        if tuple(idx.shape[1:]) != (self.grid_h, self.grid_w):
            raise ValueError(f"index grid {tuple(idx.shape[1:])} does not match {(self.grid_h, self.grid_w)}")
        bsz = int(idx.shape[0])
        blocks = self.codebook[idx].to(dtype=self.codebook.dtype)
        return self._assemble_flat_blocks(blocks.reshape(bsz * self.grid_h * self.grid_w, self.c2_ch, self.block_size, self.block_size), bsz)

    @torch.no_grad()
    def init_from_samples(self, samples: torch.Tensor) -> None:
        samples = samples.detach().float().cpu()
        if samples.ndim != 4:
            raise ValueError(f"expected samples [N,C2,P,P], got {tuple(samples.shape)}")
        expected = (self.c2_ch, self.block_size, self.block_size)
        if tuple(samples.shape[1:]) != expected:
            raise ValueError(f"sample size {tuple(samples.shape[1:])} does not match {expected}")
        n = int(samples.shape[0])
        if n < 1:
            raise ValueError("cannot initialize codebook from empty samples")
        pick = torch.randperm(n)[: self.num_codes] if n >= self.num_codes else torch.randint(0, n, (self.num_codes,))
        self.codebook.copy_(samples[pick].to(device=self.codebook.device, dtype=self.codebook.dtype))


def build_quantizer(args: argparse.Namespace, device: torch.device) -> nn.Module:
    kind = str(getattr(args, "quantizer", "vq")).lower()
    if kind == "simvq":
        return SimVQQuantizer(
            num_codes=int(args.k),
            h=int(args.latent_h),
            w=int(args.latent_w),
            beta=float(args.vq_beta),
            chunk_size=int(args.vq_chunk_size),
            proj_dim=int(args.simvq_proj_dim),
        ).to(device)
    if kind == "patch_vq":
        return PatchVQQuantizer(
            num_codes=int(args.k),
            h=int(args.latent_h),
            w=int(args.latent_w),
            patch_size=int(getattr(args, "patch_size", 4)),
            beta=float(args.vq_beta),
            chunk_size=int(args.vq_chunk_size),
        ).to(device)
    if kind in {"cross_channel_block_vq", "cc_block_vq", "block_vq"}:
        return CrossChannelBlockVQQuantizer(
            num_codes=int(args.k),
            c2_ch=int(args.latent_ch) - int(args.c1_ch),
            h=int(args.latent_h),
            w=int(args.latent_w),
            block_size=int(getattr(args, "block_size", 2)),
            beta=float(args.vq_beta),
            chunk_size=int(args.vq_chunk_size),
        ).to(device)
    if kind != "vq":
        raise ValueError(f"unknown quantizer: {kind}")
    return FullChannelQuantizer(
        num_codes=int(args.k),
        h=int(args.latent_h),
        w=int(args.latent_w),
        beta=float(args.vq_beta),
        chunk_size=int(args.vq_chunk_size),
    ).to(device)


@torch.no_grad()
def quantizer_lookup(quantizer: nn.Module, idx: torch.Tensor) -> torch.Tensor:
    if hasattr(quantizer, "decode_indices"):
        return quantizer.decode_indices(idx)
    return quantizer.codebook[idx].to(dtype=quantizer.codebook.dtype)


def soft_quantizer_lookup(quantizer: nn.Module, logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    probs = torch.softmax(logits.float() / max(float(tau), 1e-6), dim=-1)
    codebook = quantizer.codebook.float()
    if hasattr(quantizer, "decode_indices"):
        if probs.ndim != 5:
            raise ValueError(f"patch VQ logits must be [B,C,Gh,Gw,K], got {tuple(logits.shape)}")
        patches = torch.einsum("bcghk,kpq->bcghpq", probs, codebook)
        bsz, channels, gh, gw, p, _p = patches.shape
        patches = patches.reshape(bsz * channels, gh * gw, p * p).transpose(1, 2)
        return F.fold(patches, output_size=(quantizer.h, quantizer.w), kernel_size=quantizer.patch_size, stride=quantizer.patch_size).reshape(
            bsz, channels, quantizer.h, quantizer.w
        )
    return torch.einsum("bck,khw->bchw", probs, codebook)


@torch.no_grad()
def confidence_keep_mask(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    conf = torch.softmax(logits.float(), dim=-1).amax(dim=-1)
    return (conf >= float(threshold)).to(dtype=logits.dtype)


@torch.no_grad()
def vq_usage_stats(idx: torch.Tensor, quant: torch.Tensor, target: torch.Tensor, k: int) -> dict[str, float]:
    hist = torch.bincount(idx.reshape(-1).detach().cpu(), minlength=int(k)).float()
    prob = hist / hist.sum().clamp_min(1.0)
    nz = prob[prob > 0]
    used = int((hist > 0).sum().item())
    perplexity = float(torch.exp(-(nz * nz.log()).sum()).item()) if nz.numel() else 0.0
    sorted_hist = torch.sort(hist, descending=True).values
    total = float(hist.sum().item())
    top1 = float(sorted_hist[:1].sum().item() / max(total, 1.0))
    top10 = float(sorted_hist[:10].sum().item() / max(total, 1.0))
    return {
        "quant_mse": float(F.mse_loss(quant.float(), target.float()).item()),
        "perplexity": perplexity,
        "used_codes": float(used),
        "usage": used / float(k),
        "usage_top1_ratio": top1,
        "usage_top10_ratio": top10,
    }


@torch.no_grad()
def init_c2_codebook_from_samples(train_loader, encoder: nn.Module, quantizer: nn.Module, args: argparse.Namespace, device: torch.device) -> None:
    target = int(args.init_codebook_samples)
    if target <= 0:
        return
    samples = []
    seen = 0
    encoder.eval()
    print(f"collecting C2 codebook init samples: target={target}")
    while seen < target:
        for imgs, _labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            _z, z_norm, _power = cvq_io.encode_normalized(imgs, encoder, args)
            _c1, c2 = split_c1_c2(z_norm, args)
            if hasattr(quantizer, "extract_codebook_samples"):
                maps = quantizer.extract_codebook_samples(c2.detach()).float().cpu()
            else:
                maps = c2.detach().float().cpu().reshape(-1, int(args.latent_h), int(args.latent_w))
            samples.append(maps)
            seen += int(maps.shape[0])
            if seen >= target:
                break
    sample_tensor = torch.cat(samples, dim=0)[:target]
    method = str(getattr(args, "init_codebook_method", "random_samples")).lower()
    if method == "random_samples":
        quantizer.init_from_samples(sample_tensor)
        print(f"initialized C2 codebook from {target} clean C2 maps by random_samples")
        return
    if method == "kmeans":
        init_codebook_from_kmeans(
            quantizer,
            sample_tensor,
            iters=int(getattr(args, "kmeans_iters", 20)),
            chunk_size=int(getattr(args, "kmeans_chunk_size", 4096)),
            device=device,
        )
        return
    raise ValueError(f"unknown init_codebook_method: {method}")


@torch.no_grad()
def init_codebook_from_kmeans(
    quantizer: nn.Module,
    samples: torch.Tensor,
    *,
    iters: int,
    chunk_size: int,
    device: torch.device,
) -> None:
    if not hasattr(quantizer, "codebook"):
        raise TypeError("kmeans init requires quantizer.codebook")
    samples = samples.detach().float()
    codebook = quantizer.codebook
    if samples.ndim != codebook.ndim:
        raise ValueError(f"expected samples with ndim={codebook.ndim}, got {tuple(samples.shape)}")
    k = int(codebook.shape[0])
    expected = tuple(int(v) for v in codebook.shape[1:])
    if tuple(samples.shape[1:]) != expected:
        raise ValueError(f"sample size {tuple(samples.shape[1:])} does not match codebook {expected}")
    if samples.shape[0] < int(k):
        raise ValueError(f"kmeans init needs at least K samples: samples={samples.shape[0]} K={int(k)}")

    x = samples.reshape(samples.shape[0], -1).to(device=device, dtype=torch.float32)
    pick = torch.randperm(x.shape[0], device=device)[: int(k)]
    centers = x[pick].clone()
    chunk = max(1, int(chunk_size))
    print(f"running C2 codebook kmeans init: samples={x.shape[0]} K={int(k)} dim={x.shape[1]} iters={int(iters)} chunk={chunk}")

    for step in range(1, max(1, int(iters)) + 1):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(int(k), device=device, dtype=torch.float32)
        total_dist = torch.zeros((), device=device, dtype=torch.float32)
        for start in range(0, x.shape[0], chunk):
            xb = x[start : start + chunk]
            dist = xb.square().sum(dim=1, keepdim=True) + centers.square().sum(dim=1).view(1, -1) - 2.0 * xb @ centers.t()
            idx = dist.argmin(dim=1)
            sums.index_add_(0, idx, xb)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            total_dist += dist.gather(1, idx.view(-1, 1)).sum()
        empty = counts == 0
        if bool(empty.any()):
            repl = torch.randint(0, x.shape[0], (int(empty.sum().item()),), device=device)
            sums[empty] = x[repl]
            counts[empty] = 1.0
        centers = sums / counts.clamp_min(1.0).unsqueeze(1)
        print(f"kmeans init iter {step:02d}/{int(iters)} mse={float(total_dist.item() / max(1, x.shape[0] * x.shape[1])):.6g} empty={int(empty.sum().item())}")

    codebook.copy_(centers.reshape_as(codebook).to(device=codebook.device, dtype=codebook.dtype))
    print(f"initialized C2 codebook from {x.shape[0]} clean C2 maps by kmeans")


class C2IndexPredictor(nn.Module):
    def __init__(
        self,
        c1_ch: int = 16,
        c2_ch: int = 20,
        num_codes: int = 4096,
        latent_h: int = 16,
        latent_w: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.c2_ch = int(c2_ch)
        self.num_codes = int(num_codes)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.stem = nn.Sequential(
            nn.Conv2d(int(c1_ch), int(embed_dim), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(embed_dim), int(embed_dim), 3, padding=1),
        )
        self.pos = nn.Parameter(torch.zeros(1, int(latent_h) * int(latent_w), int(embed_dim)))
        self.queries = nn.Parameter(torch.randn(1, self.c2_ch, int(embed_dim)) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(heads),
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.cross = nn.MultiheadAttention(int(embed_dim), int(heads), dropout=float(dropout), batch_first=True)
        self.norm = nn.LayerNorm(int(embed_dim))
        self.head = nn.Linear(int(embed_dim), int(num_codes))

    def forward(self, c1_rx: torch.Tensor) -> torch.Tensor:
        bsz = c1_rx.shape[0]
        feat = self.stem(c1_rx).flatten(2).transpose(1, 2)
        feat = self.encoder(feat + self.pos[:, : feat.shape[1]])
        q = self.queries.expand(bsz, -1, -1)
        out, _ = self.cross(q, feat, feat, need_weights=False)
        return self.head(self.norm(out))


class C2PatchIndexPredictor(nn.Module):
    def __init__(
        self,
        c1_ch: int = 16,
        c2_ch: int = 20,
        num_codes: int = 4096,
        latent_h: int = 16,
        latent_w: int = 16,
        patch_size: int = 4,
        embed_dim: int = 256,
        dropout: float = 0.0,
        use_c2_proposal: bool = False,
    ) -> None:
        super().__init__()
        self.c2_ch = int(c2_ch)
        self.num_codes = int(num_codes)
        self.patch_size = int(patch_size)
        self.use_c2_proposal = bool(use_c2_proposal)
        if int(latent_h) % self.patch_size != 0 or int(latent_w) % self.patch_size != 0:
            raise ValueError(f"patch_size={self.patch_size} must divide latent size {(latent_h, latent_w)}")
        self.grid_h = int(latent_h) // self.patch_size
        self.grid_w = int(latent_w) // self.patch_size
        self.stem = nn.Sequential(
            nn.Conv2d(int(c1_ch), int(embed_dim), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(embed_dim), int(embed_dim), 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)
        self.channel_embed = nn.Embedding(self.c2_ch, int(embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1, self.grid_h, self.grid_w, int(embed_dim)))
        self.norm = nn.LayerNorm(int(embed_dim))
        self.drop = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(embed_dim), int(num_codes))
        if self.use_c2_proposal:
            self.proposal_head = nn.Linear(int(embed_dim), self.patch_size * self.patch_size)

    def forward(self, c1_rx: torch.Tensor) -> torch.Tensor:
        feat = self.pool(self.stem(c1_rx)).permute(0, 2, 3, 1)
        ch = self.channel_embed(torch.arange(self.c2_ch, device=c1_rx.device)).view(1, self.c2_ch, 1, 1, -1)
        tokens = self.drop(self.norm(feat.unsqueeze(1) + ch + self.pos))
        logits = self.head(tokens)
        if not self.use_c2_proposal:
            return logits
        patches = self.proposal_head(tokens)
        c2_proposal = patch_tokens_to_map(patches, self.patch_size)
        return logits, c2_proposal


class C2PatchSpatialTransformer(nn.Module):
    def __init__(
        self,
        c1_ch: int = 16,
        c2_ch: int = 20,
        num_codes: int = 4096,
        latent_h: int = 16,
        latent_w: int = 16,
        patch_size: int = 4,
        embed_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_c2_proposal: bool = False,
    ) -> None:
        super().__init__()
        self.c2_ch = int(c2_ch)
        self.num_codes = int(num_codes)
        self.patch_size = int(patch_size)
        self.use_c2_proposal = bool(use_c2_proposal)
        if int(latent_h) % self.patch_size != 0 or int(latent_w) % self.patch_size != 0:
            raise ValueError(f"patch_size={self.patch_size} must divide latent size {(latent_h, latent_w)}")
        self.grid_h = int(latent_h) // self.patch_size
        self.grid_w = int(latent_w) // self.patch_size
        self.stem = nn.Sequential(
            nn.Conv2d(int(c1_ch), int(embed_dim), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(embed_dim), int(embed_dim), 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)
        self.channel_embed = nn.Embedding(self.c2_ch, int(embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, self.c2_ch, self.grid_h, self.grid_w, int(embed_dim)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(heads),
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(int(embed_dim))
        self.head = nn.Linear(int(embed_dim), int(num_codes))
        if self.use_c2_proposal:
            self.proposal_head = nn.Linear(int(embed_dim), self.patch_size * self.patch_size)

    def forward(self, c1_rx: torch.Tensor) -> torch.Tensor:
        bsz = c1_rx.shape[0]
        feat = self.pool(self.stem(c1_rx)).permute(0, 2, 3, 1)
        feat = feat.unsqueeze(1).expand(-1, self.c2_ch, -1, -1, -1)
        ch = self.channel_embed(torch.arange(self.c2_ch, device=c1_rx.device)).view(1, self.c2_ch, 1, 1, -1)
        tokens = (feat + ch + self.pos).reshape(bsz, self.c2_ch * self.grid_h * self.grid_w, -1)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens).reshape(bsz, self.c2_ch, self.grid_h, self.grid_w, -1)
        logits = self.head(tokens)
        if not self.use_c2_proposal:
            return logits
        patches = self.proposal_head(tokens)
        c2_proposal = patch_tokens_to_map(patches, self.patch_size)
        return logits, c2_proposal


def patch_tokens_to_map(patches: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patches.ndim != 5:
        raise ValueError(f"expected patch tokens [B,C,Gh,Gw,P*P], got {tuple(patches.shape)}")
    bsz, channels, gh, gw, dim = patches.shape
    p = int(patch_size)
    if dim != p * p:
        raise ValueError(f"patch token dim={dim} does not match patch_size={p}")
    patches = patches.reshape(bsz, channels, gh, gw, p, p)
    return patches.permute(0, 1, 2, 4, 3, 5).reshape(bsz, channels, gh * p, gw * p)


def build_predictor(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if str(getattr(args, "quantizer", "vq")).lower() == "patch_vq":
        if str(getattr(args, "predictor", "")).lower() in {"patch_cnn", "cnn"}:
            return C2PatchIndexPredictor(
                c1_ch=int(args.c1_ch),
                c2_ch=int(args.latent_ch) - int(args.c1_ch),
                num_codes=int(args.k),
                latent_h=int(args.latent_h),
                latent_w=int(args.latent_w),
                patch_size=int(getattr(args, "patch_size", 4)),
                embed_dim=int(args.pred_embed_dim),
                dropout=float(args.pred_dropout),
                use_c2_proposal=bool(getattr(args, "use_c2_proposal_prior", False)),
            ).to(device)
        return C2PatchSpatialTransformer(
            c1_ch=int(args.c1_ch),
            c2_ch=int(args.latent_ch) - int(args.c1_ch),
            num_codes=int(args.k),
            latent_h=int(args.latent_h),
            latent_w=int(args.latent_w),
            patch_size=int(getattr(args, "patch_size", 4)),
            embed_dim=int(args.pred_embed_dim),
            depth=int(args.pred_depth),
            heads=int(args.pred_heads),
            mlp_ratio=float(args.pred_mlp_ratio),
            dropout=float(args.pred_dropout),
            use_c2_proposal=bool(getattr(args, "use_c2_proposal_prior", False)),
        ).to(device)
    return C2IndexPredictor(
        c1_ch=int(args.c1_ch),
        c2_ch=int(args.latent_ch) - int(args.c1_ch),
        num_codes=int(args.k),
        latent_h=int(args.latent_h),
        latent_w=int(args.latent_w),
        embed_dim=int(args.pred_embed_dim),
        depth=int(args.pred_depth),
        heads=int(args.pred_heads),
        mlp_ratio=float(args.pred_mlp_ratio),
        dropout=float(args.pred_dropout),
    ).to(device)


def topk_accuracies(logits: torch.Tensor, target: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    max_k = min(max(ks), logits.shape[-1])
    pred = logits.topk(max_k, dim=-1).indices
    out = {}
    for k in ks:
        kk = min(k, logits.shape[-1])
        ok = pred[..., :kk].eq(target.unsqueeze(-1)).any(dim=-1).float().mean()
        out[f"top{k}_acc"] = float(ok.item())
    return out


def logits_features(logits: torch.Tensor, q_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits.float(), dim=-1)
    top2 = probs.topk(2, dim=-1).values
    conf = top2[..., 0]
    margin = top2[..., 0] - top2[..., 1]
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1) / math.log(logits.shape[-1])
    q_norm = q_pred.float().square().mean(dim=(2, 3)).sqrt()
    if conf.ndim == 4:
        conf = conf.mean(dim=(2, 3))
        margin = margin.mean(dim=(2, 3))
        entropy = entropy.mean(dim=(2, 3))
    return conf, margin, entropy, q_norm


class C2Gate(nn.Module):
    def __init__(
        self,
        c1_ch: int = 16,
        c2_ch: int = 20,
        hidden: int = 128,
        c1_feat: int = 64,
    ) -> None:
        super().__init__()
        self.c2_ch = int(c2_ch)
        self.c1_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(c1_ch), int(c1_feat)),
            nn.GELU(),
        )
        self.channel_embed = nn.Embedding(self.c2_ch, int(c1_feat))
        in_dim = int(c1_feat) * 2 + 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(
        self,
        c1_rx: torch.Tensor,
        confidence: torch.Tensor,
        margin: torch.Tensor,
        entropy: torch.Tensor,
        q_norm: torch.Tensor,
    ) -> torch.Tensor:
        bsz = c1_rx.shape[0]
        c1_feat = self.c1_pool(c1_rx).unsqueeze(1).expand(-1, self.c2_ch, -1)
        ids = torch.arange(self.c2_ch, device=c1_rx.device).view(1, self.c2_ch).expand(bsz, -1)
        ch_feat = self.channel_embed(ids)
        scalars = torch.stack([confidence, margin, entropy, q_norm], dim=-1)
        return self.mlp(torch.cat([c1_feat, ch_feat, scalars], dim=-1)).squeeze(-1)


def build_gate(args: argparse.Namespace, device: torch.device) -> C2Gate:
    return C2Gate(
        c1_ch=int(args.c1_ch),
        c2_ch=int(args.latent_ch) - int(args.c1_ch),
        hidden=int(args.gate_hidden),
        c1_feat=int(args.gate_c1_feat),
    ).to(device)


@torch.no_grad()
def greedy_oracle_gate_labels(
    decoder: nn.Module,
    c1_rx: torch.Tensor,
    q_pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, c2_ch, h, w = q_pred.shape
    selected = torch.zeros_like(q_pred)
    labels = torch.zeros(bsz, c2_ch, device=q_pred.device, dtype=torch.float32)
    base = decoder(torch.cat([c1_rx, selected], dim=1)).clamp(0.0, 1.0)
    current = (base.float() - target.float()).square().mean(dim=(1, 2, 3))
    for j in range(c2_ch):
        cand = selected.clone()
        cand[:, j] = q_pred[:, j]
        recon = decoder(torch.cat([c1_rx, cand], dim=1)).clamp(0.0, 1.0)
        mse = (recon.float() - target.float()).square().mean(dim=(1, 2, 3))
        keep = mse < (current - float(margin))
        labels[:, j] = keep.float()
        if bool(keep.any()):
            selected[keep, j] = q_pred[keep, j]
            current = torch.where(keep, mse, current)
    return labels, selected


def gate_binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_b = pred.bool()
    target_b = target.bool()
    tp = (pred_b & target_b).sum().float()
    fp = (pred_b & ~target_b).sum().float()
    fn = (~pred_b & target_b).sum().float()
    acc = pred_b.eq(target_b).float().mean()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    return {
        "gate_acc": float(acc.item()),
        "gate_precision": float(precision.item()),
        "gate_recall": float(recall.item()),
    }


def add_common_cli(p: argparse.ArgumentParser, *, default_k: int = 4096) -> None:
    p.add_argument("--data-dir", type=str, default="/workspace/yongjia/datasets/DIV2K")
    p.add_argument("--save-dir", type=str, default=default_v01_save_dir(default_k))
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default="")
    p.add_argument("--init-jscc-encoder", type=str, default=default_jscc_encoder_c36_snr9())
    p.add_argument("--init-jscc-decoder", type=str, default=default_jscc_decoder_c36_snr9())
    p.add_argument("--snr-db", type=float, default=9.0)
    p.add_argument("--latent-ch", type=int, default=36)
    p.add_argument("--c1-ch", type=int, default=16)
    p.add_argument("--latent-h", type=int, default=16)
    p.add_argument("--latent-w", type=int, default=16)
    p.add_argument("--k", type=int, default=int(default_k))
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--test-batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--val-num-workers", type=int, default=8)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--latest-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--cpu", action="store_true")
