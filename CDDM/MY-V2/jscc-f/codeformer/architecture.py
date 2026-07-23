"""A 256px CodeFormer adaptation with a 16x16, 256-channel code latent.

The original CodeFormer VQGAN uses six resolution levels for 512px faces.  This
implementation deliberately has five levels, ``256 -> 128 -> 64 -> 32 -> 16``.
Consequently the code latent is always ``[B, 256, 16, 16]``.  The condition
feature list is *only* ``(128, 64, 32)``: no full-resolution 256x256 feature is
created for, or consumed by, SFT/CFT fusion.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


IMAGE_SIZE = 256
LATENT_SIZE = 16
LATENT_CHANNELS = 256
FUSE_RESOLUTIONS = (128, 64, 32)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, int(channels))
    while groups > 1 and int(channels) % groups:
        groups -= 1
    return nn.GroupNorm(groups, int(channels), eps=1e-6, affine=True)


class ResBlock(nn.Module):
    """The GroupNorm/Swish residual block used by the VQGAN backbone."""

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = int(in_channels if out_channels is None else out_channels)
        self.norm1 = _group_norm(int(in_channels))
        self.conv1 = nn.Conv2d(int(in_channels), out_channels, 3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if int(in_channels) == out_channels
            else nn.Conv2d(int(in_channels), out_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """Spatial self-attention at the 16x16 bottleneck."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = _group_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        q = self.q(h).flatten(2).transpose(1, 2)
        k = self.k(h).flatten(2)
        v = self.v(h).flatten(2).transpose(1, 2)
        weights = torch.softmax((q @ k) * (x.shape[1] ** -0.5), dim=-1)
        h = (weights @ v).transpose(1, 2).reshape_as(x)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class FuseSFTBlock(nn.Module):
    """CodeFormer-style conditional feature transformation (CFT/SFT).

    ``condition`` originates exclusively from the Layer1 reconstruction ``x1``;
    the HQ image is never an input to this module at inference.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.encode_condition = ResBlock(2 * channels, channels)
        self.scale = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, condition: Tensor, decoded: Tensor, weight: float = 1.0) -> Tensor:
        if condition.shape != decoded.shape:
            raise ValueError(
                "SFT feature shapes must match, got "
                f"condition={tuple(condition.shape)} decoded={tuple(decoded.shape)}"
            )
        condition = self.encode_condition(torch.cat([condition, decoded], dim=1))
        residual = decoded * self.scale(condition) + self.shift(condition)
        return decoded + float(weight) * residual


class ImageEncoder(nn.Module):
    """Five-level (four-downsample) VQGAN encoder for 256px RGB images."""

    resolutions = (256, 128, 64, 32, 16)

    def __init__(
        self,
        base_channels: int = 64,
        latent_channels: int = LATENT_CHANNELS,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        if int(latent_channels) != LATENT_CHANNELS:
            raise ValueError(f"latent_channels must be {LATENT_CHANNELS}, got {latent_channels}")
        self.base_channels = int(base_channels)
        self.channel_multipliers = (1, 2, 2, 4, 4)
        self.stage_channels = tuple(self.base_channels * m for m in self.channel_multipliers)
        self.input_conv = nn.Conv2d(3, self.stage_channels[0], 3, padding=1)

        stages: list[nn.Module] = []
        downsample: list[nn.Module] = []
        in_channels = self.stage_channels[0]
        for level, out_channels in enumerate(self.stage_channels):
            blocks: list[nn.Module] = []
            for _ in range(int(num_res_blocks)):
                blocks.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
            if level != len(self.stage_channels) - 1:
                downsample.append(Downsample(in_channels))
        self.stages = nn.ModuleList(stages)
        self.downsample = nn.ModuleList(downsample)
        self.mid = nn.Sequential(ResBlock(in_channels), AttentionBlock(in_channels), ResBlock(in_channels))
        self.output = nn.Sequential(
            _group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, LATENT_CHANNELS, 3, padding=1),
        )
        self.feature_channels = {
            resolution: self.stage_channels[level]
            for level, resolution in enumerate(self.resolutions)
            if resolution in FUSE_RESOLUTIONS
        }

    def forward(self, image: Tensor, return_features: bool = False) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected image [B,3,256,256], got {tuple(image.shape)}")
        h = self.input_conv(image)
        features: dict[int, Tensor] = {}
        for level, stage in enumerate(self.stages):
            h = stage(h)
            resolution = self.resolutions[level]
            if resolution in FUSE_RESOLUTIONS:
                features[resolution] = h
            if level < len(self.downsample):
                h = self.downsample[level](h)
        latent = self.output(self.mid(h))
        if tuple(latent.shape[1:]) != (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE):
            raise RuntimeError(f"encoder contract broke: got {tuple(latent.shape)}")
        return (latent, features) if return_features else latent


class ImageDecoder(nn.Module):
    """Mirror of :class:`ImageEncoder`, with 128/64/32 SFT fusion only."""

    resolutions = ImageEncoder.resolutions

    def __init__(
        self,
        base_channels: int = 64,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        base_channels = int(base_channels)
        self.stage_channels = tuple(base_channels * m for m in (1, 2, 2, 4, 4))
        in_channels = self.stage_channels[-1]
        self.input = nn.Conv2d(LATENT_CHANNELS, in_channels, 3, padding=1)
        self.mid = nn.Sequential(ResBlock(in_channels), AttentionBlock(in_channels), ResBlock(in_channels))

        stages: list[nn.Module] = []
        upsample: list[nn.Module] = []
        for level in reversed(range(len(self.stage_channels))):
            out_channels = self.stage_channels[level]
            blocks: list[nn.Module] = []
            for _ in range(int(num_res_blocks)):
                blocks.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
            if level > 0:
                upsample.append(Upsample(in_channels, self.stage_channels[level - 1]))
                in_channels = self.stage_channels[level - 1]
        self.stages = nn.ModuleList(stages)
        self.upsample = nn.ModuleList(upsample)
        self.fuse_blocks = nn.ModuleDict(
            {
                str(resolution): FuseSFTBlock(self.stage_channels[self.resolutions.index(resolution)])
                for resolution in FUSE_RESOLUTIONS
            }
        )
        self.output = nn.Sequential(
            _group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 3, 3, padding=1),
        )

    def forward(
        self,
        latent: Tensor,
        condition_features: dict[int, Tensor] | None = None,
        fusion_weight: float = 0.0,
    ) -> Tensor:
        if tuple(latent.shape[1:]) != (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE):
            raise ValueError(f"expected latent [B,256,16,16], got {tuple(latent.shape)}")
        h = self.mid(self.input(latent))
        for index, stage in enumerate(self.stages):
            h = stage(h)
            resolution = self.resolutions[-1 - index]
            if condition_features is not None and resolution in FUSE_RESOLUTIONS:
                if resolution not in condition_features:
                    raise KeyError(f"missing Layer1 x1 condition at {resolution}x{resolution}")
                h = self.fuse_blocks[str(resolution)](
                    condition_features[resolution], h, fusion_weight
                )
            if index < len(self.upsample):
                h = self.upsample[index](h)
        return torch.sigmoid(self.output(h))


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 1024, embedding_dim: int = LATENT_CHANNELS, beta: float = 0.25) -> None:
        super().__init__()
        if int(embedding_dim) != LATENT_CHANNELS:
            raise ValueError(f"embedding_dim must be {LATENT_CHANNELS}, got {embedding_dim}")
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def lookup(self, indices: Tensor) -> Tensor:
        if indices.ndim != 3 or tuple(indices.shape[-2:]) != (LATENT_SIZE, LATENT_SIZE):
            raise ValueError(f"expected indices [B,16,16], got {tuple(indices.shape)}")
        bsz = indices.shape[0]
        return (
            F.embedding(indices.long().reshape(-1), self.embedding.weight)
            .view(bsz, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def soft_lookup(self, logits: Tensor) -> Tensor:
        if logits.ndim != 3 or logits.shape[1:] != (LATENT_SIZE * LATENT_SIZE, self.codebook_size):
            raise ValueError(
                "expected logits [B,256,K], got "
                f"{tuple(logits.shape)} for K={self.codebook_size}"
            )
        bsz = logits.shape[0]
        soft_codes = torch.softmax(logits, dim=-1) @ self.embedding.weight
        return soft_codes.transpose(1, 2).reshape(bsz, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        if tuple(latent.shape[1:]) != (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE):
            raise ValueError(f"expected latent [B,256,16,16], got {tuple(latent.shape)}")
        bsz = latent.shape[0]
        flat = latent.permute(0, 2, 3, 1).reshape(-1, LATENT_CHANNELS)
        codebook = self.embedding.weight
        distance = (
            flat.float().square().sum(dim=1, keepdim=True)
            + codebook.float().square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat.float() @ codebook.float().t()
        )
        indices = distance.argmin(dim=1)
        quantized = F.embedding(indices, codebook).view(
            bsz, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS
        ).permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(quantized.float(), latent.detach().float())
        commitment_loss = F.mse_loss(quantized.detach().float(), latent.float())
        straight_through = latent + (quantized - latent).detach()
        index_map = indices.view(bsz, LATENT_SIZE, LATENT_SIZE)
        one_hot = F.one_hot(indices, num_classes=self.codebook_size).float().mean(dim=0)
        perplexity = torch.exp(-(one_hot * (one_hot + 1e-10).log()).sum())
        return straight_through, quantized, index_map, {
            "vq_loss": codebook_loss + self.beta * commitment_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }


class VQAutoencoder(nn.Module):
    """HQ stage-I codebook autoencoder, independent of Layer1."""

    def __init__(
        self,
        codebook_size: int = 1024,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = ImageEncoder(base_channels, LATENT_CHANNELS, num_res_blocks)
        self.quantizer = VectorQuantizer(codebook_size, LATENT_CHANNELS, beta)
        self.decoder = ImageDecoder(base_channels, num_res_blocks)

    def encode(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        return self.quantizer(self.encoder(image))

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        quant_st, quant, indices, stats = self.encode(image)
        reconstruction = self.decoder(quant_st)
        return {"reconstruction": reconstruction, "quantized": quant, "indices": indices, **stats}


class TransformerSALayer(nn.Module):
    """Pre-normalized self-attention/MLP block matching CodeFormer topology."""

    def __init__(self, width: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        if int(width) % int(heads):
            raise ValueError(f"transformer width {width} must divide heads {heads}")
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, int(width * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(width * mlp_ratio), width),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, position: Tensor) -> Tensor:
        normed = self.norm1(tokens)
        attended, _ = self.attn(normed + position, normed + position, normed, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class CodeFormer(nn.Module):
    """Predict VQ code indices from ``x1`` and decode with x1-only SFT features."""

    def __init__(
        self,
        vq: VQAutoencoder,
        transformer_width: int = 512,
        transformer_layers: int = 9,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.vq = vq
        self.transformer_width = int(transformer_width)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, LATENT_SIZE * LATENT_SIZE, self.transformer_width)
        )
        self.feature_embedding = nn.Linear(LATENT_CHANNELS, self.transformer_width)
        self.transformer = nn.ModuleList(
            [
                TransformerSALayer(
                    self.transformer_width,
                    int(transformer_heads),
                    transformer_mlp_ratio,
                )
                for _ in range(int(transformer_layers))
            ]
        )
        self.index_head = nn.Sequential(
            nn.LayerNorm(self.transformer_width),
            nn.Linear(self.transformer_width, self.vq.quantizer.codebook_size, bias=False),
        )
        # Do not call ``self.apply`` here: ``vq`` may already hold the trained
        # stage-I codebook/generator, and CodeFormer must not reinitialize it.
        self.feature_embedding.apply(self._init_transformer_weights)
        self.transformer.apply(self._init_transformer_weights)
        self.index_head.apply(self._init_transformer_weights)

    @staticmethod
    def _init_transformer_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def predict_logits(self, x1: Tensor) -> tuple[Tensor, dict[int, Tensor]]:
        lq_latent, features = self.vq.encoder(x1, return_features=True)
        tokens = self.feature_embedding(lq_latent.flatten(2).transpose(1, 2))
        position = self.position_embedding.expand(tokens.shape[0], -1, -1)
        for layer in self.transformer:
            tokens = layer(tokens, position)
        return self.index_head(tokens), features

    def restore(
        self,
        x1: Tensor,
        fusion_weight: float = 1.0,
        soft_decode: bool = False,
    ) -> dict[str, Tensor]:
        """Restore from Layer1 output only; ``x1`` is the deployment input."""
        logits, features = self.predict_logits(x1)
        if soft_decode:
            quantized = self.vq.quantizer.soft_lookup(logits)
            indices = logits.argmax(dim=-1).view(-1, LATENT_SIZE, LATENT_SIZE)
        else:
            indices = logits.argmax(dim=-1).view(-1, LATENT_SIZE, LATENT_SIZE)
            quantized = self.vq.quantizer.lookup(indices)
        output = self.vq.decoder(quantized, features, fusion_weight)
        return {"output": output, "logits": logits, "indices": indices, "quantized": quantized}

    def forward(self, x1: Tensor, **kwargs) -> dict[str, Tensor]:
        return self.restore(x1, **kwargs)

    def freeze_codebook_and_generator(self, freeze_encoder: bool = False) -> None:
        """Match CodeFormer's frozen codebook/generator while retaining CFT trainability."""
        for parameter in self.vq.quantizer.parameters():
            parameter.requires_grad_(False)
        for parameter in self.vq.decoder.parameters():
            parameter.requires_grad_(False)
        for fusion in self.vq.decoder.fuse_blocks.values():
            for parameter in fusion.parameters():
                parameter.requires_grad_(True)
        for parameter in self.vq.encoder.parameters():
            parameter.requires_grad_(not bool(freeze_encoder))


def trainable_parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    """Return de-duplicated parameters that have gradients enabled."""
    seen: set[int] = set()
    parameters: list[nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters
