"""分层式 Swin 语义编解码器。

编码器参考 CDDM 的 JSCC 设计：采用多 stage Swin 主干，末端再压到信道瓶颈 C。
解码器在结构上做镜像恢复，但保留更偏向重建的实现：先扩宽，再逐 stage 重建，最后输出图像。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _apply_channel_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """对 BCHW 特征图沿通道维做 LayerNorm。"""
    bsz, channels, height, width = x.shape
    x = x.flatten(2).transpose(1, 2)
    x = norm(x)
    return x.transpose(1, 2).reshape(bsz, channels, height, width)


def _infer_stage_downsample(num_stages: int, patch_size: int, stem_stride: int) -> tuple[bool, ...]:
    """根据总下采样倍数推断每个 stage 是否执行 2x 下采样。"""
    if patch_size % stem_stride != 0:
        raise ValueError(f"patch_size={patch_size} 必须能被 stem_stride={stem_stride} 整除。")
    ratio = patch_size // stem_stride
    if ratio <= 0 or ratio & (ratio - 1):
        raise ValueError(
            f"patch_size/stem_stride={ratio} 必须为 2 的整数次幂，当前 patch_size={patch_size}, stem_stride={stem_stride}。"
        )
    num_downsamples = int(math.log2(ratio))
    if num_downsamples > max(0, num_stages - 1):
        raise ValueError(
            f"stage 数量不足以实现 patch_size={patch_size} 的总下采样；"
            f"num_stages={num_stages}, 需要 {num_downsamples} 次 2x 下采样。"
        )
    flags = [False] * num_stages
    for idx in range(1, 1 + num_downsamples):
        flags[idx] = True
    return tuple(flags)


def _resolve_hierarchical_cfg(
    *,
    embed_dim: int,
    patch_size: int,
    num_heads: int,
    num_blocks: int,
    stage_embed_dims: tuple[int, ...] | None,
    stage_depths: tuple[int, ...] | None,
    stage_num_heads: tuple[int, ...] | None,
    stem_stride: int | None,
    stage_downsample: tuple[bool, ...] | None,
) -> tuple[list[int], list[int], list[int], int, list[bool]]:
    """统一旧版单尺度配置与新版分层配置。"""
    if stage_embed_dims is None or stage_depths is None or stage_num_heads is None:
        resolved_stem_stride = patch_size if stem_stride is None else stem_stride
        return [embed_dim], [num_blocks], [num_heads], resolved_stem_stride, [False]

    dims = list(stage_embed_dims)
    depths = list(stage_depths)
    heads = list(stage_num_heads)
    if not dims or len(dims) != len(depths) or len(dims) != len(heads):
        raise ValueError("stage_embed_dims、stage_depths、stage_num_heads 长度必须一致且非空。")

    resolved_stem_stride = 2 if stem_stride is None else stem_stride
    if stage_downsample is None:
        downsamples = list(_infer_stage_downsample(len(dims), patch_size, resolved_stem_stride))
    else:
        downsamples = list(stage_downsample)
        if len(downsamples) != len(dims):
            raise ValueError("stage_downsample 的长度必须与 stage_embed_dims 一致。")

    total_stride = resolved_stem_stride * (2 ** sum(downsamples))
    if total_stride != patch_size:
        raise ValueError(
            f"总下采样倍数不匹配：stem_stride={resolved_stem_stride}, "
            f"stage_downsample={downsamples} -> total_stride={total_stride}, 但 patch_size={patch_size}。"
        )
    return dims, depths, heads, resolved_stem_stride, downsamples


# ---------------------------------------------------------------------------
# 工具函数：窗口划分 / 还原
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """将特征图划分为不重叠窗口。"""
    bsz, height, width, channels = x.shape
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError(
            f"窗口大小 {window_size} 与特征尺寸 {height}x{width} 不兼容，需整除。"
        )
    x = x.view(
        bsz,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    """将窗口张量还原为特征图。"""
    num_windows = (height // window_size) * (width // window_size)
    bsz = windows.shape[0] // num_windows
    x = windows.view(
        bsz,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(bsz, height, width, -1)


# ---------------------------------------------------------------------------
# 窗口自注意力（支持 shifted window）
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """带相对位置偏置的窗口多头自注意力。"""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} 必须能被 num_heads={num_heads} 整除。")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_bias = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coords_flat = coords.flatten(1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_idx", rel.sum(-1))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, num_tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_windows, num_tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.rel_pos_bias[self.rel_idx.view(-1)].view(num_tokens, num_tokens, self.num_heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        if attn_mask is not None:
            num_windows = attn_mask.shape[0]
            attn = attn.view(batch_windows // num_windows, num_windows, self.num_heads, num_tokens, num_tokens)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, num_tokens, num_tokens)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_windows, num_tokens, channels)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Swin Transformer Block
# ---------------------------------------------------------------------------

class SwinBlock(nn.Module):
    """单个 Swin Transformer Block（含可选 shifted window）。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 4,
        shift: bool = False,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self._attn_mask_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def _compute_attn_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        key = (height, width, str(device))
        if key not in self._attn_mask_cache:
            img_mask = torch.zeros(1, height, width, 1, device=device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    img_mask[:, h_slice, w_slice, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
            self._attn_mask_cache[key] = attn_mask
        return self._attn_mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self._compute_attn_mask(height, width, x.device)
        x_windows = self.norm1(x_windows)
        attn_out = self.attn(x_windows, attn_mask)

        attn_out = attn_out.view(-1, self.window_size, self.window_size, channels)
        x = window_reverse(attn_out, self.window_size, height, width)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.permute(0, 3, 1, 2)
        x = shortcut + x

        residual = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


# ---------------------------------------------------------------------------
# Patch Embedding / Patch Merge
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """卷积 patch stem，将图像映射到第一个 stage 特征。"""

    def __init__(self, patch_size: int = 2, in_chans: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return _apply_channel_norm(x, self.norm)


class ChannelProject(nn.Module):
    """仅调整通道宽度，不改变空间分辨率。"""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return _apply_channel_norm(x, self.norm)


class PatchMerging(nn.Module):
    """2x 下采样，功能上对应 JSCC 编码器中的 PatchMerging。"""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"PatchMerging 需要偶数尺寸输入，当前为 {height}x{width}。")
        x = x.permute(0, 2, 3, 1)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(bsz, -1, 4 * channels)
        x = self.norm(x)
        x = self.reduction(x)
        return x.transpose(1, 2).reshape(bsz, -1, height // 2, width // 2)


class PatchReverseMerging(nn.Module):
    """2x 上采样，功能上对应 JSCC 解码器中的 PatchReverseMerging。"""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, out_dim * 4)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.expand(x)
        x = x.transpose(1, 2).reshape(bsz, self.out_dim * 4, height, width)
        return self.pixel_shuffle(x)


class EncoderStage(nn.Module):
    """分层编码 stage：可选下采样后接多个 SwinBlock。"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.downsample = PatchMerging(in_dim, out_dim) if downsample else None
        self.proj = ChannelProject(in_dim, out_dim) if not downsample and in_dim != out_dim else None
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim=out_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift=(idx % 2 == 1),
                )
                for idx in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        elif self.proj is not None:
            x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class DecoderStage(nn.Module):
    """分层解码 stage：先做 Swin 重建，再视需要上采样。"""

    def __init__(
        self,
        dim: int,
        next_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        upsample: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift=(idx % 2 == 1),
                )
                for idx in range(depth)
            ]
        )
        self.upsample = PatchReverseMerging(dim, next_dim) if upsample else None
        self.proj = ChannelProject(dim, next_dim) if not upsample and dim != next_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        elif self.proj is not None:
            x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# 语义编码器（支持 VAE 潜空间正则化）
# ---------------------------------------------------------------------------

class SemanticEncoder(nn.Module):
    """分层式语义编码器。

    当提供 stage 配置时：
      - `stage_depths[i]` 对应论文中的 `Ni`
      - `stage_embed_dims[i]` 对应论文中的 `Pi`
      - `len(stage_depths)` 对应论文中的 `M`
      - `embed_dim` 对应最终发送到信道的瓶颈维度 `C`
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        patch_size: int = 4,
        num_heads: int = 4,
        window_size: int = 4,
        num_blocks: int = 2,
        stage_embed_dims: tuple[int, ...] | None = None,
        stage_depths: tuple[int, ...] | None = None,
        stage_num_heads: tuple[int, ...] | None = None,
        stem_stride: int | None = None,
        stage_downsample: tuple[bool, ...] | None = None,
        use_vae: bool = False,
    ) -> None:
        super().__init__()
        self.use_vae = use_vae
        self.stage_embed_dims, self.stage_depths, self.stage_num_heads, self.stem_stride, self.stage_downsample = (
            _resolve_hierarchical_cfg(
                embed_dim=embed_dim,
                patch_size=patch_size,
                num_heads=num_heads,
                num_blocks=num_blocks,
                stage_embed_dims=stage_embed_dims,
                stage_depths=stage_depths,
                stage_num_heads=stage_num_heads,
                stem_stride=stem_stride,
                stage_downsample=stage_downsample,
            )
        )

        self.patch_embed = PatchEmbed(self.stem_stride, in_channels, self.stage_embed_dims[0])
        self.layers = nn.ModuleList()
        current_dim = self.stage_embed_dims[0]
        for out_dim, depth, heads, downsample in zip(
            self.stage_embed_dims,
            self.stage_depths,
            self.stage_num_heads,
            self.stage_downsample,
        ):
            self.layers.append(
                EncoderStage(
                    in_dim=current_dim,
                    out_dim=out_dim,
                    depth=depth,
                    num_heads=heads,
                    window_size=window_size,
                    downsample=downsample,
                )
            )
            current_dim = out_dim

        self.latent_dim = current_dim
        self.norm = nn.LayerNorm(current_dim)
        self.head = nn.Conv2d(current_dim, embed_dim, kernel_size=1)
        if use_vae:
            # 在 embed 维上做高斯 VAE，与 decoder 输入维一致
            self.vae_proj = nn.Conv2d(embed_dim, 2 * embed_dim, kernel_size=1)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = _apply_channel_norm(x, self.norm)
        return x

    def encode(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """提取特征并可选地进行 VAE 重参数化。

        Returns:
            z:      供解码器使用 [B, embed_dim, H', W']；无 VAE 为 head(features)，有 VAE 为在 **embed 空间** 上采样
            mu:     VAE 均值 [B, embed_dim, H', W']；use_vae=False 时为 None
            logvar: VAE 对数方差 [B, embed_dim, H', W']；use_vae=False 时为 None
        """
        features = self._extract_features(x)
        if self.use_vae:
            h = self.head(features)  # latent_dim -> embed_dim，与 vae_proj 的 in_ch 一致
            moments = self.vae_proj(h)
            mu, logvar = moments.chunk(2, dim=1)
            logvar = logvar.clamp(-30.0, 20.0)
            if sample:
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
            else:
                z = mu
            return z, mu, logvar
        z = self.head(features)
        return z, None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _, _ = self.encode(x, sample=self.training)
        return z


# ---------------------------------------------------------------------------
# 语义解码器
# ---------------------------------------------------------------------------

class SemanticDecoder(nn.Module):
    """分层式语义解码器，结构上与编码器大体镜像。

    对于 CDDM 对齐配置（stem_stride=2，且除首个 stage 外其余 stage 都做 2x 下采样），
    采用与原 JSCC 解码器一致的空间恢复策略：每个 decoder stage 都执行一次 2x 上采样，
    最后一个 stage 对应恢复编码器最前面的 patch stem。

    对其他历史/兼容配置，保留旧实现：仅对镜像的下采样 stage 上采样，并在末尾用
    `final_expand` 恢复 stem 的下采样倍数。
    """

    def __init__(
        self,
        out_channels: int = 3,
        embed_dim: int = 64,
        patch_size: int = 4,
        num_heads: int = 4,
        window_size: int = 4,
        num_refine_blocks: int = 2,
        stage_embed_dims: tuple[int, ...] | None = None,
        stage_depths: tuple[int, ...] | None = None,
        stage_num_heads: tuple[int, ...] | None = None,
        stem_stride: int | None = None,
        stage_downsample: tuple[bool, ...] | None = None,
    ) -> None:
        super().__init__()
        enc_dims, enc_depths, enc_heads, self.stem_stride, enc_downsample = _resolve_hierarchical_cfg(
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_heads=num_heads,
            num_blocks=max(1, num_refine_blocks),
            stage_embed_dims=stage_embed_dims,
            stage_depths=stage_depths,
            stage_num_heads=stage_num_heads,
            stem_stride=stem_stride,
            stage_downsample=stage_downsample,
        )

        self.decoder_dims = list(reversed(enc_dims))
        self.decoder_depths = list(reversed(enc_depths))
        self.decoder_heads = list(reversed(enc_heads))
        cddm_style_schedule = (
            self.stem_stride == 2
            and bool(enc_downsample)
            and not enc_downsample[0]
            and all(enc_downsample[1:])
        )
        self.stage_upsample = [True] * len(self.decoder_dims) if cddm_style_schedule else list(reversed(enc_downsample))

        self.cddm_style = cddm_style_schedule
        self.head = nn.Conv2d(embed_dim, self.decoder_dims[0], kernel_size=3, padding=1)
        self.layers = nn.ModuleList()
        current_dim = self.decoder_dims[0]
        n_stages = len(self.decoder_dims)
        for idx, (stage_dim, depth, heads, upsample) in enumerate(
            zip(self.decoder_dims, self.decoder_depths, self.decoder_heads, self.stage_upsample)
        ):
            is_last = (idx + 1 == n_stages)
            if is_last and cddm_style_schedule:
                next_dim = out_channels
            elif not is_last:
                next_dim = self.decoder_dims[idx + 1]
            else:
                next_dim = stage_dim
            self.layers.append(
                DecoderStage(
                    dim=current_dim,
                    next_dim=next_dim,
                    depth=depth,
                    num_heads=heads,
                    window_size=window_size,
                    upsample=upsample,
                )
            )
            current_dim = next_dim

        if cddm_style_schedule:
            self.final_expand = nn.Identity()
            self.refine_blocks = nn.ModuleList()
            self.output_conv = nn.Identity()
        else:
            self.final_expand = (
                nn.Sequential(
                    nn.Conv2d(current_dim, current_dim * self.stem_stride * self.stem_stride, kernel_size=1),
                    nn.PixelShuffle(self.stem_stride),
                )
                if self.stem_stride > 1
                else nn.Identity()
            )
            self.refine_blocks = nn.ModuleList(
                [
                    SwinBlock(
                        dim=current_dim,
                        num_heads=self.decoder_heads[-1],
                        window_size=window_size,
                        shift=(idx % 2 == 1),
                    )
                    for idx in range(num_refine_blocks)
                ]
            )
            hidden_dim = max(32, current_dim // 2)
            self.output_conv = nn.Sequential(
                nn.Conv2d(current_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.head(z)
        for layer in self.layers:
            x = layer(x)
        x = self.final_expand(x)
        for blk in self.refine_blocks:
            x = blk(x)
        return self.output_conv(x)
