"""DDNM 一致性修正模块与 U-Net 骨干去噪网络。

用于在特征空间（信道解码后、语义解码前）对含噪特征进行去噪修正。

包含：
  - DoubleConv / Down / Up：U-Net 基础块
  - UNetDenoiser：特征空间 U-Net 去噪器（接受含噪特征 + 条件特征 + 时间步）
  - DDNMCorrector：DDNM 一致性修正（零空间投影）
"""

from __future__ import annotations

from typing import Callable

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LinearOp = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# U-Net 基础块
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """两层 3×3 卷积 + GroupNorm + GELU。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=min(out_ch, 8), num_channels=out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=min(out_ch, 8), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


# ---------------------------------------------------------------------------
# 时间嵌入
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbed(nn.Module):
    """正弦时间步嵌入（连续时间版本）。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]（0~1 归一化时间步）
        half = self.hidden_dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.unsqueeze(1) * freq.unsqueeze(0)  # [B, half]
        emb = torch.cat([args.sin(), args.cos()], dim=1)  # [B, hidden_dim]
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# U-Net 去噪器（在特征空间工作）
# ---------------------------------------------------------------------------

class UNetDenoiser(nn.Module):
    """特征空间 U-Net 去噪器，接受：
      - x_t: 当前含噪特征 [B, channels, H', W']
      - t: 时间步 [B]（0~1）
      - cond: 条件特征（可选）；use_cond=True 时必须提供，与 x_t 同空间尺寸 [B, channels, H', W']
    输出预测噪声 ε̂: [B, channels, H', W']
    """

    def __init__(self, channels: int = 64, hidden_dim: int = 128, use_cond: bool = False) -> None:
        super().__init__()
        self.use_cond = use_cond
        c = hidden_dim
        self.time_emb = SinusoidalTimeEmbed(c)

        in_ch = channels * 2 if use_cond else channels
        self.enc0 = DoubleConv(in_ch, c)
        self.enc1 = DownBlock(c, c * 2)
        self.enc2 = DownBlock(c * 2, c * 4)

        self.bottleneck = DoubleConv(c * 4, c * 4)

        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)

        self.head = nn.Conv2d(c, channels, kernel_size=1)

        # 时间步投影到各层
        self.t_proj0 = nn.Linear(c, c)
        self.t_proj1 = nn.Linear(c, c * 2)
        self.t_proj2 = nn.Linear(c, c * 4)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)  # [B, c]

        if self.use_cond:
            if cond is None:
                raise ValueError("UNetDenoiser(use_cond=True) 需要 cond")
            inp = torch.cat([x_t, cond], dim=1)
        else:
            inp = x_t
        e0 = self.enc0(inp) + self.t_proj0(t_emb).unsqueeze(-1).unsqueeze(-1)
        e1 = self.enc1(e0) + self.t_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)
        e2 = self.enc2(e1) + self.t_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)

        b = self.bottleneck(e2)

        d2 = self.dec2(b, e1)
        d1 = self.dec1(d2, e0)

        return self.head(d1)


# ---------------------------------------------------------------------------
# DDNM+ 一致性修正器
# ---------------------------------------------------------------------------

class DDNMCorrector:
    """潜空间 DDNM+ 数据一致性修正（论文公式 15）。

    修正公式：
        ĝ_{0|t} = u_{0|t} + λ_t · (f̂ - A†A · u_{0|t})

    其中：
        u_{0|t}   : 扩散模型当前步对纯净特征的无条件估计（z0_pred）
        f̂        : 信道解码后的接收端观测特征（z_cond）
        A†A(·)   : 近似正交投影算子，由信道编解码器级联实现：
                    A†A(u) ≈ h_dec(h_enc(u))   （论文算法 1 第 3 步）
        λ_t       : 自适应权重，由扩散步噪声水平与物理层信道噪声 σ_y 联合决定
                    （论文公式 16）

    与原始 DDNM 梯度下降版本（x ← x - η·A^T(Ax-y)）相比，
    此直接修正公式在单次迭代内即可精确完成零空间/值域分解。
    """

    def __init__(self, n_iter: int = 1) -> None:
        self.n_iter = n_iter

    def correct(
        self,
        z0_pred: torch.Tensor,
        z_cond: torch.Tensor,
        a_dag_a_fn: LinearOp,
        lambda_t: float | torch.Tensor,
    ) -> torch.Tensor:
        """DDNM+ 数据一致性修正（论文公式 15）。

        Args:
            z0_pred:    扩散模型当前步的纯净特征估计 u_{0|t}
            z_cond:     信道解码后的观测特征 f̂
            a_dag_a_fn: 近似投影算子 A†A，传入 channel_decoder ∘ channel_encoder
            lambda_t:   自适应权重（标量或张量），由 σ_t 与 σ_y 计算
        """
        result = z0_pred
        for _ in range(self.n_iter):
            proj = a_dag_a_fn(result)           # A†A · u_{0|t}
            result = result + lambda_t * (z_cond - proj)   # 公式 15
        return result
