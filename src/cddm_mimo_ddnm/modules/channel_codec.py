from __future__ import annotations

import torch
import torch.nn as nn


def composed_1x1_weight(net: nn.Conv2d | nn.Sequential) -> torch.Tensor:
    """将 1×1 Conv（或两层瓶颈链）折叠为矩阵 W，满足 y = W @ x（列向量 x，通道维）。"""

    def _w(conv: nn.Conv2d) -> torch.Tensor:
        return conv.weight.squeeze(-1).squeeze(-1)

    if isinstance(net, nn.Conv2d):
        return _w(net)
    if isinstance(net, nn.Sequential):
        acc: torch.Tensor | None = None
        for m in net:
            if isinstance(m, nn.Conv2d):
                wi = _w(m)
                acc = wi if acc is None else wi @ acc
        if acc is None:
            raise ValueError("Sequential 中未找到 Conv2d")
        return acc
    raise TypeError(f"不支持的类型: {type(net)}")


class ChannelEncoder(nn.Module):
    """线性信道编码器：仅使用 1×1 Conv 映射，bias=False。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bottleneck_dim: int | None = None,
    ) -> None:
        super().__init__()
        if bottleneck_dim is not None and bottleneck_dim < min(in_channels, out_channels):
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False),
                nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ChannelDecoder(nn.Module):
    """线性信道解码器：仅使用 1×1 Conv 映射，bias=False。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bottleneck_dim: int | None = None,
    ) -> None:
        super().__init__()
        if bottleneck_dim is not None and bottleneck_dim < min(in_channels, out_channels):
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False),
                nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)
