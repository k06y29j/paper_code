from __future__ import annotations

import torch
import torch.nn as nn


def composed_1x1_weight(net: nn.Conv2d | nn.Sequential) -> torch.Tensor:
    """将 1×1 Conv 链折叠为矩阵 W，满足 y = W @ x（列向量 x，通道维）。"""

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


def _make_linear_1x1_stack(
    in_channels: int,
    out_channels: int,
    *,
    linear_depth: int,
    hidden_channels: int,
) -> nn.Conv2d | nn.Sequential:
    """构造纯线性 1x1 Conv 链。

    depth=1: in -> out
    depth=2: in -> hidden -> out
    depth=3: in -> hidden -> hidden -> out
    """
    if linear_depth < 1:
        raise ValueError(f"linear_depth 必须 >= 1，收到 {linear_depth}")
    if linear_depth == 1:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
    ]
    for _ in range(linear_depth - 2):
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False))
    layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False))
    return nn.Sequential(*layers)


class ChannelEncoder(nn.Module):
    """线性信道编码器：仅使用 1×1 Conv 映射，bias=False。

    ``linear_depth`` 用于深线性参数化实验；不加激活，因此整体仍可由
    :func:`composed_1x1_weight` 折叠成一个有效退化矩阵。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bottleneck_dim: int | None = None,
        linear_depth: int = 1,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels or in_channels)
        if linear_depth > 1:
            self.net = _make_linear_1x1_stack(
                in_channels,
                out_channels,
                linear_depth=int(linear_depth),
                hidden_channels=hidden,
            )
        elif bottleneck_dim is not None and bottleneck_dim < min(in_channels, out_channels):
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
        linear_depth: int = 1,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels or out_channels)
        if linear_depth > 1:
            self.net = _make_linear_1x1_stack(
                in_channels,
                out_channels,
                linear_depth=int(linear_depth),
                hidden_channels=hidden,
            )
        elif bottleneck_dim is not None and bottleneck_dim < min(in_channels, out_channels):
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False),
                nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)
