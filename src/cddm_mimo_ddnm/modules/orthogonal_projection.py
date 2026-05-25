from __future__ import annotations

import math

import torch
import torch.nn as nn


def make_dct_projection(
    out_dim: int = 4,
    in_dim: int = 16,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the first ``out_dim`` orthonormal DCT-II rows as A [out_dim, in_dim]."""
    if out_dim <= 0 or in_dim <= 0 or out_dim > in_dim:
        raise ValueError(f"invalid projection shape: out_dim={out_dim}, in_dim={in_dim}")
    n = torch.arange(in_dim, dtype=dtype)
    rows = []
    for k in range(out_dim):
        if k == 0:
            row = torch.full((in_dim,), 1.0 / math.sqrt(in_dim), dtype=dtype)
        else:
            row = math.sqrt(2.0 / in_dim) * torch.cos(
                math.pi * (n + 0.5) * float(k) / float(in_dim)
            )
        rows.append(row)
    return torch.stack(rows, dim=0)


class FixedOrthogonalProjector(nn.Module):
    """Fixed semi-orthogonal truncation matrix A with A A^T = I.

    Input and output tensors use BCHW layout. ``encode`` maps C=16 to C=4,
    ``decode`` applies A^T, and ``project`` applies A^T A.
    """

    def __init__(self, in_dim: int = 16, out_dim: int = 4, init: str = "dct") -> None:
        super().__init__()
        if init != "dct":
            raise ValueError(f"unsupported fixed projection init={init!r}; currently only 'dct'")
        a = make_dct_projection(out_dim=out_dim, in_dim=in_dim)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.init = init
        self.register_buffer("A", a, persistent=True)
        self.A.requires_grad_(False)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[1] != self.in_dim:
            raise ValueError(f"expected latent channels={self.in_dim}, got {z.shape[1]}")
        a = self.A.to(device=z.device, dtype=z.dtype)
        return torch.einsum("oc,bchw->bohw", a, z)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[1] != self.out_dim:
            raise ValueError(f"expected observed channels={self.out_dim}, got {y.shape[1]}")
        a_t = self.A.t().to(device=y.device, dtype=y.dtype)
        return torch.einsum("co,bohw->bchw", a_t, y)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(z))

    def null(self, z: torch.Tensor) -> torch.Tensor:
        return z - self.project(z)

    @torch.no_grad()
    def orthogonality_error(self) -> float:
        a = self.A.detach().cpu().double()
        eye = torch.eye(self.out_dim, dtype=a.dtype)
        return float((a @ a.t() - eye).abs().max().item())
