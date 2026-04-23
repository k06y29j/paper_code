"""MIMO 信道仿真与 MMSE 均衡（支持模拟 JSCC 传输）。

传输模型：y = H·x + n
MMSE 均衡：x̂ = W·y，W = (H^H·H + σ²I)^{-1}·H^H

模拟 JSCC 传输流程：
  1. 将实数特征张量重解释为复数符号（相邻两通道配对为实/虚部）
  2. 按 n_tx 分块发送，每块经随机 Rayleigh 信道 + 加性高斯噪声
  3. MMSE 均衡后还原为实数特征张量
"""

from __future__ import annotations

import math

import torch


class MIMOChannelMMSE:
    """MIMO 瑞利衰落信道 + 线性 MMSE 检测。

    支持模拟 JSCC：直接传输实数特征（无 QAM 量化），
    保持端到端可微的语义信息。
    """

    def __init__(
        self,
        n_tx: int = 2,
        n_rx: int = 2,
        snr_db: float = 10.0,
        fading: str = "rayleigh",
    ) -> None:
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.snr_db = snr_db
        self.fading = fading

    # ------------------------------------------------------------------
    # 内部：MMSE 均衡
    # ------------------------------------------------------------------

    def _mmse_detect(
        self, y: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        """MMSE 均衡并同步计算各流 SINR（论文公式 4、5）。

        Args:
            y: [n_rx]，接收信号（复数）
            h: [n_rx, n_tx]，信道矩阵（复数）
        Returns:
            x_hat:    [n_tx] 均衡后的符号估计
            sinr_mean: 所有发送流 SINR 的算术平均值
            beta:     [n_tx] 等效增益对角元（实数）

        SINR 推导（论文公式 4–5）：
            W = (H^H H + σ²I)^{-1} H^H          MMSE 滤波矩阵
            G = W H                              等效增益矩阵
            β_i = G[i,i]                         第 i 流等效信道增益（实数）
            SINR_i = β_i / (1 - β_i)            论文公式 5
        """
        snr_linear = 10.0 ** (self.snr_db / 10.0)
        # 噪声方差 σ²：与 forward() 中注入的复高斯噪声一致。
        # 线性 ChannelEncoder 不再做逐 batch 功率归一化，等效 SNR 随编码器输出幅度变化。
        sigma2 = 1.0 / snr_linear
        eye = torch.eye(self.n_tx, dtype=torch.complex64, device=y.device)

        hh = h.conj().T @ h                              # [n_tx, n_tx]
        w = torch.linalg.inv(hh + sigma2 * eye) @ h.conj().T   # [n_tx, n_rx]
        x_hat = w @ y                                    # [n_tx]

        # 计算等效增益对角元素 β_i（论文公式 4）
        g = w @ h                                        # [n_tx, n_tx] = W·H
        beta = torch.diagonal(g).real.clamp(1e-3, 1 - 1e-3)   # 数值稳定性裁剪

        # SINR_i = β_i / (1 − β_i)（论文公式 5）
        sinr = beta / (1.0 - beta)
        sinr_mean = float(sinr.mean().item())

        return x_hat, sinr_mean, beta

    # ------------------------------------------------------------------
    # 模拟 JSCC 传输（主接口）
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, float, torch.Tensor]:
        """模拟 JSCC 传输：实数特征张量经 MIMO 信道后返回均衡结果、噪声估计与等效增益统计。

        Args:
            z: [B, C, H', W']，C 必须为偶数（相邻通道对 = 复数实/虚部）
        Returns:
            z_rx:    [B, C, H', W']，均衡后的实数特征张量
            sigma_y: 全局平均等效噪声水平（论文公式 14），
                     σ_y = 1/√(mean SINR)，供 DDNM+ 计算自适应 λ_t 使用
            beta_mean: [B]，每个样本在所有 MIMO 块上 MMSE 等效增益 β 的均值，
                     用于构造线性退化中的标量尺度（diag(β) 用均值近似为 β̄·I 时，A_lin ≈ β̄·W_dec W_enc）
        """
        B, C, Hp, Wp = z.shape
        assert C % 2 == 0, "通道数 C 必须为偶数以配对为复数符号"

        # 重组为复数：[B, C/2, H', W'] complex64
        z_complex = torch.complex(z[:, 0::2, :, :], z[:, 1::2, :, :])
        # 展平为 [B, N_sym] complex，N_sym = C/2 * H' * W'
        z_flat = z_complex.reshape(B, -1)
        n_sym = z_flat.shape[1]

        # 线性编码器无固定输出功率；噪声方差仍按 snr_db 相对单位方差噪声定义
        snr_linear = 10.0 ** (self.snr_db / 10.0)
        noise_var = 1.0 / snr_linear

        # 填充使 n_sym 能整除 n_tx
        pad = (self.n_tx - (n_sym % self.n_tx)) % self.n_tx
        if pad > 0:
            z_flat = torch.cat(
                [z_flat, torch.zeros(B, pad, dtype=torch.complex64, device=z.device)], dim=1
            )
        n_blocks = z_flat.shape[1] // self.n_tx

        rx_list: list[torch.Tensor] = []
        sinr_accum: list[float] = []
        beta_mean_list: list[torch.Tensor] = []

        for bi in range(B):
            sample = z_flat[bi].view(n_blocks, self.n_tx)
            detected: list[torch.Tensor] = []
            betas_blk: list[torch.Tensor] = []
            for blk in sample:
                h_r = torch.randn(self.n_rx, self.n_tx, device=z.device) / math.sqrt(2)
                h_i = torch.randn(self.n_rx, self.n_tx, device=z.device) / math.sqrt(2)
                h = (h_r + 1j * h_i).to(torch.complex64)

                n_r = torch.randn(self.n_rx, device=z.device) * math.sqrt(noise_var / 2)
                n_i = torch.randn(self.n_rx, device=z.device) * math.sqrt(noise_var / 2)
                noise = (n_r + 1j * n_i).to(torch.complex64)

                y = h @ blk + noise
                x_hat, sinr_mean, beta = self._mmse_detect(y, h)
                detected.append(x_hat)
                sinr_accum.append(sinr_mean)
                betas_blk.append(beta)

            recv = torch.cat(detected, dim=0)  # [n_blocks * n_tx]
            if pad > 0:
                recv = recv[:-pad]
            rx_list.append(recv.unsqueeze(0))
            beta_mean_list.append(torch.cat(betas_blk, dim=0).mean())

        z_rx_flat = torch.cat(rx_list, dim=0)  # [B, N_sym] complex

        # 还原为实数特征 [B, C, H', W']（奇数通道=实部，偶数通道=虚部）
        z_rx_complex = z_rx_flat.reshape(B, C // 2, Hp, Wp)
        out = torch.empty_like(z)
        out[:, 0::2, :, :] = z_rx_complex.real
        out[:, 1::2, :, :] = z_rx_complex.imag

        # σ_y = 1/√(mean SINR)  （论文公式 14）
        mean_sinr = sum(sinr_accum) / len(sinr_accum)
        sigma_y = 1.0 / math.sqrt(mean_sinr + 1e-8)

        beta_mean = torch.stack(beta_mean_list).to(dtype=out.dtype, device=out.device)
        return out, sigma_y, beta_mean
