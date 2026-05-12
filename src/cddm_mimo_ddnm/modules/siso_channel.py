"""SISO 信道仿真（支持 AWGN / Rayleigh），含功率归一化与 MMSE 均衡。

传输流程（与用户思路一致；接收端补一步 MMSE 让 z_rx 可被 ChannelDecoder 利用）：
  1. 功率归一化：x_norm = z / √pwr  ，使 E[|x_norm|²] = 1
                 √pwr 即"归一化系数"，形状 [B]，逐样本独立计算
  2. 经过信道（按 fading 选择）：
       - AWGN     : y = x_norm + n           ，n ~ CN(0, σ²)
       - Rayleigh : y = h · x_norm + n       ，h ~ CN(0, 1) 逐符号独立衰落
     其中 σ² = 1 / 10^(SNR_dB/10) （相对单位功率符号定义）
  3. 接收端均衡（AWGN 直通；Rayleigh 用 MMSE）：
       x̂ = h*·y / (|h|² + σ²)
  4. 反归一化：z_rx = √pwr · x̂   （即"乘以归一化系数"，回到输入的功率尺度）

接口与 ``MIMOChannelMMSE.forward`` 对齐，便于在 ``pipeline.py`` 中互换：
  z_rx     : [B, C, H', W']   反归一化后的实数特征张量（与输入同形）
  sigma_y  : float            全局等效噪声标准差 σ_y = 1/√(mean SINR)，供 DDNM+ 计算 λ_t
  beta_mean: [B]              每样本所有符号位置 MMSE 等效增益 β=|h|²/(|h|²+σ²) 的均值
                              线性退化矩阵 A_lin ≈ β̄ · W_dec W_enc 中的标量。
"""

from __future__ import annotations

import math

import torch


class SISOChannel:
    """SISO 信道：AWGN / Rayleigh，含功率归一化与 MMSE 均衡（端到端可微）。

    输入/输出均为实数特征张量 [B, C, H, W]，C 必须为偶数：相邻通道对组成
    复数符号 (实部, 虚部)；每个空间位置一个独立的复数信道符号位。
    """

    def __init__(
        self,
        snr_db: float = 10.0,
        fading: str = "rayleigh",
    ) -> None:
        if fading not in ("awgn", "rayleigh"):
            raise ValueError(f"fading 必须为 'awgn' 或 'rayleigh'，收到 {fading!r}")
        self.snr_db = snr_db
        self.fading = fading

    # ------------------------------------------------------------------
    # 功率归一化（步骤 1）
    # ------------------------------------------------------------------

    @staticmethod
    def _power_normalize(
        z_complex: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """逐样本将复数符号平均功率归一化为 1。

        Args:
            z_complex: [B, C/2, H', W']，复数张量
        Returns:
            x_norm: [B, C/2, H', W']  归一化后符号（E[|x|²]=1）
            scale : [B]               归一化系数 √pwr，反归一化时再乘回
        """
        dims = tuple(range(1, z_complex.ndim))
        pwr = (z_complex.real ** 2 + z_complex.imag ** 2).mean(dim=dims)  # [B]
        scale = torch.sqrt(pwr.clamp_min(1e-12))                          # [B]
        scale_b = scale.view(-1, *([1] * (z_complex.ndim - 1)))           # 广播
        x_norm = z_complex / scale_b.to(z_complex.dtype)
        return x_norm, scale

    # ------------------------------------------------------------------
    # 主前向（归一化 → 信道 → 均衡 → 反归一化）
    # ------------------------------------------------------------------

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        """SISO 端到端信道仿真。

        Args:
            z: [B, C, H', W']，C 偶数（相邻通道对 = 复数符号实/虚部）
        Returns:
            z_rx:      [B, C, H', W']  反归一化后的实数特征张量
            sigma_y:   等效噪声标准差，σ_y = 1/√(mean SINR)
            beta_mean: [B]  每样本所有符号位置 MMSE 等效增益均值
                       AWGN 时恒为 1（无衰落）。
        """
        B, C, _Hp, _Wp = z.shape
        assert C % 2 == 0, "通道数 C 必须为偶数以配对复数符号"

        orig_dtype = z.dtype
        # torch.complex 仅支持 float16/float32/float64 的实部/虚部；AMP 下常为 bfloat16
        z_use = z.float() if orig_dtype == torch.bfloat16 else z

        # 实 → 复：偶数通道作实部、奇数通道作虚部
        z_complex = torch.complex(z_use[:, 0::2, :, :], z_use[:, 1::2, :, :])

        # ---- 1. 功率归一化（功率 → 1，除以归一化系数） ----
        x_norm, scale = self._power_normalize(z_complex)

        # ---- 2. 信道噪声（按 SNR 定义在单位功率符号上） ----
        snr_linear = 10.0 ** (self.snr_db / 10.0)
        sigma2 = 1.0 / snr_linear           # 复噪声总方差
        sigma = math.sqrt(sigma2 / 2.0)     # 实/虚部各一半
        n_r = torch.randn_like(x_norm.real) * sigma
        n_i = torch.randn_like(x_norm.imag) * sigma
        noise = torch.complex(n_r, n_i)

        if self.fading == "awgn":
            y = x_norm + noise
            x_hat = y                                                   # AWGN 无需均衡
            beta = torch.ones(B, device=z.device, dtype=z_use.dtype)
            mean_sinr = snr_linear                                      # 单位功率 + 高斯噪声
        else:  # rayleigh：h ~ CN(0, 1) 逐符号独立
            h_r = torch.randn_like(x_norm.real) / math.sqrt(2.0)
            h_i = torch.randn_like(x_norm.imag) / math.sqrt(2.0)
            h = torch.complex(h_r, h_i)

            y = h * x_norm + noise

            # MMSE 均衡：x̂ = h*·y / (|h|² + σ²)
            h_abs2 = h.real ** 2 + h.imag ** 2
            x_hat = h.conj() * y / (h_abs2 + sigma2)

            # 等效增益与 SINR（推导：SINR_MMSE = β/(1-β) = |h|²/σ²）
            beta_map = (h_abs2 / (h_abs2 + sigma2)).clamp(1e-3, 1 - 1e-3)
            sinr_map = beta_map / (1.0 - beta_map)
            beta = beta_map.mean(
                dim=tuple(range(1, beta_map.ndim))
            ).to(dtype=z_use.dtype)
            mean_sinr = float(sinr_map.mean().item())

        # ---- 3. 反归一化（乘以归一化系数 √pwr，回到输入功率尺度） ----
        scale_b = scale.view(-1, *([1] * (x_hat.ndim - 1))).to(x_hat.dtype)
        x_hat = x_hat * scale_b

        # 复 → 实：还原为 [B, C, H', W']
        out = torch.empty_like(z_use)
        out[:, 0::2, :, :] = x_hat.real
        out[:, 1::2, :, :] = x_hat.imag

        if orig_dtype == torch.bfloat16:
            out = out.to(orig_dtype)
            beta = beta.to(orig_dtype)

        sigma_y = 1.0 / math.sqrt(mean_sinr + 1e-8)
        return out, sigma_y, beta
