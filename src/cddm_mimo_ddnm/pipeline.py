"""系统主流程编排。

完整链路（遵循 FRAMEWORK.md）：
  语义编码 → 信道编码 → MIMO 信道 + MMSE → 信道解码 → DDNM+ → 语义解码

DDNM+：U-Net 为**无条件**先验（仅见 x_t 与时间步）；观测 z_cond 仅经线性伪逆一致性修正进入迭代。
线性退化采用 1×1 信道矩阵复合，并以 MIMO MMSE 等效增益 β̄ 的标量近似注入 A_lin ≈ β̄·W_dec W_enc。

CBR = 1/12（patch_size=4，每空间位置 4 个复数 = 8 个实数通道）。
"""

import torch
import torch.nn as nn

from .config import SystemConfig
from .modules.channel_codec import ChannelDecoder, ChannelEncoder, composed_1x1_weight
from .modules.ddnm import UNetDenoiser
from .modules.mimo_channel import MIMOChannelMMSE
from .modules.semantic_codec import SemanticDecoder, SemanticEncoder


class SemanticCommSystem(nn.Module):
    """端到端语义通信系统。

    三阶段训练：
      Stage 1 - 仅训练语义编解码器（SemanticEncoder + SemanticDecoder）
      Stage 2 - 冻结语义编解码器，训练信道编解码器（ChannelEncoder + ChannelDecoder）
      Stage 3 - 冻结前两阶段，训练**无条件** U-Net 扩散先验（不经信道、不注入 z_cond）
    """

    def __init__(self, cfg: SystemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        sc = cfg.semantic
        cc = cfg.channel

        # 语义编解码器
        self.semantic_encoder = SemanticEncoder(
            in_channels=sc.image_channels,
            embed_dim=sc.embed_dim,
            patch_size=sc.patch_size,
            num_heads=sc.num_heads,
            window_size=sc.window_size,
            num_blocks=sc.num_swin_blocks,
            stage_embed_dims=sc.stage_embed_dims,
            stage_depths=sc.stage_depths,
            stage_num_heads=sc.stage_num_heads,
            stem_stride=sc.stem_stride,
            stage_downsample=sc.stage_downsample,
            use_vae=sc.use_vae,
        )
        self.semantic_decoder = SemanticDecoder(
            out_channels=sc.image_channels,
            embed_dim=sc.embed_dim,
            patch_size=sc.patch_size,
            num_heads=sc.num_heads,
            window_size=sc.window_size,
            num_refine_blocks=sc.num_decoder_refine_blocks,
            stage_embed_dims=sc.stage_embed_dims,
            stage_depths=sc.stage_depths,
            stage_num_heads=sc.stage_num_heads,
            stem_stride=sc.stem_stride,
            stage_downsample=sc.stage_downsample,
        )

        # 语义特征通道数 = 编码器最后一级宽度（如 DIV2K 为 320 @ H/16×W/16），与 cfg.semantic.embed_dim 可不同
        sem_latent_c = self.semantic_encoder.latent_dim
        if cc.channel_symbols % 2 != 0:
            raise ValueError(
                f"channel_symbols={cc.channel_symbols} 须为偶数，以便 MIMO 将相邻实数通道配对为复数符号。"
            )

        self.channel_encoder = ChannelEncoder(
            in_channels=sem_latent_c,
            out_channels=cc.channel_symbols,
            bottleneck_dim=cc.channel_bottleneck_dim,
        )
        self.channel_decoder = ChannelDecoder(
            in_channels=cc.channel_symbols,
            out_channels=sem_latent_c,
            bottleneck_dim=cc.channel_bottleneck_dim,
        )

        # MIMO 信道（模拟 JSCC）
        self.mimo = MIMOChannelMMSE(
            n_tx=cfg.mimo.n_tx,
            n_rx=cfg.mimo.n_rx,
            snr_db=cfg.mimo.snr_db,
            fading=cfg.mimo.fading,
        )

        # 无条件 U-Net 先验（DDNM 理论：条件仅经线性修正，不拼接进网络）
        self.unet_denoiser = UNetDenoiser(
            channels=sem_latent_c,
            hidden_dim=cfg.diffusion.unet_hidden_dim,
            use_cond=False,
        )

        # 扩散调度参数（仅 Stage 3 使用）
        num_steps = cfg.diffusion.num_train_steps
        betas = torch.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end, num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    # ------------------------------------------------------------------
    # 阶段控制：冻结/解冻参数
    # ------------------------------------------------------------------

    def freeze_stage1(self) -> None:
        """冻结语义编解码器（Stage 2 开始时调用）。"""
        for p in self.semantic_encoder.parameters():
            p.requires_grad = False
        for p in self.semantic_decoder.parameters():
            p.requires_grad = False

    def freeze_stage2(self) -> None:
        """冻结信道编解码器（Stage 3 开始时调用）。"""
        for p in self.channel_encoder.parameters():
            p.requires_grad = False
        for p in self.channel_decoder.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Stage 1：仅语义编解码器（无信道噪声）
    # ------------------------------------------------------------------

    def forward_stage1(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """语义编解码前向：x → (z_sem, μ, log σ²) → x̂（跳过信道与 DDNM）。

        Returns:
            x_hat:  重建图像
            mu:     VAE 均值（use_vae=False 时为 None）
            logvar: VAE 对数方差（use_vae=False 时为 None）
        """
        z_sem, mu, logvar = self.semantic_encoder.encode(x, sample=self.training)
        x_hat = self.semantic_decoder(z_sem)
        return x_hat, mu, logvar

    # ------------------------------------------------------------------
    # Stage 2：语义 + 信道编解码器
    # ------------------------------------------------------------------

    def forward_stage2(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """信道编解码前向，返回 (x̂, z_sem, z_cd)。

        z_sem: 信道编码前的语义特征（作为 Stage2 损失的 target）
        z_cd:  信道解码后的重建特征
        """
        z_sem = self.semantic_encoder(x)
        z_ch = self.channel_encoder(z_sem)
        z_rx, _, _ = self.mimo.forward(z_ch)
        z_cd = self.channel_decoder(z_rx)
        x_hat = self.semantic_decoder(z_cd)
        return x_hat, z_sem, z_cd

    # ------------------------------------------------------------------
    # Stage 3：无条件扩散训练损失
    # ------------------------------------------------------------------

    def diffusion_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 3：在冻结语义潜空间上训练无条件 ε 预测（U-Net 不见 z_cd / 不经信道）。"""
        z_sem = self.semantic_encoder(x)
        bsz = z_sem.shape[0]
        t_idx = torch.randint(0, self.alpha_bars.shape[0], (bsz,), device=x.device)
        t_norm = t_idx.float() / (self.alpha_bars.shape[0] - 1)
        eps = torch.randn_like(z_sem)
        alpha_bar = self.alpha_bars[t_idx].view(-1, 1, 1, 1)
        z_t = torch.sqrt(alpha_bar) * z_sem + torch.sqrt(1 - alpha_bar) * eps
        eps_pred = self.unet_denoiser(z_t, t_norm)
        return torch.mean((eps_pred - eps) ** 2)

    # ------------------------------------------------------------------
    # 推理（采样）：全链路 + 线性伪逆 DDNM+ 修正
    # ------------------------------------------------------------------

    def _semantic_linear_chain_matrix(self) -> torch.Tensor:
        """A0 = W_dec @ W_enc，形状 [C_sem, C_sem]；与 1×1 信道及 MIMO 标量 β̄ 组合得 A_lin ≈ β̄·A0。"""
        w_enc = composed_1x1_weight(self.channel_encoder.net)
        w_dec = composed_1x1_weight(self.channel_decoder.net)
        return w_dec @ w_enc

    @staticmethod
    def _linear_ddnm_correct_batch(
        u_flat: torch.Tensor,
        z_obs_flat: torch.Tensor,
        beta_scale: torch.Tensor,
        a0: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """u ← u + λ · (r @ A^†^T)，r = z_obs − u A_lin^T，A_lin = diag_scl · A0，批量 pinv。"""
        bsz = u_flat.shape[0]
        s = beta_scale.clamp(min=1e-6).to(dtype=a0.dtype, device=a0.device).view(bsz, 1, 1)
        a_lin = s * a0.unsqueeze(0)
        a_pinv = torch.linalg.pinv(a_lin)
        pred = torch.bmm(u_flat, a_lin.transpose(-2, -1))
        r = z_obs_flat - pred
        corr = torch.bmm(r, a_pinv.transpose(-2, -1))
        return u_flat + lam * corr

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """推理前向：语义编码 → 信道 → 信道解码 → DDNM+（无条件 U-Net + 线性修正）→ 语义解码。"""
        z_sem = self.semantic_encoder(x)
        z_ch = self.channel_encoder(z_sem)
        z_rx, sigma_y, beta = self.mimo.forward(z_ch)
        z_cd = self.channel_decoder(z_rx)

        z_refined = self._ddnm_sample(
            z_cd,
            beta_mean=beta,
            num_steps=self.cfg.diffusion.num_sample_steps,
            sigma_y=sigma_y,
        )
        return self.semantic_decoder(z_refined)

    def _ddnm_sample(
        self,
        z_cond: torch.Tensor,
        beta_mean: torch.Tensor,
        num_steps: int = 50,
        sigma_y: float = 0.5,
    ) -> torch.Tensor:
        """潜空间 DDNM+：无条件 U-Net 预测 ε；数据一致性用 A_lin 的批量 Moore–Penrose 伪逆修正。

        Args:
            z_cond:    信道解码观测 f̂，与 z_sem 同形状 [B,C,H,W]
            beta_mean: [B]，MIMO MMSE 等效增益在块上的均值，用于 A_lin ≈ β̄·W_dec W_enc
        """
        z = torch.randn_like(z_cond)
        b, c, h, w = z.shape
        n_total = self.alpha_bars.shape[0]
        denom = max(n_total - 1, 1)
        step_indices = torch.linspace(n_total - 1, 0, num_steps, device=z.device).long()

        a0 = self._semantic_linear_chain_matrix().to(device=z.device, dtype=z.dtype)

        for i, idx in enumerate(step_indices):
            t_norm = torch.full((b,), float(idx.item()) / denom, device=z.device, dtype=z.dtype)
            eps_pred = self.unet_denoiser(z, t_norm)

            alpha_bar = self.alpha_bars[idx]
            if i + 1 < len(step_indices):
                alpha_bar_prev = self.alpha_bars[step_indices[i + 1]]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=z.device, dtype=z.dtype)

            z0_pred = (z - torch.sqrt(1.0 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar + 1e-8)

            a_t = alpha_bar.sqrt()
            sigma_t = (1.0 - alpha_bar).sqrt()
            threshold = float((a_t * sigma_y).item())
            if float(sigma_t.item()) >= threshold:
                lambda_t = 1.0
            else:
                lambda_t = float(sigma_t.item()) / (threshold + 1e-8)

            u_flat = z0_pred.view(b, c, -1).permute(0, 2, 1).contiguous()
            z_flat = z_cond.view(b, c, -1).permute(0, 2, 1).contiguous()
            u_flat = self._linear_ddnm_correct_batch(
                u_flat, z_flat, beta_mean.to(device=z.device, dtype=z.dtype), a0, lambda_t
            )
            z0_pred = u_flat.permute(0, 2, 1).reshape(b, c, h, w)

            z = torch.sqrt(alpha_bar_prev) * z0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred

        return z

    @torch.no_grad()
    def _ddim_sample_unconditional(
        self,
        batch_size: int,
        latent_shape: tuple[int, int, int],
        num_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """DDIM (η=0) 从标准高斯采样潜变量；latent_shape = (C, H', W')。"""
        dev = device or self.alpha_bars.device
        if num_steps is None:
            num_steps = int(self.cfg.diffusion.num_sample_steps)
        z = torch.randn(batch_size, *latent_shape, device=dev, dtype=self.alpha_bars.dtype)
        n_total = int(self.alpha_bars.shape[0])
        denom = max(n_total - 1, 1)
        step_indices = torch.linspace(n_total - 1, 0, num_steps, device=dev).long()
        for i, idx in enumerate(step_indices):
            t_norm = torch.full((batch_size,), float(idx.item()) / denom, device=dev)
            eps_pred = self.unet_denoiser(z, t_norm)
            alpha_bar = self.alpha_bars[idx]
            if i + 1 < len(step_indices):
                alpha_bar_prev = self.alpha_bars[step_indices[i + 1]]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=dev, dtype=z.dtype)
            z0_pred = (z - torch.sqrt(1.0 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar + 1e-8)
            z = torch.sqrt(alpha_bar_prev) * z0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
        return z

    @torch.no_grad()
    def sample_images_unconditional(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """纯无条件生成：DDIM 采样潜变量后语义解码（无信道观测）。"""
        self.eval()
        device = next(self.parameters()).device
        x0 = torch.zeros(
            1,
            self.cfg.semantic.image_channels,
            image_height,
            image_width,
            device=device,
            dtype=self.alpha_bars.dtype,
        )
        z_ref = self.semantic_encoder(x0)
        _, c, h, w = z_ref.shape
        z = self._ddim_sample_unconditional(
            batch_size, (c, h, w), num_steps=num_steps, device=device
        )
        return self.semantic_decoder(z).clamp(0, 1)
