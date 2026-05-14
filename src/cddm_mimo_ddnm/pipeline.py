"""系统主流程编排。

完整链路（遵循 FRAMEWORK.md）：
  语义编码 → 信道编码 → MIMO 信道 + MMSE → 信道解码 → DDNM+ → 语义解码

DDNM+：U-Net 为**无条件**先验（仅见 x_t 与时间步）；观测 z_cond 仅经线性伪逆一致性修正进入迭代。
线性退化采用 1×1 信道矩阵复合，并以 MIMO MMSE 等效增益 β̄ 的标量近似注入 A_lin ≈ β̄·W_dec W_enc。

DDNM+ 推理默认在 cfg.diffusion.latent_std 归一化空间中进行；该值应从 Stage3 checkpoint
metadata 覆盖到 SystemConfig。
"""

import torch
import torch.nn as nn

from .config import SystemConfig
from .modules.channel_codec import ChannelDecoder, ChannelEncoder, composed_1x1_weight
from .modules.ddnm import UNetDenoiser
from .modules.mimo_channel import MIMOChannelMMSE
from .modules.semantic_codec import SemanticDecoder, SemanticEncoder
from .modules.siso_channel import SISOChannel
from CDDM.Diffusion.Model import UNetUncond


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
        self.cfg.validate()
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

        # Channel codec 工作在 SemanticEncoder.forward() 的语义瓶颈空间，即 cfg.semantic.embed_dim。
        sem_latent_c = cfg.channel_input_dim

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

        # 信道模型（模拟 JSCC）：按 cfg.mimo.mode 选择 SISO / MIMO
        # 两者 forward 接口一致：(z) -> (z_rx, sigma_y, beta_mean)，
        # 因此下游 DDNM+ 修正逻辑无需任何修改。
        mode = cfg.mimo.mode.lower()
        if mode == "mimo":
            self.mimo = MIMOChannelMMSE(
                n_tx=cfg.mimo.n_tx,
                n_rx=cfg.mimo.n_rx,
                snr_db=cfg.mimo.snr_db,
                fading=cfg.mimo.fading,
            )
        elif mode == "siso":
            self.mimo = SISOChannel(
                snr_db=cfg.mimo.snr_db,
                fading=cfg.mimo.fading,
            )
        else:
            raise ValueError(
                f"cfg.mimo.mode={cfg.mimo.mode!r} 非法，应为 'siso' 或 'mimo'。"
            )

        # 无条件 U-Net 先验（DDNM 理论：条件仅经线性修正，不拼接进网络）
        self.unet_denoiser = UNetUncond(
            T=cfg.unet_uncond.T,
            ch=cfg.unet_uncond.ch,
            ch_mult=cfg.unet_uncond.ch_mult,
            attn=cfg.unet_uncond.attn,
            num_res_blocks=cfg.unet_uncond.num_res_blocks,
            dropout=cfg.unet_uncond.dropout,
            input_channel=cfg.unet_uncond.input_channel,
        )

        # 扩散调度参数（仅 Stage 3 使用）
        # 必须满足 cfg.diffusion.num_train_steps == cfg.unet_uncond.T，
        # 否则 UNetUncond 内 nn.Embedding 的索引范围与 alpha_bars 长度不一致，会越界。
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

    def forward_sc_siso(
        self,
        x: torch.Tensor,
        *,
        snr_db: float | None = None,
        fading: str | None = None,
    ) -> torch.Tensor:
        """语义编码 → SISO 信道（AWGN / 瑞利，可调 SNR）→ 语义解码。

        不经信道编解码（ChannelEncoder/Decoder）与 DDNM+；用于在语义瓶颈上单独评估物理信道影响。
        ``snr_db`` / ``fading`` 缺省时沿用 ``cfg.mimo``。

        Args:
            x: 输入图像 ``[B, C_img, H, W]``。
            snr_db: 信噪比（dB）。
            fading: ``awgn`` 或 ``rayleigh``。

        Returns:
            重建图像 ``[B, C_img, H, W]``。

        Raises:
            ValueError: 语义瓶颈通道数为奇数时无法配对复数符号。
        """
        snr = float(self.cfg.mimo.snr_db if snr_db is None else snr_db)
        fad = (self.cfg.mimo.fading if fading is None else fading).lower()
        if fad not in ("awgn", "rayleigh"):
            raise ValueError(f"fading 须为 'awgn' 或 'rayleigh'，收到 {fad!r}")

        z_sem = self.semantic_encoder(x)
        if z_sem.shape[1] % 2 != 0:
            raise ValueError(
                f"语义瓶颈通道数 {z_sem.shape[1]} 须为偶数才能走 SISO 复数符号信道；"
                "请调整 SemanticConfig.embed_dim 等为偶数。"
            )

        ch = SISOChannel(snr_db=snr, fading=fad)
        z_rx, _sigma_y, _beta = ch.forward(z_sem)
        return self.semantic_decoder(z_rx)

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
        z_cd = self.channel_decoder(z_ch)
        x_hat = self.semantic_decoder(z_cd)
        return x_hat, z_sem, z_cd

    # ------------------------------------------------------------------
    # Stage 3：无条件扩散训练损失
    # ------------------------------------------------------------------

    def diffusion_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 3：在冻结语义潜空间上训练无条件 ε 预测（U-Net 不见 z_cd / 不经信道）。"""
        z_sem = self.semantic_encoder(x) / float(self.cfg.diffusion.latent_std)
        bsz = z_sem.shape[0]
        t_idx = torch.randint(
            0, self.alpha_bars.shape[0], (bsz,), device=x.device, dtype=torch.long
        )
        eps = torch.randn_like(z_sem)
        alpha_bar = self.alpha_bars[t_idx].view(-1, 1, 1, 1)
        z_t = torch.sqrt(alpha_bar) * z_sem + torch.sqrt(1 - alpha_bar) * eps
        # UNetUncond 使用 nn.Embedding 离散时间步，需传 [0, T-1] 的整型索引
        eps_pred = self.unet_denoiser(z_t, t_idx)
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

        z_refined = self.ddnm_sample_normalized(
            z_cd,
            beta=beta,
            sigma_y=sigma_y,
            latent_std=float(self.cfg.diffusion.latent_std),
            num_steps=self.cfg.diffusion.num_sample_steps,
            t_start=self.cfg.diffusion.ddnm_t_start,
            anchor=self.cfg.diffusion.ddnm_anchor,
            blend=self.cfg.diffusion.ddnm_blend,
            repeat_per_step=self.cfg.diffusion.ddnm_repeat_per_step,
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
        step_indices = torch.linspace(n_total - 1, 0, num_steps, device=z.device).long()

        a0 = self._semantic_linear_chain_matrix().to(device=z.device, dtype=z.dtype)

        for i, idx in enumerate(step_indices):
            t_emb = torch.full((b,), idx.item(), device=z.device, dtype=torch.long)
            eps_pred = self.unet_denoiser(z, t_emb)

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
    def ddnm_sample_normalized(
        self,
        z_cond: torch.Tensor,
        beta: torch.Tensor,
        sigma_y: float,
        latent_std: float,
        num_steps: int,
        t_start: int,
        anchor: str = "zcd",
        blend: float = 1.0,
        repeat_per_step: int = 1,
    ) -> torch.Tensor:
        """潜空间 DDNM+（warm-start + 一致性修正 + 锚点融合，归一化空间迭代）。

        与 ``_ddnm_sample`` 的区别（也是经实测可严格胜过无-DDNM 基线的关键三点）：

          1. **归一化空间迭代** —— 无条件 U-Net 在 ``z / latent_std`` 空间以 min-SNR-γ 加权
             训练（见 ``train/train_unet_un.py``）。DDNM 迭代必须在归一化空间进行：
             先 ``z_cond_norm = z_cond / latent_std``，反向迭代结束后再乘回 ``latent_std``
             送给语义解码器。
          2. **Warm-start anchor**（``anchor`` 参数） —— 起始 ``z_{t_start}`` 由「锚点估计」
             前向加噪得到：``z = √α̅_{t_start} · u_anchor + √(1-α̅_{t_start}) · ε``。

               - ``"zcd"``（默认）：用 ``z_cond_norm``。``t_start=0`` 退化为「无-DDNM 直接
                 送 SC 解码器」基线，保证最差不低于该基线；``t_start>0`` 时让 U-Net 在
                 小 t 域做小幅扩散精修，并配合每步 DDNM 线性修正。
               - ``"pinv"``：用最小二乘 ``A⁺ · z_cond_norm`` —— 倾向丢掉 cc 解码器在
                 正交补里学到的有用信息；仅在极低压缩率下偶有微弱优势。
               - ``"zero"``：用 0 张量；趋近经典 DDNM+ 在 ``t_start=T-1`` 的 ε~N(0,I)，
                 实测在本 ckpt 上数值不稳定。

          3. **谐和重复 / time-travel**（``repeat_per_step``） —— 在每个 t 重复
             (U-Net + 线性修正) 多次；第 r 次后将 z 重新加噪到该 t（``z = √α̅ · z0_corr +
             √(1-α̅) · ε_new``），让 U-Net 在一致性约束下做多次去噪迭代。实测 ``r=3`` 在
             所有 (压缩率, 衰落, SNR) 上均胜过 baseline，高压缩 + 低 SNR 下提升尤为显著。

        其它：

          - ``blend`` ∈ [0,1]：最终 = blend·z_DDNM + (1-blend)·z_cond_norm；blend<1 可与
            「无-DDNM」基线融合做保险。
          - 自适应 ``λ_t``：σ_t ≥ a_t·σ_y_norm 时 λ_t=1.0（充分一致性）；否则
            λ_t = σ_t / (a_t · σ_y_norm)（小 t 时弱化修正，避免放大观测噪声）。
            ``σ_y_norm = sigma_y / latent_std``。

        Args:
            z_cond:     [B, C, H, W] 信道解码观测（未归一化，按 cfg 的 sem 潜空间尺度）。
            beta:       [B] MIMO MMSE 等效增益均值（AWGN 时恒为 1）。
            sigma_y:    信道符号空间等效噪声 std（``SISOChannel.forward`` 的第二返回值）。
            latent_std: U-Net 训练时所用的 LDM scaling（一般来自 ``unet_ckpt['latent_std']``）。
            num_steps:  反向 DDIM 步数（仅作用于 ``[t_start, 0]`` 区间）。
            t_start:    反向起始时间步 ``∈ [0, T-1]``；=0 时直接返回 ``anchor*latent_std``。
            anchor:     ``"zcd"`` / ``"pinv"`` / ``"zero"``，默认 ``"zcd"``。
            blend:      最终融合系数，默认 1.0（全用 DDNM）。
            repeat_per_step: 每个 t 重复次数，默认 1（不重复）；推荐 3。

        Returns:
            z_refined:  [B, C, H, W]，**已乘回 latent_std**，可直接送 ``semantic_decoder``。
        """
        device = z_cond.device
        z_cond_norm = z_cond / latent_std
        b, c, h, w = z_cond_norm.shape

        a0 = self._semantic_linear_chain_matrix().to(device=device, dtype=z_cond_norm.dtype)
        a_lin = beta.to(device=device, dtype=a0.dtype).clamp(min=1e-6).view(b, 1, 1) * a0.unsqueeze(0)
        a_pinv = torch.linalg.pinv(a_lin)

        z_cond_flat = z_cond_norm.view(b, c, -1).permute(0, 2, 1).contiguous()

        if anchor == "pinv":
            u_anchor_flat = torch.bmm(z_cond_flat, a_pinv.transpose(-2, -1))
            u_anchor = u_anchor_flat.permute(0, 2, 1).reshape(b, c, h, w)
        elif anchor == "zcd":
            u_anchor = z_cond_norm
        elif anchor == "zero":
            u_anchor = torch.zeros_like(z_cond_norm)
        else:
            raise ValueError(f"非法 anchor={anchor!r}（仅支持 zcd/pinv/zero）")

        if t_start <= 0:
            z_final_norm = u_anchor
        else:
            alpha_bars = self.alpha_bars.to(device=device, dtype=z_cond_norm.dtype)
            n_total = int(alpha_bars.shape[0])
            t_start = max(0, min(int(t_start), n_total - 1))

            eps_init = torch.randn_like(u_anchor)
            ab_t0 = alpha_bars[t_start]
            z = ab_t0.sqrt() * u_anchor + (1.0 - ab_t0).sqrt() * eps_init

            step_indices = torch.linspace(t_start, 0, num_steps, device=device).long()
            sigma_y_norm = float(sigma_y) / max(float(latent_std), 1e-8)

            for i, idx in enumerate(step_indices):
                alpha_bar = alpha_bars[idx]
                alpha_bar_prev = (
                    alpha_bars[step_indices[i + 1]]
                    if i + 1 < len(step_indices)
                    else torch.tensor(1.0, device=device, dtype=z.dtype)
                )
                a_t = alpha_bar.sqrt()
                sigma_t = (1.0 - alpha_bar).sqrt()
                threshold = float((a_t * sigma_y_norm).item())
                if float(sigma_t.item()) >= threshold:
                    lambda_t = 1.0
                else:
                    lambda_t = float(sigma_t.item()) / (threshold + 1e-8)

                for _r in range(max(1, int(repeat_per_step))):
                    t_emb = torch.full((b,), int(idx.item()), device=device, dtype=torch.long)
                    eps_pred = self.unet_denoiser(z, t_emb)
                    z0_pred = (z - torch.sqrt(1.0 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar + 1e-8)

                    u_flat = z0_pred.view(b, c, -1).permute(0, 2, 1).contiguous()
                    u_flat = self._linear_ddnm_correct_batch(
                        u_flat, z_cond_flat, beta.to(device=device, dtype=z.dtype), a0, lambda_t
                    )
                    z0_corr = u_flat.permute(0, 2, 1).reshape(b, c, h, w)

                    z = torch.sqrt(alpha_bar_prev) * z0_corr + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred

                    if _r + 1 < max(1, int(repeat_per_step)):
                        eps_new = torch.randn_like(z)
                        z = a_t * z0_corr + sigma_t * eps_new

            z_final_norm = z

        if blend < 1.0:
            z_final_norm = float(blend) * z_final_norm + (1.0 - float(blend)) * z_cond_norm

        return z_final_norm * latent_std

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
        step_indices = torch.linspace(n_total - 1, 0, num_steps, device=dev).long()
        for i, idx in enumerate(step_indices):
            t_emb = torch.full((batch_size,), idx.item(), device=dev, dtype=torch.long)
            eps_pred = self.unet_denoiser(z, t_emb)
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
        z = z * float(self.cfg.diffusion.latent_std)
        return self.semantic_decoder(z).clamp(0, 1)
