"""三阶段训练损失函数。

阶段1 - 语义编解码器：L2 + L1 + 感知损失 + KL 散度（use_vae=True 时）
阶段2 - 信道编解码器：图像重建损失 + 特征空间重建损失
阶段3 - 扩散去噪器：噪声预测 MSE
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 感知损失（VGG 特征）
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """基于 VGG16 的感知损失（需要 torchvision）。
    若 torchvision 不可用，则退化为 L2 损失。
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            import torchvision.models as models

            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            # 取 relu2_2（第 9 层之前的特征）
            self.feature_extractor = nn.Sequential(*list(vgg.features)[:10]).eval()
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            self._available = True
        except Exception:
            self._available = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._available:
            return F.mse_loss(pred, target)
        # 若输入单通道，复制到 3 通道
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # 需要至少 32×32 的输入，插值保障
        if pred.shape[-1] < 32:
            pred = F.interpolate(pred, size=(32, 32), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(32, 32), mode="bilinear", align_corners=False)
        device = pred.device
        self.feature_extractor.to(device)
        feat_pred = self.feature_extractor(pred)
        feat_target = self.feature_extractor(target)
        return F.mse_loss(feat_pred, feat_target.detach())


# 全局单例（避免每次重复加载 VGG）
_vgg_loss: VGGPerceptualLoss | None = None


def _get_vgg_loss() -> VGGPerceptualLoss:
    global _vgg_loss
    if _vgg_loss is None:
        _vgg_loss = VGGPerceptualLoss()
    return _vgg_loss


# ---------------------------------------------------------------------------
# VAE KL 散度损失
# ---------------------------------------------------------------------------

def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """标准 VAE KL 散度：KL(q(z|x) ∥ N(0,I))。

    推导：-0.5 * Σ(1 + log σ² - μ² - σ²)，对批次和维度取均值。

    Args:
        mu:     编码器输出的均值 [B, C, H', W']
        logvar: 编码器输出的对数方差 [B, C, H', W']
    """
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


# ---------------------------------------------------------------------------
# 阶段 1：语义编解码器损失
# ---------------------------------------------------------------------------

def semantic_codec_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor | None = None,
    logvar: torch.Tensor | None = None,
    lambda_l1: float = 0.01,
    lambda_perc: float = 0.01,
    lambda_kl: float = 1e-4,
    use_perceptual: bool = False,
) -> torch.Tensor:
    """语义重建损失。

    默认配置偏向 Stage 1 的 PSNR 优化：以 MSE 为主，辅以较弱的 L1 正则。
    如需提升主观纹理观感，可显式开启感知损失。
    """
    loss = F.mse_loss(x_hat, x)
    loss = loss + lambda_l1 * F.l1_loss(x_hat, x)
    if use_perceptual:
        loss = loss + lambda_perc * _get_vgg_loss()(x_hat, x)
    if mu is not None and logvar is not None:
        loss = loss + lambda_kl * kl_loss(mu, logvar)
    return loss


# ---------------------------------------------------------------------------
# 阶段 2：信道编解码器损失
# ---------------------------------------------------------------------------

def channel_codec_loss(
    z_before_ch: torch.Tensor,
    z_after_ch: torch.Tensor,
    lambda_l1: float = 0.5,
) -> torch.Tensor:
    """信道损失 = L2(特征空间) + λ_l1 * L1(特征空间)。

    Args:
        z_before_ch: 信道编码前的语义特征 [B, C, H', W']
        z_after_ch:  信道解码后的重建特征 [B, C, H', W']
        lambda_l1: L1 损失权重
    """
    loss = F.mse_loss(z_after_ch, z_before_ch)
    loss = loss + lambda_l1 * F.l1_loss(z_after_ch, z_before_ch)
    return loss


# ---------------------------------------------------------------------------
# 阶段 3：扩散去噪损失
# ---------------------------------------------------------------------------

def ddnm_diffusion_loss(
    eps_pred: torch.Tensor,
    eps_target: torch.Tensor,
) -> torch.Tensor:
    """扩散模型（噪声预测）训练损失 = MSE(ε̂, ε)。

    Args:
        eps_pred:   模型预测噪声 [B, C, H', W']
        eps_target: 真实噪声 [B, C, H', W']
    """
    return F.mse_loss(eps_pred, eps_target)


def ddnm_reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    lambda_l1: float = 1.0,
    lambda_perc: float = 0.1,
    use_perceptual: bool = True,
) -> torch.Tensor:
    """DDNM 最终重建损失（图像空间）= L2 + L1 + 感知损失。

    Args:
        x_hat: DDNM 采样后重建图像 [B, C, H, W]
        x:     原始图像 [B, C, H, W]

    Note:
        当前 `train.py` 的 Stage 3 使用的是 `SemanticCommSystem.diffusion_loss()`
        中的噪声预测 MSE；本函数保留为后续做重建式微调/消融实验的可选工具。
    """
    loss = F.mse_loss(x_hat, x)
    loss = loss + lambda_l1 * F.l1_loss(x_hat, x)
    if use_perceptual:
        loss = loss + lambda_perc * _get_vgg_loss()(x_hat, x)
    return loss
