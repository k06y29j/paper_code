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


@torch.jit.script
def create_window(window_size: int, sigma: float, channel: int):
    '''
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    '''
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


@torch.jit.script
def _gaussian_filter(x, window_1d, use_padding: bool):
    '''
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    '''
    C = x.shape[1]
    padding = 0
    if use_padding:
        window_size = window_1d.shape[3]
        padding = window_size // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out



@torch.jit.script
def ssim(X, Y, window, data_range: float, use_padding: bool = False):
    '''
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return:
    '''

    K1 = 0.01
    K2 = 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    # Fixed the issue that the negative value of cs_map caused ms_ssim to output Nan.
    cs_map = F.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW
    cs = cs_map.mean(dim=(1, 2, 3))

    return ssim_val, cs


@torch.jit.script
def ms_ssim(X, Y, window, data_range: float, weights, use_padding: bool = False, eps: float = 1e-8):
    '''
    interface of ms-ssim
    :param X: a batch of images, (N,C,H,W)
    :param Y: a batch of images, (N,C,H,W)
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param weights: weights for different levels
    :param use_padding: padding image before conv
    :param eps: use for avoid grad nan.
    :return:
    '''
    weights = weights[:, None]

    levels = weights.shape[0]
    vals = []
    for i in range(levels):
        ss, cs = ssim(X, Y, window=window, data_range=data_range, use_padding=use_padding)

        if i < levels - 1:
            vals.append(cs)
            X = F.avg_pool2d(X, kernel_size=2, stride=2, ceil_mode=True)
            Y = F.avg_pool2d(Y, kernel_size=2, stride=2, ceil_mode=True)
        else:
            vals.append(ss)

    vals = torch.stack(vals, dim=0)
    # Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
    vals = vals.clamp_min(eps)
    # The origin ms-ssim op.
    ms_ssim_val = torch.prod(vals[:-1] ** weights[:-1] * vals[-1:] ** weights[-1:], dim=0)
    # The new ms-ssim op. But I don't know which is best.
    # ms_ssim_val = torch.prod(vals ** weights, dim=0)
    # In this file's image training demo. I feel the old ms-ssim more better. So I keep use old ms-ssim op.
    return ms_ssim_val


class SSIM(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False):
        '''
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels (default: 3)
        :param use_padding: padding image before conv
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        self.data_range = data_range
        self.use_padding = use_padding

    @torch.jit.script_method
    def forward(self, X, Y):
        r = ssim(X, Y, window=self.window, data_range=self.data_range, use_padding=self.use_padding)
        return r[0]


class MS_SSIM(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding', 'eps']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=1.0, channel=3, use_padding=False, weights=None,
                 levels=None, eps=1e-8):
        """
        class for ms-ssim
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels
        :param use_padding: padding image before conv
        :param weights: weights for different levels. (default [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        :param levels: number of downsampling
        :param eps: Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
        """
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        self.data_range = data_range
        self.use_padding = use_padding
        self.eps = eps

        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.tensor(weights, dtype=torch.float)

        if levels is not None:
            weights = weights[:levels]
            weights = weights / weights.sum()

        self.register_buffer('weights', weights)

    @torch.jit.script_method
    def forward(self, X, Y):
        return 1 - ms_ssim(X, Y, window=self.window, data_range=self.data_range, weights=self.weights,
                       use_padding=self.use_padding, eps=self.eps)


class MSE(torch.nn.Module):
    def __init__(self, normalization=True):
        super(MSE, self).__init__()
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.normalization = normalization

    def forward(self, X, Y):
        # [-1 1] to [0 1]
        if self.normalization:
            X = (X + 1) / 2
            Y = (Y + 1) / 2
        return torch.mean(self.squared_difference(X * 255., Y * 255.))  # / 255.


class Distortion(torch.nn.Module):
    def __init__(self, args):
        super(Distortion, self).__init__()
        if args.distortion_metric == 'MSE':
            self.dist = MSE(normalization=False)
        elif args.distortion_metric == 'SSIM':
            self.dist = SSIM()
        elif args.distortion_metric == 'MS-SSIM':
            if args.trainset == 'CIFAR10':
                self.dist = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
            else:
                self.dist = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
        else:
            args.logger.info("Unknown distortion type!")
            raise ValueError

    def forward(self, X, Y, normalization=False):
        return self.dist.forward(X, Y).mean()  # / 255.

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
    use_vae: bool = False,
) -> torch.Tensor:
    """语义重建损失。

    默认配置偏向 Stage 1 的 PSNR 优化：以 MSE 为主，辅以较弱的 L1 正则。
    如需提升主观纹理观感，可显式开启感知损失。
    """
    loss = F.smooth_l1_loss(x_hat, x, beta=0.1)
    if use_vae:
        loss = loss + lambda_kl * kl_loss(mu, logvar)
    # loss = F.mse_loss(x_hat, x)
    # loss = loss + lambda_l1 * F.l1_loss(x_hat, x)
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


def min_snr_weighted_eps_loss(
    eps_pred: torch.Tensor,
    eps_target: torch.Tensor,
    alpha_bars: torch.Tensor,
    t_idx: torch.Tensor,
    gamma: float = 5.0,
) -> torch.Tensor:
    """min-SNR-γ 加权噪声预测损失（Hang et al. 2023, "Efficient Diffusion Training via Min-SNR Weighting Strategy"）。

    对于 ε-prediction 模型，权重为 w_t = min(SNR_t, γ) / SNR_t，其中 SNR_t = ᾱ_t / (1 - ᾱ_t)。
    相当于对低 SNR（高 t）放大、对高 SNR（低 t）压制，平衡不同时间步的梯度贡献。

    Args:
        eps_pred:   预测噪声 [B, C, H', W']
        eps_target: 真噪声 [B, C, H', W']
        alpha_bars: [T] 累积 ᾱ
        t_idx:      [B] 当前 batch 的整型时间步索引
        gamma:      截断阈值（推荐 5.0）
    """
    ab = alpha_bars[t_idx].to(dtype=eps_pred.dtype, device=eps_pred.device)
    snr = ab / (1.0 - ab).clamp(min=1e-8)
    w = torch.minimum(snr, torch.full_like(snr, float(gamma))) / snr.clamp(min=1e-8)
    per_sample = ((eps_pred - eps_target) ** 2).mean(dim=(1, 2, 3))
    return (w * per_sample).mean()


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
