from dataclasses import dataclass, field


@dataclass
class SemanticConfig:
    image_channels: int = 3
    # Swin Transformer 参数
    # 注意：这里的 patch_size 表示编码器的“总下采样倍数”，
    # 用来对齐原项目中 patch stem + 多个 PatchMerging 后的最终空间分辨率。
    patch_size: int = 16
    embed_dim: int = 16       # 本项目语义瓶颈维度（与原项目单段式 JSCC 的发送维度 C 不同）
    num_heads: int = 4        # 多头注意力头数
    window_size: int = 4      # Swin 窗口大小
    num_swin_blocks: int = 2  # Swin Block 数量
    # 分层 Swin-JSCC 配置；为 None 时退化为旧版单尺度结构
    stage_embed_dims: tuple[int, ...] | None = None
    stage_depths: tuple[int, ...] | None = None
    stage_num_heads: tuple[int, ...] | None = None
    stem_stride: int | None = None
    stage_downsample: tuple[bool, ...] | None = None
    # VAE 潜空间正则化（为扩散模型提供规则的高斯潜空间）
    use_vae: bool = True
    lambda_kl: float = 1e-4   # KL 散度权重
    # 解码器深度（PixelShuffle 后的 SwinBlock 数量）
    num_decoder_refine_blocks: int = 2


@dataclass
class ChannelConfig:
    # -----------------------------------------------------------------------
    # CBR 定义说明（重要：代码定义与论文公式不同，务必区分）
    # -----------------------------------------------------------------------
    # 【代码定义】CBR_code = k / (H × W × image_channels)
    #   k = 复数符号总数
    #   e.g. patch_size=4, channel_symbols=8 实数/位置 → k = 4×(H/4)×(W/4) = H×W/4
    #        CBR_code = (H×W/4) / (H×W×3) = 1/12     ← 本代码采用此定义
    #
    # 【论文定义】ρ = 2k / (H × W × 3)   （论文公式 2：每个复数符号计 2 个实数信道用量）
    #   相同系统下 ρ_paper = 2 × CBR_code = 1/6
    #
    # ⚠️  与其他工作（如 DeepJSCC 系列）对比时，需确认使用同一定义。
    #    当前系统配置 channel_symbols=8 对应论文 CBR ρ = 1/6。
    cbr: float = 1 / 12       # 代码口径：k/n_pixels（复数符号数 / 像素数）
    channel_symbols: int = 8   # 每个空间位置的实数通道数（4个复数=8实数）；须为偶数以配对复数
    # 非 None 时：信道 MLP 先压至该瓶颈维（如 320→64 为 5×），再扩到 channel_symbols 以满足 CBR
    channel_bottleneck_dim: int | None = None


@dataclass
class MIMOConfig:
    # 信道模式开关：
    #   "mimo" → MIMOChannelMMSE  使用 n_tx, n_rx, snr_db, fading
    #   "siso" → SISOChannel       仅使用 snr_db, fading（n_tx / n_rx 忽略）
    mode: str = "siso"
    n_tx: int = 2
    n_rx: int = 2
    snr_db: float = 12.0
    fading: str = "awgn"


@dataclass
class DiffusionConfig:
    num_train_steps: int = 1000
    num_sample_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    unet_hidden_dim: int = 128


@dataclass
class UNetUncondConfig:
    T: int = 1000
    ch: int = 256
    ch_mult: tuple[int, ...] = (1, 2, 2)
    attn: tuple[int, ...] = (1,)
    num_res_blocks: int = 2
    dropout: float = 0.1
    input_channel: int = 16

@dataclass
class SystemConfig:
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    mimo: MIMOConfig = field(default_factory=MIMOConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    unet_uncond: UNetUncondConfig = field(default_factory=UNetUncondConfig)

# ---------------------------------------------------------------------------
# 预设配置工厂
# ---------------------------------------------------------------------------

def get_cifar10_config() -> SystemConfig:
    """CIFAR-10 预设。

    严格对齐原项目 `CDDM/main.py` 中的 Swin-JSCC stage 配置：
      - encoder: patch stem = 2, stage dims = [128, 256], depths = [2, 4], heads = [4, 8], window = 2
      - decoder: stage dims = [256, 128], depths = [4, 2], heads = [8, 4], window = 2

    在当前实现里，`patch_size=4` 表示等效总下采样倍数：
      stem_stride=2 + 一次 stage 下采样 => H/4, W/4
    """
    return SystemConfig(
        semantic=SemanticConfig(
            image_channels=3,
            patch_size=4,
            embed_dim=24,
            num_heads=4,
            window_size=2,
            num_swin_blocks=2,
            stage_embed_dims=(128, 256),
            stage_depths=(2, 4),
            stage_num_heads=(4, 8),
            stem_stride=2,
            stage_downsample=(False, True),
            use_vae=False,
            lambda_kl=1e-4,
            num_decoder_refine_blocks=2,
        ),
    )


def get_div2k_config() -> SystemConfig:
    """DIV2K 预设。

    严格对齐 CDDM 原项目：
    https://github.com/Wireless3C-SJTU/CDDM-channel-denoising-diffusion-model-for-semantic-communication

    语义编解码器（CDDM/main.py config.DIV2K）：
      - encoder: patch_size=2, embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2],
                 num_heads=[4, 6, 8, 10], window_size=8, patch_norm=True
      - decoder: embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4]
      - 编码器输出特征维 320 @ H/16×W/16（与 stage 最后一维一致）

    信道与扩散：
      - 语义特征 [320,16,16]（256×256 输入）经信道 MLP：瓶颈 64（相对 320 压缩 5×），
        再映射为每位置 128 个实数通道（64 个复数符号/格点），总复数符号
        64×16×16 = 16384 = (256×256×3)/12，代码口径 CBR = 1/12
      - SNR=10dB、扩散 T=1000、Tmax=10 采样步；MIMO Rayleigh 可配
    """
    return SystemConfig(
        semantic=SemanticConfig(
            #checkpoint 1 : 3 stage, no vae
            # image_channels=3,
            # patch_size=8,
            # embed_dim=16,
            # num_heads=8,
            # window_size=8,
            # num_swin_blocks=3,
            # stage_embed_dims=(128, 192, 256),
            # stage_depths=(2, 2, 6),
            # stage_num_heads=(4, 6, 8),
            # stem_stride=2,
            # stage_downsample=(False, True, True),
            # use_vae=True,
            # lambda_kl=1e-4,
            # num_decoder_refine_blocks=2,

            #checkpoint 2 : 4 stage, use vae
            image_channels=3,
            patch_size=16,
            embed_dim=16,
            num_heads=8,
            window_size=8,
            num_swin_blocks=4,
            stage_embed_dims=(128, 192, 256, 320),
            stage_depths=(2, 2, 6, 2),
            stage_num_heads=(4, 6, 8, 10),
            stem_stride=2,
            stage_downsample=(False, True, True, True),
            use_vae=False,
            lambda_kl=1e-4,
            num_decoder_refine_blocks=2,
        ),
        channel=ChannelConfig(
            cbr=1 / 12,
            channel_symbols=128,
            channel_bottleneck_dim=64,
        ),
        mimo=MIMOConfig(n_tx=2, n_rx=2, snr_db=10.0, fading="rayleigh"),
        diffusion=DiffusionConfig(
            num_train_steps=1000,
            num_sample_steps=50,
            beta_start=1e-4,
            beta_end=2e-2,
            unet_hidden_dim=256,
        ),
        unet_uncond=UNetUncondConfig(
            T=1000,
            # 与 checkpoints-val/sc/sc_div2k_c{4,12,16}_best.pth 中 UNet 一致（非 Stage1 训练目标，但须能 load_state_dict）
            ch=256,
            ch_mult=(1, 2, 2),
            attn=(1,),
            num_res_blocks=2,
            dropout=0.1,
            # DIV2K 存盘权重里无条件 U 网入口通道固定为 16（与语义瓶颈 C=4/12/16 无强制一一对应）
            input_channel=16,
        ),
    )
