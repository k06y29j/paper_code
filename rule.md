1,PSNR计算方式： (各图之和/张数 ΣPSNR_i/N) ,逐图 PSNR 再平均。
2.环境是cddm_ddnm的conda环境
3.nvdia-smi查看可用GPU
4.如果在SNR 12db下，需要在12db下训练，在12db下测试
5.优化GPU利用率，应设置预加载和num-workers，以及cache_decoded，避免反复处理



6.无噪：日志保存到log-v1/two-stage 模型权重保存到checkpoints-val-v1/two-stage
  有噪：日志保存到log-v2/two-stage 模型权重保存到checkpoints-val-v2/two=stage
7.目的超过22.419 db（AWGN 12db），超过0.5db
1.有噪版本

基础模块(在各个route_a,比如swin在checkpoints-val-v1/route_a/sc_dct_c4)：

SemanticEncoder：已完成信道感知训练。Channel Encoder： 固定半正交矩阵 A，尺寸为 4×16。Channel Decoder：使用 A^T。SemanticDecoder：已完成信道感知训练。

Channel-aware SWIN + Denoiser + Refiner

在已训练好的信道感知 SWIN 编解码器和固定半正交信道矩阵 A 基础上，进一步训练接收端增强模块。训练过程中，SWIN Encoder、SWIN Decoder 和矩阵 A 均保持冻结。

首先，输入图像经过 Encoder 得到完整 16 维潜在特征 z_16，并通过 A 得到 4 维低频信道特征 z_low = A z_16。随后在 z_low 上加入 AWGN，设定训练信噪比为 12 dB，得到含噪观测 y。

第一阶段训练低频去噪器 D_low，使其从 y 中恢复干净低频特征 z_low。训练目标为最小化 D_low(y) 与 A z_16 之间的误差，从而显式去除前 4 个信道通道中的噪声。

第二阶段训练高频 refiner。将去噪后的低频特征通过 A^T 升维，得到初始 16 维特征 z_init = A^T D_low(y)。refiner 在 z_init 基础上预测高频残差 r_high，并得到最终特征 z_refined = z_init + r_high。训练目标为使 z_refined 接近原始完整特征 z_16，同时通过冻结的 Decoder 重构图像并计算图像重构损失。

该版本用于验证：在信道感知 SWIN 表征基础上，低频去噪与高频补偿是否能进一步提升 12 dB AWGN 条件下的恢复质量。



2.无噪版本

在已有的SWIN（无信道得到的）和信道编解码器的矩阵的基础上，先训练一个低频去噪器去去除前四个通道的信道噪声，再训练一个refine在前一个去噪器的基础上，去训练一个refine预测高频细节，低频去噪器和refine需要重新设计：

低频去噪器做成 **条件去噪器 Conditional Low-frequency Denoiser**，输入不仅包括含噪低频特征 `y_4`，还包括信道状态条件，比如 `SNR`、噪声方差 `σ_n²`，甚至可以进一步加入信道类型 `AWGN / Rayleigh / Rician`。

你现在只做 AWGN、SNR=12 dB，最简单有效的方案是：

**把 SNR 编码成条件向量，注入到低频去噪器的中间层。**

---

# 一、核心建模

你的低频观测为：

y = A z_16 + n

其中：

y 是 4 维含噪低频特征

A z_16 是干净低频特征

n 是 AWGN 噪声

SNR = 12 dB

低频去噪器目标是：

D_low(y, c_snr) ≈ A z_16

其中 `c_snr` 是由 SNR 或噪声方差编码得到的条件。

---

# 二、最推荐方案：FiLM 条件注入

我最推荐你用 **FiLM 条件调制**。

它的思想是：

把 SNR 编码成一个条件向量，然后生成每一层特征的缩放因子和偏置因子。

形式为：

h' = γ(c) · h + β(c)

其中：

h 是去噪器中间特征

c 是 SNR 条件编码

γ(c)、β(c) 是由小型 MLP 生成的调制参数

这样去噪器可以根据不同 SNR 自动调整去噪强度。

---

# 三、为什么 FiLM 适合你这个任务

因为低频去噪本质上是一个“噪声强度可控”的问题。

SNR 高时，y 比较可靠，去噪器应该少改：

D_low(y, high SNR) ≈ y

SNR 低时，y 噪声重，去噪器应该更多依赖学习到的低频先验：

D_low(y, low SNR) 更偏向平滑和修正

FiLM 正好适合表达这种动态行为。

---

# 四、具体网络结构建议

假设你的低频特征是：

y: B × H × W × 4

或者 PyTorch 中：

y: B × 4 × H × W

推荐低频去噪器结构：

输入：

y_4：含噪低频特征

snr_db：信噪比，例如 12

输出：

z_low_hat：去噪后的 4 维低频特征

结构：

Conv 4 → 64

Residual Block + FiLM

Residual Block + FiLM

Residual Block + FiLM

Conv 64 → 4

残差输出：z_low_hat = y + Δy

也就是让网络预测噪声残差或修正残差：

Δy = D(y, snr)

z_low_hat = y + Δy

更推荐预测残差，而不是直接预测 clean low。

---

# 五、SNR 条件怎么编码

先归一化。

Sinusoidal SNR Embedding

类似扩散模型 timestep embedding，把 SNR 做成正余弦编码：

snr_embed = SinusoidalEmbedding(snr_db)

然后经过 MLP：

c = MLP(snr_embed)

这个方案适合你后面扩展到多个 SNR，例如：

0、3、6、9、12、15 dB

如果你以后要做多 SNR 泛化，推荐这个。

# 六、训练方式推荐

多 SNR 训练

训练从离散集合采样：

SNR ∈ {0, 3, 6, 9, 12, 15}

每次生成：

y = A z_16 + n_snr

输入：

D_low(y, snr_db)

目标：

A z_16

损失：

L_low = ||D_low(y, snr_db) - A z_16||²

这样去噪器才真正学会：

“给我一个 SNR 条件，我就用对应强度去噪。”

然后你单独在 12 dB 上测试。

这才是真正的信道感知去噪器。

---

# 七、损失函数推荐

基础损失：

L_feat = ||z_low_hat - z_low||²

建议再加一个噪声残差损失：

n_hat = y - z_low_hat

n_gt = y - z_low

L_noise = ||n_hat - n_gt||²

由于：

n_gt = y - z_low

这个损失本质上和 L_feat 等价，但它能让训练目标更明确：网络是在估计噪声。

完整损失：

L_denoise = L_feat + α L_noise

一般 α = 0.1 到 0.5。

还可以加入解码低频图像约束：

x_low_hat = Decoder(A^T z_low_hat)

x_blur = GaussianBlur(x)

L_img_low = ||x_low_hat - x_blur||²

最终：

L = L_feat + α L_noise + β L_img_low

推荐初始权重：

α = 0.1

β = 0.05 或 0.1

不要一开始让图像损失太大，否则低频去噪器可能偏离真实低频特征。

---

# 九、推荐实现形式

低频去噪器可以这样设计：

```text

n_hat = network(y, c)

z_low_hat = y - n_hat

```

我更推荐预测噪声：

```text

z_low_hat = y - N_theta(y, c)

```

因为 AWGN 去噪任务中，直接预测噪声更符合物理含义。

# 十二、训练 SNR 采样策略

我建议你不要只训练 12 dB，而是这样做：

第一阶段：

SNR ∈ {6, 9, 12, 15}

让模型先学中高 SNR，避免低 SNR 把训练扰乱。

第二阶段：

SNR ∈ {0, 3, 6, 9, 12, 15}

扩展鲁棒性。

---

# 十三、和 refiner 的衔接

低频去噪器输出：

z_low_hat = D_low(y, snr)

然后升维：

z_init = A^T z_low_hat

refiner 输入建议包括三部分：

z_init

z_low_hat

snr condition

也就是说 refiner 也做成信道感知：

r_high = Refiner(z_init, z_low_hat, snr)

z_refined = z_init + r_high

并加约束：

A r_high ≈ 0

这样 refiner 知道当前信道质量：

SNR 高时可以保守补细节，SNR 低时可以更依赖先验。

---

# 十四、最终推荐方案

你的低频去噪器推荐采用：

**FiLM 条件调制残差去噪器**

完整形式：

```text

D_low(y, snr_db) = y - N_theta(y, Emb(snr_db))

```

训练目标：

```text

min ||D_low(y, snr_db) - A z_16||²

```

其中：

```text

y = A z_16 + n_snr

```


