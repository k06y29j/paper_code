# explore-2 实验总结

更新时间：2026-07-14  
范围：`MY-V2/jscc-f/explore-2` 下的 Layer 2 FSQ、VQ、连续 `q2` 预测、AR、扩散/flow 以及冻结解码器实验。

## 1. 结论先行

本目录已经证明 **Layer 2 的发送端 oracle 可以稳定获得 1 dB 以上增益，且 FSQ/VQ 码本容量增大时表达能力增强**；但在严格接收端约束下（只能使用 `z1/x1`，不能使用 `img/z2/q2/真实索引`），目前还没有任何 FSQ/VQ/AR/扩散路线达到 `PSNR(x2_hat)-PSNR(x1) >= 0.5 dB`。

| 路线 | 代表实验 | 严格接收端增益 | 发送端 oracle 增益 | 结论 |
|---|---|---:|---:|---|
| 连续 `q2` mix（非码本） | `cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-v1` | **+0.427319 dB** | 不适用 | 当前所有 receiver 路线中最高，但不是 FSQ/VQ 码本路线，且仍低于 +0.5 dB |
| FSQ K=4913 | `cnn-fsq-k4913-independent-d2-v5` | **约 +0.2598 dB** | 约 +0.9715 dB | 当前 FSQ receiver 最好；未达门槛 |
| FSQ K=125 hard | `cnn-fsq-k125-frozen-d2-combiner-hard-v1` | **+0.225616 dB**（日志末行）；最佳已保存初始化 **+0.254072 dB** | +1.103991 dB | K=125 hard 预测最接近，但未达门槛 |
| channel-VQ AR | grouped `D=512` | **+0.196760 dB** | 见 VQ oracle 表 | AR 索引路线未达门槛 |
| image-VQ diffusion | CNN image-VQ | **+0.118283 dB** | 见 VQ oracle 表 | 扩散路线未达门槛 |
| FSQ K=125 AR | joint-token AR | 最好为负增益 | FSQ oracle 为正 | teacher-forcing 与自由 rollout 存在明显鸿沟 |

因此：若只看“真正经过 FSQ 硬格点”的当前最好初始化，应使用 K=125 hard checkpoint；若只看现有 FSQ receiver 的最终完整验证结果，应使用 K=4913 independent-D2。二者都只是“最接近门槛”，不能称为达标。

## 2. 固定架构、数据和验收协议

### 2.1 两层架构

```text
发送端：img -> E1 -> z1 -> D1 -> x1
Layer 2：img, x1 -> E2 -> z2 -> quantizer -> q2 -> D2 -> u2
                 x1, u2 -> combiner -> x2

接收端：z1, x1 -> predictor/generator -> q2_hat
                 q2_hat -> D2 -> u2_hat
                 x1, u2_hat -> combiner -> x2_hat
```

接收端预测/生成 `q2_hat` 时严格禁止使用发送端信息 `img`、`z2`、真实 `q2` 或真实索引；允许使用 Layer 1 的 `z1`、`x1` 及其后产生的特征。日志中的 `oracle` 只表示发送端真实量化结果经过 D2/combiner 的上界，不是部署结果。

### 2.2 数据裁剪

- 训练：随机 `256x256` crop，并使用随机水平翻转。
- 验证：中心 `256x256` crop。
- 主要验收：DIV2K `valid100`，指标为 RGB 重建 PSNR。
- `PSNR(x1)` 取决于 Layer 1 checkpoint：CNN FSQ 分支约 `21.772013 dB`；Swin VQ 分支约 `27.565002 dB`，不同 Layer 1 的增益不可直接混合比较。

### 2.3 代码审计

`contracts.py`、`test_receiver_contract.py` 和各 receiver 日志执行了无泄漏审计。训练/验证 crop 也按上述协议实现。实验报告中凡是 `full valid100` 的结果才进入主表；smoke、preflight、单 batch 或 teacher-only 结果只作为诊断。

## 3. FSQ 实验

### 3.1 FSQ oracle：K 增大确实提升 Layer 2 表达能力

CNN Layer 1/2，FSQ 三维 latent，`K=levels^3`：

| FSQ | `PSNR(x1)` | `PSNR(x2 oracle)` | oracle 增益 | 码本/使用情况 |
|---|---:|---:|---:|---|
| K=125，`[5,5,5]` | 21.771972 | 22.876459 | **+1.104487** | 125 个格点均使用，entropy 6.538245 bit，PPL 92.9411 |
| K=729 | 约 21.772 | 22.905375 | **约 +1.133403** | 未见整体塌陷 |
| K=4913，`[17,17,17]` | 约 21.772 | 22.947125 | **约 +1.1755** | 码本表达力继续增强 |

K=125 已有足够 oracle headroom，receiver 未达标的主要瓶颈不是 FSQ 码本容量，而是从 `z1/x1` 恢复可用 `q2` 的难度。

### 3.2 K=125 receiver：直接预测 q2、hard snap 和独立 D2

| 实验 | D2/combiner | 输出形式 | `PSNR(x2_hat)` | 相对 x1 | index acc | 条件打乱下降 | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| `cnn-fsq-k125-frozen-d2-combiner-cont-v1` | 冻结 | 连续 `q2_hat` | 21.989879 | **+0.217866** | .322096 | .274918 dB | 未达标 |
| `cnn-fsq-k125-frozen-d2-combiner-hard-v1` | 冻结 | FSQ hard snap | 21.997629 | **+0.225616** | .322201 | .314029 dB | 未达标 |
| 同一 hard 最佳初始化 checkpoint | 冻结 | FSQ hard snap | 22.026085 | **+0.254072** | .325898 | .417073 dB | 当前 K=125 hard 最佳起点 |
| `cnn-fsq-k125-independent-d2-v5` | receiver 独立训练 | 连续 `q2_hat` | 21.924222 | **+0.152209** | .918620（标量） | .144669 dB | 精度高但 PSNR 低 |
| `cnn-fsq-k125-independent-hard-final-v7` | receiver 独立训练 | hard snap | 21.877952 | **+0.105938** | .287370 | .094941 dB | 失败 |
| `cnn-fsq-k125-joint-predictable-v4` | 联合可预测 | 连续 `q2_hat` | 21.808963 | **+0.036950** | .922253（标量） | .004022 dB | 典型“高准确率不等于高 PSNR” |
| `cnn-fsq-k125-sender-qhat-deploy-v11` | sender qhat 对齐 | 连续 `q2_hat` | 21.936993 | **+0.164981** | .91--.93 | .161646 dB | 未达标 |
| `cnn-fsq-k125-joint-index-v1` | 联合索引 | 离散 index | 21.927961 | **+0.155947** | .055039 | 未形成有效增益 | 未达标 |

结论：冻结 D2/combiner 可以保证 receiver 使用真实部署链路，但只靠 `z1/x1` 的直接 FSQ 预测仍只能恢复部分有效信息。独立 D2 可以提高 index acc，却可能把 D2 学成与发送端 q2 不同的坐标系，导致 PSNR 不随 accuracy 同步提高。

### 3.3 K=4913 receiver

| 实验 | 输出 | 严格 valid100 增益 | index acc | 条件打乱下降 | 结论 |
|---|---|---:|---:|---:|---|
| `cnn-fsq-k4913-direct-cont-v1` | 连续 | +0.1823 dB | .2728 | — | 未达标 |
| `cnn-fsq-k4913-direct-hard-v1` | hard | +0.0290 dB | .2652 | — | hard snap 对该训练方式不稳定 |
| `cnn-fsq-k4913-independent-d2-v5` | 连续 | **+0.2598 dB** | .7462 | .3245 dB | 当前 FSQ receiver 最好 |
| `cnn-fsq-k4913-independent-d2-final-v6` | 连续 | +0.1986 dB | .7202 | .2399 dB | 次优 |
| `cnn-fsq-k4913-joint-predictable-v3/v4` | 连续 | +0.2017/+0.2117 dB | 最高约 .7403 | — | 未达标 |
| residual anchor / rx joint fine-tune | 连续/残差 | +0.0965 至 +0.2497 dB | — | — | oracle 可能退化，不作为正式路线 |

K=4913 提高 oracle 上限，但 receiver 预测难度也提高；容量增大没有自动转化为 receiver 增益。

### 3.4 FSQ AR、flow 和 categorical diffusion

这些实验均遵守“接收端只看 `z1/x1`”的约束，且 D2/combiner 使用部署链路。

| 路线 | 代表配置 | 最佳 receiver 结果 | 观察 |
|---|---|---:|---|
| K=4913 AR index | `cnn-fsq-k4913-ar-index-h128-v13` | 约 +0.12 dB | 自回归 rollout 远低于 oracle；未达标 |
| K=4913 AR residual | `cnn-fsq-k4913-ar-residual-base-v14` | +0.096075 dB | full val index acc .266055，未达标 |
| K=125 joint-token AR | frozen D2/combiner，CE | 最好为负增益 | teacher-forced train acc 到 e15 为 .096758；full rollout val acc 约 .036289 |
| K=125 iFSQ-style continuous flow | Gaussian linear flow，MSE+cosine | 最好 -0.248606 dB（e10） | K=125 仅实现了近似路线，不是官方 iFSQ 的完整复现 |
| K=125 categorical posterior diffusion | cosine VP，NFE=12，CE | 连续 posterior mean 最好 -0.035550；hard 最好 -0.057079 | DDIM rollout 仍接近随机，未达标 |
| categorical diffusion prior scale/NFE read-only | prior scale 2/3，NFE=25 | 连续 -0.039005；hard -0.066323 | 推理步数/先验缩放不能修复训练目标错配 |
| image-VQ diffusion | CNN image-VQ | **+0.118283 dB** | final-only v5 反而 -0.5401 dB |
| channel-VQ AR | grouped D=512 | **+0.196760 dB** | soft val75 约 +0.1538 dB |

AR 的核心问题是 teacher forcing 与自由 rollout 分布不一致；扩散/flow 的核心问题是 terminal posterior 的训练目标与“经过固定 D2/combiner 后的 PSNR”不一致。当前日志没有证明继续堆 NFE 或训练 epoch 能达到 +0.5 dB。

## 4. 连续 q2（非 FSQ/VQ）基线

这部分用于确定预测难度上限，但不应被写成“利用码本信息”的成功路线。

| 实验 | 训练/冻结方式 | full valid100 增益 | 备注 |
|---|---|---:|---|
| single-generator 系列 | 不同 q2 预测器 | 最高约 +0.3970 dB | 低于门槛 |
| train-simplex top3 mix | 固定 q2 mix，单一 D2/combiner | +0.401097 dB | 非图像 ensemble；为可复现基线 |
| `cnn-continuous-q-mix-top3-logmse-e60-v1` | top3 + log-MSE | e35 最后完整验证 +0.412509 dB | 训练提前停止 |
| `cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-v1` | top3 + EMA | **+0.427319 dB** | 当前非码本 receiver 最好，仍低于 +0.5 |
| v12 read-only fixed-q diagnostic | 仅校准/诊断 | +0.402400 dB | 不是新的完整训练结论 |
| qonly/high-res residual | 不同 LR | 约 +0.287 至 +0.402 dB | 未形成门槛突破 |

此前的“冻结 D2/combiner，只训练 1027 个校准标量”的 frozen-post-q calibrator 不纳入正式路线，因为它没有利用 FSQ/VQ 码本信息，且已明确不作为用户要求的方案。

## 5. VQ：image-VQ 与 channel-VQ 的 oracle/容量实验

### 5.1 CNN image-VQ，Layer 2 embedding D=512/1024

固定 epoch25，CNN Layer 1/2，`C=256`，`PSNR(x1)=21.771992 dB`：

| embedding D | K=256 | K=1024 | K=4096 | K 随容量是否单调 |
|---:|---:|---:|---:|---|
| 512 | +1.044606 | +1.177847 | +1.211962 | 是 |
| 1024 | +1.126254 | +1.279032 | +1.283548 | 是 |

`D=1024-D=512` 的额外增益约 `+0.081647/+0.101185/+0.071585 dB`。后续 D=1024、e60 的独立 oracle 结果更高：K=256/K=1024/K=4096 分别为 **+1.383345/+1.547117/+1.650135 dB**，且 oracle capacity goal 通过。

### 5.2 CNN grouped channel-VQ，C=256

固定 epoch20：

| embedding D | K=512 | K=1024 | K=4096 | 结论 |
|---:|---:|---:|---:|---|
| 256 | +0.402444 | +0.508769 | +0.507589 | K=4096 相比 K=1024 略非单调，容量限制较明显 |
| 512 | **+0.575905** | **+0.727043** | **+0.748594** | D 增大后整体提升，容量表达有效 |

D=512 相对 D=256 的增益为 `+0.173461/+0.218274/+0.241005 dB`。D=512 的 local PPL 为 `1.9969/3.9300/14.2713`，top1 占比为 `.5280/.3063/.1343`，说明不是简单单码塌陷；shuffle drop 为 `.4105/.6152/.6028 dB`。

### 5.3 Swin global channel-VQ

Swin Layer 1/2，`C=256`，固定 epoch15：

| embedding D | K=256 | K=1024 | K=4096 |
|---:|---:|---:|---:|
| 128 | +0.317844 | +0.412309 | +0.474194 |
| 256 | +0.345848 | +0.465433 | **+0.555498** |

所有主要 K/D 组合的使用率、PPL/K、top1 和 zero/shuffle 检查通过；D=256 相比 D=128 在 K=256/1024/4096 分别增加 `+0.028004/+0.053124/+0.081304 dB`。这证明 Swin + channel-VQ 的 oracle 表达力满足要求，但 receiver AR 仍只有约 +0.2 dB。

### 5.4 VQ 实现和速率语义

- image-VQ：每个空间 token 一个 code，索引形状约 `[B,16,16]`，码本形状 `[K,D]`，名义 bits/image 为 `256 log2(K)`。
- channel-VQ：每个 channel 一个 code，global 版本原生形状 `[B,C,16,16]`，高维版本可为 `[K,1,D]`；`C` 与 embedding `D` 独立。
- grouped channel-VQ：`K` 通常需被 `C` 整除，每个 channel 在局部候选中量化；报告中使用其实际 local candidate 语义，不把 nominal K 误写成 image-VQ 的 token 数。
- Layer 2 编解码器支持独立 `cnn/swin/residual-cnn/match` 选择；Layer 1 的 CNN/Swin 与 Layer 2 quantizer 类型是两个独立维度。

## 6. 失败原因和可复用诊断

| 现象 | 日志证据 | 原因判断 |
|---|---|---|
| scalar index accuracy 很高（约 .92），PSNR 只有 +.037 dB | K=125 `joint-predictable-v4` | index accuracy 没有衡量码点误差、D2 局部敏感性和空间位置权重；错误少但可能集中在高影响 token |
| 独立 D2 的 index acc 高于直接预测，但增益仍约 +.26 dB | K=4913 `independent-d2-v5` | receiver D2 适配了预测分布，却未完全保持 sender q2 的可解码坐标系 |
| teacher-forced AR 看起来改善，free rollout 崩溃 | K=125 AR：train acc .096758 vs rollout val acc .036289 | exposure bias；训练时每一步看到真历史，推理时使用自身错误历史 |
| categorical diffusion 的 posterior mean 和 hard terminal 都为负 | K=125 CDCD 日志 | 只优化 token CE/后验，不等价于固定 D2+combiner 的重建 PSNR；采样误差会被 D2 放大 |
| 增加 NFE/prior scale 没有改善 | NFE12/25 read-only 对比 | 推理积分误差不是主瓶颈，训练目标/条件信息才是主瓶颈 |
| VQ K 增大但曲线不单调或 PPL≈1 | Swin image-VQ hard、部分 residual-cnn 分支 | 码本塌陷、commitment/usage loss 失衡或 decoder 过强；不能只看 oracle PSNR |
| 训练 epoch 指标和验证 epoch 指标差异大 | AR、image diffusion 和多种 joint predictor | train 使用 teacher-forced/发送端条件或随机 crop，valid 使用 free rollout/中心 crop；必须同时看部署闭环 full valid100 |

后续比较应至少同时记录：`PSNR(x1)`、oracle `PSNR(x2)`、receiver `PSNR(x2_hat)`、增益、q/index 误差、condition shuffle drop、zero/shuffle drop、PPL/usage/top1、是否 hard snap、是否冻结 D2/combiner，以及训练/验证是否同一 rollout 方式。

## 7. 目录与实验清单

### 7.1 主要结果文档

- [`ARCHITECTURE_AND_PLAN.md`](./ARCHITECTURE_AND_PLAN.md)：架构、无泄漏协议和阶段性计划。
- [`FSQ_LITERATURE_AND_IFSQ_AUDIT.md`](./FSQ_LITERATURE_AND_IFSQ_AUDIT.md)：FSQ/iFSQ/AR/flow/diffusion 文献路线及本地复现实验审计。
- [`results-receiver/fsq_receivers.md`](./results-receiver/fsq_receivers.md)：FSQ receiver checkpoint 元数据和严格验收表。
- [`results-vq/vq_matrix.md`](./results-vq/vq_matrix.md)：CNN/Swin × image/channel-VQ 形式化矩阵；汇总 35 个版本级 run、8 个 canonical formal run，缺失 2x2 cell 为 0，但 strict completeness 仍 FAIL（部分 canonical cell 失败或未知）。
- [`results-vq/embedding_scaling_fixed_epoch.md`](./results-vq/embedding_scaling_fixed_epoch.md)：CNN image-VQ 与 grouped channel-VQ 的 K/D scaling。
- [`results-vq/swin_global_channel_vq_embedding_scaling_epoch15.md`](./results-vq/swin_global_channel_vq_embedding_scaling_epoch15.md)：Swin channel-VQ scaling。
- [`logs-continuous-q-mix/SMOKE_RESULTS.md`](./logs-continuous-q-mix/SMOKE_RESULTS.md)：q-mix 结构 smoke；其中单 batch 数值不能当作 full valid100 结果。

### 7.2 日志目录覆盖范围

| 目录 | 文件量（含 bootstrap/preflight/eval 重复） | 主要内容 |
|---|---:|---|
| `logs` | 50 | K=4913 FSQ direct/index/AR、部分 q-only/refiner |
| `logs-125` | 8 | K=125 direct continuous/hard、joint/index、sender-qhat |
| `logs-channel-ar` | 8 | CNN/Swin channel-VQ AR，含 hard/soft/condition 变体 |
| `logs-image-diffusion` | 10 | CNN image-VQ diffusion 及 residual/final-only 变体 |
| `logs-fsq-generation` | 31 | K=125 AR、iFSQ-style flow、categorical diffusion、preflight/eval |
| `logs-continuous-q` | 15 | 单 generator、q-only、residual、high-res、ensemble |
| `logs-continuous-q-mix` | 8 | top3 q-mix、log-MSE、EMA、smoke/diagnostic |
| `logs-vq` | 94 | CNN/Swin image/channel-VQ oracle、capacity、collapse、decoder/receiver 诊断 |

日志中存在同一实验的 `bootstrap`、`preflight`、`eval` 和重启副本；比较时按实验名和 checkpoint 元数据去重，不能按文件数统计实验数。

### 7.3 代表性运行名索引

**FSQ K=4913：**

`cnn-fsq-k4913-direct-cont-v1`、`cnn-fsq-k4913-direct-hard-v1`、`cnn-fsq-k4913-independent-d2-v5`、`cnn-fsq-k4913-independent-d2-final-v6`、`cnn-fsq-k4913-independent-hard-final-v7`、`cnn-fsq-k4913-joint-predictable-v3/v4`、`cnn-fsq-k4913-ar-index-h128-v13`、`cnn-fsq-k4913-ar-residual-base-v14`、以及对应的 preinit、residual-anchor、rx-joint、qonly/refiner 诊断分支。

**FSQ K=125：**

`cnn-fsq-k125-frozen-d2-combiner-cont-v1`、`cnn-fsq-k125-frozen-d2-combiner-hard-v1`、`cnn-fsq-k125-independent-d2-v5`、`cnn-fsq-k125-independent-hard-final-v7`、`cnn-fsq-k125-joint-predictable-v4`、`cnn-fsq-k125-joint-index-v1`、`cnn-fsq-k125-sender-qhat-deploy-v11`。

**生成路线：**

K=125 `joint-token AR`、`ifsq-style continuous flow`、`categorical posterior diffusion`（continuous posterior mean/hard terminal、prior-scale/NFE read-only）；K=4913 AR index/residual；image-VQ diffusion v1--v5；channel-VQ grouped D=512 hard、soft 和 Swin global hard。

**连续 q2：**

single-generator、q-only/high-res residual、train-simplex top3、`cnn-continuous-q-mix-top3-logmse-e60-v1`、`cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-v1`、v12 read-only diagnostic。

**VQ 矩阵/容量：**

CNN/Swin Layer 1 × CNN/Swin/Residual-CNN Layer 2 × image-VQ/channel-VQ，覆盖 K=256/512/1024/4096、embedding D=128/256/512/1024，以及 collapse、RMSNorm、soft-usage、frozen decoder、receiver AR/decoder 诊断分支。正式矩阵以 `results-vq/vq_matrix.md` 为准。

## 8. 当前最合理的后续使用方式

1. 若研究重点是“码本表达力”：优先复用 CNN image-VQ `D=1024,K=4096` 或 CNN grouped channel-VQ `D=512,K=4096` 的 oracle 训练配置，并继续检查 PPL/top1/zero/shuffle，而不是只扩大 K。
2. 若研究重点是“接收端 FSQ”：优先从 K=125 hard 最佳 checkpoint 或 K=4913 independent-D2 checkpoint 出发，保持 D2/combiner 冻结，避免把 receiver 重训成另一套码本坐标系。
3. 若研究重点是 AR/扩散：先解决 free-rollout 的训练/验证闭环和面向 D2+combiner 的重建损失，再增加模型规模或采样步数；现有日志没有支持“继续堆工程参数即可过 +0.5 dB”的证据。

最终验收标准仍是严格 full valid100 的 `PSNR(x2_hat)-PSNR(x1) >= +0.5 dB`，并同时通过无泄漏、码本使用率和容量单调性检查。本目录截至本总结尚无满足全部条件的 receiver 版本。
