# explore-4 实验总结

更新时间：2026-07-16  
依据：`logs-cvq/`、`results-cvq/`、`logs-car/`、`results-car/`、checkpoint 内嵌 `args/metrics`，以及本目录的三个 launcher。

## 1. 目标与验收协议

本目录围绕 arXiv `2605.26089v2` 的 Channel-wise Vector Quantization（CVQ）和 Channel-wise AutoRegressive（CAR）展开。Layer 1 使用已经训练好的 CNN 或 Swin 编解码器并冻结；Layer 2 产生 240 个 q2 channel token。16×16 配置中，D2 前端的输入固定为：

```text
[z1[:16], q2[:240]] = 256 channels
```

前 16 个 z1 通道不做 q2 nested dropout。解码使用 residualized contract：

```text
x2 = clip(x1 + D2(z1, q2) - D2(z1, 0))
```

因此 q2=0 时严格退化为 x1，Layer 1 的 z1 只能作为 q2 残差的条件，不能绕过码本取得额外重建收益。

数据协议统一为：训练 `RandomCrop(256)+RandomHorizontalFlip`，验证 `CenterCrop(256)`。CVQ 的发送端可以使用 `img/x1` 得到真实 q2；CAR 接收端验收时不能使用 `img、z2、真实 q2、真实 index`。严格 z1-only Qwen CAR 的部署链路是：

```text
z1 -> BOS -> Qwen3 CAR 自回归生成 q2_hat -> D2(q2_hat,z1) -> combiner(x1,...)
```

这里 `x1` 只在 q2_hat 完成后进入接收解码；`train_cvq_car.py` 的旧 PaperCAR 仍是 `(z1,x1)` 条件基线，不能作为最终 z1-only 方案。

最终门槛为完整 valid100：

```text
PSNR(x2_hat) - PSNR(x1) >= +0.5 dB
drop_zero >= +0.1 dB
drop_shuffle >= +0.1 dB
condition_shuffle_drop >= +0.1 dB
receiver_only_audit = 1
```

其中 `condition_shuffle_drop` 用于确认 CAR 真正使用 z1；只看 PSNR 而忽略这一项会产生假阳性。

## 2. 代码与模型实现

| 文件 | 作用 |
|---|---|
| [`train_cvq_spatial.py`](train_cvq_spatial.py) | CNN/Swin Layer 1 冻结条件下的 Layer 2 E2、CVQ、D2 前端、nested dropout、码本健康统计和 conditional prior 训练 |
| [`cvq_predictability.py`](cvq_predictability.py) | `p(i_k\|z1,i_<k)`、soft assignment conditional-rate loss、按 channel 的 marginal entropy 统计 |
| [`train_cvq_car.py`](train_cvq_car.py) | 论文式小型 decoder-only PaperCAR，部署条件为 z1+x1，作为旧基线 |
| [`train_cvq_car_qwen.py`](train_cvq_car_qwen.py) | Qwen3-4B decoder-only CAR；z1 map prefix、BOS、CVQ map projector、hard/continuous rollout、scheduled continuous history 和 Stage-II 开关 |
| [`run_cvq_spatial.sh`](run_cvq_spatial.sh) | CVQ launcher，默认 C=240、K=16384、nested dropout alpha=0.5、10 epoch continuous residual warm-up |
| [`run_cvq_car.sh`](run_cvq_car.sh) | 旧 PaperCAR launcher |
| [`run_cvq_car_qwen.sh`](run_cvq_car_qwen.sh) | Qwen3 Stage-I launcher，默认 Qwen3-4B、K=4096、z1-only generator |

Qwen CAR 的输入序列为 `[P(z1), BOS, e(i1), ..., e(iC)]`。`e(i)` 是 CVQ 码本中的完整 channel map，经共享两层 MLP 投影到 Qwen hidden；Qwen3-4B backbone 本地可用，8B 在当前环境没有缓存。hard 路径前向严格查码本格点；continuous 路径是同一个码本上的 posterior mean `sum_i p_i e_i`，因此它利用了码本但一般不落在单一格点上。

## 3. CVQ 主实验结果

下表中的增益均是验证结果；对中途停止的运行，标明最后有效验证或 checkpoint best。`oracle_goal_met=1` 只表示码本 oracle 有正增益并通过健康检查，并不表示 CAR 已达到 +0.5 dB。

| Layer 1 | D / 空间 | K | 实验 | best/valid epoch | PSNR(x1) | PSNR(x2) | oracle 增益 | PPL | 结论 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| CNN | 256 / 16×16 | 1024 | `cvq-cnn-f16-c16plus240-k1024-e200-b8t8-v3` | e15 | 21.772 | 21.825 | +0.053 | 715.6 | 正增益太小，不能进入 CAR |
| CNN | 256 / 16×16 | 4096 | `...k4096-paper-alpha05-resdelta-warm10-b32-e200-v1` | e30 | 21.772 | 22.561 | **+0.789** | 1408.6 | 第一个稳定通过 oracle/码本健康门槛的 CNN 版本 |
| CNN | 256 / 16×16 | 4096 | `...k4096-z1condrd-freezedec-e80-v1` | best e40；末期 e60 | 21.772 | 22.702(best) | **+0.930** | 434.9(best)；末期 19.2 | best 很好，但继续训练后码本塌陷，末期不合格 |
| CNN | 256 / 16×16 | 4096 | `...k4096-z1mi-balanced-e100-v2` | best e90 | 21.772 | 22.724 | **+0.952** | 891.4 | conditional marginal entropy 后仍保持健康，是 CNN 条件-RD 的最佳发送端 |
| CNN | 256 / 16×16 | 16384 | `...k16384-paper-alpha05-resdelta-warm10-b32-e200-v7` | best e50 | 21.772 | 22.795 | **+1.023** | 4645.7 | 当前 CNN oracle 最佳，K 增大确实提升表达力且未塌陷 |
| CNN | 1024 / 32×32 | 16384 | `cvq-cnn-f32-c16plus240-k16384-paper-alpha05-resdelta-warm10-b16-acc4-e200-v1` | warm-up e9 | — | — | — | — | 只完成 continuous warm-up，未进入有效 CVQ 验证，未完成 |
| Swin | 256 / 16×16 | 1024 | `cvq-swin-f16-c16plus240-k1024-e200-b6t6-v3` | e10 | 27.565 | 27.577 | +0.012 | 634.7 | 没有可用 q2 增益 |
| Swin | 256 / 16×16 | 16384 | `...swin...sourcecomb-pca-e2e-b16a4-e60-v3` | e50 | 27.565 | 27.934 | +0.369 | 4700.3 | 码本健康，但 oracle 仍低于 CAR 所需 +0.5，不能靠预测器补足 |
| Swin | 256 / 16×16 | 16384 | `...sourcecomb-pca65536-teacher-warm5-e40-v2` | e20 | 27.565 | 27.765 | +0.200 | 4884.0 | teacher/PCA 版本低于 e2e PCA 主结果 |

### 3.1 CNN K/D 及 q2 相关性消融

`cvq-cnn-f16-d256-v1`、`...-usage-v2`、`...-anchored-v4`、`...-relevance-v3`、`...-z1q-v5` 的有效早期验证增益约为 `+0.011~+0.084 dB`。这些运行说明只给 D2 加 q2、只加 usage 或 relevance hinge，不能解决 fresh-E2 的 Layer-1 bypass/码本无效问题；正式的 residual warm-up + channel-balanced codebook initialization 才得到 K4096/K16384 的稳定结果。

CNN K16384 的无 warm-up、direct-1dB gate 和 early smoke 版本在 e1~e8 即出现 `0 dB` 附近、负增益或低 PPL；它们被 warm-up 版本取代。K1024 保留为低容量消融，虽然 PPL 尚可，但 oracle 只有约 +0.05 dB。

### 3.2 Swin 初始化与失败 probe

Swin 的 `d256-*` 试验、source-combiner smoke、PCA65536 preflight、fresh-E2、source-E2/rms 版本主要用于排查 D2 source contract、PCA adapter 和 teacher warm-up。典型结果为负增益或 q2 退化；其中 `sourcecomb-pca-e2e` 是唯一达到正 oracle 增益且码本健康的主结果，但 +0.369 dB 已构成该发送端的容量上限证据。Swin K1024 及 teacher smoke 不应继续投入 CAR。

## 4. CAR 实验结果

### 4.1 旧 PaperCAR（条件为 z1+x1）

| 实验 | K | 部署表示 | valid 增益 | 结论 |
|---|---:|---|---:|---|
| `car-cnn-f16-c240-k16384-paper-freezedec-b4-e80-v1` | 16384 | hard | -0.115 dB | 不能过线 |
| `car-cnn-f16-c240-k16384-paper-postmean-t02-freezedec-b4-e80-v2` | 16384 | posterior mean | -0.053 dB | 不能过线 |
| `car-cnn-f16-c240-k4096-paper-hard-freezedec-b4-e80-v3` | 4096 | hard | -0.153 dB | 不能过线 |

这些结果验证了“只做 teacher-forced CE + frozen decoder”不足以消除 BOS→C 的 exposure bias；同时它们的部署条件含 x1，不是当前要求的最终 z1-only 路线。

### 4.2 Qwen3-4B z1-only CAR

| 实验 | 训练/部署 | valid 结果 | teacher-history 诊断 | 结论 |
|---|---|---:|---:|---|
| `qwen3-4b-z1-k4096-stage1-e40-b4t8-v2` | frozen Qwen，hard | -0.694 dB | — | 基线失败 |
| `...condrd-e25-contmean-rd100-t1-e50-v1` | frozen Qwen，旧 continuous/argmax hybrid | -0.965 dB | mean +0.343 dB | 训练与推理 history 不一致 |
| `...condrd-e25-hardst-rd100-e50-v1` | exact hard-ST codebook | -0.975 dB | hard -0.021 dB | hard route 失败 |
| `...condrd-e30-contfeedbackfp32-rd100-t1-e55-v2` | **真正 continuous map feedback** | **best -0.253 dB（e40）** | mean +0.437~+0.445 dB | 修复了 hybrid mismatch，但仍低于 +0.5；condition shuffle 为负 |
| `...mi-balanced-e90-stage1-e40-v1` | MI-balanced sender + frozen Qwen | -0.711 dB | mean -0.304 dB | 码本 PPL 提升没有转化为 Qwen 可预测性 |
| `...scheduled-p50-s16-e8-e20-v1` 及 scheduled smoke | hard scheduled history | -0.659 dB 或更差 | — | 没有有效改善，未继续投入 |
| `qwen3-contfeedback-fp32-smoke-v2` | continuous，`max-val-batches=1` | +0.085 dB | mean +1.220 dB | 单 batch smoke，不是 valid100，不能验收 |

true continuous v2 的 e45 完整 valid100 指标为：`PSNR(x1)=21.7720`、`PSNR(x2_hat)=21.5132`、增益 `-0.2588 dB`、hard `-0.7252 dB`、teacher-history mean `+0.4455 dB`、index accuracy `0.0671`、`drop_zero=-0.2588 dB`、`condition_shuffle_drop=-0.1552 dB`。因此它是当前最接近真实连续 CAR 的版本，但没有达到门槛。

### 4.3 Qwen Stage-II probe

论文 Stage-II 要求低学习率端到端更新 Qwen。已完成显存和短程验证：Qwen3-4B 全量反传、gradient checkpointing、batch=4、effective batch=32 可以在 GPU2 上运行；单 batch probe 的数值波动很大。随后进行的 `qwen3-4b-z1-k4096-condrd-stage2-fullmetrics-e50-b4a8-lr2e6-v1` 在 e46 完整 valid100 得到：

```text
real continuous delta       -0.2056 dB
teacher-history mean delta   +0.4376 dB
teacher-history hard delta   +0.2780 dB
index accuracy                0.0653
condition_shuffle_drop       -0.1002 dB
goal_met                      0
```

它没有证明 Qwen 容量是瓶颈已被解决，因此在 e46 后停止。该运行使用 `--no-checkpoint`，只保留日志/JSON；全量 Stage-II 权重没有保存。原因是当前 workspace 用户磁盘配额为 300 GB 且已用满，尝试保存 optimizer/full Stage-II 状态时在约 3.52 GB 处写失败；本次生成的损坏 partial checkpoint 已清理。

## 5. 关键诊断与结论

1. **CNN 发送端有足够 oracle 余量。** K4096 conditional-RD 的 oracle 约 `+0.93 dB`，MI-balanced 约 `+0.95 dB`，K16384 warm-up 约 `+1.02 dB`。因此 CNN 的问题不是 D2 没有剩余信息，而是接收端无法从 z1 稳定恢复 q2。

2. **Swin 当前发送端不能作为 CAR 起点。** 最好的 Swin source-comb/PCA e2e oracle 只有 `+0.369 dB`；即使 CAR 完美也不能满足 +0.5 dB，所以应先重做 Swin CVQ/Layer-2，而不是继续换生成器。

3. **真正的瓶颈是条件熵和 exposure bias。** condrd CNN sender 的 prior CE 约 5.52 nat，真实 Qwen rollout index accuracy 约 6%~7%；e35 的 teacher continuous `+0.436 dB` 与真实 continuous `-0.287 dB` 相差约 `0.72 dB`。因此只看 teacher-forced PSNR 会误判。

4. **MI-balanced 不能只看 PPL。** 它把 PPL 从约 435 提到约 891，oracle 也略升，但 conditional prior accuracy 降到约 7.1%，Qwen teacher-history 反而更差。码本 marginal entropy 增大并不等价于 `H(I_k|z1,I_<k)` 下降。

5. **continuous route 必须保持连续 history。** 旧版本在每步把 posterior mean 再 argmax 成 index，导致训练/推理不一致。当前 `rollout_continuous()` 已经使用真实 FP32 codebook posterior mean 回馈下一步；`forward_scheduled_suffix_continuous()` 也已加入同一 history 语义。它们仍未达到 valid100 门槛。

6. **z1 使用情况必须单独验收。** 多个 Qwen 运行的 `condition_shuffle_drop` 为负，表示打乱 z1 后并未变差。当前源码已把 condition shuffle 加入 `goal_met`，旧 JSON 中的 `goal_met` 不应被重新解释为已经通过 z1 因果审计。

7. **论文 nested dropout 已在发送端实现。** q2 的前缀 nested dropout 概率 alpha=0.5，active-prefix VQ/EMA loss 与 z1 固定前缀分离；这建立了 coarse-to-fine channel order。但当前 conditional-rate 权重仍不足以把实际 Qwen 的条件熵降到可验收水平。

## 6. 当前最有价值的 checkpoint

### CNN CVQ oracle

- [CNN K16384 warm-up v7](checkpoints-cvq/cvq-cnn-f16-c16plus240-k16384-paper-alpha05-resdelta-warm10-b32-e200-v7/best.pth)：`+1.0226 dB`，PPL `4645.7`，码本健康。
- [CNN K4096 MI-balanced v2](checkpoints-cvq/cvq-cnn-f16-c16plus240-k4096-z1mi-balanced-e100-v2/best.pth)：`+0.9523 dB`，PPL `891.4`，码本健康。
- [CNN K4096 conditional-RD v1](checkpoints-cvq/cvq-cnn-f16-c16plus240-k4096-z1condrd-freezedec-e80-v1/best.pth)：best e40 `+0.9298 dB`；不要使用其末期 e60 collapsed state。
- [CNN K4096 paper baseline](checkpoints-cvq/cvq-cnn-f16-c16plus240-k4096-paper-alpha05-resdelta-warm10-b32-e200-v1/best.pth)：`+0.7891 dB`，PPL `1408.6`，是最稳的非条件 baseline。

### Qwen z1-only CAR

- [Qwen continuous FP32 v2](checkpoints-car/qwen3-4b-z1-k4096-condrd-e30-contfeedbackfp32-rd100-t1-e55-v2/best.pth)：当前真实 continuous 路线最佳 checkpoint（best e40，strict valid 增益 `-0.2532 dB`）。它是研究基线，不是验收通过版本。
- `qwen3-4b-z1-k4096-condrd-e25-contmean-rd100-t1-e50-v1`：旧 hybrid continuous，保留用于 exposure-bias 对比，不作为部署版本。

## 7. 启动方式

### CVQ

默认 launcher 是论文 K=16384、C=240、16×16、alpha=0.5：

```bash
CUDA_VISIBLE_DEVICES=0 bash MY-V2/jscc-f/explore-4/run_cvq_spatial.sh \
  cnn 4 cvq-cnn-f16-c16plus240-k16384-v1
```

K4096 需要显式覆盖 `RATES`：

```bash
CUDA_VISIBLE_DEVICES=0 RATES=4096 EPOCHS=200 BATCH_SIZE=32 \
bash MY-V2/jscc-f/explore-4/run_cvq_spatial.sh \
  cnn 4 cvq-cnn-f16-c16plus240-k4096-v1
```

32×32/D=1024 入口为 `cnn 3`，但当前目录中的对应作业只完成 warm-up，不能直接视为完成实验。

### Qwen3 z1-only CAR

以 CNN K4096 conditional-RD checkpoint 为例，连续 q2_hat 的核心参数为：

```bash
SENDER=MY-V2/jscc-f/explore-4/checkpoints-cvq/\
cvq-cnn-f16-c16plus240-k4096-z1condrd-freezedec-e80-v1/best.pth
CUDA_VISIBLE_DEVICES=2 bash MY-V2/jscc-f/explore-4/run_cvq_car_qwen.sh \
  "$SENDER" qwen3-4b-z1-k4096-continuous-v1 \
  --rollout-q-mode mean \
  --rollout-temperature 1.0 \
  --soft-temperature 1.0 \
  --lambda-recon 0 \
  --lambda-continuous-code-recon 100 \
  --freeze-receiver-decoder
```

实际使用时 `SENDER` 路径必须是一个完整参数。hard 路线使用 `--rollout-q-mode hard`，其前向 q2_hat 是精确码本 lookup；continuous 路线则必须让训练温度和 rollout 温度一致。

## 8. 原始实验清单

以下是目录中所有日志/结果的分组索引；原始数值应以对应 `.log` 和 `.json` 为准。

### CVQ CNN

- K1024：`cvq-cnn-f16-c16plus240-k1024-e200-v1`、`...-b8-v2`、`...-b8t8-v3`。
- D256 q2 probe：`cvq-cnn-f16-d256-v1`、`...-usage-v2`、`...-anchored-v4`、`...-relevance-v3`、`...-z1q-v5`。
- K4096 paper：`cvq-cnn-f16-c16plus240-k4096-paper-alpha05-resdelta-warm10-b32-e200-v1`。
- K4096 conditional：`...-z1condrd-freezedec-e80-v1`、`cvq-cnn-f16-k4096-z1condrd-smoke-v1/v2`、`...-resume-smoke-v1`、`...-z1priorcal-e55-v1`。
- K4096 MI：`cvq-cnn-f16-c16plus240-k4096-z1mi-balanced-e100-v2`。
- K16384：`...-paper-alpha05-e200-v5`、`...-resdelta-gate01-b32-e200-v6`、`...-resdelta-warm10-b32-e200-v7`、`...-e200-b8t8-alpha05-direct1db-v4`。
- D1024：`cvq-cnn-f32-c16plus240-k16384-paper-alpha05-resdelta-warm10-b16-acc4-e200-v1`（未完成）。

### CVQ Swin

- D256 probe：`cvq-swin-f16-d256-v1`、`...-usage-v2`、`...-anchored-v4`、`...-relevance-v3`、`...-z1q-v5`。
- K1024：`cvq-swin-f16-c16plus240-k1024-e200-v1`、`...-b6-v2`、`...-b6t6-v3`。
- fresh/source-E2：`...fresh-paper-alpha05-warm10-b16-e200-v1`、`...fresh-gate10-direct1db-warm10-b24-e200-v2`、`...sourcee2-paper-alpha05-resdelta-warm10-b24t4-e200-v10`、`...sourcee2-rms-paper-alpha05-resdelta-warm10-b24t4-e200-v11`。
- source-comb/PCA：`...sourcecomb-pca-e2e-b16a4-e60-v3`、`...sourcecomb-pca65536-teacher-warm5-e40-v1/v2`、`...sourcecomb-pca65536-preflight-v1`、`...sourcecomb-pca-smoke-v2`、`...sourcecomb-teacher-smoke-v1`。
- K16384 regular/gated：`...paper-alpha05-e200-v5`、`...resdelta-gate01-b24-e200-v6`、`...resdelta-warm10-b24-e200-v7`、`...resdelta-warm10-b32-acc8-e200-v8`、`...paper-resgain01-warm20-b32t4-e200-v9`、`...fresh-gate10-direct1db-warm10-b24-e200-v2`。

### CAR / Qwen

- PaperCAR：`car-cnn-f16-c240-k16384-paper-freezedec-b4-e80-v1`、`...postmean-t02-freezedec-b4-e80-v2`、`car-cnn-f16-c240-k4096-paper-hard-freezedec-b4-e80-v3`。
- Qwen Stage-I 基线：`qwen3-4b-z1-k4096-stage1-e40-b4t8-v2`、`...stage1-e5-b1-v1`、`...stage1-smoke-v1/v2`、`...e20-teacherhistory-eval-v1`。
- Qwen conditional-RD：`...condrd-e40-stage1-e40-v1`、`...condrd-e40-stage1-smoke-v1`、`...condrd-e25-contmean-rd100-t1-e50-v1`、`...condrd-e25-hardst-rd100-e50-v1`、`...condrd-e30-contfeedback-rd100-t1-e55-v1`、`...condrd-e30-contfeedbackfp32-rd100-t1-e55-v2`。
- MI-balanced Qwen：`qwen3-4b-z1-k4096-mi-balanced-e90-stage1-e40-v1`。
- scheduled sampling：`...scheduled-smoke-v1`、`...scheduled-p50-s16-e8-e20-v1`、`...scheduled-grad-smoke-v1`、`...scheduled-grad-b4-smoke-v1`、`...scheduled-grad-s32-b2-smoke-v1`、`...scheduled-grad-s64-b4-smoke-v1`、`...scheduled-grad-p50-s32-e40-v2`。
- true continuous smoke：`qwen3-contfeedback-fp32-smoke-v1`、`qwen3-contfeedback-fp32-smoke-v2`。
- Stage-II probes：`...stage2-smoke-b1-v1`（保存失败的 partial 已清理）、`...stage2-probe-b4-v1`、`...stage2-probe-b4a8-lr2e6-v1`、`...stage2-fullmetrics-e50-b4a8-lr2e6-v1`（e46 后停止、metrics-only）。

## 9. 最终状态

截至本文档更新时间，没有 CAR 版本满足严格 `PSNR(x2_hat)-PSNR(x1)>=+0.5 dB` 且 z1 condition shuffle gate。最接近的可复现实验是 CNN K4096 conditional-RD sender + Qwen3-4B true continuous feedback；它的 oracle 有约 +0.93 dB，但真实 receiver 仍为负增益。

下一步应优先改进 CVQ 阶段的 `H(I_k|z1,I_<k)`，而不是继续扩大 Qwen 或重复 scheduled sampling smoke。可保留论文的 nested channel dropout，并在保持 per-channel marginal entropy/码本健康的约束下提高 conditional-rate 目标；任何新 CAR 必须从完整 valid100 和 z1 condition-shuffle 审计开始。
