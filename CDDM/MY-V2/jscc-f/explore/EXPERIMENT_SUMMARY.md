# MY-V2/jscc-f/explore FSQ 实验与探索总结

> 更新时间：2026-07-12（Asia/Shanghai）
> 范围：`MY-V2/jscc-f/explore` 中与 CNN/Swin Layer2-FSQ、码本容量、有效表达、无 u2 teacher 和增益上限相关的训练、对照、探针及诊断。
> 状态：共享 exact multirate、共享 bitplane、独立 K729 均已完成 epoch 100；独立 K125/K4913 分别在 epoch 25/35 主动暂停且可续训；当前无本探索相关训练进程。

本文件给出结论与实验全貌。容量上限术语的严格定义另见 [CAPACITY_CEILING.md](./CAPACITY_CEILING.md)。

## 1. 结论摘要

1. **CNN 目标已达成。** 在无 u2 teacher、无 usage/KL 的 direct Layer2-FSQ 路线上，epoch 100 的 `K=125/729/4913` 得到：
   `22.876459 < 22.905375 < 22.947125 dB`，相对 `x1=21.771972 dB` 的增益为
   `+1.104487/+1.133402/+1.175153 dB`。从 epoch 50 到 100 的 11 次验证全部严格单调。

2. **Swin 的三个主目标已经在 residual 路线上同时成立。** 精确共享 `K=125/729/4913` 模型在 epoch 100 得到：
   `27.722591 < 27.747599 < 27.756767 dB`，三档都高于 `x1=27.565002 dB`，并且 zero/shuffle 消融均有效。epoch 10–100 的 19 次验证全部严格单调。

3. **增加离散维度比只提高 d=3 标量精度更有效。** 共享 bitplane residual 模型的最佳 epoch 45 在 `K=128/1024/8192` 上得到：
   `27.777931 < 27.867153 < 27.951080 dB`，增益为
   `+0.212929/+0.302151/+0.386078 dB`。全部 20 次验证都在每张图上严格递增。

4. **高 usage 不等于有效信息。** 历史 Swin Stage3 中，KL 可把 K4096 的 used/PPL 提高到 `3790/2503.90`，但早期 zero/shuffle drop 只有 `0.0037/0.0026 dB`。usage 只能作辅助指标，不能替代代码消融。

5. **Swin 的主要瓶颈不是 rounding，也不是 encoder 选码。** continuous-d3、width320 adapter 和容量探针共同表明：主要余量在有限维 latent 的表示、D2 和 combiner 优化。最终 epoch-45 五图探针中，同 K hard 搜索只比实际编码器高 `0.0001–0.0014 dB`；continuous relaxation 仍多出 `0.11–0.24 dB`，任意 u2 relaxation 可达约 `+4.60 dB`。

6. **去掉 u2 teacher 本身是可行的。** 当前成功的 CNN/Swin residual 路线均没有 u2 teacher；原始 Swin320 Layer2 的 `lambda_u2=0`，仍能达到 `42.273165 dB`。失败发生在把 `320x16x16` 连续 latent 压到低维、有限码率之后，而不是因为没有 teacher loss。

7. **原始 Swin320 的 `+14.708 dB` 只是宽架构参考。** 它使用 `320x16x16` 连续 latent，不能称作 K125/K729/K4913 的同码率上限。

## 2. 原始问题与目标

### 2.1 初始观察

- Swin 中把 GroupNorm 换成 BatchNorm 后，码本不再立刻塌陷，但利用率仍低。
- 加入 KL-usage 后 used/PPL 明显上升，但高 usage 不一定表达图像信息。
- CNN 中增加码本容量时，早期 Stage3 路线没有稳定带来更高 `psnr_final`。
- Swin 的 Layer2 combiner 同时接收 `x1` 和 Layer2 解码输出，理论上不应只有极小增益。

### 2.2 最终目标

1. CNN：FSQ 容量增大时 `psnr_final` 增大。
2. Swin：首先满足 `psnr_final > psnr_x1`；随后容量增大时 `psnr_final` 增大。
3. 码本必须具有有效表达力，而非只有高 usage。
4. 不使用 u2 teacher，直接在 Layer2 中使用 FSQ。
5. 对每档容量报告实际增益、同 K 可达包络及更松的 continuous/u2 relaxation，定位增益上限和未利用空间。

## 3. 统一实验口径

### 3.1 数据与基础模型

- 验证集：DIV2K valid，100 张图。
- Layer1 固定；`x1` 是所有 Layer2 实验共同基线。
- CNN 基线：`psnr_x1=21.771972 dB`。
- Swin 基线：`psnr_x1≈27.565002 dB`。
- 原始 Swin320 Layer2：`psnr_final=42.273165 dB`，增益 `+14.708163 dB`，latent 为 `320x16x16`。
- 所有新 Layer2 正式 launcher 使用 `num_workers=16`、`val_num_workers=4`。
- Swin BatchNorm 正式验证使用 64 个训练 batch 重新校准 running statistics。
- 当前训练 sweep 使用单一 seed `20260624`；结果还不是多 seed 统计结论。

### 3.2 容量设置

固定 `d=3`、提高标量精度：

| 标量 level | 码本容量 K | 固定 bits/token |
|---:|---:|---:|
| 5 | 125 | 7 |
| 9 | 729 | 10 |
| 17 | 4913 | 13 |

增加二值维度：

| active binary dimensions | 码本容量 K | bits/token |
|---:|---:|---:|
| 7 | 128 | 7 |
| 10 | 1024 | 10 |
| 13 | 8192 | 13 |

`L=5/9/17` 的标量网格严格嵌套；bitplane 的低码率代码也可通过追加 inactive bits 精确嵌入高码率代码集。

### 3.3 成功门槛

单档必须同时满足：

- `delta_x1 = psnr_final - psnr_x1 > 0`；
- `drop_zero = psnr_final - psnr_zero >= 0.1 dB`；
- `drop_shuffle = psnr_final - psnr_shuffle >= 0.1 dB`。

多档还检查：

- 验证集均值严格随容量递增；
- 每张图严格递增的比例；
- 相邻容量的 PSNR gain；
- PPL、used codes 仅作为辅助，不作为“有效表达”的充分条件。

## 4. 历史 Stage3 teacher-constrained 探索

入口为 [train_stage3_fsq_explore.py](./train_stage3_fsq_explore.py)，分析器为 [analyze_fsq_tokenizer.py](./analyze_fsq_tokenizer.py)。这批实验用于研究归一化、反塌陷和 usage objective，不是当前无 teacher 主路线。

### 4.1 代表性最佳 checkpoint

| 架构/设置 | K | epoch | final | x1 | used / PPL | drop0 / shuffle | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| CNN L20x16x16, usage=1e-3 | 5120 | 60 | 22.292128 | 21.771972 | 5041 / 4434.68 | 2.6383 / 2.5379 | 已超过 x1 且代码有效，但仍是 teacher-constrained 路线 |
| Swin L8, BatchNorm, no usage | 512 | 100 | 25.765914 | 27.565002 | 60 / 29.33 | 1.2304 / 1.2282 | 有效但容量利用低，final 仍低于 x1 |
| Swin L8, BatchNorm, usage=5e-3 | 512 | 55 | 25.789511 | 27.565002 | 512 / 487.28 | 1.3996 / 1.1803 | usage 提高，绝对 PSNR 几乎没改善 |
| Swin L16, BatchNorm, no usage | 4096 | 35 | 25.640775 | 27.565002 | 208 / 59.87 | 1.1431 / 1.2832 | 高容量没有带来更高 final |
| Swin L16, BatchNorm, usage=5e-3 | 4096 | 60 | 25.229007 | 27.565002 | 4062 / 3637.33 | 0.9996 / 1.0786 | 几乎满用，但 final 更低 |

### 4.2 公平 KL 早期对照

K4096、epoch 5：

| 设置 | final | used / PPL | drop0 | drop shuffle |
|---|---:|---:|---:|---:|
| 无 usage | 21.615195 | 509 / 100.08 | -1.4671 | 0.3269 |
| uniform-KL, 5e-3 | 23.670325 | 3790 / 2503.90 | 0.0037 | 0.0026 |

KL 确实让更多 code 被使用，但几乎无法通过 zero/shuffle 证明这些 code 在传递图像信息。这是“高 usage 不能代表有效表达”的直接证据。

### 4.3 空间信息探针

| checkpoint | final | x1 | unique tokens/image | token spatial change | drop0 / shuffle |
|---|---:|---:|---:|---:|---:|
| Swin L4，20 图 | 26.605173 | 28.128236 | 7.75 | 0.4535 | 1.4714 / 1.7662 |
| Swin L8 塌陷点，20 图 | 25.258064 | 28.128236 | 1.00 | 0.0000 | 0.0669 / 0.0000 |
| CNN L20 最佳点，20 图 | 23.163769 | 21.442874 | 174.8 | 0.9705 | 2.3257 / 3.7429 |

### 4.4 历史路线结论

- BatchNorm 能延缓或避免快速塌陷，但必须正确校准 running statistics。
- usage/KL 可以提高 PPL 和 used codes，却不能保证 `final>x1` 或代码相关性。
- Swin Stage3 teacher-constrained 训练长期低于 x1，因此后续转向“直接 Layer2-FSQ + final reconstruction loss”。
- 部分旧 launcher 名称与实际参数不一致，历史结果必须以 checkpoint 内嵌 `args` 为准。

## 5. 无 u2 teacher 的 direct Layer2-FSQ

核心计算图：

```text
image -> frozen E1/D1 -> x1
concat(x1, image) -> E2 -> FSQ -> D2 -> combiner(x1, u2_hat) -> final
```

训练只使用 `MSE(final, image)`；不调用 source u2 teacher，不使用 u2 image loss，也不使用 usage/KL。

主要入口是 [train_layer2_fsq_direct.py](./train_layer2_fsq_direct.py) 和 [run_layer2_fsq_direct_nested.sh](./run_layer2_fsq_direct_nested.sh)。

## 6. CNN direct nested：目标达成

设置：GroupNorm、compatible init、SafeBlend、d=3、单 seed、epoch 100。

### 6.1 epoch 100 公平比较

| K | final | gain over x1 | continuous | quant gap | drop0 | drop shuffle | used / PPL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 125 | 22.876459 | +1.104487 | 22.899974 | 0.023515 | 1.449481 | 2.288306 | 125 / 92.94 |
| 729 | 22.905375 | +1.133402 | 22.928278 | 0.022903 | 2.580539 | 2.962144 | 641 / 261.05 |
| 4913 | 22.947125 | +1.175153 | 22.963377 | 0.016252 | 2.857201 | 3.166313 | 1921 / 624.39 |

### 6.2 稳定性

- 20 个共同验证点中有 15/20 严格满足 K125 < K729 < K4913。
- epoch 50–100 的 11/11 个验证点连续严格单调。
- 20/20 个验证点三档都高于 x1，并通过 zero/shuffle 门槛。
- 最佳：K125 epoch100，K729 epoch100，K4913 epoch95（22.947396 dB）。

结论：CNN 目标在当前 seed 下已达成。限制是 K729/K4913 没有满用，且尚未做多 seed 复现。

## 7. Swin direct d=3：失败路线与定位

### 7.1 compatible init 的隐藏结构问题

Swin source Layer2 使用 320 通道 latent。direct d=3 重建模型虽然参数匹配率接近 100%，但最关键的两个头无法从 source 继承：

- E2 的 `320 -> 3` head；
- D2 的 `3 -> 320` head。

这两个头随机初始化，导致“compatible init”并没有保住 source 的完整 Layer2 表示合同。

### 7.2 BatchNorm + SafeBlend

完整 cal64 clean 结果：

| K | final@100 | gain | continuous | quant gap | drop0 / shuffle | used / PPL | alpha |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 125 | 27.751608 | +0.186606 | 27.769592 | 0.017983 | 0.1931 / 0.3284 | 125 / 71.55 | 0.10399 |
| 729 | 27.746336 | +0.181334 | 27.755888 | 0.009552 | 0.1870 / 0.3245 | 729 / 358.37 | 0.10509 |
| 4913 | 27.730556 | +0.165554 | 27.732454 | 0.001898 | 0.1716 / 0.2740 | 4068 / 2026.78 | 0.10693 |

- 20/20 个验证点都高于 x1 且代码有效。
- 只有 10/20 个共同验证点容量严格单调；epoch 100 反而随 K 下降。
- quantization gap 随 K 变小，但 final 没有上升，因此 rounding 不是主瓶颈。

### 7.3 blend-alpha 上限探针

| K | learned gain | best global-alpha gain | per-image alpha oracle | alpha=1 gain |
|---:|---:|---:|---:|---:|
| 125 | +0.186572 | +0.186527 | +0.189274 | -6.052642 |
| 729 | +0.195947 | +0.196770 | +0.200535 | -6.573134 |
| 4913 | +0.192499 | +0.192526 | +0.194961 | -6.193298 |

小 alpha 并没有隐藏一个强 Layer2 输出；codec 已与 `alpha≈0.1` 共适应，强行设 alpha=1 会灾难性下降。

### 7.4 original combiner direct

三档训练到 epoch 400 并继续到约 420–450：

- 共同 epoch 5–420 中，仅 21/84 个点容量严格。
- 仅 2/84 个点三档全部正增益。
- 只有 epoch 365 同时满足“全部正增益 + 容量严格”：

| K | final@365 | gain | continuous | drop0 / shuffle |
|---:|---:|---:|---:|---:|
| 125 | 27.585990 | +0.020989 | 27.594917 | 0.3085 / 0.3066 |
| 729 | 27.589983 | +0.024981 | 27.599592 | 0.2711 / 0.3354 |
| 4913 | 27.605229 | +0.040226 | 27.611629 | 0.2951 / 0.3683 |

去掉 blend 上限没有自动解决问题，训练波动大，偶发成功点的绝对增益也只有 0.02–0.04 dB。

## 8. Swin 表示能力对照

| 路线 | 最佳 epoch | 最佳 final | gain over x1 | 关键结论 |
|---|---:|---:|---:|---|
| continuous d=3，original combiner | 75 | 27.245272 | -0.319731 | 完全去掉 rounding 仍低于 x1 |
| exact width320 + PCA3 adapter + original | 90 | 27.340823 | -0.224179 | 保留 320 宽合同有改善，但 d=3 仍不足 |
| shared bitplane original，K8192 | 90 | 27.487404 | -0.077598 | 增加维度产生稳定容量收益，但 original combiner 仍没越过 x1 |

original shared bitplane 的三档在 20/20 个验证时刻都严格递增且消融有效，但 0/20 时刻有任一档超过 x1。它证明容量规律已经存在，绝对基线仍受 combiner 约束。

PCA 线性基准也支持这一诊断：冻结 teacher latent 的 PCA3 仅解释约 39.71% 方差；full-val continuous 只有 26.048557 dB，L5/L9/L17 为 25.640390/25.936726/26.011272 dB。该基准优化 latent MSE，不是图像 PSNR 上限。

## 9. identity-safe residual combiner

入口为 [train_layer2_fsq_adapter.py](./train_layer2_fsq_adapter.py)。设计为：

```text
correction = Conv-PReLU-Conv(concat(x1, u2_hat))
final = clip(x1 + correction, 0, 1)
```

- 复制 source combiner 的第一层 Conv 和 PReLU。
- 最后一层 Conv 零初始化，因此初始 `final == x1`，最大绝对误差实测为 0。
- 第一步先更新输出层；一次更新后梯度可进入 D2、adapter 和 E2。
- 没有全局 alpha ceiling，也不会在初始化时破坏 x1。

### 9.1 独立 K sweep：共同 epoch 25

PCA3 解释率约 54.50%，三档在共同 epoch 5/10/15/20/25 全部严格递增。

| K | final@25 | gain | drop0 | drop shuffle | used / PPL | 状态 |
|---:|---:|---:|---:|---:|---:|---|
| 125 | 27.712183 | +0.147181 | 0.135635 | 0.230235 | 125 / 31.40 | 主动暂停于 epoch25，可续训 |
| 729 | 27.725236 | +0.160234 | 0.148904 | 0.259966 | 691 / 101.83 | 后续完成 epoch100 |
| 4913 | 27.734938 | +0.169936 | 0.159778 | 0.289133 | 2595 / 411.80 | 主动暂停于 epoch35，可续训 |

K729 的 epoch100 最终值为 27.749004 dB，增益 +0.184002 dB。独立模型的不同终止 epoch 不应混作公平容量比较，因此正式结论以共享模型为准。

## 10. Swin 精确共享 K125/K729/K4913：正式主结果

入口为 [train_layer2_fsq_adapter_multirate.py](./train_layer2_fsq_adapter_multirate.py) 和 [run_layer2_fsq_adapter_multirate.sh](./run_layer2_fsq_adapter_multirate.sh)。

关键控制：

- exact source E2_320/D2_320；
- PCA `320 -> 3 -> 320` adapter；
- 同一个 z_norm、synthesis adapter、D2 和 residual combiner；
- L5/L9/L17 嵌套网格；
- mean reconstruction + monotonic hinge；
- 无 u2 teacher、无 usage/KL。

### 10.1 epoch 100

| K | final | gain | continuous | quant gap | drop0 | drop shuffle | used / PPL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 125 | 27.722591 | +0.157589 | 27.760280 | 0.037689 | 0.143052 | 0.292158 | 125 / 21.41 |
| 729 | 27.747599 | +0.182597 | 27.760280 | 0.012681 | 0.168060 | 0.338865 | 711 / 130.38 |
| 4913 | 27.756767 | +0.191764 | 27.760280 | 0.003513 | 0.177227 | 0.346813 | 3428 / 863.12 |

相邻 gain 为 `+0.025008/+0.009168 dB`。

### 10.2 稳定性

- 20 次验证中 19/20 均值严格递增；只有 epoch 5 的 L17 比 L9 低 0.000145 dB。
- epoch 10–100 全部 strict 且 goal eligible。
- epoch 100 的每图 strict ratio 为 1.0。
- 20/20 个验证点三档均高于 x1，zero/shuffle 均过门槛。
- best=goal_best=latest=epoch100。

这是“固定 d=3，只增加码本精度”下最严格、最公平的容量证据。其 K125→K4913 总收益只有 0.034176 dB，说明 d=3 本身仍是强瓶颈。

## 11. Swin shared bitplane residual：当前最高增益

入口为 [train_layer2_fsq_bitplane_multirate.py](./train_layer2_fsq_bitplane_multirate.py) 和 [run_layer2_fsq_bitplane_multirate.sh](./run_layer2_fsq_bitplane_multirate.sh)。

注意：bitplane launcher 默认 `ADAPTER_COMBINER=original`；重现成功组必须显式设置 `ADAPTER_COMBINER=residual`。

### 11.1 最佳 checkpoint：epoch 45

| bits / K | final | gain | drop0 | drop shuffle | used / PPL |
|---:|---:|---:|---:|---:|---:|
| 7 / 128 | 27.777931 | +0.212929 | 1.101498 | 0.377645 | 128 / 99.00 |
| 10 / 1024 | 27.867153 | +0.302151 | 1.190720 | 0.523744 | 1024 / 713.26 |
| 13 / 8192 | 27.951080 | +0.386078 | 1.274647 | 0.670223 | 6980 / 4626.85 |

### 11.2 epoch 100 最终点

| bits / K | final | gain | continuous | gap | drop0 / shuffle |
|---:|---:|---:|---:|---:|---:|
| 7 / 128 | 27.765957 | +0.200955 | 27.790113 | 0.024156 | 0.211530 / 0.370511 |
| 10 / 1024 | 27.857262 | +0.292260 | 27.925544 | 0.068283 | 0.302835 / 0.511513 |
| 13 / 8192 | 27.943630 | +0.378628 | 28.047877 | 0.104247 | 0.389203 / 0.653610 |

### 11.3 稳定性

- 20/20 个验证时刻均值严格递增。
- 20/20 个验证时刻 `all_rates_strict_ratio=1.0`，即每张图都严格递增。
- 20/20 个验证时刻三档都高于 x1 且代码有效。
- epoch100 的 K128→K8192 容量收益为 0.177673 dB，明显大于固定 d=3 exact sweep 的 0.034176 dB。

结论：新增离散维度比在固定三个标量上继续加细精度更能转化为图像增益。

## 12. 增益上限与可达包络

### 12.1 术语

对冻结 decoder/combiner，严格的同 K 最优代码组合无法穷举。文中使用：

- `Actual_K`：encoder 实际选择的 hard code；
- `HardOracleSearch_K`：从实际 code 出发，在同一个 hard grid 上做有限步搜索；它只是可达下界，不是数学上限；
- `ContinuousLatentRelax`：移除 rounding，仍保留 latent 维度和冻结 decoder；
- `ArbitraryU2Relax`：直接优化进入 combiner 的 u2，去掉 latent/rate/decoder 约束；
- `Source320Reference`：宽连续架构参考，不是同 K 上限。

### 12.2 旧 separate blend checkpoint：20 图

| K | actual gain | hard reachable | continuous reachable | arbitrary-u2 reachable | actual / hard |
|---:|---:|---:|---:|---:|---:|
| 125 | +0.183647 | +0.196654 | +0.246912 | +4.894742 | 93.39% |
| 729 | +0.177516 | +0.215299 | +0.240518 | +4.931150 | 82.45% |
| 4913 | +0.168510 | +0.223370 | +0.230922 | +4.997931 | 75.44% |

hard reachable 随 K 增长，但实际增益下降，说明训练没有利用增加的容量。三个 K 使用不同 checkpoint/decoder，因此这只是 decoder-specific 诊断，不能作严格容量因果结论。

### 12.3 residual bitplane 最佳 epoch 45：5 图最终探针

验证 indices 为 `0,25,50,75,99`；hard/continuous/u2 steps 为 `50/50/75`。结果保存在
[results-capacity-ceiling/bitplane-residual-goalbest-e45-5img-h50-c50-u75.json](./results-capacity-ceiling/bitplane-residual-goalbest-e45-5img-h50-c50-u75.json)。

| K | actual gain | hard reachable | hard headroom | continuous reachable | arbitrary-u2 reachable | actual / hard |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | +0.198518 | +0.198639 | 0.000121 | +0.309641 | +4.598966 | 99.94% |
| 1024 | +0.285057 | +0.285859 | 0.000802 | +0.466233 | +4.602802 | 99.72% |
| 8192 | +0.361827 | +0.363237 | 0.001409 | +0.604584 | +4.606458 | 99.61% |

- actual 与 hard reachable 在均值及 5/5 张图上都严格递增。
- actual 相邻增益为 `+0.086539/+0.076770 dB`。
- hard reachable 相邻增益为 `+0.087220/+0.077378 dB`。
- 三档 zero/shuffle 均有效。

解读：encoder 选码已接近当前冻结 decoder 的局部 hard reachable 包络；继续只改选码几乎没有空间。更大的余量位于 continuous latent、D2 和 combiner。`actual/hard≈100%` 不能解释为达到数学上限，因为 hard search 本身只是有限步可达下界。

### 12.4 source320 参考

| epoch | source final | gain over x1 |
|---:|---:|---:|
| 50 | 28.404362 | +0.839360 |
| 100 | 31.931154 | +4.366151 |
| 200 | 35.857008 | +8.292006 |
| 400 | 39.443646 | +11.878643 |
| 800 | 42.273165 | +14.708163 |

这证明 Layer2/combiner 能使用大量附加信息；但它的 320 通道连续 latent 与低码率 FSQ 不同，不能作为任何具体 K 的上限。

## 13. 因果诊断总结

目前证据形成了较一致的因果链：

1. GroupNorm→BatchNorm 解决了部分训练稳定性，但不自动赋予代码有效语义。
2. usage/KL 可提升利用率，却不能保证 zero/shuffle relevance 或最终 PSNR。
3. direct d=3 的 source-compatible 初始化缺失关键 320→3/3→320 heads。
4. continuous-d3 仍低于 x1，说明 rounding 不是主问题。
5. 保留 exact width320 E2/D2 并插入 adapter 有改善，但 original combiner 仍低于 x1。
6. identity-safe residual combiner 保住 x1，并使梯度逐步进入 D2/adapter/E2。
7. shared exact d=3 首次稳定满足 K125<K729<K4913，但增益差较小。
8. shared bitplane 通过增加表达维度获得更大、更稳定的容量收益。
9. hard-search headroom 极小，而 continuous/u2 headroom 大，后续重点应是 latent 维度、decoder 和 residual correction 的 rate-distortion 优化，不是单纯追高 usage。

## 14. 目标完成状态

| 目标 | 状态 | 证据 |
|---|---|---|
| CNN：容量增加，final 增加 | 已完成（单 seed） | epoch50–100 连续 11/11 严格；epoch100 三档为 22.876459/22.905375/22.947125 |
| Swin：final > x1 | 已完成 | exact shared 与 bitplane residual 全部验证点三档均高于 x1 |
| Swin：容量增加，final 增加 | 已完成 | exact shared epoch10–100 全严格；bitplane 20/20 且逐图 100% 严格 |
| 码本表达有效信息 | 已完成 | 所有成熟 residual 档位 zero/shuffle drop 均 >=0.1 dB |
| 不使用 u2 teacher，直接 Layer2 FSQ | 已完成 | 所有 direct/adapter/residual 主结果均为 `lambda_u2_img=0`、`lambda_usage=0` |
| 找到相应容量的增益上限 | 已建立可达包络，非数学上限 | hard/continuous/u2 ladder 已实现；epoch45 五图最终 probe 已完成 |
| 多 seed 统计 | 未完成 | 当前主要 sweep 只有 seed 20260624 |

## 15. 代码与入口索引

### 15.1 正式主路线

- [train_layer2_fsq_direct.py](./train_layer2_fsq_direct.py)：CNN direct Layer2-FSQ；Swin 旧 direct 诊断也由此入口运行。
- [run_layer2_fsq_direct_nested.sh](./run_layer2_fsq_direct_nested.sh)：单 K nested launcher，默认 workers 16/4。
- [train_layer2_fsq_adapter_multirate.py](./train_layer2_fsq_adapter_multirate.py)：Swin exact width320、共享 K125/K729/K4913 主路线。
- [run_layer2_fsq_adapter_multirate.sh](./run_layer2_fsq_adapter_multirate.sh)：默认 residual、PCA3、BN64、workers 16/4。
- [train_layer2_fsq_bitplane_multirate.py](./train_layer2_fsq_bitplane_multirate.py)：共享 7/10/13-bit dimension-growing 对照。
- [run_layer2_fsq_bitplane_multirate.sh](./run_layer2_fsq_bitplane_multirate.sh)：重现成功组时显式设 `ADAPTER_COMBINER=residual`。

### 15.2 对照路线

- [train_layer2_fsq_adapter.py](./train_layer2_fsq_adapter.py)：单 K width320 adapter，支持 original/residual。
- [run_layer2_fsq_adapter.sh](./run_layer2_fsq_adapter.sh)：默认 original；residual 实验需显式设置。
- [train_layer2_continuous_direct.py](./train_layer2_continuous_direct.py)：d=3 continuous/no-rounding 对照。
- [train_layer2_fsq_multirate.py](./train_layer2_fsq_multirate.py)：旧 shared direct d=3 control，已被 width320 adapter multirate 取代。
- [train_stage3_fsq_explore.py](./train_stage3_fsq_explore.py)：历史 teacher-constrained normalization/usage 探索。

### 15.3 分析与上限探针

- [probe_layer2_fsq_adapter_capacity_ceiling.py](./probe_layer2_fsq_adapter_capacity_ceiling.py)：当前正式 adapter/adapter-multirate/bitplane 上限阶梯。
- [probe_layer2_fsq_capacity_ceiling.py](./probe_layer2_fsq_capacity_ceiling.py)：旧 direct-only 容量阶梯。
- [probe_layer2_fsq_gain_ceiling.py](./probe_layer2_fsq_gain_ceiling.py)：blend alpha envelope。
- [analyze_fsq_tokenizer.py](./analyze_fsq_tokenizer.py)：历史 Stage3 空间信息与 zero/shuffle 分析。
- [rank_fsq_checkpoints.py](./rank_fsq_checkpoints.py)：历史 Stage3 排名。
- [rank_layer2_fsq_direct.py](./rank_layer2_fsq_direct.py)：direct Layer2 checkpoint 排名。

### 15.4 历史 launcher 清单

CNN Stage3 usage：

- `run_cnn_l20_usage_gpu1.sh`
- `run_cnn_l20_usage_e100_gpu1.sh`
- `run_cnn_l20_usage_gpu2.sh`

Swin BatchNorm/usage：

- `run_swin_l8_batch_only_gpu0.sh`
- `run_swin_l8_batch_usage_gpu0.sh`
- `run_swin_l8_usage_batch_gpu1.sh`
- `run_swin_l8_usage_group_gpu0.sh`

Swin 公平 KL/entropy-floor/stateless/dimension controls：

- `run_swin_l16_fair_b12_no_usage_gpu0.sh`
- `run_swin_l16_fair_b12_uniform_kl_gpu1.sh`
- `run_swin_l16_floor_b12_batch_gpu0.sh`
- `run_swin_l16_floor_b12_batch_stateless_gpu0.sh`
- `run_swin_d4_l8_floor_b12_batch_stateless_gpu0.sh`

这些 launcher 属于历史反塌陷/usage 探索；部分只完成 smoke、早期 epoch 或曾中断，未被提升为当前主路线。另有两个命名陷阱：`run_swin_l8_batch_only_gpu0.sh` 名称带 e10、实际配置为 400 epochs；`run_swin_l8_batch_usage_gpu0.sh` 名称带 l8/e10、当前实际为 L16/200 epochs。必须以 checkpoint 内嵌 args 为准。

## 16. 主要 artifact 位置

- 历史 Stage3：`checkpoints/`、`logs/`。
- direct CNN/Swin：`checkpoints-direct/`、`logs-direct/`。
- continuous：`checkpoints-continuous/`、`logs-continuous/`。
- 单 K adapter：`checkpoints-adapter/`、`logs-adapter/`。
- exact shared adapter：`checkpoints-adapter-multirate/`、`logs-adapter-multirate/`。
- bitplane：`checkpoints-bitplane/`、`logs-bitplane/`。
- blend alpha 结果：`results-gain-ceiling/`。
- capacity ladder：`results-capacity-ceiling/`。
- 历史 probe JSON：`cnn_l20_best_probe_20.json`、`swin_l4_best_probe*.json`、`swin_l8_best_probe_20.json`、`fsq_swin_current_rank.json`。

## 17. 复现命令

### 17.1 CNN K sweep

```bash
for levels in 5,5,5 9,9,9 17,17,17; do
  GPU=0 ARCH=cnn LEVELS="$levels" NORMALIZER=group \
    COMBINER_MODE=blend NUM_WORKERS=16 VAL_NUM_WORKERS=4 \
    bash MY-V2/jscc-f/explore/run_layer2_fsq_direct_nested.sh
done
```

实际并行运行时应为每个 K 指定不同 GPU、VERSION 和 SAVE_DIR。

### 17.2 Swin exact K125/K729/K4913

```bash
GPU=0 NUM_WORKERS=16 VAL_NUM_WORKERS=4 \
  bash MY-V2/jscc-f/explore/run_layer2_fsq_adapter_multirate.sh
```

### 17.3 Swin dimension-growing bitplane residual

```bash
GPU=0 ADAPTER_COMBINER=residual NUM_WORKERS=16 VAL_NUM_WORKERS=4 \
  bash MY-V2/jscc-f/explore/run_layer2_fsq_bitplane_multirate.sh
```

### 17.4 最终 bitplane capacity probe

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n cddm_ddnm python -u \
  MY-V2/jscc-f/explore/probe_layer2_fsq_adapter_capacity_ceiling.py \
  --checkpoint MY-V2/jscc-f/explore/checkpoints-bitplane/direct-swin-bitplane-b7x10x13-batch-pca-h320-residual-e100/jscc_f_direct-swin-bitplane-b7x10x13-batch-pca-h320-residual-e100_layer2_fsq_bitplane_swin_b7-10-13_batch_pca_h320_residual_goal_best.pth \
  --val-indices 0,25,50,75,99 \
  --hard-steps 50 --continuous-steps 50 --u2-steps 75 \
  --num-workers 16 --val-num-workers 4 --bn-calibration-batches 0 \
  --output-json MY-V2/jscc-f/explore/results-capacity-ceiling/bitplane-residual-goalbest-e45-5img-h50-c50-u75.json
```

## 18. 已完成的工程验证

- 相关 Python 入口已通过 `py_compile` 和 `--help`。
- 相关 shell launcher 已通过 `bash -n`。
- CNN/Swin direct 已做 shape smoke、one-batch train/val。
- continuous、adapter、bitplane、direct-multirate 已做 CPU shape/forward/backward smoke。
- exact adapter-multirate 做过真实 source Swin CPU 三档 forward + 一次 backward；352 个参数张量获得有限梯度。
- residual combiner 初始化实测 `final==x1`，更新后梯度可进入完整 Layer2 路径。
- checkpoint/resume 保存 RNG；adapter/residual 路线会检查 combiner 和验证协议，改变 BN/选模协议时需显式 reset best。
- 完整 epoch100 训练进一步验证了 best/goal_best/latest、BN calibration 和验证消融路径。

## 19. 限制与下一步

1. 对 CNN exact shared 和 Swin residual 主结果补 2–3 个 seed，报告均值与方差。
2. 在 epoch45 bitplane best 上把 capacity probe 从 5 图扩展到 20 或 full validation，并增加 hard-search restart；仍需称为 reachable envelope。
3. 对 exact K125/K729/K4913 的 epoch100 goal_best 运行同一 capacity ladder，比较 actual 与同 K hard envelope 是否也严格递增。
4. 继续探索在相同 bits/token 下增加有效维度、减少无效标量精度的 factorization。
5. 研究更强但 identity-safe 的 residual decoder/combiner，使 continuous/u2 headroom 转化为实际增益。
6. 独立 K125/K4913 可分别从 epoch25/35 的 latest 续训，但公平容量结论优先使用已经完成的共享 exact 模型。
7. 不再把 PPL/used codes 作为单独成功标准；所有新实验继续保留 final>x1、zero/shuffle 和逐图单调门槛。

## 20. 参考资料

- [ICLR-2024 FSQ 论文](./ICLR-2024-finite-scalar-quantization-vq-vae-made-simple-Paper-Conference.pdf)
- [iFSQ: Improving FSQ for Image Generation with 1 Line of Code](./2601.17124v2.pdf)
