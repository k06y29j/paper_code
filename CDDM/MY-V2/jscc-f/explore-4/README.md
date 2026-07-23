# explore-4：论文 2605.26089v2 的 CVQ 与 CAR

本目录独立于 `explore-2`。目标是先在冻结 CNN/Swin Layer 1 下训练不同空间尺度的 Layer 2 CVQ tokenizer，再对通过 oracle/码本健康门槛的 tokenizer 训练条件 CAR receiver。

## CVQ 阶段

`train_cvq_spatial.py` 使用新的 Layer 2 E2，而不是把 `explore-2` 固定的 `16x16` latent 插值成其他大小：

| E2 stride-2 次数 | CVQ 通道图大小 | 论文原生 C | 原生 code-vector D |
|---:|---:|---:|---:|
| 4 | `16x16` | 256 | 256 |
| 3 | `32x32` | 1024 | 1024 |
| 2 | `64x64` | 4096 |
| 5 | `8x8` | 64 |

CVQ 使用论文的 global channel codebook：每个完整通道图只有一个 index，index 形状为 `[B,C]`。当前 q2 token 数为 `C=240`；与固定 Layer-1 `z1[:,:16]` 合起来为 `16+240=256` 个 Layer-2 前端通道。低容量 `K=1024` 已保留为消融；主实验遵循论文默认的单一 `K=16384`（论文 Table 3 显示 CVQ 在 K=1024/4096/16384 持续改善且未发生 usage collapse）。`--embedding-dim` 可把完整图先投影到任意 D；当前先比较原生 `D=256`（16×16）与 `D=1024`（32×32）。

早期 fresh-E2 probe 显示 Layer-1 bypass 会使大量 code 死亡（K=4096 仅约 15 个 index 被用到，且各 K PSNR 重合）。因此正式 v2 sweep 额外启用两个与验收直接相关的训练项：可微 batch-marginal soft-usage entropy，和要求真实 q 路径优于 zero-q 路径的 q-relevance hinge。它们不改变部署量化或接收端信息边界；日志会单独报告这两个 loss。

Layer-2 解码固定采用 `D2([z1[:,:16], q2[:,:240]])`：前端 256 通道中，前 16 个是冻结 Layer-1 的 `z1`，从不参与 nested channel dropout；后 240 个是量化 q2。一个可训练的 1×1 D2 前端将这 256 通道适配到继承 D2 trunk 的 320 通道接口，因此并未把 q2 伪装成 304 个 token。最终组合严格为 `x2=clip(x1 + D2(z1,q2)-D2(z1,0))`：zero-q 恒等于 x1，z1 只能提供残差的条件，不能绕开 q2 自行提升重建。嵌套 dropout 按论文 Eq. (9) 以概率 `alpha=0.5` 对 q2 发生，另 50% step 使用完整 q2；这保证全量重建和 CAR 的由粗到细通道顺序同时被训练。这样 q2 的 K/D/空间尺度仍是唯一被扫的 Layer-2 信息，且 CAR 的 `q2_hat` 解码路径完全一致。训练发送端使用 `img,x1` 产生 q2，D2 的 z1 条件来自冻结 Layer 1；验证遵守 `RandomCrop(256)+Flip`（train）与 `CenterCrop(256)`（valid）协议。

由于 JSCC 的残差分支起点不同于论文的整图 tokenizer，主训练先进行 10 epoch continuous-q residual warm-up，学习 `D2(z1,q)-D2(z1,0)`；随后用 warm-up 后 E2 的真实 channel maps 初始化单一 `K=16384` 码本，进行硬 CVQ 训练。该两阶段 decoder-reinforcement 只改变训练阶段，不改变发送端量化或接收端信息边界。

```bash
CUDA_VISIBLE_DEVICES=0 bash MY-V2/jscc-f/explore-4/run_cvq_spatial.sh cnn 4 cvq-cnn-f16-c16plus240-k1024-v1
CUDA_VISIBLE_DEVICES=2 bash MY-V2/jscc-f/explore-4/run_cvq_spatial.sh swin 4 cvq-swin-f16-c16plus240-k1024-v1
# D=1024 的 32x32 点；默认仍是 q2 C=240、K=1024，使用小 batch 控制显存。
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=1 bash MY-V2/jscc-f/explore-4/run_cvq_spatial.sh cnn 3 cvq-cnn-f32-c16plus240-k1024-v1
```

每次训练都会报告每个 K 的 `PSNR(x2)`、相对 `x1` 增益、zero/shuffle drop、PPL/K、top1、K 容量单调性与码本秩统计。只有 oracle 增益为正、K 单调、码本无塌陷且 zero/shuffle 相关性通过的配置，才可进入 CAR。

## CAR 阶段（待 CVQ 筛选后启动）

CAR 将 index `[B,C]` 看为长度 C 的一维序列，训练 `p(i_k | i_<k, z1, x1)`；`z1/x1` 是接收端合法条件，`img/z2/q2/真实 index` 只可作为训练监督。CVQ tokenizer 训练期间的 nested channel dropout 随机保留前 `c_keep` 个 q2 通道，建立由粗到细的 channel 顺序；CAR 推理以 BOS 起始并逐 channel rollout，生成硬 codebook `q2_hat` 后通过冻结 `q2_bridge + z1[:,:16] -> D2 -> combiner` 得到 `x2_hat`。

`train_cvq_car.py` 是论文式的 causal decoder-only transformer：teacher-forced next-channel CE 与 rollout soft-code reconstruction 共同训练，验证只使用 greedy hard-code rollout。CAR 只在 CVQ 的 `oracle_goal_met=1` 后启动。

```bash
CUDA_VISIBLE_DEVICES=0 bash MY-V2/jscc-f/explore-4/run_cvq_car.sh \
  MY-V2/jscc-f/explore-4/checkpoints-cvq/<selected>/best.pth car-<selected>-k4096-v1
```

最终唯一 receiver 验收：完整 DIV2K valid100 中 `PSNR(x2_hat)-PSNR(x1) >= +0.5 dB`，同时无泄漏审计和 condition shuffle 检查通过。
