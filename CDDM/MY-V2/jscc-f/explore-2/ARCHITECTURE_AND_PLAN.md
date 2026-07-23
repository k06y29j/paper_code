# explore-2 架构、无泄漏合同与实验计划

> 更新时间：2026-07-13（Asia/Shanghai）  
> 本文只定义 `explore-2` 的实验口径和执行顺序。已运行结果与计划项严格分开；没有日志或 checkpoint 证据的项目均标为“计划”，不视为已完成。

## 1. 唯一认可的两层计算图

### 1.1 Layer1

```text
img --E1--> z1 --D1--> x1
```

- `z1` 是 Layer1 已发送并在接收端可用的信息。
- `x1` 必须由接收端用同一个 `D1(z1)` 得到；不能在接收端用原图旁路生成另一份 `x1`。

### 1.2 Layer2 发送端 oracle

```text
concat(img, x1) --E2--> z2 --quantizer--> q2 / index2
q2 --D2--> u2
concat(x1, u2) --combiner--> x2
```

这里的 `oracle` 指发送端能看到 `img`，因此能够计算 `z2`、`q2`、`index2` 和 `x2`。它回答的是“这套 Layer2 量化表示有没有重建增益和容量表达力”，不回答“接收端能否得到这份 q2”。

### 1.3 接收端推断图

```text
received z1 --D1--> x1
G(z1 and/or x1) --> q2_hat / index2_hat
q2_hat --D2--> u2_hat
concat(x1, u2_hat) --combiner--> x2_hat
```

最终目标按接收端结果验收：

```text
delta_rx = PSNR(x2_hat, img) - PSNR(x1, img)
```

CNN-FSQ receiver 的硬门槛是完整 DIV2K valid 100 图上 `delta_rx >= 0.5 dB`。`PSNR(x2)>PSNR(x1)` 只证明发送端 oracle 成立，不能替代这个门槛。

## 2. Sender oracle 与 receiver inference 必须分开记账

| 项目 | sender oracle | receiver inference |
|---|---|---|
| 当前样本可见输入 | `img, x1` | `z1` 和/或由其确定的 `x1` |
| Layer2 表示 | `z2 -> q2/index2` | `G(z1,x1) -> q2_hat/index2_hat` |
| 输出 | `x2` | `x2_hat` |
| 主要指标 | `psnr_oracle`, `delta_oracle` | `psnr_pred`, `delta_rx`, `gap_oracle` |
| 用途 | 证明 Layer2 表达力、提供训练标签 | 证明真实接收端增益 |

文档、日志和汇总表中禁止把 `x2` 写成 `x2_hat`，也禁止把 `delta_oracle` 写成接收端增益。oracle 可以作为监督上界，但不能作为部署结果。

## 3. 精确的 no-leak 合同

### 3.1 推断时允许的信息

- 接收到的 `z1`；
- 严格由 `z1` 经冻结或已声明版本的 `D1` 计算出的 `x1`；
- 模型参数、固定位置编码、公开超参数；
- AR 路线中本模型此前已经生成的 token；
- 扩散路线中接收端本地产生的随机噪声、时间步和由允许条件计算的特征。

### 3.2 推断时禁止的信息

- `img`、其 crop/feature/hash，或 `img-x1`；
- `z2`、`q2`、真实 `index2`；
- oracle `u2`、oracle `x2`，以及从它们派生的任何张量；
- AR 验证或推断时 teacher-forced 的真实 token 前缀；
- 用当前样本的任何发送端 tensor 选择 checkpoint、采样种子或后处理参数。

“训练时可作为 label”不等于“可作为 predictor 输入”。训练可以用 `img` 计算 `MSE(x2_hat,img)`，也可以用 oracle `q2/index2/x2` 计算监督损失；这些 tensor 必须位于 predictor 调用之外，且不得通过条件分支、缓存、全局变量或模块属性回流到 `G`。

### 3.3 当前代码已经实施的结构审计

当前实现不是只靠注释约定：

- [`contracts.py`](./contracts.py) 的冻结 `ReceiverCondition` 只有 `z1`、`x1` 两个字段；
- predictor 的接口必须严格是 `forward(condition)`；
- `img/z2/q2/oracle_indices` 等名字被列为 source-only；
- 每个 train/val loader 的首 batch 都调用运行时审计，检查 predictor 没有保存监督目标 tensor；
- [`train_fsq_receiver.py`](./train_fsq_receiver.py) 保存的 checkpoint 含 `receiver_contract`；日志指标 `receiver_only_audit=1` 表示该次 loader 的结构审计通过。
- [`test_receiver_contract.py`](./test_receiver_contract.py) 枚举 K125/K729/K4913 mixed-radix roundtrip，检查 FSQ predictor、channel-VQ AR、image-VQ diffusion 和固定 train-selected q mix 的接收端接口，验证 AR 未来 token 影响严格为零，并在强制禁用 E2/e3 时闭合 `q2_hat -> D2 -> combiner`。

该审计覆盖接口和已存 tensor，不是任意 Python 信息流的数学证明。以后新增 AR、diffusion、VQ predictor 时仍必须复用 `ReceiverCondition`，并增加条件 shuffle、目标前缀禁用和独立 inference-only eval。

本文更新时实际运行了 lightweight audit：mixed-radix、predictor forward、supervision invariance、FSQ raster AR 因果性、channel-VQ AR 因果性、image-VQ diffusion 合同和禁用 E2 decode 均为 `PASS`。结构通过不等于 receiver 质量达标。

### 3.4 数据裁剪合同

- 所有 DIV2K train 入口**必须**使用 `RandomCrop(256,256)+RandomHorizontalFlip+ToTensor`；
- 所有 DIV2K valid 入口**必须**使用 `CenterCrop(256,256)+ToTensor`；
- 禁止用 center-crop 训练结果替代随机裁剪训练结果；
- [`contracts.py`](./contracts.py) 的 `assert_div2k_crop_protocol` 在 FSQ、VQ、AR、diffusion 和 continuous-q loader 建立时检查这两套 transform，违约立即报错。

### 3.5 仅训练期 receiver-information teacher（不属于部署图）

[`train_receiver_information_teacher.py`](./train_receiver_information_teacher.py) 额外提供一个**明确不部署**的经验恢复能力诊断：

```text
img --frozen E1/D1--> z1,x1
Teacher(z1,x1) --> residual_hat
x_teacher = clamp(x1 + residual_hat)
```

- `Teacher.forward` 仍严格只接受 `ReceiverCondition(z1,x1)`；`img` 在 forward 返回后才作为 MSE 监督目标，`z2/q2/真实索引` 不构造也不传入；
- 该脚本复用 DIV2K loader 的 `RandomCrop` 训练、`CenterCrop` 验证，并在首 batch 执行 no-leak 审计；full-val 报告保存 `PSNR(x1)`、`PSNR(teacher)` 和 delta；
- 它只是有限模型下“接收端已有信息还可回归多少 image residual”的诊断，既不是数学上界，也**不是** `q2_hat -> D2 -> combiner` 的接收端结果；
- 其 checkpoint/JSON/Markdown 均写入 `deployment_prohibited=true`。允许的用途仅是训练期 teacher/蒸馏目标；任何部署增益仍必须单独通过第 1.3 节的 q2/index 图和完整 valid-100 验收。

## 4. 已有 CNN-FSQ oracle 证据

以下数字来自 [`../explore/EXPERIMENT_SUMMARY.md`](../explore/EXPERIMENT_SUMMARY.md) 的 6.1，均是同一口径的 epoch-100 sender-oracle 公平比较：CNN Layer1/Layer2、direct nested FSQ、`x1=21.771972 dB`。

| FSQ K | `psnr_oracle` | `delta_oracle` | continuous | quant gap | drop-zero | drop-shuffle | used / PPL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 125 | 22.876459 | +1.104487 | 22.899974 | 0.023515 | 1.449481 | 2.288306 | 125 / 92.94 |
| 729 | 22.905375 | +1.133402 | 22.928278 | 0.022903 | 2.580539 | 2.962144 | 641 / 261.05 |
| 4913 | 22.947125 | +1.175153 | 22.963377 | 0.016252 | 2.857201 | 3.166313 | 1921 / 624.39 |

这组证据说明：

1. 三档 oracle 都满足 `PSNR(x2)>PSNR(x1)`；
2. `K=125 < 729 < 4913` 时，epoch-100 PSNR 严格增加；
3. zero/shuffle drop 远大于 `0.1 dB`，不是仅靠高 usage 的无效码本；
4. 可供 receiver 争取的 oracle headroom 约为 `1.10–1.18 dB`。

它仍然不证明 `delta_rx>=0.5 dB`。默认 receiver 脚本读取的是 K4913 的 best checkpoint（日志记录 epoch 95、oracle PSNR `22.947396 dB`），这与上表的固定 epoch-100 公平数字也必须分开。

### 4.1 当前 receiver 完整验证结果

截至本文检查时，已有完成验证点包括：

| 路线 | oracle/表示 | 完整验证最佳 | `delta_rx` | `receiver_only_audit` | `goal_met` |
|---|---:|---:|---:|---:|---:|
| direct-q continuous + independent D2 | FSQ K4913 | +0.259816 dB | +0.259816 dB | 1 | 0 |
| base-initialized AR | FSQ K4913 | 约 +0.12 dB | 约 +0.12 dB | 1 | 0 |
| channel AR hard/soft | grouped channel D512 | +0.196760 dB | +0.196760 dB | 1 | 0 |
| conditional diffusion | image-VQ D512 | +0.118283 dB | +0.118283 dB | 1 | 0 |
| arbitrary-D continuous q2_hat | D512 | +0.392860 dB | +0.392860 dB | 1 | 0 |
| fixed train-simplex top-3 continuous q2 mix | D512 | +0.401097 dB | +0.401097 dB | 1 | 0 |

所有数字均来自 100 张 DIV2K valid 的 center-crop；训练始终使用 random crop。最后一行来自 `continuous_q_d512_crossmodel5_train_simplex_top3.json`：先只在 DIV2K train 的随机裁剪上选择 top-3 和全局权重 `[0.632098,0.335829,0.032072]`，再固定成一个 `q2_hat`，并且只经过一次 v9 `D2/combiner`；它不是把 `x2_hat` 做图像空间平均。它们仍未达到 `+0.5 dB`。任意 D continuous q2_hat 是 receiver 信息充分性探针，不冒充原三通道 FSQ q2；原 FSQ 合同的 residual-D2 路线单独记账。

`train_continuous_q_receiver.py` 还提供 opt-in 的 `--d2-type qonly-highres-residual`。它保留旧 `Layer1InitializedD2` 为 strict-loaded base，并添加只读取生成 `q2_hat[D]` 的四级高分辨率 residual 分支；该分支的 RGB head 为零初始化，因此从 v9 初始化时起点与旧 D2 完全相同。D2 的接口硬约束为 `forward(q2_hat)`，combiner 仍硬约束为 `forward(x1,u2)`；该变体当前只有结构/兼容 smoke，尚无 full-val 指标，不能写入上表或替代 `+0.5 dB` 门槛。

为避免把上述固定 q mix 停留在“一个旧参考 D2/combiner 的 ensemble evaluation”，新增 [`train_continuous_q_mix_receiver.py`](./train_continuous_q_mix_receiver.py)。它只接受 `eval_continuous_q_ensemble.py --selection-mode train-simplex` 保存的 JSON：脚本检查该 JSON 的选择 split 是 DIV2K train、选择 transform 是 RandomCrop、外部成员没有剩余非零权重，并冻结成员 generator 和权重。随后部署/训练图严格为：

```text
(z1,x1) -> fixed sum_i(w_i*G_i(z1,x1)) -> q2_hat
        -> one receiver D2 -> u2_hat -> one combiner(x1,u2_hat) -> x2_hat
```

唯一可训练参数是这一个 D2/combiner；`img` 只在 forward 结束后提供重建损失，`z2/q2/真实索引` 不构造。训练保持 RandomCrop，valid 保持 CenterCrop，默认 `--checkpoint-selection final`，因此验证不会重选 q 权重或训练 epoch。已通过 CPU structural smoke 和真实 DIV2K 单 batch 闭环。

q-mix 的 EMA 还有一个必须保留的实现约束：冻结 generator 为节省显存而在普通 receiver 与 EMA receiver 之间**按对象精确共享**。因此不能对整个 receiver 调用通用 in-place EMA；它会先 `mul_` 共享 generator 再读取 source，静默改变本应冻结的 q2_hat。`update_post_qmix_ema` 现在断言 `ema.generator is receiver.generator`，并且只更新两份独立的 D2/combiner state。该修复只保证固定 q-mix 不漂移，并不提高或替代 receiver 指标。

q-mix 结果按下列状态严格记账：

- 固定 train-simplex top-3、一次参考 v9 D2/combiner 的独立 full-valid-100 结果仍为 `+0.401097 dB`；
- 旧单 receiver 运行 `cnn-continuous-q-mix-top3-logmse-e60-v1` 在 train e39 前停止，最后完成的 full-valid e35 为 `+0.412509 dB`、`goal_met=0`，没有 final checkpoint 结论，不能称为运行中或成功；
- 修复 EMA 后的 `cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-v1` 已在 e35 后停止；最后完整 full-valid-100 为 `+0.427319 dB`、`goal_met=0`。它未达到 `+0.5 dB`，没有 final 结论，不能称为运行中、成功或拿中间 checkpoint 选模；
- `cnn-continuous-q-mix-top3-logmse-e60-v1` 的 latest 接 v12 latest post-q initializer 做过一次 **zero-training、fixed-q、read-only** 诊断：EMA full-valid-100 为 `+0.402400 dB`、`goal_met=0`。该状态仅用于检查 q-mix 与一个既有 post-q 模块能否闭环；它不是新训练结果，不能按历史 best 选模或替代前述固定协议结果；
- q-mix 的 `--d2-type qonly-highres-residual` 的首条低 LR run 已在完整 CenterCrop valid100 的 e15 记录 `+0.287151 dB` 后停止（e10 基线为 `+0.401999 dB`，解冻后下降，未达标）；该失败不解释为泄漏或误冻结。新 q-only branch 末端零初始化，在 `5e-5` 下 e10 的 RGB residual RMS 仅约 `8.2e-4`。因此已新增 `--d2-highres-lr`：仅对 `stem/stages/residual_head` 建立独立 optimizer group，base D2 继续用 `--d2-lr`、combiner 继续用 `--combiner-lr`；`...hlr3e4-warm20-e80-v1` 在 branch `3e-4` 的完整 CenterCrop e10 仍为 `+0.401175 dB` 后停止，未作为结论。

当前优先候选是 legacy MSE fixed-q run：在一次非结果性中断前，e40 full-valid-100 达 `+0.472027 dB`（`goal_met=0`）。新增 `--resume-checkpoint` 会严格核对固定 q mix、v9 init、训练超参数、RandomCrop/CenterCrop/no-leak 合同和 optimizer groups；只有全部一致时才恢复 raw/EMA D2+combiner、optimizer、epoch/best。该 run 已从 e41 接续，最终仍只按 final e60 checkpoint 判定。

FSQ 侧现在也有对应的 opt-in `train_fsq_receiver.py --receiver-d2-arch qonly-highres-residual`：建议从 v5 direct-FSQ K4913 checkpoint 启动，并要求 sender oracle 的 `condition_mode=none`（因此运行时断言 `tokenizer.condition(x1,z1) is None`），把 private receiver D2/combiner 严格载入为 base，再加一个只读 `[B,3,16,16] q2_hat` 的四级 RGB residual 分支。新分支 head 零初始化，所以 smoke 已验证起点与 v5 的 `D2+combiner` 完全相同；D2 的 Python 接口硬约束为 `forward(q2_hat)`，combiner 仍为 `forward(x1,u2_hat)`。可选 warmup 先冻结/eval v5 predictor、base D2 与 combiner，仅训练新分支。它同样尚无 full-val 指标，不能替代现有 `+0.259816 dB` 结果或宣称达到目标。

## 5. 目标一：CNN-FSQ receiver 的执行顺序

### Phase R0：冻结并核对 oracle

- 固定 E1、D1、E2/FSQ/D2、combiner 版本；
- 记录 `psnr_x1`、`psnr_oracle`、`delta_oracle` 和 oracle checkpoint SHA/path；
- receiver 的 checkpoint 选择不能由验证样本的发送端 tensor 决定。

### Phase R1：直接预测 q2，先做最便宜的充分性测试

使用现有 `route=direct_q`：

1. `condition_mode=z1_x1` 为主；`z1`、`x1` 是必要消融；
2. continuous 输出只作为优化和可预测性诊断；
3. 最终部署门槛必须在 hard FSQ snapping 后计算；
4. 先在 K4913 oracle 上争取最大 headroom，再复查 K125/K729 是否更易预测。

如果完整验证已经达到 `delta_rx>=0.5 dB`，直接进入复现和多 seed，不因为“生成模型更复杂”而继续引入 AR/扩散。

### Phase R2：直接预测联合索引

使用现有 `route=joint_index`，先在 K125 上验证 `[B,125,16,16]` logits 的可行性，再决定是否扩大 K。现有 `parallel_index` 把三个 FSQ 标量分别分类，可作为 K4913 的显存友好诊断，但主比较仍优先 direct-q 与 joint-index。

两条直接路线统一报告：

- `psnr_x1 / psnr_oracle / psnr_pred / delta_rx / gap_oracle`；
- hard `q_mse`、联合或分量 index accuracy；
- predicted-code used/PPL/top1；
- `z1/x1` 条件 shuffle 后的 PSNR drop；
- `receiver_only_audit`。

PSNR 是主判据；高 index accuracy 或低 q-MSE 不能替代 `delta_rx>=0.5 dB`。

#### R2b：必要时联合适配接收端 D2/combiner

现有 trainer 已支持 `--finetune-d2` 和 `--receiver-combiner {oracle,residual}`。这些模块位于 `q2_hat` 之后，只要它们仍只读取 `q2_hat/x1`，联合训练不会违反 no-leak 合同。执行顺序仍是先保存 frozen-D2/oracle-combiner baseline，再做 receiver-side adaptation；两组必须分开命名和汇报，不能把适配后的增益归因给 predictor 本身。

`residual` combiner 是 identity-safe 的 `x1 + correction(x1,u2_hat)`。它是已实现选项，但本文没有对应正式 full-val 结果，因此不声明它优于 oracle combiner。

### Phase R3：只有直接路线不足时才做 AR

启动条件：R1/R2 在足够训练预算和条件消融后仍未达到 `+0.5 dB`，且条件 shuffle 证明 `z1/x1` 对 token 有可利用信息。AR 推断只能读取已生成前缀，验证时禁止真实 token teacher forcing。

FSQ raster AR 和 base-initialized AR 已集成在 [`receiver_models.py`](./receiver_models.py) 与 [`train_fsq_receiver.py`](./train_fsq_receiver.py)。K4913 base-initialized AR 的完整验证最佳仍约 `+0.12 dB`，未达到 `+0.5 dB`，因此保留为失败证据，不再作为当前主路线。

### Phase R4：只有仍有明确条件信息余量时才做扩散

扩散用于条件分布明显多峰、直接回归平均化严重，或 AR 的顺序误差累积明显时。优先生成 hard-FSQ 可投影的 q2；任何 continuous relaxation 都需另报 snapping 后结果。

当前 diffusion 主路线转向更合适的 image-VQ embedding，入口为 [`train_image_vq_diffusion.py`](./train_image_vq_diffusion.py)；其 public deployment graph 只接受 `ReceiverCondition(z1,x1)` 并在内部采样噪声。

## 6. 目标二：CNN/Swin × image-VQ/channel-VQ

[`vq_modules.py`](./vq_modules.py) 已实现两种 tokenization 的共享 nested-prefix VQ 核心：每档 K 使用同一参数表的 `codebook[:K]`，并提供 chunked nearest lookup、usage/health 指标、zero/shuffle、初始化和 dead-code refresh。[`train_layer2_vq_nested.py`](./train_layer2_vq_nested.py) 已完成 CNN/Swin Layer1 与任意 Layer2 codec 的统一训练、full-val 与 checkpoint 合同。

### 6.1 两种 VQ 的 token 定义

Layer2 基础 latent 空间固定为 `C×16×16`，码字维度 `D` 与 C/H/W 独立：

| family | 一个 code 对应什么 | index shape | codebook embedding | 每图 token 数 |
|---|---|---|---|---:|
| image-VQ | 一个空间位置的 D 维向量 | `[B,16,16]` | `[K,D]` | 256 |
| channel-VQ | 一个 channel map 投影后的 D 维向量 | `[B,C]` | 原生 `[K,16,16]`；高维 `[K,1,D]` | C |

因此必须区分基础 latent 宽度 C 和码字维度 D：

- image-VQ：`C×16×16 -> D×16×16`，可固定 C/K 独立 sweep D；
- channel-VQ：每个 `16×16` channel map 经共享投影变为 D 维，可固定 C/K 独立 sweep D；
- sweep C 会同时改变 Layer2 基础 latent 或 token 数，不能冒充纯 embedding sweep。

名义离散码率（不含熵编码开销）为：

```text
image-VQ:   bits/image = 16*16*log2(K)
channel-VQ: bits/image = C*log2(K)
bpp = bits/image / (256*256)
```

### 6.2 第一阶段公平锚点矩阵

先固定 `C=256`，使 image-VQ 与 channel-VQ 都恰有 256 tokens/image；K 统一为 `256/1024/4096`。这是 2 个 Layer1 × 2 个 VQ family × 3 个 K，共 12 个 cell。

| Layer1 | VQ family | C | K=256 | K=1024 | K=4096 |
|---|---|---:|---|---|---|
| CNN | image-VQ | 256 | 计划 | 计划 | 计划 |
| CNN | channel-VQ | 256 | 计划 | 计划 | 计划 |
| Swin | image-VQ | 256 | 计划 | 计划 | 计划 |
| Swin | channel-VQ（global） | 256 | D128/D256 固定 epoch15 已验证，见 6.3.1 | D128/D256 固定 epoch15 已验证，见 6.3.1 | D128/D256 固定 epoch15 已验证，见 6.3.1 |

每个三档 K sweep 必须固定数据、seed、训练 epoch、optimizer、E2/D2/combiner 架构、latent shape、初始化口径和 checkpoint 选择规则；使用 `vq_modules.py` 的共享 `codebook[:K]` 精确前缀来减少独立码本优化差异。主表按同一固定 epoch 比较，另表才允许列 best checkpoint。不能用不同训练预算制造“容量增大”。

在 `C=256` 时，两种 family 的名义 bits/image 都是 `256*log2(K)`，对应 K256/K1024/K4096 的 `2048/2560/3072 bits/image`，即 `0.03125/0.0390625/0.046875 bpp`。

上段只适用于 image-VQ 与 global channel-VQ。新增的 grouped channel-VQ 使用 round-major `[Kmax,16,16]` 码本，并把通道 c 的候选限制为 `c+C*arange(K/C)`；因此其诚实名义码率是 `C*log2(K/C)`，不能与上表按同一个 K 解释。正式 grouped C256 sweep 固定 K512/K1024/K4096，即每通道 2/4/16 个候选，分别为 256/512/1024 bits/image（0.00390625/0.0078125/0.015625 bpp）。K256 只有一个候选/通道、零索引比特，只允许做结构诊断且不得晋级。正式三档全部执行 local PPL/top1、跨样本 channel variation、zero/shuffle、PSNR 与 paired gate，不设 K=C 豁免。

### 6.3 第二阶段独立 D sweep

- 固定 `C=256`、K、训练预算与 Layer2 codec，独立比较 `D=256/512/1024`；
- image-VQ 与 channel-VQ 都必须报告同 K 下的 full-val PSNR、码本 rank/PPL/top1、zero/shuffle drop；
- 另做 C sweep 时明确称为 latent-width 或 channel-token/rate sweep，不与纯 D sweep 混写；
- 已完成的 CNN image-VQ 证据：原生 `D=256,K4096` 为 `+0.9058 dB`，`D=512,K4096` 已达到至少 `+1.4114 dB`，且 D512 内部 K256/K1024/K4096 为 `+1.1680/+1.3216/+1.4114 dB`。

#### 6.3.1 已验证补充：Swin global channel-VQ 的 D128 -> D256

这组是 sender oracle 的固定 epoch 容量实验：Layer1=`swin`、Layer2=`swin`、
global channel-VQ、`C=256`、K=`256/1024/4096`、K-means 初始化、seed=`20260713`、
enhanced combiner、`--oracle-only`。训练严格使用
`RandomCrop(256)+RandomHorizontalFlip+ToTensor`，100 张 valid 严格使用
`CenterCrop(256)+ToTensor`。两份运行均固定取 epoch15，绝不混入 best checkpoint，
因此 D 的差值不是由不同选点造成。

| K | D128 delta oracle (dB) | D256 delta oracle (dB) | D256-D128 (dB) |
|---:|---:|---:|---:|
| 256 | +0.317844 | +0.345848 | +0.028004 |
| 1024 | +0.412309 | +0.465433 | +0.053124 |
| 4096 | +0.474194 | +0.555498 | +0.081304 |

两种 D 内部均满足严格 K 单调；D256 在每一个固定 K 都高于 D128。所有六个点
通过 global channel-VQ 的 non-collapse、usage scaling、zero/shuffle 与 paired-K
gate（D256 K4096 的 `used/PPL/top1/index_change=3888/2938.944/0.003633/0.994510`）。
所以该表是“码本容量有效表达”证据，而不是高 usage 的假象。最高 D256/K4096
oracle 增益仍只有 `+0.555498 dB`，低于目标一的 `>1 dB` sender 余量；并且它没有
生成 `q2_hat`，不能替代任何 receiver `x2_hat` 结果。完整原始证据见
[`results-vq/swin_global_channel_vq_embedding_scaling_epoch15.md`](./results-vq/swin_global_channel_vq_embedding_scaling_epoch15.md)。

这不把 grouped channel-VQ 的 K 语义混入 global 结果：后者的 K 是 global row
codebook 前缀，前者的 K 是 round-major 总 rows，仍按第 6.2 节的独立口径报告。
旧 `swin-image-vq-c256-d256-fullhard-fixed20-v4` 是 image-VQ 且已塌陷，明确不属于
此 D scaling 结论。

### 6.4 每个 VQ cell 的 promotion gate

1. **oracle quality**：`PSNR(x2)>PSNR(x1)`；
2. **capacity effectiveness**：固定 C 时，主表严格满足 `PSNR(K256)<PSNR(K1024)<PSNR(K4096)`；多 seed 阶段还要报告均值、方差和逐图严格递增比例；
3. **code relevance**：`drop_zero>=0.1 dB` 且 `drop_shuffle>=0.1 dB`；
4. **non-collapse**：不触发 `PPL<=1.1`、`top1_frac>=0.95` 或 token-change `<=0.01` 的塌陷警报，并报告 used/PPL/dead ratio；
5. **receiver sufficiency**：至少一条严格 no-leak 的 direct/index receiver 在完整验证上满足 `PSNR(x2_hat)>PSNR(x1)`，且 `condition_shuffle_drop>=0.1 dB`。最终若采用 AR/扩散，仍按同一 receiver 图验收。

## 7. 为什么不能只看 usage：已有塌陷证据

旧 `explore` 已给出两个直接反例：

- Swin K4096 加 uniform-KL 后 used/PPL 达到 `3790/2503.90`，但早期 `drop_zero/drop_shuffle` 只有 `0.0037/0.0026 dB`；高 occupancy 没有形成有效图像信息。
- Swin L8 塌陷点每图 unique token 为 `1.00`、spatial change 为 `0.0000`、`drop_shuffle=0.0000 dB`。

相对地，CNN L20 有 `174.8` unique tokens/image、spatial change `0.9705`，zero/shuffle drop 为 `2.3257/3.7429 dB`。因此：

- used/PPL 只做诊断；
- zero/shuffle 和最终 PSNR 是硬证据；
- 容量增大必须体现为更高重建 PSNR，而不是只体现为更大的参数量或更多 used codes。

对 receiver 还要增加 condition-shuffle：若打乱 `z1/x1` 条件后 `x2_hat` 几乎不变，则 predictor 可能只是生成了 unconditional 常量/先验，即使 PPL 很高也不能称为可预测。

## 8. 当前可执行命令

以下命令只引用已经存在的 `explore-2` 文件。

### 8.1 帮助与结构 smoke

```bash
cd /workspace/yongjia/paper_code/CDDM
conda run -n cddm_ddnm python MY-V2/jscc-f/explore-2/train_fsq_receiver.py --help

conda run -n cddm_ddnm python MY-V2/jscc-f/explore-2/train_fsq_receiver.py \
  --smoke-shapes --route direct_q --condition-mode z1_x1

conda run -n cddm_ddnm python MY-V2/jscc-f/explore-2/train_fsq_receiver.py \
  --smoke-shapes --route joint_index --condition-mode z1_x1 \
  --oracle-checkpoint MY-V2/jscc-f/explore/checkpoints-direct/direct-cnn-d3-l5x5x5-group-compatible-blend-e100/jscc_f_direct-cnn-d3-l5x5x5-group-compatible-blend-e100_layer2_fsq_direct_cnn_d3_l5x5x5_group_compatible_blend_goal_best.pth

conda run --no-capture-output -n cddm_ddnm \
  python MY-V2/jscc-f/explore-2/test_receiver_contract.py --real-k125 auto

conda run -n cddm_ddnm python MY-V2/jscc-f/explore-2/train_layer2_vq_nested.py \
  --arch cnn --vq-family image-vq --latent-c 8 --rates 5,9 \
  --rate-weights 1 --smoke-shapes
```

### 8.2 direct-q 训练

```bash
cd /workspace/yongjia/paper_code/CDDM
GPU=0 ROUTE=direct_q CONDITION_MODE=z1_x1 \
VERSION=cnn-fsq-k4913-direct-cont-rerun \
bash MY-V2/jscc-f/explore-2/run_fsq_receiver_cnn.sh

GPU=0 ROUTE=direct_q CONDITION_MODE=z1_x1 \
VERSION=cnn-fsq-k4913-direct-hard-rerun \
bash MY-V2/jscc-f/explore-2/run_fsq_receiver_cnn.sh --hard-fsq

# 已实现的 receiver-side adaptation；该命令本身不代表已有结果
GPU=0 ROUTE=direct_q CONDITION_MODE=z1_x1 \
VERSION=cnn-fsq-k4913-direct-hard-d2ft-residual-rerun \
bash MY-V2/jscc-f/explore-2/run_fsq_receiver_cnn.sh \
  --hard-fsq --finetune-d2 --receiver-combiner residual --decoder-lr 5e-5
```

### 8.3 K125 joint-index 训练

```bash
cd /workspace/yongjia/paper_code/CDDM
ORACLE=MY-V2/jscc-f/explore/checkpoints-direct/direct-cnn-d3-l5x5x5-group-compatible-blend-e100/jscc_f_direct-cnn-d3-l5x5x5-group-compatible-blend-e100_layer2_fsq_direct_cnn_d3_l5x5x5_group_compatible_blend_goal_best.pth
GPU=0 ROUTE=joint_index CONDITION_MODE=z1_x1 \
VERSION=cnn-fsq-k125-joint-index-rerun \
bash MY-V2/jscc-f/explore-2/run_fsq_receiver_cnn.sh \
  --oracle-checkpoint "$ORACLE" --lambda-index 0.002
```

### 8.4 nested image-VQ/channel-VQ

统一 trainer 已实现；以下是正式 C256 shared-prefix cell 的入口。命令存在不代表该 cell 已收敛或通过 promotion gate。

```bash
cd /workspace/yongjia/paper_code/CDDM
GPU=0 ARCH=cnn VQ_FAMILY=image-vq LATENT_C=256 RATES=256,1024,4096 \
VERSION=cnn-image-vq-c256-nested-seed20260624 \
bash MY-V2/jscc-f/explore-2/run_vq_nested.sh --seed 20260624 --codebook-init kmeans

# 同一 seed 串行运行 CNN/Swin x image/channel 四个 shared-prefix cell
GPU=0 SEED=20260624 bash MY-V2/jscc-f/explore-2/run_vq_matrix.sh --codebook-init kmeans
```

### 8.5 近期运行与诊断（均尚不能进入结果表）

以下是本次文档更新时的运行状态；它们与第 4.1、6.3.1 节的完成结果分开，
不会用于证明最终目标。

- `cnn-continuous-q-mix-top3-logmse-e60-v1`：已在 train e39 前停止，最后完整
  valid e35 为 `delta_rx=+0.412509 dB`、`goal_met=0`；没有 final 结论。
- `cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-v1`：EMA 已按第 4.1 节只更新
  D2/combiner，冻结 generator 不漂移；已在 e35 后停止，最后 full-valid-100
  `delta_rx=+0.427319 dB`、`goal_met=0`。它没有完成或达标，不作为历史 best 选模。
- v12 latest 的 zero-training fixed-q read-only diagnostic：EMA full-valid-100
  `delta_rx=+0.402400 dB`、`goal_met=0`。仅诊断 q-mix 接既有 post-q 模块的闭环，
  不作为训练结果或历史 best；q-mix highres 分支仍在实现/结构验证阶段，尚未运行。
- `swin-image-vq-c256-d256-rmsnorm-fixed5-v1`：dynamic image-VQ 分支已在 val10 后
  停止；该点虽有 `capacity_goal_met=1`，但 `noncollapse_goal_met=0`、
  `oracle_goal_met=0`、`receiver_goal_met=0`，不纳入容量成功证据。
- `swin-image-vq-c256-d256-rmsnorm-frozen-e30-v1`：冻结 E2/codebook 的 image-VQ
  control 仍在运行。val10 的 oracle/capacity/non-collapse gate 为真，但
  `receiver_goal_met=0`；它只是 control，未形成最终 sender 或 receiver 结论。

长任务由主调度者决定是否用 `setsid -f` 启动；不要重复启动同一 `VERSION`。所有新 checkpoint、log 和结果必须写入 `MY-V2/jscc-f/explore-2/`，旧 `explore` 只作为只读 oracle 来源。

## 9. 尚未实现的计划文件

以下仅是明确的占位计划，当前不存在，因此没有命令、checkpoint 或结果：

| 计划文件 | 用途 | 进入条件 |
|---|---|---|
| `train_fsq_receiver_ar.py` | receiver-only AR index generation | direct-q/joint-index 达不到 +0.5 dB |
| `train_fsq_receiver_diffusion.py` | receiver-only conditional q2 diffusion | 直接/AR 路线仍不足且条件信息证据充分 |

实现这些文件时必须留在 `explore-2`，并在日志头打印 `实验设计`、`loss设计`、`模块选择`。

## 10. 结果表的最小字段

每个正式实验至少保存：

```text
layer1_arch, quantizer_family, C, K, token_shape, embedding_shape,
bits_per_image, bpp, seed, epoch, checkpoint,
psnr_x1, psnr_oracle, delta_oracle,
psnr_pred, delta_rx, gap_oracle,
drop_zero, drop_shuffle, condition_shuffle_drop,
used_codes, perplexity, top1_frac, dead_ratio, token_change,
q_mse, index_accuracy, receiver_only_audit, goal_met
```

尚未运行的 cell 保持 `planned/not-run`，不得填入推测数值；早期 smoke 与 full-val 正式结果分表保存。
