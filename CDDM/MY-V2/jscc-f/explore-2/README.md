# JSCC-f explore-2

`explore-2` 专门承载两项工作：

1. 从接收端已有的 `z1/x1` 预测或生成 `q2_hat`，并验证 `x2_hat` 相对 `x1` 至少提升 `0.5 dB`；
2. 在 CNN/Swin Layer1 下公平探索 image-VQ/channel-VQ 的容量、有效表达和后续可预测性。

完整架构、no-leak 合同、实验矩阵和验收门槛见 [`ARCHITECTURE_AND_PLAN.md`](./ARCHITECTURE_AND_PLAN.md)。

## 当前状态

| 项目 | 状态 |
|---|---|
| CNN direct-FSQ sender oracle | 已有证据：K125/729/4913 的 epoch-100 PSNR 为 22.876459/22.905375/22.947125 dB，均高于 x1=21.771972 dB |
| FSQ direct/index/AR receiver | direct/index/AR 均已完成 full-val；历史最佳 continuous direct-q 为 `+0.2598 dB`，base-initialized AR 约 `+0.12 dB`；新增 `--receiver-d2-arch qonly-highres-residual` 从 v5 private D2/combiner 精确起步，新增支路仅读 `[B,3,16,16] q2_hat`，强制 `D2(q2_hat)`；目前仅完成结构/等价 smoke，尚无 full-val 结论 |
| receiver no-leak executable audit | mixed-radix、FSQ raster AR、global/grouped channel-VQ AR、image-VQ diffusion、continuous q、固定 train-selected q mix、禁用 E2 decode、递归 sender-target 审计与裁剪合同共 13 项全部通过 |
| channel-VQ AR receiver | grouped D512 oracle 上 hard/soft 条件逐通道 AR 均完成 full-val；最好 `+0.1968 dB`，soft val75 为 `+0.1538 dB`，未达门槛。当前 AR 同时支持 grouped 的 per-channel round 索引与 global 的 K-row 索引 |
| image-VQ diffusion receiver | 条件 DDIM/residual-mean 已完成多组 full-val；保存最佳约 `+0.1183 dB`，final-only v5 在 val40 为 `-0.5401 dB`，未达门槛 |
| arbitrary embedding VQ | image-VQ 支持 `[B,D,16,16]`；channel-VQ 支持 `[B,C,1,D]`，`D` 不再绑定 C 或 `[16,16]`；旧原生 checkpoint 精确兼容 |
| Layer1/Layer2 架构解耦 | `--arch` 只选择 E1/D1；`--layer2-arch {cnn,swin,residual-cnn,match}` 独立选择 Layer2，`match` 仅用于旧 checkpoint 兼容 |
| CNN image-VQ `D=1024` oracle | full-val K256/K1024/K4096 增益为 `+1.383/+1.547/+1.650 dB`，严格容量单调且 `oracle_goal_met=1` |
| CNN grouped channel-VQ `D=512` oracle | full-val epoch20 增益为 `+0.576/+0.727/+0.749 dB`；K、D 扩展与 local usage/shuffle 均有效 |
| continuous q2_hat receiver sufficiency / q-mix | 单 generator full-val 最高约 `+0.3970 dB`；固定 train-only top-3 q2 mix（不是 image ensemble）经一次 v9 `D2/combiner` 解码为 `+0.401097 dB`，仍未达 `+0.5 dB`。旧单 receiver 运行 `cnn-continuous-q-mix-top3-logmse-e60-v1` 在 train e39 前停止，最后完整 valid e35 为 `+0.412509 dB`、`goal_met=0`。冻结 q-mix generator 在 receiver/EMA 中**精确共享**时，EMA 只更新 D2+combiner，避免通用 whole-receiver EMA 原地改写 generator；修复后的 `cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-v1` 已在 e35 后停止，最后完整 full-valid e35 为 `+0.427319 dB`、`goal_met=0`。另有 v12 latest 的 zero-train fixed-q read-only diagnostic：EMA `+0.402400 dB`，仅诊断、未达标，且不能以历史 best 选模。首条 q-only-highres 低 LR run 在完整 CenterCrop valid e15 降至 `+0.287151 dB` 后停止；新 run `...hlr3e4-warm20-e80-v1` 正在运行，独立提高新 q-only branch 到 `3e-4`、base/combiner 保持 `5e-5/2e-5`，尚无可报告结论 |
| training-only receiver-information teacher | 新增 `(z1,x1)->x1+residual_hat` 的经验恢复能力诊断；它不产生 `q2_hat`、不执行 D2/combiner，checkpoint/report 均标为 `deployment_prohibited`，只能用作后续 q2/index receiver 的训练期 teacher |
| Swin global channel-VQ embedding scaling | **已验证 sender oracle（不是 receiver）**：Layer1/Layer2=`swin`、global channel-VQ、C256、固定 epoch15 的 D128/D256 均对 K256/1024/4096 严格增加。D128 增益为 `+0.317844/+0.412309/+0.474194 dB`，D256 为 `+0.345848/+0.465433/+0.555498 dB`；D256 相对 D128 三档均提升，所有点通过 strict-K、non-collapse、zero/shuffle gate。最高 oracle 仍低于 `+1 dB`，且没有 `x2_hat` 结论；旧 D256 image-VQ 塌陷运行明确排除 |
| Swin image-VQ 后续 | dynamic `swin-image-vq-c256-d256-rmsnorm-fixed5-v1` 已在 val10 后停止，虽 `capacity_goal_met=1`，但 `noncollapse_goal_met=0`、`oracle_goal_met=0`。冻结 E2/codebook control `...rmsnorm-frozen-e30-v1` 已完成 e30：最高 K4096 `+0.273 dB`，且 `noncollapse_goal_met=0`，不作为有效结果。D256 soft-usage 分支在 val15 恢复容量/非塌陷/strict-K，但最高 K4096 仅 `+0.335 dB`；现运行 D128 连续重建预热10轮、hard-VQ 渐变30轮的 oracle-only run，早期验证始终 hard-VQ，e40 前不据此早停 |
| q-mix resume status（取代上行旧 e35 叙述） | legacy MSE fixed-q run 的 e40 full-valid-100 为 `+0.472027 dB`、仍 `goal_met=0`；已用 `--resume-checkpoint` 严格核对固定 mix、v9 init、训练超参数、RandomCrop/CenterCrop/no-leak 合同与 optimizer groups，并从 e41 接续至 final e60。q-only-highres 的低 LR 与 `3e-4` 分支 run 分别在 e15 `+0.287151 dB`、e10 `+0.401175 dB` 后停止，均不是结论 |
| 数据裁剪合同（硬约束） | 所有入口训练固定 `RandomCrop(256)+RandomHorizontalFlip`，验证固定 `CenterCrop(256)`；loader 构建时执行硬断言 |

这里严格区分：

```text
sender oracle:    img,x1 -> E2 -> q2 -> D2/combiner -> x2
receiver inference: z1 -> x1; G(z1,x1) -> q2_hat -> D2/combiner -> x2_hat
```

oracle 的 `PSNR(x2)>PSNR(x1)` 不代表 receiver 已达成。receiver 推断禁止使用 `img/z2/q2/真实索引`；训练时它们只能作为 loss target。

## 已有文件

- [`contracts.py`](./contracts.py)：`ReceiverCondition(z1,x1)` 与 no-leak 结构审计；
- [`receiver_models.py`](./receiver_models.py)：FSQ direct/index/AR、channel-VQ 条件 AR、image-VQ 条件 diffusion；
- [`train_fsq_receiver.py`](./train_fsq_receiver.py)：加载 FSQ oracle 的 receiver-only 训练/验证，支持 frozen baseline 及 receiver-side D2/combiner adaptation；
- [`run_fsq_receiver_cnn.sh`](./run_fsq_receiver_cnn.sh)：CNN-FSQ launcher；
- [`test_receiver_contract.py`](./test_receiver_contract.py)：可执行 no-leak、mixed-radix 与无 E2 decode 审计；
- [`vq_modules.py`](./vq_modules.py)：image-VQ/channel-VQ 共享 nested-prefix 核心与 collapse diagnostics；
- [`train_layer2_vq_nested.py`](./train_layer2_vq_nested.py)：CNN/Swin 的 direct Layer2 shared-prefix VQ、receiver hard-q 闭环及统一验收；
- [`run_vq_nested.sh`](./run_vq_nested.sh)、[`run_vq_matrix.sh`](./run_vq_matrix.sh)：单 cell 与 2x2 主矩阵 launcher；
- [`run_vq_grouped_channel.sh`](./run_vq_grouped_channel.sh)：channel-owned grouped VQ 正式 `C256,K512/1024/4096` launcher；
- [`run_vq_highdim_oracle.sh`](./run_vq_highdim_oracle.sh)：任意 `embedding_dim` 的 oracle launcher；
- [`train_channel_vq_ar.py`](./train_channel_vq_ar.py)、[`run_channel_vq_ar.sh`](./run_channel_vq_ar.sh)：global/grouped channel-VQ 条件 AR 接收端；
- [`train_image_vq_diffusion.py`](./train_image_vq_diffusion.py)、[`run_image_vq_diffusion.sh`](./run_image_vq_diffusion.sh)：image-VQ q/embedding 条件扩散接收端；
- [`train_continuous_q_receiver.py`](./train_continuous_q_receiver.py)：任意 D 连续 q2_hat、Layer1 初始化 D2 和 residual/UNet receiver baseline；
- [`train_continuous_q_mix_receiver.py`](./train_continuous_q_mix_receiver.py)：读取严格的 `train-simplex` JSON，固定多 generator q2 mix，训练单一 receiver D2/combiner；默认以 final checkpoint 为结论，不用验证集选模；
- [`eval_continuous_q_mix_receiver.py`](./eval_continuous_q_mix_receiver.py)：只读地在 full valid-100 CenterCrop 上评估固定 q mix 的 raw/EMA 后端，并执行 no-leak/ablation 审计；
- [`run_continuous_q_mix_receiver.sh`](./run_continuous_q_mix_receiver.sh)：上述固定 top-3 q2 mix 的正式 launcher，默认用 v12 单 receiver 初始化但不改变 JSON 内的冻结 q 权重；
- [`train_receiver_information_teacher.py`](./train_receiver_information_teacher.py)：仅训练期的 `(z1,x1)->image/x1 residual` 经验恢复上界诊断；它有独立 no-leak/crop 合同和 full-val PSNR delta 报告，但不是部署路径；
- [`eval_continuous_q_tta.py`](./eval_continuous_q_tta.py)、[`calibrate_continuous_q_residual.py`](./calibrate_continuous_q_residual.py)：receiver-only D4 q ensemble 与只用训练集选 alpha 的残差尺度诊断；
- [`eval_continuous_q_ensemble.py`](./eval_continuous_q_ensemble.py)：单参考 D2/combiner 的多 generator q2 ensemble；`--selection-mode train-simplex` 只在随机裁剪训练集选择权重/可选 top-k 子集，随后固定权重做一次 100 张中心裁剪验证；可选 `train-gated-simplex` 仍只读 `(z1,x1)`，目前仅有合同 smoke、没有 full-val 结果；
- [`analyze_vq_matrix.py`](./analyze_vq_matrix.py)：按“Layer1 × VQ family”汇总，Layer2 架构和 D 作为每 cell 的自由变量，D scaling 另表审核；
- [`results-vq/embedding_scaling_fixed_epoch.md`](./results-vq/embedding_scaling_fixed_epoch.md)：image/channel 两族固定 epoch 的 D、K 容量对照；
- [`results-vq/swin_global_channel_vq_embedding_scaling_epoch15.md`](./results-vq/swin_global_channel_vq_embedding_scaling_epoch15.md)：Swin global channel-VQ 的 D128/D256 固定 epoch15、全 valid-100 容量与非塌陷证据；
- [`logs/`](./logs/)：当前 smoke/早期训练日志，不能自动等同于收敛结果。

## 快速检查

```bash
cd /workspace/yongjia/paper_code/CDDM

conda run -n cddm_ddnm python MY-V2/jscc-f/explore-2/train_fsq_receiver.py --help

conda run -n cddm_ddnm python MY-V2/jscc-f/explore-2/train_fsq_receiver.py \
  --smoke-shapes --route direct_q --condition-mode z1_x1

conda run --no-capture-output -n cddm_ddnm \
  python MY-V2/jscc-f/explore-2/train_fsq_receiver.py \
  --cpu --smoke-shapes --smoke-batch-size 1 \
  --route direct_q --condition-mode z1_x1 \
  --receiver-d2-arch qonly-highres-residual --receiver-d2-width 8 --receiver-d2-blocks 1 \
  --independent-receiver-d2 --finetune-d2 --receiver-combiner oracle \
  --init-receiver-checkpoint MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-independent-d2-v5/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_d2ft-oracle_independent-d2_best.pth

conda run --no-capture-output -n cddm_ddnm \
  python MY-V2/jscc-f/explore-2/train_receiver_information_teacher.py --smoke

conda run --no-capture-output -n cddm_ddnm \
  python MY-V2/jscc-f/explore-2/train_continuous_q_receiver.py --smoke-shapes

conda run --no-capture-output -n cddm_ddnm \
  python MY-V2/jscc-f/explore-2/train_continuous_q_mix_receiver.py --smoke-contract

conda run --no-capture-output -n cddm_ddnm \
  python MY-V2/jscc-f/explore-2/test_receiver_contract.py --real-k125 auto

conda run -n cddm_ddnm python MY-V2/jscc-f/explore-2/train_layer2_vq_nested.py \
  --arch swin --layer2-arch cnn --vq-family image-vq \
  --latent-c 256 --embedding-dim 512 --rates 256,1024 \
  --rate-weights 1 --smoke-shapes

conda run -n cddm_ddnm python MY-V2/jscc-f/explore-2/train_layer2_vq_nested.py \
  --arch cnn --vq-family channel-vq --channel-codebook-mode grouped \
  --latent-c 256 --embedding-dim 512 --rates 512,1024 \
  --rate-weights 1 --codebook-init channel-balanced --smoke-shapes
```

主 launcher：

```bash
GPU=0 ROUTE=direct_q CONDITION_MODE=z1_x1 \
VERSION=cnn-fsq-k4913-direct-hard-rerun \
bash MY-V2/jscc-f/explore-2/run_fsq_receiver_cnn.sh --hard-fsq
```

运行前先检查是否已有相同 `VERSION` 的进程或 checkpoint，避免重复任务和追加日志混淆。

## 执行优先级

1. 用 full-val `>1 dB` 的 oracle 作为接收端主候选；
2. channel-VQ 优先条件 AR 索引，image-VQ 优先条件 diffusion q/embedding；
3. 最终统一按 full-val `delta_rx>=0.5 dB`、zero/shuffle 和 condition-shuffle 验收；
4. image-VQ/global-channel VQ 使用 `K=256/1024/4096`；grouped channel-VQ 使用 `K=512/1024/4096`，不能把两者的 K 当作等码率；
5. 同时报告 K 增量和独立 D sweep，禁止用 used/PPL 代替 PSNR 表达力证据。

VQ 术语必须准确：`C` 是 Layer2 基础 latent 通道数，`D=--embedding-dim` 是码字维度。image-VQ 的码字为 `[D]`；channel-VQ 把每个通道 map 投影为 `[D]`。两者的 D 都可独立取 256/512/1024 等值。

## Grouped channel-VQ 合同

`--channel-codebook-mode grouped` 使用一个 round-major 共享码本；原生 D 保存为 `[Kmax,16,16]`，非原生 D 保存为 `[Kmax,1,D]`。通道 `c` 在前缀 K 中只能选择：

```text
c + C*arange(K/C)
```

因此 K 必须整除 C；每通道候选数是 `K/C`，真实名义码率为 `C*log2(K/C)` bits/image。正式 C256 三档 K512/K1024/K4096 分别是 2/4/16 候选/通道，对应 256/512/1024 bits/image（0.00390625/0.0078125/0.015625 bpp）。这与 image-VQ/global-channel 的 `log2(K)` 语义不同。

K=C 只有一个固定候选/通道，索引比特为 0，无法形成跨样本 shuffle relevance；它只能配合 `--allow-single-candidate-grouped-diagnostic` 做结构诊断，永远不能通过 promotion gate。正式三档全部必须通过 local PPL/top1、跨样本 channel variation、zero/shuffle、逐图 paired gain 和 PSNR gate。初始化和死码刷新都保持同通道来源，不允许把别的 channel map 写入该通道的 rows。

正式 CNN 入口：

```bash
cd /workspace/yongjia/paper_code/CDDM
GPU=0 ARCH=cnn SEED=20260624 \
VERSION=cnn-channel-vq-grouped-c256-k512-1024-4096-seed20260624 \
bash MY-V2/jscc-f/explore-2/run_vq_grouped_channel.sh
```

## 输出边界

所有新代码、launcher、日志、checkpoint、JSON/CSV 和总结都写入本目录。允许只读加载 `../explore` 中已经验证的 oracle checkpoint 和实验总结，但不得把新产物写回旧探索目录。

所有 receiver 与 oracle 入口均保存结构合同、日志和 checkpoint；只有完整 DIV2K valid 100 图指标会进入最终达标表，单 batch smoke 只证明计算图闭合。
