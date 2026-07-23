# Swin global channel-VQ：固定 epoch 15 的 embedding scaling

## 口径与可复现性

本报告只比较下列两个 **sender oracle** 日志中同一个验证时刻的
`[nested-vq val 015]`，不根据各自的 `best` checkpoint 选点，也不混入
receiver 结果：

- `logs-vq/swin-channel-vq-global-c256-d128-fullhard-v5.log`
- `logs-vq/swin-channel-vq-global-c256-d256-fullhard-fixed20-v6.log`

共同实验合同为 Layer1=`swin`、Layer2=`swin`、`vq_family=channel-vq`、
`channel_codebook_mode=global`、`C=256`、K=`256/1024/4096`、K-means 初始化、
seed=`20260713`、batch=`4`、enhanced combiner（width=64, blocks=8）、
`--oracle-only`、无连续 latent curriculum。嵌入维度 D 是此表的表示变量：
128 与 256。D128 总训练预算为 80 epoch，D256 为 20 epoch；本表固定取两者
epoch 15，因此不把后续 epoch 或最佳 checkpoint 当作 D 的收益。

seed、完整 VQ schedule、combiner 宽度和冻结标记已从两个对应 checkpoint 的
内嵌 `args` 交叉核对；checkpoint 只用于配置核对，**不用其可能属于其他 epoch
的 metrics** 填表。表中全部数值始终来自上述 epoch-15 日志行。

训练集采用 `RandomCrop(256)+RandomHorizontalFlip`；验证集采用
`CenterCrop(256)`，全验证集为 100 张图像（`val=100`）。两份日志的原始
运行头均明确记录了该裁剪合同和 `oracle_only_phase=True`。因此下表是
oracle `x2` 的容量结果，**不是**接收端 `x2_hat`，不能用来宣称预测/生成
q2 已达到接收端增益目标。

## 固定 epoch 15 的 oracle PSNR 与 D 增益

所有单元格都来自同一次 100 图 CenterCrop 全验证；
`D 增益 = Delta(D256) - Delta(D128)`，其中
`Delta = PSNR(x2_K) - PSNR(x1)`。

`PSNR(x1)=27.565002 dB`（两行相同）。

| K | D128 PSNR(x2) | D128 Delta x1 | D256 PSNR(x2) | D256 Delta x1 | D 增益（D256-D128） |
|---:|---:|---:|---:|---:|---:|
| 256 | 27.882846 | +0.317844 | 27.910850 | +0.345848 | +0.028004 |
| 1024 | 27.977311 | +0.412309 | 28.030435 | +0.465433 | +0.053124 |
| 4096 | 28.039196 | +0.474194 | 28.120500 | +0.555498 | +0.081304 |

结论：D 从 128 增至 256 后，三个固定 K 的 oracle Delta 均严格增加；D256
在 K4096 达到 `+0.555498 dB`。这是一份健康的固定 epoch 容量 scaling
证据，但最高 oracle 增益仍低于 `+1 dB`，且不构成 `x2_hat` 的证据。

## 严格 K 单调与非塌陷证据

运行门槛是相邻 K 的 `paired_gain >= 0.01 dB`、
`paired_strict >= 0.55`，并要求 global channel-VQ 的
`PPL/K >= 0.01`、`top1 <= 0.25`、`index_change >= 0.01`。

| D | 相邻 K | paired gain (dB) | paired strict | 严格 K 结论 |
|---:|:---|---:|---:|:---|
| 128 | 256 -> 1024 | +0.094465 | 1.0000 | PASS |
| 128 | 1024 -> 4096 | +0.061885 | 0.9900 | PASS |
| 256 | 256 -> 1024 | +0.119585 | 0.9900 | PASS |
| 256 | 1024 -> 4096 | +0.090065 | 0.9900 | PASS |

| D | K | used/K | PPL | PPL/K | top1 | index change | zero / shuffle drop (dB) |
|---:|---:|:---|---:|---:|---:|---:|:---|
| 128 | 256 | 254/256 | 122.983 | 0.480401 | 0.166445 | 0.865804 | 0.293661 / 0.447282 |
| 128 | 1024 | 1010/1024 | 611.041 | 0.596719 | 0.065313 | 0.949333 | 0.388126 / 0.593623 |
| 128 | 4096 | 3866/4096 | 2545.900 | 0.621558 | 0.019023 | 0.982824 | 0.450011 / 0.708062 |
| 256 | 256 | 254/256 | 172.688 | 0.674561 | 0.028672 | 0.966706 | 0.319276 / 0.510761 |
| 256 | 1024 | 1009/1024 | 753.864 | 0.736195 | 0.013242 | 0.985608 | 0.438861 / 0.673033 |
| 256 | 4096 | 3888/4096 | 2938.944 | 0.717516 | 0.003633 | 0.994510 | 0.528926 / 0.791582 |

两种 D 的 PPL 都随 K 严格增加，全部 operating point 满足非塌陷阈值；日志
在 epoch 15 同时给出 `capacity_goal_met=1`、`noncollapse_goal_met=1`、
`usage_scaling_goal_met=1`、`oracle_goal_met=1`。zero/shuffle drop 也均高于
`0.1 dB`，所以这里的 PSNR 和 K 单调性不是未被使用的码本假象。

## 明确排除 D256 image-VQ 旧塌陷运行

本报告的 D256 行只来自 **global channel-VQ**，绝不把
`swin-image-vq-c256-d256-fullhard-fixed20-v4.log` 当作 channel-VQ scaling
成功结果。该 image-VQ 日志在 epoch 1 的全验证已经接近单码：K256
`PPL=1.0022/top1=0.99977`，K4096 `PPL=1.0441/top1=0.99523`；其 epoch 15
仍记录 `capacity_goal_met=0`、`oracle_goal_met=0`，K1024->K4096 的 paired
gain 只有 `+0.000465 dB`。故它既不属于本表的 VQ family，也不满足严格
K 容量门槛。

## 原始证据定位

- D128 运行合同、oracle-only、100 图裁剪合同：
  `logs-vq/swin-channel-vq-global-c256-d128-fullhard-v5.log:15-23`。
- D128 固定 epoch 15 的完整验证字典（所有表格数值与 gate）：
  `logs-vq/swin-channel-vq-global-c256-d128-fullhard-v5.log:49`。
- D256 运行合同、oracle-only、100 图裁剪合同：
  `logs-vq/swin-channel-vq-global-c256-d256-fullhard-fixed20-v6.log:15-23`。
- D256 固定 epoch 15 的完整验证字典（所有表格数值与 gate）：
  `logs-vq/swin-channel-vq-global-c256-d256-fullhard-fixed20-v6.log:49`。
- global-VQ 的 strict-K/non-collapse/oracle gate 实现：
  `train_layer2_vq_nested.py:1803-1856`。
- 排除的 D256 image-VQ 早期塌陷与 epoch-15 失败：
  `logs-vq/swin-image-vq-c256-d256-fullhard-fixed20-v4.log:25,49`。

最小复核命令（不训练）：

```bash
rg -n '^\[nested-vq val 015\]' \
  MY-V2/jscc-f/explore-2/logs-vq/swin-channel-vq-global-c256-d128-fullhard-v5.log \
  MY-V2/jscc-f/explore-2/logs-vq/swin-channel-vq-global-c256-d256-fullhard-fixed20-v6.log
```
