# JSCC-f CodeFormer（256×256 适配）

这个目录以 [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) 的“VQ 码本 + Transformer 代码预测 + SFT 条件融合”思路，复现到 JSCC-f 的两层链路。

部署图如下：

```text
输入图像 -> 冻结 Layer1 (CNN 或 Swin) -> x1 [B,3,256,256]
                                       -> VQ encoder -> Transformer -> 256 个码本 ID
                                       -> VQ decoder + x1 多尺度 SFT -> x_hat [B,3,256,256]
```

训练的 HQ 图像只用于阶段 I 的 VQ 重建，以及阶段 II 的真实码本 ID / 重建监督；`CodeFormer.restore(x1)` 的推理接口只接受 `x1`，不会读取 HQ 图像、Layer1 `z1` 或任何发送端 Layer2 信息。

## 已固定的结构合同

- 输入、Layer1 `x1`、输出：`[B,3,256,256]`。
- VQ latent：`[B,256,16,16]`，共 256 个空间 token。
- 原 CodeFormer 的 512→16 五次下采样被改为 256→16 四次下采样；编码/解码 level 为 `256, 128, 64, 32, 16`。
- SFT/CFT 只融合 `128×128`、`64×64`、`32×32` 三层。`256×256` 的多尺度特征明确不构造、不融合。
- `--layer1-arch {cnn,swin}` 严格加载已有 Layer1 checkpoint；默认分别是 `jscc_f_cnn_layer1_cnn_best.pth` 与 `jscc_f_no-c1_layer1_best.pth`。

## 两阶段训练

先训练 HQ VQ autoencoder（阶段 I）：

```bash
conda run -n cddm_ddnm python MY-V2/jscc-f/codeformer/train.py \
  --stage vq --version hq-vq-1024 --batch-size 8
```

然后固定 VQ 的码本/基础 decoder 和 Layer1，以 Layer1 `x1` 训练 Transformer 与 SFT：

```bash
conda run -n cddm_ddnm python MY-V2/jscc-f/codeformer/train.py \
  --stage codeformer --version cnn-cf-v1 --layer1-arch cnn \
  --vq-ckpt MY-V2/jscc-f/codeformer/checkpoints/jsccf_codeformer_vq_hq-vq-1024_best.pth \
  --batch-size 8
```

Swin Layer1 只需改为 `--layer1-arch swin`（或显式给出与该架构匹配的 `--layer1-ckpt`）。默认阶段 II 仍可训练 VQ encoder，与原 CodeFormer 保持一致；若也要固定它，加入 `--freeze-vq-encoder`。

训练期使用 soft codebook lookup，让重建损失能反传到 Transformer；验证使用 argmax hard code，记录 `psnr_x1`、`psnr_final`、`delta_x1` 和 token 准确率。训练 smoke 不能作为质量结论。

每次运行会把 stdout/stderr 追加写入 `--save-dir/jsccf_codeformer_<stage>_<version>.log`；可用 `--log-file /path/to/run.log` 覆盖。阶段 I 的 `code_used`、`code_usage_ratio`、`code_entropy_bits`、`code_perplexity`、`code_top1_frac` 在完整 epoch 的所有 VQ token 上统计。阶段 II 以 `pred_` 和 `target_` 前缀分别记录部署 argmax 预测码和 HQ 监督码的相同指标。

## 离线形状检查

为避免默认大模型的 CPU 检查过慢，可用同一拓扑的小宽度实例：

```bash
conda run -n cddm_ddnm python MY-V2/jscc-f/codeformer/train.py \
  --smoke-shapes --cpu --base-channels 16 --transformer-width 64 \
  --transformer-layers 1 --transformer-heads 4 --codebook-size 32
```

这会断言 latent 为 `[B,256,16,16]`、logits 为 `[B,256,K]`、输出为 `[B,3,256,256]`，并打印被保留的融合尺度。

## 推理

`inference.py` 先用保存的 Layer1 生成 `x1`，然后只调用 `CodeFormer.restore(x1)`；不会把原图送进 CodeFormer：

```bash
conda run -n cddm_ddnm python MY-V2/jscc-f/codeformer/inference.py \
  --checkpoint MY-V2/jscc-f/codeformer/checkpoints/jsccf_codeformer_codeformer_cnn-cf-v1_best.pth \
  --input-path /path/to/image-or-directory --output-dir /path/to/results
```

## 上游归属

这里是针对 JSCC-f 的独立 PyTorch 实现，不含上游预训练权重。若分发上游代码或权重，请遵守 [CodeFormer 的 NTU S-Lab License 1.0](https://github.com/sczhou/CodeFormer/blob/master/LICENSE)。
