# JSCC-f CodeFormer-v2

这个目录实现两阶段的 Layer1 `x1` 条件码本重建，不复用旧 `codeformer` 的 Transformer code-prediction 训练目标。

```text
Stage1  img + frozen Layer1 -> x1
        concat(img, x1) [B,6,256,256] -> HQ encoder -> VQ -> decoder -> u2
        concat(x1, u2) [B,6,256,256] -> combiner -> x2

Stage2  frozen Layer1 -> x1 [B,3,256,256]
        x1 -> LQ encoder -> frozen Stage1 VQ -> frozen decoder + trainable SFT/CFT -> u2
        concat(x1, u2) [B,6,256,256] -> frozen combiner -> x2_hat
```

两阶段的 latent 都是 `[B,256,16,16]`。Stage1 的 HQ encoder 唯一输入是 `[img,x1]` 的六通道拼接；Stage2 的 LQ encoder 唯一输入是 `x1`，不会读取 `img`、Stage1 `z2/q2` 或 oracle code。Stage2 从 Stage1 HQ encoder 的 `x1` 输入 stem（通道 `3:6`）初始化 LQ encoder，避免将 `img` 分支权重带入部署输入。

Stage2 严格冻结 VQ codebook、decoder 基础层及 combiner；decoder 的 `fuse_blocks`（128/64/32 的 SFT/CFT）会重新设为可训练，和 LQ encoder 一起被优化。`256×256` 不参与 SFT/CFT。

## 训练

```bash
# Stage1：训练 HQ [img,x1] 六通道 VQ + decoder + combiner
conda run -n cddm_ddnm python MY-V2/jscc-f/codeformer-v2/train.py \
  --stage stage1 --version cnn-hq-v1 --layer1-arch cnn --batch-size 8

# Stage2：冻结 Stage1 codebook/decoder/combiner，只训练 LQ encoder + SFT/CFT
conda run -n cddm_ddnm python MY-V2/jscc-f/codeformer-v2/train.py \
  --stage stage2 --version cnn-lq-v1 --layer1-arch cnn --batch-size 8 \
  --stage1-ckpt MY-V2/jscc-f/codeformer-v2/checkpoints/jsccf_codeformer_v2_stage1_cnn-hq-v1_best.pth
```

日志默认保存为 `--save-dir/jsccf_codeformer_v2_<stage>_<version>.log`；可用 `--log-file` 覆盖。每个 epoch 记录 `code_used`、`code_usage_ratio`、`code_entropy_bits`、`code_perplexity`、`code_top1_frac` 与 `psnr_x1`/`psnr_x2` 或 `psnr_x2_hat`。

## Smoke

```bash
conda run -n cddm_ddnm python MY-V2/jscc-f/codeformer-v2/train.py \
  --smoke-shapes --cpu --base-channels 16 --codebook-size 32
```

它验证 Stage1 的六通道 HQ 输入和 Stage2 的 x1-only 输入、所有输出形状、Stage1 状态移交、冻结边界，以及 Stage2 LQ encoder/SFT 的反传。
