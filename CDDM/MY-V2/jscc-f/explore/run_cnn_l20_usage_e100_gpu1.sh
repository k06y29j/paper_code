#!/usr/bin/env bash
set -euo pipefail

# Promoted after the 10-epoch gate: at epoch 5 it matched the original PSNR
# while raising joint perplexity from about 185 to 4,347 / 5,120.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec conda run --no-capture-output -n cddm_ddnm python MY-V2/jscc-f/explore/train_stage3_fsq_explore.py \
  --arch cnn \
  --version cnn-l20-joint-usage-e100 \
  --save-dir MY-V2/jscc-f/explore/checkpoints/cnn-l20-joint-usage-e100 \
  --fsq-d 3 \
  --fsq-levels 20,16,16 \
  --fsq-normalizer group \
  --lambda-usage 0.001 \
  --usage-mode joint \
  --usage-tau 0.12 \
  --epochs 100 \
  --batch-size 12 \
  --test-batch 1 \
  --num-workers 0 \
  --val-num-workers 0 \
  --val-every 5 \
  --latest-every 5
