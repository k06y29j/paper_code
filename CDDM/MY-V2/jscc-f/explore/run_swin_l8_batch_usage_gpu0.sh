#!/usr/bin/env bash
set -euo pipefail

# Early feasibility gate: the baseline 8x8x8 Swin tokenizer had already
# collapsed to one joint token at epoch 5.  This run must retain non-trivial
# token entropy and q3 ablation drops before it is extended to 400 epochs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

exec conda run --no-capture-output -n cddm_ddnm python MY-V2/jscc-f/explore/train_stage3_fsq_explore.py \
  --arch swin \
  --version swin-l8-batch-usage-e10 \
  --save-dir MY-V2/jscc-f/explore/checkpoints/swin-l8-batch-usage-e10 \
  --fsq-d 3 \
  --fsq-levels 16,16,16 \
  --fsq-normalizer batch \
  --lambda-usage 0.005 \
  --usage-mode joint \
  --usage-tau 0.12 \
  --epochs 200 \
  --batch-size 12 \
  --test-batch 1 \
  --num-workers 16 \
  --val-num-workers 4 \
  --val-every 5 \
  --latest-every 5
