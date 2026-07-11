#!/usr/bin/env bash
set -euo pipefail

# Separates the two proposed anti-collapse mechanisms.  BatchNorm alone is
# the first SWIN candidate because the usage-KL=0.005 feasibility run kept all
# codes alive but consumed too much reconstruction capacity at validation.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec conda run --no-capture-output -n cddm_ddnm python MY-V2/jscc-f/explore/train_stage3_fsq_explore.py \
  --arch swin \
  --version swin-l8-batch-only-e10 \
  --save-dir MY-V2/jscc-f/explore/checkpoints/swin-l8-batch-only-e10 \
  --fsq-d 3 \
  --fsq-levels 8,8,8 \
  --fsq-normalizer batch \
  --lambda-usage 0 \
  --usage-mode joint \
  --usage-tau 0.12 \
  --epochs 400 \
  --batch-size 32 \
  --test-batch 1 \
  --num-workers 16 \
  --val-num-workers 4 \
  --val-every 5 \
  --latest-every 5
