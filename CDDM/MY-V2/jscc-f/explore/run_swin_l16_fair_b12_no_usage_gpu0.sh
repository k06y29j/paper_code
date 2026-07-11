#!/usr/bin/env bash
set -euo pipefail

# Step 1A: exact control for the historical l16 usage-KL run.  It matches
# Swin, FSQ=16x16x16, BatchNorm, batch=12, seed, and validation cadence; only
# lambda_usage changes.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec conda run --no-capture-output -n cddm_ddnm python -u MY-V2/jscc-f/explore/train_stage3_fsq_explore.py \
  --arch swin --version swin-l16-fair-b12-no-usage-e40 \
  --save-dir MY-V2/jscc-f/explore/checkpoints/swin-l16-fair-b12-no-usage-e40 \
  --fsq-d 3 --fsq-levels 16,16,16 \
  --fsq-normalizer batch --lambda-usage 0 --usage-mode joint \
  --epochs 40 --batch-size 12 --test-batch 1 --num-workers 0 --val-num-workers 0 \
  --val-every 5 --latest-every 5 \
  --selection-min-drop-zero 0.5 --selection-min-drop-shuffle 0.5
