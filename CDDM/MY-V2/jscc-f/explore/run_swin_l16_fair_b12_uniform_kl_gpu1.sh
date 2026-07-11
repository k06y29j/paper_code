#!/usr/bin/env bash
set -euo pipefail

# Step 1B: same protocol as 1A, but the original KL-to-uniform objective.
# This isolates batch-size/BatchNorm effects from the usage regularizer.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec conda run --no-capture-output -n cddm_ddnm python -u MY-V2/jscc-f/explore/train_stage3_fsq_explore.py \
  --arch swin --version swin-l16-fair-b12-uniform-kl-e40 \
  --save-dir MY-V2/jscc-f/explore/checkpoints/swin-l16-fair-b12-uniform-kl-e40 \
  --fsq-d 3 --fsq-levels 16,16,16 \
  --fsq-normalizer batch --lambda-usage 0.005 --usage-mode joint --usage-objective uniform_kl --usage-tau 0.12 \
  --epochs 40 --batch-size 12 --test-batch 1 --num-workers 0 --val-num-workers 0 \
  --val-every 5 --latest-every 5 \
  --selection-min-drop-zero 0.5 --selection-min-drop-shuffle 0.5
