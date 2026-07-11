#!/usr/bin/env bash
set -euo pipefail

# Step 4: same 4,096-word / 12-bit contract as d3 16^3, but four 8-level
# scalar dimensions give D3 one more transmitted feature channel and a larger
# nearest-grid L2 separation.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec conda run --no-capture-output -n cddm_ddnm python -u MY-V2/jscc-f/explore/train_stage3_fsq_explore.py \
  --arch swin --version swin-d4-l8-floor7p5-b12-bnstateless-e60 \
  --save-dir MY-V2/jscc-f/explore/checkpoints/swin-d4-l8-floor7p5-b12-bnstateless-e60 \
  --fsq-d 4 --fsq-levels 8,8,8,8 \
  --fsq-normalizer batch_stateless --lambda-usage 0.001 --usage-mode joint --usage-objective entropy_floor \
  --usage-target-bits 7.5 --usage-warmup-epochs 10 --usage-ramp-epochs 10 --usage-tau 0.12 \
  --epochs 60 --batch-size 12 --test-batch 1 --num-workers 0 --val-num-workers 0 \
  --val-every 5 --latest-every 5 \
  --selection-min-drop-zero 0.5 --selection-min-drop-shuffle 0.5
