#!/usr/bin/env bash
set -euo pipefail

# Step 2: do not maximize entropy.  After a reconstruction-only warmup, only
# penalize the shortfall below H=7.5 bits (PPL=181), well below uniform 12 bits.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec conda run --no-capture-output -n cddm_ddnm python -u MY-V2/jscc-f/explore/train_stage3_fsq_explore.py \
  --arch swin --version swin-l16-floor7p5-b12-batch-e60 \
  --save-dir MY-V2/jscc-f/explore/checkpoints/swin-l16-floor7p5-b12-batch-e60 \
  --fsq-d 3 --fsq-levels 16,16,16 \
  --fsq-normalizer batch --lambda-usage 0.001 --usage-mode joint --usage-objective entropy_floor \
  --usage-target-bits 7.5 --usage-warmup-epochs 10 --usage-ramp-epochs 10 --usage-tau 0.12 \
  --epochs 60 --batch-size 12 --test-batch 1 --num-workers 0 --val-num-workers 0 \
  --val-every 5 --latest-every 5 \
  --selection-min-drop-zero 0.5 --selection-min-drop-shuffle 0.5
