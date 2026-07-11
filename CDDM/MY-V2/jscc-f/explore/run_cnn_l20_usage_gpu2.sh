#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
exec conda run --no-capture-output -n cddm_ddnm python -u MY-V2/jscc-f/explore/train_stage3_fsq_explore.py \
  --arch cnn --version cnn-l20-joint-usage-e40 \
  --fsq-d 3 --fsq-levels 20,16,16 \
  --fsq-normalizer group --lambda-usage 0.0005 --usage-mode joint --usage-tau 0.12 \
  --epochs 40 --batch-size 12 --test-batch 1 --num-workers 0 --val-num-workers 0 \
  --val-every 5 --latest-every 5
