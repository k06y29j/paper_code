#!/usr/bin/env bash
set -euo pipefail

# Same 5,120-word CNN contract as the existing l20x16x16 run.  The small
# joint usage-KL favors a broader transmitted vocabulary without changing the
# reconstruction loss or the established GroupNorm tokenizer path.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec conda run --no-capture-output -n cddm_ddnm python MY-V2/jscc-f/explore/train_stage3_fsq_explore.py \
  --arch cnn \
  --version cnn-l20-joint-usage-e10 \
  --save-dir MY-V2/jscc-f/explore/checkpoints/cnn-l20-joint-usage-e10 \
  --fsq-d 3 \
  --fsq-levels 20,16,16 \
  --fsq-normalizer group \
  --lambda-usage 0.001 \
  --usage-mode joint \
  --usage-tau 0.12 \
  --epochs 10 \
  --batch-size 12 \
  --test-batch 1 \
  --num-workers 0 \
  --val-num-workers 0 \
  --val-every 5 \
  --latest-every 5
