#!/usr/bin/env bash
# Strict, read-only valid100 verification of the accepted e60 CNN q-mix receiver.
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
WORK="$ROOT/MY-V2/jscc-f/cnn-work"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/MY-V2/jscc-f/explore-2}"
GPU="${GPU:-1}"
CHECKPOINT="${CHECKPOINT:-$ARTIFACT_ROOT/checkpoints-continuous-q-mix/cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-v1/continuous_q_mix_receiver_final_e60_before_continue.pth}"
OUTPUT="${OUTPUT:-results-receiver/cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-v1_e60_strict_fullval.json}"

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$WORK/eval_continuous_q_mix_receiver.py" \
    --checkpoint "$CHECKPOINT" \
    --state both \
    --batch-size 2 \
    --test-batch 2 \
    --num-workers 0 \
    --val-num-workers 0 \
    --output "$OUTPUT"
