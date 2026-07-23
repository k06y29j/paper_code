#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
SCRIPT="$ROOT/MY-V2/jscc-f/explore-3/train_layer2_predictable_fsq.py"
VERSION="${VERSION:-k125-joint125-predictable-v1}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TEST_BATCH="${TEST_BATCH:-4}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
SAVE_DIR="$ROOT/MY-V2/jscc-f/explore-3/checkpoints/$VERSION"
LOG_FILE="$ROOT/MY-V2/jscc-f/explore-3/logs/$VERSION.log"

mkdir -p "$SAVE_DIR" "$(dirname "$LOG_FILE")"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
exec conda run --no-capture-output -n cddm_ddnm python "$SCRIPT" \
  --version "$VERSION" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch "$TEST_BATCH" \
  --num-workers "$NUM_WORKERS" \
  --val-num-workers "$VAL_NUM_WORKERS" \
  --warmup-epochs 5 \
  --shape-ramp-epochs 20 \
  --predictor-lr 5e-5 \
  --sender-lr 1e-5 \
  --decoder-lr 2e-5 \
  --lambda-oracle 1.0 \
  --lambda-hard 1.5 \
  --lambda-soft 0.25 \
  --lambda-distill 0.1 \
  --lambda-receiver-code 0.001 \
  --lambda-receiver-q 0.01 \
  --lambda-sender-shape 0.001 \
  --lambda-entropy 0.05 \
  --entropy-floor-bits 6.0 \
  --lambda-anchor 0.05 \
  --goal-delta-db 0.5 \
  --min-oracle-delta 0.8 \
  --min-condition-drop 0.1 \
  --min-pred-drop 0.1 \
  --min-oracle-drop 0.5 \
  --min-sender-entropy 5.5 \
  --min-sender-codes 100 \
  --val-every 5 \
  --latest-every 5 \
  --save-dir "$SAVE_DIR" \
  --log-file "$LOG_FILE" \
  "$@"
