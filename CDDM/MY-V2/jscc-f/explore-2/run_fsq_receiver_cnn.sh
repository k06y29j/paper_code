#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
SCRIPT="$ROOT/MY-V2/jscc-f/explore-2/train_fsq_receiver.py"
SAVE_ROOT="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-receiver"
LOG_ROOT="$ROOT/MY-V2/jscc-f/explore-2/logs"
mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

GPU="${GPU:-0}"
ROUTE="${ROUTE:-direct_q}"
CONDITION_MODE="${CONDITION_MODE:-z1_x1}"
VERSION="${VERSION:-cnn-fsq-k4913-${ROUTE}-${CONDITION_MODE}-v1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
EXTRA_ARGS=("$@")

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$SCRIPT" \
  --route "$ROUTE" \
  --condition-mode "$CONDITION_MODE" \
  --version "$VERSION" \
  --save-dir "$SAVE_ROOT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --val-num-workers "$VAL_NUM_WORKERS" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG_ROOT/${VERSION}.log"

