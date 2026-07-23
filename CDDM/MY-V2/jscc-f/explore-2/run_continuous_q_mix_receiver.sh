#!/usr/bin/env bash
# Train one receiver D2+combiner behind the fixed train-selected q2 mix.
# The JSON is a q-space ensemble selection artifact, never an image ensemble.
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
SCRIPT="$ROOT/MY-V2/jscc-f/explore-2/train_continuous_q_mix_receiver.py"
GPU="${GPU:-0}"
ENSEMBLE_SPEC="${ENSEMBLE_SPEC:-$ROOT/MY-V2/jscc-f/explore-2/results-receiver/continuous_q_d512_crossmodel5_train_simplex_top3.json}"
# v12 is the strongest known single continuous-q initialization.  It can be
# overridden, while the q-mix members/weights always remain those in SPEC.
RECEIVER_INIT_CHECKPOINT="${RECEIVER_INIT_CHECKPOINT:-$ROOT/MY-V2/jscc-f/explore-2/checkpoints-continuous-q/cnn-continuous-q-d512-hardp05-hybrid64-v12/continuous_q_receiver_best.pth}"
VERSION="${VERSION:-cnn-continuous-q-mix-top3-v1}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEST_BATCH="${TEST_BATCH:-2}"
NUM_WORKERS="${NUM_WORKERS:-12}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
VAL_EVERY="${VAL_EVERY:-5}"
SAVE_DIR="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-continuous-q-mix"
LOG_DIR="$ROOT/MY-V2/jscc-f/explore-2/logs-continuous-q-mix"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$SCRIPT" \
  --ensemble-spec "$ENSEMBLE_SPEC" \
  --receiver-init-checkpoint "$RECEIVER_INIT_CHECKPOINT" \
  --version "$VERSION" \
  --save-dir "$SAVE_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch "$TEST_BATCH" \
  --num-workers "$NUM_WORKERS" \
  --val-num-workers "$VAL_NUM_WORKERS" \
  --val-every "$VAL_EVERY" \
  --checkpoint-selection final \
  "$@" \
  2>&1 | tee -a "$LOG_DIR/${VERSION}.log"
