#!/usr/bin/env bash
# Train one receiver D2+combiner behind the fixed train-selected q2 mix.
# The JSON is a q-space ensemble selection artifact, never an image ensemble.
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
SCRIPT="$ROOT/MY-V2/jscc-f/cnn-work/train_continuous_q_mix_receiver.py"
GPU="${GPU:-0}"
WORK="$ROOT/MY-V2/jscc-f/cnn-work"
# The migrated package keeps model artifacts external by default so code and
# scripts are self-contained without duplicating multi-hundred-MB checkpoints.
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/MY-V2/jscc-f/explore-2}"
ENSEMBLE_SPEC="${ENSEMBLE_SPEC:-$WORK/configs/accepted_qmix_top3_v9_ema.json}"
RECEIVER_INIT_CHECKPOINT="${RECEIVER_INIT_CHECKPOINT:-$ARTIFACT_ROOT/checkpoints-continuous-q/cnn-continuous-q-d512-hardp05-v9/continuous_q_receiver_best.pth}"
VERSION="${VERSION:-cnn-continuous-q-mix-top3-v1}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEST_BATCH="${TEST_BATCH:-2}"
NUM_WORKERS="${NUM_WORKERS:-12}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
VAL_EVERY="${VAL_EVERY:-5}"
SAVE_DIR="$WORK/checkpoints-continuous-q-mix"
LOG_DIR="$WORK/logs-continuous-q-mix"
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
