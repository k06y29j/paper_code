#!/usr/bin/env bash
# Low-dimensional receiver-only calibration from the archived e60 EMA
# D2+combiner.  The original train-selected top-3 generators are reconstructed
# from SPEC and remain frozen/eval; validation only monitors the final-epoch
# protocol and never selects a checkpoint.
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
EXP="$ROOT/MY-V2/jscc-f/explore-2"
SCRIPT="$EXP/train_continuous_q_mix_frozen_calibrator.py"

GPU="${GPU:-0}"
VERSION="${VERSION:-cnn-continuous-q-mix-top3-frozen-calibrator-e60-v1}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEST_BATCH="${TEST_BATCH:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
VAL_EVERY="${VAL_EVERY:-2}"
SIMPLEX_LR="${SIMPLEX_LR:-1e-3}"
AFFINE_LR="${AFFINE_LR:-3e-4}"
ADAPTER_TYPE="${ADAPTER_TYPE:-none}"
ADAPTER_WIDTH="${ADAPTER_WIDTH:-32}"
ADAPTER_LR="${ADAPTER_LR:-1e-4}"

SPEC="$EXP/results-receiver/continuous_q_d512_crossmodel5_train_simplex_top3.json"
E60="$EXP/checkpoints-continuous-q-mix/cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-v1/continuous_q_mix_receiver_final_e60_before_continue.pth"
SAVE_DIR="$EXP/checkpoints-continuous-q-mix"
LOG_DIR="$EXP/logs-continuous-q-mix"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

cd "$ROOT"
if [[ "${SMOKE_CONTRACT:-0}" == "1" ]]; then
  conda run --no-capture-output -n cddm_ddnm \
    python "$SCRIPT" --smoke-contract \
    2>&1 | tee -a "$LOG_DIR/${VERSION}-contract-smoke.log"
  exit "${PIPESTATUS[0]}"
fi

env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$SCRIPT" \
    --ensemble-spec "$SPEC" \
    --e60-checkpoint "$E60" \
    --expected-start-epoch 60 \
    --expected-start-delta 0.4898087310791013 \
    --start-reproduction-tolerance 1e-4 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --test-batch "$TEST_BATCH" \
    --num-workers "$NUM_WORKERS" \
    --val-num-workers "$VAL_NUM_WORKERS" \
    --val-every "$VAL_EVERY" \
    --final-loss mse \
    --hard-example-power 0.5 \
    --hard-example-min-weight 0.25 \
    --hard-example-max-weight 4 \
    --simplex-lr "$SIMPLEX_LR" \
    --affine-lr "$AFFINE_LR" \
    --adapter-type "$ADAPTER_TYPE" \
    --adapter-width "$ADAPTER_WIDTH" \
    --adapter-lr "$ADAPTER_LR" \
    --weight-decay 0 \
    --grad-clip-norm 1 \
    --lambda-simplex-anchor 0 \
    --lambda-affine-anchor 0 \
    --lambda-adapter-anchor 0 \
    --min-delta 0.5 \
    --min-condition-drop 0.1 \
    --save-dir "$SAVE_DIR" \
    --version "$VERSION" \
    --seed 20260713 \
    "$@" \
    2>&1 | tee -a "$LOG_DIR/${VERSION}.log"
