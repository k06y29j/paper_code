#!/usr/bin/env bash
# Reproduce the CNN fixed-q-mix receiver configuration archived as
# cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-v1.
#
# This intentionally uses a fresh output-version default so the archived e60
# checkpoint is never overwritten.  Only the output directory name differs
# from the original run; model, data, q-mix, optimizer, and loss settings are
# identical.
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
EXP="$ROOT/MY-V2/jscc-f/explore-2"

GPU="${GPU:-1}"
VERSION="${VERSION:-cnn-continuous-q-mix-top3-v9mse-ema05fix-e60-repro-v1}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEST_BATCH="${TEST_BATCH:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
VAL_EVERY="${VAL_EVERY:-5}"

SPEC="$EXP/results-receiver/continuous_q_d512_crossmodel5_train_simplex_top3.json"
INIT="$EXP/checkpoints-continuous-q/cnn-continuous-q-d512-hardp05-v9/continuous_q_receiver_best.pth"

exec env \
  GPU="$GPU" \
  ENSEMBLE_SPEC="$SPEC" \
  RECEIVER_INIT_CHECKPOINT="$INIT" \
  VERSION="$VERSION" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  TEST_BATCH="$TEST_BATCH" \
  NUM_WORKERS="$NUM_WORKERS" \
  VAL_NUM_WORKERS="$VAL_NUM_WORKERS" \
  VAL_EVERY="$VAL_EVERY" \
  bash "$EXP/run_continuous_q_mix_receiver.sh" \
    --receiver-init-state auto \
    --d2-type layer1 \
    --lambda-final 1 \
    --lambda-u2 0 \
    --final-loss mse \
    --hard-example-power 0.5 \
    --hard-example-min-weight 0.25 \
    --hard-example-max-weight 4 \
    --d2-lr 1e-5 \
    --combiner-lr 2e-5 \
    --weight-decay 0 \
    --grad-clip-norm 1 \
    --ema-decay 0.5 \
    --min-delta 0.5 \
    --min-condition-drop 0.1 \
    --checkpoint-selection final \
    --seed 20260713
