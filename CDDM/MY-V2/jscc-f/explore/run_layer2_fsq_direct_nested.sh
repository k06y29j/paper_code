#!/usr/bin/env bash
set -euo pipefail

# One member of the fair nested FSQ grid.  Run with LEVELS=5,5,5 / 9,9,9 /
# 17,17,17 while holding every other environment variable fixed.  These odd
# grids are nested, so every lower-rate scalar reconstruction point also exists
# at the next rate.

ARCH="${ARCH:-cnn}"
GPU="${GPU:-0}"
FSQ_D="${FSQ_D:-3}"
LEVELS="${LEVELS:-5,5,5}"
NORMALIZER="${NORMALIZER:-group}"
CODEC_INIT="${CODEC_INIT:-compatible}"
COMBINER_MODE="${COMBINER_MODE:-original}"
FRESH_COMBINER="${FRESH_COMBINER:-0}"
BLEND_INIT="${BLEND_INIT:-0.1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
BN_CALIBRATION_BATCHES="${BN_CALIBRATION_BATCHES:-0}"
RESET_BEST_ON_RESUME="${RESET_BEST_ON_RESUME:-0}"
SEED="${SEED:-20260624}"
LEVEL_TAG="${LEVELS//,/x}"
VERSION="${VERSION:-direct-${ARCH}-d${FSQ_D}-l${LEVEL_TAG}-${NORMALIZER}-${CODEC_INIT}-${COMBINER_MODE}-e${EPOCHS}}"
SAVE_DIR="${SAVE_DIR:-MY-V2/jscc-f/explore/checkpoints-direct/${VERSION}}"

export CUDA_VISIBLE_DEVICES="${GPU}"

args=(
  --arch "${ARCH}"
  --version "${VERSION}"
  --save-dir "${SAVE_DIR}"
  --fsq-d "${FSQ_D}"
  --fsq-levels "${LEVELS}"
  --condition-mode none
  --codec-init "${CODEC_INIT}"
  --combiner-mode "${COMBINER_MODE}"
  --blend-init "${BLEND_INIT}"
  --lambda-u2-img 0
  --lambda-usage 0
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --test-batch 1
  --num-workers "${NUM_WORKERS}"
  --val-num-workers "${VAL_NUM_WORKERS}"
  --bn-calibration-batches "${BN_CALIBRATION_BATCHES}"
  --grad-clip-norm 5
  --val-every 5
  --latest-every 5
  --seed "${SEED}"
  --selection-min-delta-x1 0
  --selection-min-drop-zero 0.1
  --selection-min-drop-shuffle 0.1
)

if [[ "${NORMALIZER}" == "none" ]]; then
  args+=(--no-pre-norm)
else
  args+=(--fsq-normalizer "${NORMALIZER}")
fi
if [[ "${FRESH_COMBINER}" == "1" ]]; then
  args+=(--fresh-combiner)
fi
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "${RESUME}")
fi
if [[ "${RESET_BEST_ON_RESUME}" == "1" ]]; then
  args+=(--reset-best-on-resume)
fi

exec conda run --no-capture-output -n cddm_ddnm python -u \
  MY-V2/jscc-f/explore/train_layer2_fsq_direct.py "${args[@]}"
