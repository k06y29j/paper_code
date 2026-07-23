#!/usr/bin/env bash
set -euo pipefail

ARCH="${ARCH:-swin}"
GPU="${GPU:-0}"
FSQ_D="${FSQ_D:-3}"
NORMALIZER="${NORMALIZER:-batch}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
BN_CALIBRATION_BATCHES="${BN_CALIBRATION_BATCHES:-64}"
SEED="${SEED:-20260624}"
VERSION="${VERSION:-direct-${ARCH}-continuous-d${FSQ_D}-${NORMALIZER}-original-e${EPOCHS}}"
SAVE_DIR="${SAVE_DIR:-MY-V2/jscc-f/explore/checkpoints-continuous/${VERSION}}"

export CUDA_VISIBLE_DEVICES="${GPU}"

args=(
  --arch "${ARCH}"
  --version "${VERSION}"
  --save-dir "${SAVE_DIR}"
  --fsq-d "${FSQ_D}"
  --fsq-levels 17,17,17
  --condition-mode none
  --codec-init compatible
  --combiner-mode original
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
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "${RESUME}")
fi
if [[ "${RESET_BEST_ON_RESUME:-0}" == "1" ]]; then
  args+=(--reset-best-on-resume)
fi

exec conda run --no-capture-output -n cddm_ddnm python -u \
  MY-V2/jscc-f/explore/train_layer2_continuous_direct.py "${args[@]}"
