#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
FSQ_D="${FSQ_D:-3}"
LEVELS="${LEVELS:-17,17,17}"
NORMALIZER="${NORMALIZER:-batch}"
ADAPTER_INIT="${ADAPTER_INIT:-pca}"
PCA_INIT_BATCHES="${PCA_INIT_BATCHES:-3}"
ADAPTER_HIDDEN="${ADAPTER_HIDDEN:-320}"
ADAPTER_COMBINER="${ADAPTER_COMBINER:-original}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
BN_CALIBRATION_BATCHES="${BN_CALIBRATION_BATCHES:-64}"
SEED="${SEED:-20260624}"
LEVEL_TAG="${LEVELS//,/x}"
VERSION="${VERSION:-direct-swin-adapter-d${FSQ_D}-l${LEVEL_TAG}-${NORMALIZER}-${ADAPTER_INIT}-h${ADAPTER_HIDDEN}-${ADAPTER_COMBINER}-e${EPOCHS}}"
SAVE_DIR="${SAVE_DIR:-MY-V2/jscc-f/explore/checkpoints-adapter/${VERSION}}"

export CUDA_VISIBLE_DEVICES="${GPU}"

args=(
  --arch swin
  --version "${VERSION}"
  --save-dir "${SAVE_DIR}"
  --fsq-d "${FSQ_D}"
  --fsq-levels "${LEVELS}"
  --condition-mode none
  --combiner-mode original
  --fsq-normalizer "${NORMALIZER}"
  --lambda-u2-img 0
  --lambda-usage 0
  --adapter-init "${ADAPTER_INIT}"
  --pca-init-batches "${PCA_INIT_BATCHES}"
  --adapter-hidden "${ADAPTER_HIDDEN}"
  --adapter-combiner "${ADAPTER_COMBINER}"
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

if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "${RESUME}")
fi
if [[ "${RESET_BEST_ON_RESUME:-0}" == "1" ]]; then
  args+=(--reset-best-on-resume)
fi

exec conda run --no-capture-output -n cddm_ddnm python -u \
  MY-V2/jscc-f/explore/train_layer2_fsq_adapter.py "${args[@]}"
