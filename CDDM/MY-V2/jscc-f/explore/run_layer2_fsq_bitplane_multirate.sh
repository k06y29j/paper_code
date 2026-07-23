#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
RATE_BITS="${RATE_BITS:-7,10,13}"
NORMALIZER="${NORMALIZER:-batch}"
ADAPTER_INIT="${ADAPTER_INIT:-pca}"
PCA_INIT_BATCHES="${PCA_INIT_BATCHES:-3}"
ADAPTER_HIDDEN="${ADAPTER_HIDDEN:-320}"
ADAPTER_COMBINER="${ADAPTER_COMBINER:-original}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
BN_CALIBRATION_BATCHES="${BN_CALIBRATION_BATCHES:-64}"
RECON_WEIGHTS="${RECON_WEIGHTS:-1}"
LAMBDA_MONOTONIC="${LAMBDA_MONOTONIC:-1}"
MONOTONIC_MARGINS="${MONOTONIC_MARGINS:-1e-5}"
SEED="${SEED:-20260624}"
BIT_TAG="${RATE_BITS//,/x}"
VERSION="${VERSION:-direct-swin-bitplane-b${BIT_TAG}-${NORMALIZER}-${ADAPTER_INIT}-h${ADAPTER_HIDDEN}-${ADAPTER_COMBINER}-e${EPOCHS}}"
SAVE_DIR="${SAVE_DIR:-MY-V2/jscc-f/explore/checkpoints-bitplane/${VERSION}}"

export CUDA_VISIBLE_DEVICES="${GPU}"

args=(
  --arch swin
  --version "${VERSION}"
  --save-dir "${SAVE_DIR}"
  --rate-bits "${RATE_BITS}"
  --condition-mode none
  --combiner-mode original
  --fsq-normalizer "${NORMALIZER}"
  --lambda-u2-img 0
  --lambda-usage 0
  --adapter-init "${ADAPTER_INIT}"
  --pca-init-batches "${PCA_INIT_BATCHES}"
  --adapter-hidden "${ADAPTER_HIDDEN}"
  --adapter-combiner "${ADAPTER_COMBINER}"
  --recon-weights "${RECON_WEIGHTS}"
  --lambda-monotonic "${LAMBDA_MONOTONIC}"
  --monotonic-margins "${MONOTONIC_MARGINS}"
  --selection-min-rate-gain-db 0
  --selection-min-per-image-strict-ratio 0
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
  MY-V2/jscc-f/explore/train_layer2_fsq_bitplane_multirate.py "${args[@]}"
