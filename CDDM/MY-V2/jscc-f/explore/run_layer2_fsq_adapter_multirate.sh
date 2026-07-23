#!/usr/bin/env bash
set -euo pipefail

# One shared width-320 PCA-adapter model at K=125/729/4913.  Every rate sees
# the same d=3 normalized latent and uses the same synthesis/D2/residual
# combiner, so the capacity comparison is not confounded by separate decoders.

GPU="${GPU:-0}"
FSQ_D="${FSQ_D:-3}"
NESTED_LEVELS="${NESTED_LEVELS:-5,9,17}"
NORMALIZER="${NORMALIZER:-batch}"
ADAPTER_INIT="${ADAPTER_INIT:-pca}"
PCA_INIT_BATCHES="${PCA_INIT_BATCHES:-3}"
ADAPTER_HIDDEN="${ADAPTER_HIDDEN:-320}"
ADAPTER_COMBINER="${ADAPTER_COMBINER:-residual}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
BN_CALIBRATION_BATCHES="${BN_CALIBRATION_BATCHES:-64}"
RECON_WEIGHTS="${RECON_WEIGHTS:-1}"
LAMBDA_MONOTONIC="${LAMBDA_MONOTONIC:-1}"
MONOTONIC_MARGINS="${MONOTONIC_MARGINS:-1e-5}"
MIN_RATE_GAIN_DB="${MIN_RATE_GAIN_DB:-0}"
MIN_PER_IMAGE_STRICT_RATIO="${MIN_PER_IMAGE_STRICT_RATIO:-0}"
MIN_DELTA_X1="${MIN_DELTA_X1:-0}"
MIN_DROP_ZERO="${MIN_DROP_ZERO:-0.1}"
MIN_DROP_SHUFFLE="${MIN_DROP_SHUFFLE:-0.1}"
SEED="${SEED:-20260624}"
LEVEL_TAG="${NESTED_LEVELS//,/x}"
VERSION="${VERSION:-direct-swin-adapter-multirate-d${FSQ_D}-l${LEVEL_TAG}-${NORMALIZER}-${ADAPTER_INIT}-h${ADAPTER_HIDDEN}-${ADAPTER_COMBINER}-e${EPOCHS}}"
SAVE_DIR="${SAVE_DIR:-MY-V2/jscc-f/explore/checkpoints-adapter-multirate/${VERSION}}"

export CUDA_VISIBLE_DEVICES="${GPU}"

args=(
  --arch swin
  --preset custom
  --version "${VERSION}"
  --save-dir "${SAVE_DIR}"
  --fsq-d "${FSQ_D}"
  --nested-levels "${NESTED_LEVELS}"
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
  --selection-min-rate-gain-db "${MIN_RATE_GAIN_DB}"
  --selection-min-per-image-strict-ratio "${MIN_PER_IMAGE_STRICT_RATIO}"
  --selection-min-delta-x1 "${MIN_DELTA_X1}"
  --selection-min-drop-zero "${MIN_DROP_ZERO}"
  --selection-min-drop-shuffle "${MIN_DROP_SHUFFLE}"
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
)

if [[ -n "${LAYER2_CKPT:-}" ]]; then
  args+=(--layer2-ckpt "${LAYER2_CKPT}")
fi
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "${RESUME}")
fi
if [[ "${RESET_BEST_ON_RESUME:-0}" == "1" ]]; then
  args+=(--reset-best-on-resume)
fi

exec conda run --no-capture-output -n cddm_ddnm python -u \
  MY-V2/jscc-f/explore/train_layer2_fsq_adapter_multirate.py "${args[@]}"
