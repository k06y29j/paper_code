#!/usr/bin/env bash
set -euo pipefail

# One shared-model nested-rate experiment.  This is a monotonicity/control
# route, not the per-K upper-bound run (use run_layer2_fsq_direct_nested.sh with
# COMBINER_MODE=original for independent per-K envelopes).

ARCH="${ARCH:-swin}"
GPU="${GPU:-0}"
FSQ_D="${FSQ_D:-3}"
NESTED_LEVELS="${NESTED_LEVELS:-5,9,17}"
NORMALIZER="${NORMALIZER:-batch}"
CODEC_INIT="${CODEC_INIT:-compatible}"
COMBINER_MODE="${COMBINER_MODE:-original}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
BN_CALIBRATION_BATCHES="${BN_CALIBRATION_BATCHES:-64}"
RECON_WEIGHTS="${RECON_WEIGHTS:-1}"
LAMBDA_MONOTONIC="${LAMBDA_MONOTONIC:-1}"
MONOTONIC_MARGINS="${MONOTONIC_MARGINS:-1e-5}"
MIN_RATE_GAIN_DB="${MIN_RATE_GAIN_DB:-0}"
MIN_PER_IMAGE_STRICT_RATIO="${MIN_PER_IMAGE_STRICT_RATIO:-0}"
SEED="${SEED:-20260624}"
LEVEL_TAG="${NESTED_LEVELS//,/x}"
VERSION="${VERSION:-direct-${ARCH}-multirate-d${FSQ_D}-l${LEVEL_TAG}-${NORMALIZER}-${COMBINER_MODE}-e${EPOCHS}}"
SAVE_DIR="${SAVE_DIR:-MY-V2/jscc-f/explore/checkpoints-multirate/${VERSION}}"

export CUDA_VISIBLE_DEVICES="${GPU}"

args=(
  --arch "${ARCH}"
  --version "${VERSION}"
  --save-dir "${SAVE_DIR}"
  --fsq-d "${FSQ_D}"
  --nested-levels "${NESTED_LEVELS}"
  --condition-mode none
  --codec-init "${CODEC_INIT}"
  --combiner-mode "${COMBINER_MODE}"
  --lambda-u2-img 0
  --lambda-usage 0
  --recon-weights "${RECON_WEIGHTS}"
  --lambda-monotonic "${LAMBDA_MONOTONIC}"
  --monotonic-margins "${MONOTONIC_MARGINS}"
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
  --selection-min-rate-gain-db "${MIN_RATE_GAIN_DB}"
  --selection-min-per-image-strict-ratio "${MIN_PER_IMAGE_STRICT_RATIO}"
  --selection-min-delta-x1 0
  --selection-min-drop-zero 0.1
  --selection-min-drop-shuffle 0.1
)

if [[ "${NORMALIZER}" == "none" ]]; then
  args+=(--no-pre-norm)
else
  args+=(--fsq-normalizer "${NORMALIZER}")
fi
if [[ -n "${INIT_DIRECT_CKPT:-}" ]]; then
  args+=(--init-direct-ckpt "${INIT_DIRECT_CKPT}")
fi
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "${RESUME}")
fi
if [[ "${RESET_BEST_ON_RESUME:-0}" == "1" ]]; then
  args+=(--reset-best-on-resume)
fi

exec conda run --no-capture-output -n cddm_ddnm python -u \
  MY-V2/jscc-f/explore/train_layer2_fsq_multirate.py "${args[@]}"
