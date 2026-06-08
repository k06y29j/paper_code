#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-3}"
COARSE_SAVE="${COARSE_SAVE:-MY/checkpoints-cvq-fsq16-spatialcar-swin-coarse-a-v1}"
RESIDUAL_SAVE="${RESIDUAL_SAVE:-MY/checkpoints-cvq-fsq16-spatialcar-swin-coarse-a-residual-v1}"
INIT_CKPT="${INIT_CKPT:-MY/checkpoints-cvq-fsq16-tailfocus-e2e-vq1e3/cvq_c36_snr9_stage3_best.pth}"

STAGE5_EPOCHS="${STAGE5_EPOCHS:-120}"
RESIDUAL_EPOCHS="${RESIDUAL_EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_EVERY="${VAL_EVERY:-5}"
LATEST_EVERY="${LATEST_EVERY:-10}"
CAR_DIM="${CAR_DIM:-192}"
CAR_HEADS="${CAR_HEADS:-6}"
CAR_LAYERS="${CAR_LAYERS:-4}"
CAR_DROPOUT="${CAR_DROPOUT:-0.05}"
CAR_LR="${CAR_LR:-1e-4}"
LAMBDA_STAGE5_REC="${LAMBDA_STAGE5_REC:-100}"
LAMBDA_STAGE5_VALUE="${LAMBDA_STAGE5_VALUE:-0.5}"
STAGE5_SOFT_TAU="${STAGE5_SOFT_TAU:-1.0}"
STAGE5_FSQ_GENERATE="${STAGE5_FSQ_GENERATE:-mean}"
STAGE5_FSQ_COARSE_LEVELS="${STAGE5_FSQ_COARSE_LEVELS:-4}"
STAGE5_RESIDUAL_THRESHOLD="${STAGE5_RESIDUAL_THRESHOLD:-0.65}"
STAGE5_COARSE_GATE_TO_ZERO="${STAGE5_COARSE_GATE_TO_ZERO:-1}"

run_stage5() {
  local save_dir="$1"
  local strategy="$2"
  local epochs="$3"
  local init_car="${4:-}"
  mkdir -p "$save_dir"
  CUDA_VISIBLE_DEVICES="$GPU" \
  SAVE_DIR="$save_dir" \
  INIT_CKPT="$INIT_CKPT" \
  STAGE5_INIT_CAR_CKPT="$init_car" \
  STAGE5_EPOCHS="$epochs" \
  BATCH_SIZE="$BATCH_SIZE" \
  VAL_EVERY="$VAL_EVERY" \
  LATEST_EVERY="$LATEST_EVERY" \
  CAR_DIM="$CAR_DIM" \
  CAR_HEADS="$CAR_HEADS" \
  CAR_LAYERS="$CAR_LAYERS" \
  CAR_DROPOUT="$CAR_DROPOUT" \
  CAR_LR="$CAR_LR" \
  LAMBDA_STAGE5_REC="$LAMBDA_STAGE5_REC" \
  LAMBDA_STAGE5_VALUE="$LAMBDA_STAGE5_VALUE" \
  STAGE5_SOFT_TAU="$STAGE5_SOFT_TAU" \
  STAGE5_FSQ_GENERATE="$STAGE5_FSQ_GENERATE" \
  STAGE5_FSQ_STRATEGY="$strategy" \
  STAGE5_FSQ_COARSE_LEVELS="$STAGE5_FSQ_COARSE_LEVELS" \
  STAGE5_RESIDUAL_THRESHOLD="$STAGE5_RESIDUAL_THRESHOLD" \
  STAGE5_COARSE_GATE_TO_ZERO="$STAGE5_COARSE_GATE_TO_ZERO" \
  STAGE5_MIN_TAIL_GAIN_CAR=0.0 \
  MY/run_cvq_c36_snr9_stage5_fsq_spatial_car_swin.sh
}

mkdir -p "$COARSE_SAVE"
{
  echo "GPU=$GPU"
  echo "COARSE_SAVE=$COARSE_SAVE"
  echo "RESIDUAL_SAVE=$RESIDUAL_SAVE"
  echo "INIT_CKPT=$INIT_CKPT"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
} > "$COARSE_SAVE/coarse_then_residual.meta"

echo "=== Stage5 coarse A on GPU $GPU ==="
run_stage5 "$COARSE_SAVE" coarse_a "$STAGE5_EPOCHS"

coarse_best="$COARSE_SAVE/cvq_c36_snr9_stage5_best.pth"
if [[ -f "$coarse_best" ]]; then
  echo "=== coarse A reached positive tail gain; continue with residual from $coarse_best ==="
  run_stage5 "$RESIDUAL_SAVE" coarse_a_residual "$RESIDUAL_EPOCHS" "$coarse_best"
else
  echo "=== coarse A did not save best checkpoint with positive tail gain; residual stage skipped ==="
fi
