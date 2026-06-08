#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-3}"
STAGE5_EPOCHS="${STAGE5_EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_EVERY="${VAL_EVERY:-5}"
LATEST_EVERY="${LATEST_EVERY:-10}"
STAGE5_FSQ_STRATEGY="${STAGE5_FSQ_STRATEGY:-gated}"
STAGE5_GATE_THRESHOLD="${STAGE5_GATE_THRESHOLD:-0.60}"
STAGE5_GATE_B_THRESHOLD="${STAGE5_GATE_B_THRESHOLD:-0.75}"
LAMBDA_STAGE5_REC="${LAMBDA_STAGE5_REC:-100}"
LAMBDA_STAGE5_VALUE="${LAMBDA_STAGE5_VALUE:-0.5}"

stage3_running() {
  local stage3_dir="$1"
  pgrep -af "MY/train_cvq/main.py.*--stage 3.*--save-dir ${stage3_dir}" >/dev/null 2>&1
}

wait_for_stage3_best() {
  local stage3_dir="$1"
  local ckpt="$stage3_dir/cvq_c36_snr9_stage3_best.pth"

  while true; do
    if [[ -f "$ckpt" ]] && ! stage3_running "$stage3_dir"; then
      break
    fi
    sleep 300
  done
}

run_car() {
  local stage3_dir="$1"
  local stage5_dir="$2"
  local levels_a="$3"
  local levels_b="$4"
  local ckpt="$stage3_dir/cvq_c36_snr9_stage3_best.pth"

  wait_for_stage3_best "$stage3_dir"

  mkdir -p "$stage5_dir"
  {
    echo "CUDA_VISIBLE_DEVICES=$GPU"
    echo "STAGE3_DIR=$stage3_dir"
    echo "INIT_CKPT=$ckpt"
    echo "SAVE_DIR=$stage5_dir"
    echo "FSQ_LEVELS_A=$levels_a"
    echo "FSQ_LEVELS_B=$levels_b"
    echo "STAGE5_FSQ_STRATEGY=$STAGE5_FSQ_STRATEGY"
    echo "STAGE5_GATE_THRESHOLD=$STAGE5_GATE_THRESHOLD"
    echo "STAGE5_GATE_B_THRESHOLD=$STAGE5_GATE_B_THRESHOLD"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  } > "$stage5_dir/level_grid.meta"

  CUDA_VISIBLE_DEVICES="$GPU" \
  SAVE_DIR="$stage5_dir" \
  INIT_CKPT="$ckpt" \
  K_A="$levels_a" \
  K_B="$levels_b" \
  FSQ_LEVELS_A="$levels_a" \
  FSQ_LEVELS_B="$levels_b" \
  STAGE5_EPOCHS="$STAGE5_EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  VAL_EVERY="$VAL_EVERY" \
  LATEST_EVERY="$LATEST_EVERY" \
  STAGE5_FSQ_STRATEGY="$STAGE5_FSQ_STRATEGY" \
  STAGE5_GATE_THRESHOLD="$STAGE5_GATE_THRESHOLD" \
  STAGE5_GATE_B_THRESHOLD="$STAGE5_GATE_B_THRESHOLD" \
  LAMBDA_STAGE5_REC="$LAMBDA_STAGE5_REC" \
  LAMBDA_STAGE5_VALUE="$LAMBDA_STAGE5_VALUE" \
  MY/run_cvq_c36_snr9_stage5_fsq_spatial_car_swin.sh
}

mkdir -p MY/stage5-grid-logs

run_car \
  MY/checkpoints-cvq-fsq16x8-tailfocus-e2e-vq1e3 \
  MY/checkpoints-cvq-fsq16x8-spatialcar-swin-gated-v1 \
  16 8

run_car \
  MY/checkpoints-cvq-fsq8x8-tailfocus-e2e-vq1e3 \
  MY/checkpoints-cvq-fsq8x8-spatialcar-swin-gated-v1 \
  8 8

run_car \
  MY/checkpoints-cvq-fsq8x4-tailfocus-e2e-vq1e3 \
  MY/checkpoints-cvq-fsq8x4-spatialcar-swin-gated-v1 \
  8 4
