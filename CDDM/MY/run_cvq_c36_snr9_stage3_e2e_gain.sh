#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAVE_DIR="${SAVE_DIR:-MY/checkpoints-cvq-e2e-kmeans-gain-vq5e3-k512}"

mkdir -p "$SAVE_DIR"

SAVE_DIR="$SAVE_DIR" \
KMEANS_ITERS="${KMEANS_ITERS:-25}" \
LAMBDA_USAGE="${LAMBDA_USAGE:-0.0005}" \
LAMBDA_VQ_START="${LAMBDA_VQ_START:-0.0005}" \
LAMBDA_VQ_END="${LAMBDA_VQ_END:-0.005}" \
LAMBDA_GAIN_STAGE3="${LAMBDA_GAIN_STAGE3:-5.0}" \
STAGE3_GAIN_MARGIN_DB="${STAGE3_GAIN_MARGIN_DB:-0.2}" \
FREEZE_ENCODER_FIRST="${FREEZE_ENCODER_FIRST:-5}" \
DEAD_CODE_RESTART_EVERY="${DEAD_CODE_RESTART_EVERY:-0}" \
MY/run_cvq_c36_snr9_stage3_e2e.sh \
  2>&1 | tee "$SAVE_DIR/run_stage3_e2e_gain.console.log"
