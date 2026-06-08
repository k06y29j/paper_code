#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAVE_DIR="${SAVE_DIR:-MY/checkpoints-cvq-e2e-kmeans-vq5e3-k512}"
KMEANS_ITERS="${KMEANS_ITERS:-25}"
LAMBDA_USAGE="${LAMBDA_USAGE:-0.0005}"
LAMBDA_VQ_START="${LAMBDA_VQ_START:-0.0005}"
LAMBDA_VQ_END="${LAMBDA_VQ_END:-0.005}"

mkdir -p "$SAVE_DIR"

SAVE_DIR="$SAVE_DIR" \
KMEANS_ITERS="$KMEANS_ITERS" \
LAMBDA_USAGE="$LAMBDA_USAGE" \
LAMBDA_VQ_START="$LAMBDA_VQ_START" \
LAMBDA_VQ_END="$LAMBDA_VQ_END" \
MY/run_cvq_c36_snr9_stage3_e2e.sh \
  2>&1 | tee "$SAVE_DIR/run_stage3_e2e_kmeans.console.log"
