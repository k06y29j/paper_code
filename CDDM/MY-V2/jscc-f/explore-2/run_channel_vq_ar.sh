#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
SCRIPT="$ROOT/MY-V2/jscc-f/explore-2/train_channel_vq_ar.py"
GPU="${GPU:-0}"
ORACLE_CHECKPOINT="${ORACLE_CHECKPOINT:?set ORACLE_CHECKPOINT to a channel-vq oracle (global or grouped)}"
VERSION="${VERSION:-channel-vq-ar-v1}"
LOG_DIR="$ROOT/MY-V2/jscc-f/explore-2/logs-channel-ar"
SAVE_DIR="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-channel-ar"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$SCRIPT" \
  --oracle-checkpoint "$ORACLE_CHECKPOINT" \
  --version "$VERSION" \
  --save-dir "$SAVE_DIR" \
  "$@" \
  2>&1 | tee -a "$LOG_DIR/${VERSION}.log"
