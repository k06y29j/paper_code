#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
SCRIPT="$ROOT/MY-V2/jscc-f/explore-2/train_layer2_vq_nested.py"
ARCH="${ARCH:-cnn}"
VQ_FAMILY="${VQ_FAMILY:-image-vq}"
CHANNEL_CODEBOOK_MODE="${CHANNEL_CODEBOOK_MODE:-global}"
GPU="${GPU:-0}"
LATENT_C="${LATENT_C:-256}"
EMBEDDING_DIM="${EMBEDDING_DIM:-0}"
RATES="${RATES:-256,1024,4096}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VERSION="${VERSION:-${ARCH}-${VQ_FAMILY}-c${LATENT_C}-d${EMBEDDING_DIM}-nested-v1}"
LOG_DIR="$ROOT/MY-V2/jscc-f/explore-2/logs-vq"
SAVE_DIR="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-vq"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$SCRIPT" \
  --arch "$ARCH" \
  --vq-family "$VQ_FAMILY" \
  --channel-codebook-mode "$CHANNEL_CODEBOOK_MODE" \
  --latent-c "$LATENT_C" \
  --embedding-dim "$EMBEDDING_DIM" \
  --rates "$RATES" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --version "$VERSION" \
  --save-dir "$SAVE_DIR" \
  "$@" \
  2>&1 | tee -a "$LOG_DIR/${VERSION}.log"
