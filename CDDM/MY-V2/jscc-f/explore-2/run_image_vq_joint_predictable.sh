#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
SCRIPT="$ROOT/MY-V2/jscc-f/explore-2/train_layer2_vq_nested.py"
GPU="${GPU:-1}"
ORACLE_CHECKPOINT="${ORACLE_CHECKPOINT:?set ORACLE_CHECKPOINT to an image-vq oracle}"
VERSION="${VERSION:-cnn-image-vq-d1024-joint-predictable-v1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LOG_DIR="$ROOT/MY-V2/jscc-f/explore-2/logs-vq"
SAVE_DIR="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-vq"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$SCRIPT" \
  --arch cnn \
  --layer2-arch match \
  --vq-family image-vq \
  --channel-codebook-mode global \
  --latent-c 256 \
  --embedding-dim 1024 \
  --rates 256,1024,4096 \
  --rate-weights 1,1,1 \
  --codebook-init random \
  --combiner residual \
  --resume "$ORACLE_CHECKPOINT" \
  --reset-optimizer-on-resume \
  --reset-predictor-on-resume \
  --receiver-warmup-epochs 0 \
  --lambda-vq 0.01 \
  --vq-warmup-epochs 2 \
  --vq-ramp-epochs 12 \
  --lambda-monotonic 0.2 \
  --lambda-predict-q 0.1 \
  --lambda-predictability 0.01 \
  --lambda-predictable-z 0.01 \
  --lambda-predict-final 10 \
  --min-receiver-delta 0.5 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch 2 \
  --num-workers 8 \
  --val-num-workers 4 \
  --val-every 5 \
  --latest-every 5 \
  --seed 20260713 \
  --version "$VERSION" \
  --save-dir "$SAVE_DIR" \
  "$@" \
  2>&1 | tee -a "$LOG_DIR/${VERSION}.log"
