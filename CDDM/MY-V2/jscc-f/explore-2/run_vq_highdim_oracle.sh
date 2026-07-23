#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
RUNNER="$ROOT/MY-V2/jscc-f/explore-2/run_vq_nested.sh"
ARCH="${ARCH:-cnn}"
VQ_FAMILY="${VQ_FAMILY:-image-vq}"
CHANNEL_CODEBOOK_MODE="${CHANNEL_CODEBOOK_MODE:-global}"
GPU="${GPU:-0}"
LATENT_C="${LATENT_C:-256}"
EMBEDDING_DIM="${EMBEDDING_DIM:-512}"
RATES="${RATES:-256,1024,4096}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-20260713}"
FREEZE_ENCODER="${FREEZE_ENCODER:-1}"
VERSION="${VERSION:-${ARCH}-${VQ_FAMILY}-${CHANNEL_CODEBOOK_MODE}-c${LATENT_C}-d${EMBEDDING_DIM}-oracle-seed${SEED}}"

extra=()
if [[ "$FREEZE_ENCODER" == "1" ]]; then
  extra+=(--freeze-encoder)
fi
if [[ "$VQ_FAMILY" == "channel-vq" && "$CHANNEL_CODEBOOK_MODE" == "grouped" ]]; then
  extra+=(--codebook-init channel-balanced)
else
  extra+=(--codebook-init random)
fi

ARCH="$ARCH" \
VQ_FAMILY="$VQ_FAMILY" \
CHANNEL_CODEBOOK_MODE="$CHANNEL_CODEBOOK_MODE" \
GPU="$GPU" \
LATENT_C="$LATENT_C" \
EMBEDDING_DIM="$EMBEDDING_DIM" \
RATES="$RATES" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
VERSION="$VERSION" \
  "$RUNNER" \
  --seed "$SEED" \
  --oracle-only \
  --lambda-vq 0.01 \
  --vq-warmup-epochs 2 \
  --vq-ramp-epochs 12 \
  --lambda-monotonic 0.2 \
  --val-every 5 \
  --latest-every 5 \
  "${extra[@]}" \
  "$@"
