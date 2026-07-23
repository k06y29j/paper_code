#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
RUNNER="$ROOT/MY-V2/jscc-f/explore-2/run_vq_nested.sh"
ARCH="${ARCH:-cnn}"
GPU="${GPU:-0}"
SEED="${SEED:-20260624}"
LATENT_C="${LATENT_C:-256}"
RATES="${RATES:-512,1024,4096}"
VERSION="${VERSION:-${ARCH}-channel-vq-grouped-c${LATENT_C}-k512-1024-4096-seed${SEED}}"

# Formal grouped sweep: K is total round-major rows and each channel c may
# select only c+C*arange(K/C).  K=C is deliberately absent because it has one
# constant candidate/channel and therefore carries zero index bits.
ARCH="$ARCH" \
VQ_FAMILY=channel-vq \
CHANNEL_CODEBOOK_MODE=grouped \
GPU="$GPU" \
LATENT_C="$LATENT_C" \
RATES="$RATES" \
VERSION="$VERSION" \
  "$RUNNER" \
  --seed "$SEED" \
  --codebook-init channel-balanced \
  "$@"
