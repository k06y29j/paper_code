#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
RUNNER="$ROOT/MY-V2/jscc-f/explore-2/run_vq_nested.sh"
GPU="${GPU:-0}"
SEED="${SEED:-20260624}"

# One fair 2x2 matrix per invocation.  Each job contains the exact shared
# K=256/1024/4096 prefixes; run seeds 20260624/25/26 independently.
for arch in cnn swin; do
  for family in image-vq channel-vq; do
    ARCH="$arch" \
    VQ_FAMILY="$family" \
    GPU="$GPU" \
    LATENT_C=256 \
    RATES=256,1024,4096 \
    VERSION="${arch}-${family}-c256-nested-seed${SEED}" \
      "$RUNNER" --seed "$SEED" "$@"
  done
done

