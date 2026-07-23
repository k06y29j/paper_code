#!/usr/bin/env bash
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_cvq_car.sh <sender-best.pth> <version> [extra args...]
set -euo pipefail

SENDER=${1:?usage: $0 <sender-best.pth> <version> [extra args...]}
VERSION=${2:?usage: $0 <sender-best.pth> <version> [extra args...]}
shift 2
ROOT=/workspace/yongjia/paper_code/CDDM
EXPLORE4=${ROOT}/MY-V2/jscc-f/explore-4
mkdir -p "${EXPLORE4}/logs-car" "${EXPLORE4}/checkpoints-car" "${EXPLORE4}/results-car"

exec conda run --no-capture-output -n cddm_ddnm python "${EXPLORE4}/train_cvq_car.py" \
  --sender-checkpoint "${SENDER}" \
  --rate 4096 \
  --hidden 256 \
  --layers 6 \
  --heads 8 \
  --lambda-ce 1.0 \
  --lambda-recon 1.0 \
  --soft-temperature 1.0 \
  --epochs 80 \
  --val-every 5 \
  --batch-size 2 \
  --test-batch 2 \
  --num-workers 8 \
  --val-num-workers 4 \
  --save-dir MY-V2/jscc-f/explore-4/checkpoints-car \
  --log-json "MY-V2/jscc-f/explore-4/results-car/${VERSION}.json" \
  --version "${VERSION}" \
  "$@"
