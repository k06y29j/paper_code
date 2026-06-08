#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-12}"
EPOCHS="${EPOCHS:-600}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-20}"
SNR_LIST="${SNR_LIST:-0 6 12}"

for SNR_DB in ${SNR_LIST}; do
  export GPU SNR_DB BATCH_SIZE EPOCHS EVAL_EVERY_EPOCHS
  export TAG="snr${SNR_DB}"
  bash train/run_hierarchical_receiver_norm_swin_pretrain.sh
done
