#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-12}"
EPOCHS="${EPOCHS:-600}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-20}"

mkdir -p checkpoints-ar

launch_one() {
  local snr_db="$1"
  local gpu="$2"
  local tag="snr${snr_db}"
  local root="checkpoints-ar/twofreq_receiver_norm_swin_${tag}_v1"
  local launch_log="${root}/01_receiver_norm_swin_stage01/launcher.log"

  mkdir -p "$(dirname "${launch_log}")"
  setsid -f bash -c "
    export GPU='${gpu}'
    export SNR_DB='${snr_db}'
    export TAG='${tag}'
    export BATCH_SIZE='${BATCH_SIZE}'
    export EPOCHS='${EPOCHS}'
    export EVAL_EVERY_EPOCHS='${EVAL_EVERY_EPOCHS}'
    exec bash train/run_hierarchical_receiver_norm_swin_pretrain.sh
  " > "${launch_log}" 2>&1 < /dev/null
  echo "launched snr=${snr_db} gpu=${gpu} log=${launch_log}"
}

launch_one 0 0
launch_one 6 2
launch_one 12 3
