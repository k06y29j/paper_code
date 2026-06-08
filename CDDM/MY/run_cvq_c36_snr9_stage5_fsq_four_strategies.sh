#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_LAUNCHER="MY/run_cvq_c36_snr9_stage5_fsq_spatial_car_swin.sh"

launch_one() {
  local gpu="$1"
  local strategy="$2"
  local save_dir="$3"
  shift 3

  mkdir -p "$save_dir"
  {
    echo "gpu=$gpu"
    echo "strategy=$strategy"
    echo "save_dir=$save_dir"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "extra_env=$*"
  } > "$save_dir/detached.meta"

  setsid -f bash -c "
    export CUDA_VISIBLE_DEVICES='$gpu'
    export SAVE_DIR='$save_dir'
    export STAGE5_FSQ_STRATEGY='$strategy'
    export STAGE5_EPOCHS='120'
    export BATCH_SIZE='8'
    export TEST_BATCH='1'
    export NUM_WORKERS='16'
    export VAL_NUM_WORKERS='8'
    export VAL_EVERY='5'
    export LATEST_EVERY='10'
    export CAR_DIM='192'
    export CAR_HEADS='6'
    export CAR_LAYERS='4'
    export CAR_DROPOUT='0.05'
    export CAR_WINDOW_SIZE='4'
    export CAR_LR='1e-4'
    export LAMBDA_STAGE5_REC='100'
    export LAMBDA_STAGE5_VALUE='0.5'
    export STAGE5_SOFT_TAU='1.0'
    export STAGE5_FSQ_GENERATE='mean'
    export STAGE5_MIN_TAIL_GAIN_CAR='0.0'
    $*
    bash '$BASE_LAUNCHER'
  " > "$save_dir/detached.log" 2>&1
}

launch_one 0 two_stage "MY/checkpoints-cvq-fsq16-spatialcar-swin-two-stage-v1" \
  "export STAGE5_TWO_STAGE_A_EPOCHS='60'"

launch_one 1 gated "MY/checkpoints-cvq-fsq16-spatialcar-swin-gated-v1" \
  "export STAGE5_GATE_THRESHOLD='0.35'; export STAGE5_GATE_B_THRESHOLD='0.50'"

launch_one 2 ordinal "MY/checkpoints-cvq-fsq16-spatialcar-swin-ordinal-v1" \
  "export LAMBDA_STAGE5_ORDINAL='2.0'"

launch_one 3 maskgit "MY/checkpoints-cvq-fsq16-spatialcar-swin-maskgit-v1" \
  "export STAGE5_MASKGIT_ITERS='4'; export STAGE5_MASKGIT_MASK_PROB='0.6'"

echo "launched four FSQ Stage5 strategies"
