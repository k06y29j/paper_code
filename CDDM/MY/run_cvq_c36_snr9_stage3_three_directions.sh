#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE3_EPOCHS="${STAGE3_EPOCHS:-160}"
COMMON_STAGE3_SCORE="${STAGE3_SCORE:-psnr_vq}"

launch_one() {
  local gpu="$1"
  local mode="$2"
  local save_dir="$3"
  shift 3
  mkdir -p "$save_dir"
  {
    echo "gpu=$gpu"
    echo "mode=$mode"
    echo "save_dir=$save_dir"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "extra_env=$*"
  } > "$save_dir/launch.meta"
  setsid -f bash -lc "
    cd '$ROOT'
    export CUDA_VISIBLE_DEVICES='$gpu'
    export SAVE_DIR='$save_dir'
    export CVQ_MODE='$mode'
    export STAGE3_EPOCHS='$STAGE3_EPOCHS'
    export STAGE3_SCORE='$COMMON_STAGE3_SCORE'
    export LAMBDA_USAGE='0.0'
    export DEAD_CODE_RESTART_EVERY='0'
    $*
    MY/run_cvq_c36_snr9_stage3_e2e.sh > '$save_dir/launch.log' 2>&1
  "
  echo "launched gpu=$gpu mode=$mode save_dir=$save_dir"
}

launch_one 0 rvq MY/checkpoints-cvq-rvq2-vq5e3-k512 \
  "export RVQ_STAGES='2'; export K_A='512'; export K_B='256'; export LAMBDA_CONT_STAGE3='0.01'; export STAGE3_MIN_TAIL_GAIN_VQ='0.0'; export STAGE3_MIN_PERPLEXITY_A='10'; export STAGE3_MIN_PERPLEXITY_B='5'"

launch_one 1 patch MY/checkpoints-cvq-patch4-vq5e3-k512 \
  "export PATCH_SIZE='4'; export K_A='512'; export K_B='256'; export LAMBDA_CONT_STAGE3='0.01'; export STAGE3_MIN_TAIL_GAIN_VQ='0.0'; export STAGE3_MIN_PERPLEXITY_A='20'; export STAGE3_MIN_PERPLEXITY_B='10'"

launch_one 2 fsq MY/checkpoints-cvq-fsq16-vq1e3 \
  "export K_A='16'; export K_B='16'; export FSQ_LEVELS_A='16'; export FSQ_LEVELS_B='16'; export LAMBDA_VQ_START='0.0001'; export LAMBDA_VQ_END='0.001'; export LAMBDA_CONT_STAGE3='0.0'; export LAMBDA_TAIL_DISTILL='0.0'; export STAGE3_MIN_TAIL_GAIN_VQ='0.0'; export STAGE3_MIN_USED_A='8'; export STAGE3_MIN_USED_B='8'; export STAGE3_MIN_PERPLEXITY_A='4'; export STAGE3_MIN_PERPLEXITY_B='4'"

echo "all three Stage3 exploration jobs launched"
