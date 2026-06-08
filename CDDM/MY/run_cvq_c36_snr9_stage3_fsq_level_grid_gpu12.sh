#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE3_EPOCHS="${STAGE3_EPOCHS:-160}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_EVERY="${VAL_EVERY:-5}"
LATEST_EVERY="${LATEST_EVERY:-10}"

run_one() {
  local gpu="$1"
  local save_dir="$2"
  local levels_a="$3"
  local levels_b="$4"
  local min_used_a="$5"
  local min_used_b="$6"
  local min_ppl_a="$7"
  local min_ppl_b="$8"

  mkdir -p "$save_dir"
  {
    echo "CUDA_VISIBLE_DEVICES=$gpu"
    echo "SAVE_DIR=$save_dir"
    echo "FSQ_LEVELS_A=$levels_a"
    echo "FSQ_LEVELS_B=$levels_b"
    echo "STAGE3_EPOCHS=$STAGE3_EPOCHS"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  } > "$save_dir/level_grid.meta"

  CUDA_VISIBLE_DEVICES="$gpu" \
  SAVE_DIR="$save_dir" \
  CVQ_MODE=fsq \
  K_A="$levels_a" \
  K_B="$levels_b" \
  FSQ_LEVELS_A="$levels_a" \
  FSQ_LEVELS_B="$levels_b" \
  STAGE3_EPOCHS="$STAGE3_EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  VAL_EVERY="$VAL_EVERY" \
  LATEST_EVERY="$LATEST_EVERY" \
  STAGE3_MIN_USED_A="$min_used_a" \
  STAGE3_MIN_USED_B="$min_used_b" \
  STAGE3_MIN_PERPLEXITY_A="$min_ppl_a" \
  STAGE3_MIN_PERPLEXITY_B="$min_ppl_b" \
  MY/run_cvq_c36_snr9_stage3_fsq_tailfocus_e2e.sh
}

gpu1_worker() {
  run_one 1 MY/checkpoints-cvq-fsq8x8-tailfocus-e2e-vq1e3 8 8 4 4 2 2
  run_one 1 MY/checkpoints-cvq-fsq8x4-tailfocus-e2e-vq1e3 8 4 4 2 2 1.5
}

gpu2_worker() {
  run_one 2 MY/checkpoints-cvq-fsq16x8-tailfocus-e2e-vq1e3 16 8 8 4 4 2
}

mkdir -p MY/stage3-grid-logs
setsid -f bash -c "$(declare -f run_one gpu1_worker); cd '$ROOT'; STAGE3_EPOCHS='$STAGE3_EPOCHS' BATCH_SIZE='$BATCH_SIZE' VAL_EVERY='$VAL_EVERY' LATEST_EVERY='$LATEST_EVERY' gpu1_worker" \
  > MY/stage3-grid-logs/fsq_level_grid_gpu1.log 2>&1
setsid -f bash -c "$(declare -f run_one gpu2_worker); cd '$ROOT'; STAGE3_EPOCHS='$STAGE3_EPOCHS' BATCH_SIZE='$BATCH_SIZE' VAL_EVERY='$VAL_EVERY' LATEST_EVERY='$LATEST_EVERY' gpu2_worker" \
  > MY/stage3-grid-logs/fsq_level_grid_gpu2.log 2>&1

echo "launched FSQ level grid:"
echo "  GPU1: FSQ8x8 then FSQ8x4"
echo "  GPU2: FSQ16x8"
echo "logs:"
echo "  MY/stage3-grid-logs/fsq_level_grid_gpu1.log"
echo "  MY/stage3-grid-logs/fsq_level_grid_gpu2.log"
