#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

declare -a GPUS=(0 1 2 3)
declare -a PREDS=(resunet multiscale attn freq)
declare -a HIDDENS=(128 224 160 224)
declare -a DEPTHS=(2 8 6 12)

for i in "${!PREDS[@]}"; do
  gpu="${GPUS[$i]}"
  pred="${PREDS[$i]}"
  run_dir="checkpoints-codex/fixed_direct_${pred}_awgn12"
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}"
  (
    export HIDDEN="${HIDDENS[$i]}"
    export DEPTH="${DEPTHS[$i]}"
    export BATCH_SIZE="${BATCH_SIZE:-32}"
    export EPOCHS="${EPOCHS:-400}"
    export LAMBDA_RES="${LAMBDA_RES:-0.1}"
    export IMG_MSE_WEIGHT="${IMG_MSE_WEIGHT:-0.8}"
    export IMG_CHARB_WEIGHT="${IMG_CHARB_WEIGHT:-0.2}"
    setsid bash -c "echo \$\$ > '${run_dir}/train.pid'; exec train/run_codex_fixed_direct_predictor_awgn12.sh '${gpu}' '${pred}' '${run_dir}' > '${run_dir}/train.stdout' 2>&1" </dev/null >/dev/null 2>&1 &
  )
  echo "launched predictor=${pred} gpu=${gpu} run_dir=${run_dir}"
done
