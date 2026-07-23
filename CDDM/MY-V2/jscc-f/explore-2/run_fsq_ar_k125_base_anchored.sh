#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
TRAIN="$ROOT/MY-V2/jscc-f/explore-2/train_fsq_receiver.py"
ORACLE="$ROOT/MY-V2/jscc-f/explore/checkpoints-direct/direct-cnn-d3-l5x5x5-group-compatible-blend-e100/jscc_f_direct-cnn_d3_l5x5x5-group-compatible-blend-e100_layer2_fsq_direct_cnn_d3_l5x5x5_group_compatible_blend_goal_best.pth"
BASE="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-125/cnn-fsq-k125-frozen-d2-combiner-hard-v1/fsq_receiver_direct_q_z1_x1_hard_cnn_d3_l5x5x5_frozen_best.pth"
SAVE_ROOT="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-fsq-generation"
LOG_ROOT="$ROOT/MY-V2/jscc-f/explore-2/logs-fsq-generation"

GPU="${GPU:-2}"
VERSION="${VERSION:-cnn-fsq-k125-ar-base-anchored-v1}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-1e-5}"
INIT="${INIT:-}"
mkdir -p "$SAVE_ROOT" "$LOG_ROOT"
[[ -f "$ORACLE" ]] || { echo "missing K125 oracle: $ORACLE" >&2; exit 2; }
[[ -f "$BASE" ]] || { echo "missing K125 hard direct base: $BASE" >&2; exit 2; }

INIT_ARGS=()
if [[ -n "$INIT" ]]; then
  [[ -f "$INIT" ]] || { echo "missing AR init: $INIT" >&2; exit 2; }
  INIT_ARGS=(--init-predictor-checkpoint "$INIT")
fi

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$TRAIN" \
  --oracle-checkpoint "$ORACLE" \
  --route ar_residual_index \
  --condition-mode z1_x1 \
  --hard-fsq \
  --hidden 128 \
  --blocks 8 \
  --attention-every 2 \
  --heads 4 \
  --ar-base-checkpoint "$BASE" \
  "${INIT_ARGS[@]}" \
  --receiver-d2-arch oracle \
  --receiver-combiner oracle \
  --train-deploy-metric-batches 1 \
  --train-deploy-metric-batch-size 2 \
  --lambda-index 0.01 \
  --lambda-q 0.05 \
  --lambda-final 1 \
  --lambda-oracle 0.25 \
  --hard-example-power 0.5 \
  --hard-example-min-weight 0.25 \
  --hard-example-max-weight 4 \
  --lr "$LR" \
  --decoder-lr 0 \
  --weight-decay 1e-2 \
  --grad-clip-norm 1 \
  --goal-delta-db 0.5 \
  --min-pred-ablation-drop 0.1 \
  --min-condition-shuffle-drop 0.1 \
  --min-oracle-delta-db 0.8 \
  --min-oracle-ablation-drop 0.5 \
  --epochs "$EPOCHS" \
  --batch-size 16 \
  --test-batch 4 \
  --num-workers 12 \
  --val-num-workers 4 \
  --val-every 5 \
  --latest-every 5 \
  --save-dir "$SAVE_ROOT" \
  --version "$VERSION" \
  --seed 20260715 \
  "$@" \
  2>&1 | tee -a "$LOG_ROOT/${VERSION}.log"
