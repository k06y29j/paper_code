#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
SCRIPT="$ROOT/MY-V2/jscc-f/explore-2/train_fsq_receiver.py"
ORACLE="$ROOT/MY-V2/jscc-f/explore/checkpoints-direct/direct-cnn-d3-l5x5x5-group-compatible-blend-e100/jscc_f_direct-cnn-d3-l5x5x5-group-compatible-blend-e100_layer2_fsq_direct_cnn_d3_l5x5x5_group_compatible_blend_goal_best.pth"
SAVE_ROOT="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-fsq-generation"
LOG_ROOT="$ROOT/MY-V2/jscc-f/explore-2/logs-fsq-generation"
mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

GPU="${GPU:-1}"
VERSION="${VERSION:-cnn-fsq-k125-paper-flow-v1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-12}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
FLOW_STEPS="${FLOW_STEPS:-32}"
FLOW_NOISE="${FLOW_NOISE:-gaussian}"
EXTRA_ARGS=("$@")

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$SCRIPT" \
  --oracle-checkpoint "$ORACLE" \
  --route flow_matching \
  --condition-mode z1_x1 \
  --hard-fsq \
  --hidden 128 \
  --blocks 8 \
  --attention-every 2 \
  --heads 4 \
  --flow-sample-steps "$FLOW_STEPS" \
  --flow-sample-noise "$FLOW_NOISE" \
  --flow-sample-seed 20260713 \
  --flow-time-scale 1000 \
  --finetune-d2 \
  --independent-receiver-d2 \
  --receiver-d2-arch oracle \
  --receiver-combiner oracle \
  --lambda-flow 0.1 \
  --lambda-q 0.05 \
  --lambda-index 0.0 \
  --lambda-final 1.0 \
  --lambda-oracle 0.25 \
  --lr 2e-4 \
  --decoder-lr 5e-5 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --goal-delta-db 0.5 \
  --val-every 5 \
  --latest-every 5 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch 4 \
  --num-workers "$NUM_WORKERS" \
  --val-num-workers "$VAL_NUM_WORKERS" \
  --version "$VERSION" \
  --save-dir "$SAVE_ROOT" \
  --seed 20260713 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG_ROOT/${VERSION}.log"
