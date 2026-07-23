#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
TRAIN="$ROOT/MY-V2/jscc-f/explore-2/train_fsq_receiver.py"
ORACLE="$ROOT/MY-V2/jscc-f/explore/checkpoints-direct/direct-cnn-d3-l5x5x5-group-compatible-blend-e100/jscc_f_direct-cnn-d3-l5x5x5-group-compatible-blend-e100_layer2_fsq_direct_cnn_d3_l5x5x5_group_compatible_blend_goal_best.pth"
INIT="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-fsq-generation/cnn-fsq-k125-paper-joint-ar-frozen-sched-v2/fsq_receiver_ar_joint_index_z1_x1_hard_cnn_d3_l5x5x5_frozen_best.pth"
SAVE_ROOT="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-fsq-generation"
LOG_ROOT="$ROOT/MY-V2/jscc-f/explore-2/logs-fsq-generation"

GPU="${GPU:-2}"
VERSION="${VERSION:-cnn-fsq-k125-paper-joint-ar-rollout-aligned-v3}"
EPOCHS="${EPOCHS:-60}"
ROLLOUT_BATCH="${ROLLOUT_BATCH:-4}"
mkdir -p "$SAVE_ROOT" "$LOG_ROOT"
[[ -f "$ORACLE" ]] || { echo "missing K125 oracle: $ORACLE" >&2; exit 2; }
[[ -f "$INIT" ]] || { echo "missing AR predictor init: $INIT" >&2; exit 2; }

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$TRAIN" \
  --oracle-checkpoint "$ORACLE" \
  --route ar_joint_index \
  --condition-mode z1_x1 \
  --hard-fsq \
  --hidden 128 \
  --blocks 8 \
  --attention-every 2 \
  --heads 4 \
  --receiver-d2-arch oracle \
  --receiver-combiner oracle \
  --init-predictor-checkpoint "$INIT" \
  --ar-history-corruption-start 0.5 \
  --ar-history-corruption-end 1.0 \
  --ar-rollout-history-batch "$ROLLOUT_BATCH" \
  --train-deploy-metric-batches 1 \
  --train-deploy-metric-batch-size 2 \
  --lambda-index 0.01 \
  --lambda-q 0.05 \
  --lambda-final 1 \
  --lambda-oracle 0.25 \
  --hard-example-power 0.5 \
  --hard-example-min-weight 0.25 \
  --hard-example-max-weight 4 \
  --lr 5e-5 \
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
