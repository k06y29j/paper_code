#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
TRAIN="$ROOT/MY-V2/jscc-f/explore-2/train_fsq_receiver.py"
ORACLE="$ROOT/MY-V2/jscc-f/explore/checkpoints-direct/direct-cnn-d3-l5x5x5-group-compatible-blend-e100/jscc_f_direct-cnn-d3-l5x5x5-group-compatible-blend-e100_layer2_fsq_direct_cnn_d3_l5x5x5_group_compatible_blend_goal_best.pth"
SAVE_ROOT="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-125"
LOG_ROOT="$ROOT/MY-V2/jscc-f/explore-2/logs-125"

V5_PREDICTOR="$SAVE_ROOT/cnn-fsq-k125-independent-d2-v5/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l5x5x5_d2ft-oracle_independent-d2_best.pth"
CONT_VERSION="cnn-fsq-k125-frozen-d2-combiner-cont-v1"
HARD_VERSION="cnn-fsq-k125-frozen-d2-combiner-hard-v1"
CONT_BEST="$SAVE_ROOT/$CONT_VERSION/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l5x5x5_frozen_best.pth"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {continuous|hard} [extra trainer args...]" >&2
  exit 2
fi

STAGE="$1"
shift
GPU="${GPU:-0}"
mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

COMMON=(
  --oracle-checkpoint "$ORACLE"
  --route direct_q
  --condition-mode z1_x1
  --hidden 128
  --blocks 8
  --attention-every 2
  --heads 4
  --receiver-d2-arch oracle
  --receiver-combiner oracle
  --lambda-index 0
  --lambda-final 1
  --lambda-oracle 0.25
  --decoder-lr 0
  --weight-decay 1e-4
  --grad-clip-norm 1
  --epochs 100
  --batch-size 32
  --test-batch 4
  --num-workers 16
  --val-num-workers 4
  --val-every 5
  --latest-every 5
  --max-train-batches 0
  --max-val-batches 0
  --goal-delta-db 0.5
  --min-pred-ablation-drop 0.1
  --min-condition-shuffle-drop 0.1
  --min-oracle-delta-db 0.8
  --min-oracle-ablation-drop 0.5
  --save-dir "$SAVE_ROOT"
)

case "$STAGE" in
  continuous)
    [[ -f "$V5_PREDICTOR" ]] || { echo "missing predictor init: $V5_PREDICTOR" >&2; exit 2; }
    VERSION="$CONT_VERSION"
    ARGS=(
      --init-predictor-checkpoint "$V5_PREDICTOR"
      --lambda-q 0.02
      --lr 5e-5
      --seed 20260714
    )
    ;;
  hard)
    [[ -f "$CONT_BEST" ]] || { echo "missing continuous frozen best: $CONT_BEST" >&2; exit 2; }
    VERSION="$HARD_VERSION"
    ARGS=(
      --hard-fsq
      --init-predictor-checkpoint "$CONT_BEST"
      --lambda-q 0.05
      --lr 2e-5
      --seed 20260714
    )
    ;;
  *)
    echo "unknown stage: $STAGE (expected continuous or hard)" >&2
    exit 2
    ;;
esac

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$TRAIN" \
  "${COMMON[@]}" \
  "${ARGS[@]}" \
  --version "$VERSION" \
  "$@" \
  2>&1 | tee -a "$LOG_ROOT/${VERSION}.log"
