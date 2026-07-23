#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
TRAIN="$ROOT/MY-V2/jscc-f/explore-2/train_fsq_receiver.py"
ORACLE="$ROOT/MY-V2/jscc-f/explore/checkpoints-direct/direct-cnn-d3-l5x5x5-group-compatible-blend-e100/jscc_f_direct-cnn-d3-l5x5x5-group-compatible-blend-e100_layer2_fsq_direct_cnn_d3_l5x5x5_group_compatible_blend_goal_best.pth"
SAVE_ROOT="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-fsq-generation"
LOG_ROOT="$ROOT/MY-V2/jscc-f/explore-2/logs-fsq-generation"

MODE="${MODE:-continuous}"
case "$MODE" in
  continuous)
    GPU="${GPU:-0}"
    VERSION="${VERSION:-cnn-fsq-k125-paper-cdcd-cont-cosinevp-n12-v1}"
    MODE_ARGS=()
    ;;
  hard)
    GPU="${GPU:-2}"
    VERSION="${VERSION:-cnn-fsq-k125-paper-cdcd-hard-cosinevp-n12-v1}"
    MODE_ARGS=(--hard-fsq)
    ;;
  *)
    echo "MODE must be continuous or hard, got: $MODE" >&2
    exit 2
    ;;
esac

EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CDCD_STEPS="${CDCD_STEPS:-12}"
PRIOR_SCALE="${PRIOR_SCALE:-1.0}"
mkdir -p "$SAVE_ROOT" "$LOG_ROOT"
[[ -f "$ORACLE" ]] || { echo "missing K125 oracle: $ORACLE" >&2; exit 2; }

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$TRAIN" \
  --oracle-checkpoint "$ORACLE" \
  --route categorical_diffusion \
  --condition-mode z1_x1 \
  "${MODE_ARGS[@]}" \
  --hidden 128 \
  --blocks 8 \
  --attention-every 2 \
  --heads 4 \
  --cdcd-sample-steps "$CDCD_STEPS" \
  --cdcd-sample-seed 20260717 \
  --cdcd-time-scale 1000 \
  --cdcd-prior-scale "$PRIOR_SCALE" \
  --cdcd-schedule cosine_vp \
  --receiver-d2-arch oracle \
  --receiver-combiner oracle \
  --train-deploy-metric-batches 1 \
  --train-deploy-metric-batch-size 2 \
  --lambda-index 1 \
  --lambda-q 0 \
  --lambda-final 0 \
  --lambda-oracle 0 \
  --lambda-flow 0 \
  --lambda-zero-anchor 0 \
  --lambda-shuffle-anchor 0 \
  --hard-example-power 0 \
  --lr 1e-4 \
  --decoder-lr 0 \
  --weight-decay 0 \
  --grad-clip-norm 1 \
  --goal-delta-db 0.5 \
  --min-pred-ablation-drop 0.1 \
  --min-condition-shuffle-drop 0.1 \
  --min-oracle-delta-db 0.8 \
  --min-oracle-ablation-drop 0.5 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch 4 \
  --num-workers 12 \
  --val-num-workers 4 \
  --val-every 5 \
  --latest-every 5 \
  --save-dir "$SAVE_ROOT" \
  --version "$VERSION" \
  --seed 20260717 \
  "$@" \
  2>&1 | tee -a "$LOG_ROOT/${VERSION}.log"
