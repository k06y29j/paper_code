#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
TRAIN="$ROOT/MY-V2/jscc-f/explore-2/train_fsq_receiver.py"
ORACLE="$ROOT/MY-V2/jscc-f/explore/checkpoints-direct/direct-cnn-d3-l5x5x5-group-compatible-blend-e100/jscc_f_direct-cnn-d3-l5x5x5-group-compatible-blend-e100_layer2_fsq_direct_cnn_d3_l5x5x5_group_compatible_blend_goal_best.pth"
SAVE_ROOT="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-125"
LOG_ROOT="$ROOT/MY-V2/jscc-f/explore-2/logs-125"
mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 STAGE [extra trainer args...]" >&2
  echo "STAGE: v4-base | v5-base | v7-hard | v11-base | v21-continuous" >&2
  exit 2
fi
STAGE="$1"
shift
GPU="${GPU:-0}"

V4_VERSION="cnn-fsq-k125-joint-predictable-v4"
V5_VERSION="cnn-fsq-k125-independent-d2-v5"
V7_VERSION="cnn-fsq-k125-independent-hard-final-v7"
V11_VERSION="cnn-fsq-k125-sender-qhat-deploy-v11"
V21_VERSION="cnn-fsq-k125-sender-aligned-rxd2ft-v21"

V4_CKPT="$SAVE_ROOT/$V4_VERSION/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l5x5x5_d2ft-oracle_best.pth"
V5_CKPT="$SAVE_ROOT/$V5_VERSION/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l5x5x5_d2ft-oracle_independent-d2_best.pth"
V11_DEFAULT_CKPT="$SAVE_ROOT/$V11_VERSION/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l5x5x5_frozen_independent-d2_sender-aligned-q_best.pth"
V11_CKPT="${V11_INIT_OVERRIDE:-$V11_DEFAULT_CKPT}"

COMMON=(
  --oracle-checkpoint "$ORACLE"
  --route direct_q
  --condition-mode z1_x1
  --hidden 128
  --blocks 8
  --attention-every 2
  --heads 4
  --receiver-combiner oracle
  --goal-delta-db 0.5
  --min-pred-ablation-drop 0.1
  --min-condition-shuffle-drop 0.1
  --min-oracle-delta-db 0.8
  --min-oracle-ablation-drop 0.5
  --batch-size 32
  --test-batch 4
  --num-workers 16
  --val-num-workers 4
  --grad-clip-norm 1.0
  --save-dir "$SAVE_ROOT"
)

case "$STAGE" in
  v4-base)
    VERSION="$V4_VERSION"
    ARGS=(
      --finetune-d2
      --joint-predictable-oracle
      --lambda-q 0.005
      --lambda-index 0
      --lambda-final 1
      --lambda-oracle 0
      --lambda-sender-final 1
      --lambda-sender-predictability 0.1
      --lr 5e-5
      --decoder-lr 1e-5
      --sender-lr 1e-5
      --weight-decay 1e-4
      --epochs 40
      --val-every 5
      --latest-every 5
      --seed 20260712
    )
    ;;
  v5-base)
    [[ -f "$V4_CKPT" ]] || { echo "missing K125 v4 prerequisite: $V4_CKPT" >&2; exit 2; }
    VERSION="$V5_VERSION"
    ARGS=(
      --finetune-d2
      --independent-receiver-d2
      --init-receiver-checkpoint "$V4_CKPT"
      --lambda-q 0.02
      --lambda-index 0
      --lambda-final 1
      --lambda-oracle 0.25
      --lr 2e-5
      --decoder-lr 5e-5
      --sender-lr 5e-5
      --weight-decay 1e-4
      --epochs 100
      --val-every 5
      --latest-every 5
      --seed 20260712
    )
    ;;
  v7-hard)
    [[ -f "$V4_CKPT" ]] || { echo "missing K125 v4 prerequisite: $V4_CKPT" >&2; exit 2; }
    VERSION="$V7_VERSION"
    ARGS=(
      --hard-fsq
      --finetune-d2
      --independent-receiver-d2
      --init-receiver-checkpoint "$V4_CKPT"
      --lambda-q 0
      --lambda-index 0
      --lambda-final 1
      --lambda-oracle 0
      --lr 1e-4
      --decoder-lr 2e-4
      --sender-lr 5e-5
      --weight-decay 1e-5
      --epochs 100
      --val-every 5
      --latest-every 5
      --seed 20260712
    )
    ;;
  v11-base)
    [[ -f "$V5_CKPT" ]] || { echo "missing K125 v5 prerequisite: $V5_CKPT" >&2; exit 2; }
    VERSION="$V11_VERSION"
    ARGS=(
      --independent-receiver-d2
      --sender-aligned-q
      --init-receiver-checkpoint "$V5_CKPT"
      --lambda-q 0.02
      --lambda-index 0.01
      --lambda-final 1
      --lambda-oracle 0.25
      --sender-align-lambda-predictability 0.1
      --sender-align-lambda-final 1
      --sender-align-lambda-qhat-final 5
      --sender-align-lambda-receiver-final 1
      --sender-align-e2-lr 1e-5
      --sender-align-decoder-lr 5e-5
      --sender-align-max-q-mse 0.004
      --sender-align-min-oracle-delta-db 0.8
      --sender-align-min-oracle-ablation-drop 0.5
      --sender-deployment-min-delta-db 0.5
      --sender-deployment-min-ablation-drop 0.1
      --lr 2e-4
      --decoder-lr 5e-5
      --weight-decay 1e-4
      --epochs 40
      --val-every 1
      --latest-every 5
      --seed 20260712
    )
    ;;
  v21-continuous)
    [[ -f "$V11_CKPT" ]] || { echo "missing K125 v11 prerequisite: $V11_CKPT" >&2; exit 2; }
    VERSION="$V21_VERSION"
    ARGS=(
      --finetune-d2
      --independent-receiver-d2
      --receiver-d2-arch oracle
      --sender-aligned-q
      --init-receiver-checkpoint "$V11_CKPT"
      --lambda-q 0.02
      --lambda-index 0.01
      --lambda-final 1
      --lambda-oracle 0.25
      --sender-align-lambda-predictability 0.1
      --sender-align-lambda-final 1
      --sender-align-lambda-qhat-final 5
      --sender-align-lambda-receiver-final 1
      --sender-align-e2-lr 1e-5
      --sender-align-decoder-lr 5e-5
      --sender-align-max-q-mse 0.004
      --sender-align-min-oracle-delta-db 0.8
      --sender-align-min-oracle-ablation-drop 0.5
      --sender-deployment-min-delta-db 0.5
      --sender-deployment-min-ablation-drop 0.1
      --lr 2e-4
      --decoder-lr 5e-5
      --weight-decay 1e-4
      --epochs 40
      --val-every 2
      --latest-every 2
      --seed 20260713
    )
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

VERSION="${VERSION_OVERRIDE:-$VERSION}"

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$TRAIN" \
  "${COMMON[@]}" \
  "${ARGS[@]}" \
  --version "$VERSION" \
  "$@" \
  2>&1 | tee -a "$LOG_ROOT/${VERSION}.log"
