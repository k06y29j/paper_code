#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAVE_DIR="${SAVE_DIR:-MY/checkpoints-cvq-e2e-lowvq-k512}"
DATA_DIR="${DATA_DIR:-/workspace/yongjia/datasets/DIV2K}"
PY="${PY:-conda run -n cddm_ddnm python}"

STAGE1_CKPT="${STAGE1_CKPT:-MY/checkpoints-cvq/cvq_c36_snr9_stage1_best.pth}"
INIT_JSCC_ENCODER="${INIT_JSCC_ENCODER:-MY/checkpoints-jscc/encoder_snr9_channel_awgn_C36.pt}"
INIT_JSCC_DECODER="${INIT_JSCC_DECODER:-MY/checkpoints-jscc/decoder_snr9_channel_awgn_C36.pt}"

BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH="${TEST_BATCH:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-8}"
VAL_EVERY="${VAL_EVERY:-5}"
LATEST_EVERY="${LATEST_EVERY:-10}"
SEED="${SEED:-20260605}"
K_A="${K_A:-512}"
K_B="${K_B:-256}"
CVQ_MODE="${CVQ_MODE:-single}"
RVQ_STAGES="${RVQ_STAGES:-2}"
PATCH_SIZE="${PATCH_SIZE:-4}"
FSQ_LEVELS_A="${FSQ_LEVELS_A:-16}"
FSQ_LEVELS_B="${FSQ_LEVELS_B:-16}"
FSQ_SCALE="${FSQ_SCALE:-3.0}"
INIT_VECTORS_PER_CODE="${INIT_VECTORS_PER_CODE:-8}"
KMEANS_ITERS="${KMEANS_ITERS:-25}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-400}"
LR="${LR:-2e-5}"
CAR_ARCH="${CAR_ARCH:-auto}"
CAR_DIM="${CAR_DIM:-256}"
CAR_HEADS="${CAR_HEADS:-8}"
CAR_LAYERS="${CAR_LAYERS:-4}"
CAR_DROPOUT="${CAR_DROPOUT:-0.1}"
CAR_WINDOW_SIZE="${CAR_WINDOW_SIZE:-4}"
CAR_LR="${CAR_LR:-1e-4}"
LAMBDA_VQ_START="${LAMBDA_VQ_START:-0.0005}"
LAMBDA_VQ_END="${LAMBDA_VQ_END:-0.005}"
LAMBDA_USAGE="${LAMBDA_USAGE:-0.0005}"
LAMBDA_CONT_STAGE3="${LAMBDA_CONT_STAGE3:-0.05}"
LAMBDA_TAIL_DISTILL="${LAMBDA_TAIL_DISTILL:-0.02}"
LAMBDA_GAIN_STAGE3="${LAMBDA_GAIN_STAGE3:-0.0}"
LAMBDA_PREFIX_FLOOR_STAGE3="${LAMBDA_PREFIX_FLOOR_STAGE3:-0.0}"
STAGE3_GAIN_MARGIN_DB="${STAGE3_GAIN_MARGIN_DB:-0.2}"
STAGE3_PREFIX_FLOOR_DB="${STAGE3_PREFIX_FLOOR_DB:-25.2}"
STAGE3_SCORE="${STAGE3_SCORE:-tail_gain_vq}"
FREEZE_ENCODER_FIRST="${FREEZE_ENCODER_FIRST:-0}"
STAGE3_TRAIN_ENCODER="${STAGE3_TRAIN_ENCODER:-1}"
STAGE3_TRAIN_DECODER="${STAGE3_TRAIN_DECODER:-1}"
STAGE3_FREEZE_ENCODER_BODY="${STAGE3_FREEZE_ENCODER_BODY:-0}"
STAGE3_FREEZE_PREFIX_HEAD="${STAGE3_FREEZE_PREFIX_HEAD:-0}"
STAGE3_CAR_MODE="${STAGE3_CAR_MODE:-none}"
STAGE3_INIT_CAR_CKPT="${STAGE3_INIT_CAR_CKPT:-}"
STAGE3_CAR_START_EPOCH="${STAGE3_CAR_START_EPOCH:-1}"
LAMBDA_STAGE3_CAR_REC="${LAMBDA_STAGE3_CAR_REC:-0.0}"
LAMBDA_STAGE3_CAR_CE="${LAMBDA_STAGE3_CAR_CE:-0.0}"
LAMBDA_STAGE3_CAR_VALUE="${LAMBDA_STAGE3_CAR_VALUE:-0.0}"
LAMBDA_STAGE3_CAR_ORDINAL="${LAMBDA_STAGE3_CAR_ORDINAL:-0.0}"
LAMBDA_STAGE3_CAR_GAIN="${LAMBDA_STAGE3_CAR_GAIN:-0.0}"
STAGE3_CAR_GAIN_DB="${STAGE3_CAR_GAIN_DB:-0.5}"
STAGE3_CAR_TAIL_SCALE="${STAGE3_CAR_TAIL_SCALE:-1.0}"
STAGE3_CAR_SOFT_TAU="${STAGE3_CAR_SOFT_TAU:-1.0}"
STAGE3_CAR_ALPHA_SWEEP="${STAGE3_CAR_ALPHA_SWEEP:-}"
STAGE3_MIN_TAIL_GAIN_CAR="${STAGE3_MIN_TAIL_GAIN_CAR:-0.0}"
DEAD_CODE_RESTART_EVERY="${DEAD_CODE_RESTART_EVERY:-0}"

mkdir -p "$SAVE_DIR"

common_args=(
  --data-dir "$DATA_DIR"
  --save-dir "$SAVE_DIR"
  --snr-db 9
  --latent-ch 36
  --prefix-ch 16
  --latent-h 16
  --latent-w 16
  --init-jscc-encoder "$INIT_JSCC_ENCODER"
  --init-jscc-decoder "$INIT_JSCC_DECODER"
  --k-a "$K_A"
  --k-b "$K_B"
  --cvq-mode "$CVQ_MODE"
  --rvq-stages "$RVQ_STAGES"
  --patch-size "$PATCH_SIZE"
  --fsq-levels-a "$FSQ_LEVELS_A"
  --fsq-levels-b "$FSQ_LEVELS_B"
  --fsq-scale "$FSQ_SCALE"
  --init-vectors-per-code "$INIT_VECTORS_PER_CODE"
  --kmeans-iters "$KMEANS_ITERS"
  --car-arch "$CAR_ARCH"
  --car-dim "$CAR_DIM"
  --car-heads "$CAR_HEADS"
  --car-layers "$CAR_LAYERS"
  --car-dropout "$CAR_DROPOUT"
  --car-window-size "$CAR_WINDOW_SIZE"
  --car-lr "$CAR_LR"
  --batch-size "$BATCH_SIZE"
  --test-batch "$TEST_BATCH"
  --num-workers "$NUM_WORKERS"
  --val-num-workers "$VAL_NUM_WORKERS"
  --val-every "$VAL_EVERY"
  --latest-every "$LATEST_EVERY"
  --seed "$SEED"
)

echo "=== Stage 2: initialize CVQ from nested-dropout stage1 ==="
echo "stage1 JSCC source: $STAGE1_CKPT"
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 2 --epochs 0 --init-ckpt "$STAGE1_CKPT"

echo "=== Stage 3: constrained end-to-end JSCC + CVQ | ${STAGE3_EPOCHS} epochs ==="
echo "stage3 teacher JSCC source: $STAGE1_CKPT"
stage3_train_flags=()
if [[ "$STAGE3_TRAIN_ENCODER" == "1" || "$STAGE3_TRAIN_ENCODER" == "true" || "$STAGE3_TRAIN_ENCODER" == "yes" ]]; then
  stage3_train_flags+=(--stage3-train-encoder)
else
  stage3_train_flags+=(--no-stage3-train-encoder)
fi
if [[ "$STAGE3_TRAIN_DECODER" == "1" || "$STAGE3_TRAIN_DECODER" == "true" || "$STAGE3_TRAIN_DECODER" == "yes" ]]; then
  stage3_train_flags+=(--stage3-train-decoder)
else
  stage3_train_flags+=(--no-stage3-train-decoder)
fi
if [[ "$STAGE3_FREEZE_ENCODER_BODY" == "1" || "$STAGE3_FREEZE_ENCODER_BODY" == "true" || "$STAGE3_FREEZE_ENCODER_BODY" == "yes" ]]; then
  stage3_train_flags+=(--stage3-freeze-encoder-body)
fi
if [[ "$STAGE3_FREEZE_PREFIX_HEAD" == "1" || "$STAGE3_FREEZE_PREFIX_HEAD" == "true" || "$STAGE3_FREEZE_PREFIX_HEAD" == "yes" ]]; then
  stage3_train_flags+=(--stage3-freeze-prefix-head)
fi
if [[ -n "$STAGE3_INIT_CAR_CKPT" ]]; then
  stage3_train_flags+=(--stage3-init-car-ckpt "$STAGE3_INIT_CAR_CKPT")
fi
$PY MY/train_cvq/main.py "${common_args[@]}" \
  --stage 3 \
  --epochs "$STAGE3_EPOCHS" \
  --lr "$LR" \
  "${stage3_train_flags[@]}" \
  --freeze-encoder-first "$FREEZE_ENCODER_FIRST" \
  --lambda-vq-start "$LAMBDA_VQ_START" \
  --lambda-vq-end "$LAMBDA_VQ_END" \
  --vq-ramp-epochs 40 \
  --lambda-prefix-stage3 0.0 \
  --lambda-nested-stage3 0.0 \
  --lambda-cont-stage3 "$LAMBDA_CONT_STAGE3" \
  --lambda-tail-distill "$LAMBDA_TAIL_DISTILL" \
  --lambda-usage "$LAMBDA_USAGE" \
  --lambda-gain-stage3 "$LAMBDA_GAIN_STAGE3" \
  --lambda-prefix-floor-stage3 "$LAMBDA_PREFIX_FLOOR_STAGE3" \
  --stage3-gain-margin-db "$STAGE3_GAIN_MARGIN_DB" \
  --stage3-prefix-floor-db "$STAGE3_PREFIX_FLOOR_DB" \
  --stage3-score "$STAGE3_SCORE" \
  --stage3-car-mode "$STAGE3_CAR_MODE" \
  --stage3-car-start-epoch "$STAGE3_CAR_START_EPOCH" \
  --lambda-stage3-car-rec "$LAMBDA_STAGE3_CAR_REC" \
  --lambda-stage3-car-ce "$LAMBDA_STAGE3_CAR_CE" \
  --lambda-stage3-car-value "$LAMBDA_STAGE3_CAR_VALUE" \
  --lambda-stage3-car-ordinal "$LAMBDA_STAGE3_CAR_ORDINAL" \
  --lambda-stage3-car-gain "$LAMBDA_STAGE3_CAR_GAIN" \
  --stage3-car-gain-db "$STAGE3_CAR_GAIN_DB" \
  --stage3-car-tail-scale "$STAGE3_CAR_TAIL_SCALE" \
  --stage3-car-soft-tau "$STAGE3_CAR_SOFT_TAU" \
  --stage3-car-alpha-sweep="$STAGE3_CAR_ALPHA_SWEEP" \
  --stage3-usage-tau 10.0 \
  --stage3-teacher-ckpt "$STAGE1_CKPT" \
  --dead-code-restart-every "$DEAD_CODE_RESTART_EVERY" \
  --stage3-min-tail-gain-vq "${STAGE3_MIN_TAIL_GAIN_VQ:-0.1}" \
  --stage3-min-tail-gain-car "$STAGE3_MIN_TAIL_GAIN_CAR" \
  --stage3-min-tail-gain-cont "${STAGE3_MIN_TAIL_GAIN_CONT:-1.0}" \
  --stage3-min-psnr-prefix "${STAGE3_MIN_PSNR_PREFIX:-0.0}" \
  --stage3-min-perplexity-a "${STAGE3_MIN_PERPLEXITY_A:-28}" \
  --stage3-min-perplexity-b "${STAGE3_MIN_PERPLEXITY_B:-10}" \
  --stage3-min-used-a "${STAGE3_MIN_USED_A:-50}" \
  --stage3-min-used-b "${STAGE3_MIN_USED_B:-20}"

echo "done stage3: $SAVE_DIR"
