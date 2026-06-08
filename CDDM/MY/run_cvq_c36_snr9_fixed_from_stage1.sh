#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAVE_DIR="${SAVE_DIR:-MY/checkpoints-cvq-fixed-lowvq-k512}"
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
INIT_VECTORS_PER_CODE="${INIT_VECTORS_PER_CODE:-8}"
LAMBDA_VQ_START="${LAMBDA_VQ_START:-0.05}"
LAMBDA_VQ_END="${LAMBDA_VQ_END:-0.1}"
LAMBDA_PREFIX_STAGE3="${LAMBDA_PREFIX_STAGE3:-0.1}"
LAMBDA_NESTED_STAGE3="${LAMBDA_NESTED_STAGE3:-0.1}"
STAGE3_MIN_TAIL_GAIN_VQ="${STAGE3_MIN_TAIL_GAIN_VQ:-0.1}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-200}"
STAGE5_EPOCHS="${STAGE5_EPOCHS:-200}"
STAGE6_EPOCHS="${STAGE6_EPOCHS:-0}"
LAMBDA_STAGE6_REC="${LAMBDA_STAGE6_REC:-100}"

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
  --init-vectors-per-code "$INIT_VECTORS_PER_CODE"
  --lambda-vq-start "$LAMBDA_VQ_START"
  --lambda-vq-end "$LAMBDA_VQ_END"
  --lambda-prefix-stage3 "$LAMBDA_PREFIX_STAGE3"
  --lambda-nested-stage3 "$LAMBDA_NESTED_STAGE3"
  --stage3-min-tail-gain-vq "$STAGE3_MIN_TAIL_GAIN_VQ"
  --batch-size "$BATCH_SIZE"
  --test-batch "$TEST_BATCH"
  --num-workers "$NUM_WORKERS"
  --val-num-workers "$VAL_NUM_WORKERS"
  --val-every "$VAL_EVERY"
  --latest-every "$LATEST_EVERY"
  --seed "$SEED"
  --lambda-stage6-rec "$LAMBDA_STAGE6_REC"
)

echo "=== Stage 2: initialize CVQ from nested-dropout stage1 ==="
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 2 --epochs 0 --init-ckpt "$STAGE1_CKPT"

echo "=== Stage 3: frozen-encoder CVQ + decoder training | ${STAGE3_EPOCHS} epochs ==="
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 3 --epochs "$STAGE3_EPOCHS"

STAGE3_BEST="$SAVE_DIR/cvq_c36_snr9_stage3_best.pth"
if [[ ! -f "$STAGE3_BEST" ]]; then
  echo "stage3 did not produce a best checkpoint; tail_gain_vq did not pass the configured gate."
  echo "expected: $STAGE3_BEST"
  exit 2
fi

echo "=== Stage 4: oracle evaluation ==="
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 4 --epochs 0

echo "=== Stage 5: CAR prediction with image-metric validation | ${STAGE5_EPOCHS} epochs ==="
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 5 --epochs "$STAGE5_EPOCHS"

echo "=== Stage 6: CAR + decoder finetune/eval | ${STAGE6_EPOCHS} epochs ==="
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 6 --epochs "$STAGE6_EPOCHS"

echo "done: $SAVE_DIR"
