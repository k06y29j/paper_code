#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAVE_DIR="${SAVE_DIR:-MY/checkpoints-cvq}"
DATA_DIR="${DATA_DIR:-/workspace/yongjia/datasets/DIV2K}"
PY="${PY:-conda run -n cddm_ddnm python}"

BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH="${TEST_BATCH:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-8}"
VAL_EVERY="${VAL_EVERY:-5}"
LATEST_EVERY="${LATEST_EVERY:-10}"
SEED="${SEED:-20260605}"
K_A="${K_A:-1024}"
K_B="${K_B:-512}"
LAMBDA_STAGE6_REC="${LAMBDA_STAGE6_REC:-100}"
INIT_JSCC_ENCODER="${INIT_JSCC_ENCODER:-MY/checkpoints-jscc/encoder_snr9_channel_awgn_C36.pt}"
INIT_JSCC_DECODER="${INIT_JSCC_DECODER:-MY/checkpoints-jscc/decoder_snr9_channel_awgn_C36.pt}"

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
  --batch-size "$BATCH_SIZE"
  --test-batch "$TEST_BATCH"
  --num-workers "$NUM_WORKERS"
  --val-num-workers "$VAL_NUM_WORKERS"
  --val-every "$VAL_EVERY"
  --latest-every "$LATEST_EVERY"
  --seed "$SEED"
  --lambda-stage6-rec "$LAMBDA_STAGE6_REC"
)

echo "=== Stage 1: C=36 JSCC + tail nested dropout warmup | 800 epochs ==="
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 1 --epochs 800

echo "=== Stage 2: extract normalized tail and initialize codebook | epoch 0 ==="
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 2 --epochs 0

echo "=== Stage 3: JSCC + CVQ joint training | 200 epochs ==="
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 3 --epochs 200

echo "=== Stage 4: oracle evaluation | epoch 0 ==="
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 4 --epochs 0

echo "=== Stage 5: CAR prediction q17:36 | 200 epochs ==="
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 5 --epochs 200

echo "=== Stage 6: CAR + decoder finetune | 0 epochs ==="
$PY MY/train_cvq/main.py "${common_args[@]}" --stage 6 --epochs 0

echo "done: $SAVE_DIR"
