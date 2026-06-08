#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAVE_DIR="${SAVE_DIR:-MY/checkpoints-cvq-fsqshared16-tailfocus-e2e-vq1e3}"
SOURCE_CKPT="${SOURCE_CKPT:-MY/checkpoints-cvq-fsq8x8-tailfocus-e2e-vq1e3/cvq_c36_snr9_stage3_best.pth}"
DATA_DIR="${DATA_DIR:-/workspace/yongjia/datasets/DIV2K}"
PY="${PY:-conda run --no-capture-output -n cddm_ddnm python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH="${TEST_BATCH:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-8}"
VAL_EVERY="${VAL_EVERY:-5}"
LATEST_EVERY="${LATEST_EVERY:-10}"
SEED="${SEED:-20260608}"

FSQ_LEVELS="${FSQ_LEVELS:-16}"
INIT_VECTORS_PER_CODE="${INIT_VECTORS_PER_CODE:-8}"
MIN_INIT_VECTORS="${MIN_INIT_VECTORS:-800}"
KMEANS_ITERS="${KMEANS_ITERS:-1}"
KMEANS_CHUNK_SIZE="${KMEANS_CHUNK_SIZE:-1024}"

STAGE3_EPOCHS="${STAGE3_EPOCHS:-160}"
LR="${LR:-2e-5}"
LAMBDA_VQ_START="${LAMBDA_VQ_START:-0.0001}"
LAMBDA_VQ_END="${LAMBDA_VQ_END:-0.001}"
LAMBDA_CONT_STAGE3="${LAMBDA_CONT_STAGE3:-0.01}"
LAMBDA_TAIL_DISTILL="${LAMBDA_TAIL_DISTILL:-0.02}"
LAMBDA_PREFIX_FLOOR_STAGE3="${LAMBDA_PREFIX_FLOOR_STAGE3:-10.0}"
LAMBDA_GAIN_STAGE3="${LAMBDA_GAIN_STAGE3:-0.2}"
LAMBDA_USAGE="${LAMBDA_USAGE:-0.0}"
STAGE3_PREFIX_FLOOR_DB="${STAGE3_PREFIX_FLOOR_DB:-25.2}"
STAGE3_SCORE="${STAGE3_SCORE:-psnr_vq}"
STAGE3_MIN_PSNR_PREFIX="${STAGE3_MIN_PSNR_PREFIX:-25.2}"
STAGE3_MIN_TAIL_GAIN_VQ="${STAGE3_MIN_TAIL_GAIN_VQ:-0.5}"
STAGE3_MIN_TAIL_GAIN_CONT="${STAGE3_MIN_TAIL_GAIN_CONT:-0.5}"
STAGE3_MIN_USED_A="${STAGE3_MIN_USED_A:-8}"
STAGE3_MIN_USED_B="${STAGE3_MIN_USED_B:-8}"
STAGE3_MIN_PERPLEXITY_A="${STAGE3_MIN_PERPLEXITY_A:-4}"
STAGE3_MIN_PERPLEXITY_B="${STAGE3_MIN_PERPLEXITY_B:-4}"

mkdir -p "$SAVE_DIR"

{
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "SAVE_DIR=$SAVE_DIR"
  echo "SOURCE_CKPT=$SOURCE_CKPT"
  echo "CVQ_MODE=fsq_shared FSQ_LEVELS=$FSQ_LEVELS"
  echo "design=one shared scalar FSQ codebook for all 20 tail channels, no A/B split codebooks"
  echo "tail_focus=freeze_encoder_body:1,freeze_prefix_head:1,prefix_floor_db:$STAGE3_PREFIX_FLOOR_DB"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
} > "$SAVE_DIR/launch.meta"

common_args=(
  --data-dir "$DATA_DIR"
  --save-dir "$SAVE_DIR"
  --snr-db 9
  --latent-ch 36
  --prefix-ch 16
  --latent-h 16
  --latent-w 16
  --k-a "$FSQ_LEVELS"
  --k-b "$FSQ_LEVELS"
  --cvq-mode fsq_shared
  --fsq-levels-a "$FSQ_LEVELS"
  --fsq-levels-b "$FSQ_LEVELS"
  --init-vectors-per-code "$INIT_VECTORS_PER_CODE"
  --min-init-vectors "$MIN_INIT_VECTORS"
  --kmeans-iters "$KMEANS_ITERS"
  --kmeans-chunk-size "$KMEANS_CHUNK_SIZE"
  --batch-size "$BATCH_SIZE"
  --test-batch "$TEST_BATCH"
  --num-workers "$NUM_WORKERS"
  --val-num-workers "$VAL_NUM_WORKERS"
  --val-every "$VAL_EVERY"
  --latest-every "$LATEST_EVERY"
  --seed "$SEED"
  --car-arch legacy
)

echo "=== Stage 2: shared FSQ init from $SOURCE_CKPT ===" | tee "$SAVE_DIR/launch.console.log"
$PY MY/train_cvq/main.py "${common_args[@]}" \
  --stage 2 \
  --epochs 0 \
  --init-ckpt "$SOURCE_CKPT" 2>&1 | tee "$SAVE_DIR/stage2_launch.log"

STAGE2_CKPT="$SAVE_DIR/cvq_c36_snr9_stage2_codebook_init.pth"

echo "=== Stage 3: tail-focused shared FSQ fine-tune ===" | tee -a "$SAVE_DIR/launch.console.log"
$PY MY/train_cvq/main.py "${common_args[@]}" \
  --stage 3 \
  --epochs "$STAGE3_EPOCHS" \
  --init-ckpt "$STAGE2_CKPT" \
  --stage3-teacher-ckpt "$SOURCE_CKPT" \
  --lr "$LR" \
  --stage3-train-encoder \
  --stage3-train-decoder \
  --stage3-freeze-encoder-body \
  --stage3-freeze-prefix-head \
  --freeze-encoder-first 0 \
  --lambda-prefix-stage3 0.0 \
  --lambda-nested-stage3 0.0 \
  --lambda-vq-start "$LAMBDA_VQ_START" \
  --lambda-vq-end "$LAMBDA_VQ_END" \
  --vq-ramp-epochs 40 \
  --lambda-cont-stage3 "$LAMBDA_CONT_STAGE3" \
  --lambda-tail-distill "$LAMBDA_TAIL_DISTILL" \
  --lambda-usage "$LAMBDA_USAGE" \
  --lambda-gain-stage3 "$LAMBDA_GAIN_STAGE3" \
  --lambda-prefix-floor-stage3 "$LAMBDA_PREFIX_FLOOR_STAGE3" \
  --stage3-prefix-floor-db "$STAGE3_PREFIX_FLOOR_DB" \
  --stage3-score "$STAGE3_SCORE" \
  --dead-code-restart-every 0 \
  --stage3-min-tail-gain-vq "$STAGE3_MIN_TAIL_GAIN_VQ" \
  --stage3-min-tail-gain-cont "$STAGE3_MIN_TAIL_GAIN_CONT" \
  --stage3-min-psnr-prefix "$STAGE3_MIN_PSNR_PREFIX" \
  --stage3-min-perplexity-a "$STAGE3_MIN_PERPLEXITY_A" \
  --stage3-min-perplexity-b "$STAGE3_MIN_PERPLEXITY_B" \
  --stage3-min-used-a "$STAGE3_MIN_USED_A" \
  --stage3-min-used-b "$STAGE3_MIN_USED_B" 2>&1 | tee -a "$SAVE_DIR/launch.console.log"

echo "done: $SAVE_DIR" | tee -a "$SAVE_DIR/launch.console.log"
