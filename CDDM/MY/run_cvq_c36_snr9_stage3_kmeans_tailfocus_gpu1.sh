#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PY="${PY:-conda run --no-capture-output -n cddm_ddnm python}"

export SAVE_DIR="${SAVE_DIR:-MY/checkpoints-cvq-kmeans-tailfocus-e2e-vq2e3-k512}"
export CVQ_MODE="${CVQ_MODE:-single}"
export K_A="${K_A:-512}"
export K_B="${K_B:-256}"
export KMEANS_ITERS="${KMEANS_ITERS:-25}"
export STAGE3_EPOCHS="${STAGE3_EPOCHS:-180}"

# Keep the receiver-side predictor difficulty unchanged while making the tail
# latent/codebook/decoder co-adapt to quantized reconstruction.
export LR="${LR:-2e-5}"
export LAMBDA_VQ_START="${LAMBDA_VQ_START:-0.0001}"
export LAMBDA_VQ_END="${LAMBDA_VQ_END:-0.0025}"
export LAMBDA_USAGE="${LAMBDA_USAGE:-0.0001}"
export LAMBDA_CONT_STAGE3="${LAMBDA_CONT_STAGE3:-0.005}"
export LAMBDA_TAIL_DISTILL="${LAMBDA_TAIL_DISTILL:-0.005}"
export LAMBDA_GAIN_STAGE3="${LAMBDA_GAIN_STAGE3:-0.5}"
export LAMBDA_PREFIX_FLOOR_STAGE3="${LAMBDA_PREFIX_FLOOR_STAGE3:-10.0}"
export STAGE3_PREFIX_FLOOR_DB="${STAGE3_PREFIX_FLOOR_DB:-25.2}"
export STAGE3_SCORE="${STAGE3_SCORE:-psnr_vq}"

export STAGE3_TRAIN_ENCODER="${STAGE3_TRAIN_ENCODER:-1}"
export STAGE3_TRAIN_DECODER="${STAGE3_TRAIN_DECODER:-1}"
export STAGE3_FREEZE_ENCODER_BODY="${STAGE3_FREEZE_ENCODER_BODY:-1}"
export STAGE3_FREEZE_PREFIX_HEAD="${STAGE3_FREEZE_PREFIX_HEAD:-1}"
export DEAD_CODE_RESTART_EVERY="${DEAD_CODE_RESTART_EVERY:-0}"

export STAGE3_MIN_PSNR_PREFIX="${STAGE3_MIN_PSNR_PREFIX:-25.2}"
export STAGE3_MIN_TAIL_GAIN_VQ="${STAGE3_MIN_TAIL_GAIN_VQ:-0.05}"
export STAGE3_MIN_TAIL_GAIN_CONT="${STAGE3_MIN_TAIL_GAIN_CONT:-0.5}"
export STAGE3_MIN_USED_A="${STAGE3_MIN_USED_A:-50}"
export STAGE3_MIN_USED_B="${STAGE3_MIN_USED_B:-20}"
export STAGE3_MIN_PERPLEXITY_A="${STAGE3_MIN_PERPLEXITY_A:-20}"
export STAGE3_MIN_PERPLEXITY_B="${STAGE3_MIN_PERPLEXITY_B:-8}"

mkdir -p "$SAVE_DIR"

{
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "SAVE_DIR=$SAVE_DIR"
  echo "CVQ_MODE=$CVQ_MODE K_A=$K_A K_B=$K_B"
  echo "tail_focus=freeze_encoder_body:${STAGE3_FREEZE_ENCODER_BODY},freeze_prefix_head:${STAGE3_FREEZE_PREFIX_HEAD}"
  echo "prefix_floor_db=$STAGE3_PREFIX_FLOOR_DB lambda_prefix_floor=$LAMBDA_PREFIX_FLOOR_STAGE3"
  echo "lambda_vq=$LAMBDA_VQ_START->$LAMBDA_VQ_END lambda_usage=$LAMBDA_USAGE"
  echo "lambda_cont=$LAMBDA_CONT_STAGE3 lambda_tail_distill=$LAMBDA_TAIL_DISTILL lambda_gain=$LAMBDA_GAIN_STAGE3"
  echo "score=$STAGE3_SCORE"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
} > "$SAVE_DIR/launch.meta"

MY/run_cvq_c36_snr9_stage3_e2e.sh 2>&1 | tee "$SAVE_DIR/launch.console.log"
