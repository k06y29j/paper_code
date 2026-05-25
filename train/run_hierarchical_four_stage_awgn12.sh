#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${1:-0}"
ROOT="${2:-checkpoints-ar/hierarchical_four_stage_awgn12}"

mkdir -p "${ROOT}"

run_stage() {
  local stage="$1"
  local run_dir="$2"
  shift 2
  mkdir -p "${run_dir}"
  echo "[$(date '+%F %T')] stage=${stage} run_dir=${run_dir}" | tee -a "${ROOT}/pipeline.log"
  STAGE="${stage}" train/run_hierarchical_swin_ar_awgn12.sh "${GPU}" "${run_dir}" "$@" 2>&1 | tee -a "${run_dir}/train.stdout"
}

PRE_DIR="${ROOT}/01_hierarchical_swin_pretrain"
AR_DIR="${ROOT}/02_hierarchical_group_ar_frozen_swin"
DEC_DIR="${ROOT}/03_hierarchical_group_ar_decoder_tune"
JOINT_DIR="${ROOT}/04_hierarchical_group_ar_joint_tune"

export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
export CACHE_WORKERS="${CACHE_WORKERS:-12}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-2}"
export AR_ARCH="${AR_ARCH:-simple}"
export AR_HIDDEN="${AR_HIDDEN:-160}"
export AR_DEPTH="${AR_DEPTH:-4}"

EPOCHS="${PRETRAIN_EPOCHS:-400}" LR="${PRETRAIN_LR:-5e-5}" \
  run_stage hierarchical_swin_pretrain "${PRE_DIR}" \
  --lambda_full "${PRETRAIN_LAMBDA_FULL:-4.0}" \
  --lambda_low "${PRETRAIN_LAMBDA_LOW:-0.05}" \
  --lambda_mid "${PRETRAIN_LAMBDA_MID:-0.03}" \
  --lambda_midhigh "${PRETRAIN_LAMBDA_MIDHIGH:-0.02}" \
  --lambda_energy "${PRETRAIN_LAMBDA_ENERGY:-1e-4}" \
  --lambda_order "${PRETRAIN_LAMBDA_ORDER:-1e-3}"

export INIT_HIER_CKPT="${PRE_DIR}/hierarchical_swin_pretrain_best.pth"
EPOCHS="${AR_EPOCHS:-400}" LR="${AR_LR_BASE:-5e-5}" AR_LR="${AR_LR:-2e-4}" \
  TEACHER_FORCE_EPOCHS="${TEACHER_FORCE_EPOCHS:-50}" TEACHER_DECAY_EPOCHS="${TEACHER_DECAY_EPOCHS:-100}" \
  run_stage hierarchical_group_ar_frozen_swin "${AR_DIR}" \
  --lambda_recv "${AR_LAMBDA_RECV:-1.0}" \
  --lambda_ar "${AR_LAMBDA_AR:-0.1}"

export INIT_HIER_CKPT="${AR_DIR}/hierarchical_group_ar_frozen_swin_best.pth"
EPOCHS="${DECODER_TUNE_EPOCHS:-120}" LR="${DECODER_TUNE_LR_BASE:-5e-5}" DECODER_LR="${DECODER_TUNE_LR:-1e-5}" \
  run_stage hierarchical_group_ar_decoder_tune "${DEC_DIR}" \
  --lambda_recv "${DECODER_TUNE_LAMBDA_RECV:-1.0}" \
  --lambda_full "${DECODER_TUNE_LAMBDA_FULL:-0.2}"

export INIT_HIER_CKPT="${DEC_DIR}/hierarchical_group_ar_decoder_tune_best.pth"
EPOCHS="${JOINT_TUNE_EPOCHS:-120}" LR="${JOINT_TUNE_LR_BASE:-5e-5}" \
  DECODER_LR="${JOINT_TUNE_DECODER_LR:-5e-6}" AR_LR="${JOINT_TUNE_AR_LR:-2e-5}" \
  run_stage hierarchical_group_ar_joint_tune "${JOINT_DIR}" \
  --lambda_recv "${JOINT_TUNE_LAMBDA_RECV:-1.0}" \
  --lambda_full "${JOINT_TUNE_LAMBDA_FULL:-0.2}" \
  --lambda_low "${JOINT_TUNE_LAMBDA_LOW:-0.02}" \
  --lambda_mid "${JOINT_TUNE_LAMBDA_MID:-0.01}" \
  --lambda_midhigh "${JOINT_TUNE_LAMBDA_MIDHIGH:-0.01}" \
  --lambda_ar "${JOINT_TUNE_LAMBDA_AR:-0.05}"

echo "[$(date '+%F %T')] all stages finished" | tee -a "${ROOT}/pipeline.log"
