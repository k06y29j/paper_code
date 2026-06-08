#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ROOT="${ROOT:-checkpoints-jscc}"
DATA_DIR="${DATA_DIR:-/workspace/yongjia/datasets/DIV2K}"
SNR_LIST="${SNR_LIST:-3 9}"
EVAL_EVERY="${EVAL_EVERY:-20}"
PYTHON="${PYTHON:-/home/yongjia/.conda/envs/cddm_ddnm/bin/python}"

mkdir -p "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

for SNR in ${SNR_LIST}; do
  TAG="cddm_mse_c4_awgn_snr${SNR}"
  RUN_DIR="${ROOT}/${TAG}"
  mkdir -p "${RUN_DIR}"
  echo "[$(date '+%F %T')] start ${TAG} on GPU${GPU}"
  "${PYTHON}" train/train_cc_siso.py \
    --dataset div2k \
    --data_dir "${DATA_DIR}" \
    --embed_dim 4 \
    --no_vae \
    --lambda_kl 0 \
    --snr_db "${SNR}" \
    --fading awgn \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr 1e-4 \
    --weight_decay 0 \
    --loss_type cddm_mse \
    --amp_dtype bfloat16 \
    --num_workers 8 \
    --val_num_workers 4 \
    --prefetch_factor 4 \
    --eval_every_epochs "${EVAL_EVERY}" \
    --log_freq 50 \
    --save_dir "${RUN_DIR}" \
    --save_name "${TAG}" \
    --log_file "${RUN_DIR}/train.log"
  echo "[$(date '+%F %T')] finished ${TAG}"
done

echo "[$(date '+%F %T')] all CDDM-MSE JSCC C4 AWGN runs finished"
