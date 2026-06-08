#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
WAIT_PID="${WAIT_PID:-}"
EPOCHS="${EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ROOT="${ROOT:-checkpoints-jscc}"
DATA_DIR="${DATA_DIR:-/workspace/yongjia/datasets/DIV2K}"
SNR_LIST="${SNR_LIST:-6 9}"
EVAL_EVERY="${EVAL_EVERY:-20}"

mkdir -p "${ROOT}"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[$(date '+%F %T')] waiting for PID ${WAIT_PID} before using GPU${GPU}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date '+%F %T')] wait PID ${WAIT_PID} finished"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

for SNR in ${SNR_LIST}; do
  TAG="swin_no_vae_c4_awgn_snr${SNR}"
  RUN_DIR="${ROOT}/${TAG}"
  mkdir -p "${RUN_DIR}"
  echo "[$(date '+%F %T')] start ${TAG} on GPU${GPU}"
  conda run --no-capture-output -n cddm_ddnm \
    python train/train_cc_siso.py \
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
      --loss_type smooth_l1 \
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

echo "[$(date '+%F %T')] all JSCC C4 AWGN runs finished"
