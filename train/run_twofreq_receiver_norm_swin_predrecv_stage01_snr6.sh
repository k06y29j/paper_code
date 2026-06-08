#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
SNR_DB="${SNR_DB:-9}"
TAG="${TAG:-snr${SNR_DB}}"
BATCH_SIZE="${BATCH_SIZE:-12}"
EPOCHS="${EPOCHS:-400}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-10}"

LR="${LR:-2e-5}"
DECODER_LR="${DECODER_LR:-2e-5}"
PREDICTOR_LR="${PREDICTOR_LR:-1e-4}"
LAMBDA_RECV_PRED="${LAMBDA_RECV_PRED:-1.0}"
LAMBDA_BASE="${LAMBDA_BASE:-0.2}"
LAMBDA_ORACLE="${LAMBDA_ORACLE:-0.03}"
BASELINE_PSNR="${BASELINE_PSNR:-21.0085}"

ROOT="${ROOT:-checkpoints-ar/twofreq_receiver_norm_swin_${TAG}_v1}"
INIT_HIER_CKPT="${INIT_HIER_CKPT:-${ROOT}/01_receiver_norm_swin_stage01/twofreq_receiver_norm_swin_stage01_best.pth}"
STAGE_DIR="${STAGE_DIR:-${ROOT}/01E_receiver_norm_swin_predrecv_stage01}"

mkdir -p "${STAGE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"

args=(
  train/train_twofreq_receiver_norm_swin_predrecv_stage01.py
  --data_dir /workspace/yongjia/datasets/DIV2K
  --init_sc_encoder_ckpt checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_encoder_div2k_c16.pth
  --init_sc_decoder_ckpt checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_decoder_div2k_c16.pth
  --snr_db "${SNR_DB}"
  --baseline_psnr "${BASELINE_PSNR}"
  --batch_size "${BATCH_SIZE}"
  --num_workers 8
  --val_num_workers 4
  --cache_workers 12
  --prefetch_factor 4
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --decoder_lr "${DECODER_LR}"
  --predictor_lr "${PREDICTOR_LR}"
  --lambda_recv_pred "${LAMBDA_RECV_PRED}"
  --lambda_base "${LAMBDA_BASE}"
  --lambda_oracle "${LAMBDA_ORACLE}"
  --blur_kernel 7
  --blur_sigma 1.0
  --eval_every_epochs "${EVAL_EVERY_EPOCHS}"
  --amp_dtype bfloat16
  --save_dir "${STAGE_DIR}"
  --log_file "${STAGE_DIR}/train.log"
  --no_encoder_vae
  --lambda_kl 0
)

if [[ -n "${INIT_HIER_CKPT}" && -f "${INIT_HIER_CKPT}" ]]; then
  args+=(--init_hier_ckpt "${INIT_HIER_CKPT}")
else
  echo "warning: INIT_HIER_CKPT not found, falling back to SC encoder/decoder init: ${INIT_HIER_CKPT}" >&2
fi

exec python "${args[@]}"
