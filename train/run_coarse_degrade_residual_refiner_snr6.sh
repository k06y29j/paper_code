#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SNR_DB="${SNR_DB:-6}"
TAG="${TAG:-snr${SNR_DB}}"
BATCH_SIZE="${BATCH_SIZE:-12}"
EPOCHS="${EPOCHS:-600}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-5}"

NUM_STEPS="${NUM_STEPS:-200}"
SIGMA_MAX="${SIGMA_MAX:-0.05}"
LAMBDA_RES="${LAMBDA_RES:-0.03}"
CLIP_R="${CLIP_R:-2.0}"
LR="${LR:-1e-4}"

ROOT="${ROOT:-checkpoints-ar/twofreq_receiver_norm_swin_${TAG}_v1}"
INIT_STAGE01B_CKPT="${INIT_STAGE01B_CKPT:-${ROOT}/01B_receiver_norm_swin_predrecv_stage01/twofreq_receiver_norm_swin_predrecv_stage01_best.pth}"
STAGE_DIR="${STAGE_DIR:-${ROOT}/02C_coarse_degrade_residual_refiner_frozen_swin_${TAG}_v1}"

mkdir -p "${STAGE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"

exec conda run --no-capture-output -n cddm_ddnm \
  python train/train_coarse_degrade_residual_refiner.py \
    --data_dir /workspace/yongjia/datasets/DIV2K \
    --init_stage01b_ckpt "${INIT_STAGE01B_CKPT}" \
    --snr_db "${SNR_DB}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 8 \
    --val_num_workers 4 \
    --cache_workers 12 \
    --prefetch_factor 4 \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight_decay 0 \
    --num_steps "${NUM_STEPS}" \
    --sigma_max "${SIGMA_MAX}" \
    --lambda_res "${LAMBDA_RES}" \
    --clip_r "${CLIP_R}" \
    --unet_base 128 \
    --unet_depth 3 \
    --time_dim 256 \
    --eval_every_epochs "${EVAL_EVERY_EPOCHS}" \
    --amp_dtype bfloat16 \
    --save_dir "${STAGE_DIR}" \
    --log_file "${STAGE_DIR}/train.log" \
    --no_encoder_vae \
    --lambda_kl 0
