#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
SNR_DB="${SNR_DB:-6}"
TAG="${TAG:-snr${SNR_DB}}"
BATCH_SIZE="${BATCH_SIZE:-12}"
EPOCHS="${EPOCHS:-600}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-20}"
ROOT="${ROOT:-checkpoints-ar/twofreq_receiver_norm_swin_${TAG}_v1}"
STAGE_DIR="${ROOT}/01_receiver_norm_swin_stage01"

mkdir -p "${STAGE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
exec conda run --no-capture-output -n cddm_ddnm \
  python train/train_hierarchical_receiver_norm_swin_pretrain.py \
    --data_dir /workspace/yongjia/datasets/DIV2K \
    --init_sc_encoder_ckpt checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_encoder_div2k_c16.pth \
    --init_sc_decoder_ckpt checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_decoder_div2k_c16.pth \
    --snr_db "${SNR_DB}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 8 \
    --val_num_workers 4 \
    --cache_workers 12 \
    --prefetch_factor 4 \
    --epochs "${EPOCHS}" \
    --lr 5e-5 \
    --decoder_lr 5e-5 \
    --lambda_oracle 1.0 \
    --lambda_base 0.4 \
    --blur_kernel 7 \
    --blur_sigma 1.0 \
    --eval_every_epochs "${EVAL_EVERY_EPOCHS}" \
    --amp_dtype bfloat16 \
    --save_dir "${STAGE_DIR}" \
    --log_file "${STAGE_DIR}/train.log" \
    --no_encoder_vae \
    --lambda_kl 0
