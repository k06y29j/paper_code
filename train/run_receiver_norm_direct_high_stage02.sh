#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SNR_DB="${SNR_DB:-6}"
TAG="${TAG:-snr6}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-12}"
ROOT="${ROOT:-checkpoints-ar/hier_receiver_norm_unified_scale_swin_${TAG}_v1}"
INIT_CKPT="${INIT_CKPT:-${ROOT}/01_receiver_norm_swin_pretrain/hierarchical_receiver_norm_swin_pretrain_best.pth}"
STAGE_DIR="${STAGE_DIR:-${ROOT}/02A_direct_high_unet_frozen_swin_${TAG}}"

mkdir -p "${STAGE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
exec conda run --no-capture-output -n cddm_ddnm \
  python train/train_receiver_norm_direct_high.py \
    --data_dir /workspace/yongjia/datasets/DIV2K \
    --init_hier_ckpt "${INIT_CKPT}" \
    --snr_db "${SNR_DB}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 8 \
    --val_num_workers 4 \
    --cache_workers 12 \
    --prefetch_factor 4 \
    --epochs "${EPOCHS}" \
    --lr 1e-4 \
    --predictor_base 128 \
    --predictor_depth 3 \
    --lambda_high 0.03 \
    --eval_every_epochs 2 \
    --amp_dtype bfloat16 \
    --save_dir "${STAGE_DIR}" \
    --log_file "${STAGE_DIR}/train.log"
