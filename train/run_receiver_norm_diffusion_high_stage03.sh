#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
SNR_DB="${SNR_DB:-6}"
TAG="${TAG:-snr6}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-12}"
ROOT="${ROOT:-checkpoints-ar/hier_receiver_norm_unified_scale_swin_${TAG}_v1}"
INIT_CKPT="${INIT_CKPT:-${ROOT}/01_receiver_norm_swin_pretrain/hierarchical_receiver_norm_swin_pretrain_best.pth}"
STAGE_DIR="${STAGE_DIR:-${ROOT}/03A_diffusion_high_unet_frozen_swin_${TAG}_v1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-1000}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-linear}"
LAMBDA_X0="${LAMBDA_X0:-0.05}"
CLIP_X0="${CLIP_X0:-0}"
DDIM_STEPS_A="${DDIM_STEPS_A:-10}"
DDIM_STEPS_B="${DDIM_STEPS_B:-20}"
EVAL_EVERY="${EVAL_EVERY:-5}"
DIAG_TIMESTEPS="${DIAG_TIMESTEPS:-50,100,200}"

mkdir -p "${STAGE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
exec conda run --no-capture-output -n cddm_ddnm \
  python train/train_receiver_norm_diffusion_high.py \
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
    --weight_decay 0 \
    --diffusion_steps "${DIFFUSION_STEPS}" \
    --noise_schedule "${NOISE_SCHEDULE}" \
    --lambda_x0 "${LAMBDA_X0}" \
    --clip_x0 "${CLIP_X0}" \
    --unet_base 128 \
    --unet_depth 3 \
    --time_dim 256 \
    --ddim_steps_a "${DDIM_STEPS_A}" \
    --ddim_steps_b "${DDIM_STEPS_B}" \
    --sample_init noise \
    --diag_timesteps "${DIAG_TIMESTEPS}" \
    --eval_every_epochs "${EVAL_EVERY}" \
    --amp_dtype bfloat16 \
    --save_dir "${STAGE_DIR}" \
    --log_file "${STAGE_DIR}/train.log"
