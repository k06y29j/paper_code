#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
SNR_DB="${SNR_DB:-6}"
TAG="${TAG:-snr6}"

EPOCHS="${EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-12}"

ROOT="${ROOT:-checkpoints-ar/twofreq_receiver_norm_swin_${TAG}_v1}"
INIT_CKPT="${INIT_CKPT:-${ROOT}/01_receiver_norm_swin_stage01/twofreq_receiver_norm_swin_stage01_best.pth}"
STAGE_DIR="${STAGE_DIR:-${ROOT}/02B_zero_degrade_x0_high_unet_frozen_swin_${TAG}_v1}"

NUM_STEPS="${NUM_STEPS:-200}"
SIGMA_MAX="${SIGMA_MAX:-0.10}"
ENDPOINT_ZERO_PROB="${ENDPOINT_ZERO_PROB:-0.25}"
CLIP_X0="${CLIP_X0:-3}"

ZERO_EVAL_TIMESTEPS="${ZERO_EVAL_TIMESTEPS:-50,100,199}"
REFINE_T_STARTS="${REFINE_T_STARTS:-50,100,199}"
REFINE_STEPS="${REFINE_STEPS:-20}"
DIAG_TIMESTEPS="${DIAG_TIMESTEPS:-25,50,100,199}"
EVAL_EVERY="${EVAL_EVERY:-5}"

mkdir -p "${STAGE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"

exec conda run --no-capture-output -n cddm_ddnm \
  python train/train_receiver_norm_zero_degrade_high.py \
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
    --num_steps "${NUM_STEPS}" \
    --sigma_max "${SIGMA_MAX}" \
    --endpoint_zero_prob "${ENDPOINT_ZERO_PROB}" \
    --clip_x0 "${CLIP_X0}" \
    --unet_base 128 \
    --unet_depth 3 \
    --time_dim 256 \
    --zero_eval_timesteps "${ZERO_EVAL_TIMESTEPS}" \
    --refine_t_starts "${REFINE_T_STARTS}" \
    --refine_steps "${REFINE_STEPS}" \
    --diag_timesteps "${DIAG_TIMESTEPS}" \
    --eval_every_epochs "${EVAL_EVERY}" \
    --amp_dtype bfloat16 \
    --save_dir "${STAGE_DIR}" \
    --log_file "${STAGE_DIR}/train.log"
