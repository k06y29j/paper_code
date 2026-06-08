#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
SNR_DB="${SNR_DB:-6}"
TAG="${TAG:-snr6}"
EPOCHS="${EPOCHS:-160}"
BATCH_SIZE="${BATCH_SIZE:-12}"
ROOT="${ROOT:-checkpoints-ar/hier_receiver_norm_unified_scale_swin_${TAG}_v1}"
INIT_CKPT="${INIT_CKPT:-${ROOT}/01_receiver_norm_swin_pretrain/hierarchical_receiver_norm_swin_pretrain_best.pth}"
STAGE_DIR="${STAGE_DIR:-${ROOT}/02_receiver_norm_ar_h256d6_frozen_swin_${TAG}}"

mkdir -p "${STAGE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
exec conda run --no-capture-output -n cddm_ddnm \
  python train/train_hierarchical_swin_ar_awgn12.py \
    --data_dir /workspace/yongjia/datasets/DIV2K \
    --stage receiver_norm_ar_frozen_swin \
    --init_hier_ckpt "${INIT_CKPT}" \
    --skip_init_ar \
    --snr_db "${SNR_DB}" \
    --baseline_psnr nan \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 8 \
    --val_num_workers 4 \
    --cache_workers 12 \
    --prefetch_factor 4 \
    --epochs "${EPOCHS}" \
    --lr 5e-5 \
    --decoder_lr 0 \
    --ar_lr 1e-4 \
    --ar_arch simple \
    --ar_hidden 256 \
    --ar_depth 6 \
    --teacher_force_epochs 20 \
    --teacher_decay_epochs 60 \
    --eval_every_epochs 2 \
    --amp_dtype bfloat16 \
    --save_dir "${STAGE_DIR}" \
    --log_file "${STAGE_DIR}/train.log" \
    --no_encoder_vae \
    --lambda_kl 0 \
    --no_ar_scale \
    --receiver_observation norm \
    --lambda_recv 1.0 \
    --lambda_base 0 \
    --lambda_high_norm 0.05 \
    --high_gate_mode alpha \
    --high_gate_warmup_epochs 20
