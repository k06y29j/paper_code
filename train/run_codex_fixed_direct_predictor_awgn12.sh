#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${1:-0}"
if [[ $# -gt 0 ]]; then shift; fi
PREDICTOR="${1:-resunet}"
if [[ $# -gt 0 ]]; then shift; fi
RUN_DIR="${1:-checkpoints-codex/fixed_direct_${PREDICTOR}_awgn12}"
if [[ $# -gt 0 ]]; then shift; fi

mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec conda run --no-capture-output -n cddm_ddnm python train/train_codex_fixed_highfreq.py \
  --route direct \
  --predictor "${PREDICTOR}" \
  --direct_condition "${DIRECT_CONDITION:-compact}" \
  --data_dir /workspace/yongjia/datasets/DIV2K \
  --sc_encoder_ckpt checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_encoder_div2k_c16.pth \
  --sc_decoder_ckpt checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_decoder_div2k_c16.pth \
  --cc_dir checkpoints-val-v2/route_a/cc_dct_c4 \
  --unet_ckpt checkpoints-val-v2/route_a/unet_un/unet_un_div2k_c16_best.pth \
  --compression_ratio 0.25 \
  --snr_db 12 \
  --baseline_psnr 22.419 \
  --batch_size "${BATCH_SIZE:-32}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --val_num_workers "${VAL_NUM_WORKERS:-4}" \
  --cache_workers "${CACHE_WORKERS:-12}" \
  --prefetch_factor "${PREFETCH_FACTOR:-4}" \
  --hidden "${HIDDEN:-192}" \
  --depth "${DEPTH:-8}" \
  --lambda_res "${LAMBDA_RES:-0.1}" \
  --img_mse_weight "${IMG_MSE_WEIGHT:-0.8}" \
  --img_charb_weight "${IMG_CHARB_WEIGHT:-0.2}" \
  --epochs "${EPOCHS:-400}" \
  --lr "${LR:-2e-4}" \
  --eval_every_epochs "${EVAL_EVERY_EPOCHS:-2}" \
  --amp_dtype bfloat16 \
  --save_dir "${RUN_DIR}" \
  --log_file "${RUN_DIR}/train.log" \
  "$@"
