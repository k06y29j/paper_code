#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${1:-0}"
if [[ $# -gt 0 ]]; then shift; fi
RUN_DIR="${1:-checkpoints-codex/fixed_direct_awgn12}"
if [[ $# -gt 0 ]]; then shift; fi

mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec conda run --no-capture-output -n cddm_ddnm python train/train_codex_fixed_highfreq.py \
  --route direct \
  --data_dir /workspace/yongjia/datasets/DIV2K \
  --sc_encoder_ckpt checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_encoder_div2k_c16.pth \
  --sc_decoder_ckpt checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_decoder_div2k_c16.pth \
  --cc_dir checkpoints-val-v2/route_a/cc_dct_c4 \
  --unet_ckpt checkpoints-val-v2/route_a/unet_un/unet_un_div2k_c16_best.pth \
  --compression_ratio 0.25 \
  --snr_db 12 \
  --baseline_psnr 22.419 \
  --batch_size 32 \
  --num_workers 8 \
  --val_num_workers 4 \
  --cache_workers 12 \
  --prefetch_factor 4 \
  --hidden 224 \
  --depth 12 \
  --epochs 400 \
  --lr 2e-4 \
  --eval_every_epochs 2 \
  --amp_dtype bfloat16 \
  --save_dir "${RUN_DIR}" \
  --log_file "${RUN_DIR}/train.log" \
  "$@"
