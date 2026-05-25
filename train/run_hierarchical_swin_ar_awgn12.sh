#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${1:-0}"
if [[ $# -gt 0 ]]; then shift; fi
RUN_DIR="${1:-checkpoints-ar/hierarchical_swin_ar_awgn12}"
if [[ $# -gt 0 ]]; then shift; fi

mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_ARGS=()
if [[ -n "${INIT_HIER_CKPT:-}" ]]; then
  EXTRA_ARGS+=(--init_hier_ckpt "${INIT_HIER_CKPT}")
fi

exec conda run --no-capture-output -n cddm_ddnm python train/train_hierarchical_swin_ar_awgn12.py \
  --data_dir /workspace/yongjia/datasets/DIV2K \
  --init_sc_encoder_ckpt checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_encoder_div2k_c16.pth \
  --init_sc_decoder_ckpt checkpoints-val-v2/route_a/sc_dct_c4_awgn12/sc_decoder_div2k_c16.pth \
  --stage "${STAGE:-joint}" \
  "${EXTRA_ARGS[@]}" \
  --snr_db 12 \
  --baseline_psnr 22.419 \
  --batch_size "${BATCH_SIZE:-8}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --val_num_workers "${VAL_NUM_WORKERS:-4}" \
  --cache_workers "${CACHE_WORKERS:-12}" \
  --prefetch_factor "${PREFETCH_FACTOR:-4}" \
  --epochs "${EPOCHS:-400}" \
  --lr "${LR:-5e-5}" \
  --decoder_lr "${DECODER_LR:-0}" \
  --ar_lr "${AR_LR:-0}" \
  --ar_lr_mult "${AR_LR_MULT:-2.0}" \
  --ar_arch "${AR_ARCH:-simple}" \
  --ar_hidden "${AR_HIDDEN:-160}" \
  --ar_depth "${AR_DEPTH:-4}" \
  --teacher_force_epochs "${TEACHER_FORCE_EPOCHS:-50}" \
  --teacher_decay_epochs "${TEACHER_DECAY_EPOCHS:-100}" \
  --eval_every_epochs "${EVAL_EVERY_EPOCHS:-2}" \
  --amp_dtype bfloat16 \
  --save_dir "${RUN_DIR}" \
  --log_file "${RUN_DIR}/train.log" \
  "$@"
