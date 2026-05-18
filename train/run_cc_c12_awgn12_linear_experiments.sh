#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
PY=(conda run -n cddm_ddnm python)

COMMON_TRAIN=(
  train/train_cc_aware.py
  --out_channels 4
  --linear_depth 1
  --linear_hidden_channels 16
  --train_snr_mode fixed
  --train_snr_db 12
  --train_fading awgn
  --eval_snr_grid 0 3 6 9 12 15
  --batch_size 16
  --epochs 200
  --lr 1e-3
  --num_workers 8
  --val_num_workers 4
  --eval_every_epochs 10
)

COMMON_EVAL=(
  test/eval_all.py
  --data_dir /workspace/yongjia/datasets/DIV2K
  --sc_encoder_ckpt checkpoints-val/sc/sc_encoder_div2k_c16.pth
  --sc_decoder_ckpt checkpoints-val/sc/sc_decoder_div2k_c16.pth
  --unet_ckpt checkpoints-val/unet_un/unet_un_div2k_c16_best.pth
  --compression_ratios 0.25
  --fadings awgn
  --snrs 0 3 6 9 12 15
  --num_sample_steps 30
  --ddnm_t_start 100
  --ddnm_anchor zcd
  --ddnm_repeat_per_step 3
  --batch_size 4
  --max_batches 0
)

run_one() {
  local name="$1"
  shift
  local save_dir="checkpoints-val/cc_2/${name}"
  local log_file="log/cc_2/${name}.txt"

  echo "== Train ${name} on GPU ${GPU} =="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY[@]}" "${COMMON_TRAIN[@]}" \
    "$@" \
    --save_dir "${save_dir}" \
    --log_file "${log_file}"

  echo "== Eval ${name} =="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY[@]}" "${COMMON_EVAL[@]}" \
    --cc_dir "${save_dir}"
}

run_one "aware_awgn_snr12_c4_L1_pinv" \
  --codec_init pinv \
  --lambda_enc_orth 0

run_one "aware_awgn_snr12_c4_L1_lmmse" \
  --codec_init lmmse \
  --lambda_enc_orth 0 \
  --cov_max_batches 0

run_one "aware_awgn_snr12_c4_L1_pca_lmmse" \
  --codec_init pca_lmmse \
  --lambda_enc_orth 0 \
  --cov_max_batches 0
