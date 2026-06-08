#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-conda run --no-capture-output -n cddm_ddnm python}"

SAVE_DIR="${SAVE_DIR:-MY/checkpoints-cvq-kmeans-tailfocus-finetune-vq2e3-k512}"
INIT_CKPT="${INIT_CKPT:-MY/checkpoints-cvq-e2e-kmeans-vq5e3-k512/cvq_c36_snr9_stage3_best.pth}"
TEACHER_CKPT="${TEACHER_CKPT:-MY/checkpoints-cvq/cvq_c36_snr9_stage1_best.pth}"

mkdir -p "$SAVE_DIR"

{
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "SAVE_DIR=$SAVE_DIR"
  echo "INIT_CKPT=$INIT_CKPT"
  echo "strategy=kmeans_tailfocus_finetune"
  echo "cvq_mode=single K_A=512 K_B=256"
  echo "freeze=encoder_body_plus_prefix_head_gradient_mask"
  echo "objective=raise_psnr_vq_with_fixed_receiver_prediction_classes"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
} > "$SAVE_DIR/launch.meta"

$PY MY/train_cvq/main.py \
  --stage 3 \
  --data-dir /workspace/yongjia/datasets/DIV2K \
  --save-dir "$SAVE_DIR" \
  --init-ckpt "$INIT_CKPT" \
  --init-jscc-encoder MY/checkpoints-jscc/encoder_snr9_channel_awgn_C36.pt \
  --init-jscc-decoder MY/checkpoints-jscc/decoder_snr9_channel_awgn_C36.pt \
  --snr-db 9 \
  --latent-ch 36 \
  --prefix-ch 16 \
  --latent-h 16 \
  --latent-w 16 \
  --cvq-mode single \
  --k-a 512 \
  --k-b 256 \
  --epochs "${STAGE3_EPOCHS:-140}" \
  --batch-size 8 \
  --test-batch 1 \
  --num-workers 16 \
  --val-num-workers 8 \
  --val-every 5 \
  --latest-every 10 \
  --seed 20260605 \
  --lr "${LR:-2e-5}" \
  --decoder-lr "${DECODER_LR:-2e-5}" \
  --stage3-train-encoder \
  --stage3-train-decoder \
  --stage3-freeze-encoder-body \
  --stage3-freeze-prefix-head \
  --freeze-encoder-first 0 \
  --lambda-prefix-stage3 0.0 \
  --lambda-nested-stage3 0.0 \
  --lambda-cont-stage3 "${LAMBDA_CONT_STAGE3:-0.005}" \
  --lambda-tail-distill "${LAMBDA_TAIL_DISTILL:-0.005}" \
  --lambda-vq-start "${LAMBDA_VQ_START:-0.0005}" \
  --lambda-vq-end "${LAMBDA_VQ_END:-0.002}" \
  --vq-ramp-epochs "${VQ_RAMP_EPOCHS:-40}" \
  --lambda-usage "${LAMBDA_USAGE:-0.00005}" \
  --lambda-gain-stage3 "${LAMBDA_GAIN_STAGE3:-0.5}" \
  --stage3-gain-margin-db "${STAGE3_GAIN_MARGIN_DB:-0.2}" \
  --lambda-prefix-floor-stage3 "${LAMBDA_PREFIX_FLOOR_STAGE3:-10.0}" \
  --stage3-prefix-floor-db "${STAGE3_PREFIX_FLOOR_DB:-25.2}" \
  --stage3-teacher-ckpt "$TEACHER_CKPT" \
  --stage3-score psnr_vq \
  --stage3-min-psnr-prefix "${STAGE3_MIN_PSNR_PREFIX:-25.2}" \
  --stage3-min-tail-gain-vq "${STAGE3_MIN_TAIL_GAIN_VQ:-0.05}" \
  --stage3-min-tail-gain-cont "${STAGE3_MIN_TAIL_GAIN_CONT:-0.5}" \
  --stage3-min-used-a "${STAGE3_MIN_USED_A:-50}" \
  --stage3-min-used-b "${STAGE3_MIN_USED_B:-20}" \
  --stage3-min-perplexity-a "${STAGE3_MIN_PERPLEXITY_A:-20}" \
  --stage3-min-perplexity-b "${STAGE3_MIN_PERPLEXITY_B:-8}" \
  --dead-code-restart-every 0 \
  2>&1 | tee "$SAVE_DIR/launch.console.log"
