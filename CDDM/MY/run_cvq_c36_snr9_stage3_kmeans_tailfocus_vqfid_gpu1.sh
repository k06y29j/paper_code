#!/usr/bin/env bash
set -euo pipefail

# Ordinary VQ fine-tune from the existing kmeans Stage 3 best.
# Keeps K_A/K_B unchanged, but puts more pressure on tail quantization fidelity.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export SAVE_DIR="${SAVE_DIR:-MY/checkpoints-cvq-kmeans-tailfocus-finetune-vqfid-k512}"
export STAGE3_EPOCHS="${STAGE3_EPOCHS:-160}"
export LR="${LR:-2e-5}"
export DECODER_LR="${DECODER_LR:-2e-5}"
export LAMBDA_VQ_START="${LAMBDA_VQ_START:-0.003}"
export LAMBDA_VQ_END="${LAMBDA_VQ_END:-0.008}"
export VQ_RAMP_EPOCHS="${VQ_RAMP_EPOCHS:-40}"
export LAMBDA_USAGE="${LAMBDA_USAGE:-0.0001}"
export LAMBDA_CONT_STAGE3="${LAMBDA_CONT_STAGE3:-0.02}"
export LAMBDA_TAIL_DISTILL="${LAMBDA_TAIL_DISTILL:-0.01}"
export LAMBDA_GAIN_STAGE3="${LAMBDA_GAIN_STAGE3:-0.2}"
export LAMBDA_PREFIX_FLOOR_STAGE3="${LAMBDA_PREFIX_FLOOR_STAGE3:-0.0}"
export STAGE3_MIN_TAIL_GAIN_VQ="${STAGE3_MIN_TAIL_GAIN_VQ:-0.05}"
export STAGE3_MIN_TAIL_GAIN_CONT="${STAGE3_MIN_TAIL_GAIN_CONT:-0.5}"
export STAGE3_MIN_PSNR_PREFIX="${STAGE3_MIN_PSNR_PREFIX:-25.2}"

MY/run_cvq_c36_snr9_stage3_kmeans_tailfocus_finetune_gpu1.sh
