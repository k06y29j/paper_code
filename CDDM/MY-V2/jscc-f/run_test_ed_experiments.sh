#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-1}"
DATA_DIR="${DATA_DIR:-/workspace/yongjia/datasets/DIV2K}"
SAVE_DIR="${SAVE_DIR:-MY-V2/jscc-f/checkpoints/test_ed}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-24}"
TEST_BATCH="${TEST_BATCH:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-8}"
LR="${LR:-1e-4}"
LR_CODEBOOK="${LR_CODEBOOK:-1e-4}"
LAMBDA_VQ="${LAMBDA_VQ:-0.1}"
LAMBDA_SSIM="${LAMBDA_SSIM:-1.0}"
LAMBDA_GAN="${LAMBDA_GAN:-0.0}"
GAN_START_EPOCH="${GAN_START_EPOCH:-1}"
GAN_LOSS="${GAN_LOSS:-hinge}"
DISC_FACTOR="${DISC_FACTOR:-1.0}"
DISC_LAYERS="${DISC_LAYERS:-3}"
DISC_NDF="${DISC_NDF:-64}"
DISC_LR="${DISC_LR:-1e-4}"
DISC_WEIGHT_DECAY="${DISC_WEIGHT_DECAY:-1e-4}"
BETA_COMMIT="${BETA_COMMIT:-0.25}"
VAL_EVERY="${VAL_EVERY:-5}"
LATEST_EVERY="${LATEST_EVERY:-10}"
SEED="${SEED:-20260701}"

export CUDA_VISIBLE_DEVICES="$GPU"

if [[ -n "${EXPERIMENTS:-}" ]]; then
  read -r -a EXP_LIST <<< "$EXPERIMENTS"
else
  EXP_LIST=(
    exp01_swin_channel
    exp02_swin_image
    exp03_cnn_channel
    exp04_cnn_image
    exp05_cnn_noquant
    exp06_swin_channel_simvq
    exp07_swin_image_simvq
    exp08_cnn_channel_simvq
    exp09_cnn_image_simvq
    exp10_swin320_channel_simvq
    exp11_swin320_image_simvq
    exp12_swin320_channel
  )
fi

for EXP in "${EXP_LIST[@]}"; do
  echo "=== running ${EXP} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
  conda run -n cddm_ddnm python MY-V2/jscc-f/test_ed.py \
    --experiment "$EXP" \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --test-batch "$TEST_BATCH" \
    --num-workers "$NUM_WORKERS" \
    --val-num-workers "$VAL_NUM_WORKERS" \
    --lr "$LR" \
    --lr-codebook "$LR_CODEBOOK" \
    --lambda-vq "$LAMBDA_VQ" \
    --lambda-ssim "$LAMBDA_SSIM" \
    --lambda-gan "$LAMBDA_GAN" \
    --gan-start-epoch "$GAN_START_EPOCH" \
    --gan-loss "$GAN_LOSS" \
    --disc-factor "$DISC_FACTOR" \
    --disc-layers "$DISC_LAYERS" \
    --disc-ndf "$DISC_NDF" \
    --disc-lr "$DISC_LR" \
    --disc-weight-decay "$DISC_WEIGHT_DECAY" \
    --beta-commit "$BETA_COMMIT" \
    --val-every "$VAL_EVERY" \
    --latest-every "$LATEST_EVERY" \
    --seed "$SEED" \
    "$@"
done
