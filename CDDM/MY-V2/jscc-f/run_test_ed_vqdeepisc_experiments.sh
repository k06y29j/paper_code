#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-1}"
DATA_DIR="${DATA_DIR:-/workspace/yongjia/datasets/DIV2K}"
SAVE_DIR="${SAVE_DIR:-MY-V2/jscc-f/checkpoints/test_ed_vqdeepisc}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-24}"
TEST_BATCH="${TEST_BATCH:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-8}"
LR="${LR:-2e-4}"
LR_CODEBOOK="${LR_CODEBOOK:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_MIN="${LR_MIN:-1e-6}"
LR_SCHEDULER_T_MAX="${LR_SCHEDULER_T_MAX:-300}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5}"
LAMBDA_VQ="${LAMBDA_VQ:-1.0}"
LAMBDA_SSIM="${LAMBDA_SSIM:-0.0}"
LAMBDA_GAN="${LAMBDA_GAN:-0.0}"
GAN_START_EPOCH="${GAN_START_EPOCH:-1}"
GAN_LOSS="${GAN_LOSS:-hinge}"
DISC_FACTOR="${DISC_FACTOR:-1.0}"
DISC_LAYERS="${DISC_LAYERS:-3}"
DISC_NDF="${DISC_NDF:-64}"
DISC_LR="${DISC_LR:-1e-4}"
DISC_WEIGHT_DECAY="${DISC_WEIGHT_DECAY:-1e-4}"
IMAGE_K="${IMAGE_K:-1024}"
BETA_COMMIT="${BETA_COMMIT:-0.25}"
BETA_KLD="${BETA_KLD:-0.05}"
USAGE_KLD_TAU="${USAGE_KLD_TAU:-1.0}"
VQ_EMA_DECAY="${VQ_EMA_DECAY:-0.99}"
VQ_EMA_EPS="${VQ_EMA_EPS:-1e-5}"
VAL_EVERY="${VAL_EVERY:-5}"
LATEST_EVERY="${LATEST_EVERY:-10}"
SEED="${SEED:-20260702}"

export CUDA_VISIBLE_DEVICES="$GPU"

if [[ -n "${EXPERIMENTS:-}" ]]; then
  read -r -a EXP_LIST <<< "$EXPERIMENTS"
else
  EXP_LIST=(
    exp13_swin_image_ema
    exp14_swin320_image_ema
    exp15_swin_image_ema_init_layer1
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
    --weight-decay "$WEIGHT_DECAY" \
    --lr-scheduler "$LR_SCHEDULER" \
    --lr-min "$LR_MIN" \
    --lr-scheduler-t-max "$LR_SCHEDULER_T_MAX" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
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
    --image-k "$IMAGE_K" \
    --beta-commit "$BETA_COMMIT" \
    --beta-kld "$BETA_KLD" \
    --usage-kld-tau "$USAGE_KLD_TAU" \
    --vq-ema-decay "$VQ_EMA_DECAY" \
    --vq-ema-eps "$VQ_EMA_EPS" \
    --val-every "$VAL_EVERY" \
    --latest-every "$LATEST_EVERY" \
    --seed "$SEED" \
    "$@"
done
