#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yongjia/paper_code/CDDM"
SCRIPT="$ROOT/MY-V2/jscc-f/explore-2/train_layer2_vq_nested.py"
SAVE_ROOT="$ROOT/MY-V2/jscc-f/explore-2/checkpoints-vq"
LOG_ROOT="$ROOT/MY-V2/jscc-f/explore-2/logs-vq"
mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

GPU="${GPU:-0}"
ARCH="${ARCH:-cnn}"
VQ_FAMILY="${VQ_FAMILY:-image-vq}"
CHANNEL_MODE="${CHANNEL_MODE:-global}"
LATENT_C="${LATENT_C:-256}"
PHASE="${PHASE:-q-pretrain}"
RESUME="${RESUME:?set RESUME to the oracle or preceding receiver checkpoint}"
EPOCHS="${EPOCHS:?set EPOCHS to an absolute epoch greater than the resume epoch}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
SEED="${SEED:-20260624}"

if [[ "$VQ_FAMILY" == "channel-vq" && "$CHANNEL_MODE" == "grouped" ]]; then
  RATES="${RATES:-512,1024,4096}"
  CODEBOOK_INIT="${CODEBOOK_INIT:-channel-balanced}"
else
  RATES="${RATES:-256,1024,4096}"
  CODEBOOK_INIT="${CODEBOOK_INIT:-random}"
fi

VERSION="${VERSION:-${ARCH}-${VQ_FAMILY}-${CHANNEL_MODE}-c${LATENT_C}-rx-${PHASE}}"
EXTRA_ARGS=("$@")

COMMON=(
  --arch "$ARCH"
  --vq-family "$VQ_FAMILY"
  --channel-codebook-mode "$CHANNEL_MODE"
  --latent-c "$LATENT_C"
  --rates "$RATES"
  --rate-weights 1
  --codebook-init "$CODEBOOK_INIT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --val-num-workers "$VAL_NUM_WORKERS"
  --seed "$SEED"
  --version "$VERSION"
  --save-dir "$SAVE_ROOT"
  --resume "$RESUME"
  --freeze-encoder
  --freeze-codebook
  --lambda-vq 0
  --dead-refresh-every 0
  --receiver-only
  --receiver-phase "$PHASE"
  --receiver-stack independent
  --reset-optimizer-on-resume
  --val-every 1
)

case "$PHASE" in
  q-pretrain)
    PHASE_ARGS=(
      --reset-predictor-on-resume
      --lambda-predict-q 1
      --lambda-predict-final 0
      --lambda-predictability 0
      --predictor-lr "${PREDICTOR_LR:-1e-4}"
    )
    ;;
  decoder-warmup)
    anchor="${ANCHOR_WEIGHT:-0.1}"
    decoder_lr="${RECEIVER_DECODER_LR:-2e-5}"
    if [[ "$ARCH" == "swin" ]]; then
      anchor="${ANCHOR_WEIGHT:-0.05}"
      decoder_lr="${RECEIVER_DECODER_LR:-1e-5}"
    fi
    PHASE_ARGS=(
      --lambda-predict-q 0
      --lambda-predict-final 1
      --lambda-predictability 0
      --lambda-zero-anchor "$anchor"
      --lambda-shuffle-anchor "$anchor"
      --lambda-condition-anchor "$anchor"
      --receiver-decoder-lr "$decoder_lr"
    )
    ;;
  joint)
    anchor="${ANCHOR_WEIGHT:-0.1}"
    decoder_lr="${RECEIVER_DECODER_LR:-2e-5}"
    if [[ "$ARCH" == "swin" ]]; then
      anchor="${ANCHOR_WEIGHT:-0.05}"
      decoder_lr="${RECEIVER_DECODER_LR:-1e-5}"
    fi
    PHASE_ARGS=(
      --lambda-predict-q "${LAMBDA_PREDICT_Q:-0.002}"
      --lambda-predict-final 1
      --lambda-predictability 0
      --lambda-zero-anchor "$anchor"
      --lambda-shuffle-anchor "$anchor"
      --lambda-condition-anchor "$anchor"
      --predictor-lr "${PREDICTOR_LR:-5e-5}"
      --receiver-decoder-lr "$decoder_lr"
    )
    ;;
  *)
    echo "PHASE must be q-pretrain, decoder-warmup, or joint" >&2
    exit 2
    ;;
esac

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n cddm_ddnm \
  python "$SCRIPT" \
  "${COMMON[@]}" \
  "${PHASE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG_ROOT/${VERSION}.log"
