#!/usr/bin/env bash
# Paper 2605.26089v2 CVQ capacity sweep.
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_cvq_spatial.sh cnn 4 cvq-cnn-f16-d256-v1
set -euo pipefail

ARCH=${1:?usage: $0 <cnn|swin> <e2-downsamples> <version> [extra args...]}
DOWNSAMPLES=${2:?usage: $0 <cnn|swin> <e2-downsamples> <version> [extra args...]}
VERSION=${3:?usage: $0 <cnn|swin> <e2-downsamples> <version> [extra args...]}
shift 3

# The active 16x16 configuration reserves 16 Layer-1 z1 channels and trains
# 240 quantized q2 channels, i.e. the new D2 front end is [16 + 240] = 256.
# Environment overrides keep this launcher usable for later C/D ablations.
LATENT_C=${LATENT_C:-240}
EMBEDDING_DIM=${EMBEDDING_DIM:-0}
BATCH_SIZE=${BATCH_SIZE:-4}
TEST_BATCH=${TEST_BATCH:-${BATCH_SIZE}}
# The CVQ paper's K=16,384 tokenizer uses batch 256.  On a single GPU,
# ACCUM_STEPS reproduces its codebook-update token statistics without
# changing the reconstruction or commitment objective.
ACCUM_STEPS=${ACCUM_STEPS:-1}
EPOCHS=${EPOCHS:-200}
CONTINUOUS_WARMUP_EPOCHS=${CONTINUOUS_WARMUP_EPOCHS:-10}
# For a high-fidelity Layer1, the residual target is much smaller.  This is
# a representation scale (not an added loss) and can be raised explicitly.
INITIAL_RESIDUAL_GAIN=${INITIAL_RESIDUAL_GAIN:-0.01}
SOURCE_E2_INIT=${SOURCE_E2_INIT:-0}
CHANNEL_RMS_NORMALIZE=${CHANNEL_RMS_NORMALIZE:-0}
# Paper 2605.26089v2 uses K=16,384 by default; the earlier K=1,024 run is
# retained as the low-capacity ablation, not the main tokenizer setting.
RATES=${RATES:-16384}
DROPOUT_ALPHA=${DROPOUT_ALPHA:-0.5}
LAMBDA_DIRECT_GAIN=${LAMBDA_DIRECT_GAIN:-0}
DIRECT_GAIN_DB=${DIRECT_GAIN_DB:-0}
QUERY_CHUNK_SIZE=${QUERY_CHUNK_SIZE:-1024}
CODEBOOK_CHUNK_SIZE=${CODEBOOK_CHUNK_SIZE:-4096}
# D2 has a fixed 320-channel source contract.  Keep the first 16 channels as
# the Layer-1 z1 base and quantize/ablate only the remaining q2 branch.
D2_Z1_CHANNELS=${D2_Z1_CHANNELS:-16}

ROOT=/workspace/yongjia/paper_code/CDDM
EXPLORE4=${ROOT}/MY-V2/jscc-f/explore-4
mkdir -p "${EXPLORE4}/logs-cvq" "${EXPLORE4}/checkpoints-cvq" "${EXPLORE4}/results-cvq"
SOURCE_E2_INIT_ARG=()
if [[ "${SOURCE_E2_INIT}" == "1" ]]; then
  SOURCE_E2_INIT_ARG=(--source-e2-init)
fi
CHANNEL_RMS_NORMALIZE_ARG=()
if [[ "${CHANNEL_RMS_NORMALIZE}" == "1" ]]; then
  CHANNEL_RMS_NORMALIZE_ARG=(--channel-rms-normalize)
fi

exec conda run --no-capture-output -n cddm_ddnm python "${EXPLORE4}/train_cvq_spatial.py" \
  --arch "${ARCH}" \
  --e2-downsamples "${DOWNSAMPLES}" \
  "${SOURCE_E2_INIT_ARG[@]}" \
  "${CHANNEL_RMS_NORMALIZE_ARG[@]}" \
  --latent-c "${LATENT_C}" \
  --d2-z1-channels "${D2_Z1_CHANNELS}" \
  --embedding-dim "${EMBEDDING_DIM}" \
  --rates "${RATES}" \
  --rate-weights 1 \
  --nested-channel-dropout \
  --nested-dropout-prob "${DROPOUT_ALPHA}" \
  --batch-size "${BATCH_SIZE}" \
  --test-batch "${TEST_BATCH}" \
  --accum-steps "${ACCUM_STEPS}" \
  --num-workers 8 \
  --val-num-workers 4 \
  --epochs "${EPOCHS}" \
  --continuous-warmup-epochs "${CONTINUOUS_WARMUP_EPOCHS}" \
  --initial-residual-gain "${INITIAL_RESIDUAL_GAIN}" \
  --val-every 5 \
  --latest-every 5 \
  --lr 1e-4 \
  --lambda-vq 0.02 \
  --lambda-monotonic 0.1 \
  --query-chunk-size "${QUERY_CHUNK_SIZE}" \
  --codebook-chunk-size "${CODEBOOK_CHUNK_SIZE}" \
  --codebook-init-batches 16 \
  --codebook-init-max-samples 65536 \
  --save-dir MY-V2/jscc-f/explore-4/checkpoints-cvq \
  --log-json "MY-V2/jscc-f/explore-4/results-cvq/${VERSION}.json" \
  --version "${VERSION}" \
  --lambda-direct-gain "${LAMBDA_DIRECT_GAIN}" \
  --direct-gain-db "${DIRECT_GAIN_DB}" \
  "$@"
