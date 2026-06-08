#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SNR="${1:?usage: $0 SNR_DB GPU ROOT_DIR}"
GPU="${2:?usage: $0 SNR_DB GPU ROOT_DIR}"
ROOT="${3:?usage: $0 SNR_DB GPU ROOT_DIR}"
shift 3
BASELINE="${BASELINE_PSNR:-nan}"

TAG="awgn${SNR}"
STAGE1_DIR="${ROOT}/01_light_ar_joint_no_sideinfo_${TAG}"
STAGE2_DIR="${ROOT}/02_no_scale_ar_h256d6_frozen_${TAG}"

mkdir -p "${STAGE1_DIR}" "${STAGE2_DIR}"
printf "%s\n" "$$" > "${ROOT}/pipeline.pid"

log_root() {
  echo "[$(date '+%F %T')] $*" | tee -a "${ROOT}/pipeline.log"
}

log_root "start no-side-info pipeline SNR=${SNR}dB GPU=${GPU}"
log_root "stage1 train Swin + light AR with norm receiver -> ${STAGE1_DIR}"
env STAGE="joint" \
  AR_ARCH="simple" AR_HIDDEN="160" AR_DEPTH="4" \
  EPOCHS="400" LR="5e-5" AR_LR="0" DECODER_LR="0" \
  TEACHER_FORCE_EPOCHS="50" TEACHER_DECAY_EPOCHS="100" \
  train/run_hierarchical_swin_ar_awgn12.sh "${GPU}" "${STAGE1_DIR}" \
    --snr_db "${SNR}" --baseline_psnr "${BASELINE}" \
    --no_ar_scale --receiver_observation norm \
    "$@" \
  2>&1 | tee -a "${STAGE1_DIR}/train.stdout"

INIT_CKPT="${STAGE1_DIR}/hierarchical_swin_ar_awgn12_best.pth"
if [[ ! -f "${INIT_CKPT}" ]]; then
  log_root "missing stage1 checkpoint: ${INIT_CKPT}"
  exit 1
fi

log_root "stage2 freeze Swin and train no-scale simple h256d6 AR -> ${STAGE2_DIR}"
env INIT_HIER_CKPT="${INIT_CKPT}" \
  STAGE="hierarchical_group_ar_frozen_swin" \
  AR_ARCH="simple" AR_HIDDEN="256" AR_DEPTH="6" \
  EPOCHS="240" LR="5e-5" AR_LR="2e-4" \
  TEACHER_FORCE_EPOCHS="60" TEACHER_DECAY_EPOCHS="120" \
  train/run_hierarchical_swin_ar_awgn12.sh "${GPU}" "${STAGE2_DIR}" \
    --snr_db "${SNR}" --baseline_psnr "${BASELINE}" \
    --lambda_recv 1.0 --lambda_ar 0.2 \
    --no_ar_scale --receiver_observation norm \
    "$@" \
  2>&1 | tee -a "${STAGE2_DIR}/train.stdout"

log_root "done no-side-info pipeline SNR=${SNR}dB"
