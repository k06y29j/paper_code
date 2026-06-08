#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python MY/eval_cvq/coarse_to_fine.py \
  --checkpoint MY/checkpoints-cvq/cvq_c36_snr9_stage3_best.pth \
  --snr-db 9 \
  --latent-ch 36 \
  --prefix-ch 16
