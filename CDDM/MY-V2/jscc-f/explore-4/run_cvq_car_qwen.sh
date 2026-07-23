#!/usr/bin/env bash
# Paper-2605.26089v2 CAR Stage-I launcher: z1-only Qwen3 receiver generator.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 SENDER_CHECKPOINT VERSION [extra train_cvq_car_qwen.py args]" >&2
  exit 2
fi

SENDER_CHECKPOINT=$1
VERSION=$2
shift 2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$ROOT/../../.." && pwd)
cd "$REPO"

RATE=${RATE:-4096}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-4B}
BATCH_SIZE=${BATCH_SIZE:-4}
# The Qwen3-4B backbone is frozen in Stage-I.  Autoregressive validation has
# no activation-gradient cost, so use the spare memory to generate several
# images in parallel instead of serializing 100 x 240 decoder steps.
TEST_BATCH=${TEST_BATCH:-8}
EPOCHS=${EPOCHS:-40}
VAL_EVERY=${VAL_EVERY:-10}

# This environment cannot reach Hugging Face directly; hf-mirror serves the
# same public Qwen repository.  A local model path remains valid for QWEN_MODEL.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
exec conda run --no-capture-output -n cddm_ddnm python "$ROOT/train_cvq_car_qwen.py" \
  --sender-checkpoint "$SENDER_CHECKPOINT" --rate "$RATE" --qwen-model "$QWEN_MODEL" \
  --epochs "$EPOCHS" --val-every "$VAL_EVERY" --batch-size "$BATCH_SIZE" --test-batch "$TEST_BATCH" \
  --freeze-receiver-decoder --rollout-q-mode hard --rollout-temperature 0.2 \
  --save-dir "MY-V2/jscc-f/explore-4/checkpoints-car" \
  --log-json "MY-V2/jscc-f/explore-4/results-car/${VERSION}.json" --version "$VERSION" "$@"
