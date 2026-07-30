#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

project_path() {
    if [[ "$1" = /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s/%s\n' "$PROJECT_ROOT" "${1#./}"
    fi
}

PYTHON="${PYTHON:-python}"
DATA_ROOT="$(project_path "${DATA_ROOT:-data/Kvasir-SEG}")"
EMBEDDING_ROOT="$(project_path "${EMBEDDING_ROOT:-${DATA_ROOT}/embeddings}")"
PSEUDO_HIGH_ROOT="$(project_path "${PSEUDO_HIGH_ROOT:-${DATA_ROOT}/pseudo_masks/labelshigh}")"
PSEUDO_LOW_ROOT="$(project_path "${PSEUDO_LOW_ROOT:-${DATA_ROOT}/pseudo_masks/labelslow}")"
OUTPUT_ROOT="$(project_path "${OUTPUT_ROOT:-outputs}")"
RESULTS_DIR="$(project_path "${RESULTS_DIR:-${OUTPUT_ROOT}/results}")"
LOG_DIR="$(project_path "${LOG_DIR:-${OUTPUT_ROOT}/logs}")"
CHECKPOINT_DIR="$(project_path "${CHECKPOINT_DIR:-${OUTPUT_ROOT}/checkpoints}")"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SPATIAL_SIZE="${SPATIAL_SIZE:-224}"
DATA_SIZE_RATE="${DATA_SIZE_RATE:-1}"
MAX_ITE="${MAX_ITE:-15000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FP16="${FP16:-1}"
VAL_STEP="${VAL_STEP:-100}"
LOG_STEP="${LOG_STEP:-100}"
PPC_WEIGHT="${PPC_WEIGHT:-1.0}"
FEATURE_DISTILLATION_WEIGHT="${FEATURE_DISTILLATION_WEIGHT:-0.5}"
IMAGE_PROTO_BLEND="${IMAGE_PROTO_BLEND:-0.25}"
EVAL_ONLY="${EVAL_ONLY:-0}"
CKPT_PATH="$(project_path "${CKPT_PATH:-${CHECKPOINT_DIR}/AutoGazeSeg_kvasir_seed${SEED}.pth}")"
SAVE_NAME="${SAVE_NAME:-AutoGazeSeg_kvasir_seed${SEED}}"

if [[ "$EVAL_ONLY" != "0" && "$EVAL_ONLY" != "1" ]]; then
    echo "EVAL_ONLY must be 0 or 1." >&2
    exit 2
fi
if [[ "$FP16" != "0" && "$FP16" != "1" ]]; then
    echo "FP16 must be 0 or 1." >&2
    exit 2
fi

COMMON_ARGS=(
    --method autogazeseg
    --model autogazeseg
    --data kvasir
    --root "$DATA_ROOT"
    --embedding_root "$EMBEDDING_ROOT"
    --pseudo_high_root "$PSEUDO_HIGH_ROOT"
    --pseudo_low_root "$PSEUDO_LOW_ROOT"
    --exp_result_path "$RESULTS_DIR"
    --log_path "$LOG_DIR"
    --params_path "$CHECKPOINT_DIR"
    --spatial_size "$SPATIAL_SIZE"
    --in_channels 3
    --opt sgd
    --lr 1e-2
    --lr_min 1e-4
    --lr_scheduler cos
    --weight_decay 4e-4
    --batch_size "$BATCH_SIZE"
    --max_ite "$MAX_ITE"
    --num_worker "$NUM_WORKERS"
    --val_step "$VAL_STEP"
    --log_step "$LOG_STEP"
    --ppc_weight "$PPC_WEIGHT"
    --feature_distillation_weight "$FEATURE_DISTILLATION_WEIGHT"
    --image_proto_blend "$IMAGE_PROTO_BLEND"
    --data_size_rate "$DATA_SIZE_RATE"
    --device "$DEVICE"
    --seed "$SEED"
)
if [[ "$FP16" == "1" ]]; then
    COMMON_ARGS+=(--fp16)
fi

if [[ "$EVAL_ONLY" == "0" ]]; then
    "$PYTHON" "$PROJECT_ROOT/run.py" "${COMMON_ARGS[@]}"
fi

if [[ ! -e "$CKPT_PATH" ]]; then
    echo "Checkpoint not found: $CKPT_PATH" >&2
    exit 1
fi

"$PYTHON" "$PROJECT_ROOT/run.py" \
    "${COMMON_ARGS[@]}" \
    --test \
    --ckpt_path "$CKPT_PATH" \
    --save_name "$SAVE_NAME"
