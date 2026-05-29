#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_GPU=${ORIGINAL_GPU:-0}
V3_GPU=${V3_GPU:-1}
ORIGINAL_PORT=${ORIGINAL_PORT:-29521}
V3_PORT=${V3_PORT:-29522}
LOG_DIR=${LOG_DIR:-local/logs/train_eval_original_and_v3_mmlu_15k_1gpu_each}

ORIGINAL_TRAIN_CONFIG=${ORIGINAL_TRAIN_CONFIG:-recipe/train_recipe/C2C_original_0.6+0.5_MMLU_15k_1gpu_parallel.json}
V3_TRAIN_CONFIG=${V3_TRAIN_CONFIG:-recipe/train_recipe/C2C_v3_hidden_0.6+0.5_MMLU_15k_1gpu_parallel.json}
ORIGINAL_EVAL_CONFIG=${ORIGINAL_EVAL_CONFIG:-recipe/eval_recipe/mmlu_redux_original_mmlu_15k_1gpu_parallel.yaml}
V3_EVAL_CONFIG=${V3_EVAL_CONFIG:-recipe/eval_recipe/mmlu_redux_v3_hidden_mmlu_15k_1gpu_parallel.yaml}

mkdir -p "${LOG_DIR}"

cleanup() {
    kill "${original_train_pid:-}" "${v3_train_pid:-}" "${original_eval_pid:-}" "${v3_eval_pid:-}" 2>/dev/null || true
}
trap cleanup INT TERM

echo "Starting training jobs..."
CUDA_VISIBLE_DEVICES="${ORIGINAL_GPU}" torchrun --nproc_per_node=1 --master_port="${ORIGINAL_PORT}" \
    script/train/SFT_train.py --config "${ORIGINAL_TRAIN_CONFIG}" \
    > "${LOG_DIR}/train_original_c2c_gpu${ORIGINAL_GPU}.log" 2>&1 &
original_train_pid=$!

CUDA_VISIBLE_DEVICES="${V3_GPU}" torchrun --nproc_per_node=1 --master_port="${V3_PORT}" \
    script/train/SFT_train_v3.py --config "${V3_TRAIN_CONFIG}" \
    > "${LOG_DIR}/train_v3_hidden_gpu${V3_GPU}.log" 2>&1 &
v3_train_pid=$!

echo "Original C2C train PID ${original_train_pid} on GPU ${ORIGINAL_GPU}"
echo "V3 hidden train PID ${v3_train_pid} on GPU ${V3_GPU}"
echo "Train logs:"
echo "  ${LOG_DIR}/train_original_c2c_gpu${ORIGINAL_GPU}.log"
echo "  ${LOG_DIR}/train_v3_hidden_gpu${V3_GPU}.log"

set +e
wait "${original_train_pid}"
original_train_status=$?
wait "${v3_train_pid}"
v3_train_status=$?
set -e

if [[ "${original_train_status}" -ne 0 || "${v3_train_status}" -ne 0 ]]; then
    echo "Training failed: original=${original_train_status}, v3=${v3_train_status}" >&2
    exit 1
fi

echo "Training complete. Starting mmlu-redux evaluation jobs..."
CUDA_VISIBLE_DEVICES="${ORIGINAL_GPU}" python script/evaluation/unified_evaluator.py \
    --config "${ORIGINAL_EVAL_CONFIG}" \
    > "${LOG_DIR}/eval_mmlu_redux_original_c2c_gpu${ORIGINAL_GPU}.log" 2>&1 &
original_eval_pid=$!

CUDA_VISIBLE_DEVICES="${V3_GPU}" python script/evaluation/unified_evaluator_v3.py \
    --config "${V3_EVAL_CONFIG}" \
    > "${LOG_DIR}/eval_mmlu_redux_v3_hidden_gpu${V3_GPU}.log" 2>&1 &
v3_eval_pid=$!

echo "Original C2C eval PID ${original_eval_pid} on GPU ${ORIGINAL_GPU}"
echo "V3 hidden eval PID ${v3_eval_pid} on GPU ${V3_GPU}"
echo "Eval logs:"
echo "  ${LOG_DIR}/eval_mmlu_redux_original_c2c_gpu${ORIGINAL_GPU}.log"
echo "  ${LOG_DIR}/eval_mmlu_redux_v3_hidden_gpu${V3_GPU}.log"

set +e
wait "${original_eval_pid}"
original_eval_status=$?
wait "${v3_eval_pid}"
v3_eval_status=$?
set -e
trap - INT TERM

if [[ "${original_eval_status}" -ne 0 || "${v3_eval_status}" -ne 0 ]]; then
    echo "Evaluation failed: original=${original_eval_status}, v3=${v3_eval_status}" >&2
    exit 1
fi

echo "Both training and mmlu-redux evaluation jobs completed successfully."
