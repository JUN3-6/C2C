#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_GPU=${ORIGINAL_GPU:-0}
V3_GPU=${V3_GPU:-1}
ORIGINAL_PORT=${ORIGINAL_PORT:-29521}
V3_PORT=${V3_PORT:-29522}
LOG_DIR=${LOG_DIR:-local/logs/original_and_v3_mmlu_15k_1gpu_each}

ORIGINAL_CONFIG=${ORIGINAL_CONFIG:-recipe/train_recipe/C2C_original_0.6+0.5_MMLU_15k_1gpu_parallel.json}
V3_CONFIG=${V3_CONFIG:-recipe/train_recipe/C2C_v3_hidden_0.6+0.5_MMLU_15k_1gpu_parallel.json}

mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="${ORIGINAL_GPU}" torchrun --nproc_per_node=1 --master_port="${ORIGINAL_PORT}" \
    script/train/SFT_train.py --config "${ORIGINAL_CONFIG}" \
    > "${LOG_DIR}/original_c2c_gpu${ORIGINAL_GPU}.log" 2>&1 &
original_pid=$!

CUDA_VISIBLE_DEVICES="${V3_GPU}" torchrun --nproc_per_node=1 --master_port="${V3_PORT}" \
    script/train/SFT_train_v3.py --config "${V3_CONFIG}" \
    > "${LOG_DIR}/v3_hidden_gpu${V3_GPU}.log" 2>&1 &
v3_pid=$!

cleanup() {
    kill "${original_pid}" "${v3_pid}" 2>/dev/null || true
}
trap cleanup INT TERM

echo "Original C2C PID ${original_pid} on GPU ${ORIGINAL_GPU}"
echo "V3 hidden C2C PID ${v3_pid} on GPU ${V3_GPU}"
echo "Logs:"
echo "  ${LOG_DIR}/original_c2c_gpu${ORIGINAL_GPU}.log"
echo "  ${LOG_DIR}/v3_hidden_gpu${V3_GPU}.log"

set +e
wait "${original_pid}"
original_status=$?
wait "${v3_pid}"
v3_status=$?
set -e
trap - INT TERM

if [[ "${original_status}" -ne 0 || "${v3_status}" -ne 0 ]]; then
    echo "Training failed: original=${original_status}, v3=${v3_status}" >&2
    exit 1
fi

echo "Both training jobs completed successfully."
