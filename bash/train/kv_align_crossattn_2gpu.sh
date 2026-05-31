#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
MASTER_PORT=${MASTER_PORT:-29519}
LOG_DIR=${LOG_DIR:-local/logs/kv_align_crossattn_2gpu}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/train_kv_align_crossattn_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_kv_align_crossattn_0.6+0.5_MMLU_15k_2gpu.json
