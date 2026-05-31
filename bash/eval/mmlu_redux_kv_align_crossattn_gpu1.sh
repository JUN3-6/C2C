#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
LOG_DIR=${LOG_DIR:-local/logs/kv_align_crossattn_1gpu}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/eval_mmlu_redux_kv_align_crossattn_gpu1_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python script/evaluation/unified_evaluator.py \
    --config recipe/eval_recipe/mmlu_redux_kv_align_crossattn_1gpu.yaml
