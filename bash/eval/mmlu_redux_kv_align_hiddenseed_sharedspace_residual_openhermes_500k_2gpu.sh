#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
LOG_DIR=${LOG_DIR:-local/logs/kv_align_hiddenseed_sharedspace_residual_openhermes_500k_bs2_2gpu}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/eval_mmlu_redux_kv_align_hiddenseed_sharedspace_residual_openhermes_500k_bs2_2gpu_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python script/evaluation/unified_evaluator.py \
    --config recipe/eval_recipe/mmlu_redux_kv_align_hiddenseed_sharedspace_residual_openhermes_500k_2gpu.yaml
