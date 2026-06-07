#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG_DIR=${LOG_DIR:-local/logs/top5_qwen3_4b_mmlu15k_sharer_mismatch_2gpu}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/top5_qwen3_4b_mmlu15k_sharer_mismatch_2gpu_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Start time: $(date)"

CONFIGS=(
  "recipe/eval_recipe/mmlu_redux_sharer_mismatch_original_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_sharer_mismatch_kv_align_hiddenseed_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_sharer_mismatch_kv_align_receiverseed_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_sharer_mismatch_kv_align_sourcequery_receivermemory_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_sharer_mismatch_kv_align_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo
  echo "Evaluate sharer mismatch: ${config}"
  python script/evaluation/unified_evaluator.py --config "${config}"
done

echo
echo "Done: $(date)"
