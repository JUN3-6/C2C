#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG_DIR=${LOG_DIR:-local/logs/mmlu_redux_kv_content_ablation}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/kv_content_global_exact_original_c2c_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

CONFIG=${CONFIG:-recipe/eval_recipe/mmlu_redux_kv_content_ablation_global_exact_original_c2c_mmlu_15k.yaml}
RESULT_DIR=${RESULT_DIR:-local/final_results/0.6+0.5B_C2C_original_MMLU_15k_kv_content_global_exact_seed0_mmlu_redux}

python script/evaluation/unified_evaluator_kv_content_ablation.py --config "${CONFIG}"

python script/analysis/analyze_kv_content_ablation_results.py \
  --result original_c2c_global_exact "${RESULT_DIR}" \
  --output-json "${RESULT_DIR}/kv_content_ablation_summary_latest.json"

echo "Done."
