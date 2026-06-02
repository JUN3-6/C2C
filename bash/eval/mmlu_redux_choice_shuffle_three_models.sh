#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG_DIR=${LOG_DIR:-local/logs/mmlu_redux_choice_shuffle_three_models}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/choice_shuffle_three_models_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

ORIGINAL_CONFIG=${ORIGINAL_CONFIG:-recipe/eval_recipe/mmlu_redux_choice_shuffle_original_c2c_mmlu_15k.yaml}
RECEIVER_SEED_CONFIG=${RECEIVER_SEED_CONFIG:-recipe/eval_recipe/mmlu_redux_choice_shuffle_kv_align_receiverseed_sharedspace_residual_mmlu_15k_gate_0_bs16_2gpu.yaml}
SHARER_QUERY_CONFIG=${SHARER_QUERY_CONFIG:-recipe/eval_recipe/mmlu_redux_choice_shuffle_kv_align_sharerquery_receivermemory_sharedspace_residual_mmlu_15k_gate_0_bs16_2gpu.yaml}

echo "[1/3] Original C2C choice-shuffle eval"
python script/evaluation/unified_evaluator_choice_shuffle.py --config "${ORIGINAL_CONFIG}"

echo "[2/3] Receiver-seed KVAlign choice-shuffle eval"
python script/evaluation/unified_evaluator_choice_shuffle.py --config "${RECEIVER_SEED_CONFIG}"

echo "[3/3] Sharer-query receiver-memory KVAlign choice-shuffle eval"
python script/evaluation/unified_evaluator_choice_shuffle.py --config "${SHARER_QUERY_CONFIG}"

echo "[summary] Choice-shuffle result summary"
python script/analysis/analyze_choice_shuffle_results.py \
  --result original_c2c local/final_results/0.6+0.5B_C2C_original_MMLU_15k_choice_shuffle_seed0_mmlu_redux \
  --result receiver_seed local/final_results/0.6+0.5B_C2C_kv_align_receiverseed_sharedspace_residual_MMLU_15k_gate_0_bs16_2gpu_choice_shuffle_seed0_mmlu_redux \
  --result sharer_query local/final_results/0.6+0.5B_C2C_kv_align_sharerquery_receivermemory_sharedspace_residual_MMLU_15k_gate_0_bs16_2gpu_choice_shuffle_seed0_mmlu_redux

echo "Done."
