#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MASTER_PORT_BASE=${MASTER_PORT_BASE:-29620}
LOG_DIR=${LOG_DIR:-local/logs/top5_qwen3_4b_mmlu15k_bs8_2gpu_train_eval}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/top5_qwen3_4b_mmlu15k_bs8_2gpu_train_eval_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Start time: $(date)"

NPROC=${NPROC:-2}

NAMES=(
  "original"
  "hiddenseed_sharedspace_residual_gate_0"
  "receiverseed_sharedspace_residual_gate_0"
  "sourcequery_receivermemory_sharedspace_residual_gate_0"
  "sharedspace_residual_gate_0"
)

TRAIN_CONFIGS=(
  "recipe/train_recipe/C2C_original_qwen3_0.6b+qwen3_4b_MMLU_15k_2gpu.json"
  "recipe/train_recipe/C2C_kv_align_hiddenseed_sharedspace_residual_gate_0_qwen3_0.6b+qwen3_4b_MMLU_15k_2gpu.json"
  "recipe/train_recipe/C2C_kv_align_receiverseed_sharedspace_residual_gate_0_qwen3_0.6b+qwen3_4b_MMLU_15k_2gpu.json"
  "recipe/train_recipe/C2C_kv_align_sourcequery_receivermemory_sharedspace_residual_gate_0_qwen3_0.6b+qwen3_4b_MMLU_15k_2gpu.json"
  "recipe/train_recipe/C2C_kv_align_sharedspace_residual_gate_0_qwen3_0.6b+qwen3_4b_MMLU_15k_2gpu.json"
)

EVAL_CONFIGS=(
  "recipe/eval_recipe/mmlu_redux_original_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_kv_align_hiddenseed_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_kv_align_receiverseed_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_kv_align_sourcequery_receivermemory_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_kv_align_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  train_config="${TRAIN_CONFIGS[$i]}"
  eval_config="${EVAL_CONFIGS[$i]}"
  master_port=$((MASTER_PORT_BASE + i))

  echo
  echo "[$((i + 1))/5] Train ${name}"
  echo "Config: ${train_config}"
  torchrun --nproc_per_node="${NPROC}" --master_port="${master_port}" script/train/SFT_train.py \
    --config "${train_config}"

  echo
  echo "[$((i + 1))/5] Evaluate ${name} on mmlu-redux"
  echo "Config: ${eval_config}"
  python script/evaluation/unified_evaluator.py \
    --config "${eval_config}"
done

echo
echo "Done: $(date)"
