#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
MASTER_PORT=${MASTER_PORT:-29546}
LOG_DIR=${LOG_DIR:-local/logs/kv_align_sourcequery_receivermemory_sharedspace_residual_mmlu_15k_gate_0_bs16_2gpu}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/train_eval_gate_0_bs16_2gpu_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "[1/2] Train source-query receiver-memory shared-space residual MMLU-15k gate init 0.0"
torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_kv_align_sourcequery_receivermemory_sharedspace_residual_gate_0_0.6+0.5_MMLU_15k_bs16_2gpu.json

echo "[2/2] Evaluate source-query receiver-memory shared-space residual MMLU-15k gate init 0.0 on mmlu-redux"
python script/evaluation/unified_evaluator.py \
    --config recipe/eval_recipe/mmlu_redux_kv_align_sourcequery_receivermemory_sharedspace_residual_mmlu_15k_gate_0_bs16_2gpu.yaml
