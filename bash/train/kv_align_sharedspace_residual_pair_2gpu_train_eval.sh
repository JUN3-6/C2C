#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
MASTER_PORT_HIDDENSEED=${MASTER_PORT_HIDDENSEED:-29534}
MASTER_PORT_SHAREDSPACE=${MASTER_PORT_SHAREDSPACE:-29535}
LOG_DIR=${LOG_DIR:-local/logs/kv_align_sharedspace_residual_pair_2gpu}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/train_eval_kv_align_sharedspace_residual_pair_2gpu_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "[1/4] Train hidden-seed shared-space residual"
torchrun --nproc_per_node=2 --master_port="${MASTER_PORT_HIDDENSEED}" script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_kv_align_hiddenseed_sharedspace_residual_0.6+0.5_MMLU_15k_2gpu.json

echo "[2/4] Evaluate hidden-seed shared-space residual on mmlu-redux"
python script/evaluation/unified_evaluator.py \
    --config recipe/eval_recipe/mmlu_redux_kv_align_hiddenseed_sharedspace_residual_2gpu.yaml

echo "[3/4] Train shared-space residual"
torchrun --nproc_per_node=2 --master_port="${MASTER_PORT_SHAREDSPACE}" script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_kv_align_sharedspace_residual_0.6+0.5_MMLU_15k_2gpu.json

echo "[4/4] Evaluate shared-space residual on mmlu-redux"
python script/evaluation/unified_evaluator.py \
    --config recipe/eval_recipe/mmlu_redux_kv_align_sharedspace_residual_2gpu.yaml
