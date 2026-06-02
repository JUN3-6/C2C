#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
MASTER_PORT_M1=${MASTER_PORT_M1:-29540}
MASTER_PORT_0=${MASTER_PORT_0:-29541}
LOG_DIR=${LOG_DIR:-local/logs/kv_align_hiddenseed_sharedspace_residual_openhermes_500k_gate_sweep_m1_0_bs8_2gpu}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/train_eval_gate_sweep_m1_0_bs8_2gpu_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "[1/4] Train gate init -1.0"
torchrun --nproc_per_node=2 --master_port="${MASTER_PORT_M1}" script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_kv_align_hiddenseed_sharedspace_residual_gate_m1_0.6+0.5_OpenHermes_500k_bs8_2gpu.json

echo "[2/4] Evaluate gate init -1.0 on mmlu-redux"
python script/evaluation/unified_evaluator.py \
    --config recipe/eval_recipe/mmlu_redux_kv_align_hiddenseed_sharedspace_residual_openhermes_500k_gate_m1_bs8_2gpu.yaml

echo "[3/4] Train gate init 0.0"
torchrun --nproc_per_node=2 --master_port="${MASTER_PORT_0}" script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_kv_align_hiddenseed_sharedspace_residual_gate_0_0.6+0.5_OpenHermes_500k_bs8_2gpu.json

echo "[4/4] Evaluate gate init 0.0 on mmlu-redux"
python script/evaluation/unified_evaluator.py \
    --config recipe/eval_recipe/mmlu_redux_kv_align_hiddenseed_sharedspace_residual_openhermes_500k_gate_0_bs8_2gpu.yaml
