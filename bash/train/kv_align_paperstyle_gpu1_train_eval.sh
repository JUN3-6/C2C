#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
LOG_DIR=${LOG_DIR:-local/logs/kv_align_paperstyle_gpu1}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/train_eval_kv_align_paperstyle_gpu1_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_kv_align_paperstyle_0.6+0.5_MMLU_15k_gpu1.json

python script/evaluation/unified_evaluator.py \
    --config recipe/eval_recipe/mmlu_redux_kv_align_paperstyle_gpu1.yaml
