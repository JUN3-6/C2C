#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
MASTER_PORT=${MASTER_PORT:-29514}

torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" script/train/SFT_train_v4.py \
    --config recipe/train_recipe/C2C_v4_kv_hidden_0.6+0.5_2gpu.json

python script/evaluation/unified_evaluator_v4.py \
    --config recipe/eval_recipe/mmlu_redux_v4_kv_hidden.yaml
