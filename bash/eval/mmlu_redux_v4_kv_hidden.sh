#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

python script/evaluation/unified_evaluator_v4.py \
    --config recipe/eval_recipe/mmlu_redux_v4_kv_hidden.yaml
