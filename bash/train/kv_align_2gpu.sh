#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
MASTER_PORT=${MASTER_PORT:-29516}

torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_kv_align_0.6+0.5_MMLU_15k_2gpu.json
