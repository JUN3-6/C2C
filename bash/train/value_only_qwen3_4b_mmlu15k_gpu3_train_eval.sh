#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export NPROC=${NPROC:-1}
export EVAL_GPU_IDS=${EVAL_GPU_IDS:-0}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-64}
export RUN_VARIANTS=${RUN_VARIANTS:-"original hiddenseed receiverseed sourcequery_receivermemory sharedspace"}
export LOG_DIR=${LOG_DIR:-local/logs/value_only_qwen3_4b_mmlu15k_gpu3_train_eval}
export TMP_CONFIG_DIR=${TMP_CONFIG_DIR:-local/tmp/value_only_qwen3_4b_mmlu15k_gpu3_train_eval}

bash bash/train/value_only_qwen3_4b_mmlu15k_gpu0_train_eval.sh
