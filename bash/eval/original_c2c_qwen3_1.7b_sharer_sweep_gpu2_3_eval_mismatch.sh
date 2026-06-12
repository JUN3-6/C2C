#!/usr/bin/env bash
set -euo pipefail

export EVAL_ONLY=${EVAL_ONLY:-1}
export EVAL_USE_PHYSICAL_GPU_IDS=${EVAL_USE_PHYSICAL_GPU_IDS:-1}
export PHYSICAL_VISIBLE_DEVICES=${PHYSICAL_VISIBLE_DEVICES:-0,1,2,3}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-64}
export EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-16}

bash bash/train/original_c2c_qwen3_1.7b_sharer_sweep_gpu2_3_train_eval_mismatch.sh
