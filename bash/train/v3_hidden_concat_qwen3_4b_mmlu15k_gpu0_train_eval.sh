#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MASTER_PORT=${MASTER_PORT:-29830}
LOG_DIR=${LOG_DIR:-local/logs/v3_hidden_concat_qwen3_4b_mmlu15k_gpu0_train_eval}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/v3_hidden_concat_qwen3_4b_mmlu15k_gpu0_train_eval_${TIMESTAMP}.log}
TRAIN_CONFIG=${TRAIN_CONFIG:-recipe/train_recipe/C2C_v3_hidden_concat_qwen3_0.6b+qwen3_4b_MMLU_15k_gpu0_1gpu.json}
EVAL_CONFIG=${EVAL_CONFIG:-recipe/eval_recipe/mmlu_redux_v3_hidden_concat_qwen3_0.6b_qwen3_4b_mmlu_15k_gpu0_1gpu.yaml}
NPROC=${NPROC:-1}
SKIP_TRAIN_IF_FINAL=${SKIP_TRAIN_IF_FINAL:-1}
SKIP_EVAL_IF_SUMMARY=${SKIP_EVAL_IF_SUMMARY:-0}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

json_get_output_dir() {
  python - "$1" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    print(json.load(f)["output"]["output_dir"])
PY
}

yaml_get_output_dir() {
  python - "$1" <<'PY'
import sys
import yaml
with open(sys.argv[1]) as f:
    print(yaml.safe_load(f)["output"]["output_dir"])
PY
}

checkpoint_dir="$(json_get_output_dir "${TRAIN_CONFIG}")"
eval_output_dir="$(yaml_get_output_dir "${EVAL_CONFIG}")"

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC=${NPROC}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "EVAL_CONFIG=${EVAL_CONFIG}"
echo "Checkpoint dir: ${checkpoint_dir}"
echo "Eval output dir: ${eval_output_dir}"
echo "Start time: $(date)"

if [[ "${SKIP_TRAIN_IF_FINAL}" == "1" && -d "${checkpoint_dir}/final" ]]; then
  echo "Skip training because ${checkpoint_dir}/final exists."
else
  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" script/train/SFT_train_v3.py \
    --config "${TRAIN_CONFIG}"
fi

if [[ "${SKIP_EVAL_IF_SUMMARY}" == "1" ]] && compgen -G "${eval_output_dir}/*summary.json" > /dev/null; then
  echo "Skip eval because summary exists in ${eval_output_dir}."
else
  python script/evaluation/unified_evaluator_v3.py \
    --config "${EVAL_CONFIG}"
fi

echo "Done: $(date)"
