#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MASTER_PORT_BASE=${MASTER_PORT_BASE:-29920}
LOG_DIR=${LOG_DIR:-local/logs/original_c2c_qwen3_1.7b_sharer_sweep_gpu2_3_train_eval_mismatch}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
MAIN_LOG=${MAIN_LOG:-${LOG_DIR}/main_${TIMESTAMP}.log}
SKIP_TRAIN_IF_FINAL=${SKIP_TRAIN_IF_FINAL:-1}
SKIP_EVAL_IF_SUMMARY=${SKIP_EVAL_IF_SUMMARY:-0}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${MAIN_LOG}") 2>&1

NAMES=(
  "qwen3_4b_to_qwen3_1.7b_original"
  "qwen2.5_3b_to_qwen3_1.7b_original"
)

GPUS=(
  "2"
  "3"
)

TRAIN_CONFIGS=(
  "recipe/train_recipe/C2C_original_qwen3_1.7b+qwen3_4b_MMLU_15k_gpu2_1gpu.json"
  "recipe/train_recipe/C2C_original_qwen3_1.7b+qwen2.5_3b_instruct_MMLU_15k_gpu3_1gpu.json"
)

EVAL_CONFIGS=(
  "recipe/eval_recipe/mmlu_redux_original_qwen3_1.7b_qwen3_4b_mmlu_15k_gpu2_1gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_original_qwen3_1.7b_qwen2.5_3b_instruct_mmlu_15k_gpu3_1gpu.yaml"
)

MISMATCH_CONFIGS=(
  "recipe/eval_recipe/mmlu_redux_sharer_mismatch_original_qwen3_1.7b_qwen3_4b_mmlu_15k_gpu2_1gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_sharer_mismatch_original_qwen3_1.7b_qwen2.5_3b_instruct_mmlu_15k_gpu3_1gpu.yaml"
)

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

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_one() {
  local idx="$1"
  local name="${NAMES[$idx]}"
  local gpu="${GPUS[$idx]}"
  local train_config="${TRAIN_CONFIGS[$idx]}"
  local eval_config="${EVAL_CONFIGS[$idx]}"
  local mismatch_config="${MISMATCH_CONFIGS[$idx]}"
  local master_port=$((MASTER_PORT_BASE + idx))
  local job_log="${LOG_DIR}/${name}_${TIMESTAMP}.log"
  local checkpoint_dir
  local eval_output_dir
  local mismatch_output_dir

  checkpoint_dir="$(json_get_output_dir "${train_config}")"
  eval_output_dir="$(yaml_get_output_dir "${eval_config}")"
  mismatch_output_dir="$(yaml_get_output_dir "${mismatch_config}")"

  (
    set -euo pipefail
    exec > >(tee -a "${job_log}") 2>&1
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"

    echo "Job: ${name}"
    echo "Physical GPU: ${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "Train config: ${train_config}"
    echo "Eval config: ${eval_config}"
    echo "Mismatch config: ${mismatch_config}"
    echo "Checkpoint dir: ${checkpoint_dir}"
    echo "Eval output dir: ${eval_output_dir}"
    echo "Mismatch output dir: ${mismatch_output_dir}"
    echo "Start time: $(date)"

    if [[ "${SKIP_TRAIN_IF_FINAL}" == "1" && -d "${checkpoint_dir}/final" ]]; then
      echo "Skip training because ${checkpoint_dir}/final exists."
    else
      run_or_print torchrun --nproc_per_node=1 --master_port="${master_port}" script/train/SFT_train.py \
        --config "${train_config}"
    fi

    if [[ "${SKIP_EVAL_IF_SUMMARY}" == "1" ]] && compgen -G "${eval_output_dir}/*summary.json" > /dev/null; then
      echo "Skip normal eval because summary exists in ${eval_output_dir}."
    else
      run_or_print python script/evaluation/unified_evaluator.py \
        --config "${eval_config}"
    fi

    if [[ "${SKIP_EVAL_IF_SUMMARY}" == "1" ]] && compgen -G "${mismatch_output_dir}/*summary.json" > /dev/null; then
      echo "Skip mismatch eval because summary exists in ${mismatch_output_dir}."
    else
      run_or_print python script/evaluation/unified_evaluator.py \
        --config "${mismatch_config}"
    fi

    echo "Done ${name}: $(date)"
  )
}

echo "Logging to ${MAIN_LOG}"
echo "Start time: $(date)"
echo "SKIP_TRAIN_IF_FINAL=${SKIP_TRAIN_IF_FINAL}"
echo "SKIP_EVAL_IF_SUMMARY=${SKIP_EVAL_IF_SUMMARY}"
echo "DRY_RUN=${DRY_RUN}"

pids=()
for idx in "${!NAMES[@]}"; do
  run_one "${idx}" &
  pids+=("$!")
  echo "Started ${NAMES[$idx]} on physical GPU ${GPUS[$idx]} with PID ${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "One or more jobs failed." >&2
  exit "${status}"
fi

echo "All jobs completed: $(date)"
