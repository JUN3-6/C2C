#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG_DIR=${LOG_DIR:-local/logs/mmlu_redux_openhermes_fuser_forced_label_positions}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/openhermes_fuser_forced_label_positions_${TIMESTAMP}.log}
TARGET_LABELS=${TARGET_LABELS:-"A B C D"}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
SUMMARY_OUT=${SUMMARY_OUT:-local/final_results/openhermes_fuser_forced_label_positions_summary_latest.json}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

JOBS=(
  "qwen3_0.6b+qwen3_4b_Fuser|Qwen/Qwen3-0.6B|Qwen/Qwen3-4B|qwen3_0.6b_qwen3_4b_OpenHermes_Fuser"
  "qwen3_0.6b+qwen2.5_0.5b_Fuser|Qwen/Qwen3-0.6B|Qwen/Qwen2.5-0.5B-Instruct|qwen3_0.6b_qwen2.5_0.5b_OpenHermes_Fuser"
)

run_one() {
  local fuser_subdir="$1"
  local base_model="$2"
  local teacher_model="$3"
  local run_id="$4"
  local target_label="$5"

  FUSER_SUBDIR="${fuser_subdir}" \
    BASE_MODEL="${base_model}" \
    TEACHER_MODEL="${teacher_model}" \
    RUN_ID="${run_id}" \
    TARGET_LABEL="${target_label}" \
    bash bash/eval/mmlu_redux_hf_fuser_forced_label.sh
}

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TARGET_LABELS=${TARGET_LABELS}"
echo "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
echo "SUMMARY_OUT=${SUMMARY_OUT}"

FAILED=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r fuser_subdir base_model teacher_model run_id <<< "${job}"
  for target_label in ${TARGET_LABELS}; do
    echo
    echo "============================================================"
    echo "Running ${run_id}, forced ${target_label}"
    echo "============================================================"
    if run_one "${fuser_subdir}" "${base_model}" "${teacher_model}" "${run_id}" "${target_label}"; then
      echo "[done] ${run_id} forced ${target_label}"
    else
      echo "[failed] ${run_id} forced ${target_label}" >&2
      FAILED+=("${run_id}:forced_${target_label}")
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        exit 1
      fi
    fi
  done
done

SUMMARY_ARGS=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r _ _ _ run_id <<< "${job}"
  for target_label in ${TARGET_LABELS}; do
    SUMMARY_ARGS+=(--result "${run_id}_forced_${target_label}" "local/final_results/${run_id}_forced_${target_label}_mmlu_redux")
  done
done

echo
echo "[summary] Forced-label result summary"
python script/analysis/analyze_forced_label_results.py \
  "${SUMMARY_ARGS[@]}" \
  --output-json "${SUMMARY_OUT}"

if (( ${#FAILED[@]} > 0 )); then
  printf 'Failed jobs:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi

echo "Done."
