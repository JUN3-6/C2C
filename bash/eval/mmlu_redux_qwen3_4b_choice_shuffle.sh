#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG_DIR=${LOG_DIR:-local/logs/mmlu_redux_qwen3_4b_choice_shuffle}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/qwen3_4b_choice_shuffle_${TIMESTAMP}.log}
RUN_IN_PARALLEL=${RUN_IN_PARALLEL:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

JOBS=(
  "qwen3_0.6b+qwen3_4b_Fuser|Qwen/Qwen3-0.6B|Qwen/Qwen3-4B|qwen3_0.6b_qwen3_4b_OpenHermes_Fuser_choice_shuffle"
  "qwen3_0.6b+qwen3_4b_base_Fuser|Qwen/Qwen3-0.6B|Qwen/Qwen3-4B-Base|qwen3_0.6b_qwen3_4b_base_OpenHermes_Fuser_choice_shuffle"
)

run_one() {
  local fuser_subdir="$1"
  local base_model="$2"
  local teacher_model="$3"
  local run_id="$4"

  FUSER_SUBDIR="${fuser_subdir}" \
    BASE_MODEL="${base_model}" \
    TEACHER_MODEL="${teacher_model}" \
    RUN_ID="${run_id}" \
    bash bash/eval/mmlu_redux_hf_fuser_choice_shuffle.sh
}

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "RUN_IN_PARALLEL=${RUN_IN_PARALLEL}"

if [[ "${RUN_IN_PARALLEL}" == "1" ]]; then
  PIDS=()
  NAMES=()
  for job in "${JOBS[@]}"; do
    IFS='|' read -r fuser_subdir base_model teacher_model run_id <<< "${job}"
    echo "[launch] ${run_id}"
    run_one "${fuser_subdir}" "${base_model}" "${teacher_model}" "${run_id}" &
    PIDS+=("$!")
    NAMES+=("${run_id}")
  done

  FAILED=()
  for idx in "${!PIDS[@]}"; do
    if wait "${PIDS[$idx]}"; then
      echo "[done] ${NAMES[$idx]}"
    else
      echo "[failed] ${NAMES[$idx]}" >&2
      FAILED+=("${NAMES[$idx]}")
    fi
  done

  if (( ${#FAILED[@]} > 0 )); then
    printf 'Failed jobs:\n'
    printf '  %s\n' "${FAILED[@]}"
    exit 1
  fi
  exit 0
fi

FAILED=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r fuser_subdir base_model teacher_model run_id <<< "${job}"
  echo
  echo "============================================================"
  echo "Running ${run_id}"
  echo "============================================================"
  if run_one "${fuser_subdir}" "${base_model}" "${teacher_model}" "${run_id}"; then
    echo "[done] ${run_id}"
  else
    echo "[failed] ${run_id}" >&2
    FAILED+=("${run_id}")
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
done

if (( ${#FAILED[@]} > 0 )); then
  printf 'Failed jobs:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi

echo "Done."
