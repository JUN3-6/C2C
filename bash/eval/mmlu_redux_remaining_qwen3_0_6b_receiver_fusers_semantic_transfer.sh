#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG_DIR=${LOG_DIR:-local/logs/mmlu_redux_remaining_qwen3_0_6b_receiver_fusers}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/remaining_qwen3_0_6b_receiver_fusers_${TIMESTAMP}.log}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
RUN_IN_PARALLEL=${RUN_IN_PARALLEL:-0}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
echo "RUN_IN_PARALLEL=${RUN_IN_PARALLEL}"

# Already covered separately:
# - qwen3_0.6b+qwen2.5_0.5b_Fuser
# - qwen3_0.6b+qwen3_4b_Fuser
JOBS=(
  "qwen3_0.6b+llam3.2_1b_Fuser|Qwen/Qwen3-0.6B|meta-llama/Llama-3.2-1B-Instruct|qwen3_0.6b_llama3.2_1b_OpenHermes_Fuser"
  "qwen3_0.6b+qwen2.5_1.5b_math_Fuser|Qwen/Qwen3-0.6B|Qwen/Qwen2.5-Math-1.5B-Instruct|qwen3_0.6b_qwen2.5_math_1.5b_OpenHermes_Fuser"
  "qwen3_0.6b+qwen3_4b_base_Fuser|Qwen/Qwen3-0.6B|Qwen/Qwen3-4B-Base|qwen3_0.6b_qwen3_4b_base_OpenHermes_Fuser"
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
    bash bash/eval/mmlu_redux_hf_fuser_semantic_transfer.sh
}

if [[ "${RUN_IN_PARALLEL}" == "1" ]]; then
  echo
  echo "Launching all remaining fuser experiments concurrently."
  echo "All jobs share CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; use only if GPU memory is sufficient."

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
    pid="${PIDS[$idx]}"
    name="${NAMES[$idx]}"
    if wait "${pid}"; then
      echo "[done] ${name}"
    else
      echo "[failed] ${name}" >&2
      FAILED+=("${name}")
    fi
  done

  if (( ${#FAILED[@]} > 0 )); then
    echo
    echo "Completed with failures:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
  fi

  echo
  echo "All remaining Qwen3-0.6B receiver fuser experiments completed."
  exit 0
fi

FAILED=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r fuser_subdir base_model teacher_model run_id <<< "${job}"
  echo
  echo "============================================================"
  echo "Running ${run_id}"
  echo "  fuser=${fuser_subdir}"
  echo "  base=${base_model}"
  echo "  teacher=${teacher_model}"
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
  echo
  echo "Completed with failures:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi

echo
echo "All remaining Qwen3-0.6B receiver fuser experiments completed."
