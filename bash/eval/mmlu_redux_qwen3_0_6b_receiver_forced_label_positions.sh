#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B}
RUN_ID=${RUN_ID:-qwen3_0.6b_receiver_only}
TARGET_LABELS=${TARGET_LABELS:-"A B C D"}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
SUMMARY_OUT=${SUMMARY_OUT:-local/final_results/qwen3_0.6b_receiver_only_forced_label_positions_summary_latest.json}

LOG_DIR=${LOG_DIR:-local/logs/mmlu_redux_qwen3_0_6b_receiver_forced_label_positions}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/qwen3_0_6b_receiver_forced_label_positions_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "RUN_ID=${RUN_ID}"
echo "TARGET_LABELS=${TARGET_LABELS}"
echo "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
echo "SUMMARY_OUT=${SUMMARY_OUT}"

FAILED=()
for target_label in ${TARGET_LABELS}; do
  echo
  echo "============================================================"
  echo "Running receiver-only ${RUN_ID}, forced ${target_label}"
  echo "============================================================"
  if MODEL_NAME="${MODEL_NAME}" \
    RUN_ID="${RUN_ID}" \
    TARGET_LABEL="${target_label}" \
    bash bash/eval/mmlu_redux_hf_receiver_forced_label.sh; then
    echo "[done] ${RUN_ID} forced ${target_label}"
  else
    echo "[failed] ${RUN_ID} forced ${target_label}" >&2
    FAILED+=("${RUN_ID}:forced_${target_label}")
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
done

SUMMARY_ARGS=()
for target_label in ${TARGET_LABELS}; do
  SUMMARY_ARGS+=(--result "${RUN_ID}_forced_${target_label}" "local/final_results/${RUN_ID}_forced_${target_label}_mmlu_redux")
done

echo
echo "[summary] Receiver-only forced-label result summary"
python script/analysis/analyze_forced_label_results.py \
  "${SUMMARY_ARGS[@]}" \
  --output-json "${SUMMARY_OUT}"

if (( ${#FAILED[@]} > 0 )); then
  printf 'Failed jobs:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi

echo "Done."
