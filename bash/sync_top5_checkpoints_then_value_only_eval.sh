#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-oem@166.104.35.125}
PORT=${PORT:-20022}
REMOTE_ROOT=${REMOTE_ROOT:-/data/kjh/C2C}
LOG_DIR=${LOG_DIR:-local/logs/sync_top5_checkpoints_then_value_only_eval}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/sync_eval_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}" local/checkpoints
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "REMOTE=${REMOTE}"
echo "PORT=${PORT}"
echo "REMOTE_ROOT=${REMOTE_ROOT}"
echo "Start time: $(date)"

CHECKPOINT_DIRS=(
  "qwen3_0.6b_qwen3_4b_C2C_original_repro_MMLU_15k_bs4_acc32_2gpu"
  "qwen3_0.6b_qwen3_4b_C2C_kv_align_hiddenseed_sharedspace_residual_MMLU_15k_gate_0_hfloss_acc32_bs4_2gpu"
  "qwen3_0.6b_qwen3_4b_C2C_kv_align_receiverseed_sharedspace_residual_MMLU_15k_gate_0_hfloss_acc32_bs4_2gpu"
  "qwen3_0.6b_qwen3_4b_C2C_kv_align_sourcequery_receivermemory_sharedspace_residual_MMLU_15k_gate_0_hfloss_acc32_bs4_2gpu"
  "qwen3_0.6b_qwen3_4b_C2C_kv_align_sharedspace_residual_MMLU_15k_gate_0_hfloss_acc32_bs4_2gpu"
)

echo
echo "Fetching checkpoint final directories from remote..."
ssh -p "${PORT}" "${REMOTE}" "REMOTE_ROOT='${REMOTE_ROOT}' bash -s" <<'REMOTE_SCRIPT' | tar -xf -
set -euo pipefail
cd "${REMOTE_ROOT}"

dirs=(
  "qwen3_0.6b_qwen3_4b_C2C_original_repro_MMLU_15k_bs4_acc32_2gpu"
  "qwen3_0.6b_qwen3_4b_C2C_kv_align_hiddenseed_sharedspace_residual_MMLU_15k_gate_0_hfloss_acc32_bs4_2gpu"
  "qwen3_0.6b_qwen3_4b_C2C_kv_align_receiverseed_sharedspace_residual_MMLU_15k_gate_0_hfloss_acc32_bs4_2gpu"
  "qwen3_0.6b_qwen3_4b_C2C_kv_align_sourcequery_receivermemory_sharedspace_residual_MMLU_15k_gate_0_hfloss_acc32_bs4_2gpu"
  "qwen3_0.6b_qwen3_4b_C2C_kv_align_sharedspace_residual_MMLU_15k_gate_0_hfloss_acc32_bs4_2gpu"
)

paths=()
for d in "${dirs[@]}"; do
  ckpt="local/checkpoints/${d}"
  if [[ -d "${ckpt}/final" ]]; then
    echo "[remote] include ${ckpt}/final" >&2
    [[ -f "${ckpt}/config.json" ]] && paths+=("${ckpt}/config.json")
    paths+=("${ckpt}/final")
  else
    echo "[remote] missing ${ckpt}/final" >&2
  fi
done

if [[ "${#paths[@]}" -eq 0 ]]; then
  echo "[remote] no checkpoint paths found" >&2
  exit 42
fi

tar -cf - "${paths[@]}"
REMOTE_SCRIPT

echo
echo "Local checkpoint status:"
missing=0
for d in "${CHECKPOINT_DIRS[@]}"; do
  path="local/checkpoints/${d}/final/projector_config.json"
  if [[ -f "${path}" ]]; then
    echo "OK      ${path}"
  else
    echo "MISSING ${path}"
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "Some checkpoints are still missing; value-only eval will skip those entries."
fi

echo
echo "Running value-only eval..."
SKIP_EVAL_IF_SUMMARY=${SKIP_EVAL_IF_SUMMARY:-0} \
  bash bash/eval/value_only_existing_top5_qwen3_4b_mmlu15k_gpu0.sh

echo
echo "Done: $(date)"
