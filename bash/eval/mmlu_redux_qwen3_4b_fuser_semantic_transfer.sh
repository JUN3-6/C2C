#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG_DIR=${LOG_DIR:-local/logs/mmlu_redux_qwen3_4b_fuser_semantic_transfer}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/qwen3_4b_fuser_semantic_transfer_${TIMESTAMP}.log}

HF_REPO=${HF_REPO:-nics-efc/C2C_Fuser}
HF_SUBDIR=${HF_SUBDIR:-qwen3_0.6b+qwen3_4b_Fuser}
HF_LOCAL_DIR=${HF_LOCAL_DIR:-local/hf_models/C2C_Fuser}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${HF_LOCAL_DIR}/${HF_SUBDIR}/final}

NORMAL_CONFIG=${NORMAL_CONFIG:-recipe/eval_recipe/mmlu_redux_qwen3_0_6b_qwen3_4b_fuser.yaml}
SWAP_CONFIG=${SWAP_CONFIG:-recipe/eval_recipe/mmlu_redux_kv_content_ablation_global_exact_qwen3_0_6b_qwen3_4b_fuser.yaml}
NORMAL_RESULT_DIR=${NORMAL_RESULT_DIR:-local/final_results/qwen3_0.6b_qwen3_4b_C2C_Fuser_mmlu_redux}
SWAP_RESULT_DIR=${SWAP_RESULT_DIR:-local/final_results/qwen3_0.6b_qwen3_4b_C2C_Fuser_kv_content_global_exact_seed0_mmlu_redux}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HF_REPO=${HF_REPO}"
echo "HF_SUBDIR=${HF_SUBDIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"

if [[ ! -f "${CHECKPOINT_DIR}/projector_config.json" ]]; then
  echo "[download] Fetching ${HF_REPO}/${HF_SUBDIR} into ${HF_LOCAL_DIR}"
  python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${HF_REPO}",
    allow_patterns=["${HF_SUBDIR}/*"],
    local_dir="${HF_LOCAL_DIR}",
)
PY
fi

if [[ ! -f "${CHECKPOINT_DIR}/projector_config.json" ]]; then
  echo "Missing ${CHECKPOINT_DIR}/projector_config.json after download" >&2
  exit 1
fi

echo "[1/3] Normal Qwen3-0.6B + Qwen3-4B C2C fuser MMLU-Redux eval"
python script/evaluation/unified_evaluator_choice_shuffle.py --config "${NORMAL_CONFIG}"

echo "[2/3] KV-content ablation with same-length donor sharer KV"
python script/evaluation/unified_evaluator_kv_content_ablation.py --config "${SWAP_CONFIG}"

echo "[3/3] Paired normal-vs-swap comparison"
python script/analysis/compare_normal_vs_kv_content_ablation.py \
  --normal-dir "${NORMAL_RESULT_DIR}" \
  --swap-dir "${SWAP_RESULT_DIR}" \
  --output-json "${SWAP_RESULT_DIR}/normal_vs_kv_content_swap_latest.json"

echo "[summary] Swap-only donor diagnostics"
python script/analysis/analyze_kv_content_ablation_results.py \
  --result qwen3_4b_fuser_global_exact "${SWAP_RESULT_DIR}" \
  --output-json "${SWAP_RESULT_DIR}/kv_content_ablation_summary_latest.json"

echo "Done."
