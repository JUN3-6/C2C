#!/usr/bin/env bash
set -euo pipefail

: "${FUSER_SUBDIR:?Set FUSER_SUBDIR, e.g. qwen3_0.6b+qwen3_4b_Fuser}"
: "${BASE_MODEL:?Set BASE_MODEL, e.g. Qwen/Qwen3-0.6B}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL, e.g. Qwen/Qwen3-4B}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

HF_REPO=${HF_REPO:-nics-efc/C2C_Fuser}
HF_LOCAL_DIR=${HF_LOCAL_DIR:-local/hf_models/C2C_Fuser}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${HF_LOCAL_DIR}/${FUSER_SUBDIR}/final}
KV_SWAP_SEED=${KV_SWAP_SEED:-0}
EVAL_GPU_IDS=${EVAL_GPU_IDS:-0,1}
RUN_ID=${RUN_ID:-$(python - <<'PY'
import os, re
name = os.environ["FUSER_SUBDIR"].replace("+", "_")
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
PY
)}

NORMAL_RESULT_DIR=${NORMAL_RESULT_DIR:-local/final_results/${RUN_ID}_mmlu_redux}
SWAP_RESULT_DIR=${SWAP_RESULT_DIR:-local/final_results/${RUN_ID}_kv_content_global_exact_seed${KV_SWAP_SEED}_mmlu_redux}
COMPARE_JSON=${COMPARE_JSON:-${SWAP_RESULT_DIR}/normal_vs_kv_content_swap_latest.json}
FORCE_RERUN=${FORCE_RERUN:-0}
CONFIG_DIR=${CONFIG_DIR:-local/tmp/hf_fuser_semantic_transfer_configs}
NORMAL_CONFIG=${NORMAL_CONFIG:-${CONFIG_DIR}/${RUN_ID}_normal.yaml}
SWAP_CONFIG=${SWAP_CONFIG:-${CONFIG_DIR}/${RUN_ID}_kv_content_global_exact_seed${KV_SWAP_SEED}.yaml}

LOG_DIR=${LOG_DIR:-local/logs/mmlu_redux_hf_fuser_semantic_transfer}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_ID}_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}" "${CONFIG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "EVAL_GPU_IDS=${EVAL_GPU_IDS}"
echo "HF_REPO=${HF_REPO}"
echo "FUSER_SUBDIR=${FUSER_SUBDIR}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TEACHER_MODEL=${TEACHER_MODEL}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "RUN_ID=${RUN_ID}"
echo "COMPARE_JSON=${COMPARE_JSON}"
echo "FORCE_RERUN=${FORCE_RERUN}"

if [[ "${FORCE_RERUN}" != "1" && -f "${COMPARE_JSON}" ]]; then
  echo "[skip] Comparison JSON already exists: ${COMPARE_JSON}"
  echo "Set FORCE_RERUN=1 to rerun this fuser."
  exit 0
fi

if [[ ! -f "${CHECKPOINT_DIR}/projector_config.json" ]]; then
  echo "[download] Fetching ${HF_REPO}/${FUSER_SUBDIR} into ${HF_LOCAL_DIR}"
  python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${HF_REPO}",
    allow_patterns=["${FUSER_SUBDIR}/*"],
    local_dir="${HF_LOCAL_DIR}",
)
PY
fi

if [[ ! -f "${CHECKPOINT_DIR}/projector_config.json" ]]; then
  echo "Missing ${CHECKPOINT_DIR}/projector_config.json after download" >&2
  exit 1
fi
FUSER_CONFIG_PATH=${FUSER_CONFIG_PATH:-${HF_LOCAL_DIR}/${FUSER_SUBDIR}/config.json}
if [[ ! -f "${FUSER_CONFIG_PATH}" ]]; then
  echo "Missing ${FUSER_CONFIG_PATH} after download" >&2
  exit 1
fi

export BASE_MODEL TEACHER_MODEL CHECKPOINT_DIR NORMAL_RESULT_DIR SWAP_RESULT_DIR NORMAL_CONFIG SWAP_CONFIG
export KV_SWAP_SEED EVAL_GPU_IDS RUN_ID FUSER_CONFIG_PATH
python - <<'PY'
import json
import os
import yaml

def gpu_ids():
    return [int(part.strip()) for part in os.environ["EVAL_GPU_IDS"].split(",") if part.strip()]

def base_config(output_dir, cache_suffix):
    with open(os.environ["FUSER_CONFIG_PATH"]) as f:
        fuser_config = json.load(f)
    fuser_model_config = fuser_config.get("model", {})
    rosetta_config = {
        "base_model": os.environ["BASE_MODEL"],
        "teacher_model": os.environ["TEACHER_MODEL"],
        "is_do_alignment": bool(fuser_model_config.get("is_do_alignment", False)),
        "alignment_strategy": fuser_model_config.get("alignment_strategy", "first"),
        "checkpoints_dir": os.environ["CHECKPOINT_DIR"],
        "include_response": bool(fuser_model_config.get("include_response", False)),
    }
    if fuser_model_config.get("multi_source_fusion_mode") is not None:
        rosetta_config["multi_source_fusion_mode"] = fuser_model_config["multi_source_fusion_mode"]

    return {
        "model": {
            "model_name": "Rosetta",
            "rosetta_config": rosetta_config,
            "generation_config": {
                "do_sample": False,
                "max_new_tokens": 64,
            },
        },
        "output": {
            "output_dir": output_dir,
        },
        "eval": {
            "dataset": "mmlu-redux",
            "gpu_ids": gpu_ids(),
            "answer_method": "generate",
            "use_cot": False,
            "use_template": True,
            "sample_interval": 1,
            "dataset_cache_dir": f"local/hf_datasets_cache/{cache_suffix}",
            "math_grading_method": "comprehensive",
        },
    }

run_id = os.environ["RUN_ID"]
normal = base_config(os.environ["NORMAL_RESULT_DIR"], f"mmlu_redux_{run_id}_eval")
swap = base_config(os.environ["SWAP_RESULT_DIR"], f"mmlu_redux_{run_id}_kv_content_global_exact_seed{os.environ['KV_SWAP_SEED']}_eval")
swap["eval"]["kv_content_ablation"] = {
    "enabled": True,
    "mode": "global_exact",
    "seed": int(os.environ["KV_SWAP_SEED"]),
    "skip_unmatched": True,
    "require_label_mismatch": False,
}

for path, cfg in [
    (os.environ["NORMAL_CONFIG"], normal),
    (os.environ["SWAP_CONFIG"], swap),
]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote {path}")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, stopping before evaluation."
  exit 0
fi

echo "[1/3] Normal C2C fuser MMLU-Redux eval"
python script/evaluation/unified_evaluator_choice_shuffle.py --config "${NORMAL_CONFIG}"

echo "[2/3] KV-content ablation with same-length donor sharer KV"
python script/evaluation/unified_evaluator_kv_content_ablation.py --config "${SWAP_CONFIG}"

echo "[3/3] Paired normal-vs-swap comparison"
python script/analysis/compare_normal_vs_kv_content_ablation.py \
  --normal-dir "${NORMAL_RESULT_DIR}" \
  --swap-dir "${SWAP_RESULT_DIR}" \
  --output-json "${COMPARE_JSON}"

echo "[summary] Swap-only donor diagnostics"
python script/analysis/analyze_kv_content_ablation_results.py \
  --result "${RUN_ID}_global_exact" "${SWAP_RESULT_DIR}" \
  --output-json "${SWAP_RESULT_DIR}/kv_content_ablation_summary_latest.json"

echo "Done."
