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
SHUFFLE_SEED=${SHUFFLE_SEED:-0}
EVAL_GPU_IDS=${EVAL_GPU_IDS:-$(python - <<'PY'
import os

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
count = len([part for part in visible.split(",") if part.strip()])
print(",".join(str(i) for i in range(max(count, 1))))
PY
)}
RUN_ID=${RUN_ID:-$(python - <<'PY'
import os, re
name = os.environ["FUSER_SUBDIR"].replace("+", "_")
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
PY
)}

RESULT_DIR=${RESULT_DIR:-local/final_results/${RUN_ID}_choice_shuffle_seed${SHUFFLE_SEED}_mmlu_redux}
SUMMARY_SENTINEL=${SUMMARY_SENTINEL:-${RESULT_DIR}/choice_shuffle_summary_latest.txt}
FORCE_RERUN=${FORCE_RERUN:-0}
CONFIG_DIR=${CONFIG_DIR:-local/tmp/hf_fuser_choice_shuffle_configs}
CONFIG=${CONFIG:-${CONFIG_DIR}/${RUN_ID}_choice_shuffle_seed${SHUFFLE_SEED}.yaml}

LOG_DIR=${LOG_DIR:-local/logs/mmlu_redux_hf_fuser_choice_shuffle}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_ID}_choice_shuffle_seed${SHUFFLE_SEED}_${TIMESTAMP}.log}

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
echo "RESULT_DIR=${RESULT_DIR}"
echo "SUMMARY_SENTINEL=${SUMMARY_SENTINEL}"
echo "FORCE_RERUN=${FORCE_RERUN}"

if [[ "${FORCE_RERUN}" != "1" && -f "${SUMMARY_SENTINEL}" ]]; then
  echo "[skip] Choice-shuffle summary already exists: ${SUMMARY_SENTINEL}"
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

export BASE_MODEL TEACHER_MODEL CHECKPOINT_DIR RESULT_DIR CONFIG SHUFFLE_SEED EVAL_GPU_IDS RUN_ID FUSER_CONFIG_PATH
python - <<'PY'
import json
import os
import yaml

def gpu_ids():
    return [int(part.strip()) for part in os.environ["EVAL_GPU_IDS"].split(",") if part.strip()]

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

cfg = {
    "model": {
        "model_name": "Rosetta",
        "rosetta_config": rosetta_config,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 64,
        },
    },
    "output": {
        "output_dir": os.environ["RESULT_DIR"],
    },
    "eval": {
        "dataset": "mmlu-redux",
        "gpu_ids": gpu_ids(),
        "answer_method": "generate",
        "use_cot": False,
        "use_template": True,
        "sample_interval": 1,
        "dataset_cache_dir": f"local/hf_datasets_cache/mmlu_redux_{os.environ['RUN_ID']}_choice_shuffle_seed{os.environ['SHUFFLE_SEED']}_eval",
        "choice_shuffle": {
            "enabled": True,
            "seed": int(os.environ["SHUFFLE_SEED"]),
        },
        "math_grading_method": "comprehensive",
    },
}

os.makedirs(os.path.dirname(os.environ["CONFIG"]), exist_ok=True)
with open(os.environ["CONFIG"], "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Wrote {os.environ['CONFIG']}")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, stopping before evaluation."
  exit 0
fi

echo "[1/2] Choice-shuffle eval"
python script/evaluation/unified_evaluator_choice_shuffle.py --config "${CONFIG}"

echo "[2/2] Choice-shuffle summary"
mkdir -p "${RESULT_DIR}"
python script/analysis/analyze_choice_shuffle_results.py \
  --result "${RUN_ID}_choice_shuffle" "${RESULT_DIR}" | tee "${SUMMARY_SENTINEL}"

echo "Done."
