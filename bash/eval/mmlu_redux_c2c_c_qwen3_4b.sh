#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

RUN_ID=${RUN_ID:-qwen3_0.6b_qwen3_4b_C2C_C_openhermes_500k}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-0.6B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-4B}
HF_REPO=${HF_REPO:-nics-efc/C2C_Fuser}
HF_LOCAL_DIR=${HF_LOCAL_DIR:-local/hf_models/C2C_Fuser}
HF_FUSER_SUBDIR=${HF_FUSER_SUBDIR:-}

if [[ -n "${HF_FUSER_SUBDIR}" ]]; then
  CHECKPOINT_DIR=${CHECKPOINT_DIR:-${HF_LOCAL_DIR}/${HF_FUSER_SUBDIR}/final}
else
  CHECKPOINT_DIR=${CHECKPOINT_DIR:-local/checkpoints/${RUN_ID}/final}
fi

RESULT_DIR=${RESULT_DIR:-local/final_results/${RUN_ID}_mmlu_redux}
SUMMARY_SENTINEL=${SUMMARY_SENTINEL:-${RESULT_DIR}/summary.done.json}
TMP_DIR=${TMP_DIR:-local/tmp/${RUN_ID}}
EVAL_CONFIG=${EVAL_CONFIG:-${TMP_DIR}/eval_mmlu_redux.yaml}

LOG_DIR=${LOG_DIR:-local/logs/${RUN_ID}}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/eval_${RUN_ID}_${TIMESTAMP}.log}

FORCE_RERUN=${FORCE_RERUN:-0}
DRY_RUN=${DRY_RUN:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
EVAL_GPU_IDS=${EVAL_GPU_IDS:-$(python - <<'PY'
import os
n = len([part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()])
print(",".join(str(i) for i in range(max(n, 1))))
PY
)}

mkdir -p "${LOG_DIR}" "${TMP_DIR}" "${RESULT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "EVAL_GPU_IDS=${EVAL_GPU_IDS}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TEACHER_MODEL=${TEACHER_MODEL}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "SUMMARY_SENTINEL=${SUMMARY_SENTINEL}"
echo "FORCE_RERUN=${FORCE_RERUN}"

if [[ "${FORCE_RERUN}" != "1" && -f "${SUMMARY_SENTINEL}" ]]; then
  echo "[skip] Summary sentinel exists: ${SUMMARY_SENTINEL}"
  echo "Set FORCE_RERUN=1 to rerun."
  exit 0
fi

if [[ -n "${HF_FUSER_SUBDIR}" && ! -f "${CHECKPOINT_DIR}/projector_config.json" ]]; then
  echo "[download] Fetching ${HF_REPO}/${HF_FUSER_SUBDIR} into ${HF_LOCAL_DIR}"
  python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${HF_REPO}",
    allow_patterns=["${HF_FUSER_SUBDIR}/*"],
    local_dir="${HF_LOCAL_DIR}",
)
PY
fi

if [[ ! -f "${CHECKPOINT_DIR}/projector_config.json" ]]; then
  echo "Missing ${CHECKPOINT_DIR}/projector_config.json" >&2
  exit 1
fi

export BASE_MODEL TEACHER_MODEL CHECKPOINT_DIR RESULT_DIR EVAL_CONFIG EVAL_GPU_IDS MAX_NEW_TOKENS
python - <<'PY'
import os
import yaml

def gpu_ids():
    return [int(part.strip()) for part in os.environ["EVAL_GPU_IDS"].split(",") if part.strip()]

cfg = {
    "model": {
        "model_name": "Rosetta",
        "rosetta_config": {
            "base_model": os.environ["BASE_MODEL"],
            "teacher_model": os.environ["TEACHER_MODEL"],
            "is_do_alignment": False,
            "alignment_strategy": "first",
            "checkpoints_dir": os.environ["CHECKPOINT_DIR"],
            "include_response": False,
        },
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
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
        "dataset_cache_dir": "local/hf_datasets_cache/mmlu_redux_c2c_c_qwen3_0_6b_qwen3_4b_eval",
        "math_grading_method": "comprehensive",
    },
}

os.makedirs(os.path.dirname(os.environ["EVAL_CONFIG"]), exist_ok=True)
with open(os.environ["EVAL_CONFIG"], "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Wrote {os.environ['EVAL_CONFIG']}")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, stopping after config generation."
  exit 0
fi

python script/evaluation/unified_evaluator.py --config "${EVAL_CONFIG}"

SUMMARY_FILE=$(python - <<PY
from pathlib import Path
result_dir = Path("${RESULT_DIR}")
matches = sorted(result_dir.glob("*mmlu-redux_generate*_summary.json"), key=lambda p: p.stat().st_mtime)
print(matches[-1] if matches else "")
PY
)
if [[ -z "${SUMMARY_FILE}" ]]; then
  echo "No summary JSON found under ${RESULT_DIR}" >&2
  exit 1
fi

python - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
sentinel = Path("${SUMMARY_SENTINEL}")
sentinel.parent.mkdir(parents=True, exist_ok=True)
with sentinel.open("w") as f:
    json.dump({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "summary_file": "${SUMMARY_FILE}",
        "checkpoint_dir": "${CHECKPOINT_DIR}",
        "eval_config": "${EVAL_CONFIG}",
    }, f, indent=2)
print(f"Wrote {sentinel}")
PY

echo "Done."
