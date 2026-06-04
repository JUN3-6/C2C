#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_NAME:?Set MODEL_NAME, e.g. Qwen/Qwen3-0.6B}"
: "${TARGET_LABEL:?Set TARGET_LABEL to one of A, B, C, D}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

EVAL_GPU_IDS=${EVAL_GPU_IDS:-$(python - <<'PY'
import os

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
count = len([part for part in visible.split(",") if part.strip()])
print(",".join(str(i) for i in range(max(count, 1))))
PY
)}
RUN_ID=${RUN_ID:-$(python - <<'PY'
import os, re
name = os.environ["MODEL_NAME"].replace("/", "_")
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
PY
)}
TARGET_LABEL=$(echo "${TARGET_LABEL}" | tr '[:lower:]' '[:upper:]')

RESULT_DIR=${RESULT_DIR:-local/final_results/${RUN_ID}_forced_${TARGET_LABEL}_mmlu_redux}
SUMMARY_SENTINEL=${SUMMARY_SENTINEL:-${RESULT_DIR}/forced_label_summary_latest.txt}
FORCE_RERUN=${FORCE_RERUN:-0}
CONFIG_DIR=${CONFIG_DIR:-local/tmp/hf_receiver_forced_label_configs}
CONFIG=${CONFIG:-${CONFIG_DIR}/${RUN_ID}_forced_${TARGET_LABEL}.yaml}

LOG_DIR=${LOG_DIR:-local/logs/mmlu_redux_hf_receiver_forced_label}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_ID}_forced_${TARGET_LABEL}_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}" "${CONFIG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "EVAL_GPU_IDS=${EVAL_GPU_IDS}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "TARGET_LABEL=${TARGET_LABEL}"
echo "RUN_ID=${RUN_ID}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "SUMMARY_SENTINEL=${SUMMARY_SENTINEL}"
echo "FORCE_RERUN=${FORCE_RERUN}"

case "${TARGET_LABEL}" in
  A|B|C|D) ;;
  *) echo "TARGET_LABEL must be A, B, C, or D" >&2; exit 1 ;;
esac

if [[ "${FORCE_RERUN}" != "1" && -f "${SUMMARY_SENTINEL}" ]]; then
  echo "[skip] Forced-label summary already exists: ${SUMMARY_SENTINEL}"
  echo "Set FORCE_RERUN=1 to rerun this target."
  exit 0
fi

export MODEL_NAME RESULT_DIR CONFIG TARGET_LABEL EVAL_GPU_IDS RUN_ID
python - <<'PY'
import os
import yaml


def gpu_ids():
    return [int(part.strip()) for part in os.environ["EVAL_GPU_IDS"].split(",") if part.strip()]


target_label = os.environ["TARGET_LABEL"]
cfg = {
    "model": {
        "model_name": os.environ["MODEL_NAME"],
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
        "dataset_cache_dir": (
            f"local/hf_datasets_cache/mmlu_redux_{os.environ['RUN_ID']}"
            f"_forced_{target_label}_eval"
        ),
        "forced_label": {
            "enabled": True,
            "target_label": target_label,
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

echo "[1/2] Receiver-only forced-label eval: target ${TARGET_LABEL}"
python script/evaluation/unified_evaluator_forced_label.py --config "${CONFIG}"

echo "[2/2] Forced-label summary"
mkdir -p "${RESULT_DIR}"
python script/analysis/analyze_forced_label_results.py \
  --result "${RUN_ID}_forced_${TARGET_LABEL}" "${RESULT_DIR}" | tee "${SUMMARY_SENTINEL}"

echo "Done."
