#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN_ID=${RUN_ID:-qwen3_0.6b_qwen3_4b_C2C_C_openhermes_500k}
TRAIN_TEMPLATE=${TRAIN_TEMPLATE:-recipe/train_recipe/C2C_C_qwen3_0.6b+qwen3_4b_openhermes_500k_8gpu.json}
TMP_DIR=${TMP_DIR:-local/tmp/${RUN_ID}}
TRAIN_CONFIG=${TRAIN_CONFIG:-${TMP_DIR}/train.json}
EVAL_CONFIG=${EVAL_CONFIG:-${TMP_DIR}/eval_mmlu_redux.yaml}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-local/checkpoints/${RUN_ID}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/final}
RESULT_DIR=${RESULT_DIR:-local/final_results/${RUN_ID}_mmlu_redux}
SUMMARY_SENTINEL=${SUMMARY_SENTINEL:-${RESULT_DIR}/summary.done.json}

LOG_DIR=${LOG_DIR:-local/logs/${RUN_ID}}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/train_eval_${RUN_ID}_${TIMESTAMP}.log}

FORCE_RERUN=${FORCE_RERUN:-0}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_EVAL=${RUN_EVAL:-1}
DRY_RUN=${DRY_RUN:-0}
MASTER_PORT=${MASTER_PORT:-29537}
MACRO_BATCH_SIZE=${MACRO_BATCH_SIZE:-256}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-2048}
NUM_SAMPLES=${NUM_SAMPLES:-500000}
WANDB_MODE=${WANDB_MODE:-disabled}
WANDB_PROJECT=${WANDB_PROJECT:-C2C}
WANDB_ENTITY=${WANDB_ENTITY:-}
LOSS_CHUNK_SIZE=${LOSS_CHUNK_SIZE:-128}

mkdir -p "${LOG_DIR}" "${TMP_DIR}" "${RESULT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

visible_gpu_count() {
  python - <<'PY'
import os
visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
print(len(visible))
PY
}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(visible_gpu_count)}
EVAL_GPU_IDS=${EVAL_GPU_IDS:-$(python - <<'PY'
import os
n = len([part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()])
print(",".join(str(i) for i in range(max(n, 1))))
PY
)}

GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$(python - <<PY
macro = int("${MACRO_BATCH_SIZE}")
per_device = int("${PER_DEVICE_TRAIN_BATCH_SIZE}")
nproc = int("${NPROC_PER_NODE}")
den = per_device * nproc
if den <= 0:
    raise SystemExit("per-device batch and nproc must be positive")
if macro % den != 0:
    raise SystemExit(f"MACRO_BATCH_SIZE={macro} is not divisible by PER_DEVICE_TRAIN_BATCH_SIZE*NPROC_PER_NODE={den}")
print(macro // den)
PY
)}

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "EVAL_GPU_IDS=${EVAL_GPU_IDS}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "SUMMARY_SENTINEL=${SUMMARY_SENTINEL}"
echo "FORCE_RERUN=${FORCE_RERUN}"

if [[ "${FORCE_RERUN}" != "1" && -f "${SUMMARY_SENTINEL}" ]]; then
  echo "[skip] Summary sentinel exists: ${SUMMARY_SENTINEL}"
  echo "Set FORCE_RERUN=1 to rerun."
  exit 0
fi

export TRAIN_TEMPLATE TRAIN_CONFIG EVAL_CONFIG CHECKPOINT_ROOT CHECKPOINT_DIR RESULT_DIR
export RUN_ID NPROC_PER_NODE PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
export MAX_LENGTH NUM_SAMPLES WANDB_MODE WANDB_PROJECT WANDB_ENTITY EVAL_GPU_IDS
export LOSS_CHUNK_SIZE

python - <<'PY'
import json
import os
import yaml

def gpu_ids():
    return [int(part.strip()) for part in os.environ["EVAL_GPU_IDS"].split(",") if part.strip()]

with open(os.environ["TRAIN_TEMPLATE"]) as f:
    train_cfg = json.load(f)

train_cfg["training"]["num_processes"] = int(os.environ["NPROC_PER_NODE"])
train_cfg["training"]["per_device_train_batch_size"] = int(os.environ["PER_DEVICE_TRAIN_BATCH_SIZE"])
train_cfg["training"]["gradient_accumulation_steps"] = int(os.environ["GRADIENT_ACCUMULATION_STEPS"])
train_cfg["training"]["max_length"] = int(os.environ["MAX_LENGTH"])
train_cfg["training"]["loss_chunk_size"] = int(os.environ["LOSS_CHUNK_SIZE"])
train_cfg["output"]["output_dir"] = os.environ["CHECKPOINT_ROOT"]
train_cfg["output"]["wandb_config"]["mode"] = os.environ["WANDB_MODE"]
train_cfg["output"]["wandb_config"]["project"] = os.environ["WANDB_PROJECT"]
train_cfg["output"]["wandb_config"]["entity"] = os.environ["WANDB_ENTITY"] or None
train_cfg["output"]["wandb_config"]["run_name"] = os.environ["RUN_ID"]
train_cfg["data"]["kwargs"]["num_samples"] = int(os.environ["NUM_SAMPLES"])
train_cfg["data"]["kwargs"]["max_word_count"] = int(os.environ["MAX_LENGTH"])

os.makedirs(os.path.dirname(os.environ["TRAIN_CONFIG"]), exist_ok=True)
with open(os.environ["TRAIN_CONFIG"], "w") as f:
    json.dump(train_cfg, f, indent=2)
print(f"Wrote {os.environ['TRAIN_CONFIG']}")

eval_cfg = {
    "model": {
        "model_name": "Rosetta",
        "rosetta_config": {
            "base_model": "Qwen/Qwen3-0.6B",
            "teacher_model": "Qwen/Qwen3-4B",
            "is_do_alignment": False,
            "alignment_strategy": "first",
            "checkpoints_dir": os.environ["CHECKPOINT_DIR"],
            "include_response": False,
        },
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
        "dataset_cache_dir": "local/hf_datasets_cache/mmlu_redux_c2c_c_qwen3_0_6b_qwen3_4b_openhermes_500k_eval",
        "math_grading_method": "comprehensive",
    },
}
with open(os.environ["EVAL_CONFIG"], "w") as f:
    yaml.safe_dump(eval_cfg, f, sort_keys=False)
print(f"Wrote {os.environ['EVAL_CONFIG']}")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, stopping after config generation."
  exit 0
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  if [[ "${FORCE_RERUN}" != "1" && -f "${CHECKPOINT_DIR}/projector_config.json" ]]; then
    echo "[skip] Final checkpoint already exists: ${CHECKPOINT_DIR}"
  else
    echo "[1/2] Train C2C-C fuser on OpenHermes 500k"
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
      script/train/SFT_train.py --config "${TRAIN_CONFIG}"
  fi
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  if [[ ! -f "${CHECKPOINT_DIR}/projector_config.json" ]]; then
    echo "Missing ${CHECKPOINT_DIR}/projector_config.json; cannot evaluate." >&2
    exit 1
  fi

  echo "[2/2] Evaluate C2C-C fuser on MMLU-Redux"
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
fi

echo "Done."
