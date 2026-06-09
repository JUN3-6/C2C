#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint_final_dir> [run_name]" >&2
  echo "Example: GPU_IDS=1 $0 local/checkpoints/MY_RUN/final MY_RUN_source_mismatch" >&2
  exit 1
fi

CHECKPOINT_DIR="$1"
RUN_NAME="${2:-$(basename "$(dirname "$CHECKPOINT_DIR")")_source_mismatch}"
SAFE_RUN_NAME="${RUN_NAME//\//_}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-4B}"
GPU_IDS="${GPU_IDS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MISMATCH_OFFSET="${MISMATCH_OFFSET:-1}"
MISMATCH_PADDING_SIDE="${MISMATCH_PADDING_SIDE:-left}"
USE_TEMPLATE="${USE_TEMPLATE:-true}"
USE_COT="${USE_COT:-false}"
IS_DO_ALIGNMENT="${IS_DO_ALIGNMENT:-false}"
INCLUDE_RESPONSE="${INCLUDE_RESPONSE:-false}"
MULTI_SOURCE_FUSION_MODE="${MULTI_SOURCE_FUSION_MODE:-parallel}"
OUTPUT_DIR="${OUTPUT_DIR:-local/final_results/${SAFE_RUN_NAME}_mmlu_redux}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-local/hf_datasets_cache/mmlu_redux_mismatch_eval}"
LOG_DIR="${LOG_DIR:-local/logs/eval}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$DATASET_CACHE_DIR"

CONFIG_PATH="${LOG_DIR}/${SAFE_RUN_NAME}_mmlu_redux_source_mismatch.yaml"
LOG_PATH="${LOG_DIR}/${SAFE_RUN_NAME}_mmlu_redux_source_mismatch_$(date +%Y%m%d_%H%M%S).log"

export CHECKPOINT_DIR BASE_MODEL TEACHER_MODEL GPU_IDS MAX_NEW_TOKENS
export MISMATCH_OFFSET MISMATCH_PADDING_SIDE USE_TEMPLATE USE_COT
export IS_DO_ALIGNMENT INCLUDE_RESPONSE MULTI_SOURCE_FUSION_MODE
export OUTPUT_DIR DATASET_CACHE_DIR

python - "$CONFIG_PATH" <<'PY'
import os
import sys
import yaml

def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

def gpu_ids(value: str):
    return [int(x) for x in value.replace(" ", "").split(",") if x != ""]

config_path = sys.argv[1]
config = {
    "model": {
        "model_name": "Rosetta",
        "rosetta_config": {
            "base_model": os.environ["BASE_MODEL"],
            "teacher_model": os.environ["TEACHER_MODEL"],
            "is_do_alignment": as_bool(os.environ["IS_DO_ALIGNMENT"]),
            "alignment_strategy": "longest",
            "checkpoints_dir": os.environ["CHECKPOINT_DIR"],
            "include_response": as_bool(os.environ["INCLUDE_RESPONSE"]),
            "multi_source_fusion_mode": os.environ["MULTI_SOURCE_FUSION_MODE"],
        },
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
        },
    },
    "output": {
        "output_dir": os.environ["OUTPUT_DIR"],
    },
    "eval": {
        "dataset": "mmlu-redux",
        "gpu_ids": gpu_ids(os.environ["GPU_IDS"]),
        "answer_method": "generate",
        "use_cot": as_bool(os.environ["USE_COT"]),
        "use_template": as_bool(os.environ["USE_TEMPLATE"]),
        "sample_interval": 1,
        "dataset_cache_dir": os.environ["DATASET_CACHE_DIR"],
        "source_prompt_mismatch": True,
        "source_prompt_mismatch_offset": int(os.environ["MISMATCH_OFFSET"]),
        "source_prompt_mismatch_padding_side": os.environ["MISMATCH_PADDING_SIDE"],
        "math_grading_method": "comprehensive",
    },
}

with open(config_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY

echo "Config: $CONFIG_PATH"
echo "Log:    $LOG_PATH"
echo "Output: $OUTPUT_DIR"

python script/evaluation/unified_evaluator.py --config "$CONFIG_PATH" 2>&1 | tee "$LOG_PATH"
