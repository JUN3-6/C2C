#!/usr/bin/env bash

set -euo pipefail

export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONFIG_PATH="${ROOT_DIR}/local/checkpoints/0.6+0.5B_C2C_general_again/config.json"
OUTPUT_DIR="/home/june/workspace/C2C_routing/local/router_runs/0.6+0.5B_C2C_general_again_router"
DEVICE="cuda:0"
DTYPE="bfloat16"
MAX_LENGTH="2048"
SHARD_SIZE="4000"
TRAIN_RATIO="0.90"
EPOCHS="10"
BATCH_SIZE="256"
HIDDEN_DIM="1024"
LOG_EVERY="10"
LR_SCHEDULER_TYPE="constant"
LR_WARMUP_RATIO="0.05"
LR_WARMUP_STEPS="-1"
LR_MIN_RATIO="0.1"
GAIN_LOSS_WEIGHT="0.1"
ACTION_CLASS_WEIGHT_MODE="none"
ACTION_CLASS_EFFECTIVE_NUM_BETA="0.9999"
ACTION_CLASS_MAX_WEIGHT="20.0"
ACTION_CLASS_SKIP_MULTIPLIER="1.0"
FUSE_THRESHOLD="0.5"
SELECTION_TEMPERATURE="1.0"
BEST_METRIC="mean_routed_gain"
ROUTER_FEATURE_SOURCE="hidden"
SEED="42"
NUM_SAMPLES_OVERRIDE=""
USE_WANDB="1"
WANDB_PROJECT="C2C"
WANDB_ENTITY="june6-hanyang-university"
WANDB_MODE="online"
WANDB_RUN_NAME=""
WANDB_TAGS=()
DEFAULT_BANK_DIRS=(
  "${ROOT_DIR}/local/checkpoints/0.6+0.5B_C2C_general_again/final"
)
BANK_DIRS=("${DEFAULT_BANK_DIRS[@]}")
CUSTOM_BANK_DIRS="0"

usage() {
  echo "Usage: $0 --output-dir <dir> [options]"
  echo
  echo "Options:"
  echo "  --config <path>          Train-style config JSON/YAML."
  echo "  --bank-dir <path>        Projector bank directory (required). Can be provided multiple times."
  echo "  --output-dir <dir>       Output directory for shards, dataset, split, and router."
  echo "  --device <device>        Device for label generation and training. Default: ${DEVICE}"
  echo "  --dtype <dtype>          Model dtype. Default: ${DTYPE}"
  echo "  --max-length <int>       Max sequence length. Default: ${MAX_LENGTH}"
  echo "  --shard-size <int>       Label generation shard size. Default: ${SHARD_SIZE}"
  echo "  --num-samples <int>      Override data.kwargs.num_samples from config."
  echo "  --train-ratio <float>    Train split ratio. Default: ${TRAIN_RATIO}"
  echo "  --epochs <int>           Router training epochs. Default: ${EPOCHS}"
  echo "  --batch-size <int>       Router training batch size. Default: ${BATCH_SIZE}"
  echo "  --hidden-dim <int>       Router hidden dimension. Default: ${HIDDEN_DIM}"
  echo "  --log-every <int>        W&B step logging interval during router training. Default: ${LOG_EVERY}"
  echo "  --lr-scheduler-type <m>  LR scheduler: constant, linear, cosine. Default: ${LR_SCHEDULER_TYPE}"
  echo "  --lr-warmup-ratio <f>    Warmup ratio over total steps. Default: ${LR_WARMUP_RATIO}"
  echo "  --lr-warmup-steps <int>  Warmup steps override (-1 uses ratio). Default: ${LR_WARMUP_STEPS}"
  echo "  --lr-min-ratio <f>       Final LR ratio for linear/cosine. Default: ${LR_MIN_RATIO}"
  echo "  --gain-loss-weight <f>   Weight for gain-aware objective. Default: ${GAIN_LOSS_WEIGHT}"
  echo "  --action-class-weight-mode <m>  Action CE class weighting: none, inverse, effective_num. Default: ${ACTION_CLASS_WEIGHT_MODE}"
  echo "  --action-class-effective-num-beta <f>  Beta for effective_num mode. Default: ${ACTION_CLASS_EFFECTIVE_NUM_BETA}"
  echo "  --action-class-max-weight <f>   Max clip for class weights (<=0 disables). Default: ${ACTION_CLASS_MAX_WEIGHT}"
  echo "  --action-class-skip-multiplier <f>  Extra multiplier for skip class. Default: ${ACTION_CLASS_SKIP_MULTIPLIER}"
  echo "  --fuse-threshold <f>     Threshold for hard fuse decision in val metrics. Default: ${FUSE_THRESHOLD}"
  echo "  --selection-temperature <f>  Softmax temperature for gain objective. Default: ${SELECTION_TEMPERATURE}"
  echo "  --best-metric <name>     Checkpoint metric: loss, mean_routed_gain, gain_capture, auto. Default: ${BEST_METRIC}"
  echo "  --router-feature-source <s>  Router feature source: kv, hidden, projector_in, projector_in_stats, projector_in_pooled, or projector_in_binned. Default: ${ROUTER_FEATURE_SOURCE}"
  echo "  --seed <int>             Split seed. Default: ${SEED}"
  echo "  --wandb                  Enable W&B logging for router training (default: enabled)."
  echo "  --no-wandb               Disable W&B logging."
  echo "  --wandb-project <name>   Override W&B project."
  echo "  --wandb-entity <name>    Override W&B entity."
  echo "  --wandb-mode <mode>      W&B mode: online, offline, or disabled."
  echo "  --wandb-run-name <name>  Override W&B run name."
  echo "  --wandb-tag <tag>        Repeatable W&B tag."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --bank-dir)
      if [[ "${CUSTOM_BANK_DIRS}" == "0" ]]; then
        BANK_DIRS=()
        CUSTOM_BANK_DIRS="1"
      fi
      BANK_DIRS+=("$2")
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --max-length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --shard-size)
      SHARD_SIZE="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES_OVERRIDE="$2"
      shift 2
      ;;
    --train-ratio)
      TRAIN_RATIO="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --hidden-dim)
      HIDDEN_DIM="$2"
      shift 2
      ;;
    --log-every)
      LOG_EVERY="$2"
      shift 2
      ;;
    --lr-scheduler-type)
      LR_SCHEDULER_TYPE="$2"
      shift 2
      ;;
    --lr-warmup-ratio)
      LR_WARMUP_RATIO="$2"
      shift 2
      ;;
    --lr-warmup-steps)
      LR_WARMUP_STEPS="$2"
      shift 2
      ;;
    --lr-min-ratio)
      LR_MIN_RATIO="$2"
      shift 2
      ;;
    --gain-loss-weight)
      GAIN_LOSS_WEIGHT="$2"
      shift 2
      ;;
    --action-class-weight-mode)
      ACTION_CLASS_WEIGHT_MODE="$2"
      shift 2
      ;;
    --action-class-effective-num-beta)
      ACTION_CLASS_EFFECTIVE_NUM_BETA="$2"
      shift 2
      ;;
    --action-class-max-weight)
      ACTION_CLASS_MAX_WEIGHT="$2"
      shift 2
      ;;
    --action-class-skip-multiplier)
      ACTION_CLASS_SKIP_MULTIPLIER="$2"
      shift 2
      ;;
    --fuse-threshold)
      FUSE_THRESHOLD="$2"
      shift 2
      ;;
    --selection-temperature)
      SELECTION_TEMPERATURE="$2"
      shift 2
      ;;
    --best-metric)
      BEST_METRIC="$2"
      shift 2
      ;;
    --router-feature-source)
      ROUTER_FEATURE_SOURCE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --wandb)
      USE_WANDB="1"
      shift
      ;;
    --no-wandb)
      USE_WANDB="0"
      shift
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --wandb-run-name)
      WANDB_RUN_NAME="$2"
      shift 2
      ;;
    --wandb-tag)
      WANDB_TAGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "--output-dir is required." >&2
  usage >&2
  exit 1
fi

if [[ ${#BANK_DIRS[@]} -eq 0 ]]; then
  echo "At least one --bank-dir is required." >&2
  usage >&2
  exit 1
fi

if [[ "${USE_WANDB}" == "1" ]]; then
  readarray -t WANDB_DEFAULTS < <(
    cd "${ROOT_DIR}" && python - <<'PY' "${CONFIG_PATH}"
import json
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as f:
    if config_path.suffix == ".json":
        cfg = json.load(f)
    else:
        cfg = yaml.safe_load(f)

wandb_cfg = cfg.get("output", {}).get("wandb_config", {})
print(wandb_cfg.get("project", "C2C"))
print(wandb_cfg.get("entity", ""))
print(wandb_cfg.get("run_name", "router_training"))
PY
  )

  if [[ -z "${WANDB_PROJECT}" ]]; then
    WANDB_PROJECT="${WANDB_DEFAULTS[0]}"
  fi
  if [[ -z "${WANDB_ENTITY}" ]]; then
    WANDB_ENTITY="${WANDB_DEFAULTS[1]}"
  fi
  if [[ -z "${WANDB_RUN_NAME}" ]]; then
    WANDB_RUN_NAME="${WANDB_DEFAULTS[2]}_router"
  fi
  if [[ -z "${WANDB_MODE}" ]]; then
    WANDB_MODE="online"
  fi
fi

mkdir -p "${OUTPUT_DIR}/shards" "${OUTPUT_DIR}/logs"

LABEL_EXTRA_ARGS=()
if [[ -n "${NUM_SAMPLES_OVERRIDE}" ]]; then
  LABEL_EXTRA_ARGS+=( --num-samples-override "${NUM_SAMPLES_OVERRIDE}" )
fi

echo "Computing filtered dataset size..."
TOTAL_COUNT="$(
  cd "${ROOT_DIR}" && python - <<'PY' "${CONFIG_PATH}" "${MAX_LENGTH}" "${NUM_SAMPLES_OVERRIDE}" | tail -n 1
import json
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from rosetta.model.aligner import AlignmentStrategy, TokenAligner
from rosetta.train.dataset_adapters import (
    AlignedChatDataset,
    ChatDataset,
    create_dataset,
)
from rosetta.utils.evaluate import set_default_chat_template

config_path = Path(sys.argv[1])
max_length = int(sys.argv[2])
num_samples_override = sys.argv[3]
with config_path.open("r", encoding="utf-8") as f:
    if config_path.suffix == ".json":
        cfg = json.load(f)
    else:
        cfg = yaml.safe_load(f)

model_cfg = cfg["model"]
data_cfg = cfg["data"]
if num_samples_override:
    data_cfg = dict(data_cfg)
    kwargs = dict(data_cfg.get("kwargs", {}))
    kwargs["num_samples"] = int(num_samples_override)
    data_cfg["kwargs"] = kwargs
message_dataset = create_dataset(data_cfg["type"], **data_cfg.get("kwargs", {}))

base_tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
set_default_chat_template(base_tokenizer, model_cfg["base_model"])

if model_cfg.get("is_do_alignment", False):
    teacher_tokenizer = AutoTokenizer.from_pretrained(model_cfg["teacher_model"])
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
    set_default_chat_template(teacher_tokenizer, model_cfg["teacher_model"])
    aligner = TokenAligner(
        slm_tokenizer=base_tokenizer,
        llm_tokenizer=teacher_tokenizer,
        strategy=AlignmentStrategy(model_cfg.get("alignment_strategy", "first")),
    )
    dataset = AlignedChatDataset(message_dataset, aligner, max_length=max_length)
else:
    dataset = ChatDataset(message_dataset, base_tokenizer, max_length=max_length)

print(len(dataset))
PY
)"

echo "Filtered supervised dataset size: ${TOTAL_COUNT}"

START_INDEX=0
while [[ "${START_INDEX}" -lt "${TOTAL_COUNT}" ]]; do
  END_INDEX=$((START_INDEX + SHARD_SIZE))
  if [[ "${END_INDEX}" -gt "${TOTAL_COUNT}" ]]; then
    END_INDEX="${TOTAL_COUNT}"
  fi

  SHARD_PATH="${OUTPUT_DIR}/shards/router_labels_${START_INDEX}_${END_INDEX}.pt"
  LOG_PATH="${OUTPUT_DIR}/logs/labels_${START_INDEX}_${END_INDEX}.log"

  if [[ -f "${SHARD_PATH}" ]]; then
    echo "Skipping existing shard ${SHARD_PATH}"
  else
    echo "Generating shard ${START_INDEX}:${END_INDEX}"
    cd "${ROOT_DIR}"
    /usr/bin/time -f "elapsed=%E maxrss=%MKB" \
      python script/train/generate_router_labels.py \
        --config "${CONFIG_PATH}" \
        --projector-bank-dirs "${BANK_DIRS[@]}" \
        --output-path "${SHARD_PATH}" \
        --start-index "${START_INDEX}" \
        --end-index "${END_INDEX}" \
        --device "${DEVICE}" \
        --dtype "${DTYPE}" \
        --max-length "${MAX_LENGTH}" \
        --router-feature-source "${ROUTER_FEATURE_SOURCE}" \
        "${LABEL_EXTRA_ARGS[@]}" \
      2>&1 | tee "${LOG_PATH}"
  fi

  START_INDEX="${END_INDEX}"
done

cd "${ROOT_DIR}"
SHARD_FILES=( "${OUTPUT_DIR}"/shards/router_labels_*.pt )
echo "Building consolidated router dataset..."
python script/train/build_router_dataset.py \
  --input-shards "${SHARD_FILES[@]}" \
  --output-path "${OUTPUT_DIR}/router_dataset_full.pt" \
  2>&1 | tee "${OUTPUT_DIR}/logs/build_dataset.log"

echo "Splitting train/val dataset..."
python - <<'PY' "${OUTPUT_DIR}/router_dataset_full.pt" "${OUTPUT_DIR}" "${TRAIN_RATIO}" "${SEED}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/split_dataset.log"
import math
import sys
from pathlib import Path

import torch

dataset_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
train_ratio = float(sys.argv[3])
seed = int(sys.argv[4])

payload = torch.load(dataset_path, map_location="cpu")
num_examples = payload["pooled_feature"].shape[0]
num_train = max(1, min(num_examples - 1, int(math.floor(num_examples * train_ratio))))

generator = torch.Generator().manual_seed(seed)
perm = torch.randperm(num_examples, generator=generator)
train_idx = perm[:num_train]
val_idx = perm[num_train:]

def subset(data, indices):
    subset_payload = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            subset_payload[key] = value.index_select(0, indices)
        elif isinstance(value, list):
            subset_payload[key] = [value[int(i)] for i in indices.tolist()]
        else:
            subset_payload[key] = value
    return subset_payload

train_payload = subset(payload, train_idx)
val_payload = subset(payload, val_idx)

torch.save(train_payload, output_dir / "router_dataset_train.pt")
torch.save(val_payload, output_dir / "router_dataset_val.pt")
torch.save(
    {
        "train_indices": train_idx,
        "val_indices": val_idx,
        "num_examples": num_examples,
        "train_ratio": train_ratio,
        "seed": seed,
    },
    output_dir / "router_dataset_split_meta.pt",
)

print(
    {
        "num_examples": num_examples,
        "num_train": int(train_idx.numel()),
        "num_val": int(val_idx.numel()),
        "train_ratio": train_ratio,
        "seed": seed,
    }
)
PY

TRAIN_ROUTER_ARGS=(
  --train-data "${OUTPUT_DIR}/router_dataset_train.pt"
  --val-data "${OUTPUT_DIR}/router_dataset_val.pt"
  --output-dir "${OUTPUT_DIR}/router"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --hidden-dim "${HIDDEN_DIM}"
  --log-every "${LOG_EVERY}"
  --lr-scheduler-type "${LR_SCHEDULER_TYPE}"
  --lr-warmup-ratio "${LR_WARMUP_RATIO}"
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --lr-min-ratio "${LR_MIN_RATIO}"
  --gain-loss-weight "${GAIN_LOSS_WEIGHT}"
  --action-class-weight-mode "${ACTION_CLASS_WEIGHT_MODE}"
  --action-class-effective-num-beta "${ACTION_CLASS_EFFECTIVE_NUM_BETA}"
  --action-class-max-weight "${ACTION_CLASS_MAX_WEIGHT}"
  --action-class-skip-multiplier "${ACTION_CLASS_SKIP_MULTIPLIER}"
  --fuse-threshold "${FUSE_THRESHOLD}"
  --selection-temperature "${SELECTION_TEMPERATURE}"
  --best-metric "${BEST_METRIC}"
  --router-feature-source "${ROUTER_FEATURE_SOURCE}"
  --device "${DEVICE}"
)

if [[ "${USE_WANDB}" == "1" ]]; then
  TRAIN_ROUTER_ARGS+=(
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-mode "${WANDB_MODE}"
    --wandb-run-name "${WANDB_RUN_NAME}"
  )
  if [[ -n "${WANDB_ENTITY}" ]]; then
    TRAIN_ROUTER_ARGS+=( --wandb-entity "${WANDB_ENTITY}" )
  fi
  for tag in "${WANDB_TAGS[@]}"; do
    TRAIN_ROUTER_ARGS+=( --wandb-tag "${tag}" )
  done
fi

echo "Training router..."
python script/train/train_router.py \
  "${TRAIN_ROUTER_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/train_router.log"

echo "Router pipeline complete: ${OUTPUT_DIR}"
