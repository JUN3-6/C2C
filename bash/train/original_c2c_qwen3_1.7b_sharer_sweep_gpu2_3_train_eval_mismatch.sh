#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MASTER_PORT_BASE=${MASTER_PORT_BASE:-29920}
LOG_DIR=${LOG_DIR:-local/logs/original_c2c_qwen3_1.7b_sharer_sweep_gpu2_3_train_eval_mismatch}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
MAIN_LOG=${MAIN_LOG:-${LOG_DIR}/main_${TIMESTAMP}.log}
TMP_CONFIG_DIR=${TMP_CONFIG_DIR:-local/tmp/original_c2c_qwen3_1.7b_sharer_sweep_gpu2_3_train_eval_mismatch/${TIMESTAMP}}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-64}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-16}
EVAL_ONLY=${EVAL_ONLY:-0}
SKIP_TRAIN_IF_FINAL=${SKIP_TRAIN_IF_FINAL:-1}
SKIP_EVAL_IF_SUMMARY=${SKIP_EVAL_IF_SUMMARY:-0}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${LOG_DIR}" "${TMP_CONFIG_DIR}"
exec > >(tee -a "${MAIN_LOG}") 2>&1

NAMES=(
  "qwen3_4b_to_qwen3_1.7b_original"
  "qwen2.5_3b_to_qwen3_1.7b_original"
)

GPUS=(
  "2"
  "3"
)

TRAIN_CONFIGS=(
  "recipe/train_recipe/C2C_original_qwen3_1.7b+qwen3_4b_MMLU_15k_gpu2_1gpu.json"
  "recipe/train_recipe/C2C_original_qwen3_1.7b+qwen2.5_3b_instruct_MMLU_15k_gpu3_1gpu.json"
)

EVAL_CONFIGS=(
  "recipe/eval_recipe/mmlu_redux_original_qwen3_1.7b_qwen3_4b_mmlu_15k_gpu2_1gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_original_qwen3_1.7b_qwen2.5_3b_instruct_mmlu_15k_gpu3_1gpu.yaml"
)

MISMATCH_CONFIGS=(
  "recipe/eval_recipe/mmlu_redux_sharer_mismatch_original_qwen3_1.7b_qwen3_4b_mmlu_15k_gpu2_1gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_sharer_mismatch_original_qwen3_1.7b_qwen2.5_3b_instruct_mmlu_15k_gpu3_1gpu.yaml"
)

json_get_output_dir() {
  python - "$1" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    print(json.load(f)["output"]["output_dir"])
PY
}

yaml_get_output_dir() {
  python - "$1" <<'PY'
import sys
import yaml
with open(sys.argv[1]) as f:
    print(yaml.safe_load(f)["output"]["output_dir"])
PY
}

materialize_configs() {
  local idx="$1"
  local gpu="$2"
  local train_config="$3"
  local eval_config="$4"
  local mismatch_config="$5"
  local out_dir="${TMP_CONFIG_DIR}/job${idx}_gpu${gpu}"
  mkdir -p "${out_dir}"

  python - \
    "${train_config}" \
    "${eval_config}" \
    "${mismatch_config}" \
    "${out_dir}" \
    "${gpu}" \
    "${PER_DEVICE_BATCH_SIZE}" \
    "${GRAD_ACCUM_STEPS}" \
    "${EVAL_MAX_NEW_TOKENS}" <<'PY'
import json
import re
import sys
from pathlib import Path

import yaml

train_path, eval_path, mismatch_path, out_dir, gpu, batch_size, grad_accum, eval_max_new_tokens = sys.argv[1:9]
batch_size_i = int(batch_size)
grad_accum_i = int(grad_accum)
eval_max_new_tokens_i = int(eval_max_new_tokens)
out = Path(out_dir)


def apply_suffix(text: str) -> str:
    text = re.sub(r"_bs\d+_acc\d+_", f"_bs{batch_size_i}_acc{grad_accum_i}_", text)
    text = re.sub(r"_gpu\d+_1gpu", f"_gpu{gpu}_1gpu", text)
    return text


def apply_eval_suffix(text: str) -> str:
    text = apply_suffix(text)
    if f"_gen{eval_max_new_tokens_i}" not in text:
        text = f"{text}_gen{eval_max_new_tokens_i}"
    return text


with open(train_path) as f:
    train = json.load(f)

train["training"]["per_device_train_batch_size"] = batch_size_i
train["training"]["gradient_accumulation_steps"] = grad_accum_i
train["training"]["num_processes"] = 1

old_train_output = train["output"]["output_dir"]
new_train_output = apply_suffix(old_train_output)
train["output"]["output_dir"] = new_train_output

wandb_config = train.get("output", {}).get("wandb_config", {})
if "run_name" in wandb_config:
    wandb_config["run_name"] = apply_suffix(wandb_config["run_name"])

data_kwargs = train.get("data", {}).get("kwargs", {})
if "cache_dir" in data_kwargs:
    data_kwargs["cache_dir"] = apply_suffix(data_kwargs["cache_dir"])


def rewrite_eval(path: str, suffix: str):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["rosetta_config"]["checkpoints_dir"] = f"{new_train_output}/final"
    cfg.setdefault("model", {}).setdefault("generation_config", {})
    cfg["model"]["generation_config"]["do_sample"] = False
    cfg["model"]["generation_config"]["max_new_tokens"] = eval_max_new_tokens_i
    cfg["eval"]["gpu_ids"] = [0]
    cfg["eval"]["max_new_tokens"] = eval_max_new_tokens_i
    cfg["output"]["output_dir"] = apply_eval_suffix(cfg["output"]["output_dir"])
    if "dataset_cache_dir" in cfg["eval"]:
        cfg["eval"]["dataset_cache_dir"] = apply_suffix(cfg["eval"]["dataset_cache_dir"])
    dest = out / suffix
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return dest


train_dest = out / "train.json"
with open(train_dest, "w") as f:
    json.dump(train, f, indent=4)

eval_dest = rewrite_eval(eval_path, "eval.yaml")
mismatch_dest = rewrite_eval(mismatch_path, "mismatch.yaml")

print(train_dest)
print(eval_dest)
print(mismatch_dest)
PY
}

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_one() {
  local idx="$1"
  local name="${NAMES[$idx]}"
  local gpu="${GPUS[$idx]}"
  local train_config="${TRAIN_CONFIGS[$idx]}"
  local eval_config="${EVAL_CONFIGS[$idx]}"
  local mismatch_config="${MISMATCH_CONFIGS[$idx]}"
  local master_port=$((MASTER_PORT_BASE + idx))
  local job_log="${LOG_DIR}/${name}_${TIMESTAMP}.log"
  local materialized
  local checkpoint_dir
  local eval_output_dir
  local mismatch_output_dir

  mapfile -t materialized < <(materialize_configs "${idx}" "${gpu}" "${train_config}" "${eval_config}" "${mismatch_config}")
  train_config="${materialized[0]}"
  eval_config="${materialized[1]}"
  mismatch_config="${materialized[2]}"

  checkpoint_dir="$(json_get_output_dir "${train_config}")"
  eval_output_dir="$(yaml_get_output_dir "${eval_config}")"
  mismatch_output_dir="$(yaml_get_output_dir "${mismatch_config}")"

  (
    set -euo pipefail
    exec > >(tee -a "${job_log}") 2>&1
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"

    echo "Job: ${name}"
    echo "Physical GPU: ${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
    echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
    echo "EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS}"
    echo "EVAL_ONLY=${EVAL_ONLY}"
    echo "Train config: ${train_config}"
    echo "Eval config: ${eval_config}"
    echo "Mismatch config: ${mismatch_config}"
    echo "Checkpoint dir: ${checkpoint_dir}"
    echo "Eval output dir: ${eval_output_dir}"
    echo "Mismatch output dir: ${mismatch_output_dir}"
    echo "Start time: $(date)"

    if [[ "${EVAL_ONLY}" == "1" ]]; then
      if [[ ! -d "${checkpoint_dir}/final" ]]; then
        echo "EVAL_ONLY=1 but checkpoint is missing: ${checkpoint_dir}/final" >&2
        exit 1
      fi
      echo "Skip training because EVAL_ONLY=1."
    elif [[ "${SKIP_TRAIN_IF_FINAL}" == "1" && -d "${checkpoint_dir}/final" ]]; then
      echo "Skip training because ${checkpoint_dir}/final exists."
    else
      run_or_print torchrun --nproc_per_node=1 --master_port="${master_port}" script/train/SFT_train.py \
        --config "${train_config}"
    fi

    if [[ "${SKIP_EVAL_IF_SUMMARY}" == "1" ]] && compgen -G "${eval_output_dir}/*summary.json" > /dev/null; then
      echo "Skip normal eval because summary exists in ${eval_output_dir}."
    else
      run_or_print python script/evaluation/unified_evaluator.py \
        --config "${eval_config}"
    fi

    if [[ "${SKIP_EVAL_IF_SUMMARY}" == "1" ]] && compgen -G "${mismatch_output_dir}/*summary.json" > /dev/null; then
      echo "Skip mismatch eval because summary exists in ${mismatch_output_dir}."
    else
      run_or_print python script/evaluation/unified_evaluator.py \
        --config "${mismatch_config}"
    fi

    echo "Done ${name}: $(date)"
  )
}

echo "Logging to ${MAIN_LOG}"
echo "Start time: $(date)"
echo "TMP_CONFIG_DIR=${TMP_CONFIG_DIR}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS}"
echo "EVAL_ONLY=${EVAL_ONLY}"
echo "SKIP_TRAIN_IF_FINAL=${SKIP_TRAIN_IF_FINAL}"
echo "SKIP_EVAL_IF_SUMMARY=${SKIP_EVAL_IF_SUMMARY}"
echo "DRY_RUN=${DRY_RUN}"

pids=()
for idx in "${!NAMES[@]}"; do
  run_one "${idx}" &
  pids+=("$!")
  echo "Started ${NAMES[$idx]} on physical GPU ${GPUS[$idx]} with PID ${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "One or more jobs failed." >&2
  exit "${status}"
fi

echo "All jobs completed: $(date)"
