#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash bash/train/value_only_qwen3_4b_mmlu15k_gpu0_train_eval.sh

Environment:
  RUN_VARIANTS="original hiddenseed receiverseed sourcequery_receivermemory sharedspace"
  PER_DEVICE_BATCH_SIZE=1
  GRAD_ACCUM_STEPS=64
  NPROC=1
  EVAL_GPU_IDS="0"
  CUDA_VISIBLE_DEVICES=0
  DRY_RUN=1
EOF
  exit 0
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG_DIR=${LOG_DIR:-local/logs/value_only_qwen3_4b_mmlu15k_gpu0_train_eval}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/value_only_train_eval_${TIMESTAMP}.log}
TMP_CONFIG_DIR=${TMP_CONFIG_DIR:-local/tmp/value_only_qwen3_4b_mmlu15k_gpu0_train_eval}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29840}
NPROC=${NPROC:-1}
EVAL_GPU_IDS=${EVAL_GPU_IDS:-0}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-64}
RUN_VARIANTS=${RUN_VARIANTS:-original}
SKIP_TRAIN_IF_FINAL=${SKIP_TRAIN_IF_FINAL:-1}
SKIP_EVAL_IF_SUMMARY=${SKIP_EVAL_IF_SUMMARY:-0}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${LOG_DIR}" "${TMP_CONFIG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC=${NPROC}"
echo "EVAL_GPU_IDS=${EVAL_GPU_IDS}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "RUN_VARIANTS=${RUN_VARIANTS}"
echo "DRY_RUN=${DRY_RUN}"
echo "Start time: $(date)"

variant_train_config() {
  case "$1" in
    original)
      echo "recipe/train_recipe/C2C_original_qwen3_0.6b+qwen3_4b_MMLU_15k_2gpu.json"
      ;;
    hiddenseed)
      echo "recipe/train_recipe/C2C_kv_align_hiddenseed_sharedspace_residual_gate_0_qwen3_0.6b+qwen3_4b_MMLU_15k_2gpu_hfloss_acc32.json"
      ;;
    receiverseed)
      echo "recipe/train_recipe/C2C_kv_align_receiverseed_sharedspace_residual_gate_0_qwen3_0.6b+qwen3_4b_MMLU_15k_2gpu_hfloss_acc32.json"
      ;;
    sourcequery_receivermemory)
      echo "recipe/train_recipe/C2C_kv_align_sourcequery_receivermemory_sharedspace_residual_gate_0_qwen3_0.6b+qwen3_4b_MMLU_15k_2gpu_hfloss_acc32.json"
      ;;
    sharedspace)
      echo "recipe/train_recipe/C2C_kv_align_sharedspace_residual_gate_0_qwen3_0.6b+qwen3_4b_MMLU_15k_2gpu_hfloss_acc32.json"
      ;;
    *)
      echo "Unknown variant: $1" >&2
      return 1
      ;;
  esac
}

variant_eval_config() {
  case "$1" in
    original)
      echo "recipe/eval_recipe/mmlu_redux_original_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
      ;;
    hiddenseed)
      echo "recipe/eval_recipe/mmlu_redux_kv_align_hiddenseed_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu_hfloss_acc32.yaml"
      ;;
    receiverseed)
      echo "recipe/eval_recipe/mmlu_redux_kv_align_receiverseed_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu_hfloss_acc32.yaml"
      ;;
    sourcequery_receivermemory)
      echo "recipe/eval_recipe/mmlu_redux_kv_align_sourcequery_receivermemory_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu_hfloss_acc32.yaml"
      ;;
    sharedspace)
      echo "recipe/eval_recipe/mmlu_redux_kv_align_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu_hfloss_acc32.yaml"
      ;;
    *)
      echo "Unknown variant: $1" >&2
      return 1
      ;;
  esac
}

make_train_config() {
  local variant="$1"
  local src="$2"
  local dst="$3"
  python - "$variant" "$src" "$dst" "${PER_DEVICE_BATCH_SIZE}" "${GRAD_ACCUM_STEPS}" "${NPROC}" <<'PY'
import json
import sys

variant, src, dst, bs, accum, nproc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
with open(src) as f:
    cfg = json.load(f)

cfg["model"]["projector_value_only"] = True
cfg["training"]["num_processes"] = nproc
cfg["training"]["per_device_train_batch_size"] = bs
cfg["training"]["gradient_accumulation_steps"] = accum
cfg["training"]["loss_type"] = "hf_internal"

base_out = cfg["output"]["output_dir"].rstrip("/")
suffix = f"value_only_bs{bs}_acc{accum}_{nproc}gpu"
cfg["output"]["output_dir"] = f"{base_out}_{suffix}"
cfg["output"]["wandb_config"]["run_name"] = f"{cfg['output']['wandb_config']['run_name']}_{suffix}"
cfg["data"]["kwargs"]["cache_dir"] = cfg["data"]["kwargs"].get("cache_dir", "local/hf_datasets_cache/mmlu") + f"_{suffix}"

with open(dst, "w") as f:
    json.dump(cfg, f, indent=4)
PY
}

make_eval_config() {
  local src="$1"
  local dst="$2"
  local checkpoint_dir="$3"
  local variant="$4"
  python - "$src" "$dst" "$checkpoint_dir" "$variant" "${PER_DEVICE_BATCH_SIZE}" "${GRAD_ACCUM_STEPS}" "${NPROC}" ${EVAL_GPU_IDS} <<'PY'
import sys
import yaml

src, dst, checkpoint_dir, variant, bs, accum, nproc, *eval_gpu_ids = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)

suffix = f"value_only_bs{bs}_acc{accum}_{nproc}gpu"
cfg["model"]["rosetta_config"]["checkpoints_dir"] = checkpoint_dir.rstrip("/") + "/final"
cfg["eval"]["gpu_ids"] = [int(gpu_id) for gpu_id in eval_gpu_ids] or [0]
cfg["output"]["output_dir"] = f"local/final_results/qwen3_0.6b_qwen3_4b_{variant}_MMLU_15k_{suffix}_mmlu_redux"
cfg["eval"]["dataset_cache_dir"] = cfg["eval"].get("dataset_cache_dir", "local/hf_datasets_cache/mmlu_redux") + f"_{variant}_{suffix}"

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

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

for variant in ${RUN_VARIANTS}; do
  src_train="$(variant_train_config "${variant}")"
  src_eval="$(variant_eval_config "${variant}")"
  train_config="${TMP_CONFIG_DIR}/${variant}_value_only_train.json"
  eval_config="${TMP_CONFIG_DIR}/${variant}_value_only_eval.yaml"

  make_train_config "${variant}" "${src_train}" "${train_config}"
  checkpoint_dir="$(json_get_output_dir "${train_config}")"
  make_eval_config "${src_eval}" "${eval_config}" "${checkpoint_dir}" "${variant}"
  eval_output_dir="$(yaml_get_output_dir "${eval_config}")"

  echo
  echo "Train value-only variant: ${variant}"
  echo "Train config: ${train_config}"
  echo "Checkpoint dir: ${checkpoint_dir}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: generated configs only."
    continue
  fi

  if [[ "${SKIP_TRAIN_IF_FINAL}" == "1" && -d "${checkpoint_dir}/final" ]]; then
    echo "Skip training because ${checkpoint_dir}/final exists."
  else
    torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT_BASE}" script/train/SFT_train.py \
      --config "${train_config}"
  fi

  echo
  echo "Evaluate value-only variant: ${variant}"
  echo "Eval config: ${eval_config}"
  echo "Eval output dir: ${eval_output_dir}"

  if [[ "${SKIP_EVAL_IF_SUMMARY}" == "1" ]] && compgen -G "${eval_output_dir}/*summary.json" > /dev/null; then
    echo "Skip eval because summary exists in ${eval_output_dir}."
  else
    python script/evaluation/unified_evaluator_value_only.py --config "${eval_config}"
  fi
done

echo
echo "Done: $(date)"
