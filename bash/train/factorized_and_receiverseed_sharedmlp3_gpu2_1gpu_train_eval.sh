#!/usr/bin/env bash
set -euo pipefail

# Use physical GPU 2 only. Inside this process it is visible as cuda:0.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MASTER_PORT_BASE=${MASTER_PORT_BASE:-29820}
NPROC=${NPROC:-1}
LOG_DIR=${LOG_DIR:-local/logs/factorized_and_receiverseed_sharedmlp3_gpu2_1gpu_train_eval}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/factorized_and_receiverseed_sharedmlp3_gpu2_1gpu_train_eval_${TIMESTAMP}.log}
SKIP_TRAIN_IF_FINAL=${SKIP_TRAIN_IF_FINAL:-1}
SKIP_EVAL_IF_SUMMARY=${SKIP_EVAL_IF_SUMMARY:-0}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC=${NPROC}"
echo "SKIP_TRAIN_IF_FINAL=${SKIP_TRAIN_IF_FINAL}"
echo "SKIP_EVAL_IF_SUMMARY=${SKIP_EVAL_IF_SUMMARY}"
echo "Start time: $(date)"

NAMES=(
  "factorized_kv"
  "receiverseed_sharedmlp3"
)

TRAIN_CONFIGS=(
  "recipe/train_recipe/C2C_factorized_kv_qwen3_0.6b+qwen3_4b_MMLU_15k_hfloss_eff256_gpu2_1gpu.json"
  "recipe/train_recipe/C2C_kv_align_receiverseed_sharedspace_residual_gate_0_sharedmlp3_qwen3_0.6b+qwen3_4b_MMLU_15k_hfloss_eff256_gpu2_1gpu.json"
)

EVAL_CONFIGS=(
  "recipe/eval_recipe/mmlu_redux_factorized_kv_qwen3_0.6b_qwen3_4b_mmlu_15k_hfloss_eff256_gpu2_1gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_kv_align_receiverseed_sharedspace_residual_gate_0_sharedmlp3_qwen3_0.6b_qwen3_4b_mmlu_15k_hfloss_eff256_gpu2_1gpu.yaml"
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

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  train_config="${TRAIN_CONFIGS[$i]}"
  eval_config="${EVAL_CONFIGS[$i]}"
  master_port=$((MASTER_PORT_BASE + i))
  checkpoint_dir="$(json_get_output_dir "${train_config}")"
  eval_output_dir="$(yaml_get_output_dir "${eval_config}")"

  echo
  echo "[$((i + 1))/2] Train ${name}"
  echo "Config: ${train_config}"
  echo "Checkpoint dir: ${checkpoint_dir}"

  if [[ "${SKIP_TRAIN_IF_FINAL}" == "1" && -d "${checkpoint_dir}/final" ]]; then
    echo "Skip training because ${checkpoint_dir}/final exists."
  else
    torchrun --nproc_per_node="${NPROC}" --master_port="${master_port}" script/train/SFT_train.py \
      --config "${train_config}"
  fi

  echo
  echo "[$((i + 1))/2] Evaluate ${name} on mmlu-redux"
  echo "Config: ${eval_config}"
  echo "Eval output dir: ${eval_output_dir}"

  if [[ "${SKIP_EVAL_IF_SUMMARY}" == "1" ]] && compgen -G "${eval_output_dir}/*summary.json" > /dev/null; then
    echo "Skip eval because summary exists in ${eval_output_dir}."
  else
    python script/evaluation/unified_evaluator.py \
      --config "${eval_config}"
  fi
done

echo
echo "Done: $(date)"
