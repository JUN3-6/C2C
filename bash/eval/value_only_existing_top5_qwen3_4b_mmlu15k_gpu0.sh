#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG_DIR=${LOG_DIR:-local/logs/value_only_existing_top5_qwen3_4b_mmlu15k_gpu0}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/value_only_eval_${TIMESTAMP}.log}
TMP_CONFIG_DIR=${TMP_CONFIG_DIR:-local/tmp/value_only_existing_top5_qwen3_4b_mmlu15k_gpu0}
SKIP_EVAL_IF_SUMMARY=${SKIP_EVAL_IF_SUMMARY:-0}

mkdir -p "${LOG_DIR}" "${TMP_CONFIG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Start time: $(date)"

NAMES=(
  "original_hfloss_acc32"
  "hiddenseed_hfloss_acc32"
  "receiverseed_hfloss_acc32"
  "sourcequery_receivermemory_hfloss_acc32"
  "sharedspace_hfloss_acc32"
)

EVAL_CONFIGS=(
  "recipe/eval_recipe/mmlu_redux_original_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu.yaml"
  "recipe/eval_recipe/mmlu_redux_kv_align_hiddenseed_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu_hfloss_acc32.yaml"
  "recipe/eval_recipe/mmlu_redux_kv_align_receiverseed_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu_hfloss_acc32.yaml"
  "recipe/eval_recipe/mmlu_redux_kv_align_sourcequery_receivermemory_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu_hfloss_acc32.yaml"
  "recipe/eval_recipe/mmlu_redux_kv_align_sharedspace_residual_gate_0_qwen3_0.6b_qwen3_4b_mmlu_15k_2gpu_hfloss_acc32.yaml"
)

make_value_only_config() {
  local src="$1"
  local dst="$2"
  python - "$src" "$dst" <<'PY'
import sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = yaml.safe_load(f)

cfg["eval"]["gpu_ids"] = [0]
cfg["output"]["output_dir"] = cfg["output"]["output_dir"].rstrip("/") + "_value_only_eval"
cfg["eval"]["dataset_cache_dir"] = cfg["eval"].get("dataset_cache_dir", "local/hf_datasets_cache/mmlu_redux") + "_value_only"

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

yaml_get() {
  python - "$1" "$2" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

cur = cfg
for part in sys.argv[2].split("."):
    cur = cur[part]
print(cur)
PY
}

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  src_config="${EVAL_CONFIGS[$i]}"
  value_config="${TMP_CONFIG_DIR}/${name}_value_only.yaml"
  make_value_only_config "${src_config}" "${value_config}"

  checkpoint_dir="$(yaml_get "${value_config}" "model.rosetta_config.checkpoints_dir")"
  output_dir="$(yaml_get "${value_config}" "output.output_dir")"

  echo
  echo "[$((i + 1))/${#NAMES[@]}] Value-only eval: ${name}"
  echo "Config: ${value_config}"
  echo "Checkpoint: ${checkpoint_dir}"
  echo "Output: ${output_dir}"

  if [[ ! -f "${checkpoint_dir}/projector_config.json" ]]; then
    echo "Skip: missing ${checkpoint_dir}/projector_config.json"
    continue
  fi

  if [[ "${SKIP_EVAL_IF_SUMMARY}" == "1" ]] && compgen -G "${output_dir}/*summary.json" > /dev/null; then
    echo "Skip: summary already exists in ${output_dir}"
    continue
  fi

  python script/evaluation/unified_evaluator_value_only.py --config "${value_config}"
done

echo
echo "Value-only eval summaries:"
find local/final_results -maxdepth 2 -path '*value_only_eval/*summary.json' -print | sort | while read -r summary; do
  python - "$summary" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
acc = data.get("overall_accuracy")
if acc is None:
    acc = data.get("accuracy")
print(f"{acc:.4f}\t{path}" if isinstance(acc, (float, int)) else f"NA\t{path}")
PY
done

echo "Done: $(date)"
