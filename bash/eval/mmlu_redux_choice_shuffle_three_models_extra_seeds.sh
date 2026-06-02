#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG_DIR=${LOG_DIR:-local/logs/mmlu_redux_choice_shuffle_three_models_extra_seeds}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/choice_shuffle_three_models_extra_seeds_${TIMESTAMP}.log}
CONFIG_TMP_DIR=${CONFIG_TMP_DIR:-local/tmp/mmlu_redux_choice_shuffle_extra_seed_configs}
SEEDS=${SEEDS:-"1 2"}

mkdir -p "${LOG_DIR}" "${CONFIG_TMP_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SEEDS=${SEEDS}"
echo "Temporary configs: ${CONFIG_TMP_DIR}"

ORIGINAL_CONFIG=${ORIGINAL_CONFIG:-recipe/eval_recipe/mmlu_redux_choice_shuffle_original_c2c_mmlu_15k.yaml}
RECEIVER_SEED_CONFIG=${RECEIVER_SEED_CONFIG:-recipe/eval_recipe/mmlu_redux_choice_shuffle_kv_align_receiverseed_sharedspace_residual_mmlu_15k_gate_0_bs16_2gpu.yaml}
SHARER_QUERY_CONFIG=${SHARER_QUERY_CONFIG:-recipe/eval_recipe/mmlu_redux_choice_shuffle_kv_align_sharerquery_receivermemory_sharedspace_residual_mmlu_15k_gate_0_bs16_2gpu.yaml}

make_seed_config() {
  local base_config="$1"
  local seed="$2"
  python - "${base_config}" "${seed}" "${CONFIG_TMP_DIR}" <<'PY'
import pathlib
import sys
import yaml

base_config = pathlib.Path(sys.argv[1])
seed = int(sys.argv[2])
out_dir = pathlib.Path(sys.argv[3])

with base_config.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("eval", {}).setdefault("choice_shuffle", {})["enabled"] = True
cfg["eval"]["choice_shuffle"]["seed"] = seed

out_path = out_dir / f"{base_config.stem}_seed{seed}.yaml"
with out_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(out_path)
PY
}

run_eval() {
  local label="$1"
  local base_config="$2"
  local seed="$3"
  local seed_config
  seed_config=$(make_seed_config "${base_config}" "${seed}")
  echo "[${label}] choice-shuffle seed=${seed}"
  echo "  config=${seed_config}"
  python script/evaluation/unified_evaluator_choice_shuffle.py --config "${seed_config}"
}

for seed in ${SEEDS}; do
  echo "===== choice-shuffle seed ${seed} ====="
  run_eval "original_c2c" "${ORIGINAL_CONFIG}" "${seed}"
  run_eval "receiver_seed" "${RECEIVER_SEED_CONFIG}" "${seed}"
  run_eval "sharer_query" "${SHARER_QUERY_CONFIG}" "${seed}"
done

echo "[summary] All choice-shuffle CSVs in the shared result folders"
python script/analysis/analyze_choice_shuffle_results.py --all \
  --result original_c2c local/final_results/0.6+0.5B_C2C_original_MMLU_15k_choice_shuffle_seed0_mmlu_redux \
  --result receiver_seed local/final_results/0.6+0.5B_C2C_kv_align_receiverseed_sharedspace_residual_MMLU_15k_gate_0_bs16_2gpu_choice_shuffle_seed0_mmlu_redux \
  --result sharer_query local/final_results/0.6+0.5B_C2C_kv_align_sharerquery_receivermemory_sharedspace_residual_MMLU_15k_gate_0_bs16_2gpu_choice_shuffle_seed0_mmlu_redux

echo "Done."
