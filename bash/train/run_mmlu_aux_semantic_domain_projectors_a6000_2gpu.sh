#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ENV_NAME="${ENV_NAME:-route}"
GPU_IDS="${GPU_IDS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

RECIPES=(
  "recipe/train_recipe/C2C_0.6+0.5_mmlu_aux_0_15k_semantic_p1_a6000_2gpu_bs8.json"
  "recipe/train_recipe/C2C_0.6+0.5_mmlu_aux_0_15k_semantic_p2_a6000_2gpu_bs8.json"
  "recipe/train_recipe/C2C_0.6+0.5_mmlu_aux_0_15k_semantic_p3_a6000_2gpu_bs8.json"
)

PORTS=(29751 29752 29753)

output_dir_for_recipe() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg["output"]["output_dir"])
PY
}

for i in "${!RECIPES[@]}"; do
  recipe="${RECIPES[$i]}"
  port="${PORTS[$i]}"
  output_dir="$(output_dir_for_recipe "${recipe}")"

  if [[ -f "${output_dir}/final/projector_config.json" ]]; then
    echo "Skipping completed projector: ${output_dir}/final"
    continue
  fi

  echo "Starting semantic-domain projector training on GPUs ${GPU_IDS}: ${recipe}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" conda run --no-capture-output -n "${ENV_NAME}" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${port}" \
      script/train/SFT_train.py --config "${recipe}"
done

echo "All semantic-domain A6000 2GPU projector trainings are complete."
