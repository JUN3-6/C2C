#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_SCRIPT="${ROOT_DIR}/bash/train/run_router_full_pipeline.sh"

DEFAULT_CONFIG="${ROOT_DIR}/local/checkpoints/0.6+0.5B_C2C_general_again/config.json"
DEFAULT_OUTPUT="/home/june/workspace/C2C_routing/local/router_runs/0.6+0.5B_C2C_general_again_router_low_lambda"
DEFAULT_GAIN_LOSS_WEIGHT="0.03"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base pipeline script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

bash "${BASE_SCRIPT}" \
  --config "${DEFAULT_CONFIG}" \
  --output-dir "${DEFAULT_OUTPUT}" \
  --gain-loss-weight "${DEFAULT_GAIN_LOSS_WEIGHT}" \
  "$@"
