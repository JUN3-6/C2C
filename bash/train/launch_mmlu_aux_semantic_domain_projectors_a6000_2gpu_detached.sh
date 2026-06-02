#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p local/logs/mmlu_aux_semantic_a6000_2gpu
stamp="$(date +%Y%m%d_%H%M%S)"
log_path="local/logs/mmlu_aux_semantic_a6000_2gpu/train_${stamp}.log"
pid_path="local/logs/mmlu_aux_semantic_a6000_2gpu/latest.pid"

ENV_NAME="${ENV_NAME:-route}" \
GPU_IDS="${GPU_IDS:-0,1}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-2}" \
setsid bash bash/train/run_mmlu_aux_semantic_domain_projectors_a6000_2gpu.sh \
  > "${log_path}" 2>&1 < /dev/null &

pid="$!"
echo "${pid}" > "${pid_path}"

echo "Started detached A6000 2GPU semantic-domain projector training."
echo "PID: ${pid}"
echo "Log: ${log_path}"
