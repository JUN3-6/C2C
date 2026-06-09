#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

RECEIVER_MODEL="${RECEIVER_MODEL:-Qwen/Qwen3-0.6B}"

SHARER_SMALL_MODEL="${SHARER_SMALL_MODEL:-Qwen/Qwen2.5-0.5B}"
SHARER_MID_MODEL="${SHARER_MID_MODEL:-Qwen/Qwen3-4B}"
SHARER_LARGE_MODEL="${SHARER_LARGE_MODEL:-Qwen/Qwen3-8B}"
SHARER_SMALL_LABEL="${SHARER_SMALL_LABEL:-qwen2p5_0p5b}"
SHARER_MID_LABEL="${SHARER_MID_LABEL:-qwen3_4b}"
SHARER_LARGE_LABEL="${SHARER_LARGE_LABEL:-qwen3_8b}"

FIT_DEVICE="${FIT_DEVICE:-cuda:0}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,2}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-script/calibration/prompts/mmlu_redux_disjoint_128_oneline.txt}"
MAX_PROMPTS="${MAX_PROMPTS:-128}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
RIDGE="${RIDGE:-3.0}"
BLEND_ALPHA="${BLEND_ALPHA:-0.52}"
POSTPROCESS_MODE="${POSTPROCESS_MODE:-direct}"
MAPPING="${MAPPING:-k_nearest}"
K="${K:-1}"
TARGET_LAYERS="${TARGET_LAYERS:-all}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
FORCE_REFIT="${FORCE_REFIT:-0}"
MISMATCH_OFFSET="${MISMATCH_OFFSET:-1}"

LOG_DIR="${LOG_DIR:-local/logs/closed_form_qwen_receiver_sharer_sweep_gpu02}"
mkdir -p "$LOG_DIR"

checkpoint_dir_for() {
  local label="$1"
  echo "local/checkpoints/${label}_closed_form_kv_${MAPPING}_ridge${RIDGE}_a${BLEND_ALPHA}/final"
}

fit_one() {
  local label="$1"
  local source_model="$2"
  local checkpoint_dir
  checkpoint_dir="$(checkpoint_dir_for "$label")"
  local fit_log="${LOG_DIR}/${label}_fit_$(date +%Y%m%d_%H%M%S).log"

  echo "============================================================"
  echo "[Calibration] ${label}"
  echo "Receiver:    ${RECEIVER_MODEL}"
  echo "Source:      ${source_model}"
  echo "Checkpoint:  ${checkpoint_dir}"
  echo "Fit device:  ${FIT_DEVICE}"
  echo "============================================================"

  if [[ "${FORCE_REFIT}" == "1" || ! -f "${checkpoint_dir}/projector_config.json" ]]; then
    mkdir -p "$(dirname "$checkpoint_dir")"
    python script/calibration/fit_closed_form_kv_align.py \
      --receiver-model "$RECEIVER_MODEL" \
      --source-model "$source_model" \
      --output-dir "$checkpoint_dir" \
      --device "$FIT_DEVICE" \
      --dtype bfloat16 \
      --mapping "$MAPPING" \
      --k "$K" \
      --target-layers "$TARGET_LAYERS" \
      --max-prompts "$MAX_PROMPTS" \
      --max-length "$MAX_LENGTH" \
      --ridge "$RIDGE" \
      --blend-alpha "$BLEND_ALPHA" \
      --postprocess-mode "$POSTPROCESS_MODE" \
      --calibration-text-file "$CALIBRATION_PROMPTS" \
      2>&1 | tee "$fit_log"
  else
    echo "Skip fitting; existing checkpoint found at ${checkpoint_dir}"
  fi
}

eval_one() {
  local label="$1"
  local source_model="$2"
  local checkpoint_dir
  checkpoint_dir="$(checkpoint_dir_for "$label")"

  echo "============================================================"
  echo "[Benchmark] ${label}"
  echo "Receiver:    ${RECEIVER_MODEL}"
  echo "Source:      ${source_model}"
  echo "Checkpoint:  ${checkpoint_dir}"
  echo "Eval GPUs:   ${EVAL_GPU_IDS}"
  echo "============================================================"

  BASE_MODEL="$RECEIVER_MODEL" \
  TEACHER_MODEL="$source_model" \
  GPU_IDS="$EVAL_GPU_IDS" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  INCLUDE_RESPONSE=false \
  MULTI_SOURCE_FUSION_MODE=parallel \
  OUTPUT_DIR="local/final_results/${label}_closed_form_kv_${MAPPING}_ridge${RIDGE}_a${BLEND_ALPHA}_mmlu_redux" \
  LOG_DIR="$LOG_DIR" \
    bash bash/eval/mmlu_redux_rosetta_equal.sh \
      "$checkpoint_dir" \
      "${label}_closed_form_kv_${MAPPING}_ridge${RIDGE}_a${BLEND_ALPHA}"

  BASE_MODEL="$RECEIVER_MODEL" \
  TEACHER_MODEL="$source_model" \
  GPU_IDS="$EVAL_GPU_IDS" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  INCLUDE_RESPONSE=false \
  MULTI_SOURCE_FUSION_MODE=parallel \
  MISMATCH_OFFSET="$MISMATCH_OFFSET" \
  OUTPUT_DIR="local/final_results/${label}_closed_form_kv_${MAPPING}_ridge${RIDGE}_a${BLEND_ALPHA}_mmlu_redux_mismatch_offset${MISMATCH_OFFSET}" \
  LOG_DIR="$LOG_DIR" \
    bash bash/eval/mmlu_redux_source_mismatch.sh \
      "$checkpoint_dir" \
      "${label}_closed_form_kv_${MAPPING}_ridge${RIDGE}_a${BLEND_ALPHA}_mismatch_offset${MISMATCH_OFFSET}"
}

LABELS=(
  "qwen3_receiver__${SHARER_SMALL_LABEL}"
  "qwen3_receiver__${SHARER_MID_LABEL}"
  "qwen3_receiver__${SHARER_LARGE_LABEL}"
)
SOURCE_MODELS=(
  "$SHARER_SMALL_MODEL"
  "$SHARER_MID_MODEL"
  "$SHARER_LARGE_MODEL"
)

echo "Phase 1/2: fit closed-form KV projectors for all model pairs"
for idx in "${!LABELS[@]}"; do
  fit_one "${LABELS[$idx]}" "${SOURCE_MODELS[$idx]}"
done

echo "Phase 2/2: run clean and mismatch MMLU-Redux benchmarks for all model pairs"
for idx in "${!LABELS[@]}"; do
  eval_one "${LABELS[$idx]}" "${SOURCE_MODELS[$idx]}"
done
