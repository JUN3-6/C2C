#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG_DIR=${LOG_DIR:-local/logs/value_only_top5_earlyskip_qwen3_4b_mmlu15k_gpu0}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/value_only_earlyskip_${TIMESTAMP}.log}
TMP_CONFIG_DIR=${TMP_CONFIG_DIR:-local/tmp/value_only_top5_earlyskip_qwen3_4b_mmlu15k_gpu0}
SKIP_EVAL_IF_SUMMARY=${SKIP_EVAL_IF_SUMMARY:-0}
CHECK_SUBJECT=${CHECK_SUBJECT:-abstract_algebra}
FOLLOWUP_SUBJECTS=${FOLLOWUP_SUBJECTS:-"anatomy astronomy"}
COLLAPSE_RATIO_THRESHOLD=${COLLAPSE_RATIO_THRESHOLD:-0.30}

mkdir -p "${LOG_DIR}" "${TMP_CONFIG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CHECK_SUBJECT=${CHECK_SUBJECT}"
echo "FOLLOWUP_SUBJECTS=${FOLLOWUP_SUBJECTS}"
echo "COLLAPSE_RATIO_THRESHOLD=${COLLAPSE_RATIO_THRESHOLD}"
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

make_config() {
  local src="$1"
  local dst="$2"
  local name="$3"
  local phase="$4"
  shift 4
  python - "$src" "$dst" "$name" "$phase" "$@" <<'PY'
import sys
import yaml

src, dst, name, phase, *subjects = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)

cfg["eval"]["gpu_ids"] = [0]
cfg["eval"]["subjects"] = subjects
cfg["output"]["output_dir"] = (
    cfg["output"]["output_dir"].rstrip("/") + f"_value_only_earlyskip_{phase}"
)
cfg["eval"]["dataset_cache_dir"] = (
    cfg["eval"].get("dataset_cache_dir", "local/hf_datasets_cache/mmlu_redux")
    + f"_value_only_earlyskip_{phase}_{name}"
)

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

run_eval() {
  local config="$1"
  local output_dir="$2"
  if [[ "${SKIP_EVAL_IF_SUMMARY}" == "1" ]] && compgen -G "${output_dir}/*summary.json" > /dev/null; then
    echo "Skip eval because summary exists in ${output_dir}."
    return
  fi
  python script/evaluation/unified_evaluator_value_only.py --config "${config}"
}

analyze_collapse() {
  local output_dir="$1"
  python - "$output_dir" "${COLLAPSE_RATIO_THRESHOLD}" <<'PY'
import csv
import glob
import json
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
threshold = float(sys.argv[2])

summary_files = sorted(output_dir.glob("*summary.json"))
cot_files = sorted(output_dir.glob("*_cot.csv"))

accuracy = None
if summary_files:
    with summary_files[-1].open() as f:
        summary = json.load(f)
    accuracy = summary.get("overall_accuracy", summary.get("accuracy"))

rows = []
if cot_files:
    with cot_files[-1].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

def text_is_collapsed(text: str) -> bool:
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    patterns = [
        r"(Question){5,}",
        r"(answer){8,}",
        r"(correct){8,}",
        r"(\.){16,}",
        r"(\*){16,}",
        r"([A-D]\.){8,}",
    ]
    return any(re.search(pattern, compact, re.IGNORECASE) for pattern in patterns)

collapse_count = 0
empty_pred_count = 0
for row in rows:
    text = row.get("cot_output") or row.get("generated_output") or row.get("output") or ""
    pred = (row.get("pred") or "").strip()
    if text_is_collapsed(text):
        collapse_count += 1
    if not pred:
        empty_pred_count += 1

total = len(rows)
collapse_ratio = collapse_count / total if total else 0.0
empty_pred_ratio = empty_pred_count / total if total else 0.0
accuracy_value = float(accuracy) if isinstance(accuracy, (float, int)) else None
collapsed = (
    collapse_ratio >= threshold
    or (empty_pred_ratio >= 0.90 and (accuracy_value is None or accuracy_value <= 0.02))
)

print(
    "ANALYSIS "
    f"accuracy={accuracy if accuracy is not None else 'NA'} "
    f"rows={total} collapse_ratio={collapse_ratio:.4f} "
    f"empty_pred_ratio={empty_pred_ratio:.4f} "
    f"collapsed={int(collapsed)} "
    f"summary={summary_files[-1] if summary_files else 'NA'} "
    f"cot={cot_files[-1] if cot_files else 'NA'}"
)
raise SystemExit(10 if collapsed else 0)
PY
}

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  src_config="${EVAL_CONFIGS[$i]}"
  check_config="${TMP_CONFIG_DIR}/${name}_value_only_check_${CHECK_SUBJECT}.yaml"
  make_config "${src_config}" "${check_config}" "${name}" "check_${CHECK_SUBJECT}" "${CHECK_SUBJECT}"

  checkpoint_dir="$(yaml_get "${check_config}" "model.rosetta_config.checkpoints_dir")"
  check_output_dir="$(yaml_get "${check_config}" "output.output_dir")"

  echo
  echo "[$((i + 1))/${#NAMES[@]}] Check collapse: ${name}"
  echo "Config: ${check_config}"
  echo "Checkpoint: ${checkpoint_dir}"
  echo "Output: ${check_output_dir}"

  if [[ ! -f "${checkpoint_dir}/projector_config.json" ]]; then
    echo "ELIMINATED ${name}: missing ${checkpoint_dir}/projector_config.json"
    continue
  fi

  run_eval "${check_config}" "${check_output_dir}"

  set +e
  analyze_collapse "${check_output_dir}"
  collapse_status=$?
  set -e
  if [[ "${collapse_status}" == "10" ]]; then
    echo "ELIMINATED ${name}: collapse detected on ${CHECK_SUBJECT}"
    continue
  elif [[ "${collapse_status}" != "0" ]]; then
    echo "ELIMINATED ${name}: collapse analysis failed with status ${collapse_status}"
    continue
  fi

  echo "SURVIVED ${name}: no collapse detected on ${CHECK_SUBJECT}; running follow-up subjects"
  followup_config="${TMP_CONFIG_DIR}/${name}_value_only_followup.yaml"
  make_config "${src_config}" "${followup_config}" "${name}" "followup" ${FOLLOWUP_SUBJECTS}
  followup_output_dir="$(yaml_get "${followup_config}" "output.output_dir")"
  echo "Follow-up config: ${followup_config}"
  echo "Follow-up output: ${followup_output_dir}"
  run_eval "${followup_config}" "${followup_output_dir}"
done

echo
echo "Value-only early-skip summaries:"
find local/final_results -maxdepth 2 -path '*value_only_earlyskip*/*summary.json' -print | sort | while read -r summary; do
  python - "$summary" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
acc = data.get("overall_accuracy", data.get("accuracy"))
print(f"{acc:.4f}\t{path}" if isinstance(acc, (float, int)) else f"NA\t{path}")
PY
done

echo "Done: $(date)"
