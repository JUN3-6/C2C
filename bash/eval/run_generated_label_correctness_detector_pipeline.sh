#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DUAL_GENERATE_CONFIG="${DUAL_GENERATE_CONFIG:-recipe/eval_recipe/unified_eval_mmlu_aux15k_dual_generate_legacy_full.yaml}"
LOGITS_CSV="${LOGITS_CSV:-local/final_results/mmlu_aux15k_dual_logits_legacy_full/Rosetta_mmlu-auxiliary_dual_logits_select_20260505_014833_cot.csv}"
LABEL_DIR="${LABEL_DIR:-local/final_results/mmlu_aux15k_dual_generate_legacy_full_v2}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-local/final_results/mmlu_aux15k_correctness_detector_compact_v1_torch_mlp_genlabel_v1}"
BENCH_OUTPUT_DIR="${BENCH_OUTPUT_DIR:-local/final_results/c2c_general_correctness_detector_aux_compact_v1_torch_mlp_genlabel_v1}"
GENERATED_RECIPE="${GENERATED_RECIPE:-recipe/eval_recipe/unified_eval_c2c_general_correctness_detector_aux_compact_v1_torch_mlp_genlabel_v1.yaml}"
EXPECTED_ROWS="${EXPECTED_ROWS:-15000}"
DUAL_GENERATE_CHUNK_SIZE="${DUAL_GENERATE_CHUNK_SIZE:-1000}"
CHUNK_DIR="${CHUNK_DIR:-${LABEL_DIR}/chunks}"
MERGED_LABEL_CSV="${MERGED_LABEL_CSV:-${LABEL_DIR}/Rosetta_mmlu-auxiliary_dual_generate_genlabel_${EXPECTED_ROWS}_merged_cot.csv}"

WANDB_PROJECT="${WANDB_PROJECT:-C2C}"
WANDB_ENTITY="${WANDB_ENTITY:-june6-hanyang-university}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mmlu_aux15k_correctness_detector_compact_v1_torch_mlp_genlabel_v1}"

csv_data_rows() {
  local path="$1"
  python - "$path" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    with path.open(newline="") as f:
        print(max(0, sum(1 for _ in csv.reader(f)) - 1))
PY
}

latest_cot_csv_in() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*_cot.csv' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
}

write_chunk_recipe() {
  local start="$1"
  local end="$2"
  local output_dir="$3"
  local recipe_path="$4"
  python - "$DUAL_GENERATE_CONFIG" "$start" "$end" "$output_dir" "$recipe_path" <<'PY'
import sys
from pathlib import Path

import yaml

base_config, start, end, output_dir, recipe_path = sys.argv[1:6]
with open(base_config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config["output"]["output_dir"] = output_dir
config["eval"]["sample_start"] = int(start)
config["eval"]["sample_end"] = int(end)
path = Path(recipe_path)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
PY
}

merge_chunk_csvs() {
  python - "$MERGED_LABEL_CSV" "$CHUNK_DIR" "$EXPECTED_ROWS" <<'PY'
import csv
import sys
from pathlib import Path

merged_path = Path(sys.argv[1])
chunk_dir = Path(sys.argv[2])
expected_rows = int(sys.argv[3])

csv_paths = sorted(chunk_dir.glob("chunk_*_*/**/*_cot.csv"))
if not csv_paths:
    raise SystemExit(f"No chunk CSVs found under {chunk_dir}")

fieldnames = None
rows_written = 0
merged_path.parent.mkdir(parents=True, exist_ok=True)
with merged_path.open("w", newline="", encoding="utf-8") as out_f:
    writer = None
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as in_f:
            reader = csv.DictReader(in_f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
            for row in reader:
                writer.writerow(row)
                rows_written += 1

if rows_written != expected_rows:
    raise SystemExit(
        f"Merged {rows_written} rows, expected {expected_rows}: {merged_path}"
    )
print(merged_path)
PY
}

label_csv="$MERGED_LABEL_CSV"
label_rows=0
if [[ -f "$label_csv" ]]; then
  label_rows="$(csv_data_rows "$label_csv")"
fi

if [[ "${label_rows}" -lt "${EXPECTED_ROWS}" ]]; then
  echo "[1/4] Generating dual-generate labels in chunks with ${DUAL_GENERATE_CONFIG}"
  mkdir -p "$CHUNK_DIR"
  start=0
  while [[ "$start" -lt "$EXPECTED_ROWS" ]]; do
    end=$((start + DUAL_GENERATE_CHUNK_SIZE))
    if [[ "$end" -gt "$EXPECTED_ROWS" ]]; then
      end="$EXPECTED_ROWS"
    fi
    chunk_expected=$((end - start))
    chunk_output_dir="${CHUNK_DIR}/chunk_${start}_${end}"
    chunk_recipe="${CHUNK_DIR}/chunk_${start}_${end}.yaml"
    chunk_csv="$(latest_cot_csv_in "$chunk_output_dir" || true)"
    chunk_rows=0
    if [[ -n "${chunk_csv}" ]]; then
      chunk_rows="$(csv_data_rows "$chunk_csv")"
    fi
    if [[ "${chunk_rows}" -lt "${chunk_expected}" ]]; then
      rm -rf "$chunk_output_dir"
      write_chunk_recipe "$start" "$end" "$chunk_output_dir" "$chunk_recipe"
      echo "Generating label chunk ${start}:${end}"
      conda run --no-capture-output -n route python script/evaluation/unified_evaluator.py \
        --config "$chunk_recipe"
      chunk_csv="$(latest_cot_csv_in "$chunk_output_dir")"
      chunk_rows="$(csv_data_rows "$chunk_csv")"
    else
      echo "Reusing label chunk ${start}:${end} (${chunk_rows} rows)"
    fi
    if [[ "${chunk_rows}" -lt "${chunk_expected}" ]]; then
      echo "Chunk ${start}:${end} has only ${chunk_rows}/${chunk_expected} rows" >&2
      exit 1
    fi
    start="$end"
  done
  merge_chunk_csvs
  label_rows="$(csv_data_rows "$label_csv")"
fi

if [[ "${label_rows}" -lt "${EXPECTED_ROWS}" ]]; then
  echo "Label CSV has only ${label_rows} rows: ${label_csv}" >&2
  exit 1
fi
echo "Using label CSV: ${label_csv} (${label_rows} rows)"

positive_class_weight="$(
  python - "$label_csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
pos = 0
neg = 0
with path.open(newline="") as f:
    for row in csv.DictReader(f):
        true = str(row.get("true_answer", "")).strip().upper()
        receiver = str(row.get("dual_receiver_pred", "")).strip().upper()
        fusion = str(row.get("dual_fusion_pred", "")).strip().upper()
        receiver_ok = bool(receiver) and receiver == true
        fusion_ok = bool(fusion) and fusion == true
        if receiver_ok and not fusion_ok:
            pos += 1
        elif fusion_ok and not receiver_ok:
            neg += 1
print(1.0 if pos == 0 else neg / pos)
PY
)"
echo "Using positive_class_weight=${positive_class_weight}"

echo "[2/4] Training generated-label correctness detector"
conda run --no-capture-output -n route python script/evaluation/train_fusion_receiver_correctness_detector_torch.py \
  --input-csv "$LOGITS_CSV" \
  --label-csv "$label_csv" \
  --output-dir "$TRAIN_OUTPUT_DIR" \
  --feature-set compact_v1 \
  --architecture mlp \
  --hidden-dim 64 \
  --dropout 0.1 \
  --epochs 100 \
  --batch-size 512 \
  --lr 1e-3 \
  --lr-schedule constant \
  --weight-decay 1e-3 \
  --positive-class-weight "$positive_class_weight" \
  --val-fraction 0.2 \
  --fixed-threshold 0.796 \
  --checkpoint-metric val_auc \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-run-name "$WANDB_RUN_NAME"

detector_config="$(
  find "$TRAIN_OUTPUT_DIR" -maxdepth 1 -type f -name '*_config.json' -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
)"
if [[ -z "${detector_config}" ]]; then
  echo "Could not find detector config in ${TRAIN_OUTPUT_DIR}" >&2
  exit 1
fi
echo "Using detector config: ${detector_config}"

echo "[3/4] Writing benchmark recipe ${GENERATED_RECIPE}"
python - "$detector_config" "$BENCH_OUTPUT_DIR" "$GENERATED_RECIPE" <<'PY'
import sys
from pathlib import Path

detector_config, output_dir, recipe_path = sys.argv[1:4]
text = f"""model:
  model_name: Rosetta
  rosetta_config:
    base_model: Qwen/Qwen3-0.6B
    teacher_model: Qwen/Qwen2.5-0.5B-Instruct
    is_do_alignment: false
    alignment_strategy: "longest"
    checkpoints_dir: local/checkpoints/0.6+0.5B_C2C_general_again/final
    include_response: false
    static_gate_enabled: false
    entropy_gate_enabled: false
    multi_source_fusion_mode: sequential
    update_decode_past: false

  generation_config:
    do_sample: false
    max_new_tokens: 64

output:
  output_dir: {output_dir}

eval:
  dataset: mmlu-redux
  gpu_ids: [0]
  answer_method: fusion_receiver_correctness_detector
  correctness_detector_config: {detector_config}
  use_cot: false
  use_template: true
  sample_interval: 1
  math_grading_method: "comprehensive"
"""
path = Path(recipe_path)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(text, encoding="utf-8")
PY

echo "[4/4] Running mmlu-redux benchmark"
conda run --no-capture-output -n route python script/evaluation/unified_evaluator.py \
  --config "$GENERATED_RECIPE"

echo "Generated-label correctness detector pipeline complete."
