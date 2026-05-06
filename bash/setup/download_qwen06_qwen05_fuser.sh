#!/usr/bin/env bash
set -euo pipefail

REPO_ID=${REPO_ID:-nics-efc/C2C_Fuser}
SUBDIR=${SUBDIR:-qwen3_0.6b+qwen2.5_0.5b_Fuser}
ROOT_DIR=${ROOT_DIR:-$(pwd)}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-0.6+0.5B_C2C_general_again}
DEST_DIR=${DEST_DIR:-$ROOT_DIR/local/checkpoints/$CHECKPOINT_NAME}
CACHE_DIR=${CACHE_DIR:-$ROOT_DIR/local/hf_downloads/C2C_Fuser}

mkdir -p "$CACHE_DIR" "$(dirname "$DEST_DIR")"

python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${REPO_ID}",
    repo_type="model",
    allow_patterns=["${SUBDIR}/*"],
    local_dir="${CACHE_DIR}",
    local_dir_use_symlinks=False,
)
PY

SRC_DIR="$CACHE_DIR/$SUBDIR"
if [[ ! -d "$SRC_DIR" ]]; then
  echo "Downloaded subdir not found: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
rsync -a "$SRC_DIR"/ "$DEST_DIR"/

if [[ ! -f "$DEST_DIR/final/projector_config.json" ]]; then
  python - <<PY
import json
from pathlib import Path

final_dir = Path("${DEST_DIR}") / "final"
projector_ids = sorted(
    int(path.stem.split("_")[1])
    for path in final_dir.glob("projector_*.pt")
)
if not projector_ids:
    raise SystemExit(f"No projector_*.pt files found in {final_dir}")
if projector_ids != list(range(len(projector_ids))):
    raise SystemExit(f"Projector ids are not contiguous from 0: {projector_ids[:5]} ... {projector_ids[-5:]}")

mapping = {"0": {"1": {}}}
for target_layer in projector_ids:
    source_layer = max(0, target_layer - 4)
    mapping["0"]["1"][str(target_layer)] = [[source_layer, target_layer]]

out_path = final_dir / "projector_config.json"
out_path.write_text(json.dumps(mapping), encoding="utf-8")
print(f"Wrote {out_path}")
PY
fi

echo "Downloaded fuser checkpoint to: $DEST_DIR"
echo "CONFIG=$DEST_DIR/config.json"
echo "BANK_DIR=$DEST_DIR/final"
