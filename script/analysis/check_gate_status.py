"""
Check gate open/close status of trained C2CProjector checkpoints.

For each projector in a checkpoint directory, reads key_gate_logit and
value_gate_logit from the saved .pt file and reports whether the gate
would be open (logit > 0) or closed (logit <= 0) at inference time.

Usage:
    python check_gate_status.py --checkpoint-dir <path> [options]
"""

import argparse
import json
import os
import re
import sys

import torch


def load_gate_logits(pt_path: str) -> dict:
    """Extract gate logit scalars from a projector state dict."""
    state = torch.load(pt_path, map_location="cpu")
    return {
        "key_gate_logit": state.get("key_gate_logit", None),
        "value_gate_logit": state.get("value_gate_logit", None),
    }


def build_layer_map(proj_config: dict) -> dict:
    """
    Invert projector_config.json into a lookup:
        proj_idx -> (target_layer_idx, source_layer_idx)
    """
    layer_map = {}
    for target_model_idx, src_dict in proj_config.items():
        for source_model_idx, layer_dict in src_dict.items():
            for target_layer, entries in layer_dict.items():
                for src_layer, proj_idx in entries:
                    layer_map[int(proj_idx)] = {
                        "target_model": int(target_model_idx),
                        "source_model": int(source_model_idx),
                        "target_layer": int(target_layer),
                        "source_layer": int(src_layer),
                    }
    return layer_map


def check_gate_status(checkpoint_dir: str, verbose: bool = True) -> list:
    """
    Load all projectors in checkpoint_dir and return their gate status.

    Returns:
        List of dicts, one per projector, sorted by projector index.
    """
    pt_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)],
        key=lambda f: int(re.search(r"\d+", f).group()),
    )

    if not pt_files:
        raise FileNotFoundError(f"No projector_*.pt files found in {checkpoint_dir}")

    # Load projector config for layer mapping (optional)
    layer_map = {}
    cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            layer_map = build_layer_map(json.load(f))

    results = []
    for pt_file in pt_files:
        proj_idx = int(re.search(r"\d+", pt_file).group())
        pt_path = os.path.join(checkpoint_dir, pt_file)
        gates = load_gate_logits(pt_path)

        k = gates["key_gate_logit"]
        v = gates["value_gate_logit"]

        if k is None or v is None:
            entry = {
                "proj_idx": proj_idx,
                "key_gate_logit": None,
                "value_gate_logit": None,
                "key_open": None,
                "value_open": None,
                "gate_open": None,
                "layer_info": layer_map.get(proj_idx, {}),
            }
        else:
            k_val = float(k.item())
            v_val = float(v.item())
            k_open = k_val > 0
            v_open = v_val > 0
            entry = {
                "proj_idx": proj_idx,
                "key_gate_logit": k_val,
                "value_gate_logit": v_val,
                "key_open": k_open,
                "value_open": v_open,
                "gate_open": k_open or v_open,
                "layer_info": layer_map.get(proj_idx, {}),
            }
        results.append(entry)

    return results


def print_table(results: list, show_layer_info: bool = True) -> None:
    has_layer = any(r["layer_info"] for r in results)

    # Header
    cols = ["Proj", "target_layer", "source_layer", "key_gate_logit", "value_gate_logit",
            "key_open", "val_open", "gate_open"]
    if not (has_layer and show_layer_info):
        cols = [c for c in cols if c not in ("target_layer", "source_layer")]

    widths = {
        "Proj": 5, "target_layer": 12, "source_layer": 12,
        "key_gate_logit": 16, "value_gate_logit": 17,
        "key_open": 9, "val_open": 9, "gate_open": 9,
    }
    header = " | ".join(f"{c:>{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))

    open_count = 0
    closed_count = 0
    unknown_count = 0

    for r in results:
        if r["key_gate_logit"] is None:
            row_vals = {
                "Proj": str(r["proj_idx"]),
                "target_layer": r["layer_info"].get("target_layer", "-"),
                "source_layer": r["layer_info"].get("source_layer", "-"),
                "key_gate_logit": "N/A",
                "value_gate_logit": "N/A",
                "key_open": "N/A",
                "val_open": "N/A",
                "gate_open": "N/A",
            }
            unknown_count += 1
        else:
            row_vals = {
                "Proj": str(r["proj_idx"]),
                "target_layer": str(r["layer_info"].get("target_layer", "-")),
                "source_layer": str(r["layer_info"].get("source_layer", "-")),
                "key_gate_logit": f"{r['key_gate_logit']:.6f}",
                "value_gate_logit": f"{r['value_gate_logit']:.6f}",
                "key_open": str(r["key_open"]),
                "val_open": str(r["value_open"]),
                "gate_open": str(r["gate_open"]),
            }
            if r["gate_open"]:
                open_count += 1
            else:
                closed_count += 1

        if not (has_layer and show_layer_info):
            row_vals = {k: v for k, v in row_vals.items() if k not in ("target_layer", "source_layer")}

        print(" | ".join(f"{row_vals[c]:>{widths[c]}}" for c in cols))

    total = len(results)
    print()
    print(f"Total projectors : {total}")
    print(f"  Gate OPEN      : {open_count}  ({100*open_count/total:.1f}%)")
    print(f"  Gate CLOSED    : {closed_count}  ({100*closed_count/total:.1f}%)  ← skippable at inference")
    if unknown_count:
        print(f"  Unknown        : {unknown_count}")


def save_json(results: list, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check gate open/close status of trained C2CProjector checkpoints."
    )
    parser.add_argument(
        "--checkpoint-dir", "-d",
        required=True,
        help="Path to checkpoint directory containing projector_*.pt files.",
    )
    parser.add_argument(
        "--save-json", "-o",
        metavar="PATH",
        default=None,
        help="If provided, save full results to this JSON file.",
    )
    parser.add_argument(
        "--no-layer-info",
        action="store_true",
        help="Hide target/source layer columns even when projector_config.json is present.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        print(f"Error: {args.checkpoint_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Checkpoint dir: {args.checkpoint_dir}\n")
    results = check_gate_status(args.checkpoint_dir)
    print_table(results, show_layer_info=not args.no_layer_info)

    if args.save_json:
        save_json(results, args.save_json)


if __name__ == "__main__":
    main()
