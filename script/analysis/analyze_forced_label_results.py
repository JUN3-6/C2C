#!/usr/bin/env python3
"""Summarize MMLU-Redux forced-label evaluation outputs."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


LABELS = list("ABCD")


def _cot_csvs(path: Path, include_all: bool = False) -> List[Path]:
    if path.is_file():
        return [path]
    matches = sorted(path.glob("*_cot.csv"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No *_cot.csv found under {path}")
    return matches if include_all else [matches[-1]]


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _dist(values: Iterable[str], total: int) -> str:
    counts = Counter(v for v in values if v)
    return " ".join(
        f"{label}:{counts[label]}({counts[label] / total:.1%})"
        for label in LABELS
    )


def summarize_csv(name: str, csv_path: Path) -> Dict[str, object]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    n = len(rows)
    pred_values = [row.get("pred", "").strip().upper() for row in rows]
    forced_values = [row.get("shuffled_true_answer", "").strip().upper() for row in rows]
    original_values = [row.get("original_true_answer", "").strip().upper() for row in rows]
    target_counts = Counter(v for v in forced_values if v)
    forced_label = target_counts.most_common(1)[0][0] if target_counts else ""
    moved = [
        row for row in rows
        if row.get("original_true_answer") and row.get("shuffled_true_answer")
        and row.get("original_true_answer") != row.get("shuffled_true_answer")
    ]
    moved_n = len(moved)

    acc = sum(_as_bool(row.get("is_correct", "")) for row in rows) / n
    pred_forced = sum(pred == forced_label for pred in pred_values) / n if forced_label else 0.0
    pred_original = sum(
        pred == old for pred, old in zip(pred_values, original_values)
        if old
    ) / n
    moved_new = (
        sum(row.get("pred", "").strip().upper() == row.get("shuffled_true_answer", "").strip().upper() for row in moved) / moved_n
        if moved_n
        else 0.0
    )
    moved_old = (
        sum(row.get("pred", "").strip().upper() == row.get("original_true_answer", "").strip().upper() for row in moved) / moved_n
        if moved_n
        else 0.0
    )

    print(f"\n{name}")
    print(f"  csv: {csv_path}")
    print(f"  forced label: {forced_label}")
    print(f"  n={n} accuracy(forced labels)={acc:.4%}")
    print(f"  pred dist:          {_dist(pred_values, n)}")
    print(f"  forced true dist:   {_dist(forced_values, n)}")
    print(f"  original true dist: {_dist(original_values, n)}")
    print(f"  pred==forced_label: {pred_forced:.4%}")
    print(f"  pred==old_true:     {pred_original:.4%}")
    print(f"  moved-label samples: {moved_n}/{n} ({moved_n / n:.1%})")
    print(f"  on moved labels: pred==forced_true {moved_new:.4%}, pred==old_true {moved_old:.4%}")

    return {
        "name": name,
        "csv": str(csv_path),
        "forced_label": forced_label,
        "n": n,
        "accuracy": acc,
        "pred_forced_label_rate": pred_forced,
        "pred_old_true_rate": pred_original,
        "moved_n": moved_n,
        "moved_new_true_rate": moved_new,
        "moved_old_true_rate": moved_old,
        "pred_distribution": dict(Counter(v for v in pred_values if v)),
        "original_true_distribution": dict(Counter(v for v in original_values if v)),
    }


def summarize(name: str, path: Path, include_all: bool = False) -> List[Dict[str, object]]:
    return [
        summarize_csv(name, csv_path)
        for csv_path in _cot_csvs(path, include_all=include_all)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="Model/target name and result directory or *_cot.csv path.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Summarize every *_cot.csv under each result directory instead of only the latest.",
    )
    parser.add_argument("--output-json", help="Optional path to write a machine-readable summary.")
    args = parser.parse_args()

    summaries: List[Dict[str, object]] = []
    for name, path in args.result:
        summaries.extend(summarize(name, Path(path), include_all=args.all))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved summary JSON to {out}")


if __name__ == "__main__":
    main()
