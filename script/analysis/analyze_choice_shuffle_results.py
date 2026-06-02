#!/usr/bin/env python3
"""Summarize MMLU-Redux choice-shuffle evaluation outputs."""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


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


def summarize_csv(name: str, csv_path: Path) -> Dict[str, float]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    n = len(rows)
    moved = [
        row for row in rows
        if row.get("original_true_answer") and row.get("shuffled_true_answer")
        and row.get("original_true_answer") != row.get("shuffled_true_answer")
    ]
    moved_n = len(moved)
    acc = sum(_as_bool(row.get("is_correct", "")) for row in rows) / n
    pred_values = [row.get("pred", "").strip() for row in rows]
    shuffled_values = [row.get("shuffled_true_answer", "").strip() for row in rows]
    original_values = [row.get("original_true_answer", "").strip() for row in rows]
    pred_old_on_moved = (
        sum(row.get("pred", "").strip() == row.get("original_true_answer", "").strip() for row in moved) / moved_n
        if moved_n
        else 0.0
    )
    pred_new_on_moved = (
        sum(row.get("pred", "").strip() == row.get("shuffled_true_answer", "").strip() for row in moved) / moved_n
        if moved_n
        else 0.0
    )

    seed = rows[0].get("choice_shuffle_seed", "")
    suffix = f" seed={seed}" if seed != "" else ""
    print(f"\n{name}{suffix}")
    print(f"  csv: {csv_path}")
    print(f"  n={n} accuracy(new shuffled labels)={acc:.4%}")
    print(f"  pred dist:           {_dist(pred_values, n)}")
    print(f"  shuffled true dist:  {_dist(shuffled_values, n)}")
    print(f"  original true dist:  {_dist(original_values, n)}")
    print(f"  moved-label samples: {moved_n}/{n} ({moved_n / n:.1%})")
    print(f"  on moved labels: pred==new_true {pred_new_on_moved:.4%}, pred==old_true {pred_old_on_moved:.4%}")
    return {
        "accuracy": acc,
        "moved_new_true_rate": pred_new_on_moved,
        "moved_old_true_rate": pred_old_on_moved,
    }


def summarize(name: str, path: Path, include_all: bool = False) -> None:
    csv_paths = _cot_csvs(path, include_all=include_all)
    for csv_path in csv_paths:
        summarize_csv(name, csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="Model name and result directory or *_cot.csv path.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Summarize every *_cot.csv under each result directory instead of only the latest.",
    )
    args = parser.parse_args()

    for name, path in args.result:
        summarize(name, Path(path), include_all=args.all)


if __name__ == "__main__":
    main()
