#!/usr/bin/env python
"""Rescore cached dual-generate candidates without running generation again.

This is useful when receiver/fusion outputs are already saved in a CoT CSV and
we only want to test a different selector policy.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


def choice_index(answer: Optional[str]) -> Optional[int]:
    if not answer:
        return None
    answer = str(answer).strip().upper()
    if len(answer) != 1:
        return None
    idx = ord(answer) - ord("A")
    return idx if 0 <= idx < 10 else None


def word_count(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(str(text).strip().split())


def exact_answer(text: Optional[str]) -> bool:
    if not text:
        return False
    return re.fullmatch(
        r"\s*The correct answer is [A-J]\.?\s*",
        str(text),
        flags=re.IGNORECASE,
    ) is not None


def is_correct(row: Dict[str, str]) -> bool:
    return str(row.get("is_correct", "")).strip().lower() == "true"


def row_key(row: Dict[str, str]) -> Tuple[str, str]:
    return str(row.get("subject", "")), str(row.get("question_id", ""))


def receiver_allowed(
    text: Optional[str],
    *,
    max_words: Optional[int],
    exact_answer_only: bool,
) -> bool:
    if max_words is not None and word_count(text) > max_words:
        return False
    if exact_answer_only and not exact_answer(text):
        return False
    return True


def select_row(
    row: Dict[str, str],
    *,
    margin: float,
    max_words: Optional[int],
    exact_answer_only: bool,
) -> Tuple[str, str]:
    receiver_pred = (row.get("dual_receiver_pred") or "").strip().upper()
    fusion_pred = (row.get("dual_fusion_pred") or "").strip().upper()
    receiver_valid = choice_index(receiver_pred) is not None
    fusion_valid = choice_index(fusion_pred) is not None

    if receiver_valid and not fusion_valid:
        return "receiver", "receiver_only_valid_rescore"
    if fusion_valid and not receiver_valid:
        return "fusion", "fusion_only_valid_rescore"
    if receiver_valid and fusion_valid and receiver_pred == fusion_pred:
        return "fusion", "same_answer_rescore"
    if not (receiver_valid and fusion_valid):
        return "fusion", "default_fusion_rescore"

    try:
        receiver_score = float(row.get("dual_receiver_score") or "-inf")
        fusion_score = float(row.get("dual_fusion_score") or "-inf")
    except ValueError:
        return "fusion", "missing_score_rescore"

    allowed = receiver_allowed(
        row.get("dual_receiver_output"),
        max_words=max_words,
        exact_answer_only=exact_answer_only,
    )
    if receiver_score > fusion_score + margin and allowed:
        return "receiver", "receiver_source_confidence_rescore"
    if not allowed:
        return "fusion", "fusion_source_confidence_receiver_guarded_rescore"
    return "fusion", "fusion_source_confidence_rescore"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-csv", required=True)
    parser.add_argument("--all-fusion-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin", type=float, default=0.65)
    parser.add_argument("--receiver-max-words", type=int)
    parser.add_argument("--receiver-exact-answer-only", action="store_true")
    parser.add_argument("--tag", default="guarded_rescore")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dual_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    all_rows = {}
    if args.all_fusion_csv:
        with open(args.all_fusion_csv, newline="") as f:
            for row in csv.DictReader(f):
                all_rows[row_key(row)] = row

    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    selected_counts = Counter()
    help_count = 0
    harm_count = 0
    same_count = 0
    rescored_rows = []

    for row in rows:
        row = dict(row)
        selected_source, reason = select_row(
            row,
            margin=args.margin,
            max_words=args.receiver_max_words,
            exact_answer_only=args.receiver_exact_answer_only,
        )
        if selected_source == "receiver":
            pred = (row.get("dual_receiver_pred") or "").strip().upper()
            output = row.get("dual_receiver_output") or ""
        else:
            pred = (row.get("dual_fusion_pred") or "").strip().upper()
            output = row.get("dual_fusion_output") or ""

        correct = pred == (row.get("true_answer") or "").strip().upper()
        row["pred"] = pred
        row["cot_pred"] = pred
        row["cot_output"] = output
        row["is_correct"] = str(bool(correct))
        row["dual_selected_source"] = selected_source
        row["dual_selector_reason"] = reason
        rescored_rows.append(row)

        subject = row.get("subject", "")
        subject_stats[subject]["total"] += 1
        subject_stats[subject]["correct"] += int(correct)
        selected_counts[selected_source] += 1

        if all_rows:
            baseline = all_rows.get(row_key(row))
            if baseline is not None:
                base_correct = is_correct(baseline)
                if correct and not base_correct:
                    help_count += 1
                elif base_correct and not correct:
                    harm_count += 1
                else:
                    same_count += 1

    total = sum(v["total"] for v in subject_stats.values())
    correct = sum(v["correct"] for v in subject_stats.values())
    subjects = {
        subject: (stats["correct"] / stats["total"] if stats["total"] else 0.0)
        for subject, stats in sorted(subject_stats.items())
    }
    summary = {
        "answer_method": "dual_generate_rescore",
        "selector": "source_confidence_guarded",
        "margin": args.margin,
        "receiver_max_words": args.receiver_max_words,
        "receiver_exact_answer_only": args.receiver_exact_answer_only,
        "overall_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "selected_source_counts": dict(selected_counts),
        "subjects": subjects,
    }
    if all_rows:
        baseline_correct = sum(is_correct(all_rows[row_key(row)]) for row in rows if row_key(row) in all_rows)
        baseline_total = sum(1 for row in rows if row_key(row) in all_rows)
        summary["all_fusion_accuracy"] = (
            baseline_correct / baseline_total if baseline_total else 0.0
        )
        summary["all_fusion_correct"] = baseline_correct
        summary["all_fusion_total"] = baseline_total
        summary["help"] = help_count
        summary["harm"] = harm_count
        summary["net"] = help_count - harm_count
        summary["same"] = same_count

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{args.tag}_{timestamp}_cot.csv"
    summary_path = output_dir / f"{args.tag}_{timestamp}_summary.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rescored_rows)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved rescored CSV to {csv_path}")
    print(f"Saved summary to {summary_path}")
    print(
        "accuracy="
        f"{summary['overall_accuracy'] * 100:.2f}% "
        f"({correct}/{total})"
    )
    if all_rows:
        print(
            "all_fusion="
            f"{summary['all_fusion_accuracy'] * 100:.2f}% "
            f"({summary['all_fusion_correct']}/{summary['all_fusion_total']}) "
            f"help={help_count} harm={harm_count} net={help_count - harm_count}"
        )
    print(f"selected_source_counts={dict(selected_counts)}")


if __name__ == "__main__":
    main()
