import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _latest_csv(result_dir: Path) -> Path:
    csv_files = sorted(result_dir.glob("*_cot.csv"), key=lambda path: path.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError(f"No *_cot.csv files found in {result_dir}")
    return csv_files[-1]


def _read_rows(result_dir: Path, use_all: bool) -> Tuple[List[Dict], List[Path]]:
    csv_files = sorted(result_dir.glob("*_cot.csv"), key=lambda path: path.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError(f"No *_cot.csv files found in {result_dir}")
    selected = csv_files if use_all else [csv_files[-1]]
    rows: List[Dict] = []
    for csv_file in selected:
        with open(csv_file, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows, selected


def summarize_rows(name: str, rows: List[Dict]) -> Dict:
    n = len(rows)
    correct = sum(1 for row in rows if _as_bool(row.get("is_correct", False)))
    matched = [row for row in rows if _as_bool(row.get("kv_ablation_matched", False))]
    pred_counter = Counter(row.get("pred") or "" for row in rows)
    true_counter = Counter(row.get("true_answer") or "" for row in rows)
    donor_true_counter = Counter(row.get("donor_true_answer") or "" for row in matched)

    donor_eval_rows = [row for row in matched if row.get("pred") and row.get("donor_true_answer")]
    donor_match = sum(1 for row in donor_eval_rows if row.get("pred") == row.get("donor_true_answer"))
    donor_label_mismatch_rows = [
        row for row in donor_eval_rows
        if str(row.get("donor_label_matches_target", "")).strip().lower() == "false"
    ]
    donor_match_on_mismatch = sum(
        1 for row in donor_label_mismatch_rows
        if row.get("pred") == row.get("donor_true_answer")
    )

    by_subject = defaultdict(list)
    for row in rows:
        by_subject[row.get("subject", "")].append(row)

    subject_summary = {}
    for subject, subject_rows in by_subject.items():
        subject_n = len(subject_rows)
        subject_correct = sum(1 for row in subject_rows if _as_bool(row.get("is_correct", False)))
        subject_donor_rows = [
            row for row in subject_rows
            if row.get("pred") and row.get("donor_true_answer")
        ]
        subject_donor_match = sum(
            1 for row in subject_donor_rows
            if row.get("pred") == row.get("donor_true_answer")
        )
        subject_summary[subject] = {
            "n": subject_n,
            "accuracy": subject_correct / subject_n if subject_n else 0.0,
            "pred_matches_donor_true_rate": (
                subject_donor_match / len(subject_donor_rows)
                if subject_donor_rows else 0.0
            ),
            "pred_distribution": dict(Counter(row.get("pred") or "" for row in subject_rows)),
        }

    return {
        "name": name,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "matched_n": len(matched),
        "matched_rate": len(matched) / n if n else 0.0,
        "pred_distribution": dict(pred_counter),
        "true_answer_distribution": dict(true_counter),
        "donor_true_answer_distribution": dict(donor_true_counter),
        "pred_matches_donor_true_rate": donor_match / len(donor_eval_rows) if donor_eval_rows else 0.0,
        "pred_matches_donor_true_on_label_mismatch_rate": (
            donor_match_on_mismatch / len(donor_label_mismatch_rows)
            if donor_label_mismatch_rows else 0.0
        ),
        "donor_label_mismatch_n": len(donor_label_mismatch_rows),
        "subjects": subject_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize KV-content ablation results")
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("NAME", "DIR"),
        required=True,
        help="Result name and directory containing *_cot.csv",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Read all *_cot.csv files in each directory instead of only the latest one",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the summary JSON",
    )
    args = parser.parse_args()

    summaries = {}
    for name, directory in args.result:
        result_dir = Path(directory)
        rows, files = _read_rows(result_dir, args.all)
        summary = summarize_rows(name, rows)
        summary["files"] = [str(path) for path in files]
        summaries[name] = summary

        print(f"\n[{name}]")
        print(f"files: {', '.join(str(path) for path in files)}")
        print(f"n={summary['n']} matched={summary['matched_n']} ({summary['matched_rate'] * 100:.2f}%)")
        print(f"accuracy={summary['accuracy'] * 100:.2f}%")
        print(f"pred distribution={summary['pred_distribution']}")
        print(f"true distribution={summary['true_answer_distribution']}")
        print(f"donor true distribution={summary['donor_true_answer_distribution']}")
        print(f"pred == donor true={summary['pred_matches_donor_true_rate'] * 100:.2f}%")
        print(
            "pred == donor true when donor label differs="
            f"{summary['pred_matches_donor_true_on_label_mismatch_rate'] * 100:.2f}% "
            f"(n={summary['donor_label_mismatch_n']})"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
