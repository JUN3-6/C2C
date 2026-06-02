import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def latest_csv(result_dir: Path) -> Path:
    csv_files = sorted(result_dir.glob("*_cot.csv"), key=lambda path: path.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError(f"No *_cot.csv files found in {result_dir}")
    return csv_files[-1]


def read_latest_rows(result_dir: Path) -> Tuple[List[Dict], Path]:
    csv_file = latest_csv(result_dir)
    with open(csv_file, newline="") as f:
        return list(csv.DictReader(f)), csv_file


def row_key(row: Dict) -> Tuple[str, int]:
    return row.get("subject", ""), int(row.get("question_id", -1))


def accuracy(rows: List[Dict]) -> float:
    return sum(1 for row in rows if as_bool(row.get("is_correct"))) / len(rows) if rows else 0.0


def summarize_subjects(paired: List[Tuple[Dict, Dict]]) -> Dict[str, Dict]:
    by_subject = defaultdict(list)
    for normal, swapped in paired:
        by_subject[normal.get("subject", "")].append((normal, swapped))

    out = {}
    for subject, pairs in by_subject.items():
        normal_rows = [normal for normal, _ in pairs]
        swapped_rows = [swapped for _, swapped in pairs]
        normal_acc = accuracy(normal_rows)
        swapped_acc = accuracy(swapped_rows)
        out[subject] = {
            "n": len(pairs),
            "normal_accuracy": normal_acc,
            "swapped_accuracy": swapped_acc,
            "delta_swapped_minus_normal": swapped_acc - normal_acc,
            "normal_pred_distribution": dict(Counter(row.get("pred") or "" for row in normal_rows)),
            "swapped_pred_distribution": dict(Counter(row.get("pred") or "" for row in swapped_rows)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare normal C2C eval against KV-content ablation")
    parser.add_argument("--normal-dir", required=True, help="Directory containing normal *_cot.csv")
    parser.add_argument("--swap-dir", required=True, help="Directory containing KV-content ablation *_cot.csv")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path")
    parser.add_argument("--top-k-subjects", type=int, default=10)
    args = parser.parse_args()

    normal_rows, normal_csv = read_latest_rows(Path(args.normal_dir))
    swap_rows, swap_csv = read_latest_rows(Path(args.swap_dir))

    normal_by_key = {row_key(row): row for row in normal_rows}
    swap_by_key = {row_key(row): row for row in swap_rows}
    common_keys = sorted(set(normal_by_key) & set(swap_by_key))
    paired = [(normal_by_key[key], swap_by_key[key]) for key in common_keys]

    normal_common = [normal for normal, _ in paired]
    swap_common = [swapped for _, swapped in paired]
    normal_acc = accuracy(normal_common)
    swap_acc = accuracy(swap_common)

    both_correct = sum(
        1 for normal, swapped in paired
        if as_bool(normal.get("is_correct")) and as_bool(swapped.get("is_correct"))
    )
    normal_only = sum(
        1 for normal, swapped in paired
        if as_bool(normal.get("is_correct")) and not as_bool(swapped.get("is_correct"))
    )
    swap_only = sum(
        1 for normal, swapped in paired
        if not as_bool(normal.get("is_correct")) and as_bool(swapped.get("is_correct"))
    )
    both_wrong = len(paired) - both_correct - normal_only - swap_only
    pred_changed = sum(1 for normal, swapped in paired if normal.get("pred") != swapped.get("pred"))

    donor_rows = [
        swapped for _, swapped in paired
        if swapped.get("pred") and swapped.get("donor_true_answer")
    ]
    donor_match = sum(1 for row in donor_rows if row.get("pred") == row.get("donor_true_answer"))
    donor_label_mismatch_rows = [
        row for row in donor_rows
        if str(row.get("donor_label_matches_target", "")).strip().lower() == "false"
    ]
    donor_match_on_mismatch = sum(
        1 for row in donor_label_mismatch_rows
        if row.get("pred") == row.get("donor_true_answer")
    )

    subject_summary = summarize_subjects(paired)
    top_drops = sorted(
        subject_summary.items(),
        key=lambda item: item[1]["delta_swapped_minus_normal"],
    )[: args.top_k_subjects]
    top_gains = sorted(
        subject_summary.items(),
        key=lambda item: item[1]["delta_swapped_minus_normal"],
        reverse=True,
    )[: args.top_k_subjects]

    summary = {
        "normal_csv": str(normal_csv),
        "swap_csv": str(swap_csv),
        "normal_n": len(normal_rows),
        "swap_n": len(swap_rows),
        "paired_n": len(paired),
        "normal_accuracy_on_overlap": normal_acc,
        "swapped_accuracy_on_overlap": swap_acc,
        "delta_swapped_minus_normal": swap_acc - normal_acc,
        "transition_counts": {
            "both_correct": both_correct,
            "normal_only_correct": normal_only,
            "swap_only_correct": swap_only,
            "both_wrong": both_wrong,
        },
        "prediction_changed_rate": pred_changed / len(paired) if paired else 0.0,
        "normal_pred_distribution": dict(Counter(row.get("pred") or "" for row in normal_common)),
        "swapped_pred_distribution": dict(Counter(row.get("pred") or "" for row in swap_common)),
        "pred_matches_donor_true_rate": donor_match / len(donor_rows) if donor_rows else 0.0,
        "pred_matches_donor_true_on_label_mismatch_rate": (
            donor_match_on_mismatch / len(donor_label_mismatch_rows)
            if donor_label_mismatch_rows else 0.0
        ),
        "donor_label_mismatch_n": len(donor_label_mismatch_rows),
        "subjects": subject_summary,
    }

    print(f"normal_csv={normal_csv}")
    print(f"swap_csv={swap_csv}")
    print(f"paired_n={len(paired)}")
    print(f"normal overlap accuracy={normal_acc * 100:.2f}%")
    print(f"swapped overlap accuracy={swap_acc * 100:.2f}%")
    print(f"delta swapped-normal={(swap_acc - normal_acc) * 100:.2f}pp")
    print(f"transition_counts={summary['transition_counts']}")
    print(f"prediction_changed_rate={summary['prediction_changed_rate'] * 100:.2f}%")
    print(f"normal_pred_distribution={summary['normal_pred_distribution']}")
    print(f"swapped_pred_distribution={summary['swapped_pred_distribution']}")
    print(f"pred == donor true={summary['pred_matches_donor_true_rate'] * 100:.2f}%")
    print(
        "pred == donor true when donor label differs="
        f"{summary['pred_matches_donor_true_on_label_mismatch_rate'] * 100:.2f}% "
        f"(n={summary['donor_label_mismatch_n']})"
    )
    print("\nTop subject drops (swapped - normal):")
    for subject, stats in top_drops:
        print(
            f"  {subject}: {stats['delta_swapped_minus_normal'] * 100:.2f}pp "
            f"({stats['normal_accuracy'] * 100:.2f}% -> {stats['swapped_accuracy'] * 100:.2f}%, n={stats['n']})"
        )
    print("\nTop subject gains (swapped - normal):")
    for subject, stats in top_gains:
        print(
            f"  {subject}: {stats['delta_swapped_minus_normal'] * 100:.2f}pp "
            f"({stats['normal_accuracy'] * 100:.2f}% -> {stats['swapped_accuracy'] * 100:.2f}%, n={stats['n']})"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
