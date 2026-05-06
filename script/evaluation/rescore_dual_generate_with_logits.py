#!/usr/bin/env python
"""Rescore cached dual-generate outputs with freshly recomputed option logits.

The original dual-generate CSV stores the generated receiver/fusion answers, but
not the full option probability vectors. This script reuses the generated text
and only recomputes the fixed-prefix option logits, making it possible to sweep
selectors such as entropy or top-1 margin without running generation again.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.model.wrapper import RosettaModel
from rosetta.utils.evaluate import build_prompt, get_option_token_ids, load_rosetta_model, set_default_chat_template
from script.evaluation.unified_evaluator import UnifiedEvaluator


def choice_index(answer: Optional[str], num_options: int) -> Optional[int]:
    if not answer:
        return None
    answer = str(answer).strip().upper()
    if len(answer) != 1:
        return None
    idx = ord(answer) - ord("A")
    return idx if 0 <= idx < num_options else None


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


def entropy(probs: np.ndarray) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def top1_margin(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    if probs.size < 2:
        return 0.0
    top2 = np.partition(probs, -2)[-2:]
    return float(top2.max() - top2.min())


def parse_float_list(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def option_count(row: Dict[str, str]) -> int:
    count = 0
    for letter in "ABCDEFGHIJ":
        if str(row.get(letter, "") or "").strip():
            count += 1
    return max(2, count)


def prompt_from_row(row: Dict[str, str], *, use_cot: bool, use_template: bool) -> str:
    choices = ""
    for letter in "ABCDEFGHIJ":
        value = str(row.get(letter, "") or "").strip()
        if value:
            choices += f"{letter}. {value}\n"
    return build_prompt(
        dataset="mmlu-redux",
        locale="",
        question=str(row.get("question", "")),
        choices=choices,
        use_cot=use_cot,
        use_template=use_template,
    )


@torch.no_grad()
def compute_option_probs(
    *,
    evaluator: UnifiedEvaluator,
    model: RosettaModel,
    tokenizer,
    llm_tokenizer,
    device: torch.device,
    prompt: str,
    option_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    proportion = evaluator.eval_config.get("kv_cache_proportion", 1.0)
    order_mode = evaluator.eval_config.get("kv_cache_order_mode", "front")
    prepared = evaluator.prepare_model_inputs(
        prompt=prompt,
        tokenizer=tokenizer,
        device=device,
        model_type="rosetta",
        llm_tokenizer=llm_tokenizer,
        answer_method="logits",
        proportion=proportion,
        order_mode=order_mode,
    )

    option_index = torch.tensor(option_ids, dtype=torch.long, device=device)

    receiver_inputs = evaluator._first_model_inputs(prepared["inputs"])
    receiver_model = model.model_list[model.base_model_idx]
    receiver_outputs = receiver_model(**receiver_inputs)
    receiver_logits = receiver_outputs.logits[0, -1].index_select(0, option_index)
    receiver_probs = torch.softmax(receiver_logits.float(), dim=-1)

    fusion_outputs = model.forward(**prepared["inputs"])
    fusion_logits = fusion_outputs.logits[0, -1].index_select(0, option_index)
    fusion_probs = torch.softmax(fusion_logits.float(), dim=-1)

    return (
        receiver_probs.detach().cpu().numpy(),
        fusion_probs.detach().cpu().numpy(),
    )


def score_candidate(
    selector: str,
    *,
    pred: str,
    own_probs: np.ndarray,
    other_probs: np.ndarray,
    num_options: int,
) -> float:
    pred_idx = choice_index(pred, num_options)
    if selector == "source_confidence":
        return float(own_probs[pred_idx]) if pred_idx is not None else float("-inf")
    if selector == "source_entropy":
        return -entropy(own_probs)
    if selector == "source_top1_margin":
        return top1_margin(own_probs)
    if selector == "other_verifier":
        return float(other_probs[pred_idx]) if pred_idx is not None else float("-inf")
    raise ValueError(f"Unknown selector: {selector}")


def select_record(
    record: Dict[str, Any],
    *,
    selector: str,
    margin: float,
    max_words: Optional[int],
    exact_answer_only: bool,
) -> Tuple[str, str, float, float]:
    row = record["row"]
    num_options = record["num_options"]
    receiver_pred = (row.get("dual_receiver_pred") or "").strip().upper()
    fusion_pred = (row.get("dual_fusion_pred") or "").strip().upper()
    receiver_valid = choice_index(receiver_pred, num_options) is not None
    fusion_valid = choice_index(fusion_pred, num_options) is not None

    if receiver_valid and not fusion_valid:
        return "receiver", "receiver_only_valid_logits_rescore", float("inf"), float("-inf")
    if fusion_valid and not receiver_valid:
        return "fusion", "fusion_only_valid_logits_rescore", float("-inf"), float("inf")
    if receiver_valid and fusion_valid and receiver_pred == fusion_pred:
        return "fusion", "same_answer_logits_rescore", float("nan"), float("nan")
    if not (receiver_valid and fusion_valid):
        return "fusion", "default_fusion_logits_rescore", float("nan"), float("nan")

    receiver_probs = record["receiver_probs"]
    fusion_probs = record["fusion_probs"]
    receiver_score = score_candidate(
        selector,
        pred=receiver_pred,
        own_probs=receiver_probs,
        other_probs=fusion_probs,
        num_options=num_options,
    )
    fusion_score = score_candidate(
        selector,
        pred=fusion_pred,
        own_probs=fusion_probs,
        other_probs=receiver_probs,
        num_options=num_options,
    )
    allowed = receiver_allowed(
        row.get("dual_receiver_output"),
        max_words=max_words,
        exact_answer_only=exact_answer_only,
    )
    if receiver_score > fusion_score + margin and allowed:
        return "receiver", f"receiver_{selector}_logits_rescore", receiver_score, fusion_score
    reason = (
        f"fusion_{selector}_logits_rescore"
        if allowed
        else f"fusion_{selector}_receiver_guarded_logits_rescore"
    )
    return "fusion", reason, receiver_score, fusion_score


def evaluate_records(
    records: Iterable[Dict[str, Any]],
    *,
    selector: str,
    margin: float,
    max_words: Optional[int],
    exact_answer_only: bool,
    all_rows: Dict[Tuple[str, str], Dict[str, str]],
) -> Dict[str, Any]:
    selected_counts = Counter()
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    total = 0
    help_count = 0
    harm_count = 0
    same_count = 0

    for record in records:
        row = record["row"]
        selected_source, _, _, _ = select_record(
            record,
            selector=selector,
            margin=margin,
            max_words=max_words,
            exact_answer_only=exact_answer_only,
        )
        pred_key = "dual_receiver_pred" if selected_source == "receiver" else "dual_fusion_pred"
        pred = (row.get(pred_key) or "").strip().upper()
        ok = pred == (row.get("true_answer") or "").strip().upper()
        correct += int(ok)
        total += 1
        selected_counts[selected_source] += 1
        subject = row.get("subject", "")
        subject_stats[subject]["correct"] += int(ok)
        subject_stats[subject]["total"] += 1

        baseline = all_rows.get(row_key(row))
        if baseline is not None:
            base_correct = is_correct(baseline)
            if ok and not base_correct:
                help_count += 1
            elif base_correct and not ok:
                harm_count += 1
            else:
                same_count += 1

    subjects = {
        subject: stats["correct"] / stats["total"]
        for subject, stats in sorted(subject_stats.items())
        if stats["total"]
    }
    summary = {
        "selector": selector,
        "margin": margin,
        "overall_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "selected_source_counts": dict(selected_counts),
        "subjects": subjects,
    }
    if all_rows:
        baseline_correct = sum(
            is_correct(all_rows[row_key(record["row"])])
            for record in records
            if row_key(record["row"]) in all_rows
        )
        baseline_total = sum(
            1 for record in records if row_key(record["row"]) in all_rows
        )
        summary.update(
            {
                "all_fusion_accuracy": baseline_correct / baseline_total
                if baseline_total
                else 0.0,
                "all_fusion_correct": baseline_correct,
                "all_fusion_total": baseline_total,
                "help": help_count,
                "harm": harm_count,
                "net": help_count - harm_count,
                "same": same_count,
            }
        )
    return summary


def write_rescored_csv(
    *,
    records: List[Dict[str, Any]],
    fieldnames: List[str],
    path: Path,
    selector: str,
    margin: float,
    max_words: Optional[int],
    exact_answer_only: bool,
) -> None:
    extra_fields = [
        "dual_receiver_score_logits_rescore",
        "dual_fusion_score_logits_rescore",
        "dual_receiver_probs",
        "dual_fusion_probs",
    ]
    output_fieldnames = list(fieldnames)
    for name in extra_fields:
        if name not in output_fieldnames:
            output_fieldnames.append(name)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record["row"])
            selected_source, reason, receiver_score, fusion_score = select_record(
                record,
                selector=selector,
                margin=margin,
                max_words=max_words,
                exact_answer_only=exact_answer_only,
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
            row["dual_receiver_score"] = receiver_score
            row["dual_fusion_score"] = fusion_score
            row["dual_receiver_score_logits_rescore"] = receiver_score
            row["dual_fusion_score_logits_rescore"] = fusion_score
            row["dual_receiver_probs"] = json.dumps(record["receiver_probs"].tolist())
            row["dual_fusion_probs"] = json.dumps(record["fusion_probs"].tolist())
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dual-csv", required=True)
    parser.add_argument("--all-fusion-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--selectors",
        default="source_entropy,source_top1_margin",
        help="Comma-separated selectors: source_confidence,source_entropy,source_top1_margin",
    )
    parser.add_argument(
        "--margins",
        default="0,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.9",
        help="Comma-separated margin sweep values.",
    )
    parser.add_argument("--receiver-max-words", type=int)
    parser.add_argument("--receiver-exact-answer-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-best-csv", action="store_true")
    parser.add_argument("--tag", default="logits_rescore")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as f:
        config = yaml.safe_load(f)
    config["eval"]["answer_method"] = "dual_generate"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dual_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if args.limit is not None:
        rows = rows[: args.limit]

    all_rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    if args.all_fusion_csv:
        with open(args.all_fusion_csv, newline="") as f:
            all_rows = {row_key(row): row for row in csv.DictReader(f)}

    if not torch.cuda.is_available():
        raise RuntimeError("This script expects CUDA for Rosetta rescoring.")
    device = torch.device(f"cuda:{config['eval'].get('gpu_ids', [0])[0]}")
    torch.cuda.set_device(device)

    evaluator = UnifiedEvaluator(config)
    model, tokenizer = load_rosetta_model(
        evaluator.model_config,
        evaluator.eval_config,
        device=device,
        generation_config=evaluator.generation_config,
    )
    model.eval()

    rosetta_cfg = evaluator.model_config.get("rosetta_config", {})
    is_do_alignment = evaluator.model_config.get(
        "is_do_alignment",
        rosetta_cfg.get("is_do_alignment", False),
    )
    llm_tokenizer = None
    if is_do_alignment and rosetta_cfg.get("teacher_model"):
        llm_tokenizer = AutoTokenizer.from_pretrained(str(rosetta_cfg["teacher_model"]))
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        set_default_chat_template(llm_tokenizer, str(rosetta_cfg["teacher_model"]))

    selectors = [part.strip() for part in args.selectors.split(",") if part.strip()]
    margins = parse_float_list(args.margins)
    use_cot = bool(evaluator.eval_config.get("use_cot", False))
    use_template = bool(evaluator.eval_config.get("use_template", True))

    option_ids_cache: Dict[int, List[int]] = {}
    records: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="Recomputing option logits"):
        num_options = option_count(row)
        option_ids = option_ids_cache.get(num_options)
        if option_ids is None:
            option_ids = get_option_token_ids(tokenizer, num_options)
            option_ids_cache[num_options] = option_ids
        prompt = prompt_from_row(row, use_cot=use_cot, use_template=use_template)
        receiver_probs, fusion_probs = compute_option_probs(
            evaluator=evaluator,
            model=model,
            tokenizer=tokenizer,
            llm_tokenizer=llm_tokenizer,
            device=device,
            prompt=prompt,
            option_ids=option_ids,
        )
        records.append(
            {
                "row": row,
                "num_options": num_options,
                "receiver_probs": receiver_probs,
                "fusion_probs": fusion_probs,
            }
        )

    sweep = []
    best_summary = None
    for selector in selectors:
        for margin in margins:
            summary = evaluate_records(
                records,
                selector=selector,
                margin=margin,
                max_words=args.receiver_max_words,
                exact_answer_only=args.receiver_exact_answer_only,
                all_rows=all_rows,
            )
            sweep.append(summary)
            key = (
                summary["correct"],
                summary.get("net", -10**9),
                -summary.get("harm", 10**9),
                -summary["selected_source_counts"].get("receiver", 0),
            )
            if best_summary is None or key > best_summary["_key"]:
                best_summary = dict(summary)
                best_summary["_key"] = key

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_path = output_dir / f"{args.tag}_{timestamp}_sweep.json"
    with sweep_path.open("w") as f:
        json.dump(
            {
                "config": str(config_path),
                "dual_csv": str(args.dual_csv),
                "all_fusion_csv": str(args.all_fusion_csv) if args.all_fusion_csv else None,
                "receiver_max_words": args.receiver_max_words,
                "receiver_exact_answer_only": args.receiver_exact_answer_only,
                "sweep": sweep,
                "best": {k: v for k, v in (best_summary or {}).items() if k != "_key"},
            },
            f,
            indent=2,
        )

    print(f"Saved sweep to {sweep_path}")
    print("selector\tmargin\tacc\tcorrect\thelp\tharm\tnet\treceiver\tfusion")
    for summary in sweep:
        print(
            f"{summary['selector']}\t{summary['margin']:.4g}\t"
            f"{summary['overall_accuracy'] * 100:.2f}\t{summary['correct']}\t"
            f"{summary.get('help', 0)}\t{summary.get('harm', 0)}\t"
            f"{summary.get('net', 0):+d}\t"
            f"{summary['selected_source_counts'].get('receiver', 0)}\t"
            f"{summary['selected_source_counts'].get('fusion', 0)}"
        )

    if best_summary:
        print(
            "BEST "
            f"selector={best_summary['selector']} margin={best_summary['margin']} "
            f"accuracy={best_summary['overall_accuracy'] * 100:.2f}% "
            f"correct={best_summary['correct']}/{best_summary['total']} "
            f"help={best_summary.get('help', 0)} harm={best_summary.get('harm', 0)} "
            f"net={best_summary.get('net', 0):+d} "
            f"selected={best_summary['selected_source_counts']}"
        )

        if args.save_best_csv:
            csv_path = output_dir / (
                f"{args.tag}_{best_summary['selector']}_m{best_summary['margin']:.4g}_{timestamp}_cot.csv"
            )
            write_rescored_csv(
                records=records,
                fieldnames=fieldnames,
                path=csv_path,
                selector=best_summary["selector"],
                margin=float(best_summary["margin"]),
                max_words=args.receiver_max_words,
                exact_answer_only=args.receiver_exact_answer_only,
            )
            summary_path = output_dir / (
                f"{args.tag}_{best_summary['selector']}_m{best_summary['margin']:.4g}_{timestamp}_summary.json"
            )
            with summary_path.open("w") as f:
                json.dump({k: v for k, v in best_summary.items() if k != "_key"}, f, indent=2)
            print(f"Saved best CSV to {csv_path}")
            print(f"Saved best summary to {summary_path}")


if __name__ == "__main__":
    main()
