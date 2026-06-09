import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.utils.evaluate import (
    build_prompt,
    extract_answer_from_content,
    load_rosetta_model,
)
from script.evaluation.unified_evaluator import DATASET_CONFIGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched MMLU-Redux generate eval for Rosetta closed-form KV checkpoints.")
    parser.add_argument("--config", default="recipe/eval_recipe/closed_form_kv_direct_layers8_27_a0p52_mmlu_redux_generate.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-subject sample limit.")
    parser.add_argument("--subjects", default=None, help="Optional comma-separated subject list.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--response-holdout-tokens",
        type=int,
        default=None,
        help=(
            "Number of trailing response-prefix tokens to keep receiver-only. "
            "Default keeps the whole response prefix receiver-only."
        ),
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_mmlu_redux_answer(example: Dict[str, Any]) -> Optional[str]:
    error_type = example.get("error_type", "")
    if error_type in ["no_correct_answer", "expert"]:
        return None

    if error_type == "wrong_groundtruth" and example.get("correct_answer") is not None:
        answer = example["correct_answer"]
        if isinstance(answer, str):
            answer = answer.strip()
            if answer in ["0", "1", "2", "3"]:
                answer_num = int(answer)
            elif answer in ["A", "B", "C", "D"]:
                answer_num = ord(answer) - ord("A")
            else:
                return None
        else:
            answer_num = int(answer)
    else:
        answer_num = int(example["answer"])

    if answer_num < 0 or answer_num > 3:
        return None
    return chr(65 + answer_num)


def format_mmlu_redux_prompt(example: Dict[str, Any], use_cot: bool, use_template: bool) -> str:
    choices = ""
    for idx, choice in enumerate(example["choices"]):
        choices += f"{chr(65 + idx)}. {choice}\n"
    return build_prompt(
        dataset="mmlu-redux",
        locale="",
        question=example["question"],
        choices=choices,
        use_cot=use_cot,
        use_template=use_template,
    )


def make_chat_text(tokenizer, prompt: str, response_text: str) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text + response_text


def make_kv_cache_index(instruction_length: int, response_length: int, device: torch.device) -> List[torch.Tensor]:
    instruction = torch.tensor([1, 0], dtype=torch.long, device=device).repeat(instruction_length, 1).unsqueeze(0)
    response = torch.tensor([-1, 0], dtype=torch.long, device=device).repeat(response_length, 1).unsqueeze(0)
    return [instruction, response]


def batched(items: List[Dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    texts: List[str],
    response_length: int,
    response_text: str,
    device: torch.device,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)

    full_length = input_ids.shape[1]
    instruction_length = full_length - response_length
    if instruction_length <= 0:
        raise ValueError(f"instruction_length must be positive, got {instruction_length}")

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        kv_cache_index=make_kv_cache_index(instruction_length, response_length, device),
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    generated = outputs[:, full_length:]
    strict_re = re.compile(r"^The correct answer is [A-D]\.$")
    results = []
    for token_ids in generated:
        suffix = tokenizer.decode(token_ids, skip_special_tokens=True)
        content = f"{response_text} {suffix.lstrip()}".strip()
        results.append({
            "content": content,
            "pred": extract_answer_from_content(content),
            "strict_format": bool(strict_re.match(content)),
            "generated_token_count": int(token_ids.shape[0]),
        })
    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_config = DATASET_CONFIGS["mmlu-redux"]
    eval_config = config["eval"]
    output_dir = Path(args.output_dir or config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_subjects = args.subjects.split(",") if args.subjects else eval_config.get("subjects")
    if args.subjects is None and "subjects" in eval_config and eval_config.get("limit") is not None:
        # A smoke recipe may include subjects+limit. For full benchmark runs, ignore
        # those unless --subjects/--limit were explicitly supplied.
        requested_subjects = None
    subjects = requested_subjects or dataset_config["subjects"]
    subjects = [subject.strip() for subject in subjects]

    device = torch.device(args.device)
    print(f"Loading Rosetta model on {device}")
    model, tokenizer = load_rosetta_model(
        config["model"],
        eval_config,
        device=device,
        generation_config=config["model"].get("generation_config", {}),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    response_text = eval_config.get("response_text", "The correct answer is")
    response_length = len(tokenizer(response_text, add_special_tokens=False).input_ids)
    kv_response_length = response_length if args.response_holdout_tokens is None else args.response_holdout_tokens
    if kv_response_length < 1:
        raise ValueError("response-holdout-tokens must be at least 1 because the final section produces logits")
    if kv_response_length > response_length:
        raise ValueError(
            f"response-holdout-tokens={kv_response_length} exceeds response prefix token length {response_length}"
        )
    max_new_tokens = int(config["model"].get("generation_config", {}).get("max_new_tokens", 2))
    use_cot = bool(eval_config.get("use_cot", False))
    use_template = bool(eval_config.get("use_template", True))
    limit = args.limit

    summary = {
        "dataset": "mmlu-redux",
        "answer_method": "generate",
        "constrain_option_tokens": False,
        "config": args.config,
        "checkpoint": config["model"]["rosetta_config"]["checkpoints_dir"],
        "batch_size": args.batch_size,
        "response_text": response_text,
        "response_length": response_length,
        "kv_response_length": kv_response_length,
        "max_new_tokens": max_new_tokens,
        "subjects": {},
        "total": 0,
        "correct": 0,
        "strict_format": 0,
        "parsable": 0,
        "skipped": 0,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    predictions = []
    start_time = time.perf_counter()

    for subject in subjects:
        raw_dataset = load_dataset(dataset_config["dataset_name"], subject)[dataset_config["test_split"]]
        items = []
        skipped = 0
        for idx, example in enumerate(raw_dataset):
            true_answer = parse_mmlu_redux_answer(example)
            if true_answer is None:
                skipped += 1
                continue
            prompt = format_mmlu_redux_prompt(example, use_cot=use_cot, use_template=use_template)
            text = make_chat_text(tokenizer, prompt, response_text)
            items.append({"idx": idx, "text": text, "answer": true_answer})
            if limit is not None and len(items) >= limit:
                break

        subject_correct = 0
        subject_total = 0
        subject_strict = 0
        subject_parsable = 0
        iterator = tqdm(list(batched(items, args.batch_size)), desc=subject, leave=False)
        for batch in iterator:
            generated = generate_batch(
                model=model,
                tokenizer=tokenizer,
                texts=[item["text"] for item in batch],
                response_length=kv_response_length,
                response_text=response_text,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            for item, result in zip(batch, generated):
                is_correct = result["pred"] == item["answer"]
                subject_correct += int(is_correct)
                subject_total += 1
                subject_strict += int(result["strict_format"])
                subject_parsable += int(result["pred"] is not None)
                if args.save_predictions:
                    predictions.append({
                        "subject": subject,
                        "index": item["idx"],
                        "prediction": result["pred"],
                        "answer": item["answer"],
                        "correct": bool(is_correct),
                        "strict_format": result["strict_format"],
                        "content": result["content"],
                    })
            acc = subject_correct / subject_total if subject_total else 0.0
            strict_rate = subject_strict / subject_total if subject_total else 0.0
            iterator.set_postfix(acc=f"{acc:.3f}", strict=f"{strict_rate:.3f}")

        subject_acc = subject_correct / subject_total if subject_total else 0.0
        subject_strict_rate = subject_strict / subject_total if subject_total else 0.0
        subject_parsable_rate = subject_parsable / subject_total if subject_total else 0.0
        summary["subjects"][subject] = {
            "accuracy": subject_acc,
            "correct": subject_correct,
            "total": subject_total,
            "strict_format": subject_strict,
            "strict_format_rate": subject_strict_rate,
            "parsable": subject_parsable,
            "parsable_rate": subject_parsable_rate,
            "skipped": skipped,
        }
        summary["correct"] += subject_correct
        summary["total"] += subject_total
        summary["strict_format"] += subject_strict
        summary["parsable"] += subject_parsable
        summary["skipped"] += skipped
        running_acc = summary["correct"] / summary["total"] if summary["total"] else 0.0
        running_strict = summary["strict_format"] / summary["total"] if summary["total"] else 0.0
        print(
            f"{subject}: {subject_acc * 100:.2f}% "
            f"({subject_correct}/{subject_total}, strict {subject_strict_rate * 100:.2f}%, skipped {skipped}) "
            f"| running {running_acc * 100:.2f}%, strict {running_strict * 100:.2f}%"
        )

    elapsed = time.perf_counter() - start_time
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["elapsed_seconds"] = elapsed
    summary["overall_accuracy"] = summary["correct"] / summary["total"] if summary["total"] else 0.0
    summary["overall_strict_format_rate"] = summary["strict_format"] / summary["total"] if summary["total"] else 0.0
    summary["overall_parsable_rate"] = summary["parsable"] / summary["total"] if summary["total"] else 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"closed_form_kv_mmlu_redux_generate_{timestamp}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.save_predictions:
        pred_path = output_dir / f"closed_form_kv_mmlu_redux_generate_{timestamp}_predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        print(f"Predictions saved to {pred_path}")
    print(f"Summary saved to {summary_path}")
    print(
        f"Overall accuracy: {summary['overall_accuracy'] * 100:.2f}% "
        f"({summary['correct']}/{summary['total']}), "
        f"strict format: {summary['overall_strict_format_rate'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
