import argparse
import copy
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.utils.evaluate import load_rosetta_model
from script.evaluation.run_closed_form_mmlu_redux_generate import (
    batched,
    format_mmlu_redux_prompt,
    generate_batch,
    load_config,
    make_chat_text,
    parse_mmlu_redux_answer,
)
from script.evaluation.unified_evaluator import DATASET_CONFIGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile single target-layer closed-form KV injection on MMLU-Redux.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-per-subject", type=int, default=2)
    parser.add_argument("--subjects", default=None, help="Comma-separated subject list. Default: all MMLU-Redux subjects.")
    parser.add_argument("--layers", default="all", help="'all' or comma-separated target layer indices.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--response-holdout-tokens",
        type=int,
        default=None,
        help="Number of trailing response-prefix tokens to keep receiver-only.",
    )
    return parser.parse_args()


def parse_layers(spec: str, full_projector_dict: Dict[int, Any]) -> List[int]:
    if spec != "all":
        return [int(item.strip()) for item in spec.split(",") if item.strip()]

    layers = set()
    for sources in full_projector_dict.get(0, {}).values():
        layers.update(int(layer) for layer in sources.keys())
    return sorted(layers)


def subset_projector_dict(full_projector_dict: Dict[int, Any], target_layer: int) -> Dict[int, Any]:
    subset: Dict[int, Any] = {}
    for target_model_idx, sources in full_projector_dict.items():
        for source_model_idx, layer_map in sources.items():
            if target_layer not in layer_map:
                continue
            subset.setdefault(target_model_idx, {}).setdefault(source_model_idx, {})[target_layer] = copy.deepcopy(
                layer_map[target_layer]
            )
    return subset


def is_repetition(content: str, response_text: str) -> bool:
    if content.count(response_text) > 1:
        return True
    if re.search(r"\b([A-D])(?:\s+\1\b){4,}", content):
        return True
    return False


def load_items(subjects: List[str], limit_per_subject: Optional[int], use_cot: bool, use_template: bool, tokenizer) -> List[Dict[str, Any]]:
    dataset_config = DATASET_CONFIGS["mmlu-redux"]
    items: List[Dict[str, Any]] = []
    for subject in subjects:
        raw_dataset = load_dataset(dataset_config["dataset_name"], subject)[dataset_config["test_split"]]
        subject_count = 0
        for idx, example in enumerate(raw_dataset):
            true_answer = parse_mmlu_redux_answer(example)
            if true_answer is None:
                continue
            prompt = format_mmlu_redux_prompt(example, use_cot=use_cot, use_template=use_template)
            items.append({"subject": subject, "idx": idx, "prompt": prompt, "answer": true_answer})
            subject_count += 1
            if limit_per_subject is not None and subject_count >= limit_per_subject:
                break
    return items


@torch.no_grad()
def evaluate_items(
    *,
    label: str,
    model,
    tokenizer,
    items: List[Dict[str, Any]],
    projector_dict: Dict[int, Any],
    response_text: str,
    response_length: int,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    save_predictions: bool,
) -> Dict[str, Any]:
    model.projector_dict = copy.deepcopy(projector_dict)
    rows = []
    correct = 0
    strict = 0
    parsable = 0
    repeated = 0
    generated_tokens = 0
    seen = 0

    eval_items_with_text = [
        {
            **item,
            "text": make_chat_text(tokenizer, item["prompt"], response_text),
        }
        for item in items
    ]

    iterator = tqdm(list(batched(eval_items_with_text, batch_size)), desc=label, leave=False)
    for batch in iterator:
        generated = generate_batch(
            model=model,
            tokenizer=tokenizer,
            texts=[item["text"] for item in batch],
            response_length=response_length,
            response_text=response_text,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        for item, result in zip(batch, generated):
            pred = result["pred"]
            content = result["content"]
            is_correct = pred == item["answer"]
            is_repeated = is_repetition(content, response_text)
            correct += int(is_correct)
            strict += int(result["strict_format"])
            parsable += int(pred is not None)
            repeated += int(is_repeated)
            generated_tokens += int(result["generated_token_count"])
            seen += 1
            if save_predictions:
                rows.append(
                    {
                        "profile": label,
                        "subject": item["subject"],
                        "index": item["idx"],
                        "prediction": pred,
                        "answer": item["answer"],
                        "correct": bool(is_correct),
                        "strict_format": bool(result["strict_format"]),
                        "repetition": bool(is_repeated),
                        "content": content,
                    }
                )
        iterator.set_postfix(
            acc=f"{correct / max(1, seen):.3f}",
            strict=f"{strict / max(1, seen):.3f}",
            rep=f"{repeated / max(1, seen):.3f}",
        )

    total = len(items)
    return {
        "label": label,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "strict_format": strict,
        "strict_format_rate": strict / total if total else 0.0,
        "parsable": parsable,
        "parsable_rate": parsable / total if total else 0.0,
        "repetition": repeated,
        "repetition_rate": repeated / total if total else 0.0,
        "avg_generated_tokens": generated_tokens / total if total else 0.0,
        "predictions": rows if save_predictions else None,
    }


def metric_for_layer(metrics: Dict[str, Any], target_layer: int) -> Dict[str, Any]:
    layer = metrics.get("layers", {}).get(str(target_layer))
    if not layer:
        return {}
    key = layer.get("key", {})
    value = layer.get("value", {})
    result = {
        "source_layer": layer.get("source_layer"),
        "projector_idx": layer.get("projector_idx"),
        "key_cosine": key.get("cosine"),
        "key_rmse": key.get("rmse"),
        "key_target_norm": key.get("target_norm"),
        "value_cosine": value.get("cosine"),
        "value_rmse": value.get("rmse"),
        "value_target_norm": value.get("target_norm"),
        "value_pred_norm": value.get("pred_norm"),
    }
    if value.get("target_norm"):
        result["value_nrmse"] = value["rmse"] / value["target_norm"]
        result["value_pred_norm_ratio"] = value.get("pred_norm", 0.0) / value["target_norm"]
    if key.get("target_norm"):
        result["key_nrmse"] = key["rmse"] / key["target_norm"]
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    eval_config = config["eval"]
    dataset_config = DATASET_CONFIGS["mmlu-redux"]
    output_dir = Path(args.output_dir or config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, tokenizer = load_rosetta_model(
        config["model"],
        eval_config,
        device=device,
        generation_config=config["model"].get("generation_config", {}),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_projector_dict = copy.deepcopy(model.projector_dict)
    target_layers = parse_layers(args.layers, full_projector_dict)
    requested_subjects = args.subjects.split(",") if args.subjects else dataset_config["subjects"]
    subjects = [subject.strip() for subject in requested_subjects]
    use_cot = bool(eval_config.get("use_cot", False))
    use_template = bool(eval_config.get("use_template", True))
    response_text = eval_config.get("response_text", "The correct answer is")
    full_response_length = len(tokenizer(response_text, add_special_tokens=False).input_ids)
    response_length = full_response_length if args.response_holdout_tokens is None else args.response_holdout_tokens
    max_new_tokens = int(config["model"].get("generation_config", {}).get("max_new_tokens", 2))
    items = load_items(subjects, args.limit_per_subject, use_cot, use_template, tokenizer)

    checkpoint_dir = Path(config["model"]["rosetta_config"]["checkpoints_dir"])
    metrics_path = checkpoint_dir / "closed_form_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    print(
        f"Profiling {len(target_layers)} target layers on {len(items)} examples "
        f"({len(subjects)} subjects, limit_per_subject={args.limit_per_subject})"
    )

    started = datetime.now().isoformat(timespec="seconds")
    start_time = time.perf_counter()
    all_predictions = []
    results = []

    baseline = evaluate_items(
        label="receiver_only",
        model=model,
        tokenizer=tokenizer,
        items=items,
        projector_dict={},
        response_text=response_text,
        response_length=response_length,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
        save_predictions=args.save_predictions,
    )
    if baseline["predictions"] is not None:
        all_predictions.extend(baseline.pop("predictions"))
    results.append(baseline)
    print(
        f"receiver_only: acc={baseline['accuracy'] * 100:.2f}% "
        f"strict={baseline['strict_format_rate'] * 100:.2f}% "
        f"rep={baseline['repetition_rate'] * 100:.2f}%"
    )

    for target_layer in target_layers:
        projector_dict = subset_projector_dict(full_projector_dict, target_layer)
        result = evaluate_items(
            label=f"target_{target_layer}",
            model=model,
            tokenizer=tokenizer,
            items=items,
            projector_dict=projector_dict,
            response_text=response_text,
            response_length=response_length,
            device=device,
            batch_size=args.batch_size,
            max_new_tokens=max_new_tokens,
            save_predictions=args.save_predictions,
        )
        if result["predictions"] is not None:
            all_predictions.extend(result.pop("predictions"))
        result["target_layer"] = target_layer
        result["delta_accuracy_vs_receiver"] = result["accuracy"] - baseline["accuracy"]
        result["delta_strict_vs_receiver"] = result["strict_format_rate"] - baseline["strict_format_rate"]
        result.update(metric_for_layer(metrics, target_layer))
        results.append(result)
        source_layer = result.get("source_layer")
        print(
            f"target_{target_layer:02d}"
            f"{'' if source_layer is None else f'->source_{source_layer:02d}'}: "
            f"acc={result['accuracy'] * 100:.2f}% "
            f"strict={result['strict_format_rate'] * 100:.2f}% "
            f"rep={result['repetition_rate'] * 100:.2f}% "
            f"d_acc={result['delta_accuracy_vs_receiver'] * 100:+.2f}pp "
            f"v_cos={result.get('value_cosine')}"
        )

    elapsed = time.perf_counter() - start_time
    summary = {
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "config": args.config,
        "checkpoint": str(checkpoint_dir),
        "subjects": subjects,
        "limit_per_subject": args.limit_per_subject,
        "num_examples": len(items),
        "batch_size": args.batch_size,
        "max_new_tokens": max_new_tokens,
        "response_text": response_text,
        "response_length": response_length,
        "results": results,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"layer_profile_{timestamp}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_predictions:
        pred_path = output_dir / f"layer_profile_{timestamp}_predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in all_predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Predictions saved to {pred_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
