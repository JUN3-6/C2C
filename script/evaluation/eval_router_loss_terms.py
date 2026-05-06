#!/usr/bin/env python
"""
Evaluate router loss terms on benchmark data.

This script computes benchmark-side quantities aligned with router training:
  improvement = CE_receiver - CE_fusion
  total_loss = BCE + selection_loss_weight * selection_CE
               + gain_loss_weight * (-expected_routed_gain)

Notes:
- It currently targets 4-way multiple-choice benchmarks with MMLU-Redux-style
  prompting and answer parsing.
- For a single fusion bank (K=1), selection CE is effectively 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.model.router import load_router
from rosetta.model.wrapper import hybrid_to_dynamic
from rosetta.utils.evaluate import get_option_token_ids, load_rosetta_model
from script.evaluation.unified_evaluator import UnifiedEvaluator


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        if path.endswith((".yml", ".yaml")):
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported config format: {path}")


def _resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _router_files(router_dir: Path) -> Tuple[Path, Path]:
    json_candidates = [router_dir / "router.json", router_dir / "router_config.json"]
    pt_candidates = [router_dir / "router.pt", router_dir / "router.bin"]

    config_path = next((p for p in json_candidates if p.exists()), None)
    weight_path = next((p for p in pt_candidates if p.exists()), None)
    if config_path is None:
        raise FileNotFoundError(f"Router config not found in {router_dir}")
    if weight_path is None:
        raise FileNotFoundError(f"Router weights not found in {router_dir}")
    return config_path, weight_path


def _compute_prefill_caches(
    inputs: Dict[str, Any],
    base_model,
    teacher_model,
):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    position_ids = inputs["position_ids"]
    kv_sections = inputs["kv_cache_index"]

    if isinstance(input_ids, list):
        base_input_ids = input_ids[0]
        base_attention_mask = attention_mask[0]
        teacher_input_ids = input_ids[1]
        teacher_attention_mask = attention_mask[1]
    else:
        base_input_ids = input_ids
        base_attention_mask = attention_mask
        teacher_input_ids = input_ids
        teacher_attention_mask = attention_mask

    if len(kv_sections) <= 1:
        raise ValueError("Expected at least one prefill section and one response section.")

    base_cache = None
    teacher_cache = None
    start = 0
    for section in kv_sections[:-1]:
        sec_len = int(section.shape[1])
        end = start + sec_len
        section_pos = position_ids[:, start:end]

        base_out = base_model(
            input_ids=base_input_ids[:, start:end],
            attention_mask=base_attention_mask[:, :end],
            position_ids=section_pos,
            past_key_values=base_cache,
            use_cache=True,
            return_dict=True,
        )
        base_cache = hybrid_to_dynamic(base_out.past_key_values)

        teacher_out = teacher_model(
            input_ids=teacher_input_ids[:, start:end],
            attention_mask=teacher_attention_mask[:, :end],
            position_ids=section_pos,
            past_key_values=teacher_cache,
            use_cache=True,
            return_dict=True,
        )
        teacher_cache = hybrid_to_dynamic(teacher_out.past_key_values)
        start = end

    return base_cache, teacher_cache


def _sample_indices(length: int, sample_interval: int, limit: Any) -> List[int]:
    indices = list(range(0, length, sample_interval))
    if isinstance(limit, int) and limit > 0:
        return indices[:limit]
    if isinstance(limit, (list, tuple)) and len(limit) == 2:
        start, end = limit
        start = 0 if start is None else int(start)
        end = length if end is None else int(end)
        return [i for i in indices if start <= i < end]
    return indices


def _compute_expected_gain(
    binary_logit: torch.Tensor,
    selection_logits: torch.Tensor,
    improvements: torch.Tensor,
    selection_temperature: float,
) -> torch.Tensor:
    if selection_temperature <= 0:
        raise ValueError(f"selection_temperature must be > 0, got {selection_temperature}")
    bank_probs = torch.softmax(selection_logits / selection_temperature, dim=-1)
    expected_bank_gain = torch.sum(bank_probs * improvements, dim=-1)
    fuse_prob = torch.sigmoid(binary_logit)
    return fuse_prob * expected_bank_gain


def _build_targets(improvements: torch.Tensor, skip_margin: float) -> Tuple[int, int]:
    positive_scores = improvements.clone()
    positive_scores[positive_scores <= skip_margin] = float("-inf")
    if torch.isinf(positive_scores).all():
        return 0, -1
    bank_target = int(torch.argmax(positive_scores, dim=-1).item())
    return 1, bank_target


def _compute_expected_gain_single_bank(
    binary_logit: torch.Tensor,
    improvement: torch.Tensor,
) -> torch.Tensor:
    return torch.sigmoid(binary_logit) * improvement


def _safe_mean(total: float, count: int) -> float:
    return float(total) / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Evaluation YAML used for benchmark prompt/model setup.")
    parser.add_argument("--router-dir", required=True, help="Router directory containing router.json and router.pt.")
    parser.add_argument("--output-json", required=True, help="Path to save aggregated metrics JSON.")
    parser.add_argument("--device", help="Device override (e.g., cuda:0).")
    parser.add_argument("--skip-margin", type=float, default=1e-6)
    parser.add_argument("--fuse-threshold", type=float, default=0.5)
    parser.add_argument("--selection-temperature", type=float, default=1.0)
    parser.add_argument("--selection-loss-weight", type=float, default=1.0)
    parser.add_argument("--gain-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--single-bank-loss",
        action="store_true",
        default=True,
        help=(
            "Compute loss in single-bank form: selection CE is forced to 0 and "
            "expected_gain = sigmoid(z) * improvement."
        ),
    )
    parser.add_argument(
        "--full-router-loss",
        dest="single_bank_loss",
        action="store_false",
        help=(
            "Use full router loss (requires improvements for all router banks). "
            "Currently unsupported in this script."
        ),
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        default=os.environ.get("TQDM_DISABLE", "").lower() in {"1", "true", "yes"},
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional subject override. Default uses eval.subjects or dataset defaults.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional per-subject cap overriding eval.limit.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    evaluator = UnifiedEvaluator(config)
    if evaluator.dataset_name != "mmlu-redux":
        raise ValueError(
            "This script currently supports `mmlu-redux` only. "
            f"Got dataset={evaluator.dataset_name!r}."
        )

    device = _resolve_device(args.device)
    fusion_model, tokenizer = load_rosetta_model(
        evaluator.model_config,
        evaluator.eval_config,
        device=device,
        generation_config=evaluator.generation_config,
    )
    fusion_model.eval()

    base_model = fusion_model.model_list[0].eval()
    teacher_model = fusion_model.model_list[1].eval()

    router_config_path, router_weight_path = _router_files(Path(args.router_dir))
    router = load_router(str(router_config_path)).to(device).eval()
    router_sd = torch.load(router_weight_path, map_location=device)
    router.load_state_dict(router_sd, strict=False)
    router_num_banks = int(getattr(router, "num_banks", 1))
    if (not args.single_bank_loss) and router_num_banks != 1:
        raise ValueError(
            f"--full-router-loss requested but router has num_banks={router_num_banks} "
            "while this run only computes one-bank improvements."
        )

    eval_cfg = evaluator.eval_config
    sample_interval = int(eval_cfg.get("sample_interval", 1))
    limit = args.limit if args.limit is not None else eval_cfg.get("limit", None)
    use_cot = bool(eval_cfg.get("use_cot", False))
    use_template = bool(eval_cfg.get("use_template", True))
    proportion = float(eval_cfg.get("kv_cache_proportion", 1.0))
    order_mode = str(eval_cfg.get("kv_cache_order_mode", "front"))

    response_text = eval_cfg.get("response_text", "The correct answer is")
    response_length = len(tokenizer(response_text, add_special_tokens=False).input_ids)
    option_ids = get_option_token_ids(tokenizer, 4)

    dataset_name = evaluator.dataset_config["dataset_name"]
    split_name = evaluator.dataset_config["test_split"]
    subjects = args.subjects or eval_cfg.get("subjects") or evaluator.dataset_config["subjects"]

    totals = defaultdict(float)
    counts = defaultdict(int)
    subject_stats: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for subject in subjects:
            dataset = load_dataset(dataset_name, subject)[split_name]
            indices = _sample_indices(len(dataset), sample_interval=sample_interval, limit=limit)

            sub_totals = defaultdict(float)
            sub_counts = defaultdict(int)
            skipped_no_answer = 0
            skipped_error = 0

            iterator = tqdm(indices, desc=f"loss-eval {subject}", disable=args.disable_tqdm)
            for i in iterator:
                try:
                    example = dataset[i]
                    true_answer = evaluator.parse_answer(example)
                    if true_answer is None:
                        skipped_no_answer += 1
                        continue

                    answer_idx = ord(true_answer) - ord("A")
                    if answer_idx < 0 or answer_idx >= 4:
                        skipped_no_answer += 1
                        continue

                    prompt = evaluator._format_mmlu_redux_example(
                        example,
                        use_cot=use_cot,
                        use_template=use_template,
                    )
                    prepared = evaluator.prepare_model_inputs(
                        prompt=prompt,
                        tokenizer=tokenizer,
                        device=device,
                        model_type="rosetta",
                        llm_tokenizer=None,
                        answer_method="logits",
                        proportion=proportion,
                        order_mode=order_mode,
                    )
                    inputs = prepared["inputs"]

                    receiver_out = base_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        return_dict=True,
                    )
                    fusion_out = fusion_model.forward(**inputs)

                    receiver_option_logits = receiver_out.logits[0, -1, option_ids]
                    fusion_option_logits = fusion_out.logits[0, -1, option_ids]
                    receiver_log_probs = F.log_softmax(receiver_option_logits, dim=-1)
                    fusion_log_probs = F.log_softmax(fusion_option_logits, dim=-1)

                    ce_receiver = float(-receiver_log_probs[answer_idx].item())
                    ce_fusion = float(-fusion_log_probs[answer_idx].item())
                    improvement = ce_receiver - ce_fusion
                    improvements = torch.tensor([[improvement]], device=device, dtype=torch.float32)

                    base_cache, source_cache = _compute_prefill_caches(
                        inputs=inputs,
                        base_model=base_model,
                        teacher_model=teacher_model,
                    )
                    router_out = router(
                        base_cache=base_cache,
                        source_cache=source_cache,
                    )
                    binary_logit = router_out.binary_logits
                    selection_logits = router_out.selection_logits

                    binary_target, bank_target_int = _build_targets(improvements[0], skip_margin=args.skip_margin)
                    binary_target_t = torch.tensor([binary_target], device=device, dtype=torch.float32)

                    binary_loss = F.binary_cross_entropy_with_logits(
                        binary_logit,
                        binary_target_t,
                    )

                    if args.single_bank_loss:
                        selection_loss = selection_logits.new_zeros(())
                        expected_gain = _compute_expected_gain_single_bank(
                            binary_logit=binary_logit,
                            improvement=improvements[:, 0],
                        )
                    else:
                        if bank_target_int >= 0:
                            bank_target_t = torch.tensor([bank_target_int], device=device, dtype=torch.long)
                            selection_loss = F.cross_entropy(selection_logits, bank_target_t)
                        else:
                            selection_loss = selection_logits.new_zeros(())
                        expected_gain = _compute_expected_gain(
                            binary_logit=binary_logit,
                            selection_logits=selection_logits,
                            improvements=improvements,
                            selection_temperature=args.selection_temperature,
                        )
                    gain_loss = -expected_gain.mean()
                    total_loss = (
                        binary_loss
                        + args.selection_loss_weight * selection_loss
                        + args.gain_loss_weight * gain_loss
                    )

                    fuse_prob = torch.sigmoid(binary_logit)[0]
                    pred_fuse = bool(fuse_prob.item() > args.fuse_threshold)
                    pred_bank = int(torch.argmax(selection_logits[0], dim=-1).item())
                    pred_gain_if_fuse = float(improvements[0, pred_bank].item())
                    routed_gain = pred_gain_if_fuse if pred_fuse else 0.0
                    oracle_gain = float(torch.clamp(improvements[0].max(), min=0.0).item())
                    wrong_bank_gain_delta = max(0.0, oracle_gain - pred_gain_if_fuse) if pred_fuse else 0.0
                    harm = 1 if (pred_fuse and pred_gain_if_fuse < 0.0) else 0

                    row = {
                        "ce_receiver": ce_receiver,
                        "ce_fusion": ce_fusion,
                        "improvement": improvement,
                        "binary_target": float(binary_target),
                        "bank_target": float(bank_target_int),
                        "binary_loss": float(binary_loss.item()),
                        "selection_loss": float(selection_loss.item()),
                        "gain_loss": float(gain_loss.item()),
                        "expected_gain": float(expected_gain.mean().item()),
                        "total_loss": float(total_loss.item()),
                        "fuse_prob": float(fuse_prob.item()),
                        "pred_fuse": float(int(pred_fuse)),
                        "pred_gain_if_fuse": float(pred_gain_if_fuse),
                        "routed_gain": float(routed_gain),
                        "oracle_gain": float(oracle_gain),
                        "wrong_bank_gain_delta": float(wrong_bank_gain_delta),
                        "harm": float(harm),
                        "input_len": float(inputs["input_ids"].shape[1]),
                        "response_len": float(response_length),
                    }

                    for k, v in row.items():
                        sub_totals[k] += v
                        totals[k] += v
                    sub_counts["examples"] += 1
                    counts["examples"] += 1

                except Exception:
                    skipped_error += 1
                    continue

            n_sub = sub_counts["examples"]
            if n_sub > 0:
                subject_stats[subject] = {
                    "examples": int(n_sub),
                    "skipped_no_answer": int(skipped_no_answer),
                    "skipped_error": int(skipped_error),
                    "mean_ce_receiver": _safe_mean(sub_totals["ce_receiver"], n_sub),
                    "mean_ce_fusion": _safe_mean(sub_totals["ce_fusion"], n_sub),
                    "mean_improvement": _safe_mean(sub_totals["improvement"], n_sub),
                    "positive_improvement_rate": _safe_mean(sub_totals["binary_target"], n_sub),
                    "mean_total_loss": _safe_mean(sub_totals["total_loss"], n_sub),
                    "mean_binary_loss": _safe_mean(sub_totals["binary_loss"], n_sub),
                    "mean_selection_loss": _safe_mean(sub_totals["selection_loss"], n_sub),
                    "mean_gain_loss": _safe_mean(sub_totals["gain_loss"], n_sub),
                    "mean_expected_gain": _safe_mean(sub_totals["expected_gain"], n_sub),
                    "fuse_rate": _safe_mean(sub_totals["pred_fuse"], n_sub),
                    "harm_rate": _safe_mean(sub_totals["harm"], n_sub),
                    "mean_routed_gain": _safe_mean(sub_totals["routed_gain"], n_sub),
                    "mean_oracle_gain": _safe_mean(sub_totals["oracle_gain"], n_sub),
                }
            else:
                subject_stats[subject] = {
                    "examples": 0,
                    "skipped_no_answer": int(skipped_no_answer),
                    "skipped_error": int(skipped_error),
                }

    n = counts["examples"]
    if n <= 0:
        raise RuntimeError("No valid examples were processed.")

    mean_oracle_gain = _safe_mean(totals["oracle_gain"], n)
    mean_routed_gain = _safe_mean(totals["routed_gain"], n)
    gain_capture = (totals["routed_gain"] / max(totals["oracle_gain"], 1e-8))

    summary = {
        "config_path": str(Path(args.config).resolve()),
        "router_dir": str(Path(args.router_dir).resolve()),
        "dataset": evaluator.dataset_name,
        "subjects": list(subjects),
        "num_examples": int(n),
        "hyperparams": {
            "skip_margin": args.skip_margin,
            "fuse_threshold": args.fuse_threshold,
            "selection_temperature": args.selection_temperature,
            "selection_loss_weight": args.selection_loss_weight,
            "gain_loss_weight": args.gain_loss_weight,
            "single_bank_loss": args.single_bank_loss,
            "router_num_banks": router_num_banks,
        },
        "metrics": {
            "mean_ce_receiver": _safe_mean(totals["ce_receiver"], n),
            "mean_ce_fusion": _safe_mean(totals["ce_fusion"], n),
            "mean_improvement": _safe_mean(totals["improvement"], n),
            "positive_improvement_rate": _safe_mean(totals["binary_target"], n),
            "mean_total_loss": _safe_mean(totals["total_loss"], n),
            "mean_binary_loss": _safe_mean(totals["binary_loss"], n),
            "mean_selection_loss": _safe_mean(totals["selection_loss"], n),
            "mean_gain_loss": _safe_mean(totals["gain_loss"], n),
            "mean_expected_gain": _safe_mean(totals["expected_gain"], n),
            "fuse_rate": _safe_mean(totals["pred_fuse"], n),
            "skip_rate": 1.0 - _safe_mean(totals["pred_fuse"], n),
            "harm_rate": _safe_mean(totals["harm"], n),
            "mean_pred_gain_if_fuse": _safe_mean(totals["pred_gain_if_fuse"], n),
            "mean_routed_gain": mean_routed_gain,
            "mean_oracle_gain": mean_oracle_gain,
            "gain_capture": float(gain_capture),
            "mean_wrong_bank_gain_delta": _safe_mean(totals["wrong_bank_gain_delta"], n),
            "mean_fuse_prob": _safe_mean(totals["fuse_prob"], n),
            "mean_input_len": _safe_mean(totals["input_len"], n),
            "mean_response_len": _safe_mean(totals["response_len"], n),
        },
        "subject_metrics": subject_stats,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {out_path}")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
