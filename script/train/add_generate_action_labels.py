#!/usr/bin/env python
"""
Attach generation-correctness action labels to an existing router feature dataset.

The existing router datasets already contain pooled features. This script keeps
those features and re-labels each example by actually generating with:
  action 0: receiver-only
  action 1..K: fusion with projector bank K

This aligns the router target with generate-time benchmark accuracy instead of
option-token CE or option-logit proxy accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.model.wrapper import RosettaModel
from rosetta.train.dataset_adapters import create_dataset
from rosetta.utils.evaluate import (
    _adjust_config_indices,
    _load_projectors_and_config,
    build_prompt as build_eval_prompt,
    extract_answer_from_content,
    set_default_chat_template,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


CHOICE_LABELS = ["A", "B", "C", "D"]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".json"):
            return json.load(f)
        if config_path.endswith((".yml", ".yaml")):
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported config format: {config_path}")


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_message_dataset(data_config: Dict[str, Any]):
    dataset_type = data_config["type"]
    kwargs = data_config.get("kwargs", {})
    return create_dataset(dataset_type, **kwargs)


def load_models(model_config: Dict[str, Any], device: torch.device, dtype: torch.dtype):
    base_model_path = model_config["base_model"]
    teacher_model_path = model_config["teacher_model"]

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    set_default_chat_template(base_tokenizer, base_model_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map={"": device},
    ).eval()

    teacher_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": device},
    }
    if teacher_model_path == "google/gemma-3-1b-it":
        teacher_kwargs["sliding_window"] = 4096
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        **teacher_kwargs,
    ).eval()
    return base_model, teacher_model, base_tokenizer


def load_projector_banks(bank_dirs: Sequence[str], device: torch.device):
    projector_list = []
    bank_dicts = []
    for bank_dir in bank_dirs:
        bank_projectors, _, raw_config = _load_projectors_and_config(bank_dir, device)
        proj_offset = len(projector_list)
        projector_list.extend(bank_projectors)

        if raw_config.get("format") == "projector_banks_v1":
            bank_configs = raw_config.get("bank_configs", [])
        else:
            bank_configs = [raw_config]

        for bank_config in bank_configs:
            bank_dicts.append(
                _adjust_config_indices(
                    bank_config,
                    proj_offset,
                    actual_source_idx=1,
                )
            )
    if not bank_dicts:
        raise ValueError("No projector banks were loaded.")
    return projector_list, bank_dicts


def _parse_mmlu_user_prompt(user_prompt: str) -> tuple[str, str]:
    question_prefix = "Question:"
    choices_marker = "\n\nChoices:\n"
    if user_prompt.startswith(question_prefix) and choices_marker in user_prompt:
        question, choices = user_prompt[len(question_prefix):].split(choices_marker, 1)
        return question.strip(), choices.strip() + "\n"
    if "\nChoices:\n" in user_prompt:
        question, choices = user_prompt.split("\nChoices:\n", 1)
        question = question.removeprefix(question_prefix).strip()
        return question, choices.strip() + "\n"
    return user_prompt.strip(), ""


def build_generation_prompt(
    messages: List[Dict[str, str]],
    tokenizer,
    prompt_style: str,
) -> str:
    if prompt_style == "dataset":
        user_messages = messages[:-1]
    elif prompt_style == "eval_mmlu":
        question, choices = _parse_mmlu_user_prompt(messages[0].get("content", ""))
        eval_prompt = build_eval_prompt(
            dataset="mmlu-redux",
            locale="",
            question=question,
            choices=choices,
            use_cot=False,
            use_template=True,
        )
        user_messages = [{"role": "user", "content": eval_prompt}]
    else:
        raise ValueError(f"Unsupported prompt_style={prompt_style}")

    return tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def make_rosetta_inputs(text: str, tokenizer, device: torch.device):
    tokenized = tokenizer(text, return_tensors="pt").to(device)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    full_length = int(input_ids.shape[1])
    response_length = 1
    instruction_length = max(0, full_length - response_length)

    sections = []
    if instruction_length > 0:
        sections.append(
            torch.tensor([1, 0], dtype=torch.long, device=device)
            .repeat(instruction_length, 1)
            .unsqueeze(0)
        )
    sections.append(
        torch.tensor([-1, 0], dtype=torch.long, device=device)
        .repeat(response_length, 1)
        .unsqueeze(0)
    )
    position_ids = attention_mask.long().cumsum(-1) - 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "kv_cache_index": sections,
    }


def decode_new_tokens(tokenizer, sequences: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(
        sequences[0, prompt_len:],
        skip_special_tokens=True,
    ).strip()


def generate_receiver(
    base_model,
    tokenizer,
    text: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    tokenized = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = base_model.generate(
            **tokenized,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return decode_new_tokens(tokenizer, outputs, tokenized["input_ids"].shape[1])


def generate_fusion(
    rosetta_model: RosettaModel,
    tokenizer,
    text: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    inputs = make_rosetta_inputs(text, tokenizer, device)
    with torch.no_grad():
        outputs = rosetta_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return decode_new_tokens(tokenizer, outputs, inputs["input_ids"].shape[1])


def extract_gold(messages: List[Dict[str, str]]) -> Optional[str]:
    if not messages:
        return None
    answer = extract_answer_from_content(messages[-1].get("content", ""))
    if answer in CHOICE_LABELS:
        return answer
    return None


def slice_tensor_or_list(value, indices: torch.Tensor):
    if torch.is_tensor(value):
        return value.index_select(0, indices)
    if isinstance(value, list) and len(value) == int(indices.numel()):
        # Already sliced by caller.
        return value
    if isinstance(value, list):
        return [value[int(i)] for i in indices.tolist()]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--projector-bank-dirs", nargs="+", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--sample-strategy",
        choices=["first", "random"],
        default="first",
        help="How to choose examples from the feature/message dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-style",
        choices=["dataset", "eval_mmlu"],
        default="dataset",
        help="Use original dataset chat prompt or the MMLU-redux evaluation prompt.",
    )
    parser.add_argument("--device")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--update-decode-past",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether RosettaModel.generate updates current_past after each decode step. "
            "Default: use model.update_decode_past from config, falling back to True."
        ),
    )
    parser.add_argument(
        "--tie-action",
        choices=["fuse", "skip"],
        default="fuse",
        help="Hard target for both-correct or both-wrong ties.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_config = cfg["model"]
    data_config = cfg["data"]

    feature_payload = torch.load(args.input_dataset, map_location="cpu")
    if "pooled_feature" not in feature_payload:
        raise KeyError(f"{args.input_dataset} is missing `pooled_feature`.")
    num_features = int(feature_payload["pooled_feature"].shape[0])

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    base_model, teacher_model, tokenizer = load_models(model_config, device, dtype)
    message_dataset = build_message_dataset(data_config)
    if len(message_dataset) < num_features:
        raise ValueError(
            f"Message dataset has {len(message_dataset)} samples, but feature dataset has "
            f"{num_features} samples."
        )

    projector_list, bank_dicts = load_projector_banks(args.projector_bank_dirs, device)
    if len(bank_dicts) != 1:
        raise ValueError("Generation-action labeling currently expects exactly one bank.")
    update_decode_past = (
        bool(model_config.get("update_decode_past", True))
        if args.update_decode_past is None
        else bool(args.update_decode_past)
    )

    rosetta_model = RosettaModel(
        model_list=[base_model, teacher_model],
        base_model_idx=0,
        projector_list=projector_list,
        include_response=model_config.get("include_response", False),
        multi_source_fusion_mode=model_config.get("multi_source_fusion_mode", "sequential"),
        static_gate_enabled=True,
        entropy_gate_enabled=False,
        update_decode_past=update_decode_past,
    ).to(device).eval()
    rosetta_model.set_projector_banks(bank_dicts)
    rosetta_model.projector_dict = bank_dicts[0]

    start = max(0, args.start_index)
    candidate_stop = num_features
    if start >= candidate_stop:
        raise ValueError(f"Empty slice: start={start}, stop={candidate_stop}")

    candidate_indices = torch.arange(start, candidate_stop, dtype=torch.long)
    if args.sample_strategy == "random":
        generator = torch.Generator().manual_seed(args.seed)
        candidate_indices = candidate_indices[
            torch.randperm(int(candidate_indices.numel()), generator=generator)
        ]

    if args.max_samples is not None:
        candidate_indices = candidate_indices[: args.max_samples]
    selected_indices = candidate_indices.tolist()
    if not selected_indices:
        raise ValueError("No examples were selected.")

    receiver_corrects = []
    fusion_corrects = []
    receiver_preds = []
    fusion_preds = []
    receiver_outputs = []
    fusion_outputs = []
    golds = []
    action_targets = []
    skipped_no_gold = 0

    for idx in tqdm(selected_indices, desc="generate-action-labels"):
        messages = message_dataset[idx]
        gold = extract_gold(messages)
        if gold is None:
            skipped_no_gold += 1
        prompt = build_generation_prompt(messages, tokenizer, prompt_style=args.prompt_style)

        receiver_text = generate_receiver(
            base_model,
            tokenizer,
            prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        fusion_text = generate_fusion(
            rosetta_model,
            tokenizer,
            prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        receiver_pred = extract_answer_from_content(receiver_text)
        fusion_pred = extract_answer_from_content(fusion_text)
        receiver_ok = bool(gold is not None and receiver_pred == gold)
        fusion_ok = bool(gold is not None and fusion_pred == gold)

        if receiver_ok and not fusion_ok:
            action_target = 0
        elif fusion_ok and not receiver_ok:
            action_target = 1
        else:
            action_target = 1 if args.tie_action == "fuse" else 0

        golds.append(gold or "")
        receiver_preds.append(receiver_pred or "")
        fusion_preds.append(fusion_pred or "")
        receiver_outputs.append(receiver_text)
        fusion_outputs.append(fusion_text)
        receiver_corrects.append(float(receiver_ok))
        fusion_corrects.append(float(fusion_ok))
        action_targets.append(action_target)

    selected = torch.tensor(selected_indices, dtype=torch.long)
    output = {}
    for key, value in feature_payload.items():
        if torch.is_tensor(value) and int(value.shape[0]) == num_features:
            output[key] = value.index_select(0, selected)
        elif isinstance(value, list) and len(value) == num_features:
            output[key] = [value[int(i)] for i in selected.tolist()]
        else:
            output[key] = value

    receiver_correct = torch.tensor(receiver_corrects, dtype=torch.float32)
    fusion_correct = torch.tensor(fusion_corrects, dtype=torch.float32).unsqueeze(-1)
    action_target = torch.tensor(action_targets, dtype=torch.long)

    output["receiver_correct"] = receiver_correct
    output["fusion_corrects"] = fusion_correct
    output["action_utilities"] = torch.cat([receiver_correct.unsqueeze(-1), fusion_correct], dim=-1)
    output["action_target"] = action_target
    output["binary_target"] = (action_target > 0).long()
    output["bank_target"] = torch.where(
        action_target > 0,
        action_target - 1,
        torch.full_like(action_target, -1),
    )
    output["generation_gold"] = golds
    output["receiver_generate_pred"] = receiver_preds
    output["fusion_generate_pred"] = fusion_preds
    output["receiver_generate_output"] = receiver_outputs
    output["fusion_generate_output"] = fusion_outputs
    output["generation_action_labeling"] = {
        "config": args.config,
        "input_dataset": args.input_dataset,
        "projector_bank_dirs": list(args.projector_bank_dirs),
        "start_index": start,
        "candidate_stop_index": candidate_stop,
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "prompt_style": args.prompt_style,
        "selected_indices": selected_indices,
        "max_new_tokens": args.max_new_tokens,
        "tie_action": args.tie_action,
        "update_decode_past": update_decode_past,
        "skipped_no_gold": skipped_no_gold,
    }
    output["generation_source_indices"] = selected

    output_path = Path(args.output_dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    n = len(action_targets)
    receiver_acc = float(receiver_correct.mean().item()) if n else 0.0
    fusion_acc = float(fusion_correct.mean().item()) if n else 0.0
    oracle_acc = float(torch.maximum(receiver_correct, fusion_correct.squeeze(-1)).mean().item()) if n else 0.0
    counts = torch.bincount(action_target, minlength=2).tolist()
    print(f"Saved {n} examples to {output_path}")
    print(
        "Generation label stats: "
        f"receiver_acc={receiver_acc:.4f} fusion_acc={fusion_acc:.4f} "
        f"oracle_acc={oracle_acc:.4f} action_counts={counts}"
    )


if __name__ == "__main__":
    main()
