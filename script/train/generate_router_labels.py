#!/usr/bin/env python
"""
Generate offline router labels from raw response CE.

This script:
1. Loads a supervised chat dataset.
2. Computes receiver-only CE on assistant response tokens.
3. Computes bank-wise fusion CE using pre-trained projector checkpoints.
4. Extracts deterministic pooled routing features from receiver/sharer prefill KV.
5. Saves per-example labels and features for downstream router training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.model.aligner import AlignmentStrategy, TokenAligner
from rosetta.model.router import SimpleKVRouter
from rosetta.model.router_features import (
    PROJECTOR_IN_FEATURE_SOURCES,
    extract_projector_in_feature,
)
from rosetta.model.wrapper import RosettaModel, hybrid_to_dynamic
from rosetta.train.dataset_adapters import (
    AlignedChatDataset,
    ChatDataset,
    RosettaDataCollator,
    create_dataset,
)
from rosetta.utils.evaluate import (
    _adjust_config_indices,
    _load_projectors_and_config,
    set_default_chat_template,
)


class LocalMessagesDataset(Dataset):
    def __init__(self, file_path: str):
        self.samples = self._load_samples(Path(file_path))

    @staticmethod
    def _load_samples(path: Path) -> List[List[Dict[str, str]]]:
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix == ".jsonl":
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            raise ValueError(f"Unsupported local dataset format: {path.suffix}")

        samples = []
        for record in records:
            if isinstance(record, list):
                samples.append(record)
            elif isinstance(record, dict):
                if "messages" in record:
                    samples.append(record["messages"])
                elif "conversations" in record:
                    samples.append(record["conversations"])
                else:
                    raise KeyError(
                        "Local dataset records must contain `messages` or `conversations`."
                    )
            else:
                raise ValueError("Unsupported local dataset record type.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".json"):
            return json.load(f)
        if config_path.endswith((".yml", ".yaml")):
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported config format for {config_path}")


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def build_message_dataset(data_config: Dict[str, Any]) -> Dataset:
    if data_config.get("messages_path"):
        return LocalMessagesDataset(data_config["messages_path"])

    dataset_type = data_config["type"]
    kwargs = data_config.get("kwargs", {})
    return create_dataset(dataset_type, **kwargs)


def build_supervised_dataset(
    message_dataset: Dataset,
    base_tokenizer,
    teacher_tokenizer,
    model_config: Dict[str, Any],
    max_length: int,
):
    aligner = None
    if model_config.get("is_do_alignment", False):
        strategy = model_config.get("alignment_strategy", "first")
        aligner = TokenAligner(
            slm_tokenizer=base_tokenizer,
            llm_tokenizer=teacher_tokenizer,
            strategy=AlignmentStrategy(strategy),
        )
        dataset = AlignedChatDataset(message_dataset, aligner, max_length=max_length)
    else:
        dataset = ChatDataset(message_dataset, base_tokenizer, max_length=max_length)

    collator = RosettaDataCollator(
        slm_tokenizer=base_tokenizer,
        llm_tokenizer=teacher_tokenizer,
        max_length=max_length,
        aligner=aligner,
        do_alignment=model_config.get("is_do_alignment", False),
    )
    return dataset, collator


def compute_average_ce(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    token_loss = flat_loss.view_as(shift_labels)
    token_mask = shift_labels.ne(-100)
    valid_loss = token_loss[token_mask]
    if valid_loss.numel() == 0:
        return float("nan"), 0
    return float(valid_loss.mean().item()), int(valid_loss.numel())


def compute_skip_and_bank_labels(
    improvements: torch.Tensor,
    skip_margin: float,
) -> Dict[str, Any]:
    score_skip = torch.tensor(0.0, dtype=improvements.dtype)
    positive_scores = improvements.clone()
    positive_scores[positive_scores <= skip_margin] = float("-inf")

    if positive_scores.numel() == 0 or torch.all(torch.isinf(positive_scores)):
        return {
            "best_action": "skip",
            "binary_target": 0,
            "bank_target": -1,
            "action_target": 0,
        }

    best_bank = int(torch.argmax(positive_scores).item())
    best_bank_score = positive_scores[best_bank]
    if torch.isinf(best_bank_score) or best_bank_score <= score_skip:
        return {
            "best_action": "skip",
            "binary_target": 0,
            "bank_target": -1,
            "action_target": 0,
        }
    return {
        "best_action": f"bank_{best_bank}",
        "binary_target": 1,
        "bank_target": best_bank,
        "action_target": best_bank + 1,
    }


def hash_messages(messages: Sequence[Dict[str, Any]]) -> str:
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_models_and_tokenizers(
    model_config: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
):
    base_model_path = model_config["base_model"]
    teacher_model_path = model_config["teacher_model"]

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    set_default_chat_template(base_tokenizer, base_model_path)

    teacher_tokenizer = None
    if model_config.get("is_do_alignment", False):
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
            teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
        set_default_chat_template(teacher_tokenizer, teacher_model_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map={"": device},
    ).eval()

    teacher_model_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": device},
    }
    if teacher_model_path == "google/gemma-3-1b-it":
        teacher_model_kwargs["sliding_window"] = 4096
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        **teacher_model_kwargs,
    ).eval()

    return base_model, teacher_model, base_tokenizer, teacher_tokenizer


def load_projector_banks(
    bank_dirs: Iterable[str],
    device: torch.device,
) -> Tuple[List[torch.nn.Module], List[Dict[str, Any]]]:
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
    return projector_list, bank_dicts


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, list):
            if value and torch.is_tensor(value[0]):
                moved[key] = [item.to(device) for item in value]
            else:
                moved[key] = value
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_prefill_caches(
    batch: Dict[str, Any],
    base_model,
    teacher_model,
):
    if isinstance(batch["input_ids"], list):
        base_input_ids = batch["input_ids"][0]
        base_attention_mask = batch["attention_mask"][0]
        teacher_input_ids = batch["input_ids"][1]
        teacher_attention_mask = batch["attention_mask"][1]
    else:
        base_input_ids = batch["input_ids"]
        base_attention_mask = batch["attention_mask"]
        teacher_input_ids = batch["input_ids"]
        teacher_attention_mask = batch["attention_mask"]

    position_ids = batch["position_ids"]
    kv_sections = batch["kv_cache_index"]
    if len(kv_sections) <= 1:
        raise ValueError("Routing features require at least one prefill section before response.")

    current_base_cache = None
    current_teacher_cache = None
    base_last_hidden = None
    teacher_last_hidden = None
    start = 0
    for section in kv_sections[:-1]:
        sec_len = section.shape[1]
        end = start + sec_len
        section_position_ids = position_ids[:, start:end]

        base_output = base_model(
            input_ids=base_input_ids[:, start:end],
            attention_mask=base_attention_mask[:, :end],
            position_ids=section_position_ids,
            past_key_values=current_base_cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        current_base_cache = hybrid_to_dynamic(base_output.past_key_values)
        base_last_hidden = base_output.hidden_states[-1][:, -1, :].detach()

        teacher_output = teacher_model(
            input_ids=teacher_input_ids[:, start:end],
            attention_mask=teacher_attention_mask[:, :end],
            position_ids=section_position_ids,
            past_key_values=current_teacher_cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        current_teacher_cache = hybrid_to_dynamic(teacher_output.past_key_values)
        teacher_last_hidden = teacher_output.hidden_states[-1][:, -1, :].detach()
        start = end

    if base_last_hidden is None or teacher_last_hidden is None:
        raise ValueError("Could not extract prefill terminal hidden states.")
    return current_base_cache, current_teacher_cache, base_last_hidden, teacher_last_hidden


def compute_receiver_ce(
    batch: Dict[str, Any],
    base_model,
) -> Tuple[float, int]:
    if isinstance(batch["input_ids"], list):
        input_ids = batch["input_ids"][0]
        attention_mask = batch["attention_mask"][0]
    else:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

    outputs = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=batch["labels"],
        return_dict=True,
    )

    return compute_average_ce(outputs.logits, batch["labels"])


def compute_bank_ce(
    rosetta_model: RosettaModel,
    batch: Dict[str, Any],
    bank_dict: Dict[str, Any],
) -> Tuple[float, int]:
    rosetta_model.projector_dict = bank_dict
    outputs = rosetta_model.forward(
        kv_cache_index=batch["kv_cache_index"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        labels=batch["labels"],
        use_cache=True,
    )
    last_section_len = batch["kv_cache_index"][-1].shape[1]
    response_labels = batch["labels"][:, -last_section_len:]
    num_response_tokens = int(response_labels.ne(-100).sum().item())
    return float(outputs.loss.item()), num_response_tokens


def maybe_slice_dataset(
    dataset: Dataset,
    max_samples: Optional[int],
    start_index: int,
    end_index: Optional[int],
):
    dataset_len = len(dataset)
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    if start_index > dataset_len:
        raise ValueError(
            f"start_index {start_index} is out of bounds for dataset of size {dataset_len}."
        )

    stop_index = dataset_len if end_index is None else min(end_index, dataset_len)
    if stop_index < start_index:
        raise ValueError("end_index must be greater than or equal to start_index.")

    indices = range(start_index, stop_index)
    if max_samples is None:
        return indices
    return range(start_index, min(start_index + max_samples, stop_index))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to train-style JSON/YAML config.")
    parser.add_argument(
        "--projector-bank-dirs",
        nargs="+",
        required=True,
        help="One or more bank checkpoint directories.",
    )
    parser.add_argument("--output-path", required=True, help="Output .pt file.")
    parser.add_argument("--max-samples", type=int, help="Optional maximum number of examples to process.")
    parser.add_argument(
        "--num-samples-override",
        type=int,
        help="Override data.kwargs.num_samples before building the dataset.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Inclusive dataset start index for sharded/resumable generation.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Exclusive dataset end index for sharded/resumable generation.",
    )
    parser.add_argument("--device", help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--dtype", default="bfloat16", help="Model load dtype.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--skip-margin", type=float, default=1e-6)
    parser.add_argument(
        "--router-feature-source",
        choices=[
            "kv",
            "hidden",
            "projector_in",
            "projector_in_stats",
            "projector_in_pooled",
            "projector_in_binned",
        ],
        default="hidden",
        help="Router feature source for label generation.",
    )
    parser.add_argument(
        "--messages-path",
        help="Optional local JSON/JSONL messages file. Overrides data config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_config = cfg["model"]
    data_config = cfg["data"]
    if args.num_samples_override is not None:
        data_config = dict(data_config)
        kwargs = dict(data_config.get("kwargs", {}))
        kwargs["num_samples"] = args.num_samples_override
        data_config["kwargs"] = kwargs
    if args.messages_path:
        data_config = dict(data_config)
        data_config["messages_path"] = args.messages_path

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)

    base_model, teacher_model, base_tokenizer, teacher_tokenizer = load_models_and_tokenizers(
        model_config,
        device,
        dtype,
    )
    message_dataset = build_message_dataset(data_config)
    supervised_dataset, collator = build_supervised_dataset(
        message_dataset,
        base_tokenizer,
        teacher_tokenizer,
        model_config,
        max_length=args.max_length,
    )

    projector_list, bank_dicts = load_projector_banks(args.projector_bank_dirs, device)
    if not bank_dicts:
        raise ValueError("No projector banks were loaded.")
    if args.router_feature_source in PROJECTOR_IN_FEATURE_SOURCES and len(bank_dicts) != 1:
        raise ValueError(
            f"{args.router_feature_source} router features currently require exactly one bank."
        )

    rosetta_model = RosettaModel(
        model_list=[base_model, teacher_model],
        base_model_idx=0,
        projector_list=projector_list,
        include_response=model_config.get("include_response", False),
        multi_source_fusion_mode=model_config.get("multi_source_fusion_mode", "sequential"),
        static_gate_enabled=False,
        entropy_gate_enabled=False,
    ).to(device).eval()

    feature_extractor = SimpleKVRouter(
        num_banks=len(bank_dicts),
        feature_source=args.router_feature_source,
    ).to(device).eval()

    results = []
    indices = maybe_slice_dataset(
        supervised_dataset,
        args.max_samples,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    disable_tqdm = os.environ.get("TQDM_DISABLE", "").lower() in {"1", "true", "yes"}
    for idx in tqdm(indices, desc="router-labels", disable=disable_tqdm):
        feature = supervised_dataset[idx]
        batch = collator([feature])
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            (
                base_cache,
                teacher_cache,
                base_last_hidden,
                teacher_last_hidden,
            ) = compute_prefill_caches(
                batch,
                base_model=base_model,
                teacher_model=teacher_model,
            )
            if args.router_feature_source == "hidden":
                pooled_feature = feature_extractor.extract_query_feature_from_hidden(
                    base_last_hidden,
                    teacher_last_hidden,
                )[0].detach().cpu()
            elif args.router_feature_source == "kv":
                pooled_feature = feature_extractor.extract_query_feature(
                    base_cache,
                    teacher_cache,
                )[0].detach().cpu()
            elif args.router_feature_source in PROJECTOR_IN_FEATURE_SOURCES:
                pooled_feature = extract_projector_in_feature(
                    base_cache=base_cache,
                    source_cache=teacher_cache,
                    projector_list=projector_list,
                    projector_bank_config=bank_dicts[0],
                    base_model_idx=0,
                    source_model_idx=1,
                    new_length=None,
                    feature_source=args.router_feature_source,
                )[0].detach().cpu()
            else:
                raise ValueError(
                    f"Unsupported router_feature_source={args.router_feature_source}"
                )

            ce_receiver, num_response_tokens = compute_receiver_ce(batch, base_model)
            ce_fusions = []
            for bank_dict in bank_dicts:
                ce_fusion, _ = compute_bank_ce(rosetta_model, batch, bank_dict)
                ce_fusions.append(ce_fusion)

        ce_fusions_tensor = torch.tensor(ce_fusions, dtype=torch.float32)
        improvements = torch.tensor(ce_receiver, dtype=torch.float32) - ce_fusions_tensor
        label_info = compute_skip_and_bank_labels(
            improvements,
            skip_margin=args.skip_margin,
        )
        raw_messages = message_dataset[idx]
        results.append(
            {
                "sample_idx": idx,
                "sample_id": hash_messages(raw_messages),
                "messages": raw_messages,
                "pooled_feature": pooled_feature,
                "ce_receiver": float(ce_receiver),
                "ce_fusions": ce_fusions_tensor,
                "improvements": improvements,
                "num_response_tokens": num_response_tokens,
                "feature_source": args.router_feature_source,
                **label_info,
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config_path": args.config,
            "base_model": model_config["base_model"],
            "teacher_model": model_config["teacher_model"],
            "projector_bank_dirs": list(args.projector_bank_dirs),
            "feature_source": args.router_feature_source,
            "examples": results,
        },
        output_path,
    )
    print(f"Saved {len(results)} labeled examples to {output_path}")


if __name__ == "__main__":
    main()
