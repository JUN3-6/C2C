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
    POSTFUSION_DELTA_FEATURE_SOURCES,
    POSTFUSION_FEATURE_SOURCES,
    POSTFUSION_OPTION_LOGIT_FEATURE_SOURCES,
    POSTFUSION_PROBE_LOGIT_FEATURE_SOURCES,
    PROJECTOR_IN_FEATURE_SOURCES,
    extract_hidden_binned_feature,
    extract_option_logits_feature,
    extract_postfusion_delta_feature,
    extract_probe_logits_feature,
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
    get_option_token_ids,
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


def locate_option_target(
    labels: torch.Tensor,
    option_token_ids: Sequence[int],
) -> Dict[str, int]:
    if labels.ndim != 2:
        raise ValueError(f"labels must be rank-2 [batch, seq], got shape={tuple(labels.shape)}")
    if labels.shape[0] != 1:
        raise ValueError(
            "Option-token labeling currently expects batch size 1 during shard generation."
        )

    token_to_option = {int(token_id): idx for idx, token_id in enumerate(option_token_ids)}
    shift_labels = labels[:, 1:]
    for pos in range(int(shift_labels.shape[1])):
        token_id = int(shift_labels[0, pos].item())
        if token_id in token_to_option:
            return {
                "shift_position": pos,
                "option_index": token_to_option[token_id],
                "token_id": token_id,
            }

    raise ValueError(
        "Could not find any option token in supervised labels. "
        "For `option_token_ce`, use datasets where assistant targets explicitly contain an "
        "A/B/C/D-style answer token."
    )


def compute_option_token_ce(
    logits: torch.Tensor,
    option_token_ids: Sequence[int],
    option_target: Dict[str, int],
) -> Tuple[float, int]:
    stats = compute_option_token_stats(
        logits,
        option_token_ids=option_token_ids,
        option_target=option_target,
    )
    return float(stats["ce"]), 1


def compute_option_token_stats(
    logits: torch.Tensor,
    option_token_ids: Sequence[int],
    option_target: Dict[str, int],
) -> Dict[str, Any]:
    shift_logits = logits[:, :-1, :].contiguous()
    pos = int(option_target["shift_position"])
    target_idx = int(option_target["option_index"])
    if pos < 0 or pos >= int(shift_logits.shape[1]):
        raise ValueError(
            f"Option target position {pos} is out of bounds for shifted logits length "
            f"{int(shift_logits.shape[1])}."
        )

    option_ids = torch.tensor(option_token_ids, dtype=torch.long, device=shift_logits.device)
    option_logits = shift_logits[0, pos, option_ids].unsqueeze(0)
    option_target_tensor = torch.tensor([target_idx], dtype=torch.long, device=shift_logits.device)
    ce = F.cross_entropy(option_logits, option_target_tensor)
    pred_idx = int(torch.argmax(option_logits, dim=-1).item())
    return {
        "ce": float(ce.item()),
        "pred_option": pred_idx,
        "gold_option": target_idx,
        "correct": bool(pred_idx == target_idx),
        "option_logits": option_logits.detach().float().cpu().squeeze(0),
    }


def remap_option_target_for_shift_length(
    option_target: Dict[str, int],
    *,
    full_shift_length: int,
    target_shift_length: int,
) -> Dict[str, int]:
    pos = int(option_target["shift_position"])
    if pos < target_shift_length:
        return option_target

    offset = full_shift_length - target_shift_length
    remapped_pos = pos - offset
    if remapped_pos < 0 or remapped_pos >= target_shift_length:
        raise ValueError(
            "Failed to remap option target position for truncated logits: "
            f"full_shift_length={full_shift_length}, "
            f"target_shift_length={target_shift_length}, "
            f"original_pos={pos}, remapped_pos={remapped_pos}."
        )

    remapped = dict(option_target)
    remapped["shift_position"] = int(remapped_pos)
    return remapped


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
    base_last_logits = None
    prefill_attention_mask = None
    prefill_position_ids = None
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
        base_last_logits = base_output.logits[:, -1, :].detach()
        prefill_attention_mask = base_attention_mask[:, :end]
        prefill_position_ids = section_position_ids[:, -1:]
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
    return (
        current_base_cache,
        current_teacher_cache,
        base_last_hidden,
        teacher_last_hidden,
        base_last_logits,
        prefill_attention_mask,
        prefill_position_ids,
    )


def build_probe_inputs(
    *,
    prefill_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    cache_length: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    probe_input_ids = torch.argmax(prefill_logits.float(), dim=-1, keepdim=True)
    probe_attention_mask = None
    if attention_mask is not None:
        ones = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        probe_attention_mask = torch.cat([attention_mask, ones], dim=1)
    if position_ids is not None:
        probe_position_ids = position_ids[:, -1:] + 1
    else:
        probe_position_ids = torch.full(
            (probe_input_ids.size(0), 1),
            int(cache_length),
            dtype=torch.long,
            device=probe_input_ids.device,
        )
    return probe_input_ids, probe_attention_mask, probe_position_ids


def build_fixed_probe_inputs(
    *,
    probe_token_ids: Sequence[int],
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    cache_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if not probe_token_ids:
        raise ValueError("probe_token_ids must not be empty for option-logit features.")
    batch_size = 1 if attention_mask is None else int(attention_mask.size(0))
    probe_len = len(probe_token_ids)
    probe_input_ids = torch.tensor(
        [int(token_id) for token_id in probe_token_ids],
        dtype=torch.long,
        device=device,
    ).view(1, probe_len).expand(batch_size, -1).contiguous()

    probe_attention_mask = None
    if attention_mask is not None:
        ones = torch.ones(
            (batch_size, probe_len),
            dtype=attention_mask.dtype,
            device=device,
        )
        probe_attention_mask = torch.cat([attention_mask, ones], dim=1)

    if position_ids is not None:
        start = position_ids[:, -1:] + 1
    else:
        start = torch.full(
            (batch_size, 1),
            int(cache_length),
            dtype=torch.long,
            device=device,
        )
    offsets = torch.arange(probe_len, dtype=torch.long, device=device).view(1, -1)
    probe_position_ids = start + offsets
    return probe_input_ids, probe_attention_mask, probe_position_ids


def probe_base_model_logits(
    base_model,
    *,
    cache,
    probe_input_ids: torch.Tensor,
    probe_attention_mask: Optional[torch.Tensor],
    probe_position_ids: torch.Tensor,
) -> torch.Tensor:
    output = base_model(
        input_ids=probe_input_ids,
        attention_mask=probe_attention_mask,
        position_ids=probe_position_ids,
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    return output.logits[:, -1, :].detach()


def compute_receiver_ce(
    batch: Dict[str, Any],
    base_model,
    label_metric: str,
    option_token_ids: Optional[Sequence[int]] = None,
    option_target: Optional[Dict[str, int]] = None,
    return_option_stats: bool = False,
) -> Tuple[float, int] | Tuple[float, int, Optional[Dict[str, Any]]]:
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

    if label_metric == "response_ce":
        ce, token_count = compute_average_ce(outputs.logits, batch["labels"])
        if return_option_stats:
            return ce, token_count, None
        return ce, token_count
    if label_metric == "option_token_ce":
        if option_token_ids is None or option_target is None:
            raise ValueError("option_token_ids and option_target are required for option_token_ce.")
        stats = compute_option_token_stats(
            outputs.logits,
            option_token_ids=option_token_ids,
            option_target=option_target,
        )
        if return_option_stats:
            return float(stats["ce"]), 1, stats
        return float(stats["ce"]), 1
    raise ValueError(f"Unsupported label_metric: {label_metric}")


def compute_bank_ce(
    rosetta_model: RosettaModel,
    batch: Dict[str, Any],
    bank_dict: Dict[str, Any],
    label_metric: str,
    option_token_ids: Optional[Sequence[int]] = None,
    option_target: Optional[Dict[str, int]] = None,
    return_option_stats: bool = False,
) -> Tuple[float, int] | Tuple[float, int, Optional[Dict[str, Any]]]:
    rosetta_model.projector_dict = bank_dict
    outputs = rosetta_model.forward(
        kv_cache_index=batch["kv_cache_index"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        labels=batch["labels"],
        use_cache=True,
    )
    if label_metric == "response_ce":
        metric_ce = float(outputs.loss.item())
        metric_tokens = int(batch["labels"][:, 1:].ne(-100).sum().item())
        stats = None
    elif label_metric == "option_token_ce":
        if option_token_ids is None or option_target is None:
            raise ValueError("option_token_ids and option_target are required for option_token_ce.")
        full_shift_length = int(batch["labels"][:, 1:].shape[1])
        target_shift_length = int(outputs.logits[:, :-1, :].shape[1])
        bank_option_target = remap_option_target_for_shift_length(
            option_target,
            full_shift_length=full_shift_length,
            target_shift_length=target_shift_length,
        )
        stats = compute_option_token_stats(
            outputs.logits,
            option_token_ids=option_token_ids,
            option_target=bank_option_target,
        )
        metric_ce = float(stats["ce"])
        metric_tokens = 1
    else:
        raise ValueError(f"Unsupported label_metric: {label_metric}")
    if return_option_stats:
        return metric_ce, metric_tokens, stats
    return metric_ce, metric_tokens


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
            "hidden_binned",
            "projector_in",
            "projector_in_stats",
            "projector_in_pooled",
            "projector_in_binned",
            "postfusion_delta_stats",
            "postfusion_probe_logits",
            "postfusion_option_logits",
            "postfusion_option_logits_hidden_binned",
        ],
        default="hidden",
        help="Router feature source for label generation.",
    )
    parser.add_argument(
        "--router-input-dim",
        type=int,
        default=896,
        help="Router input dim for feature sources that need a fixed output dim, e.g. hidden_binned.",
    )
    parser.add_argument(
        "--label-metric",
        choices=["response_ce", "option_token_ce"],
        default="response_ce",
        help=(
            "Label CE objective. "
            "`response_ce`: mean CE over supervised response tokens. "
            "`option_token_ce`: CE on A/B/C/D answer token only (MMLU-style)."
        ),
    )
    parser.add_argument(
        "--num-options",
        type=int,
        default=4,
        help="Number of options for option_token_ce mode (max 10).",
    )
    parser.add_argument(
        "--messages-path",
        help="Optional local JSON/JSONL messages file. Overrides data config.",
    )
    parser.add_argument(
        "--option-response-text",
        default="The correct answer is",
        help=(
            "Fixed assistant prefix to probe before A/B/C/D option logits when "
            "--router-feature-source=postfusion_option_logits."
        ),
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
    if (
        args.router_feature_source in (PROJECTOR_IN_FEATURE_SOURCES | POSTFUSION_FEATURE_SOURCES)
        and len(bank_dicts) != 1
    ):
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
    rosetta_model.set_projector_banks(bank_dicts)

    feature_extractor = SimpleKVRouter(
        num_banks=len(bank_dicts),
        feature_source=args.router_feature_source,
        input_dim=args.router_input_dim
        if args.router_feature_source in {"hidden_binned", "postfusion_option_logits_hidden_binned"}
        else 18,
    ).to(device).eval()
    option_token_ids = (
        get_option_token_ids(base_tokenizer, args.num_options)
        if args.label_metric == "option_token_ce"
        else None
    )
    option_feature_token_ids = None
    option_response_token_ids = None
    if args.router_feature_source in POSTFUSION_OPTION_LOGIT_FEATURE_SOURCES:
        option_feature_token_ids = get_option_token_ids(base_tokenizer, args.num_options)
        option_response_token_ids = base_tokenizer(
            args.option_response_text,
            add_special_tokens=False,
        ).input_ids
        if not option_response_token_ids:
            raise ValueError(
                f"--option-response-text produced no tokens: {args.option_response_text!r}"
            )

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
            option_target = None
            if args.label_metric == "option_token_ce":
                option_target = locate_option_target(
                    batch["labels"],
                    option_token_ids=option_token_ids,
                )

            (
                base_cache,
                teacher_cache,
                base_last_hidden,
                teacher_last_hidden,
                base_last_logits,
                prefill_attention_mask,
                prefill_position_ids,
            ) = compute_prefill_caches(
                batch,
                base_model=base_model,
                teacher_model=teacher_model,
            )
            if args.router_feature_source in {"hidden", "hidden_binned"}:
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
            elif args.router_feature_source in POSTFUSION_DELTA_FEATURE_SOURCES:
                prefill_length = int(base_cache.key_cache[0].shape[2])
                fused_cache = rosetta_model._apply_projector_bank_to_cache(
                    base_output_kv_cache=base_cache,
                    source_output_kv_cache=teacher_cache,
                    source_model_idx=1,
                    new_length=prefill_length,
                    bank_selection=None,
                    default_bank_idx=0,
                )
                pooled_feature = extract_postfusion_delta_feature(
                    base_cache=base_cache,
                    fused_cache=fused_cache,
                    projector_bank_config=bank_dicts[0],
                    base_model_idx=0,
                    source_model_idx=1,
                    new_length=prefill_length,
                    feature_source=args.router_feature_source,
                )[0].detach().cpu()
            elif args.router_feature_source in POSTFUSION_PROBE_LOGIT_FEATURE_SOURCES:
                prefill_length = int(base_cache.key_cache[0].shape[2])
                fused_cache = rosetta_model._apply_projector_bank_to_cache(
                    base_output_kv_cache=base_cache,
                    source_output_kv_cache=teacher_cache,
                    source_model_idx=1,
                    new_length=prefill_length,
                    bank_selection=None,
                    default_bank_idx=0,
                )
                probe_input_ids, probe_attention_mask, probe_position_ids = build_probe_inputs(
                    prefill_logits=base_last_logits,
                    attention_mask=prefill_attention_mask,
                    position_ids=prefill_position_ids,
                    cache_length=prefill_length,
                )
                receiver_probe_logits = probe_base_model_logits(
                    base_model,
                    cache=base_cache,
                    probe_input_ids=probe_input_ids,
                    probe_attention_mask=probe_attention_mask,
                    probe_position_ids=probe_position_ids,
                )
                fused_probe_logits = probe_base_model_logits(
                    base_model,
                    cache=fused_cache,
                    probe_input_ids=probe_input_ids,
                    probe_attention_mask=probe_attention_mask,
                    probe_position_ids=probe_position_ids,
                )
                pooled_feature = extract_probe_logits_feature(
                    receiver_logits=receiver_probe_logits,
                    fused_logits=fused_probe_logits,
                    prefill_logits=base_last_logits,
                    feature_source=args.router_feature_source,
                )[0].detach().cpu()
            elif args.router_feature_source in POSTFUSION_OPTION_LOGIT_FEATURE_SOURCES:
                prefill_length = int(base_cache.key_cache[0].shape[2])
                fused_cache = rosetta_model._apply_projector_bank_to_cache(
                    base_output_kv_cache=base_cache,
                    source_output_kv_cache=teacher_cache,
                    source_model_idx=1,
                    new_length=prefill_length,
                    bank_selection=None,
                    default_bank_idx=0,
                )
                probe_input_ids, probe_attention_mask, probe_position_ids = build_fixed_probe_inputs(
                    probe_token_ids=option_response_token_ids,
                    attention_mask=prefill_attention_mask,
                    position_ids=prefill_position_ids,
                    cache_length=prefill_length,
                    device=base_cache.key_cache[0].device,
                )
                receiver_probe_logits = probe_base_model_logits(
                    base_model,
                    cache=base_cache,
                    probe_input_ids=probe_input_ids,
                    probe_attention_mask=probe_attention_mask,
                    probe_position_ids=probe_position_ids,
                )
                fused_probe_logits = probe_base_model_logits(
                    base_model,
                    cache=fused_cache,
                    probe_input_ids=probe_input_ids,
                    probe_attention_mask=probe_attention_mask,
                    probe_position_ids=probe_position_ids,
                )
                pooled_feature = extract_option_logits_feature(
                    receiver_logits=receiver_probe_logits,
                    fused_logits=fused_probe_logits,
                    option_token_ids=option_feature_token_ids,
                    prefill_logits=base_last_logits,
                    feature_source=args.router_feature_source,
                )
                if args.router_feature_source == "postfusion_option_logits_hidden_binned":
                    hidden_feature_dim = args.router_input_dim - int(pooled_feature.size(-1))
                    if hidden_feature_dim <= 0 or hidden_feature_dim % 4 != 0:
                        raise ValueError(
                            "Hybrid option+hidden feature requires "
                            f"router_input_dim={args.router_input_dim} to exceed "
                            f"option_dim={pooled_feature.size(-1)} by 4 * bins."
                        )
                    hidden_feature = extract_hidden_binned_feature(
                        base_last_hidden=base_last_hidden,
                        source_last_hidden=teacher_last_hidden,
                        bins_per_model=hidden_feature_dim // 4,
                    ).to(device=pooled_feature.device, dtype=pooled_feature.dtype)
                    pooled_feature = torch.cat([pooled_feature, hidden_feature], dim=-1)
                pooled_feature = pooled_feature[0].detach().cpu()
            else:
                raise ValueError(
                    f"Unsupported router_feature_source={args.router_feature_source}"
                )

            ce_receiver, metric_token_count, receiver_option_stats = compute_receiver_ce(
                batch,
                base_model,
                label_metric=args.label_metric,
                option_token_ids=option_token_ids,
                option_target=option_target,
                return_option_stats=True,
            )
            ce_fusions = []
            fusion_option_stats = []
            for bank_dict in bank_dicts:
                ce_fusion, _, bank_option_stats = compute_bank_ce(
                    rosetta_model,
                    batch,
                    bank_dict,
                    label_metric=args.label_metric,
                    option_token_ids=option_token_ids,
                    option_target=option_target,
                    return_option_stats=True,
                )
                ce_fusions.append(ce_fusion)
                fusion_option_stats.append(bank_option_stats)

        ce_fusions_tensor = torch.tensor(ce_fusions, dtype=torch.float32)
        improvements = torch.tensor(ce_receiver, dtype=torch.float32) - ce_fusions_tensor
        receiver_correct = (
            None
            if receiver_option_stats is None
            else bool(receiver_option_stats["correct"])
        )
        fusion_corrects = [
            bool(stats["correct"]) if stats is not None else False
            for stats in fusion_option_stats
        ]
        receiver_pred_option = (
            None
            if receiver_option_stats is None
            else int(receiver_option_stats["pred_option"])
        )
        fusion_pred_options = [
            int(stats["pred_option"]) if stats is not None else -1
            for stats in fusion_option_stats
        ]
        gold_option = (
            None
            if receiver_option_stats is None
            else int(receiver_option_stats["gold_option"])
        )
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
                "num_response_tokens": int(batch["labels"][:, 1:].ne(-100).sum().item()),
                "metric_token_count": metric_token_count,
                "label_metric": args.label_metric,
                "option_target": option_target,
                "option_gold": gold_option,
                "receiver_pred_option": receiver_pred_option,
                "fusion_pred_options": torch.tensor(fusion_pred_options, dtype=torch.long),
                "receiver_correct": receiver_correct,
                "fusion_corrects": torch.tensor(fusion_corrects, dtype=torch.bool),
                "feature_source": args.router_feature_source,
                "option_token_ids": (
                    None
                    if option_feature_token_ids is None
                    else torch.tensor(option_feature_token_ids, dtype=torch.long)
                ),
                "option_response_token_ids": (
                    None
                    if option_response_token_ids is None
                    else torch.tensor(option_response_token_ids, dtype=torch.long)
                ),
                "option_response_text": (
                    args.option_response_text
                    if option_response_token_ids is not None
                    else None
                ),
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
            "label_metric": args.label_metric,
            "feature_source": args.router_feature_source,
            "option_token_ids": (
                None
                if option_feature_token_ids is None
                else torch.tensor(option_feature_token_ids, dtype=torch.long)
            ),
            "option_response_token_ids": (
                None
                if option_response_token_ids is None
                else torch.tensor(option_response_token_ids, dtype=torch.long)
            ),
            "option_response_text": (
                args.option_response_text if option_response_token_ids is not None else None
            ),
            "examples": results,
        },
        output_path,
    )
    print(f"Saved {len(results)} labeled examples to {output_path}")


if __name__ == "__main__":
    main()
