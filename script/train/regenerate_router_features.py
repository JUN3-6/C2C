#!/usr/bin/env python
"""
Regenerate router features while reusing existing CE labels/improvements.

This is useful when the dataset, receiver/sharer, and projector bank are
unchanged, but the router input feature has changed. The script rebuilds the
same supervised dataset order, computes only prefill caches and router features,
then copies all non-feature fields from an existing router dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
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
    build_prompt as build_eval_prompt,
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
                    if line:
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
    return create_dataset(data_config["type"], **data_config.get("kwargs", {}))


def build_supervised_dataset(
    message_dataset: Dataset,
    base_tokenizer,
    teacher_tokenizer,
    model_config: Dict[str, Any],
    max_length: int,
):
    aligner = None
    if model_config.get("is_do_alignment", False):
        aligner = TokenAligner(
            slm_tokenizer=base_tokenizer,
            llm_tokenizer=teacher_tokenizer,
            strategy=AlignmentStrategy(model_config.get("alignment_strategy", "first")),
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

    return base_model, teacher_model, base_tokenizer, teacher_tokenizer


def load_projector_banks(
    bank_dirs: List[str],
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
    *,
    need_hidden: bool,
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
            output_hidden_states=need_hidden,
            return_dict=True,
        )
        current_base_cache = hybrid_to_dynamic(base_output.past_key_values)
        base_last_logits = base_output.logits[:, -1, :].detach()
        prefill_attention_mask = base_attention_mask[:, :end]
        prefill_position_ids = section_position_ids[:, -1:]
        if need_hidden:
            base_last_hidden = base_output.hidden_states[-1][:, -1, :].detach()

        teacher_output = teacher_model(
            input_ids=teacher_input_ids[:, start:end],
            attention_mask=teacher_attention_mask[:, :end],
            position_ids=section_position_ids,
            past_key_values=current_teacher_cache,
            use_cache=True,
            output_hidden_states=need_hidden,
            return_dict=True,
        )
        current_teacher_cache = hybrid_to_dynamic(teacher_output.past_key_values)
        if need_hidden:
            teacher_last_hidden = teacher_output.hidden_states[-1][:, -1, :].detach()
        start = end

    return (
        current_base_cache,
        current_teacher_cache,
        base_last_hidden,
        teacher_last_hidden,
        base_last_logits,
        prefill_attention_mask,
        prefill_position_ids,
    )


def _parse_mmlu_user_prompt(user_prompt: str) -> Tuple[str, str]:
    question_prefix = "Question:"
    choices_marker = "\n\nChoices:\n"
    if user_prompt.startswith(question_prefix) and choices_marker in user_prompt:
        question, choices = user_prompt[len(question_prefix):].split(choices_marker, 1)
        return question.strip(), choices.strip() + "\n"
    if "\nChoices:\n" in user_prompt:
        question, choices = user_prompt.split("\nChoices:\n", 1)
        return question.removeprefix(question_prefix).strip(), choices.strip() + "\n"
    return user_prompt.strip(), ""


def build_eval_mmlu_text(messages: List[Dict[str, str]], tokenizer) -> str:
    question, choices = _parse_mmlu_user_prompt(messages[0].get("content", ""))
    eval_prompt = build_eval_prompt(
        dataset="mmlu-redux",
        locale="",
        question=question,
        choices=choices,
        use_cot=False,
        use_template=True,
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": eval_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def make_generate_prefill_batch(text: str, tokenizer, device: torch.device) -> Dict[str, Any]:
    tokenized = tokenizer(text, return_tensors="pt").to(device)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    full_length = int(input_ids.shape[1])
    response_length = 1
    instruction_length = full_length - response_length
    if instruction_length <= 0:
        raise ValueError("Eval-prompt feature regeneration needs at least one prefill token.")

    kv_cache_index = [
        torch.tensor([1, 0], dtype=torch.long, device=device)
        .repeat(instruction_length, 1)
        .unsqueeze(0),
        torch.tensor([-1, 0], dtype=torch.long, device=device)
        .repeat(response_length, 1)
        .unsqueeze(0),
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": attention_mask.long().cumsum(-1) - 1,
        "kv_cache_index": kv_cache_index,
    }


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
    probe_token_ids: List[int],
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
        probe_token_ids,
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


def regenerate_feature(
    *,
    feature_source: str,
    feature_extractor: SimpleKVRouter,
    base_cache,
    teacher_cache,
    base_last_hidden,
    teacher_last_hidden,
    base_last_logits,
    prefill_attention_mask,
    prefill_position_ids,
    base_model,
    projector_list: List[torch.nn.Module],
    bank_dicts: List[Dict[str, Any]],
    rosetta_model: Optional[RosettaModel] = None,
    option_token_ids: Optional[List[int]] = None,
    option_response_token_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    if feature_source in {"hidden", "hidden_binned"}:
        if base_last_hidden is None or teacher_last_hidden is None:
            raise ValueError("hidden feature requested but hidden states were not captured.")
        return feature_extractor.extract_query_feature_from_hidden(
            base_last_hidden,
            teacher_last_hidden,
        )[0].detach().cpu()
    if feature_source == "kv":
        return feature_extractor.extract_query_feature(base_cache, teacher_cache)[0].detach().cpu()
    if feature_source in PROJECTOR_IN_FEATURE_SOURCES:
        if len(bank_dicts) != 1:
            raise ValueError(
                f"{feature_source} feature regeneration currently requires one bank."
            )
        return extract_projector_in_feature(
            base_cache=base_cache,
            source_cache=teacher_cache,
            projector_list=projector_list,
            projector_bank_config=bank_dicts[0],
            base_model_idx=0,
            source_model_idx=1,
            new_length=None,
            feature_source=feature_source,
        )[0].detach().cpu()
    if feature_source in POSTFUSION_DELTA_FEATURE_SOURCES:
        if len(bank_dicts) != 1:
            raise ValueError(
                f"{feature_source} feature regeneration currently requires one bank."
            )
        if rosetta_model is None:
            raise ValueError("post-fusion feature regeneration requires a RosettaModel.")
        prefill_length = int(base_cache.key_cache[0].shape[2])
        fused_cache = rosetta_model._apply_projector_bank_to_cache(
            base_output_kv_cache=base_cache,
            source_output_kv_cache=teacher_cache,
            source_model_idx=1,
            new_length=prefill_length,
            bank_selection=None,
            default_bank_idx=0,
        )
        return extract_postfusion_delta_feature(
            base_cache=base_cache,
            fused_cache=fused_cache,
            projector_bank_config=bank_dicts[0],
            base_model_idx=0,
            source_model_idx=1,
            new_length=prefill_length,
            feature_source=feature_source,
        )[0].detach().cpu()
    if feature_source in POSTFUSION_PROBE_LOGIT_FEATURE_SOURCES:
        if len(bank_dicts) != 1:
            raise ValueError(
                f"{feature_source} feature regeneration currently requires one bank."
            )
        if rosetta_model is None:
            raise ValueError("post-fusion feature regeneration requires a RosettaModel.")
        if base_last_logits is None:
            raise ValueError("post-fusion probe logits require captured prefill logits.")
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
        receiver_logits = probe_base_model_logits(
            base_model,
            cache=base_cache,
            probe_input_ids=probe_input_ids,
            probe_attention_mask=probe_attention_mask,
            probe_position_ids=probe_position_ids,
        )
        fused_logits = probe_base_model_logits(
            base_model,
            cache=fused_cache,
            probe_input_ids=probe_input_ids,
            probe_attention_mask=probe_attention_mask,
            probe_position_ids=probe_position_ids,
        )
        return extract_probe_logits_feature(
            receiver_logits=receiver_logits,
            fused_logits=fused_logits,
            prefill_logits=base_last_logits,
            feature_source=feature_source,
        )[0].detach().cpu()
    if feature_source in POSTFUSION_OPTION_LOGIT_FEATURE_SOURCES:
        if len(bank_dicts) != 1:
            raise ValueError(
                f"{feature_source} feature regeneration currently requires one bank."
            )
        if rosetta_model is None:
            raise ValueError("post-fusion option-logit feature regeneration requires a RosettaModel.")
        if option_token_ids is None or option_response_token_ids is None:
            raise ValueError(
                "post-fusion option-logit features require option_token_ids and "
                "option_response_token_ids."
            )
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
            device=base_last_logits.device if base_last_logits is not None else base_cache.key_cache[0].device,
        )
        receiver_logits = probe_base_model_logits(
            base_model,
            cache=base_cache,
            probe_input_ids=probe_input_ids,
            probe_attention_mask=probe_attention_mask,
            probe_position_ids=probe_position_ids,
        )
        fused_logits = probe_base_model_logits(
            base_model,
            cache=fused_cache,
            probe_input_ids=probe_input_ids,
            probe_attention_mask=probe_attention_mask,
            probe_position_ids=probe_position_ids,
        )
        pooled_feature = extract_option_logits_feature(
            receiver_logits=receiver_logits,
            fused_logits=fused_logits,
            option_token_ids=option_token_ids,
            prefill_logits=base_last_logits,
            feature_source=feature_source,
        )
        if feature_source == "postfusion_option_logits_hidden_binned":
            if base_last_hidden is None or teacher_last_hidden is None:
                raise ValueError(
                    "Hybrid option+hidden feature requested but hidden states were not captured."
                )
            hidden_feature_dim = int(feature_extractor.input_dim) - int(pooled_feature.size(-1))
            if hidden_feature_dim <= 0 or hidden_feature_dim % 4 != 0:
                raise ValueError(
                    "Hybrid option+hidden feature requires input_dim to exceed "
                    f"option_dim={pooled_feature.size(-1)} by 4 * bins, got "
                    f"input_dim={feature_extractor.input_dim}."
                )
            hidden_feature = extract_hidden_binned_feature(
                base_last_hidden=base_last_hidden,
                source_last_hidden=teacher_last_hidden,
                bins_per_model=hidden_feature_dim // 4,
            ).to(device=pooled_feature.device, dtype=pooled_feature.dtype)
            pooled_feature = torch.cat([pooled_feature, hidden_feature], dim=-1)
        return pooled_feature[0].detach().cpu()
    raise ValueError(f"Unsupported feature_source={feature_source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset", required=True, help="Existing router dataset .pt.")
    parser.add_argument("--output-path", required=True, help="Output router dataset .pt.")
    parser.add_argument("--config", required=True, help="Train-style config JSON/YAML.")
    parser.add_argument(
        "--projector-bank-dirs",
        nargs="+",
        required=True,
        help="One or more projector bank directories.",
    )
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
        required=True,
        help="Feature type to regenerate.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--router-input-dim",
        type=int,
        default=630,
        help=(
            "Router input dim for fixed-size features such as hidden_binned or "
            "postfusion_option_logits_hidden_binned."
        ),
    )
    parser.add_argument(
        "--num-samples-override",
        type=int,
        help="Override data.kwargs.num_samples before building the dataset.",
    )
    parser.add_argument(
        "--messages-path",
        help="Optional local JSON/JSONL messages file. Overrides data config.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["dataset", "eval_mmlu"],
        default="dataset",
        help=(
            "Feature prompt source. `dataset` keeps the supervised training prompt; "
            "`eval_mmlu` rebuilds the MMLU-redux generate prompt used by evaluation."
        ),
    )
    parser.add_argument(
        "--source-indices-key",
        help=(
            "Optional tensor key in the input dataset containing original message-dataset "
            "indices for each row, e.g. generation_source_indices."
        ),
    )
    parser.add_argument(
        "--option-response-text",
        default="The correct answer is",
        help=(
            "Fixed assistant prefix to probe before A/B/C/D option logits when "
            "--router-feature-source=postfusion_option_logits."
        ),
    )
    parser.add_argument(
        "--num-options",
        type=int,
        default=4,
        help="Number of option tokens for postfusion_option_logits.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start row in the existing router dataset.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Exclusive end row in the existing router dataset.",
    )
    args = parser.parse_args()

    input_dataset = torch.load(args.input_dataset, map_location="cpu")
    num_existing = int(input_dataset["pooled_feature"].shape[0])
    start_index = int(args.start_index)
    end_index = num_existing if args.end_index is None else min(int(args.end_index), num_existing)
    if start_index < 0 or start_index > end_index:
        raise ValueError(f"Invalid start/end range: {start_index}:{end_index}")
    row_indices = list(range(start_index, end_index))

    if args.source_indices_key:
        if args.source_indices_key not in input_dataset:
            raise KeyError(
                f"{args.input_dataset} is missing --source-indices-key={args.source_indices_key!r}"
            )
        source_indices_value = input_dataset[args.source_indices_key]
        if not torch.is_tensor(source_indices_value):
            raise TypeError(
                f"{args.source_indices_key} must be a tensor, got {type(source_indices_value)}"
            )
        if int(source_indices_value.shape[0]) < end_index:
            raise ValueError(
                f"{args.source_indices_key} has only {int(source_indices_value.shape[0])} rows, "
                f"but requested up to {end_index}."
            )
        message_indices = [int(source_indices_value[row].item()) for row in row_indices]
    else:
        message_indices = row_indices

    cfg = load_config(args.config)
    model_config = cfg["model"]
    data_config = cfg["data"]
    if args.prompt_style == "eval_mmlu" and model_config.get("is_do_alignment", False):
        raise ValueError("prompt_style=eval_mmlu is currently supported only without token alignment.")
    if args.num_samples_override is not None:
        data_config = dict(data_config)
        kwargs = dict(data_config.get("kwargs", {}))
        kwargs["num_samples"] = args.num_samples_override
        data_config["kwargs"] = kwargs
    if args.messages_path:
        data_config = dict(data_config)
        data_config["messages_path"] = args.messages_path

    device = torch.device(args.device)
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
    max_message_index = max(message_indices) if message_indices else -1
    if len(supervised_dataset) <= max_message_index:
        raise ValueError(
            f"Supervised dataset has only {len(supervised_dataset)} examples, "
            f"but requested message index {max_message_index}."
        )

    projector_list, bank_dicts = load_projector_banks(args.projector_bank_dirs, device)
    rosetta_model = None
    if args.router_feature_source in POSTFUSION_FEATURE_SOURCES:
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
    option_token_ids = None
    option_response_token_ids = None
    if args.router_feature_source in POSTFUSION_OPTION_LOGIT_FEATURE_SOURCES:
        option_token_ids = get_option_token_ids(base_tokenizer, args.num_options)
        option_response_token_ids = base_tokenizer(
            args.option_response_text,
            add_special_tokens=False,
        ).input_ids
        if not option_response_token_ids:
            raise ValueError(
                f"--option-response-text produced no tokens: {args.option_response_text!r}"
            )
    feature_extractor = SimpleKVRouter(
        num_banks=max(1, len(bank_dicts)),
        input_dim=args.router_input_dim
        if args.router_feature_source in {"hidden_binned", "postfusion_option_logits_hidden_binned"}
        else 18,
        feature_source=args.router_feature_source,
        option_token_ids=option_token_ids,
        option_response_token_ids=option_response_token_ids,
        option_response_text=args.option_response_text
        if args.router_feature_source in POSTFUSION_OPTION_LOGIT_FEATURE_SOURCES
        else None,
    ).to(device).eval()

    regenerated_features = []
    need_hidden = args.router_feature_source in {
        "hidden",
        "hidden_binned",
        "postfusion_option_logits_hidden_binned",
    }
    for row_idx, message_idx in tqdm(
        zip(row_indices, message_indices),
        total=len(row_indices),
        desc="regenerate-router-features",
    ):
        if args.prompt_style == "eval_mmlu":
            del row_idx
            text = build_eval_mmlu_text(message_dataset[message_idx], base_tokenizer)
            batch = make_generate_prefill_batch(text, base_tokenizer, device)
        else:
            del row_idx
            feature = supervised_dataset[message_idx]
            batch = collator([feature])
            batch = move_batch_to_device(batch, device)
        with torch.no_grad():
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
                base_model,
                teacher_model,
                need_hidden=need_hidden,
            )
            regenerated_features.append(
                regenerate_feature(
                    feature_source=args.router_feature_source,
                    feature_extractor=feature_extractor,
                    base_cache=base_cache,
                    teacher_cache=teacher_cache,
                    base_last_hidden=base_last_hidden,
                    teacher_last_hidden=teacher_last_hidden,
                    base_last_logits=base_last_logits,
                    prefill_attention_mask=prefill_attention_mask,
                    prefill_position_ids=prefill_position_ids,
                    base_model=base_model,
                    projector_list=projector_list,
                    bank_dicts=bank_dicts,
                    rosetta_model=rosetta_model,
                    option_token_ids=option_token_ids,
                    option_response_token_ids=option_response_token_ids,
                )
            )

    output_dataset = {}
    for key, value in input_dataset.items():
        if isinstance(value, torch.Tensor):
            output_dataset[key] = value[start_index:end_index].clone()
        elif isinstance(value, list):
            output_dataset[key] = value[start_index:end_index]
        else:
            output_dataset[key] = value

    output_dataset["pooled_feature"] = torch.stack(regenerated_features, dim=0).float()
    output_dataset["feature_source"] = args.router_feature_source
    if option_token_ids is not None:
        output_dataset["option_token_ids"] = torch.tensor(option_token_ids, dtype=torch.long)
    if option_response_token_ids is not None:
        output_dataset["option_response_token_ids"] = torch.tensor(
            option_response_token_ids,
            dtype=torch.long,
        )
        output_dataset["option_response_text"] = args.option_response_text
    output_dataset["feature_regeneration"] = {
        "source_dataset": args.input_dataset,
        "config": args.config,
        "projector_bank_dirs": list(args.projector_bank_dirs),
        "start_index": start_index,
        "end_index": end_index,
        "prompt_style": args.prompt_style,
        "source_indices_key": args.source_indices_key,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_dataset, output_path)
    print(
        f"Saved {len(output_dataset['pooled_feature'])} examples with "
        f"{args.router_feature_source} features to {output_path}"
    )


if __name__ == "__main__":
    main()
