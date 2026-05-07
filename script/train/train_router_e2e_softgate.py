#!/usr/bin/env python
"""
End-to-end soft-gate router training.

Stage-1 default:
  - receiver/base model: frozen
  - sharer/source model: frozen
  - projector bank: frozen
  - router: trainable

The router is trained from response CE directly instead of offline hard labels.
For one projector bank, the soft prefill cache is:

    soft_cache = base_cache + p_fuse * (projected_cache - base_cache)

At inference the saved router is still the normal hard argmax router.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers.cache_utils import DynamicCache
from transformers.optimization import get_scheduler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.model.projector import save_projector
from rosetta.model.router import SharedLatentFusionRouter, SimpleKVRouter, save_router
from rosetta.model.wrapper import hybrid_to_dynamic
from script.train.generate_router_labels_option_ce import (
    build_message_dataset,
    build_supervised_dataset,
    get_option_token_ids,
    load_config,
    load_models_and_tokenizers,
    load_projector_banks,
    locate_option_target,
    move_batch_to_device,
    parse_dtype,
    remap_option_target_for_shift_length,
    resolve_device,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def configure_projector_training(
    projectors: Sequence[torch.nn.Module],
    mode: str,
    trainable_name_substrings: Sequence[str],
) -> None:
    for projector in projectors:
        projector.eval()
        for name, param in projector.named_parameters():
            if mode == "frozen":
                param.requires_grad = False
            elif mode == "full":
                param.requires_grad = True
                projector.train()
            elif mode == "partial":
                param.requires_grad = any(token in name for token in trainable_name_substrings)
                if param.requires_grad:
                    projector.train()
            else:
                raise ValueError(f"Unsupported projector_train_mode={mode}")


def set_module_dropout(module: torch.nn.Module, dropout_p: Optional[float]) -> None:
    if dropout_p is None or dropout_p < 0:
        return
    if dropout_p > 1:
        raise ValueError("--expert-dropout must be <= 1")
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Dropout):
            submodule.p = float(dropout_p)


def parse_device_list(devices: str) -> List[torch.device]:
    parsed = [torch.device(item.strip()) for item in devices.split(",") if item.strip()]
    if not parsed:
        raise ValueError("--projector-devices must contain at least one device.")
    return parsed


def distribute_projectors(
    projectors: Sequence[torch.nn.Module],
    devices: Sequence[torch.device],
) -> None:
    if not devices:
        return
    counts = {str(device): 0 for device in devices}
    for idx, projector in enumerate(projectors):
        target_device = devices[idx % len(devices)]
        projector.to(target_device)
        counts[str(target_device)] = counts.get(str(target_device), 0) + 1
    print(
        "Projector device placement: "
        + ", ".join(f"{device}={count}" for device, count in counts.items())
    )


def first_parameter_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    for param in module.parameters():
        return param.device
    return fallback


def _offset_bank_config_projector_indices(config: Any, offset: int) -> Any:
    if isinstance(config, dict):
        return {
            key: _offset_bank_config_projector_indices(value, offset)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [_offset_bank_config_projector_indices(value, offset) for value in config]
    if isinstance(config, tuple):
        if len(config) == 2:
            return (int(config[0]), int(config[1]) + int(offset))
        return tuple(_offset_bank_config_projector_indices(value, offset) for value in config)
    return config


def clone_single_bank_as_experts(
    projector_list: Sequence[torch.nn.Module],
    bank_dicts: Sequence[Dict[str, Any]],
    *,
    num_experts: int,
) -> Tuple[List[torch.nn.Module], List[Dict[str, Any]]]:
    if num_experts <= 0:
        return list(projector_list), list(bank_dicts)
    if len(bank_dicts) == num_experts:
        return list(projector_list), list(bank_dicts)
    if len(bank_dicts) != 1:
        raise ValueError(
            "--num-experts can only clone a single source bank, or match the "
            f"already-loaded bank count. got loaded_banks={len(bank_dicts)} "
            f"num_experts={num_experts}"
        )
    if not projector_list:
        raise ValueError("Cannot clone experts from an empty projector list.")

    per_bank = len(projector_list)
    expert_projectors: List[torch.nn.Module] = []
    expert_bank_configs: List[Dict[str, Any]] = []
    for expert_idx in range(num_experts):
        offset = expert_idx * per_bank
        expert_projectors.extend(copy.deepcopy(projector) for projector in projector_list)
        expert_bank_configs.append(
            _offset_bank_config_projector_indices(bank_dicts[0], offset)
        )
    print(
        "Cloned single projector bank into experts: "
        f"source_projectors={per_bank} experts={num_experts} "
        f"total_projectors={len(expert_projectors)}"
    )
    return expert_projectors, expert_bank_configs


def scale_linear_initialization(module: torch.nn.Module, scale: float) -> None:
    if scale == 1.0:
        return
    if scale <= 0:
        raise ValueError("--router-init-scale must be positive.")
    with torch.no_grad():
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.Linear):
                submodule.weight.mul_(scale)
                if submodule.bias is not None:
                    submodule.bias.mul_(scale)


class MixedMessageDataset(Dataset):
    def __init__(
        self,
        components: Sequence[Tuple[str, Dataset, Sequence[int]]],
        *,
        seed: int,
    ) -> None:
        self.components = [(name, dataset) for name, dataset, _ in components]
        self.items: List[Tuple[int, int]] = []
        self.source_counts: Dict[str, int] = {}
        for component_idx, (name, _dataset, indices) in enumerate(components):
            self.source_counts[name] = len(indices)
            self.items.extend((component_idx, int(idx)) for idx in indices)
        random.Random(seed).shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        component_idx, source_idx = self.items[idx]
        _name, dataset = self.components[component_idx]
        return dataset[source_idx]


def _with_data_sample_limit(data_config: Dict[str, Any], sample_limit: int) -> Dict[str, Any]:
    copied = dict(data_config)
    kwargs = dict(copied.get("kwargs", {}))
    kwargs["num_samples"] = sample_limit
    copied["kwargs"] = kwargs
    return copied


def _dataset_source_name(
    config_path: Optional[str],
    data_config: Dict[str, Any],
    ordinal: int,
) -> str:
    dataset_type = str(data_config.get("type", f"dataset{ordinal}"))
    if config_path:
        return f"{Path(config_path).parent.name}:{dataset_type}"
    return f"primary:{dataset_type}"


def build_training_message_dataset(
    *,
    primary_data_config: Dict[str, Any],
    mix_config_paths: Sequence[str],
    num_samples: int,
    data_pool_samples: Optional[int],
    seed: int,
) -> Dataset:
    if not mix_config_paths:
        sample_limit = max(num_samples, data_pool_samples or num_samples)
        return build_message_dataset(
            _with_data_sample_limit(primary_data_config, sample_limit)
        )

    source_configs: List[Tuple[Optional[str], Dict[str, Any]]] = [(None, primary_data_config)]
    for path in mix_config_paths:
        source_cfg = load_config(path)
        if "data" not in source_cfg:
            raise KeyError(f"Mix config is missing `data`: {path}")
        source_configs.append((path, dict(source_cfg["data"])))

    num_sources = len(source_configs)
    base_count = num_samples // num_sources
    remainder = num_samples % num_sources
    components = []
    for source_idx, (config_path, data_config) in enumerate(source_configs):
        take_count = base_count + (1 if source_idx < remainder else 0)
        pool_limit = max(take_count, data_pool_samples or take_count)
        source_name = _dataset_source_name(config_path, data_config, source_idx)
        dataset = build_message_dataset(
            _with_data_sample_limit(data_config, pool_limit)
        )
        if len(dataset) < take_count:
            raise ValueError(
                f"Dataset source {source_name} only has {len(dataset)} examples, "
                f"but {take_count} were requested."
            )
        indices = list(range(len(dataset)))
        random.Random(seed + 1009 * (source_idx + 1)).shuffle(indices)
        components.append((source_name, dataset, indices[:take_count]))

    mixed = MixedMessageDataset(components, seed=seed)
    print(
        "Mixed dataset sampling: "
        + ", ".join(f"{name}={count}" for name, count in mixed.source_counts.items())
        + f" total={len(mixed)} seed={seed}"
    )
    return mixed


def select_model_inputs(batch: Dict[str, Any]):
    if isinstance(batch["input_ids"], list):
        base_input_ids = batch["input_ids"][0]
        base_attention_mask = batch["attention_mask"][0]
        source_input_ids = batch["input_ids"][1]
        source_attention_mask = batch["attention_mask"][1]
    else:
        base_input_ids = batch["input_ids"]
        base_attention_mask = batch["attention_mask"]
        source_input_ids = batch["input_ids"]
        source_attention_mask = batch["attention_mask"]
    return base_input_ids, base_attention_mask, source_input_ids, source_attention_mask


def section_offsets(kv_sections: Sequence[torch.Tensor]) -> List[int]:
    offsets = [0]
    for section in kv_sections:
        offsets.append(offsets[-1] + int(section.shape[1]))
    return offsets


def compute_prefill_state(
    batch: Dict[str, Any],
    base_model,
    source_model,
):
    base_input_ids, base_attention_mask, source_input_ids, source_attention_mask = select_model_inputs(batch)
    position_ids = batch["position_ids"]
    kv_sections = batch["kv_cache_index"]
    if len(kv_sections) <= 1:
        raise ValueError("E2E router training needs at least one prefill section and one response section.")

    base_cache = None
    source_cache = None
    base_last_hidden = None
    source_last_hidden = None
    offsets = section_offsets(kv_sections)

    with torch.no_grad():
        for section_idx in range(len(kv_sections) - 1):
            start = offsets[section_idx]
            end = offsets[section_idx + 1]
            pos = position_ids[:, start:end]

            base_out = base_model(
                input_ids=base_input_ids[:, start:end],
                attention_mask=base_attention_mask[:, :end],
                position_ids=pos,
                past_key_values=base_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            base_cache = hybrid_to_dynamic(base_out.past_key_values)
            base_last_hidden = base_out.hidden_states[-1][:, -1, :].detach()

            source_out = source_model(
                input_ids=source_input_ids[:, start:end],
                attention_mask=source_attention_mask[:, :end],
                position_ids=pos,
                past_key_values=source_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            source_cache = hybrid_to_dynamic(source_out.past_key_values)
            source_last_hidden = source_out.hidden_states[-1][:, -1, :].detach()

    if base_cache is None or source_cache is None:
        raise RuntimeError("Failed to build prefill caches.")
    return base_cache, source_cache, base_last_hidden, source_last_hidden, offsets[-2], offsets[-1]


def _entry_pairs(entry: Any) -> List[Tuple[int, int]]:
    if isinstance(entry, tuple):
        return [entry]
    return [(int(src_layer), int(projector_idx)) for src_layer, projector_idx in entry]


def make_dynamic_cache(keys: Sequence[torch.Tensor], values: Sequence[torch.Tensor]) -> DynamicCache:
    return hybrid_to_dynamic(DynamicCache.from_legacy_cache(list(zip(keys, values))))


def infer_base_kv_shape(model) -> Tuple[int, int]:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Cannot infer KV shape from model without config.")
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    num_heads = int(
        getattr(config, "num_key_value_heads", None)
        or getattr(config, "num_attention_heads", 0)
        or 0
    )
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = int(getattr(config, "hidden_size", 0) or 0)
        attn_heads = int(getattr(config, "num_attention_heads", 0) or 0)
        if hidden_size <= 0 or attn_heads <= 0:
            raise ValueError("Cannot infer head_dim from model config.")
        head_dim = hidden_size // attn_heads
    head_dim = int(head_dim)
    if num_layers <= 0 or num_heads <= 0 or head_dim <= 0:
        raise ValueError(
            f"Invalid inferred KV shape: layers={num_layers}, "
            f"kv_heads={num_heads}, head_dim={head_dim}"
        )
    return num_layers, num_heads * head_dim


def interpolate_cache(
    *,
    base_cache: DynamicCache,
    all_fuse_cache: DynamicCache,
    bank_probability: torch.Tensor,
) -> DynamicCache:
    base_cache = hybrid_to_dynamic(base_cache)
    all_fuse_cache = hybrid_to_dynamic(all_fuse_cache)
    gate = bank_probability.view(-1, 1, 1, 1)
    keys = []
    values = []
    for base_key, fuse_key, base_value, fuse_value in zip(
        base_cache.key_cache,
        all_fuse_cache.key_cache,
        base_cache.value_cache,
        all_fuse_cache.value_cache,
    ):
        gate_t = gate.to(device=base_key.device, dtype=base_key.dtype)
        keys.append(base_key + gate_t * (fuse_key - base_key))
        values.append(base_value + gate_t * (fuse_value - base_value))
    return make_dynamic_cache(keys, values)


def mix_bank_caches(
    *,
    bank_caches: Sequence[DynamicCache],
    bank_probabilities: torch.Tensor,
) -> DynamicCache:
    if not bank_caches:
        raise ValueError("mix_bank_caches requires at least one bank cache.")
    bank_caches = [hybrid_to_dynamic(cache) for cache in bank_caches]
    probs = bank_probabilities.float()
    if probs.ndim != 2:
        raise ValueError(f"bank_probabilities must be [B, K], got {tuple(probs.shape)}")
    if probs.size(-1) != len(bank_caches):
        raise ValueError(
            f"Probability/action count mismatch: probs={probs.size(-1)} "
            f"caches={len(bank_caches)}"
        )
    num_layers = len(bank_caches[0].key_cache)
    mixed_keys = []
    mixed_values = []
    for layer_idx in range(num_layers):
        key_terms = []
        value_terms = []
        for expert_idx, cache in enumerate(bank_caches):
            key = cache.key_cache[layer_idx]
            value = cache.value_cache[layer_idx]
            weight = probs[:, expert_idx].to(device=key.device, dtype=key.dtype).view(-1, 1, 1, 1)
            key_terms.append(weight * key)
            value_terms.append(weight.to(device=value.device, dtype=value.dtype) * value)
        mixed_keys.append(torch.stack(key_terms, dim=0).sum(dim=0))
        mixed_values.append(torch.stack(value_terms, dim=0).sum(dim=0))
    return make_dynamic_cache(mixed_keys, mixed_values)


def soft_fuse_cache(
    *,
    base_cache: DynamicCache,
    source_cache: DynamicCache,
    projector_list: Sequence[torch.nn.Module],
    bank_config: Dict[str, Any],
    bank_probability: torch.Tensor,
    base_model_idx: int = 0,
    source_model_idx: int = 1,
) -> DynamicCache:
    base_cache = hybrid_to_dynamic(base_cache)
    source_cache = hybrid_to_dynamic(source_cache)
    fused_keys = list(base_cache.key_cache)
    fused_values = list(base_cache.value_cache)

    target_map = bank_config.get(base_model_idx, bank_config.get(str(base_model_idx), {}))
    source_map = target_map.get(source_model_idx, target_map.get(str(source_model_idx), {}))
    if not source_map:
        return make_dynamic_cache(fused_keys, fused_values)

    for target_layer_idx_raw, entry in source_map.items():
        target_layer_idx = int(target_layer_idx_raw)
        base_key = base_cache.key_cache[target_layer_idx]
        base_value = base_cache.value_cache[target_layer_idx]
        target_kv = (base_key, base_value)

        projected_pairs = []
        for source_layer_idx, projector_idx in _entry_pairs(entry):
            projector = projector_list[int(projector_idx)]
            projector_device = first_parameter_device(projector, base_key.device)
            source_kv = (
                source_cache.key_cache[int(source_layer_idx)].to(projector_device),
                source_cache.value_cache[int(source_layer_idx)].to(projector_device),
            )
            projector_target_kv = (
                base_key.to(projector_device),
                base_value.to(projector_device),
            )
            grad_context = nullcontext() if any(p.requires_grad for p in projector.parameters()) else torch.no_grad()
            with grad_context:
                projected_pairs.append(projector(source_kv, projector_target_kv))

        if not projected_pairs:
            continue

        projected_key, projected_value = projected_pairs[0]
        gate = bank_probability.to(device=base_key.device, dtype=base_key.dtype).view(-1, 1, 1, 1)
        projected_key = projected_key.to(device=base_key.device, dtype=base_key.dtype)
        projected_value = projected_value.to(device=base_value.device, dtype=base_value.dtype)
        fused_keys[target_layer_idx] = base_key + gate * (projected_key - base_key)
        fused_values[target_layer_idx] = base_value + gate * (projected_value - base_value)

    return make_dynamic_cache(fused_keys, fused_values)


def response_ce_from_cache(
    *,
    base_model,
    batch: Dict[str, Any],
    past_key_values: DynamicCache,
    response_start: int,
    response_end: int,
    label_metric: str,
    option_token_ids: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, int]:
    base_input_ids, base_attention_mask, _, _ = select_model_inputs(batch)
    position_ids = batch["position_ids"]
    labels = batch["labels"]

    response_labels = labels[:, response_start:response_end]
    if label_metric == "response_ce":
        valid_tokens = int(response_labels[:, 1:].ne(-100).sum().item())
    elif label_metric == "option_token_ce":
        valid_tokens = int(labels.size(0))
    else:
        raise ValueError(f"Unsupported label_metric={label_metric}")
    if valid_tokens <= 0:
        zero = torch.zeros((), device=base_input_ids.device, dtype=torch.float32)
        return zero, 0

    outputs = base_model(
        input_ids=base_input_ids[:, response_start:response_end],
        attention_mask=base_attention_mask[:, :response_end],
        position_ids=position_ids[:, response_start:response_end],
        past_key_values=past_key_values,
        labels=response_labels,
        use_cache=False,
        return_dict=True,
    )
    if label_metric == "response_ce":
        return outputs.loss, valid_tokens

    if option_token_ids is None:
        raise ValueError("option_token_ids are required for option_token_ce.")
    option_targets = locate_option_targets_batch(labels, option_token_ids)
    full_shift_length = int(labels[:, 1:].shape[1])
    target_shift_length = int(outputs.logits[:, :-1, :].shape[1])
    offset = full_shift_length - target_shift_length
    remapped_positions = option_targets["shift_positions"] - int(offset)
    if torch.any(remapped_positions < 0) or torch.any(remapped_positions >= target_shift_length):
        raise ValueError(
            "Failed to remap at least one option target position for truncated logits: "
            f"full_shift_length={full_shift_length}, target_shift_length={target_shift_length}, "
            f"positions={option_targets['shift_positions'].detach().cpu().tolist()}, "
            f"remapped={remapped_positions.detach().cpu().tolist()}."
        )
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    option_ids = torch.tensor(
        option_token_ids,
        dtype=torch.long,
        device=shift_logits.device,
    )
    row_indices = torch.arange(shift_logits.size(0), device=shift_logits.device)
    option_logits = shift_logits[
        row_indices,
        remapped_positions.to(shift_logits.device),
    ]
    option_logits = option_logits.index_select(dim=-1, index=option_ids)
    return (
        F.cross_entropy(option_logits.float(), option_targets["option_indices"]),
        int(shift_logits.size(0)),
    )


def compute_router_feature(
    router: torch.nn.Module,
    base_last_hidden: torch.Tensor,
    source_last_hidden: torch.Tensor,
    base_cache: Optional[DynamicCache] = None,
    fused_cache: Optional[DynamicCache] = None,
    new_length: Optional[int] = None,
) -> torch.Tensor:
    feature_source = getattr(router, "feature_source", None)
    if feature_source == "shared_latent_receiver":
        if base_cache is None:
            raise ValueError("shared_latent_receiver feature requires base_cache.")
        return router.extract_query_feature_from_fused_cache(
            base_cache,
            new_length=new_length,
        )
    if feature_source == "shared_latent_fusion":
        if fused_cache is None:
            raise ValueError("shared_latent_fusion feature requires fused_cache.")
        return router.extract_query_feature_from_fused_cache(
            fused_cache,
            new_length=new_length,
        )
    if feature_source == "shared_latent_pair":
        if base_cache is None or fused_cache is None:
            raise ValueError("shared_latent_pair feature requires base_cache and fused_cache.")
        return router.extract_query_feature_from_cache_pair(
            base_cache,
            fused_cache,
            new_length=new_length,
        )
    return router.extract_query_feature_from_hidden(base_last_hidden, source_last_hidden).detach()


def locate_option_targets_batch(
    labels: torch.Tensor,
    option_token_ids: Sequence[int],
) -> Dict[str, torch.Tensor]:
    if labels.ndim != 2:
        raise ValueError(f"labels must be rank-2 [batch, seq], got shape={tuple(labels.shape)}")
    token_to_option = {int(token_id): idx for idx, token_id in enumerate(option_token_ids)}
    shift_labels = labels[:, 1:]
    positions = []
    option_indices = []
    token_ids = []
    for row_idx in range(int(shift_labels.shape[0])):
        found = None
        for pos in range(int(shift_labels.shape[1])):
            token_id = int(shift_labels[row_idx, pos].item())
            if token_id in token_to_option:
                found = (pos, token_to_option[token_id], token_id)
                break
        if found is None:
            raise ValueError(
                "Could not find any option token in supervised labels for "
                f"batch row {row_idx}."
            )
        positions.append(found[0])
        option_indices.append(found[1])
        token_ids.append(found[2])
    device = labels.device
    return {
        "shift_positions": torch.tensor(positions, dtype=torch.long, device=device),
        "option_indices": torch.tensor(option_indices, dtype=torch.long, device=device),
        "token_ids": torch.tensor(token_ids, dtype=torch.long, device=device),
    }


def estimate_feature_stats(
    *,
    dataset,
    collator,
    indices: Sequence[int],
    device: torch.device,
    base_model,
    source_model,
    router: SimpleKVRouter,
    max_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    for idx in indices[:max_samples]:
        batch = collator([dataset[idx]])
        batch = move_batch_to_device(batch, device)
        try:
            _, _, base_hidden, source_hidden, _, _ = compute_prefill_state(
                batch,
                base_model=base_model,
                source_model=source_model,
            )
            features.append(compute_router_feature(router, base_hidden, source_hidden).cpu())
        except Exception as exc:
            print(f"Skipping feature-stat sample idx={idx}: {exc}")
    if not features:
        raise RuntimeError("No features were collected for input standardization.")
    stacked = torch.cat(features, dim=0).float()
    return stacked.mean(dim=0), stacked.std(dim=0).clamp_min(1e-6)


def compute_batch_losses(
    *,
    batch: Dict[str, Any],
    base_model,
    source_model,
    projector_list: Sequence[torch.nn.Module],
    bank_config: Dict[str, Any],
    router: SimpleKVRouter,
    fusion_cost_weight: float,
    label_metric: str,
    option_token_ids: Optional[Sequence[int]],
    load_balance_loss_weight: float = 0.0,
    load_balance_loss_type: str = "mse",
    straight_through_routing: bool = False,
    no_skip_routing: bool = False,
    bank_configs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    base_cache, source_cache, base_hidden, source_hidden, response_start, response_end = compute_prefill_state(
        batch,
        base_model=base_model,
        source_model=source_model,
    )
    all_fuse_cache = None
    bank1_soft = None
    if no_skip_routing:
        if bank_configs is None or len(bank_configs) < 1:
            raise ValueError("no-skip routing expects at least one projector bank.")
        bank_caches = [
            soft_fuse_cache(
                base_cache=base_cache,
                source_cache=source_cache,
                projector_list=projector_list,
                bank_config=current_bank_config,
                bank_probability=torch.ones(
                    base_cache.key_cache[0].size(0),
                    device=base_cache.key_cache[0].device,
                ),
            )
            for current_bank_config in bank_configs
        ]
        if getattr(router, "feature_source", None) == "shared_latent_pair":
            if len(bank_caches) != 2:
                raise ValueError("shared_latent_pair no-skip routing expects exactly two banks.")
            feature = compute_router_feature(
                router,
                base_hidden,
                source_hidden,
                base_cache=bank_caches[0],
                fused_cache=bank_caches[1],
            )
        elif getattr(router, "feature_source", None) == "shared_latent_fusion":
            feature = compute_router_feature(
                router,
                base_hidden,
                source_hidden,
                fused_cache=bank_caches[0],
            )
        else:
            feature = compute_router_feature(
                router,
                base_hidden,
                source_hidden,
                base_cache=base_cache,
                fused_cache=bank_caches[0],
            )
        router_out = router.forward_features(feature)
        action_logits_fp32 = router_out.action_logits.float()
        probs = torch.softmax(action_logits_fp32, dim=-1)
        if probs.size(-1) != len(bank_caches):
            raise ValueError(
                "No-skip E2E expects one action per bank: "
                f"logits={probs.size(-1)} banks={len(bank_caches)}."
            )
        if straight_through_routing:
            hard_action = torch.argmax(action_logits_fp32, dim=-1)
            hard_probs = F.one_hot(hard_action, num_classes=len(bank_caches)).to(dtype=probs.dtype)
            bank_probabilities = hard_probs + probs - probs.detach()
        else:
            hard_action = torch.argmax(action_logits_fp32.detach(), dim=-1)
            bank_probabilities = probs
        fused_cache = mix_bank_caches(
            bank_caches=bank_caches,
            bank_probabilities=bank_probabilities,
        )
    elif getattr(router, "feature_source", None) in {"shared_latent_fusion", "shared_latent_pair"}:
        all_fuse_cache = soft_fuse_cache(
            base_cache=base_cache,
            source_cache=source_cache,
            projector_list=projector_list,
            bank_config=bank_config,
            bank_probability=torch.ones(
                base_cache.key_cache[0].size(0),
                device=base_cache.key_cache[0].device,
            ),
        )
    if not no_skip_routing:
        feature = compute_router_feature(
            router,
            base_hidden,
            source_hidden,
            base_cache=base_cache,
            fused_cache=all_fuse_cache,
        )
        router_out = router.forward_features(feature)
        probs = router_out.action_probabilities
        if probs.size(-1) != 2:
            raise ValueError("Soft-gate E2E currently expects one bank: actions=[skip, bank0].")
        p_fuse_soft = probs[:, 1].float()
        if straight_through_routing:
            hard_action = torch.argmax(router_out.action_logits.float(), dim=-1)
            p_fuse_hard = (hard_action > 0).to(dtype=p_fuse_soft.dtype)
            # Forward uses the hard top-1 route; backward follows the soft router probability.
            p_fuse = p_fuse_hard + p_fuse_soft - p_fuse_soft.detach()
        else:
            p_fuse = p_fuse_soft

        if all_fuse_cache is None:
            fused_cache = soft_fuse_cache(
                base_cache=base_cache,
                source_cache=source_cache,
                projector_list=projector_list,
                bank_config=bank_config,
                bank_probability=p_fuse,
            )
        else:
            fused_cache = interpolate_cache(
                base_cache=base_cache,
                all_fuse_cache=all_fuse_cache,
                bank_probability=p_fuse,
            )
    task_loss, valid_tokens = response_ce_from_cache(
        base_model=base_model,
        batch=batch,
        past_key_values=fused_cache,
        response_start=response_start,
        response_end=response_end,
        label_metric=label_metric,
        option_token_ids=option_token_ids,
    )
    if valid_tokens <= 0:
        return None

    if no_skip_routing:
        p_fuse = bank_probabilities.detach().max(dim=-1).values
        p_fuse_soft = probs.detach().max(dim=-1).values
    cost_loss = p_fuse.mean() if not no_skip_routing else task_loss.new_zeros(())
    load_balance_loss = action_load_balance_loss(
        router_out.action_logits,
        loss_type=load_balance_loss_type,
    ) if load_balance_loss_weight > 0 else task_loss.new_zeros(())
    total_loss = task_loss + fusion_cost_weight * cost_loss + load_balance_loss_weight * load_balance_loss
    return {
        "loss": total_loss,
        "task_loss": task_loss.detach(),
        "cost_loss": cost_loss.detach(),
        "load_balance_loss": load_balance_loss.detach(),
        "p_fuse": p_fuse.detach(),
        "p_fuse_soft": p_fuse_soft.detach(),
        "expert_entropy": (-(probs.detach() * probs.detach().clamp_min(1e-8).log()).sum(dim=-1).mean()
                           if no_skip_routing else task_loss.new_zeros(())),
        "selected_expert": (hard_action.detach().float().mean() if no_skip_routing else task_loss.new_zeros(())),
        "valid_tokens": torch.tensor(valid_tokens, device=task_loss.device),
    }


def action_load_balance_loss(action_logits: torch.Tensor, *, loss_type: str = "mse") -> torch.Tensor:
    probs = torch.softmax(action_logits.float(), dim=-1)
    num_actions = probs.size(-1)
    importance = probs.mean(dim=0)
    if loss_type == "mse":
        target = torch.full_like(importance, 1.0 / max(num_actions, 1))
        return num_actions * torch.sum((importance - target).pow(2))
    if loss_type == "switch":
        selected = torch.argmax(probs.detach(), dim=-1)
        load = F.one_hot(selected, num_classes=num_actions).float().mean(dim=0)
        return num_actions * torch.sum(importance * load)
    raise ValueError(f"Unsupported load balance loss type: {loss_type}")


def compute_batch_action_ce_losses(
    *,
    batch: Dict[str, Any],
    base_model,
    source_model,
    projector_list: Sequence[torch.nn.Module],
    bank_config: Dict[str, Any],
    router: torch.nn.Module,
    label_metric: str,
    option_token_ids: Optional[Sequence[int]],
    oracle_ce_margin: float,
    action_class_weights: Optional[torch.Tensor] = None,
    load_balance_loss_weight: float = 0.0,
    load_balance_loss_type: str = "mse",
    no_skip_routing: bool = False,
    bank_configs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    base_cache, source_cache, base_hidden, source_hidden, response_start, response_end = compute_prefill_state(
        batch,
        base_model=base_model,
        source_model=source_model,
    )
    if no_skip_routing:
        if bank_configs is None or len(bank_configs) < 1:
            raise ValueError("no-skip action CE expects at least one projector bank.")
        bank_caches = [
            soft_fuse_cache(
                base_cache=base_cache,
                source_cache=source_cache,
                projector_list=projector_list,
                bank_config=current_bank_config,
                bank_probability=torch.ones(
                    base_cache.key_cache[0].size(0),
                    device=base_cache.key_cache[0].device,
                ),
            )
            for current_bank_config in bank_configs
        ]
        if getattr(router, "feature_source", None) == "shared_latent_pair":
            if len(bank_caches) != 2:
                raise ValueError("shared_latent_pair no-skip action CE expects exactly two banks.")
            feature = compute_router_feature(
                router,
                base_hidden,
                source_hidden,
                base_cache=bank_caches[0],
                fused_cache=bank_caches[1],
            )
        elif getattr(router, "feature_source", None) == "shared_latent_fusion":
            feature = compute_router_feature(
                router,
                base_hidden,
                source_hidden,
                fused_cache=bank_caches[0],
            )
        else:
            feature = compute_router_feature(
                router,
                base_hidden,
                source_hidden,
                base_cache=base_cache,
                fused_cache=bank_caches[0],
            )
        router_out = router.forward_features(feature)
        if router_out.action_logits.size(-1) != len(bank_caches):
            raise ValueError(
                "No-skip action CE expects one router action per bank: "
                f"logits={router_out.action_logits.size(-1)} banks={len(bank_caches)}."
            )

        with torch.no_grad():
            bank_losses = []
            valid_tokens = 0
            for bank_cache in bank_caches:
                bank_loss, current_valid = response_ce_from_cache(
                    base_model=base_model,
                    batch=batch,
                    past_key_values=bank_cache,
                    response_start=response_start,
                    response_end=response_end,
                    label_metric=label_metric,
                    option_token_ids=option_token_ids,
                )
                if current_valid <= 0:
                    return None
                valid_tokens = current_valid
                bank_losses.append(bank_loss.detach())
            loss_stack = torch.stack(bank_losses).view(1, -1)
        target_action = torch.argmin(loss_stack, dim=-1)

        ce_weights = None
        if action_class_weights is not None:
            ce_weights = action_class_weights.to(
                device=router_out.action_logits.device,
                dtype=torch.float32,
            )
        action_logits_fp32 = router_out.action_logits.float()
        action_loss = F.cross_entropy(
            action_logits_fp32,
            target_action,
            weight=ce_weights,
        )
        load_balance_loss = action_logits_fp32.new_zeros(())
        if load_balance_loss_weight > 0:
            load_balance_loss = action_load_balance_loss(
                action_logits_fp32,
                loss_type=load_balance_loss_type,
            )
        total_loss = action_loss + load_balance_loss_weight * load_balance_loss

        pred_action = torch.argmax(action_logits_fp32.detach(), dim=-1)
        routed_loss = loss_stack.gather(1, pred_action.view(-1, 1)).mean()
        oracle_loss = loss_stack.min(dim=1).values.mean()
        probs = router_out.action_probabilities.detach()
        selected_prob = probs.gather(1, pred_action.view(-1, 1)).squeeze(-1)
        return {
            "loss": total_loss,
            "action_loss": action_loss.detach(),
            "load_balance_loss": load_balance_loss.detach(),
            "task_loss": routed_loss.detach(),
            "receiver_loss": loss_stack[:, 0].mean().detach(),
            "fusion_loss": loss_stack.mean().detach(),
            "oracle_loss": oracle_loss.detach(),
            "p_fuse": selected_prob.detach(),
            "target_fuse": target_action.float().detach(),
            "hard_fuse": pred_action.float().detach(),
            "action_correct": (pred_action == target_action).float().detach(),
            "valid_tokens": torch.tensor(valid_tokens, device=action_loss.device),
        }

    all_fuse_cache = soft_fuse_cache(
        base_cache=base_cache,
        source_cache=source_cache,
        projector_list=projector_list,
        bank_config=bank_config,
        bank_probability=torch.ones(
            base_cache.key_cache[0].size(0),
            device=base_cache.key_cache[0].device,
        ),
    )
    feature = compute_router_feature(
        router,
        base_hidden,
        source_hidden,
        base_cache=base_cache,
        fused_cache=all_fuse_cache,
    )
    router_out = router.forward_features(feature)
    if router_out.action_logits.size(-1) != 2:
        raise ValueError("Action-CE training currently expects one bank: actions=[skip, bank0].")

    with torch.no_grad():
        receiver_loss, valid_tokens = response_ce_from_cache(
            base_model=base_model,
            batch=batch,
            past_key_values=base_cache,
            response_start=response_start,
            response_end=response_end,
            label_metric=label_metric,
            option_token_ids=option_token_ids,
        )
        if valid_tokens <= 0:
            return None
        fusion_loss, _ = response_ce_from_cache(
            base_model=base_model,
            batch=batch,
            past_key_values=all_fuse_cache,
            response_start=response_start,
            response_end=response_end,
            label_metric=label_metric,
            option_token_ids=option_token_ids,
        )
        target_action = torch.where(
            (receiver_loss - fusion_loss) > oracle_ce_margin,
            torch.ones((), dtype=torch.long, device=router_out.action_logits.device),
            torch.zeros((), dtype=torch.long, device=router_out.action_logits.device),
        ).view(1)

    ce_weights = None
    if action_class_weights is not None:
        ce_weights = action_class_weights.to(
            device=router_out.action_logits.device,
            dtype=torch.float32,
        )
    action_logits_fp32 = router_out.action_logits.float()
    action_loss = F.cross_entropy(
        action_logits_fp32,
        target_action,
        weight=ce_weights,
    )
    load_balance_loss = action_logits_fp32.new_zeros(())
    if load_balance_loss_weight > 0:
        load_balance_loss = action_load_balance_loss(
            action_logits_fp32,
            loss_type=load_balance_loss_type,
        )
    total_loss = action_loss + load_balance_loss_weight * load_balance_loss

    pred_action = torch.argmax(action_logits_fp32.detach(), dim=-1)
    pred_fuse = pred_action > 0
    routed_loss = torch.where(
        pred_fuse,
        fusion_loss.detach().view(1),
        receiver_loss.detach().view(1),
    ).mean()
    oracle_loss = torch.minimum(receiver_loss.detach(), fusion_loss.detach())
    p_fuse = router_out.action_probabilities[:, 1].detach()
    return {
        "loss": total_loss,
        "action_loss": action_loss.detach(),
        "load_balance_loss": load_balance_loss.detach(),
        "task_loss": routed_loss.detach(),
        "receiver_loss": receiver_loss.detach(),
        "fusion_loss": fusion_loss.detach(),
        "oracle_loss": oracle_loss.detach(),
        "p_fuse": p_fuse,
        "target_fuse": target_action.float().detach(),
        "hard_fuse": pred_fuse.float().detach(),
        "action_correct": (pred_action == target_action).float().detach(),
        "valid_tokens": torch.tensor(valid_tokens, device=action_loss.device),
    }


def compute_batch_delta_regression_losses(
    *,
    batch: Dict[str, Any],
    base_model,
    source_model,
    projector_list: Sequence[torch.nn.Module],
    bank_config: Dict[str, Any],
    router: torch.nn.Module,
    label_metric: str,
    option_token_ids: Optional[Sequence[int]],
    delta_mean: float,
    delta_std: float,
) -> Optional[Dict[str, torch.Tensor]]:
    base_cache, source_cache, base_hidden, source_hidden, response_start, response_end = compute_prefill_state(
        batch,
        base_model=base_model,
        source_model=source_model,
    )
    all_fuse_cache = soft_fuse_cache(
        base_cache=base_cache,
        source_cache=source_cache,
        projector_list=projector_list,
        bank_config=bank_config,
        bank_probability=torch.ones(
            base_cache.key_cache[0].size(0),
            device=base_cache.key_cache[0].device,
        ),
    )
    feature = compute_router_feature(
        router,
        base_hidden,
        source_hidden,
        base_cache=base_cache,
        fused_cache=all_fuse_cache,
    )
    router_out = router.forward_features(feature)
    score = router_out.binary_logits

    with torch.no_grad():
        receiver_loss, valid_tokens = response_ce_from_cache(
            base_model=base_model,
            batch=batch,
            past_key_values=base_cache,
            response_start=response_start,
            response_end=response_end,
            label_metric=label_metric,
            option_token_ids=option_token_ids,
        )
        if valid_tokens <= 0:
            return None
        fusion_loss, _ = response_ce_from_cache(
            base_model=base_model,
            batch=batch,
            past_key_values=all_fuse_cache,
            response_start=response_start,
            response_end=response_end,
            label_metric=label_metric,
            option_token_ids=option_token_ids,
        )
        delta = (receiver_loss - fusion_loss).view_as(score)
        target_score = (delta - float(delta_mean)) / max(float(delta_std), 1e-6)

    loss = F.smooth_l1_loss(score, target_score)
    pred_fuse = score.detach() > 0
    routed_loss = torch.where(
        pred_fuse,
        fusion_loss.detach().view_as(score),
        receiver_loss.detach().view_as(score),
    ).mean()
    target_fuse = delta.detach() > 0
    return {
        "loss": loss,
        "regression_loss": loss.detach(),
        "task_loss": routed_loss.detach(),
        "receiver_loss": receiver_loss.detach(),
        "fusion_loss": fusion_loss.detach(),
        "oracle_loss": torch.minimum(receiver_loss.detach(), fusion_loss.detach()),
        "delta": delta.detach(),
        "target_score": target_score.detach(),
        "score": score.detach(),
        "p_fuse": torch.sigmoid(score.detach()),
        "target_fuse": target_fuse.float(),
        "hard_fuse": pred_fuse.float(),
        "action_correct": (pred_fuse == target_fuse).float(),
        "valid_tokens": torch.tensor(valid_tokens, device=loss.device),
    }


@torch.no_grad()
def estimate_oracle_action_counts(
    *,
    dataset,
    collator,
    indices: Sequence[int],
    device: torch.device,
    base_model,
    source_model,
    projector_list: Sequence[torch.nn.Module],
    bank_config: Dict[str, Any],
    label_metric: str,
    option_token_ids: Optional[Sequence[int]],
    oracle_ce_margin: float,
    oracle_ce_margin_quantile: Optional[float],
    max_samples: int,
    hist_bins: int,
) -> Dict[str, Any]:
    receiver_loss_sum = 0.0
    fusion_loss_sum = 0.0
    deltas = []
    valid = 0
    scan_indices = indices if max_samples <= 0 else indices[:max_samples]
    for scan_pos, idx in enumerate(scan_indices, start=1):
        batch = collator([dataset[idx]])
        batch = move_batch_to_device(batch, device)
        try:
            base_cache, source_cache, _, _, response_start, response_end = compute_prefill_state(
                batch,
                base_model=base_model,
                source_model=source_model,
            )
            all_fuse_cache = soft_fuse_cache(
                base_cache=base_cache,
                source_cache=source_cache,
                projector_list=projector_list,
                bank_config=bank_config,
                bank_probability=torch.ones(
                    base_cache.key_cache[0].size(0),
                    device=base_cache.key_cache[0].device,
                ),
            )
            receiver_loss, valid_tokens = response_ce_from_cache(
                base_model=base_model,
                batch=batch,
                past_key_values=base_cache,
                response_start=response_start,
                response_end=response_end,
                label_metric=label_metric,
                option_token_ids=option_token_ids,
            )
            if valid_tokens <= 0:
                continue
            fusion_loss, _ = response_ce_from_cache(
                base_model=base_model,
                batch=batch,
                past_key_values=all_fuse_cache,
                response_start=response_start,
                response_end=response_end,
                label_metric=label_metric,
                option_token_ids=option_token_ids,
            )
            delta = float((receiver_loss - fusion_loss).item())
            deltas.append(delta)
            receiver_loss_sum += float(receiver_loss.item())
            fusion_loss_sum += float(fusion_loss.item())
            valid += 1
        except Exception as exc:
            print(f"Skipping class-weight scan sample idx={idx}: {exc}")
            continue
        if scan_pos == 1 or scan_pos % 100 == 0:
            print(
                "class-weight scan "
                f"{scan_pos}/{len(scan_indices)} valid={valid}"
            )

    if not deltas:
        raise RuntimeError("No valid deltas were collected during oracle action scan.")

    delta_tensor = torch.tensor(deltas, dtype=torch.float32)
    effective_margin = float(oracle_ce_margin)
    if oracle_ce_margin_quantile is not None:
        effective_margin = float(torch.quantile(delta_tensor, oracle_ce_margin_quantile).item())

    targets = (delta_tensor > effective_margin).long()
    counts = torch.bincount(targets, minlength=2).long()
    raw_counts = torch.bincount((delta_tensor > 0).long(), minlength=2).long()
    eps = 1e-6
    hist_bins = max(1, int(hist_bins))
    hist = torch.histc(
        delta_tensor,
        bins=hist_bins,
        min=float(delta_tensor.min().item()),
        max=float(delta_tensor.max().item()),
    )
    hist_edges = torch.linspace(
        float(delta_tensor.min().item()),
        float(delta_tensor.max().item()),
        steps=hist_bins + 1,
    )
    quantile_points = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    quantiles = {
        f"q{int(q * 100):02d}": float(torch.quantile(delta_tensor, q).item())
        for q in quantile_points
    }
    delta_mean = float(delta_tensor.mean().item())
    delta_std = float(delta_tensor.std(unbiased=False).item())
    fuse_better_ratio = float((delta_tensor > 0).float().mean().item())
    strong_fuse_ratio = float((delta_tensor > eps).float().mean().item())
    skip_better_ratio = float((delta_tensor < 0).float().mean().item())
    near_zero_ratio = float(delta_tensor.abs().le(eps).float().mean().item())

    print("Delta diagnostics (delta = receiver_ce - fusion_ce)")
    print(f"  mean delta: {delta_mean:.6f}")
    print(f"  std delta: {delta_std:.6f}")
    print(f"  fuse better ratio (delta > 0): {fuse_better_ratio:.6f}")
    print(f"  strong fuse ratio (delta > {eps:g}): {strong_fuse_ratio:.6f}")
    print(f"  skip better ratio (delta < 0): {skip_better_ratio:.6f}")
    print(f"  near-zero ratio (abs(delta) <= {eps:g}): {near_zero_ratio:.6f}")
    print(f"  raw margin=0 counts [skip,fuse]: {raw_counts.tolist()}")
    print(f"  effective margin: {effective_margin:.6f}")
    if oracle_ce_margin_quantile is not None:
        print(f"  effective margin source: quantile={oracle_ce_margin_quantile:.3f}")
    print(f"  effective target counts [skip,fuse]: {counts.tolist()}")
    print(f"  quantiles: {quantiles}")
    print("  histogram:")
    for bin_idx in range(hist_bins):
        print(
            "    "
            f"[{float(hist_edges[bin_idx]):.6f}, {float(hist_edges[bin_idx + 1]):.6f}) "
            f"{int(hist[bin_idx].item())}"
        )

    return {
        "counts": counts,
        "raw_counts": raw_counts,
        "num_valid": valid,
        "receiver_loss_mean": receiver_loss_sum / max(valid, 1),
        "fusion_loss_mean": fusion_loss_sum / max(valid, 1),
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_min": float(delta_tensor.min().item()),
        "delta_max": float(delta_tensor.max().item()),
        "fuse_better_ratio": fuse_better_ratio,
        "strong_fuse_ratio": strong_fuse_ratio,
        "skip_better_ratio": skip_better_ratio,
        "near_zero_ratio": near_zero_ratio,
        "effective_margin": effective_margin,
        "oracle_ce_margin_quantile": oracle_ce_margin_quantile,
        "quantiles": quantiles,
        "hist_counts": [int(x) for x in hist.tolist()],
        "hist_edges": [float(x) for x in hist_edges.tolist()],
    }


def build_action_class_weights(
    *,
    counts: torch.Tensor,
    mode: str,
    max_weight: float,
    skip_multiplier: float,
) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    if mode != "inverse":
        raise ValueError(f"Unsupported action_class_weight_mode={mode}")
    if skip_multiplier <= 0:
        raise ValueError("--action-class-skip-multiplier must be > 0")

    counts_f = counts.float().clamp_min(1.0)
    total = counts_f.sum().clamp_min(1.0)
    weights = total / (counts_f.numel() * counts_f)
    weights[0] = weights[0] * float(skip_multiplier)
    if max_weight > 0:
        weights = weights.clamp_max(float(max_weight))
    # Keep the average weight near 1 so LR remains comparable across runs.
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights.float()


@torch.no_grad()
def evaluate_soft(
    *,
    dataset,
    collator,
    indices: Sequence[int],
    device: torch.device,
    base_model,
    source_model,
    projector_list: Sequence[torch.nn.Module],
    bank_config: Dict[str, Any],
    router: SimpleKVRouter,
    fusion_cost_weight: float,
    label_metric: str,
    option_token_ids: Optional[Sequence[int]],
    load_balance_loss_weight: float,
    load_balance_loss_type: str,
    straight_through_routing: bool,
    no_skip_routing: bool,
    bank_configs: Optional[Sequence[Dict[str, Any]]],
    max_samples: int,
) -> Dict[str, float]:
    router.eval()
    totals = {
        "loss": 0.0,
        "task_loss": 0.0,
        "cost_loss": 0.0,
        "load_balance_loss": 0.0,
        "p_fuse": 0.0,
        "p_fuse_soft": 0.0,
        "hard_fuse": 0.0,
        "count": 0,
    }
    for idx in indices[:max_samples]:
        batch = collator([dataset[idx]])
        batch = move_batch_to_device(batch, device)
        try:
            result = compute_batch_losses(
                batch=batch,
                base_model=base_model,
                source_model=source_model,
                projector_list=projector_list,
                bank_config=bank_config,
                router=router,
                fusion_cost_weight=fusion_cost_weight,
                label_metric=label_metric,
                option_token_ids=option_token_ids,
                load_balance_loss_weight=load_balance_loss_weight,
                load_balance_loss_type=load_balance_loss_type,
                straight_through_routing=straight_through_routing,
                no_skip_routing=no_skip_routing,
                bank_configs=bank_configs,
            )
        except Exception as exc:
            print(f"Skipping eval sample idx={idx}: {exc}")
            continue
        if result is None:
            continue
        totals["loss"] += float(result["loss"].item())
        totals["task_loss"] += float(result["task_loss"].item())
        totals["cost_loss"] += float(result["cost_loss"].item())
        totals["load_balance_loss"] += float(result["load_balance_loss"].item())
        totals["p_fuse"] += float(result["p_fuse"].mean().item())
        totals["p_fuse_soft"] += float(result.get("p_fuse_soft", result["p_fuse"]).mean().item())
        totals["hard_fuse"] += float((result["p_fuse"] > 0.5).float().mean().item())
        totals["count"] += 1

    count = max(int(totals["count"]), 1)
    return {
        "loss": totals["loss"] / count,
        "task_loss": totals["task_loss"] / count,
        "cost_loss": totals["cost_loss"] / count,
        "load_balance_loss": totals["load_balance_loss"] / count,
        "p_fuse": totals["p_fuse"] / count,
        "p_fuse_soft": totals["p_fuse_soft"] / count,
        "hard_fuse_rate": totals["hard_fuse"] / count,
        "num_eval": int(totals["count"]),
    }


@torch.no_grad()
def evaluate_action_ce(
    *,
    dataset,
    collator,
    indices: Sequence[int],
    device: torch.device,
    base_model,
    source_model,
    projector_list: Sequence[torch.nn.Module],
    bank_config: Dict[str, Any],
    router: torch.nn.Module,
    label_metric: str,
    option_token_ids: Optional[Sequence[int]],
    oracle_ce_margin: float,
    action_class_weights: Optional[torch.Tensor],
    load_balance_loss_weight: float,
    load_balance_loss_type: str,
    no_skip_routing: bool,
    bank_configs: Optional[Sequence[Dict[str, Any]]],
    max_samples: int,
) -> Dict[str, float]:
    router.eval()
    totals = {
        "loss": 0.0,
        "task_loss": 0.0,
        "receiver_loss": 0.0,
        "fusion_loss": 0.0,
        "oracle_loss": 0.0,
        "action_loss": 0.0,
        "load_balance_loss": 0.0,
        "p_fuse": 0.0,
        "hard_fuse": 0.0,
        "target_fuse": 0.0,
        "action_correct": 0.0,
        "count": 0,
    }
    for idx in indices[:max_samples]:
        batch = collator([dataset[idx]])
        batch = move_batch_to_device(batch, device)
        try:
            result = compute_batch_action_ce_losses(
                batch=batch,
                base_model=base_model,
                source_model=source_model,
                projector_list=projector_list,
                bank_config=bank_config,
                router=router,
                label_metric=label_metric,
                option_token_ids=option_token_ids,
                oracle_ce_margin=oracle_ce_margin,
                action_class_weights=action_class_weights,
                load_balance_loss_weight=load_balance_loss_weight,
                load_balance_loss_type=load_balance_loss_type,
                no_skip_routing=no_skip_routing,
                bank_configs=bank_configs,
            )
        except Exception as exc:
            print(f"Skipping eval sample idx={idx}: {exc}")
            continue
        if result is None:
            continue
        totals["loss"] += float(result["loss"].item())
        totals["task_loss"] += float(result["task_loss"].item())
        totals["receiver_loss"] += float(result["receiver_loss"].item())
        totals["fusion_loss"] += float(result["fusion_loss"].item())
        totals["oracle_loss"] += float(result["oracle_loss"].item())
        totals["action_loss"] += float(result["action_loss"].item())
        totals["load_balance_loss"] += float(result["load_balance_loss"].item())
        totals["p_fuse"] += float(result["p_fuse"].mean().item())
        totals["hard_fuse"] += float(result["hard_fuse"].mean().item())
        totals["target_fuse"] += float(result["target_fuse"].mean().item())
        totals["action_correct"] += float(result["action_correct"].mean().item())
        totals["count"] += 1

    count = max(int(totals["count"]), 1)
    return {
        "loss": totals["loss"] / count,
        "task_loss": totals["task_loss"] / count,
        "receiver_loss": totals["receiver_loss"] / count,
        "fusion_loss": totals["fusion_loss"] / count,
        "oracle_loss": totals["oracle_loss"] / count,
        "action_loss": totals["action_loss"] / count,
        "load_balance_loss": totals["load_balance_loss"] / count,
        "p_fuse": totals["p_fuse"] / count,
        "hard_fuse_rate": totals["hard_fuse"] / count,
        "target_fuse_rate": totals["target_fuse"] / count,
        "action_acc": totals["action_correct"] / count,
        "num_eval": int(totals["count"]),
    }


@torch.no_grad()
def evaluate_delta_regression(
    *,
    dataset,
    collator,
    indices: Sequence[int],
    device: torch.device,
    base_model,
    source_model,
    projector_list: Sequence[torch.nn.Module],
    bank_config: Dict[str, Any],
    router: torch.nn.Module,
    label_metric: str,
    option_token_ids: Optional[Sequence[int]],
    delta_mean: float,
    delta_std: float,
    max_samples: int,
) -> Dict[str, float]:
    router.eval()
    rows = []
    total_loss = 0.0
    for idx in indices[:max_samples]:
        batch = collator([dataset[idx]])
        batch = move_batch_to_device(batch, device)
        try:
            result = compute_batch_delta_regression_losses(
                batch=batch,
                base_model=base_model,
                source_model=source_model,
                projector_list=projector_list,
                bank_config=bank_config,
                router=router,
                label_metric=label_metric,
                option_token_ids=option_token_ids,
                delta_mean=delta_mean,
                delta_std=delta_std,
            )
        except Exception as exc:
            print(f"Skipping eval sample idx={idx}: {exc}")
            continue
        if result is None:
            continue
        total_loss += float(result["loss"].item())
        rows.append(
            {
                "score": float(result["score"].view(-1)[0].item()),
                "delta": float(result["delta"].view(-1)[0].item()),
                "receiver_loss": float(result["receiver_loss"].item()),
                "fusion_loss": float(result["fusion_loss"].item()),
            }
        )

    if not rows:
        return {
            "loss": 0.0,
            "task_loss": 0.0,
            "p_fuse": 0.0,
            "hard_fuse_rate": 0.0,
            "target_fuse_rate": 0.0,
            "action_acc": 0.0,
            "num_eval": 0,
        }

    scores = torch.tensor([row["score"] for row in rows], dtype=torch.float32)
    deltas = torch.tensor([row["delta"] for row in rows], dtype=torch.float32)
    receiver_losses = torch.tensor([row["receiver_loss"] for row in rows], dtype=torch.float32)
    fusion_losses = torch.tensor([row["fusion_loss"] for row in rows], dtype=torch.float32)

    def routed_for_mask(mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, fusion_losses, receiver_losses)

    receiver_mean = float(receiver_losses.mean().item())
    fusion_mean = float(fusion_losses.mean().item())
    oracle_mean = float(torch.minimum(receiver_losses, fusion_losses).mean().item())
    target_fuse = deltas > 0
    pred_fuse_zero = scores > 0
    routed_zero = routed_for_mask(pred_fuse_zero)

    centered_scores = scores - scores.mean()
    centered_deltas = deltas - deltas.mean()
    denom = centered_scores.norm() * centered_deltas.norm()
    pearson = (
        0.0
        if float(denom.item()) <= 1e-12
        else float((centered_scores * centered_deltas).sum().div(denom).item())
    )

    threshold_candidates = torch.unique(scores).tolist()
    threshold_candidates.extend(
        [float(torch.quantile(scores, q).item()) for q in torch.linspace(0, 1, 21)]
    )
    best = None
    for threshold in sorted(set(float(x) for x in threshold_candidates)):
        mask = scores > threshold
        routed = routed_for_mask(mask)
        routed_mean = float(routed.mean().item())
        item = {
            "threshold": threshold,
            "routed_loss": routed_mean,
            "fuse_rate": float(mask.float().mean().item()),
            "gain_vs_receiver": receiver_mean - routed_mean,
            "gain_vs_all_fuse": fusion_mean - routed_mean,
        }
        if best is None or item["routed_loss"] < best["routed_loss"]:
            best = item

    top_metrics = {}
    for rho in (0.1, 0.2, 0.3, 0.5):
        threshold = float(torch.quantile(scores, 1.0 - rho).item())
        mask = scores > threshold
        routed = routed_for_mask(mask)
        routed_mean = float(routed.mean().item())
        key = f"top{int(rho * 100)}"
        top_metrics[f"{key}_threshold"] = threshold
        top_metrics[f"{key}_fuse_rate"] = float(mask.float().mean().item())
        top_metrics[f"{key}_routed_loss"] = routed_mean
        top_metrics[f"{key}_gain_vs_receiver"] = receiver_mean - routed_mean
        top_metrics[f"{key}_gain_vs_all_fuse"] = fusion_mean - routed_mean

    return {
        "loss": total_loss / len(rows),
        "task_loss": float(routed_zero.mean().item()),
        "receiver_loss": receiver_mean,
        "fusion_loss": fusion_mean,
        "oracle_loss": oracle_mean,
        "p_fuse": float(torch.sigmoid(scores).mean().item()),
        "hard_fuse_rate": float(pred_fuse_zero.float().mean().item()),
        "target_fuse_rate": float(target_fuse.float().mean().item()),
        "action_acc": float((pred_fuse_zero == target_fuse).float().mean().item()),
        "score_mean": float(scores.mean().item()),
        "score_std": float(scores.std(unbiased=False).item()),
        "delta_mean": float(deltas.mean().item()),
        "delta_std": float(deltas.std(unbiased=False).item()),
        "score_delta_pearson": pearson,
        "threshold_sweep_threshold": float(best["threshold"]),
        "threshold_sweep_fuse_rate": float(best["fuse_rate"]),
        "threshold_sweep_routed_loss": float(best["routed_loss"]),
        "threshold_sweep_gain_vs_receiver": float(best["gain_vs_receiver"]),
        "threshold_sweep_gain_vs_all_fuse": float(best["gain_vs_all_fuse"]),
        "num_eval": len(rows),
        **top_metrics,
    }


@torch.no_grad()
def sweep_hard_thresholds(
    *,
    dataset,
    collator,
    indices: Sequence[int],
    device: torch.device,
    base_model,
    source_model,
    projector_list: Sequence[torch.nn.Module],
    bank_config: Dict[str, Any],
    router: SimpleKVRouter,
    label_metric: str,
    option_token_ids: Optional[Sequence[int]],
    fusion_cost_weight: float,
    thresholds: Sequence[float],
    max_samples: int,
) -> Dict[str, Any]:
    rows = []
    for idx in indices[:max_samples]:
        batch = collator([dataset[idx]])
        batch = move_batch_to_device(batch, device)
        try:
            base_cache, source_cache, base_hidden, source_hidden, response_start, response_end = compute_prefill_state(
                batch,
                base_model=base_model,
                source_model=source_model,
            )
            all_fuse_cache = None
            if getattr(router, "feature_source", None) in {"shared_latent_fusion", "shared_latent_pair"}:
                all_fuse_cache = soft_fuse_cache(
                    base_cache=base_cache,
                    source_cache=source_cache,
                    projector_list=projector_list,
                    bank_config=bank_config,
                    bank_probability=torch.ones(
                        base_cache.key_cache[0].size(0),
                        device=base_cache.key_cache[0].device,
                    ),
                )
            feature = compute_router_feature(
                router,
                base_hidden,
                source_hidden,
                base_cache=base_cache,
                fused_cache=all_fuse_cache,
            )
            p_fuse = float(router.forward_features(feature).action_probabilities[:, 1].item())

            # Build the all-fusion cache before any response forward. Hugging Face
            # DynamicCache objects are mutable and can be extended by model.forward.
            if all_fuse_cache is None:
                all_fuse_cache = soft_fuse_cache(
                    base_cache=base_cache,
                    source_cache=source_cache,
                    projector_list=projector_list,
                    bank_config=bank_config,
                    bank_probability=torch.ones(1, device=device),
                )
            receiver_loss, valid = response_ce_from_cache(
                base_model=base_model,
                batch=batch,
                past_key_values=base_cache,
                response_start=response_start,
                response_end=response_end,
                label_metric=label_metric,
                option_token_ids=option_token_ids,
            )
            if valid <= 0:
                continue
            fusion_loss, _ = response_ce_from_cache(
                base_model=base_model,
                batch=batch,
                past_key_values=all_fuse_cache,
                response_start=response_start,
                response_end=response_end,
                label_metric=label_metric,
                option_token_ids=option_token_ids,
            )
            rows.append(
                {
                    "p_fuse": p_fuse,
                    "receiver_loss": float(receiver_loss.item()),
                    "fusion_loss": float(fusion_loss.item()),
                }
            )
        except Exception as exc:
            print(f"Skipping threshold sample idx={idx}: {exc}")

    if not rows:
        return {"num_eval": 0, "thresholds": []}

    threshold_results = []
    receiver_mean = sum(row["receiver_loss"] for row in rows) / len(rows)
    fusion_mean = sum(row["fusion_loss"] for row in rows) / len(rows)
    for threshold in thresholds:
        routed_losses = [
            row["fusion_loss"] if row["p_fuse"] > threshold else row["receiver_loss"]
            for row in rows
        ]
        fuse_rate = sum(1 for row in rows if row["p_fuse"] > threshold) / len(rows)
        routed_mean = sum(routed_losses) / len(routed_losses)
        routed_objectives = [
            row["fusion_loss"] + fusion_cost_weight
            if row["p_fuse"] > threshold
            else row["receiver_loss"]
            for row in rows
        ]
        routed_objective_mean = sum(routed_objectives) / len(routed_objectives)
        threshold_results.append(
            {
                "threshold": float(threshold),
                "routed_loss": routed_mean,
                "routed_objective": routed_objective_mean,
                "gain_vs_receiver": receiver_mean - routed_mean,
                "gain_vs_all_fuse": fusion_mean - routed_mean,
                "fuse_rate": fuse_rate,
            }
        )
    best = min(threshold_results, key=lambda item: item["routed_loss"])
    best_objective = min(threshold_results, key=lambda item: item["routed_objective"])
    return {
        "num_eval": len(rows),
        "receiver_loss": receiver_mean,
        "all_fuse_loss": fusion_mean,
        "best": best,
        "best_objective": best_objective,
        "thresholds": threshold_results,
    }


def save_projector_checkpoint(
    output_dir: Path,
    projector_list: Sequence[torch.nn.Module],
    bank_config: Any,
) -> None:
    proj_dir = output_dir / "projector"
    proj_dir.mkdir(parents=True, exist_ok=True)
    for idx, projector in enumerate(projector_list):
        torch.save(projector.state_dict(), proj_dir / f"projector_{idx}.pt")
        save_projector(projector, str(proj_dir / f"projector_{idx}.json"))
    if isinstance(bank_config, (list, tuple)):
        payload = {
            "format": "projector_banks_v1",
            "bank_configs": list(bank_config),
        }
    else:
        payload = bank_config
    with open(proj_dir / "projector_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--projector-bank-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument(
        "--data-pool-samples",
        type=int,
        default=None,
        help=(
            "Load this many dataset examples before randomly selecting --num-samples. "
            "Use this to avoid training only on the dataset prefix."
        ),
    )
    parser.add_argument(
        "--mix-data-config",
        action="append",
        default=[],
        help=(
            "Additional config path whose `data` section is mixed with the primary "
            "--config data. --num-samples is split evenly across all sources."
        ),
    )
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--threshold-sweep-samples", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override training.max_length from the main config.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fusion-cost-weight", type=float, default=0.05)
    parser.add_argument(
        "--no-skip-routing",
        action="store_true",
        help="Route directly among projector banks. Action 0 means bank0, not skip.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=0,
        help=(
            "If >0 and a single projector bank is loaded, clone it into this many "
            "independent trainable expert banks."
        ),
    )
    parser.add_argument(
        "--training-objective",
        choices=[
            "task_ce_softgate",
            "task_ce_straight_through",
            "action_ce_oracle",
            "delta_regression",
        ],
        default="task_ce_softgate",
        help=(
            "`task_ce_softgate` optimizes response/option CE through a soft "
            "receiver-fusion interpolation. `task_ce_straight_through` uses a hard "
            "top-1 skip/fuse route in the forward pass and soft router gradients "
            "in the backward pass. `action_ce_oracle` computes receiver "
            "and all-fusion CE, turns the better side into a skip/fuse label, "
            "and trains encoder+router with action cross entropy only. "
            "`delta_regression` trains binary_logits to predict normalized "
            "receiver_ce - fusion_ce with SmoothL1."
        ),
    )
    parser.add_argument(
        "--oracle-ce-margin",
        type=float,
        default=0.0,
        help="Fuse label requires receiver_ce - fusion_ce > margin in action_ce_oracle mode.",
    )
    parser.add_argument(
        "--oracle-ce-margin-quantile",
        type=float,
        default=None,
        help=(
            "If set, compute the fuse margin from the train delta distribution. "
            "Example: 0.7 means only the top 30%% delta samples become fuse labels."
        ),
    )
    parser.add_argument(
        "--delta-hist-bins",
        type=int,
        default=21,
        help="Number of bins printed for delta diagnostics during action_ce_oracle scan.",
    )
    parser.add_argument(
        "--action-class-weight-mode",
        choices=["none", "inverse"],
        default="none",
        help="Class weighting for action_ce_oracle CE loss.",
    )
    parser.add_argument(
        "--action-class-weight-scan-samples",
        type=int,
        default=0,
        help="Number of train examples to scan for class weights. 0 scans all train examples.",
    )
    parser.add_argument(
        "--action-class-max-weight",
        type=float,
        default=10.0,
        help="Clip action class weights before normalization. Use <=0 to disable clipping.",
    )
    parser.add_argument(
        "--action-class-skip-multiplier",
        type=float,
        default=1.0,
        help="Extra multiplier on skip class weight before normalization.",
    )
    parser.add_argument(
        "--label-metric",
        choices=["response_ce", "option_token_ce"],
        default="response_ce",
    )
    parser.add_argument("--num-options", type=int, default=4)
    parser.add_argument(
        "--router-feature-source",
        choices=[
            "hidden_binned",
            "shared_latent_receiver",
            "shared_latent_fusion",
            "shared_latent_pair",
        ],
        default="hidden_binned",
    )
    parser.add_argument("--router-input-dim", type=int, default=896)
    parser.add_argument("--router-hidden-dim", type=int, default=96)
    parser.add_argument("--router-layers", type=int, default=2)
    parser.add_argument("--router-dropout", type=float, default=0.3)
    parser.add_argument(
        "--router-init-scale",
        type=float,
        default=1.0,
        help="Scale Linear weights/biases after router initialization. Switch-style experiments use 0.1.",
    )
    parser.add_argument(
        "--load-balance-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary router load-balance regularization weight.",
    )
    parser.add_argument(
        "--load-balance-loss-type",
        choices=["mse", "switch"],
        default="mse",
        help="`mse` balances mean routing probabilities; `switch` uses Switch-style importance*load.",
    )
    parser.add_argument("--latent-num-kv-layers", type=int, default=0)
    parser.add_argument("--latent-kv-dim", type=int, default=0)
    parser.add_argument("--latent-encoder-dim", type=int, default=1024)
    parser.add_argument("--latent-shared-dim", type=int, default=1024)
    parser.add_argument("--latent-num-heads", type=int, default=16)
    parser.add_argument("--latent-ffn-mult", type=float, default=4.0)
    parser.add_argument("--latent-dropout", type=float, default=0.0)
    parser.add_argument("--latent-pooling", choices=["last", "mean", "last_mean"], default="last")
    parser.add_argument(
        "--latent-max-encoder-tokens",
        type=int,
        default=0,
        help=(
            "Optional cap on fused KV tokens consumed by the shared-latent encoder. "
            "0 keeps the full prefill slice."
        ),
    )
    parser.add_argument("--input-standardization", action="store_true")
    parser.add_argument("--feature-stat-samples", type=int, default=512)
    parser.add_argument(
        "--projector-train-mode",
        choices=["frozen", "partial", "full"],
        default="frozen",
    )
    parser.add_argument(
        "--projector-trainable-name-substring",
        action="append",
        default=["key_proj_out", "value_proj_out", "key_scalar_head", "value_scalar_head"],
    )
    parser.add_argument(
        "--expert-dropout",
        type=float,
        default=-1.0,
        help=(
            "If >=0, overwrite Dropout.p inside projector experts. "
            "Switch-style fine-tuning commonly uses large expert dropout."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--projector-devices",
        default=None,
        help=(
            "Comma-separated devices for projector modules, e.g. cuda:0,cuda:1. "
            "This shards expert projector parameters and Adam states across GPUs; "
            "base/source/router remain on --device."
        ),
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--router-dtype",
        default=None,
        help="Optional dtype for the trainable router only. Defaults to --dtype.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="C2C")
    parser.add_argument("--wandb-entity", default="june6-hanyang-university")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    if args.train_ratio <= 0.0 or args.train_ratio >= 1.0:
        raise ValueError("--train-ratio must be in (0, 1).")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if args.train_batch_size <= 0:
        raise ValueError("--train-batch-size must be positive.")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.num_experts < 0:
        raise ValueError("--num-experts must be >= 0")
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError("--max-length must be positive when set.")
    if args.data_pool_samples is not None and args.data_pool_samples <= 0:
        raise ValueError("--data-pool-samples must be positive when set.")
    if args.train_batch_size != 1 and args.training_objective not in {
        "task_ce_softgate",
        "task_ce_straight_through",
    }:
        raise ValueError(
            "--train-batch-size > 1 is currently supported for task_ce_softgate "
            "and task_ce_straight_through only."
        )
    if args.action_class_weight_scan_samples < 0:
        raise ValueError("--action-class-weight-scan-samples must be >= 0")
    if args.action_class_skip_multiplier <= 0:
        raise ValueError("--action-class-skip-multiplier must be > 0")
    if args.oracle_ce_margin_quantile is not None and not (
        0.0 <= args.oracle_ce_margin_quantile <= 1.0
    ):
        raise ValueError("--oracle-ce-margin-quantile must be in [0, 1]")
    if args.delta_hist_bins <= 0:
        raise ValueError("--delta-hist-bins must be positive")
    if args.router_init_scale <= 0:
        raise ValueError("--router-init-scale must be positive")
    if args.load_balance_loss_weight < 0:
        raise ValueError("--load-balance-loss-weight must be >= 0")
    if args.no_skip_routing and args.training_objective not in {
        "task_ce_softgate",
        "task_ce_straight_through",
        "action_ce_oracle",
    }:
        raise ValueError("--no-skip-routing currently supports task CE and action CE objectives only.")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    model_config = cfg["model"]
    data_config = dict(cfg["data"])

    device = resolve_device(args.device)
    projector_devices = (
        parse_device_list(args.projector_devices)
        if args.projector_devices
        else [device]
    )
    dtype = parse_dtype(args.dtype)
    router_dtype = parse_dtype(args.router_dtype) if args.router_dtype else dtype
    base_model, source_model, base_tokenizer, source_tokenizer = load_models_and_tokenizers(
        model_config,
        device,
        dtype,
    )
    freeze_module(base_model)
    freeze_module(source_model)

    message_dataset = build_training_message_dataset(
        primary_data_config=data_config,
        mix_config_paths=args.mix_data_config,
        num_samples=args.num_samples,
        data_pool_samples=args.data_pool_samples,
        seed=args.seed,
    )
    supervised_dataset, collator = build_supervised_dataset(
        message_dataset,
        base_tokenizer,
        source_tokenizer,
        model_config,
        max_length=int(args.max_length or cfg["training"].get("max_length", 1024)),
    )
    option_token_ids = (
        get_option_token_ids(base_tokenizer, args.num_options)
        if args.label_metric == "option_token_ce"
        else None
    )

    projector_list, bank_dicts = load_projector_banks(args.projector_bank_dir, device)
    projector_list, bank_dicts = clone_single_bank_as_experts(
        projector_list,
        bank_dicts,
        num_experts=args.num_experts,
    )
    if args.no_skip_routing:
        if len(bank_dicts) < 1:
            raise ValueError("--no-skip-routing expects at least one projector bank.")
    elif len(bank_dicts) != 1:
        raise ValueError("Soft-gate E2E currently expects exactly one projector bank.")
    bank_config = bank_dicts[0]
    for projector in projector_list:
        set_module_dropout(projector, args.expert_dropout)
    configure_projector_training(
        projector_list,
        mode=args.projector_train_mode,
        trainable_name_substrings=args.projector_trainable_name_substring,
    )
    distribute_projectors(projector_list, projector_devices)

    num_examples = min(args.num_samples, len(supervised_dataset))
    all_indices = list(range(len(supervised_dataset)))
    random.Random(args.seed).shuffle(all_indices)
    selected_indices = all_indices[:num_examples]
    train_count = max(1, int(num_examples * args.train_ratio))
    train_indices = selected_indices[:train_count]
    val_indices = selected_indices[train_count:] or selected_indices[: min(args.eval_samples, len(selected_indices))]
    print(
        "Dataset sampling: "
        f"loaded_pool={len(supervised_dataset)} selected={num_examples} "
        f"train={len(train_indices)} val={len(val_indices)} seed={args.seed}"
    )

    if args.router_feature_source in {
        "shared_latent_receiver",
        "shared_latent_fusion",
        "shared_latent_pair",
    } and args.input_standardization:
        print(
            "Warning: disabling --input-standardization for shared-latent features "
            "because the encoder is trainable and its feature distribution changes."
        )
        args.input_standardization = False

    if args.router_feature_source in {
        "shared_latent_receiver",
        "shared_latent_fusion",
        "shared_latent_pair",
    }:
        inferred_layers, inferred_kv_dim = infer_base_kv_shape(base_model)
        latent_layers = args.latent_num_kv_layers or inferred_layers
        latent_kv_dim = args.latent_kv_dim or inferred_kv_dim
        router_num_banks = len(bank_dicts) if args.no_skip_routing else 1
        router = SharedLatentFusionRouter(
            num_banks=router_num_banks,
            num_kv_layers=latent_layers,
            kv_dim=latent_kv_dim,
            encoder_dim=args.latent_encoder_dim,
            shared_dim=args.latent_shared_dim,
            num_encoder_heads=args.latent_num_heads,
            encoder_ffn_mult=args.latent_ffn_mult,
            encoder_dropout=args.latent_dropout,
            max_encoder_tokens=(
                args.latent_max_encoder_tokens
                if args.latent_max_encoder_tokens > 0
                else None
            ),
            pooling=args.latent_pooling,
            router_hidden_dim=args.router_hidden_dim,
            router_layers=args.router_layers,
            router_dropout=args.router_dropout,
            feature_source=args.router_feature_source,
            no_skip_routing=args.no_skip_routing,
            dtype=router_dtype,
        ).to(device)
        print(
            "Shared-latent fusion router: "
            f"layers={latent_layers} kv_dim={latent_kv_dim} "
            f"encoder_dim={args.latent_encoder_dim} shared_dim={args.latent_shared_dim} "
            f"heads={args.latent_num_heads} input_dim={router.input_dim}"
        )
    else:
        router_num_banks = len(bank_dicts) if args.no_skip_routing else 1
        router = SimpleKVRouter(
            num_banks=router_num_banks,
            input_dim=args.router_input_dim,
            hidden_dim=args.router_hidden_dim,
            token_mlp_layers=args.router_layers,
            dropout=args.router_dropout,
            input_standardization=args.input_standardization,
            feature_source=args.router_feature_source,
            no_skip_routing=args.no_skip_routing,
            dtype=router_dtype,
        ).to(device)

    scale_linear_initialization(router, args.router_init_scale)
    if args.router_init_scale != 1.0:
        print(f"Scaled router Linear initialization by {args.router_init_scale:g}")

    action_class_weights = None
    action_label_scan = None
    if args.no_skip_routing and args.training_objective == "action_ce_oracle":
        if args.action_class_weight_mode != "none":
            raise ValueError(
                "No-skip action CE does not yet support action class weight scanning. "
                "Use --action-class-weight-mode none."
            )
        print("Skipping receiver/fusion delta scan for no-skip bank action CE.")
    elif args.training_objective in {"action_ce_oracle", "delta_regression"}:
        print(
            "Scanning receiver/fusion CE deltas for oracle action diagnostics..."
        )
        action_label_scan = estimate_oracle_action_counts(
            dataset=supervised_dataset,
            collator=collator,
            indices=train_indices,
            device=device,
            base_model=base_model,
            source_model=source_model,
            projector_list=projector_list,
            bank_config=bank_config,
            label_metric=args.label_metric,
            option_token_ids=option_token_ids,
            oracle_ce_margin=args.oracle_ce_margin,
            oracle_ce_margin_quantile=args.oracle_ce_margin_quantile,
            max_samples=args.action_class_weight_scan_samples,
            hist_bins=args.delta_hist_bins,
        )
        args.oracle_ce_margin = float(action_label_scan["effective_margin"])
        if (
            args.training_objective == "action_ce_oracle"
            and args.action_class_weight_mode != "none"
        ):
            action_class_weights = build_action_class_weights(
                counts=action_label_scan["counts"],
                mode=args.action_class_weight_mode,
                max_weight=args.action_class_max_weight,
                skip_multiplier=args.action_class_skip_multiplier,
            )
        print(
            "Action class counts/weights "
            f"counts={action_label_scan['counts'].tolist()} "
            f"weights={None if action_class_weights is None else [float(x) for x in action_class_weights.tolist()]} "
            f"receiver_loss_mean={action_label_scan['receiver_loss_mean']:.4f} "
            f"fusion_loss_mean={action_label_scan['fusion_loss_mean']:.4f} "
            f"effective_margin={args.oracle_ce_margin:.6f}"
        )

    if args.input_standardization:
        mean, std = estimate_feature_stats(
            dataset=supervised_dataset,
            collator=collator,
            indices=train_indices,
            device=device,
            base_model=base_model,
            source_model=source_model,
            router=router,
            max_samples=min(args.feature_stat_samples, len(train_indices)),
        )
        router.set_input_standardization(mean, std)
        print(
            "Input standardization enabled: "
            f"mean_abs={mean.abs().mean().item():.6g} std_mean={std.mean().item():.6g}"
        )

    trainable_params = list(router.parameters())
    for projector in projector_list:
        trainable_params.extend([p for p in projector.parameters() if p.requires_grad])

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    train_batches_per_epoch = math.ceil(len(train_indices) / args.train_batch_size)
    updates_per_epoch = math.ceil(train_batches_per_epoch / args.gradient_accumulation_steps)
    total_steps = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    wandb_module = None
    if args.wandb:
        import wandb

        wandb_module = wandb
        action_label_scan_config = None
        if action_label_scan is not None:
            action_label_scan_config = {
                "counts": action_label_scan["counts"].tolist(),
                "raw_counts": action_label_scan["raw_counts"].tolist(),
                "num_valid": int(action_label_scan["num_valid"]),
                "receiver_loss_mean": float(action_label_scan["receiver_loss_mean"]),
                "fusion_loss_mean": float(action_label_scan["fusion_loss_mean"]),
                "delta_mean": float(action_label_scan["delta_mean"]),
                "delta_std": float(action_label_scan["delta_std"]),
                "delta_min": float(action_label_scan["delta_min"]),
                "delta_max": float(action_label_scan["delta_max"]),
                "fuse_better_ratio": float(action_label_scan["fuse_better_ratio"]),
                "strong_fuse_ratio": float(action_label_scan["strong_fuse_ratio"]),
                "skip_better_ratio": float(action_label_scan["skip_better_ratio"]),
                "near_zero_ratio": float(action_label_scan["near_zero_ratio"]),
                "effective_margin": float(action_label_scan["effective_margin"]),
                "oracle_ce_margin_quantile": action_label_scan["oracle_ce_margin_quantile"],
                "quantiles": action_label_scan["quantiles"],
                "hist_counts": action_label_scan["hist_counts"],
                "hist_edges": action_label_scan["hist_edges"],
            }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=args.wandb_run_name or output_dir.name,
            dir=str(output_dir),
            config={
                **vars(args),
                "num_train": len(train_indices),
                "num_val": len(val_indices),
                "projector_train_mode": args.projector_train_mode,
                "action_class_weights": (
                    None
                    if action_class_weights is None
                    else [float(x) for x in action_class_weights.tolist()]
                ),
                "action_label_scan": action_label_scan_config,
            },
        )

    shutil.copy2(args.config, output_dir / "config.json")
    with open(output_dir / "e2e_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(
        f"E2E soft-gate training: train={len(train_indices)} val={len(val_indices)} "
        f"batch_size={args.train_batch_size} steps={total_steps} "
        f"projector_mode={args.projector_train_mode}"
    )

    best_val = float("inf")
    best_state = None
    best_projector_state = None
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        router.train()
        random.Random(args.seed + epoch).shuffle(train_indices)
        running = {
            "loss": 0.0,
            "task": 0.0,
            "cost": 0.0,
            "action": 0.0,
            "load_balance": 0.0,
            "p": 0.0,
            "p_soft": 0.0,
            "expert_entropy": 0.0,
            "selected_expert": 0.0,
            "n": 0,
        }

        epoch_batches = [
            train_indices[start : start + args.train_batch_size]
            for start in range(0, len(train_indices), args.train_batch_size)
        ]
        for local_step, batch_indices in enumerate(epoch_batches, start=1):
            batch = collator([supervised_dataset[idx] for idx in batch_indices])
            batch = move_batch_to_device(batch, device)
            try:
                if args.training_objective == "action_ce_oracle":
                    result = compute_batch_action_ce_losses(
                        batch=batch,
                        base_model=base_model,
                        source_model=source_model,
                        projector_list=projector_list,
                        bank_config=bank_config,
                        router=router,
                        label_metric=args.label_metric,
                        option_token_ids=option_token_ids,
                        oracle_ce_margin=args.oracle_ce_margin,
                        action_class_weights=action_class_weights,
                        load_balance_loss_weight=args.load_balance_loss_weight,
                        load_balance_loss_type=args.load_balance_loss_type,
                        no_skip_routing=args.no_skip_routing,
                        bank_configs=bank_dicts,
                    )
                elif args.training_objective == "delta_regression":
                    if action_label_scan is None:
                        raise RuntimeError("delta_regression requires delta scan statistics.")
                    result = compute_batch_delta_regression_losses(
                        batch=batch,
                        base_model=base_model,
                        source_model=source_model,
                        projector_list=projector_list,
                        bank_config=bank_config,
                        router=router,
                        label_metric=args.label_metric,
                        option_token_ids=option_token_ids,
                        delta_mean=float(action_label_scan["delta_mean"]),
                        delta_std=float(action_label_scan["delta_std"]),
                    )
                else:
                    result = compute_batch_losses(
                        batch=batch,
                        base_model=base_model,
                        source_model=source_model,
                        projector_list=projector_list,
                        bank_config=bank_config,
                        router=router,
                        fusion_cost_weight=args.fusion_cost_weight,
                        label_metric=args.label_metric,
                        option_token_ids=option_token_ids,
                        load_balance_loss_weight=args.load_balance_loss_weight,
                        load_balance_loss_type=args.load_balance_loss_type,
                        straight_through_routing=args.training_objective
                        == "task_ce_straight_through",
                        no_skip_routing=args.no_skip_routing,
                        bank_configs=bank_dicts,
                    )
            except Exception as exc:
                print(f"Skipping train batch idx={batch_indices}: {exc}")
                continue
            if result is None:
                continue

            loss = result["loss"] / args.gradient_accumulation_steps
            loss.backward()

            running["loss"] += float(result["loss"].detach().item())
            running["task"] += float(result["task_loss"].item())
            running["cost"] += float(result.get("cost_loss", torch.zeros_like(result["loss"])).item())
            running["action"] += float(result.get("action_loss", result["loss"]).item())
            running["load_balance"] += float(
                result.get("load_balance_loss", torch.zeros_like(result["loss"])).item()
            )
            running["p"] += float(result["p_fuse"].mean().item())
            running["p_soft"] += float(result.get("p_fuse_soft", result["p_fuse"]).mean().item())
            running["expert_entropy"] += float(
                result.get("expert_entropy", torch.zeros_like(result["loss"])).item()
            )
            running["selected_expert"] += float(
                result.get("selected_expert", torch.zeros_like(result["loss"])).item()
            )
            running.setdefault("target_fuse", 0.0)
            running.setdefault("hard_fuse", 0.0)
            running.setdefault("action_correct", 0.0)
            running["target_fuse"] += float(result.get("target_fuse", result["p_fuse"] * 0).mean().item())
            running["hard_fuse"] += float(result.get("hard_fuse", (result["p_fuse"] > 0.5).float()).mean().item())
            running["action_correct"] += float(result.get("action_correct", result["p_fuse"] * 0).mean().item())
            running["n"] += 1

            if local_step % args.gradient_accumulation_steps == 0 or local_step == len(epoch_batches):
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step == 1 or global_step % args.log_every == 0:
                    denom = max(running["n"], 1)
                    payload = {
                        "trainer/global_step": global_step,
                        "train/loss": running["loss"] / denom,
                        "train/task_loss": running["task"] / denom,
                        "train/fusion_cost": running["cost"] / denom,
                        "train/action_loss": running["action"] / denom,
                        "train/load_balance_loss": running["load_balance"] / denom,
                        "train/p_fuse": running["p"] / denom,
                        "train/p_fuse_soft": running["p_soft"] / denom,
                        "train/expert_entropy": running["expert_entropy"] / denom,
                        "train/selected_expert_mean": running["selected_expert"] / denom,
                        "train/target_fuse_rate": running["target_fuse"] / denom,
                        "train/hard_fuse_rate": running["hard_fuse"] / denom,
                        "train/action_acc": running["action_correct"] / denom,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch + min(
                            local_step * args.train_batch_size,
                            len(train_indices),
                        ) / max(len(train_indices), 1),
                    }
                    print(
                        f"step={global_step} epoch={epoch + 1} "
                        f"loss={payload['train/loss']:.4f} "
                        f"action={payload['train/action_loss']:.4f} "
                        f"lb={payload['train/load_balance_loss']:.4f} "
                        f"task={payload['train/task_loss']:.4f} "
                        f"p_fuse={payload['train/p_fuse']:.4f} "
                        f"p_soft={payload['train/p_fuse_soft']:.4f} "
                        f"expert_H={payload['train/expert_entropy']:.4f} "
                        f"target_fuse={payload['train/target_fuse_rate']:.4f} "
                        f"acc={payload['train/action_acc']:.4f}"
                    )
                    if wandb_module is not None:
                        wandb.log(payload, step=global_step)
                    running = {
                        "loss": 0.0,
                        "task": 0.0,
                        "cost": 0.0,
                        "action": 0.0,
                        "load_balance": 0.0,
                        "p": 0.0,
                        "p_soft": 0.0,
                        "expert_entropy": 0.0,
                        "selected_expert": 0.0,
                        "target_fuse": 0.0,
                        "hard_fuse": 0.0,
                        "action_correct": 0.0,
                        "n": 0,
                    }

                if global_step % args.eval_every == 0 or global_step == total_steps:
                    if args.training_objective == "action_ce_oracle":
                        val_metrics = evaluate_action_ce(
                            dataset=supervised_dataset,
                            collator=collator,
                            indices=val_indices,
                            device=device,
                            base_model=base_model,
                            source_model=source_model,
                            projector_list=projector_list,
                            bank_config=bank_config,
                            router=router,
                            label_metric=args.label_metric,
                            option_token_ids=option_token_ids,
                            oracle_ce_margin=args.oracle_ce_margin,
                            action_class_weights=action_class_weights,
                            load_balance_loss_weight=args.load_balance_loss_weight,
                            load_balance_loss_type=args.load_balance_loss_type,
                            no_skip_routing=args.no_skip_routing,
                            bank_configs=bank_dicts,
                            max_samples=args.eval_samples,
                        )
                    elif args.training_objective == "delta_regression":
                        if action_label_scan is None:
                            raise RuntimeError("delta_regression requires delta scan statistics.")
                        val_metrics = evaluate_delta_regression(
                            dataset=supervised_dataset,
                            collator=collator,
                            indices=val_indices,
                            device=device,
                            base_model=base_model,
                            source_model=source_model,
                            projector_list=projector_list,
                            bank_config=bank_config,
                            router=router,
                            label_metric=args.label_metric,
                            option_token_ids=option_token_ids,
                            delta_mean=float(action_label_scan["delta_mean"]),
                            delta_std=float(action_label_scan["delta_std"]),
                            max_samples=args.eval_samples,
                        )
                    else:
                        val_metrics = evaluate_soft(
                            dataset=supervised_dataset,
                            collator=collator,
                            indices=val_indices,
                            device=device,
                            base_model=base_model,
                            source_model=source_model,
                            projector_list=projector_list,
                            bank_config=bank_config,
                            router=router,
                            fusion_cost_weight=args.fusion_cost_weight,
                            label_metric=args.label_metric,
                            option_token_ids=option_token_ids,
                            load_balance_loss_weight=0.0,
                            load_balance_loss_type=args.load_balance_loss_type,
                            straight_through_routing=args.training_objective
                            == "task_ce_straight_through",
                            no_skip_routing=args.no_skip_routing,
                            bank_configs=bank_dicts,
                            max_samples=args.eval_samples,
                        )
                    print(
                        f"eval step={global_step} "
                        f"val_loss={val_metrics['loss']:.4f} "
                        f"val_task={val_metrics['task_loss']:.4f} "
                        f"val_p_fuse={val_metrics['p_fuse']:.4f} "
                        f"hard_fuse={val_metrics['hard_fuse_rate']:.4f} "
                        f"target_fuse={val_metrics.get('target_fuse_rate', 0.0):.4f} "
                        f"acc={val_metrics.get('action_acc', 0.0):.4f} "
                        f"corr={val_metrics.get('score_delta_pearson', 0.0):.4f} "
                        f"best_gain={val_metrics.get('threshold_sweep_gain_vs_receiver', 0.0):.4f}"
                    )
                    if wandb_module is not None:
                        val_payload = {
                            "val/loss": val_metrics["loss"],
                            "val/task_loss": val_metrics["task_loss"],
                            "val/p_fuse": val_metrics["p_fuse"],
                            "val/p_fuse_soft": val_metrics.get("p_fuse_soft", val_metrics["p_fuse"]),
                            "val/hard_fuse_rate": val_metrics["hard_fuse_rate"],
                            "val/num_eval": val_metrics["num_eval"],
                        }
                        if "cost_loss" in val_metrics:
                            val_payload["val/fusion_cost"] = val_metrics["cost_loss"]
                        for key in (
                            "action_loss",
                            "load_balance_loss",
                            "receiver_loss",
                            "fusion_loss",
                            "oracle_loss",
                            "target_fuse_rate",
                            "action_acc",
                            "score_mean",
                            "score_std",
                            "delta_mean",
                            "delta_std",
                            "score_delta_pearson",
                            "threshold_sweep_threshold",
                            "threshold_sweep_fuse_rate",
                            "threshold_sweep_routed_loss",
                            "threshold_sweep_gain_vs_receiver",
                            "threshold_sweep_gain_vs_all_fuse",
                            "top10_routed_loss",
                            "top10_gain_vs_receiver",
                            "top20_routed_loss",
                            "top20_gain_vs_receiver",
                            "top30_routed_loss",
                            "top30_gain_vs_receiver",
                            "top50_routed_loss",
                            "top50_gain_vs_receiver",
                        ):
                            if key in val_metrics:
                                val_payload[f"val/{key}"] = val_metrics[key]
                        wandb.log(val_payload, step=global_step)
                    if val_metrics["loss"] < best_val:
                        best_val = val_metrics["loss"]
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in router.state_dict().items()
                        }
                        if args.projector_train_mode != "frozen":
                            best_projector_state = [
                                {
                                    key: value.detach().cpu().clone()
                                    for key, value in projector.state_dict().items()
                                }
                                for projector in projector_list
                            ]
        torch.cuda.empty_cache()

    if best_state is not None:
        router.load_state_dict(best_state)
    if best_projector_state is not None:
        for projector, state in zip(projector_list, best_projector_state):
            projector.load_state_dict(state)
    router_dir = output_dir / "router"
    router_dir.mkdir(parents=True, exist_ok=True)
    save_router(router.cpu(), str(router_dir / "router.json"))
    torch.save(router.state_dict(), router_dir / "router.pt")

    if args.projector_train_mode != "frozen":
        save_projector_checkpoint(
            output_dir,
            projector_list,
            bank_dicts if args.no_skip_routing else bank_config,
        )

    if args.no_skip_routing:
        sweep = {
            "mode": "no_skip_bank_routing",
            "note": "Threshold sweep is not applicable: argmax selects bank0 or bank1.",
        }
    else:
        thresholds = [round(x / 20.0, 2) for x in range(1, 20)]
        sweep = sweep_hard_thresholds(
            dataset=supervised_dataset,
            collator=collator,
            indices=val_indices,
            device=device,
            base_model=base_model,
            source_model=source_model,
            projector_list=projector_list,
            bank_config=bank_config,
            router=router.to(device),
            label_metric=args.label_metric,
            option_token_ids=option_token_ids,
            fusion_cost_weight=args.fusion_cost_weight,
            thresholds=thresholds,
            max_samples=args.threshold_sweep_samples,
        )
    with open(output_dir / "threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)
    print(f"Saved router to {router_dir}")
    print(f"Threshold sweep: {json.dumps(sweep.get('best', {}), indent=2)}")

    if wandb_module is not None:
        wandb_module.run.summary["best_val_loss"] = best_val
        if sweep.get("best"):
            wandb_module.run.summary["best_threshold"] = sweep["best"]["threshold"]
            wandb_module.run.summary["best_threshold_routed_loss"] = sweep["best"]["routed_loss"]
        if sweep.get("best_objective"):
            wandb_module.run.summary["best_objective_threshold"] = sweep["best_objective"]["threshold"]
            wandb_module.run.summary["best_threshold_routed_objective"] = sweep["best_objective"]["routed_objective"]
        wandb_module.finish()


if __name__ == "__main__":
    main()
