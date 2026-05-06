"""
Standalone entropy-gated Rosetta wrapper.

This file is intentionally independent from wrapper.py so it can replace that
module directly without importing its implementation.
"""

import json
import math
from typing import List, Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from rosetta.model.projector import Projector
from rosetta.model.sampling import sample_token

try:
    from transformers.generation.utils import (
        GreedySearchDecoderOnlyOutput,
        SampleDecoderOnlyOutput,
    )
except Exception:
    GreedySearchDecoderOnlyOutput = None
    SampleDecoderOnlyOutput = None


def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
    new_cache = DynamicCache()
    for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
        new_cache.key_cache.append(k.clone().detach())
        new_cache.value_cache.append(v.clone().detach())
    return new_cache


def hybrid_to_dynamic(hybrid_cache):
    if hybrid_cache is None:
        return None
    if isinstance(hybrid_cache, DynamicCache):
        return hybrid_cache

    if hasattr(hybrid_cache, "key_cache") and hasattr(hybrid_cache, "value_cache"):
        keys = hybrid_cache.key_cache
        values = hybrid_cache.value_cache
        assert len(keys) == len(values), "key/value layers do not match"
        legacy_cache = [(k, v) for k, v in zip(keys, values)]
        return DynamicCache.from_legacy_cache(legacy_cache)

    raise TypeError(f"Unsupported cache type: {type(hybrid_cache)}")


class RosettaModel(nn.Module):
    """
    Drop in replacement for the standard transformers LLM models, like Qwen3ForCausalLM

    entropy-based source gating before cache-to-cache fusion using the rule:

        gate = [h_s < threshold_abs] & [h_s - h_r < threshold_rel]

    where h_s and h_r are normalized next-token entropies for the sharer/source
    and receiver/base model respectively.
    """

    def __init__(
        self,
        model_list: List[PreTrainedModel],
        base_model_idx=0,
        projector_list: List[Projector] = [],
        include_response: bool = False,
        multi_source_fusion_mode: str = "parallel",
        threshold_abs: float = 0.45,
        threshold_rel: float = 0.15,
        normalize_entropy: bool = True,
        entropy_gate_enabled: bool = True,
        entropy_eps: float = 1e-12,
    ):
        super().__init__()
        self.base_model_idx = base_model_idx
        self.model_list = nn.ModuleList(model_list)

        device = model_list[base_model_idx].device
        dtype = model_list[base_model_idx].dtype
        self.projector_list = nn.ModuleList(projector_list).to(device=device, dtype=dtype)

        self.projector_dict = {}
        self.kv_cache_dict = {}
        self._generation_hook_handlers = []

        self.include_response = include_response
        if multi_source_fusion_mode not in ["sequential", "parallel"]:
            raise ValueError(
                "multi_source_fusion_mode must be 'sequential' or 'parallel', "
                f"got '{multi_source_fusion_mode}'"
            )
        self.multi_source_fusion_mode = multi_source_fusion_mode

        self.threshold_abs = threshold_abs
        self.threshold_rel = threshold_rel
        self.normalize_entropy = normalize_entropy
        self.entropy_gate_enabled = entropy_gate_enabled
        self.entropy_eps = entropy_eps
        self.last_entropy_gate_state = {} #for debugging

    @property
    def device(self):
        return self.model_list[self.base_model_idx].device

    def to(self, device):
        super().to(device)
        for model in self.model_list:
            model.to(device)
        for projector in self.projector_list:
            projector.to(device)
        return self

    def _use_entropy_gate(self) -> bool:
        if not self.entropy_gate_enabled:
            return False
        return (not self.training) and (
            torch.is_inference_mode_enabled() or not torch.is_grad_enabled()
        )

    def _compute_entropy(self, logits) -> Optional[float]:
        if logits is None:
            return None
        next_token_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        probs = torch.softmax(next_token_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(self.entropy_eps))).sum(dim=-1)
        if self.normalize_entropy and next_token_logits.size(-1) > 1:
            entropy = entropy / math.log(next_token_logits.size(-1))
        return float(entropy.mean().detach().cpu())

    def _should_fuse_source(
        self,
        curr_base_H: Optional[float],
        curr_source_H: Optional[float],
    ) -> bool:
        if not self._use_entropy_gate():
            return True
        if curr_base_H is None or curr_source_H is None:
            return True
        abs_gate = curr_source_H < self.threshold_abs
        rel_gate = (curr_base_H - curr_source_H) < self.threshold_rel
        return bool(abs_gate and rel_gate)

    def _patch_base_attention_with_cache( #TODO : 그냥 하던대로 쓰던 코드 사용할 것.
        self,
        fused_kv_cache: DynamicCache,
        new_length: int,
    ):
        hook_handlers = []
        for i in range(self.model_list[self.base_model_idx].config.num_hidden_layers):
            attn = self.model_list[self.base_model_idx].model.layers[i].self_attn
            new_k = fused_kv_cache.key_cache[i][:, :, -new_length:, :]
            new_v = fused_kv_cache.value_cache[i][:, :, -new_length:, :]
            orig_forward = RosettaModel._monkeypatch_qwen3_attention_forward(
                attn, new_k, new_v
            )
            hook_handlers.append((attn, orig_forward))
        return hook_handlers

    def set_projector_config(
        self,
        source_model_idx: int,
        source_model_layer_idx: int,
        target_model_idx: int,
        target_model_layer_idx: int,
        projector_idx: int,
    ):
        if target_model_idx not in self.projector_dict:
            self.projector_dict[target_model_idx] = {}
        if source_model_idx not in self.projector_dict[target_model_idx]:
            self.projector_dict[target_model_idx][source_model_idx] = {}
        layer_entry = self.projector_dict[target_model_idx][source_model_idx].get(
            target_model_layer_idx
        )
        if layer_entry is None:
            self.projector_dict[target_model_idx][source_model_idx][
                target_model_layer_idx
            ] = [(source_model_layer_idx, projector_idx)]
        else:
            layer_entry.append((source_model_layer_idx, projector_idx))

    @staticmethod
    def _is_gate_open(proj) -> bool:
        fn = getattr(proj, "is_gate_open", None)
        return fn() if fn is not None else True

    def _get_active_source_models(self) -> set:
        active = set()
        if self.base_model_idx not in self.projector_dict:
            return active
        for src_idx, layer_map in self.projector_dict[self.base_model_idx].items():
            for _, entry in layer_map.items():
                for _, proj_idx in entry:
                    if self._is_gate_open(self.projector_list[proj_idx]):
                        active.add(src_idx)
                        break
                if src_idx in active:
                    break
        return active

    def load_projector(self, projector_list):
        self.projector_list: List[Projector] = projector_list

    def get_projector(
        self,
        source_model_idx,
        source_model_layer_idx,
        target_model_idx,
        target_model_layer_idx,
    ):
        pair_list = self.projector_dict[target_model_idx][source_model_idx][
            target_model_layer_idx
        ]
        if len(pair_list) == 0:
            raise ValueError("No projector configured for the given target layer")
        for src_layer, projector_id in pair_list:
            if src_layer == source_model_layer_idx:
                return self.projector_list[projector_id]
        return self.projector_list[pair_list[0][1]]

    @staticmethod
    def load_json(file_name):
        with open(file_name, "r") as f:
            result = json.load(f)
        return result

    @staticmethod
    def _convert_dict_keys_to_ints(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                if isinstance(key, str) and key.lstrip("-").isdigit():
                    new_key = int(key)
                else:
                    new_key = key
                new_obj[new_key] = RosettaModel._convert_dict_keys_to_ints(value)
            return new_obj
        if isinstance(obj, list):
            return [RosettaModel._convert_dict_keys_to_ints(v) for v in obj]
        return obj

    def save_projector_config(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.projector_dict, f)

    def load_projector_config(self, config_path):
        if config_path.endswith(".json"):
            loaded = RosettaModel.load_json(config_path)
            self.projector_dict = RosettaModel._convert_dict_keys_to_ints(loaded)

    def set_kv_cache_dict(self, source_model_idx, target_model_idx, cache):
        if target_model_idx not in self.kv_cache_dict:
            self.kv_cache_dict[target_model_idx] = {}
        if cache is None:
            self.kv_cache_dict[target_model_idx][source_model_idx] = DynamicCache()
        else:
            self.kv_cache_dict[target_model_idx][source_model_idx] = cache

    @staticmethod
    def _monkeypatch_qwen3_attention_forward(attn_module, new_k_cache, new_v_cache):
        import types

        from transformers.models.qwen3.modeling_qwen3 import (  # type: ignore
            ALL_ATTENTION_FUNCTIONS,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        orig_forward = attn_module.forward

        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
                1, 2
            )
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
                1, 2
            )
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if new_k_cache is not None and new_v_cache is not None:
                if key_states.shape == new_k_cache.shape:
                    key_states = new_k_cache
                if value_states.shape == new_v_cache.shape:
                    value_states = new_v_cache

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if (
                    self.config._attn_implementation == "sdpa"
                    and kwargs.get("output_attentions", False)
                ):
                    attention_interface = eager_attention_forward
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[
                        self.config._attn_implementation
                    ]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        attn_module.forward = types.MethodType(patched_forward, attn_module)
        return orig_forward

    def register_hooks(
        self,
        input_ids,
        attention_mask,
        position_ids,
        base_kv_cache,
        source_model_idx,
        source_kv_cache,
    ):
        has_open_gate = False
        if (
            self.base_model_idx in self.projector_dict
            and source_model_idx in self.projector_dict[self.base_model_idx]
        ):
            for _, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
                for _, proj_idx in entry:
                    if self._is_gate_open(self.projector_list[proj_idx]):
                        has_open_gate = True
                        break
                if has_open_gate:
                    break

        base_kv_copy = clone_kv_cache(base_kv_cache)
        base_output = self.model_list[self.base_model_idx].forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=base_kv_copy,
            labels=None,
            use_cache=True,
        )
        base_output_kv_cache = base_output.past_key_values
        curr_base_H = self._compute_entropy(base_output.logits)

        if not has_open_gate:
            fused_kv_cache = clone_kv_cache(base_output_kv_cache)
            hook_handlers = self._patch_base_attention_with_cache(
                fused_kv_cache, input_ids.shape[1]
            )
            return hook_handlers, base_output_kv_cache, source_kv_cache

        source_kv_copy = clone_kv_cache(source_kv_cache)
        new_length = input_ids.shape[1]
        source_output = self.model_list[source_model_idx].forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=source_kv_copy,
            labels=None,
            use_cache=True,
        )
        source_output_kv_cache = source_output.past_key_values
        curr_source_H = self._compute_entropy(source_output.logits)
        
        print(f"curr_base_H: {curr_base_H}, curr_source_H: {curr_source_H}")
        allow_entropy_fusion = self._should_fuse_source(curr_base_H, curr_source_H)

        self.last_entropy_gate_state = {
            "mode": "register_hooks",
            "base_entropy": curr_base_H,
            "sources": {
                source_model_idx: {
                    "source_entropy": curr_source_H,
                    "relative_entropy_gap": (
                        None
                        if curr_base_H is None or curr_source_H is None
                        else curr_base_H - curr_source_H
                    ),
                    "allow_fusion": allow_entropy_fusion,
                }
            },
        }

        if not allow_entropy_fusion:
            fused_kv_cache = clone_kv_cache(base_output_kv_cache)
            hook_handlers = self._patch_base_attention_with_cache(
                fused_kv_cache, new_length
            )
            return hook_handlers, base_output_kv_cache, source_output_kv_cache

        fused_kv_cache = clone_kv_cache(base_output_kv_cache)

        for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
            base_key_cache, base_value_cache = base_output_kv_cache[target_layer_idx]
            new_base_key_cache = base_key_cache[:, :, -new_length:, :]
            new_base_value_cache = base_value_cache[:, :, -new_length:, :]
            new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

            projected_kv_list = []
            for source_model_layer_idx, projector_idx in entry:
                if not self._is_gate_open(self.projector_list[projector_idx]):
                    continue
                source_key_cache, source_value_cache = source_output_kv_cache[
                    source_model_layer_idx
                ]
                new_source_key_cache = source_key_cache[:, :, -new_length:, :]
                new_source_value_cache = source_value_cache[:, :, -new_length:, :]
                new_source_kv_cache = (new_source_key_cache, new_source_value_cache)
                projected_key, projected_value = self.projector_list[projector_idx].forward(
                    new_source_kv_cache,
                    new_base_kv_cache,
                )
                projected_kv_list.append((projected_key, projected_value))

            if not projected_kv_list:
                continue

            agg_key, agg_value = projected_kv_list[0]
            fused_kv_cache.key_cache[target_layer_idx][:, :, -new_length:, :] = agg_key
            fused_kv_cache.value_cache[target_layer_idx][:, :, -new_length:, :] = agg_value

        hook_handlers = self._patch_base_attention_with_cache(fused_kv_cache, new_length)
        return hook_handlers, base_output_kv_cache, source_output_kv_cache

    def remove_hooks(self, hook_handlers):
        for attn, orig_forward in hook_handlers:
            attn.forward = orig_forward

    def forward(
        self,
        kv_cache_index: Optional[List] = None,
        input_ids: Optional[Union[torch.LongTensor, List[torch.LongTensor]]] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        *args,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if isinstance(input_ids, list):
            base_input_ids = input_ids[self.base_model_idx] if input_ids is not None else None
            base_attention_mask = (
                attention_mask[self.base_model_idx] if attention_mask is not None else None
            )
            _, seqlen = base_input_ids.size() if base_input_ids is not None else (0, 0)
        else:
            base_input_ids = input_ids
            base_attention_mask = attention_mask
            _, seqlen = input_ids.size() if input_ids is not None else (0, 0)

        if seqlen > 1:
            self.kv_cache_dict = dict()

        num_sections = len(kv_cache_index) if kv_cache_index is not None else 1
        section_lengths = (
            [kv_cache_index[i].shape[1] for i in range(num_sections)]
            if kv_cache_index is not None
            else [seqlen]
        )
        section_starts = [0]
        for length in section_lengths:
            section_starts.append(section_starts[-1] + length)

        curr_base_kv_cache = past_key_values

        for i in range(num_sections):
            start = section_starts[i]
            end = section_starts[i + 1]
            prefill_input_ids = base_input_ids[:, start:end] if base_input_ids is not None else None
            prefill_attention_mask = (
                base_attention_mask[:, :end] if base_attention_mask is not None else None
            )
            prefill_position_ids = (
                position_ids[:, start:end] if position_ids is not None else None
            )
            prefill_labels = labels[:, start:end] if labels is not None else None

            if i == num_sections - 1:
                if self.include_response:
                    hook_handlers, base_output_kv_cache, source_output_kv_cache = self.register_hooks(
                        input_ids=prefill_input_ids,
                        attention_mask=prefill_attention_mask,
                        position_ids=prefill_position_ids,
                        base_kv_cache=self.kv_cache_dict[self.base_model_idx][self.base_model_idx],
                        source_model_idx=1,
                        source_kv_cache=self.kv_cache_dict[self.base_model_idx][1],
                    )

                output = self.model_list[self.base_model_idx].forward(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask,
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    *args,
                    **kwargs,
                )

                if self.include_response:
                    self.remove_hooks(hook_handlers)
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(
                        base_output_kv_cache
                    )
                    self.kv_cache_dict[self.base_model_idx][1] = clone_kv_cache(
                        source_output_kv_cache
                    )
            else:
                output = self.model_list[self.base_model_idx].forward(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask,
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    *args,
                    **kwargs,
                )

                if self.base_model_idx not in self.kv_cache_dict:
                    self.kv_cache_dict[self.base_model_idx] = {}
                if self.base_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = None
                self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(
                    output.past_key_values
                )

                curr_base_kv_cache: DynamicCache = output.past_key_values
                curr_base_H = self._compute_entropy(output.logits)

                active_source_models = self._get_active_source_models()
                entropy_gate_sources = {}
                self.last_entropy_gate_state = {
                    "mode": "forward",
                    "section_idx": i,
                    "base_entropy": curr_base_H,
                    "sources": {},
                }

                for source_model_idx in range(1, len(self.model_list)):
                    if source_model_idx not in active_source_models:
                        continue
                    if self.base_model_idx not in self.kv_cache_dict:
                        self.kv_cache_dict[self.base_model_idx] = {}
                    if source_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                        self.kv_cache_dict[self.base_model_idx][source_model_idx] = None

                    if isinstance(input_ids, list):
                        source_input_ids = input_ids[source_model_idx]
                        source_attention_mask = (
                            attention_mask[source_model_idx] if attention_mask is not None else None
                        )
                        source_prefill_input_ids = (
                            source_input_ids[:, start:end] if source_input_ids is not None else None
                        )
                        source_prefill_attention_mask = (
                            source_attention_mask[:, :end]
                            if source_attention_mask is not None
                            else None
                        )
                    else:
                        source_prefill_input_ids = prefill_input_ids
                        source_prefill_attention_mask = prefill_attention_mask

                    model = self.model_list[source_model_idx]
                    was_training = model.training
                    had_gc = getattr(model, "is_gradient_checkpointing", False)

                    try:
                        if was_training:
                            model.eval()
                        if had_gc:
                            model.gradient_checkpointing_disable()

                        with torch.no_grad():
                            out = model(
                                input_ids=source_prefill_input_ids,
                                attention_mask=source_prefill_attention_mask,
                                position_ids=prefill_position_ids,
                                past_key_values=self.kv_cache_dict[self.base_model_idx][source_model_idx],
                                use_cache=True,
                                return_dict=True,
                            )
                            curr_source_kv_cache = out.past_key_values
                            curr_source_H = self._compute_entropy(out.logits)
                    finally:
                        if had_gc:
                            model.gradient_checkpointing_enable()
                        if was_training:
                            model.train()

                    allow_entropy_fusion = self._should_fuse_source(
                        curr_base_H, curr_source_H
                    )
                    entropy_gate_sources[source_model_idx] = allow_entropy_fusion
                    self.last_entropy_gate_state["sources"][source_model_idx] = {
                        "source_entropy": curr_source_H,
                        "relative_entropy_gap": (
                            None
                            if curr_base_H is None or curr_source_H is None
                            else curr_base_H - curr_source_H
                        ),
                        "allow_fusion": allow_entropy_fusion,
                    }

                    curr_source_kv_cache = hybrid_to_dynamic(curr_source_kv_cache)
                    self.kv_cache_dict[self.base_model_idx][source_model_idx] = clone_kv_cache(
                        curr_source_kv_cache
                    )

                if self.base_model_idx in self.projector_dict:
                    sharer_mask = kv_cache_index[i][0][0][0].item()
                    if sharer_mask > 0:
                        base_cache = clone_kv_cache(curr_base_kv_cache)
                        parallel_delta_cache = (
                            {} if self.multi_source_fusion_mode == "parallel" else None
                        )

                        for source_model_idx in self.projector_dict[self.base_model_idx].keys():
                            if not (sharer_mask & (1 << (source_model_idx - 1))):
                                continue
                            if not entropy_gate_sources.get(source_model_idx, True):
                                continue

                            if self.multi_source_fusion_mode == "sequential":
                                base_cache_ref = curr_base_kv_cache
                            else:
                                base_cache_ref = base_cache

                            for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
                                base_key_cache, base_value_cache = base_cache_ref[target_layer_idx]
                                new_base_key_cache = base_key_cache[:, :, start:end, :]
                                new_base_value_cache = base_value_cache[:, :, start:end, :]
                                new_base_kv_cache = (
                                    new_base_key_cache,
                                    new_base_value_cache,
                                )

                                projected_kv_list = []
                                for source_model_layer_idx, projector_idx in entry:
                                    if not self._is_gate_open(self.projector_list[projector_idx]):
                                        continue
                                    source_key_cache, source_value_cache = self.kv_cache_dict[
                                        self.base_model_idx
                                    ][source_model_idx][source_model_layer_idx]
                                    new_source_key_cache = source_key_cache[:, :, start:end, :]
                                    new_source_value_cache = source_value_cache[:, :, start:end, :]
                                    new_source_kv_cache = (
                                        new_source_key_cache,
                                        new_source_value_cache,
                                    )
                                    projected_key, projected_value = self.projector_list[
                                        projector_idx
                                    ].forward(new_source_kv_cache, new_base_kv_cache)
                                    projected_kv_list.append((projected_key, projected_value))

                                if not projected_kv_list:
                                    continue

                                agg_key, agg_value = projected_kv_list[0]

                                if self.multi_source_fusion_mode == "sequential":
                                    curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = agg_key
                                    curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = agg_value
                                else:
                                    if target_layer_idx not in parallel_delta_cache:
                                        parallel_delta_cache[target_layer_idx] = (
                                            torch.zeros_like(new_base_key_cache),
                                            torch.zeros_like(new_base_value_cache),
                                        )
                                    delta_key, delta_value = parallel_delta_cache[target_layer_idx]
                                    delta_key = delta_key + (agg_key - new_base_key_cache)
                                    delta_value = delta_value + (agg_value - new_base_value_cache)
                                    parallel_delta_cache[target_layer_idx] = (
                                        delta_key,
                                        delta_value,
                                    )

                        if self.multi_source_fusion_mode == "parallel":
                            for target_layer_idx, (delta_key, delta_value) in parallel_delta_cache.items():
                                base_key_cache, base_value_cache = base_cache[target_layer_idx]
                                base_key_slice = base_key_cache[:, :, start:end, :]
                                base_value_slice = base_value_cache[:, :, start:end, :]
                                curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = (
                                    base_key_slice + delta_key
                                )
                                curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = (
                                    base_value_slice + delta_value
                                )

                output.past_key_values = curr_base_kv_cache

        return output

    @torch.no_grad()
    def generate(
        self,
        kv_cache_index,
        input_ids,
        max_new_tokens: Optional[int] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        do_sample: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        max_length: Optional[int] = None,
        use_cache: bool = True,
        streamer=None,
        *args,
        **kwargs,
    ):
        self.kv_cache_dict = dict()

        if isinstance(input_ids, list):
            base_input_ids_for_len = input_ids[self.base_model_idx]
        else:
            base_input_ids_for_len = input_ids
        prompt_len = base_input_ids_for_len.size(1)

        base_model = self.model_list[self.base_model_idx]
        gen_cfg = getattr(base_model, "generation_config", None)
        cfg_obj = gen_cfg if gen_cfg is not None else getattr(base_model, "config", None)
        if eos_token_id is None and cfg_obj is not None:
            eos_token_id = getattr(cfg_obj, "eos_token_id", None)
        if pad_token_id is None and cfg_obj is not None:
            pad_token_id = getattr(cfg_obj, "pad_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0]

        if max_new_tokens is None:
            if max_length is not None:
                if max_length <= prompt_len:
                    max_new_tokens = 0
                else:
                    max_new_tokens = max_length - prompt_len
            else:
                raise ValueError("Provide max_new_tokens or max_length")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        if isinstance(input_ids, list):
            base_input_ids = input_ids[self.base_model_idx]
            base_attention_mask = (
                attention_mask[self.base_model_idx] if attention_mask is not None else None
            )
        else:
            base_input_ids = input_ids
            base_attention_mask = attention_mask

        if base_attention_mask is None:
            base_attention_mask = torch.ones_like(
                base_input_ids, dtype=torch.long, device=base_input_ids.device
            )

        batch_size = base_input_ids.size(0)

        prefill_output = self.forward(
            kv_cache_index=kv_cache_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            *args,
            **kwargs,
        )

        current_past = prefill_output.past_key_values
        all_input_ids = base_input_ids
        current_attention_mask = base_attention_mask

        if streamer is not None:
            streamer.put(base_input_ids)

        eos_set = None
        if eos_token_id is not None:
            eos_set = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
        finished = torch.zeros(batch_size, dtype=torch.bool, device=all_input_ids.device)

        last_logits = prefill_output.logits[:, -1, :]

        if do_sample is None:
            do_sample = False
        effective_temperature = temperature if do_sample else 0.0

        collect_scores = bool(return_dict_in_generate) and bool(output_scores)
        scores = []

        for _ in range(max_new_tokens):
            if collect_scores:
                scores.append(last_logits)

            adjusted_logits = last_logits
            if (
                (repetition_penalty is not None and repetition_penalty != 1.0)
                or (presence_penalty is not None and presence_penalty != 0.0)
                or (frequency_penalty is not None and frequency_penalty != 0.0)
            ):
                adjusted_logits = last_logits.clone()
                vocab_size = adjusted_logits.size(-1)
                for b in range(batch_size):
                    seq_tokens = all_input_ids[b]
                    if seq_tokens.numel() == 0:
                        continue
                    counts = torch.bincount(seq_tokens, minlength=vocab_size)
                    if counts.dtype not in (torch.float32, torch.float64):
                        counts = counts.to(adjusted_logits.dtype)
                    if presence_penalty and presence_penalty != 0.0:
                        presence_mask = counts > 0
                        if presence_mask.any():
                            adjusted_logits[b, presence_mask] = (
                                adjusted_logits[b, presence_mask] - presence_penalty
                            )
                    if frequency_penalty and frequency_penalty != 0.0:
                        adjusted_logits[b] = adjusted_logits[b] - frequency_penalty * counts
                    if repetition_penalty and repetition_penalty != 1.0:
                        rep_mask = counts > 0
                        if rep_mask.any():
                            pos_mask = rep_mask & (adjusted_logits[b] > 0)
                            neg_mask = rep_mask & ~pos_mask
                            if pos_mask.any():
                                adjusted_logits[b, pos_mask] = (
                                    adjusted_logits[b, pos_mask] / repetition_penalty
                                )
                            if neg_mask.any():
                                adjusted_logits[b, neg_mask] = (
                                    adjusted_logits[b, neg_mask] * repetition_penalty
                                )

            next_token = sample_token(
                adjusted_logits,
                temperature=effective_temperature,
                top_p=top_p,
                top_k=top_k,
            )
            if not isinstance(next_token, torch.Tensor):
                next_token = torch.tensor(
                    [next_token], device=all_input_ids.device, dtype=torch.long
                ).repeat(batch_size)

            if eos_set is not None:
                just_finished = torch.zeros_like(finished)
                for eid in eos_set:
                    just_finished |= next_token == eid
                finished = finished | just_finished
                if pad_token_id is not None:
                    next_token = torch.where(
                        finished,
                        torch.tensor(
                            pad_token_id,
                            device=next_token.device,
                            dtype=next_token.dtype,
                        ),
                        next_token,
                    )

            next_token_unsqueezed = next_token.unsqueeze(1)
            all_input_ids = torch.cat([all_input_ids, next_token_unsqueezed], dim=1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        device=current_attention_mask.device,
                        dtype=current_attention_mask.dtype,
                    ),
                ],
                dim=1,
            )

            if streamer is not None:
                streamer.put(next_token_unsqueezed)

            if eos_set is not None and torch.all(finished):
                break

            kv_cache_index = [
                torch.tensor([-1, 0], dtype=torch.long)
                .repeat(1, 1)
                .unsqueeze(0)
                .to(all_input_ids.device)
            ]

            decode_output = self.forward(
                kv_cache_index=kv_cache_index,
                input_ids=next_token_unsqueezed,
                attention_mask=current_attention_mask,
                position_ids=None,
                past_key_values=current_past,
                use_cache=True,
                *args,
                **kwargs,
            )
            last_logits = decode_output.logits[:, -1, :]

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if (
                GreedySearchDecoderOnlyOutput is not None
                and SampleDecoderOnlyOutput is not None
            ):
                if do_sample:
                    return SampleDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
                return GreedySearchDecoderOnlyOutput(
                    sequences=all_input_ids,
                    scores=scores if collect_scores else None,
                )

            result = {"sequences": all_input_ids}
            if collect_scores:
                result["scores"] = scores
            return ModelOutput(**result)
        return all_input_ids
