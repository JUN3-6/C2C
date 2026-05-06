"""
The ensemble of multiple standard transformers LLM models, with automatic kv-cache projection. It shares the same interface as the standard transformers LLM models.
"""

from typing import Any, Dict, List, Optional, Union
import torch
import math
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import json

from rosetta.model.projector import Projector
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
from rosetta.model.sampling import sample_token
from transformers.utils import ModelOutput
try:
    from transformers.generation.utils import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
except Exception:
    GreedySearchDecoderOnlyOutput = None
    SampleDecoderOnlyOutput = None

def _attach_legacy_cache_views(kv_cache: DynamicCache) -> DynamicCache:
    if kv_cache is None:
        return None
    if hasattr(kv_cache, "key_cache") and hasattr(kv_cache, "value_cache"):
        return kv_cache
    if hasattr(kv_cache, "layers"):
        kv_cache.key_cache = [layer.keys for layer in kv_cache.layers]
        kv_cache.value_cache = [layer.values for layer in kv_cache.layers]
    return kv_cache

def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
    kv_cache = _attach_legacy_cache_views(kv_cache)
    legacy_cache = [(k.clone().detach(), v.clone().detach()) for k, v in zip(kv_cache.key_cache, kv_cache.value_cache)]
    return _attach_legacy_cache_views(DynamicCache.from_legacy_cache(legacy_cache))

def hybrid_to_dynamic(hybrid_cache):
    if hybrid_cache is None:
        return None
    if isinstance(hybrid_cache, DynamicCache):
        return _attach_legacy_cache_views(hybrid_cache)

    # 手动从 HybridCache 提取
    if hasattr(hybrid_cache, "key_cache") and hasattr(hybrid_cache, "value_cache"):
        keys = hybrid_cache.key_cache
        values = hybrid_cache.value_cache
        assert len(keys) == len(values), "key/value layers do not match"

        legacy_cache = [(k, v) for k, v in zip(keys, values)]
        return _attach_legacy_cache_views(DynamicCache.from_legacy_cache(legacy_cache))

    raise TypeError(f"Unsupported cache type: {type(hybrid_cache)}")

class RosettaModel(nn.Module):
    """
    Drop in replacement for the standard transformers LLM models, like Qwen3ForCausalLM
    """
    def __init__(
        self, 
        model_list: List[PreTrainedModel], 
        base_model_idx = 0, 
        projector_list: List[Projector] = [], 
        router: Optional[nn.Module] = None,
        projector_bank_dicts: Optional[List[Dict[str, Any]]] = None,
        router_fuse_threshold: float = 0.5,
        include_response: bool = False, 
        multi_source_fusion_mode: str = "parallel",
        static_gate_enabled: bool = True,
        threshold_abs: float = 0.45,
        threshold_rel: float = 0.15,
        normalize_entropy: bool = True,
        entropy_gate_enabled: bool = True, 
        entropy_eps: float = 1e-12,
        update_decode_past: bool = True,
    ):
        super().__init__()
        # model list: a list of model, model 0 by default is the base model
        # projector list: a list of projector
        # standard init with additional model list parameter
        # kv-cache dict: key (source_model_idx, target_model_idx), value (Cache), assume only convert at prefill with one type of model
        # projector dict: key (source_model_idx, target_model_idx) value dict(key (source_model_layer_idx, M_target value )

        self.base_model_idx = base_model_idx
        self.model_list = nn.ModuleList(model_list)

        device = model_list[base_model_idx].device
        dtype = model_list[base_model_idx].dtype
        self.projector_list = nn.ModuleList(projector_list).to(device=device, dtype=dtype)
        self.router = router.to(device=device) if router is not None else None

        self.projector_dict = {}
        self.projector_bank_dicts: List[Dict[str, Any]] = []
        self.kv_cache_dict = {}
        self._generation_hook_handlers = []
        self.router_fuse_threshold = router_fuse_threshold
        self.last_routing_state = {}
        self.routing_stats = {
            "decision_total": 0,
            "fuse_total": 0,
            "skip_total": 0,
            "bank_counts": {},
        }
        self._active_routing_bank_idx: Optional[int] = None

        # Multi-source fusion mode:
        # "sequential" (default): each source updates base cache iteratively
        # "parallel": all sources project from clean base cache, then sum projections
        #
        # `include_response` keeps its legacy meaning: continue fusion during
        # decode/response tokens. Routed inference uses a separate prefill-only
        # path and does not force this flag on.
        self.include_response = include_response
        if multi_source_fusion_mode not in ["sequential", "parallel"]:
            raise ValueError(f"multi_source_fusion_mode must be 'sequential' or 'parallel', got '{multi_source_fusion_mode}'")
        self.multi_source_fusion_mode = multi_source_fusion_mode

        # Routing replaces the legacy gates for the routed path.
        self.static_gate_enabled = False if self.router is not None else static_gate_enabled
        self.threshold_abs = threshold_abs
        self.threshold_rel = threshold_rel
        self.normalize_entropy = normalize_entropy
        self.entropy_gate_enabled = False if self.router is not None else entropy_gate_enabled
        self.entropy_eps = entropy_eps
        self.update_decode_past = update_decode_past
        self.last_entropy_gate_state = {} #for debugging
        self._prefill_entropy_gate_override = None
        self._reset_entropy_gate_stats()

        if projector_bank_dicts:
            self.set_projector_banks(projector_bank_dicts)

    @property
    def device(self):
        return self.model_list[self.base_model_idx].device

    @property
    def routing_enabled(self) -> bool:
        return self.router is not None

    def _router_uses_hidden_features(self) -> bool:
        if not self.routing_enabled:
            return False
        return getattr(self.router, "feature_source", "kv") in {"hidden", "hidden_binned"}

    def _router_needs_hidden_states(self) -> bool:
        if not self.routing_enabled:
            return False
        return getattr(self.router, "feature_source", "kv") in {
            "hidden",
            "hidden_binned",
            "postfusion_option_logits_hidden_binned",
        }

    def _router_uses_projector_in_features(self) -> bool:
        if not self.routing_enabled:
            return False
        return getattr(self.router, "feature_source", "kv") in PROJECTOR_IN_FEATURE_SOURCES

    def _router_uses_postfusion_features(self) -> bool:
        if not self.routing_enabled:
            return False
        return getattr(self.router, "feature_source", "kv") in POSTFUSION_FEATURE_SOURCES

    def _router_uses_postfusion_delta_features(self) -> bool:
        if not self.routing_enabled:
            return False
        return getattr(self.router, "feature_source", "kv") in POSTFUSION_DELTA_FEATURE_SOURCES

    def _router_uses_postfusion_probe_logits_features(self) -> bool:
        if not self.routing_enabled:
            return False
        return getattr(self.router, "feature_source", "kv") in POSTFUSION_PROBE_LOGIT_FEATURE_SOURCES

    def _router_uses_postfusion_option_logits_features(self) -> bool:
        if not self.routing_enabled:
            return False
        return getattr(self.router, "feature_source", "kv") in POSTFUSION_OPTION_LOGIT_FEATURE_SOURCES

    def _router_uses_shared_latent_fusion_features(self) -> bool:
        if not self.routing_enabled:
            return False
        return (
            getattr(self.router, "feature_source", "kv")
            in {"shared_latent_receiver", "shared_latent_fusion", "shared_latent_pair"}
            and hasattr(self.router, "extract_query_feature_from_fused_cache")
        )
    
    def to(self, device):
        """
        Move the RosettaModel and all underlying models and projectors to the specified device.
        """
        super().to(device)
        for model in self.model_list:
            model.to(device)
        for projector in self.projector_list:
            projector.to(device)
        if self.router is not None:
            self.router.to(device)
        return self

    def _use_entropy_gate(self) -> bool:
        if not self.entropy_gate_enabled:
            return False
        return (not self.training) and (
            torch.is_inference_mode_enabled() or not torch.is_grad_enabled()
        )

    def _reset_entropy_gate_stats(self) -> None:
        self.entropy_gate_stats = {
            "opportunity_total": 0,
            "checked_total": 0,
            "allowed_total": 0,
            "blocked_total": 0,
        }

    def _record_entropy_gate_decision(
        self,
        allow_fusion: bool,
        gate_was_checked: bool,
    ) -> None:
        self.entropy_gate_stats["opportunity_total"] += 1
        if gate_was_checked:
            self.entropy_gate_stats["checked_total"] += 1
        if allow_fusion:
            self.entropy_gate_stats["allowed_total"] += 1
        else:
            self.entropy_gate_stats["blocked_total"] += 1

    def _compute_entropy(self, logits) -> Optional[float]:
        if logits is None:
            return None
        next_token_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        probs = torch.softmax(next_token_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(self.entropy_eps))).sum(dim=-1)
        if self.normalize_entropy and next_token_logits.size(-1) > 1:
            entropy = entropy / math.log(next_token_logits.size(-1))
        return float(entropy.mean().detach().cpu())

    def _get_top_token_id(self, logits) -> Optional[int]:
        if logits is None:
            return None
        next_token_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        top_token_id = torch.argmax(next_token_logits, dim=-1)
        return int(top_token_id[0].detach().cpu().item())

    def _should_fuse_source(
        self,
        curr_base_H: Optional[float],
        curr_source_H: Optional[float],
    ) -> bool:
        if not self._use_entropy_gate():
            return True
        if curr_base_H is None or curr_source_H is None:
            return True
        abs_gate = curr_base_H > self.threshold_abs
        rel_gate = (curr_source_H - curr_base_H) < self.threshold_rel
        return bool(abs_gate and rel_gate)

    def _compute_model_terminal_entropy(
        self,
        model_idx: int,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache] = None,
    ) -> Optional[float]:
        if input_ids is None:
            return None

        model = self.model_list[model_idx]
        was_training = model.training
        had_gc = getattr(model, "is_gradient_checkpointing", False)

        try:
            if was_training:
                model.eval()
            if had_gc:
                model.gradient_checkpointing_disable()

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=False,
                    return_dict=True,
                )
            return self._compute_entropy(out.logits)
        finally:
            if had_gc:
                model.gradient_checkpointing_enable()
            if was_training:
                model.train()

    def _compute_generate_prefill_entropy_override(
        self,
        kv_cache_index,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
    ):
        if not self._use_entropy_gate():
            return None
        if kv_cache_index is None:
            return None

        selected_sources = set()
        for section in kv_cache_index:
            if section is None or section.numel() == 0:
                continue
            sharer_mask = int(section[0][0][0].item())
            if sharer_mask <= 0:
                continue
            for source_model_idx in range(1, len(self.model_list)):
                if sharer_mask & (1 << (source_model_idx - 1)):
                    selected_sources.add(source_model_idx)

        if not selected_sources:
            return None

        if isinstance(input_ids, list):
            base_input_ids = input_ids[self.base_model_idx]
            base_attention_mask = (
                attention_mask[self.base_model_idx] if attention_mask is not None else None
            )
        else:
            base_input_ids = input_ids
            base_attention_mask = attention_mask

        base_entropy = self._compute_model_terminal_entropy(
            model_idx=self.base_model_idx,
            input_ids=base_input_ids,
            attention_mask=base_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        state = {
            "mode": "generate_prefill",
            "entropy_source": "full_prompt_terminal",
            "base_entropy": base_entropy,
            "sources": {},
        }

        for source_model_idx in sorted(selected_sources):
            if isinstance(input_ids, list):
                source_input_ids = input_ids[source_model_idx]
                source_attention_mask = (
                    attention_mask[source_model_idx] if attention_mask is not None else None
                )
            else:
                source_input_ids = input_ids
                source_attention_mask = attention_mask

            source_entropy = self._compute_model_terminal_entropy(
                model_idx=source_model_idx,
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                position_ids=position_ids,
                past_key_values=None,
            )
            allow_fusion = self._should_fuse_source(base_entropy, source_entropy)
            state["sources"][source_model_idx] = {
                "source_entropy": source_entropy,
                "relative_entropy_gap": (
                    None
                    if base_entropy is None or source_entropy is None
                    else source_entropy - base_entropy
                ),
                "allow_fusion": allow_fusion,
            }

        return state

    @staticmethod
    def _bank_layout_signature(bank_config: Dict[str, Any]) -> Dict[str, Any]:
        def normalize(obj):
            if isinstance(obj, dict):
                return {int(k): normalize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [[int(src_layer), 0] for src_layer, _ in obj]
            return obj

        return normalize(bank_config)

    def _validate_projector_banks(self, bank_dicts: List[Dict[str, Any]]) -> None:
        if not bank_dicts:
            return

        reference_layout = self._bank_layout_signature(bank_dicts[0])
        for bank_idx, bank_config in enumerate(bank_dicts):
            if self._bank_layout_signature(bank_config) != reference_layout:
                raise ValueError(
                    "All projector banks must share the same source/target topology. "
                    f"Bank 0 and bank {bank_idx} differ."
                )

        if self.routing_enabled and len(self.model_list) != 2:
            raise ValueError(
                "Routing v1 supports single-sharer only. Provide exactly one receiver "
                "and one sharer model."
            )

    def set_projector_banks(self, bank_dicts: List[Dict[str, Any]]) -> None:
        normalized = [
            RosettaModel._convert_dict_keys_to_ints(bank_config)
            for bank_config in bank_dicts
        ]
        self._validate_projector_banks(normalized)
        self.projector_bank_dicts = normalized
        self.projector_dict = normalized[0] if normalized else {}

    def _get_projector_config(
        self,
        bank_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.projector_bank_dicts:
            return self.projector_dict

        if bank_idx is None:
            bank_idx = 0
        if bank_idx < 0 or bank_idx >= len(self.projector_bank_dicts):
            raise IndexError(
                f"bank_idx={bank_idx} is out of range for {len(self.projector_bank_dicts)} banks"
            )
        return self.projector_bank_dicts[bank_idx]

    def save_projector_bank_config(self, file_name: str) -> None:
        payload = {
            "format": "projector_banks_v1",
            "num_banks": len(self.projector_bank_dicts),
            "bank_configs": self.projector_bank_dicts,
        }
        with open(file_name, "w") as f:
            json.dump(payload, f)

    def _routing_source_model_idx(self) -> int:
        if len(self.model_list) != 2:
            raise ValueError(
                "Routing v1 supports single-sharer only and expects model_list=[receiver, sharer]"
            )
        return 1 if self.base_model_idx == 0 else 0
    
    # set projector 
    def set_projector_config(self, 
                        source_model_idx: int, 
                        source_model_layer_idx: int, 
                        target_model_idx: int,
                        target_model_layer_idx: int, 
                        projector_idx: int):
        """
        Set the projector configuration
        Args:
            source_model_idx: int, the index of the source model
            source_model_layer_idx: int, the index of the source model layer
            target_model_idx: int, the index of the target model
            target_model_layer_idx: int, the index of the target model layer
            projector_idx: int, the index of the projector

        The projector dict structure supports multiple projectors per target layer.
        Structure:
        {
            target_model_idx: {
                source_model_idx: {
                    target_model_layer_idx: [(source_model_layer_idx, projector_idx), ...]
                }
            }
        }
        Repeated calls for the same (target, source, target_layer) append additional pairs.
        """

        if target_model_idx not in self.projector_dict.keys():
            self.projector_dict[target_model_idx] = {}
        if source_model_idx not in self.projector_dict[target_model_idx].keys():
            self.projector_dict[target_model_idx][source_model_idx] = {}
        # Accumulate list of (source_layer, projector_idx) for this target layer
        layer_entry = self.projector_dict[target_model_idx][source_model_idx].get(target_model_layer_idx)
        if layer_entry is None:
            self.projector_dict[target_model_idx][source_model_idx][target_model_layer_idx] = [(source_model_layer_idx, projector_idx)]
        else:
            layer_entry.append((source_model_layer_idx, projector_idx))

    def _is_gate_open(self, proj) -> bool:
        """Safe gate check with fallback for projectors without is_gate_open()."""
        if not self.static_gate_enabled:
            return True
        fn = getattr(proj, 'is_gate_open', None)
        return fn() if fn is not None else True

    def _get_active_source_models(self) -> set:
        """Return source model indices that have at least one projector with an open gate.
        When a source has no open gates, its entire forward pass can be skipped."""
        active = set()
        projector_config = self._get_projector_config()
        if self.base_model_idx not in projector_config:
            return active
        for src_idx, layer_map in projector_config[self.base_model_idx].items():
            for _, entry in layer_map.items():
                for _, proj_idx in entry:
                    if self._is_gate_open(self.projector_list[proj_idx]):
                        active.add(src_idx)
                        break
                if src_idx in active:
                    break
        return active

    def _has_cached_source_pair(self, source_model_idx: Optional[int]) -> bool:
        if source_model_idx is None:
            return False
        return (
            self.base_model_idx in self.kv_cache_dict
            and self.base_model_idx in self.kv_cache_dict[self.base_model_idx]
            and source_model_idx in self.kv_cache_dict[self.base_model_idx]
            and self.kv_cache_dict[self.base_model_idx][self.base_model_idx] is not None
            and self.kv_cache_dict[self.base_model_idx][source_model_idx] is not None
        )

    def _should_use_last_section_fusion(
        self,
        *,
        seqlen: int,
        source_model_idx: Optional[int],
    ) -> bool:
        if not self._has_cached_source_pair(source_model_idx):
            return False
        if self.routing_enabled:
            # Routing now applies banked fusion during prefill sections.
            # Keep legacy last-section hook fusion only when decode-time
            # include_response behavior is explicitly requested.
            return self.include_response and seqlen > 1
        return self.include_response
    
    def load_projector(self, projector_list):
        self.projector_list = nn.ModuleList(projector_list).to(device=self.device)

    def get_projector(self, 
                        source_model_idx, 
                        source_model_layer_idx, 
                        target_model_idx,
                        target_model_layer_idx,
                        bank_idx: Optional[int] = None):
        projector_config = self._get_projector_config(bank_idx)
        pair_list = projector_config[target_model_idx][source_model_idx][target_model_layer_idx]
        if len(pair_list) == 0:
            raise ValueError("No projector configured for the given target layer")
        # Prefer exact source layer match
        for src_layer, projector_id in pair_list:
            if src_layer == source_model_layer_idx:
                return self.projector_list[projector_id]
        # Fallback: return the first projector
        return self.projector_list[pair_list[0][1]]

    @staticmethod
    def load_json(file_name):
        with open(file_name, "r") as f:
            result = json.load(f)
        return result
    
    @staticmethod
    def _convert_dict_keys_to_ints(obj):
        """
        Recursively convert dictionary keys that look like integers back to int.
        This reverses json.dump's coercion of dict keys to strings.
        """
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                if isinstance(key, str) and key.lstrip('-').isdigit():
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
        if self.projector_bank_dicts:
            if file_name.endswith("projector_config.json"):
                bank_file = file_name.replace(
                    "projector_config.json",
                    "projector_bank_config.json",
                )
            else:
                bank_file = file_name + ".banks"
            self.save_projector_bank_config(bank_file)

    
    def load_projector_config(self, config_path):
        if config_path.endswith(".json"):
            loaded = RosettaModel.load_json(config_path)
            if loaded.get("format") == "projector_banks_v1":
                bank_configs = loaded.get("bank_configs", [])
                self.set_projector_banks(bank_configs)
            else:
                self.projector_dict = RosettaModel._convert_dict_keys_to_ints(loaded)
                self.projector_bank_dicts = []

    def set_kv_cache_dict(self, source_model_idx, target_model_idx, cache):
        if target_model_idx not in self.kv_cache_dict.keys():
            self.kv_cache_dict[target_model_idx] = {}
        if cache is None:
            # Initialize with a DynamicCache instead of RosettaCache for now
            self.kv_cache_dict[target_model_idx][source_model_idx] = DynamicCache() # noqa, maybe we should use RosettaCache here
        else:
            self.kv_cache_dict[target_model_idx][source_model_idx] = cache

    @staticmethod
    def _monkeypatch_qwen3_attention_forward(attn_module, new_k_cache, new_v_cache):
        """
        Monkeypatch Qwen3Attention.forward so that *current step* attention uses the
        provided key/value (in cache space) before computing attention.

        This avoids editing transformers' Qwen3 code while ensuring the modified KV
        is used in the same forward pass (not just for the next token).

        new_k_cache/new_v_cache: (B, kv_heads, q_len, head_dim) in the SAME space as
        Qwen3Attention's key_states/value_states AFTER k_norm + RoPE (k) and reshape (v).
        """
        import types

        # Lazy imports to avoid hard dependency at module import time
        from transformers.models.qwen3.modeling_qwen3 import (  # type: ignore
            apply_rotary_pos_emb,
            eager_attention_forward,
            ALL_ATTENTION_FUNCTIONS,
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
            # This is essentially Qwen3Attention.forward with one injection point.
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # === Injection point (before cache update & attention) ===
            # Replace current-token key/value with provided cache-space tensors.
            # Expect same shape as key_states/value_states at this moment:
            # (B, kv_heads, q_len, head_dim)
            if new_k_cache is not None and new_v_cache is not None:
                # Only replace if compatible
                if key_states.shape == new_k_cache.shape:
                    key_states = new_k_cache
                if value_states.shape == new_v_cache.shape:
                    value_states = new_v_cache

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    # fall back to eager, same as upstream behavior (warning omitted here)
                    attention_interface = eager_attention_forward
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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

    def _apply_projector_bank_to_cache(
        self,
        base_output_kv_cache: DynamicCache,
        source_output_kv_cache: DynamicCache,
        source_model_idx: int,
        new_length: int,
        bank_selection: Optional[torch.Tensor] = None,
        default_bank_idx: int = 0,
    ) -> DynamicCache:
        fused_kv_cache = clone_kv_cache(base_output_kv_cache)

        if bank_selection is None:
            bank_to_mask = {default_bank_idx: None}
        else:
            bank_to_mask = {}
            for bank_idx in torch.unique(bank_selection).tolist():
                bank_idx = int(bank_idx)
                if bank_idx < 0:
                    continue
                bank_to_mask[bank_idx] = bank_selection == bank_idx

        for bank_idx, batch_mask in bank_to_mask.items():
            projector_config = self._get_projector_config(bank_idx)
            if (
                self.base_model_idx not in projector_config
                or source_model_idx not in projector_config[self.base_model_idx]
            ):
                continue

            for target_layer_idx, entry in projector_config[self.base_model_idx][
                source_model_idx
            ].items():
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
                    projected_key, projected_value = self.projector_list[
                        projector_idx
                    ].forward(
                        new_source_kv_cache,
                        new_base_kv_cache,
                    )
                    projected_kv_list.append((projected_key, projected_value))

                if not projected_kv_list:
                    continue

                agg_key, agg_value = projected_kv_list[0]
                target_key_slice = fused_kv_cache.key_cache[target_layer_idx][
                    :, :, -new_length:, :
                ]
                target_value_slice = fused_kv_cache.value_cache[target_layer_idx][
                    :, :, -new_length:, :
                ]
                agg_key = agg_key.to(dtype=target_key_slice.dtype, device=target_key_slice.device)
                agg_value = agg_value.to(dtype=target_value_slice.dtype, device=target_value_slice.device)
                if batch_mask is None:
                    target_key_slice.copy_(agg_key)
                    target_value_slice.copy_(agg_value)
                else:
                    target_key_slice[batch_mask] = agg_key[batch_mask]
                    target_value_slice[batch_mask] = agg_value[batch_mask]

        return fused_kv_cache

    def _extract_projector_in_router_feature(
        self,
        *,
        base_output_kv_cache: DynamicCache,
        source_output_kv_cache: DynamicCache,
        source_model_idx: int,
        new_length: Optional[int],
    ) -> torch.Tensor:
        if len(self.projector_bank_dicts) != 1:
            raise RuntimeError(
                "Projector-in router features currently require exactly one projector bank."
            )
        return extract_projector_in_feature(
            base_cache=base_output_kv_cache,
            source_cache=source_output_kv_cache,
            projector_list=list(self.projector_list),
            projector_bank_config=self._get_projector_config(0),
            base_model_idx=self.base_model_idx,
            source_model_idx=source_model_idx,
            new_length=new_length,
            feature_source=getattr(self.router, "feature_source", "projector_in"),
        )

    def _extract_postfusion_router_feature(
        self,
        *,
        base_output_kv_cache: DynamicCache,
        source_output_kv_cache: DynamicCache,
        source_model_idx: int,
        new_length: Optional[int],
    ) -> torch.Tensor:
        if len(self.projector_bank_dicts) != 1:
            raise RuntimeError(
                "Post-fusion router features currently require exactly one projector bank."
            )
        candidate_fused_cache = self._apply_projector_bank_to_cache(
            base_output_kv_cache=base_output_kv_cache,
            source_output_kv_cache=source_output_kv_cache,
            source_model_idx=source_model_idx,
            new_length=new_length,
            bank_selection=None,
            default_bank_idx=0,
        )
        return extract_postfusion_delta_feature(
            base_cache=base_output_kv_cache,
            fused_cache=candidate_fused_cache,
            projector_bank_config=self._get_projector_config(0),
            base_model_idx=self.base_model_idx,
            source_model_idx=source_model_idx,
            new_length=new_length,
            feature_source=getattr(self.router, "feature_source", "postfusion_delta_stats"),
        )

    def _build_probe_inputs(
        self,
        *,
        prefill_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        cache_length: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        next_token_logits = prefill_logits[:, -1, :] if prefill_logits.dim() == 3 else prefill_logits
        probe_input_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)

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

    def _build_fixed_probe_inputs(
        self,
        *,
        probe_token_ids: List[int],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        cache_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if not probe_token_ids:
            raise RuntimeError(
                "postfusion_option_logits router requires option_response_token_ids "
                "stored in the router config."
            )
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

    def _probe_base_model_logits(
        self,
        *,
        cache: DynamicCache,
        probe_input_ids: torch.Tensor,
        probe_attention_mask: Optional[torch.Tensor],
        probe_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model_list[self.base_model_idx]
        was_training = model.training
        had_gc = getattr(model, "is_gradient_checkpointing", False)
        try:
            if was_training:
                model.eval()
            if had_gc:
                model.gradient_checkpointing_disable()
            with torch.no_grad():
                output = model(
                    input_ids=probe_input_ids,
                    attention_mask=probe_attention_mask,
                    position_ids=probe_position_ids,
                    past_key_values=clone_kv_cache(cache),
                    use_cache=False,
                    return_dict=True,
                )
            return output.logits[:, -1, :].detach()
        finally:
            if had_gc:
                model.gradient_checkpointing_enable()
            if was_training:
                model.train()

    def _extract_postfusion_probe_logits_router_feature(
        self,
        *,
        base_output_kv_cache: DynamicCache,
        source_output_kv_cache: DynamicCache,
        source_model_idx: int,
        new_length: Optional[int],
        prefill_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if len(self.projector_bank_dicts) != 1:
            raise RuntimeError(
                "Post-fusion probe-logit router features currently require exactly one projector bank."
            )
        candidate_fused_cache = self._apply_projector_bank_to_cache(
            base_output_kv_cache=base_output_kv_cache,
            source_output_kv_cache=source_output_kv_cache,
            source_model_idx=source_model_idx,
            new_length=new_length,
            bank_selection=None,
            default_bank_idx=0,
        )
        cache_length = int(base_output_kv_cache.key_cache[0].shape[2])
        probe_input_ids, probe_attention_mask, probe_position_ids = self._build_probe_inputs(
            prefill_logits=prefill_logits,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_length=cache_length,
        )
        receiver_logits = self._probe_base_model_logits(
            cache=base_output_kv_cache,
            probe_input_ids=probe_input_ids,
            probe_attention_mask=probe_attention_mask,
            probe_position_ids=probe_position_ids,
        )
        fused_logits = self._probe_base_model_logits(
            cache=candidate_fused_cache,
            probe_input_ids=probe_input_ids,
            probe_attention_mask=probe_attention_mask,
            probe_position_ids=probe_position_ids,
        )
        return extract_probe_logits_feature(
            receiver_logits=receiver_logits,
            fused_logits=fused_logits,
            prefill_logits=prefill_logits,
            feature_source=getattr(self.router, "feature_source", "postfusion_probe_logits"),
        )

    def _extract_postfusion_option_logits_router_feature(
        self,
        *,
        base_output_kv_cache: DynamicCache,
        source_output_kv_cache: DynamicCache,
        source_model_idx: int,
        new_length: Optional[int],
        prefill_logits: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        base_last_hidden: Optional[torch.Tensor] = None,
        source_last_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(self.projector_bank_dicts) != 1:
            raise RuntimeError(
                "Post-fusion option-logit router features currently require exactly one projector bank."
            )
        option_token_ids = getattr(self.router, "option_token_ids", None)
        option_response_token_ids = getattr(self.router, "option_response_token_ids", None)
        if not option_token_ids or not option_response_token_ids:
            raise RuntimeError(
                "postfusion_option_logits router requires option_token_ids and "
                "option_response_token_ids stored in the router config."
            )
        candidate_fused_cache = self._apply_projector_bank_to_cache(
            base_output_kv_cache=base_output_kv_cache,
            source_output_kv_cache=source_output_kv_cache,
            source_model_idx=source_model_idx,
            new_length=new_length,
            bank_selection=None,
            default_bank_idx=0,
        )
        cache_length = int(base_output_kv_cache.key_cache[0].shape[2])
        probe_input_ids, probe_attention_mask, probe_position_ids = self._build_fixed_probe_inputs(
            probe_token_ids=option_response_token_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_length=cache_length,
            device=base_output_kv_cache.key_cache[0].device,
        )
        receiver_logits = self._probe_base_model_logits(
            cache=base_output_kv_cache,
            probe_input_ids=probe_input_ids,
            probe_attention_mask=probe_attention_mask,
            probe_position_ids=probe_position_ids,
        )
        fused_logits = self._probe_base_model_logits(
            cache=candidate_fused_cache,
            probe_input_ids=probe_input_ids,
            probe_attention_mask=probe_attention_mask,
            probe_position_ids=probe_position_ids,
        )
        option_feature = extract_option_logits_feature(
            receiver_logits=receiver_logits,
            fused_logits=fused_logits,
            option_token_ids=option_token_ids,
            prefill_logits=prefill_logits,
            feature_source=getattr(self.router, "feature_source", "postfusion_option_logits"),
        )
        if getattr(self.router, "feature_source", "kv") != "postfusion_option_logits_hidden_binned":
            return option_feature

        if base_last_hidden is None or source_last_hidden is None:
            raise RuntimeError(
                "postfusion_option_logits_hidden_binned requires terminal hidden states."
            )
        hidden_feature_dim = int(getattr(self.router, "input_dim", 0)) - int(option_feature.size(-1))
        if hidden_feature_dim <= 0 or hidden_feature_dim % 4 != 0:
            raise RuntimeError(
                "Invalid hybrid router input_dim: expected option feature dim plus "
                f"4 * bins, got input_dim={getattr(self.router, 'input_dim', None)} "
                f"option_dim={option_feature.size(-1)}"
            )
        hidden_feature = extract_hidden_binned_feature(
            base_last_hidden=base_last_hidden,
            source_last_hidden=source_last_hidden,
            bins_per_model=hidden_feature_dim // 4,
        ).to(device=option_feature.device, dtype=option_feature.dtype)
        return torch.cat([option_feature, hidden_feature], dim=-1)

    def _run_router(
        self,
        base_output_kv_cache: DynamicCache,
        source_output_kv_cache: DynamicCache,
        pooled_feature: Optional[torch.Tensor] = None,
    ) -> dict:
        if not self.routing_enabled:
            raise RuntimeError("Router is not configured")

        routing = self.router.predict(
            base_cache=base_output_kv_cache,
            source_cache=source_output_kv_cache,
            fuse_threshold=self.router_fuse_threshold,
            pooled_feature=pooled_feature,
        )
        self.last_routing_state = {
            "binary_logits": routing["binary_logits"].detach().cpu(),
            "selection_logits": routing["selection_logits"].detach().cpu(),
            "fuse_probability": routing["fuse_probability"].detach().cpu(),
            "should_fuse": routing["should_fuse"].detach().cpu(),
            "selected_bank": routing["selected_bank"].detach().cpu(),
            "pooled_feature": routing["pooled_feature"].detach().cpu(),
        }
        if "action_logits" in routing:
            self.last_routing_state["action_logits"] = routing["action_logits"].detach().cpu()
        if "action_probabilities" in routing:
            self.last_routing_state["action_probabilities"] = routing[
                "action_probabilities"
            ].detach().cpu()
        if "selected_action" in routing:
            self.last_routing_state["selected_action"] = routing["selected_action"].detach().cpu()
        self._record_routing_decision(routing)
        if routing["selected_bank"].numel() == 1:
            selected = int(routing["selected_bank"][0].item())
            self._active_routing_bank_idx = selected if selected >= 0 else None
        else:
            self._active_routing_bank_idx = None
        return routing

    def _record_routing_decision(self, routing: dict) -> None:
        should_fuse = routing["should_fuse"].detach().cpu().view(-1).bool()
        selected_bank = routing["selected_bank"].detach().cpu().view(-1).long()
        decision_total = int(should_fuse.numel())
        fuse_total = int(should_fuse.sum().item())
        skip_total = decision_total - fuse_total
        self.routing_stats["decision_total"] += decision_total
        self.routing_stats["fuse_total"] += fuse_total
        self.routing_stats["skip_total"] += skip_total
        bank_counts = self.routing_stats.setdefault("bank_counts", {})
        for bank_idx in selected_bank[should_fuse].tolist():
            key = str(int(bank_idx))
            bank_counts[key] = int(bank_counts.get(key, 0)) + 1

    def register_hooks(
        self, 
        input_ids, 
        attention_mask, 
        position_ids, 
        base_kv_cache, 
        source_model_idx, 
        source_kv_cache
    ):

        base_kv_copy = clone_kv_cache(base_kv_cache)
        source_kv_copy = clone_kv_cache(source_kv_cache)

        new_length = input_ids.shape[1]
        router_needs_hidden = self._router_needs_hidden_states()
        router_uses_hidden = self._router_uses_hidden_features()
        router_uses_projector_in = self._router_uses_projector_in_features()
        router_uses_postfusion_delta = self._router_uses_postfusion_delta_features()
        router_uses_postfusion_probe_logits = (
            self._router_uses_postfusion_probe_logits_features()
        )
        router_uses_postfusion_option_logits = (
            self._router_uses_postfusion_option_logits_features()
        )
        router_uses_shared_latent_fusion = (
            self._router_uses_shared_latent_fusion_features()
        )

        base_output = self.model_list[self.base_model_idx].forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=base_kv_copy,
            labels=None,
            use_cache=True,
            output_hidden_states=router_needs_hidden,
            return_dict=True,
        )
        base_output_kv_cache = hybrid_to_dynamic(base_output.past_key_values)
        curr_base_H = self._compute_entropy(base_output.logits)
        curr_base_top_token_id = self._get_top_token_id(base_output.logits)
        
        source_output = self.model_list[source_model_idx].forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=source_kv_copy,
            labels=None,
            use_cache=True,
            output_hidden_states=router_needs_hidden,
            return_dict=True,
        )
        source_output_kv_cache = hybrid_to_dynamic(source_output.past_key_values)
        curr_source_H = self._compute_entropy(source_output.logits)
        curr_source_top_token_id = self._get_top_token_id(source_output.logits)

        fused_kv_cache = clone_kv_cache(base_output_kv_cache)
        if self.routing_enabled:
            pooled_feature = None
            if router_uses_hidden:
                pooled_feature = self.router.extract_query_feature_from_hidden(
                    base_output.hidden_states[-1][:, -1, :],
                    source_output.hidden_states[-1][:, -1, :],
                )
            elif router_uses_projector_in:
                pooled_feature = self._extract_projector_in_router_feature(
                    base_output_kv_cache=base_output_kv_cache,
                    source_output_kv_cache=source_output_kv_cache,
                    source_model_idx=source_model_idx,
                    new_length=new_length,
                )
            elif router_uses_postfusion_delta:
                pooled_feature = self._extract_postfusion_router_feature(
                    base_output_kv_cache=base_output_kv_cache,
                    source_output_kv_cache=source_output_kv_cache,
                    source_model_idx=source_model_idx,
                    new_length=new_length,
                )
            elif router_uses_postfusion_probe_logits:
                pooled_feature = self._extract_postfusion_probe_logits_router_feature(
                    base_output_kv_cache=base_output_kv_cache,
                    source_output_kv_cache=source_output_kv_cache,
                    source_model_idx=source_model_idx,
                    new_length=new_length,
                    prefill_logits=base_output.logits,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            elif router_uses_postfusion_option_logits:
                pooled_feature = self._extract_postfusion_option_logits_router_feature(
                    base_output_kv_cache=base_output_kv_cache,
                    source_output_kv_cache=source_output_kv_cache,
                    source_model_idx=source_model_idx,
                    new_length=new_length,
                    prefill_logits=base_output.logits,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    base_last_hidden=base_output.hidden_states[-1][:, -1, :]
                    if router_needs_hidden
                    else None,
                    source_last_hidden=source_output.hidden_states[-1][:, -1, :]
                    if router_needs_hidden
                    else None,
                )
            elif router_uses_shared_latent_fusion:
                if (
                    getattr(self.router, "no_skip_routing", False)
                    and getattr(self.router, "feature_source", "kv") == "shared_latent_pair"
                    and len(self.projector_bank_dicts) >= 2
                ):
                    bank0_cache = self._apply_projector_bank_to_cache(
                        base_output_kv_cache=base_output_kv_cache,
                        source_output_kv_cache=source_output_kv_cache,
                        source_model_idx=source_model_idx,
                        new_length=new_length,
                        bank_selection=None,
                        default_bank_idx=0,
                    )
                    bank1_cache = self._apply_projector_bank_to_cache(
                        base_output_kv_cache=base_output_kv_cache,
                        source_output_kv_cache=source_output_kv_cache,
                        source_model_idx=source_model_idx,
                        new_length=new_length,
                        bank_selection=None,
                        default_bank_idx=1,
                    )
                    pooled_feature = self.router.extract_query_feature_from_cache_pair(
                        bank0_cache,
                        bank1_cache,
                        new_length=new_length,
                    )
                elif getattr(self.router, "feature_source", "kv") == "shared_latent_receiver":
                    pooled_feature = self.router.extract_query_feature_from_fused_cache(
                        base_output_kv_cache,
                        new_length=new_length,
                    )
                else:
                    candidate_fused_cache = self._apply_projector_bank_to_cache(
                        base_output_kv_cache=base_output_kv_cache,
                        source_output_kv_cache=source_output_kv_cache,
                        source_model_idx=source_model_idx,
                        new_length=new_length,
                        bank_selection=None,
                        default_bank_idx=0,
                    )
                    if getattr(self.router, "feature_source", "kv") == "shared_latent_pair":
                        pooled_feature = self.router.extract_query_feature_from_cache_pair(
                            base_output_kv_cache,
                            candidate_fused_cache,
                            new_length=new_length,
                        )
                    else:
                        pooled_feature = self.router.extract_query_feature_from_fused_cache(
                            candidate_fused_cache,
                            new_length=new_length,
                        )
            routing = self._run_router(
                base_output_kv_cache=base_output_kv_cache,
                source_output_kv_cache=source_output_kv_cache,
                pooled_feature=pooled_feature,
            )
            fused_kv_cache = self._apply_projector_bank_to_cache(
                base_output_kv_cache=base_output_kv_cache,
                source_output_kv_cache=source_output_kv_cache,
                source_model_idx=source_model_idx,
                new_length=new_length,
                bank_selection=routing["selected_bank"],
            )
            self.last_entropy_gate_state = {
                "mode": "routing",
                "base_entropy": curr_base_H,
                "base_top_token_id": curr_base_top_token_id,
                "sources": {
                    source_model_idx: {
                        "source_entropy": curr_source_H,
                        "source_top_token_id": curr_source_top_token_id,
                        "allow_fusion": routing["should_fuse"].detach().cpu(),
                    }
                },
            }
        else:
            allow_entropy_fusion = self._should_fuse_source(curr_base_H, curr_source_H)
            self._record_entropy_gate_decision(
                allow_fusion=allow_entropy_fusion,
                gate_was_checked=(
                    self._use_entropy_gate()
                    and curr_base_H is not None
                    and curr_source_H is not None
                ),
            )

            self.last_entropy_gate_state = {
                "mode": "register_hooks",
                "base_entropy": curr_base_H,
                "base_top_token_id": curr_base_top_token_id,
                "sources": {
                    source_model_idx: {
                        "source_entropy": curr_source_H,
                        "source_top_token_id": curr_source_top_token_id,
                        "relative_entropy_gap": (
                            None
                            if curr_base_H is None or curr_source_H is None
                            else curr_source_H - curr_base_H
                        ),
                        "allow_fusion": allow_entropy_fusion,
                    }
                },
            }

            if allow_entropy_fusion:
                fused_kv_cache = self._apply_projector_bank_to_cache(
                    base_output_kv_cache=base_output_kv_cache,
                    source_output_kv_cache=source_output_kv_cache,
                    source_model_idx=source_model_idx,
                    new_length=new_length,
                    default_bank_idx=0,
                )

        # Monkeypatch attention forward so the modified KV is used in *this* forward pass.
        hook_handlers = []  # list of (attn_module, orig_forward)
        for i in range(self.model_list[self.base_model_idx].config.num_hidden_layers):
            attn = self.model_list[self.base_model_idx].model.layers[i].self_attn
            new_k = fused_kv_cache.key_cache[i][:, :, -new_length:, :]
            new_v = fused_kv_cache.value_cache[i][:, :, -new_length:, :]
            orig_forward = RosettaModel._monkeypatch_qwen3_attention_forward(attn, new_k, new_v)
            hook_handlers.append((attn, orig_forward))

        return hook_handlers, base_output_kv_cache, source_output_kv_cache
    
    def remove_hooks(self, hook_handlers):
        # Restore monkeypatched forwards
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
        # **kwargs: Unpack[KwargsForCausalLM],
        *args,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass
        
        kv_cache_index: List of tensors with shape (B, sec_seq_len, 2).
            The first element [i][0][0][0] controls sharer selection:
            - -1: No projection (receiver only, skip all sharers)
            - 0: Self projection (receiver projects from itself) - not currently used
            - >0: Bitmask selecting sharers (1 (001)=sharer1, 2 (010)=sharer2, 3 (011)=both, 7 (111)=all three)
            Each bit corresponds to a sharer: bit i selects sharer at model_list[i+1].
        
        input_ids: If LongTensor, same input for all models. If List, per-model inputs.
        """

        # Handle different input formats: if input_ids is a list, use per-model inputs
        if isinstance(input_ids, list):
            # Use list format: different input_ids and attention_mask for each model
            base_input_ids = input_ids[self.base_model_idx] if input_ids is not None else None
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
            _, seqlen = base_input_ids.size() if base_input_ids is not None else (0, 0)
        else:
            # Use tensor format: same input_ids and attention_mask for all models (backward compatibility)
            base_input_ids = input_ids
            base_attention_mask = attention_mask
            _, seqlen = input_ids.size() if input_ids is not None else (0, 0)

        if seqlen > 1:
            self.kv_cache_dict = dict()
            self.last_routing_state = {}
            self._active_routing_bank_idx = None
            
        num_sections = len(kv_cache_index) if kv_cache_index is not None else 1

        section_lengths = [kv_cache_index[i].shape[1] for i in range(num_sections)] if kv_cache_index is not None else [seqlen]
        section_starts = [0]
        for l in section_lengths:
            section_starts.append(section_starts[-1] + l)
        
        curr_base_kv_cache = hybrid_to_dynamic(past_key_values)
        routed_source_model_idx = None
        router_needs_hidden = self._router_needs_hidden_states()
        router_uses_hidden = self._router_uses_hidden_features()
        router_uses_projector_in = self._router_uses_projector_in_features()
        router_uses_postfusion_delta = self._router_uses_postfusion_delta_features()
        router_uses_postfusion_probe_logits = (
            self._router_uses_postfusion_probe_logits_features()
        )
        router_uses_postfusion_option_logits = (
            self._router_uses_postfusion_option_logits_features()
        )
        router_uses_shared_latent_fusion = (
            self._router_uses_shared_latent_fusion_features()
        )
        if len(self.model_list) > 1:
            routed_source_model_idx = (
                self._routing_source_model_idx() if self.routing_enabled else 1
            )

        for i in range(num_sections):
            start = section_starts[i]
            end = section_starts[i + 1]
            prefill_input_ids = base_input_ids[:, start:end] if base_input_ids is not None else None
            prefill_attention_mask = base_attention_mask[:, :end] if base_attention_mask is not None else None
            prefill_position_ids = position_ids[:, start:end] if position_ids is not None else None
            prefill_labels = labels[:, start:end] if labels is not None else None

            if i == num_sections - 1:
                use_last_section_fusion = self._should_use_last_section_fusion(
                    seqlen=seqlen,
                    source_model_idx=routed_source_model_idx,
                )

                if use_last_section_fusion:
                    hook_handlers, base_output_kv_cache, source_output_kv_cache = self.register_hooks(
                        input_ids=prefill_input_ids,
                        attention_mask=prefill_attention_mask,
                        position_ids=prefill_position_ids,
                        base_kv_cache=self.kv_cache_dict[self.base_model_idx][self.base_model_idx],
                        source_model_idx=routed_source_model_idx,
                        source_kv_cache=self.kv_cache_dict[self.base_model_idx][routed_source_model_idx],
                    )

                # calculate target model kvcache
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
                    **kwargs
                )

                if use_last_section_fusion:
                    self.last_entropy_gate_state["post_fusion_entropy"] = self._compute_entropy(output.logits)
                    self.remove_hooks(hook_handlers)

                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(output.past_key_values)
                    self.kv_cache_dict[self.base_model_idx][routed_source_model_idx] = clone_kv_cache(source_output_kv_cache)
                else:
                    if self.base_model_idx not in self.kv_cache_dict:
                        self.kv_cache_dict[self.base_model_idx] = {}
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(output.past_key_values)

            else:

                output = self.model_list[self.base_model_idx].forward(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask, 
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,
                    use_cache=use_cache, 
                    output_attentions=output_attentions,
                    output_hidden_states=(output_hidden_states or router_needs_hidden),
                    *args,
                    **kwargs
                )
                base_last_hidden_for_routing = (
                    output.hidden_states[-1][:, -1, :].detach()
                    if router_needs_hidden
                    else None
                )

                if self.base_model_idx not in self.kv_cache_dict:
                    self.kv_cache_dict[self.base_model_idx] = {}
                if self.base_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = None
                self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(output.past_key_values)

                curr_base_kv_cache: DynamicCache = hybrid_to_dynamic(output.past_key_values)
                entropy_override = self._prefill_entropy_gate_override
                curr_base_H = (
                    entropy_override["base_entropy"]
                    if entropy_override is not None
                    else self._compute_entropy(output.logits)
                )
                curr_base_top_token_id = self._get_top_token_id(output.logits)

                # Pre-check: skip source models whose projectors all have closed gates
                if self.routing_enabled:
                    active_source_models = {self._routing_source_model_idx()}
                else:
                    active_source_models = self._get_active_source_models()
                entropy_gate_sources = {}
                self.last_entropy_gate_state = {
                    "mode": "forward",
                    "entropy_source": (
                        "full_prompt_terminal"
                        if entropy_override is not None
                        else "section_terminal"
                    ),
                    "section_idx": i,
                    "base_entropy": curr_base_H,
                    "base_top_token_id": curr_base_top_token_id,
                    "sources": {},
                }
                section_source_last_hidden: Dict[int, torch.Tensor] = {}
                
                for source_model_idx in range(1, len(self.model_list)):
                    if source_model_idx not in active_source_models:
                        continue
                    if self.base_model_idx not in self.kv_cache_dict:
                        self.kv_cache_dict[self.base_model_idx] = {}
                    if source_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                        self.kv_cache_dict[self.base_model_idx][source_model_idx] = None

                    # Get model-specific input_ids and attention_mask
                    if isinstance(input_ids, list):
                        source_input_ids = input_ids[source_model_idx]
                        source_attention_mask = attention_mask[source_model_idx] if attention_mask is not None else None
                        source_prefill_input_ids = source_input_ids[:, start:end] if source_input_ids is not None else None
                        source_prefill_attention_mask = source_attention_mask[:, :end] if source_attention_mask is not None else None
                    else:
                        # Backward compatibility: use same input for all models
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
                                output_hidden_states=router_needs_hidden,
                                return_dict=True,
                            )
                            curr_source_kv_cache = hybrid_to_dynamic(out.past_key_values)
                            curr_source_H = self._compute_entropy(out.logits)
                            curr_source_top_token_id = self._get_top_token_id(out.logits)
                            if router_needs_hidden:
                                section_source_last_hidden[source_model_idx] = (
                                    out.hidden_states[-1][:, -1, :].detach()
                                )
                    finally:
                        if had_gc:
                            model.gradient_checkpointing_enable()
                        if was_training:
                            model.train()
                            
                    override_source = None
                    if entropy_override is not None:
                        override_source = entropy_override["sources"].get(source_model_idx)

                    if override_source is not None:
                        source_entropy_for_gate = override_source["source_entropy"]
                    else:
                        source_entropy_for_gate = curr_source_H

                    # Prefill always applies source fusion; decode-time gating remains
                    # in register_hooks() when include_response=True.
                    allow_entropy_fusion = True
                    entropy_gate_sources[source_model_idx] = allow_entropy_fusion
                    self.last_entropy_gate_state["sources"][source_model_idx] = {
                        "source_entropy": source_entropy_for_gate,
                        "source_top_token_id": curr_source_top_token_id,
                        "relative_entropy_gap": (
                            None
                            if curr_base_H is None or source_entropy_for_gate is None
                            else source_entropy_for_gate - curr_base_H
                        ),
                        "allow_fusion": allow_entropy_fusion,
                    }
                    
                    curr_source_kv_cache = hybrid_to_dynamic(curr_source_kv_cache)
                    self.kv_cache_dict[self.base_model_idx][source_model_idx] = clone_kv_cache(curr_source_kv_cache)

                # Routing path: apply selected bank on each prefill section so the
                # execution path matches non-routing all-fusion timing.
                if self.routing_enabled and routed_source_model_idx is not None:
                    section_sharer_mask = None
                    if kv_cache_index is not None:
                        try:
                            section_sharer_mask = kv_cache_index[i][0][0][0].item()
                        except Exception:
                            section_sharer_mask = None

                    if section_sharer_mask is None or section_sharer_mask > 0:
                        source_cache_for_routing = self.kv_cache_dict[self.base_model_idx].get(
                            routed_source_model_idx
                        )
                        if source_cache_for_routing is not None:
                            pooled_feature = None
                            if router_uses_hidden:
                                source_last_hidden_for_routing = section_source_last_hidden.get(
                                    routed_source_model_idx
                                )
                                if (
                                    base_last_hidden_for_routing is None
                                    or source_last_hidden_for_routing is None
                                ):
                                    raise RuntimeError(
                                        "Routing(hidden) requested but terminal hidden states "
                                        "were not captured for base/source models."
                                    )
                                pooled_feature = self.router.extract_query_feature_from_hidden(
                                    base_last_hidden_for_routing,
                                    source_last_hidden_for_routing,
                                )
                            elif router_uses_projector_in:
                                pooled_feature = self._extract_projector_in_router_feature(
                                    base_output_kv_cache=curr_base_kv_cache,
                                    source_output_kv_cache=source_cache_for_routing,
                                    source_model_idx=routed_source_model_idx,
                                    new_length=end - start,
                                )
                            elif router_uses_postfusion_delta:
                                pooled_feature = self._extract_postfusion_router_feature(
                                    base_output_kv_cache=curr_base_kv_cache,
                                    source_output_kv_cache=source_cache_for_routing,
                                    source_model_idx=routed_source_model_idx,
                                    new_length=end - start,
                                )
                            elif router_uses_postfusion_probe_logits:
                                pooled_feature = self._extract_postfusion_probe_logits_router_feature(
                                    base_output_kv_cache=curr_base_kv_cache,
                                    source_output_kv_cache=source_cache_for_routing,
                                    source_model_idx=routed_source_model_idx,
                                    new_length=end - start,
                                    prefill_logits=output.logits,
                                    attention_mask=prefill_attention_mask,
                                    position_ids=prefill_position_ids,
                                )
                            elif router_uses_postfusion_option_logits:
                                pooled_feature = self._extract_postfusion_option_logits_router_feature(
                                    base_output_kv_cache=curr_base_kv_cache,
                                    source_output_kv_cache=source_cache_for_routing,
                                    source_model_idx=routed_source_model_idx,
                                    new_length=end - start,
                                    prefill_logits=output.logits,
                                    attention_mask=prefill_attention_mask,
                                    position_ids=prefill_position_ids,
                                    base_last_hidden=base_last_hidden_for_routing,
                                    source_last_hidden=section_source_last_hidden.get(
                                        routed_source_model_idx
                                    ),
                                )
                            elif router_uses_shared_latent_fusion:
                                if (
                                    getattr(self.router, "no_skip_routing", False)
                                    and getattr(self.router, "feature_source", "kv") == "shared_latent_pair"
                                    and len(self.projector_bank_dicts) >= 2
                                ):
                                    bank0_cache = self._apply_projector_bank_to_cache(
                                        base_output_kv_cache=curr_base_kv_cache,
                                        source_output_kv_cache=source_cache_for_routing,
                                        source_model_idx=routed_source_model_idx,
                                        new_length=end - start,
                                        bank_selection=None,
                                        default_bank_idx=0,
                                    )
                                    bank1_cache = self._apply_projector_bank_to_cache(
                                        base_output_kv_cache=curr_base_kv_cache,
                                        source_output_kv_cache=source_cache_for_routing,
                                        source_model_idx=routed_source_model_idx,
                                        new_length=end - start,
                                        bank_selection=None,
                                        default_bank_idx=1,
                                    )
                                    pooled_feature = self.router.extract_query_feature_from_cache_pair(
                                        bank0_cache,
                                        bank1_cache,
                                        new_length=end - start,
                                    )
                                elif getattr(self.router, "feature_source", "kv") == "shared_latent_receiver":
                                    pooled_feature = self.router.extract_query_feature_from_fused_cache(
                                        curr_base_kv_cache,
                                        new_length=end - start,
                                    )
                                else:
                                    candidate_fused_cache = self._apply_projector_bank_to_cache(
                                        base_output_kv_cache=curr_base_kv_cache,
                                        source_output_kv_cache=source_cache_for_routing,
                                        source_model_idx=routed_source_model_idx,
                                        new_length=end - start,
                                        bank_selection=None,
                                        default_bank_idx=0,
                                    )
                                    if getattr(self.router, "feature_source", "kv") == "shared_latent_pair":
                                        pooled_feature = self.router.extract_query_feature_from_cache_pair(
                                            curr_base_kv_cache,
                                            candidate_fused_cache,
                                            new_length=end - start,
                                        )
                                    else:
                                        pooled_feature = self.router.extract_query_feature_from_fused_cache(
                                            candidate_fused_cache,
                                            new_length=end - start,
                                        )
                            routing = self._run_router(
                                base_output_kv_cache=curr_base_kv_cache,
                                source_output_kv_cache=source_cache_for_routing,
                                pooled_feature=pooled_feature,
                            )
                            curr_base_kv_cache = self._apply_projector_bank_to_cache(
                                base_output_kv_cache=curr_base_kv_cache,
                                source_output_kv_cache=source_cache_for_routing,
                                source_model_idx=routed_source_model_idx,
                                new_length=end - start,
                                bank_selection=routing["selected_bank"],
                            )

                # calculate source model kvcache and apply projections
                if (not self.routing_enabled) and self.base_model_idx in self.projector_dict:
                    # Iterate over all source models in projector_dict
                    sharer_mask = kv_cache_index[i][0][0][0].item()
                    if sharer_mask > 0:
                        base_cache = clone_kv_cache(curr_base_kv_cache)

                        # For parallel mode, accumulate residuals for each target layer
                        parallel_delta_cache = {} if self.multi_source_fusion_mode == "parallel" else None
                        
                        # Compute and apply projections (shared logic for both modes)
                        for source_model_idx in self.projector_dict[self.base_model_idx].keys():
                            # Check if this sharer is selected: bit (source_model_idx - 1)
                            if not (sharer_mask & (1 << (source_model_idx - 1))):
                                continue
                            if not entropy_gate_sources.get(source_model_idx, True):
                                continue
                            if self.multi_source_fusion_mode == "sequential":
                                base_cache_ref = curr_base_kv_cache
                            else:
                                # Parallel: always project from the clean cloned base cache
                                base_cache_ref = base_cache

                            for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
                                # Get base KV cache slice for projection
                                base_key_cache, base_value_cache = base_cache_ref[target_layer_idx]
                                new_base_key_cache = base_key_cache[:, :, start:end, :]
                                new_base_value_cache = base_value_cache[:, :, start:end, :]
                                new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

                                pair_list = entry

                                projected_kv_list = []
                                source_kv_list = []
                                for source_model_layer_idx, projector_idx in pair_list:
                                    if not self._is_gate_open(self.projector_list[projector_idx]):
                                        continue
                                    source_key_cache, source_value_cache = self.kv_cache_dict[self.base_model_idx][source_model_idx][source_model_layer_idx]
                                    new_source_key_cache = source_key_cache[:, :, start:end, :]
                                    new_source_value_cache = source_value_cache[:, :, start:end, :]
                                    new_source_kv_cache = (new_source_key_cache, new_source_value_cache)
                                    projected_key, projected_value = self.projector_list[projector_idx].forward(
                                        new_source_kv_cache,
                                        new_base_kv_cache
                                    )
                                    projected_kv_list.append((projected_key, projected_value))
                                    source_kv_list.append(new_source_kv_cache)

                                if not projected_kv_list:
                                    continue

                                # Use first projector result
                                agg_key, agg_value = projected_kv_list[0]

                                # Collect or apply projection based on mode
                                if self.multi_source_fusion_mode == "sequential":
                                    # Sequential: apply immediately so next source sees updated cache
                                    curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = agg_key
                                    curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = agg_value
                                else:
                                    # Parallel: accumulate residuals (agg - base) for this target layer
                                    if target_layer_idx not in parallel_delta_cache:
                                        parallel_delta_cache[target_layer_idx] = (
                                            torch.zeros_like(new_base_key_cache),
                                            torch.zeros_like(new_base_value_cache),
                                        )
                                    delta_key, delta_value = parallel_delta_cache[target_layer_idx]
                                    delta_key = delta_key + (agg_key - new_base_key_cache)
                                    delta_value = delta_value + (agg_value - new_base_value_cache)
                                    parallel_delta_cache[target_layer_idx] = (delta_key, delta_value)

                        # For parallel mode, apply all accumulated residuals in one shot
                        if self.multi_source_fusion_mode == "parallel":
                            for target_layer_idx, (delta_key, delta_value) in parallel_delta_cache.items():
                                base_key_cache, base_value_cache = base_cache[target_layer_idx]
                                base_key_slice = base_key_cache[:, :, start:end, :]
                                base_value_slice = base_value_cache[:, :, start:end, :]
                                curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = base_key_slice + delta_key
                                curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = base_value_slice + delta_value

                self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(curr_base_kv_cache)
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
        streamer = None,
        *args,
        **kwargs,
    ):
        """
        New generation loop without using the base model's generate.
        - Uses this module's forward for prefill and per-token decode.
        - Samples tokens via rosetta.model.sampling.sample_token.
        Returns a tensor of shape [batch, prompt_len + generated_len] for the base model stream.
        """

        self.kv_cache_dict = dict()

        # Derive number of tokens to generate
        # If max_new_tokens not provided, infer from max_length
        if isinstance(input_ids, list):
            base_input_ids_for_len = input_ids[self.base_model_idx]
        else:
            base_input_ids_for_len = input_ids
        prompt_len = base_input_ids_for_len.size(1)

        # Default eos/pad from base model tokenizer/config if not provided
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

        # Resolve base inputs
        if isinstance(input_ids, list):
            base_input_ids = input_ids[self.base_model_idx]
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
        else:
            base_input_ids = input_ids
            base_attention_mask = attention_mask

        if base_attention_mask is None:
            base_attention_mask = torch.ones_like(base_input_ids, dtype=torch.long, device=base_input_ids.device)

        batch_size = base_input_ids.size(0)

        # Prefill to build caches and obtain initial logits
        self._reset_entropy_gate_stats()
        self._prefill_entropy_gate_override = self._compute_generate_prefill_entropy_override(
            kv_cache_index=kv_cache_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        try:
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
        finally:
            self._prefill_entropy_gate_override = None

        current_past = prefill_output.past_key_values
        all_input_ids = base_input_ids
        current_attention_mask = base_attention_mask

        # Initialize streamer with prompt if provided
        if streamer is not None:
            streamer.put(base_input_ids)

        # EOS handling setup
        eos_set = None
        if eos_token_id is not None:
            eos_set = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
        finished = torch.zeros(batch_size, dtype=torch.bool, device=all_input_ids.device)

        # Start from last prefill logits
        last_logits = prefill_output.logits[:, -1, :]

        # Determine sampling mode
        if do_sample is None:
            do_sample = False
        effective_temperature = temperature if do_sample else 0.0

        # Optional scores collection
        collect_scores = bool(return_dict_in_generate) and bool(output_scores)
        scores = []

        for _ in range(max_new_tokens):
            if collect_scores:
                scores.append(last_logits)
            # Apply repetition/presence/frequency penalties to logits before sampling
            adjusted_logits = last_logits
            if (
                (repetition_penalty is not None and repetition_penalty != 1.0) or
                (presence_penalty is not None and presence_penalty != 0.0) or
                (frequency_penalty is not None and frequency_penalty != 0.0)
            ):
                adjusted_logits = last_logits.clone()
                vocab_size = adjusted_logits.size(-1)
                # Per-batch penalty application for clarity and correctness
                for b in range(batch_size):
                    seq_tokens = all_input_ids[b]
                    if seq_tokens.numel() == 0:
                        continue
                    counts = torch.bincount(seq_tokens, minlength=vocab_size)
                    if counts.dtype != torch.float32 and counts.dtype != torch.float64:
                        counts = counts.to(adjusted_logits.dtype)
                    # Presence penalty: penalize any token that has appeared
                    if presence_penalty and presence_penalty != 0.0:
                        presence_mask = counts > 0
                        if presence_mask.any():
                            adjusted_logits[b, presence_mask] = adjusted_logits[b, presence_mask] - presence_penalty
                    # Frequency penalty: penalize proportionally to frequency
                    if frequency_penalty and frequency_penalty != 0.0:
                        adjusted_logits[b] = adjusted_logits[b] - frequency_penalty * counts
                    # Repetition penalty (HF-style): divide positive logits, multiply negative logits
                    if repetition_penalty and repetition_penalty != 1.0:
                        rep_mask = counts > 0
                        if rep_mask.any():
                            pos_mask = rep_mask & (adjusted_logits[b] > 0)
                            neg_mask = rep_mask & ~pos_mask
                            if pos_mask.any():
                                adjusted_logits[b, pos_mask] = adjusted_logits[b, pos_mask] / repetition_penalty
                            if neg_mask.any():
                                adjusted_logits[b, neg_mask] = adjusted_logits[b, neg_mask] * repetition_penalty

            # Sample next token
            next_token = sample_token(adjusted_logits, temperature=effective_temperature, top_p=top_p, top_k=top_k)
            if not isinstance(next_token, torch.Tensor):
                next_token = torch.tensor([next_token], device=all_input_ids.device, dtype=torch.long).repeat(batch_size)

            # Apply EOS logic
            if eos_set is not None:
                just_finished = torch.zeros_like(finished)
                for eid in eos_set:
                    just_finished |= (next_token == eid)
                finished = finished | just_finished
                if pad_token_id is not None:
                    next_token = torch.where(
                        finished,
                        torch.tensor(pad_token_id, device=next_token.device, dtype=next_token.dtype),
                        next_token,
                    )

            # Append sampled token
            next_token_unsqueezed = next_token.unsqueeze(1)
            all_input_ids = torch.cat([all_input_ids, next_token_unsqueezed], dim=1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones((batch_size, 1), device=current_attention_mask.device, dtype=current_attention_mask.dtype),
                ],
                dim=1,
            )

            # Stream the new token if streamer provided
            if streamer is not None:
                streamer.put(next_token_unsqueezed)

            # Early stop if all sequences finished
            if eos_set is not None and torch.all(finished):
                break

            # Decode one step using cached states; pass base-stream tensors
            kv_cache_index = [torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(all_input_ids.device)]

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
            if self.update_decode_past:
                current_past = decode_output.past_key_values
            last_logits = decode_output.logits[:, -1, :]

        # End streaming if streamer provided
        if streamer is not None:
            streamer.end()

        # Return style compatible with HF generate
        if return_dict_in_generate:
            if GreedySearchDecoderOnlyOutput is not None and SampleDecoderOnlyOutput is not None:
                if do_sample:
                    return SampleDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
                else:
                    return GreedySearchDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
            # Fallback to generic ModelOutput
            result = {"sequences": all_input_ids}
            if collect_scores:
                result["scores"] = scores
            return ModelOutput(**result)
        return all_input_ids
