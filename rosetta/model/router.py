"""
Router module for query-level C2C bank selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache

from rosetta.utils.registry import (
    capture_init_args,
    create_registry,
    load_object,
    save_object,
)


ROUTER_REGISTRY, register_router, get_router_class = create_registry(
    "router",
    case_insensitive=True,
)


def _ensure_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    if not isinstance(cache, DynamicCache):
        raise TypeError(f"Expected DynamicCache, got {type(cache)}")
    return cache


def _cache_key_layers(cache: DynamicCache):
    if hasattr(cache, "key_cache"):
        return cache.key_cache
    return [layer.keys for layer in cache.layers]


def _cache_value_layers(cache: DynamicCache):
    if hasattr(cache, "value_cache"):
        return cache.value_cache
    return [layer.values for layer in cache.layers]


@dataclass
class RouterOutput:
    pooled_feature: Tensor
    action_logits: Tensor

    @property
    def action_probabilities(self) -> Tensor:
        probs = torch.softmax(self.action_logits.float(), dim=-1)
        return probs.to(dtype=self.action_logits.dtype)

    @property
    def skip_probability(self) -> Tensor:
        return self.action_probabilities[:, 0]

    @property
    def fuse_probability(self) -> Tensor:
        return 1.0 - self.skip_probability

    @property
    def selection_logits(self) -> Tensor:
        # Bank logits only (exclude class-0 skip action).
        return self.action_logits[:, 1:]

    @property
    def binary_logits(self) -> Tensor:
        # Log-odds of fuse-vs-skip induced by multiclass logits.
        fuse_logit = torch.logsumexp(self.selection_logits, dim=-1)
        skip_logit = self.action_logits[:, 0]
        return fuse_logit - skip_logit


@register_router
@capture_init_args
class SimpleKVRouter(nn.Module):
    """
    Lightweight router that turns receiver/sharer KV caches into a query-level
    feature and predicts one multiclass action:
      action 0 -> skip
      action i -> fuse with bank (i - 1), for i in [1, num_banks].

    The built-in KV feature extractor keeps hidden structure while staying query-level:
    1) mean over tokens (N) per layer,
    2) mean over layers (L),
    3) flatten (H, D) into a hidden-size-like vector.

    Final KV/hidden feature = [base, source]. Other feature_source values, such
    as projector_in/projector_in_stats/projector_in_pooled/projector_in_binned,
    must provide pooled_feature directly.
    """

    def __init__(
        self,
        num_banks: int,
        input_dim: int = 18,
        hidden_dim: int = 128,
        token_mlp_layers: int = 2,
        dropout: float = 0.0,
        input_layer_norm: bool = False,
        input_standardization: bool = False,
        eps: float = 1e-6,
        feature_source: str = "kv",
        option_token_ids: Optional[Sequence[int]] = None,
        option_response_token_ids: Optional[Sequence[int]] = None,
        option_response_text: Optional[str] = None,
        decision_policy: str = "argmax",
        skip_threshold: float = 0.5,
        no_skip_routing: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if num_banks <= 0:
            raise ValueError(f"num_banks must be positive, got {num_banks}")
        if token_mlp_layers < 0:
            raise ValueError(
                f"token_mlp_layers must be non-negative, got {token_mlp_layers}"
            )
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if feature_source not in {
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
            "hybrid_projector_hidden_binned",
        }:
            raise ValueError(
                "feature_source must be one of "
                "['kv', 'hidden', 'hidden_binned', 'projector_in', "
                "'projector_in_stats', 'projector_in_pooled', 'projector_in_binned', "
                "'postfusion_delta_stats', 'postfusion_probe_logits', "
                "'postfusion_option_logits', 'postfusion_option_logits_hidden_binned', "
                "'hybrid_projector_hidden_binned'], "
                f"got {feature_source}"
            )
        if feature_source == "hidden_binned" and input_dim % 4 != 0:
            raise ValueError(
                "hidden_binned expects input_dim divisible by 4 "
                "([base mean/std bins, source mean/std bins]), "
                f"got input_dim={input_dim}"
            )
        if decision_policy not in {"argmax", "skip_threshold", "option_confidence", "score_threshold"}:
            raise ValueError(
                "decision_policy must be one of "
                "['argmax', 'skip_threshold', 'option_confidence', 'score_threshold'], "
                f"got {decision_policy}"
            )
        if decision_policy != "score_threshold" and not 0.0 <= float(skip_threshold) <= 1.0:
            raise ValueError(f"skip_threshold must be in [0, 1], got {skip_threshold}")

        self.num_banks = num_banks
        self.no_skip_routing = bool(no_skip_routing)
        self.num_actions = num_banks if self.no_skip_routing else num_banks + 1
        self.input_dim = input_dim
        # Backward-compatible alias used by older tests/utilities.
        self.token_feature_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_layer_norm = input_layer_norm
        self.input_standardization = input_standardization
        self.eps = eps
        self.feature_source = feature_source
        self.option_token_ids = (
            [int(token_id) for token_id in option_token_ids]
            if option_token_ids is not None
            else None
        )
        self.option_response_token_ids = (
            [int(token_id) for token_id in option_response_token_ids]
            if option_response_token_ids is not None
            else None
        )
        self.option_response_text = option_response_text
        self.decision_policy = decision_policy
        self.skip_threshold = float(skip_threshold)

        self.register_buffer("input_mean", torch.zeros(input_dim, dtype=dtype), persistent=True)
        self.register_buffer("input_std", torch.ones(input_dim, dtype=dtype), persistent=True)
        self.input_norm = nn.LayerNorm(input_dim, dtype=dtype) if input_layer_norm else nn.Identity()
        layers = []
        in_dim = self.input_dim
        for _ in range(token_mlp_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, dtype=dtype))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.feature_encoder = nn.Sequential(*layers)
        self.action_head = nn.Linear(in_dim, self.num_actions, dtype=dtype)

    def set_input_standardization(self, mean: Tensor, std: Tensor) -> None:
        if mean.numel() != self.input_dim or std.numel() != self.input_dim:
            raise ValueError(
                "Input standardization stats must match router input_dim: "
                f"mean={mean.numel()} std={std.numel()} input_dim={self.input_dim}"
            )
        device = self.input_mean.device
        dtype = self.input_mean.dtype
        self.input_mean.copy_(mean.detach().to(device=device, dtype=dtype).view(-1))
        self.input_std.copy_(
            std.detach().to(device=device, dtype=dtype).view(-1).clamp_min(self.eps)
        )

    def _mean_token_hidden(self, key: Tensor, value: Tensor) -> Tensor:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "Expected cache tensors with shape (B, H, N, D), "
                f"got key={tuple(key.shape)} value={tuple(value.shape)}"
            )
        seq_len = min(key.size(2), value.size(2))
        if seq_len <= 0:
            raise ValueError("Cannot build routing feature from an empty cache slice")

        key_mean = key[:, :, :seq_len, :].float().mean(dim=2)
        value_mean = value[:, :, :seq_len, :].float().mean(dim=2)
        return 0.5 * (key_mean + value_mean)

    def _layer_mean_hidden(self, keys, values, num_layers: int) -> Tensor:
        pooled_layers = []
        for layer_idx in range(num_layers):
            pooled_layers.append(
                self._mean_token_hidden(
                    keys[layer_idx],
                    values[layer_idx],
                )
            )
        return torch.stack(pooled_layers, dim=1).mean(dim=1)

    def _build_feature_from_hidden(self, base_hidden: Tensor, source_hidden: Tensor) -> Tensor:
        if self.feature_source == "hidden_binned":
            bins_per_model = self.input_dim // 4
            return torch.cat(
                [
                    self._bin_hidden_vector(base_hidden, bins_per_model),
                    self._bin_hidden_vector(source_hidden, bins_per_model),
                ],
                dim=-1,
            )

        base_flat = base_hidden.flatten(start_dim=1)
        source_flat = source_hidden.flatten(start_dim=1)
        if base_flat.size(-1) != source_flat.size(-1):
            common_dim = min(base_flat.size(-1), source_flat.size(-1))
            if common_dim <= 0:
                raise ValueError(
                    f"Invalid flattened dims: base={base_flat.size(-1)} "
                    f"source={source_flat.size(-1)}"
                )
            base_flat = base_flat[:, :common_dim]
            source_flat = source_flat[:, :common_dim]
        return torch.cat([base_flat, source_flat], dim=-1)

    def _bin_hidden_vector(self, hidden: Tensor, bins: int) -> Tensor:
        if hidden.ndim != 2:
            raise ValueError(f"Expected hidden with shape (B, D), got {tuple(hidden.shape)}")
        hidden_1d = hidden.unsqueeze(1)
        bin_mean = F.adaptive_avg_pool1d(hidden_1d, bins).squeeze(1)
        bin_second_moment = F.adaptive_avg_pool1d(hidden_1d.square(), bins).squeeze(1)
        bin_std = (bin_second_moment - bin_mean.square()).clamp_min(0.0).sqrt()
        return torch.cat([bin_mean, bin_std], dim=-1)

    def extract_query_feature(
        self,
        base_cache: DynamicCache,
        source_cache: DynamicCache,
    ) -> Tensor:
        base_cache = _ensure_dynamic_cache(base_cache)
        source_cache = _ensure_dynamic_cache(source_cache)
        base_keys = _cache_key_layers(base_cache)
        base_values = _cache_value_layers(base_cache)
        source_keys = _cache_key_layers(source_cache)
        source_values = _cache_value_layers(source_cache)

        if len(base_keys) == 0 or len(source_keys) == 0:
            raise ValueError("Routing requires non-empty base/source caches")

        num_layers = min(len(base_keys), len(source_keys))
        if num_layers <= 0:
            raise ValueError("Routing requires at least one shared layer")

        base_hidden = self._layer_mean_hidden(base_keys, base_values, num_layers)
        source_hidden = self._layer_mean_hidden(source_keys, source_values, num_layers)
        return self._build_feature_from_hidden(base_hidden, source_hidden)

    def extract_query_feature_from_hidden(
        self,
        base_last_hidden: Tensor,
        source_last_hidden: Tensor,
    ) -> Tensor:
        if base_last_hidden.ndim == 1:
            base_last_hidden = base_last_hidden.unsqueeze(0)
        if source_last_hidden.ndim == 1:
            source_last_hidden = source_last_hidden.unsqueeze(0)
        if base_last_hidden.ndim != 2 or source_last_hidden.ndim != 2:
            raise ValueError(
                "Expected hidden states with shape (B, D), "
                f"got base={tuple(base_last_hidden.shape)} "
                f"source={tuple(source_last_hidden.shape)}"
            )
        return self._build_feature_from_hidden(base_last_hidden, source_last_hidden)

    def forward_features(self, pooled_feature: Tensor) -> RouterOutput:
        pooled_feature = pooled_feature.to(
            device=self.action_head.weight.device,
            dtype=self.action_head.weight.dtype,
        )
        if pooled_feature.size(-1) != self.input_dim:
            raise ValueError(
                "Router input dimension mismatch: "
                f"expected {self.input_dim}, got {pooled_feature.size(-1)}"
            )
        if self.input_standardization:
            pooled_feature = (pooled_feature - self.input_mean) / self.input_std.clamp_min(
                self.eps
            )
        encoded_feature = self.feature_encoder(self.input_norm(pooled_feature))
        action_logits = self.action_head(encoded_feature)
        return RouterOutput(
            pooled_feature=pooled_feature,
            action_logits=action_logits,
        )

    def forward(
        self,
        base_cache: Optional[DynamicCache] = None,
        source_cache: Optional[DynamicCache] = None,
        pooled_feature: Optional[Tensor] = None,
    ) -> RouterOutput:
        if pooled_feature is None:
            if self.feature_source != "kv":
                raise ValueError(
                    f"feature_source={self.feature_source} requires pooled_feature."
                )
            if base_cache is None or source_cache is None:
                raise ValueError(
                    "base_cache/source_cache are required when pooled_feature is not provided."
                )
            pooled_feature = self.extract_query_feature(base_cache, source_cache)
        return self.forward_features(pooled_feature)

    @torch.no_grad()
    def predict(
        self,
        base_cache: Optional[DynamicCache] = None,
        source_cache: Optional[DynamicCache] = None,
        fuse_threshold: float = 0.5,
        pooled_feature: Optional[Tensor] = None,
    ) -> dict:
        output = self.forward(
            base_cache=base_cache,
            source_cache=source_cache,
            pooled_feature=pooled_feature,
        )
        action_logits = output.action_logits
        action_probabilities = output.action_probabilities
        fuse_probability = output.fuse_probability
        # Default inference policy is one-shot multiclass action selection:
        #   action 0 -> skip
        #   action i -> bank (i - 1)
        # For high-precision skip-detector routers, keep fusion as the default
        # and skip only when p(skip) crosses the validation-tuned threshold.
        del fuse_threshold
        if self.decision_policy == "skip_threshold":
            pred_bank = torch.argmax(output.selection_logits, dim=-1)
            selected_action = torch.where(
                output.skip_probability > self.skip_threshold,
                torch.zeros_like(pred_bank),
                pred_bank + 1,
            )
        elif self.decision_policy == "option_confidence":
            if self.feature_source not in {
                "postfusion_option_logits",
                "postfusion_option_logits_hidden_binned",
            }:
                raise ValueError(
                    "option_confidence policy requires postfusion option-logit "
                    f"features, got feature_source={self.feature_source}"
                )
            if not self.option_token_ids:
                raise ValueError("option_confidence policy requires option_token_ids.")
            raw_feature = pooled_feature
            if raw_feature is None:
                raw_feature = output.pooled_feature
                if self.input_standardization:
                    raw_feature = raw_feature * self.input_std.clamp_min(self.eps) + self.input_mean
            raw_feature = raw_feature.to(
                device=action_logits.device,
                dtype=action_logits.dtype,
            )
            num_options = len(self.option_token_ids)
            stats_dim = 5 * num_options + 8
            min_dim = 2 * stats_dim
            if raw_feature.size(-1) < min_dim:
                raise ValueError(
                    "option_confidence policy expected receiver and fused option "
                    f"stats with at least {min_dim} dims, got {raw_feature.size(-1)}"
                )
            receiver_probs = raw_feature[:, 2 * num_options : 3 * num_options]
            fused_probs = raw_feature[
                :, stats_dim + 2 * num_options : stats_dim + 3 * num_options
            ]
            receiver_conf = receiver_probs.max(dim=-1).values
            fused_conf = fused_probs.max(dim=-1).values
            pred_bank = torch.argmax(output.selection_logits, dim=-1)
            selected_action = torch.where(
                (receiver_conf - fused_conf) > self.skip_threshold,
                torch.zeros_like(pred_bank),
                pred_bank + 1,
            )
        elif self.decision_policy == "score_threshold":
            pred_bank = torch.argmax(output.selection_logits, dim=-1)
            selected_action = torch.where(
                output.binary_logits > self.skip_threshold,
                pred_bank + 1,
                torch.zeros_like(pred_bank),
            )
        else:
            selected_action = torch.argmax(action_logits, dim=-1)
        if self.no_skip_routing:
            should_fuse = torch.ones_like(selected_action, dtype=torch.bool)
            selected_bank = selected_action
        else:
            should_fuse = selected_action > 0
            selected_bank = torch.where(
                should_fuse,
                selected_action - 1,
                torch.full_like(selected_action, -1),
            )
        return {
            "pooled_feature": output.pooled_feature,
            "action_logits": action_logits,
            "action_probabilities": action_probabilities,
            "skip_probability": output.skip_probability,
            "binary_logits": output.binary_logits,
            "selection_logits": output.selection_logits,
            "fuse_probability": fuse_probability,
            "should_fuse": should_fuse,
            "selected_bank": selected_bank,
            "selected_action": selected_action,
        }

    def _migrate_legacy_state_dict(self, state_dict: dict) -> tuple[dict, bool]:
        if "action_head.weight" in state_dict and "action_head.bias" in state_dict:
            return state_dict, False

        has_legacy = (
            "binary_head.weight" in state_dict
            and "binary_head.bias" in state_dict
            and "selection_head.weight" in state_dict
            and "selection_head.bias" in state_dict
        )
        if not has_legacy:
            return state_dict, False

        migrated = dict(state_dict)
        device = self.action_head.weight.device
        dtype = self.action_head.weight.dtype

        action_w = torch.zeros_like(self.action_head.weight, device=device, dtype=dtype)
        action_b = torch.zeros_like(self.action_head.bias, device=device, dtype=dtype)

        binary_w = migrated["binary_head.weight"].to(device=device, dtype=dtype)
        binary_b = migrated["binary_head.bias"].to(device=device, dtype=dtype)
        selection_w = migrated["selection_head.weight"].to(device=device, dtype=dtype)
        selection_b = migrated["selection_head.bias"].to(device=device, dtype=dtype)

        banks_to_copy = min(self.num_banks, selection_w.size(0))
        if banks_to_copy > 0:
            if self.num_banks == 1:
                # Exact migration for one-bank legacy routers:
                # old p(fuse)=sigmoid(z_binary), old bank is always 0.
                action_w[1] = binary_w[0]
                action_b[1] = binary_b[0]
            else:
                # Approximate migration for multi-bank legacy routers.
                action_w[1 : 1 + banks_to_copy] = selection_w[:banks_to_copy] + binary_w
                action_b[1 : 1 + banks_to_copy] = selection_b[:banks_to_copy] + binary_b

        migrated["action_head.weight"] = action_w
        migrated["action_head.bias"] = action_b

        for old_key in (
            "binary_head.weight",
            "binary_head.bias",
            "selection_head.weight",
            "selection_head.bias",
        ):
            migrated.pop(old_key, None)
        return migrated, True

    def load_state_dict(self, state_dict, strict: bool = True):
        migrated_state, migrated = self._migrate_legacy_state_dict(state_dict)
        if migrated:
            print(
                "Migrated legacy router checkpoint (binary_head/selection_head) "
                "to multiclass action_head."
            )
        return super().load_state_dict(migrated_state, strict=strict)


class _LayerwiseCrossAttentionBlock(nn.Module):
    """One paper-style translator layer: previous state queries one KV layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_mult: float = 4.0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        ffn_dim = max(hidden_dim, int(hidden_dim * ffn_mult))
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_dim // num_heads)
        self.attn_dropout_p = float(dropout)
        self.query_norm = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.memory_norm = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim, dtype=dtype),
            nn.Dropout(dropout),
        )

    def forward(self, state: Tensor, layer_memory: Tensor) -> Tensor:
        attn_dtype = self.q_proj.weight.dtype
        query = self.query_norm(state).to(dtype=attn_dtype)
        memory = self.memory_norm(layer_memory).to(dtype=attn_dtype)
        batch_size, query_len, _ = query.shape
        key_len = memory.size(1)
        q = self.q_proj(query)
        k = self.k_proj(memory)
        v = self.v_proj(memory)

        def split_heads(x: Tensor, seq_len: int) -> Tensor:
            return (
                x.view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

        q = split_heads(q, query_len)
        k = split_heads(k, key_len)
        v = split_heads(v, key_len)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, self.hidden_dim)
        )
        attn_out = self.o_proj(attn_out)
        state = state + self.dropout(attn_out)
        return state + self.ffn(self.ffn_norm(state))


class LocalToSharedKVEncoder(nn.Module):
    """
    Paper-style local KV -> shared latent encoder.

    The cache is represented as local blocks with shape (B, S, L, D), where D is
    the flattened KV-head dimension. Key and value blocks use separate input and
    output projections, while sharing the layer-wise cross-attention stack.
    """

    def __init__(
        self,
        num_kv_layers: int,
        kv_dim: int,
        encoder_dim: int = 1024,
        shared_dim: int = 1024,
        num_heads: int = 16,
        ffn_mult: float = 4.0,
        dropout: float = 0.0,
        max_encoder_tokens: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if num_kv_layers <= 0:
            raise ValueError(f"num_kv_layers must be positive, got {num_kv_layers}")
        if kv_dim <= 0:
            raise ValueError(f"kv_dim must be positive, got {kv_dim}")
        if encoder_dim <= 0 or shared_dim <= 0:
            raise ValueError("encoder_dim and shared_dim must be positive")
        if max_encoder_tokens is not None and max_encoder_tokens <= 0:
            raise ValueError("--max_encoder_tokens must be positive when set")

        self.num_kv_layers = int(num_kv_layers)
        self.kv_dim = int(kv_dim)
        self.encoder_dim = int(encoder_dim)
        self.shared_dim = int(shared_dim)
        self.max_encoder_tokens = max_encoder_tokens

        self.key_in_norm = nn.LayerNorm(kv_dim, dtype=dtype)
        self.value_in_norm = nn.LayerNorm(kv_dim, dtype=dtype)
        self.key_in = nn.Linear(kv_dim, encoder_dim, dtype=dtype)
        self.value_in = nn.Linear(kv_dim, encoder_dim, dtype=dtype)
        self.input_act = nn.GELU()

        self.blocks = nn.ModuleList(
            [
                _LayerwiseCrossAttentionBlock(
                    hidden_dim=encoder_dim,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    dtype=dtype,
                )
                for _ in range(num_kv_layers)
            ]
        )

        concat_dim = num_kv_layers * encoder_dim
        self.key_out_norm = nn.LayerNorm(concat_dim, dtype=dtype)
        self.value_out_norm = nn.LayerNorm(concat_dim, dtype=dtype)
        self.key_out = nn.Linear(concat_dim, shared_dim, dtype=dtype)
        self.value_out = nn.Linear(concat_dim, shared_dim, dtype=dtype)
        self.output_act = nn.GELU()

    def _cache_to_blocks(
        self,
        fused_cache: DynamicCache,
        *,
        new_length: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        fused_cache = _ensure_dynamic_cache(fused_cache)
        keys = _cache_key_layers(fused_cache)
        values = _cache_value_layers(fused_cache)
        if len(keys) < self.num_kv_layers or len(values) < self.num_kv_layers:
            raise ValueError(
                f"Fused cache has {len(keys)} key layers/{len(values)} value layers, "
                f"but encoder expects {self.num_kv_layers}."
            )

        seq_len = min(
            min(int(layer.size(2)) for layer in keys[: self.num_kv_layers]),
            min(int(layer.size(2)) for layer in values[: self.num_kv_layers]),
        )
        if new_length is not None:
            seq_len = min(seq_len, int(new_length))
        if self.max_encoder_tokens is not None:
            seq_len = min(seq_len, int(self.max_encoder_tokens))
        if seq_len <= 0:
            raise ValueError("Cannot encode an empty fused KV cache slice.")

        key_layers = []
        value_layers = []
        for layer_idx in range(self.num_kv_layers):
            key = keys[layer_idx][:, :, -seq_len:, :]
            value = values[layer_idx][:, :, -seq_len:, :]
            batch, heads, tokens, head_dim = key.shape
            key_flat = key.transpose(1, 2).contiguous().view(batch, tokens, heads * head_dim)
            value_flat = (
                value.transpose(1, 2).contiguous().view(batch, tokens, heads * head_dim)
            )
            if key_flat.size(-1) != self.kv_dim or value_flat.size(-1) != self.kv_dim:
                raise ValueError(
                    "KV flattened dimension mismatch: "
                    f"expected {self.kv_dim}, got key={key_flat.size(-1)} "
                    f"value={value_flat.size(-1)} at layer {layer_idx}"
                )
            key_layers.append(key_flat)
            value_layers.append(value_flat)

        key_block = torch.stack(key_layers, dim=2)
        value_block = torch.stack(value_layers, dim=2)
        dtype = self.key_in.weight.dtype
        device = self.key_in.weight.device
        return (
            key_block.to(device=device, dtype=dtype),
            value_block.to(device=device, dtype=dtype),
        )

    def _encode_one_type(
        self,
        block: Tensor,
        *,
        input_norm: nn.LayerNorm,
        input_proj: nn.Linear,
        output_norm: nn.LayerNorm,
        output_proj: nn.Linear,
    ) -> Tensor:
        hidden = self.input_act(input_proj(input_norm(block)))
        state = hidden[:, :, 0, :]
        layer_outputs = []
        for layer_idx, block_module in enumerate(self.blocks):
            state = block_module(state, hidden[:, :, layer_idx, :])
            layer_outputs.append(state)
        concat = torch.cat(layer_outputs, dim=-1)
        return self.output_act(output_proj(output_norm(concat)))

    def forward(
        self,
        fused_cache: DynamicCache,
        *,
        new_length: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        key_block, value_block = self._cache_to_blocks(
            fused_cache,
            new_length=new_length,
        )
        key_shared = self._encode_one_type(
            key_block,
            input_norm=self.key_in_norm,
            input_proj=self.key_in,
            output_norm=self.key_out_norm,
            output_proj=self.key_out,
        )
        value_shared = self._encode_one_type(
            value_block,
            input_norm=self.value_in_norm,
            input_proj=self.value_in,
            output_norm=self.value_out_norm,
            output_proj=self.value_out,
        )
        return key_shared, value_shared


@register_router
@capture_init_args
class SharedLatentFusionRouter(nn.Module):
    """
    Router whose feature is produced by a paper-style KV -> shared latent
    encoder, followed by an MLP action head.

    In shared_latent_receiver mode, this router consumes only the receiver
    prefill cache. In shared_latent_fusion mode, it consumes the all-fusion
    candidate cache. In shared_latent_pair mode, it compares the receiver-only
    cache and all-fusion candidate cache through the same encoder.
    """

    def __init__(
        self,
        num_banks: int,
        num_kv_layers: int,
        kv_dim: int,
        encoder_dim: int = 1024,
        shared_dim: int = 1024,
        num_encoder_heads: int = 16,
        encoder_ffn_mult: float = 4.0,
        encoder_dropout: float = 0.0,
        max_encoder_tokens: Optional[int] = None,
        pooling: str = "last",
        router_hidden_dim: int = 512,
        router_layers: int = 2,
        router_dropout: float = 0.1,
        input_layer_norm: bool = True,
        feature_source: str = "shared_latent_fusion",
        decision_policy: str = "argmax",
        skip_threshold: float = 0.5,
        no_skip_routing: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if num_banks <= 0:
            raise ValueError(f"num_banks must be positive, got {num_banks}")
        if feature_source not in {
            "shared_latent_receiver",
            "shared_latent_fusion",
            "shared_latent_pair",
        }:
            raise ValueError(
                "SharedLatentFusionRouter only supports "
                "feature_source in {'shared_latent_receiver', "
                "'shared_latent_fusion', 'shared_latent_pair'}."
            )
        if pooling not in {"last", "mean", "last_mean"}:
            raise ValueError("pooling must be one of: last, mean, last_mean")
        if decision_policy not in {"argmax", "skip_threshold", "score_threshold"}:
            raise ValueError(
                "decision_policy must be one of: argmax, skip_threshold, score_threshold"
            )
        if decision_policy != "score_threshold" and not 0.0 <= float(skip_threshold) <= 1.0:
            raise ValueError(f"skip_threshold must be in [0, 1], got {skip_threshold}")

        self.num_banks = int(num_banks)
        self.no_skip_routing = bool(no_skip_routing)
        self.num_actions = self.num_banks if self.no_skip_routing else self.num_banks + 1
        self.feature_source = feature_source
        self.pooling = pooling
        self.decision_policy = decision_policy
        self.skip_threshold = float(skip_threshold)
        self.shared_dim = int(shared_dim)
        pooled_multiplier = 2 if pooling == "last_mean" else 1
        self.single_cache_feature_dim = 2 * pooled_multiplier * int(shared_dim)
        self.input_dim = self.single_cache_feature_dim
        if feature_source == "shared_latent_pair":
            self.input_dim = self.single_cache_feature_dim * 4
        self.token_feature_dim = self.input_dim

        self.encoder = LocalToSharedKVEncoder(
            num_kv_layers=num_kv_layers,
            kv_dim=kv_dim,
            encoder_dim=encoder_dim,
            shared_dim=shared_dim,
            num_heads=num_encoder_heads,
            ffn_mult=encoder_ffn_mult,
            dropout=encoder_dropout,
            max_encoder_tokens=max_encoder_tokens,
            dtype=dtype,
        )

        self.input_norm = nn.LayerNorm(self.input_dim, dtype=dtype) if input_layer_norm else nn.Identity()
        layers = []
        in_dim = self.input_dim
        for _ in range(router_layers):
            layers.append(nn.Linear(in_dim, router_hidden_dim, dtype=dtype))
            layers.append(nn.GELU())
            if router_dropout > 0:
                layers.append(nn.Dropout(router_dropout))
            in_dim = router_hidden_dim
        self.feature_encoder = nn.Sequential(*layers)
        self.action_head = nn.Linear(in_dim, self.num_actions, dtype=dtype)

    def _pool_shared(self, shared_sequence: Tensor) -> Tensor:
        if self.pooling == "last":
            return shared_sequence[:, -1, :]
        if self.pooling == "mean":
            return shared_sequence.mean(dim=1)
        return torch.cat(
            [shared_sequence[:, -1, :], shared_sequence.mean(dim=1)],
            dim=-1,
        )

    def extract_query_feature_from_fused_cache(
        self,
        fused_cache: DynamicCache,
        *,
        new_length: Optional[int] = None,
    ) -> Tensor:
        key_shared, value_shared = self.encoder(fused_cache, new_length=new_length)
        return torch.cat(
            [self._pool_shared(key_shared), self._pool_shared(value_shared)],
            dim=-1,
        )

    def extract_query_feature_from_cache_pair(
        self,
        receiver_cache: DynamicCache,
        fused_cache: DynamicCache,
        *,
        new_length: Optional[int] = None,
    ) -> Tensor:
        receiver_feature = self.extract_query_feature_from_fused_cache(
            receiver_cache,
            new_length=new_length,
        )
        fusion_feature = self.extract_query_feature_from_fused_cache(
            fused_cache,
            new_length=new_length,
        )
        return torch.cat(
            [
                receiver_feature,
                fusion_feature,
                fusion_feature - receiver_feature,
                (fusion_feature - receiver_feature).abs(),
            ],
            dim=-1,
        )

    def forward_features(self, pooled_feature: Tensor) -> RouterOutput:
        pooled_feature = pooled_feature.to(
            device=self.action_head.weight.device,
            dtype=self.action_head.weight.dtype,
        )
        if pooled_feature.size(-1) != self.input_dim:
            raise ValueError(
                "Router input dimension mismatch: "
                f"expected {self.input_dim}, got {pooled_feature.size(-1)}"
            )
        encoded = self.feature_encoder(self.input_norm(pooled_feature))
        return RouterOutput(
            pooled_feature=pooled_feature,
            action_logits=self.action_head(encoded),
        )

    def forward(
        self,
        fused_cache: Optional[DynamicCache] = None,
        pooled_feature: Optional[Tensor] = None,
        new_length: Optional[int] = None,
        **_: object,
    ) -> RouterOutput:
        if pooled_feature is None:
            if fused_cache is None:
                raise ValueError("SharedLatentFusionRouter requires fused_cache or pooled_feature.")
            pooled_feature = self.extract_query_feature_from_fused_cache(
                fused_cache,
                new_length=new_length,
            )
        return self.forward_features(pooled_feature)

    @torch.no_grad()
    def predict(
        self,
        base_cache: Optional[DynamicCache] = None,
        source_cache: Optional[DynamicCache] = None,
        fused_cache: Optional[DynamicCache] = None,
        fuse_threshold: float = 0.5,
        pooled_feature: Optional[Tensor] = None,
        new_length: Optional[int] = None,
    ) -> dict:
        del base_cache, source_cache, fuse_threshold
        output = self.forward(
            fused_cache=fused_cache,
            pooled_feature=pooled_feature,
            new_length=new_length,
        )
        action_logits = output.action_logits
        if self.decision_policy == "skip_threshold":
            pred_bank = torch.argmax(output.selection_logits, dim=-1)
            selected_action = torch.where(
                output.skip_probability > self.skip_threshold,
                torch.zeros_like(pred_bank),
                pred_bank + 1,
            )
        elif self.decision_policy == "score_threshold":
            pred_bank = torch.argmax(output.selection_logits, dim=-1)
            selected_action = torch.where(
                output.binary_logits > self.skip_threshold,
                pred_bank + 1,
                torch.zeros_like(pred_bank),
            )
        else:
            selected_action = torch.argmax(action_logits, dim=-1)
        if self.no_skip_routing:
            should_fuse = torch.ones_like(selected_action, dtype=torch.bool)
            selected_bank = selected_action
        else:
            should_fuse = selected_action > 0
            selected_bank = torch.where(
                should_fuse,
                selected_action - 1,
                torch.full_like(selected_action, -1),
            )
        return {
            "pooled_feature": output.pooled_feature,
            "action_logits": action_logits,
            "action_probabilities": output.action_probabilities,
            "skip_probability": output.skip_probability,
            "binary_logits": output.binary_logits,
            "selection_logits": output.selection_logits,
            "fuse_probability": output.fuse_probability,
            "should_fuse": should_fuse,
            "selected_bank": selected_bank,
            "selected_action": selected_action,
        }


def save_router(obj: nn.Module, file_path: str) -> None:
    save_object(obj, file_path)


def load_router(file_path: str, override_args: Optional[dict] = None) -> nn.Module:
    return load_object(file_path, get_router_class, override_args)
