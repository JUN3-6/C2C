"""
Shared router feature extractors.

These helpers are used by both offline label generation and routed inference so
the router sees the same feature family in training and evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache


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


def _lookup_int_key(mapping: Dict[Any, Any], key: int):
    if key in mapping:
        return mapping[key]
    text_key = str(key)
    if text_key in mapping:
        return mapping[text_key]
    raise KeyError(key)


def _iter_bank_entries(
    projector_bank_config: Dict[str, Any],
    *,
    base_model_idx: int,
    source_model_idx: int,
) -> Iterable[tuple[int, int, int]]:
    base_config = _lookup_int_key(projector_bank_config, base_model_idx)
    source_config = _lookup_int_key(base_config, source_model_idx)
    for target_layer_idx in sorted(int(k) for k in source_config.keys()):
        entries = _lookup_int_key(source_config, target_layer_idx)
        for source_layer_idx, projector_idx in entries:
            yield int(target_layer_idx), int(source_layer_idx), int(projector_idx)


PROJECTOR_IN_FEATURE_SOURCES = {
    "projector_in",
    "projector_in_stats",
    "projector_in_pooled",
    "projector_in_binned",
}

POSTFUSION_DELTA_FEATURE_SOURCES = {
    "postfusion_delta_stats",
}

POSTFUSION_PROBE_LOGIT_FEATURE_SOURCES = {
    "postfusion_probe_logits",
}

POSTFUSION_OPTION_LOGIT_FEATURE_SOURCES = {
    "postfusion_option_logits",
    "postfusion_option_logits_hidden_binned",
}

POSTFUSION_FEATURE_SOURCES = (
    POSTFUSION_DELTA_FEATURE_SOURCES
    | POSTFUSION_PROBE_LOGIT_FEATURE_SOURCES
    | POSTFUSION_OPTION_LOGIT_FEATURE_SOURCES
)

# Keep flattened router inputs compact. With the current 28-layer MMLU bank,
# 2 bins yields 28 layers * 24 stats = 672 dimensions.
PROJECTOR_IN_BIN_COUNT = 2


def _pooled_activation_stats(hidden: Tensor) -> Tensor:
    hidden = hidden.float()
    last = hidden[:, -1, :]
    token_mean = hidden.mean(dim=-1)
    token_rms = hidden.pow(2).mean(dim=-1).sqrt()
    feature_mean = hidden.mean(dim=1)
    delta = last - feature_mean
    hidden_dim = max(1, int(hidden.size(-1)))
    scale = hidden_dim ** 0.5

    return torch.cat(
        [
            hidden.mean(dim=(1, 2), keepdim=False).unsqueeze(-1),
            hidden.std(dim=(1, 2), unbiased=False, keepdim=False).unsqueeze(-1),
            hidden.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1),
            hidden.pow(2).mean(dim=(1, 2), keepdim=False).sqrt().unsqueeze(-1),
            last.mean(dim=-1, keepdim=True),
            last.std(dim=-1, unbiased=False, keepdim=True),
            last.abs().mean(dim=-1, keepdim=True),
            last.norm(dim=-1, keepdim=True) / scale,
            token_mean.std(dim=-1, unbiased=False, keepdim=True),
            token_rms.mean(dim=-1, keepdim=True),
            delta.abs().mean(dim=-1, keepdim=True),
            delta.norm(dim=-1, keepdim=True) / scale,
        ],
        dim=-1,
    )


def _adaptive_binned_stats(vector: Tensor, *, num_bins: int = PROJECTOR_IN_BIN_COUNT) -> Tensor:
    vector = vector.float()
    if vector.ndim != 2:
        raise ValueError(f"Expected vector with shape (B, D), got {tuple(vector.shape)}")
    pooled_mean = F.adaptive_avg_pool1d(vector.unsqueeze(1), num_bins).squeeze(1)
    pooled_square_mean = F.adaptive_avg_pool1d(vector.pow(2).unsqueeze(1), num_bins).squeeze(1)
    pooled_std = (pooled_square_mean - pooled_mean.pow(2)).clamp_min(0).sqrt()
    return torch.cat([pooled_mean, pooled_std], dim=-1)


def _projector_input_hidden(
    projector: nn.Module,
    source_kv: tuple[Tensor, Tensor],
    target_kv: tuple[Tensor, Tensor],
    *,
    new_length: Optional[int],
    feature_source: str,
) -> Tensor:
    source_key, source_value = source_kv
    target_key, target_value = target_kv

    seq_len = min(
        int(source_key.size(2)),
        int(source_value.size(2)),
        int(target_key.size(2)),
        int(target_value.size(2)),
    )
    if new_length is not None:
        seq_len = min(seq_len, int(new_length))
    if seq_len <= 0:
        raise ValueError("Cannot build projector_in feature from an empty cache slice")

    source_key = source_key[:, :, -seq_len:, :]
    source_value = source_value[:, :, -seq_len:, :]
    target_key = target_key[:, :, -seq_len:, :]
    target_value = target_value[:, :, -seq_len:, :]

    batch_size, source_heads, _, source_dim = source_key.shape
    _, target_heads, _, target_dim = target_key.shape

    source_key_flat = (
        source_key.transpose(1, 2).contiguous().view(batch_size, seq_len, source_heads * source_dim)
    )
    source_value_flat = (
        source_value.transpose(1, 2).contiguous().view(batch_size, seq_len, source_heads * source_dim)
    )
    target_key_flat = (
        target_key.transpose(1, 2).contiguous().view(batch_size, seq_len, target_heads * target_dim)
    )
    target_value_flat = (
        target_value.transpose(1, 2).contiguous().view(batch_size, seq_len, target_heads * target_dim)
    )

    key_cat = torch.cat([source_key_flat, target_key_flat], dim=-1)
    value_cat = torch.cat([source_value_flat, target_value_flat], dim=-1)

    key_in = getattr(projector, "key_in", None)
    value_in = getattr(projector, "value_in", None)
    if key_in is None or value_in is None:
        raise TypeError(
            f"Projector {type(projector).__name__} does not expose key_in/value_in layers"
        )

    device = key_in.weight.device
    dtype = key_in.weight.dtype
    key_hidden = key_in(key_cat.to(device=device, dtype=dtype))
    value_hidden = value_in(value_cat.to(device=device, dtype=dtype))

    key_hidden = key_hidden.float()
    value_hidden = value_hidden.float()

    # Keep layer information outside this helper; summarize only the token axis.
    if feature_source == "projector_in":
        return torch.cat(
            [
                key_hidden.mean(dim=1),
                value_hidden.mean(dim=1),
            ],
            dim=-1,
        )
    if feature_source == "projector_in_stats":
        key_mean = key_hidden.mean(dim=1)
        value_mean = value_hidden.mean(dim=1)
        key_std = key_hidden.std(dim=1, unbiased=False)
        value_std = value_hidden.std(dim=1, unbiased=False)
        key_last = key_hidden[:, -1, :]
        value_last = value_hidden[:, -1, :]
        key_delta = key_last - key_mean
        value_delta = value_last - value_mean
        key_hidden_dim = max(1, int(key_hidden.size(-1)))
        value_hidden_dim = max(1, int(value_hidden.size(-1)))
        key_scale = key_hidden_dim ** 0.5
        value_scale = value_hidden_dim ** 0.5

        # Keep the original dense mean representation, then append a small set
        # of per-layer scalar summaries so dataset size stays close to v1.
        key_stats = torch.cat(
            [
                key_std.mean(dim=-1, keepdim=True),
                key_std.norm(dim=-1, keepdim=True) / key_scale,
                key_delta.abs().mean(dim=-1, keepdim=True),
                key_delta.norm(dim=-1, keepdim=True) / key_scale,
                key_mean.norm(dim=-1, keepdim=True) / key_scale,
                key_last.norm(dim=-1, keepdim=True) / key_scale,
            ],
            dim=-1,
        )
        value_stats = torch.cat(
            [
                value_std.mean(dim=-1, keepdim=True),
                value_std.norm(dim=-1, keepdim=True) / value_scale,
                value_delta.abs().mean(dim=-1, keepdim=True),
                value_delta.norm(dim=-1, keepdim=True) / value_scale,
                value_mean.norm(dim=-1, keepdim=True) / value_scale,
                value_last.norm(dim=-1, keepdim=True) / value_scale,
            ],
            dim=-1,
        )
        return torch.cat(
            [
                key_mean,
                value_mean,
                key_stats,
                value_stats,
            ],
            dim=-1,
        )
    if feature_source == "projector_in_pooled":
        return torch.cat(
            [
                _pooled_activation_stats(key_hidden),
                _pooled_activation_stats(value_hidden),
            ],
            dim=-1,
        )
    if feature_source == "projector_in_binned":
        key_mean = key_hidden.mean(dim=1)
        value_mean = value_hidden.mean(dim=1)
        key_std = key_hidden.std(dim=1, unbiased=False)
        value_std = value_hidden.std(dim=1, unbiased=False)
        key_delta = key_hidden[:, -1, :] - key_mean
        value_delta = value_hidden[:, -1, :] - value_mean
        return torch.cat(
            [
                _adaptive_binned_stats(key_mean),
                _adaptive_binned_stats(value_mean),
                _adaptive_binned_stats(key_std),
                _adaptive_binned_stats(value_std),
                _adaptive_binned_stats(key_delta),
                _adaptive_binned_stats(value_delta),
            ],
            dim=-1,
        )
    raise ValueError(f"Unsupported projector feature_source={feature_source}")


@torch.no_grad()
def extract_projector_in_feature(
    *,
    base_cache: DynamicCache,
    source_cache: DynamicCache,
    projector_list: List[nn.Module],
    projector_bank_config: Dict[str, Any],
    base_model_idx: int = 0,
    source_model_idx: int = 1,
    new_length: Optional[int] = None,
    feature_source: str = "projector_in",
) -> Tensor:
    """
    Build a router feature from the first learned projector representation.

    For each target layer in the selected bank:
      1. flatten source/target KV heads,
      2. concatenate source and target KV,
      3. run only projector.key_in/value_in,
      4. summarize tokens according to feature_source,
      5. concatenate layer features in target-layer order.
    """
    if feature_source not in PROJECTOR_IN_FEATURE_SOURCES:
        raise ValueError(
            f"Unsupported projector feature_source={feature_source}. "
            f"Expected one of {sorted(PROJECTOR_IN_FEATURE_SOURCES)}."
        )

    base_cache = _ensure_dynamic_cache(base_cache)
    source_cache = _ensure_dynamic_cache(source_cache)
    base_keys = _cache_key_layers(base_cache)
    base_values = _cache_value_layers(base_cache)
    source_keys = _cache_key_layers(source_cache)
    source_values = _cache_value_layers(source_cache)

    layer_features = []
    for target_layer_idx, source_layer_idx, projector_idx in _iter_bank_entries(
        projector_bank_config,
        base_model_idx=base_model_idx,
        source_model_idx=source_model_idx,
    ):
        projector = projector_list[projector_idx]
        feature = _projector_input_hidden(
            projector,
            source_kv=(source_keys[source_layer_idx], source_values[source_layer_idx]),
            target_kv=(base_keys[target_layer_idx], base_values[target_layer_idx]),
            new_length=new_length,
            feature_source=feature_source,
        )
        layer_features.append(feature)

    if not layer_features:
        raise ValueError("No projector entries were available for projector_in feature")
    return torch.cat(layer_features, dim=-1)


def _cache_delta_stats(base: Tensor, fused: Tensor, *, eps: float = 1e-6) -> Tensor:
    base = base.float()
    fused = fused.float()
    delta = fused - base

    batch_size = base.size(0)
    base_flat = base.reshape(batch_size, -1)
    fused_flat = fused.reshape(batch_size, -1)
    delta_flat = delta.reshape(batch_size, -1)

    base_rms = base_flat.pow(2).mean(dim=-1, keepdim=True).sqrt()
    fused_rms = fused_flat.pow(2).mean(dim=-1, keepdim=True).sqrt()
    delta_rms = delta_flat.pow(2).mean(dim=-1, keepdim=True).sqrt()
    delta_abs_mean = delta_flat.abs().mean(dim=-1, keepdim=True)
    delta_abs_max = delta_flat.abs().amax(dim=-1, keepdim=True)
    delta_ratio = delta_rms / base_rms.clamp_min(eps)

    base_fused_cos = F.cosine_similarity(base_flat, fused_flat, dim=-1).unsqueeze(-1)
    base_delta_cos = F.cosine_similarity(base_flat, delta_flat, dim=-1).unsqueeze(-1)

    token_delta_rms = delta.pow(2).mean(dim=(1, 3)).sqrt()
    token_base_rms = base.pow(2).mean(dim=(1, 3)).sqrt()
    token_delta_mean = token_delta_rms.mean(dim=-1, keepdim=True)
    token_delta_std = token_delta_rms.std(dim=-1, unbiased=False, keepdim=True)
    last_delta_ratio = token_delta_rms[:, -1:].div(token_base_rms[:, -1:].clamp_min(eps))

    signed_shift = (base_flat * delta_flat).mean(dim=-1, keepdim=True) / (
        base_rms * delta_rms
    ).clamp_min(eps)

    return torch.cat(
        [
            base_rms,
            fused_rms,
            delta_rms,
            delta_abs_mean,
            delta_abs_max,
            delta_ratio,
            base_fused_cos,
            base_delta_cos,
            token_delta_mean,
            token_delta_std,
            last_delta_ratio,
            signed_shift,
        ],
        dim=-1,
    )


@torch.no_grad()
def extract_postfusion_delta_feature(
    *,
    base_cache: DynamicCache,
    fused_cache: DynamicCache,
    projector_bank_config: Dict[str, Any],
    base_model_idx: int = 0,
    source_model_idx: int = 1,
    new_length: Optional[int] = None,
    feature_source: str = "postfusion_delta_stats",
) -> Tensor:
    """
    Build a compact router feature after candidate fusion has been applied.

    The feature keeps the projector bank's target-layer order and summarizes how
    much the candidate fused cache moved receiver key/value tensors. If the
    router predicts skip, inference can discard this candidate and keep the
    original receiver cache.
    """
    if feature_source not in POSTFUSION_DELTA_FEATURE_SOURCES:
        raise ValueError(
            f"Unsupported post-fusion feature_source={feature_source}. "
            f"Expected one of {sorted(POSTFUSION_DELTA_FEATURE_SOURCES)}."
        )

    base_cache = _ensure_dynamic_cache(base_cache)
    fused_cache = _ensure_dynamic_cache(fused_cache)
    base_keys = _cache_key_layers(base_cache)
    base_values = _cache_value_layers(base_cache)
    fused_keys = _cache_key_layers(fused_cache)
    fused_values = _cache_value_layers(fused_cache)

    seen_layers = set()
    layer_features = []
    for target_layer_idx, _, _ in _iter_bank_entries(
        projector_bank_config,
        base_model_idx=base_model_idx,
        source_model_idx=source_model_idx,
    ):
        if target_layer_idx in seen_layers:
            continue
        seen_layers.add(target_layer_idx)

        key_seq_len = min(
            int(base_keys[target_layer_idx].size(2)),
            int(fused_keys[target_layer_idx].size(2)),
        )
        value_seq_len = min(
            int(base_values[target_layer_idx].size(2)),
            int(fused_values[target_layer_idx].size(2)),
        )
        seq_len = min(key_seq_len, value_seq_len)
        if new_length is not None:
            seq_len = min(seq_len, int(new_length))
        if seq_len <= 0:
            raise ValueError("Cannot build post-fusion feature from an empty cache slice")

        base_key = base_keys[target_layer_idx][:, :, -seq_len:, :]
        fused_key = fused_keys[target_layer_idx][:, :, -seq_len:, :]
        base_value = base_values[target_layer_idx][:, :, -seq_len:, :]
        fused_value = fused_values[target_layer_idx][:, :, -seq_len:, :]
        layer_features.append(
            torch.cat(
                [
                    _cache_delta_stats(base_key, fused_key),
                    _cache_delta_stats(base_value, fused_value),
                ],
                dim=-1,
            )
        )

    if not layer_features:
        raise ValueError("No projector entries were available for post-fusion feature")
    return torch.cat(layer_features, dim=-1)


def _next_token_logits(logits: Tensor) -> Tensor:
    if logits.ndim == 3:
        return logits[:, -1, :]
    if logits.ndim == 2:
        return logits
    raise ValueError(f"Expected logits with shape (B, V) or (B, T, V), got {tuple(logits.shape)}")


def _logit_distribution_stats(logits: Tensor, *, top_k: int = 8, eps: float = 1e-8) -> Tensor:
    logits = _next_token_logits(logits).float()
    vocab_size = max(1, int(logits.size(-1)))
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
    normalized_entropy = entropy / max(1.0, float(torch.log(torch.tensor(float(vocab_size))).item()))

    k = min(top_k, vocab_size)
    top_logits, _ = torch.topk(logits, k=k, dim=-1)
    top_probs, _ = torch.topk(probs, k=k, dim=-1)
    if k < top_k:
        top_logits = F.pad(top_logits, (0, top_k - k))
        top_probs = F.pad(top_probs, (0, top_k - k))

    top2_gap = (
        (top_logits[:, 0] - top_logits[:, 1]).unsqueeze(-1)
        if vocab_size > 1
        else torch.zeros_like(entropy)
    )
    top_prob_gap = (
        (top_probs[:, 0] - top_probs[:, 1]).unsqueeze(-1)
        if vocab_size > 1
        else torch.zeros_like(entropy)
    )

    centered = logits - logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    z = centered / std
    top_z, _ = torch.topk(z, k=k, dim=-1)
    if k < top_k:
        top_z = F.pad(top_z, (0, top_k - k))

    def top_mass(width: int) -> Tensor:
        kk = min(width, vocab_size)
        return torch.topk(probs, k=kk, dim=-1).values.sum(dim=-1, keepdim=True)

    return torch.cat(
        [
            normalized_entropy,
            top_probs[:, :1],
            top_prob_gap,
            top2_gap,
            logits.mean(dim=-1, keepdim=True),
            logits.std(dim=-1, unbiased=False, keepdim=True),
            logits.amax(dim=-1, keepdim=True),
            logits.amin(dim=-1, keepdim=True),
            top_mass(5),
            top_mass(10),
            top_mass(20),
            top_probs,
            top_logits,
            top_z,
        ],
        dim=-1,
    )


@torch.no_grad()
def extract_probe_logits_feature(
    *,
    receiver_logits: Tensor,
    fused_logits: Tensor,
    prefill_logits: Optional[Tensor] = None,
    feature_source: str = "postfusion_probe_logits",
    eps: float = 1e-8,
) -> Tensor:
    """
    Compact feature from receiver-only and candidate-fused probe logits.

    The probe token is chosen by the caller, usually receiver prefill top-1.
    This feature is intentionally post-fusion: it summarizes the actual output
    distribution shift caused by using the fused cache for one cheap probe step.
    """
    if feature_source not in POSTFUSION_PROBE_LOGIT_FEATURE_SOURCES:
        raise ValueError(
            f"Unsupported probe-logit feature_source={feature_source}. "
            f"Expected one of {sorted(POSTFUSION_PROBE_LOGIT_FEATURE_SOURCES)}."
        )

    receiver = _next_token_logits(receiver_logits).float()
    fused = _next_token_logits(fused_logits).float()
    if receiver.shape != fused.shape:
        raise ValueError(
            "receiver/fused logits must have the same shape: "
            f"receiver={tuple(receiver.shape)} fused={tuple(fused.shape)}"
        )

    receiver_probs = torch.softmax(receiver, dim=-1)
    fused_probs = torch.softmax(fused, dim=-1)
    receiver_log_probs = torch.log_softmax(receiver, dim=-1)
    fused_log_probs = torch.log_softmax(fused, dim=-1)
    midpoint = 0.5 * (receiver_probs + fused_probs)
    midpoint_log = midpoint.clamp_min(eps).log()

    tv = 0.5 * (receiver_probs - fused_probs).abs().sum(dim=-1, keepdim=True)
    js = 0.5 * (
        (receiver_probs * (receiver_log_probs - midpoint_log)).sum(dim=-1, keepdim=True)
        + (fused_probs * (fused_log_probs - midpoint_log)).sum(dim=-1, keepdim=True)
    )
    js = js / torch.log(torch.tensor(2.0, device=js.device, dtype=js.dtype))
    logit_cos = F.cosine_similarity(
        receiver - receiver.mean(dim=-1, keepdim=True),
        fused - fused.mean(dim=-1, keepdim=True),
        dim=-1,
    ).unsqueeze(-1)

    receiver_top = torch.topk(receiver_probs, k=min(20, receiver_probs.size(-1)), dim=-1).indices
    fused_top = torch.topk(fused_probs, k=min(20, fused_probs.size(-1)), dim=-1).indices

    def overlap(width: int) -> Tensor:
        k = min(width, receiver_top.size(1), fused_top.size(1))
        same = (
            receiver_top[:, :k].unsqueeze(-1)
            == fused_top[:, :k].unsqueeze(1)
        ).any(dim=-1)
        return same.float().sum(dim=-1, keepdim=True) / max(1, k)

    receiver_top1 = receiver_probs.argmax(dim=-1, keepdim=True)
    fused_top1 = fused_probs.argmax(dim=-1, keepdim=True)
    receiver_prob_at_fused_top1 = receiver_probs.gather(1, fused_top1)
    fused_prob_at_receiver_top1 = fused_probs.gather(1, receiver_top1)
    top1_changed = (receiver_top1 != fused_top1).float()

    cross_stats = torch.cat(
        [
            tv,
            js,
            logit_cos,
            overlap(1),
            overlap(5),
            overlap(10),
            overlap(20),
            top1_changed,
            receiver_prob_at_fused_top1,
            fused_prob_at_receiver_top1,
        ],
        dim=-1,
    )

    pieces = [
        _logit_distribution_stats(receiver),
        _logit_distribution_stats(fused),
        cross_stats,
    ]
    if prefill_logits is not None:
        prefill = _next_token_logits(prefill_logits).float()
        if prefill.shape == receiver.shape:
            pieces.append(_logit_distribution_stats(prefill))
            prefill_probs = torch.softmax(prefill, dim=-1)
            pieces.append(
                torch.cat(
                    [
                        prefill_probs.gather(1, receiver_top1),
                        prefill_probs.gather(1, fused_top1),
                        F.cosine_similarity(
                            prefill - prefill.mean(dim=-1, keepdim=True),
                            fused - fused.mean(dim=-1, keepdim=True),
                            dim=-1,
                        ).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
    return torch.cat(pieces, dim=-1)


def _gather_option_logits(logits: Tensor, option_token_ids: Iterable[int]) -> Tensor:
    logits = _next_token_logits(logits).float()
    option_ids = torch.tensor(
        [int(token_id) for token_id in option_token_ids],
        dtype=torch.long,
        device=logits.device,
    )
    if option_ids.numel() <= 0:
        raise ValueError("option_token_ids must contain at least one token id.")
    if int(option_ids.min().item()) < 0 or int(option_ids.max().item()) >= logits.size(-1):
        raise ValueError(
            "option_token_ids contain ids outside the logits vocabulary: "
            f"vocab={logits.size(-1)} ids={option_ids.detach().cpu().tolist()}"
        )
    return logits.index_select(dim=-1, index=option_ids)


def _option_distribution_stats(option_logits: Tensor, *, eps: float = 1e-8) -> Tensor:
    option_logits = option_logits.float()
    num_options = max(1, int(option_logits.size(-1)))
    option_probs = torch.softmax(option_logits, dim=-1)
    option_log_probs = torch.log_softmax(option_logits, dim=-1)
    sorted_logits = torch.sort(option_logits, dim=-1, descending=True).values
    sorted_probs = torch.sort(option_probs, dim=-1, descending=True).values
    top2_logit_gap = (
        (sorted_logits[:, 0] - sorted_logits[:, 1]).unsqueeze(-1)
        if num_options > 1
        else torch.zeros(option_logits.size(0), 1, device=option_logits.device)
    )
    top2_prob_gap = (
        (sorted_probs[:, 0] - sorted_probs[:, 1]).unsqueeze(-1)
        if num_options > 1
        else torch.zeros(option_logits.size(0), 1, device=option_logits.device)
    )
    entropy = -(option_probs * option_log_probs).sum(dim=-1, keepdim=True)
    normalized_entropy = entropy / max(1.0, float(torch.log(torch.tensor(float(num_options))).item()))
    pred = torch.argmax(option_logits, dim=-1)
    pred_one_hot = F.one_hot(pred, num_classes=num_options).float()
    centered = option_logits - option_logits.mean(dim=-1, keepdim=True)
    std = option_logits.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    z = centered / std
    return torch.cat(
        [
            option_logits,
            option_log_probs,
            option_probs,
            z,
            option_logits.mean(dim=-1, keepdim=True),
            option_logits.std(dim=-1, unbiased=False, keepdim=True),
            option_logits.amax(dim=-1, keepdim=True),
            option_logits.amin(dim=-1, keepdim=True),
            top2_logit_gap,
            sorted_probs[:, :1],
            top2_prob_gap,
            normalized_entropy,
            pred_one_hot,
        ],
        dim=-1,
    )


def extract_hidden_binned_feature(
    *,
    base_last_hidden: Tensor,
    source_last_hidden: Tensor,
    bins_per_model: int,
) -> Tensor:
    if bins_per_model <= 0:
        raise ValueError(f"bins_per_model must be positive, got {bins_per_model}")
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

    def _bin(hidden: Tensor) -> Tensor:
        hidden = hidden.float().unsqueeze(1)
        bin_mean = F.adaptive_avg_pool1d(hidden, bins_per_model).squeeze(1)
        bin_second_moment = F.adaptive_avg_pool1d(hidden.square(), bins_per_model).squeeze(1)
        bin_std = (bin_second_moment - bin_mean.square()).clamp_min(0.0).sqrt()
        return torch.cat([bin_mean, bin_std], dim=-1)

    return torch.cat([_bin(base_last_hidden), _bin(source_last_hidden)], dim=-1)


@torch.no_grad()
def extract_option_logits_feature(
    *,
    receiver_logits: Tensor,
    fused_logits: Tensor,
    option_token_ids: Iterable[int],
    prefill_logits: Optional[Tensor] = None,
    feature_source: str = "postfusion_option_logits",
    eps: float = 1e-8,
) -> Tensor:
    """
    MMLU-style post-fusion feature over the A/B/C/D candidate logits.

    The caller probes the model after a fixed answer prefix, e.g.
    "The correct answer is", then this function summarizes only option-token
    logits. It never uses the gold option, so it can also run at inference time.
    """
    if feature_source not in POSTFUSION_OPTION_LOGIT_FEATURE_SOURCES:
        raise ValueError(
            f"Unsupported option-logit feature_source={feature_source}. "
            f"Expected one of {sorted(POSTFUSION_OPTION_LOGIT_FEATURE_SOURCES)}."
        )

    receiver_options = _gather_option_logits(receiver_logits, option_token_ids)
    fused_options = _gather_option_logits(fused_logits, option_token_ids)
    if receiver_options.shape != fused_options.shape:
        raise ValueError(
            "receiver/fused option logits must have the same shape: "
            f"receiver={tuple(receiver_options.shape)} fused={tuple(fused_options.shape)}"
        )

    receiver_probs = torch.softmax(receiver_options, dim=-1)
    fused_probs = torch.softmax(fused_options, dim=-1)
    receiver_log_probs = torch.log_softmax(receiver_options, dim=-1)
    fused_log_probs = torch.log_softmax(fused_options, dim=-1)
    midpoint = 0.5 * (receiver_probs + fused_probs)
    midpoint_log = midpoint.clamp_min(eps).log()

    receiver_pred = torch.argmax(receiver_options, dim=-1, keepdim=True)
    fused_pred = torch.argmax(fused_options, dim=-1, keepdim=True)
    pred_changed = (receiver_pred != fused_pred).float()
    receiver_prob_at_fused_pred = receiver_probs.gather(1, fused_pred)
    fused_prob_at_receiver_pred = fused_probs.gather(1, receiver_pred)
    tv = 0.5 * (receiver_probs - fused_probs).abs().sum(dim=-1, keepdim=True)
    js = 0.5 * (
        (receiver_probs * (receiver_log_probs - midpoint_log)).sum(dim=-1, keepdim=True)
        + (fused_probs * (fused_log_probs - midpoint_log)).sum(dim=-1, keepdim=True)
    )
    js = js / torch.log(torch.tensor(2.0, device=js.device, dtype=js.dtype))
    logit_cos = F.cosine_similarity(
        receiver_options - receiver_options.mean(dim=-1, keepdim=True),
        fused_options - fused_options.mean(dim=-1, keepdim=True),
        dim=-1,
    ).unsqueeze(-1)

    pieces = [
        _option_distribution_stats(receiver_options, eps=eps),
        _option_distribution_stats(fused_options, eps=eps),
        fused_options - receiver_options,
        fused_log_probs - receiver_log_probs,
        fused_probs - receiver_probs,
        torch.cat(
            [
                tv,
                js,
                logit_cos,
                pred_changed,
                receiver_prob_at_fused_pred,
                fused_prob_at_receiver_pred,
            ],
            dim=-1,
        ),
    ]
    if prefill_logits is not None:
        prefill_options = _gather_option_logits(prefill_logits, option_token_ids)
        if prefill_options.shape == receiver_options.shape:
            prefill_probs = torch.softmax(prefill_options, dim=-1)
            pieces.extend(
                [
                    _option_distribution_stats(prefill_options, eps=eps),
                    receiver_options - prefill_options,
                    fused_options - prefill_options,
                    receiver_probs - prefill_probs,
                    fused_probs - prefill_probs,
                ]
            )
    return torch.cat(pieces, dim=-1)
