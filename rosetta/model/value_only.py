"""Utilities for value-only KV projection ablations.

The patch preserves receiver keys and lets projector-produced values pass
through unchanged. It is intentionally runtime-only so existing checkpoints can
be evaluated without rewriting saved projector files.
"""

from __future__ import annotations

import types
from typing import Any


def apply_projector_value_only(model: Any) -> None:
    """Patch every Rosetta projector to inject values only.

    For per-layer projectors:
        (projected_key, projected_value) -> (receiver_key, projected_value)

    For full-cache projectors:
        [(projected_key_i, projected_value_i)] -> [(receiver_key_i, projected_value_i)]
    """

    if not hasattr(model, "projector_list"):
        return

    for projector in model.projector_list:
        if getattr(projector, "_value_only_patch", False):
            continue

        original_forward = projector.forward

        def value_only_forward(self_projector, source_kv, target_kv, *args, _orig=original_forward, **kwargs):
            _, projected_value = _orig(source_kv, target_kv, *args, **kwargs)
            target_key, _ = target_kv
            return target_key, projected_value

        projector.forward = types.MethodType(value_only_forward, projector)

        if hasattr(projector, "forward_cache"):
            original_forward_cache = projector.forward_cache

            def value_only_forward_cache(
                self_projector,
                source_layers,
                target_layers,
                *args,
                _orig=original_forward_cache,
                **kwargs,
            ):
                projected_layers = _orig(source_layers, target_layers, *args, **kwargs)
                return [
                    (target_key, projected_value)
                    for (target_key, _), (_, projected_value) in zip(target_layers, projected_layers)
                ]

            projector.forward_cache = types.MethodType(value_only_forward_cache, projector)

        projector._value_only_patch = True
