"""
RosettaModel wrapper for C2C v4 projectors.

V4 projectors need both the original source KV slice and the mapped
source/target hidden states. This wrapper extends wrapper_v3 without changing
the original C2C or v3 entrypoints.
"""

from rosetta.model.wrapper_v3 import RosettaModel as HiddenRosettaModel


class RosettaModel(HiddenRosettaModel):
    @staticmethod
    def _uses_source_kv_and_hidden_states(projector) -> bool:
        return bool(getattr(projector, "uses_source_kv_and_hidden_states", False))

    def _project_with_source(
        self,
        projector,
        source_model_layer_idx: int,
        target_model_layer_idx: int,
        source_kv_cache,
        source_hidden_states,
        target_hidden_states,
        target_kv,
        start: int,
        end: int,
    ):
        if self._uses_source_kv_and_hidden_states(projector):
            source_key_cache, source_value_cache = source_kv_cache[source_model_layer_idx]
            source_kv = (
                source_key_cache[:, :, start:end, :],
                source_value_cache[:, :, start:end, :],
            )
            source_hidden = self._get_hidden_for_layer(
                source_hidden_states,
                source_model_layer_idx,
                end - start,
                "source",
            )
            target_hidden = self._get_hidden_for_layer(
                target_hidden_states,
                target_model_layer_idx,
                end - start,
                "target",
            )
            return projector.forward((source_kv, source_hidden, target_hidden), target_kv)

        return super()._project_with_source(
            projector=projector,
            source_model_layer_idx=source_model_layer_idx,
            target_model_layer_idx=target_model_layer_idx,
            source_kv_cache=source_kv_cache,
            source_hidden_states=source_hidden_states,
            target_hidden_states=target_hidden_states,
            target_kv=target_kv,
            start=start,
            end=end,
        )
