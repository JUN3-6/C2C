"""
C2C v4 projector.

This variant keeps the original C2C KV input path and adds hidden-state
features before the input down-projection:

    source KV flat + target KV flat + source hidden + target hidden
        -> Linear down projection -> original C2C MLP/fusion/gate path
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from rosetta.model.projector import C2CProjector, Projector
from rosetta.utils.registry import (
    capture_init_args,
    get_projector_class,
    load_object,
    register_model,
    save_object,
)


def _nvtx_push(name: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


@register_model
@capture_init_args
class C2CKVHiddenProjectorV4(C2CProjector):
    """
    C2C v4 projector.

    Inputs:
        source_kv: source key/value for the mapped source layer.
        target_kv: receiver key/value for the mapped target layer.
        source_hidden: hidden_states[source_layer].
        target_hidden: hidden_states[target_layer].

    The downstream projection, dynamic weighting, gates, and residual write are
    intentionally the same as the original C2C projector.
    """

    uses_source_hidden_states = True
    uses_target_hidden_states = True
    uses_source_kv_and_hidden_states = True

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        source_num_heads: int = 1,
        target_num_heads: int = 1,
        source_hidden_dim: Optional[int] = None,
        target_hidden_dim: Optional[int] = None,
        intermediate_dim: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.001,
        anneal_steps: int = 1929,
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
    ):
        captured_init_args = dict(getattr(self, "_init_args", {}))

        super().__init__(
            source_dim=source_dim,
            target_dim=target_dim,
            source_num_heads=source_num_heads,
            target_num_heads=target_num_heads,
            intermediate_dim=intermediate_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            initial_temperature=initial_temperature,
            final_temperature=final_temperature,
            anneal_steps=anneal_steps,
            dtype=dtype,
            zero_init=zero_init,
        )

        # C2CProjector is also decorated with capture_init_args. Restore this
        # class' full init args so checkpoint JSON keeps the hidden dimensions.
        self._init_args = captured_init_args

        self.source_hidden_dim = source_hidden_dim if source_hidden_dim is not None else source_dim * source_num_heads
        self.target_hidden_dim = target_hidden_dim if target_hidden_dim is not None else target_dim * target_num_heads
        self.hidden_dim = hidden_dim

        source_kv_dim = source_dim * source_num_heads
        target_kv_dim = target_dim * target_num_heads
        input_dim = source_kv_dim + target_kv_dim + self.source_hidden_dim + self.target_hidden_dim

        self.key_in = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.value_in = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)

    @staticmethod
    def _slice_hidden(hidden_state: Tensor, seq_len: int, name: str) -> Tensor:
        if hidden_state.dim() != 3:
            raise ValueError(f"{name} must have shape (B, N, hidden_size); got {tuple(hidden_state.shape)}")
        if hidden_state.size(1) != seq_len:
            if hidden_state.size(1) < seq_len:
                raise ValueError(f"{name} sequence length is shorter than KV length: {hidden_state.size(1)} < {seq_len}")
            hidden_state = hidden_state[:, -seq_len:, :]
        return hidden_state

    def forward(
        self,
        source_payload,
        target_kv: Tuple[Tensor, Tensor],
        position_ids: Optional[Tensor] = None,
        max_pos: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if not isinstance(source_payload, (tuple, list)) or len(source_payload) != 3:
            raise ValueError("C2CKVHiddenProjectorV4 expects (source_kv, source_hidden, target_hidden)")

        source_kv, source_hidden_state, target_hidden_state = source_payload
        source_key, source_value = source_kv
        target_key, target_value = target_kv

        batch_size, source_heads, seq_len, source_head_dim = source_key.shape
        _, target_heads, target_seq_len, target_head_dim = target_key.shape
        if target_seq_len != seq_len:
            raise ValueError(f"source and target KV lengths must match; got {seq_len} and {target_seq_len}")

        source_hidden_state = self._slice_hidden(source_hidden_state, seq_len, "source_hidden_state")
        target_hidden_state = self._slice_hidden(target_hidden_state, seq_len, "target_hidden_state")

        if source_hidden_state.size(0) != batch_size or target_hidden_state.size(0) != batch_size:
            raise ValueError("source/target hidden batch sizes must match KV batch size")
        if source_hidden_state.size(-1) != self.source_hidden_dim:
            raise ValueError(
                "source hidden size must match source_hidden_dim; "
                f"got {source_hidden_state.size(-1)} and {self.source_hidden_dim}"
            )
        if target_hidden_state.size(-1) != self.target_hidden_dim:
            raise ValueError(
                "target hidden size must match target_hidden_dim; "
                f"got {target_hidden_state.size(-1)} and {self.target_hidden_dim}"
            )

        _nvtx_push("fuser_v4.kv_hidden_concat")
        source_key_flat = source_key.transpose(1, 2).contiguous().view(batch_size, seq_len, source_heads * source_head_dim)
        source_value_flat = source_value.transpose(1, 2).contiguous().view(batch_size, seq_len, source_heads * source_head_dim)
        target_key_flat = target_key.transpose(1, 2).contiguous().view(batch_size, seq_len, target_heads * target_head_dim)
        target_value_flat = target_value.transpose(1, 2).contiguous().view(batch_size, seq_len, target_heads * target_head_dim)

        key_cat = torch.cat(
            [
                source_key_flat.to(dtype=target_key.dtype),
                target_key_flat,
                source_hidden_state.to(dtype=target_key.dtype),
                target_hidden_state.to(dtype=target_key.dtype),
            ],
            dim=-1,
        )
        value_cat = torch.cat(
            [
                source_value_flat.to(dtype=target_value.dtype),
                target_value_flat,
                source_hidden_state.to(dtype=target_value.dtype),
                target_hidden_state.to(dtype=target_value.dtype),
            ],
            dim=-1,
        )

        key_hidden = self.key_in(key_cat)
        value_hidden = self.value_in(value_cat)
        _nvtx_pop()

        _nvtx_push("fuser_v4.projection")
        key_hidden = self.key_mlp1(key_hidden)
        value_hidden = self.value_mlp1(value_hidden)
        _nvtx_pop()

        _nvtx_push("fuser_v4.feature_fusion")
        key_proj_hidden = self.key_proj_out(self.key_proj_mlp2(key_hidden))
        value_proj_hidden = self.value_proj_out(self.value_proj_mlp2(value_hidden))
        projected_key = key_proj_hidden.view(batch_size, seq_len, target_heads, target_head_dim).transpose(1, 2)
        projected_value = value_proj_hidden.view(batch_size, seq_len, target_heads, target_head_dim).transpose(1, 2)
        _nvtx_pop()

        _nvtx_push("fuser_v4.dynamic_weight")
        key_scalar = self.key_scalar_head(self.key_scalar_mlp2(key_hidden))
        value_scalar = self.value_scalar_head(self.value_scalar_mlp2(value_hidden))
        key_scalar = key_scalar.permute(0, 2, 1).unsqueeze(-1)
        value_scalar = value_scalar.permute(0, 2, 1).unsqueeze(-1)
        _nvtx_pop()

        _nvtx_push("fuser_v4.gate")
        key_gate_logit = self.key_gate_logit.view(1, 1, 1, 1)
        value_gate_logit = self.value_gate_logit.view(1, 1, 1, 1)
        if self.training and self.use_gumbel:
            u1 = torch.rand(batch_size, target_heads, seq_len, 1, device=key_gate_logit.device, dtype=key_gate_logit.dtype)
            u2 = torch.rand(batch_size, target_heads, seq_len, 1, device=value_gate_logit.device, dtype=value_gate_logit.dtype)
            g1 = -torch.log(-torch.log(u1 + 1e-20) + 1e-20)
            g2 = -torch.log(-torch.log(u2 + 1e-20) + 1e-20)
            key_gate = torch.sigmoid((key_gate_logit + g1) / self.gate_temperature)
            value_gate = torch.sigmoid((value_gate_logit + g2) / self.gate_temperature)
        else:
            key_gate = (key_gate_logit > 0).float()
            value_gate = (value_gate_logit > 0).float()
        _nvtx_pop()

        _nvtx_push("fuser_v4.dynamic_weight_norm")
        norm_key_scalar = torch.sigmoid(key_scalar / self.scalar_temperature)
        norm_value_scalar = torch.sigmoid(value_scalar / self.scalar_temperature)
        _nvtx_pop()

        _nvtx_push("fuser_v4.write_cache")
        output_key = target_key + key_gate * norm_key_scalar * projected_key
        output_value = target_value + value_gate * norm_value_scalar * projected_value
        _nvtx_pop()

        try:
            self.last_norm_key_scalar = norm_key_scalar.detach().cpu()
            self.last_norm_value_scalar = norm_value_scalar.detach().cpu()
            self.last_key_gate_logit = float(self.key_gate_logit.detach().cpu().item())
            self.last_value_gate_logit = float(self.value_gate_logit.detach().cpu().item())
        except Exception:
            pass

        return output_key, output_value


register_model("C2CProjectorV4")(C2CKVHiddenProjectorV4)
register_model("KVHiddenC2CProjectorV4")(C2CKVHiddenProjectorV4)


def save_projector(obj: Projector, file_path: str) -> None:
    save_object(obj, file_path)


def load_projector(file_path: str, override_args: Optional[dict] = None) -> Projector:
    return load_object(file_path, get_projector_class, override_args)


def create_projector(projector_type: str, **kwargs) -> Projector:
    cls = get_projector_class(projector_type)
    return cls(**kwargs)
