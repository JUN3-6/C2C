import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from rosetta.model.router import SimpleKVRouter, load_router, save_router
from rosetta.model.wrapper import RosettaModel


def _make_cache(num_layers=2, batch_size=2, num_heads=3, seq_len=4, head_dim=5):
    legacy = []
    for layer_idx in range(num_layers):
        base = torch.arange(
            batch_size * num_heads * seq_len * head_dim,
            dtype=torch.float32,
        ).reshape(batch_size, num_heads, seq_len, head_dim)
        legacy.append((base + layer_idx, base + layer_idx + 0.5))
    return DynamicCache.from_legacy_cache(legacy)


def _make_logic_only_model(*, routing_enabled: bool, include_response: bool):
    model = RosettaModel.__new__(RosettaModel)
    nn.Module.__init__(model)
    model.base_model_idx = 0
    model.router = nn.Identity() if routing_enabled else None
    model.include_response = include_response
    model.kv_cache_dict = {
        0: {
            0: object(),
            1: object(),
        }
    }
    return model


def test_simple_kv_router_predict_shapes():
    router = SimpleKVRouter(num_banks=2, input_dim=60, hidden_dim=16)
    base_cache = _make_cache()
    source_cache = _make_cache()

    result = router.predict(base_cache, source_cache)

    assert result["pooled_feature"].shape == (2, 60)
    assert result["binary_logits"].shape == (2,)
    assert result["selection_logits"].shape == (2, 2)
    assert result["should_fuse"].shape == (2,)
    assert result["selected_bank"].shape == (2,)


def test_router_save_and_load_roundtrip(tmp_path):
    router = SimpleKVRouter(num_banks=3, input_dim=60, hidden_dim=12)
    config_path = tmp_path / "router.json"
    weight_path = tmp_path / "router.pt"

    save_router(router, str(config_path))
    torch.save(router.state_dict(), weight_path)

    loaded = load_router(str(config_path))
    loaded.load_state_dict(torch.load(weight_path))

    assert isinstance(loaded, SimpleKVRouter)
    assert loaded.num_banks == 3
    assert loaded.input_dim == 60
    assert loaded.hidden_dim == 12


def test_projector_banks_require_shared_layout():
    model = RosettaModel.__new__(RosettaModel)
    nn.Module.__init__(model)
    model.router = None
    model.model_list = [object(), object()]
    model.projector_dict = {}
    model.projector_bank_dicts = []

    bank0 = {0: {1: {0: [(0, 0)], 1: [(1, 1)]}}}
    bank1 = {0: {1: {0: [(0, 2)], 1: [(1, 3)]}}}
    model.set_projector_banks([bank0, bank1])

    assert len(model.projector_bank_dicts) == 2
    assert model.projector_dict == bank0

    bad_bank = {0: {1: {0: [(0, 4)]}}}
    try:
        model.set_projector_banks([bank0, bad_bank])
    except ValueError as exc:
        assert "same source/target topology" in str(exc)
    else:
        raise AssertionError("Expected mismatched bank layout to raise ValueError")


def test_routing_last_section_fusion_is_prefill_only():
    model = _make_logic_only_model(routing_enabled=True, include_response=False)

    assert model._should_use_last_section_fusion(seqlen=8, source_model_idx=1)
    assert not model._should_use_last_section_fusion(seqlen=1, source_model_idx=1)


def test_non_routing_last_section_fusion_respects_include_response():
    model = _make_logic_only_model(routing_enabled=False, include_response=False)
    assert not model._should_use_last_section_fusion(seqlen=8, source_model_idx=1)

    model.include_response = True
    assert model._should_use_last_section_fusion(seqlen=1, source_model_idx=1)
