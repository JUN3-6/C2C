import torch

from script.train.generate_router_labels import compute_skip_and_bank_labels


def test_compute_skip_and_bank_labels_prefers_skip_for_non_positive():
    labels = compute_skip_and_bank_labels(
        torch.tensor([-0.1, 0.0, 1e-7], dtype=torch.float32),
        skip_margin=1e-6,
    )
    assert labels["best_action"] == "skip"
    assert labels["binary_target"] == 0
    assert labels["bank_target"] == -1


def test_compute_skip_and_bank_labels_selects_best_bank():
    labels = compute_skip_and_bank_labels(
        torch.tensor([0.01, 0.2, 0.05], dtype=torch.float32),
        skip_margin=1e-6,
    )
    assert labels["best_action"] == "bank_1"
    assert labels["binary_target"] == 1
    assert labels["bank_target"] == 1
