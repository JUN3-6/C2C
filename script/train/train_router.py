#!/usr/bin/env python
"""
Train the lightweight C2C router from pooled offline features.

Training now uses a multiclass action objective:
  action 0 -> skip
  action i -> fuse with bank (i - 1), for i in [1, num_banks]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rosetta.model.router import SimpleKVRouter, save_router


def _load_dataset(path: Path):
    payload = torch.load(path, map_location="cpu")
    if "pooled_feature" not in payload:
        raise KeyError(f"Dataset {path} is missing key: pooled_feature")

    has_action = "action_target" in payload
    has_binary_bank = "binary_target" in payload and "bank_target" in payload
    if not has_action and not has_binary_bank:
        raise KeyError(
            f"Dataset {path} must contain `action_target` or both "
            "`binary_target` and `bank_target`."
        )
    return payload


def _optional_int_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().view(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return None


def _infer_num_banks(dataset_dict) -> int:
    if "improvements" in dataset_dict:
        return int(dataset_dict["improvements"].shape[-1])

    candidates = []
    if "bank_target" in dataset_dict:
        bank_target = dataset_dict["bank_target"].long()
        if bank_target.numel() > 0:
            candidates.append(max(0, int(bank_target.max().item()) + 1))
    if "action_target" in dataset_dict:
        action_target = dataset_dict["action_target"].long()
        if action_target.numel() > 0:
            candidates.append(max(0, int(action_target.max().item())))

    if not candidates:
        raise ValueError("Cannot infer num_banks from dataset.")
    return max(1, max(candidates))


def _ensure_improvements(dataset_dict, num_banks: int) -> bool:
    if "improvements" in dataset_dict:
        return True
    if "ce_receiver" in dataset_dict and "ce_fusions" in dataset_dict:
        ce_receiver = dataset_dict["ce_receiver"].float()
        ce_fusions = dataset_dict["ce_fusions"].float()
        dataset_dict["improvements"] = ce_receiver.unsqueeze(-1) - ce_fusions
        return True

    num_examples = int(dataset_dict["pooled_feature"].shape[0])
    dataset_dict["improvements"] = torch.zeros(num_examples, num_banks, dtype=torch.float32)
    return False


def _ensure_action_targets(dataset_dict, num_banks: int) -> None:
    if "action_target" in dataset_dict:
        action_target = dataset_dict["action_target"].long()
    elif "binary_target" in dataset_dict and "bank_target" in dataset_dict:
        binary_target = dataset_dict["binary_target"].long()
        bank_target = dataset_dict["bank_target"].long()
        action_target = torch.zeros_like(binary_target)
        fuse_mask = binary_target > 0
        if fuse_mask.any():
            fuse_bank = bank_target[fuse_mask]
            if (fuse_bank < 0).any() or (fuse_bank >= num_banks).any():
                raise ValueError("bank_target out of range for fused samples.")
            action_target[fuse_mask] = fuse_bank + 1
    else:
        improvements = dataset_dict["improvements"].float()
        best_bank = torch.argmax(improvements, dim=-1)
        best_gain = improvements.gather(1, best_bank.unsqueeze(-1)).squeeze(-1)
        should_fuse = best_gain > 0.0
        action_target = torch.where(
            should_fuse,
            best_bank + 1,
            torch.zeros_like(best_bank),
        ).long()

    if action_target.numel() == 0:
        raise ValueError("action_target is empty.")
    if int(action_target.min().item()) < 0:
        raise ValueError("action_target must be >= 0.")
    if int(action_target.max().item()) > num_banks:
        raise ValueError(
            f"action_target max={int(action_target.max().item())} exceeds num_banks={num_banks}"
        )

    dataset_dict["action_target"] = action_target.long()
    dataset_dict["binary_target"] = (action_target > 0).long()
    dataset_dict["bank_target"] = torch.where(
        action_target > 0,
        action_target - 1,
        torch.full_like(action_target, -1),
    ).long()


def _ensure_action_target_probs(
    dataset_dict,
    num_actions: int,
    mode: str,
    temperature: float,
    ce_tiebreaker_alpha: float,
) -> bool:
    if temperature <= 0:
        raise ValueError("--action-utility-temperature must be > 0")
    if ce_tiebreaker_alpha < 0:
        raise ValueError("--action-utility-ce-tiebreaker-alpha must be >= 0")

    mode = mode.lower()
    if mode == "hard":
        dataset_dict["_use_soft_action_targets"] = False
        if "action_utilities" not in dataset_dict:
            skip_util = torch.zeros(
                dataset_dict["improvements"].shape[0],
                1,
                dtype=torch.float32,
            )
            dataset_dict["action_utilities"] = torch.cat(
                [skip_util, dataset_dict["improvements"].float()],
                dim=-1,
            )
        return False

    if mode == "skip_detector":
        if "action_utilities" not in dataset_dict:
            if "receiver_correct" not in dataset_dict or "fusion_corrects" not in dataset_dict:
                raise KeyError(
                    "skip_detector mode requires `action_utilities` or "
                    "(`receiver_correct`, `fusion_corrects`) in the dataset."
                )
            dataset_dict["action_utilities"] = torch.cat(
                [
                    dataset_dict["receiver_correct"].float().unsqueeze(-1),
                    dataset_dict["fusion_corrects"].float(),
                ],
                dim=-1,
            )

        utilities = dataset_dict["action_utilities"].float()
        if utilities.shape[-1] != num_actions:
            raise ValueError(
                f"action_utilities last dim={utilities.shape[-1]} does not match "
                f"num_actions={num_actions}"
            )

        receiver_utility = utilities[:, 0]
        fusion_utility, best_bank = utilities[:, 1:].max(dim=-1)
        skip_positive = receiver_utility > fusion_utility
        # Ties default to fuse so the deployed policy remains all-fusion unless
        # the detector is confident. With decision_margin weighting and
        # --action-sample-weight-min=0, exact ties receive zero loss weight.
        action_target = torch.where(
            skip_positive,
            torch.zeros_like(best_bank),
            best_bank + 1,
        ).long()
        dataset_dict["action_target"] = action_target
        dataset_dict["binary_target"] = (action_target > 0).long()
        dataset_dict["bank_target"] = torch.where(
            action_target > 0,
            action_target - 1,
            torch.full_like(action_target, -1),
        ).long()
        dataset_dict["_use_soft_action_targets"] = False
        return False

    if mode != "accuracy_utility":
        raise ValueError(
            f"Unsupported action_target_mode={mode}. "
            "Use one of: hard, accuracy_utility, skip_detector."
        )
    if "action_utilities" not in dataset_dict:
        if "receiver_correct" not in dataset_dict or "fusion_corrects" not in dataset_dict:
            raise KeyError(
                "accuracy_utility mode requires `action_utilities` or "
                "(`receiver_correct`, `fusion_corrects`) in the dataset."
            )
        dataset_dict["action_utilities"] = torch.cat(
            [
                dataset_dict["receiver_correct"].float().unsqueeze(-1),
                dataset_dict["fusion_corrects"].float(),
            ],
            dim=-1,
        )

    utilities = dataset_dict["action_utilities"].float()
    if utilities.shape[-1] != num_actions:
        raise ValueError(
            f"action_utilities last dim={utilities.shape[-1]} does not match "
            f"num_actions={num_actions}"
        )

    if ce_tiebreaker_alpha > 0:
        if "ce_receiver" not in dataset_dict or "ce_fusions" not in dataset_dict:
            raise KeyError(
                "CE tie-breaker requires `ce_receiver` and `ce_fusions` in the dataset."
            )
        ce_values = torch.cat(
            [
                dataset_dict["ce_receiver"].float().unsqueeze(-1),
                dataset_dict["ce_fusions"].float(),
            ],
            dim=-1,
        )
        utilities = utilities - ce_tiebreaker_alpha * ce_values
        dataset_dict["action_utilities"] = utilities

    target_probs = torch.softmax(utilities / temperature, dim=-1)
    dataset_dict["action_target_probs"] = target_probs.float()
    dataset_dict["_use_soft_action_targets"] = True

    hard_target = torch.argmax(utilities, dim=-1).long()
    dataset_dict["action_target"] = hard_target
    dataset_dict["binary_target"] = (hard_target > 0).long()
    dataset_dict["bank_target"] = torch.where(
        hard_target > 0,
        hard_target - 1,
        torch.full_like(hard_target, -1),
    ).long()
    return True


def _compute_action_class_weights(
    action_target: torch.Tensor,
    num_actions: int,
    mode: str,
    effective_num_beta: float,
    max_weight: float,
    skip_multiplier: float,
) -> Optional[torch.Tensor]:
    mode = mode.lower()
    if mode == "none":
        return None

    counts = torch.bincount(action_target.long(), minlength=num_actions).float()
    present_mask = counts > 0

    if mode == "inverse":
        weights = counts.sum() / counts.clamp_min(1.0)
    elif mode == "effective_num":
        beta = effective_num_beta
        if beta <= 0.0 or beta >= 1.0:
            raise ValueError(
                f"effective_num_beta must be in (0, 1), got {effective_num_beta}"
            )
        effective_num = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
        weights = 1.0 / effective_num.clamp_min(1e-12)
    else:
        raise ValueError(
            f"Unsupported action_class_weight_mode={mode}. "
            "Use one of: none, inverse, effective_num."
        )

    # Keep zero-count classes from exploding and normalize for stable scale.
    weights = torch.where(present_mask, weights, torch.zeros_like(weights))
    present_weights = weights[present_mask]
    if present_weights.numel() == 0:
        return None
    weights = weights / present_weights.mean().clamp_min(1e-8)

    if skip_multiplier != 1.0:
        weights[0] = weights[0] * skip_multiplier

    if max_weight > 0:
        weights = torch.clamp(weights, min=0.0, max=max_weight)

    # Re-normalize after optional skip scaling / clipping.
    present_weights = weights[present_mask]
    weights = weights / present_weights.mean().clamp_min(1e-8)
    return weights.float()


def _make_loader(dataset_dict, batch_size: int, shuffle: bool, num_actions: int) -> DataLoader:
    if dataset_dict.get("_use_soft_action_targets", False):
        action_target_probs = dataset_dict["action_target_probs"].float()
    else:
        action_target_probs = torch.empty(
            dataset_dict["pooled_feature"].shape[0],
            0,
            dtype=torch.float32,
        )
    action_utilities = dataset_dict.get("action_utilities")
    if action_utilities is None:
        skip_util = torch.zeros(
            dataset_dict["improvements"].shape[0],
            1,
            dtype=torch.float32,
        )
        action_utilities = torch.cat(
            [skip_util, dataset_dict["improvements"].float()],
            dim=-1,
        )
    dataset = TensorDataset(
        dataset_dict["pooled_feature"].float(),
        dataset_dict["action_target"].long(),
        dataset_dict["binary_target"].long(),
        dataset_dict["bank_target"].long(),
        dataset_dict["improvements"].float(),
        action_target_probs.float(),
        action_utilities.float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _expected_gain(output, improvements, selection_temperature: float):
    if selection_temperature <= 0:
        raise ValueError(f"selection_temperature must be > 0, got {selection_temperature}")

    action_logits = output.action_logits / selection_temperature
    action_probs = torch.softmax(action_logits, dim=-1)

    per_bank_gain = improvements.to(device=action_probs.device, dtype=action_probs.dtype)
    skip_gain = torch.zeros(
        per_bank_gain.size(0),
        1,
        device=per_bank_gain.device,
        dtype=per_bank_gain.dtype,
    )
    action_gain = torch.cat([skip_gain, per_bank_gain], dim=-1)
    expected_routed_gain = torch.sum(action_probs * action_gain, dim=-1)
    return expected_routed_gain


def _resolve_gain_loss_weight_for_epoch(
    *,
    base_gain_loss_weight: float,
    epoch_index: int,
    gain_ce_only_epochs: int,
    gain_ramp_epochs: int,
) -> float:
    if base_gain_loss_weight <= 0.0:
        return 0.0
    if gain_ce_only_epochs < 0:
        raise ValueError(f"gain_ce_only_epochs must be >= 0, got {gain_ce_only_epochs}")
    if gain_ramp_epochs < 0:
        raise ValueError(f"gain_ramp_epochs must be >= 0, got {gain_ramp_epochs}")

    epoch_one_based = epoch_index + 1
    if epoch_one_based <= gain_ce_only_epochs:
        return 0.0
    if gain_ramp_epochs == 0:
        return base_gain_loss_weight

    ramp_step = epoch_one_based - gain_ce_only_epochs
    ramp_ratio = min(max(ramp_step / float(gain_ramp_epochs), 0.0), 1.0)
    return float(base_gain_loss_weight * ramp_ratio)


def _compute_losses(
    router,
    pooled_feature,
    action_target,
    improvements,
    selection_temperature: float,
    action_class_weights: Optional[torch.Tensor] = None,
    action_label_smoothing: float = 0.0,
    action_sample_weight_mode: str = "none",
    action_sample_weight_scale: float = 1.0,
    action_sample_weight_min: float = 0.2,
    action_decision_balance_power: float = 1.0,
    action_target_probs: Optional[torch.Tensor] = None,
    action_utilities: Optional[torch.Tensor] = None,
):
    output = router.forward_features(pooled_feature)
    ce_weights = None
    if action_class_weights is not None:
        ce_weights = action_class_weights.to(
            device=output.action_logits.device,
            dtype=output.action_logits.dtype,
        )
    if action_target_probs is None:
        action_loss_per_sample = F.cross_entropy(
            output.action_logits,
            action_target,
            weight=ce_weights,
            label_smoothing=action_label_smoothing,
            reduction="none",
        )
    else:
        if action_class_weights is not None:
            raise ValueError("action_class_weights are not supported with soft action targets.")
        if action_label_smoothing > 0:
            raise ValueError("action_label_smoothing is not supported with soft action targets.")
        target_probs = action_target_probs.to(
            device=output.action_logits.device,
            dtype=output.action_logits.dtype,
        )
        log_probs = torch.log_softmax(output.action_logits, dim=-1)
        action_loss_per_sample = -(target_probs * log_probs).sum(dim=-1)
    if action_sample_weight_mode == "none":
        action_loss = action_loss_per_sample.mean()
    elif action_sample_weight_mode in {"decision_margin", "decision_margin_balanced"}:
        if action_sample_weight_scale <= 0.0:
            raise ValueError("--action-sample-weight-scale must be > 0")
        if action_sample_weight_min < 0.0 or action_sample_weight_min > 1.0:
            raise ValueError("--action-sample-weight-min must be in [0, 1]")
        if action_decision_balance_power < 0.0:
            raise ValueError("--action-decision-balance-power must be >= 0")

        if action_utilities is not None:
            # For accuracy-utility training this makes both-correct/both-wrong
            # ties low-weight, and focuses learning on samples where skip/fuse
            # changes the final option correctness.
            action_values = action_utilities.to(
                device=output.action_logits.device,
                dtype=output.action_logits.dtype,
            )
        else:
            skip_values = torch.zeros(
                improvements.size(0),
                1,
                device=improvements.device,
                dtype=improvements.dtype,
            )
            action_values = torch.cat([skip_values, improvements], dim=1)
        target_values = action_values.gather(1, action_target.unsqueeze(-1)).squeeze(-1)
        not_target = torch.ones_like(action_values, dtype=torch.bool)
        not_target.scatter_(1, action_target.unsqueeze(-1), False)
        runner_up_values = action_values.masked_fill(~not_target, -torch.inf).max(dim=1).values
        decision_margin = (target_values - runner_up_values).clamp_min(0.0)
        confidence = (decision_margin / action_sample_weight_scale).clamp(0.0, 1.0)
        sample_weights = action_sample_weight_min + (1.0 - action_sample_weight_min) * confidence

        if action_sample_weight_mode == "decision_margin_balanced":
            decision_mask = decision_margin > 0
            if decision_mask.any():
                decision_targets = action_target[decision_mask]
                counts = torch.bincount(
                    decision_targets,
                    minlength=action_values.size(1),
                ).to(device=sample_weights.device, dtype=sample_weights.dtype)
                present = counts > 0
                class_weights = torch.ones_like(counts)
                present_count = present.float().sum().clamp_min(1.0)
                class_weights[present] = counts[present].sum() / (
                    present_count * counts[present].clamp_min(1.0)
                )
                class_weights = class_weights.pow(action_decision_balance_power)
                sample_weights = sample_weights * torch.where(
                    decision_mask,
                    class_weights.gather(0, action_target),
                    torch.ones_like(sample_weights),
                )

        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
        action_loss = (action_loss_per_sample * sample_weights).mean()
    else:
        raise ValueError(
            f"Unsupported action_sample_weight_mode={action_sample_weight_mode}. "
            "Use one of: none, decision_margin, decision_margin_balanced."
        )

    expected_routed_gain = _expected_gain(
        output=output,
        improvements=improvements,
        selection_temperature=selection_temperature,
    )
    gain_loss = -expected_routed_gain.mean()
    return output, action_loss, gain_loss, expected_routed_gain


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "constant":
        return None

    if total_steps <= 0:
        return None
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if min_lr_ratio <= 0.0 or min_lr_ratio > 1.0:
        raise ValueError(f"min_lr_ratio must be in (0, 1], got {min_lr_ratio}")
    if scheduler_type not in {"linear", "cosine"}:
        raise ValueError(
            f"Unsupported lr_scheduler_type={scheduler_type}. "
            "Use one of: constant, linear, cosine."
        )

    warmup_steps = min(warmup_steps, total_steps)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        decay_total = max(1, total_steps - warmup_steps)
        decay_step = min(max(current_step - warmup_steps, 0), decay_total)
        progress = float(decay_step) / float(decay_total)

        if scheduler_type == "linear":
            return (1.0 - progress) * (1.0 - min_lr_ratio) + min_lr_ratio

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _evaluate(
    router,
    loader,
    device,
    action_loss_weight: float,
    gain_loss_weight: float,
    fuse_threshold: float,
    selection_temperature: float,
    action_class_weights: Optional[torch.Tensor] = None,
    decision_policy: str = "argmax",
    skip_threshold: float = 0.5,
    sweep_skip_threshold: bool = False,
    skip_threshold_steps: int = 101,
):
    router.eval()
    total_loss = 0.0
    total_action_loss = 0.0
    total_gain_loss = 0.0
    total_expected_gain = 0.0
    total_examples = 0
    correct_action = 0
    correct_binary = 0
    correct_bank = 0
    bank_examples = 0
    total_routed_gain = 0.0
    total_oracle_gain = 0.0
    total_fuse = 0
    total_harm = 0
    total_wrong_bank_gain_delta = 0.0
    total_routed_utility = 0.0
    total_oracle_utility = 0.0
    total_receiver_utility = 0.0
    total_all_fuse_utility = 0.0
    sweep_skip_scores = []
    sweep_pred_banks = []
    sweep_action_utilities = []
    sweep_improvements = []
    sweep_action_targets = []

    with torch.no_grad():
        for (
            pooled_feature,
            action_target,
            binary_target,
            bank_target,
            improvements,
            action_target_probs,
            action_utilities,
        ) in loader:
            pooled_feature = pooled_feature.to(device)
            action_target = action_target.to(device)
            binary_target = binary_target.to(device)
            bank_target = bank_target.to(device)
            improvements = improvements.to(device)
            action_target_probs = action_target_probs.to(device)
            action_utilities = action_utilities.to(device)
            if action_target_probs.numel() == 0:
                action_target_probs = None

            output, action_loss, gain_loss, expected_routed_gain = _compute_losses(
                router,
                pooled_feature,
                action_target,
                improvements,
                selection_temperature=selection_temperature,
                action_class_weights=action_class_weights,
                action_target_probs=action_target_probs,
            )
            loss = (
                action_loss_weight * action_loss
                + gain_loss_weight * gain_loss
            )

            batch_size = pooled_feature.size(0)
            total_loss += float(loss.item()) * batch_size
            total_action_loss += float(action_loss.item()) * batch_size
            total_gain_loss += float(gain_loss.item()) * batch_size
            total_expected_gain += float(expected_routed_gain.sum().item())
            total_examples += batch_size

            pred_bank = torch.argmax(output.selection_logits, dim=-1)
            if decision_policy == "skip_threshold":
                pred_action = torch.where(
                    output.skip_probability > skip_threshold,
                    torch.zeros_like(pred_bank),
                    pred_bank + 1,
                )
            elif decision_policy == "argmax":
                pred_action = torch.argmax(output.action_logits, dim=-1)
            else:
                raise ValueError(
                    f"Unsupported decision_policy={decision_policy}. "
                    "Use one of: argmax, skip_threshold."
                )
            pred_fuse = pred_action > 0
            pred_bank = torch.where(
                pred_fuse,
                pred_action - 1,
                torch.zeros_like(pred_action),
            )
            if sweep_skip_threshold:
                sweep_skip_scores.append(output.skip_probability.detach().cpu())
                sweep_pred_banks.append(torch.argmax(output.selection_logits, dim=-1).detach().cpu())
                sweep_action_utilities.append(action_utilities.detach().cpu())
                sweep_improvements.append(improvements.detach().cpu())
                sweep_action_targets.append(action_target.detach().cpu())

            correct_action += int((pred_action == action_target).sum().item())
            binary_pred = pred_fuse.long()
            correct_binary += int((binary_pred == binary_target).sum().item())

            positive_mask = bank_target >= 0
            if positive_mask.any():
                correct_bank += int((pred_bank[positive_mask] == bank_target[positive_mask]).sum().item())
                bank_examples += int(positive_mask.sum().item())

            pred_gain_if_fuse = improvements.gather(1, pred_bank.unsqueeze(-1)).squeeze(-1)
            routed_gain = torch.where(
                pred_fuse,
                pred_gain_if_fuse,
                torch.zeros_like(pred_gain_if_fuse),
            )
            oracle_gain = improvements.max(dim=1).values.clamp_min(0.0)
            wrong_bank_gain_delta = torch.where(
                pred_fuse,
                (oracle_gain - pred_gain_if_fuse).clamp_min(0.0),
                torch.zeros_like(pred_gain_if_fuse),
            )

            total_routed_gain += float(routed_gain.sum().item())
            total_oracle_gain += float(oracle_gain.sum().item())
            total_fuse += int(pred_fuse.sum().item())
            total_harm += int((pred_fuse & (pred_gain_if_fuse < 0)).sum().item())
            total_wrong_bank_gain_delta += float(wrong_bank_gain_delta.sum().item())
            pred_utility = action_utilities.gather(1, pred_action.unsqueeze(-1)).squeeze(-1)
            total_routed_utility += float(pred_utility.sum().item())
            total_oracle_utility += float(action_utilities.max(dim=1).values.sum().item())
            total_receiver_utility += float(action_utilities[:, 0].sum().item())
            if action_utilities.size(1) > 1:
                total_all_fuse_utility += float(action_utilities[:, 1:].max(dim=1).values.sum().item())

    fuse_rate = total_fuse / max(total_examples, 1)
    skip_rate = 1.0 - fuse_rate
    gain_capture = total_routed_gain / max(total_oracle_gain, 1e-8)

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "action_loss": total_action_loss / max(total_examples, 1),
        "gain_loss": total_gain_loss / max(total_examples, 1),
        "mean_expected_gain": total_expected_gain / max(total_examples, 1),
        "action_acc": correct_action / max(total_examples, 1),
        "binary_acc": correct_binary / max(total_examples, 1),
        "bank_acc": correct_bank / max(bank_examples, 1),
        "mean_routed_gain": total_routed_gain / max(total_examples, 1),
        "mean_oracle_gain": total_oracle_gain / max(total_examples, 1),
        "gain_capture": gain_capture,
        "harm_rate": total_harm / max(total_examples, 1),
        "harm_given_fuse_rate": total_harm / max(total_fuse, 1),
        "fuse_rate": fuse_rate,
        "skip_rate": skip_rate,
        "wrong_bank_gain_delta": total_wrong_bank_gain_delta / max(total_examples, 1),
        "mean_routed_utility": total_routed_utility / max(total_examples, 1),
        "mean_oracle_utility": total_oracle_utility / max(total_examples, 1),
        "mean_receiver_utility": total_receiver_utility / max(total_examples, 1),
        "mean_all_fuse_utility": total_all_fuse_utility / max(total_examples, 1),
    }

    if sweep_skip_threshold and sweep_skip_scores:
        if skip_threshold_steps <= 1:
            raise ValueError("--skip-threshold-steps must be > 1 when threshold sweep is enabled.")
        skip_scores = torch.cat(sweep_skip_scores, dim=0).float()
        pred_banks = torch.cat(sweep_pred_banks, dim=0).long()
        action_utilities_all = torch.cat(sweep_action_utilities, dim=0).float()
        improvements_all = torch.cat(sweep_improvements, dim=0).float()
        action_targets_all = torch.cat(sweep_action_targets, dim=0).long()

        best_utility = float("-inf")
        best_threshold = 1.0
        best_skip_rate = 0.0
        best_gain = 0.0
        best_acc = 0.0
        for threshold in torch.linspace(0.0, 1.0, steps=skip_threshold_steps):
            pred_skip = skip_scores > float(threshold.item())
            pred_action = torch.where(
                pred_skip,
                torch.zeros_like(pred_banks),
                pred_banks + 1,
            )
            pred_utility = action_utilities_all.gather(
                1,
                pred_action.unsqueeze(-1),
            ).squeeze(-1)
            pred_gain_if_fuse = improvements_all.gather(
                1,
                pred_banks.unsqueeze(-1),
            ).squeeze(-1)
            routed_gain = torch.where(
                pred_skip,
                torch.zeros_like(pred_gain_if_fuse),
                pred_gain_if_fuse,
            )
            mean_utility = float(pred_utility.mean().item())
            if mean_utility > best_utility:
                best_utility = mean_utility
                best_threshold = float(threshold.item())
                best_skip_rate = float(pred_skip.float().mean().item())
                best_gain = float(routed_gain.mean().item())
                best_acc = float((pred_action == action_targets_all).float().mean().item())

        metrics.update(
            {
                "threshold_sweep_mean_routed_utility": best_utility,
                "threshold_sweep_skip_threshold": best_threshold,
                "threshold_sweep_skip_rate": best_skip_rate,
                "threshold_sweep_mean_routed_gain": best_gain,
                "threshold_sweep_action_acc": best_acc,
            }
        )

    return metrics


def _maybe_init_wandb(
    args,
    train_dataset,
    val_dataset,
    num_banks: int,
    pooled_feature_dim: int,
    output_dir: Path,
    action_class_weights: Optional[torch.Tensor],
):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging was requested, but the `wandb` package is not installed."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.wandb_run_name or output_dir.name

    run_config = {
        "train_data": str(args.train_data),
        "val_data": str(args.val_data) if args.val_data else None,
        "num_train_examples": int(train_dataset["pooled_feature"].shape[0]),
        "num_val_examples": int(val_dataset["pooled_feature"].shape[0]) if val_dataset is not None else 0,
        "pooled_feature_dim": pooled_feature_dim,
        "num_banks": num_banks,
        "num_actions": num_banks + 1,
        "router_feature_source": args.router_feature_source,
        "option_response_text": train_dataset.get("option_response_text"),
        "hidden_dim": args.hidden_dim,
        "token_mlp_layers": args.token_mlp_layers,
        "dropout": args.dropout,
        "input_layer_norm": args.input_layer_norm,
        "input_standardization": args.input_standardization,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "lr_warmup_ratio": args.lr_warmup_ratio,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_min_ratio": args.lr_min_ratio,
        "action_loss_weight": args.action_loss_weight,
        "action_label_smoothing": args.action_label_smoothing,
        "action_sample_weight_mode": args.action_sample_weight_mode,
        "action_sample_weight_scale": args.action_sample_weight_scale,
        "action_sample_weight_min": args.action_sample_weight_min,
        "action_decision_balance_power": args.action_decision_balance_power,
        "gain_loss_weight": args.gain_loss_weight,
        "gain_ce_only_epochs": args.gain_ce_only_epochs,
        "gain_ramp_epochs": args.gain_ramp_epochs,
        "fuse_threshold": args.fuse_threshold,
        "selection_temperature": args.selection_temperature,
        "decision_policy": args.decision_policy,
        "skip_threshold": args.skip_threshold,
        "sweep_skip_threshold": args.sweep_skip_threshold,
        "skip_threshold_steps": args.skip_threshold_steps,
        "best_metric": args.best_metric,
        "device": str(args.device),
        "log_every": args.log_every,
        "training_objective": "multiclass_action_ce_plus_gain",
        "action_class_weight_mode": args.action_class_weight_mode,
        "action_target_mode": args.action_target_mode,
        "action_utility_temperature": args.action_utility_temperature,
        "action_utility_ce_tiebreaker_alpha": args.action_utility_ce_tiebreaker_alpha,
        "action_class_effective_num_beta": args.action_class_effective_num_beta,
        "action_class_max_weight": args.action_class_max_weight,
        "action_class_skip_multiplier": args.action_class_skip_multiplier,
        "wandb_val_metrics_mode": args.wandb_val_metrics_mode,
        "action_class_weights": (
            action_class_weights.tolist() if action_class_weights is not None else None
        ),
    }

    wandb.init(
        project=args.wandb_project or "router_training",
        entity=args.wandb_entity or None,
        name=run_name,
        mode=args.wandb_mode,
        tags=args.wandb_tags or None,
        dir=str(output_dir),
        config=run_config,
    )
    return wandb


def _build_val_wandb_payload(val_metrics: dict, mode: str) -> dict:
    if mode == "compact":
        # Keep only val metrics that are easy to compare with train metrics
        # plus skip rate (requested signal).
        metric_keys = [
            "loss",
            "action_loss",
            "gain_loss",
            "mean_expected_gain",
            "skip_rate",
            "mean_routed_utility",
            "mean_oracle_utility",
            "mean_receiver_utility",
            "mean_all_fuse_utility",
            "threshold_sweep_mean_routed_utility",
            "threshold_sweep_skip_threshold",
            "threshold_sweep_skip_rate",
        ]
    elif mode == "full":
        metric_keys = [
            "loss",
            "action_loss",
            "gain_loss",
            "mean_expected_gain",
            "action_acc",
            "binary_acc",
            "bank_acc",
            "mean_routed_gain",
            "mean_oracle_gain",
            "gain_capture",
            "harm_rate",
            "harm_given_fuse_rate",
            "fuse_rate",
            "skip_rate",
            "wrong_bank_gain_delta",
            "mean_routed_utility",
            "mean_oracle_utility",
            "mean_receiver_utility",
            "mean_all_fuse_utility",
            "threshold_sweep_mean_routed_utility",
            "threshold_sweep_skip_threshold",
            "threshold_sweep_skip_rate",
            "threshold_sweep_mean_routed_gain",
            "threshold_sweep_action_acc",
        ]
    else:
        raise ValueError(
            f"Unsupported wandb_val_metrics_mode={mode}. "
            "Use one of: compact, full."
        )

    return {
        f"val/{key}": val_metrics[key]
        for key in metric_keys
        if key in val_metrics
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True, help="Training dataset (.pt).")
    parser.add_argument("--val-data", help="Optional validation dataset (.pt).")
    parser.add_argument("--output-dir", required=True, help="Directory to save router artifacts.")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--router-feature-source",
        default="auto",
        choices=[
            "auto",
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
        ],
        help=(
            "Feature source that router should expect at inference time. "
            "`auto` uses dataset['feature_source'] when present, else `kv`."
        ),
    )
    parser.add_argument("--token-mlp-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--input-layer-norm",
        action="store_true",
        help="Apply LayerNorm to pooled router features before the MLP.",
    )
    parser.add_argument(
        "--input-standardization",
        action="store_true",
        help=(
            "Standardize pooled router features with train-set per-dimension "
            "mean/std stored in the router checkpoint."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--lr-scheduler-type",
        default="constant",
        choices=["constant", "linear", "cosine"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--lr-warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio over total training steps (ignored if --lr-warmup-steps >= 0).",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=-1,
        help="Explicit warmup steps override. Use -1 to derive from --lr-warmup-ratio.",
    )
    parser.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.1,
        help="Final LR ratio relative to base LR for linear/cosine schedulers.",
    )
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--action-target-mode",
        default="hard",
        choices=["hard", "accuracy_utility", "skip_detector"],
        help=(
            "`hard` trains on action_target CE. `accuracy_utility` builds soft "
            "targets from [receiver_correct, fusion_corrects], matching option "
            "argmax accuracy instead of CE improvement. `skip_detector` treats "
            "receiver-only-correct examples as skip positives, fusion-only-correct "
            "examples as fuse positives, and leaves ties for sample weighting."
        ),
    )
    parser.add_argument(
        "--action-utility-temperature",
        type=float,
        default=0.25,
        help="Softmax temperature for accuracy utility targets.",
    )
    parser.add_argument(
        "--action-utility-ce-tiebreaker-alpha",
        type=float,
        default=0.0,
        help="Optional small CE tie-breaker alpha added as utility -= alpha * CE.",
    )
    parser.add_argument(
        "--action-label-smoothing",
        type=float,
        default=0.0,
        help=(
            "Label smoothing for the training action CE loss. Validation loss "
            "remains hard-label CE so runs stay comparable."
        ),
    )
    parser.add_argument(
        "--action-sample-weight-mode",
        default="none",
        choices=["none", "decision_margin", "decision_margin_balanced"],
        help=(
            "Optional train-only per-sample weighting for action CE. "
            "`decision_margin` downweights ambiguous samples whose best action "
            "barely beats the runner-up action value. With accuracy_utility "
            "targets, this uses [receiver_correct, fusion_corrects], so "
            "both-correct/both-wrong ties get low weight. "
            "`decision_margin_balanced` additionally balances decisive action "
            "classes within each batch."
        ),
    )
    parser.add_argument(
        "--action-sample-weight-scale",
        type=float,
        default=1.0,
        help=(
            "Decision-margin value mapped to full sample weight when "
            "--action-sample-weight-mode=decision_margin."
        ),
    )
    parser.add_argument(
        "--action-sample-weight-min",
        type=float,
        default=0.2,
        help="Minimum normalized confidence before batch mean re-normalization.",
    )
    parser.add_argument(
        "--action-decision-balance-power",
        type=float,
        default=1.0,
        help=(
            "Strength for `decision_margin_balanced` class balancing. "
            "0 disables the class-balance multiplier, 1 applies full inverse "
            "frequency, and values in between apply a softer correction."
        ),
    )
    parser.add_argument(
        "--selection-loss-weight",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--gain-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--action-class-weight-mode",
        default="none",
        choices=["none", "inverse", "effective_num"],
        help=(
            "Class reweighting mode for multiclass action CE. "
            "`inverse` uses inverse frequency, `effective_num` uses Class-Balanced Loss weighting."
        ),
    )
    parser.add_argument(
        "--action-class-effective-num-beta",
        type=float,
        default=0.9999,
        help="Beta for effective_num weighting mode (must be in (0, 1)).",
    )
    parser.add_argument(
        "--action-class-max-weight",
        type=float,
        default=20.0,
        help="Optional max clip for action class weights. Use <=0 to disable clipping.",
    )
    parser.add_argument(
        "--action-class-skip-multiplier",
        type=float,
        default=1.0,
        help="Optional extra multiplier applied to skip class (action=0).",
    )
    parser.add_argument(
        "--gain-ce-only-epochs",
        type=int,
        default=4,
        help="Initial epochs where gain loss is disabled (CE-only stabilization).",
    )
    parser.add_argument(
        "--gain-ramp-epochs",
        type=int,
        default=6,
        help="Epochs to linearly ramp gain loss weight from 0 to --gain-loss-weight.",
    )
    parser.add_argument("--fuse-threshold", type=float, default=0.5)
    parser.add_argument(
        "--decision-policy",
        default="argmax",
        choices=["argmax", "skip_threshold"],
        help=(
            "Validation/deployed router policy. `skip_threshold` keeps fusion as "
            "default and skips only when p(skip) > --skip-threshold."
        ),
    )
    parser.add_argument(
        "--skip-threshold",
        type=float,
        default=0.5,
        help="Initial/fixed skip probability threshold for --decision-policy=skip_threshold.",
    )
    parser.add_argument(
        "--sweep-skip-threshold",
        action="store_true",
        help="Sweep skip thresholds on validation and report the best routed utility.",
    )
    parser.add_argument(
        "--skip-threshold-steps",
        type=int,
        default=201,
        help="Number of evenly spaced thresholds in [0, 1] for validation sweep.",
    )
    parser.add_argument("--selection-temperature", type=float, default=1.0)
    parser.add_argument(
        "--best-metric",
        default="auto",
        choices=[
            "auto",
            "loss",
            "mean_routed_gain",
            "gain_capture",
            "action_acc",
            "mean_routed_utility",
            "threshold_sweep_mean_routed_utility",
        ],
        help=(
            "Metric for checkpoint selection. "
            "`auto` uses `mean_routed_gain` when validation improvements exist, else `loss`."
        ),
    )
    parser.add_argument("--log-every", type=int, default=10, help="W&B step logging interval.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", help="W&B project name.")
    parser.add_argument("--wandb-entity", help="W&B entity/team name.")
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode. Use `online` for live dashboards or `offline` for later sync.",
    )
    parser.add_argument("--wandb-run-name", help="Optional explicit W&B run name.")
    parser.add_argument(
        "--wandb-val-metrics-mode",
        default="compact",
        choices=["compact", "full"],
        help=(
            "Which validation metrics to log to W&B. "
            "`compact` logs only val/loss, val/action_loss, val/gain_loss, "
            "val/mean_expected_gain, and val/skip_rate."
        ),
    )
    parser.add_argument(
        "--wandb-tag",
        dest="wandb_tags",
        action="append",
        default=[],
        help="Repeatable W&B tag.",
    )
    args = parser.parse_args()

    if args.selection_loss_weight is not None:
        print(
            "Warning: --selection-loss-weight is deprecated in multiclass mode. "
            "Mapping it to --action-loss-weight."
        )
        args.action_loss_weight = args.selection_loss_weight
    if args.lr_warmup_ratio < 0.0 or args.lr_warmup_ratio >= 1.0:
        raise ValueError("--lr-warmup-ratio must be in [0, 1).")
    if args.lr_warmup_steps < -1:
        raise ValueError("--lr-warmup-steps must be >= -1.")
    if args.lr_min_ratio <= 0.0 or args.lr_min_ratio > 1.0:
        raise ValueError("--lr-min-ratio must be in (0, 1].")
    if args.gain_ce_only_epochs < 0:
        raise ValueError("--gain-ce-only-epochs must be >= 0")
    if args.gain_ramp_epochs < 0:
        raise ValueError("--gain-ramp-epochs must be >= 0")
    if args.action_label_smoothing < 0.0 or args.action_label_smoothing >= 1.0:
        raise ValueError("--action-label-smoothing must be in [0, 1).")
    if args.action_sample_weight_scale <= 0.0:
        raise ValueError("--action-sample-weight-scale must be > 0")
    if args.action_sample_weight_min < 0.0 or args.action_sample_weight_min > 1.0:
        raise ValueError("--action-sample-weight-min must be in [0, 1]")
    if args.action_decision_balance_power < 0.0:
        raise ValueError("--action-decision-balance-power must be >= 0")
    if args.action_utility_temperature <= 0.0:
        raise ValueError("--action-utility-temperature must be > 0")
    if args.action_utility_ce_tiebreaker_alpha < 0.0:
        raise ValueError("--action-utility-ce-tiebreaker-alpha must be >= 0")
    if not 0.0 <= args.skip_threshold <= 1.0:
        raise ValueError("--skip-threshold must be in [0, 1]")
    if args.skip_threshold_steps <= 1:
        raise ValueError("--skip-threshold-steps must be > 1")
    if args.action_target_mode == "accuracy_utility":
        if args.action_class_weight_mode != "none":
            raise ValueError("accuracy_utility soft targets require --action-class-weight-mode none")
        if args.action_label_smoothing > 0.0:
            raise ValueError("accuracy_utility soft targets require --action-label-smoothing 0")
    if args.action_target_mode == "skip_detector":
        if args.action_sample_weight_mode == "none":
            print(
                "Warning: skip_detector usually needs "
                "--action-sample-weight-mode decision_margin_balanced so tie "
                "examples can be downweighted."
            )

    train_dataset = _load_dataset(Path(args.train_data))
    val_dataset = _load_dataset(Path(args.val_data)) if args.val_data else None
    dataset_feature_source = str(train_dataset.get("feature_source", "kv"))
    if dataset_feature_source not in {
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
            f"Unsupported dataset feature_source={dataset_feature_source}. "
            "Expected one of: kv, hidden, hidden_binned, projector_in, "
            "projector_in_stats, projector_in_pooled, projector_in_binned, "
            "postfusion_delta_stats, postfusion_probe_logits, "
            "postfusion_option_logits, postfusion_option_logits_hidden_binned, "
            "hybrid_projector_hidden_binned."
        )
    if args.router_feature_source == "auto":
        args.router_feature_source = dataset_feature_source
    if val_dataset is not None and "feature_source" in val_dataset:
        val_feature_source = str(val_dataset["feature_source"])
        if val_feature_source != args.router_feature_source:
            raise ValueError(
                "Train/val router feature_source mismatch: "
                f"train={args.router_feature_source}, val={val_feature_source}"
            )
    print(f"Router feature source: {args.router_feature_source}")
    pooled_feature_dim = int(train_dataset["pooled_feature"].shape[-1])
    if val_dataset is not None:
        val_feature_dim = int(val_dataset["pooled_feature"].shape[-1])
        if val_feature_dim != pooled_feature_dim:
            raise ValueError(
                "Train/val pooled feature dims do not match: "
                f"train={pooled_feature_dim}, val={val_feature_dim}"
            )

    num_banks = _infer_num_banks(train_dataset)
    train_has_improvements = _ensure_improvements(train_dataset, num_banks=num_banks)
    _ensure_action_targets(train_dataset, num_banks=num_banks)
    if val_dataset is not None:
        val_has_improvements = _ensure_improvements(val_dataset, num_banks=num_banks)
        _ensure_action_targets(val_dataset, num_banks=num_banks)
    else:
        val_has_improvements = False

    if args.gain_loss_weight > 0 and not train_has_improvements:
        print(
            "Warning: dataset has no `improvements`; "
            "gain-aware term is disabled by forcing --gain-loss-weight=0.0"
        )
        args.gain_loss_weight = 0.0

    num_actions = num_banks + 1
    train_has_soft_targets = _ensure_action_target_probs(
        train_dataset,
        num_actions=num_actions,
        mode=args.action_target_mode,
        temperature=args.action_utility_temperature,
        ce_tiebreaker_alpha=args.action_utility_ce_tiebreaker_alpha,
    )
    if val_dataset is not None:
        _ensure_action_target_probs(
            val_dataset,
            num_actions=num_actions,
            mode=args.action_target_mode,
            temperature=args.action_utility_temperature,
            ce_tiebreaker_alpha=args.action_utility_ce_tiebreaker_alpha,
        )
    train_action_counts = torch.bincount(
        train_dataset["action_target"].long(),
        minlength=num_actions,
    ).long()
    if train_has_soft_targets:
        print("Using soft accuracy-utility action targets.")
    action_class_weights = _compute_action_class_weights(
        action_target=train_dataset["action_target"],
        num_actions=num_actions,
        mode=args.action_class_weight_mode,
        effective_num_beta=args.action_class_effective_num_beta,
        max_weight=args.action_class_max_weight,
        skip_multiplier=args.action_class_skip_multiplier,
    )
    print(f"Train action counts (0=skip,1..K=banks): {train_action_counts.tolist()}")
    if action_class_weights is None:
        print("Action class weighting: disabled")
    else:
        print(f"Action class weights: {[float(x) for x in action_class_weights.tolist()]}")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    router = SimpleKVRouter(
        num_banks=num_banks,
        input_dim=pooled_feature_dim,
        hidden_dim=args.hidden_dim,
        token_mlp_layers=args.token_mlp_layers,
        dropout=args.dropout,
        input_layer_norm=args.input_layer_norm,
        input_standardization=args.input_standardization,
        feature_source=args.router_feature_source,
        option_token_ids=_optional_int_list(train_dataset.get("option_token_ids")),
        option_response_token_ids=_optional_int_list(
            train_dataset.get("option_response_token_ids")
        ),
        option_response_text=train_dataset.get("option_response_text"),
        decision_policy=args.decision_policy,
        skip_threshold=args.skip_threshold,
    ).to(device)
    if args.input_standardization:
        feature_stats = train_dataset["pooled_feature"].float()
        feature_mean = feature_stats.mean(dim=0)
        feature_std = feature_stats.std(dim=0, unbiased=False).clamp_min(1e-6)
        router.set_input_standardization(feature_mean, feature_std)
        print(
            "Input standardization enabled: "
            f"mean_abs={float(feature_mean.abs().mean()):.6g} "
            f"std_mean={float(feature_std.mean()):.6g}"
        )

    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_actions=num_actions,
    )
    val_loader = (
        _make_loader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_actions=num_actions,
        )
        if val_dataset is not None
        else None
    )
    total_steps = args.epochs * max(len(train_loader), 1)
    if args.lr_warmup_steps >= 0:
        warmup_steps = min(args.lr_warmup_steps, total_steps)
    else:
        warmup_steps = min(int(total_steps * args.lr_warmup_ratio), total_steps)
    lr_scheduler = _build_lr_scheduler(
        optimizer,
        scheduler_type=args.lr_scheduler_type,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=args.lr_min_ratio,
    )
    print(
        "LR scheduler: "
        f"type={args.lr_scheduler_type}, total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, min_ratio={args.lr_min_ratio}"
    )

    wandb_module = _maybe_init_wandb(
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_banks=num_banks,
        pooled_feature_dim=pooled_feature_dim,
        output_dir=output_dir,
        action_class_weights=action_class_weights,
    )

    best_state = None
    if args.best_metric == "auto":
        checkpoint_metric = "mean_routed_gain" if val_has_improvements else "loss"
    else:
        checkpoint_metric = args.best_metric
    maximize_metric = checkpoint_metric in {
        "mean_routed_gain",
        "gain_capture",
        "action_acc",
        "mean_routed_utility",
        "threshold_sweep_mean_routed_utility",
    }
    best_metric = float("-inf") if maximize_metric else float("inf")
    best_epoch: Optional[int] = None
    best_skip_threshold = float(args.skip_threshold)
    global_step = 0
    print(f"Checkpoint selection metric: {checkpoint_metric} (maximize={maximize_metric})")

    for epoch in range(args.epochs):
        router.train()
        effective_gain_loss_weight = _resolve_gain_loss_weight_for_epoch(
            base_gain_loss_weight=args.gain_loss_weight,
            epoch_index=epoch,
            gain_ce_only_epochs=args.gain_ce_only_epochs,
            gain_ramp_epochs=args.gain_ramp_epochs,
        )
        epoch_loss = 0.0
        epoch_action_loss = 0.0
        epoch_gain_loss = 0.0
        epoch_expected_gain = 0.0
        epoch_examples = 0

        for step, batch in enumerate(train_loader, start=1):
            (
                pooled_feature,
                action_target,
                _,
                _,
                improvements,
                action_target_probs,
                _action_utilities,
            ) = batch
            pooled_feature = pooled_feature.to(device)
            action_target = action_target.to(device)
            improvements = improvements.to(device)
            action_target_probs = action_target_probs.to(device)
            if action_target_probs.numel() == 0:
                action_target_probs = None

            _, action_loss, gain_loss, expected_routed_gain = _compute_losses(
                router,
                pooled_feature,
                action_target,
                improvements,
                selection_temperature=args.selection_temperature,
                action_class_weights=action_class_weights,
                action_label_smoothing=args.action_label_smoothing,
                action_sample_weight_mode=args.action_sample_weight_mode,
                action_sample_weight_scale=args.action_sample_weight_scale,
                action_sample_weight_min=args.action_sample_weight_min,
                action_decision_balance_power=args.action_decision_balance_power,
                action_target_probs=action_target_probs,
                action_utilities=_action_utilities,
            )
            loss = (
                args.action_loss_weight * action_loss
                + effective_gain_loss_weight * gain_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            batch_size = pooled_feature.size(0)
            epoch_loss += float(loss.item()) * batch_size
            epoch_action_loss += float(action_loss.item()) * batch_size
            epoch_gain_loss += float(gain_loss.item()) * batch_size
            epoch_expected_gain += float(expected_routed_gain.sum().item())
            epoch_examples += batch_size
            global_step += 1

            if wandb_module is not None and (
                global_step == 1
                or global_step % args.log_every == 0
                or step == len(train_loader)
            ):
                wandb_module.log(
                    {
                        "global_step": global_step,
                        "epoch": epoch + step / max(len(train_loader), 1),
                        "train/step_loss": float(loss.item()),
                        "train/step_action_loss": float(action_loss.item()),
                        "train/step_gain_loss": float(gain_loss.item()),
                        "train/step_expected_gain": float(expected_routed_gain.mean().item()),
                        "train/gain_loss_weight_effective": float(effective_gain_loss_weight),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=global_step,
                )

        train_metrics = {
            "loss": epoch_loss / max(epoch_examples, 1),
            "action_loss": epoch_action_loss / max(epoch_examples, 1),
            "gain_loss": epoch_gain_loss / max(epoch_examples, 1),
            "mean_expected_gain": epoch_expected_gain / max(epoch_examples, 1),
        }

        if val_loader is not None:
            val_metrics = _evaluate(
                router,
                val_loader,
                device,
                action_loss_weight=args.action_loss_weight,
                gain_loss_weight=effective_gain_loss_weight,
                fuse_threshold=args.fuse_threshold,
                selection_temperature=args.selection_temperature,
                action_class_weights=action_class_weights,
                decision_policy=args.decision_policy,
                skip_threshold=args.skip_threshold,
                sweep_skip_threshold=args.sweep_skip_threshold,
                skip_threshold_steps=args.skip_threshold_steps,
            )
            score = val_metrics[checkpoint_metric]
            threshold_text = ""
            if "threshold_sweep_mean_routed_utility" in val_metrics:
                threshold_text = (
                    f" sweep_utility={val_metrics['threshold_sweep_mean_routed_utility']:.4f} "
                    f"sweep_t={val_metrics['threshold_sweep_skip_threshold']:.3f} "
                    f"sweep_skip={val_metrics['threshold_sweep_skip_rate']:.4f}"
                )
            print(
                f"epoch={epoch + 1} "
                f"gain_w={effective_gain_loss_weight:.4f} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_mean_routed_gain={val_metrics['mean_routed_gain']:.4f} "
                f"val_routed_utility={val_metrics['mean_routed_utility']:.4f} "
                f"val_gain_capture={val_metrics['gain_capture']:.4f} "
                f"val_action_acc={val_metrics['action_acc']:.4f} "
                f"val_binary_acc={val_metrics['binary_acc']:.4f} "
                f"val_bank_acc={val_metrics['bank_acc']:.4f} "
                f"{checkpoint_metric}={score:.4f}"
                f"{threshold_text}"
            )
            is_better = score > best_metric if maximize_metric else score < best_metric
            if is_better:
                best_metric = score
                best_epoch = epoch + 1
                best_skip_threshold = float(
                    val_metrics.get("threshold_sweep_skip_threshold", args.skip_threshold)
                )
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in router.state_dict().items()
                }
        else:
            score = train_metrics["loss"]
            best_metric = score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in router.state_dict().items()
            }
            val_metrics = None
            print(
                f"epoch={epoch + 1} "
                f"gain_w={effective_gain_loss_weight:.4f} "
                f"train_loss={train_metrics['loss']:.4f}"
            )

        if wandb_module is not None:
            log_payload = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "train/epoch_loss": train_metrics["loss"],
                "train/epoch_action_loss": train_metrics["action_loss"],
                "train/epoch_gain_loss": train_metrics["gain_loss"],
                "train/epoch_mean_expected_gain": train_metrics["mean_expected_gain"],
                "train/gain_loss_weight_effective": float(effective_gain_loss_weight),
                "best/metric": best_metric,
                "best/epoch": best_epoch,
                "best/metric_name": checkpoint_metric,
            }
            if val_metrics is not None:
                log_payload.update(
                    _build_val_wandb_payload(
                        val_metrics=val_metrics,
                        mode=args.wandb_val_metrics_mode,
                    )
                )
            wandb_module.log(log_payload, step=global_step)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.decision_policy == "skip_threshold":
        router.skip_threshold = best_skip_threshold
        router.decision_policy = "skip_threshold"
        if hasattr(router, "_init_args"):
            router._init_args["decision_policy"] = "skip_threshold"
            router._init_args["skip_threshold"] = best_skip_threshold
    save_router(router.cpu(), str(output_dir / "router.json"))
    if best_state is not None:
        router.load_state_dict(best_state)
    torch.save(router.state_dict(), output_dir / "router.pt")
    print(f"Saved router to {output_dir}")

    if wandb_module is not None:
        wandb_module.run.summary["best_metric"] = best_metric
        wandb_module.run.summary["best_metric_name"] = checkpoint_metric
        wandb_module.run.summary["best_epoch"] = best_epoch
        wandb_module.run.summary["router_output_dir"] = str(output_dir)
        wandb_module.finish()


if __name__ == "__main__":
    main()
