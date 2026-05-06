#!/usr/bin/env python
"""
Build a pooled-feature routing dataset from precomputed feature/improvement shards.

Expected shard formats:
1. A list of dict examples.
2. A dict of batched tensors/lists.

Each example must provide:
- pooled_feature
- improvements, or (ce_receiver and ce_fusions)

This builder also creates:
- binary_target (0=skip, 1=fuse)
- bank_target (-1 for skip, else bank index)
- action_target (0=skip, else bank index + 1)

Optional fields are preserved when possible.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import torch


def _load_examples(path: Path) -> List[Dict]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if "examples" in payload:
            return payload["examples"]

        if "pooled_feature" in payload:
            examples = []
            num_examples = len(payload["pooled_feature"])
            for idx in range(num_examples):
                example = {}
                for key, value in payload.items():
                    if isinstance(value, torch.Tensor):
                        example[key] = value[idx]
                    elif isinstance(value, list):
                        example[key] = value[idx]
                    else:
                        example[key] = value
                examples.append(example)
            return examples

    raise ValueError(f"Unsupported shard format in {path}")


def _to_tensor_list(values) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.float()
    return torch.tensor(values, dtype=torch.float32)


def _compute_improvements(example: Dict) -> torch.Tensor:
    if "improvements" in example:
        return _to_tensor_list(example["improvements"])

    if "ce_receiver" in example and "ce_fusions" in example:
        ce_receiver = float(example["ce_receiver"])
        ce_fusions = _to_tensor_list(example["ce_fusions"])
        return ce_receiver - ce_fusions

    raise KeyError(
        "Each example must contain `improvements` or (`ce_receiver`, `ce_fusions`)."
    )


def _build_labels(
    improvements: torch.Tensor,
    skip_margin: float,
):
    masked = improvements.clone()
    masked[masked <= skip_margin] = float("-inf")
    best_bank = int(torch.argmax(masked).item()) if masked.numel() > 0 else -1
    best_improvement = float(masked[best_bank].item()) if best_bank >= 0 else float("-inf")

    if best_bank < 0 or best_improvement == float("-inf"):
        best_action = "skip"
        binary_target = 0
        bank_target = -1
        action_target = 0
    else:
        best_action = f"bank_{best_bank}"
        binary_target = 1
        bank_target = best_bank
        action_target = best_bank + 1

    return {
        "binary_target": binary_target,
        "bank_target": bank_target,
        "action_target": action_target,
        "best_action": best_action,
    }


def build_dataset(shard_paths: Iterable[Path], skip_margin: float) -> Dict[str, torch.Tensor]:
    pooled_features = []
    improvements_list = []
    ce_receivers = []
    ce_fusions_list = []
    receiver_corrects = []
    fusion_corrects_list = []
    option_golds = []
    receiver_pred_options = []
    fusion_pred_options_list = []
    binary_targets = []
    bank_targets = []
    action_targets = []
    best_actions = []
    feature_sources = set()
    option_token_ids = None
    option_response_token_ids = None
    option_response_text = None

    for shard_path in shard_paths:
        for example in _load_examples(shard_path):
            improvements = _compute_improvements(example)
            labels = _build_labels(improvements, skip_margin=skip_margin)

            pooled_feature = example["pooled_feature"]
            if not isinstance(pooled_feature, torch.Tensor):
                pooled_feature = torch.tensor(pooled_feature, dtype=torch.float32)

            pooled_features.append(pooled_feature.float())
            improvements_list.append(improvements.float())
            if "ce_receiver" in example:
                ce_receivers.append(float(example["ce_receiver"]))
            if "ce_fusions" in example:
                ce_fusions_list.append(_to_tensor_list(example["ce_fusions"]).float())
            if example.get("receiver_correct") is not None:
                receiver_corrects.append(float(bool(example["receiver_correct"])))
            if "fusion_corrects" in example and example["fusion_corrects"] is not None:
                fusion_corrects_list.append(_to_tensor_list(example["fusion_corrects"]).float())
            if example.get("option_gold") is not None:
                option_golds.append(int(example["option_gold"]))
            if example.get("receiver_pred_option") is not None:
                receiver_pred_options.append(int(example["receiver_pred_option"]))
            if "fusion_pred_options" in example and example["fusion_pred_options"] is not None:
                fusion_pred_options_list.append(_to_tensor_list(example["fusion_pred_options"]).long())
            binary_targets.append(labels["binary_target"])
            bank_targets.append(labels["bank_target"])
            action_targets.append(labels["action_target"])
            best_actions.append(labels["best_action"])
            if "feature_source" in example and example["feature_source"] is not None:
                feature_sources.add(str(example["feature_source"]))
            if option_token_ids is None and example.get("option_token_ids") is not None:
                option_token_ids = _to_tensor_list(example["option_token_ids"]).long()
            if (
                option_response_token_ids is None
                and example.get("option_response_token_ids") is not None
            ):
                option_response_token_ids = _to_tensor_list(
                    example["option_response_token_ids"]
                ).long()
            if option_response_text is None and example.get("option_response_text") is not None:
                option_response_text = str(example["option_response_text"])

    if not pooled_features:
        raise ValueError("No examples were loaded from the provided shards.")
    if len(feature_sources) > 1:
        raise ValueError(
            f"Inconsistent feature_source across shards/examples: {sorted(feature_sources)}"
        )

    feature_source = next(iter(feature_sources)) if feature_sources else "kv"

    dataset = {
        "pooled_feature": torch.stack(pooled_features, dim=0),
        "improvements": torch.stack(improvements_list, dim=0),
        "binary_target": torch.tensor(binary_targets, dtype=torch.long),
        "bank_target": torch.tensor(bank_targets, dtype=torch.long),
        "action_target": torch.tensor(action_targets, dtype=torch.long),
        "best_action": best_actions,
    }
    num_examples = len(pooled_features)
    if len(ce_receivers) == num_examples:
        dataset["ce_receiver"] = torch.tensor(ce_receivers, dtype=torch.float32)
    if len(ce_fusions_list) == num_examples:
        dataset["ce_fusions"] = torch.stack(ce_fusions_list, dim=0).float()
    if len(receiver_corrects) == num_examples and len(fusion_corrects_list) == num_examples:
        receiver_correct = torch.tensor(receiver_corrects, dtype=torch.float32)
        fusion_corrects = torch.stack(fusion_corrects_list, dim=0).float()
        dataset["receiver_correct"] = receiver_correct
        dataset["fusion_corrects"] = fusion_corrects
        dataset["action_utilities"] = torch.cat(
            [receiver_correct.unsqueeze(-1), fusion_corrects],
            dim=-1,
        )
    if len(option_golds) == num_examples:
        dataset["option_gold"] = torch.tensor(option_golds, dtype=torch.long)
    if len(receiver_pred_options) == num_examples:
        dataset["receiver_pred_option"] = torch.tensor(receiver_pred_options, dtype=torch.long)
    if len(fusion_pred_options_list) == num_examples:
        dataset["fusion_pred_options"] = torch.stack(fusion_pred_options_list, dim=0).long()
    if option_token_ids is not None:
        dataset["option_token_ids"] = option_token_ids
    if option_response_token_ids is not None:
        dataset["option_response_token_ids"] = option_response_token_ids
    if option_response_text is not None:
        dataset["option_response_text"] = option_response_text
    dataset["feature_source"] = feature_source
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-shards",
        nargs="+",
        required=True,
        help="Input .pt shards containing pooled router features and improvements.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to the consolidated router dataset (.pt).",
    )
    parser.add_argument(
        "--skip-margin",
        type=float,
        default=1e-6,
        help="Improvements at or below this value are treated as skip.",
    )
    args = parser.parse_args()

    dataset = build_dataset(
        shard_paths=[Path(path) for path in args.input_shards],
        skip_margin=args.skip_margin,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    print(f"Saved {len(dataset['pooled_feature'])} examples to {output_path}")


if __name__ == "__main__":
    main()
