#!/usr/bin/env python3
"""MMLU-Redux evaluator with forced correct-answer label positions.

This keeps the standard generate/logits evaluation path intact, but rewrites
each MMLU-Redux example so the correct option is placed at a configured label
position. It is intended as a label-prior diagnostic.
"""

import argparse
from typing import Any, Dict

import torch.multiprocessing as mp
import yaml

from script.evaluation.unified_evaluator_choice_shuffle import (
    UnifiedEvaluator as ChoiceShuffleUnifiedEvaluator,
)


class ForcedLabelUnifiedEvaluator(ChoiceShuffleUnifiedEvaluator):
    def _maybe_shuffle_mmlu_redux_choices(
        self,
        example: Dict[str, Any],
        subject: str,
        question_id: int,
    ) -> Dict[str, Any]:
        forced_cfg = self.eval_config.get("forced_label", {})
        if not isinstance(forced_cfg, dict) or not bool(forced_cfg.get("enabled", False)):
            return example
        if self.dataset_name != "mmlu-redux":
            return example

        choices = list(example.get("choices", []))
        if len(choices) < 2:
            return example

        old_answer_num = self._parse_mmlu_redux_answer_index(example)
        if old_answer_num is None:
            return example

        target_label = str(forced_cfg.get("target_label", "C")).strip().upper()
        if len(target_label) != 1 or target_label < "A" or target_label > "Z":
            raise ValueError(f"Invalid forced_label.target_label: {target_label!r}")
        target_idx = ord(target_label) - ord("A")
        if target_idx >= len(choices):
            return example

        old_answer_num = int(old_answer_num)
        if old_answer_num < 0 or old_answer_num >= len(choices):
            return example
        perm = list(range(len(choices)))
        perm[target_idx], perm[old_answer_num] = perm[old_answer_num], perm[target_idx]

        forced = dict(example)
        forced["choices"] = [choices[idx] for idx in perm]
        forced["answer"] = int(target_idx)
        if example.get("correct_answer") is not None:
            forced["correct_answer"] = str(target_idx)

        # Reuse the existing choice-shuffle CSV columns so downstream analysis
        # can compare original vs remapped labels without changing the writer.
        forced["_choice_shuffle_enabled"] = True
        forced["_choice_shuffle_seed"] = f"forced-{target_label}"
        forced["_choice_shuffle_perm"] = ",".join(chr(65 + idx) for idx in perm)
        forced["_choice_shuffle_original_true_answer"] = self._answer_index_to_letter(old_answer_num)
        forced["_choice_shuffle_shuffled_true_answer"] = self._answer_index_to_letter(target_idx)
        return forced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="eval_recipe/unified_eval.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("Using config: ", args.config)
    evaluator = ForcedLabelUnifiedEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    import torch._dynamo as dynamo

    dynamo.config.cache_size_limit = 64
    mp.set_start_method("spawn", force=True)
    main()
