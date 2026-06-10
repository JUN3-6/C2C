"""Unified evaluator entrypoint with value-only projector ablation enabled."""

import argparse

import torch.multiprocessing as mp
import yaml

from rosetta.model.value_only import apply_projector_value_only
from script.evaluation.unified_evaluator_choice_shuffle import UnifiedEvaluator as BaseUnifiedEvaluator


class UnifiedEvaluator(BaseUnifiedEvaluator):
    def _patch_rosetta_projector_dtype(self, model) -> None:
        super()._patch_rosetta_projector_dtype(model)
        apply_projector_value_only(model)
        print("[value-only ablation] receiver keys preserved; projector values injected")


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script with value-only KV ablation")
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
    evaluator = UnifiedEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    import torch._dynamo as dynamo

    dynamo.config.cache_size_limit = 64
    mp.set_start_method("spawn", force=True)
    main()
