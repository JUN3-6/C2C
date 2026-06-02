"""
KV-content ablation evaluator for Rosetta/C2C models.

This evaluator keeps the receiver prompt from sample i, but replaces the
sharer prefill prompt with a donor sample j that has the same tokenized prompt
length. It is intended to test whether C2C gains depend on the semantic content
of the sharer KV cache or mostly on projector/prior effects.

The standard evaluator is left unchanged. This file subclasses the local
choice-shuffle evaluator to reuse its model loading, formatting, and dtype
patches, then adds only the source-KV swap behavior and extra logging.
"""

import argparse
import csv
import hashlib
import json
import random
import types
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from datasets import load_dataset

from rosetta.model.wrapper import RosettaModel
from script.evaluation.unified_evaluator_choice_shuffle import (
    DATASET_CONFIGS,
    UnifiedEvaluator as BaseUnifiedEvaluator,
)


class KVContentAblationEvaluator(BaseUnifiedEvaluator):
    """Evaluate with donor sharer KV caches for MMLU-Redux."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kv_ablation_config = self.eval_config.get("kv_content_ablation", {})
        self._kv_ablation_index_built = False
        self._kv_ablation_records: List[Dict[str, Any]] = []
        self._kv_ablation_donors: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._kv_ablation_donor_by_prompt: Dict[str, Dict[str, Any]] = {}
        self._kv_ablation_log_meta: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._kv_ablation_pending_source_inputs: Optional[Dict[str, Any]] = None
        self._kv_ablation_index_stats: Dict[str, Any] = {}

    def _kv_ablation_enabled(self) -> bool:
        cfg = self.kv_ablation_config
        return isinstance(cfg, dict) and bool(cfg.get("enabled", False))

    def _prompt_chat_text(self, prompt: str, tokenizer) -> str:
        messages = [{"role": "user", "content": prompt}]
        if self.eval_config.get("answer_method") == "logits":
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            text += self.eval_config.get("response_text", "The correct answer is")
            return text

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _prompt_token_length(self, prompt: str, tokenizer) -> int:
        text = self._prompt_chat_text(prompt, tokenizer)
        return int(tokenizer(text, return_tensors="pt")["input_ids"].shape[1])

    def _build_mmlu_redux_records(self, tokenizer) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        dataset_cache_dir = self.eval_config.get("dataset_cache_dir")
        load_kwargs = {"cache_dir": dataset_cache_dir} if dataset_cache_dir else {}

        for subject in DATASET_CONFIGS["mmlu-redux"]["subjects"]:
            dataset = load_dataset(DATASET_CONFIGS["mmlu-redux"]["dataset_name"], subject, **load_kwargs)
            split = dataset[DATASET_CONFIGS["mmlu-redux"]["test_split"]]
            for question_id, raw_example in enumerate(split):
                example = dict(raw_example)
                answer_num = self._parse_mmlu_redux_answer_index(example)
                if answer_num is None:
                    continue

                prompt = self._format_mmlu_redux_example(
                    example,
                    use_cot=self.eval_config["use_cot"],
                    use_template=self.eval_config["use_template"],
                )
                records.append(
                    {
                        "subject": subject,
                        "question_id": int(question_id),
                        "question": str(example.get("question", "")),
                        "true_answer": self._answer_index_to_letter(answer_num),
                        "prompt": prompt,
                        "length": self._prompt_token_length(prompt, tokenizer),
                    }
                )

        return records

    def _select_donor(
        self,
        record: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        seed: int,
        require_label_mismatch: bool,
    ) -> Optional[Dict[str, Any]]:
        pool = [
            candidate
            for candidate in candidates
            if not (
                candidate["subject"] == record["subject"]
                and int(candidate["question_id"]) == int(record["question_id"])
            )
        ]
        if require_label_mismatch:
            pool = [candidate for candidate in pool if candidate["true_answer"] != record["true_answer"]]
        if not pool:
            return None

        key = "||".join(
            [
                str(seed),
                str(record["subject"]),
                str(record["question_id"]),
                str(record["question"]),
            ]
        )
        donor_idx = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(pool)
        return pool[donor_idx]

    def _ensure_kv_ablation_index(self, tokenizer) -> None:
        if self._kv_ablation_index_built or not self._kv_ablation_enabled():
            return
        if self.dataset_name != "mmlu-redux":
            raise ValueError("kv_content_ablation is currently implemented for mmlu-redux only")

        cfg = self.kv_ablation_config
        mode = str(cfg.get("mode", "global_exact"))
        if mode != "global_exact":
            raise ValueError(f"Unsupported kv_content_ablation mode: {mode}")

        seed = int(cfg.get("seed", 0))
        require_label_mismatch = bool(cfg.get("require_label_mismatch", False))
        records = self._build_mmlu_redux_records(tokenizer)

        by_length: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_length[int(record["length"])].append(record)

        donors: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for record in records:
            candidates = by_length[int(record["length"])]
            donor = self._select_donor(record, candidates, seed, require_label_mismatch)
            if donor is not None:
                donors[(record["subject"], int(record["question_id"]))] = donor

        coverage = len(donors) / len(records) if records else 0.0
        singleton_lengths = sum(1 for group in by_length.values() if len(group) == 1)
        self._kv_ablation_index_stats = {
            "mode": mode,
            "seed": seed,
            "records": len(records),
            "matched": len(donors),
            "coverage": coverage,
            "unique_lengths": len(by_length),
            "singleton_length_groups": singleton_lengths,
            "require_label_mismatch": require_label_mismatch,
        }
        print(
            "[KV-content ablation] Built global_exact donor index: "
            f"{len(donors)}/{len(records)} matched ({coverage * 100:.2f}%), "
            f"{singleton_lengths} singleton length groups, seed={seed}, "
            f"require_label_mismatch={require_label_mismatch}"
        )

        self._kv_ablation_records = records
        self._kv_ablation_donors = donors
        self._kv_ablation_index_built = True

    def _maybe_shuffle_mmlu_redux_choices(
        self,
        example: Dict[str, Any],
        subject: str,
        question_id: int,
    ) -> Dict[str, Any]:
        example = super()._maybe_shuffle_mmlu_redux_choices(example, subject, question_id)
        if not self._kv_ablation_enabled() or self.dataset_name != "mmlu-redux":
            return example

        self._ensure_kv_ablation_index(self.tokenizer)

        out = dict(example)
        target_key = (subject, int(question_id))
        donor = self._kv_ablation_donors.get(target_key)
        skip_unmatched = bool(self.kv_ablation_config.get("skip_unmatched", True))

        target_answer_num = self._parse_mmlu_redux_answer_index(out)
        target_answer = self._answer_index_to_letter(target_answer_num)
        target_prompt = self._format_mmlu_redux_example(
            out,
            use_cot=self.eval_config["use_cot"],
            use_template=self.eval_config["use_template"],
        )
        target_length = self._prompt_token_length(target_prompt, self.tokenizer)

        meta = {
            "kv_ablation_enabled": True,
            "kv_ablation_mode": self.kv_ablation_config.get("mode", "global_exact"),
            "kv_ablation_seed": int(self.kv_ablation_config.get("seed", 0)),
            "kv_ablation_matched": donor is not None,
            "target_prompt_length": target_length,
            "donor_subject": "",
            "donor_question_id": "",
            "donor_true_answer": "",
            "donor_prompt_length": "",
            "donor_label_matches_target": "",
            "donor_match_delta": "",
        }

        if donor is None:
            out["_kv_ablation_skip"] = bool(skip_unmatched)
            self._kv_ablation_log_meta[target_key] = meta
            return out

        meta.update(
            {
                "donor_subject": donor["subject"],
                "donor_question_id": int(donor["question_id"]),
                "donor_true_answer": donor["true_answer"],
                "donor_prompt_length": int(donor["length"]),
                "donor_label_matches_target": bool(donor["true_answer"] == target_answer),
                "donor_match_delta": int(donor["length"]) - int(target_length),
            }
        )

        self._kv_ablation_donor_by_prompt[target_prompt] = donor
        self._kv_ablation_log_meta[target_key] = meta
        out["_kv_ablation_skip"] = False
        return out

    def parse_answer(self, example: Dict[str, Any]) -> Optional[str]:
        if example.get("_kv_ablation_skip"):
            return None
        return super().parse_answer(example)

    def prepare_model_inputs(
        self,
        prompt: str,
        tokenizer,
        device: torch.device,
        model_type: str,
        llm_tokenizer: Optional[Any],
        answer_method: str,
        proportion: float = 1.0,
        order_mode: str = "front",
    ):
        prepared = BaseUnifiedEvaluator.prepare_model_inputs(
            self,
            prompt=prompt,
            tokenizer=tokenizer,
            device=device,
            model_type=model_type,
            llm_tokenizer=llm_tokenizer,
            answer_method=answer_method,
            proportion=proportion,
            order_mode=order_mode,
        )
        self._kv_ablation_pending_source_inputs = None

        if not self._kv_ablation_enabled() or model_type != "rosetta" or self.dataset_name != "mmlu-redux":
            return prepared

        donor = self._kv_ablation_donor_by_prompt.get(prompt)
        if donor is None:
            return prepared

        donor_prepared = BaseUnifiedEvaluator.prepare_model_inputs(
            self,
            prompt=donor["prompt"],
            tokenizer=tokenizer,
            device=device,
            model_type=model_type,
            llm_tokenizer=llm_tokenizer,
            answer_method=answer_method,
            proportion=proportion,
            order_mode=order_mode,
        )

        target_ids = prepared["inputs"]["input_ids"]
        donor_ids = donor_prepared["inputs"]["input_ids"]
        if isinstance(target_ids, list) or isinstance(donor_ids, list):
            raise ValueError("kv_content_ablation currently expects non-aligned Rosetta tensor inputs")
        if target_ids.shape != donor_ids.shape:
            raise ValueError(
                "KV donor prompt length mismatch after tokenization: "
                f"target={tuple(target_ids.shape)}, donor={tuple(donor_ids.shape)}, "
                f"donor={donor['subject']}#{donor['question_id']}"
            )

        self._kv_ablation_pending_source_inputs = {
            "input_ids": donor_ids,
            "attention_mask": donor_prepared["inputs"].get("attention_mask"),
            "subject": donor["subject"],
            "question_id": int(donor["question_id"]),
        }
        return prepared

    def _patch_rosetta_kv_content_ablation(self, model) -> None:
        if not self._kv_ablation_enabled() or not isinstance(model, RosettaModel):
            return
        if getattr(model, "_kv_content_ablation_forward_patch", False):
            return

        evaluator = self
        original_forward = model.forward

        def patched_forward(self_model, *args, **kwargs):
            donor_inputs = evaluator._kv_ablation_pending_source_inputs
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")

            if (
                donor_inputs is not None
                and torch.is_tensor(input_ids)
                and input_ids.ndim == 2
                and input_ids.shape[1] > 1
            ):
                donor_ids = donor_inputs["input_ids"].to(device=input_ids.device)
                if donor_ids.shape != input_ids.shape:
                    raise ValueError(
                        "KV donor input_ids shape changed before Rosetta forward: "
                        f"target={tuple(input_ids.shape)}, donor={tuple(donor_ids.shape)}"
                    )

                num_models = len(getattr(self_model, "model_list", []))
                if num_models < 2:
                    return original_forward(*args, **kwargs)

                kwargs["input_ids"] = [input_ids] + [donor_ids] * (num_models - 1)

                if attention_mask is not None:
                    donor_mask = donor_inputs.get("attention_mask")
                    if donor_mask is not None:
                        donor_mask = donor_mask.to(
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        )
                    kwargs["attention_mask"] = [attention_mask] + [donor_mask] * (num_models - 1)

            return original_forward(*args, **kwargs)

        model.forward = types.MethodType(patched_forward, model)
        model._kv_content_ablation_forward_patch = True

    def evaluate_subject(self, subject: str, model, tokenizer, device: torch.device, model_type: str = "hf", llm_tokenizer: Optional[Any] = None):
        self._patch_rosetta_kv_content_ablation(model)
        result = super().evaluate_subject(subject, model, tokenizer, device, model_type, llm_tokenizer)
        cors, acc, probs, length_stats, cot_logs = result
        for row in cot_logs:
            key = (row.get("subject"), int(row.get("question_id")))
            row.update(self._kv_ablation_log_meta.get(key, {}))
            if "donor_true_answer" in row and row.get("pred"):
                row["pred_matches_donor_true_answer"] = bool(row["pred"] == row["donor_true_answer"])
        return cors, acc, probs, length_stats, cot_logs

    def evaluate_on_gpu(self, rank: int, gpu_id: int, subjects: List[str], return_dict):
        super().evaluate_on_gpu(rank, gpu_id, subjects, return_dict)
        if rank in return_dict:
            result = dict(return_dict[rank])
            result["kv_ablation_index_stats"] = dict(self._kv_ablation_index_stats)
            return_dict[rank] = result

    def merge_results(self, results_by_rank: Dict):
        for result in results_by_rank.values():
            stats = result.get("kv_ablation_index_stats")
            if stats:
                self._kv_ablation_index_stats = dict(stats)
                break
        return super().merge_results(results_by_rank)

    def save_results(self, all_cors, subject_cors, subcat_cors, cat_cors, all_length_stats, all_cot_logs):
        if self.dataset_name != "longbench":
            overall_accuracy = np.mean(np.concatenate(all_cors)) if all_cors else 0
        else:
            overall_accuracy = 0

        summary = {
            "model": self.model_config["model_name"],
            "dataset": self.dataset_name,
            "answer_method": self.eval_config["answer_method"],
            "overall_accuracy": float(overall_accuracy),
            "subjects": subject_cors,
            "kv_content_ablation": self.kv_ablation_config,
            "kv_content_ablation_index": self._kv_ablation_index_stats,
        }

        if self.dataset_name == "mmlu-redux":
            summary["categories"] = {
                cat: float(np.mean(np.concatenate(cors))) if cors else 0.0
                for cat, cors in cat_cors.items()
            }
            summary["subcategories"] = {
                subcat: float(np.mean(np.concatenate(cors))) if cors else 0.0
                for subcat, cors in subcat_cors.items()
            }

        if all_length_stats:
            summary["length_statistics"] = self._compute_length_statistics(all_length_stats)

        model_name_for_file = self.model_config["model_name"].split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_file = self.output_dir / f"{model_name_for_file}_{self.dataset_name}_{self.eval_config['answer_method']}_{timestamp}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

        if all_length_stats:
            detailed_length_file = self.output_dir / f"{model_name_for_file}_{self.dataset_name}_{self.eval_config['answer_method']}_{timestamp}_length.json"
            with open(detailed_length_file, "w") as f:
                json.dump(all_length_stats, f, indent=2)
            print(f"Detailed length statistics saved to {detailed_length_file}")

        if all_cot_logs and self.dataset_name != "longbench":
            cot_csv_file = self.output_dir / f"{model_name_for_file}_{self.dataset_name}_{self.eval_config['answer_method']}_{timestamp}_cot.csv"
            fieldnames = [
                "subject",
                "question_id",
                "question",
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "true_answer",
                "choice_shuffle_enabled",
                "choice_shuffle_seed",
                "choice_shuffle_perm",
                "original_true_answer",
                "shuffled_true_answer",
                "kv_ablation_enabled",
                "kv_ablation_mode",
                "kv_ablation_seed",
                "kv_ablation_matched",
                "target_prompt_length",
                "donor_subject",
                "donor_question_id",
                "donor_true_answer",
                "donor_prompt_length",
                "donor_match_delta",
                "donor_label_matches_target",
                "pred_matches_donor_true_answer",
                "pred",
                "is_correct",
                "answer_method",
                "cot_pred",
                "cot_input_length",
                "cot_gen_length",
                "cot_output",
                "answer_latency_ms",
                "extraction_method_used",
                "ground_truth_normalized",
                "extracted_normalized",
            ]
            with open(cot_csv_file, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in all_cot_logs:
                    writer.writerow(row)
            print(f"CoT outputs saved to {cot_csv_file}")

        print("\nEvaluation complete!")
        if self.dataset_name != "longbench":
            print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="KV-content ablation evaluator")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("Using config: ", args.config)
    evaluator = KVContentAblationEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    import torch._dynamo as dynamo

    dynamo.config.cache_size_limit = 64
    mp.set_start_method("spawn", force=True)
    main()
