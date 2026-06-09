import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.model.projector import ClosedFormKVAlignProjector, save_projector
from rosetta.model.wrapper import RosettaModel
from rosetta.train.model_utils import k_nearest_sources, last_aligned_sources


DEFAULT_PROMPTS = [
    "Explain why the sky appears blue in two concise sentences.",
    "Solve the arithmetic problem step by step: 17 * 23 + 41.",
    "Choose the best answer. Which organ pumps blood through the human body? A. Lung B. Heart C. Liver D. Kidney",
    "Translate to Korean: Machine learning models can share latent representations.",
    "Write a Python function that returns the factorial of a non-negative integer.",
    "A train travels 180 kilometers in 3 hours. What is its average speed?",
    "Summarize the main tradeoff between speed and accuracy in model routing.",
    "Given A, B, C, D options, answer only with the letter: Water freezes at what temperature? A. 0 C B. 10 C C. 50 C D. 100 C",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit closed-form KV cache alignment projectors.")
    parser.add_argument("--receiver-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--source-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", default="local/checkpoints/qwen3_4b_to_0p6b_closed_form_kv/final")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--mapping", default="last_aligned", choices=["last_aligned", "k_nearest", "fixed_offset"])
    parser.add_argument(
        "--source-offset",
        type=int,
        default=None,
        help="For --mapping fixed_offset, map target layer t to source layer t + source_offset.",
    )
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--target-layers", default="all", help="'all' or comma-separated receiver layer indices.")
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    parser.add_argument(
        "--postprocess-mode",
        default="direct",
        choices=[
            "direct",
            "key_direct",
            "value_direct",
            "value_norm_residual",
            "kv_norm_residual",
            "value_delta_residual",
            "kv_delta_residual",
            "value_norm_delta_residual",
            "kv_token_norm_direct",
            "kv_norm_delta_residual",
        ],
    )
    parser.add_argument("--norm-ratio", type=float, default=1.0)
    parser.add_argument("--calibration-text-file", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-prompt", default="Explain in one sentence why latent KV alignment might help model collaboration.")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def load_prompts(path: str | None, max_prompts: int) -> List[str]:
    if path is None:
        prompts = DEFAULT_PROMPTS
    else:
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError("no calibration prompts available")
    return prompts


def selected_target_layers(spec: str, num_target_layers: int) -> List[int]:
    if spec == "all":
        return list(range(num_target_layers))
    layers = [int(item.strip()) for item in spec.split(",") if item.strip()]
    bad = [idx for idx in layers if idx < 0 or idx >= num_target_layers]
    if bad:
        raise ValueError(f"target layer out of range: {bad}")
    return layers


def get_cache_layer(cache, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    return cache[layer_idx]


def flatten_valid(kv: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # KV cache: [B, H, N, D] -> rows at valid token positions: [valid_tokens, H * D]
    batch, heads, seq_len, head_dim = kv.shape
    flat = kv.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim)
    mask = attention_mask[:, :seq_len].bool()
    return flat[mask].to(torch.float32).cpu()


def fit_ridge(source: torch.Tensor, target: torch.Tensor, ridge: float, use_bias: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be 2D matrices")
    if source.shape[0] != target.shape[0]:
        raise ValueError(f"sample mismatch: {source.shape[0]} vs {target.shape[0]}")

    x = source
    if use_bias:
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype)
        x = torch.cat([x, ones], dim=1)

    gram = x.T @ x
    reg = ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    if use_bias:
        reg[-1, -1] = 0.0
    rhs = x.T @ target
    weight = torch.linalg.solve(gram + reg, rhs)

    pred = x @ weight
    mse = F.mse_loss(pred, target).item()
    cosine = F.cosine_similarity(pred, target, dim=-1).mean().item()
    target_norm = target.norm(dim=-1).mean().item()
    pred_norm = pred.norm(dim=-1).mean().item()
    return weight, {
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "cosine": float(cosine),
        "target_norm": float(target_norm),
        "pred_norm": float(pred_norm),
    }


def fixed_offset_sources(num_target_layers: int, num_source_layers: int, source_offset: int) -> Dict[int, List[int]]:
    mapping: Dict[int, List[int]] = {}
    for target_layer in range(num_target_layers):
        source_layer = target_layer + source_offset
        if 0 <= source_layer < num_source_layers:
            mapping[target_layer] = [source_layer]
    return mapping


def build_mapping(
    mapping_name: str,
    num_target_layers: int,
    num_source_layers: int,
    k: int,
    source_offset: int | None,
) -> Dict[int, List[int]]:
    if mapping_name == "last_aligned":
        return last_aligned_sources(num_target_layers, num_source_layers, k)
    if mapping_name == "k_nearest":
        return k_nearest_sources(num_target_layers, num_source_layers, k)
    if mapping_name == "fixed_offset":
        if source_offset is None:
            raise ValueError("--source-offset is required for --mapping fixed_offset")
        return fixed_offset_sources(num_target_layers, num_source_layers, source_offset)
    raise ValueError(f"unknown mapping: {mapping_name}")


def iter_layer_pairs(mapping: Dict[int, List[int]], target_layers: Iterable[int]) -> Iterable[Tuple[int, int]]:
    for target_layer in target_layers:
        sources = mapping.get(target_layer, [])
        if not sources:
            continue
        yield target_layer, sources[0]


@torch.no_grad()
def forward_cache_only(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    base_model = getattr(model, "model", None)
    if base_model is not None:
        output = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    else:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    return output.past_key_values


@torch.no_grad()
def run_models(receiver_model, source_model, tokenizer, prompts: List[str], device: str, max_length: int):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    receiver_cache = forward_cache_only(receiver_model, input_ids, attention_mask)
    source_cache = forward_cache_only(source_model, input_ids, attention_mask)
    return input_ids, attention_mask, receiver_cache, source_cache


@torch.no_grad()
def collect_calibration_matrices(
    receiver_model,
    source_model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_length: int,
    batch_size: int,
    pairs: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], Dict[str, object]]:
    if batch_size <= 0:
        raise ValueError("--calibration-batch-size must be positive")

    chunks: Dict[Tuple[int, int], Dict[str, object]] = {
        pair: {
            "x_key": [],
            "y_key": [],
            "x_value": [],
            "y_value": [],
            "source_dim": None,
            "target_dim": None,
            "source_num_heads": None,
            "target_num_heads": None,
        }
        for pair in pairs
    }

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        end = start + len(batch_prompts)
        print(f"  - calibration batch {start + 1}-{end}/{len(prompts)}")
        _, attention_mask, receiver_cache, source_cache = run_models(
            receiver_model,
            source_model,
            tokenizer,
            batch_prompts,
            device,
            max_length,
        )
        attention_mask_cpu = attention_mask.detach().cpu()

        for target_layer, source_layer in pairs:
            target_key, target_value = get_cache_layer(receiver_cache, target_layer)
            source_key, source_value = get_cache_layer(source_cache, source_layer)
            bucket = chunks[(target_layer, source_layer)]
            if bucket["source_dim"] is None:
                bucket["source_dim"] = int(source_key.shape[-1])
                bucket["target_dim"] = int(target_key.shape[-1])
                bucket["source_num_heads"] = int(source_key.shape[1])
                bucket["target_num_heads"] = int(target_key.shape[1])
            bucket["x_key"].append(flatten_valid(source_key, attention_mask_cpu))
            bucket["y_key"].append(flatten_valid(target_key, attention_mask_cpu))
            bucket["x_value"].append(flatten_valid(source_value, attention_mask_cpu))
            bucket["y_value"].append(flatten_valid(target_value, attention_mask_cpu))

        del attention_mask, attention_mask_cpu, receiver_cache, source_cache
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    result: Dict[Tuple[int, int], Dict[str, object]] = {}
    for pair, tensors in chunks.items():
        result[pair] = {
            "x_key": torch.cat(tensors["x_key"], dim=0),
            "y_key": torch.cat(tensors["y_key"], dim=0),
            "x_value": torch.cat(tensors["x_value"], dim=0),
            "y_value": torch.cat(tensors["y_value"], dim=0),
            "source_dim": tensors["source_dim"],
            "target_dim": tensors["target_dim"],
            "source_num_heads": tensors["source_num_heads"],
            "target_num_heads": tensors["target_num_heads"],
        }
    return result


def save_checkpoint(output_dir: Path, projectors: List[ClosedFormKVAlignProjector], projector_config: Dict, metrics: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, projector in enumerate(projectors):
        save_projector(projector, str(output_dir / f"projector_{idx}.json"))
        torch.save(projector.state_dict(), output_dir / f"projector_{idx}.pt")
    with open(output_dir / "projector_config.json", "w", encoding="utf-8") as f:
        json.dump(projector_config, f, indent=2)
    with open(output_dir / "closed_form_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


@torch.no_grad()
def smoke_test(
    receiver_model,
    source_model,
    tokenizer,
    projectors: List[ClosedFormKVAlignProjector],
    projector_config: Dict,
    prompt: str,
    device: str,
) -> None:
    rosetta = RosettaModel(
        model_list=[receiver_model, source_model],
        base_model_idx=0,
        projector_list=projectors,
        multi_source_fusion_mode="parallel",
    ).to(device).eval()
    rosetta.projector_dict = RosettaModel._convert_dict_keys_to_ints(projector_config)

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    if seq_len < 2:
        raise ValueError("smoke prompt must produce at least two tokens")

    first_len = seq_len - 1
    kv_cache_index = [
        torch.tensor([1, 0], dtype=torch.long, device=device).repeat(first_len, 1).unsqueeze(0),
        torch.tensor([-1, 0], dtype=torch.long, device=device).repeat(1, 1).unsqueeze(0),
    ]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    output = rosetta(
        kv_cache_index=kv_cache_index,
        input_ids=[input_ids, input_ids],
        attention_mask=[attention_mask, attention_mask],
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    next_token = int(torch.argmax(output.logits[:, -1, :], dim=-1)[0].detach().cpu().item())
    print(f"[smoke] logits shape={tuple(output.logits.shape)} next_token={next_token} text={tokenizer.decode([next_token])!r}")


def main() -> None:
    args = parse_args()
    device = args.device
    dtype = get_dtype(args.dtype)
    output_dir = Path(args.output_dir)
    prompts = load_prompts(args.calibration_text_file, args.max_prompts)

    print(f"Loading receiver: {args.receiver_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    receiver_model = AutoModelForCausalLM.from_pretrained(
        args.receiver_model,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=args.trust_remote_code,
    ).eval()
    print(f"Loading source: {args.source_model}")
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=args.trust_remote_code,
    ).eval()

    num_target_layers = receiver_model.config.num_hidden_layers
    num_source_layers = source_model.config.num_hidden_layers
    target_layers = selected_target_layers(args.target_layers, num_target_layers)
    mapping = build_mapping(args.mapping, num_target_layers, num_source_layers, args.k, args.source_offset)
    pairs = list(iter_layer_pairs(mapping, target_layers))
    if not pairs:
        raise ValueError("no layer pairs selected")

    print(
        f"Running calibration forward on {len(prompts)} prompts, "
        f"max_length={args.max_length}, batch_size={args.calibration_batch_size}"
    )
    calibration_matrices = collect_calibration_matrices(
        receiver_model,
        source_model,
        tokenizer,
        prompts,
        device,
        args.max_length,
        args.calibration_batch_size,
        pairs,
    )

    projectors: List[ClosedFormKVAlignProjector] = []
    projector_config = {0: {1: {}}}
    metrics = {
        "receiver_model": args.receiver_model,
        "source_model": args.source_model,
        "mapping": args.mapping,
        "source_offset": args.source_offset,
        "k": args.k,
        "ridge": args.ridge,
        "blend_alpha": args.blend_alpha,
        "postprocess_mode": args.postprocess_mode,
        "norm_ratio": args.norm_ratio,
        "num_prompts": len(prompts),
        "max_length": args.max_length,
        "layers": {},
    }

    for projector_idx, (target_layer, source_layer) in enumerate(pairs):
        matrices = calibration_matrices[(target_layer, source_layer)]
        x_key = matrices["x_key"]
        y_key = matrices["y_key"]
        x_value = matrices["x_value"]
        y_value = matrices["y_value"]

        key_matrix, key_metrics = fit_ridge(x_key, y_key, args.ridge, use_bias=True)
        value_matrix, value_metrics = fit_ridge(x_value, y_value, args.ridge, use_bias=True)

        projector = ClosedFormKVAlignProjector(
            source_dim=int(matrices["source_dim"]),
            target_dim=int(matrices["target_dim"]),
            source_num_heads=int(matrices["source_num_heads"]),
            target_num_heads=int(matrices["target_num_heads"]),
            use_bias=True,
            blend_alpha=args.blend_alpha,
            postprocess_mode=args.postprocess_mode,
            norm_ratio=args.norm_ratio,
            dtype=torch.float32,
        )
        projector.set_alignment(key_matrix, value_matrix)
        projectors.append(projector)
        projector_config[0][1][target_layer] = [(source_layer, projector_idx)]
        metrics["layers"][str(target_layer)] = {
            "source_layer": source_layer,
            "projector_idx": projector_idx,
            "num_samples": int(x_key.shape[0]),
            "key": key_metrics,
            "value": value_metrics,
        }
        print(
            f"[fit] target={target_layer:02d} source={source_layer:02d} "
            f"K rmse={key_metrics['rmse']:.4f} cos={key_metrics['cosine']:.4f} "
            f"V rmse={value_metrics['rmse']:.4f} cos={value_metrics['cosine']:.4f}"
        )

    save_checkpoint(output_dir, projectors, projector_config, metrics)
    print(f"Saved closed-form KV checkpoint to {output_dir}")

    if args.smoke_test:
        smoke_test(
            receiver_model,
            source_model,
            tokenizer,
            [p.to(device) for p in projectors],
            projector_config,
            args.smoke_prompt,
            device,
        )


if __name__ == "__main__":
    main()
