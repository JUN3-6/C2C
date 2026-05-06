"""
Common evaluation utilities for benchmark tasks.

This module provides shared functions for model evaluation across different benchmarks
like MMLU-Redux and MMMLU.
"""

import re
import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.model.projector import load_projector
from rosetta.model.router import load_router
from rosetta.model.wrapper import RosettaModel
from rosetta.model.oracle import OracleRosettaModel

def build_prompt(dataset: str, locale: str, question: str, choices: str, use_cot: bool, use_template: bool = True) -> str:
    """
    Build a localized prompt for a given dataset and locale.

    Currently supports:
    - dataset: "mmmlu"
      - locale: "SW_KE" (Swahili). Other locales fall back to English.

    Args:
        dataset: Dataset identifier (e.g., "mmmlu")
        locale: Locale/subject code (e.g., "SW_KE")
        question: Question text
        choices: Formatted choices string
        use_cot: Whether to include CoT instruction

    Returns:
        Localized prompt string
    """
    
        # Unified default English templates (shared by MMLU and MMMLU)
    if not use_cot:
        template = """Accurately answer the following question:

{{question}}

Choices:
{{choices}}

Instructions:
- Carefully read the question and all options.
- Select the single most correct answer.
- Respond ONLY in the following format: "The correct answer is A/B/C/D".
- Do not include any explanations, additional text, or punctuation besides the answer.

The correct answer is"""

    else:
        template = """Accurately answer the following question:
                   
{{question}}

Choices:
{{choices}}

Instructions:
- Carefully read the question and all options.
- Let's think step by step and explain your reasoning briefly.
- Then give the final answer starting with The correct answer is"""

    prompt = template.replace("{{question}}", question)
    prompt = prompt.replace("{{choices}}", choices)

    if not use_template:
        prompt = question + "\n\nChoices:\n" + choices

    return prompt


def parse_answer(answer_str: str) -> List[str]:
    """
    Parse answer string to extract valid answer options.
    Converts digits 0/1/2/3 to letters A/B/C/D.
    
    Args:
        answer_str: String containing answer digits
        
    Returns:
        List of parsed answer letters
    """
    if not isinstance(answer_str, str):
        return []
    valid_digits = [c for c in answer_str if c in {'0','1','2','3'}]
    return sorted(list({
        chr(65 + int(d))  # 0->A, 1->B, 2->C, 3->D
        for d in valid_digits
    }))


def extract_answer_from_content(text: str) -> Optional[str]:
    """
    Extract answer from model output with robust multi-pattern matching.
    Supports multiple languages and response formats.
    
    Args:
        text: Model output text
        
    Returns:
        Extracted answer letter or None
    """
    text = text.strip()
    if not text:
        return None

    # Define multiple answer patterns for different languages and formats
    answer_patterns = [
        # English patterns
        r'Answer:\s*(.*)',
        r'answer:\s*(.*)',
        r'ANSWER:\s*(.*)',
        r'Your answer:\s*(.*)',
        r'your answer:\s*(.*)',
        r'YOUR ANSWER:\s*(.*)',
        r'The answer is\s*(.*)',
        r'the answer is\s*(.*)',
        r'THE ANSWER IS\s*(.*)',
        r'Correct answer is\s*(.*)',
        r'correct answer is\s*(.*)',
        r'Correct answer is:\s*(.*)',
        r'correct answer is:\s*(.*)',
        r'Correct answer:\s*(.*)',
        r'correct answer:\s*(.*)',
        r'CORRECT ANSWER:\s*(.*)',
        
        # Swahili patterns
        r'Jibu lako:\s*(.*)',
        r'jibu lako:\s*(.*)',
        r'JIBU LAKO:\s*(.*)',
        r'Jibu:\s*(.*)',
        r'jibu:\s*(.*)',
        r'JIBU:\s*(.*)',
        r'Jibu sahihi:\s*(.*)',
        r'jibu sahihi:\s*(.*)',
        r'JIBU SAHIHI:\s*(.*)',
        
        # Other common patterns
        r'Response:\s*(.*)',
        r'response:\s*(.*)',
        r'RESPONSE:\s*(.*)',
        r'Choice:\s*(.*)',
        r'choice:\s*(.*)',
        r'CHOICE:\s*(.*)',
        r'Option:\s*(.*)',
        r'option:\s*(.*)',
        r'OPTION:\s*(.*)',
    ]
    
    # 1. Try to match any of the answer patterns
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_part = match.group(1).strip()
            # Search for first A-D letter in the matched part
            for char in answer_part:
                if char in {'A', 'B', 'C', 'D'}:
                    return char
    
    # 2. Look for standalone A-D letters that are likely answers
    # Prioritize letters at the end of text or with clear answer-like context
    standalone_patterns = [
        r'\b([A-D])(?:\s*[.,!?:)]?\s*$)',  # A-D at end of text with optional punctuation
        r'\b([A-D])(?:\s*[.,!?:)]\s)',     # A-D followed by punctuation and space
        r'(?:^|\s)([A-D])(?:\s*$)',        # A-D at start or with word boundary at end
    ]
    
    for pattern in standalone_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Check if this looks like mathematical expressions rather than answers
            math_indicators = ['+', '-', '*', '/', '=', '^', 'x^', 'y^', 'z^', 'mod', 'sqrt', 'sin', 'cos', 'tan']
            has_math = any(indicator in text for indicator in math_indicators)
            has_answer_indicators = any(phrase in text.lower() for phrase in ['jibu', 'answer', 'choice', 'option', 'response', 'correct', 'sahihi'])
            
            # If it has math indicators but no answer indicators, it's likely mathematical notation
            if has_math and not has_answer_indicators:
                continue  # Skip this match, try next pattern
            
            return matches[-1].upper()
    
    # 3. Fallback: find all A-D letters but be more selective
    all_letters = re.findall(r'\b([A-D])\b', text, re.IGNORECASE)
    if all_letters:
        # Check if this looks like mathematical expressions rather than answers
        math_indicators = ['+', '-', '*', '/', '=', '^', 'x^', 'y^', 'z^', 'mod', 'sqrt', 'sin', 'cos', 'tan']
        has_math = any(indicator in text for indicator in math_indicators)
        has_answer_indicators = any(phrase in text.lower() for phrase in ['jibu', 'answer', 'choice', 'option', 'response', 'correct', 'sahihi'])
        
        # If it has math indicators but no answer indicators, it's likely mathematical notation
        if has_math and not has_answer_indicators:
            return None
        
        # Otherwise, return the last letter found
        return all_letters[-1].upper()
    
    # 3. Search backwards for any A-D letter as fallback
    for char in reversed(text):
        if char in {'A', 'B', 'C', 'D'}:
            return char

    return None


def apply_generation_config(model: Any, generation_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Apply generation configuration to a model and handle sampling parameters.
    
    This function applies the provided generation config to the model and removes
    sampling parameters (temperature, top_p, top_k, min_p) when do_sample=False
    to avoid warnings from the transformers library. If no config is provided,
    it defaults to greedy decoding with cleaned sampling parameters.
    
    Args:
        model: Model object with generation_config attribute
        generation_config: Optional generation configuration dictionary.
                          If None, defaults to greedy decoding (do_sample=False).
    """
    if not hasattr(model, 'generation_config'):
        return
    
    # If no config provided, default to greedy decoding
    if not generation_config:
        generation_config = {'do_sample': False}
    
    # Apply all configuration parameters
    for key, value in generation_config.items():
        setattr(model.generation_config, key, value)
    
    # Disable sampling parameters if do_sample=False to avoid warnings
    # We set them to None instead of deleting, since some model code may
    # access these attributes unconditionally.
    if not generation_config.get('do_sample', True):
        sampling_params = ['temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty']
        for param in sampling_params:
            try:
                setattr(model.generation_config, param, None)
            except Exception:
                # If the backend does not allow setting, ignore silently
                pass


def _find_router_config_file(router_dir: str) -> Optional[str]:
    for name in ("router.json", "router_config.json"):
        path = os.path.join(router_dir, name)
        if os.path.exists(path):
            return path
    return None


def _find_router_weight_file(router_dir: str) -> Optional[str]:
    for name in ("router.pt", "router.bin"):
        path = os.path.join(router_dir, name)
        if os.path.exists(path):
            return path
    return None


def _load_router_module(
    router_dir: Optional[str],
    device: torch.device,
    num_banks: int,
):
    if router_dir is None:
        return None

    router_config_path = _find_router_config_file(router_dir)
    if router_config_path is None:
        raise FileNotFoundError(
            f"Could not find router config in {router_dir}. "
            "Expected router.json or router_config.json."
        )

    router = load_router(router_config_path, override_args={"num_banks": num_banks})
    router = router.to(device)

    router_weights_path = _find_router_weight_file(router_dir)
    if router_weights_path is not None:
        state_dict = torch.load(router_weights_path, map_location=device)
        router.load_state_dict(state_dict, strict=False)
    return router


def _load_projectors_and_config(
    checkpoint_dir: str,
    device: torch.device,
):
    if checkpoint_dir is None:
        raise KeyError(
            "checkpoints_dir must be provided under model.rosetta_config or eval config"
        )
    num_projectors = len(
        [f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)]
    )
    projector_list = []
    projector_classes = []
    for proj_idx in range(num_projectors):
        json_cfg = os.path.join(checkpoint_dir, f"projector_{proj_idx}.json")
        proj = load_projector(json_cfg)
        projector_classes.append(proj.__class__.__name__)
        proj = proj.to(device)
        pt_path = os.path.join(checkpoint_dir, f"projector_{proj_idx}.pt")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=device)
            proj.load_state_dict(state_dict, strict=False)
        projector_list.append(proj)

    config_path = os.path.join(checkpoint_dir, "projector_bank_config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(checkpoint_dir, "projector_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find projector config in {checkpoint_dir}"
        )

    with open(config_path, "r") as f:
        config = json.load(f)
    return projector_list, projector_classes, config


def _adjust_config_indices(config_dict, proj_offset, actual_source_idx=None):
    adjusted = {}
    for target_model_idx, sources in config_dict.items():
        adjusted[int(target_model_idx)] = {}
        for source_model_idx, layers in sources.items():
            actual_src_idx = (
                actual_source_idx if actual_source_idx is not None else int(source_model_idx)
            )
            adjusted[int(target_model_idx)][actual_src_idx] = {}
            for target_layer_idx, mappings in layers.items():
                adjusted_mappings = []
                for source_layer_idx, idx in mappings:
                    adjusted_mappings.append((source_layer_idx, idx + proj_offset))
                adjusted[int(target_model_idx)][actual_src_idx][
                    int(target_layer_idx)
                ] = adjusted_mappings
    return adjusted


def set_default_chat_template(tokenizer, model_name: str):
    """
    Set default chat template for models without one.
    
    Args:
        tokenizer: Tokenizer object
        model_name: Name of the model
    """
    if tokenizer.chat_template is None:
        if "UlizaLlama3".lower() in model_name.lower():
            tokenizer.chat_template = (
                "{%- for message in messages %}"
                "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}"
                "{{- message['content'] }}"
                "{{- '<|eot_id|>' }}"
                "{%- endfor %}"
                "{%- if add_generation_prompt %}"
                "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                "{%- endif %}"
            )
        else:
            print(f"Model {model_name} has no chat template, setting default template...")
            default_template = """{% for message in messages %}{% if message['role'] == 'user' %}### Human: {{ message['content'] }}{% elif message['role'] == 'assistant' %}### Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}
    {% endif %}{% endfor %}{% if add_generation_prompt %}
    ### Assistant:{% endif %}"""
            tokenizer.chat_template = default_template
            print("Default chat template has been set.")
    else:
        print(f"Model {model_name} already has a chat template.")


def load_hf_model(model_name: str, device: torch.device, generation_config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    """
    Load Hugging Face model and tokenizer.
    
    Args:
        model_name: Model name or path
        device: Device to load model on
        generation_config: Optional generation configuration to apply
        
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name),
        trust_remote_code=True,
        padding_side='left'
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check and set chat template
    set_default_chat_template(tokenizer, model_name)

    if model_name == "google/gemma-3-1b-it":
        torch._dynamo.config.cache_size_limit = 64
        model = AutoModelForCausalLM.from_pretrained(
            str(model_name),
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            sliding_window=4096
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_name),
            torch_dtype=torch.bfloat16,
            device_map={"": device}
    ).eval()
    
    # Apply generation config
    apply_generation_config(model, generation_config)
    
    return model, tokenizer


def load_rosetta_model(model_config: Dict[str, Any], eval_config: Dict[str, Any], 
                      device: torch.device, generation_config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    """
    Load Rosetta model with projectors.
    
    Args:
        model_config: Model configuration dict
        eval_config: Evaluation configuration dict
        device: Device to load model on
        generation_config: Optional generation configuration to apply
        
    Returns:
        Tuple of (rosetta_model, tokenizer)
    """
    # Prefer checkpoints_dir under model.rosetta_config; fall back to eval config for backward compatibility
    rosetta_config = model_config["rosetta_config"]
    slm_model_path = rosetta_config["base_model"]
    teacher_model_config = rosetta_config["teacher_model"]
    routing_config = rosetta_config.get("routing", {})
    static_gate_enabled = rosetta_config.get("static_gate_enabled", True)
    entropy_gate_enabled = rosetta_config.get("entropy_gate_enabled", True)
    threshold_abs = rosetta_config.get("threshold_abs", 0.45)
    threshold_rel = rosetta_config.get("threshold_rel", 0.15)
    include_response = rosetta_config.get("include_response", False)
    multi_source_fusion_mode = rosetta_config.get("multi_source_fusion_mode", "sequential")
    update_decode_past = rosetta_config.get("update_decode_past", True)
    router_fuse_threshold = routing_config.get(
        "fuse_threshold",
        rosetta_config.get("router_fuse_threshold", 0.5),
    )
    projector_bank_dirs = routing_config.get(
        "projector_bank_dirs",
        rosetta_config.get("projector_bank_dirs"),
    )
    router_dir = routing_config.get("router_dir", rosetta_config.get("router_dir"))
    checkpoint_dir_value = rosetta_config.get(
        "checkpoints_dir",
        eval_config.get("checkpoints_dir"),
    )
    
    llm_configs = []  # List of (model_path, checkpoint_dir) tuples
    
    if isinstance(teacher_model_config, str):
        checkpoint_dir = checkpoint_dir_value
        if projector_bank_dirs is None and isinstance(checkpoint_dir_value, list):
            projector_bank_dirs = checkpoint_dir_value
            checkpoint_dir = checkpoint_dir_value[0] if checkpoint_dir_value else None
        llm_configs.append((teacher_model_config, checkpoint_dir))
    
    elif isinstance(teacher_model_config, dict):
        if projector_bank_dirs is not None or router_dir is not None:
            raise ValueError(
                "Routing v1 supports single-teacher only. "
                "Use a single teacher_model with projector_bank_dirs/router_dir."
            )
        checkpoints_dir = checkpoint_dir_value or []
        model_items = list(teacher_model_config.items())
        
        if len(checkpoints_dir) != len(model_items):
            raise ValueError(f"Number of checkpoints ({len(checkpoints_dir)}) must match number of models ({len(model_items)})")
        
        for (model_name, model_path), ckpt_dir in zip(model_items, checkpoints_dir):
            llm_configs.append((model_path, ckpt_dir))

    # Load tokenizer
    slm_tokenizer = AutoTokenizer.from_pretrained(str(slm_model_path))
    set_default_chat_template(slm_tokenizer, slm_model_path)
    
    # Load SLM model
    slm_model = AutoModelForCausalLM.from_pretrained(
        str(slm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    # Apply generation config to SLM
    apply_generation_config(slm_model, generation_config)
    
    # Load LLM models
    llm_models = []
    for llm_model_path, _ in llm_configs:
        if llm_model_path == "google/gemma-3-1b-it":
            llm_model = AutoModelForCausalLM.from_pretrained(
                str(llm_model_path),
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                sliding_window=4096
            ).eval()
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(
                str(llm_model_path),
                torch_dtype=torch.bfloat16,
                device_map={"": device}
            ).eval()
        
        # Apply generation config to LLM
        apply_generation_config(llm_model, generation_config)
        llm_models.append(llm_model)

    model_list = [slm_model] + llm_models

    if projector_bank_dirs is not None or router_dir is not None:
        if not isinstance(teacher_model_config, str):
            raise ValueError("Routing loader requires a single teacher_model string")
        if projector_bank_dirs is None:
            checkpoint_dir = llm_configs[0][1]
            if checkpoint_dir is None:
                raise KeyError(
                    "projector_bank_dirs or checkpoints_dir must be provided for routing"
                )
            projector_bank_dirs = [checkpoint_dir]

        projector_list = []
        projector_bank_dicts = []
        expected_projector_classes = None

        for bank_dir in projector_bank_dirs:
            bank_projectors, projector_classes, raw_config = _load_projectors_and_config(
                bank_dir,
                device,
            )
            if raw_config.get("format") == "projector_banks_v1":
                bank_configs = raw_config.get("bank_configs", [])
                if len(bank_configs) != 1:
                    raise ValueError(
                        "projector_bank_dirs must point to single-bank checkpoint directories"
                    )
                raw_config = bank_configs[0]

            if expected_projector_classes is None:
                expected_projector_classes = projector_classes
            elif projector_classes != expected_projector_classes:
                raise ValueError(
                    "All routing banks must share the same projector type ordering"
                )

            proj_offset = len(projector_list)
            projector_list.extend(bank_projectors)
            projector_bank_dicts.append(
                _adjust_config_indices(raw_config, proj_offset, actual_source_idx=1)
            )

        if len(projector_bank_dicts) > 1 and router_dir is None:
            candidate_router_dir = projector_bank_dirs[0]
            if _find_router_config_file(candidate_router_dir) is not None:
                router_dir = candidate_router_dir
            else:
                raise ValueError(
                    "Multiple projector banks require router_dir (or router files "
                    "co-located in the first bank directory)."
                )

        router = _load_router_module(router_dir, device, len(projector_bank_dicts))
        rosetta_model = RosettaModel(
            model_list=model_list,
            base_model_idx=0,
            projector_list=projector_list,
            router=router,
            projector_bank_dicts=projector_bank_dicts,
            router_fuse_threshold=router_fuse_threshold,
            include_response=include_response,
            multi_source_fusion_mode=multi_source_fusion_mode,
            static_gate_enabled=static_gate_enabled,
            entropy_gate_enabled=entropy_gate_enabled,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            update_decode_past=update_decode_past,
        ).to(device).eval()
        return rosetta_model, slm_tokenizer

    projector_list = []
    projector_offsets = [0]
    raw_projector_configs = []

    for _, checkpoint_dir in llm_configs:
        bank_projectors, _, raw_config = _load_projectors_and_config(checkpoint_dir, device)
        if raw_config.get("format") == "projector_banks_v1":
            bank_configs = raw_config.get("bank_configs", [])
            if len(bank_configs) != 1:
                raise ValueError(
                    "Legacy multi-source loading expects one bank per checkpoint directory"
                )
            raw_config = bank_configs[0]
        projector_list.extend(bank_projectors)
        raw_projector_configs.append(raw_config)
        projector_offsets.append(len(projector_list))

    rosetta_model = RosettaModel(
        model_list=model_list,
        base_model_idx=0,
        projector_list=projector_list,
        include_response=include_response,
        multi_source_fusion_mode=multi_source_fusion_mode,
        static_gate_enabled=static_gate_enabled,
        entropy_gate_enabled=entropy_gate_enabled,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        update_decode_past=update_decode_past,
    ).to(device).eval()

    for llm_idx, raw_config in enumerate(raw_projector_configs):
        adjusted_config = _adjust_config_indices(
            raw_config,
            projector_offsets[llm_idx],
            actual_source_idx=llm_idx + 1,
        )
        for target_idx, sources in adjusted_config.items():
            if target_idx not in rosetta_model.projector_dict:
                rosetta_model.projector_dict[target_idx] = {}
            for source_idx, layers in sources.items():
                if source_idx not in rosetta_model.projector_dict[target_idx]:
                    rosetta_model.projector_dict[target_idx][source_idx] = {}
                rosetta_model.projector_dict[target_idx][source_idx].update(layers)

    return rosetta_model, slm_tokenizer


def load_oracle_rosetta_model(model_config: Dict[str, Any], eval_config: Dict[str, Any], 
                      device: torch.device) -> Tuple[Any, Any]:
    """
    Load Rosetta model with projectors.
    
    Args:
        model_config: Model configuration dict
        eval_config: Evaluation configuration dict
        device: Device to load model on
        
    Returns:
        Tuple of (rosetta_model, tokenizer)
    """
    # Prefer checkpoints_dir under model.rosetta_config; fall back to eval config for backward compatibility
    rosetta_config = model_config["rosetta_config"]
    checkpoint_dir = rosetta_config.get("checkpoints_dir", eval_config.get("checkpoints_dir"))
    if checkpoint_dir is None:
        raise KeyError("checkpoints_dir must be provided under model.rosetta_config (preferred) or eval config (legacy)")
    slm_model_path = rosetta_config["base_model"]
    llm_model_path = rosetta_config["teacher_model"]

    # Load tokenizer
    slm_tokenizer = AutoTokenizer.from_pretrained(str(slm_model_path))
    set_default_chat_template(slm_tokenizer, slm_model_path)
    
    # Load models
    slm_model = AutoModelForCausalLM.from_pretrained(
        str(slm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        str(llm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    # Load projectors
    num_projectors = len([f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)])
    projector_list = []
    for t in range(num_projectors):
        json_cfg = os.path.join(checkpoint_dir, f"projector_{t}.json")
        proj = load_projector(json_cfg)
        proj = proj.to(device)
        pt_path = os.path.join(checkpoint_dir, f"projector_{t}.pt")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=device)
            proj.load_state_dict(state_dict, strict=False)
        projector_list.append(proj)
    
    # Initialize Rosetta model
    rosetta_model = OracleRosettaModel(
        model_list=[slm_model, llm_model],
        base_model_idx=0,
        projector_list=projector_list,
    ).to(device).eval()

    # Load projector mapping configs
    proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
    rosetta_model.load_projector_config(proj_cfg_path)

    return rosetta_model, slm_tokenizer


def get_option_token_ids(tokenizer, num_options: int = 4) -> List[int]:
    """
    Get token IDs for options A, B, C, D (or more up to J).
    
    Args:
        tokenizer: Tokenizer object
        num_options: Number of options to get (default 4 for A-D, max 10 for A-J)
        
    Returns:
        List of token IDs for options
    """
    # Limit to maximum of 10 options (A-J)
    num_options = min(num_options, 10)
    option_ids = []
    for i in range(num_options):
        letter = chr(65 + i)  # A=65, B=66, etc.
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        option_ids.append(ids[0] if ids else tokenizer.eos_token_id)
    return option_ids

"""
Deprecated
"""

@torch.no_grad()
def generate_answer_with_logits(model, tokenizer, prompt: str, option_ids: List[int], 
                               device: torch.device, model_type: str = "hf") -> Tuple[str, np.ndarray]:
    """
    Generate answer using logits method.
    
    Args:
        model: Model object
        tokenizer: Tokenizer object
        prompt: Input prompt
        option_ids: Token IDs for options A, B, C, D
        device: Device to run on
        model_type: Type of model ("rosetta", "qwen", or "hf")
        
    Returns:
        Tuple of (predicted_answer, probabilities)
    """
    messages = [{
        "role": "user",
        "content": prompt
    }]
    
    # Try to apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False if model_type == "qwen" else None
        )
    except Exception as e:
        print(f"Failed to apply chat template for {model_type} model: {e}")
        text = f"### Human: {prompt}\n### Assistant:"
    
    text += "The correct answer is"
    input_ids = tokenizer(text, return_tensors="pt").to(device)['input_ids']
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    
    if model_type == "rosetta":
        instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(input_ids.shape[1]-1, 1).unsqueeze(0).to(device)
        response_index = torch.tensor([[-1, 0]], dtype=torch.long).unsqueeze(0)
        outputs = model.forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            kv_cache_index=[instruction_index, response_index]
        )
    else:
        outputs = model(input_ids)
    
    logits = outputs.logits[0, -1]
    option_logits = torch.tensor([
        logits[option_ids[0]].item(),
        logits[option_ids[1]].item(),
        logits[option_ids[2]].item(),
        logits[option_ids[3]].item()
    ])
    
    probs = torch.nn.functional.softmax(option_logits, dim=0).numpy()
    pred = chr(65 + np.argmax(probs))
    return pred, probs


@torch.no_grad()
def generate_answer_with_generate(model, tokenizer, prompt: str, device: torch.device,
                                 model_type: str = "hf") -> Tuple[str, np.ndarray, int, int, str]:
    """
    Generate answer using text generation method.
    
    Args:
        model: Model object
        tokenizer: Tokenizer object
        prompt: Input prompt
        device: Device to run on
        model_type: Type of model ("rosetta" or "hf")
        
    Returns:
        Tuple of (predicted_answer, probabilities, input_length, generation_length, generated_text)
    """
    messages = [{
        "role": "user",
        "content": prompt
    }]
    
    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception as e:
        print(f"Failed to apply chat template: {e}")
        text = f"### Human: {prompt}\n### Assistant:"

    # Prepare model input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generation parameters
    sampling_params = {
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'min_p': 0.0,
        'repetition_penalty': 1.2,
        'max_new_tokens': 1024
    }
    
    # Generate text
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        **sampling_params
    )
    
    # Parse output
    if isinstance(model, RosettaModel):
        generated_ids = outputs[0]
    else:
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n")
    
    # Extract answer
    pred = extract_answer_from_content(content)
    
    # Return uniform distribution for generate method
    probs = np.array([0.25, 0.25, 0.25, 0.25])

    input_length = inputs.input_ids.shape[1]
    gen_length = generated_ids.shape[0]

    return pred, probs, input_length, gen_length, content
