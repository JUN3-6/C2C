"""
Unified evaluator entrypoint for C2C v4 checkpoints.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch._dynamo as dynamo
import torch.multiprocessing as mp

import rosetta.utils.evaluate as eval_utils
from rosetta.model.projector_v4 import load_projector
from rosetta.model.wrapper_v4 import RosettaModel
from script.evaluation import unified_evaluator as base_eval


eval_utils.load_projector = load_projector
eval_utils.RosettaModel = RosettaModel
base_eval.load_rosetta_model = eval_utils.load_rosetta_model
base_eval.RosettaModel = RosettaModel


if __name__ == "__main__":
    dynamo.config.cache_size_limit = 64
    mp.set_start_method("spawn", force=True)
    base_eval.main()
