"""
SFT entrypoint for hidden-state C2C v3.

This reuses the upstream SFT_train.py training loop and only swaps the Rosetta
wrapper. Importing rosetta.model.projector_v3 registers C2CHiddenStateProjector
in the existing projector registry, so the original config-driven projector
creation path continues to work.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.model import projector_v3  # noqa: F401
from rosetta.model.wrapper_v3 import RosettaModel
from script.train import SFT_train as sft


sft.RosettaModel = RosettaModel


if __name__ == "__main__":
    sft.main()
