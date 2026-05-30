"""
SFT entrypoint for C2C v4.

The training loop is reused from SFT_train.py. This file only registers the v4
projector and swaps in the v4 Rosetta wrapper.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.model import projector_v4  # noqa: F401
from rosetta.model.wrapper_v4 import RosettaModel
from script.train import SFT_train as sft


sft.RosettaModel = RosettaModel


if __name__ == "__main__":
    sft.main()
