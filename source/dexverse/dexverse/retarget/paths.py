# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Where the cross-hand retargeting reads demos and writes its outputs.

Outputs go to ``$DEXVERSE_RETARGET_RUNS`` (default ``<repo>/results/retarget``, git-ignored):

    <runs>/<task>/<source pickle stem>/epNNN/reference.npz          stage 1
    <runs>/<task>/<source pickle stem>/epNNN/<robot_type>/...       stages 2-4
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def runs_root() -> Path:
    return Path(os.environ.get("DEXVERSE_RETARGET_RUNS") or REPO / "results" / "retarget")


def run_dir(task: str, demos: str, episode: int) -> Path:
    d = runs_root() / task.split(":")[-1] / Path(demos).stem / f"ep{episode:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_demos(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
