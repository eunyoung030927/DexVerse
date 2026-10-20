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


# Output-format versions: a stage's output is reused only if it was written by the current definition.
#   reference.npz 2: contacts on the tip's collision surface (2026-10-13; 1: tip origin)
#   kinematic.npz 2: hand-volume penetration term + step cap (2026-10-13; 1: fingertip-only IK)
REFERENCE_VERSION = 2
KINEMATIC_VERSION = 2


def output_version(npz_path) -> int:
    """Version stamp of a stage-1/2 npz (0 if the file is missing or unreadable, 1 if unstamped)."""
    import json  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    try:
        with np.load(npz_path, allow_pickle=True) as z:
            if "version" in z:
                return int(z["version"])
            if "meta" in z:
                return int(json.loads(str(z["meta"])).get("version", 1))
            return 1
    except Exception:
        return 0
