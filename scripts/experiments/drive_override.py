# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run scripts/record_demos.py with the robot's finger drives overridden (no robot config change).

``DEXVERSE_DRIVE_OVERRIDE`` is applied to ``env_cfg.scene.robot.actuators`` right before ``gym.make``, to every
actuator whose name contains the given substring and not "base" (the floating wrist):

* ``"<substring>:<attr>=<value>,...;..."``  scalar values, e.g. ``"hand:effort_limit_sim=1"``;
* ``"@<hand>"``  the official / reference drives of ``<hand>`` (``data_collection/official_finger_drives.json``).

Spec format and merge rules: ``dexverse.data_collection.finger_drives``. Every change is printed.

    DEXVERSE_DRIVE_OVERRIDE=@allegro python scripts/experiments/drive_override.py --task ... --robot_type ... \
        --replay_demos <demos.pkl> --replay_states ...      (record_demos.py arguments, run from the repo root)
"""

import os
import runpy
import sys

import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "source", "dexverse"))
from dexverse.data_collection.finger_drives import apply_drive_spec  # noqa: E402

SPEC = os.environ.get("DEXVERSE_DRIVE_OVERRIDE", "").strip()
_make = gym.make


def make(task, *a, cfg=None, **kw):
    if cfg is not None and SPEC:
        apply_drive_spec(cfg, SPEC, log=lambda m: print(m, flush=True))
    return _make(task, *a, cfg=cfg, **kw)


gym.make = make
print(f"[drive_override] spec={SPEC!r}", flush=True)
sys.argv = ["scripts/record_demos.py"] + sys.argv[1:]
runpy.run_path("scripts/record_demos.py", run_name="__main__")
