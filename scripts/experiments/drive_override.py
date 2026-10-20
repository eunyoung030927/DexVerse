# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run scripts/record_demos.py with the robot's finger drives overridden (no robot config change).

``DEXVERSE_DRIVE_OVERRIDE`` is applied to ``env_cfg.scene.robot.actuators`` right before ``gym.make``, to every
actuator whose name contains the given substring and not "base" (the floating wrist):

* ``"<substring>:<attr>=<value>,...;..."``  scalar values, e.g. ``"hand:effort_limit_sim=1"``;
* ``"@<hand>"``  the ``override`` entry of ``<hand>`` in ``DEXVERSE_DRIVE_SPECS`` (default
  ``source/dexverse/dexverse/data_collection/official_finger_drives.json``, the official / reference drives):
  ``{"<substring>": {"<attr>": scalar | {joint regex: value}, "_drop": [group keys]}}``.

A dict value is merged into the actuator's existing dict (keys in ``_drop`` removed first, e.g. a group key replaced
by per-joint keys); otherwise it replaces the value. Every change is printed.

    DEXVERSE_DRIVE_OVERRIDE=@allegro python scripts/experiments/drive_override.py --task ... --robot_type ... \
        --replay_demos <demos.pkl> --replay_states ...      (record_demos.py arguments, run from the repo root)
"""

import json
import os
import runpy
import sys

import gymnasium as gym

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPECS = os.environ.get("DEXVERSE_DRIVE_SPECS") or os.path.join(
    ROOT, "source", "dexverse", "dexverse", "data_collection", "official_finger_drives.json")
SPEC = os.environ.get("DEXVERSE_DRIVE_OVERRIDE", "").strip()
_make = gym.make


def _parse() -> dict:
    if SPEC.startswith("@"):
        entry = json.load(open(SPECS))[SPEC[1:]]
        return entry.get("override", entry)
    out = {}
    for part in filter(None, SPEC.split(";")):
        sub, kv = part.split(":", 1)
        out[sub] = {k: float(v) for k, v in (a.split("=") for a in kv.split(","))}
    return out


def _apply(cfg):
    acts = cfg.scene.robot.actuators
    for sub, attrs in _parse().items():
        hit = [n for n in acts if sub in n and "base" not in n]
        if not hit:
            raise KeyError(f"[drive_override] no actuator matches {sub!r}: {list(acts)}")
        drop = attrs.get("_drop", [])
        for n in hit:
            for k, v in attrs.items():
                if k == "_drop":
                    continue
                old = getattr(acts[n], k)
                if isinstance(v, dict) and isinstance(old, dict):
                    new = {kk: vv for kk, vv in old.items() if kk not in drop}
                    new.update(v)
                else:
                    new = v
                setattr(acts[n], k, new)
                print(f"[drive_override] {n}.{k}: {old} -> {new}", flush=True)


def make(task, *a, cfg=None, **kw):
    if cfg is not None and SPEC:
        _apply(cfg)
    return _make(task, *a, cfg=cfg, **kw)


gym.make = make
print(f"[drive_override] spec={SPEC!r}", flush=True)
sys.argv = ["scripts/record_demos.py"] + sys.argv[1:]
runpy.run_path("scripts/record_demos.py", run_name="__main__")
