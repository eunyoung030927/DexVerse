# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Finger drive overrides of the floating hands (no robot config change).

A spec is applied to ``env_cfg.scene.robot.actuators`` before ``gym.make``, to every actuator whose name contains
the given substring and not "base" (the floating wrist):

* ``"<substring>:<attr>=<value>,...;..."``  scalar values, e.g. ``"hand:effort_limit_sim=1"``;
* ``"@<hand>"``  the ``override`` entry of ``<hand>`` in ``official_finger_drives.json`` (next to this module, or
  ``$DEXVERSE_DRIVE_SPECS``): the official / reference drives, ``{"<substring>": {"<attr>": scalar |
  {joint regex: value}, "_drop": [group keys]}}``.

A dict value is merged into the actuator's existing dict (keys in ``_drop`` removed first, e.g. a group key replaced
by per-joint keys); otherwise it replaces the value. Values and sources: ``docs/finger_drives_official.md``.
"""

from __future__ import annotations

import json
import os

SPECS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "official_finger_drives.json")


def parse_drive_spec(spec: str) -> dict:
    spec = (spec or "").strip()
    if not spec:
        return {}
    if spec.startswith("@"):
        entry = json.load(open(os.environ.get("DEXVERSE_DRIVE_SPECS") or SPECS_PATH))[spec[1:]]
        return entry.get("override", entry)
    out = {}
    for part in filter(None, spec.split(";")):
        sub, kv = part.split(":", 1)
        out[sub] = {k: float(v) for k, v in (a.split("=") for a in kv.split(","))}
    return out


def apply_drive_spec(env_cfg, spec: str, log=print) -> list[str]:
    """Apply ``spec`` to the robot actuators of ``env_cfg``; returns the change lines (also logged)."""
    acts = env_cfg.scene.robot.actuators
    lines = []
    for sub, attrs in parse_drive_spec(spec).items():
        hit = [n for n in acts if sub in n and "base" not in n]
        if not hit:
            raise KeyError(f"[finger_drives] no actuator matches {sub!r}: {list(acts)}")
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
                lines.append(f"[finger_drives] {n}.{k}: {old} -> {new}")
                if log:
                    log(lines[-1])
    return lines
