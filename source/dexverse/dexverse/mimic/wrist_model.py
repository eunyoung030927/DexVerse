# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-form floating-hand wrist kinematics and its per-task cache (numpy / scipy only, no Isaac Sim).

:class:`FloatingHandMimicEnv` measures the wrist model of every hand when it starts and writes it, with the
action-column -> joint mapping, to ``<cache dir>/<task>__<robot_type>.json`` (``$DEXVERSE_MIMIC_CACHE`` or
``~/.cache/dexverse/mimic``). The offline annotator reads that file to compute palm poses from recorded joint
states without starting the simulator.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

_AXIS_LETTERS = "XYZ"
CACHE_VERSION = 1


class WristModel:
    """Closed-form palm pose <-> wrist joint values of one floating hand (all quantities in the robot root frame)."""

    def __init__(self, trans_axes, rot_axes, centre, palm_offset, palm_rot0):
        self.A = np.asarray(trans_axes, dtype=np.float64).T          # (3, 3): columns = prismatic axes
        self.w = np.asarray(rot_axes, dtype=np.float64)              # (3, 3): rows = revolute axes, chain order
        self.c = np.asarray(centre, dtype=np.float64)
        self.d = np.asarray(palm_offset, dtype=np.float64)
        self.R0 = palm_rot0 if isinstance(palm_rot0, Rotation) else Rotation.from_matrix(np.asarray(palm_rot0))
        # The revolute axes of every DexVerse floating hand are signed coordinate axes -> an Euler sequence.
        idx, sign = [], []
        for k in range(3):
            i = int(np.argmax(np.abs(self.w[k])))
            if abs(self.w[k, i]) < 0.999:
                raise RuntimeError(f"wrist revolute axis {k} = {self.w[k]} is not a coordinate axis")
            idx.append(i)
            sign.append(float(np.sign(self.w[k, i])))
        if len(set(idx)) != 3:
            raise RuntimeError(f"wrist revolute axes {idx} are not three distinct axes")
        self.seq = "".join(_AXIS_LETTERS[i] for i in idx)          # intrinsic: R = R_a(t1) R_b(t2) R_c(t3)
        self.sign = np.asarray(sign)

    def to_dict(self) -> dict:
        return {"trans_axes": self.A.T.tolist(), "rot_axes": self.w.tolist(), "centre": self.c.tolist(),
                "palm_offset": self.d.tolist(), "palm_rot0": self.R0.as_matrix().tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> WristModel:
        return cls(d["trans_axes"], d["rot_axes"], d["centre"], d["palm_offset"], np.asarray(d["palm_rot0"]))

    def wrist_rot(self, q_r) -> Rotation:
        return Rotation.from_euler(self.seq, np.asarray(q_r, dtype=np.float64) * self.sign)

    def forward(self, q_t, q_r) -> tuple[np.ndarray, Rotation]:
        rw = self.wrist_rot(q_r)
        return self.c + self.A @ np.asarray(q_t, dtype=np.float64) + rw.apply(self.d), rw * self.R0

    def inverse(self, pos, rot: Rotation, q_r_ref) -> tuple[np.ndarray, np.ndarray]:
        """Wrist joint values for a palm pose; among the Euler solutions the one nearest ``q_r_ref``."""
        rw = rot * self.R0.inv()
        t = rw.as_euler(self.seq)
        ref = np.asarray(q_r_ref, dtype=np.float64) * self.sign
        best, best_d = None, np.inf
        for cand in (t, np.array([t[0] + np.pi, np.pi - t[1], t[2] + np.pi]),
                     np.array([t[0] + np.pi, -np.pi - t[1], t[2] + np.pi])):
            if (Rotation.from_euler(self.seq, cand) * rw.inv()).magnitude() > 1e-6:
                continue
            cand = cand + 2.0 * np.pi * np.round((ref - cand) / (2.0 * np.pi))
            dist = float(np.abs(cand - ref).sum())
            if dist < best_d:
                best, best_d = cand, dist
        q_r = best * self.sign
        q_t = np.linalg.solve(self.A, np.asarray(pos, dtype=np.float64) - self.c - rw.apply(self.d))
        return q_t, q_r


def cache_dir() -> str:
    return os.environ.get("DEXVERSE_MIMIC_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "dexverse",
                                                                   "mimic")


def cache_path(task: str, robot_type: str) -> str:
    return os.path.join(cache_dir(), f"{task.split(':')[-1]}__{robot_type}.json")


def save_cache(task: str, robot_type: str, data: dict) -> str:
    path = cache_path(task, robot_type)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": CACHE_VERSION, "task": task, "robot_type": robot_type, **data}, f, indent=1)
    os.replace(tmp, path)
    return path


def load_cache(task: str, robot_type: str) -> dict | None:
    path = cache_path(task, robot_type)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if data.get("version") == CACHE_VERSION else None
