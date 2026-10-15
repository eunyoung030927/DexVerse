# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decode the EE action versions recorded by :mod:`lerobot_spool` (e.g. a policy trained on one) to env actions.

:class:`EEActionDecoder` maps rows of ``action.ee_abs`` / ``action.ee_abs_rot6d`` / ``action.ee_delta`` /
``action.ee_delta_init`` back to the env action: 6D -> rotation by Gram-Schmidt (Zhou et al. 2019), deltas
composed onto the previous command / the measured start pose, and the wrist rotation decomposed into the
virtual-joint values nearest the previous command, so a rotation crossing +-pi gives no jump at the joint level.
``scripts/data_tools/check_ee_decode.py`` checks the round trip on a dataset.

Like :mod:`lerobot_spool`, the module body is stdlib + numpy (scipy inside functions), so it loads BY PATH
outside Isaac Sim; it then loads its sibling ``lerobot_spool.py`` by path as well.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np


def _load_spool():
    try:
        from . import lerobot_spool  # noqa: PLC0415  (imported as part of the dexverse package)

        return lerobot_spool
    except ImportError:
        pass
    name = "_dexverse_lerobot_spool"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerobot_spool.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_S = _load_spool()


def rot6d_to_rotation(x):
    """Rotation from 6D vectors in the layout of :func:`rot6d` (first two matrix COLUMNS) by Gram-Schmidt
    (Zhou et al. 2019): any (N, 6) whose halves are not parallel gives a valid rotation, so raw network outputs
    need no other projection. Returns a scipy Rotation of length N."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    x = np.asarray(x, dtype=np.float64).reshape(-1, 6)
    c0 = x[:, :3] / np.linalg.norm(x[:, :3], axis=1, keepdims=True)
    c1 = x[:, 3:] - (c0 * x[:, 3:]).sum(1, keepdims=True) * c0
    c1 /= np.linalg.norm(c1, axis=1, keepdims=True)
    return Rotation.from_matrix(np.stack([c0, c1, np.cross(c0, c1)], axis=2))


def wrist_rot_joint_values(rot, hand, q_ref, gn_iters: int = 3) -> np.ndarray:
    """Revolute virtual-joint values (chain order, (N, 3)) whose product of exponentials is ``rot`` (Rotation,
    N): the inverse of the rotation half of :func:`lerobot_spool._chain_pose`. Of the two decompositions (Davenport angles
    about the measured axes, orthogonalised for the initial guess) and their 2*pi copies, the one nearest
    ``q_ref`` ((N, 3) or (3,), e.g. the previous command) is taken, so a rotation crossing +-pi stays continuous
    at the joint level; Gauss-Newton on the exact measured axes then removes the orthogonalisation error. Near
    gimbal lock (middle joint at its singular angle) the split between the outer joints is ill-conditioned.
    Joint limits are not applied."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    axes = np.asarray([j["axis"] for j in hand["rot"]], dtype=np.float64)
    if len(axes) != 3:
        raise ValueError(f"wrist decoding needs 3 revolute joints, the {hand['side']} hand has {len(axes)}")
    n1 = axes[0] / np.linalg.norm(axes[0])
    n2 = axes[1] - (axes[1] @ n1) * n1
    n2 /= np.linalg.norm(n2)
    n3 = axes[2] - (axes[2] @ n2) * n2
    n3 /= np.linalg.norm(n3)
    q = rot.as_davenport(np.stack([n1, n2, n3]), "intrinsic")
    lam = np.arctan2(np.cross(n1, n2) @ n3, n1 @ n3)
    alt = q + np.array([np.pi, 0.0, np.pi])
    alt[:, 1] = 2.0 * lam - q[:, 1]
    q_ref = np.broadcast_to(np.asarray(q_ref, dtype=np.float64), q.shape)
    cands = [c + 2.0 * np.pi * np.round((q_ref - c) / (2.0 * np.pi)) for c in (q, alt)]
    pick = np.linalg.norm(cands[1] - q_ref, axis=1) < np.linalg.norm(cands[0] - q_ref, axis=1)
    q = np.where(pick[:, None], cands[1], cands[0])

    def err(qq):
        return (_S._chain_pose(np.zeros((len(qq), len(hand["trans"]))), qq, hand)[1] * rot.inv()).as_rotvec()

    eps = 1e-7
    for _ in range(gn_iters):
        e0 = err(q)
        jac = np.stack([(err(q + eps * np.eye(3)[k]) - e0) / eps for k in range(3)], axis=2)  # (N, 3, 3)
        jt = jac.transpose(0, 2, 1)
        q = q - np.linalg.solve(jt @ jac + 1e-12 * np.eye(3), (jt @ e0[:, :, None]))[:, :, 0]
    return q


class EEActionDecoder:
    """EE-layout action rows (one of the four ``action.ee_*`` keys, e.g. a policy output) -> env actions.

    ``hands`` is ``meta/isaac_tasks.json["ee_hands"]``, ``n_action`` the env action dimension. Stateful per
    episode: ``action.ee_delta`` composes onto the previously decoded command (the home pose after
    :meth:`reset`), ``action.ee_delta_init`` onto the measured start pose given to :meth:`reset` (the first
    frame's ``observation.state.ee`` or ``observation.state.ee_rot6d`` row), and each rotation is decomposed
    into the joint values nearest the previous command. Finger columns pass through. Rows must be fed in
    execution order; a chunk (K, n) is decoded row by row.
    """

    def __init__(self, hands: list[dict], n_action: int, kind: str):
        if kind not in _S.EE_ACTION_BLOCKS:
            raise ValueError(f"kind must be one of {list(_S.EE_ACTION_BLOCKS)}, got {kind!r}")
        self.hands, self.n_action, self.kind = hands, int(n_action), kind
        self.items = _S._ee_layout_items(self.n_action, hands)
        self.width = len(_S.EE_ACTION_BLOCKS[kind])
        self.reset()

    def _split(self, row, width):
        """``{("wrist", hand_i) | ("col", c): values}`` of one EE-layout row whose wrist blocks are ``width`` wide."""
        out, k = {}, 0
        for kind, i in self.items:
            w = width if kind == "wrist" else 1
            out[(kind, i)] = row[k:k + w]
            k += w
        if k != len(row):
            raise ValueError(f"EE row has {len(row)} columns, the layout needs {k}")
        return out

    @staticmethod
    def _pose(block):
        """(p, Rotation) of a wrist block in the rotvec (6) or 6D (9) layout."""
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        if len(block) == 6:
            return block[:3], Rotation.from_rotvec(block[3:6])
        return block[:3], rot6d_to_rotation(block[3:9])[0]

    def reset(self, start_state_ee=None):
        """Start an episode at the home pose; ``start_state_ee`` (measured start pose row) is required for
        ``action.ee_delta_init``."""
        zero = np.zeros((1, self.n_action))
        self._prev = []
        for h in self.hands:
            p, rot, _, _ = _S.wrist_command_poses(zero, h)
            self._prev.append({"p": p[0], "rot": rot[0], "q_r": _S._wrist_targets(zero, h["rot"])[0]})
        self._start = None
        if start_state_ee is not None:
            row = np.asarray(start_state_ee, dtype=np.float64).reshape(-1)
            width = 6 if len(row) == len(self.items) + 5 * len(self.hands) else 9
            blocks = self._split(row, width)
            self._start = [self._pose(blocks[("wrist", i)]) for i in range(len(self.hands))]

    def __call__(self, ee) -> np.ndarray:
        """Env actions (K, n_action) float32 for EE rows (K, n) or (n,)."""
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        if self.kind == "action.ee_delta_init" and self._start is None:
            raise ValueError("action.ee_delta_init needs the episode's measured start pose (reset(start_state_ee))")
        rows = np.atleast_2d(np.asarray(ee, dtype=np.float64))
        out = np.zeros((len(rows), self.n_action))
        for t, row in enumerate(rows):
            blocks = self._split(row, self.width)
            for i, h in enumerate(self.hands):
                b, prev = blocks[("wrist", i)], self._prev[i]
                if self.kind in ("action.ee_abs", "action.ee_abs_rot6d"):
                    p, rot = self._pose(b)
                elif self.kind == "action.ee_delta":
                    p, rot = prev["p"] + b[:3], Rotation.from_rotvec(b[3:6]) * prev["rot"]
                else:
                    p0, r0 = self._start[i]
                    p, rot = p0 + b[:3], Rotation.from_rotvec(b[3:6]) * r0
                q_r = wrist_rot_joint_values(Rotation.from_quat(rot.as_quat()[None]), h, prev["q_r"])[0]
                q_t = p @ np.linalg.pinv(np.asarray([j["axis"] for j in h["trans"]], dtype=np.float64))
                for j, q in zip(h["trans"] + h["rot"], np.concatenate([q_t, q_r])):
                    out[t, j["idx"]] = (q - j["offset"]) / j["scale"]
                self._prev[i] = {"p": p, "rot": rot, "q_r": q_r}
            for (kind, c), v in blocks.items():
                if kind == "col":
                    out[t, c] = v[0]
        return out.astype(np.float32)
