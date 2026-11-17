"""Batched numpy forward kinematics of the simulated floating hands, built from the env's own USD joint table
(scripts/extract_usd_kinematics.py -> runs/stage1/usd_kin/<hand>.json). Body frames = Isaac body link frames.

T_parent_child(q) = T(pos0, rot0) @ M(q) @ inv(T(pos1, rot1)); M = rotation about / translation along the joint's
local axis. The world pose of the root body (base) is the articulation root pose (pkl initial_state root_pose).
"""
from __future__ import annotations

import json

import numpy as np

USD_KIN_DIR = "/workspace/xhand/runs/stage1/usd_kin"
_AX = {"X": 0, "Y": 1, "Z": 2}


def quat_to_mat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_quat(R):
    """(...,3,3) -> (...,4) wxyz."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = np.zeros(R.shape[:-2] + (4,))
    c0 = tr > 0
    s = np.sqrt(np.maximum(tr + 1.0, 1e-12)) * 2
    q0 = np.stack([0.25 * s, (R[..., 2, 1] - R[..., 1, 2]) / s, (R[..., 0, 2] - R[..., 2, 0]) / s,
                   (R[..., 1, 0] - R[..., 0, 1]) / s], -1)
    q = np.where(c0[..., None], q0, q)
    for i, (a, b, c) in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
        cond = (~c0) & (R[..., a, a] >= R[..., b, b]) & (R[..., a, a] >= R[..., c, c])
        s = np.sqrt(np.maximum(1.0 + R[..., a, a] - R[..., b, b] - R[..., c, c], 1e-12)) * 2
        qq = np.zeros(R.shape[:-2] + (4,))
        qq[..., 0] = (R[..., c, b] - R[..., b, c]) / s
        qq[..., 1 + a] = 0.25 * s
        qq[..., 1 + b] = (R[..., b, a] + R[..., a, b]) / s
        qq[..., 1 + c] = (R[..., c, a] + R[..., a, c]) / s
        q = np.where(cond[..., None], qq, q)
        c0 = c0 | cond
    return q * np.where(q[..., :1] < 0, -1.0, 1.0)


def make_T(pos, quat):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(np.asarray(quat))
    T[:3, 3] = pos
    return T


def axis_rot(axis_idx, ang):
    """(B,) angles -> (B,3,3) rotation about a principal axis."""
    c, s = np.cos(ang), np.sin(ang)
    B = ang.shape[0]
    R = np.zeros((B, 3, 3))
    i, j, k = axis_idx, (axis_idx + 1) % 3, (axis_idx + 2) % 3
    R[:, i, i] = 1
    R[:, j, j] = c
    R[:, k, k] = c
    R[:, j, k] = -s
    R[:, k, j] = s
    return R


def rot6d_from_mat(R):
    """first two COLUMNS of R (x and y axes), (...,6)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def mat_from_rot6d(d6):
    a, b = d6[..., :3], d6[..., 3:6]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (x * b).sum(-1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], -1)


def so3_log(R):
    """(B,3,3) -> rotation vectors (B,3)."""
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    th = np.arccos(cos)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)
    small = th < 1e-6
    k = np.where(small[..., None], 0.5 * v, th[..., None] * v / (2 * np.where(small, 1.0, s)[..., None]))
    # near pi: fall back to the diagonal
    near = th > np.pi - 1e-3
    if near.any():
        Rn = R[near]
        ax = np.sqrt(np.maximum((np.diagonal(Rn, axis1=-2, axis2=-1) + 1) / 2, 0))
        sg = np.sign(np.stack([Rn[:, 2, 1] - Rn[:, 1, 2], Rn[:, 0, 2] - Rn[:, 2, 0], Rn[:, 1, 0] - Rn[:, 0, 1]], -1))
        sg[sg == 0] = 1
        k[near] = ax * sg * th[near][:, None]
    return k


class HandKin:
    """Kinematic tree of one hand. ``joint_order`` = the joint-name order of q vectors passed to fk()."""

    def __init__(self, hand: str, joint_order: list[str] | None = None, usd_json: str | None = None):
        d = json.load(open(usd_json or f"{USD_KIN_DIR}/{hand}.json"))
        self.hand = hand
        js = [j for j in d["joints"] if j["body0"] is not None and j["body1"] is not None and not j["excluded"]]
        short = lambda p: p.split("/")[-1]  # noqa: E731
        self.children = {}
        self.joint = {}
        for j in js:
            b0, b1 = short(j["body0"]), short(j["body1"])
            typ = {"PhysicsRevoluteJoint": "rev", "PhysicsPrismaticJoint": "pri", "PhysicsFixedJoint": "fix"}[j["type"]]
            self.joint[b1] = {
                "name": j["name"], "parent": b0, "type": typ,
                "T0": make_T(j["pos0"], j["rot0"]), "T1inv": np.linalg.inv(make_T(j["pos1"], j["rot1"])),
                "axis": _AX.get(j.get("axis") or "X", 0), "lower": j.get("lower"), "upper": j.get("upper"),
            }
            self.children.setdefault(b0, []).append(b1)
        roots = [short(j["body1"]) for j in d["joints"] if j["body0"] is None]
        self.root = roots[0]
        self.order = []
        stack = [self.root]
        while stack:
            b = stack.pop(0)
            self.order.append(b)
            stack += self.children.get(b, [])
        self.movable = [self.joint[b]["name"] for b in self.order[1:] if self.joint[b]["type"] != "fix"]
        self.joint_order = list(joint_order) if joint_order is not None else self.movable
        self.qidx = {n: i for i, n in enumerate(self.joint_order)}
        missing = [n for n in self.movable if n not in self.qidx]
        if missing:
            raise ValueError(f"joints not in joint_order: {missing}")
        self.body_joint = {b: self.joint[b]["name"] for b in self.order[1:]}
        lim = {self.joint[b]["name"]: (self.joint[b]["lower"], self.joint[b]["upper"]) for b in self.order[1:]
               if self.joint[b]["type"] != "fix"}
        self.lower = np.array([lim[n][0] for n in self.joint_order], dtype=np.float64)
        self.upper = np.array([lim[n][1] for n in self.joint_order], dtype=np.float64)

    def set_limits(self, names, lower, upper):
        for n, lo, hi in zip(names, lower, upper):
            if n in self.qidx:
                self.lower[self.qidx[n]], self.upper[self.qidx[n]] = lo, hi

    def fk(self, q, root_T=None, bodies=None, with_joint_frames=False):
        """q (B,nq) in joint_order -> {body: (B,4,4)} world poses (root_T: (4,4) or (B,4,4), default identity).
        with_joint_frames: also return {joint_name: (origin (B,3), axis_w (B,3), type)} for Jacobians."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        B = q.shape[0]
        Tw = {}
        rT = np.eye(4) if root_T is None else np.asarray(root_T, dtype=np.float64)
        Tw[self.root] = np.broadcast_to(rT, (B, 4, 4)).copy()
        jf = {}
        need = None
        if bodies is not None:
            need = set()
            for b in bodies:
                while b is not None and b not in need:
                    need.add(b)
                    b = self.joint[b]["parent"] if b in self.joint else None
        for b in self.order[1:]:
            if need is not None and b not in need:
                continue
            j = self.joint[b]
            Tp = Tw[j["parent"]] @ j["T0"]                      # joint frame on the parent side (B,4,4)
            if j["type"] == "fix":
                M = None
            else:
                qq = q[:, self.qidx[j["name"]]]
                M = np.tile(np.eye(4), (B, 1, 1))
                if j["type"] == "rev":
                    M[:, :3, :3] = axis_rot(j["axis"], qq)
                else:
                    M[:, j["axis"], 3] = qq
                if with_joint_frames:
                    jf[j["name"]] = (Tp[:, :3, 3].copy(), Tp[:, :3, j["axis"]].copy(), j["type"])
            Tc = Tp if M is None else Tp @ M
            Tw[b] = Tc @ j["T1inv"]
        if bodies is not None:
            Tw = {b: Tw[b] for b in bodies}
        return (Tw, jf) if with_joint_frames else Tw

    def chain_joints(self, body):
        """movable joint names from the root to ``body``."""
        out = []
        b = body
        while b in self.joint:
            if self.joint[b]["type"] != "fix":
                out.append(self.joint[b]["name"])
            b = self.joint[b]["parent"]
        return out[::-1]

    def rel_T(self, parent_body, child_body):
        """constant transform parent->child when only fixed joints lie between them (zero config)."""
        Tw = self.fk(np.zeros((1, len(self.joint_order))))
        return np.linalg.inv(Tw[parent_body][0]) @ Tw[child_body][0]
