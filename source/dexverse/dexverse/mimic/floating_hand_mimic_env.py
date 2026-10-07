# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Mimic (MimicGen / DexMimicGen) environment for DexVerse floating hands.

End effector = each hand's palm body. Floating hands are driven by six virtual wrist joints (three prismatic,
then three revolute) through absolute joint-position action terms, so a target palm pose maps to wrist joint
targets in closed form once the wrist kinematics are known. At construction the env measures them in the robot
root frame by moving the wrist joints kinematically (no physics step) and reading the palm pose:

* prismatic axes ``a_i`` and revolute axes ``w_k`` (product of exponentials, chain order),
* the common rotation centre ``c`` of the revolute joints and the palm offset ``d`` from it,
* the palm orientation ``R0`` at the zero wrist configuration,

so that ``palm_rot(q) = Rw(q_r) R0`` and ``palm_pos(q) = c + A q_t + Rw(q_r) d`` with
``Rw = exp(q_1 w_1) exp(q_2 w_2) exp(q_3 w_3)``. The model is checked against the simulator at random wrist
configurations before it is used. Finger joints are the "gripper" actions and are copied from the source demos
(DexMimicGen style). Poses are expressed relative to the env origin, like :meth:`get_object_poses`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLMimicEnv
from scipy.spatial.transform import Rotation

_AXIS_LETTERS = "XYZ"


def _rot_from_matrix(m: np.ndarray) -> Rotation:
    return Rotation.from_matrix(np.asarray(m, dtype=np.float64))


class _WristModel:
    """Closed-form palm pose <-> wrist joint values of one floating hand (all quantities in the robot root frame)."""

    def __init__(self, trans_axes, rot_axes, centre, palm_offset, palm_rot0):
        self.A = np.asarray(trans_axes, dtype=np.float64).T          # (3, 3): columns = prismatic axes
        self.w = np.asarray(rot_axes, dtype=np.float64)              # (3, 3): rows = revolute axes, chain order
        self.c = np.asarray(centre, dtype=np.float64)
        self.d = np.asarray(palm_offset, dtype=np.float64)
        self.R0 = palm_rot0
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


class FloatingHandMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic API for DexVerse floating-hand tasks (eef names = retargeter layout sides, e.g. "right", "left")."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._robot = self.scene["robot"]
        self._hands = self._resolve_hands()
        self._measure_wrist_models()

    # ----- layout -------------------------------------------------------------------------------------
    def _resolve_hands(self) -> dict[str, dict]:
        from importlib import import_module  # noqa: PLC0415

        from dexverse.devices.retargeters.simple_relative_retargeting import (  # noqa: PLC0415
            SIMPLE_RETARGETER_LAYOUT_SOURCES,
        )

        robot_type = str(getattr(self.cfg, "robot_type", "") or "")
        if robot_type not in SIMPLE_RETARGETER_LAYOUT_SOURCES:
            raise RuntimeError(f"[mimic] no retargeter layout for robot_type {robot_type!r}")
        module_name, attr = SIMPLE_RETARGETER_LAYOUT_SOURCES[robot_type]
        layout = getattr(import_module(module_name), attr)

        # action column -> (joint id, scale, offset) of the joint-position term consuming it
        entries = []
        for term_name in self.action_manager.active_terms:
            term = self.action_manager.get_term(term_name)
            ids = term._joint_ids
            ids = list(range(self._robot.num_joints))[ids] if isinstance(ids, slice) else list(ids)
            for k in range(int(term.action_dim)):
                scale = term._scale if isinstance(term._scale, float) else float(term._scale[0, k])
                offset = term._offset if isinstance(term._offset, float) else float(term._offset[0, k])
                entries.append((int(ids[k]), float(scale), float(offset)))
        if len(entries) != int(self.action_manager.total_action_dim):
            raise RuntimeError("[mimic] action columns do not match the action dim")

        rc = getattr(self.cfg, "robot_config", None)
        hands, covered = {}, set()
        for side, h in layout["hands"].items():
            if h.get("wrist_rot_repr", "euler") != "euler":
                raise RuntimeError(f"[mimic] {robot_type} {side}: wrist_rot_repr {h.get('wrist_rot_repr')} unsupported")
            cols = lambda idx: [{"col": int(i), "joint_id": entries[i][0], "scale": entries[i][1],  # noqa: E731
                                 "offset": entries[i][2]} for i in idx]
            rot = sorted(cols(h["wrist_rot_indices"]), key=lambda j: j["joint_id"])  # sim joint order = chain order
            trans = sorted(cols(h["wrist_trans_indices"]), key=lambda j: j["joint_id"])
            if max(j["joint_id"] for j in trans) > min(j["joint_id"] for j in rot):
                raise RuntimeError(f"[mimic] {side}: translation joints are not all before the rotation joints")
            name = None
            if rc is not None:
                name = (getattr(rc, f"{side}_palm_body_name", None) if len(layout["hands"]) > 1 else None) \
                    or getattr(rc, "palm_body_name", None)
            found = self._robot.find_bodies(name)[0] if name else []
            if not found:
                raise RuntimeError(f"[mimic] {side}: palm body {name!r} not found")
            fingers = [int(i) for i in h["finger_indices"]]
            hands[side] = {"trans": trans, "rot": rot, "fingers": fingers, "palm_id": int(found[0])}
            covered |= {j["col"] for j in trans + rot} | set(fingers)
        if covered != set(range(len(entries))):
            raise RuntimeError(f"[mimic] layout leaves action columns {sorted(set(range(len(entries))) - covered)} "
                               "unassigned")
        return hands

    # ----- wrist kinematics ---------------------------------------------------------------------------
    def _root_pose_rel(self, env_id: int) -> tuple[np.ndarray, Rotation]:
        p = (self._robot.data.root_pos_w[env_id] - self.scene.env_origins[env_id]).double().cpu().numpy()
        q = self._robot.data.root_quat_w[env_id].double().cpu().numpy()  # wxyz
        return p, Rotation.from_quat([q[1], q[2], q[3], q[0]])

    def _measure_wrist_models(self, theta: float = 0.6, eps_t: float = 0.05):
        robot, sim = self._robot, self.sim
        view = robot.root_physx_view
        q0 = robot.data.joint_pos.clone()
        v0 = robot.data.joint_vel.clone()
        rq = robot.data.root_quat_w[0].double().cpu().numpy()
        base = Rotation.from_quat([rq[1], rq[2], rq[3], rq[0]])
        root_p = robot.data.root_pos_w[0].double().cpu().numpy()

        def set_q(q):
            robot.write_joint_state_to_sim(q, torch.zeros_like(v0))
            sim.forward()

        def palm(bid):
            tf = view.get_link_transforms()[0, bid].double().cpu().numpy()
            return tf[:3], Rotation.from_quat(tf[3:7])  # PhysX: position + quaternion (x, y, z, w)

        self._models = {}
        try:
            for side, h in self._hands.items():
                wrist = h["trans"] + h["rot"]
                qz = q0.clone()
                for j in wrist:
                    qz[:, j["joint_id"]] = 0.0
                set_q(qz)
                p0, r0 = palm(h["palm_id"])
                rot_axes, rows, rhs = [], [], []
                for j in h["rot"]:
                    for sgn in (1.0, -1.0):
                        q1 = qz.clone()
                        q1[:, j["joint_id"]] = sgn * theta
                        set_q(q1)
                        p1, r1 = palm(h["palm_id"])
                        rk = (r1 * r0.inv()).as_matrix()
                        if sgn > 0:
                            rot_axes.append((r1 * r0.inv()).as_rotvec() / theta)
                        # p1 - p0 = (Rk - I)(p0 - c)  ->  (Rk - I) c = (Rk - I) p0 - (p1 - p0)
                        rows.append(rk - np.eye(3))
                        rhs.append((rk - np.eye(3)) @ p0 - (p1 - p0))
                centre_w = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
                trans_axes = []
                for j in h["trans"]:
                    q1 = qz.clone()
                    q1[:, j["joint_id"]] = eps_t
                    set_q(q1)
                    p1, _ = palm(h["palm_id"])
                    trans_axes.append((p1 - p0) / eps_t)
                model = _WristModel(
                    trans_axes=[base.inv().apply(a) for a in trans_axes],
                    rot_axes=[base.inv().apply(w) for w in rot_axes],
                    centre=base.inv().apply(centre_w - root_p),
                    palm_offset=base.inv().apply(p0 - centre_w),
                    palm_rot0=base.inv() * r0,
                )
                # check the model against the simulator at random wrist configurations
                rng = np.random.default_rng(0)
                worst_p, worst_r = 0.0, 0.0
                for _ in range(4):
                    qt = rng.uniform(-0.05, 0.05, 3)
                    qr = rng.uniform(-0.8, 0.8, 3)
                    q1 = qz.clone()
                    for k, j in enumerate(h["trans"]):
                        q1[:, j["joint_id"]] = float(qt[k])
                    for k, j in enumerate(h["rot"]):
                        q1[:, j["joint_id"]] = float(qr[k])
                    set_q(q1)
                    ps, rs = palm(h["palm_id"])
                    pm, rm = model.forward(qt, qr)
                    worst_p = max(worst_p, float(np.linalg.norm(base.apply(pm) + root_p - ps)))
                    worst_r = max(worst_r, float(np.degrees(((base * rm) * rs.inv()).magnitude())))
                    it, ir = model.inverse(pm, rm, qr)
                    if np.abs(it - qt).max() > 1e-6 or np.abs(ir - qr).max() > 1e-6:
                        raise RuntimeError(f"[mimic] {side}: wrist inverse does not invert the model ({it}, {ir})")
                if worst_p > 1e-3 or worst_r > 0.1:
                    raise RuntimeError(f"[mimic] {side}: wrist model off the simulator by {worst_p * 1000:.2f} mm / "
                                       f"{worst_r:.3f} deg; the virtual wrist is not translate-then-rotate about a "
                                       "common centre")
                print(f"[mimic] {side} wrist model: Euler {model.seq} signs {model.sign.tolist()}, check "
                      f"{worst_p * 1000:.3f} mm / {worst_r:.4f} deg")
                self._models[side] = model
        finally:
            robot.write_joint_state_to_sim(q0, v0)
            sim.forward()

    def _joint_values(self, h: dict, values: torch.Tensor, kind: str) -> np.ndarray:
        return np.array([float(values[j["col"]]) * j["scale"] + j["offset"] for j in h[kind]])

    # ----- Mimic API ----------------------------------------------------------------------------------
    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        pose = self._robot.data.body_link_pose_w[env_ids, self._hands[eef_name]["palm_id"]]
        pos = pose[:, :3] - self.scene.env_origins[env_ids]
        rot = _quat_wxyz_to_matrix(pose[:, 3:7])
        return _make_pose(pos, rot)

    def target_eef_pose_to_action(self, target_eef_pose_dict: dict, gripper_action_dict: dict,
                                  action_noise_dict: dict | None = None, env_id: int = 0) -> torch.Tensor:
        action = torch.zeros(int(self.action_manager.total_action_dim), device=self.device)
        root_p, root_r = self._root_pose_rel(env_id)
        joint_pos = self._robot.data.joint_pos[env_id]
        for side, h in self._hands.items():
            T = target_eef_pose_dict[side].double().cpu().numpy()
            pos_b = root_r.inv().apply(T[:3, 3] - root_p)
            rot_b = root_r.inv() * _rot_from_matrix(T[:3, :3])
            q_r_ref = np.array([float(joint_pos[j["joint_id"]]) for j in h["rot"]])
            q_t, q_r = self._models[side].inverse(pos_b, rot_b, q_r_ref)
            if action_noise_dict is not None and action_noise_dict.get(side):
                noise = float(action_noise_dict[side])
                q_t = q_t + noise * np.random.randn(3)
                q_r = q_r + noise * np.random.randn(3)
            for k, j in enumerate(h["trans"]):
                action[j["col"]] = (q_t[k] - j["offset"]) / j["scale"]
            for k, j in enumerate(h["rot"]):
                action[j["col"]] = (q_r[k] - j["offset"]) / j["scale"]
            action[h["fingers"]] = gripper_action_dict[side].to(self.device, dtype=action.dtype)
        return action

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        poses = {side: [] for side in self._hands}
        for env_id in range(action.shape[0]):
            root_p, root_r = self._root_pose_rel(env_id)
            for side, h in self._hands.items():
                pos_b, rot_b = self._models[side].forward(self._joint_values(h, action[env_id], "trans"),
                                                          self._joint_values(h, action[env_id], "rot"))
                T = np.eye(4)
                T[:3, :3] = (root_r * rot_b).as_matrix()
                T[:3, 3] = root_r.apply(pos_b) + root_p
                poses[side].append(T)
        return {side: torch.tensor(np.stack(v), dtype=torch.float32, device=self.device) for side, v in poses.items()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {side: actions[..., h["fingers"]] for side, h in self._hands.items()}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """``object_lifted``: the task object is >= ``mimic_lift_height`` above its spawn height (grasp done)."""
        if env_ids is None:
            env_ids = slice(None)
        obj = self.scene["object"]
        z = obj.data.root_pos_w[env_ids, 2] - self.scene.env_origins[env_ids, 2]
        z0 = obj.data.default_root_state[env_ids, 2]
        height = float(getattr(self.cfg, "mimic_lift_height", 0.03))
        return {"object_lifted": (z - z0) > height}

    def serialize(self):
        return dict(env_name=self.cfg.env_name, type=2, env_kwargs=dict())


def _quat_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    import isaaclab.utils.math as math_utils  # noqa: PLC0415

    return math_utils.matrix_from_quat(q)


def _make_pose(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    import isaaclab.utils.math as math_utils  # noqa: PLC0415

    return math_utils.make_pose(pos, rot)
