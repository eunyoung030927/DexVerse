# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A DexVerse task built N times for retargeting: state access, kinematic teleport, physics stepping.

The task is built exactly as ``record_demos.py`` builds it (``dexverse.mimic.env_setup.build_env_cfg``: the
task config for the requested ``robot_type``, cameras stripped), so the robot, its actuators and the scene
physics are the benchmark's own. Only the stepping is done here: the retargeting writes robot joint targets
directly and steps the simulator ``decimation`` times per control frame, the same physics a DexVerse
``env.step`` runs, without the action / observation / event managers in between.

Poses are env-local (env origin subtracted), quaternions wxyz, like ``scene.get_state(is_relative=True)``.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply

from . import hands

# Rigid objects that are part of the set, not of the task: never contact targets, never manipulated.
STATIC_PREFIXES = ("table", "success_marker", "ground", "goal")   # goal_frame / goal_pad: markers


def build(task: str, robot_type: str, num_envs: int, device: str, *, gravity: bool = True):
    """Create the task env (Isaac app must be running). Returns ``(env, success_term)``."""
    import gymnasium as gym  # noqa: PLC0415

    import dexverse.tasks  # noqa: F401,PLC0415
    from dexverse.mimic.env_setup import build_env_cfg, resolve_success_term  # noqa: PLC0415

    cfg, success = build_env_cfg(task, robot_type, device=device, num_envs=num_envs, keep_cameras=False)
    if not gravity:
        cfg.sim.gravity = (0.0, 0.0, 0.0)
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    return RetargetEnv(env, robot_type), resolve_success_term(env, success)


class RetargetEnv:
    def __init__(self, env, robot_type: str):
        self.env = env
        self.scene = env.scene
        self.sim = env.sim
        self.device = env.device
        self.num_envs = env.num_envs
        self.physics_dt = env.physics_dt
        self.decimation = env.cfg.decimation
        self.step_dt = self.physics_dt * self.decimation
        self.spec = hands.get(robot_type)
        self.robot: Articulation = self.scene["robot"]
        self.joint_names = list(self.robot.joint_names)
        bodies = list(self.robot.body_names)
        self.n_hands = self.spec.n_hands
        self.palm_ids = torch.tensor([bodies.index(p) for p in self.spec.palms], device=self.device)
        self.slots = self.spec.tip_slots()                 # 5 * hand + finger, over the robot's hands
        self.tip_ids = torch.tensor([bodies.index(self.spec.tips[s]) for s in self.slots], device=self.device)
        self.wrist_ids = torch.tensor([self.joint_names.index(j) for j in self.spec.wrist_joints], device=self.device)
        self._action_map = self._build_action_map()
        # finger joints = every joint the env action commands that is not a wrist joint (a joint the action does
        # not command -- Sharpa bimanual's head, Inspire's mimic joints -- is never given a residual)
        commanded = {j for j, _, _, _ in self._action_map}
        self.finger_ids = torch.tensor(
            [i for i, n in enumerate(self.joint_names) if i in commanded and n not in self.spec.wrist_joints],
            device=self.device)
        lim = self.robot.data.soft_joint_pos_limits[0]
        self.q_lo, self.q_hi = lim[:, 0].clone(), lim[:, 1].clone()
        # mimic coupling q = C x + d over the free (non-mimic) joints x; identity for fully actuated hands
        mimic = {self.joint_names.index(j): (self.joint_names.index(drv), m, b) for j, drv, m, b in self.spec.mimic}
        self.mimic_ids = torch.tensor(sorted(mimic), dtype=torch.long, device=self.device)
        self.free_ids = torch.tensor([i for i in range(len(self.joint_names)) if i not in mimic],
                                     dtype=torch.long, device=self.device)
        self.C = torch.zeros(len(self.joint_names), len(self.free_ids), device=self.device)
        self.d = torch.zeros(len(self.joint_names), device=self.device)
        col = {int(i): k for k, i in enumerate(self.free_ids.tolist())}
        self.C[self.free_ids, torch.arange(len(self.free_ids), device=self.device)] = 1.0
        for j, (drv, m, b) in mimic.items():
            self.C[j, col[drv]], self.d[j] = m, b
        self.q_target = self.robot.data.joint_pos.clone()
        # task objects: dynamic rigid objects and the links of non-robot articulations
        self.objects: dict[str, RigidObject | Articulation] = {}
        for name, a in self.scene.rigid_objects.items():
            if not name.startswith(STATIC_PREFIXES):
                self.objects[name] = a
        for name, a in self.scene.articulations.items():
            if name != "robot":
                self.objects[name] = a
        self.obj_ids = list(self.objects)
        self.art_ids = [k for k, a in self.objects.items() if isinstance(a, Articulation)]
        self.bodies = []                                   # (obj_id, body name, index in the asset)
        for k, a in self.objects.items():
            if isinstance(a, Articulation):
                self.bodies += [(k, b, i) for i, b in enumerate(a.body_names)]
            else:
                self.bodies.append((k, k, 0))

    # ----------------------------------------------------------------- reads
    def origins(self):
        return self.scene.env_origins

    def tips(self) -> torch.Tensor:
        """(N, n_tips, 3) fingertip body origins, in ``spec.tip_slots()`` order."""
        return self.robot.data.body_pos_w[:, self.tip_ids] - self.origins()[:, None]

    def palm_pose(self) -> torch.Tensor:
        """(N, n_hands, 7) palm pose per hand (right, then left)."""
        return torch.cat([self.robot.data.body_pos_w[:, self.palm_ids] - self.origins()[:, None],
                          self.robot.data.body_quat_w[:, self.palm_ids]], -1)

    def object_body_poses(self) -> torch.Tensor:
        o = self.origins()
        out = [torch.cat([self.objects[k].data.body_pos_w[:, i] - o, self.objects[k].data.body_quat_w[:, i]], -1)
               for k, _, i in self.bodies]
        return torch.stack(out, 1) if out else torch.zeros(self.num_envs, 0, 7, device=self.device)

    def object_roots(self) -> torch.Tensor:
        """(N, n_obj, 7) root poses in ``obj_ids`` order."""
        o = self.origins()[:, None]
        pos = torch.stack([a.data.root_pos_w for a in self.objects.values()], 1) - o
        quat = torch.stack([a.data.root_quat_w for a in self.objects.values()], 1)
        return torch.cat([pos, quat], -1)

    def object_qpos(self) -> dict[str, torch.Tensor]:
        return {k: self.objects[k].data.joint_pos for k in self.art_ids}

    def body_prim_paths(self) -> list[str]:
        out = []
        for k, _, i in self.bodies:
            a = self.objects[k]
            out.append(a.root_physx_view.link_paths[0][i] if k in self.art_ids else a.root_physx_view.prim_paths[0])
        return out

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Free-joint values (..., n_free) -> full joint vector (..., n_joints), mimic joints following."""
        return x @ self.C.T + self.d

    def calibrate_mimic(self, steps: int = 120) -> dict:
        """Fit each mimic joint's ``multiplier * driver + offset`` on the simulator itself (objects parked, every
        driver swept over its range across the envs, physics settled ``steps`` frames) and use that in ``expand``.
        The USD's PhysX mimic constraint is what stage 3 runs under, and it need not equal the URDF numbers (units,
        conversion); returns the fit per mimic joint. No-op for a fully actuated hand."""
        if len(self.mimic_ids) == 0:
            return {}
        n = self.num_envs
        home = self.robot.data.default_joint_pos[0].clone()
        x = home[self.free_ids].expand(n, -1).clone()
        drivers = sorted({self.joint_names.index(d) for _, d, _, _ in self.spec.mimic})
        col = {int(i): k for k, i in enumerate(self.free_ids.tolist())}
        # one driver at a time (the others at home): closing every finger at once makes them collide
        per = n // len(drivers)
        assert per >= 4, (n, len(drivers))
        block = {}
        for k, dj in enumerate(drivers):
            ids = torch.arange(k * per, (k + 1) * per, device=self.device)
            frac = torch.linspace(0.0, 1.0, per, device=self.device)
            x[ids, col[dj]] = self.q_lo[dj] + frac * (self.q_hi[dj] - self.q_lo[dj])
            block[dj] = ids
        self.park_static()
        self.teleport(q=self.expand(x))
        for _ in range(steps):
            self.physics_step()
        q = self.robot.data.joint_pos.clone()                           # (n, J) settled under the constraint
        cmd = self.expand(x)
        fit = {}
        for j, d, m0, b0 in self.spec.mimic:
            ji, di = self.joint_names.index(j), self.joint_names.index(d)
            ids = block[di]
            # a sample whose driver did not stay where it was put was pushed by a self-collision (Inspire thumb at
            # home yaw below pitch 0.22): not the constraint, so not fitted
            ok = (q[ids, di] - cmd[ids, di]).abs() < 2e-3
            qd, qj = q[ids][ok, di], q[ids][ok, ji]
            if ok.sum() >= 4:
                A = torch.stack([qd, torch.ones_like(qd)], 1)
                (m, b), *_ = torch.linalg.lstsq(A, qj[:, None])
                m, b = float(m), float(b)
            else:
                m, b = m0, b0
            resid = float((qj - (m * qd + b)).abs().max()) if len(qj) else float("nan")
            self.C[ji, col[di]], self.d[ji] = m, b
            fit[j] = {"driver": d, "multiplier": round(m, 5), "offset": round(b, 5), "max_resid_rad": round(resid, 5),
                      "samples": int(ok.sum()), "urdf": [m0, b0]}
        self.teleport(q=home)
        return fit

    # ----------------------------------------------------------------- state
    def get_state(self):
        return self.scene.get_state(is_relative=True)

    def set_state(self, state, env_ids=None):
        """Write a ``scene.get_state(is_relative=True)``-style state (batch 1 is broadcast) and refresh buffers."""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self.scene.reset_to(_broadcast(state, len(ids), self.device), env_ids=ids, is_relative=True)
        q = self.robot.data.joint_pos[ids].clone()
        self.robot.set_joint_position_target(q, env_ids=ids)
        self.q_target[ids] = q
        self.sim.forward()
        self.scene.update(dt=0.0)

    def teleport(self, q=None, roots=None, art=None, env_ids=None):
        """Kinematic write (zero velocity) of robot joints / object roots / articulation joints, no physics step."""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        n = len(ids)
        if q is not None:
            q = q.expand(n, -1).clone() if q.dim() == 1 else q
            self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
            self.robot.set_joint_position_target(q, env_ids=ids)
            self.q_target[ids] = q
        if roots is not None:
            o = self.origins()[ids]
            for j, a in enumerate(self.objects.values()):
                p = roots[..., j, :].expand(n, -1).clone() if roots.dim() == 2 else roots[:, j].clone()
                p[:, :3] += o
                a.write_root_pose_to_sim(p, env_ids=ids)
                a.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=ids)
        for k, v in (art or {}).items():
            v = v.expand(n, -1).clone() if v.dim() == 1 else v
            self.objects[k].write_joint_state_to_sim(v, torch.zeros_like(v), env_ids=ids)
            self.objects[k].set_joint_position_target(v, env_ids=ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

    def park_static(self, offset=(0.0, 0.0, -50.0)):
        """Move every non-robot asset far away (robot-only kinematics)."""
        for name, a in list(self.scene.rigid_objects.items()) + [
            (n, x) for n, x in self.scene.articulations.items() if n != "robot"
        ]:
            p = a.data.root_pose_w.clone()
            p[:, :3] += torch.tensor(offset, device=self.device)
            a.write_root_pose_to_sim(p)
            a.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, device=self.device))
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

    # ----------------------------------------------------------------- step
    def physics_step(self, hook=None):
        """One control frame: hold ``q_target`` for ``decimation`` physics steps (``hook()`` after each)."""
        for _ in range(self.decimation):
            self.robot.set_joint_position_target(self.q_target)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            if hook is not None:
                hook()

    # --------------------------------------------------------------- actions
    def _build_action_map(self):
        """Per action column: joint index, scale, offset (JointPositionAction: target = raw * scale + offset)."""
        cols = []
        for name, term in self.env.action_manager._terms.items():
            ids = term._joint_ids
            ids = list(range(len(self.joint_names))) if isinstance(ids, slice) else list(ids)
            scale = term._scale
            off = term._offset
            for c, j in enumerate(ids):
                s = float(scale[0, c]) if torch.is_tensor(scale) else float(scale)
                o = float(off[0, c]) if torch.is_tensor(off) else float(off)
                cols.append((j, s, o, type(term).__name__))
        kinds = {c[3] for c in cols}
        if kinds - {"JointPositionAction"}:
            raise RuntimeError(f"retargeting writes absolute joint targets; action terms are {sorted(kinds)}")
        return cols

    def targets_to_action(self, q_target: torch.Tensor) -> np.ndarray:
        """Joint targets (n_joints,) -> the env action that commands them."""
        q = q_target.detach().cpu().numpy()
        return np.array([(q[j] - o) / s for j, s, o, _ in self._action_map], dtype=np.float32)


def _broadcast(state, n, device):
    if isinstance(state, dict):
        return {k: _broadcast(v, n, device) for k, v in state.items()}
    t = torch.as_tensor(np.asarray(state) if not torch.is_tensor(state) else state, device=device, dtype=torch.float32)
    if t.dim() == 1:
        t = t[None]
    return t.expand(n, *t.shape[1:]).clone() if t.shape[0] == 1 else t


def to_numpy_state(state, i=0):
    """Env ``i`` of a scene state as numpy with a leading batch of 1 (the pickle layout of record_demos)."""
    if isinstance(state, dict):
        return {k: to_numpy_state(v, i) for k, v in state.items()}
    return state[i:i + 1].detach().cpu().numpy()


def quat_apply_np(q, v):
    return quat_apply(torch.as_tensor(q), torch.as_tensor(v)).numpy()
