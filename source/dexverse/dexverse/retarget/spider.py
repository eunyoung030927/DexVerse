# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The SPIDER rollout environment on a DexVerse task: target hand, task objects, the retargeted reference.

Ported from bench2dex_retarget retarget/spider_env.py (SPIDER v2). Control per env:

  wrist    a residual on the six virtual wrist joints around the kinematic reference
           (``wrist_trans_scale`` m, ``wrist_rot_scale`` rad per unit action) -- a floating hand needs no IK.
  fingers  a residual on every finger joint, +-``finger_res_cap`` rad around the reference.
  gate     residuals are faded in only around the frames the hand manipulates something
           (``gate_margin``, ``gate_ramp``); elsewhere the hand replays the reference.
  ema      the residual is low-passed: res = ema * previous + (1 - ema) * new (the reference is not).
  lead     the reference command leads ``ref_q`` by each joint's PD lag D/K (see ``update_command``).

Object-side guidance is SPIDER's virtual contact spring: each demo contact point on a body is pulled towards the
fingertip that touched it, capped at a few body weights; the gain is annealed to 0 over the optimisation
iterations and committed frames always run with gain 0.

DexVerse controls at 60 Hz (Bench2Dex: 20 Hz); every frame count below is the Bench2Dex value x 3 (same time).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_conjugate, quat_mul

from .env import RetargetEnv, to_numpy_state

ROT_NONE = ("sphere", "ball", "apple", "orange", "lemon")
ROT_TILT = ("cup", "mug", "bowl", "plate", "bottle", "can", "glass", "pot")


@dataclass
class SpiderCfg:
    gate_margin: int = 45
    gate_ramp: int = 15
    action_ema: float = 0.8 ** (1.0 / 3.0)   # the 20 Hz value 0.8, per 60 Hz frame
    wrist_trans_scale: float = 0.05
    wrist_rot_scale: float = 0.5
    finger_res_cap: float = 0.6


class SpiderEnv:
    def __init__(self, renv: RetargetEnv, ref: dict, kin: dict, cfg: SpiderCfg, success_term, success_steps: int):
        self.e = renv
        self.cfg = cfg
        dev = self.device = renv.device
        self.num_envs = renv.num_envs
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.meta = ref["meta"]
        assert self.meta["obj_ids"] == renv.obj_ids, (self.meta["obj_ids"], renv.obj_ids)
        self.T = self.meta["n_frames"]
        perm = [list(kin["joint_names"]).index(n) for n in renv.joint_names]
        self.ref_q = t(kin["q"][:, perm])
        self.slots = renv.slots                                           # target finger slots
        self.ref_tips = t(kin["tips"])[:, self.slots]                     # (T,F,3)
        self.ref_body = t(ref["body_pose"])
        self.ref_root = t(ref["obj_root"])
        self.ref_art = {k: t(ref[f"art/{k}"]) for k in renv.art_ids}
        self.contact_body = torch.as_tensor(ref["contact_body"], dtype=torch.long, device=dev)[:, self.slots]
        self.contact_local = t(ref["contact_local"])[:, self.slots]
        self.active = np.asarray(ref["active"])                           # (T,)
        self.body_obj = torch.as_tensor(self.meta["body_obj"], device=dev)
        self.gate = t(self._make_gate(self.active, cfg.gate_margin, cfg.gate_ramp))   # (T,)
        # PD lag compensation: a joint under implicit PD (stiffness K, damping D, velocity target 0) settles
        # D/K seconds behind a moving position target (0.2 s on the floating wrists), so the command leads the
        # reference by that much: cmd = q_ref + (D/K) * dq_ref/dt (position targets only, like the env action)
        r = renv.robot
        self.tau = (r.data.joint_damping[0] / r.data.joint_stiffness[0].clamp(min=1e-6)).to(dev)
        self.update_command()
        self.wrist_ids = renv.wrist_ids
        self.finger_ids = renv.finger_ids
        self.n_act = 6 + len(self.finger_ids)
        self.actions = torch.zeros(self.num_envs, self.n_act, device=dev)
        self.res = torch.zeros(self.num_envs, len(renv.joint_names), device=dev)
        self.frame = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        n = self.num_envs
        m, inert = [], []
        for k, _, i in renv.bodies:
            v = renv.objects[k].root_physx_view
            mm, ii = v.get_masses().reshape(n, -1), v.get_inertias().reshape(n, -1, 9)
            m.append(mm[:, i])
            inert.append(ii[:, i][:, [0, 4, 8]].mean(-1))
        self.body_mass = torch.stack(m, 1)[..., None].to(dev)
        self.body_inertia = torch.stack(inert, 1)[..., None].to(dev)
        self.asset_bodies = {k: [j for j, (kk, _, _) in enumerate(renv.bodies) if kk == k] for k in renv.obj_ids}
        mode = ["none" if any(s in k.lower() for s in ROT_NONE) else "tilt" if any(s in k.lower() for s in ROT_TILT)
                else "full" for k in self._object_names()]
        self.rot_full = torch.tensor([x == "full" for x in mode], device=dev)
        self.rot_tilt = torch.tensor([x == "tilt" for x in mode], device=dev)
        self.guide_gain = 0.0
        self.hold = False
        self.success = SuccessTracker(renv, success_term, success_steps)

    def _object_names(self):
        """Asset names used to pick the rotation-error mode: the USD file of each object, else its key."""
        out = []
        for k, a in self.e.objects.items():
            spawn = getattr(a.cfg, "spawn", None)
            out.append(f"{k} {getattr(spawn, 'usd_path', '') or ''}")
        return out

    def update_command(self):
        """``ref_cmd``: the joint targets that make the PD-driven hand follow ``ref_q``."""
        v = torch.zeros_like(self.ref_q)
        fps = 1.0 / self.e.step_dt
        v[1:-1] = (self.ref_q[2:] - self.ref_q[:-2]) * (0.5 * fps)
        self.ref_cmd = torch.max(torch.min(self.ref_q + self.tau * v, self.e.q_hi), self.e.q_lo)

    @staticmethod
    def _make_gate(active, margin, ramp):
        T = len(active)
        idx = np.where(active >= 0)[0]
        if not len(idx):
            return np.zeros(T, np.float32)
        dd = np.abs(np.arange(T)[:, None] - idx[None]).min(1)
        return np.clip(1.0 - (dd - margin) / max(ramp, 1), 0.0, 1.0).astype(np.float32)

    def f(self, offset=0):
        return torch.clamp(self.frame + offset, max=self.T - 1)

    # ------------------------------------------------------------- reads
    def tips(self):
        return self.e.tips()

    def obj_rot_dist(self, quat, ref):
        full = axis_angle_from_quat(quat_mul(ref.reshape(-1, 4), quat_conjugate(quat.reshape(-1, 4)))).norm(dim=-1)
        full = full.reshape(quat.shape[:-1])
        z = torch.zeros_like(quat[..., 1:])
        z[..., 2] = 1.0
        a = quat_apply(quat.reshape(-1, 4), z.reshape(-1, 3)).reshape(z.shape)
        b = quat_apply(ref.reshape(-1, 4), z.reshape(-1, 3)).reshape(z.shape)
        tilt = torch.acos((a * b).sum(-1).clamp(-1.0, 1.0))
        return torch.where(self.rot_full, full, torch.where(self.rot_tilt, tilt, torch.zeros_like(full)))

    def demo_contact_world(self, body_pose, f):
        """Demo contact points of frame f placed on the current body poses: (N,F,3), valid (N,F), body idx."""
        cb = self.contact_body[f]
        valid = cb >= 0
        idx = cb.clamp(min=0)
        p = torch.gather(body_pose[..., :3], 1, idx[..., None].expand(-1, -1, 3))
        q = torch.gather(body_pose[..., 3:7], 1, idx[..., None].expand(-1, -1, 4))
        loc = self.contact_local[f]
        return quat_apply(q.reshape(-1, 4), loc.reshape(-1, 3)).reshape(p.shape) + p, valid, idx

    # ------------------------------------------------------------- control
    def set_action(self, actions):
        cfg, e = self.cfg, self.e
        self.actions = actions.clamp(-1.0, 1.0)
        f1 = self.f(1)
        if self.hold:
            e.q_target[:] = self.ref_cmd[f1]
            return
        g = self.gate[f1][:, None]
        a = self.actions * g
        scale = torch.tensor([cfg.wrist_trans_scale] * 3 + [cfg.wrist_rot_scale] * 3, device=self.device)
        res = torch.zeros_like(e.q_target)
        res[:, self.wrist_ids] = scale * a[:, :6]
        res[:, self.finger_ids] = cfg.finger_res_cap * a[:, 6:]
        # the low-pass acts on the residual only: filtering the whole target (Bench2Dex) would add ~0.25 s of
        # lag to the moving reference on top of the PD lag
        self.res = cfg.action_ema * self.res + (1.0 - cfg.action_ema) * res
        e.q_target[:] = torch.max(torch.min(self.ref_cmd[f1] + self.res, e.q_hi), e.q_lo)

    def step(self, actions=None, hook=None):
        """One control frame on every env. ``actions=None`` holds the current targets."""
        if actions is not None:
            self.set_action(actions)

        def sub():
            if self.hold:
                f1 = self.f(1)
                self._write_objects(self.ref_root[f1], {k: v[f1] for k, v in self.ref_art.items()})
            else:
                self._apply_guidance()
            if hook is not None:
                hook()

        self._apply_guidance() if not self.hold else None
        self.e.physics_step(hook=sub)
        if actions is not None:
            self.frame += 1

    def _apply_guidance(self):
        e = self.e
        n, dev = self.num_envs, self.device
        B = len(e.bodies)
        F = torch.zeros(n, B, 3, device=dev)
        Tq = torch.zeros(n, B, 3, device=dev)
        if self.guide_gain > 0.0 and B:
            body = e.object_body_poses()
            cp, valid, idx = self.demo_contact_world(body, self.f(1))
            fi = self.guide_gain * (self.tips() - cp) * valid[..., None].float()
            bpos = torch.gather(body[..., :3], 1, idx[..., None].expand(-1, -1, 3))
            ti = torch.cross(cp - bpos, fi, dim=-1)
            F.scatter_add_(1, idx[..., None].expand(-1, -1, 3), fi)
            Tq.scatter_add_(1, idx[..., None].expand(-1, -1, 3), ti)
            held = torch.zeros(n, B, 1, device=dev).scatter_add_(1, idx[..., None], valid[..., None].float()) > 0
            lin = torch.stack([e.objects[k].data.body_lin_vel_w[:, i] for k, _, i in e.bodies], 1)
            F = F - 2.0 * self.body_mass * lin * held
            cap = 3.0 * 9.81 * self.body_mass
            F = F * torch.clamp(cap / F.norm(dim=-1, keepdim=True).clamp(min=1e-6), max=1.0)
            tcap = 80.0 * self.body_inertia
            Tq = Tq * torch.clamp(tcap / Tq.norm(dim=-1, keepdim=True).clamp(min=1e-9), max=1.0)
        for k, ids in self.asset_bodies.items():
            a = e.objects[k]
            a.permanent_wrench_composer.set_forces_and_torques(
                forces=F[:, ids].contiguous(), torques=Tq[:, ids].contiguous(),
                body_ids=[e.bodies[j][2] for j in ids] if isinstance(a, Articulation) else None, is_global=True)

    def _write_objects(self, roots, art):
        e = self.e
        n = self.num_envs
        o = e.origins()
        zero = torch.zeros(n, 6, device=self.device)
        for j, (k, a) in enumerate(e.objects.items()):
            p = roots[j].expand(n, -1).clone() if roots.dim() == 2 else roots[:, j].clone()
            p[:, :3] += o
            a.write_root_pose_to_sim(p)
            a.write_root_velocity_to_sim(zero)
            if k in art:
                v = art[k].expand(n, -1) if art[k].dim() == 1 else art[k]
                a.write_joint_state_to_sim(v, torch.zeros_like(v))

    # ------------------------------------------------------------- reset / snapshots
    def initial_state(self, src_initial_state):
        """The episode's initial scene state with the target hand at the reference's first frame."""
        st = copy.deepcopy(src_initial_state)
        r = self.e.robot
        q0 = self.ref_q[0].cpu().numpy()[None]
        root = r.data.default_root_state[0].cpu().numpy()
        st["articulation"]["robot"] = {"root_pose": root[None, :7].astype(np.float32),
                                       "root_velocity": np.zeros((1, 6), np.float32),
                                       "joint_position": q0.astype(np.float32),
                                       "joint_velocity": np.zeros_like(q0, dtype=np.float32)}
        return st

    def reset(self, state):
        self.e.set_state(state)
        self.frame[:] = 0
        self.actions[:] = 0.0
        self.res[:] = 0.0
        self.e.q_target[:] = self.ref_cmd[0]
        self.success.reset()

    def snapshot(self, i=0):
        return {"state": to_numpy_state(self.e.get_state(), i), "q_target": self.e.q_target[i].clone(),
                "actions": self.actions[i].clone(), "res": self.res[i].clone(), "frame": int(self.frame[i])}

    def restore(self, s):
        self.e.set_state(s["state"])
        self.e.q_target[:] = s["q_target"]
        self.e.robot.set_joint_position_target(self.e.q_target)
        self.actions[:] = s["actions"]
        self.res[:] = s["res"]
        self.frame[:] = s["frame"]


class SuccessTracker:
    """The task's success term on env 0, with record_demos' rule: success held ``n`` consecutive frames."""

    def __init__(self, renv: RetargetEnv, term, n: int):
        self.env = renv.env
        self.term = term
        self.n = int(n)
        self.history: list[bool] = []

    def reset(self):
        self.history = []
        if hasattr(self.term.func, "reset"):
            try:
                self.term.func.reset()
            except TypeError:
                pass

    def evaluate(self) -> torch.Tensor:
        return self.term.func(self.env, **self.term.params)

    def update(self) -> bool:
        ok = bool(self.evaluate()[0])
        self.history.append(ok)
        return ok

    def done_frame(self):
        run = 0
        for i, ok in enumerate(self.history):
            run = run + 1 if ok else 0
            if run >= self.n:
                return i
        return None

    def state(self):
        return len(self.history)

    def load(self, n):
        del self.history[n:]
