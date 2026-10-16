# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrist force/torque ("wrench") logging of the floating hands, in the layout of the real arm's wrist F/T sensor.

A floating hand hangs from six virtual wrist joints (see ``robot_agents/floating_wrist.py``); the joint between the
virtual chain and the hand plays the part of the arm flange / wrist sensor. After every physics substep (read-only
hook after ``scene.update``) the recorder reads PhysX's incoming joint force of the hand MOUNT body (the first real
link after the virtual chain: Shadow ``palm``, Allegro ``base_link``, Inspire ``hand_root``, LEAP ``palm_lower``,
Sharpa ``*_hand_C_MC``, Wuji ``*_palm_link``) and stores, per hand:

* ``observation.wrench`` [6 per hand] = ``Fx, Fy, Fz`` (N), ``Tx, Ty, Tz`` (Nm): the wrench the ENVIRONMENT exerts on
  the robot through the wrist, i.e. minus the force / torque the virtual arm applies to the hand -- the sign
  convention of the real recorder (robo-synapse ``Arm.external_wrench``). Axes: the mount body's link frame (moves
  with the hand, like the real ``tool0_controller`` frame); torque about the mount joint's anchor point on the hand.
  Hand gravity is disabled in DexVerse, so the value is already "tared": 0 at rest without contact; it contains the
  contact forces on the hand, the weight of a held object and the hand's inertia (-m a).
* ``observation.wrist_drive_saturated`` [1 per hand] uint8: 1 if a virtual wrist joint's PD drive hit its effort
  limit during the control step -- the wrench is then capped by the virtual arm, not by the contact.

Bimanual: right hand first, then left (the order of ``observation.touch``). Reduction over the physics substeps of
the control step: as for tactile (``DEXVERSE_TACTILE_REDUCE``: ``mean`` = average, default; ``last``). The frame
right after a reset / state restore is zeros. ``DEXVERSE_WRENCH=0`` turns it off.
"""

from __future__ import annotations

import os
import re

import numpy as np

ENABLE_ENV = "DEXVERSE_WRENCH"
WRENCH_FEATURE = "observation.wrench"
SATURATED_FEATURE = "observation.wrist_drive_saturated"
WRENCH_COMPONENTS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")

VIRTUAL_JOINT = re.compile(r"^(?:(?:right|left|rh|lh)_)?(?:[xyz]_(?:translation|rotation)_joint|[xyz]_joint|roll_joint"
                           r"|pitch_joint|yaw_joint|wrist_(?:trans|rot)_[xyz])$")
_SIDE = re.compile(r"^(right|left|rh|lh)_")

# PhysX get_link_incoming_joint_force(): force / torque the incoming joint applies to the CHILD link (the parent
# frame stated in Isaac Lab's docstring does not hold), in the child-side joint frame (body1 localRot1), torque about
# that frame's origin (body1 localPos1). Verified by Newton balance (joint force = sum m a - contact force on the hand,
# gravity off) on replayed demos, 10/2026: Sharpa VR 0.2% force / 0.8% torque residual, Inspire VR 6.4% / 3.4%,
# Shadow 8-21% force, the excess at grasp impacts (frame median 0.02-0.13 N). The mount joints of all six hands
# have identity localRot1 and zero localPos1, so the axes are the mount body's link frame and the torque is about
# its origin. scripts/experiments/wrist_wrench_probe.py + wrist_wrench_probe_analyze.py.
PHYSX_SIGN = 1.0

WRENCH_DEFINITIONS = {
    WRENCH_FEATURE: "per hand [Fx,Fy,Fz,Tx,Ty,Tz] (N, Nm): wrench the environment exerts on the robot through the "
                    "virtual wrist = minus the force / torque the virtual arm applies to the hand (PhysX incoming "
                    "joint force of the hand mount body); axes = mount body link frame; torque about the mount joint "
                    "anchor on the hand. Hand gravity is off, so it is 'tared': contact forces on the hand + weight "
                    "of a held object - hand inertia (m a). Bimanual: right hand first.",
    SATURATED_FEATURE: "per hand uint8: 1 if a virtual wrist joint's PD drive reached its effort limit in a physics "
                       "substep of the control step (the wrench is then capped by the virtual arm)",
    "real_format": "real UR7e + RH5DG2 export (robo-synapse): observation.wrench float32 [6] Fx,Fy,Fz,Tx,Ty,Tz in N / "
                   "Nm, frame tool0_controller, environment-on-robot sign, tared per episode and EMA-smoothed; "
                   "stored next to observation.state, not inside it",
    "differences_to_real": "no sensor noise / EMA; axes and torque reference of the mount body instead of tool0; "
                           "capped by the virtual wrist drive (wrist_drive.effort_limit, per axis); no residual of the "
                           "hand weight when the wrist tilts (gravity off instead of a one-pose tare)",
}


def wrench_enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "1").strip().lower() not in ("0", "false", "off", "no")


def _np(x):
    return np.array(x.detach().to("cpu").numpy() if hasattr(x, "detach") else x, dtype=np.float64, copy=True)


def _qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _side_of(name: str, robot_type: str) -> str:
    m = _SIDE.match(name)
    if m:
        return "right" if m.group(1) in ("right", "rh") else "left"
    return "left" if str(robot_type).endswith("_left") else "right"


def find_wrist_mounts(env) -> list[dict]:
    """Per hand (right first): the mount body (first real link after the virtual wrist chain), its incoming joint
    and that joint's child-side local frame, the virtual wrist joint ids. From the USD stage of env 0."""
    from pxr import UsdPhysics  # noqa: PLC0415

    robot = env.scene["robot"]
    bnames = list(robot.body_names)
    joints = []
    authored = {}   # USD-authored link masses: PhysX gives massless dummy links a default 1 kg at runtime
    for prim in env.sim.stage.Traverse():
        if not str(prim.GetPath()).startswith("/World/envs/env_0/"):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            authored[prim.GetName()] = (UsdPhysics.MassAPI(prim).GetMassAttr().Get()
                                        if prim.HasAPI(UsdPhysics.MassAPI) else None)
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        b0 = [str(t).rsplit("/", 1)[-1] for t in j.GetBody0Rel().GetTargets()]
        b1 = [str(t).rsplit("/", 1)[-1] for t in j.GetBody1Rel().GetTargets()]
        if not b1 or b1[0] not in bnames:
            continue
        q1 = j.GetLocalRot1Attr().Get()
        joints.append({"name": prim.GetName(), "parent": b0[0] if b0 else None, "child": b1[0],
                       "pos1": np.array(j.GetLocalPos1Attr().Get() or (0.0, 0.0, 0.0), float),
                       "q1": np.array([1.0, 0.0, 0.0, 0.0] if q1 is None else [q1.GetReal(), *q1.GetImaginary()])})
    virt = [j for j in joints if VIRTUAL_JOINT.match(j["name"])]
    ends = [j for j in virt if j["child"] not in {v["parent"] for v in virt}]
    if not ends:
        raise RuntimeError("[wrench] no virtual wrist joint chain found (not a floating hand?)")
    mounts = []
    for e in ends:
        c = e["child"]
        if (authored.get(c) or 0.0) > 1e-4:
            mj = e
        else:   # an (authored-)massless dummy closes the chain and the hand hangs from it by one fixed joint
            nxt = [j for j in joints if j["parent"] == c and not VIRTUAL_JOINT.match(j["name"])]
            if len(nxt) != 1:
                raise RuntimeError(f"[wrench] chain end {c!r}: expected one joint to the hand, got {nxt}")
            mj = nxt[0]
        side = _side_of(mj["child"], getattr(env.cfg, "robot_type", ""))
        jn = robot.joint_names
        wrist_ids = [i for i, n in enumerate(jn) if VIRTUAL_JOINT.match(n) and
                     (len(ends) == 1 or _side_of(n, "") == side)]
        mounts.append({"side": side, "body": mj["child"], "body_id": bnames.index(mj["child"]), "joint": mj["name"],
                       "R1": _qmat(mj["q1"]), "pos1": mj["pos1"], "wrist_joint_ids": wrist_ids})
    mounts.sort(key=lambda m: 0 if m["side"] == "right" else 1)
    return mounts


class WristWrenchLogger:
    """Per-substep wrist wrench + drive saturation, reduced per control step (module docstring). Env 0."""

    def __init__(self, env, reduce: str = "mean"):
        self.env = env
        self.robot = env.scene["robot"]
        self.reduce = reduce
        self.mounts = find_wrist_mounts(env)
        self.n_hands = len(self.mounts)
        self.body_ids = [m["body_id"] for m in self.mounts]
        # raw (child joint frame, about the anchor) -> mount body axes, about the anchor, environment-on-robot sign
        self.M = [-PHYSX_SIGN * m["R1"] for m in self.mounts]
        self.decimation = int(env.cfg.decimation)
        self.physics_dt = float(env.physics_dt)
        self._reset_acc()
        if self.reduce == "mean":
            scene = env.scene
            if getattr(scene, "_tactile_pad_pose_hooks", None) is None:   # same read-only hook as tactile_pads
                orig = scene.update
                scene._tactile_pad_pose_hooks = []

                def update(dt, _orig=orig, _scene=scene):
                    _orig(dt)
                    for fn in _scene._tactile_pad_pose_hooks:
                        fn()

                scene.update = update
            scene._tactile_pad_pose_hooks.append(self._on_substep)

    def names(self) -> list[str]:
        pre = [""] if self.n_hands == 1 else [f"{m['side']}." for m in self.mounts]
        return [f"{p}{c}" for p in pre for c in WRENCH_COMPONENTS]

    def side_names(self) -> list[str]:
        return [m["side"] for m in self.mounts]

    def _reset_acc(self):
        self._n = 0
        self._sum = np.zeros((len(getattr(self, "mounts", [])), 6))
        self._sat = np.zeros(len(getattr(self, "mounts", [])), bool)

    def _read(self):
        raw = _np(self.robot.root_physx_view.get_link_incoming_joint_force()[0])[self.body_ids]   # (H, 6)
        w = np.stack([np.r_[M @ r[:3], M @ r[3:]] for M, r in zip(self.M, raw)])
        d = self.robot.data
        comp = _np(d.computed_torque[0])
        lim = _np(d.joint_effort_limits[0])
        sat = np.array([bool((np.abs(comp[m["wrist_joint_ids"]]) >= 0.999 * lim[m["wrist_joint_ids"]]).any())
                        for m in self.mounts])
        return w, sat

    def _on_substep(self):
        w, sat = self._read()
        self._sum += w
        self._sat |= sat
        self._n += 1

    def mark_reset(self):
        self._reset_acc()

    def zeros(self):
        return np.zeros(6 * self.n_hands, np.float32), np.zeros(self.n_hands, np.uint8)

    def capture(self, force_zero: bool):
        """(wrench [6 H] float32, saturated [H] uint8, substeps) of the frame; zeros after a reset."""
        if force_zero:
            self.mark_reset()
            return (*self.zeros(), 0)
        if self.reduce == "last":
            w, sat = self._read()
            return w.reshape(-1).astype(np.float32), sat.astype(np.uint8), 1
        n = self._n
        if n == 0:
            self._reset_acc()
            return (*self.zeros(), 0)
        out = ((self._sum / n).reshape(-1).astype(np.float32), self._sat.astype(np.uint8), n)
        self._reset_acc()
        return out

    def info(self) -> dict:
        d = self.robot.data
        drive = []
        for m in self.mounts:
            ids = m["wrist_joint_ids"]
            drive.append({"side": m["side"], "joints": [self.robot.joint_names[i] for i in ids],
                          "stiffness": _np(d.joint_stiffness[0])[ids].tolist(),
                          "damping": _np(d.joint_damping[0])[ids].tolist(),
                          "effort_limit": _np(d.joint_effort_limits[0])[ids].tolist(),
                          "velocity_limit": _np(d.joint_vel_limits[0])[ids].tolist()})
        return {"reduce": self.reduce, "substeps_per_frame": self.decimation, "physics_dt": self.physics_dt,
                "hands": [{"side": m["side"], "mount_body": m["body"], "mount_joint": m["joint"],
                           "anchor_in_mount_body_m": m["pos1"].tolist()} for m in self.mounts],
                "names": self.names(), "units": {"force": "N", "torque": "Nm"}, "wrist_drive": drive,
                "definitions": WRENCH_DEFINITIONS,
                "switches": {ENABLE_ENV: "0 disables wrench logging",
                             "DEXVERSE_TACTILE_REDUCE": "mean (default) | last, shared with tactile"}}
