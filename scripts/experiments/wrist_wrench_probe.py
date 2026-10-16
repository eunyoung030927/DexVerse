# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe of the floating-wrist joint wrench (experiment, not a tool).

Replays one episode of a trajectory pickle (reset, restore the initial state, feed the recorded actions) and, after
EVERY physics substep (read-only hook after ``scene.update``), stores for env 0:

* the raw PhysX incoming joint force (``get_link_incoming_joint_force``, 6) of the hand mount body (the first real
  link after the virtual wrist chain) and of the last virtual-chain link;
* poses of those bodies and of their parents, and the USD local frames of their incoming joints;
* masses, COM positions and COM linear velocities of every hand body (Newton: sum m a);
* object-filtered contact forces (normal + friction) on every hand body, their sum and their moment about the world
  origin (normal rows at the contact points, friction rows at their patch anchors), and the unfiltered net normal
  force of every hand body (to find frames where the hand touches nothing but the object).

    /workspace/isaaclab/_isaac_sim/python.sh wrist_wrench_probe.py --task Dexverse-GraspKettle-v0 \
        --robot_type floating_shadow_right --pickle demos.pkl --episode 0 --out probe.npz --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--robot_type", required=True)
parser.add_argument("--pickle", required=True)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--max_steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", rendering_mode="performance")
args = parser.parse_args()
app = AppLauncher(vars(args)).app

import pickle  # noqa: E402
import re  # noqa: E402

import dexverse.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dexverse.data_collection.tactile import RawContactView, resolve_targets  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from dexverse.teleop_utils.episode_task_state import reset_managers_after_state_restore  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

VIRTUAL = re.compile(r"^(?:(?:right|left|rh|lh)_)?(?:[xyz]_(?:translation|rotation)_joint|[xyz]_joint|roll_joint"
                     r"|pitch_joint|yaw_joint|wrist_(?:trans|rot)_[xyz])$")

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg = type(cfg)(robot_type=args.robot_type)
cfg.sim.device = args.device
cfg.scene.num_envs = 1
cfg.terminations.success = None
cfg.terminations.time_out = None
cfg.recorders = {}
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
bnames = list(robot.body_names)


def _np(x):
    return np.array(x.detach().to("cpu").numpy() if hasattr(x, "detach") else x, dtype=np.float64, copy=True)


# ---- articulation structure from the USD stage of env 0
stage = env.sim.stage
root = robot.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace(".*", "")
joints = []
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if not p.startswith("/World/envs/env_0/") or not prim.IsA(UsdPhysics.Joint):
        continue
    j = UsdPhysics.Joint(prim)
    b0 = [str(t).rsplit("/", 1)[-1] for t in j.GetBody0Rel().GetTargets()]
    b1 = [str(t).rsplit("/", 1)[-1] for t in j.GetBody1Rel().GetTargets()]
    if not b1 or b1[0] not in bnames:
        continue
    joints.append({"name": prim.GetName(), "type": prim.GetTypeName(), "parent": b0[0] if b0 else None,
                   "child": b1[0], "pos0": list(j.GetLocalPos0Attr().Get() or (0, 0, 0)),
                   "q0": j.GetLocalRot0Attr().Get(), "pos1": list(j.GetLocalPos1Attr().Get() or (0, 0, 0)),
                   "q1": j.GetLocalRot1Attr().Get()})


def _quat(q):
    return [1.0, 0.0, 0.0, 0.0] if q is None else [q.GetReal(), *q.GetImaginary()]


for j in joints:
    j["q0"], j["q1"] = _quat(j["q0"]), _quat(j["q1"])
virt = [j for j in joints if VIRTUAL.match(j["name"])]
masses = _np(robot.root_physx_view.get_masses()[0])
# AUTHORED masses (USD MassAPI): PhysX gives massless dummy links a default mass at runtime
authored = {}
for prim in stage.Traverse():
    if str(prim.GetPath()).startswith("/World/envs/env_0/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        mv = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
        authored[prim.GetName()] = mv
print("RUNTIME MASS of virtual-chain links", {j["child"]: round(float(masses[bnames.index(j["child"])]), 4)
                                               for j in virt})
# chain ends = children of virtual joints that are not the parent of another virtual joint
ends = [j for j in virt if j["child"] not in {v["parent"] for v in virt}]
mounts = []
for e in ends:
    c = e["child"]
    if (authored.get(c) or 0.0) > 1e-4:
        mounts.append({"body": c, "joint": e, "end": c})
    else:  # authored-massless dummy at the chain end: the hand hangs from it by a fixed joint
        nxt = [j for j in joints if j["parent"] == c and not VIRTUAL.match(j["name"])]
        assert len(nxt) == 1, (c, nxt)
        mounts.append({"body": nxt[0]["child"], "joint": nxt[0], "end": c})
print("MOUNTS", [(m["end"], m["joint"]["name"], m["body"]) for m in mounts])
# hand bodies = the subtree hanging from each mount body
children = {}
for j in joints:
    if not VIRTUAL.match(j["name"]):
        children.setdefault(j["parent"], []).append(j["child"])
hand_bodies = []
for mt in mounts:
    stack = [mt["body"]]
    while stack:
        b = stack.pop()
        if b not in hand_bodies:
            hand_bodies.append(b)
            stack += children.get(b, [])
hand_bodies = [b for b in bnames if b in hand_bodies]
hand_ids = [bnames.index(b) for b in hand_bodies]
mount_ids = [bnames.index(m["body"]) for m in mounts]
end_ids = [bnames.index(m["end"]) for m in mounts]
parent_ids = [bnames.index(m["joint"]["parent"]) for m in mounts]
print("hand bodies", len(hand_bodies), "mass", masses[hand_ids].sum())

targets = resolve_targets(env.cfg)
cv = RawContactView(env, hand_bodies, targets, max_contacts_per_body=32)
# the recorder's own logger, fed by the same substeps (its hook is installed before the probe's wrapper below)
from dexverse.data_collection.wrist_wrench import WristWrenchLogger  # noqa: E402

wl = WristWrenchLogger(env, "mean")
print("LOGGER MOUNTS", [(m["side"], m["body"], m["joint"]) for m in wl.mounts], "names", wl.names())
logged, logged_sat = [], []

log = {k: [] for k in ("raw_mount", "raw_end", "pos_mount", "quat_mount", "pos_parent", "quat_parent", "pos_end",
                       "quat_end", "com_pos", "com_vel", "f_obj", "f_obj_normal", "m_obj_O", "net_normal", "step")}
state = {"step": -1, "on": False}


def on_substep():
    if not state["on"]:
        return
    d = robot.data
    raw = _np(robot.root_physx_view.get_link_incoming_joint_force()[0])     # (B, 6)
    log["raw_mount"].append(raw[mount_ids])
    log["raw_end"].append(raw[end_ids])
    pw, qw = _np(d.body_link_pos_w[0]), _np(d.body_link_quat_w[0])
    log["pos_mount"].append(pw[mount_ids]); log["quat_mount"].append(qw[mount_ids])
    log["pos_parent"].append(pw[parent_ids]); log["quat_parent"].append(qw[parent_ids])
    log["pos_end"].append(pw[end_ids]); log["quat_end"].append(qw[end_ids])
    log["com_pos"].append(_np(d.body_com_pos_w[0])[hand_ids])
    log["com_vel"].append(_np(d.body_com_lin_vel_w[0])[hand_ids])
    c = cv.read(raw=True)
    rows = [cv.names.index(b) for b in hand_bodies]
    f_obj = (c["fmat"].sum(1) + c["fric"].sum(1))[rows]                     # (H, 3) world
    log["f_obj"].append(f_obj)
    log["f_obj_normal"].append(c["fmat"].sum(1)[rows])
    log["net_normal"].append(c["net"][rows])
    # moment about the world origin: normal rows at contact points, friction rows at patch anchors
    forces = c["contact"][0].reshape(-1)
    points, normals = c["contact"][1].reshape(-1, 3), c["contact"][2].reshape(-1, 3)
    cnt = int(np.asarray(c["contact"][4]).sum())
    st = np.asarray(c["contact"][5]).reshape(-1).astype(int)
    ct = np.asarray(c["contact"][4]).reshape(-1).astype(int)
    m = np.zeros(3)
    for s, n in zip(st, ct):
        if n:
            fv = forces[s:s + n, None] * normals[s:s + n]
            m += np.cross(points[s:s + n], fv).sum(0)
    ff, fp = c["friction"][0].reshape(-1, 3), c["friction"][1].reshape(-1, 3)
    fs, fc = np.asarray(c["friction"][3]).reshape(-1).astype(int), np.asarray(c["friction"][2]).reshape(-1).astype(int)
    for s, n in zip(fs, fc):
        if n:
            m += np.cross(fp[s:s + n], np.nan_to_num(ff[s:s + n])).sum(0)
    log["m_obj_O"].append(m)
    log["step"].append(state["step"])


scene = env.scene
_orig = scene.update


def _update(dt):
    _orig(dt)
    on_substep()


scene.update = _update

with open(args.pickle, "rb") as f:
    ep = [e for e in pickle.load(f)["episodes"] if len(e.get("actions", []))][args.episode]


def to_t(x):
    if isinstance(x, dict):
        return {k: to_t(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=env.device)
    return x


env.sim.reset()
env.reset()
env.reset_to(to_t(ep["initial_state"]), env_ids=None, is_relative=True)
reset_managers_after_state_restore(env)
wl.mark_reset()
acts = ep["actions"][: args.max_steps or None]
state["on"] = True
with torch.inference_mode():
    for t, a in enumerate(acts):
        state["step"] = t
        env.step(torch.as_tensor(np.asarray(a, dtype=np.float32).reshape(1, -1), device=env.device))
        w, sat, _n = wl.capture(False)
        logged.append(w)
        logged_sat.append(sat)
arr = {k: np.asarray(v) for k, v in log.items()}
np.savez(args.out, masses=masses[hand_ids], physics_dt=float(env.physics_dt), hand_bodies=np.array(hand_bodies),
         mounts=np.array([m["body"] for m in mounts]), ends=np.array([m["end"] for m in mounts]),
         mount_joint_pos1=np.array([m["joint"]["pos1"] for m in mounts]),
         mount_joint_q1=np.array([m["joint"]["q1"] for m in mounts]),
         mount_joint_pos0=np.array([m["joint"]["pos0"] for m in mounts]),
         mount_joint_q0=np.array([m["joint"]["q0"] for m in mounts]),
         mount_joint_names=np.array([m["joint"]["name"] for m in mounts]), logged=np.asarray(logged),
         logged_sat=np.asarray(logged_sat), logger_mounts=np.array([m["body"] for m in wl.mounts]), **arr)
print("PROBE_DONE", {k: v.shape for k, v in arr.items()})
env.close()
app.close()
