# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent physical verification of the --tactile contact forces (experiment, not a tool).

Replays one episode of a trajectory pickle (same flow as tactile_replay_check.py) and records, after EVERY physics
substep (hook on ``env.scene.update``):

1. every robot link vs the target object(s): normal (force matrix) + friction + raw contact points
   (``vf_all`` ContactSensor over ``Robot/.*``), plus each link's unfiltered net normal force (all partners);
2. object side (rigid-object tasks): object vs every robot link (Newton-3 check), object vs table, object unfiltered
   net normal, object COM velocity / pose and mass -> m a = sum F + m g per substep and per control step;
3. joint level: contact generalized force tau_c = sum_i J(p_i)^T f_i from the raw per-contact buffers
   (normal + friction at their own contact points), PD torque estimates (Isaac Lab applied_torque = pre-step state,
   and k (q* - q) - d qd with the post-step state), PhysX DOF projected joint forces, gravity / Coriolis
   compensation forces and M qdd;
4. the --tactile feature itself (TactileReader at control-step end) for comparison with the independent sensors
   and with a recorded spool (``--spool_npz``: frame t of the dataset = end of control step t-1).

Optionally (``--image_steps``) a close-up camera renders given control steps looking at ``--image_body``
(rendering mid-episode is kept out of the force runs: it is a separate invocation).

    /workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/tactile_force_verify.py --task Dexverse-PickCube-v0 \
        --robot_type floating_allegro_right --pickle demo.pkl --out /root/tactile_test/verify/al --headless \
        --enable_cameras [--spool_npz .../ep_000000/data.npz] [--image_steps 300 400 --image_body thdistal]
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--robot_type", required=True)
parser.add_argument("--pickle", required=True)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--spool_npz", default=None)
parser.add_argument("--image_steps", type=int, nargs="*", default=[])
parser.add_argument("--image_body", nargs="*", default=[])
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--object_sensors", action="store_true",
                    help="also put ContactSensors on the object (needs PhysxContactReportAPI on it, which CHANGES the "
                         "replayed trajectory -- see the report's spool comparison)")
parser.add_argument("--net_sensor", action="store_true", help="separate unfiltered ContactSensor over the hand links")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", rendering_mode="performance")
args = parser.parse_args()
app = AppLauncher(vars(args)).app


def _exit_on_error(exc_type, exc, tb):
    import traceback  # noqa: PLC0415

    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    os._exit(1)


sys.excepthook = _exit_on_error

import pickle  # noqa: E402

import dexverse  # noqa: E402
import dexverse.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dexverse.data_collection.tactile import (  # noqa: E402
    TactileReader,
    add_tactile_sensors,
    articulation_link_prims,
)
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from dexverse.teleop_utils.episode_task_state import reset_managers_after_state_restore  # noqa: E402
from isaaclab.sensors import CameraCfg, ContactSensorCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

print("DEXVERSE_FILE", dexverse.__file__, flush=True)
os.makedirs(args.out, exist_ok=True)

# ------------------------------------------------------------------ config
cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg = type(cfg)(robot_type=args.robot_type)
cfg.sim.device = args.device
cfg.scene.num_envs = 1
cfg.terminations.success = None
cfg.terminations.time_out = None
cfg.recorders = {}
tac = add_tactile_sensors(cfg)
targets = tac["targets"]
robot_links = articulation_link_prims(cfg.scene.robot)
print(f"[verify] {len(robot_links)} robot links; targets {targets}", flush=True)
cfg.scene.vf_all = ContactSensorCfg(prim_path=f"{cfg.scene.robot.prim_path}/.*", filter_prim_paths_expr=list(targets),
                                    track_friction_forces=True, track_contact_points=True,
                                    max_contact_data_count_per_prim=32)
if args.net_sensor:
    cfg.scene.vf_all_net = ContactSensorCfg(prim_path=f"{cfg.scene.robot.prim_path}/.*")
has_obj = getattr(cfg.scene, "object", None) is not None
has_table = getattr(cfg.scene, "table", None) is not None
if has_obj and args.object_sensors:
    op = cfg.scene.object.prim_path
    # object-side sensors need PhysxContactReportAPI on the object (report-only; the spool comparison in the
    # report checks that the replay is unchanged by it)
    cfg.scene.object.spawn.activate_contact_sensors = True
    cfg.scene.vf_obj_net = ContactSensorCfg(prim_path=op)
    cfg.scene.vf_obj_hand = ContactSensorCfg(prim_path=op, filter_prim_paths_expr=list(robot_links),
                                             track_friction_forces=True, max_contact_data_count_per_prim=256)
    if has_table:
        cfg.scene.vf_obj_table = ContactSensorCfg(prim_path=op, filter_prim_paths_expr=[cfg.scene.table.prim_path],
                                                  track_friction_forces=True, max_contact_data_count_per_prim=64)
if args.image_steps:
    cfg.scene.vf_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/VerifyCam", height=480, width=480, data_types=["rgb"],
                                 spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 5.0)))

env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["object"] if has_obj else None
reader = TactileReader(env, tac["bodies"], tac["targets"])
vf_all = env.scene.sensors["vf_all"]
vf_net = env.scene.sensors["vf_all_net"] if args.net_sensor else vf_all  # net_forces_w = all partners, normal only
assert vf_all.contact_physx_view.filter_count == len(targets), "target filter mismatch"
sens_names = list(vf_all.body_names)
net_names = list(vf_net.body_names)
assert sens_names == net_names, "sensor body order differs"
rb = {n: i for i, n in enumerate(robot.body_names)}
sens2rob = np.array([rb[n] for n in sens_names])
pad_sens_idx = [sens_names.index(b) for b in tac["bodies"]]
obj_table_ok = False
if has_obj and args.object_sensors:
    vf_oh = env.scene.sensors["vf_obj_hand"]
    print("[verify] object-hand filter count", vf_oh.contact_physx_view.filter_count, "of", len(robot_links), flush=True)
    if has_table:
        obj_table_ok = env.scene.sensors["vf_obj_table"].contact_physx_view.filter_count == 1
    print("[verify] object-table sensor ok:", obj_table_ok, flush=True)
fixed = bool(robot.is_fixed_base)
view = robot.root_physx_view
dof_off = 0 if fixed else 6
phys_dt = env.physics_dt
print(f"[verify] fixed base {fixed}, dofs {robot.num_joints}, bodies {robot.num_bodies}, physics dt {phys_dt}, "
      f"decimation {env.cfg.decimation}", flush=True)
stiff = robot.data.joint_stiffness[0].cpu().numpy()
damp = robot.data.joint_damping[0].cpu().numpy()
g_vec = np.array(env.cfg.sim.gravity, dtype=np.float64)


def _t(x):
    # COPY: on the CPU pipeline .numpy() aliases Isaac Lab's in-place-updated buffers (stored rows would all change)
    return np.array(x.detach().to("cpu").numpy() if hasattr(x, "detach") else x, copy=True)


# ------------------------------------------------------------------ kinematic tree from the USD joints
# PhysX get_jacobians() raises "Unable to obtain Jacobian shape" for these articulations on the CPU pipeline, so the
# contact generalized force is built geometrically: joint anchor + axis in world from the child link pose and the
# joint's localPos1 / localRot1 / axis; the convention is checked against measured link velocities.
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

_stage = get_current_stage()
_root = _stage.GetPrimAtPath("/World/envs/env_0/Robot")
parent, jinfo = {}, {}
for prim in Usd.PrimRange(_root, Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdPhysics.Joint):
        continue
    J0 = UsdPhysics.Joint(prim)
    b0, b1 = J0.GetBody0Rel().GetTargets(), J0.GetBody1Rel().GetTargets()
    if not b0 or not b1:
        continue
    c, pa = b1[0].name, b0[0].name
    if c not in rb or pa not in rb:
        continue
    lp1 = np.array(J0.GetLocalPos1Attr().Get(), dtype=np.float64)
    q1 = J0.GetLocalRot1Attr().Get()
    q1 = np.array([q1.GetReal(), *q1.GetImaginary()], dtype=np.float64)
    kind, axis = "fixed", None
    if prim.IsA(UsdPhysics.RevoluteJoint):
        kind, axis = "rev", UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        kind, axis = "pri", UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get()
    dof = robot.joint_names.index(prim.GetName()) if (kind != "fixed" and prim.GetName() in robot.joint_names) else None
    parent[c] = pa
    jinfo[c] = {"name": prim.GetName(), "kind": kind, "axis": axis, "lp1": lp1, "q1": q1, "dof": dof}
anc = {}
for _n in robot.body_names:
    chain, cur = [], _n
    while cur in parent:
        if jinfo[cur]["dof"] is not None:
            chain.append(cur)
        cur = parent[cur]
    anc[_n] = chain
print(f"[verify] kinematic tree: {len(jinfo)} joints, {sum(j['dof'] is not None for j in jinfo.values())} with a dof "
      f"of {robot.num_joints}", flush=True)


def _qmat(q):  # wxyz -> R
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


_E = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}


def joint_frames(lp, lq):
    """World anchor + axis of every dof joint, keyed by child link name."""
    out = {}
    for c, j in jinfo.items():
        if j["dof"] is None:
            continue
        Rc = _qmat(lq[rb[c]])
        out[c] = (lp[rb[c]] + Rc @ j["lp1"], Rc @ _qmat(j["q1"]) @ _E[j["axis"]])
    return out


def gen_force(bname, p, f, frames):
    tau = np.zeros(robot.num_joints)
    for c in anc[bname]:
        j = jinfo[c]
        a_pos, a_ax = frames[c]
        tau[j["dof"]] += a_ax @ np.cross(p - a_pos, f) if j["kind"] == "rev" else a_ax @ f
    return tau


_api_err = {}


def _try(fn, n):
    try:
        return _t(fn()[0]).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        _api_err.setdefault(fn.__name__, repr(exc))
        return np.full(n, np.nan)


# link inertial data for the Newton-Euler generalized inertial / gravity forces (PhysX mass matrix / gravity
# compensation are unavailable here, same as the Jacobian)
_mass_b = _t(view.get_masses()[0]).astype(np.float64).reshape(-1)
_inert_b = _t(view.get_inertias()[0]).astype(np.float64).reshape(-1, 3, 3)
_grav_on = np.ones(robot.num_bodies, dtype=bool)
for _bn in robot.body_names:
    _p = _stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{_bn}")
    _a = _p.GetAttribute("physxRigidBody:disableGravity") if _p.IsValid() else None
    if _a is not None and _a.IsValid() and _a.Get():
        _grav_on[rb[_bn]] = False
print(f"[verify] robot links with gravity enabled: {int(_grav_on.sum())}/{robot.num_bodies}; hand mass "
      f"{_mass_b.sum():.3f} kg", flush=True)
_prev = {}


def inertial_and_gravity(frames):
    """tau_in = sum_b J_com,b^T [m a ; I alpha + w x I w]  (= M qdd + c), tau_g = sum_b J_com,b^T m g (enabled links)."""
    cp = _t(robot.data.body_com_pos_w[0]).astype(np.float64)
    cq = _t(robot.data.body_com_quat_w[0]).astype(np.float64)
    v = _t(robot.data.body_com_lin_vel_w[0]).astype(np.float64)
    w = _t(robot.data.body_com_ang_vel_w[0]).astype(np.float64)
    tau_in, tau_g = np.full(robot.num_joints, np.nan), np.zeros(robot.num_joints)
    have = "v" in _prev
    if have:
        tau_in = np.zeros(robot.num_joints)
    for n in robot.body_names:
        b = rb[n]
        if not anc[n]:
            continue
        fg = _mass_b[b] * g_vec if _grav_on[b] else np.zeros(3)
        if have:
            R = _qmat(cq[b])
            Iw = R @ _inert_b[b] @ R.T
            fa = _mass_b[b] * (v[b] - _prev["v"][b]) / phys_dt
            ma = Iw @ ((w[b] - _prev["w"][b]) / phys_dt) + np.cross(w[b], Iw @ w[b])
        for c in anc[n]:
            j = jinfo[c]
            a_pos, a_ax = frames[c]
            if j["kind"] == "rev":
                tau_g[j["dof"]] += a_ax @ np.cross(cp[b] - a_pos, fg)
                if have:
                    tau_in[j["dof"]] += a_ax @ (np.cross(cp[b] - a_pos, fa) + ma)
            else:
                tau_g[j["dof"]] += a_ax @ fg
                if have:
                    tau_in[j["dof"]] += a_ax @ fa
    _prev["v"], _prev["w"] = v, w
    return tau_in, tau_g


# ------------------------------------------------------------------ raw-contact generalized force
sign_checks = []
jac_checks = []


def contact_tau(frames):
    """tau = sum_i J(p_i)^T f_i over every hand contact with the targets (normal and friction at their own points)."""
    forces, pts, nrm, _sep, cnt, st = vf_all.contact_physx_view.get_contact_data(dt=phys_dt)
    fr, fpts, fcnt, fst = vf_all.contact_physx_view.get_friction_data(dt=phys_dt)
    forces, pts, nrm, cnt, st = _t(forces).reshape(-1), _t(pts).reshape(-1, 3), _t(nrm).reshape(-1, 3), _t(cnt), _t(st)
    fr, fpts, fcnt, fst = _t(fr).reshape(-1, 3), _t(fpts).reshape(-1, 3), _t(fcnt), _t(fst)
    nb = len(sens_names)
    cnt, st = cnt.reshape(nb, -1), st.reshape(nb, -1)
    fcnt, fst = fcnt.reshape(nb, -1), fst.reshape(nb, -1)
    fm = _t(vf_all.data.force_matrix_w[0])  # (B, M, 3)
    ffm = np.nan_to_num(_t(vf_all.data.friction_forces_w[0]))      # (B, M, 3)
    cpm = _t(vf_all.data.contact_pos_w[0])                         # (B, M, 3) mean contact point, NaN if none
    tau_pair, tau_pt = np.zeros(robot.num_joints), np.zeros(robot.num_joints)
    for b in range(nb):
        bname = sens_names[b]
        for m in range(cnt.shape[1]):
            fpair = fm[b, m] + ffm[b, m]
            if np.linalg.norm(fpair) > 0 and np.isfinite(cpm[b, m]).all():
                # (A) pair resultant at the mean contact point (approximation of the patch moment)
                tau_pair += gen_force(bname, cpm[b, m], fpair, frames)
            c, s0 = int(cnt[b, m]), int(st[b, m])
            if c > 0:
                fsum = (forces[s0:s0 + c, None] * nrm[s0:s0 + c]).sum(0)
                ref = fm[b, m]
                if np.linalg.norm(ref) > 1e-3:
                    sign_checks.append(float(np.dot(fsum, ref) / np.dot(ref, ref)))
                # (B) per contact point: normal sign aligned with the pair's normal resultant, friction per point
                for k in range(c):
                    nk = nrm[s0 + k] * (1.0 if np.dot(nrm[s0 + k], ref) >= 0 else -1.0)
                    tau_pt += gen_force(bname, pts[s0 + k], forces[s0 + k] * nk, frames)
            c, s0 = int(fcnt[b, m]), int(fst[b, m])
            for k in range(c):
                tau_pt += gen_force(bname, fpts[s0 + k], fr[s0 + k], frames)
    return tau_pair, tau_pt


NORMAL_SIGN = 1.0

# ------------------------------------------------------------------ per-substep recorder
rec = {k: [] for k in ("step", "lnk_n", "lnk_f", "lnk_net", "q", "qd", "qtgt", "applied", "proj", "gcomp", "ccomp",
                       "Mqdd_M", "tau_n", "tau_f", "obj_vel", "obj_pos", "obj_net", "obj_hand_n", "obj_hand_f",
                       "obj_table_n", "obj_table_f", "tau_in", "tau_g", "incoming", "b_cpos", "b_cquat", "b_cvel",
                       "b_cang", "j_anchor", "j_axis", "obj_lpos")}
step_rec = {k: [] for k in ("pad_force_w", "pad_force", "pad_normal", "pad_net", "lnk_n", "lnk_f", "lnk_net")}
state = {"step": -1, "on": False}


def record_substep():
    s = state["step"]
    lp = _t(robot.data.body_link_pos_w[0]).astype(np.float64)
    lq = _t(robot.data.body_link_quat_w[0]).astype(np.float64)
    qd = _t(robot.data.joint_vel[0]).astype(np.float64)
    frames = joint_frames(lp, lq)
    if len(jac_checks) < 400:
        bl = _t(robot.data.body_link_lin_vel_w[0]).astype(np.float64)
        for n in robot.body_names:
            vb = bl[rb[n]]
            if not anc[n] or np.linalg.norm(vb) < 0.02:
                continue
            pred = np.zeros(3)
            for c in anc[n]:
                j = jinfo[c]
                a_pos, a_ax = frames[c]
                pred += qd[j["dof"]] * (np.cross(a_ax, lp[rb[n]] - a_pos) if j["kind"] == "rev" else a_ax)
            jac_checks.append((float(np.dot(pred, vb) / np.dot(vb, vb)), float(np.linalg.norm(pred - vb) / np.linalg.norm(vb))))
    tn, tf = contact_tau(frames)
    rec["step"].append(s)
    rec["lnk_n"].append(_t(vf_all.data.force_matrix_w[0]).sum(1))
    rec["lnk_f"].append(_t(vf_all.data.friction_forces_w[0]).sum(1))
    rec["lnk_net"].append(_t(vf_net.data.net_forces_w[0]))
    rec["q"].append(_t(robot.data.joint_pos[0]))
    rec["qd"].append(qd)
    rec["qtgt"].append(_t(robot.data.joint_pos_target[0]))
    rec["applied"].append(_t(robot.data.applied_torque[0]))
    nd = robot.num_joints + dof_off
    rec["proj"].append(_try(view.get_dof_projected_joint_forces, nd)[dof_off:])
    rec["gcomp"].append(_try(view.get_gravity_compensation_forces, nd)[dof_off:])
    rec["ccomp"].append(_try(view.get_coriolis_and_centrifugal_compensation_forces, nd)[dof_off:])
    M = _try(view.get_generalized_mass_matrices, nd * nd)
    rec["Mqdd_M"].append(M.reshape(nd, nd)[dof_off:, dof_off:])
    t_in, t_g = inertial_and_gravity(frames)
    rec["tau_in"].append(t_in)
    rec["b_cpos"].append(_t(robot.data.body_com_pos_w[0]))
    rec["b_cquat"].append(_t(robot.data.body_com_quat_w[0]))
    rec["b_cvel"].append(_t(robot.data.body_com_lin_vel_w[0]))
    rec["b_cang"].append(_t(robot.data.body_com_ang_vel_w[0]))
    anch, axs = np.full((robot.num_joints, 3), np.nan), np.full((robot.num_joints, 3), np.nan)
    for c, (ap, ax) in frames.items():
        anch[jinfo[c]["dof"]], axs[jinfo[c]["dof"]] = ap, ax
    rec["j_anchor"].append(anch)
    rec["j_axis"].append(axs)
    if obj is not None:
        rec["obj_lpos"].append(_t(obj.data.root_link_pos_w[0]))
    rec["tau_g"].append(t_g)
    rec["tau_n"].append(tn)
    rec["tau_f"].append(tf)
    if obj is not None:
        rec["obj_vel"].append(_t(obj.data.root_com_lin_vel_w[0]))
        rec["obj_pos"].append(_t(obj.data.root_com_pos_w[0]))
        if args.object_sensors:
            rec["obj_net"].append(_t(env.scene.sensors["vf_obj_net"].data.net_forces_w[0, 0]))
            oh = env.scene.sensors["vf_obj_hand"].data
            rec["obj_hand_n"].append(_t(oh.force_matrix_w[0, 0]).sum(0))
            rec["obj_hand_f"].append(_t(oh.friction_forces_w[0, 0]).sum(0))
        if obj_table_ok:
            ot = env.scene.sensors["vf_obj_table"].data
            rec["obj_table_n"].append(_t(ot.force_matrix_w[0, 0]).sum(0))
            rec["obj_table_f"].append(_t(ot.friction_forces_w[0, 0]).sum(0))


_orig_update = env.scene.update


def _hooked_update(dt):
    _orig_update(dt)
    if state["on"]:
        record_substep()


env.scene.update = _hooked_update

# ------------------------------------------------------------------ replay
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
mass = float(_t(obj.root_physx_view.get_masses()[0]).sum()) if obj is not None else float("nan")
print(f"[verify] object mass {mass}", flush=True)
actions = ep["actions"][: args.max_steps] if args.max_steps else ep["actions"]
images = []
with torch.inference_mode():
    for t, a in enumerate(actions):
        state["step"], state["on"] = t, True
        env.step(torch.as_tensor(np.asarray(a, dtype=np.float32).reshape(1, -1), device=env.device))
        state["on"] = False
        if t % 50 == 0:
            print(f"[verify] step {t}/{len(actions)}", flush=True)
        r = reader.read_torch()
        step_rec["pad_force_w"].append(_t(r["force_w"]))
        step_rec["pad_force"].append(_t(r["force"]))
        step_rec["pad_normal"].append(_t(r["normal"]))
        step_rec["pad_net"].append(_t(r["net_normal"]))
        if t in args.image_steps:
            import cv2  # noqa: PLC0415

            cam = env.scene["vf_cam"]
            for bname in args.image_body:
                bi = robot.body_names.index(bname)
                tgt = robot.data.body_link_pos_w[0, bi]
                ctr = obj.data.root_pos_w[0] if obj is not None else env.scene["articulation"].data.root_pos_w[0]
                d = torch.nn.functional.normalize(tgt - ctr, dim=0)
                for vi, (off, up) in enumerate(((0.10, 0.04), (0.07, -0.06))):
                    side = torch.nn.functional.normalize(torch.linalg.cross(d, torch.tensor([0.0, 0.0, 1.0])), dim=0)
                    eye = tgt + d * off + side * up + torch.tensor([0.0, 0.0, 0.03])
                    cam.set_world_poses_from_view(eye.unsqueeze(0), tgt.unsqueeze(0))
                    for _ in range(3):
                        env.sim.render()
                    cam.update(0.0, force_recompute=True)
                    img = _t(cam.data.output["rgb"][0, ..., :3]).astype(np.uint8)
                    path = os.path.join(args.out, f"img_step{t:04d}_{bname}_v{vi}.png")
                    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    images.append(path)

R = {k: np.asarray(v) for k, v in rec.items() if len(v)}
S = {k: np.asarray(v) for k, v in step_rec.items() if len(v)}
arm = _t(robot.data.joint_armature[0])
eff = _t(robot.data.joint_effort_limits[0]) if hasattr(robot.data, "joint_effort_limits") else np.full_like(arm, np.inf)
plim = _t(robot.data.joint_pos_limits[0])
vlim = _t(robot.data.joint_vel_limits[0]) if hasattr(robot.data, "joint_vel_limits") else np.full_like(arm, np.inf)
fric = _t(robot.data.joint_friction_coeff[0]) if hasattr(robot.data, "joint_friction_coeff") else np.zeros_like(arm)
np.savez(os.path.join(args.out, "verify.npz"), mass=mass, phys_dt=phys_dt, step_dt=env.step_dt, stiff=stiff, damp=damp,
         armature=arm, joint_friction=fric, effort_limit=eff, pos_limits=plim, vel_limits=vlim, grav_on=_grav_on, link_mass=_mass_b, link_inertia=_inert_b,
         anc_mat=np.array([[jinfo[c]["dof"] == d for d in range(robot.num_joints)] for c in []] or
                          [[any(jinfo[c]["dof"] == d for c in anc[n]) for d in range(robot.num_joints)]
                           for n in robot.body_names]),
         dof_kind=np.array([next((j["kind"] for j in jinfo.values() if j["dof"] == d), "none")
                            for d in range(robot.num_joints)]), body_names=np.array(robot.body_names),
         sens_names=np.array(sens_names), pad_bodies=np.array(tac["bodies"]), joint_names=np.array(robot.joint_names),
         **R, **{"S_" + k: v for k, v in S.items()})
print("[verify] normal-sign checks (sum f_i n_i . force_matrix / |fm|^2): n", len(sign_checks),
      "median", float(np.median(sign_checks)) if sign_checks else None,
      "min/max", (float(np.min(sign_checks)), float(np.max(sign_checks))) if sign_checks else None, flush=True)
print("[verify] kinematic check (pred link vel . measured / |measured|^2, rel err): n", len(jac_checks), "median",
      np.median(np.array(jac_checks), axis=0).tolist() if jac_checks else None, flush=True)
api_ok = {k: bool(np.isfinite(R[k]).all()) for k in ("proj", "gcomp", "ccomp", "Mqdd_M") if k in R}
print("[verify] PhysX API availability:", api_ok, "errors:", _api_err, flush=True)
print("[verify] images:", images, flush=True)
json.dump({"sign_checks_median": float(np.median(sign_checks)) if sign_checks else None,
           "kin_check_ratio_relerr": np.median(np.array(jac_checks), axis=0).tolist() if jac_checks else None,
           "api_available": api_ok,
           "fixed_base": fixed, "decimation": int(env.cfg.decimation), "obj_table_ok": obj_table_ok, "mass": mass,
           "images": images}, open(os.path.join(args.out, "run_info.json"), "w"), indent=1)
print("VERIFY_RUN_DONE", flush=True)
os._exit(0)
