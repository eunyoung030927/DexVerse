# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Raw per-contact data (points / normals / separations / forces) of hand-object contacts (experiment, not a tool).

Replays one episode of a trajectory pickle and, after EVERY physics substep, reads the PhysX RigidContactView of a
ContactSensor over all hand links (filtered against the task's target) the careful way:

* ``get_contact_data`` and ``get_friction_data`` return PERSISTENT buffers, and both write the SAME
  ``_contact_count_buffer`` / ``_start_indices_buffer`` attributes of the view -> everything is deep-copied right
  after each call (the diagnostics below record what goes wrong otherwise);
* per sensor-filter pair the contact rows are ``[start, start + count)`` of the contact call's own count/start.

It stores the raw rows, the aggregated sensor outputs (force matrix, friction, mean contact point), all body poses
and the collision geometry of every robot link and of the target (body-local frames) for
``tactile_rawcontact_analyze.py``.

    PYTHONPATH=<worktree>/source/dexverse /workspace/isaaclab/_isaac_sim/python.sh \
        scripts/experiments/tactile_rawcontact_probe.py --task Dexverse-PickCube-v0 \
        --robot_type floating_allegro_right --pickle demo.pkl --out <dir> --headless --enable_cameras
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--robot_type", required=True)
parser.add_argument("--pickle", required=True)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--per_prim", type=int, default=64, help="max_contact_data_count_per_prim of the probe sensor")
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
import trimesh  # noqa: E402
from dexverse.data_collection.tactile import TactileReader, add_tactile_sensors, articulation_link_prims  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from dexverse.teleop_utils.episode_task_state import reset_managers_after_state_restore  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

print("DEXVERSE_FILE", dexverse.__file__, flush=True)
os.makedirs(args.out, exist_ok=True)


def C(x):
    """Deep copy to numpy (CPU tensors from PhysX / Isaac Lab alias persistent buffers)."""
    return np.array(x.detach().to("cpu").numpy() if hasattr(x, "detach") else x, copy=True)


# ------------------------------------------------------------------ env
cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg = type(cfg)(robot_type=args.robot_type)
cfg.sim.device = args.device
cfg.scene.num_envs = 1
cfg.terminations.success = None
cfg.terminations.time_out = None
cfg.recorders = {}
tac = add_tactile_sensors(cfg)
targets = tac["targets"]
cfg.scene.rc_all = ContactSensorCfg(prim_path=f"{cfg.scene.robot.prim_path}/.*", filter_prim_paths_expr=list(targets),
                                    track_friction_forces=True, track_contact_points=True,
                                    max_contact_data_count_per_prim=args.per_prim)
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
reader = TactileReader(env, tac["bodies"], tac["targets"])
sens = env.scene.sensors["rc_all"]
view = sens.contact_physx_view
names = list(sens.body_names)
B, M = len(names), view.filter_count
assert M == len(targets)
cap = int(view.max_contact_data_count)
dt = env.physics_dt
print(f"[probe] sensor bodies {B}, filters {M}, buffer capacity {cap}, physics dt {dt}", flush=True)

# target bodies (rigid object root or articulation links)
obj = env.scene["object"] if getattr(cfg.scene, "object", None) is not None else None
art = env.scene["articulation"] if obj is None and getattr(cfg.scene, "articulation", None) is not None else None
tgt_names = [t.rsplit("/", 1)[-1] for t in targets]

# ------------------------------------------------------------------ collision geometry (body-local frames)
stage = get_current_stage()
xc = UsdGeom.XformCache()


def _mesh_from_prim(p):
    if p.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(p)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
        cnt = np.asarray(m.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
        faces, k = [], 0
        for c in cnt:
            for j in range(1, c - 1):
                faces.append([idx[k], idx[k + j], idx[k + j + 1]])
            k += c
        return pts, np.asarray(faces, dtype=np.int64)
    if p.IsA(UsdGeom.Sphere):
        t = trimesh.creation.icosphere(subdivisions=3, radius=UsdGeom.Sphere(p).GetRadiusAttr().Get())
    elif p.IsA(UsdGeom.Cube):
        s = UsdGeom.Cube(p).GetSizeAttr().Get()
        t = trimesh.creation.box(extents=(s, s, s))
    elif p.IsA(UsdGeom.Capsule):
        g = UsdGeom.Capsule(p)
        t = trimesh.creation.capsule(height=g.GetHeightAttr().Get(), radius=g.GetRadiusAttr().Get())
        ax = g.GetAxisAttr().Get()
        if ax == "X":
            t.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        elif ax == "Y":
            t.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif p.IsA(UsdGeom.Cylinder):
        g = UsdGeom.Cylinder(p)
        t = trimesh.creation.cylinder(radius=g.GetRadiusAttr().Get(), height=g.GetHeightAttr().Get())
        ax = g.GetAxisAttr().Get()
        if ax == "X":
            t.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        elif ax == "Y":
            t.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    else:
        return None, None
    return np.asarray(t.vertices), np.asarray(t.faces)


def body_colliders(body_path: str):
    body = stage.GetPrimAtPath(body_path)
    if not body.IsValid():
        return []
    bw = Gf.Transform(xc.GetLocalToWorldTransform(body))
    scale = tuple(bw.GetScale())
    if max(abs(x - 1.0) for x in scale) > 1e-6:
        print(f"[probe] body {body_path} has scale {scale}: colliders are expressed in its UNSCALED frame "
              "(= the PhysX body frame)", flush=True)
    bw.SetScale(Gf.Vec3d(1.0, 1.0, 1.0))
    inv = bw.GetMatrix().GetInverse()
    out = []
    for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if p != body and p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue  # nested body
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        en = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
        if en is False:
            continue
        v, f = _mesh_from_prim(p)
        if v is None:
            print(f"[probe] unsupported collider type {p.GetTypeName()} at {p.GetPath()}", flush=True)
            continue
        rel = xc.GetLocalToWorldTransform(p) * inv  # row-vector convention: x_body = x_local * rel
        R = np.array([[rel[i][j] for j in range(4)] for i in range(4)])
        vb = np.c_[v, np.ones(len(v))] @ R
        approx = "none"
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            approx = str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get())
        elif not p.IsA(UsdGeom.Mesh):
            approx = "primitive:" + p.GetTypeName()
        out.append({"prim": str(p.GetPath()), "approx": approx, "verts": vb[:, :3].astype(np.float32),
                    "faces": f.astype(np.int32)})
    return out


rp = "/World/envs/env_0/Robot"
geom = {"robot": {n: body_colliders(f"{rp}/{n}") for n in robot.body_names}, "target": {}}
for t in targets:
    geom["target"][t.rsplit("/", 1)[-1]] = body_colliders(t.replace("{ENV_REGEX_NS}", "/World/envs/env_0"))
nr = sum(len(v) for v in geom["robot"].values())
print(f"[probe] colliders: robot {nr}, target {[(k, len(v), [c['approx'] for c in v]) for k, v in geom['target'].items()]}",
      flush=True)

# ------------------------------------------------------------------ per-substep raw reader
REC = {"crow": [], "frow": [], "fm": [], "ffm": [], "cpos": [], "body_pose": [], "tgt_pose": [], "sub": [], "step": []}
DIAG = {"sat": [], "repeat_equal": [], "count_overwritten": [], "start_overwritten": [], "old_ratio": [], "n_contacts": [],
        "n_friction": []}
state = {"step": -1, "on": False, "k": 0}


def tgt_pose():
    if obj is not None:
        return np.r_[C(obj.data.root_link_pos_w[0]), C(obj.data.root_link_quat_w[0])][None]
    out = []
    for n in tgt_names:
        i = art.body_names.index(n)
        out.append(np.r_[C(art.data.body_link_pos_w[0, i]), C(art.data.body_link_quat_w[0, i])])
    return np.asarray(out)


def record():
    k = state["k"]
    view = sens.contact_physx_view   # re-fetch: env.sim.reset() re-creates the physics views
    ret = view.get_contact_data(dt=dt)
    f, p, n, s, c, st = (C(x) for x in ret)          # copies, taken right after the call
    c_ref, st_ref = ret[4], ret[5]                   # live buffer references (aliasing test)
    f2, p2, _n2, _s2, c2, st2 = (C(x) for x in view.get_contact_data(dt=dt))
    DIAG["repeat_equal"].append(bool(np.array_equal(f, f2) and np.array_equal(p, p2) and np.array_equal(c, c2)))
    fr, fp, fc, fst = (C(x) for x in view.get_friction_data(dt=dt))
    # did the friction call overwrite the contact call's count / start buffers?
    DIAG["count_overwritten"].append(int((C(c_ref) != c).sum()))
    DIAG["start_overwritten"].append(int((C(st_ref) != st).sum()))
    f, n, s = f.reshape(-1), n.reshape(-1, 3), s.reshape(-1)
    p, fr, fp = p.reshape(-1, 3), fr.reshape(-1, 3), fp.reshape(-1, 3)
    c, st, fc, fst = c.reshape(B, M), st.reshape(B, M), fc.reshape(B, M), fst.reshape(B, M)
    DIAG["sat"].append(int(c.sum()) >= cap or int(fc.sum()) >= cap)
    DIAG["n_contacts"].append(int(c.sum()))
    DIAG["n_friction"].append(int(fc.sum()))
    fm = C(sens.data.force_matrix_w[0])
    ffm = np.nan_to_num(C(sens.data.friction_forces_w[0]))
    cpos = C(sens.data.contact_pos_w[0])
    rows, frows = [], []
    for b in range(B):
        for m in range(M):
            ci, si = int(c[b, m]), int(st[b, m])
            for i in range(si, si + ci):
                rows.append([k, b, m, *p[i], *n[i], s[i], f[i]])
            # the earlier (buggy) parse: contact rows sliced with the FRICTION call's count/start
            if np.linalg.norm(fm[b, m]) > 1e-3 and int(fc[b, m]) > 0:
                fi, fsi = int(fc[b, m]), int(fst[b, m])
                old = (f[fsi:fsi + fi, None] * n[fsi:fsi + fi]).sum(0)
                DIAG["old_ratio"].append(float(np.dot(old, fm[b, m]) / np.dot(fm[b, m], fm[b, m])))
            ci, si = int(fc[b, m]), int(fst[b, m])
            for i in range(si, si + ci):
                frows.append([k, b, m, *fp[i], *fr[i]])
    REC["crow"].extend(rows)
    REC["frow"].extend(frows)
    REC["fm"].append(fm.astype(np.float32))
    REC["ffm"].append(ffm.astype(np.float32))
    REC["cpos"].append(cpos.astype(np.float32))
    REC["body_pose"].append(np.c_[C(robot.data.body_link_pos_w[0]), C(robot.data.body_link_quat_w[0])])
    REC["tgt_pose"].append(tgt_pose())
    REC["sub"].append(k)
    REC["step"].append(state["step"])
    state["k"] += 1


_orig_update = env.scene.update


def _hooked_update(dt):
    _orig_update(dt)
    if state["on"]:
        record()


env.scene.update = _hooked_update

# ------------------------------------------------------------------ replay
with open(args.pickle, "rb") as fh:
    ep = [e for e in pickle.load(fh)["episodes"] if len(e.get("actions", []))][args.episode]


def to_t(x):
    if isinstance(x, dict):
        return {kk: to_t(v) for kk, v in x.items()}
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=env.device)
    return x


env.sim.reset()
env.reset()
env.reset_to(to_t(ep["initial_state"]), env_ids=None, is_relative=True)
reset_managers_after_state_restore(env)
_v = sens.contact_physx_view
FILTER_PATHS = [list(x) for x in _v.filter_paths]
SENSOR_PATHS = list(_v.sensor_paths)
print(f"[probe] filter paths of sensor 0 (column order of the force matrix / count buffers): {FILTER_PATHS[0]}", flush=True)
print(f"[probe] all sensors share that filter order: {all(fp == FILTER_PATHS[0] for fp in FILTER_PATHS)}", flush=True)
actions = ep["actions"][: args.max_steps] if args.max_steps else ep["actions"]
pad_w = []
with torch.inference_mode():
    for t, a in enumerate(actions):
        state["step"], state["on"] = t, True
        env.step(torch.as_tensor(np.asarray(a, dtype=np.float32).reshape(1, -1), device=env.device))
        state["on"] = False
        pad_w.append(C(reader.read_torch()["force_w"]))
        if t % 100 == 0:
            print(f"[probe] step {t}/{len(actions)}", flush=True)

np.savez_compressed(
    os.path.join(args.out, "raw.npz"),
    crow=np.asarray(REC["crow"], dtype=np.float64).reshape(-1, 11),
    frow=np.asarray(REC["frow"], dtype=np.float64).reshape(-1, 9),
    fm=np.asarray(REC["fm"]), ffm=np.asarray(REC["ffm"]), cpos=np.asarray(REC["cpos"]),
    body_pose=np.asarray(REC["body_pose"]), tgt_pose=np.asarray(REC["tgt_pose"]), sub=np.asarray(REC["sub"]),
    step=np.asarray(REC["step"]), pad_w=np.asarray(pad_w), sens_names=np.array(names),
    robot_body_names=np.array(robot.body_names), tgt_names=np.array(tgt_names), pad_bodies=np.array(tac["bodies"]),
    dt=dt, cap=cap, filter_paths=np.array(FILTER_PATHS), sensor_paths=np.array(SENSOR_PATHS), **{"d_" + k: np.asarray(v) for k, v in DIAG.items()})
with open(os.path.join(args.out, "geom.pkl"), "wb") as fh:
    pickle.dump(geom, fh)
d = {k: np.asarray(v) for k, v in DIAG.items()}
print(f"[probe] substeps {len(REC['sub'])}, contact rows {len(REC['crow'])}, friction rows {len(REC['frow'])}", flush=True)
print(f"[probe] buffer saturated substeps: {int(d['sat'].sum())}; max contacts in a substep {d['n_contacts'].max()} "
      f"(capacity {cap})", flush=True)
print(f"[probe] repeated get_contact_data identical: {int(d['repeat_equal'].sum())}/{len(d['repeat_equal'])}", flush=True)
print(f"[probe] friction call overwrote contact count buffer in {int((d['count_overwritten'] > 0).sum())} substeps, "
      f"start buffer in {int((d['start_overwritten'] > 0).sum())}", flush=True)
if len(d["old_ratio"]):
    print(f"[probe] OLD parse ratio (contact rows sliced with friction count/start): median {np.median(d['old_ratio']):.3f}, "
          f"p5/p95 {np.percentile(d['old_ratio'], 5):.3f}/{np.percentile(d['old_ratio'], 95):.3f}", flush=True)
print("PROBE_DONE", flush=True)
os._exit(0)
