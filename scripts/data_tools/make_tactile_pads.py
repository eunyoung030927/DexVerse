# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write the virtual fingerprint-pad definition of a hand: ``data_collection/tactile_pads/<robot_type>.npz``.

Spawns the hand (any task that accepts ``--robot_type``), reads its collision shapes and joints from the live stage
(physics is not stepped and nothing is modified) and, per finger:

* distal bundle = the distal phalanx D (the body whose incoming joint is the last revolute joint before the
  fingertip marker) + every link fixed to D (recursively);
* finger frame in D: o = anchor of D's revolute joint, a = its axis, u = direction from o to the centre of the
  bundle surface (orthogonalised to a), palmar h = s * (a x u) with s = +1 if the joint's upper limit is the larger
  one in magnitude (positive = flexion), w = u x h;
* surface samples of every bundle link's collision shapes (convex hull for convex approximations, the mesh
  otherwise), about one per (--spacing)^2, with outward normals;
* PAD = samples with  n . h >= cos(--max_angle)  (palmar-facing: back and sides excluded) and
  s >= s_max - L_pad  with  L_pad = clip(--pad_frac * L_bundle, --pad_min, --pad_max)  (from the base of the finger
  pad to the tip, the proximal / joint-crease part excluded), L_bundle = s_max - max(s_min, 0).

Extra masks: ``elastomer`` (Sharpa: every sample of the ``*_elastomer`` link = the physical pad body, used as ground
truth). A JSON summary and a PNG of every finger (side + palmar views) go next to the npz (``--out_dir`` for the
figures).

    PYTHONPATH=<worktree>/source/dexverse /workspace/isaaclab/_isaac_sim/python.sh scripts/data_tools/make_tactile_pads.py \
        --robot_type floating_allegro_right --headless --enable_cameras
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot_type", required=True)
parser.add_argument("--task", default="Dexverse-PickCube-v0")
parser.add_argument("--spacing", type=float, default=0.0006, help="surface sample spacing (m)")
parser.add_argument("--max_angle", type=float, default=60.0, help="max angle (deg) between pad normal and palmar dir")
parser.add_argument("--pad_frac", type=float, default=0.8)
parser.add_argument("--pad_min", type=float, default=0.012)
parser.add_argument("--pad_max", type=float, default=0.022)
parser.add_argument("--out_dir", default=None, help="folder for the PNG / JSON copy (default: next to the npz)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
app = AppLauncher(vars(args)).app


def _exit_on_error(exc_type, exc, tb):
    import traceback  # noqa: PLC0415

    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    os._exit(1)


sys.excepthook = _exit_on_error

import dexverse  # noqa: E402
import dexverse.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from dexverse.data_collection.tactile_pads import PAD_DIR, _qmat, pad_definition_path  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

print("DEXVERSE_FILE", dexverse.__file__, flush=True)
cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg = type(cfg)(robot_type=args.robot_type)
cfg.sim.device = args.device
cfg.scene.num_envs = 1
env = gym.make(args.task, cfg=cfg).unwrapped
env.sim.reset()
env.reset()
robot = env.scene["robot"]
rc = env.cfg.robot_config
stage = get_current_stage()
xc = UsdGeom.XformCache()
RP = "/World/envs/env_0/Robot"
bodies = list(robot.body_names)


# ------------------------------------------------------------------ joints (child body -> joint)
def _q(qf):
    return np.array([qf.GetReal(), *qf.GetImaginary()], dtype=np.float64)


joints = {}
for p in Usd.PrimRange(stage.GetPrimAtPath(RP), Usd.TraverseInstanceProxies()):
    if not p.IsA(UsdPhysics.Joint):
        continue
    j = UsdPhysics.Joint(p)
    b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
    if not b1 or b1[0].name not in bodies:
        continue
    info = {"name": p.GetName(), "parent": b0[0].name if b0 else None, "kind": "fixed"}
    if p.IsA(UsdPhysics.RevoluteJoint):
        r = UsdPhysics.RevoluteJoint(p)
        ax = {"X": 0, "Y": 1, "Z": 2}[r.GetAxisAttr().Get()]
        info.update({"kind": "revolute", "anchor": np.array(j.GetLocalPos1Attr().Get(), float),
                     "axis": _qmat(_q(j.GetLocalRot1Attr().Get()))[:, ax],
                     "lower": r.GetLowerLimitAttr().Get(), "upper": r.GetUpperLimitAttr().Get()})
    elif p.IsA(UsdPhysics.PrismaticJoint):
        info["kind"] = "prismatic"
    joints[b1[0].name] = info
children = {}
for c, info in joints.items():
    children.setdefault(info["parent"], []).append(c)


# ------------------------------------------------------------------ colliders (unscaled body frame)
def _gprim_mesh(p):
    if p.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(p)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
        cnt, idx = np.asarray(m.GetFaceVertexCountsAttr().Get()), np.asarray(m.GetFaceVertexIndicesAttr().Get())
        faces, k = [], 0
        for c in cnt:
            faces += [[idx[k], idx[k + j], idx[k + j + 1]] for j in range(1, c - 1)]
            k += c
        return trimesh.Trimesh(pts, np.asarray(faces), process=False)
    for typ, mk in ((UsdGeom.Sphere, lambda g: trimesh.creation.icosphere(3, radius=g.GetRadiusAttr().Get())),
                    (UsdGeom.Cube, lambda g: trimesh.creation.box(extents=[g.GetSizeAttr().Get()] * 3)),
                    (UsdGeom.Capsule, lambda g: trimesh.creation.capsule(height=g.GetHeightAttr().Get(),
                                                                         radius=g.GetRadiusAttr().Get())),
                    (UsdGeom.Cylinder, lambda g: trimesh.creation.cylinder(radius=g.GetRadiusAttr().Get(),
                                                                           height=g.GetHeightAttr().Get()))):
        if p.IsA(typ):
            g = typ(p)
            t = mk(g)
            if typ in (UsdGeom.Capsule, UsdGeom.Cylinder):
                axn = g.GetAxisAttr().Get()
                if axn == "X":
                    t.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
                elif axn == "Y":
                    t.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            return t
    return None


def body_meshes(name):
    body = stage.GetPrimAtPath(f"{RP}/{name}")
    bw = Gf.Transform(xc.GetLocalToWorldTransform(body))
    bw.SetScale(Gf.Vec3d(1, 1, 1))
    inv = bw.GetMatrix().GetInverse()
    out = []
    for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if p != body and p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if not p.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False:
            continue
        approx = str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()) if p.HasAPI(
            UsdPhysics.MeshCollisionAPI) else ("primitive" if not p.IsA(UsdGeom.Mesh) else "none")
        geoms = [p] if p.IsA(UsdGeom.Gprim) else [d for d in Usd.PrimRange(p, Usd.TraverseInstanceProxies())
                                                    if d.IsA(UsdGeom.Gprim)]   # CollisionAPI on an Xform
        for g in geoms:
            t = _gprim_mesh(g)
            if t is None:
                continue
            rel = xc.GetLocalToWorldTransform(g) * inv
            T = np.array([[rel[i][j] for j in range(4)] for i in range(4)]).T   # column-vector form
            t.apply_transform(T)
            if approx in ("convexHull", "boundingCube", "boundingSphere", "primitive", "convexDecomposition"):
                t = t.convex_hull if approx != "convexDecomposition" else t
            out.append((str(g.GetPath()), approx, t))
    return out


# ------------------------------------------------------------------ fingers
def distal_of(marker):
    b = marker
    while b in joints and joints[b]["kind"] == "fixed":
        b = joints[b]["parent"]
    if joints.get(b, {}).get("kind") != "revolute":
        raise RuntimeError(f"no revolute distal joint above {marker}")
    return b


def fixed_descendants(b):
    out = [b]
    for c in children.get(b, []):
        if joints[c]["kind"] == "fixed":
            out += fixed_descendants(c)
    return out


markers = list(rc.fingertip_body_names)
palms = [rc.right_palm_body_name] + ([rc.left_palm_body_name] if getattr(rc, "left_palm_body_name", None) else [])
palms = list(dict.fromkeys(p for p in palms if p))
pose = np.c_[robot.data.body_link_pos_w[0].cpu().numpy(), robot.data.body_link_quat_w[0].cpu().numpy()].astype(float)


def T_of(name):
    i = bodies.index(name)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _qmat(pose[i, 3:]), pose[i, :3]
    return T


ang_cos = np.cos(np.radians(args.max_angle))
fingers, distal, links, link_finger = [], [], [], []
per_link, summary = [], {"robot_type": args.robot_type, "params": vars(args) | {"enable_cameras": None}, "fingers": []}
rng = np.random.default_rng(0)
for fi, mk in enumerate(markers):
    D = distal_of(mk)
    bundle = [b for b in fixed_descendants(D) if body_meshes(b)]
    label = D
    TD_inv = np.linalg.inv(T_of(D))
    J = joints[D]
    o, a = J["anchor"], J["axis"] / np.linalg.norm(J["axis"])
    flex = 1.0 if abs(J["upper"]) >= abs(J["lower"]) else -1.0
    samp = {}
    for b in bundle:
        P, Nn, A = [], [], 0.0
        for path, approx, t in body_meshes(b):
            n = int(np.clip(t.area / args.spacing ** 2, 200, 12000))
            pts, fidx = trimesh.sample.sample_surface(t, n, seed=int(rng.integers(1 << 30)))
            P.append(pts)
            Nn.append(t.face_normals[fidx])
            A += t.area
        P, Nn = np.concatenate(P), np.concatenate(Nn)
        Tb = TD_inv @ T_of(b)
        samp[b] = (P, Nn, Tb, A)
    allD = np.concatenate([s[0] @ s[2][:3, :3].T + s[2][:3, 3] for s in samp.values()])
    u = allD.mean(0) - o
    u -= a * np.dot(u, a)
    u /= np.linalg.norm(u)
    h = flex * np.cross(a, u)
    w = np.cross(u, h)
    s_all = (allD - o) @ u
    s_max, s_min = s_all.max(), s_all.min()
    Lb = s_max - max(s_min, 0.0)
    Lpad = float(np.clip(args.pad_frac * Lb, args.pad_min, args.pad_max))
    finfo = {"finger": label, "marker": mk, "distal": D, "joint": J["name"], "limits_deg": [J["lower"], J["upper"]],
             "flex_sign": flex, "bundle": bundle, "L_bundle_mm": Lb * 1e3, "L_pad_mm": Lpad * 1e3,
             "frame_in_distal": {"o": o.tolist(), "u": u.tolist(), "h_palmar": h.tolist(), "a": a.tolist()}}
    pad_area = 0.0
    for b in bundle:
        P, Nn, Tb, A = samp[b]
        xD = P @ Tb[:3, :3].T + Tb[:3, 3]
        nD = Nn @ Tb[:3, :3].T
        s = (xD - o) @ u
        pad = (nD @ h >= ang_cos) & (s >= s_max - Lpad)
        pad75 = (nD @ h >= np.cos(np.radians(75.0))) & (s >= s_max - Lpad)   # sensitivity variant (wider angle)
        el = np.full(len(P), "elastomer" in b)
        per_link.append({"link": b, "finger": fi, "samples": P.astype(np.float32), "normals": Nn.astype(np.float32),
                         "pad": pad, "pad75": pad75, "elastomer": el, "xD": xD, "nD": nD})
        links.append(b)
        link_finger.append(fi)
        pad_area += A * pad.mean()
        finfo.setdefault("links", []).append({"link": b, "samples": int(len(P)), "area_mm2": A * 1e6,
                                              "pad_frac": float(pad.mean()), "pad_area_mm2": A * pad.mean() * 1e6})
    pd_ = np.concatenate([q["xD"][q["pad"]] for q in per_link if q["finger"] == fi])
    if len(pd_):
        finfo["pad_extent_mm"] = {"s": [float(((pd_ - o) @ u).min() * 1e3), float(((pd_ - o) @ u).max() * 1e3)],
                                  "w": [float(((pd_ - o) @ w).min() * 1e3), float(((pd_ - o) @ w).max() * 1e3)],
                                  "h": [float(((pd_ - o) @ h).min() * 1e3), float(((pd_ - o) @ h).max() * 1e3)]}
    finfo["pad_area_mm2"] = pad_area * 1e6
    fingers.append(label)
    distal.append(D)
    summary["fingers"].append(finfo)
    print(json.dumps({k: v for k, v in finfo.items() if k not in ("links", "frame_in_distal")}), flush=True)

for p in palms:
    if not body_meshes(p):
        print(f"[pads] WARNING palm {p} has no colliders", flush=True)
summary["palms"] = palms
os.makedirs(PAD_DIR, exist_ok=True)
arrs = {"fingers": np.array(fingers), "distal": np.array(distal), "links": np.array(links),
        "link_finger": np.array(link_finger), "palms": np.array(palms)}
for i, q in enumerate(per_link):
    arrs[f"samples_{i}"] = q["samples"]
    arrs[f"normals_{i}"] = q["normals"]
    arrs[f"mask_pad_{i}"] = q["pad"]
    arrs[f"mask_pad75_{i}"] = q["pad75"]
    if any("elastomer" in x for x in links):
        arrs[f"mask_elastomer_{i}"] = q["elastomer"]
for i, f in enumerate(summary["fingers"]):
    fr = f["frame_in_distal"]
    arrs[f"frame_{i}"] = np.array([fr["o"], fr["u"], fr["h_palmar"], fr["a"]])
np.savez_compressed(pad_definition_path(args.robot_type), **arrs)
with open(pad_definition_path(args.robot_type).replace(".npz", ".json"), "w") as fh:
    json.dump(summary, fh, indent=1, default=float)

# ------------------------------------------------------------------ figure
out_dir = args.out_dir or PAD_DIR
os.makedirs(out_dir, exist_ok=True)
nf = len(fingers)
fig, axs = plt.subplots(nf, 2, figsize=(9, 3.2 * nf), squeeze=False)
for fi in range(nf):
    fr = summary["fingers"][fi]["frame_in_distal"]
    o, u, h = np.array(fr["o"]), np.array(fr["u"]), np.array(fr["h_palmar"])
    w = np.cross(u, h)
    for q in [q for q in per_link if q["finger"] == fi]:
        s, hh, ww = (q["xD"] - o) @ u * 1e3, (q["xD"] - o) @ h * 1e3, (q["xD"] - o) @ w * 1e3
        for ax, y in ((axs[fi, 0], hh), (axs[fi, 1], ww)):
            sub = rng.choice(len(s), min(len(s), 4000), replace=False)
            c = np.where(q["pad"][sub], "tab:red", "lightgray")
            ax.scatter(s[sub], y[sub], s=1.5, c=c, marker="o" if "elastomer" not in q["link"] and "tip" not in q["link"] else "x")
    axs[fi, 0].set_title(f"{fingers[fi]}: side view (s along finger, h palmar up)", fontsize=8)
    axs[fi, 1].set_title(f"{fingers[fi]}: palmar view (s, w lateral); red = pad", fontsize=8)
    for ax in axs[fi]:
        ax.set_aspect("equal")
        ax.set_xlabel("s [mm]", fontsize=7)
        ax.tick_params(labelsize=6)
    axs[fi, 0].set_ylabel("h [mm]", fontsize=7)
    axs[fi, 1].set_ylabel("w [mm]", fontsize=7)
fig.suptitle(f"{args.robot_type} virtual fingerprint pads (o = distal joint anchor)", fontsize=9)
fig.tight_layout()
png = os.path.join(out_dir, f"{args.robot_type}_pads.png")
fig.savefig(png, dpi=130)
print("[pads] wrote", pad_definition_path(args.robot_type), "and", png, flush=True)
print("PADS_DONE", flush=True)
os._exit(0)
