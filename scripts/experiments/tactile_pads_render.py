# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Picture of the virtual tactile pads ON the hand (experiment): every collision shape of the hand at a lightly
curled pose, shaded grey, with the pad definition (``tactile_pads/<robot_type>.npz``) painted on it -- fingertip
pads red, palm zones palm_l / palm_c / palm_r red / green / blue. Palmar and side views per hand.

Spawns the hand like make_tactile_pads.py (physics is not stepped; the joints are written once).

    python scripts/experiments/tactile_pads_render.py --robot_type floating_allegro_bimanual --out <png> --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot_type", required=True)
parser.add_argument("--task", default="Dexverse-PickCube-v0")
parser.add_argument("--out", required=True)
parser.add_argument("--curl", type=float, default=0.25, help="fraction of each finger joint's range toward flexion")
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

import dexverse.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from dexverse.data_collection.tactile_pads import _qmat, pad_definition_path  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg = type(cfg)(robot_type=args.robot_type)
cfg.sim.device = args.device
cfg.scene.num_envs = 1
env = gym.make(args.task, cfg=cfg).unwrapped
env.sim.reset()
env.reset()
robot = env.scene["robot"]
stage = get_current_stage()
RP = "/World/envs/env_0/Robot"
bodies = list(robot.body_names)
D = np.load(pad_definition_path(args.robot_type))

# lightly curled fingers (virtual wrist joints untouched), written to the sim and read back as link poses
q = robot.data.default_joint_pos.clone()
lo, hi = robot.data.soft_joint_pos_limits[0, :, 0], robot.data.soft_joint_pos_limits[0, :, 1]
virt = [i for i, n in enumerate(robot.joint_names) if any(k in n for k in ("translation", "rotation", "_x_joint",
                                                                            "_y_joint", "_z_joint", "roll", "pitch",
                                                                            "yaw"))]
fing = [i for i in range(len(robot.joint_names)) if i not in virt]
big = torch.where(hi[fing].abs() >= lo[fing].abs(), hi[fing], lo[fing])
q[0, fing] = torch.clamp(args.curl * big, lo[fing], hi[fing])
robot.write_joint_state_to_sim(q, torch.zeros_like(q))
env.sim.forward()
env.scene.update(0.0)
pose = np.c_[robot.data.body_link_pos_w[0].cpu().numpy(), robot.data.body_link_quat_w[0].cpu().numpy()].astype(float)


def T_world(name):
    i = bodies.index(name)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _qmat(pose[i, 3:]), pose[i, :3]
    return T


xc = UsdGeom.XformCache()


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
    for typ, mk in ((UsdGeom.Sphere, lambda g: trimesh.creation.icosphere(2, radius=g.GetRadiusAttr().Get())),
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
    """Collision shapes of a body in its unscaled body frame (as make_tactile_pads.py)."""
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
                                                    if d.IsA(UsdGeom.Gprim)]
        for g in geoms:
            t = _gprim_mesh(g)
            if t is None:
                continue
            rel = xc.GetLocalToWorldTransform(g) * inv
            t.apply_transform(np.array([[rel[i][j] for j in range(4)] for i in range(4)]).T)
            if approx in ("convexHull", "boundingCube", "boundingSphere", "primitive"):
                t = t.convex_hull
            out.append(t)
    return out


def tf(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


# ------------------------------------------------------------------ geometry per hand
names = [str(x) for x in D["hand_names"]] if "hand_names" in D.files else ["right"]
palms = [str(x) for x in D["palms"]]
links = [str(x) for x in D["links"]]
link_finger = D["link_finger"]
finger_hand = D["finger_hand"] if "finger_hand" in D.files else np.zeros(len(D["fingers"]), int)
palm_links = [str(x) for x in D["palm_links"]] if "palm_links" in D.files else []
ZC = {0: (0.78, 0.78, 0.78), 1: (0.84, 0.15, 0.16), 2: (0.17, 0.63, 0.17), 3: (0.12, 0.47, 0.71)}


def hand_of_body(b):
    """Index of the palm a body hangs from (walk the parent joints), or None."""
    parents = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(RP), Usd.TraverseInstanceProxies()):
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if b1 and b0:
                parents[b1[0].name] = b0[0].name
    side = None
    for k, n in enumerate(names):
        tag = {"right": ("right", "rh_"), "left": ("left", "lh_")}[n]
        if b.lower().startswith(tag) or tag[0] in b.lower():
            side = k
    if side is None and len(names) == 1:
        side = 0
    return side


panels = []
for h, hname in enumerate(names):
    tris, cols = [], []
    for b in bodies:
        if any(k in b for k in ("translation", "rotation")) or hand_of_body(b) != h:
            continue
        T = T_world(b)
        for m in body_meshes(b):
            v = tf(T, np.asarray(m.vertices))
            tris.append(v[np.asarray(m.faces)])
    pad_pts, pal_pts, pal_col = [], [], []
    for li, ln in enumerate(links):
        f = int(link_finger[li])
        if int(finger_hand[f]) != h or f"samples_{li}" not in D.files:
            continue
        s, m = D[f"samples_{li}"], D[f"mask_pad_{li}"]
        pad_pts.append(tf(T_world(ln), s[m]))
    for j, ln in enumerate(palm_links):
        if int(D["palm_link_palm"][j]) != h:
            continue
        # palm samples live in the frame of their own link (palm_links[j]; palm_link_palm = its palm)
        s, z = D[f"palm_samples_{j}"], D[f"palm_zone_{j}"]
        keep = z > 0
        pal_pts.append(tf(T_world(ln), s[keep]))
        pal_col += [ZC[int(x)] for x in z[keep]]
    pad = np.concatenate(pad_pts) if pad_pts else np.zeros((0, 3))
    pal = np.concatenate(pal_pts) if pal_pts else np.zeros((0, 3))
    T = T_world(palms[h])
    frame = D[f"palm_frame_{h}"] if f"palm_frame_{h}" in D.files else None
    panels.append((hname, tris, pad, pal, np.array(pal_col).reshape(-1, 3), T, frame))


def view_dirs(T, frame):
    """World directions of the palmar normal h and the thumb-side axis t (palm frame rows: centre, u, h, t)."""
    h, t = (T[:3, :3] @ np.asarray(frame[2]), T[:3, :3] @ np.asarray(frame[3])) if frame is not None else \
        (T[:3, :3][:, 2], T[:3, :3][:, 1])
    return h / np.linalg.norm(h), t / np.linalg.norm(t)


fig = plt.figure(figsize=(7 * len(panels), 12))
rng = np.random.default_rng(0)
for c, (hname, tris, pad, pal, palc, T, frame) in enumerate(panels):
    allv = np.concatenate([t.reshape(-1, 3) for t in tris])
    ctr = (allv.max(0) + allv.min(0)) / 2
    rad = (allv.max(0) - allv.min(0)).max() / 2
    hdir, tdir = view_dirs(T, frame)
    for r, (title, d) in enumerate((("palmar view", hdir), ("thumb-side view", 0.6 * tdir + 0.8 * hdir))):
        ax = fig.add_subplot(2, len(panels), r * len(panels) + c + 1, projection="3d")
        d = d / np.linalg.norm(d)
        az = np.degrees(np.arctan2(d[1], d[0]))
        el = np.degrees(np.arcsin(np.clip(d[2], -1, 1)))
        light = d
        polys = np.concatenate(tris)
        n = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.45 + 0.5 * np.abs(n @ light)
        pc = Poly3DCollection(polys, facecolors=np.c_[np.repeat(shade[:, None], 3, 1) * 0.85, np.full(len(shade), 0.55)],
                              edgecolors="none")
        ax.add_collection3d(pc)
        # points on the surface facing the camera only (cheap visibility: normal-free, offset toward the camera)
        for pts, col, sz in ((pad, (0.86, 0.1, 0.12), 2.0), (pal, palc, 1.6)):
            if len(pts) == 0:
                continue
            sel = rng.choice(len(pts), min(len(pts), 6000), replace=False)
            p = pts[sel] + 0.0015 * d
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=sz, c=col if isinstance(col, tuple) else col[sel], depthshade=False)
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=el, azim=az)
        ax.set_axis_off()
        ax.set_title(f"{hname} hand - {title}", fontsize=11)
fig.suptitle(f"{args.robot_type}: tactile pads on the hand collision shapes\n"
             "red = fingertip pads; palm zones red palm_l (thumb side) / green palm_c / blue palm_r (little side)",
             fontsize=12)
fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fig.savefig(args.out, dpi=110)
print("[render] wrote", args.out, flush=True)
env.close()
app.close()
