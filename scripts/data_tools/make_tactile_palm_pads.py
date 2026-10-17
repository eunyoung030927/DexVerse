# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Add the virtual PALM pads (3 zones like the RH5DG2 palm sensor) and the real-hand touch layout to an existing
pad definition ``data_collection/tactile_pads/<robot_type>.npz`` (written by make_tactile_pads.py; the finger pad
arrays are copied unchanged).

Spawns the hand, reads collision shapes / joints / the reset pose from the live stage (physics is not stepped and
nothing is modified) and, per palm body P of the definition:

* palm bundle = every body rigidly connected to P (fixed joints, up and down; e.g. Allegro ``palm`` + its parent
  ``base_link``) that has colliders and is not a finger-pad link;
* palm frame in P (at the reset pose): palmar normal h = mean palmar direction of the non-thumb fingers' distal
  frames (their ``frame_i`` h) expressed in P; distal axis u = from the centroid of the bundle surface toward the
  mean base (first link below the palm) of the non-thumb fingers, orthogonalised to h; thumb-side axis t = from the
  little-most to the index finger base, orthogonalised to h and u (anatomical: the same rule for a left hand);
* palmar face = bundle surface samples with  n . h >= cos(--max_angle)  (back and sides excluded) inside the palm
  band along u: from the finger bases (median of the non-thumb bases, + 10 mm) back by --band_factor x the lateral
  span of the non-thumb bases (wrist / forearm parts of the palm body excluded);
* lateral range: the non-thumb finger-base span widened by half the mean base spacing on each side (the RH5DG2 palm
  sensor spans the index..little bases);
* reference pose: every joint at 0 clipped to its limits (kinematics only);
* zones: the palmar face's extent along t cut into three equal widths: palm_l (thumb side, zone 1), palm_c
  (zone 2), palm_r (little-finger side, zone 3); every other sample = non-pad (zone 0).

Touch layout per finger: canonical pad name (little / ring / middle / index / thumb, from the link-name table
below; checked against the base order along t), its hand (palm it hangs from), and the pad axes in the distal frame
[u (distal), h (palmar normal), w (thumb side)] with w = +-(u x h): sign toward the palm's t for non-thumb fingers;
for the thumb toward the palm's distal axis u (fallback: the palmar normal) -- recorded as ``fty_rule``.

Reference (RH5DG2 real hand, read-only from the ro1 URDF rh5dg2_right.urdf / robosynapse profile): ONE palm sensor
link ``right_palm_force_sensor`` (mesh about 61 x 28 mm, centred on the palm midline about 40 mm proximal of the
finger MCP joints) read out as three zones palm_r / palm_c / palm_l; the robosynapse touch_layout puts palm_l next
to the thumb (facing view). Its zone boundaries are not published, so the zones here are equal thirds of the
simulated hand's palmar face; RH5DG2 numbers are stored in the json for comparison.

    PYTHONPATH=<worktree>/source/dexverse /workspace/isaaclab/isaaclab.sh -p scripts/data_tools/make_tactile_palm_pads.py \
        --robot_type floating_shadow_right --headless --out_dir <dir>
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot_type", required=True)
parser.add_argument("--task", default="Dexverse-PickCube-v0")
parser.add_argument("--spacing", type=float, default=0.0012, help="palm surface sample spacing (m)")
parser.add_argument("--max_angle", type=float, default=60.0, help="max angle (deg) between face normal and palmar normal")
parser.add_argument("--band_factor", type=float, default=1.5,
                    help="palm band length below the finger bases, in lateral finger-base spans")
parser.add_argument("--out_dir", default=None, help="folder for the PNG (default: next to the npz)")
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
from dexverse.data_collection.tactile_pads import PAD_DIR, PALM_ZONES, TOUCH_PADS, _qmat, pad_definition_path  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg, prune_stale_obs_refs, strip_camera_cfgs  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

print("DEXVERSE_FILE", dexverse.__file__, flush=True)
cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg = type(cfg)(robot_type=args.robot_type)
cfg.sim.device = args.device
cfg.scene.num_envs = 1
if not args.enable_cameras:
    cfg = prune_stale_obs_refs(strip_camera_cfgs(cfg))
env = gym.make(args.task, cfg=cfg).unwrapped
env.sim.reset()
env.reset()
robot = env.scene["robot"]
# reference pose: every joint at 0 clipped to its limits (open hand) so the fingers' palmar directions agree with
# the palm's; only kinematics are refreshed (no physics step)
import torch  # noqa: E402

lim = robot.data.joint_pos_limits[0]
q0 = torch.clamp(torch.zeros_like(robot.data.joint_pos[0]), lim[:, 0], lim[:, 1])[None]
robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
env.scene.write_data_to_sim()
env.sim.forward()
env.scene.update(dt=env.physics_dt)
stage = get_current_stage()
xc = UsdGeom.XformCache()
RP = "/World/envs/env_0/Robot"
bodies = list(robot.body_names)
pose = np.c_[robot.data.body_link_pos_w[0].cpu().numpy(), robot.data.body_link_quat_w[0].cpu().numpy()].astype(float)
NPZ = pad_definition_path(args.robot_type)
Z0 = dict(np.load(NPZ))
for k in [k for k in Z0 if k.startswith(("palm_samples_", "palm_normals_", "palm_zone_", "palm_frame_", "touch_axes_",
                                          "palm_link_T_"))
          or k in ("palm_links", "palm_link_palm", "finger_canon", "finger_hand", "hand_names")]:
    del Z0[k]                                   # regenerate palm / touch entries only
fingers = [str(x) for x in Z0["fingers"]]
distal = [str(x) for x in Z0["distal"]]
pad_links = set(str(x) for x in Z0["links"])
palms = [str(x) for x in Z0["palms"]]


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
    joints[b1[0].name] = {"name": p.GetName(), "parent": b0[0].name if b0 else None,
                          "kind": "revolute" if p.IsA(UsdPhysics.RevoluteJoint) else (
                              "prismatic" if p.IsA(UsdPhysics.PrismaticJoint) else "fixed")}
children = {}
for c, info in joints.items():
    children.setdefault(info["parent"], []).append(c)


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
                                                    if d.IsA(UsdGeom.Gprim)]
        for g in geoms:
            t = _gprim_mesh(g)
            if t is None:
                continue
            rel = xc.GetLocalToWorldTransform(g) * inv
            T = np.array([[rel[i][j] for j in range(4)] for i in range(4)]).T
            t.apply_transform(T)
            if approx in ("convexHull", "boundingCube", "boundingSphere", "primitive"):
                t = t.convex_hull
            out.append((str(g.GetPath()), approx, t))
    return out


def fixed_descendants(b):
    out = [b]
    for c in children.get(b, []):
        if joints[c]["kind"] == "fixed":
            out += fixed_descendants(c)
    return out


def T_of(name):
    i = bodies.index(name)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _qmat(pose[i, 3:]), pose[i, :3]
    return T


def canon(name):
    n = name.lower()
    for side in ("right_", "left_", "rh_", "lh_"):   # rh_ / lh_: Shadow bimanual
        if n.startswith(side):
            n = n[len(side):]
    table = {"link_3_0": "index", "link_7_0": "middle", "link_11_0": "ring", "link_15_0": "thumb",
             "thdistal": "thumb", "ffdistal": "index", "mfdistal": "middle", "rfdistal": "ring", "lfdistal": "little",
             "finger1_link4": "thumb", "finger2_link4": "index", "finger3_link4": "middle", "finger4_link4": "ring",
             "finger5_link4": "little", "thumb_distal": "thumb", "index_intermediate": "index",
             "middle_intermediate": "middle", "ring_intermediate": "ring", "pinky_intermediate": "little",
             "thumb_fingertip": "thumb", "fingertip": "index", "fingertip_2": "middle", "fingertip_3": "ring",
             "thumb_dp": "thumb", "index_dp": "index", "middle_dp": "middle", "ring_dp": "ring", "pinky_dp": "little"}
    if n not in table:
        raise RuntimeError(f"[palm] no canonical finger name for {name}")
    return table[n]


RAW_DIR_NOTE = ("raw direction conventions (robosynapse profile.py touch_layout, back-computed from a DISPLAY "
                "calibration, to be re-verified on the hardware): fingertip pads 0 deg = distal, counter-clockwise "
                "toward the thumb side; palm_l / palm_c 0 deg = thumb side, clockwise toward distal; palm_r = the "
                "same with only the distal-proximal axis mirrored (0 deg = thumb side, 90 deg = proximal), not a "
                "180 deg rotation")
rng = np.random.default_rng(0)
cos_m = np.cos(np.radians(args.max_angle))
finger_canon = [canon(f) for f in fingers]


def fixed_group(P):
    """bodies rigidly connected to P (fixed joints up and down), e.g. Allegro 'palm' marker + its parent base_link"""
    top = P
    while top in joints and joints[top]["kind"] == "fixed" and joints[top]["parent"] in bodies:
        top = joints[top]["parent"]
    return fixed_descendants(top)


palm_group = {P: fixed_group(P) for P in palms}


def hand_of(link):
    b, chain = link, [link]
    while b is not None and not any(b in g for g in palm_group.values()):
        b = joints.get(b, {}).get("parent")
        chain.append(b)
    if b is None:
        raise RuntimeError(f"[palm] {link} does not hang from a palm {palms}: chain {chain}")
    return [i for i, P in enumerate(palms) if b in palm_group[P]][0]


finger_hand = [hand_of(d) for d in distal]
hand_names = ["left" if ("left" in p.lower() or p.lower().startswith("lh_")) else "right" for p in palms]
if len(set(hand_names)) != len(hand_names):
    raise RuntimeError(f"[palm] cannot name the hands of palms {palms}")
summary = {"robot_type": args.robot_type, "params": {"spacing": args.spacing, "max_angle": args.max_angle,
                                                     "band_factor": args.band_factor, "task": args.task},
           "finger_canon": dict(zip(fingers, finger_canon)), "finger_hand": dict(zip(fingers, finger_hand)),
           "hand_names": hand_names, "palms": [], "touch": {"pad_order": list(TOUCH_PADS), "fingers": []},
           "rh5dg2_reference": {
               "source": "ro1 robosynapse rh5dg2_right.urdf (right_palm_force_sensor link + STL) and "
                         "robots/ur7/profile.py touch_layout (read-only)",
               "palm_sensor_mesh_extent_mm": [61.1, 28.3, 11.8],
               "palm_sensor_origin_in_palm_m": [0.021234, 0.0001482, 0.1121],
               "finger_mcp_origins_in_palm_m": {"index": [0.0111, 0.033249, 0.15225],
                                                 "middle": [0.0155, 0.0075087, 0.14468],
                                                 "ring": [0.0111, -0.0056513, 0.15225],
                                                 "little": [0.0111, -0.027151, 0.15225]},
               "palm_link_mesh_extent_mm": [51.0, 95.4, 126.5],
               "touch_layout_facing_view_x": {"thumb": 0.22, "palm_l": 0.44, "palm_c": 0.54, "palm_r": 0.64,
                                              "little": 0.70},
               "note": "one physical palm sensor read as 3 zones; zone boundaries unpublished; " + RAW_DIR_NOTE}}
arrs = {}
palm_links, palm_link_palm = [], []
palm_frames = {}
fig_data = []
for pi_, P in enumerate(palms):
    TP_inv = np.linalg.inv(T_of(P))
    bundle = [b for b in palm_group[P] if b not in pad_links and body_meshes(b)]
    if not bundle:
        print(f"[palm] WARNING {P}: no colliders in the palm bundle -> no palm zones", flush=True)
        continue
    samp = []
    for b in bundle:
        Pts, Nn = [], []
        for path, approx, t in body_meshes(b):
            n = int(np.clip(t.area / args.spacing ** 2, 300, 20000))
            pts, fidx = trimesh.sample.sample_surface(t, n, seed=int(rng.integers(1 << 30)))
            Pts.append(pts)
            Nn.append(t.face_normals[fidx])
        Pts, Nn = np.concatenate(Pts), np.concatenate(Nn)
        Tb = TP_inv @ T_of(b)
        samp.append((b, Pts, Nn, Pts @ Tb[:3, :3].T + Tb[:3, 3], Nn @ Tb[:3, :3].T, Tb))
    allP = np.concatenate([s[3] for s in samp])
    # palmar normal from the non-thumb fingers' palmar directions (reset pose)
    mine = [i for i in range(len(fingers)) if finger_hand[i] == pi_]
    nonthumb = [i for i in mine if finger_canon[i] != "thumb"]
    hs = []
    for i in nonthumb:
        RdP = (TP_inv @ T_of(distal[i]))[:3, :3]
        hs.append(RdP @ Z0[f"frame_{i}"][2])
    h = np.mean(hs, 0)
    h /= np.linalg.norm(h)

    def base_of(d):
        b = d
        while joints[b]["parent"] not in palm_group[P]:
            b = joints[b]["parent"]
        return (TP_inv @ T_of(b))[:3, 3]

    base = {finger_canon[i]: base_of(distal[i]) for i in mine}
    c_all = allP.mean(0)
    u = np.mean([base[finger_canon[i]] for i in nonthumb], 0) - c_all
    u -= h * (u @ h)
    u /= np.linalg.norm(u)
    order = [c for c in ("index", "middle", "ring", "little") if c in base]
    t = base[order[0]] - base[order[-1]]
    t -= h * (t @ h) + u * (t @ u)
    t /= np.linalg.norm(t)
    lat = {c: float((base[c] - c_all) @ t * 1e3) for c in base}
    mono = all(lat[order[k]] > lat[order[k + 1]] for k in range(len(order) - 1))
    if not mono:
        raise RuntimeError(f"[palm] finger bases not ordered index->little along the thumb-side axis: {lat}")
    thumb_side_ok = ("thumb" not in lat) or lat["thumb"] > lat[order[-1]]
    # palm band along the distal axis: from the finger bases (median of the non-thumb bases) back by
    # --band_factor x the lateral span of the non-thumb bases; excludes wrist / forearm parts of the palm body
    ub = np.array([(base[finger_canon[i]] - c_all) @ u for i in nonthumb])
    span = abs((base[order[0]] - base[order[-1]]) @ t)
    u_hi, u_lo = float(ub.max() + 0.01), float(np.median(ub) - args.band_factor * span)
    uu = (allP - c_all) @ u
    # lateral range: the non-thumb finger-base span (like the RH5DG2 palm sensor, which spans index..little bases)
    # widened by half the mean base spacing on each side
    bt = np.sort([(base[c] - c_all) @ t for c in order])
    marg = 0.5 * float(np.mean(np.diff(bt))) if len(bt) > 1 else 0.01
    t_lo, t_hi = float(bt[0] - marg), float(bt[-1] + marg)
    lt = (allP - c_all) @ t
    pad_all = (np.concatenate([s[4] @ h >= cos_m for s in samp]) & (uu >= u_lo) & (uu <= u_hi) & (lt >= t_lo)
               & (lt <= t_hi))
    hs_std = float(np.degrees(np.arccos(np.clip(min(float(x @ h) / np.linalg.norm(x) for x in hs), -1, 1))))
    tmin, tmax = lt[pad_all].min(), lt[pad_all].max()
    e1, e2 = tmin + (tmax - tmin) / 3, tmin + 2 * (tmax - tmin) / 3
    c_pad = allP[pad_all].mean(0)
    palm_frames[pi_] = np.array([c_pad, u, h, t])
    base_tu = {cn: [float((bp - c_pad) @ t * 1e3), float((bp - c_pad) @ u * 1e3)] for cn, bp in base.items()}
    pinfo = {"palm": P, "hand": hand_names[pi_], "bundle": bundle,
             "frame_in_palm": {"centre_palmar_face": c_pad.tolist(), "u_distal": u.tolist(), "h_palmar": h.tolist(),
                               "t_thumb_side": t.tolist(), "handedness_det_u_h_t": float(np.linalg.det(np.c_[u, h, t]))},
             "finger_base_t_u_mm_from_face_centre": base_tu, "base_order_ok": mono, "thumb_on_thumb_side": bool(thumb_side_ok),
             "palmar_face_lateral_extent_mm": [float((tmin - (c_pad - c_all) @ t) * 1e3),
                                               float((tmax - (c_pad - c_all) @ t) * 1e3)],
             "zone_edges_lateral_mm_from_face_centre": [float((e - (c_pad - c_all) @ t) * 1e3) for e in (e1, e2)],
             "zone_width_mm": float((tmax - tmin) / 3 * 1e3),
             "palm_band_rule": f"palmar-facing AND distal coordinate within [median non-thumb base - {args.band_factor} x "
                               "lateral base span, max base + 10 mm]",
             "palm_band_u_mm_from_face_centre": [float((u_lo - (c_pad - c_all) @ u) * 1e3),
                                                 float((u_hi - (c_pad - c_all) @ u) * 1e3)],
             "lateral_rule": "non-thumb finger-base span +- half the mean base spacing (RH5DG2 palm sensor spans the "
                             "index..little bases: 61 mm vs 60 mm)",
             "finger_palmar_dir_max_dev_deg": hs_std, "reference_pose": "all joints 0 clipped to limits",
             "links": []}
    off = 0
    for (b, Pts, Nn, xP, nP, Tb) in samp:
        k = len(Pts)
        pad = pad_all[off:off + k]
        l_ = lt[off:off + k]
        off += k
        zone = np.zeros(k, np.int8)
        zone[pad & (l_ >= e2)] = 1          # palm_l: thumb side
        zone[pad & (l_ >= e1) & (l_ < e2)] = 2
        zone[pad & (l_ < e1)] = 3           # palm_r: little-finger side
        j = len(palm_links)
        palm_links.append(b)
        palm_link_palm.append(pi_)
        arrs[f"palm_samples_{j}"] = Pts.astype(np.float32)
        arrs[f"palm_normals_{j}"] = Nn.astype(np.float32)
        arrs[f"palm_zone_{j}"] = zone
        arrs[f"palm_link_T_{j}"] = Tb                      # link frame -> palm body frame (rigid, figures)
        s_u = (xP - c_pad) @ u
        pinfo["links"].append({"link": b, "samples": int(k), "palmar_face_frac": float(pad.mean()),
                               "zone_samples": {zn: int((zone == zi + 1).sum()) for zi, zn in enumerate(PALM_ZONES)}})
        fig_data.append((pi_, b, xP, zone, s_u))
    pz = np.concatenate([q[4][q[3] > 0] for q in fig_data if q[0] == pi_])
    pinfo["palmar_face_distal_extent_mm"] = [float(pz.min() * 1e3), float(pz.max() * 1e3)]
    summary["palms"].append(pinfo)
    print(json.dumps({k: v for k, v in pinfo.items() if k not in ("links", "frame_in_palm")}), flush=True)
    arrs[f"palm_frame_{pi_}"] = palm_frames[pi_]
if len(palm_frames) != len(palms):
    raise RuntimeError("[palm] every palm needs zones for the touch layout (palm frame missing)")
# finger pad axes [u, h, w_thumb] in the distal frame
for i, f in enumerate(fingers):
    fr = Z0[f"frame_{i}"]
    o, u, h = fr[0], fr[1], fr[2]
    w = np.cross(u, h)
    pi_ = finger_hand[i]
    R_dP = (np.linalg.inv(T_of(distal[i])) @ T_of(palms[pi_]))[:3, :3]     # palm -> distal
    _c, uP, hP, tP = palm_frames[pi_]
    if finger_canon[i] != "thumb":
        dot = float(w @ (R_dP @ tP))
        rule = "toward the palm thumb-side axis t"
    else:
        dot = float(w @ (R_dP @ uP))
        rule = "thumb: toward the palm distal axis u"
        if abs(dot) < 0.3:
            dot = float(w @ (R_dP @ hP))
            rule = "thumb: toward the palm palmar normal h (|w.u| < 0.3)"
    sgn = 1.0 if dot >= 0 else -1.0
    arrs[f"touch_axes_{i}"] = np.array([u, h, sgn * w])
    summary["touch"]["fingers"].append({"finger": f, "canon": finger_canon[i], "hand": hand_names[pi_],
                                        "fty_rule": rule, "alignment_dot": dot, "weak_alignment": abs(dot) < 0.3})
summary["touch"]["slots_valid"] = {hand_names[h]: {pad: int(pad in [finger_canon[i] for i in range(len(fingers))
                                                                    if finger_hand[i] == h] or pad.startswith("palm"))
                                                   for pad in TOUCH_PADS} for h in range(len(palms))}
arrs.update({"palm_links": np.array(palm_links), "palm_link_palm": np.array(palm_link_palm),
             "finger_canon": np.array(finger_canon), "finger_hand": np.array(finger_hand),
             "hand_names": np.array(hand_names)})
np.savez_compressed(NPZ, **Z0, **arrs)
js = NPZ.replace(".npz", ".json")
J = json.load(open(js))
J["palm_pads"] = summary
json.dump(J, open(js, "w"), indent=1, default=float)

# ------------------------------------------------------------------ figure: zones (palmar view + side view)
out_dir = args.out_dir or PAD_DIR
os.makedirs(out_dir, exist_ok=True)
colors = {0: "#dddddd", 1: "tab:red", 2: "tab:green", 3: "tab:blue"}
fig, axs = plt.subplots(len(palms), 2, figsize=(10, 4.6 * len(palms)), squeeze=False)
for pi_ in range(len(palms)):
    c, u, h, t = palm_frames[pi_]
    for (pp, b, xP, zone, s_u) in [q for q in fig_data if q[0] == pi_]:
        sub = rng.choice(len(xP), min(len(xP), 6000), replace=False)
        cc = [colors[int(z)] for z in zone[sub]]
        axs[pi_, 0].scatter((xP[sub] - c) @ t * 1e3, (xP[sub] - c) @ u * 1e3, s=1.2, c=cc)
        axs[pi_, 1].scatter((xP[sub] - c) @ u * 1e3, (xP[sub] - c) @ h * 1e3, s=1.2, c=cc)
    pinfo = summary["palms"][pi_]
    for cn, (tv, uv) in pinfo["finger_base_t_u_mm_from_face_centre"].items():
        axs[pi_, 0].plot(tv, uv, "kx", ms=6)
        axs[pi_, 0].annotate(cn + " base", (tv, uv), fontsize=7, ha="center", va="bottom")
    axs[pi_, 0].set_title(f"{palms[pi_]} ({hand_names[pi_]}): palmar view, x = thumb side t, y = distal u\n"
                          "red palm_l (thumb side) / green palm_c / blue palm_r (little side) / grey non-pad",
                          fontsize=8)
    axs[pi_, 1].set_title("side view: x = distal u, y = palmar normal h", fontsize=8)
    for ax in axs[pi_]:
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("mm", fontsize=7)
fig.suptitle(f"{args.robot_type}: virtual palm zones (palmar face = normal within {args.max_angle:.0f} deg of h)",
             fontsize=9)
fig.tight_layout()
png = os.path.join(out_dir, f"{args.robot_type}_palm_zones.png")
fig.savefig(png, dpi=130)
print("[palm] wrote", NPZ, js, png, flush=True)
print("PALM_DONE", flush=True)
os._exit(0)
