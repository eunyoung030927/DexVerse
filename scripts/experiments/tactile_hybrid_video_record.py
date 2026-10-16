# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 1 of the hybrid tactile videos (experiment): replay a demo with cameras ON and record, per control step,

* third-person and wrist RGB (``--image_size``, default 240x320),
* every force-carrying raw contact of each finger's distal bundle (virtual-pad reader of
  ``dexverse.data_collection.tactile_pads``: deep-copied raw rows, zero-force rows dropped, points placed with the
  pose of the end of the previous physics substep) expressed in the DISTAL link frame: point, normal force vector
  and a friction share (the link's aggregated friction distributed over its normal rows by normal-force magnitude:
  PhysX friction rows sit at friction-patch anchors, not at the contact points),
* the target bodies' poses at the same instant and the target collision surfaces (dense samples, body frame) for
  the gap field used by the TacRGB-like height maps.

Physics is untouched (read-only sensors; no TacSL / compliant / SDF). Stage 2 = tactile_hybrid_video_render.py.

    PYTHONPATH=<worktree>/source/dexverse /workspace/isaaclab/_isaac_sim/python.sh \
        scripts/experiments/tactile_hybrid_video_record.py --task Dexverse-PickCube-v0 \
        --robot_type floating_allegro_right --pickle demo.pkl --out <dir> --headless --enable_cameras
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--robot_type", required=True)
parser.add_argument("--pickle", default=None)
parser.add_argument("--h5", default=None)
parser.add_argument("--actions_npz", default=None, help="scripted rollout: npz with 'actions' (T, A) [+ 'q0']")
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--xemb_conv", default=None,
                    help="xemb conv_<task>.npz: replay ep<episode>/act_<--xemb_hand> converted from the Shadow demo "
                         "--pickle (scene restored from that demo, robot left at its own reset state = xemb variant A)")
parser.add_argument("--xemb_hand", default=None)
parser.add_argument("--hold", type=int, default=0, help="repeat the last action this many steps")
parser.add_argument("--out", required=True)
parser.add_argument("--image_size", default="240x320")
parser.add_argument("--max_steps", type=int, default=0)
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
from dexverse.data_collection.tactile import resolve_targets  # noqa: E402
from dexverse.data_collection.tactile_pads import (  # noqa: E402
    TactilePadReader,
    _np,
    _qmat,
    add_tactile_pad_sensor,
)
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from dexverse.teleop_utils.episode_task_state import reset_managers_after_state_restore  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

print("DEXVERSE_FILE", dexverse.__file__, flush=True)
os.makedirs(args.out, exist_ok=True)
H, W = (int(v) for v in args.image_size.lower().split("x"))
cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg = type(cfg)(robot_type=args.robot_type)
cfg.sim.device = args.device
cfg.scene.num_envs = 1
cfg.terminations.success = None
cfg.terminations.time_out = None
cfg.recorders = {}
cams = []
for name in ("third_person_camera", "wrist_camera", "right_wrist_camera", "left_wrist_camera"):
    c = getattr(cfg.scene, name, None)
    if c is None:
        continue
    c.height, c.width = H, W
    c.data_types = ["rgb"]
    cams.append(name)
for group in ("depth", "pointcloud"):
    if getattr(cfg.observations, group, None) is not None:
        setattr(cfg.observations, group, None)
third = "third_person_camera"
wrist = next(n for n in ("wrist_camera", "right_wrist_camera", "left_wrist_camera") if n in cams)
targets = resolve_targets(cfg)
add_tactile_pad_sensor(cfg, targets, args.robot_type)
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
reader = TactilePadReader(env, args.robot_type, targets)
D = reader.d
tgt_names = [t.rsplit("/", 1)[-1] for t in targets]
obj = env.scene["object"] if getattr(cfg.scene, "object", None) is not None else None
art = env.scene["articulation"] if obj is None and getattr(cfg.scene, "articulation", None) is not None else None


def tgt_pose():
    if obj is not None:
        return np.r_[_np(obj.data.root_link_pos_w[0]), _np(obj.data.root_link_quat_w[0])][None]
    return np.asarray([np.r_[_np(art.data.body_link_pos_w[0, art.body_names.index(n)]),
                             _np(art.data.body_link_quat_w[0, art.body_names.index(n)])] for n in tgt_names])


tcache = []


def _tgt_hook():
    tcache.append(tgt_pose())
    if len(tcache) > 2:
        tcache.pop(0)


env.scene._tactile_pad_pose_hooks.append(_tgt_hook)


# ------------------------------------------------------------------ target surfaces (unscaled body frame)
def surface_samples(body_path, n_max=200000, spacing=0.0008):
    stage = get_current_stage()
    xc = UsdGeom.XformCache()
    body = stage.GetPrimAtPath(body_path)
    bw = Gf.Transform(xc.GetLocalToWorldTransform(body))
    bw.SetScale(Gf.Vec3d(1, 1, 1))
    inv = bw.GetMatrix().GetInverse()
    pts = []
    for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        geoms = [p] if p.IsA(UsdGeom.Gprim) else [d for d in Usd.PrimRange(p, Usd.TraverseInstanceProxies())
                                                    if d.IsA(UsdGeom.Gprim)]
        for g in geoms:
            if g.IsA(UsdGeom.Mesh):
                m = UsdGeom.Mesh(g)
                v = np.asarray(m.GetPointsAttr().Get(), float)
                cnt, idx = np.asarray(m.GetFaceVertexCountsAttr().Get()), np.asarray(m.GetFaceVertexIndicesAttr().Get())
                f, k = [], 0
                for c in cnt:
                    f += [[idx[k], idx[k + j], idx[k + j + 1]] for j in range(1, c - 1)]
                    k += c
                t = trimesh.Trimesh(v, np.asarray(f), process=False)
            elif g.IsA(UsdGeom.Cube):
                t = trimesh.creation.box(extents=[UsdGeom.Cube(g).GetSizeAttr().Get()] * 3)
            elif g.IsA(UsdGeom.Sphere):
                t = trimesh.creation.icosphere(4, radius=UsdGeom.Sphere(g).GetRadiusAttr().Get())
            elif g.IsA(UsdGeom.Cylinder):
                c = UsdGeom.Cylinder(g)
                t = trimesh.creation.cylinder(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
            else:
                continue
            rel = xc.GetLocalToWorldTransform(g) * inv
            t.apply_transform(np.array([[rel[i][j] for j in range(4)] for i in range(4)]).T)
            n = int(np.clip(t.area / spacing ** 2, 2000, n_max))
            pts.append(np.r_[t.vertices, trimesh.sample.sample_surface(t, n, seed=0)[0]])
    return np.concatenate(pts).astype(np.float32) if pts else np.zeros((0, 3), np.float32)


tgt_surf = {n: surface_samples(t.replace("{ENV_REGEX_NS}", "/World/envs/env_0")) for n, t in zip(tgt_names, targets)}
print(f"[hv] targets {[(n, len(v)) for n, v in tgt_surf.items()]}; cameras {third}, {wrist} at {H}x{W}", flush=True)


# ------------------------------------------------------------------ episode
def to_t(x):
    if isinstance(x, dict):
        return {k: to_t(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=env.device)
    return x


env.sim.reset()
env.reset()
source = ""
if args.pickle and args.xemb_conv:
    with open(args.pickle, "rb") as fh:
        ep = [e for e in pickle.load(fh)["episodes"] if len(e.get("actions", []))][args.episode]
    live = env.scene.get_state(is_relative=True)
    rec_state = to_t(ep["initial_state"])
    state = {cat: dict(ents) for cat, ents in live.items()}
    for cat, ents in rec_state.items():
        if isinstance(ents, dict):
            for name, st in ents.items():
                if name != "robot":
                    state.setdefault(cat, {})[name] = st
    env.reset_to(state, env_ids=None, is_relative=True)
    reset_managers_after_state_restore(env)
    actions = [np.asarray(a, dtype=np.float32) for a in np.load(args.xemb_conv)[f"ep{args.episode}/act_{args.xemb_hand}"]]
    source = (f"retargeted (xemb DexPilot conversion, variant A) from Shadow demo {os.path.basename(os.path.dirname(args.pickle))} "
              f"ep {args.episode}")
elif args.pickle:
    with open(args.pickle, "rb") as fh:
        ep = [e for e in pickle.load(fh)["episodes"] if len(e.get("actions", []))][args.episode]
    env.reset_to(to_t(ep["initial_state"]), env_ids=None, is_relative=True)
    reset_managers_after_state_restore(env)
    actions = [np.asarray(a, dtype=np.float32).reshape(-1) for a in ep["actions"]]
    source = f"pickle {os.path.basename(args.pickle)} ep {args.episode}"
elif args.h5:
    import h5py  # noqa: PLC0415

    f = h5py.File(args.h5, "r")
    meta = json.loads(f.attrs["names_json"])
    eps = sorted(f["episodes"].keys())
    g = f["episodes"][eps[args.episode]]
    jn = meta["robot_joint_names"]
    order = [jn.index(n) for n in robot.joint_names]
    q0 = np.asarray(g["obs_hand_specific/joint_pos"][0], dtype=np.float32)[order]
    qd0 = np.asarray(g["obs_hand_specific/joint_vel"][0], dtype=np.float32)[order]
    robot.write_joint_state_to_sim(torch.tensor(q0)[None], torch.tensor(qd0)[None])
    robot.set_joint_position_target(torch.tensor(q0)[None])
    for name in meta["rigid_objects"]:
        o_ = env.scene[name]
        o_.write_root_pose_to_sim(torch.tensor(np.asarray(g[f"obs_object_world/{name}/root_pose"][0], np.float32))[None])
        o_.write_root_velocity_to_sim(torch.tensor(np.asarray(g[f"obs_object_world/{name}/root_vel"][0], np.float32))[None])
    env.scene.write_data_to_sim()
    env.sim.forward()
    actions = [np.asarray(a, dtype=np.float32) for a in g["action_env"][:]]
    source = f"xemb phase-1 HDF5 {os.path.basename(os.path.dirname(args.h5))}/{os.path.basename(args.h5)} {eps[args.episode]}"
else:
    z = np.load(args.actions_npz)
    if "q0" in z.files:
        q0 = torch.tensor(z["q0"], dtype=torch.float32)[None]
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        robot.set_joint_position_target(q0)
        env.scene.write_data_to_sim()
        env.sim.forward()
    actions = [np.asarray(a, dtype=np.float32) for a in z["actions"]]
    source = f"scripted rollout {os.path.basename(args.actions_npz)}"
if args.hold:
    actions = actions + [actions[-1]] * args.hold
if args.max_steps:
    actions = actions[: args.max_steps]
reader.mark_reset()
T = len(actions)
img3 = np.lib.format.open_memmap(os.path.join(args.out, "third.npy"), mode="w+", dtype=np.uint8, shape=(T, H, W, 3))
imgw = np.lib.format.open_memmap(os.path.join(args.out, "wrist.npy"), mode="w+", dtype=np.uint8, shape=(T, H, W, 3))
rows, dpose, tposes, tot_d = [], [], [], []
prow, pzone, pleft, touch = [], [], [], []          # palm zones (definition with palm zones only)
palm_rob = [robot.body_names.index(n) for n in D["palms"]]
plp = dict(zip(D["palm_links"], D["palm_link_palm"]))
# friction per bundle link of the read (contact-view row order: reader.sens_names)
didx = [robot.body_names.index(n) for n in D["distal"]]
with torch.inference_mode():
    for t, a in enumerate(actions):
        env.step(torch.as_tensor(a.reshape(1, -1), device=env.device))
        for arr, cam in ((img3, third), (imgw, wrist)):
            rgb = env.scene[cam].data.output["rgb"][0, ..., :3].to("cpu").numpy()
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            arr[t] = rgb
        out = reader.read(keep_rows=True)
        lp_prev, R_prev = reader.last_diag["prev_pose"]
        fr_link = reader.last_diag["fric_link"]                                  # (B, 3) friction per link
        link_idx = {n: i for i, n in enumerate(reader.sens_names)}
        tot_d.append(out["total"])
        dpose.append(np.stack([np.r_[lp_prev[i], R_prev[i].reshape(-1)] for i in didx]))
        tposes.append(tcache[0] if len(tcache) == 2 else tgt_pose())
        for (ln, xl, vec, _isn, _lab, _dp, _dn) in reader.last_diag["rows_detail"]:
            fi = reader.link_finger[ln]
            rl = robot.body_names.index(ln)
            rd = didx[fi]
            xw = xl @ R_prev[rl].T + lp_prev[rl]                     # pre-substep world position
            xd = (xw - lp_prev[rd]) @ R_prev[rd]                      # distal-link frame
            fn_d = vec @ R_prev[rd]                                   # normal force on the finger, distal frame
            mag = np.linalg.norm(vec, axis=1)
            ff_d = (fr_link[link_idx[ln]] @ R_prev[rd])[None] * (mag / max(mag.sum(), 1e-12))[:, None]
            for k in range(len(xd)):
                rows.append([t, fi, *xd[k], *fn_d[k], *ff_d[k]])
        if reader.zones_on:
            # palm rows: point in the palm body frame (pre-substep pose), normal force + allocated friction in the palm
            # frame (CURRENT orientation, as the reader's palm_zone / touch). Friction per zone = reader zone force -
            # normal rows of that zone, shared over the zone's normal rows by |normal| (zone sums = the reader's)
            R_cur = _qmat(_np(robot.data.body_link_quat_w[0]))
            zw = reader.last_diag["palm_zone_w"]                       # (n_palms, 4, 3) world [non-pad, l, c, r]
            per = {}
            for (ln, xl, vec, zl) in reader.last_diag["palm_rows_detail"]:
                p = plp[ln]
                rl, rp = robot.body_names.index(ln), palm_rob[p]
                xw = xl @ R_prev[rl].T + lp_prev[rl]
                per.setdefault(p, []).append(((xw - lp_prev[rp]) @ R_prev[rp], vec, zl))
            lz = np.zeros((len(palm_rob), 4, 3))
            for p in range(len(palm_rob)):
                rp = palm_rob[p]
                if p in per:
                    X = np.concatenate([q[0] for q in per[p]])
                    V = np.concatenate([q[1] for q in per[p]])
                    Zl = np.concatenate([q[2] for q in per[p]])
                else:
                    X, V, Zl = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int)
                mag = np.linalg.norm(V, axis=1)
                FF = np.zeros_like(V)
                for k in range(4):
                    sel = Zl == k
                    fr = zw[p, k] - V[sel].sum(0)
                    if sel.any() and mag[sel].sum() > 0:
                        FF[sel] = fr[None] * (mag[sel] / mag[sel].sum())[:, None]
                    else:
                        lz[p, k] = fr @ R_cur[rp]                    # friction without a normal row in the zone
                for k in range(len(X)):
                    prow.append([t, p, *X[k], *(V[k] @ R_cur[rp]), *(FF[k] @ R_cur[rp]), Zl[k]])
            pzone.append(np.einsum("pji,pkj->pki", R_cur[palm_rob], zw))
            pleft.append(lz)
        if reader.touch_on:
            touch.append(out["touch"])
        if t % 100 == 0:
            print(f"[hv] step {t}/{T}", flush=True)
img3.flush()
imgw.flush()
# fixed transforms bundle link -> distal link (for the pad samples of child links)
lp_now, R_now = _np(robot.data.body_link_pos_w[0]), _qmat(_np(robot.data.body_link_quat_w[0]))
link_rel = {}
for ln, fi in zip(D["links"], D["link_finger"]):
    il, idd = robot.body_names.index(ln), didx[int(fi)]
    Rr = R_now[idd].T @ R_now[il]
    link_rel[f"rel_{ln}"] = np.r_[(lp_now[il] - lp_now[idd]) @ R_now[idd], Rr.reshape(-1)]
np.savez_compressed(os.path.join(args.out, "contacts.npz"), **link_rel, rows=np.asarray(rows, np.float64).reshape(-1, 11),
                    distal_pose=np.asarray(dpose), tgt_pose=np.asarray(tposes), total=np.asarray(tot_d),
                    tgt_names=np.array(tgt_names), fingers=np.array(D["fingers"]),
                    step_dt=env.step_dt, **{f"surf_{n}": v for n, v in tgt_surf.items()},
                    **({"palm_rows": np.asarray(prow, np.float64).reshape(-1, 12), "palm_zone": np.asarray(pzone),
                        "palm_left": np.asarray(pleft), "palms": np.array(D["palms"])} if reader.zones_on else {}),
                    **({"touch": np.asarray(touch), "touch_names": np.array(reader.touch_names())}
                       if reader.touch_on else {}))
json.dump({"task": args.task, "robot_type": args.robot_type, "source": source, "steps": T, "image_hw": [H, W],
           "third_cam": third, "wrist_cam": wrist, "step_dt": env.step_dt},
          open(os.path.join(args.out, "info.json"), "w"), indent=1)
print(f"[hv] {T} steps, {len(rows)} pad-bundle contact rows, {len(prow)} palm rows -> {args.out}", flush=True)
print("HV_RECORD_DONE", flush=True)
os._exit(0)
