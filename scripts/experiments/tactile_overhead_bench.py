# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overhead / latency benchmark of the tactile logging paths on a replayed demo (experiment, measurement only).

The process builds the env the way ``record_demos.py`` does for a LeRobot replay (CPU physics, rendering mode
performance, task cameras RGB at ``--image_size``) with the sensor set of ``--sensors``:

* ``none``     -> condition 1 (no tactile sensors)
* ``tactile``  -> condition 2 (``--tactile`` per-pad sensors)
* ``pads``     -> conditions 3, 4a, 4b, 4c (``--tactile`` + ``--tactile_pads``), run one after the other in the same
                  process: 3 = pad split only, 4a = 3 + online TacFF only, 4b = 3 + online TacRGB only,
                  4c = 3 + both (TacRGB = Taxim on CUDA; 4bC / 4cC = the same with Taxim on CPU)

and replays the episode ``--repeats`` times per condition. Per control step it times (perf_counter, CUDA
synchronised where a GPU tensor is touched) the env step split into physics (``sim.step``) / render
(``sim.render``) / the rest, the substep pose-cache hook, camera capture (RGB of both cameras to host uint8),
state read, writing the frame to a buffer file on local disk, the ``--tactile`` read, the pad read (+ separately
measured raw-read / aggregate / classification parts), TacFF and TacRGB generation, and the latency from the end
of the LAST physics substep to the moment each tactile product is in hand. Nothing in the feature code is changed:
timers wrap instance methods of this process only.

    PYTHONPATH=<worktree>/source/dexverse /workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/tactile_overhead_bench.py \
        --task Dexverse-PickCube-v0 --robot_type floating_allegro_right --pickle demo.pkl --sensors pads \
        --out <dir> --headless --enable_cameras
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--robot_type", required=True)
parser.add_argument("--pickle", default=None)
parser.add_argument("--h5", default=None)
parser.add_argument("--xemb_conv", default=None)
parser.add_argument("--xemb_hand", default=None)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--sensors", choices=["none", "tactile", "pads"], required=True)
parser.add_argument("--conditions", default=None, help="comma list among 1,2,3,4a,4b,4c,4bC,4cC,3d (default by --sensors)")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--image_size", default="480x640")
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--no_cameras", action="store_true", help="strip the task cameras (hosts where RTX rendering fails)")
parser.add_argument("--out", required=True)
parser.add_argument("--gelsight_dir", default="/workspace/dexverse/tactile_results_251008/inputs")
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

import importlib.util  # noqa: E402
import pickle  # noqa: E402
import types  # noqa: E402

import cv2  # noqa: E402
import dexverse  # noqa: E402
import dexverse.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from dexverse.data_collection.tactile import TactileReader, add_tactile_sensors, resolve_targets  # noqa: E402
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
from scipy.spatial import cKDTree  # noqa: E402

print("DEXVERSE_FILE", dexverse.__file__, flush=True)
os.makedirs(args.out, exist_ok=True)
conds = args.conditions.split(",") if args.conditions else {"none": ["1"], "tactile": ["2"],
                                                             "pads": ["3", "4a", "4b", "4c", "4bC", "3d"]}[args.sensors]
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
if args.no_cameras:
    from dexverse.tasks.utils import prune_stale_obs_refs, strip_camera_cfgs  # noqa: PLC0415

    cfg = prune_stale_obs_refs(strip_camera_cfgs(cfg))
    rec_cams = []
else:
    rec_cams = ["third_person_camera",
                next(n for n in ("wrist_camera", "right_wrist_camera", "left_wrist_camera") if n in cams)]
targets = resolve_targets(cfg)
tac = add_tactile_sensors(cfg) if args.sensors in ("tactile", "pads") else None
if args.sensors == "pads":
    add_tactile_pad_sensor(cfg, targets, args.robot_type)
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
treader = TactileReader(env, tac["bodies"], tac["targets"]) if tac else None
reader = TactilePadReader(env, args.robot_type, targets) if args.sensors == "pads" else None

# ------------------------------------------------------------------ timers (instance-level wrappers, this process only)
ACC = {"physics": 0.0, "render": 0.0, "hook": 0.0, "scene_update": 0.0, "obs": 0.0}
MARK = {"phys_end": 0.0}
_sim_step, _sim_render = env.sim.step, env.sim.render


def _t_step(*a, **k):
    t0 = time.perf_counter()
    r = _sim_step(*a, **k)
    MARK["phys_end"] = time.perf_counter()
    ACC["physics"] += MARK["phys_end"] - t0
    return r


def _t_render(*a, **k):
    t0 = time.perf_counter()
    r = _sim_render(*a, **k)
    ACC["render"] += time.perf_counter() - t0
    return r


env.sim.step, env.sim.render = _t_step, _t_render
if reader is not None:
    _orig_hook = reader._on_substep

    def _t_hook():
        t0 = time.perf_counter()
        _orig_hook()
        ACC["hook"] += time.perf_counter() - t0

    hooks = env.scene._tactile_pad_pose_hooks
    hooks[hooks.index(_orig_hook)] = _t_hook
_scene_update = env.scene.update


def _t_update(dt):
    t0 = time.perf_counter()
    _scene_update(dt)
    ACC["scene_update"] += time.perf_counter() - t0


env.scene.update = _t_update
_obs_compute = env.observation_manager.compute


def _t_obs(*a, **k):
    t0 = time.perf_counter()
    r = _obs_compute(*a, **k)
    ACC["obs"] += time.perf_counter() - t0
    return r


env.observation_manager.compute = _t_obs

# ------------------------------------------------------------------ online TacFF / TacRGB (same algorithms as the video renderer)
FF = RG = None
if reader is not None:
    D = reader.d
    Zd = np.load(D["path"])
    nf = reader.nf
    lp0, R0 = _np(robot.data.body_link_pos_w[0]), _qmat(_np(robot.data.body_link_quat_w[0]))
    didx = [robot.body_names.index(n) for n in D["distal"]]
    FF = []
    for fi in range(nf):
        fr = Zd[f"frame_{fi}"]
        o, u, h = fr[0], fr[1], fr[2]
        w = np.cross(u, h)
        S, Mk = [], []
        for li, ln in enumerate(D["links"]):
            if D["link_finger"][li] != fi:
                continue
            x = D["samples"][ln]
            if ln != D["distal"][fi]:
                il, idd = robot.body_names.index(ln), didx[fi]
                x = ((x @ R0[il].T + lp0[il]) - lp0[idd]) @ R0[idd]
            S.append(x)
            Mk.append(D["masks"][ln])
        S, Mk = np.concatenate(S), np.concatenate(Mk)
        padsw = np.c_[(S[Mk] - o) @ u, (S[Mk] - o) @ w] * 1e3
        smin, smax, wmin, wmax = padsw[:, 0].min(), padsw[:, 0].max(), padsw[:, 1].min(), padsw[:, 1].max()
        GS, GW = np.meshgrid(np.linspace(smax, smin, 16), np.linspace(wmin, wmax, 12), indexing="ij")
        nodes = np.c_[GS.ravel(), GW.ravel()]
        tp = cKDTree(padsw)
        cs, cw = (smin + smax) / 2, (wmin + wmax) / 2
        PS, PW = np.meshgrid(np.linspace(cs + 14, cs - 14, 160), np.linspace(cw - 10.5, cw + 10.5, 120), indexing="ij")
        dd, ii = tp.query(np.c_[PS.ravel(), PW.ravel()], k=8)
        wk = 1.0 / np.maximum(dd, 0.05)
        FF.append({"o": o, "u": u, "w": w, "nodes": nodes, "node_on": tp.query(nodes)[0] < 1.5,
                   "pix_on": dd[:, 0] < 1.0,
                   "pad3d": (S[Mk][ii] * wk[..., None]).sum(1) / wk.sum(1)[:, None], "tree3d": cKDTree(S), "mask": Mk})

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
            for g in ([p] if p.IsA(UsdGeom.Gprim) else [d for d in Usd.PrimRange(p, Usd.TraverseInstanceProxies())
                                                         if d.IsA(UsdGeom.Gprim)]):
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
                    cy = UsdGeom.Cylinder(g)
                    t = trimesh.creation.cylinder(radius=cy.GetRadiusAttr().Get(), height=cy.GetHeightAttr().Get())
                else:
                    continue
                rel = xc.GetLocalToWorldTransform(g) * inv
                t.apply_transform(np.array([[rel[i][j] for j in range(4)] for i in range(4)]).T)
                pts.append(np.r_[t.vertices, trimesh.sample.sample_surface(t, int(np.clip(t.area / spacing ** 2, 2000, n_max)),
                                                                          seed=0)[0]])
        return np.concatenate(pts)

    tgt_names = [t.rsplit("/", 1)[-1] for t in targets]
    SURF = {n: surface_samples(t.replace("{ENV_REGEX_NS}", "/World/envs/env_0")) for n, t in zip(tgt_names, targets)}
    STREE = {n: cKDTree(v) for n, v in SURF.items()}
    obj = env.scene["object"] if getattr(cfg.scene, "object", None) is not None else None
    art = env.scene["articulation"] if obj is None else None

    def tgt_poses():
        if obj is not None:
            return [(_np(obj.data.root_link_pos_w[0]), _qmat(_np(obj.data.root_link_quat_w[0])))]
        return [(_np(art.data.body_link_pos_w[0, art.body_names.index(n)]),
                 _qmat(_np(art.data.body_link_quat_w[0, art.body_names.index(n)]))) for n in tgt_names]

    def surf_dist(n, x, k=20):
        d, idx = STREE[n].query(x, k=k)
        nb = SURF[n][idx]
        c = nb.mean(1)
        cov = np.einsum("nki,nkj->nij", nb - c[:, None], nb - c[:, None])
        nrm = np.linalg.eigh(cov)[1][:, :, 0]
        return np.where(d[:, 0] < 0.004, np.abs(np.einsum("ni,ni->n", x - c, nrm)), d[:, 0])

    def load_taxim(device):
        for name in ("isaaclab", "isaaclab.utils", "isaaclab.utils.assets"):
            sys.modules.setdefault(name, types.ModuleType(name))
        if not hasattr(sys.modules["isaaclab.utils.assets"], "retrieve_file_path"):
            sys.modules["isaaclab.utils.assets"].retrieve_file_path = lambda path, **kw: path
        spec = importlib.util.spec_from_file_location(
            "visuotactile_render_bench",
            "/workspace/isaaclab/source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/visuotactile_render.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        c = types.SimpleNamespace(base_data_path=args.gelsight_dir, sensor_data_dir_name="gelsight_r15_data",
                                  background_path="bg.jpg", calib_path="polycalib.npz", real_background="real_bg.npy",
                                  image_height=320, image_width=240, num_bins=120, mm_per_pixel=0.0877)
        return mod.GelsightRender(c, device=device)

    # NOTE: sys.modules stubs for isaaclab.utils.assets would shadow the real package for later imports; Isaac Lab
    # is fully imported by now, so only the renderer module sees them (the real module object is kept if present).
    TAXIM = {}


def tacff(rows_detail, lp_prev, R_prev, fr_link):
    """Node normal force (nf, 192) and node shear (nf, 192, 2) from this step's pad rows."""
    nn = np.zeros((reader.nf, 192))
    ns = np.zeros((reader.nf, 192, 2))
    padF = np.zeros(reader.nf)
    for (ln, xl, vec, _isn, lab, _dp, _dn) in rows_detail:
        fi = reader.link_finger[ln]
        if not lab.any():
            continue
        rl, rd = robot.body_names.index(ln), didx[fi]
        xd = ((xl @ R_prev[rl].T + lp_prev[rl]) - lp_prev[rd]) @ R_prev[rd]
        mag = np.linalg.norm(vec, axis=1)
        ff = (fr_link[reader.sens_names.index(ln)] @ R_prev[rd])[None] * (mag / max(mag.sum(), 1e-12))[:, None]
        f = FF[fi]
        xd, mag, ff = xd[lab], mag[lab], ff[lab]
        sw = np.c_[(xd - f["o"]) @ f["u"], (xd - f["o"]) @ f["w"]] * 1e3
        sh = np.c_[ff @ f["u"], ff @ f["w"]]
        padF[fi] += mag.sum()
        sig = np.clip(mag ** (1 / 3), 1.0, 4.0)
        wgt = np.exp(-((f["nodes"][None] - sw[:, None]) ** 2).sum(-1) / (2 * sig[:, None] ** 2)) * f["node_on"][None]
        wgt /= np.maximum(wgt.sum(1, keepdims=True), 1e-12)
        nn[fi] += (mag[:, None] * wgt).sum(0)
        ns[fi] += np.einsum("kn,kc->nc", wgt, sh)
    return nn, ns, padF


def tacrgb(padF, lp_prev, R_prev, device):
    if device not in TAXIM:
        TAXIM[device] = load_taxim(device)
    tp = tgt_poses()
    imgs = []
    for fi in range(reader.nf):
        f = FF[fi]
        ind = np.zeros(len(f["pix_on"]))
        if padF[fi] > 0.05:
            xw = f["pad3d"][f["pix_on"]] @ R_prev[didx[fi]].T + lp_prev[didx[fi]]
            g = np.full(len(xw), np.inf)
            for (pt, Rt), n in zip(tp, tgt_names):
                g = np.minimum(g, surf_dist(n, (xw - pt) @ Rt))
            delta = min(0.6e-3 * (padF[fi] / 5.0) ** (2 / 3), 2e-3)
            ind[f["pix_on"]] = np.clip(delta - (g - g.min()), 0, None)
        hm = cv2.resize(cv2.GaussianBlur(ind.reshape(160, 120).astype(np.float32), (0, 0), 1.5), (240, 320))
        imgs.append(hm)
    out = TAXIM[device].render(torch.tensor(np.stack(imgs), device=device))
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return out.to("cpu").numpy()


# ------------------------------------------------------------------ episode source
def to_t(x):
    if isinstance(x, dict):
        return {k: to_t(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=env.device)
    return x


def load_episode():
    if args.pickle and args.xemb_conv:
        with open(args.pickle, "rb") as fh:
            ep = [e for e in pickle.load(fh)["episodes"] if len(e.get("actions", []))][args.episode]
        acts = [np.asarray(a, np.float32) for a in np.load(args.xemb_conv)[f"ep{args.episode}/act_{args.xemb_hand}"]]
        return ("xemb", ep), acts
    if args.pickle:
        with open(args.pickle, "rb") as fh:
            ep = [e for e in pickle.load(fh)["episodes"] if len(e.get("actions", []))][args.episode]
        return ("pickle", ep), [np.asarray(a, np.float32).reshape(-1) for a in ep["actions"]]
    import h5py  # noqa: PLC0415

    f = h5py.File(args.h5, "r")
    meta = json.loads(f.attrs["names_json"])
    g = f["episodes"][sorted(f["episodes"].keys())[args.episode]]
    return ("h5", (meta, g)), [np.asarray(a, np.float32) for a in g["action_env"][:]]


SRC, ACTIONS = load_episode()
if args.max_steps:
    ACTIONS = ACTIONS[: args.max_steps]


def start_episode():
    env.sim.reset()
    env.reset()
    kind, ep = SRC
    if kind == "pickle":
        env.reset_to(to_t(ep["initial_state"]), env_ids=None, is_relative=True)
        reset_managers_after_state_restore(env)
    elif kind == "xemb":
        live = env.scene.get_state(is_relative=True)
        st = {c: dict(e) for c, e in live.items()}
        for c, ents in to_t(ep["initial_state"]).items():
            if isinstance(ents, dict):
                for n, s in ents.items():
                    if n != "robot":
                        st.setdefault(c, {})[n] = s
        env.reset_to(st, env_ids=None, is_relative=True)
        reset_managers_after_state_restore(env)
    else:
        meta, g = ep
        jn = meta["robot_joint_names"]
        order = [jn.index(n) for n in robot.joint_names]
        q0 = torch.tensor(np.asarray(g["obs_hand_specific/joint_pos"][0], np.float32)[order])[None]
        robot.write_joint_state_to_sim(q0, torch.tensor(np.asarray(g["obs_hand_specific/joint_vel"][0], np.float32)[order])[None])
        robot.set_joint_position_target(q0)
        for name in meta["rigid_objects"]:
            o_ = env.scene[name]
            o_.write_root_pose_to_sim(torch.tensor(np.asarray(g[f"obs_object_world/{name}/root_pose"][0], np.float32))[None])
            o_.write_root_velocity_to_sim(torch.tensor(np.asarray(g[f"obs_object_world/{name}/root_vel"][0], np.float32))[None])
        env.scene.write_data_to_sim()
        env.sim.forward()
    if reader is not None:
        reader.mark_reset()


# ------------------------------------------------------------------ run
buf_path = os.path.join("/root", f"bench_buf_{os.getpid()}.u8")
results = []
PER_STEP = {}   # per condition / repeat: (T, 13) [t, step, physics, render, obs, tactile, pad, tacff, tacrgb,
#                 camera+state+buffer, total, step_return_latency, pad normal rows (-1 = no pad reader)] (s)
torch.set_grad_enabled(False)
print(f"[bench] pid {os.getpid()} conditions {conds} steps {len(ACTIONS)} image {H}x{W} cams {rec_cams}", flush=True)
main_conds = [c for c in conds if c != "3d"]
for rep in range(args.repeats):
    # rotate the condition order every repeat (drift / thermal / neighbour-load bias), diagnostic episode last
    order = main_conds[rep % len(main_conds):] + main_conds[: rep % len(main_conds)] + (["3d"] if "3d" in conds and rep == 0 else [])
    for cond in order:
        start_episode()
        buf = open(buf_path, "wb", buffering=8 << 20)
        do_ff = cond in ("4a", "4c", "4cC")
        do_rgb = cond in ("4b", "4c", "4bC", "4cC")
        seg = {k: [] for k in ("step", "physics", "render", "hook", "scene_update", "obs", "step_other", "camera", "state",
                               "buffer", "tactile", "pad", "tacff", "tacrgb", "total", "lat_step_return", "lat_tactile",
                               "lat_pad", "lat_tacff", "lat_tacrgb")}
        diag = {"raw_read_copy": [], "aggregate_and_pose_read": [], "classify": [], "rows": []} if cond == "3d" else {}
        per_step = []
        cpu0, wall0 = time.process_time(), time.perf_counter()
        for t, a in enumerate(ACTIONS):
            for k in ACC:
                ACC[k] = 0.0
            t0 = time.perf_counter()
            env.step(torch.as_tensor(a.reshape(1, -1), device=env.device))
            t1 = time.perf_counter()
            pe = MARK["phys_end"]
            lat = {"lat_step_return": t1 - pe}
            # tactile products first (latency = end of the last physics substep -> product in hand)
            if treader is not None:
                treader.read()
            t2 = time.perf_counter()
            if treader is not None:
                lat["lat_tactile"] = t2 - pe
            if reader is not None:
                reader.read(keep_rows=do_ff or do_rgb)
            t3 = time.perf_counter()
            if reader is not None:
                lat["lat_pad"] = t3 - pe
            if do_ff:
                lp_prev, R_prev = reader.last_diag["prev_pose"]
                fr_link = np.nan_to_num(_np(reader.sensor.data.friction_forces_w[0])).sum(1)
                nn, ns, padF = tacff(reader.last_diag["rows_detail"], lp_prev, R_prev, fr_link)
            t4 = time.perf_counter()
            if do_ff:
                lat["lat_tacff"] = t4 - pe
            if do_rgb:
                lp_prev, R_prev = reader.last_diag["prev_pose"]
                if not do_ff:  # TacRGB alone: per-finger pad normal force from this step's rows (part of its cost)
                    padF = np.zeros(reader.nf)
                    for (ln, _xl, vec, _isn, lab, _dp, _dn) in reader.last_diag["rows_detail"]:
                        padF[reader.link_finger[ln]] += np.linalg.norm(vec[lab], axis=1).sum()
                tacrgb(padF, lp_prev, R_prev, "cpu" if cond.endswith("C") else "cuda:0")
            t5 = time.perf_counter()
            if do_rgb:
                lat["lat_tacrgb"] = t5 - pe
            # camera capture as the LeRobot recorder (RGB of both cameras to host uint8), state, frame write
            imgs = []
            for cname in rec_cams:
                rgb = env.scene[cname].data.output["rgb"][0, ..., :3].to("cpu").numpy()
                imgs.append(rgb if rgb.dtype == np.uint8 else np.clip(rgb * 255, 0, 255).astype(np.uint8))
            t6 = time.perf_counter()
            js = np.r_[robot.data.joint_pos[0].cpu().numpy(), robot.data.joint_vel[0].cpu().numpy()]
            t7 = time.perf_counter()
            for im in imgs:
                buf.write(np.ascontiguousarray(im).tobytes())
            buf.write(js.astype(np.float32).tobytes())
            t8 = time.perf_counter()
            if cond == "3d" and t >= args.warmup:
                # pad-read breakdown (diagnostic episode only, outside the timed segments): raw contact + friction rows
                # read & copy, aggregated sensor buffers, nearest-sample classification of this step's rows
                view = reader.sensor.contact_physx_view
                ta = time.perf_counter()
                [_np(x) for x in view.get_contact_data(dt=reader.dt)]
                [_np(x) for x in view.get_friction_data(dt=reader.dt)]
                tb = time.perf_counter()
                _np(reader.sensor.data.force_matrix_w[0])
                _np(reader.sensor.data.friction_forces_w[0])
                _np(robot.data.body_link_pos_w[0])
                _np(robot.data.body_link_quat_w[0])
                tc = time.perf_counter()
                reader.read(keep_rows=True)
                rows = reader.last_diag["rows_detail"]
                td = time.perf_counter()
                for (ln, xl, *_r) in rows:
                    reader.classify(ln, xl)
                te = time.perf_counter()
                diag["raw_read_copy"].append(tb - ta)
                diag["aggregate_and_pose_read"].append(tc - tb)
                diag["classify"].append(te - td)
                diag["rows"].append(sum(len(r[1]) for r in rows))
            per_step.append([t, t1 - t0, ACC["physics"], ACC["render"], ACC["obs"], t2 - t1, t3 - t2, t4 - t3, t5 - t4,
                             t8 - t5, t8 - t0, t1 - pe, reader.last_diag.get("rows", 0) if reader is not None else -1])
            if t >= args.warmup:
                seg["step"].append(t1 - t0)
                seg["physics"].append(ACC["physics"])
                seg["render"].append(ACC["render"])
                seg["hook"].append(ACC["hook"])
                seg["scene_update"].append(ACC["scene_update"] - ACC["hook"])
                seg["obs"].append(ACC["obs"])
                seg["step_other"].append(t1 - t0 - ACC["physics"] - ACC["render"] - ACC["scene_update"] - ACC["obs"])
                seg["tactile"].append(t2 - t1)
                seg["pad"].append(t3 - t2)
                seg["tacff"].append(t4 - t3)
                seg["tacrgb"].append(t5 - t4)
                seg["camera"].append(t6 - t5)
                seg["state"].append(t7 - t6)
                seg["buffer"].append(t8 - t7)
                seg["total"].append(t8 - t0)
                for k, v in lat.items():
                    seg[k].append(v)
        wall = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
        buf.close()
        pad_parts = None
        if diag:
            pad_parts = {k: float(np.median(v) * 1e3) for k, v in diag.items() if k != "rows"}
            pad_parts["rows_median"] = float(np.median(diag["rows"]))
            pad_parts["rows_max"] = int(np.max(diag["rows"]))
        r = {"cond": cond, "rep": rep, "steps": len(ACTIONS), "timed_steps": len(seg["step"]), "episode_wall_s": wall,
             "cpu_cores_avg": cpu / wall, "pad_parts": pad_parts,
             "ms": {k: {"median": float(np.median(v) * 1e3), "p95": float(np.percentile(v, 95) * 1e3),
                        "mean": float(np.mean(v) * 1e3)} for k, v in seg.items() if len(v)}}
        results.append(r)
        PER_STEP[f"{cond}_r{rep}"] = np.asarray(per_step, np.float64)
        print(f"[bench] cond {cond} rep {rep}: total median {r['ms']['total']['median']:.2f} ms (step "
              f"{r['ms']['step']['median']:.2f}, cam {r['ms']['camera']['median']:.2f}, pad {r['ms']['pad']['median']:.2f}, "
              f"ff {r['ms']['tacff']['median']:.2f}, rgb {r['ms']['tacrgb']['median']:.2f}) episode {wall:.1f}s cpu {cpu / wall:.2f}",
              flush=True)
if os.path.exists(buf_path):
    os.remove(buf_path)
info = {"no_cameras": bool(args.no_cameras), "task": args.task, "robot_type": args.robot_type, "sensors": args.sensors, "image_hw": [H, W],
        "physics_dt": env.physics_dt, "step_dt": env.step_dt, "decimation": env.cfg.decimation,
        "torch_cuda_max_mem_MB": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
        "results": results}
json.dump(info, open(os.path.join(args.out, f"bench_{args.sensors}.json"), "w"), indent=1)
np.savez_compressed(os.path.join(args.out, f"steps_{args.sensors}.npz"), **PER_STEP)
print("BENCH_DONE", flush=True)
os._exit(0)
