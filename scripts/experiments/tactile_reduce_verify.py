# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validation of the substep-averaged default tactile logging (experiment).

Replays one demo pickle through the real LeRobot spool recorder (cameras ON, 120x160, no writer process) with the
default tactile logging in ``mean`` mode and the logger's substep dump enabled, and checks per frame:

* the recorded frame values (observation.tactile.*, observation.tactile.pad.*, palm, palm zones, touch) against
  the direct average of the dumped substep readings (and the force-weighted pad centre);
* pad + non-pad = bundle total and palm zones + palm non-pad = palm total (on the averaged values);
* the object's Newton balance per control step while lifted, m (v_t - v_{t-1}) / dt_ctrl - m g against the hand's
  force on the object (own read-only contact view over all robot links, every substep) averaged over the substeps
  vs the last substep only;
* frame-to-frame jitter of the pad forces / touch fn: mean vs last substep (both from the same dump);
* time per control step spent in the substep hook and in the capture.

    PYTHONPATH=<worktree>/source/dexverse /workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/tactile_reduce_verify.py \
        --task Dexverse-GraspCup-v0 --robot_type floating_shadow_right --pickle demo.pkl --out <dir> --headless --enable_cameras
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
parser.add_argument("--pickle", required=True)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--image_size", default="120x160")
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
from dexverse.data_collection import lerobot_spool as LS  # noqa: E402
from dexverse.data_collection.tactile import _glob  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from dexverse.teleop_utils.episode_task_state import reset_managers_after_state_restore  # noqa: E402

print("DEXVERSE_FILE", dexverse.__file__, flush=True)
os.environ.pop("DEXVERSE_TACTILE", None)
os.environ["DEXVERSE_TACTILE_REDUCE"] = "mean"
os.makedirs(args.out, exist_ok=True)
H, W = (int(v) for v in args.image_size.lower().split("x"))
cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg = type(cfg)(robot_type=args.robot_type)
cfg.sim.device = args.device
cfg.scene.num_envs = 1
cfg.terminations.success = None
cfg.terminations.time_out = None
cfg.recorders = {}
for name in ("third_person_camera", "wrist_camera", "right_wrist_camera", "left_wrist_camera"):
    c = getattr(cfg.scene, name, None)
    if c is not None:
        c.height, c.width = H, W
        c.data_types = ["rgb"]
for group in ("depth", "pointcloud"):
    if getattr(cfg.observations, group, None) is not None:
        setattr(cfg.observations, group, None)
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
root = os.path.join(args.out, "ds")
rec = LS.LeRobotSpoolRecorder(env, cfg, task_id=args.task, root=root, launch_writer=False)
log = rec.tactile_log
assert log is not None, rec.tactile_status
log.debug_substeps = []

# timing wrappers (this process only)
TM = {"hook": 0.0, "capture": 0.0}
_hook = log._on_substep


def _timed_hook():
    t0 = time.perf_counter()
    _hook()
    TM["hook"] += time.perf_counter() - t0


hooks = env.scene._tactile_pad_pose_hooks
hooks[hooks.index(_hook)] = _timed_hook
_pose_hook = log.pads._on_substep


def _timed_pose():
    t0 = time.perf_counter()
    _pose_hook()
    TM["hook"] += time.perf_counter() - t0


hooks[hooks.index(_pose_hook)] = _timed_pose

# force ON the object BY the hand every substep: own read-only contact view over EVERY robot link filtered against
# the object (the object itself has no contact-report API, so its unfiltered net force is not available)
obj = env.scene["object"] if getattr(cfg.scene, "object", None) is not None else None
OBJ = {"net": [], "z": [], "mass": None}
if obj is not None:
    from dexverse.data_collection.tactile import RawContactView  # noqa: E402

    _allv = RawContactView(env, list(robot.body_names), [obj.cfg.prim_path])
    OBJ["mass"] = float(obj.root_physx_view.get_masses()[0].sum())

    def _obj_hook():
        d = _allv.read()
        OBJ["net"].append(-(d["fmat"] + d["fric"]).sum((0, 1)))          # reaction: force on the object
        OBJ["z"].append(float(obj.data.root_pos_w[0, 2]))

    hooks.append(_obj_hook)


def to_t(x):
    if isinstance(x, dict):
        return {k: to_t(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=env.device)
    return x


with open(args.pickle, "rb") as fh:
    ep = [e for e in pickle.load(fh)["episodes"] if len(e.get("actions", []))][args.episode]
env.sim.reset()
env.reset()
env.reset_to(to_t(ep["initial_state"]), env_ids=None, is_relative=True)
reset_managers_after_state_restore(env)
actions = [np.asarray(a, np.float32).reshape(-1) for a in ep["actions"]]
rec.begin_episode({"source": "tactile_reduce_verify"})
rec.mark_needs_render()
FR, SUBS, VEL, STEPT, ORNG = [], [], [], [], []
with torch.inference_mode():
    for t, a in enumerate(actions):
        at = torch.as_tensor(a.reshape(1, -1), device=env.device)
        n_dbg = len(log.debug_substeps)
        TM["hook"] = 0.0
        t0 = time.perf_counter()
        rec.capture(at)
        TM["capture"] = time.perf_counter() - t0
        FR.append({"tactile": rec._ep["tactile"][-1], "pads": rec._ep["tactile_pads"][-1]})
        SUBS.append(None)
        o0 = len(OBJ["net"])
        t1 = time.perf_counter()
        env.step(at)
        t2 = time.perf_counter()
        ORNG.append((o0, len(OBJ["net"])))
        SUBS[-1] = (n_dbg, len(log.debug_substeps))       # substeps of THIS step = frame t+1
        STEPT.append((t2 - t1, TM["hook"], TM["capture"]))
        if obj is not None:
            VEL.append(obj.data.root_com_lin_vel_w[0].cpu().numpy().astype(np.float64).copy())
        if t % 100 == 0:
            print(f"[rv] step {t}/{len(actions)}", flush=True)
final = rec.commit_episode()
D = log.debug_substeps
T = len(actions)
res = {"task": args.task, "robot_type": args.robot_type, "steps": T, "decimation": log.decimation,
       "physics_dt": log.physics_dt, "spool_episode": final}
# ---------------------------------------------------------------- (1) frame = average of its substeps
err = {}
nsub = []
for t in range(1, T):
    a, b = SUBS[t - 1]                  # substeps of env.step(t-1) -> frame t
    subs = D[a:b]
    nsub.append(len(subs))
    if not subs:
        continue
    fr = FR[t]
    for k, v in fr["tactile"].items():
        ref = np.mean([s["tactile"][k] for s in subs], 0)
        err[f"tactile.{k}"] = max(err.get(f"tactile.{k}", 0.0), float(np.abs(v - ref).max()))
    for k in ("total", "pad", "nonpad", "pad_w", "nonpad_w", "palm", "palm_zone", "palm_nonpad"):
        if k in fr["pads"]:
            ref = np.mean([s["pads"][k] for s in subs], 0)
            err[f"pads.{k}"] = max(err.get(f"pads.{k}", 0.0), float(np.abs(fr["pads"][k] - ref).max()))
    cn = np.sum([s["cen_num"] for s in subs], 0)
    cw = np.sum([s["cen_w"] for s in subs], 0)
    ref = np.where(cw[:, None] > 0, cn / np.maximum(cw, 1e-12)[:, None], 0.0)
    err["pads.pad_centroid"] = max(err.get("pads.pad_centroid", 0.0), float(np.abs(fr["pads"]["pad_centroid"] - ref).max()))
    if "touch" in fr["pads"]:
        pm = np.mean([s["pads"]["pad"] for s in subs], 0)
        zm = np.mean([s["pads"]["palm_zone"] for s in subs], 0) if "palm_zone" in subs[0]["pads"] else None
        ref = log.pads._touch(pm, zm)[0]
        err["touch"] = max(err.get("touch", 0.0), float(np.abs(fr["pads"]["touch"] - ref).max()))
res["frame_vs_direct_substep_average_max_abs"] = err
res["substeps_per_frame"] = {str(k): int(v) for k, v in zip(*np.unique(nsub, return_counts=True))}
# ---------------------------------------------------------------- (2) sums on the averaged values
P = {k: np.stack([f["pads"][k] for f in FR]) for k in FR[0]["pads"]}
res["pad_plus_nonpad_minus_total_max_N"] = float(np.abs(P["pad"] + P["nonpad"] - P["total"]).max())
if "palm_zone" in P:
    res["palm_zones_plus_nonpad_minus_palm_max_N"] = float(np.abs(P["palm_zone"].sum(2) + P["palm_nonpad"] - P["palm"]).max())
# ---------------------------------------------------------------- (3) jitter mean vs last
last_pad = np.zeros_like(P["pad"])
last_fn = np.zeros((T, P["touch"].shape[1] // 3)) if "touch" in P else None
for t in range(1, T):
    a, b = SUBS[t - 1]
    if b > a:
        last_pad[t] = D[b - 1]["pads"]["pad"]
        if last_fn is not None:
            zm = D[b - 1]["pads"].get("palm_zone")
            last_fn[t] = log.pads._touch(D[b - 1]["pads"]["pad"], zm)[0].reshape(-1, 3)[:, 0]
mean_mag = np.linalg.norm(P["pad"], axis=-1)
last_mag = np.linalg.norm(last_pad, axis=-1)
live = (mean_mag[1:] > 0.1) & (mean_mag[:-1] > 0.1)
res["jitter_pad_force_frame_to_frame_median_N"] = {
    "mean": float(np.median(np.abs(np.diff(mean_mag, axis=0))[live])) if live.any() else None,
    "last": float(np.median(np.abs(np.diff(last_mag, axis=0))[live])) if live.any() else None}
if last_fn is not None:
    mfn = P["touch"].reshape(T, -1, 3)[:, :, 0]
    lv = (mfn[1:] > 0.1) & (mfn[:-1] > 0.1)
    res["jitter_touch_fn_frame_to_frame_median_N"] = {
        "mean": float(np.median(np.abs(np.diff(mfn, axis=0))[lv])) if lv.any() else None,
        "last": float(np.median(np.abs(np.diff(last_fn, axis=0))[lv])) if lv.any() else None}
    res["touch_fn_substep_max_ge_mean"] = bool((P["touch_fn_substep_max"] >= mfn - 1e-5).all())
# ---------------------------------------------------------------- (4) object Newton balance per control step
if obj is not None and len(VEL) > 2:
    g = np.asarray(env.cfg.sim.gravity, np.float64)
    m = OBJ["mass"]
    dtc = log.physics_dt * log.decimation
    net = np.asarray(OBJ["net"])                       # per substep
    z_lift = float(np.asarray(OBJ["z"])[: max(ORNG[0][1], 1)].min()) + 0.02   # 2 cm above the start height
    r_mean, r_last = [], []
    for t in range(1, len(VEL)):
        a_, b_ = ORNG[t]                               # substeps of env.step(t): VEL[t-1] -> VEL[t]
        if b_ <= a_:
            continue
        f = net[a_:b_]
        lhs = m * (VEL[t] - VEL[t - 1]) / dtc - m * g
        zz = np.asarray(OBJ["z"][a_:b_])
        if np.linalg.norm(f.mean(0)) < 0.5 or zz.min() < z_lift:   # on / near the table or not held
            continue
        r_mean.append(np.linalg.norm(lhs - f.mean(0)))
        r_last.append(np.linalg.norm(lhs - f[-1]))
    res["object_newton_balance"] = {"mass_kg": m, "frames": len(r_mean),
                                    "resid_median_N": {"mean": float(np.median(r_mean)) if r_mean else None,
                                                       "last": float(np.median(r_last)) if r_last else None},
                                    "resid_p95_N": {"mean": float(np.percentile(r_mean, 95)) if r_mean else None,
                                                    "last": float(np.percentile(r_last, 95)) if r_last else None},
                                    "note": "m (v_t - v_{t-1}) / dt_ctrl - m g vs the hand's force on the object "
                                            "(all robot links, normal + friction), frames with the object lifted >= 2 cm "
                                            "and |mean force| >= 0.5 N"}
# ---------------------------------------------------------------- (5) timing
st = np.asarray(STEPT[20:])
res["timing_ms_median"] = {"env_step_incl_hooks": float(np.median(st[:, 0]) * 1e3),
                           "substep_hooks_per_step": float(np.median(st[:, 1]) * 1e3),
                           "capture": float(np.median(st[:, 2]) * 1e3)}
res["tactile_status"] = {k: v for k, v in rec.tactile_status.items() if k != "definition"}
json.dump(res, open(os.path.join(args.out, "reduce_verify.json"), "w"), indent=1, default=float)
print(json.dumps(res, default=float), flush=True)
print("RV_DONE", flush=True)
os._exit(0)
