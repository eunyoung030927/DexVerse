# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validation of the virtual palm zones + real-hand ``observation.touch`` layout on a replayed demo (experiment).

Replays a demo through the REAL LeRobot spool recorder (``LeRobotSpoolRecorder.capture`` before every env.step,
``commit_episode`` at the end, no writer process) with ``--sensors on`` (= record_demos --tactile --tactile_pads) or
``--sensors off`` (no tactile sensors). Cameras are off (hosts where RTX rendering fails): the recorder's camera
list is replaced by an empty one in THIS process only (no camera features); everything else is the recorder's code.

Per control step it stores the scene state (``scene.get_state``), joint state and action for bit-identity
comparisons between runs, and with sensors on also the palm zone forces, palm total / non-pad, the palm residual
(zones + non-pad vs aggregated sensor), ``touch`` / ``touch_valid`` / ``touch_contact`` and the palm contact points
(extra ``read(keep_rows=True)`` after the capture: read-only).

    PYTHONPATH=<worktree>/source/dexverse /workspace/isaaclab/isaaclab.sh -p scripts/experiments/tactile_palm_verify.py \
        --task Dexverse-GraspCup-v0 --robot_type floating_shadow_right --pickle demo.pkl --sensors on --out <dir> --headless
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
parser.add_argument("--xemb_conv", default=None)
parser.add_argument("--xemb_hand", default=None)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--sensors", choices=["on", "off"], required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--spool_root", default=None, help="local dir for the LeRobot spool (default <out>/ds)")
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
from dexverse.data_collection import lerobot_spool as LS  # noqa: E402
from dexverse.data_collection.tactile import add_tactile_sensors, resolve_targets  # noqa: E402
from dexverse.data_collection.tactile_pads import add_tactile_pad_sensor  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg, prune_stale_obs_refs, strip_camera_cfgs  # noqa: E402
from dexverse.teleop_utils.episode_task_state import reset_managers_after_state_restore  # noqa: E402

print("DEXVERSE_FILE", dexverse.__file__, flush=True)
os.makedirs(args.out, exist_ok=True)
cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg = type(cfg)(robot_type=args.robot_type)
cfg.sim.device = args.device
cfg.scene.num_envs = 1
cfg.terminations.success = None
cfg.terminations.time_out = None
cfg.recorders = {}
cfg = prune_stale_obs_refs(strip_camera_cfgs(cfg))
tac = pads = None
if args.sensors == "on":
    tac = add_tactile_sensors(cfg)
    pads = add_tactile_pad_sensor(cfg, resolve_targets(cfg), args.robot_type)
    pads["variant"] = "pad"
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
LS.LeRobotSpoolRecorder._resolve_cameras = lambda self, env_cfg: []        # cameras off: no image features
root = args.spool_root or os.path.join(args.out, "ds")
rec = LS.LeRobotSpoolRecorder(env, cfg, task_id=args.task, root=root, launch_writer=False, tactile=tac,
                              tactile_pads=pads)
reader = rec.tactile_pads


def to_t(x):
    if isinstance(x, dict):
        return {k: to_t(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=env.device)
    return x


def flat(x, pre=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{pre}/{k}"))
    elif isinstance(x, torch.Tensor):
        out[pre] = x.detach().cpu().numpy().copy()
    return out


env.sim.reset()
env.reset()
if args.pickle and args.xemb_conv:
    with open(args.pickle, "rb") as fh:
        ep = [e for e in pickle.load(fh)["episodes"] if len(e.get("actions", []))][args.episode]
    live = env.scene.get_state(is_relative=True)
    st = {c: dict(e) for c, e in live.items()}
    for c, ents in to_t(ep["initial_state"]).items():
        if isinstance(ents, dict):
            for n, s_ in ents.items():
                if n != "robot":
                    st.setdefault(c, {})[n] = s_
    env.reset_to(st, env_ids=None, is_relative=True)
    reset_managers_after_state_restore(env)
    actions = [np.asarray(a, np.float32) for a in np.load(args.xemb_conv)[f"ep{args.episode}/act_{args.xemb_hand}"]]
elif args.pickle:
    with open(args.pickle, "rb") as fh:
        ep = [e for e in pickle.load(fh)["episodes"] if len(e.get("actions", []))][args.episode]
    env.reset_to(to_t(ep["initial_state"]), env_ids=None, is_relative=True)
    reset_managers_after_state_restore(env)
    actions = [np.asarray(a, np.float32).reshape(-1) for a in ep["actions"]]
else:
    import h5py  # noqa: PLC0415

    f = h5py.File(args.h5, "r")
    meta = json.loads(f.attrs["names_json"])
    g = f["episodes"][sorted(f["episodes"].keys())[args.episode]]
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
    actions = [np.asarray(a, np.float32) for a in g["action_env"][:]]
if args.max_steps:
    actions = actions[: args.max_steps]

rec.begin_episode({"source": "tactile_palm_verify"})
rec.mark_needs_render()
S = {"joint_pos": [], "joint_vel": [], "action": []}
SCENE = {}
O = {k: [] for k in ("touch", "touch_valid", "touch_contact", "palm", "palm_zone", "palm_nonpad", "palm_resid", "pad")}
PTS = []      # (step, palm index, link, x_link, |F|, zone)
with torch.inference_mode():
    for t, a in enumerate(actions):
        at = torch.as_tensor(a.reshape(1, -1), device=env.device)
        rec.capture(at)
        if reader is not None and t > 0:
            r = reader.read(keep_rows=True)           # read-only repeat of the capture's read, for rows + diag
            for k in ("touch", "touch_valid", "touch_contact", "palm", "palm_zone", "palm_nonpad", "pad"):
                O[k].append(r[k])
            O["palm_resid"].append(reader.last_diag.get("palm_resid_max_N", 0.0))
            for (ln, xl, vec, zl) in reader.last_diag["palm_rows_detail"]:
                pi = reader.d["palm_link_palm"][reader.d["palm_links"].index(ln)]
                PTS.append((t, pi, ln, xl, np.linalg.norm(vec, axis=1), zl))
        env.step(at)
        S["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        S["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        S["action"].append(a.copy())
        for k, v in flat(env.scene.get_state(is_relative=True)).items():
            SCENE.setdefault(k, []).append(v)
        if t % 100 == 0:
            print(f"[pv] step {t}/{len(actions)}", flush=True)
final = rec.commit_episode()
np.savez_compressed(os.path.join(args.out, "states.npz"), **{k: np.asarray(v) for k, v in S.items()},
                    **{"scene" + k.replace("/", "|"): np.asarray(v) for k, v in SCENE.items()})
res = {"robot_type": args.robot_type, "task": args.task, "sensors": args.sensors, "steps": len(actions),
       "spool_episode": final}
if reader is not None:
    A = {k: np.asarray(v) for k, v in O.items() if len(v)}
    np.savez_compressed(os.path.join(args.out, "touch_steps.npz"), names=np.array(reader.touch_names()), **A)
    with open(os.path.join(args.out, "palm_points.pkl"), "wb") as fh:
        pickle.dump(PTS, fh)
    zsum = A["palm_zone"].sum(2) + A["palm_nonpad"]                  # (T, np, 3)
    res["palm_zone_plus_nonpad_minus_palm_total_max_N"] = float(np.abs(zsum - A["palm"]).max())
    res["palm_resid_vs_aggregated_sensor_max_N"] = float(A["palm_resid"].max())
    pt = np.linalg.norm(A["palm"], axis=-1)
    zn = np.linalg.norm(A["palm_zone"], axis=-1)
    live = pt > 0.05
    res["palm_frames_in_contact"] = live.sum(0).tolist()
    res["palm_zone_share_of_palm_total"] = [float(zn[live[:, p], p].sum() / max(pt[live[:, p], p].sum(), 1e-9))
                                            for p in range(pt.shape[1])]
    res["palm_max_N"] = pt.max(0).tolist()
    T = A["touch"]
    names = reader.touch_names()
    slots = reader.touch_slot_names()
    rng_ = {}
    for j, sn in enumerate(slots):
        fn, fx, fy = T[:, 3 * j], T[:, 3 * j + 1], T[:, 3 * j + 2]
        rng_[sn] = {"valid": int(A["touch_valid"][0, j]), "fn_max": float(fn.max()), "fn_min": float(fn.min()),
                    "ftx_min": float(fx.min()), "ftx_max": float(fx.max()), "fty_min": float(fy.min()),
                    "fty_max": float(fy.max()), "contact_frames": int(A["touch_contact"][:, j].sum())}
    res["touch_ranges"] = rng_
    # consistency: finger fn/ftx/fty magnitude = |pad| ; palm zone |touch| = |zone force|
    pm = np.linalg.norm(A["pad"], axis=-1)
    tm = np.linalg.norm(T.reshape(len(T), -1, 3), axis=-1)
    errs = []
    for i, slot in reader.touch_finger_slot.items():
        errs.append(np.abs(tm[:, slot] - pm[:, i]).max())
    res["touch_norm_vs_pad_norm_max_N"] = float(max(errs))
    sp = os.path.join(final, "data.npz") if final else None
    if sp and os.path.exists(sp):
        z = np.load(sp)
        res["spool_keys"] = sorted(k for k in z.files if "touch" in k or "palm" in k)
        res["spool_touch_shape_dtype"] = {k: [list(z[k].shape), str(z[k].dtype)] for k in res["spool_keys"]}
json.dump(res, open(os.path.join(args.out, "palm_verify.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "touch_ranges"}, default=float), flush=True)
print("PV_DONE", flush=True)
os._exit(0)
