# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render retargeting outputs of one robot type to mp4 (+ a sidecar json the web UI plots), states only, no physics.

One Isaac launch renders several jobs of the same robot type (the env is built once). A job is ``kind:in:out``:

  source:<demos.pkl>#<episode>:<out.mp4>    the source demo episode (its recorded states)
  kinematic:<target dir>:<out.mp4>          stage 2: kinematic.npz joints + the demo object motion (reference.npz)
  traj:<x_traj.pkl.gz | x_process.pkl.gz>:<out.mp4>
                                            stage 3: a SPIDER result or its execution log (rollbacks included)

  python scripts/retarget/render_retarget.py --robot_type floating_allegro_right --task Dexverse-GraspBleach-v1 \
      --job kinematic:<dir>:<dir>/viz/kinematic.mp4 --headless
"""

import argparse
import gzip
import json
import os
import pickle
import sys
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--robot_type", required=True)
ap.add_argument("--job", action="append", required=True, help="kind:input:output.mp4 (repeatable)")
ap.add_argument("--camera", default="third_person_camera")
ap.add_argument("--size", default="480x640", help="HxW")
ap.add_argument("--stride", type=int, default=1, help="render every n-th state")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import dexverse.tasks  # noqa: E402,F401
from dexverse.retarget.env import RetargetEnv  # noqa: E402
from dexverse.tasks.utils import prune_stale_obs_refs  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def to_torch(x, dev):
    if isinstance(x, dict):
        return {k: to_torch(v, dev) for k, v in x.items()}
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)


def scene_from(env, state):
    """The current scene state with every non-robot entry taken from ``state`` (a source demo's initial state
    sets up the scene for another hand)."""
    cur = env.scene.get_state(is_relative=True)
    for g, d in cur.items():
        for k in d:
            if k != "robot" and k in state.get(g, {}):
                d[k] = to_torch(state[g][k], env.device)
    return cur


def build_env():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg = type(cfg)(robot_type=args.robot_type)
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    h, w = (int(v) for v in args.size.split("x"))
    cam = getattr(cfg.scene, args.camera)
    cam.height, cam.width, cam.data_types = h, w, ["rgb"]
    for name in ("wrist_camera", "right_wrist_camera", "left_wrist_camera"):
        if getattr(cfg.scene, name, None) is not None and name != args.camera:
            setattr(cfg.scene, name, None)
    for group in ("depth", "pointcloud"):
        if getattr(cfg.observations, group, None) is not None:
            setattr(cfg.observations, group, None)
    cfg = prune_stale_obs_refs(cfg)
    cfg.terminations = None
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    return env


class Recorder:
    """Grabs the camera after every scene write and streams the frames into an mp4 (written to a temp name)."""

    def __init__(self, env, out, fps):
        self.env, self.out = env, out
        self.cam = env.scene[args.camera]
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        self.tmp = out + ".part.mp4"
        self.w = imageio.get_writer(self.tmp, fps=fps, codec="libx264", quality=7, macro_block_size=8)

    def grab(self):
        self.env.sim.render()
        self.env.scene.update(dt=0.0)
        self.cam.update(dt=0.0, force_recompute=True)
        self.w.append_data(self.cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))

    def close(self, side):
        self.w.close()
        os.replace(self.tmp, self.out)
        with open(os.path.splitext(self.out)[0] + ".json", "w") as f:
            json.dump(side, f)


def play_states(env, states, out, fps, side):
    rec = Recorder(env, out, fps)
    for st in states[::args.stride]:
        env.scene.reset_to(to_torch(st, env.device), env_ids=None, is_relative=True)
        env.sim.forward()
        rec.grab()
    rec.close(side)


def err_list(err):
    return [None if e is None else np.round(100 * np.asarray(e, dtype=float), 2).tolist() for e in err]


def job_source(env, src, out):
    path, ep = src.rsplit("#", 1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    states = data["episodes"][int(ep)]["states"]
    fps = round(1.0 / env.step_dt)
    play_states(env, states, out, fps, {"kind": "source", "fps": fps, "stride": args.stride,
                                        "n": len(states[::args.stride]), "robot_type": data["robot_type"]})


def job_kinematic(env, d, out):
    rdir = os.path.dirname(os.path.normpath(d))
    ref = np.load(os.path.join(rdir, "reference.npz"), allow_pickle=True)
    meta = json.loads(str(ref["meta"]))
    kin = np.load(os.path.join(d, "kinematic.npz"), allow_pickle=True)
    renv = RetargetEnv(env, args.robot_type)
    with open(meta["demos"], "rb") as f:
        init = pickle.load(f)["episodes"][meta["episode"]]["initial_state"]
    env.scene.reset_to(scene_from(env, init), env_ids=None, is_relative=True)
    names = [str(n) for n in kin["joint_names"]]
    order = [names.index(n) for n in renv.joint_names]
    q = torch.as_tensor(kin["q"][:, order], dtype=torch.float32, device=env.device)
    roots = torch.as_tensor(ref["obj_root"], dtype=torch.float32, device=env.device)
    art = {k: torch.as_tensor(ref[f"art/{k}"], dtype=torch.float32, device=env.device) for k in renv.art_ids}
    T = min(len(q), len(roots))
    fps = round(1.0 / renv.step_dt)
    rec = Recorder(env, out, fps)
    for t in range(0, T, args.stride):
        renv.teleport(q=q[t], roots=roots[t], art={k: v[t] for k, v in art.items()})
        rec.grab()
    tip_err = kin["tip_err"] if "tip_err" in kin else None
    rec.close({"kind": "kinematic", "fps": fps, "stride": args.stride, "n": len(range(0, T, args.stride)),
               "robot_type": args.robot_type, "report": json.loads(str(kin["report"])) if "report" in kin else None,
               "tip_err_cm": None if tip_err is None else np.round(100 * np.nan_to_num(tip_err), 2).tolist(),
               "segments": meta.get("segments")})


def job_traj(env, path, out):
    with gzip.open(path, "rb") as f:
        tr = pickle.load(f)
    fps = tr.get("fps") or round(1.0 / (env.physics_dt * env.cfg.decimation))
    if "frame" in tr:                                   # execution log (stage3 *_process.pkl.gz)
        side = {"kind": "process", "status": tr.get("status"), "obj_ids": tr.get("obj_ids"),
                "frame": tr["frame"], "err_cm": err_list(tr["err"]), "success": tr["success"],
                "events": tr["events"], "result": tr.get("result")}
    else:                                               # final committed trajectory (stage3 *_traj.pkl.gz)
        n = len(tr["states"])
        side = {"kind": "traj", "frame": list(range(n)), "err_cm": err_list([None] + list(tr["err"])),
                "success": [False] + [bool(x) for x in tr.get("success", [])], "events": []}
        res = os.path.join(os.path.dirname(path), os.path.basename(path).replace("_traj.pkl.gz", ".json"))
        if os.path.exists(res):
            side["result"] = json.load(open(res))
            side["obj_ids"] = list(side["result"].get("obj_err_mean_cm", {}))
    s = args.stride
    for k in ("frame", "err_cm", "success"):
        side[k] = side[k][::s]
    side.update({"fps": fps, "stride": s, "n": len(side["frame"]), "robot_type": args.robot_type,
                 "source_mtime": os.path.getmtime(path)})
    play_states(env, tr["states"], out, fps, side)


def main():
    env = build_env()
    for job in args.job:
        kind, src, out = job.split(":", 2)
        t0 = time.time()
        try:
            {"source": job_source, "kinematic": job_kinematic, "traj": job_traj}[kind](env, src, out)
            print(f"RESULT render {kind} {src} -> {out} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:                          # one bad job must not lose the others
            import traceback

            traceback.print_exc()
            print(f"RESULT render FAILED {kind} {src}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
