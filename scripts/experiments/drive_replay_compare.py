# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop action replay of recorded demos with a given finger drive, every episode on video (experiment).

record_demos keeps the LeRobot frames of successful episodes only; this replays the recorded actions continuously
(one state restore at the episode start, velocity targets zeroed) and writes, per episode, a third-person mp4 and
per-frame metrics, success or not:

* ``success``: the task success term on 10 consecutive frames (record_demos rule) -> ``success_frame``;
* ``obj_err_cm``: distance of every rigid object of the recorded state from its recorded position (max over objects);
* ``joint_err``: max |joint position target - joint position| over the finger joints (rad) -- a weak drive that
  cannot follow the recorded targets shows here first;
* ``tip_force``: max |contact force| on the fingertip pad bodies from the task objects (N, last substep).

    python scripts/experiments/drive_replay_compare.py --task Dexverse-FunctionalHammerStrike-v1 \
        --robot_type floating_allegro_right --demos <demos.pkl> --episodes 0 1 2 --drive @allegro --out <dir> \
        --wrist_camera_body_collision on --headless --enable_cameras
"""

import argparse
import json
import os
import pickle
import re
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--robot_type", required=True)
ap.add_argument("--demos", required=True)
ap.add_argument("--episodes", type=int, nargs="+", required=True)
ap.add_argument("--drive", default="", help="finger drive spec (dexverse.data_collection.finger_drives), '' = config")
ap.add_argument("--out", required=True)
ap.add_argument("--hold_steps", type=int, default=120, help="frames holding the last action (record_demos replay: 120)")
ap.add_argument("--success_steps", type=int, default=10)
ap.add_argument("--video_hw", default="240x320")
ap.add_argument("--wrist_video", action="store_true", help="also write the wrist camera (hand close-up) mp4")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--wrist_camera_body_collision", choices=("on", "off"), default=None)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import dexverse.tasks  # noqa: E402, F401
from dexverse.data_collection.finger_drives import apply_drive_spec  # noqa: E402
from dexverse.data_collection.tactile import TactileReader, resolve_targets, tactile_body_names  # noqa: E402
from dexverse.mimic.env_setup import build_env_cfg, resolve_success_term  # noqa: E402
from dexverse.scene_state import clear_joint_velocity_targets  # noqa: E402
from dexverse.teleop_utils.episode_task_state import reset_managers_after_state_restore  # noqa: E402
from dexverse.teleop_utils.replay_teleop import _to_torch, set_wrist_camera_body_collision  # noqa: E402

WRIST_RE = re.compile(r"(translation|rotation|^(right_|left_)?(x|y|z|roll|pitch|yaw)_joint$)")
os.makedirs(args.out, exist_ok=True)
H, W = (int(v) for v in args.video_hw.split("x"))

data = pickle.load(open(args.demos, "rb"))
# same seed as record_demos replays with (reset events randomise with it)
cfg, success = build_env_cfg(args.task, args.robot_type, device=args.device, keep_cameras=True,
                             seed=data.get("seed"))
cam = cfg.scene.third_person_camera
cam.height, cam.width = H, W
WRIST = next((n for n in ("wrist_camera", "right_wrist_camera") if getattr(cfg.scene, n, None) is not None), None)
if args.wrist_video and WRIST:
    wc = getattr(cfg.scene, WRIST)
    wc.height, wc.width = H, W
cfg.episode_length_s = 1.0e6
if args.wrist_camera_body_collision:
    set_wrist_camera_body_collision(cfg, args.wrist_camera_body_collision == "on")
drive_lines = apply_drive_spec(cfg, args.drive) if args.drive else []
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
term = resolve_success_term(env, success)
robot = env.scene["robot"]
finger = torch.tensor([i for i, n in enumerate(robot.joint_names) if not WRIST_RE.search(n)], device=env.device)
PADS = tactile_body_names(args.robot_type, list(cfg.robot_config.fingertip_body_names), list(robot.body_names))
tac = TactileReader(env, PADS, resolve_targets(env.cfg, None))
episodes = data["episodes"]
summary = {"task": args.task, "robot_type": args.robot_type, "demos": args.demos, "drive": args.drive,
           "drive_changes": drive_lines, "pads": PADS, "episodes": []}


def object_positions(state):
    """{name: (3,) relative position} of the rigid objects of a scene state (recorded numpy or live torch)."""
    out = {}
    for name, s in (state.get("rigid_object") or {}).items():
        p = s["root_pose"]
        p = p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
        out[name] = np.asarray(p, dtype=np.float64).reshape(-1, 7)[0, :3]
    return out


for e in args.episodes:
    ep = episodes[e]
    t0 = time.time()
    env.reset()
    env.reset_to(_to_torch(ep["initial_state"], env.device), env_ids=None, is_relative=True)
    clear_joint_velocity_targets(env.scene)
    reset_managers_after_state_restore(env)
    actions = np.asarray(ep["actions"], np.float32)
    states = ep.get("states")
    T = len(actions) + args.hold_steps
    writer = imageio.get_writer(os.path.join(args.out, f"ep{e:03d}.mp4"), fps=args.fps, codec="libx264",
                                quality=6, macro_block_size=8)
    wwriter = (imageio.get_writer(os.path.join(args.out, f"ep{e:03d}_wrist.mp4"), fps=args.fps, codec="libx264", quality=6,
                                  macro_block_size=8) if args.wrist_video and WRIST else None)
    run, success_frame = 0, -1
    pad_force = []
    obj_err, joint_err, tip_force, succ = [], [], [], []
    for t in range(T):
        a = torch.as_tensor(actions[min(t, len(actions) - 1)], device=env.device).unsqueeze(0)
        env.step(a)
        frame = env.scene["third_person_camera"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        writer.append_data(frame)
        if wwriter is not None:
            wwriter.append_data(env.scene[WRIST].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
        live = object_positions(env.scene.get_state(is_relative=True))
        rec = object_positions(states[min(t + 1, len(states) - 1)]) if states else {}
        errs = [np.linalg.norm(live[k] - rec[k]) for k in rec if k in live]
        obj_err.append(round(100 * max(errs), 2) if errs else None)
        jt = (robot.data.joint_pos_target[0, finger] - robot.data.joint_pos[0, finger]).abs().max().item()
        joint_err.append(round(jt, 4))
        fp = np.linalg.norm(np.asarray(tac.read()["force"]).reshape(-1, 3), axis=1)
        pad_force.append([round(float(v), 1) for v in fp])
        tip_force.append(round(float(fp.max()), 1))
        s = bool(term.func(env, **term.params)[0])
        succ.append(int(s))
        run = run + 1 if s else 0
        if success_frame < 0 and run >= args.success_steps:
            success_frame = t
    writer.close()
    if wwriter is not None:
        wwriter.close()
    rec_row = {"episode": e, "source_episode": ep.get("episode_index", e), "num_actions": len(actions),
               "success": success_frame >= 0, "success_frame": success_frame,
               "obj_err_cm_final": obj_err[len(actions) - 1], "obj_err_cm_max": max(x for x in obj_err if x is not None)
               if any(x is not None for x in obj_err) else None,
               "joint_err_max": max(joint_err), "tip_force_max": max(tip_force),
               "frames": {"obj_err_cm": obj_err, "joint_err": joint_err, "tip_force": tip_force, "success": succ,
                          "pad_force": pad_force},
               "video": f"ep{e:03d}.mp4", "seconds": round(time.time() - t0, 1)}
    json.dump(rec_row, open(os.path.join(args.out, f"ep{e:03d}.json"), "w"))
    summary["episodes"].append({k: v for k, v in rec_row.items() if k != "frames"})
    print(f"[compare] ep{e}: success={rec_row['success']} frame={success_frame} obj_err_final="
          f"{rec_row['obj_err_cm_final']} cm joint_err_max={rec_row['joint_err_max']:.3f} "
          f"tip_force_max={rec_row['tip_force_max']} ({rec_row['seconds']} s)", flush=True)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
os._exit(0)   # Isaac Sim may hang in app.close() after the run
