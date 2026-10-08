# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render one episode of a trajectory pickle by its recorded states (no physics) to an mp4 from the task camera.

Works for the source demos and for retargeted pickles alike, so the two can be compared side by side.

  python scripts/retarget/render_states.py --pickle <pkl> --episode 0 --out ep0.mp4 --headless --enable_cameras
"""

import argparse
import os
import pickle
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--pickle", required=True)
ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--camera", default="third_person_camera")
ap.add_argument("--size", default="480x640", help="HxW")
ap.add_argument("--fps", type=int, default=60)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import dexverse.tasks  # noqa: E402,F401
from dexverse.tasks.utils import prune_stale_obs_refs  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def to_torch(x, dev):
    if isinstance(x, dict):
        return {k: to_torch(v, dev) for k, v in x.items()}
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)


def main():
    with open(args.pickle, "rb") as f:
        data = pickle.load(f)
    task, robot_type = data["task"], data["robot_type"]
    ep = data["episodes"][args.episode]
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg = type(cfg)(robot_type=robot_type)
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
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    camera = env.scene[args.camera]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=7, macro_block_size=8)
    states = ep["states"]
    for t, st in enumerate(states):
        env.scene.reset_to(to_torch(st, env.device), env_ids=None, is_relative=True)
        env.sim.forward()
        env.sim.render()
        env.scene.update(dt=0.0)
        camera.update(dt=0.0, force_recompute=True)
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        writer.append_data(rgb)
    writer.close()
    print(f"RESULT render {task} {robot_type} ep{args.episode}: {len(states)} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
