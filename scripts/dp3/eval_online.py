#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained DP3 policy online in a DexVerse simulator.

Usage::

    python scripts/dp3/eval_online.py \\
        --ckpt runs/dp3_graspPan/best.pt \\
        --task Dexverse-GraspPan-v0 \\
        --observation_preset pointcloud \\
        --num_episodes 20

``--enable_cameras`` is auto-set when the preset includes a camera-driven
group (``pointcloud`` / ``3view_pointcloud``); pass ``--no_auto_enable_cameras``
if you need to override that behavior.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from isaaclab.app import AppLauncher

# Presets that need camera sensors live in the scene.
_CAMERA_PRESETS = {
    "rgb",
    "rgb_depth",
    "rgbd",
    "pointcloud",
    "3view_rgb",
    "3view_rgb_depth",
    "3view_rgbd",
    "3view_pointcloud",
}

parser = argparse.ArgumentParser(description="Evaluate DP3 policy online in DexVerse simulation.")
parser.add_argument("--ckpt", type=str, required=True, help="Path to DP3 training checkpoint")
parser.add_argument("--task", type=str, required=True, help="DexVerse task name")
parser.add_argument("--output_dir", type=str, default="runs/dp3_eval_online")
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--replan_interval", type=int, default=8)
parser.add_argument("--num_points", type=int, default=512)
parser.add_argument(
    "--observation_preset",
    type=str,
    default="pointcloud",
    help=(
        "Observation preset to apply to the env cfg before gym.make() "
        "(default: pointcloud). Must match the preset used when generating demos."
    ),
)
parser.add_argument(
    "--no_auto_enable_cameras",
    action="store_true",
    default=False,
    help="Disable the auto-setting of --enable_cameras based on the preset.",
)
parser.add_argument(
    "--json_path", type=str, default=None, help="Path to template JSON spec (for *Template environments)"
)
parser.add_argument(
    "--record_video", action="store_true", default=False, help="Record one MP4 per episode from a scene camera."
)
parser.add_argument(
    "--video_camera",
    type=str,
    default="third_person_camera",
    help="Name of the scene Camera sensor to render from (default: third_person_camera).",
)
parser.add_argument("--video_fps", type=int, default=30, help="Frames per second for written MP4s (default: 30).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force headless. Without this the app boots a windowed RTX pipeline and the
# first camera render inside env.reset() spins indefinitely on a display-less
# host. create_demo_files_sequential.py forces headless for the same reason.
args.headless = True

# Isaac Lab refuses to spawn Camera sensors unless --enable_cameras is set.
# Mirror create_demo_files_sequential.py's auto-enable behavior so the user doesn't have
# to remember the flag when picking a camera-driven preset.
if not args.no_auto_enable_cameras and args.observation_preset in _CAMERA_PRESETS:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    from dexverse.IL.dp3.config import DP3OnlineEvalConfig
    from dexverse.IL.dp3.online_eval import eval_main

    cfg = DP3OnlineEvalConfig(
        ckpt=args.ckpt,
        task=args.task,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        device=args.device,
        replan_interval=args.replan_interval,
        num_points=args.num_points,
        observation_preset=args.observation_preset,
        record_video=args.record_video,
        video_camera=args.video_camera,
        video_fps=args.video_fps,
    )
    if args.json_path is not None:
        cfg.json_path = args.json_path
    eval_main(cfg)


if __name__ == "__main__":
    main()
    simulation_app.close()
