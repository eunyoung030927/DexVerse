#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained Diffusion Policy checkpoint closed-loop in simulation.

Usage::

    python scripts/diffusion/eval_online.py \
        --task Dexverse-GraspKettle-v0 \
        --ckpt runs/dp_Dexverse-GraspKettle-v0/best.pt \
        --output_dir runs/dp_Dexverse-GraspKettle-v0/eval \
        --observation_preset state --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", required=True)
parser.add_argument("--ckpt", required=True, help="Checkpoint from scripts/diffusion/train.py.")
parser.add_argument("--output_dir", default="runs/diffusion_eval")
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--sample_steps", type=int, default=20, help="DDPM denoising steps per replan.")
parser.add_argument("--replan_interval", type=int, default=4, help="Env steps between replans.")
parser.add_argument("--observation_preset", default=None,
                    help="Observation preset to narrow the live env to. Must match the --obs-groups "
                         "preset the demos were replayed with (typically 'state').")
parser.add_argument("--reset_from_dataset", default=None,
                    help="Training HDF5 to pin each episode's start to a recorded initial_state, "
                         "instead of the env's reset randomization. Diagnostic.")
parser.add_argument("--video", action="store_true", help="Record one MP4 per episode under <output_dir>/videos/.")
parser.add_argument("--video_length", type=int, default=None, help="Frames per recorded episode (default: --max_steps).")
parser.add_argument("--json_path", default=None, help="Task-specific object/scene JSON.")
parser.add_argument("--policy_device", default=None,
                    help="Device for policy inference (default: --device); physics stays on --device.")
parser.add_argument("--seed", type=int, default=None, help="Env seed (use distinct seeds for parallel shards).")
parser.add_argument("--robot_type", default=None,
                    help="Override the task's hand, e.g. floating_allegro_right (match the training demos).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# DexVerse scenes declare tiled cameras even when the policy is state-only, and
# Isaac Sim refuses to spawn them unless cameras are enabled.
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    from dexverse.IL.diffusion.config import EvalConfig
    from dexverse.IL.diffusion.online_eval import evaluate

    fields = {field: getattr(args, field) for field in EvalConfig.__dataclass_fields__ if hasattr(args, field)}
    evaluate(EvalConfig(**fields))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # simulation_app.close() can end the process before Python prints the traceback.
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
