# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check task configurations and bounded reset/step smoke tests in this checkout.

Run with the existing Isaac Lab Python environment. Source selection is explicit
so an editable installation pointing at a different checkout cannot mask errors.
Use --configs_only for every registered task, or --tasks followed by task IDs to
run a small number of resets with the camera-free state preset.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "dexverse"))

# Isaac Sim teardown/reinitialization can stall when multiple scenes run in one
# process. Dispatch physics checks to bounded workers before launching Kit.
if "--configs_only" not in sys.argv:
    dispatch = argparse.ArgumentParser(add_help=False)
    dispatch.add_argument("--tasks", nargs="+")
    dispatch.add_argument("--report", type=Path, default=Path("outputs/task_upgrades/checks.json"))
    selection, remaining = dispatch.parse_known_args()
    if selection.tasks and len(selection.tasks) > 1:
        selection.report.parent.mkdir(parents=True, exist_ok=True)
        combined = {"tasks": {}}
        for task in selection.tasks:
            task_report = selection.report.parent / f"{selection.report.stem}_{task}.json"
            command = [sys.executable, __file__, *remaining, "--tasks", task, "--report", str(task_report)]
            try:
                completed = subprocess.run(command, timeout=150, check=False)
                if task_report.is_file():
                    result = json.loads(task_report.read_text())
                    combined["source"] = result["source"]
                    combined["tasks"].update(result["tasks"])
                else:
                    combined["tasks"][task] = {"status": "failed", "error": f"worker exited {completed.returncode}"}
            except subprocess.TimeoutExpired:
                if task_report.is_file():
                    result = json.loads(task_report.read_text())
                    combined["tasks"].update(result["tasks"])
                    combined["tasks"][task]["cleanup_timeout"] = True
                else:
                    combined["tasks"][task] = {"status": "failed", "error": "worker exceeded 150 seconds"}
            selection.report.write_text(json.dumps(combined, indent=2) + "\n")
        raise SystemExit(int(any(r["status"] == "failed" for r in combined["tasks"].values())))

from isaaclab.app import AppLauncher  # noqa: E402
from dexverse.teleop_utils.debug_visualization import (  # noqa: E402
    add_debug_visualization_args, configure_v1_debug_visualization,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tasks", nargs="+", default=None)
parser.add_argument("--configs_only", action="store_true")
parser.add_argument("--resets", type=int, default=3)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42, help="Nonnegative validation seed; reset i uses seed+i.")
parser.add_argument(
    "--check_seed_repeatability", action="store_true",
    help="Verify each seeded scene/task reset repeats after an intervening different-seed reset.",
)
parser.add_argument("--report", type=Path, default=Path("outputs/task_upgrades/checks.json"))
parser.add_argument("--record_fixture", type=Path, default=None, help="Save the last reset as a replay fixture.")
parser.add_argument("--inspect_visuals", action="store_true", help="Record marker visibility/purpose and retargeter debug flags.")
parser.add_argument(
    "--image", type=Path, default=None, help="Save a third-person RGB frame (requires --enable_cameras)."
)
AppLauncher.add_app_launcher_args(parser)
add_debug_visualization_args(parser)
args = parser.parse_args()
if args.seed < 0 or args.seed + max(100, args.resets) + args.resets > 2**32 - 1:
    parser.error("--seed must leave room for the validation/reset seed range in [0, 2**32-1]")
if args.resets < 1 or args.steps < 1:
    parser.error("--resets and --steps must be positive")
app = AppLauncher(args).app

import dexverse  # noqa: E402
import dexverse.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg, prune_stale_obs_refs, strip_camera_cfgs  # noqa: E402
from dexverse.benchmark import action_layout, task_identity  # noqa: E402
from dexverse.teleop_utils.episode_task_state import (  # noqa: E402
    capture_episode_task_state,
    restore_episode_task_state,
    reset_managers_after_state_restore,
)


def compare_state(left, right):
    """Compare reset scene/command/task tensors, allowing small float noise."""
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare_state(left[key], right[key])
    else:
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def main():
    source = Path(dexverse.__file__).resolve()
    if not source.is_relative_to(REPO_ROOT):
        raise RuntimeError(f"Wrong checkout imported: {source}")
    tasks = args.tasks or sorted(s.id for s in gym.registry.values() if s.id.startswith("Dexverse-"))
    report = {"source": str(source), "configs_only": args.configs_only, "tasks": {}}
    failed = False
    for task in tasks:
        env = None
        try:
            cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
            if args.enable_debug_vis is not None:
                cfg = type(cfg)(enable_debug_vis=args.enable_debug_vis)
                cfg.sim.device = args.device
                cfg.scene.num_envs = args.num_envs
            configure_v1_debug_visualization(cfg, cues_in_rgb=args.cues_in_rgb)
            cfg.observation_preset = "state"
            cfg._apply_observation_preset("state")
            if not args.enable_cameras:
                strip_camera_cfgs(cfg)
                prune_stale_obs_refs(cfg)
            cfg.seed = args.seed
            cfg.sim.render_interval = cfg.decimation
            result = {
                "robot_type": cfg.robot_type, "status": "configured", "seed": args.seed,
                "enable_debug_vis": cfg.enable_debug_vis, "cues_in_rgb": args.cues_in_rgb,
            }
            report["tasks"][task] = result
            if not args.configs_only:
                env = gym.make(task, cfg=cfg).unwrapped
                terminations = {}
                reset_seeds = [args.seed + index for index in range(args.resets)]
                perturbation_seeds = [seed + max(100, args.resets) for seed in reset_seeds]
                for reset_index in range(args.resets):
                    env.reset(seed=reset_seeds[reset_index])
                    # Finite observations alone cannot detect goals accidentally
                    # placed on another cloned scene's table.
                    for name in env.command_manager.active_terms:
                        command = env.command_manager.get_term(name)
                        if (type(command).__name__ == "ObjectUniformPoseCommand"
                                and type(command).__module__.startswith("dexverse.baseline_v1.")
                                and command.cfg.use_world_frame):
                            ranges = command.cfg.ranges
                            low = torch.tensor([ranges.pos_x[0], ranges.pos_y[0], ranges.pos_z[0]], device=env.device)
                            high = torch.tensor([ranges.pos_x[1], ranges.pos_y[1], ranges.pos_z[1]], device=env.device)
                            local = command.command[:, :3] - env.scene.env_origins
                            assert ((local >= low - 1e-5) & (local <= high + 1e-5)).all(), (
                                f"{name}: sampled goal lies outside its own scene's configured ranges"
                            )
                    snapshot = capture_episode_task_state(env)
                    # A second reset changes random commands/choices. Restore
                    # scene and task state, as the sequential converter does.
                    scene_state = env.scene.get_state(is_relative=True)
                    env.reset(seed=perturbation_seeds[reset_index])
                    if args.check_seed_repeatability:
                        env.reset(seed=reset_seeds[reset_index])
                        compare_state(scene_state, env.scene.get_state(is_relative=True))
                        compare_state(snapshot, capture_episode_task_state(env))
                        # Still test state restoration from an unrelated reset.
                        env.reset(seed=perturbation_seeds[reset_index])
                    env.reset_to(scene_state, env_ids=None, is_relative=True)
                    reset_managers_after_state_restore(env)
                    skipped = restore_episode_task_state(env, snapshot)
                    if skipped:
                        raise RuntimeError(f"Task state did not round trip: {skipped}")
                    replayed = capture_episode_task_state(env)

                    compare_state(snapshot, replayed)
                    env.command_manager.compute(dt=0.0)
                    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
                    states = [env.scene.get_state(is_relative=True)]
                    actions = []

                    def assert_finite(value):
                        if isinstance(value, dict):
                            for child in value.values():
                                assert_finite(child)
                        elif isinstance(value, torch.Tensor) and value.is_floating_point():
                            assert torch.isfinite(value).all(), "Non-finite simulation state or observation"

                    for _ in range(args.steps):
                        obs, reward, *_ = env.step(action)
                        assert_finite(obs)
                        assert_finite(reward)
                        if args.record_fixture:
                            states.append(env.scene.get_state(is_relative=True))
                            actions.append(action[0].detach().cpu().numpy())
                        for name in env.termination_manager.active_terms:
                            count = int(env.termination_manager.get_term(name).sum().item())
                            terminations[name] = terminations.get(name, 0) + count
                if args.record_fixture:
                    if env.num_envs != 1:
                        raise ValueError("--record_fixture requires --num_envs 1")

                    def cpu(value):
                        if isinstance(value, dict):
                            return {key: cpu(child) for key, child in value.items()}
                        if isinstance(value, torch.Tensor):
                            return value.detach().cpu().numpy()
                        return value

                    fixture = {
                        "format": "dexverse_trajectory",
                        "schema_version": 5,
                        "task": task,
                        "env_name": task,
                        "robot_type": cfg.robot_type,
                        "seed": args.seed,
                        "reset_seed": reset_seeds[reset_index],
                        **task_identity(task),
                        "action_layout": action_layout(env),
                        "record_state": True,
                        "num_episodes": 1,
                        "episodes": [
                            {
                                "episode_index": 0,
                                "episode_name": "reset_smoke",
                                "success": False,
                                "initial_state": cpu(states[0]),
                                "task_state": cpu(snapshot),
                                "states": [cpu(state) for state in states],
                                "actions": actions,
                            }
                        ],
                    }
                    args.record_fixture.parent.mkdir(parents=True, exist_ok=True)
                    with args.record_fixture.open("wb") as stream:
                        pickle.dump(fixture, stream)
                if args.image:
                    import matplotlib.pyplot as plt

                    for _ in range(3):
                        env.sim.render()
                    camera = env.scene["third_person_camera"]
                    rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                    args.image.parent.mkdir(parents=True, exist_ok=True)
                    plt.imsave(args.image, rgb)
                    result["image"] = str(args.image)
                if args.inspect_visuals:
                    from pxr import Usd, UsdGeom
                    import omni.usd

                    stage = omni.usd.get_context().get_stage()
                    result["visual_markers"] = {
                        str(prim.GetPath()): {
                            "purpose": UsdGeom.Imageable(prim).ComputePurpose(),
                            "visibility": UsdGeom.Imageable(prim).ComputeVisibility(),
                        }
                        for prim in Usd.PrimRange(stage.GetPseudoRoot())
                        if prim.IsA(UsdGeom.PointInstancer)
                    }
                    result["retargeter_debug_vis"] = [
                        getattr(rtg, "debug_vis", None)
                        for device in getattr(getattr(cfg, "teleop_devices", None), "devices", {}).values()
                        for rtg in getattr(device, "retargeters", [])
                    ]
                result.update(
                    status="stepped",
                    resets=args.resets,
                    steps_per_reset=args.steps,
                    num_envs=env.num_envs,
                    action_dim=env.action_manager.total_action_dim,
                    termination_counts=terminations,
                    reset_seeds=reset_seeds,
                    perturbation_seeds=perturbation_seeds,
                    seed_repeatability="passed" if args.check_seed_repeatability else "not_checked",
                )
            report["tasks"][task] = result
            print(f"[task-upgrades] PASS {task}: {result}", flush=True)
        except Exception:
            failed = True
            report["tasks"][task] = {"status": "failed", "error": traceback.format_exc()}
            print(f"[task-upgrades] FAIL {task}\n{traceback.format_exc()}", flush=True)
        finally:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            # A physics worker owns just one scene. Kit shutdown tears it down;
            # do not separately destroy and recreate the simulation context.
    return int(failed)


try:
    exit_code = main()
finally:
    if args.configs_only:
        app.close()
# Physics workers have already flushed their result and own no recordings.
# Kit teardown can hang on marker subscriptions; terminate this bounded worker
# directly after flushing stdout/stderr, letting the OS release its GPU context.
if not args.configs_only:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
raise SystemExit(exit_code)
