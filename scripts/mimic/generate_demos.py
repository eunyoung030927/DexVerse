# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate demos with Isaac Lab Mimic from an annotated dataset and save them as a DexVerse trajectory pickle.

Runs the Isaac Lab Mimic data generator (object-centric segment transform + stitching, fingers copied from the
source demos) in the floating-hand Mimic env until ``--num_demos`` generated demos succeed, then writes them as a
trajectory pickle (initial scene state + actions, same schema as record_demos) that pickle_to_lerobot.sh turns
into a LeRobot dataset.

    /workspace/isaaclab/_isaac_sim/python.sh scripts/mimic/generate_demos.py --input <annotated.hdf5> \
        --num_demos 30 --output_pickle <generated.pkl>
"""

import argparse
import json
import os
import pickle
import random

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--input", required=True, help="annotated HDF5 from annotate_pickles.py (its .json sidecar too)")
ap.add_argument("--num_demos", type=int, required=True, help="number of SUCCESSFUL generated demos to keep")
ap.add_argument("--output_pickle", required=True)
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument("--seed", type=int, default=1, help="datagen seed (source selection, noise)")
ap.add_argument("--env_seed", type=int, default=None, help="env seed (scene randomization); default: datagen seed")
ap.add_argument("--max_num_failures", type=int, default=1000)
ap.add_argument("--num_success_steps", type=int, default=10,
                help="consecutive success steps a generated demo must hold (record_demos default)")
ap.add_argument("--subtasks", type=int, default=None, choices=(1, 2), help="override the annotation sidecar")
ap.add_argument("--second_ref", default="auto",
                help="object frame of the second subtask; auto: the scene's success_marker (pour / place "
                "target) if it has one, else object")
ap.add_argument("--object_ref", default="auto",
                help="object frame of the first (or only) subtask; auto: object if the scene has one, else the task "
                "articulation (e.g. the laptop)")
ap.add_argument("--keep_failed", action="store_true",
                help="also export failed attempts (<output>_failed.hdf5) for analysis")
ap.add_argument("--action_noise", type=float, default=0.002, help="noise on the wrist joint targets (m / rad)")
AppLauncher.add_app_launcher_args(ap)
# enable_cameras: rendering every step changes the physics outcome; keep it as on the recording / replay path
ap.set_defaults(device="cpu", headless=True, enable_cameras=True)
args = ap.parse_args()

with open(os.path.splitext(args.input)[0] + ".json", encoding="utf-8") as f:
    SIDECAR = json.load(f)
TASK, ROBOT = SIDECAR["task"], SIDECAR["robot_type"]

app = AppLauncher(args).app

import sys  # noqa: E402
import traceback  # noqa: E402


def _die(exc_type, exc, tb):
    """An error after the app started: print it and exit at once (closing Kit after a failure can hang)."""
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = _die

import asyncio  # noqa: E402

import dexverse.tasks  # noqa: E402, F401
import isaaclab_tasks  # noqa: E402, F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dexverse.mimic.env_setup import (  # noqa: E402
    attach_mimic_cfg,
    build_env_cfg,
    layout_sides,
    make_env,
    resolve_object_ref,
    resolve_success_term,
)
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402
import isaaclab_mimic.datagen.generation as G  # noqa: E402
from isaaclab_mimic.datagen.generation import setup_async_generation  # noqa: E402

out_pickle = os.path.abspath(args.output_pickle)
os.makedirs(os.path.dirname(out_pickle), exist_ok=True)
gen_hdf5 = os.path.splitext(out_pickle)[0] + ".hdf5"

env_seed = args.env_seed if args.env_seed is not None else args.seed
env_cfg, success_term = build_env_cfg(TASK, ROBOT, device=args.device, num_envs=args.num_envs, seed=env_seed)
num_subtasks = int(args.subtasks or SIDECAR.get("subtasks", 2))
second_ref = args.second_ref
if second_ref == "auto":
    # The pour / place target is the marker, not the object pose at the end of the grasp subtask (GraspCup, public
    # Shadow demos: 5/23 attempts succeed with success_marker vs 4/58 with object).
    second_ref = "success_marker" if getattr(env_cfg.scene, "success_marker", None) is not None else "object"
if args.num_envs != 1:
    raise SystemExit("[mimic] --num_envs must be 1: every attempt starts from env.sim.reset(), which resets all envs")
object_ref = resolve_object_ref(env_cfg, args.object_ref)
print(f"[mimic] subtasks {num_subtasks}, object_ref {object_ref}, second_ref {second_ref}", flush=True)
attach_mimic_cfg(env_cfg, layout_sides(ROBOT), num_subtasks=num_subtasks, second_ref=second_ref,
                 object_ref=object_ref,
                 num_trials=args.num_demos, seed=args.seed, action_noise=args.action_noise,
                 max_num_failures=args.max_num_failures)
rec = ActionStateRecorderManagerCfg()
rec.dataset_export_dir_path = os.path.dirname(gen_hdf5)
rec.dataset_filename = os.path.splitext(os.path.basename(gen_hdf5))[0]
rec.dataset_export_mode = (DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES if args.keep_failed
                           else DatasetExportMode.EXPORT_SUCCEEDED_ONLY)
env_cfg.datagen_config.generation_keep_failed = bool(args.keep_failed)
env_cfg.recorders = rec
env = make_env(env_cfg)
success_term = resolve_success_term(env, success_term)


class _HeldSuccess:
    """Success only once the task success held for ``n`` consecutive steps (record_demos' rule), so every demo the
    generator keeps also counts as a success when pickle_to_lerobot replays it. Reset at every attempt."""

    def __init__(self, func, n):
        self.func, self.n, self.count = func, int(n), None

    def reset(self):
        self.count = None

    def __call__(self, env, **params):
        s = self.func(env, **params).bool()
        if self.count is None or self.count.shape != s.shape:
            self.count = torch.zeros(s.shape, dtype=torch.int64, device=s.device)
        self.count = torch.where(s, self.count + 1, torch.zeros_like(self.count))
        return self.count >= self.n


held = _HeldSuccess(success_term.func, args.num_success_steps)
success_term.func = held


def _env_loop(env, reset_queue, action_queue, loop):
    """Isaac Lab Mimic's ``env_loop``, except that every attempt starts from a freshly reset simulation
    (``env.sim.reset()`` before ``env.reset()``), as record_demos does between demos. PhysX keeps contact / solver
    state across ``env.reset()``: without the sim reset a generated attempt cannot be replayed from its recorded
    initial state + actions (the replay diverges at the first contacts)."""
    env_ids = torch.tensor([0], dtype=torch.int64, device=env.device)
    prev = 0
    with torch.inference_mode():
        while True:
            while action_queue.qsize() != env.num_envs:
                loop.run_until_complete(asyncio.sleep(0))
                while not reset_queue.empty():
                    env_ids[0] = reset_queue.get_nowait()
                    env.sim.reset()
                    env.reset(env_ids=env_ids)
                    held.reset()
                    reset_queue.task_done()
            actions = torch.zeros(env.action_space.shape)
            for _ in range(env.num_envs):
                env_id, action = loop.run_until_complete(action_queue.get())
                actions[env_id] = action
            env.step(actions)
            for _ in range(env.num_envs):
                action_queue.task_done()
            if prev != G.num_attempts:
                prev = G.num_attempts
                rate = 100.0 * G.num_success / max(G.num_attempts, 1)
                print(f"{G.num_success}/{G.num_attempts} ({rate:.1f}%) successful demos generated by mimic", flush=True)
                if G.num_success >= env.cfg.datagen_config.generation_num_trials:
                    break
            if env.sim.is_stopped():
                break
    env.close()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.reset()
comps = setup_async_generation(env=env, num_envs=args.num_envs, input_file=os.path.abspath(args.input),
                               success_term=success_term)
tasks = asyncio.ensure_future(asyncio.gather(*comps["tasks"]))
try:
    _env_loop(env, comps["reset_queue"], comps["action_queue"], comps["event_loop"])
finally:
    tasks.cancel()
    try:
        comps["event_loop"].run_until_complete(tasks)
    except asyncio.CancelledError:
        pass


def _to_numpy(x):
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


handler = HDF5DatasetFileHandler()
handler.open(gen_hdf5)
episodes = []
for name in handler.get_episode_names():
    ep = handler.load_episode(name, "cpu")
    actions = ep.data["actions"].detach().cpu().numpy().astype(np.float32)
    episodes.append({"episode_index": len(episodes), "episode_name": f"demo_{len(episodes)}", "reset_id": None,
                     "initial_state": _to_numpy(ep.data["initial_state"]), "goal_pose": None,
                     "actions": actions, "success": True, "num_steps": int(actions.shape[0])})
handler.close()
payload = dict(SIDECAR["pickle_metadata"])
payload.update({
    "record_state": False,
    "generator": {"type": "isaaclab_mimic", "annotated_dataset": os.path.abspath(args.input),
                  "subtasks": num_subtasks, "object_ref": object_ref, "second_ref": second_ref, "seed": args.seed,
                  "env_seed": env_seed,
                  "action_noise": args.action_noise, "attempts": int(G.num_attempts),
                  "successes": int(G.num_success), "sources": SIDECAR.get("sources")},
    "episodes": episodes,
})
with open(out_pickle, "wb") as f:
    pickle.dump(payload, f)
rate = 100.0 * G.num_success / max(G.num_attempts, 1)
print(f"[mimic] generated {len(episodes)} demos ({G.num_success}/{G.num_attempts} attempts succeeded, {rate:.0f}%) "
      f"-> {out_pickle}", flush=True)
app.close()
