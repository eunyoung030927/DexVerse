# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Annotate recorded DexVerse trajectory pickles for Isaac Lab Mimic (MimicGen / DexMimicGen).

Replays every episode headless in the floating-hand Mimic env (initial scene state + recorded actions, CPU
physics: bit-exact on the recording machine) and records the datagen info the generator needs -- palm poses,
object poses, the target palm poses of the actions and the subtask termination signals -- into an Isaac Lab HDF5
dataset. Episodes that do not end in success (or never lift the object, with two subtasks) are skipped.

    /workspace/isaaclab/_isaac_sim/python.sh scripts/mimic/annotate_pickles.py <pickle or folder> [...] \
        --output <annotated.hdf5>

Normally run through scripts/teleop_tools/mimicgen.sh. A ``<output>.json`` sidecar keeps the task, robot type,
source episodes and the source pickle metadata for ``generate_demos.py``.
"""

import argparse
import glob
import json
import os
import pickle

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("inputs", nargs="+", help="trajectory pickles or folders (searched recursively for *.pkl)")
ap.add_argument("--output", required=True, help="annotated Isaac Lab Mimic HDF5 to write")
ap.add_argument("--subtasks", type=int, default=2, choices=(1, 2),
                help="2: grasp (until the object is lifted) + the rest, both object-centric; 1: whole demo")
ap.add_argument("--max_episodes", type=int, default=None, help="annotate at most this many source episodes")
ap.add_argument("--num_success_steps", type=int, default=10,
                help="consecutive success steps that end a demo (record_demos default); the replay is cut there")
ap.add_argument("--hold_steps", type=int, default=120,
                help="steps the last action is held when success is not reached yet (record_demos replay default)")
AppLauncher.add_app_launcher_args(ap)
# enable_cameras: rendering every step changes the physics outcome; keep it as on the recording / replay path
ap.set_defaults(device="cpu", headless=True, enable_cameras=True)
args = ap.parse_args()


def _collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.pkl"), recursive=True))
        else:
            out.append(p)
    return out


pickles = _collect(args.inputs)
payloads = []
for p in pickles:
    with open(p, "rb") as f:
        payloads.append((p, pickle.load(f)))
if not payloads:
    raise SystemExit("[mimic] no pickles found")
tasks = {(d.get("task") or d.get("env_name"), d.get("robot_type")) for _, d in payloads}
if len(tasks) != 1:
    raise SystemExit(f"[mimic] pickles mix tasks / robot types: {sorted(tasks)}")
TASK, ROBOT = tasks.pop()
if not TASK or not ROBOT:
    raise SystemExit("[mimic] pickle metadata lacks task or robot_type")

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

import dexverse.tasks  # noqa: E402, F401
import isaaclab_tasks  # noqa: E402, F401
import torch  # noqa: E402
from dexverse.mimic.env_setup import (  # noqa: E402
    attach_mimic_cfg,
    build_env_cfg,
    layout_sides,
    make_env,
    resolve_success_term,
)
from dexverse.mimic.recorders import MimicAnnotationRecorderManagerCfg  # noqa: E402
from dexverse.teleop_utils.replay_teleop import _to_torch  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402

out_dir = os.path.dirname(os.path.abspath(args.output))
os.makedirs(out_dir, exist_ok=True)
env_cfg, success_term = build_env_cfg(TASK, ROBOT, device=args.device)
attach_mimic_cfg(env_cfg, layout_sides(ROBOT), num_subtasks=args.subtasks)
rec = MimicAnnotationRecorderManagerCfg()
rec.dataset_export_dir_path = out_dir
rec.dataset_filename = os.path.splitext(os.path.basename(args.output))[0]
rec.dataset_export_mode = DatasetExportMode.EXPORT_ALL
env_cfg.recorders = rec
env = make_env(env_cfg)
success_term = resolve_success_term(env, success_term)
env.reset()

exported, sources, skipped = 0, [], []
with torch.inference_mode():
    for path, payload in payloads:
        for ep_i, ep in enumerate(payload.get("episodes", [])):
            if args.max_episodes is not None and exported + len(skipped) >= args.max_episodes:
                break
            actions = torch.as_tensor(ep.get("actions", []), dtype=torch.float32)
            if actions.ndim != 2 or len(actions) == 0:
                continue
            env.sim.reset()
            env.recorder_manager.reset()
            env.reset_to(_to_torch(ep["initial_state"], env.device), None, is_relative=True)
            # Same rule as record_demos --replay_demos: the demo ends once success held for --num_success_steps
            # steps; after the last action it is held for up to --hold_steps more (public demos can be cut short).
            run, ok, n = 0, False, 0
            for t in range(len(actions) + args.hold_steps):
                env.step(actions[min(t, len(actions) - 1)].reshape(1, -1).to(env.device))
                n += 1
                run = run + 1 if bool(success_term.func(env, **success_term.params)[0]) else 0
                if run >= args.num_success_steps:
                    ok = True
                    break
            reason = "" if ok else f"replay never held success ({len(actions)} actions + {args.hold_steps} hold)"
            if ok and args.subtasks == 2:
                sig = env.recorder_manager.get_episode(0).data["obs"]["datagen_info"]["subtask_term_signals"]
                lifted = sig["object_lifted"]
                lifted = (torch.cat([torch.as_tensor(x).flatten() for x in lifted]) if isinstance(lifted, list)
                          else torch.as_tensor(lifted).flatten()).bool()
                if not bool(lifted.any()) or bool(lifted[0]):
                    ok, reason = False, "object_lifted never switches 0 -> 1"
            if ok:
                env.recorder_manager.set_success_to_episodes(
                    None, torch.tensor([[True]], dtype=torch.bool, device=env.device))
                env.recorder_manager.export_episodes()
                sources.append({"pickle": os.path.abspath(path), "episode": ep_i, "num_steps": n,
                                "demo": f"demo_{exported}"})
                exported += 1
                print(f"[mimic] annotated {os.path.basename(path)} episode {ep_i} ({n} of {len(actions)} steps)",
                      flush=True)
            else:
                skipped.append({"pickle": os.path.abspath(path), "episode": ep_i, "reason": reason})
                print(f"[mimic] SKIPPED {os.path.basename(path)} episode {ep_i}: {reason}", flush=True)
            env.recorder_manager.reset()

meta = {k: v for k, v in payloads[0][1].items() if k != "episodes"}
with open(os.path.splitext(args.output)[0] + ".json", "w", encoding="utf-8") as f:
    json.dump({"task": TASK, "robot_type": ROBOT, "subtasks": args.subtasks, "annotated": exported,
               "sources": sources, "skipped": skipped, "pickle_metadata": meta}, f, indent=1, default=str)
print(f"[mimic] {exported} annotated, {len(skipped)} skipped -> {os.path.abspath(args.output)}", flush=True)
env.close()
app.close()
