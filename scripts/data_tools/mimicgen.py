# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MimicGen / DexMimicGen for DexVerse: recorded pickles -> more successful demos -> LeRobot dataset.

    scripts/teleop_tools/mimicgen.sh <pickle or folder> [...] --num_demos 30 [--subtasks 2] [--seed 1]

Three steps on this machine (headless Isaac Sim, except the annotation when the kinematics cache exists):

1. ``scripts/mimic/annotate_pickles.py``: palm / object poses and subtask signals of the source demos, computed
   from their recorded per-step states (``--replay``: by replaying them)
   -> ``<datasets_dir>/mimicgen/<task>-<robot_type>/<stamp>/annotated.hdf5``
2. ``scripts/mimic/generate_demos.py``: Isaac Lab Mimic generation until ``--num_demos`` generated demos succeed
   -> ``.../<stamp>/generated.pkl`` (trajectory pickle, same schema as record_demos)
3. ``scripts/teleop_tools/pickle_to_lerobot.sh`` on that pickle (skip with ``--no_lerobot``)
   -> LeRobot dataset ``<datasets_dir>/<task>-<robot_type>-mimicgen`` (kept apart from the human demos)
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import pickle
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
_SPOOL_PY = os.path.join(_REPO, "source", "dexverse", "dexverse", "data_collection", "lerobot_spool.py")


def _load_spool():
    spec = importlib.util.spec_from_file_location("_dexverse_lerobot_spool", _SPOOL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="trajectory pickles or folders (one task + robot type)")
    ap.add_argument("--num_demos", type=int, required=True, help="successful generated demos to produce")
    ap.add_argument("--robot_type", default=None, help="only use the pickles of this robot type (a task folder can "
                    "hold several)")
    ap.add_argument("--subtasks", type=int, default=2, choices=(1, 2))
    ap.add_argument("--second_ref", default="auto",
                    help="object frame of the second subtask (auto: success_marker if the task has one, else object)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--replay", action="store_true",
                    help="annotate by replaying the demos in the simulator instead of reading their recorded states")
    ap.add_argument("--action_noise", type=float, default=0.002)
    ap.add_argument("--max_num_failures", type=int, default=1000)
    ap.add_argument("--datasets_dir", default=None, help="default: same as record_demos")
    ap.add_argument("--no_lerobot", action="store_true", help="stop after the generated pickle")
    args = ap.parse_args(argv)

    S = _load_spool()
    pickles = []
    for p in args.inputs:
        pickles += sorted(glob.glob(os.path.join(p, "**", "*.pkl"), recursive=True)) if os.path.isdir(p) else [p]
    if not pickles:
        raise SystemExit("[mimicgen] no pickles found")
    kinds = {}
    for p in pickles:
        with open(p, "rb") as f:
            head = pickle.load(f)
        kinds[p] = (head.get("task") or head.get("env_name"), head.get("robot_type"))
    if args.robot_type:
        pickles = [p for p in pickles if kinds[p][1] == args.robot_type]
        if not pickles:
            raise SystemExit(f"[mimicgen] no pickles of robot type {args.robot_type}")
    found = sorted({kinds[p] for p in pickles}, key=str)
    if len(found) != 1:
        raise SystemExit(f"[mimicgen] the pickles mix tasks / robot types {found}; pass --robot_type or one folder")
    task, robot = found[0]
    print(f"[mimicgen] {len(pickles)} pickles of {task} / {robot}", flush=True)
    datasets_dir = args.datasets_dir or S.default_datasets_dir()
    name = S.default_dataset_name(task, robot)
    run_dir = os.path.join(datasets_dir, "mimicgen", name, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    python = os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"), "_isaac_sim", "python.sh")
    annotated = os.path.join(run_dir, "annotated.hdf5")
    generated = os.path.join(run_dir, "generated.pkl")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    def run(cmd, log_name, product):
        """Run one Isaac Sim step; success = its product file exists (Isaac Sim exit codes are unreliable)."""
        log = os.path.join(run_dir, log_name)
        print(f"[mimicgen] {' '.join(cmd)}\n[mimicgen]   log: {log}", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                    cwd=_REPO, bufsize=1)
            for line in proc.stdout:
                f.write(line)
                if line.startswith(("[mimic]", "Traceback")) or "successful demos generated" in line:
                    print("  " + line.rstrip(), flush=True)
            rc = proc.wait()
        if not os.path.exists(product):
            raise SystemExit(f"[mimicgen] step failed (exit {rc}, no {os.path.basename(product)}), see {log}")

    run([python, os.path.join(_REPO, "scripts", "mimic", "annotate_pickles.py"), *pickles, "--output", annotated,
         "--subtasks", str(args.subtasks), "--headless", *(["--replay"] if args.replay else [])],
        "1_annotate.log", os.path.splitext(annotated)[0] + ".json")
    run([python, os.path.join(_REPO, "scripts", "mimic", "generate_demos.py"), "--input", annotated,
         "--num_demos", str(args.num_demos), "--output_pickle", generated, "--seed", str(args.seed),
         "--action_noise", str(args.action_noise), "--max_num_failures", str(args.max_num_failures),
         "--second_ref", args.second_ref, "--headless"], "2_generate.log", generated)
    if args.no_lerobot:
        print(f"[mimicgen] generated pickle: {generated}")
        return 0
    root = os.path.join(datasets_dir, f"{name}-mimicgen")
    cmd = [os.path.join(_REPO, "scripts", "teleop_tools", "pickle_to_lerobot.sh"), generated, "--lerobot_root", root,
           "--datasets_dir", datasets_dir]
    print(f"[mimicgen] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=_REPO, env=env)


if __name__ == "__main__":
    sys.exit(main())
