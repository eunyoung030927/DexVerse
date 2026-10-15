# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build LeRobot datasets from recorded trajectory pickles by replaying them headless.

VR sessions record only the trajectory pickle (``record_demos --no_lerobot``): rendering the dataset cameras
inside an XR session is what stalls it. This tool replays every episode of each pickle in the simulator (initial
scene state + recorded actions, ``record_demos --replay_demos``), renders the cameras and writes the same LeRobot
dataset a live recording would have produced (``<datasets_dir>/<task>-<robot_type>``; episodes are appended).
An episode whose replay does not reach the task's success condition is dropped and reported.

    scripts/teleop_tools/pickle_to_lerobot.sh <pickle or folder> [...]          # e.g. /workspace/local/datasets/trajectories
    scripts/teleop_tools/pickle_to_lerobot.sh <...> --dry_run                   # list what would be converted

Folders are searched recursively for ``*.pkl``. Every converted pickle is logged in
``<datasets_dir>/lerobot_conversions.jsonl`` and skipped next time (``--force`` converts it again, which appends
its episodes a second time). The replayed pickle and the replay log go to
``<datasets_dir>/trajectories_replayed/`` next to the originals. Replay is open-loop, so convert on the machine
(and sim device) the demos were recorded on.

Run with Isaac Sim's python (the wrapper does); each pickle starts one headless Isaac Sim process.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
_SPOOL_PY = os.path.join(_REPO, "source", "dexverse", "dexverse", "data_collection", "lerobot_spool.py")
LEDGER = "lerobot_conversions.jsonl"


def _spool_module():
    spec = importlib.util.spec_from_file_location("lerobot_spool", _SPOOL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_pickles(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d != "trajectories_replayed")
                out += [os.path.join(root, f) for f in sorted(files) if f.endswith(".pkl")]
        elif p.endswith(".pkl") and os.path.isfile(p):
            out.append(p)
        else:
            print(f"[convert] skipping {p}: not a .pkl file or a folder")
    return list(dict.fromkeys(out))


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        d = pickle.load(f)
    eps = d.get("episodes", []) or []
    return {
        "task": d.get("task") or d.get("env_name"),
        "robot_type": d.get("robot_type"),
        "sim_device": d.get("sim_device"),
        "num_episodes": sum(1 for e in eps if len(e.get("actions", [])) > 0),
    }


def _key(path: str) -> dict:
    st = os.stat(path)
    return {"pickle": path, "size": st.st_size, "mtime": int(st.st_mtime)}


def load_ledger(datasets_dir: str) -> list[dict]:
    path = os.path.join(datasets_dir, LEDGER)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def already_converted(ledger: list[dict], path: str) -> dict | None:
    k = _key(path)
    for row in ledger:
        if all(row.get(f) == v for f, v in k.items()):
            return row
    return None


def replayed_paths(datasets_dir: str, pickle_path: str) -> tuple[str, str]:
    traj = os.path.join(datasets_dir, "trajectories")
    rel = os.path.relpath(pickle_path, traj) if pickle_path.startswith(traj + os.sep) else os.path.basename(pickle_path)
    stem = os.path.splitext(rel)[0]
    base = os.path.join(datasets_dir, "trajectories_replayed", stem)
    return base + "_replay.pkl", base + "_replay.log"


def wait_for_writer(spool: str, timeout_s: float = 1800.0) -> str:
    """Wait until the spool writer of a finished session logs ``exit``; returns its last line."""
    log = os.path.join(spool, "writer.log")
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        try:
            with open(log, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            last = lines[-1] if lines else ""
        except OSError:
            pass
        if "] exit " in last:
            return last
        time.sleep(3)
    return last or "(no writer log)"


def main(argv=None) -> int:
    S = _spool_module()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="trajectory pickles and/or folders (searched recursively for *.pkl)")
    ap.add_argument("--datasets_dir", default=S.default_datasets_dir(),
                    help="base folder of the LeRobot datasets (default: %(default)s)")
    ap.add_argument("--lerobot_root", default=None,
                    help="exact dataset folder for ALL given pickles (default <datasets_dir>/<task>-<robot_type>)")
    ap.add_argument("--force", action="store_true", help="convert pickles that the ledger lists as done")
    ap.add_argument("--dry_run", action="store_true", help="only list what would be converted")
    args, extra = ap.parse_known_args(argv)  # extra args go to record_demos (e.g. --lerobot_image_size 240x320)

    datasets_dir = os.path.abspath(args.datasets_dir)
    pickles = find_pickles(args.paths)
    if not pickles:
        print("[convert] no pickles found")
        return 1
    ledger = load_ledger(datasets_dir)
    python = os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"), "_isaac_sim", "python.sh")
    record_demos = os.path.join(_REPO, "scripts", "record_demos.py")

    jobs = []
    for p in pickles:
        try:
            h = read_header(p)
        except Exception as exc:  # noqa: BLE001
            print(f"[convert] skip {p}: cannot read ({exc!r})")
            continue
        if not h["task"] or not h["robot_type"] or h["num_episodes"] == 0:
            print(f"[convert] skip {p}: task={h['task']} robot_type={h['robot_type']} episodes={h['num_episodes']}")
            continue
        done = already_converted(ledger, p)
        if done and not args.force:
            print(f"[convert] skip {p}: converted {done.get('time')} -> {done.get('lerobot_root')} (use --force)")
            continue
        root = args.lerobot_root or os.path.join(datasets_dir, S.default_dataset_name(h["task"], h["robot_type"]))
        jobs.append((p, h, os.path.abspath(root)))

    for p, h, root in jobs:
        print(f"[convert] {p}\n          {h['task']} / {h['robot_type']} / {h['num_episodes']} episode(s) -> {root}")
    if args.dry_run or not jobs:
        return 0

    summary = []
    for i, (p, h, root) in enumerate(jobs, 1):
        replay_pkl, replay_log = replayed_paths(datasets_dir, p)
        os.makedirs(os.path.dirname(replay_pkl), exist_ok=True)
        cmd = [python, record_demos, "--task", h["task"], "--robot_type", h["robot_type"], "--replay_demos", p,
               "--dataset_file", replay_pkl, "--num_demos", "0", "--headless", "--datasets_dir", datasets_dir,
               "--lerobot_root", root, *extra]
        print(f"\n[convert] ({i}/{len(jobs)}) replaying {os.path.basename(p)} ... log {replay_log}", flush=True)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        t0 = time.time()
        with open(replay_log, "w", encoding="utf-8") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=_REPO, env=env)
        with open(replay_log, encoding="utf-8", errors="replace") as f:
            text = f.read()
        n_ok = len(re.findall(r"^\[lerobot\] episode spooled", text, flags=re.M))
        failed = re.findall(r"^\[replay\] episode (\d+): no success", text, flags=re.M)
        print(f"[convert] replay exit {rc}: {n_ok}/{h['num_episodes']} episode(s) reproduced"
              + (f", not reproduced: {', '.join(failed)}" if failed else "") + f" ({time.time() - t0:.0f} s)")
        writer = wait_for_writer(S.spool_dir_for(root))
        print(f"[convert] writer: {writer}")
        row = {**_key(p), "task": h["task"], "robot_type": h["robot_type"], "episodes": h["num_episodes"],
               "reproduced": n_ok, "not_reproduced": [int(x) for x in failed], "lerobot_root": root,
               "replay_pickle": replay_pkl, "replay_log": replay_log, "exit_code": rc,
               "time": datetime.datetime.now().isoformat(timespec="seconds")}
        if rc == 0:
            with open(os.path.join(datasets_dir, LEDGER), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary.append(row)

    print("\n[convert] summary")
    for r in summary:
        print(f"  {os.path.basename(r['pickle'])}: {r['reproduced']}/{r['episodes']} -> {r['lerobot_root']}"
              + ("" if r["exit_code"] == 0 else f"  (FAILED, exit {r['exit_code']}, see {r['replay_log']})"))
    return 0 if all(r["exit_code"] == 0 for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
