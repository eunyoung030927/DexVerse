# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep videos of every retargeting output up to date, success or failure, for the web UI (webui.py).

Polls a runs root (the run_queue layout <runs>/<task>/<demos stem>/epNNN/) and renders what is new or changed
with render_retarget.py, one Isaac process per (task, robot type), one at a time:

  epNNN/viz/source.mp4                       the source demo episode
  epNNN/<target>/viz/kinonly_process.mp4     stage 2 joints replayed under physics (run_target's kinonly check)
  epNNN/<target>/viz/<tag>_process.mp4       stage 3 execution log, rollbacks included (re-rendered as it grows)
  epNNN/<target>/viz/<tag>_traj.mp4          stage 3 result, when there is no execution log (older runs)

Each mp4 has a sidecar .json (per-frame object error, success, drop events) the UI plots under the video.
No Isaac import here: run it with any python, it calls $PY (default: this interpreter) for the renders.

  python scripts/retarget/render_watch.py --runs <artifacts dir> [--gpu 0] [--once]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = os.environ.get("PY", sys.executable)
FAILED = {}                                            # out mp4 -> input mtime whose render failed (not retried)


def newer(src: Path, out: Path, min_age: float) -> bool:
    """out is missing, or older than src and at least min_age seconds old (a growing log is not re-rendered
    on every write). A render that failed is retried only once its input changes."""
    if FAILED.get(str(out)) == src.stat().st_mtime:
        return False
    if not out.exists():
        return True
    m = out.stat().st_mtime
    return src.stat().st_mtime > m and time.time() - m >= min_age


def pending(runs: Path, min_age: float):
    """{(task, robot_type): [job, ...]} of everything to (re)render."""
    todo = {}
    for ref in sorted(runs.glob("*/*/ep[0-9][0-9][0-9]/reference.npz")):
        ep = ref.parent
        task = ep.parents[1].name
        try:
            meta = json.loads(ref.with_suffix(".json").read_text())
        except Exception:
            continue
        out = ep / "viz" / "source.mp4"
        if newer(ref, out, 0):
            todo.setdefault((task, meta["source"]), []).append(f"source:{meta['demos']}#{meta['episode']}:{out}")
        for d in sorted(p for p in ep.iterdir() if p.is_dir() and (p / "kinematic.npz").exists()):
            jobs = []
            procs = {p.name[: -len("_process.pkl.gz")] for p in d.glob("*_process.pkl.gz")}
            for tag in sorted(procs):
                src, out = d / f"{tag}_process.pkl.gz", d / "viz" / f"{tag}_process.mp4"
                if newer(src, out, min_age):
                    jobs.append(f"traj:{src}:{out}")
            for src in sorted(d.glob("*_traj.pkl.gz")):
                tag = src.name[: -len("_traj.pkl.gz")]
                if src.is_symlink() or tag in procs:
                    continue
                out = d / "viz" / f"{tag}_traj.mp4"
                if newer(src, out, 0):
                    jobs.append(f"traj:{src}:{out}")
            if jobs:
                todo.setdefault((task, d.name), []).extend(jobs)
    return todo


def render(task, robot_type, jobs, gpu, log):
    cmd = [PY, "-u", str(HERE / "render_retarget.py"), "--task", task, "--robot_type", robot_type,
           "--headless", "--device", "cuda:0"]
    for j in jobs:
        cmd += ["--job", j]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": f"{REPO / 'source' / 'dexverse'}:{os.environ.get('PYTHONPATH', '')}"}
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n--- {time.strftime('%F %T')} {task} {robot_type}: {len(jobs)} jobs\n")
        f.flush()
        try:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env=env, timeout=3600)
        except subprocess.TimeoutExpired:
            f.write("--- TIMEOUT\n")
    for j in jobs:
        kind, src, out = j.split(":", 2)
        src = os.path.join(os.path.dirname(os.path.dirname(out)), "reference.npz") if kind == "source" else src
        src = os.path.join(src, "kinematic.npz") if kind == "kinematic" else src
        if not os.path.exists(out) or os.path.getmtime(out) < t0:
            FAILED[out] = os.path.getmtime(src)
            print(f"[{time.strftime('%F %T')}] FAILED {out} (see {log})", flush=True)
    print(f"[{time.strftime('%F %T')}] rendered {task} {robot_type}: {len(jobs)} jobs ({time.time() - t0:.0f}s)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="runs root ($DEXVERSE_RETARGET_RUNS of the queue)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--min_age", type=float, default=300.0, help="s between re-renders of a growing execution log")
    ap.add_argument("--log", default=None, help="render output log (default <runs>/../logs/render.log)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    runs = Path(a.runs)
    log = Path(a.log) if a.log else runs.parent / "logs" / "render.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        for (task, robot_type), jobs in pending(runs, a.min_age).items():
            render(task, robot_type, jobs, a.gpu, log)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
