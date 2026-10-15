# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch driver for the reset-distribution audit.

Runs ``sample_reset_distributions.py`` once per task (one Isaac process each,
round-robin over the given GPUs), then aggregates every task's ``stats.json``
into ``<out_dir>/report.md``. Plain python -- no Isaac import here.

    python scripts/env_tools/run_reset_audit.py --tasks baseline --gpus 1,3,5
    python scripts/env_tools/run_reset_audit.py --tasks Dexverse-PushT-v0 Dexverse-OpenLaptop-v0 --gpus 5
    python scripts/env_tools/run_reset_audit.py --report_only            # re-aggregate existing results
    python scripts/env_tools/run_reset_audit.py --replot                 # re-run stats/plots offline, then report

``--tasks all`` enumerates every registered ``Dexverse-*`` id in a helper Isaac
process (slow start-up, ~1 min).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "source" / "dexverse"
SAMPLER = Path(__file__).resolve().parent / "sample_reset_distributions.py"

sys.path.insert(0, str(SRC_DIR))
from dexverse.analysis.reset_audit import analyze_task_dir, plot_grid, write_report  # noqa: E402
from dexverse.benchmark import baseline_tasks  # noqa: E402

_LIST_ENVS_SNIPPET = r"""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import dexverse.tasks, gymnasium as gym
print("\n".join(sorted(s.id for s in gym.registry.values() if s.id.startswith("Dexverse-"))))
app.close()
"""


def _list_all_tasks(python: str) -> list[str]:
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([python, "-c", _LIST_ENVS_SNIPPET], env=env, capture_output=True, text=True, check=False)
    tasks = [line.strip() for line in out.stdout.splitlines() if line.strip().startswith("Dexverse-")]
    if not tasks:
        raise SystemExit(f"could not enumerate tasks:\n{out.stderr[-2000:]}")
    return tasks


def _run_one(task: str, gpu: str, args, log_dir: Path) -> tuple[str, int, float]:
    log_path = log_dir / f"{task}.log"
    cmd = [
        args.python,
        str(SAMPLER),
        "--task",
        task,
        "--num_samples",
        str(args.num_samples),
        "--num_envs",
        str(args.num_envs),
        "--settle_steps",
        str(args.settle_steps),
        "--probe_resets",
        str(args.probe_resets),
        "--seed",
        str(args.seed),
        "--out_dir",
        str(args.out_dir),
        "--headless",
    ]
    if args.robot_type:
        cmd += ["--robot_type", args.robot_type]
    env = dict(os.environ)
    env.pop("DISPLAY", None)  # forwarded X breaks headless GLX init
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(cmd) + f"\n(CUDA_VISIBLE_DEVICES={gpu})\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return task, proc.returncode, time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=["baseline"], help="'baseline', 'all', or explicit gym ids.")
    parser.add_argument("--version", choices=["v0", "v1", "all"], default="v0", help="Version selected by --tasks baseline.")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids; one task per GPU at a time.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Interpreter with Isaac Sim / Isaac Lab.")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--settle_steps", type=int, default=30)
    parser.add_argument("--probe_resets", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot_type", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs/reset_audit")
    parser.add_argument("--skip_existing", action="store_true", help="Skip tasks that already have samples.npz.")
    parser.add_argument("--report_only", action="store_true", help="Only aggregate existing stats.json into report.md.")
    parser.add_argument("--replot", action="store_true", help="Re-run stats/flags/plots offline for existing samples, then report.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    args.out_dir = out_dir
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.tasks == ["baseline"]:
        tasks = list(baseline_tasks(args.version))
    elif args.tasks == ["all"]:
        tasks = _list_all_tasks(args.python)
    else:
        tasks = list(args.tasks)

    failures: dict[str, str] = {}
    if args.replot:
        for t in tasks:
            d = out_dir / t
            if (d / "samples.npz").is_file():
                try:
                    stats = analyze_task_dir(d)
                    print(f"[audit] {t}: {', '.join(f['flag'] for f in stats['flags']) or 'no flags'}")
                except Exception as exc:  # noqa: BLE001
                    failures[t] = f"replot failed: {exc}"
                    print(f"[audit] {t}: replot failed: {exc}")
    elif not args.report_only:
        todo = [t for t in tasks if not (args.skip_existing and (out_dir / t / "samples.npz").is_file())]
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        print(f"[audit] {len(todo)} task(s) on GPUs {gpus} with {args.python}")
        # one worker per GPU; each worker pulls the next task from the shared queue
        queue = list(todo)
        results = []

        def worker(gpu: str):
            while queue:
                task = queue.pop(0)
                print(f"[audit] gpu {gpu}: {task} ...", flush=True)
                res = _run_one(task, gpu, args, log_dir)
                results.append(res)
                print(f"[audit] gpu {gpu}: {task} -> rc={res[1]} ({res[2] / 60:.1f} min)", flush=True)

        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(worker, gpus))
        for task, rc, _ in results:
            if rc != 0 or not (out_dir / task / "stats.json").is_file():
                tail = ""
                lp = log_dir / f"{task}.log"
                if lp.is_file():
                    lines = lp.read_text(errors="replace").splitlines()
                    tail = " | ".join(lines[-3:])[-400:]
                failures[task] = f"exit code {rc}; see logs/{task}.log -- {tail}"

    report = write_report([out_dir / t for t in tasks], out_dir / "report.md", failures=failures)
    print(f"[audit] report: {report}")
    try:
        grid = plot_grid([out_dir / t for t in tasks], out_dir / "grid.png", title=f"Reset positions (top-down) -- {out_dir.name}")
        print(f"[audit] grid: {grid}")
    except Exception as exc:  # noqa: BLE001
        print(f"[audit] grid failed: {exc}")
    if failures:
        print(f"[audit] {len(failures)} task(s) failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
