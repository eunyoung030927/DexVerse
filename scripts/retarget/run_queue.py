# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retarget whole tasks: a queue of (task, episode, target hand) items spread over the GPUs, PER_GPU at once.

Per episode: stage 1 once (in a free slot), then scripts/retarget/run_target.py per target hand (stage 2, SPIDER
with retries). Finished items are skipped, so the command can be stopped and re-run. The cluster runs it from
ondemand/retarget_worker.sbatch; it works the same on a plain GPU box.

  Targets: --hands are hand families; each becomes the source's robot_type with the hand swapped
  (floating_shadow_right -> floating_allegro_right, floating_shadow_bimanual -> floating_allegro_bimanual).
  A robot_type without a spec in dexverse/retarget/hands.py is skipped with a log line (none of the 5 hands today).

  Progress: one line per finished item in this log, and experiments/STATUS.md + experiments/<task>/<target>/
  (init_experiments.py, status_report.py), refreshed every 2 minutes; $DEXVERSE_EXPERIMENTS moves it (cluster).

  python scripts/retarget/run_queue.py --tasks Dexverse-GraspBleach-v1 --demos_root demos/demonstrations/v1 \
      --hands allegro leap wuji --gpus 0 --per_gpu 3

Memory: one SPIDER process with 1024 samples takes a few GB of VRAM; a CUDA OOM corrupts PhysX state.
"""

import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_target import episode_dir, experiments_root, runs_root  # noqa: E402

ALL_HANDS = ["allegro", "inspire", "leap", "sharpa", "wuji"]
FINAL = {"recorded", "failed_all_attempts", "stage2_failed"}


def supported_robot_types() -> set[str]:
    """Robot types with a retargeting hand spec; hands.py is pure Python, loaded without the dexverse package."""
    spec = importlib.util.spec_from_file_location("hands", REPO / "source/dexverse/dexverse/retarget/hands.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hands"] = mod                                 # dataclasses look the module up while decorating
    spec.loader.exec_module(mod)
    return set(mod.HANDS)


def find_demos(root: Path, task: str) -> Path:
    hits = sorted(root.glob(f"*/{task}/demos.pkl"))
    if not hits:
        raise FileNotFoundError(f"no {root}/*/{task}/demos.pkl")
    return hits[0]


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True, help="task ids, e.g. Dexverse-GraspBleach-v1")
    ap.add_argument("--demos_root", default=str(REPO / "demos" / "demonstrations" / "v1"),
                    help="<root>/<category>/<task>/demos.pkl (the release layout)")
    ap.add_argument("--episodes", type=int, nargs="+", default=list(range(50)), help="source episode indices")
    ap.add_argument("--hands", nargs="+", default=ALL_HANDS, help=f"target hand families (default {ALL_HANDS})")
    ap.add_argument("--gpus", nargs="+", default=["0"])
    ap.add_argument("--per_gpu", type=int, default=2, help="max concurrent sessions per GPU")
    ap.add_argument("--min_free_gb", type=float, default=0.0,
                    help="start a session on a GPU only if this much VRAM is free there (0: slots only). Sessions "
                         "started in the last 2 min count as --session_gb each, since they have not allocated yet")
    ap.add_argument("--session_gb", type=float, default=6.0, help="VRAM one SPIDER session ends up using")
    ap.add_argument("--max_attempts", type=int, default=3)
    ap.add_argument("--spider_args", default="")
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--retry_failed", action="store_true",
                    help="also re-run items that ended failed_all_attempts (new attempts if --max_attempts grew) "
                         "or stage2_failed (stage 2 is re-run; a crash there is a bug, not a verdict)")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    py = os.environ.get("PY", sys.executable)
    supported = supported_robot_types()
    final = {"recorded"} if a.retry_failed else FINAL
    free = [g for g in a.gpus for _ in range(a.per_gpu)]
    lock = threading.Condition()
    logdir = runs_root() / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    env_base = {**os.environ, "PY": py,
                "PYTHONPATH": f"{REPO / 'source' / 'dexverse'}:{os.environ.get('PYTHONPATH', '')}"}

    recent = []                                              # (time, gpu) of sessions started in the last 2 min

    def free_vram_gb():
        """Free VRAM per entry of --gpus (nvidia-smi rows in the same order as the job's GPUs); None if unknown."""
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout.split()
            rows = [float(x) / 1024 for x in out]
            return dict(zip(a.gpus, rows)) if len(rows) >= len(a.gpus) else None
        except Exception:  # noqa: BLE001
            return None

    def acquire():
        """A free slot, on the GPU with the most usable VRAM; waits while every GPU is full. Items are offered in
        queue order (task 1's episodes first, then task 2, ...), so the GPUs fill in that order without idling."""
        with lock:
            while True:
                if free:
                    if a.min_free_gb <= 0:
                        return free.pop(0)
                    now = time.time()
                    recent[:] = [(t, g) for t, g in recent if now - t < 120]
                    vram = free_vram_gb()
                    if vram is None:
                        return free.pop(0)
                    usable = {g: vram[g] - a.session_gb * sum(1 for _, x in recent if x == g) for g in set(free)}
                    best = max(usable, key=usable.get)
                    if usable[best] >= a.min_free_gb:
                        free.remove(best)
                        recent.append((now, best))
                        return best
                lock.wait(timeout=30)                        # re-check VRAM every 30 s

    def release(g):
        with lock:
            free.append(g)
            lock.notify()

    def run(cmd, gpu, logf, timeout):
        with open(logf, "a") as f:
            try:
                return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=REPO, timeout=timeout,
                                      env={**env_base, "CUDA_VISIBLE_DEVICES": str(gpu)}).returncode
            except subprocess.TimeoutExpired:
                return 124

    # (task, demos, source robot_type, episodes, targets)
    plan = []
    for task in a.tasks:
        demos = find_demos(Path(a.demos_root), task)
        with open(demos, "rb") as f:
            d = pickle.load(f)
        src, n_eps = d["robot_type"], len(d["episodes"])
        del d
        targets = []
        for hand in a.hands:
            rt = src.replace(src.split("_")[1], hand, 1)              # floating_<hand>_<right|left|bimanual>
            if rt == src:
                continue
            if rt not in supported:
                log(f"skip {task} -> {rt}: not supported by the retargeting yet (dexverse/retarget/hands.py)")
                continue
            targets.append(rt)
        eps = [e for e in a.episodes if e < n_eps]
        plan.append((task, demos, src, eps, targets))
        log(f"{task}: source {src}, {len(eps)} episodes x targets {targets}")

    stop = threading.Event()
    if not a.dry_run:
        import status_report  # noqa: PLC0415
        subprocess.run([py, str(HERE / "init_experiments.py"), "--demos_root", a.demos_root, "--tasks", *a.tasks,
                        "--hands", *a.hands])

        def report():
            try:
                status_report.write(experiments_root(), status_report.collect(runs_root()), runs_root())
            except Exception as e:  # noqa: BLE001  (a report error must not stop the queue)
                log(f"status report failed: {e}")

        def reporter():
            while not stop.wait(120):
                report()
        report()
        threading.Thread(target=reporter, daemon=True).start()

    def target_job(task, demos, ep, tgt, gpu):
        try:
            cmd = [py, "-u", HERE / "run_target.py", "--task", task, "--demos", demos, "--episode", ep, "--target", tgt,
                   "--max_attempts", a.max_attempts, "--spider_args", a.spider_args, "--seed_offset", a.seed_offset]
            run(cmd, gpu, logdir / f"{task}_ep{ep:03d}_{tgt}.log", 72 * 3600)
            st = episode_dir(task, str(demos), ep) / tgt / "status.json"
            log(f"{task} ep{ep} -> {tgt}: {json.load(open(st)).get('state') if st.exists() else 'no status'}")
        finally:
            release(gpu)

    threads = []
    for task, demos, src, eps, targets in plan:
        for ep in eps:
            todo = []
            for tgt in targets:
                st = episode_dir(task, str(demos), ep) / tgt / "status.json"
                if st.exists() and json.load(open(st)).get("state") in final:
                    continue
                todo.append(tgt)
            if not todo:
                continue
            ref = episode_dir(task, str(demos), ep) / "reference.npz"
            if not ref.exists():
                if a.dry_run:
                    log(f"would run stage 1 for {task} ep{ep}")
                else:
                    gpu = acquire()
                    log(f"stage 1 {task} ep{ep} on GPU {gpu}")
                    run([py, "-u", HERE / "stage1_reference.py", "--task", task, "--demos", demos, "--episode", ep,
                              "--headless", "--device", "cuda:0"], gpu, logdir / f"{task}_ep{ep:03d}_stage1.log", 3600)
                    release(gpu)
                    if not ref.exists():                   # (Isaac scripts can exit 0 after an exception)
                        log(f"stage 1 failed for {task} ep{ep} (see logs/)")
                        continue
            for tgt in todo:
                if a.dry_run:
                    log(f"would run {task} ep{ep} {src} -> {tgt}")
                    continue
                gpu = acquire()
                log(f"start {task} ep{ep} {src} -> {tgt} on GPU {gpu}")
                th = threading.Thread(target=target_job, args=(task, demos, ep, tgt, gpu))
                th.start()
                threads.append(th)
                time.sleep(10)                       # stagger Kit start-ups
    for th in threads:
        th.join()
    stop.set()
    if not a.dry_run:
        report()
    log("ALL DONE")


if __name__ == "__main__":
    main()
