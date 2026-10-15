# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Everything for one (task, source episode, target hand): stage 2 once, then SPIDER until the task succeeds.

  stage 2   kinematic retarget (skipped if kinematic.npz exists)
  kinonly   the kinematic joints PD-tracked under physics with no optimisation (stage3 --iters 0, a few minutes),
            judged by the same success term. If it succeeds it is the result and SPIDER is skipped.
  stage 3   SPIDER; the verdict is the task's own success term held for 10 frames. On failure retry with a new
            seed and more search -- attempt a: --seed a, --num_samples 1024 * (1 + a // 2), --iters 5 + a --
            up to --max_attempts (the Bench2Dex cluster schedule). The successful attempt is linked as
            spider.json / spider_traj.pkl.gz, which stage 4 (upload_task.py) collects into the trajectory pickle.

Every step skips what is on disk, so a resubmitted job resumes. Each Isaac stage is its own process; the
interpreter is $PY or this one. Status: <runs>/<task>/<demos stem>/epNNN/<target>/status.json

  python scripts/retarget/run_target.py --task Dexverse-GraspBleach-v1 --demos <demos.pkl> --episode 0 \
      --target floating_allegro_right [--max_attempts 3]
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


def runs_root() -> Path:
    return Path(os.environ.get("DEXVERSE_RETARGET_RUNS") or REPO / "results" / "retarget")


def experiments_root() -> Path:
    """Tracked progress tables (experiments/); the cluster points DEXVERSE_EXPERIMENTS outside the checkout."""
    return Path(os.environ.get("DEXVERSE_EXPERIMENTS") or REPO / "experiments")


def episode_dir(task: str, demos: str, episode: int) -> Path:
    """Same layout as dexverse.retarget.paths.run_dir (kept import-free: no Isaac Sim here)."""
    return runs_root() / task.split(":")[-1] / Path(demos).stem / f"ep{episode:03d}"


def run(log: Path, cmd: list, timeout: int) -> int:
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'source' / 'dexverse'}:{os.environ.get('PYTHONPATH', '')}"}
    with open(log, "a") as f:
        f.write(f"\n--- {time.strftime('%F %T')} {' '.join(map(str, cmd))}\n")
        f.flush()
        try:
            return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                                  cwd=REPO, env=env).returncode
        except subprocess.TimeoutExpired:
            f.write(f"--- TIMEOUT after {timeout}s\n")
            return 124


def link_success(d: Path, tag: str) -> None:
    """spider.json / spider_traj.pkl.gz -> the successful attempt (what stage 4 reads by default)."""
    for suffix in (".json", "_traj.pkl.gz"):
        dst = d / f"spider{suffix}"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(f"{tag}{suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--target", required=True, help="target robot_type, e.g. floating_allegro_right")
    ap.add_argument("--max_attempts", type=int, default=3)
    ap.add_argument("--seed_offset", type=int, default=0, help="attempt K uses seed K + offset")
    ap.add_argument("--spider_args", default="", help="extra stage-3 arguments for every attempt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_kinonly", action="store_true", help="skip the kinematic-only physics check")
    a = ap.parse_args()

    d = episode_dir(a.task, a.demos, a.episode) / a.target
    d.mkdir(parents=True, exist_ok=True)
    log, status_path = d / "run.log", d / "status.json"
    status = json.load(open(status_path)) if status_path.exists() else {"attempts": []}
    status.update({"task": a.task, "episode": a.episode, "target": a.target})

    def save(**kw):
        status.update(kw)
        json.dump(status, open(status_path, "w"), indent=1)

    common = ["--task", a.task, "--demos", a.demos, "--episode", str(a.episode), "--target", a.target]
    isaac = ["--headless", "--device", a.device]
    try:
        # the Isaac scripts can exit 0 after an exception (Kit shutdown), so judge each stage by its output file
        rc = run(log, [PY, "-u", HERE / "stage2_kinematic.py", *common, *isaac], 6 * 3600)   # bimanual + busy GPU: ~2.5 h
        if not (d / "kinematic.npz").exists():
            return save(state="stage2_failed", error=f"no kinematic.npz (rc={rc}, see run.log)")
        good = next((x for x in status["attempts"] if x.get("success")), None)
        if not good and not a.no_kinonly:
            kin_path = d / "kinonly.json"
            if not kin_path.exists():
                run(log, [PY, "-u", HERE / "stage3_spider.py", *common, "--tag", "kinonly", "--iters", "0",
                          "--num_samples", "8", "--max_retries", "0", "--no_hold_pass", "--ckpt_every", "0",
                          *isaac], 3600)
            if kin_path.exists():
                res = json.load(open(kin_path))
                entry = {"tag": "kinonly", "success": bool(res["task_success"]), "minutes": res.get("minutes"),
                         "success_frame": res.get("success_frame")}
                status["attempts"] = [x for x in status["attempts"] if x["tag"] != "kinonly"] + [entry]
                save(state="retrying")
                if entry["success"]:
                    good = entry
        for att in range(a.max_attempts):
            if good:
                break
            tag = f"spider_a{att}"
            res_path = d / f"{tag}.json"
            if not res_path.exists():
                extra = ["--seed", str(att + a.seed_offset), "--num_samples", str(1024 * (1 + att // 2)),
                         "--iters", str(5 + att)]
                rc = run(log, [PY, "-u", HERE / "stage3_spider.py", *common, "--tag", tag, *extra,
                               *a.spider_args.split(), *isaac], 8 * 3600)
                if not res_path.exists():
                    status["attempts"] = [x for x in status["attempts"] if x["tag"] != tag]
                    status["attempts"].append({"tag": tag, "success": False, "error": f"stage3 rc={rc}"})
                    save(state="retrying")
                    continue
            res = json.load(open(res_path))
            entry = {"tag": tag, "success": bool(res["task_success"]), "minutes": res.get("minutes"),
                     "success_frame": res.get("success_frame")}
            status["attempts"] = [x for x in status["attempts"] if x["tag"] != tag] + [entry]
            save(state="retrying")
            if entry["success"]:
                good = entry
        if not good:
            return save(state="failed_all_attempts")
        link_success(d, good["tag"])
        save(state="recorded")
    finally:
        print(f"RESULT run_target {a.task} ep{a.episode} -> {a.target}: {status.get('state')}", flush=True)


if __name__ == "__main__":
    main()
