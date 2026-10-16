# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a retargeting run in the dv-demos layout (the Dexverse-CSI/dv-demos dataset repo), for upload_dv_demos.py:

    <out>/v1/retargeting/<category>/<task>/<robot_type>/epNNN.pkl     one successful source episode each (stage 4,
                                                                       record_demos format, NNN = source index)
    <out>/v1/retargeting/<category>/<task>/<robot_type>/index.json    every source episode's state

Every (task, target hand) of the queue gets its folder and index.json even before any episode succeeds, so the repo
shows the whole plan. index.json episode states: spider | kinematic (uploaded), failed, running, pending.
Pure Python + numpy (no simulator); arm_joint_meta.json must exist for the successful runs (arm_joint_meta.py).

  python scripts/retarget/export_dv_demos.py --runs <artifacts> --demos_root <demos v1 root> --tasks ... \
      [--hands allegro leap wuji sharpa inspire] --out <artifacts>/dv-demos
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_queue import ALL_HANDS, find_demos  # noqa: E402

VERSION = "v1"


def episode_state(d: Path) -> dict:
    st = json.load(open(d / "status.json")) if (d / "status.json").exists() else None
    if st is None:
        return {"state": "running" if d.exists() else "pending"}
    att = st.get("attempts", [])
    if st.get("state") == "recorded":
        good = next(a for a in att if a.get("success"))
        return {"state": "kinematic" if good["tag"] == "kinonly" else "spider", "success_frame": good.get("success_frame")}
    if st.get("state") in ("failed_all_attempts", "stage2_failed"):
        why = []
        for a in att:
            r = json.load(open(d / f"{a['tag']}.json")) if (d / f"{a['tag']}.json").exists() else {}
            why.append(a["tag"] + (f" aborted at frame {r['aborted_frame']}" if r.get("aborted_frame") is not None
                                   else " no success" if r else f" {a.get('error', 'error')}"))
        return {"state": "failed", "why": "; ".join(why) or st.get("error") or st["state"]}
    return {"state": "running"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--demos_root", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--hands", nargs="+", default=ALL_HANDS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", default=os.environ.get("HSCHO_SERVER", ""))
    ap.add_argument("--run_id", default="")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    py = os.environ.get("PY", sys.executable)
    for task in a.tasks:
        demos = find_demos(Path(a.demos_root), task)
        category = demos.parent.parent.name
        with open(demos, "rb") as f:
            src = pickle.load(f)
        src_type, n_src = src["robot_type"], len(src["episodes"])
        del src
        for hand in a.hands:
            tgt = src_type.replace(src_type.split("_")[1], hand, 1)
            dst = out / VERSION / "retargeting" / category / task / tgt
            dst.mkdir(parents=True, exist_ok=True)
            eps, ok = {}, []
            for i in range(n_src):
                s = episode_state(runs / task / demos.stem / f"ep{i:03d}" / tgt)
                eps[str(i)] = s
                if s["state"] in ("spider", "kinematic"):
                    ok.append(i)
            if ok:
                r = subprocess.run([py, str(HERE / "stage4_record.py"), "--task", task, "--demos", str(demos),
                                    "--target", tgt, "--episodes", *map(str, ok), "--per_episode_dir", str(dst)],
                                   capture_output=True, text=True, env={**os.environ, "DEXVERSE_RETARGET_RUNS": str(runs)})
                print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-500:], flush=True)
            for p in dst.glob("ep[0-9][0-9][0-9].pkl"):              # an episode no longer successful is not kept
                if int(p.stem[2:]) not in ok:
                    p.unlink()
            counts = {k: sum(1 for v in eps.values() if v["state"] == k)
                      for k in ("spider", "kinematic", "failed", "running", "pending")}
            index = {"task": task, "category": category, "robot_type": tgt, "source_robot_type": src_type,
                     "source": f"dexverse/DexVerse_release demonstrations/{VERSION}/{category}/{task}/demos.pkl",
                     "source_episodes": n_src, "counts": counts, "server": a.server, "run_id": a.run_id,
                     "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "episodes": eps}
            with open(dst / "index.json", "w") as f:
                json.dump(index, f, indent=1)
            print(f"{task} {tgt}: {counts}", flush=True)


if __name__ == "__main__":
    main()
