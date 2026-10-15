# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One round of the cluster's dv-demos upload (run by ondemand/dv_upload.sh inside the Isaac container).

  1. arm_joint_meta.json for recorded results whose stage-3 trajectory predates storing it (stage 4 needs it for
     record_demos' arm_joint_actions); one short Isaac launch per (task, target hand)
  2. scripts/retarget/export_dv_demos.py -> <runs>/../dv-demos: epNNN.pkl per success + index.json per (task, hand)
  3. scripts/retarget/upload_dv_demos.py --src that export -> Dexverse-CSI/dv-demos, merged per episode with the repo
     (the central PC uploads the servers' results to the same repo; neither removes the other's)

Env: TASKS, HANDS, DEMOS_ROOT, PY, HF_TOKEN, DEXVERSE_RETARGET_RUNS, SLURM_JOB_ID.
"""

import gzip
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
S = REPO / "scripts" / "retarget"
PY = os.environ.get("PY", sys.executable)
RUNS = Path(os.environ.get("DEXVERSE_RETARGET_RUNS", "/workspace/dexverse/runs/retarget"))
OUT = RUNS.parent / "dv-demos"
TASKS = os.environ["TASKS"].split()
HANDS = os.environ.get("HANDS", "allegro inspire leap sharpa wuji").split()


def run(cmd):
    print("$ " + " ".join(map(str, cmd[1:4])) + " ...", flush=True)
    return subprocess.run(list(map(str, cmd)), cwd=REPO).returncode


def main():
    need = {}
    for st in sorted(RUNS.glob("*/demos/ep*/*/status.json")):
        d = st.parent
        task = d.parent.parent.parent.name
        if task not in TASKS or json.load(open(st)).get("state") != "recorded":
            continue
        if (d / "arm_joint_meta.json").exists() or (d / ".arm_meta_in_traj").exists():
            continue
        with gzip.open(d / "spider_traj.pkl.gz", "rb") as f:
            if pickle.load(f).get("arm_joint_meta") is not None:
                (d / ".arm_meta_in_traj").touch()
                continue
        need.setdefault((task, d.name), []).append(d)
    for (task, tgt), dirs in sorted(need.items()):
        run([PY, "-u", S / "arm_joint_meta.py", "--task", task, "--target", tgt, "--headless",
             *[x for d in dirs for x in ("--out", d)]])
    rc = run([PY, S / "export_dv_demos.py", "--runs", RUNS, "--demos_root", os.environ["DEMOS_ROOT"],
              "--tasks", *TASKS, "--hands", *HANDS, "--out", OUT, "--server", "slurm",
              "--run_id", f"slurm-{os.environ.get('SLURM_JOB_ID', '')}"])
    if rc:
        sys.exit(f"export failed (rc={rc}); nothing uploaded this round")
    sys.exit(run([PY, S / "upload_dv_demos.py", "--src", OUT]))


if __name__ == "__main__":
    main()
