# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 4: collect the successful stage-3 runs of one (task, demo pickle, target hand) into a trajectory pickle.

Pure Python, no simulator. Each successful run becomes one episode in record_demos' format (initial state,
per-step actions, the T+1 scene states), cut where record_demos ends a demo: at the frame the task success has
held ``success_steps`` consecutive frames. Episodes keep their source: ``retarget_source`` names the source
pickle, episode and hand. The file metadata is the source pickle's, with ``robot_type`` set to the target.

The states are the record (stage 3 restores a snapshot at every commit), so build the LeRobot dataset by
replaying states, not actions.

  python scripts/retarget/stage4_record.py --task Dexverse-GraspCup-v0 --demos <demos.pkl> \
      --target floating_allegro_right [--episodes 0 1 2] [--tag spider]
"""

import argparse
import gzip
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source", "dexverse"))
from dexverse.retarget import paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--demos", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--episodes", type=int, nargs="*", default=None, help="source episodes (default: all with a result)")
ap.add_argument("--tag", default="spider")
ap.add_argument("--out", default="", help="output pickle (default <runs>/<task>/<demos stem>/<target>.pkl)")
args = ap.parse_args()


def main():
    src = paths.load_demos(args.demos)
    base = paths.runs_root() / args.task.split(":")[-1] / os.path.splitext(os.path.basename(args.demos))[0]
    eps = args.episodes if args.episodes is not None else sorted(
        int(p.name[2:]) for p in base.glob("ep[0-9][0-9][0-9]") if (p / args.target / f"{args.tag}.json").exists())
    episodes, skipped = [], []
    for i in eps:
        d = base / f"ep{i:03d}" / args.target
        res_path = d / f"{args.tag}.json"
        if not res_path.exists():
            skipped.append((i, "no result"))
            continue
        res = json.load(open(res_path))
        if not res["task_success"]:
            skipped.append((i, "failed"))
            continue
        with gzip.open(d / f"{args.tag}_traj.pkl.gz", "rb") as fh:
            traj = pickle.load(fh)
        n = int(res["success_frame"]) + 1                          # actions up to the held-success frame
        sep = src["episodes"][i]
        episodes.append({
            "episode_index": len(episodes), "episode_name": f"demo_{len(episodes)}",
            "initial_state": traj["initial_state"], "goal_pose": sep.get("goal_pose"),
            "actions": np.stack(traj["actions"][:n]).astype(np.float32),
            "states": traj["states"][: n + 1], "success": True, "num_steps": n,
            "retarget_source": {"demos": os.path.abspath(args.demos), "episode": i, "robot_type": src["robot_type"],
                                "method": "spider", "seed": res.get("seed"), "result": str(res_path)},
        })
    meta = {k: v for k, v in src.items() if k not in ("episodes", "num_episodes", "release_provenance")}
    meta.update(robot_type=args.target, num_episodes=len(episodes),
                retarget={"source_robot_type": src["robot_type"], "source_demos": os.path.abspath(args.demos),
                          "replay": "states (stage 3 restores a snapshot every commit; do not replay actions)"})
    out = args.out or str(base / f"{args.target}.pkl")
    with open(out, "wb") as fh:
        pickle.dump({**meta, "episodes": episodes}, fh)
    print(f"RESULT stage4 {args.task} -> {args.target}: {len(episodes)} episodes -> {out}; skipped {skipped}")


if __name__ == "__main__":
    main()
