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
ap.add_argument("--per_episode_dir", default="", help="instead: one pickle per source episode, <dir>/epNNN.pkl "
                "(NNN = source episode index; each a record_demos pickle holding that one episode)")
args = ap.parse_args()


def main():
    src = paths.load_demos(args.demos)
    base = paths.runs_root() / args.task.split(":")[-1] / os.path.splitext(os.path.basename(args.demos))[0]
    eps = args.episodes if args.episodes is not None else sorted(
        int(p.name[2:]) for p in base.glob("ep[0-9][0-9][0-9]") if (p / args.target / f"{args.tag}.json").exists())
    episodes, skipped, layout, arm_meta = [], [], None, None
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
        # one unreadable episode is skipped (listed in the RESULT line), not the whole target's export
        try:
            with gzip.open(d / f"{args.tag}_traj.pkl.gz", "rb") as fh:
                traj = pickle.load(fh)
            layout = traj.get("action_layout", layout)
            n = int(res["success_frame"]) + 1                          # actions up to the held-success frame
            sep = src["episodes"][i]
            actions = np.stack(traj["actions"][:n]).astype(np.float32)
            # record_demos' arm_joint_actions: the commanded arm joint targets per step = action * scale + offset
            am = traj.get("arm_joint_meta")
            if am is None and (d / "arm_joint_meta.json").exists():
                am = json.load(open(d / "arm_joint_meta.json"))
            if am is None:
                raise RuntimeError("no arm_joint_meta (run scripts/retarget/arm_joint_meta.py for results older than it)")
            arm_meta = am if arm_meta is None else arm_meta
            extra = {}
            if am["arm_joint_names"]:
                col = {j: (c, s_, o) for c, (j, s_, o) in enumerate(am["action_map"])}
                extra["arm_joint_actions"] = np.stack(
                    [actions[:, col[j][0]] * col[j][1] + col[j][2] for j in am["arm_joint_ids"]], 1).astype(np.float32)
            # the run that produced it: SPIDER, or the kinematic joints alone (run_target's kinonly check)
            method = "kinematic" if res.get("args", {}).get("tag") == "kinonly" else "spider"
            episodes.append({
                "episode_index": len(episodes), "episode_name": f"demo_{len(episodes)}",
                "initial_state": traj["initial_state"], "goal_pose": sep.get("goal_pose"),
                "actions": actions, **extra,
                "states": traj["states"][: n + 1], "success": True, "num_steps": n,
                "retarget_source": {"demos": os.path.abspath(args.demos), "episode": i, "robot_type": src["robot_type"],
                                    "method": method, "seed": res.get("seed"), "result": str(res_path.resolve())},
            })
        except Exception as e:  # noqa: BLE001
            skipped.append((i, f"{type(e).__name__}: {e}"[:200]))
            print(f"[stage4] skip ep{i:03d}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    meta = {k: v for k, v in src.items() if k not in ("episodes", "num_episodes", "release_provenance")}
    if layout is not None:                                         # the source's layout is the source hand's
        meta["action_layout"] = layout
    if arm_meta is not None:                                       # likewise the arm joints (record_demos metadata)
        meta["arm_joint_names"] = arm_meta["arm_joint_names"]
    meta.update(robot_type=args.target, num_episodes=len(episodes),
                retarget={"source_robot_type": src["robot_type"], "source_demos": os.path.abspath(args.demos),
                          "replay": "states (stage 3 restores a snapshot every commit; do not replay actions)"})
    if args.per_episode_dir:
        os.makedirs(args.per_episode_dir, exist_ok=True)
        for ep in episodes:
            i = ep["retarget_source"]["episode"]
            one = {**ep, "episode_index": 0, "episode_name": "demo_0"}
            with open(os.path.join(args.per_episode_dir, f"ep{i:03d}.pkl"), "wb") as fh:
                pickle.dump({**meta, "num_episodes": 1, "episodes": [one]}, fh)
        print(f"RESULT stage4 {args.task} -> {args.target}: {len(episodes)} episode pickles -> {args.per_episode_dir}; "
              f"skipped {skipped}")
        return
    out = args.out or str(base / f"{args.target}.pkl")
    with open(out, "wb") as fh:
        pickle.dump({**meta, "episodes": episodes}, fh)
    print(f"RESULT stage4 {args.task} -> {args.target}: {len(episodes)} episodes -> {out}; skipped {skipped}")


if __name__ == "__main__":
    main()
