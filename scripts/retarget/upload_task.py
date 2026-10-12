# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push a task's retargeted episodes for each target hand -- only if EVERY source episode succeeded
(or, with --allow_partial, whatever succeeded, listing the rest in failures.json).

Per target: source episodes 0..N-1 (N = --episodes) must each have a successful SPIDER run
(<runs>/<task>/demos/epNNN/<target>/status.json "recorded"). Stage 4 collects them into one trajectory pickle
(record_demos format, the colleague's recording format; LeRobot is built from it later by state replay,
scripts/teleop_tools/pickle_to_lerobot.sh). One commit to <ns>/dexverse-<task>-<hand>-retargeting adds

    demonstrations/<version>/<category>/<task>/demos.pkl     (the release layout of dexverse/DexVerse_release)
    README.md                                                dataset card
    failures.json                                            source episodes left out, with why

If any episode is missing nothing is pushed and the missing ones are listed (exit code 2). Episodes already in
the Hub pickle count as present and are merged, so a later run elsewhere that retargeted only the missing ones
completes the repo (and rewrites the card as 50/50) -- the Bench2Dex upload convention.

  python scripts/retarget/upload_task.py --task Dexverse-GraspBleach-v1 [--hands "allegro wuji"] [--episodes 50] [--dry_run]
"""

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_queue import ALL_HANDS, find_demos, supported_robot_types  # noqa: E402
from run_target import REPO, episode_dir  # noqa: E402

CARD = """---
license: other
task_categories:
- robotics
tags:
- dexverse
- dexterous-manipulation
- cross-embodiment
- retargeting
- spider
pretty_name: "DexVerse {task}: {target} (SPIDER retargeting)"
---

# DexVerse `{task}` retargeted to `{target}`

The released DexVerse demonstrations of **{task}** ([dexverse/DexVerse_release]
(https://huggingface.co/datasets/dexverse/DexVerse_release) `{src_path}`, source hand `{source}`) retargeted to
**`{target}`** with SPIDER (sampling-based physics retargeting, arXiv 2511.09484) in the unmodified task.

- **Episodes:** {n_up} of {n_src} source episodes ({missing_note})
- **Success:** every episode is a success of the task's own `success` term held for 10 consecutive steps
  (the `record_demos` rule) on the recorded run, and is cut at that step.
- **Format:** one trajectory pickle in `record_demos` format (`initial_state`, per-step `actions` in the target
  hand's env action layout, the T+1 scene `states`); `retarget_source` in each episode names the source episode.
- **Replay by states, not actions:** SPIDER restores a snapshot every commit, so open-loop action replay does not
  reproduce the run. Build LeRobot with `scripts/teleop_tools/pickle_to_lerobot.sh demos.pkl` (it detects the
  `retarget` metadata and replays the recorded states, rendering the cameras).
- **Generator:** [eunyoung030927/DexVerse `retarget/shadow-cross-hand`](https://github.com/eunyoung030927/DexVerse/tree/retarget/shadow-cross-hand)
  (`scripts/retarget/`, `docs/retarget_cross_hand.md`)

## Layout

```
{repo_path}   {n_up} episodes
```

## Not included

{missing_list}

## License

Follows the DexVerse release terms of the source demonstrations.
"""


def token():
    t = os.environ.get("HF_TOKEN")
    if not t and (Path.home() / ".hf_token").exists():
        t = (Path.home() / ".hf_token").read_text().strip()
    if not t:
        sys.exit("FATAL: no HF token (HF_TOKEN or ~/.hf_token)")
    return t


def retry(fn, what, tries=8):
    from huggingface_hub.utils import HfHubHTTPError
    for i in range(tries):
        try:
            return fn()
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code not in (409, 412, 429, 500, 502, 503, 504) or i == tries - 1:
                raise
            wait = min(600, 15 * 2 ** i) * (0.5 + random.random())
            print(f"  {what}: HTTP {code}, retry {i + 1}/{tries - 1} in {wait:.0f}s", flush=True)
            time.sleep(wait)


def repo_id(api, task: str, target: str) -> str:
    ns = os.environ.get("HF_NAMESPACE") or api.whoami()["name"]
    hand = target.split("_")[1]
    return f"{ns}/dexverse-{task.removeprefix('Dexverse-').lower()}-{hand}-retargeting"


def check(task: str, demos: Path, target: str, episodes: list[int]) -> tuple[list[int], dict]:
    """(source episodes with a successful run, {episode: why it is missing})."""
    ok, missing = [], {}
    for e in episodes:
        d = episode_dir(task, str(demos), e) / target
        st = d / "status.json"
        s = json.load(open(st)) if st.exists() else {"state": "not run"}
        if s.get("state") == "recorded" and (d / "spider.json").exists():
            ok.append(e)
        else:
            missing[f"{e:06d}"] = {"state": s.get("state"), "attempts": len(s.get("attempts", []))}
    return ok, missing


def build_pickle(task, demos, target, local_eps, hub_pkl, out):
    """Stage 4 on the local successful runs, merged with the Hub pickle's episodes; sorted by source episode."""
    py = os.environ.get("PY", sys.executable)
    tmp = f"{out}.local"
    subprocess.run([py, str(HERE / "stage4_record.py"), "--task", task, "--demos", str(demos), "--target", target,
                    "--episodes", *map(str, local_eps), "--out", tmp], check=True, cwd=REPO)
    with open(tmp, "rb") as f:
        data = pickle.load(f)
    os.remove(tmp)
    by_src = {ep["retarget_source"]["episode"]: ep for ep in data["episodes"]}
    if hub_pkl is not None:
        for ep in hub_pkl["episodes"]:
            by_src.setdefault(ep["retarget_source"]["episode"], ep)
    episodes = [by_src[k] for k in sorted(by_src)]
    for i, ep in enumerate(episodes):
        ep["episode_index"], ep["episode_name"] = i, f"demo_{i}"
    data["episodes"], data["num_episodes"] = episodes, len(episodes)
    with open(out, "wb") as f:
        pickle.dump(data, f)
    return sorted(by_src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="e.g. Dexverse-GraspBleach-v1")
    ap.add_argument("--demos_root", default=str(REPO / "demos" / "demonstrations" / "v1"))
    ap.add_argument("--hands", default=" ".join(ALL_HANDS))
    ap.add_argument("--episodes", type=int, default=50, help="source episodes 0..N-1 that must all succeed")
    ap.add_argument("--allow_partial", action="store_true", help="push what succeeded even if some episodes are missing")
    ap.add_argument("--dry_run", action="store_true", help="check only, push nothing")
    a = ap.parse_args()

    demos = find_demos(Path(a.demos_root), a.task)
    version, category = demos.parents[2].name, demos.parents[1].name     # .../<version>/<category>/<task>/demos.pkl
    with open(demos, "rb") as f:
        source = pickle.load(f)["robot_type"]
    supported = supported_robot_types()
    targets = [source.replace(source.split("_")[1], h, 1) for h in a.hands.split()]
    targets = [t for t in targets if t != source and t in supported]
    episodes = list(range(a.episodes))
    repo_path = f"demonstrations/{version}/{category}/{a.task}/demos.pkl"

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi(token=token()) if (not a.dry_run or os.environ.get("HF_TOKEN")) else None
    rc = 0
    for tgt in targets:
        ok, missing = check(a.task, demos, tgt, episodes)
        hub_pkl = None
        if api is not None:
            rid = repo_id(api, a.task, tgt)
            try:
                if repo_path in retry(lambda: api.list_repo_files(rid, repo_type="dataset"), "list"):
                    with open(hf_hub_download(rid, repo_path, repo_type="dataset", token=token()), "rb") as f:
                        hub_pkl = pickle.load(f)
            except Exception:  # noqa: BLE001  (repo not created yet)
                hub_pkl = None
            if hub_pkl is not None:
                on_hub = {f"{ep['retarget_source']['episode']:06d}" for ep in hub_pkl["episodes"]}
                for k in on_hub & set(missing):
                    missing.pop(k)
        n_ok = a.episodes - len(missing)
        if missing and not a.allow_partial:
            rc = 2
            print(f"RESULT upload_task {a.task} -> {tgt}: NOT UPLOADED, {n_ok}/{a.episodes} succeeded; missing "
                  + ", ".join(f"{k} ({v['state']})" for k, v in missing.items()), flush=True)
            continue
        if a.dry_run:
            print(f"RESULT upload_task {a.task} -> {tgt}: ready, {n_ok}/{a.episodes} ({len(ok)} local)"
                  + (f", missing {sorted(missing)}" if missing else "") + " [dry run]", flush=True)
            continue
        if not ok and hub_pkl is None:
            print(f"RESULT upload_task {a.task} -> {tgt}: nothing to upload", flush=True)
            continue
        rid = repo_id(api, a.task, tgt)
        retry(lambda: api.create_repo(rid, repo_type="dataset", private=False, exist_ok=True), "create_repo")
        with tempfile.TemporaryDirectory() as tmp:
            have = build_pickle(a.task, demos, tgt, ok, hub_pkl, f"{tmp}/demos.pkl")
            missing_list = "\n".join(f"- source episode {k}: {v['state']} ({v['attempts']} attempts)"
                                     for k, v in missing.items()) or "None."
            card = CARD.format(task=a.task, target=tgt, source=source, n_up=len(have), n_src=a.episodes,
                               src_path=f"demonstrations/{version}/{category}/{a.task}/demos.pkl", repo_path=repo_path,
                               missing_note="all included" if not missing else f"{len(missing)} left out, listed below",
                               missing_list=missing_list)
            open(f"{tmp}/README.md", "w").write(card)
            json.dump(missing, open(f"{tmp}/failures.json", "w"), indent=1)
            ops = [CommitOperationAdd(repo_path, f"{tmp}/demos.pkl"), CommitOperationAdd("README.md", f"{tmp}/README.md"),
                   CommitOperationAdd("failures.json", f"{tmp}/failures.json")]
            retry(lambda: api.create_commit(rid, repo_type="dataset", operations=ops,
                                            commit_message=f"{a.task} {tgt}: {len(have)}/{a.episodes} episodes"), "commit")
        print(f"RESULT upload_task {a.task} -> {tgt}: UPLOADED {len(have)}/{a.episodes} to {rid}"
              + (f"; missing {sorted(missing)}" if missing else ""), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
