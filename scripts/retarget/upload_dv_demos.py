# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Upload dv-demos exports (export_dv_demos.py) to the HF dataset repo Dexverse-CSI/dv-demos, merged per episode.

Two uploaders write the same repo: the central PC (the synced experiments tree of the servers, default sources)
and the SLURM cluster job (--src <its export>, token from $HF_TOKEN). Each run merges its exports with what is on
the repo, episode by episode, so neither removes the other's results:

    <export>/v1/retargeting/<category>/<task>/<robot_type>/{epNNN.pkl, index.json}  ->  repo, same paths

- per episode, a success (spider / kinematic) beats running > failed > pending; a tie keeps the repo's entry;
  the same server's newer state replaces its own older one (running -> failed), but never a success.
- the episode's pickle is uploaded only when this side's success wins. Nothing is ever deleted from the repo.
- the root README.md progress table is rebuilt from the merged index.json of every folder.
- nothing is sent when the merge equals the repo.
The repo is created private if missing. Needs huggingface_hub.

  python scripts/retarget/upload_dv_demos.py [--src <export> ...] [--experiments /mnt/d/hscho_share/experiments]
      [--repo Dexverse-CSI/dv-demos] [--token_file /mnt/d/confidentials/hf_token.json] [--dry_run]
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
README = """---
license: other
task_categories:
- robotics
tags:
- dexverse
- dexterous-manipulation
- retargeting
---

# dv-demos

DexVerse demonstrations, by version. `v1/retargeting/` holds the released DexVerse v1 demonstrations
(`dexverse/DexVerse_release`, source hand Shadow) retargeted to other hands in the unmodified tasks.

## Layout

```
v1/retargeting/<category>/<task>/<robot_type>/epNNN.pkl    one source episode (NNN = its index in the source)
v1/retargeting/<category>/<task>/<robot_type>/index.json   state of every source episode
```

- `epNNN.pkl`: `record_demos` trajectory pickle (`format: dexverse_trajectory`, schema 5) holding that one episode:
  `initial_state`, per-step `actions` (target hand's action layout), `arm_joint_actions`, the scene `states`,
  cut at the frame the task success has held 10 frames. `retarget_source` gives the source episode and the method.
- Method (`index.json` / `retarget_source.method`): `kinematic` = the kinematic retarget already succeeds when its
  joints are tracked under physics; `spider` = SPIDER physics retargeting (arXiv 2511.09484) refined it.
- Replay by **states**, not actions: SPIDER restores a snapshot every commit, so open-loop action replay is not
  guaranteed to reproduce the episode.
- index.json states: `spider`, `kinematic` (file present), `failed` (with why), `running`, `pending` (not run yet).

## Progress

| category | task | robot_type | spider | kinematic | failed | running | pending |
|---|---|---|---|---|---|---|---|
{rows}
"""


RANK = {"spider": 3, "kinematic": 3, "running": 2, "failed": 1, "pending": 0}
COUNT_KEYS = ("spider", "kinematic", "failed", "running", "pending")


def ok(e: dict) -> bool:
    return e["state"] in ("spider", "kinematic")


def episodes_of(index: dict) -> dict:
    """index.json episodes, each tagged with the server it came from (older indexes only name it per folder)."""
    return {k: {**v, "server": v.get("server", index.get("server", ""))} for k, v in index["episodes"].items()}


def wins(new: dict, old: dict) -> bool:
    """Does episode entry `new` replace `old`? (a tie keeps `old`)"""
    if new["server"] == old["server"]:                     # its own newer state, but a success stays a success
        return ok(new) or not ok(old)
    return RANK[new["state"]] > RANK[old["state"]]


def load_token(token_file: str) -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    raw = Path(token_file).read_text().strip()
    return json.loads(raw)["token"] if raw.startswith("{") else raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", default=[], help="export root(s) (.../dv-demos); default: every "
                    "<experiments>/*/dexverse-retarget/*/artifacts/dv-demos")
    ap.add_argument("--experiments", default=str(HERE.parents[4] / "experiments"))
    ap.add_argument("--repo", default="Dexverse-CSI/dv-demos")
    ap.add_argument("--token_file", default="/mnt/d/confidentials/hf_token.json", help="JSON {token} or plain; $HF_TOKEN wins")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--state", default="", help="(unused: an unchanged merge is detected against the repo)")
    a = ap.parse_args()
    roots = [Path(s) for s in a.src] or sorted(Path(a.experiments).glob("*/dexverse-retarget/*/artifacts/dv-demos"))

    # local: merge every export root per episode; remember where each winning success pickle is
    local = {}                                             # rel folder -> {"index": meta, "eps": {ep: entry}, "pkl": {ep: path}}
    for root in roots:
        for idx in sorted(root.glob("v*/retargeting/*/*/*/index.json")):
            rel = idx.parent.relative_to(root).as_posix()
            d = json.load(open(idx))
            f = local.setdefault(rel, {"index": d, "eps": {}, "pkl": {}})
            for k, e in episodes_of(d).items():
                if k not in f["eps"] or wins(e, f["eps"][k]):
                    f["eps"][k] = e
                    p = idx.parent / f"ep{int(k):03d}.pkl"
                    f["pkl"].pop(k, None)
                    if ok(e) and p.exists():
                        f["pkl"][k] = p

    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

    token = load_token(a.token_file)
    api = HfApi(token=token)
    if not a.dry_run:
        api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    try:
        remote_files = set(api.list_repo_files(a.repo, repo_type="dataset"))
    except Exception as e:  # noqa: BLE001
        if not a.dry_run:
            raise
        print(f"(dry run) repo not readable: {e}")
        remote_files = set()
    tmp = Path(tempfile.mkdtemp(prefix="dv-demos-"))
    remote = {}
    for fn in sorted(x for x in remote_files if x.endswith("/index.json") and "/retargeting/" in x):
        p = hf_hub_download(a.repo, fn, repo_type="dataset", token=token, cache_dir=str(tmp / "cache"))
        remote[fn.rsplit("/", 1)[0]] = json.load(open(p))

    stage = tmp / "stage"
    stage.mkdir()
    rows, changed, n_pkl = [], [], 0
    for rel in sorted(set(local) | set(remote)):
        r = remote.get(rel)
        meta = dict(r or local[rel]["index"])
        eps = episodes_of(r) if r else {}
        lf = local.get(rel, {"eps": {}, "pkl": {}})
        up = []
        for k, e in lf["eps"].items():
            if k not in eps or wins(e, eps[k]):
                eps[k] = e
                if k in lf["pkl"] and f"{rel}/ep{int(k):03d}.pkl" not in remote_files:
                    up.append(k)
                elif k in lf["pkl"] and not (r and ok(episodes_of(r).get(k, {"state": "pending"}))):
                    up.append(k)                           # replaces a pickle the repo still holds from an old success
        eps = dict(sorted(eps.items(), key=lambda kv: int(kv[0])))
        servers = sorted({e["server"] for e in eps.values() if e["server"]})
        meta.update({"episodes": eps, "counts": {c: sum(e["state"] == c for e in eps.values()) for c in COUNT_KEYS},
                     "server": ",".join(servers)})
        meta.pop("run_id", None)
        c = meta["counts"]
        rows.append(f"| {meta['category']} | {meta['task']} | {meta['robot_type']} | {c['spider']} | {c['kinematic']} | "
                    f"{c['failed']} | {c['running']} | {c['pending']} |")
        same = r is not None and {**r, "updated": None, "run_id": None} == {**meta, "updated": None, "run_id": None}
        if same and not up:
            continue
        meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (stage / rel).mkdir(parents=True, exist_ok=True)
        with open(stage / rel / "index.json", "w") as fh:
            json.dump(meta, fh, indent=1)
        for k in up:
            shutil.copy2(lf["pkl"][k], stage / rel / f"ep{int(k):03d}.pkl")
        n_pkl += len(up)
        changed.append(f"{rel}: {c} (+{len(up)} pkl)")
    if changed or "README.md" not in remote_files:
        (stage / "README.md").write_text(README.replace("{rows}", "\n".join(rows)))
    print(f"{len(rows)} task x hand folders; {len(changed)} changed, {n_pkl} episode pickles to send")
    for line in changed:
        print("  " + line)
    if not changed and "README.md" in remote_files:
        print("unchanged on the repo, nothing sent")
    elif not a.dry_run:
        api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(stage), path_in_repo="",
                          commit_message=f"retargeting: +{n_pkl} episodes, {len(changed)} task x hand folders updated")
        print(f"uploaded to https://huggingface.co/datasets/{a.repo}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
