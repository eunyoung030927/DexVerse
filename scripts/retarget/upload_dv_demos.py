# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Upload the dv-demos exports of every server (export_dv_demos.py) to the HF dataset repo Dexverse-CSI/dv-demos.

Runs on the central PC: it reads the synced experiments tree (hscho_share) and holds the HF token there.

    <experiments>/<server>/dexverse-retarget/<run_id>/artifacts/dv-demos/v1/retargeting/...  ->  repo v1/retargeting/...

The servers' tasks are disjoint, so the trees are merged as they are; the root README.md gets a progress table
built from every index.json. The repo is created private if missing. Needs huggingface_hub (any python with it).

  python scripts/retarget/upload_dv_demos.py [--experiments /mnt/d/hscho_share/experiments] [--repo Dexverse-CSI/dv-demos]
      [--token_file /mnt/d/confidentials/hf_token.json] [--dry_run]
"""

import argparse
import json
import shutil
import tempfile
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default=str(HERE.parents[4] / "experiments"))
    ap.add_argument("--repo", default="Dexverse-CSI/dv-demos")
    ap.add_argument("--token_file", default="/mnt/d/confidentials/hf_token.json")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--state", default="", help="file with the content hash of the last upload: skip if unchanged")
    a = ap.parse_args()
    exp = Path(a.experiments)
    stage = Path(tempfile.mkdtemp(prefix="dv-demos-"))
    rows = []
    for root in sorted(exp.glob("*/dexverse-retarget/*/artifacts/dv-demos")):
        for idx in sorted(root.glob("v*/retargeting/*/*/*/index.json")):
            rel = idx.parent.relative_to(root)
            dst = stage / rel
            if dst.exists():                               # the same (task, hand) from an older run: newest wins
                old = json.load(open(dst / "index.json"))
                if old.get("updated", "") >= json.load(open(idx)).get("updated", ""):
                    continue
                shutil.rmtree(dst)
            shutil.copytree(idx.parent, dst)
    for idx in sorted(stage.glob("v*/retargeting/*/*/*/index.json")):
        d = json.load(open(idx))
        c = d["counts"]
        rows.append(f"| {d['category']} | {d['task']} | {d['robot_type']} | {c['spider']} | {c['kinematic']} | "
                    f"{c['failed']} | {c['running']} | {c['pending']} |")
    (stage / "README.md").write_text(README.replace("{rows}", "\n".join(rows)))
    files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
    n_pkl = sum(f.endswith(".pkl") for f in files)
    print(f"staged {len(files)} files ({n_pkl} episode pickles, {len(rows)} task x hand folders) in {stage}")
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    for f in files:                                        # content only ("updated" stamps change every export)
        b = (stage / f).read_bytes()
        if f.endswith("index.json"):
            d = json.loads(b)
            d.pop("updated", None)
            b = json.dumps(d, sort_keys=True).encode()
        h.update(f.encode() + b"\0" + b)
    digest = h.hexdigest()
    if a.state and Path(a.state).exists() and Path(a.state).read_text().strip() == digest:
        shutil.rmtree(stage)
        print("unchanged since the last upload, nothing sent")
        return
    if a.dry_run:
        print("\n".join(files[:40]))
        return
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(token=json.load(open(a.token_file))["token"])
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(stage), path_in_repo="",
                      commit_message=f"retargeting: {n_pkl} episodes, {len(rows)} task x hand folders",
                      delete_patterns=["v*/retargeting/**/ep*.pkl"])
    shutil.rmtree(stage)
    if a.state:
        Path(a.state).write_text(digest)
    print(f"uploaded to https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
