# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Progress of every retargeting run under $DEXVERSE_RETARGET_RUNS, written into experiments/:
experiments/STATUS.md (all task x target pairs) and experiments/<task>/<target>/STATUS.md, results.csv.

Reads <task>/demos/epNNN/<target>/status.json (written by run_target.py) and, while a pair is still running, the
last progress line of its run.log. The cluster points DEXVERSE_EXPERIMENTS at <runs>/../report so the git checkout
stays clean. run_queue.py refreshes it every 2 minutes.

  python scripts/retarget/status_report.py              # once
  python scripts/retarget/status_report.py --loop 300   # refresh every 5 min
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_target import experiments_root, runs_root  # noqa: E402

DONE_OK = {"recorded"}
DONE_BAD = {"failed_all_attempts", "stage2_failed"}


def tail_progress(d: Path) -> str:
    log = d / "run.log"
    if not log.exists():
        return ""
    txt = log.read_bytes()[-200_000:].decode(errors="ignore")
    hits = re.findall(r"(frame \d+/\d+|drop of \[[^\]]*\] at frame \d+[^\n]*|RESULT [^\n]*)", txt)
    return hits[-1][:110] if hits else ""


def collect(runs: Path):
    rows = []
    dirs = {p.parent for pat in ("status.json", "run.log") for p in runs.glob(f"*/*/ep[0-9][0-9][0-9]/*/{pat}")}
    for d in sorted(dirs):
        st = d / "status.json"
        try:
            s = json.load(open(st)) if st.exists() else {"state": "running"}   # first attempt still in stage 2/3
        except (OSError, ValueError):
            continue
        att = s.get("attempts", [])
        state = s.get("state", "?")
        good = next((x for x in att if x.get("success")), None)
        rows.append({"task": d.parent.parent.parent.name, "episode": d.parent.name, "target": d.name, "state": state,
                     "attempts": len(att), "success": state in DONE_OK,
                     "success_attempt": good["tag"] if good else "",
                     "minutes": round(sum(x.get("minutes") or 0 for x in att), 1),
                     "done": state in DONE_OK | DONE_BAD,
                     "progress": "" if state in DONE_OK | DONE_BAD else tail_progress(d)})
    return rows


def write(exp: Path, rows, runs: Path):
    """experiments/STATUS.md (all pairs) + experiments/<task>/<target>/STATUS.md, results.csv."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    fields = ["task", "episode", "target", "state", "attempts", "success", "success_attempt", "minutes"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["target"])].append(r)
    for (task, tgt), rs in groups.items():
        d = exp / task / tgt
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rs:
                w.writerow({k: r[k] for k in fields})
        ok = sum(r["success"] for r in rs)
        bad = sum(r["done"] and not r["success"] for r in rs)
        table = ["| episode | state | attempts | minutes | last event |", "|---|---|---|---|---|"]
        table += [f"| {r['episode']} | {r['state']} | {r['attempts']} | {r['minutes']} | {r['progress'] or '-'} |" for r in rs]
        (d / "STATUS.md").write_text("\n".join(
            [f"# {task} -> {tgt}: status", "", f"Updated: {stamp}", "",
             f"{ok} success / {bad} failed / {len(rs) - ok - bad} in progress (episodes started so far)", ""] + table) + "\n")
    L = ["# Retargeting status", "", f"Updated: {stamp}  ·  data: `{runs}`", "",
         f"Total: {sum(r['success'] for r in rows)} success / {sum(r['done'] and not r['success'] for r in rows)} failed / "
         f"{sum(not r['done'] for r in rows)} in progress", "",
         "| task | target | success | failed | in progress |", "|---|---|---|---|---|"]
    for (t, g), rs in sorted(groups.items()):
        c = Counter("ok" if r["success"] else "fail" if r["done"] else "running" for r in rs)
        L.append(f"| [{t}]({t}/) | [{g}]({t}/{g}/STATUS.md) | {c['ok']} | {c['fail']} | {c['running']} |")
    (exp / "STATUS.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between refreshes (0: once)")
    a = ap.parse_args()
    runs, exp = runs_root(), experiments_root()
    runs.mkdir(parents=True, exist_ok=True)
    exp.mkdir(parents=True, exist_ok=True)
    while True:
        write(exp, collect(runs), runs)
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
