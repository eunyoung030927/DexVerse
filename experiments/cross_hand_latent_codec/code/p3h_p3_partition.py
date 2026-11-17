#!/usr/bin/env python3
"""P3 two-machine partition (work1 RTX 3090 + work2 RTX 4090). New file, stdlib only, imported by
p3h_p3_driver.py (work1) and p3h_p3_driver_w2.py (work2); identical copy on both machines.

Single authority = work1:<P3>/partition.json (static; synced to work2:<P3>/ctl/partition.json by the PC relay) and
work1:<P3>/sel/final_split.json (written once by the work1 driver at the first SELECT; synced to work2:<P3>/ctl/).
Every job has exactly one owner, so the two drivers never launch the same job:
  sel   : owner work2 iff hand in partition["sel_work2"]["hands"] and the job name is not in ["exclude"]
          (exclude = sel jobs work1 had already attempted when the split was made), else work1.
  final : unit = (hand, first episode of the 10-ep chunk), all three arms of a unit on the same machine
          (between-arm comparisons per episode stay on one GPU). owner work2 iff the unit is in
          final_split["work2_units"]; while final_split is missing the owner is None (nobody launches).
  partition["disabled"] = true -> everything work1 (manual fallback, only if work2 is known to be stopped).
"""
import json
from pathlib import Path

HANDS = ("sharpa", "shadow", "wuji")
FINAL_LOS = (0, 10, 20, 30, 40)


def parse(name):
    kind, armstep, handep = name.split("/")
    arm, step = armstep.split("_")
    hand, eps = handep.split("_ep")
    return kind, arm, step, hand, int(eps.split("-")[0])


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None


def owner(name, part, split):
    if not part or part.get("disabled"):
        return "work1"
    kind, arm, step, hand, lo = parse(name)
    if kind == "sel":
        s = part["sel_work2"]
        return "work2" if hand in s["hands"] and name not in s["exclude"] else "work1"
    if split is None:
        return None
    return "work2" if [hand, lo] in split["work2_units"] else "work1"


def machine_speed(p3, machine):
    """aggregate env steps / s of a machine over its finished selection jobs: workers / (wall s per step per worker)."""
    wall = steps = 0.0
    workers = []
    for mf in Path(p3).glob("sel/*/*/machine.json"):
        m = load(mf)
        met = load(mf.parent / "metrics.json")
        if not m or not met or m.get("machine") != machine:
            continue
        workers.append(int(m.get("workers", 3)))
        for e in met["episodes"]:
            wall += e["wall_s"]
            steps += e["trace_len"] - 1
    if steps < 3000 or not workers:
        return None
    return max(workers) / (wall / steps), wall / steps, max(workers)


def compute_final_split(p3, part, now):
    """work2 share of the 15 final units proportional to measured speed; units taken in part['work2_unit_order']."""
    if part.get("final_work2_units") is not None:  # manual override
        units = part["final_work2_units"]
        return {"work2_units": units, "n_work2": len(units), "rule": "manual override (partition.json)", "time": now}
    v1, v2 = machine_speed(p3, "work1"), machine_speed(p3, "work2")
    if v2 is None:
        n2, why = 0, "no measured work2 selection jobs -> all finals on work1"
    elif v1 is None:
        n2, why = len(part["work2_unit_order"]) // 2, "no measured work1 rate -> half"
    else:
        n2 = round(15 * v2[0] / (v1[0] + v2[0]))
        why = f"15 units x v2/(v1+v2), v = workers / (s per step per worker)"
    units = [list(u) for u in part["work2_unit_order"][:n2]]
    return {"work2_units": units, "n_work2": n2, "rule": why, "speed_work1": v1, "speed_work2": v2, "time": now}
