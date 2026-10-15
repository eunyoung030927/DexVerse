#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit teleop demo sets for validity, coverage, horizon slack and signal quality.

Reads the session pickles (``record_demos.py`` output) and, when available, the
replayed HDF5 (``create_demo_files_sequential.py`` output) of one or more
``<category>/<Task>/<session>`` demo sets and prints, per set:

  A1 criterion replay   -- does the task's success term fire when the demo is replayed,
                           and how late (should be ~end of episode);
  A2 rule violations    -- failure terminations tripped by any demo;
  A3 post-success tail  -- steps recorded after the first success;
  B5 reset coverage     -- initial-pose spread per scene entity, quartile occupancy
                           (against ``--range`` when given, else the demos' own span);
  B6 goal balance       -- counts of discrete per-episode goals (``task_state`` buffers);
  B8 horizon slack      -- episode lengths vs the eval horizon;
  C13 signal quality    -- idle lead/trail, idle fraction, action jerk, per session.

Requires NO Isaac; runs on the raw files. Examples::

    python scripts/demo_tools/audit_demos.py \\
        --task functional/Dexverse-FunctionalPourMug-v0/0823_zhuxiu --horizon 1200 \\
        --range PourMug:object:-0.10,0.20,-0.35,0.35

    python scripts/demo_tools/audit_demos.py --task bimanual/Dexverse-BimanualCartonRegrasp-v0/0823_zhuxiu \\
        --task articulation/Dexverse-CutStripScissors-v0/0824_zhuxiu_final50 --json audit.json

NOTE: A1-A3 need an H5 replayed with a replayer that advances ``episode_length_buf``
during set-state replay (fixed 2026-08-25). Older H5s report 0 fires for stage-machine /
cut-sweep tasks -- re-replay before trusting A1 for those.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DEMOS_ROOT = os.path.join(REPO_ROOT, "source", "dexverse", "demonstrations")
DEFAULT_H5_ROOT = os.path.join(REPO_ROOT, "outputs", "demos_h5")


def _yaw_deg(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _quartiles(u: np.ndarray) -> list[int]:
    return np.histogram(np.clip(u, 0, 1), bins=4, range=(0, 1))[0].tolist()


def _idle(actions: np.ndarray, thr: float = 1e-3) -> tuple[int, int, float]:
    if len(actions) < 2:
        return 0, 0, 0.0
    still = np.abs(np.diff(actions, axis=0)).max(axis=1) < thr
    moving = ~still
    lead = int(np.argmax(moving)) if moving.any() else len(still)
    trail = int(np.argmax(moving[::-1])) if moving.any() else len(still)
    return lead, trail, float(still.mean())


def load_set(demos_root: str, rel: str) -> dict:
    session_dir = os.path.join(demos_root, rel)
    pkls = sorted(glob.glob(os.path.join(session_dir, "*.pkl")))
    if not pkls:
        raise SystemExit(f"no pickles under {session_dir}")
    episodes, sessions, meta = [], [], None
    for p in pkls:
        with open(p, "rb") as fp:
            d = pickle.load(fp)
        if d.get("format") != "dexverse_trajectory":
            print(f"  [skip] {os.path.basename(p)}: format={d.get('format')!r}")
            continue
        meta = meta or {k: d.get(k) for k in ("task", "robot_type", "schema_version")}
        for e in d.get("episodes", []):
            episodes.append(e)
            sessions.append(os.path.basename(p))
    return {"rel": rel, "episodes": episodes, "sessions": sessions, "meta": meta or {}, "session_dir": session_dir}


def find_h5(h5_root: str, rel: str) -> str | None:
    session = rel.rstrip("/").split("/")[-1]
    cand = os.path.join(h5_root, rel, f"{session}.pointcloud.seq.demo.h5")
    if os.path.isfile(cand):
        return cand
    hits = sorted(glob.glob(os.path.join(h5_root, rel, "*.seq.demo.h5")))
    return hits[0] if hits else None


def audit_replay(h5_path: str | None) -> dict:
    out = {"h5": h5_path}
    if h5_path is None:
        return out
    import h5py

    with h5py.File(h5_path, "r") as hf:
        demos = list(hf["data"].keys())
        fires, first_frac, tails, no_attr, violations = [], [], [], [], {}
        for k in demos:
            g = hf["data"][k]
            if not bool(g.attrs.get("success", False)):
                no_attr.append(k)
            terms = g["terminations"] if "terminations" in g else {}
            if "success" in terms:
                s = terms["success"][...].astype(bool)
                if s.any():
                    f = int(np.argmax(s))
                    fires.append(k)
                    first_frac.append(f / max(1, len(s)))
                    tails.append(len(s) - f)
            for t in terms:
                if t in ("success", "time_out"):
                    continue
                if terms[t][...].astype(bool).any():
                    violations.setdefault(t, []).append(k)
        out.update(
            demos=len(demos),
            attr_success=len(demos) - len(no_attr),
            fires=len(fires),
            first_success_pct_median=float(np.median(first_frac) * 100) if first_frac else None,
            tail_median=float(np.median(tails)) if tails else None,
            tail_max=int(max(tails)) if tails else None,
            no_fire=[k for k in demos if k not in fires],
            violations={t: len(v) for t, v in violations.items()},
        )
    return out


def audit_coverage(episodes: list, ranges: dict[str, tuple[float, float, float, float]]) -> dict:
    st0 = episodes[0].get("initial_state", {})
    entities = [("rigid_object", k) for k in st0.get("rigid_object", {}) if "table" not in k] + [
        ("articulation", k) for k in st0.get("articulation", {}) if k != "robot"
    ]
    out = {}
    for grp, name in entities:
        try:
            P = np.array([np.asarray(e["initial_state"][grp][name]["root_pose"]).reshape(-1)[:7] for e in episodes])
        except Exception:
            continue
        yaw = _yaw_deg(P[:, 3:7])
        rec = {
            "x": [float(P[:, 0].min()), float(P[:, 0].max()), float(P[:, 0].std())],
            "y": [float(P[:, 1].min()), float(P[:, 1].max()), float(P[:, 1].std())],
            "yaw_deg": [float(yaw.min()), float(yaw.max()), float(yaw.std())],
        }
        if name in ranges:
            x0, x1, y0, y1 = ranges[name]
            rec["range"] = ranges[name]
            rec["quartiles_x"] = _quartiles((P[:, 0] - x0) / max(1e-9, x1 - x0))
            rec["quartiles_y"] = _quartiles((P[:, 1] - y0) / max(1e-9, y1 - y0))
            rec["outside_range"] = int(((P[:, 0] < x0) | (P[:, 0] > x1) | (P[:, 1] < y0) | (P[:, 1] > y1)).sum())
        elif P[:, 0].std() > 1e-4 or P[:, 1].std() > 1e-4:
            rec["quartiles_x"] = _quartiles((P[:, 0] - P[:, 0].min()) / max(1e-9, np.ptp(P[:, 0])))
            rec["quartiles_y"] = _quartiles((P[:, 1] - P[:, 1].min()) / max(1e-9, np.ptp(P[:, 1])))
        out[f"{grp}/{name}"] = rec
    return out


def audit_goals(episodes: list) -> dict:
    counts: dict[str, dict] = {}
    for e in episodes:
        ts = e.get("task_state") or {}
        for bname, val in (ts.get("env_buffers") or {}).items():
            v = np.asarray(val)
            if v.size == 1 and float(v).is_integer():
                counts.setdefault(bname, {})
                counts[bname][int(v)] = counts[bname].get(int(v), 0) + 1
    return {b: dict(sorted(c.items())) for b, c in counts.items()}


def audit_horizon(episodes: list, horizon: int) -> dict:
    L = np.array([len(e.get("actions", [])) for e in episodes])
    return {
        "horizon": horizon,
        "len_min": int(L.min()), "len_median": float(np.median(L)), "len_mean": float(L.mean()), "len_max": int(L.max()),
        "over_60pct": int((L > 0.6 * horizon).sum()), "over_80pct": int((L > 0.8 * horizon).sum()), "over_horizon": int((L > horizon).sum()),
    }


def audit_signal(episodes: list, sessions: list) -> dict:
    per = {}
    for e, s in zip(episodes, sessions):
        a = np.asarray(e.get("actions", []), dtype=np.float32)
        if a.ndim != 2 or len(a) < 3:
            continue
        lead, trail, frac = _idle(a)
        jerk = float(np.abs(np.diff(a, n=2, axis=0)).mean())
        per.setdefault(s, []).append((lead, trail, frac, jerk, len(a)))
    out = {}
    for s, rows in per.items():
        r = np.array(rows, dtype=np.float64)
        out[s] = {"episodes": len(rows), "idle_lead_med": float(np.median(r[:, 0])), "idle_trail_med": float(np.median(r[:, 1])),
                  "idle_frac_mean": float(r[:, 2].mean()), "jerk_mean": float(r[:, 3].mean()), "len_mean": float(r[:, 4].mean())}
    allr = np.array([row for rows in per.values() for row in rows], dtype=np.float64)
    out["_all"] = {"episodes": int(len(allr)), "idle_frac_mean": float(allr[:, 2].mean()), "jerk_mean": float(allr[:, 3].mean()),
                   "jerk_p90": float(np.percentile(allr[:, 3], 90))} if len(allr) else {}
    return out


def flags_for(rep: dict) -> list[str]:
    f = []
    r = rep["replay"]
    if r.get("h5") is None:
        f.append("A1: no replay H5 found -> criterion/violation audits skipped")
    else:
        if r["fires"] < r["demos"]:
            f.append(f"A1: success term fired in only {r['fires']}/{r['demos']} replayed demos"
                     + (" (0 fires: if this H5 predates the step-counter replay fix of 2026-08-25, re-replay)" if r["fires"] == 0 else ""))
        if r.get("first_success_pct_median") is not None and r["first_success_pct_median"] < 85:
            f.append(f"A1: success fires early (median {r['first_success_pct_median']:.0f}% of episode) -> criterion may be too loose or demos overrun")
        if r.get("tail_max") and r["tail_max"] > 60:
            f.append(f"A3: up to {r['tail_max']} steps recorded after success -> trim demos at first success (+margin)")
        for t, n in r.get("violations", {}).items():
            f.append(f"A2: {n} demos trip failure rule '{t}'")
    for ent, c in rep["coverage"].items():
        if "range" in c:
            exp = rep["n"] / 4
            for ax in ("x", "y"):
                q = c[f"quartiles_{ax}"]
                if min(q) < 0.5 * exp:
                    f.append(f"B5: {ent} {ax}-quartiles {q} uneven vs configured range (expect ~{exp:.0f} each)")
            if c.get("outside_range"):
                f.append(f"B5: {ent}: {c['outside_range']} demos start outside the configured range")
    for b, counts in rep["goals"].items():
        exp = rep["n"] / max(1, len(counts))
        low = {k: v for k, v in counts.items() if v < 0.6 * exp}
        if low:
            f.append(f"B6: goal '{b}' imbalanced: {counts} (under-represented: {low})")
    h = rep["horizon"]
    if h["over_horizon"]:
        f.append(f"B8: {h['over_horizon']} demos LONGER than the {h['horizon']}-step horizon (unfinishable at eval)")
    if h["len_median"] > 0.6 * h["horizon"] or h["over_80pct"] > 0.1 * rep["n"]:
        f.append(f"B8: little horizon slack (median {h['len_median']:.0f}, {h['over_80pct']} demos > 80% of horizon)")
    sig = rep["signal"]
    allj = sig.get("_all", {}).get("jerk_mean")
    for s, v in sig.items():
        if s == "_all":
            continue
        if allj and v["jerk_mean"] > 1.8 * allj:
            f.append(f"C13: session {s} is jerky (jerk {v['jerk_mean']:.4f} vs set mean {allj:.4f})")
        if v["idle_frac_mean"] > 0.15:
            f.append(f"C13: session {s} idle {v['idle_frac_mean']*100:.0f}% of steps")
    return f


def print_report(rep: dict) -> None:
    print(f"\n=== {rep['rel']}  ({rep['n']} episodes, {rep['meta'].get('robot_type')}, {len(rep['signal']) - 1} session file(s)) ===")
    r = rep["replay"]
    if r.get("h5"):
        print(f"A1 criterion replay : success fires in {r['fires']}/{r['demos']} demos"
              + (f", median at {r['first_success_pct_median']:.0f}% of episode" if r["first_success_pct_median"] is not None else "")
              + f" | teleop success attr: {r['attr_success']}/{r['demos']}")
        print(f"A2 rule violations  : {r['violations'] or 'none'}")
        print(f"A3 post-success tail: median {r['tail_median']} / max {r['tail_max']} steps" if r["tail_max"] is not None else "A3 post-success tail: n/a")
    else:
        print("A1-A3: no replay H5 found")
    print("B5 reset coverage   :")
    for ent, c in rep["coverage"].items():
        line = f"   {ent:32s} x[{c['x'][0]:+.3f},{c['x'][1]:+.3f}] y[{c['y'][0]:+.3f},{c['y'][1]:+.3f}] yaw[{c['yaw_deg'][0]:+.0f},{c['yaw_deg'][1]:+.0f}]deg"
        if "quartiles_x" in c:
            line += f"  quartiles x{c['quartiles_x']} y{c['quartiles_y']}" + (" (vs configured range)" if "range" in c else " (own span)")
        print(line)
    print(f"B6 goal balance     : {rep['goals'] or 'no discrete goals recorded'}")
    h = rep["horizon"]
    print(f"B8 horizon slack    : len min/med/max {h['len_min']}/{h['len_median']:.0f}/{h['len_max']} vs horizon {h['horizon']} | >60%: {h['over_60pct']}  >80%: {h['over_80pct']}  >100%: {h['over_horizon']}")
    print("C13 signal quality  :")
    for s, v in rep["signal"].items():
        if s == "_all":
            continue
        print(f"   {s[-40:]:40s} n={v['episodes']:3d} idle lead/trail {v['idle_lead_med']:.0f}/{v['idle_trail_med']:.0f} idle {v['idle_frac_mean']*100:4.1f}% jerk {v['jerk_mean']:.4f} len {v['len_mean']:.0f}")
    print("FLAGS:" if rep["flags"] else "FLAGS: none")
    for fl in rep["flags"]:
        print(f"   ! {fl}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", required=True, help="<category>/<Task>/<session> (repeatable)")
    ap.add_argument("--demos-root", default=DEFAULT_DEMOS_ROOT)
    ap.add_argument("--h5-root", default=DEFAULT_H5_ROOT)
    ap.add_argument("--h5", action="append", default=[], help="Explicit replay H5 for the task at the same position (optional).")
    ap.add_argument("--horizon", type=int, default=1200, help="Eval horizon in env steps (default 1200).")
    ap.add_argument(
        "--range",
        action="append",
        default=[],
        help="Configured reset range 'TASKSUBSTR:entity:x0,x1,y0,y1' (world frame, init offsets applied); "
        "TASKSUBSTR selects which --task it applies to (substring match). Repeatable.",
    )
    ap.add_argument("--json", default=None, help="Write the full report to this JSON file.")
    args = ap.parse_args()

    range_specs = []
    for r in args.range:
        task_sub, name, vals = r.split(":", 2)
        range_specs.append((task_sub, name, tuple(float(v) for v in vals.split(","))))
    reports = []
    for i, rel in enumerate(args.task):
        ds = load_set(args.demos_root, rel)
        h5 = args.h5[i] if i < len(args.h5) else find_h5(args.h5_root, rel)
        ranges = {name: vals for task_sub, name, vals in range_specs if task_sub in rel}
        rep = {"rel": rel, "n": len(ds["episodes"]), "meta": ds["meta"],
               "replay": audit_replay(h5), "coverage": audit_coverage(ds["episodes"], ranges),
               "goals": audit_goals(ds["episodes"]), "horizon": audit_horizon(ds["episodes"], args.horizon),
               "signal": audit_signal(ds["episodes"], ds["sessions"])}
        rep["flags"] = flags_for(rep)
        reports.append(rep)
        print_report(rep)
    if args.json:
        with open(args.json, "w") as fp:
            json.dump(reports, fp, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
