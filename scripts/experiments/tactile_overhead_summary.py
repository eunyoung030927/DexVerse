# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarise the tactile overhead campaign (stdlib only; runs with any python3).

    python3 scripts/experiments/tactile_overhead_summary.py <overhead dir>

Reads raw/<hand>_<res>_<sensors>/bench_<sensors>.json, render/*.json, tacsl_work2/*.json and gpumon.txt and writes
summary.txt (tables) + summary.json next to them.
"""

import glob
import json
import os
import statistics as st
import sys

O = sys.argv[1]
AMP1 = len(sys.argv) > 2 and sys.argv[2] == "amp1"
RES = [("na", "cameras OFF (amp1)")] if AMP1 else [("480", "480x640"), ("120", "120x160")]


def jobname(hand, res, s):
    return f"{hand}_{s}" if res == "na" else f"{hand}_{res}_{s}"


HANDS = [("sh2", "Shadow GraspCup"), ("al", "Allegro PickCube (appendix)"), ("sh", "Shadow GraspCup (pre-crash, partial)"), ("sp", "Sharpa bimanual OpenLaptop"),
         ("in", "Inspire GraspCup"), ("wj", "Wuji GraspCup"), ("lp", "LEAP GraspCup")]
if AMP1:
    HANDS = [("amp1_sh", "Shadow GraspCup (amp1, cameras off)")]
CONDS = ["1", "2", "3", "4a", "4b", "4c", "4bC"]
CNAME = {"1": "1 none", "2": "2 --tactile", "3": "3 +pads", "4a": "4a +TacFF", "4b": "4b +TacRGB(GPU)",
         "4c": "4c +both(GPU)", "4bC": "4bC +TacRGB(CPU)"}
out_lines = []


def p(*a):
    out_lines.append(" ".join(str(x) for x in a))


def load(job):
    d = os.path.join(O, "raw", job + "_r2")      # rerun (with per-step arrays) wins over the first run
    if not os.path.isdir(d):
        d = os.path.join(O, "raw", job)
    f = glob.glob(os.path.join(d, "bench_*.json"))
    if not f:
        return None
    j = json.load(open(f[0]))
    log = open(os.path.join(d, "log.txt")).read().split("\n")
    t0 = float(log[0].split()[1])
    t1 = [float(x.split()[2]) for x in log if x.startswith("EXIT ")]
    j["_win"] = (t0, t1[0] if t1 else None)
    return j


# ------------------------------------------------------------------ GPU memory per job from the host sampler
mon = []
for _gm in glob.glob(os.path.join(O, "gpumon*.txt")):
    for ln in open(_gm):
        parts = ln.rstrip("\n").split("|")
        if len(parts) < 4:
            continue
        t, pids, g, apps = float(parts[0]), parts[1].split(), parts[2], parts[3]
        mem = {a.split(",")[0]: float(a.split(",")[1]) for a in apps.split(";") if "," in a}
        mon.append((t, sum(mem.get(x, 0.0) for x in pids), float(g.split(",")[0]), float(g.split(",")[1])))


def gpu_peak(win):
    if not mon or win[1] is None:
        return None, None
    v = [(m, u) for (t, m, _g, u) in mon if win[0] <= t <= win[1]]
    if not v:
        return None, None
    return max(m for m, _ in v), st.mean(u for _, u in v)


def agg(results, cond, key, stat="median"):
    v = [r["ms"][key][stat] for r in results if r["cond"] == cond and key in r["ms"]]
    return (st.median(v), min(v), max(v)) if v else None


def collect(res):
    rows = {}
    for hand, _ in HANDS:
        jobs = {s: load(jobname(hand, res, s)) for s in ("none", "tactile", "pads")}
        R = {}
        for s, j in jobs.items():
            if j is None:
                continue
            gp, gu = gpu_peak(j["_win"])
            for r in j["results"]:
                r["_gpu_peak_MB"], r["_gpu_util"] = gp, gu
            for c in sorted({r["cond"] for r in j["results"]}):
                rr = [r for r in j["results"] if r["cond"] == c]
                R[c] = {"total": agg(rr, c, "total"), "total_p95": agg(rr, c, "total", "p95"),
                        "seg": {k: agg(rr, c, k)[0] for k in rr[0]["ms"]},
                        "seg_p95": {k: agg(rr, c, k, "p95")[0] for k in rr[0]["ms"]},
                        "wall": st.mean(r["episode_wall_s"] for r in rr), "steps": rr[0]["steps"],
                        "cpu": st.mean(r["cpu_cores_avg"] for r in rr), "gpu_peak_MB": gp, "gpu_util": gu,
                        "pad_parts": rr[0].get("pad_parts"), "reps": len(rr)}
        rows[hand] = R
    return rows


summary = {}
for res, label in RES:
    rows = collect(res)
    summary[res] = rows
    if not any(rows.values()):
        continue
    p(f"\n## A. step time per control step, camera {label} (median of 3 per-episode medians [min-max], ms)")
    p("hand | " + " | ".join(CNAME[c] for c in CONDS))
    for hand, hn in HANDS:
        R = rows[hand]
        if not R:
            continue
        cells = []
        for c in CONDS:
            if c in R and R[c]["total"]:
                m, lo, hi = R[c]["total"]
                cells.append(f"{m:.1f} [{lo:.1f}-{hi:.1f}]")
            else:
                cells.append("-")
        p(f"{hn} | " + " | ".join(cells))
    p(f"\n## A. increase vs cond 1 (none) and vs cond 3 (pads, no images), camera {label} (ms, %)")
    p("hand | " + " | ".join(f"{CNAME[c]} vs1" for c in CONDS[1:]) + " | " + " | ".join(f"{CNAME[c]} vs3" for c in ("4a", "4b", "4c", "4bC")))
    for hand, hn in HANDS:
        R = rows[hand]
        if "1" not in R:
            continue
        b1 = R["1"]["total"][0]
        cells = []
        for c in CONDS[1:]:
            cells.append(f"+{R[c]['total'][0] - b1:.1f} ({(R[c]['total'][0] / b1 - 1) * 100:+.0f}%)" if c in R else "-")
        b3 = R["3"]["total"][0] if "3" in R else None
        for c in ("4a", "4b", "4c", "4bC"):
            cells.append(f"+{R[c]['total'][0] - b3:.1f} ({(R[c]['total'][0] / b3 - 1) * 100:+.0f}%)" if c in R and b3 else "-")
        p(f"{hn} | " + " | ".join(cells))
    p(f"\n## A. p95 total (ms), episode wall (s), CPU cores used (avg), GPU mem peak of the bench process (MB), camera {label}")
    p("hand | cond | p95 | episode s (steps) | sim-time ratio | cpu | gpu MB | others' gpu util% (whole GPU, mean)")
    for hand, hn in HANDS:
        R = rows[hand]
        for c in CONDS:
            if c not in R:
                continue
            x = R[c]
            p(f"{hn} | {c} | {x['total_p95'][0]:.1f} | {x['wall']:.1f} ({x['steps']}) | "
              f"{x['wall'] / (x['steps'] * 1 / 60):.2f}x real time | {x['cpu']:.2f} | {x['gpu_peak_MB']} | "
              f"{x['gpu_util']:.0f}" if x["gpu_util"] is not None else f"{hn} | {c} | {x['total_p95'][0]:.1f} | {x['wall']:.1f}")
    p(f"\n## A. breakdown (median ms) camera {label}")
    keys = ["physics", "render", "obs", "scene_update", "hook", "step_other", "step", "tactile", "pad", "tacff", "tacrgb",
            "camera", "state", "buffer", "total"]
    p("hand | cond | " + " | ".join(keys))
    for hand, hn in HANDS:
        R = rows[hand]
        for c in CONDS:
            if c in R:
                p(f"{hn} | {c} | " + " | ".join(f"{R[c]['seg'].get(k, 0):.2f}" for k in keys))
    if res in ("480", "na"):
        p("\n## pad read breakdown (diagnostic episode 3d, median per step over the episode, ms; rows = contact rows/step)")
        p("hand | raw rows read+copy | aggregate+pose copy | classify | rows median | rows max | pad read (cond 3) | pose-cache hook")
        for hand, hn in HANDS:
            R = rows[hand]
            if "3d" in R and R["3d"]["pad_parts"]:
                q = R["3d"]["pad_parts"]
                p(f"{hn} | {q['raw_read_copy']:.3f} | {q['aggregate_and_pose_read']:.3f} | {q['classify']:.3f} | "
                  f"{q['rows_median']:.0f} | {q['rows_max']} | {R['3']['seg']['pad']:.2f} | {R['3']['seg']['hook']:.2f}")
        p("\n## B. latency: end of the LAST physics substep -> product in hand (median / p95 ms), condition 4c (4a for TacFF)")
        p("hand | env.step return | +tactile | +pad | +TacFF | +TacRGB(GPU) | +TacRGB(CPU, 4bC)")
        for hand, hn in HANDS:
            R = rows[hand]
            if "4c" not in R:
                continue
            s, s9 = R["4c"]["seg"], R["4c"]["seg_p95"]
            fa = R["4a"]["seg"], R["4a"]["seg_p95"]
            bc = R["4bC"]["seg"], R["4bC"]["seg_p95"]
            p(f"{hn} | {s['lat_step_return']:.1f} / {s9['lat_step_return']:.1f} | {s['lat_tactile']:.1f} / {s9['lat_tactile']:.1f} | "
              f"{s['lat_pad']:.1f} / {s9['lat_pad']:.1f} | {fa[0]['lat_tacff']:.1f} / {fa[1]['lat_tacff']:.1f} | "
              f"{s['lat_tacrgb']:.1f} / {s9['lat_tacrgb']:.1f} | {bc[0]['lat_tacrgb']:.1f} / {bc[1]['lat_tacrgb']:.1f}")

# ------------------------------------------------------------------ contact-phase view from the per-step arrays
# per-step columns: 0 t, 1 step, 2 physics, 3 render, 4 obs, 5 tactile, 6 pad, 7 tacff, 8 tacrgb, 9 cam+state+buf,
#                   10 total, 11 step-return latency, 12 pad normal rows
try:
    import numpy as np
except ImportError:
    np = None
summary["phase"] = {}
if np is not None:
    for res, label in RES:
        hdr = False
        for hand, hn in HANDS:
            Z = {}
            for s_ in ("none", "tactile", "pads"):
                f = os.path.join(O, "raw", jobname(hand, res, s_) + "_r2", f"steps_{s_}.npz")
                if not os.path.exists(f):
                    f = os.path.join(O, "raw", jobname(hand, res, s_), f"steps_{s_}.npz")
                if os.path.exists(f):
                    z = np.load(f)
                    Z.update({k: z[k] for k in z.files})
            k3 = [k for k in Z if k.startswith("3_r")]
            if not k3:
                continue
            rows = Z[k3[0]][:, 12]
            T = len(rows)
            con = np.zeros(T, bool)
            con[20:] = rows[20:] > 0
            free = np.zeros(T, bool)
            free[20:] = rows[20:] == 0
            if not hdr:
                p(f"\n## A'. contact phase (pad normal rows > 0 in cond 3, same step indices in every condition) vs free "
                  f"phase, steps >= 20, 3 repeats pooled, camera {label}; ms median / p95")
                p("hand | phase (steps) | cond | total | step (env.step) | physics | --tactile read | pad read | TacFF | TacRGB "
                  "| total - cond1 (median)")
                hdr = True
            res_ph = {}
            for ph, msk in (("contact", con), ("free", free)):
                base = None
                for c in CONDS:
                    arr = [Z[k] for k in Z if k.split("_r")[0] == c and len(Z[k]) == T]
                    if not arr or not msk.any():
                        continue
                    A = np.concatenate([x[msk] for x in arr]) * 1e3
                    md = lambda j: float(np.median(A[:, j]))  # noqa: E731
                    p9 = lambda j: float(np.percentile(A[:, j], 95))  # noqa: E731
                    if c == "1":
                        base = md(10)
                    res_ph[f"{ph}_{c}"] = {"n": int(len(A)), **{f"{n}_med": md(j) for n, j in
                                           (("total", 10), ("step", 1), ("physics", 2), ("tactile", 5), ("pad", 6),
                                            ("tacff", 7), ("tacrgb", 8))},
                                           **{f"{n}_p95": p9(j) for n, j in (("total", 10), ("pad", 6), ("tacff", 7),
                                                                             ("tacrgb", 8))}}
                    p(f"{hn} | {ph} ({int(msk.sum())}) | {CNAME[c]} | {md(10):.2f} / {p9(10):.2f} | {md(1):.2f} | "
                      f"{md(2):.2f} | {md(5):.2f} | {md(6):.2f} / {p9(6):.2f} | {md(7):.3f} / {p9(7):.3f} | "
                      f"{md(8):.2f} / {p9(8):.2f} | " + (f"+{md(10) - base:.2f}" if base is not None else "-"))
            res_ph["rows_contact_median"] = float(np.median(rows[con])) if con.any() else 0.0
            res_ph["rows_max"] = float(rows.max())
            summary["phase"][f"{hand}_{res}"] = res_ph
            p(f"{hn} | contact rows/step median {res_ph['rows_contact_median']:.0f}, max {res_ph['rows_max']:.0f}")

# ------------------------------------------------------------------ C. offline render
p("\n## C. offline render (tactile_hybrid_video_render.py, existing 240x320 recordings): per frame ms and per episode s")
p("hand | frames | rows | TacFF only: field s / tile ms/f / compose ms/f / total s (ms/f) | TacRGB only: taxim load s / tile ms/f / total s (ms/f) | both (2 passes) total s")
summary["render"] = {}


def rmean(files):
    J = [json.load(open(f)) for f in files]
    return {k: (st.median(j[k] for j in J) if isinstance(J[0][k], (int, float)) else J[0][k]) for k in J[0]}, len(J)


for hand, hn in HANDS:
    fa = sorted(glob.glob(os.path.join(O, "render", f"{hand}_tacff*.json")))
    fb = sorted(glob.glob(os.path.join(O, "render", f"{hand}_tacrgb*.json")))
    if not (fa and fb):
        continue
    (A, na), (B, nb) = rmean(fa), rmean(fb)
    summary["render"][hand] = {"tacff": A, "tacrgb": B, "runs": [na, nb]}
    p(f"{hn} (median of {na}/{nb} runs) | {A['frames']} | {A['contact_rows']} | {A['field_s']:.2f} / {A['tile_ms_per_frame']:.2f} / "
      f"{A['compose_ms_per_frame']:.2f} / {A['total_s']:.1f} ({A['wall_ms_per_frame']:.1f}) | {B['taxim_load_s']:.2f} / "
      f"{B['tile_ms_per_frame']:.2f} / {B['total_s']:.1f} ({B['wall_ms_per_frame']:.1f}) | {A['total_s'] + B['total_s']:.1f}")

# ------------------------------------------------------------------ D. TacSL
p("\n## D. TacSL (tacsl_sharpa_smoke.py test scene, GPU physics): ms/step median, work1 (10/8) vs work2 (now)")
W1 = {"t_none": 9.37, "t_none_cam": 11.12, "t_ff5": 13.60, "t_cam5": 41.67}
summary["tacsl"] = {}
for n, v in W1.items():
    f = os.path.join(O, "tacsl_work2", f"{n}.json")
    w2 = json.load(open(f))["timing"]["median_ms"] if os.path.exists(f) else None
    summary["tacsl"][n] = {"work1": v, "work2": w2}
    p(f"{n} | {v:.1f} | {w2:.1f}" if w2 else f"{n} | {v:.1f} | -")

open(os.path.join(O, "summary.txt"), "w").write("\n".join(out_lines) + "\n")
json.dump(summary, open(os.path.join(O, "summary.json"), "w"), indent=1, default=str)
print("\n".join(out_lines))
