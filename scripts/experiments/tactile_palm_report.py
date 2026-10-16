# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report of the palm-zone / observation.touch validation runs of tactile_palm_verify.py (experiment; numpy only).

    python scripts/experiments/tactile_palm_report.py --runs /root/tactile_palm --out <results dir>

Per hand <h> the runs pv_<h>_on1, pv_<h>_on2 (sensors on, twice) and pv_<h>_off (no tactile sensors):
(b) bit identity of the per-step scene state / joint state / action (on1 vs on2, on1 vs off), of the touch outputs
and of the spooled episode arrays (on1 vs on2; common keys on1 vs off); (a) palm zones + non-pad vs palm total and
vs the aggregated sensor; (d) observation.touch ranges next to the real RH5DG2 export stats; (c) palm contact
points over the zone definition (figure per hand).
"""

import argparse
import glob
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="/root/tactile_palm")
ap.add_argument("--out", required=True)
ap.add_argument("--pads_dir", default="/workspace/dexverse/DexVerse-tactile/source/dexverse/dexverse/data_collection/tactile_pads")
args = ap.parse_args()
HANDS = [("sh", "floating_shadow_right", "Shadow GraspCup"), ("sp", "floating_sharpa_bimanual", "Sharpa bimanual OpenLaptop"),
         ("al", "floating_allegro_right", "Allegro PickCube"), ("in", "floating_inspire_right", "Inspire GraspCup"),
         ("wj", "floating_wuji_right", "Wuji GraspCup"), ("sr", "floating_sharpa_right", "Sharpa right GraspCup"),
         ("lp", "floating_leap_right", "LEAP GraspCup (retargeted)")]
REAL = {"thumb.fn_max": 30.00, "tangential_abs_max_approx": 30.0,
        "source": "HF RoboSynapse/teleop_ur7e-rh5dg2_unhook-mug-and-place-on-plate stats (given by the coordinator)"}
os.makedirs(args.out, exist_ok=True)
L = []


def P(*a):
    L.append(" ".join(str(x) for x in a))
    print(*a, flush=True)


def eq_npz(a, b, keys=None):
    za, zb = np.load(a), np.load(b)
    ks = sorted(set(za.files) & set(zb.files)) if keys is None else keys
    diff = [k for k in ks if not (za[k].shape == zb[k].shape and np.array_equal(za[k], zb[k]))]
    return len(ks), diff, sorted(set(za.files) ^ set(zb.files))


summary = {}
P("## (b) bit identity (same server amp1, cameras off, CPU physics)")
P("hand | steps | scene/joint/action on1 vs on2 | on1 vs off | touch outputs on1 vs on2 | spool data.npz on1 vs on2 | spool common keys on1 vs off")
for h, rt, hn in HANDS:
    d = {r: os.path.join(args.runs, f"pv_{h}_{r}", "o") for r in ("on1", "on2", "off")}
    if not all(os.path.exists(os.path.join(x, "states.npz")) for x in d.values()):
        P(f"{hn} | missing runs")
        continue
    s12 = eq_npz(d["on1"] + "/states.npz", d["on2"] + "/states.npz")
    s1o = eq_npz(d["on1"] + "/states.npz", d["off"] + "/states.npz")
    t12 = eq_npz(d["on1"] + "/touch_steps.npz", d["on2"] + "/touch_steps.npz")
    sp = {r: (glob.glob(os.path.join(d[r], "ds.spool", "ep_*", "data.npz")) or [None])[0] for r in d}
    p12 = eq_npz(sp["on1"], sp["on2"]) if sp["on1"] and sp["on2"] else None
    p1o = eq_npz(sp["on1"], sp["off"]) if sp["on1"] and sp["off"] else None
    steps = int(np.load(d["on1"] + "/states.npz")["action"].shape[0])

    def fmt(x):
        return "n/a" if x is None else (f"{x[0]} keys identical" if not x[1] else f"DIFF {x[1][:4]}")

    P(f"{hn} | {steps} | {fmt(s12)} | {fmt(s1o)} | {fmt(t12)} | {fmt(p12)} | {fmt(p1o)} (only-on keys {len(p1o[2]) if p1o else '-'})")
    summary[h] = {"steps": steps, "states_on1_on2_diff": s12[1], "states_on1_off_diff": s1o[1], "touch_on1_on2_diff": t12[1],
                  "spool_on1_on2_diff": p12[1] if p12 else None, "spool_on1_off_common_diff": p1o[1] if p1o else None,
                  "spool_only_in_one": p1o[2] if p1o else None}

P("\n## (a) palm: zones (palm_l + palm_c + palm_r) + palm non-pad vs palm total (per frame max |diff|), vs aggregated sensor")
P("hand | palm | contact frames (|palm| > 0.05 N) | max |palm| N | zone share of palm force | max|l+c+r+nonpad - total| N | max resid vs aggregated sensor N")
for h, rt, hn in HANDS:
    f = os.path.join(args.runs, f"pv_{h}_on1", "o", "palm_verify.json")
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    for p in range(len(r["palm_max_N"])):
        P(f"{hn} | {p} | {r['palm_frames_in_contact'][p]} | {r['palm_max_N'][p]:.2f} | {r['palm_zone_share_of_palm_total'][p]:.3f} | "
          f"{r['palm_zone_plus_nonpad_minus_palm_total_max_N']:.2e} | {r['palm_resid_vs_aggregated_sensor_max_N']:.2e}")
    summary.setdefault(h, {}).update({k: r[k] for k in r if k.startswith("palm") or k.startswith("touch_norm") or k.startswith("spool_")})

P("\n## (d) observation.touch ranges (N, sensors on run 1) next to the real RH5DG2 export (thumb.fn max 30.00, tangential about +-30)")
P("hand | pad | valid | fn max | ftx min..max | fty min..max | contact frames (fn > 0.1 N)")
for h, rt, hn in HANDS:
    f = os.path.join(args.runs, f"pv_{h}_on1", "o", "palm_verify.json")
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    for sn, q in r["touch_ranges"].items():
        if not q["valid"]:
            P(f"{hn} | {sn} | 0 | - | - | - | -")
            continue
        P(f"{hn} | {sn} | 1 | {q['fn_max']:.2f} | {q['ftx_min']:.2f}..{q['ftx_max']:.2f} | {q['fty_min']:.2f}..{q['fty_max']:.2f} | "
          f"{q['contact_frames']}")
    summary.setdefault(h, {})["touch_ranges"] = r["touch_ranges"]
    summary[h]["touch_norm_vs_pad_norm_max_N"] = r.get("touch_norm_vs_pad_norm_max_N")
P(f"real: {REAL}")

# ------------------------------------------------------------------ (c) figures: palm contact points over the zones
colors = {0: "#dddddd", 1: "#f4a6a6", 2: "#a6e3a6", 3: "#a6c8f4"}
pcol = {0: "k", 1: "red", 2: "green", 3: "blue"}
for h, rt, hn in HANDS:
    pts_f = os.path.join(args.runs, f"pv_{h}_on1", "o", "palm_points.pkl")
    if not os.path.exists(pts_f):
        continue
    PTS = pickle.load(open(pts_f, "rb"))
    z = np.load(os.path.join(args.pads_dir, f"{rt}.npz"))
    links = [str(x) for x in z["palm_links"]]
    lp = [int(x) for x in z["palm_link_palm"]]
    npalm = len(z["palms"])
    fig, axs = plt.subplots(npalm, 2, figsize=(10, 4.6 * npalm), squeeze=False)
    rng = np.random.default_rng(0)
    for p in range(npalm):
        c, u, hh, t = z[f"palm_frame_{p}"]
        for j, ln in enumerate(links):
            if lp[j] != p:
                continue
            Tb = z[f"palm_link_T_{j}"] if f"palm_link_T_{j}" in z.files else np.eye(4)
            S = z[f"palm_samples_{j}"].astype(float) @ Tb[:3, :3].T + Tb[:3, 3]
            Zn = z[f"palm_zone_{j}"]
            sub = rng.choice(len(S), min(len(S), 6000), replace=False)
            cc = [colors[int(v)] for v in Zn[sub]]
            axs[p, 0].scatter((S[sub] - c) @ t * 1e3, (S[sub] - c) @ u * 1e3, s=1, c=cc)
            axs[p, 1].scatter((S[sub] - c) @ u * 1e3, (S[sub] - c) @ hh * 1e3, s=1, c=cc)
        mine = [q for q in PTS if q[1] == p]
        if mine:
            X, F, Zl = [], [], []
            for (_t, _p, ln, xl, fm, zl) in mine:
                j = links.index(ln)
                Tb = z[f"palm_link_T_{j}"] if f"palm_link_T_{j}" in z.files else np.eye(4)
                X.append(xl @ Tb[:3, :3].T + Tb[:3, 3])
                F.append(fm)
                Zl.append(zl)
            X, F, Zl = np.concatenate(X), np.concatenate(F), np.concatenate(Zl)
            sel = rng.choice(len(X), min(len(X), 800), replace=False)
            sz = 6 + 40 * np.clip(F[sel] / max(F.max(), 1e-9), 0, 1)
            for ax, (a1, a2) in ((axs[p, 0], (t, u)), (axs[p, 1], (u, hh))):
                ax.scatter((X[sel] - c) @ a1 * 1e3, (X[sel] - c) @ a2 * 1e3, s=sz, c=[pcol[int(v)] for v in Zl[sel]],
                           edgecolors="k", linewidths=0.2)
            share = {n: float(F[Zl == k].sum() / max(F.sum(), 1e-9)) for k, n in ((1, "palm_l"), (2, "palm_c"),
                                                                                     (3, "palm_r"), (0, "non-pad"))}
        else:
            share = {}
        axs[p, 0].set_title(f"{hn} palm {p}: palmar view (x thumb side t, y distal u) | contact normal-force share "
                            + ", ".join(f"{k} {v:.2f}" for k, v in share.items()), fontsize=7)
        axs[p, 1].set_title("side view (x distal u, y palmar normal h); pink/green/blue = palm_l/c/r zones, "
                            "points = contacts (red/green/blue = zone, black = non-pad)", fontsize=7)
        for ax in axs[p]:
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            ax.set_xlabel("mm", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{rt}_palm_contacts.png"), dpi=130)
    plt.close(fig)
open(os.path.join(args.out, "palm_report.txt"), "w").write("\n".join(L) + "\n")
json.dump(summary, open(os.path.join(args.out, "palm_report.json"), "w"), indent=1, default=float)
print("REPORT_DONE")
