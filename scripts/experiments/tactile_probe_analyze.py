# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Penetration depth at tactile outliers (>100 N) from DEXVERSE_TACTILE_PROBE files.
  python scripts/experiments/tactile_probe_analyze.py <probe dir>... --out results.json
Per dataset: substep rows, share of pad readings >100 N, and for those readings the minimum contact separation
(negative = penetration) - median, p90 depth, share deeper than 0.2 / 0.5 / 1 mm - split by first substep after a
state set or not. Same for 1-100 N contacts as the reference.
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--out", required=True)
args = ap.parse_args()


def depth_stats(sep):
    sep = sep[np.isfinite(sep)]
    if sep.size == 0:
        return {"n": 0}
    pen = -sep * 1000  # mm, positive = penetration
    return {"n": int(sep.size), "pen_median_mm": float(np.median(pen)), "pen_p90_mm": float(np.quantile(pen, 0.9)),
            "pen_max_mm": float(pen.max()), "gt0.2mm": float((pen > 0.2).mean()), "gt0.5mm": float((pen > 0.5).mean()),
            "gt1mm": float((pen > 1.0).mean())}


res = {}
for d in args.dirs:
    z = np.load(os.path.join(d, "probe.npz"))
    f, sep, first = z["force"], z["min_sep"], z["first_after_restore"].astype(bool)
    pads = [str(p).split("/")[-1] for p in z["pads"]]
    hi, mid = f > 100, (f > 1) & (f <= 100)
    r = {"rows": int(len(f)), "frames": int(z["frame"].max() + 1), "pads": pads,
         "first_after_restore_rows": float(first.mean()),
         "gt100_share": float(hi.mean()), "max_force": float(f.max()),
         "gt100": depth_stats(sep[hi]), "contact_1_100": depth_stats(sep[mid]),
         "gt100_first": depth_stats(sep[hi & first[:, None]]), "gt100_later": depth_stats(sep[hi & ~first[:, None]]),
         "gt100_share_first": float(hi[first].mean()) if first.any() else None,
         "gt100_share_later": float(hi[~first].mean()),
         "per_pad_gt100_share": {p: float(hi[:, i].mean()) for i, p in enumerate(pads)}}
    if hi.any():
        c = np.corrcoef(f[hi], -sep[hi])[0, 1] if np.isfinite(sep[hi]).all() and hi.sum() > 2 else None
        r["corr_force_pen_gt100"] = None if c is None or not np.isfinite(c) else float(c)
    res[os.path.basename(d.rstrip("/"))] = r
json.dump(res, open(args.out, "w"), indent=1)
for k, r in res.items():
    g, m = r["gt100"], r["contact_1_100"]
    print(f"{k:34s} rows {r['rows']:6d} >100N {100 * r['gt100_share']:5.2f}% max {r['max_force']:6.0f}N | "
          f">100N pen med {g.get('pen_median_mm', float('nan')):5.2f} p90 {g.get('pen_p90_mm', float('nan')):5.2f} "
          f">0.2 {100 * g.get('gt0.2mm', float('nan')):4.0f}% >0.5 {100 * g.get('gt0.5mm', float('nan')):4.0f}% | "
          f"1-100N pen med {m.get('pen_median_mm', float('nan')):5.2f} >0.5 {100 * m.get('gt0.5mm', float('nan')):4.0f}% | "
          f">100N share first/later {100 * (r['gt100_share_first'] or 0):5.2f}/{100 * r['gt100_share_later']:5.2f}%")
