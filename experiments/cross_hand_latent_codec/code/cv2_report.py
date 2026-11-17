"""codec v2 (cv2, 261115): gate table from cv2_measure outputs (artifacts/codec_v2/measure/<name>.json). New file.
Gates = DESIGN.md sec.4 (pre-registered). Plain python3 + json only.
  python3 xhand_latent/cv2_report.py [measure_dir] > artifacts/codec_v2/measure/gates.md
"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/codec_v2/measure")
ORDER = ["E2", "e2c", "ft0", "cyc01", "cyc05", "p05", "cyc05p05", "adv01", "adv05", "advm05", "clj05", "clf05", "cyc05clf05",
         "cyc05p05f002", "cyc05p05f01", "cyc05clf05f002"]
R = {p.stem: json.loads(p.read_text()) for p in D.glob("*.json")}
names = [n for n in ORDER if n in R] + sorted(n for n in R if n not in ORDER)
E = R["E2"]
e_rmse = E["quality"]["joint_rmse_rad"]
e_pad = E["quality"]["pad_residual_mm_mean"]


def g(r):
    m = lambda s: r[s]["dist"]["MEAN"]
    out = {
        "G1 lin GEO5/SYN5/GEO3": (r["GEO_ALL5"]["probe_raw"]["acc"], r["SYN_ALL5"]["probe_raw"]["acc"], r["GEO_H3"]["probe_raw"]["acc"]),
        "G2 same/within GEO5/SYN5": (m("GEO_ALL5")["same/within"], m("SYN_ALL5")["same/within"]),
        "G3 frac_hand SYN5/GEO5": (r["SYN_ALL5"]["two_way"]["frac_hand"], r["GEO_ALL5"]["two_way"]["frac_hand"]),
        "G4 rmse/E2* max": max(r["quality"]["joint_rmse_rad"][h] / e_rmse[h] for h in e_rmse),
        "G5 pad mm": r["quality"]["pad_residual_mm_mean"],
    }
    p1 = out["G1 lin GEO5/SYN5/GEO3"]
    passes = {
        "G1": p1[0] <= .40 and p1[1] <= .40 and p1[2] <= .50,
        "G2": all(x <= .60 for x in out["G2 same/within GEO5/SYN5"]),
        "G3": all(x <= .10 for x in out["G3 frac_hand SYN5/GEO5"]),
        "G4": all(r["quality"]["joint_rmse_rad"][h] <= max(1.15 * e_rmse[h], e_rmse[h] + .005) for h in e_rmse),
        "G5": out["G5 pad mm"] <= 1.10 * e_pad,
    }
    return out, passes


f = lambda x: "/".join(f"{v:.2f}" for v in x) if isinstance(x, tuple) else f"{x:.2f}"
print("| codec | G1 lin GEO5/SYN5/GEO3 (≤.40/.40/.50) | G2 same/within GEO5/SYN5 (≤.60) | G3 frac_hand SYN5/GEO5 (≤.10) "
      "| G4 max rmse/E2* | G5 pad mm (≤%.2f) | pass |" % (1.1 * e_pad))
print("|---|---|---|---|---|---|---|")
for n in names:
    o, p = g(R[n])
    marks = [f"{f(v)} {'✓' if p[k[:2]] else '✗'}" for k, v in o.items()]
    print(f"| {n} | " + " | ".join(marks) + f" | {'PASS' if all(p.values()) else 'fail'} |")
print()
print("| codec | kNN raw GEO5/SYN5 | kNN centered GEO5/SYN5 | same/rand GEO5/SYN5 | offset/within GEO5/SYN5 | RT5 drift cross/rand "
      "| RT5 lin | play lin | rmse allegro/leap/shadow/wuji/sharpa | K1 lin / η² / off÷within | K2 allegro kNN ratio vs H3-ref / vs LOHO / in95 / bin | K2 leap (same) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for n in names:
    r = R[n]
    m = lambda s: r[s]["dist"]["MEAN"]
    q = r["quality"]["joint_rmse_rad"]
    row = [f"{r['GEO_ALL5']['knn15_raw']:.2f}/{r['SYN_ALL5']['knn15_raw']:.2f}",
           f"{r['GEO_ALL5']['knn15_centered']:.2f}/{r['SYN_ALL5']['knn15_centered']:.2f}",
           f"{m('GEO_ALL5')['same/rand']:.2f}/{m('SYN_ALL5')['same/rand']:.2f}",
           f"{m('GEO_ALL5')['offset/within']:.2f}/{m('SYN_ALL5')['offset/within']:.2f}",
           f"{r['RT_ALL5']['rt_drift_cross/rand']:.2f}", f"{r['RT_ALL5']['probe_raw']['acc']:.2f}",
           f"{r['PLAY_unpaired_ALL5']['probe_raw']:.2f}",
           "/".join(f"{q[h]:.3f}" for h in ("allegro", "leap", "shadow", "wuji", "sharpa"))]
    if "KETTLE_K1" in r:
        k1, k2 = r["KETTLE_K1"], r["KETTLE_K2"]
        row.append(f"{k1['probe_raw']['acc']:.2f} / {k1['var']['eta2_hand']:.2f} / {k1['offset/within_mean']:.2f}")
        for u in ("allegro", "leap"):
            row.append(f"{k2[u]['knn10_median_ratio']:.1f} / {k2[u].get('knn10_median_ratio_vs_loho', float('nan')):.2f} / "
                       f"{k2[u]['frac_inside_ref_p95']:.2f} / {k2[u]['binary_lr_acc']:.2f}")
    else:
        row += ["-", "-", "-"]
    print(f"| {n} | " + " | ".join(row) + " |")
