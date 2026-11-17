"""codec v3 gate table (DESIGN.md §5). Reads artifacts/codec_v3/measure/<id>.json (cv3_measure) and measure/cv2/<id>.json
(unmodified cv2_measure). Plain python3.  python3 xhand_latent/cv3_report.py [measure_dir] > artifacts/codec_v3/measure/gates.md"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/codec_v3/measure")
R = {p.stem: json.loads(p.read_text()) for p in D.glob("*.json")}
R = {k: v for k, v in R.items() if isinstance(v, dict) and "collapse" in v}
C2 = {p.stem: json.loads(p.read_text()) for p in (D / "cv2").glob("*.json")}
BASE = ["E2", "cyc05p05", "cyc05clf05f002"]
names = [n for n in BASE if n in R] + sorted(n for n in R if n not in BASE)
E, V = R["E2"], R["cyc05clf05f002"]
e_rmse = C2["E2"]["quality"]["joint_rmse_rad"]
e_pad = C2["E2"]["quality"]["pad_residual_mm_mean"]
e_cat = E["retarget"]["playval"]["by_category"]
has_tax = "TAXC_ALL5" in V
W1SETS = ["GEO_ALL5", "GEO_H3"] + (["TAXC_ALL5"] if has_tax else [])


def wr(r, s):
    return r[s]["white"]["same/rand"] if s in r else float("nan")


def gates(n):
    r, c = R[n], C2.get(n)
    w1 = [wr(r, s) for s in W1SETS]
    w1p = all(x <= 0.8 * wr(V, s) and x < wr(E, s) for x, s in zip(w1, W1SETS))
    col = r["collapse"]["pooled_playval"]
    ph = min(r["GEO_ALL5"]["perhand_pr"].values())
    w2p = col["pc1_share"] <= 0.30 and col["participation_ratio"] >= 6 and ph >= 3
    p3, p5 = r["GEO_H3"]["handid"]["probe_std_lr"], r["GEO_ALL5"]["handid"]["probe_std_lr"]
    v3, v5 = V["GEO_H3"]["handid"]["probe_std_lr"], V["GEO_ALL5"]["handid"]["probe_std_lr"]
    w3p = p3 <= v3 - 0.10 and p5 <= v5 - 0.10
    if c:
        rm = c["quality"]["joint_rmse_rad"]
        w4 = max(rm[h] / e_rmse[h] for h in e_rmse)
        w4p = all(rm[h] <= max(1.15 * e_rmse[h], e_rmse[h] + .005) for h in e_rmse)
        pad = c["quality"]["pad_residual_mm_mean"]
    else:
        w4, w4p, pad = float("nan"), False, r["retarget"]["playval"]["overall_G5style"]
    cat = r["retarget"]["playval"]["by_category"]
    w5p = pad <= 1.10 * e_pad and all(cat[k] <= 1.20 * e_cat[k] for k in e_cat)
    return dict(w1=w1, w1p=w1p, pc1=col["pc1_share"], pr=col["participation_ratio"], phpr=ph, w2p=w2p, p3=p3, p5=p5, w3p=w3p,
                w4=w4, w4p=w4p, pad=pad, cat=cat, w5p=w5p)


ck = lambda b: "✓" if b else "✗"
print(f"W1 sets: {', '.join(W1SETS)} (whitened same/rand-cross; pass ≤ 0.8×v2 and < E2*)\n")
print("| codec | W1 white same/rand | W2 PC1 / PR / min per-hand PR | W3 probe GEO-H3 / GEO-ALL5 | W4 max rmse/E2* | W5 pad mm (5↔5, 5→4, 4→5, 4↔4) | W1–W5 |")
print("|---|---|---|---|---|---|---|")
for n in names:
    g = gates(n)
    cat = g["cat"]
    allp = g["w1p"] and g["w2p"] and g["w3p"] and g["w4p"] and g["w5p"]
    print(f"| {n} | {'/'.join(f'{x:.2f}' for x in g['w1'])} {ck(g['w1p'])} | {g['pc1']:.2f} / {g['pr']:.1f} / {g['phpr']:.1f} {ck(g['w2p'])} "
          f"| {g['p3']:.2f} / {g['p5']:.2f} {ck(g['w3p'])} | {g['w4']:.2f} {ck(g['w4p'])} "
          f"| {g['pad']:.2f} ({cat.get('5<->5', 0):.1f}, {cat.get('5->4', 0):.1f}, {cat.get('4->5', 0):.1f}, {cat.get('4<->4', 0):.1f}) {ck(g['w5p'])} "
          f"| {'PASS' if allp else 'fail'} |")
print()
print("| codec | QDA GEO-H3 / ALL5 | kNN raw H3 / ALL5 | kNN white H3 / ALL5 | 4v5 / within-5 | white same/within GEO5 / GEO3 | SYN5 / CL5 white same/rand"
      + (" | TAXC H3 / TAXS ALL5 white s/r | TAX pad mm (test / held-out class)" if has_tax else "")
      + " | Kettle white in-p95 allegro / leap ; kNN ratio | v2-G1 lin GEO5/GEO3 | v2-G2 same/within GEO5 |")
print("|---|---|---|---|---|---|---|" + ("---|---|" if has_tax else "") + "---|---|---|")
for n in names:
    r, c = R[n], C2.get(n, {})
    h3, h5 = r["GEO_H3"]["handid"], r["GEO_ALL5"]["handid"]
    k = r.get("KETTLE_K2_white", {})
    row = [f"{h3['qda']:.2f} / {h5['qda']:.2f}", f"{h3['knn15_raw']:.2f} / {h5['knn15_raw']:.2f}",
           f"{h3['knn15_white']:.2f} / {h5['knn15_white']:.2f}", f"{h5['probe_4vs5']:.2f} / {h5['probe_within5']:.2f}",
           f"{r['GEO_ALL5']['white']['same/within']:.2f} / {r['GEO_H3']['white']['same/within']:.2f}",
           f"{wr(r, 'SYN_ALL5'):.2f} / {wr(r, 'CL_ALL5'):.2f}"]
    if has_tax:
        row += [f"{wr(r, 'TAXC_H3'):.2f} / {wr(r, 'TAXS_ALL5'):.2f}",
                f"{r['retarget']['tax_test']['overall_G5style']:.2f} / {r['retarget']['tax_heldout_class']['overall_G5style']:.2f}"]
    row += [f"{k.get('allegro', {}).get('frac_inside_ref_p95', float('nan')):.2f} / {k.get('leap', {}).get('frac_inside_ref_p95', float('nan')):.2f} ; "
            f"{k.get('allegro', {}).get('knn10_median_ratio', float('nan')):.1f} / {k.get('leap', {}).get('knn10_median_ratio', float('nan')):.1f}"]
    if c:
        row += [f"{c['GEO_ALL5']['probe_raw']['acc']:.2f} / {c['GEO_H3']['probe_raw']['acc']:.2f}",
                f"{c['GEO_ALL5']['dist']['MEAN']['same/within']:.2f}"]
    else:
        row += ["–", "–"]
    print(f"| {n} | " + " | ".join(row) + " |")
