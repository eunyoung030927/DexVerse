"""cv3alt: markdown gate tables from artifacts/codec_v3alt/measure/*.json (host python3, json only).
  python3 xhand_latent/cv3alt_report.py > artifacts/codec_v3alt/measure/gates.md"""
import json
import os

M = "artifacts/codec_v3alt/measure"
ORDER = ["E2", "v2clf", "v2p05", "playik_w", "playik_w3", "playik_w10", "sobol_w", "human_w", "mix_w"]
R = {n: json.load(open(f"{M}/{n}.json")) for n in ORDER if os.path.exists(f"{M}/{n}.json")}
E2 = R["E2"]["quality"]["joint_rmse_rad"]
LIM = {h: max(1.15 * v, v + 0.005) for h, v in E2.items()}
f = lambda x: f"{x:.2f}"
print("## Gate table (v2 GEO pair sets = identical scenes to codec_v2; G1–G5 thresholds as v2 DESIGN §4)\n")
print("| codec | G1 lin GEO5/SYN5/GEO3 | G2 Euclid same/within GEO5/SYN5 | G3 frac_hand SYN5/GEO5 | G4 max rmse/limit | G5 pad mm | "
      "whitened R_w GEO5/GEO3 | v3-white same/rand GEO5/GEO3 | QDA(cov) GEO5 | per-hand-white kNN GEO5/GEO3 | play-val PC1 / PR |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for n, r in R.items():
    g5, g3, s5 = r["v2GEO_ALL5"], r["v2GEO_H3"], r["v2SYN_ALL5"]
    q = r["quality"]
    g4 = max(q["joint_rmse_rad"][h] / LIM[h] for h in LIM)
    c = r.get("collapse_playval", {})
    print(f"| {n} | {f(g5['probe_raw'])}/{f(s5['probe_raw'])}/{f(g3['probe_raw'])} | {f(g5['R_euclid_same/within'])}/{f(s5['R_euclid_same/within'])} | "
          f"{s5['frac_hand']:.3f}/{g5['frac_hand']:.3f} | {g4:.2f}{' ✗' if g4 > 1 else ''} | {q['pad_residual_mm_mean']:.2f}{' ✗' if q['pad_residual_mm_mean'] > 6.80 else ''} | "
          f"{f(g5['R_whitened'])}/{f(g3['R_whitened'])} | {f(g5['v3white']['same/rand'])}/{f(g3['v3white']['same/rand'])} | {f(g5['gauss']['qda0'])} | "
          f"{f(g5['knn15_perhand_whitened'])}/{f(g3['knn15_perhand_whitened'])} | {c.get('pc1_share', float('nan')):.2f} / {c.get('participation_ratio', float('nan')):.1f} |")
print("\n## Held-out sets of the cv3alt datasets (lin probe / v3-white same/rand / QDA(cov) / per-hand-white kNN), ALL5 and H3\n")
sets = ["playik", "sobol", "human"]
print("| codec | " + " | ".join(f"{s} ALL5 | {s} H3" for s in sets) + " |")
print("|---|" + "---|" * (2 * len(sets)))
for n, r in R.items():
    cells = []
    for s in sets:
        for t in ("ALL5", "H3"):
            b = r.get(f"{s}_{t}")
            cells.append("–" if b is None else f"{f(b['probe_raw'])} / {f(b['v3white']['same/rand'])} / {f(b['gauss']['qda0'])} / {f(b['knn15_perhand_whitened'])}")
    print(f"| {n} | " + " | ".join(cells) + " |")
print("\n## Retarget pad residual per category (mm, E2* metric) — play val / v2 GEO pairs / playik val\n")
print("| codec | 5↔5 | 5→4 | 4→5 | 4↔4 (play val) | GEO 5↔5/5→4/4→5/4↔4 | playik-val 5↔5/5→4/4→5/4↔4 |")
print("|---|---|---|---|---|---|---|")
for n, r in R.items():
    c = r["retarget_playval"]["by_category"]; g = r["retarget_v2GEO"]["by_category"]; p = r.get("retarget_playik_ALL5", {}).get("by_category", {})
    k = ["5<->5", "5->4", "4->5", "4<->4"]
    print(f"| {n} | " + " | ".join(f(c[x]) for x in k) + " | " + "/".join(f(g[x]) for x in k) + " | " + ("/".join(f(p[x]) for x in k) if p else "–") + " |")
print("\n## Kettle (K1 H3 η²; K2 allegro/leap: Euclid kNN ratio, frac inside H3 p95, whitened frac inside)\n")
print("| codec | K1 η²_hand | allegro ratio / in-p95 / white in-p95 | leap ratio / in-p95 / white in-p95 |")
print("|---|---|---|---|")
for n, r in R.items():
    k = r.get("KETTLE")
    if not k:
        continue
    a, l = k["K2"]["allegro"], k["K2"]["leap"]
    print(f"| {n} | {k['K1']['var']['eta2_hand']:.2f} | {a['knn10_median_ratio']:.1f} / {a['frac_inside_ref_p95']:.2f} / {a['whitened_frac_inside_ref_p95']:.2f} | "
          f"{l['knn10_median_ratio']:.1f} / {l['frac_inside_ref_p95']:.2f} / {l['whitened_frac_inside_ref_p95']:.2f} |")
print("\n## DP criterion 1 (vs v2clf on GEO): R_w lower on ALL5 and H3, probe lower by ≥0.05 on one with the other not worse by >0.02, G4+G5 pass\n")
v = R["v2clf"]
for n, r in R.items():
    if n in ("E2", "v2clf", "v2p05"):
        continue
    rw = all(r[f"v2GEO_{t}"]["R_whitened"] < v[f"v2GEO_{t}"]["R_whitened"] for t in ("ALL5", "H3"))
    d = [v[f"v2GEO_{t}"]["probe_raw"] - r[f"v2GEO_{t}"]["probe_raw"] for t in ("ALL5", "H3")]
    pr = max(d) >= 0.05 and min(d) >= -0.02
    q = r["quality"]; g45 = max(q["joint_rmse_rad"][h] / LIM[h] for h in LIM) <= 1 and q["pad_residual_mm_mean"] <= 6.80
    print(f"- {n}: R_w lower {rw}, probe {pr} (Δ ALL5 {d[0]:+.2f}, H3 {d[1]:+.2f}), G4/G5 {g45} → {'PASS' if rw and pr and g45 else 'fail'}")
