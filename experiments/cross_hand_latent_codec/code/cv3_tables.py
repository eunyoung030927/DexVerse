"""Extra RESULTS tables from measure JSONs (plain python3)."""
import json
from pathlib import Path
M = Path("artifacts/codec_v3/measure")
names = ["E2", "cyc05p05", "cyc05clf05f002", "w_cl", "wv_cl", "wb_cl_l", "wn_cl_l", "v2n_cl", "wb_tax_l", "wb_cltax_l"]
names = [n for n in names if (M / f"{n}.json").exists()]
cats = ["5<->5", "5->4", "4->5", "4<->4"]
print("| codec | play val: 5↔5 / 5→4 / 4→5 / 4↔4 / all | taxonomy test (held-out samples + classes) | v2 closure pairs (cl_func) |")
print("|---|---|---|---|")
for n in names:
    r = json.loads((M / f"{n}.json").read_text())["retarget"]
    f = lambda k: " / ".join(f"{r[k]['by_category'].get(c, float('nan')):.1f}" for c in cats) + f" / {r[k]['overall_G5style']:.2f}"
    print(f"| {n} | {f('playval')} | {f('tax_test') if 'tax_test' in r else '–'} | {f('cl')} |")
print()
print("| codec | K1 Kettle H3 η²_hand | K2 kNN ratio allegro / leap (Euclid) | K2 in-H3-p95 allegro / leap (Euclid) | K2 whitened kNN ratio vs LOHO allegro / leap | K2 whitened in-p95 | RT5 drift cross/rand |")
print("|---|---|---|---|---|---|---|")
for n in names:
    c = json.loads((M / "cv2" / f"{n}.json").read_text()); w = json.loads((M / f"{n}.json").read_text())["KETTLE_K2_white"]
    k2 = c["KETTLE_K2"]
    print(f"| {n} | {c['KETTLE_K1']['var']['eta2_hand']:.2f} | {k2['allegro']['knn10_median_ratio']:.1f} / {k2['leap']['knn10_median_ratio']:.1f} | "
          f"{k2['allegro']['frac_inside_ref_p95']:.2f} / {k2['leap']['frac_inside_ref_p95']:.2f} | "
          f"{w['allegro']['knn10_median_ratio_vs_loho']:.2f} / {w['leap']['knn10_median_ratio_vs_loho']:.2f} | "
          f"{w['allegro']['frac_inside_ref_p95']:.2f} / {w['leap']['frac_inside_ref_p95']:.2f} | {c['RT_ALL5']['rt_drift_cross/rand']:.2f} |")
