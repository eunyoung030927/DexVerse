"""CV2DP summary (261115): tables, exact McNemar and failure diagnostics from artifacts/cv2dp/{sel,final}/*/metrics.json
and traces. New file. Container python (numpy), cwd latent_codec_v1:  python -m xhand_latent.cv2dp_summary
Writes artifacts/cv2dp/cv2dp_summary.json, cv2dp_episodes.csv and the auto part of RESULTS_CV2DP.md (RESULTS_AUTO.md).
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

CV = Path("artifacts/cv2dp")
TRAIN = ("shadow", "wuji")
TEST = "sharpa"
HANDS = (TEST,) + TRAIN


def metrics(d):
    p = d / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def rows(kind, arm, step, hand):
    """{episode: record(+ max_rise_cm)} over every chunk dir of (kind, arm, step, hand)."""
    out = {}
    dirs = list((CV / kind / f"{arm}_{step}").glob(f"{hand}_ep*"))
    if kind == "final" and hand in TRAIN:  # eps 0-9 of the training hands = selection rows of the best step
        dirs += list((CV / "sel" / f"{arm}_{step}").glob(f"{hand}_ep*"))
    for d in dirs:
        m = metrics(d)
        if not m:
            continue
        for e in m["episodes"]:
            r = dict(e)
            tp = d / "traces" / f"ep{e['episode']:03d}.npz"
            if tp.exists():
                z = np.load(tp)["obj"][:, 2]
                r["max_rise_cm"] = float((z.max() - z[min(10, len(z) - 1)]) * 100)
            out[e["episode"]] = r
    return out


def mcnemar(a, b):
    """a, b: {ep: success}. exact two-sided binomial test on discordant pairs."""
    eps = sorted(set(a) & set(b))
    n10 = sum(1 for e in eps if a[e] and not b[e])
    n01 = sum(1 for e in eps if b[e] and not a[e])
    n = n10 + n01
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(n10, n01) + 1)) / 2 ** n)
    return {"n_pairs": len(eps), "a_only": n10, "b_only": n01, "both": sum(1 for e in eps if a[e] and b[e]),
            "neither": sum(1 for e in eps if not a[e] and not b[e]), "p_exact": p}


def diag(R):
    if not R:
        return {}
    v = list(R.values())
    rise = [r.get("max_rise_cm", np.nan) for r in v]
    fails = [r for r in v if not r["success"]]
    return {"n": len(v), "success": sum(r["success"] for r in v),
            "lift_ep": sum(1 for r in v if r.get("t_lift", -1) >= 0),
            "rise_ge_1cm": int(np.sum(np.array(rise) >= 1.0)), "rise_ge_5cm": int(np.sum(np.array(rise) >= 5.0)),
            "median_max_rise_cm": float(np.nanmedian(rise)),
            "median_max_rise_cm_failures": float(np.nanmedian([r.get("max_rise_cm", np.nan) for r in fails])) if fails else None,
            "decode_fail_eps": sum(1 for r in v if r.get("decode_fail_steps", 0) > 0),
            "decode_fail_steps_total": int(sum(r.get("decode_fail_steps", 0) for r in v)),
            "mean_steps_success": float(np.mean([r["steps"] for r in v if r["success"]])) if any(r["success"] for r in v) else None}


def main():
    arms = json.loads((CV / "ctl/arms.json").read_text()) if (CV / "ctl/arms.json").exists() else {}
    S = {"arms": arms, "selection": {}, "final": {}, "mcnemar_sharpa": {}, "diag": {}}
    md = ["## Selection (training hands shadow + wuji, eps 0-9 each, /20)", ""]
    csv = ["arm,step,kind,hand,episode,success,steps,t_lift,max_rise_cm,decode_fail_steps"]
    succ = {}
    for arm in arms:
        bp = CV / "sel" / f"{arm}_best.json"
        b = json.loads(bp.read_text()) if bp.exists() else None
        S["selection"][arm] = b
        if b:
            if len(md) == 2:
                md += ["| arm | " + " | ".join(f"ep{50 * (i + 1)} ({s})" for i, s in enumerate(b["totals"])) + " | best |",
                       "|---|" + "---|" * (len(b["totals"]) + 1)]
            md.append(f"| {arm} | " + " | ".join(
                f"{b['per_hand'][s]['shadow']}+{b['per_hand'][s]['wuji']}={b['totals'][s]}" for s in b["totals"])
                + f" | {b['best']} |")
            for s in b["totals"]:
                for h in TRAIN:
                    for e, r in sorted(rows("sel", arm, s, h).items()):
                        csv.append(f"{arm},{s},sel,{h},{e},{int(r['success'])},{r['steps']},{r.get('t_lift', -1)},"
                                   f"{r.get('max_rise_cm', float('nan')):.2f},{r.get('decode_fail_steps', 0)}")
    md += ["", "(cells: shadow + wuji = total /20)", "", "## Final (best snapshot)", "",
           "| arm (best step) | sharpa (unseen) | shadow (train) | wuji (train) |", "|---|---|---|---|"]
    for arm in arms:
        b = S["selection"][arm]
        if not b:
            continue
        cells = []
        for h in HANDS:
            R = rows("final", arm, b["best"], h)
            succ[(arm, h)] = {e: bool(r["success"]) for e, r in R.items()}
            S["final"][f"{arm}/{h}"] = [sum(succ[(arm, h)].values()), len(R)]
            S["diag"][f"{arm}/{h}"] = diag(R)
            cells.append(f"{sum(succ[(arm, h)].values())}/{len(R)}" if R else "-")
            for e, r in sorted(R.items()):
                if not (h in TRAIN and e < 10):
                    csv.append(f"{arm},{b['best']},final,{h},{e},{int(r['success'])},{r['steps']},{r.get('t_lift', -1)},"
                               f"{r.get('max_rise_cm', float('nan')):.2f},{r.get('decode_fail_steps', 0)}")
        md.append(f"| {arm} ({b['best']}) | " + " | ".join(cells) + " |")
    md += ["", "## Exact McNemar on sharpa (paired by episode)", "",
           "| A vs B | pairs | A only | B only | both | neither | p (exact, two-sided) |", "|---|---|---|---|---|---|---|"]
    names = [a for a in arms if (a, TEST) in succ]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = mcnemar(succ[(a, TEST)], succ[(b, TEST)])
            S["mcnemar_sharpa"][f"{a}_vs_{b}"] = m
            md.append(f"| {a} vs {b} | {m['n_pairs']} | {m['a_only']} | {m['b_only']} | {m['both']} | {m['neither']} | "
                      f"{m['p_exact']:.4g} |")
    md += ["", "## Failure diagnostics (final cells)", "",
           "| arm / hand | n | success | lifted (t_lift>=0) | max rise >=1 cm | >=5 cm | median max rise cm (all / failures) "
           "| eps with decode failures (steps) |", "|---|---|---|---|---|---|---|---|"]
    for k, d in S["diag"].items():
        if d:
            mf = d["median_max_rise_cm_failures"]
            md.append(f"| {k} | {d['n']} | {d['success']} | {d['lift_ep']} | {d['rise_ge_1cm']} | {d['rise_ge_5cm']} | "
                      f"{d['median_max_rise_cm']:.2f} / {'-' if mf is None else f'{mf:.2f}'} | "
                      f"{d['decode_fail_eps']} ({d['decode_fail_steps_total']}) |")
    (CV / "cv2dp_summary.json").write_text(json.dumps(S, indent=1))
    (CV / "cv2dp_episodes.csv").write_text("\n".join(csv) + "\n")
    (CV / "RESULTS_AUTO.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
