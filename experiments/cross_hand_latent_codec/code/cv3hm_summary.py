"""CV3HM summary (261117; adapted from cv3dp_summary.py, unchanged original: paths/arm names cv3hm, no cv2dp/H2 context,
R0 ceiling table and the PREREG floor rule added). Original docstring:
CV3DP summary (261116): tables + the PREREG tests (exact McNemar etc.) from the cv3dp sel/final metrics.json files.
New file, python3 stdlib only (host). cwd = latent_codec_v1:  python3 xhand_latent/cv3dp_summary.py
Results root: artifacts/codec_v3/cv3dp/eval_work1 if it exists (work1 evaluated; copied by the PC relay), else
artifacts/codec_v3/cv3dp. Context (descriptive only, different training hands):
  cv2dp Ze2/Zv2: artifacts/codec_v3/cv3dp/ext/cv2dp/{sel/<arm>_best.json, final/<arm>_<step>/sharpa_ep*/metrics.json}
  H2 Z/D:        artifacts/pooled3hand/ho/h2/{sel/<arm>_best.json, final/<arm>_<step>/<hand>_ep*/metrics.json}
Writes artifacts/codec_v3/cv3dp/{cv3hm_summary.json, cv3hm_episodes.csv, RESULTS_AUTO.md}.
"""
from __future__ import annotations

import json
from math import comb, sqrt
from pathlib import Path

CV = Path("artifacts/codec_v3/cv3hm")
ROOT = CV
TRAIN = ("shadow", "wuji", "allegro")
A, B = "Zw3h", "Ze3h"


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def cell(root, kind, arm, step, pat):
    """{episode: success} over every chunk dir matching pat under root/kind/arm_step."""
    out = {}
    for d in sorted((root / kind / f"{arm}_{step}").glob(pat)):
        m = jl(d / "metrics.json")
        for e in (m or {}).get("episodes", []):
            out[e["episode"]] = bool(e["success"])
    return out


def mcnemar(a, b):
    keys = sorted(set(a) & set(b))
    n10 = sum(1 for k in keys if a[k] and not b[k])
    n01 = sum(1 for k in keys if b[k] and not a[k])
    n = n10 + n01
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(n10, n01) + 1)) / 2 ** n)
    N = len(keys)
    d = (n10 - n01) / N if N else 0.0
    se = sqrt(max(n10 + n01 - (n10 - n01) ** 2 / N, 0)) / N if N else 0.0
    return {"n": N, "a": sum(a[k] for k in keys), "b": sum(b[k] for k in keys), "a_only": n10, "b_only": n01,
            "both": sum(1 for k in keys if a[k] and b[k]), "neither": sum(1 for k in keys if not a[k] and not b[k]),
            "delta": n10 - n01, "ci95_delta_frac": [round(d - 1.96 * se, 3), round(d + 1.96 * se, 3)], "p_exact": p}


def sign_test(x, y):
    pos = sum(1 for i, j in zip(x, y) if i > j)
    neg = sum(1 for i, j in zip(x, y) if i < j)
    n = pos + neg
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(pos, neg) + 1)) / 2 ** n)
    return {"a_more": pos, "b_more": neg, "ties": len(x) - n, "p_exact": p}


def verdict(m, n):
    if m["n"] < n:
        return f"incomplete ({m['n']}/{n})"
    if m["p_exact"] < 0.05 and m["delta"] >= 5 * n / 50:
        return "Zw3h better"
    if m["p_exact"] < 0.05 and m["delta"] <= -5 * n / 50:
        return "Zw3h worse"
    return "no difference detected"


def fmt(m):
    return (f"{m['a']} vs {m['b']} (n {m['n']}; Zw3h-only {m['a_only']}, Ze3h-only {m['b_only']}; Δ {m['delta']:+d}, "
            f"95% CI {m['ci95_delta_frac'][0]:+.2f}..{m['ci95_delta_frac'][1]:+.2f}; p {m['p_exact']:.3g})")


def main():
    arms = jl(CV / "ctl/arms.json") or {}
    S = {"root": str(ROOT), "selection": {}, "final": {}, "tests": {}, "context": {}}
    md = [f"(auto, results root `{ROOT}`)", "", "### Selection (training hands, eps 0-9, /30; held-out hands unused)", ""]
    csv = ["arm,step,kind,hand,seed,episode,success"]
    best = {}
    for arm in arms:
        b = jl(ROOT / "sel" / f"{arm}_best.json")
        S["selection"][arm] = b
        if not b:
            md.append(f"- {arm}: 대기")
            continue
        best[arm] = b["best"]
        if not md[-1].startswith("|"):
            md += ["| arm | " + " | ".join(f"ep{50 * (i + 1)} ({s})" for i, s in enumerate(b["totals"])) + " | best |",
                   "|---|" + "---|" * (len(b["totals"]) + 1)]
        md.append(f"| {arm} | " + " | ".join("+".join(str(b["per_hand"][s][h]) for h in TRAIN) + f"={b['totals'][s]}"
                                            for s in b["totals"]) + f" | {b['best']} |")
    md += ["", "(cells: shadow+wuji+allegro = total /30)", ""]
    F = {}
    for arm, st in best.items():
        F[(arm, "sharpa")] = cell(ROOT, "final", arm, st, "sharpa_ep*")
        F[(arm, "leap")] = cell(ROOT, "final", arm, st, "leap_ep*")  # 261116 amendment: eps 0-49 x 1 run
        for h in TRAIN:
            c = cell(ROOT, "final", arm, st, f"{h}_ep*")
            c.update(cell(ROOT, "sel", arm, st, f"{h}_ep*"))  # eps 0-9 = selection rows of the best step
            F[(arm, h)] = c
        for (a_, h), c in F.items():
            if a_ == arm:
                for e, v in sorted(c.items()):
                    s = ""
                    csv.append(f"{arm},{st},{'final' if not (h in TRAIN and e < 10) else 'sel'},{h},{s},{e},{int(v)}")
    md += ["### Final (best snapshot)", "",
           "| arm (best) | sharpa held-out /50 | leap held-out /50 | leap eps 0-9 /10 | shadow /50 | wuji /50 | allegro /50 | train total /150 |",
           "|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        if arm not in best:
            md.append(f"| {arm} | 대기 | 대기 | 대기 | 대기 | 대기 | 대기 | 대기 |")
            continue
        g = lambda h: F.get((arm, h), {})  # noqa: E731
        lp = g("leap")
        S["final"][arm] = {h: [sum(g(h).values()), len(g(h))] for h in ("sharpa", "leap") + TRAIN}
        tr = sum(sum(g(h).values()) for h in TRAIN), sum(len(g(h)) for h in TRAIN)
        md.append(f"| {arm} ({best[arm]}) | {sum(g('sharpa').values())}/{len(g('sharpa'))} | "
                  f"{sum(lp.values())}/{len(lp)} | {sum(v for e, v in lp.items() if e < 10)}/{sum(1 for e in lp if e < 10)} | "
                  + " | ".join(f"{sum(g(h).values())}/{len(g(h))}" for h in TRAIN) + f" | {tr[0]}/{tr[1]} |")
    if A in best and B in best:
        T = S["tests"]
        T["P1_sharpa"] = m = mcnemar(F[(A, "sharpa")], F[(B, "sharpa")]); m["verdict"] = verdict(m, 50)
        T["P2_leap"] = m = mcnemar(F.get((A, "leap"), {}), F.get((B, "leap"), {})); m["verdict"] = verdict(m, 50)
        ho_a = {("s", k): v for k, v in F[(A, "sharpa")].items()} | {("l", k): v for k, v in F.get((A, "leap"), {}).items()}
        ho_b = {("s", k): v for k, v in F[(B, "sharpa")].items()} | {("l", k): v for k, v in F.get((B, "leap"), {}).items()}
        T["P3_heldout_pooled"] = mcnemar(ho_a, ho_b)
        for h in TRAIN:
            T[f"T_{h}"] = mcnemar(F[(A, h)], F[(B, h)])
            T[f"T_{h}_eps10-49"] = mcnemar({e: v for e, v in F[(A, h)].items() if e >= 10},
                                          {e: v for e, v in F[(B, h)].items() if e >= 10})
        ta = {(h, e): v for h in TRAIN for e, v in F[(A, h)].items()}
        tb = {(h, e): v for h in TRAIN for e, v in F[(B, h)].items()}
        T["T_pooled"] = m = mcnemar(ta, tb)
        m["noninferior_margin12"] = m["n"] == 150 and m["a"] >= m["b"] - 12
        T["T_pooled_eps10-49"] = mcnemar({k: v for k, v in ta.items() if k[1] >= 10}, {k: v for k, v in tb.items() if k[1] >= 10})
        md += ["", "### PREREG tests (Zw3h vs Ze3h, exact two-sided McNemar)", "",
               f"- **P1 sharpa** (paired by episode): {fmt(T['P1_sharpa'])} → **{T['P1_sharpa']['verdict']}**",
               f"- **P2 leap** (paired by episode, eps 0-49): {fmt(T['P2_leap'])} → **{T['P2_leap']['verdict']}**",
               f"- **P3 held-out pooled** (sharpa 50 + leap 50): {fmt(T['P3_heldout_pooled'])}",
               f"- **T training hands pooled** /150: {fmt(T['T_pooled'])}; non-inferior (Zw3h ≥ Ze3h − 12): "
               f"**{T['T_pooled']['noninferior_margin12']}**; eps 10-49 only (/120): {fmt(T['T_pooled_eps10-49'])}"]
        md += [f"  - {h}: {fmt(T[f'T_{h}'])}; eps 10-49: {fmt(T[f'T_{h}_eps10-49'])}" for h in TRAIN]
    # ---- R0 decode-replay ceiling (descriptive) + floor rule (PREREG)
    md += ["", "### R0 decode-replay ceiling (dataset rows, no policy; training hands eps 0-9, /10)", "",
           "| codec | " + " | ".join(TRAIN) + " |", "|---|" + "---|" * len(TRAIN)]
    for tag in ("v3wbcll", "e2"):
        row = []
        for h in TRAIN:
            m = jl(CV / "r0" / f"R0_{tag}" / f"{h}_ep0-9" / "metrics.json")
            row.append(f"{sum(e[success] for e in m[episodes])}/{len(m[episodes])}" if m else "대기")
        S["context"][f"R0_{tag}"] = row
        md.append(f"| {tag} | " + " | ".join(row) + " |")
    if A in best and B in best:
        T = S["tests"]
        for k, h in (("P1_sharpa", "sharpa"), ("P2_leap", "leap")):
            m = T[k]
            if m["n"] == 50 and m["a"] <= 2 and m["b"] <= 2:
                m["verdict"] = "floor; uninformative"
        tp = T["T_pooled"]
        if tp["n"] == 150 and tp["a"] <= 15 and tp["b"] <= 15:
            tp["floor"] = True
        md += ["", f"Floor rule: P1 {T[P1_sharpa][verdict]}, P2 {T[P2_leap][verdict]}, "
                   f"training pooled floor {tp.get(floor, False)}"]
    (CV / "cv3hm_summary.json").write_text(json.dumps(S, indent=1, default=str))
    (CV / "cv3hm_episodes.csv").write_text("\n".join(csv) + "\n")
    (CV / "RESULTS_AUTO.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
