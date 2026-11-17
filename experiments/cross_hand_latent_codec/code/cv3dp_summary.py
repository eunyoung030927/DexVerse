"""CV3DP summary (261116; 261117 A2/A3 update): tables + the PREREG tests (exact McNemar etc.) from the cv3dp sel/final
metrics.json files. New file, python3 stdlib only (host). cwd = latent_codec_v1:  python3 xhand_latent/cv3dp_summary.py
Results root: artifacts/codec_v3/cv3dp (work1-owned job dirs are mirrored into the same tree by the PC split relay).
A2: success = success at or before step 1500 (records with a 3000 cap are re-scored with the same rule; the context groups
too, so every cell in this file uses one cap).
Tests: P1 sharpa / P2 leap / P3 held-out pooled / T training hands for Zw3 vs Ze3 (PREREG); D1 Dw3 vs Zw3, D2 De3 vs Ze3,
D3 Dw3 vs De3 (A3), each with the same sub-tests.
Context (descriptive only, different training hands):
  cv2dp Ze2/Zv2: artifacts/codec_v3/cv3dp/ext/cv2dp/{sel/<arm>_best.json, final/<arm>_<step>/sharpa_ep*/metrics.json}
  H2 Z/D:        artifacts/pooled3hand/ho/h2/{sel/<arm>_best.json, final/<arm>_<step>/<hand>_ep*/metrics.json}
Writes artifacts/codec_v3/cv3dp/{cv3dp_summary.json, cv3dp_episodes.csv, RESULTS_AUTO.md}.
"""
from __future__ import annotations

import json
from math import comb, sqrt
from pathlib import Path

CV = Path("artifacts/codec_v3/cv3dp")
ROOT = CV
TRAIN = ("shadow", "wuji", "allegro")
HELD = ("sharpa", "leap")
CAP = 1500
COMPARISONS = (("P", "Zw3", "Ze3", "codec (Z wrist): wb_cl_l vs E2*"),
               ("D1", "Dw3", "Zw3", "wrist method D vs Z (wb_cl_l z)"),
               ("D2", "De3", "Ze3", "wrist method D vs Z (E2* z)"),
               ("D3", "Dw3", "De3", "codec (D wrist): wb_cl_l vs E2*"))


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def ok(e):
    return bool(e["success"]) and int(e.get("steps", 0)) <= CAP


def cell(root, kind, arm, step, pat):
    """{episode: success (A2 cap)} over every chunk dir matching pat under root/kind/arm_step."""
    out = {}
    for d in sorted((root / kind / f"{arm}_{step}").glob(pat)):
        m = jl(d / "metrics.json")
        for e in (m or {}).get("episodes", []):
            out[e["episode"]] = ok(e)
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


def verdict(m, n, A):
    if m["n"] < n:
        return f"incomplete ({m['n']}/{n})"
    if m["p_exact"] < 0.05 and m["delta"] >= 5 * n / 50:
        return f"{A} better"
    if m["p_exact"] < 0.05 and m["delta"] <= -5 * n / 50:
        return f"{A} worse"
    return "no difference detected"


def fmt(m, A, B):
    return (f"{A} {m['a']} vs {B} {m['b']} (n {m['n']}; {A}-only {m['a_only']}, {B}-only {m['b_only']}; Δ {m['delta']:+d}, "
            f"95% CI {m['ci95_delta_frac'][0]:+.2f}..{m['ci95_delta_frac'][1]:+.2f}; exact p {m['p_exact']:.3g})")


def compare(F, A, B):
    T = {}
    for h in HELD:
        T[h] = m = mcnemar(F.get((A, h), {}), F.get((B, h), {}))
        m["verdict"] = verdict(m, 50, A)
    T["heldout_pooled"] = mcnemar({(h, e): v for h in HELD for e, v in F.get((A, h), {}).items()},
                                  {(h, e): v for h in HELD for e, v in F.get((B, h), {}).items()})
    for h in TRAIN:
        T[h] = mcnemar(F.get((A, h), {}), F.get((B, h), {}))
        T[f"{h}_eps10-49"] = mcnemar({e: v for e, v in F.get((A, h), {}).items() if e >= 10},
                                     {e: v for e, v in F.get((B, h), {}).items() if e >= 10})
    ta = {(h, e): v for h in TRAIN for e, v in F.get((A, h), {}).items()}
    tb = {(h, e): v for h in TRAIN for e, v in F.get((B, h), {}).items()}
    T["train_pooled"] = m = mcnemar(ta, tb)
    m["noninferior_margin12"] = (m["a"] >= m["b"] - 12) if m["n"] == 150 else None
    T["train_pooled_eps10-49"] = mcnemar({k: v for k, v in ta.items() if k[1] >= 10}, {k: v for k, v in tb.items() if k[1] >= 10})
    return T


def main():
    arms = jl(CV / "ctl/arms.json") or {}
    S = {"root": str(ROOT), "cap_steps": CAP, "selection": {}, "final": {}, "tests": {}, "context": {}}
    md = [f"(auto; results root `{ROOT}`; success = success within {CAP} steps, A2)", "",
          "### Selection (training hands, eps 0-9, /30; held-out hands unused)", ""]
    csv = ["arm,step,kind,hand,episode,success_cap1500"]
    best = {}
    for arm in arms:
        b = jl(ROOT / "sel" / f"{arm}_best.json")
        S["selection"][arm] = b
        if not b:
            continue
        best[arm] = b["best"]
    if best:
        steps = list(next(iter(S["selection"][a] for a in best))["totals"])
        md += ["| arm | " + " | ".join(f"ep{50 * (i + 1)} ({s})" for i, s in enumerate(steps)) + " | best |",
               "|---|" + "---|" * (len(steps) + 1)]
    for arm in arms:
        b = S["selection"][arm]
        if not b:
            md.append(f"| {arm} | " + " | ".join("대기" for _ in range(5)) + " | 대기 |")
            continue
        md.append(f"| {arm} | " + " | ".join("+".join(str(b["per_hand"][s][h]) for h in TRAIN) + f"={b['totals'][s]}"
                                            for s in b["totals"]) + f" | {b['best']} |")
    md += ["", "(cells: shadow+wuji+allegro = total /30, as recorded by the driver)", ""]
    F = {}
    for arm, st in best.items():
        for h in HELD:
            F[(arm, h)] = cell(ROOT, "final", arm, st, f"{h}_ep*")
        for h in TRAIN:
            c = cell(ROOT, "final", arm, st, f"{h}_ep*")
            c.update(cell(ROOT, "sel", arm, st, f"{h}_ep*"))  # eps 0-9 = selection rows of the best step
            F[(arm, h)] = c
        for h in HELD + TRAIN:
            for e, v in sorted(F[(arm, h)].items()):
                csv.append(f"{arm},{st},{'sel' if (h in TRAIN and e < 10) else 'final'},{h},{e},{int(v)}")
    md += ["### Final (best snapshot)", "",
           "| arm (best) | sharpa held-out /50 | leap held-out /50 | shadow /50 | wuji /50 | allegro /50 | train total /150 |",
           "|---|---|---|---|---|---|---|"]
    for arm in arms:
        if arm not in best:
            md.append(f"| {arm} | 대기 | 대기 | 대기 | 대기 | 대기 | 대기 |")
            continue
        g = lambda h: F.get((arm, h), {})  # noqa: E731
        S["final"][arm] = {h: [sum(g(h).values()), len(g(h))] for h in HELD + TRAIN}
        tr = sum(sum(g(h).values()) for h in TRAIN), sum(len(g(h)) for h in TRAIN)
        md.append(f"| {arm} ({best[arm]}) | " + " | ".join(f"{sum(g(h).values())}/{len(g(h))}" for h in HELD + TRAIN)
                  + f" | {tr[0]}/{tr[1]} |")
    for tag, A, B, what in COMPARISONS:
        if A not in best or B not in best:
            md += ["", f"### {tag}: {A} vs {B} ({what}) — 대기"]
            continue
        T = S["tests"][tag] = compare(F, A, B)
        md += ["", f"### {tag}: {A} vs {B} ({what}), exact two-sided McNemar on episode pairs", "",
               f"- **sharpa**: {fmt(T['sharpa'], A, B)} → **{T['sharpa']['verdict']}**",
               f"- **leap**: {fmt(T['leap'], A, B)} → **{T['leap']['verdict']}**",
               f"- held-out pooled (n 100): {fmt(T['heldout_pooled'], A, B)}",
               f"- training hands pooled /150: {fmt(T['train_pooled'], A, B)}; non-inferior ({A} ≥ {B} − 12): "
               f"**{T['train_pooled']['noninferior_margin12']}**",
               f"  - eps 10-49 only (/120): {fmt(T['train_pooled_eps10-49'], A, B)}"]
        md += [f"  - {h}: {fmt(T[h], A, B)}" for h in TRAIN]
    # ---- context: cv2dp Ze2/Zv2 (sharpa), H2 Z/D (sharpa, leap) - different training hands, descriptive only
    ctx = {}
    ext = CV / "ext/cv2dp"
    for arm in ("Ze2", "Zv2"):
        b = jl(ext / "sel" / f"{arm}_best.json")
        ctx[arm] = {"train": "shadow+wuji", "best": b and b["best"],
                    "sharpa": cell(ext, "final", arm, b["best"], "sharpa_ep*") if b else {}, "leap": {}}
    h2 = Path("artifacts/pooled3hand/ho/h2")
    for arm in ("Z", "D"):
        b = jl(h2 / "sel" / f"{arm}_best.json")
        ctx[f"H2-{arm}"] = {"train": "shadow+allegro", "best": b and b["best"],
                            "sharpa": cell(h2, "final", arm, b["best"], "sharpa_ep*") if b else {},
                            "leap": cell(h2, "final", arm, b["best"], "leap_ep*") if b else {}}
    frac = lambda c, miss: f"{sum(c.values())}/{len(c)}" if c else miss  # noqa: E731
    mc = lambda m: "-" if not m or not m["n"] else f"{m['a']} vs {m['b']}, p {m['p_exact']:.3g} (n {m['n']})"  # noqa: E731
    md += ["", "### Context (different training hands → descriptive, confounded; same 1500-step cap)", "",
           "| group | training hands | best | sharpa | leap | sharpa: Zw3 vs it | sharpa: Ze3 vs it | leap: Zw3 vs it | leap: Ze3 vs it |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k, v in ctx.items():
        s, l = v["sharpa"], v["leap"]
        r = {f"{h}_{a}": (mcnemar(F[(a, h)], c) if c and (a, h) in F else None)
             for h, c in (("sharpa", s), ("leap", l)) for a in ("Zw3", "Ze3")}
        S["context"][k] = {"train": v["train"], "best": v["best"], "sharpa": [sum(s.values()), len(s)],
                           "leap": [sum(l.values()), len(l)], **r}
        md.append(f"| {k} | {v['train']} | {v['best'] or '대기'} | {frac(s, '대기')} | "
                  f"{frac(l, '-' if k in ('Ze2', 'Zv2') else '대기')} | {mc(r['sharpa_Zw3'])} | {mc(r['sharpa_Ze3'])} | "
                  f"{mc(r['leap_Zw3'])} | {mc(r['leap_Ze3'])} |")
    md += ["", "H2 leap: eps 0-9, extended to 0-49 only where its prereg allowed. All cells re-scored with the 1500-step cap."]
    (CV / "cv3dp_summary.json").write_text(json.dumps(S, indent=1, default=str))
    (CV / "cv3dp_episodes.csv").write_text("\n".join(csv) + "\n")
    (CV / "RESULTS_AUTO.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
