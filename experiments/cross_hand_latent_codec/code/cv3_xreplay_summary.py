"""Summarize codec v3 cross-hand Kettle replay (artifacts/codec_v3/xreplay/*/replay/*.jsonl) and native round trip
(artifacts/codec_v3/replay/out/*.jsonl). Plain python3. Exact two-sided McNemar (binomial on discordant pairs) vs e2s and v2f."""
import json, sys
from math import comb
from pathlib import Path
R = Path("artifacts/codec_v3")
rows = {}
for f in sorted(R.glob("xreplay/*/replay/*.jsonl")):
    for l in f.read_text().splitlines():
        r = json.loads(l); c = r["variant"].removeprefix("arm_")
        rows[(c, r["replay_hand"], r["episode"])] = bool(r["success"])
conds = sorted({k[0] for k in rows}, key=lambda c: (c not in ("e2s", "v2p", "v2f"), c))
hands = ["allegro", "leap", "shadow", "wuji"]
def mcn(a, b):
    keys = [k for k in rows if k[0] == a and (b, k[1], k[2]) in rows]
    n01 = sum(rows[k] and not rows[(b, k[1], k[2])] for k in keys); n10 = sum((not rows[k]) and rows[(b, k[1], k[2])] for k in keys)
    n = n01 + n10
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p
print("| condition | " + " | ".join(hands) + " | total /40 | vs E2* (+/−, p) | vs v2f (+/−, p) |")
print("|---|" + "---|" * (len(hands) + 3))
for c in conds:
    per = [sum(rows.get((c, h, e), False) for e in range(10)) for h in hands]
    n = [sum((c, h, e) in rows for e in range(10)) for h in hands]
    a = mcn(c, "e2s") if c != "e2s" else None; b = mcn(c, "v2f") if c != "v2f" else None
    fmt = lambda t: f"+{t[0]}/−{t[1]}, p={t[2]:.2f}" if t else "–"
    print(f"| {c} | " + " | ".join(f"{p}/{m}" for p, m in zip(per, n)) + f" | {sum(per)}/{sum(n)} | {fmt(a)} | {fmt(b)} |")
nat = {}
for f in sorted(R.glob("replay/out/*.jsonl")):
    for l in f.read_text().splitlines():
        r = json.loads(l)
        if r.get("variant") == "native":
            continue
        nat.setdefault((f.stem.split("__")[-1], r.get("replay_hand", r.get("target"))), []).append((r["episode"], bool(r["success"])))
if nat:
    print("\nnative round trip (Z arm, eps 0-9):")
    for (arm, h), v in sorted(nat.items()):
        v = dict(v); print(f"  {arm} {h}: {sum(v.values())}/{len(v)} failed eps {[e for e, s in sorted(v.items()) if not s]}")
