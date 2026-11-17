"""cv3alt: tally replay rows (host python3, no deps). Cross-hand: artifacts/codec_v3alt/replay/out/*.jsonl (variant = arm_<cond>);
round trip: artifacts/codec_v3alt/rt/out/*.jsonl (variant codec vs native). Prints per condition x hand successes and per-episode
vectors (for exact McNemar), writes artifacts/codec_v3alt/replay/tally.json."""
import glob
import json
import math
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "artifacts/codec_v3alt"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(pattern):
    rows = defaultdict(dict)
    for f in glob.glob(pattern):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            hand = r.get("replay_hand") or r.get("target")
            rows[(r["variant"], hand)][r["episode"]] = bool(r["success"])
    return rows


x = load(f"{ROOT}/replay/out/*.jsonl")
rt = load(f"{ROOT}/rt/out/*.jsonl")
out = {"cross": {}, "roundtrip": {}}
conds = sorted({v for v, _ in x})
for v in conds:
    per = {h: sum(e.values()) for (vv, h), e in x.items() if vv == v}
    n = {h: len(e) for (vv, h), e in x.items() if vv == v}
    out["cross"][v] = {"per_hand": per, "n": n, "total": sum(per.values()), "n_total": sum(n.values())}
    print("X", v, per, f"{sum(per.values())}/{sum(n.values())}")
ref = "arm_e2ref"
for v in conds:
    if v == ref:
        continue
    b = c = 0
    for (vv, h), e in x.items():
        if vv != v or (ref, h) not in x:
            continue
        for ep, s in e.items():
            s0 = x[(ref, h)].get(ep)
            if s0 is None:
                continue
            b += s and not s0; c += s0 and not s
    out["cross"][v]["vs_e2ref_gain_loss_p"] = [b, c, mcnemar(b, c)]
    print("  vs e2ref", v, b, c, round(mcnemar(b, c), 3))
for (v, h), e in sorted(rt.items()):
    out["roundtrip"].setdefault(v, {})[h] = f"{sum(e.values())}/{len(e)}"
print("RT", json.dumps(out["roundtrip"]))
json.dump(out, open(f"{ROOT}/replay/tally.json", "w"), indent=1)
