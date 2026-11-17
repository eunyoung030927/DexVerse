"""codec v3 Phase B coverage metrics (TAXONOMY_SPEC §6.3) for taxonomy pairs vs play_v1 (unpaired) vs v2 cl_func pairs. CPU.
C1 fingertip voxel occupancy: tips / s in the palm frame, 0.05 grid, per finger; occupancy = |vox(data) ∩ vox(ref)| / |vox(ref)|,
   ref = 50k uniform-joint Monte-Carlo samples (reachable set). Each dataset subsampled to the same N (default 5000).
C2 thumb–finger pairwise-distance space (|T-I|,|T-M|,|T-R|,|I-M|,|M-R|)/s: 0.1 grid occupancy vs ref and covering radius
   (95th pct over 3000 ref points of the distance to the nearest data point).
C3 Kapandji reach (taxonomy only, valid fraction per K posture); C4 (posture x aperture quartile) cells with >= 8 rows valid
   on all 5 hands / on H3.
  python.sh -m xhand_latent.cv3_coverage --tax artifacts/codec_v3/data/tax_pairs.npz --out artifacts/codec_v3/data/coverage.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .arm_train import load_arm
from .cv2_measure import E2
from .cv2_train import ClosureSampler

HANDS = ("allegro", "leap", "shadow", "wuji", "sharpa")


def tips(fk, q, s):
    with torch.no_grad():
        return fk(torch.as_tensor(q, dtype=torch.float32)).numpy() / s


def vox(T, nf, g=0.05):
    return {(f, *v) for f in range(nf) for v in np.floor(T[:, f] / g).astype(int)}


def pdist5(T):
    return np.stack([np.linalg.norm(T[:, 0] - T[:, 1], axis=1), np.linalg.norm(T[:, 0] - T[:, 2], axis=1),
                     np.linalg.norm(T[:, 0] - T[:, 3], axis=1), np.linalg.norm(T[:, 1] - T[:, 2], axis=1),
                     np.linalg.norm(T[:, 2] - T[:, 3], axis=1)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tax", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    torch.set_num_threads(8)
    rng = np.random.default_rng(0)
    _, fks, p0 = load_arm(E2)
    sc = p0["config"]["hand_scales_m"]
    play = {h: np.load(f"artifacts/play_v1/{h}.npz", allow_pickle=True) for h in HANDS}
    d = np.load(args.tax, allow_pickle=True)
    V = d["valid"]
    smp = ClosureSampler(play, fks, sc, "cpu", mode="func"); smp.rng = np.random.default_rng(9)
    cl = {h: q.numpy() for h, q in smp.sample(args.n).items()}
    sets = {}
    for i, h in enumerate(HANDS):
        tq = d[f"q_{h}"][V[:, i]]
        tq_all5 = d[f"q_{h}"][V.all(1)]
        pl = play[h]["train"]
        sets[h] = {"taxonomy": tq, "taxonomy_all5_paired": tq_all5, "play_v1": pl, "v2_cl_func": cl[h],
                   "play_v1_pinch_synergy_quarter": pl[len(pl) // 2:]}
    res = {"n_per_set": args.n, "per_hand": {}}
    for h in HANDS:
        fk = fks[h]; s = sc[h]; nf = 5 if bool(fk.valid.all()) else 4
        ref = rng.uniform(-0.98, 0.98, (50000, fk.dof)).astype(np.float32)
        Tr = tips(fk, ref, s); Vr = vox(Tr, nf); Pr = pdist5(Tr); Cr = {tuple(c) for c in np.floor(Pr / 0.1).astype(int)}
        refp = torch.tensor(Pr[rng.choice(len(Pr), 3000, replace=False)])
        out = {"ref_voxels": len(Vr), "ref_pd_cells": len(Cr)}
        for name, Q in sets[h].items():
            if len(Q) == 0:
                out[name] = None
                continue
            Q = Q[rng.choice(len(Q), min(args.n, len(Q)), replace=False)]
            T = tips(fk, Q, s); vv = vox(T, nf); P = pdist5(T)
            cc = {tuple(c) for c in np.floor(P / 0.1).astype(int)}
            cov_r = torch.cdist(refp, torch.tensor(P)).min(1).values.numpy()
            out[name] = {"n": int(len(Q)), "C1_voxel_occupancy": len(vv & Vr) / len(Vr), "C2_pd_cell_occupancy": len(cc & Cr) / len(Cr),
                         "C2_covering_radius_p95": float(np.percentile(cov_r, 95)), "C2_covering_radius_median": float(np.median(cov_r))}
        res["per_hand"][h] = out
        print(h, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in out.items()}), flush=True)
    names = [str(x) for x in d["class_names"]]
    cid = d["class_id"]; th = d["theta"]
    h3 = [HANDS.index(h) for h in ("sharpa", "shadow", "wuji")]
    c4 = {"all5": 0, "h3": 0, "cells": 0}
    for c in np.unique(cid):
        m = cid == c
        a = th[m, 0]
        if np.all(np.isnan(a)):
            continue
        qs = np.nanquantile(a, [0.25, 0.5, 0.75])
        b = np.searchsorted(qs, a)
        for k in range(4):
            mm = b == k
            c4["cells"] += 1
            c4["all5"] += int(V[m][mm].all(1).sum() >= 8)
            c4["h3"] += int(V[m][mm][:, h3].all(1).sum() >= 8)
    res["C4_cells_ge8"] = c4
    res["C3_kapandji_valid_frac"] = {names[c]: {h: float(V[cid == c][:, i].mean()) for i, h in enumerate(HANDS)}
                                     for c in np.unique(cid) if names[c].startswith("S-K")}
    for key in ("taxonomy", "taxonomy_all5_paired", "play_v1", "v2_cl_func", "play_v1_pinch_synergy_quarter"):
        vals = [res["per_hand"][h][key] for h in HANDS if res["per_hand"][h].get(key)]
        res.setdefault("summary", {})[key] = {m: {"mean": float(np.mean([v[m] for v in vals])), "min": float(np.min([v[m] for v in vals]))}
                                              for m in ("C1_voxel_occupancy", "C2_pd_cell_occupancy", "C2_covering_radius_p95")}
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1), json.dumps(c4), flush=True)


if __name__ == "__main__":
    main()
