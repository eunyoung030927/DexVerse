"""cv3alt (261116): coverage of the cv3alt pair datasets with the SAME C1/C2 definitions as the v3 agent's cv3_coverage.py
(TAXONOMY_SPEC §6.3; helper functions copied verbatim, same seed / N / reference): C1 fingertip voxel occupancy (tips/s, 0.05 grid,
vs 50k uniform-joint reachable-set reference), C2 thumb-finger pairwise-distance cell occupancy (0.1 grid) and covering radius.
Extra (C3-like, no taxonomy labels here): opposition rate = fraction with min thumb-finger tip distance < 0.10 s.
Compared sets per hand: play_v1 (unpaired), v2 cl_func pairs, v2 SYN prototype pairs, and each cv3alt dataset (valid rows of
that hand, and rows valid on all 5 hands). New file. CPU.
  python.sh -m xhand_latent.cv3alt_coverage --out artifacts/codec_v3alt/data/coverage.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .arm_train import load_arm
from .cv2_measure import E2
from .cv2_train import ClosureSampler, SynSampler

HANDS = ("allegro", "leap", "shadow", "wuji", "sharpa")
DATA = {"playik": "artifacts/codec_v3alt/data/playik.npz", "sobol": "artifacts/codec_v3alt/data/sobol.npz",
        "human": "artifacts/codec_v3alt/data/human.npz"}


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
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    torch.set_num_threads(8)
    rng = np.random.default_rng(0)
    _, fks, p0 = load_arm(E2)
    sc = p0["config"]["hand_scales_m"]
    play = {h: np.load(f"artifacts/play_v1/{h}.npz", allow_pickle=True) for h in HANDS}
    smp = ClosureSampler(play, fks, sc, "cpu", mode="func"); smp.rng = np.random.default_rng(9)
    cl = {h: q.numpy() for h, q in smp.sample(args.n).items()}
    syn = {h: q.numpy() for h, q in SynSampler(play, HANDS, "cpu").sample(args.n, torch.Generator().manual_seed(5)).items()}
    D = {k: np.load(p, allow_pickle=True) for k, p in DATA.items() if Path(p).exists()}
    res = {"n_per_set": args.n, "per_hand": {}, "datasets": {}}
    for k, d in D.items():
        M = d["pmask_train"].astype(bool); F = d["feas_train"].astype(bool)
        res["datasets"][k] = {"n_train_scenes": int(len(M)), "n_val_scenes": int(len(d["pmask_val"])),
                              "feasible_rate": {h: float(F[:, i].mean()) for i, h in enumerate(HANDS)},
                              "pair_mask_rate": {h: float(M[:, i].mean()) for i, h in enumerate(HANDS)},
                              "all5_paired_rate": float(M.all(1).mean()), "h3_paired_rate": float(M[:, 2:].all(1).mean())}
    for h in HANDS:
        i = HANDS.index(h)
        fk = fks[h]; s = sc[h]; nf = 5 if bool(fk.valid.all()) else 4
        sets = {"play_v1": play[h]["train"], "v2_cl_func": cl[h], "v2_syn": syn[h]}
        for k, d in D.items():
            M = d["pmask_train"].astype(bool)
            sets[k] = d[f"Q_{h}_train"][M[:, i]]
            sets[f"{k}_all5_paired"] = d[f"Q_{h}_train"][M.all(1)]
        ref = rng.uniform(-0.98, 0.98, (50000, fk.dof)).astype(np.float32)
        Tr = tips(fk, ref, s); Vr = vox(Tr, nf); Pr = pdist5(Tr); Cr = {tuple(c) for c in np.floor(Pr / 0.1).astype(int)}
        refp = torch.tensor(Pr[rng.choice(len(Pr), 3000, replace=False)])
        out = {"ref_voxels": len(Vr), "ref_pd_cells": len(Cr)}
        for name, Q in sets.items():
            if len(Q) == 0:
                out[name] = None
                continue
            Q = Q[rng.choice(len(Q), min(args.n, len(Q)), replace=False)]
            T = tips(fk, Q, s); vv = vox(T, nf); P = pdist5(T)
            cc = {tuple(c) for c in np.floor(P / 0.1).astype(int)}
            cov_r = torch.cdist(refp, torch.tensor(P)).min(1).values.numpy()
            out[name] = {"n": int(len(Q)), "C1_voxel_occupancy": len(vv & Vr) / len(Vr), "C2_pd_cell_occupancy": len(cc & Cr) / len(Cr),
                         "C2_covering_radius_p95": float(np.percentile(cov_r, 95)),
                         "C2_covering_radius_median": float(np.median(cov_r)),
                         "opposition_rate": float((P[:, :3].min(1) < 0.10).mean())}
        res["per_hand"][h] = out
        print(h, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in out.items()}), flush=True)
    keys = [k for k in res["per_hand"]["sharpa"] if isinstance(res["per_hand"]["sharpa"][k], dict)]
    for key in keys:
        vals = [res["per_hand"][h][key] for h in HANDS if res["per_hand"][h].get(key)]
        res.setdefault("summary", {})[key] = {m: {"mean": float(np.mean([v[m] for v in vals])), "min": float(np.min([v[m] for v in vals]))}
                                              for m in ("C1_voxel_occupancy", "C2_pd_cell_occupancy", "C2_covering_radius_p95",
                                                        "opposition_rate")}
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
