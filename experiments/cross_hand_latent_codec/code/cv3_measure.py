"""codec v3 (cv3, 261116) measurement ADDITIONS (DESIGN.md §4). The v2-comparable metrics come from the unmodified
cv2_measure (run separately with --out artifacts/codec_v3/measure/cv2). New file, CPU.
Per codec (artifacts/codec_v3/measure/<name>.json):
  whitening W = ZCA of the codec's pooled play-val z (4000/hand x 5); collapse stats (PC1 share, PR, per-hand PR)
  paired sets (codec-independent): GEO_ALL5/GEO_H3/SYN_ALL5 (v2 pairsets.npz), CL_ALL5 (v2 cl_func pair draw, cached),
     TAXC (taxonomy held-out classes), TAXS (held-out samples of training classes) -> ALL5 (masked) and H3
     whitened same/rand-cross, same/within; std linear probe, QDA, kNN-15 raw and whitened; ALL5: 4-vs-5, within-5
  retarget pad residual per category (5<->5, 5->4, 4->5, 4<->4) with play val[:2048], TAXC+TAXS and CL as SOURCE postures
  Kettle K2 (allegro/leap vs H3 pool) in whitened coordinates
Container python, cwd /workspace/xhand/latent_codec_v1:
  CUDA_VISIBLE_DEVICES= nice -n 10 python.sh -m xhand_latent.cv3_measure --codec NAME=PATH [--tax artifacts/codec_v3/data/tax_pairs.npz]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .arm_pads import align_pair
from .arm_train import load_arm
from .cv2_measure import ALL5, E2, H3, kettle_action_q, knn_acc, probe, split_groups
from .cv2_separability import FIVE, cov, eig_stats, gauss_classify
from .cv2_train import ClosureSampler
from .cv3_train import patch_contacts
from .kinematics import FINGERS


def wmat(Z, ridge=1e-6):
    e, V = np.linalg.eigh(cov(Z))
    e = np.maximum(e, 0) + ridge * max(e.mean(), 1e-12)
    return V @ np.diag(e ** -0.5) @ V.T


def pr_stats(Z):
    s = eig_stats(cov(Z))
    return {"pc1_share": s["eig_frac"][0], "participation_ratio": s["participation_ratio"], "n90": s["n90"], "n99": s["n99"]}


def wdist(Z, M, W, hands, seed=0):
    """Z (S,H,D), M (S,H) bool. Whitened same / rand-cross / within (mean over hand pairs of mean distances)."""
    S, H, _ = Z.shape
    Zw = Z @ W
    rng = np.random.default_rng(seed)
    rows = {}
    for i in range(H):
        for j in range(i + 1, H):
            v = np.where(M[:, i] & M[:, j])[0]
            if len(v) < 20:
                continue
            p = rng.permutation(v)
            p = np.where(p == v, np.roll(p, 1), p)
            same = np.linalg.norm(Zw[v, i] - Zw[v, j], axis=1).mean()
            rand = np.linalg.norm(Zw[v, i] - Zw[p, j], axis=1).mean()
            within = 0.5 * (np.linalg.norm(Zw[v, i] - Zw[p, i], axis=1).mean() + np.linalg.norm(Zw[v, j] - Zw[p, j], axis=1).mean())
            rows[f"{hands[i]}-{hands[j]}"] = {"same": float(same), "rand_cross": float(rand), "within": float(within),
                                              "same/rand": float(same / rand), "same/within": float(same / within), "n": int(len(v))}
    out = {"pairs": rows}
    for k in ("same/rand", "same/within"):
        out[k] = float(np.mean([r[k] for r in rows.values()])) if rows else None
    return out


def handid(Z, M, W, hands):
    S, H, D = Z.shape
    X = Z.reshape(S * H, D).astype(np.float64); y = np.tile(np.arange(H), S); g = np.repeat(np.arange(S), H)
    keep = M.reshape(-1)
    X, y, g = X[keep], y[keep], g[keep]
    tr, te = split_groups(g)
    R = {"n": int(len(y)), "chance": 1.0 / H, "probe_std_lr": probe(X, y, g, H, "raw", split=(tr, te))[0]["acc"],
         "qda": gauss_classify(X[tr], y[tr], X[te], y[te], H, "qda"),
         "knn15_raw": knn_acc(X, y, tr, te), "knn15_white": knn_acc(X @ W, y, tr, te)}
    if H == 5:
        five = [i for i, h in enumerate(hands) if h in FIVE]
        y45 = np.isin(y, five).astype(int)
        R["probe_4vs5"] = probe(X, y45, g, 2, "raw", split=(tr, te))[0]["acc"]
        m5 = np.isin(y, five)
        y5 = np.searchsorted(np.array(five), y[m5])
        R["probe_within5"] = probe(X[m5], y5, g[m5], 3, "raw", split=(tr[m5], te[m5]))[0]["acc"]
    return R


@torch.no_grad()
def retarget(codecs, fks, Q, M, hands):
    """Q {hand: (N,dof) normalized source postures}, M (N,H) bool validity. Pad residual (E2* metric) per ordered pair."""
    pads = {h: fks[h].pads(torch.tensor(Q[h])) for h in hands}
    z = {h: codecs[h].encode(torch.tensor(Q[h]))[0] for h in hands}
    pairs = {}
    for a, A in enumerate(hands):
        for b, B in enumerate(hands):
            if a == b:
                continue
            v = torch.tensor(M[:, a])
            if int(v.sum()) < 20:
                continue
            qd = codecs[B].decoder(z[A][v])
            pb = fks[B].pads(qd)
            pa = tuple(x[v] for x in pads[A])
            res = align_pair(fks[A], fks[B], pa, pb)
            e = res["err"].norm(dim=-1).numpy() * 1000; w = res["w"].numpy(); F = len(res["fingers"])
            pf = {FINGERS[fi]: float((e[:, j] * w[:, j]).sum() / w[:, j].sum()) for j, fi in enumerate(res["fingers"])}
            pf["thumb"] = float((e[:, F:2 * F] * w).sum() / w.sum())
            pairs[f"{A}->{B}"] = pf
    cat = {}
    for key, pf in pairs.items():
        A, B = key.split("->")
        c = ("5" if A in FIVE else "4") + ("<->" if (A in FIVE) == (B in FIVE) else "->") + ("5" if B in FIVE else "4")
        cat.setdefault(c, []).append(float(np.mean(list(pf.values()))))
    allf = {f: np.mean([p[f] for p in pairs.values() if f in p]) for f in FINGERS if any(f in p for p in pairs.values())}
    return {"by_category": {c: float(np.mean(v)) for c, v in sorted(cat.items())},
            "overall_G5style": float(np.mean(list(allf.values()))), "pairs": pairs}


def load_tax(path, part):
    d = np.load(path, allow_pickle=True)
    th = [str(h) for h in d["hands"]]
    m = d["split"] == part
    Q = {h: d[f"q_{h}"][m].astype(np.float32) for h in th}
    V = d["valid"][m][:, [th.index(h) for h in ALL5]].astype(bool)
    return Q, V, d["class_id"][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", action="append", required=True)
    ap.add_argument("--out", default="artifacts/codec_v3/measure")
    ap.add_argument("--sets", default="artifacts/codec_v2/measure/pairsets.npz")
    ap.add_argument("--cl-sets", default="artifacts/codec_v3/measure/clpairs.npz")
    ap.add_argument("--tax", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-kettle", action="store_true")
    ap.add_argument("--max-rows", type=int, default=3000, help="cap per paired set (speed)")
    args = ap.parse_args()
    patch_contacts()
    torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    play = {h: np.load(f"artifacts/play_v1/{h}.npz", allow_pickle=True) for h in ALL5}
    _, fks, p0 = load_arm(E2)
    sets = dict(np.load(args.sets, allow_pickle=True))
    if not Path(args.cl_sets).exists():  # v2 functional-closure pair draw (codec-independent, cached once)
        smp = ClosureSampler(play, fks, p0["config"]["hand_scales_m"], "cpu", mode="func")
        smp.rng = np.random.default_rng(2024)
        q = smp.sample(2000)
        np.savez_compressed(args.cl_sets, **{f"CL_{h}": q[h].numpy() for h in ALL5})
    cl = dict(np.load(args.cl_sets))
    PS = {}
    for kind in ("GEO", "SYN"):
        for tag, hands in (("ALL5", ALL5), ("H3", H3)):
            if kind == "SYN" and tag == "H3":
                continue
            Q = {h: sets[f"{kind}_{tag}_{h}"][:args.max_rows] for h in hands}
            PS[f"{kind}_{tag}"] = (hands, Q, np.ones((len(Q[hands[0]]), len(hands)), bool))
    PS["CL_ALL5"] = (ALL5, {h: cl[f"CL_{h}"] for h in ALL5}, np.ones((len(cl["CL_allegro"]), 5), bool))
    if args.tax:
        for part, nm in ((2, "TAXC"), (1, "TAXS")):
            Q, V, _ = load_tax(args.tax, part)
            if len(V) > args.max_rows:
                sel = np.random.default_rng(0).choice(len(V), args.max_rows, replace=False)
                Q = {h: q[sel] for h, q in Q.items()}; V = V[sel]
            PS[f"{nm}_ALL5"] = (ALL5, Q, V)
            hi = [ALL5.index(h) for h in H3]
            m3 = V[:, hi].all(1)
            PS[f"{nm}_H3"] = (H3, {h: Q[h][m3] for h in H3}, V[m3][:, hi])
    kq = None if args.no_kettle else {h: kettle_action_q(fks, h, range(10)) for h in ALL5}
    for spec in args.codec:
        name, path = spec.split("=", 1)
        codecs, cfks, payload = load_arm(path)

        @torch.no_grad()
        def enc(h, q):
            return codecs[h].encode(torch.as_tensor(np.asarray(q), dtype=torch.float32))[0].numpy().astype(np.float64)

        R = {"codec": path, "step": payload.get("step"), "cv3": payload["config"].get("cv3"), "cv2": payload["config"].get("cv2")}
        Zp = {h: enc(h, play[h]["val"][:4000]) for h in ALL5}
        pool = np.concatenate(list(Zp.values()))
        W = wmat(pool)
        R["collapse"] = {"pooled_playval": pr_stats(pool), "perhand_playval": {h: pr_stats(Zp[h]) for h in ALL5}}
        for sname, (hands, Q, M) in PS.items():
            Z = np.stack([enc(h, Q[h]) for h in hands], 1)
            blk = {"n_scenes": int(len(M)), "white": wdist(Z, M, W, hands), "handid": handid(Z, M, W, hands)}
            if sname == "GEO_ALL5":
                blk["perhand_pr"] = {h: pr_stats(Z[:, i])["participation_ratio"] for i, h in enumerate(hands)}
            R[sname] = blk
            print(name, sname, json.dumps({"w_same/rand": blk["white"]["same/rand"], "w_same/within": blk["white"]["same/within"],
                                           **{k: v for k, v in blk["handid"].items() if k != "n"}}), flush=True)
        # retarget
        rt = {"playval": retarget(codecs, cfks, {h: play[h]["val"][:2048] for h in ALL5}, np.ones((2048, 5), bool), ALL5),
              "cl": retarget(codecs, cfks, {h: cl[f"CL_{h}"] for h in ALL5}, np.ones((len(cl["CL_allegro"]), 5), bool), ALL5)}
        if args.tax:
            Qc, Vc, _ = load_tax(args.tax, 2); Qs, Vs, _ = load_tax(args.tax, 1)
            Qt = {h: np.concatenate([Qc[h], Qs[h]]) for h in ALL5}
            rt["tax_test"] = retarget(codecs, cfks, Qt, np.concatenate([Vc, Vs]), ALL5)
            rt["tax_heldout_class"] = retarget(codecs, cfks, Qc, Vc, ALL5)
        R["retarget"] = rt
        print(name, "retarget", json.dumps({k: v["by_category"] | {"all": v["overall_G5style"]} for k, v in rt.items()}), flush=True)
        if kq is not None:
            norm = lambda h, qf: cfks[h].normalize(torch.tensor(np.asarray(qf), dtype=torch.float32)).clamp(-1, 1)
            Zk = {h: [enc(h, norm(h, qf)) @ W for qf in kq[h]] for h in ALL5}
            P = torch.tensor(np.concatenate([z for h in H3 for z in Zk[h][:8]]))
            knnd = lambda Zq: torch.cdist(torch.tensor(Zq), P).topk(10, largest=False).values.mean(1).numpy()
            dref = knnd(np.concatenate([z for h in H3 for z in Zk[h][8:]]))
            k2 = {"ref_median": float(np.median(dref)), "ref_p95": float(np.percentile(dref, 95))}
            lo = []
            for h in H3:
                Po = torch.tensor(np.concatenate([z for o in H3 if o != h for z in Zk[o][:8]]))
                lo.append(float(np.median(torch.cdist(torch.tensor(np.concatenate(Zk[h])), Po).topk(10, largest=False).values.mean(1).numpy())))
            k2["loho_ref_median"] = float(np.mean(lo))
            for u in ("allegro", "leap"):
                du = knnd(np.concatenate(Zk[u]))
                k2[u] = {"knn10_median_ratio": float(np.median(du) / np.median(dref)),
                         "knn10_median_ratio_vs_loho": float(np.median(du) / k2["loho_ref_median"]),
                         "frac_inside_ref_p95": float((du <= k2["ref_p95"]).mean())}
            R["KETTLE_K2_white"] = k2
            print(name, "kettle_white", json.dumps(k2), flush=True)
        (out / f"{name}.json").write_text(json.dumps(R, indent=1))
        print("[cv3_measure] DONE", name, flush=True)


if __name__ == "__main__":
    main()
