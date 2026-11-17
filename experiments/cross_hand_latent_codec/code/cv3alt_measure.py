"""cv3alt (261116) measurement: v2-comparable gates + cause-1/cause-2 separability metrics + per-category retarget residual,
on (i) the v2 codec-independent pair sets (codec_v2/measure/pairsets.npz, identical scenes) and (ii) each cv3alt dataset's
held-out split. New file; probe/kNN/var/dist functions imported from cv2_measure, Gaussian/eig helpers from cv2_separability,
Kettle K1/K2 code copied from cv2_measure.main (unchanged logic). CPU.
  CUDA_VISIBLE_DEVICES= nice -n 10 python.sh -m xhand_latent.cv3alt_measure --codec NAME=PATH [--codec ...] --out artifacts/codec_v3alt/measure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .arm_pads import align_pair
from .arm_train import load_arm, objective, summarize_diag
from .cv2_measure import ALL5, H3, kettle_action_q, kettle_state_q, knn_acc, probe, split_groups, var_hand, dist_table, two_way
from .cv2_separability import cov, eig_stats, gauss_classify, inv_sqrt
from .kinematics import FINGERS

FIVE = ("sharpa", "shadow", "wuji")
V2SETS = "artifacts/codec_v2/measure/pairsets.npz"
DATASETS = {"playik": "artifacts/codec_v3alt/data/playik.npz", "sobol": "artifacts/codec_v3alt/data/sobol.npz",
            "human": "artifacts/codec_v3alt/data/human.npz"}


def whitened_ratio(Z, seed=0):
    """Z (S,H,D) paired. Pooled whitening, then mean same-scene cross-hand distance / same-hand different-scene distance."""
    S, H, D = Z.shape
    X = Z.reshape(-1, D).astype(np.float64)
    W = inv_sqrt(cov(X), 1e-6)
    Zw = ((Z.reshape(-1, D) - X.mean(0)) @ W).reshape(S, H, D)
    perm = np.random.default_rng(seed).permutation(S)
    perm = np.where(perm == np.arange(S), (perm + 1) % S, perm)
    same, within = [], []
    for i in range(H):
        for j in range(i + 1, H):
            same.append(np.linalg.norm(Zw[:, i] - Zw[:, j], axis=1).mean())
            within.append(0.5 * (np.linalg.norm(Zw[:, i] - Zw[perm, i], axis=1).mean()
                                 + np.linalg.norm(Zw[:, j] - Zw[perm, j], axis=1).mean()))
    return float(np.mean(same) / np.mean(within))


def v3_ratios(Z, Wp, mp, seed=0):
    """v3-DESIGN definition: ZCA from the codec's own pooled play-val z; same-scene cross-hand distance divided by (a) random
    cross-hand distance (same/rand-cross) and (b) same-hand different-scene distance (same/within)."""
    S, H, D = Z.shape
    Zw = ((Z.reshape(-1, D) - mp) @ Wp).reshape(S, H, D)
    perm = np.random.default_rng(seed).permutation(S)
    perm = np.where(perm == np.arange(S), (perm + 1) % S, perm)
    same, rand, within = [], [], []
    for i in range(H):
        for j in range(i + 1, H):
            same.append(np.linalg.norm(Zw[:, i] - Zw[:, j], axis=1).mean())
            rand.append(np.linalg.norm(Zw[:, i] - Zw[perm, j], axis=1).mean())
            within.append(0.5 * (np.linalg.norm(Zw[:, i] - Zw[perm, i], axis=1).mean()
                                 + np.linalg.norm(Zw[:, j] - Zw[perm, j], axis=1).mean()))
    return {"same/rand": float(np.mean(same) / np.mean(rand)), "same/within": float(np.mean(same) / np.mean(within))}


def alt_block(Z, hands, name, Wp=None, mp=None):
    S, H, D = Z.shape
    X = Z.reshape(S * H, D).astype(np.float64); y = np.tile(np.arange(H), S); g = np.repeat(np.arange(S), H)
    tr, te = split_groups(g)
    R = {"n_scenes": int(S), "hands": list(hands)}
    R["probe_raw"] = probe(X, y, g, H, "raw", split=(tr, te))[0]["acc"]
    R["R_whitened"] = whitened_ratio(Z)
    R["R_euclid_same/within"] = dist_table(Z, hands)["MEAN"]["same/within"]
    if Wp is not None:
        R["v3white"] = v3_ratios(Z, Wp, mp)
    R["frac_hand"] = two_way(Z)["frac_hand"]
    R["gauss"] = {m: gauss_classify(X[tr], y[tr], X[te], y[te], H, m) for m in ("nc", "qda", "qda0")}
    Xw = X.copy()
    for k in range(H):
        mk = X[tr & (y == k)].mean(0); Wk = inv_sqrt(cov(X[tr & (y == k)]), 1e-3)
        Xw[y == k] = (X[y == k] - mk) @ Wk
    R["knn15_raw"] = knn_acc(X, y, tr, te)
    R["knn15_perhand_whitened"] = knn_acc(Xw, y, tr, te)
    e = np.clip(np.linalg.eigvalsh(cov(X))[::-1], 0, None)
    R["pc1_share"] = float(e[0] / e.sum())
    R["perhand"] = {h: {k: v for k, v in eig_stats(cov(X[y == i])).items() if k != "eig_frac"} for i, h in enumerate(hands)}
    if H == 5:
        y45 = np.isin(np.array(hands)[y], FIVE).astype(int)
        R["probe_4vs5"] = probe(X, y45, g, 2, "raw", split=(tr, te))[0]["acc"]
    print(name, json.dumps({"lin": round(R["probe_raw"], 3), "Rw": round(R["R_whitened"], 3), "v3w": R.get("v3white"), "Reu": round(R["R_euclid_same/within"], 3),
                            "qda0": R["gauss"]["qda0"], "wknn": R["knn15_perhand_whitened"], "pc1": round(R["pc1_share"], 3),
                            "PR": {h: round(v["participation_ratio"], 1) for h, v in R["perhand"].items()}}), flush=True)
    return R


def retarget_pairs(codecs, fks, Q, n=2048):
    """Per ordered pair A->B: Dec_B(Enc_A(q_A)) pad-contact residual (E2* metric, pair-weighted per finger, mm) on source
    postures Q[A] (normalized). Returns per-pair dict + per-category summary."""
    pairs = {}
    with torch.no_grad():
        for a in Q:
            qa = torch.as_tensor(Q[a][:n], dtype=torch.float32)
            za = codecs[a].encode(qa)[0]
            pa = fks[a].pads(qa)
            for b in Q:
                if b == a:
                    continue
                res = align_pair(fks[a], fks[b], pa, fks[b].pads(codecs[b].decoder(za)))
                e = res["err"].norm(dim=-1).numpy() * 1000; w = res["w"].numpy(); F = len(res["fingers"])
                pf = {FINGERS[fi]: float((e[:, j] * w[:, j]).sum() / w[:, j].sum()) for j, fi in enumerate(res["fingers"])}
                pf["thumb"] = float((e[:, F:2 * F] * w).sum() / w.sum())
                pairs[f"{a}->{b}"] = pf
    cat = {}
    for key, pf in pairs.items():
        a, b = key.split("->")
        c = ("5" if a in FIVE else "4") + ("<->" if (a in FIVE) == (b in FIVE) else "->") + ("5" if b in FIVE else "4")
        cat.setdefault(c, []).append(float(np.mean(list(pf.values()))))
    allf = {f: np.mean([p[f] for p in pairs.values() if f in p]) for f in FINGERS}
    return {"by_category": {c: float(np.mean(v)) for c, v in sorted(cat.items())}, "overall_G5style": float(np.mean(list(allf.values()))),
            "pairs": pairs}


def kettle(codecs, cfks, kq, enc):
    """Verbatim logic of cv2_measure.main's Kettle K1/K2 block."""
    norm = lambda h, qf: cfks[h].normalize(torch.tensor(np.asarray(qf), dtype=torch.float32)).clamp(-1, 1)
    rng = np.random.default_rng(0)
    KZ, KY, KE = [], [], []
    for k, h in enumerate(H3):
        for e, qf in enumerate(kq["state"][h]):
            z = enc(h, norm(h, qf))[::4]
            KZ.append(z); KY.append(np.full(len(z), k)); KE.append(np.full(len(z), e))
    KZ, KY, KE = np.concatenate(KZ), np.concatenate(KY), np.concatenate(KE)
    bal = np.zeros(len(KY), bool)
    for part in (KE < 40, KE >= 40):
        m = min(((KY == k) & part).sum() for k in range(3))
        for k in range(3):
            ii = np.where((KY == k) & part)[0]
            bal[rng.choice(ii, m, replace=False)] = True
    KZ, KY, KE = KZ[bal], KY[bal], KE[bal]
    k1 = {"n": int(len(KY)), "probe_raw": probe(KZ, KY, KE, 3, split=(KE < 40, KE >= 40))[0]["acc"], "var": var_hand(KZ, KY)}
    Zk = {h: [enc(h, norm(h, qf)) for qf in kq["action"][h]] for h in ALL5}
    pool = np.concatenate([z for h in H3 for z in Zk[h][:8]])
    refz = np.concatenate([z for h in H3 for z in Zk[h][8:]])
    P = torch.tensor(pool)

    def knnd(Zq, k=10):
        return torch.cdist(torch.tensor(Zq), P).topk(k, largest=False).values.mean(1).numpy()
    dref = knnd(refz)
    k2 = {"ref_median": float(np.median(dref)), "ref_p95": float(np.percentile(dref, 95))}
    lo = []
    for h in H3:
        Po = torch.tensor(np.concatenate([z for o in H3 if o != h for z in Zk[o][:8]]))
        lo.append(np.median(torch.cdist(torch.tensor(np.concatenate(Zk[h])), Po).topk(10, largest=False).values.mean(1).numpy()))
    k2["loho_ref_median"] = float(np.mean(lo))
    for u in ("allegro", "leap"):
        zu = np.concatenate(Zk[u]); du = knnd(zu)
        k2[u] = {"knn10_median_ratio": float(np.median(du) / np.median(dref)),
                 "knn10_median_ratio_vs_loho": float(np.median(du) / k2["loho_ref_median"]),
                 "frac_inside_ref_p95": float((du <= k2["ref_p95"]).mean())}
    # whitened K2 (cause-1-proof): same kNN after pooled whitening of all Kettle z
    allz = np.concatenate([z for h in ALL5 for z in Zk[h]]).astype(np.float64)
    Wt = inv_sqrt(cov(allz), 1e-6); mu = allz.mean(0)
    Pw = torch.tensor((pool - mu) @ Wt)

    def knnw(Zq, k=10):
        return torch.cdist(torch.tensor((Zq - mu) @ Wt), Pw).topk(k, largest=False).values.mean(1).numpy()
    drw = knnw(refz)
    for u in ("allegro", "leap"):
        du = knnw(np.concatenate(Zk[u]))
        k2[u]["whitened_knn10_median_ratio"] = float(np.median(du) / np.median(drw))
        k2[u]["whitened_frac_inside_ref_p95"] = float((du <= np.percentile(drw, 95)).mean())
    return {"K1": k1, "K2": k2}


def held_out_sets(name, path):
    d = np.load(path, allow_pickle=True)
    order = [str(s) for s in d["hands"]]
    M = d["pmask_val"].astype(bool)
    out = {}
    for tag, hands in (("ALL5", ALL5), ("H3", H3)):
        keep = M[:, [order.index(h) for h in hands]].all(1)
        if keep.sum() < 50:
            continue
        idx = np.where(keep)[0][:3000]
        out[f"{name}_{tag}"] = {h: d[f"Q_{h}_val"][idx] for h in hands}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", action="append", required=True)
    ap.add_argument("--out", default="artifacts/codec_v3alt/measure")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-kettle", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    play = {h: np.load(f"artifacts/play_v1/{h}.npz", allow_pickle=True) for h in ALL5}
    v2 = dict(np.load(V2SETS, allow_pickle=True))
    sets = {f"v2{kind}_{tag}": {h: v2[f"{kind}_{tag}_{h}"] for h in (ALL5 if tag == "ALL5" else H3)}
            for kind in ("GEO", "SYN") for tag in ("ALL5", "H3")}
    for n, p in DATASETS.items():
        if Path(p).exists():
            sets.update(held_out_sets(n, p))
    _, fks0, _ = load_arm("artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt")
    kq = None
    if not args.no_kettle:
        kq = {"state": {h: kettle_state_q(fks0, h, range(50)) for h in H3},
              "action": {h: kettle_action_q(fks0, h, range(10)) for h in ALL5}}
    for spec in args.codec:
        name, path = spec.split("=", 1)
        codecs, cfks, payload = load_arm(path)

        @torch.no_grad()
        def enc(h, q):
            return codecs[h].encode(torch.as_tensor(np.asarray(q), dtype=torch.float32))[0].numpy()
        R = {"codec": path, "step": payload.get("step"), "cv3alt": payload["config"].get("cv3alt"), "cv2": payload["config"].get("cv2")}
        Zpv = np.concatenate([enc(h, play[h]["val"][:4000]) for h in ALL5]).astype(np.float64)
        mp = Zpv.mean(0); Wp = inv_sqrt(cov(Zpv), 1e-3)
        ev = np.clip(np.linalg.eigvalsh(cov(Zpv))[::-1], 0, None)
        R["collapse_playval"] = {"pc1_share": float(ev[0] / ev.sum()), "participation_ratio": float(ev.sum() ** 2 / np.square(ev).sum())}
        for sname, Q in sets.items():
            hands = list(Q)
            Z = np.stack([enc(h, Q[h]) for h in hands], 1)
            R[sname] = alt_block(Z, hands, f"{name} {sname}", Wp, mp)
        with torch.no_grad():
            vb = {h: torch.tensor(play[h]["val"][:2048]) for h in cfks}
            _, _, values, stats, diag = objective(vb, codecs, cfks, payload["config"]["beta"], payload["config"]["lam"], "e2",
                                                  sample=False, diagnostics=True)
        q = summarize_diag(diag, cfks)
        q["joint_rmse_rad"] = {h: float((cfks[h].denormalize(a) - cfks[h].denormalize(b)).square().mean().sqrt())
                               for h, (a, b) in stats.items()}
        q["pad_residual_mm_mean"] = float(np.mean(list(q["per_finger_contact_residual_mm_pairweighted"].values())))
        R["quality"] = q
        R["retarget_playval"] = retarget_pairs(codecs, cfks, {h: play[h]["val"][:2048] for h in ALL5})
        R["retarget_v2GEO"] = retarget_pairs(codecs, cfks, {h: v2[f"GEO_ALL5_{h}"] for h in ALL5})
        for sname, Q in sets.items():
            if sname.endswith("_ALL5") and not sname.startswith("v2"):
                R[f"retarget_{sname}"] = retarget_pairs(codecs, cfks, Q)
        print(name, "quality", json.dumps({"pad": round(q["pad_residual_mm_mean"], 3), "rmse": q["joint_rmse_rad"],
                                           "cat": R["retarget_playval"]["by_category"]}), flush=True)
        if kq is not None:
            R["KETTLE"] = kettle(codecs, cfks, kq, enc)
            print(name, "kettle", json.dumps({"K1": R["KETTLE"]["K1"]["probe_raw"], "eta2": R["KETTLE"]["K1"]["var"]["eta2_hand"],
                                              "K2": {u: R["KETTLE"]["K2"][u] for u in ("allegro", "leap")}}), flush=True)
        (out / f"{name}.json").write_text(json.dumps(R, indent=1))
        print("[cv3alt_measure] DONE", name, flush=True)


if __name__ == "__main__":
    main()
