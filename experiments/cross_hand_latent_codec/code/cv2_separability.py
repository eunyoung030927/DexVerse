"""codec v2 follow-up (261116): WHY is the hand still linearly readable from z, the little finger (pinky) question, per-pair
retarget residual, and a small visualization JSON. New file, CPU only, offline (no training).
Container python, cwd /workspace/xhand/latent_codec_v1:
  CUDA_VISIBLE_DEVICES= nice -n 10 python.sh -m xhand_latent.cv2_separability --out artifacts/codec_v2/viz
Reuses cv2_measure (probe = standardized multinomial LR, grouped scene split; pair sets measure/pairsets.npz).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .arm_train import load_arm, objective
from .cv2_measure import ALL5, E2, H3, fit_lr, kettle_action_q, knn_acc, probe, split_groups, var_hand, dist_table, two_way
from .kinematics import FINGERS

CODECS = {"E2*": E2, "cyc05p05": "artifacts/codec_v2/runs/cyc05p05/best.pt",
          "cyc05clf05f002": "artifacts/codec_v2/runs/cyc05clf05f002/best.pt"}
MEASURE = {"E2*": "E2", "cyc05p05": "cyc05p05", "cyc05clf05f002": "cyc05clf05f002"}
FIVE = ("sharpa", "shadow", "wuji")
FOUR = ("allegro", "leap")
L2S = (1e-4, 1e-3, 1e-2, 1e-1)


def r3(a):
    return [round(float(x), 3) for x in a]


# ---------------- linear algebra helpers ----------------
def cov(X):
    Xc = X - X.mean(0)
    return Xc.T @ Xc / len(X)


def inv_sqrt(S, ridge=1e-6):
    e, V = np.linalg.eigh(S)
    e = np.maximum(e, 0) + ridge * max(e.mean(), 1e-12)
    return V @ np.diag(e ** -0.5) @ V.T


def scatter(X, y, K):
    mu = X.mean(0)
    D = X.shape[1]
    Sw = np.zeros((D, D)); Sb = np.zeros((D, D))
    for k in range(K):
        Xk = X[y == k]; mk = Xk.mean(0)
        Sw += (Xk - mk).T @ (Xk - mk)
        Sb += len(Xk) * np.outer(mk - mu, mk - mu)
    return Sw / len(X), Sb / len(X)


def lda_dirs(X, y, K, ridge=1e-6):
    Sw, Sb = scatter(X, y, K)
    W = inv_sqrt(Sw, ridge)
    e, U = np.linalg.eigh(W @ Sb @ W)
    order = np.argsort(e)[::-1][:K - 1]
    dirs = (W @ U[:, order]).T
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True), e[order], Sw, Sb


def probe_acc(X, y, g, K, split, l2s=L2S):
    res, (W, b, mu, sd) = probe(X, y, g, K, "raw", l2s=l2s, split=split)
    tr, te = split
    pred = (((X[te] - mu) / sd) @ W + b).argmax(1)
    conf = np.zeros((K, K), int)
    for a, p in zip(y[te], pred):
        conf[a, p] += 1
    return float(res["acc"]), conf


def gauss_classify(Xtr, ytr, Xte, yte, K, mode, ridge=1e-4):
    """mode: 'nc' Euclidean nearest centroid | 'lda' shared cov | 'qda' per-hand mean+cov | 'qda0' per-hand cov, means removed."""
    D = Xtr.shape[1]
    mus = [Xtr[ytr == k].mean(0) for k in range(K)]
    if mode == "qda0":
        Xte = Xte.copy()
        for k in range(K):  # test points centred with their own hand's TRAIN mean (= covariance-shape-only test)
            Xte[yte == k] -= mus[k]
        Xtr = Xtr.copy()
        for k in range(K):
            Xtr[ytr == k] -= mus[k]
        mus = [np.zeros(D)] * K
    covs = [cov(Xtr[ytr == k]) for k in range(K)]
    pooled = sum(covs) / K
    scores = []
    for k in range(K):
        if mode == "nc":
            s = -np.square(Xte - mus[k]).sum(1)
        else:
            S = pooled if mode == "lda" else covs[k]
            S = S + ridge * np.trace(S) / D * np.eye(D)
            Si = np.linalg.inv(S)
            d = Xte - mus[k]
            s = -0.5 * np.einsum("ni,ij,nj->n", d, Si, d) - (0.5 * np.linalg.slogdet(S)[1] if mode != "lda" else 0)
        scores.append(s)
    return float((np.argmax(np.stack(scores, 1), 1) == yte).mean())


def eig_stats(S, dof=None):
    e = np.clip(np.linalg.eigvalsh(S)[::-1], 0, None)
    c = np.cumsum(e) / e.sum()
    out = {"total_var": float(e.sum()), "participation_ratio": float(e.sum() ** 2 / np.square(e).sum()),
           "n90": int(np.searchsorted(c, 0.90) + 1), "n99": int(np.searchsorted(c, 0.99) + 1),
           "n999": int(np.searchsorted(c, 0.999) + 1), "eig_frac": r3(e / e.sum())}
    if dof is not None and dof < len(e):
        out["frac_var_beyond_dof"] = float(e[dof:].sum() / e.sum())
    return out


def subspace_overlap(Sa, Sb_, frac=0.9):
    def top(S):
        e, V = np.linalg.eigh(S); e, V = e[::-1], V[:, ::-1]
        k = int(np.searchsorted(np.cumsum(e) / e.sum(), frac) + 1)
        return V[:, :k]
    A, B = top(Sa), top(Sb_)
    return float(np.square(A.T @ B).sum() / min(A.shape[1], B.shape[1])), A.shape[1], B.shape[1]


# ---------------- separability block ----------------
def sep_block(Z, LV, hands, subspaces, name):
    S, H, D = Z.shape
    X = Z.reshape(S * H, D).astype(np.float64); y = np.tile(np.arange(H), S); g = np.repeat(np.arange(S), H)
    split = split_groups(g); tr, te = split
    R = {"n_scenes": S, "hands": list(hands)}
    acc, conf = probe_acc(X, y, g, H, split)
    R["probe_raw"], R["confusion_raw"] = acc, conf.tolist()
    if H == 5:  # 4-vs-5 grouping inside the 5-way confusion
        five = [i for i, h in enumerate(hands) if h in FIVE]
        four = [i for i, h in enumerate(hands) if h in FOUR]
        R["frac_errors_within_5finger"] = float(sum(conf[a, b] for a in five for b in five if a != b) / max(1, conf.sum() - np.trace(conf)))
        R["frac_errors_within_4finger"] = float(sum(conf[a, b] for a in four for b in four if a != b) / max(1, conf.sum() - np.trace(conf)))
        y45 = np.isin(y, five).astype(int)
        R["probe_4vs5_binary"] = probe_acc(X, y45, g, 2, split)[0]
    # PCA (train split)
    m = X[tr].mean(0)
    e, U = np.linalg.eigh(cov(X[tr])); e, U = e[::-1], U[:, ::-1]
    Xp = (X - m) @ U
    R["pca_eig_frac"] = r3(e / e.sum())
    R["probe_global_whitened"] = probe_acc(Xp / np.sqrt(e + 1e-12 * e.max()), y, g, H, split)[0]
    R["probe_topk"] = {k: probe_acc(Xp[:, :k], y, g, H, split)[0] for k in (1, 2, 4, 8, 12, 16, 24)}
    R["probe_without_topk"] = {k: probe_acc(Xp[:, k:], y, g, H, split)[0] for k in (2, 4, 8, 12, 16, 24)}
    # per-PC hand share and Fisher ratio
    eta, fis = [], []
    for i in range(D):
        v = Xp[:, i]; mu = v.mean()
        sb = sum((y == k).sum() * (v[y == k].mean() - mu) ** 2 for k in range(H)); st = np.square(v - mu).sum()
        eta.append(sb / st); fis.append(sb / max(st - sb, 1e-12))
    R["pc_eta2_hand"], R["pc_fisher"] = r3(eta), r3(fis)
    R["pc_argmax_fisher"] = int(np.argmax(fis))
    # LDA directions (train split), evaluated on test split
    dirs, fr, Sw, Sb = lda_dirs(X[tr], y[tr], H)
    Swte, Sbte = scatter(X[te], y[te], H)
    Ste = Swte + Sbte
    St = Sw + Sb
    lda = []
    for w, f in zip(dirs, fr):
        energy = np.square(U.T @ w)  # energy over pooled PCs (train)
        lda.append({"fisher_train": float(f), "fisher_test": float(w @ Sbte @ w / (w @ Swte @ w)),
                    "eta2_test": float(w @ Sbte @ w / (w @ Ste @ w)),
                    "within_var_rel_mean_dir": float(w @ Sw @ w / (np.trace(Sw) / D)),
                    "total_var_rel_mean_dir": float(w @ St @ w / (np.trace(St) / D)),
                    "pcs_with_more_variance": int((e > w @ St @ w).sum()),
                    "energy_top8_pcs": float(energy[:8].sum()), "energy_pcs_17_32": float(energy[16:].sum()),
                    "per_hand_within_var_rel": {h: float(w @ cov(X[tr][y[tr] == k]) @ w / (np.trace(cov(X[tr][y[tr] == k])) / D))
                                                for k, h in enumerate(hands)}})
    R["lda"] = lda
    R["hand_share_total_var"] = var_hand(X, y)["eta2_hand"]
    R["two_way"] = two_way(Z)
    R["same/within"] = dist_table(Z, hands)["MEAN"]["same/within"]
    # Gaussian classifiers: Euclidean means vs Mahalanobis means vs covariance shape
    R["gauss"] = {mode: gauss_classify(X[tr], y[tr], X[te], y[te], H, mode) for mode in ("nc", "lda", "qda", "qda0")}
    # per-hand mean removal / per-hand whitening (train stats) -> nonlinear probes
    Xc = X.copy(); Xw = X.copy()
    for k in range(H):
        mk = X[tr & (y == k)].mean(0); Wk = inv_sqrt(cov(X[tr & (y == k)]), 1e-3)
        Xc[y == k] -= mk; Xw[y == k] = (X[y == k] - mk) @ Wk
    sq = lambda A: np.concatenate([A, A ** 2], 1)
    R["centered"] = {"knn15": knn_acc(Xc, y, tr, te), "diagquad_lr": probe_acc(sq(Xc), y, g, H, split)[0]}
    R["perhand_whitened"] = {"knn15": knn_acc(Xw, y, tr, te), "diagquad_lr": probe_acc(sq(Xw), y, g, H, split)[0]}
    # covariance shape / support
    covs = {h: cov(X[y == k]) for k, h in enumerate(hands)}
    R["perhand_cov"] = {h: eig_stats(covs[h]) for h in hands}
    R["subspace_overlap90"] = {f"{a}-{b}": subspace_overlap(covs[a], covs[b])[0] for i, a in enumerate(hands) for b in hands[i + 1:]}
    maha = {}
    for a in hands:
        Si = np.linalg.inv(covs[a] + 1e-4 * np.trace(covs[a]) / D * np.eye(D)); mu = X[y == hands.index(a)].mean(0)
        maha[a] = {b: float(np.median(np.einsum("ni,ij,nj->n", X[y == j] - mu, Si, X[y == j] - mu)) / D) for j, b in enumerate(hands)}
    R["median_mahalanobis2_over_D"] = maha  # row a = Gaussian of hand a, col b = points of hand b (diag ~ 1)
    # encoder posterior variance as a signature
    if LV is not None:
        L = LV.reshape(S * H, D).astype(np.float64)
        R["logvar"] = {"probe": probe_acc(L, y, g, H, split)[0],
                       "mean_logvar_per_hand": {h: float(L[y == k].mean()) for k, h in enumerate(hands)},
                       "mean_std_per_hand": {h: float(np.exp(0.5 * L[y == k]).mean()) for k, h in enumerate(hands)}}
    # subspace removal (pinky / ring control)
    sub = {}
    for sname, P in subspaces.items():
        Q = np.eye(D) - P @ P.T
        sub[sname] = {"k": int(P.shape[1]), "probe_without": probe_acc(X @ Q, y, g, H, split)[0],
                      "probe_only": probe_acc(X @ P, y, g, H, split)[0],
                      "lda1_energy_in": float(np.square(P.T @ dirs[0]).sum()),
                      "perhand_var_frac_in": {h: float(np.trace(P.T @ covs[h] @ P) / np.trace(covs[h])) for h in hands}}
        if H == 5:
            sub[sname]["probe_4vs5_without"] = probe_acc(X @ Q, np.isin(y, [hands.index(h) for h in FIVE]).astype(int), g, 2, split)[0]
    R["subspace_removal"] = sub
    print(name, json.dumps({"lin": R["probe_raw"], "white": R["probe_global_whitened"], "top8": R["probe_topk"][8],
                            "w/o top8": R["probe_without_topk"][8], "gauss": R["gauss"], "c_knn": R["centered"]["knn15"],
                            "w_knn": R["perhand_whitened"]["knn15"], "lda_within_rel": lda[0]["within_var_rel_mean_dir"],
                            "sub": {k: (v["probe_without"], v["probe_only"]) for k, v in sub.items()}}), flush=True)
    return R


# ---------------- pinky ----------------
def finger_cols(fk):
    return {f: [fk.qidx[n] for n in fk.spec["finger"][f]["chain_joints"]] for f in FINGERS if fk.spec["finger"][f]}


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1e-3):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    A = Xtr - mx
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Ytr - my))
    P = (Xte - mx) @ W + my
    return float(1 - np.square(P - Yte).sum() / np.square(Yte - Yte.mean(0)).sum())


def closure(fk, q, q_open, fi, scale):
    with torch.no_grad():
        t = fk(torch.as_tensor(q, dtype=torch.float32))[:, fi].numpy()
        t0 = fk(torch.as_tensor(q_open[None], dtype=torch.float32))[:, fi].numpy()
    return np.linalg.norm(t - t0, axis=1) / scale


def dstats(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "p05": float(np.percentile(v, 5)),
            "p50": float(np.median(v)), "p95": float(np.percentile(v, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/codec_v2/viz")
    ap.add_argument("--sets", default="artifacts/codec_v2/measure/pairsets.npz")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-viz", type=int, default=250)
    ap.add_argument("--n-kettle", type=int, default=150)
    ap.add_argument("--no-tsne", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    play = {h: np.load(f"artifacts/play_v1/{h}.npz", allow_pickle=True) for h in ALL5}
    sets = dict(np.load(args.sets, allow_pickle=True))
    _, fks0, _ = load_arm(E2)
    kq = {h: kettle_action_q(fks0, h, range(10)) for h in ALL5}
    rng = np.random.default_rng(0)
    nS = len(sets["GEO_ALL5_sharpa"])
    viz_s = np.sort(rng.choice(nS, args.n_viz, replace=False))
    viz_pairs = []
    for _ in range(40):
        s = int(rng.integers(args.n_viz)); a, b = sorted(rng.choice(5, 2, replace=False).tolist())
        viz_pairs.append([a * args.n_viz + s, b * args.n_viz + s])
    ket_sel = {h: np.sort(rng.choice(sum(len(x) for x in kq[h]), args.n_kettle, replace=False)) for h in ALL5}
    ALLRES, VIZ = {}, {}
    for cname, path in CODECS.items():
        codecs, fks, payload = load_arm(path)
        scales = payload["config"]["hand_scales_m"]

        @torch.no_grad()
        def enc(h, q):
            m, lv = codecs[h].encode(torch.as_tensor(np.asarray(q), dtype=torch.float32))
            return m.numpy().astype(np.float64), lv.numpy().astype(np.float64)

        @torch.no_grad()
        def dec(h, z):
            return codecs[h].decoder(torch.as_tensor(np.asarray(z), dtype=torch.float32)).numpy()

        norm = lambda h, q: fks[h].normalize(torch.tensor(np.asarray(q), dtype=torch.float32)).clamp(-1, 1).numpy()
        RES = {"codec": path, "step": payload.get("step")}
        # ---- pinky / finger subspaces (from play val, 5-finger hands) ----
        val = {h: play[h]["val"].astype(np.float32) for h in ALL5}
        zval = {h: enc(h, val[h])[0] for h in ALL5}
        sens, dz_store = {}, {}
        for h in ALL5:
            cols = finger_cols(fks[h]); q = val[h][:2000]; z = zval[h][:2000]
            perm = np.random.default_rng(1).permutation(len(val[h]))[:2000]
            base = float(np.linalg.norm(z - zval[h][perm], axis=1).mean())
            sens[h] = {"random_pair_dist": base}
            for f, c in cols.items():
                q2 = q.copy(); q2[:, c] = val[h][perm][:, c]
                dz = enc(h, q2)[0] - z
                dz_store[(h, f)] = dz
                sens[h][f] = float(np.linalg.norm(dz, axis=1).mean() / base)
        RES["finger_swap_dz_rel"] = sens

        def finger_subspace(f, frac=0.9, k=None):
            M = np.concatenate([dz_store[(h, f)] / np.linalg.norm(dz_store[(h, f)], axis=1).mean() for h in FIVE])
            _, s, Vt = np.linalg.svd(M, full_matrices=False)
            c = np.cumsum(s ** 2) / np.sum(s ** 2)
            k = int(np.searchsorted(c, frac) + 1) if k is None else k
            return Vt[:k].T, k, r3(c[:12])
        Pl, kl, cl = finger_subspace("little")
        Pr, _, cr = finger_subspace("ring", k=kl)
        Pi, _, ci = finger_subspace("index", k=kl)
        RES["finger_subspace"] = {"k_little90": kl, "little_cum_energy": cl, "ring_cum_energy": cr, "index_cum_energy": ci,
                                  "overlap_little_ring": float(np.square(Pl.T @ Pr).sum() / kl),
                                  "overlap_little_index": float(np.square(Pl.T @ Pi).sum() / kl)}
        subspaces = {"little": Pl, "ring": Pr, "index": Pi}
        # ---- Q1 separability on GEO sets + play val ----
        for tag, hands in (("GEO_ALL5", ALL5), ("GEO_H3", H3)):
            ZL = [enc(h, sets[f"{tag}_{h}"]) for h in hands]
            Z = np.stack([a for a, _ in ZL], 1); LV = np.stack([b for _, b in ZL], 1)
            RES[tag] = sep_block(Z, LV, hands, subspaces, f"{cname} {tag}")
        RES["PLAY_perhand_cov"] = {h: eig_stats(cov(zval[h]), fks[h].dof) for h in ALL5}
        RES["PLAY_logvar_mean"] = {h: float(enc(h, val[h][:2000])[1].mean()) for h in ALL5}
        # ---- Kettle binary H3 vs 4-finger (episode split), with subspace removal ----
        ZK = {h: [enc(h, norm(h, q))[0] for q in kq[h]] for h in ALL5}
        kb = {}
        for u in FOUR:
            Xb, yb, eb = [], [], []
            for lab, hs in ((0, H3), (1, (u,))):
                for h in hs:
                    for e_, z in enumerate(ZK[h]):
                        Xb.append(z); yb.append(np.full(len(z), lab)); eb.append(np.full(len(z), e_))
            Xb, yb, eb = np.concatenate(Xb), np.concatenate(yb), np.concatenate(eb)
            bal = np.zeros(len(yb), bool); r_ = np.random.default_rng(0)
            for part in (eb < 8, eb >= 8):
                mm = min(((yb == c) & part).sum() for c in range(2))
                for c in range(2):
                    bal[r_.choice(np.where((yb == c) & part)[0], mm, replace=False)] = True
            Xb, yb, eb = Xb[bal], yb[bal], eb[bal]
            sp = (eb < 8, eb >= 8)
            kb[u] = {"raw": probe_acc(Xb, yb, eb, 2, sp)[0],
                     "nearest_centroid": gauss_classify(Xb[sp[0]], yb[sp[0]], Xb[sp[1]], yb[sp[1]], 2, "nc"),
                     **{f"without_{s}": probe_acc(Xb @ (np.eye(32) - P @ P.T), yb, eb, 2, sp)[0] for s, P in subspaces.items()},
                     "only_little": probe_acc(Xb @ Pl, yb, eb, 2, sp)[0]}
        RES["KETTLE_binary_H3_vs_4f"] = kb
        print(cname, "kettle binary", json.dumps(kb), flush=True)
        # ---- Q2 per-finger reconstruction (play val[:4000]) ----
        rec = {}
        for h in ALL5:
            q = val[h][:4000]; qh = dec(h, zval[h][:4000]); cols = finger_cols(fks[h])
            with torch.no_grad():
                T = fks[h](torch.tensor(q)).numpy(); Th = fks[h](torch.tensor(qh)).numpy()
            qd = fks[h].denormalize(torch.tensor(q)).numpy(); qhd = fks[h].denormalize(torch.tensor(qh)).numpy()
            rec[h] = {f: {"joint_rmse_rad": float(np.sqrt(np.square(qd[:, cols[f]] - qhd[:, cols[f]]).mean())),
                          "tip_err_mm_mean": float(np.linalg.norm(T[:, fi] - Th[:, fi], axis=1).mean() * 1000),
                          "r2_norm_joints": float(1 - np.square(q[:, cols[f]] - qh[:, cols[f]]).sum() / np.square(q[:, cols[f]] - q[:, cols[f]].mean(0)).sum())}
                      for fi, f in enumerate(FINGERS) if f in cols}
        RES["perfinger_recon_playval"] = rec
        # ---- Q2 linear decodability of each finger from z (ridge, train 20k -> val 4000) ----
        lin = {}
        for h in FIVE:
            cols = finger_cols(fks[h]); tri = np.random.default_rng(2).choice(len(play[h]["train"]), 20000, replace=False)
            qtr = play[h]["train"][tri].astype(np.float32); ztr = enc(h, qtr)[0]; qte = val[h][:4000]; zte = zval[h][:4000]
            lin[h] = {}
            for f, c in cols.items():
                oth = [i for i in range(fks[h].dof) if i not in c]
                lin[h][f] = {"r2_from_z": ridge_r2(ztr, qtr[:, c], zte, qte[:, c]),
                             "r2_from_other_joints": ridge_r2(qtr[:, oth], qtr[:, c], qte[:, oth], qte[:, c])}
        RES["finger_linear_r2"] = lin
        # ---- Q2 Dec_5(z_4): pinky (little) state decoded from 4-finger z ----
        dec5 = {}
        for B in FIVE:
            fk = fks[B]; cols = finger_cols(fk)
            pn = [str(s) for s in play[B]["prototype_names"]]; qo = play[B]["prototypes"][pn.index("open")]
            lc = lambda q: closure(fk, q, qo, 4, scales[B]); rc = lambda q: closure(fk, q, qo, 3, scales[B])
            srcs = {"native_play": val[B][:4000], "native_kettle": norm(B, np.concatenate(kq[B]))}
            for A in FOUR:
                srcs[f"from_{A}_play"] = dec(B, zval[A][:4000])
                srcs[f"from_{A}_kettle"] = dec(B, np.concatenate(ZK[A]))
            srcs["from_other5_play"] = np.concatenate([dec(B, zval[A][:2000]) for A in FIVE if A != B])
            srcs["from_other5_kettle"] = np.concatenate([dec(B, np.concatenate(ZK[A])) for A in FIVE if A != B])
            dec5[B] = {}
            for s, q in srcs.items():
                L, Rg = lc(q), rc(q)
                dec5[B][s] = {"little_closure": dstats(L), "ring_closure": dstats(Rg),
                              "corr_little_ring": float(np.corrcoef(L, Rg)[0, 1]),
                              "mean_abs_little_minus_ring": float(np.abs(L - Rg).mean()),
                              "little_joint_mean_norm": r3(q[:, cols["little"]].mean(0)),
                              "little_joint_std_norm": r3(q[:, cols["little"]].std(0))}
        RES["dec5_from_4f_little"] = dec5
        # ---- Q3 per-pair retarget pad residual (G5 metric) on play val[:2048] ----
        with torch.no_grad():
            vb = {h: torch.tensor(play[h]["val"][:2048]) for h in fks}
            _, _, _, _, diag = objective(vb, codecs, fks, payload["config"]["beta"], payload["config"]["lam"], "e2",
                                         sample=False, diagnostics=True)
        pairs = {}
        for key, d in diag.items():
            e = d["err"].norm(dim=-1).numpy() * 1000; w = d["w"].numpy(); F = len(d["fingers"])
            pf = {FINGERS[fi]: float((e[:, j] * w[:, j]).sum() / w[:, j].sum()) for j, fi in enumerate(d["fingers"])}
            pf["thumb"] = float((e[:, F:2 * F] * w).sum() / w.sum())
            pf["unweighted_mean_all_pts"] = float(e[:, :2 * F].mean())
            pairs[key] = pf
        cat = {}
        for key, pf in pairs.items():
            a, b = key.split("->")
            c = ("5" if a in FIVE else "4") + ("<->" if (a in FIVE) == (b in FIVE) else "->") + ("5" if b in FIVE else "4")
            cat.setdefault(c, []).append(pf)
        def summarize(lst):
            fs = [f for f in FINGERS if f in lst[0]]
            per = {f: float(np.mean([p[f] for p in lst])) for f in fs}
            return {"n_pairs": len(lst), "mean_over_fingers": float(np.mean(list(per.values()))),
                    "mean_over_thumb_index_middle_ring": float(np.mean([per[f] for f in ("thumb", "index", "middle", "ring")])),
                    "per_finger": per, "unweighted_mean_all_pts": float(np.mean([p["unweighted_mean_all_pts"] for p in lst]))}
        RES["retarget_pairs"] = pairs
        RES["retarget_by_category"] = {c: summarize(v) for c, v in sorted(cat.items())}
        # G5-style overall (per finger averaged over pairs that have it, then mean) for sanity vs measure json
        allf = {f: np.mean([p[f] for p in pairs.values() if f in p]) for f in FINGERS}
        RES["retarget_G5_recomputed"] = float(np.mean(list(allf.values())))
        print(cname, "retarget", json.dumps({c: round(v["mean_over_fingers"], 2) for c, v in RES["retarget_by_category"].items()}),
              "G5", round(RES["retarget_G5_recomputed"], 3), flush=True)
        # ---- viz ----
        Zv = np.concatenate([enc(h, sets[f"GEO_ALL5_{h}"][viz_s])[0] for h in ALL5])
        hv = np.repeat(np.arange(5), args.n_viz)
        mu = Zv.mean(0); ev, Uv = np.linalg.eigh(cov(Zv)); ev, Uv = ev[::-1], Uv[:, ::-1]
        Y2 = (Zv - mu) @ Uv[:, :2]
        Zk = np.concatenate([np.concatenate(ZK[h])[ket_sel[h]] for h in ALL5]); hk = np.repeat(np.arange(5), args.n_kettle)
        K2 = (Zk - mu) @ Uv[:, :2]
        m = json.load(open(f"artifacts/codec_v2/measure/{MEASURE[cname]}.json"))
        stats = {"probe_GEO_ALL5": m["GEO_ALL5"]["probe_raw"]["acc"], "probe_GEO_H3": m["GEO_H3"]["probe_raw"]["acc"],
                 "probe_SYN_ALL5": m["SYN_ALL5"]["probe_raw"]["acc"],
                 "same_within_GEO_ALL5": m["GEO_ALL5"]["dist"]["MEAN"]["same/within"],
                 "same_within_GEO_H3": m["GEO_H3"]["dist"]["MEAN"]["same/within"],
                 "hand_var_share_GEO_ALL5": m["GEO_ALL5"]["two_way"]["frac_hand"],
                 "hand_var_share_GEO_H3": m["GEO_H3"]["two_way"]["frac_hand"],
                 "kettle_binary_probe_allegro": m["KETTLE_K2"]["allegro"]["binary_lr_acc"],
                 "kettle_binary_probe_leap": m["KETTLE_K2"]["leap"]["binary_lr_acc"],
                 "kettle_frac_inside_H3_p95_allegro": m["KETTLE_K2"]["allegro"]["frac_inside_ref_p95"],
                 "kettle_frac_inside_H3_p95_leap": m["KETTLE_K2"]["leap"]["frac_inside_ref_p95"],
                 "pad_residual_mm": m["quality"]["pad_residual_mm_mean"],
                 "probe_GEO_ALL5_euclid_nearest_centroid": RES["GEO_ALL5"]["gauss"]["nc"],
                 "probe_GEO_ALL5_top8pc": RES["GEO_ALL5"]["probe_topk"][8],
                 "probe_GEO_ALL5_without_top8pc": RES["GEO_ALL5"]["probe_without_topk"][8]}
        VIZ[cname] = {"pca": {"x": r3(Y2[:, 0]), "y": r3(Y2[:, 1]), "hand": hv.tolist()},
                      "pairs": viz_pairs,
                      "kettle": {"x": r3(K2[:, 0]), "y": r3(K2[:, 1]), "hand": hk.tolist()},
                      "stats": {k: round(float(v), 4) for k, v in stats.items()},
                      "pca_var": r3(ev[:2] / ev.sum())}
        if not args.no_tsne:
            T2 = tsne(Zv)
            VIZ[cname]["tsne"] = {"x": r3(T2[:, 0]), "y": r3(T2[:, 1]), "hand": hv.tolist()}
        ALLRES[cname] = RES
        (out / "separability_results.json").write_text(json.dumps(ALLRES, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print("[done codec]", cname, round(time.time() - t0), "s", flush=True)
    notes = {
        "hands": "index into the top-level 'hands' list (allegro, leap = 4-finger; shadow, wuji, sharpa = 5-finger)",
        "pca": "z = encoder mean on the codec-independent GEO-val ALL5 pair set (artifacts/codec_v2/measure/pairsets.npz), "
               f"{args.n_viz} scenes x 5 hands, the SAME scenes for every codec. Points are hand-major: index = hand*{args.n_viz} + scene. "
               "2-component PCA fit on the pooled z of these points, per codec.",
        "tsne": "exact t-SNE (own torch implementation, perplexity 30, 1000 iters, PCA init) on the same z, per codec",
        "pairs": "40 same-scene cross-hand links [i, j] into the pca/tsne point lists (same links for every codec); "
                 "GEO pairs = nearest neighbour by palm-frame fingertips / hand scale (4 fingers when 4-finger hands are involved)",
        "kettle": f"Kettle commanded finger q (data/replay eps 0-9), {args.n_kettle} random frames per hand for all 5 hands, "
                  "encoded and projected with the SAME PCA as 'pca' (not refit)",
        "stats": "probe_* = linear hand probe accuracy (standardized multinomial LR, grouped scene split; chance 0.2 ALL5 / 0.33 H3); "
                 "same_within = same-scene cross-hand distance / same-hand different-scene distance; hand_var_share = two-way "
                 "frac_hand; kettle_* = binary H3-vs-unseen probe and fraction inside H3 kNN p95 (from artifacts/codec_v2/measure); "
                 "*_euclid_nearest_centroid / *_top8pc / *_without_top8pc from this analysis",
        "pca_var": "explained variance ratio of PC1, PC2 on the pooled viz z",
        "source": "xhand_latent/cv2_separability.py, analysis in artifacts/codec_v2/viz/RESULTS_SEPARABILITY.md"}
    (out / "latent_viz.json").write_text(json.dumps({"codecs": VIZ, "hands": list(ALL5), "notes": notes}, separators=(",", ":")))
    print("[cv2_separability] DONE", round(time.time() - t0), "s", flush=True)


def tsne(X, perp=30.0, iters=1000, seed=0):
    X = torch.tensor(X, dtype=torch.float64); N = len(X)
    D = torch.cdist(X, X) ** 2
    D = D / D[D > 0].median()
    beta = torch.ones(N, dtype=torch.float64); lo = torch.zeros(N, dtype=torch.float64); hi = torch.full((N,), float("inf"), dtype=torch.float64)
    logU = np.log(perp); eye = torch.eye(N, dtype=torch.bool)
    for _ in range(100):
        P = torch.exp(-D * beta[:, None]).masked_fill(eye, 0); sP = P.sum(1).clamp_min(1e-300)
        Hh = torch.log(sP) + beta * (D * P).sum(1) / sP
        up = Hh > logU
        lo = torch.where(up, beta, lo); hi = torch.where(up, hi, beta)
        beta = torch.where(up, torch.where(torch.isinf(hi), beta * 2, (beta + hi) / 2), (beta + lo) / 2)
    P = P / sP[:, None]; P = (P + P.T) / (2 * N); P = P.clamp_min(1e-12)
    Xc = X - X.mean(0); _, _, V = torch.linalg.svd(Xc, full_matrices=False)
    Y = Xc @ V[:2].T; Y = Y / Y[:, 0].std() * 1e-4
    vel = torch.zeros_like(Y); gains = torch.ones_like(Y); lr = max(N / 12.0, 50.0)
    for it in range(iters):
        ex = 12.0 if it < 250 else 1.0; mom = 0.5 if it < 250 else 0.8
        num = 1.0 / (1.0 + torch.cdist(Y, Y) ** 2); num = num.masked_fill(eye, 0)
        Q = (num / num.sum()).clamp_min(1e-12)
        PQ = (ex * P - Q) * num
        grad = 4 * ((PQ.sum(1))[:, None] * Y - PQ @ Y)
        gains = torch.where((grad > 0) != (vel > 0), gains + 0.2, gains * 0.8).clamp_min(0.01)
        vel = mom * vel - lr * gains * grad
        Y = Y + vel; Y = Y - Y.mean(0)
    return Y.numpy()


if __name__ == "__main__":
    main()
