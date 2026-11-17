"""codec v2 (cv2, 261115) measurement: zpair metrics (artifacts/pooled3hand/ident/zpair/zpair.py, zpair2.py; probe /
variance / distance functions copied verbatim) for any HandVAE checkpoint, plus codec quality and Kettle OOD. New file. CPU.
Pair sets are codec-independent and cached once (--sets npz) so every codec sees identical scenes:
  SYN  : 3000 clean synergy scenes (zpair construction, seed 0), H3 and ALL5
  GEO  : val pool, zpair procedure (anchor rotation, NN by palm-frame tips / hand scale, better-matched half), H3 and ALL5
  RT   : codec-defined Enc_B(Dec_B(z_A)) on play val (2000 source frames per hand)
Quality: arm_train.objective on play val[:2048] (encoder mean): per-hand joint RMSE, per-finger pad residual (pair-weighted).
Kettle K1: LeRobot observation.state, sharpa/shadow/wuji 50 eps, every 4th frame, train ep0-39 / test ep40-49 (as zpair).
Kettle K2 (OOD): commanded finger q of data/replay pkl eps 0-9, 5 hands; pool = H3 eps 0-7, ref = H3 eps 8-9.
Container python, cwd /workspace/xhand/latent_codec_v1:
  CUDA_VISIBLE_DEVICES= nice -n 10 python.sh -m xhand_latent.cv2_measure --codec NAME=PATH --out artifacts/codec_v2/measure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .arm_train import load_arm, objective, summarize_diag
from .data import feature_names, load_trajectory, trajectory_path

ALL5 = ("allegro", "leap", "shadow", "wuji", "sharpa")
H3 = ("sharpa", "shadow", "wuji")
TASK = "Dexverse-GraspKettle-v0"
DATA = "data/replay"
LEROBOT = Path("/workspace/xhand/data/dv-demos-v0/v0/teleop_lerobot/functional") / TASK
E2 = "artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt"


# ---------------- probe / variance / distance (verbatim from zpair.py / zpair2.py, rng made explicit) ----------------
def fit_lr(Xtr, ytr, Xte, l2, K, iters=150):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float64); B = torch.tensor((Xte - mu) / sd, dtype=torch.float64)
    y = torch.tensor(ytr)
    W = torch.zeros(A.shape[1], K, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(A @ W + b, y) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (B @ W + b).argmax(1).numpy(), (W.detach().numpy(), b.detach().numpy(), mu, sd)


def center(X, y, tr, K, scale=False):
    X = X.copy()
    for k in range(K):
        m = tr & (y == k)
        mu = X[m].mean(0); sd = X[m].std(0) + 1e-6
        X[y == k] -= mu
        if scale:
            X[y == k] /= sd
    return X


def split_groups(group, test_frac=0.2, seed=0):
    ug = np.unique(group)
    r = np.random.default_rng(seed).permutation(ug)
    teg = set(r[:max(1, int(len(ug) * test_frac))].tolist())
    te = np.array([g in teg for g in group])
    return ~te, te


def probe(X, y, group, K, mode="raw", l2s=(1e-4, 1e-3, 1e-2, 1e-1), test_frac=0.2, seed=0, split=None):
    tr, te = split if split is not None else split_groups(group, test_frac, seed)
    if mode != "raw":
        X = center(X, y, tr, K, scale=(mode == "cscale"))
    gtr = group[tr]
    fold = (np.searchsorted(np.unique(gtr), gtr) * 7919) % 4
    best, bacc = None, -1
    for l2 in l2s:
        acc = np.mean([(fit_lr(X[tr][fold != f], y[tr][fold != f], X[tr][fold == f], l2, K)[0] == y[tr][fold == f]).mean()
                       for f in range(4)])
        if acc > bacc:
            best, bacc = l2, acc
    pred, model = fit_lr(X[tr], y[tr], X[te], best, K)
    return {"acc": float((pred == y[te]).mean()), "cv": float(bacc), "l2": best,
            "per_hand": [float((pred[y[te] == k] == k).mean()) for k in range(K)],
            "n_tr": int(tr.sum()), "n_te": int(te.sum()), "chance": 1.0 / K}, model


def knn_acc(X, y, tr, te, k=15):
    P = torch.tensor(X[tr]); out = []
    for s in range(0, te.sum(), 2000):
        D = torch.cdist(torch.tensor(X[te][s:s + 2000]), P)
        nb = D.topk(k, largest=False).indices.numpy()
        out.append(np.array([np.bincount(y[tr][r], minlength=y.max() + 1).argmax() for r in nb]))
    return float((np.concatenate(out) == y[te]).mean())


def var_hand(X, y):
    mu = X.mean(0)
    K = y.max() + 1
    sb = sum((y == k).sum() * np.square(X[y == k].mean(0) - mu).sum() for k in range(K))
    st = np.square(X - mu).sum()
    return {"eta2_hand": float(sb / st), "fisher_tr": float(sb / (st - sb))}


def two_way(Z):
    mu = Z.mean((0, 1))
    a = Z.mean(0) - mu
    b = Z.mean(1) - mu
    S, H, _ = Z.shape
    st = np.square(Z - mu).sum()
    sh = S * np.square(a).sum(); ss = H * np.square(b).sum()
    return {"frac_hand": float(sh / st), "frac_scene": float(ss / st), "frac_resid": float((st - sh - ss) / st)}


def dist_table(Z, hands, seed=0):
    S, H, _ = Z.shape
    perm = np.random.default_rng(seed).permutation(S)
    perm = np.where(perm == np.arange(S), (perm + 1) % S, perm)
    Zc = Z - Z.mean(0, keepdims=True)
    rows = {}
    for i in range(H):
        for j in range(i + 1, H):
            same = np.linalg.norm(Z[:, i] - Z[:, j], axis=1).mean()
            rand = np.linalg.norm(Z[:, i] - Z[perm, j], axis=1).mean()
            within = 0.5 * (np.linalg.norm(Z[:, i] - Z[perm, i], axis=1).mean()
                            + np.linalg.norm(Z[:, j] - Z[perm, j], axis=1).mean())
            off = np.linalg.norm(Z[:, i].mean(0) - Z[:, j].mean(0))
            samec = np.linalg.norm(Zc[:, i] - Zc[:, j], axis=1).mean()
            rows[f"{hands[i]}-{hands[j]}"] = {"same": float(same), "rand_cross": float(rand), "within_diff": float(within),
                                              "same/rand": float(same / rand), "same/within": float(same / within),
                                              "offset": float(off), "offset/within": float(off / within),
                                              "same_centered": float(samec), "same_centered/rand": float(samec / rand)}
    keys = list(next(iter(rows.values())).keys())
    rows["MEAN"] = {k: float(np.mean([r[k] for r in rows.values()])) for k in keys}
    return rows


def paired_block(Z, hands, name):
    S, H, D = Z.shape
    X = Z.reshape(S * H, D); y = np.tile(np.arange(H), S); g = np.repeat(np.arange(S), H)
    tr, te = split_groups(g)
    Xc = center(X, y, tr, H)
    res = {"n_scenes": S, "hands": list(hands), "probe_raw": probe(X, y, g, H, "raw", split=(tr, te))[0],
           "knn15_raw": knn_acc(X, y, tr, te), "knn15_centered": knn_acc(Xc, y, tr, te),
           "var": var_hand(X, y), "two_way": two_way(Z), "dist": dist_table(Z, hands)}
    m = res["dist"]["MEAN"]
    print(name, json.dumps({"lin": res["probe_raw"]["acc"], "knn": res["knn15_raw"], "knn_c": res["knn15_centered"],
                            "frac_hand": res["two_way"]["frac_hand"], "same/within": m["same/within"],
                            "same/rand": m["same/rand"]}), flush=True)
    return res


# ---------------- codec-independent pair sets ----------------
def make_sets(play, fks, scales, path, S=3000, seed=0):
    if Path(path).exists():
        return dict(np.load(path, allow_pickle=True))
    rng = np.random.default_rng(seed)
    sets = {}
    for tag, hands in (("H3", H3), ("ALL5", ALL5)):
        names = sorted(set.intersection(*[set(str(s) for s in play[h]["prototype_names"]) for h in hands]))
        i = rng.integers(len(names), size=S); j = rng.integers(len(names), size=S); r = rng.uniform(0, 1, S)
        for h in hands:
            pn = [str(s) for s in play[h]["prototype_names"]]
            P = play[h]["prototypes"]
            sets[f"SYN_{tag}_{h}"] = np.stack([P[pn.index(names[a])] * rr + P[pn.index(names[b])] * (1 - rr)
                                               for a, b, rr in zip(i, j, r)]).astype(np.float32)
    # GEO (zpair.geo_scenes, val pool)
    for tag, hands in (("H3", H3), ("ALL5", ALL5)):
        nf = 5 if all(bool(fks[h].valid.all()) for h in hands) else 4
        Q = {h: play[h]["val"] for h in hands}
        with torch.no_grad():
            T = {h: (fks[h](torch.as_tensor(Q[h]))[:, :nf] / scales[h]).reshape(len(Q[h]), -1).numpy() for h in hands}
        H = len(hands)
        anchors = [(s % H, rng.integers(len(Q[hands[s % H]]))) for s in range(S)]
        sel = np.zeros((S, H), int); res = np.zeros((S, H))
        for a in range(H):
            rows = [s for s in range(S) if anchors[s][0] == a]
            ai = np.array([anchors[s][1] for s in rows])
            sel[rows, a] = ai
            for b in range(H):
                if b == a:
                    continue
                D = torch.cdist(torch.tensor(T[hands[a]][ai]), torch.tensor(T[hands[b]]))
                v, k = D.min(1)
                sel[rows, b] = k.numpy(); res[rows, b] = v.numpy()
        tip = res.sum(1) / (H - 1) / np.sqrt(nf) * 1000 * np.mean([scales[h] for h in hands])
        keep = tip <= np.median(tip)
        for b, h in enumerate(hands):
            sets[f"GEO_{tag}_{h}"] = Q[h][sel[keep, b]].astype(np.float32)
        sets[f"GEO_{tag}_tip_mm_median_kept"] = np.array(np.median(tip[keep]))
    np.savez_compressed(path, **sets)
    return sets


# ---------------- Kettle ----------------
def kettle_state_q(fks, hand, eps):
    """finger joint rows (rad, fk.joint_names order) of LeRobot observation.state per episode (= HoCodec.q_from_state +
    finger_rows)."""
    fk = fks[hand]
    _, sn = feature_names(DATA, TASK, hand)
    out = []
    for e in eps:
        t = pq.read_table(LEROBOT / f"floating_{hand}_right/data/chunk-000/file-{e:03d}.parquet", columns=["observation.state"])
        st = np.array(t.column("observation.state").to_pylist(), np.float64)
        out.append(st[:, [sn.index(n) for n in fk.joint_names]])
    return out


def kettle_action_q(fks, hand, eps):
    """commanded finger q (rad) = P3HCodec.q_from_action + finger_rows: action + robotmeta default offset (the wrist-only
    inferred offset of P3HCodec does not touch finger joints), clipped to the reference limits."""
    fk = fks[hand]
    an, _ = feature_names(DATA, TASK, hand)
    meta = fk.spec.get("robotmeta")
    default = (np.array(meta["action_offset"])[[meta["action_joint_names"].index(n) for n in an]] if meta
               else np.array([.3 if n == "z_translation_joint" else 0 for n in an]))
    lo, hi = np.asarray(fk.reference.lower), np.asarray(fk.reference.upper)
    cols = [an.index(n) for n in fk.joint_names]
    ri = [fk.reference.qidx[n] for n in fk.joint_names]
    out = []
    for e in eps:
        a = np.asarray(load_trajectory(trajectory_path(DATA, TASK, hand, e))["episodes"][0]["actions"], np.float64)
        out.append(np.clip(a[:, cols] + default[cols], lo[ri], hi[ri]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", action="append", required=True, help="NAME=PATH (repeatable)")
    ap.add_argument("--out", default="artifacts/codec_v2/measure")
    ap.add_argument("--sets", default="artifacts/codec_v2/measure/pairsets.npz")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-kettle", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    play = {h: np.load(f"artifacts/play_v1/{h}.npz", allow_pickle=True) for h in ALL5}
    _, fks, p0 = load_arm(E2)
    scales = p0["config"]["hand_scales_m"]
    sets = make_sets(play, fks, scales, args.sets)
    kq = None
    if not args.no_kettle:
        kq = {"state": {h: kettle_state_q(fks, h, range(50)) for h in H3},
              "action": {h: kettle_action_q(fks, h, range(10)) for h in ALL5}}
    for spec in args.codec:
        name, path = spec.split("=", 1)
        codecs, cfks, payload = load_arm(path)

        @torch.no_grad()
        def enc(h, q):
            return codecs[h].encode(torch.as_tensor(np.asarray(q), dtype=torch.float32))[0].numpy()

        @torch.no_grad()
        def dec(h, z):
            return codecs[h].decoder(torch.as_tensor(np.asarray(z), dtype=torch.float32)).numpy()

        R = {"codec": path, "step": payload.get("step"), "cv2": payload["config"].get("cv2"),
             "centered": "cv2_center" in payload["config"]}
        for kind in ("SYN", "GEO"):
            for tag, hands in (("H3", H3), ("ALL5", ALL5)):
                Z = np.stack([enc(h, sets[f"{kind}_{tag}_{h}"]) for h in hands], 1)
                R[f"{kind}_{tag}"] = paired_block(Z, hands, f"{name} {kind}_{tag}")
        # RT
        for tag, hands in (("H3", H3), ("ALL5", ALL5)):
            H = len(hands)
            src = {h: enc(h, play[h]["val"][:2000]) for h in hands}
            Zrt = {(a, b): enc(B, dec(B, src[A])) for a, A in enumerate(hands) for b, B in enumerate(hands)}
            Zs = np.concatenate([np.stack([Zrt[(a, b)] for b in range(H)], 1) for a in range(H)])
            blk = paired_block(Zs, hands, f"{name} RT_{tag}")
            cross = float(np.mean([np.linalg.norm(Zrt[(a, b)] - src[A], axis=1).mean()
                                   for a, A in enumerate(hands) for b in range(H) if b != a]))
            selfd = float(np.mean([np.linalg.norm(Zrt[(a, a)] - src[A], axis=1).mean() for a, A in enumerate(hands)]))
            blk["rt_drift_cross"], blk["rt_drift_self"] = cross, selfd
            blk["rt_drift_cross/rand"] = cross / blk["dist"]["MEAN"]["rand_cross"]
            R[f"RT_{tag}"] = blk
        # unpaired play val probe (informational)
        X = np.concatenate([enc(h, play[h]["val"][:4000]) for h in ALL5]); y = np.repeat(np.arange(5), 4000)
        R["PLAY_unpaired_ALL5"] = {"probe_raw": probe(X, y, np.arange(len(X)), 5)[0]["acc"], "var": var_hand(X, y)}
        # quality
        with torch.no_grad():
            vb = {h: torch.tensor(play[h]["val"][:2048]) for h in cfks}
            _, _, values, stats, diag = objective(vb, codecs, cfks, payload["config"]["beta"], payload["config"]["lam"], "e2",
                                                  sample=False, diagnostics=True)
            q = summarize_diag(diag, cfks)
            q["joint_rmse_rad"] = {h: float((cfks[h].denormalize(a) - cfks[h].denormalize(b)).square().mean().sqrt())
                                   for h, (a, b) in stats.items()}
            q["rec"], q["align_raw"] = float(values["rec"]), float(values["align_raw"])
            pw = q["per_finger_contact_residual_mm_pairweighted"]
            q["pad_residual_mm_mean"] = float(np.mean(list(pw.values())))
        R["quality"] = q
        print(name, "quality", json.dumps({"pad_mm": q["pad_residual_mm_mean"], "rmse": q["joint_rmse_rad"]}), flush=True)
        # Kettle
        if kq is not None:
            norm = lambda h, qf: cfks[h].normalize(torch.tensor(np.asarray(qf), dtype=torch.float32)).clamp(-1, 1)
            # K1 (zpair: every 4th frame, balanced per hand within train / test)
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
            k1 = {"n": int(len(KY)), "probe_raw": probe(KZ, KY, KE, 3, split=(KE < 40, KE >= 40))[0],
                  "knn15_raw": knn_acc(KZ, KY, KE < 40, KE >= 40), "var": var_hand(KZ, KY)}
            mus = [KZ[KY == k].mean(0) for k in range(3)]
            within = float(np.mean([np.linalg.norm(KZ[KY == k] - KZ[KY == k][rng.permutation((KY == k).sum())], axis=1).mean()
                                    for k in range(3)]))
            k1["offset_pairs"] = {f"{H3[i]}-{H3[j]}": float(np.linalg.norm(mus[i] - mus[j])) for i in range(3) for j in range(i + 1, 3)}
            k1["within_rms_dist"] = within
            k1["offset/within_mean"] = float(np.mean(list(k1["offset_pairs"].values())) / within)
            R["KETTLE_K1"] = k1
            # K2 OOD
            Zk = {h: [enc(h, norm(h, qf)) for qf in kq["action"][h]] for h in ALL5}
            pool = np.concatenate([z for h in H3 for z in Zk[h][:8]])
            refz = np.concatenate([z for h in H3 for z in Zk[h][8:]])
            P = torch.tensor(pool)

            def knnd(Zq, k=10):
                return torch.cdist(torch.tensor(Zq), P).topk(k, largest=False).values.mean(1).numpy()
            dref = knnd(refz)
            k2 = {"ref_median": float(np.median(dref)), "ref_p95": float(np.percentile(dref, 95))}
            # leave-one-hand-out reference: a training hand's frames vs the pool of the OTHER two H3 hands (eps 0-7);
            # = how far a hand the pool has never seen sits when the codec is as aligned as it is among H3
            lo = []
            for h in H3:
                Po = torch.tensor(np.concatenate([z for o in H3 if o != h for z in Zk[o][:8]]))
                lo.append(np.median(torch.cdist(torch.tensor(np.concatenate(Zk[h])), Po).topk(10, largest=False).values.mean(1).numpy()))
            k2["loho_ref_median_per_hand"] = [float(x) for x in lo]
            k2["loho_ref_median"] = float(np.mean(lo))
            h3mu = {h: np.concatenate(Zk[h]).mean(0) for h in H3}
            h3off = float(np.mean([np.linalg.norm(h3mu[a] - h3mu[b]) for i, a in enumerate(H3) for b in H3[i + 1:]]))
            h3c = np.mean(list(h3mu.values()), 0)
            for u in ("allegro", "leap"):
                zu = np.concatenate(Zk[u]); du = knnd(zu)
                # binary H3 vs u, episode-grouped (train eps 0-7, test 8-9), balanced
                Xb, yb, eb = [], [], []
                for lab, hs in ((0, H3), (1, (u,))):
                    for h in hs:
                        for e, z in enumerate(Zk[h]):
                            Xb.append(z); yb.append(np.full(len(z), lab)); eb.append(np.full(len(z), e))
                Xb, yb, eb = np.concatenate(Xb), np.concatenate(yb), np.concatenate(eb)
                bal = np.zeros(len(yb), bool)
                for part in (eb < 8, eb >= 8):
                    m = min(((yb == c) & part).sum() for c in range(2))
                    for c in range(2):
                        ii = np.where((yb == c) & part)[0]
                        bal[rng.choice(ii, m, replace=False)] = True
                Xb, yb, eb = Xb[bal], yb[bal], eb[bal]
                k2[u] = {"knn10_median_ratio": float(np.median(du) / np.median(dref)),
                         "knn10_median_ratio_vs_loho": float(np.median(du) / k2["loho_ref_median"]),
                         "frac_inside_ref_p95": float((du <= k2["ref_p95"]).mean()),
                         "binary_lr_acc": probe(Xb, yb, eb, 2, split=(eb < 8, eb >= 8))[0]["acc"],
                         "mean_dist_to_H3_center/H3_offsets": float(np.linalg.norm(zu.mean(0) - h3c) / h3off)}
            R["KETTLE_K2"] = k2
            print(name, "kettle", json.dumps({"K1_lin": k1["probe_raw"]["acc"], "K1_eta2": k1["var"]["eta2_hand"],
                                              "K1_off/within": k1["offset/within_mean"], "K2": k2}), flush=True)
        (out / f"{name}.json").write_text(json.dumps(R, indent=1))
        print("[cv2_measure] DONE", name, flush=True)


if __name__ == "__main__":
    main()
