"""P0a (pooled3hand_design_261112 §4.1/§5/§6-3): segmentation, K / f_task training (unpaired, kettle frame), offline
Fisher-ratio alignment for raw / K / G2 f / f_task / F. New file. Container python on work2, cwd latent_codec_v1:
  python.sh -m xhand_latent.p3h_train
Inputs: artifacts/pooled3hand/p0a/extract_<hand>.npz (p3h_extract.py).
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .p3h_offsets import HANDS3, G2F, FNet, Hand, KNet, inv, make_T  # noqa: F401

OUT = Path("artifacts/pooled3hand/p0a")
SEGS = ("approach", "grasp", "lift", "hold")
NB = 20
W_ROT = 0.10  # m per rad, as G2
SIGMA = 0.05
N_TR = 40
DEV = "cuda"


# ----------------------------------------------------------------------------- segmentation
def ang_np(R1, R2):
    c = (np.trace(np.swapaxes(R1, -1, -2) @ R2, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def first_run(r, n=5, start=0):
    for t in range(start, len(r) - n + 1):
        if r[t:t + n].all():
            return t
    return -1


def segment(O, W, C):
    """O, W (T,4,4) kettle / canonical wrist world poses, C (T,8) touch_contact. Returns dict."""
    T = len(O)
    z = O[:, 2, 3] - O[10, 2, 3]  # frames 0-~5 = kettle settling after reset
    t_lift = first_run(z > 0.01, 5, 10)
    out = {"T": T, "t_lift": int(t_lift)}
    if t_lift < 0:
        out["ok"] = False
        return out
    zmax = z.max()
    t_top = int(np.argmax(z >= 0.9 * zmax))
    Wk = np.linalg.inv(O) @ W
    dp = np.linalg.norm(Wk[:, :3, 3] - Wk[t_lift, :3, 3], axis=-1) * 1000
    dr = ang_np(Wk[:, :3, :3], Wk[t_lift:t_lift + 1, :3, :3])
    near = (dp < 20) & (dr < 10)
    t_g = t_lift
    while t_g > 0 and near[t_g - 1]:
        t_g -= 1
    # diagnostics: design contact rule (thumb + >=2 finger slots, >=5 steps) and tilt>15 deg pour rule
    fing, th = (C[:, :4] > 0).sum(1), C[:, 4] > 0
    t_c = first_run(th & (fing >= 2), 5)
    t_any = first_run((C > 0).any(1), 5)
    tilt = ang_np(O[:, :3, :3], O[10:11, :3, :3])
    t_pour = first_run(tilt > 15, 5, t_lift)
    out.update({"t_g": int(t_g), "t_top": int(t_top), "zmax_cm": float(zmax * 100),
                "t_contact_design": int(t_c), "t_contact_any": int(t_any), "t_pour_tilt15": int(t_pour),
                "tilt_max_after_lift_deg": float(tilt[t_lift:].max())})
    out["ok"] = bool(0 < t_g < t_lift < t_top < T - 1)
    out["flag_short_grasp"] = bool(t_lift - t_g < 3)
    return out


def bin_index(seg):
    b = [0, seg["t_g"], seg["t_lift"], seg["t_top"], seg["T"] - 1]
    s = (np.arange(NB) + 0.5) / NB
    return np.concatenate([np.round(b[j] + s * (b[j + 1] - b[j])).astype(int) for j in range(4)])


# ----------------------------------------------------------------------------- data
class HandData:
    def __init__(self, hand):
        d = np.load(OUT / f"extract_{hand}.npz")
        self.name = hand
        L = d["lengths"]
        self.L, self.off = L, np.r_[0, np.cumsum(L)]
        self.n_ep = len(L)
        W, O = d["wrist_w"].astype(np.float64), d["obj_w"].astype(np.float64)
        self.Wk_np = np.linalg.inv(O) @ W
        self.segs, self.idx = [], np.zeros((self.n_ep, 4 * NB), np.int64)
        for e in range(self.n_ep):
            s = slice(self.off[e], self.off[e + 1])
            sg = segment(O[s], W[s], d["contact"][s])
            self.segs.append(sg)
            if sg["ok"]:
                self.idx[e] = self.off[e] + bin_index(sg)
        self.ok = np.array([s["ok"] for s in self.segs])
        self.Wk = torch.tensor(self.Wk_np, dtype=torch.float32, device=DEV)
        self.q = torch.tensor(d["q"], dtype=torch.float32, device=DEV)
        self.H = Hand(hand, DEV)
        self.lo, self.hi = self.H.fk.lower, self.H.fk.upper
        self.tr_eps = [e for e in range(N_TR) if self.ok[e]]
        self.va_eps = [e for e in range(N_TR, self.n_ep) if self.ok[e]]
        pairs = np.concatenate([np.arange(self.off[e], self.off[e + 1] - 1) for e in self.tr_eps])
        self.pairs = torch.tensor(pairs, device=DEV)
        self.idx_t = torch.tensor(self.idx, device=DEV)

    def noisy(self, q):
        return torch.maximum(torch.minimum(q + SIGMA * torch.randn_like(q), self.hi), self.lo)


def off_of(model, kind, D, hi, q):
    if kind == "K":
        return model(hi, q.shape[0])
    return model(*D.H.features(q))


@torch.no_grad()
def all_offsets(kind, model, D, hi, chunk=4096):
    out = []
    for a in range(0, len(D.q), chunk):
        q = D.q[a:a + chunk]
        if kind == "raw":
            out.append(torch.eye(4, device=DEV).expand(len(q), 4, 4))
        elif kind == "F_TPF":
            out.append(D.H.e0_frame(q))
        else:
            out.append(off_of(model, kind, D, hi, q))
    return torch.cat(out)


# ----------------------------------------------------------------------------- training
def loss_terms(model, kind, Ds, cfg, noise=True):
    Es, Ts = [], []
    for hi, D in enumerate(Ds):
        idx = D.idx_t[D.tr_eps].reshape(-1)
        q = D.q[idx]
        if noise and kind != "K":
            q = D.noisy(q)
        T = off_of(model, kind, D, hi, q)
        Es.append((D.Wk[idx] @ inv(T)).reshape(len(D.tr_eps), 4 * NB, 4, 4))
        Ts.append(T.reshape(len(D.tr_eps), 4 * NB, 4, 4))
    mu = torch.stack([E[..., :3, 3].mean(0) for E in Es])  # (H,B,3)
    Rb = torch.stack([E[..., :3, :3].mean(0) for E in Es])  # (H,B,3,3)
    sd = torch.stack([E[..., :3, 3].std(0) for E in Es])
    rho = torch.stack([(E[..., :3, :3] - E[..., :3, :3].mean(0, keepdim=True)).flatten(-2).norm(dim=-1).mean(0) for E in Es])
    match = cov = 0
    n = len(Ds)
    for a in range(n):
        for b in range(a + 1, n):
            match = match + (mu[a] - mu[b]).norm(dim=-1) + W_ROT * (Rb[a] - Rb[b]).flatten(-2).norm(dim=-1) / math.sqrt(2)
            cov = cov + (sd[a] - sd[b]).norm(dim=-1) + W_ROT * (rho[a] - rho[b]).abs()
    npair = n * (n - 1) / 2
    match, cov = match.mean() / npair, cov.mean() / npair
    tm = torch.stack([T[..., :3, 3].mean(0) for T in Ts]).mean(0)  # (B,3) mean over hands of per-hand mean
    Rm = torch.stack([T[..., :3, :3].mean(0) for T in Ts]).mean(0)
    eye = torch.eye(3, device=DEV)
    zero = tm.norm(dim=-1).mean() + W_ROT * (Rm - eye).flatten(-2).norm(dim=-1).mean() / math.sqrt(2)
    l2 = torch.stack([(T[..., :3, 3].norm(dim=-1) + W_ROT * (T[..., :3, :3] - eye).flatten(-2).norm(dim=-1) / math.sqrt(2)).mean()
                      for T in Ts]).mean()
    smooth = torch.zeros((), device=DEV)
    if kind != "K" and cfg["smooth"] > 0:
        sm = []
        for hi, D in enumerate(Ds):
            p = D.pairs[torch.randint(len(D.pairs), (2048,), device=DEV)]
            q1, q2 = D.q[p], D.q[p + 1]
            if noise:
                eps = SIGMA * torch.randn_like(q1)
                q1 = torch.maximum(torch.minimum(q1 + eps, D.hi), D.lo)
                q2 = torch.maximum(torch.minimum(q2 + eps, D.hi), D.lo)
            T = off_of(model, kind, D, hi, torch.cat([q1, q2]))
            T1, T2 = T[:2048], T[2048:]
            sm.append(((T1[:, :3, 3] - T2[:, :3, 3]).norm(dim=-1) +
                       W_ROT * (T1[:, :3, :3] - T2[:, :3, :3]).flatten(-2).norm(dim=-1) / math.sqrt(2)).mean())
        smooth = torch.stack(sm).mean()
    total = match + cfg["cov"] * cov + cfg["zero"] * zero + cfg["smooth"] * smooth + cfg["l2"] * l2
    return total, {"match_mm": match.item() * 1000, "cov_mm": float(cov) * 1000, "zero_mm": zero.item() * 1000,
                   "smooth_mm_per_frame": smooth.item() * 1000, "l2_mm": l2.item() * 1000}


def train(kind, Ds, cfg, log, steps):
    torch.manual_seed(cfg.get("seed", 0))
    if kind == "K":
        model = KNet(len(Ds)).to(DEV)
        lr = 3e-3
    else:
        model = FNet().to(DEV)
        if cfg.get("init") == "f_id":
            model.load_state_dict(torch.load(G2F, map_location=DEV))
        lr = 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4 if kind != "K" else 0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        loss, terms = loss_terms(model, kind, Ds, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % 500 == 0 or s == steps - 1:
            log(f"   [{kind} {cfg['name']}] step {s} loss={loss.item() * 1000:.2f} " +
                " ".join(f"{k}={v:.2f}" for k, v in terms.items()))
    return model.eval()


# ----------------------------------------------------------------------------- metrics
def proj_np(M):
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    U = U.copy()
    U[..., :, 2] *= d[..., None]
    return U @ Vt


def fisher_from(P, R):
    """P (H,n,B,3) mm, R (H,n,B,3,3). Returns per-bin arrays."""
    H, n = P.shape[:2]
    mu_a = P.mean(1)
    mu = mu_a.mean(0)
    btw = ((mu_a - mu) ** 2).sum(-1).mean(0)
    wit = ((P - mu_a[:, None]) ** 2).sum(-1).sum(1).mean(0) / (n - 1)
    Rb_a = proj_np(R.mean(1))
    Rb = proj_np(Rb_a.mean(0))
    btw_r = (ang_np(Rb_a, Rb[None]) ** 2).mean(0)
    wit_r = ((ang_np(R, Rb_a[:, None]) ** 2).sum(1) / (n - 1)).mean(0)
    pd, pr = [], []
    for a in range(H):
        for b in range(a + 1, H):
            pd.append(np.linalg.norm(mu_a[a] - mu_a[b], axis=-1))
            pr.append(ang_np(Rb_a[a], Rb_a[b]))
    return {"fr_pos": btw / wit, "fr_rot": btw_r / wit_r, "pair_mm": np.mean(pd, 0), "pair_deg": np.mean(pr, 0),
            "within_mm": np.sqrt(wit), "within_deg": np.sqrt(wit_r)}


def summarize(m):
    out = {}
    for k, v in m.items():
        out[k] = {"overall_median": float(np.median(v)), "overall_mean": float(np.mean(v)),
                  **{s: float(np.median(v[j * NB:(j + 1) * NB])) for j, s in enumerate(SEGS)}}
    return out


def fisher(Es, Ds, split, n_perm=20, seed=0):
    """Es: list per hand of (N_h,4,4) numpy EE_common in kettle frame (all frames)."""
    ns, Ps, Rs = [], [], []
    for E, D in zip(Es, Ds):
        eps = D.tr_eps if split == "train" else D.va_eps
        I = D.idx[eps]  # (n,B)
        ns.append(len(eps))
        Ps.append(E[I, :3, 3] * 1000)
        Rs.append(E[I, :3, :3])
    n = min(ns)
    P = np.stack([p[:n] for p in Ps])
    R = np.stack([r[:n] for r in Rs])
    res = summarize(fisher_from(P, R))
    rng = np.random.default_rng(seed)
    H = P.shape[0]
    nulls = {"fr_pos": [], "fr_rot": []}
    Pf, Rf = P.reshape(H * n, *P.shape[2:]), R.reshape(H * n, *R.shape[2:])
    for _ in range(n_perm):
        perm = rng.permutation(H * n)
        m = fisher_from(Pf[perm].reshape(P.shape), Rf[perm].reshape(R.shape))
        for k in nulls:
            nulls[k].append(np.median(m[k]))
    res["null_perm_overall_median"] = {k: float(np.median(v)) for k, v in nulls.items()}
    res["n_per_hand"] = n
    return res


def score(res):
    return 0.5 * (res["fr_pos"]["overall_median"] + res["fr_rot"]["overall_median"])


def ee_common_all(offs, Ds):
    return [D.Wk_np @ np.linalg.inv(T.double().cpu().numpy()) for T, D in zip(offs, Ds)]


def offset_mag(offs, Ds):
    out = {}
    for T, D in zip(offs, Ds):
        T = T.double().cpu().numpy()
        va = np.concatenate([np.arange(D.off[e], D.off[e + 1]) for e in D.va_eps])
        out[D.name] = {"mean_pos_mm": float(np.linalg.norm(T[va, :3, 3], axis=-1).mean() * 1000),
                       "mean_rot_deg": float(ang_np(T[va, :3, :3], np.eye(3)[None]).mean()),
                       "spread_over_frames_pos_mm": float(np.linalg.norm(T[va, :3, 3] - T[va, :3, 3].mean(0), axis=-1).mean() * 1000)}
    return out


def roundtrip(offs, Ds):
    worst = 0.0
    for T, D in zip(offs, Ds):
        T = T.double().cpu().numpy()
        E = D.Wk_np @ np.linalg.inv(T)
        worst = max(worst, float(np.abs(E @ T - D.Wk_np).max()))
    return worst


# ----------------------------------------------------------------------------- main
CONFIGS = [
    {"name": "c0_base", "cov": 0.5, "zero": 1.0, "smooth": 10.0, "l2": 0.05},
    {"name": "c1_nocov", "cov": 0.0, "zero": 1.0, "smooth": 10.0, "l2": 0.05},
    {"name": "c2_zero0.3", "cov": 0.5, "zero": 0.3, "smooth": 10.0, "l2": 0.05},
    {"name": "c3_smooth50", "cov": 0.5, "zero": 1.0, "smooth": 50.0, "l2": 0.05},
    {"name": "c4_smooth2_cov1", "cov": 1.0, "zero": 1.0, "smooth": 2.0, "l2": 0.05},
    {"name": "c5_base_init_fid", "cov": 0.5, "zero": 1.0, "smooth": 10.0, "l2": 0.05, "init": "f_id"},
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--configs", default="all")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "train_log.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    Ds = [HandData(h) for h in HANDS3]
    seg_report = {}
    for D in Ds:
        S = D.segs
        okS = [s for s in S if s["ok"]]
        seg_report[D.name] = {
            "n_ep": D.n_ep, "seg_ok": int(D.ok.sum()), "flag_short_grasp": int(sum(s.get("flag_short_grasp", False) for s in S)),
            "design_contact_rule_found": float(np.mean([s.get("t_contact_design", -1) >= 0 for s in S])),
            "any_contact_found": float(np.mean([s.get("t_contact_any", -1) >= 0 for s in S])),
            "tilt15_pour_found": float(np.mean([s.get("t_pour_tilt15", -1) >= 0 for s in S])),
            **{f"{k}_median": float(np.median([s[k] for s in okS])) for k in ("T", "t_g", "t_lift", "t_top", "zmax_cm",
                                                                             "tilt_max_after_lift_deg")},
            "grasp_len_median": float(np.median([s["t_lift"] - s["t_g"] for s in okS])),
            "episodes": S}
        log(f"[seg] {D.name}: ok {D.ok.sum()}/{D.n_ep} train {len(D.tr_eps)} val {len(D.va_eps)} " +
            json.dumps({k: v for k, v in seg_report[D.name].items() if k != "episodes"}))
    (OUT / "segmentation.json").write_text(json.dumps(seg_report, indent=1))

    results = {"config": {"W_ROT_m_per_rad": W_ROT, "sigma_rad": SIGMA, "bins_per_segment": NB, "segments": SEGS,
                          "steps": a.steps, "split": "eps 0-39 train / 40-49 val per hand",
                          "segmentation": "t_lift: kettle z > z(frame10)+1cm for 5 steps; t_g: earliest t<=t_lift from which "
                                          "the kettle-frame wrist stays within 20 mm / 10 deg of its t_lift pose; t_top: "
                                          "kettle z >= 0.9 max; segments approach/grasp/lift/hold"},
               "segmentation_summary": {h: {k: v for k, v in r.items() if k != "episodes"} for h, r in seg_report.items()}}
    methods = {}

    def evaluate(name, offs):
        Es = ee_common_all(offs, Ds)
        r = {"val": fisher(Es, Ds, "val"), "train": fisher(Es, Ds, "train"), "offset_mag_val": offset_mag(offs, Ds),
             "roundtrip_max_abs": roundtrip(offs, Ds)}
        r["val_score"] = score(r["val"])
        methods[name] = r
        v = r["val"]
        log(f"[eval] {name:24s} val FRpos {v['fr_pos']['overall_median']:.3f} FRrot {v['fr_rot']['overall_median']:.3f} "
            f"pair {v['pair_mm']['overall_median']:.1f}mm/{v['pair_deg']['overall_median']:.1f}deg "
            f"within {v['within_mm']['overall_median']:.1f}mm/{v['within_deg']['overall_median']:.1f}deg "
            f"| train FRpos {r['train']['fr_pos']['overall_median']:.3f} FRrot {r['train']['fr_rot']['overall_median']:.3f} "
            f"| rt {r['roundtrip_max_abs']:.1e} ({time.time() - t0:.0f}s)")
        return r

    evaluate("raw", [all_offsets("raw", None, D, i) for i, D in enumerate(Ds)])
    g2 = FNet().to(DEV)
    g2.load_state_dict(torch.load(G2F, map_location=DEV))
    g2.eval()
    evaluate("g2f", [all_offsets("f", g2, D, i) for i, D in enumerate(Ds)])
    # F: pure T_world->F and with shared mean-zero normaliser M (se3 mean of inv(T_P->F) over train frames, 3 hands)
    TPF = [all_offsets("F_TPF", None, D, i) for i, D in enumerate(Ds)]
    evaluate("F_pure", [inv(T) for T in TPF])
    tr_frames = [torch.tensor(np.concatenate([np.arange(D.off[e], D.off[e + 1]) for e in D.tr_eps]), device=DEV) for D in Ds]
    invs = torch.cat([inv(T[f]) for T, f in zip(TPF, tr_frames)]).double()
    from pd2.train_f import proj_so3
    M = make_T(proj_so3(invs[:, :3, :3].mean(0)), invs[:, :3, 3].mean(0)).float()
    evaluate("F", [inv(T @ M) for T in TPF])
    torch.save({"M": M.cpu(), "note": "EE_common(F) = EE_a @ T_P->F(q) @ M; shared constant over hands"}, OUT / "F_M.pt")

    cfgs = CONFIGS if a.configs == "all" else [c for c in CONFIGS if c["name"] in a.configs.split(",")]
    sweep, k_done = {}, {}
    for cfg in cfgs:
        log(f"[sweep] {cfg}")
        fm = train("f", Ds, cfg, log, a.steps)
        r = evaluate(f"f_task:{cfg['name']}", [all_offsets("f", fm, D, i) for i, D in enumerate(Ds)])
        torch.save({"state_dict": fm.state_dict(), "config": cfg, "hands": list(HANDS3)}, OUT / f"f_task_{cfg['name']}.pt")
        sweep[cfg["name"]] = {"f_task_val_score": r["val_score"]}
        kkey = (cfg["cov"], cfg["zero"], cfg["l2"])
        if kkey not in k_done:
            km = train("K", Ds, cfg, log, min(a.steps, 2000))
            rk = evaluate(f"K:{cfg['name']}", [all_offsets("K", km, D, i) for i, D in enumerate(Ds)])
            torch.save({"K": km.matrices().detach().cpu(), "hands": list(HANDS3), "config": cfg}, OUT / f"K_{cfg['name']}.pt")
            k_done[kkey] = cfg["name"]
            sweep[cfg["name"]]["K_val_score"] = rk["val_score"]
        results["sweep"], results["methods"] = sweep, methods
        (OUT / "analysis_p0a.json").write_text(json.dumps(results, indent=1))
    # selection on validation score = 0.5 (median FR_pos + median FR_rot)
    fn = min([m for m in methods if m.startswith("f_task:")], key=lambda m: methods[m]["val_score"])
    fs = min([m for m in methods if m.startswith("f_task:") and "init_fid" not in m], key=lambda m: methods[m]["val_score"])
    kn = min([m for m in methods if m.startswith("K:")], key=lambda m: methods[m]["val_score"])
    shutil.copy(OUT / f"f_task_{fn.split(':')[1]}.pt", OUT / "f_task_best.pt")
    shutil.copy(OUT / f"K_{kn.split(':')[1]}.pt", OUT / "K.pt")
    results["selected"] = {"f_task_best": fn, "f_task_best_scratch": fs, "K": kn,
                           "rule": "min val 0.5*(median FR_pos + median FR_rot) over 80 bins"}
    results["sweep"], results["methods"] = sweep, methods
    results["wall_s"] = time.time() - t0
    (OUT / "analysis_p0a.json").write_text(json.dumps(results, indent=1))
    log(f"[done] selected {results['selected']} wall {results['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
