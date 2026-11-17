"""codec v3 (cv3, 261116): losses that constrain EVERY z direction. New file; cv2_*.py and arm_train.py are NOT modified.
Design + pre-registered gates: artifacts/codec_v3/DESIGN.md.

Diagnosis of v2 (artifacts/codec_v2/viz/RESULTS_SEPARABILITY.md): L_cyc/V and L_pair/V were divided by the (non-detached)
total z variance V, so the model inflated one direction (PC1 86-93 %) and left the hand in low-variance tail directions.

L = L_E2*(rec + lam*align + beta*KL; arm_train.objective, unmodified)
    + lam_cyc * L_cyc + lam_pair * L_pair           (norm 'white': squared errors in WHITENED coordinates / D;
                                                     norm 'trace': v2 behaviour, / V)
    + lam_var * L_var + lam_cov * L_cov             (VICReg: per-dim std floor relative to the RMS std + off-diagonal cov)
    + lam_bt * L_bt                                 (Barlow Twins / CCA-style cross-correlation -> identity on paired z)
    + lam_coral * L_coral                           (per-hand mean + covariance matching on paired scenes, whitened)
    + lam_fisher * tr(Sw^-1 Sb)                     (v2 Fisher term, optional)
    + lam_null * L_null                             (A4: z content in each decoder's null space pulled to the pooled mean)
Whitening W = (C + ridge*mean_eig*I)^(-1/2) (ZCA) of the pooled native-batch z covariance, running EMA, DETACHED: the
alignment losses cannot be lowered by rescaling or by inflating one direction, and every direction weighs the same.
Pair data: 'cl_func' (v2 functional closure pairs), 'syn' (v2 prototype synergies), 'tax' (taxonomy IK pairs, npz from
cv3_taxonomy.py, per-hand validity masks, train split only), 'cl_func+tax' (half/half).
Checkpoint format = arm_train (HandVAE per hand, arm_mode e2) -> load_arm / HoCodec / p3h_roundtrip / arm_prepare work unchanged.
Container python, cwd /workspace/xhand/latent_codec_v1:
  python.sh -m xhand_latent.cv3_train --output artifacts/codec_v3/runs/<id> --norm white --lam-cyc 0.05 --lam-pair 0.05
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import arm_pads
from .arm_pads import TAU, PadFK
from .arm_train import build, objective, summarize_diag
from .cv2_train import E2, ClosureSampler, SynSampler, cycle_pairs, fisher_loss


# ---------------- memory-lean contacts (identical math; avoids the (B,F,K,K,3) difference tensor) ----------------
def contacts_cdist(pts, nrm, fingers, tau=TAU):
    thumb, tn = pts[:, 0], nrm[:, 0]
    f_pts, f_nrm = pts[:, fingers], nrm[:, fingers]
    B, F, K, _ = f_pts.shape
    dist = torch.cdist(f_pts.reshape(B, F * K, 3), thumb, compute_mode="donot_use_mm_for_euclid_dist").reshape(B, F, K, -1)
    a = torch.softmax(-dist.min(-1).values / tau, -1)
    b = torch.softmax(-dist.min(-2).values / tau, -1)
    cf = (a[..., None] * f_pts).sum(-2)
    nf = torch.nn.functional.normalize((a[..., None] * f_nrm).sum(-2), dim=-1, eps=1e-8)
    ct = (b[..., None] * thumb[:, None]).sum(-2)
    nt = torch.nn.functional.normalize((b[..., None] * tn[:, None]).sum(-2), dim=-1, eps=1e-8)
    return cf, ct, nf, nt


def patch_contacts():
    arm_pads.contacts = contacts_cdist


# ---------------- whitening ----------------
class Whitener:
    """Running (EMA) pooled z covariance -> detached ZCA whitening matrix W. sq(d) = ||d W||^2 / D."""

    def __init__(self, dim, momentum=0.9, ridge=1e-3):
        self.D, self.mom, self.ridge = dim, momentum, ridge
        self.C = None
        self.W = None

    @torch.no_grad()
    def update(self, Z):
        Z = Z.detach().double()
        C = torch.cov(Z.T, correction=0)
        self.C = C if self.C is None else self.mom * self.C + (1 - self.mom) * C
        e, V = torch.linalg.eigh(self.C)
        e = e.clamp_min(0) + self.ridge * e.clamp_min(0).mean()
        self.W = ((V * e.rsqrt()) @ V.T).float()
        self.eig = e.flip(0)

    def apply(self, Z):
        return Z @ self.W

    def sq(self, d):
        return (d @ self.W).square().sum(-1) / self.D

    def state(self):
        e = self.eig
        return {"pc1_share": float(e[0] / e.sum()), "participation_ratio": float(e.sum() ** 2 / e.square().sum())}


def zvar(z):
    return (z - z.mean(0)).square().sum(-1).mean()


# ---------------- loss terms ----------------
def cycle_terms(codecs, mu, pairs, noise, dist):
    """-> mean over pairs of mean dist(z^_B - z_A), list of (z_A, z^_B) for BT."""
    total, tuples = 0., []
    for a, b in pairs:
        with torch.no_grad():
            qt = codecs[b].decoder(mu[a])
            if noise > 0:
                qt = (qt + noise * torch.randn_like(qt)).clamp(-1, 1)
        zh = codecs[b].encode(qt)[0]
        total = total + dist(zh - mu[a]).mean()
        tuples.append((mu[a], zh))
    return total / max(1, len(pairs)), tuples


def encode_scenes(codecs, hands, qs):
    return torch.stack([codecs[h].encode(qs[h])[0] for h in hands], 1)  # (S, H, D)


def pair_term(Z, M, dist):
    """Z (S,H,D), M (S,H) float mask; distance of each valid hand's z to the masked scene mean."""
    w = M / M.sum(1, keepdim=True).clamp_min(1)
    zbar = (Z * w[..., None]).sum(1, keepdim=True)
    d = dist((Z - zbar).reshape(-1, Z.shape[-1])).reshape(M.shape)
    return (d * M).sum() / M.sum().clamp_min(1)


def scene_tuples(Z, M, min_rows=32):
    out = []
    H = Z.shape[1]
    for i in range(H):
        for j in range(i + 1, H):
            m = (M[:, i] > 0) & (M[:, j] > 0)
            if int(m.sum()) >= min_rows:
                out.append((Z[m, i], Z[m, j]))
    return out


def vicreg_terms(Z, gamma, eps=1e-4):
    """Scale-free VICReg on pooled z: hinge on std_d / RMS-std (detached) and off-diagonal covariance / mean var."""
    Zc = Z - Z.mean(0)
    var = Zc.square().mean(0)
    mv = var.mean().detach()
    std = torch.sqrt(var + eps * mv)
    lv = torch.relu(gamma - std / mv.sqrt()).mean()
    C = Zc.T @ Zc / len(Z) / mv
    off = C - torch.diag(torch.diagonal(C))
    lc = off.square().sum() / Z.shape[1]
    return lv, lc


def vicreg_perhand_var(Zs, gamma, ref_mv, eps=1e-4):
    """per-hand std floor relative to the POOLED RMS std (forces each hand to use every direction)."""
    out = 0.
    for Z in Zs:
        var = (Z - Z.mean(0)).square().mean(0)
        out = out + torch.relu(gamma - torch.sqrt(var + eps * ref_mv) / ref_mv.sqrt()).mean()
    return out / len(Zs)


def bt_term(tuples, lam_off=None, eps=1e-5):
    """Barlow Twins: per-view batch standardization, cross-correlation C -> I. (sum(1-Cii)^2 + lam_off sum Cij^2)/D."""
    tot = 0.
    for A, B in tuples:
        D = A.shape[1]
        lo = 1.0 / D if lam_off is None else lam_off
        As = (A - A.mean(0)) / (A.std(0) + eps)
        Bs = (B - B.mean(0)) / (B.std(0) + eps)
        C = As.T @ Bs / len(A)
        on = (1 - torch.diagonal(C)).square().sum()
        off = (C - torch.diag(torch.diagonal(C))).square().sum()
        tot = tot + (on + lo * off) / D
    return tot / max(1, len(tuples))


def coral_term(Z, M, wh):
    """Per-hand mean + covariance matching on PAIRED scenes in whitened coordinates (only rows valid for every hand used)."""
    keep = M.sum(1) == M.shape[1]
    if int(keep.sum()) < 32:
        keep = torch.ones_like(keep)
    Zw = wh.apply(Z[keep].reshape(-1, Z.shape[-1])).reshape(int(keep.sum()), Z.shape[1], -1)
    mus = Zw.mean(0)  # (H, D)
    mbar = mus.mean(0)
    covs = torch.stack([torch.cov(Zw[:, h].T, correction=0) for h in range(Zw.shape[1])])
    cbar = covs.mean(0)
    D = Z.shape[-1]
    return ((mus - mbar).square().sum(-1) / D).mean() + ((covs - cbar).square().sum((-1, -2)) / D).mean()


def null_term(codecs, hands, mu, n=64, tol=1e-3):
    """Decoder-null-space penalty (A4): for each hand, z directions its decoder ignores at that point (null space of
    dDec_h/dz, detached basis) should carry no hand-specific content -> squared deviation of z from the POOLED batch mean
    inside that null space, / pooled mean variance (detached). Free for reconstruction by construction."""
    from torch.func import jacrev, vmap
    pool = torch.cat([mu[h] for h in hands])
    m = pool.mean(0).detach()
    mv = (pool - pool.mean(0)).square().sum(-1).mean().detach() / pool.shape[1]
    tot = 0.
    for h in hands:
        z = mu[h][:n]
        dec = codecs[h].decoder
        with torch.no_grad():
            J = vmap(jacrev(lambda zz: dec(zz[None])[0]))(z.detach())  # (n, dof, D)
            U, S, Vh = torch.linalg.svd(J, full_matrices=True)
            r = (S > tol * S.amax(-1, keepdim=True)).sum(-1)  # (n,)
            D = z.shape[1]
            mask = (torch.arange(D, device=z.device)[None] >= r[:, None]).float()  # null rows of Vh
            Nb = Vh * mask[..., None]  # (n, D, D), rows = null basis vectors
        proj = torch.einsum("nij,nj->ni", Nb, z - m)
        tot = tot + proj.square().sum(-1).mean() / mv
    return tot / len(hands) / pool.shape[1]


def matched_samples(Z, M, mu, pairs, codecs, n_rt=128):
    """(z, hand-label) on matched content: valid paired scenes + cycle pairs (for the optional Fisher term)."""
    H = Z.shape[1]
    Zs, Ys = [], []
    for h in range(H):
        m = M[:, h] > 0
        Zs.append(Z[m, h]); Ys.append(torch.full((int(m.sum()),), h, device=Z.device))
    return Zs, Ys


# ---------------- taxonomy pairs ----------------
class TaxSampler:
    """Paired taxonomy postures (cv3_taxonomy.py npz): q_<hand> (N, dof) normalized, valid (N, H) in `hands` order,
    split (N,) 0 = train, 1 = held-out sample, 2 = held-out class. Samples train rows, class-balanced."""

    def __init__(self, path, hands, device, noise=0.0):
        d = np.load(path, allow_pickle=True)
        th = [str(h) for h in d["hands"]]
        self.hands, self.device, self.noise = hands, device, noise
        tr = d["split"] == 0
        self.q = {h: torch.tensor(d[f"q_{h}"][tr], dtype=torch.float32, device=device) for h in hands}
        self.M = torch.tensor(d["valid"][tr][:, [th.index(h) for h in hands]], dtype=torch.float32, device=device)
        cls = d["class_id"][tr]
        self.groups = [torch.tensor(np.where(cls == c)[0], device=device) for c in np.unique(cls)]
        self.rng = np.random.default_rng(11)
        self.n = int(tr.sum())

    def sample(self, S, gen=None):
        ci = self.rng.integers(len(self.groups), size=S)
        idx = torch.stack([self.groups[c][int(self.rng.integers(len(self.groups[c])))] for c in ci])
        out = {h: self.q[h][idx] for h in self.hands}
        if self.noise > 0:
            out = {h: (q + self.noise * torch.randn_like(q)).clamp(-0.995, 0.995) for h, q in out.items()}
        return out, self.M[idx]


class PairData:
    def __init__(self, kind, play, fks, scales, hands, device, tax_path=None):
        self.kind, self.hands = kind, hands
        self.src = []
        for k in kind.split("+"):
            if k == "syn":
                self.src.append(("syn", SynSampler(play, hands, device)))
            elif k in ("cl_func", "cl_joint"):
                self.src.append((k, ClosureSampler(play, fks, scales, device, mode=k[3:])))
            elif k == "tax":
                self.src.append(("tax", TaxSampler(tax_path, hands, device)))
            else:
                raise ValueError(k)

    def sample(self, S, gen=None):
        qs, Ms = {h: [] for h in self.hands}, []
        n = S // len(self.src)
        for i, (k, smp) in enumerate(self.src):
            s = S - n * (len(self.src) - 1) if i == 0 else n
            if k == "tax":
                q, M = smp.sample(s, gen)
            else:
                q = smp.sample(s, gen)
                M = torch.ones(s, len(self.hands), device=q[self.hands[0]].device)
            for h in self.hands:
                qs[h].append(q[h])
            Ms.append(M)
        return {h: torch.cat(v) for h, v in qs.items()}, torch.cat(Ms)

    def info(self):
        out = {}
        for k, smp in self.src:
            if hasattr(smp, "cmax"):
                out[k] = {"closure_common_range": smp.cmax.tolist()}
            if k == "tax":
                out[k] = {"n_train_rows": smp.n, "n_classes": len(smp.groups)}
        return out


# ---------------- all v3 terms ----------------
def v3_terms(args, codecs, hands, batch, pairs, pd_q, pd_M, wh, train=True):
    mu = {h: codecs[h].encode(q)[0] for h, q in batch.items()}
    Zpool = torch.cat([mu[h] for h in hands])
    if train:
        wh.update(Zpool)
    V = zvar(Zpool)
    if args.norm == "white":
        dist = wh.sq
    else:
        dist = lambda d: d.square().sum(-1) / V
    T = {}
    zero = V.new_zeros(())
    need_cyc = args.lam_cyc > 0 or args.lam_bt > 0
    if need_cyc:
        T["cyc"], cyc_tuples = cycle_terms(codecs, mu, pairs, args.cyc_noise if train else 0.0, dist)
    else:
        T["cyc"], cyc_tuples = zero, []
    Z = encode_scenes(codecs, hands, pd_q) if pd_q is not None else None
    T["pair"] = pair_term(Z, pd_M, dist) if Z is not None and args.lam_pair > 0 else zero
    if args.lam_var > 0 or args.lam_cov > 0:
        T["var"], T["cov"] = vicreg_terms(Zpool, args.vic_gamma)
        if args.vic_perhand:
            mv = (Zpool - Zpool.mean(0)).square().mean().detach()
            T["var"] = 0.5 * (T["var"] + vicreg_perhand_var([mu[h] for h in hands], args.vic_gamma, mv))
    else:
        T["var"] = T["cov"] = zero
    if args.lam_bt > 0:
        tuples = cyc_tuples + (scene_tuples(Z, pd_M) if Z is not None else [])
        T["bt"] = bt_term(tuples, args.bt_off)
    else:
        T["bt"] = zero
    T["coral"] = coral_term(Z, pd_M, wh) if args.lam_coral > 0 and Z is not None else zero
    T["null"] = null_term(codecs, hands, mu) if args.lam_null > 0 else zero
    if args.lam_fisher > 0 and Z is not None:
        Zs, Ys = matched_samples(Z, pd_M, mu, pairs, codecs)
        for a, b in pairs:  # RT pairs as in v2
            za = mu[a][:128]
            with torch.no_grad():
                qt = codecs[b].decoder(za)
            Zs += [za, codecs[b].encode(qt)[0]]
            Ys += [torch.full((len(za),), hands.index(a), device=za.device), torch.full((len(za),), hands.index(b), device=za.device)]
        T["fisher"] = fisher_loss(torch.cat(Zs), torch.cat(Ys), len(hands))
    else:
        T["fisher"] = zero
    lam = {"cyc": args.lam_cyc, "pair": args.lam_pair, "var": args.lam_var, "cov": args.lam_cov, "bt": args.lam_bt,
           "coral": args.lam_coral, "fisher": args.lam_fisher, "null": args.lam_null}
    extra = sum(lam[k] * T[k] for k in lam)
    T["V"] = V
    return extra, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=E2)
    ap.add_argument("--play", default="artifacts/play_v1")
    ap.add_argument("--metadata", default="metadata")
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--pair-scenes", type=int, default=256)
    ap.add_argument("--pair-data", default="cl_func", help="'+'-joined: syn, cl_func, cl_joint, tax")
    ap.add_argument("--tax", default="artifacts/codec_v3/data/tax_pairs.npz")
    ap.add_argument("--norm", choices=("white", "trace"), default="white")
    ap.add_argument("--lam-cyc", type=float, default=0.05)
    ap.add_argument("--lam-pair", type=float, default=0.05)
    ap.add_argument("--cyc-noise", type=float, default=0.02)
    ap.add_argument("--lam-var", type=float, default=0.0)
    ap.add_argument("--lam-cov", type=float, default=0.0)
    ap.add_argument("--vic-gamma", type=float, default=0.5)
    ap.add_argument("--vic-perhand", action="store_true")
    ap.add_argument("--lam-bt", type=float, default=0.0)
    ap.add_argument("--bt-off", type=float, default=None)
    ap.add_argument("--lam-coral", type=float, default=0.0)
    ap.add_argument("--lam-fisher", type=float, default=0.0)
    ap.add_argument("--lam-null", type=float, default=0.0, help="A4 decoder-null-space penalty (null_term)")
    ap.add_argument("--white-mom", type=float, default=0.9)
    ap.add_argument("--white-ridge", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--validate-interval", type=int, default=500)
    ap.add_argument("--memory-fraction", type=float, default=0.12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe-only", action="store_true", help="print step-0 term values and exit (calibration)")
    args = ap.parse_args()
    patch_contacts()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    saved = torch.load(args.init, map_location=args.device, weights_only=False)
    e2cfg = saved["config"]
    hands = e2cfg["hands"].split(",")
    beta, lam = e2cfg["beta"], e2cfg["lam"]
    fks = {h: PadFK(h, args.metadata).to(args.device) for h in hands}
    codecs = build(fks, "e2", e2cfg["latent_dim"]).to(args.device)
    codecs.load_state_dict(saved["state_dict"])
    play, train, val = {}, {}, {}
    for h in hands:
        play[h] = np.load(Path(args.play) / f"{h}.npz", allow_pickle=True)
        if list(play[h]["joint_names"]) != fks[h].joint_names:
            raise ValueError(f"joint order mismatch {h}")
        train[h] = torch.tensor(play[h]["train"], device=args.device)
        val[h] = torch.tensor(play[h]["val"], device=args.device)
    pairs = cycle_pairs(fks)
    pdata = PairData(args.pair_data, play, fks, e2cfg["hand_scales_m"], hands, args.device, args.tax)
    need_pd = any(x > 0 for x in (args.lam_pair, args.lam_bt, args.lam_coral, args.lam_fisher))
    gval = torch.Generator(device=args.device).manual_seed(123)
    pd_val = pdata.sample(2048, gval)
    wh = Whitener(e2cfg["latent_dim"], args.white_mom, args.white_ridge)
    with torch.no_grad():
        wh.update(torch.cat([codecs[h].encode(q[:2048])[0] for h, q in val.items()]))
    if args.probe_only:
        codecs.eval()
        with torch.no_grad():
            vb = {h: q[:2048] for h, q in val.items()}
            score, _, values, _, _ = objective(vb, codecs, fks, beta, lam, "e2", sample=False)
            extra, T = v3_terms(args, codecs, hands, vb, pairs, *pd_val, wh, train=False)
        print(json.dumps({"E2_loss": float(score), "rec": float(values["rec"]), "extra": float(extra),
                          **{k: float(v) for k, v in T.items()}, **wh.state()}), flush=True)
        if args.device == "cuda":
            print("peak_MiB", torch.cuda.max_memory_allocated() / 2**20)
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for n in ("train.jsonl", "validation.jsonl"):
        if (out / n).exists():
            (out / n).unlink()
    config = {**e2cfg, "output": args.output, "steps": args.steps, "lr": args.lr, "seed": args.seed,
              "init": args.init, "init_info": {"path": args.init, "step": saved["step"]}, "arm_mode": "e2",
              "cv3": {**{k: v for k, v in vars(args).items() if k not in ("output", "init", "play", "metadata")},
                      "cycle_pairs": [f"{a}->{b}" for a, b in pairs], "pair_info": pdata.info()},
              "loss": "E2* + lam_cyc*L_cyc + lam_pair*L_pair (whitened or trace) + VICReg + BT + CORAL + Fisher (cv3_train.py)",
              "optimizer": "fresh Adam"}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    opt = torch.optim.Adam(codecs.parameters(), lr=args.lr)
    best, start, prev = float("inf"), time.monotonic(), time.monotonic()
    for step in range(0, args.steps + 1):
        if step > 0:
            codecs.train()
            w = min(1., step / args.warmup) if args.warmup else 1.
            batch = {h: q[torch.randint(len(q), (args.batch_size,), device=args.device)] for h, q in train.items()}
            loss, _, values, _, _ = objective(batch, codecs, fks, beta, lam, "e2")
            pd_q, pd_M = pdata.sample(args.pair_scenes) if need_pd else (None, None)
            extra, T = v3_terms(args, codecs, hands, batch, pairs, pd_q, pd_M, wh, train=True)
            total = loss + w * extra
            if not torch.isfinite(total):
                raise RuntimeError(f"nonfinite at {step}")
            opt.zero_grad(set_to_none=True)
            total.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(codecs.parameters(), 10))
            opt.step()
            if step == 1 or step % args.log_interval == 0:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                now = time.monotonic()
                row = {"step": step, "total": float(total.detach()), **{k: float(v.detach()) for k, v in values.items()},
                       **{k: float(v.detach()) for k, v in T.items()}, "w": w, **wh.state(),
                       "grad_norm": gn, "wall_s": now - start, "interval_s": now - prev,
                       "peak_MiB": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None}
                prev = now
                with (out / "train.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        if step % args.validate_interval == 0 or step == args.steps:
            codecs.eval()
            with torch.no_grad():
                vb = {h: q[:2048] for h, q in val.items()}
                score, _, values, stats, diag = objective(vb, codecs, fks, beta, lam, "e2", sample=False, diagnostics=True)
                extra, T = v3_terms(args, codecs, hands, vb, pairs, *pd_val, wh, train=False)
                total = float(score) + float(extra)
                row = {"step": step, "total": total, **{k: float(v) for k, v in values.items()},
                       **{k: float(v) for k, v in T.items()}, **wh.state(), **summarize_diag(diag, fks)}
                row["joint_rmse_rad"] = {h: float((fks[h].denormalize(a) - fks[h].denormalize(b)).square().mean().sqrt())
                                         for h, (a, b) in stats.items()}
            with (out / "validation.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({"validation": row}), flush=True)
            if step == 0:
                continue
            payload = {"config": config, "step": step, "state_dict": codecs.state_dict(), "validation": row}
            torch.save(payload, out / "last.tmp"); os.replace(out / "last.tmp", out / "last.pt")
            if total < best:
                best = total
                torch.save(payload, out / "best.tmp"); os.replace(out / "best.tmp", out / "best.pt")
    (out / "done.json").write_text(json.dumps({"steps": args.steps, "wall_s": time.monotonic() - start, "best_total": best}))
    print("[cv3_train] DONE", flush=True)


if __name__ == "__main__":
    main()
