"""cv3alt (261116): fine-tune E2* toward a hand-invariant z with dense PAIRED functional posture data (D1 playik / D2 human /
D3 sobol / mix). New file; arm_train.objective is reused verbatim for the E2* terms. Design: artifacts/codec_v3alt/DESIGN.md §3.

L = L_E2*(rec + lam*align + beta*KL) + lam_w*L_wpair + lam_v*L_var + lam_c*L_cov + lam_k*L_coral
  L_wpair: whitened paired MSE on matched scenes: mean over valid (s,h) of (z_hs - zbar_s)^T Sigma^-1 (z_hs - zbar_s) / D,
           Sigma = pooled covariance of the pair batch (all hands, with gradient) + ridge*mean-eig*I  (linear-map invariant)
  L_var  : VICReg variance hinge mean_d relu(1 - std_d/gamma_d), gamma_d = std of E2* z on play val (fixed)
  L_cov  : VICReg covariance, sum of squared off-diagonal correlations / D
  L_coral: per-hand covariance in the pooled-whitened space vs their mean, |C_h - Cbar|_F^2 / D^2
Checkpoint format = arm_train (HandVAE per hand, arm_mode "e2"): load_arm / HoCodec / p3h_roundtrip / cv2_measure work unchanged.
Container python, cwd /workspace/xhand/latent_codec_v1:
  python.sh -m xhand_latent.cv3alt_train --output artifacts/codec_v3alt/runs/<id> --pairs artifacts/codec_v3alt/data/playik.npz
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


def contacts_cdist(pts, nrm, fingers, tau=TAU):
    """Memory-lean copy of arm_pads.contacts (identical math; copied from the v3 agent's cv3_train.py, read-only):
    avoids the (B,F,K,K,3) difference tensor. Patched in with --lean-contacts."""
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

E2 = "artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt"
HANDS5 = ("allegro", "leap", "shadow", "wuji", "sharpa")


class PairData:
    """npz with Q_<hand>_<split> (N,dof) normalized joints, pmask_<split> (N,5 in HANDS5 order) bool (pair-loss mask)."""

    def __init__(self, paths, hands, device, split="train"):
        self.sets = []
        for p in paths:
            d = np.load(p, allow_pickle=True)
            Q = {h: torch.tensor(d[f"Q_{h}_{split}"], device=device) for h in hands}
            order = [str(s) for s in d["hands"]]
            M = torch.tensor(d[f"pmask_{split}"][:, [order.index(h) for h in hands]], device=device)
            keep = M.sum(1) >= 2
            self.sets.append(({h: q[keep] for h, q in Q.items()}, M[keep]))
        self.device = device

    def sample(self, S, gen=None):
        per = [S // len(self.sets) + (1 if i < S % len(self.sets) else 0) for i in range(len(self.sets))]
        Qs, Ms = {}, []
        for (Q, M), n in zip(self.sets, per):
            idx = torch.randint(len(M), (n,), device=self.device, generator=gen)
            for h in Q:
                Qs.setdefault(h, []).append(Q[h][idx])
            Ms.append(M[idx])
        return {h: torch.cat(v) for h, v in Qs.items()}, torch.cat(Ms)

    def all(self):
        Qs = {h: torch.cat([Q[h] for Q, _ in self.sets]) for h in self.sets[0][0]}
        return Qs, torch.cat([M for _, M in self.sets])


def pooled_cov(Zp, ridge):
    C = torch.cov(Zp.T)
    return C + ridge * C.diagonal().mean() * torch.eye(len(C), device=Zp.device, dtype=Zp.dtype)


def pair_terms(codecs, hands, Q, M, gamma, ridge=1e-4):
    Z = torch.stack([codecs[h].encode(Q[h])[0] for h in hands], 1)  # (S,H,D)
    Mf = M.to(Z.dtype)
    S, H, D = Z.shape
    zbar = (Z * Mf[..., None]).sum(1) / Mf.sum(1, keepdim=True)
    dev = (Z - zbar[:, None]) * Mf[..., None]
    Zp = Z[M]
    C = pooled_cov(Zp, ridge)
    L = torch.linalg.cholesky(C)
    wdev = torch.linalg.solve_triangular(L, dev.reshape(-1, D).T, upper=False).T  # (S*H, D)
    wpair = wdev.square().sum(-1).sum() / Mf.sum() / D
    euc = dev.square().sum(-1).sum() / Mf.sum() / ((Zp - Zp.mean(0)).square().sum(-1).mean())
    std = Zp.std(0)
    var = torch.relu(1 - std / gamma).mean()
    corr = torch.corrcoef(Zp.T)
    cov = (corr - torch.diag(corr.diagonal())).square().sum() / D
    Ch = []
    mu = Zp.mean(0)
    for k in range(H):
        zk = Z[M[:, k], k]
        if len(zk) < D + 2:
            continue
        wk = torch.linalg.solve_triangular(L, (zk - zk.mean(0)).T, upper=False)
        Ch.append(wk @ wk.T / (len(zk) - 1))
    coral = (torch.stack(Ch) - torch.stack(Ch).mean(0)).square().sum((1, 2)).mean() / D ** 2 if len(Ch) > 1 else Z.new_zeros(())
    return {"wpair": wpair, "euc_ratio": euc, "var": var, "cov": cov, "coral": coral}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=E2)
    ap.add_argument("--play", default="artifacts/play_v1")
    ap.add_argument("--metadata", default="metadata")
    ap.add_argument("--pairs", required=True, help="comma list of pair npz (sampled equally)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--pair-scenes", type=int, default=256)
    ap.add_argument("--lam-w", type=float, default=0.05)
    ap.add_argument("--lam-v", type=float, default=0.02)
    ap.add_argument("--lam-c", type=float, default=0.01)
    ap.add_argument("--lam-k", type=float, default=0.02)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--validate-interval", type=int, default=500)
    ap.add_argument("--memory-fraction", type=float, default=0.12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lean-contacts", action="store_true")
    args = ap.parse_args()
    if args.lean_contacts:
        arm_pads.contacts = contacts_cdist
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
        train[h] = torch.tensor(play[h]["train"], device=args.device)
        val[h] = torch.tensor(play[h]["val"], device=args.device)
    paths = args.pairs.split(",")
    pdata = PairData(paths, hands, args.device, "train")
    pval = PairData(paths, hands, args.device, "val")
    gval = torch.Generator(device=args.device).manual_seed(123)
    pv_q, pv_m = pval.sample(2048, gval)
    with torch.no_grad():  # gamma = per-dim std of E2* z on play val (fixed target scale for the VICReg hinge)
        gamma = torch.cat([codecs[h].encode(val[h][:2048])[0] for h in hands]).std(0).clamp_min(1e-3)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for n in ("train.jsonl", "validation.jsonl"):
        if (out / n).exists():
            (out / n).unlink()
    config = {**e2cfg, "output": args.output, "steps": args.steps, "lr": args.lr, "seed": args.seed,
              "init": args.init, "init_info": {"path": args.init, "step": saved["step"]}, "arm_mode": "e2",
              "cv3alt": {"pairs": paths, "lam_w": args.lam_w, "lam_v": args.lam_v, "lam_c": args.lam_c, "lam_k": args.lam_k,
                         "ridge": args.ridge, "pair_scenes": args.pair_scenes, "warmup": args.warmup,
                         "n_train_scenes": [int(len(M)) for _, M in pdata.sets],
                         "n_val_scenes": [int(len(M)) for _, M in pval.sets]},
              "loss": "E2* + lam_w*L_wpair + lam_v*L_var + lam_c*L_cov + lam_k*L_coral (cv3alt_train.py)",
              "optimizer": "fresh Adam"}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    opt = torch.optim.Adam(codecs.parameters(), lr=args.lr)
    best, start, prev = float("inf"), time.monotonic(), time.monotonic()
    lam_vec = {"wpair": args.lam_w, "var": args.lam_v, "cov": args.lam_c, "coral": args.lam_k}
    for step in range(0, args.steps + 1):
        if step > 0:
            codecs.train()
            w = min(1., step / args.warmup) if args.warmup else 1.
            batch = {h: q[torch.randint(len(q), (args.batch_size,), device=args.device)] for h, q in train.items()}
            loss, _, values, _, _ = objective(batch, codecs, fks, beta, lam, "e2")
            Q, M = pdata.sample(args.pair_scenes)
            pt = pair_terms(codecs, hands, Q, M, gamma, args.ridge)
            extra = sum(lam_vec[k] * pt[k] for k in lam_vec)
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
                       **{k: float(v.detach()) for k, v in pt.items()}, "w": w, "grad_norm": gn,
                       "wall_s": now - start, "interval_s": now - prev,
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
                pt = pair_terms(codecs, hands, pv_q, pv_m, gamma, args.ridge)
                total = float(score) + sum(lam_vec[k] * float(pt[k]) for k in lam_vec)
                row = {"step": step, "total": total, **{k: float(v) for k, v in values.items()},
                       **{f"pv_{k}": float(v) for k, v in pt.items()}, **summarize_diag(diag, fks)}
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
    print("[cv3alt_train] DONE", flush=True)


if __name__ == "__main__":
    main()
