"""codec v2 (cv2, 261115): fine-tune E2* toward a hand-invariant z. New file (arm_train.py is NOT modified; its objective is
reused verbatim for the E2* terms). Design: artifacts/codec_v2/DESIGN.md.

L = L_E2*(rec + lam*align + beta*KL; arm_train.objective) + lam_cyc*L_cyc + lam_pair*L_pair
  L_cyc : ordered pairs A->B with fingers(A) subset of fingers(B) (5->5, 4->4, 4->5; 5->4 skipped).
          z_A = mu_A(q_A); q~ = stopgrad(Dec_B(z_A)) + N(0, cyc_noise^2) (normalized joints, clamped to [-1,1]);
          L_cyc = mean ||mu_B(q~) - z_A||^2 / V,  V = mean ||z - mean z||^2 over the batch's native z (not detached).
  L_pair: synergy scenes (common prototypes open/fist/curl_thumb/index/middle/ring), same (i, j, r) for every hand,
          + N(0, 0.055) per hand (play_v1 synergy noise); L_pair = mean_s mean_h ||z_hs - mean_h z_hs||^2 / V.
Checkpoint format = arm_train (HandVAE per hand, config arm_mode "e2"), so load_arm / HoCodec / p3h_roundtrip work unchanged.
Container python, cwd /workspace/xhand/latent_codec_v1:
  python.sh -m xhand_latent.cv2_train --output artifacts/codec_v2/runs/<id> --lam-cyc 0.05 --lam-pair 0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .arm_pads import PadFK
from .arm_train import build, objective, summarize_diag

E2 = "artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt"
COMMON_PROTOS = ("open", "fist", "curl_thumb", "curl_index", "curl_middle", "curl_ring")


def finger_set(fk):
    return frozenset(i for i in range(5) if bool(fk.valid[i]))


def cycle_pairs(fks):
    return [(a, b) for a in fks for b in fks if a != b and finger_set(fks[a]) <= finger_set(fks[b])]


def zvar(z):
    return (z - z.mean(0)).square().sum(-1).mean()


def cycle_loss(codecs, mu, pairs, noise, gen=None):
    """mu: {hand: encoder mean of native batch}. Returns sum over pairs of mean squared z error (unnormalized)."""
    total = 0.
    for a, b in pairs:
        with torch.no_grad():
            qt = codecs[b].decoder(mu[a])
            if noise > 0:
                qt = (qt + noise * torch.randn(qt.shape, device=qt.device, generator=gen)).clamp(-1, 1)
        zh = codecs[b].encode(qt)[0]
        total = total + (zh - mu[a]).square().sum(-1).mean()
    return total / len(pairs)


class SynSampler:
    def __init__(self, play, hands, device, noise=0.055):
        self.hands, self.noise, self.device = hands, noise, device
        self.P = {}
        for h in hands:
            names = [str(s) for s in play[h]["prototype_names"]]
            self.P[h] = torch.tensor(np.stack([play[h]["prototypes"][names.index(n)] for n in COMMON_PROTOS]),
                                     dtype=torch.float32, device=device)

    def sample(self, S, gen=None, noise=None):
        noise = self.noise if noise is None else noise
        K = len(COMMON_PROTOS)
        i = torch.randint(K, (S,), device=self.device, generator=gen)
        j = torch.randint(K, (S,), device=self.device, generator=gen)
        r = torch.rand(S, 1, device=self.device, generator=gen)
        out = {}
        for h in self.hands:
            q = self.P[h][i] * r + self.P[h][j] * (1 - r)
            if noise > 0:
                q = q + noise * torch.randn(q.shape, device=self.device, generator=gen)
            out[h] = q.clamp(-0.995, 0.995)
        return out


class ClosureSampler:
    """Paired kinematic closure scenes (FK only). Per scene one closure value per SHARED finger (thumb/index/middle/ring);
    q_h = open_h + sum_f s_hf * (fist_h - open_h)[chain f] + N(0, noise); little (5-finger hands) follows ring
    (s_little = s_ring + N(0, 0.1)). Abduction joints stay open (prototype convention).
      mode 'joint': s_hf = u_f (equal joint fraction for every hand).
      mode 'func' : flexion endpoint = 0.98 (normalized upper limit) instead of the 0.65-fist; u_f is a FUNCTIONAL closure c = |tip_f(s) - tip_f(open)| / hand scale (palm-frame fingertip travel);
                    s_hf = c_hf^-1(u_f) per hand (monotone table, 201 points), u_f ~ U[0, min_h c_hf(1)] (common range)."""

    def __init__(self, play, fks, scales, device, mode="joint", noise=0.03, grid=201):
        self.mode, self.noise, self.device, self.hands = mode, noise, device, list(fks)
        self.open, self.delta, self.chain, self.table = {}, {}, {}, {}
        sg = np.linspace(0, 1, grid)
        self.cmax = np.full(4, np.inf)
        for h, fk in fks.items():
            names = [str(s) for s in play[h]["prototype_names"]]
            o = play[h]["prototypes"][names.index("open")]; c = play[h]["prototypes"][names.index("fist")]
            if mode == "func":  # sweep flexion joints to (almost) their upper limit so every hand reaches the common range
                c = np.where(np.abs(c - o) > 1e-6, 0.98, o)
            self.open[h], self.delta[h] = o.astype(np.float32), (c - o).astype(np.float32)
            self.chain[h] = {}
            for fi, f in enumerate(("thumb", "index", "middle", "ring", "little")):
                info = fk.spec["finger"][f]
                if not info:
                    continue
                idx = [fk.qidx[n] for n in info["chain_joints"]]
                self.chain[h][fi] = idx
                if fi < 4:
                    q = np.repeat(o[None], grid, 0).astype(np.float32)
                    q[:, idx] = o[idx] + sg[:, None] * (c - o)[idx]
                    with torch.no_grad():
                        tip = fk.cpu()(torch.tensor(q))[:, fi].numpy()
                    fk.to(device)
                    cf = np.maximum.accumulate(np.linalg.norm(tip - tip[0], axis=1) / scales[h])
                    self.table[(h, fi)] = cf
                    self.cmax[fi] = min(self.cmax[fi], cf[-1])
        self.sg = sg
        self.rng = np.random.default_rng(7)

    def s_for(self, h, fi, u):
        if self.mode == "joint":
            return u
        cf = self.table[(h, fi)]
        return np.interp(u * self.cmax[fi], cf, self.sg)  # u in [0,1] -> functional value in the common range

    def sample(self, S, gen=None, noise=None):
        noise = self.noise if noise is None else noise
        rng = self.rng if gen is None else np.random.default_rng(int(torch.randint(1 << 30, (1,), generator=gen, device=gen.device)))
        u = rng.uniform(0, 1, (S, 4))
        lit = rng.normal(0, 0.1, S)
        out = {}
        for h in self.hands:
            q = np.repeat(self.open[h][None], S, 0)
            for fi, idx in self.chain[h].items():
                s = self.s_for(h, fi, u[:, fi]) if fi < 4 else np.clip(self.s_for(h, 3, u[:, 3]) + lit, 0, 1)
                q[:, idx] += s[:, None] * self.delta[h][idx]
            if noise > 0:
                q = q + rng.normal(0, noise, q.shape)
            out[h] = torch.tensor(np.clip(q, -0.995, 0.995), dtype=torch.float32, device=self.device)
        return out


def fisher_loss(Z, Y, K, ridge=1e-3):
    """tr(Sw^-1 Sb) of hand labels on matched-content z (= linear discriminability; scale/whitening invariant)."""
    mu = Z.mean(0)
    Sw = Z.new_zeros(Z.shape[1], Z.shape[1]); Sb = torch.zeros_like(Sw)
    for k in range(K):
        Zk = Z[Y == k]
        if len(Zk) < 2:
            continue
        mk = Zk.mean(0)
        Sw = Sw + (Zk - mk).T @ (Zk - mk)
        Sb = Sb + len(Zk) * torch.outer(mk - mu, mk - mu)
    Sw = Sw / len(Z); Sb = Sb / len(Z)
    Sw = Sw + ridge * Sw.diagonal().mean() * torch.eye(len(Sw), device=Z.device)
    return torch.trace(torch.linalg.solve(Sw, Sb))


def pair_loss(codecs, qs):
    Z = torch.stack([codecs[h].encode(q)[0] for h, q in qs.items()], 1)  # (S, H, D)
    return (Z - Z.mean(1, keepdim=True)).square().sum(-1).mean()


class HandAdversary(torch.nn.Module):
    """Hand classifier on standardized z (BatchNorm, no affine = the probe's per-dim standardization); hidden=0 -> linear."""

    def __init__(self, dim, n_hands, hidden=0):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(dim, affine=False)
        self.net = (torch.nn.Linear(dim, n_hands) if hidden == 0 else
                    torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                                        torch.nn.ReLU(), torch.nn.Linear(hidden, n_hands)))

    def forward(self, z):
        return self.net(self.bn(z))


def adv_samples(codecs, hands, syn_q, mu, pairs, n_rt=128):
    """Matched-content z with encoder-hand labels: SYN scenes (every hand) + RT pairs (z_A -> label A,
    Enc_B(stopgrad Dec_B(z_A)) -> label B) over the cycle pairs. Gradients flow into the encoders."""
    Z, Y = [], []
    idx = {h: i for i, h in enumerate(hands)}
    if syn_q is not None:
        for h, q in syn_q.items():
            Z.append(codecs[h].encode(q)[0]); Y.append(torch.full((len(q),), idx[h], device=q.device))
    for a, b in pairs:
        za = mu[a][:n_rt]
        with torch.no_grad():
            qt = codecs[b].decoder(za)
        Z += [za, codecs[b].encode(qt)[0]]
        Y += [torch.full((len(za),), idx[a], device=za.device), torch.full((len(za),), idx[b], device=za.device)]
    return torch.cat(Z), torch.cat(Y)


def v2_terms(codecs, batch, pairs, syn_q, noise, gen=None):
    mu = {h: codecs[h].encode(q)[0] for h, q in batch.items()}
    V = zvar(torch.cat(list(mu.values())))
    cyc = cycle_loss(codecs, mu, pairs, noise, gen) / V
    pr = pair_loss(codecs, syn_q) / V if syn_q is not None else V.new_zeros(())
    return cyc, pr, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=E2)
    ap.add_argument("--play", default="artifacts/play_v1")
    ap.add_argument("--metadata", default="metadata")
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--pair-scenes", type=int, default=256)
    ap.add_argument("--lam-cyc", type=float, default=0.0)
    ap.add_argument("--lam-pair", type=float, default=0.0)
    ap.add_argument("--cyc-noise", type=float, default=0.02)
    ap.add_argument("--lam-adv", type=float, default=0.0, help="hand-confusion weight (adversary on matched-content z)")
    ap.add_argument("--adv-hidden", type=int, default=0, help="0 = linear adversary, else MLP width")
    ap.add_argument("--adv-lr", type=float, default=1e-3)
    ap.add_argument("--pair-data", choices=("syn", "cl_joint", "cl_func"), default="syn",
                    help="paired scenes for L_pair / adversary / Fisher: prototype synergies or kinematic closure sweeps")
    ap.add_argument("--lam-fisher", type=float, default=0.0, help="tr(Sw^-1 Sb) hand discriminability on matched z")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--validate-interval", type=int, default=500)
    ap.add_argument("--memory-fraction", type=float, default=0.12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
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
    syn = (SynSampler(play, hands, args.device) if args.pair_data == "syn" else
           ClosureSampler(play, fks, e2cfg["hand_scales_m"], args.device, mode=args.pair_data[3:]))
    gval = torch.Generator(device=args.device).manual_seed(123)
    syn_val = syn.sample(2048, gval)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for n in ("train.jsonl", "validation.jsonl"):
        if (out / n).exists():
            (out / n).unlink()
    config = {**e2cfg, "output": args.output, "steps": args.steps, "lr": args.lr, "seed": args.seed,
              "init": args.init, "init_info": {"path": args.init, "step": saved["step"]}, "arm_mode": "e2",
              "cv2": {"lam_cyc": args.lam_cyc, "lam_pair": args.lam_pair, "cyc_noise": args.cyc_noise,
                      "pair_scenes": args.pair_scenes, "warmup": args.warmup, "cycle_pairs": [f"{a}->{b}" for a, b in pairs],
                      "common_prototypes": list(COMMON_PROTOS), "pair_noise": syn.noise,
                      "lam_adv": args.lam_adv, "adv_hidden": args.adv_hidden, "adv_lr": args.adv_lr,
                      "pair_data": args.pair_data, "lam_fisher": args.lam_fisher,
                      "closure_common_range": (syn.cmax.tolist() if hasattr(syn, "cmax") else None)},
              "loss": "E2* (rec + lam*align + beta*KL) + lam_cyc*L_cyc/V + lam_pair*L_pair/V (cv2_train.py)",
              "optimizer": "fresh Adam"}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    opt = torch.optim.Adam(codecs.parameters(), lr=args.lr)
    adv = HandAdversary(e2cfg["latent_dim"], len(hands), args.adv_hidden).to(args.device) if args.lam_adv > 0 else None
    adv_opt = torch.optim.Adam(adv.parameters(), lr=args.adv_lr) if adv is not None else None
    logK = float(np.log(len(hands)))
    best, start, prev = float("inf"), time.monotonic(), time.monotonic()
    for step in range(0, args.steps + 1):
        if step > 0:
            codecs.train()
            w = min(1., step / args.warmup) if args.warmup else 1.
            batch = {h: q[torch.randint(len(q), (args.batch_size,), device=args.device)] for h, q in train.items()}
            loss, _, values, _, _ = objective(batch, codecs, fks, beta, lam, "e2")
            need_q = args.lam_pair > 0 or args.lam_adv > 0 or args.lam_fisher > 0
            syn_q = syn.sample(args.pair_scenes) if need_q else None
            if args.lam_cyc > 0 or args.lam_pair > 0:
                cyc, pr, V = v2_terms(codecs, batch, pairs if args.lam_cyc > 0 else pairs[:1], syn_q, args.cyc_noise)
            else:
                cyc = pr = V = loss.new_zeros(())
            total = loss + w * (args.lam_cyc * cyc + args.lam_pair * pr)
            conf = adv_acc = fis = loss.new_zeros(())
            if args.lam_fisher > 0:
                mu = {h: codecs[h].encode(q)[0] for h, q in batch.items()}
                Zf, Yf = adv_samples(codecs, hands, syn_q, mu, pairs)
                fis = fisher_loss(Zf, Yf, len(hands))
                total = total + w * args.lam_fisher * fis
            if adv is not None:
                mu = {h: codecs[h].encode(q)[0] for h, q in batch.items()}
                Za, Ya = adv_samples(codecs, hands, syn_q, mu, pairs)
                adv.train()
                lc = torch.nn.functional.cross_entropy(adv(Za.detach()), Ya)
                adv_opt.zero_grad(set_to_none=True); lc.backward(); adv_opt.step()
                logits = adv(Za)
                adv_acc = (logits.argmax(1) == Ya).float().mean()
                conf = -torch.log_softmax(logits, 1).mean() - logK  # CE to uniform minus its minimum (>= 0)
                total = total + w * args.lam_adv * conf
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
                       "cyc": float(cyc.detach()), "pair": float(pr.detach()), "V": float(V.detach()), "w": w,
                       "conf": float(conf.detach()), "adv_acc": float(adv_acc), "fisher": float(fis.detach()),
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
                cyc, pr, V = v2_terms(codecs, vb, pairs, syn_val, 0.0)
                total = float(score) + args.lam_cyc * float(cyc) + args.lam_pair * float(pr)
                extra = {}
                if args.lam_fisher > 0 or args.lam_adv > 0:
                    mu = {h: codecs[h].encode(q)[0] for h, q in vb.items()}
                    Zf, Yf = adv_samples(codecs, hands, syn_val, mu, pairs, n_rt=512)
                    extra["fisher"] = float(fisher_loss(Zf, Yf, len(hands)))
                    total += args.lam_fisher * extra["fisher"]
                if adv is not None and step > 0:
                    adv.eval()
                    mu = {h: codecs[h].encode(q)[0] for h, q in vb.items()}
                    Za, Ya = adv_samples(codecs, hands, syn_val, mu, pairs, n_rt=512)
                    lg = adv(Za)
                    vconf = float(-torch.log_softmax(lg, 1).mean() - logK)
                    extra = {"conf": vconf, "adv_acc": float((lg.argmax(1) == Ya).float().mean())}
                    total += args.lam_adv * vconf
                row = {"step": step, "total": total, **{k: float(v) for k, v in values.items()}, "cyc": float(cyc),
                       "pair": float(pr), "V": float(V), **extra, **summarize_diag(diag, fks)}
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
    print("[cv2_train] DONE", flush=True)


if __name__ == "__main__":
    main()
