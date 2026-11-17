"""E1/E2 (Kabsch-in-loop) and E3 (learned wrist head) training for artifacts/arm_latent_v1.

loss E1/E2 = rec + lam*align + beta*KL, align = mean over ordered cross pairs of the
normalized weighted Kabsch residual^2 between pad correspondence sets.
loss E3    = rec + lam*(cross + self) + beta*KL, decoder also outputs T_{palm_B in F}.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .arm_pads import (F_FINGERS, PadFK, align_pair, common_fingers, contacts, f_frame, kabsch,
                       pair_weights, point_set, rot6d_to_matrix, weighted_residual2)
from .kinematics import FINGERS
from .model import HandVAE, mlp

TRANS_UNIT = 0.1  # metres per unit of the translation head


class WristVAE(nn.Module):
    """HandVAE whose decoder additionally outputs the palm pose in the source F frame."""

    def __init__(self, dof, latent_dim=32, head="shared"):
        super().__init__()
        self.dof, self.head_kind = dof, head
        self.encoder = mlp([dof, 64, 128, 64])
        self.mean = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        if head == "shared":  # one decoder MLP outputs [q, wrist]
            self.head = mlp([latent_dim, 64, 128, 64, dof + 9])
        else:  # separate q decoder and wrist head, both from z
            self.qdec = mlp([latent_dim, 64, 128, 64, dof])
            self.whead = mlp([latent_dim, 64, 128, 64, 9])

    def encode(self, q):
        hidden = self.encoder(q)
        return self.mean(hidden), self.logvar(hidden).clamp(-12, 6)

    def decode_full(self, z):
        out = self.head(z) if self.head_kind == "shared" else torch.cat([self.qdec(z), self.whead(z)], -1)
        d6 = out[:, self.dof + 3:] + torch.tensor([1., 0, 0, 0, 1, 0], device=z.device, dtype=z.dtype)
        return torch.tanh(out[:, :self.dof]), rot6d_to_matrix(d6), TRANS_UNIT * out[:, self.dof:self.dof + 3]

    def decoder(self, z):
        return self.decode_full(z)[0]

    def forward(self, q, sample=True):
        mean, logvar = self.encode(q)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        return self.decoder(z), z, mean, logvar


def build(fks, mode, latent_dim=32, head="shared"):
    if mode == "e3":
        return nn.ModuleDict({h: WristVAE(fk.dof, latent_dim, head) for h, fk in fks.items()})
    return nn.ModuleDict({h: HandVAE(fk.dof, latent_dim) for h, fk in fks.items()})


def decoder_params(codecs):
    return [p for m in codecs.values() for n, p in m.named_parameters() if n.startswith(("decoder", "head", "qdec", "whead"))]


def load_arm(path, metadata="metadata", device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    hands = cfg["hands"].split(",")
    fks = {h: PadFK(h, metadata).to(device) for h in hands}
    codecs = build(fks, cfg.get("arm_mode", "e1"), cfg["latent_dim"], cfg.get("e3_head", "shared")).to(device)
    codecs.load_state_dict(payload["state_dict"])
    codecs.eval()
    return codecs, fks, payload


def true_f_inverse(pads, fk):
    """Pose of palm in F: R = R_F^T, t = -R_F^T o (F from index/middle/ring contacts)."""
    cf, ct, _, _ = contacts(pads[0], pads[1], F_FINGERS)
    w = pair_weights(cf, ct)
    RF, o = f_frame(cf, ct, w)
    Rt = RF.transpose(-1, -2)
    return Rt, -torch.einsum("bij,bj->bi", Rt, o)


def objective(batch, codecs, fks, beta, lam, mode, sample=True, diagnostics=False):
    hands = list(batch)
    rec = kl = selfl = batch[hands[0]].new_zeros(())
    latents, pads, stats, finv = {}, {}, {}, {}
    for h, q in batch.items():
        mean, logvar = codecs[h].encode(q)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        latents[h] = z
        pads[h] = fks[h].pads(q)
        if mode == "e3":
            decoded, Rh, th = codecs[h].decode_full(z)
            with torch.no_grad():
                finv[h] = true_f_inverse(pads[h], fks[h])
            selfl = selfl + ((th - finv[h][1]).square().sum(-1) / fks[h].scale ** 2
                             + (Rh - finv[h][0]).square().sum((-1, -2))).mean()
        else:
            decoded = codecs[h].decoder(z)
        rec = rec + (decoded - q).square().mean()
        kl = kl + (-0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1)).mean()
        stats[h] = (decoded, q)
    rec, kl, selfl = rec / len(hands), kl / len(hands), selfl / len(hands)
    align, count, diag = rec.new_zeros(()), 0, {}
    for tgt in hands:
        sources = [s for s in hands if s != tgt]
        Z = torch.cat([latents[s] for s in sources])
        if mode == "e3":
            qd, Rd, td = codecs[tgt].decode_full(Z)
        else:
            qd = codecs[tgt].decoder(Z)
        pb_all = fks[tgt].pads(qd)
        n = batch[sources[0]].shape[0]
        for i, src in enumerate(sources):
            sl = slice(i * n, (i + 1) * n)
            pb = tuple(x[sl] for x in pb_all)
            if mode == "e3":
                fingers = common_fingers(fks[src], fks[tgt])
                ca = contacts(pads[src][0], pads[src][1], fingers)
                cb = contacts(pb[0], pb[1], fingers)
                w = pair_weights(ca[0], ca[1])
                pa, ww = point_set(*ca, w, fks[src].anchors)
                pbp, _ = point_set(*cb, w, fks[tgt].anchors)
                RA, tA = finv[src]
                paF = torch.einsum("bij,bnj->bni", RA, pa) + tA[:, None]
                r2, err = weighted_residual2(pbp, paF, ww, Rd[sl], td[sl])
                r2 = r2 / fks[src].scale ** 2
                if diagnostics:
                    with torch.no_grad():
                        Rk, tk = kabsch(pbp, pa, ww)
                        _, kerr = weighted_residual2(pbp, pa, ww, Rk, tk)
                        # gap between E3 palm pose and Kabsch-optimal palm pose, both in palm_A coords
                        RFa, oFa = RA.transpose(-1, -2), -torch.einsum("bji,bj->bi", RA, tA)
                        Rp = RFa @ Rd[sl]; tp = torch.einsum("bij,bj->bi", RFa, td[sl]) + oFa
                        ang = torch.arccos(((torch.einsum("bij,bij->b", Rp, Rk) - 1) / 2).clamp(-1, 1))
                    diag[f"{src}->{tgt}"] = {"err": err.detach(), "kabsch_err": kerr, "fingers": fingers, "w": w,
                                             "gap_pos_mm": (tp - tk).norm(dim=-1) * 1000, "gap_rot_deg": ang.rad2deg()}
            else:
                res = align_pair(fks[src], fks[tgt], pads[src], pb)
                r2 = res["r2"]
                if diagnostics:
                    diag[f"{src}->{tgt}"] = {"err": res["err"].detach(), "fingers": res["fingers"], "w": res["w"]}
            align = align + r2.mean()
            count += 1
    align = align / count
    terms = {"rec": rec, "align": lam * align, "kl": beta * kl}
    if mode == "e3":
        terms["self"] = lam * selfl
    total = sum(terms.values())
    values = {"loss": total, "rec": rec, "align_raw": align, "align": lam * align, "kl": kl, "lam": torch.tensor(lam)}
    if mode == "e3":
        values["self_raw"] = selfl
    return total, terms, values, stats, diag


def summarize_diag(diag, fks):
    """Per-finger (incl. thumb) residual mm, aggregated over pairs."""
    per = {f: [] for f in FINGERS}
    perw = {f: [] for f in FINGERS}
    kab = {f: [] for f in FINGERS}
    gap_p, gap_r = [], []
    for key, d in diag.items():
        F = len(d["fingers"])
        e = d["err"].norm(dim=-1) * 1000
        w = d["w"]
        for j, fi in enumerate(d["fingers"]):
            per[FINGERS[fi]].append(float(e[:, j].mean()))
            perw[FINGERS[fi]].append(float((e[:, j] * w[:, j]).sum() / w[:, j].sum()))
        per["thumb"].append(float(e[:, F:2 * F].mean()))
        perw["thumb"].append(float((e[:, F:2 * F] * w).sum() / w.sum()))
        if "kabsch_err" in d:
            ke = d["kabsch_err"].norm(dim=-1) * 1000
            for j, fi in enumerate(d["fingers"]):
                kab[FINGERS[fi]].append(float(ke[:, j].mean()))
            kab["thumb"].append(float(ke[:, F:2 * F].mean()))
            gap_p.append(float(d["gap_pos_mm"].median())); gap_r.append(float(d["gap_rot_deg"].median()))
    out = {"per_finger_contact_residual_mm": {f: float(np.mean(v)) for f, v in per.items() if v},
           "per_finger_contact_residual_mm_pairweighted": {f: float(np.mean(v)) for f, v in perw.items() if v}}
    if gap_p:
        out["kabsch_refit_per_finger_mm"] = {f: float(np.mean(v)) for f, v in kab.items() if v}
        out["e3_vs_kabsch_wrist_gap_median_mm"] = float(np.mean(gap_p))
        out["e3_vs_kabsch_wrist_gap_median_deg"] = float(np.mean(gap_r))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["e1", "e2", "e3"], required=True)
    parser.add_argument("--hands", default="allegro,leap,shadow,wuji,sharpa")
    parser.add_argument("--metadata", default="metadata")
    parser.add_argument("--play", default="artifacts/play_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=7500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1e-5)
    parser.add_argument("--lam", type=float, default=50.)
    parser.add_argument("--lam-warmup", type=int, default=0)
    parser.add_argument("--init", help="warm-start weights (E2)")
    parser.add_argument("--e3-head", choices=["shared", "separate"], default="shared")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--validate-interval", type=int, default=500)
    parser.add_argument("--memory-fraction", type=float, default=0.12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    hands = args.hands.split(",")
    fks = {h: PadFK(h, args.metadata).to(args.device) for h in hands}
    codecs = build(fks, args.mode, args.latent_dim, args.e3_head).to(args.device)
    init_info = None
    if args.init:
        saved = torch.load(args.init, map_location=args.device, weights_only=False)
        if saved["config"]["hands"] != args.hands:
            raise ValueError("init hand set differs")
        codecs.load_state_dict(saved["state_dict"])
        init_info = {"path": args.init, "step": saved["step"]}
    train, val = {}, {}
    for h in hands:
        data = np.load(Path(args.play) / f"{h}.npz")
        if list(data["joint_names"]) != fks[h].joint_names:
            raise ValueError(f"Joint ordering mismatch: {h}")
        train[h] = torch.tensor(data["train"], device=args.device)
        val[h] = torch.tensor(data["val"], device=args.device)
    optimizer = torch.optim.Adam(codecs.parameters(), lr=args.lr)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "validation.jsonl"):
        if (out / name).exists():
            (out / name).unlink()
    config = {**vars(args), "arm_mode": args.mode, "encoder_type": "raw", "init_info": init_info,
              "parameters": sum(p.numel() for p in codecs.parameters()),
              "hand_dofs": {h: fks[h].dof for h in hands}, "hand_scales_m": {h: fks[h].scale for h in hands},
              "loss": ("rec + lam*(F-frame cross residual^2 + self pose loss) + beta*KL" if args.mode == "e3"
                       else "rec + lam*normalized weighted Kabsch pad residual^2 (all ordered cross pairs) + beta*KL"),
              "kl_warmup_steps": 0 if args.init else 500, "optimizer": "fresh Adam"}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    params = decoder_params(codecs)
    if args.steps == 0:  # evaluation of an existing checkpoint with the same validation metrics
        codecs.eval()
        with torch.no_grad():
            vb = {h: q[:2048] for h, q in val.items()}
            _, _, values, stats, diag = objective(vb, codecs, fks, args.beta, args.lam, args.mode,
                                                  sample=False, diagnostics=True)
            row = {"step": 0, **{k: float(v) for k, v in values.items()}, **summarize_diag(diag, fks)}
            row["joint_rmse_rad"] = {h: float((fks[h].denormalize(a) - fks[h].denormalize(b)).square().mean().sqrt())
                                     for h, (a, b) in stats.items()}
        (out / "validation.jsonl").write_text(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        return
    start = previous = time.monotonic()
    best = float("inf")
    for step in range(1, args.steps + 1):
        codecs.train()
        batch = {h: q[torch.randint(len(q), (args.batch_size,), device=args.device)] for h, q in train.items()}
        beta = args.beta if args.init else args.beta * min(1., step / 500)
        lam = args.lam * (min(1., step / args.lam_warmup) if args.lam_warmup else 1.)
        optimizer.zero_grad(set_to_none=True)
        loss, terms, values, _, _ = objective(batch, codecs, fks, beta, lam, args.mode)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        log = step == 1 or step % args.log_interval == 0
        grad_norms = {}
        if log:
            for k, term in terms.items():
                gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                grad_norms[k] = float(torch.sqrt(sum((g.square().sum() for g in gs if g is not None),
                                                     torch.zeros((), device=loss.device))))
        loss.backward()
        total_norm = float(torch.nn.utils.clip_grad_norm_(codecs.parameters(), 10))
        optimizer.step()
        if log:
            if args.device == "cuda":
                torch.cuda.synchronize()
            now = time.monotonic()
            row = {"step": step, **{k: float(v.detach()) for k, v in values.items()},
                   "beta": beta, "grad_norm_decoder": grad_norms, "grad_norm_total_preclip": total_norm,
                   "wall_s": now - start, "interval_s": now - previous,
                   "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None}
            with (out / "train.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            previous = now
        if step % args.validate_interval == 0 or step == args.steps:
            codecs.eval()
            with torch.no_grad():
                vb = {h: q[:2048] for h, q in val.items()}
                score, _, values, stats, diag = objective(vb, codecs, fks, args.beta, args.lam, args.mode,
                                                          sample=False, diagnostics=True)
                row = {"step": step, **{k: float(v) for k, v in values.items()}, **summarize_diag(diag, fks)}
                row["joint_rmse_rad"] = {h: float((fks[h].denormalize(a) - fks[h].denormalize(b)).square().mean().sqrt())
                                         for h, (a, b) in stats.items()}
            with (out / "validation.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            payload = {"config": config, "step": step, "state_dict": codecs.state_dict(),
                       "optimizer": optimizer.state_dict(), "validation": row}
            torch.save(payload, out / "last.tmp"); os.replace(out / "last.tmp", out / "last.pt")
            if float(score) < best:
                best = float(score)
                torch.save(payload, out / "best.tmp"); os.replace(out / "best.tmp", out / "best.pt")
            print(json.dumps({"validation": row, "best": best}), flush=True)
    (out / "done.json").write_text(json.dumps({"steps": args.steps, "wall_s": time.monotonic() - start,
                                              "best_loss": best}, indent=2))


if __name__ == "__main__":
    main()
