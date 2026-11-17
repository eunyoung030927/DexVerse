"""cv3alt (261116): build cross-hand PAIRED posture datasets (DESIGN §2). New file. CPU (nice).
  part  : python.sh -m xhand_latent.cv3alt_build part --method playik --source sharpa --split train --n 16000 --out <dir>
          python.sh -m xhand_latent.cv3alt_build part --method sobol --split train --n 16384 --out <dir>
          python.sh -m xhand_latent.cv3alt_build part --method human --out <dir>
  merge : python.sh -m xhand_latent.cv3alt_build merge --method playik --parts <dir> --out artifacts/codec_v3alt/data/playik.npz
Output npz: hands (HANDS5), Q_<hand>_<split> (N,dof) normalized joints, pmask_<split> (N,5) pair-loss mask, feas_<split> (N,5),
src_<split> (N,) source hand index (-1 = none), residual arrays res_<metric>_<hand>_<split>.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .arm_pads import align_pair
from .arm_train import load_arm
from .cv3alt_func import W, W_HYB, FuncHand, feasible, human_desc, residuals, solve, target_from

HANDS5 = ("allegro", "leap", "shadow", "wuji", "sharpa")
E2 = "artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt"
LITTLE_DEV = 0.25  # |u_little - u_ring| (finger lengths) above which a 5-finger-sourced scene is not paired with the 4-finger hands (C2)


def load_hands(device="cpu"):
    play = {h: np.load(f"artifacts/play_v1/{h}.npz", allow_pickle=True) for h in HANDS5}
    return play, {h: FuncHand(h, play[h]).to(device) for h in HANDS5}


def e2_pad_mm(Hs, Ht, qs_or_pads, qt):
    with torch.no_grad():
        pa = qs_or_pads if isinstance(qs_or_pads, tuple) else Hs.fk.pads(qs_or_pads)
        r = align_pair(Hs.fk, Ht.fk, pa, Ht.fk.pads(qt))
    return (r["r2"].sqrt() * Hs.fk.scale * 1000).cpu().numpy()


def part_playik(args, play, H):
    src = args.source
    rng = np.random.default_rng(args.seed + HANDS5.index(src))
    pool = play[src][args.split]
    idx = np.sort(rng.choice(len(pool), min(args.n, len(pool)), replace=False))
    q = torch.tensor(pool[idx], device=args.device)
    codecs, _, _ = load_arm(E2, device=args.device)
    wts = W_HYB if args.wts == "hyb" else W
    with torch.no_grad():
        d = H[src].desc(q); sp = H[src].fk.pads(q); z = codecs[src].encode(q)[0]
    out = {"idx": idx, f"Q_{src}": q.cpu().numpy(), "feas": {src: np.ones(len(q), bool)}, "res": {}}
    for tgt in HANDS5:
        if tgt == src:
            continue
        t0 = time.time()
        t, fw = target_from(src, d, H[src].valid, H[tgt].valid, src_pads=sp)
        with torch.no_grad():
            qd = codecs[tgt].decoder(z)
        inits = [qd.clone(), H[tgt].q_open[None].expand(len(q), -1).clone()]
        qb, lb = solve(H[tgt], t, fw, inits, iters=args.iters, wts=wts, src_fk=H[src].fk, chunk=args.chunk)
        r = residuals(H[tgt], qb, t, fw)
        out[f"Q_{tgt}"] = qb.cpu().numpy()
        pad = e2_pad_mm(H[src], H[tgt], sp, qb)
        # HYB feasibility: E2*-metric pad residual, functional finger shape, pinch preservation (DESIGN amendment C2)
        out["feas"][tgt] = (pad <= 12.0) & (r["shape_mean_fingers"] <= 0.4) & (r["opp_pinch_max_mm"] <= 10.0)
        out["res"][tgt] = {"shape": r["shape_mean_fingers"], "pinch_mm": r["opp_pinch_max_mm"], "thumb_mm": r["thumb_mm"],
                           "e2pad_mm": pad, "e2pad_mm_E2dec": e2_pad_mm(H[src], H[tgt], sp, qd),
                           "loss": lb.cpu().numpy()}
        print(json.dumps({"src": src, "tgt": tgt, "n": len(q), "s": round(time.time() - t0, 1),
                          "feas": float(out["feas"][tgt].mean()), "shape_med": float(np.median(r["shape_mean_fingers"])),
                          "e2pad_med": float(np.median(out["res"][tgt]["e2pad_mm"])),
                          "e2pad_E2dec_med": float(np.median(out["res"][tgt]["e2pad_mm_E2dec"]))}), flush=True)
    # little-finger rule for pairing with 4-finger hands (5-finger source only)
    out["little_dev"] = ((d["u"][:, 4] - d["u"][:, 3]).norm(dim=-1).cpu().numpy() if H[src].valid[4] else np.zeros(len(q)))
    return out


def sobol_params(n, seed):
    eng = torch.quasirandom.SobolEngine(15, scramble=True, seed=seed)
    return eng.draw(n).numpy().astype(np.float64)


def sobol_targets(P):
    """P (N,15) in [0,1] -> per-finger target u (N,5,3) in the finger frame + thumb params (k, g)."""
    N = len(P)
    u = np.zeros((N, 5, 3))
    th_lo, th_hi, r_lo, r_hi, ps = -0.15, 1.9, 0.45, 1.0, 0.3
    vals = {}
    for j, fi in enumerate((1, 2, 3)):
        th = th_lo + (th_hi - th_lo) * P[:, 3 * j]; r = r_lo + (r_hi - r_lo) * P[:, 3 * j + 1]; psi = -ps + 2 * ps * P[:, 3 * j + 2]
        vals[fi] = (th, r, psi)
    th, r, psi = vals[3]  # little = ring + bounded perturbation (natural ring-little coupling; lets 4-finger hands pair)
    vals[4] = (np.clip(th + 0.3 * (P[:, 9] - 0.5), th_lo, th_hi), np.clip(r + 0.15 * (P[:, 10] - 0.5), r_lo, r_hi),
               np.clip(psi - 0.1 + 0.1 * (P[:, 11] - 0.5), -ps, ps))
    for fi, (th, r, psi) in vals.items():
        u[:, fi] = r[:, None] * np.stack([np.cos(th) * np.cos(psi), np.cos(th) * np.sin(psi), -np.sin(th)], 1)
    k, g, nflip = P[:, 12], P[:, 13], P[:, 14]
    return u, k, g


def part_sobol(args, play, H):
    P = sobol_params(args.n, args.seed + (0 if args.split == "train" else 1000))
    u, k, g = sobol_targets(P)
    N = len(P)
    out = {"params": P, "feas": {}, "res": {}}
    dev = args.device
    for h in args.hands.split(","):
        t0 = time.time()
        Hh = H[h]
        # stage 1: fingers only (thumb fixed open), target finger-frame shape u
        tgt = {"u": torch.tensor(u, dtype=torch.float32, device=dev), "m": torch.zeros(N, 5, 3, device=dev),
               "o": torch.zeros(N, 5, 3, device=dev), "t": torch.zeros(N, 3, device=dev), "nT": torch.zeros(N, 3, device=dev)}
        fw = np.array([0.0] + [1.0 if Hh.valid[f] else 0.0 for f in range(1, 5)], np.float32)
        thumb_cols = [Hh.fk.qidx[n] for n in Hh.fk.spec["finger"]["thumb"]["chain_joints"]]
        mask = np.ones(Hh.dof, np.float32); mask[thumb_cols] = 0
        w1 = {"u": 1.0, "m": 0.0, "o": 0.0, "t": 0.0, "nT": 0.0, "uT": 0.0, "reg": 1e-3, "align": 0.0}
        q_open = Hh.q_open[None].expand(N, -1).clone()
        inits = [q_open.clone(), torch.where(torch.tensor(mask, dtype=torch.bool, device=dev), torch.zeros(N, Hh.dof, device=dev), q_open)]
        q1, _ = solve(Hh, tgt, fw, inits, iters=args.iters, mask=mask, wts=w1, chunk=args.chunk)
        with torch.no_grad():
            d1 = Hh.desc(q1); dopen = Hh.desc(Hh.q_open[None])
        ue = (d1["u"] - tgt["u"]).norm(dim=-1).cpu().numpy()  # (N,5)
        # stage 2: thumb target = fraction g between the (interpolated) target-finger pad and the thumb-open pad position
        nf = 3  # I, M, R are common to every hand
        kk = np.clip(k * (nf - 1), 0, nf - 1 - 1e-6); i0 = np.floor(kk).astype(int); fr = kk - i0
        c = d1["c"].cpu().numpy(); n = d1["n"].cpu().numpy()
        Pp = c[np.arange(N), 1 + i0] * (1 - fr)[:, None] + c[np.arange(N), 2 + i0] * fr[:, None]
        Pn = n[np.arange(N), 1 + i0] * (1 - fr)[:, None] + n[np.arange(N), 2 + i0] * fr[:, None]
        Pn /= np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-9
        contact = Pp + 0.006 * Pn  # stand-off in front of the pad
        cT_open = dopen["c"][0, 0].cpu().numpy()
        cT = contact + g[:, None] * (cT_open[None] - contact)
        tgt2 = {"u": torch.zeros(N, 5, 3, device=dev), "m": torch.zeros(N, 5, 3, device=dev), "o": torch.zeros(N, 5, 3, device=dev),
                "t": torch.tensor(cT / Hh.s, dtype=torch.float32, device=dev), "nT": torch.tensor(-Pn, dtype=torch.float32, device=dev)}
        w2 = {"u": 0.0, "m": 0.0, "o": 0.0, "t": 10.0, "nT": 0.2, "uT": 0.0, "reg": 1e-3, "align": 0.0}
        mask2 = 1 - mask
        fw2 = np.zeros(5, np.float32)
        q2, _ = solve(Hh, tgt2, fw2, [q1.clone(), torch.where(torch.tensor(mask2, dtype=torch.bool, device=dev),
                                                              torch.zeros(N, Hh.dof, device=dev), q1)],
                      iters=args.iters, mask=mask2, wts=w2, chunk=args.chunk)
        with torch.no_grad():
            d2 = Hh.desc(q2)
        te = (d2["c"][:, 0].cpu().numpy() - cT)
        te = np.linalg.norm(te, axis=1) * 1000
        shape = (ue[:, 1:] * fw[1:]).sum(1) / fw[1:].sum()
        out[f"Q_{h}"] = q2.cpu().numpy()
        out["feas"][h] = (shape <= 0.15) & (te <= np.where(g < 0.3, 10.0, 25.0))
        out["res"][h] = {"shape": shape, "thumb_mm": te, "shape_per_finger": ue}
        print(json.dumps({"hand": h, "n": N, "s": round(time.time() - t0, 1), "feas": float(out["feas"][h].mean()),
                          "shape_med": float(np.median(shape)), "thumb_med": float(np.median(te)),
                          "thumb_p90": float(np.percentile(te, 90))}), flush=True)
    return out


def part_human(args, play, H):
    d = np.load(args.human, allow_pickle=True)
    J = d["right"]
    keep = np.concatenate([[True], np.abs(np.diff(J, axis=0)).sum((1, 2)) > 0]) & (np.abs(J).sum((1, 2)) > 0)
    J = J[keep]
    hd, info = human_desc(J)
    N = len(J)
    hv = np.ones(5, bool)
    out = {"frame_idx": np.where(keep)[0], "feas": {}, "res": {}, "human_info": info}
    wts = {**W, "uT": 0.0, "t": 0.1}
    codecs, _, _ = load_arm(E2)
    for h in HANDS5:
        t0 = time.time()
        t, fw = target_from("human", hd, hv, H[h].valid)
        inits = [H[h].q_open[None].expand(N, -1).clone(), torch.zeros(N, H[h].dof)]
        qb, lb = solve(H[h], t, fw, inits, iters=args.iters, wts=wts)
        r = residuals(H[h], qb, t, fw)
        out[f"Q_{h}"] = qb.numpy()
        out["feas"][h] = feasible(r, thumb_thr=np.inf)
        out["res"][h] = {"shape": r["shape_mean_fingers"], "pinch_mm": r["opp_pinch_max_mm"], "thumb_mm": r["thumb_mm"],
                         "loss": lb.numpy()}
        print(json.dumps({"hand": h, "n": N, "s": round(time.time() - t0, 1), "feas": float(out["feas"][h].mean()),
                          "shape_med": float(np.median(r["shape_mean_fingers"])),
                          "pinch_p90": float(np.percentile(r["opp_pinch_max_mm"], 90)),
                          "has_pinch": float(r["has_pinch"].mean())}), flush=True)
    out["little_dev"] = (hd["u"][:, 4] - hd["u"][:, 3]).norm(dim=-1).numpy()
    # cross-check: E2* pad residual between robot pairs of this dataset (sharpa as reference source)
    out["e2pad_vs_sharpa"] = {h: e2_pad_mm(H["sharpa"], H[h], torch.tensor(out["Q_sharpa"]), torch.tensor(out[f"Q_{h}"]))
                              for h in HANDS5 if h != "sharpa"}
    return out


def merge(args):
    parts = sorted(Path(args.parts).glob(f"{args.method}_*.pt"))
    if args.method == "sobol":  # per-hand part files of one split -> one record per split
        recs = {}
        for p in parts:
            o = torch.load(p, weights_only=False)
            r = recs.setdefault(o["split"], {"source": None, "split": o["split"], "params": o["params"], "feas": {}, "res": {}})
            assert np.array_equal(r["params"], o["params"])
            r["feas"].update(o["feas"]); r["res"].update(o["res"])
            r.update({k: v for k, v in o.items() if k.startswith("Q_")})
        Path(args.parts, "_merged").mkdir(exist_ok=True)
        parts = []
        for split, r in recs.items():
            missing = [h for h in HANDS5 if f"Q_{h}" not in r]
            if missing:
                raise SystemExit(f"sobol {split}: missing hands {missing}")
            pth = Path(args.parts, "_merged", f"sobol_{split}.pt"); torch.save(r, pth); parts.append(pth)
    D = {"hands": np.array(HANDS5)}
    summary = {"method": args.method, "parts": [str(p) for p in parts]}
    for split in ("train", "val"):
        Qs = {h: [] for h in HANDS5}; F, PM, SRC, RES = [], [], [], {}
        for p in parts:
            o = torch.load(p, weights_only=False)
            if o["split"] != split:
                continue
            n = len(o[f"Q_{HANDS5[0]}"])
            for h in HANDS5:
                Qs[h].append(o[f"Q_{h}"].astype(np.float32))
            f = np.stack([o["feas"][h] for h in HANDS5], 1)
            pm = f.copy()
            ld = o.get("little_dev", np.zeros(n))
            for j, h in enumerate(HANDS5):
                if h in ("allegro", "leap"):
                    pm[:, j] &= ld <= LITTLE_DEV
            F.append(f); PM.append(pm)
            SRC.append(np.full(n, HANDS5.index(o["source"]) if o.get("source") in HANDS5 else -1))
            keys = sorted({k for r in o["res"].values() for k, v in r.items() if np.ndim(v) == 1})
            for h in HANDS5:
                for k in keys:
                    v = o["res"].get(h, {}).get(k)
                    RES.setdefault(f"res_{k}_{h}_{split}", []).append(
                        np.asarray(v, np.float32) if v is not None else np.full(n, np.nan, np.float32))
        if not F:
            continue
        for h in HANDS5:
            D[f"Q_{h}_{split}"] = np.concatenate(Qs[h])
        D[f"feas_{split}"] = np.concatenate(F); D[f"pmask_{split}"] = np.concatenate(PM); D[f"src_{split}"] = np.concatenate(SRC)
        for k, v in RES.items():
            D[k] = np.concatenate(v)
        fe, pm = D[f"feas_{split}"], D[f"pmask_{split}"]
        summary[split] = {"n_scenes": int(len(fe)), "feasible_rate": {h: float(fe[:, j].mean()) for j, h in enumerate(HANDS5)},
                          "pair_mask_rate": {h: float(pm[:, j].mean()) for j, h in enumerate(HANDS5)},
                          "all5_valid": float(pm.all(1).mean()), "h3_valid": float(pm[:, 2:].all(1).mean())}
    np.savez_compressed(args.out, **D)
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("part", "merge"))
    ap.add_argument("--method", choices=("playik", "sobol", "human"), required=True)
    ap.add_argument("--source", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--wts", choices=("w", "hyb"), default="hyb")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--human", default="artifacts/codec_v3alt/data/human/xr_hand_dump.npz")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--hands", default=",".join(HANDS5), help="sobol: hands to solve in this process")
    ap.add_argument("--memory-fraction", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == "merge":
        return merge(args)
    if args.device == "cuda" and args.memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    play, H = load_hands(args.device if args.method in ("playik", "sobol") else "cpu")
    t0 = time.time()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    if args.method == "playik":
        o = part_playik(args, play, H); o["source"] = args.source; o["split"] = args.split
        torch.save(o, outdir / f"playik_{args.source}_{args.split}.pt")
    elif args.method == "sobol":
        o = part_sobol(args, play, H); o["source"] = None; o["split"] = args.split
        torch.save(o, outdir / f"sobol_{args.split}_{args.hands.replace(',', '-')}.pt")
    else:
        o = part_human(args, play, H)
        N = len(o["frame_idx"]); cut = int(N * (1 - args.val_frac))
        for split, sl in (("train", slice(0, cut)), ("val", slice(cut, N))):
            p = {"source": None, "split": split, "frame_idx": o["frame_idx"][sl], "human_info": o["human_info"],
                 "little_dev": o["little_dev"][sl], "feas": {h: v[sl] for h, v in o["feas"].items()},
                 "res": {h: {k: v[sl] for k, v in r.items()} for h, r in o["res"].items()}}
            for h in HANDS5:
                p[f"Q_{h}"] = o[f"Q_{h}"][sl]
            torch.save(p, outdir / f"human_{split}.pt")
        torch.save({"e2pad_vs_sharpa": o["e2pad_vs_sharpa"], "human_info": o["human_info"]}, outdir / "human_extra.ptx")
    print(json.dumps({"done": args.method, "wall_s": time.time() - t0}), flush=True)


if __name__ == "__main__":
    main()
