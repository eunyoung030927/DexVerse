"""codec v3 diagnostic (261116): intra-hand floor of the GEO whitened same/rand ratio. For each hand, anchors from play val
are paired with their nearest neighbour IN THE SAME HAND (other val rows) by palm-frame fingertips / hand scale (the GEO
matching rule; better-matched half kept). Two z of the same hand with (almost) the same fingertip geometry differ only in
geometry-silent detail (redundant DOF, phalanx split, abduction) -> whitened same/rand of these pairs is the floor that
any cross-hand alignment measured on fingertip-matched pairs can reach. CPU.
  python.sh -m xhand_latent.cv3_floor --codec NAME=PATH ... --out artifacts/codec_v3/measure/floor.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .arm_train import load_arm
from .cv2_measure import ALL5, E2
from .cv3_measure import wmat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", action="append", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default="artifacts/codec_v3/measure/floor.json")
    ap.add_argument("--target-mm", default="nn,16,25", help="nn = nearest neighbour; numbers = partner whose tip residual is closest to that many mm (GEO-H3 16, GEO-ALL5 25)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    play = {h: np.load(f"artifacts/play_v1/{h}.npz", allow_pickle=True)["val"] for h in ALL5}
    _, fks, p0 = load_arm(E2)
    sc = p0["config"]["hand_scales_m"]
    rng = np.random.default_rng(0)
    pairs = {}
    for h in ALL5:
        Q = play[h]
        nf = 5 if bool(fks[h].valid.all()) else 4
        with torch.no_grad():
            T = (fks[h](torch.tensor(Q))[:, :nf] / sc[h]).reshape(len(Q), -1)
        a = rng.choice(len(Q), args.n, replace=False)
        D = torch.cdist(T[a], T); D[torch.arange(args.n), torch.tensor(a)] = float("inf")
        mm = D / np.sqrt(nf) * 1000 * sc[h]
        for tg in args.target_mm.split(","):
            if tg == "nn":
                v, k = mm.min(1); keep = (v <= v.median()).numpy()
            else:
                k = (mm - float(tg)).abs().argmin(1); v = mm[torch.arange(args.n), k]; keep = np.ones(args.n, bool)
            pairs[(h, tg)] = (a[keep], k.numpy()[keep], float(v[keep].median()))
    res = {}
    for spec in args.codec:
        name, path = spec.split("=", 1)
        codecs, _, _ = load_arm(path)
        enc = lambda h, q: codecs[h].encode(torch.tensor(q))[0].detach().numpy().astype(np.float64)
        Zp = {h: enc(h, play[h][:4000]) for h in ALL5}
        W = wmat(np.concatenate(list(Zp.values())))
        R = {}
        for (h, tg), (a, b, tipmm) in pairs.items():
            za, zb = enc(h, play[h][a]) @ W, enc(h, play[h][b]) @ W
            perm = rng.permutation(len(b))
            same = np.linalg.norm(za - zb, axis=1).mean(); rand = np.linalg.norm(za - zb[perm], axis=1).mean()
            R.setdefault(tg, {})[h] = {"same/rand_intrahand": float(same / rand), "tip_mm_median": tipmm}
        for tg in R:
            R[tg]["mean"] = float(np.mean([R[tg][h]["same/rand_intrahand"] for h in ALL5]))
        res[name] = R
        print(name, json.dumps({tg: {h: round(v["same/rand_intrahand"], 3) if isinstance(v, dict) else round(v, 3) for h, v in r.items()} for tg, r in R.items()}), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
