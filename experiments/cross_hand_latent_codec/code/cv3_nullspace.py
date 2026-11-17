"""codec v3 analysis (261116): is the residual hand signal in z FUNCTIONALLY silent? For each hand h, the functional subspace
of z at a point = row space of J_h(z) = d[pad centroids + tips (palm frame) / s] / dz through Dec_h; the complement is the
functional null space (z directions that change joints but not fingertip/pad geometry, e.g. redundant DOF, hand-specific
posture details). We measure, on GEO-val ALL5/H3 z (encoder means):
  - energy of the hand-discriminative directions (LDA, train split, ridge) inside the per-hand functional null space
    (averaged over hands and points) vs a random-direction baseline (= null dim / D);
  - share of within-hand z variance in the functional null space.
CPU.  python.sh -m xhand_latent.cv3_nullspace --codec NAME=PATH ... --out artifacts/codec_v3/measure/nullspace.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .arm_train import load_arm
from .cv2_measure import ALL5, H3, split_groups
from .cv2_separability import lda_dirs


def jac_rows(codecs, fk, h, Z, tol=1e-3):
    """per point: orthonormal basis of the functional row space (D x r) of d geom / dz."""
    s = fk.scale
    bases, qbases = [], []
    for z in Z:
        z = torch.tensor(z, dtype=torch.float32)

        def g(zz):
            q = codecs[h].decoder(zz[None])
            pts, _, tips = fk.pads(q)
            nf = 5 if bool(fk.valid.all()) else 4
            return torch.cat([pts[0, :nf].mean(1), tips[0, :nf]], 0).reshape(-1) / s
        J = torch.autograd.functional.jacobian(g, z).numpy()  # (G, D)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        r = int((S > tol * S.max()).sum())
        bases.append(Vt[:r].T)
        Jq = torch.autograd.functional.jacobian(lambda zz: codecs[h].decoder(zz[None])[0], z).numpy()
        U, S, Vt = np.linalg.svd(Jq, full_matrices=False)
        qbases.append(Vt[:int((S > tol * S.max()).sum())].T)
    return bases, qbases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", action="append", required=True)
    ap.add_argument("--sets", default="artifacts/codec_v2/measure/pairsets.npz")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="artifacts/codec_v3/measure/nullspace.json")
    args = ap.parse_args()
    torch.set_num_threads(8)
    sets = dict(np.load(args.sets, allow_pickle=True))
    res = {}
    for spec in args.codec:
        name, path = spec.split("=", 1)
        codecs, fks, _ = load_arm(path)
        R = {}
        for tag, hands in (("GEO_ALL5", ALL5), ("GEO_H3", H3)):
            with torch.no_grad():
                Z = np.stack([codecs[h].encode(torch.tensor(sets[f"{tag}_{h}"]))[0].numpy() for h in hands], 1).astype(np.float64)
            S, H, D = Z.shape
            X = Z.reshape(S * H, D); y = np.tile(np.arange(H), S); g = np.repeat(np.arange(S), H)
            tr, te = split_groups(g)
            dirs, fr, _, _ = lda_dirs(X[tr], y[tr], H, ridge=1e-3)
            rng = np.random.default_rng(0)
            idx = rng.choice(S, args.n, replace=False)
            per = {}
            for k, h in enumerate(hands):
                B, Bq = jac_rows(codecs, fks[h], h, Z[idx, k])
                rdim = float(np.mean([b.shape[1] for b in B]))
                null_e = [float(np.mean([1 - np.square(b.T @ w).sum() for b in B])) for w in dirs]
                Zc = Z[:, k] - Z[:, k].mean(0)
                C = Zc.T @ Zc / len(Zc)
                null_var = float(np.mean([1 - np.trace(b.T @ C @ b) / np.trace(C) for b in B]))
                per[h] = {"decoder_rank": float(np.mean([b.shape[1] for b in Bq])),
                          "lda_decoder_null_energy": [float(np.mean([1 - np.square(b.T @ w).sum() for b in Bq])) for w in dirs],
                          "functional_rank": rdim, "null_dim": D - rdim, "random_baseline_null_energy": (D - rdim) / D,
                          "lda_null_energy": null_e, "within_hand_var_null_share": null_var}
            R[tag] = {"per_hand": per, "lda_fisher_train": [float(f) for f in fr],
                      "mean_lda1_null_energy": float(np.mean([p["lda_null_energy"][0] for p in per.values()])),
                      "mean_random_baseline": float(np.mean([p["random_baseline_null_energy"] for p in per.values()])),
                      "mean_lda1_decoder_null_energy": float(np.mean([p["lda_decoder_null_energy"][0] for p in per.values()])),
                      "mean_decoder_rank": float(np.mean([p["decoder_rank"] for p in per.values()])),
                      "mean_functional_rank": float(np.mean([p["functional_rank"] for p in per.values()])),
                      "mean_within_var_null_share": float(np.mean([p["within_hand_var_null_share"] for p in per.values()]))}
            print(name, tag, json.dumps({k: v for k, v in R[tag].items() if k != "per_hand"}), flush=True)
        res[name] = R
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
