"""cv3alt (261116): latent visualization JSON in the schema of artifacts/codec_v2/viz/latent_viz.json (same GEO-val scenes,
same 40 pair links, same Kettle frame selection: identical rng sequence as cv2_separability.main). New file. CPU.
  python.sh -m xhand_latent.cv3alt_viz --codec NAME=PATH [...] --out artifacts/codec_v3alt/viz/latent_viz_v3alt.json
Stats come from artifacts/codec_v3alt/measure/<NAME>.json (cv3alt_measure).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .arm_train import load_arm
from .cv2_measure import ALL5, E2, kettle_action_q
from .cv2_separability import cov, r3, tsne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", action="append", required=True)
    ap.add_argument("--out", default="artifacts/codec_v3alt/viz/latent_viz_v3alt.json")
    ap.add_argument("--measure", default="artifacts/codec_v3alt/measure")
    ap.add_argument("--sets", default="artifacts/codec_v2/measure/pairsets.npz")
    ap.add_argument("--n-viz", type=int, default=250)
    ap.add_argument("--n-kettle", type=int, default=150)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
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
    VIZ = {}
    for spec in args.codec:
        name, path = spec.split("=", 1)
        codecs, fks, payload = load_arm(path)

        @torch.no_grad()
        def enc(h, q):
            return codecs[h].encode(torch.as_tensor(np.asarray(q), dtype=torch.float32))[0].numpy()
        Zv = np.concatenate([enc(h, sets[f"GEO_ALL5_{h}"][viz_s]) for h in ALL5])
        hv = np.repeat(np.arange(5), args.n_viz)
        mu = Zv.mean(0); ev, Uv = np.linalg.eigh(cov(Zv)); ev, Uv = ev[::-1], Uv[:, ::-1]
        Y2 = (Zv - mu) @ Uv[:, :2]
        norm = lambda h, qf: fks[h].normalize(torch.tensor(np.asarray(qf), dtype=torch.float32)).clamp(-1, 1)
        Zk = np.concatenate([np.concatenate([enc(h, norm(h, qf)) for qf in kq[h]])[ket_sel[h]] for h in ALL5])
        hk = np.repeat(np.arange(5), args.n_kettle)
        K2 = (Zk - mu) @ Uv[:, :2]
        m = json.load(open(Path(args.measure) / f"{name}.json"))
        k2 = m.get("KETTLE", {}).get("K2", {})
        stats = {"probe_GEO_ALL5": m["v2GEO_ALL5"]["probe_raw"], "probe_GEO_H3": m["v2GEO_H3"]["probe_raw"],
                 "probe_SYN_ALL5": m["v2SYN_ALL5"]["probe_raw"],
                 "same_within_GEO_ALL5": m["v2GEO_ALL5"]["R_euclid_same/within"],
                 "same_within_GEO_H3": m["v2GEO_H3"]["R_euclid_same/within"],
                 "whitened_same_within_GEO_ALL5": m["v2GEO_ALL5"]["R_whitened"],
                 "whitened_same_within_GEO_H3": m["v2GEO_H3"]["R_whitened"],
                 "hand_var_share_GEO_ALL5": m["v2GEO_ALL5"]["frac_hand"], "hand_var_share_GEO_H3": m["v2GEO_H3"]["frac_hand"],
                 "qda_cov_shape_GEO_ALL5": m["v2GEO_ALL5"]["gauss"]["qda0"],
                 "knn_perhand_whitened_GEO_ALL5": m["v2GEO_ALL5"]["knn15_perhand_whitened"],
                 "pc1_share_GEO_ALL5": m["v2GEO_ALL5"]["pc1_share"],
                 "pad_residual_mm": m["quality"]["pad_residual_mm_mean"],
                 "probe_GEO_ALL5_euclid_nearest_centroid": m["v2GEO_ALL5"]["gauss"]["nc"]}
        for u in ("allegro", "leap"):
            if u in k2:
                stats[f"kettle_frac_inside_H3_p95_{u}"] = k2[u]["frac_inside_ref_p95"]
                stats[f"kettle_whitened_frac_inside_H3_p95_{u}"] = k2[u]["whitened_frac_inside_ref_p95"]
        T2 = tsne(Zv)
        VIZ[name] = {"pca": {"x": r3(Y2[:, 0]), "y": r3(Y2[:, 1]), "hand": hv.tolist()}, "pairs": viz_pairs,
                     "kettle": {"x": r3(K2[:, 0]), "y": r3(K2[:, 1]), "hand": hk.tolist()},
                     "stats": {k: round(float(v), 4) for k, v in stats.items()}, "pca_var": r3(ev[:2] / ev.sum()),
                     "tsne": {"x": r3(T2[:, 0]), "y": r3(T2[:, 1]), "hand": hv.tolist()}}
        print("[viz]", name, flush=True)
    notes = {
        "hands": "index into the top-level 'hands' list (allegro, leap = 4-finger; shadow, wuji, sharpa = 5-finger)",
        "pca": f"z = encoder mean on the codec-independent GEO-val ALL5 pair set (artifacts/codec_v2/measure/pairsets.npz), "
               f"{args.n_viz} scenes x 5 hands, the SAME scenes as artifacts/codec_v2/viz/latent_viz.json. Hand-major: index = hand*{args.n_viz} + scene. "
               "2-component PCA fit on the pooled z of these points, per codec.",
        "tsne": "exact t-SNE (cv2_separability.tsne, perplexity 30, 1000 iters) on the same z, per codec",
        "pairs": "40 same-scene cross-hand links [i, j] (identical to the v2 file)",
        "kettle": f"Kettle commanded finger q (data/replay eps 0-9), {args.n_kettle} frames per hand (same selection as v2), same PCA",
        "stats": "from artifacts/codec_v3alt/measure/<codec>.json: probe_* = standardized linear hand probe; same_within = Euclidean "
                 "same-scene/same-hand-diff-scene; whitened_* = same ratio after pooled whitening (cause-1-proof); qda_cov_shape = "
                 "Gaussian classifier with per-hand means removed; knn_perhand_whitened = kNN-15 after per-hand whitening (cause 2)",
        "pca_var": "explained variance ratio of PC1, PC2 on the pooled viz z",
        "source": "xhand_latent/cv3alt_viz.py; analysis in artifacts/codec_v3alt/RESULTS_V3ALT.md"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"codecs": VIZ, "hands": list(ALL5), "notes": notes}, separators=(",", ":")))
    print("[cv3alt_viz] DONE", flush=True)


if __name__ == "__main__":
    main()
