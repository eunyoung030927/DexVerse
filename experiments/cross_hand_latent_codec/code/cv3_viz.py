"""codec v3 latent_viz.json, SAME schema and SAME point selection as artifacts/codec_v2/viz/latent_viz.json (cv2_separability):
250 GEO-val ALL5 scenes x 5 hands (rng 0), 40 same-scene links, 150 Kettle frames per hand projected with the same PCA,
t-SNE (cv2_separability.tsne). Stats come from artifacts/codec_v3/measure/<name>.json (+ cv2/<name>.json). Extra keys per
codec: 'pca_white' (PCA after ZCA whitening with the codec's pooled play-val z) and 'tax' (held-out taxonomy class points,
if the taxonomy file exists), both optional.
  python.sh -m xhand_latent.cv3_viz --codec NAME=PATH ... --out artifacts/codec_v3/viz/latent_viz.json
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
from .cv3_measure import load_tax, wmat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", action="append", required=True)
    ap.add_argument("--out", default="artifacts/codec_v3/viz/latent_viz.json")
    ap.add_argument("--sets", default="artifacts/codec_v2/measure/pairsets.npz")
    ap.add_argument("--measure", default="artifacts/codec_v3/measure")
    ap.add_argument("--tax", default="artifacts/codec_v3/data/tax_pairs.npz")
    ap.add_argument("--n-viz", type=int, default=250)
    ap.add_argument("--n-kettle", type=int, default=150)
    ap.add_argument("--no-tsne", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    play = {h: np.load(f"artifacts/play_v1/{h}.npz", allow_pickle=True) for h in ALL5}
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
    tax = None
    if Path(args.tax).exists():
        Qc, Vc, cc = load_tax(args.tax, 2)
        sel = np.where(Vc.all(1))[0]
        sel = np.sort(np.random.default_rng(1).choice(sel, min(100, len(sel)), replace=False))
        names = [str(x) for x in np.load(args.tax, allow_pickle=True)["class_names"]]
        tax = ({h: Qc[h][sel] for h in ALL5}, [names[c] for c in cc[sel]])
    VIZ = {}
    for spec in args.codec:
        cname, path = spec.split("=", 1)
        codecs, fks, payload = load_arm(path)

        @torch.no_grad()
        def enc(h, q):
            return codecs[h].encode(torch.as_tensor(np.asarray(q), dtype=torch.float32))[0].numpy().astype(np.float64)
        norm = lambda h, q: fks[h].normalize(torch.tensor(np.asarray(q), dtype=torch.float32)).clamp(-1, 1)
        ZK = {h: [enc(h, norm(h, q)) for q in kq[h]] for h in ALL5}
        Zv = np.concatenate([enc(h, sets[f"GEO_ALL5_{h}"][viz_s]) for h in ALL5])
        hv = np.repeat(np.arange(5), args.n_viz)
        mu = Zv.mean(0); ev, Uv = np.linalg.eigh(cov(Zv)); ev, Uv = ev[::-1], Uv[:, ::-1]
        Y2 = (Zv - mu) @ Uv[:, :2]
        Zk = np.concatenate([np.concatenate(ZK[h])[ket_sel[h]] for h in ALL5]); hk = np.repeat(np.arange(5), args.n_kettle)
        K2 = (Zk - mu) @ Uv[:, :2]
        W = wmat(np.concatenate([enc(h, play[h]["val"][:4000]) for h in ALL5]))
        Zw = Zv @ W; mw = Zw.mean(0); ew, Uw = np.linalg.eigh(cov(Zw)); ew, Uw = ew[::-1], Uw[:, ::-1]
        Yw = (Zw - mw) @ Uw[:, :2]; Kw = (Zk @ W - mw) @ Uw[:, :2]
        stats = {}
        mp, mp2 = Path(args.measure) / f"{cname}.json", Path(args.measure) / "cv2" / f"{cname}.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            stats.update({"probe_GEO_ALL5": m["GEO_ALL5"]["handid"]["probe_std_lr"], "probe_GEO_H3": m["GEO_H3"]["handid"]["probe_std_lr"],
                          "qda_GEO_H3": m["GEO_H3"]["handid"]["qda"], "knn_white_GEO_H3": m["GEO_H3"]["handid"]["knn15_white"],
                          "white_same_rand_GEO_ALL5": m["GEO_ALL5"]["white"]["same/rand"], "white_same_rand_GEO_H3": m["GEO_H3"]["white"]["same/rand"],
                          "pc1_share_playval": m["collapse"]["pooled_playval"]["pc1_share"],
                          "participation_ratio_playval": m["collapse"]["pooled_playval"]["participation_ratio"],
                          "pad_residual_mm": m["retarget"]["playval"]["overall_G5style"],
                          "kettle_white_in_p95_allegro": m["KETTLE_K2_white"]["allegro"]["frac_inside_ref_p95"],
                          "kettle_white_in_p95_leap": m["KETTLE_K2_white"]["leap"]["frac_inside_ref_p95"]})
            if "TAXC_ALL5" in m:
                stats["white_same_rand_TAX_heldout_class"] = m["TAXC_ALL5"]["white"]["same/rand"]
        if mp2.exists():
            m2 = json.loads(mp2.read_text())
            stats.update({"same_within_GEO_ALL5": m2["GEO_ALL5"]["dist"]["MEAN"]["same/within"],
                          "hand_var_share_GEO_ALL5": m2["GEO_ALL5"]["two_way"]["frac_hand"],
                          "kettle_frac_inside_H3_p95_allegro": m2["KETTLE_K2"]["allegro"]["frac_inside_ref_p95"],
                          "kettle_frac_inside_H3_p95_leap": m2["KETTLE_K2"]["leap"]["frac_inside_ref_p95"]})
        VIZ[cname] = {"pca": {"x": r3(Y2[:, 0]), "y": r3(Y2[:, 1]), "hand": hv.tolist()}, "pairs": viz_pairs,
                      "kettle": {"x": r3(K2[:, 0]), "y": r3(K2[:, 1]), "hand": hk.tolist()},
                      "stats": {k: round(float(v), 4) for k, v in stats.items()}, "pca_var": r3(ev[:2] / ev.sum()),
                      "pca_white": {"x": r3(Yw[:, 0]), "y": r3(Yw[:, 1]), "hand": hv.tolist(), "kettle_x": r3(Kw[:, 0]),
                                    "kettle_y": r3(Kw[:, 1]), "var": r3(ew[:2] / ew.sum())}}
        if tax is not None:
            Zt = np.concatenate([enc(h, tax[0][h]) for h in ALL5]); T2 = (Zt - mu) @ Uv[:, :2]
            VIZ[cname]["tax"] = {"x": r3(T2[:, 0]), "y": r3(T2[:, 1]), "hand": np.repeat(np.arange(5), len(tax[1])).tolist(),
                                 "posture": tax[1]}
        if not args.no_tsne:
            T2 = tsne(Zv)
            VIZ[cname]["tsne"] = {"x": r3(T2[:, 0]), "y": r3(T2[:, 1]), "hand": hv.tolist()}
        print("viz", cname, flush=True)
    notes = {
        "hands": "index into 'hands' (allegro, leap = 4-finger; shadow, wuji, sharpa = 5-finger)",
        "pca": f"z = encoder mean on GEO-val ALL5 (artifacts/codec_v2/measure/pairsets.npz), {args.n_viz} scenes x 5 hands, same scenes as "
               "codec_v2/viz/latent_viz.json; hand-major index = hand*250 + scene; 2-D PCA per codec",
        "tsne": "exact t-SNE (cv2_separability.tsne, perplexity 30, 1000 iters) on the same z",
        "pairs": "40 same-scene cross-hand links (same as v2 viz)",
        "kettle": "Kettle commanded finger q (data/replay eps 0-9), 150 frames per hand, projected with the same PCA",
        "pca_white": "same points after ZCA whitening with the codec's pooled play-val z (4000/hand), then 2-D PCA; kettle_x/y projected likewise",
        "tax": "held-out-class taxonomy postures valid on all 5 hands (up to 100 rows x 5 hands, hand-major), projected with the same PCA; posture = class id",
        "stats": "from artifacts/codec_v3/measure/<name>.json and measure/cv2/<name>.json (see RESULTS_CV3.md)",
        "pca_var": "explained variance ratio PC1, PC2", "source": "xhand_latent/cv3_viz.py"}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"codecs": VIZ, "hands": list(ALL5), "notes": notes}, separators=(",", ":")))
    print("[cv3_viz] DONE", out, flush=True)


if __name__ == "__main__":
    main()
