"""codec v2 (cv2, 261115) V0 / "+c": per-hand offset removal baked into a HandVAE checkpoint. New file.
o_h = mean_s z_h(SYN calibration scene s) - mean_k mean_s z_k  (clean synergy scenes, seed --seed, all hands in the ckpt)
Exact reparameterisation: Enc mean bias -= o_h ; Dec first Linear bias += W0 @ o_h  ->  z' = z - o_h, Dec'(z') = Dec(z).
Checks: max |Dec'(Enc'(q)) - Dec(Enc(q))| and max |z' - (z - o_h)| on play val.
Container python, cwd /workspace/xhand/latent_codec_v1:
  python.sh -m xhand_latent.cv2_center --ckpt <in best.pt> --out <dir>/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .arm_train import load_arm
from .cv2_train import SynSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--play", default="artifacts/play_v1")
    ap.add_argument("--scenes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(4)
    codecs, fks, payload = load_arm(args.ckpt)
    hands = list(fks)
    play = {h: np.load(Path(args.play) / f"{h}.npz", allow_pickle=True) for h in hands}
    syn = SynSampler(play, hands, "cpu", noise=0.0)
    q = syn.sample(args.scenes, torch.Generator().manual_seed(args.seed))
    with torch.no_grad():
        m = {h: codecs[h].encode(q[h])[0].mean(0) for h in hands}
        ref = torch.stack(list(m.values())).mean(0)
        off = {h: m[h] - ref for h in hands}
        val = {h: torch.tensor(play[h]["val"][:2048]) for h in hands}
        before = {h: (codecs[h].encode(val[h])[0], codecs[h].decoder(codecs[h].encode(val[h])[0])) for h in hands}
        for h in hands:
            c = codecs[h]
            c.mean.bias -= off[h]
            l0 = c.decoder[0]
            l0.bias += l0.weight @ off[h]
        check = {}
        for h in hands:
            z2 = codecs[h].encode(val[h])[0]
            d2 = codecs[h].decoder(z2)
            check[h] = {"z_shift_err": float((z2 - (before[h][0] - off[h])).abs().max()),
                        "decode_err": float((d2 - before[h][1]).abs().max()), "offset_norm": float(off[h].norm())}
    print(json.dumps(check), flush=True)
    assert all(v["decode_err"] < 1e-4 and v["z_shift_err"] < 1e-4 for v in check.values()), check
    cfg = dict(payload["config"])
    cfg["cv2_center"] = {"source": args.ckpt, "scenes": args.scenes, "seed": args.seed,
                         "offsets": {h: off[h].tolist() for h in hands}, "check": check}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "step": payload.get("step"), "state_dict": codecs.state_dict(),
                "validation": payload.get("validation")}, out)
    (out.parent / "center.json").write_text(json.dumps(cfg["cv2_center"], indent=1))
    print("[cv2_center] DONE", out, flush=True)


if __name__ == "__main__":
    main()
