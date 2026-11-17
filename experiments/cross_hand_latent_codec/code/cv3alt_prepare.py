"""cv3alt (261116): cross-hand retarget replay inputs (Sharpa -> allegro/leap/shadow/wuji) for any HandVAE checkpoint.
Read-only reuse of arm_prepare (kind=kabsch, stage=cohort): identical source commands (bit-exact check vs the frozen 11/10
native commands), identical finger decoding Dec_B(Enc_A(q_A)), identical pad-surface Kabsch wrist + smoothing + wrist_solve,
identical target roots. Only the output directory differs (artifacts/codec_v3alt/replay/inputs). New file.
  python.sh -m xhand_latent.cv3alt_prepare --condition <name> --checkpoint <ckpt> [--tasks Dexverse-GraspKettle-v0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .arm_pads import PadFK, align_pair
from .arm_prepare import OLD, TARGETS, decode, read_npz, relative, source_commands
from .arm_train import load_arm
from .decode_replay import canonical_world, wrist_solve
from .reference_kin import make_T

OUT = Path("artifacts/codec_v3alt/replay")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tasks", default="Dexverse-GraspKettle-v0")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--smooth", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(4)
    sf = PadFK("sharpa")
    fks = {h: PadFK(h) for h in TARGETS}
    codecs, _, payload = load_arm(args.checkpoint)
    report = {"condition": args.condition, "checkpoint": args.checkpoint,
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), "step": payload["step"],
              "tasks": {}}
    eps = list(range(args.episodes))
    for task in args.tasks.split(","):
        src = source_commands(task, eps, Path("data/replay"), sf)
        native = read_npz(OLD / "frozen" / task / "sharpa_native.npz")
        for e, s in src.items():
            k = f"ep{e:03d}"
            if not (np.array_equal(native[k], s["q"].astype(np.float32)) and
                    np.array_equal(native[f"init_{k}"], s["initial"].astype(np.float32))):
                raise ValueError(f"source command reproduction failed {task} {k}")
        (OUT / "inputs" / task).mkdir(parents=True, exist_ok=True)
        for target in TARGETS:
            tf = fks[target]
            frozen = read_npz(OLD / "frozen" / task / f"{target}_decoded_from_sharpa.npz")
            result = {"joint_names": np.array(tf.reference.joint_order)}
            diags = []
            for e, s in src.items():
                k = f"ep{e:03d}"
                all_q = s["all_q"]
                q_t, src_norm, _ = decode(codecs, sf, tf, all_q, target=target)
                troot = frozen[f"root_{k}"]
                tq = torch.tensor(q_t[:, [tf.reference.qidx[n] for n in tf.joint_names]], dtype=torch.float32)
                with torch.no_grad():
                    res = align_pair(sf, tf, sf.pads(src_norm), tf.pads(tq, normalized=False))
                Rrel, trel = res["R"].double().numpy(), res["t"].double().numpy()
                T = relative(Rrel, trel, args.smooth)
                source_palm = canonical_world(sf, all_q, make_T(s["root"][:3], s["root"][3:]))
                solved, errors = wrist_solve(tf, source_palm @ T, make_T(troot[:3], troot[3:]), q_t)
                solved = solved.astype(np.float32)
                result[k] = solved[1:]; result[f"init_{k}"] = solved[0]; result[f"root_{k}"] = np.asarray(troot, np.float32)
                diags.append({"episode": e, **errors,
                              "kabsch_residual_mm_median": float(res["r2"].sqrt().median() * sf.scale * 1000)})
            path = OUT / "inputs" / task / f"{target}_{args.condition}.npz"
            np.savez_compressed(path, **result)
            report["tasks"].setdefault(task, {})[target] = {"file": str(path), "episodes": diags,
                                                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            print(json.dumps({"task": task, "target": target, "condition": args.condition,
                              "kabsch_mm_med": float(np.median([d["kabsch_residual_mm_median"] for d in diags]))}), flush=True)
    (OUT / f"prepare_{args.condition}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
