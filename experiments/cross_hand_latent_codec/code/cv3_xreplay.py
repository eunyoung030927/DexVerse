"""codec v3 cross-hand Kettle replay inputs + manifests (261116). New file; arm_prepare / arm_campaign are NOT modified.
Same pipeline as the 11/11 arm_latent_v1 cohort (E2* 77/200; Kettle 34/40): Sharpa demos eps 0-9 -> Enc_sharpa -> Dec_B,
target wrist = pad-surface weighted Kabsch (arm_pads.align_pair) smoothed (window 5) -> decode_replay.wrist_solve, target root
from the frozen 11/10 files; source commands checked bit-exact against frozen sharpa_native.npz. Only the output root
(artifacts/codec_v3/xreplay) and the task filter (Kettle) differ.
  prepare : python -m xhand_latent.cv3_xreplay prepare --condition <c> --checkpoint <ckpt>   (lerobot env python, as 11/11)
  manifest: python -m xhand_latent.cv3_xreplay manifest --conditions c1,c2 --inputs-dir inputs_initfix --phase <p>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .kinematics import HandFK

TARGETS = ["allegro", "leap", "shadow", "wuji"]

ROOT = Path("artifacts/codec_v3/xreplay")
TASK = "Dexverse-GraspKettle-v0"
EPS = list(range(10))


def prepare(args):
    from .arm_pads import PadFK, align_pair
    from .arm_prepare import OLD, decode, read_npz, relative, source_commands
    from .arm_train import load_arm
    from .decode_replay import canonical_world, wrist_solve
    from .reference_kin import make_T
    torch.set_num_threads(4)
    sf = PadFK("sharpa")
    fks = {h: PadFK(h) for h in TARGETS}
    codecs, _, payload = load_arm(args.checkpoint)
    out_dir = ROOT / "inputs" / TASK
    out_dir.mkdir(parents=True, exist_ok=True)
    src = source_commands(TASK, EPS, Path("data/replay"), sf)
    native = read_npz(OLD / "frozen" / TASK / "sharpa_native.npz")
    for e, s in src.items():
        k = f"ep{e:03d}"
        if not (np.array_equal(native[k], s["q"].astype(np.float32)) and np.array_equal(native[f"init_{k}"], s["initial"].astype(np.float32))):
            raise ValueError(f"source command reproduction failed {k}")
    report = {"condition": args.condition, "checkpoint": args.checkpoint, "checkpoint_step": payload["step"],
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), "smooth": args.smooth, "targets": {}}
    for target in TARGETS:
        tf = fks[target]
        frozen = read_npz(OLD / "frozen" / TASK / f"{target}_decoded_from_sharpa.npz")
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
            T = relative(res["R"].double().numpy(), res["t"].double().numpy(), args.smooth)
            source_palm = canonical_world(sf, all_q, make_T(s["root"][:3], s["root"][3:]))
            solved, errors = wrist_solve(tf, source_palm @ T, make_T(troot[:3], troot[3:]), q_t)
            solved = solved.astype(np.float32)
            result[k] = solved[1:]; result[f"init_{k}"] = solved[0]; result[f"root_{k}"] = np.asarray(troot, np.float32)
            diags.append({"episode": e, **errors, "kabsch_residual_mm_median": float(res["r2"].sqrt().median() * sf.scale * 1000)})
        path = out_dir / f"{target}_{args.condition}.npz"
        np.savez_compressed(path, **result)
        report["targets"][target] = {"file": str(path), "episodes": diags, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(json.dumps({"target": target, "condition": args.condition}), flush=True)
    (ROOT / f"prepare_{args.condition}.json").write_text(json.dumps(report, indent=2))


def manifest(args):
    phase = ROOT / args.phase
    for d in ("manifests", "replay", "logs"):
        (phase / d).mkdir(parents=True, exist_ok=True)
    jobs_out = []
    for hand in args.targets.split(","):
        fk = HandFK(hand)
        jobs = []
        for c in args.conditions.split(","):
            f = ROOT / args.inputs_dir / TASK / f"{hand}_{c}.npz"
            if not f.exists() and not (ROOT / "inputs" / TASK / f.name).exists():
                raise FileNotFoundError(f)
            jobs.append({"condition": f"arm_{c}", "file": str(f), "finger_reference": str(f), "finger_mode": "target_decoded",
                         "target_hand": hand, "role": c, "EE_definition": "pad-surface correspondence wrist (arm_latent_v1)",
                         "initial_state_protocol": "Source Sharpa scene; target root from target episode; codec-mapped robot initial state"})
        m = phase / "manifests" / f"{TASK}__{hand}.json"
        m.write_text(json.dumps({"protocol_id": f"codec_v3_xreplay_{args.phase}", "task": TASK, "source": "sharpa",
                                 "replay_hand": hand, "run": "codec_v3_xreplay", "wrist_joints": fk.spec["wrist_joints"],
                                 "jobs": jobs}, indent=2))
        jobs_out.append({"hand": hand, "manifest": str(m), "output": str(phase / "replay" / f"{TASK}__{hand}.jsonl"),
                         "expected_rows": len(jobs) * len(EPS)})
    (phase / "campaign.json").write_text(json.dumps({"phase": args.phase, "conditions": args.conditions.split(","),
                                                     "inputs_dir": args.inputs_dir, "jobs": jobs_out}, indent=2))
    print(json.dumps(jobs_out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--condition", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--smooth", type=int, default=5)
    m = sub.add_parser("manifest"); m.add_argument("--conditions", required=True); m.add_argument("--phase", required=True)
    m.add_argument("--inputs-dir", default="inputs_initfix"); m.add_argument("--targets", default=",".join(TARGETS))
    args = ap.parse_args()
    {"prepare": prepare, "manifest": manifest}[args.cmd](args)


if __name__ == "__main__":
    main()
