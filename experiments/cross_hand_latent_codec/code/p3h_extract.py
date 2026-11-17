"""P0a (pooled3hand_design_261112 §4.1) step 1: per-step extraction from dv-demos-v0 GraspKettle native pkls. New file.

Per hand/episode/state: canonical palm ("wrist") world pose (decode_replay.canonical_world on recorded joint_position +
recorded root pose, i.e. same canonical palm frame as E2*/G3), finger q (pd2 Hand.jn order, physical rad), kettle
(rigid_object.object) and mug world poses, LeRobot touch_contact (8 slots little..thumb, palm_r/c/l; frame k = state k).
Container python, cwd /workspace/xhand/latent_codec_v1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .arm_pads import PadFK
from .data import load_trajectory
from .decode_replay import canonical_world
from .reference_kin import make_T

DATA = Path("/workspace/xhand/data/dv-demos-v0/v0")
TASK = "Dexverse-GraspKettle-v0"
HANDS3 = ("sharpa", "shadow", "wuji")
OUT = Path("artifacts/pooled3hand/p0a")


def pose7_to_T(p):
    p = np.asarray(p, np.float64).reshape(-1, 7)
    return np.stack([make_T(x[:3], x[3:]) for x in p])


def extract(hand, n_ep=50):
    fk = PadFK(hand)
    jn = list(fk.joint_names)  # finger joints (pd2 Hand.jn order)
    out = {"lengths": [], "wrist_w": [], "obj_w": [], "mug_w": [], "q": [], "contact": [], "root_w": [],
           "obj_init_w": [], "success": []}
    base = DATA / "teleop_lerobot/functional" / TASK / f"floating_{hand}_right"
    for e in range(n_ep):
        d = load_trajectory(DATA / "teleop/functional" / TASK / f"floating_{hand}_right" / f"ep{e:03d}.pkl")
        ep = d["episodes"][0]
        names = d["action_layout"]["robot_joint_names"]
        S = ep["states"]
        jp = np.stack([np.asarray(s["articulation"]["robot"]["joint_position"])[0] for s in S]).astype(np.float64)
        root = np.stack([np.asarray(s["articulation"]["robot"]["root_pose"])[0] for s in S]).astype(np.float64)
        obj = np.stack([np.asarray(s["rigid_object"]["object"]["root_pose"])[0] for s in S]).astype(np.float64)
        mug = np.stack([np.asarray(s["rigid_object"]["mug"]["root_pose"])[0] for s in S]).astype(np.float64)
        q_ref = jp[:, [names.index(n) for n in fk.reference.joint_order]]
        rootT = pose7_to_T(root)
        if np.abs(root - root[:1]).max() < 1e-6:
            P = canonical_world(fk, q_ref, rootT[0])
        else:
            P = np.stack([canonical_world(fk, q_ref[i:i + 1], rootT[i])[0] for i in range(len(q_ref))])
        t = pq.read_table(base / "data/chunk-000" / f"file-{e:03d}.parquet",
                          columns=["episode_index", "frame_index", "observation.touch_contact", "observation.state"])
        assert set(t.column("episode_index").to_pylist()) == {e}
        tc = np.array(t.column("observation.touch_contact").to_pylist(), np.float32)
        if len(tc) != ep["num_steps"] or len(S) != ep["num_steps"] + 1:
            raise ValueError(f"{hand} ep{e}: lerobot {len(tc)} vs pkl steps {ep['num_steps']} states {len(S)}")
        tc = np.concatenate([tc, tc[-1:]])  # state N (after last action) gets the last rendered frame's contact
        out["lengths"].append(len(S))
        out["wrist_w"].append(P.astype(np.float32))
        out["obj_w"].append(pose7_to_T(obj).astype(np.float32))
        out["mug_w"].append(pose7_to_T(mug).astype(np.float32))
        out["q"].append(jp[:, [names.index(n) for n in jn]].astype(np.float32))
        out["contact"].append(tc)
        out["root_w"].append(root.astype(np.float32))
        out["success"].append(bool(ep["success"]))
        print(f"[extract] {hand} ep{e:03d} T={len(S)}", flush=True)
    res = {k: np.concatenate(v) for k, v in out.items() if k not in ("lengths", "success", "obj_init_w")}
    res["lengths"] = np.array(out["lengths"])
    res["success"] = np.array(out["success"])
    res["joint_names"] = np.array(jn)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hands = sys.argv[1:] or HANDS3
    for h in hands:
        r = extract(h)
        np.savez_compressed(OUT / f"extract_{h}.npz", **r)
        print(f"[extract] saved {h} frames={len(r['q'])} eps={len(r['lengths'])} dof={r['q'].shape[1]}", flush=True)


if __name__ == "__main__":
    main()
