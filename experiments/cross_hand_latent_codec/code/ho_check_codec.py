"""ho check (a) (261115): HoCodec (ho_codec.py) vs P3HNoHotCodec bit-identity on sharpa/shadow/wuji, and the offline
Z/D encode->decode round trip of recorded pkl actions for allegro/leap (+ the 3 hands for reference); offsets sanity;
writes replay rows (<arm>_<hand>_ep<k>_action) for the in-sim check (b) (ho_eval_pooled.py --replay_npz). New file.
Container python, cwd /workspace/xhand/latent_codec_v1:
  P3H_PD2=artifacts/pooled3hand/p0b/ext/pd2_work2 python.sh -m xhand_latent.ho_check_codec [--f-task CKPT --tag T]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import load_trajectory, trajectory_path
from .ho_codec import HANDS_ALL, HoCodec
from .p3h_nh_codec import P3HNoHotCodec
from .p3h_policy_codec import DATA, E2, F_LOADER, F_TASK

TASK = "Dexverse-GraspKettle-v0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f-task", default=F_TASK)
    ap.add_argument("--hands", default=",".join(HANDS_ALL))
    ap.add_argument("--episodes", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--replay-eps", default="0,1,2")
    ap.add_argument("--tag", default="h1")
    ap.add_argument("--out", default="artifacts/pooled3hand/ho/check")
    ap.add_argument("--no-identity", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hands = args.hands.split(",")
    eps = [int(e) for e in args.episodes.split(",")]
    reps = [int(e) for e in args.replay_eps.split(",")]
    cc = HoCodec(E2, args.f_task, F_LOADER, DATA, hands=hands)
    rep = {"f_task": args.f_task, "hands": {}}
    if not args.no_identity:
        old = P3HNoHotCodec(E2, args.f_task, F_LOADER, DATA)
        ident = {}
        for h in ("sharpa", "shadow", "wuji"):
            if h not in hands:
                continue
            ep = load_trajectory(trajectory_path(DATA, TASK, h, 0))["episodes"][0]
            rootT = cc.root_T(np.asarray(ep["initial_state"]["articulation"]["robot"]["root_pose"])[0])
            a = np.asarray(ep["actions"], np.float32)
            st = a  # any rows in state layout (state names == action names for these hands) suffice for identity
            r = {}
            for arm in ("Z", "D"):
                r[arm] = {"state_eq": bool(np.array_equal(cc.encode_state(h, st, rootT, arm), old.encode_state(h, st, rootT, arm))),
                          "action_eq": bool(np.array_equal(cc.encode_action(h, a, rootT, arm), old.encode_action(h, a, rootT, arm))),
                          "offset_eq": bool(np.array_equal(cc.offset[h], old.offset[h]))}
            ident[h] = r
        rep["identity_vs_P3HNoHotCodec"] = ident
        print(json.dumps({"identity": ident}), flush=True)
    replay = {}
    for h in hands:
        fk = cc.fks[h]
        an = cc.names[h][0]
        meta = fk.spec.get("robotmeta")
        hr = {"n_action": len(an), "n_finger": len(fk.joint_names), "robotmeta": bool(meta),
              "offset": np.round(cc.offset[h], 5).tolist()}
        res, fin_res = [], []
        rt = {a: {"wrist_max": 0.0, "finger_mae": [], "finger_max": 0.0, "solve_mm": 0.0, "solve_deg": 0.0, "fail": 0}
              for a in ("Z", "D")}
        for e in eps:
            d = load_trajectory(trajectory_path(DATA, TASK, h, e))
            ep = d["episodes"][0]
            names = d["action_layout"]["robot_joint_names"]
            a = np.asarray(ep["actions"], np.float64)
            assert a.shape[1] == len(an), (h, a.shape, len(an))
            q0 = np.asarray(ep["initial_state"]["articulation"]["robot"]["joint_position"])[0]
            q0 = q0[[names.index(n) for n in an]]
            res.append(q0 - (a[0] + cc.offset[h]))  # initial state vs first command (PD tracking residual)
            rootT = cc.root_T(np.asarray(ep["initial_state"]["articulation"]["robot"]["root_pose"])[0])
            seed = cc.q_from_action(h, a)
            fi = cc.j_finger[h]
            if e in reps:  # arm C = recorded actions zero-padded (decode = slice): native replay baseline
                replay[f"C_{h}_ep{e}_action"] = cc.encode_action(h, a, rootT, "C")
            for arm in ("Z", "D"):
                v = cc.encode_action(h, a, rootT, arm)
                if e in reps:
                    replay[f"{arm}_{h}_ep{e}_action"] = v
                try:
                    act, _, st = cc.decode_action(h, v, rootT, seed, arm)
                except ValueError as exc:
                    rt[arm]["fail"] += 1
                    print(h, arm, e, "decode fail", str(exc)[:120], flush=True)
                    continue
                rt[arm]["wrist_max"] = max(rt[arm]["wrist_max"], float(np.abs(act[:, :6] - a[:, :6]).max()))
                fd = np.abs(act[:, fi] - a[:, fi])
                rt[arm]["finger_mae"].append(float(fd.mean()))
                rt[arm]["finger_max"] = max(rt[arm]["finger_max"], float(fd.max()))
                rt[arm]["solve_mm"] = max(rt[arm]["solve_mm"], float(st.get("max_wrist_position_error_mm", 0)))
                rt[arm]["solve_deg"] = max(rt[arm]["solve_deg"], float(st.get("max_wrist_rotation_error_deg", 0)))
        for arm in rt:
            rt[arm]["finger_mae"] = float(np.mean(rt[arm]["finger_mae"])) if rt[arm]["finger_mae"] else None
        res = np.stack(res)
        hr["init_minus_first_cmd"] = {"wrist_abs_med": np.round(np.median(np.abs(res[:, :6]), 0), 4).tolist(),
                                      "finger_abs_med_mean": float(np.median(np.abs(res[:, 6:]), 0).mean()),
                                      "finger_abs_max": float(np.abs(res[:, 6:]).max())}
        if meta:
            dflt = np.array(meta["action_offset"])[[meta["action_joint_names"].index(n) for n in an]]
            hr["offset_minus_robotmeta"] = np.round(cc.offset[h] - dflt, 5).tolist()
        hr["roundtrip_eps"] = eps
        hr["roundtrip"] = rt
        rep["hands"][h] = hr
        print(json.dumps({h: hr}), flush=True)
    np.savez_compressed(out / f"ho_dsrows_{args.tag}.npz", **replay)
    rep["replay_npz"] = str(out / f"ho_dsrows_{args.tag}.npz")
    (out / f"codec_check_{args.tag}.json").write_text(json.dumps(rep, indent=1))
    print("[ho_check] DONE", flush=True)


if __name__ == "__main__":
    main()
