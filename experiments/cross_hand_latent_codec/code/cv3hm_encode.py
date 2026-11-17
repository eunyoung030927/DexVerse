"""CV3HM (261117): 3-hand (shadow + wuji + allegro) nohot Z rows for the codec-v3 DP test on
Dexverse-FunctionalHammerStrike-v0 (held-out test hands sharpa, leap). Copy of cv3dp_encode.py (unchanged original) with:
TASK = FunctionalHammerStrike, OUT = artifacts/codec_v3/cv3hm, and NO reference rows (no earlier Hammer Z set exists),
so the bit-exact E2*-vs-reference check of cv3dp is replaced by:
  - feature names: the Hammer LeRobot info.json state/action names == the names the codec takes from the Kettle replay
    meta (HoCodec reads names/offsets from data/replay GraspKettle; same robot configs) -> asserted.
  - wrist offset: the floating-wrist recorded offset inferred on Hammer pkl eps 0-9 (data.infer_recorded_offset, same
    rule) vs the codec's offset (inferred on Kettle eps 0-9); max abs difference reported for all 5 hands (incl. the
    held-out sharpa/leap, whose decode at eval uses the same codec offsets).
  - palm columns (0-8) are codec-independent: the cross-codec equality is asserted later by cv3hm_build (both npz).
  - decode sanity on eps 0-9: finger MAE of Dec_hand(z) vs commanded q (as cv3dp).
Output: artifacts/codec_v3/cv3hm/encoded_<tag>_<hand>.npz (state_Z, action_Z, lengths, root7), encode_<tag>.json.
Container python (lerobot env), cwd /workspace/xhand/latent_codec_v1:
  python -m xhand_latent.cv3hm_encode --tag e2 --ckpt artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt
  python -m xhand_latent.cv3hm_encode --tag v3wbcll --ckpt artifacts/codec_v3/runs/wb_cl_l/best.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .data import infer_recorded_offset, load_trajectory, trajectory_path
from .ho_codec import E2, HoCodec

TASK = "Dexverse-FunctionalHammerStrike-v0"
HANDS = ("shadow", "wuji", "allegro")
TEST_HANDS = ("sharpa", "leap")
LEROBOT = Path("/workspace/xhand/data/dv-demos-v0/v0/teleop_lerobot/functional") / TASK
PKL_ROOT = "/workspace/xhand/data/dv-demos-v0"
OUT = Path("artifacts/codec_v3/cv3hm")


def offset_check(cc, hand):
    an, _ = cc.names[hand]
    wr = cc.fks[hand].spec["wrist_joints"]
    off_h, info = infer_recorded_offset(PKL_ROOT, TASK, hand, an, cc.offset[hand], 10, wrist_names=wr)
    return {"max_abs_offset_diff_hammer_vs_codec": float(np.abs(np.asarray(off_h) - np.asarray(cc.offset[hand])).max()),
            "hammer_wrist_offset_spread_max": float(np.max(info["wrist_offset_spread"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default=E2)
    args = ap.parse_args()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    cc = HoCodec(args.ckpt, need_f=False, hands=HANDS)
    is_e2 = Path(args.ckpt).resolve() == Path(E2).resolve()
    rep = {"tag": args.tag, "task": TASK, "ckpt": args.ckpt, "ckpt_md5": hashlib.md5(Path(args.ckpt).read_bytes()).hexdigest(),
           "is_e2": is_e2, "hands": {}, "offset_check_test_hands": {}}
    for hand in HANDS:
        info = json.loads((LEROBOT / f"floating_{hand}_right/meta/info.json").read_text())["features"]
        an, sn = cc.names[hand]
        assert list(info["action"]["names"]) == list(an) and list(info["observation.state"]["names"]) == list(sn), hand
        st_all, ac_all, lengths, roots, fin = [], [], [], [], []
        for e in range(50):
            t = pq.read_table(LEROBOT / f"floating_{hand}_right/data/chunk-000/file-{e:03d}.parquet",
                              columns=["episode_index", "frame_index", "observation.state", "action"])
            assert set(t.column("episode_index").to_pylist()) == {e}
            assert t.column("frame_index").to_pylist() == list(range(t.num_rows))
            st = np.array(t.column("observation.state").to_pylist(), np.float32)
            ac = np.array(t.column("action").to_pylist(), np.float32)
            ep = load_trajectory(trajectory_path(PKL_ROOT, TASK, hand, e))["episodes"][0]
            assert len(ep["actions"]) == len(ac), (hand, e, len(ep["actions"]), len(ac))
            root7 = np.asarray(ep["initial_state"]["articulation"]["robot"]["root_pose"])[0]
            rootT = cc.root_T(root7)
            s = cc.encode_state(hand, st, rootT, "Z")
            a = cc.encode_action(hand, ac, rootT, "Z")
            assert s.shape == (len(st), 41) and a.shape == (len(ac), 41) and np.isfinite(s).all() and np.isfinite(a).all()
            if e < 10:
                q_cmd = cc.q_from_action(hand, ac)
                _, q_dec, _ = cc.decode_action(hand, a, rootT, q_cmd, "Z")
                fin.append(float(np.abs(cc.finger_rows(hand, q_dec) - cc.finger_rows(hand, q_cmd)).mean()))
            st_all.append(s); ac_all.append(a); lengths.append(len(st)); roots.append(root7)
        np.savez_compressed(OUT / f"encoded_{args.tag}_{hand}.npz", state_Z=np.concatenate(st_all),
                            action_Z=np.concatenate(ac_all), lengths=np.array(lengths), root7=np.stack(roots))
        hr = {"frames": int(sum(lengths)), "decode_finger_mae_rad_eps0-9": float(np.mean(fin)),
              "z_action_std_per_dim_mean": float(np.concatenate(ac_all)[:, 9:].std(0).mean()), **offset_check(cc, hand)}
        rep["hands"][hand] = hr
        print(json.dumps({hand: hr}), flush=True)
    ct = HoCodec(args.ckpt, need_f=False, hands=TEST_HANDS)
    for hand in TEST_HANDS:
        rep["offset_check_test_hands"][hand] = offset_check(ct, hand)
    print(json.dumps({"offset_check_test_hands": rep["offset_check_test_hands"]}), flush=True)
    (OUT / f"encode_{args.tag}.json").write_text(json.dumps(rep, indent=1))
    print(f"[cv3hm_encode] {args.tag} DONE", flush=True)


if __name__ == "__main__":
    main()
