"""CV3DP (261116): 3-hand (shadow + wuji + allegro) nohot Z rows for the codec-v3 DP test (held-out test hands sharpa,
leap). Copy of cv2dp_encode.py (unchanged original) with only: HANDS += allegro, OUT = artifacts/codec_v3/cv3dp, and the
reference rows taken per hand (shadow/wuji: pooled3hand_nohot/kettle3_Z; allegro: ho2hand/sa_Z, the E2* H2 shadow+allegro
nohot Z set). ho_codec / p3h_* are imported unchanged.
For each hand and GraspKettle eps 0-49 (dv-demos-v0 teleop LeRobot parquet + pkl root pose, same per-episode ops as
ho_encode.py): Z state/action rows = [canonical palm pos 3, rot6d 6, z 32 = Enc_hand mean(q_finger)] with codec --ckpt.
Checks (written to artifacts/codec_v3/cv3dp/encode_<tag>.json):
  - reference rows for the same hand/episode; E2* (--tag e2) must reproduce them bit-exactly; any other codec must
    reproduce the palm columns (0-8) bit-exactly (palm = canonical_world(q), codec-independent) and only differ in z.
  - decode sanity on eps 0-9: finger MAE of Dec_hand(z) vs commanded q (as ho_encode).
Output: artifacts/codec_v3/cv3dp/encoded_<tag>_<hand>.npz (state_Z, action_Z, lengths, root7).
Container python (lerobot env), cwd /workspace/xhand/latent_codec_v1:
  python -m xhand_latent.cv3dp_encode --tag e2 --ckpt artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt
  python -m xhand_latent.cv3dp_encode --tag v3wbcll --ckpt artifacts/codec_v3/runs/wb_cl_l/best.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .data import load_trajectory, trajectory_path
from .ho_codec import E2, HoCodec

TASK = "Dexverse-GraspKettle-v0"
HANDS = ("shadow", "wuji", "allegro")
LEROBOT = Path("/workspace/xhand/data/dv-demos-v0/v0/teleop_lerobot/functional") / TASK
PKL_ROOT = "/workspace/xhand/data/dv-demos-v0"
OUT = Path("artifacts/codec_v3/cv3dp")
REF = {"shadow": "/workspace/xhand/datasets/pooled3hand_nohot/kettle3_Z",
       "wuji": "/workspace/xhand/datasets/pooled3hand_nohot/kettle3_Z",
       "allegro": "/workspace/xhand/datasets/ho2hand/sa_Z"}


def ref_rows(root, hands):
    """{(hand, src_ep): (state, action)} from a pooled LeRobot set with meta/p3h_pooled.json."""
    root = Path(root)
    meta = json.loads((root / "meta/p3h_pooled.json").read_text())
    out = {}
    for e in meta["episodes"]:
        if e["hand"] not in hands:
            continue
        t = pq.read_table(root / f"data/chunk-000/file-{e['episode_index']:03d}.parquet",
                          columns=["episode_index", "observation.state", "action"])
        assert set(t.column("episode_index").to_pylist()) == {e["episode_index"]}
        out[(e["hand"], e["source_episode"])] = (np.array(t.column("observation.state").to_pylist(), np.float32),
                                                 np.array(t.column("action").to_pylist(), np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default=E2)
    args = ap.parse_args()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    cc = HoCodec(args.ckpt, need_f=False, hands=HANDS)
    ref = {}
    for root in sorted(set(REF.values())):
        ref.update(ref_rows(root, tuple(h for h in HANDS if REF[h] == root)))
    is_e2 = Path(args.ckpt).resolve() == Path(E2).resolve()
    rep = {"tag": args.tag, "ckpt": args.ckpt, "ckpt_md5": hashlib.md5(Path(args.ckpt).read_bytes()).hexdigest(),
           "ref_rows": REF, "is_e2": is_e2, "hands": {}}
    for hand in HANDS:
        st_all, ac_all, lengths, roots = [], [], [], []
        palm_eq, full_eq, zdiff, fin = True, True, [], []
        for e in range(50):
            t = pq.read_table(LEROBOT / f"floating_{hand}_right/data/chunk-000/file-{e:03d}.parquet",
                              columns=["episode_index", "frame_index", "observation.state", "action"])
            assert set(t.column("episode_index").to_pylist()) == {e}
            assert t.column("frame_index").to_pylist() == list(range(t.num_rows))
            st = np.array(t.column("observation.state").to_pylist(), np.float32)
            ac = np.array(t.column("action").to_pylist(), np.float32)
            ep = load_trajectory(trajectory_path(PKL_ROOT, TASK, hand, e))["episodes"][0]
            root7 = np.asarray(ep["initial_state"]["articulation"]["robot"]["root_pose"])[0]
            rootT = cc.root_T(root7)
            s = cc.encode_state(hand, st, rootT, "Z")
            a = cc.encode_action(hand, ac, rootT, "Z")
            assert s.shape == (len(st), 41) and a.shape == (len(ac), 41) and np.isfinite(s).all() and np.isfinite(a).all()
            rs, ra = ref[(hand, e)]
            assert rs.shape == s.shape and ra.shape == a.shape, (hand, e, rs.shape, s.shape)
            palm_eq &= bool(np.array_equal(s[:, :9], rs[:, :9]) and np.array_equal(a[:, :9], ra[:, :9]))
            full_eq &= bool(np.array_equal(s, rs) and np.array_equal(a, ra))
            zdiff.append(float(np.abs(a[:, 9:] - ra[:, 9:]).mean()))
            if e < 10:
                q_cmd = cc.q_from_action(hand, ac)
                _, q_dec, _ = cc.decode_action(hand, a, rootT, q_cmd, "Z")
                fin.append(float(np.abs(cc.finger_rows(hand, q_dec) - cc.finger_rows(hand, q_cmd)).mean()))
            st_all.append(s); ac_all.append(a); lengths.append(len(st)); roots.append(root7)
        np.savez_compressed(OUT / f"encoded_{args.tag}_{hand}.npz", state_Z=np.concatenate(st_all),
                            action_Z=np.concatenate(ac_all), lengths=np.array(lengths), root7=np.stack(roots))
        hr = {"frames": int(sum(lengths)), "ref": REF[hand], "palm_cols_equal_ref": palm_eq, "rows_equal_ref": full_eq,
              "z_mean_abs_diff_vs_ref_action": float(np.mean(zdiff)), "decode_finger_mae_rad_eps0-9": float(np.mean(fin)),
              "z_action_std_per_dim_mean": float(np.concatenate(ac_all)[:, 9:].std(0).mean())}
        assert palm_eq, f"{hand}: palm columns differ from the reference set"
        if is_e2:
            assert full_eq, f"{hand}: E2* rows differ from the reference set"
        rep["hands"][hand] = hr
        print(json.dumps({hand: hr}), flush=True)
    (OUT / f"encode_{args.tag}.json").write_text(json.dumps(rep, indent=1))
    print(f"[cv3dp_encode] {args.tag} DONE", flush=True)


if __name__ == "__main__":
    main()
