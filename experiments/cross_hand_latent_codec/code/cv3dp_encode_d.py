"""CV3DP amendment A3 (261117): D wrist columns (EE_common = canonical palm @ inv(f_task(q_finger))) for the 3-hand
(shadow + wuji + allegro) cv3dp sets Dw3 / De3. New file; cv3dp_encode.py / ho_codec / p3h_* are imported unchanged.
EE_common is codec-independent (f_task is evaluated on the RECORDED finger q, as p3h_policy_codec.encode_q arm D), so
one palm block per hand serves both Dw3 (wb_cl_l z) and De3 (E2* z); the z columns are taken from the existing
encoded_<tag>_<hand>.npz (identical rows to wsa_Zw3 / wsa_Ze3).
f_task = artifacts/pooled3hand/p0a/f_task_best.pt (P0a, trained on Kettle sharpa/shadow/wuji; allegro out-of-set).
Per hand, GraspKettle eps 0-49 (same parquet + pkl root pose as cv3dp_encode.py):
  state_D9  = HoCodec.encode_state(hand, state, rootT, 'D')[:, :9]
  action_D9 = HoCodec.encode_action(hand, action, rootT, 'D')[:, :9]
Checks (artifacts/codec_v3/cv3dp/encode_dpalm.json):
  - E2* Z rows of this codec instance == encoded_e2_<hand>.npz (row alignment; same op path).
  - D rows (E2* z) z columns == encoded_e2 z; palm D cols vs pooled3hand_nohot/kettle3_D (shadow, wuji; same source eps 0-49).
  - held-out round trip (sharpa, leap demos pkl eps 0-9): action -> D vec (codec tag) -> decode_action(arm D, seed = q_cmd)
    -> canonical palm error vs the recorded commanded palm (mm / deg) + finger MAE; Z for reference.
Output: artifacts/codec_v3/cv3dp/encoded_dpalm_<hand>.npz (state_D9, action_D9, lengths, root7).
Isaac python in the xhand container (same as P1/H2 D encoding), cwd /workspace/xhand/latent_codec_v1:
  P3H_PD2=artifacts/pooled3hand/p0b/ext/pd2_work2 /workspace/isaaclab/_isaac_sim/python.sh -m xhand_latent.cv3dp_encode_d
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .cv3dp_encode import LEROBOT, OUT, PKL_ROOT, TASK, ref_rows
from .data import load_trajectory, trajectory_path
from .decode_replay import canonical_world
from .ho_codec import DATA, E2, F_LOADER, HoCodec
from .p3h_roundtrip import pose_err

HANDS = ("shadow", "wuji", "allegro")
HELD = ("sharpa", "leap")
F_TASK = "artifacts/pooled3hand/p0a/f_task_best.pt"
REF_D = "/workspace/xhand/datasets/pooled3hand_nohot/kettle3_D"
CODECS = {"e2": E2, "v3wbcll": "artifacts/codec_v3/runs/wb_cl_l/best.pt"}
DEMOS = "artifacts/codec_v3/cv3dp/demos/Dexverse-GraspKettle-v0_floating_{}_right.pkl"


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def decode_rows(cc, hand, vec, rootT, seed, arm):
    """batch decode; on a wrist-solve ValueError fall back to rows one by one (failed rows -> NaN)."""
    try:
        _, q, _ = cc.decode_action(hand, vec, rootT, seed, arm)
        return q, 0
    except ValueError:
        out, fails = np.full_like(seed, np.nan), 0
        for i in range(len(vec)):
            try:
                out[i] = cc.decode_action(hand, vec[i:i + 1], rootT, seed[i], arm)[1][0]
            except ValueError:
                fails += 1
        return out, fails


def main():
    torch.set_num_threads(8)
    t0 = time.time()
    cc = HoCodec(E2, F_TASK, F_LOADER, DATA, hands=HANDS)
    refD = ref_rows(REF_D, ("shadow", "wuji"))
    rep = {"f_task": F_TASK, "f_task_md5": md5(F_TASK), "e2": E2, "ref_D": REF_D, "hands": {}, "roundtrip": {}}
    for hand in HANDS:
        e2 = np.load(OUT / f"encoded_e2_{hand}.npz")
        off = np.r_[0, np.cumsum(e2["lengths"])]
        sD, aD, lengths, roots = [], [], [], []
        z_eq, zrow_eq, refcmp = True, True, []
        for e in range(50):
            t = pq.read_table(LEROBOT / f"floating_{hand}_right/data/chunk-000/file-{e:03d}.parquet",
                              columns=["episode_index", "frame_index", "observation.state", "action"])
            assert set(t.column("episode_index").to_pylist()) == {e}
            st = np.array(t.column("observation.state").to_pylist(), np.float32)
            ac = np.array(t.column("action").to_pylist(), np.float32)
            ep = load_trajectory(trajectory_path(PKL_ROOT, TASK, hand, e))["episodes"][0]
            root7 = np.asarray(ep["initial_state"]["articulation"]["robot"]["root_pose"])[0]
            assert np.array_equal(root7, e2["root7"][e])
            rootT = cc.root_T(root7)
            sz, az = cc.encode_state(hand, st, rootT, "Z"), cc.encode_action(hand, ac, rootT, "Z")
            sd, ad = cc.encode_state(hand, st, rootT, "D"), cc.encode_action(hand, ac, rootT, "D")
            sl = slice(off[e], off[e + 1])
            assert len(st) == e2["lengths"][e]
            zrow_eq &= bool(np.array_equal(sz, e2["state_Z"][sl]) and np.array_equal(az, e2["action_Z"][sl]))
            z_eq &= bool(np.array_equal(sd[:, 9:], e2["state_Z"][sl][:, 9:]) and np.array_equal(ad[:, 9:], e2["action_Z"][sl][:, 9:]))
            assert np.isfinite(sd).all() and np.isfinite(ad).all()
            if (hand, e) in refD:
                rs, ra = refD[(hand, e)]
                refcmp.append({"ep": e, "state_eq": bool(np.array_equal(sd, rs)), "action_eq": bool(np.array_equal(ad, ra)),
                               "palm_max_abs": float(max(np.abs(sd[:, :9] - rs[:, :9]).max(), np.abs(ad[:, :9] - ra[:, :9]).max())),
                               "z_max_abs": float(max(np.abs(sd[:, 9:] - rs[:, 9:]).max(), np.abs(ad[:, 9:] - ra[:, 9:]).max()))})
            sD.append(sd[:, :9]); aD.append(ad[:, :9]); lengths.append(len(st)); roots.append(root7)
        np.savez_compressed(OUT / f"encoded_dpalm_{hand}.npz", state_D9=np.concatenate(sD), action_D9=np.concatenate(aD),
                            lengths=np.array(lengths), root7=np.stack(roots))
        SD, SZ = np.concatenate(aD), e2["action_Z"][:, :9]
        hr = {"frames": int(sum(lengths)), "Z_rows_equal_encoded_e2": zrow_eq, "D_z_cols_equal_encoded_e2": z_eq,
              "action_D_minus_Z_palm_pos_mean_abs_m": float(np.abs(SD[:, :3] - SZ[:, :3]).mean()),
              "action_D_pos_std": SD[:, :3].std(0).round(5).tolist(), "action_Z_pos_std": SZ[:, :3].std(0).round(5).tolist()}
        if refcmp:
            hr["vs_kettle3_D"] = {"episodes": len(refcmp), "all_rows_bit_equal": all(r["state_eq"] and r["action_eq"] for r in refcmp),
                                  "palm_max_abs": max(r["palm_max_abs"] for r in refcmp),
                                  "z_max_abs": max(r["z_max_abs"] for r in refcmp)}
        assert zrow_eq and z_eq, hr
        rep["hands"][hand] = hr
        print(json.dumps({hand: hr}), flush=True)

    # held-out round trip
    for tag, ck in CODECS.items():
        hc = HoCodec(ck, F_TASK, F_LOADER, DATA, hands=HELD)
        rep["roundtrip"][tag] = {"ckpt": ck, "ckpt_md5": md5(ck)}
        for hand in HELD:
            with open(DEMOS.format(hand), "rb") as f:
                payload = pickle.load(f)
            fk = hc.fks[hand]
            res = {}
            for arm in ("D", "Z"):
                P, R, F, fails, n = [], [], [], 0, 0
                for k in range(10):
                    ep = payload["episodes"][k]
                    ac = np.asarray(ep["actions"], np.float32)
                    root7 = np.asarray(ep["initial_state"]["articulation"]["robot"]["root_pose"], np.float64)[0]
                    rootT = hc.root_T(root7)
                    q_cmd = hc.q_from_action(hand, ac)
                    vec = hc.encode_action(hand, ac, rootT, arm)
                    q_dec, nf = decode_rows(hc, hand, vec, rootT, q_cmd, arm)
                    fails += nf; n += len(ac)
                    ok = np.isfinite(q_dec).all(1)
                    d, r = pose_err(canonical_world(fk, q_dec[ok], rootT), canonical_world(fk, q_cmd[ok], rootT))
                    P.append(d); R.append(r)
                    F.append(np.abs(hc.finger_rows(hand, q_dec[ok]) - hc.finger_rows(hand, q_cmd[ok])).mean(1))
                P, R, F = np.concatenate(P), np.concatenate(R), np.concatenate(F)
                res[arm] = {"frames": n, "solve_fail_rows": fails,
                            "pos_mm": {"mean": float(P.mean()), "median": float(np.median(P)), "p95": float(np.percentile(P, 95)), "max": float(P.max())},
                            "rot_deg": {"mean": float(R.mean()), "median": float(np.median(R)), "p95": float(np.percentile(R, 95)), "max": float(R.max())},
                            "finger_mae_rad": float(F.mean())}
            rep["roundtrip"][tag][hand] = res
            print(json.dumps({tag: {hand: res}}), flush=True)
    rep["wall_s"] = round(time.time() - t0, 1)
    (OUT / "encode_dpalm.json").write_text(json.dumps(rep, indent=1))
    print("[cv3dp_encode_d] DONE", flush=True)


if __name__ == "__main__":
    main()
