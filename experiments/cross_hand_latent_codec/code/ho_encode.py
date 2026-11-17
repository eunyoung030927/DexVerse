"""HO H2 step 1 (261115): encode per-frame observation.state / action of the GraspKettle LeRobot sets of the H2 training
hands (shadow, allegro) for arms Z and D WITHOUT one-hot, with ho_codec.HoCodec (D uses the shadow+allegro f_task
ho/h2/f_task/f_task_best.pt). Same per-episode ops as p3h_encode.py (P1). New file.
Checks: shadow Z rows == P1 encoded_shadow (state_Z minus one-hot, action_Z) bit-exactly (same codec path);
root pose constant per hand; decode sanity on eps 0-9 (finger / palm error, as p3h_encode).
Output: artifacts/pooled3hand/ho/h2/encoded_<hand>.npz (state_Z, action_Z, state_D, action_D, lengths, root7),
        artifacts/pooled3hand/ho/h2/encode_sanity.json
Container python, cwd /workspace/xhand/latent_codec_v1:
  P3H_PD2=artifacts/pooled3hand/p0b/ext/pd2_work2 python.sh -m xhand_latent.ho_encode
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .data import load_trajectory, trajectory_path
from .decode_replay import canonical_world
from .ho_codec import E2, F_LOADER, F_TASK_SA, DATA, HoCodec, action_dim, state_dim
from .p3h_roundtrip import pose_err

TASK = "Dexverse-GraspKettle-v0"
HANDS = ("shadow", "allegro")
ARMS = ("Z", "D")
LEROBOT = Path("/workspace/xhand/data/dv-demos-v0/v0/teleop_lerobot/functional") / TASK
OUT = Path("artifacts/pooled3hand/ho/h2")
P1 = Path("artifacts/pooled3hand/p1")
PKL_ROOT = "/workspace/xhand/data/dv-demos-v0"


def main():
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    cc = HoCodec(E2, F_TASK_SA, F_LOADER, DATA, hands=HANDS)
    rep = {"e2_ckpt": E2, "f_task_ckpt": F_TASK_SA, "f_task_md5": hashlib.md5(Path(F_TASK_SA).read_bytes()).hexdigest(),
           "hands": {}}
    for hand in HANDS:
        fk = cc.fks[hand]
        out = {f"{k}_{a}": [] for a in ARMS for k in ("state", "action")}
        lengths, roots = [], []
        hr = {"offset": cc.offset[hand].tolist(), "names_action": cc.names[hand][0]}
        acc = {a: {"fin": [], "fin_max": 0.0, "mm": [], "deg": [], "mm_max": 0.0, "deg_max": 0.0} for a in ARMS}
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
            lengths.append(len(st))
            roots.append(root7)
            for arm in ARMS:
                s = cc.encode_state(hand, st, rootT, arm)
                a = cc.encode_action(hand, ac, rootT, arm)
                assert s.shape == (len(st), state_dim(arm)) and a.shape == (len(ac), action_dim(arm)), (arm, s.shape)
                assert np.isfinite(s).all() and np.isfinite(a).all()
                out[f"state_{arm}"].append(s)
                out[f"action_{arm}"].append(a)
            if e < 10:
                q_cmd = cc.q_from_action(hand, ac)
                P = canonical_world(fk, q_cmd, rootT)
                qf_true = cc.finger_rows(hand, q_cmd)
                for arm in ARMS:
                    act, q_dec, _ = cc.decode_action(hand, out[f"action_{arm}"][-1], rootT, q_cmd, arm)
                    fd = np.abs(cc.finger_rows(hand, q_dec) - qf_true)
                    d, r = pose_err(canonical_world(fk, q_dec, rootT), P)
                    A = acc[arm]
                    A["fin"].append(fd.mean()); A["fin_max"] = max(A["fin_max"], float(fd.max()))
                    A["mm"].append(d.mean()); A["deg"].append(r.mean())
                    A["mm_max"] = max(A["mm_max"], float(d.max())); A["deg_max"] = max(A["deg_max"], float(r.max()))
        res = {k: np.concatenate(v) for k, v in out.items()}
        res["lengths"] = np.array(lengths)
        res["root7"] = np.stack(roots)
        np.savez_compressed(OUT / f"encoded_{hand}.npz", **res)
        hr["frames"] = int(sum(lengths))
        hr["root_unique"] = np.unique(np.round(np.stack(roots), 6), axis=0).tolist()
        assert len(hr["root_unique"]) == 1, hr["root_unique"]
        for arm in ARMS:
            A = acc[arm]
            hr[f"decode_{arm}_eps0-9"] = {"finger_mae_rad": float(np.mean(A["fin"])), "finger_max_rad": A["fin_max"],
                                          "palm_mm_mean": float(np.mean(A["mm"])), "palm_mm_max": A["mm_max"],
                                          "palm_deg_mean": float(np.mean(A["deg"])), "palm_deg_max": A["deg_max"]}
        if (P1 / f"encoded_{hand}.npz").exists():
            E1 = np.load(P1 / f"encoded_{hand}.npz")
            hr["Z_equals_P1"] = {"state": bool(np.array_equal(res["state_Z"], E1["state_Z"][:, :-3])),
                                 "action": bool(np.array_equal(res["action_Z"], E1["action_Z"])),
                                 "lengths": bool(np.array_equal(res["lengths"], E1["lengths"]))}
            hr["D_vs_P1_3hand_f_task_max_abs"] = float(np.abs(res["action_D"] - E1["action_D"]).max())
            assert all(hr["Z_equals_P1"].values()), hr["Z_equals_P1"]
        rep["hands"][hand] = hr
        print(json.dumps({hand: {k: v for k, v in hr.items() if k not in ("names_action", "offset")}}), flush=True)
    (OUT / "encode_sanity.json").write_text(json.dumps(rep, indent=1))
    print("[ho_encode] DONE", flush=True)


if __name__ == "__main__":
    main()
