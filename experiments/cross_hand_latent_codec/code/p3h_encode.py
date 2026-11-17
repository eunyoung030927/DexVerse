"""pooled3hand P1 step 1: encode per-frame observation.state / action of the GraspKettle LeRobot sets (sharpa, shadow,
wuji) for arms C, Z, D with the shared codec (p3h_policy_codec.P3HCodec), plus decode sanity checks. New file (261113).

Output: artifacts/pooled3hand/p1/encoded_<hand>.npz  (state_<ARM>, action_<ARM>, lengths, episode)
        artifacts/pooled3hand/p1/encode_sanity.json
Container python (/isaac-sim/python.sh), cwd /workspace/xhand/latent_codec_v1:
  python -m xhand_latent.p3h_encode
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
from .p3h_policy_codec import ARMS, E2, F_TASK, HANDS3, TASK, P3HCodec, action_dim, state_dim
from .p3h_roundtrip import codec_roundtrip, native_commands, pose_err

LEROBOT = Path("/workspace/xhand/data/dv-demos-v0/v0/teleop_lerobot/functional") / TASK
OUT = Path("artifacts/pooled3hand/p1")
PKL_ROOT = "/workspace/xhand/data/dv-demos-v0"  # all 50 eps (data/replay holds eps 0-9 only, same bytes)


def main():
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    cc = P3HCodec()
    rep = {"e2_ckpt": E2, "e2_sha256": hashlib.sha256(Path(E2).read_bytes()).hexdigest(),
           "f_task_ckpt": F_TASK, "f_task_md5": hashlib.md5(Path(F_TASK).read_bytes()).hexdigest(), "hands": {}}
    for hand in HANDS3:
        fk = cc.fks[hand]
        wi = [fk.reference.qidx[n] for n in fk.spec["wrist_joints"]]
        out = {f"{k}_{a}": [] for a in ARMS for k in ("state", "action")}
        lengths, roots = [], []
        hr = {"offset": cc.offset[hand].tolist(), "names_action": cc.names[hand][0], "episodes_checked": [],
              "C_max_abs_err": 0.0}
        acc = {a: {"fin": [], "fin_max": 0.0, "mm": [], "deg": [], "mm_max": 0.0, "deg_max": 0.0, "act_abs": []}
               for a in ("Z", "D")}
        native, _ = native_commands(hand, fk, list(range(10)), data=cc.data)
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
                assert s.shape == (len(st), state_dim(arm)) and a.shape == (len(ac), action_dim(arm)), (arm, s.shape, a.shape)
                assert np.isfinite(s).all() and np.isfinite(a).all()
                out[f"state_{arm}"].append(s)
                out[f"action_{arm}"].append(a)
            # --- sanity: C exact
            dec_c, _, _ = cc.decode_action(hand, out["action_C"][-1], rootT, None, "C")
            hr["C_max_abs_err"] = max(hr["C_max_abs_err"], float(np.abs(dec_c - ac).max()))
            if e < 10:
                q_cmd = cc.q_from_action(hand, ac)
                # identity vs P0b round-trip inputs: same q rows, same z
                assert np.array_equal(q_cmd, native[e]["all_q"][1:]), f"{hand} ep{e}: q_cmd != P0b native all_q"
                _, z_ref = codec_roundtrip(cc.codecs[hand], fk, native[e]["all_q"])
                z_max = float(np.abs(out["action_Z"][-1][:, 9:] - z_ref[1:]).max())
                hr["episodes_checked"].append({"ep": e, "q_cmd_equals_p0b": True, "z_vs_p0b_max_abs": z_max})
                P = canonical_world(fk, q_cmd, rootT)
                qf_true = cc.finger_rows(hand, q_cmd)
                for arm in ("Z", "D"):
                    act, q_dec, _ = cc.decode_action(hand, out[f"action_{arm}"][-1], rootT, q_cmd, arm)
                    fd = np.abs(cc.finger_rows(hand, q_dec) - qf_true)
                    d, r = pose_err(canonical_world(fk, q_dec, rootT), P)
                    A = acc[arm]
                    A["fin"].append(fd.mean())
                    A["fin_max"] = max(A["fin_max"], float(fd.max()))
                    A["mm"].append(d.mean()); A["deg"].append(r.mean())
                    A["mm_max"] = max(A["mm_max"], float(d.max())); A["deg_max"] = max(A["deg_max"], float(r.max()))
                    wa = [i for i, n in enumerate(cc.names[hand][0]) if n in fk.spec["wrist_joints"]]
                    A["act_abs"].append(np.abs(act[:, wa] - ac[:, wa]).max())
            print(f"[encode] {hand} ep{e:03d} T={len(st)}", flush=True)
        res = {k: np.concatenate(v) for k, v in out.items()}
        res["lengths"] = np.array(lengths)
        res["root7"] = np.stack(roots)
        np.savez_compressed(OUT / f"encoded_{hand}.npz", **res)
        hr["frames"] = int(sum(lengths))
        hr["root_unique"] = np.unique(np.round(np.stack(roots), 6), axis=0).tolist()
        hr["z_vs_p0b_max_abs"] = max(x["z_vs_p0b_max_abs"] for x in hr["episodes_checked"])
        for arm in ("Z", "D"):
            A = acc[arm]
            hr[f"decode_{arm}_eps0-9"] = {"finger_mae_rad": float(np.mean(A["fin"])), "finger_max_rad": A["fin_max"],
                                          "palm_mm_mean": float(np.mean(A["mm"])), "palm_mm_max": A["mm_max"],
                                          "palm_deg_mean": float(np.mean(A["deg"])), "palm_deg_max": A["deg_max"],
                                          "wrist_action_abs_err_max": float(np.max(A["act_abs"]))}
        rep["hands"][hand] = hr
        print(json.dumps({hand: {k: v for k, v in hr.items() if k not in ("episodes_checked", "names_action", "offset")}}),
              flush=True)
    (OUT / "encode_sanity.json").write_text(json.dumps(rep, indent=1))
    print("[encode] DONE", flush=True)


if __name__ == "__main__":
    main()
