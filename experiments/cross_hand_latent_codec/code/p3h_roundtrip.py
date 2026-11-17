"""P0b (pooled3hand_design_261112 sec.6 rule 4): same-hand round-trip (encode -> decode) replay inputs. New file.

Per hand (sharpa/shadow/wuji) and GraspKettle episode 0-9, writes replay_sim.py inputs (decoded npz format) per arm:
  <out>/decoded/<ARM>/<task>/<hand>/<hand>_native.npz   native commands, bit-exact decode_replay.main convention
  <out>/decoded/<ARM>/<task>/<hand>/<hand>_codec.npz    round-trip commands (initial row/root = native, untouched)
Arms (rows = command rows; initial state is the pkl initial_state for every arm):
  Z: fingers q -> E2* enc (mean) -> z -> E2* dec -> q_hat; wrist joints unchanged (as decode_replay.convert same-hand).
  E: P = canonical palm (decode_replay.canonical_world); T_pF(q) = F frame in palm coords (arm_pads: pad-surface contacts of
     index/middle/ring vs thumb, origin = weighted pinch-contact centre, x = thumb->finger, z = palm normal orthogonalised).
     EE_common = P @ T_pF(q);  P_hat = EE_common @ T_pF(q_hat)^-1;  fingers = q_hat; floating wrist re-solved (wrist_solve).
  D: EE_common = P @ f(q)^-1;  P_hat = EE_common @ f(q_hat);  f = learned f_task (P0a), via --f-task-ckpt/--f-task-loader.
Sanity (E, D): with q_hat = q the solve must reproduce the recorded wrist (max joint / palm-pose error reported).
Codec input = clipped command rows (decode_replay convention); native npz = unclipped commands (decode_replay).
Container python (no Isaac), cwd /workspace/xhand/latent_codec_v1:
  python -m xhand_latent.p3h_roundtrip --arms Z,E
  python -m xhand_latent.p3h_roundtrip --arms D --f-task-ckpt <f_task_best.pt> --f-task-loader <p3h_offsets.py>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .arm_pads import F_FINGERS, contacts, f_frame, pair_weights
from .arm_train import load_arm
from .data import feature_names, infer_recorded_offset, load_trajectory, trajectory_path
from .decode_replay import canonical_world, wrist_solve
from .reference_kin import make_T, so3_log

TASK = "Dexverse-GraspKettle-v0"
HANDS = ("sharpa", "shadow", "wuji")
E2 = "artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt"


def inv(T):
    out = np.zeros_like(T)
    R, t = T[..., :3, :3], T[..., :3, 3]
    out[..., :3, :3] = np.swapaxes(R, -1, -2)
    out[..., :3, 3] = -np.einsum("...ji,...j->...i", R, t)
    out[..., 3, 3] = 1
    return out


def pose_err(A, B):
    d = np.linalg.norm(A[:, :3, 3] - B[:, :3, 3], axis=-1) * 1000
    r = np.rad2deg(np.linalg.norm(so3_log(A[:, :3, :3] @ np.swapaxes(B[:, :3, :3], -1, -2)), axis=-1))
    return d, r


def native_commands(hand, fk, episodes, data="data/replay"):
    """Exactly decode_replay.main: q = recorded actions + offset (wrist offset inferred on eps 0-9)."""
    action_names, _ = feature_names(data, TASK, hand)
    meta = fk.spec.get("robotmeta")
    default = np.array(meta["action_offset"]) if meta else np.array([.3 if n == "z_translation_joint" else 0 for n in action_names])
    if meta:
        default = default[[meta["action_joint_names"].index(n) for n in action_names]]
    offset, stats = infer_recorded_offset(data, TASK, hand, action_names, default, 10, wrist_names=fk.spec["wrist_joints"])
    out = {}
    for e in episodes:
        d = load_trajectory(trajectory_path(data, TASK, hand, e))
        ep = d["episodes"][0]
        names = d["action_layout"]["robot_joint_names"]
        a = np.asarray(ep["actions"], dtype=np.float64)
        q = np.zeros((len(a), len(fk.reference.joint_order)))
        for i, n in enumerate(action_names):
            q[:, fk.reference.qidx[n]] = (a + offset)[:, i]
        clipped = np.clip(q, fk.reference.lower, fk.reference.upper)
        root = np.asarray(ep["initial_state"]["articulation"]["robot"]["root_pose"])[0]
        init = np.asarray(ep["initial_state"]["articulation"]["robot"]["joint_position"])[0]
        initial = init[[names.index(n) for n in fk.reference.joint_order]]
        out[e] = {"q": q, "all_q": np.concatenate([initial[None], clipped]), "initial": initial, "root": root,
                  "clamp_fraction": float(np.mean(np.abs(q - clipped) > 1e-6))}
    return out, {"offset": offset.tolist(), **{k: v for k, v in stats.items() if k != "offset"}}


def finger_rows(fk, q_rows):
    return q_rows[:, [fk.reference.qidx[n] for n in fk.joint_names]]


@torch.no_grad()
def codec_roundtrip(codec, fk, all_q):
    qf = torch.tensor(finger_rows(fk, all_q), dtype=torch.float32)
    z = codec.encode(fk.normalize(qf).clamp(-1, 1))[0]
    return fk.denormalize(codec.decoder(z)).double().numpy(), z.numpy()


@torch.no_grad()
def T_palm_F(fk, qf):
    """F frame pose in canonical palm coords from physical finger rows (n, dof) -> (n,4,4)."""
    pts, nrm, _ = fk.pads(torch.tensor(qf, dtype=torch.float32), normalized=False)
    cf, ct, _, _ = contacts(pts, nrm, F_FINGERS)
    RF, o = f_frame(cf, ct, pair_weights(cf, ct))
    T = np.tile(np.eye(4), (len(qf), 1, 1))
    T[:, :3, :3] = RF.double().numpy()
    T[:, :3, 3] = o.double().numpy()
    return T


def load_f_task(ckpt, loader, kind="f_task"):
    """D adapter for the P0a loader (work2 xhand_latent/p3h_offsets.py, 261112 version):
         Offsets(kind, path).offset(hand, q) -> (B,4,4) torch, q = physical finger rad in PadFK.joint_names (= Hand.jn) order,
         convention EE_common = EE @ inv(off), EE = EE_common @ off  (identical to the design's f(q,a)).
       Its pd2 dependency must be work2's copy (work1's pd2/train_f.py differs): set P3H_PD2 to that copy.
       Fallback: a module-level load_f_task(ckpt) / load(ckpt) returning callable(hand, q) -> (n,4,4)."""
    spec = importlib.util.spec_from_file_location("p3h_offsets_ext", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p3h_offsets_ext"] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "Offsets"):
        off = mod.Offsets(kind, ckpt)
        return lambda hand, q: off.offset(hand, np.asarray(q, np.float32)).double().cpu().numpy()
    for name in ("load_f_task", "load"):
        if hasattr(mod, name):
            return getattr(mod, name)(ckpt)
    raise AttributeError(f"{loader}: expected Offsets, load_f_task(ckpt) or load(ckpt)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="Z,E")
    p.add_argument("--episodes", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--hands", default=",".join(HANDS))
    p.add_argument("--checkpoint", default=E2)
    p.add_argument("--out", default="artifacts/pooled3hand/p0b")
    p.add_argument("--f-task-ckpt")
    p.add_argument("--f-task-loader", help="path to P0a p3h_offsets.py (copied to work1)")
    p.add_argument("--f-task-kind", default="f_task", help="Offsets kind for arm D (f_task; K/F/g2f for checks)")
    p.add_argument("--tag", help="output tag for roundtrip_prep_<tag>.json/csv and decoded/<tag> (default = arms joined)")
    p.add_argument("--native-check", default="artifacts/decoded/all_five_v1",
                   help="decode_replay native npz to compare bit-exactly (eps 0-9)")
    args = p.parse_args()
    torch.set_num_threads(4)
    arms = args.arms.split(",")
    episodes = [int(e) for e in args.episodes.split(",")]
    out = Path(args.out)
    codecs, fks, payload = load_arm(args.checkpoint)
    f_task = load_f_task(args.f_task_ckpt, args.f_task_loader, args.f_task_kind) if "D" in arms else None
    report = {"checkpoint": args.checkpoint, "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "checkpoint_step": payload.get("step"), "arms": arms, "episodes": episodes, "hands": {},
              "f_task_ckpt": args.f_task_ckpt, "f_task_loader": args.f_task_loader}
    csv_rows = []
    for hand in args.hands.split(","):
        fk = fks[hand]
        src, off = native_commands(hand, fk, episodes)
        fi = [fk.reference.qidx[n] for n in fk.joint_names]
        wi = [fk.reference.qidx[n] for n in fk.spec["wrist_joints"]]
        native = {"joint_names": np.array(fk.reference.joint_order)}
        results = {a: {"joint_names": np.array(fk.reference.joint_order)} for a in arms}
        hrep = {"offset": off, "native_check_vs_all_five_v1": None, "episodes": {}}
        chk = Path(args.native_check) / TASK / hand / f"{hand}_native.npz"
        ref = np.load(chk) if chk.exists() else None
        for e in episodes:
            k = f"ep{e:03d}"
            s = src[e]
            native[k], native[f"init_{k}"], native[f"root_{k}"] = (s["q"].astype(np.float32), s["initial"].astype(np.float32),
                                                                 s["root"].astype(np.float32))
            if ref is not None and k in ref.files:
                same = all(np.array_equal(ref[x], native[x]) for x in (k, f"init_{k}", f"root_{k}"))
                prev = hrep["native_check_vs_all_five_v1"]
                hrep["native_check_vs_all_five_v1"] = same if prev in (None, True) else False
            all_q = s["all_q"]
            rootT = make_T(s["root"][:3], s["root"][3:])
            qhat_f, _ = codec_roundtrip(codecs[hand], fk, all_q)
            q_z = all_q.copy()
            q_z[:, fi] = qhat_f
            P = canonical_world(fk, all_q, rootT)
            f_true = finger_rows(fk, all_q)
            diff = np.abs(qhat_f[1:] - f_true[1:])
            er = {"n_rows": int(len(all_q) - 1), "clamp_fraction": s["clamp_fraction"],
                  "finger_abs_err_rad_mean": float(diff.mean()),
                  "finger_rmse_rad": float(np.sqrt(np.square(diff).mean())),
                  "finger_abs_err_rad_p95": float(np.percentile(diff, 95)),
                  "finger_abs_err_rad_max": float(diff.max())}
            for arm in arms:
                extra = {}
                if arm == "Z":
                    q_arm = q_z
                else:
                    if arm == "E":
                        A_true, A_hat = T_palm_F(fk, f_true), T_palm_F(fk, qhat_f)
                        common = P @ A_true
                        desired = common @ inv(A_hat)
                        sanity_desired = common @ inv(A_true)
                    else:
                        A_true, A_hat = np.asarray(f_task(hand, f_true)), np.asarray(f_task(hand, qhat_f))
                        common = P @ inv(A_true)
                        desired = common @ A_hat
                        sanity_desired = common @ A_true
                    # sanity: q_hat = q must give back the recorded wrist exactly
                    s_q, _ = wrist_solve(fk, sanity_desired, rootT, all_q)
                    sd, sr = pose_err(canonical_world(fk, s_q, rootT), P)
                    extra = {"sanity_max_wrist_joint_diff": float(np.abs(s_q[1:, wi] - all_q[1:, wi]).max()),
                             "sanity_max_palm_mm": float(sd[1:].max()), "sanity_max_palm_deg": float(sr[1:].max()),
                             "sanity_composition_max_mm": float(pose_err(sanity_desired, P)[0].max()),
                             "F_shift_mm_mean": float(pose_err(A_hat[1:], A_true[1:])[0].mean()),
                             "F_shift_deg_mean": float(pose_err(A_hat[1:], A_true[1:])[1].mean())}
                    try:
                        q_arm, s_err2 = wrist_solve(fk, desired, rootT, q_z)
                        extra.update(s_err2)
                    except ValueError as exc:
                        extra["solve_error"] = str(exc)[:200]
                        er[arm] = extra
                        continue
                    assert np.array_equal(q_arm[:, fi], q_z[:, fi])
                d, r = pose_err(canonical_world(fk, q_arm, rootT)[1:], P[1:])
                lo, hi = fk.reference.lower[wi], fk.reference.upper[wi]
                extra.update({"wrist_dev_mm_mean": float(d.mean()), "wrist_dev_mm_p95": float(np.percentile(d, 95)),
                              "wrist_dev_mm_max": float(d.max()), "wrist_dev_deg_mean": float(r.mean()),
                              "wrist_dev_deg_p95": float(np.percentile(r, 95)), "wrist_dev_deg_max": float(r.max()),
                              "wrist_joint_at_limit_frac": float(np.mean((q_arm[1:, wi] <= lo + 1e-6) | (q_arm[1:, wi] >= hi - 1e-6)))})
                er[arm] = extra
                res = results[arm]
                res[k] = q_arm[1:].astype(np.float32)
                res[f"init_{k}"] = s["initial"].astype(np.float32)  # native initial state (pkl), never decoded
                res[f"root_{k}"] = s["root"].astype(np.float32)
                csv_rows.append({"hand": hand, "arm": arm, "episode": e,
                                 **{x: v for x, v in er.items() if not isinstance(v, dict)},
                                 **{x: v for x, v in extra.items() if isinstance(v, (int, float))}})
            hrep["episodes"][k] = er
            print(json.dumps({"hand": hand, "episode": e, "finger_mae": round(er["finger_abs_err_rad_mean"], 4),
                              **{a: {x: round(v, 3) for x, v in er[a].items() if isinstance(v, float)} for a in arms if a in er}}),
                  flush=True)
        for arm in arms:
            d = out / "decoded" / (f"{arm}_{args.tag}" if args.tag else arm) / TASK / hand
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{hand}_native.npz", **native)
            if len([x for x in results[arm] if x.startswith("ep")]) == len(episodes):
                np.savez_compressed(d / f"{hand}_codec.npz", **results[arm])
            else:
                print(f"[p3h] WARN {arm} {hand}: not all episodes solved; codec npz not written", flush=True)
        report["hands"][hand] = hrep
    tag = args.tag or "_".join(arms)
    (out / f"roundtrip_prep_{tag}.json").write_text(json.dumps(report, indent=1))
    keys = sorted({k for r in csv_rows for k in r}, key=lambda x: (x not in ("hand", "arm", "episode"), x))
    with open(out / f"roundtrip_prep_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)
    print("[p3h] prep DONE", flush=True)


if __name__ == "__main__":
    main()
