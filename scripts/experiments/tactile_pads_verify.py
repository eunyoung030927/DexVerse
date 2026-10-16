# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validation of the virtual fingerprint pads (``--tactile_pads``) on a replayed demo (experiment, not a tool).

Replays a trajectory pickle (record_demos replay flow) or an xemb phase-1 HDF5 episode (object pose + robot joint
state restored, ``action_env`` replayed) with the pad sensor and the per-pad ``--tactile`` sensors, and per control
step compares:

* the pad split computed with the backward-integrated link poses (what the recorder uses) against the split with
  the TRUE pose before the last substep (hooked on every substep) and with the post-substep pose (no correction):
  point placement error (mm) and force-weighted label agreement;
* pad + non-pad against the bundle total (aggregated sensor);
* Sharpa: the ``elastomer`` variant (pad = the elastomer link surface) against the elastomer's own --tactile force
  (ground truth) and the fingerprint pad against both;

and writes per-finger statistics, the per-step outputs and a figure (pad samples + contact points).

    PYTHONPATH=<worktree>/source/dexverse /workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/tactile_pads_verify.py \
        --task Dexverse-PickCube-v0 --robot_type floating_allegro_right --pickle demo.pkl --out <dir> --headless --enable_cameras
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--robot_type", required=True)
parser.add_argument("--pickle", default=None)
parser.add_argument("--h5", default=None)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--thr", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", rendering_mode="performance")
args = parser.parse_args()
app = AppLauncher(vars(args)).app


def _exit_on_error(exc_type, exc, tb):
    import traceback  # noqa: PLC0415

    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    os._exit(1)


sys.excepthook = _exit_on_error

import pickle  # noqa: E402

import dexverse  # noqa: E402
import dexverse.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dexverse.data_collection.tactile import TactileReader, add_tactile_sensors  # noqa: E402
from dexverse.data_collection.tactile_pads import (  # noqa: E402
    TactilePadReader,
    _np,
    _qmat,
    add_tactile_pad_sensor,
    load_pad_definition,
)
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from dexverse.teleop_utils.episode_task_state import reset_managers_after_state_restore  # noqa: E402

print("DEXVERSE_FILE", dexverse.__file__, flush=True)
os.makedirs(args.out, exist_ok=True)
cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg = type(cfg)(robot_type=args.robot_type)
cfg.sim.device = args.device
cfg.scene.num_envs = 1
cfg.terminations.success = None
cfg.terminations.time_out = None
cfg.recorders = {}
tac = add_tactile_sensors(cfg)
add_tactile_pad_sensor(cfg, tac["targets"], args.robot_type)
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
treader = TactileReader(env, tac["bodies"], tac["targets"])
reader = TactilePadReader(env, args.robot_type, tac["targets"])
D = reader.d
has_el = "elastomer" in str(np.load(D["path"]).files)
reader_el = TactilePadReader(env, args.robot_type, tac["targets"], variant="elastomer") if has_el else None
reader_w = TactilePadReader(env, args.robot_type, tac["targets"], variant="pad75")
nf = reader.nf
print(f"[pads-verify] fingers {D['fingers']}; elastomer variant {has_el}", flush=True)

state = {"on": False, "poses": []}
_orig = env.scene.update


def _hook(dt):
    _orig(dt)
    if state["on"]:
        state["poses"].append((_np(robot.data.body_link_pos_w[0]), _qmat(_np(robot.data.body_link_quat_w[0]))))


env.scene.update = _hook


# ------------------------------------------------------------------ episode source
def to_t(x):
    if isinstance(x, dict):
        return {k: to_t(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=env.device)
    return x


env.sim.reset()
env.reset()
if args.pickle:
    with open(args.pickle, "rb") as fh:
        ep = [e for e in pickle.load(fh)["episodes"] if len(e.get("actions", []))][args.episode]
    env.reset_to(to_t(ep["initial_state"]), env_ids=None, is_relative=True)
    reset_managers_after_state_restore(env)
    actions = [np.asarray(a, dtype=np.float32).reshape(-1) for a in ep["actions"]]
else:
    import h5py  # noqa: PLC0415

    f = h5py.File(args.h5, "r")
    meta = json.loads(f.attrs["names_json"])
    eps = sorted(f["episodes"].keys())
    g = f["episodes"][eps[args.episode]]
    jn = meta["robot_joint_names"]
    q0 = np.asarray(g["obs_hand_specific/joint_pos"][0], dtype=np.float32)
    qd0 = np.asarray(g["obs_hand_specific/joint_vel"][0], dtype=np.float32)
    order = [jn.index(n) for n in robot.joint_names]
    robot.write_joint_state_to_sim(torch.tensor(q0[order])[None], torch.tensor(qd0[order])[None])
    robot.set_joint_position_target(torch.tensor(q0[order])[None])
    for name in meta["rigid_objects"]:
        pz = np.asarray(g[f"obs_object_world/{name}/root_pose"][0], dtype=np.float32)
        vz = np.asarray(g[f"obs_object_world/{name}/root_vel"][0], dtype=np.float32)
        obj = env.scene[name]
        obj.write_root_pose_to_sim(torch.tensor(pz)[None])
        obj.write_root_velocity_to_sim(torch.tensor(vz)[None])
    env.scene.write_data_to_sim()
    env.sim.forward()
    actions = [np.asarray(a, dtype=np.float32) for a in g["action_env"][:]]
    print(f"[pads-verify] h5 episode {eps[args.episode]}: {len(actions)} actions", flush=True)
if args.max_steps:
    actions = actions[: args.max_steps]

# ------------------------------------------------------------------ replay + per-step comparisons
OUT = {k: [] for k in ("total", "pad", "nonpad", "pad_centroid", "pad_w", "nonpad_w", "palm", "el_pad", "el_non", "w75_pad",
                       "tac_force_w", "resid")}
PE = {"err_est_mm": [], "err_post_mm": [], "err_bwd_mm": [], "w": [], "agree_est": [], "agree_post": [],
      "agree_bwd": [], "src": []}
BND = {"rows": 0, "bnd": 0, "f": 0.0, "fb": 0.0}
PTS = {i: [] for i in range(nf)}           # (step, xD, |F|, label) of contact rows for the figure
with torch.inference_mode():
    for t, a in enumerate(actions):
        state["poses"] = []
        state["on"] = True
        env.step(torch.as_tensor(a.reshape(1, -1), device=env.device))
        state["on"] = False
        r = reader.read(keep_rows=True)
        dg = reader.last_diag
        PE["src"].append(dg["pose_source"])
        rows_est = dg["rows_detail"]
        BND["rows"] += dg["rows"]
        BND["bnd"] += dg["boundary_rows"]
        BND["f"] += dg["force_rows"]
        BND["fb"] += dg["force_boundary"]
        OUT["resid"].append(dg["sum_resid_max_N"])
        for k in ("total", "pad", "nonpad", "pad_centroid", "pad_w", "nonpad_w", "palm"):
            OUT[k].append(r[k])
        OUT["tac_force_w"].append(treader.read()["force_w"])
        if len(state["poses"]) >= 2 and rows_est:
            lp_t, R_t = state["poses"][-2]
            lp_c, R_c = state["poses"][-1]
            reader.read(poses=(lp_t, R_t), keep_rows=True)
            rows_true = reader.last_diag["rows_detail"]
            reader.read(poses=(lp_c, R_c), keep_rows=True)
            rows_post = reader.last_diag["rows_detail"]
            reader.read(pose_source="backward", keep_rows=True)
            rows_bwd = reader.last_diag["rows_detail"]
            for re_, rt_, rp_, rb_ in zip(rows_est, rows_true, rows_post, rows_bwd):
                xe, vec, le = re_[1], re_[2], re_[4]
                xt, lt = rt_[1], rt_[4]
                xp, lp_ = rp_[1], rp_[4]
                xb, lb_ = rb_[1], rb_[4]
                m = np.linalg.norm(vec, axis=1)
                PE["err_est_mm"] += list(np.linalg.norm(xe - xt, axis=1) * 1e3)
                PE["err_post_mm"] += list(np.linalg.norm(xp - xt, axis=1) * 1e3)
                PE["err_bwd_mm"] += list(np.linalg.norm(xb - xt, axis=1) * 1e3)
                PE["w"] += list(m)
                PE["agree_est"] += list(le == lt)
                PE["agree_post"] += list(lp_ == lt)
                PE["agree_bwd"] += list(lb_ == lt)
        for (ln, xe, vec, isn, le, _a, _b) in rows_est:
            fi = reader.link_finger[ln]
            dist = D["distal"][fi]
            if ln != dist:  # express child-link points in the distal frame (fixed relative pose)
                i_l, i_d = robot.body_names.index(ln), robot.body_names.index(dist)
                lp_now, R_now = _np(robot.data.body_link_pos_w[0]), _qmat(_np(robot.data.body_link_quat_w[0]))
                xw = xe @ R_now[i_l].T + lp_now[i_l]
                xd = (xw - lp_now[i_d]) @ R_now[i_d]
            else:
                xd = xe
            PTS[fi].append((t, xd, np.linalg.norm(vec, axis=1), le, isn))
        if reader_el is not None:
            re_ = reader_el.read()
            OUT["el_pad"].append(re_["pad_w"])
            OUT["el_non"].append(re_["nonpad_w"])
        OUT["w75_pad"].append(reader_w.read()["pad_w"])
        if t % 100 == 0:
            print(f"[pads-verify] step {t}/{len(actions)}", flush=True)

A = {k: np.asarray(v) for k, v in OUT.items() if len(v)}
np.savez_compressed(os.path.join(args.out, "pads_steps.npz"), fingers=np.array(D["fingers"]),
                    pad_bodies=np.array(tac["bodies"]), **A)
res = {"robot_type": args.robot_type, "steps": len(actions), "fingers": D["fingers"]}

# ------------------------------------------------------------------ statistics
tot = np.linalg.norm(A["total"], axis=-1)
padm = np.linalg.norm(A["pad"], axis=-1)
nonm = np.linalg.norm(A["nonpad"], axis=-1)
P = lambda *x: print(*x, flush=True)  # noqa: E731
P("\n## per finger (contact frames = |bundle total| > thr)")
P(f"{'finger':22s} {'frames':>6s} {'pad only':>8s} {'nonpad only':>11s} {'mixed':>6s} {'pad share':>9s} {'max tot N':>9s} "
  f"{'centroid s,h,w med (mm)':>24s}")
fin = []
for i, fname in enumerate(D["fingers"]):
    live = tot[:, i] > args.thr
    if not live.any():
        P(f"{fname:22s} {0:6d}")
        fin.append({"finger": fname, "frames": 0})
        continue
    p_, n_ = padm[live, i] > args.thr, nonm[live, i] > args.thr
    fr = np.load(D["path"])[f"frame_{i}"]
    o, u, h = fr[0], fr[1], fr[2]
    w = np.cross(u, h)
    cen = A["pad_centroid"][live, i][p_]
    cs = (np.median((cen - o) @ u) * 1e3, np.median((cen - o) @ h) * 1e3, np.median((cen - o) @ w) * 1e3) if len(cen) else (np.nan,) * 3
    row = {"finger": fname, "frames": int(live.sum()), "pad_only_pct": float((p_ & ~n_).mean() * 100),
           "nonpad_only_pct": float((~p_ & n_).mean() * 100), "mixed_pct": float((p_ & n_).mean() * 100),
           "pad_share": float(padm[live, i].sum() / max(tot[live, i].sum(), 1e-9)), "max_total_N": float(tot[:, i].max()),
           "pad_centroid_med_mm_s_h_w": [float(x) for x in cs]}
    fin.append(row)
    P(f"{fname:22s} {row['frames']:6d} {row['pad_only_pct']:8.1f} {row['nonpad_only_pct']:11.1f} {row['mixed_pct']:6.1f} "
      f"{row['pad_share']:9.3f} {row['max_total_N']:9.2f}   {cs[0]:6.1f} {cs[1]:6.1f} {cs[2]:6.1f}")
w75 = np.linalg.norm(A["w75_pad"], axis=-1)
for i, row in enumerate(fin):
    live = tot[:, i] > args.thr
    if live.any():
        row["pad_share_75deg"] = float(w75[live, i].sum() / max(tot[live, i].sum(), 1e-9))
P("pad share with the 75 deg variant: " + ", ".join(f"{r['finger']} {r.get('pad_share_75deg', float('nan')):.3f}" for r in fin))
res["fingers_stats"] = fin
palm = np.linalg.norm(A["palm"], axis=-1)
res["palm"] = {"frames": int((palm > args.thr).sum(0).max()), "max_N": float(palm.max())}
P(f"palm: frames in contact {(palm > args.thr).sum(0).tolist()}, max {palm.max():.2f} N")
P(f"\npad + nonpad vs bundle total: max |resid| over all steps {A['resid'].max():.3e} N")
res["sum_resid_max_N"] = float(A["resid"].max())
if PE["w"]:
    e, ep_, eb, wv = (np.array(PE[k]) for k in ("err_est_mm", "err_post_mm", "err_bwd_mm", "w"))
    ae, ap, ab = (np.array(PE[k]) for k in ("agree_est", "agree_post", "agree_bwd"))
    srcs = {k: PE["src"].count(k) for k in set(PE["src"])}
    P(f"pose source used by the recorder path: {srcs}")
    for lab, x, ag in (("recorder (substep cache)", e, ae), ("backward integration", eb, ab), ("uncorrected (post)", ep_, ap)):
        P(f"  {lab:26s}: point error vs TRUE pre-substep pose median {np.median(x):.4f} p95 {np.percentile(x, 95):.4f} "
          f"max {x.max():.4f} mm; pad label agreement (force-weighted) {(wv * ag).sum() / wv.sum():.5f}")
    res["pose_correction"] = {lab: {"mm_median": float(np.median(x)), "mm_p95": float(np.percentile(x, 95)),
                                    "mm_max": float(x.max()), "agree_w": float((wv * ag).sum() / wv.sum())}
                              for lab, x, ag in (("cache", e, ae), ("backward", eb, ab), ("post", ep_, ap))}
    res["pose_correction"]["n_rows"] = int(len(e))
    res["pose_source_counts"] = srcs
fa = np.array(reader._fric_anchor_mm)
if len(fa):
    P(f"friction anchor -> nearest normal contact point of the same pair: median {np.median(fa):.3f} p95 "
      f"{np.percentile(fa, 95):.3f} max {fa.max():.3f} mm (n={len(fa)})")
    res["friction_anchor_mm"] = {"median": float(np.median(fa)), "p95": float(np.percentile(fa, 95)), "max": float(fa.max())}
P(f"boundary rows (|d_pad - d_nonpad| < 1 mm): {BND['bnd']}/{BND['rows']} rows, force share {BND['fb'] / max(BND['f'], 1e-9):.4f}")
res["boundary"] = {"rows": BND["rows"], "boundary_rows": BND["bnd"], "force_share": BND["fb"] / max(BND["f"], 1e-9)}
if reader_el is not None:
    # ground truth: elastomer link force from the --tactile reader (per elastomer pad body, world frame)
    tb = list(tac["bodies"])
    gt = np.zeros_like(A["el_pad"])
    for i, dist in enumerate(D["distal"]):
        el = dist.replace("_DP", "_elastomer")
        if el in tb:
            gt[:, i] = A["tac_force_w"][:, tb.index(el)]
    d_el = np.linalg.norm(A["el_pad"] - gt, axis=-1)
    live = np.linalg.norm(gt, axis=-1) > args.thr
    P(f"\n## Sharpa ground truth: 'elastomer' virtual pad vs elastomer sensor: max |diff| {d_el.max():.3e} N "
      f"(rel median {np.median(d_el[live] / np.linalg.norm(gt[live], axis=-1)) if live.any() else 0:.2e})")
    fp = np.linalg.norm(A["pad_w"], axis=-1)
    g_ = np.linalg.norm(gt, axis=-1)
    both = (np.linalg.norm(A["total"], axis=-1) > args.thr)
    P(f"fingerprint pad vs elastomer truth over bundle-contact frames: corr |F| {np.corrcoef(fp[both], g_[both])[0, 1]:.4f}, "
      f"median |diff| {np.median(np.abs(fp[both] - g_[both])):.3f} N, median |diff|/|total| "
      f"{np.median(np.abs(fp[both] - g_[both]) / np.linalg.norm(A['total'], axis=-1)[both]):.3f}")
    for i, fname in enumerate(D["fingers"]):
        lv = both[:, i]
        if lv.sum() > 2:
            P(f"   {fname:22s} frames {lv.sum():4d}  elastomer share {g_[lv, i].sum() / max(np.linalg.norm(A['total'], axis=-1)[lv, i].sum(), 1e-9):.3f}"
              f"  fingerprint share {fp[lv, i].sum() / max(np.linalg.norm(A['total'], axis=-1)[lv, i].sum(), 1e-9):.3f}"
              f"  corr {np.corrcoef(fp[lv, i], g_[lv, i])[0, 1] if fp[lv, i].std() > 0 and g_[lv, i].std() > 0 else float('nan'):.3f}")
    res["sharpa_truth"] = {"elastomer_variant_maxdiff_N": float(d_el.max()),
                           "fingerprint_vs_elastomer_corr": float(np.corrcoef(fp[both], g_[both])[0, 1])}
json.dump(res, open(os.path.join(args.out, "pads_verify.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ figure: pad samples + contact points
z = np.load(D["path"])
fig, axs = plt.subplots(nf, 2, figsize=(9, 3.2 * nf), squeeze=False)
rng = np.random.default_rng(0)
for i, fname in enumerate(D["fingers"]):
    fr = z[f"frame_{i}"]
    o, u, h = fr[0], fr[1], fr[2]
    w = np.cross(u, h)
    dist = D["distal"][i]
    for li, ln in enumerate(D["links"]):
        if D["link_finger"][li] != i:
            continue
        S, Mk = D["samples"][ln], D["masks"][ln]
        if ln != dist:
            i_l, i_d = robot.body_names.index(ln), robot.body_names.index(dist)
            lp_now, R_now = _np(robot.data.body_link_pos_w[0]), _qmat(_np(robot.data.body_link_quat_w[0]))
            S = ((S @ R_now[i_l].T + lp_now[i_l]) - lp_now[i_d]) @ R_now[i_d]
        sub = rng.choice(len(S), min(len(S), 3000), replace=False)
        for ax, y in ((axs[i, 0], h), (axs[i, 1], w)):
            ax.scatter((S[sub] - o) @ u * 1e3, (S[sub] - o) @ y * 1e3, s=1, c=np.where(Mk[sub], "#f4a6a6", "#dddddd"))
    pts = PTS[i]
    if pts:
        xs = np.concatenate([p[1] for p in pts])
        fs = np.concatenate([p[2] for p in pts])
        lb = np.concatenate([p[3] for p in pts])
    if pts and len(xs):
        sel = rng.choice(len(xs), min(len(xs), 600), replace=False)
        for ax, y in ((axs[i, 0], h), (axs[i, 1], w)):
            ax.scatter((xs[sel] - o) @ u * 1e3, (xs[sel] - o) @ y * 1e3, s=4 + 30 * np.clip(fs[sel] / max(fs.max(), 1e-9), 0, 1),
                       c=np.where(lb[sel], "red", "blue"), edgecolors="k", linewidths=0.2)
    axs[i, 0].set_title(f"{fname}: side (s, h palmar)  red/blue = pad/non-pad contacts", fontsize=7)
    axs[i, 1].set_title(f"{fname}: palmar view (s, w)", fontsize=7)
    for ax in axs[i]:
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
fig.suptitle(f"{args.robot_type}: virtual pads (pink) and replayed contact points", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(args.out, f"{args.robot_type}_contacts.png"), dpi=130)
print("PADS_VERIFY_DONE", flush=True)
os._exit(0)
