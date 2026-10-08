# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Round-trip check of the EE action decoder: env action -> four EE versions -> EEActionDecoder -> env action.

usage: python scripts/data_tools/check_ee_decode.py <dataset root> [...] [--synthetic] [--tol 1e-4]

Runs in any env with numpy / scipy / pandas (the lerobot env); no Isaac Sim. For each dataset the EE columns
are the stored ones (v2 datasets) or are recomputed from ``action`` and the sim-order joint state (v1 datasets,
which stored only ``action.ee_delta``). ``--synthetic`` adds, per dataset's hand layout, a trajectory whose
wrist rotation joints sweep past +-pi (x 0 -> 7 rad, z +-3.5 rad) and a 6D decode of noisy outputs.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DECODER_PY = os.path.normpath(os.path.join(_HERE, "..", "..", "source", "dexverse", "dexverse", "data_collection",
                                            "ee_action_decoder.py"))


def _load_decoder_module():
    """The stdlib+numpy decoder module (and through it lerobot_spool.py), loaded BY PATH (importing the dexverse
    package pulls Isaac Lab in)."""
    spec = importlib.util.spec_from_file_location("_dexverse_ee_action_decoder", _DECODER_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_dexverse_ee_action_decoder"] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load_decoder_module()
S = D._S


def wrist_errors(dec, act, hands):
    """(max |d trans| m, max |d rot joint| rad, max |d finger|) between decoded and original env actions."""
    t_idx = [j["idx"] for h in hands for j in h["trans"]]
    r_idx = [j["idx"] for h in hands for j in h["rot"]]
    f_idx = [c for c in range(act.shape[1]) if c not in set(t_idx + r_idx)]
    d = np.abs(dec.astype(np.float64) - act.astype(np.float64))
    return d[:, t_idx].max(), d[:, r_idx].max(), (d[:, f_idx].max() if f_idx else 0.0)


def round_trip(ee, act, start_ee, hands, tol, label):
    ok = True
    for kind in S.EE_ACTION_BLOCKS:
        dec = D.EEActionDecoder(hands, act.shape[1], kind)
        dec.reset(start_ee if kind == "action.ee_delta_init" else None)
        et, er, ef = wrist_errors(dec(ee[kind]), act, hands)
        good = max(et, er, ef) < tol
        ok &= good
        print(f"   {label:10s} {kind:22s} trans {et:.1e} m  rot {er:.1e} rad  finger {ef:.1e}  {'OK' if good else 'FAIL'}")
    return ok


def load(root):
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    hands = json.load(open(os.path.join(root, "meta", "isaac_tasks.json")))["ee_hands"]
    import pandas as pd  # noqa: PLC0415

    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(root, "data", "*", "*.parquet")))])
    df = df.sort_values("index")
    return info, hands, df


def check_dataset(root, tol):
    info, hands, df = load(root)
    feats = info["features"]
    stored = "action.ee_abs" in feats
    print(f"== {root}: {df['episode_index'].nunique()} ep, {len(df)} frames, "
          f"{'stored' if stored else 'recomputed (v1)'} EE columns, hands {[h['side'] for h in hands]}")
    ok = True
    for e, g in df.groupby("episode_index"):
        act = np.stack(g["action"].to_numpy()).astype(np.float64)
        if stored:
            ee = {k: np.stack(g[k].to_numpy()) for k in S.EE_ACTION_BLOCKS}
            start = np.stack(g["observation.state.ee"].to_numpy())[0]
        else:
            q = np.stack(g["observation.state"].to_numpy()).astype(np.float64)  # v1: sim joint order
            ee = S.ee_action_columns(act, q, hands)
            start = S.ee_state_columns(q[:1], hands, list(range(act.shape[1])))["observation.state.ee"][0]
        ok &= round_trip(ee, act, start, hands, tol, f"ep{e} T={len(act)}")
    return hands, ok


def check_synthetic(hands, n_action, tol, rng):
    """Wrist rotation joints sweep past +-pi; the decoder must give back the same (unwrapped) joint values."""
    T = 600
    s = np.linspace(0.0, 1.0, T)
    act = np.zeros((T, n_action))
    wrist = set()
    for h in hands:
        traj = [7.0 * s, 1.2 * np.sin(6 * np.pi * s), 3.5 * np.sin(4 * np.pi * s)]  # chain order
        for j, q in zip(h["rot"], traj):
            act[:, j["idx"]] = (q - j["offset"]) / j["scale"]
        for k, j in enumerate(h["trans"]):
            act[:, j["idx"]] = 0.1 * np.sin(2 * np.pi * s + k)
        wrist |= {j["idx"] for j in h["trans"] + h["rot"]}
    for c in set(range(n_action)) - wrist:
        act[:, c] = rng.uniform(-1, 1) * np.sin(2 * np.pi * s * rng.uniform(1, 3))
    n_joints = max(n_action, 1 + max(j["joint_id"] for h in hands for j in h["trans"] + h["rot"]))
    q0 = np.zeros((1, n_joints))
    for h in hands:
        for j in h["trans"] + h["rot"]:
            q0[0, j["joint_id"]] = act[0, j["idx"]] * j["scale"] + j["offset"]
    ee = S.ee_action_columns(act, q0, hands)
    start = S.ee_state_columns(q0, hands, list(range(n_action)))["observation.state.ee"][0]
    ok = round_trip(ee, act, start, hands, tol, "synthetic")

    # Noisy 6D outputs (not orthonormal): Gram-Schmidt still gives a rotation; report how far it lands.
    noisy = ee["action.ee_abs_rot6d"].astype(np.float64).copy()
    items = S._ee_layout_items(n_action, hands)
    col, r6_cols = 0, []
    for kind, _ in items:
        if kind == "wrist":
            r6_cols += list(range(col + 3, col + 9))
            col += 9
        else:
            col += 1
    noisy[:, r6_cols] += rng.normal(0.0, 0.05, (T, len(r6_cols)))
    dec = D.EEActionDecoder(hands, n_action, "action.ee_abs_rot6d")
    out = dec(noisy).astype(np.float64)
    worst = jump = 0.0
    for h in hands:
        idx = [j["idx"] for j in h["rot"]]
        tgt = [act[:, i] for i in idx]
        got = [out[:, i] for i in idx]
        r_t = S._chain_pose(np.zeros((T, len(h["trans"]))), np.stack(tgt, 1), h)[1]
        r_g = S._chain_pose(np.zeros((T, len(h["trans"]))), np.stack(got, 1), h)[1]
        worst = max(worst, np.degrees((r_g * r_t.inv()).magnitude()).max())
        jump = max(jump, np.abs(np.diff(np.stack(got, 1), axis=0)).max())
    print(f"   synthetic  6D + N(0, 0.05) noise: rotation error max {worst:.2f} deg, "
          f"max joint step {jump:.3f} rad (no +-2pi jumps: {jump < 1.0})")
    return ok and jump < 1.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    ok = True
    for root in args.roots:
        hands, good = check_dataset(root, args.tol)
        ok &= good
        if args.synthetic:
            n_action = json.load(open(os.path.join(root, "meta", "info.json")))["features"]["action"]["shape"][0]
            ok &= check_synthetic(hands, n_action, args.tol, rng)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
