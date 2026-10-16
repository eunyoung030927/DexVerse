# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the recorder's WristWrenchLogger frames against the probe's per-substep data of the same run.

logged[t] = frame value after env.step t (mean over that step's substeps), environment-on-robot, mount body axes.
Expected: -(mean over the substeps of the raw incoming joint force) (exact: the conversion), and, by Newton
(gravity off), -R^T (sum m a - F_contact) for the force and -R^T (sum (c - p) x m a - (M_O - p x F_contact)) for the
torque about the mount body origin p (rotational inertia ignored; single-hand runs only).
"""
import sys

import numpy as np


def qmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


for path in sys.argv[1:]:
    d = np.load(path)
    dt = float(d["physics_dt"])
    step = d["step"]
    T = len(d["logged"])
    names = [str(x) for x in d["hand_bodies"]]
    mounts = [str(x) for x in d["mounts"]]
    lm = [str(x) for x in d["logger_mounts"]]
    logged = d["logged"].reshape(T, len(lm), 6)
    print(f"\n===== {path}: {T} frames, logger mounts {lm}, saturated frames {d['logged_sat'].sum(0).tolist()}")
    for h, mb in enumerate(mounts):
        li = lm.index(mb)
        sel = (np.array([n.startswith(mb.split("_")[0] + "_") for n in names]) if len(mounts) > 1
               else np.ones(len(names), bool))
        m = d["masses"][sel]
        v = d["com_vel"][:, sel]
        c = d["com_pos"][:, sel]
        a = np.zeros_like(v)
        a[1:] = (v[1:] - v[:-1]) / dt
        fc = d["f_obj"][:, sel].sum(1)
        ma = (m[:, None] * a).sum(1)
        R = qmat(d["quat_mount"][:, h])
        p = d["pos_mount"][:, h]
        pred_F = np.einsum("tji,tj->ti", R, ma - fc)                     # body axes, force on the hand by the joint
        pred_T = np.einsum("tji,tj->ti", R, np.cross(c - p[:, None], m[None, :, None] * a).sum(1)
                           - (d["m_obj_O"] - np.cross(p, fc)))
        raw = d["raw_mount"][:, h]
        exp_raw, exp_F, exp_T = [], [], []
        for t in range(T):
            k = step == t
            exp_raw.append(-raw[k].mean(0))
            exp_F.append(-pred_F[k].mean(0))
            exp_T.append(-pred_T[k].mean(0))
        exp_raw, exp_F, exp_T = map(np.asarray, (exp_raw, exp_F, exp_T))
        lg = logged[:, li]
        ok = np.arange(T) >= 2
        conv = np.abs(lg - exp_raw)[ok].max()
        rF = np.linalg.norm(lg[:, :3] - exp_F, axis=1)[ok]
        sF = np.linalg.norm(exp_F, axis=1)[ok]
        print(f"  [{mb}] logged vs -mean(raw): max |diff| {conv:.2e} (conversion)")
        print(f"         force vs Newton: median |diff| {np.median(rF):.3f} N, p95 {np.percentile(rF, 95):.3f} N, "
              f"rms {np.sqrt((rF ** 2).mean()):.3f} N vs rms |F| {np.sqrt((sF ** 2).mean()):.3f} N; "
              f"max |F| logged {np.linalg.norm(lg[:, :3], axis=1).max():.2f} N")
        if len(mounts) == 1:
            rT = np.linalg.norm(lg[:, 3:] - exp_T, axis=1)[ok]
            sT = np.linalg.norm(exp_T, axis=1)[ok]
            print(f"         torque vs Newton: median |diff| {np.median(rT):.4f} Nm, p95 {np.percentile(rT, 95):.4f}, "
                  f"rms |T| {np.sqrt((sT ** 2).mean()):.4f} Nm")
