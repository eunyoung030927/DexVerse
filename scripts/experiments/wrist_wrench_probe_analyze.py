# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Find the frame / reference point / sign of PhysX's incoming joint force at the hand mount, by Newton balance.

For the hand (every non-virtual body downstream of the mount joint; gravity off):
    F_joint_on_hand = sum_i m_i a_i - F_contact            (world)
    T_joint_on_hand(A) = sum_i (c_i - A) x m_i a_i - (M_O - A x F_contact)    (rotational inertia ignored)
Valid substeps: the hand touches nothing but the object (unfiltered net normal == object normal).

    python analyze.py probe.npz [probe2.npz ...]
"""
import sys

import numpy as np


def qmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def rot(R, v):
    return np.einsum("...ij,...j->...i", R, v)


for path in sys.argv[1:]:
    d = np.load(path)
    dt = float(d["physics_dt"])
    names = [str(x) for x in d["hand_bodies"]]
    mounts = [str(x) for x in d["mounts"]]
    m = d["masses"]
    step = d["step"]
    print(f"\n===== {path}  substeps={len(step)}  mounts={mounts}  joints={list(d['mount_joint_names'])}")
    for h, mb in enumerate(mounts):
        # bodies of this hand: by side prefix when bimanual
        if len(mounts) > 1:
            pre = mb.split("_")[0] + "_"
            sel = np.array([n.startswith(pre) for n in names])
        else:
            sel = np.ones(len(names), bool)
        mm = m[sel]
        v = d["com_vel"][:, sel]
        c = d["com_pos"][:, sel]
        a = np.zeros_like(v)
        a[1:] = (v[1:] - v[:-1]) / dt
        newstep = np.r_[True, step[1:] != step[:-1]]
        first = np.r_[True, np.zeros(len(step) - 1, bool)]
        ma = (mm[:, None] * a).sum(1)                                   # (T, 3)
        fc = d["f_obj"][:, sel].sum(1)
        fcn = d["f_obj_normal"][:, sel].sum(1)
        nn = d["net_normal"][:, sel].sum(1)
        only_obj = np.linalg.norm(nn - fcn, axis=1) < 0.05 + 0.02 * np.linalg.norm(fcn, axis=1)
        valid = only_obj & ~first
        valid[:3] = False
        pred_F = ma - fc
        raw = d["raw_mount"][:, h]
        Rc = qmat(d["quat_mount"][:, h])
        Rp = qmat(d["quat_parent"][:, h])
        R1 = qmat(np.asarray(d["mount_joint_q1"][h], float))
        R0 = qmat(np.asarray(d["mount_joint_q0"][h], float))
        cands = {"child": Rc, "child*q1": Rc @ R1, "parent": Rp, "parent*q0": Rp @ R0,
                 "world": np.broadcast_to(np.eye(3), Rc.shape)}
        scale = np.sqrt(np.mean(np.sum(pred_F[valid] ** 2, 1)))
        print(f"  [{mb}] hand mass {mm.sum():.3f} kg, valid substeps {valid.sum()}/{len(valid)} "
              f"(contact |F| p50 {np.median(np.linalg.norm(fc[valid], axis=1)):.2f} N), rms |pred F| {scale:.2f} N")
        best = None
        for name, R in cands.items():
            for sgn in (1, -1):
                F = sgn * rot(R, raw[:, :3])
                res = np.sqrt(np.mean(np.sum((F - pred_F)[valid] ** 2, 1)))
                print(f"    F frame {name:10s} sign {sgn:+d}: rms resid {res:8.3f} N")
                if best is None or res < best[0]:
                    best = (res, name, sgn, R)
        res, name, sgn, R = best
        print(f"  -> force: frame {name}, sign {sgn:+d}, rms resid {res:.3f} N "
              f"({100 * res / max(scale, 1e-9):.1f}% of rms force)")
        # torque about candidate anchors, using the best frame for the torque too
        pc = d["pos_mount"][:, h]
        pp = d["pos_parent"][:, h]
        anchors = {"child origin": pc, "child + pos1": pc + rot(Rc, np.asarray(d["mount_joint_pos1"][h], float)),
                   "parent + pos0": pp + rot(Rp, np.asarray(d["mount_joint_pos0"][h], float))}
        T_raw = sgn * rot(R, raw[:, 3:])
        for an, A in anchors.items():
            pred_T = np.cross(c - A[:, None], mm[None, :, None] * a).sum(1) - (d["m_obj_O"] - np.cross(A, fc))
            # also only use calmer substeps (inertia term ignored for rotations)
            calm = valid & (np.linalg.norm(ma, axis=1) < 0.5 * np.linalg.norm(fc, axis=1) + 0.5)
            sc = np.sqrt(np.mean(np.sum(pred_T[calm] ** 2, 1)))
            r = np.sqrt(np.mean(np.sum((T_raw - pred_T)[calm] ** 2, 1)))
            print(f"    torque about {an:14s}: rms resid {r:7.4f} Nm vs rms {sc:7.4f} Nm "
                  f"({100 * r / max(sc, 1e-9):5.1f}%), "
                  f"calm substeps {calm.sum()}")
