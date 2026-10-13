# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tables for tactile_force_verify.py output (numpy only):  python tactile_force_verify_report.py <out_dir> [focus regex]"""

import json
import re
import sys

import numpy as np

out = sys.argv[1]
focus = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
spool_npz = sys.argv[3] if len(sys.argv) > 3 else None
Z = np.load(out + "/verify.npz")
info = json.load(open(out + "/run_info.json"))
names = list(Z["sens_names"])
pads = list(Z["pad_bodies"])
jn = list(Z["joint_names"])
step = Z["step"]
dt, sdt, m = float(Z["phys_dt"]), float(Z["step_dt"]), float(Z["mass"])
T = int(step.max()) + 1
last = np.array([np.where(step == t)[0][-1] for t in range(T)])          # substep index at each control-step end
F_sub = Z["lnk_n"] + Z["lnk_f"]                                          # (K, B, 3) force on each link from targets
F = F_sub[last]
Fn = Z["lnk_n"][last]
NET = Z["lnk_net"][last]
other_n_sub = np.linalg.norm(Z["lnk_net"] - Z["lnk_n"], axis=-1)         # normal from non-target partners
res = {"run": info}
P = lambda *a: print(*a, flush=True)  # noqa: E731

# ------------------------------------------------------------------ 1. per-link contact census
mag = np.linalg.norm(F, axis=-1)
magsub = np.linalg.norm(F_sub, axis=-1)
netmag = np.linalg.norm(NET, axis=-1)
P(f"\n## 1. per-link contact with the target(s) (control-step end = recorded sample; T={T} steps, {len(step)} substeps)")
P(f"{'link':28s} {'pad':3s} {'frames':>6s} {'substeps':>8s} {'max|F|N':>8s} {'med|F|':>7s} {'sum|F|dt':>8s}  {'net(all partners) frames/max':>28s}")
rows = []
for b, n in enumerate(names):
    c = mag[:, b] > 1e-3
    show = c.any() or (focus is not None and focus.search(n)) or n in pads
    row = {"link": n, "pad": n in pads, "frames": int(c.sum()), "substeps": int((magsub[:, b] > 1e-3).sum()),
           "max": float(mag[:, b].max()), "med": float(np.median(mag[c, b])) if c.any() else 0.0,
           "impulse": float(magsub[:, b].sum() * dt), "net_frames": int((netmag[:, b] > 1e-3).sum()),
           "net_max": float(netmag[:, b].max())}
    rows.append(row)
    if show:
        P(f"{n:28s} {'*' if row['pad'] else '':3s} {row['frames']:6d} {row['substeps']:8d} {row['max']:8.2f} {row['med']:7.2f} "
          f"{row['impulse']:8.3f}  {row['net_frames']:6d} / {row['net_max']:7.2f}")
res["links"] = rows
pad_idx = [names.index(p) for p in pads]
tot_imp = magsub.sum()
pad_imp = magsub[:, pad_idx].sum()
anyc = mag.sum(1) > 1e-3
vec_share = np.linalg.norm(F[:, pad_idx].sum(1), axis=-1)[anyc] / np.maximum(np.linalg.norm(F.sum(1), axis=-1)[anyc], 1e-9)
P(f"pad share of |F| impulse over all hand links: {pad_imp / max(tot_imp, 1e-12) * 100:.1f}%  "
  f"(per contact frame |sum pads F| / |sum all F|: median {np.median(vec_share) * 100 if anyc.any() else 0:.1f}%)")
res["pad_share_impulse_pct"] = float(pad_imp / max(tot_imp, 1e-12) * 100)

# ------------------------------------------------------------------ 2. feature vs independent sensors / spool
S_fw = Z["S_pad_force_w"]
indep = F[:, pad_idx]
d1 = np.abs(S_fw - indep).max()
P(f"\n## 2a. --tactile feature force_w vs independent all-link sensor (same instant): max |diff| = {d1:.3e} N")
res["feature_vs_indep_maxdiff"] = float(d1)
if spool_npz:
    sp = np.load(spool_npz)
    fw = sp["observation.tactile.force_w"].reshape(len(sp["observation.tactile.force_w"]), -1, 3)
    n = min(len(fw) - 1, T)
    d2 = np.abs(fw[1:n + 1] - S_fw[:n]).max()
    P(f"## 2b. recorded spool force_w[t+1] vs this run's step-end value[t] (t<{n}): max |diff| = {d2:.3e} N; "
      f"spool frame0 max |F| = {np.abs(fw[0]).max():.3e}")
    res["spool_vs_run_maxdiff"] = float(d2)

if "obj_vel" in Z.files:
    g = np.array([0.0, 0.0, -9.81])
    v, p = Z["obj_vel"], Z["obj_pos"]
    hand_side = F_sub.sum(1)                       # sum of forces ON the hand links FROM the object
    hand_side_n = Z["lnk_n"].sum(1)
    if "obj_hand_n" in Z.files:
        hand_on_obj = Z["obj_hand_n"] + Z["obj_hand_f"]
        n3 = np.linalg.norm(hand_on_obj + hand_side, axis=-1)
        act = np.linalg.norm(hand_side, axis=-1) > 1e-3
        P(f"\n## 2c. Newton-3 object side vs hand side: max {n3[act].max() if act.any() else 0:.2e} N")
    hand_on_obj = -hand_side                       # Newton 3 (object-side sensors perturb the replay, see report)
    ok = np.r_[False, False, np.diff(step)[1:] >= 0]
    vfd = np.zeros_like(p)
    vfd[1:] = (p[1:] - p[:-1]) / dt                # PhysX semi-implicit Euler: p_k = p_k-1 + v_k dt
    P(f"\nobject velocity buffer check: median |dp/dt - root_com_lin_vel_w| = {np.median(np.linalg.norm(vfd[1:] - v[1:], axis=1)):.4f} m/s"
      f" (reported |v| range {np.linalg.norm(v, axis=1).min():.3f}..{np.linalg.norm(v, axis=1).max():.3f}) -> using dp/dt")
    res["obj_vel_buffer_err"] = float(np.median(np.linalg.norm(vfd[1:] - v[1:], axis=1)))
    v = vfd
    a = np.zeros_like(v)
    a[1:] = (v[1:] - v[:-1]) / dt
    z0 = p[0, 2]
    lifted = (p[:, 2] > z0 + 0.03) & ok            # >3 cm above its start: only the hand can touch it
    P(f"\n## 2d. object balance m a = F_hand + m g on lifted substeps   (m = {m:.4f} kg, m g = {m * 9.81:.3f} N; "
      f"substep dt {dt:.5f}, step dt {sdt:.5f}, lifted substeps {lifted.sum()} / {len(v)})")
    P(f"{'variant':58s} {'frames':>6s} {'median|res|':>11s} {'p95|res|':>9s} {'med|F_hand|':>11s}")
    stats = {}

    def show(label, resid, sel, fref):
        r = np.linalg.norm(resid[sel], axis=-1)
        stats[label] = {"n": int(sel.sum()), "median": float(np.median(r)) if sel.any() else None,
                        "p95": float(np.percentile(r, 95)) if sel.any() else None}
        if sel.any():
            P(f"{label:58s} {sel.sum():6d} {np.median(r):11.4f} {np.percentile(r, 95):9.4f} "
              f"{np.median(np.linalg.norm(fref[sel], axis=-1)):11.3f}")

    show("A substep: F(k) vs (p_k - 2p_k-1 + p_k-2)/dt^2  [time-consistent]", m * a - (hand_on_obj + m * g), lifted, hand_on_obj)
    show("B substep: normal only (friction dropped)", m * a - (-hand_side_n + m * g), lifted, hand_on_obj)
    show("C substep: friction sign flipped", m * a - (-(hand_side_n - (hand_side - hand_side_n)) + m * g), lifted,
         hand_on_obj)
    pads_only = -F_sub[:, pad_idx].sum(1)
    show("D substep: pad links only (other links dropped)", m * a - (pads_only + m * g), lifted, hand_on_obj)
    vs, ps = v[last], p[last]
    lift_s = (ps[:, 2] > z0 + 0.03) & np.r_[False, False, np.ones(T - 2, bool)]
    Fs = hand_on_obj[last]
    a_c = np.gradient(vs, sdt, axis=0)
    a_b = np.zeros_like(vs)
    a_b[1:] = (vs[1:] - vs[:-1]) / sdt
    show("E step: last-substep F vs central-diff a (= earlier check)", m * a_c - (Fs + m * g), lift_s, Fs)
    show("F step: last-substep F vs backward-diff a", m * a_b - (Fs + m * g), lift_s, Fs)
    Fmean = np.stack([hand_on_obj[step == t].mean(0) for t in range(T)])
    show("G step: substep-mean F vs backward-diff a", m * a_b - (Fmean + m * g), lift_s, Fs)
    Fpad_s = pads_only[last]
    show("H step: pads only, last-substep, central diff", m * a_c - (Fpad_s + m * g), lift_s, Fs)
    res["balance"] = stats
    res["lifted_substeps"] = int(lifted.sum())

# ------------------------------------------------------------------ 3. joint torques
tau_c = Z["tau_n"]          # pair resultant (normal+friction) at the mean contact point
tau_pt = Z["tau_f"]         # per contact point (normal sign aligned to the pair resultant)
q, qd, qt = Z["q"], Z["qd"], Z["qtgt"]
stiff, damp = Z["stiff"], Z["damp"]
pd_raw = stiff * (qt - q) - damp * qd
eff = Z["effort_limit"] if "effort_limit" in Z.files else np.full_like(stiff, np.inf)
pd_post = np.clip(pd_raw, -eff, eff)           # PhysX drive force is capped at the joint's max force
applied, proj = Z["applied"], Z["proj"]
qdd = np.zeros_like(qd)
qdd[1:] = (qd[1:] - qd[:-1]) / dt
# generalized inertial force (Newton-Euler over the links, = M qdd + c without armature) + armature * qdd,
# from finite differences of the link COM poses (the velocity buffers are checked against them)
cpos, cquat = Z["b_cpos"].astype(np.float64), Z["b_cquat"].astype(np.float64)
K = len(step)
vb = np.zeros_like(cpos)
vb[1:] = (cpos[1:] - cpos[:-1]) / dt
ab = np.zeros_like(cpos)
ab[1:] = (vb[1:] - vb[:-1]) / dt


def _qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


qrel = _qmul(cquat[1:], cquat[:-1] * np.array([1, -1, -1, -1]))
qrel *= np.sign(qrel[..., :1] + 1e-12)
vn = np.linalg.norm(qrel[..., 1:], axis=-1, keepdims=True)
wb = np.zeros_like(cpos)
wb[1:] = 2 * np.arctan2(vn, qrel[..., :1]) * qrel[..., 1:] / np.maximum(vn, 1e-12) / dt
alb = np.zeros_like(cpos)
alb[1:] = (wb[1:] - wb[:-1]) / dt
P(f"\nrobot link velocity buffers vs finite differences: median |dx/dt - body_com_lin_vel_w| = "
  f"{np.median(np.linalg.norm(vb[1:] - Z['b_cvel'][1:], axis=-1)):.4f} m/s, |w_fd - body_com_ang_vel_w| = "
  f"{np.median(np.linalg.norm(wb[1:] - Z['b_cang'][1:], axis=-1)):.4f} rad/s")


def _qm(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


Rb = _qm(cquat)                                                          # (K, B, 3, 3)
Iw = Rb @ Z["link_inertia"][None] @ np.swapaxes(Rb, -1, -2)
fa = Z["link_mass"][None, :, None] * ab                                   # (K, B, 3)
ma = np.einsum("kbij,kbj->kbi", Iw, alb) + np.cross(wb, np.einsum("kbij,kbj->kbi", Iw, wb))
anch, axs, A = Z["j_anchor"], Z["j_axis"], Z["anc_mat"].astype(float)   # (K,D,3), (K,D,3), (B,D)
rev = (Z["dof_kind"] == "rev")
tau_in_fd = np.zeros((K, len(jn)))
for d in range(len(jn)):
    bs = np.where(A[:, d] > 0)[0]
    if not len(bs):
        continue
    if rev[d]:
        mom = np.cross(cpos[:, bs] - anch[:, None, d], fa[:, bs]) + ma[:, bs]
        tau_in_fd[:, d] = np.einsum("kbi,ki->k", mom, axs[:, d])
    else:
        tau_in_fd[:, d] = np.einsum("kbi,ki->k", fa[:, bs], axs[:, d])
tau_in_fd[:2] = np.nan
Mqdd = tau_in_fd + Z["armature"] * qdd
gc = -Z["tau_g"]            # "compensation" convention: tau_drive + tau_c = M qdd + c - tau_g
cc = np.zeros_like(gc)
virt = re.compile(r"translation|rotation|trans_|rot_|_(x|y|z)_joint|roll|pitch|yaw|wrist")
fj = [i for i, n in enumerate(jn) if not virt.search(n)]
fin_qd = np.abs(qd[:, fj]).max(1)
clean = (other_n_sub.max(1) < 0.05) & np.r_[False, np.diff(step) >= 0]
A_mat = Z["anc_mat"]                                          # (B_robot, D): link b is distal to dof d
rbn = list(Z["body_names"])
sens2rob = np.array([rbn.index(n) for n in names])
other_rob = np.zeros((len(step), len(rbn)))
other_rob[:, sens2rob] = other_n_sub
lim = Z["pos_limits"]
vlim = Z["vel_limits"] if "vel_limits" in Z.files else np.full(len(jn), np.inf)
at_lim = (q < lim[None, :, 0] + 0.005) | (q > lim[None, :, 1] - 0.005) | (np.abs(qd) > 0.8 * vlim[None, :])
valid = np.isfinite(Mqdd).all(1) & np.r_[False, False, np.diff(step)[1:] >= 0]
P(f"\n## 3. joint torques  tau_drive + tau_contact + tau_gravity = M qdd + c   (per DOF)")
P("tau_contact = sum over target contacts of J(p)^T F with the pair resultant (normal+friction) at the mean contact point;")
P("tau_drive = k (q* - q) - d qd with the post-substep state, clipped to the effort limit (implicit PD);")
P("M qdd + c from Newton-Euler over the distal links (finite-difference COM accelerations) + armature qdd.")
P(f"robot links with gravity enabled: {int(Z['grav_on'].sum())}/{len(Z['grav_on'])} (tau_gravity = 0 if none); "
  f"PhysX DOF projected joint forces / mass matrix: {'available' if np.isfinite(proj).all() else 'UNAVAILABLE on this backend'}")
P("substeps used per joint: valid, |tau_c|>2 mNm, no non-target contact on any link distal to the joint, joint not within"
  " 5 mrad of a position limit, |qd| < 1 rad/s and < 80% of its velocity limit")
P(f"{'joint':22s} {'n':>4s} {'k':>6s} {'med|tc|':>8s} {'med|PD|':>8s} {'med|Mqdd|':>9s} {'corr(PD,-tc)':>12s} "
  f"{'slope':>6s} {'med|res|/med|tc|':>16s}")
jt = []
for j in fj:
    sub_other = other_rob[:, A_mat[:, j]].max(1) if A_mat[:, j].any() else np.zeros(len(step))
    sel = valid & (np.abs(tau_c[:, j]) > 2e-3) & (sub_other < 0.05) & ~at_lim[:, j] & (np.abs(qd[:, j]) < 1.0)
    if sel.sum() < 5:
        continue
    tc, pdp = tau_c[sel, j], pd_post[sel, j]
    resid = pdp + tc - Mqdd[sel, j] - gc[sel, j]
    cr = float(np.corrcoef(pdp, -tc)[0, 1]) if np.std(tc) > 0 and np.std(pdp) > 0 else float("nan")
    slope = float(np.dot(pdp, -tc) / np.dot(tc, tc))
    row = {"joint": jn[j], "n": int(sel.sum()), "k": float(stiff[j]), "tc": float(np.median(np.abs(tc))),
           "pd": float(np.median(np.abs(pdp))), "Mqdd": float(np.median(np.abs(Mqdd[sel, j]))), "corr": cr,
           "slope": slope, "rel_res": float(np.median(np.abs(resid)) / max(np.median(np.abs(tc)), 1e-12))}
    jt.append(row)
    P(f"{jn[j]:22s} {row['n']:4d} {row['k']:6.2f} {row['tc']:8.4f} {row['pd']:8.4f} {row['Mqdd']:9.4f} {cr:12.3f} "
      f"{slope:6.3f} {row['rel_res']:16.3f}")
res["joints"] = jt
if jt:
    allc = np.array([r["corr"] for r in jt]); alls = np.array([r["slope"] for r in jt]); allr = np.array([r["rel_res"] for r in jt])
    P(f"summary over {len(jt)} joints: median corr {np.nanmedian(allc):.3f}, median slope {np.median(alls):.3f}, "
      f"median rel residual {np.median(allr):.3f}")
fin_qd = np.abs(qd[:, fj]).max(1)

# peak-force anatomy
k = int(np.argmax(magsub.max(1)))
b = int(np.argmax(magsub[k]))
P(f"\n## 3b. peak contact: substep {k} (step {step[k]}), link {names[b]}, |F| = {magsub[k, b]:.2f} N "
  f"(normal {np.linalg.norm(Z['lnk_n'][k, b]):.2f}, friction {np.linalg.norm(Z['lnk_f'][k, b]):.2f})")
P(f"{'joint':22s} {'k':>6s} {'limit':>6s} {'atlim':>5s} {'q*-q (rad)':>10s} {'PDpost Nm':>10s} {'applied':>8s} {'tau_c Nm':>9s} {'Mqdd':>8s} {'lever |tc|/|F| mm':>17s}")
peak = []
for j in fj:
    if abs(tau_c[k, j]) > 0.01:
        lev = abs(tau_c[k, j]) / max(magsub[k].sum(), 1e-9) * 1e3
        P(f"{jn[j]:22s} {stiff[j]:6.2f} {eff[j]:6.2f} {str(bool(at_lim[k, j])):>5s} {qt[k, j] - q[k, j]:10.4f} {pd_post[k, j]:10.4f} {applied[k, j]:8.4f} "
          f"{tau_c[k, j]:9.4f} {Mqdd[k, j]:8.4f} {lev:17.1f}")
        peak.append({"joint": jn[j], "k": float(stiff[j]), "dq": float(qt[k, j] - q[k, j]), "pd_post": float(pd_post[k, j]),
                     "tau_c": float(tau_c[k, j]), "Mqdd": float(Mqdd[k, j])})
res["peak"] = {"substep": k, "step": int(step[k]), "link": names[b], "F": float(magsub[k, b]), "joints": peak,
               "other_partner_normal_at_peak": float(other_n_sub[k].max()), "finger_qd_max": float(fin_qd[k])}
P(f"at peak: max non-target normal on any link {other_n_sub[k].max():.2f} N, max finger |qd| {fin_qd[k]:.3f} rad/s")
json.dump(res, open(out + "/report.json", "w"), indent=1)
P("REPORT_DONE")
