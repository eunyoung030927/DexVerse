# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analysis of tactile_rawcontact_probe.py output (experiment): raw-contact consistency, position reliability and
position-based pad / non-pad splitting of distal-segment contacts against link-level ground truth.

Run with Isaac Sim's kit python + standalone pxr (no simulation):

    L=/workspace/isaaclab/_isaac_sim/extscache/omni.usd.libs-...; PYTHONPATH=$L:/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages \
    LD_LIBRARY_PATH=$L/bin /workspace/isaaclab/_isaac_sim/python.sh tactile_rawcontact_analyze.py \
        --run <probe out> [--run2 <second probe out>] --robot_usd <hand usd> --hand allegro|shadow|sharpa
"""

import argparse
import json
import pickle

import numpy as np
import trimesh
from pxr import Usd, UsdPhysics
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--run2", default=None)
ap.add_argument("--robot_usd", required=True)
ap.add_argument("--hand", required=True, choices=["allegro", "shadow", "sharpa"])
ap.add_argument("--thr", type=float, default=0.05, help="N; a link / class counts as in contact above this")
ap.add_argument("--s_frac", type=float, default=0.5, help="R2 pad rule: distal fraction of the finger segment")
ap.add_argument("--palmar", type=int, default=1, help="R2 also requires the palmar half (h >= centre line)")
ap.add_argument("--tag", default="", help="suffix of the output json files")
ap.add_argument("--pose_lag", type=int, default=1,
                help="body poses of substep k - lag are used for the contacts of substep k (contacts are generated "
                     "before the substep integrates the poses; 0 = poses after the substep)")
args = ap.parse_args()
P = lambda *a: print(*a, flush=True)  # noqa: E731

Z = np.load(args.run + "/raw.npz")
G = pickle.load(open(args.run + "/geom.pkl", "rb"))
names = list(Z["sens_names"])
rbn = list(Z["robot_body_names"])
tgt = list(Z["tgt_names"])
TGT_ORDER = list(tgt)                          # order of tgt_pose rows (= the configured target list)
if "filter_paths" in Z.files:
    fcol = [str(x).rsplit("/", 1)[-1] for x in Z["filter_paths"][0]]
    assert all([str(x).rsplit("/", 1)[-1] for x in r] == fcol for r in Z["filter_paths"]), "per-sensor filter order differs"
    print(f"filter column order (PhysX): {fcol}; configured target order: {tgt}", flush=True)
    tgt = fcol                                  # column m of the contact buffers is body tgt[m]
TPOSE_IDX = np.array([TGT_ORDER.index(n) for n in tgt])
crow, frow = Z["crow"], Z["frow"]
fm, ffm, cpos = Z["fm"].astype(np.float64), Z["ffm"].astype(np.float64), Z["cpos"].astype(np.float64)
bpose, tpose = Z["body_pose"].astype(np.float64), Z["tgt_pose"].astype(np.float64)
K, B, M = fm.shape[0], fm.shape[1], fm.shape[2]
res = {"run": args.run, "hand": args.hand, "substeps": K, "contact_rows": len(crow), "friction_rows": len(frow)}
ck, cb, cm = crow[:, 0].astype(int), crow[:, 1].astype(int), crow[:, 2].astype(int)
cp, cn, csep, cf = crow[:, 3:6], crow[:, 6:9], crow[:, 9], crow[:, 10]
fk, fb, fmm = frow[:, 0].astype(int), frow[:, 1].astype(int), frow[:, 2].astype(int)
fpnt, fvec = frow[:, 3:6], frow[:, 6:9]

# ======================================================================= 1. raw vs aggregated
P(f"## 1. raw contact rows vs aggregated sensor outputs  ({args.run})")
P(f"substeps {K}, contact rows {len(crow)}, friction rows {len(frow)}, max contacts/substep {Z['d_n_contacts'].max()} "
  f"(capacity {int(Z['cap'])}, saturated substeps {int(Z['d_sat'].sum())})")
P(f"repeat get_contact_data identical {int(Z['d_repeat_equal'].sum())}/{K}; friction call overwrote the contact "
  f"call's count buffer in {int((Z['d_count_overwritten'] > 0).sum())} substeps, start buffer in "
  f"{int((Z['d_start_overwritten'] > 0).sum())}")
if len(Z["d_old_ratio"]):
    o = Z["d_old_ratio"]
    P(f"OLD parse (contact rows sliced with the friction call's count/start): ratio to force_matrix median {np.median(o):.3f} "
      f"p5 {np.percentile(o, 5):.3f} p95 {np.percentile(o, 95):.3f}  (n={len(o)})")
key = ck * (B * M) + cb * M + cm
fkey = fk * (B * M) + fb * M + fmm
uk, inv = np.unique(key, return_inverse=True)
raw_sum = np.zeros((len(uk), 3))
np.add.at(raw_sum, inv, cf[:, None] * cn)
w = np.zeros(len(uk))
np.add.at(w, inv, np.abs(cf))
pmean = np.zeros((len(uk), 3))
np.add.at(pmean, inv, cp)
cnt = np.bincount(inv)
pmean /= cnt[:, None]
pw = np.zeros((len(uk), 3))
np.add.at(pw, inv, np.abs(cf)[:, None] * cp)
pw /= np.maximum(w, 1e-12)[:, None]
kk, bb, mm = uk // (B * M), (uk % (B * M)) // M, uk % M
ref = fm[kk, bb, mm]
refc = cpos[kk, bb, mm]
sel = np.linalg.norm(ref, axis=1) > 1e-3
ratio = np.einsum("ij,ij->i", raw_sum[sel], ref[sel]) / np.einsum("ij,ij->i", ref[sel], ref[sel])
relerr = np.linalg.norm(raw_sum[sel] - ref[sel], axis=1) / np.linalg.norm(ref[sel], axis=1)
d_mean = np.linalg.norm(pmean - refc, axis=1) * 1e3
d_w = np.linalg.norm(pw - refc, axis=1) * 1e3
d_w[~(w > 1e-6)] = np.nan          # pairs whose rows all carry zero force
align = np.einsum("ij,ij->i", cn, fm[ck, cb, cm]) >= 0
P(f"pairs with contact rows {len(uk)} ({sel.sum()} with |force_matrix|>1 mN); pairs with force_matrix but no rows: "
  f"{int((np.linalg.norm(fm, axis=-1) > 1e-3).sum() - sel.sum())}")
P(f"NEW parse sum(f_i n_i) vs force_matrix: ratio median {np.median(ratio):.6f} min {ratio.min():.6f} max {ratio.max():.6f}; "
  f"rel err median {np.median(relerr):.2e} max {relerr.max():.2e}")
P(f"contact normals pointing along the pair's force_matrix: {align.mean() * 100:.2f}% of rows; force scalars >= 0: "
  f"{(cf >= 0).mean() * 100:.2f}%")
P(f"unweighted mean of raw points vs contact_pos_w: max {np.nanmax(d_mean):.3e} mm (indexing check)")
P(f"pairs whose contact rows all have zero force: {int((~(w > 1e-6)).sum())} of {len(w)}; "
  f"rows with zero force: {(cf <= 0).sum()} of {len(cf)}")
P(f"force-weighted mean of raw points vs contact_pos_w (unweighted): median {np.nanmedian(d_w):.3f} mm, "
  f"p95 {np.nanpercentile(d_w, 95):.3f} mm, max {np.nanmax(d_w):.3f} mm")
fuk, finv = np.unique(fkey, return_inverse=True)
fsum = np.zeros((len(fuk), 3))
np.add.at(fsum, finv, fvec)
fref = ffm[fuk // (B * M), (fuk % (B * M)) // M, fuk % M]
fs = np.linalg.norm(fref, axis=1) > 1e-3
ferr = np.linalg.norm(fsum[fs] - fref[fs], axis=1) / np.linalg.norm(fref[fs], axis=1)
P(f"friction rows sum vs friction_forces_w: rel err median {np.median(ferr):.2e} max {ferr.max():.2e} (n={fs.sum()})")
P(f"rows per pair: median {np.median(cnt):.0f}, max {cnt.max()}; separation: median {np.median(csep) * 1e3:.3f} mm, "
  f"min {csep.min() * 1e3:.3f}, max {csep.max() * 1e3:.3f} mm")
res["raw_vs_agg"] = {"ratio_median": float(np.median(ratio)), "relerr_max": float(relerr.max()),
                     "normal_aligned_pct": float(align.mean() * 100), "wmean_vs_cpos_mm_median": float(np.nanmedian(d_w)),
                     "wmean_vs_cpos_mm_p95": float(np.nanpercentile(d_w, 95)), "friction_relerr_max": float(ferr.max()),
                     "old_ratio_median": float(np.median(Z["d_old_ratio"])) if len(Z["d_old_ratio"]) else None,
                     "count_overwritten_substeps": int((Z["d_count_overwritten"] > 0).sum()),
                     "saturated": int(Z["d_sat"].sum())}

# ======================================================================= 2. reproducibility
if args.run2:
    Z2 = np.load(args.run2 + "/raw.npz")
    same = {k: bool(np.array_equal(Z[k], Z2[k])) for k in ("crow", "frow", "fm", "ffm", "body_pose", "tgt_pose")}
    P(f"\n## 2(i). second replay {args.run2}: bit-identical {same}")
    if not same["crow"] and Z["crow"].shape == Z2["crow"].shape:
        P(f"   max |diff| contact rows {np.abs(Z['crow'] - Z2['crow']).max():.3e}")
    res["repro_bit_identical"] = same


# ======================================================================= geometry helpers
def qmat(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


CONVEX = ("convexHull", "boundingCube", "boundingSphere")


class Shape:
    def __init__(self, cols):
        self.parts = []
        for c in cols:
            v, f = c["verts"].astype(np.float64), c["faces"]
            if c["approx"] in CONVEX or c["approx"].startswith("primitive"):
                h = trimesh.convex.convex_hull(v)
                self.parts.append(("convex", h.face_normals.copy(), np.einsum("ij,ij->i", h.face_normals, h.triangles[:, 0])))
            else:
                # non-convex (convexDecomposition / triangle mesh / sdf): PhysX's own decomposition is not
                # available -> UNSIGNED distance to the source mesh surface (dense surface samples + KD-tree)
                m = trimesh.Trimesh(v, f, process=False)
                n = int(min(400000, max(20000, m.area / (0.0005 ** 2))))
                pts = np.r_[m.vertices, trimesh.sample.sample_surface(m, n, seed=0)[0]]
                self.parts.append(("mesh", cKDTree(pts), None))
        self.approx = [c["approx"] for c in cols]
        self.verts = np.concatenate([c["verts"] for c in cols]).astype(np.float64) if cols else np.zeros((0, 3))

    def sd(self, x):
        """signed distance (m), > 0 outside; union of parts (min)."""
        out = np.full(len(x), np.inf)
        for kind, a, b in self.parts:
            if kind == "convex":
                d = (x @ a.T - b[None]).max(1)
            else:
                d = a.query(x)[0]
            out = np.minimum(out, d)
        return out


def to_local(x, pose):
    R = qmat(pose[..., 3:7])
    return np.einsum("...ji,...j->...i", R, x - pose[..., :3])


rshape = {n: Shape(G["robot"][n]) for n in rbn if G["robot"].get(n)}
PADV = {}
tshape = {n: Shape(G["target"][n]) for n in tgt}
s2r = np.array([rbn.index(n) for n in names])
lag = lambda k: np.maximum(k - args.pose_lag, 0)  # noqa: E731

# ======================================================================= 2(iii) point-to-surface distances
P("\n## 2(iii). raw contact points vs collision surfaces (signed distance, + outside, mm)")
xl = to_local(cp, bpose[lag(ck), s2r[cb]])
xt = to_local(cp, tpose[lag(ck), TPOSE_IDX[cm]])
xl0 = to_local(cp, bpose[ck, s2r[cb]])
xt0 = to_local(cp, tpose[ck, TPOSE_IDX[cm]])
d_link = np.full(len(cp), np.nan)
for b in np.unique(cb):
    n = names[b]
    if n in rshape:
        i = cb == b
        d_link[i] = rshape[n].sd(xl[i])
d_tgt = np.full(len(cp), np.nan)
for m in np.unique(cm):
    i = cm == m
    d_tgt[i] = tshape[tgt[m]].sd(xt[i])
d_link0 = np.full(len(cp), np.nan)
d_tgt0 = np.full(len(cp), np.nan)
for b in np.unique(cb):
    if names[b] in rshape:
        d_link0[cb == b] = rshape[names[b]].sd(xl0[cb == b])
for m in np.unique(cm):
    d_tgt0[cm == m] = tshape[tgt[m]].sd(xt0[cm == m])
a0, b0 = np.abs(d_link0[np.isfinite(d_link0)]) * 1e3, np.abs(d_tgt0) * 1e3
P(f"(poses AFTER the substep, lag 0: link |d| median {np.median(a0):.3f} p95 {np.percentile(a0, 95):.3f}; target |d| median "
  f"{np.median(b0):.3f} p95 {np.percentile(b0, 95):.3f})")
P(f"poses of substep k-{args.pose_lag} (before the solve that produced the contacts):")
for lab, d in (("hand link collider", d_link), ("target collider", d_tgt)):
    a = np.abs(d[np.isfinite(d)]) * 1e3
    P(f"{lab:20s}: |d| median {np.median(a):.3f}, p95 {np.percentile(a, 95):.3f}, max {a.max():.3f}; "
      f"signed median {np.nanmedian(d) * 1e3:+.3f}")
P(f"target collider approximation: {[(k, sorted(set(v.approx))) for k, v in tshape.items()]} "
  "(non-convex approximations: UNSIGNED distance to the source mesh, not to PhysX's decomposition hulls)")
res["dist_link_mm"] = {"median": float(np.nanmedian(np.abs(d_link)) * 1e3), "p95": float(np.nanpercentile(np.abs(d_link), 95) * 1e3)}
res["dist_tgt_mm"] = {"median": float(np.nanmedian(np.abs(d_tgt)) * 1e3), "p95": float(np.nanpercentile(np.abs(d_tgt), 95) * 1e3)}
# compare with the separation reported per contact: the gap between the two surfaces at the point
P(f"sanity: separation reported by PhysX median {np.median(csep) * 1e3:.3f} mm vs d_link+d_tgt median "
  f"{np.nanmedian(d_link + d_tgt) * 1e3:.3f} mm")

# ======================================================================= 2(iv) continuity
P("\n## 2(iv). frame-to-frame continuity (link-local frame, consecutive substeps both in contact)")
pwl = to_local(pw, bpose[lag(kk), s2r[bb]])
pair = np.where(w > 1e-6, bb * M + mm, -1)
jumps, nn = [], []
for pr in np.unique(pair[pair >= 0]):
    i = np.where(pair == pr)[0]
    i = i[np.argsort(kk[i])]
    ks = kk[i]
    cons = np.where(np.diff(ks) == 1)[0]
    jumps += list(np.linalg.norm(pwl[i[cons + 1]] - pwl[i[cons]], axis=1) * 1e3)
for k0 in range(K - 1):
    a_ = np.where(ck == k0)[0]
    b_ = np.where(ck == k0 + 1)[0]
    if len(a_) and len(b_):
        for bi in np.intersect1d(cb[a_], cb[b_]):
            pa = xl[a_[cb[a_] == bi]]
            pb = xl[b_[cb[b_] == bi]]
            dd = np.linalg.norm(pb[:, None] - pa[None], axis=-1).min(1)
            nn += list(dd * 1e3)
jumps, nn = np.array(jumps), np.array(nn)
P(f"force-weighted pair point jump: median {np.median(jumps):.3f} mm, p95 {np.percentile(jumps, 95):.3f}, "
  f"max {jumps.max():.3f} (n={len(jumps)})")
P(f"per-point nearest-neighbour move: median {np.median(nn):.3f} mm, p95 {np.percentile(nn, 95):.3f} mm (n={len(nn)})")
res["continuity_mm"] = {"pair_jump_median": float(np.median(jumps)), "pair_jump_p95": float(np.percentile(jumps, 95)),
                        "nn_median": float(np.median(nn)), "nn_p95": float(np.percentile(nn, 95))}

# ======================================================================= joint frames from the USD (child frame)
st = Usd.Stage.Open(args.robot_usd)
jinfo = {}
for p in st.Traverse(Usd.TraverseInstanceProxies()):
    if not (p.IsA(UsdPhysics.RevoluteJoint)):
        continue
    j = UsdPhysics.RevoluteJoint(p)
    b1 = j.GetBody1Rel().GetTargets()
    if not b1:
        continue
    q1 = j.GetLocalRot1Attr().Get()
    R1 = qmat(np.array([q1.GetReal(), *q1.GetImaginary()]))
    ax = {"X": 0, "Y": 1, "Z": 2}[j.GetAxisAttr().Get()]
    jinfo[b1[0].name] = {"joint": p.GetName(), "anchor": np.array(j.GetLocalPos1Attr().Get(), float), "axis": R1[:, ax]}


def finger_frame(distal, pad):
    """o = distal joint anchor, u = finger direction, h = palmar direction (where +flexion moves the tip)."""
    J = jinfo[distal]
    o, a = J["anchor"], J["axis"] / np.linalg.norm(J["axis"])
    V = rshape[distal].verts
    if pad and pad in rshape:
        # pad link colliders expressed in the distal frame through the (fixed) relative pose of the first substep
        rel = to_local(to_world(rshape[pad].verts, bpose[0, rbn.index(pad)]), bpose[0, rbn.index(distal)])
        PADV[distal] = rel
        V = np.r_[V, rel]
    c = V.mean(0)
    u = c - o
    u -= a * np.dot(u, a)
    u /= np.linalg.norm(u)
    h = np.cross(a, u)
    # extend u to the far tip: farthest vertex along u
    return o, u, h, V, J["joint"]


def to_world(x, pose):
    R = qmat(pose[3:7])
    return x @ R.T + pose[:3]


# ======================================================================= 3/4. pad vs non-pad splitting
if args.hand == "allegro":
    groups = {f: (f"link_{i}_0", f"link_{i}_0_tip") for f, i in (("index", 3), ("middle", 7), ("ring", 11), ("thumb", 15))}
elif args.hand == "sharpa":
    groups = {f"{s}_{f}": (f"{s}_{f}_DP", f"{s}_{f}_elastomer") for s in ("right", "left")
              for f in ("thumb", "index", "middle", "ring", "pinky") if f"{s}_{f}_DP" in rbn}
else:
    groups = {f: (f"{f}distal", None) for f in ("th", "ff", "mf", "rf", "lf")}
P(f"\n## 3/4. position-based pad split ({args.hand}); rule R2 = s >= {args.s_frac:.2f} L (distal part) AND h >= h_c "
  "(palmar of the segment centre line), in the distal link frame; R1 = nearer collision surface (pad link vs phalanx)")
rows_out = []
F_true_all, F_est_all, F_tot_all, cls_true, cls_est1, cls_est2 = [], [], [], [], [], []
agg = {"n": 0, "w": 0.0, "ok1": 0, "ok2": 0, "w1": 0.0, "w2": 0.0}
shadow_tab = []
for g, (dist, pad) in groups.items():
    if dist not in names:
        continue
    o, u, h, V, jn = finger_frame(dist, pad)
    sV, hV = (V - o) @ u, (V - o) @ h
    L, hc = sV.max(), np.median(hV)
    bd = names.index(dist)
    bp = names.index(pad) if pad and pad in names else None
    # rows of the group (contact + friction), in the distal frame
    gi = np.where((cb == bd) | ((cb == bp) if bp is not None else False))[0]
    gf = np.where((fb == bd) | ((fb == bp) if bp is not None else False))[0]
    if not len(gi):
        P(f"{g:14s}: no contact rows")
        continue
    xd = to_local(cp[gi], bpose[lag(ck[gi]), rbn.index(dist)])
    xdf = to_local(fpnt[gf], bpose[lag(fk[gf]), rbn.index(dist)])
    r2 = lambda x: (((x - o) @ u) >= args.s_frac * L) & ((((x - o) @ h) >= hc) | (not args.palmar))  # noqa: E731
    lab2, lab2f = r2(xd), r2(xdf)
    truth = (cb[gi] == bp) if bp is not None else None
    truthf = (fb[gf] == bp) if bp is not None else None
    if bp is not None:
        xp = to_local(cp[gi], bpose[lag(ck[gi]), rbn.index(pad)])
        xpf = to_local(fpnt[gf], bpose[lag(fk[gf]), rbn.index(pad)])
        lab1 = rshape[pad].sd(xp) < rshape[dist].sd(xd)
        lab1f = rshape[pad].sd(xpf) < rshape[dist].sd(xdf)
    # per-row force vectors
    fvr = cf[gi, None] * cn[gi]
    wrow = np.linalg.norm(fvr, axis=1)
    out = {"group": g, "rows": int(len(gi)), "L_mm": float(L * 1e3), "joint": jn}
    sr, hr = ((xd - o) @ u) / L, ((xd - o) @ h) - hc
    if bp is not None and truth.any():
        pv = PADV[dist]
        out.update({"padlink_s_range": [float(((pv - o) @ u).min() / L), float(((pv - o) @ u).max() / L)],
                    "padlink_h_range_mm": [float((((pv - o) @ h) - hc).min() * 1e3), float((((pv - o) @ h) - hc).max() * 1e3)],
                    "truthpad_s_med": float(np.median(sr[truth])), "truthpad_h_med_mm": float(np.median(hr[truth]) * 1e3),
                    "truthpad_palmar_frac": float((hr[truth] >= 0).mean())})
        if (~truth).any():
            out.update({"truthnon_s_med": float(np.median(sr[~truth])), "truthnon_h_med_mm": float(np.median(hr[~truth]) * 1e3)})
    else:
        out.update({"s_med": float(np.median(sr)), "h_med_mm": float(np.median(hr) * 1e3), "palmar_frac": float((hr >= 0).mean())})
    if bp is not None:
        acc1 = float((lab1 == truth).mean()); acc2 = float((lab2 == truth).mean())
        wacc1 = float((wrow * (lab1 == truth)).sum() / wrow.sum()); wacc2 = float((wrow * (lab2 == truth)).sum() / wrow.sum())
        out.update({"acc_R1": acc1, "acc_R2": acc2, "wacc_R1": wacc1, "wacc_R2": wacc2, "truth_pad_frac": float(truth.mean())})
        agg["n"] += len(gi); agg["ok1"] += int((lab1 == truth).sum()); agg["ok2"] += int((lab2 == truth).sum())
        agg["w"] += wrow.sum(); agg["w1"] += (wrow * (lab1 == truth)).sum(); agg["w2"] += (wrow * (lab2 == truth)).sum()
    # per-substep forces: truth from the pad link's aggregated sensor values, estimates from classified rows
    ks = np.unique(np.r_[ck[gi], fk[gf]])
    for k in ks:
        ia, ib = ck[gi] == k, fk[gf] == k
        tot = fvr[ia].sum(0) + fvec[gf][ib].sum(0)
        est = {}
        for nm, la, lf in (("R2", lab2, lab2f), ("R1", lab1 if bp is not None else None, lab1f if bp is not None else None)):
            if la is None:
                continue
            est[nm] = (fvr[ia & la].sum(0) + fvec[gf][ib & lf].sum(0), fvr[ia & ~la].sum(0) + fvec[gf][ib & ~lf].sum(0))
        rec = {"k": int(k), "tot": float(np.linalg.norm(tot)), "R2_pad": float(np.linalg.norm(est["R2"][0])),
               "R2_non": float(np.linalg.norm(est["R2"][1]))}
        if bp is not None:
            tp = fm[k, bp].sum(0) + ffm[k, bp].sum(0)
            tn = fm[k, bd].sum(0) + ffm[k, bd].sum(0)
            rec.update({"true_pad": float(np.linalg.norm(tp)), "true_non": float(np.linalg.norm(tn)),
                        "R1_pad": float(np.linalg.norm(est["R1"][0])), "R1_non": float(np.linalg.norm(est["R1"][1]))})
        rows_out.append({"group": g, **rec})
    rec_g = [r for r in rows_out if r["group"] == g]
    if bp is not None:
        def cls(p_, n_):
            return 0 if (p_ > args.thr and n_ <= args.thr) else (1 if (n_ > args.thr and p_ <= args.thr) else
                                                                 (2 if p_ > args.thr else 3))
        ct = np.array([cls(r["true_pad"], r["true_non"]) for r in rec_g])
        c1 = np.array([cls(r["R1_pad"], r["R1_non"]) for r in rec_g])
        c2 = np.array([cls(r["R2_pad"], r["R2_non"]) for r in rec_g])
        tpv = np.array([r["true_pad"] for r in rec_g]); e1 = np.array([r["R1_pad"] for r in rec_g]); e2 = np.array([r["R2_pad"] for r in rec_g])
        tt = np.array([r["tot"] for r in rec_g])
        live = tt > args.thr
        out.update({"frames": int(live.sum()), "mixed_true_pct": float((ct[live] == 2).mean() * 100),
                    "cls_acc_R1": float((c1[live] == ct[live]).mean()), "cls_acc_R2": float((c2[live] == ct[live]).mean()),
                    "corr_R1": float(np.corrcoef(tpv[live], e1[live])[0, 1]) if live.sum() > 2 and tpv[live].std() > 0 else float("nan"),
                    "corr_R2": float(np.corrcoef(tpv[live], e2[live])[0, 1]) if live.sum() > 2 and tpv[live].std() > 0 else float("nan"),
                    "mae_R1_over_tot": float(np.median(np.abs(e1[live] - tpv[live]) / tt[live])),
                    "mae_R2_over_tot": float(np.median(np.abs(e2[live] - tpv[live]) / tt[live])),
                    "pad_share_true": float(tpv[live].sum() / max(tt[live].sum(), 1e-9))})
        F_true_all += list(tpv[live]); F_est_all += list(e2[live]); F_tot_all += list(tt[live])
    else:
        e2 = np.array([r["R2_pad"] for r in rec_g]); n2 = np.array([r["R2_non"] for r in rec_g]); tt = np.array([r["tot"] for r in rec_g])
        live = tt > args.thr
        c2 = np.array([0 if (p_ > args.thr and n_ <= args.thr) else (1 if (n_ > args.thr and p_ <= args.thr) else 2)
                       for p_, n_ in zip(e2[live], n2[live])])
        out.update({"frames": int(live.sum()), "pad_only_pct": float((c2 == 0).mean() * 100),
                    "nonpad_only_pct": float((c2 == 1).mean() * 100), "mixed_pct": float((c2 == 2).mean() * 100),
                    "pad_share_R2": float(e2[live].sum() / max(tt[live].sum(), 1e-9)),
                    "max_pad_N": float(e2.max()), "max_tot_N": float(tt.max())})
    shadow_tab.append(out)
    P(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}))
if agg["n"]:
    P(f"ALL groups, per row: R1 acc {agg['ok1'] / agg['n']:.4f} (force-weighted {agg['w1'] / agg['w']:.4f}), "
      f"R2 acc {agg['ok2'] / agg['n']:.4f} (force-weighted {agg['w2'] / agg['w']:.4f}), rows {agg['n']}")
    res["row_acc"] = {"R1": agg["ok1"] / agg["n"], "R1_w": agg["w1"] / agg["w"], "R2": agg["ok2"] / agg["n"], "R2_w": agg["w2"] / agg["w"]}
res["groups"] = shadow_tab
if args.hand == "shadow":
    # thumb middle phalanx contacts next to the distal split
    if "thmiddle" in names:
        bt = names.index("thmiddle")
        tm = np.linalg.norm(fm[:, bt].sum(1) + ffm[:, bt].sum(1), axis=1)
        P(f"thmiddle: substeps in contact {(tm > args.thr).sum()}, max {tm.max():.2f} N, median in contact "
          f"{np.median(tm[tm > args.thr]) if (tm > args.thr).any() else 0:.2f} N")
        res["thmiddle"] = {"substeps": int((tm > args.thr).sum()), "max": float(tm.max())}
json.dump(res, open(args.run + f"/analysis{args.tag}.json", "w"), indent=1, default=float)
json.dump(rows_out, open(args.run + f"/pad_frames{args.tag}.json", "w"))
P("ANALYSIS_DONE")
