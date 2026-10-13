# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacFF hybrid tactile video with the PALM zones (experiment; no simulation).

Input: a tactile_hybrid_video_record.py run made with a pad definition that has palm zones (``palm_rows`` /
``palm_zone`` / ``touch`` in contacts.npz). Per hand, two tile rows in the ``observation.touch`` pad order:

* fingertip pads little, ring, middle, index, thumb (missing finger = grey "n/a" tile): the TacFF of
  tactile_hybrid_video_render.py (16 x 12 nodes over the pad's (s, w) plane, Hertz-like Gaussian kernel
  sigma = clip(F^(1/3), 1, 4) mm, colour = normal force per node, arrow = shear per node), with w = the pad's
  thumb-side axis of the touch layout (tile: up = distal, right = thumb side);
* palm zones palm_r, palm_c, palm_l: the zone's palmar face on its (t thumb side, u distal) plane, 16 rows along u
  and as many columns as keep the cells about square; every palm contact row spreads (with the same kernel) its
  compressive normal component -F.h and its tangential (F.u, F.t), F = normal row + its share of the zone's
  friction, so the node sums equal ``observation.touch``'s palm slots (checked per frame). Palm contacts outside the
  zones (non-pad) are printed as a number next to the palm row.

Scales are fixed for every hand (same N_REF / S_REF as the fingertip TacFF). Layout: left third-person camera |
centre tiles (per hand: fingertip row + palm row) | right wrist camera; 30 fps H.264 yuv420p.
"""

import argparse
import json
import os
import subprocess

import cv2
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--rec", required=True)
ap.add_argument("--pads", required=True, help="tactile_pads/<robot_type>.npz (with palm zones)")
ap.add_argument("--out", required=True, help="output .mp4")
ap.add_argument("--title", default="")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--thr", type=float, default=0.1, help="N: contact threshold (= touch_contact)")
args = ap.parse_args()

N_REF, S_REF, S_ARROW, S_CAP = 2.0, 0.5, 1.0, 1.8     # same fixed scales as tactile_hybrid_video_render.py
PADS = ("little", "ring", "middle", "index", "thumb", "palm_r", "palm_c", "palm_l")
info = json.load(open(os.path.join(args.rec, "info.json")))
C = np.load(os.path.join(args.rec, "contacts.npz"))
img3 = np.load(os.path.join(args.rec, "third.npy"), mmap_mode="r")
imgw = np.load(os.path.join(args.rec, "wrist.npy"), mmap_mode="r")
Z = np.load(args.pads)
fingers = [str(x) for x in Z["fingers"]]
distal = [str(x) for x in Z["distal"]]
links = [str(x) for x in Z["links"]]
link_finger = Z["link_finger"].astype(int)
canon = [str(x) for x in Z["finger_canon"]]
fhand = [int(x) for x in Z["finger_hand"]]
hands = [str(x) for x in Z["hand_names"]]
palm_links = [str(x) for x in Z["palm_links"]]
palm_link_palm = [int(x) for x in Z["palm_link_palm"]]
nf, nh = len(fingers), len(hands)
T = int(info["steps"])
dt = float(info["step_dt"])
rows = C["rows"]
prows = C["palm_rows"]
pzone = C["palm_zone"]          # (T, n_palms, 4, 3) palm frame [non-pad, l, c, r]
pleft = C["palm_left"]
touch = C["touch"]               # (T, nh * 24)


def kernel_spread(nodes, node_on, X, Fn, Sh):
    """Gaussian spread (sigma = clip(Fn^(1/3), 1, 4) mm) of per-row normal Fn and shear Sh (2) onto nodes (mm)."""
    nn = np.zeros(len(nodes))
    ns = np.zeros((len(nodes), 2))
    for k in range(len(X)):
        sig = float(np.clip(max(Fn[k], 0.0) ** (1 / 3), 1.0, 4.0))
        wgt = np.exp(-((nodes - X[k]) ** 2).sum(1) / (2 * sig ** 2)) * node_on
        if wgt.sum() <= 1e-300:
            wgt = np.zeros(len(nodes))
            wgt[np.argmin(((nodes - X[k]) ** 2).sum(1))] = 1.0
        wgt /= wgt.sum()
        nn += Fn[k] * wgt
        ns += wgt[:, None] * Sh[k][None]
    return nn, ns


# ------------------------------------------------------------------ fingertip pads (as the TacFF renderer, w = thumb side)
F = []
for fi in range(nf):
    fr = Z[f"frame_{fi}"]
    o = fr[0]
    u, h, w = Z[f"touch_axes_{fi}"]
    S, Mk = [], []
    for li, ln in enumerate(links):
        if link_finger[li] != fi:
            continue
        x = Z[f"samples_{li}"].astype(np.float64)
        if ln != distal[fi]:
            rel = C[f"rel_{ln}"]
            x = x @ rel[3:].reshape(3, 3).T + rel[:3]
        S.append(x)
        Mk.append(Z[f"mask_pad_{li}"].astype(bool))
    S, Mk = np.concatenate(S), np.concatenate(Mk)
    padsw = np.c_[(S[Mk] - o) @ u, (S[Mk] - o) @ w] * 1e3
    smin, smax, wmin, wmax = padsw[:, 0].min(), padsw[:, 0].max(), padsw[:, 1].min(), padsw[:, 1].max()
    GS, GW = np.meshgrid(np.linspace(smax, smin, 16), np.linspace(wmin, wmax, 12), indexing="ij")
    nodes = np.c_[GS.ravel(), GW.ravel()]
    F.append({"o": o, "u": u, "w": w, "tree3d": cKDTree(S), "mask": Mk, "nodes": nodes, "shape": (16, 12),
              "node_on": cKDTree(padsw).query(nodes)[0] < 1.5})
fN = np.zeros((T, nf, 192))
fS = np.zeros((T, nf, 192, 2))
fF = np.zeros((T, nf))
fSh = np.zeros((T, nf))
for t in range(T):
    rr = rows[rows[:, 0] == t] if len(rows) else rows
    for fi in range(nf):
        q = rr[rr[:, 1] == fi]
        if not len(q):
            continue
        f = F[fi]
        x, fn, ff = q[:, 2:5], q[:, 5:8], q[:, 8:11]
        on = f["mask"][f["tree3d"].query(x)[1]]
        if not on.any():
            continue
        x, fn, ff = x[on], fn[on], ff[on]
        mag = np.linalg.norm(fn, axis=1)
        sw = np.c_[(x - f["o"]) @ f["u"], (x - f["o"]) @ f["w"]] * 1e3
        sh = np.c_[ff @ f["u"], ff @ f["w"]]
        fF[t, fi], fSh[t, fi] = mag.sum(), np.linalg.norm(sh.sum(0))
        fN[t, fi], fS[t, fi] = kernel_spread(f["nodes"], f["node_on"], sw, mag, sh)

# ------------------------------------------------------------------ palm zones
P = {}            # (palm, zone 1..3) -> grid
for j, ln in enumerate(palm_links):
    p = palm_link_palm[j]
    Tb = Z[f"palm_link_T_{j}"]
    S = Z[f"palm_samples_{j}"].astype(np.float64) @ Tb[:3, :3].T + Tb[:3, 3]
    zz = Z[f"palm_zone_{j}"]
    for k in (1, 2, 3):
        P.setdefault((p, k), []).append(S[zz == k])
GRID = {}
for (p, k), lst in P.items():
    S = np.concatenate(lst)
    c, u, h, t_ = Z[f"palm_frame_{p}"]
    tu = np.c_[(S - c) @ t_, (S - c) @ u] * 1e3
    tmin, tmax, umin, umax = tu[:, 0].min(), tu[:, 0].max(), tu[:, 1].min(), tu[:, 1].max()
    cell = (umax - umin) / 15
    ncol = int(max(3, round((tmax - tmin) / max(cell, 1e-6)) + 1))
    GU, GT = np.meshgrid(np.linspace(umax, umin, 16), np.linspace(tmin, tmax, ncol), indexing="ij")
    nodes = np.c_[GU.ravel(), GT.ravel()]                       # (u, t) like the finger tiles' (s, w)
    GRID[(p, k)] = {"nodes": nodes, "shape": (16, ncol), "c": c, "u": u, "h": h, "t": t_,
                    "node_on": cKDTree(tu[:, ::-1]).query(nodes)[0] < 0.75 * max(cell, (tmax - tmin) / max(ncol - 1, 1)),
                    "size_mm": [float(tmax - tmin), float(umax - umin)]}
n_palm = pzone.shape[1]
pN = {key: np.zeros((T, np.prod(g["shape"]))) for key, g in GRID.items()}
pS = {key: np.zeros((T, np.prod(g["shape"]), 2)) for key, g in GRID.items()}
pFn = np.zeros((T, n_palm, 4))
pSh = np.zeros((T, n_palm, 4))
resid = []
for t in range(T):
    rr = prows[prows[:, 0] == t] if len(prows) else prows
    for (p, k), g in GRID.items():
        q = rr[(rr[:, 1] == p) & (rr[:, 11] == k)]
        Fv = q[:, 5:8] + q[:, 8:11]                              # normal row + its friction share (palm frame)
        X = np.c_[(q[:, 2:5] - g["c"]) @ g["u"], (q[:, 2:5] - g["c"]) @ g["t"]] * 1e3
        fnr = -(Fv @ g["h"])
        shr = np.c_[Fv @ g["u"], Fv @ g["t"]]
        lf = pleft[t, p, k]
        if np.linalg.norm(lf) > 0:                               # zone friction without a normal row: zone centre
            X = np.r_[X, [[np.mean(g["nodes"][:, 0]), np.mean(g["nodes"][:, 1])]]]
            fnr = np.r_[fnr, -(lf @ g["h"])]
            shr = np.r_[shr, [[lf @ g["u"], lf @ g["t"]]]]
        if len(X):
            pN[(p, k)][t], pS[(p, k)][t] = kernel_spread(g["nodes"], g["node_on"], X, fnr, shr)
        zf = pzone[t, p, k]
        pFn[t, p, k] = -(zf @ g["h"])
        pSh[t, p, k] = np.hypot(zf @ g["u"], zf @ g["t"])
        hand = p if nh > 1 else 0
        slot = hand * 8 + PADS.index(("palm_l", "palm_c", "palm_r")[k - 1])
        tile = np.r_[pN[(p, k)][t].sum(), pS[(p, k)][t].sum(0)]
        resid.append(np.abs(tile - touch[t, 3 * slot:3 * slot + 3]).max())
    for p in range(n_palm):
        pFn[t, p, 0] = np.linalg.norm(pzone[t, p, 0])
resid = np.asarray(resid)

# ------------------------------------------------------------------ drawing
TILE_W, TILE_H, LAB = 150, 210, 30


def col(v):
    a = float(np.clip(v / N_REF, 0, 1))
    return (int(255 * min(1, 2 * a)), int(200 * min(1, 2 * (1 - a))), 0)


def draw_grid(canvas, nodes_shape, node_on, nn, ns):
    R_, C_ = nodes_shape
    mx, my = 12, 10
    cw_ = (TILE_W - 2 * mx) / max(C_ - 1, 1)
    ch_ = (TILE_H - 2 * my) / max(R_ - 1, 1)
    cell = min(cw_, ch_)
    ox = (TILE_W - cell * (C_ - 1)) / 2
    oy = LAB + (TILE_H - cell * (R_ - 1)) / 2
    rad = max(2, int(cell * 0.33))
    for kk in range(R_ * C_):
        r, c = divmod(kk, C_)
        cx, cy = int(ox + c * cell), int(oy + r * cell)
        if not node_on[kk]:
            cv2.circle(canvas, (cx, cy), 1, (70, 70, 70), -1)
            continue
        n = nn[kk]
        cv2.circle(canvas, (cx, cy), rad if n > 1e-3 else max(1, rad - 1), col(n) if n > 1e-3 else (0, 150, 0), -1)
        sv = ns[kk]
        ls = np.linalg.norm(sv)
        if ls > 0.02:
            L = min(S_ARROW * ls / S_REF, S_CAP) * cell
            d = sv / ls
            cv2.arrowedLine(canvas, (cx, cy), (int(cx + d[1] * L), int(cy - d[0] * L)), (255, 255, 255), 1,
                            tipLength=0.35)


def label(canvas, name, fn, sh, active):
    fn = 0.0 if abs(fn) < 5e-3 else fn                          # no "-0.0"
    cv2.putText(canvas, name, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Fn {fn:5.1f}N sh {sh:4.1f}N", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (180, 0, 0) if active else (90, 90, 90), 1, cv2.LINE_AA)


def finger_tile(t, hand, pad):
    canvas = np.full((TILE_H + LAB, TILE_W, 3), 255, np.uint8)
    canvas[LAB:] = 25
    idx = [i for i in range(nf) if fhand[i] == hand and canon[i] == pad]
    if not idx:
        canvas[LAB:] = 120
        cv2.putText(canvas, pad, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "n/a (no finger)", (20, LAB + TILE_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        return canvas
    fi = idx[0]
    draw_grid(canvas, F[fi]["shape"], F[fi]["node_on"], fN[t, fi], fS[t, fi])
    label(canvas, pad, fF[t, fi], fSh[t, fi], fF[t, fi] > args.thr)
    return canvas


def palm_tile(t, p, k):
    canvas = np.full((TILE_H + LAB, TILE_W, 3), 255, np.uint8)
    canvas[LAB:] = 25
    g = GRID.get((p, k))
    name = ("palm_l", "palm_c", "palm_r")[k - 1]
    if g is None:
        canvas[LAB:] = 120
        label(canvas, name + " (no zone)", 0, 0, False)
        return canvas
    draw_grid(canvas, g["shape"], g["node_on"], pN[(p, k)][t], pS[(p, k)][t])
    label(canvas, f"{name} {g['size_mm'][0]:.0f}x{g['size_mm'][1]:.0f}mm", pFn[t, p, k], pSh[t, p, k],
          pFn[t, p, k] > args.thr)
    return canvas


GAP = 4
mid_w = 5 * TILE_W + 6 * GAP
mid_h = nh * 2 * (TILE_H + LAB + GAP) + GAP + nh * 16
HEAD = 34
side_h = mid_h
side_w = int(round(side_h * info["image_hw"][1] / info["image_hw"][0] / 2) * 2)
WID = side_w * 2 + mid_w + 8
WID += WID % 2
HGT = HEAD + mid_h + (HEAD + mid_h) % 2


def frame(t):
    fr = np.full((HGT, WID, 3), 255, np.uint8)
    fr[HEAD:HEAD + side_h, :side_w] = cv2.resize(np.asarray(img3[t]), (side_w, side_h))
    fr[HEAD:HEAD + side_h, side_w + mid_w + 8:side_w * 2 + mid_w + 8] = cv2.resize(np.asarray(imgw[t]), (side_w, side_h))
    x_mid = side_w + 4
    y = HEAD + GAP
    for hand in range(nh):
        p = hand if n_palm > 1 else 0
        cv2.putText(fr, f"{hands[hand]} hand: fingertip pads (tile: up = distal, right = thumb side)", (x_mid + GAP, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        y += 16
        for c, pad in enumerate(PADS[:5]):
            x0 = x_mid + GAP + c * (TILE_W + GAP)
            fr[y:y + TILE_H + LAB, x0:x0 + TILE_W] = finger_tile(t, hand, pad)
        y += TILE_H + LAB + GAP
        for c, k in enumerate((3, 2, 1)):            # palm_r, palm_c, palm_l (touch order)
            x0 = x_mid + GAP + c * (TILE_W + GAP)
            fr[y:y + TILE_H + LAB, x0:x0 + TILE_W] = palm_tile(t, p, k)
        x0 = x_mid + GAP + 3 * (TILE_W + GAP)
        nonpad = pFn[t, p, 0]
        for ii, txt in enumerate(("palm zones (tile: up = distal,", "right = thumb side)",
                                  f"palm outside zones (non-pad):", f"|F| = {nonpad:5.2f} N")):
            cv2.putText(fr, txt, (x0 + 4, y + 40 + 18 * ii), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (180, 0, 0) if (ii == 3 and nonpad > args.thr) else (60, 60, 60), 1, cv2.LINE_AA)
        y += TILE_H + LAB + GAP
    txt = (f"{args.title} | TacFF + palm zones (fixed scale: red = {N_REF:.0f} N/node) | t {t * dt:5.2f}s frame {t}/{T} | "
           f"video {args.fps * args.stride * dt:.2f}x speed")
    cv2.putText(fr, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return fr


import imageio_ffmpeg  # noqa: E402

ff = imageio_ffmpeg.get_ffmpeg_exe()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WID}x{HGT}",
                         "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                         "-preset", "medium", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
palm_tot = pFn[:, :, 1:].clip(0).sum((1, 2))
pm_t = int(np.argmax(palm_tot)) - int(np.argmax(palm_tot)) % args.stride if palm_tot.max() > 0 else -1
want = {int(np.argmax(fF.sum(1))), int(0.75 * (T - 1))}
want = {t - t % args.stride for t in want} | ({pm_t} if pm_t >= 0 else set())
for t in range(0, T, args.stride):
    fr = frame(t)
    proc.stdin.write(fr.tobytes())
    if t in want:
        tag = "palmmax_" if t == pm_t else ""
        cv2.imwrite(args.out.replace(".mp4", f"_{tag}frame{t:04d}.png"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
proc.stdin.close()
proc.wait()
# ------------------------------------------------------------------ check summary
res = {"rec": args.rec, "steps": T, "hands": hands, "frames_written": len(range(0, T, args.stride)),
       "palm_tile_vs_touch_resid_max_N": float(resid.max()) if len(resid) else 0.0,
       "palm_tile_vs_touch_resid_median_N": float(np.median(resid)) if len(resid) else 0.0, "palms": []}
for p in range(n_palm):
    zc = (pFn[:, p, 1:] > args.thr)
    arrows = 0
    for k in (1, 2, 3):
        if (p, k) in pS:
            arrows += int(((np.linalg.norm(pS[(p, k)], axis=-1) > 0.02).any(1) & zc[:, k - 1]).sum())
    res["palms"].append({
        "palm": int(p), "hand": hands[p if nh > 1 else 0],
        "zone_contact_frames_any": int(zc.any(1).sum()),
        "zone_contact_frames": {n: int(zc[:, k - 1].sum()) for k, n in ((1, "palm_l"), (2, "palm_c"), (3, "palm_r"))},
        "max_Fn_N": {n: float(pFn[:, p, k].max()) for k, n in ((1, "palm_l"), (2, "palm_c"), (3, "palm_r"))},
        "max_sh_N": {n: float(pSh[:, p, k].max()) for k, n in ((1, "palm_l"), (2, "palm_c"), (3, "palm_r"))},
        "nonpad_frames": int((pFn[:, p, 0] > args.thr).sum()), "nonpad_max_N": float(pFn[:, p, 0].max()),
        "zone_contact_frames_with_arrows": arrows,
        "zone_grid": {n: {"shape": list(GRID[(p, k)]["shape"]), "size_mm_t_u": GRID[(p, k)]["size_mm"]}
                      for k, n in ((1, "palm_l"), (2, "palm_c"), (3, "palm_r")) if (p, k) in GRID}})
res["png_frames"] = sorted(want)
res["palm_max_frame"] = int(np.argmax(palm_tot)) if palm_tot.max() > 0 else None
json.dump(res, open(args.out.replace(".mp4", "_check.json"), "w"), indent=1)
print(f"[render-palm] wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, {WID}x{HGT}); resid max "
      f"{res['palm_tile_vs_touch_resid_max_N']:.2e} N; palms {[(q['zone_contact_frames_any'], q['max_Fn_N']) for q in res['palms']]}",
      flush=True)
