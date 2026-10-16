# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2 of the hybrid tactile videos (experiment): ManiFeel-style videos from a tactile_hybrid_video_record run.

No simulation. Per finger, the virtual fingerprint pad (``tactile_pads/<robot_type>.npz``, variant ``pad`` = 60 deg
or ``pad75``) is unrolled onto its (s, w) plane (s along the finger, tip at the top; w lateral) and two tactile
images are synthesised from the recorded raw contacts:

* TacFF: a 16 x 12 node grid over the pad; every pad contact spreads its normal force and its friction share
  (projected onto the pad plane) over the nodes with a Gaussian kernel whose width grows like a Hertz contact radius
  (sigma = clip(1 mm * (F / 1 N)^(1/3), 1, 4) mm). Node colour = normal force per node (green -> red, saturating at
  N_REF), arrow = shear per node (length = S_ARROW cells per S_REF newton, capped). The scales are FIXED for every
  hand and episode.
* TacRGB-like: height map = indentation delta(F) = 0.6 mm (F / 5 N)^(2/3) (<= 2 mm) minus the gap between each pad
  surface point and the target's collision surface (so the contact patch follows the object's shape), rendered with
  Isaac Lab TacSL's Taxim renderer (GelsightRender, GELSIGHT_R15 calibration, 320 x 240 px = 28 x 21 mm).

Layout: left third-person camera | centre per-finger tactile tiles | right wrist camera; 30 fps H.264 (yuv420p).
"""

import argparse
import importlib.util
import json
import time
import os
import subprocess
import sys
import types

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--rec", required=True)
ap.add_argument("--pads", required=True, help="tactile_pads/<robot_type>.npz")
ap.add_argument("--variant", default="pad")
ap.add_argument("--mode", choices=["tacff", "tacrgb"], required=True)
ap.add_argument("--out", required=True, help="output .mp4")
ap.add_argument("--title", default="")
ap.add_argument("--gelsight_dir", default="/workspace/dexverse/tactile_results_251008/inputs")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--stride", type=int, default=1, help="use every n-th control step")
ap.add_argument("--frames_png", type=int, nargs="*", default=None, help="control steps to also save as PNG")
args = ap.parse_args()
_T0 = time.perf_counter()   # timing (TACTILE_RENDER_TIMING=<json path> writes a breakdown; output unchanged)
_TIM = {"tile": 0.0, "compose": 0.0, "pipe": 0.0}

N_REF = 2.0      # N per node at full red
S_REF = 0.5      # N per node -> S_ARROW cells
S_ARROW = 1.0
S_CAP = 1.8      # max arrow length in cells
THR = 0.05       # N: a pad counts as touched above this

info = json.load(open(os.path.join(args.rec, "info.json")))
C = np.load(os.path.join(args.rec, "contacts.npz"))
img3 = np.load(os.path.join(args.rec, "third.npy"), mmap_mode="r")
imgw = np.load(os.path.join(args.rec, "wrist.npy"), mmap_mode="r")
Zd = np.load(args.pads)
fingers = [str(x) for x in Zd["fingers"]]
distal = [str(x) for x in Zd["distal"]]
links = [str(x) for x in Zd["links"]]
link_finger = Zd["link_finger"].astype(int)
nf = len(fingers)
T = int(info["steps"])
dt = float(info["step_dt"])
rows = C["rows"]            # t, finger, x(3) distal frame, fn(3) distal frame, ff(3) distal frame
dpose = C["distal_pose"]    # (T, nf, 12): pos(3) + R(9) of the distal link (pre-substep)
tpose = C["tgt_pose"]       # (T, M, 7)
tgt_names = [str(x) for x in C["tgt_names"]]


def short(name):
    n = name
    for side in ("right_", "left_"):
        if n.startswith(side):
            return ("R " if side == "right_" else "L ") + short(n[len(side):])
    table = {"link_3_0": "index", "link_7_0": "middle", "link_11_0": "ring", "link_15_0": "thumb", "thdistal": "thumb",
             "ffdistal": "index", "mfdistal": "middle", "rfdistal": "ring", "lfdistal": "little",
             "finger1_link4": "thumb", "finger2_link4": "index", "finger3_link4": "middle", "finger4_link4": "ring",
             "finger5_link4": "little", "thumb_distal": "thumb", "index_intermediate": "index",
             "middle_intermediate": "middle", "ring_intermediate": "ring", "pinky_intermediate": "little",
             "thumb_fingertip": "thumb", "fingertip": "index", "fingertip_2": "middle", "fingertip_3": "ring",
             "thumb_DP": "thumb", "index_DP": "index", "middle_DP": "middle", "ring_DP": "ring", "pinky_DP": "little"}
    return table.get(n, n)


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------------------ pads in the distal frame
F = []
for fi in range(nf):
    fr = Zd[f"frame_{fi}"]
    o, u, h = fr[0], fr[1], fr[2]
    w = np.cross(u, h)
    S, Mk = [], []
    for li, ln in enumerate(links):
        if link_finger[li] != fi:
            continue
        x = Zd[f"samples_{li}"].astype(np.float64)
        if ln != distal[fi]:
            rel = C[f"rel_{ln}"]
            x = x @ rel[3:].reshape(3, 3).T + rel[:3]
        S.append(x)
        Mk.append(Zd[f"mask_{args.variant}_{li}"].astype(bool))
    S, Mk = np.concatenate(S), np.concatenate(Mk)
    sw = np.c_[(S - o) @ u, (S - o) @ w] * 1e3          # mm
    padsw = sw[Mk]
    smin, smax = padsw[:, 0].min(), padsw[:, 0].max()
    wmin, wmax = padsw[:, 1].min(), padsw[:, 1].max()
    cs, cw = (smin + smax) / 2, (wmin + wmax) / 2
    # TacFF nodes (16 rows along s, 12 columns along w)
    gs = np.linspace(smax, smin, 16)
    gw = np.linspace(wmin, wmax, 12)
    GS, GW = np.meshgrid(gs, gw, indexing="ij")
    nodes = np.c_[GS.ravel(), GW.ravel()]
    tree_pad = cKDTree(padsw)
    node_on = tree_pad.query(nodes)[0] < 1.5
    # TacRGB pixels at half resolution (160 x 120) over 28 x 21 mm around the pad centre
    ps = np.linspace(cs + 14.0, cs - 14.0, 160)
    pw = np.linspace(cw - 10.5, cw + 10.5, 120)
    PS, PW = np.meshgrid(ps, pw, indexing="ij")
    pix = np.c_[PS.ravel(), PW.ravel()]
    dd, ii = tree_pad.query(pix, k=8)
    pix_on = dd[:, 0] < 1.0
    wk = 1.0 / np.maximum(dd, 0.05)
    pad3d = (S[Mk][ii] * wk[..., None]).sum(1) / wk.sum(1)[:, None]   # smoothed pad surface point per pixel
    F.append({"o": o, "u": u, "h": h, "w": w, "tree3d": cKDTree(S), "mask": Mk, "nodes": nodes, "node_on": node_on,
              "pix_on": pix_on, "pad3d": pad3d, "win": (smin, smax, wmin, wmax)})
tgt_pts = {n: C[f"surf_{n}"].astype(np.float64) for n in tgt_names}
tgt_tree = {n: cKDTree(v) for n, v in tgt_pts.items()}


def surf_dist(n, x, k=20):
    """Distance to the target surface: plane fitted to the k nearest surface samples (removes the sampling
    quantisation a plain nearest-sample distance would print into the height map)."""
    d, idx = tgt_tree[n].query(x, k=k)
    nb = tgt_pts[n][idx]                                   # (N, k, 3)
    c = nb.mean(1)
    cov = np.einsum("nki,nkj->nij", nb - c[:, None], nb - c[:, None])
    _, vec = np.linalg.eigh(cov)
    nrm = vec[:, :, 0]
    return np.where(d[:, 0] < 0.004, np.abs(np.einsum("ni,ni->n", x - c, nrm)), d[:, 0])

_T1 = time.perf_counter()
# ------------------------------------------------------------------ per-frame pad quantities
by_t = {}
for r in rows:
    by_t.setdefault(int(r[0]), []).append(r)
padF = np.zeros((T, nf))      # sum |normal| of pad rows
padS = np.zeros((T, nf))      # |sum projected friction| of pad rows
nonF = np.zeros((T, nf))
node_n = np.zeros((T, nf, 16 * 12))
node_s = np.zeros((T, nf, 16 * 12, 2))
for t, rr in by_t.items():
    rr = np.asarray(rr)
    for fi in range(nf):
        q = rr[rr[:, 1] == fi]
        if not len(q):
            continue
        f = F[fi]
        x, fn, ff = q[:, 2:5], q[:, 5:8], q[:, 8:11]
        on_pad = f["mask"][f["tree3d"].query(x)[1]]
        mag = np.linalg.norm(fn, axis=1)
        nonF[t, fi] = mag[~on_pad].sum()
        if not on_pad.any():
            continue
        x, mag, ff = x[on_pad], mag[on_pad], ff[on_pad]
        sw = np.c_[(x - f["o"]) @ f["u"], (x - f["o"]) @ f["w"]] * 1e3
        sh = np.c_[ff @ f["u"], ff @ f["w"]]
        padF[t, fi] = mag.sum()
        padS[t, fi] = np.linalg.norm(sh.sum(0))
        for k in range(len(x)):
            sig = float(np.clip(1.0 * mag[k] ** (1 / 3), 1.0, 4.0))
            wgt = np.exp(-((f["nodes"] - sw[k]) ** 2).sum(1) / (2 * sig ** 2)) * f["node_on"]
            if wgt.sum() <= 0:
                wgt = np.zeros(len(wgt))
                wgt[np.argmin(((f["nodes"] - sw[k]) ** 2).sum(1))] = 1.0
            wgt /= wgt.sum()
            node_n[t, fi] += mag[k] * wgt
            node_s[t, fi] += wgt[:, None] * sh[k][None]
print(f"[render] pad normal per node p95 {np.percentile(node_n[node_n > 1e-3], 95) if (node_n > 1e-3).any() else 0:.3f} N, "
      f"shear per node p95 {np.percentile(np.linalg.norm(node_s, axis=-1)[node_n > 1e-3], 95) if (node_n > 1e-3).any() else 0:.3f} N",
      flush=True)


_T2 = time.perf_counter()
# ------------------------------------------------------------------ Taxim renderer (pure function, stubbed loader)
def load_taxim():
    for name in ("isaaclab", "isaaclab.utils", "isaaclab.utils.assets"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["isaaclab.utils.assets"].retrieve_file_path = lambda path, **kw: path
    spec = importlib.util.spec_from_file_location(
        "visuotactile_render",
        "/workspace/isaaclab/source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/visuotactile_render.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = types.SimpleNamespace(base_data_path=args.gelsight_dir, sensor_data_dir_name="gelsight_r15_data",
                                background_path="bg.jpg", calib_path="polycalib.npz", real_background="real_bg.npy",
                                image_height=320, image_width=240, num_bins=120, mm_per_pixel=0.0877)
    return mod.GelsightRender(cfg, device="cpu")


taxim = load_taxim() if args.mode == "tacrgb" else None
_T3 = time.perf_counter()
bg_rgb = None


def tac_rgb(t, fi):
    global bg_rgb
    f = F[fi]
    ind = np.zeros(len(f["pix_on"]))
    Fp = padF[t, fi]
    if Fp > THR:
        p, R = dpose[t, fi, :3], dpose[t, fi, 3:].reshape(3, 3)
        xw = f["pad3d"][f["pix_on"]] @ R.T + p
        g = np.full(len(xw), np.inf)
        for m, n in enumerate(tgt_names):
            Rt, pt = qmat(tpose[t, m, 3:7]), tpose[t, m, :3]
            g = np.minimum(g, surf_dist(n, (xw - pt) @ Rt))
        delta = min(0.6e-3 * (Fp / 5.0) ** (2 / 3), 2e-3)
        ind_on = np.clip(delta - (g - g.min()), 0, None)
        ind[f["pix_on"]] = ind_on
    hm = cv2.GaussianBlur(ind.reshape(160, 120).astype(np.float32), (0, 0), 1.5)   # ~0.26 mm: gel smoothing
    hm = cv2.resize(hm, (240, 320), interpolation=cv2.INTER_LINEAR)
    img = taxim.render(torch.tensor(hm)[None].float())[0].numpy()
    return img


# ------------------------------------------------------------------ drawing
TILE_W, TILE_H, LAB = 203, 270, 34


def col(v):
    a = float(np.clip(v / N_REF, 0, 1))
    return (int(255 * min(1, 2 * a)), int(200 * min(1, 2 * (1 - a))), 0)   # RGB green -> yellow -> red


def tile(t, fi):
    f = F[fi]
    canvas = np.full((TILE_H + LAB, TILE_W, 3), 255, np.uint8)
    if args.mode == "tacrgb":
        im = tac_rgb(t, fi)
        canvas[LAB:] = cv2.resize(im, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    else:
        canvas[LAB:] = 25
        smin, smax, wmin, wmax = f["win"]
        mx, my = 16, 14
        cw_ = (TILE_W - 2 * mx) / 11
        ch_ = (TILE_H - 2 * my) / 15
        cell = min(cw_, ch_)
        for k, (s_, w_) in enumerate(f["nodes"]):
            r, c = divmod(k, 12)
            cx, cy = int(mx + c * cw_), int(LAB + my + r * ch_)
            if not f["node_on"][k]:
                cv2.circle(canvas, (cx, cy), 2, (70, 70, 70), -1)
                continue
            n = node_n[t, fi, k]
            cv2.circle(canvas, (cx, cy), 5 if n > 1e-3 else 3, col(n) if n > 1e-3 else (0, 150, 0), -1)
            sv = node_s[t, fi, k]
            ls = np.linalg.norm(sv)
            if ls > 0.02:
                L = min(S_ARROW * ls / S_REF, S_CAP) * cell
                d = sv / ls
                # image axes: x = w (right), y = -s (tip at the top)
                tip = (int(cx + d[1] * L), int(cy - d[0] * L))
                cv2.arrowedLine(canvas, (cx, cy), tip, (255, 255, 255), 2, tipLength=0.35)
    name = short(fingers[fi])
    cv2.putText(canvas, name, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Fn {padF[t, fi]:5.1f}N  sh {padS[t, fi]:4.1f}N", (6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (180, 0, 0) if padF[t, fi] > THR else (90, 90, 90), 1, cv2.LINE_AA)
    return canvas


rows_n = 2 if nf > 5 else 1
per_row = int(np.ceil(nf / rows_n))
mid_w = per_row * TILE_W + (per_row + 1) * 4
mid_h = rows_n * (TILE_H + LAB) + (rows_n + 1) * 4
HEAD = 30
side_h = mid_h
side_w = int(round(side_h * info["image_hw"][1] / info["image_hw"][0] / 2) * 2)
WID = side_w * 2 + mid_w + 8
WID += WID % 2
HGT = HEAD + mid_h + (HEAD + mid_h) % 2


def frame(t):
    fr = np.full((HGT, WID, 3), 255, np.uint8)
    fr[HEAD:HEAD + side_h, :side_w] = cv2.resize(np.asarray(img3[t]), (side_w, side_h))
    fr[HEAD:HEAD + side_h, side_w + mid_w + 8:side_w * 2 + mid_w + 8] = cv2.resize(np.asarray(imgw[t]), (side_w, side_h))
    for fi in range(nf):
        r, c = divmod(fi, per_row)
        x0 = side_w + 4 + 4 + c * (TILE_W + 4)
        y0 = HEAD + 4 + r * (TILE_H + LAB + 4)
        _a = time.perf_counter()
        fr[y0:y0 + TILE_H + LAB, x0:x0 + TILE_W] = tile(t, fi)
        _TIM["tile"] += time.perf_counter() - _a
    txt = (f"{args.title} | {args.mode.upper()} ({'60' if args.variant == 'pad' else '75'} deg pad) | t {t * dt:5.2f}s "
           f"frame {t}/{T} | video {args.fps * args.stride * dt:.2f}x speed")
    cv2.putText(fr, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return fr


import imageio_ffmpeg  # noqa: E402

ff = imageio_ffmpeg.get_ffmpeg_exe()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WID}x{HGT}",
                         "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                         "-preset", "medium", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
if args.frames_png is None:   # auto: max pad shear, max pad normal, 3/4 of the episode
    want_png = {int(np.argmax(padS.sum(1))), int(np.argmax(padF.sum(1))), int(0.75 * (T - 1))}
    want_png = {t - t % args.stride for t in want_png}
else:
    want_png = set(args.frames_png)
for t in range(0, T, args.stride):
    _a = time.perf_counter()
    fr = frame(t)
    _b = time.perf_counter()
    _TIM["compose"] += _b - _a
    proc.stdin.write(fr.tobytes())
    _TIM["pipe"] += time.perf_counter() - _b
    if t in want_png:
        cv2.imwrite(args.out.replace(".mp4", f"_frame{t:04d}.png"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
proc.stdin.close()
_T4 = time.perf_counter()
proc.wait()
_T5 = time.perf_counter()
if os.environ.get("TACTILE_RENDER_TIMING"):
    _n = len(range(0, T, args.stride))
    json.dump({"mode": args.mode, "frames": _n, "fingers": nf, "contact_rows": int(len(rows)),
               "setup_s": _T1 - _T0, "field_s": _T2 - _T1, "taxim_load_s": _T3 - _T2,
               "tile_ms_per_frame": _TIM["tile"] / _n * 1e3,
               "compose_ms_per_frame": (_TIM["compose"] - _TIM["tile"]) / _n * 1e3,
               "pipe_ms_per_frame": _TIM["pipe"] / _n * 1e3, "frames_loop_s": _T4 - _T3, "encode_tail_s": _T5 - _T4,
               "total_s": _T5 - _T0, "wall_ms_per_frame": (_T5 - _T0) / _n * 1e3},
              open(os.environ["TACTILE_RENDER_TIMING"], "w"), indent=1)
np.savez_compressed(args.out.replace(".mp4", "_series.npz"), padF=padF, padS=padS, nonF=nonF,
                    fingers=np.array(fingers), dt=dt, obj_z=tpose[:, 0, 2])
print(f"[render] wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, {WID}x{HGT}, {len(range(0, T, args.stride))} frames)")
