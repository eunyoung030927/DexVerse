"""cv3alt (261116): functional hand descriptors + batched functional IK for cross-hand PAIRED posture data.
New file (no existing module is modified). Design: artifacts/codec_v3alt/DESIGN.md §2.

Descriptor of a posture q of hand h (palm frame: x distal, y thumbward, z = x cross y; metres):
  c_f  pad centre (mean of PadFK's 64 pad samples of finger f), n_f pad normal (normalised mean normal)
  b_f  base = origin of the joint frame of the finger's first chain joint (fixed in the palm)
  R_f  finger frame at the OPEN prototype: x = c_f(open) - b_f, z = palm z orthogonalised; l_f = |c_f(open) - b_f|
  u_f  = R_f^T (c_f - b_f) / l_f   (finger shape, length/splay/mounting invariant)
  m_f  = R_f^T n_f                 (pad normal in the finger frame)
  o_f  = (c_f - c_T) / s           (opposition vector thumb->finger, s = hand scale = |middle tip| at q=0, PadFK.scale)
  t    = c_T / s, nT = n_T         (thumb pad position / normal in the palm frame)
"""
from __future__ import annotations

import numpy as np
import torch

from .arm_pads import PadFK, align_pair

FING = ("thumb", "index", "middle", "ring", "little")
W = {"u": 1.0, "m": 0.3, "o": 2.0, "t": 0.5, "nT": 0.2, "uT": 0.2, "reg": 1e-3, "align": 0.0}
# W_HYB: functional terms + the E2* retarget metric itself (pad-surface Kabsch contact residual, /s^2) as the main term;
# absolute thumb position dropped (Kabsch already removes the rigid palm placement)
W_HYB = {"u": 0.5, "m": 0.15, "o": 1.0, "t": 0.0, "nT": 0.1, "uT": 0.1, "reg": 1e-3, "align": 300.0}


def _open_proto(play_npz):
    names = [str(s) for s in play_npz["prototype_names"]]
    return np.asarray(play_npz["prototypes"][names.index("open")], np.float32)


class FuncHand:
    def __init__(self, hand, play_npz, metadata="metadata"):
        self.hand = hand
        self.fk = PadFK(hand, metadata)
        self.fk.eval()
        self.dof = self.fk.dof
        self.valid = np.array([bool(v) for v in self.fk.valid.tolist()])
        self.s = float(self.fk.scale)
        self.q_open = torch.tensor(_open_proto(play_npz))
        # base = joint-frame origin of first chain joint (parent pose @ t0), at q = 0 (parent of the first joint is palm-fixed)
        first = {}
        for fi, f in enumerate(FING):
            info = self.fk.spec["finger"][f]
            if info:
                first[info["chain_joints"][0]] = fi
        poses = self.fk.poses(torch.zeros(1, self.dof), normalized=True)
        base = np.zeros((5, 3), np.float32)
        for i, (body, parent, typ, axis, qi) in enumerate(self.fk.nodes):
            jname = self.fk.reference.joint[body]["name"]
            if jname in first:
                T = poses[parent] @ getattr(self.fk, f"t0_{i}")
                base[first[jname]] = T[0, :3, 3].detach().numpy()
        self.base = torch.tensor(base)
        with torch.no_grad():
            c, n = self.pads_cn(self.q_open[None])
        c = c[0]
        R = np.zeros((5, 3, 3), np.float32); L = np.ones(5, np.float32)
        for fi in range(5):
            if not self.valid[fi]:
                R[fi] = np.eye(3); continue
            v = (c[fi] - self.base[fi]).numpy()
            L[fi] = np.linalg.norm(v)
            x = v / L[fi]
            z = np.array([0, 0, 1.0]) - x[2] * x
            z /= np.linalg.norm(z)
            R[fi] = np.stack([x, np.cross(z, x), z], 1)  # columns = x, y, z
        self.R, self.L = torch.tensor(R), torch.tensor(L)

    def to(self, device):
        self.fk.to(device)
        self.base, self.R, self.L, self.q_open = (t.to(device) for t in (self.base, self.R, self.L, self.q_open))
        return self

    def pads_cn(self, q):
        pts, nrm, _ = self.fk.pads(q)
        c = pts.mean(2)
        n = torch.nn.functional.normalize(nrm.mean(2), dim=-1, eps=1e-8)
        return c, n

    def desc(self, q):
        c, n = self.pads_cn(q)
        return _desc_from(self, c, n)


def _desc_from(H, c, n):
    u = torch.einsum("fji,bfj->bfi", H.R, c - H.base[None]) / H.L[None, :, None]
    m = torch.einsum("fji,bfj->bfi", H.R, n)
    o = (c - c[:, :1]) / H.s
    return {"u": u, "m": m, "o": o, "t": c[:, 0] / H.s, "nT": n[:, 0], "c": c, "n": n}


def target_from(src, d, src_valid, tgt_valid, src_pads=None):
    """Functional target for a target hand from source descriptors d (torch dict) -> (target dict, finger weights (5,)).
    src_pads = source PadFK.pads(q) (pts, nrm, tips) for the E2*-metric term."""
    tgt = {k: d[k].clone() for k in ("u", "m", "o", "t", "nT")}
    if src_pads is not None:
        tgt["pp"], tgt["pn"], tgt["pt"] = src_pads
    fw = np.array([1.0 if (src_valid[f] and tgt_valid[f]) else 0.0 for f in range(5)], np.float32)
    if tgt_valid[4] and not src_valid[4]:  # 4 -> 5: little copies the ring's functional shape (ulnar-most rule)
        for k in ("u", "m"):
            tgt[k][:, 4] = d[k][:, 3]
        fw[4] = 0.5  # shape only (opposition weight for little is 0, see ik_loss)
    return tgt, fw


def ik_loss(H, q, tgt, fw, opp_w=None, wts=W, src_fk=None):
    if wts.get("align", 0) > 0 and src_fk is not None:
        pts, nrm, tips = H.fk.pads(q)
        c = pts.mean(2); n = torch.nn.functional.normalize(nrm.mean(2), dim=-1, eps=1e-8)
        d = _desc_from(H, c, n)
        r2 = align_pair(src_fk, H.fk, (tgt["pp"], tgt["pn"], tgt["pt"]), (pts, nrm, tips), differentiable_svd=True)["r2"]
    else:
        d = H.desc(q); r2 = None
    fwt = torch.tensor(fw, device=q.device)
    lu = ((d["u"] - tgt["u"]).square().sum(-1) * fwt).sum(-1)
    lm = ((1 - (d["m"] * tgt["m"]).sum(-1)) * fwt).sum(-1)
    fw_fing = fwt.clone(); fw_fing[0] = 0
    lu = lu - (1 - wts["uT"]) * (d["u"][:, 0] - tgt["u"][:, 0]).square().sum(-1) * fwt[0]
    if opp_w is None:
        gap = tgt["o"].norm(dim=-1)
        opp_w = 1 + 10 * torch.exp(-gap / 0.08)
    ow = fw_fing.clone(); ow[4] = ow[4] if fw[4] == 1.0 else 0.0
    lo = ((d["o"] - tgt["o"]).square().sum(-1) * opp_w * ow).sum(-1)
    lt = (d["t"] - tgt["t"]).square().sum(-1)
    lnt = 1 - (d["nT"] * tgt["nT"]).sum(-1)
    tot = wts["u"] * lu + wts["m"] * lm + wts["o"] * lo + wts["t"] * lt + wts["nT"] * lnt + wts["reg"] * q.square().sum(-1)
    if r2 is not None:
        tot = tot + wts["align"] * r2
    return tot, d


def solve(H, tgt, fw, inits, iters=250, lr=0.05, mask=None, wts=W, chunk=8000, src_fk=None):
    """Batched multi-start Adam IK. tgt: dict of (N,...) tensors; inits: list of (N,dof) tensors. mask (dof,) = movable joints.
    Returns best q (N,dof) and its loss."""
    N = len(inits[0]); dev = inits[0].device
    best_q = torch.empty(N, H.dof, device=dev); best_l = torch.full((N,), float("inf"), device=dev)
    mk = torch.ones(H.dof, device=dev) if mask is None else torch.as_tensor(mask, dtype=torch.float32, device=dev)
    for s in range(0, N, chunk):
        sl = slice(s, min(N, s + chunk))
        t = {k: v[sl] for k, v in tgt.items()}
        for q0 in inits:
            base = q0[sl].clone()
            x = base.clone().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=lr)
            for it in range(iters):
                q = x * mk + base * (1 - mk)
                loss, _ = ik_loss(H, q, t, fw, wts=wts, src_fk=src_fk)
                opt.zero_grad(set_to_none=True)
                loss.sum().backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(-0.995, 0.995)
                if it == int(iters * 0.6):
                    for g in opt.param_groups:
                        g["lr"] = lr * 0.3
            with torch.no_grad():
                q = x * mk + base * (1 - mk)
                loss, _ = ik_loss(H, q, t, fw, wts=wts, src_fk=src_fk)
                better = loss < best_l[sl]
                bq = best_q[sl]; bq[better] = q[better]; best_q[sl] = bq
                bl = best_l[sl]; bl[better] = loss[better]; best_l[sl] = bl
    return best_q, best_l


@torch.no_grad()
def residuals(H, q, tgt, fw):
    """Per-sample residual summary: shape error (fraction of finger length) per finger, opposition error (mm on target hand),
    thumb position error (mm), normal angle (deg)."""
    d = H.desc(q)
    fwt = torch.tensor(fw, device=q.device)
    ue = (d["u"] - tgt["u"]).norm(dim=-1)  # (N,5) in finger lengths
    oe = (d["o"] - tgt["o"]).norm(dim=-1) * H.s * 1000  # mm
    te = (d["t"] - tgt["t"]).norm(dim=-1) * H.s * 1000
    me = torch.rad2deg(torch.acos((d["m"] * tgt["m"]).sum(-1).clamp(-1, 1)))
    shape_fing = (ue[:, 1:] * fwt[1:]).sum(-1) / fwt[1:].clamp_min(0).sum().clamp_min(1)
    ow = fwt.clone(); ow[0] = 0; ow[4] = ow[4] if fw[4] == 1.0 else 0.0
    gap = tgt["o"].norm(dim=-1) * H.s * 1000
    pinch = (gap < 30) & (ow[None] > 0)
    opp_pinch_max = torch.where(pinch, oe, torch.zeros_like(oe)).max(-1).values
    c = lambda t: t.detach().cpu().numpy()
    return {"shape_err": c(ue), "opp_mm": c(oe), "thumb_mm": c(te), "normal_deg": c(me),
            "shape_mean_fingers": c(shape_fing), "opp_pinch_max_mm": c(opp_pinch_max), "has_pinch": c(pinch.any(-1))}


def feasible(res, shape_thr=0.15, pinch_thr=10.0, thumb_thr=25.0):
    return (res["shape_mean_fingers"] <= shape_thr) & (res["opp_pinch_max_mm"] <= pinch_thr) & (res["thumb_mm"] <= thumb_thr)


# ------------------------------------------------------------------ human (OpenXR 26 joints) descriptors
XR = ['palm', 'wrist', 'thumb_metacarpal', 'thumb_proximal', 'thumb_distal', 'thumb_tip', 'index_metacarpal', 'index_proximal',
      'index_intermediate', 'index_distal', 'index_tip', 'middle_metacarpal', 'middle_proximal', 'middle_intermediate',
      'middle_distal', 'middle_tip', 'ring_metacarpal', 'ring_proximal', 'ring_intermediate', 'ring_distal', 'ring_tip',
      'little_metacarpal', 'little_proximal', 'little_intermediate', 'little_distal', 'little_tip']


def _quat_R(q):  # [w,x,y,z] -> R
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def human_desc(J, pad_frac=0.3):
    """J (N,26,7) OpenXR right-hand joints [x,y,z,qw,qx,qy,qz] -> descriptor dict (torch) in the robot palm-frame convention.
    pad centre = tip - pad_frac*(tip - distal); pad normal = -Y of the distal joint frame (OpenXR: +Y dorsal);
    base: thumb_metacarpal / *_proximal (MCP); straight-finger direction for fingers = metacarpal->proximal bone, thumb = median
    direction metacarpal->pad over the 10 % most extended frames; l_f = bone-length sum base->pad; s = |wrist->middle_prox| + l_middle."""
    P = J[..., :3].astype(np.float64)
    ix = {n: i for i, n in enumerate(XR)}
    w, mi, mm, mr = P[:, ix["wrist"]], P[:, ix["index_proximal"]], P[:, ix["middle_proximal"]], P[:, ix["ring_proximal"]]
    x = mm - w; x /= np.linalg.norm(x, axis=-1, keepdims=True)
    y = mi - mr; y -= x * (x * y).sum(-1, keepdims=True); y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    Rp = np.stack([x, y, z], -1)  # palm frame columns (world)

    def to_palm(p):
        return np.einsum("nji,nj->ni", Rp, p - w)
    c = np.zeros((len(P), 5, 3)); n = np.zeros((len(P), 5, 3)); b = np.zeros((len(P), 5, 3))
    lengths = np.zeros(5)
    for fi, f in enumerate(FING):
        tip, dist = P[:, ix[f + "_tip"]], P[:, ix[f + "_distal"]]
        pad = tip - pad_frac * (tip - dist)
        c[:, fi] = to_palm(pad)
        Rd = _quat_R(J[:, ix[f + "_distal"], 3:7].astype(np.float64))
        n[:, fi] = np.einsum("nji,nj->ni", Rp, -Rd[:, :, 1])
        bn = f + ("_metacarpal" if f == "thumb" else "_proximal")
        b[:, fi] = to_palm(P[:, ix[bn]])
        chain = ([f + "_metacarpal", f + "_proximal", f + "_distal"] if f == "thumb" else
                 [f + "_proximal", f + "_intermediate", f + "_distal"])
        seg = sum(np.median(np.linalg.norm(P[:, ix[a]] - P[:, ix[bb]], axis=-1)) for a, bb in zip(chain[:-1], chain[1:]))
        lengths[fi] = seg + (1 - pad_frac) * np.median(np.linalg.norm(tip - dist, axis=-1))
    s = np.median(np.linalg.norm(P[:, ix["middle_proximal"]] - w, axis=-1)) + lengths[2] / (1 - 0.0)
    base = np.median(b, 0)
    R = np.zeros((5, 3, 3))
    for fi, f in enumerate(FING):
        if f == "thumb":
            v = c[:, 0] - base[0]; ext = np.linalg.norm(v, axis=-1)
            sel = ext >= np.quantile(ext, 0.9)
            xd = np.median(v[sel] / ext[sel, None], 0)
        else:
            xd = np.median(to_palm(P[:, ix[f + "_proximal"]]) - to_palm(P[:, ix[f + "_metacarpal"]]), 0)
        xd /= np.linalg.norm(xd)
        zz = np.array([0, 0, 1.0]) - xd[2] * xd; zz /= np.linalg.norm(zz)
        R[fi] = np.stack([xd, np.cross(zz, xd), zz], 1)
    u = np.einsum("fji,nfj->nfi", R, c - base[None]) / lengths[None, :, None]
    m = np.einsum("fji,nfj->nfi", R, n)
    o = (c - c[:, :1]) / s
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    return {"u": T(u), "m": T(m), "o": T(o), "t": T(c[:, 0] / s), "nT": T(n[:, 0]), "c": T(c), "n": T(n)}, \
        {"s": float(s), "lengths": lengths.tolist()}
