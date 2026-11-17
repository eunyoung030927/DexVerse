"""codec v3 Phase B (261116): taxonomy-based PAIRED posture data from artifacts/codec_v3/TAXONOMY_SPEC.md (Feix-corrected,
TAXONOMY_SPEC_CHANGES.md applied). New file, offline data generation only (FK-gradient IK, no simulation).

Every static posture = functional template (DESIGN.md §3):
  obj   'pinch'  : listed contact points of the fingers at distance a*s from the thumb contact point (pad/tip/side)
        'sphere' : contact points on a sphere of diameter a*s (free centre); 'cyl' : on a cylinder (axis ~ palm y, rolled
                   by theta_roll), 'palm' flag = palm point on the object surface (palm opposition)
        'kap'    : thumb pad onto a Kapandji landmark;  'none' : flexion / spread / thumb only
  flex  per finger level E/S/M/F (or a range 'S-M', or a (proximal, distal) tuple for hooks), spread N/m/W, thumb ABD/ADD
Points (palm frame, metres; s = hand scale = |middle tip| at the open pose): 'pad' = pad-sample centroid (+ mean normal),
'tip' = fingertip link origin, 'PP'/'MP'/'DP' = phalanx bone midpoints (+0.04 s finger half-thickness on object surfaces),
'rside'/'uside' + PP/MP/DP = bone midpoint +/- 0.045 s along palm y (radial/ulnar side), 'palm' = mid-palm surface point.
Same theta (aperture, spread, per-finger flexion position inside the level, cylinder roll, IK seed) for every hand.
4-finger hands: the ulnar-most (ring) finger takes the little-finger role -> little constraints are dropped (ring keeps its
own); postures that need the little finger itself are masked ('L4' = invalid on 4-finger hands); wrap apertures above
0.75 * a_max are masked on 4-finger hands (spec §5.2, draft value). A hand's sample is valid if: contact / surface error
within tolerance, no pad-pad interpenetration between non-contact fingers (min distance >= 2 mm), joints inside
+-0.95 (normalized). Sample kept if >= 3 hands valid. Dynamic primitives: joint-space interpolation between the paired IK
solutions of their start/end static postures (same t for all hands) where kinematically definable; the others are logged.
Container python, cwd /workspace/xhand/latent_codec_v1:
  python.sh -m xhand_latent.cv3_taxonomy --out artifacts/codec_v3/data/tax_pairs.npz [--per-posture 256] [--coverage]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .arm_pads import PadFK
from .kinematics import FINGERS
from .play import prototypes

HANDS = ("allegro", "leap", "shadow", "wuji", "sharpa")
FOUR = ("allegro", "leap")
FI = {"T": 0, "I": 1, "M": 2, "R": 3, "L": 4}
LEVEL = {"E": (0.0, 0.10), "S": (0.10, 0.35), "M": (0.35, 0.65), "F": (0.65, 0.92)}
SPREAD = {"N": (0.0, 0.15), "m": (0.30, 0.60), "W": (0.70, 1.0)}
THICK = 0.04   # finger half thickness / s (bone point -> surface)
SIDE = 0.045   # lateral side offset / s
WRAP4 = 0.75   # 4-finger wrap aperture factor (draft)


def lvl(code):
    if "-" in code:
        a, b = code.split("-")
        return LEVEL[a][0], LEVEL[b][1]
    return LEVEL[code]


def P(pid, fam, obj, a=None, contacts=(), flex=None, spread="N", thumb=None, palm=False, need_l=False, wrap=False,
      kap=None, feix=None, name=""):
    return dict(id=pid, fam=fam, obj=obj, a=a, contacts=list(contacts), flex=flex or {}, spread=spread, thumb=thumb,
                palm=palm, need_l=need_l, wrap=wrap, kap=kap, feix=feix, name=name)


W4 = [("I", "PP"), ("I", "MP"), ("I", "DP"), ("M", "PP"), ("M", "MP"), ("M", "DP"), ("R", "PP"), ("R", "MP"), ("R", "DP"),
      ("L", "PP"), ("L", "MP"), ("L", "DP")]
ALLF = {"I": "M", "M": "M", "R": "M", "L": "M"}


def F(**kw):
    return {k: v for k, v in kw.items()}


# ---- static postures (Feix-corrected spec; S-P13 = S-I01 and S-R08 = S-P08 deduplicated; S-N14 = S-I04 and S-N10 = S-N01
# removed as kinematic duplicates; Feix #13 Precision Sphere -> S-R19, Feix #32 Ventral -> S-I07 added) ----
POSTURES = [
    # Power (palm opposition: VF1 = palm)
    P("S-P01", "power", "cyl", (0.35, 0.55), W4 + [("T", "pad")], F(I="S-M", M="S-M", R="S-M", L="S-M"), "m", "ABD", palm=True, wrap=True, feix=1, name="Large diameter"),
    P("S-P02", "power", "cyl", (0.12, 0.25), W4 + [("T", "pad")], F(I="M-F", M="M-F", R="M-F", L="M-F"), "N", "ABD", palm=True, wrap=True, feix=2, name="Small diameter"),
    P("S-P03", "power", "cyl", (0.20, 0.35), W4 + [("T", "pad")], F(I="M", M="M", R="M", L="M"), "N", "ABD", palm=True, wrap=True, feix=3, name="Medium wrap"),
    P("S-P04", "power", "cyl", (0.15, 0.30), W4 + [("T", "pad|rside:I:PP")], F(I="M-F", M="M-F", R="M-F", L="M-F"), "N", "ADD", palm=True, wrap=True, feix=4, name="Adducted thumb"),
    P("S-P05", "power", "cyl", (0.15, 0.25), W4 + [("T", "pad")], F(I="M-F", M="M-F", R="M-F", L="M-F"), "N", "ADD", palm=True, wrap=True, feix=5, name="Light tool"),
    P("S-P06", "power", "sphere", (0.45, 0.80), [("I", "MP"), ("I", "DP"), ("M", "MP"), ("M", "DP"), ("R", "MP"), ("R", "DP"), ("L", "MP"), ("L", "DP"), ("T", "pad")], F(I="S-M", M="S-M", R="S-M", L="S-M"), "W", "ABD", palm=True, wrap=True, feix=11, name="Power sphere"),
    P("S-P07", "power", "sphere", (0.50, 0.80), [("I", "DP"), ("M", "DP"), ("R", "DP"), ("L", "DP"), ("T", "pad")], F(I="S", M="S", R="S", L="S"), "W", "ABD", palm=True, wrap=True, feix=10, name="Power disk"),
    P("S-P08", "power", "sphere", (0.35, 0.60), [("I", "pad"), ("M", "pad"), ("R", "pad"), ("T", "pad")], F(I="M", M="M", R="M", L="M"), "m", "ABD", palm=True, feix=26, name="Sphere 4-finger (pad)"),
    P("S-P09", "power", "sphere", (0.20, 0.40), [("I", "DP"), ("M", "DP"), ("R", "DP"), ("L", "DP")], F(I="S-M", M="S-M", R="S-M", L="S-M"), "N", "ADD", palm=True, wrap=True, feix=30, name="Palmar"),
    P("S-P10", "power", "cyl", (0.10, 0.30), [("T", "pad"), ("I", "MP"), ("I", "DP")], F(I="M", M="F", R="F", L="F"), "N", "ABD", feix=31, name="Ring"),
    P("S-P11", "power", "none", None, [], F(I=("E-S", "F"), M=("E-S", "F"), R=("E-S", "F"), L=("E-S", "F")), "N", "ADD", feix=15, name="Fixed hook"),
    P("S-P12", "intermediate", "pinch", (0.10, 0.20), [("T", "pad"), ("I", "rside:MP")], F(I="M", M="M", R="M", L="M"), "N", "ADD", feix=29, name="Stick"),
    P("S-P14", "power", "pinch", (0.03, 0.12), [("T", "pad"), ("I", "pad"), ("M", "pad"), ("R", "pad")], F(I="E", M="E", R="E", L="E"), "N", "ABD", feix=18, name="Extension type"),
    P("S-P15", "power", "cyl", (0.12, 0.25), [x for x in W4 if x[0] != "I"], F(I="E", M="M-F", R="M-F", L="M-F"), "N", "ADD", palm=True, wrap=True, feix=17, name="Index finger extension"),
    P("S-P16", "power", "cyl", (0.10, 0.20), [("I", "DP"), ("M", "DP"), ("R", "DP"), ("L", "DP"), ("T", "pad")], F(I="M-F", M="M-F", R="M-F", L="M-F"), "N", "ABD", feix=19, name="Distal type"),
    # Intermediate (side opposition)
    P("S-I01", "intermediate", "pinch", (0.02, 0.08), [("T", "pad"), ("I", "rside:MP")], F(I="S-M", M="F", R="F", L="F"), "N", "ADD", feix=16, name="Lateral pinch"),
    P("S-I02", "intermediate", "pinch", (0.03, 0.10), [("T", "pad"), ("M", "rside:DP")], F(I="M", M="M", R="F", L="F"), "N", "ADD", feix=25, name="Lateral tripod"),
    P("S-I03", "intermediate", "pinch", (0.02, 0.06), [("I", "uside:MP"), ("M", "rside:MP")], F(I="M-F", M="M-F", R="F", L="F"), "N", "ADD", name="Stick between fingers"),
    P("S-I04", "intermediate", "pinch", (0.01, 0.04), [("I", "uside:MP"), ("M", "rside:MP")], F(I="E-S", M="E-S", R="E-S", L="E-S"), "N", "ABD", feix=23, name="Adduction grip"),
    P("S-I05", "precision", "pinch", (0.02, 0.06), [("T", "pad"), ("M", "rside:DP"), ("I", "pad")], F(I="M", M="M", R="F", L="F"), "N", "ABD", feix=20, name="Writing tripod"),
    P("S-I06", "intermediate", "pinch", (0.03, 0.08), [("T", "pad"), ("M", "rside:DP"), ("R", "rside:DP")], F(I="M", M="M", R="M", L="F"), "N", "ABD", feix=21, name="Tripod variation"),
    P("S-I07", "intermediate", "pinch", (0.05, 0.15), [("T", "pad"), ("I", "rside:PP")], F(I="S", M="S-M", R="S-M", L="S-M"), "N", "ADD", feix=32, name="Ventral"),
    # Precision (pad opposition)
    P("S-R01", "precision", "pinch", (0.01, 0.05), [("T", "tip"), ("I", "tip")], F(I="M", M="F", R="F", L="F"), "N", "ABD", feix=24, name="Tip pinch"),
    P("S-R02", "precision", "pinch", (0.01, 0.08), [("T", "pad"), ("I", "pad")], F(I="M-F", M="F", R="F", L="F"), "N", "ABD", feix=9, name="Palmar pinch"),
    P("S-R03", "precision", "pinch", (0.03, 0.10), [("T", "pad"), ("I", "pad")], F(I="F", M="F", R="F", L="F"), "N", "ABD", feix=33, name="Inferior pincer"),
    P("S-R04", "precision", "pinch", (0.02, 0.10), [("T", "tip"), ("I", "tip"), ("M", "tip")], F(I="M", M="M", R="F", L="F"), "N", "ABD", feix=14, name="Tripod"),
    P("S-R05", "precision", "pinch", (0.03, 0.12), [("T", "tip"), ("I", "tip"), ("M", "tip"), ("R", "tip")], F(I="M", M="M", R="M", L="F"), "m", "ABD", name="Quadpod"),
    P("S-R06", "precision", "pinch", (0.03, 0.12), [("T", "tip"), ("I", "tip"), ("M", "tip"), ("R", "tip"), ("L", "tip")], F(I="M", M="M", R="M", L="M"), "m", "ABD", name="Five-pad pinch"),
    P("S-R07", "precision", "sphere", (0.20, 0.45), [("T", "pad"), ("I", "pad"), ("M", "pad")], F(I="S-M", M="S-M", R="F", L="F"), "m", "ABD", feix=28, name="Sphere 3-finger"),
    P("S-R09", "precision", "sphere", (0.25, 0.55), [("T", "pad"), ("I", "pad"), ("M", "pad"), ("R", "pad"), ("L", "pad")], F(I="S", M="S", R="S", L="S"), "W", "ABD", feix=12, name="Precision disk"),
    P("S-R10", "precision", "sphere", (0.15, 0.40), [("T", "pad"), ("I", "pad")], F(I="S-M", M="F", R="F", L="F"), "N", "ABD", name="Precision sphere 2-finger"),
    P("S-R11", "precision", "pinch", (0.02, 0.10), [("T", "pad"), ("I", "pad"), ("M", "pad"), ("R", "pad"), ("L", "pad")], F(I="M", M="M", R="M", L="M"), "N", "ABD", feix=6, name="Prismatic 4-finger"),
    P("S-R12", "precision", "pinch", (0.02, 0.10), [("T", "pad"), ("I", "pad"), ("M", "pad"), ("R", "pad")], F(I="M", M="M", R="M", L="F"), "N", "ABD", feix=7, name="Prismatic 3-finger"),
    P("S-R13", "precision", "pinch", (0.02, 0.08), [("T", "pad"), ("I", "pad"), ("M", "pad")], F(I="M", M="M", R="F", L="F"), "N", "ABD", feix=8, name="Prismatic 2-finger"),
    P("S-R14", "precision", "pinch", (0.02, 0.08), [("T", "pad"), ("I", "pad"), ("M", "pad"), ("R", "pad"), ("L", "pad")], F(I="E-S", M="E-S", R="E-S", L="E-S"), "N", "ADD", feix=22, name="Parallel extension"),
    P("S-R15", "precision", "pinch", (0.01, 0.05), [("T", "pad"), ("I", "rside:MP")], F(I="M", M="F", R="F", L="F"), "N", "ADD", name="Pad-to-side pinch"),
    P("S-R16", "precision", "pinch", (0.01, 0.02), [("T", "tip"), ("I", "tip")], F(I="S", M="F", R="F", L="F"), "N", "ABD", name="Distal pad (thin)"),
    P("S-R17", "precision", "pinch", (0.01, 0.04), [("T", "tip"), ("I", "tip"), ("M", "tip")], F(I="M-F", M="M-F", R="F", L="F"), "N", "ABD", name="Small-ball pinch"),
    P("S-R18", "precision", "pinch", (0.05, 0.20), [("T", "tip"), ("I", "tip")], F(I="E-S", M="F", R="F", L="F"), "N", "ABD", name="Extension pinch"),
    P("S-R19", "precision", "sphere", (0.15, 0.35), [("T", "pad"), ("I", "pad"), ("M", "pad"), ("R", "pad"), ("L", "pad")], F(I="S-M", M="S-M", R="S-M", L="S-M"), "m", "ABD", feix=13, name="Precision sphere"),
    # Kapandji (thumb pad onto landmark; other fingers neutral / slight flexion)
    *[P(f"S-K{k:02d}", "kapandji", "kap", None, [], F(I="E-S", M="E-S", R="E-S", L="E-S"), "N", None, kap=lm, need_l=nl, name=f"Kapandji {k}")
      for k, lm, nl in ((1, ("I", "rside:PP"), False), (2, ("I", "rside:MP"), False), (3, ("I", "tip"), False), (4, ("M", "tip"), False),
                        (5, ("R", "tip"), False), (6, ("L", "tip"), True), (7, ("L", "DP"), True), (8, ("L", "MP"), True),
                        (9, ("U", "PP"), False), (10, ("U", "palm"), False))],
    # Non-prehensile / anchors / daily
    P("S-N01", "nonpre", "none", None, [], F(I="E", M="E", R="E", L="E", T="E"), "N", "ADD", name="Open flat hand"),
    P("S-N02", "nonpre", "none", None, [], F(I="E", M="E", R="E", L="E", T="E"), "W", "ABD", name="Open spread"),
    P("S-N03", "nonpre", "none", None, [], F(I="F", M="F", R="F", L="F", T="M"), "N", "ADD", name="Fist"),
    P("S-N04", "nonpre", "none", None, [], F(I="F", M="F", R="F", L="F", T="E"), "N", "ABD", name="Fist, thumb extended"),
    P("S-N05", "nonpre", "none", None, [], F(I="E", M="F", R="F", L="F", T="M"), "N", "ADD", name="Pointing"),
    P("S-N06", "nonpre", "none", None, [], F(I="E", M="E", R="F", L="F", T="M"), "W", "ADD", name="V sign"),
    P("S-N07", "nonpre", "none", None, [], F(I="E", M="E", R="E", L="F", T="M"), "m", "ADD", need_l=True, name="Three-finger extension"),
    P("S-N08", "nonpre", "pinch", (0.005, 0.02), [("T", "tip"), ("I", "tip")], F(I="M", M="E", R="E", L="E"), "m", "ABD", name="OK sign"),
    P("S-N09", "nonpre", "none", None, [], F(I=("E-S", "F"), M=("E-S", "F"), R=("E-S", "F"), L=("E-S", "F"), T="S"), "N", "ADD", name="Claw"),
    P("S-N11", "nonpre", "none", None, [], F(I="S", M="F", R="F", L="F", T="M"), "N", "ADD", name="Button press (index)"),
    P("S-N12", "nonpre", "none", None, [], F(I="F", M="F", R="F", L="F", T="S"), "N", "ABD", name="Thumb press"),
    P("S-N13", "nonpre", "none", None, [], F(I="S", M="S", R="S", L="S", T="S"), "N", "ABD", name="Scoop / cup"),
    P("S-N15", "nonpre", "none", None, [], F(I="E-S", M="E-S", R="E-S", L="E-S", T="E-S"), "N", "ADD", name="Palm-up present"),
    P("S-N16", "nonpre", "sphere", (0.15, 0.30), [("T", "pad"), ("I", "pad"), ("M", "pad")], F(I="S-M", M="S-M", R="F", L="F"), "m", "ABD", name="Grip-and-twist prep"),
    P("S-N17", "nonpre", "pinch", (0.01, 0.03), [("T", "tip"), ("M", "pad")], F(I="F", M="F", R="F", L="F"), "N", "ABD", name="Snap prep"),
    P("S-N18", "nonpre", "count", None, [], {}, "m", "ADD", name="Counting (E/F subsets)"),
]
# counting patterns (I, M, R, L, T): 1 = extended
COUNT = [(1, 0, 0, 0, 0), (1, 1, 0, 0, 0), (1, 1, 1, 0, 0), (1, 1, 1, 1, 0), (1, 1, 1, 1, 1), (0, 0, 0, 0, 1), (0, 0, 0, 1, 0),
         (1, 0, 0, 1, 0), (0, 1, 0, 0, 0), (1, 0, 0, 0, 1), (0, 1, 1, 1, 0), (1, 1, 0, 0, 1)]
# dynamic primitives: (id, start, end, theta override) -> joint-space interpolation of paired IK solutions
DYNAMIC = [("D01", "S-R01", "S-R18", "Pinch open/close"), ("D02", "S-R04", "S-R05", "Dynamic tripod"),
           ("D03", "S-P09", "S-P03", "Squeeze"), ("D09", "S-R11", "S-P09", "Finger-to-palm translation"),
           ("D10", "S-P09", "S-R04", "Palm-to-finger translation"), ("D15", "S-K03", "S-K05", "Kapandji sweep")]
DYNAMIC_SKIPPED = {"D04": "rock (object rotation, needs dynamics)", "D05": "twiddle (rolling contact)",
                   "D06": "radial/ulnar roll (rolling contact)", "D07": "linear step (finger gaiting / regrasp)",
                   "D08": "rotary step (regrasp)", "D11": "simple shift (sliding contact)", "D12": "complex shift (sliding)",
                   "D13": "simple rotation (object spin)", "D14": "complex rotation (contact release)"}
HELDOUT = ["S-P03", "S-P10", "S-I02", "S-R05", "S-R13", "S-R19", "S-K04", "S-N06", "S-N16", "D02"]


class Hand:
    def __init__(self, h, play, device):
        self.h, self.device = h, device
        fk = PadFK(h)
        self.fk = fk.to(device)
        self.four = h in FOUR
        names = [str(s) for s in play["prototype_names"]]
        self.open = torch.tensor(play["prototypes"][names.index("open")], device=device)
        proto, pn = prototypes(fk.cpu()); fk.to(device)
        fist = torch.tensor(proto[pn.index("fist")], device=device)
        self.abd = (fist - self.open).abs() < 1e-6  # joints not flexed by the fist prototype = abduction joints
        self.chain = {}
        for f in FINGERS:
            info = fk.spec["finger"][f]
            if info:
                self.chain[f[0].upper() if f != "little" else "L"] = [fk.qidx[n] for n in info["chain_joints"]]
        self.chain = {("T" if k == "T" else k): v for k, v in self.chain.items()}
        self.flexj = {f: [j for j in idx if (f == "T" or not bool(self.abd[j]))] for f, idx in self.chain.items()}
        self.abdj = {f: [j for j in idx if bool(self.abd[j])] for f, idx in self.chain.items() if f != "T"}
        # body chain per finger (kinematic order) for phalanx points
        self.bodies = {}
        for f, idx in self.chain.items():
            nodes = [n for n in fk.nodes if n[4] is not None and n[4] in idx]
            self.bodies[f] = [n[0] for n in nodes]
        self.s = fk.scale
        with torch.no_grad():
            q0 = self.open[None].clone()
            T0 = self.fk.pads(q0)[2]
            # abduction sign: +1 if increasing q moves the tip away from the middle finger (index: +y, ring/little: -y)
            self.abd_sign = {}
            for f, js in self.abdj.items():
                for j in js:
                    q1 = q0.clone(); q1[0, j] += 0.05
                    dy = float(self.fk.pads(q1)[2][0, FI[f], 1] - T0[0, FI[f], 1])
                    want = {"I": 1.0, "M": 0.0, "R": -1.0, "L": -1.0}[f]
                    self.abd_sign[j] = (np.sign(dy) * want) if want != 0 else 0.0
            anc = self.fk.anchors  # wrist, index, middle, ring MCP (palm frame)
            self.palm = (0.4 * anc[0] + 0.6 * anc[1:].mean(0)).clone()
            self.palm[2] -= 0.03 * self.s
            self.ulnar_palm = (0.4 * anc[0] + 0.6 * anc[3]).clone(); self.ulnar_palm[2] -= 0.03 * self.s
            # thumb palmar depth range (uniform thumb joints, other joints open)
            g = torch.Generator(device="cpu").manual_seed(0)
            qq = self.open[None].repeat(4000, 1).cpu()
            tj = self.chain["T"]
            qq[:, tj] = torch.rand(4000, len(tj), generator=g) * 1.9 - 0.95
            depth = -self.fk.cpu().pads(qq)[2][:, 0, 2].numpy() / self.s
            self.fk.to(device)
            self.depth_q = (float(np.percentile(depth, 30)), float(np.percentile(depth, 60)))

    def ulnar(self):
        return "R" if self.four else "L"

    def fk_all(self, q):
        """one FK pass: poses + pad points/normals + tips (same math as PadFK.pads)."""
        fk = self.fk
        poses = fk.poses(q)
        T = torch.stack([poses[b] for b in fk.pad_links], 1)
        R, t = T[:, fk.pad_link, :3, :3], T[:, fk.pad_link, :3, 3]
        pts = torch.einsum("bfkij,fkj->bfki", R, fk.pad_pts) + t
        nrm = torch.einsum("bfkij,fkj->bfki", R, fk.pad_nrm)
        mask = fk.pad_valid.to(q)[None, :, None, None]
        zero = q.new_zeros((q.shape[0], 3))
        tips = torch.stack([poses[fk.spec["finger"][f]["tip_link"]][:, :3, 3] if fk.spec["finger"][f] else zero
                            for f in FINGERS], 1)
        return poses, pts * mask, nrm * mask, tips

    def points(self, q):
        """-> dict of named points (B,3) + pad normals (+ raw pads under key 'pads')."""
        poses, pts, nrm, tips = self.fk_all(q)
        out = {"pads": pts}
        for f, fi in FI.items():
            if f not in self.chain:
                continue
            out[(f, "pad")] = pts[:, fi].mean(1)
            out[(f, "padn")] = torch.nn.functional.normalize(nrm[:, fi].mean(1), dim=-1)
            out[(f, "tip")] = tips[:, fi]
            J = [poses[b][:, :3, 3] for b in self.bodies[f]]
            uniq = [J[0]]
            for p in J[1:]:
                if float((p - uniq[-1]).norm(dim=-1).max()) > 0.003:
                    uniq.append(p)
            seq = uniq[-3:] + [tips[:, fi]] if len(uniq) >= 3 else uniq + [tips[:, fi]]
            while len(seq) < 4:
                seq = [seq[0]] + seq
            for k, part in enumerate(("PP", "MP", "DP")):
                out[(f, part)] = 0.5 * (seq[k] + seq[k + 1])
        return out

    def point(self, pts, f, part):
        """named contact point -> (point (B,3), surface offset in s units, normal or None)."""
        if f == "U":
            f = self.ulnar()
            if part == "palm":
                return self.ulnar_palm[None].expand(pts[("T", "pad")].shape[0], 3), 0.0, None
        if part in ("pad", "tip"):
            return pts[(f, part)], 0.0, (pts[(f, "padn")] if part == "pad" else None)
        if ":" in part:
            side, seg = part.split(":")
            yv = torch.zeros(3, device=self.device); yv[1] = SIDE * self.s * (1 if side == "rside" else -1)
            return pts[(f, seg)] + yv, 0.0, None
        return pts[(f, part)], THICK, None

    def flex_frac(self, q, f, joints=None):
        js = joints if joints is not None else self.flexj[f]
        o = self.open[js]
        return ((q[:, js] - o) / (1 - o).clamp_min(0.2)).clamp(-0.5, 1.5)


def sample_theta(n, seed, dim=12):
    return torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed).draw(n).double().numpy()


def solve(hand, post, theta, iters=350, device="cpu"):
    """IK for one hand, one posture, N samples (theta rows). Returns q (N,dof), valid (N,), err (N,), info."""
    N = len(theta)
    H = hand
    s = H.s
    if post["need_l"] and H.four:  # posture needs the little finger itself: masked on 4-finger hands, not solved
        return H.open[None].repeat(N, 1), torch.zeros(N, dtype=torch.bool, device=device), torch.full((N,), float("nan")), {}
    a_lo, a_hi = post["a"] if post["a"] else (0, 0)
    a = np.exp(np.log(max(a_lo, 1e-3)) + theta[:, 0] * (np.log(max(a_hi, 2e-3)) - np.log(max(a_lo, 1e-3)))) if post["a"] else np.zeros(N)
    a_t = torch.tensor(a, dtype=torch.float32, device=device)
    sp_lo, sp_hi = SPREAD[post["spread"]]
    sig = torch.tensor(sp_lo + theta[:, 1] * (sp_hi - sp_lo), dtype=torch.float32, device=device)
    roll = torch.tensor((theta[:, 2] - 0.5) * np.deg2rad(60), dtype=torch.float32, device=device)
    fl = {f: theta[:, 3 + i] for i, f in enumerate("TIMRL")}
    count = None
    flex = dict(post["flex"])
    if post["obj"] == "count":
        count = np.array(COUNT)[(theta[:, 8] * len(COUNT)).astype(int) % len(COUNT)]
    # ---- targets ----
    q0 = H.open[None].repeat(N, 1).clone()
    tflex, wflex = {}, {}
    contact_f = {c[0] for c in post["contacts"]}
    for f in "IMRLT":
        if f not in H.chain:
            continue
        if count is not None:
            ext = count[:, "IMRLT".index(f)]
            lo = np.where(ext == 1, LEVEL["E"][0], LEVEL["F"][0]); hi = np.where(ext == 1, LEVEL["E"][1], LEVEL["F"][1])
            tgt = lo + fl[f] * (hi - lo)
            tflex[f] = ("all", torch.tensor(tgt, dtype=torch.float32, device=device))
        elif f in flex:
            v = flex[f]
            if isinstance(v, tuple):  # (proximal, distal) levels: hook / claw
                (pl, ph), (dl, dh) = lvl(v[0]), lvl(v[1])
                tflex[f] = ("split", torch.tensor(pl + fl[f] * (ph - pl), dtype=torch.float32, device=device),
                            torch.tensor(dl + fl[f] * (dh - dl), dtype=torch.float32, device=device))
            else:
                lo, hi = lvl(v)
                tflex[f] = ("all", torch.tensor(lo + fl[f] * (hi - lo), dtype=torch.float32, device=device))
        else:
            continue
        wflex[f] = 0.3 if (f in contact_f or post["obj"] in ("cyl", "sphere") and f != "T") else 1.0
        js = H.flexj[f]
        o = H.open[js]
        tg = tflex[f][1] if tflex[f][0] == "all" else 0.5 * (tflex[f][1] + tflex[f][2])
        q0[:, js] = o + tg[:, None] * (1 - o)
    abd_t = {}
    for f, js in H.abdj.items():
        for j in js:
            sgn = H.abd_sign.get(j, 0.0)
            lim = 0.95 if sgn > 0 else -0.95
            abd_t[j] = H.open[j] + sgn * sig * (lim - H.open[j]) if sgn != 0 else H.open[j].expand(N)
            q0[:, j] = abd_t[j]
    g = torch.Generator(device="cpu").manual_seed(int(theta[0, 9] * 1e6))
    tj = H.chain["T"]
    if "T" not in tflex:
        q0[:, tj] = (torch.rand(N, len(tj), generator=g) * 1.0 - 0.3).to(device)
    q = q0.clamp(-0.95, 0.95).clone().requires_grad_(True)
    r = (a_t * s / 2)
    c0 = torch.zeros(N, 3, device=device)
    c0[:, 0] = 0.55 * float(H.fk.anchors[2, 0]) + 0.3 * s * 0.5
    c0[:, 2] = -(r + 0.03 * s) if post["palm"] else -0.25 * s
    cen = c0.clone().requires_grad_(True)
    params = [q, cen]
    opt = torch.optim.Adam([{"params": [q], "lr": 0.04}, {"params": [cen], "lr": 0.004}])
    u = torch.stack([-torch.sin(roll), torch.cos(roll), torch.zeros_like(roll)], -1)  # cylinder axis ~ palm y, rolled about z
    conts = [(f, p) for f, p in post["contacts"] if not (f == "L" and H.four)]
    pinch_pairs = []
    if post["obj"] == "pinch":
        base = conts[0]
        pinch_pairs = [(base, c) for c in conts[1:]]
    if post["obj"] == "kap":
        lmf, lmp = post["kap"]
        pinch_pairs = [(("T", "pad"), (lmf, lmp))]
    dep_lo, dep_hi = H.depth_q

    def terms(q, cen):
        pts = H.points(q)
        T = {}
        errs = []
        cl = q.new_zeros(N)
        for (fa, pa), (fb, pb) in pinch_pairs:
            pa_, oa, na = H.point(pts, fa, "pad" if "|" in pa else pa)
            pb_, ob, nb = H.point(pts, fb, pb)
            tgt = a_t * s + (oa + ob) * s + (SIDE * 0 if post["obj"] == "kap" else 0)
            if post["obj"] == "kap":
                tgt = torch.full_like(a_t, 0.045 * s if (":" in pb or pb in ("PP", "MP", "DP")) else 0.01 * s)
            d = (pa_ - pb_).norm(dim=-1)
            e = (d - tgt) / s
            errs.append(e.abs())
            cl = cl + e.square()
            if na is not None and nb is not None:
                cl = cl + 0.02 * torch.relu((na * nb).sum(-1) + 0.3)
        if post["obj"] in ("cyl", "sphere"):
            for f, p in conts:
                if "|" in p:  # e.g. thumb 'pad|rside:I:PP' -> thumb pad onto the index radial side (adducted thumb)
                    _, rest = p.split("|"); side, f2, seg = rest.split(":")
                    pa_, _, _ = H.point(pts, "T", "pad"); pb_, _, _ = H.point(pts, f2, f"{side}:{seg}")
                    e = ((pa_ - pb_).norm(dim=-1) - 0.03 * s) / s
                    errs.append(e.abs()); cl = cl + e.square()
                    continue
                pp, off, nn = H.point(pts, f, p)
                v = pp - cen
                if post["obj"] == "cyl":
                    v = v - (v * u).sum(-1, keepdim=True) * u
                d = v.norm(dim=-1)
                e = (d - r - off * s) / s
                errs.append(e.abs())
                cl = cl + e.square()
                if nn is not None:
                    cl = cl + 0.02 * torch.relu(0.3 + (nn * v).sum(-1) / d.clamp_min(1e-6))
            if post["palm"]:
                v = H.palm[None] - cen
                if post["obj"] == "cyl":
                    v = v - (v * u).sum(-1, keepdim=True) * u
                e = (v.norm(dim=-1) - r) / s
                errs.append(e.abs()); cl = cl + e.square()
        T["contact"] = cl
        fx = q.new_zeros(N)
        for f, tg in tflex.items():
            js = H.flexj[f]
            if tg[0] == "all":
                fx = fx + wflex[f] * (H.flex_frac(q, f).mean(1) - tg[1]).square()
            else:
                k = max(1, len(js) // 3)
                prox = [j for j in js if j in H.chain[f][:max(1, len(H.chain[f]) - 2)]][:k] or js[:1]
                dist = [j for j in js if j not in prox] or js[-1:]
                fx = fx + wflex[f] * ((H.flex_frac(q, f, prox).mean(1) - tg[1]).square() + (H.flex_frac(q, f, dist).mean(1) - tg[2]).square())
        T["flex"] = fx
        T["spread"] = sum((q[:, j] - t).square() for j, t in abd_t.items()) if abd_t else q.new_zeros(N)
        depth = -pts[("T", "tip")][:, 2] / s
        th = q.new_zeros(N)
        if post["thumb"] == "ADD":
            th = torch.relu(depth - dep_lo).square()
        elif post["thumb"] == "ABD":
            th = torch.relu(dep_hi - depth).square()
        T["thumb"] = th
        # interpenetration between pads of fingers that are not in contact with each other (2 mm)
        pads = pts["pads"]
        pen = q.new_zeros(N)
        fl_ = [f for f in "TIMRL" if f in H.chain]
        allowed = {frozenset((c[0][0], c[1][0])) for c in pinch_pairs}
        if post["obj"] in ("cyl", "sphere"):
            allowed |= {frozenset(("T", f)) for f, _ in conts}
        for i, fa in enumerate(fl_):
            for fb in fl_[i + 1:]:
                if frozenset((fa, fb)) in allowed:
                    continue
                dm = torch.cdist(pads[:, FI[fa]], pads[:, FI[fb]], compute_mode="donot_use_mm_for_euclid_dist").amin((1, 2))
                pen = pen + torch.relu(0.004 - dm).square() / s ** 2
        T["pen"] = pen
        return T, (torch.stack(errs, 1).amax(1) if errs else q.new_zeros(N)), pts

    wts = {"contact": 100.0, "flex": 1.0, "spread": 0.3, "thumb": 5.0, "pen": 200.0}
    for it in range(iters):
        opt.zero_grad()
        T, _, _ = terms(q, cen)
        loss = sum(wts[k] * v.mean() for k, v in T.items())
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.clamp_(-0.95, 0.95)
    with torch.no_grad():
        T, err, pts = terms(q, cen)
        tol = 0.02 if post["obj"] in ("pinch", "kap") else 0.04
        valid = err <= tol
        valid &= T["pen"] * s ** 2 <= (0.002) ** 2
        flex_off = torch.zeros(N, device=device)
        for f, tg in tflex.items():
            if wflex[f] >= 1.0 and tg[0] == "all":
                flex_off = torch.maximum(flex_off, (H.flex_frac(q, f).mean(1) - tg[1]).abs())
        valid &= flex_off <= 0.2
        if post["wrap"] and H.four:
            valid &= torch.tensor(a <= WRAP4 * a_hi, device=device)
        if post["need_l"] and H.four:
            valid[:] = False
        if count is not None and H.four:
            valid &= torch.tensor(count[:, 2] == count[:, 3], device=device)  # L must equal R for the ulnar-role mapping
    return q.detach(), valid, err, {"a": a, "flex_off": flex_off.cpu().numpy()}


def coverage(Q, fks, scales, grid=0.05, ref=None):
    """C1 fingertip voxel occupancy and C2 pairwise-distance grid occupancy (vs reference sets)."""
    out = {}
    for h, q in Q.items():
        if len(q) == 0:
            out[h] = None
            continue
        with torch.no_grad():
            tips = fks[h](torch.tensor(q, dtype=torch.float32)).numpy() / scales[h]
        nf = 5 if bool(fks[h].valid.all()) else 4
        vox = set()
        for f in range(nf):
            for v in np.floor(tips[:, f] / grid).astype(int):
                vox.add((f, *v))
        T, I, M, R = tips[:, 0], tips[:, 1], tips[:, 2], tips[:, 3]
        pd = np.stack([np.linalg.norm(T - I, axis=1), np.linalg.norm(T - M, axis=1), np.linalg.norm(T - R, axis=1),
                       np.linalg.norm(I - M, axis=1), np.linalg.norm(M - R, axis=1)], 1)
        cells = {tuple(c) for c in np.floor(pd / 0.1).astype(int)}
        out[h] = {"vox": vox, "pd": pd, "cells": cells}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/codec_v3/data/tax_pairs.npz")
    ap.add_argument("--per-posture", type=int, default=256)
    ap.add_argument("--iters", type=int, default=350)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--only", default=None, help="comma list of posture ids (debug)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--memory-fraction", type=float, default=0.05)
    ap.add_argument("--part", default=None, help="k/n: solve postures k, k+n, ... and save a shard (no merge)")
    ap.add_argument("--merge", type=int, default=0, help="1: merge every <out>.part*.npz shard into --out")
    ap.add_argument("--tag", default=None, help="shard file tag (default k-of-n)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    t0 = time.time()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    posts = [p for p in POSTURES if args.only is None or p["id"] in args.only.split(",")]
    gidx = {p["id"]: i for i, p in enumerate(POSTURES)}
    if not args.merge:
        k, n = (int(x) for x in (args.part or "0/1").split("/"))
        play = {h: np.load(f"artifacts/play_v1/{h}.npz", allow_pickle=True) for h in HANDS}
        hands = {h: Hand(h, play[h], args.device) for h in HANDS}
        shard = {"meta": json.dumps({"depth_quantiles": {h: hands[h].depth_q for h in HANDS},
                                     "hand_scale_m": {h: hands[h].s for h in HANDS}})}
        for p in posts[k::n]:
            th = sample_theta(args.per_posture, seed=1000 + gidx[p["id"]])
            V, E = [], []
            for h in HANDS:
                q, v, e, info = solve(hands[h], p, th, args.iters, args.device)
                shard[f"{p['id']}|q_{h}"] = q.cpu().numpy(); V.append(v.cpu().numpy()); E.append(e.cpu().numpy())
            shard[f"{p['id']}|V"] = np.stack(V, 1); shard[f"{p['id']}|E"] = np.stack(E, 1); shard[f"{p['id']}|th"] = th
            print(p["id"], json.dumps({h: round(float(V[i].mean()), 2) for i, h in enumerate(HANDS)}), f"{time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(f"{out}.part{args.tag or f'{k}of{n}'}.npz", **shard)
        print("[cv3_taxonomy] SHARD DONE", k, n, f"{time.time() - t0:.0f}s", flush=True)
        return
    sol, meta = {}, None
    for f in sorted(out.parent.glob(out.name + ".part*.npz")):
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        for key in d.files:
            if "|" in key:
                pid, f = key.split("|")
                sol.setdefault(pid, {})[f] = d[key]
    rows_q = {h: [] for h in HANDS}
    rows_v, rows_c, rows_th, rows_err = [], [], [], []
    log = {"postures": {}, "dynamic": {}, "dynamic_skipped": DYNAMIC_SKIPPED, "heldout_classes": HELDOUT, **meta}
    posts = [p for p in posts if p["id"] in sol]
    classes = [p["id"] for p in posts] + [d[0] for d in DYNAMIC]
    for ci, p in enumerate(posts):
        S = sol[p["id"]]
        V, E, th = S["V"], S["E"], S["th"]
        keep = V.sum(1) >= 3
        for h in HANDS:
            rows_q[h].append(S[f"q_{h}"][keep])
        rows_v.append(V[keep]); rows_err.append(E[keep]); rows_th.append(th[keep]); rows_c.append(np.full(int(keep.sum()), ci))
        log["postures"][p["id"]] = {"name": p["name"], "family": p["fam"], "feix": p["feix"], "obj": p["obj"],
                                    "valid_frac": {h: float(V[:, i].mean()) for i, h in enumerate(HANDS)},
                                    "median_err_s": {h: float(np.median(E[:, i])) for i, h in enumerate(HANDS)},
                                    "kept_rows": int(keep.sum()),
                                    "infeasible_hands": [h for i, h in enumerate(HANDS) if V[:, i].mean() < 0.10]}
    rng = np.random.default_rng(5)
    for di, (did, s0, s1, nm) in enumerate(DYNAMIC):
        if s0 not in sol or s1 not in sol:
            continue
        ci = len(posts) + di
        A, B = sol[s0], sol[s1]
        n = min(len(A["V"]), len(B["V"]))
        t = rng.uniform(0, 1, n)[:, None]
        V = A["V"][:n] & B["V"][:n]
        keep = V.sum(1) >= 3
        for h in HANDS:
            rows_q[h].append(((1 - t) * A[f"q_{h}"][:n] + t * B[f"q_{h}"][:n])[keep])
        rows_v.append(V[keep]); rows_err.append(np.zeros((int(keep.sum()), 5)))
        th = np.full((int(keep.sum()), A["th"].shape[1]), np.nan); th[:, 0] = t[keep, 0]; rows_th.append(th)
        rows_c.append(np.full(int(keep.sum()), ci))
        log["dynamic"][did] = {"name": nm, "start": s0, "end": s1, "kept_rows": int(keep.sum()),
                               "valid_frac": {h: float(V[:, i].mean()) for i, h in enumerate(HANDS)}}
    Qall = {h: np.concatenate(rows_q[h]).astype(np.float32) for h in HANDS}
    Vall = np.concatenate(rows_v); Call = np.concatenate(rows_c); Th = np.concatenate(rows_th); Err = np.concatenate(rows_err)
    split = np.zeros(len(Call), np.int8)
    ho = np.isin(np.array(classes)[Call], HELDOUT)
    split[ho] = 2
    r = np.random.default_rng(3).uniform(0, 1, len(Call))
    split[(~ho) & (r < 0.10)] = 1
    np.savez_compressed(out, hands=np.array(HANDS), valid=Vall, class_id=Call, class_names=np.array(classes),
                        family=np.array([p["fam"] for p in posts] + ["dynamic"] * len(DYNAMIC)), split=split, theta=Th,
                        err_s=Err, **{f"q_{h}": Qall[h] for h in HANDS})
    log["rows"] = {"total": int(len(Call)), "train": int((split == 0).sum()), "heldout_sample": int((split == 1).sum()),
                   "heldout_class": int((split == 2).sum()),
                   "valid_rows_per_hand": {h: int(Vall[:, i].sum()) for i, h in enumerate(HANDS)},
                   "all5_valid_rows": int(Vall.all(1).sum())}
    log["infeasible_items"] = {pid: v["infeasible_hands"] for pid, v in log["postures"].items() if v["infeasible_hands"]}
    out.with_suffix(".json").write_text(json.dumps(log, indent=1))
    print("[cv3_taxonomy] DONE", json.dumps(log["rows"]), flush=True)


if __name__ == "__main__":
    main()
