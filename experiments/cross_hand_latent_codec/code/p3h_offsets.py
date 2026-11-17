"""P0a (pooled3hand_design_261112 §4.1): standalone loader / apply for the hand-offset modules. New file.

Convention (same as G2/G3): off(q, a) is a 4x4 transform in the canonical palm ("wrist") frame P of latent_codec_v1
(origin wrist keypoint, x distal, y thumbward, z dorsal; metres).
    EE_common = EE_a @ inv(off(q_a, a))        EE_a = EE_common @ off(q_a, a)
EE_a = canonical palm pose (decode_replay.canonical_world), any outer frame (world or kettle frame).

kinds:
  "raw"    off = I
  "K"      per-hand constant (artifacts/pooled3hand/p0a/K.pt)
  "f_task" G2 architecture pd2.train_f.F on pd2.common.Hand.features(q) (p0a/f_task_best.pt)
  "g2f"    G2 f_id (play_data2/f_pilot_v0/f_id.pt), unchanged
  "F"      training-free pinch frame: off = inv(T_P->F(q) @ M), T_P->F = pd2 Hand.e0_frame (arm_pads.f_frame on
           softmin pad contacts, fingers index/middle/ring, pair weights; z = palm dorsal orthogonalised); M = shared
           constant mean-zero normaliser (p0a/F_M.pt; identity -> pure T_world->F).
q: physical rad, finger joints only, in PadFK(hand).joint_names order (= Hand.jn); see q_from_named().

Dependencies (copy to work1 together): latent_codec_v1/xhand_latent/{arm_pads,kinematics,reference_kin}.py +
latent_codec_v1/metadata/{hands,raw_kin}/<hand>.json, play_data2/code/pd2/{common,train_f,__init__}.py, DexVerse
tactile_pads/floating_<hand>_right.npz (PAD_DIR in pd2.common / arm_pads), torch, numpy, scipy. Paths are set by
P3H_LC / P3H_PD2 env vars (defaults = work2 container paths).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

LC = os.environ.get("P3H_LC", "/workspace/xhand/latent_codec_v1")
PD2 = os.environ.get("P3H_PD2", "/workspace/xhand/play_data2/code")
for p in (LC, PD2):
    if p not in sys.path:
        sys.path.insert(0, p)
from pd2.common import Hand  # noqa: E402
from pd2.train_f import F as FNet, inv, make_T, rot6d_to_R  # noqa: E402

HANDS3 = ("sharpa", "shadow", "wuji")
P0A = os.environ.get("P3H_P0A", f"{LC}/artifacts/pooled3hand/p0a")
G2F = os.environ.get("P3H_G2F", "/workspace/xhand/play_data2/f_pilot_v0/f_id.pt")


class KNet(nn.Module):
    """Per-hand constant 9D offset (pos + rot6d delta), zero-initialised (= identity)."""

    def __init__(self, n=3):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(n, 9))
        self.register_buffer("r6", torch.tensor([1., 0, 0, 0, 1, 0]))

    def forward(self, i, B):
        o = self.p[i]
        T = make_T(rot6d_to_R(o[3:] + self.r6)[None], o[None, :3] * 0.1)
        return T.expand(B, 4, 4)

    def matrices(self):
        return torch.stack([self.forward(i, 1)[0] for i in range(self.p.shape[0])])


class Offsets:
    def __init__(self, kind, path=None, device="cpu", hands=HANDS3):
        self.kind, self.dev = kind, device
        self.hands = {h: Hand(h, device) for h in hands}
        self.order = list(hands)
        if kind == "f_task" or kind == "g2f":
            path = path or (f"{P0A}/f_task_best.pt" if kind == "f_task" else G2F)
            sd = torch.load(path, map_location=device)
            sd = sd.get("state_dict", sd)
            self.net = FNet().to(device)
            self.net.load_state_dict(sd)
            self.net.eval()
        elif kind == "K":
            d = torch.load(path or f"{P0A}/K.pt", map_location=device)
            self.K = {h: d["K"][d["hands"].index(h)].to(device).float() for h in hands}
        elif kind == "F":
            if path == "identity":
                self.M = torch.eye(4, device=device)
            else:
                self.M = torch.load(path or f"{P0A}/F_M.pt", map_location=device)["M"].to(device).float()
        elif kind != "raw":
            raise ValueError(kind)

    def q_from_named(self, hand, names, values):
        """values (B, len(names)) in any joint order (e.g. a pkl joint_position row) -> (B, dof) in Hand.jn order."""
        names = list(names)
        return np.asarray(values)[..., [names.index(n) for n in self.hands[hand].jn]]

    @torch.no_grad()
    def offset(self, hand, q):
        """q (B,dof) physical rad (numpy or torch) -> off (B,4,4) torch float32."""
        H = self.hands[hand]
        q = torch.as_tensor(np.asarray(q) if not torch.is_tensor(q) else q, dtype=torch.float32, device=self.dev)
        B = q.shape[0]
        if self.kind == "raw":
            return torch.eye(4, device=self.dev).expand(B, 4, 4).clone()
        if self.kind == "K":
            return self.K[hand][None].expand(B, 4, 4).clone()
        if self.kind == "F":
            return inv(H.e0_frame(q) @ self.M)
        return self.net(*H.features(q))

    def ee_common(self, hand, q, ee):
        """ee (B,4,4) canonical palm pose (numpy/torch) -> EE_common (B,4,4) float64 numpy."""
        off = self.offset(hand, q).double().cpu().numpy()
        return np.asarray(ee, np.float64) @ np.linalg.inv(off)

    def ee_from_common(self, hand, q, ee_common):
        off = self.offset(hand, q).double().cpu().numpy()
        return np.asarray(ee_common, np.float64) @ off
