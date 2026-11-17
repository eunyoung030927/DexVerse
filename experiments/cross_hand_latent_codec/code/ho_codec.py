"""ho (H1/H2 cross-hand follow-up of pooled3hand_nohot, 261115): no-one-hot Z/D codec for an ARBITRARY hand list.
New file. p3h_policy_codec.py / p3h_nh_codec.py / p3h_offsets.py / p3h_roundtrip.py are NOT modified; this module
subclasses P3HNoHotCodec and only replaces the three HANDS3-bound pieces:
  1. __init__: names/offsets/J-checks loop over `hands` (default HANDS_ALL), same ops as P3HCodec.__init__ (robotmeta
     default offset, floating-wrist offset inferred on pkl eps 0-9 of data/replay) and P3HNoHotCodec.__init__.
  2. f_task loader: p3h_offsets.Offsets(kind, ckpt, hands=hands) (Offsets accepts hands=; load_f_task does not pass it).
     Same callable signature/return as p3h_roundtrip.load_f_task.
  3. encode_state (Z/D/C): body only (P3HCodec.encode_state minus the one-hot; P3HNoHotCodec drops it AFTER calling
     one_hot(hand) which raises for hands outside HANDS3). J: unchanged (P3HNoHotCodec.encode_state).
encode_action / decode_action / q_* conversions are inherited unchanged -> for sharpa/shadow/wuji the rows are
bit-identical to P3HNoHotCodec (checked by ho_check_codec.py).
Container python, cwd /workspace/xhand/latent_codec_v1; P3H_PD2 must point to artifacts/pooled3hand/p0b/ext/pd2_work2.
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import torch

from .arm_train import load_arm
from .data import feature_names, infer_recorded_offset
from .p3h_nh_codec import ARMS, WRIST, P3HNoHotCodec, action_dim, state_dim  # noqa: F401
from .p3h_policy_codec import DATA, E2, F_LOADER, F_TASK, LATENT, PAD_DIM, TASK  # noqa: F401

HANDS_ALL = ("sharpa", "shadow", "wuji", "allegro", "leap")
F_TASK_SA = "artifacts/pooled3hand/ho/h2/f_task/f_task_best.pt"  # H2: f_task retrained on shadow+allegro only


def load_f_task_hands(ckpt, loader, hands, kind="f_task"):
    """= p3h_roundtrip.load_f_task, but Offsets(kind, ckpt, hands=hands)."""
    spec = importlib.util.spec_from_file_location("p3h_offsets_ext", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p3h_offsets_ext"] = mod
    spec.loader.exec_module(mod)
    off = mod.Offsets(kind, ckpt, hands=tuple(hands))
    return lambda hand, q: off.offset(hand, np.asarray(q, np.float32)).double().cpu().numpy()


class HoCodec(P3HNoHotCodec):
    def __init__(self, checkpoint=E2, f_task_ckpt=F_TASK, f_task_loader=F_LOADER, data=DATA, need_f=True,
                 hands=HANDS_ALL):
        # --- P3HCodec.__init__ with the hand list as a parameter (no super().__init__: it loops over HANDS3) ---
        self.hands = tuple(hands)
        self.codecs, self.fks, self.payload = load_arm(checkpoint)
        missing = [h for h in self.hands if h not in self.fks]
        if missing:
            raise RuntimeError(f"E2* checkpoint has no codec for {missing}")
        self.f_task = load_f_task_hands(f_task_ckpt, f_task_loader, self.hands) if need_f else None
        self.data = data
        self.names, self.offset = {}, {}
        for hand in self.hands:
            fk = self.fks[hand]
            action_names, state_names = feature_names(data, TASK, hand)
            self.names[hand] = (action_names, state_names)
            meta = fk.spec.get("robotmeta")
            default = np.array(meta["action_offset"]) if meta else np.array(
                [.3 if n == "z_translation_joint" else 0 for n in action_names])
            if meta:
                default = default[[meta["action_joint_names"].index(n) for n in action_names]]
            self.offset[hand], _ = infer_recorded_offset(data, TASK, hand, action_names, default, 10,
                                                         wrist_names=fk.spec["wrist_joints"])
        # --- P3HNoHotCodec.__init__ checks/J tables with the hand list ---
        self.j_wrist, self.j_finger = {}, {}
        for hand in self.hands:
            fk = self.fks[hand]
            an, sn = self.names[hand]
            wr = set(fk.spec["wrist_joints"])
            if len(wr) != WRIST or set(an[:WRIST]) != wr or set(sn[:WRIST]) != wr or list(an[:WRIST]) != list(sn[:WRIST]):
                raise RuntimeError(f"{hand}: LeRobot dims 0-5 are not the {WRIST} floating wrist joints")
            if wr & set(fk.joint_names) or set(an[WRIST:]) != set(fk.joint_names):
                raise RuntimeError(f"{hand}: action dims 6.. != E2* finger joints")
            self.j_wrist[hand] = list(an[:WRIST])
            self.j_finger[hand] = [an.index(n) for n in fk.joint_names]

    def encode_state(self, hand, state, rootT, arm):
        s = np.atleast_2d(np.asarray(state, np.float32))
        if arm == "J":
            return super().encode_state(hand, s, rootT, arm)
        body = self._pad(s) if arm == "C" else self.encode_q(hand, self.q_from_state(hand, s), rootT, arm)
        assert body.shape[1] == state_dim(arm)
        return np.ascontiguousarray(body, np.float32)
