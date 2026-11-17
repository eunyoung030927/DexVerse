"""pooled3hand_nohot (follow-up of design 261112, approved 261114): state/action codec for the pooled DP arms trained
WITHOUT the hand one-hot. New file (261114); p3h_policy_codec.py (arms C/Z/D of P3, with one-hot) is unchanged and is
the base class here, so C/Z/D encode/decode are bit-identical to P3 except that the 3 one-hot dims are not appended.

Arms (datasets /workspace/xhand/datasets/pooled3hand_nohot/kettle3_<arm>):
  C (C'): state = action = LeRobot vector zero-padded to 28                         (28 / 28)
  Z (Z'): [palm pos 3, palm rot6d 6, z 32]                                           (41 / 41)
  J (new): [native floating-wrist joints 6 (LeRobot vector dims 0-5), z 32]          (38 / 38)
           wrist dims = the hand's floating wrist joints (x, y, z, yaw/pitch/roll; checked against fk.spec['wrist_joints']
           in __init__ for every hand); state wrist = observed joint position, action wrist = env action (target - offset),
           exactly C's first 6 dims; z = E2* encoder mean of the finger rows of q (identical to Z's z: same q, same ops).
           decode: q_finger_hat = Dec_hand(z); wrist action passed through unchanged (no wrist solve);
                   action = [vec wrist | q_finger_hat - offset] in LeRobot action-name order.
  D (D'): [EE_common pos 3, rot6d 6, z 32]                                           (41 / 41)
Container python (/isaac-sim/python.sh), cwd /workspace/xhand/latent_codec_v1.
"""
from __future__ import annotations

import numpy as np
import torch

from .p3h_policy_codec import DATA, E2, F_LOADER, F_TASK, HANDS3, LATENT, PAD_DIM, TASK, P3HCodec  # noqa: F401

ARMS = ("C", "Z", "J", "D")
WRIST = 6
ONE_HOT = False


def state_dim(arm):
    return {"C": PAD_DIM, "Z": 9 + LATENT, "J": WRIST + LATENT, "D": 9 + LATENT}[arm]


def action_dim(arm):
    return state_dim(arm)


class P3HNoHotCodec(P3HCodec):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.j_wrist, self.j_finger = {}, {}
        for hand in HANDS3:
            fk = self.fks[hand]
            an, sn = self.names[hand]
            wr = set(fk.spec["wrist_joints"])
            if len(wr) != WRIST or set(an[:WRIST]) != wr or set(sn[:WRIST]) != wr or list(an[:WRIST]) != list(sn[:WRIST]):
                raise RuntimeError(f"{hand}: LeRobot dims 0-5 are not the {WRIST} floating wrist joints: {an[:WRIST]} vs {sorted(wr)}")
            if wr & set(fk.joint_names) or set(an[WRIST:]) != set(fk.joint_names):
                raise RuntimeError(f"{hand}: action dims 6.. != E2* finger joints")
            self.j_wrist[hand] = list(an[:WRIST])
            self.j_finger[hand] = [an.index(n) for n in fk.joint_names]  # action column of each E2* finger row

    @torch.no_grad()
    def z_from_q(self, hand, q):
        fk = self.fks[hand]
        qf = self.finger_rows(hand, q)
        return self.codecs[hand].encode(fk.normalize(torch.tensor(qf, dtype=torch.float32)).clamp(-1, 1))[0].numpy()

    def encode_action(self, hand, action, rootT, arm):
        a = np.atleast_2d(np.asarray(action, np.float32))
        if arm == "J":
            return np.concatenate([a[:, :WRIST], self.z_from_q(hand, self.q_from_action(hand, a))], -1).astype(np.float32)
        return super().encode_action(hand, a, rootT, arm)

    def encode_state(self, hand, state, rootT, arm):
        s = np.atleast_2d(np.asarray(state, np.float32))
        if arm == "J":
            return np.concatenate([s[:, :WRIST], self.z_from_q(hand, self.q_from_state(hand, s))], -1).astype(np.float32)
        out = super().encode_state(hand, s, rootT, arm)  # body + one-hot (P3) ...
        assert out.shape[1] == state_dim(arm) + len(HANDS3)
        return np.ascontiguousarray(out[:, :state_dim(arm)])  # ... one-hot dropped

    @torch.no_grad()
    def decode_action(self, hand, vec, rootT, seed_q, arm):
        if arm != "J":
            return super().decode_action(hand, vec, rootT, seed_q, arm)
        fk = self.fks[hand]
        vec = np.atleast_2d(np.asarray(vec, np.float64))
        if not np.isfinite(vec).all():
            raise ValueError("non-finite policy action")
        qf = fk.denormalize(self.codecs[hand].decoder(torch.as_tensor(vec[:, WRIST:], dtype=torch.float32))).double().numpy()
        n_act = len(self.names[hand][0])
        act = np.zeros((len(vec), n_act))
        act[:, :WRIST] = vec[:, :WRIST]
        act[:, self.j_finger[hand]] = qf - self.offset[hand][self.j_finger[hand]]
        return act, self.q_from_action(hand, act), {}
