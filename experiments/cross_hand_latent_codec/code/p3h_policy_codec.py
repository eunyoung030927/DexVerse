"""pooled3hand (design 261112 §4, P1/P3): ONE shared state/action codec for the pooled DP arms C, Z, D. New file (261113).

Used by P1 dataset building (p3h_build_dataset.py) and must be used unchanged by P3 evaluation hooks, so that training
and evaluation encode/decode with the identical function. Building blocks are the P0b round-trip functions
(p3h_roundtrip.py: codec mean encoding, f_task loader, inv) and policy_codec/decode_replay (canonical palm, wrist solve).

Conventions (verified 261113 on GraspKettle sharpa/shadow/wuji):
  LeRobot `action`[k] == pkl actions[k] (env action space = joint target - action_offset);
  LeRobot `observation.state`[k] == pkl states[k] joint_position (physical rad / m); names = info.json feature names.
  root pose is constant per hand (pkl initial_state root_pose); env provides it at evaluation.
  q_cmd = clip(action + offset, joint limits)     (decode_replay / P0b native_commands convention, offset inferred on eps 0-9)
  q_obs = observation.state                       (no offset, no clip)
Arms (one-hot over HANDS3 is appended to the STATE only):
  C: vec = LeRobot vector (state or action, LeRobot name order) zero-padded to PAD_DIM=28.  decode = slice [:n].
  Z: vec = [palm pos 3, palm rot6d 6, z 32]  palm = canonical_world(q) (world frame), z = E2* encoder mean.
  D: vec = [EE_common pos 3, rot6d 6, z 32]  EE_common = palm @ inv(f_task(q_finger)).
  decode Z/D: q_finger_hat = Dec_hand(z); desired palm = vec pose (Z) or EE_common @ f_task(q_finger_hat) (D);
              wrist joints = floating-wrist solve (decode_replay.wrist_solve); action = q[action names] - offset.
rot6d = first two columns of R (reference_kin.rot6d_from_mat / mat_from_rot6d).
Container python (/isaac-sim/python.sh), cwd /workspace/xhand/latent_codec_v1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .arm_train import load_arm
from .data import feature_names, infer_recorded_offset
from .decode_replay import canonical_world, wrist_solve
from .p3h_roundtrip import inv, load_f_task
from .reference_kin import make_T, mat_from_rot6d, rot6d_from_mat

TASK = "Dexverse-GraspKettle-v0"
HANDS3 = ("sharpa", "shadow", "wuji")
ARMS = ("C", "Z", "D")
PAD_DIM = 28
LATENT = 32
E2 = "artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt"
F_TASK = "artifacts/pooled3hand/p0a/f_task_best.pt"
F_LOADER = "xhand_latent/p3h_offsets.py"
DATA = "data/replay"  # same pkl bytes as /workspace/xhand/data/dv-demos-v0 (md5 checked 261113); LeRobot meta for names


def one_hot(hand, n=1):
    v = np.zeros((n, len(HANDS3)), np.float32)
    v[:, HANDS3.index(hand)] = 1
    return v


def state_dim(arm):
    return (PAD_DIM if arm == "C" else 9 + LATENT) + len(HANDS3)


def action_dim(arm):
    return PAD_DIM if arm == "C" else 9 + LATENT


class P3HCodec:
    def __init__(self, checkpoint=E2, f_task_ckpt=F_TASK, f_task_loader=F_LOADER, data=DATA, need_f=True):
        self.codecs, self.fks, self.payload = load_arm(checkpoint)
        self.f_task = load_f_task(f_task_ckpt, f_task_loader, "f_task") if need_f else None
        self.data = data
        self.names, self.offset = {}, {}
        for hand in HANDS3:
            fk = self.fks[hand]
            action_names, state_names = feature_names(data, TASK, hand)
            self.names[hand] = (action_names, state_names)
            # identical to p3h_roundtrip.native_commands (decode_replay.main): default from robotmeta, wrist inferred eps 0-9
            meta = fk.spec.get("robotmeta")
            default = np.array(meta["action_offset"]) if meta else np.array(
                [.3 if n == "z_translation_joint" else 0 for n in action_names])
            if meta:
                default = default[[meta["action_joint_names"].index(n) for n in action_names]]
            self.offset[hand], _ = infer_recorded_offset(data, TASK, hand, action_names, default, 10,
                                                         wrist_names=fk.spec["wrist_joints"])

    # ---------- joint-space conversions ----------
    def q_from_action(self, hand, action):
        """LeRobot/env action rows (n, n_act) -> clipped full q (n, len(reference.joint_order)), float64."""
        fk = self.fks[hand]
        names = self.names[hand][0]
        a = np.atleast_2d(np.asarray(action, np.float64))
        q = np.zeros((len(a), len(fk.reference.joint_order)))
        for i, n in enumerate(names):
            q[:, fk.reference.qidx[n]] = (a + self.offset[hand])[:, i]
        return np.clip(q, fk.reference.lower, fk.reference.upper)

    def q_from_state(self, hand, state):
        fk = self.fks[hand]
        names = self.names[hand][1]
        s = np.atleast_2d(np.asarray(state, np.float64))
        q = np.zeros((len(s), len(fk.reference.joint_order)))
        for i, n in enumerate(names):
            q[:, fk.reference.qidx[n]] = s[:, i]
        return q

    def action_from_q(self, hand, q):
        fk = self.fks[hand]
        names = self.names[hand][0]
        return np.asarray(q)[:, [fk.reference.qidx[n] for n in names]] - self.offset[hand]

    def finger_rows(self, hand, q):
        fk = self.fks[hand]
        return np.asarray(q)[:, [fk.reference.qidx[n] for n in fk.joint_names]]

    @staticmethod
    def root_T(root7):
        root7 = np.asarray(root7, np.float64).reshape(7)
        return make_T(root7[:3], root7[3:])

    # ---------- encode ----------
    @torch.no_grad()
    def encode_q(self, hand, q, rootT, arm):
        """full q rows -> (n, 41) for arm Z/D (same ops as p3h_roundtrip.codec_roundtrip + P0b D composition)."""
        fk = self.fks[hand]
        qf = self.finger_rows(hand, q)
        z = self.codecs[hand].encode(fk.normalize(torch.tensor(qf, dtype=torch.float32)).clamp(-1, 1))[0].numpy()
        P = canonical_world(fk, q, rootT)
        if arm == "Z":
            T = P
        elif arm == "D":
            T = P @ inv(np.asarray(self.f_task(hand, qf)))
        else:
            raise ValueError(arm)
        return np.concatenate([T[:, :3, 3], rot6d_from_mat(T[:, :3, :3]), z], -1).astype(np.float32)

    def encode_action(self, hand, action, rootT, arm):
        a = np.atleast_2d(np.asarray(action, np.float32))
        if arm == "C":
            return self._pad(a)
        return self.encode_q(hand, self.q_from_action(hand, a), rootT, arm)

    def encode_state(self, hand, state, rootT, arm):
        s = np.atleast_2d(np.asarray(state, np.float32))
        body = self._pad(s) if arm == "C" else self.encode_q(hand, self.q_from_state(hand, s), rootT, arm)
        return np.concatenate([body, one_hot(hand, len(s))], -1).astype(np.float32)

    @staticmethod
    def _pad(x):
        out = np.zeros((len(x), PAD_DIM), np.float32)
        out[:, :x.shape[1]] = x
        return out

    # ---------- decode ----------
    @torch.no_grad()
    def decode_action(self, hand, vec, rootT, seed_q, arm):
        """policy action rows (n, action_dim) -> (env action rows (n, n_act), full q rows, wrist_solve stats).
        seed_q: (n_joints,) or (n, n_joints) full q seed (wrist Euler-branch choice; e.g. current q_obs)."""
        fk = self.fks[hand]
        vec = np.atleast_2d(np.asarray(vec, np.float64))
        n_act = len(self.names[hand][0])
        if arm == "C":
            a = vec[:, :n_act]
            return a, self.q_from_action(hand, a), {}
        if not np.isfinite(vec).all():
            raise ValueError("non-finite policy action")
        qf = fk.denormalize(self.codecs[hand].decoder(torch.as_tensor(vec[:, 9:], dtype=torch.float32))).double().numpy()
        q = np.broadcast_to(np.asarray(seed_q, np.float64), (len(vec), len(fk.reference.joint_order))).copy()
        for i, name in enumerate(fk.joint_names):
            q[:, fk.reference.qidx[name]] = qf[:, i]
        T = np.broadcast_to(np.eye(4), (len(vec), 4, 4)).copy()
        T[:, :3, :3] = mat_from_rot6d(vec[:, 3:9])
        T[:, :3, 3] = vec[:, :3]
        desired = T if arm == "Z" else T @ np.asarray(self.f_task(hand, qf))
        q, stats = wrist_solve(fk, desired, rootT, q)
        return self.action_from_q(hand, q), q, stats
