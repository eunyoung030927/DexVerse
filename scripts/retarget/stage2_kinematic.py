# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2: kinematic retargeting, source fingertips -> target floating hand (wrist + finger IK).

Both robots live in the same task scene over the same objects, so target fingertip slot k goes to source fingertip
slot k (slot = 5 * hand + finger, thumb..little; a source finger the target lacks is dropped; a bimanual source maps
right hand to right hand and left to left). Tips in contact weigh 10, free tips 2. Each palm's orientation is kept
softly on the source palm's rotation from its home pose, applied to the target's home palm (every floating hand
homes palm-down over the table, which is what the teleop retargeter assumes too); a small pull keeps each frame
near the previous one, and frames are solved in order, each warm-started from the last. All wrists and fingers of
the robot are solved together (12 wrist + 2 x finger joints for a bimanual target). An underactuated hand's mimic
joints (Inspire: 6 per hand, ``hands.py``) are not variables: they follow their driver, ``q = C x + d``, with the
coefficients measured on the simulator first (``RetargetEnv.calibrate_mimic``), i.e. what the PhysX mimic constraint
actually holds in the task.

Kinematics come from the target hand's own USD in the simulator: the Jacobian is a finite difference over N =
joints + 1 copies of the hand written kinematically (no physics step), so nothing collides and nothing moves
between the reads.

Ported from bench2dex_retarget tools/retarget/stage2_kinematic.py (UR5 arm IK -> floating-hand virtual joints).

  python scripts/retarget/stage2_kinematic.py --task Dexverse-GraspCup-v0 --demos <demos.pkl> --episode 0 \
      --target floating_allegro_right --headless
"""

import argparse
import json
import sys
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--demos", required=True)
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--target", required=True, help="target robot_type, e.g. floating_allegro_right")
ap.add_argument("--iters0", type=int, default=300, help="iterations on the first frame")
ap.add_argument("--iters", type=int, default=8, help="iterations per later frame (warm-started)")
ap.add_argument("--eps", type=float, default=1e-3, help="finite-difference step (m / rad)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dexverse.retarget import env as renv  # noqa: E402
from dexverse.retarget import hands, paths  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul  # noqa: E402

W_TIP_CONTACT = 10.0
W_TIP_FREE = 2.0
W_PALM_ROT = 0.3
W_REG = 0.05
DAMPING = 1e-3


def main():
    rdir = paths.run_dir(args.task, args.demos, args.episode)
    out_dir = rdir / args.target
    out = out_dir / "kinematic.npz"
    if out.exists():
        print(f"skip: {out} exists", flush=True)
        return
    ref = np.load(rdir / "reference.npz", allow_pickle=True)
    meta = json.loads(str(ref["meta"]))
    T = meta["n_frames"]
    H = meta.get("n_hands", 1)
    if H != hands.get(args.target).n_hands:
        raise RuntimeError(f"source has {H} hand(s), target {args.target} {hands.get(args.target).n_hands}")
    env, _ = renv.build(args.task, args.target, 64 if H == 1 else 72, args.device, gravity=False)   # >= free + 1
    dev, names = env.device, env.joint_names
    free = env.free_ids                                               # IK variables: every non-mimic joint
    J = len(free)
    assert env.num_envs >= J + 1, (env.num_envs, J)
    fd_ids = torch.arange(J + 1, device=dev)
    x_lo, x_hi = env.q_lo[free], env.q_hi[free]
    coupling = env.calibrate_mimic()                                   # measured on the sim; {} for actuated hands
    if coupling:
        print("mimic coupling (simulator fit):", json.dumps(coupling), flush=True)
    slots = [s for s in env.slots if s in meta["source_slots"]]
    sel = [env.slots.index(s) for s in slots]                       # rows of env.tips() used
    print(f"target {args.target}: {len(names)} joints ({J} free, {len(env.mimic_ids)} mimic), fingers {slots}; limits "
          + ", ".join(f"{n} [{env.q_lo[i]:.2f},{env.q_hi[i]:.2f}]" for i, n in enumerate(names[:6])), flush=True)

    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
    tgt_tips = t(ref["src_tips"])[:, slots]                          # (T,F,3)
    w_tip = torch.where(torch.as_tensor(ref["contact_body"], device=dev)[:, slots] >= 0, W_TIP_CONTACT, W_TIP_FREE)
    src_palm, src_home = t(ref["src_palm"]).reshape(T, H, 7), t(ref["src_palm_home"]).reshape(H, 7)

    home = env.robot.data.default_joint_pos[0].clone()
    env.teleport(q=home)
    home = home[free]
    tgt_home_q = env.palm_pose()[0, :, 3:7].clone()                                   # (H,4)
    # palm rotation target per hand: the source palm's rotation from its home, applied to the target's home palm
    d_rot = quat_mul(src_palm[:, :, 3:7].reshape(-1, 4), quat_conjugate(src_home[:, 3:7]).repeat(T, 1))
    tgt_palm_q = quat_mul(d_rot, tgt_home_q.repeat(T, 1)).reshape(T, H, 4)

    eye = torch.eye(J, device=dev)
    probe_dx = torch.cat([args.eps * eye, torch.zeros(1, J, device=dev)], 0)   # env J = the point itself

    def features(x):
        """free joints x (J+1,J) -> tips (J+1,F*3), palm quats (J+1,H,4)."""
        env.teleport(q=env.expand(x), env_ids=fd_ids)
        return env.tips()[: J + 1, sel].reshape(J + 1, -1), env.palm_pose()[: J + 1, :, 3:7]

    def rot_err(qa, qb):
        """axis-angle of qa * conj(qb) per hand, flattened: (..., H, 4) x (..., H, 4) -> (..., H*3)."""
        aa = axis_angle_from_quat(quat_mul(qa.reshape(-1, 4), quat_conjugate(qb.reshape(-1, 4))))
        return aa.reshape(*qa.shape[:-2], H * 3)

    def solve(f, x, x_reg, iters):
        for _ in range(iters):
            q = torch.max(torch.min(x[None] + probe_dx, x_hi), x_lo)
            tips, quat = features(q)
            base_tips, base_quat = tips[J], quat[J]
            dq = q[:J] - q[J]
            dd = torch.diagonal(dq).clamp(min=1e-9) * torch.sign(torch.diagonal(dq) + 1e-12)
            Jt = ((tips[:J] - base_tips) / dd[:, None]).T                                  # (F*3, J)
            Jr = (rot_err(quat[:J], base_quat[None].expand(J, -1, -1)) / dd[:, None]).T     # (H*3, J)
            w = w_tip[f].repeat_interleave(3)
            e_tip = (tgt_tips[f].reshape(-1) - base_tips) * w
            e_rot = W_PALM_ROT * rot_err(tgt_palm_q[f], base_quat)
            e = torch.cat([e_tip, e_rot, W_REG * (x_reg - x)])
            Jx = torch.cat([Jt * w[:, None], W_PALM_ROT * Jr, W_REG * eye])
            H = Jx.T @ Jx + DAMPING * eye
            x = torch.max(torch.min(x + torch.linalg.solve(H, Jx.T @ e), x_hi), x_lo)
        return x

    xs = [solve(0, home, home, args.iters0)]
    for f in range(1, T):
        xs.append(solve(f, xs[-1], xs[-1], args.iters))
        if f % 50 == 0:
            print(f"  frame {f}/{T}", flush=True)
    q = env.expand(torch.stack(xs))                                   # (T, n_joints), mimic joints filled in

    tips_all, palm_all = [], []
    for s in range(0, T, env.num_envs):
        e = min(s + env.num_envs, T)
        env.teleport(q=q[s:e], env_ids=torch.arange(e - s, device=dev))
        tips_all.append(env.tips()[: e - s])
        palm_all.append(env.palm_pose()[: e - s])
    tips_h, palm = torch.cat(tips_all), torch.cat(palm_all)
    tips5 = torch.full((T, 5 * H, 3), float("nan"), device=dev)
    tips5[:, env.slots] = tips_h
    err = (tips_h[:, sel] - tgt_tips).norm(dim=-1)
    rot_e = rot_err(tgt_palm_q, palm[:, :, 3:7]).reshape(T, H, 3).norm(dim=-1)           # (T,H)
    contact = w_tip == W_TIP_CONTACT
    rep = {"tip_err_cm_mean": round(100 * err.mean().item(), 2),
           "tip_err_cm_contact": round(100 * err[contact].mean().item(), 2) if contact.any() else None,
           "tip_err_cm_p95": round(100 * err.quantile(0.95).item(), 2),
           "tip_err_cm_per_finger": [round(100 * v, 2) for v in err.mean(0).tolist()],
           "palm_rot_err_deg_mean": [round(float(np.degrees(v)), 1) for v in rot_e.mean(0).tolist()],
           "q_jump_max_rad": round((q[1:] - q[:-1]).abs().max().item(), 3),
           "at_limit_frac": round(((q[:, free] <= x_lo + 1e-4) | (q[:, free] >= x_hi - 1e-4)).float().mean().item(), 3),
           "fingers": slots}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, q=q.cpu().numpy(), joint_names=np.array(names), tips=tips5.cpu().numpy(),
                        palm=palm.cpu().numpy(), tip_err=err.cpu().numpy(), report=np.array(json.dumps(rep)),
                        coupling=np.array(json.dumps(coupling)))
    print(f"RESULT stage2 {args.task} ep{args.episode} -> {args.target}: {rep}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
