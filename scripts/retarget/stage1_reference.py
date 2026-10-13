# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 1: the retargeting reference of one recorded episode, read off the simulator.

The episode's recorded scene states (``states[t]``, the state before action t, as record_demos keeps them) are
written into the source hand's task env, a chunk of frames at a time (one frame per env, no physics step), and
everything the later stages need is read from the same USDs the benchmark simulates:

  body_pose      (T,B,7)  every task-object body (rigid objects; each link of a non-robot articulation)
  obj_root       (T,O,7)  task-object root poses (as recorded)
  src_tips       (T,K,3)  source fingertips, slot 5*hand+finger (thumb..little; NaN where the hand has no such
                          finger); K = 5 for a single hand, 10 for a bimanual robot (right, then left)
  src_palm       (T,H,7)  source palm pose per hand;  src_palm_home (H,7) at the robot's home joints
  contact_body   (T,K)    body each tip touches (-1: none): the tip link's COLLISION surface within 2 cm (not its
                          origin -- Shadow's thumb tip origin sits ~3.5 cm behind its pad, so origin distance missed
                          every thumb contact)
  contact_local  (T,K,3)  closest surface point, in that body's frame
  active         (T,)     body the hand manipulates: moving, and within 3 cm of a tip

Ported from bench2dex_retarget tools/retarget/stage1_reference.py (floating hands, single or bimanual, DexVerse states).

  python scripts/retarget/stage1_reference.py --task Dexverse-GraspCup-v0 \
      --demos <demos.pkl> --episode 0 --headless
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--demos", required=True, help="trajectory pickle (record_demos format, with per-step states)")
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--chunk", type=int, default=128, help="frames written at once (one per env)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dexverse.retarget import env as renv  # noqa: E402
from dexverse.retarget import geometry, hands, paths  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_conjugate  # noqa: E402

NEAR_DIST = 0.03     # the hand manipulates a moving body whose surface is this close to one of its tips
MOVING_SPEED = 0.03  # m/s, a body point moving faster than this counts as moving
DILATE = 15          # frames (60 Hz): grasp onset / release belong to the manipulation


def stack_states(states):
    """List of batch-1 state dicts -> one batched state dict."""
    first = states[0]
    if isinstance(first, dict):
        return {k: stack_states([s[k] for s in states]) for k in first}
    return np.concatenate([np.asarray(s, dtype=np.float32).reshape(1, -1) for s in states], 0)


def main():
    demos = paths.load_demos(args.demos)
    ep = demos["episodes"][args.episode]
    src_type = demos["robot_type"]
    states = ep["states"]
    T = len(states)
    if T != len(ep["actions"]) + 1:
        raise RuntimeError(f"episode {args.episode}: {T} states for {len(ep['actions'])} actions (need T+1)")
    out_dir = paths.run_dir(args.task, args.demos, args.episode)
    out = out_dir / "reference.npz"
    if out.exists() and paths.output_version(out) >= paths.REFERENCE_VERSION:
        print(f"skip: {out} exists (version {paths.REFERENCE_VERSION})", flush=True)
        return
    env, _ = renv.build(args.task, src_type, min(args.chunk, T), args.device, gravity=False)
    dev, N = env.device, env.num_envs
    spec = env.spec
    K = 5 * spec.n_hands                                              # fingertip slots over the robot's hands

    import omni.usd  # noqa: PLC0415

    meshes = geometry.body_meshes(omni.usd.get_context().get_stage(), env.body_prim_paths())
    for (k, b, _), m in zip(env.bodies, meshes):
        print(f"  body {k}/{b}: {'no mesh' if m is None else f'{len(m[0])} verts {len(m[1])} faces'}", flush=True)

    # collision surface of every tip (link frame): contact is judged on it, not on the tip origin. A tip body that
    # is a bare frame (Shadow's thtip / fftip: no collision geometry) uses the nearest body that has some -- its
    # distal link.
    link_paths = list(env.robot.root_physx_view.link_paths[0])
    all_samples = geometry.collision_samples(omni.usd.get_context().get_stage(), link_paths, 48)
    env.teleport(q=env.robot.data.default_joint_pos[0], env_ids=torch.arange(1, device=dev))
    bpos0 = env.robot.data.body_pos_w[0]
    with_coll = [i for i, p in enumerate(all_samples) if p is not None]
    cbody = []
    for i in env.tip_ids.tolist():
        if all_samples[i] is not None:
            cbody.append(i)
        else:
            dd = torch.stack([(bpos0[c] - bpos0[i]).norm() for c in with_coll])
            cbody.append(with_coll[int(dd.argmin())])
    cbody_t = torch.tensor(cbody, device=dev)
    tip_samples = [torch.as_tensor(all_samples[c], device=dev) for c in cbody]
    print("contact bodies per tip slot:", {spec.slot_name(s): env.robot.body_names[c] for s, c in zip(env.slots, cbody)},
          flush=True)

    tips, cpos, cquat, palm, body, roots = [], [], [], [], [], []
    for s in range(0, T, N):
        e = min(s + N, T)
        ids = torch.arange(e - s, device=dev)
        env.set_state(stack_states(states[s:e]), env_ids=ids)
        tips.append(env.tips()[: e - s])
        cpos.append(env.robot.data.body_pos_w[: e - s, cbody_t] - env.origins()[: e - s, None])
        cquat.append(env.robot.data.body_quat_w[: e - s, cbody_t])
        palm.append(env.palm_pose()[: e - s])
        body.append(env.object_body_poses()[: e - s])
        roots.append(env.object_roots()[: e - s])
    tips_h, cpos, cquat, palm, body, roots = (torch.cat(x) for x in (tips, cpos, cquat, palm, body, roots))
    # palm at the hand's home joints: stage 2 maps palm rotations through each hand's home pose
    env.teleport(q=env.robot.data.default_joint_pos[0], env_ids=torch.arange(1, device=dev))
    palm_home = env.palm_pose()[0]                                    # (H,7)
    rec = np.stack([np.asarray(s["rigid_object"][env.obj_ids[0]]["root_pose"]).reshape(-1)
                    for s in states]) if env.obj_ids and env.obj_ids[0] not in env.art_ids else None
    if rec is not None:
        print(f"written vs recorded root of {env.obj_ids[0]}: "
              f"{100 * np.abs(roots[:, 0, :3].cpu().numpy() - rec[:, :3]).max():.4f} cm max", flush=True)
    tips5 = torch.full((T, K, 3), float("nan"), device=dev)
    tips5[:, env.slots] = tips_h

    # contacts: signed distance of every tip link's collision surface to every body surface
    B = len(env.bodies)
    dist = torch.full((T, K, B), float("inf"), device=dev)
    local = torch.zeros(T, K, B, 3, device=dev)
    have = torch.isfinite(tips5[..., 0])
    for bi, m in enumerate(meshes):
        if m is None:
            continue
        V = torch.as_tensor(m[0], dtype=torch.float32, device=dev)
        Fc = torch.as_tensor(m[1], device=dev)
        p, qb = body[:, bi, :3], body[:, bi, 3:7]
        for j, slot in enumerate(env.slots):
            S = len(tip_samples[j])
            pw = cpos[:, j, None] + quat_apply(cquat[:, j, None].expand(-1, S, -1).reshape(-1, 4),
                                               tip_samples[j].repeat(T, 1)).reshape(T, S, 3)
            loc = quat_apply(quat_conjugate(qb).repeat_interleave(S, 0), (pw - p[:, None]).reshape(-1, 3))
            d, cpt = geometry.signed_distance(loc, V, Fc)
            d, cpt = d.reshape(T, S), cpt.reshape(T, S, 3)
            d = torch.where(d < -geometry.MAX_DEPTH, torch.full_like(d, float("inf")), d)
            dmin, k = d.min(1)
            dist[:, slot, bi] = dmin
            local[:, slot, bi] = cpt[torch.arange(T, device=dev), k]
    dist = geometry.effective_distance(dist)
    dist[~have] = float("inf")
    dmin, nearest = dist.min(-1)
    contact_body = torch.where(dmin < geometry.CONTACT_DIST, nearest, torch.full_like(nearest, -1))
    contact_local = torch.take_along_dim(local, nearest[..., None, None].expand(-1, -1, 1, 3), 2)[:, :, 0]

    # active body: moving (speed bound: origin speed + angular speed x body radius) and near a tip
    fps = 1.0 / env.step_dt
    radius = torch.tensor([0.0 if m is None else float(np.linalg.norm(m[0], axis=1).max()) for m in meshes],
                          device=dev)
    lin = torch.zeros(T, B, device=dev)
    lin[1:] = (body[1:, :, :3] - body[:-1, :, :3]).norm(dim=-1) * fps
    rot = torch.zeros(T, B, device=dev)
    dq = (body[1:, :, 3:7] * body[:-1, :, 3:7]).sum(-1).abs().clamp(max=1.0)
    rot[1:] = 2.0 * torch.acos(dq) * fps
    moving = (lin + rot * radius) > MOVING_SPEED
    pad = torch.nn.functional.pad(moving.float().T[None], (DILATE, DILATE))[0]
    moving = torch.nn.functional.max_pool1d(pad[None], 2 * DILATE + 1, stride=1)[0].T > 0
    hd = dist.min(1).values                                           # (T,B)
    cand = torch.where(moving & (hd < NEAR_DIST), hd, torch.full_like(hd, float("inf")))
    v, a = cand.min(1)
    act = torch.where(torch.isfinite(v), a, torch.full_like(a, -1)).cpu().numpy()
    for t in range(1, T - 1):                                         # fill dropouts of <= DILATE frames
        if act[t] == -1:
            prev, nxt = act[max(0, t - DILATE):t], act[t + 1:t + 1 + DILATE]
            if len(prev) and prev[-1] >= 0 and prev[-1] in nxt:
                act[t] = prev[-1]

    names = [f"{k}/{b}" if k in env.art_ids else k for k, b, _ in env.bodies]
    segs, start = [], None
    for t, x in enumerate(list(act) + [-1]):
        if start is not None and x != act[start]:
            if t - start >= 30:
                segs.append({"body": names[act[start]], "start": start, "end": t})
            start = None
        if start is None and x >= 0:
            start = t
    meta = {"version": paths.REFERENCE_VERSION,
            "task": args.task, "demos": os.path.abspath(args.demos), "episode": args.episode, "source": src_type,
            "n_frames": T, "fps": fps, "obj_ids": env.obj_ids, "bodies": names,
            "body_obj": [env.obj_ids.index(k) for k, _, _ in env.bodies],
            "art": {k: list(env.objects[k].joint_names) for k in env.art_ids},
            "segments": segs, "fingers": list(hands.FINGERS), "n_hands": spec.n_hands,
            "hands": [h.side for h in spec.hands], "source_slots": env.slots}
    arrays = {"body_pose": body, "obj_root": roots, "src_tips": tips5, "src_palm": palm, "src_palm_home": palm_home,
              "contact_body": contact_body, "contact_local": contact_local, "contact_dist": dmin}
    arrays = {k: v.cpu().numpy() for k, v in arrays.items()}
    arrays["active"] = act
    for k in env.art_ids:
        arrays[f"art/{k}"] = np.stack([np.asarray(s["articulation"][k]["joint_position"]).reshape(-1) for s in states])
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
    (out_dir / "reference.json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(out_dir / "meshes.npz", **{f"{i}/v": m[0] for i, m in enumerate(meshes) if m is not None},
                        **{f"{i}/f": m[1] for i, m in enumerate(meshes) if m is not None})
    ct = (arrays["contact_body"] >= 0).mean(0)
    print(f"contact fraction per finger {dict(zip([spec.slot_name(i) for i in range(K)], np.round(ct, 2).tolist()))}")
    print(json.dumps(segs, indent=1))
    print(f"RESULT stage1 {args.task} ep{args.episode}: {T} frames, {B} bodies -> {out}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
