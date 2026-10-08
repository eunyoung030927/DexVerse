# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 3: SPIDER physics retargeting (Meta FAIR, arXiv 2511.09484) of one episode onto one target hand.

Ported from bench2dex_retarget tools/retarget/stage3_spider.py with its "v2" settings; frame counts are x3 for
DexVerse's 60 Hz control (same time spans). No learning: a receding-horizon sampling optimiser (MPPI) refines
the kinematic reference so the target hand reproduces the demo object motion under the task's own physics.

  * Every replanning step the current state is broadcast to all N envs; N residual sequences (knots every
    ``knot`` frames over ``horizon`` frames, linearly interpolated) are rolled out and scored; the nominal plan
    is the cost-weighted mean (MPPI), ``iters`` times.
  * Virtual contact guidance pulls demo contact points towards the fingertips during optimisation, annealed to
    0; committed frames always run without it.
  * Only frames near manipulation are optimised (gate), and only when manipulation is within ``gate_look``.
  * Backtracking: when a manipulated object jumps away from its reference, go back ``backtrack`` commits and
    re-plan with more effort, up to ``max_retries`` times per restart frame, then carry on (never stops early).
  * After the demo the last command is held for ``settle`` frames; success keeps being checked.
  * The committed execution (env 0) is kept as a DexVerse trajectory: the scene state before every action and
    the env action that commands the executed joint targets. Every commit starts from a restored snapshot, so
    re-executing the actions open-loop is not guaranteed to reproduce the run: the states are the record.
  * Success = the task's success term held ``success_steps`` consecutive frames (record_demos' rule).

  python scripts/retarget/stage3_spider.py --task Dexverse-GraspCup-v0 --demos <demos.pkl> --episode 0 \
      --target floating_allegro_right --headless
"""

import argparse
import gzip
import json
import os
import pickle
import sys
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--demos", required=True)
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--num_samples", type=int, default=1024)
ap.add_argument("--horizon", type=int, default=90)
ap.add_argument("--knot", type=int, default=15)
ap.add_argument("--commit", type=int, default=15)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--sigma", type=float, default=0.5)
ap.add_argument("--guide_gain", type=float, default=300.0, help="N/m, contact spring at the first iteration")
ap.add_argument("--squeeze", type=float, default=0.02, help="m; contact targets moved this far into the body")
ap.add_argument("--max_retries", type=int, default=3, help="backtracks per restart frame")
ap.add_argument("--backtrack", type=int, default=2, help="commits to go back after a drop")
ap.add_argument("--gate_look", type=int, default=90, help="optimise only when manipulation is this many frames ahead")
ap.add_argument("--drop_jump", type=float, default=0.025, help="m; error increase per commit counted as a drop")
ap.add_argument("--settle", type=int, default=120)
ap.add_argument("--success_steps", type=int, default=10, help="consecutive success frames (record_demos default)")
ap.add_argument("--no_hold_pass", action="store_true")
ap.add_argument("--max_frames", type=int, default=0, help="debug: stop the episode early")
ap.add_argument("--hold_only", action="store_true", help="debug: run the hold pass and exit")
ap.add_argument("--tag", default="spider")
ap.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dexverse.retarget import env as renv_mod  # noqa: E402
from dexverse.retarget import paths  # noqa: E402
from dexverse.retarget.env import to_numpy_state  # noqa: E402
from dexverse.retarget.spider import SpiderCfg, SpiderEnv  # noqa: E402

ART_W = 10.0


def step_cost(env: SpiderEnv):
    f = env.f()
    roots = env.e.object_roots()
    ref = env.ref_root[f]
    c = (10.0 * (ref[..., :3] - roots[..., :3]).norm(dim=-1)
         + 0.5 * env.obj_rot_dist(roots[..., 3:7], ref[..., 3:7])).sum(-1)
    for k, v in env.e.object_qpos().items():
        c = c + ART_W * (env.ref_art[k][f] - v).abs().sum(-1)
    body = env.e.object_body_poses()
    ctgt, valid, idx = env.demo_contact_world(body, f)
    if args.squeeze > 0:
        centre = torch.gather(body[..., :3], 1, idx[..., None].expand(-1, -1, 3))
        inward = centre - ctgt
        ctgt = ctgt + args.squeeze * inward / inward.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    tips = env.tips()
    cd = (ctgt - tips).norm(dim=-1)
    v = valid.float()
    c = c + 5.0 * (cd * v).sum(-1) / v.sum(-1).clamp(min=1)
    c = c + 1.0 * (env.ref_tips[f] - tips).norm(dim=-1).mean(-1)
    return torch.nan_to_num(c, nan=1e3, posinf=1e3)


def obj_err(env: SpiderEnv):
    f = env.f()
    roots = env.e.object_roots()
    e = (env.ref_root[f][0, :, :3] - roots[0, :, :3]).norm(dim=-1)
    for k, v in env.e.object_qpos().items():
        j = env.e.obj_ids.index(k)
        e[j] = e[j] + (env.ref_art[k][f][0] - v[0]).abs().sum()
    return e


def knots_to_actions(knots, H, knot):
    N, K, A = knots.shape
    t = torch.arange(H, device=knots.device, dtype=torch.float32) / knot
    i0 = t.floor().long().clamp(max=K - 1)
    i1 = (i0 + 1).clamp(max=K - 1)
    w = (t - i0.float())[None, :, None]
    return knots[:, i0] * (1 - w) + knots[:, i1] * w


def hold_pass(env: SpiderEnv, state0):
    """Reference post-processing: PD-track the kinematic reference with objects held on the demo."""
    env.reset(state0)
    env.hold, env.guide_gain = True, 0.0
    zero = torch.zeros(env.num_envs, env.n_act, device=env.device)
    q, tips = [env.e.robot.data.joint_pos[0].clone()], [env.tips()[0].clone()]
    th = time.time()
    for _ in range(env.T - 1):
        env.step(zero)
        q.append(env.e.robot.data.joint_pos[0].clone())
        tips.append(env.tips()[0].clone())
    env.hold = False
    torch.cuda.synchronize()
    print(f"  hold pass: {env.T - 1} frames x {env.num_envs} envs in {time.time() - th:.1f}s "
          f"({1000 * (time.time() - th) / (env.T - 1):.1f} ms/frame)", flush=True)
    q, tips = torch.stack(q), torch.stack(tips)
    shift = (tips - env.ref_tips[: len(q)]).norm(dim=-1)
    qerr = (q - env.ref_q[: len(q)]).abs()
    print(f"  hold pass: reached-vs-kinematic tip shift mean {100 * shift.mean():.2f} cm, "
          f"p95 {100 * shift.quantile(0.95):.2f} cm; per-30-frame means "
          f"{np.round(100 * shift.mean(1).cpu().numpy()[::30], 1).tolist()}; joint error max per joint (wrist "
          f"{np.round(qerr[:, env.wrist_ids].max(0).values.cpu().numpy(), 3).tolist()}, fingers "
          f"{np.round(qerr[:, env.finger_ids].max().item(), 3)})", flush=True)
    env.ref_q[: len(q)], env.ref_tips[: len(q)] = q, tips
    env.update_command()


def main():
    rdir = paths.run_dir(args.task, args.demos, args.episode)
    out_dir = rdir / args.target
    res_path = out_dir / f"{args.tag}.json"
    if res_path.exists():
        print(f"skip: {res_path} exists", flush=True)
        return
    ref = dict(np.load(rdir / "reference.npz", allow_pickle=True))
    ref["meta"] = json.loads(str(ref["meta"]))
    kin = dict(np.load(out_dir / "kinematic.npz", allow_pickle=True))
    demos = paths.load_demos(args.demos)
    src_ep = demos["episodes"][args.episode]

    renv, success_term = renv_mod.build(args.task, args.target, args.num_samples, args.device)
    env = SpiderEnv(renv, ref, kin, SpiderCfg(), success_term, args.success_steps)
    state0 = env.initial_state(src_ep["initial_state"])
    if not args.no_hold_pass:
        hold_pass(env, state0)
    if args.hold_only:
        return
    if args.max_frames:
        env.T = min(env.T, args.max_frames)
    T, dev, A = env.T, env.device, env.n_act
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.reset(state0)
    H, knot, S = args.horizon, args.knot, args.commit
    K = H // knot + 1
    nominal = torch.zeros(K, A, device=dev)
    sigma = torch.full((A,), args.sigma, device=dev)
    body_obj = env.body_obj.cpu().numpy()

    # the committed run (env 0): DexVerse trajectory layout -- states[t] before actions[t]
    traj = {"states": [to_numpy_state(renv.get_state())], "actions": [], "err": []}
    t0, n_opt = time.time(), 0
    history, retries, level, level_until = [], {}, 0, 0

    def commit_step(a):
        env.guide_gain = 0.0
        env.step(None if a is None else a[None].expand(env.num_envs, -1))
        traj["actions"].append(renv.targets_to_action(renv.q_target[0]))
        traj["states"].append(to_numpy_state(renv.get_state()))
        traj["err"].append(obj_err(env).cpu().numpy())
        env.success.update()

    while int(env.frame[0]) < T - 3:
        f = int(env.frame[0])
        history.append((env.snapshot(), len(traj["actions"]), nominal.clone(), env.success.state()))
        h = min(H, T - 3 - f)
        look = min(h, args.gate_look) if args.gate_look > 0 else h
        gate = env.gate[f + 1:f + 1 + look].max().item() if h > 0 else 0.0
        if gate > 0:
            s = env.snapshot()
            iters = args.iters * (1 + level)
            for it in range(iters):
                env.guide_gain = args.guide_gain * (1.0 - it / max(iters - 1, 1))
                env.restore(s)
                eps = torch.randn(env.num_envs, K, A, device=dev) * sigma * (1.0 + 0.5 * level) * (1.0 - 0.6 * it / iters)
                eps[0] = 0.0
                knots = (nominal[None] + eps).clamp(-1.0, 1.0)
                acts = knots_to_actions(knots, h, knot)
                cost = torch.zeros(env.num_envs, device=dev)
                for k in range(h):
                    env.step(acts[:, k])
                    cost += step_cost(env) * (1.0 + 2.0 * (k == h - 1))
                cost += 0.05 * (knots ** 2).sum((1, 2))
                adv = cost - cost.min()
                w = torch.softmax(-adv / (0.1 * adv.mean().clamp(min=1e-6)), 0)
                nominal = (w[:, None, None] * knots).sum(0)
            n_opt += 1
            env.guide_gain = 0.0
            env.restore(s)
            commit = knots_to_actions(nominal[None], min(S, h), knot)[0]
        else:
            commit = torch.zeros(min(S, max(h, 1)), A, device=dev)
        for k in range(commit.shape[0]):
            commit_step(commit[k])
        P = env.e.object_roots()[0, :, :3]
        if not torch.isfinite(P).all():
            raise RuntimeError(f"PhysX state corrupted at frame {int(env.frame[0])} - result discarded")
        fa = min(int(env.frame[0]), T - 1)
        active = {int(body_obj[a]) for a in env.active[max(0, fa - 15):fa + 1] if a >= 0}
        prev_len = history[-1][1]
        if active and prev_len > 0:
            e_now, e_prev = traj["err"][-1], traj["err"][prev_len - 1]
            dropped = [o for o in active if e_now[o] > 0.04 and e_now[o] - e_prev[o] > args.drop_jump]
            if dropped:
                back = max(len(history) - args.backtrack, 0)
                key = history[back][0]["frame"]
                if retries.get(key, 0) < args.max_retries:
                    retries[key] = level = retries.get(key, 0) + 1
                    snap, n_keep, nominal, sstate = history[back]
                    nominal = nominal.clone()
                    del history[back:]
                    del traj["actions"][n_keep:], traj["err"][n_keep:], traj["states"][n_keep + 1:]
                    env.success.load(sstate)
                    env.restore(snap)
                    level_until = key + (args.backtrack + 1) * S + H
                    print(f"  drop of {[env.e.obj_ids[x] for x in dropped]} at frame {fa}: back to {key}, "
                          f"retry {level}/{args.max_retries} with {args.iters * (1 + level)} iterations", flush=True)
                    continue
        if int(env.frame[0]) >= level_until:
            level = 0
        shift = max(commit.shape[0] // knot, 1)
        nominal = torch.cat([nominal[shift:], torch.zeros(shift, A, device=dev)], 0)
        if f // 60 != int(env.frame[0]) // 60:
            print(f"  frame {int(env.frame[0])}/{T} obj err cm {np.round(100 * traj['err'][-1], 1).tolist()} "
                  f"success now {env.success.history[-1]} ({time.time() - t0:.0f}s, {n_opt} optimised)", flush=True)

    for _ in range(args.settle):
        commit_step(None)

    done = env.success.done_frame()
    er = np.array(traj["err"])
    res = {"task": args.task, "episode": args.episode, "source": demos["robot_type"], "target": args.target,
           "frames": len(er), "settle_frames": args.settle, "seed": args.seed,
           "task_success": done is not None, "success_frame": done,
           "instant_success_frames": int(sum(env.success.history)),
           "minutes": round((time.time() - t0) / 60, 1), "optimised_steps": n_opt,
           "retries": {str(k): v for k, v in retries.items()}, "args": vars(args),
           "obj_err_mean_cm": dict(zip(env.e.obj_ids, (100 * er.mean(0)).round(2).tolist())),
           "obj_err_final_cm": dict(zip(env.e.obj_ids, (100 * er[-1]).round(2).tolist()))}
    out_dir.mkdir(parents=True, exist_ok=True)
    traj["success"] = list(env.success.history)
    traj["initial_state"] = state0
    traj["joint_names"] = renv.joint_names
    with gzip.open(out_dir / f"{args.tag}_traj.pkl.gz", "wb") as fh:
        pickle.dump(traj, fh)
    json.dump(res, open(res_path, "w"), indent=1)
    print(f"RESULT stage3 {args.task} ep{args.episode} {res['source']}->{args.target}: "
          f"{'SUCCESS' if res['task_success'] else 'fail'} ({res['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
