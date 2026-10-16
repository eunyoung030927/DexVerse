# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that a hand (``robot_type``) loads and simulates in a task: build the task as record_demos does, hold the
hand at its home joints for a few seconds through ``env.step``, and report what went wrong, if anything.

Checks (second half of the hold, after settling): the env builds; the robot config's palm / fingertip bodies exist; every action term is a joint-position
action covering the robot; holding home keeps the hand still (joint drift) and finite; the task objects stay on
the table (no explosion from initial interpenetration); the success term evaluates.

  python scripts/retarget/check_hand_in_task.py --task Dexverse-GraspCup-v0 --robot_type floating_allegro_right \
      --headless
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--robot_type", required=True)
ap.add_argument("--steps", type=int, default=120, help="env steps (60 Hz) holding the home pose")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dexverse.tasks  # noqa: E402,F401
from dexverse.mimic.env_setup import build_env_cfg, resolve_success_term  # noqa: E402


def main():
    rep = {"task": args.task, "robot_type": args.robot_type, "ok": False, "problems": []}
    t0 = time.time()
    try:
        cfg, success = build_env_cfg(args.task, args.robot_type, device=args.device, num_envs=1, keep_cameras=False)
        env = gym.make(args.task, cfg=cfg).unwrapped
        env.reset()
    except Exception as exc:  # noqa: BLE001
        rep["problems"].append(f"build failed: {type(exc).__name__}: {str(exc)[:300]}")
        return rep
    rep["build_s"] = round(time.time() - t0, 1)
    robot = env.scene["robot"]
    rep["joints"] = len(robot.joint_names)
    rep["action_dim"] = int(env.action_manager.total_action_dim)
    bodies = set(robot.body_names)
    rc = cfg.robot_config
    want = [b for b in [getattr(rc, "palm_body_name", None), getattr(rc, "right_palm_body_name", None),
                        getattr(rc, "left_palm_body_name", None), *(getattr(rc, "fingertip_body_names", None) or [])] if b]
    missing = [b for b in want if b not in bodies]
    if missing:
        rep["problems"].append(f"robot_config bodies missing from the articulation: {missing}")
    rep["fingertips"] = len(getattr(rc, "fingertip_body_names", None) or [])

    # action that commands the home joints: JointPositionAction target = raw * scale + offset
    raw = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    col, covered = 0, set()
    home = robot.data.default_joint_pos[0]
    for name, term in env.action_manager._terms.items():
        kind = type(term).__name__
        ids = term._joint_ids
        ids = list(range(len(robot.joint_names))) if isinstance(ids, slice) else list(ids)
        if kind != "JointPositionAction":
            rep["problems"].append(f"action term {name} is {kind} (expected JointPositionAction)")
        scale, off = term._scale, term._offset
        for c, j in enumerate(ids):
            s = float(scale[0, c]) if torch.is_tensor(scale) else float(scale)
            o = float(off[0, c]) if torch.is_tensor(off) else float(off)
            raw[0, col + c] = (home[j] - o) / s
            covered.add(j)
        col += term.action_dim
    uncovered = [robot.joint_names[j] for j in range(len(robot.joint_names)) if j not in covered]
    if uncovered:
        rep["uncovered_joints"] = uncovered       # mimic / passive joints are expected here

    # first half: the hand moves from its (possibly randomized) reset pose to home and objects settle;
    # second half: everything must stay put
    objs = {k: a for k, a in env.scene.rigid_objects.items() if not k.startswith(("table", "success_marker"))}
    try:
        for _ in range(args.steps // 2):
            env.step(raw)
        p0 = {k: a.data.root_pos_w[0].clone() for k, a in objs.items()}
        q0 = robot.data.joint_pos[0].clone()
        rep["home_err_max"] = round(float((q0 - home)[sorted(covered)].abs().max()), 4)
        for _ in range(args.steps - args.steps // 2):
            env.step(raw)
    except Exception as exc:  # noqa: BLE001
        rep["problems"].append(f"step failed: {type(exc).__name__}: {str(exc)[:300]}")
        return rep
    q = robot.data.joint_pos[0]
    if not torch.isfinite(q).all():
        rep["problems"].append("robot joints became NaN/inf")
    drift = (q - q0).abs()
    act_ids = sorted(covered)
    rep["joint_drift_max"] = round(float(drift[act_ids].max()), 4) if act_ids else None
    worst = int(drift.argmax())
    rep["joint_drift_worst"] = robot.joint_names[worst]
    if rep["joint_drift_max"] is not None and rep["joint_drift_max"] > 0.05:
        rep["problems"].append(f"hand does not hold home: {robot.joint_names[worst]} moved {float(drift[worst]):.3f}")
    moved = {k: round(float((a.data.root_pos_w[0] - p0[k]).norm()), 4) for k, a in objs.items()}
    rep["object_moved_m"] = moved
    flying = {k: v for k, v in moved.items() if v > 0.05}
    if flying:
        rep["problems"].append(f"objects moved > 5 cm in the second half while the hand held still: {flying}")
    try:
        term = resolve_success_term(env, success)
        rep["success_now"] = bool(term.func(env, **term.params)[0])
    except Exception as exc:  # noqa: BLE001
        rep["problems"].append(f"success term failed: {type(exc).__name__}: {str(exc)[:200]}")
    rep["ok"] = not rep["problems"]
    rep["total_s"] = round(time.time() - t0, 1)
    return rep


if __name__ == "__main__":
    r = main()
    print("RESULT " + json.dumps(r), flush=True)
    sys.stdout.flush()
    os._exit(0)
