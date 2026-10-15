"""Bounded automatic-catch checks: cam, snap, hold, release and spring return.

Places joints at each physical test's initial state, then lets actual contact
and configured drives act. A separate predicate check seeds synthetic grasp
door-contact history to isolate hands-free dwell; this is not a teleop demo.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/dexverse"))
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, default=ROOT / "outputs/cutaway_door/physics.json")
parser.add_argument("--opening-torque", type=float, default=1.0, help="Diagnostic hinge torque in Nm; latch remains passive.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import dexverse.tasks  # noqa: E402,F401
from dexverse.tasks.utils import parse_env_cfg, strip_camera_cfgs, prune_stale_obs_refs  # noqa: E402
from dexverse.baseline_v1.mdp import door_latch  # noqa: E402


def main():
    cfg = parse_env_cfg("Dexverse-OpenDoor-v1", device=args.device, num_envs=1)
    cfg._apply_observation_preset("state")
    strip_camera_cfgs(cfg)
    prune_stale_obs_refs(cfg)
    env = gym.make("Dexverse-OpenDoor-v1", cfg=cfg).unwrapped
    asset = env.scene["articulation"]
    door_id = asset.find_joints("door_hinge")[0][0]
    latch_id = asset.find_joints("latch_slide")[0][0]
    results = {}

    def initialize(angle, height=0.0):
        env.reset(seed=42)
        robot = env.scene["robot"]
        root = robot.data.root_state_w.clone()
        root[:, 0] -= 2.0  # Move the hand away, not a prop holding the door.
        robot.write_root_pose_to_sim(root[:, :7])
        q = asset.data.default_joint_pos.clone()
        q[:, door_id], q[:, latch_id] = angle, height
        asset.write_joint_state_to_sim(q, torch.zeros_like(q))
        target = torch.zeros_like(q)
        target[:, door_id] = cfg.door_preload_angle
        asset.set_joint_position_target(target)
        asset.set_joint_effort_target(torch.zeros_like(q))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(cfg.sim.dt)

    def simulate(steps):
        angles, heights, support = [], [], []
        for _ in range(steps):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(cfg.sim.dt)
            angles.append(float(asset.data.joint_pos[0, door_id]))
            heights.append(float(asset.data.joint_pos[0, latch_id]))
            force = env.scene["door_latch_contact"].data.force_matrix_w
            support.append(float(force.norm(dim=-1).sum()) if force is not None else 0.0)
        assert torch.isfinite(asset.data.joint_pos).all()
        return {"angle": angles, "latch_height": heights, "support_force": support}

    initialize(0.9)
    env.episode_length_buf.fill_(1)
    assert not door_latch.door_reclosed(env).item()
    results["unlatched_closes"] = simulate(480)
    env.episode_length_buf.fill_(2)
    assert door_latch.door_reclosed(env).item(), "Reclosed failure did not trigger after opening"
    print("FREE_CLOSE", results["unlatched_closes"]["angle"][::60], flush=True)
    # Open using applied hinge torque as a stand-in for the hand, never by
    # teleporting through the catch or directly driving the latch joint.
    initialize(0.9)
    effort = torch.zeros_like(asset.data.joint_pos)
    effort[:, door_id] = args.opening_torque
    asset.set_joint_effort_target(effort)
    approach = {"angle": [], "latch_height": [], "support_force": []}
    for _ in range(600):
        sample = simulate(1)
        for key in approach:
            approach[key].extend(sample[key])
        if sample["angle"][-1] >= 1.6057:  # ~92 degrees, then release opening effort.
            break
    results["automatic_engagement"] = approach
    asset.set_joint_effort_target(torch.zeros_like(effort))
    print("AUTO_APPROACH", approach["angle"][-1], max(approach["latch_height"]), flush=True)
    results["catch_holds"] = simulate(360)
    env.episode_length_buf.fill_(1)
    assert not door_latch.door_latched_and_released(env).item(), "No recorded hand release yet"
    # Isolate the final support/release gate. This is synthetic history for a
    # predicate test, not a teleop demonstration or proof of hand reachability.
    env.cutaway_door_progress[:, 1] = 1
    for index in range(2, 62):
        simulate(cfg.decimation)
        env.episode_length_buf.fill_(index)
        success = door_latch.door_latched_and_released(env)
        assert bool(success.item()) == (index == 61), "Physical release dwell counted incorrectly"
    print(
        "CATCH_HOLD",
        results["catch_holds"]["angle"][::60],
        results["catch_holds"]["latch_height"][::60],
        results["catch_holds"]["support_force"][::60],
        flush=True,
    )
    q = asset.data.joint_pos.clone()
    q[:, latch_id] = 0.05
    asset.write_joint_state_to_sim(q, torch.zeros_like(q))
    target = torch.zeros_like(q)
    target[:, door_id] = cfg.door_preload_angle
    target[:, latch_id] = 0.05  # Diagnostic release, not needed for task success.
    asset.set_joint_position_target(target)
    results["raised_catch_releases"] = simulate(480)
    print("RELEASE", results["raised_catch_releases"]["angle"][::60], flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    checks = {
        "automatically_cammed": max(approach["latch_height"]) > 0.025,
        "reached_open_angle": approach["angle"][-1] >= 1.6057,
        "spring_returned_catch": max(results["catch_holds"]["latch_height"][-120:]) < 0.012,
        "unlatched_closes": results["unlatched_closes"]["angle"][-1] < 0.09,
        "catch_holds": min(results["catch_holds"]["angle"][-120:]) > 1.46,
        "support_measured": max(results["catch_holds"]["support_force"][-120:]) > 0.05,
        "raised_catch_releases": results["raised_catch_releases"]["angle"][-1] < 0.10,
    }
    args.report.write_text(
        json.dumps(
            {
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "dt": cfg.sim.dt,
                "opening_torque_nm": args.opening_torque,
                "release_dwell_steps": cfg.hold_steps,
                "synthetic_prior_door_contact": True,
                "tests": results,
            },
            indent=2,
        )
    )
    assert all(checks.values()), checks
    assert results["unlatched_closes"]["angle"][-1] < 0.09, "Closing spring does not close the free door"
    assert min(results["catch_holds"]["angle"][-120:]) > 1.46, "Catch did not physically hold open"
    assert max(results["catch_holds"]["support_force"][-120:]) > 0.05, "No catch support contact"
    assert results["raised_catch_releases"]["angle"][-1] < 0.10, "Door remains locked after catch clears"
    print("CUTAWAY_PHYSICS: automatic cam/snap, contact retention, release dwell and spring return passed", flush=True)


try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
