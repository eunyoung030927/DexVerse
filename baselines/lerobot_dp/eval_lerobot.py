#!/usr/bin/env python
"""Closed-loop evaluation of a LeRobot Diffusion Policy from the recorded starts of its demo pickle (main env, Isaac Sim).

Task, hand and starts all come from --demos: episode k restores episode k of that pickle (initial_state, active masks,
task_state). Cameras render at the dataset's 480x640 and images go to the policy without resizing; observation.state is
joint_pos in LeRobotSpoolRecorder order, read before env.step like the recording. No evaluation seed is set.

  python eval_lerobot.py --demos <demos.pkl> --ckpt <pretrained_model> --output_dir <dir> --headless --device cpu
"""
import argparse
import json
import os
import pickle
import secrets
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--demos", required=True, help="Trajectory pickle of the dataset the policy was trained on.")
parser.add_argument("--ckpt", required=True, help="pretrained_model folder of a checkpoint.")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--episodes", default=None, help="Comma list of pickle episodes (0-based); default all 50.")
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--save_first_frames", action="store_true", help="Save the first images of each episode (checks).")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(rendering_mode="performance", device="cpu")  # record_demos defaults, used by the LeRobot conversion
args = parser.parse_args()
args.enable_cameras = True
if os.environ.get("DEXVERSE_FINGER_DRIVE"):
    sys.exit("DEXVERSE_FINGER_DRIVE is set: the teleop datasets use the robot config drives")
simulation_app = AppLauncher(args).app

HERE = Path(__file__).resolve().parent
IMAGE_SIZE = (480, 640)
CAMERAS = ("third_person_camera", "wrist_camera", "right_wrist_camera", "left_wrist_camera")
LEROBOT_PYTHON = "/workspace/envs/dexverse/lerobot/bin/python"


def check_policy_matches(payload: dict) -> None:
    config = Path(args.ckpt) / "train_config.json"
    if config.exists():
        root = str(json.loads(config.read_text())["dataset"]["root"]).lower()
        task = payload["env_name"].removeprefix("Dexverse-").split("-")[0].lower()
        if task not in root or payload["robot_type"] not in root:
            sys.exit(f"{args.ckpt} was trained on {root}, not on {payload['env_name']} {payload['robot_type']}")


def make_env(payload: dict):
    import gymnasium as gym

    import dexverse.tasks  # noqa: F401
    from dexverse.IL.diffusion.online_eval import _SuccessProbe
    from dexverse.replay_rigid_object import configure_replay_rigid_objects
    from dexverse.tasks.utils import parse_env_cfg
    from dexverse.teleop_utils.debug_visualization import configure_v1_debug_visualization
    from dexverse.teleop_utils.replay_teleop import recorded_wrist_camera_body_collision, set_wrist_camera_body_collision

    task, robot_type = payload["env_name"], payload["robot_type"]
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1, json_path=payload.get("json_path"))
    env_cfg = type(env_cfg)(robot_type=robot_type, enable_debug_vis=True)  # the datasets show the goal cues
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 1
    configure_v1_debug_visualization(env_cfg, cues_in_rgb=True)
    env_cfg.seed = payload["seed"]
    set_wrist_camera_body_collision(env_cfg, recorded_wrist_camera_body_collision(payload, robot_type))
    for name in CAMERAS:  # same override as record_demos._configure_lerobot_cameras
        cam = getattr(env_cfg.scene, name, None)
        if cam is not None:
            cam.height, cam.width = IMAGE_SIZE
            cam.data_types = ["rgb"]
    for group in ("depth", "pointcloud"):
        if getattr(env_cfg.observations, group, None) is not None:
            setattr(env_cfg.observations, group, None)
    env_cfg.recorders = {}
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    probe = _SuccessProbe(env_cfg.terminations.success)
    env_cfg.terminations.success = None
    configure_replay_rigid_objects(env_cfg.scene)
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(env, video_folder=f"{args.output_dir}/videos", episode_trigger=lambda _: True,
                                       video_length=args.max_steps, disable_logger=True)
    return env, env_cfg, probe


def camera_keys(env_cfg) -> dict:
    scene = env_cfg.scene
    wrist = next(n for n in CAMERAS[1:] if getattr(scene, n, None) is not None)
    keys = {"observation.images.third_person": "third_person_camera", "observation.images.eye_in_hand": wrist}
    if wrist == "right_wrist_camera" and getattr(scene, "left_wrist_camera", None) is not None:
        keys["observation.images.eye_in_hand_left"] = "left_wrist_camera"
    return keys


def state_order(env) -> list:
    robot = env.scene["robot"]
    ids = []
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        term_ids = term._joint_ids
        term_ids = list(range(robot.num_joints))[term_ids] if isinstance(term_ids, slice) else list(term_ids)
        ids += [int(i) for i in term_ids[: int(term.action_dim)]]
    return ids + [i for i in range(robot.num_joints) if i not in set(ids)]


def restore_episode(env, episode: dict) -> float:
    """Restore a recorded start the way the demo replay does; returns the largest error against the recording."""
    import torch

    sys.path.insert(0, f"{os.environ['DEXVERSE_ROOT']}/scripts/demo_tools")
    from _active_object_masks import apply_recorded_active_masks

    from dexverse.scene_state import clear_joint_velocity_targets
    from dexverse.teleop_utils.episode_task_state import reset_managers_after_state_restore, restore_episode_task_state
    from dexverse.teleop_utils.replay_teleop import _to_torch

    env_ids = torch.tensor([0], device=env.device)
    state = _to_torch(episode["initial_state"], env.device)
    env.reset_to(state, env_ids, is_relative=True)
    clear_joint_velocity_targets(env.scene, env_ids)
    apply_recorded_active_masks(env, episode, env_ids)
    skipped = restore_episode_task_state(env, episode.get("task_state"), env_index=0,
                                         legacy_goal_pose=episode.get("goal_pose"))
    if skipped:
        raise RuntimeError(f"task state not restored: {skipped}")
    reset_managers_after_state_restore(env)

    def largest(recorded, live):
        if isinstance(recorded, dict):
            return max((largest(recorded[k], live[k]) for k in recorded), default=0.0)
        return float((recorded.float() - live.float()).abs().max()) if recorded.numel() else 0.0

    return largest(state, env.scene.get_state(is_relative=True))


def observe(env, keys, order, refresh) -> dict:
    if refresh:  # after a restore the cameras still hold the previous render
        env.sim.render()
    message = {}
    for key, name in keys.items():
        cam = env.scene[name]
        if refresh:
            cam.update(0.0, force_recompute=True)
        image = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        message[key] = (image.shape, "uint8", image.tobytes())
    state = env.scene["robot"].data.joint_pos[0, order].cpu().numpy().astype("float32")
    message["observation.state"] = (state.shape, "float32", state.tobytes())
    return message


def start_server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    authkey = secrets.token_hex(16)
    environ = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH", "CONDA_PREFIX")}
    server = subprocess.Popen([LEROBOT_PYTHON, str(HERE / "policy_server.py"), "--ckpt", args.ckpt, "--port", str(port),
                               "--authkey", authkey], env=environ)
    for _ in range(300):
        try:
            return server, Client(("127.0.0.1", port), authkey=authkey.encode())
        except ConnectionRefusedError:
            if server.poll() is not None:
                raise RuntimeError("policy server exited")
            time.sleep(1)
    raise RuntimeError("policy server did not start")


def main():
    import numpy as np
    import torch

    with open(args.demos, "rb") as f:
        payload = pickle.load(f)
    demos = payload["episodes"]
    if len(demos) != 50:
        raise ValueError(f"{args.demos}: expected 50 episodes, got {len(demos)}")
    check_policy_matches(payload)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = [int(i) for i in args.episodes.split(",")] if args.episodes else list(range(len(demos)))
    server, conn = start_server()
    env, env_cfg, probe = make_env(payload)
    base = env.unwrapped
    keys, order = camera_keys(env_cfg), state_order(base)

    results = []
    for done, index in enumerate(indices):
        env.reset()
        with torch.inference_mode():
            restore_error = restore_episode(base, demos[index])
        conn.send(("reset", None))
        conn.recv()
        succeeded, steps = False, args.max_steps
        for step in range(args.max_steps):
            message = observe(base, keys, order, refresh=step == 0)
            if step == 0 and args.save_first_frames:
                np.savez(output_dir / f"first_ep{index:03d}.npz", **{
                    key: np.frombuffer(value[2], dtype=value[1]).reshape(value[0]) for key, value in message.items()})
            conn.send(("act", message))
            action = torch.frombuffer(bytearray(conn.recv()), dtype=torch.float32).to(base.device)
            _, _, terminated, truncated, _ = env.step(action.unsqueeze(0))
            succeeded = succeeded or probe(base)
            if (step + 1) % 500 == 0:  # heartbeat: a slow 3000-step episode (>30 min on a shared GPU) must not look like a hang
                print(f"[eval] episode={index} step={step + 1}", flush=True)
            if succeeded or bool(terminated[0]) or bool(truncated[0]):
                steps = step + 1
                break
        results.append({"episode": index, "source_episode": demos[index].get("episode_index"), "success": succeeded,
                        "steps": steps, "restore_error": restore_error})
        print(f"[eval] {done + 1}/{len(indices)} episode={index} restore_error={restore_error:.2e} "
              f"success={'yes' if succeeded else 'no'} steps={steps} "
              f"running_success={sum(r['success'] for r in results)}/{done + 1}", flush=True)
        (output_dir / "metrics.json").write_text(json.dumps(
            {"success_rate": sum(r["success"] for r in results) / len(results), "num_episodes": len(results),
             "task": payload["env_name"], "robot_type": payload["robot_type"], "demos": args.demos,
             "image_size": IMAGE_SIZE, "episodes": results}, indent=2))
    conn.send(("close", None))
    server.wait(timeout=60)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
