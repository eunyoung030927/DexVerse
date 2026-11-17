#!/usr/bin/env python
"""handmask (261115): pixel-exact hand masks for the Dexverse GraspKettle teleop demos by STATE REPLAY.

For each recorded episode (dv-demos-v0 teleop pickle, record_state=True): restore the initial scene exactly as the
policy eval does (p3h_nh_eval_pooled.restore_episode), then for every LeRobot frame k write the recorded scene state
states[k] (record_demos captures frame k BEFORE action k; states[0] = initial state), render, and read the same
TiledCameras the dataset uses (third_person_camera, wrist_camera; 480x640, rendering_mode performance, debug cues in
RGB, as record_demos / eval) with an added instance_id_segmentation_fast annotator.  Hand mask = pixels whose instance
prim path lies under the env's Robot prim (all hand links).  Nothing about the scene or cameras is changed except the
extra annotator.
Outputs <output>/<hand>/ep<k>.npz : masks_<cam> packbits (T,480,640 bool, axis=-1), area_<cam> (T,), and rgb_<cam>
for every --rgb_every-th frame (alignment check against the dataset frames), labels (hand prim paths seen).
Isaac python in the xhand container, cwd /workspace/xhand/latent_codec_v1:
  python.sh -m xhand_latent.hm_render_masks --hand wuji --episodes 0-49 --output artifacts/handmask/masks --headless
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hand", required=True)
parser.add_argument("--task", default="Dexverse-GraspKettle-v0")
parser.add_argument("--episodes", default="0-49")
parser.add_argument("--output", required=True)
parser.add_argument("--rgb_every", type=int, default=25)
parser.add_argument("--renders_per_frame", type=int, default=1)
parser.add_argument("--max_frames", type=int, default=None, help="debug: limit frames per episode")
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(rendering_mode="performance", device="cpu")
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

LC = "/workspace/xhand/latent_codec_v1"
sys.path.insert(0, LC)
from xhand_latent.data import load_trajectory, trajectory_path  # noqa: E402
from xhand_latent.hm_seg import add_segmentation, hand_mask_from_camera  # noqa: E402

IMAGE_SIZE = (480, 640)
CAMERAS = ("third_person_camera", "wrist_camera", "right_wrist_camera", "left_wrist_camera")
DATA_ROOT = "/workspace/xhand/data/dv-demos-v0"


def parse_eps(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def make_env(payload: dict):  # = p3h_nh_eval_pooled.make_env (no video) + segmentation annotator on the dataset cameras
    import gymnasium as gym

    import dexverse.tasks  # noqa: F401
    from dexverse.replay_rigid_object import configure_replay_rigid_objects
    from dexverse.tasks.utils import parse_env_cfg
    from dexverse.teleop_utils.debug_visualization import configure_v1_debug_visualization
    from dexverse.teleop_utils.replay_teleop import recorded_wrist_camera_body_collision, set_wrist_camera_body_collision

    task, robot_type = payload["env_name"], payload["robot_type"]
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1, json_path=payload.get("json_path"))
    env_cfg = type(env_cfg)(robot_type=robot_type, enable_debug_vis=True)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 1
    configure_v1_debug_visualization(env_cfg, cues_in_rgb=True)
    env_cfg.seed = payload["seed"]
    set_wrist_camera_body_collision(env_cfg, recorded_wrist_camera_body_collision(payload, robot_type))
    for name in CAMERAS:
        cam = getattr(env_cfg.scene, name, None)
        if cam is not None:
            cam.height, cam.width = IMAGE_SIZE
            cam.data_types = ["rgb"]
    for name in ("third_person_camera_left", "third_person_camera_right"):  # unused extra views (speed only)
        if getattr(env_cfg.scene, name, None) is not None:
            setattr(env_cfg.scene, name, None)
    for group in ("depth", "pointcloud"):
        if getattr(env_cfg.observations, group, None) is not None:
            setattr(env_cfg.observations, group, None)
    env_cfg.recorders = {}
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.success = None
    configure_replay_rigid_objects(env_cfg.scene)
    wrist = next(n for n in CAMERAS[1:] if getattr(env_cfg.scene, n, None) is not None)
    cams = {"tp": "third_person_camera", "eih": wrist}
    for n in cams.values():
        add_segmentation(getattr(env_cfg.scene, n))
    env = gym.make(task, cfg=env_cfg)
    return env, cams


def restore_episode(env, episode: dict) -> float:  # = p3h_nh_eval_pooled.restore_episode
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


def main():
    from dexverse.scene_state import reset_scene_to
    from dexverse.teleop_utils.replay_teleop import _to_torch

    out = Path(args.output) / args.hand
    out.mkdir(parents=True, exist_ok=True)
    eps = parse_eps(args.episodes)
    payload0 = load_trajectory(trajectory_path(DATA_ROOT, args.task, args.hand, eps[0]))
    env, cams = make_env(payload0)
    base = env.unwrapped
    env.reset()
    log = open(out / "render_log.jsonl", "a")
    for e in eps:
        dst = out / f"ep{e:03d}.npz"
        if dst.exists() and not args.overwrite:
            continue
        payload = load_trajectory(trajectory_path(DATA_ROOT, args.task, args.hand, e))
        assert payload["robot_type"] == payload0["robot_type"] and payload["seed"] == payload0["seed"]
        ep = payload["episodes"][0]
        T = len(ep["actions"])
        if args.max_frames:
            T = min(T, args.max_frames)
        t0 = time.time()
        env.reset()
        with torch.inference_mode():
            rerr = restore_episode(base, ep)
        masks = {c: np.zeros((T, *IMAGE_SIZE), bool) for c in cams}
        rgbs = {c: [] for c in cams}
        labels = set()
        for k in range(T):
            with torch.inference_mode():
                reset_scene_to(base.scene, _to_torch(ep["states"][k], base.device), env_ids=None, is_relative=True)
                base.sim.forward()
                base.scene.update(dt=0.0)
            for _ in range(args.renders_per_frame):
                base.sim.render()
            for c, name in cams.items():
                cam = base.scene[name]
                cam.update(0.0, force_recompute=True)
                m, lab = hand_mask_from_camera(cam)
                masks[c][k] = m.cpu().numpy()
                labels |= set(lab)
                if k % args.rgb_every == 0:
                    rgbs[c].append(cam.data.output["rgb"][0, ..., :3].cpu().numpy().copy())  # cpu device: copy, buffer is reused
        area = {c: masks[c].reshape(T, -1).mean(1) for c in cams}
        np.savez_compressed(dst, **{f"masks_{c}": np.packbits(masks[c], axis=-1) for c in cams},
                            **{f"area_{c}": area[c].astype(np.float32) for c in cams},
                            **{f"rgb_{c}": np.stack(rgbs[c]) for c in cams}, rgb_frames=np.arange(0, T, args.rgb_every),
                            T=np.int32(T), labels=np.array(sorted(labels)))
        rec = {"episode": e, "T": T, "restore_error": rerr, "wall_s": round(time.time() - t0, 1),
               **{f"area_{c}_mean": float(area[c].mean()) for c in cams}, "n_labels": len(labels)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print("[hm_render]", json.dumps(rec), flush=True)
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
