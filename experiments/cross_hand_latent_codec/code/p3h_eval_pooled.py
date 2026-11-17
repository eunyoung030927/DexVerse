#!/usr/bin/env python
"""P3 (pooled3hand_design_261112 sec.5): closed-loop evaluation of a POOLED 3-hand LeRobot DP (arm C/Z/D). New file.

Derived from lerobot_dp/eval_lerobot.py (unchanged original: /workspace/xhand/lerobot_dp_pretrained/code/lerobot_dp/),
identical env construction, initial_state restore, camera images, policy server (same policy_server.py: DDPM 100 steps,
n_action_steps 8 from the checkpoint config), success probe, max_steps and no evaluation seed. Added hooks only:
  state : observation.state = P3HCodec.encode_state(hand, joint_pos[LeRobot order], rootT, arm)  (incl. 3-way one-hot)
  action: env action = P3HCodec.decode_action(hand, policy_vec, rootT, seed_q = current observed q, arm)[0]
          (C: slice to the hand's action dims; Z/D: E2* Dec_hand(z) + [palm | EE_common o f_task(q_hat)] + floating-wrist
          solve). Every action of the 8-step chunk is decoded when it is executed, seeded with the q observed then.
  rootT : pkl episode initial_state robot root_pose (as P1 dataset building); checked against the live root each episode.
  trace : per step q (LeRobot state order), kettle (rigid object "object") and robot root pose relative to the env origin,
          raw policy vector -> <output_dir>/traces/ep<k>.npz.  At episode end t_lift / t_g are computed with the P0a
          segmentation (p3h_train.segment, kinematic rule, touch_contact unused) and the canonical-palm pose in the kettle
          frame at t_g is stored in metrics.json (wrist_k_tg: pos m + rot6d).  After success the rollout continues
          --post_success_steps steps for the trace only (success / steps are fixed at the success step, as in A).
  --replay_npz: check mode, no policy: feed pooled-dataset action rows (<arm>_<hand>_ep<k>_action) through the same
          decode path, then hold the last row 120 steps (replay_sim --hold-steps default).
  --resume (default): episodes already in metrics.json are kept and skipped.
Isaac python in the xhand container:
  python.sh p3h_eval_pooled.py --arm Z --demos <pkl> --ckpt <pretrained_model> --output_dir <dir> --episodes 0,1 \
      --max_steps 3000 --no-video --headless
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
parser.add_argument("--arm", required=True, choices=("C", "Z", "D"))
parser.add_argument("--demos", required=True)
parser.add_argument("--ckpt", default=None, help="pretrained_model folder (not needed with --replay_npz)")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--episodes", default=None)
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--post_success_steps", type=int, default=30)
parser.add_argument("--replay_npz", default=None, help="check (b): dataset action rows instead of the policy")
parser.add_argument("--hold_steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(rendering_mode="performance", device="cpu")
args = parser.parse_args()
args.enable_cameras = True
if os.environ.get("DEXVERSE_FINGER_DRIVE"):
    sys.exit("DEXVERSE_FINGER_DRIVE is set: the teleop datasets use the robot config drives")
simulation_app = AppLauncher(args).app

LC = "/workspace/xhand/latent_codec_v1"
DP = "/workspace/xhand/lerobot_dp_pretrained/code/lerobot_dp"
os.environ.setdefault("P3H_PD2", f"{LC}/artifacts/pooled3hand/p0b/ext/pd2_work2")
sys.path.insert(0, DP)
sys.path.insert(0, LC)

# ---- verbatim copy of eval_lerobot.py helpers (env, cameras, state order, restore, observe, policy server) ----
HERE = Path(DP)  # the original policy_server.py is used unchanged
IMAGE_SIZE = (480, 640)
CAMERAS = ("third_person_camera", "wrist_camera", "right_wrist_camera", "left_wrist_camera")
LEROBOT_PYTHON = "/workspace/envs/dexverse/lerobot/bin/python"


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


# ---- end of verbatim copy ----


def check_policy_matches():
    if args.ckpt is None:
        return {}
    cfg = json.loads((Path(args.ckpt) / "train_config.json").read_text())
    root = str(cfg["dataset"]["root"]).rstrip("/")
    if not root.endswith(f"pooled3hand/kettle3_{args.arm}"):
        sys.exit(f"{args.ckpt} was trained on {root}, not on pooled3hand/kettle3_{args.arm}")
    pc = json.loads((Path(args.ckpt) / "config.json").read_text())
    return {"dataset_root": root, "n_obs_steps": pc["n_obs_steps"], "horizon": pc["horizon"],
            "n_action_steps": pc["n_action_steps"], "num_inference_steps": pc.get("num_inference_steps"),
            "num_train_timesteps": pc.get("num_train_timesteps"), "state_dim": pc["input_features"]["observation.state"]["shape"],
            "action_dim": pc["output_features"]["action"]["shape"]}


def load_codec():
    cwd = os.getcwd()
    os.chdir(LC)  # E2* metadata / data paths are relative to latent_codec_v1
    try:
        from xhand_latent.p3h_policy_codec import DATA, E2, F_LOADER, F_TASK, P3HCodec, state_dim, action_dim
        from xhand_latent.p3h_train import segment
        codec = P3HCodec(f"{LC}/{E2}", f"{LC}/{F_TASK}", f"{LC}/{F_LOADER}", f"{LC}/{DATA}")
    finally:
        os.chdir(cwd)
    return codec, segment, state_dim, action_dim


def main():
    import numpy as np
    import torch

    from xhand_latent.decode_replay import canonical_world
    from xhand_latent.p3h_roundtrip import inv
    from xhand_latent.reference_kin import make_T, rot6d_from_mat

    with open(args.demos, "rb") as f:
        payload = pickle.load(f)
    demos = payload["episodes"]
    if len(demos) != 50:
        raise ValueError(f"{args.demos}: expected 50 episodes, got {len(demos)}")
    hand = payload["robot_type"].removeprefix("floating_").removesuffix("_right")
    pol_cfg = check_policy_matches()
    codec, segment, state_dim, action_dim = load_codec()
    if pol_cfg and (pol_cfg["state_dim"] != [state_dim(args.arm)] or pol_cfg["action_dim"] != [action_dim(args.arm)]):
        sys.exit(f"policy dims {pol_cfg} do not match arm {args.arm}")
    fk = codec.fks[hand]
    action_names, state_names = codec.names[hand]
    output_dir = Path(args.output_dir)
    (output_dir / "traces").mkdir(parents=True, exist_ok=True)
    indices = [int(i) for i in args.episodes.split(",")] if args.episodes else list(range(len(demos)))
    results = []
    mpath = output_dir / "metrics.json"
    if args.resume and mpath.exists():
        old = json.loads(mpath.read_text())
        if old.get("arm") == args.arm and old.get("ckpt") == args.ckpt:
            results = [r for r in old["episodes"] if r["episode"] in indices]
    done_eps = {r["episode"] for r in results}
    replay = np.load(args.replay_npz) if args.replay_npz else None
    server = conn = None
    if replay is None:
        server, conn = start_server()
    env, env_cfg, probe = make_env(payload)
    base = env.unwrapped
    keys, order = camera_keys(env_cfg), state_order(base)
    robot = base.scene["robot"]
    live_names = [robot.joint_names[i] for i in order]
    if live_names != list(state_names):
        raise RuntimeError(f"env state order != codec state names:\n{live_names}\n{list(state_names)}")
    n_act = base.action_manager.total_action_dim
    if n_act != len(action_names):
        raise RuntimeError(f"env action dim {n_act} != codec action names {len(action_names)}")
    obj = base.scene["object"]
    origin = base.scene.env_origins[0]

    def write_metrics():
        rs = sorted(results, key=lambda r: r["episode"])
        mpath.write_text(json.dumps(
            {"success_rate": sum(r["success"] for r in rs) / max(1, len(rs)), "num_episodes": len(rs),
             "task": payload["env_name"], "robot_type": payload["robot_type"], "hand": hand, "arm": args.arm,
             "ckpt": args.ckpt, "policy_config": pol_cfg, "replay_npz": args.replay_npz, "demos": args.demos,
             "image_size": IMAGE_SIZE, "post_success_steps": args.post_success_steps, "episodes": rs}, indent=2))

    for done, index in enumerate(indices):
        if index in done_eps:
            continue
        ep = demos[index]
        env.reset()
        with torch.inference_mode():
            restore_error = restore_episode(base, ep)
        root7 = np.asarray(ep["initial_state"]["articulation"]["robot"]["root_pose"], np.float64)[0]
        rootT = codec.root_T(root7)
        live_root = (robot.data.root_state_w[0, :7] - torch.cat([origin, origin.new_zeros(4)])).cpu().numpy()
        root_err = float(np.abs(live_root[:3] - root7[:3]).max())
        if replay is None:
            conn.send(("reset", None))
            conn.recv()
            rows = None
        else:
            rows = replay[f"{args.arm}_{hand}_ep{index}_action"]
        succeeded, steps, post = False, args.max_steps, 0
        limit = args.max_steps if rows is None else min(args.max_steps, len(rows) + args.hold_steps)
        qs, objs, vecs, solve_err, t_codec, decode_fail, prev_act = [], [], [], [], 0.0, 0, None
        wall = time.monotonic()
        for step in range(limit + args.post_success_steps):
            message = observe(base, keys, order, refresh=step == 0)
            shape, dtype, data = message["observation.state"]
            raw = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            t0 = time.perf_counter()
            st = codec.encode_state(hand, raw, rootT, args.arm)[0].astype(np.float32)
            t_codec += time.perf_counter() - t0
            message["observation.state"] = (st.shape, "float32", st.tobytes())
            qs.append(raw)
            objs.append((obj.data.root_state_w[0, :7] - torch.cat([origin, origin.new_zeros(4)])).cpu().numpy())
            if rows is None:
                conn.send(("act", message))
                vec = np.frombuffer(bytearray(conn.recv()), dtype=np.float32)
            else:
                vec = rows[min(step, len(rows) - 1)]
            vecs.append(vec)
            t0 = time.perf_counter()
            try:
                act, _, stats = codec.decode_action(hand, vec, rootT, codec.q_from_state(hand, raw)[0], args.arm)
                if stats:
                    solve_err.append([stats.get("max_wrist_position_error_mm", np.nan),
                                      stats.get("max_wrist_rotation_error_deg", np.nan)])
            except ValueError as exc:  # wrist solve infeasible (no Euler branch / translation limit / FK gate)
                decode_fail += 1
                if decode_fail <= 3:
                    print(f"[eval] episode={index} step={step} decode failed: {str(exc)[:160]}", flush=True)
                # hold: previous decoded action, or (first step) the current observed q as target
                act = prev_act if prev_act is not None else codec.action_from_q(hand, codec.q_from_state(hand, raw))
            t_codec += time.perf_counter() - t0
            prev_act = act
            action = torch.as_tensor(act[0], dtype=torch.float32, device=base.device)
            _, _, terminated, truncated, _ = env.step(action.unsqueeze(0))
            if succeeded:
                post += 1
                if post >= args.post_success_steps or bool(terminated[0]) or bool(truncated[0]):
                    break
                continue
            succeeded = succeeded or probe(base)
            if (step + 1) % 500 == 0:
                print(f"[eval] episode={index} step={step + 1}", flush=True)
            if succeeded:
                steps = step + 1
                if args.post_success_steps <= 0:
                    break
                continue
            if bool(terminated[0]) or bool(truncated[0]) or step + 1 >= limit:
                steps = step + 1
                break
        # final observed state after the last step (dataset frame convention: state k precedes action k)
        qs.append(robot.data.joint_pos[0, order].cpu().numpy().astype(np.float32))
        objs.append((obj.data.root_state_w[0, :7] - torch.cat([origin, origin.new_zeros(4)])).cpu().numpy())
        Q = np.stack(qs).astype(np.float64)
        O7 = np.stack(objs).astype(np.float64)
        q_ref = codec.q_from_state(hand, Q)
        W = canonical_world(fk, q_ref, rootT)
        O = np.stack([make_T(x[:3], x[3:]) for x in O7])
        seg = segment(O, W, np.zeros((len(O), 8), np.float32)) if len(O) > 16 else {"ok": False, "t_lift": -1}
        rec = {"episode": index, "source_episode": ep.get("episode_index"), "success": succeeded, "steps": steps,
               "restore_error": restore_error, "root_pos_err": root_err, "wall_s": round(time.monotonic() - wall, 1),
               "codec_ms_per_step": round(1000 * t_codec / max(1, len(vecs)), 3), "trace_len": len(O), "decode_fail_steps": decode_fail,
               "t_lift": seg.get("t_lift", -1), "t_g": seg.get("t_g", -1), "seg_ok": bool(seg.get("ok", False))}
        if seg.get("t_g", -1) is not None and seg.get("t_g", -1) >= 0:
            tg = seg["t_g"]
            Wk = inv(O[tg:tg + 1]) @ W[tg:tg + 1]
            rec["wrist_k_tg"] = np.concatenate([Wk[0, :3, 3], rot6d_from_mat(Wk[:, :3, :3])[0]]).round(6).tolist()
            Wl = inv(O[seg["t_lift"]:seg["t_lift"] + 1]) @ W[seg["t_lift"]:seg["t_lift"] + 1]
            rec["wrist_k_tlift"] = np.concatenate([Wl[0, :3, 3], rot6d_from_mat(Wl[:, :3, :3])[0]]).round(6).tolist()
        if solve_err:
            se = np.asarray(solve_err, np.float64)
            rec["wrist_solve_err_mm_max"] = float(np.nanmax(se[:, 0]))
            rec["wrist_solve_err_deg_max"] = float(np.nanmax(se[:, 1]))
        np.savez_compressed(output_dir / "traces" / f"ep{index:03d}.npz", q=Q.astype(np.float32), obj=O7.astype(np.float32),
                            root=root7.astype(np.float32), vec=np.stack(vecs).astype(np.float32),
                            state_names=np.array(state_names), success_step=np.int32(steps if succeeded else -1))
        results.append(rec)
        print(f"[eval] {done + 1}/{len(indices)} episode={index} restore_error={restore_error:.2e} "
              f"success={'yes' if succeeded else 'no'} steps={steps} t_g={rec['t_g']} wall={rec['wall_s']}s "
              f"codec_ms={rec['codec_ms_per_step']} running_success={sum(r['success'] for r in results)}/{len(results)}",
              flush=True)
        write_metrics()
    write_metrics()
    if conn is not None:
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
