"""Physics replay of native and decoded physical joint targets on work1.

Uses the same recorded scene reset and success-term hold as the existing
cross-hand replay harness. Live joint names, scales and offsets are inspected
at runtime. This script never loads a VLA/policy or trains any parameters.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--hand", required=True)
parser.add_argument("--robot-mode", choices=("right","bimanual"), default="right")
parser.add_argument("--sources", required=True)
parser.add_argument("--decoded", required=True)
parser.add_argument("--data", default="data/replay")
parser.add_argument("--output", required=True)
parser.add_argument("--native-reference", default="artifacts/native_replay")
parser.add_argument("--native-only", action="store_true")
parser.add_argument("--contact-trace", action="store_true", help="Separate diagnostic traces; no changes to actions or success criterion")
parser.add_argument("--scene-trace",action="store_true",help="Task-independent full scene/EE trace without changing physics settings")
parser.add_argument("--diag-trace",action="store_true",help="diag_ceiling 261111: scene trace + per-object hand-link contact forces (reporting only; default off)")
parser.add_argument("--native-episodes",type=int,help="Limit ancillary native-health episodes independently of main conditions")
parser.add_argument('--resolve-scene-entities',action='store_true',help='Resolve configured success/failure body and joint names before manual evaluation')
parser.add_argument("--diagnostic-manifest", help="Explicit same-hand frozen-finger wrist interventions, separate from standard codec replay")
parser.add_argument("--no-auto-reset", action="store_true", help="Disable automatic terminations; stop manually at the original non-success terminal condition")
parser.add_argument("--object-material",choices=("as_recorded","configured"),default="as_recorded",
                    help="friction fix 261111: env.sim.reset() rebuilds PhysX and drops DexVerse's startup physics-material "
                         "events (object friction falls to 0.5; record_demos.py does the same, so demos were recorded that way). "
                         "'configured' re-applies them after every sim.reset(); default keeps the recorded-demo condition.")
parser.add_argument("--episode-indices", help="Comma-separated recorded episode indices, for controlled diagnostic repairs")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--hold-steps", type=int, default=120)
parser.add_argument("--success-steps", type=int, default=10)
parser.add_argument("--leap-asset", default="assets/floating_leap_right/floating_leap_hand.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
episode_indices=[int(e) for e in args.episode_indices.split(',')] if args.episode_indices else list(range(args.episodes))
if len(set(episode_indices))!=len(episode_indices) or any(e<0 or e>=args.episodes for e in episode_indices):
    raise ValueError('Invalid episode selection')
if args.contact_trace and args.scene_trace:raise ValueError('Choose one trace mode')
if args.diag_trace and (args.contact_trace or args.scene_trace):raise ValueError('Choose one trace mode')
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import dexverse.tasks  # noqa: F401,E402
import isaaclab_tasks  # noqa: F401,E402
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from dexverse.scene_state import clear_joint_velocity_targets  # noqa: E402
from dexverse.teleop_utils.episode_task_state import reset_managers_after_state_restore  # noqa: E402
from dexverse.teleop_utils.replay_teleop import recorded_wrist_camera_body_collision, set_wrist_camera_body_collision  # noqa: E402
from isaaclab.managers import ManagerTermBase  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

from .data import load_trajectory, trajectory_path  # noqa: E402


def to_torch(value, device):
    if isinstance(value, dict):
        return {k: to_torch(v, device) for k, v in value.items()}
    return torch.as_tensor(value, device=device) if isinstance(value, np.ndarray) else value


def main():
    robot_type = f"floating_{args.hand}_{args.robot_mode}"
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if getattr(cfg, "robot_type", None) != robot_type:
        cfg = type(cfg)(robot_type=robot_type)
        cfg.sim.device = args.device
        cfg.scene.num_envs = 1
    cfg.env_name = args.task
    os.environ["DEXVERSE_EPISODE_SEED"] = "0"
    from dexverse.data_collection.finger_drives import apply_env_finger_drive
    apply_env_finger_drive(cfg)
    if args.contact_trace:
        # DexVerse adds the rigid-body API after its inner USD spawner returns.
        # Enable reporting after that custom spawner, rather than inside it.
        from .contact_trace import contact_reporting_spawner
        cfg.scene.object.spawn.func = contact_reporting_spawner(cfg.scene.object.spawn.func)
        cfg.scene.lazy_sensor_update = False
    diag_objects = []
    if args.diag_trace:
        from .diag_trace import wrap_spawners
        diag_objects = wrap_spawners(cfg)
    if args.robot_mode == "bimanual":
        assets = {"allegro":"floating_allegro_hand_bimanual/floating_allegro_hand_bimanual.usd",
                  "leap":"bimanual_leap/bimanual_leap_hand.usd","shadow":"floating_shadow_bimanual/floating_shadow_bimanual.usd",
                  "wuji":"floating_wuji_hand_bimanual/floating_wuji_hand_bimanual.usd",
                  "sharpa":"floating_sharpa_wave_bimanual/dual_sharpa_wave/dual_sharpa_wave.usd"}
        asset = Path("assets")/assets[args.hand]
        if not asset.is_file():
            raise FileNotFoundError(asset)
        cfg.scene.robot.spawn.usd_path = str(asset.resolve())
    elif args.hand == "leap":
        asset = Path(args.leap_asset).resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        cfg.scene.robot.spawn.usd_path = str(asset)
    recorded = load_trajectory(trajectory_path(args.data, args.task, args.hand, 0))
    set_wrist_camera_body_collision(cfg, recorded_wrist_camera_body_collision(recorded, robot_type))
    success_term = cfg.terminations.success
    manual_failure_terms=[]
    if args.no_auto_reset:
        for name in dir(cfg.terminations):
            if name.startswith('_') or name in ('success','time_out'):continue
            term=getattr(cfg.terminations,name)
            if term is not None and hasattr(term,'func'):
                manual_failure_terms.append((name,term))
                setattr(cfg.terminations,name,None)
    cfg.terminations.success = cfg.terminations.time_out = None
    cfg.recorders = {}
    cfg.observations.policy.concatenate_terms = False
    for name in ("third_person_camera", "wrist_camera", "right_wrist_camera", "left_wrist_camera"):
        camera = getattr(cfg.scene, name, None)
        if camera is not None:
            camera.height = camera.width = 64
            camera.data_types = ["rgb"]
    for name in ("depth", "pointcloud"):
        if getattr(cfg.observations, name, None) is not None:
            setattr(cfg.observations, name, None)
    env = gym.make(args.task, cfg=cfg).unwrapped
    if args.resolve_scene_entities:
        def resolve_entities(value):
            if isinstance(value,SceneEntityCfg):value.resolve(env.scene)
            elif isinstance(value,dict):
                for v in value.values():resolve_entities(v)
            elif isinstance(value,(tuple,list)):
                for v in value:resolve_entities(v)
        resolve_entities(success_term.params)
        for _,term in manual_failure_terms:resolve_entities(term.params)
    func = success_term.func
    if isinstance(func, type) and issubclass(func, ManagerTermBase):
        func = func(success_term, env)
    failure_funcs=[]
    for name,term in manual_failure_terms:
        failure_func=term.func
        if isinstance(failure_func,type) and issubclass(failure_func,ManagerTermBase):
            failure_func=failure_func(term,env)
        failure_funcs.append((name,failure_func,term.params))
    robot, manager = env.scene["robot"], env.action_manager
    startup_terms = list(env.event_manager.active_terms.get("startup", []))
    material_terms = [n for n in startup_terms if "material" in n]
    if args.object_material == "configured" and not material_terms:
        raise ValueError("--object-material configured but the env has no startup material event")
    def live_materials():
        out = {}
        for name, obj in env.scene.rigid_objects.items():
            m = obj.root_physx_view.get_material_properties()[0].detach().cpu().numpy()
            out[name] = np.unique(np.round(m, 4), axis=0).tolist()
        return out
    material_after_init = live_materials()
    def reapply_startup_materials():
        for n in material_terms:
            term_cfg = env.event_manager.get_term_cfg(n)
            term_cfg.func(env, None, **term_cfg.params)
    tracer = None
    if args.contact_trace:
        from .contact_trace import ContactTrace
        tracer = ContactTrace(env, args.output, func, success_term.params)
    elif args.scene_trace:
        from .scene_trace import SceneTrace
        tracer=SceneTrace(env,args.output,args.hand,func,success_term.params)
    elif args.diag_trace:
        from .diag_trace import DiagTrace
        tracer=DiagTrace(env,args.output,args.hand,func,success_term.params,diag_objects)
    fk_checked = False
    action_names, offsets, scales = [], [], []
    for term_name in manager.active_terms:
        term = manager.get_term(term_name)
        names = list(term._joint_names)
        action_names.extend(names)
        for output, attribute, default in ((offsets, "_offset", 0.), (scales, "_scale", 1.)):
            value = getattr(term, attribute, default)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy().reshape(-1)
                output.extend(value[:len(names)].tolist() if len(value) >= len(names) else [float(value[0])] * len(names))
            else:
                output.extend([float(value)] * len(names))
    offsets, scales = np.array(offsets), np.array(scales)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    live = {"task": args.task, "hand": args.hand, "joint_names": list(robot.joint_names),
            "action_joint_names": action_names, "action_offset": offsets.tolist(), "action_scale": scales.tolist(),
            "joint_limits": robot.data.joint_pos_limits[0].cpu().numpy().tolist(), "step_dt": env.step_dt}
    live.update({"object_material_mode": args.object_material, "startup_event_terms": startup_terms,
                 "material_terms": material_terms, "material_after_env_init": material_after_init})
    output.with_suffix(".runtime.json").write_text(json.dumps(live, indent=2))
    reference = Path(args.native_reference) / f"{args.task}__{args.hand}.jsonl"
    reference.parent.mkdir(parents=True, exist_ok=True)
    native_file = Path(args.decoded) / args.task / args.hand / f"{args.hand}_native.npz"
    with np.load(native_file, allow_pickle=False) as native_data:
        digest = hashlib.sha256()
        for key in sorted(native_data.files):
            digest.update(key.encode())
            digest.update(native_data[key].tobytes())
        native_initials = {key:native_data[key].copy() for key in native_data.files if key.startswith(("init_","root_"))}
    profile = {"runtime": live, "native_targets_sha256": digest.hexdigest(),
               "hold_steps": args.hold_steps, "success_steps": args.success_steps}
    if args.object_material != "as_recorded":
        profile["object_material"] = args.object_material
    if args.no_auto_reset:
        profile['termination_control']='all automatic terms disabled; original non-success terms stop replay manually'
    if args.resolve_scene_entities:profile['scene_entity_resolution']='explicit_task_body_and_joint_selection_v1'
    profile_path = reference.with_suffix(".profile.json")
    if profile_path.exists() and json.loads(profile_path.read_text()) != profile:
        raise ValueError("Shared native reference protocol differs")
    profile_path.write_text(json.dumps(profile, indent=2))
    native_rows = {json.loads(line)["episode"]: json.loads(line) for line in reference.read_text().splitlines()} if reference.exists() else {}
    existing = set()
    if output.is_file():
        for line in output.read_text().splitlines():
            r = json.loads(line)
            existing.add((r["source"], r["variant"], r["episode"]))
    diagnostic_jobs = {}
    if args.diagnostic_manifest:
        manifest = json.loads(Path(args.diagnostic_manifest).read_text())
        if manifest['task'] != args.task or manifest.get('replay_hand',manifest['source']) != args.hand or args.robot_mode != 'right':
            raise ValueError('Diagnostic manifest task/hand mismatch')
        diagnostic_jobs = {j['condition']: j for j in manifest['jobs']}
        if len(diagnostic_jobs) != len(manifest['jobs']) or any(not k.replace('_','').isalnum() or k in ('native','codec') for k in diagnostic_jobs):
            raise ValueError('Diagnostic condition names must be distinct')
        jobs = [(args.hand, 'native')] + [(manifest['source'], k) for k in diagnostic_jobs]
    else:
        jobs = [(args.hand, "native")] + ([] if args.native_only else [(s, "codec") for s in args.sources.split(",")])
    rows = []
    for source, variant in jobs:
        diagnostic = diagnostic_jobs.get(variant)
        path = Path(diagnostic['file']) if diagnostic else Path(args.decoded) / args.task / args.hand / f"{source}_{variant}.npz"
        decoded = np.load(path, allow_pickle=False)
        finger_reference = np.load(diagnostic['finger_reference'], allow_pickle=False) if diagnostic else None
        names = list(decoded["joint_names"])
        if set(names) != set(robot.joint_names):
            raise ValueError(f"Decoded/live joints differ: {set(names) ^ set(robot.joint_names)}")
        to_actions = [names.index(n) for n in action_names]
        to_live = [names.index(n) for n in robot.joint_names]
        selected=episode_indices[:args.native_episodes] if variant=='native' and args.native_episodes is not None else episode_indices
        for episode in selected:
            if (source, variant, episode) in existing:
                continue
            if variant == "native" and episode in native_rows:
                row = {**native_rows[episode], "reused_native": True, "native_reference": str(reference)}
                with output.open("a") as stream:
                    stream.write(json.dumps(row)+"\n")
                continue
            key = f"ep{episode:03d}"
            q_targets = decoded[key]
            if diagnostic:
                wrist_names = set(manifest['wrist_joints'])
                finger_indices = [i for i,n in enumerate(names) if n not in wrist_names]
                ref_order = [list(finger_reference['joint_names']).index(names[i]) for i in finger_indices]
                if not np.array_equal(q_targets[:,finger_indices], finger_reference[key][:,ref_order]):
                    raise ValueError('Diagnostic changed frozen finger commands')
                if not np.array_equal(decoded[f'init_{key}'][finger_indices],finger_reference[f'init_{key}'][ref_order]):
                    raise ValueError('Diagnostic changed finger initial state')
                if not np.array_equal(decoded[f'root_{key}'],native_initials[f'root_{key}']):
                    raise ValueError('Diagnostic changed articulation root')
            actions = ((q_targets[:, to_actions] - offsets) / scales).astype(np.float32)
            initial = decoded[f"init_{key}"][to_live][None]
            if source == args.hand and variant == "codec":
                if not np.array_equal(decoded[f"init_{key}"],native_initials[f"init_{key}"]) or not np.array_equal(decoded[f"root_{key}"],native_initials[f"root_{key}"]):
                    raise ValueError("Same-hand replay must preserve the native initial state")
            source_episode = load_trajectory(trajectory_path(args.data, args.task, source, episode))["episodes"][0]
            from .scene_trace import state_digest
            scene_only={k:{n:v for n,v in values.items() if not(k=='articulation' and n=='robot')} for k,values in source_episode['initial_state'].items()}
            scene_hash=state_digest(scene_only)
            state = copy.deepcopy(source_episode["initial_state"])
            state["articulation"]["robot"] = {"root_pose": decoded[f"root_{key}"][None],
                                                "root_velocity": np.zeros((1, 6), dtype=np.float32),
                                                "joint_position": initial.astype(np.float32),
                                                "joint_velocity": np.zeros_like(initial, dtype=np.float32)}
            env.sim.reset()
            if args.object_material == "configured":
                reapply_startup_materials()
            env.reset()
            env.reset_to(to_torch(state, env.device), env_ids=None, is_relative=True)
            clear_joint_velocity_targets(env.scene)
            reset_managers_after_state_restore(env)
            episode_materials = live_materials()
            recorded_states = source_episode.get("states") if (variant == "native" and source == args.hand) else None
            object_errors = []
            if tracer:
                tracer.begin(source, variant, episode, q_targets, names)
            count, success, wall = 0, False, time.monotonic()
            non_success_termination=None
            tracking_sum, tracking_n = 0., 0
            for step in range(len(actions) + args.hold_steps):
                index = min(step, len(actions) - 1)
                step_output=env.step(torch.tensor(actions[index], device=env.device)[None])
                if args.no_auto_reset and (bool(step_output[2][0]) or bool(step_output[3][0])):
                    raise ValueError('Unexpected automatic termination while all terms should be disabled')
                if not fk_checked:
                    # Required gate for the independently extracted bimanual
                    # assets. Validate exact native FK, not mirror similarity.
                    from .reference_kin import HandKin, make_T
                    label = args.hand+("_bimanual" if args.robot_mode == "bimanual" else "")
                    kin = HandKin(label,list(robot.joint_names),f"metadata/raw_kin/{label}.json")
                    bodies = [n for n in robot.body_names if n in kin.order]
                    if len(bodies) < 6:
                        raise ValueError("Too few native bodies for FK validation")
                    pose = robot.data.root_state_w[0,:7].cpu().numpy()
                    actual_q = robot.data.joint_pos.cpu().numpy()
                    expected = kin.fk(actual_q,root_T=make_T(pose[:3],pose[3:]),bodies=bodies)
                    errors = [np.linalg.norm(expected[n][0,:3,3]-robot.data.body_pos_w[0,list(robot.body_names).index(n)].cpu().numpy()) for n in bodies]
                    check = {"robot_mode":args.robot_mode,"bodies_checked":len(bodies),"max_native_body_error_mm":float(max(errors)*1000)}
                    output.with_suffix(".fk_validation.json").write_text(json.dumps(check,indent=2))
                    if max(errors) > .001:
                        raise ValueError(f"Live native FK mismatch: {check}")
                    fk_checked = True
                if recorded_states is not None and step + 1 < len(recorded_states) and "object" in env.scene.rigid_objects:
                    recorded_pos = np.asarray(recorded_states[step + 1]["rigid_object"]["object"]["root_pose"])[0, :3]
                    live_pos = (env.scene["object"].data.root_pos_w[0] - env.scene.env_origins[0]).detach().cpu().numpy()
                    object_errors.append(float(np.linalg.norm(live_pos - recorded_pos)) * 1000)
                error = robot.data.joint_pos[0].detach().cpu().numpy() - q_targets[index, to_live]
                tracking_sum += float(np.square(error).sum())
                tracking_n += error.size
                ok = bool(func(env, **success_term.params)[0])
                if tracer:
                    tracer.record(step, index, ok)
                for name,failure_func,params in failure_funcs:
                    if bool(failure_func(env,**params)[0]):
                        non_success_termination=name
                        break
                if non_success_termination:
                    break
                count = count + 1 if ok else 0
                if count >= args.success_steps:
                    success = True
                    break
            row = {"task": args.task, "source": source, "target": args.hand, "variant": variant,
                   "episode": episode, "success": success, "steps": step + 1,
                   "joint_tracking_rmse": (tracking_sum / tracking_n)**.5,
                   "wall_s": time.monotonic() - wall}
            if args.no_auto_reset:
                row['auto_reset_disabled']=True
                row['non_success_termination']=non_success_termination
            row["initial_state_protocol"] = "native" if source == args.hand else "mapped_cross_hand"
            row["object_material_mode"] = args.object_material
            row["object_material_live"] = episode_materials.get("object")
            if object_errors:
                row["object_vs_recorded_mm"] = {"mean": float(np.mean(object_errors)), "max": float(np.max(object_errors)),
                                                "final": object_errors[-1], "n": len(object_errors)}
            row['replay_hand']=args.hand
            row['scene_initial_sha256']=scene_hash
            if diagnostic:
                row['initial_state_protocol'] = diagnostic.get('initial_state_protocol','same_source_scene_and_finger_initial_state; only wrist initial position/commands intervened')
                row['wrist_intervention'] = {k:v for k,v in diagnostic.items() if k not in ('file','finger_reference')}
            if tracer:
                row['state_diagnostics' if (args.scene_trace or args.diag_trace) else 'contact_diagnostics'] = tracer.finish(success)
            with output.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if variant == "native":
                with reference.open("a") as stream:
                    stream.write(json.dumps(row)+"\n")
            print("[latent-replay] " + json.dumps(row), flush=True)
            rows.append(row)
    output.with_suffix(".done.json").write_text(json.dumps({"completed": True, "new_rows": len(rows)}))
    print("[latent-replay] DONE", flush=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
