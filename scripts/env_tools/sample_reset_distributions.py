# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sample a task's reset distribution and dump it for the reset audit.

Instantiates one Dexverse task headlessly, resets it ``--num_samples`` times and
records, for every reset, the env-relative pose of every scene entity, the goal
command(s), the robot palm position, and what happens during a short hold
(``--settle_steps`` zero/hold actions): which termination terms fire and how far
things move. Results go to ``<out_dir>/<task>/{samples.npz,meta.json}`` and are
immediately analysed into ``stats.json`` + ``<task>.png`` by
:mod:`dexverse.analysis.reset_audit` (which is simulator-free, so re-plotting
later does not need Isaac).

Typical use (headless server; DISPLAY must not leak into Isaac):

    env -u DISPLAY CUDA_VISIBLE_DEVICES=5 python scripts/env_tools/sample_reset_distributions.py \\
        --task Dexverse-PushT-v0 --num_samples 500 --headless

Notes on the sampling protocol (see the plan / docstrings for the evidence):

* ``--num_envs`` defaults to 1. World-frame goal commands (``use_world_frame``)
  are sampled in absolute world coordinates, so with several envs only the env at
  the origin gets an on-table goal; running one env keeps env-relative == world.
  ``--num_envs N>1`` is available as a cross-check and is reported as such.
* Command metrics are zeroed by ``CommandTerm.reset()`` and only refreshed at the
  *end* of ``env.step()``, after terminations were evaluated. The main loop calls
  ``command_manager.compute(dt=0.0)`` right after each reset so goal-distance
  successes are evaluated against the real goal; a short raw *probe* phase records
  how the env behaves without that fix (``probe/success_step1``).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sample reset distributions of a Dexverse task.")
parser.add_argument("--task", type=str, required=True, help="Gym id, e.g. Dexverse-PushT-v0.")
parser.add_argument("--num_samples", type=int, default=100, help="Number of reset samples to record.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (default 1; >1 only as a cross-check).")
parser.add_argument("--settle_steps", type=int, default=30, help="Hold-action steps after each reset.")
parser.add_argument("--probe_resets", type=int, default=8, help="Raw resets (no metric refresh) probing step-1 success; runs after the main loop.")
parser.add_argument("--probe_steps", type=int, default=15, help="Hold steps per probe reset (>1 so PhysX is never reset right after a single step).")
parser.add_argument("--seed", type=int, default=0, help="Seed for the env / sampling RNG.")
parser.add_argument("--out_dir", type=str, default="outputs/reset_audit", help="Output root (per-task subdir is created).")
parser.add_argument("--robot_type", type=str, default=None, help="Optional robot variant override.")
parser.add_argument("--json_path", type=str, default=None, help="Template JSON spec for *Template environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--no_plot", action="store_true", default=False, help="Skip stats/plot after sampling.")
parser.add_argument("--debug_done", type=int, default=0, help="Print entity poses for the first N terminations seen during the hold.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # this tool never needs a window

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Line-buffer stdout so progress survives redirection to a log file.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass

# Make sure *this* checkout's dexverse is the one imported, regardless of which
# editable install the interpreter carries (several conda envs on the box point
# at other checkouts).
_REPO_SRC = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "dexverse"))
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import dexverse  # noqa: E402
import dexverse.tasks  # noqa: F401, E402
import isaaclab_tasks  # noqa: F401, E402
from dexverse.analysis.reset_audit import analyze_task_dir  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from dexverse.tasks.utils.scene_cleanup import prune_stale_obs_refs, strip_camera_cfgs  # noqa: E402
from dexverse.teleop_utils.range_vis import collect_range_boxes  # noqa: E402
from isaaclab.envs.mdp.actions import JointPositionAction, RelativeJointPositionAction  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

_loaded = os.path.realpath(os.path.dirname(os.path.dirname(dexverse.__file__)))
if _loaded != _REPO_SRC:
    raise RuntimeError(f"dexverse resolved to {_loaded}, expected {_REPO_SRC}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


def _numeric_params(params: dict) -> dict:
    """JSON-safe subset of a manager term's params (numbers, strings, small lists, entity names)."""
    out = {}
    for k, v in (params or {}).items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float, str)):
            out[k] = v
        elif isinstance(v, SceneEntityCfg):
            out[k] = {"entity": v.name, "body_names": v.body_names, "joint_names": v.joint_names}
        elif isinstance(v, (list, tuple)) and len(v) <= 12 and all(isinstance(x, (int, float)) for x in v):
            out[k] = list(v)
        elif isinstance(v, dict) and all(isinstance(x, (int, float, list, tuple)) for x in v.values()):
            out[k] = {kk: (list(vv) if isinstance(vv, (list, tuple)) else vv) for kk, vv in v.items()}
    return out


def _term_name(func) -> str:
    return getattr(func, "__name__", type(func).__name__)


def _reset_event_meta(env) -> list[dict]:
    """Ordered list of reset events with the bits the audit cares about."""
    events = []
    active = getattr(env.event_manager, "active_terms", {})
    reset_names = active.get("reset", []) if isinstance(active, dict) else []
    for name in reset_names:
        term = getattr(env.cfg.events, name, None)
        if term is None:
            continue
        params = term.params or {}
        entity = None
        for key in ("asset_cfg", "object_cfg", "target_cfg"):
            v = params.get(key)
            if isinstance(v, SceneEntityCfg):
                entity = v.name
                break
        ref = params.get("reference_asset_cfg")
        events.append(
            {
                "name": name,
                "func": _term_name(term.func),
                "entity": entity,
                "reference_entity": ref.name if isinstance(ref, SceneEntityCfg) else None,
                "min_xy_distance": params.get("min_xy_distance"),
                "pose_range": {k: list(v) for k, v in params["pose_range"].items()} if isinstance(params.get("pose_range"), dict) else None,
            }
        )
    return events


def _termination_meta(env) -> dict:
    out = {}
    for name in env.termination_manager.active_terms:
        term = getattr(env.cfg.terminations, name, None)
        if term is None:
            continue
        out[name] = {"func": _term_name(term.func), "params": _numeric_params(term.params), "time_out": bool(getattr(term, "time_out", False))}
    return out


def _command_meta(env) -> dict:
    out = {}
    for name in env.command_manager.active_terms:
        term = env.command_manager.get_term(name)
        cfg = term.cfg
        ranges = getattr(cfg, "ranges", None)
        out[name] = {
            "class": type(term).__name__,
            "use_world_frame": bool(getattr(cfg, "use_world_frame", False)),
            "position_only": bool(getattr(cfg, "position_only", False)),
            "asset_name": getattr(cfg, "asset_name", None),
            "object_name": getattr(cfg, "object_name", None),
            "ranges": {k: list(getattr(ranges, k)) for k in ("pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw") if ranges is not None and getattr(ranges, k, None) is not None},
        }
    return out


def _palm_bodies(env) -> dict[str, int]:
    """``{label: body_index}`` for the robot palm body / bodies."""
    robot = env.scene["robot"]
    rc = getattr(env.cfg, "robot_config", None)
    names = []
    if rc is not None:
        if getattr(rc, "left_palm_body_name", None) and getattr(rc, "right_palm_body_name", None):
            names = [("right", rc.right_palm_body_name), ("left", rc.left_palm_body_name)]
        elif getattr(rc, "palm_body_name", None):
            names = [("palm", rc.palm_body_name)]
    out = {}
    for label, body in names:
        try:
            ids, _ = robot.find_bodies(body)
        except Exception:  # noqa: BLE001
            ids = []
        if ids:
            out[label] = int(ids[0])
    return out


def _hold_action(env) -> torch.Tensor:
    """Action that keeps the robot where it is.

    Zero is a hold for ``JointPositionAction`` with ``use_default_offset`` (target =
    default joint pos = reset pose) and for ``RelativeJointPositionAction``. Any
    other term gets its slice filled from the current joint positions (absolute
    targets) so we never inject motion.
    """
    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    offset = 0
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        dim = term.action_dim
        if isinstance(term, RelativeJointPositionAction) or (
            isinstance(term, JointPositionAction) and getattr(term.cfg, "use_default_offset", True)
        ):
            pass  # zeros hold
        elif isinstance(term, JointPositionAction):
            joint_pos = term._asset.data.joint_pos[:, term._joint_ids]  # noqa: SLF001
            scale = term._scale  # noqa: SLF001
            action[:, offset : offset + dim] = (joint_pos - term._offset) / scale  # noqa: SLF001
            print(f"[reset_audit] action term '{name}' is absolute without default offset; holding current joint pos")
        else:
            print(f"[reset_audit] WARNING: action term '{name}' ({type(term).__name__}) -- zero may not be a hold")
        offset += dim
    return action


def _snapshot(env, palm: dict[str, int]) -> dict[str, np.ndarray]:
    """Env-relative poses of everything, per env (arrays of shape (num_envs, ...))."""
    state = env.scene.get_state(is_relative=True)
    out: dict[str, np.ndarray] = {}
    for kind in ("rigid_object", "articulation"):
        for name, entry in state.get(kind, {}).items():
            out[f"{kind}/{name}/root_pose"] = _np(entry["root_pose"])
            if kind == "articulation" and "joint_position" in entry:
                out[f"{kind}/{name}/joint_pos"] = _np(entry["joint_position"])
    origins = _np(env.scene.env_origins)
    for name in env.command_manager.active_terms:
        term = env.command_manager.get_term(name)
        pw = getattr(term, "pose_command_w", None)
        if pw is not None:
            pw = _np(pw)
            pw[:, :3] -= origins
            out[f"commands/{name}/pose_w"] = pw
        out[f"commands/{name}/command"] = _np(term.command)
    body_pos = _np(env.scene["robot"].data.body_pos_w)  # (num_envs, num_bodies, 3)
    for label, idx in palm.items():
        out[f"palm/{label}/pos"] = body_pos[:, idx, :] - origins
    # every robot body (palm, fingers, ...) so the audit can measure hand<->object clearance
    out["robot/body_pos"] = body_pos - origins[:, None, :]
    return out


def _stack(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = rows[0].keys()
    return {k: np.concatenate([r[k] for r in rows], axis=0) for k in keys}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        json_path=args_cli.json_path,
    )
    if args_cli.robot_type is not None:
        if not hasattr(env_cfg, "robot_type"):
            raise ValueError(f"Task '{args_cli.task}' does not expose robot_type.")
        cfg_cls = type(env_cfg)
        env_cfg = cfg_cls(robot_type=args_cli.robot_type)
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.sim.use_fabric = not args_cli.disable_fabric
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # State-only observations, no cameras (no --enable_cameras needed), no recorders.
    if hasattr(env_cfg, "_apply_observation_preset"):
        try:
            env_cfg._apply_observation_preset("state")  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            print(f"[reset_audit] observation preset 'state' not applied: {exc}")
    strip_camera_cfgs(env_cfg)
    prune_stale_obs_refs(env_cfg)
    env_cfg.recorders = {}
    # keep terminations (we read them); make sure time_out never trips during the hold
    if getattr(env_cfg.terminations, "time_out", None) is not None:
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 60.0)

    env = gym.make(args_cli.task, cfg=env_cfg, disable_env_checker=True).unwrapped
    print(f"[reset_audit] env created in {time.time() - t0:.1f}s; num_envs={env.num_envs}, device={env.device}")

    with torch.inference_mode():
        samples, meta = _sample(env, env_cfg, t0)

    out_dir = Path(args_cli.out_dir) / args_cli.task
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "samples.npz", **samples)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[reset_audit] wrote {out_dir / 'samples.npz'} ({meta['num_samples']} samples) and meta.json")

    env.close()

    if not args_cli.no_plot:
        stats = analyze_task_dir(out_dir)
        flags = ", ".join(f["flag"] for f in stats.get("flags", [])) or "none"
        print(f"[reset_audit] flags: {flags}")
        print(f"[reset_audit] figure: {out_dir / stats.get('png', '')}")


def _sample(env, env_cfg, t0: float) -> tuple[dict, dict]:
    """Run probe + main sampling loop; returns (samples, meta). Call under inference_mode."""
    palm = _palm_bodies(env)
    hold = _hold_action(env)
    term_names = list(env.termination_manager.active_terms)
    origins_np = _np(env.scene.env_origins)

    # ---------------- main sampling loop -------------------------------------------
    env.reset(seed=args_cli.seed)
    rounds = int(math.ceil(args_cli.num_samples / env.num_envs))
    reset_rows, settled_rows = [], []
    first_step = {name: [] for name in term_names}
    done_step_rows, origin_rows, env_index_rows = [], [], []
    range_boxes = None
    debug_left = [int(args_cli.debug_done)]
    for r in range(rounds):
        env.reset()
        # refresh command metrics / world goal for the just-sampled commands (see module docstring)
        if env.command_manager.active_terms:
            env.command_manager.compute(dt=0.0)
        if range_boxes is None:
            try:
                boxes = collect_range_boxes(env, env_index=0)
                o0 = origins_np[0]
                range_boxes = [
                    {
                        "name": b.name,
                        "kind": b.kind,
                        "entity": b.entity,
                        "center": [float(b.center[i] - o0[i]) for i in range(3)],
                        "half_extent": [float(v) for v in b.half_extent],
                        "rot_range": {k: [float(v[0]), float(v[1])] for k, v in b.rot_range.items()},
                        "degenerate": bool(b.degenerate),
                    }
                    for b in boxes
                ]
            except Exception as exc:  # noqa: BLE001
                print(f"[reset_audit] collect_range_boxes failed: {exc}")
                range_boxes = []
        snap0 = _snapshot(env, palm)
        n = env.num_envs
        fired = {name: np.full((n,), -1, dtype=np.int64) for name in term_names}
        done_at = np.full((n,), -1, dtype=np.int64)
        for k in range(1, args_cli.settle_steps + 1):
            _, _, terminated, truncated, _ = env.step(hold)
            done = _np(terminated | truncated).astype(bool)
            for name in term_names:
                flag = _np(env.termination_manager.get_term(name)).astype(bool)
                newly = flag & (fired[name] < 0) & (done_at < 0)
                fired[name][newly] = k
            newly_done = done & (done_at < 0)
            done_at[newly_done] = k
            if args_cli.debug_done > 0 and newly_done.any() and debug_left[0] > 0:
                debug_left[0] -= 1
                which = [name for name in term_names if fired[name][newly_done].max() == k]
                snap = _snapshot(env, palm)
                for key, arr in snap.items():
                    if key.endswith("/root_pose") or key.startswith("palm/") or key.endswith("/pose_w"):
                        print(f"[reset_audit][debug] step {k} terms={which} {key} = {np.round(arr[newly_done][0][:3], 3).tolist()}")
        snap1 = _snapshot(env, palm)
        # envs that terminated were auto-reset inside step(): their settled state is a new sample -> mask
        masked = done_at >= 0
        for key, arr in snap1.items():
            if masked.any():
                arr = arr.copy()
                arr[masked] = np.nan
            snap1[key] = arr
        reset_rows.append(snap0)
        settled_rows.append({"settled/" + k: v for k, v in snap1.items()})
        for name in term_names:
            first_step[name].append(fired[name])
        done_step_rows.append(done_at)
        origin_rows.append(origins_np.copy())
        env_index_rows.append(np.arange(n))
        if (r + 1) % max(1, rounds // 10) == 0 or r == rounds - 1:
            print(f"[reset_audit] round {r + 1}/{rounds} ({(r + 1) * n} samples), {time.time() - t0:.0f}s")

    # ---------------- probe: raw env behaviour right after reset (no metric refresh) ---
    # Runs AFTER the main loop and holds for ``--probe_steps`` per reset: resetting
    # PhysX again after a single step, a dozen times in a row, was observed to make
    # some assets (bleach bottle, cup) permanently lose collision with the table.
    probe_success, probe_success_any, probe_obj_pose = [], [], []
    obj_key = "rigid_object/object/root_pose" if "object" in env.scene.rigid_objects else None
    if args_cli.probe_resets > 0 and "success" in term_names:
        env.reset(seed=args_cli.seed + 12345)
        n_probe_rounds = int(math.ceil(args_cli.probe_resets / env.num_envs))
        for _ in range(n_probe_rounds):
            env.reset()
            snap = _snapshot(env, palm)
            if obj_key and obj_key in snap:
                probe_obj_pose.append(snap[obj_key])
            any_fired = np.zeros((env.num_envs,), dtype=bool)
            for k in range(1, max(1, args_cli.probe_steps) + 1):
                env.step(hold)
                flag = _np(env.termination_manager.get_term("success")).astype(bool)
                if k == 1:
                    probe_success.append(flag)
                any_fired |= flag
            probe_success_any.append(any_fired)
        probe_success = np.concatenate(probe_success)[: args_cli.probe_resets]
        probe_success_any = np.concatenate(probe_success_any)[: args_cli.probe_resets]
        print(
            f"[reset_audit] probe ({probe_success.size} raw resets x {args_cli.probe_steps} steps): "
            f"success on step 1 in {probe_success.mean() * 100:.0f}%, within the window in {probe_success_any.mean() * 100:.0f}%"
        )
    else:
        probe_success = np.zeros((0,), dtype=bool)
        probe_success_any = np.zeros((0,), dtype=bool)

    samples = _stack(reset_rows)
    samples.update(_stack(settled_rows))
    for name in term_names:
        samples[f"term_first_step/{name}"] = np.concatenate(first_step[name])
    samples["done_step"] = np.concatenate(done_step_rows)
    samples["env_origin"] = np.concatenate(origin_rows)
    samples["env_index"] = np.concatenate(env_index_rows)
    samples = {k: v[: args_cli.num_samples] for k, v in samples.items()}
    samples["probe/success_step1"] = probe_success
    samples["probe/success_any"] = probe_success_any
    if probe_obj_pose:
        samples["probe/object_root_pose"] = np.concatenate(probe_obj_pose)[: args_cli.probe_resets]

    # ---------------- meta ---------------------------------------------------------
    state0 = env.scene.get_state(is_relative=True)
    entities = {kind: sorted(state0.get(kind, {}).keys()) for kind in ("rigid_object", "articulation")}
    default_root, default_joint, joint_names = {}, {}, {}
    for kind in ("rigid_object", "articulation"):
        for name in entities[kind]:
            asset = env.scene[name]
            drs = _np(asset.data.default_root_state)[0]
            default_root[name] = [float(v) for v in drs[:7]]
            if kind == "articulation":
                default_joint[name] = [float(v) for v in _np(asset.data.default_joint_pos)[0]]
                joint_names[name] = list(asset.joint_names)
    table_half = None
    try:
        size = env.cfg.scene.table.spawn.size
        table_half = [float(size[0]) / 2, float(size[1]) / 2]
    except Exception:  # noqa: BLE001
        pass
    meta = {
        "task": args_cli.task,
        "robot_type": getattr(env.cfg, "robot_type", None),
        "seed": args_cli.seed,
        "num_samples": int(min(args_cli.num_samples, rounds * env.num_envs)),
        "num_envs": int(env.num_envs),
        "settle_steps": int(args_cli.settle_steps),
        "probe_resets": int(probe_success.size),
        "probe_steps": int(args_cli.probe_steps),
        "step_dt": float(env.step_dt),
        "episode_length_s": float(env_cfg.episode_length_s),
        "entities": entities,
        "joint_names": joint_names,
        "default_root_state": default_root,
        "default_joint_pos": default_joint,
        "palm_bodies": {k: int(v) for k, v in palm.items()},
        "robot_body_names": list(env.scene["robot"].body_names),
        "table_half_xy": table_half,
        "range_boxes": range_boxes or [],
        "reset_events": _reset_event_meta(env),
        "termination_terms": _termination_meta(env),
        "commands": _command_meta(env),
        "action_terms": {n: type(env.action_manager.get_term(n)).__name__ for n in env.action_manager.active_terms},
        "elapsed_s": float(time.time() - t0),
    }
    return samples, meta


if __name__ == "__main__":
    main()
    simulation_app.close()
