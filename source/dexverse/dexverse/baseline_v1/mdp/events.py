# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event helpers for manager-based tasks (non-termination utilities)."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from .resets import reset_joints_to_init
from .utils import resolve_env_ids, resolve_joint_ids


def reset_board_and_switch_xy(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    board_cfg: SceneEntityCfg,
    switch_cfg: SceneEntityCfg,
    switch_joint_cfg: SceneEntityCfg | None = None,
    xy_range: tuple[float, float] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
):
    """Reset board and switch positions with the same XY offset."""
    board = env.scene[board_cfg.name]
    switch = env.scene[switch_cfg.name]

    env_ids_t = resolve_env_ids(env, env_ids)

    if x_range is None:
        if xy_range is None:
            raise ValueError("reset_board_and_switch_xy requires xy_range or x_range.")
        x_range = xy_range
    if y_range is None:
        if xy_range is None:
            raise ValueError("reset_board_and_switch_xy requires xy_range or y_range.")
        y_range = xy_range

    # sample offsets
    offsets = torch.zeros((env_ids_t.shape[0], 3), device=env.device)
    offsets[:, 0].uniform_(x_range[0], x_range[1])
    offsets[:, 1].uniform_(y_range[0], y_range[1])

    board_state = board.data.default_root_state[env_ids_t].clone()
    switch_state = switch.data.default_root_state[env_ids_t].clone()
    board_state[:, :3] += offsets
    switch_state[:, :3] += offsets

    board.write_root_pose_to_sim(board_state[:, 0:7], env_ids=env_ids_t)
    switch.write_root_pose_to_sim(switch_state[:, 0:7], env_ids=env_ids_t)
    if switch_joint_cfg is not None:
        reset_joints_to_init(env, env_ids_t, switch_joint_cfg)


def reset_book_cluster_and_command(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    left_book_cfg: SceneEntityCfg,
    right_book_cfg: SceneEntityCfg,
    y_range: tuple[float, float],
    target_pos_x: float,
    target_pos_z: float,
    target_pitch_rad: float,
    command_name: str = "object_pose",
):
    """Reset the three-book cluster with one shared Y offset and align the goal pose."""
    env_ids_t = resolve_env_ids(env, env_ids)
    num_envs = env_ids_t.shape[0]
    if num_envs == 0:
        return

    y_offsets = torch.empty(num_envs, device=env.device)
    y_offsets.uniform_(y_range[0], y_range[1])

    target_book = env.scene[object_cfg.name]
    left_book = env.scene[left_book_cfg.name]
    right_book = env.scene[right_book_cfg.name]

    # Target book is dynamic: reset pose and zero velocity.
    target_root_state = target_book.data.default_root_state[env_ids_t].clone()
    target_root_state[:, 1] += y_offsets
    target_root_state[:, 7:13] = 0.0
    target_book.write_root_pose_to_sim(target_root_state[:, 0:7], env_ids=env_ids_t)
    target_book.write_root_velocity_to_sim(target_root_state[:, 7:13], env_ids=env_ids_t)

    # Neighbor books are kinematic in this task: only write pose.
    for asset in (left_book, right_book):
        root_state = asset.data.default_root_state[env_ids_t].clone()
        root_state[:, 1] += y_offsets
        asset.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids_t)

    command_term = env.command_manager.get_term(command_name)
    command_term.pose_command_b[env_ids_t, 0] = target_pos_x
    command_term.pose_command_b[env_ids_t, 1] = y_offsets
    command_term.pose_command_b[env_ids_t, 2] = target_pos_z

    zero = torch.zeros(num_envs, device=env.device)
    quat = quat_from_euler_xyz(zero, torch.full_like(zero, target_pitch_rad), zero)
    command_term.pose_command_b[env_ids_t, 3:] = quat

    if getattr(command_term.cfg, "use_world_frame", False):
        command_term.pose_command_w[env_ids_t, :3] = command_term.pose_command_b[env_ids_t, :3]
        command_term.pose_command_w[env_ids_t, 3:] = quat


def set_joint_position_limits(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lower: float | None = None,
    upper: float | None = None,
):
    """Set joint position limits for an articulation asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)
    env_ids_t = resolve_env_ids(env, env_ids)

    limits = asset.data.default_joint_pos_limits.clone()
    # Outer-product indexing: (N, 1) x (1, J) -> every selected joint of every
    # selected env. Pairing the two flat index tensors directly only works when
    # one of them has length 1 (e.g. single-env teleop).
    joint_ids_t = torch.as_tensor(joint_ids, device=limits.device, dtype=torch.long)
    rows, cols = env_ids_t[:, None], joint_ids_t[None, :]
    if lower is not None:
        limits[rows, cols, 0] = lower
    if upper is not None:
        limits[rows, cols, 1] = upper
    # Clamp to PhysX supported range for revolute joints.
    limits[rows, cols] = torch.clamp(limits[rows, cols], -2.0 * torch.pi, 2.0 * torch.pi)
    limits_to_set = limits[rows, cols]  # (N, J, 2)
    asset.write_joint_position_limit_to_sim(
        limits_to_set, joint_ids=joint_ids, env_ids=env_ids_t, warn_limit_violation=False
    )


def _resolve_single_body_index(env: ManagerBasedRLEnv, body_cfg: SceneEntityCfg) -> int:
    """Body index for a one-body ``SceneEntityCfg`` (resolving it lazily if needed)."""
    if isinstance(body_cfg.body_ids, slice):
        body_cfg.resolve(env.scene)
    ids = body_cfg.body_ids
    if isinstance(ids, slice):
        return 0
    return int(ids[0])


def over_center_hinge(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    over_center_angle: float,
    close_torque: float,
    lock_torque: float,
    transition_width: float = 0.05,
    open_sign: float = -1.0,
    max_open_angle: float | None = None,
):
    """Bistable ("over-centre") spring on a revolute joint, as a per-step feed-forward torque.

    Emulates a stapler-style top hinge. With ``phi`` the opening angle:

    * ``phi < over_center_angle - transition_width``: constant closing torque
      ``close_torque`` (N*m) -- the spring pulls the top shut all the way.
    * the ``transition_width`` band below the over-centre angle: the closing
      torque ramps linearly to zero (a sharp crossing, so the part's own
      gravity cannot pull it open before the over-centre point).
    * ``phi >= over_center_angle``: a small opening ``lock_torque`` growing
      toward the open limit parks the joint there ("locked open") until it is
      pushed back across the over-centre angle, whereupon the closing branch
      takes over and snaps it shut.

    ``open_sign`` is the joint direction that opens (``-1`` when the open limit
    is the negative one). Register with ``mode="interval",
    interval_range_s=(0.0, 0.0)`` so it runs every env step, and give the joint
    an implicit actuator with ``stiffness=0`` (the effort target is applied as
    feed-forward on top of the drive's damping / joint friction).
    ``max_open_angle`` defaults to the joint's opening limit.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)
    env_ids_t = resolve_env_ids(env, env_ids)
    if env_ids_t.shape[0] == 0:
        return
    joint_ids_t = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    q = asset.data.joint_pos[env_ids_t][:, joint_ids_t]
    phi = open_sign * q  # opening angle >= 0
    if max_open_angle is None:
        limits = asset.data.joint_pos_limits[env_ids_t][:, joint_ids_t]  # (N, J, 2)
        phi_max = limits[..., 1] if open_sign > 0 else -limits[..., 0]
    else:
        phi_max = torch.full_like(phi, float(max_open_angle))
    oc = float(over_center_angle)
    w = max(float(transition_width), 1e-6)
    closing = float(close_torque) * torch.clamp((oc - phi) / w, 0.0, 1.0)
    lock_frac = torch.clamp((phi - oc) / (phi_max - oc).clamp_min(1e-6), 0.0, 1.0)
    locking = float(lock_torque) * lock_frac
    tau = torch.where(phi < oc, -open_sign * closing, open_sign * locking)
    asset.set_joint_effort_target(tau, joint_ids=joint_ids, env_ids=env_ids_t)


def follow_body_when_stage(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    target_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
    task_key: str,
    stage_name: str,
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    persistent: bool = True,
):
    """Pin a rigid object to an articulation body once a stage-graph stage is latched.

    Every env step (register with ``mode="interval", interval_range_s=(0, 0)``)
    the envs whose ``stage_name`` flag is set get ``target`` teleported to
    ``body`` pose * (``local_pos``, ``local_quat``) with zero velocity -- a
    "seated" object riding its holder (e.g. a loaded staple strip in a
    magazine, standing in for the follower spring). Envs where the stage is
    not (yet) reached are untouched.
    """
    from .stage_machine import evaluate_stage_graph

    env_ids_t = resolve_env_ids(env, env_ids)
    if env_ids_t.shape[0] == 0:
        return
    flags = evaluate_stage_graph(env, task_key=task_key, persistent=persistent)
    mask = flags[stage_name][env_ids_t]
    if not bool(mask.any()):
        return
    ids = env_ids_t[mask]
    body: Articulation = env.scene[body_cfg.name]
    bidx = _resolve_single_body_index(env, body_cfg)
    body_pos = body.data.body_pos_w[ids, bidx]
    body_quat = body.data.body_quat_w[ids, bidx]
    n = ids.shape[0]
    offset = torch.tensor(local_pos, device=env.device, dtype=body_pos.dtype).unsqueeze(0).expand(n, -1)
    lq = torch.tensor(local_quat, device=env.device, dtype=body_quat.dtype).unsqueeze(0).expand(n, -1)
    pos = body_pos + quat_apply(body_quat, offset)
    quat = quat_mul(body_quat, lq)
    target: RigidObject = env.scene[target_cfg.name]
    target.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
    target.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device, dtype=pos.dtype), env_ids=ids)


def hide_prims(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    prim_paths: list[str],
):
    """Make USD prims invisible (per env; paths may contain ``{ENV_REGEX_NS}``).

    Physics is unaffected (colliders keep colliding). Register as
    ``mode="prestartup"`` or ``"startup"``; e.g. to blank out a part modelled
    into an asset that the task supplies separately (a stapler's built-in
    staple row when the task is to load one).
    """
    from isaaclab.sim.utils.queries import find_matching_prims
    from pxr import UsdGeom

    for pattern in prim_paths:
        prims = find_matching_prims(pattern.format(ENV_REGEX_NS=env.scene.env_regex_ns))
        if not prims:
            raise RuntimeError(f"hide_prims: no prims match '{pattern}'.")
        for prim in prims:
            UsdGeom.Imageable(prim).MakeInvisible()

def debug_reset_reasons(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    prefix: str = "Env reset",
):
    """Print reset reasons for each env_id based on active termination terms."""
    env_ids_t = resolve_env_ids(env, env_ids)

    term_names = env.termination_manager.active_terms
    if not term_names:
        for env_id in env_ids_t.tolist():
            print(f"{prefix} [{env_id}]: no termination terms active")
        return

    term_buffers = {name: env.termination_manager.get_term(name) for name in term_names}
    for env_id in env_ids_t.tolist():
        reasons = [name for name in term_names if bool(term_buffers[name][env_id])]
        if reasons:
            print(f"{prefix} [{env_id}]: {', '.join(reasons)}")
        else:
            print(f"{prefix} [{env_id}]: unknown")


def update_articulation_root_from_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    target_cfg: SceneEntityCfg,
    source_cfg: SceneEntityCfg,
):
    """Update an articulation root pose to follow a rigid object's root pose."""
    target: Articulation = env.scene[target_cfg.name]
    source = env.scene[source_cfg.name]

    env_ids_t = resolve_env_ids(env, env_ids)

    pos = source.data.root_pos_w[env_ids_t]
    quat = source.data.root_quat_w[env_ids_t]
    root_pose = torch.cat([pos, quat], dim=-1)
    target.write_root_pose_to_sim(root_pose, env_ids=env_ids_t)


def update_marker_from_body(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    marker_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Update a marker pose to follow a given articulation body with a local offset."""
    marker: RigidObject = env.scene[marker_cfg.name]
    asset: Articulation = env.scene[body_cfg.name]

    if isinstance(body_cfg.body_ids, slice) and body_cfg.body_ids == slice(None) and body_cfg.body_names is not None:
        body_cfg.resolve(env.scene)
    body_ids = body_cfg.body_ids
    if isinstance(body_ids, slice):
        body_id = 0
    else:
        body_id = body_ids[0]

    env_ids_t = resolve_env_ids(env, env_ids)

    body_pose = asset.data.body_pose_w[env_ids_t, body_id]
    offset_t = torch.tensor(offset, device=env.device, dtype=body_pose.dtype).unsqueeze(0).repeat(env_ids_t.shape[0], 1)
    pos = body_pose[:, 0:3] + quat_apply(body_pose[:, 3:7], offset_t)
    root_pose = torch.cat([pos, body_pose[:, 3:7]], dim=-1)
    marker.write_root_pose_to_sim(root_pose, env_ids=env_ids_t)


def filter_collision_pairs(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    source_prim_path: str,
    target_prim_paths: list[str],
):
    """Author PhysX filtered pairs so ``source_prim_path`` never collides with
    ``target_prim_paths`` (same-env pairs only).

    Must be registered with ``mode="prestartup"``: it edits the USD stage
    (``UsdPhysics.FilteredPairsAPI``) and so has to run after scene cloning
    but *before* PhysX parses the stage -- exactly when prestartup events run
    (they also require ``replicate_physics=False``, the dexverse default).
    Paths may contain ``{ENV_REGEX_NS}``; each matched source prim is paired
    only with the target paths of its own ``env_N`` instance. Idempotent
    (re-running rewrites the same relationship targets).
    """
    import re

    from isaaclab.sim.utils.queries import find_matching_prims
    from pxr import UsdPhysics

    source_pattern = source_prim_path.format(ENV_REGEX_NS=env.scene.env_regex_ns)
    env_prefix_re = re.compile(env.scene.env_regex_ns.replace(".*", "[0-9]+"))
    prims = find_matching_prims(source_pattern)
    if not prims:
        raise RuntimeError(f"filter_collision_pairs: no prims match '{source_pattern}'.")
    for prim in prims:
        path = prim.GetPath().pathString
        match = env_prefix_re.match(path)
        if match is None:
            raise RuntimeError(f"filter_collision_pairs: '{path}' is outside the env namespace.")
        env_prefix = match.group(0)
        api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = api.CreateFilteredPairsRel()
        rel.ClearTargets(True)
        for target in target_prim_paths:
            rel.AddTarget(target.format(ENV_REGEX_NS=env_prefix))
