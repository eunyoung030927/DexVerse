"""Contact-based cutaway-door evaluation; the catch is entirely physical."""

import torch

from dexverse.teleop_utils.door_latch_progress import new_door_latch_state, update_door_latch_state

BUFFER = "cutaway_door_progress"


def reset_door_latch(env, env_ids, buffer_name=BUFFER, door_preload_angle=-0.20):
    state = getattr(env, buffer_name, None)
    if state is None:
        state = new_door_latch_state(env.num_envs, env.device)
        setattr(env, buffer_name, state)
    ids = slice(None) if env_ids is None else env_ids
    state[ids] = 0
    state[ids, 5] = -1
    asset = env.scene["articulation"]
    # Reset explicit drive targets too: never retain teleop/debug targets.
    targets = torch.zeros_like(asset.data.default_joint_pos[ids])
    targets[:, asset.find_joints("door_hinge")[0][0]] = door_preload_angle
    asset.set_joint_position_target(targets, env_ids=env_ids)
    asset.set_joint_velocity_target(torch.zeros_like(asset.data.default_joint_vel[ids]), env_ids=env_ids)


def _force(env, name):
    matrix = env.scene[name].data.force_matrix_w
    if matrix is None:
        raise RuntimeError(f"{name} requires filtered contact reporting")
    return matrix.norm(dim=-1).reshape(env.num_envs, -1).sum(dim=1)


def _evaluate(env, hold_steps=60):
    asset = env.scene["articulation"]
    door_id = asset.find_joints("door_hinge")[0][0]
    latch_id = asset.find_joints("latch_slide")[0][0]
    q, qd = asset.data.joint_pos, asset.data.joint_vel
    # Actual filtered forces, not broad proximity regions. A nearby released
    # hand is allowed, but supporting either the panel or catch resets dwell.
    door_touch = _force(env, "door_hand_contact") > 0.01
    latch_touch = _force(env, "latch_hand_contact") > 0.01
    clear = ~door_touch & ~latch_touch
    state = getattr(env, BUFFER, None)
    if state is None:
        state = new_door_latch_state(env.num_envs, env.device)
        setattr(env, BUFFER, state)
    return update_door_latch_state(
        state,
        env.episode_length_buf,
        q[:, door_id],
        qd[:, door_id],
        q[:, latch_id],
        qd[:, latch_id],
        door_touch,
        _force(env, "door_latch_contact") > 0.05,
        clear,
        hold_steps,
    )


def door_latched_and_released(env, hold_steps=60):
    return _evaluate(env, hold_steps)[0]


def door_reclosed(env, hold_steps=60):
    return _evaluate(env, hold_steps)[1]


def door_latch_signals(env, hold_steps=60):
    state = getattr(env, BUFFER, None)
    if state is None:
        return torch.zeros((env.num_envs, 5), device=env.device)
    result = state[:, :5].clone()
    result[:, 4] /= float(hold_steps)
    return result
