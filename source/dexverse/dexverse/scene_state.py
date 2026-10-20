# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Writing a recorded scene state without leaving hidden drive targets behind.

``InteractiveScene.reset_to`` (and ``ManagerBasedEnv.reset_to``, which calls it) writes the joint state of every
articulation and then also sets its PD targets to it: position target = written joint position and **velocity
target = written joint velocity**. Actions only set position targets, so that velocity target stays until the next
restore, and the drive's damping term keeps pushing every joint toward the velocity it had at the restored instant
(``damping * (v_target - v)``): a hidden force that a continuous rollout never has (its velocity targets are zero).

Measured 2026-10-19 (GraspPan, SPIDER env): restoring a snapshot of the env's own state changed the object velocity
one step later by 24-25 mm/s (run-to-run noise 3-7 mm/s); setting only the velocity target to the current joint
velocity reproduced it, writing only the joint state did not. SPIDER restores every 15-step commit, so its actions
were optimised under that force and failed in continuous replay (0-4% vs 4-17/45 with restores).

Use :func:`reset_scene_to` / :func:`clear_joint_velocity_targets` wherever a state is restored and physics steps
afterwards; continuous rollouts keep zero velocity targets.
"""

from __future__ import annotations

import torch


def clear_joint_velocity_targets(scene, env_ids=None) -> None:
    """Zero the joint velocity targets of every articulation of ``scene`` (all envs if ``env_ids`` is None)."""
    for art in scene.articulations.values():
        n = art.data.joint_vel_target.shape[0] if env_ids is None else len(env_ids)
        art.set_joint_velocity_target(
            torch.zeros(n, art.num_joints, device=art.data.joint_vel_target.device), env_ids=env_ids)


def reset_scene_to(scene, state, env_ids=None, is_relative: bool = True) -> None:
    """``scene.reset_to`` followed by :func:`clear_joint_velocity_targets` (same arguments)."""
    scene.reset_to(state, env_ids=env_ids, is_relative=is_relative)
    clear_joint_velocity_targets(scene, env_ids)
