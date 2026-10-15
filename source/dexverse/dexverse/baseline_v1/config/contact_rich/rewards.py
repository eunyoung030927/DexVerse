# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dense reward functions for the contact-rich insertion task family.

These shape the stages the sparse ``engaged``/``success`` bonuses skip: carry
the held part toward the receptacle opening, orient it for insertion, and
grade the insertion depth. Written for the pen/pen-holder task but generic to
any two-ended held part with a fixed upright receptacle — offsets and
thresholds come in as params from the task cfg (which owns the geometry
constants), e.g. ``DenseInsertPenRewardsCfg`` in ``insertpen_cfg.py``.

All terms gate on the held part being lifted off the table (``min_lift``)
so they cannot pull the hand toward the receptacle before a grasp exists,
and the depth term additionally gates on being XY-centered over the opening
so resting the part on the rim scores nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from ...mdp.utils import asset_axis_w, root_height_delta, world_points_from_local

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _held_tip_positions_w(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    tip_local_offsets: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
    """World positions of the held part's tip reference points, ``(E, T, 3)``."""
    tips = torch.tensor(tip_local_offsets, device=env.device, dtype=torch.float32)
    return world_points_from_local(env, held_cfg, tips)


def _opening_position_w(
    env: ManagerBasedRLEnv,
    fixed_cfg: SceneEntityCfg,
    target_local_offset: tuple[float, float, float],
) -> torch.Tensor:
    """World position of the receptacle opening center, ``(E, 3)``."""
    point = torch.tensor([target_local_offset], device=env.device, dtype=torch.float32)
    return world_points_from_local(env, fixed_cfg, point).squeeze(1)


def _lift_gate(env: ManagerBasedRLEnv, held_cfg: SceneEntityCfg, min_lift: float) -> torch.Tensor:
    """1.0 where the held part has risen at least ``min_lift`` above its spawn."""
    if min_lift <= 0.0:
        return torch.ones(env.num_envs, device=env.device)
    return (root_height_delta(env.scene[held_cfg.name]) >= min_lift).float()


def pen_tip_to_opening_reward(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    tip_local_offsets: tuple[tuple[float, float, float], ...],
    target_local_offset: tuple[float, float, float],
    distance_gain: float = 8.0,
    min_lift: float = 0.03,
) -> torch.Tensor:
    """Exponential decay of the closest tip-to-opening distance, lift-gated."""
    tips_w = _held_tip_positions_w(env, held_cfg, tip_local_offsets)
    opening_w = _opening_position_w(env, fixed_cfg, target_local_offset)
    distance = torch.norm(tips_w - opening_w[:, None, :], dim=-1).min(dim=1).values
    return torch.exp(-distance_gain * distance) * _lift_gate(env, held_cfg, min_lift)


def pen_axis_vertical_alignment_reward(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0),
    min_lift: float = 0.03,
) -> torch.Tensor:
    """|cos| between the held part's long axis and world up, lift-gated.

    Sign-agnostic — either end may point down. The lift gate keeps a part
    standing on the table from collecting the reward.
    """
    axis_w = asset_axis_w(env, asset_cfg=held_cfg, axis_local=axis_local)
    alignment = torch.abs(axis_w[:, 2]).clamp(0.0, 1.0)
    return alignment * _lift_gate(env, held_cfg, min_lift)


def pen_insertion_depth_reward(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    tip_local_offsets: tuple[tuple[float, float, float], ...],
    target_local_offset: tuple[float, float, float],
    center_dist_thresh: float,
    depth_range: float = 0.03,
) -> torch.Tensor:
    """Graded insertion depth below the rim, gated on being XY-centered.

    For each tip: depth = rim_z - tip_z (world frame; the receptacle stands
    upright), clamped to ``[0, depth_range]`` and normalized, and counted only
    while that tip is within ``center_dist_thresh`` of the opening center in
    XY. The best tip wins, so either end of the part may be inserted.
    """
    tips_w = _held_tip_positions_w(env, held_cfg, tip_local_offsets)
    opening_w = _opening_position_w(env, fixed_cfg, target_local_offset)
    xy_dist = torch.norm(tips_w[..., :2] - opening_w[:, None, :2], dim=-1)
    depth = (opening_w[:, None, 2] - tips_w[..., 2]).clamp(0.0, depth_range) / max(depth_range, 1e-6)
    per_tip = depth * (xy_dist <= center_dist_thresh).float()
    return per_tip.max(dim=1).values
