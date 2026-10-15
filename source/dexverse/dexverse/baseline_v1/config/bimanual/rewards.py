# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configurations for the bimanual task family.

Holds the coordinated-lift reward presets plus the two-hand reward functions
they use. The ``Dense*`` preset is the PPO-ready variant: per-hand reach (so
each hand is independently pulled to the object), lift progress graded toward
the success height, a both-hands grasp-gated lift term, and keep-level
shaping. Fingertip wiring happens via :func:`wire_bimanual_reward_fingertips`
from ``BimanualLiftObjectEnvCfg.__post_init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ...mdp.utils import root_height_delta

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def two_hand_reach_reward(
    env: ManagerBasedRLEnv,
    right_fingers_cfg: SceneEntityCfg,
    left_fingers_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_gain: float = 5.0,
) -> torch.Tensor:
    """Mean of per-hand exponential reach rewards.

    Each hand's mean fingertip distance decays independently, so one hand
    parked on the object cannot satisfy the term alone — unlike a single
    reach summed/meaned over all ten fingertips.
    """
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    reward = torch.zeros(env.num_envs, device=env.device)
    for fingers_cfg in (right_fingers_cfg, left_fingers_cfg):
        asset = env.scene[fingers_cfg.name]
        tips = asset.data.body_pos_w[:, fingers_cfg.body_ids]
        distance = torch.norm(tips - object_pos[:, None, :], dim=-1).mean(dim=-1)
        reward = reward + 0.5 * torch.exp(-distance_gain * distance)
    return reward


def bimanual_grasp_lift_reward(
    env: ManagerBasedRLEnv,
    right_fingers_cfg: SceneEntityCfg,
    left_fingers_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    threshold: float = 0.08,
    min_fingers_per_hand: int = 2,
    min_height: float = 0.15,
) -> torch.Tensor:
    """Lift progress gated on both hands being in contact range of the object.

    A hand counts as grasping when at least ``min_fingers_per_hand`` of its
    fingertips are within ``threshold`` of the object root. Deliberately does
    not read the lift action: bimanual action layouts make a fixed action
    index unreliable, and the height delta itself is the trusted signal.
    """
    object_asset = env.scene[object_cfg.name]
    object_pos = object_asset.data.root_pos_w
    grasping = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for fingers_cfg in (right_fingers_cfg, left_fingers_cfg):
        asset = env.scene[fingers_cfg.name]
        tips = asset.data.body_pos_w[:, fingers_cfg.body_ids]
        distance = torch.norm(tips - object_pos[:, None, :], dim=-1)
        grasping = grasping & ((distance <= threshold).sum(dim=1) >= min_fingers_per_hand)
    lift = torch.clamp(root_height_delta(object_asset) / max(min_height, 1e-6), 0.0, 1.0)
    return grasping * lift


def wire_bimanual_reward_fingertips(rewards, fingertip_body_names: list[str]) -> None:
    """Point the bimanual reward terms at the active robot's fingertip bodies.

    Handles both reach-term shapes: the legacy all-tips ``object_ee_distance``
    (``asset_cfg``) and the per-hand terms (``right_fingers_cfg`` /
    ``left_fingers_cfg``, split by the Shadow bimanual ``rh_``/``lh_`` body
    name prefixes; a hand with no prefixed names falls back to all tips).
    """
    right = [name for name in fingertip_body_names if name.startswith("rh_")]
    left = [name for name in fingertip_body_names if name.startswith("lh_")]
    for term_name in ("fingers_to_object", "bimanual_grasp_lift"):
        term = getattr(rewards, term_name, None)
        if term is None:
            continue
        if "asset_cfg" in term.params:
            term.params["asset_cfg"].body_names = fingertip_body_names
        else:
            term.params["right_fingers_cfg"] = SceneEntityCfg("robot", body_names=right or fingertip_body_names)
            term.params["left_fingers_cfg"] = SceneEntityCfg("robot", body_names=left or fingertip_body_names)


@configclass
class BimanualLiftRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reach + height-based lift signal (works with any fingertip count)."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            # body_names wired to fingertip_body_names in __post_init__.
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )

    object_lift_height = RewTerm(
        func=mdp.object_lift_height,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("object"), "min_height": 0.0},
    )


@configclass
class DenseBimanualLiftRewardsCfg(BimanualLiftRewardsCfg):
    """PPO-ready dense preset for bimanual coordinated lifts.

    ``min_height`` on the lift terms is graded toward ``lift_height`` (wired
    in ``BimanualLiftObjectEnvCfg.__post_init__``), and the fingertip cfgs of
    the per-hand terms are wired by :func:`wire_bimanual_reward_fingertips`.
    """

    fingers_to_object = RewTerm(
        func=two_hand_reach_reward,
        weight=2.0,
        params={
            "right_fingers_cfg": SceneEntityCfg("robot", body_names=None),
            "left_fingers_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "distance_gain": 5.0,
        },
    )

    object_lift_height = RewTerm(
        func=mdp.object_lift_height,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("object"), "min_height": 0.15},
    )

    bimanual_grasp_lift = RewTerm(
        func=bimanual_grasp_lift_reward,
        weight=1.0,
        params={
            "right_fingers_cfg": SceneEntityCfg("robot", body_names=None),
            "left_fingers_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.08,
            "min_fingers_per_hand": 2,
            "min_height": 0.15,
        },
    )

    # Keep the object level while carrying it (trays/cartons tip easily);
    # rises from 0 at ~20 deg of tilt to 1 when upright.
    keep_level = RewTerm(
        func=mdp.tilt_angle_reward,
        weight=1.0,
        params={
            "threshold_rad": 0.35,
            "axis_local": (0.0, 0.0, 1.0),
            "world_axis": (0.0, 0.0, 1.0),
            "tilt_ge": False,
            "object_cfg": SceneEntityCfg("object"),
        },
    )
