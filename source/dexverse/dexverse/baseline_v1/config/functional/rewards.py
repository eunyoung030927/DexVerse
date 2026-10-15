# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configurations for the functional task family.

Holds the reward-term presets for functional grasping / pouring / tool-use
tasks so per-task reward shaping lives next to the task configs rather than in
the shared ``tasks/mdp/rewards.py`` (which only hosts generic primitives).

The ``Dense*`` presets are the PPO-ready variants: they keep the legacy term
names (so ``__post_init__`` wiring keeps working) but use mean-reduced reach
and name-scoped grasp conditions that are correct for five-fingertip hands.
"""

from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp

POUR_ANGLE_RAD = math.radians(100.0)

# Substrings identifying the thumb fingertip across supported hands
# (Shadow: ``thtip`` / ``rh_thtip`` / ``lh_thtip``; DexHand: ``thumb_fingertip``).
_THUMB_NAME_HINTS = ("thtip", "thumb")


def split_thumb_fingertip_names(fingertip_body_names: list[str]) -> tuple[list[str], list[str]]:
    """Split fingertip body names into (thumb names, non-thumb names)."""
    thumbs = [name for name in fingertip_body_names if any(hint in name.lower() for hint in _THUMB_NAME_HINTS)]
    others = [name for name in fingertip_body_names if name not in thumbs]
    return thumbs, others


def wire_fingertip_reward_bodies(rewards, fingertip_body_names: list[str]) -> None:
    """Point the reach/grasp reward terms at the active robot's fingertip bodies.

    Handles both grasp-term shapes: the legacy ``lift_when_grasping_reward``
    (positional ``asset_cfg``) and ``grasp_lift_action_reward`` (name-scoped
    ``fingers_cfg`` / ``thumb_cfg``).
    """
    fingers_to_object = getattr(rewards, "fingers_to_object", None)
    if fingers_to_object is not None:
        fingers_to_object.params["asset_cfg"].body_names = fingertip_body_names
    term = getattr(rewards, "lift_when_grasping", None)
    if term is None:
        return
    if "asset_cfg" in term.params:
        term.params["asset_cfg"].body_names = fingertip_body_names
    else:
        thumbs, others = split_thumb_fingertip_names(fingertip_body_names)
        term.params["thumb_cfg"] = SceneEntityCfg("robot", body_names=thumbs) if thumbs else None
        term.params["fingers_cfg"] = SceneEntityCfg("robot", body_names=others or fingertip_body_names)


@configclass
class FunctionalGraspingRewardsCfg(dexverse_base_env.RewardsCfg):
    """Pickup-style shaping + goal tracking and success bonus."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )

    lift_when_grasping = RewTerm(
        func=mdp.lift_when_grasping_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.08,
        },
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=2.0,
        params={"std": 0.15, "command_name": "object_pose"},
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=8.0,
        params={"pos_std": 0.05, "rot_std": None, "command_name": "object_pose"},
    )


@configclass
class DenseFunctionalGraspingRewardsCfg(FunctionalGraspingRewardsCfg):
    """PPO-ready dense preset: mean-reduced reach + name-scoped grasp-lift."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "reduce": "mean",
        },
        weight=2.0,
    )

    lift_when_grasping = RewTerm(
        func=mdp.grasp_lift_action_reward,
        weight=0.5,
        params={
            "object_cfg": SceneEntityCfg("object"),
            # fingers_cfg / thumb_cfg body names are wired from the active
            # robot's fingertips by wire_fingertip_reward_bodies().
            "fingers_cfg": SceneEntityCfg("robot", body_names=None),
            "thumb_cfg": None,
            "threshold": 0.08,
        },
    )


@configclass
class DenseHammerStrikeRewardsCfg(DenseFunctionalGraspingRewardsCfg):
    """PPO-ready dense preset for hammer strike.

    The goal command does not exist on this task, so the goal-tracking terms
    are disabled at the class level. The strike itself is shaped by driving
    the hammer head toward the nail head and by graded nail compression.
    """

    position_tracking = None
    success = None

    # Hammer-head point -> nail-head point. Local offsets are wired in
    # ``HammerStrikeEnvFloatingDexHandRightCfg.__post_init__`` (the nail-head
    # offset depends on the board scale; the hammer-head offset is a
    # leaf-tunable attribute).
    head_to_nail = RewTerm(
        func=mdp.point_to_point_distance_exp,
        weight=3.0,
        params={
            "source_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("hammer_target"),
            "source_local_offset": (0.0, 0.0, 0.0),
            "target_local_offset": (0.0, 0.0, 0.0),
            "distance_gain": 8.0,
        },
    )

    # Graded nail compression toward the success press distance. Joint names
    # and the threshold are wired in the leaf ``__post_init__``.
    nail_press = RewTerm(
        func=mdp.joint_displacement_reward,
        weight=8.0,
        params={
            "threshold_m": 0.06,
            "asset_cfg": SceneEntityCfg("hammer_target"),
        },
    )


@configclass
class FunctionalPourRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reach, grasp, and tilt shaping for pouring."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )

    lift_when_grasping = RewTerm(
        func=mdp.lift_when_grasping_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.08,
        },
    )

    tilt_reward = RewTerm(
        func=mdp.tilt_angle_reward,
        weight=5.0,
        params={
            "threshold_rad": POUR_ANGLE_RAD,
            "axis_local": (0.0, 0.0, 1.0),
            "world_axis": (0.0, 0.0, 1.0),
            "tilt_ge": True,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class DenseFunctionalPourRewardsCfg(FunctionalPourRewardsCfg):
    """PPO-ready dense preset: pour shaping plus graded lift progress.

    ``lift_progress`` grades 0 -> 1 up to the pour lift height (``min_height``
    is wired from ``pour_lift_height`` in ``FunctionalPourEnvCfg.__post_init__``)
    so the tilt term is reachable through a learnable lift stage.
    """

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "reduce": "mean",
        },
        weight=2.0,
    )

    lift_when_grasping = RewTerm(
        func=mdp.grasp_lift_action_reward,
        weight=0.5,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "fingers_cfg": SceneEntityCfg("robot", body_names=None),
            "thumb_cfg": None,
            "threshold": 0.08,
        },
    )

    lift_progress = RewTerm(
        func=mdp.object_lift_height,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("object"), "min_height": 0.2},
    )
