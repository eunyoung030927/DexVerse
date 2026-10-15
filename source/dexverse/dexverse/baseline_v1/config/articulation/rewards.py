# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configurations for the articulation task family.

The ``Dense*`` presets are the PPO-ready variants. Term param wiring
(fingertip body names, ``success_joint_names``, ``dense_reach_body_names``)
happens generically in ``ArticulationBaseEnvCfg.__post_init__`` — see the
"dense reward wiring" block there — so leaves only pick a preset and, where
needed, name the moving link via ``dense_reach_body_names``.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .articulation_base.articulation_base_cfg import ARTICULATION_KEY, ArticulationBaseRewardsCfg


@configclass
class DenseArticulationRewardsCfg(ArticulationBaseRewardsCfg):
    """PPO-ready preset for fixed-root articulations (faucet, cabinet, ...).

    Keeps the threshold-normalized ``open_amount`` term (auto-wired from
    ``success_threshold``) and adds fingertip reach shaping toward the
    articulation so exploration finds the handle at all.
    """

    fingers_to_part = RewTerm(
        func=mdp.fingertips_to_body_distance,
        weight=1.5,
        params={
            # target body names come from ``dense_reach_body_names`` (None ->
            # mean over all links); fingertip body names are wired from the
            # active robot config.
            "target_cfg": SceneEntityCfg(ARTICULATION_KEY),
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "distance_gain": 5.0,
            "reduce": "mean",
        },
    )


@configclass
class DenseFixateArticulationRewardsCfg(DenseArticulationRewardsCfg):
    """PPO-ready preset for free-root fixate-then-manipulate articulations.

    * ``open_amount`` is swapped to init-relative range progress, matching the
      ``joint_relative_move(mode="progress")`` success criterion these tasks
      use (the ``articulation_base`` threshold auto-wiring only applies to
      ``joint_open_reward`` and skips this term by design).
    * ``stabilize_lin``/``stabilize_ang`` penalize root motion of the free
      articulation — the pin-the-base sub-behavior is otherwise unrewarded
      (a toppled/slid object only shows up as an ``out_of_bound`` termination).
      Keep these small relative to the positive terms so ending the episode
      early never beats making progress.
    """

    open_amount = RewTerm(
        func=mdp.joint_range_progress_from_init,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=".*")},
    )

    stabilize_lin = RewTerm(
        func=mdp.root_lin_vel_norm,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg(ARTICULATION_KEY)},
    )

    stabilize_ang = RewTerm(
        func=mdp.root_ang_vel_norm,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg(ARTICULATION_KEY)},
    )


@configclass
class DenseSlideKnifeRewardsCfg(DenseFixateArticulationRewardsCfg):
    """Utility-knife preset: fixate shaping plus the lift the success requires.

    ``joint_moved_and_object_lifted`` success also demands the knife be held
    above the table; ``knife_lift`` grades that sub-goal (``min_height`` is
    wired from ``success_lift_min_height`` in the leaf ``__post_init__``).
    """

    knife_lift = RewTerm(
        func=mdp.object_lift_height,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg(ARTICULATION_KEY), "min_height": 0.2},
    )
