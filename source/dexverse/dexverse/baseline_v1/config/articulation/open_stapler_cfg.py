# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixate-then-manipulate: press a stapler head (synthesis/green stapler)."""

import math

from isaaclab.actuators import ImplicitActuatorCfg

from dexverse.assets import SYNTHESIS_DIR
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .base_cfg import FixateArticulationEnvFloatingDexHandRightCfg

STAPLER_USD_PATH = str(SYNTHESIS_DIR / "green stapler" / "model_stapler_1.usd")


@configclass
class OpenStaplerEnvFloatingDexHandRightCfg(FixateArticulationEnvFloatingDexHandRightCfg):
    """Open the stapler head up onto the body while the base is free.

    Elements: ``E_shell_1`` (top shell), ``E_body_2`` (body / anvil),
    ``E_stapler_needle_3`` (inner plunger). The main rotary hinge between
    shell and body is what we ask the policy to actuate. A typical solution
    pins the body on the table while the palm / fingers press the shell
    downwards.
    """

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = STAPLER_USD_PATH
    articulation_scale: tuple = (1.0, 1.0, 1.0)
    articulation_init_pos: tuple = (0.0, 0.0, 0.0)
    # Rotate +90 deg about z (counterclockwise when viewed from +z to -z).
    articulation_init_rot: tuple = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    # Staplers sit on a flat base; a small half-height keeps the anvil on
    # the table.
    articulation_half_height_est: float = 0.003  # reset audit (Aug 2026): seated ~3 mm above rest instead of dropping (was 0.03)
    # Reset audit (Aug 2026): x was pinned, giving a 1-D spawn line. Randomize
    # both table axes; the band stays in front of the bimanual palms (x=-0.25).
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.15, 0.15],
        "y": [-0.2, 0.2],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        # Reset audit (Aug 2026): uniform yaw -- the object can face any way,
        # like the functional tasks; the two floating hands can approach from
        # any side.
        "yaw": [-3.14159, 3.14159],
    }

    success_joint_names: list[str] = ["RevoluteJoint_stapler_1_up"]
    success_threshold: float = 0.2

    # Spring-loaded shell hinge (reset audit, Aug 2026: the hinge used to be a free
    # joint that stayed wherever it was put -- too simple). An implicit position
    # drive with the setpoint at the rest angle acts like the stapler's spring:
    # lifting the shell works against ``stiffness`` (N*m/rad) and it snaps back
    # when released. 1.0 N*m/rad ~= 0.56 N*m at the 32-degree success angle
    # (~7 N at the shell tip). Set stiffness to 0.0 to recover the free hinge.
    articulation_hinge_stiffness: float = 1.0
    articulation_hinge_damping: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        if self.articulation_hinge_stiffness > 0.0 or self.articulation_hinge_damping > 0.0:
            self.scene.articulation.actuators = {
                "stapler_hinge": ImplicitActuatorCfg(
                    joint_names_expr=self.success_joint_names,
                    effort_limit_sim=100.0,
                    velocity_limit_sim=100.0,
                    stiffness=self.articulation_hinge_stiffness,
                    damping=self.articulation_hinge_damping,
                ),
            }
            # keep the drive setpoint at the rest angle every episode
            self.events.reset_stapler_hinge_target = EventTerm(
                func=mdp.set_joint_position_target_to_init,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names)},
            )
        self.terminations.success.func = mdp.joint_relative_move
        # Deep press: 80% of the shell hinge travel.
        self.terminations.success.params = {
            "threshold": self.success_threshold,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "progress",
            "op": ">=",
            "reduce": "any",
        }
