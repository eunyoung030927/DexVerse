# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp a knocked-over cup, stand it upright and set it under the dispenser.

The pour tasks (``PourCan`` / ``GraspCup``) are all grasp-lift-tilt; this one
exercises **reorientation and precise upright placement** instead. The green
cup of :mod:`grasp_cup_cfg` spawns *lying on its side* (or upside down) at a
random pose on the table; the robot has to pick it up, turn it upright and
stand it on the drip tray of a counter-top **dispenser**, exactly under the
nozzle, then let go.

* Dispenser (:func:`dexverse.assets.desk_props_builder.build_dispenser`): a
  kinematic prop -- drip tray, back column, arm and a nozzle whose bottom sits
  ``spout_height`` = 16 cm above the tray (the cup is 9.6 cm tall), with a
  green target disk on the tray under the nozzle (the vision landmark). It
  spawns at the far side of the table facing the robot, with xy / yaw jitter;
  the cup spawns clear of it.
* Cup spawn (:func:`mdp.reset_root_pose_uniform_orientations`): rest orientation
  drawn from ``cup_spawn_weights`` (side / upside-down / upright), random xy and
  yaw; the draw is kept in ``env.cup_spawn_orientation`` (recorded with the
  episode task state).
* Success (stage graph ``cup_upright -> cup_placed``,
  :func:`mdp.object_placed_at_spot`): the cup's base within ``spot_xy_tol`` of
  the tray centre and ``spot_z_tol`` of the tray top, upright within
  ``upright_tol_deg``, at rest, **and released** (every robot body farther than
  the cup radius + ``release_clearance`` from the cup's axis), for
  ``placed_hold_steps`` consecutive steps.
* Failure: the cup-scaled rim forbidden zone (fingers in the drink) becomes an
  immediate failure (``rim_touch_is_failure``); cup off the table.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from dexverse.assets.desk_props_builder import ensure_dispenser_asset
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..grasping.forbidden_zones import ForbiddenZone, split_zones
from .grasp_cup_cfg import GraspCupEnvFloatingDexHandRightCfg

STAGE_KEY = "place_cup_under_dispenser"
CUP_SPAWN_BUFFER = "cup_spawn_orientation"
# Green cup (Cup_B0CMD4LX4D), root at the base centre. Task-local scale is
# 10% smaller than the previous dispenser cup (0.9); leave GraspCup v0 alone.
CUP_SCALE = 0.81
CUP_RADIUS = 0.0425 * CUP_SCALE
CUP_HEIGHT = 0.118 * CUP_SCALE
CUP_UP_AXIS_LOCAL = (0.0, 0.0, 1.0)
_S45 = math.sqrt(0.5)
# rest orientations: (quat wxyz, height of the cup root above the table top)
CUP_REST_ORIENTATIONS = {
    "side": ((_S45, _S45, 0.0, 0.0), CUP_RADIUS + 0.003),
    "upside_down": ((0.0, 1.0, 0.0, 0.0), CUP_HEIGHT + 0.003),
    "upright": ((1.0, 0.0, 0.0, 0.0), 0.003),
}
CUP_REST_ORDER = tuple(CUP_REST_ORIENTATIONS)

DISPENSER_NAME = "dispenser"
# Tray long enough in x that the column's front face stays 5 cm behind the spot
# (cup radius now 3.4 cm): a cup on the spot must not touch the column.
DISPENSER_PARAMS = dict(
    spout_height=0.16,
    tray_size=(0.22, 0.16, 0.012),
    column_size=(0.06, 0.06, 0.26),
    body_color=(0.55, 0.55, 0.55),
    nozzle_color=(0.35, 0.35, 0.35),
    # Keep the placement target green and distinct from the grey hardware.
    target_color=(0.15, 0.75, 0.25),
)
DISPENSER_USD_PATH, DISPENSER_MANIFEST = ensure_dispenser_asset(DISPENSER_NAME, **DISPENSER_PARAMS)
SPOT_LOCAL: tuple[float, float, float] = tuple(float(v) for v in DISPENSER_MANIFEST["spot_local"])


def make_dispenser_cfg(usd_path: str, init_pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Dispenser",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=None,  # authored per prim (the target disk has none)
            mass_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class PlaceCupUnderDispenserEnvFloatingDexHandRightCfg(GraspCupEnvFloatingDexHandRightCfg):
    """Right the knocked-over cup and stand it on the dispenser tray under the nozzle."""

    # Scale both the rendered mesh and its authored convex colliders. Root-frame
    # zones/observation offsets are metric, so they must be scaled explicitly.
    object_scale: tuple[float, float, float] = (CUP_SCALE, CUP_SCALE, CUP_SCALE)
    forbidden_zones: tuple[ForbiddenZone, ...] = (
        ForbiddenZone(
            kind="cylinder", center=(0.0, 0.0, 0.11 * CUP_SCALE),
            radius=0.035 * CUP_SCALE, half_height=0.005 * CUP_SCALE,
        ),
    )
    pour_goal_object_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.1 * CUP_SCALE)

    # ---- cup spawn ---------------------------------------------------------------
    cup_spawn_weights: dict[str, float] = {"side": 0.8, "upside_down": 0.2, "upright": 0.0}
    # x starts at -0.08: the hand rests at x=-0.25 with fingertips reaching ~-0.13,
    # so a tall inverted cup dropped further back would land on them.
    object_reset_x_range: tuple[float, float] = (-0.08, 0.12)
    object_reset_y_range: tuple[float, float] = (-0.20, 0.20)
    object_reset_yaw_range: tuple[float, float] = (-math.pi, math.pi)
    min_cup_dispenser_xy_distance: float = 0.22

    # ---- dispenser spawn: far side, open side toward the robot -------------------
    dispenser_init_xy: tuple[float, float] = (0.30, 0.0)
    dispenser_pose_range: dict[str, list[float]] = {"x": [-0.04, 0.04], "y": [-0.15, 0.15], "yaw": [-0.25, 0.25]}

    # ---- success ----------------------------------------------------------------
    upright_stage_tol_deg: float = 15.0  # stage 1: cup standing up (anywhere)
    spot_xy_tol: float = 0.03
    spot_z_tol: float = 0.02
    upright_tol_deg: float = 10.0
    placed_max_lin_speed: float = 0.05
    placed_hold_steps: int = 30
    release_clearance: float = 0.02
    rim_touch_is_failure: bool = True
    episode_length_s_override: float = 25.0

    # no pour goal sphere / progress sphere: the dispenser's disk is the target
    pour_show_progress_marker: bool = False
    pour_goal_marker_camera_visible: bool = False

    def __post_init__(self):
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        self.scene.dispenser = make_dispenser_cfg(
            str(DISPENSER_USD_PATH), (self.dispenser_init_xy[0], self.dispenser_init_xy[1], table_top_z)
        )
        super().__post_init__()
        self.episode_length_s = self.episode_length_s_override

        # --- resets: dispenser first, then the cup clear of it, then the marker ----
        self.events.reset_object = None
        self.events.reset_dispenser = EventTerm(
            func=mdp.reset_root_pose_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("dispenser"),
                "pose_range": {
                    "x": list(self.dispenser_pose_range["x"]),
                    "y": list(self.dispenser_pose_range["y"]),
                    "z": [0.0, 0.0],
                    "roll": [0.0, 0.0],
                    "pitch": [0.0, 0.0],
                    "yaw": list(self.dispenser_pose_range["yaw"]),
                },
            },
        )
        self.events.reset_cup = EventTerm(
            func=mdp.reset_root_pose_uniform_orientations,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "pose_range": {
                    "x": list(self.object_reset_x_range),
                    "y": list(self.object_reset_y_range),
                    "yaw": list(self.object_reset_yaw_range),
                },
                "orientations": [CUP_REST_ORIENTATIONS[name] for name in CUP_REST_ORDER],
                "weights": [float(self.cup_spawn_weights.get(name, 0.0)) for name in CUP_REST_ORDER],
                "table_top_z": table_top_z,
                "buffer_name": CUP_SPAWN_BUFFER,
                "reference_asset_cfg": SceneEntityCfg("dispenser"),
                "min_xy_distance": self.min_cup_dispenser_xy_distance,
                "max_attempts": 40,
            },
        )
        # the (hidden) pour goal marker just tracks the spot (re-added here so it
        # runs after the dispenser reset; the inherited slot ran before it)
        self.scene.success_marker.spawn.visible = False
        self.events.reset_success_marker = None
        self.events.sync_marker_to_spot = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("success_marker"),
                "source_cfg": SceneEntityCfg("dispenser"),
                "source_local_offset": SPOT_LOCAL,
                "quat": (1.0, 0.0, 0.0, 0.0),
            },
        )
        # clear latched stages every reset (the cup's init pose is upright, which
        # would otherwise latch ``cup_upright`` from the construction-time eval)
        self.events.reset_stage_state = EventTerm(
            func=mdp.reset_stage_graph_state, mode="reset", params={"task_key": STAGE_KEY}
        )

        # --- stage graph: upright, then placed under the nozzle and released -------
        mdp.register_stage_graph(
            STAGE_KEY,
            mdp.StageGraphSpec(
                stages=(
                    mdp.StageSpec(
                        name="cup_upright",
                        func=mdp.object_axis_upright,
                        params={
                            "object_cfg": SceneEntityCfg("object"),
                            "axis_local": CUP_UP_AXIS_LOCAL,
                            "min_cos": math.cos(math.radians(self.upright_stage_tol_deg)),
                        },
                    ),
                    mdp.StageSpec(
                        name="cup_placed",
                        func=mdp.object_placed_at_spot,
                        params={
                            "cache_key": STAGE_KEY,
                            "spot_cfg": SceneEntityCfg("dispenser"),
                            "spot_local": SPOT_LOCAL,
                            "object_cfg": SceneEntityCfg("object"),
                            "xy_tol": self.spot_xy_tol,
                            "z_tol": self.spot_z_tol,
                            "axis_local": CUP_UP_AXIS_LOCAL,
                            "upright_cos": math.cos(math.radians(self.upright_tol_deg)),
                            "max_lin_speed": self.placed_max_lin_speed,
                            "hold_steps": self.placed_hold_steps,
                            "robot_cfg": self.forbidden_zone_asset_cfg(),
                            "object_radius": CUP_RADIUS,
                            "object_axis_length": CUP_HEIGHT,
                            "release_clearance": self.release_clearance,
                        },
                        deps=("cup_upright",),
                    ),
                ),
                terminal_stage="cup_placed",
                ordering_mode="strict",
                success_mode="substage",
            ),
            override=True,
        )
        self.terminations.success = DoneTerm(func=mdp.stage_success, params={"task_key": STAGE_KEY, "persistent": True})
        if self.rim_touch_is_failure:
            sphere_zones, box_zones, cylinder_zones = split_zones(self.forbidden_zones)
            self.terminations.rim_touched = DoneTerm(
                func=mdp.robot_in_forbidden_zones,
                params={
                    "asset_cfg": self.forbidden_zone_asset_cfg(),
                    "object_cfg": SceneEntityCfg("object"),
                    "sphere_zones": sphere_zones,
                    "box_zones": box_zones,
                    "cylinder_zones": cylinder_zones,
                },
            )

        # --- rewards: no tilt shaping here; stage bonus ------------------------------
        self.rewards.tilt_reward = None
        if getattr(self.rewards, "lift_progress", None) is not None:
            self.rewards.lift_progress = None
        self.rewards.stage_bonus = RewTerm(
            func=mdp.stage_success_reward, weight=5.0, params={"task_key": STAGE_KEY, "persistent": True}
        )

        # --- observations: spot in ``goal``; dispenser pose + stage signals in state
        @configclass
        class GoalObsCfg(ObsGroup):
            spot_pos_b = ObsTerm(
                func=mdp.object_local_point_pos_b,
                params={"object_cfg": SceneEntityCfg("dispenser"), "local_offset": SPOT_LOCAL},
                noise=Unoise(n_min=-0.0, n_max=0.0),
            )

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 0

        self.observations.goal = GoalObsCfg()
        state = self.observations.state
        if state is not None:
            state.dispenser_pos_b = ObsTerm(
                func=mdp.asset_pos_b,
                params={"asset_cfg": SceneEntityCfg("dispenser")},
                noise=Unoise(n_min=-0.0, n_max=0.0),
            )
            state.dispenser_quat_b = ObsTerm(
                func=mdp.object_quat_b,
                params={"object_cfg": SceneEntityCfg("dispenser")},
                noise=Unoise(n_min=-0.0, n_max=0.0),
            )
            state.stage_signals = ObsTerm(func=mdp.stage_signals, params={"task_key": STAGE_KEY, "persistent": True})
