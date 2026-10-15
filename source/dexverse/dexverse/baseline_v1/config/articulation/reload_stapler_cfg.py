# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open a free stapler and place its base on a raised target table.

The stapler starts near the table centre facing either left or right. The task
requires opening its top past the over-centre detent, moving the whole stapler
onto a marked target area at any orientation. It must then rest on the platform
without either hand touching it for a short dwell. Opening and placement may
be completed in either order.

The target height is randomized per episode by selecting one of several small
tables whose feet remain seated on the main table. Four camera-visible green
spheres mark the target footprint on the selected table. Contact with any
target table while the stapler is still near closed immediately fails.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from dexverse.assets.small_table_builder import ensure_small_table_asset
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .articulation_base.articulation_base_cfg import ARTICULATION_KEY
from .articulation_base.usd_helpers import replace_joint_with_fixed_parent
from .open_stapler_cfg import STAPLER_USD_PATH, OpenStaplerEnvFloatingDexHandRightCfg

TOP_HINGE = "RevoluteJoint_stapler_1_up"
MAGAZINE_HINGE = "RevoluteJoint_stapler_1_middle"
STAPLER_LINKS = ("E_body_2", "E_shell_1", "E_stapler_needle_3")
FIXED_MAGAZINE_USD_PATH = replace_joint_with_fixed_parent(
    STAPLER_USD_PATH,
    joint_name=MAGAZINE_HINGE,
    parent_body_path="/root/E_body_2",
)
TOP_HINGE_TRAVEL_DEG = 160.0

# Bounds of the stapler base link in its root frame, measured from the source
# asset. Success aligns the footprint centre (rather than the slightly
# off-centre physics root) with the goal entity.
STAPLER_BASE_HALF_EXTENTS = (0.09725, 0.035)
STAPLER_BASE_CENTER_LOCAL = (-0.00415, 0.0018, 0.0)

GOAL_MARKER_RADIUS = 0.012
TARGET_TABLE_HEIGHTS = (0.08, 0.12, 0.16)
TARGET_TABLE_HALF_EXTENTS = (
    STAPLER_BASE_HALF_EXTENTS[0] + 0.05,
    STAPLER_BASE_HALF_EXTENTS[1] + 0.05,
)
TARGET_TABLE_ASSETS = tuple(
    ensure_small_table_asset(
        f"stapler_target_table_{int(round(height * 100)):02d}cm",
        depth=2.0 * TARGET_TABLE_HALF_EXTENTS[0],
        width=2.0 * TARGET_TABLE_HALF_EXTENTS[1],
        height=height,
        top_thickness=0.012,
        leg_size=0.020,
        color=(0.48, 0.36, 0.22),
    )[0]
    for height in TARGET_TABLE_HEIGHTS
)
TARGET_TABLE_KEYS = tuple(f"stapler_target_table_{index}" for index in range(len(TARGET_TABLE_HEIGHTS)))
TARGET_TABLE_PRIM_PATHS = tuple(
    f"{{ENV_REGEX_NS}}/StaplerTargetTable{index}" for index in range(len(TARGET_TABLE_HEIGHTS))
)
GOAL_FRAME_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/StaplerGoalPose",
    spawn=sim_utils.CuboidCfg(
        size=(0.001, 0.001, 0.001),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visible=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
)


def _robot_contact_filter_paths(usd_path: str, prim_path: str) -> list[str]:
    """Resolve one robot rigid-body path per contact-filter column.

    PhysX requires each filter expression to match one body per environment.
    A Robot/.* filter also matches joints/material scopes and cannot represent
    both hands as a single column. Read the selected robot USD without editing
    it; retain only the environment wildcard in the spawned paths.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None or not stage.GetDefaultPrim():
        raise ValueError(f"Could not resolve robot USD default prim for stapler contact filters: {usd_path}")
    root = stage.GetDefaultPrim()
    paths = []
    for body in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if body.HasAPI(UsdPhysics.RigidBodyAPI):
            relative = str(body.GetPath().MakeRelativePath(root.GetPath()))
            paths.append(prim_path if relative == "." else f"{prim_path}/{relative}")
    if not paths:
        raise ValueError(f"No robot rigid bodies found for stapler contact filters: {usd_path}")
    return paths


@configclass
class ReloadStaplerEnvFloatingDexHandRightCfg(OpenStaplerEnvFloatingDexHandRightCfg):
    """Open the stapler and place it on a randomly elevated target table."""

    articulation_usd_path: str = FIXED_MAGAZINE_USD_PATH
    episode_length_s_override: float = 30.0

    # Tight positional variance, with a random left/right initial heading.
    # The authored nominal rotation is +90 degrees, so yaw offsets of 0 and pi
    # produce final world headings of +90 and -90 degrees.
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.025, 0.025],
        "y": [-0.025, 0.025],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [math.radians(-10.0), math.radians(10.0)],
    }
    articulation_yaw_choices: tuple[float, float] = (0.0, math.pi)

    # The goal is independently randomized on the far side of the table. Its
    # four corners move and rotate with this pose, showing both the target area
    # and its sampled orientation.
    # Keep the raised furniture clear of the initial stapler footprint even at
    # the closest sampled poses/orientations, avoiding reset-time collisions.
    goal_init_xy: tuple[float, float] = (0.25, 0.0)
    # Match the horizontal left/right axis used at initialization. Since the
    # four-corner goal treats 180-degree reversals as equivalent, one +90-degree
    # nominal frame represents both left- and right-facing placements.
    goal_init_rot: tuple[float, float, float, float] = (
        math.sqrt(0.5),
        0.0,
        0.0,
        math.sqrt(0.5),
    )
    goal_xy_range: dict[str, list[float]] = {
        "x": [-0.04, 0.04],
        "y": [-0.12, 0.12],
    }
    goal_yaw_range: tuple[float, float] = (math.radians(-15.0), math.radians(15.0))
    goal_pos_tol: float = 0.05
    goal_z_tol: float = 0.02
    goal_corner_marker_radius: float = GOAL_MARKER_RADIUS
    target_table_heights: tuple[float, ...] = TARGET_TABLE_HEIGHTS
    closed_contact_max_progress: float = 0.20
    target_contact_force_threshold: float = 0.25
    release_contact_force_threshold: float = 0.10
    # A padded volume around the complete open stapler. Any robot body inside
    # it resets the release dwell, even before physical contact occurs.
    release_forbidden_zone_center: tuple[float, float, float] = (
        STAPLER_BASE_CENTER_LOCAL[0],
        STAPLER_BASE_CENTER_LOCAL[1],
        0.10,
    )
    release_forbidden_zone_half_extents: tuple[float, float, float] = (
        STAPLER_BASE_HALF_EXTENTS[0] + 0.03,
        STAPLER_BASE_HALF_EXTENTS[1] + 0.03,
        0.12,
    )
    success_hold_steps: int = 60

    # Over-centre top hinge: it pulls closed before the detent and parks open
    # after crossing it.
    articulation_hinge_stiffness: float = 0.0
    articulation_hinge_damping: float = 0.02
    hinge_friction: float = 0.02
    hinge_over_center_deg: float = 150.0
    hinge_close_torque: float = 0.25
    hinge_transition_deg: float = 3.0
    hinge_lock_torque: float = 0.05
    base_mass_kg: float = 0.25
    top_open_deg: float = 152.0

    stapler_static_friction: float = 2.0
    stapler_dynamic_friction: float = 1.8

    def __post_init__(self):
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        goal_z = table_top_z + self.target_table_heights[0] + self.articulation_half_height_est
        self.scene.stapler_goal = GOAL_FRAME_CFG.replace(
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(self.goal_init_xy[0], self.goal_init_xy[1], goal_z),
                rot=self.goal_init_rot,
            )
        )
        for index, (key, prim_path, usd_path) in enumerate(
            zip(TARGET_TABLE_KEYS, TARGET_TABLE_PRIM_PATHS, TARGET_TABLE_ASSETS, strict=True)
        ):
            setattr(
                self.scene,
                key,
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=str(usd_path),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            rigid_body_enabled=True,
                            kinematic_enabled=True,
                            disable_gravity=True,
                        ),
                        # Collision boxes are authored in the procedural USD.
                        collision_props=None,
                        mass_props=None,
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(self.goal_init_xy[0], self.goal_init_xy[1], table_top_z if index == 0 else -2.0),
                        rot=self.goal_init_rot,
                    ),
                ),
            )
        super().__post_init__()
        self.events.reset_articulation.params["yaw_choices"] = self.articulation_yaw_choices
        self.events.reset_stapler_target = EventTerm(
            func=mdp.reset_random_support_and_goal,
            mode="reset",
            params={
                "support_cfgs": tuple(SceneEntityCfg(key) for key in TARGET_TABLE_KEYS),
                "support_heights": self.target_table_heights,
                "goal_cfg": SceneEntityCfg("stapler_goal"),
                "goal_xy_range": self.goal_xy_range,
                "support_base_z": table_top_z,
                "goal_z_clearance": self.articulation_half_height_est,
                "yaw_range": self.goal_yaw_range,
            },
        )

        # The filtered sensors see target-table contacts only, including hits
        # by the shell and the fixed magazine rather than just the base.
        self.scene.articulation.spawn.activate_contact_sensors = True
        target_filter_paths = list(TARGET_TABLE_PRIM_PATHS)
        robot_filter_paths = _robot_contact_filter_paths(
            self.scene.robot.spawn.usd_path, self.scene.robot.prim_path,
        )
        target_contact_sensor_names = []
        robot_contact_sensor_names = []
        for link_name in STAPLER_LINKS:
            sensor_name = f"{link_name}_target_tables_s"
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    # After the prepared USD is spawned, these rigid bodies are
                    # direct children of /Articulation (confirmed by the live
                    # articulation/contact-report initialization log).
                    prim_path=f"{{ENV_REGEX_NS}}/Articulation/{link_name}",
                    filter_prim_paths_expr=target_filter_paths,
                ),
            )
            target_contact_sensor_names.append(sensor_name)
            robot_sensor_name = f"{link_name}_robot_s"
            setattr(
                self.scene,
                robot_sensor_name,
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Articulation/{link_name}",
                    # Sense every rigid body below the robot root, so release
                    # includes palms and intermediate finger links as well as
                    # fingertips.
                    filter_prim_paths_expr=list(robot_filter_paths),
                ),
            )
            robot_contact_sensor_names.append(robot_sensor_name)

        drive_props = self.scene.articulation.spawn.joint_drive_props
        if drive_props is None:
            self.scene.articulation.spawn.joint_drive_props = sim_utils.JointDrivePropertiesCfg(drive_type="force")
        else:
            drive_props.drive_type = "force"
        self.scene.articulation.actuators = {
            "stapler_top": ImplicitActuatorCfg(
                joint_names_expr=[TOP_HINGE],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=self.articulation_hinge_damping,
                friction=self.hinge_friction,
            ),
        }
        self.events.reset_stapler_hinge_target = None
        self.events.stapler_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(ARTICULATION_KEY),
                "static_friction_range": (self.stapler_static_friction, self.stapler_static_friction),
                "dynamic_friction_range": (self.stapler_dynamic_friction, self.stapler_dynamic_friction),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
            },
        )
        self.events.set_stapler_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, body_names=[STAPLER_LINKS[0]]),
                "mass_distribution_params": (self.base_mass_kg, self.base_mass_kg),
                "operation": "abs",
                "recompute_inertia": True,
            },
        )
        self.events.over_center_top_hinge = EventTerm(
            func=mdp.over_center_hinge,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[TOP_HINGE]),
                "over_center_angle": math.radians(self.hinge_over_center_deg),
                "close_torque": self.hinge_close_torque,
                "lock_torque": self.hinge_lock_torque,
                "transition_width": math.radians(self.hinge_transition_deg),
                "open_sign": -1.0,
                "max_open_angle": math.radians(TOP_HINGE_TRAVEL_DEG),
            },
        )

        # Success requires opening, placement with physical platform contact,
        # and complete hand release for a consecutive dwell. The explicit
        # spatial zone catches nearby hand links; contact sensors remain a
        # second safeguard for geometries whose body origins sit outside it.
        forbidden_center = self.release_forbidden_zone_center
        forbidden_half = self.release_forbidden_zone_half_extents
        self.terminations.success.func = mdp.joint_pose_contact_release_hold_success
        self.terminations.success.params = {
            "threshold": self.top_open_deg / TOP_HINGE_TRAVEL_DEG,
            "goal_cfg": SceneEntityCfg("stapler_goal"),
            "support_sensor_names": target_contact_sensor_names,
            "robot_sensor_names": robot_contact_sensor_names,
            "forbidden_box_zones": [(*forbidden_center, *forbidden_half)],
            "forbidden_asset_cfg": SceneEntityCfg(
                "robot",
                body_names=self.robot_config.forbidden_zone_body_names,
            ),
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[TOP_HINGE]),
            "center_local": STAPLER_BASE_CENTER_LOCAL,
            "pos_tol": self.goal_pos_tol,
            "z_tol": self.goal_z_tol,
            # Placement orientation is unrestricted.
            "yaw_tol": None,
            "mode": "progress",
            "op": ">=",
            "reduce": "any",
            "support_force_threshold": self.target_contact_force_threshold,
            "robot_force_threshold": self.release_contact_force_threshold,
            "hold_steps": self.success_hold_steps,
        }
        self.terminations.unsafe_slam = None
        self.terminations.closed_on_target_table = DoneTerm(
            func=mdp.joint_near_init_and_contact,
            params={
                "sensor_names": target_contact_sensor_names,
                "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[TOP_HINGE]),
                "max_progress": self.closed_contact_max_progress,
                "force_threshold": self.target_contact_force_threshold,
            },
        )

        # Replace loading-stage reward with a sparse bonus for this task.
        self.rewards.stage_bonus = RewTerm(
            func=mdp.joint_moved_and_pose_near_goal,
            weight=5.0,
            params={
                key: value
                for key, value in self.terminations.success.params.items()
                if key
                not in {
                    "support_sensor_names",
                    "robot_sensor_names",
                    "forbidden_box_zones",
                    "forbidden_asset_cfg",
                    "support_force_threshold",
                    "robot_force_threshold",
                    "hold_steps",
                }
            },
        )

        @configclass
        class GoalObsCfg(ObsGroup):
            goal_pos_b = ObsTerm(
                func=mdp.asset_pos_b,
                params={"asset_cfg": SceneEntityCfg("stapler_goal")},
                noise=Unoise(n_min=-0.0, n_max=0.0),
            )
            goal_quat_b = ObsTerm(
                func=mdp.object_quat_b,
                params={"object_cfg": SceneEntityCfg("stapler_goal")},
                noise=Unoise(n_min=-0.0, n_max=0.0),
            )

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 0

        self.observations.goal = GoalObsCfg()

        hx, hy = TARGET_TABLE_HALF_EXTENTS
        marker_z = self.goal_corner_marker_radius - self.articulation_half_height_est
        corners = [(sx * hx, sy * hy, marker_z) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
        self.add_scene_vis_term(
            "stapler_goal_corners",
            ObsTerm(
                func=mdp.local_points_vis,
                params={
                    "asset_cfg": SceneEntityCfg("stapler_goal"),
                    "local_points": corners,
                    "radius": self.goal_corner_marker_radius,
                    "color": (0.15, 0.75, 0.25),
                    "opacity": 1.0,
                    "prim_path": "/Visuals/StaplerGoalCorners",
                    "hide_from_cameras": False,
                },
            ),
        )
