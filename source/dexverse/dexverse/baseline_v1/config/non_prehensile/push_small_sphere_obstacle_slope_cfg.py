# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for pushing a small sphere uphill while avoiding obstacles.

Ten thin vertical rods are sampled uniformly over the full usable slope.
``mdp.reset_rod_obstacles_sequential`` keeps every new rod at least one sphere
diameter plus margin away from the surfaces of all earlier rods. No strata,
corridor, or other deliberately safe path is reserved.
"""

import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from ..robot_init import set_robot_wrist_init_world_pos

SMALL_SPHERE_RADIUS = 0.08
SMALL_SPHERE_MASS_KG = 0.50
SMALL_SPHERE_STATIC_FRICTION = 0.3
SMALL_SPHERE_DYNAMIC_FRICTION = 0.2
TARGET_OPACITY = 0.30
# Sphere start / goal lateral randomization (m). Reset audit (Aug 2026): both
# used to be (almost) fixed on the centerline (start y +-0.04, goal fixed);
# they now slide along y independently. Side guards sit at |y| = 0.47 and
# the sphere radius is 0.08, so +-0.30 keeps a >= 9 cm margin to the guards.
RESET_Y_RANGE = 0.30
GOAL_Y_RANGE = 0.30
# Success needs the ball past the goal's x/z line AND within this lateral
# band of the goal's y (None -> y-agnostic, the pre-audit behaviour).
GOAL_Y_TOLERANCE_M: float | None = 0.15
SUCCESS_THRESHOLD_M = 0.08
TARGET_LINE_TOLERANCE_M = 0.0
BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.6
# Initial palm world position in m. Drives a joint translation on floating
# Shadow/Leap and an arm IK target on UR10e (= floating_shadow base + 0.12).
ROBOT_INIT_PALM_WORLD_X = -0.63

SLOPE_LENGTH = 0.80
SLOPE_WIDTH = 0.90
SLOPE_THICKNESS = 0.04
SLOPE_ANGLE_DEG = 15.0
SLOPE_ANGLE_RAD = math.radians(SLOPE_ANGLE_DEG)
SLOPE_CENTER_X = 0.18
SLOPE_CENTER_Y = 0.0
SLOPE_HEIGHT_OFFSET = -0.04
GOAL_OFFSET_FROM_TOP_M = 0.12
START_OFFSET_FROM_SLOPE_M = 0.10
SIDE_GUARD_THICKNESS = 0.04
SIDE_GUARD_HEIGHT = 0.16
SIDE_GUARD_COLOR = (0.30, 0.30, 0.35)

MAX_NUM_OBSTACLES = 10
OBSTACLE_RADIUS = 0.020
# Keep enough height to block the sphere reliably; "smaller" refers to the
# rod footprint, which is what allows more non-clustered samples.
OBSTACLE_HEIGHT = 0.10
GOAL_TANGENT = 0.5 * SLOPE_LENGTH - GOAL_OFFSET_FROM_TOP_M
OBSTACLE_TANGENT_RANGE = (-0.30, GOAL_TANGENT - OBSTACLE_RADIUS - 0.08)
# Keep a sphere-sized *free gap* between rod surfaces, plus a small margin.
# Centre spacing therefore includes the diameter of both rods as well as the
# sphere. The margin avoids marginal contacts from wedging the sphere.
OBSTACLE_GAP_MARGIN = 0.015
OBSTACLE_MIN_CENTER_SPACING = 2.0 * SMALL_SPHERE_RADIUS + 2.0 * OBSTACLE_RADIUS + OBSTACLE_GAP_MARGIN
OBSTACLE_CLEARANCE_M = 0.02
OBSTACLE_LATERAL_LIMIT = 0.5 * SLOPE_WIDTH - OBSTACLE_CLEARANCE_M - OBSTACLE_RADIUS
if OBSTACLE_LATERAL_LIMIT <= 0.0:
    raise ValueError("Slope is too narrow to keep obstacle clearance from the side guards.")
OBSTACLE_COLOR = (0.25, 0.25, 0.30)

SLOPE_TANGENT = (math.cos(SLOPE_ANGLE_RAD), 0.0, math.sin(SLOPE_ANGLE_RAD))
SLOPE_NORMAL = (-math.sin(SLOPE_ANGLE_RAD), 0.0, math.cos(SLOPE_ANGLE_RAD))
SLOPE_QUAT = (math.cos(SLOPE_ANGLE_RAD / 2.0), 0.0, -math.sin(SLOPE_ANGLE_RAD / 2.0), 0.0)


def _surface_centerline_point(
    slope_center: tuple[float, float, float],
    tangent_scale: float,
    normal_scale: float,
) -> tuple[float, float, float]:
    """Return a point on the slope centerline using tangent/normal offsets."""
    return (
        slope_center[0] + tangent_scale * SLOPE_TANGENT[0] + normal_scale * SLOPE_NORMAL[0],
        slope_center[1] + tangent_scale * SLOPE_TANGENT[1] + normal_scale * SLOPE_NORMAL[1],
        slope_center[2] + tangent_scale * SLOPE_TANGENT[2] + normal_scale * SLOPE_NORMAL[2],
    )


def _make_obstacle_cfg(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CylinderCfg(
            radius=OBSTACLE_RADIUS,
            height=OBSTACLE_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=OBSTACLE_COLOR, roughness=0.9),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=SLOPE_QUAT),
    )


def _make_side_guard_cfg(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(SLOPE_LENGTH, SIDE_GUARD_THICKNESS, SIDE_GUARD_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=SIDE_GUARD_COLOR, roughness=0.9),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=SLOPE_QUAT),
    )


SMALL_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.SphereCfg(
        radius=SMALL_SPHERE_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        # Low-friction surface so the sphere slides readily across the hand.
        # ``min`` ensures a grippy fingertip material cannot dominate the pair.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=SMALL_SPHERE_STATIC_FRICTION,
            dynamic_friction=SMALL_SPHERE_DYNAMIC_FRICTION,
            restitution=0.0,
            friction_combine_mode="min",
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=SMALL_SPHERE_MASS_KG),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.55, 0.18)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)

SLOPE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Slope",
    spawn=sim_utils.CuboidCfg(
        size=(SLOPE_LENGTH, SLOPE_WIDTH, SLOPE_THICKNESS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.35, 0.25), roughness=0.9),
        visible=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=SLOPE_QUAT),
)

SIDE_GUARD_LEFT_CFG = _make_side_guard_cfg("SideGuardLeft")
SIDE_GUARD_RIGHT_CFG = _make_side_guard_cfg("SideGuardRight")
OBSTACLE_CFGS = tuple(_make_obstacle_cfg(f"Obstacle{obstacle_idx}") for obstacle_idx in range(MAX_NUM_OBSTACLES))


def object_passed_target_xz_line(
    env,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    tolerance: float = TARGET_LINE_TOLERANCE_M,
    y_tolerance: float | None = GOAL_Y_TOLERANCE_M,
) -> torch.Tensor:
    """Success once the ball has crossed the target's x and z coordinates and,
    when ``y_tolerance`` is set, is within that lateral distance of the target's y."""
    obj = env.scene[object_cfg.name]
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    obj_pos = obj.data.root_pos_w
    passed = (obj_pos[:, 0] >= target_pos[:, 0] - tolerance) & (obj_pos[:, 2] >= target_pos[:, 2] - tolerance)
    if y_tolerance is not None:
        passed &= (obj_pos[:, 1] - target_pos[:, 1]).abs() <= float(y_tolerance)
    return passed


@configclass
class PushSmallSphereObstacleSlopeCommandsCfg(dexverse_base_env.CommandsCfg):
    """Command terms for the obstacle-slope pushing task."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,  # goal marker: always on, camera-visible
        use_world_frame=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="object",
        position_only=True,
    )


@configclass
class PushSmallSphereObstacleSlopeObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for push-sphere-with-obstacles on a slope.

    Sphere position / orientation and the (kinematic) obstacle states in
    ``proprio``; sphere velocities in ``privileged``. Commanded goal pose
    lives in ``goal``.
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        # Obstacle states are kinematic (velocities = 0) so we keep them as
        # the original body_state_b 13-vec rather than splitting per axis.
        obstacle_0_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_0"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_1_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_1"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_2_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_2"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_3_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_3"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_4_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_4"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_5_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_5"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_6_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_6"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_7_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_7"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_8_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_8"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_9_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_9"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))

    @configclass
    class GoalObsCfg(ObsGroup):
        target_object_pose_b = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class PushSmallSphereObstacleSlopeRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for the obstacle-slope pushing task."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 5.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            # Mean over fingertips: the summed default saturates to ~0 with
            # the five-fingertip Shadow hand.
            "reduce": "mean",
        },
        weight=1.5,
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=5.0,
        params={
            "std": 0.14,
            "command_name": "object_pose",
        },
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=10.0,
        params={
            "pos_std": SUCCESS_THRESHOLD_M,
            "rot_std": None,
            "command_name": "object_pose",
        },
    )


@configclass
class PushSmallSphereObstacleSlopeTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for the obstacle-slope pushing task."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=object_passed_target_xz_line,
        params={
            "command_name": "object_pose",
            "tolerance": TARGET_LINE_TOLERANCE_M,
            "y_tolerance": GOAL_Y_TOLERANCE_M,
        },
    )


@configclass
class PushSmallSphereObstacleSlopeEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for the obstacle-slope task."""

    reset_obstacles = EventTerm(
        func=mdp.reset_rod_obstacles_sequential,
        mode="reset",
        params={
            "obstacle_asset_cfgs": tuple(SceneEntityCfg(f"obstacle_{i}") for i in range(MAX_NUM_OBSTACLES)),
            "slope_center": (SLOPE_CENTER_X, SLOPE_CENTER_Y, 0.0),
            "slope_quat": SLOPE_QUAT,
            "tangent_range": OBSTACLE_TANGENT_RANGE,
            "lateral_limit": OBSTACLE_LATERAL_LIMIT,
            "obstacle_radius": OBSTACLE_RADIUS,
            "normal_offset": 0.5 * SLOPE_THICKNESS + 0.5 * OBSTACLE_HEIGHT,
            "min_center_spacing": OBSTACLE_MIN_CENTER_SPACING,
        },
    )


@configclass
class PushSmallSphereObstacleSlopeSceneCfg(dexverse_base_env.SceneCfg):
    object: RigidObjectCfg = SMALL_OBJECT_CFG
    slope: RigidObjectCfg = SLOPE_CFG
    side_guard_left: RigidObjectCfg = SIDE_GUARD_LEFT_CFG
    side_guard_right: RigidObjectCfg = SIDE_GUARD_RIGHT_CFG
    obstacle_0: RigidObjectCfg = OBSTACLE_CFGS[0]
    obstacle_1: RigidObjectCfg = OBSTACLE_CFGS[1]
    obstacle_2: RigidObjectCfg = OBSTACLE_CFGS[2]
    obstacle_3: RigidObjectCfg = OBSTACLE_CFGS[3]
    obstacle_4: RigidObjectCfg = OBSTACLE_CFGS[4]
    obstacle_5: RigidObjectCfg = OBSTACLE_CFGS[5]
    obstacle_6: RigidObjectCfg = OBSTACLE_CFGS[6]
    obstacle_7: RigidObjectCfg = OBSTACLE_CFGS[7]
    obstacle_8: RigidObjectCfg = OBSTACLE_CFGS[8]
    obstacle_9: RigidObjectCfg = OBSTACLE_CFGS[9]


@configclass
class PushSmallSphereObstacleSlopeEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Push a smaller sphere uphill while navigating around randomized obstacles."""

    supports_object_pose_command: bool = True

    commands: PushSmallSphereObstacleSlopeCommandsCfg = PushSmallSphereObstacleSlopeCommandsCfg()
    observations: PushSmallSphereObstacleSlopeObservationsCfg = PushSmallSphereObstacleSlopeObservationsCfg()
    rewards: PushSmallSphereObstacleSlopeRewardsCfg = PushSmallSphereObstacleSlopeRewardsCfg()
    terminations: PushSmallSphereObstacleSlopeTerminationsCfg = PushSmallSphereObstacleSlopeTerminationsCfg()
    events: PushSmallSphereObstacleSlopeEventCfg = PushSmallSphereObstacleSlopeEventCfg()
    scene: PushSmallSphereObstacleSlopeSceneCfg = PushSmallSphereObstacleSlopeSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=SMALL_OBJECT_CFG,
        slope=SLOPE_CFG,
        side_guard_left=SIDE_GUARD_LEFT_CFG,
        side_guard_right=SIDE_GUARD_RIGHT_CFG,
        obstacle_0=OBSTACLE_CFGS[0],
        obstacle_1=OBSTACLE_CFGS[1],
        obstacle_2=OBSTACLE_CFGS[2],
        obstacle_3=OBSTACLE_CFGS[3],
        obstacle_4=OBSTACLE_CFGS[4],
        obstacle_5=OBSTACLE_CFGS[5],
        obstacle_6=OBSTACLE_CFGS[6],
        obstacle_7=OBSTACLE_CFGS[7],
        obstacle_8=OBSTACLE_CFGS[8],
        obstacle_9=OBSTACLE_CFGS[9],
    )

    def __post_init__(self):
        super().__post_init__()

        set_robot_wrist_init_world_pos(self, x=ROBOT_INIT_PALM_WORLD_X)

        self.episode_length_s = 20.0
        self.commands.object_pose.resampling_time_range = (self.episode_length_s + 1.0, self.episode_length_s + 1.0)
        self.commands.object_pose.position_only = True
        self.commands.object_pose.use_world_frame = True

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        slope_center_z = (
            table_top_z
            + 0.5 * (SLOPE_THICKNESS * math.cos(SLOPE_ANGLE_RAD) + SLOPE_LENGTH * math.sin(SLOPE_ANGLE_RAD))
            + SLOPE_HEIGHT_OFFSET
        )
        slope_center = (SLOPE_CENTER_X, SLOPE_CENTER_Y, slope_center_z)
        self.scene.slope.init_state.pos = slope_center
        self.scene.slope.init_state.rot = SLOPE_QUAT
        self.events.reset_obstacles.params["slope_center"] = slope_center
        side_guard_center_y = 0.5 * SLOPE_WIDTH + 0.5 * SIDE_GUARD_THICKNESS
        side_guard_normal_offset = 0.5 * SLOPE_THICKNESS + 0.5 * SIDE_GUARD_HEIGHT
        for side_guard, center_y in (
            (self.scene.side_guard_left, side_guard_center_y),
            (self.scene.side_guard_right, -side_guard_center_y),
        ):
            side_guard.init_state.pos = _surface_centerline_point(
                slope_center=slope_center,
                tangent_scale=0.0,
                normal_scale=side_guard_normal_offset,
            )
            side_guard.init_state.pos = (
                side_guard.init_state.pos[0],
                slope_center[1] + center_y,
                side_guard.init_state.pos[2],
            )
            side_guard.init_state.rot = SLOPE_QUAT

        lower_surface_point = _surface_centerline_point(
            slope_center=slope_center,
            tangent_scale=-0.5 * SLOPE_LENGTH,
            normal_scale=0.5 * SLOPE_THICKNESS,
        )
        sphere_start = (
            lower_surface_point[0] - START_OFFSET_FROM_SLOPE_M,
            lower_surface_point[1],
            table_top_z + SMALL_SPHERE_RADIUS,
        )
        self.scene.object.init_state.pos = sphere_start
        self.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        goal_surface_point = _surface_centerline_point(
            slope_center=slope_center,
            tangent_scale=0.5 * SLOPE_LENGTH - GOAL_OFFSET_FROM_TOP_M,
            normal_scale=0.5 * SLOPE_THICKNESS,
        )
        goal_center = (
            goal_surface_point[0] + SMALL_SPHERE_RADIUS * SLOPE_NORMAL[0],
            goal_surface_point[1] + SMALL_SPHERE_RADIUS * SLOPE_NORMAL[1],
            goal_surface_point[2] + SMALL_SPHERE_RADIUS * SLOPE_NORMAL[2],
        )
        self.commands.object_pose.ranges.pos_x = (goal_center[0], goal_center[0])
        # goal slides along the slope's lateral axis (sampled once per episode)
        self.commands.object_pose.ranges.pos_y = (goal_center[1] - GOAL_Y_RANGE, goal_center[1] + GOAL_Y_RANGE)
        self.commands.object_pose.ranges.pos_z = (goal_center[2], goal_center[2])

        self.commands.object_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=SMALL_SPHERE_RADIUS,
                    visible=True,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.20, 0.60, 0.90),
                        opacity=TARGET_OPACITY,
                    ),
                )
            },
        )
        self.commands.object_pose.success_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "failure": sim_utils.SphereCfg(
                    radius=SMALL_SPHERE_RADIUS,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.55, 0.18)),
                ),
                "success": sim_utils.SphereCfg(
                    radius=SMALL_SPHERE_RADIUS,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.80, 0.25)),
                ),
            },
        )

        if self.events.reset_object is not None:
            self.events.reset_object.params["pose_range"] = {
                "x": [0.0, 0.0],
                "y": [-RESET_Y_RANGE, RESET_Y_RANGE],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-math.pi, math.pi],
            }

        # Nominal poses are used before the first reset; subsequent resets use
        # the sequential 2-D rod sampler above.
        obstacle_normal_offset = 0.5 * SLOPE_THICKNESS + 0.5 * OBSTACLE_HEIGHT
        tangent_span = OBSTACLE_TANGENT_RANGE[1] - OBSTACLE_TANGENT_RANGE[0]
        for obstacle_idx in range(MAX_NUM_OBSTACLES):
            tangent = OBSTACLE_TANGENT_RANGE[0] + (obstacle_idx + 0.5) * tangent_span / MAX_NUM_OBSTACLES
            lateral = -0.25 if obstacle_idx % 2 == 0 else 0.25
            obstacle = getattr(self.scene, f"obstacle_{obstacle_idx}")
            obstacle.init_state.pos = _surface_centerline_point(
                slope_center=slope_center,
                tangent_scale=tangent,
                normal_scale=obstacle_normal_offset,
            )
            obstacle.init_state.pos = (
                obstacle.init_state.pos[0],
                slope_center[1] + lateral,
                obstacle.init_state.pos[2],
            )
            obstacle.init_state.rot = SLOPE_QUAT

        if self.terminations.object_out_of_bound is not None:
            table_size = self.scene.table.spawn.size
            half_x = table_size[0] * 0.5
            half_y = table_size[1] * 0.5
            self.terminations.object_out_of_bound.params["in_bound_range"] = {
                "x": (-half_x, half_x),
                "y": (-half_y, half_y),
                "z": (BOUND_Z_MIN, BOUND_Z_MAX),
            }

        mdp.setup_fingertip_contact_observation(self)
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names


@configclass
class PushSmallSphereObstacleSlopeEnvFloatingDexHandRightCfg(PushSmallSphereObstacleSlopeEnvCfg):
    """Obstacle-slope pushing config for floating dex hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
