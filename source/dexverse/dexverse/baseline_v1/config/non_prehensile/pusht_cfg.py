# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push-T v1 with ten goal-local orange rods, 15 cm gaps and XY/yaw reachability."""

import isaaclab.sim as sim_utils
from dexverse.assets import DEXVERSE_AUTHORED_ASSETS_DIR
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
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
from ...mdp.pusht_obstacles import (
    ROD_COLOR, ROD_HEIGHT, ROD_NAMES, ROD_RADIUS,
    reset_push_t_with_rods, rod_positions_in_table,
)
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop

PUSH_T_ASSET_DIR = DEXVERSE_AUTHORED_ASSETS_DIR / "push_t"
TEE_USD_PATH = str((PUSH_T_ASSET_DIR / "tee.usda").resolve())
GOAL_TEE_USD_PATH = str((PUSH_T_ASSET_DIR / "tee_goal_2d.usda").resolve())

TEE_THICKNESS_M = 0.04
TEE_HALF_THICKNESS_M = 0.5 * TEE_THICKNESS_M
TABLE_CLEARANCE_M = 0.001
GOAL_MARKER_Z_OFFSET_M = 0.003

GOAL_OFFSET_XY = (-0.156, -0.100)

SPAWN_BOX_X_OFFSET = -0.25
SPAWN_BOX_Y_OFFSET = -0.25
SPAWN_BOX_X_LENGTH = 0.50
SPAWN_BOX_Y_LENGTH = 0.50

OVERLAP_SUCCESS_THRESHOLD = 0.70  # Rod navigation is harder; final alignment is more forgiving.
BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.5
# Non-prehensile: the tee must stay on the table. The episode fails the moment
# the tee root rises this far above its resting height — catches both lifting
# and tipping onto an edge, while leaving room for contact jitter during pushes.
LIFT_FAIL_HEIGHT_M = 0.03


def _make_rod(index):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/PushTRod{index}",
        spawn=sim_utils.CylinderCfg(
            radius=ROD_RADIUS, height=ROD_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=ROD_COLOR, roughness=0.55),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.62)),
    )


TEE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        func=dexverse_base_env.spawn_usd_with_rigid_properties,
        usd_path=TEE_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=4,
            max_contact_impulse=1.0e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

GOAL_TEE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/GoalTee",
    spawn=sim_utils.UsdFileCfg(
        func=dexverse_base_env.spawn_usd_with_rigid_properties,
        usd_path=GOAL_TEE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=4,
            max_contact_impulse=1.0e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visible=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)


@configclass
class PushTObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for the rigid push-T task.

    The state policy sees tee/goal poses and all ten rod positions. Full
    tee/goal body states (including velocity) remain in the privileged group.
    """

    @configclass
    class StateObsCfg(ObsGroup):
        tee_pose = ObsTerm(func=mdp.body_pose_b, params={
            "body_asset_cfg": SceneEntityCfg("object"), "base_asset_cfg": SceneEntityCfg("table"),
        })
        goal_pose = ObsTerm(func=mdp.body_pose_b, params={
            "body_asset_cfg": SceneEntityCfg("goal_tee"), "base_asset_cfg": SceneEntityCfg("table"),
        })
        rod_positions = ObsTerm(func=rod_positions_in_table)

    state: StateObsCfg = StateObsCfg()

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        rod_positions = ObsTerm(func=rod_positions_in_table)
        tee_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("object"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        goal_tee_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("goal_tee"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        tee_state_rel_goal = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("object"), "base_asset_cfg": SceneEntityCfg("goal_tee")},
        )

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


@configclass
class PushTRewardsCfg(dexverse_base_env.RewardsCfg):
    """Rewards for PushT task."""

    pose_dense = RewTerm(
        func=mdp.push_t_pose_dense_reward,
        weight=1.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "goal_cfg": SceneEntityCfg("goal_tee"),
            "rotation_weight": 0.5,
            "translation_weight": 0.5,
            "translation_distance_gain": 5.0,
            "overlap_success_threshold": OVERLAP_SUCCESS_THRESHOLD,
            "success_bonus": 3.0,
            "overlap_point_step": 0.005,
        },
    )


@configclass
class PushTTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for PushT task."""

    success = DoneTerm(
        func=mdp.push_t_success,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "goal_cfg": SceneEntityCfg("goal_tee"),
            "success_threshold": OVERLAP_SUCCESS_THRESHOLD,
            "overlap_point_step": 0.005,
        },
    )

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Instant failure when the tee leaves the tabletop plane (lifted or tipped
    # on an edge). x/y are left effectively unbounded — leaving the table is
    # ``object_out_of_bound``'s job. The z cap is finalized in
    # ``__post_init__`` from the tee's resting height.
    object_lifted = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1e3, 1e3), "y": (-1e3, 1e3), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PushTEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for PushT task."""

    reset_object = EventTerm(
        func=reset_push_t_with_rods,
        mode="reset",
        params={},
    )
    reset_goal_tee = None  # Joint reset above owns the goal and obstacle poses.


@configclass
class PushTSceneCfg(dexverse_base_env.SceneCfg):
    object: RigidObjectCfg = TEE_CFG
    goal_tee: RigidObjectCfg = GOAL_TEE_CFG
    push_t_rod_0: RigidObjectCfg = _make_rod(0)
    push_t_rod_1: RigidObjectCfg = _make_rod(1)
    push_t_rod_2: RigidObjectCfg = _make_rod(2)
    push_t_rod_3: RigidObjectCfg = _make_rod(3)
    push_t_rod_4: RigidObjectCfg = _make_rod(4)
    push_t_rod_5: RigidObjectCfg = _make_rod(5)
    push_t_rod_6: RigidObjectCfg = _make_rod(6)
    push_t_rod_7: RigidObjectCfg = _make_rod(7)
    push_t_rod_8: RigidObjectCfg = _make_rod(8)
    push_t_rod_9: RigidObjectCfg = _make_rod(9)


@configclass
class PushTEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """PushT task configuration (base, robot-agnostic)."""

    observations: PushTObservationsCfg = PushTObservationsCfg()
    rewards: PushTRewardsCfg = PushTRewardsCfg()
    terminations: PushTTerminationsCfg = PushTTerminationsCfg()
    events: PushTEventCfg = PushTEventCfg()
    scene: PushTSceneCfg = PushTSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=TEE_CFG,
        goal_tee=GOAL_TEE_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        # ``body_state_b`` (non-vis variant) for hand tips — the live
        # visualizer slows rendering and isn't needed here.
        if hasattr(self.observations.privileged, "hand_tips_state_b"):
            self.observations.privileged.hand_tips_state_b.func = mdp.body_state_b

        if hasattr(self.events, "object_physics_material"):
            self.events.object_physics_material = None
        if hasattr(self.events, "object_scale_mass"):
            self.events.object_scale_mass = None

        table_size = self.scene.table.spawn.size
        table_pos = self.scene.table.init_state.pos
        table_top_z = table_pos[2] + table_size[2] * 0.5
        tee_root_z = table_top_z + TEE_HALF_THICKNESS_M + TABLE_CLEARANCE_M
        goal_root_z = table_top_z + GOAL_MARKER_Z_OFFSET_M
        if self.scene.table.init_state.rot != (1.0, 0.0, 0.0, 0.0):
            raise ValueError("Push-T clearance sampler requires an axis-aligned tabletop")
        self.events.reset_object.params = {
            "table_size": tuple(table_size[:2]), "table_center": tuple(table_pos[:2]),
            "spawn_x": (GOAL_OFFSET_XY[0] + SPAWN_BOX_X_OFFSET,
                        GOAL_OFFSET_XY[0] + SPAWN_BOX_X_OFFSET + SPAWN_BOX_X_LENGTH),
            "spawn_y": (GOAL_OFFSET_XY[1] + SPAWN_BOX_Y_OFFSET,
                        GOAL_OFFSET_XY[1] + SPAWN_BOX_Y_OFFSET + SPAWN_BOX_Y_LENGTH),
        }
        for index, name in enumerate(ROD_NAMES):
            getattr(self.scene, name).init_state.pos = (
                table_pos[0] + (index % 4 - 1.5) * 0.42,
                table_pos[1] + (index // 4 - 1) * 0.42, table_top_z + ROD_HEIGHT / 2,
            )

        self.scene.object.init_state.pos = (GOAL_OFFSET_XY[0], GOAL_OFFSET_XY[1], tee_root_z)
        self.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.goal_tee.init_state.pos = (GOAL_OFFSET_XY[0], GOAL_OFFSET_XY[1], goal_root_z)
        self.scene.goal_tee.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        if self.terminations.object_out_of_bound is not None:
            half_x = table_size[0] * 0.5
            half_y = table_size[1] * 0.5
            self.terminations.object_out_of_bound.params["in_bound_range"] = {
                "x": (-half_x, half_x),
                "y": (-half_y, half_y),
                "z": (BOUND_Z_MIN, BOUND_Z_MAX),
            }

        if self.terminations.object_lifted is not None:
            self.terminations.object_lifted.params["in_bound_range"] = {
                "x": (-1e3, 1e3),
                "y": (-1e3, 1e3),
                "z": (BOUND_Z_MIN, tee_root_z + LIFT_FAIL_HEIGHT_M),
            }

        self.episode_length_s = 25  # Extra time to push around rods and realign.

        mdp.setup_fingertip_contact_observation(self)


@configclass
class PushTEnvFloatingDexHandRightCfg(PushTEnvCfg):
    """PushT environment configuration for floating dexterous hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
