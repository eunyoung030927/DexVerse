# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functional manipulation: hammer strike.

The robot must grasp the hammer by its handle and drive a nail into a board.
The target is a nail-on-a-board articulation (a single prismatic joint along
``-Z`` so positive joint values mean the nail is being pushed *into* the
board) authored at ``DEXVERSE_AUTHORED_ARTICULATIONS_DIR/nail_board``.

Success requires:

- Forbidden-zone clearance on the hammer (fingertips off the hammer head).
- Affinity-zone overlap between the hammer head and the nail.
- Compression of the nail's prismatic joint past a tunable threshold.

Nail dynamics:
    Gravity is disabled for the nail and the prismatic joint has zero
    stiffness, only a small damping term. A strike imparts a velocity
    impulse along the joint axis; damping bleeds the velocity off and the
    nail settles at whatever displacement was reached. Tune
    ``nail_board_damping`` if the nail moves too easily (raise) or strikes
    barely register (lower). The previous high-stiffness +
    ``velocity_limit_sim`` "break-through" mechanic has been removed.

Hammer stand (small table):
    Like the retrieve-tray rack, the hammer starts on furniture rather than
    a solid block: a procedural small table (top + four legs,
    :mod:`dexverse.assets.small_table_builder`) so the episode begins with
    an affordance (hammer on a stand) before the strike. The table is
    kinematic; ``table_clearance`` is set to its top-face height so the
    hammer rests on the board. At each reset the table is snapped under the
    hammer via ``mdp.sync_object`` (world-aligned, with a per-env xy jitter),
    so the hammer — whose world xy and yaw are randomized by ``reset_object``
    — starts flat at a random position and heading on the stand top.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from dexverse.assets import DEXVERSE_AUTHORED_ARTICULATIONS_DIR
from dexverse.assets.small_table_builder import ensure_small_table_asset
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..grasping.forbidden_zones import ForbiddenZone, split_zones
from .base_cfg import (
    YCB_DIR,
    FunctionalGraspingEnvCfg,
    FunctionalGraspingEnvFloatingDexHandRightCfg,
    FunctionalGraspingObservationsCfg,
)
from .rewards import DenseHammerStrikeRewardsCfg

HAMMER_USD_PATH = str(YCB_DIR / "048_hammer_usd" / "048_hammer.usd")

# Head-down vertical pose: -90° about world +y rotates the asset's local +x
# (handle axis) onto world -z, putting the head face flat on the tabletop and
# the handle pointing straight up. Quaternion is (w, x, y, z).

HAMMER_ROT_INIT = (1, 0, 0, 0)
HAMMER_MASS = 0.6

NAIL_BOARD_USD_PATH = str(DEXVERSE_AUTHORED_ARTICULATIONS_DIR / "nail_board" / "nail_board.usd")
NAIL_BOARD_DEFAULT_SCALE = (1.0, 1.0, 1.0)
# Half thickness of the board in metres at unit scale (URDF box is 0.10 m
# tall, frame at centre); apply the asset scale to recover the actual
# half-height. The URDF was scaled 2x at source rather than via USD
# scale because USD scaling did not also scale the prismatic joint range.
NAIL_BOARD_UNIT_HALF_HEIGHT = 0.05
NAIL_BOARD_HALF_HEIGHT = NAIL_BOARD_UNIT_HALF_HEIGHT * NAIL_BOARD_DEFAULT_SCALE[2]
NAIL_PRISMATIC_JOINT = "nail_prismatic"
# Nail head centre, at q=0, expressed in the (unscaled) board root frame
# (URDF: board top at z=+0.05, nail head visual centred at +0.084 above
# the nail link origin which sits on the board top).
NAIL_HEAD_UNIT_Z = 0.05 + 0.084
# Default forbidden-zone sphere radius around the nail head (in metres,
# applied in the *scaled* board frame). Tuned so a fingertip touching the
# head is rejected but the hammer head can still strike from above.
NAIL_HEAD_FORBIDDEN_RADIUS = 0.025

# Scene field name for the nail-on-a-board target.
TARGET_KEY = "hammer_target"

# Procedural small table the hammer rests on (retrieve-tray-style affordance).
# Large square top so the downscaled hammer sits fully on it with fixed yaw.
# Offset (0, 0) centres the table under the hammer root. Oak MDL matches the
# main table legs / drill workpiece.
HAMMER_STAND_TABLE_PARAMS = dict(
    depth=0.40,
    width=0.40,
    height=0.06,
    top_thickness=0.012,
    leg_size=0.022,
)
HAMMER_STAND_USD_PATH, HAMMER_STAND_MANIFEST = ensure_small_table_asset(
    "hammer_stand_table",
    **HAMMER_STAND_TABLE_PARAMS,
)
HAMMER_STAND_TOP_HEIGHT = float(HAMMER_STAND_MANIFEST["z_top"])
HAMMER_STAND_DEFAULT_XY_OFFSET = (0.0, 0.0)
HAMMER_STAND_MATERIAL_PATH = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak.mdl"


def _make_hammer_stand_spawner(
    *,
    static_friction: float,
    dynamic_friction: float,
    friction_combine_mode: str,
):
    """UsdFileCfg cannot take ``physics_material`` (shape spawners only).

    Wrap the usual USD spawn and bind a PhysX material onto the stand's
    collision prims so friction / combine-mode actually take effect.
    """
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from isaaclab.sim.utils import bind_physics_material, clone
    from isaaclab.sim import schemas

    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
        friction_combine_mode=friction_combine_mode,
    )

    @clone
    def spawn(prim_path, cfg, *args, **kwargs):
        prim = spawn_from_usd(prim_path, cfg, *args, **kwargs)
        path = prim.GetPath().pathString
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(path, cfg.rigid_props)
        if cfg.collision_props is not None:
            schemas.define_collision_properties(path, cfg.collision_props)
        if cfg.mass_props is not None:
            schemas.define_mass_properties(path, cfg.mass_props)
        mat_path = f"{path}/Looks/PhysicsMaterial"
        material_cfg.func(mat_path, material_cfg)
        bind_physics_material(path, mat_path)
        return prim

    return spawn


def _make_nail_board_articulation_cfg(
    *,
    init_pos: tuple[float, float, float],
    scale: tuple[float, float, float],
    stiffness: float,
    damping: float,
    effort_limit_sim: float,
    velocity_limit_sim: float,
) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/HammerTarget",
        spawn=sim_utils.UsdFileCfg(
            usd_path=NAIL_BOARD_USD_PATH,
            scale=scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={NAIL_PRISMATIC_JOINT: 0.0},
        ),
        actuators={
            "nail_spring": ImplicitActuatorCfg(
                joint_names_expr=[NAIL_PRISMATIC_JOINT],
                stiffness=stiffness,
                damping=damping,
                effort_limit_sim=effort_limit_sim,
                velocity_limit_sim=velocity_limit_sim,
            ),
        },
    )


@configclass
class HammerStrikeSceneCfg(FunctionalGraspingEnvCfg.FunctionalGraspingSceneCfg):
    """Scene with the hammer ``object``, the nail-on-a-board target, and a
    small table stand lifting the hammer off the tabletop."""

    # Placeholder; the actual init pose / actuator gains are written in
    # ``HammerStrikeEnvCfg.__post_init__`` so subclasses can tweak them.
    hammer_target: ArticulationCfg = _make_nail_board_articulation_cfg(
        init_pos=(0.0, 0.0, 0.0),
        scale=NAIL_BOARD_DEFAULT_SCALE,
        stiffness=0.0,
        damping=0.1,
        effort_limit_sim=1.0,
        velocity_limit_sim=1000.0,
    )


@configclass
class HammerStrikeObservationsCfg(FunctionalGraspingObservationsCfg):
    """Observation layout for hammer strike without an ``object_pose`` command."""

    @configclass
    class StateObsCfg(FunctionalGraspingObservationsCfg.StateObsCfg):
        # Current nail press depth (a joint position) — observable state.
        nail_press = ObsTerm(
            func=mdp.max_joint_pos_signed,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "asset_cfg": SceneEntityCfg(TARGET_KEY, joint_names=[NAIL_PRISMATIC_JOINT]),
            },
        )

    @configclass
    class GoalObsCfg(ObsGroup):
        goal_pos_b = ObsTerm(
            func=mdp.asset_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg(TARGET_KEY)},
        )
        required_press_distance = ObsTerm(func=mdp.scalar_obs, params={"value": 0.0})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    state: StateObsCfg = StateObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class HammerStrikeEnvFloatingDexHandRightCfg(FunctionalGraspingEnvFloatingDexHandRightCfg):
    """Hammer strike: grasp + affinity overlap + nail compression success."""

    usd_path: str = HAMMER_USD_PATH
    object_mass: float = HAMMER_MASS
    # Slightly under unit scale so the full hammer fits on the stand top.
    object_scale: tuple[float, float, float] = (0.95, 0.95, 0.95)
    # Vertical (head-down) pose: distance from the asset origin to the head's
    # bottom face is roughly half the hammer's total length. Tune in the
    # viewer if the hammer floats above or sinks into the table.
    object_half_height: float = 0.0  # reset audit (Aug 2026): seated ~3 mm above rest instead of dropping (was 0.02, dropped 2.5 cm)
    object_static_friction: float | None = 2.0
    object_dynamic_friction: float | None = 2.0
    object_friction_combine_mode: str = "average"
    object_init_rot: tuple[float, float, float, float] = HAMMER_ROT_INIT
    object_collision_enabled: bool = False

    # Spawn a bit toward -x of the table centre (near the hand side, but not
    # so close that the stand crowds the palm). Both the hammer's world xy and
    # its yaw are randomized; the stand stays world-aligned with the main
    # table, so the yaw jitter also randomizes the hammer's heading *on* the
    # stand top. ``hammer_on_stand_xy_range`` below adds the translational
    # randomization relative to the stand.
    object_init_x_offset: float = -0.05
    object_init_y_offset: float = -0.15
    object_reset_x_range: tuple[float, float] = (-0.15, 0.05)
    object_reset_y_range: tuple[float, float] = (-0.25, -0.05)
    # Full-circle yaw: the handle may point in any direction on the stand.
    object_reset_yaw_range: tuple[float, float] = (-3.14, 3.14)
    # object_reset_roll_range: tuple[float, float] = (-1.57,-1.57)
    # object_reset_pitch_range: tuple[float, float] = (0.3,0.3)
    # Nail-on-a-board target placement (offsets from the table centre).
    nail_board_init_x_offset: float = 0.0
    nail_board_init_y_offset: float = 0.2
    nail_board_reset_x_range: tuple[float, float] = (-0.05, 0.2)
    nail_board_reset_y_range: tuple[float, float] = (0.0, 0.1)
    nail_board_scale: tuple[float, float, float] = NAIL_BOARD_DEFAULT_SCALE
    nail_board_half_height: float = NAIL_BOARD_HALF_HEIGHT

    # Damping-only actuator: zero stiffness (no restoring spring), small
    # damping bleeds off post-strike velocity, no velocity cap so a strike
    # can actually move the nail. See module docstring.
    nail_board_stiffness: float = 0.0
    nail_board_damping: float = 0.3
    nail_board_effort_limit_sim: float = 1.0
    nail_board_velocity_limit_sim: float = 10.0

    # (Hammer-frame) forbidden zones intentionally empty: success is gated on
    # the nail-prismatic joint compression plus the target-frame forbidden
    # zone below. Re-populate ``forbidden_zones`` to add a
    # hand-stay-off-the-hammer-head requirement.
    forbidden_zones: tuple[ForbiddenZone, ...] = ()

    # Forbidden zones in the *target asset* (nail+board) local frame. Kept for the
    # debug visualiser only: since Aug 2026 the "do not touch the nail" rule is a
    # *termination* (``nail_touched`` below, attached to the moving nail body)
    # rather than a success gate -- gating success only meant "press it with a
    # finger, retract, still succeed".
    target_forbidden_zones: tuple[ForbiddenZone, ...] = (
        ForbiddenZone(
            kind="sphere",
            center=(0.0, 0.0, NAIL_HEAD_UNIT_Z * NAIL_BOARD_DEFAULT_SCALE[2]),
            radius=NAIL_HEAD_FORBIDDEN_RADIUS,
        ),
    )
    # Failure termination: any robot body within ``nail_touch_radius`` of the nail
    # head centre (nail link frame, ``nail_head_local_offset``) ends the episode.
    # 3.5 cm ~ nail head radius + fingertip radius; the hammer is a separate rigid
    # body, so striking with it never trips this.
    nail_touch_radius: float = 0.035
    nail_head_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.084 * NAIL_BOARD_DEFAULT_SCALE[2])
    nail_body_name: str = "nail"

    # Failure termination: the hammer *handle* touched the nail (a handle strike
    # instead of a head strike). The nail head centre is tested against a capsule
    # around the handle's centre line in the hammer's local frame. Geometry from
    # the scaled (0.95) mesh: the handle runs diagonally from (+0.032, -0.175) at
    # its end to (-0.046, +0.035) where the head begins (head spans y in
    # [0.04, 0.14], striking face at x = -0.122), z-centre 0.015, radius ~0.017.
    # The capsule stops 1.5 cm short of the head; radius = handle r (0.017) +
    # nail head r (0.025) + margin. Validated on the 0821 demos: a proper head
    # strike never brings the nail head closer than 0.074 m to this segment.
    handle_touch_radius: float = 0.05
    handle_segment_local_start: tuple[float, float, float] = (0.032, -0.175, 0.015)
    handle_segment_local_end: tuple[float, float, float] = (-0.042, 0.020, 0.015)

    # Small table (top + legs) the hammer rests on. Top height is the
    # procedural asset's ``z_top`` (also used as ``table_clearance``).
    # ``hammer_stand_xy_offset`` is in the hammer's local frame; the table
    # inherits hammer yaw via ``quat_local``.
    hammer_stand_height: float = HAMMER_STAND_TOP_HEIGHT
    hammer_stand_xy_offset: tuple[float, float] = HAMMER_STAND_DEFAULT_XY_OFFSET
    # Per-axis uniform xy jitter (m, world frame) applied to the stand when it
    # snaps under the already-reset hammer. Shifting the stand under the
    # hammer is equivalent to spawning the hammer at a random spot on the
    # stand top. At +/-0.07 with full yaw the slim end of the hammer can
    # overhang the 0.40 m top by a few cm at extreme draws, but the centre
    # of mass stays far inside the top face, so the hammer never tips.
    hammer_on_stand_xy_range: tuple[float, float] = (-0.07, 0.07)
    # Low stand friction so the hammer can slide on the top. Use ``min``
    # combine so these values win against the hammer's higher friction (2.0);
    # with ``average`` the contact would still feel sticky.
    hammer_stand_static_friction: float = 0.3
    hammer_stand_dynamic_friction: float = 0.3
    hammer_stand_friction_combine_mode: str = "min"

    # Press distance (m) the nail must travel relative to its init pose.
    press_distance_m: float = 0.06
    # ``displacement`` measures abs(q - q_init); ``ratio`` uses joint range.
    press_mode: str = "displacement"

    # Hammer-head centre in the hammer's local frame (metres, scaled asset).
    # Handle along local +x, head toward -x. Tuned for ``object_scale`` ~0.95.
    hammer_head_local_offset: tuple[float, float, float] = (-0.095, 0.0, 0.0)

    # Minimum xy distance (m) the nail-board target must keep from the
    # hammer spawn. Enforced via rejection sampling at reset time.
    min_object_goal_xy_distance: float = 0.10

    # Override the scene to include the nail-board target articulation.
    scene: HammerStrikeSceneCfg = HammerStrikeSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
    )
    observations: HammerStrikeObservationsCfg = HammerStrikeObservationsCfg()
    rewards: DenseHammerStrikeRewardsCfg = DenseHammerStrikeRewardsCfg()

    def __post_init__(self):
        # Lift the hammer onto the small-table top (read by the base
        # ``__post_init__``). Asset frame origin is the footprint bottom.
        self.table_clearance = self.hammer_stand_height
        super().__post_init__()

        # Position the nail-board on top of the table at the configured offset.
        # Board frame is at the *centre* of the board, so add half the board
        # thickness to land its bottom face on the table top.
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        table_pos = self.scene.table.init_state.pos

        # Spawn the kinematic small table under the hammer (same role as the
        # retrieve-tray rack: furniture that creates a pre-grasp affordance).
        # Initial xy is nominal; ``reset_hammer_stand`` snaps it to the
        # post-reset hammer pose. Root z = main-table top (bottom-origin USD).
        stand_x = table_pos[0] + self.object_init_x_offset + self.hammer_stand_xy_offset[0]
        stand_y = table_pos[1] + self.object_init_y_offset + self.hammer_stand_xy_offset[1]
        self.scene.hammer_stand = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/HammerStand",
            spawn=sim_utils.UsdFileCfg(
                func=_make_hammer_stand_spawner(
                    static_friction=self.hammer_stand_static_friction,
                    dynamic_friction=self.hammer_stand_dynamic_friction,
                    friction_combine_mode=self.hammer_stand_friction_combine_mode,
                ),
                usd_path=str(HAMMER_STAND_USD_PATH),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                # Collision is authored per box prim in the generated USD.
                collision_props=None,
                mass_props=None,
                # Same Oak wood as the main table legs / drill workpiece.
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path=HAMMER_STAND_MATERIAL_PATH,
                    project_uvw=True,
                    texture_scale=(0.5, 0.5),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(stand_x, stand_y, table_top_z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self.scene.hammer_target = _make_nail_board_articulation_cfg(
            init_pos=(
                table_pos[0] + self.nail_board_init_x_offset,
                table_pos[1] + self.nail_board_init_y_offset,
                table_top_z + self.nail_board_half_height,
            ),
            scale=self.nail_board_scale,
            stiffness=self.nail_board_stiffness,
            damping=self.nail_board_damping,
            effort_limit_sim=self.nail_board_effort_limit_sim,
            velocity_limit_sim=self.nail_board_velocity_limit_sim,
        )

        # Attach reset events for the nail-board (root pose + prismatic joint).
        # The excluding variant rejection-samples the nail-board xy so it
        # stays at least ``min_object_goal_xy_distance`` from the hammer's
        # already-reset xy.
        self.events.reset_hammer_target = EventTerm(
            func=mdp.reset_root_pose_uniform_excluding,
            mode="reset",
            params={
                "pose_range": {
                    "x": list(self.nail_board_reset_x_range),
                    "y": list(self.nail_board_reset_y_range),
                    "z": [0.0, 0.0],
                    "roll": [0.0, 0.0],
                    "pitch": [0.0, 0.0],
                    "yaw": [0.0, 0.0],
                },
                "asset_cfg": SceneEntityCfg(TARGET_KEY),
                "reference_asset_cfg": SceneEntityCfg("object"),
                "min_xy_distance": self.min_object_goal_xy_distance,
            },
        )
        self.events.reset_hammer_target_joints = EventTerm(
            func=mdp.reset_joints_to_init,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(TARGET_KEY, joint_names=[NAIL_PRISMATIC_JOINT]),
            },
        )

        # Snap the small table under the hammer. Use a *world* identity quat
        # (not quat_local) so the stand stays axis-aligned with the main
        # table -- two nested squares, not a diamond on a square. xy tracks
        # the hammer plus a per-env jitter (``hammer_on_stand_xy_range``) so
        # the hammer lands at a random spot on the stand top; bottom-origin
        # USD sits on the main tabletop.
        stand_z_offset = -(self.object_half_height + self.hammer_stand_height)
        self.events.reset_hammer_stand = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("hammer_stand"),
                "source_cfg": SceneEntityCfg("object"),
                "source_local_offset": (
                    self.hammer_stand_xy_offset[0],
                    self.hammer_stand_xy_offset[1],
                    0.0,
                ),
                "xy_jitter_range": {
                    "x": list(self.hammer_on_stand_xy_range),
                    "y": list(self.hammer_on_stand_xy_range),
                },
                "z_offset": stand_z_offset,
                "quat": (1.0, 0.0, 0.0, 0.0),
            },
        )

        # No object-pose goal command: the goal is encoded by affinity overlap
        # and nail compression instead (the goal-tracking reward terms are
        # ``None`` at the class level on ``DenseHammerStrikeRewardsCfg``).
        self.commands.object_pose = None
        # Wire the dense strike terms: nail-head point in the scaled board
        # frame, hammer-head point from the leaf-tunable offset, and nail
        # compression graded toward the success press distance.
        self.rewards.head_to_nail.params.update({
            "source_local_offset": self.hammer_head_local_offset,
            "target_local_offset": (0.0, 0.0, NAIL_HEAD_UNIT_Z * self.nail_board_scale[2]),
        })
        self.rewards.nail_press.params.update({
            "threshold_m": self.press_distance_m,
            "asset_cfg": SceneEntityCfg(TARGET_KEY, joint_names=[NAIL_PRISMATIC_JOINT]),
        })
        # ``goal_pos_b.asset_cfg`` and ``nail_press.asset_cfg`` are fixed to
        # ``TARGET_KEY`` at the class level (see ``HammerStrikeObservationsCfg``).
        # Only the leaf-tunable ``press_distance_m`` needs propagation.
        self.observations.goal.required_press_distance.params["value"] = self.press_distance_m

        # Replace success termination with the hammer-strike-specific success.
        # Success is instantaneous -- the nail is pressed past the threshold -- so
        # no contact/zone conditions are attached to it (Aug 2026; the target
        # zones now feed the ``nail_touched`` failure termination instead).
        sphere_zones, box_zones, cylinder_zones = split_zones(self.forbidden_zones)
        self.terminations.success = DoneTerm(
            func=mdp.hammer_strike_success,
            params={
                "press_threshold_m": self.press_distance_m,
                "press_mode": self.press_mode,
                "sphere_zones": sphere_zones,
                "box_zones": box_zones,
                "cylinder_zones": cylinder_zones,
                "target_sphere_zones": [],
                "target_box_zones": [],
                "target_cylinder_zones": [],
                "asset_cfg": self.forbidden_zone_asset_cfg(),  # every robot body (cross-embodiment)
                "object_cfg": SceneEntityCfg("object"),
                "target_asset_cfg": SceneEntityCfg(TARGET_KEY),
                "target_joint_cfg": SceneEntityCfg(TARGET_KEY, joint_names=[NAIL_PRISMATIC_JOINT]),
            },
        )
        # Operator cues (forbidden / contact zones from the functional base).
        # The nail-head no-touch sphere visual was removed; ``nail_touched``
        # below still enforces the rule without drawing a red marker.
        self.configure_debug_vis()

        # Failure: the hand touched the nail head (any robot body inside the sphere).
        self.terminations.nail_touched = DoneTerm(
            func=mdp.hand_near_body_point,
            params={
                "robot_cfg": self.forbidden_zone_asset_cfg(),  # every robot body unless the robot cfg narrows it
                "target_cfg": SceneEntityCfg(TARGET_KEY, body_names=[self.nail_body_name]),
                "local_offset": tuple(self.nail_head_local_offset),
                "radius": self.nail_touch_radius,
            },
        )

        # Failure: the hammer handle touched the nail head (handle strike).
        self.terminations.handle_touched_nail = DoneTerm(
            func=mdp.body_point_near_object_segment,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "target_cfg": SceneEntityCfg(TARGET_KEY, body_names=[self.nail_body_name]),
                "target_local_offset": tuple(self.nail_head_local_offset),
                "segment_start": tuple(self.handle_segment_local_start),
                "segment_end": tuple(self.handle_segment_local_end),
                "radius": self.handle_touch_radius,
            },
        )
