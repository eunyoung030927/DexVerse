# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual: retrieve a food tray from a random level of a rack and set it
level over a target on the tabletop -- without spilling the food.

Builds on :mod:`lift_tray_cfg` (same tray + loose ``food`` prop, same Shadow
bimanual robot) and adds

* a procedural multi-level **rack** (:mod:`dexverse.assets.rack_builder`, one
  kinematic rigid body, open at the front and both sides) standing at the far
  side of the table with the opening toward the robot; x/y/yaw jitter per reset;
* a per-reset **spawn level**: :func:`mdp.reset_object_on_rack_level` puts the
  tray on a uniformly chosen shelf (manifest gives every shelf-top height), the
  food is then re-seated on the tray;
* a flat **goal pad** on the tabletop, randomised in front of the rack;
* success :func:`mdp.carrier_held_aloft_with_cargo` -- tray within
  ``goal_xy_threshold`` of the pad, held in the aerial height band, level, still, food still
  on the tray; failure :func:`mdp.cargo_off_carrier` -- food slid off / fell.

Compared to the plain lift the grasp is constrained by the shelf above/below
(only side/front access, limited clearance) and the episode needs a level carry
plus a stable aerial hold. Rewards are inherited from the bimanual lift base for
now (reach + lift shaping); a task-specific dense preset is TODO before RL.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from dexverse.assets.rack_builder import ensure_rack_asset

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .base_cfg import SUCCESS_MARKER_QUAT, BimanualLiftObservationsCfg, BimanualLiftTerminationsCfg
from .lift_tray_cfg import FOOD_REST_QUAT, FOOD_Z_OFFSET, LiftTrayEnvFloatingShadowBimanualCfg


def _quat_mul_wxyz(a, b):
    """Hamilton product of two (w, x, y, z) quaternions (a * b)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def food_rest_quat_yawed(yaw_rad: float) -> tuple[float, float, float, float]:
    """``FOOD_REST_QUAT`` (lying flat) rotated by ``yaw_rad`` about the z axis.

    Used as the food's orientation *in the tray's frame*: the tray asset's
    long side is its local x, the flat-lying food's long side is its local x
    too, so ``yaw_rad=0`` lays the food along the tray's long side.
    """
    import math

    qz = (math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0))
    return _quat_mul_wxyz(qz, FOOD_REST_QUAT)

# ---------------------------------------------------------------------------
# Rack (generated on first use; the assets tree is not tracked by git)
# ---------------------------------------------------------------------------
RACK_NAME = "rack_3level"
# Aug 2026 (after teleop feedback): lower and smaller rack -- level spacing 0.15
# (13 cm clearance above each shelf), 0.36 x 0.48 m footprint, lowest shelf 8 cm
# above the table -- placed closer to the hands (see rack_init_x below).
RACK_PARAMS = dict(levels=3, spacing=0.15, depth=0.36, width=0.48, board_thickness=0.02, post_size=0.03, bottom_gap=0.08, top_margin=0.03)
RACK_USD_PATH, RACK_MANIFEST = ensure_rack_asset(RACK_NAME, **RACK_PARAMS)
RACK_LEVEL_Z_TOPS: list[float] = [float(lvl["z_top"]) for lvl in RACK_MANIFEST["levels"]]

# Tray footprint (asset frame, before the Rz(90) rest rotation): 0.335 x 0.265 m.
TRAY_HALF_EXTENTS_XY: tuple[float, float] = (0.15, 0.115)  # slightly inside the rim
GOAL_PAD_SIZE: tuple[float, float, float] = (0.30, 0.30, 0.004)


def make_rack_cfg(usd_path: str, init_pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            # collision is authored per box prim in the generated USD
            collision_props=None,
            mass_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


GOAL_PAD_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/GoalPad",
    spawn=sim_utils.CuboidCfg(
        size=GOAL_PAD_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.75, 0.25), emissive_color=(0.0, 0.15, 0.0), roughness=1.0),
        visible=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
)


@configclass
class RetrieveTrayObservationsCfg(BimanualLiftObservationsCfg):
    """Lift-tray observations + rack pose / spawn level in ``state`` and the goal
    pad position in ``goal``."""

    @configclass
    class StateObsCfg(BimanualLiftObservationsCfg.StateObsCfg):
        rack_pos_b = ObsTerm(func=mdp.asset_pos_b, params={"asset_cfg": SceneEntityCfg("rack")}, noise=Unoise(n_min=-0.0, n_max=0.0))
        rack_quat_b = ObsTerm(func=mdp.object_quat_b, params={"object_cfg": SceneEntityCfg("rack")}, noise=Unoise(n_min=-0.0, n_max=0.0))
        rack_level_height = ObsTerm(func=mdp.rack_level_height, noise=Unoise(n_min=-0.0, n_max=0.0))
        food_pos_b = ObsTerm(func=mdp.asset_pos_b, params={"asset_cfg": SceneEntityCfg("food")}, noise=Unoise(n_min=-0.0, n_max=0.0))

    @configclass
    class GoalObsCfg(ObsGroup):
        goal_pad_pos_b = ObsTerm(func=mdp.asset_pos_b, params={"asset_cfg": SceneEntityCfg("goal_pad")}, noise=Unoise(n_min=-0.0, n_max=0.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    state: StateObsCfg = StateObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class RetrieveTrayEventCfg(dexverse_base_env.EventCfg):
    """Reset order matters: rack -> tray on a level of the rack -> goal pad ->
    food re-seated on the tray -> (invisible) success marker."""

    reset_object = None  # replaced by reset_tray_on_rack below

    # Rack anywhere in a band across the far half of the table (offsets from its
    # init pose), always turned so its open front faces the hands' start position
    # (midpoint of the two palms at their reset pose -- a fixed env-frame point,
    # set from ``rack_face_point`` in __post_init__), plus a little yaw jitter.
    reset_rack = EventTerm(
        func=mdp.reset_root_pose_uniform_facing,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("rack"),
            "pose_range": {"x": [-0.03, 0.03], "y": [-0.22, 0.22], "z": [0.0, 0.0]},
            "face_asset_cfg": None,
            "face_point": (-0.25, 0.0),
            "open_dir_local": (-1.0, 0.0),
            "yaw_jitter": (-0.1, 0.1),
        },
    )
    reset_tray_on_rack = EventTerm(
        func=mdp.reset_object_on_rack_level,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "rack_cfg": SceneEntityCfg("rack"),
            "level_z_tops": RACK_LEVEL_Z_TOPS,
            "level_x_range": (-0.05, -0.01),  # toward the shelf front (open side is -x): closer to the hands
            "level_y_range": (-0.04, 0.04),
            "yaw_range": (-0.1, 0.1),
            "z_offset": 0.03,  # tray root-to-bottom + clearance, overwritten in __post_init__
        },
    )
    reset_goal_pad = EventTerm(
        func=mdp.reset_root_pose_uniform_excluding,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("goal_pad"),
            # offsets from the pad's init pose (env origin, on the tabletop); rejection-
            # sampled to stay clear of the (randomly placed, yawed) rack footprint.
            # Band pulled toward the rack (teleop feedback Aug 2026) for a shorter carry.
            "pose_range": {"x": [-0.28, -0.02], "y": [-0.30, 0.30], "z": [0.0, 0.0], "roll": [0.0, 0.0], "pitch": [0.0, 0.0], "yaw": [0.0, 0.0]},
            "reference_asset_cfg": SceneEntityCfg("rack"),
            "min_xy_distance": 0.35,  # rack front face 0.18 + pad half 0.15 + 2 cm slack
            "max_attempts": 40,
        },
    )
    # placeholders: lift_tray_cfg.__post_init__ assigns the real terms (order kept)
    reset_food_velocity = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("food")})
    reset_food = EventTerm(func=mdp.sync_object, mode="reset", params={"target_cfg": SceneEntityCfg("food"), "source_cfg": SceneEntityCfg("object"), "z_offset": FOOD_Z_OFFSET, "quat": FOOD_REST_QUAT})
    reset_success_marker = EventTerm(
        func=mdp.sync_object,
        mode="reset",
        params={"target_cfg": SceneEntityCfg("success_marker"), "source_cfg": SceneEntityCfg("object"), "z_offset": 0.2, "quat": SUCCESS_MARKER_QUAT},
    )


@configclass
class RetrieveTrayTerminationsCfg(BimanualLiftTerminationsCfg):
    cargo_dropped = DoneTerm(
        func=mdp.cargo_off_carrier,
        params={
            "cargo_cfg": SceneEntityCfg("food"),
            "carrier_cfg": SceneEntityCfg("object"),
            "carrier_half_extents_xy": TRAY_HALF_EXTENTS_XY,
            "z_min": -0.03,
            "z_max": 0.20,
        },
    )


@configclass
class RetrieveTrayFromRackEnvFloatingShadowBimanualCfg(LiftTrayEnvFloatingShadowBimanualCfg):
    """Retrieve the loaded tray and hold it level in the goal region above the pad."""

    # ---- rack placement (env frame; the rack's -x face is the open front) ----
    # Band centre; reset_rack samples x +-0.05, y +-0.25 around it and aims the open
    # front at the hands' start. With the smaller rack (0.36 x 0.48) this band keeps
    # the front posts >= ~6 cm from the fingertips at their start pose and the far
    # corners on the table (audit-checked, Aug 2026).
    # Moved closer to the hands (0.29 -> 0.26, teleop feedback Aug 2026); the tighter
    # x/y jitter above caps the aim yaw so the front posts stay clear of the
    # fingertips' start pose (audit-checked).
    # Pulled closer again (0.27 -> 0.25, teleop feedback Aug 2026); audit-checked
    # against the palms' start pose.
    rack_init_x: float = 0.25
    rack_init_y: float = 0.0
    # PhysX collision offsets authored on the tray's collision meshes at spawn
    # (shared ``author_collision_offsets`` mechanism, same as the articulation
    # tasks' ``articulation_contact_offset``): ``tray_contact_offset`` is the
    # distance at which contacts are generated, ``tray_rest_offset`` lifts the
    # effective resting surface above the visual mesh.
    tray_contact_offset: float | None = 0.02
    tray_rest_offset: float | None = 0.002
    # Food heading ON THE TRAY (tray-frame yaw, rad). The food is re-seated
    # with ``sync_object(quat_local=...)`` so it follows the tray's yaw on the
    # rack; 0 lays the 23.6 cm prop along the tray's 33.5 cm long side. The
    # previous world-fixed placement had it across the tray's 26.5 cm short
    # side (inner bed ~23 cm) -> it overlapped the rim at spawn (teleop
    # feedback, Aug 2026); relative to that, 0 here is the requested 90 deg
    # turn. Applied to the spawn pose too.
    food_yaw_offset_rad: float = 0.0
    # Seat height of the food root above the tray root at spawn / re-seat.
    # Laid in the bed (not propped on the rim) the prop rests ~2 mm above the
    # tray root, so 5 mm leaves ~3 mm to settle under gravity (audit, Aug 2026;
    # the parent's 10 mm was tuned for the rim-propped rest height).
    food_z_offset: float = 0.005
    # Env-frame point the rack's open front is aimed at: the hands' start (midpoint of
    # the two Shadow palms at their reset pose, x=-0.25, y=+-0.3). Body poses are not
    # refreshed between reset events, so this is a fixed point rather than a live read.
    rack_face_point: tuple[float, float] = (-0.25, 0.0)
    # ---- task thresholds (in-air hold goal) ----
    # The goal is to HOLD the tray level in the air over the pad, not to set it
    # down (a table pad was too easy to slam onto; teleop feedback Aug 2026):
    # tray root within ``goal_xy_threshold`` of the pad centre, between
    # ``goal_z_above_pad`` above the pad root, tilt < ``goal_max_tilt_rad``,
    # slow, food aboard -- for ``goal_hold_steps`` consecutive env steps.
    goal_xy_threshold: float = 0.08
    goal_z_above_pad: tuple[float, float] = (0.12, 0.28)
    goal_hold_steps: int = 30
    goal_max_tilt_rad: float = 0.174532925  # 10 deg, same as the lift-tray level gate
    goal_max_lin_speed: float = 0.10
    goal_rest_z_tol: float = 0.03  # legacy (table-rest success), unused by the hold goal
    # Radius of the 8 corner spheres marking the hold band (vision cue).
    goal_corner_marker_radius: float = 0.012
    episode_length_s_override: float = 20.0

    observations: RetrieveTrayObservationsCfg = RetrieveTrayObservationsCfg()
    events: RetrieveTrayEventCfg = RetrieveTrayEventCfg()
    terminations: RetrieveTrayTerminationsCfg = RetrieveTrayTerminationsCfg()

    def __post_init__(self):
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        # Scene additions must exist before the base builds the scene.
        self.scene.rack = make_rack_cfg(str(RACK_USD_PATH), (self.rack_init_x, self.rack_init_y, table_top_z))
        self.scene.goal_pad = GOAL_PAD_CFG.replace(
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, table_top_z + GOAL_PAD_SIZE[2] / 2.0), rot=(1.0, 0.0, 0.0, 0.0))
        )

        super().__post_init__()  # lift_tray: tray + food, base: table placement, cameras, rewards, teleop

        # Re-spawn the tray through the offset-authoring func (same USD/props,
        # plus the collision offsets above).
        tray_spawn = self.scene.object.spawn
        self.scene.object.spawn = dexverse_base_env.CollisionOffsetUsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties_and_collision_offsets,
            usd_path=tray_spawn.usd_path,
            scale=tray_spawn.scale,
            rigid_props=tray_spawn.rigid_props,
            collision_props=None,
            mass_props=tray_spawn.mass_props,
            contact_offset=self.tray_contact_offset,
            rest_offset=self.tray_rest_offset,
        )

        # Food laid along the tray's long side, following the tray's yaw: the
        # re-seat uses a tray-frame quat (``quat_local``) instead of the parent's
        # world-fixed ``quat``; the spawn pose composes it with the tray's init
        # rotation so the first frame already matches.
        food_quat_local = food_rest_quat_yawed(self.food_yaw_offset_rad)
        self.scene.food.init_state.rot = _quat_mul_wxyz(tuple(self.scene.object.init_state.rot), food_quat_local)
        self.events.reset_food.params.pop("quat", None)
        self.events.reset_food.params["quat_local"] = food_quat_local
        self.events.reset_food.params["z_offset"] = self.food_z_offset
        fx, fy, _ = self.scene.food.init_state.pos
        self.scene.food.init_state.pos = (fx, fy, float(self.scene.object.init_state.pos[2]) + self.food_z_offset)

        # Tray on the shelf: same root-to-bottom + clearance as on the table.
        self.events.reset_tray_on_rack.params["z_offset"] = self.object_half_height + self.table_clearance
        self.events.reset_rack.params["face_point"] = tuple(self.rack_face_point)

        # Success: tray held level IN THE AIR over the pad (height band above the
        # pad root), slow, food aboard, for ``goal_hold_steps`` consecutive steps.
        self.terminations.success.func = mdp.carrier_held_aloft_with_cargo
        self.terminations.success.params = {
            "carrier_cfg": SceneEntityCfg("object"),
            "goal_cfg": SceneEntityCfg("goal_pad"),
            "cargo_cfg": SceneEntityCfg("food"),
            "xy_threshold": self.goal_xy_threshold,
            "z_above_goal": tuple(self.goal_z_above_pad),
            "max_tilt_rad": self.goal_max_tilt_rad,
            "max_lin_speed": self.goal_max_lin_speed,
            "hold_steps": self.goal_hold_steps,
            "carrier_half_extents_xy": TRAY_HALF_EXTENTS_XY,
            "cargo_z_min": -0.03,
            "cargo_z_max": 0.20,
        }
        # Tell the policy the target height band (mid), next to the pad position.
        if getattr(self.observations, "goal", None) is not None:
            self.observations.goal.goal_height_above_pad = ObsTerm(
                func=mdp.scalar_obs, params={"value": 0.5 * (self.goal_z_above_pad[0] + self.goal_z_above_pad[1])}
            )
        # Vision-policy cue (always on, camera-visible): small spheres at the 8
        # corners of the hold band over the pad. The filled box and the pad on
        # the table were removed -- both blocked the view (teleop feedback).
        z_lo, z_hi = self.goal_z_above_pad
        hx, hy = GOAL_PAD_SIZE[0] / 2.0, GOAL_PAD_SIZE[1] / 2.0
        corners = [(sx * hx, sy * hy, z) for z in (z_lo, z_hi) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
        self.add_scene_vis_term(
            "hold_band_corners",
            ObsTerm(
                func=mdp.local_points_vis,
                params={
                    "asset_cfg": SceneEntityCfg("goal_pad"),
                    "local_points": corners,
                    "radius": self.goal_corner_marker_radius,
                    "color": (0.15, 0.75, 0.25),
                    "opacity": 1.0,
                    "prim_path": "/Visuals/HoldBandCorners",
                    "hide_from_cameras": False,
                },
            ),
        )
        # The pad stays as the (invisible, non-colliding) goal frame the corners
        # and the success term hang off.
        self.scene.goal_pad.spawn.visible = False
        # The invisible lift marker is meaningless here; keep it parked out of the way.
        self.scene.success_marker.spawn.visible = False
