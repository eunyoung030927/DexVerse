# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two-phase scissors: open the blades, then snip a paper strip.

Extends :mod:`squeeze_scissors_cfg` with a functional phase. A kinematic
strip-on-a-stand prop (:func:`dexverse.assets.cut_props_builder.
build_strip_stand`: base + post + clamp holding a cantilevered paper strip)
stands at the table centre, its free edge aimed at the hands' start; the
scissors rig spawns around it. The robot must open the scissors, bring them so
the strip sits between the blades, and close them on it.

The paper has its normal authored collider and exchanges contact forces with
the complete scissors, including both blade links. Success is the contact
predicate :func:`mdp.scissors_inner_blades_touch_strip`: a long rectangular
zone along each blade's inward-facing cutting edge defines its functional
region, and both regions must touch opposite sides of the paper for a short
dwell, with their midpoint within its inset top face. No particular hinge
velocity or final joint angle is required; the paper also has full collision.

Stage graph: ``jaws_open`` (the parent's open-success predicate, latched --
the scissors must have been opened first) -> ``strip_cut``. A stricter
finger-keep-away term (``hand_near_body_point`` on the strip) is available if
demos show operators nudging the stand; the stage-2 predicate itself involves
only the scissors' joints and the strip/blade frames, so fingers cannot fake
it.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from dexverse.assets.cut_props_builder import ensure_strip_stand_asset
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .articulation_base.articulation_base_cfg import ARTICULATION_KEY
from .squeeze_scissors_cfg import SqueezeScissorsEnvFloatingDexHandRightCfg

STAGE_KEY = "cut_strip_scissors"
STAND_NAME = "strip_stand"
STAND_PARAMS = dict(
    base_size=0.12,
    base_height=0.02,
    post_size=0.03,
    post_height=0.115,
    clamp_size=(0.04, 0.035, 0.015),
    strip_length=0.14,
    strip_width=0.03,
    strip_thickness=0.0015,
    cut_point_inset=0.025,
    # Orange PLA-like stand so it reads as a 3D-printed fixture (Aug 2026).
    stand_color=(0.92, 0.42, 0.08),
)
STAND_USD_PATH, STAND_MANIFEST = ensure_strip_stand_asset(STAND_NAME, **STAND_PARAMS)
CUT_POINT_LOCAL: tuple[float, float, float] = tuple(STAND_MANIFEST["cut_point"])
# The whole paper strip as a box in the stand root frame; blade contact may
# occur anywhere along it.
_FREE_EDGE = STAND_MANIFEST["free_edge_local"]
STRIP_BOX_CENTER_LOCAL: tuple[float, float, float] = (
    float(_FREE_EDGE[0]) + STAND_PARAMS["strip_length"] / 2.0,
    0.0,
    float(STAND_MANIFEST["z_strip"]),
)
STRIP_BOX_HALF_EXTENTS: tuple[float, float, float] = (
    STAND_PARAMS["strip_length"] / 2.0,
    STAND_PARAMS["strip_width"] / 2.0,
    STAND_PARAMS["strip_thickness"] / 2.0,
)

# Scissor blade links (scissors010): the two halves hinged on the pivot screw
# ``body_19`` (= the articulation root). Both blade link frames sit at the
# pivot with the jaw along local +y and the shear line near local x = 0.
SCISSOR_BLADE_BODIES = ("Left_scissor_handle_3", "Right_cut_handle_6")


def make_strip_stand_cfg(usd_path: str, init_pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/StripStand",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=None,  # authored per box prim
            mass_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class CutStripScissorsEnvFloatingDexHandRightCfg(SqueezeScissorsEnvFloatingDexHandRightCfg):
    """Open the scissors, then close them on the strip's cut point."""

    # Two-phase episode: grasp + open + carry to the strip + snip.
    # 20 s = 1200 steps at 60 Hz (was 15 s): the task is long -- teleop demos run
    # 550-1250 steps -- so the eval budget matches the other functional tasks.
    episode_length_s_override: float = 20.0

    # Goal-at-centre layout: the strip stand sits at the exact table centre
    # (no jitter -- easy, consistent reach from the hands' start at x=-0.2),
    # its strip aimed exactly at the hands; the scissors rig is then
    # rejection-sampled *around* it (see ``articulation_reset_pose_range``
    # below + the rig reset in __post_init__).
    stand_init_x: float = 0.0
    stand_init_y: float = 0.0
    stand_pose_range: dict[str, list[float]] = {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}
    # Aim the strip's free edge (local -x) at the hands' start, with the
    # original slight heading variation so a right-hand snip stays natural.
    stand_face_point: tuple[float, float] = (-0.25, 0.0)
    stand_yaw_jitter: tuple[float, float] = (-0.26, 0.26)
    # Scissors-rig spawn band: a compact segment on the hand side of the
    # centred stand; the point-based exclusion below carves out the overlap
    # region.
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.30, -0.05],
        "y": [-0.22, 0.22],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        # Blade / jaws (tool local +y) toward the operator's LEFT (world +y,
        # the hands face +x): yaw about 0 only, so the left hand takes the
        # body and the right hand the functional end (teleop feedback, Aug 2026).
        "yaw": [-0.4, 0.4],
    }
    # Rig-vs-stand exclusion: the scissors + their two stands form a ~0.26 m
    # bar along local y centred on the pivot (stands at y=+-0.09); three check
    # points along it, each kept ``min_scissors_stand_xy_distance`` from the
    # stand centre (stand-block half-diagonal ~0.04 + strip-stand base
    # half-diagonal ~0.09). The strip itself cantilevers 0.14 m above the
    # scissors' resting height, so only the base footprint matters.
    scissors_rig_exclusion_points: tuple[tuple[float, float, float], ...] = (
        (0.0, -0.12, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.12, 0.0),
    )
    min_scissors_stand_xy_distance: float = 0.14

    # Continuous rectangular inner-blade region in each blade link frame. Only
    # the distal (tip-side) half of the cutting edge is active, so both blades
    # can touch the strip only at a relatively small jaw angle. The contact
    # cross-section remains forgiving; the success predicate separately
    # requires the blade zones to straddle the paper, rejecting flat placement.
    blade_zone_y_range: tuple[float, float] = (0.0485, 0.095)
    blade_zone_half_extents: tuple[float, float, float] = (0.008, 0.02325, 0.004)
    blade_contact_inflation: tuple[float, float, float] = (0.004, 0.004, 0.006)
    # The midpoint between the two inner edges must pass through this inset
    # top-face area, preventing contact against a vertical side from counting.
    strip_cut_edge_margin_xy: tuple[float, float] = (0.008, 0.004)
    blade_contact_hold_steps: int = 3

    def configure_debug_vis(self) -> None:
        super().configure_debug_vis()
        if not self.enable_debug_vis:
            return
        # Sphere on the strip's cut point: the operator's "cut here" cue.
        self.add_scene_vis_term(
            "cut_point",
            ObsTerm(
                func=mdp.body_point_zone_vis,
                params={
                    "target_cfg": SceneEntityCfg("strip_stand"),
                    "local_offset": CUT_POINT_LOCAL,
                    "radius": 0.02,
                    "color": (0.15, 0.75, 0.25),
                    "opacity": 0.5,
                    "prim_path": "/Visuals/CutPoint",
                    "hide_from_cameras": True,
                },
            ),
        )
        # Blue rectangles exactly matching the functional contact regions.
        zone_mid_y = 0.5 * (self.blade_zone_y_range[0] + self.blade_zone_y_range[1])
        zone_size = tuple(2.0 * value for value in self.blade_zone_half_extents)
        for index, body_name in enumerate(SCISSOR_BLADE_BODIES):
            self.add_scene_vis_term(
                f"blade_edge_{index}",
                ObsTerm(
                    func=mdp.body_box_zone_vis,
                    params={
                        "target_cfg": SceneEntityCfg(ARTICULATION_KEY, body_names=[body_name]),
                        "local_offset": (0.0, zone_mid_y, 0.0),
                        "size": zone_size,
                        "color": (0.15, 0.45, 0.85),
                        "opacity": 0.5,
                        "prim_path": f"/Visuals/BladeEdge{index}",
                        "hide_from_cameras": True,
                    },
                ),
            )

    def __post_init__(self):
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        self.scene.strip_stand = make_strip_stand_cfg(
            str(STAND_USD_PATH), (self.stand_init_x, self.stand_init_y, table_top_z)
        )

        super().__post_init__()

        # --- reset order: stand at the centre FIRST (strip aimed at the
        # hands), then the scissors rig (scissors + padding stands as one
        # rigid transform) rejection-sampled around it. The inherited
        # ``reset_articulation`` fires before any appended event, so null it
        # and re-add the rig reset after the stand with the exclusion
        # reference (close_folder_insert_shelf pattern).
        rig_params = dict(self.events.reset_articulation.params)
        self.events.reset_articulation = None
        self.events.reset_strip_stand = EventTerm(
            func=mdp.reset_root_pose_uniform_facing,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("strip_stand"),
                "pose_range": self.stand_pose_range,
                "face_asset_cfg": None,
                "face_point": tuple(self.stand_face_point),
                "open_dir_local": (-1.0, 0.0),
                "yaw_jitter": tuple(self.stand_yaw_jitter),
            },
        )
        self.events.reset_scissors_rig = EventTerm(
            func=mdp.reset_articulation_with_supports_uniform,
            mode="reset",
            params={
                **rig_params,
                "reference_asset_cfg": SceneEntityCfg("strip_stand"),
                "min_xy_distance": self.min_scissors_stand_xy_distance,
                "exclusion_points_local": list(self.scissors_rig_exclusion_points),
                "max_attempts": 60,
            },
        )

        # --- stage graph: jaws_open (latched) -> strip_cut ----------------------
        mdp.register_stage_graph(
            STAGE_KEY,
            mdp.StageGraphSpec(
                stages=(
                    mdp.StageSpec(
                        name="jaws_open",
                        func=mdp.joint_relative_move,
                        params={
                            "threshold": self.success_threshold,
                            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=self.success_joint_names),
                            "mode": "progress",
                            "op": ">=",
                            "reduce": "any",
                        },
                    ),
                    mdp.StageSpec(
                        name="strip_cut",
                        func=mdp.scissors_inner_blades_touch_strip,
                        params={
                            "scissors_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=self.success_joint_names),
                            "blade_body_names": SCISSOR_BLADE_BODIES,
                            "blade_zone_center_local": (
                                0.0,
                                0.5 * (self.blade_zone_y_range[0] + self.blade_zone_y_range[1]),
                                0.0,
                            ),
                            "blade_zone_half_extents": self.blade_zone_half_extents,
                            "strip_cfg": SceneEntityCfg("strip_stand"),
                            "strip_box_center_local": STRIP_BOX_CENTER_LOCAL,
                            "strip_box_half_extents": STRIP_BOX_HALF_EXTENTS,
                            "contact_inflation": self.blade_contact_inflation,
                            "strip_edge_margin_xy": self.strip_cut_edge_margin_xy,
                            "hold_steps": self.blade_contact_hold_steps,
                        },
                        deps=("jaws_open",),
                    ),
                ),
                terminal_stage="strip_cut",
                ordering_mode="strict",
                success_mode="substage",
            ),
            override=True,
        )
        self.terminations.success.func = mdp.stage_success
        self.terminations.success.params = {"task_key": STAGE_KEY, "persistent": True}

        # --- rewards: fixate preset plus carry-to-strip and stage shaping -------
        # ``point_to_point_distance_exp`` is valid here because the scissors'
        # pivot screw IS the articulation root.
        self.rewards.pivot_to_cutpoint = RewTerm(
            func=mdp.point_to_point_distance_exp,
            weight=1.5,
            params={
                "source_cfg": SceneEntityCfg(ARTICULATION_KEY),
                "source_local_offset": (
                    0.0,
                    0.5 * (self.blade_zone_y_range[0] + self.blade_zone_y_range[1]),
                    0.0,
                ),
                "target_cfg": SceneEntityCfg("strip_stand"),
                "target_local_offset": CUT_POINT_LOCAL,
                "distance_gain": 10.0,
            },
        )
        self.rewards.stage_bonus = RewTerm(
            func=mdp.stage_success_reward,
            weight=5.0,
            params={"task_key": STAGE_KEY, "persistent": True},
        )

        # --- observations --------------------------------------------------------
        state = self.observations.state
        if state is not None:
            state.stand_pos_b = ObsTerm(
                func=mdp.asset_pos_b,
                params={"asset_cfg": SceneEntityCfg("strip_stand")},
                noise=Unoise(n_min=-0.0, n_max=0.0),
            )
            state.stand_quat_b = ObsTerm(
                func=mdp.object_quat_b,
                params={"object_cfg": SceneEntityCfg("strip_stand")},
                noise=Unoise(n_min=-0.0, n_max=0.0),
            )
            state.stage_signals = ObsTerm(
                func=mdp.stage_signals, params={"task_key": STAGE_KEY, "persistent": True}
            )
        if getattr(self.observations, "goal", None) is None:

            @configclass
            class GoalObsCfg(ObsGroup):
                cut_point_b = ObsTerm(
                    func=mdp.object_local_point_pos_b,
                    params={"object_cfg": SceneEntityCfg("strip_stand"), "local_offset": CUT_POINT_LOCAL},
                    noise=Unoise(n_min=-0.0, n_max=0.0),
                )

                def __post_init__(self):
                    self.enable_corruption = True
                    self.concatenate_terms = True
                    self.history_length = 0

            self.observations.goal = GoalObsCfg()

        # --- operator cues -------------------------------------------------------
        self.configure_debug_vis()
