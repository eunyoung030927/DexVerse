# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two-phase utility knife: slide the blade out, then cut a taped seam.

Extends :mod:`slide_utility_knife_cfg` with a functional phase. A kinematic
"mimicked box" workpiece (:func:`dexverse.assets.cut_props_builder.
build_taped_seam_workpiece`: two cuboid blocks separated by a thin open slot,
covered by a colored tape strip) stands on the table. The robot must extend
the blade and drag the blade tip through the tape along the whole seam.

Cutting is measured, never force-simulated (see :mod:`...mdp.cut_sweep`):

- A prestartup :func:`mdp.filter_collision_pairs` event makes the ``tape``
  prim collide with everything *except* the blade link, so the extended blade
  phantoms through the tape into the slot while the knife body cannot; the
  unfiltered slot walls physically guide the blade along the seam.
- A per-env **cut frontier** advances only while the blade is extended, the
  tip is pressed through the tape into the slot near the seam line, the knife
  heading tracks the seam, and the tip is *at* the frontier -- advancing at a
  capped rate. Teleporting or tapping the far end accrues nothing.

Success = the frontier passes ``cut_complete_frac`` of the seam. The parent's
0.2 m lift gate is retired: reaching the seam already forces holding the
knife. Stage graph (blade_extended -> seam_cut) provides latched stage
signals / progress via the standard stage-machine interface.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from dexverse.assets.cut_props_builder import ensure_taped_seam_workpiece_asset
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
from .slide_utility_knife_cfg import SlideUtilityKnifeEnvFloatingDexHandRightCfg

STAGE_KEY = "cut_seam_utility_knife"
WORKPIECE_NAME = "taped_seam_workpiece"
WORKPIECE_PARAMS = dict(
    seam_length=0.10,
    slot_width=0.012,
    block_width=0.07,
    block_height=0.045,
    tape_thickness=0.0025,
    tape_width=0.05,
    # Darker kraft-cardboard body so it reads as a shipping box (Aug 2026).
    block_color=(0.42, 0.27, 0.14),
)
WORKPIECE_USD_PATH, WORKPIECE_MANIFEST = ensure_taped_seam_workpiece_asset(WORKPIECE_NAME, **WORKPIECE_PARAMS)
SEAM_START_LOCAL: tuple[float, float, float] = tuple(WORKPIECE_MANIFEST["seam"]["start"])
SEAM_END_LOCAL: tuple[float, float, float] = tuple(WORKPIECE_MANIFEST["seam"]["end"])
SLOT_CENTER_LOCAL: tuple[float, float, float] = tuple(WORKPIECE_MANIFEST["slot"]["center"])
SLOT_HALF_EXTENTS: tuple[float, float, float] = tuple(WORKPIECE_MANIFEST["slot"]["half_extents"])
Z_BLOCK_TOP: float = float(WORKPIECE_MANIFEST["z_block_top"])

# Knife link names (synthesis/utility knife002). The sweep's reference point is
# the blade's *cutting-edge tip corner* in the *scaled* (1.3x) blade-link
# frame: the blade is a trapezoid whose cutting edge is the -x side of the
# plate (it reaches the tip at y=0.109 unscaled; the +x spine side stops at
# y=0.100), so the corner sits at (-0.009, 0.109, -0.0065) unscaled. Using the
# edge (not the plate mid-width, 11.7 mm up the plate) makes the dip gate mean
# "the edge is through the tape" rather than demanding a ~15 mm deep stroke.
KNIFE_BLADE_BODY = "E_link_blade_2"
KNIFE_TIP_LOCAL_OFFSET: tuple[float, float, float] = (-0.0117, 0.142, -0.0045)


def make_workpiece_cfg(usd_path: str, init_pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
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
class CutSeamUtilityKnifeEnvFloatingDexHandRightCfg(SlideUtilityKnifeEnvFloatingDexHandRightCfg):
    """Slide the blade out, then drag the tip through the taped seam."""

    # Two-phase episode: grasp + extend + carry + gated 10 cm drag.
    episode_length_s_override: float = 20.0

    # Goal-at-centre layout: the workpiece sits at the exact table centre
    # (no translation jitter -- easy, consistent reach from the hands' start
    # at x=-0.2) with only a slight heading jitter about "seam along world y"
    # so a right-hand drag stays natural; the knife rig is then
    # rejection-sampled *around* it (see ``articulation_reset_pose_range``
    # below + the rig reset in __post_init__).
    workpiece_init_x: float = 0.0
    workpiece_init_y: float = 0.0
    workpiece_pose_range: dict[str, list[float]] = {
        "x": [0.0, 0.0],
        "y": [0.0, 0.0],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.26, 0.26],
    }
    # Cut direction as seen by the operator (hands face +x, so their left is
    # +y): True drags the blade from +y to -y, i.e. left -> right.
    cut_left_to_right: bool = True
    # Knife-rig spawn band: a compact segment on the hand side of the centred
    # workpiece; the point-based exclusion below carves out the overlap region.
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.32, -0.05],
        "y": [-0.22, 0.22],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        # Blade / jaws (tool local +y) toward the operator's LEFT (world +y,
        # the hands face +x): yaw about 0 only, so the left hand takes the
        # body and the right hand the functional end (teleop feedback, Aug 2026).
        "yaw": [-0.4, 0.4],
    }
    # Rig-vs-workpiece exclusion: the knife + its two stands form a ~0.26 m
    # bar along the knife's local +y (stands at y=0 and y=0.2); three check
    # points along it, each kept ``min_knife_workpiece_xy_distance`` from the
    # workpiece centre (stand half-diagonal ~0.04 + knife half-width ~0.03 +
    # workpiece half-diagonal ~0.08 at the 0.10 m seam). The rig may sit much
    # closer than a single root radius whenever it points away from the box.
    knife_rig_exclusion_points: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.2, 0.0))
    min_knife_workpiece_xy_distance: float = 0.16

    # --- sweep gates (see mdp.CutSweepSpec) --------------------------------
    cut_complete_frac: float = 0.95
    cut_lateral_tol: float = 0.012
    cut_min_dip: float = 0.003
    cut_backlash: float = 0.02
    cut_max_step_advance: float = 0.01
    cut_align_threshold_cos: float = 0.90
    # Blade-extension gate reuses the parent success threshold (progress units).

    def configure_debug_vis(self) -> None:
        super().configure_debug_vis()
        if not self.enable_debug_vis:
            return
        # Red -> green sphere riding the cut frontier. (A translucent green
        # slab over the slot was tried and removed -- it blocked the
        # operator's view of the seam; the yellow tape strip itself is the
        # "cut here" cue.)
        self.add_scene_vis_term(
            "cut_frontier",
            ObsTerm(func=mdp.cut_frontier_vis, params={"task_key": STAGE_KEY, "radius": 0.015}),
        )

    def __post_init__(self):
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        self.scene.workpiece = make_workpiece_cfg(
            str(WORKPIECE_USD_PATH), (self.workpiece_init_x, self.workpiece_init_y, table_top_z)
        )

        super().__post_init__()

        # --- reset order: workpiece at the centre FIRST, then the knife rig
        # (knife + support stands as one rigid transform) rejection-sampled
        # around it. The inherited ``reset_articulation`` fires before any
        # appended event, so null it and re-add the rig reset after the
        # workpiece with the exclusion reference (close_folder_insert_shelf
        # pattern).
        rig_params = dict(self.events.reset_articulation.params)
        self.events.reset_articulation = None
        self.events.reset_workpiece = EventTerm(
            func=mdp.reset_root_pose_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("workpiece"),
                "pose_range": self.workpiece_pose_range,
            },
        )
        self.events.reset_knife_rig = EventTerm(
            func=mdp.reset_articulation_with_supports_uniform,
            mode="reset",
            params={
                **rig_params,
                "reference_asset_cfg": SceneEntityCfg("workpiece"),
                "min_xy_distance": self.min_knife_workpiece_xy_distance,
                "exclusion_points_local": list(self.knife_rig_exclusion_points),
                "max_attempts": 60,
            },
        )

        # --- phantom tape: blade passes through, everything else collides ----
        self.events.filter_tape_blade = EventTerm(
            func=mdp.filter_collision_pairs,
            mode="prestartup",
            params={
                "source_prim_path": "{ENV_REGEX_NS}/Workpiece/" + WORKPIECE_MANIFEST["tape_prim"],
                "target_prim_paths": ["{ENV_REGEX_NS}/Articulation/" + KNIFE_BLADE_BODY],
            },
        )

        # --- cut sweep registration ------------------------------------------
        mdp.register_cut_sweep(
            STAGE_KEY,
            mdp.CutSweepSpec(
                workpiece_asset_name="workpiece",
                seam_start_local=SEAM_END_LOCAL if self.cut_left_to_right else SEAM_START_LOCAL,
                seam_end_local=SEAM_START_LOCAL if self.cut_left_to_right else SEAM_END_LOCAL,
                tool_asset_name=ARTICULATION_KEY,
                tip_body_name=KNIFE_BLADE_BODY,
                tip_local_offset=KNIFE_TIP_LOCAL_OFFSET,
                ext_joint_name=self.success_joint_names[0],
                ext_threshold=self.success_threshold,
                lateral_tol=self.cut_lateral_tol,
                z_surface_local=Z_BLOCK_TOP,
                min_dip=self.cut_min_dip,
                align_axis_tool_local=(0.0, 1.0, 0.0),
                align_threshold_cos=self.cut_align_threshold_cos,
                backlash=self.cut_backlash,
                max_step_advance=self.cut_max_step_advance,
                complete_frac=self.cut_complete_frac,
            ),
            override=True,
        )

        # --- stage graph: blade_extended -> seam_cut --------------------------
        mdp.register_stage_graph(
            STAGE_KEY,
            mdp.StageGraphSpec(
                stages=(
                    mdp.StageSpec(
                        name="blade_extended",
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
                        name="seam_cut",
                        func=mdp.cut_sweep_complete,
                        params={"task_key": STAGE_KEY},
                        deps=("blade_extended",),
                    ),
                ),
                terminal_stage="seam_cut",
                ordering_mode="strict",
                success_mode="substage",
            ),
            override=True,
        )

        # --- success: the seam is cut (lift gate retired -- reaching the seam
        # already forces holding the knife) ------------------------------------
        self.terminations.success.func = mdp.stage_success
        self.terminations.success.params = {"task_key": STAGE_KEY, "persistent": True}

        # --- rewards: keep the fixate preset (parent wired knife_lift before we
        # null it), add sweep shaping ------------------------------------------
        self.rewards.knife_lift = None
        self.rewards.tip_to_frontier = RewTerm(
            func=mdp.cut_sweep_tip_tracking_reward,
            weight=1.5,
            params={"task_key": STAGE_KEY, "distance_gain": 10.0, "require_ext": True},
        )
        self.rewards.cut_progress = RewTerm(
            func=mdp.cut_sweep_progress_reward,
            weight=5.0,
            params={"task_key": STAGE_KEY},
        )
        self.rewards.stage_bonus = RewTerm(
            func=mdp.stage_success_reward,
            weight=5.0,
            params={"task_key": STAGE_KEY, "persistent": True},
        )

        # --- observations ------------------------------------------------------
        state = self.observations.state
        if state is not None:
            state.workpiece_pos_b = ObsTerm(
                func=mdp.asset_pos_b,
                params={"asset_cfg": SceneEntityCfg("workpiece")},
                noise=Unoise(n_min=-0.0, n_max=0.0),
            )
            state.workpiece_quat_b = ObsTerm(
                func=mdp.object_quat_b,
                params={"object_cfg": SceneEntityCfg("workpiece")},
                noise=Unoise(n_min=-0.0, n_max=0.0),
            )
            state.stage_signals = ObsTerm(
                func=mdp.stage_signals, params={"task_key": STAGE_KEY, "persistent": True}
            )
            state.cut_frontier = ObsTerm(func=mdp.cut_sweep_frontier, params={"task_key": STAGE_KEY})
        if getattr(self.observations, "goal", None) is None:
            seam_start = SEAM_END_LOCAL if self.cut_left_to_right else SEAM_START_LOCAL
            seam_end = SEAM_START_LOCAL if self.cut_left_to_right else SEAM_END_LOCAL

            @configclass
            class GoalObsCfg(ObsGroup):
                seam_start_b = ObsTerm(
                    func=mdp.object_local_point_pos_b,
                    params={"object_cfg": SceneEntityCfg("workpiece"), "local_offset": seam_start},
                    noise=Unoise(n_min=-0.0, n_max=0.0),
                )
                seam_end_b = ObsTerm(
                    func=mdp.object_local_point_pos_b,
                    params={"object_cfg": SceneEntityCfg("workpiece"), "local_offset": seam_end},
                    noise=Unoise(n_min=-0.0, n_max=0.0),
                )

                def __post_init__(self):
                    self.enable_corruption = True
                    self.concatenate_terms = True
                    self.history_length = 0

            self.observations.goal = GoalObsCfg()

        # --- operator cues -----------------------------------------------------
        self.configure_debug_vis()
