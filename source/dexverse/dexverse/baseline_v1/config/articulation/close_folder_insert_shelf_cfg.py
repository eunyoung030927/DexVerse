# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual: close an open folder, rotate it upright, and insert it in a slot.

Three-stage extension of :mod:`open_flat_folder_cfg` (same folder asset and
Shadow bimanual robot): the folder spawns lying *open* on the table; the robot
must swing the cover shut, lift the closed folder with two hands, rotate it so
it stands vertically like a file, and insert it into the shelf's single narrow
vertical slot (:mod:`dexverse.assets.shelf_builder`: base + back + sides + two
groups of smaller stored-folder props). The slot is open at the front and top.
Its 8 cm opening closely fits the closed folder's roughly 6.2 cm thickness, while its
32 cm height accommodates the folder's approximately 28 cm height. The folder
stands like a file: its thickness spans the narrow slot and its 23 cm broad
dimension extends into the shelf. The shelf stands at the far side of the
table, randomly placed, with its opening turned toward the hands' start.

Success :func:`mdp.articulation_closed_and_inserted`: hinge closed (< 20 deg),
folder root inserted past the slot front, folder thickness axis aligned with
the narrow slot-width axis, and the folder's height axis aligned with shelf up. It
must be standing vertically and at rest. Failure: folder out of
bounds. Rewards are inherited from the fixate-articulation preset (joint
progress from init counts closing as progress); a task-specific dense preset is
TODO before RL.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from dexverse.assets.shelf_builder import ensure_slot_shelf_asset

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .articulation_base.articulation_base_cfg import ARTICULATION_KEY
from .open_flat_folder_cfg import OpenFlatFolderEnvFloatingDexHandRightCfg

# Folder closed is ~0.23 x 0.28 x 0.062 m at scale 1.2. The narrow vertical
# opening follows its 6.2 cm thickness with ~1.8 cm total side clearance.
FOLDER_CLOSED_THICKNESS = 0.062
SLOT_THICKNESS_CLEARANCE = 0.018
SHELF_NAME = "slot_shelf_vertical_folder_thickness"
SHELF_PARAMS = dict(
    inner_depth=0.30,
    slot_width=FOLDER_CLOSED_THICKNESS + SLOT_THICKNESS_CLEARANCE,
    filler_width=0.12,
    height=0.32,
    panel_thickness=0.02,
    top_closed=False,
    slot_orientation="vertical",
    vertical_filler_style="folder_props",
)
SHELF_USD_PATH, SHELF_MANIFEST = ensure_slot_shelf_asset(SHELF_NAME, **SHELF_PARAMS)
SLOT_CENTER_LOCAL: tuple[float, float, float] = tuple(SHELF_MANIFEST["slot"]["center"])
SLOT_HALF_EXTENTS: tuple[float, float, float] = tuple(SHELF_MANIFEST["slot"]["half_extents"])
SHELF_FOOTPRINT_HALF_EXTENTS: tuple[float, float] = (
    float(SHELF_MANIFEST["footprint"]["x_half"]),
    float(SHELF_MANIFEST["footprint"]["y_half"]),
)
# At the 3.2 rad reset angle, the cover unfolds along local +x. These values
# conservatively enclose both scaled folder panels in the tabletop plane.
FOLDER_OPEN_FOOTPRINT_CENTER_LOCAL = (0.11, 0.0)
FOLDER_OPEN_FOOTPRINT_HALF_EXTENTS = (0.22, 0.14)
FOLDER_OPEN_JOINT_POS = 3.2  # rad, hinge range is [0, 3.32]; cover lying flat open


def make_shelf_cfg(usd_path: str, init_pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            collision_props=None,  # authored per box prim
            mass_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class CloseFolderInsertShelfEnvFloatingDexHandRightCfg(OpenFlatFolderEnvFloatingDexHandRightCfg):
    """Close the folder, rotate it vertically, and insert it like a file."""

    # Folder starts open flat.
    articulation_init_joint_pos: dict[str, float] = {"RevoluteJoint_file_9_up": FOLDER_OPEN_JOINT_POS}
    # Same per-collider collision offsets as the tray (retrieve_tray_from_rack):
    # the folder flaps are thin, so engaging finger contacts earlier
    # (contact_offset) and settling them slightly above the mesh (rest_offset)
    # stabilises the bimanual grasp.
    articulation_contact_offset: float | None = 0.02
    articulation_rest_offset: float | None = 0.002
    # Open, the folder rests with its root ~3.3 cm above the table (reset audit, Aug
    # 2026: seated at the closed-folder height it pushed itself up 2.8 cm); seat it
    # ~3 mm above that rest height instead.
    articulation_half_height_est: float = 0.036
    # Folder spawn box (offsets from its seated init pose at the table centre) --
    # kept clear of the shelf band and in front of the bimanual palms (x=-0.25).
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.25, -0.05],
        "y": [-0.25, 0.25],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [math.radians(-45.0-90), math.radians(45.0-90)],
    }
    # Shelf band (offsets from shelf_init_x/y); the open front is aimed at the
    # hands' start ``shelf_face_point`` by reset_root_pose_uniform_facing.
    # x in [0.26, 0.32], y +-0.25: 16 cm closer to the hands than the previous
    # band. Even at the angled approach extrema, its footprint remains on the
    # 0.75 m-half-width tabletop.
    shelf_init_x: float = 0.29
    shelf_init_y: float = 0.0
    shelf_pose_range: dict[str, list[float]] = {"x": [-0.03, 0.03], "y": [-0.25, 0.25], "z": [0.0, 0.0]}
    shelf_face_point: tuple[float, float] = (-0.25, 0.0)
    # Keep the opening up front: rotate at most 45 degrees either way relative
    # to the heading that points the opening toward the hands.
    shelf_yaw_jitter: tuple[float, float] = (math.radians(-45.0), math.radians(45.0))
    # Reject actual oriented-footprint overlap at reset. This allows the shelf
    # to move closer without the old 0.58 m circumscribed-radius rule discarding
    # safe poses. A 1 cm gap absorbs footprint/model approximation error.
    folder_shelf_reset_clearance: float = 0.01

    closed_threshold_rad: float = 0.35
    max_axis_angle_rad: float = 0.3491  # 20 deg from vertical
    max_lin_speed: float = 0.05
    # The scaled folder reaches about 12 cm from its root along the insertion
    # axis. Requiring the root 15 cm past the opening puts the whole folder
    # inside, with its front edge roughly 3 cm behind the shelf front.
    min_insertion_depth: float = 0.15
    episode_length_s_override: float = 25.0

    def configure_debug_vis(self) -> None:
        super().configure_debug_vis()
        if not self.enable_debug_vis:
            return
        c, h = SLOT_CENTER_LOCAL, SLOT_HALF_EXTENTS
        self.add_scene_vis_term(
            "slot_target",
            ObsTerm(
                func=mdp.forbidden_zones_vis,
                params={
                    "box_zones": [(c[0], c[1], c[2], h[0], h[1], h[2])],
                    "object_cfg": SceneEntityCfg("shelf"),
                    "color": (0.15, 0.75, 0.25),
                    "opacity": 0.25,
                    "prim_path_prefix": "/Visuals/SlotTarget",
                },
            ),
        )

    def __post_init__(self):
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        self.scene.shelf = make_shelf_cfg(str(SHELF_USD_PATH), (self.shelf_init_x, self.shelf_init_y, table_top_z))

        super().__post_init__()

        # --- reset order: shelf first, then the folder kept clear of it ------------
        # The base's ``reset_articulation`` (uniform pose) fires before any event we
        # add here, so replace it: null it and re-add the folder reset *after* the
        # shelf placement, using the shelf as an exclusion reference.
        folder_pose_range = dict(self.events.reset_articulation.params["pose_range"]) if self.events.reset_articulation is not None else self.articulation_reset_pose_range
        self.events.reset_articulation = None
        self.events.reset_shelf = EventTerm(
            func=mdp.reset_root_pose_uniform_facing,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("shelf"),
                "pose_range": self.shelf_pose_range,
                "face_asset_cfg": None,
                "face_point": tuple(self.shelf_face_point),
                "open_dir_local": (-1.0, 0.0),
                "yaw_jitter": tuple(self.shelf_yaw_jitter),
            },
        )
        self.events.reset_folder = EventTerm(
            func=mdp.reset_root_pose_uniform_excluding,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(ARTICULATION_KEY),
                "pose_range": folder_pose_range,
                "reference_asset_cfg": SceneEntityCfg("shelf"),
                "footprint_half_extents": FOLDER_OPEN_FOOTPRINT_HALF_EXTENTS,
                "footprint_center_local": FOLDER_OPEN_FOOTPRINT_CENTER_LOCAL,
                "reference_footprint_half_extents": SHELF_FOOTPRINT_HALF_EXTENTS,
                "footprint_clearance": self.folder_shelf_reset_clearance,
                "max_attempts": 100,
            },
        )
        # joints (open) are re-seated by the base's reset_articulation_joints, which
        # runs before reset_folder -- fine, joint state is independent of the root pose.

        # --- success --------------------------------------------------------------
        self.terminations.success.func = mdp.articulation_closed_and_inserted
        self.terminations.success.params = {
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=self.success_joint_names),
            "shelf_cfg": SceneEntityCfg("shelf"),
            "slot_center_local": SLOT_CENTER_LOCAL,
            "slot_half_extents": SLOT_HALF_EXTENTS,
            "xy_margin": 0.02,
            "z_margin": 0.02,
            "closed_threshold": self.closed_threshold_rad,
            # Closed folder axes: local x = broad width, local y = height,
            # local z = thickness. Insert it like a file: thickness spans the
            # narrow shelf width (+y), broad width extends through the depth
            # (+/-x), and height points up (+z).
            "thickness_axis_local": (0.0, 0.0, 1.0),
            "slot_width_axis_local": tuple(SHELF_MANIFEST["slot"]["width_axis_local"]),
            "upright_axis_local": (0.0, 1.0, 0.0),
            "slot_up_axis_local": (0.0, 0.0, 1.0),
            "slot_front_x_local": float(SHELF_MANIFEST["slot"]["front_x"]),
            "min_insertion_depth": self.min_insertion_depth,
            "max_axis_angle_rad": self.max_axis_angle_rad,
            "max_lin_speed": self.max_lin_speed,
        }

        # --- operator cue: translucent slot box on the shelf (teleop only) ------------
        self.configure_debug_vis()

        # --- observations: shelf pose + slot centre ---------------------------------
        state = self.observations.state
        if state is not None:
            state.shelf_pos_b = ObsTerm(func=mdp.asset_pos_b, params={"asset_cfg": SceneEntityCfg("shelf")}, noise=Unoise(n_min=-0.0, n_max=0.0))
            state.shelf_quat_b = ObsTerm(func=mdp.object_quat_b, params={"object_cfg": SceneEntityCfg("shelf")}, noise=Unoise(n_min=-0.0, n_max=0.0))
        if getattr(self.observations, "goal", None) is None:

            @configclass
            class GoalObsCfg(ObsGroup):
                slot_pos_b = ObsTerm(
                    func=mdp.object_local_point_pos_b,
                    params={"object_cfg": SceneEntityCfg("shelf"), "local_offset": SLOT_CENTER_LOCAL},
                    noise=Unoise(n_min=-0.0, n_max=0.0),
                )

                def __post_init__(self):
                    self.enable_corruption = True
                    self.concatenate_terms = True
                    self.history_length = 0

            self.observations.goal = GoalObsCfg()
