# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-tilted-door task: the door frame lies tilted instead of standing.

Extends :class:`OpenDoorEnvFloatingDexHandRightCfg` (which stays registered):
each episode draws ONE of two rest modes for the whole frame + leaf assembly —
**slanted** like a basement bulkhead door (45 deg from vertical) or **flat**
like a ground/cellar hatch (90 deg, door plane horizontal) — via
``mdp.reset_root_pose_uniform_orientations`` (the drawn index is recorded in
``env.door_tilt_choice``). The success criterion is inherited unchanged: hinge
the leaf past the parent's open ratio; in the tilted modes that means lifting
the leaf up against gravity.

Padding: the frame is lifted a per-mode clearance above the tabletop so the
**backside of the latch mechanism never touches the ground** when the leaf
lies in the frame plane. Four cosmetic corner posts (kinematic, collision
disabled, always world-vertical) are re-seated under the frame corners every
reset so the lifted frame reads as a supported structure; their excess length
sinks invisibly into the table.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation as R

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .opendoor_cfg import DOOR_ROT, OpenDoorEnvFloatingDexHandRightCfg


def _tilted_door_rot(tilt_deg: float) -> tuple[float, float, float, float]:
    """Standing door rot composed with a world-frame pitch of ``tilt_deg``.

    Positive pitch about world +y tips the top of the door away from the robot
    (toward +x), so the handle face ends up facing upward along the slant.
    scipy quats are xyzw; Isaac's are wxyz.
    """
    standing = R.from_quat((*DOOR_ROT[1:], DOOR_ROT[0]))
    qx, qy, qz, qw = (R.from_euler("y", math.radians(tilt_deg)) * standing).as_quat()
    return (float(qw), float(qx), float(qy), float(qz))


def _build_pad_post_cfg(
    index: int,
    *,
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """Cosmetic vertical post under one frame corner (faucet-pedestal style).

    Kinematic, gravity-free, collision disabled — it never touches the hand or
    the door; it only fills the visible gap between the tabletop and the
    lifted/tilted frame. A rigid object (not a bare visual prim) so the
    per-corner ``sync_object`` reset event can re-seat it each episode.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DoorPadPost" + str(index),
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class OpenTiltedDoorEnvFloatingDexHandRightCfg(OpenDoorEnvFloatingDexHandRightCfg):
    """Open-door on a slanted (basement) or flat (ground-hatch) frame."""

    # The tilted door plane extends from the root toward +x, so pull the base
    # back toward the robot to keep the handle over the hand's workspace
    # (standing parent uses x = 0.3). TODO: tune after previewing.
    articulation_init_pos: tuple[float, float, float] = (-0.35, 0.0, 0.0)

    # ---- Tilt modes -------------------------------------------------------
    # (tilt from vertical in deg, frame lift above the tabletop in m). The
    # lift is the "padding": clearance so the latch backside on the underside
    # of the leaf plane never touches the ground. Measured from the USD: the
    # latch protrudes 0.03 m behind the leaf plane (0.045 m at the 1.5 asset
    # scale), so the flat mode's 0.08 m lift leaves ~3.5 cm of clearance;
    # slanted only needs a small seat (the latch is high up the slant).
    tilt_slanted_deg: float = 45.0
    tilt_flat_deg: float = 90.0
    slanted_lift: float = 0.05
    flat_lift: float = 0.08
    # Categorical weights (slanted, flat) for the per-episode draw.
    tilt_weights: tuple[float, float] = (0.5, 0.5)

    # ---- Cosmetic corner posts -------------------------------------------
    # Frame-corner points in the door asset's LOCAL (USD) frame, PRE-scaled by
    # the 1.5 asset scale (``sync_object`` rotates offsets but does not apply
    # the spawn mesh scale). From the USD bbox: frame x in [-0.125, 0.125],
    # z in [0.114, 0.335] (the frame plane is local x-z; local +y is the
    # handle-side normal). The posts' tops are synced to these points each
    # reset; excess post length sinks invisibly into the table.
    pad_corner_offsets_local: tuple[tuple[float, float, float], ...] = (
        (-0.1875, 0.0, 0.171),
        (0.1875, 0.0, 0.171),
        (-0.1875, 0.0, 0.5025),
        (0.1875, 0.0, 0.5025),
    )
    pad_post_size: tuple[float, float, float] = (0.06, 0.06, 0.5)
    pad_post_color: tuple[float, float, float] = (0.55, 0.5, 0.45)

    def __post_init__(self):
        super().__post_init__()

        # Replace the parent's plain pose reset with the categorical
        # rest-orientation reset: same x/y jitter, per-mode quat + lift, and
        # the drawn mode index recorded in ``env.door_tilt_choice`` (part of
        # the recorded scene state for demos).
        pose_range = self.articulation_reset_pose_range or {}
        self.events.reset_articulation.func = mdp.reset_root_pose_uniform_orientations
        self.events.reset_articulation.params = {
            "asset_cfg": SceneEntityCfg("articulation"),
            "pose_range": {
                "x": tuple(pose_range.get("x", (0.0, 0.0))),
                "y": tuple(pose_range.get("y", (0.0, 0.0))),
                "yaw": tuple(pose_range.get("yaw", (0.0, 0.0))),
            },
            "orientations": [
                (_tilted_door_rot(self.tilt_slanted_deg), self.slanted_lift),
                (_tilted_door_rot(self.tilt_flat_deg), self.flat_lift),
            ],
            "table_top_z": dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT,
            "weights": list(self.tilt_weights),
            "buffer_name": "door_tilt_choice",
        }

        # Corner posts: top face at the frame corner, world-vertical at any
        # tilt (``quat`` pins the world orientation; only the corner point
        # follows the door's frame). Added after super().__post_init__() so
        # they fire after ``reset_articulation`` reads the drawn tilt.
        post_h = float(self.pad_post_size[2])
        for i, corner in enumerate(self.pad_corner_offsets_local):
            setattr(
                self.scene,
                f"door_pad_post_{i}",
                _build_pad_post_cfg(i, size=self.pad_post_size, color=self.pad_post_color),
            )
            setattr(
                self.events,
                f"reset_door_pad_{i}",
                EventTerm(
                    func=mdp.sync_object,
                    mode="reset",
                    params={
                        "target_cfg": SceneEntityCfg(f"door_pad_post_{i}"),
                        "source_cfg": SceneEntityCfg("articulation"),
                        "source_local_offset": tuple(corner),
                        "z_offset": -post_h / 2.0,
                        "quat": (1.0, 0.0, 0.0, 0.0),
                    },
                ),
            )
