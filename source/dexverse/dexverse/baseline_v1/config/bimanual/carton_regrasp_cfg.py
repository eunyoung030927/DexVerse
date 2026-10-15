# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual carton hold by a per-episode **target face** into a goal box.

The plain :mod:`lift_carton_cfg` only asks for the box to rise, which any fixed
two-hand hold satisfies. Here every episode draws ONE target face of the carton
(``[+x, -x, +y, -y, +z, -z]`` of the box frame, marked by four red spheres). Contact is not
restricted; the sampled face specifies which face must point down at the goal.

The goal is a **carton-shaped cue in the air**: an invisible goal frame is
snapped next to the carton's spawn (small offset, mostly toward the hands) and
a cue box -- the carton's own shape enlarged to ``goal_cue_scale`` (0.85 vs the
carton's 0.65), oriented for the required pose -- floats ``goal_center_height``
above it. Success = the carton's centre inside the cue's slack (it fits when
correctly oriented), its yaw within ``yaw_tol_deg`` of the cue's (a cuboid),
**standing on its target face** (that face's normal points down within
``upright_tol_deg`` -- so a sideways friction pinch never counts), slowly,
for ``hold_steps`` consecutive env steps.
Eight green spheres mark the cue's corners (camera-visible, per-episode
orientation -- a vision-policy landmark).

Success :func:`mdp.object_fit_in_cue_success`; failure: box out of
bounds. Rewards are inherited from the bimanual lift base (reach + lift
shaping); a task-specific dense preset (goal-box shaping) is TODO before RL.
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

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .carton_face_cue import CartonFaceCornerCue
from .lift_carton_cfg import LiftCartonEnvFloatingShadowBimanualCfg

# Carton (model_carton.usd) in the object frame (root on the bottom face,
# footprint centred). At unit scale: x in [-0.199, 0.203], y in [-0.235, 0.236],
# z in [0, 0.285]; this task uses ``CARTON_SCALE`` (0.65 -- slightly smaller
# than the lift task's 0.75, teleop feedback Aug 2026) and derives every
# dimension from it.
CARTON_SCALE = 0.65
_UNIT_HALF_X, _UNIT_HALF_Y, _UNIT_HEIGHT = 0.150 / 0.75, 0.176 / 0.75, 0.214 / 0.75
_UNIT_CENTER_XY = (0.0015 / 0.75, 0.0005 / 0.75)
CARTON_HALF_X, CARTON_HALF_Y, CARTON_HEIGHT = _UNIT_HALF_X * CARTON_SCALE, _UNIT_HALF_Y * CARTON_SCALE, _UNIT_HEIGHT * CARTON_SCALE
CARTON_CENTER_XY = (_UNIT_CENTER_XY[0] * CARTON_SCALE, _UNIT_CENTER_XY[1] * CARTON_SCALE)
# Four non-colliding red spheres mark the selected face's rectangle. Their
# centres sit at its corners, exposing a 2 cm radius silhouette to depth/XYZ.
FACE_CORNER_RADIUS = 0.020
_cx, _cy, _cz = CARTON_CENTER_XY[0], CARTON_CENTER_XY[1], CARTON_HEIGHT / 2.0
_HALF = {"x": CARTON_HALF_X, "y": CARTON_HALF_Y, "z": CARTON_HEIGHT / 2.0}
_CENTER = {"x": _cx, "y": _cy, "z": _cz}
# faces ordered [+x, -x, +y, -y, +z, -z]
_FACES = [("x", +1), ("x", -1), ("y", +1), ("y", -1), ("z", +1), ("z", -1)]
_AXES = ("x", "y", "z")
GRASP_FACE_BUFFER = "carton_grasp_face"


def _vec(**kw):
    return tuple(kw.get(a, 0.0) for a in _AXES)


def _face_corners() -> list:
    """Four object-frame corners per target face, ordered [+x,-x,+y,-y,+z,-z]."""
    out = []
    for axis, sign in _FACES:
        others = [a for a in _AXES if a != axis]
        corners = []
        for first in (-1, 1):
            for second in (-1, 1):
                c = dict(_CENTER)
                c[axis] += sign * _HALF[axis]
                c[others[0]] += first * _HALF[others[0]]
                c[others[1]] += second * _HALF[others[1]]
                corners.append(_vec(**c))
        out.append(corners)
    return out


# Object-frame outward normal of each face (must point DOWN at the goal).
DOWN_NORMAL_PER_FACE = [_vec(**{axis: float(sign)}) for axis, sign in _FACES]
# Object-frame axis that maps to the goal frame's +x in the required pose (see
# ``cue_half_extents_per_face``): carton z for +-x faces, carton x otherwise.
HORIZONTAL_AXIS_PER_FACE = [_vec(z=1.0) if axis == "x" else _vec(x=1.0) for axis, _ in _FACES]


def cue_half_extents_per_face(cue_scale: float) -> list:
    """Half extents (goal frame, world-aligned) of the carton-shaped cue at
    ``cue_scale`` for each target face being the bottom: the face's axis is
    vertical; the other two carton axes map to world x/y as (x->x, y->y) for
    +-z, (z->x, y->y) for +-x and (x->x, z->y) for +-y."""
    r = cue_scale / CARTON_SCALE
    hx, hy, hz = CARTON_HALF_X * r, CARTON_HALF_Y * r, CARTON_HEIGHT / 2.0 * r
    out = []
    for axis, _ in _FACES:
        if axis == "x":
            out.append((hz, hy, hx))
        elif axis == "y":
            out.append((hx, hz, hy))
        else:
            out.append((hx, hy, hz))
    return out
FACE_CORNERS_PER_FACE = _face_corners()

# Invisible, non-colliding goal frame on the table; the goal box hangs above it.
GOAL_FRAME_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/GoalFrame",
    spawn=sim_utils.CuboidCfg(
        size=(0.02, 0.02, 0.004),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.75, 0.25)),
        visible=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
)


@configclass
class CartonRegraspEnvFloatingShadowBimanualCfg(LiftCartonEnvFloatingShadowBimanualCfg):
    """Place the carton on the sampled target face inside the aerial goal box."""

    # Slightly smaller carton than the lift task.
    scale: tuple[float, float, float] = (CARTON_SCALE, CARTON_SCALE, CARTON_SCALE)

    # Box upright on the table, random position and yaw.
    obj_x_range: tuple[float, float] = (-0.05, 0.10)
    obj_y_range: tuple[float, float] = (-0.15, 0.15)
    obj_yaw_range: tuple[float, float] = (-math.pi, math.pi)

    # Per-face sampling weights [+x, -x, +y, -y, +z, -z]; the bottom face (-z)
    # can only be held from below, the top (+z) from above.
    grasp_face_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # ---- goal box (goal-frame coordinates; the frame sits on the table) ----
    # The frame is snapped next to the carton's spawn every reset (a small
    # world-frame offset, mostly toward the hands): the task is about bimanual
    # coordination on the target face, not about carrying the box across the
    # table (teleop feedback, Aug 2026). Offsets are from the carton root.
    goal_init_x: float = 0.0
    goal_init_y: float = 0.0
    goal_offset_x_range: tuple[float, float] = (-0.10, 0.0)
    goal_offset_y_range: tuple[float, float] = (-0.08, 0.08)
    # Carton-shaped cue: the carton's own shape at ``goal_cue_scale`` (vs
    # CARTON_SCALE 0.65), centred ``goal_center_height`` above the goal frame,
    # oriented for the required "target face down" pose. Success = the
    # carton's centre within the cue's slack (cue half - carton half: at 0.85
    # about +-4.0 / 4.7 / 2.9 cm per axis) + the target face's normal within
    # ``upright_tol_deg`` of straight down + the carton's horizontal axis within
    # ``yaw_tol_deg`` of the cue's (the cue is a cuboid) + slow, for hold_steps
    # consecutive env steps. Relaxed from 0.70 / 20 deg / 0.10 m/s (no successes
    # in teleop, Aug 2026); position, yaw and tilt are gated separately.
    goal_cue_scale: float = 0.85
    goal_center_height: float = 0.22
    upright_tol_deg: float = 35.0
    yaw_tol_deg: float = 25.0
    goal_max_lin_speed: float = 0.15
    hold_steps: int = 30
    # Visual cue only: 8 spheres at the cue box corners (per-episode shape).
    goal_corner_marker_radius: float = 0.02
    episode_length_s_override: float = 20.0

    def __post_init__(self):
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        self.scene.goal_frame = GOAL_FRAME_CFG.replace(
            init_state=RigidObjectCfg.InitialStateCfg(pos=(self.goal_init_x, self.goal_init_y, table_top_z + 0.002), rot=(1.0, 0.0, 0.0, 0.0))
        )
        super().__post_init__()

        # --- per-episode target face (sampled before anything reads it) ---------
        self.events.sample_grasp_face = EventTerm(
            func=mdp.sample_env_choice,
            mode="reset",
            params={"buffer_name": GRASP_FACE_BUFFER, "num_choices": 6, "weights": list(self.grasp_face_weights)},
        )
        # --- goal frame: snapped next to the carton after its reset (appended
        # events run later), world-aligned, at the carton's bottom height ----------
        self.events.reset_goal_frame = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("goal_frame"),
                "source_cfg": SceneEntityCfg("object"),
                "z_offset": 0.0,
                "quat": (1.0, 0.0, 0.0, 0.0),
                "xy_jitter_range": {"x": list(self.goal_offset_x_range), "y": list(self.goal_offset_y_range)},
            },
        )

        # --- success: carton fits the cue, standing on the target face + slow ---
        cue_half = cue_half_extents_per_face(self.goal_cue_scale)
        cue_center = (0.0, 0.0, self.goal_center_height)
        self.terminations.success.func = mdp.object_fit_in_cue_success
        self.terminations.success.params = {
            "cue_half_extents_per_choice": cue_half,
            "object_half_extents_per_choice": cue_half_extents_per_face(CARTON_SCALE),
            "down_normals_per_choice": DOWN_NORMAL_PER_FACE,
            "horizontal_axes_per_choice": HORIZONTAL_AXIS_PER_FACE,
            "object_center_local": (_cx, _cy, _cz),
            "buffer_name": GRASP_FACE_BUFFER,
            "goal_cfg": SceneEntityCfg("goal_frame"),
            "cue_center_local": cue_center,
            "object_cfg": SceneEntityCfg("object"),
            "upright_cos": math.cos(math.radians(self.upright_tol_deg)),
            "yaw_cos": math.cos(math.radians(self.yaw_tol_deg)),
            "max_lin_speed": self.goal_max_lin_speed,
            "hold_steps": self.hold_steps,
        }
        # Lift-height reward / marker graded to the cue's root height for an
        # upright carton; the base's lift marker itself is hidden (the corner
        # spheres are the goal).
        self.lift_height = max(self.goal_center_height - CARTON_HEIGHT / 2.0, 0.05)
        self.events.reset_success_marker.params["z_offset"] = self.lift_height
        self.scene.success_marker.spawn.visible = False

        # --- observations: target face + goal box in ``goal`` ---------------------
        @configclass
        class GoalObsCfg(ObsGroup):
            grasp_face = ObsTerm(
                func=mdp.env_choice_one_hot,
                params={"buffer_name": GRASP_FACE_BUFFER, "num_choices": 6},
                noise=Unoise(n_min=-0.0, n_max=0.0),
            )
            goal_frame_pos_b = ObsTerm(func=mdp.asset_pos_b, params={"asset_cfg": SceneEntityCfg("goal_frame")}, noise=Unoise(n_min=-0.0, n_max=0.0))
            goal_height_above_frame = ObsTerm(func=mdp.scalar_obs, params={"value": self.goal_center_height})

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 0

        self.observations.goal = GoalObsCfg()

        # --- target face (red spheres) + goal-box corners: camera-visible -----
        self.add_scene_vis_term(
            "graspable_face",
            ObsTerm(
                func=CartonFaceCornerCue,
                params={
                    "points_per_choice": FACE_CORNERS_PER_FACE,
                    "buffer_name": GRASP_FACE_BUFFER,
                    "object_cfg": SceneEntityCfg("object"),
                    "radius": FACE_CORNER_RADIUS,
                    "color": (0.9, 0.03, 0.03),
                },
            ),
        )
        corners_per_face = [
            [(cue_center[0] + sx * h[0], cue_center[1] + sy * h[1], cue_center[2] + sz * h[2]) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
            for h in cue_half
        ]
        self.add_scene_vis_term(
            "goal_box_corners",
            ObsTerm(
                func=mdp.env_choice_points_vis,
                params={
                    "asset_cfg": SceneEntityCfg("goal_frame"),
                    "buffer_name": GRASP_FACE_BUFFER,
                    "points_per_choice": corners_per_face,
                    "radius": self.goal_corner_marker_radius,
                    "color": (0.15, 0.75, 0.25),
                    "opacity": 1.0,
                    "prim_path": "/Visuals/CartonGoalCorners",
                    "hide_from_cameras": False,
                },
            ),
        )
