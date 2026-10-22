# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Floating (virtual-joint) Inspire hand variants and tabletop helpers.

Single-hand (right / left) and bimanual floating Inspire hands using virtual
prismatic/revolute joints for wrist translation and rotation. The palm-down
home orientation is baked into the converted USDs
(``floating_inspire_hand_*/``), so the CFG base rotations stay identity.

The Inspire hand is underactuated: 6 driver joints + 6 mimic joints (kept as
coupled joints in the USD). The 6 mimics get a zero-gain actuator group (covered
so every DOF is assigned, but left to the PhysX mimic coupling), and only the 6
drivers are exposed to the action layout. The coupling is made rigid at spawn
(see :func:`_spawn_inspire_usd`), like the real hand's linkage.

Configurations:

* :obj:`FLOATING_INSPIRE_RIGHT_CFG`    - single floating right Inspire hand.
* :obj:`FLOATING_INSPIRE_LEFT_CFG`     - single floating left Inspire hand.
* :obj:`FLOATING_INSPIRE_BIMANUAL_CFG` - bimanual floating Inspire hands.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils.queries import find_matching_prim_paths
from isaaclab.sim.utils.stage import get_current_stage
from pxr import Usd
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.utils import configclass

from .. import TabletopRobotSetup, dex_retargeting_hand_spec
from ..floating_wrist import UPSTREAM_ALLEGRO_WRIST_DRIVE, wrist_drive_terms

_ASSET_DIR = Path(__file__).resolve().parent
_RETARGET_DIR = _ASSET_DIR / "retarget"

# All 12 finger DOFs (must all be assigned to an actuator group); the 6 drivers
# are commanded by the action layout, the 6 mimics follow via PhysX coupling.
_FINGER_JOINTS = [
    "index_proximal_joint", "index_intermediate_joint",
    "middle_proximal_joint", "middle_intermediate_joint",
    "pinky_proximal_joint", "pinky_intermediate_joint",
    "ring_proximal_joint", "ring_intermediate_joint",
    "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint", "thumb_distal_joint",
]
_FINGER_DRIVERS = [
    "index_proximal_joint", "middle_proximal_joint",
    "pinky_proximal_joint", "ring_proximal_joint",
    "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
]
_FINGER_MIMICS = [
    "index_intermediate_joint", "middle_intermediate_joint",
    "pinky_intermediate_joint", "ring_intermediate_joint",
    "thumb_intermediate_joint", "thumb_distal_joint",
]
_FINGERTIP_SUFFIXES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

# Armature added to the proximal (driver) and intermediate (mimic) finger
# joints' joint-space inertia, to help stabilize the underactuated coupling.
_FINGER_ARMATURE = 0.02

_SINGLE_TRANS_EXPR = "(x|y|z)_translation_joint"
_SINGLE_ROT_EXPR = "(x|y|z)_rotation_joint"

# Mimic constraint at driver=0 is value = multiplier*0 + offset. The 4 finger
# intermediates use offset=-0.04545; the thumb mimics use offset=0. The init
# pose MUST satisfy this, else the solver pops the finger off its joint.
# Use a tiny epsilon inside the URDF lower bound: USD stores angular limits in
# degrees and PhysX round-trips them, so -0.04545 ends up ~6e-9 below the
# reported limit and fails Isaac Lab's strict default-pos validation.
_MIMIC_INIT = {
    "index_intermediate_joint": -0.0454,
    "middle_intermediate_joint": -0.0454,
    "ring_intermediate_joint": -0.0454,
    "pinky_intermediate_joint": -0.0454,
    "thumb_intermediate_joint": 0.0,
    "thumb_distal_joint": 0.0,
}


def _finger_joint_names(prefix: str = "") -> list[str]:
    return [f"{prefix}{n}" for n in _FINGER_JOINTS]


def _finger_driver_names(prefix: str = "") -> list[str]:
    return [f"{prefix}{n}" for n in _FINGER_DRIVERS]


def _finger_mimic_names(prefix: str = "") -> list[str]:
    return [f"{prefix}{n}" for n in _FINGER_MIMICS]


# PhysX mimic offset of the four finger intermediates, in degrees (USD angular unit). The converted USDs store the
# URDF value -0.04545 rad as -0.04545 deg; the PhysX sign convention also flips it, so the coupling that matches
# _MIMIC_INIT (q_mimic = 1.064 * q_driver - 0.04545 rad) needs +2.604 deg (measured).
_FINGER_MIMIC_OFFSET_DEG = 2.604095


def _make_mimic_joints_rigid(prim_path: str) -> None:
    """Rigid mimic coupling for every Inspire hand spawned at ``prim_path`` (regex allowed).

    The converted USDs author the PhysX mimic joints as soft springs (naturalFrequency 25, dampingRatio 0.005): a
    finger blocked by an object bent its distal phalanx back by up to ~0.5 rad. naturalFrequency 0 makes the
    coupling a hard constraint, and the finger intermediates get the offset in the right unit and sign.
    """
    stage = get_current_stage()
    for root in find_matching_prim_paths(prim_path):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            natural_frequency = prim.GetAttribute("physxMimicJoint:rotZ:naturalFrequency")
            if not natural_frequency.IsValid():
                continue
            natural_frequency.Set(0.0)
            name = prim.GetName()
            if name.endswith("intermediate_joint") and "thumb" not in name:
                prim.GetAttribute("physxMimicJoint:rotZ:offset").Set(_FINGER_MIMIC_OFFSET_DEG)


def _spawn_inspire_usd(prim_path: str, cfg: sim_utils.UsdFileCfg, *args, **kwargs) -> Usd.Prim:
    """``spawn_from_usd`` followed by :func:`_make_mimic_joints_rigid`."""
    prim = spawn_from_usd(prim_path, cfg, *args, **kwargs)
    _make_mimic_joints_rigid(prim_path)
    return prim


def _finger_home(prefix: str = "") -> dict[str, float]:
    """Home joint positions for all 12 finger DOFs: drivers at 0, mimics at the
    constraint-satisfying value for driver=0."""
    return {f"{prefix}{n}": _MIMIC_INIT.get(n, 0.0) for n in _FINGER_JOINTS}


# ---------------------------------------------------------------------------
# Single floating right / left Inspire hand
# ---------------------------------------------------------------------------

FLOATING_INSPIRE_RIGHT_BASE_POS = (-0.1, 0.0, 0.45)
FLOATING_INSPIRE_RIGHT_BASE_ROT = (1.0, 0.0, 0.0, 0.0)  # orientation baked in URDF
FLOATING_INSPIRE_LEFT_BASE_POS = FLOATING_INSPIRE_RIGHT_BASE_POS
FLOATING_INSPIRE_LEFT_BASE_ROT = FLOATING_INSPIRE_RIGHT_BASE_ROT

_SINGLE_HOME_JOINTS = {
    "x_translation_joint": 0.0,
    "y_translation_joint": 0.0,
    "z_translation_joint": 0.3,
    "x_rotation_joint": 0.0,
    "y_rotation_joint": 0.0,
    "z_rotation_joint": 0.0,
    **_finger_home(),
}

FLOATING_INSPIRE_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=_spawn_inspire_usd,
        usd_path=str((_ASSET_DIR / "floating_inspire_hand_right" / "floating_inspire_hand_right.usd").resolve()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=FLOATING_INSPIRE_RIGHT_BASE_POS,
        rot=FLOATING_INSPIRE_RIGHT_BASE_ROT,
        joint_pos=dict(_SINGLE_HOME_JOINTS),
    ),
    actuators={
        "floating_base": ImplicitActuatorCfg(
            joint_names_expr=[_SINGLE_TRANS_EXPR, _SINGLE_ROT_EXPR],
            **wrist_drive_terms(_SINGLE_TRANS_EXPR, _SINGLE_ROT_EXPR, UPSTREAM_ALLEGRO_WRIST_DRIVE),
        ),
        "hand_drivers": ImplicitActuatorCfg(
            joint_names_expr=_finger_driver_names(),
            effort_limit_sim=5.0,
            stiffness=20.0,
            damping=2.0,
            armature=_FINGER_ARMATURE,
            velocity_limit_sim=6.0,
            friction=0.01,
        ),
        # Zero-gain: PhysX mimic coupling controls these; an active drive fights
        # the coupling and blows the fingers apart.
        "hand_mimics": ImplicitActuatorCfg(
            joint_names_expr=_finger_mimic_names(),
            effort_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
            armature=_FINGER_ARMATURE,
            velocity_limit_sim=6.0,
            friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_INSPIRE_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=_spawn_inspire_usd,
        usd_path=str((_ASSET_DIR / "floating_inspire_hand_left" / "floating_inspire_hand_left.usd").resolve()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=FLOATING_INSPIRE_LEFT_BASE_POS,
        rot=FLOATING_INSPIRE_LEFT_BASE_ROT,
        joint_pos=dict(_SINGLE_HOME_JOINTS),
    ),
    actuators={
        "floating_base": ImplicitActuatorCfg(
            joint_names_expr=[_SINGLE_TRANS_EXPR, _SINGLE_ROT_EXPR],
            **wrist_drive_terms(_SINGLE_TRANS_EXPR, _SINGLE_ROT_EXPR, UPSTREAM_ALLEGRO_WRIST_DRIVE),
        ),
        "hand_drivers": ImplicitActuatorCfg(
            joint_names_expr=_finger_driver_names(),
            effort_limit_sim=10.0,
            stiffness=20.0,
            damping=0.1,
            armature=_FINGER_ARMATURE,
            velocity_limit_sim=6.0,
            friction=0.01,
        ),
        "hand_mimics": ImplicitActuatorCfg(
            joint_names_expr=_finger_mimic_names(),
            effort_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
            armature=_FINGER_ARMATURE,
            velocity_limit_sim=6.0,
            friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_INSPIRE_RIGHT_PALM_BODY_NAME = "hand_base_link"
FLOATING_INSPIRE_RIGHT_FINGERTIP_BODY_NAMES = list(_FINGERTIP_SUFFIXES)
FLOATING_INSPIRE_RIGHT_HAND_TIPS_BODY_NAMES = (
    [FLOATING_INSPIRE_RIGHT_PALM_BODY_NAME] + FLOATING_INSPIRE_RIGHT_FINGERTIP_BODY_NAMES
)
FLOATING_INSPIRE_RIGHT_WRIST_JOINT_NAME = _SINGLE_ROT_EXPR
FLOATING_INSPIRE_RIGHT_ARM_JOINT_NAMES_EXPR = [_SINGLE_TRANS_EXPR]

FLOATING_INSPIRE_LEFT_PALM_BODY_NAME = FLOATING_INSPIRE_RIGHT_PALM_BODY_NAME
FLOATING_INSPIRE_LEFT_FINGERTIP_BODY_NAMES = FLOATING_INSPIRE_RIGHT_FINGERTIP_BODY_NAMES
FLOATING_INSPIRE_LEFT_HAND_TIPS_BODY_NAMES = FLOATING_INSPIRE_RIGHT_HAND_TIPS_BODY_NAMES
FLOATING_INSPIRE_LEFT_WRIST_JOINT_NAME = FLOATING_INSPIRE_RIGHT_WRIST_JOINT_NAME
FLOATING_INSPIRE_LEFT_ARM_JOINT_NAMES_EXPR = FLOATING_INSPIRE_RIGHT_ARM_JOINT_NAMES_EXPR


@configclass
class FloatingInspireSingleAbsJointPosActionCfg:
    """Absolute joint-position control for a single floating Inspire hand."""

    translation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["x_translation_joint", "y_translation_joint", "z_translation_joint"],
        scale=1.0,
        preserve_order=True,
    )
    rotation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["x_rotation_joint", "y_rotation_joint", "z_rotation_joint"],
        scale=1.0,
        preserve_order=True,
    )
    finger_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=_finger_driver_names(),  # 6 drivers; mimics follow
        scale=1.0,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class FloatingInspireSingleRelJointPosActionCfg:
    """Relative joint-position control for a single floating Inspire hand."""

    translation_action = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[_SINGLE_TRANS_EXPR], scale=0.1
    )
    rotation_action = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[_SINGLE_ROT_EXPR], scale=0.1
    )
    finger_action = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=_finger_driver_names(), scale=0.1
    )


# ---------------------------------------------------------------------------
# Bimanual floating Inspire hands
# ---------------------------------------------------------------------------

FLOATING_INSPIRE_BIMANUAL_BASE_POS = (-0.1, 0.0, 0.6)
FLOATING_INSPIRE_BIMANUAL_BASE_ROT = (1.0, 0.0, 0.0, 0.0)

_BIMANUAL_TRANS_EXPR = "(right|left)_wrist_trans_(x|y|z)"
_BIMANUAL_ROT_EXPR = "(right|left)_wrist_rot_(x|y|z)"

_BIMANUAL_WRIST_JOINT_NAMES = [
    f"{side}_wrist_{kind}_{axis}"
    for side in ("right", "left")
    for kind in ("trans", "rot")
    for axis in ("x", "y", "z")
]
_BIMANUAL_FINGER_DRIVERS = _finger_driver_names("right_") + _finger_driver_names("left_")

_BIMANUAL_HOME_JOINTS: dict[str, float] = {}
for _side in ("right", "left"):
    _BIMANUAL_HOME_JOINTS.update(
        {
            f"{_side}_wrist_trans_x": 0.0,
            f"{_side}_wrist_trans_y": 0.0,
            f"{_side}_wrist_trans_z": 0.3,
            f"{_side}_wrist_rot_x": 0.0,
            f"{_side}_wrist_rot_y": 0.0,
            f"{_side}_wrist_rot_z": 0.0,
        }
    )
    _BIMANUAL_HOME_JOINTS.update(_finger_home(f"{_side}_"))

FLOATING_INSPIRE_BIMANUAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=_spawn_inspire_usd,
        usd_path=str((_ASSET_DIR / "floating_inspire_hand_bimanual" / "floating_inspire_hand_bimanual.usd").resolve()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=FLOATING_INSPIRE_BIMANUAL_BASE_POS,
        rot=FLOATING_INSPIRE_BIMANUAL_BASE_ROT,
        joint_pos=dict(_BIMANUAL_HOME_JOINTS),
    ),
    actuators={
        "floating_base": ImplicitActuatorCfg(
            joint_names_expr=[_BIMANUAL_TRANS_EXPR, _BIMANUAL_ROT_EXPR],
            **wrist_drive_terms(_BIMANUAL_TRANS_EXPR, _BIMANUAL_ROT_EXPR, UPSTREAM_ALLEGRO_WRIST_DRIVE),
        ),
        "hand_drivers": ImplicitActuatorCfg(
            joint_names_expr=_finger_driver_names("right_") + _finger_driver_names("left_"),
            effort_limit_sim=10.0,
            stiffness=20.0,
            damping=0.5,
            armature=_FINGER_ARMATURE,
            velocity_limit_sim=6.0,
            friction=0.01,
        ),
        # Zero-gain like the single hands: a drive on the mimics pulls them back to their default target and
        # fights the coupling (the fingers stopped ~0.9 rad short of the coupled pose).
        "hand_mimics": ImplicitActuatorCfg(
            joint_names_expr=_finger_mimic_names("right_") + _finger_mimic_names("left_"),
            effort_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
            armature=_FINGER_ARMATURE,
            velocity_limit_sim=6.0,
            friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_INSPIRE_BIMANUAL_RIGHT_PALM_BODY_NAME = "right_hand_base_link"
FLOATING_INSPIRE_BIMANUAL_LEFT_PALM_BODY_NAME = "left_hand_base_link"
FLOATING_INSPIRE_BIMANUAL_PALM_BODY_NAME = FLOATING_INSPIRE_BIMANUAL_RIGHT_PALM_BODY_NAME
FLOATING_INSPIRE_BIMANUAL_FINGERTIP_BODY_NAMES = [
    f"right_{s}" for s in _FINGERTIP_SUFFIXES
] + [f"left_{s}" for s in _FINGERTIP_SUFFIXES]
FLOATING_INSPIRE_BIMANUAL_HAND_TIPS_BODY_NAMES = [
    FLOATING_INSPIRE_BIMANUAL_RIGHT_PALM_BODY_NAME,
    FLOATING_INSPIRE_BIMANUAL_LEFT_PALM_BODY_NAME,
] + FLOATING_INSPIRE_BIMANUAL_FINGERTIP_BODY_NAMES
FLOATING_INSPIRE_BIMANUAL_ARM_JOINT_NAMES_EXPR = ["(right|left)_wrist_trans_(x|y|z)"]


@configclass
class FloatingInspireBimanualAbsJointPosActionCfg:
    """Absolute joint-position control for bimanual floating Inspire hands."""

    wrist = JointPositionActionCfg(
        asset_name="robot",
        joint_names=_BIMANUAL_WRIST_JOINT_NAMES,
        scale=1.0,
        preserve_order=True,
    )
    fingers = JointPositionActionCfg(
        asset_name="robot",
        joint_names=_BIMANUAL_FINGER_DRIVERS,  # 6 drivers per hand; mimics follow
        scale=1.0,
        preserve_order=True,
        use_default_offset=False,
    )


# ---------------------------------------------------------------------------
# Retargeter layouts + dex-retargeting (teleop). Only the right DexPilot config
# + dexrot URDF exist; left/bimanual reuse the right config as a placeholder.
# Only the 6 driver joints are commanded; the 6 mimics follow via coupling.
# ---------------------------------------------------------------------------

_DEX_FINGER_NAMES = tuple(_FINGER_DRIVERS)

FLOATING_INSPIRE_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 12,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "xyz",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "finger_indices": tuple(range(6, 12)),
            "finger_joint_names": _DEX_FINGER_NAMES,
            "finger_permutation": tuple(range(6)),
        }
    },
}
FLOATING_INSPIRE_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 12,
    "hands": {
        "left": dict(FLOATING_INSPIRE_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT["hands"]["right"]),
    },
}
FLOATING_INSPIRE_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 24,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "xyz",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "finger_indices": tuple(range(12, 18)),
            "finger_joint_names": _DEX_FINGER_NAMES,
            "finger_permutation": tuple(range(6)),
        },
        "left": {
            "wrist_trans_indices": (6, 7, 8),
            "wrist_rot_indices": (9, 10, 11),
            "wrist_rot_order": "xyz",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "finger_indices": tuple(range(18, 24)),
            "finger_joint_names": _DEX_FINGER_NAMES,
            "finger_permutation": tuple(range(6)),
        },
    },
}

FLOATING_INSPIRE_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {
        "right": dex_retargeting_hand_spec(
            _RETARGET_DIR, "right", _RETARGET_DIR / "inspire_hand_right" / "inspire_hand_right_dexrot.urdf"
        )
    }
}
# inspire ships only a right config; left/bimanual reuse it as a placeholder.
FLOATING_INSPIRE_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {"left": FLOATING_INSPIRE_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING["hands"]["right"]}
}
FLOATING_INSPIRE_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {
        "right": FLOATING_INSPIRE_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING["hands"]["right"],
        "left": FLOATING_INSPIRE_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING["hands"]["left"],
    }
}


# ---------------------------------------------------------------------------
# Tabletop env-setup builders
# ---------------------------------------------------------------------------


def build_tabletop_floating_inspire_right_setup(prim_path: str = "{ENV_REGEX_NS}/Robot") -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating right Inspire hand."""
    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_INSPIRE_RIGHT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_INSPIRE_RIGHT_FINGERTIP_BODY_NAMES,
            "dummy_mass_body_names": [".*(thumb|index|middle|ring|pinky)_tip"],
            "hand_tips_body_names": FLOATING_INSPIRE_RIGHT_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": FLOATING_INSPIRE_RIGHT_WRIST_JOINT_NAME,
            "arm_joint_names_expr": FLOATING_INSPIRE_RIGHT_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_INSPIRE_RIGHT_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_INSPIRE_RIGHT_CFG.init_state.replace(
                pos=FLOATING_INSPIRE_RIGHT_BASE_POS,
                rot=FLOATING_INSPIRE_RIGHT_BASE_ROT,
            ),
        ),
        actions=FloatingInspireSingleAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": _finger_driver_names(),
            "wrist_position_offset": (0.0, 0.0, 0.0),
            "retargeter_config_filename": "floating_inspire_right",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


def build_tabletop_floating_inspire_left_setup(prim_path: str = "{ENV_REGEX_NS}/Robot") -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating left Inspire hand.

    NOTE: requires floating_inspire_hand_left.usd (convert the left URDF first).
    """
    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_INSPIRE_LEFT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_INSPIRE_LEFT_FINGERTIP_BODY_NAMES,
            "dummy_mass_body_names": [".*(thumb|index|middle|ring|pinky)_tip"],
            "hand_tips_body_names": FLOATING_INSPIRE_LEFT_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": FLOATING_INSPIRE_LEFT_WRIST_JOINT_NAME,
            "arm_joint_names_expr": FLOATING_INSPIRE_LEFT_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_INSPIRE_LEFT_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_INSPIRE_LEFT_CFG.init_state.replace(
                pos=FLOATING_INSPIRE_LEFT_BASE_POS,
                rot=FLOATING_INSPIRE_LEFT_BASE_ROT,
            ),
        ),
        actions=FloatingInspireSingleAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": _finger_driver_names(),
            "wrist_position_offset": (0.0, 0.0, 0.0),
            "retargeter_config_filename": "floating_inspire_left",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


def build_tabletop_floating_inspire_bimanual_setup(prim_path: str = "{ENV_REGEX_NS}/Robot") -> TabletopRobotSetup:
    """Return the tabletop env setup for floating bimanual Inspire hands."""
    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_INSPIRE_BIMANUAL_PALM_BODY_NAME,
            "right_palm_body_name": FLOATING_INSPIRE_BIMANUAL_RIGHT_PALM_BODY_NAME,
            "left_palm_body_name": FLOATING_INSPIRE_BIMANUAL_LEFT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_INSPIRE_BIMANUAL_FINGERTIP_BODY_NAMES,
            "dummy_mass_body_names": [".*(thumb|index|middle|ring|pinky)_tip"],
            "hand_tips_body_names": FLOATING_INSPIRE_BIMANUAL_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": None,
            "arm_joint_names_expr": FLOATING_INSPIRE_BIMANUAL_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_INSPIRE_BIMANUAL_CFG.replace(prim_path=prim_path),
        actions=FloatingInspireBimanualAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": [],
            "wrist_position_offset": (0.0, 0.0, 0.0),
            "retargeter_config_filename": "floating_inspire_bimanual",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


__all__ = [
    "FLOATING_INSPIRE_RIGHT_CFG",
    "FLOATING_INSPIRE_LEFT_CFG",
    "FLOATING_INSPIRE_BIMANUAL_CFG",
    "FLOATING_INSPIRE_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_INSPIRE_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_INSPIRE_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_INSPIRE_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FLOATING_INSPIRE_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FLOATING_INSPIRE_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FloatingInspireSingleAbsJointPosActionCfg",
    "FloatingInspireSingleRelJointPosActionCfg",
    "FloatingInspireBimanualAbsJointPosActionCfg",
    "build_tabletop_floating_inspire_right_setup",
    "build_tabletop_floating_inspire_left_setup",
    "build_tabletop_floating_inspire_bimanual_setup",
]
