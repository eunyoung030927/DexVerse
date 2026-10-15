"""Pull the narrow cutout, clear it for the automatic latch, then release."""

import math
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from dexverse.assets import DEXVERSE_AUTHORED_ARTICULATIONS_DIR
from ... import dexverse_base_env_cfg as base, mdp
from ...mdp import door_latch
from .articulation_base import ArticulationBaseEnvFloatingDexHandRightCfg

DOOR_USD_PATH = DEXVERSE_AUTHORED_ARTICULATIONS_DIR / "cutaway_door/cutaway_door.usd"


@configclass
class OpenCutawayDoorEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Single right hand; two passive spring joints and contact-only retention."""

    articulation_usd_path: str = str(DOOR_USD_PATH)
    articulation_scale: tuple = (1.0, 1.0, 1.0)
    # Local +X panel maps to world +Y; it opens toward the hand (world -X).
    articulation_init_pos: tuple = (0.12, -0.17, 0.0)
    articulation_init_rot: tuple = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    articulation_init_joint_pos: dict = {"door_hinge": 0.0, "latch_slide": 0.0}
    articulation_half_height_est: float = 0.003
    articulation_fix_root_link: bool = True
    articulation_collision_approximation: str | None = None
    articulation_contact_offset: float = 0.002
    articulation_rest_offset: float = 0.0
    articulation_reset_pose_range: dict = {"x": [-0.015, 0.015], "y": [-0.025, 0.025], "yaw": [-0.10, 0.10]}
    success_joint_names: list = ["door_hinge"]
    door_spring_stiffness: float = 0.20
    door_spring_damping: float = 0.12
    door_preload_angle: float = -0.20  # ~0.04 Nm toward the closed hard stop even at q=0.
    latch_spring_stiffness: float = 12.0
    latch_spring_damping: float = 1.0
    hold_steps: int = 60
    episode_length_s_override: float = 30.0
    rewards = base.RewardsCfg()
    terminations = base.TerminationsCfg()

    def __post_init__(self):
        if not Path(self.articulation_usd_path).is_file():
            raise FileNotFoundError(
                "New cutaway-door USD is missing. Run from the feature worktree: "
                "python scripts/run_local.py scripts/asset_tools/convert_cutaway_door.py --headless"
            )
        super().__post_init__()
        self.episode_length_s = self.episode_length_s_override
        spawn = self.scene.articulation.spawn
        spawn.articulation_props.enabled_self_collisions = True  # Door/catch contact is essential.
        spawn.articulation_props.solver_position_iteration_count = 32
        spawn.articulation_props.solver_velocity_iteration_count = 4
        # Default stabilization can arrest a weak spring before it reaches its
        # rest angle. This mechanism must close under small residual torques.
        spawn.articulation_props.sleep_threshold = 0.0
        spawn.articulation_props.stabilization_threshold = 0.0
        spawn.rigid_props.sleep_threshold = 0.0
        spawn.rigid_props.stabilization_threshold = 0.0
        spawn.activate_contact_sensors = True
        spawn.joint_drive_props.drive_type = "force"
        self.scene.articulation.actuators = {
            "door_spring": ImplicitActuatorCfg(
                joint_names_expr=["door_hinge"],
                stiffness=self.door_spring_stiffness,
                damping=self.door_spring_damping,
                effort_limit_sim=3.0,
                velocity_limit_sim=2.0,
                friction=0.0,  # Avoid static joint friction defeating the gentle closing spring.
            ),
            "catch_spring": ImplicitActuatorCfg(
                joint_names_expr=["latch_slide"],
                stiffness=self.latch_spring_stiffness,
                damping=self.latch_spring_damping,
                effort_limit_sim=5.0,
                velocity_limit_sim=0.5,
                friction=0.0,
            ),
        }
        # GPU contact filters require one explicit rigid-body path per column,
        # not a wildcard matching the entire hand articulation.
        from pxr import Usd, UsdPhysics

        robot_stage = Usd.Stage.Open(self.scene.robot.spawn.usd_path)
        robot_root = robot_stage.GetDefaultPrim().GetPath()
        robot_filters = [
            "{ENV_REGEX_NS}/Robot/" + str(prim.GetPath().MakeRelativePath(robot_root))
            for prim in robot_stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if not robot_filters:
            raise ValueError("Could not resolve robot rigid bodies for door contact filters")
        # Replace inherited articulation-root filters with real collider links.
        for tip in self.robot_config.fingertip_body_names:
            sensor = getattr(self.scene, f"{tip}_articulation_s", None)
            if sensor is not None:
                sensor.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Articulation/door"]
        for name, body, target in (
            ("door_hand_contact", "door", "Robot/.*"),
            ("latch_hand_contact", "latch", "Robot/.*"),
            ("door_latch_contact", "door", "Articulation/latch"),
        ):
            setattr(
                self.scene,
                name,
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Articulation/{body}",
                    filter_prim_paths_expr=robot_filters if target == "Robot/.*" else [f"{{ENV_REGEX_NS}}/{target}"],
                ),
            )
        self.events.reset_door_latch = EventTerm(
            func=door_latch.reset_door_latch,
            mode="reset",
            params={"buffer_name": door_latch.BUFFER, "door_preload_angle": self.door_preload_angle},
        )
        self.terminations.success = DoneTerm(
            func=door_latch.door_latched_and_released,
            params={"hold_steps": self.hold_steps},
        )
        self.terminations.door_reclosed = DoneTerm(
            func=door_latch.door_reclosed,
            params={"hold_steps": self.hold_steps},
        )
        self.rewards.open_angle = RewTerm(
            func=mdp.joint_open_fraction_reward,
            weight=1.0,
            params={
                "close_angle_rad": 0.0,
                "open_angle_rad": math.pi / 2,
                "asset_cfg": SceneEntityCfg("articulation", joint_names=["door_hinge"]),
            },
        )
        if self.observations.state is not None:
            self.observations.state.articulation_joint_pos.params["asset_cfg"] = SceneEntityCfg(
                "articulation", joint_names=["door_hinge", "latch_slide"]
            )
            self.observations.state.door_latch_signals = ObsTerm(
                func=door_latch.door_latch_signals, params={"hold_steps": self.hold_steps}
            )
        if self.observations.privileged is not None:
            self.observations.privileged.articulation_joint_vel.params["asset_cfg"] = SceneEntityCfg(
                "articulation", joint_names=["door_hinge", "latch_slide"]
            )
