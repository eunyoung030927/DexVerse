"""Shared launch-time visualization controls, without simulator imports at load."""

import argparse


def add_debug_visualization_args(parser, *, teleop=False):
    parser.add_argument(
        "--enable_debug_vis", action=argparse.BooleanOptionalAction,
        default=True if teleop else None,
        help=(
            "Enable operator debug cues (zones, reference frames and v1 hand-tracking dots). "
            "Goal-defining markers remain visible. Teleop/recording defaults to on; other tools "
            "keep the task default. Camera visibility is controlled separately by --cues_in_rgb."
        ),
    )
    parser.add_argument(
        "--cues_in_rgb", action=argparse.BooleanOptionalAction, default=teleop,
        help=(
            "For v1, render enabled debug cues in camera RGB as well as the viewport. "
            "Defaults on in teleop/recording for CloudXR/AVP, which may hide guide prims; "
            "otherwise defaults off. --no-cues_in_rgb tags cues as guide for clean camera RGB. "
            "Does not enable disabled cues or hide task goals. v0 rendering is unchanged."
        ),
    )


def configure_v1_debug_visualization(env_cfg, *, cues_in_rgb=False):
    """Apply after config/retargeter overrides, before gym.make and device creation.

    Task cues are created by config __post_init__, so enable_debug_vis must be
    supplied when rebuilding the config. Never clear scene_vis wholesale: it
    also contains task-defining goals. Camera-purpose policy is process-wide,
    launch-time only; this is not a runtime UI toggle for existing prims.
    """
    if not type(env_cfg).__module__.startswith("dexverse.baseline_v1."):
        return
    from dexverse.baseline_v1.visual_purpose import set_camera_hiding_enabled

    set_camera_hiding_enabled(not cues_in_rgb)
    devices = getattr(getattr(env_cfg, "teleop_devices", None), "devices", {})
    for device in devices.values():
        for retargeter in getattr(device, "retargeters", []):
            if hasattr(retargeter, "task_version"):
                retargeter.task_version = 1
                retargeter.debug_vis = bool(env_cfg.enable_debug_vis)


def show_pour_goal_marker(env_cfg) -> bool:
    """Show the goal sphere of the v0 pour-style tasks (GraspBleach / GraspPan / GraspKettle / GraspCup / PourCan /
    PourMug): at the goal position, red -> green as the tilt condition is met.

    The v0 configs leave it off (``pour_show_progress_marker = False``). Teleoperation turned it on
    (cross_dex ``show_goal_marker``), so the operator aimed at it, but replay / LeRobot rendering and policy
    evaluation did not: the recorded images had no goal although success needs the object within
    ``pour_goal_xy_threshold`` of it (2026-11-15, GraspBleach). Call this on a built env cfg in every such tool so
    that teleoperation, rendered datasets and evaluation show the same scene. Same observation term as the config
    adds with ``pour_show_progress_marker`` and ``enable_debug_vis`` on; v1 configs (shown by default) and tasks
    without the marker are left as they are. Returns True when the marker is on after the call.
    """
    if type(env_cfg).__module__.startswith("dexverse.baseline_v1."):
        return False
    if not hasattr(env_cfg, "pour_show_progress_marker") or getattr(env_cfg.scene, "success_marker", None) is None:
        return False
    scene_vis = getattr(env_cfg.observations, "scene_vis", None)
    if scene_vis is not None and hasattr(scene_vis, "pour_progress_marker_vis"):
        return True
    from dexverse.tasks import dexverse_base_env_cfg as base
    from dexverse.tasks import mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg

    if scene_vis is None:
        env_cfg.observations.scene_vis = base.ObservationsCfg.SceneVisObsCfg()
    env_cfg.observations.scene_vis.pour_progress_marker_vis = ObsTerm(
        func=mdp.pour_progress_marker_vis,
        params={
            "goal_asset_cfg": SceneEntityCfg("success_marker"),
            "object_cfg": SceneEntityCfg("object"),
            "primary_threshold_rad": env_cfg.pour_angle_rad,
            "primary_axis_local": env_cfg.pour_axis_local,
            "primary_world_axis": env_cfg.pour_world_axis,
            "primary_tilt_ge": env_cfg.pour_tilt_ge,
            "plane_threshold_rad": env_cfg.pour_plane_angle_threshold_rad,
            "plane_axis_local": env_cfg.pour_plane_axis_local,
            "plane_normal": env_cfg.pour_plane_normal,
            "radius": env_cfg.pour_progress_marker_radius,
            "opacity": env_cfg.pour_progress_marker_opacity,
            "num_color_steps": env_cfg.pour_progress_marker_num_color_steps,
            "prim_path_prefix": "/Visuals/PourProgressMarker",
        },
    )
    print("[goal-marker] pour goal sphere on (red -> green as the tilt condition is met)")
    return True
