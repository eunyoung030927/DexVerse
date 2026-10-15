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
