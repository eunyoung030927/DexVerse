# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone VR app: calibrate the controller-to-hand rotation of a hand-mounted Quest controller.

An empty scene shows two hand skeletons (joints + bones):

* green: the target, a fixed open flat hand in front of you, palm down with the fingers pointing
         forward, with its wrist (big green sphere) and dotted wrist axes (X magenta, Y cyan, Z white);
* yellow: your wrist (controller pose + pivot offset);
* red:   after a capture, your hand (live glove finger shape) with the new rotation and its wrist axes
         (frame marker: X red, Y green, Z blue), to check the result.

By default the background is transparent, so with passthrough you see your real hand and align it
to the green hand directly (``--no_passthrough`` draws a ground and sky). A rotation already saved in
the JSON is not used: the red hand only appears after this run's capture.

Bring the yellow sphere onto the green wrist and hold your real hand like the green hand, then press
A (right) / X (left): the rotation that maps the controller onto the target hand (averaged over
0.5 s) is saved as ``mount_rot_ctrl_to_wrist`` in the pivot JSON and the red hand appears with it, so
you can check it lies on the green one. Press again to redo.
B / Y: move the target in front of your current view; hold 2 s to quit.

``--mode handtracking`` measures instead of asking you to align: the headset's optical hand tracking
gives the hand (red skeleton, wrist axes) and the rotation is ``mean(R_controller^T R_hand)`` over the
capture (press A/X, keep still). It also reports the distance between the hand-tracking wrist and the
pivot wrist (yellow), a check of the pivot calibration. It needs the headset to stream hand tracking
while the controllers are active (Quest multimodal); gloves can make hand tracking less accurate.

Needs the Manus bridge and the CloudXR runtime, like the teleop::

    scripts/teleop_tools/run_teleop.sh scripts/teleop_tools/calibrate_controller_rotation.py \\
        --side right --offset_json /workspace/datasets/dexverse/calibration/controller_wrist_offset_both.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate the controller-to-hand rotation (standalone VR app).")
parser.add_argument("--side", choices=("right", "left"), default="right")
parser.add_argument("--offset_json", required=True, help="pivot_calibration.py output; the rotation is saved into it")
parser.add_argument("--mode", choices=("target", "handtracking"), default="target",
                    help="target: align your hand to the green hand; handtracking: take the hand orientation from "
                         "the headset's optical hand tracking (needs controllers and hand tracking at the same time)")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=47000)
parser.add_argument("--target_forward", type=float, default=0.35, help="target distance in front of the head (m)")
parser.add_argument("--target_down", type=float, default=0.12, help="target height below the head (m)")
parser.add_argument("--no_passthrough", action="store_true",
                    help="draw a ground and sky instead of a transparent background (passthrough)")
# Accepted for run_teleop.sh compatibility; unused here.
parser.add_argument("--teleop_device", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--enable_pinocchio", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.xr = True
simulation_app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
from dexverse.devices.manus.manus_device import (  # noqa: E402
    ManusBridgeClient,
    keypoints_to_openxr_joints,
    openxr_wrist_frame,
    snap_signed_permutation,
)
from dexverse.teleop_utils.xr_session import request_ar_session  # noqa: E402
from isaaclab.devices import DeviceBase  # noqa: E402
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES  # noqa: E402
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg, XrCfg  # noqa: E402
from isaaclab.devices.retargeter_base import RetargeterBase  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

POSE = DeviceBase.MotionControllerDataRowIndex.POSE.value
INPUTS = DeviceBase.MotionControllerDataRowIndex.INPUTS.value
BUTTON_0 = DeviceBase.MotionControllerInputIndex.BUTTON_0.value  # A (right) / X (left)
BUTTON_1 = DeviceBase.MotionControllerInputIndex.BUTTON_1.value  # B (right) / Y (left)
CONTROLLER = {"right": DeviceBase.TrackingTarget.CONTROLLER_RIGHT, "left": DeviceBase.TrackingTarget.CONTROLLER_LEFT}
CAPTURE_FRAMES = 45  # ~0.5 s at 90 Hz
QUIT_HOLD_S = 2.0


def rot_from_wxyz(q) -> np.ndarray:
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def hand_markers(path: str, color: tuple[float, float, float], radius: float) -> VisualizationMarkers:
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path=path, markers={"joint": sim_utils.SphereCfg(
        radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))}))


# Target wrist axes as dotted lines in their own colours (the frame-marker USD has fixed RGB axes).
TARGET_AXIS_COLORS = {"x": (1.0, 0.0, 1.0), "y": (0.0, 0.9, 1.0), "z": (1.0, 1.0, 1.0)}  # magenta, cyan, white
AXIS_LENGTH, AXIS_DOTS = 0.12, 10


def axis_markers(path: str) -> VisualizationMarkers:
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path=path, markers={
        name: sim_utils.SphereCfg(radius=0.004, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
        for name, color in TARGET_AXIS_COLORS.items()}))


def draw_axes(markers: VisualizationMarkers, pos: np.ndarray, rot: np.ndarray) -> None:
    steps = np.linspace(AXIS_LENGTH / AXIS_DOTS, AXIS_LENGTH, AXIS_DOTS)
    points = np.concatenate([pos + steps[:, None] * rot[:, i] for i in range(3)])
    markers.visualize(translations=points, marker_indices=np.repeat(np.arange(3), AXIS_DOTS))


FINGERS = ("index", "middle", "ring", "little")
BONES = [("wrist", "thumb_metacarpal"), ("thumb_metacarpal", "thumb_proximal"), ("thumb_proximal", "thumb_distal"),
         ("thumb_distal", "thumb_tip")]
for _f in FINGERS:
    _chain = ["wrist"] + [f"{_f}_{j}" for j in ("metacarpal", "proximal", "intermediate", "distal", "tip")]
    BONES += list(zip(_chain[:-1], _chain[1:]))
BONES += [("index_proximal", "middle_proximal"), ("middle_proximal", "ring_proximal"), ("ring_proximal", "little_proximal")]
BONE_DOTS = 6


def skeleton_markers(path: str, color: tuple[float, float, float]) -> VisualizationMarkers:
    """Joints (bigger spheres) + bones (rows of small spheres), one colour."""
    mat = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path=path, markers={
        "joint": sim_utils.SphereCfg(radius=0.007, visual_material=mat),
        "bone": sim_utils.SphereCfg(radius=0.003, visual_material=mat)}))


def draw_skeleton(markers: VisualizationMarkers, joints: dict[str, np.ndarray]) -> None:
    pts = [np.asarray(joints[n][:3]) for n in HAND_JOINT_NAMES if n != "palm"]
    t = np.linspace(0.0, 1.0, BONE_DOTS + 2)[1:-1, None]
    bones = [np.asarray(joints[a][:3]) * (1 - t) + np.asarray(joints[b][:3]) * t for a, b in BONES]
    points = np.concatenate([np.stack(pts), *bones])
    markers.visualize(translations=points, marker_indices=np.r_[np.zeros(len(pts), int), np.ones(len(points) - len(pts), int)])


def flat_hand_template(side: str) -> dict[str, np.ndarray]:
    """Fixed open flat hand (fingers together) in the OpenXR wrist frame: +Y back of hand, -Z toward the
    fingers. X = Y x Z puts the thumb at -X for a right hand, +X for a left hand."""
    s = -1.0 if side == "right" else 1.0  # thumb side sign along X
    pts = {"wrist": (0.0, 0.0, 0.0),
           "thumb_metacarpal": (0.022, 0.0, -0.025), "thumb_proximal": (0.042, -0.005, -0.050),
           "thumb_distal": (0.052, -0.008, -0.080), "thumb_tip": (0.058, -0.010, -0.100)}
    for f, x, length in (("index", 0.022, 1.0), ("middle", 0.0, 1.08), ("ring", -0.020, 1.0), ("little", -0.038, 0.82)):
        pts[f"{f}_metacarpal"] = (x * 0.5, 0.0, -0.02)
        z = -0.085 * (0.95 if f == "little" else 1.0)
        pts[f"{f}_proximal"] = (x, 0.0, z)
        for j, seg in (("intermediate", 0.042), ("distal", 0.026), ("tip", 0.022)):
            z -= seg * length
            pts[f"{f}_{j}"] = (x, 0.0, z)
    pts["palm"] = (0.0, 0.0, -0.05)
    return {k: np.array([s * v[0], v[1], v[2]]) for k, v in pts.items()}


def place_skeleton(template: dict[str, np.ndarray], pos: np.ndarray, rot: np.ndarray) -> dict[str, np.ndarray]:
    return {k: pos + rot @ v for k, v in template.items()}


def frame_markers(path: str) -> VisualizationMarkers:
    cfg = FRAME_MARKER_CFG.copy()
    cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
    return VisualizationMarkers(cfg.replace(prim_path=path))


def place_target(head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Target wrist in front of / below the head; palm down, fingers along the horizontal view direction."""
    fwd = -rot_from_wxyz(head[3:])[:, 2]
    fwd[2] = 0.0
    fwd /= max(np.linalg.norm(fwd), 1e-6)
    pos = head[:3] + args.target_forward * fwd - np.array([0.0, 0.0, args.target_down])
    y, z = np.array([0.0, 0.0, 1.0]), -fwd  # OpenXR wrist: +Y back of hand, -Z toward the fingers
    return pos, np.stack([np.cross(y, z), y, z], axis=1)


def save_rotation(mount_rot: np.ndarray, method: str = "calibrate_controller_rotation.py (target hand: palm down, fingers forward)") -> None:
    with open(args.offset_json) as f:
        calib = json.load(f)
    calib[args.side]["mount_rot_ctrl_to_wrist"] = mount_rot.round(6).tolist()
    calib[args.side]["mount_rot_date"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    calib[args.side]["mount_rot_method"] = method
    with open(args.offset_json, "w") as f:
        json.dump(calib, f, indent=2)
        f.write("\n")


def run_handtracking(xr, offset: np.ndarray, live_skel, live_wrist, live_axes) -> None:
    """Rotation from optical hand tracking: R_mount = mean(R_ctrl^T R_hand) while A/X capture runs."""
    first_ctrl, last_wrist, capture = None, None, []
    a_prev, b_down_since, frame, fresh = False, None, 0, False
    print("[rot] handtracking mode: keep the hand in view, still, then A/X. Hold B/Y 2 s to quit.")
    while simulation_app.is_running():
        simulation_app.update()
        frame += 1
        data = xr._get_raw_data()
        ctrl = data.get(CONTROLLER[args.side])
        hand = data.get(DeviceBase.TrackingTarget.HAND_RIGHT if args.side == "right" else DeviceBase.TrackingTarget.HAND_LEFT)
        wrist = None if not isinstance(hand, dict) else hand.get("wrist")
        ctrl_ok = ctrl is not None and len(ctrl) > 0
        if ctrl_ok:
            pose = np.asarray(ctrl[POSE], dtype=np.float64)
            first_ctrl = pose.copy() if first_ctrl is None else first_ctrl
            ctrl_ok = np.abs(pose - first_ctrl).max() > 1e-6
        if wrist is not None:
            w = np.asarray(wrist, dtype=np.float64)
            fresh = bool(np.any(w[:3])) and (last_wrist is None or not np.array_equal(w, last_wrist))
            last_wrist = w
        else:
            fresh = False
        if frame % 90 == 0:
            msg = f"[rot] controller={'tracked' if ctrl_ok else 'none'} hand tracking={'tracked' if fresh else 'none/stale'}"
            if ctrl_ok and fresh:
                r_ctrl = rot_from_wxyz(pose[3:])
                pivot_wrist = pose[:3] + r_ctrl @ offset
                msg += f", pivot wrist vs hand-tracking wrist {np.linalg.norm(pivot_wrist - last_wrist[:3]) * 100:.1f} cm"
            print(msg)
        if not ctrl_ok:
            continue
        r_ctrl = rot_from_wxyz(pose[3:])
        live_wrist.visualize(translations=(pose[:3] + r_ctrl @ offset)[None])
        if fresh:
            draw_skeleton(live_skel, hand)
            live_axes.visualize(translations=last_wrist[:3][None], orientations=last_wrist[3:][None])
        a_now = ctrl[INPUTS][BUTTON_0] > 0.5
        if a_now and not a_prev and not capture:
            capture = [None]
            print("[rot] capturing, keep still with the hand in view ...")
        elif capture and fresh:
            capture.append(r_ctrl.T @ rot_from_wxyz(last_wrist[3:]))
            if len(capture) > CAPTURE_FRAMES:
                samples = np.stack(capture[1:])
                mean = R.from_matrix(samples).mean()
                spread = max(np.degrees(np.linalg.norm((R.from_matrix(m) * mean.inv()).as_rotvec())) for m in samples)
                save_rotation(mean.as_matrix(), method="calibrate_controller_rotation.py --mode handtracking")
                print(f"[rot] saved controller->hand rotation to {args.offset_json} "
                      f"({np.degrees(np.linalg.norm(mean.as_rotvec())):.0f} deg; samples spread up to {spread:.1f} deg, "
                      f"{len(samples)} frames). Large spread = noisy hand tracking: retry or use --mode target.")
                capture = []
        a_prev = a_now
        b_now = ctrl[INPUTS][BUTTON_1] > 0.5
        if b_now and b_down_since is None:
            b_down_since = time.monotonic()
        if not b_now:
            b_down_since = None
        if b_down_since is not None and time.monotonic() - b_down_since > QUIT_HOLD_S:
            print("[rot] quit")
            return


def main() -> None:
    with open(args.offset_json) as f:
        calib = json.load(f)
    if args.side not in calib:
        raise SystemExit(f"{args.offset_json} has no '{args.side}' entry: run the pivot calibration for this hand first")
    offset = np.asarray(calib[args.side]["offset_ctrl_m"], dtype=np.float64)
    mount_rot = np.eye(3)  # a saved rotation is not used here: this run measures a new one
    print(f"[rot] {args.side}: pivot offset {np.round(offset, 4).tolist()} m, "
          f"{'saved' if 'mount_rot_ctrl_to_wrist' in calib[args.side] else 'no'} rotation yet")

    sim = SimulationContext(SimulationCfg(dt=1 / 90))
    if args.no_passthrough:
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    else:
        # Passthrough: no ground / sky, and a zero-alpha background so the headset shows the real room
        # (your real hand and the controller) with only the markers on top.
        import carb

        settings = carb.settings.get_settings()
        settings.set_bool("/rtx/post/backgroundZeroAlpha/enabled", True)
        settings.set_bool("/rtx/post/backgroundZeroAlpha/backgroundComposite", False)
        sim_utils.DistantLightCfg(intensity=3000.0).func("/World/light", sim_utils.DistantLightCfg(intensity=3000.0))
    xr = OpenXRDevice(OpenXRDeviceCfg(xr_cfg=XrCfg(anchor_pos=(0.0, 0.0, 0.0))))
    xr._required_features = {RetargeterBase.Requirement.MOTION_CONTROLLER}
    if args.mode == "handtracking":
        xr._required_features.add(RetargeterBase.Requirement.HAND_TRACKING)
    target_skel = skeleton_markers("/Visuals/calib_target_hand", (0.1, 0.9, 0.2))
    target_wrist = hand_markers("/Visuals/calib_target_wrist", (0.1, 0.9, 0.2), 0.02)
    live_skel = skeleton_markers("/Visuals/calib_live_hand", (0.95, 0.15, 0.1))
    live_wrist = hand_markers("/Visuals/calib_live_wrist", (1.0, 0.85, 0.0), 0.02)
    target_axes = axis_markers("/Visuals/calib_target_axes")
    live_axes = frame_markers("/Visuals/calib_live_axes")
    template = flat_hand_template(args.side)
    calibrated = False  # red hand and its axes are drawn once this run has captured a rotation
    live_skel.set_visibility(False)
    live_axes.set_visibility(False)
    sim.reset()
    request_ar_session(simulation_app)
    if args.mode == "handtracking":
        run_handtracking(xr, offset, live_skel, live_wrist, live_axes)
        return
    glove = ManusBridgeClient(args.host, args.port, 0.5)

    first_ctrl, target, local_map = None, None, None
    capture: list[np.ndarray] = []
    b_down_since, b_prev, a_prev, frame = None, False, False, 0
    print("[rot] Yellow = your wrist. Put it on the green wrist, hold your hand like the green hand, then A/X. "
          "B/Y: re-place the target, hold 2 s to quit.")
    while simulation_app.is_running():
        simulation_app.update()
        frame += 1
        ctrl = xr._get_raw_data().get(CONTROLLER[args.side])
        head = np.asarray(xr._calculate_headpose(), dtype=np.float64)
        gloves = glove.gloves()
        if ctrl is None or len(ctrl) == 0 or args.side not in gloves:
            if frame % 270 == 0:
                print(f"[rot] waiting: controller={'ok' if ctrl is not None and len(ctrl) else 'none'} "
                      f"glove={'ok' if args.side in gloves else 'none'} (bridge {glove.err or 'ok'})")
            continue
        pose = np.asarray(ctrl[POSE], dtype=np.float64)
        first_ctrl = pose.copy() if first_ctrl is None else first_ctrl
        if np.abs(pose - first_ctrl).max() < 1e-6:
            continue  # placeholder pose until the headset streams the controller
        if target is None:
            if not np.any(head[:3]):
                continue
            target = place_target(head)
            print(f"[rot] target placed at {np.round(target[0], 3).tolist()}")

        kp, wq = gloves[args.side]
        geometric = openxr_wrist_frame(kp, args.side)
        if geometric is None or not np.isfinite(kp).all():
            continue
        if wq is not None:  # glove wrist frame from the Manus wrist node, as in ManusDevice
            node = R.from_quat(wq).as_matrix()
            local_map = snap_signed_permutation(node.T @ geometric) if local_map is None else local_map
            glove_frame = node @ local_map
        else:
            glove_frame = geometric

        r_ctrl = rot_from_wxyz(pose[3:])
        wrist_pos, wrist_rot = pose[:3] + r_ctrl @ offset, r_ctrl @ mount_rot
        live = keypoints_to_openxr_joints(kp, glove_frame, wrist_pos, wrist_rot)
        draw_skeleton(target_skel, place_skeleton(template, target[0], target[1]))  # fixed flat hand
        target_wrist.visualize(translations=target[0][None])
        draw_axes(target_axes, target[0], target[1])
        live_wrist.visualize(translations=wrist_pos[None])
        if calibrated:
            draw_skeleton(live_skel, live)
            live_axes.visualize(translations=wrist_pos[None], orientations=np.roll(R.from_matrix(wrist_rot).as_quat(), 1)[None])
        if frame % 90 == 0:
            msg = f"[rot] wrist to target wrist: {np.linalg.norm(wrist_pos - target[0]) * 100:.1f} cm"
            if calibrated:
                ang = np.degrees(np.linalg.norm(R.from_matrix(target[1] @ wrist_rot.T).as_rotvec()))
                msg += f", hand vs target rotation {ang:.0f} deg"
            print(msg)

        a_now = ctrl[INPUTS][BUTTON_0] > 0.5
        if a_now and not a_prev and not capture:
            capture = [r_ctrl]
            print("[rot] capturing, hold still ...")
        elif capture:
            capture.append(r_ctrl)
            if len(capture) >= CAPTURE_FRAMES:
                r_mean = R.from_matrix(np.stack(capture)).mean().as_matrix()
                spread = max(np.degrees(np.linalg.norm(R.from_matrix(c @ r_mean.T).as_rotvec())) for c in capture)
                mount_rot = r_mean.T @ target[1]
                save_rotation(mount_rot)
                calibrated = True
                live_skel.set_visibility(True)
                live_axes.set_visibility(True)
                print(f"[rot] saved controller->hand rotation to {args.offset_json} "
                      f"({np.degrees(np.linalg.norm(R.from_matrix(mount_rot).as_rotvec())):.0f} deg; "
                      f"hand moved up to {spread:.1f} deg while capturing). The red hand now uses it.")
                capture = []
        a_prev = a_now

        b_now = ctrl[INPUTS][BUTTON_1] > 0.5
        if b_now and not b_prev:
            b_down_since = time.monotonic()
        if b_now and b_down_since is not None and time.monotonic() - b_down_since > QUIT_HOLD_S:
            print("[rot] quit")
            break
        if not b_now and b_prev and b_down_since is not None:
            target = place_target(head)
            print(f"[rot] target moved to {np.round(target[0], 3).tolist()}")
        b_prev = b_now


if __name__ == "__main__":
    main()
    # SimulationApp.close() can hang in an XR session and keep the headset session: exit directly.
    sys.stdout.flush()
    os._exit(0)
