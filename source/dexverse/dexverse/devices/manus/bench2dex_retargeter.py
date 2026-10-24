# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DexVerse hand retargeter whose finger part is Bench2Dex's Manus-tuned ``RetargetBridge``.

The wrist command and the action layout stay those of :class:`SimpleRelativeRetargeter`; only
the finger retargeting is replaced. ``RetargetBridge`` (vendored from the Bench2Dex teleop code
in ``bench2dex/``) maps 21 MediaPipe keypoints in a canonical wrist frame to robot joint angles
with DexPilot, using per-hand YAMLs (``bench2dex/hand_cfgs``) whose scale / offset / joint locks
were tuned on the Manus skeleton, and the Bench2Dex URDF-Zoo URDFs (``$URDF_ZOO_DIR``).

Input is the OpenXR-style hand payload produced by :class:`~dexverse.devices.manus.ManusDevice`
(or OpenXR hand tracking). Output joints are matched to the robot action by joint name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase
from scipy.spatial.transform import Rotation as R

from ..retargeters.simple_relative_retargeting import SimpleRelativeRetargeter, SimpleRelativeRetargeterCfg

# OpenXR joint names in MediaPipe 21-keypoint order (wrist, thumb CMC/MCP/IP/TIP, then
# MCP/PIP/DIP/TIP per finger; the finger metacarpals are skipped, as in Bench2Dex).
_MEDIAPIPE_FROM_OPENXR = (
    "wrist",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "little_proximal", "little_intermediate", "little_distal", "little_tip",
)

# Columns = RetargetBridge canonical axes expressed in the OpenXR wrist joint frame.
_OPENXR_JOINT_TO_CANONICAL = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# DexVerse hand family -> Bench2Dex hand_type (hand_cfgs/<hand_type>.yml).
_BENCH2DEX_HAND_TYPE = {
    "shadow": "shadow",
    "allegro": "allegro",
    "leap": "leap",
    "sharpa": "sharpa",
    "wuji": "wuji",
    "inspire": "rh56dfx",
}

_TARGET_SIDE = {DeviceBase.TrackingTarget.HAND_LEFT: "left", DeviceBase.TrackingTarget.HAND_RIGHT: "right"}


def _hand_family(robot_type: str) -> str:
    parts = robot_type.split("_")  # floating_<hand>_<side>
    if len(parts) < 3 or parts[0] != "floating":
        raise ValueError(f"Bench2Dex finger retargeting expects floating_<hand>_<side>, got '{robot_type}'.")
    return parts[1]


class Bench2DexFingerRetargeter(SimpleRelativeRetargeter):
    """Relative wrist (displacement from the START pose) + Bench2Dex ``RetargetBridge`` fingers."""

    def __init__(self, cfg: Bench2DexFingerRetargeterCfg):
        cfg.initialize_dex_retargeting = False  # the bridge replaces the DexVerse dex-retargeting YAMLs
        super().__init__(cfg)
        from .bench2dex.retarget_bridge import RetargetBridge

        family = _hand_family(cfg.robot_type)
        hand_type = cfg.bench2dex_hand_type or _BENCH2DEX_HAND_TYPE.get(family)
        if hand_type is None:
            raise ValueError(f"No Bench2Dex retarget config for hand family '{family}'.")
        self._bridges = {}
        for hand in self._tracked_hands:
            bridge = RetargetBridge(hand_type, _TARGET_SIDE[hand], input_device=cfg.bench2dex_input_device)
            self._bridges[hand] = bridge
            robot = bridge._retargeting.optimizer.robot
            self._dex_output_joint_names[hand] = [str(n) for n in robot.dof_joint_names]
            self._dex_to_action_finger_indices[hand] = self._build_finger_name_mapping(hand)
            missing = [
                name
                for name, idx in zip(self._layout["hands"][hand].get("finger_joint_names", ()), self._dex_to_action_finger_indices[hand])
                if idx is None
            ]
            print(f"[Bench2DexFingerRetargeter] {hand.name}: {bridge}" + (f", unmatched action joints {missing}" if missing else ""))
        self._calls = 0
        self.robot_wrist_pose_w: np.ndarray | None = None

    def set_robot_wrist_pose(self, pose_w: np.ndarray) -> None:
        """Robot wrist world pose [x, y, z, qw, qx, qy, qz], pushed every step by teleop_agent / record_demos."""
        self.robot_wrist_pose_w = np.asarray(pose_w, dtype=np.float64)

    def _retarget_hand_fingers(self, hand_data: Any, hand: DeviceBase.TrackingTarget) -> np.ndarray:
        bridge = self._bridges.get(hand)
        n_out = len(self._dex_output_joint_names.get(hand, []))
        if bridge is None or not isinstance(hand_data, dict) or "wrist" not in hand_data:
            return np.zeros(n_out, dtype=np.float32)
        try:
            keypoints = np.stack([np.asarray(hand_data[n], dtype=np.float32)[:3] for n in _MEDIAPIPE_FROM_OPENXR])
        except KeyError:
            return np.zeros(n_out, dtype=np.float32)
        wrist_quat = np.asarray(hand_data["wrist"], dtype=np.float64)[3:]  # OpenXR wrist frame, wxyz
        rot = R.from_quat([wrist_quat[1], wrist_quat[2], wrist_quat[3], wrist_quat[0]]).as_matrix()
        # RetargetBridge wrist frame from the OpenXR wrist joint frame: (+Y, -X, +Z), i.e. the joint
        # frame turned +90 deg about Z (Bench2Dex xr_hand_source.OPENXR_JOINT_TO_CANONICAL, verified
        # against the URDF-Zoo hands' zero-pose fingertips).
        wrist_frame = (rot @ _OPENXR_JOINT_TO_CANONICAL).astype(np.float32)
        self._calls += 1
        if self._calls % 120 == 0:
            bridge.reload_tunables_if_changed()  # edit hand_cfgs/*.yml scale / offset live
        with torch.enable_grad(), torch.inference_mode(False):
            qpos = bridge.retarget_keypoints(keypoints, wrist_frame)
        return np.zeros(n_out, dtype=np.float32) if qpos is None else np.asarray(qpos, dtype=np.float32)


@dataclass
class Bench2DexFingerRetargeterCfg(SimpleRelativeRetargeterCfg):
    """Configuration for :class:`Bench2DexFingerRetargeter`."""

    bench2dex_hand_type: str | None = None
    """Override the Bench2Dex hand config (``hand_cfgs/<name>.yml``); default from ``robot_type``."""
    bench2dex_input_device: str = "manus"
    """``"manus"`` uses the Manus-tuned hand scale / offset, ``"xr"`` the OpenXR ones."""
    retargeter_type: type[RetargeterBase] = Bench2DexFingerRetargeter
