# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manus glove teleop device that feeds the existing DexVerse hand retargeters.

The glove skeleton comes from the Manus bridge (``robo-viz/manus/bridge.py``), which owns
the dongle through the Manus SDK and streams line-delimited JSON over TCP (default
``127.0.0.1:47000``). Each message carries, per glove, the 25 raw-skeleton nodes in WebXR
order (0 wrist, thumb 1-4, index 5-9, middle 10-14, ring 15-19, little 20-24, metres) and the
wrist-node quaternion.

:class:`ManusDevice` turns that into the payload :class:`isaaclab.devices.OpenXRDevice` produces
for hand tracking: ``{HAND_LEFT/HAND_RIGHT: {joint_name: [x, y, z, qw, qx, qy, qz]}}`` over the
26 OpenXR joints, so the DexVerse hand retargeters work unchanged. The finger shape is expressed
in the glove's wrist frame (Manus wrist node, mapped to the OpenXR joint convention: +Y out of the
back of the hand, -Z toward the fingertips) and re-placed at a wrist pose chosen by ``wrist_mode``:

* ``"fixed"``: a fixed wrist, palm down with the fingers along +X (fingers only, no headset).
* ``"controller"``: the Quest controller mounted on the back of the same hand. The pivot
  calibration (``controller_offset_json``, from ``pivot_calibration.py``) moves the tracked point
  to the wrist rotation centre; the controller-to-hand rotation is taken at START, with the hand
  held like the robot hand.
* ``"handtracking"``: the headset's optical hand tracking (OpenXR wrist joint).

With ``xr_cfg`` (a VR session) the device also owns an :class:`OpenXRDevice` without retargeters:
it places the XR anchor, streams the controller / hand-tracking wrist, and relays the headset
menu's START / STOP / RESET. On START the anchor is moved so the operator's hand sits on the robot
hand (``recenter_on_start``), then the retargeter calibrates its wrist reference.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from scipy.spatial.transform import Rotation as R

# WebXR 25-node index -> OpenXR 26-joint name (OpenXR adds "palm" in front).
_WEBXR_TO_OPENXR = HAND_JOINT_NAMES[1:]
_INDEX_PROXIMAL, _MIDDLE_METACARPAL, _MIDDLE_PROXIMAL, _LITTLE_PROXIMAL = 6, 10, 11, 21

# OpenXR wrist frame for a palm-down hand with fingers along world +X:
# columns are the wrist X, Y, Z axes in world coordinates (Y = back of hand = +Z_w, Z = -X_w).
_PALM_DOWN_FINGERS_X = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# Inverse of this maps a robot wrist orientation back to an OpenXR wrist orientation: the DexVerse
# retargeters normalise the OpenXR wrist with R_hand * Ry(90) * Rx(-90).
_DEXVERSE_HAND_NORMALIZE = R.from_euler("y", 90, degrees=True) * R.from_euler("x", -90, degrees=True)

_SIDES = {"right": DeviceBase.TrackingTarget.HAND_RIGHT, "left": DeviceBase.TrackingTarget.HAND_LEFT}
_CONTROLLERS = {"right": DeviceBase.TrackingTarget.CONTROLLER_RIGHT, "left": DeviceBase.TrackingTarget.CONTROLLER_LEFT}

# Frames between moving the anchor / re-orienting the wrist on START and the retargeter's calibration,
# so the calibration sees the updated wrist.
_START_DELAY_FRAMES = 3


def _rot_from_wxyz(q: np.ndarray) -> np.ndarray:
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def openxr_wrist_frame(kp: np.ndarray, side: str) -> np.ndarray | None:
    """Wrist frame (3x3, columns X/Y/Z) in the OpenXR joint convention, from 25 WebXR keypoints."""
    z = kp[0] - kp[_MIDDLE_PROXIMAL]  # toward the wrist, so -Z points to the fingers
    a = kp[_INDEX_PROXIMAL] - kp[0]
    b = kp[_LITTLE_PROXIMAL] - kp[0]
    back = np.cross(b, a) if side == "right" else np.cross(a, b)
    zn = np.linalg.norm(z)
    if zn < 1e-6:
        return None
    z = z / zn
    y = back - back.dot(z) * z
    yn = np.linalg.norm(y)
    if yn < 1e-6:
        return None
    y = y / yn
    return np.stack([np.cross(y, z), y, z], axis=1)


def snap_signed_permutation(m: np.ndarray) -> np.ndarray:
    """Nearest signed permutation matrix (each column / row has a single +-1)."""
    p = np.zeros((3, 3))
    used = set()
    for col in np.argsort(-np.abs(m).max(axis=0)):
        row = max((r for r in range(3) if r not in used), key=lambda r: abs(m[r, col]))
        used.add(row)
        p[row, col] = np.sign(m[row, col])
    return p


def keypoints_to_openxr_joints(
    kp: np.ndarray, frame: np.ndarray, wrist_pos: np.ndarray, wrist_rot: np.ndarray
) -> dict[str, np.ndarray]:
    """Glove keypoints (25, 3) with glove wrist frame ``frame`` -> 26 OpenXR joints re-placed at
    ``wrist_pos`` / ``wrist_rot`` (both 3x3 in the OpenXR joint convention)."""
    local = (kp - kp[0]) @ frame  # keypoints in the glove wrist frame
    world = wrist_pos[None, :] + local @ wrist_rot.T
    q = R.from_matrix(wrist_rot).as_quat()  # xyzw
    quat_wxyz = np.array([q[3], q[0], q[1], q[2]])
    joints = {name: np.concatenate([world[i], quat_wxyz]).astype(np.float32) for i, name in enumerate(_WEBXR_TO_OPENXR)}
    palm = 0.5 * (world[_MIDDLE_METACARPAL] + world[_MIDDLE_PROXIMAL])
    joints["palm"] = np.concatenate([palm, quat_wxyz]).astype(np.float32)
    return joints


class ManusBridgeClient:
    """Background TCP client for the Manus bridge; keeps only the latest message."""

    def __init__(self, host: str, port: int, stale_s: float):
        self.host, self.port, self.stale_s = host, int(port), float(stale_s)
        self._lock = threading.Lock()
        self._msg: dict | None = None
        self._t = 0.0
        self.connected = False
        self.err = ""
        self.n = 0
        threading.Thread(target=self._run, daemon=True, name="manus-bridge-client").start()

    def _run(self) -> None:
        while True:
            try:
                with socket.create_connection((self.host, self.port), timeout=2.0) as s:
                    s.settimeout(2.0)
                    self.connected, self.err = True, ""
                    buf = b""
                    while True:
                        chunk = s.recv(65536)
                        if not chunk:
                            raise ConnectionError("bridge closed")
                        buf += chunk
                        *lines, buf = buf.split(b"\n")
                        if lines:
                            msg = json.loads(lines[-1])  # latest line only
                            with self._lock:
                                self._msg, self._t = msg, time.monotonic()
                                self.n += len(lines)
            except Exception as exc:  # noqa: BLE001 - reconnect on any socket / parse error
                self.connected = False
                self.err = f"no Manus bridge at {self.host}:{self.port} ({type(exc).__name__})"
                time.sleep(1.0)

    def gloves(self) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
        """``{side: (kp (25, 3), wrist quaternion xyzw or None)}`` for gloves seen within ``stale_s``."""
        with self._lock:
            msg, t = self._msg, self._t
        if msg is None or time.monotonic() - t > self.stale_s:
            return {}
        out = {}
        for g in msg.get("gloves", []):
            if g.get("age_s", 0.0) <= self.stale_s and g.get("side") in _SIDES:
                wq = g.get("wq")
                out[g["side"]] = (
                    np.asarray(g["kp"], dtype=np.float64).reshape(25, 3),
                    None if wq is None else np.asarray(wq, dtype=np.float64),
                )
        return out


class ManusDevice(DeviceBase):
    """Manus glove device emitting OpenXR-style hand joints for the DexVerse retargeters."""

    def __init__(self, cfg: ManusDeviceCfg, retargeters: list | None = None):
        super().__init__(retargeters)
        if cfg.wrist_mode not in ("fixed", "controller", "handtracking"):
            raise ValueError(f"wrist_mode must be 'fixed', 'controller' or 'handtracking', got '{cfg.wrist_mode}'")
        self.cfg = cfg
        self._client = ManusBridgeClient(cfg.host, cfg.port, cfg.stale_s)
        self._callbacks: dict[str, Callable] = {}
        default = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self._last = {side: {name: default.copy() for name in HAND_JOINT_NAMES} for side in _SIDES}
        self._seen = {side: False for side in _SIDES}  # glove data received
        self._local_to_openxr: dict[str, np.ndarray] = {}  # Manus wrist-node axes -> OpenXR wrist frame
        self._started = False
        self._pending_start = 0
        self._last_log = 0.0
        # Tracked wrist (controller / hand tracking): pose per side, world position + OpenXR wrist frame.
        self._wrist: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._wrist_first_pose: dict[str, np.ndarray] = {}
        self._ctrl_rot: dict[str, np.ndarray] = {}
        self._mount_rot = {side: np.eye(3) for side in _SIDES}  # controller frame -> OpenXR wrist frame
        self._ctrl_offset = {side: np.zeros(3) for side in _SIDES}
        if cfg.controller_offset_json:
            with open(cfg.controller_offset_json) as f:
                calib = json.load(f)
            for side in _SIDES:
                if side in calib:
                    self._ctrl_offset[side] = np.asarray(calib[side]["offset_ctrl_m"], dtype=np.float64)
            print(f"[manus] controller->wrist offsets from {cfg.controller_offset_json}: "
                  + ", ".join(f"{s}={np.round(v, 4).tolist()}" for s, v in self._ctrl_offset.items()))

        self._xr = None
        if cfg.xr_cfg is not None:
            try:
                from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg
                from isaaclab.devices.retargeter_base import RetargeterBase

                self._xr = OpenXRDevice(OpenXRDeviceCfg(xr_cfg=cfg.xr_cfg, sim_device=cfg.sim_device))
                # No retargeter is attached to this device, so request the wrist stream explicitly.
                self._xr._required_features = {
                    "controller": {RetargeterBase.Requirement.MOTION_CONTROLLER},
                    "handtracking": {RetargeterBase.Requirement.HAND_TRACKING},
                }.get(cfg.wrist_mode, set())
            except Exception as exc:  # noqa: BLE001 - XR extensions are absent outside an XR app
                print(f"[manus] no XR session ({type(exc).__name__}: {exc}); running without headset commands")
        if cfg.wrist_mode != "fixed" and self._xr is None:
            raise RuntimeError(f"wrist_mode='{cfg.wrist_mode}' needs an XR session (xr_cfg and a running CloudXR runtime)")

    def __str__(self) -> str:
        xr = ", xr" if self._xr is not None else ""
        return f"ManusDevice(bridge={self.cfg.host}:{self.cfg.port}, wrist_mode={self.cfg.wrist_mode}{xr})"

    def reset(self) -> None:
        self._started = False
        if self._xr is not None:
            self._xr.reset()

    def add_callback(self, key: Any, func: Callable) -> None:
        self._callbacks[key] = func
        if self._xr is not None:
            # START from the headset menu goes through _request_start so it can recenter first.
            self._xr.add_callback(key, self._request_start if key == "START" else func)

    def status(self) -> dict:
        return {"connected": self._client.connected, "err": self._client.err, "msgs": self._client.n,
                "sides": sorted(self._client.gloves())}

    # ----------------------------------------------------------------- START: align operator and robot

    def _request_start(self) -> None:
        """START: take the controller-to-hand rotation and move the XR anchor so the operator's hand
        sits on the robot hand, then fire the START callbacks (the retargeter's wrist calibration)
        once the updated wrist has reached the retargeter."""
        rotated = self.cfg.wrist_mode == "controller" and self._calibrate_mount_rotation()
        moved = self.cfg.recenter_on_start and self.cfg.wrist_mode != "fixed" and self._recenter()
        if rotated or moved:
            self._pending_start = _START_DELAY_FRAMES
        elif "START" in self._callbacks:
            self._callbacks["START"]()

    def _robot_wrist_pose(self) -> np.ndarray | None:
        """Robot wrist world pose [x, y, z, qw, qx, qy, qz] pushed to the retargeter by the teleop scripts."""
        for r in self._retargeters:
            pose = getattr(r, "robot_wrist_pose_w", None)
            if pose is not None:
                return np.asarray(pose, dtype=np.float64)
        return None

    def _robot_sides(self) -> list[str]:
        """Tracked sides the robot is driven by (sides the retargeter tracks, all if unknown)."""
        robot = [side for side, target in _SIDES.items() for r in self._retargeters
                 if target in getattr(r, "_tracked_hands", ())]
        return [s for s in (robot or _SIDES) if s in self._wrist]

    def _calibrate_mount_rotation(self) -> bool:
        """Controller-to-hand rotation, assuming the hand is held like the robot hand at START.

        The robot wrist orientation is mapped back to the OpenXR wrist convention, so afterwards the
        tracked wrist, the red keypoints and the robot hand share one orientation. Without a robot
        wrist pose the reference is palm down with the fingers toward the headset's horizontal forward.
        """
        sides = [s for s in self._robot_sides() if s in self._ctrl_rot]
        if not sides:
            print("[manus] rotation calibration skipped (controller not tracked yet)")
            return False
        pose = self._robot_wrist_pose()
        if pose is not None:
            ref = (R.from_matrix(_rot_from_wxyz(pose[3:])) * _DEXVERSE_HAND_NORMALIZE.inv()).as_matrix()
            how = "robot hand orientation"
        else:
            fwd = -_rot_from_wxyz(np.asarray(self._xr._calculate_headpose(), dtype=np.float64)[3:])[:, 2]
            fwd[2] = 0.0
            fwd /= max(np.linalg.norm(fwd), 1e-6)
            y, z = np.array([0.0, 0.0, 1.0]), -fwd  # back of hand up, -Z toward the fingers
            ref, how = np.stack([np.cross(y, z), y, z], axis=1), "palm down, fingers forward"
        for side in sides:
            self._mount_rot[side] = self._ctrl_rot[side].T @ ref
            angle = np.degrees(np.linalg.norm(R.from_matrix(self._mount_rot[side]).as_rotvec()))
            print(f"[manus] {side} controller->hand rotation calibrated from {how} ({angle:.0f} deg)")
        return True

    def _recenter(self) -> bool:
        """Translate the XR anchor by (robot wrist - operator wrist) so the two coincide."""
        pose = self._robot_wrist_pose()
        sides = [s for s in self._robot_sides() if self._seen[s]]
        if pose is None or not sides:
            print("[manus] recenter skipped (no robot wrist pose, or the robot's hand is not tracked yet)")
            return False
        side = sides[0]
        delta = pose[:3] - self._wrist[side][0]
        from isaacsim.core.prims import SingleXFormPrim

        anchor = SingleXFormPrim(self._xr._xr_anchor_headset_path)
        pos, rot = anchor.get_world_pose()
        anchor.set_world_pose(position=np.asarray(pos, dtype=np.float64) + delta, orientation=rot)
        print(f"[manus] recentered: XR anchor moved by {np.round(delta, 3).tolist()} m ({side} hand -> robot wrist)")
        return True

    # ----------------------------------------------------------------- wrist sources

    def _track(self, side: str, pose: np.ndarray, what: str) -> bool:
        """Before the headset streams a source its pose is a constant placeholder (the XR anchor, or
        zeros); a tracked pose always jitters, so a side counts as tracked once its pose first changes."""
        if side in self._wrist:
            return True
        first = self._wrist_first_pose.setdefault(side, pose.copy())
        if np.abs(pose - first).max() < 1e-6:
            return False
        print(f"[manus] {side} {what} tracked at {np.round(pose[:3], 3).tolist()}")
        return True

    def _update_controller_wrists(self) -> None:
        """Wrist = controller pose (row: x, y, z, qw, qx, qy, qz) moved by the pivot offset, rotated by the mount."""
        data = self._xr._get_raw_data()
        for side, target in _CONTROLLERS.items():
            ctrl = data.get(target)
            if ctrl is None or len(ctrl) == 0:
                continue
            pose = np.asarray(ctrl[DeviceBase.MotionControllerDataRowIndex.POSE.value], dtype=np.float64)
            if not np.isfinite(pose).all():
                continue
            newly = side not in self._wrist
            if not self._track(side, pose, "controller"):
                continue
            rot = _rot_from_wxyz(pose[3:])
            self._ctrl_rot[side] = rot
            self._wrist[side] = (pose[:3] + rot @ self._ctrl_offset[side], rot @ self._mount_rot[side])
            if newly and self._started:
                self._request_start()  # recalibrate on the real controller pose

    def _update_tracked_hand_wrists(self) -> None:
        """Wrist = Quest optical hand-tracking wrist joint (already in the OpenXR joint convention)."""
        data = self._xr._get_raw_data()
        for side, target in _SIDES.items():
            joints = data.get(target)
            wrist = joints.get("wrist") if isinstance(joints, dict) else None
            if wrist is None:
                continue
            pose = np.asarray(wrist, dtype=np.float64)  # x, y, z, qw, qx, qy, qz
            if np.isfinite(pose).all() and self._track(side, pose, "hand (headset tracking)"):
                self._wrist[side] = (pose[:3], _rot_from_wxyz(pose[3:]))

    def _glove_frame(self, side: str, kp: np.ndarray, wq: np.ndarray | None) -> np.ndarray | None:
        """Glove wrist frame (OpenXR convention) from the Manus wrist node, like Bench2Dex.

        Which local axis of the wrist node is the finger / back-of-hand direction depends on the SDK
        coordinate-system settings, so the map is found once per glove from the palm keypoints and
        snapped to a signed axis permutation. Falls back to the keypoint frame without a quaternion.
        """
        geometric = openxr_wrist_frame(kp, side)
        if wq is None or geometric is None:
            return geometric
        node_rot = R.from_quat(wq).as_matrix()
        if side not in self._local_to_openxr:
            m = node_rot.T @ geometric
            p = snap_signed_permutation(m)
            err = np.degrees(np.arccos(np.clip((np.trace(p.T @ m) - 1) / 2, -1.0, 1.0)))
            self._local_to_openxr[side] = p
            print(f"[manus] {side} wrist-node axes -> OpenXR wrist frame:\n{p.astype(int)}  (palm-vs-node offset {err:.1f} deg)")
        return node_rot @ self._local_to_openxr[side]

    # ----------------------------------------------------------------- DeviceBase

    def _get_raw_data(self) -> dict:
        if self._pending_start:
            self._pending_start -= 1
            if not self._pending_start and "START" in self._callbacks:
                self._callbacks["START"]()
        if self.cfg.wrist_mode == "controller":
            self._update_controller_wrists()
        elif self.cfg.wrist_mode == "handtracking":
            self._update_tracked_hand_wrists()
        for side, (kp, wq) in self._client.gloves().items():
            if not np.isfinite(kp).all():
                continue
            if self.cfg.wrist_mode == "fixed":
                offset = np.array(self.cfg.left_offset if side == "left" else (0.0, 0.0, 0.0))
                pos, rot = np.asarray(self.cfg.wrist_pos, dtype=np.float64) + offset, _PALM_DOWN_FINGERS_X
            elif side in self._wrist:
                pos, rot = self._wrist[side]
            else:
                continue  # wait for the headset so START calibrates on a tracked wrist
            frame = self._glove_frame(side, kp, wq)
            if frame is not None:
                self._last[side] = keypoints_to_openxr_joints(kp, frame, pos, rot)
                self._seen[side] = True
        if self.cfg.auto_start and not self._started and any(self._seen.values()):
            self._started = True
            self._request_start()
        now = time.monotonic()
        if self.cfg.log_every_s and now - self._last_log > self.cfg.log_every_s:
            self._last_log = now
            st = self.status()
            print(f"[manus] bridge={'ok' if st['connected'] else st['err']} msgs={st['msgs']} gloves={st['sides']}")
        return {target: self._last[side] for side, target in _SIDES.items()}


@dataclass
class ManusDeviceCfg(DeviceCfg):
    """Configuration for :class:`ManusDevice`."""

    host: str = "127.0.0.1"
    port: int = 47000
    stale_s: float = 0.5
    wrist_mode: str = "fixed"  # "fixed" | "controller" | "handtracking"
    controller_offset_json: str | None = None  # pivot_calibration.py output (controller -> wrist offset)
    recenter_on_start: bool = True  # on START move the XR anchor so your hand sits on the robot hand
    auto_start: bool = True  # start (and calibrate) on the first glove frame instead of the headset START
    wrist_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # "fixed" mode wrist position
    left_offset: tuple[float, float, float] = (0.0, 0.3, 0.0)  # "fixed" mode left-hand offset
    log_every_s: float = 5.0
    xr_cfg: Any = None  # isaaclab XrCfg; set in a VR session (anchor, wrist stream, headset START/STOP)
    retargeters: list = field(default_factory=list)
    class_type: type[DeviceBase] = ManusDevice


def _bench2dex_fingers(retargeter_cfg):
    """Same retargeter settings, with the fingers retargeted by Bench2Dex's Manus-tuned RetargetBridge."""
    from dataclasses import fields

    from dexverse.devices.retargeters import SimpleRelativeRetargeterCfg

    from .bench2dex_retargeter import Bench2DexFingerRetargeterCfg

    if type(retargeter_cfg) is not SimpleRelativeRetargeterCfg:
        print(f"[manus] {type(retargeter_cfg).__name__} kept (Bench2Dex fingers replace the relative retargeter only)")
        return retargeter_cfg
    kwargs = {f.name: getattr(retargeter_cfg, f.name) for f in fields(SimpleRelativeRetargeterCfg) if f.name != "retargeter_type"}
    return Bench2DexFingerRetargeterCfg(**kwargs)


def add_manus_device(
    devices_cfg,
    name: str = "manus",
    source: str = "handtracking",
    use_xr: bool = False,
    fingers: str = "bench2dex",
    **overrides,
) -> None:
    """Add a Manus device that reuses the retargeters (and, with ``use_xr``, the XR anchor) of an existing entry.

    ``fingers="bench2dex"`` (default) retargets the fingers with Bench2Dex's Manus-tuned ``RetargetBridge``;
    ``fingers="dexverse"`` keeps the DexVerse dex-retargeting YAMLs (tuned for Quest hand tracking).
    """
    src = devices_cfg.devices[source]
    if use_xr:
        overrides.setdefault("xr_cfg", getattr(src, "xr_cfg", None))
    retargeters = list(src.retargeters)
    if fingers == "bench2dex":
        retargeters = [_bench2dex_fingers(r) for r in retargeters]
    elif fingers != "dexverse":
        raise ValueError(f"fingers must be 'bench2dex' or 'dexverse', got '{fingers}'")
    devices_cfg.devices[name] = ManusDeviceCfg(retargeters=retargeters, sim_device=src.sim_device, **overrides)
