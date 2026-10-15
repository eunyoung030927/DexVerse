# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XR session helpers for CloudXR / OpenXR hand-tracking teleoperation.

Two small pieces shared by ``teleop_agent.py`` and ``record_demos.py``:

* :func:`request_ar_session` starts the AR session programmatically, so the
  stream begins as soon as an XR client connects. Headless runs already
  auto-enable AR through ``isaaclab.python.xr.openxr.headless.kit``
  (``xr.profile.ar.enabled = true``); in GUI runs this replaces clicking
  **Start AR** in the viewport. The call is idempotent.
* :class:`XrStreamLogger` prints a one-line hand-tracking summary every N frames
  (wrist position, number of joints with a non-zero position, and a ``STALE``
  flag when the wrist pose is bit-for-bit frozen). It is the quickest way to
  tell "stream not arriving" (``0/26`` — check the runtime's push-device
  setting and the headset's hand-tracking permissions) from "hand left the
  tracking volume" (``STALE``; OpenXRDevice holds the last valid pose).

Both functions must be called after the Kit app has launched.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def request_ar_session(simulation_app: Any, num_updates: int = 10) -> bool:
    """Enable the AR profile so the XR session starts without the viewport button.

    Args:
        simulation_app: The running ``SimulationApp``.
        num_updates: App updates to pump afterwards so ``omni.kit.xr.profile.ar``
            activates the session.

    Returns:
        True if the request was issued, False if the XR core is unavailable.
    """
    try:
        from omni.kit.xr.core import XRCore

        XRCore.get_singleton().request_enable_profile("ar")
    except Exception as exc:  # noqa: BLE001 - XR extension missing or not loaded
        logger.warning(f"[xr] could not enable the AR profile programmatically: {exc}")
        return False
    for _ in range(num_updates):
        simulation_app.update()
    print("[xr] AR profile enabled; waiting for the XR client to connect.")
    return True


class XrStreamLogger:
    """Periodic hand-tracking stream summary for an Isaac Lab ``OpenXRDevice``.

    If the ``DEXVERSE_XR_DUMP`` environment variable names a ``.npz`` path, every frame's raw
    hand joints are also recorded there (``right``/``left``: ``(T, 26, 7)`` as
    ``[x, y, z, qw, qx, qy, qz]`` in ``joint_names`` order), rewritten every 60 frames so a
    Ctrl+C loses at most one second. Used to replay real operator hands through every
    robot's retargeting offline.
    """

    _DUMP_FLUSH_EVERY = 60

    def __init__(self, teleop_interface: Any, every_n_frames: int):
        self._device = teleop_interface
        self._every = int(every_n_frames)
        self._frame = 0
        self._last_wrist: dict[str, np.ndarray] = {}
        self._dump_path = os.environ.get("DEXVERSE_XR_DUMP") or None
        self._dump: dict[str, list[np.ndarray]] = {"right": [], "left": []}
        if self._dump_path:
            from isaaclab.devices.openxr.common import HAND_JOINT_NAMES

            self._joint_names = list(HAND_JOINT_NAMES)
            print(f"[xr] recording raw hand joints to {self._dump_path}")

    def _record(self) -> None:
        for side in ("right", "left"):
            poses = getattr(self._device, f"_previous_joint_poses_{side}", None) or {}
            self._dump[side].append(
                np.stack([np.asarray(poses.get(n, np.zeros(7)), dtype=np.float32) for n in self._joint_names])
            )
        if len(self._dump["right"]) % self._DUMP_FLUSH_EVERY == 0:
            np.savez(
                self._dump_path,
                right=np.stack(self._dump["right"]),
                left=np.stack(self._dump["left"]),
                joint_names=np.array(self._joint_names),
            )

    def step(self) -> None:
        """Call once per loop iteration; logs every ``every_n_frames`` frames."""
        if self._dump_path:
            self._record()
        if self._every <= 0:
            return
        self._frame += 1
        if self._frame % self._every:
            return
        parts = []
        for side, attr in (("R", "_previous_joint_poses_right"), ("L", "_previous_joint_poses_left")):
            poses = getattr(self._device, attr, None)
            if not poses:
                parts.append(f"{side}: n/a")
                continue
            wrist = np.asarray(poses.get("wrist", np.zeros(7)), dtype=np.float32)
            nonzero = sum(1 for p in poses.values() if np.any(np.asarray(p)[:3] != 0.0))
            previous = self._last_wrist.get(side)
            stale = previous is not None and nonzero > 0 and np.array_equal(previous, wrist)
            self._last_wrist[side] = wrist.copy()
            parts.append(
                f"{side}: wrist=({wrist[0]:+.3f},{wrist[1]:+.3f},{wrist[2]:+.3f})"
                f" nonzero={nonzero}/{len(poses)}{' STALE' if stale else ''}"
            )
        print(f"[xr] f={self._frame:>6}  " + "  ".join(parts))
