# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record Quest controller poses over CloudXR for pivot calibration.

Run against the CloudXR runtime like the teleop scripts::

    scripts/teleop_tools/start_cloudxr_runtime.sh                  # terminal 1
    scripts/teleop_tools/run_teleop.sh scripts/teleop_tools/record_controller_poses.py \\
        --out /workspace/datasets/dexverse/calibration/pivot_$(date +%Y%m%d-%H%M%S).npz

Procedure (per hand, controller mounted on the glove):
  1. Rest the wrist rotation centre on a fixed point (table edge) and keep it there.
  2. Hold that controller's TRIGGER (press it with the other hand, or a helper does)
     and rotate the hand slowly in every direction (flex/extend, side to side,
     twist) for 20-30 s. Release to pause; samples are recorded only while held.
  3. Repeat for the other hand. Press A or X to finish: the poses are saved and
     ``pivot_calibration.py`` is run on them.

Poses are the Isaac Lab ``OpenXRDevice`` motion-controller poses in sim world
(``[x, y, z, qw, qx, qy, qz]``), i.e. the same values the teleop retargeters see.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record Quest controller poses for pivot calibration.")
parser.add_argument("--out", type=str, required=True, help="output .npz path")
parser.add_argument("--trigger_threshold", type=float, default=0.5, help="trigger value that counts as held")
parser.add_argument("--log_every", type=int, default=90, help="print a stream summary every N frames (0: off)")
# Accepted for run_teleop.sh compatibility; unused here.
parser.add_argument("--teleop_device", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--enable_pinocchio", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.xr = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
from dexverse.teleop_utils.xr_session import request_ar_session  # noqa: E402
from isaaclab.devices import DeviceBase  # noqa: E402
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg, XrCfg  # noqa: E402
from isaaclab.devices.retargeter_base import RetargeterBase  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pivot_calibration import calibrate  # noqa: E402

POSE = DeviceBase.MotionControllerDataRowIndex.POSE.value
INPUTS = DeviceBase.MotionControllerDataRowIndex.INPUTS.value
TRIGGER = DeviceBase.MotionControllerInputIndex.TRIGGER.value
BUTTON_0 = DeviceBase.MotionControllerInputIndex.BUTTON_0.value  # A (right) / X (left)

SIDES = {"right": DeviceBase.TrackingTarget.CONTROLLER_RIGHT, "left": DeviceBase.TrackingTarget.CONTROLLER_LEFT}


def save(out: Path, samples: dict[str, list]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{side: np.asarray(s, dtype=np.float64).reshape(-1, 7) for side, s in samples.items()})


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1 / 90))
    device = OpenXRDevice(OpenXRDeviceCfg(xr_cfg=XrCfg()))
    # No retargeter is attached, so request the controller stream explicitly.
    device._required_features = {RetargeterBase.Requirement.MOTION_CONTROLLER}
    sim.reset()
    request_ar_session(simulation_app)

    out = Path(args_cli.out)
    samples: dict[str, list] = {side: [] for side in SIDES}
    held = {side: False for side in SIDES}
    frame = 0
    print("[pivot] Hold a controller's trigger and rotate the hand about the fixed wrist. A/X = finish.")
    while simulation_app.is_running():
        simulation_app.update()
        frame += 1
        data = device._get_raw_data()
        finish = False
        for side, target in SIDES.items():
            ctrl = data.get(target)
            if ctrl is None or len(ctrl) == 0:
                continue
            pressed = ctrl[INPUTS][TRIGGER] > args_cli.trigger_threshold
            if pressed != held[side]:
                held[side] = pressed
                print(f"[pivot] {side}: {'recording' if pressed else 'paused'} ({len(samples[side])} samples)")
            if pressed:
                samples[side].append(np.asarray(ctrl[POSE], dtype=np.float64).copy())
            finish |= ctrl[INPUTS][BUTTON_0] > 0.5
        if args_cli.log_every and frame % args_cli.log_every == 0:
            parts = []
            for side, target in SIDES.items():
                ctrl = data.get(target)
                if ctrl is None or len(ctrl) == 0:
                    parts.append(f"{side}=none")
                else:
                    x, y, z = ctrl[POSE][:3]
                    parts.append(f"{side}=({x:+.3f},{y:+.3f},{z:+.3f}) n={len(samples[side])}")
            print(f"[pivot] f={frame} " + "  ".join(parts))
        if frame % 900 == 0:
            save(out, samples)  # periodic save so a crash loses little
        if finish:
            break

    save(out, samples)
    print(f"[pivot] saved {out}: " + ", ".join(f"{s}={len(v)}" for s, v in samples.items()))
    for side, s in samples.items():
        if len(s) < 10:
            continue
        res = calibrate(np.asarray(s))
        r = res["offset_ctrl_m"]
        print(
            f"[pivot] {side}: offset=({r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f}) m  RMS={res['rms_mm']:.1f} mm"
            + "".join(f"\n[pivot]   WARNING: {w}" for w in res["warnings"])
        )
    print(f"[pivot] save the result with: python {Path(__file__).with_name('pivot_calibration.py')} {out}")


if __name__ == "__main__":
    main()
    # SimulationApp.close() can hang in an XR session and the half-closed app keeps the headset
    # session, so the next run gets no controller data: exit the process directly.
    sys.stdout.flush()
    os._exit(0)
