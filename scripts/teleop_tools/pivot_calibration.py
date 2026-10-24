# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pivot calibration: controller -> wrist offset for a hand-mounted VR controller.

The operator keeps the wrist rotation centre fixed (e.g. resting on a table edge)
and rotates the hand in many directions while the controller pose is recorded.
Every sample then satisfies::

    t_i + R_i @ r = p          (r: wrist point in the controller frame, p: fixed pivot in world)

which is linear in ``(r, p)``. Stacking ``[R_i, -I] @ [r; p] = -t_i`` over all samples
and solving least squares gives the offset ``r`` used at teleop time as
``p_wrist = t_ctrl + R_ctrl @ r``.

Pure numpy, no Isaac Sim needed. Input is the ``.npz`` written by
``record_controller_poses.py`` (``right`` / ``left``: ``(N, 7)`` poses as
``[x, y, z, qw, qx, qy, qz]``)::

    python scripts/teleop_tools/pivot_calibration.py poses.npz --out controller_wrist_offset.json
    python scripts/teleop_tools/pivot_calibration.py --selftest
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

# Quality thresholds for the printed verdict.
MIN_SAMPLES = 100
MIN_ROTATION_SPAN_DEG = 30.0  # smallest principal spread of the rotation axes
GOOD_RMS_MM = 5.0
OUTLIER_FLOOR_M = 0.005  # never reject samples closer than this to the fit


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz quaternions -> (N, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )


def _solve(t: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(t)
    A = np.concatenate([R, -np.broadcast_to(np.eye(3), (n, 3, 3))], axis=2).reshape(3 * n, 6)
    x, *_ = np.linalg.lstsq(A, -t.reshape(3 * n), rcond=None)
    r, p = x[:3], x[3:]
    residual = np.linalg.norm(t + R @ r - p, axis=1)
    return r, p, residual


def rotation_span_deg(R: np.ndarray) -> np.ndarray:
    """Principal spreads (deg) of the samples' rotations relative to their mean.

    The smallest value tells whether the hand was rotated about enough different
    axes; rotating about a single axis leaves the offset along that axis unobservable.
    """
    R_ref = R[len(R) // 2]
    rel = np.einsum("ij,njk->nik", R_ref.T, R)
    angle = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))
    axis = np.stack([rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]], axis=1)
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    rotvec = np.where(norm > 1e-9, axis / np.maximum(norm, 1e-9), 0.0) * angle[:, None]
    rotvec -= rotvec.mean(axis=0)
    sv = np.linalg.svd(rotvec, compute_uv=False) / np.sqrt(len(rotvec))
    return np.degrees(2 * sv)  # ~ +-1 sigma span


def calibrate(poses: np.ndarray, outlier_iters: int = 2) -> dict:
    """Fit the pivot for one hand. ``poses``: (N, 7) ``[x, y, z, qw, qx, qy, qz]``."""
    poses = np.asarray(poses, dtype=np.float64)
    keep = np.isfinite(poses).all(axis=1) & (np.linalg.norm(poses[:, 3:], axis=1) > 0.5)
    # Drop frames where tracking froze (bit-identical consecutive poses).
    keep[1:] &= np.any(poses[1:] != poses[:-1], axis=1)
    t = poses[keep, :3]
    R = quat_wxyz_to_rotmat(poses[keep, 3:])
    if len(t) < 10:
        raise ValueError(f"only {len(t)} valid samples")

    used = np.ones(len(t), dtype=bool)
    for _ in range(outlier_iters + 1):
        r, p, _ = _solve(t[used], R[used])
        residual = np.linalg.norm(t + R @ r - p, axis=1)
        med = np.median(residual[used])
        mad = np.median(np.abs(residual[used] - med))
        new_used = residual < max(med + 4.0 * 1.4826 * mad, OUTLIER_FLOOR_M)
        if np.array_equal(new_used, used):
            break
        used = new_used

    span = rotation_span_deg(R[used])
    rms_mm = float(np.sqrt(np.mean(residual[used] ** 2)) * 1000)
    warnings = []
    if used.sum() < MIN_SAMPLES:
        warnings.append(f"few samples ({used.sum()} < {MIN_SAMPLES}): record longer")
    if span.min() < MIN_ROTATION_SPAN_DEG:
        warnings.append(
            f"rotation too narrow (smallest span {span.min():.0f} deg < {MIN_ROTATION_SPAN_DEG:.0f}):"
            " also tilt / twist the hand about other axes"
        )
    if rms_mm > GOOD_RMS_MM:
        warnings.append(f"RMS {rms_mm:.1f} mm > {GOOD_RMS_MM} mm: the wrist probably moved, keep it fixed")
    return {
        "offset_ctrl_m": [float(v) for v in r],
        "offset_norm_m": float(np.linalg.norm(r)),
        "pivot_world_m": [float(v) for v in p],
        "rms_mm": rms_mm,
        "max_err_mm": float(residual[used].max() * 1000),
        "n_used": int(used.sum()),
        "n_valid": int(len(t)),
        "rotation_span_deg": [float(v) for v in span],
        "warnings": warnings,
    }


def _selftest() -> None:
    rng = np.random.default_rng(0)
    r_true = np.array([0.02, -0.04, 0.09])
    p_true = np.array([0.3, 0.1, 1.0])
    rv = rng.normal(scale=0.5, size=(600, 3))
    ang = np.linalg.norm(rv, axis=1, keepdims=True)
    q = np.concatenate([np.cos(ang / 2), np.sin(ang / 2) * rv / ang], axis=1)
    R = quat_wxyz_to_rotmat(q)
    t = p_true - R @ r_true + rng.normal(scale=0.002, size=(600, 3))  # 2 mm tracking noise
    t[:20] += 0.1  # glitches
    res = calibrate(np.concatenate([t, q], axis=1))
    err = np.linalg.norm(np.array(res["offset_ctrl_m"]) - r_true) * 1000
    print(json.dumps(res, indent=2))
    print(f"[selftest] offset error {err:.2f} mm -> {'OK' if err < 2.0 else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", nargs="?", help="poses recorded by record_controller_poses.py")
    parser.add_argument("--out", default=None, help="output JSON (default: controller_wrist_offset.json next to npz)")
    parser.add_argument("--selftest", action="store_true", help="fit synthetic data and check the error")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.npz:
        parser.error("npz path required (or --selftest)")

    data = np.load(args.npz)
    result = {
        "description": "wrist point in the controller frame: p_wrist = t_ctrl + R_ctrl @ offset_ctrl_m",
        "pose_source": "Isaac Lab OpenXRDevice motion controller pose [x, y, z, qw, qx, qy, qz]",
        "source_npz": str(Path(args.npz).resolve()),
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    for side in ("right", "left"):
        if side not in data.files or len(data[side]) == 0:
            continue
        try:
            res = calibrate(data[side])
        except ValueError as exc:
            print(f"[{side}] skipped: {exc}")
            continue
        result[side] = res
        r = res["offset_ctrl_m"]
        print(
            f"[{side}] offset = ({r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f}) m  |r| = {res['offset_norm_m'] * 100:.1f} cm"
            f"  RMS = {res['rms_mm']:.1f} mm  samples = {res['n_used']}/{res['n_valid']}"
            f"  span = {', '.join(f'{s:.0f}' for s in res['rotation_span_deg'])} deg"
        )
        for w in res["warnings"]:
            print(f"[{side}]   WARNING: {w}")
        if not res["warnings"]:
            print(f"[{side}]   OK")

    if "right" not in result and "left" not in result:
        raise SystemExit("no hand calibrated")
    out = Path(args.out) if args.out else Path(args.npz).with_name("controller_wrist_offset.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
