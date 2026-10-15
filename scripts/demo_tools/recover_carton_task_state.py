# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Back-fill the per-episode target face into old carton-regrasp recordings.

``Dexverse-BimanualCartonRegrasp-v0`` pickles recorded before schema 4 (Aug 23
2026 sessions) carry the goal frame in every scene state (``rigid_object/goal_frame``,
so its position and yaw replay correctly) but **not** the per-episode env buffer
``carton_grasp_face`` -- the sampled target face that selects the yellow face
patches, the goal-cue corner set and the success gates. Replaying such a pickle
re-samples the face, so the cue, the markers in rendered observations and the
success check no longer match what the operator saw.

The face is recoverable without the buffer: the success condition in force at
recording time (``forbidden_faces_fit_in_cue_success`` / ``object_fit_in_cue_success``)
requires the target face's outward normal to point within ``upright_tol_deg``
(35 deg) of straight down for ``hold_steps`` consecutive steps, while every other
face of a cuboid is at least 55 deg away. So for a successful episode the face
whose normal points down during the final hold window *is* the sampled face.
The tool also cross-checks the recovered face against the rest of the success
gates (carton centre inside the cue's slack above the recorded goal frame, yaw
alignment of the face's horizontal axis with the goal frame) so a corrupted or
non-success episode is flagged instead of silently labelled.

Recovered faces are written as ``episode["task_state"] = {"env_buffers":
{"carton_grasp_face": int64}}`` -- the exact layout ``record_demos.py`` (schema 4)
produces -- so ``create_demo_files_sequential.py`` restores them through
``restore_episode_task_state`` with no further changes. The pickle's
``schema_version`` is bumped to 4 and a ``task_state_recovery`` provenance block
is added at the top level.

Usage::

    # dry run: print the recovered face + checks for every episode, write nothing
    python scripts/demo_tools/recover_carton_task_state.py <pickle-or-dir> [...]

    # write next to the inputs as <stem>.recovered.pkl
    python scripts/demo_tools/recover_carton_task_state.py <pickle-or-dir> --write

    # upgrade in place (original kept as <name>.pkl.schema3.bak)
    python scripts/demo_tools/recover_carton_task_state.py <pickle-or-dir> --inplace
"""

from __future__ import annotations

import argparse
import math
import pickle
import shutil
from pathlib import Path

import numpy as np

# --- carton geometry / face conventions -----------------------------------------
# Mirrors ``dexverse.tasks.config.bimanual.carton_regrasp_cfg`` (importing the cfg
# needs a running Isaac app, and this tool is meant to run on a plain python).
CARTON_SCALE = 0.65
_UNIT_HALF_X, _UNIT_HALF_Y, _UNIT_HEIGHT = 0.150 / 0.75, 0.176 / 0.75, 0.214 / 0.75
_UNIT_CENTER_XY = (0.0015 / 0.75, 0.0005 / 0.75)
CARTON_HALF = np.array([_UNIT_HALF_X, _UNIT_HALF_Y, _UNIT_HEIGHT / 2.0]) * CARTON_SCALE
CARTON_CENTER_LOCAL = np.array([_UNIT_CENTER_XY[0] * CARTON_SCALE, _UNIT_CENTER_XY[1] * CARTON_SCALE, CARTON_HALF[2]])
# faces ordered [+x, -x, +y, -y, +z, -z] -- the ``carton_grasp_face`` buffer value
FACE_NAMES = ["+x", "-x", "+y", "-y", "+z", "-z"]
FACE_AXIS = [0, 0, 1, 1, 2, 2]
DOWN_NORMAL_PER_FACE = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
# object-frame axis that maps to the goal frame's +x in the required pose
HORIZONTAL_AXIS_PER_FACE = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64)

BUFFER_NAME = "carton_grasp_face"
TASK_NAME = "Dexverse-BimanualLiftCarton-v1"
LEGACY_TASK_NAME = "Dexverse-BimanualCartonRegrasp-v0"


def _cue_half_extents(face: int, cue_scale: float) -> np.ndarray:
    """Goal-frame half extents of the carton-shaped cue with ``face`` at the bottom."""
    r = cue_scale / CARTON_SCALE
    hx, hy, hz = CARTON_HALF * r
    axis = FACE_AXIS[face]
    if axis == 0:
        return np.array([hz, hy, hx])
    if axis == 1:
        return np.array([hx, hz, hy])
    return np.array([hx, hy, hz])


def _rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _pose(state: dict, entity: str) -> np.ndarray:
    return np.asarray(state["rigid_object"][entity]["root_pose"], dtype=np.float64).reshape(-1)[:7]


def recover_face(
    episode: dict,
    *,
    hold_steps: int,
    upright_tol_deg: float,
    yaw_tol_deg: float,
    cue_scale: float,
    goal_center_height: float,
) -> dict:
    """Recover the target face of one episode from its final ``hold_steps`` states.

    Returns a dict with ``face`` (int), the per-face mean down-cosine over the
    window, and the success-gate cross-checks evaluated at the last state.
    """
    states = episode.get("states")
    if not states:
        raise ValueError("episode has no per-step 'states' (recorded without --record_state)")
    window = states[-hold_steps:]
    down_cos = np.zeros((len(window), 6))
    for i, st in enumerate(window):
        R = _rot(_pose(st, "object")[3:])
        down_cos[i] = -(R @ DOWN_NORMAL_PER_FACE.T)[2]  # cos(angle to straight down) per face
    mean_cos = down_cos.mean(axis=0)
    order = np.argsort(mean_cos)[::-1]
    face, runner_up = int(order[0]), int(order[1])

    # Cross-checks against the other success gates at the final state.
    last = states[-1]
    obj, goal = _pose(last, "object"), _pose(last, "goal_frame")
    R_o, R_g = _rot(obj[3:]), _rot(goal[3:])
    centre_w = obj[:3] + R_o @ CARTON_CENTER_LOCAL
    centre_g = R_g.T @ (centre_w - goal[:3]) - np.array([0.0, 0.0, goal_center_height])
    slack = _cue_half_extents(face, cue_scale) - _cue_half_extents(face, CARTON_SCALE)
    yaw_cos = abs(float((R_o @ HORIZONTAL_AXIS_PER_FACE[face]) @ (R_g @ np.array([1.0, 0.0, 0.0]))))
    upright_cos = math.cos(math.radians(upright_tol_deg))
    return {
        "face": face,
        "face_name": FACE_NAMES[face],
        "mean_down_cos": mean_cos,
        "min_down_cos": float(down_cos[:, face].min()),
        "runner_up": runner_up,
        "runner_up_cos": float(mean_cos[runner_up]),
        "upright_ok": bool(down_cos[:, face].min() >= upright_cos),
        # a cuboid's other faces are >= 90 deg away from the down face; anything
        # closer than the upright tolerance itself means the pose is ambiguous
        "unambiguous": bool(mean_cos[runner_up] < upright_cos),
        "centre_in_cue": bool(np.all(np.abs(centre_g) <= slack + 1e-6)),
        "centre_offset": centre_g,
        "cue_slack": slack,
        "yaw_ok": bool(yaw_cos >= math.cos(math.radians(yaw_tol_deg))),
        "yaw_cos": yaw_cos,
    }


def _iter_pickles(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*.pkl") if ".recovered" not in q.name))
        else:
            out.append(p)
    return out


def process_pickle(path: Path, args) -> tuple[int, int]:
    with open(path, "rb") as fp:
        payload = pickle.load(fp)
    task = payload.get("env_name") or payload.get("task")
    if task not in (TASK_NAME, LEGACY_TASK_NAME) and not args.force_task:
        print(f"[skip] {path}: task {task!r} is not {TASK_NAME} (pass --force-task to override)")
        return 0, 0
    episodes = payload.get("episodes", [])
    print(f"\n=== {path} (schema {payload.get('schema_version')}, seed {payload.get('seed')}, {len(episodes)} episodes)")
    n_ok = n_bad = 0
    face_hist = np.zeros(6, dtype=int)
    for ep in episodes:
        idx = ep.get("episode_index", "?")
        existing = (ep.get("task_state") or {}).get("env_buffers", {}).get(BUFFER_NAME)
        if existing is not None and not args.overwrite_existing:
            print(f"  ep{idx:>3}: already has {BUFFER_NAME}={int(np.asarray(existing).reshape(-1)[0])} -- kept (use --overwrite-existing to recompute)")
            n_ok += 1
            continue
        try:
            r = recover_face(
                ep,
                hold_steps=args.hold_steps,
                upright_tol_deg=args.upright_tol_deg,
                yaw_tol_deg=args.yaw_tol_deg,
                cue_scale=args.goal_cue_scale,
                goal_center_height=args.goal_center_height,
            )
        except ValueError as exc:
            print(f"  ep{idx:>3}: FAILED -- {exc}")
            n_bad += 1
            continue
        success = ep.get("success")
        checks = [r["upright_ok"], r["unambiguous"], r["centre_in_cue"], r["yaw_ok"], success is True]
        ok = all(checks)
        flag = "ok " if ok else "?? "
        print(
            f"  ep{idx:>3}: {flag}face={r['face']} ({r['face_name']})  down_cos min={r['min_down_cos']:.3f}"
            f" runner-up={FACE_NAMES[r['runner_up']]}:{r['runner_up_cos']:.2f}"
            f"  yaw_cos={r['yaw_cos']:.3f}  centre_off=({r['centre_offset'][0]:+.3f},{r['centre_offset'][1]:+.3f},{r['centre_offset'][2]:+.3f})"
            f" slack=({r['cue_slack'][0]:.3f},{r['cue_slack'][1]:.3f},{r['cue_slack'][2]:.3f})  success={success}"
        )
        if not ok:
            reasons = [
                name for name, c in zip(["upright", "unambiguous", "centre_in_cue", "yaw", "success_flag"], checks) if not c
            ]
            print(f"          checks failed: {', '.join(reasons)}")
            if not args.allow_failed_checks:
                n_bad += 1
                continue
        task_state = dict(ep.get("task_state") or {})
        env_buffers = dict(task_state.get("env_buffers") or {})
        env_buffers[BUFFER_NAME] = np.asarray(r["face"], dtype=np.int64)
        task_state["env_buffers"] = env_buffers
        ep["task_state"] = task_state
        face_hist[r["face"]] += 1
        n_ok += 1

    print(f"  recovered {n_ok}/{len(episodes)} episodes; face histogram {dict(zip(FACE_NAMES, face_hist.tolist()))}; flagged {n_bad}")
    if n_bad and not args.allow_failed_checks:
        print(f"  [warn] {n_bad} episode(s) failed the consistency checks and were left without a face")

    if args.write or args.inplace:
        payload["schema_version"] = max(int(payload.get("schema_version", 0)), 4)
        payload["task_state_recovery"] = {
            "tool": "scripts/demo_tools/recover_carton_task_state.py",
            "buffer": BUFFER_NAME,
            "method": "face whose outward normal points down over the final hold window of a successful episode",
            "hold_steps": args.hold_steps,
            "upright_tol_deg": args.upright_tol_deg,
            "yaw_tol_deg": args.yaw_tol_deg,
            "goal_cue_scale": args.goal_cue_scale,
            "goal_center_height": args.goal_center_height,
            "original_schema_version": int(payload.get("schema_version_original", payload.get("schema_version", 0))),
            "episodes_recovered": n_ok,
            "episodes_flagged": n_bad,
        }
        if args.inplace:
            backup = path.with_name(path.name + ".schema3.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
            out_path = path
        else:
            out_path = path.with_name(path.stem + ".recovered.pkl")
        with open(out_path, "wb") as fp:
            pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  wrote {out_path}" + (f" (backup: {backup})" if args.inplace else ""))
    return n_ok, n_bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="Pickle files or directories (searched recursively for *.pkl).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Write <stem>.recovered.pkl next to each input.")
    mode.add_argument("--inplace", action="store_true", help="Overwrite each input (keeps <name>.pkl.schema3.bak).")
    parser.add_argument("--overwrite-existing", action="store_true", help="Recompute faces even when task_state already has one.")
    parser.add_argument("--allow-failed-checks", action="store_true", help="Still write a face for episodes that fail the consistency checks.")
    parser.add_argument("--force-task", action="store_true", help="Process pickles whose task name is not the carton task.")
    # Success-gate parameters in force when the demos were recorded (carton_regrasp_cfg.py @ 8df6b57).
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--upright-tol-deg", type=float, default=35.0)
    parser.add_argument("--yaw-tol-deg", type=float, default=25.0)
    parser.add_argument("--goal-cue-scale", type=float, default=0.85)
    parser.add_argument("--goal-center-height", type=float, default=0.22)
    args = parser.parse_args()

    total_ok = total_bad = 0
    for path in _iter_pickles(args.paths):
        ok, bad = process_pickle(path, args)
        total_ok += ok
        total_bad += bad
    print(f"\nDone: {total_ok} episodes recovered, {total_bad} flagged." + ("" if (args.write or args.inplace) else " (dry run -- pass --write or --inplace to save)"))


if __name__ == "__main__":
    main()
