# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-distribution audit: statistics, flags, plots and report.

The Isaac-side sampler (``scripts/env_tools/sample_reset_distributions.py``)
resets a task many times and dumps, per task directory:

* ``samples.npz`` -- stacked per-sample arrays (see :func:`load_task_dir` for the
  key layout), all positions **env-relative** (metres) and quaternions ``(w,x,y,z)``.
* ``meta.json``   -- task id, entity lists, declared randomization boxes,
  termination-term parameters, reset-event order, defaults, ...

Everything in this module is simulator-free (numpy + matplotlib) so the numbers
can be re-analysed and re-plotted anywhere:

* :func:`resolve_roles`  -- which entity is *the object*, which is *the goal*.
* :func:`compute_stats`  -- spread / coverage / init->goal distance / settle-drift
  / termination statistics.
* :func:`flag_task`      -- turns stats into a list of named problems.
* :func:`plot_task`      -- one PNG per task (XY scatter + histograms + text).
* :func:`write_report`   -- aggregates every task directory into ``report.md``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from dexverse.teleop_utils.diversity_core import _circular_std_deg, quat_yaw
from dexverse.benchmark import BASELINE_TASKS, BASELINE_PAIRS, LEGACY_UPGRADE_TASKS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

#: Per-task role hints. ``goal`` is either a scene entity name or
#: ``"command:<term>"``. Anything not listed here goes through the generic
#: fallback in :func:`resolve_roles`. ``extra`` lists props worth plotting even
#: when they do not move (context for the operator's eye).
TASK_ROLES: dict[str, dict] = {
    "Dexverse-GraspBleach-v0": {"object": "object", "goal": "success_marker"},
    "Dexverse-GraspPan-v0": {"object": "object", "goal": "success_marker", "extra": ["stove"]},
    "Dexverse-GraspKettle-v0": {"object": "object", "goal": "success_marker", "extra": ["mug"]},
    "Dexverse-GraspCup-v0": {"object": "object", "goal": "success_marker"},
    "Dexverse-RemoveCupFromRack-v0": {"object": "object", "goal": "command:object_pose", "extra": ["cup_holder"]},
    "Dexverse-FunctionalPourCan-v0": {"object": "object", "goal": "success_marker", "extra": ["bowl"]},
    "Dexverse-FunctionalPourMug-v0": {"object": "object", "goal": "success_marker"},
    "Dexverse-FunctionalHammerStrike-v0": {"object": "object", "goal": "hammer_target", "extra": ["hammer_stand"]},
    "Dexverse-OpenFaucet-v0": {"object": "articulation", "goal": None, "extra": ["sink", "faucet_support"]},
    "Dexverse-OpenLaptop-v0": {"object": "articulation", "goal": None},
    "Dexverse-SqueezeScissors-v0": {"object": "articulation", "goal": None},
    "Dexverse-SlideUtilityKnife-v0": {"object": "articulation", "goal": None},
    "Dexverse-OpenStapler-v0": {"object": "articulation", "goal": None},
    "Dexverse-OpenFlatFolder-v0": {"object": "articulation", "goal": None},
    "Dexverse-BimanualLiftTray-v0": {"object": "object", "goal": None, "extra": ["food"]},
    "Dexverse-BimanualLiftCarton-v0": {"object": "object", "goal": None},
    "Dexverse-InsertPen-v0": {"object": "pen", "goal": "pen_holder"},
    "Dexverse-PushSmallSphereObstacleSlope-v0": {"object": "object", "goal": "command:object_pose"},
    "Dexverse-PushT-v0": {"object": "object", "goal": "goal_tee"},
    "Dexverse-BimanualRetrieveTrayFromRack-v0": {"object": "object", "goal": "goal_pad", "extra": ["rack", "food"]},
    "Dexverse-CloseFolderInsertShelf-v0": {"object": "articulation", "goal": "shelf", "extra": []},
    "Dexverse-BimanualCartonRegrasp-v0": {"object": "object", "goal": "goal_frame"},
    "Dexverse-CutSeamUtilityKnife-v0": {"object": "articulation", "goal": "workpiece"},
    "Dexverse-CutStripScissors-v0": {"object": "articulation", "goal": "strip_stand"},
    "Dexverse-ReloadStapler-v0": {"object": "articulation", "goal": "stapler_goal"},
    "Dexverse-PlaceCupUnderDispenser-v0": {"object": "object", "goal": "dispenser"},
    "Dexverse-OpenTiltedDoor-v0": {"object": "articulation", "goal": None},
}

# Preserve legacy-role hints for offline private data, but canonical v1 roles
# follow the upgraded implementation rather than the original task's name.
for _v0, _v1 in BASELINE_PAIRS.items():
    TASK_ROLES[_v1] = dict(TASK_ROLES[_v0])
for _legacy, _v1 in LEGACY_UPGRADE_TASKS.items():
    TASK_ROLES[_v1] = dict(TASK_ROLES[_legacy])
# These props were added only by v1.
TASK_ROLES["Dexverse-FunctionalHammerStrike-v0"] = {"object": "object", "goal": "hammer_target"}
TASK_ROLES["Dexverse-OpenFaucet-v0"] = {"object": "articulation", "goal": None}

_GOAL_NAME_HINTS = ("goal", "success_marker", "target", "holder", "bowl", "mug")
_STATIC_NAME_HINTS = ("table", "plane", "leg", "wall", "light", "camera")

#: Termination-term parameter names that denote a *distance* threshold, in
#: order of preference. ``kind`` says which distance the threshold applies to.
_DISTANCE_PARAM_KEYS: tuple[tuple[str, str], ...] = (
    ("position_threshold", "3d"),
    ("goal_xy_threshold", "xy"),
    ("xy_threshold", "xy"),
    ("center_dist_thresh", "xy"),
    ("threshold", "3d"),
)
#: ``threshold`` is only a distance for these success functions (for
#: articulation ``joint_*`` terms it is radians/metres of joint travel).
_DISTANCE_THRESHOLD_FUNCS = (
    "object_at_goal_position",
    "object_at_goal_pose",
    "success_no_forbidden_contact",
    "success_with_contact_zones",
    "insert_peg_success",
    "plug_charger_pose_success",
)

#: Default flag thresholds (metres / degrees / fractions). Override via
#: ``flag_task(..., thresholds={...})``.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "degenerate_xy_std_m": 0.01,
    "degenerate_yaw_std_deg": 2.0,
    "one_d_axis_extent_m": 0.01,
    "one_d_other_axis_min_m": 0.05,
    "narrow_extent_m": 0.10,
    "overlap_dist_multiplier": 2.0,
    "overlap_frac": 0.05,
    "trivial_frac": 0.01,
    "broken_frac": 0.02,
    "drift_pos_m": 0.03,
    "drift_yaw_deg": 10.0,
    "palm_drift_m": 0.02,
    "table_half_m": 0.75,
    "step1_spurious_frac": 0.5,
}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_task_dir(task_dir: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load ``samples.npz`` + ``meta.json`` from one task directory.

    ``samples.npz`` keys (``N`` = number of recorded samples):

    * ``rigid_object/<name>/root_pose`` (N,7), ``articulation/<name>/root_pose`` (N,7),
      ``articulation/<name>/joint_pos`` (N,J)
    * ``commands/<name>/pose_w`` (N,7) env-relative goal pose, ``commands/<name>/command`` (N,C)
    * ``palm/<label>/pos`` (N,3)
    * ``settled/...`` -- same keys, state after the settle steps (NaN where the env
      terminated and was auto-reset before the snapshot)
    * ``term_first_step/<term>`` (N,) int, first settle step at which the term fired, -1 = never
    * ``done_step`` (N,) int, first settle step at which the env was done, -1 = never
    * ``probe/success_step1`` (P,) bool -- raw env behaviour on the first step after reset
    * ``env_origin`` (N,3), ``env_index`` (N,)
    """
    task_dir = Path(task_dir)
    with np.load(task_dir / "samples.npz") as data:
        samples = {k: data[k] for k in data.files}
    meta = json.loads((task_dir / "meta.json").read_text())
    return samples, meta


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


def _entity_kinds(meta: dict) -> dict[str, str]:
    """``{entity_name: "rigid_object" | "articulation"}`` from meta."""
    out = {}
    for kind in ("rigid_object", "articulation"):
        for name in meta.get("entities", {}).get(kind, []):
            out[name] = kind
    return out


def _pose_key(samples: dict, meta: dict, entity: str, settled: bool = False) -> str | None:
    kind = _entity_kinds(meta).get(entity)
    if kind is None:
        return None
    key = f"{kind}/{entity}/root_pose"
    if settled:
        key = "settled/" + key
    return key if key in samples else None


def resolve_roles(samples: dict, meta: dict) -> dict:
    """Decide object / goal / extra entities for a task.

    Uses :data:`TASK_ROLES` when the task is listed, otherwise a generic guess:
    the ``object`` scene key (or the single articulation), the first command
    term as goal, else the first entity whose name looks like a goal.
    """
    task = meta.get("task", "")
    kinds = _entity_kinds(meta)
    commands = list(meta.get("commands", {}).keys())
    hint = dict(TASK_ROLES.get(task, {}))

    obj = hint.get("object")
    if obj not in kinds:
        if "object" in kinds:
            obj = "object"
        elif "articulation" in kinds:
            obj = "articulation"
        else:
            arts = [n for n, k in kinds.items() if k == "articulation"]
            obj = arts[0] if arts else next(iter(kinds), None)

    goal = hint.get("goal", "auto")
    if goal == "auto":
        goal = None
        if commands:
            goal = f"command:{commands[0]}"
        else:
            for name in kinds:
                if name != obj and any(h in name.lower() for h in _GOAL_NAME_HINTS):
                    goal = name
                    break
    elif goal is not None and not goal.startswith("command:") and goal not in kinds:
        logger.warning("reset_audit: goal entity '%s' not in scene for %s; ignoring", goal, task)
        goal = None
    elif goal is not None and goal.startswith("command:") and goal.split(":", 1)[1] not in commands:
        logger.warning("reset_audit: goal command '%s' not active for %s; ignoring", goal, task)
        goal = None

    extra = [e for e in hint.get("extra", []) if e in kinds]
    # Also plot every entity that actually moves between resets (spread > 2 mm).
    for name, kind in kinds.items():
        if name in (obj, goal) or name in extra:
            continue
        if any(h in name.lower() for h in _STATIC_NAME_HINTS):
            continue
        key = f"{kind}/{name}/root_pose"
        if key in samples and np.nanmax(np.nanstd(samples[key][:, :2], axis=0)) > 0.002:
            extra.append(name)

    dist = _distance_threshold(meta)
    return {"object": obj, "goal": goal, "extra": extra, **dist}


def _distance_threshold(meta: dict) -> dict:
    """Pull the success distance threshold (if any) out of the termination params."""
    terms = meta.get("termination_terms", {})
    succ = terms.get("success")
    if not succ:
        return {"distance_threshold_m": None, "distance_kind": None, "success_func": None}
    func = succ.get("func", "")
    params = succ.get("params", {})
    for key, kind in _DISTANCE_PARAM_KEYS:
        if key not in params:
            continue
        if key == "threshold" and func not in _DISTANCE_THRESHOLD_FUNCS:
            continue
        val = params[key]
        if isinstance(val, (int, float)):
            return {"distance_threshold_m": float(val), "distance_kind": kind, "success_func": func}
    return {"distance_threshold_m": None, "distance_kind": None, "success_func": func}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _finite_rows(arr: np.ndarray) -> np.ndarray:
    return arr[np.all(np.isfinite(arr.reshape(arr.shape[0], -1)), axis=1)]


def _yaw_deg(quat: np.ndarray) -> np.ndarray:
    return np.degrees(quat_yaw(quat))


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def _pose_stats(pose: np.ndarray, xy_bin_m: float = 0.02, yaw_bin_deg: float = 30.0) -> dict:
    """Spread / coverage of an (N,7) env-relative pose array."""
    pose = _finite_rows(np.asarray(pose, dtype=np.float64))
    if pose.shape[0] == 0:
        return {"n": 0}
    xyz = pose[:, :3]
    yaw = _yaw_deg(pose[:, 3:7])
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    xy_bins = {tuple(np.floor(p / xy_bin_m).astype(int)) for p in xyz[:, :2]}
    yaw_bins = {int(math.floor(((y % 360.0)) / yaw_bin_deg)) for y in yaw}
    return {
        "n": int(pose.shape[0]),
        "mean": [float(v) for v in xyz.mean(axis=0)],
        "std": [float(v) for v in xyz.std(axis=0)],
        "min": [float(v) for v in lo],
        "max": [float(v) for v in hi],
        "extent": [float(v) for v in (hi - lo)],
        "yaw_mean_deg": float(np.degrees(np.angle(np.exp(1j * np.radians(yaw)).mean()))),
        "yaw_std_deg": _circular_std_deg(yaw),
        "yaw_min_deg": float(yaw.min()),
        "yaw_max_deg": float(yaw.max()),
        "xy_bins_occupied": len(xy_bins),
        "yaw_bins_occupied": len(yaw_bins),
    }


def _drift_stats(before: np.ndarray, after: np.ndarray) -> dict:
    """Position / yaw change between the reset snapshot and the settled snapshot."""
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    ok = np.all(np.isfinite(after.reshape(after.shape[0], -1)), axis=1) & np.all(
        np.isfinite(before.reshape(before.shape[0], -1)), axis=1
    )
    if ok.sum() == 0:
        return {"n": 0}
    dpos = np.linalg.norm(after[ok, :3] - before[ok, :3], axis=1)
    dyaw = np.abs(_wrap_deg(_yaw_deg(after[ok, 3:7]) - _yaw_deg(before[ok, 3:7])))
    return {
        "n": int(ok.sum()),
        "pos_mean_m": float(dpos.mean()),
        "pos_p95_m": float(np.percentile(dpos, 95)),
        "pos_max_m": float(dpos.max()),
        "yaw_mean_deg": float(dyaw.mean()),
        "yaw_p95_deg": float(np.percentile(dyaw, 95)),
        "yaw_max_deg": float(dyaw.max()),
        "_dpos": dpos,
        "_dyaw": dyaw,
    }


def _goal_positions(samples: dict, meta: dict, roles: dict) -> np.ndarray | None:
    goal = roles.get("goal")
    if goal is None:
        return None
    if goal.startswith("command:"):
        key = f"commands/{goal.split(':', 1)[1]}/pose_w"
        return samples[key] if key in samples else None
    key = _pose_key(samples, meta, goal)
    return samples[key] if key else None


def _declared_box(meta: dict, entity: str | None, kind: str | None = None) -> dict | None:
    """First declared range box for ``entity`` (optionally of ``kind``)."""
    if entity is None:
        return None
    for box in meta.get("range_boxes", []):
        if box.get("entity") == entity and (kind is None or box.get("kind") == kind):
            return box
    return None


def _boxes_overlap_xy(a: dict | None, b: dict | None) -> float | None:
    """xy overlap area / area of the smaller box (0..1); None if not computable."""
    if a is None or b is None:
        return None
    ac, ah = np.asarray(a["center"][:2]), np.asarray(a["half_extent"][:2])
    bc, bh = np.asarray(b["center"][:2]), np.asarray(b["half_extent"][:2])
    lo = np.maximum(ac - ah, bc - bh)
    hi = np.minimum(ac + ah, bc + bh)
    inter = float(np.prod(np.clip(hi - lo, 0.0, None)))
    area_a, area_b = float(np.prod(2 * ah)), float(np.prod(2 * bh))
    smaller = min(area_a, area_b)
    if smaller <= 1e-8:
        # Degenerate box (a fixed point / line): overlap = point inside the other box.
        inside = np.all(hi - lo >= -1e-9)
        return 1.0 if inside else 0.0
    return inter / smaller


def compute_stats(samples: dict, meta: dict, roles: dict | None = None) -> dict:
    """Compute every number the flags and the report need. JSON-safe output."""
    roles = roles or resolve_roles(samples, meta)
    kinds = _entity_kinds(meta)
    n = int(meta.get("num_samples", 0)) or next((v.shape[0] for k, v in samples.items() if k.startswith("rigid")), 0)
    stats: dict = {"task": meta.get("task"), "num_samples": n, "roles": roles, "entities": {}, "drift": {}}

    # -- per-entity spread + drift ------------------------------------------
    for name, kind in kinds.items():
        key = f"{kind}/{name}/root_pose"
        if key not in samples:
            continue
        stats["entities"][name] = _pose_stats(samples[key])
        stats["entities"][name]["kind"] = kind
        skey = "settled/" + key
        if skey in samples:
            d = _drift_stats(samples[key], samples[skey])
            stats["drift"][name] = {k: v for k, v in d.items() if not k.startswith("_")}
            if d.get("n"):
                dp, dy = d["_dpos"], d["_dyaw"]
                stats["drift"][name]["frac_moved"] = float(
                    np.mean((dp > DEFAULT_THRESHOLDS["drift_pos_m"]) | (dy > DEFAULT_THRESHOLDS["drift_yaw_deg"]))
                )
        jkey = f"{kind}/{name}/joint_pos"
        if jkey in samples and samples[jkey].size:
            jp = _finite_rows(samples[jkey])
            defaults = np.asarray(meta.get("default_joint_pos", {}).get(name, [0.0] * jp.shape[1]), dtype=np.float64)
            if defaults.shape[0] != jp.shape[1]:
                defaults = np.zeros(jp.shape[1])
            rel = jp - defaults[None, :]
            stats["entities"][name]["joint_names"] = meta.get("joint_names", {}).get(name, [])
            stats["entities"][name]["joint_init_mean"] = [float(v) for v in jp.mean(axis=0)]
            stats["entities"][name]["joint_init_std"] = [float(v) for v in jp.std(axis=0)]
            stats["entities"][name]["joint_init_rel_extent"] = [float(v) for v in (rel.max(axis=0) - rel.min(axis=0))]

    # -- palm ------------------------------------------------------------------
    stats["palm"] = {}
    for key, arr in samples.items():
        if key.startswith("palm/") and key.endswith("/pos"):
            label = key.split("/")[1]
            arr = _finite_rows(arr)
            entry = {"mean": [float(v) for v in arr.mean(axis=0)]} if arr.size else {}
            skey = "settled/" + key
            if skey in samples:
                s = samples[skey]
                ok = np.all(np.isfinite(s), axis=1) & np.all(np.isfinite(samples[key]), axis=1)
                if ok.any():
                    dp = np.linalg.norm(s[ok] - samples[key][ok], axis=1)
                    entry["drift_mean_m"] = float(dp.mean())
                    entry["drift_max_m"] = float(dp.max())
            stats["palm"][label] = entry

    # -- hand <-> object clearance at reset ------------------------------------------
    obj_key = _pose_key(samples, meta, roles["object"]) if roles.get("object") else None
    stats["hand_object"] = None
    if obj_key and "robot/body_pos" in samples:
        bp = samples["robot/body_pos"]  # (N, B, 3)
        op = samples[obj_key][:, None, :3]
        ok = np.all(np.isfinite(op[:, 0, :]), axis=1)
        if ok.any():
            d3 = np.linalg.norm(bp[ok] - op[ok], axis=2)  # (N, B)
            dxy = np.linalg.norm(bp[ok][:, :, :2] - op[ok][:, :, :2], axis=2)
            names = meta.get("robot_body_names", [])
            closest = int(np.argmin(d3.min(axis=0)))
            stats["hand_object"] = {
                "min_3d_m": float(d3.min()),
                "p05_3d_m": float(np.percentile(d3.min(axis=1), 5)),
                "median_3d_m": float(np.median(d3.min(axis=1))),
                "min_xy_m": float(dxy.min()),
                "median_xy_m": float(np.median(dxy.min(axis=1))),
                "closest_body": names[closest] if closest < len(names) else str(closest),
                "frac_within_8cm": float(np.mean(d3.min(axis=1) < 0.08)),
                "frac_xy_within_5cm": float(np.mean(dxy.min(axis=1) < 0.05)),
            }
            stats["_hand_d3"] = d3.min(axis=1)

    # -- goal ------------------------------------------------------------------
    goal_pos = _goal_positions(samples, meta, roles)
    stats["goal"] = None
    if goal_pos is not None:
        gstats = _pose_stats(goal_pos)
        gstats["source"] = roles["goal"]
        gxy = _finite_rows(goal_pos)[:, :2]
        half = DEFAULT_THRESHOLDS["table_half_m"]
        gstats["frac_off_table"] = float(np.mean(np.any(np.abs(gxy) > half, axis=1))) if gxy.size else 0.0
        stats["goal"] = gstats

    # -- init -> goal distance ---------------------------------------------------
    stats["init_goal"] = None
    if obj_key and goal_pos is not None:
        obj = samples[obj_key]
        ok = np.all(np.isfinite(obj[:, :3]), axis=1) & np.all(np.isfinite(goal_pos[:, :3]), axis=1)
        if ok.any():
            d3 = np.linalg.norm(obj[ok, :3] - goal_pos[ok, :3], axis=1)
            dxy = np.linalg.norm(obj[ok, :2] - goal_pos[ok, :2], axis=1)
            dyaw = np.abs(_wrap_deg(_yaw_deg(obj[ok, 3:7]) - _yaw_deg(goal_pos[ok, 3:7])))
            thr = roles.get("distance_threshold_m")
            kind = roles.get("distance_kind") or "3d"
            dsel = dxy if kind == "xy" else d3
            ig = {
                "n": int(ok.sum()),
                "xy_min": float(dxy.min()),
                "xy_p05": float(np.percentile(dxy, 5)),
                "xy_median": float(np.median(dxy)),
                "xy_max": float(dxy.max()),
                "d3_min": float(d3.min()),
                "d3_median": float(np.median(d3)),
                "yaw_diff_median_deg": float(np.median(dyaw)),
                "yaw_diff_min_deg": float(dyaw.min()),
                "distance_threshold_m": thr,
                "distance_kind": kind,
            }
            if thr is not None:
                ig["frac_within_threshold"] = float(np.mean(dsel < thr))
                ig["frac_within_2x_threshold"] = float(np.mean(dsel < DEFAULT_THRESHOLDS["overlap_dist_multiplier"] * thr))
            stats["init_goal"] = ig
            stats["_dxy"] = dxy
            stats["_d3"] = d3

    # -- declared boxes overlap ------------------------------------------------
    obj_box = _declared_box(meta, roles.get("object"), "spawn") or _declared_box(meta, roles.get("object"))
    goal_box = None
    if roles.get("goal"):
        if roles["goal"].startswith("command:"):
            gname = roles["goal"].split(":", 1)[1]
            goal_box = next((b for b in meta.get("range_boxes", []) if b.get("name") == gname), None)
        else:
            goal_box = _declared_box(meta, roles["goal"])
    stats["declared_overlap_xy"] = _boxes_overlap_xy(obj_box, goal_box)

    # -- min-distance constraints -----------------------------------------------
    stats["min_dist_checks"] = []
    for ev in meta.get("reset_events", []):
        min_d = ev.get("min_xy_distance")
        ref = ev.get("reference_entity")
        ent = ev.get("entity")
        if min_d is None or ref is None or ent is None:
            continue
        ka, kb = _pose_key(samples, meta, ent), _pose_key(samples, meta, ref)
        if not ka or not kb:
            continue
        a, b = samples[ka], samples[kb]
        ok = np.all(np.isfinite(a[:, :2]), axis=1) & np.all(np.isfinite(b[:, :2]), axis=1)
        dxy = np.linalg.norm(a[ok, :2] - b[ok, :2], axis=1)
        stats["min_dist_checks"].append(
            {
                "event": ev.get("name"),
                "entity": ent,
                "reference": ref,
                "min_xy_distance": float(min_d),
                "observed_min": float(dxy.min()) if dxy.size else None,
                "frac_violated": float(np.mean(dxy < float(min_d) - 1e-6)) if dxy.size else 0.0,
            }
        )

    # -- terminations during settle -----------------------------------------------
    stats["terminations"] = {}
    for key, arr in samples.items():
        if not key.startswith("term_first_step/"):
            continue
        term = key.split("/", 1)[1]
        fired = arr >= 0
        entry = {
            "frac_fired": float(fired.mean()) if arr.size else 0.0,
            "frac_fired_step1": float(np.mean(arr == 1)) if arr.size else 0.0,
            "frac_fired_after_step1": float(np.mean(arr >= 2)) if arr.size else 0.0,
        }
        if fired.any():
            entry["first_step_median"] = float(np.median(arr[fired]))
        stats["terminations"][term] = entry
    if "done_step" in samples:
        stats["frac_done_during_settle"] = float(np.mean(samples["done_step"] >= 0))
    probe = samples.get("probe/success_step1")
    probe_any = samples.get("probe/success_any")
    stats["probe"] = {
        "n": int(probe.shape[0]) if probe is not None else 0,
        "steps": int(meta.get("probe_steps", 1)),
        "success_step1_rate": float(np.mean(probe)) if probe is not None and probe.size else None,
        "success_any_rate": float(np.mean(probe_any)) if probe_any is not None and probe_any.size else None,
    }

    # -- multi-env sanity ---------------------------------------------------------
    stats["multi_env"] = None
    if "env_origin" in samples and meta.get("num_envs", 1) > 1 and goal_pos is not None and roles["goal"].startswith("command:"):
        origins = samples["env_origin"]
        far = np.linalg.norm(origins[:, :2], axis=1) > 1e-6
        if far.any():
            gxy = goal_pos[far, :2]
            stats["multi_env"] = {
                "n_far_env_samples": int(far.sum()),
                "goal_xy_abs_mean_far_envs": float(np.mean(np.linalg.norm(gxy, axis=1))),
                "goal_xy_abs_mean_origin_env": float(
                    np.mean(np.linalg.norm(goal_pos[~far, :2], axis=1)) if (~far).any() else float("nan")
                ),
            }
    return stats


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def flag_task(stats: dict, thresholds: dict | None = None) -> list[dict]:
    """Return ``[{"flag": NAME, "detail": str}, ...]`` for one task's stats."""
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    flags: list[dict] = []
    roles = stats.get("roles", {})
    obj = roles.get("object")
    es = stats.get("entities", {}).get(obj, {}) if obj else {}

    def add(flag: str, detail: str) -> None:
        flags.append({"flag": flag, "detail": detail})

    if es.get("n"):
        sx, sy = es["std"][0], es["std"][1]
        ex, ey = es["extent"][0], es["extent"][1]
        ys = es["yaw_std_deg"]
        if max(sx, sy) < t["degenerate_xy_std_m"] and ys < t["degenerate_yaw_std_deg"]:
            add("DEGENERATE_INIT", f"{obj}: xy std ({sx:.3f},{sy:.3f}) m, yaw std {ys:.1f} deg -- effectively fixed")
        else:
            if (ex < t["one_d_axis_extent_m"] and ey > t["one_d_other_axis_min_m"]) or (
                ey < t["one_d_axis_extent_m"] and ex > t["one_d_other_axis_min_m"]
            ):
                add("ONE_D_INIT", f"{obj}: extent x={ex:.3f} m, y={ey:.3f} m -- one axis pinned")
            elif ex < t["narrow_extent_m"] and ey < t["narrow_extent_m"]:
                add("NARROW_INIT", f"{obj}: extent x={ex:.3f} m, y={ey:.3f} m (< {t['narrow_extent_m']} m both)")

    ig = stats.get("init_goal")
    if ig:
        f2 = ig.get("frac_within_2x_threshold")
        if f2 is not None and f2 >= t["overlap_frac"]:
            add(
                "INIT_GOAL_OVERLAP",
                f"{f2 * 100:.1f}% of resets start within {t['overlap_dist_multiplier']:.0f}x the success "
                f"threshold ({ig['distance_threshold_m']:.3f} m {ig['distance_kind']}); min dist {ig['xy_min']:.3f} m",
            )
        elif f2 is None and ig.get("xy_min", 1.0) < 0.03:
            add("INIT_GOAL_OVERLAP", f"object and goal can spawn on top of each other (min xy dist {ig['xy_min']:.3f} m)")
        f1 = ig.get("frac_within_threshold")
        if f1 is not None and f1 >= t["trivial_frac"]:
            add("TRIVIAL_AT_RESET", f"{f1 * 100:.1f}% of resets already satisfy the goal-distance criterion")
    ov = stats.get("declared_overlap_xy")
    if ov is not None and ov > 0.3 and not any(f["flag"] == "INIT_GOAL_OVERLAP" for f in flags):
        add("INIT_GOAL_OVERLAP", f"declared object and goal boxes overlap ({ov * 100:.0f}% of the smaller box)")

    succ = stats.get("terminations", {}).get("success")
    if succ and succ.get("frac_fired_after_step1", 0.0) >= t["trivial_frac"]:
        add("TRIVIAL_AT_RESET", f"success termination fires within the settle window in {succ['frac_fired_after_step1'] * 100:.1f}% of resets")
    if succ and succ.get("frac_fired_step1", 0.0) >= t["step1_spurious_frac"]:
        add("STEP1_SPURIOUS_SUCCESS", f"success fires on the first step after reset in {succ['frac_fired_step1'] * 100:.0f}% of resets (stale command metrics)")
    probe = stats.get("probe", {})
    if probe.get("success_step1_rate") is not None and probe["success_step1_rate"] >= t["step1_spurious_frac"]:
        if not any(f["flag"] == "STEP1_SPURIOUS_SUCCESS" for f in flags):
            add("STEP1_SPURIOUS_SUCCESS", f"raw probe: success on step 1 in {probe['success_step1_rate'] * 100:.0f}% of resets")

    for term, entry in stats.get("terminations", {}).items():
        if term in ("success", "time_out"):
            continue
        if entry.get("frac_fired", 0.0) >= t["broken_frac"]:
            add("BROKEN_SPAWN", f"termination '{term}' fires during settle in {entry['frac_fired'] * 100:.1f}% of resets")
    for name, d in stats.get("drift", {}).items():
        if name == "robot" or any(h in name.lower() for h in _STATIC_NAME_HINTS):
            continue
        if d.get("n") and d.get("frac_moved", 0.0) >= t["broken_frac"]:
            add(
                "UNSETTLED_SPAWN",
                f"{name} moves after reset (>{t['drift_pos_m']} m or >{t['drift_yaw_deg']} deg) in {d['frac_moved'] * 100:.1f}% of resets; "
                f"pos p95 {d['pos_p95_m']:.3f} m, yaw p95 {d['yaw_p95_deg']:.1f} deg",
            )
    for label, p in stats.get("palm", {}).items():
        if p.get("drift_max_m", 0.0) > t["palm_drift_m"]:
            add("PALM_DRIFT", f"palm '{label}' moves up to {p['drift_max_m']:.3f} m during the hold -- hold action is not a hold")

    ho = stats.get("hand_object")
    if ho and ho.get("frac_within_8cm", 0.0) >= t["overlap_frac"]:
        add(
            "HAND_INIT_OVERLAP",
            f"{ho['frac_within_8cm'] * 100:.1f}% of resets put the object within 8 cm of a hand body "
            f"(closest: {ho['closest_body']}, min {ho['min_3d_m']:.3f} m, median {ho['median_3d_m']:.3f} m)",
        )
    g = stats.get("goal")
    if g and g.get("frac_off_table", 0.0) > 0.0:
        add("GOAL_OFF_TABLE", f"{g['frac_off_table'] * 100:.1f}% of goals outside the table footprint")
    for c in stats.get("min_dist_checks", []):
        if c.get("frac_violated", 0.0) > 0.0:
            add("MIN_DIST_VIOLATED", f"{c['event']}: {c['frac_violated'] * 100:.1f}% closer than {c['min_xy_distance']} m to {c['reference']} (min {c['observed_min']:.3f} m)")
    me = stats.get("multi_env")
    if me and me["goal_xy_abs_mean_far_envs"] > t["table_half_m"]:
        add(
            "GOAL_NOT_ENV_RELATIVE",
            f"for envs away from the origin the goal lies {me['goal_xy_abs_mean_far_envs']:.2f} m from their own table centre "
            f"(origin env: {me['goal_xy_abs_mean_origin_env']:.2f} m) -- world-frame goal ranges are not offset by env_origins",
        )
    return flags


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Categorical palette (light surface), fixed order; marker shape is the
# secondary encoding so identity never rides on colour alone.
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e6e5e1"
_SURFACE = "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_2, labelsize=8)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _bins(values: np.ndarray, lo: float | None = None, n: int = 30, min_span: float = 0.1) -> np.ndarray:
    """Histogram edges that stay readable when all values are (nearly) identical."""
    values = np.asarray(values, dtype=np.float64)
    vmin = float(values.min()) if lo is None else lo
    vmax = float(values.max())
    if vmax - vmin < min_span:
        pad = 0.5 * (min_span - (vmax - vmin))
        vmin, vmax = (vmin - pad, vmax + pad) if lo is None else (lo, vmin + min_span)
    return np.linspace(vmin, vmax, n + 1)


def _draw_box(ax, box: dict, color: str, label: str | None) -> None:
    from matplotlib.patches import Rectangle

    c, h = box["center"], box["half_extent"]
    if h[0] < 1e-4 and h[1] < 1e-4:
        ax.plot([c[0]], [c[1]], marker="+", color=color, markersize=10, mew=1.5, linestyle="none", label=label)
        return
    ax.add_patch(
        Rectangle(
            (c[0] - h[0], c[1] - h[1]),
            max(2 * h[0], 0.004),
            max(2 * h[1], 0.004),
            fill=False,
            edgecolor=color,
            linewidth=1.2,
            linestyle="--",
            label=label,
        )
    )


def _series_for(samples: dict, meta: dict, roles: dict) -> tuple[list, np.ndarray | None]:
    """Ordered (label, pose-array) list for object / goal / extras."""
    series: list[tuple[str, np.ndarray | None]] = []
    if roles.get("object"):
        series.append((f"object: {roles['object']}", _pose_key(samples, meta, roles["object"])))
    goal_pos = _goal_positions(samples, meta, roles)
    if goal_pos is not None:
        series.append((f"goal: {roles['goal']}", None))
    for e in roles.get("extra", []):
        series.append((e, _pose_key(samples, meta, e)))
    return series, goal_pos


def draw_xy_panel(ax, samples: dict, meta: dict, roles: dict, compact: bool = False, title: str | None = None):
    """Top-down XY scatter of object / goal / extras with table outline, declared
    boxes and palm markers. ``compact`` trims labels/legend for grid use.
    Returns ``(series, goal_pos)`` for callers that plot the same entities elsewhere."""
    from matplotlib.patches import Rectangle

    half = float(meta.get("table_half_xy", [0.75, 0.75])[0]) if isinstance(meta.get("table_half_xy"), list) else 0.75
    half_y = float(meta.get("table_half_xy", [0.75, 0.75])[1]) if isinstance(meta.get("table_half_xy"), list) else 0.75
    ax.add_patch(Rectangle((-half, -half_y), 2 * half, 2 * half_y, fill=False, edgecolor=_INK_2, linewidth=1.0))
    series, goal_pos = _series_for(samples, meta, roles)
    n_pts = 0
    slot = 0
    for label, key in series:
        if key is None and label.startswith("goal:"):
            arr = goal_pos
        elif key is not None:
            arr = samples[key]
        else:
            continue
        arr = _finite_rows(arr)
        if arr.size == 0:
            continue
        n_pts = max(n_pts, arr.shape[0])
        color, marker = _SERIES[slot % len(_SERIES)], _MARKERS[slot % len(_MARKERS)]
        short = label if not compact else label.split(": ", 1)[-1][:14]
        ax.scatter(arr[:, 0], arr[:, 1], s=8 if compact else 14, c=color, marker=marker, alpha=0.65, linewidths=0,
                   label=short if compact else f"{label} (n={arr.shape[0]})")
        # yaw ticks (heading of the body x-axis) for the object and goal only
        if slot < 2 and arr.shape[0] <= 600:
            yaw = quat_yaw(arr[:, 3:7])
            L = 0.02
            ax.plot(
                np.stack([arr[:, 0], arr[:, 0] + L * np.cos(yaw)], axis=1).T,
                np.stack([arr[:, 1], arr[:, 1] + L * np.sin(yaw)], axis=1).T,
                color=color, linewidth=0.5, alpha=0.35,
            )
        slot += 1
    drawn = set()
    for box in meta.get("range_boxes", []):
        color = "#4a3aa7" if box.get("kind") == "goal" else "#199e70"
        label = None if compact else (f"declared {box.get('kind')} box" if box.get("kind") not in drawn else None)
        drawn.add(box.get("kind"))
        _draw_box(ax, box, color, label)
    for key, arr in samples.items():
        if key.startswith("palm/") and key.endswith("/pos"):
            arr = _finite_rows(arr)
            if arr.size:
                ax.scatter(arr[:, 0], arr[:, 1], s=40 if compact else 60, c=_INK, marker="x", linewidths=1.5,
                           label=None if compact else f"palm init ({key.split('/')[1]})")
    ax.set_aspect("equal")
    ax.set_xlim(-half - 0.1, half + 0.1)
    ax.set_ylim(-half_y - 0.1, half_y + 0.1)
    if compact:
        ax.set_xticks([-0.5, 0.0, 0.5])
        ax.set_yticks([-0.5, 0.0, 0.5])
        ax.tick_params(labelsize=6)
        ax.set_title(title or meta.get("task", ""), color=_INK, fontsize=8, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=5.5, frameon=False, handlelength=1.0, ncol=3, columnspacing=0.8)
    else:
        ax.set_xlabel("x (m, env frame; +x = away from robot)", color=_INK_2, fontsize=9)
        ax.set_ylabel("y (m)", color=_INK_2, fontsize=9)
        ax.set_title(title or "Reset positions (top-down)", color=_INK, fontsize=11, loc="left")
        ax.legend(loc="upper center", fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.09), ncol=2)
    return series, goal_pos


def plot_grid(task_dirs: list[str | Path], out_png: str | Path, ncols: int = 5, title: str | None = None) -> Path:
    """One compact XY panel per task directory, laid out in a grid."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    dirs = [Path(d) for d in task_dirs if (Path(d) / "samples.npz").is_file()]
    n = len(dirs)
    nrows = int(math.ceil(n / ncols)) if n else 1
    fig = Figure(figsize=(3.4 * ncols, 3.7 * nrows + 0.5), dpi=120)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(_SURFACE)
    gs = fig.add_gridspec(nrows, ncols, wspace=0.25, hspace=0.45, top=0.94, bottom=0.04, left=0.04, right=0.99)
    for i, d in enumerate(dirs):
        samples, meta = load_task_dir(d)
        stats_path = d / "stats.json"
        roles = json.loads(stats_path.read_text())["roles"] if stats_path.is_file() else resolve_roles(samples, meta)
        flags = json.loads(stats_path.read_text()).get("flags", []) if stats_path.is_file() else []
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        _style(ax)
        name = meta.get("task", d.name).replace("Dexverse-", "").replace("-v0", "")
        sub = f"{name}  (n={meta.get('num_samples', '?')})"
        draw_xy_panel(ax, samples, meta, roles, compact=True, title=sub)
        if flags:
            ax.text(0.99, 0.99, "\n".join(f["flag"] for f in flags), ha="right", va="top", fontsize=5.5, color="#e34948", transform=ax.transAxes)
    key = "blue = object, orange = goal, other colours = props;  X = palm at reset;  dashed = declared spawn (green) / goal (violet) box;  ticks = heading"
    fig.suptitle((title or "Reset positions (top-down) -- all tasks") + "\n" + key, color=_INK, fontsize=11, x=0.04, y=0.995, ha="left", va="top")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=_SURFACE, bbox_inches="tight")
    return out_png



def plot_task(samples: dict, meta: dict, stats: dict, out_png: str | Path, flags: list[dict] | None = None) -> Path:
    """Render the per-task audit figure and return its path."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    roles = stats["roles"]
    kinds = _entity_kinds(meta)
    task = meta.get("task", "?")
    flags = flags if flags is not None else []

    fig = Figure(figsize=(16, 9), dpi=110)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(_SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], height_ratios=[1, 1], wspace=0.28, hspace=0.32)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_yaw = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[0, 2])
    ax_aux = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[1, 2])
    for ax in (ax_xy, ax_yaw, ax_dist, ax_aux):
        _style(ax)
    ax_txt.set_axis_off()

    # ---- (1) top-down scatter --------------------------------------------------
    series, goal_pos = draw_xy_panel(ax_xy, samples, meta, roles, compact=False)

    # ---- (2) yaw histograms ------------------------------------------------------
    plotted = 0
    for slot_i, (label, key) in enumerate(series[:4]):
        arr = goal_pos if key is None and label.startswith("goal:") else (samples[key] if key else None)
        if arr is None:
            continue
        arr = _finite_rows(arr)
        if arr.size == 0:
            continue
        yaw = _yaw_deg(arr[:, 3:7])
        if _circular_std_deg(yaw) < 0.5:
            continue
        ax_yaw.hist(yaw, bins=36, range=(-180, 180), color=_SERIES[slot_i % len(_SERIES)], alpha=0.6, label=label, histtype="stepfilled", edgecolor="none")
        plotted += 1
    ax_yaw.set_xlim(-180, 180)
    ax_yaw.set_xlabel("yaw (deg)", color=_INK_2, fontsize=9)
    ax_yaw.set_title("Yaw at reset" + ("" if plotted else " (no yaw randomisation)"), color=_INK, fontsize=11, loc="left")
    if plotted:
        ax_yaw.legend(fontsize=7, frameon=False)

    # ---- (3) init -> goal distance -------------------------------------------------
    ig = stats.get("init_goal")
    if ig and "_dxy" in stats:
        d = stats["_dxy"] if ig["distance_kind"] == "xy" else stats["_d3"]
        ax_dist.hist(d, bins=_bins(d, lo=0.0, n=40, min_span=0.05), color=_SERIES[0], alpha=0.75, histtype="stepfilled", edgecolor="none")
        thr = ig.get("distance_threshold_m")
        if thr is not None:
            ax_dist.axvline(thr, color="#e34948", linewidth=1.5, label=f"success threshold {thr:.3f} m")
            ax_dist.axvline(2 * thr, color="#e34948", linewidth=1.0, linestyle=":", label="2x threshold")
            ax_dist.legend(fontsize=7, frameon=False)
        ax_dist.set_xlabel(f"object -> goal distance ({ig['distance_kind']}, m)", color=_INK_2, fontsize=9)
        ax_dist.set_title(f"Init->goal distance (median {ig['xy_median']:.3f} m xy)", color=_INK, fontsize=11, loc="left")
    else:
        ax_dist.text(0.5, 0.5, "no spatial goal\n(lift / joint / overlap criterion)", ha="center", va="center", color=_INK_2, fontsize=9, transform=ax_dist.transAxes)
        ax_dist.set_title("Init->goal distance", color=_INK, fontsize=11, loc="left")

    # ---- (4) articulation joints or settle drift ------------------------------------
    obj = roles.get("object")
    jkey = f"articulation/{obj}/joint_pos" if obj and kinds.get(obj) == "articulation" else None
    if jkey and jkey in samples and samples[jkey].size:
        jp = _finite_rows(samples[jkey])
        names = meta.get("joint_names", {}).get(obj, [f"j{i}" for i in range(jp.shape[1])])
        varying = [i for i in range(jp.shape[1]) if jp[:, i].std() > 1e-4] or list(range(min(jp.shape[1], 3)))
        for k, i in enumerate(varying[:6]):
            ax_aux.hist(jp[:, i], bins=_bins(jp[:, i], min_span=0.05), alpha=0.6, color=_SERIES[k % len(_SERIES)], label=names[i] if i < len(names) else f"j{i}", histtype="stepfilled", edgecolor="none")
        ax_aux.set_xlabel("joint position at reset (rad / m)", color=_INK_2, fontsize=9)
        ax_aux.set_title("Articulation init joints" + ("" if any(jp[:, i].std() > 1e-4 for i in range(jp.shape[1])) else " (fixed)"), color=_INK, fontsize=11, loc="left")
        ax_aux.legend(fontsize=7, frameon=False)
    else:
        drift_any = False
        k = 0
        for name in stats.get("drift", {}):
            if name == "robot" or any(h in name.lower() for h in _STATIC_NAME_HINTS):
                continue
            key = f"{kinds[name]}/{name}/root_pose"
            skey = "settled/" + key
            if skey not in samples:
                continue
            dd = _drift_stats(samples[key], samples[skey])
            if not dd.get("n") or dd["pos_max_m"] < 1e-4:
                continue
            ax_aux.hist(dd["_dpos"] * 100.0, bins=_bins(dd["_dpos"] * 100.0, lo=0.0, min_span=0.5), alpha=0.5, color=_SERIES[k % len(_SERIES)], label=name, histtype="stepfilled", edgecolor="none")
            drift_any = True
            k += 1
        ax_aux.set_xlabel(f"position change during {meta.get('settle_steps', '?')} hold steps (cm)", color=_INK_2, fontsize=9)
        ax_aux.set_title("Settle drift" + ("" if drift_any else " (nothing moved > 0.1 mm)"), color=_INK, fontsize=11, loc="left")
        ax_aux.ticklabel_format(axis="x", useOffset=False, style="plain")
        if drift_any:
            ax_aux.set_xlim(left=0.0)
        if drift_any:
            ax_aux.legend(fontsize=7, frameon=False)

    # ---- (5) text ----------------------------------------------------------------
    lines = [f"{task}", f"robot: {meta.get('robot_type', '?')}   samples: {stats.get('num_samples')}   settle: {meta.get('settle_steps', '?')} steps"]
    if obj and obj in stats["entities"] and stats["entities"][obj].get("n"):
        e = stats["entities"][obj]
        lines.append(f"object {obj}: xy extent ({e['extent'][0]:.2f}, {e['extent'][1]:.2f}) m, xy std ({e['std'][0]:.3f}, {e['std'][1]:.3f}) m")
        lines.append(f"    yaw std {e['yaw_std_deg']:.1f} deg, 2cm-bins {e['xy_bins_occupied']}, 30deg-bins {e['yaw_bins_occupied']}")
    g = stats.get("goal")
    if g and g.get("n"):
        lines.append(f"goal {roles['goal']}: xy extent ({g['extent'][0]:.2f}, {g['extent'][1]:.2f}) m, off-table {g['frac_off_table'] * 100:.0f}%")
    if ig:
        thr = ig.get("distance_threshold_m")
        s = f"init->goal xy dist: min {ig['xy_min']:.3f}, p5 {ig['xy_p05']:.3f}, median {ig['xy_median']:.3f} m"
        if thr is not None:
            s += f"; within thr {ig['frac_within_threshold'] * 100:.1f}%, within 2x {ig['frac_within_2x_threshold'] * 100:.1f}%"
        lines.append(s)
    ho = stats.get("hand_object")
    if ho:
        lines.append(f"hand->object at reset: min {ho['min_3d_m']:.3f} m 3D ({ho['closest_body']}), median {ho['median_3d_m']:.3f} m; within 8 cm {ho['frac_within_8cm'] * 100:.0f}%")
    succ = stats.get("terminations", {}).get("success")
    if succ:
        lines.append(f"success fired in settle: step1 {succ['frac_fired_step1'] * 100:.0f}%, later {succ['frac_fired_after_step1'] * 100:.1f}%")
    pr = stats.get("probe", {})
    if pr.get("success_step1_rate") is not None:
        lines.append(
            f"raw probe ({pr['n']} resets x {pr.get('steps', 1)} steps): success on step 1 = {pr['success_step1_rate'] * 100:.0f}%, "
            f"within window = {(pr.get('success_any_rate') or 0.0) * 100:.0f}%"
        )
    other = [f"{k} {v['frac_fired'] * 100:.1f}%" for k, v in stats.get("terminations", {}).items() if k not in ("success", "time_out") and v["frac_fired"] > 0]
    if other:
        lines.append("other terminations: " + ", ".join(other))
    for c in stats.get("min_dist_checks", []):
        lines.append(f"min-dist {c['entity']}-{c['reference']} >= {c['min_xy_distance']}: observed min {c['observed_min']:.3f} m, violated {c['frac_violated'] * 100:.1f}%")
    lines.append("")
    lines.append("FLAGS: " + (", ".join(f["flag"] for f in flags) if flags else "none"))
    for f in flags:
        lines.append(f"  - {f['flag']}: {f['detail']}")
    ax_txt.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", fontsize=7.5, family="monospace", color=_INK, transform=ax_txt.transAxes, wrap=True)

    fig.suptitle(f"Reset-distribution audit -- {task}", color=_INK, fontsize=13, x=0.01, ha="left")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=_SURFACE, bbox_inches="tight")
    return out_png


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def analyze_task_dir(task_dir: str | Path, thresholds: dict | None = None, plot: bool = True) -> dict:
    """Load one task directory, compute stats + flags, write ``stats.json`` and the PNG."""
    task_dir = Path(task_dir)
    samples, meta = load_task_dir(task_dir)
    roles = resolve_roles(samples, meta)
    stats = compute_stats(samples, meta, roles)
    flags = flag_task(stats, thresholds)
    stats["flags"] = flags
    if plot:
        png = plot_task(samples, meta, stats, task_dir / f"{meta.get('task', task_dir.name)}.png", flags)
        stats["png"] = png.name
    (task_dir / "stats.json").write_text(json.dumps(_json_safe(stats), indent=2))
    return stats


def _fmt(v, nd=3, dash="--"):
    if v is None:
        return dash
    try:
        if isinstance(v, float) and not math.isfinite(v):
            return dash
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def write_report(task_dirs: list[str | Path], out_md: str | Path, failures: dict[str, str] | None = None) -> Path:
    """Aggregate ``stats.json`` of every task dir into one markdown report."""
    out_md = Path(out_md)
    rows = []
    for d in task_dirs:
        d = Path(d)
        sp = d / "stats.json"
        if not sp.is_file():
            continue
        rows.append((d, json.loads(sp.read_text())))

    lines = ["# Reset-distribution audit", ""]
    lines.append(f"{len(rows)} task(s) audited. Positions are env-relative metres; the table is 1.5 x 1.5 m centred at the origin, +x away from the robot.")
    lines.append("")
    lines.append("Flags: `DEGENERATE_INIT` fixed init pose · `ONE_D_INIT` one xy axis pinned · `NARROW_INIT` both xy extents < 10 cm · "
                 "`INIT_GOAL_OVERLAP` object can start (almost) at the goal · `TRIVIAL_AT_RESET` success criterion met at reset · "
                 "`BROKEN_SPAWN` a non-success termination fires while holding still · `UNSETTLED_SPAWN` an entity moves >3 cm / >10 deg while holding still (spawned above / inside its support) · `GOAL_OFF_TABLE` · `MIN_DIST_VIOLATED` · "
                 "`STEP1_SPURIOUS_SUCCESS` success fires on the first step because command metrics are stale · `HAND_INIT_OVERLAP` object spawns within 8 cm of a hand body · `PALM_DRIFT` hold action moved the hand.")
    lines.append("")
    lines.append("| task | object | obj xy extent (m) | obj yaw std (deg) | goal | goal xy extent (m) | init->goal xy min / median (m) | thr (m) | hand->obj min / med (m) | success@reset | step1 raw | flags |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d, s in rows:
        roles = s.get("roles", {})
        obj = roles.get("object")
        e = s.get("entities", {}).get(obj, {}) if obj else {}
        g = s.get("goal") or {}
        ig = s.get("init_goal") or {}
        succ = s.get("terminations", {}).get("success", {})
        probe = s.get("probe", {})
        flags = ", ".join(f["flag"] for f in s.get("flags", [])) or "-"
        ext = f"{_fmt(e['extent'][0], 2)} x {_fmt(e['extent'][1], 2)}" if e.get("extent") else "--"
        gext = f"{_fmt(g['extent'][0], 2)} x {_fmt(g['extent'][1], 2)}" if g.get("extent") else "--"
        igs = f"{_fmt(ig.get('xy_min'))} / {_fmt(ig.get('xy_median'))}" if ig else "--"
        s_at_reset = _fmt(ig.get("frac_within_threshold") * 100, 1) + "%" if ig and ig.get("frac_within_threshold") is not None else (
            _fmt(succ.get("frac_fired_after_step1", 0.0) * 100, 1) + "%*" if succ else "--"
        )
        step1 = _fmt(probe.get("success_step1_rate") * 100, 0) + "%" if probe.get("success_step1_rate") is not None else "--"
        ho = s.get("hand_object") or {}
        hos = f"{_fmt(ho.get('min_3d_m'))} / {_fmt(ho.get('median_3d_m'))}" if ho else "--"
        lines.append(
            f"| {s.get('task')} | {obj} | {ext} | {_fmt(e.get('yaw_std_deg'), 1)} | {roles.get('goal') or '-'} | {gext} | {igs} | "
            f"{_fmt(ig.get('distance_threshold_m'))} | {hos} | {s_at_reset} | {step1} | {flags} |"
        )
    lines.append("")
    lines.append("`success@reset`: fraction of resets whose object->goal distance is already below the success threshold "
                 "(or, `*`, the fraction where the env's `success` termination fired during the hold after step 1). "
                 "`step1 raw`: fraction of raw resets where `success` fired on the very first step (stale command metrics bug).")
    lines.append("")
    if failures:
        lines.append("## Failed tasks")
        lines.append("")
        for t, why in failures.items():
            lines.append(f"- `{t}`: {why}")
        lines.append("")
    lines.append("## Per-task figures")
    lines.append("")
    for d, s in rows:
        lines.append(f"### {s.get('task')}")
        lines.append("")
        for f in s.get("flags", []):
            lines.append(f"- **{f['flag']}** -- {f['detail']}")
        if not s.get("flags"):
            lines.append("- no flags")
        for c in s.get("min_dist_checks", []):
            lines.append(f"- min-distance constraint `{c['event']}`: {c['entity']} vs {c['reference']} >= {c['min_xy_distance']} m, observed min {_fmt(c['observed_min'])} m")
        png = s.get("png")
        if png:
            rel = Path(d).relative_to(out_md.parent) if str(Path(d).resolve()).startswith(str(out_md.parent.resolve())) else Path(d)
            lines.append("")
            lines.append(f"![{s.get('task')}]({rel / png})")
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    return out_md
