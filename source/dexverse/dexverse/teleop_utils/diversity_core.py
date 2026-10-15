# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline demo-diversity metrics computed directly from teleop trajectory pickles.

Everything in this module runs on the raw ``dexverse_trajectory`` pickles written
by ``record_demos.py`` -- no simulator, no dexverse imports, numpy only (scipy
optional, for fast k-NN). It powers ``analyze_demo_diversity.py`` and is written
so that replay-derived features (true fingertip contact modes from stage-2
instrumentation) can reuse the same entropy / perplexity / saturation machinery
on their categorical labels.

What is measured, and from what:

* **Initial-state coverage** -- per-episode initial poses of every scene entity
  whose reset pose actually varies across episodes (data-driven detection), plus
  the recorded goal command when present. Reported as per-dimension dispersion
  and as occupied discretization bins, whose permutation-averaged accumulation
  curve is the "marginal coverage gain per added demo" artifact.
* **Strategy modes (kinematic proxies)** -- the approach direction of each palm
  relative to the manipulated entity at *manipulation onset* (the first step the
  entity measurably moves), binned into octants; plus a hand-retreat/return
  counter as a regrasp *proxy*. True contact modes need the replay pass; these
  are the closest sim-free stand-ins and are labeled as proxies in the report.
* **Effective strategy count** -- Shannon entropy over the observed mode
  distribution, reported as perplexity ``exp(H)``.
* **Conditional multimodality** -- k-NN in whitened state space over all
  (state, action) pairs, restricted to neighbors from *other* episodes; the
  action spread among neighbors relative to the global action spread. Near zero
  means the demos are effectively one deterministic policy on shared states.

Palm positions are recovered without forward kinematics by exploiting the
floating-hand rig: the articulation root never moves and the first virtual
joints are pure x/y/z translations (chain order verified against
``robot_agents/*/floating.py`` and the recorded joint values). Single-hand
robots use ``joints[0:3]``; bimanual robots interleave the two chains
(``[rh_x, lh_x, rh_y, lh_y, rh_z, lh_z, ...]``) and each chain hangs off a
fixed mount offset from the articulation root (``rh_base``/``lh_base`` at
y = -0.3 / +0.3 for the shadow bimanual rig). Arm-mounted embodiments (no
``floating_`` prefix) get no palm-based features here -- those need replay.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - scipy is present in practice
    cKDTree = None

# Entities that mirror other state for visualization only; keeping them would
# double-count the goal in coverage metrics.
_VISUAL_ONLY_ENTITIES = ("success_marker",)

# Per-hand mount offset of each translation-joint chain from the articulation
# root, keyed by the hand-family token inside robot_type. Values documented in
# ``robot_agents/shadow/floating.py`` (read from the bimanual USD). Families not
# listed fall back to (0, -/+0.3, 0) with a warning, since every bimanual rig in
# the repo uses the same layout convention.
_BIMANUAL_MOUNT_OFFSETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "shadow": ((0.0, -0.3, 0.0), (0.0, 0.3, 0.0)),
}
_DEFAULT_BIMANUAL_MOUNT = ((0.0, -0.3, 0.0), (0.0, 0.3, 0.0))

_FLOATING_SINGLE_RE = re.compile(r"^floating_([a-z0-9]+)_(right|left)$")
_FLOATING_BIMANUAL_RE = re.compile(r"^floating_([a-z0-9]+)_bimanual$")


"""
Quaternion helpers (wxyz, Isaac Lab convention).
"""


def quat_canonical(q: np.ndarray) -> np.ndarray:
    """Fix the double-cover sign so identical rotations compare equal."""
    q = np.asarray(q, dtype=np.float64)
    sign = np.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q * sign


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.maximum(w * w + x * x + y * y + z * z, 1e-12)
    w, x, y, z = w / np.sqrt(n), x / np.sqrt(n), y / np.sqrt(n), z / np.sqrt(n)
    mat = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1 - 2 * (y * y + z * z)
    mat[..., 0, 1] = 2 * (x * y - w * z)
    mat[..., 0, 2] = 2 * (x * z + w * y)
    mat[..., 1, 0] = 2 * (x * y + w * z)
    mat[..., 1, 1] = 1 - 2 * (x * x + z * z)
    mat[..., 1, 2] = 2 * (y * z - w * x)
    mat[..., 2, 0] = 2 * (x * z - w * y)
    mat[..., 2, 1] = 2 * (y * z + w * x)
    mat[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return mat


def quat_yaw(q: np.ndarray) -> np.ndarray:
    """Heading of the body x-axis in the world xy-plane, radians."""
    mat = quat_to_mat(q)
    return np.arctan2(mat[..., 1, 0], mat[..., 0, 0])


def tilt_from_initial(quats: np.ndarray) -> np.ndarray:
    """Angle (rad) between the initially-up body axis and world up, per step.

    Yaw-invariant by construction: spinning in place reads as zero tilt, which
    keeps PushT rotations out of the "tilt" diagnostic.
    """
    quats = np.asarray(quats, dtype=np.float64)
    mats = quat_to_mat(quats)
    local_up = mats[0].T @ np.array([0.0, 0.0, 1.0])
    up_world = mats @ local_up
    return np.arccos(np.clip(up_world[..., 2], -1.0, 1.0))


def geodesic_angle_from_initial(quats: np.ndarray) -> np.ndarray:
    """Full rotation angle (rad) between each orientation and the first one."""
    q = quat_canonical(quats)
    dots = np.abs(np.sum(q * q[0], axis=-1))
    return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))


"""
Pickle loading.
"""


@dataclass
class EpisodeArrays:
    """One demo episode parsed into plain arrays. Poses are (pos3, quat4) wxyz."""

    actions: np.ndarray  # (T, A)
    joint_pos: np.ndarray  # (T+1, J) robot joints, per-step
    robot_root: np.ndarray  # (7,) static for floating rigs
    rigid_poses: dict[str, np.ndarray]  # name -> (T+1, 7)
    art_joints: dict[str, np.ndarray]  # non-robot articulation -> (T+1, Jk)
    art_roots: dict[str, np.ndarray]  # non-robot articulation -> (7,) initial root
    goal_pose: np.ndarray | None  # flattened goal command, if recorded
    success: bool
    num_steps: int


@dataclass
class TaskDemos:
    task: str
    category: str
    robot_type: str | None
    source_files: list[str]
    episodes: list[EpisodeArrays] = field(default_factory=list)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)


def _flat(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _parse_episode(ep: dict) -> EpisodeArrays | None:
    states = ep.get("states")
    actions = np.asarray(ep.get("actions"), dtype=np.float64)
    if not isinstance(states, list) or len(states) < 2 or actions.ndim != 2 or actions.shape[0] == 0:
        return None
    arts = states[0].get("articulation", {}) or {}
    if "robot" not in arts:
        return None

    joint_pos = np.stack([_flat(s["articulation"]["robot"]["joint_position"]) for s in states])
    robot_root = _flat(states[0]["articulation"]["robot"]["root_pose"])[:7]

    rigid_names = list((states[0].get("rigid_object", {}) or {}).keys())
    rigid_poses = {
        name: np.stack([_flat(s["rigid_object"][name]["root_pose"])[:7] for s in states]) for name in rigid_names
    }
    art_names = [n for n in arts.keys() if n != "robot"]
    art_joints = {
        name: np.stack([_flat(s["articulation"][name]["joint_position"]) for s in states]) for name in art_names
    }
    art_roots = {name: _flat(states[0]["articulation"][name]["root_pose"])[:7] for name in art_names}

    goal = ep.get("goal_pose")
    goal = _flat(goal) if goal is not None else None

    return EpisodeArrays(
        actions=actions,
        joint_pos=joint_pos,
        robot_root=robot_root,
        rigid_poses=rigid_poses,
        art_joints=art_joints,
        art_roots=art_roots,
        goal_pose=goal,
        success=bool(ep.get("success")),
        num_steps=int(ep.get("num_steps", actions.shape[0])),
    )


def discover_task_pickles(task_dir: Path) -> list[Path]:
    """Prefer the merged ``<dirname>.pkl``; otherwise every session pickle."""
    merged = task_dir / f"{task_dir.name}.pkl"
    if merged.is_file():
        return [merged]
    return sorted(task_dir.glob("*.pkl"))


def load_task_demos(task_dir: Path, include_failed: bool = False) -> TaskDemos:
    """Load all demos for one task directory (``<category>/<task>/``)."""
    paths = discover_task_pickles(task_dir)
    if not paths:
        raise FileNotFoundError(f"No pickles under {task_dir}")
    task_name, robot_type = task_dir.name, None
    demos = TaskDemos(
        task=task_name,
        category=task_dir.parent.name,
        robot_type=None,
        source_files=[str(p) for p in paths],
    )
    for path in paths:
        with open(path, "rb") as fp:
            payload = pickle.load(fp)
        if payload.get("format") != "dexverse_trajectory":
            continue
        robot_type = robot_type or payload.get("robot_type")
        for ep in payload.get("episodes") or []:
            if not include_failed and not ep.get("success"):
                continue
            parsed = _parse_episode(ep)
            if parsed is not None:
                demos.episodes.append(parsed)
    demos.robot_type = robot_type
    return demos


def demos_from_episodes(
    episodes: list[dict],
    *,
    task: str,
    category: str = "",
    robot_type: str | None = None,
    include_failed: bool = False,
    history_dir: Path | None = None,
    exclude: Path | None = None,
) -> TaskDemos:
    """Build a :class:`TaskDemos` from live, in-memory episode dicts.

    ``episodes`` are the raw dicts ``TrajectoryPickleRecorder`` accumulates in
    ``_episodes`` -- after ``finalize_episode`` they carry exactly the schema
    :func:`_parse_episode` expects, so a live teleop session can be scored
    without waiting for the pickle to be re-read.

    When ``history_dir`` is given, previously recorded pickles in that directory
    are loaded first so the numbers describe the whole dataset for the task, not
    just this session. ``exclude`` skips one path -- pass the session's own
    output file, which ``finalize_episode`` keeps flushed and would otherwise
    double-count every in-memory episode.
    """
    demos = TaskDemos(task=task, category=category, robot_type=robot_type, source_files=[])

    if history_dir is not None and history_dir.is_dir():
        exclude_resolved = exclude.resolve() if exclude is not None else None
        for path in sorted(history_dir.glob("*.pkl")):
            if exclude_resolved is not None and path.resolve() == exclude_resolved:
                continue
            try:
                with open(path, "rb") as fp:
                    payload = pickle.load(fp)
            except Exception:  # a half-written sibling session must not kill teleop
                continue
            if payload.get("format") != "dexverse_trajectory":
                continue
            demos.source_files.append(str(path))
            demos.robot_type = demos.robot_type or payload.get("robot_type")
            for ep in payload.get("episodes") or []:
                if not include_failed and not ep.get("success"):
                    continue
                parsed = _parse_episode(ep)
                if parsed is not None:
                    demos.episodes.append(parsed)

    for ep in episodes:
        if not include_failed and not ep.get("success"):
            continue
        parsed = _parse_episode(ep)
        if parsed is not None:
            demos.episodes.append(parsed)
    return demos


"""
Entity selection: which scene entity does this task manipulate?
"""


def _rigid_move_range(poses: np.ndarray) -> float:
    return float(np.abs(poses[:, :3] - poses[0, :3]).max())


def _art_move_range(joints: np.ndarray) -> float:
    return float(np.abs(joints - joints[0]).max())


def select_manipulated_entity(demos: TaskDemos, probe_episodes: int = 10) -> tuple[str, str]:
    """Return ``(group, name)`` of the manipulated entity.

    Priority: a rigid object literally named ``object`` that moves; else the
    most-moving rigid object; else the most-moving non-robot articulation.
    Max motion is taken over a probe of episodes so a single quiet demo cannot
    misroute the choice. Visual-only entities are excluded outright.
    """
    probe = demos.episodes[: max(1, probe_episodes)]
    rigid_move: dict[str, float] = {}
    art_move: dict[str, float] = {}
    for ep in probe:
        for name, poses in ep.rigid_poses.items():
            if name in _VISUAL_ONLY_ENTITIES:
                continue
            rigid_move[name] = max(rigid_move.get(name, 0.0), _rigid_move_range(poses))
        for name, joints in ep.art_joints.items():
            art_move[name] = max(art_move.get(name, 0.0), _art_move_range(joints))

    if rigid_move.get("object", 0.0) > 0.02:
        return "rigid_object", "object"
    moving_rigids = {n: v for n, v in rigid_move.items() if v > 0.02}
    if moving_rigids:
        return "rigid_object", max(moving_rigids, key=moving_rigids.get)
    if art_move:
        return "articulation", max(art_move, key=art_move.get)
    if rigid_move:
        return "rigid_object", max(rigid_move, key=rigid_move.get)
    raise ValueError(f"{demos.task}: no candidate manipulated entity found")


def entity_positions(ep: EpisodeArrays, group: str, name: str) -> np.ndarray:
    """Per-step world position (T+1, 3) of the entity's reference point."""
    if group == "rigid_object":
        return ep.rigid_poses[name][:, :3]
    # Articulated fixtures do not translate; their root is the reference point.
    return np.broadcast_to(ep.art_roots[name][:3], (ep.joint_pos.shape[0], 3))


"""
Palm recovery from the floating-rig virtual joints.
"""


@dataclass
class PalmModel:
    """Extracts per-hand palm(=wrist chain origin) positions from robot joints."""

    kind: str  # "single" | "bimanual" | "none"
    reason: str = ""
    mount_offsets: tuple[tuple[float, float, float], ...] = ()
    hand_labels: tuple[str, ...] = ()

    def positions(self, ep: EpisodeArrays) -> np.ndarray | None:
        """(T+1, n_hands, 3) world positions, or None if unsupported."""
        if self.kind == "single":
            trans = ep.joint_pos[:, :3]
            return (ep.robot_root[:3] + trans)[:, None, :]
        if self.kind == "bimanual":
            # Interleaved virtual chains: joints[[0,2,4]] is one hand's x/y/z
            # translation, joints[[1,3,5]] the other's (verified against the
            # recorded init values 0.5/0.5/0/0/0.3/0.3).
            out = np.empty((ep.joint_pos.shape[0], 2, 3))
            for slot in (0, 1):
                trans = ep.joint_pos[:, [slot, slot + 2, slot + 4]]
                out[:, slot, :] = ep.robot_root[:3] + np.asarray(self.mount_offsets[slot]) + trans
            return out
        return None


def build_palm_model(demos: TaskDemos) -> PalmModel:
    robot_type = demos.robot_type or ""
    if _FLOATING_SINGLE_RE.match(robot_type):
        model = PalmModel(kind="single", hand_labels=("hand",))
    elif _FLOATING_BIMANUAL_RE.match(robot_type):
        family = _FLOATING_BIMANUAL_RE.match(robot_type).group(1)
        offsets = _BIMANUAL_MOUNT_OFFSETS.get(family, _DEFAULT_BIMANUAL_MOUNT)
        # Slot->side assignment: the chain mounted at negative y is the one
        # whose recovered positions must start at root_y - 0.3. Both slots init
        # identically in joint space, so try both assignments and keep the one
        # whose hands do not cross at t=0 (they start mirrored about the root).
        model = PalmModel(kind="bimanual", mount_offsets=offsets, hand_labels=("hand_neg_y", "hand_pos_y"))
    else:
        return PalmModel(kind="none", reason=f"robot_type={robot_type!r} is not a floating rig; needs replay FK")

    # Workspace sanity check: recovered palms must be near the table, else the
    # virtual-joint layout assumption does not hold for this rig.
    probe = demos.episodes[: min(5, len(demos.episodes))]
    for ep in probe:
        pos = model.positions(ep)
        if pos is None or not (np.all(np.abs(pos[..., :2]) < 1.5) and np.all(pos[..., 2] > -0.2) and np.all(pos[..., 2] < 2.0)):
            return PalmModel(kind="none", reason="virtual-joint palm recovery failed the workspace sanity check")
    return model


"""
Manipulation onset, approach octants, regrasp proxy, motion diagnostics.
"""


def manipulation_onset(ep: EpisodeArrays, group: str, name: str, pos_eps: float = 0.01, joint_eps: float = 0.01) -> int:
    """First step index where the manipulated entity has measurably moved.

    Falls back to the step of closest palm approach -- or 0 -- when the entity
    never crosses the threshold (should not happen in successful demos).
    """
    if group == "rigid_object":
        poses = ep.rigid_poses[name]
        moved = np.abs(poses[:, :3] - poses[0, :3]).max(axis=1) > pos_eps
        tilted = tilt_from_initial(poses[:, 3:7]) > np.deg2rad(5.0)
        hits = np.flatnonzero(moved | tilted)
    else:
        joints = ep.art_joints[name]
        hits = np.flatnonzero(np.abs(joints - joints[0]).max(axis=1) > joint_eps)
    return int(hits[0]) if hits.size else 0


def approach_octant(
    palm_pos: np.ndarray,
    entity_pos: np.ndarray,
    entity_yaw: float,
    azimuth_bins: int = 4,
) -> tuple[str, float, float]:
    """Bin the palm->entity offset into an azimuth x elevation octant.

    Azimuth is measured in the entity's yaw-derotated frame so that "approached
    the handle side" is the same mode regardless of how the object spawned.
    Returns ``(label, azimuth_deg, elevation_deg)``.
    """
    offset = np.asarray(palm_pos, dtype=np.float64) - np.asarray(entity_pos, dtype=np.float64)
    az = np.arctan2(offset[1], offset[0]) - entity_yaw
    az = (az + np.pi) % (2.0 * np.pi) - np.pi
    elev = np.arctan2(offset[2], max(np.hypot(offset[0], offset[1]), 1e-9))
    az_bin = int(((az + np.pi) / (2.0 * np.pi)) * azimuth_bins) % azimuth_bins
    label = f"az{az_bin}_{'up' if elev >= 0.0 else 'dn'}"
    return label, float(np.degrees(az)), float(np.degrees(elev))


def retreat_return_count(
    palm_dist: np.ndarray, onset: int, rise_near: float = 0.04, rise_far: float = 0.10
) -> int:
    """Regrasp *proxy*: after onset, count near -> far -> near excursions.

    The near/far bands sit ``rise_near`` / ``rise_far`` above the episode's own
    post-onset closest approach, because the wrist-chain origin we can recover
    without FK carries an embodiment-dependent forearm offset that makes any
    absolute threshold meaningless. Hysteresis between the bands keeps jitter
    from counting. True regrasps (contact loss and recovery) need the stage-2
    contact replay; this proxies them with wrist-to-entity distance.
    """
    d = np.asarray(palm_dist, dtype=np.float64)[onset:]
    if d.size == 0:
        return 0
    near_threshold = float(d.min()) + rise_near
    far_threshold = float(d.min()) + rise_far
    count = 0
    state_near = bool(d[0] < near_threshold)
    for value in d[1:]:
        if state_near and value > far_threshold:
            state_near = False
        elif not state_near and value < near_threshold:
            state_near = True
            count += 1
    # The first entry into "near" is the initial approach, not a re-grasp.
    return max(0, count - (0 if bool(d[0] < near_threshold) else 1))


def motion_diagnostics(ep: EpisodeArrays, group: str, name: str) -> dict[str, float]:
    """Cheap physical strategy descriptors of the manipulated entity."""
    if group == "rigid_object":
        poses = ep.rigid_poses[name]
        pos = poses[:, :3]
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        return {
            "max_lift_m": float((pos[:, 2] - pos[0, 2]).max()),
            "max_tilt_deg": float(np.degrees(tilt_from_initial(poses[:, 3:7]).max())),
            "max_rotation_deg": float(np.degrees(geodesic_angle_from_initial(poses[:, 3:7]).max())),
            "net_xy_m": float(np.linalg.norm(pos[-1, :2] - pos[0, :2])),
            "path_length_m": float(steps.sum()),
        }
    joints = ep.art_joints[name]
    travel = np.abs(joints - joints[0])
    return {
        "max_joint_travel": float(travel.max()),
        "final_joint_travel": float(np.abs(joints[-1] - joints[0]).max()),
        "n_direction_reversals": int(
            np.sum(np.abs(np.diff(np.sign(np.diff(joints[:, int(np.argmax(travel.max(axis=0)))])))) > 0) // 2
        ),
    }


"""
Initial-state coverage.
"""


def _circular_std_deg(angles_deg: np.ndarray) -> float:
    """Circular standard deviation in degrees (wrap-safe, unlike np.std)."""
    rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    resultant = float(np.abs(np.exp(1j * rad).mean()))
    if resultant >= 1.0:
        return 0.0
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(resultant, 1e-12)))))


def _initial_pose_rows(demos: TaskDemos) -> dict[str, np.ndarray]:
    """Per-entity (N, 7) initial poses across episodes, plus goal_pose rows."""
    rows: dict[str, list[np.ndarray]] = {}
    for ep in demos.episodes:
        for name, poses in ep.rigid_poses.items():
            if name in _VISUAL_ONLY_ENTITIES:
                continue
            rows.setdefault(name, []).append(poses[0])
        for name, joints in ep.art_joints.items():
            root = np.zeros(7)
            root[: min(7, ep.art_roots[name].size)] = ep.art_roots[name][:7]
            # Fixtures encode their randomization in the root pose; joint inits
            # are appended separately below via a pseudo-row when they vary.
            rows.setdefault(name, []).append(root)
        if ep.goal_pose is not None and ep.goal_pose.size >= 3:
            row = np.zeros(7)
            row[: min(7, ep.goal_pose.size)] = ep.goal_pose[:7]
            rows.setdefault("goal_pose", []).append(row)
    return {name: np.stack(v) for name, v in rows.items() if len(v) == len(demos.episodes)}


def initial_state_coverage(
    demos: TaskDemos,
    xy_bin_m: float = 0.02,
    yaw_bin_deg: float = 30.0,
    min_pos_std_m: float = 0.002,
    min_yaw_std_deg: float = 1.0,
) -> dict:
    """Dispersion and occupied-bin coverage over entities whose reset varies."""
    pose_rows = _initial_pose_rows(demos)
    per_entity: dict[str, dict] = {}
    episode_bins: list[set] = [set() for _ in range(demos.num_episodes)]

    for name, rows in pose_rows.items():
        pos = rows[:, :3]
        yaw = np.degrees(quat_yaw(rows[:, 3:7])) if np.any(rows[:, 3:7]) else np.zeros(len(rows))
        pos_std = pos.std(axis=0)
        yaw_std_deg = _circular_std_deg(yaw)
        randomized = bool(pos_std.max() > min_pos_std_m or yaw_std_deg > min_yaw_std_deg)
        if not randomized:
            continue
        bins_x = np.floor(pos[:, 0] / xy_bin_m).astype(int)
        bins_y = np.floor(pos[:, 1] / xy_bin_m).astype(int)
        bins_yaw = np.floor(yaw / yaw_bin_deg).astype(int)
        occupied = set()
        for i in range(len(rows)):
            key = (name, int(bins_x[i]), int(bins_y[i]), int(bins_yaw[i]))
            occupied.add(key)
            episode_bins[i].add(key)
        per_entity[name] = {
            "pos_std_m": [float(v) for v in pos_std],
            "xy_range_m": [float(np.ptp(pos[:, 0])), float(np.ptp(pos[:, 1]))],
            "yaw_std_deg": yaw_std_deg,
            "n_bins_occupied": len(occupied),
        }

    # Articulated fixtures can also randomize their initial joint values (a
    # laptop's starting lid angle); cover them with their own binning.
    art_common = set.intersection(*[set(ep.art_joints) for ep in demos.episodes]) if demos.episodes else set()
    for name in sorted(art_common):
        inits = np.stack([ep.art_joints[name][0] for ep in demos.episodes])
        if inits.std(axis=0).max() <= 0.005:
            continue
        joint_bin = 0.05  # rad or m per bin; virtual fixtures mix both units
        keys = [tuple((name, "joints", *np.floor(row / joint_bin).astype(int))) for row in inits]
        for i, key in enumerate(keys):
            episode_bins[i].add(key)
        per_entity[f"{name}(joints)"] = {
            "joint_init_std": [float(v) for v in inits.std(axis=0)],
            "n_bins_occupied": len(set(keys)),
        }

    return {
        "bin_params": {"xy_bin_m": xy_bin_m, "yaw_bin_deg": yaw_bin_deg},
        "randomized_entities": sorted(per_entity.keys()),
        "per_entity": per_entity,
        "episode_bins": episode_bins,  # consumed by saturation_curve, stripped from JSON
    }


"""
Entropy / perplexity and saturation.
"""


def entropy_perplexity(counts: dict[str, int]) -> dict:
    total = float(sum(counts.values()))
    if total <= 0:
        return {"counts": counts, "distinct": 0, "entropy_nats": 0.0, "perplexity": 0.0, "top_share": 0.0}
    probs = np.array([c / total for c in counts.values() if c > 0], dtype=np.float64)
    entropy = float(-(probs * np.log(probs)).sum())
    return {
        "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "distinct": int(len(probs)),
        "entropy_nats": entropy,
        "perplexity": float(np.exp(entropy)),
        "top_share": float(probs.max()),
    }


def saturation_curve(per_episode_items: list, n_permutations: int = 200, seed: int = 0) -> dict:
    """Permutation-averaged cumulative distinct-item curve over added demos.

    ``per_episode_items`` holds one hashable label *or* one set of hashables per
    episode. Returns the mean curve, the marginal gain per added demo, and
    ``n95`` -- how many demos reach 95% of the final coverage, i.e. the point
    past which additional demos stop adding information of this kind.
    """
    sets = [items if isinstance(items, (set, frozenset)) else {items} for items in per_episode_items]
    n = len(sets)
    if n == 0:
        return {"curve": [], "marginal": [], "n95": 0, "final": 0}
    rng = np.random.default_rng(seed)
    acc = np.zeros(n, dtype=np.float64)
    for _ in range(n_permutations):
        seen: set = set()
        for pos, idx in enumerate(rng.permutation(n)):
            seen |= sets[idx]
            acc[pos] += len(seen)
    curve = acc / n_permutations
    marginal = np.diff(np.concatenate([[0.0], curve]))
    final = curve[-1]
    n95 = int(np.argmax(curve >= 0.95 * final) + 1) if final > 0 else 0
    return {
        "curve": [float(v) for v in curve],
        "marginal": [float(v) for v in marginal],
        "n95": n95,
        "final": float(final),
    }


"""
Conditional multimodality: k-NN action spread in whitened state space.
"""


def _episode_state_features(
    ep: EpisodeArrays, moving_rigids: list[str], art_names: list[str], goal_dim: int
) -> np.ndarray:
    """(T, D) per-step state features for the T recorded actions (pre-step state)."""
    steps = ep.actions.shape[0]
    parts = [ep.joint_pos[:steps]]
    for name in moving_rigids:
        poses = ep.rigid_poses[name][:steps]
        parts.extend([poses[:, :3], quat_canonical(poses[:, 3:7])])
    for name in art_names:
        parts.append(ep.art_joints[name][:steps])
    if goal_dim > 0:
        parts.append(np.broadcast_to(ep.goal_pose[:goal_dim], (steps, goal_dim)))
    return np.concatenate(parts, axis=1)


def knn_conditional_multimodality(
    demos: TaskDemos,
    k: int = 8,
    max_pool: int = 60_000,
    max_queries: int = 8_000,
    seed: int = 0,
) -> dict:
    """Action variability among nearest *cross-episode* state neighbors.

    Both states and actions are whitened per-dimension over the pooled dataset.
    The headline number is ``spread_ratio``: mean per-dimension action std among
    each query's k nearest other-episode neighbors, divided by the global std
    (== 1 after whitening). ~0 means one deterministic policy; ~1 means actions
    at similar states are as varied as the whole dataset. ``random_baseline``
    repeats the computation with random other-episode samples in place of
    neighbors and should sit near 1.
    """
    if demos.num_episodes < 2:
        return {"skipped": "needs >= 2 episodes"}

    # Feature layout must be identical across episodes.
    rigid_names = sorted(set.intersection(*[set(ep.rigid_poses) for ep in demos.episodes]) - set(_VISUAL_ONLY_ENTITIES))
    moving = [
        n for n in rigid_names if max(_rigid_move_range(ep.rigid_poses[n]) for ep in demos.episodes[:10]) > 0.005
    ]
    # Randomized-but-static entities (receptacles, goals) are part of the state
    # too: conditioning on them keeps "similar state" honest across episodes.
    static_randomized = [
        n
        for n in rigid_names
        if n not in moving and np.std([ep.rigid_poses[n][0, :3] for ep in demos.episodes], axis=0).max() > 0.002
    ]
    art_names = sorted(set.intersection(*[set(ep.art_joints) for ep in demos.episodes]))
    feature_rigids = moving + static_randomized
    goal_sizes = {0 if ep.goal_pose is None else min(7, ep.goal_pose.size) for ep in demos.episodes}
    goal_dim = goal_sizes.pop() if len(goal_sizes) == 1 else 0

    states, actions, episode_ids = [], [], []
    for idx, ep in enumerate(demos.episodes):
        feats = _episode_state_features(ep, feature_rigids, art_names, goal_dim)
        states.append(feats)
        actions.append(ep.actions)
        episode_ids.append(np.full(feats.shape[0], idx))
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    episode_ids = np.concatenate(episode_ids)

    rng = np.random.default_rng(seed)
    if states.shape[0] > max_pool:
        keep = np.sort(rng.choice(states.shape[0], size=max_pool, replace=False))
        states, actions, episode_ids = states[keep], actions[keep], episode_ids[keep]

    def whiten(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = x.std(axis=0)
        live = std > 1e-8
        return (x[:, live] - x[:, live].mean(axis=0)) / std[live], live

    s_w, _ = whiten(states)
    a_w, a_live = whiten(actions)
    if a_w.shape[1] == 0 or s_w.shape[1] == 0:
        return {"skipped": "no varying state or action dimensions"}

    n = s_w.shape[0]
    queries = np.sort(rng.choice(n, size=min(max_queries, n), replace=False))
    _, ep_counts = np.unique(episode_ids, return_counts=True)
    k_search = int(min(n, k + ep_counts.max()))

    if cKDTree is not None:
        tree = cKDTree(s_w)
        spreads, state_dists = [], []
        for chunk in np.array_split(queries, max(1, len(queries) // 2048)):
            dist, idx = tree.query(s_w[chunk], k=k_search, workers=-1)
            for row, (di, ii) in enumerate(zip(dist, idx)):
                qi = chunk[row]
                mask = episode_ids[ii] != episode_ids[qi]
                picked = ii[mask][:k]
                if picked.size < max(2, k // 2):
                    continue
                spreads.append(a_w[picked].std(axis=0).mean())
                state_dists.append(di[mask][: picked.size].mean())
    else:  # chunked brute force
        spreads, state_dists = [], []
        for qi in queries:
            d = np.linalg.norm(s_w - s_w[qi], axis=1)
            d[episode_ids == episode_ids[qi]] = np.inf
            picked = np.argpartition(d, k)[:k]
            spreads.append(a_w[picked].std(axis=0).mean())
            state_dists.append(float(d[picked].mean()))

    if not spreads:
        return {"skipped": "no queries with enough cross-episode neighbors"}

    # Null control: random other-episode samples instead of nearest neighbors.
    rand_spreads = []
    for qi in rng.choice(queries, size=min(1000, len(queries)), replace=False):
        others = np.flatnonzero(episode_ids != episode_ids[qi])
        picked = rng.choice(others, size=min(k, others.size), replace=False)
        rand_spreads.append(a_w[picked].std(axis=0).mean())

    return {
        "k": k,
        "n_pool": int(n),
        "n_queries": int(len(spreads)),
        "state_dim": int(s_w.shape[1]),
        "action_dim_used": int(a_w.shape[1]),
        "action_dims_dropped": int((~a_live).sum()),
        "spread_ratio": float(np.mean(spreads)),
        "spread_ratio_median": float(np.median(spreads)),
        "random_baseline": float(np.mean(rand_spreads)),
        # Neighbor and random spreads share the same small-sample bias (std of
        # k samples underestimates sigma), so their ratio is the fair headline:
        # 1.0 = actions at similar states as varied as anywhere; 0 = one policy.
        "spread_vs_baseline": float(np.mean(spreads) / max(np.mean(rand_spreads), 1e-9)),
        "mean_neighbor_state_dist": float(np.mean(state_dists)),
        "state_features": {"moving": moving, "static_randomized": static_randomized, "articulations": art_names},
    }


"""
Task-level orchestration.
"""


def _resolve_bimanual_slots(demos: TaskDemos, model: PalmModel, group: str, name: str) -> PalmModel:
    """Pick the slot->mount-offset assignment that the data supports.

    Both interleave slots start at identical joint values, so the pickle alone
    cannot say which chain carries which +/-y mount. The wrong assignment puts
    every palm ~0.6 m off in y; the right one puts palms near the manipulated
    entity at onset. Try both, keep the closer.
    """
    if model.kind != "bimanual":
        return model
    candidates = [model.mount_offsets, tuple(reversed(model.mount_offsets))]
    scores = []
    for offsets in candidates:
        trial = PalmModel(kind="bimanual", mount_offsets=offsets, hand_labels=model.hand_labels)
        dists = []
        for ep in demos.episodes[: min(8, len(demos.episodes))]:
            onset = manipulation_onset(ep, group, name)
            palms = trial.positions(ep)[onset]
            entity = entity_positions(ep, group, name)[onset]
            dists.append(np.linalg.norm(palms - entity, axis=1).mean())
        scores.append(float(np.mean(dists)))
    best = candidates[int(np.argmin(scores))]
    return PalmModel(kind="bimanual", mount_offsets=best, hand_labels=model.hand_labels)


def analyze_task(
    task_dir: Path,
    *,
    include_failed: bool = False,
    xy_bin_m: float = 0.02,
    yaw_bin_deg: float = 30.0,
    azimuth_bins: int = 4,
    knn_k: int = 8,
    knn_max_pool: int = 60_000,
    knn_max_queries: int = 8_000,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict:
    """Run every offline diversity metric for one task directory; JSON-safe result."""
    demos = load_task_demos(task_dir, include_failed=include_failed)
    return analyze_demos(
        demos,
        include_failed=include_failed,
        xy_bin_m=xy_bin_m,
        yaw_bin_deg=yaw_bin_deg,
        azimuth_bins=azimuth_bins,
        knn_k=knn_k,
        knn_max_pool=knn_max_pool,
        knn_max_queries=knn_max_queries,
        n_permutations=n_permutations,
        seed=seed,
    )


def analyze_demos(
    demos: TaskDemos,
    *,
    include_failed: bool = False,
    xy_bin_m: float = 0.02,
    yaw_bin_deg: float = 30.0,
    azimuth_bins: int = 4,
    knn_k: int = 8,
    knn_max_pool: int = 60_000,
    knn_max_queries: int = 8_000,
    n_permutations: int = 200,
    seed: int = 0,
    include_knn: bool = True,
) -> dict:
    """Run the offline diversity metrics over an already-loaded :class:`TaskDemos`.

    Split out of :func:`analyze_task` so live teleop can score episodes it is
    holding in memory without a pickle round-trip. ``include_knn=False`` skips
    :func:`knn_conditional_multimodality`, which is the only stage whose cost
    grows with dataset size (~0.6 s at 50 episodes) -- the rest is milliseconds
    and is safe to recompute after every recorded trajectory.
    """
    warnings_list: list[str] = []
    if demos.num_episodes == 0:
        return {"task": demos.task, "category": demos.category, "error": "no episodes"}

    group, name = select_manipulated_entity(demos)
    palm_model = build_palm_model(demos)
    if palm_model.kind == "none":
        warnings_list.append(f"palm-based features skipped: {palm_model.reason}")
    palm_model = _resolve_bimanual_slots(demos, palm_model, group, name)

    # --- initial-state coverage + its saturation ---------------------------
    coverage = initial_state_coverage(demos, xy_bin_m=xy_bin_m, yaw_bin_deg=yaw_bin_deg)
    episode_bins = coverage.pop("episode_bins")
    coverage["saturation"] = saturation_curve(episode_bins, n_permutations=n_permutations, seed=seed)

    # --- per-episode strategy features -------------------------------------
    octant_labels: list = []
    octant_details: list[dict] = []
    regrasp_counts: list[int] = []
    onset_steps: list[int] = []
    motions: list[dict[str, float]] = []
    for ep in demos.episodes:
        onset = manipulation_onset(ep, group, name)
        onset_steps.append(onset)
        motions.append(motion_diagnostics(ep, group, name))
        palms = palm_model.positions(ep)
        if palms is None:
            continue
        entity_pos = entity_positions(ep, group, name)
        if group == "rigid_object":
            yaw = float(quat_yaw(ep.rigid_poses[name][onset, 3:7]))
        else:
            yaw = float(quat_yaw(ep.art_roots[name][3:7]))
        hand_labels = []
        detail = {}
        for hand in range(palms.shape[1]):
            label, az, elev = approach_octant(palms[onset, hand], entity_pos[onset], yaw, azimuth_bins)
            hand_labels.append(label)
            detail[palm_model.hand_labels[hand]] = {"octant": label, "azimuth_deg": az, "elevation_deg": elev}
        octant_labels.append(hand_labels[0] if len(hand_labels) == 1 else "|".join(hand_labels))
        octant_details.append(detail)
        dist = np.linalg.norm(palms - entity_pos[:, None, :], axis=2).min(axis=1)
        regrasp_counts.append(retreat_return_count(dist, onset))

    strategy: dict = {
        "onset_step": {
            "mean": float(np.mean(onset_steps)),
            "median": float(np.median(onset_steps)),
            "max": int(np.max(onset_steps)),
        },
        "motion": {
            key: {
                "mean": float(np.mean([m[key] for m in motions])),
                "std": float(np.std([m[key] for m in motions])),
                "min": float(np.min([m[key] for m in motions])),
                "max": float(np.max([m[key] for m in motions])),
            }
            for key in motions[0]
        },
    }
    if octant_labels:
        counts: dict[str, int] = {}
        for label in octant_labels:
            counts[label] = counts.get(label, 0) + 1
        strategy["approach_octant"] = entropy_perplexity(counts)
        strategy["approach_octant"]["basis"] = (
            "wrist-chain origin from virtual translation joints; palm-surface offset"
            " not included (exact palm poses come from the replay pass)"
        )
        all_az = [d[h]["azimuth_deg"] for d in octant_details for h in d]
        all_elev = [d[h]["elevation_deg"] for d in octant_details for h in d]
        strategy["approach_octant"]["azimuth_circ_std_deg"] = _circular_std_deg(np.array(all_az))
        strategy["approach_octant"]["elevation_mean_deg"] = float(np.mean(all_elev))
        strategy["approach_octant"]["per_episode"] = octant_details
        strategy["approach_octant"]["saturation"] = saturation_curve(
            octant_labels, n_permutations=n_permutations, seed=seed
        )
        strategy["regrasp_proxy"] = {
            "mean": float(np.mean(regrasp_counts)),
            "nonzero_episodes": int(np.sum(np.array(regrasp_counts) > 0)),
            "max": int(np.max(regrasp_counts)),
        }

    # --- conditional multimodality -----------------------------------------
    if include_knn:
        knn = knn_conditional_multimodality(
            demos, k=knn_k, max_pool=knn_max_pool, max_queries=knn_max_queries, seed=seed
        )
    else:
        knn = {"skipped": "include_knn=False"}

    return {
        "task": demos.task,
        "category": demos.category,
        "robot_type": demos.robot_type,
        "n_episodes": demos.num_episodes,
        "manipulated_entity": {"group": group, "name": name},
        "palm_model": {"kind": palm_model.kind, "hand_labels": list(palm_model.hand_labels)},
        "initial_state_coverage": coverage,
        "strategy": strategy,
        "conditional_multimodality": knn,
        "warnings": warnings_list,
        "params": {
            "xy_bin_m": xy_bin_m,
            "yaw_bin_deg": yaw_bin_deg,
            "azimuth_bins": azimuth_bins,
            "knn_k": knn_k,
            "n_permutations": n_permutations,
            "seed": seed,
            "include_failed": include_failed,
        },
    }
