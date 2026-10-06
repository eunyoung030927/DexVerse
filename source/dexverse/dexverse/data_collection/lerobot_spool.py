# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a LeRobot v3 dataset LIVE from ``record_demos.py`` (``--lerobot_root``) without blocking the sim.

Same on-disk SPOOL format and the same LeRobot layout as the isaac-tasks collectors (UR7e + RH5DG2), so the
datasets load with the same training code: ``observation.images.third_person``,
``observation.images.eye_in_hand``, ``observation.state``, ``action`` (+ ``meta/isaac_tasks.json`` and
``meta/isaac_tasks_episodes.jsonl``).

TWO HALVES, ONE FILE FORMAT:

* :class:`LeRobotSpoolRecorder` runs inside ``record_demos.py`` (post-app). Every recorded control step it
  captures the third-person and wrist camera RGB, the full joint state and the action BEFORE ``env.step``
  applies the action (LeRobot convention: frame t = ``(o_t, a_t)``), streams the frames to
  ``ep_XXXXXX.tmp/`` as raw uint8 and, when the demo succeeds, atomically renames it to ``ep_XXXXXX/``.
  A failed / reset / abandoned demo is deleted -- its frames never reach the dataset.
* ``scripts/data_tools/lerobot_spool_writer.py`` is a separate CPU-only process (its own Python env with
  ``lerobot``; Isaac Sim's interpreter is never touched) that consumes the ready episodes with
  ``LeRobotDataset.add_frame`` / ``save_episode`` (AV1 videos), deletes each consumed entry and finalizes
  on the ``DONE`` sentinel. The recorder launches it; it also runs standalone on a leftover spool.

SPOOL LAYOUT (``<lerobot_root>.spool/``)::

    spool.json                 dataset-level contract: repo_id, root, fps, features, task, cameras
    last_seq                   last sequence number handed out (kept monotonic across consumed entries)
    ep_000007.tmp/             a demo being recorded (deleted if it does not succeed)
    ep_000006/                 a READY episode
        third_person.u8, eye_in_hand.u8   raw uint8 frames, (T, H, W, 3) row-major
        data.npz               keep_idx (T,) + one (T, ...) array per non-video feature, keyed by feature name
        meta.json              per-episode metadata
    DONE                       the recorder has finished: finalize once the queue is empty
    writer.lock / writer.log   the writer's exclusive lock and its log

STATE. ``observation.state`` holds every robot joint position, ordered like ``action``: first the joints the
action drives, in action-column order (so ``state[:len(action)]`` lines up with ``action`` name by name), then
the remaining (mimic / passive) joints in sim order.

ACTIONS. ``action`` is exactly what ``env.step`` consumed (the task's absolute joint-position action terms:
wrist virtual joints relative to the home pose + finger joint targets), so replaying the rows open-loop
reproduces the demo. Four end-effector versions are stored next to it (floating hands with a retargeter layout).
Each has the layout of ``action`` with every hand's six wrist columns replaced, in place, by that hand's wrist
block; the finger columns are the absolute finger targets of ``action`` in all of them:

* ``action.ee_abs``         ``[x, y, z, rx, ry, rz]``: absolute commanded wrist pose, rotation as a rotation
                            vector (rad), in the hand's base frame.
* ``action.ee_abs_rot6d``   ``[x, y, z, r6d_0..r6d_5]``: same pose with the 6D rotation (first two COLUMNS of the
                            rotation matrix, ``[R[:,0], R[:,1]]``, Zhou et al. 2019).
* ``action.ee_delta``       ``[dx, dy, dz, drx, dry, drz]``: commanded motion since the PREVIOUS step,
                            ``p_t - p_{t-1}`` and ``R_t R_{t-1}^T`` as a rotation vector (first step relative to the
                            home pose), so composition recovers ``ee_abs``.
* ``action.ee_delta_init``  ``[dx, dy, dz, drx, dry, drz]``: commanded pose relative to the MEASURED wrist pose at
                            the episode's first frame, ``p_t - p_0`` and ``R_t R_0^T`` -- "absolute delta" from the
                            start, which a policy can apply to the pose it observed when the episode began.

The matching measured wrist poses are ``observation.state.ee`` / ``observation.state.ee_rot6d`` (same layouts,
finger columns = measured finger joint positions). Poses come from the virtual-joint values (targets for the
actions, joint positions for the state) by the product of exponentials over joint axes MEASURED in the simulator
at startup, so no per-hand Euler convention is assumed; ``meta.json`` reports how far the measured palm rotation
is from the command (``ee_rot_check_deg``).

IMPORT NOTE. The module body is stdlib + numpy only: the writer loads it BY PATH outside Isaac Sim.
Everything that needs torch / Isaac Lab / scipy is imported inside :class:`LeRobotSpoolRecorder`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

# ----- the spool file format (shared by the recorder and lerobot_spool_writer.py) -----------------
SPOOL_FORMAT = "dexverse lerobot spool v1"
SPOOL_INFO = "spool.json"
DONE_SENTINEL = "DONE"
WRITER_LOCK = "writer.lock"
WRITER_LOG = "writer.log"
DATA_FILE = "data.npz"
META_FILE = "meta.json"
WRITTEN_MARK = "WRITTEN"
SEQ_FILE = "last_seq"
TMP_SUFFIX = ".tmp"
_READY_RE = re.compile(r"^ep_(\d{6})$")
_ANY_RE = re.compile(r"^ep_(\d{6})(\.tmp)?$")

# LeRobot feature key -> raw frame file. Keys match the isaac-tasks / DexSteer datasets.
THIRD_PERSON_KEY = "observation.images.third_person"
EYE_IN_HAND_KEY = "observation.images.eye_in_hand"
_POSE6 = ("ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz")
_POSE9 = ("ee_x", "ee_y", "ee_z") + tuple(f"ee_r6d_{i}" for i in range(6))
_DELTA6 = ("ee_dx", "ee_dy", "ee_dz", "ee_drx", "ee_dry", "ee_drz")
# feature key -> names of one hand's wrist block
EE_ACTION_BLOCKS = {
    "action.ee_abs": _POSE6,
    "action.ee_abs_rot6d": _POSE9,
    "action.ee_delta": _DELTA6,
    "action.ee_delta_init": _DELTA6,
}
EE_STATE_BLOCKS = {
    "observation.state.ee": _POSE6,
    "observation.state.ee_rot6d": _POSE9,
}

# Datasets go to local disk: CIFS / NFS mounts corrupt or stall the writer's large sequential writes.
DATASETS_DIR_ENV = "DEXVERSE_LEROBOT_DIR"
_NETWORK_FS = ("cifs", "smb3", "smbfs", "nfs", "nfs4", "fuse.sshfs", "9p")


def default_datasets_dir() -> str:
    """``$DEXVERSE_LEROBOT_DIR``, else ``/workspace/local/datasets`` (n1 host mount) if it exists, else
    ``/root/dexverse_datasets``."""
    env = os.environ.get(DATASETS_DIR_ENV)
    if env:
        return os.path.abspath(env)
    if os.path.isdir("/workspace/local/datasets"):
        return "/workspace/local/datasets"
    return "/root/dexverse_datasets"


def default_dataset_name(task_id: str, robot_type: str | None) -> str:
    """``Dexverse-GraspCup-v0`` + ``floating_shadow_right`` -> ``graspcup-v0-floating_shadow_right``."""
    name = re.sub(r"^Dexverse-", "", task_id.split(":")[-1]).lower()
    return f"{name}-{robot_type or 'default'}"


def filesystem_type(path) -> str:
    """Filesystem type of the mount holding ``path`` (nearest existing parent), from /proc/mounts."""
    p = os.path.abspath(str(path))
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    p = os.path.realpath(p)
    best, fstype = "", "unknown"
    try:
        with open("/proc/mounts", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                if (p == mnt or p.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
                    best, fstype = mnt, parts[2]
    except OSError:
        pass
    return fstype


def spool_dir_for(root) -> str:
    """The spool directory that belongs to a LeRobot dataset root: ``<root>.spool``."""
    return os.path.abspath(str(root)).rstrip("/") + ".spool"


def list_ready(spool) -> list[str]:
    """Ready (fully written) episode directories, oldest first."""
    if not os.path.isdir(spool):
        return []
    out = []
    for name in sorted(os.listdir(spool)):
        p = os.path.join(spool, name)
        if _READY_RE.match(name) and os.path.isdir(p) and os.path.exists(os.path.join(p, META_FILE)):
            out.append(p)
    return out


def list_tmp(spool) -> list[str]:
    if not os.path.isdir(spool):
        return []
    return [os.path.join(spool, n) for n in sorted(os.listdir(spool)) if n.startswith("ep_") and n.endswith(TMP_SUFFIX)]


def next_seq(spool) -> int:
    seqs = [int(m.group(1)) for n in (os.listdir(spool) if os.path.isdir(spool) else []) for m in [_ANY_RE.match(n)] if m]
    try:
        with open(os.path.join(spool, SEQ_FILE), encoding="utf-8") as f:
            seqs.append(int(f.read().strip()))
    except (OSError, ValueError):
        pass
    return (max(seqs) + 1) if seqs else 0


class _Indexed:
    """``frames[t]`` -> ``mm[keep[t]]`` without materialising the whole episode."""

    def __init__(self, mm, keep):
        self.mm, self.keep = mm, keep

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, t):
        return self.mm[int(self.keep[t])]


def load_episode(ep_dir) -> dict:
    """Open one ready spool episode: ``{"meta", "arrays": {feature: (T, ...)}, "frames": {key: (T,H,W,3)}}``."""
    with open(os.path.join(ep_dir, META_FILE), encoding="utf-8") as f:
        meta = json.load(f)
    d = np.load(os.path.join(ep_dir, DATA_FILE))
    keep = d["keep_idx"].astype(np.int64)
    arrays = {k: d[k].astype(np.float32) for k in d.files if k != "keep_idx"}
    frames = {}
    n_cap = int(meta["n_captured"])
    for key, fname in meta["frame_files"].items():
        h, w, c = meta["image_shapes"][key]
        mm = np.memmap(os.path.join(ep_dir, fname), dtype=np.uint8, mode="r", shape=(n_cap, h, w, c))
        frames[key] = _Indexed(mm, keep)
    return {"meta": meta, "arrays": arrays, "frames": frames, "num_frames": int(len(keep))}


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v


def _write_json_atomic(path, obj):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def default_task_string(task_id: str) -> str:
    """Language instruction for a DexVerse task id (DexVerse ships none): ``Dexverse-PickCube-v0`` ->
    ``"Pick cube"``. Override with ``--lerobot_task`` for anything that needs a better sentence."""
    name = task_id.split(":")[-1]
    name = re.sub(r"^Dexverse-", "", name)
    name = re.sub(r"-v\d+$", "", name).replace("LongHorizon-", "")
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", name)
    text = " ".join(words).lower()
    return text[:1].upper() + text[1:] if text else task_id


def _wrist_targets(actions: np.ndarray, joints: list[dict]) -> np.ndarray:
    """Joint targets the env commands for these action entries: ``a * scale + offset`` (T, n)."""
    a = np.asarray(actions, dtype=np.float64)
    return np.stack([a[:, j["idx"]] * j["scale"] + j["offset"] for j in joints], axis=1)


def wrist_command_poses(actions: np.ndarray, hand: dict):
    """Commanded wrist pose per step in the hand's base frame from the virtual-joint targets, by the product of
    exponentials with the joint axes MEASURED in the simulator (``hand["trans"]`` / ``hand["rot"]`` entries:
    ``idx`` action column, ``scale``, ``offset``, ``axis`` in the base frame at the chain's zero configuration;
    ``rot`` in chain order from the base). Returns ``(p (T,3), R (scipy Rotation, T), p_home, R_home)``; home is
    the zero action (the task's default wrist pose)."""
    p, rot = _chain_pose(_wrist_targets(actions, hand["trans"]), _wrist_targets(actions, hand["rot"]), hand)
    zero = np.zeros((1, len(np.asarray(actions)[0])))
    p_home, rot_home = _chain_pose(_wrist_targets(zero, hand["trans"]), _wrist_targets(zero, hand["rot"]), hand)
    return p, rot, p_home[0], rot_home[0]


def _chain_pose(q_t, q_r, hand):
    """Wrist pose of the virtual-joint values ``q_t`` (T, n_trans) / ``q_r`` (T, n_rot): translate along the
    measured prismatic axes, then rotate through the revolute chain (product of exponentials)."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415  (scipy ships with Isaac Sim)

    p = np.asarray(q_t, dtype=np.float64) @ np.asarray([j["axis"] for j in hand["trans"]], dtype=np.float64)
    q_r = np.asarray(q_r, dtype=np.float64)
    rot = Rotation.identity(len(q_r))
    for k, j in enumerate(hand["rot"]):
        rot = rot * Rotation.from_rotvec(np.outer(q_r[:, k], np.asarray(j["axis"], dtype=np.float64)))
    return p, rot


def wrist_state_poses(joint_pos: np.ndarray, hand: dict):
    """Measured wrist pose per frame from the virtual-joint POSITIONS (``joint_pos`` in sim joint order)."""
    q = np.asarray(joint_pos, dtype=np.float64)
    return _chain_pose(q[:, [j["joint_id"] for j in hand["trans"]]], q[:, [j["joint_id"] for j in hand["rot"]]], hand)


def rot6d(rot) -> np.ndarray:
    """6D rotation ``[R[:,0], R[:,1]]`` (first two columns of the matrix) per row, (T, 6)."""
    m = rot.as_matrix()
    return m[:, :, :2].transpose(0, 2, 1).reshape(len(m), 6)


def _ee_layout_items(n_cols: int, hands: list[dict]):
    """Columns of an EE vector: ``("wrist", hand_i)`` at the position of the hand's first wrist action column,
    ``("col", c)`` for every non-wrist (finger) action column, in action order."""
    first = {min(j["idx"] for j in h["trans"] + h["rot"]): i for i, h in enumerate(hands)}
    wrist_cols = {j["idx"] for h in hands for j in h["trans"] + h["rot"]}
    return [("wrist", first[c]) if c in first else ("col", c) for c in range(n_cols) if c in first or c not in wrist_cols]


def ee_names(action_names: list[str], hands: list[dict], block: tuple) -> list[str]:
    """Names of an EE vector whose wrist blocks are named ``block`` (``right_``/``left_`` prefix if bimanual)."""
    out = []
    for kind, i in _ee_layout_items(len(action_names), hands):
        if kind == "wrist":
            prefix = "" if len(hands) == 1 else f"{hands[i]['side']}_"
            out += [prefix + n for n in block]
        else:
            out.append(action_names[i])
    return out


def _assemble(hands, items, blocks, fingers):
    """Concatenate wrist blocks (per hand, (T, k)) and finger columns (``fingers(c)`` -> (T,)) per layout."""
    cols = [blocks[i] if kind == "wrist" else np.asarray(fingers(i), dtype=np.float64)[:, None] for kind, i in items]
    return np.hstack(cols).astype(np.float32)


def ee_action_columns(actions: np.ndarray, joint_pos: np.ndarray, hands: list[dict]) -> dict:
    """The four EE action versions (see the module docstring) of one episode: ``{feature key: (T, n)}``.
    ``actions`` (T, n_action) as consumed by env.step, ``joint_pos`` (T, n_joints) in sim order (frame 0 is the
    measured start pose for ``ee_delta_init``)."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    a = np.asarray(actions, dtype=np.float64)
    items = _ee_layout_items(a.shape[1], hands)
    blocks = {k: [] for k in EE_ACTION_BLOCKS}
    for h in hands:
        p, rot, p_home, rot_home = wrist_command_poses(a, h)
        p_prev = np.vstack([p_home[None], p[:-1]])
        home = Rotation.from_quat(rot_home.as_quat()[None])
        rot_prev = Rotation.concatenate([home, rot[:-1]]) if len(a) > 1 else home
        p0, r0 = wrist_state_poses(np.asarray(joint_pos)[:1], h)
        blocks["action.ee_abs"].append(np.hstack([p, rot.as_rotvec()]))
        blocks["action.ee_abs_rot6d"].append(np.hstack([p, rot6d(rot)]))
        blocks["action.ee_delta"].append(np.hstack([p - p_prev, (rot * rot_prev.inv()).as_rotvec()]))
        blocks["action.ee_delta_init"].append(np.hstack([p - p0[0], (rot * r0[0].inv()).as_rotvec()]))
    return {k: _assemble(hands, items, b, lambda c: a[:, c]) for k, b in blocks.items()}


def ee_state_columns(joint_pos: np.ndarray, hands: list[dict], action_joint_ids: list[int]) -> dict:
    """Measured wrist poses in the layouts of ``action.ee_abs`` / ``action.ee_abs_rot6d``: ``{key: (T, n)}``;
    finger columns are the measured positions of the joints the finger action columns drive."""
    q = np.asarray(joint_pos, dtype=np.float64)
    items = _ee_layout_items(len(action_joint_ids), hands)
    blocks = {k: [] for k in EE_STATE_BLOCKS}
    for h in hands:
        p, rot = wrist_state_poses(q, h)
        blocks["observation.state.ee"].append(np.hstack([p, rot.as_rotvec()]))
        blocks["observation.state.ee_rot6d"].append(np.hstack([p, rot6d(rot)]))
    return {k: _assemble(hands, items, b, lambda c: q[:, action_joint_ids[c]]) for k, b in blocks.items()}


# =====================================================================================================
# RECORDER SIDE (inside record_demos.py, after the app has started)
# =====================================================================================================


class LeRobotSpoolRecorder:
    """Live LeRobot recording for ``record_demos.py``.

    Lifecycle (driven by the trajectory recorder): :meth:`begin_episode` -> :meth:`capture` once per control
    step BEFORE ``env.step`` -> :meth:`commit_episode` on success or :meth:`discard_episode` otherwise ->
    :meth:`close` at the end of the session.
    """

    def __init__(self, env, env_cfg, *, task_id: str, root: str, repo_id: str | None = None,
                 task: str | None = None, max_pending: int = 4, launch_writer: bool = True,
                 writer_python: str | None = None, image_threads: int = 8):
        self.env = env
        self.task_id = task_id
        self.root = os.path.abspath(root)
        self.spool = spool_dir_for(self.root)
        fstype = filesystem_type(self.root)
        if fstype in _NETWORK_FS:
            raise RuntimeError(f"[lerobot] {self.root} is on a {fstype} network mount; record LeRobot datasets to "
                               f"local disk (--lerobot_root or ${DATASETS_DIR_ENV}).")
        self.robot_type = str(getattr(env_cfg, "robot_type", "") or "unknown")
        self.repo_id = repo_id or f"local/dexverse-{re.sub(r'^Dexverse-', '', task_id).lower()}-{self.robot_type}"
        self.task = task or default_task_string(task_id)
        self.max_pending = int(max_pending)
        self.image_threads = int(image_threads)
        self.writer_python = writer_python
        self.robot = env.scene["robot"]

        step_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        fps = 1.0 / step_dt
        if abs(fps - round(fps)) > 1e-6:
            raise ValueError(f"LeRobot needs an integer fps; this task steps at {fps} Hz.")
        self.fps = int(round(fps))

        self.cameras = self._resolve_cameras(env_cfg)
        self.image_shapes = {}
        for key, cam_name, _ in self.cameras:
            cam_cfg = getattr(env_cfg.scene, cam_name)
            self.image_shapes[key] = [int(cam_cfg.height), int(cam_cfg.width), 3]

        self.action_names = self._action_names()
        self.action_joint_ids = [e[0] for e in self._action_entries()]
        # observation.state: the action's joints in action order, then the remaining joints in sim order
        rest = [i for i in range(self.robot.num_joints) if i not in set(self.action_joint_ids)]
        if len(set(self.action_joint_ids)) != len(self.action_joint_ids):
            raise RuntimeError("[lerobot] two action columns drive the same joint; cannot align state with action")
        self.state_order = self.action_joint_ids + rest
        self.state_names = [self.robot.joint_names[i] for i in self.state_order]
        self.ee_hands = self._ee_layout()
        self.palm_ids, self.palm_names = self._palm_bodies(env_cfg)
        if self.ee_hands:
            if len(self.palm_ids) != len(self.ee_hands):
                raise RuntimeError("[lerobot] could not resolve the palm body of every hand (robot_config.*palm_body_name)")
            for h in self.ee_hands:
                if max(j["joint_id"] for j in h["trans"]) > min(j["joint_id"] for j in h["rot"]):
                    raise RuntimeError(f"[lerobot] {h['side']} wrist: translation joints are not all before the "
                                       "rotation joints in the chain; ee_delta assumes translate-then-rotate")
            self._measure_wrist_axes()

        self._features = self._build_features()
        os.makedirs(self.spool, exist_ok=True)
        self._check_or_write_spool_info()
        # A DONE left by the previous session into the same root would make this session's writer stop early.
        if os.path.exists(os.path.join(self.spool, DONE_SENTINEL)):
            os.remove(os.path.join(self.spool, DONE_SENTINEL))
        self._writer = None
        if launch_writer:
            self._launch_writer()

        self._ep = None
        self._needs_render = True
        self.n_committed = 0
        print(f"[lerobot] recording {self.task_id} ({self.robot_type}) at {self.fps} fps -> {self.root} "
              f"(repo_id {self.repo_id}, task {self.task!r}); spool {self.spool}")

    # ----- static description ------------------------------------------------------------------------
    def _resolve_cameras(self, env_cfg):
        scene = env_cfg.scene
        third = "third_person_camera"
        wrist = next((n for n in ("wrist_camera", "right_wrist_camera", "left_wrist_camera")
                      if getattr(scene, n, None) is not None), None)
        if getattr(scene, third, None) is None or wrist is None:
            raise RuntimeError(
                "LeRobot recording needs scene.third_person_camera and a wrist camera; got "
                f"third_person={getattr(scene, third, None) is not None}, wrist={wrist}. "
                "Run with cameras enabled (record_demos does this when --lerobot_root is set).")
        return [(THIRD_PERSON_KEY, third, "third_person.u8"), (EYE_IN_HAND_KEY, wrist, "eye_in_hand.u8")]

    def _action_names(self):
        names = []
        manager = self.env.action_manager
        for term_name in manager.active_terms:
            term = manager.get_term(term_name)
            joint_names = list(getattr(term, "_joint_names", []) or [])
            if len(joint_names) != int(term.action_dim):
                joint_names = [f"{term_name}_{i}" for i in range(int(term.action_dim))]
            names += joint_names
        if len(names) != int(manager.total_action_dim):
            raise RuntimeError(f"action names {len(names)} != action dim {manager.total_action_dim}")
        return names

    def _action_entries(self):
        """Per action column: ``(joint_id, scale, offset)`` of the joint-position term that consumes it."""
        entries = []
        manager = self.env.action_manager
        for term_name in manager.active_terms:
            term = manager.get_term(term_name)
            ids = term._joint_ids
            ids = list(range(self.robot.num_joints))[ids] if isinstance(ids, slice) else list(ids)
            for k in range(int(term.action_dim)):
                scale = term._scale if isinstance(term._scale, float) else float(term._scale[0, k])
                offset = term._offset if isinstance(term._offset, float) else float(term._offset[0, k])
                entries.append((int(ids[k]), float(scale), float(offset)))
        return entries

    def _ee_layout(self):
        """Wrist virtual-joint columns per hand (from the retargeter layout), with their joint ids/scale/offset."""
        from importlib import import_module  # noqa: PLC0415

        from dexverse.devices.retargeters.simple_relative_retargeting import (  # noqa: PLC0415
            SIMPLE_RETARGETER_LAYOUT_SOURCES,
        )

        if self.robot_type not in SIMPLE_RETARGETER_LAYOUT_SOURCES:
            return []
        module_name, attr = SIMPLE_RETARGETER_LAYOUT_SOURCES[self.robot_type]
        layout = getattr(import_module(module_name), attr)
        entries = self._action_entries()
        hands = []
        for side, h in layout["hands"].items():
            if h.get("wrist_rot_repr", "euler") != "euler":
                continue

            def cols(idx_list):
                return [{"idx": int(i), "name": self.action_names[i], "joint_id": entries[i][0],
                         "scale": entries[i][1], "offset": entries[i][2]} for i in idx_list]

            rot = sorted(cols(h["wrist_rot_indices"]), key=lambda j: j["joint_id"])  # sim (BFS) order = chain order
            hands.append({"side": side, "trans": cols(h["wrist_trans_indices"]), "rot": rot})
        return hands

    def _palm_bodies(self, env_cfg):
        """Palm body id per entry of ``self.ee_hands`` (wrist-pose checks / axis measurement)."""
        rc = getattr(env_cfg, "robot_config", None)
        ids, names = [], []
        for h in self.ee_hands:
            name = None
            if rc is not None:
                name = (getattr(rc, f"{h['side']}_palm_body_name", None) if len(self.ee_hands) > 1 else None) \
                    or getattr(rc, "palm_body_name", None)
            found = self.robot.find_bodies(name)[0] if name else []
            if not found:
                return [], []
            ids.append(int(found[0]))
            names.append(name)
        return ids, names

    def _measure_wrist_axes(self, eps_rot: float = 0.1, eps_trans: float = 0.05):
        """Measure every virtual wrist joint's axis in the hand base frame, at the chain's zero configuration,
        by moving it alone (kinematics only, no physics step) and reading the palm pose. The joint state is
        restored afterwards; ``record_demos`` resets the env before recording anyway."""
        import torch  # noqa: PLC0415
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        robot, sim = self.robot, self.env.sim
        view = robot.root_physx_view
        q0 = robot.data.joint_pos.clone()
        v0 = robot.data.joint_vel.clone()
        rq = robot.data.root_quat_w[0].detach().to("cpu").numpy().astype(np.float64)
        base = Rotation.from_quat([rq[1], rq[2], rq[3], rq[0]])

        def set_q(q):
            robot.write_joint_state_to_sim(q, torch.zeros_like(v0))
            sim.forward()

        def palm(bid):
            tf = view.get_link_transforms()[0, bid].detach().to("cpu").numpy().astype(np.float64)
            return tf[:3], Rotation.from_quat(tf[3:7])   # PhysX tensors: position + quaternion (x, y, z, w)

        try:
            for h, bid in zip(self.ee_hands, self.palm_ids):
                qz = q0.clone()
                for j in h["trans"] + h["rot"]:
                    qz[0, j["joint_id"]] = 0.0
                set_q(qz)
                p0, r0 = palm(bid)
                for j in h["rot"]:
                    q1 = qz.clone()
                    q1[0, j["joint_id"]] = eps_rot
                    set_q(q1)
                    _, r1 = palm(bid)
                    j["axis"] = base.inv().apply((r1 * r0.inv()).as_rotvec() / eps_rot).tolist()
                for j in h["trans"]:
                    q1 = qz.clone()
                    q1[0, j["joint_id"]] = eps_trans
                    set_q(q1)
                    p1, _ = palm(bid)
                    j["axis"] = base.inv().apply((p1 - p0) / eps_trans).tolist()
        finally:
            robot.write_joint_state_to_sim(q0, v0)
            sim.forward()
        for h in self.ee_hands:
            for j in h["trans"] + h["rot"]:
                norm = float(np.linalg.norm(j["axis"]))
                if abs(norm - 1.0) > 0.05:
                    raise RuntimeError(f"[lerobot] measured axis of {j['name']} has norm {norm:.3f} (expected 1)")

    def _build_features(self):
        feats = {}
        for key, _cam, _f in self.cameras:
            h, w, c = self.image_shapes[key]
            feats[key] = {"dtype": "video", "shape": [c, h, w], "names": ["channel", "height", "width"]}
        feats["observation.state"] = {"dtype": "float32", "shape": [len(self.state_names)], "names": self.state_names}
        feats["action"] = {"dtype": "float32", "shape": [len(self.action_names)], "names": self.action_names}
        if self.ee_hands:
            for key, block in {**EE_STATE_BLOCKS, **EE_ACTION_BLOCKS}.items():
                names = ee_names(self.action_names, self.ee_hands, block)
                feats[key] = {"dtype": "float32", "shape": [len(names)], "names": names}
        return feats

    def _spool_info(self):
        return {
            "format": SPOOL_FORMAT,
            "repo_id": self.repo_id,
            "root": self.root,
            "fps": self.fps,
            "robot_type": f"dexverse_{self.robot_type}",
            "task": self.task,
            "env_task_id": self.task_id,
            "features": self._features,
            "image_shapes": self.image_shapes,
            "camera_mapping": {key: cam for key, cam, _ in self.cameras},
            "frame_files": {key: f for key, _, f in self.cameras},
            "state_definition": "robot.data.joint_pos (rad / m for the virtual wrist prismatics) captured before the "
                                "action of the same frame is applied; the action's joints first in action-column "
                                "order (state[:len(action)] matches action by name), then the other joints",
            "action_definition": "env.step action: the task's absolute joint-position action terms in term order "
                                 "(wrist virtual joints relative to the home pose, finger joint targets)",
            "ee_definitions": {
                "layout": "the layout of `action` with each hand's 6 wrist columns replaced in place by its wrist "
                          "block; finger columns = absolute finger targets (actions) / measured finger joint "
                          "positions (state)",
                "pose": "wrist pose in the hand's base frame from virtual-joint values (targets for actions, "
                        "joint positions for state) by the product of exponentials of the measured axes",
                "action.ee_abs": "[x,y,z,rx,ry,rz] absolute commanded pose, rotation vector (rad)",
                "action.ee_abs_rot6d": "[x,y,z,r6d_0..5] absolute commanded pose, 6D rotation = [R[:,0], R[:,1]]",
                "action.ee_delta": "[dx,dy,dz,drx,dry,drz] p_t - p_{t-1}, rotvec(R_t R_{t-1}^T): since the previous "
                                   "command (first frame: since the home pose)",
                "action.ee_delta_init": "[dx,dy,dz,drx,dry,drz] p_t - p_0, rotvec(R_t R_0^T) relative to the "
                                        "MEASURED wrist pose of the episode's first frame",
                "observation.state.ee": "[x,y,z,rx,ry,rz] measured wrist pose, rotation vector",
                "observation.state.ee_rot6d": "[x,y,z,r6d_0..5] measured wrist pose, 6D rotation",
            } if self.ee_hands else None,
            "ee_hands": self.ee_hands,
            "alignment": "frame t = (observation before action t, action t)",
        }

    def _check_or_write_spool_info(self):
        path = os.path.join(self.spool, SPOOL_INFO)
        info = self._spool_info()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            for k in ("repo_id", "root", "fps", "features"):
                if json.dumps(_jsonable(old.get(k)), sort_keys=True) != json.dumps(_jsonable(info.get(k)), sort_keys=True):
                    raise RuntimeError(f"[lerobot] existing spool {self.spool} has a different {k!r}; use another "
                                       f"--lerobot_root (old {old.get(k)!r}, new {info.get(k)!r})")
        _write_json_atomic(path, info)

    # ----- writer process ----------------------------------------------------------------------------
    def _writer_cmd(self):
        script = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                                               "scripts", "data_tools", "lerobot_spool_writer.py"))
        python = self.writer_python or "/opt/conda/envs/lerobot/bin/python"
        return [python, script, "--spool", self.spool, "--image_threads", str(self.image_threads),
                "--follow", "--parent_pid", str(os.getpid())]

    def _launch_writer(self):
        cmd = self._writer_cmd()
        if not os.path.exists(cmd[0]):
            raise RuntimeError(f"[lerobot] writer python {cmd[0]} not found (create the 'lerobot' conda env, see "
                               "docs/teleop_quest_cloudxr6.md) or pass --lerobot_python / --no_spool_writer")
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX")}
        env.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"})
        log = open(os.path.join(self.spool, WRITER_LOG), "a", encoding="utf-8")
        self._writer = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        print(f"[lerobot] spool writer started (pid {self._writer.pid}); log {os.path.join(self.spool, WRITER_LOG)}")

    def writer_alive(self) -> bool:
        return self._writer is not None and self._writer.poll() is None

    def _wait_for_room(self):
        while len(list_ready(self.spool)) >= self.max_pending:
            if self._writer is not None and not self.writer_alive():
                raise RuntimeError(f"[lerobot] spool writer exited (see {os.path.join(self.spool, WRITER_LOG)}) "
                                   f"with {len(list_ready(self.spool))} episodes pending")
            time.sleep(0.5)

    # ----- per episode -------------------------------------------------------------------------------
    def begin_episode(self, info: dict | None = None):
        if self._ep is not None:
            self.discard_episode("restarted")
        seq = next_seq(self.spool)
        with open(os.path.join(self.spool, SEQ_FILE), "w", encoding="utf-8") as f:
            f.write(str(seq))
        tmp = os.path.join(self.spool, f"ep_{seq:06d}{TMP_SUFFIX}")
        os.makedirs(tmp)
        self._ep = {
            "seq": seq, "dir": tmp, "info": dict(info or {}), "t0": time.time(), "capture_s": 0.0,
            "files": {key: open(os.path.join(tmp, f), "wb", buffering=8 << 20) for key, _c, f in self.cameras},
            "state": [], "action": [], "palm_quat": [], "n": 0,
        }
        self._needs_render = True

    def mark_needs_render(self):
        """Call after anything that moves the scene outside env.step (reset, reset_to)."""
        self._needs_render = True

    def capture(self, action):
        """Record frame t: images + joint state BEFORE ``env.step(action)``, plus ``action``."""
        ep = self._ep
        if ep is None:
            return
        t0 = time.time()
        force = self._needs_render
        if force:
            # After a reset the cameras still hold the pre-reset render; refresh once.
            self.env.sim.render()
            self._needs_render = False
        for key, cam_name, _f in self.cameras:
            cam = self.env.scene[cam_name]
            if force:
                cam.update(0.0, force_recompute=True)
            # otherwise env.step's scene.update marked the camera outdated and .data re-reads the latest render
            rgb = cam.data.output["rgb"][0, ..., :3]
            img = rgb.to("cpu").numpy()
            if img.dtype != np.uint8:
                img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            if list(img.shape) != self.image_shapes[key]:
                raise RuntimeError(f"[lerobot] {cam_name} gave {img.shape}, expected {self.image_shapes[key]}")
            ep["files"][key].write(np.ascontiguousarray(img).tobytes())
        ep["state"].append(self.robot.data.joint_pos[0].detach().to("cpu").numpy().astype(np.float32))
        ep["action"].append(np.asarray(action.detach().to("cpu").numpy() if hasattr(action, "detach") else action,
                                       dtype=np.float32).reshape(-1))
        if self.palm_ids:
            q = self.robot.data.body_link_pose_w[0, self.palm_ids, 3:7].detach().to("cpu").numpy()
            ep["palm_quat"].append(q.astype(np.float64))
        ep["n"] += 1
        ep["capture_s"] += time.time() - t0

    def _ee_rot_check(self, actions):
        """Angle (deg) between the measured palm rotation (relative to the first frame, in the hand base frame)
        and the commanded one decoded from the wrist Euler triple -- a check of the ee_delta convention."""
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        ep = self._ep
        if not self.ee_hands or not ep["palm_quat"] or len(self.palm_ids) < len(self.ee_hands):
            return None
        root_q = self.robot.data.root_quat_w[0].detach().to("cpu").numpy()
        base = Rotation.from_quat([root_q[1], root_q[2], root_q[3], root_q[0]])
        pq = np.asarray(ep["palm_quat"])  # (T, n_palms, 4) wxyz
        res = {}
        for i, h in enumerate(self.ee_hands):
            _p, cmd, _ph, rot_home = wrist_command_poses(actions, h)
            meas = Rotation.from_quat(pq[:, i][:, [1, 2, 3, 0]])
            # Frame 0 is captured right after the reset, i.e. at the home pose: the measured chain rotation is the
            # palm rotation since frame 0 (in the base frame) applied on top of the home command.
            meas_abs = base.inv() * meas * meas[0].inv() * base * rot_home
            # measured state t+1 is the result of command t
            n = len(cmd) - 1
            if n < 1:
                continue
            ang = np.degrees((meas_abs[1:] * cmd[:n].inv()).magnitude())
            res[h["side"]] = {"median": float(np.median(ang)), "p95": float(np.percentile(ang, 95)),
                              "max": float(ang.max())}
        return res

    def commit_episode(self, extra_meta: dict | None = None):
        ep = self._ep
        if ep is None:
            return None
        self._ep = None
        for f in ep["files"].values():
            f.flush()
            os.fsync(f.fileno())
            f.close()
        T = int(ep["n"])
        if T == 0:
            shutil.rmtree(ep["dir"], ignore_errors=True)
            return None
        joint_pos = np.stack(ep["state"])               # sim joint order
        action = np.stack(ep["action"])
        arrays = {"keep_idx": np.arange(T, dtype=np.int64), "observation.state": joint_pos[:, self.state_order],
                  "action": action}
        self._ep = ep                                   # _ee_rot_check reads palm_quat from the active episode
        try:
            ee_check = self._ee_rot_check(action.astype(np.float64)) if self.ee_hands else None
        finally:
            self._ep = None
        if self.ee_hands:
            arrays.update(ee_state_columns(joint_pos, self.ee_hands, self.action_joint_ids))
            arrays.update(ee_action_columns(action, joint_pos, self.ee_hands))
        np.savez(os.path.join(ep["dir"], DATA_FILE), **arrays)
        meta = {
            "task": self.task,
            "env_task_id": self.task_id,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "num_frames": T,
            "n_captured": T,
            "image_shapes": self.image_shapes,
            "frame_files": {key: f for key, _c, f in self.cameras},
            "ee_rot_check_deg": ee_check,
            "timing": {"episode_wall_s": time.time() - ep["t0"], "capture_overhead_s": ep["capture_s"]},
            **ep["info"],
            **(extra_meta or {}),
        }
        _write_json_atomic(os.path.join(ep["dir"], META_FILE), meta)
        final = ep["dir"][: -len(TMP_SUFFIX)]
        os.rename(ep["dir"], final)
        self.n_committed += 1
        print(f"[lerobot] episode spooled: {os.path.basename(final)} ({T} frames, capture "
              f"{ep['capture_s'] / max(T, 1) * 1000:.1f} ms/frame)")
        self._wait_for_room()
        return final

    def discard_episode(self, reason: str = ""):
        ep = self._ep
        if ep is None:
            return
        self._ep = None
        for f in ep["files"].values():
            try:
                f.close()
            except OSError:
                pass
        shutil.rmtree(ep["dir"], ignore_errors=True)
        if ep["n"]:
            print(f"[lerobot] discarded {ep['n']} frames ({reason})")

    def close(self):
        self.discard_episode("session ended")
        with open(os.path.join(self.spool, DONE_SENTINEL), "w", encoding="utf-8") as f:
            f.write(str(time.time()))
        pending = len(list_ready(self.spool))
        if self._writer is not None:
            print(f"[lerobot] {self.n_committed} episode(s) recorded; the writer keeps encoding {pending} pending "
                  f"episode(s) in the background (pid {self._writer.pid}, log {os.path.join(self.spool, WRITER_LOG)}).")
        else:
            print(f"[lerobot] {self.n_committed} episode(s) spooled, {pending} pending. Write them with:\n"
                  f"  /opt/conda/envs/lerobot/bin/python scripts/data_tools/lerobot_spool_writer.py --spool {self.spool}")
