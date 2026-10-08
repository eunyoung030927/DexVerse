# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Mimic annotations computed from the per-step states a trajectory pickle already records (no simulator).

record_demos keeps the full scene state before every step (``states[t]``, ``states[0]`` = initial state), so the
datagen info the replay annotator would record can be computed directly:

* palm pose = wrist model (from the kinematics cache) at the recorded wrist joint positions, in the robot root
  pose of that step,
* target palm pose = wrist model at the wrist joint targets of ``actions[t]``,
* object poses = the recorded rigid-object root poses,
* ``object_lifted`` = object height above its spawn height > the lift threshold.

The HDF5 layout is the one Isaac Lab's ``HDF5DatasetFileHandler`` writes and ``DataGenInfoPool`` reads.
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .wrist_model import WristModel


def has_states(ep: dict) -> bool:
    """True when the episode records a state before every action (``len(states) == len(actions) + 1``)."""
    states, actions = ep.get("states"), ep.get("actions")
    return isinstance(states, list) and actions is not None and len(states) == len(actions) + 1


def _pose_matrices(root_pose: np.ndarray) -> np.ndarray:
    """(T, 7) position + quaternion (w, x, y, z) -> (T, 4, 4)."""
    m = np.tile(np.eye(4), (len(root_pose), 1, 1))
    m[:, :3, :3] = Rotation.from_quat(root_pose[:, [4, 5, 6, 3]]).as_matrix()
    m[:, :3, 3] = root_pose[:, :3]
    return m


def _stack(states: list) -> dict | np.ndarray:
    """List of per-step nested state dicts with (1, k) leaves -> nested dict of (T, k) arrays."""
    first = states[0]
    if isinstance(first, dict):
        return {k: _stack([s[k] for s in states]) for k in first}
    return np.concatenate([np.asarray(s, dtype=np.float32).reshape(1, -1) for s in states], axis=0)


def _palm_poses(model: WristModel, q_t: np.ndarray, q_r: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
    """Palm poses (T, 4, 4) relative to the env origin for wrist values (T, 3) + (T, 3) and robot root poses (T, 7)."""
    rw = Rotation.from_euler(model.seq, q_r * model.sign)
    pos_b = model.c + q_t @ model.A.T + rw.apply(model.d)
    rot_b = rw * model.R0
    root_r = Rotation.from_quat(root_pose[:, [4, 5, 6, 3]])
    m = np.tile(np.eye(4), (len(q_t), 1, 1))
    m[:, :3, :3] = (root_r * rot_b).as_matrix()
    m[:, :3, 3] = root_r.apply(pos_b) + root_pose[:, :3]
    return m


def _joint_values(cols: list, values: np.ndarray, from_actions: bool) -> np.ndarray:
    if from_actions:
        return np.stack([values[:, j["col"]] * j["scale"] + j["offset"] for j in cols], axis=1)
    return np.stack([values[:, j["joint_id"]] for j in cols], axis=1)


def annotate_episode(ep: dict, cache: dict, subtasks: int) -> tuple[dict | None, str]:
    """Datagen info + Isaac Lab episode data for one pickle episode, or ``(None, reason)`` if it cannot be used."""
    if not ep.get("success", True):
        return None, "episode is not marked successful"
    actions = np.asarray(ep["actions"], dtype=np.float32)
    if actions.ndim != 2 or len(actions) == 0:
        return None, "no actions"
    if actions.shape[1] != cache["action_dim"]:
        return None, f"action dim {actions.shape[1]} != {cache['action_dim']} of the kinematics cache"
    T = len(actions)
    states = _stack(ep["states"][:T])                        # state before each action
    robot = states["articulation"]["robot"]
    if robot["joint_position"].shape[1] != len(cache["joint_names"]):
        return None, "joint count differs from the kinematics cache"
    root = robot["root_pose"].astype(np.float64)
    q = robot["joint_position"].astype(np.float64)
    a = actions.astype(np.float64)

    eef, target = {}, {}
    for side, h in cache["hands"].items():
        model = WristModel.from_dict(h["model"])
        eef[side] = _palm_poses(model, _joint_values(h["trans"], q, False), _joint_values(h["rot"], q, False), root)
        target[side] = _palm_poses(model, _joint_values(h["trans"], a, True), _joint_values(h["rot"], a, True), root)
    rigid = dict(states.get("rigid_object", {}))
    # non-robot articulations (e.g. the laptop of OpenLaptop) are object frames too (see get_object_poses)
    rigid.update({n: s for n, s in states.get("articulation", {}).items() if n != "robot" and n not in rigid})
    missing = [n for n in cache["object_names"] if n not in rigid]
    if missing:
        return None, f"states lack rigid objects {missing}"
    objects = {n: _pose_matrices(rigid[n]["root_pose"].astype(np.float64)) for n in cache["object_names"]}
    lift = cache["lift"]
    if lift is None or "object" not in rigid:
        if subtasks == 2:
            return None, "no lifted object in this task: use --subtasks 1"
        lifted = np.zeros(T, dtype=bool)
    else:
        lifted = (rigid["object"]["root_pose"][:, 2] - lift["object_default_z"]) > lift["height"]
    if subtasks == 2 and (not lifted.any() or lifted[0]):
        return None, "object_lifted never switches 0 -> 1"

    datagen_info = {
        "eef_pose": {k: v.astype(np.float32) for k, v in eef.items()},
        "object_pose": {k: v.astype(np.float32) for k, v in objects.items()},
        "target_eef_pose": {k: v.astype(np.float32) for k, v in target.items()},
        "subtask_term_signals": {"object_lifted": lifted.astype(bool)},
    }
    data = {
        "actions": actions,
        "initial_state": _stack([ep["initial_state"]]),
        "states": states,
        "obs": {"actions": actions, "datagen_info": datagen_info},
    }
    return data, ""


def _write_group(group: h5py.Group, data: dict):
    for k, v in data.items():
        if isinstance(v, dict):
            _write_group(group.create_group(k), v)
        else:
            group.create_dataset(k, data=np.asarray(v))


def write_hdf5(path: str, task: str, demos: list[dict]):
    """Write episodes in the Isaac Lab HDF5 dataset layout (``data/demo_i``, ``env_args``, ``total``)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("data")
        g.attrs["env_args"] = json.dumps({"env_name": task.split(":")[-1], "type": 2})
        total = 0
        for i, data in enumerate(demos):
            d = g.create_group(f"demo_{i}")
            n = int(len(data["actions"]))
            d.attrs["num_samples"] = n
            d.attrs["success"] = True
            _write_group(d, data)
            total += n
        g.attrs["total"] = total
