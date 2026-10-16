# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A teleop "device" that replays the actions of a recorded DexVerse trajectory pickle.

``record_demos.py --replay_demos <session.pkl>`` drives the normal recording loop with it instead of a VR
headset: for every episode in the pickle it restores the recorded initial scene state, presses START and
feeds the recorded actions one per step (then holds the last one). Uses:

* validating the recording pipeline (pickle + ``--lerobot_root``) end to end without a headset;
* re-exporting existing pickles (e.g. the released DexVerse demos) through the live LeRobot recorder.

The recorder still decides success with the task's own success term, exactly as for a human demo. Actions are
replayed open-loop from the recorded initial state, so a demo is reproduced only as far as the physics is.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable

import numpy as np
import torch


WRIST_CAMERA_BODY_COLLISION_KEY = "wrist_camera_body_collision"
"""Trajectory pickle metadata key: whether the wrist camera body box collided while the pickle was recorded."""

_WRIST_CAMERA_BODY_ATTRS = ("wrist_camera_body", "right_wrist_camera_body", "left_wrist_camera_body")


def recorded_wrist_camera_body_collision(metadata: dict, robot_type: str | None = None) -> bool:
    """The wrist camera body collision a trajectory pickle has to be replayed with.

    Pickles written since 10/2026 store it under ``wrist_camera_body_collision``. Older pickles were all recorded
    with the box colliding; the Shadow ones (the released DexVerse demos and Shadow VR sessions) are replayed with
    it so they reproduce, while older pickles of the other hands replay without it, like new recordings.

    Args:
        metadata: Top-level pickle payload (the ``episodes`` entry is not needed).
        robot_type: Robot variant used when the pickle does not name one (the env's ``robot_type``).
    """
    if WRIST_CAMERA_BODY_COLLISION_KEY in metadata:
        return bool(metadata[WRIST_CAMERA_BODY_COLLISION_KEY])
    return str(metadata.get("robot_type") or robot_type or "").startswith("floating_shadow")


def _wrist_camera_body_collision_props(env_cfg) -> list:
    props = []
    for name in _WRIST_CAMERA_BODY_ATTRS:
        body = getattr(env_cfg.scene, name, None)
        prop = getattr(getattr(body, "spawn", None), "collision_props", None)
        if prop is not None:
            props.append(prop)
    return props


def set_wrist_camera_body_collision(env_cfg, enabled: bool) -> int:
    """Turn the wrist camera body box collision on / off (single-hand and bimanual); returns the boxes changed."""
    props = _wrist_camera_body_collision_props(env_cfg)
    for prop in props:
        prop.collision_enabled = bool(enabled)
    return len(props)


def wrist_camera_body_collision(env_cfg) -> bool | None:
    """Whether the env cfg's wrist camera body box collides (None when the scene has no box)."""
    props = _wrist_camera_body_collision_props(env_cfg)
    return any(bool(prop.collision_enabled) for prop in props) if props else None


def _to_torch(data, device):
    if isinstance(data, dict):
        return {k: _to_torch(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_to_torch(v, device) for v in data)
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data, device=device)
    return data


class ReplayDemoDevice:
    """Feeds the actions of a trajectory pickle into ``record_demos.py``'s loop."""

    def __init__(self, env, pickle_path: str, *, hold_steps: int = 120, max_episodes: int | None = None):
        with open(pickle_path, "rb") as f:
            payload = pickle.load(f)
        self.env = env
        self.source_path = os.path.abspath(pickle_path)
        self.episodes = [ep for ep in payload.get("episodes", []) if len(ep.get("actions", [])) > 0]
        if max_episodes is not None:
            self.episodes = self.episodes[: int(max_episodes)]
        if not self.episodes:
            raise ValueError(f"{pickle_path} has no episodes with actions")
        self.source = {k: payload.get(k) for k in ("task", "robot_type", "seed", "sim_device")}
        self.hold_steps = int(hold_steps)
        self._callbacks: dict[str, Callable] = {}
        self._ep_idx = -1
        self._t = 0
        self._started = False
        self._reset_requested = False
        self.finished = False

    def __str__(self) -> str:
        return f"ReplayDemoDevice({len(self.episodes)} episodes from task {self.source.get('task')})"

    def add_callback(self, key: str, func: Callable) -> None:
        self._callbacks[key] = func

    @property
    def current_episode(self) -> dict | None:
        return self.episodes[self._ep_idx] if 0 <= self._ep_idx < len(self.episodes) else None

    def reset(self) -> None:
        """Called by record_demos right after every env reset: load the next episode's initial state."""
        self._ep_idx += 1
        self._t = 0
        self._started = False
        self._reset_requested = False
        ep = self.current_episode
        if ep is None:
            self.finished = True
            return
        state = _to_torch(ep["initial_state"], self.env.device)
        self.env.reset_to(state, env_ids=None, is_relative=True)
        # rebase terms that capture a reference at reset (e.g. OpenLaptop base_displaced) on the restored scene
        from .episode_task_state import reset_managers_after_state_restore  # noqa: PLC0415

        reset_managers_after_state_restore(self.env)
        print(f"[replay] episode {self._ep_idx + 1}/{len(self.episodes)} "
              f"(source episode {ep.get('episode_index')}, {len(ep['actions'])} actions)")

    def advance(self) -> torch.Tensor:
        ep = self.current_episode
        if ep is None:
            self.finished = True
            return torch.zeros(self.env.action_manager.total_action_dim, device=self.env.device)
        if not self._started:
            self._started = True
            if "START" in self._callbacks:
                self._callbacks["START"]()
        actions = ep["actions"]
        n = len(actions)
        if self._t >= n + self.hold_steps and not self._reset_requested:
            # the demo did not reach the success condition while holding its last action: give up on it
            self._reset_requested = True
            print(f"[replay] episode {self._ep_idx + 1}: no success after {n} actions + {self.hold_steps} hold steps")
            if "RESET" in self._callbacks:
                self._callbacks["RESET"]()
        a = np.asarray(actions[min(self._t, n - 1)], dtype=np.float32).reshape(-1)
        self._t += 1
        return torch.as_tensor(a, device=self.env.device)
