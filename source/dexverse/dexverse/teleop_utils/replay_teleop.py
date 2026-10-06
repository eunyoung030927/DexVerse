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

import pickle
from collections.abc import Callable

import numpy as np
import torch


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
