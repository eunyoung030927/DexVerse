# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop controller around a trained diffusion policy.

Receding horizon: the policy samples an ``action_horizon``-long chunk, the runner
plays ``replan_interval`` actions from it, then samples again. Sampling is the
expensive part, so replanning less often trades reactivity for throughput.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch

from .config import PolicyConfig
from .model import DiffusionPolicy
from .normalization import Stats, denormalize, normalize, stats_to
from .obs_state import build_state


class DiffusionPolicyRunner:
    def __init__(
        self,
        policy: DiffusionPolicy,
        stats: dict[str, Stats],
        *,
        device: torch.device,
        num_envs: int = 1,
        sample_steps: int = 20,
        replan_interval: int = 4,
    ):
        self.policy = policy
        self.stats = stats
        self.device = device
        self.num_envs = num_envs
        self.sample_steps = sample_steps
        self.replan_interval = replan_interval
        self.state_history: deque[torch.Tensor] = deque(maxlen=policy.cfg.obs_horizon)
        self.pending_actions: deque[torch.Tensor] = deque()
        self.steps_since_replan = 0

    def reset(self) -> None:
        self.state_history.clear()
        self.pending_actions.clear()
        self.steps_since_replan = 0

    def act(self, obs: dict[str, Any]) -> torch.Tensor:
        """Return the next action for the current observation."""
        state = build_state(obs, self.num_envs)
        if state.numel() != self.policy.cfg.state_dim:
            raise RuntimeError(
                f"State dim mismatch: env produced {state.numel()}, checkpoint expects "
                f"{self.policy.cfg.state_dim}. The live env's observation groups differ from the "
                f"ones recorded in the training data -- check --observation_preset."
            )
        self.state_history.append(state)
        # At episode start there is no history yet; repeat the first state.
        while len(self.state_history) < self.policy.cfg.obs_horizon:
            self.state_history.appendleft(self.state_history[0])

        if not self.pending_actions or self.steps_since_replan >= self.replan_interval:
            obs_history = torch.stack(list(self.state_history)).unsqueeze(0).to(self.device)
            chunk = self.policy.sample_chunk(normalize(obs_history, self.stats["state"]), steps=self.sample_steps)
            self.pending_actions = deque(denormalize(chunk[0], self.stats["action"]).unbind(dim=0))
            self.steps_since_replan = 0

        self.steps_since_replan += 1
        return self.pending_actions.popleft()


def load_policy(ckpt_path: str, *, task: str, device: torch.device) -> tuple[DiffusionPolicy, dict[str, Stats]]:
    """Load a training checkpoint and return the policy (EMA weights) and its stats."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if checkpoint.get("task") != task:
        raise ValueError(f"Checkpoint was trained on '{checkpoint.get('task')}', not '{task}'.")

    policy = DiffusionPolicy(PolicyConfig(**checkpoint["policy_cfg"]))
    policy.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    policy.to(device).eval()
    return policy, stats_to(checkpoint["stats"], device)
