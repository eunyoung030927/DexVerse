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
from .obs_images import build_images
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
        sampler: str = "ddpm",
    ):
        self.policy = policy
        self.stats = stats
        self.device = device
        self.num_envs = num_envs
        self.sample_steps = sample_steps
        self.sampler = sampler
        self.replan_interval = replan_interval
        self.state_history: deque[torch.Tensor] = deque(maxlen=policy.cfg.obs_horizon)
        self.image_history: deque[dict[str, torch.Tensor]] = deque(maxlen=policy.cfg.obs_horizon)
        self.pending_actions: deque[torch.Tensor] = deque()
        self.steps_since_replan = 0

    def reset(self) -> None:
        self.state_history.clear()
        self.image_history.clear()
        self.pending_actions.clear()
        self.steps_since_replan = 0

    def act(self, obs: dict[str, Any]) -> torch.Tensor:
        """Return the next action for the current observation."""
        state = build_state(obs, self.num_envs)
        if self.policy.cfg.state_index:
            if state.numel() != self.policy.cfg.full_state_dim:
                raise RuntimeError(
                    f"Full state dim mismatch: env produced {state.numel()}, training data had "
                    f"{self.policy.cfg.full_state_dim} -- check --observation_preset."
                )
            state = state[torch.as_tensor(self.policy.cfg.state_index, dtype=torch.long)]
        if state.numel() != self.policy.cfg.state_dim:
            raise RuntimeError(
                f"State dim mismatch: env produced {state.numel()}, checkpoint expects "
                f"{self.policy.cfg.state_dim}. The live env's observation groups differ from the "
                f"ones recorded in the training data -- check --observation_preset."
            )
        self.state_history.append(state)
        cfg = self.policy.cfg
        if cfg.image_keys:
            # Looked up by camera name, so each encoder gets its own camera.
            self.image_history.append(build_images(obs, tuple(cfg.image_keys), cfg.image_size, self.num_envs))
        # At episode start there is no history yet; repeat the first observation.
        while len(self.state_history) < cfg.obs_horizon:
            self.state_history.appendleft(self.state_history[0])
        while cfg.image_keys and len(self.image_history) < cfg.obs_horizon:
            self.image_history.appendleft(self.image_history[0])

        if not self.pending_actions or self.steps_since_replan >= self.replan_interval:
            obs_history = torch.stack(list(self.state_history)).unsqueeze(0).to(self.device)
            images = None
            if cfg.image_keys:
                images = {
                    key: torch.stack([frame[key] for frame in self.image_history]).unsqueeze(0).to(self.device)
                    for key in cfg.image_keys
                }
            chunk = self.policy.sample_chunk(normalize(obs_history, self.stats["state"]), steps=self.sample_steps, images=images, sampler=self.sampler)
            self.pending_actions = deque(denormalize(chunk[0], self.stats["action"]).unbind(dim=0))
            self.steps_since_replan = 0

        self.steps_since_replan += 1
        return self.pending_actions.popleft()


def load_policy(ckpt_path: str, *, task: str, device: torch.device) -> tuple[DiffusionPolicy, dict[str, Stats]]:
    """Load a training checkpoint and return the policy (EMA weights) and its stats."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if checkpoint.get("task") != task:
        raise ValueError(f"Checkpoint was trained on '{checkpoint.get('task')}', not '{task}'.")

    policy_cfg = dict(checkpoint["policy_cfg"])
    for key in ("image_keys", "state_index", "unet_block_out_channels"):
        if key in policy_cfg:
            policy_cfg[key] = tuple(policy_cfg[key])
    policy = DiffusionPolicy(PolicyConfig(**policy_cfg))
    policy.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    policy.to(device).eval()
    return policy, stats_to(checkpoint["stats"], device)
