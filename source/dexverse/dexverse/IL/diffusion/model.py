# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion Policy network: a 1-D conditional UNet denoiser over action chunks.

The model predicts the noise added to a normalized action chunk of shape
``(action_horizon, action_dim)``, conditioned on the flattened observation
history. Conditioning is FiLM-style: the observation embedding is added to the
timestep embedding, which every residual block in the UNet already consumes.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import PolicyConfig


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        UNet1DModel = _import_unet()
        self.cfg = cfg
        self.unet = UNet1DModel(
            in_channels=cfg.action_dim,
            out_channels=cfg.action_dim,
            layers_per_block=cfg.layers_per_block,
            block_out_channels=tuple(cfg.unet_block_out_channels),
            down_block_types=("DownResnetBlock1D",) * 3,
            up_block_types=("UpResnetBlock1D",) * 3,
            mid_block_type="MidResTemporalBlock1D",
            out_block_type="OutConv1DBlock",
            time_embedding_type="positional",
            use_timestep_embedding=True,
            act_fn="mish",
        )
        temb_channels = int(cfg.unet_block_out_channels[0])
        self.obs_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim * cfg.obs_horizon, cfg.obs_embed_dim),
            nn.Mish(),
            nn.Linear(cfg.obs_embed_dim, temb_channels),
        )
        # ``UNet1DModel.forward`` takes no conditioning argument, so the only way
        # in without forking diffusers is its timestep embedding: wrap ``time_mlp``
        # once here and set the addend per forward pass.
        self.unet.time_mlp = _ConditionedTimeEmbedding(self.unet.time_mlp)

    def forward(self, noisy_action: torch.Tensor, timesteps: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        """Predict the noise in ``(B, action_horizon, action_dim)`` action chunks.

        ``obs_history`` is ``(B, obs_horizon, state_dim)``, already normalized.
        """
        if obs_history.ndim != 3:
            raise ValueError(f"obs_history must be rank-3, got shape {tuple(obs_history.shape)}.")

        obs_embed = self.obs_encoder(obs_history.reshape(obs_history.shape[0], -1).to(noisy_action.dtype))
        self.unet.time_mlp.cond = obs_embed
        try:
            # diffusers' 1-D UNet is channels-first; our chunks are time-first.
            predicted = self.unet(noisy_action.transpose(1, 2), timesteps).sample
        finally:
            self.unet.time_mlp.cond = None
        return predicted.transpose(1, 2)

    def make_noise_scheduler(self):
        from diffusers import DDPMScheduler

        return DDPMScheduler(
            num_train_timesteps=self.cfg.num_train_timesteps,
            beta_schedule=self.cfg.beta_schedule,
            prediction_type=self.cfg.prediction_type,
        )

    @torch.no_grad()
    def sample_chunk(self, obs_history: torch.Tensor, *, steps: int) -> torch.Tensor:
        """Denoise a fresh action chunk from noise, conditioned on ``obs_history``."""
        device = obs_history.device
        scheduler = self.make_noise_scheduler()
        scheduler.set_timesteps(steps, device=device)
        action = torch.randn(
            (obs_history.shape[0], self.cfg.action_horizon, self.cfg.action_dim),
            device=device,
        )
        for timestep in scheduler.timesteps:
            batched_timestep = torch.full((action.shape[0],), int(timestep), device=device, dtype=torch.long)
            noise = self.forward(action, batched_timestep, obs_history)
            action = scheduler.step(noise, timestep, action).prev_sample
        return action


class _ConditionedTimeEmbedding(nn.Module):
    """diffusers' ``time_mlp``, plus an observation embedding when one is set."""

    def __init__(self, time_mlp: nn.Module):
        super().__init__()
        self.time_mlp = time_mlp
        self.cond: torch.Tensor | None = None

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        embedding = self.time_mlp(timesteps)
        if self.cond is None:
            return embedding
        return embedding + self.cond.to(embedding.device, embedding.dtype)


def _import_unet():
    try:
        from diffusers import UNet1DModel
    except ImportError as exc:
        raise ImportError("The diffusion baseline needs `diffusers`: pip install diffusers") from exc
    return UNet1DModel
