# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conditional 1-D UNet of the official Diffusion Policy (Chi et al., 2023).

Re-implementation of ``diffusion_policy/model/diffusion/conditional_unet1d.py``
(real-stanford/diffusion_policy, MIT): every residual block is FiLM-conditioned on
``[diffusion-step embedding, global observation condition]``, with a per-channel
scale and bias when ``cond_predict_scale`` is set. Used by ``PolicyConfig.unet_type
= "chi"``; the default ``"diffusers"`` path keeps the original DexVerse baseline.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -scale)
        emb = x.float()[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.out_channels = out_channels
        self.cond_predict_scale = cond_predict_scale
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2 if cond_predict_scale else out_channels))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            out = embed[:, 0] * out + embed[:, 1]
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        all_dims = [input_dim, *down_dims]
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed)
        )
        cond_dim = dsed + global_cond_dim
        block = lambda i, o: ConditionalResidualBlock1D(  # noqa: E731
            i, o, cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
        )
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([block(mid_dim, mid_dim), block(mid_dim, mid_dim)])
        self.down_modules = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        block(dim_in, dim_out),
                        block(dim_out, dim_out),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if ind < len(in_out) - 1 else nn.Identity(),
                    ]
                )
                for ind, (dim_in, dim_out) in enumerate(in_out)
            ]
        )
        self.up_modules = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        block(dim_out * 2, dim_in),
                        block(dim_in, dim_in),
                        nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if ind < len(in_out) - 1 else nn.Identity(),
                    ]
                )
                for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:]))
            ]
        )
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups), nn.Conv1d(start_dim, input_dim, 1)
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """``sample`` (B, T, input_dim), ``timestep`` (B,), ``global_cond`` (B, global_cond_dim) -> (B, T, input_dim)."""
        x = sample.transpose(1, 2)
        global_feature = torch.cat([self.diffusion_step_encoder(timestep), global_cond], dim=-1)
        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet2(resnet(x, global_feature), global_feature)
            skips.append(x)
            x = downsample(x)
        for mid in self.mid_modules:
            x = mid(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = upsample(resnet2(resnet(x, global_feature), global_feature))
        return self.final_conv(x).transpose(1, 2)
