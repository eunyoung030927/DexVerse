# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion Policy network: a 1-D conditional UNet denoiser over action chunks.

The model predicts the noise added to a normalized action chunk of shape
``(action_horizon, action_dim)``, conditioned on the flattened observation
history. Conditioning is FiLM-style: the observation embedding is added to the
timestep embedding, which every residual block in the UNet already consumes.

Image policies (``PolicyConfig.image_keys`` non-empty) give every camera its own
encoder -- ResNet18 trained from scratch with GroupNorm and a spatial-softmax head,
the image variant of Chi et al. -- keyed by the camera's term name. Per step the
camera features are concatenated with the low-dim state before the obs MLP.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import PolicyConfig
from .unet_chi import ConditionalUnet1D


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        # One encoder per camera, addressed by the camera's term name (never by position).
        self.image_encoders = nn.ModuleDict(
            {key: ImageEncoder(cfg.image_crop, cfg.num_keypoints, cfg.image_feature_dim) for key in cfg.image_keys}
        )
        step_dim = cfg.state_dim + len(cfg.image_keys) * cfg.image_feature_dim
        if cfg.unet_type == "chi":
            # Official Diffusion Policy: the flattened observation features are the global FiLM condition.
            self.unet = ConditionalUnet1D(
                input_dim=cfg.action_dim,
                global_cond_dim=step_dim * cfg.obs_horizon,
                diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
                down_dims=tuple(cfg.unet_block_out_channels),
                kernel_size=cfg.kernel_size,
                n_groups=cfg.n_groups,
                cond_predict_scale=cfg.cond_predict_scale,
            )
            self.obs_encoder = None
            return
        if cfg.unet_type != "diffusers":
            raise ValueError(f"unet_type must be 'diffusers' or 'chi', got '{cfg.unet_type}'.")
        UNet1DModel = _import_unet()
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
            nn.Linear(step_dim * cfg.obs_horizon, cfg.obs_embed_dim),
            nn.Mish(),
            nn.Linear(cfg.obs_embed_dim, temb_channels),
        )
        # ``UNet1DModel.forward`` takes no conditioning argument, so the only way
        # in without forking diffusers is its timestep embedding: wrap ``time_mlp``
        # once here and set the addend per forward pass.
        self.unet.time_mlp = _ConditionedTimeEmbedding(self.unet.time_mlp)

    def encode_obs(self, obs_history: torch.Tensor, images: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Observation embedding ``(B, temb)`` from the state history and, for image policies, the camera history.

        ``obs_history`` is ``(B, obs_horizon, state_dim)``, already normalized; ``images`` maps every
        ``cfg.image_keys`` camera to ``(B, obs_horizon, 3, H, W)`` uint8.
        """
        if obs_history.ndim != 3:
            raise ValueError(f"obs_history must be rank-3, got shape {tuple(obs_history.shape)}.")
        features = [obs_history.float()]
        if self.cfg.image_keys:
            if images is None or set(images) != set(self.cfg.image_keys):
                raise KeyError(f"Policy expects cameras {list(self.cfg.image_keys)}, got {sorted(images or {})}.")
            for key in self.cfg.image_keys:
                frames = images[key]
                batch, horizon = frames.shape[:2]
                feat = self.image_encoders[key](frames.reshape(batch * horizon, *frames.shape[2:]))
                features.append(feat.reshape(batch, horizon, -1))
        step = torch.cat(features, dim=-1).reshape(features[0].shape[0], -1)
        return step if self.obs_encoder is None else self.obs_encoder(step)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        obs_history: torch.Tensor | None = None,
        images: dict[str, torch.Tensor] | None = None,
        *,
        obs_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the noise in ``(B, action_horizon, action_dim)`` action chunks.

        Conditioned on ``obs_embed`` when given (sampling encodes once per chunk), else on
        ``encode_obs(obs_history, images)``.
        """
        if obs_embed is None:
            obs_embed = self.encode_obs(obs_history, images)
        if self.cfg.unet_type == "chi":
            return self.unet(noisy_action, timesteps, obs_embed.to(noisy_action.dtype))
        self.unet.time_mlp.cond = obs_embed.to(noisy_action.dtype)
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
    def sample_chunk(
        self, obs_history: torch.Tensor, *, steps: int, images: dict[str, torch.Tensor] | None = None,
        sampler: str = "ddpm",
    ) -> torch.Tensor:
        """Denoise a fresh action chunk from noise, conditioned on ``obs_history`` (and ``images``).

        ``sampler`` = ``ddpm`` (ancestral, the training scheduler) or ``ddim`` (deterministic DDIM, eta 0, on the
        same noise schedule; the official Diffusion Policy fast-inference option)."""
        device = obs_history.device
        obs_embed = self.encode_obs(obs_history, images)
        scheduler = self.make_noise_scheduler()
        if sampler == "ddim":
            from diffusers import DDIMScheduler

            scheduler = DDIMScheduler.from_config(scheduler.config, clip_sample=True, set_alpha_to_one=True, steps_offset=0)
        elif sampler != "ddpm":
            raise ValueError(f"sampler must be 'ddpm' or 'ddim', got '{sampler}'.")
        scheduler.set_timesteps(steps, device=device)
        action = torch.randn(
            (obs_history.shape[0], self.cfg.action_horizon, self.cfg.action_dim),
            device=device,
        )
        for timestep in scheduler.timesteps:
            batched_timestep = torch.full((action.shape[0],), int(timestep), device=device, dtype=torch.long)
            noise = self.forward(action, batched_timestep, obs_embed=obs_embed)
            action = scheduler.step(noise, timestep, action).prev_sample
        return action


class ImageEncoder(nn.Module):
    """One camera: uint8 ``(N, 3, H, W)`` -> ``(N, feature_dim)``.

    ResNet18 from scratch with every BatchNorm swapped for GroupNorm (16 channels per
    group; BatchNorm statistics break under EMA and small eval batches), then a
    spatial softmax over ``num_keypoints`` learned keypoint maps and a linear layer.
    The input is cropped to ``crop``: a random crop per image while training, the
    center crop at inference.
    """

    def __init__(self, crop: int, num_keypoints: int, feature_dim: int):
        super().__init__()
        from torchvision.models import resnet18

        backbone = resnet18(weights=None)
        _batchnorm_to_groupnorm(backbone)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # drop avgpool + fc
        self.crop = crop
        self.keypoints = nn.Conv2d(512, num_keypoints, kernel_size=1)
        self.out = nn.Linear(num_keypoints * 2, feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float() / 255.0  # [0, 1], as robomimic feeds the official image DP
        if self.crop and self.crop < x.shape[-1]:
            x = _random_crop(x, self.crop) if self.training else _center_crop(x, self.crop)
        x = self.keypoints(self.backbone(x))
        n, k, h, w = x.shape
        attention = torch.softmax(x.reshape(n, k, h * w), dim=-1)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        expected = torch.stack(
            [(attention * grid_x.reshape(1, 1, -1)).sum(-1), (attention * grid_y.reshape(1, 1, -1)).sum(-1)], dim=-1
        )
        return self.out(expected.reshape(n, k * 2))


def _batchnorm_to_groupnorm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            _batchnorm_to_groupnorm(child)


def _center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    top = (x.shape[-2] - size) // 2
    left = (x.shape[-1] - size) // 2
    return x[..., top : top + size, left : left + size]


def _random_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """Independent random crop per image of ``(N, C, H, W)``."""
    n = x.shape[0]
    tops = torch.randint(0, x.shape[-2] - size + 1, (n,), device=x.device)
    lefts = torch.randint(0, x.shape[-1] - size + 1, (n,), device=x.device)
    rows = (tops[:, None] + torch.arange(size, device=x.device)[None]).reshape(n, 1, size, 1)
    cols = (lefts[:, None] + torch.arange(size, device=x.device)[None]).reshape(n, 1, 1, size)
    batch = torch.arange(n, device=x.device).reshape(n, 1, 1, 1)
    channels = torch.arange(x.shape[1], device=x.device).reshape(1, -1, 1, 1)
    return x[batch, channels, rows, cols]


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
